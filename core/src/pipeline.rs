use ash::vk;
use std::sync::Arc;

use super::context::VkContext;
use super::device::DeviceContext;
use crate::shader::ShaderModule;
use crate::vertex::Vertex;


#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct PushConstants {
    pub model: [[f32; 4]; 4], 
}

impl PushConstants {
    pub fn identity() -> Self {
        let id = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        Self { model: id }
    }

    pub fn from_model_matrix(model: [[f32; 4]; 4]) -> Self {
        Self { model }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }

    pub fn push_range() -> vk::PushConstantRange {
        vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<Self>() as u32)
    }
}

pub struct ShadowMapPushConstants {
    pub proj_view: [[f32; 4]; 4],
    pub model: [[f32; 4]; 4],
}

impl ShadowMapPushConstants {
    pub fn new(model: [[f32; 4]; 4], proj_view: [[f32; 4]; 4]) -> Self {
        Self { proj_view, model }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }

    pub fn push_range() -> vk::PushConstantRange {
        vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<Self>() as u32)
    }
}
// ---------------------------------------------------------------------------
// GraphicsPipeline
// ---------------------------------------------------------------------------

pub struct RenderPipeline {
    ctx: Arc<VkContext>,
    pub handle: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub renderpass: vk::RenderPass,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub viewports: Vec<vk::Viewport>,
    pub scissors: Vec<vk::Rect2D>,
}

impl RenderPipeline {
    pub fn create(
        ctx: Arc<VkContext>,
        global: vk::DescriptorSetLayout,
        material: vk::DescriptorSetLayout,
    ) -> anyhow::Result<Self> {
        let mut pipeline = RenderPipeline {
            ctx,
            handle: vk::Pipeline::null(),
            layout: vk::PipelineLayout::null(),
            renderpass: vk::RenderPass::null(),
            framebuffers: Vec::new(),
            viewports: Vec::new(),
            scissors: Vec::new(),
        };

        pipeline
            .create_renderpass()
            .create_layout(global, material)
            .create_framebuffers()
            .create_pipeline()?;

        Ok(pipeline)
    }

    // --- Render pass -------------------------------------------------------

    fn create_renderpass(&mut self) -> &mut Self {
        let attachments = [
            // Color attachment
            vk::AttachmentDescription::default()
                .format(self.ctx.surface_format().format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
            // Depth attachment
            vk::AttachmentDescription::default()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        ];

        let color_ref = [vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
        let depth_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        // Subpass dependency: wait for colour output from previous frame
        let dependencies = [vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .dependency_flags(vk::DependencyFlags::empty())];

        let subpass = vk::SubpassDescription::default()
            .color_attachments(&color_ref)
            .depth_stencil_attachment(&depth_ref)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);

        self.renderpass = unsafe {
            self.ctx
                .device()
                .create_render_pass(
                    &vk::RenderPassCreateInfo::default()
                        .attachments(&attachments)
                        .subpasses(std::slice::from_ref(&subpass))
                        .dependencies(&dependencies),
                    None,
                )
                .expect("Failed to create render pass")
        };

        self
    }

    // --- Pipeline layout (push constants) ----------------------------------

    fn create_layout(
        &mut self,
        global: vk::DescriptorSetLayout,
        material: vk::DescriptorSetLayout,
    ) -> &mut Self {
        let ranges = [PushConstants::push_range()];
        let layouts = [global, material];
        self.layout = unsafe {
            self.ctx
                .device()
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&layouts)
                        .push_constant_ranges(&ranges),
                    None,
                )
                .expect("Failed to create pipeline layout")
        };
        self
    }

    // --- Framebuffers ------------------------------------------------------

    fn create_framebuffers(&mut self) -> &mut Self {
        let depth_view = self.ctx.depth_image_view();
        let resolution = self.ctx.surface_resolution();

        self.framebuffers = self
            .ctx
            .present_image_views()
            .iter()
            .enumerate()
            .map(|(i, &color_view)| {
                let attachments = [color_view, depth_view];
                unsafe {
                    self.ctx
                        .device()
                        .create_framebuffer(
                            &vk::FramebufferCreateInfo::default()
                                .render_pass(self.renderpass)
                                .attachments(&attachments)
                                .width(resolution.width)
                                .height(resolution.height)
                                .layers(1),
                            None,
                        )
                        .unwrap_or_else(|_| panic!("Failed to create framebuffer {}", i))
                }
            })
            .collect();
        self
    }

    // --- Graphics pipeline -------------------------------------------------

    fn create_pipeline(&mut self) -> anyhow::Result<&mut Self> {
        let vertex_shader =
            ShaderModule::load_from_file(Arc::clone(&self.ctx.device_ctx), "shaders/vert.spv")?;

        let fragment_shader =
            ShaderModule::load_from_file(Arc::clone(&self.ctx.device_ctx), "shaders/frag.spv")?;

        let shader_entry = c"main";
        let stages = vec![
            vk::PipelineShaderStageCreateInfo::default()
                .module(vertex_shader.module)
                .name(shader_entry)
                .stage(vk::ShaderStageFlags::VERTEX),
            vk::PipelineShaderStageCreateInfo::default()
                .module(fragment_shader.module)
                .name(shader_entry)
                .stage(vk::ShaderStageFlags::FRAGMENT),
        ];

        let attribute_descriptions = Vertex::get_attribute_descriptions();
        let binding_descriptions = [Vertex::get_binding_description()];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_attribute_descriptions(&attribute_descriptions)
            .vertex_binding_descriptions(&binding_descriptions);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let resolution = self.ctx.surface_resolution();
        self.viewports = vec![
            vk::Viewport::default()
                .x(0.0)
                .y(0.0)
                .width(resolution.width as f32)
                .height(resolution.height as f32)
                .min_depth(0.0)
                .max_depth(1.0),
        ];
        self.scissors = vec![resolution.into()];

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&self.viewports)
            .scissors(&self.scissors);

        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK);

        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let noop_stencil = vk::StencilOpState::default()
            .fail_op(vk::StencilOp::KEEP)
            .pass_op(vk::StencilOp::KEEP)
            .depth_fail_op(vk::StencilOp::KEEP)
            .compare_op(vk::CompareOp::ALWAYS);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .front(noop_stencil)
            .back(noop_stencil)
            .max_depth_bounds(1.0);

        let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];

        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        self.handle = unsafe {
            self.ctx
                .device()
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .stages(&stages)
                        .vertex_input_state(&vertex_input)
                        .input_assembly_state(&input_assembly)
                        .viewport_state(&viewport_state)
                        .rasterization_state(&rasterization)
                        .multisample_state(&multisample)
                        .depth_stencil_state(&depth_stencil)
                        .color_blend_state(&color_blend)
                        .dynamic_state(&dynamic_state)
                        .layout(self.layout)
                        .render_pass(self.renderpass)],
                    None,
                )
                .expect("Failed to create graphics pipeline")[0]
        };

        Ok(self)
    }

    pub fn begin_render_pass(
        &self,
        cmd: vk::CommandBuffer,
        clear_color: [f32; 4],
        present_index: usize,
    ) {
        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: clear_color,
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];

        let render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.renderpass)
            .framebuffer(self.framebuffers[present_index])
            .render_area(self.ctx.surface_resolution().into())
            .clear_values(&clear_values);

        unsafe {
            self.ctx.device().cmd_begin_render_pass(
                cmd,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );

            self.ctx
                .device()
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.handle);
            self.ctx.device().cmd_set_viewport(cmd, 0, &self.viewports);
            self.ctx.device().cmd_set_scissor(cmd, 0, &self.scissors);
        }
    }
}

impl Drop for RenderPipeline {
    fn drop(&mut self) {
        unsafe {
            let device = self.ctx.device();
            for &fb in &self.framebuffers {
                device.destroy_framebuffer(fb, None);
            }
            device.destroy_render_pass(self.renderpass, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_pipeline(self.handle, None);
        }
    }
}

pub struct ShadowPipeline {
    ctx: Arc<DeviceContext>,
    render_pass: vk::RenderPass,
    handle: vk::Pipeline,
    layout: vk::PipelineLayout,
    framebuffer: vk::Framebuffer,
}

impl ShadowPipeline {
    pub fn new(
        ctx: Arc<DeviceContext>,
        view: vk::ImageView,
        res: vk::Extent2D,
    ) -> anyhow::Result<Self> {
        let render_pass = Self::create_renderpass(Arc::clone(&ctx));
        let framebuffer =
            Self::create_framebuffer(Arc::clone(&ctx), view, render_pass, res.clone());
        let layout = Self::create_pipeline_layout(Arc::clone(&ctx));
        let handle = Self::create_pipeline(Arc::clone(&ctx), layout, render_pass, res)?;

        Ok(Self {
            ctx,
            render_pass,
            handle,
            layout,
            framebuffer,
        })
    }

    fn create_renderpass(ctx: Arc<DeviceContext>) -> vk::RenderPass {
        let depth_attachment = vk::AttachmentDescription::default()
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let depth_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&[])
            .depth_stencil_attachment(&depth_ref);

        let deps = [
            vk::SubpassDependency::default()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                .src_access_mask(vk::AccessFlags::SHADER_READ)
                .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dependency_flags(vk::DependencyFlags::BY_REGION),
            vk::SubpassDependency::default()
                .src_subpass(0)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags::LATE_FRAGMENT_TESTS)
                .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .dependency_flags(vk::DependencyFlags::BY_REGION),
        ];

        let attachments = [depth_attachment];
        let subpasses = [subpass];
        let create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&deps);

        unsafe {
            ctx.device
                .create_render_pass(&create_info, None)
                .unwrap_or_else(|_| panic!("failed to create shadow map renderpass!"))
        }
    }
    fn create_framebuffer(
        ctx: Arc<DeviceContext>,
        depth_view: vk::ImageView,
        render_pass: vk::RenderPass,
        res: vk::Extent2D,
    ) -> vk::Framebuffer {
        let attachments = [depth_view];

        let create_info = vk::FramebufferCreateInfo::default()
            .attachment_count(1)
            .attachments(&attachments)
            .width(res.width)
            .height(res.height)
            .render_pass(render_pass)
            .layers(1);

        unsafe {
            ctx.device
                .create_framebuffer(&create_info, None)
                .unwrap_or_else(|_| panic!("Failed to create shadowmap framebuffer"))
        }
    }

    fn create_pipeline_layout(ctx: Arc<DeviceContext>) -> vk::PipelineLayout {
        let ranges = [ShadowMapPushConstants::push_range()];
        unsafe {
            ctx.device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges), // fix this
                    None,
                )
                .expect("Failed to create pipeline layout")
        }
    }

    fn create_pipeline(
        ctx: Arc<DeviceContext>,
        layout: vk::PipelineLayout,
        render_pass: vk::RenderPass,
        res: vk::Extent2D,
    ) -> anyhow::Result<vk::Pipeline> {
        let vertex_shader = ShaderModule::load_from_file(Arc::clone(&ctx), "shaders/shadow.spv")?;

        let shader_entry = c"main";
        let stages = vec![
            vk::PipelineShaderStageCreateInfo::default()
                .module(vertex_shader.module)
                .name(shader_entry)
                .stage(vk::ShaderStageFlags::VERTEX),
        ];

        let attribute_descriptions = Vertex::get_attribute_descriptions();
        let binding_descriptions = [Vertex::get_binding_description()];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_attribute_descriptions(&attribute_descriptions)
            .vertex_binding_descriptions(&binding_descriptions);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        //let resolution = vk::Extent2D::default().height(2048u32).width(2048u32);
        let viewports = vec![
            vk::Viewport::default()
                .x(0.0)
                .y(0.0)
                .width(res.width as f32)
                .height(res.height as f32)
                .min_depth(0.0)
                .max_depth(1.0),
        ];
        let scissors = vec![res.into()];

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::FRONT)
            .depth_bias_enable(true)
            .depth_bias_constant_factor(4.0)
            .depth_bias_slope_factor(2.5);

        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let noop_stencil = vk::StencilOpState::default()
            .fail_op(vk::StencilOp::KEEP)
            .pass_op(vk::StencilOp::KEEP)
            .depth_fail_op(vk::StencilOp::KEEP)
            .compare_op(vk::CompareOp::ALWAYS);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .front(noop_stencil)
            .back(noop_stencil)
            .max_depth_bounds(1.0);

        let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];

        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        Ok(unsafe {
            ctx.device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .stages(&stages)
                        .vertex_input_state(&vertex_input)
                        .input_assembly_state(&input_assembly)
                        .viewport_state(&viewport_state)
                        .rasterization_state(&rasterization)
                        .multisample_state(&multisample)
                        .depth_stencil_state(&depth_stencil)
                        .color_blend_state(&color_blend)
                        .dynamic_state(&dynamic_state)
                        .layout(layout)
                        .render_pass(render_pass)],
                    None,
                )
                .expect("Failed to create graphics pipeline")[0]
        })
    }

    pub fn handle(&self) -> vk::Pipeline {
        self.handle
    }

    pub fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }

    pub fn render_pass(&self) -> vk::RenderPass {
        self.render_pass
    }

    pub fn framebuffer(&self) -> vk::Framebuffer {
        self.framebuffer
    }

    pub fn begin_render_pass(&self, cmd: vk::CommandBuffer, resolution: vk::Extent2D) {
        let clear_values = [vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        }];

        let begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass())
            .framebuffer(self.framebuffer())
            .render_area(resolution.into())
            .clear_values(&clear_values);

        unsafe {
            self.ctx
                .device
                .cmd_begin_render_pass(cmd, &begin_info, vk::SubpassContents::INLINE);

            let viewports = [vk::Viewport::default()
                .x(0.0)
                .y(0.0)
                .width(resolution.width as f32)
                .height(resolution.height as f32)
                .min_depth(0.0)
                .max_depth(1.0)];
            self.ctx.device.cmd_set_viewport(cmd, 0, &viewports);

            let scissors = [vk::Rect2D::from(resolution)];
            self.ctx.device.cmd_set_scissor(cmd, 0, &scissors);

            self.ctx
                .device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.handle);
        }
    }
}

impl Drop for ShadowPipeline {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device.destroy_render_pass(self.render_pass, None);
            self.ctx.device.destroy_framebuffer(self.framebuffer, None);
            self.ctx.device.destroy_pipeline_layout(self.layout, None);
            self.ctx.device.destroy_pipeline(self.handle, None);
        }
    }
}
