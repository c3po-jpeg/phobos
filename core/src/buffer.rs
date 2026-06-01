use crate::device::DeviceContext;
use crate::utils::find_memorytype_index;

use ash::vk;
use gpu_allocator::MemoryLocation;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// BufferUsage flags
// ---------------------------------------------------------------------------

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug)]
    pub struct BufferUsage: u32 {
        const VERTEX       = 0b0000_0001;
        const INDEX        = 0b0000_0010;
        const UNIFORM      = 0b0000_0100;
        const STORAGE      = 0b0000_1000;
        const TRANSFER_SRC = 0b0001_0000;
        const TRANSFER_DST = 0b0010_0000;
    }
}

impl BufferUsage {
    pub fn to_vk_usage(self) -> vk::BufferUsageFlags {
        let mut flags = vk::BufferUsageFlags::empty();
        if self.contains(Self::VERTEX) {
            flags |= vk::BufferUsageFlags::VERTEX_BUFFER;
        }
        if self.contains(Self::INDEX) {
            flags |= vk::BufferUsageFlags::INDEX_BUFFER;
        }
        if self.contains(Self::UNIFORM) {
            flags |= vk::BufferUsageFlags::UNIFORM_BUFFER;
        }
        if self.contains(Self::STORAGE) {
            flags |= vk::BufferUsageFlags::STORAGE_BUFFER;
        }
        if self.contains(Self::TRANSFER_SRC) {
            flags |= vk::BufferUsageFlags::TRANSFER_SRC;
        }
        if self.contains(Self::TRANSFER_DST) {
            flags |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        flags
    }

    /// CPU-visible buffers: uniforms and storage need frequent CPU writes.
    /// Everything else (vertex, index) prefers GPU-only memory.
    fn preferred_memory_location(self) -> MemoryLocation {
        if self.intersects(Self::UNIFORM | Self::STORAGE) {
            MemoryLocation::CpuToGpu
        } else {
            MemoryLocation::GpuOnly
        }
    }

    fn is_cpu_visible(self) -> bool {
        self.preferred_memory_location() == MemoryLocation::CpuToGpu
    }
}

// ---------------------------------------------------------------------------
// Buffer
// ---------------------------------------------------------------------------

pub struct Buffer {
    ctx: Arc<DeviceContext>,
    pub(crate) raw: vk::Buffer,
    memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    pub usage: BufferUsage,
}

impl Buffer {
    /// Allocate an uninitialised buffer of `size` bytes.
    pub fn new(
        ctx: Arc<DeviceContext>,
        size: vk::DeviceSize,
        usage: BufferUsage,
    ) -> anyhow::Result<Self> {
        Self::new_internal(ctx, size, usage)
    }

    pub fn new_with_data<T: Copy>(
        ctx: Arc<DeviceContext>,
        data: &[T],
        usage: BufferUsage,
    ) -> anyhow::Result<Self> {
        let size = std::mem::size_of_val(data) as vk::DeviceSize;
        assert!(size > 0, "Buffer::new_with_data called with empty slice");

        if usage.is_cpu_visible() {
            // Direct map — no staging needed
            let mut buffer = Self::new_internal(Arc::clone(&ctx), size, usage)?;
            buffer.write(data)?;
            Ok(buffer)
        } else {
            let mut staging = Self::new_staging(Arc::clone(&ctx), size)?;
            staging.write(data)?;

            let dst =
                Self::new_internal(Arc::clone(&ctx), size, usage | BufferUsage::TRANSFER_DST)?;

            copy_buffer(Arc::clone(&ctx), staging.raw, dst.raw, size)?;
            Ok(dst)
        }
    }

    /// Map and write `data` into a CPU-visible buffer.
    pub fn write<T: Copy>(&mut self, data: &[T]) -> anyhow::Result<()> {
        let size = std::mem::size_of_val(data) as vk::DeviceSize;

        let ptr = unsafe {
            self.ctx
                .device
                .map_memory(self.memory, 0, size, vk::MemoryMapFlags::empty())?
        };

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                ptr as *mut u8,
                size as usize,
            );

            self.ctx.device.unmap_memory(self.memory);
        }
        Ok(())
    }

    // ---- private helpers -----------------------------------------------

    fn new_internal(
        ctx: Arc<DeviceContext>,
        size: vk::DeviceSize,
        usage: BufferUsage,
    ) -> anyhow::Result<Self> {
        Self::new_with_location(ctx, size, usage)
    }

    /// A CpuToGpu staging buffer (TRANSFER_SRC only, always CPU-visible).
    fn new_staging(ctx: Arc<DeviceContext>, size: vk::DeviceSize) -> anyhow::Result<Self> {
        // Staging buffers are always CPU-visible (CpuToGpu), regardless of the
        // TRANSFER_SRC usage flag which isn't in the is_cpu_visible() set.
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let raw = unsafe { ctx.device.create_buffer(&buffer_info, None)? };
        let requirements = unsafe { ctx.device.get_buffer_memory_requirements(raw) };

        // Staging buffers MUST be CPU-visible
        let memory_properties = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        let memory_type_index = find_memorytype_index(
            &requirements,
            &ctx.device_memory_properties,
            memory_properties,
        );

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index.expect("could not find memory index!"));

        let memory = unsafe { ctx.device.allocate_memory(&alloc_info, None)? };

        unsafe {
            ctx.device.bind_buffer_memory(raw, memory, 0)?;
        }

        Ok(Self {
            ctx,
            raw,
            memory,
            size,
            usage: BufferUsage::TRANSFER_SRC,
        })
    }

    fn new_with_location(
        ctx: Arc<DeviceContext>,
        size: vk::DeviceSize,
        usage: BufferUsage,
    ) -> anyhow::Result<Self> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage.to_vk_usage())
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let raw = unsafe { ctx.device.create_buffer(&buffer_info, None)? };
        let requirements = unsafe { ctx.device.get_buffer_memory_requirements(raw) };

        let memory_properties = if usage.is_cpu_visible() {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        } else {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        };

        let memory_type_index = find_memorytype_index(
            &requirements,
            &ctx.device_memory_properties,
            memory_properties,
        );

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index.expect("could not find memory index!"));

        let memory = unsafe { ctx.device.allocate_memory(&alloc_info, None)? };

        unsafe {
            ctx.device.bind_buffer_memory(raw, memory, 0)?;
        }

        Ok(Self {
            ctx,
            raw,
            memory,
            size,
            usage,
        })
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device.destroy_buffer(self.raw, None);
            self.ctx.device.free_memory(self.memory, None);
        }
    }
}

// ---------------------------------------------------------------------------
// Free function — one-shot GPU copy
// ---------------------------------------------------------------------------

fn copy_buffer(
    ctx: Arc<DeviceContext>,
    src: vk::Buffer,
    dst: vk::Buffer,
    size: vk::DeviceSize,
) -> anyhow::Result<()> {
    let device = &ctx.device;
    let queue = ctx.graphics_queue;
    let queue_index = ctx.graphics_queue_index;

    let pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                .queue_family_index(queue_index),
            None,
        )?
    };

    let cmd = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0]
    };

    unsafe {
        device.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        device.cmd_copy_buffer(
            cmd,
            src,
            dst,
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            }],
        );
        device.end_command_buffer(cmd)?;

        let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        device.queue_submit(
            queue,
            &[vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd))],
            fence,
        )?;
        // Wait for the copy to finish before returning
        device.wait_for_fences(&[fence], true, u64::MAX)?;
        device.destroy_fence(fence, None);

        device.free_command_buffers(pool, &[cmd]);
        device.destroy_command_pool(pool, None);
    }

    Ok(())
}
