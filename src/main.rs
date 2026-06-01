use anyhow;
use std::sync::Arc;

mod app;
mod camera;
mod object;
mod scene;

use app::App;
use scene::Scene;

use math::vec3::Vec3;

use physics::{collider::Collider, collider::SphereCollider, rigidbody::RigidBodyBuilder};

use geometry::Shape;

use crate::object::Material;

fn main() -> anyhow::Result<()> {
    // Create the app with the scene reference
    let mut app = App::new();
    // Create the scene
    let scene = Scene::new();

    scene.add_object(
        Material::stone(false, 8.0, 0.4),
        Shape::UVSphere {
            radius: 0.6,
            segments: 40,
            rings: 40,
            color: Some([0.3, 0.6, 0.7]),
        },
        RigidBodyBuilder::new(Collider::Sphere(SphereCollider::new(0.6)))
            .position(Vec3::new(0.0, 10.0, 0.0))
            .mass(1.0)
            .restitution(0.5)
            .build(), // using  kilograms
    );

    scene.add_object(
        Material::polished(true, 8.0, 0.4),
        Shape::UVSphere {
            radius: 0.6,
            segments: 40,
            rings: 40,
            color: None,
        },
        RigidBodyBuilder::new(Collider::Sphere(SphereCollider::new(0.6)))
            .position(Vec3::new(3.0, 10.0, 0.0))
            .mass(1.0)
            .restitution(0.6)
            .build(),
    );

    scene.add_object(
        Material::rubber(false, 10.0, 0.4),
        Shape::CubeSphere {
            radius: 0.6,
            subdivisions: 50,
            color: None,
        },
        RigidBodyBuilder::new(Collider::Sphere(SphereCollider::new(0.6)))
            .position(Vec3::new(-3.0, 10.0, 0.0))
            .mass(1.0)
            .restitution(0.4)
            .build(),
    );

    // floor
    scene.add_object(
        Material::polished(true, 400.0, 0.05),
        Shape::CubeSphere {
            radius: 1000.0,
            subdivisions: 400,
            color: Some([1.0; 3]),
        },
        RigidBodyBuilder::new(Collider::Sphere(SphereCollider::new(1000.0)))
            .position(Vec3::new(0.0, -1002.0, 0.0))
            .mass(f32::INFINITY) // static objects have infinite mass
            .restitution(1.0)
            .build(),
    );

    app.set_scene(Arc::new(scene));
    // Run the application
    app::run(app)?;

    Ok(())
}
