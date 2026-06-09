use math::vec3::Vec3;

use crate::{collider::Collider, rigidbody::RigidBody};

type BodyHandle = usize;

// ── Contact ───────────────────────────────────────────────────────────────────

pub struct Contact {
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    pub pt_a_world: Vec3,
    pub pt_b_world: Vec3,
    /// Normal pointing from B → A (push-A-away direction)
    pub toi: f32,
    pub normal: Vec3,
    pub has_collision: bool,
}

impl Contact {
    fn new(a: BodyHandle, b: BodyHandle) -> Self {
        Self {
            body_a: a,
            body_b: b,
            pt_a_world: Vec3::ZERO,
            pt_b_world: Vec3::ZERO,
            toi: 0.0,
            normal: Vec3::ZERO,
            has_collision: false,
        }
    }
}

pub fn test_sphere_sphere(a: BodyHandle, b: BodyHandle, bodies: &[RigidBody]) -> Contact {
    let mut contact = Contact::new(a, b);

    let ra = sphere_radius(bodies[a].collider()).unwrap();
    let rb = sphere_radius(bodies[b].collider()).unwrap();
    let rab = ra + rb;

    let ab = bodies[b].position - bodies[a].position;
    if ab.len_sqrd() > rab * rab {
        return contact;
    }

    contact.has_collision = true;
    contact.normal = ab.normalize();
    contact.pt_a_world = bodies[a].position + contact.normal * ra;
    contact.pt_b_world = bodies[b].position - contact.normal * rb;
    contact
}

pub fn sphere_sphere_dynamic(
    a: BodyHandle,
    b: BodyHandle,
    bodies: &[RigidBody],
    dt: f32,
) -> Contact {
    let mut contact = Contact::new(a, b);

    let relative_vel = bodies[a].velocity - bodies[b].velocity;
    let ra = sphere_radius(bodies[a].collider()).unwrap();
    let rb = sphere_radius(bodies[b].collider()).unwrap();

    let start_pt_a = bodies[a].position;
    let end_pt_b = bodies[a].position + relative_vel * dt;
    let ray_dir = end_pt_b - start_pt_a;

    let mut t0 = 0.0;
    let mut t1 = 0.0;
    if ray_dir.len_sqrd() < 0.001 * 0.001 {
        let ab = bodies[b].position - bodies[a].position;
        let radius = ra + rb + 0.001;
        if ab.len_sqrd() < radius * radius {
            contact.normal = ab.normalize();
            contact.pt_a_world = bodies[a].position + contact.normal * ra;
            contact.pt_b_world = bodies[b].position - contact.normal * rb;
            return contact;
        }
    } else {
        if !test_ray_sphere(
            start_pt_a,
            ray_dir,
            bodies[b].position,
            ra + rb,
            &mut t0,
            &mut t1,
        ) {
            return contact;
        }
    }

    t0 *= dt;
    t1 *= dt;

    if t1 < 0.0 {
        return contact;
    }

    contact.toi = if t0 < 0.0 { 0.0 } else { t0 };
    if contact.toi > dt {
        return contact;
    }

    let new_pos_a = bodies[a].position + bodies[a].velocity * contact.toi;
    let new_pos_b = bodies[b].position + bodies[b].velocity * contact.toi;
    let ab = new_pos_b - new_pos_a;
    contact.normal = ab.normalize();
    contact.has_collision = true;
    contact.pt_a_world = new_pos_a + ab * ra;
    contact.pt_b_world = new_pos_b + ab * rb;

    contact
}

pub fn test_ray_sphere(
    ray_start: Vec3,
    ray_dir: Vec3,
    sphere_center: Vec3,
    sphere_radius: f32,
    t1: &mut f32,
    t2: &mut f32,
) -> bool {
    let m = sphere_center - ray_start;
    let a = ray_dir.dot(&ray_dir);
    let b = m.dot(&ray_dir);
    let c = m.dot(&m) - sphere_radius * sphere_radius;
    let delta = b * b - a * c;
    let inv_a = 1.0 / a;

    if delta < 0.0 {
        return false;
    }

    let sqrt_delta = delta.sqrt();
    *t1 = (b - sqrt_delta) * inv_a;
    *t2 = (b + sqrt_delta) * inv_a;

    true
}

// ── Contact resolution ────────────────────────────────────────────────────────

pub fn resolve_contact(contact: &Contact, bodies: &mut Vec<RigidBody>) {
    if !contact.has_collision {
        return;
    }

    let a = contact.body_a;
    let b = contact.body_b;

    let p_on_a = contact.pt_a_world;
    let p_on_b = contact.pt_b_world;

    let inv_mass_a = bodies[a].inv_mass();
    let inv_mass_b = bodies[b].inv_mass();
    let total_inv_mass = inv_mass_a + inv_mass_b;

    let inv_inertia_a = bodies[a].inv_inertia_tensor_world();
    let inv_inertia_b = bodies[b].inv_inertia_tensor_world();

    let n = contact.normal;

    // r = contact point relative to centre of mass
    let ra = p_on_a - bodies[a].center_of_mass_world();
    let rb = p_on_b - bodies[b].center_of_mass_world();

    // ── Restitution impulse ───────────────────────────────────────────────────

    let elasticity = bodies[a].restitution * bodies[b].restitution;

    let vel_a = bodies[a].velocity + bodies[a].angular_velocity.cross(&ra);
    let vel_b = bodies[b].velocity + bodies[b].angular_velocity.cross(&rb);
    let vab = vel_a - vel_b;

    let ang_factor_a = (inv_inertia_a * ra.cross(&n)).cross(&ra);
    let ang_factor_b = (inv_inertia_b * rb.cross(&n)).cross(&rb);
    let angular_factor = (ang_factor_a + ang_factor_b).dot(&n);

    let j = (1.0 + elasticity) * vab.dot(&n) / (total_inv_mass + angular_factor);
    let impulse = n * j;

    bodies[a].apply_impulse_at_point(-impulse, p_on_a);
    bodies[b].apply_impulse_at_point(impulse, p_on_b);

    // ── Friction impulse ──────────────────────────────────────────────────────

    let friction = bodies[a].friction * bodies[b].friction;

    // Recompute velocities at contact after restitution impulse was applied
    let vel_a = bodies[a].velocity + bodies[a].angular_velocity.cross(&ra);
    let vel_b = bodies[b].velocity + bodies[b].angular_velocity.cross(&rb);
    let vab = vel_a - vel_b;

    let vel_normal = n * n.dot(&vab);
    let vel_tangent = vab - vel_normal;
    let tang_len_sq = vel_tangent.len_sqrd();

    if tang_len_sq > 1e-10 {
        let tang_dir = vel_tangent.normalize();

        let ang_fric_a = (inv_inertia_a * ra.cross(&tang_dir)).cross(&ra);
        let ang_fric_b = (inv_inertia_b * rb.cross(&tang_dir)).cross(&rb);
        let inv_inertia_tang = (ang_fric_a + ang_fric_b).dot(&tang_dir);

        let reduced_mass = 1.0 / (total_inv_mass + inv_inertia_tang);
        let friction_impulse = vel_tangent * (-reduced_mass * friction);

        bodies[a].apply_impulse_at_point(friction_impulse, p_on_a);
        bodies[b].apply_impulse_at_point(-friction_impulse, p_on_b);
    }

    let ds = contact.pt_b_world - contact.pt_a_world;
    let ta = inv_mass_a / total_inv_mass;
    let tb = inv_mass_b / total_inv_mass;

    bodies[a].position = bodies[a].position + ds * ta;
    bodies[b].position = bodies[b].position - ds * tb;
}

fn sphere_radius(ct: &Collider) -> anyhow::Result<f32> {
    match ct {
        Collider::Sphere(sphere) => Ok(sphere.radius),
        //_ => Err(anyhow::anyhow!("expected Sphere collider")),
    }
}
