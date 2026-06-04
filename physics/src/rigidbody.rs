use math::{mat3x3::Mat3x3, mat4x4::Mat4x4, quaternion::Quat, transform::Transform, vec3::Vec3};

use crate::collider::Collider;

// ── RigidBody ─────────────────────────────────────────────────────────────────
//
// Invariant: every RigidBody always has a collider.
// This is enforced by making the constructor private and requiring callers
// to go through RigidBodyBuilder, which cannot produce a body without one.

pub struct RigidBody {
    // Physical properties
    pub mass: f32,
    pub restitution: f32,
    pub friction: f32,

    // State
    pub position: Vec3,
    pub orientation: Quat,
    pub velocity: Vec3,
    pub angular_velocity: Vec3,

    // Shape — always present, never None
    collider: Collider,
}

impl RigidBody {
    /// Only entry point: go through the builder.
    pub fn builder(collider: Collider) -> RigidBodyBuilder {
        RigidBodyBuilder::new(collider)
    }

    // ── Collider delegation ───────────────────────────────────────────────────

    pub fn collider(&self) -> &Collider {
        &self.collider
    }

    pub fn inv_mass(&self) -> f32 {
        if self.mass.is_infinite() { 0.0 } else { 1.0 / self.mass }
    }

    /// Inverse inertia tensor in body space, already scaled by inv_mass.
    pub fn inv_inertia_tensor_body(&self) -> Mat3x3 {
        self.collider.inertia_tensor().inverse() * self.inv_mass()
    }

    /// Inverse inertia tensor rotated into world space.
    /// I_world^-1 = R * I_body^-1 * R^T
    pub fn inv_inertia_tensor_world(&self) -> Mat3x3 {
        let r = self.orientation.to_mat3x3();
        r * self.inv_inertia_tensor_body() * r.transpose()
    }

    /// CoM in body space (comes from the collider shape).
    pub fn center_of_mass_body(&self) -> Vec3 {
        self.collider.center_of_mass()
    }

    /// CoM transformed into world space.
    pub fn center_of_mass_world(&self) -> Vec3 {
        self.position + self.orientation.rotate_vector(self.center_of_mass_body())
    }

    // ── Transform ─────────────────────────────────────────────────────────────

    pub fn transform_matrix(&self) -> Mat4x4 {
        Transform::default()
            .translation(self.position)
            .orientation(self.orientation.normalize())
            .to_mat()
    }

    // ── Impulse application ───────────────────────────────────────────────────

    pub fn apply_impulse_linear(&mut self, impulse: Vec3) {
        self.velocity = self.velocity + impulse * self.inv_mass();
    }

    /// `impulse` here is an angular momentum impulse (torque * dt).
    pub fn apply_impulse_angular(&mut self, impulse: Vec3) {
        self.angular_velocity =
            self.angular_velocity + self.inv_inertia_tensor_world() * impulse;

        const MAX_ANG_VEL: f32 = 30.0;
        if self.angular_velocity.len_sqrd() > (MAX_ANG_VEL * MAX_ANG_VEL) {
            self.angular_velocity = self.angular_velocity.normalize() * MAX_ANG_VEL;
        }
    }

    /// Apply a linear + angular impulse at a world-space contact point.
    pub fn apply_impulse_at_point(&mut self,impulse: Vec3, point: Vec3) {
        if self.mass.is_infinite() {
            return;
        }
        self.apply_impulse_linear(impulse);
        let r = point - self.center_of_mass_world();
        self.apply_impulse_angular(r.cross(&impulse));
    }

    // ── Integration ───────────────────────────────────────────────────────────

    pub fn update(&mut self, dt: f32) {
        if self.mass.is_infinite() {
            return;
        }

        // Linear integration
        self.position = self.position + self.velocity * dt;

        //   α = I^-1 * (ω × (I * ω))
        let cm = self.center_of_mass_world();
        let cm_to_pos = self.position - cm;

        let r = self.orientation.to_mat3x3();
        let inertia_world = r * self.collider.inertia_tensor() * r.transpose();
        let alpha = inertia_world.inverse()
            * self.angular_velocity.cross(&(inertia_world * self.angular_velocity));
        self.angular_velocity = self.angular_velocity + alpha * dt;

        let d_angle = self.angular_velocity * dt;
        let angle = d_angle.len();
        let dq = if angle > 1e-8 {
            Quat::from_radians(angle, d_angle)
        } else {
            Quat::identity()
        };

        self.orientation = (dq * self.orientation).normalize();
        self.position = cm + dq.rotate_vector(cm_to_pos);
    }
}

pub struct RigidBodyBuilder {
    collider:         Collider,
    mass:             f32,
    restitution:      f32,
    friction:         f32,
    position:         Vec3,
    orientation:      Quat,
    velocity:         Vec3,
    angular_velocity: Vec3,
}

impl RigidBodyBuilder {
    pub fn new(collider: Collider) -> Self {
        Self {
            collider,
            mass:             10.0,
            restitution:      0.5,
            friction:         0.5,
            position:         Vec3::ZERO,
            orientation:      Quat::identity(),
            velocity:         Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
        }
    }

    pub fn mass(mut self, v: f32) -> Self              { self.mass = v; self }
    pub fn restitution(mut self, v: f32) -> Self       { self.restitution = v; self }
    pub fn friction(mut self, v: f32) -> Self          { self.friction = v; self }
    pub fn position(mut self, v: Vec3) -> Self         { self.position = v; self }
    pub fn orientation(mut self, v: Quat) -> Self      { self.orientation = v; self }
    pub fn velocity(mut self, v: Vec3) -> Self         { self.velocity = v; self }
    pub fn angular_velocity(mut self, v: Vec3) -> Self { self.angular_velocity = v; self }

    /// Consumes the builder and produces a valid RigidBody.
    pub fn build(self) -> RigidBody {
        RigidBody {
            collider:         self.collider,
            mass:             self.mass,
            restitution:      self.restitution,
            friction:         self.friction,
            position:         self.position,
            orientation:      self.orientation,
            velocity:         self.velocity,
            angular_velocity: self.angular_velocity,
        }
    }
}
