use super::{mat4x4::*, quaternion::*, vec3::Vec3};

#[derive(Clone, Debug, Copy)]
pub struct Transform {
    pub translation: Vec3,
    pub scaling    : Vec3,
    pub orientation: Quat,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            orientation: Quat::identity(),
            scaling    : Vec3::ONE,
        }
    }
}

impl Transform {
    pub fn new(scaling: Vec3, translation: Vec3, orientation: Quat) -> Self {
        Self {
            translation,
            scaling,
            orientation,
        }
    }

    pub fn scaling(mut self, scaling: Vec3) -> Self {
        self.scaling = scaling;
        self
    }

    pub fn orientation(mut self, orientation: Quat) -> Self{
        self.orientation = orientation;
        self
    }

    pub fn translation(mut self, trans: Vec3) -> Self{
        self.translation = trans;
        self
    }

    pub fn lerp(&self, other: &Self, factor: f32) -> Transform {
        Self {
            translation: self.translation.mix(other.translation, factor),
            scaling    : self.scaling.mix(other.scaling, factor),
            orientation: self.orientation.nlerp(other.orientation, factor),
        }
    }

    pub fn from_mat(mat: &Mat4x4) -> Self {
        let mut transform = Self::default();

        let translation = Vec3 {
            x: mat.data[0][3],
            y: mat.data[1][3],
            z: mat.data[2][3],
        };

        let orientation = mat.to_quat();
        let d = &mat.data;
        let rot_scale_mat = Mat4x4 {
            data: [
                [d[0][0], d[0][1], d[0][2], 0.0],
                [d[1][0], d[1][1], d[1][2], 0.0],
                [d[2][0], d[2][1], d[2][2], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        };
        let inv_rot_mat = orientation.inverse().to_mat4x4();
        let scale_skew_mat = rot_scale_mat * inv_rot_mat;

        let scaling = Vec3::new(
            scale_skew_mat.data[0][0],
            scale_skew_mat.data[1][1],
            scale_skew_mat.data[2][2],
        );

        transform.translation = translation;
        transform.orientation = orientation;
        transform.scaling     = scaling;

        transform
    }

    pub fn to_mat(&self) -> Mat4x4 {
        let mut x = self.orientation.rotate_vector(Vec3::X);
        let mut y = self.orientation.rotate_vector(Vec3::Y);
        let mut z = self.orientation.rotate_vector(Vec3::Z);

        x = x * self.scaling.x;
        y = y * self.scaling.y;
        z = z * self.scaling.z;

        let p = self.translation;

        Mat4x4 {
            data: [
                [x.x, y.x, z.x, p.x],
                [x.y, y.y, z.y, p.y],
                [x.z, y.z, z.z, p.z],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn inverse(&self) -> Self {
        let mut inv = Transform::default();

        inv.orientation = self.orientation.inverse();

        inv.scaling.x = 1.0 / self.scaling.x;
        inv.scaling.y = 1.0 / self.scaling.y;
        inv.scaling.z = 1.0 / self.scaling.z;

        let inv_trans = -self.translation;
        inv.translation = inv.orientation.rotate_vector(inv.scaling * inv_trans);

        inv
    }

    pub fn combine(&self, rhs: &Self) -> Self {
        let mut out = Transform::default();

        out.scaling     = self.scaling * rhs.scaling;

        out.orientation = self.orientation * rhs.orientation;
        //mhhhh have no idea what this is
        out.translation = self.orientation.rotate_vector(self.scaling * rhs.translation);

        out.translation = self.translation + out.translation;

        out
    }
}
