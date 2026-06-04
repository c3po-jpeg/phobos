// home made quaternion math lib cause i have a big ego.
// "john vince - quaternions for for computer graphics" was a massive help along with
// "gabor szauer - hands on c++ game animation programming packt", both great books.
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

use crate::mat3x3::Mat3x3;

use super::{mat4x4::*, vec3::Vec3};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub s: f32,
}

impl Quat {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        s: 0.0,
    };

    pub fn identity() -> Self {
        Self::new(0.0, 0.0, 0.0, 1.0)
    }

    pub fn new(x: f32, y: f32, z: f32, s: f32) -> Self {
        Self { x, y, z, s }
    }
    /// get quaternion from array
    pub fn from(a: &[f32; 4]) -> Self {
        Self::new(a[0], a[1], a[2], a[3])
    }
    pub fn to_array(&self) -> [f32; 4] {
        [self.x, self.y, self.z, self.s]
    }
    /// halves the angle and creates a quaternion from it and the specified axis
    /// and also axis is normalized so no worries
    /// resulting quaternion intended to be used with 'to_mat' function
    pub fn from_degrees(angle: f32, axis: Vec3) -> Self {
        Self::from_radians(angle.to_radians(), axis)
    }

    pub fn from_radians(angle: f32, axis: Vec3) -> Self {
        let s = f32::sin(angle / 2.0);
        let c = f32::cos(angle / 2.0);

        let unit_axis = Vec3::normalize(&axis);

        let x = s * unit_axis.x;
        let y = s * unit_axis.y;
        let z = s * unit_axis.z;
        let s = c;

        Self { x, y, z, s }
    }

    pub fn rotation_x(angle: f32) -> Self {
        Self::from_degrees(angle, Vec3::X)
    }

    pub fn rotation_y(angle: f32) -> Self {
        Self::from_degrees(angle, Vec3::Y)
    }

    pub fn rotation_z(angle: f32) -> Self {
        Self::from_degrees(angle, Vec3::Z)
    }

    pub fn norm(&self) -> f32 {
        let x2 = f32::powf(self.x, 2.0);
        let y2 = f32::powf(self.y, 2.0);
        let z2 = f32::powf(self.z, 2.0);
        let s2 = f32::powf(self.s, 2.0);
        f32::sqrt(x2 + y2 + z2 + s2)
    }

    pub fn normalize(&self) -> Self {
        let n = self.norm();

        if n.abs() < f32::EPSILON {
                return Self::identity();
        }

        let inv = 1.0 / n;

        Self {
            x: (inv * self.x),
            y: (inv * self.y),
            z: (inv * self.z),
            s: (inv * self.s),
        }
    }

    pub fn conjugate(&self) -> Self {
        Self {
            x: (-self.x),
            y: (-self.y),
            z: (-self.z),
            s: (self.s),
        }
    }

    pub fn inverse(&self) -> Self {
        let len_sq = self.x * self.x + self.y * self.y + self.z * self.z + self.s * self.s;
        let inv_len = 1.0 / len_sq;
        self.conjugate() * inv_len
    }

    pub fn dot(&self, rhs: &Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.s * rhs.s
    }

    pub fn nlerp(&self, other: Self, c: f32) -> Quat {
        (*self + (other - *self) * c).normalize()
    }

    pub fn axis(&self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    /// creates a rotation matrix from a quaternion
    pub fn to_mat3x3(&self) -> Mat3x3 {
        let x2 = f32::powf(self.x, 2.0);
        let y2 = f32::powf(self.y, 2.0);
        let z2 = f32::powf(self.z, 2.0);
        // first row
        let xx = 1.0 - 2.0 * (y2 + z2);
        let xy = 2.0 * (self.x * self.y - self.s * self.z);
        let xz = 2.0 * (self.x * self.z + self.s * self.y);
        // second row
        let yx = 2.0 * (self.x * self.y + self.s * self.z);
        let yy = 1.0 - 2.0 * (x2 + z2);
        let yz = 2.0 * (self.y * self.z - self.s * self.x);
        // third row
        let zx = 2.0 * (self.x * self.z - self.s * self.y);
        let zy = 2.0 * (self.y * self.z + self.s * self.x);
        let zz = 1.0 - 2.0 * (x2 + y2);

        Mat3x3 {
            data: [
                [ xx,  xy,  xz],
                [ yx,  yy,  yz],
                [ zx,  zy,  zz],
            ],
        }
    }

    /// rotate around a specified axis
    /// creates a rotation matrix from a quaternion
    pub fn to_mat4x4(&self) -> Mat4x4 {
        let d = self.to_mat3x3().data;

        Mat4x4 {
            data: [
                [d[0][0], d[0][1], d[0][2], 0.0],
                [d[1][0], d[1][1], d[1][2], 0.0],
                [d[2][0], d[2][1], d[2][2], 0.0],
                [0.0,     0.0,     0.0,     1.0],
            ],
        }
    }

    /// r = (q * v' * q^-1).xyz
    pub fn rotate_vector(&self, v: Vec3) -> Vec3 {
        let a = self.axis() * 2.0 * Vec3::dot(&self.axis(), &v);
        let b = v * (self.s * self.s - Vec3::dot(&self.axis(), &self.axis()));
        let c = Vec3::cross(&self.axis(), &v) * 2.0 * self.s;

        a + b + c
    }
}

use std::ops::*;

impl Sub for Quat {
    type Output = Quat;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            s: self.s - rhs.s,
        }
    }
}

impl Add for Quat {
    type Output = Quat;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            s: self.s + rhs.s,
        }
    }
}

impl Neg for Quat {
    type Output = Quat;
    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            s: -self.s,
        }
    }
}
impl Mul<f32> for Quat {
    type Output = Quat;
    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
            s: self.s * rhs,
        }
    }
}
impl Mul<Quat> for f32 {
    type Output = Quat;
    fn mul(self, rhs: Quat) -> Self::Output {
        Quat {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
            s: self * rhs.s,
        }
    }
}

impl Mul<Quat> for Quat {
    type Output = Quat;
    fn mul(self, rhs: Quat) -> Self::Output {
        Self {
            x: self.s * rhs.x + self.x * rhs.s + self.y * rhs.z - self.z * rhs.y,
            y: self.s * rhs.y + self.y * rhs.s + self.z * rhs.x - self.x * rhs.z,
            z: self.s * rhs.z + self.z * rhs.s + self.x * rhs.y - self.y * rhs.x,
            s: self.s * rhs.s - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
        }
    }
}
