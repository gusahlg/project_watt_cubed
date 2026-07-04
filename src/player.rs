//! player.rs holds the player's position and view orientation, and derives the
//! render camera from them. Input modules mutate this; the world reads its
//! [`Aabb`] for collision.
use voxel_engine::{Camera3D, Vec3};

use crate::math::{Aabb, Bounded};

/// Half the size of the player's collision box, measured from the eye position.
pub const PLAYER_HALF: Vec3 = Vec3::new(0.3, 0.9, 0.3);

/// The player: where they are and where they're looking.
pub struct Player {
    /// Eye position in world space.
    pub position: Vec3,
    /// Yaw in radians (rotation around the Y axis / left-right look).
    pub yaw: f32,
    /// Pitch in radians (up-down look), clamped by the look controller.
    pub pitch: f32,
    /// Current vertical velocity, driven by gravity and jumping.
    pub velocity_y: f32,
    /// Whether the player is standing on solid ground this frame.
    pub on_ground: bool,
    /// When true, gravity is disabled and the player can move freely up/down.
    pub fly: bool,
}

impl Player {
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            yaw: 0.0,
            pitch: 0.0,
            velocity_y: 0.0,
            on_ground: false,
            fly: false,
        }
    }

    /// Full view direction, including pitch.
    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
    }

    /// The forward and right basis vectors on the XZ plane, used for ground
    /// movement. Returned together because they share one `sin`/`cos` of the yaw,
    /// and both come out unit length already (no normalize needed).
    pub fn movement_basis(&self) -> (Vec3, Vec3) {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let forward = Vec3::new(cos_yaw, 0.0, sin_yaw);
        let right = Vec3::new(-sin_yaw, 0.0, cos_yaw);
        (forward, right)
    }

    /// Build the engine camera that looks out from the player's eye.
    pub fn camera(&self) -> Camera3D {
        self.camera_with_fov(70.0)
    }

    /// Like [`camera`](Self::camera) but with a caller-chosen vertical field of
    /// view in degrees, so the FOV graphics setting can drive the render camera.
    pub fn camera_with_fov(&self, fovy: f32) -> Camera3D {
        Camera3D {
            position: self.position,
            target: self.position + self.forward(),
            up: Vec3::new(0.0, 1.0, 0.0),
            fovy,
        }
    }
}

impl Bounded for Player {
    fn aabb(&self) -> Aabb {
        Aabb::new(self.position, PLAYER_HALF)
    }
}
