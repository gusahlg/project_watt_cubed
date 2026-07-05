//! player.rs holds the player's position and view orientation, and derives the
//! render camera from them. Input modules mutate this; the world reads its
//! [`Aabb`] for collision.
//!
//! Positions and velocities are `f64` so play stays precise out to the world
//! border (see [`math`](crate::math)); view angles stay `f32` — a radian needs
//! no more precision, only positions accumulate magnitude.
use voxel_engine::{Camera3D, DVec3, Vec3};

use crate::math::{Aabb, Bounded};

/// Half the size of the player's collision box, measured from the eye position.
pub const PLAYER_HALF: DVec3 = DVec3::new(0.3, 0.9, 0.3);

/// The player: where they are and where they're looking.
pub struct Player {
    /// Eye position in world space.
    pub position: DVec3,
    /// Yaw in radians (rotation around the Y axis / left-right look).
    pub yaw: f32,
    /// Pitch in radians (up-down look), clamped by the look controller.
    pub pitch: f32,
    /// Current vertical velocity, driven by gravity and jumping.
    pub velocity_y: f64,
    /// Whether the player is standing on solid ground this frame.
    pub on_ground: bool,
    /// When true, gravity is disabled and the player can move freely up/down.
    pub fly: bool,
}

impl Player {
    pub fn new(position: DVec3) -> Self {
        Self {
            position,
            yaw: 0.0,
            pitch: 0.0,
            velocity_y: 0.0,
            on_ground: false,
            fly: false,
        }
    }

    /// Full view direction, including pitch. Built from `f64` trig of the
    /// `f32` angles so adding it to an `f64` position loses nothing.
    pub fn forward(&self) -> DVec3 {
        let (yaw, pitch) = (self.yaw as f64, self.pitch as f64);
        DVec3::new(
            yaw.cos() * pitch.cos(),
            pitch.sin(),
            yaw.sin() * pitch.cos(),
        )
    }

    /// The forward and right basis vectors on the XZ plane, used for ground
    /// movement. Returned together because they share one `sin`/`cos` of the yaw,
    /// and both come out unit length already (no normalize needed).
    pub fn movement_basis(&self) -> (DVec3, DVec3) {
        let (sin_yaw, cos_yaw) = (self.yaw as f64).sin_cos();
        let forward = DVec3::new(cos_yaw, 0.0, sin_yaw);
        let right = DVec3::new(-sin_yaw, 0.0, cos_yaw);
        (forward, right)
    }

    /// Build the engine camera that looks out from the player's eye.
    pub fn camera(&self) -> Camera3D {
        self.camera_with_fov(70.0)
    }

    /// Like [`camera`](Self::camera) but with a caller-chosen vertical field of
    /// view in degrees, so the FOV graphics setting can drive the render camera.
    ///
    /// CAMERA REBASE: the engine is `f32`, so instead of handing it a huge
    /// world-space eye position (whose f32 rounding would make far terrain
    /// jitter), the camera sits at the origin looking along the view
    /// direction, and every 3D draw is made camera-relative (chunk meshes via
    /// per-draw offsets, peers by subtracting the eye) — see
    /// [`Game::draw`](crate::game::Game).
    pub fn camera_with_fov(&self, fovy: f32) -> Camera3D {
        Camera3D {
            position: Vec3::ZERO,
            target: self.forward().as_vec3(),
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
