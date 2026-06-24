//! player.rs holds the player's position and view orientation, and derives the
//! render camera from them. Input modules mutate this; the world reads its
//! [`Aabb`] for collision.
use raylib::prelude::*;

use crate::math::{Aabb, Bounded};

/// Half the size of the player's collision box, measured from the eye position.
pub const PLAYER_HALF: Vector3 = Vector3 {
    x: 0.3,
    y: 0.9,
    z: 0.3,
};

/// The player: where they are and where they're looking.
pub struct Player {
    /// Eye position in world space.
    pub position: Vector3,
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
    pub fn new(position: Vector3) -> Self {
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
    pub fn forward(&self) -> Vector3 {
        Vector3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
    }

    /// View direction flattened onto the XZ plane, for ground movement.
    pub fn forward_flat(&self) -> Vector3 {
        Vector3::new(self.yaw.cos(), 0.0, self.yaw.sin()).normalize()
    }

    /// The rightward direction on the XZ plane (forward rotated 90°).
    pub fn right_flat(&self) -> Vector3 {
        let f = self.forward_flat();
        Vector3::new(-f.z, 0.0, f.x)
    }

    /// Build the raylib camera that looks out from the player's eye.
    pub fn camera(&self) -> Camera3D {
        Camera3D::perspective(
            self.position,
            self.position + self.forward(),
            Vector3::new(0.0, 1.0, 0.0),
            70.0,
        )
    }
}

impl Bounded for Player {
    fn aabb(&self) -> Aabb {
        Aabb::new(self.position, PLAYER_HALF)
    }
}
