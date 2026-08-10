//! look.rs turns mouse movement into changes in the player's view direction,
//! letting the user turn and look around.
use voxel_engine::Vec2;

use crate::player::Player;

/// Radians per pixel of rotation.
pub const SENSITIVITY: f32 = 0.0025;
/// ~88°, just short of straight up/down.
pub const PITCH_LIMIT: f32 = 1.54;

/// Apply a raw look delta via the one [`Orientation::look`](crate::camera::Orientation::look) clamp.
pub fn apply(player: &mut Player, d: Vec2) {
    player.orientation.look(d, SENSITIVITY);
}
