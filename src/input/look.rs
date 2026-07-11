//! look.rs turns mouse movement into changes in the player's view direction,
//! letting the user turn and look around.
use voxel_engine::Vec2;

use crate::player::Player;

/// Radians per pixel of rotation.
pub const SENSITIVITY: f32 = 0.0025;
/// ~88°, just short of straight up/down.
pub const PITCH_LIMIT: f32 = 1.54;

/// Apply look delta, clamping pitch to prevent vertical inversion.
pub fn apply(player: &mut Player, look: Vec2) {
    player.yaw += look.x;
    player.pitch = (player.pitch + look.y).clamp(-PITCH_LIMIT, PITCH_LIMIT);
}
