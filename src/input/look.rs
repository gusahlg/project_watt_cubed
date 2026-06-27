//! look.rs turns mouse movement into changes in the player's view direction,
//! letting the user turn and look around.
use raylib::prelude::*;

use crate::player::Player;

const SENSITIVITY: f32 = 0.0025; // radians of rotation per pixel of mouse movement
const PITCH_LIMIT: f32 = 1.54; // also radians, ~88°, just short of straight up/down

/// Update the player's yaw and pitch from this frame's mouse movement. 
pub fn update(player: &mut Player, rl: &RaylibHandle) {
    // gets the direction of the mouse moving
    let delta = rl.get_mouse_delta();

    player.yaw += delta.x * SENSITIVITY;
    player.pitch -= delta.y * SENSITIVITY;

    // Clamp it so that you can't look straight up or straight down
    player.pitch = player.pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);
}
