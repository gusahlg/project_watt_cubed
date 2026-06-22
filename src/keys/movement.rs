/// movement.rs is the file that does the movement logic and handles key input
use raylib::prelude::*;

const MOVE_SPEED: f32 = 0.1;

pub fn move_player_camera(cam: &mut Camera3D, direction: Direction) {
    let mut dx = 0.0;
    let mut dz = 0.0;

    match direction.forward_back {
        AxisZ::Forward => dz -= MOVE_SPEED,
        AxisZ::Backward => dz += MOVE_SPEED,
        AxisZ::None => {}
    }

    match direction.left_right {
        AxisX::Left => dx -= MOVE_SPEED,
        AxisX::Right => dx += MOVE_SPEED,
        AxisX::None => {}
    }

    // Move both the camera position and target,
    // so the camera moves without changing where it is looking.
    cam.position.x += dx;
    cam.position.z += dz;

    cam.target.x += dx;
    cam.target.z += dz;
}

pub struct Direction {
    forward_back: AxisZ,
    left_right: AxisX,
}

pub enum AxisZ {
    Forward,
    Backward,
    None,
}

pub enum AxisX {
    Left,
    Right,
    None,
}

pub fn get_move_direction(rl: &RaylibHandle) -> Direction {
    let forward_back = if rl.is_key_down(KeyboardKey::KEY_W) {
        AxisZ::Forward
    } else if rl.is_key_down(KeyboardKey::KEY_S) {
        AxisZ::Backward
    } else {
        AxisZ::None
    };

    let left_right = if rl.is_key_down(KeyboardKey::KEY_A) {
        AxisX::Left
    } else if rl.is_key_down(KeyboardKey::KEY_D) {
        AxisX::Right
    } else {
        AxisX::None
    };

    Direction {
        forward_back,
        left_right,
    }
}
