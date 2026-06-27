//! app.rs ties the systems together: it owns the window, world, and player, and
//! runs the update/draw loop. Keeping this state in one struct keeps `main` tiny
//! and makes the per-frame flow easy to follow.
use raylib::prelude::*;

use crate::input::{look, movement};
use crate::player::Player;
use crate::render::Render;
use crate::world::World;

const STARTING_WINDOW_WIDTH: i32 = 1280;
const STARTING_WINDOW_HEIGHT: i32 = 720;
const TARGET_FPS: u32 = 100;
const HELP_TEXT: &str = "WASD move  |  mouse look  |  Space jump  |  F fly  |  Tab free cursor  |  Esc quit";

/// The whole game: window handle, world, player, and transient UI state.
pub struct App {
    rl: RaylibHandle,
    thread: RaylibThread,
    world: World,
    player: Player,
    /// When locked the mouse drives the camera; when unlocked the cursor moves
    /// freely around the window.
    mouse_locked: bool,
}

impl App {
    /// Open the window and build the initial game state.
    pub fn new() -> Self {
        let (mut rl, thread) = raylib::init()
            .size(STARTING_WINDOW_WIDTH, STARTING_WINDOW_HEIGHT)
            .title("voxel prototype")
            .build();

        rl.set_target_fps(TARGET_FPS);
        rl.disable_cursor();

        let world = World::generate();
        // Spawn above the terrain so the player falls and lands on the surface.
        let player = Player::new(Vector3::new(8.0, 40.0, 8.0));

        Self {
            rl,
            thread,
            world,
            player,
            mouse_locked: true,
        }
    }

    /// Run the game until the window is closed.
    pub fn run(mut self) {
        while !self.rl.window_should_close() {
            self.update();
            self.draw();
        }
    }

    /// Advance one frame of simulation from input.
    fn update(&mut self) {
        // Delta time keeps movement speed consistent regardless of frame rate.
        let dt = self.rl.get_frame_time();

        if self.rl.is_key_pressed(KeyboardKey::KEY_TAB) {
            self.toggle_mouse();
        }

        // Only steer the camera with the mouse while it's locked.
        if self.mouse_locked {
            look::update(&mut self.player, &self.rl);
        }

        let input = movement::MoveInput::from_input(&self.rl);
        movement::update_player(&mut self.player, &self.world, &input, dt);
    }

    fn toggle_mouse(&mut self) {
        self.mouse_locked = !self.mouse_locked;
        if self.mouse_locked {
            self.rl.disable_cursor();
        } else {
            self.rl.enable_cursor();
        }
    }

    /// Render the world and the HUD.
    fn draw(&mut self) {
        let camera = self.player.camera();

        let mut d = self.rl.begin_drawing(&self.thread);
        d.clear_background(Color::SKYBLUE);

        {
            let mut d3 = d.begin_mode3D(camera);
            self.world.render(&mut d3);
        }

        d.draw_text(HELP_TEXT, 10, 10, 20, Color::DARKGRAY);
        d.draw_fps(10, 40);
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}
