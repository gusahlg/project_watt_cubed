//! app.rs ties the systems together: it owns the window, world, and player, and
//! runs the update/draw loop. Keeping this state in one struct keeps `main` tiny
//! and makes the per-frame flow easy to follow.
use raylib::prelude::*;

use crate::command;
use crate::console::{self, Console};
use crate::input::{look, movement};
use crate::player::Player;
use crate::render::Render;
use crate::world::World;

const STARTING_WINDOW_WIDTH: i32 = 1280;
const STARTING_WINDOW_HEIGHT: i32 = 720;
// Currently not in use for testing purposes
const TARGET_FPS: u32 = 100;
const HELP_TEXT: &str =
    "WASD move  |  mouse look  |  Space jump  |  F fly  |  Tab free cursor  |  T command  |  Esc quit";

/// The whole game: window handle, world, player, and transient UI state.
///
/// Field order is also drop order: `world` owns the GPU chunk models, which must
/// be freed (`UnloadModel`) while the GL context is still alive, so it is declared
/// before `rl` (whose drop closes the window).
pub struct App {
    world: World,
    player: Player,
    /// The in-game console / chat line; swallows game input while it's open.
    console: Console,
    /// When locked the mouse drives the camera; when unlocked the cursor moves
    /// freely around the window.
    mouse_locked: bool,
    rl: RaylibHandle,
    thread: RaylibThread,
}

impl App {
    /// Open the window and build the initial game state.
    pub fn new() -> Self {
        let (mut rl, thread) = raylib::init()
            .size(STARTING_WINDOW_WIDTH, STARTING_WINDOW_HEIGHT)
            .title("voxel prototype")
            .build();

        // Temporary diasable
        // rl.set_target_fps(TARGET_FPS);
        rl.disable_cursor();
        // We manage quitting ourselves so Esc can close the console instead of
        // always exiting the game (see `update`).
        rl.set_exit_key(None);

        // Generate the voxel data, then upload the chunk geometry to the GPU once.
        let mut world = World::generate();
        world.build_meshes(&mut rl, &thread);

        // Spawn above the terrain so the player falls and lands on the surface.
        let player = Player::new(Vector3::new(8.0, 40.0, 8.0));

        Self {
            world,
            player,
            console: Console::new(),
            mouse_locked: true,
            rl,
            thread,
        }
    }

    /// Run the game until the window is closed or the player quits.
    pub fn run(mut self) {
        while !self.rl.window_should_close() {
            if self.update() {
                break;
            }
            self.draw();
        }
    }

    /// Advance one frame of simulation from input. Returns `true` to quit.
    fn update(&mut self) -> bool {
        // Delta time keeps movement speed consistent regardless of frame rate.
        let dt = self.rl.get_frame_time();

        // While the console is open it captures all typing; the world is frozen so
        // movement keys land in the input box instead of moving the player.
        if self.console.is_open() {
            if let Some(line) = self.console.handle_input(&mut self.rl) {
                self.console.print(format!("> {line}"));
                for out in command::execute(&line, &mut self.player) {
                    self.console.print(out);
                }
            }
            return false;
        }

        // Esc quits when the console is closed (it closes the console when open,
        // handled above).
        if self.rl.is_key_pressed(KeyboardKey::KEY_ESCAPE) {
            return true;
        }

        // Open the console with `T`, or `/` to start a command straight away.
        let slash = self.rl.is_key_pressed(KeyboardKey::KEY_SLASH);
        if slash || self.rl.is_key_pressed(KeyboardKey::KEY_T) {
            self.console.open(slash);
            // Drop the keystroke that opened it so it doesn't appear in the input.
            while self.rl.get_char_pressed().is_some() {}
            return false;
        }

        if self.rl.is_key_pressed(KeyboardKey::KEY_TAB) {
            self.toggle_mouse();
        }

        // Only steer the camera with the mouse while it's locked.
        if self.mouse_locked {
            look::update(&mut self.player, &self.rl);
        }

        let input = movement::MoveInput::from_input(&self.rl);
        movement::update_player(&mut self.player, &self.world, &input, dt);
        false
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

        // Build the coordinate readout and centre it before borrowing `rl` for
        // drawing (`measure_text`/screen size need the handle).
        let p = self.player.position;
        let coord_text = format!("X: {:.1}    Y: {:.1}    Z: {:.1}", p.x, p.y, p.z);
        let coord_fs = 26;
        let screen_w = self.rl.get_screen_width();
        let screen_h = self.rl.get_screen_height();
        let coord_x = (screen_w - self.rl.measure_text(&coord_text, coord_fs)) / 2;

        let mut d = self.rl.begin_drawing(&self.thread);
        d.clear_background(Color::SKYBLUE);

        {
            let mut d3 = d.begin_mode3D(camera);
            self.world.render(&mut d3);
        }

        // Coordinate HUD, centred at the top of the screen.
        console::shadowed(&mut d, &coord_text, coord_x, 12, coord_fs, Color::WHITE);

        // FPS and controls in the top-left; the console owns the bottom strip.
        d.draw_fps(10, 12);
        console::shadowed(&mut d, HELP_TEXT, 10, 40, 18, Color::RAYWHITE);

        // Console / chat overlay on top of everything.
        self.console.draw(&mut d, screen_w, screen_h);
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}
