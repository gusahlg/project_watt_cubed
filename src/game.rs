//! game.rs owns the in-world state — world, player, physics, console — and runs a
//! frame of it: input, movement, block interaction, mods, streaming, and drawing.
//! The window and the menu/play state machine live one level up in [`app`](crate::app);
//! a `Game` is handed the window each frame and reports back whether to keep playing
//! or return to the menu.
use raylib::prelude::*;

use crate::block::AIR;
use crate::command;
use crate::console::{self, Console};
use crate::input::{look, movement};
use crate::interact;
use crate::mods::{ModContext, Mods};
use crate::player::Player;
use crate::render::Render;
use crate::sim::Simulation;
use crate::world::World;

/// How far the player can reach to break a block, in world units.
const REACH: f32 = 6.0;
const HELP_TEXT: &str = "WASD move | mouse look | Space jump | F fly | LMB break | I inventory | Tab cursor | T cmd | Esc menu";

/// What a game update wants the app to do next.
pub enum Signal {
    /// Keep playing.
    Continue,
    /// Leave to the start menu (the app saves on the way out).
    ExitToMenu,
}

/// The live world the player is in.
pub struct Game {
    world: World,
    player: Player,
    sim: Simulation,
    console: Console,
    /// When locked the mouse drives the camera and clicks break blocks; when
    /// unlocked the cursor moves freely (for windows/menus).
    mouse_locked: bool,
    /// The save slot this world belongs to.
    save_name: String,
}

impl Game {
    pub fn new(world: World, player: Player, save_name: String) -> Self {
        Self {
            world,
            player,
            sim: Simulation::new(),
            console: Console::new(),
            mouse_locked: true,
            save_name,
        }
    }

    pub fn save_name(&self) -> &str {
        &self.save_name
    }
    pub fn world(&self) -> &World {
        &self.world
    }
    pub fn player(&self) -> &Player {
        &self.player
    }

    /// Capture the cursor when (re)entering play.
    pub fn on_enter(&mut self, rl: &mut RaylibHandle) {
        self.mouse_locked = true;
        rl.disable_cursor();
    }

    /// Advance one frame. Returns [`Signal::ExitToMenu`] when the player leaves.
    pub fn update(&mut self, rl: &mut RaylibHandle, thread: &RaylibThread, mods: &mut Mods) -> Signal {
        let dt = rl.get_frame_time();

        // While the console is open it captures all typing; the world is frozen.
        if self.console.is_open() {
            if let Some(line) = self.console.handle_input(rl) {
                self.console.print(format!("> {line}"));
                for out in command::execute(&line, &mut self.player, &self.world) {
                    self.console.print(out);
                }
            }
            return Signal::Continue;
        }

        // Esc (console closed) leaves to the menu.
        if rl.is_key_pressed(KeyboardKey::KEY_ESCAPE) {
            return Signal::ExitToMenu;
        }

        // Open the console with `T`, or `/` to start a command straight away.
        let slash = rl.is_key_pressed(KeyboardKey::KEY_SLASH);
        if slash || rl.is_key_pressed(KeyboardKey::KEY_T) {
            self.console.open(slash);
            while rl.get_char_pressed().is_some() {}
            return Signal::Continue;
        }

        if rl.is_key_pressed(KeyboardKey::KEY_TAB) {
            self.toggle_mouse(rl);
        }

        if self.mouse_locked {
            look::update(&mut self.player, rl);
        }

        let input = movement::MoveInput::from_input(rl);
        movement::update_player(&mut self.player, &self.world, &input, dt);

        // Break the aimed-at block into its elements while actually aiming (cursor
        // locked, not navigating a free cursor).
        if self.mouse_locked && rl.is_mouse_button_pressed(MouseButton::MOUSE_BUTTON_LEFT) {
            self.break_block(mods);
        }

        // Mods run once per frame here — never inside the voxel loop.
        {
            let mut ctx = ModContext {
                player: &mut self.player,
                world: &mut self.world,
                screen_w: rl.get_screen_width(),
                screen_h: rl.get_screen_height(),
                capturing_text: false,
            };
            mods.update(rl, &mut ctx);
        }

        // Load/mesh/unload chunks around the player, then step physics.
        self.world.stream(self.player.position, rl, thread);
        self.sim.advance(&mut self.world, dt);
        Signal::Continue
    }

    /// Break the block the player is looking at, handing its elements to the mods.
    fn break_block(&mut self, mods: &mut Mods) {
        let Some(hit) = interact::raycast(&self.world, self.player.position, self.player.forward(), REACH)
        else {
            return;
        };
        let (x, y, z) = hit.block;
        let id = self.world.block_at(x, y, z);
        // Snapshot the block's elements before it's removed.
        let elements = self.world.registry().block(id).composition.elements();
        self.world.set_block(x, y, z, AIR);
        mods.on_block_break(&elements, &self.world);
    }

    fn toggle_mouse(&mut self, rl: &mut RaylibHandle) {
        self.mouse_locked = !self.mouse_locked;
        if self.mouse_locked {
            rl.disable_cursor();
        } else {
            rl.enable_cursor();
        }
    }

    /// Render the world and HUD (owns its own draw pass for the frame).
    pub fn draw(&mut self, rl: &mut RaylibHandle, thread: &RaylibThread, mods: &Mods) {
        let camera = self.player.camera();

        let p = self.player.position;
        let coord_text = format!("X: {:.1}    Y: {:.1}    Z: {:.1}", p.x, p.y, p.z);
        let coord_fs = 26;
        let screen_w = rl.get_screen_width();
        let screen_h = rl.get_screen_height();
        let coord_x = (screen_w - rl.measure_text(&coord_text, coord_fs)) / 2;

        let mut d = rl.begin_drawing(thread);
        d.clear_background(Color::SKYBLUE);

        {
            let mut d3 = d.begin_mode3D(camera);
            self.world.render(&mut d3);
        }

        // Aiming crosshair at the screen centre.
        let (cx, cy) = (screen_w / 2, screen_h / 2);
        let cross = Color::new(255, 255, 255, 180);
        d.draw_line(cx - 8, cy, cx + 8, cy, cross);
        d.draw_line(cx, cy - 8, cx, cy + 8, cross);

        console::shadowed(&mut d, &coord_text, coord_x, 12, coord_fs, Color::WHITE);
        d.draw_fps(10, 12);
        console::shadowed(&mut d, HELP_TEXT, 10, 40, 16, Color::RAYWHITE);

        // Enabled mods draw their HUD over the world, under the console.
        mods.draw(&mut d, screen_w, screen_h);
        self.console.draw(&mut d, screen_w, screen_h);
    }
}
