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
use crate::net::chat;
use crate::net::client::{Connection, Incoming};
use crate::player::Player;
use crate::render::Render;
use crate::save;
use crate::sim::Simulation;
use crate::world::World;

/// How far the player can reach to break a block, in world units.
const REACH: f32 = 6.0;
const HELP_TEXT: &str = "WASD move | mouse look | Space jump | F fly | LMB break | I inventory | Tab cursor | T chat/cmd | Esc menu";
/// Half-extents of another player's drawn body — matches the collision box in
/// [`player`](crate::player::PLAYER_HALF).
const PEER_HALF: Vector3 = Vector3 { x: 0.3, y: 0.9, z: 0.3 };
/// Peers past this distance get no floating name tag (it would be unreadable).
const TAG_RANGE: f32 = 90.0;

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
    /// The live server connection when playing multiplayer; `None` in singleplayer.
    /// The player simulates locally and the server keeps everyone in sync.
    net: Option<Connection>,
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
            net: None,
        }
    }

    /// Attach a server connection, turning this into a multiplayer session.
    pub fn with_net(mut self, net: Connection) -> Self {
        self.net = Some(net);
        self
    }

    pub fn save_name(&self) -> &str {
        &self.save_name
    }

    /// Whether this is a networked session (its world is a server mirror, not a
    /// local save, so the app does not autosave it).
    pub fn is_multiplayer(&self) -> bool {
        self.net.is_some()
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

        // Drain the server first so edits and chat keep flowing even while the
        // console is open or the player stands still.
        if self.apply_net_events() {
            self.console.print("* disconnected from server".to_string());
            return Signal::ExitToMenu;
        }

        // While the console is open it captures all typing; the world is frozen
        // (locally — other players keep moving over the network).
        if self.console.is_open() {
            if let Some(line) = self.console.handle_input(rl) {
                self.submit_line(line);
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

        // Report our own state to the server (throttled + heartbeat inside).
        if let Some(net) = &mut self.net {
            net.send_move(self.player.position, self.player.yaw, self.player.pitch);
        }

        // Load/mesh/unload chunks around the player, then step physics.
        self.world.stream(self.player.position, rl, thread);
        self.sim.advance(&mut self.world, dt);
        Signal::Continue
    }

    /// Drain queued server messages: apply world edits, surface chat, and report a
    /// lost connection. Returns `true` if the server dropped us.
    fn apply_net_events(&mut self) -> bool {
        let events = match &mut self.net {
            Some(net) => net.poll(),
            None => return false,
        };
        let mut disconnected = false;
        for event in events {
            match event {
                Incoming::Edit { x, y, z, spec } => {
                    // Resolve the portable spec against our own palette, then apply.
                    let id = save::parse_block(&mut self.world, &spec);
                    self.world.set_block(x, y, z, id);
                }
                Incoming::Chat { from_name, channel, text } => {
                    let scope = if channel == chat::GLOBAL { "[global] " } else { "" };
                    self.console.print(format!("{scope}<{from_name}> {text}"));
                }
                Incoming::Disconnected => disconnected = true,
            }
        }
        disconnected
    }

    /// Handle one submitted console line. A leading `/` is always a local command; in
    /// multiplayer any other line is chat (a leading `!` sends it to global chat),
    /// while in singleplayer it stays a command as before.
    fn submit_line(&mut self, line: String) {
        if !line.starts_with('/') {
            if let Some(net) = &mut self.net {
                let (channel, text) = match line.strip_prefix('!') {
                    Some(rest) => (chat::GLOBAL, rest.trim().to_string()),
                    None => (chat::LOCAL, line),
                };
                if !text.is_empty() {
                    // The server echoes chat back to us, so we don't print it here.
                    net.send_chat(channel, text);
                }
                return;
            }
        }
        self.console.print(format!("> {line}"));
        for out in command::execute(&line, &mut self.player, &self.world) {
            self.console.print(out);
        }
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
        // Tell the server (it validates and relays to everyone else). We apply
        // locally above for a responsive feel; the server is still authoritative.
        if let Some(net) = &mut self.net {
            net.send_edit(x, y, z, "air".to_string());
        }
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

        // Gather the other players to draw, projecting a head point to screen space
        // now (while we still hold `rl`) for the floating name tags.
        let peers = self.peer_draws(rl, camera);
        let online = self.net.as_ref().map(|net| net.peers().count() + 1);

        let mut d = rl.begin_drawing(thread);
        d.clear_background(Color::SKYBLUE);

        {
            let mut d3 = d.begin_mode3D(camera);
            self.world.render(&mut d3);
            // Other players: a body box and a small head, tinted per player.
            for peer in &peers {
                let (bw, bh, bd) = (PEER_HALF.x * 2.0, PEER_HALF.y * 2.0, PEER_HALF.z * 2.0);
                d3.draw_cube(peer.pos, bw, bh, bd, peer.color);
                d3.draw_cube_wires(peer.pos, bw, bh, bd, Color::BLACK);
                let head = peer.pos + Vector3::new(0.0, PEER_HALF.y + 0.2, 0.0);
                d3.draw_cube(head, 0.4, 0.4, 0.4, peer.color);
            }
        }

        // Aiming crosshair at the screen centre.
        let (cx, cy) = (screen_w / 2, screen_h / 2);
        let cross = Color::new(255, 255, 255, 180);
        d.draw_line(cx - 8, cy, cx + 8, cy, cross);
        d.draw_line(cx, cy - 8, cx, cy + 8, cross);

        console::shadowed(&mut d, &coord_text, coord_x, 12, coord_fs, Color::WHITE);
        d.draw_fps(10, 12);
        console::shadowed(&mut d, HELP_TEXT, 10, 40, 16, Color::RAYWHITE);

        // Floating name tags over each visible player.
        for peer in &peers {
            if let Some(tag) = peer.tag {
                let fs = 18;
                let tw = d.measure_text(&peer.name, fs);
                console::shadowed(&mut d, &peer.name, tag.x as i32 - tw / 2, tag.y as i32, fs, Color::WHITE);
            }
        }
        if let Some(count) = online {
            let text = format!("players online: {count}");
            let w = d.measure_text(&text, 20);
            console::shadowed(&mut d, &text, screen_w - w - 12, 12, 20, Color::LIME);
        }

        // Enabled mods draw their HUD over the world, under the console.
        mods.draw(&mut d, screen_w, screen_h);
        self.console.draw(&mut d, screen_w, screen_h);
    }

    /// Build the per-frame draw data for other players, projecting a head point to
    /// screen space for the name tag (only for peers in front and within range).
    fn peer_draws(&self, rl: &RaylibHandle, camera: Camera3D) -> Vec<PeerDraw> {
        let Some(net) = &self.net else { return Vec::new() };
        let eye = self.player.position;
        let forward = self.player.forward();
        net.peers()
            .map(|peer| {
                let head = peer.pos + Vector3::new(0.0, PEER_HALF.y + 0.4, 0.0);
                let to_head = head - eye;
                let visible = to_head.dot(forward) > 0.0 && to_head.length() <= TAG_RANGE;
                let tag = visible.then(|| rl.get_world_to_screen(head, camera));
                PeerDraw {
                    pos: peer.pos,
                    color: peer_color(&peer.name),
                    name: peer.name.clone(),
                    tag,
                }
            })
            .collect()
    }
}

/// Everything needed to draw one other player this frame.
struct PeerDraw {
    pos: Vector3,
    color: Color,
    name: String,
    /// Screen position for the name tag, or `None` when off-screen/behind us.
    tag: Option<Vector2>,
}

/// A stable, cheerful colour for a player, hashed from their name so the same player
/// keeps the same tint across clients.
fn peer_color(name: &str) -> Color {
    const PALETTE: [Color; 6] = [
        Color { r: 230, g: 90, b: 90, a: 255 },
        Color { r: 90, g: 170, b: 230, a: 255 },
        Color { r: 110, g: 210, b: 120, a: 255 },
        Color { r: 230, g: 190, b: 90, a: 255 },
        Color { r: 200, g: 120, b: 220, a: 255 },
        Color { r: 240, g: 150, b: 90, a: 255 },
    ];
    // FNV-1a over the name, then index the palette.
    let mut h: u32 = 2166136261;
    for b in name.bytes() {
        h = (h ^ b as u32).wrapping_mul(16777619);
    }
    PALETTE[h as usize % PALETTE.len()]
}
