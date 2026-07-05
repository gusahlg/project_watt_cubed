//! game.rs owns the in-world state — world, player, physics, console — and runs a
//! frame of it: input, movement, block interaction, mods, streaming, and drawing.
//! The window and the menu/play state machine live one level up in [`app`](crate::app);
//! a `Game` is handed the engine each frame and reports back whether to keep playing
//! or return to the menu.
use voxel_engine::{Camera3D, Color, DVec3, Engine, Key, MouseButton, Vec2, Vec3};

use crate::block::AIR;
use crate::command;
use crate::console::{self, Console};
use crate::input::{look, movement};
use crate::interact;
use crate::math::{Aabb, Bounded};
use crate::mods::{ModContext, Mods};
use crate::net::chat;
use crate::net::client::{Connection, Incoming};
use crate::player::Player;
use crate::save;
use crate::settings::Settings;
use crate::sim::Simulation;
use crate::world::World;

/// How far the player can reach to break a block, in world units.
const REACH: f64 = 6.0;
const HELP_TEXT: &str = "WASD move | mouse look | Space jump | F fly | LMB break | I inventory | C craft | Tab cursor | T chat/cmd | Esc menu";
/// Half-extents of another player's drawn body — matches the collision box in
/// [`player`](crate::player::PLAYER_HALF). `f64` like all position math; cast
/// to `f32` only for the (camera-relative) draw calls.
const PEER_HALF: DVec3 = DVec3::new(0.3, 0.9, 0.3);
/// Peers past this distance get no floating name tag (it would be unreadable).
const TAG_RANGE: f64 = 90.0;

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
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }
    pub fn player(&self) -> &Player {
        &self.player
    }
    pub fn player_mut(&mut self) -> &mut Player {
        &mut self.player
    }

    /// Capture the cursor when (re)entering play.
    pub fn on_enter(&mut self, eng: &mut Engine) {
        self.mouse_locked = true;
        eng.disable_cursor();
    }

    /// Return the world's GPU meshes to the engine (called before the game is
    /// dropped when leaving to the menu).
    pub fn free_gpu(&mut self, eng: &mut Engine) {
        self.world.free_meshes(eng);
    }

    /// Advance one frame. Returns [`Signal::ExitToMenu`] when the player leaves.
    pub fn update(
        &mut self,
        eng: &mut Engine,
        mods: &mut Mods,
        settings: &mut Settings,
    ) -> Signal {
        // Clamp dt so a stall (window minimized, world load hitch) becomes one
        // slightly-long step instead of a single giant physics step that would
        // tunnel the player through terrain.
        let dt = eng.frame_time().min(0.1);

        // Drain the server first so edits and chat keep flowing even while the
        // console is open or the player stands still.
        if self.apply_net_events() {
            self.console.print("* disconnected from server".to_string());
            return Signal::ExitToMenu;
        }

        // Report our own state to the server every frame — this doubles as the
        // keepalive heartbeat, so it must run even while the console is open
        // (otherwise the server's idle timeout kicks a chatting player).
        if let Some(net) = &mut self.net {
            net.send_move(self.player.position, self.player.yaw, self.player.pitch);
        }

        // While the console is open it captures all typing; the world is frozen
        // (locally — other players keep moving over the network).
        if self.console.is_open() {
            if let Some(line) = self.console.handle_input(eng) {
                self.submit_line(line, eng, settings);
            }
            return Signal::Continue;
        }

        // Esc (console closed) leaves to the menu.
        if eng.is_key_pressed(Key::Escape) {
            return Signal::ExitToMenu;
        }

        // Open the console with `T`, or `/` to start a command straight away.
        let slash = eng.is_key_pressed(Key::Slash);
        if slash || eng.is_key_pressed(Key::T) {
            self.console.open(slash);
            while eng.get_char_pressed().is_some() {}
            return Signal::Continue;
        }

        if eng.is_key_pressed(Key::Tab) {
            self.toggle_mouse(eng);
        }

        if self.mouse_locked {
            look::update(&mut self.player, eng);
        }

        let input = movement::MoveInput::from_input(eng);
        movement::update_player(&mut self.player, &self.world, &input, dt);

        // Break the aimed-at block into its elements while actually aiming (cursor
        // locked, not navigating a free cursor).
        if self.mouse_locked && eng.is_mouse_button_pressed(MouseButton::Left) {
            self.break_block(mods);
        }

        // Mods run once per frame here — never inside the voxel loop.
        let placements = {
            let mut ctx = ModContext {
                player: &mut self.player,
                world: &mut self.world,
                screen_w: eng.screen_width(),
                screen_h: eng.screen_height(),
                capturing_text: false,
                mouse_locked: self.mouse_locked,
                placements: Vec::new(),
            };
            mods.update(eng, &mut ctx);
            ctx.placements
        };
        self.apply_placements(placements);

        // Load/mesh/unload chunks around the player, then step physics.
        self.world.stream(self.player.position, eng);
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
    fn submit_line(&mut self, line: String, eng: &mut Engine, settings: &mut Settings) {
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
        let before = settings.clone();
        for out in command::execute(&line, &mut self.player, &self.world, settings) {
            self.console.print(out);
        }
        // A `/gfx` command edits settings; push the result to the engine and
        // world, and persist it, only when something actually changed.
        if *settings != before {
            settings.apply(eng);
            self.world.set_view_radius(settings.render_distance);
            settings.save();
        }
    }

    /// Break the block the player is looking at, handing its elements to the mods.
    fn break_block(&mut self, mods: &mut Mods) {
        let Some(hit) =
            interact::raycast(&self.world, self.player.position, self.player.forward(), REACH)
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

    /// Apply the block placements mods queued this frame. A placement lands only
    /// in an air cell that doesn't overlap the player. Well-behaved mods (the
    /// crafting mod) ran this exact check before queueing — and before spending a
    /// block on it — so within one frame the two always agree; re-checking here is
    /// a cheap invariant against a mod that queues without validating.
    fn apply_placements(&mut self, placements: Vec<(i32, i32, i32, crate::block::BlockId)>) {
        for (x, y, z, id) in placements {
            if self.world.block_at(x, y, z) != AIR {
                continue;
            }
            // Overlap check in f64: at far coordinates an f32 cell centre
            // would land whole blocks away from the real cell.
            let cell = Aabb::new(
                DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5),
                DVec3::splat(0.5),
            );
            if cell.intersects(&self.player.aabb()) {
                continue;
            }
            self.world.set_block(x, y, z, id);
            // Tell the server in the same portable spec form saves use; it
            // validates and relays, exactly like breaking does with "air".
            if let Some(net) = &mut self.net {
                let spec = save::block_spec(&self.world, id);
                net.send_edit(x, y, z, spec);
            }
        }
    }

    fn toggle_mouse(&mut self, eng: &mut Engine) {
        self.mouse_locked = !self.mouse_locked;
        if self.mouse_locked {
            eng.disable_cursor();
        } else {
            eng.enable_cursor();
        }
    }

    /// Render the world and HUD (owns its own draw pass for the frame).
    ///
    /// CAMERA REBASE: the camera sits at `Vec3::ZERO` looking along the view
    /// direction ([`Player::camera_with_fov`](crate::player::Player)), and
    /// every 3D draw is camera-relative — the world passes per-chunk offsets
    /// to `draw_mesh`, peers subtract the eye. All differences are taken in
    /// `f64` first, so only *small* camera-local values ever reach the `f32`
    /// GPU path; the world can be 1e9 blocks wide without a vertex jittering.
    pub fn draw(&mut self, eng: &mut Engine, mods: &mut Mods, fov: f32) {
        let camera = self.player.camera_with_fov(fov);
        let cam_pos = self.player.position;

        let p = self.player.position;
        let coord_text = format!("X: {:.1}    Y: {:.1}    Z: {:.1}", p.x, p.y, p.z);
        let coord_fs = 26;
        let screen_w = eng.screen_width();
        let screen_h = eng.screen_height();
        let coord_x = (screen_w - eng.measure_text(&coord_text, coord_fs)) / 2;

        // Gather the other players to draw (camera-relative), projecting a head
        // point to screen space for the floating name tags.
        let peers = self.peer_draws(eng, &camera);
        let online = self.net.as_ref().map(|net| net.peers().count() + 1);

        let mut f = eng.begin_frame(Color::SKYBLUE);

        {
            let mut f3 = f.begin_3d(&camera);
            self.world.render(&mut f3, cam_pos);
            // Other players: a body box and a small head, tinted per player.
            // `peer.pos` is already camera-relative (see `peer_draws`).
            for peer in &peers {
                let body = (PEER_HALF * 2.0).as_vec3();
                f3.draw_cube(peer.pos, body, peer.color);
                f3.draw_cube_wires(peer.pos, body, Color::BLACK);
                let head = peer.pos + Vec3::new(0.0, PEER_HALF.y as f32 + 0.2, 0.0);
                f3.draw_cube(head, Vec3::splat(0.4), peer.color);
            }
        }

        // Aiming crosshair at the screen centre.
        let (cx, cy) = (screen_w / 2, screen_h / 2);
        let cross = Color::new(255, 255, 255, 180);
        f.draw_line(cx - 8, cy, cx + 8, cy, cross);
        f.draw_line(cx, cy - 8, cx, cy + 8, cross);

        console::shadowed(&mut f, &coord_text, coord_x, 12, coord_fs, Color::WHITE);
        f.draw_fps(10, 12);
        console::shadowed(&mut f, HELP_TEXT, 10, 40, 16, Color::RAYWHITE);

        // Floating name tags over each visible player.
        for peer in &peers {
            if let Some(tag) = peer.tag {
                let fs = 18;
                let tw = f.measure_text(&peer.name, fs);
                console::shadowed(
                    &mut f,
                    &peer.name,
                    tag.x as i32 - tw / 2,
                    tag.y as i32,
                    fs,
                    Color::WHITE,
                );
            }
        }
        if let Some(count) = online {
            let text = format!("players online: {count}");
            let w = f.measure_text(&text, 20);
            console::shadowed(&mut f, &text, screen_w - w - 12, 12, 20, Color::LIME);
        }

        // Enabled mods draw their HUD over the world, under the console.
        mods.draw(&mut f, screen_w, screen_h);
        self.console.draw(&mut f, screen_w, screen_h);
    }

    /// Build the per-frame draw data for other players, projecting a head point to
    /// screen space for the name tag (only for peers in front and within range).
    /// The in-front/range filters run in `f64`; the projection takes the
    /// CAMERA-RELATIVE head with the origin-based camera, matching the scene.
    fn peer_draws(&self, eng: &Engine, camera: &Camera3D) -> Vec<PeerDraw> {
        let Some(net) = &self.net else { return Vec::new() };
        let eye = self.player.position;
        let forward = self.player.forward();
        net.peers()
            .map(|peer| {
                let head = peer.pos + DVec3::new(0.0, PEER_HALF.y + 0.4, 0.0);
                let to_head = head - eye;
                let visible = to_head.dot(forward) > 0.0 && to_head.length() <= TAG_RANGE;
                let tag = visible.then(|| eng.world_to_screen(to_head.as_vec3(), camera));
                PeerDraw {
                    pos: (peer.pos - eye).as_vec3(),
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
    /// Camera-relative position (world position minus the eye, subtracted in
    /// f64, then narrowed) — safe to hand to the f32 immediate draws.
    pos: Vec3,
    color: Color,
    name: String,
    /// Screen position for the name tag, or `None` when off-screen/behind us.
    tag: Option<Vec2>,
}

/// A stable, cheerful colour for a player, hashed from their name so the same player
/// keeps the same tint across clients.
fn peer_color(name: &str) -> Color {
    const PALETTE: [Color; 6] = [
        Color::new(230, 90, 90, 255),
        Color::new(90, 170, 230, 255),
        Color::new(110, 210, 120, 255),
        Color::new(230, 190, 90, 255),
        Color::new(200, 120, 220, 255),
        Color::new(240, 150, 90, 255),
    ];
    // FNV-1a over the name, then index the palette.
    let mut h: u32 = 2166136261;
    for b in name.bytes() {
        h = (h ^ b as u32).wrapping_mul(16777619);
    }
    PALETTE[h as usize % PALETTE.len()]
}
