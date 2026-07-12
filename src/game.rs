//! game.rs owns the in-world state — world, player, physics, console — and runs a
//! frame of it: input, movement, block interaction, mods, streaming, and drawing.
//! The window and the menu/play state machine live one level up in [`app`](crate::app);
//! a `Game` is handed the engine each frame and reports back whether to keep playing
//! or return to the menu.
use std::time::Instant;

use voxel_engine::{Camera3D, Color, DVec3, Engine, IVec2, Key, Vec2};

use crate::avatar::Pose;
use crate::block::AIR;
use crate::camera::{CameraMode, FlyAxes, GameCamera, ViewPose};
use crate::command;
use crate::harness::{CameraPose, DebugView};
use crate::console::{self, Console};
use crate::ui::{self, Anchor, Theme};
use crate::input::intent::{GameplayAxis, GameplayEvent, GameplayState, GlobalEvent, MenuEvent};
use crate::input::router::{Context, Router, View};
use crate::input::{look, movement};
use crate::interact;
use crate::math::{Aabb, Bounded};
use crate::minimap::{Minimap, MinimapConfig};
use crate::mods::{ModContext, Mods};
use crate::net::chat;
use crate::net::client::{Connection, Incoming};
use crate::player::Player;
use crate::presence::{self, Gait, RenderPose, Stance, TagVisibility, WireAction};
use crate::save;
use crate::settings::Settings;
use crate::sim::Simulation;
use crate::sky::Sky;
use crate::world::World;

/// How far the player can reach to break a block: 6 m, in world units.
const REACH: f64 = 6.0 * crate::math::PER_METER;

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
    /// First/third person and freecam modes, plus shake effects.
    camera: GameCamera,
    sim: Simulation,
    console: Console,
    /// The save slot this world belongs to.
    save_name: String,
    /// The live server connection when playing multiplayer; `None` in singleplayer.
    /// The player simulates locally and the server keeps everyone in sync.
    net: Option<Connection>,
    /// Animation state for the player's own third-person body — the same
    /// machine each remote player carries.
    local_anim: presence::Animator,
    /// The local walk-cycle phase, accumulated from horizontal travel, same as a peer's.
    local_gait: f64,
    /// Cached HUD coordinate line: the displayed values change far less often
    /// than the frame rate, so the format!/measure pair runs only on change.
    coord_cache: (i64, i64, i64, String),
    /// In-world UI look and HUD visibility (see [`ui::Theme`]).
    theme: Theme,
    /// Day/night clock, atmosphere colour, weather, and the lighting edge into
    /// voxel shading (see [`crate::sky`]).
    sky: Sky,
    /// Top-down minimap: throttled terrain raster drawn in the HUD corner.
    minimap: Minimap,
    /// What the app renders: `Normal` play, or `TerrainKey` for the
    /// harness's sky-hole detector (flat terrain key, sky/fog passes disabled).
    debug_view: DebugView,
    /// Monotone frame counter driving the temporal (dither) sequence — the game
    /// owns this truth; the engine exposes no frame index.
    frame_index: u64,
    /// Golden-harness capture mode: the pose is pinned by [`teleport`](Self::teleport)
    /// and the clock by [`set_day`](Self::set_day), so `update` consumes NO live
    /// input and does NOT advance the clock. Without this a stray desktop cursor
    /// or keypress (the window has focus while the harness runs) rotates the
    /// camera or drifts physics, and per-frame `sky.tick` walks the day off its
    /// pin over the streaming frames before capture — either silently corrupts
    /// the blessed shot.
    scripted: bool,
    /// The typed look/lane config this game draws with (was the `WATT_CLOUDS`/
    /// `WATT_WEATHER` env reads). `compose` reads the per-frame look lanes from
    /// it. The live path uses [`RenderConfig::default`]; the harness pins
    /// [`RenderConfig::golden`] via [`scripted`](Self::scripted). Read per frame,
    /// so live toggling is possible if ever wanted.
    render: crate::render_config::RenderConfig,
}

impl Game {
    pub fn new(world: World, player: Player, save_name: String) -> Self {
        Self {
            world,
            player,
            camera: GameCamera::new(),
            sim: Simulation::new(),
            console: Console::new(),
            save_name,
            net: None,
            local_anim: presence::Animator::default(),
            local_gait: 0.0,
            coord_cache: (i64::MIN, i64::MIN, i64::MIN, String::new()),
            theme: Theme::new(),
            sky: Sky::new(),
            minimap: Minimap::new(MinimapConfig::DEFAULT),
            debug_view: DebugView::Normal,
            frame_index: 0,
            scripted: false,
            render: crate::render_config::RenderConfig::default(),
        }
    }

    /// Build a headless, deterministic game for the golden-shot harness:
    /// a fresh world at `seed` and a player at the origin. The harness teleports
    /// the camera per shot ([`teleport`](Self::teleport)) and selects what to
    /// render with [`set_debug_view`](Self::set_debug_view).
    /// Pushed at world entry and on each in-game `/gfx` edit to adopt new render config.
    pub fn set_render_config(&mut self, render: crate::render_config::RenderConfig) {
        self.render = render;
    }

    pub fn scripted(seed: u64, render: crate::render_config::RenderConfig) -> Game {
        let world = World::with_config(seed as i64, render);
        let player = Player::new(DVec3::new(0.0, 80.0, 0.0));
        let mut g = Game::new(world, player, "scripted".to_string());
        g.scripted = true;
        g.render = render;
        g
    }

    /// Select what the app renders for a capture.
    pub fn set_debug_view(&mut self, view: DebugView) {
        self.debug_view = view;
    }

    /// Place the player (and thus the render camera) at `pose`. The harness
    /// drives the same `Player` → `camera_with_fov` path the game uses.
    pub fn teleport(&mut self, pose: CameraPose) {
        self.player.position = pose.pos;
        self.player.yaw = pose.yaw;
        self.player.pitch = pose.pitch;
    }

    /// Pin the day/night clock fraction (0.5 = noon, 0.0 = midnight). Fixes
    /// each shot's lighting before capture, driving the same `SkyClock` the
    /// `/time` command and net sync do.
    pub fn set_day(&mut self, day: f64) {
        self.sky.clock.set_day(day);
    }

    /// Swap the atmosphere colour table.
    pub fn set_palette(&mut self, palette: crate::sky::Palette) {
        self.sky.atmosphere.palette = palette;
    }

    /// Attach a server connection, turning this into a multiplayer session.
    pub fn with_net(mut self, net: Connection) -> Self {
        self.net = Some(net);
        self
    }

    pub fn save_name(&self) -> &str {
        &self.save_name
    }

    /// Surface a status line in the in-world console (save/load notices).
    pub fn notify(&mut self, line: impl Into<String>) {
        self.console.print(line);
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
    pub fn on_enter(&mut self, eng: &mut Engine, router: &mut Router) {
        router.set_captured(true);
        eng.disable_cursor();
    }

    /// Return the world's GPU meshes to the engine (called before the game is
    /// dropped when leaving to the menu).
    pub fn free_gpu(&mut self, eng: &mut Engine) {
        self.world.free_meshes(eng);
    }

    /// Advance one frame. Returns Signal::ExitToMenu when the player leaves.
    pub fn update(
        &mut self,
        eng: &mut Engine,
        router: &mut Router,
        mods: &mut Mods,
        settings: &mut Settings,
    ) -> Signal {
        // Clamp dt so a stall (window minimized, world load hitch) becomes one
        // slightly-long step instead of a single giant physics step that would
        // tunnel the player through terrain.
        let dt = eng.frame_time().min(0.1);

        // Scripted golden capture: pose pinned by `teleport`, clock pinned by
        // `set_day`. Consume no live input and do not advance the clock — only
        // step the deterministic world so terrain streams in before the frame is
        // grabbed. See the `scripted` field for why this can't be optional.
        if self.scripted {
            self.world.stream(self.player.position, eng);
            self.sim.advance(&mut self.world, dt);
            return Signal::Continue;
        }

        // Advance the day/night clock (singleplayer drives it locally; a server
        // sync overrides `day` on arrival).
        self.sky.tick(dt as f64);

        // Keep the HUD text scale in sync with the persisted setting.
        self.theme.scale = settings.ui_scale;

        // Drain the server first so edits and chat keep flowing even while the
        // console is open or the player stands still.
        let net_disconnected = {
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::NetEvents);
            self.apply_net_events()
        };
        if net_disconnected {
            self.console.print("* disconnected from server".to_string());
            return Signal::ExitToMenu;
        }

        // Report our own state to the server every frame — this doubles as the
        // keepalive heartbeat, so it must run even while the console is open
        // (otherwise the server's idle timeout kicks a chatting player).
        if let Some(net) = &mut self.net {
            net.send_move(
                self.player.position,
                self.player.yaw,
                self.player.pitch,
                Stance::of_player(&self.player),
            );
        }

        // The exclusive context for this frame: Text while the console captures
        // typing (the world freezes locally — peers keep moving over the net).
        router.set_context(if self.console.is_open() { Context::Text } else { Context::Gameplay });

        // The once-per-frame transition: snapshot this frame's intents into
        // locals, then drop the router borrow before any `&mut Engine` side
        // effects (screenshot, cursor grab, console open) run below.
        let mut is_text = false;
        let mut text_chars: Vec<char> = Vec::new();
        let mut text_edit = None;
        let mut move_input = None;
        let mut look_delta = Vec2::ZERO;
        let mut fly_axes = FlyAxes::default();
        let (mut do_break, mut do_place) = (false, false);
        let (mut toggle_inventory, mut toggle_crafting) = (false, false);
        let (mut nav_up, mut nav_down, mut nav_confirm) = (false, false, false);
        let (mut open_console, mut open_chat, mut toggle_capture) = (false, false, false);
        // Global events are evaluated unconditionally below, so bind once (no
        // dead initializer).
        let (g_escape, g_hud, g_shot, g_minimap);
        {
            let input = router.frame(eng, dt);
            match input.view() {
                View::Gameplay(gp) => {
                    move_input = Some(movement::MoveInput::from_view(&gp));
                    look_delta = gp.look();
                    // Same axes the player reads, reinterpreted by the freecam rig
                    // when the camera is detached (the two never both consume them).
                    fly_axes = FlyAxes {
                        forward: gp.axis(GameplayAxis::MoveZ) as f64,
                        right: gp.axis(GameplayAxis::MoveX) as f64,
                        up: gp.axis(GameplayAxis::MoveY) as f64,
                        boost: gp.state(GameplayState::Sprint),
                    };
                    do_break = gp.event(GameplayEvent::Break);
                    do_place = gp.event(GameplayEvent::Place);
                    toggle_inventory = gp.event(GameplayEvent::ToggleInventory);
                    toggle_crafting = gp.event(GameplayEvent::ToggleCrafting);
                    open_console = gp.event(GameplayEvent::OpenConsole);
                    open_chat = gp.event(GameplayEvent::OpenChat);
                    toggle_capture = gp.event(GameplayEvent::ToggleCapture);
                    nav_up = gp.overlay_nav(MenuEvent::Up);
                    nav_down = gp.overlay_nav(MenuEvent::Down);
                    nav_confirm = gp.overlay_nav(MenuEvent::Confirm);
                }
                View::Text(t) => {
                    is_text = true;
                    text_chars = t.chars().collect();
                    text_edit = t.edit();
                }
                // The game never routes to Menu; treat it like Text (inert).
                View::Menu(_) => is_text = true,
            }
            let global = input.global();
            g_escape = global.event(GlobalEvent::Escape);
            g_hud = global.event(GlobalEvent::CycleHud);
            g_shot = global.event(GlobalEvent::Screenshot);
            g_minimap = global.event(GlobalEvent::MinimapMode);
        }

        // Text context: the console owns all input; nothing else runs. Esc is
        // the game's call (the Text view has no bindable events), and here it
        // means "close the console", never "leave the world".
        if is_text {
            if g_escape {
                self.console.close();
                return Signal::Continue;
            }
            if let Some(line) = self.console.handle_input(&text_chars, text_edit) {
                self.submit_line(line, eng, settings);
            }
            return Signal::Continue;
        }

        // Esc closes an in-world mod overlay before leaving the world.
        if g_escape {
            if mods.close_overlay() {
                return Signal::Continue;
            }
            return Signal::ExitToMenu;
        }

        // Open the console: `/` (OpenConsole) pre-fills a slash, `T` (OpenChat)
        // does not. Drain the char queue so the opening key isn't also typed.
        if open_console || open_chat {
            self.console.open(open_console);
            while eng.get_char_pressed().is_some() {}
            return Signal::Continue;
        }

        if toggle_capture {
            toggle_mouse(eng, router);
        }
        if g_hud {
            self.theme.cycle_hud();
        }
        if g_shot {
            match eng.screenshot() {
                Some(path) => println!("screenshot queued: {}", path.display()),
                None => eprintln!("screenshot could not be queued"),
            }
        }
        if g_minimap {
            self.minimap.toggle_orientation();
        }

        // F5 cycles first/third-back/third-front; F6 toggles freecam.
        if eng.is_key_pressed(Key::F5) {
            self.camera.cycle_person();
        }
        if eng.is_key_pressed(Key::F6) {
            self.camera.toggle_freecam(&self.player, &self.world, settings.fov);
        }
        self.camera.fx.update(dt);

        // Exactly one thing consumes look/move per frame: the detached rig
        // (player frozen) or the player. See `CameraMode`.
        let detached = if let Some(rig) = self.camera.free_rig() {
            rig.look(look_delta);
            rig.fly(fly_axes, dt);
            true
        } else {
            // Look (inert while uncaptured — the query already zeroed the delta).
            look::apply(&mut self.player, look_delta);

            if let Some(input) = &move_input {
                let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::Physics);
                movement::update_player(&mut self.player, &self.world, input, dt);
            }
            // Advance the local walk cycle from horizontal travel, mirroring
            // how peers' phases accumulate from their snapshots.
            let v = self.player.velocity();
            self.local_gait += (v.x * v.x + v.z * v.z).sqrt() * dt as f64 * presence::STRIDE_FREQ;
            false
        };

        // Break is capture-gated in the query; freecam additionally can't act
        // on the world (the crosshair isn't where the player aims).
        if do_break && !detached {
            self.break_block(mods);
        }

        // Mods run once per frame here — never inside the voxel loop.
        let placements = {
            let mut ctx = ModContext {
                player: &mut self.player,
                world: &mut self.world,
                screen_w: eng.screen_width(),
                screen_h: eng.screen_height(),
                place: do_place,
                toggle_inventory,
                toggle_crafting,
                nav_up,
                nav_down,
                nav_confirm,
                placements: Vec::new(),
            };
            mods.update(eng, &mut ctx);
            ctx.placements
        };
        self.apply_placements(placements);

        // Load/mesh/unload chunks around the camera (the player, unless the
        // freecam rig has flown elsewhere), then step physics.
        let stream_center = match &self.camera.mode {
            CameraMode::Free { rig, .. } => rig.pos,
            CameraMode::Person(_) => self.player.position,
        };
        self.world.stream(stream_center, eng);

        // Refresh minimap (throttled).
        let p = self.player.position;
        let player_col = IVec2::new(p.x.floor() as i32, p.z.floor() as i32);
        self.minimap
            .refresh(eng, &self.world, player_col, Instant::now());
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
                    // The server echoes our OWN edits back too (that server-ordered
                    // echo is what converges racing edits on one cell); re-applying
                    // an edit we already made locally is harmless, just a redundant
                    // dirty-remesh per own edit — acceptable.
                    let id = save::parse_block(&mut self.world, &spec);
                    self.world.set_block(x, y, z, id);
                }
                Incoming::Chat { from_name, channel, text } => {
                    // Colour the scope tag and name so chat scans at a glance: a gold
                    // [global] tag, a blue <name>, and the message body white.
                    let name = ui::Line::of(ui::Role::Accent, format!("<{from_name}> "));
                    let line = if channel == chat::GLOBAL {
                        ui::Line::of(ui::Role::Warning, "[global] ")
                            .then(ui::Role::Accent, format!("<{from_name}> "))
                    } else {
                        name
                    };
                    self.console.push(line.then(ui::Role::Muted, text));
                }
                Incoming::Joined { name } => {
                    self.console.push(ui::Line::of(ui::Role::Positive, format!("* {name} joined")));
                }
                Incoming::Left { name } => {
                    self.console.push(ui::Line::of(ui::Role::Muted, format!("* {name} left")));
                }
                Incoming::Time { day } => self.sky.clock.set_day(day as f64),
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
        self.console.echo(&line);
        let before = settings.clone();
        let day_before = self.sky.clock.day();
        // Each output line already carries its role (System output vs Error
        // rejection), so there is nothing to guess — just show them.
        for out in command::execute(&line, &mut self.player, &self.world, settings, &mut self.sky) {
            self.console.push(out);
        }
        // A `/gfx` command edits settings; push the result to the engine and
        // world, and persist it, only when something actually changed.
        if *settings != before {
            settings.apply(eng);
            self.world.set_view_radius(settings.render_distance);
            self.world.set_lighting(settings.lighting, eng);
            // Occlusion/lod2 are entry-only (a live world stays as constructed); the
            // per-frame look lanes (clouds/weather) update immediately here.
            self.render = settings.render_config();
            settings.save();
        }
        // A `/time` change is shared: tell the server so every client's clock
        // follows (the server relays it and hands it to future joiners).
        if self.sky.clock.day() != day_before {
            if let Some(net) = &mut self.net {
                net.send_set_time(self.sky.clock.day() as f32);
            }
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
        self.local_anim.on_action(WireAction::Swing);
        // Tell the server (it validates and relays to everyone else). We apply
        // locally above for a responsive feel; the server is still authoritative.
        if let Some(net) = &mut self.net {
            net.send_edit(x, y, z, "air".to_string());
            net.send_swing();
        }
    }

    /// Apply the block placements mods queued this frame. A placement lands only
    /// in a non-obstacle cell (air, or a liquid it replaces) that doesn't overlap
    /// the player. Well-behaved mods (the crafting mod) ran an equivalent check
    /// before queueing — and before spending a block on it — so within one frame
    /// the two always agree; re-checking here is a cheap guard against a mod that
    /// queues without validating.
    fn apply_placements(&mut self, placements: Vec<(i32, i32, i32, crate::block::BlockId)>) {
        for (x, y, z, id) in placements {
            // Lands in any non-obstacle cell — air, or a passable liquid it replaces
            // (raycast hands back a liquid `previous` when aiming through water).
            if self.world.is_obstacle(x, y, z) {
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
            self.local_anim.on_action(WireAction::Swing);
            // Tell the server in the same portable spec form saves use; it
            // validates and relays, exactly like breaking does with "air".
            if let Some(net) = &mut self.net {
                let spec = save::block_spec(&self.world, id);
                net.send_edit(x, y, z, spec);
                net.send_swing();
            }
        }
    }

    /// Render the world and HUD.
    ///
    /// The camera is at the origin looking along the view direction, and all
    /// 3D draws are camera-relative. Differences are computed at f64 precision
    /// before narrowing to f32 for the GPU, keeping far terrain stable.
    pub fn draw(&mut self, eng: &mut Engine, mods: &mut Mods, fov: f32, shake: f32) {
        // The one pose this frame renders from: mode observation plus effects.
        let pose = self.camera.pose(&self.player, &self.world, fov, shake);
        let camera = pose.camera3d();
        let cam_pos = pose.eye;

        let p = self.player.position;
        // 0.1-block display resolution: only re-format when a shown digit moves.
        let key = ((p.x * 10.0) as i64, (p.y * 10.0) as i64, (p.z * 10.0) as i64);
        if (key.0, key.1, key.2) != (self.coord_cache.0, self.coord_cache.1, self.coord_cache.2) {
            let text = format!("X: {:.1}    Y: {:.1}    Z: {:.1}", p.x, p.y, p.z);
            self.coord_cache = (key.0, key.1, key.2, text);
        }
        let coord_text = self.coord_cache.3.clone();
        // Scripted (harness) frames pin the readout: a live FPS number is the
        // one nondeterministic pixel region in an otherwise reproducible shot,
        // and golden diffs must only ever see real rendering drift.
        let fps_text =
            if self.scripted { "-- FPS".to_string() } else { format!("{:2} FPS", eng.fps()) };
        let screen_w = eng.screen_width();
        let screen_h = eng.screen_height();
        let screen = (screen_w, screen_h);

        // `dt` steps each peer's animator (body-yaw follow, stance blend, swing).
        let dt = eng.frame_time();
        let peers = self.peer_draws(eng, &camera, &pose, dt);
        let online = self.net.as_ref().map(|net| net.peers().count() + 1);
        let ping_ms = self.net.as_ref().and_then(|net| net.ping_ms());

        // Compose the single per-frame lighting truth: the source for the
        // engine's per-frame UBO for sky/fog and avatar key lighting. The UBO is
        // the only path; legacy push lanes have been retired.
        //
        // Exposure is the render thread's latest metered+smoothed value,
        // sourced through `Engine::exposure_for_compose`; temporal smoothing
        // already happened render-side, so frame delta is passed only for
        // signature symmetry (unused there).
        // Allow pinning exposure to a fixed default for stable bless/debug output.
        static EXPOSURE_ON: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| !matches!(std::env::var("WATT_EXPOSURE").as_deref(), Ok("0")));
        let exposure = if *EXPOSURE_ON {
            eng.exposure_for_compose(dt)
        } else {
            voxel_engine::skeleton::Exposure::DEFAULT
        };
        // Scripted (harness) frames pin the dither phase: capture lands on an
        // arbitrary frame index under uncapped pacing, and a cycling blue-noise
        // phase is per-run LSB wobble on gradient/blend surfaces (water) that a
        // golden diff must never see.
        let dither_frame = if self.scripted { 0 } else { self.frame_index };
        let snapshot = crate::frame_snapshot::compose(
            &self.sky,
            pose.eye,
            dither_frame,
            exposure,
            &self.render,
        );
        let frame_uniforms = voxel_engine::skeleton::FrameUniformsGpu::from(&snapshot);
        self.frame_index = self.frame_index.wrapping_add(1);

        // TerrainKey: flat terrain, sky/fog disabled, magenta clear for the
        // sky-hole detector. Normal: real clear, no debug flat.
        let (clear, debug_flat) = match self.debug_view {
            DebugView::Normal => (self.sky.clear(), None),
            // Pure-magenta endpoints (255/0) decode identically under sRGB and raw
            // normalize, so the sky-hole detector's HDR key value is unchanged.
            DebugView::TerrainKey => {
                (crate::harness::SKY_KEY.to_linear(), Some(crate::harness::TERRAIN_KEY))
            }
        };

        let mut f = eng.begin_frame(clear);

        {
            // The pose's f64 eye is the render-space origin for camera rebase:
            // TAA's translation reprojection depends on this.
            // Lighting is decided when the 3D scope opens (no post-hoc setter):
            // the composed per-frame UBO carries the lighting truth in every mode
            // (the renderer overlays the debug-flat reserved key for TerrainKey).
            let mut f3 = f.begin_3d(
                &camera,
                pose.eye,
                voxel_engine::Lighting::Composed(frame_uniforms),
            );
            f3.set_debug_flat(debug_flat);
            if matches!(self.debug_view, DebugView::Normal) {
                let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::ListSky);
                self.sky.draw(&mut f3);
            }
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::ListWorld);
            self.world.render(&mut f3, cam_pos);
            // Other players: a six-box humanoid, head tracking their look and
            // body lazily following, limbs swinging with their gait. Poses are
            // already camera-relative (see peer_draws).
            for peer in &peers {
                Pose::resolve(&peer.pose, &peer.rig).draw(&mut f3, peer.color);
            }
            // The player's own body, whenever the camera can see it (third
            // person and freecam) — the same humanoid + animator the peers use.
            if self.camera.shows_body() {
                let feet = DVec3::new(
                    self.player.position.x,
                    self.player.feet_y(),
                    self.player.position.z,
                );
                let v = self.player.velocity();
                let speed = ((v.x * v.x + v.z * v.z).sqrt()) as f32;
                let rp = RenderPose {
                    feet: (feet - pose.eye).as_vec3(),
                    yaw: self.player.yaw,
                    pitch: self.player.pitch,
                    stance: Stance::of_player(&self.player),
                    gait: Gait::new(self.local_gait as f32, speed),
                };
                let rig = self.local_anim.step(&rp, dt);
                Pose::resolve(&rp, &rig).draw(&mut f3, peer_color(&self.save_name));
            }
        }

        let _hud = voxel_engine::profile::scope(voxel_engine::profile::Meter::ListHud);
        // Draw minimap.
        let player_col = IVec2::new(
            self.player.position.x.floor() as i32,
            self.player.position.z.floor() as i32,
        );
        self.minimap.draw(&mut f, screen, player_col, self.player.yaw);

        let theme = &self.theme;

        // Reticle and world-space name tags: shown in every mode but fully-off.
        if theme.hud.shows_world_ui() {
            theme.crosshair.draw(&mut f, screen);

            // Floating name tags over each visible player, in the peer's own
            // tint, fading with distance and dimming when terrain occludes the
            // head (instead of drawing full-strength through walls).
            for peer in &peers {
                if let Some((tag, alpha)) = peer.tag {
                    let fs = theme.fs(18);
                    let tw = f.measure_text(&peer.name, fs);
                    let c = peer.color;
                    console::shadowed(
                        &mut f,
                        &peer.name,
                        tag.x as i32 - tw / 2,
                        tag.y as i32,
                        fs,
                        Color::new(c.r, c.g, c.b, (alpha * 255.0) as u8),
                    );
                }
            }
        }

        // Informational HUD text: coords, help, FPS, player count. Full mode only.
        if theme.hud.shows_info() {
            ui::label(&mut f, theme, screen, Anchor::Top, (0, 12), 26, ui::Role::Primary.color(), &coord_text);
            ui::label(&mut f, theme, screen, Anchor::TopLeft, (10, 12), 20, ui::Role::Positive.color(), &fps_text);
            if let Some(count) = online {
                let text = match ping_ms {
                    Some(ms) => format!("players online: {count}   {ms} ms"),
                    None => format!("players online: {count}"),
                };
                ui::label(&mut f, theme, screen, Anchor::TopRight, (-12, 180), 20, ui::Role::Positive.color(), &text);
            }
        }

        // Enabled mods contribute their HUD as data; the core renders it over the
        // world, under the console. Mods never touch the frame themselves.
        let hud = mods.hud(&self.world, screen);
        ui::render_hud(&mut f, theme, screen, &hud);
        self.console.draw(&mut f, screen_w, screen_h);
    }

    /// Build the per-frame draw data for other players. `&mut self` because animator state advances here.
    fn peer_draws(&mut self, eng: &Engine, camera: &Camera3D, pose: &ViewPose, dt: f32) -> Vec<PeerDraw> {
        let Some(net) = &mut self.net else { return Vec::new() };
        let world = &self.world;
        let eye = pose.eye;
        let forward = pose.forward();
        let now = Instant::now();
        net.peers_mut()
            .map(|peer| {
                let r = peer.sample(now);
                let rp = RenderPose {
                    feet: (r.pos - eye).as_vec3(),
                    yaw: r.yaw,
                    pitch: r.pitch,
                    stance: r.stance,
                    gait: Gait::new(r.phase, r.speed),
                };
                let rig = peer.anim.step(&rp, dt);
                let head = r.pos + DVec3::new(0.0, Pose::HEAD_TOP as f64 + 0.2, 0.0);
                let to_head = head - eye;
                let dist = to_head.length();
                let tag = if to_head.dot(forward) > 0.0 && dist > 1e-6 {
                    // Raycast to dim tag when terrain occludes the head.
                    let occluded =
                        interact::raycast(world, eye, to_head / dist, (dist - 0.5).max(0.0))
                            .is_some();
                    match TagVisibility::of(dist, occluded) {
                        TagVisibility::Hidden => None,
                        TagVisibility::Visible { alpha } => {
                            Some((eng.world_to_screen(to_head.as_vec3(), camera), alpha))
                        }
                    }
                } else {
                    None
                };
                PeerDraw {
                    pose: rp,
                    rig,
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
    /// Camera-relative render pose (world minus eye, subtracted in f64, then
    /// narrowed) — safe to hand to the f32 immediate draws.
    pose: RenderPose,
    /// This frame's animator output (body yaw, stance blend, action swing).
    rig: presence::RigParams,
    color: Color,
    name: String,
    /// Screen position + fade alpha for the name tag, or `None` when
    /// off-screen, behind us, or out of range.
    tag: Option<(Vec2, f32)>,
}

/// Toggle capture and sync cursor grab with the OS.
fn toggle_mouse(eng: &mut Engine, router: &mut Router) {
    let captured = !router.captured();
    router.set_captured(captured);
    if captured {
        eng.disable_cursor();
    } else {
        eng.enable_cursor();
    }
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
    let h = crate::hash::fnv1a_32(name.as_bytes());
    PALETTE[h as usize % PALETTE.len()]
}
