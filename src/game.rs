//! game.rs owns the in-world state — world, player, physics, console — and runs a
//! frame of it: input, movement, block interaction, mods, streaming, and drawing.
//! The window and the menu/play state machine live one level up in [`app`](crate::app);
//! a `Game` is handed the engine each frame and reports back whether to keep playing
//! or return to the menu.
use std::time::Instant;

use voxel_engine::{Camera3D, Color, DVec3, Engine, IVec2, Key, Vec2};

use crate::audio::{AudioCtx, AudioDirector, PeerPose, PlayerPose, SoundEvent, SoundSystem, UiSound};
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
use crate::presence::{self, Eye, Feet, Gait, RenderPose, Stance, TagVisibility, WireAction};
use crate::save;
use crate::sched::Ctx as SchedCtx;
use crate::settings::Settings;
use crate::sim::Simulation;
use crate::sky::Sky;
use crate::world::World;

/// What a game update wants the app to do next.
pub enum Signal {
    /// Keep playing.
    Continue,
    /// Leave to the start menu (the app saves on the way out).
    ExitToMenu,
}

/// Everything [`Game::compose_phase`] decides before frame recording starts:
/// the camera pose, the composed per-frame lighting truth, peer render poses,
/// and the cached HUD strings — handed read-only to the scene and HUD phases.
struct Scene {
    pose: ViewPose,
    camera: Camera3D,
    peers: Vec<PeerDraw>,
    frame_uniforms: voxel_engine::skeleton::FrameUniformsGpu,
    clear: voxel_engine::LinearRgb,
    debug_flat: Option<Color>,
    coord_text: String,
    fps_text: String,
    screen: (i32, i32),
    online: Option<usize>,
    ping_ms: Option<u32>,
    dt: f32,
}

/// One frame's routed input, snapshotted into plain data by
/// [`Game::input_phase`] so the router borrow ends before later phases take
/// `&mut Engine`. Which fields are live depends on the frame's exclusive
/// context: `is_text` carries the console's typing, everything else gameplay.
#[derive(Default)]
struct FrameInput {
    is_text: bool,
    text_chars: Vec<char>,
    text_edit: Option<crate::input::intent::EditKey>,
    move_input: Option<movement::MoveInput>,
    look_delta: Vec2,
    fly_axes: FlyAxes,
    do_break: bool,
    do_place: bool,
    toggle_inventory: bool,
    toggle_crafting: bool,
    nav_up: bool,
    nav_down: bool,
    nav_confirm: bool,
    open_console: bool,
    open_chat: bool,
    toggle_capture: bool,
    /// Push-to-talk held this frame (level, not edge — see `GameplayState::PushToTalk`).
    ptt: bool,
    g_escape: bool,
    g_hud: bool,
    g_shot: bool,
    g_minimap: bool,
}

/// One optimistic edit awaiting the server's verdict: everything needed to
/// undo it if the verdict is a rejection.
struct PendingEdit {
    cell: (i32, i32, i32),
    /// What the cell held before the optimistic apply.
    prev: crate::block::BlockId,
    kind: PendingKind,
}

/// The economy side of a pending edit — what to give back on rejection.
enum PendingKind {
    /// Breaking awarded these elements; a rejection revokes them.
    Break(Vec<crate::block::ElementId>),
    /// Placing spent one crafted block of this id; a rejection refunds it.
    Place(crate::block::BlockId),
}

/// The live world the player is in.
pub struct Game {
    world: World,
    player: Player,
    /// First/third person and freecam modes, plus shake effects.
    camera: GameCamera,
    console: Console,
    /// The save slot this world belongs to.
    save_name: String,
    /// The live server connection when playing multiplayer; `None` in singleplayer.
    /// The player simulates locally and the server keeps everyone in sync.
    net: Option<Connection>,
    /// Rollback bookkeeping for optimistic edits awaiting a server verdict,
    /// keyed by the connection's request id: what the cell held before, and
    /// what the economy optimistically did (loot gained, item spent).
    pending_edits: std::collections::HashMap<u32, PendingEdit>,
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
    /// The producer scheduler owns the fixed-tick
    /// sim lane (registered in [`Game::new`]); other lanes still run directly
    /// in `stream_phase` and migrate in one at a time.
    sched: crate::sched::Scheduler,
    /// The autosave interval gate on the scheduler's frame clock
    /// (replaces the autosaver's `Instant` throttle). App reads
    /// [`Game::autosave_due`] and clears it via [`Game::mark_autosave`]; the
    /// save itself stays in `App` (its inputs — mods, meta, slot — live there).
    autosave_interval: crate::sched::IntervalHandle,
    /// The minimap throttle gate on the scheduler's frame clock
    /// (replaces the minimap's `last_refresh: Instant`). The recenter half of
    /// the gate stays a state comparison inside [`Minimap`].
    minimap_interval: crate::sched::IntervalHandle,
}

impl Game {
    pub fn new(mut world: World, player: Player, save_name: String) -> Self {
        let mut sched = crate::sched::Scheduler::new();
        // The fixed-tick sim runs through the scheduler. A pure clock lane is
        // never "starved", so its forward-progress floor is effectively infinite
        // — it fires only when whole ticks are due.
        let sim_id =
            sched.register(Simulation::manifest(), Box::new(Simulation::new()), u32::MAX);
        sched.set_meter(sim_id, voxel_engine::profile::Meter::Physics);
        // The autosave and minimap throttles are scheduler interval gates.
        let autosave_interval =
            sched.register_interval(crate::save::autosave::AUTOSAVE_INTERVAL.as_secs_f32());
        let minimap_interval =
            sched.register_interval(MinimapConfig::DEFAULT.refresh_every.as_secs_f32());
        // The stream lanes are call-point-driven producers (see
        // sched::Scheduler::manual); World drives them via run_manual at their
        // exact positions in stream(). One registration point keeps the lane
        // types private to crate::world.
        let stream_lanes = crate::world::lanes::StreamLanes::register(&mut sched);
        world.set_stream_lanes(stream_lanes);
        Self {
            world,
            player,
            camera: GameCamera::new(),
            console: Console::new(),
            save_name,
            net: None,
            pending_edits: std::collections::HashMap::new(),
            local_anim: presence::Animator::default(),
            local_gait: 0.0,
            coord_cache: (i64::MIN, i64::MIN, i64::MIN, String::new()),
            theme: Theme::new(),
            sky: Sky::new(),
            minimap: Minimap::new(MinimapConfig::DEFAULT),
            debug_view: DebugView::Normal,
            scripted: false,
            render: crate::render_config::RenderConfig::default(),
            sched,
            autosave_interval,
            minimap_interval,
        }
    }

    /// Whether the autosave interval has elapsed on the scheduler's frame clock.
    /// `App` gates a new save on this ∧ the world being dirty.
    pub fn autosave_due(&self) -> bool {
        self.sched.interval_due(self.autosave_interval)
    }

    /// Clear the autosave interval when a save is attempted — matches the old
    /// `last_attempt` bump, so the next save is a full interval out.
    pub fn mark_autosave(&mut self) {
        self.sched.interval_reset(self.autosave_interval);
    }

    /// Build a headless, deterministic game for the golden-shot harness:
    /// a fresh world at `seed` and a player at the origin. The harness teleports
    /// the camera per shot ([`teleport`](Self::teleport)) and selects what to
    /// render with [`set_debug_view`](Self::set_debug_view).
    /// Pushed at world entry and on each in-game `/gfx` edit to adopt new render config.
    pub fn set_render_config(&mut self, render: crate::render_config::RenderConfig) {
        self.render = render;
    }

    /// Push every live-applicable setting into this game: engine values, the
    /// world's view radius and lighting lane, and the per-frame look config.
    /// THE one path — world entry and in-game `/gfx` edits both come through
    /// here, so they can never drift apart. (World-construction lanes
    /// — occlusion/lod2 — stay entry-only by design; see `App::enter_game`.)
    pub fn apply_settings(&mut self, eng: &mut Engine, settings: &mut Settings) {
        settings.apply(eng);
        self.world.set_view_radius(settings.render_distance);
        self.world.set_lighting(settings.lighting, eng);
        self.world.set_ao(settings.ao, eng);
        self.render = settings.render_config();
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
        self.player.orientation.yaw = pose.yaw;
        self.player.orientation.pitch = pose.pitch;
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
    /// One frame of in-world logic, as a sequence of named phases. Each phase
    /// is a plain method — the flow reads top to bottom and any early Signal
    /// short-circuits the rest of the frame, exactly as before the split.
    pub fn update(
        &mut self,
        eng: &mut Engine,
        router: &mut Router,
        mods: &mut Mods,
        settings: &mut Settings,
        sound: &mut SoundSystem,
        audio: &mut AudioDirector,
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
            self.world.stream(self.player.position, eng, &mut self.sched);
            let clocks = self.sched.clocks(dt);
            let mut sched_ctx = SchedCtx::new(&mut self.world, Some(&mut *eng));
            self.sched.tick(&mut sched_ctx, &clocks);
            return Signal::Continue;
        }

        // Advance the day/night clock (singleplayer drives it locally; a server
        // sync overrides `day` on arrival). The day_night lane freezes it at
        // the current time of day — permanent daylight without a special case
        // anywhere downstream (compose still reads the clock every frame).
        if self.render.day_night {
            self.sky.tick(dt as f64);
        }

        // Keep the HUD text scale in sync with the persisted setting.
        self.theme.scale = settings.ui_scale;

        // This frame's unrecoverable audio facts, accumulated across the
        // phases and folded by the director. Everything else — footsteps, splash, the
        // underwater bed, voice sessions — the director DERIVES from the readout.
        let mut events: Vec<SoundEvent> = Vec::new();

        if let Some(signal) = self.net_phase(mods, &mut events) {
            return signal;
        }
        let input = self.input_phase(eng, router, dt);
        // The overlay may consume the frame (console typing, opening chat): movement,
        // interaction and streaming run only on an unconsumed ("active") frame, exactly
        // as before. Audio, though, commits EVERY frame so voice/emitters/faults — and
        // the `/voicetest` cue submitted while the console is open — stay live; only a
        // real exit short-circuits it.
        let active =
            match self.overlay_phase(&input, eng, router, mods, settings, sound, &mut events) {
                Some(Signal::ExitToMenu) => return Signal::ExitToMenu,
                Some(Signal::Continue) => false,
                None => {
                    let detached = self.motion_phase(&input, dt);
                    self.interact_phase(&input, detached, eng, mods, &mut events);
                    self.stream_phase(eng, dt);
                    true
                }
            };
        self.commit_audio(dt, &input, sound, audio, settings, events, active);
        Signal::Continue
    }

    /// Drain server events and send our heartbeat. `Some(ExitToMenu)` when the
    /// server dropped us. Runs before input so edits and chat keep flowing even
    /// while the console is open or the player stands still — and the move
    /// report doubles as the keepalive, so it too runs unconditionally.
    fn net_phase(&mut self, mods: &mut Mods, events: &mut Vec<SoundEvent>) -> Option<Signal> {
        let net_disconnected = {
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::NetEvents);
            self.apply_net_events(mods, events)
        };
        if net_disconnected {
            self.console.print("* disconnected from server".to_string());
            return Some(Signal::ExitToMenu);
        }
        if let Some(net) = &mut self.net {
            net.send_move(
                self.player.position,
                self.player.orientation.yaw,
                self.player.orientation.pitch,
                Stance::of_player(&self.player),
            );
        }
        None
    }

    /// The once-per-frame router transition: pick the exclusive context (Text
    /// while the console captures typing) and snapshot every intent into plain
    /// data, so the router borrow ends before any `&mut Engine` side effects
    /// (screenshot, cursor grab, console open) run in later phases.
    fn input_phase(&mut self, eng: &mut Engine, router: &mut Router, dt: f32) -> FrameInput {
        router.set_context(if self.console.is_open() { Context::Text } else { Context::Gameplay });

        let mut f = FrameInput::default();
        let input = router.frame(eng, dt);
        match input.view() {
            View::Gameplay(gp) => {
                f.move_input = Some(movement::MoveInput::from_view(&gp));
                f.look_delta = gp.look();
                // Same axes the player reads, reinterpreted by the freecam rig
                // when the camera is detached (the two never both consume them).
                f.fly_axes = FlyAxes {
                    forward: gp.axis(GameplayAxis::MoveZ) as f64,
                    right: gp.axis(GameplayAxis::MoveX) as f64,
                    up: gp.axis(GameplayAxis::MoveY) as f64,
                    boost: gp.state(GameplayState::Sprint),
                };
                f.do_break = gp.event(GameplayEvent::Break);
                f.do_place = gp.event(GameplayEvent::Place);
                f.toggle_inventory = gp.event(GameplayEvent::ToggleInventory);
                f.toggle_crafting = gp.event(GameplayEvent::ToggleCrafting);
                f.open_console = gp.event(GameplayEvent::OpenConsole);
                f.open_chat = gp.event(GameplayEvent::OpenChat);
                f.toggle_capture = gp.event(GameplayEvent::ToggleCapture);
                f.ptt = gp.state(GameplayState::PushToTalk);
                f.nav_up = gp.overlay_nav(MenuEvent::Up);
                f.nav_down = gp.overlay_nav(MenuEvent::Down);
                f.nav_confirm = gp.overlay_nav(MenuEvent::Confirm);
            }
            View::Text(t) => {
                f.is_text = true;
                f.text_chars = t.chars().collect();
                f.text_edit = t.edit();
            }
            // The game never routes to Menu; treat it like Text (inert).
            View::Menu(_) => f.is_text = true,
        }
        let global = input.global();
        f.g_escape = global.event(GlobalEvent::Escape);
        f.g_hud = global.event(GlobalEvent::CycleHud);
        f.g_shot = global.event(GlobalEvent::Screenshot);
        f.g_minimap = global.event(GlobalEvent::MinimapMode);
        f
    }

    /// Console, escape routing, and the global toggles (mouse capture, HUD
    /// cycle, screenshot, minimap, camera modes). `Some` consumes the frame:
    /// while typing, nothing below the console runs.
    fn overlay_phase(
        &mut self,
        input: &FrameInput,
        eng: &mut Engine,
        router: &mut Router,
        mods: &mut Mods,
        settings: &mut Settings,
        sound: &mut SoundSystem,
        events: &mut Vec<SoundEvent>,
    ) -> Option<Signal> {
        // Text context: the console owns all input; nothing else runs. Esc is
        // the game's call (the Text view has no bindable events), and here it
        // means "close the console", never "leave the world".
        if input.is_text {
            if input.g_escape {
                self.console.close();
                return Some(Signal::Continue);
            }
            if let Some(line) = self.console.handle_input(&input.text_chars, input.text_edit) {
                self.submit_line(line, eng, settings, sound, events);
            }
            return Some(Signal::Continue);
        }

        // Esc closes an in-world mod overlay before leaving the world.
        if input.g_escape {
            if mods.close_overlay() {
                return Some(Signal::Continue);
            }
            return Some(Signal::ExitToMenu);
        }

        // Open the console: `/` (OpenConsole) pre-fills a slash, `T` (OpenChat)
        // does not. Drain the char queue so the opening key isn't also typed.
        if input.open_console || input.open_chat {
            self.console.open(input.open_console);
            while eng.get_char_pressed().is_some() {}
            return Some(Signal::Continue);
        }

        if input.toggle_capture {
            toggle_mouse(eng, router);
        }
        if input.g_hud {
            self.theme.cycle_hud();
        }
        if input.g_shot {
            match eng.screenshot() {
                Some(path) => println!("screenshot queued: {}", path.display()),
                None => eprintln!("screenshot could not be queued"),
            }
        }
        if input.g_minimap {
            self.minimap.toggle_orientation();
        }

        // F5 cycles first/third-back/third-front; F6 toggles freecam.
        if eng.is_key_pressed(Key::F5) {
            self.camera.cycle_person();
        }
        if eng.is_key_pressed(Key::F6) {
            // Reattaching after the rig flew far away resumes physics at the
            // frozen player, whose chunks may have streamed out (the centre
            // followed the camera). Restore the collision halo synchronously
            // BEFORE the toggle so the first reattached step never runs
            // against unloaded air.
            if self.camera.free_rig().is_some() {
                self.world.prepare_around(self.player.position);
            }
            self.camera.toggle_freecam(&self.player, &self.world, settings.fov);
        }
        None
    }

    /// Camera effects plus movement. Exactly one thing consumes look/move per
    /// frame — the detached freecam rig (player frozen) or the player; returns
    /// whether the rig had it (see `CameraMode`).
    fn motion_phase(&mut self, input: &FrameInput, dt: f32) -> bool {
        self.camera.fx.update(dt);
        if let Some(rig) = self.camera.free_rig() {
            rig.look(input.look_delta);
            rig.fly(input.fly_axes, dt);
            true
        } else {
            // Look (inert while uncaptured — the query already zeroed the delta).
            look::apply(&mut self.player, input.look_delta);

            if let Some(mi) = &input.move_input {
                let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::Physics);
                movement::update_player(&mut self.player, &self.world, mi, dt);
            }
            // Advance the local walk cycle from horizontal travel so the third-person
            // body animates. The AUDIO gait (footstep phase-crossings) is derived
            // inside the director from the same speed — this one drives rendering only.
            let v = self.player.velocity();
            let speed = (v.x * v.x + v.z * v.z).sqrt();
            self.local_gait += speed * dt as f64 * presence::STRIDE_FREQ;
            false
        }
    }

    /// World edits: block breaking, then the mods' frame hooks and whatever
    /// placements they queued. Mods run once per frame here — never inside the
    /// voxel loop.
    fn interact_phase(
        &mut self,
        input: &FrameInput,
        detached: bool,
        eng: &mut Engine,
        mods: &mut Mods,
        events: &mut Vec<SoundEvent>,
    ) {
        // Break is capture-gated in the query; freecam additionally can't act
        // on the world (the crosshair isn't where the player aims).
        if input.do_break && !detached {
            self.break_block(mods, events);
        }

        let placements = {
            let mut ctx = ModContext {
                player: &mut self.player,
                world: &mut self.world,
                screen_w: eng.screen_width(),
                screen_h: eng.screen_height(),
                // A detached camera cannot perform player-origin actions: its
                // crosshair no longer represents the frozen player's aim.
                place: input.do_place && !detached,
                toggle_inventory: input.toggle_inventory,
                toggle_crafting: input.toggle_crafting,
                nav_up: input.nav_up,
                nav_down: input.nav_down,
                nav_confirm: input.nav_confirm,
                placements: Vec::new(),
            };
            mods.update(eng, &mut ctx);
            ctx.placements
        };
        self.apply_placements(placements, events);
    }

    /// Load/mesh/unload chunks around the camera (the player, unless the
    /// freecam rig has flown elsewhere), refresh the minimap (throttled), and
    /// step the simulation.
    fn stream_phase(&mut self, eng: &mut Engine, dt: f32) {
        // The scheduler drives the fixed-tick sim lane. Its clock
        // (fixed-tick accumulator + catch-up cap) is derived once per frame
        // here; other lanes still run directly below until they migrate in.
        let clocks = self.sched.clocks(dt);
        let mut sched_ctx = SchedCtx::new(&mut self.world, Some(&mut *eng));
        self.sched.tick(&mut sched_ctx, &clocks);

        let stream_center = match &self.camera.mode {
            CameraMode::Free { rig, .. } => rig.pos,
            CameraMode::Person(_) => self.player.position,
        };
        self.world.stream(stream_center, eng, &mut self.sched);

        let p = self.player.position;
        let player_col = IVec2::new(p.x.floor() as i32, p.z.floor() as i32);
        // The minimap throttle rides the scheduler's interval gate (advanced in
        // clocks() above); the recenter half stays inside Minimap::due. Reset
        // the gate whenever a rebuild actually happens (either trigger).
        let due = self.sched.interval_due(self.minimap_interval);
        if self.minimap.refresh(eng, &self.world, player_col, due) {
            self.sched.interval_reset(self.minimap_interval);
        }
    }

    /// Hand this frame's readout to the audio director: it folds
    /// the drained `events`, derives the rest from the trace (footsteps, splash,
    /// underwater bed, voice sessions), commits the [`AudioFrame`], and services the
    /// voice/capture path. The window, medium, gait and session bookkeeping that used
    /// to live here are the director's now.
    fn commit_audio(
        &mut self,
        dt: f32,
        input: &FrameInput,
        sound: &mut SoundSystem,
        audio: &mut AudioDirector,
        settings: &Settings,
        events: Vec<SoundEvent>,
        active: bool,
    ) {
        // THE per-frame peer sample: one `Instant`, consumed by the director for
        // both remote footsteps and voice sessions. `peer_draws` in draw() keeps its
        // own richer sample — it runs in the separate draw() call, steps each peer's
        // animator, and needs render fields absent from `PeerPose`.
        let now = Instant::now();
        let peers: Vec<PeerPose> = self
            .net
            .as_ref()
            .map(|net| {
                net.peers()
                    .map(|p| {
                        let r = p.sample(now);
                        PeerPose {
                            id: p.id(),
                            at: r.pos.0,
                            visible: p.visible(),
                            phase: r.phase,
                            speed: r.speed,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        // On a console-owned frame the player isn't stepped, so freeze the listener
        // velocity: a stale walk speed would fire phantom footsteps in the director.
        let velocity = if active { self.player.velocity() } else { DVec3::ZERO };
        let player = PlayerPose {
            pos: self.player.position,
            yaw: self.player.orientation.yaw,
            pitch: self.player.orientation.pitch,
            velocity,
            on_ground: self.player.on_ground(),
        };

        let ctx = AudioCtx {
            dt,
            player,
            ptt: input.ptt,
            voice_enabled: settings.voice_enabled,
            events,
            peers: &peers,
            world: &self.world,
            net: self.net.as_mut(),
            console: &mut self.console,
        };
        audio.frame(ctx, sound);
    }

    /// Drain queued server messages: apply world edits, resolve our own edit
    /// verdicts (rolling back rejected predictions), surface chat, and report
    /// a lost connection. Returns `true` if the server dropped us.
    fn apply_net_events(&mut self, mods: &mut Mods, events: &mut Vec<SoundEvent>) -> bool {
        let incoming = match &mut self.net {
            Some(net) => net.poll(),
            None => return false,
        };
        let mut disconnected = false;
        for event in incoming {
            match event {
                Incoming::Edit { x, y, z, spec } => {
                    // Resolve the portable spec against our own palette, then
                    // apply. The connection already dropped stale revisions,
                    // and our own edits come back as acks, not broadcasts.
                    // Snapshot the cell first so a remote break names the block that
                    // WAS there (its sound class), not a generic default.
                    let prev = self.world.block_at(x, y, z);
                    let id = save::parse_block(&mut self.world, &spec);
                    self.world.set_block(x, y, z, id);
                    let at = cell_center(x, y, z);
                    events.push(if id == AIR {
                        SoundEvent::BlockBroken { at, block: prev }
                    } else {
                        SoundEvent::BlockPlaced { at, block: id }
                    });
                }
                Incoming::EditAccepted { req } => {
                    // Prediction confirmed: the optimistic apply IS the truth.
                    self.pending_edits.remove(&req);
                }
                Incoming::EditRejected { req, restore } => {
                    let Some(pending) = self.pending_edits.remove(&req) else { continue };
                    if restore {
                        let (x, y, z) = pending.cell;
                        self.world.set_block(x, y, z, pending.prev);
                    }
                    match pending.kind {
                        PendingKind::Break(elements) => mods.on_break_rejected(&elements),
                        PendingKind::Place(id) => mods.on_place_rejected(id, &self.world),
                    }
                }
                Incoming::Position { pos } => {
                    // Authoritative snap-back (refused teleport or implausible
                    // move): land safely, exactly like a local teleport.
                    self.world.prepare_around(pos);
                    self.player.position = pos;
                    self.player.cancel_fall();
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
                Incoming::Time { day, day_secs } => {
                    // The server owns the shared clock: phase AND cycle length.
                    self.sky.clock.set_day(day as f64);
                    self.sky.day_length = crate::sky::DayLength::clamped(day_secs as f64);
                }
                Incoming::Disconnected => disconnected = true,
                Incoming::PeerSwing { id } => {
                    // The swing edge → a whoosh at the peer's current position. The
                    // local animator update already happened in `Connection::apply`.
                    if let Some(peer) = self.net.as_ref().and_then(|net| net.peers().find(|p| p.id() == id))
                    {
                        events.push(SoundEvent::PeerSwing { at: peer.sample(Instant::now()).pos.0 });
                    }
                }
            }
        }
        disconnected
    }

    /// Handle one submitted console line. A leading `/` is always a local command; in
    /// multiplayer any other line is chat (a leading `!` sends it to global chat),
    /// while in singleplayer it stays a command as before.
    fn submit_line(
        &mut self,
        line: String,
        eng: &mut Engine,
        settings: &mut Settings,
        sound: &mut SoundSystem,
        events: &mut Vec<SoundEvent>,
    ) {
        // `/voicetest` plays the canned UI cue; emit it as a fact and let the
        // director route it (it runs this frame even though the console owns input).
        // `execute` below prints the acknowledgement.
        if line.trim() == "/voicetest" {
            events.push(SoundEvent::Ui(UiSound::VoiceTest));
        }
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
        let day_len_before = self.sky.day_length;
        let pos_before = self.player.position;
        // Each output line already carries its role (System output vs Error
        // rejection), so there is nothing to guess — just show them.
        for out in command::execute(&line, &mut self.player, &mut self.world, settings, &mut self.sky) {
            self.console.push(out);
        }
        // A `/gfx` command edits settings; push the result through the one
        // application path and persist it, only when something actually changed.
        if *settings != before {
            self.apply_settings(eng, settings);
            settings.save();
            // Push the audio mix through the one committer whenever `/gfx`-style
            // settings edits touch a volume/mute/deafen row.
            sound.set_mix(settings.mix_change());
        }
        // A `/time` change is shared: tell the server so every client's clock
        // follows (the server relays it and hands it to future joiners).
        if self.sky.clock.day() != day_before {
            if let Some(net) = &mut self.net {
                net.send_set_time(self.sky.clock.day() as f32);
            }
        }
        // The cycle LENGTH is server-owned in multiplayer: a local change
        // would silently desync every clock's advance rate.
        if self.sky.day_length != day_len_before && self.net.is_some() {
            self.sky.day_length = day_len_before;
            self.console.print("* day length is set by the server".to_string());
        }
        // A `/tp` is a position discontinuity: ordinary moves are envelope-
        // checked server-side, so report it as an explicit teleport (the
        // server may still snap us back if teleports are disabled).
        if self.player.position != pos_before {
            if let Some(net) = &mut self.net {
                net.send_teleport(self.player.position);
            }
        }
    }

    /// Break the block the player is looking at, handing its elements to the mods.
    fn break_block(&mut self, mods: &mut Mods, events: &mut Vec<SoundEvent>) {
        let Some(hit) =
            interact::raycast(
                &self.world,
                self.player.position,
                self.player.forward(),
                interact::REACH,
            )
        else {
            return;
        };
        let (x, y, z) = hit.block;
        let id = self.world.block_at(x, y, z);
        // Snapshot the block's elements before it's removed.
        let elements = self.world.registry().block(id).composition.elements();
        // Report the broken block; the director derives its class-specific cue.
        events.push(SoundEvent::BlockBroken { at: cell_center(x, y, z), block: id });
        self.world.set_block(x, y, z, AIR);
        mods.on_block_break(&elements, &self.world);
        self.local_anim.on_action(WireAction::Swing);
        // Tell the server (it validates and relays to everyone else). The
        // apply above is a PREDICTION for responsiveness: the ack rolls it
        // back — cell and loot both — if we lose the race for this cell.
        if let Some(net) = &mut self.net {
            let req = net.send_edit(x, y, z, "air".to_string());
            self.pending_edits.insert(
                req,
                PendingEdit {
                    cell: (x, y, z),
                    prev: id,
                    kind: PendingKind::Break(elements.to_vec()),
                },
            );
            net.send_swing();
        }
    }

    /// Apply the block placements mods queued this frame. A placement lands only
    /// in a non-obstacle cell (air, or a liquid it replaces) that doesn't overlap
    /// the player. Well-behaved mods (the crafting mod) ran an equivalent check
    /// before queueing — and before spending a block on it — so within one frame
    /// the two always agree; re-checking here is a cheap guard against a mod that
    /// queues without validating.
    fn apply_placements(
        &mut self,
        placements: Vec<(i32, i32, i32, crate::block::BlockId)>,
        events: &mut Vec<SoundEvent>,
    ) {
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
            let prev = self.world.block_at(x, y, z);
            // Report the placed block; the director derives its class-specific cue.
            events.push(SoundEvent::BlockPlaced { at: cell_center(x, y, z), block: id });
            self.world.set_block(x, y, z, id);
            self.local_anim.on_action(WireAction::Swing);
            // Tell the server in the same portable spec form saves use; it
            // validates and relays, exactly like breaking does with "air".
            // The spent crafted block is refunded if the server says no.
            if let Some(net) = &mut self.net {
                let spec = save::block_spec(&self.world, id);
                let req = net.send_edit(x, y, z, spec);
                self.pending_edits.insert(
                    req,
                    PendingEdit { cell: (x, y, z), prev, kind: PendingKind::Place(id) },
                );
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
        let scene = self.compose_phase(eng, fov, shake);
        let mut f = eng.begin_frame(scene.clear);
        self.scene_phase(&mut f, &scene);
        self.hud_phase(&mut f, mods, &scene);
    }

    /// Everything a frame needs decided BEFORE recording starts: the camera
    /// pose, the per-frame lighting truth (the engine UBO's single source),
    /// peer render poses, and the cached HUD strings.
    fn compose_phase(&mut self, eng: &mut Engine, fov: f32, shake: f32) -> Scene {
        // The one pose this frame renders from: mode observation plus effects.
        let pose = self.camera.pose(&self.player, &self.world, fov, shake);
        let camera = pose.camera3d();

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
        let screen = (eng.screen_width(), eng.screen_height());

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
        let snapshot = crate::frame_snapshot::compose(
            &self.sky,
            pose.eye,
            exposure,
            &self.render,
        );
        let frame_uniforms = voxel_engine::skeleton::FrameUniformsGpu::from(&snapshot);

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

        Scene {
            pose,
            camera,
            peers,
            frame_uniforms,
            clear,
            debug_flat,
            coord_text,
            fps_text,
            screen,
            online,
            ping_ms,
            dt,
        }
    }

    /// The 3D scope: sky, world, and every humanoid, all camera-relative.
    fn scene_phase(&mut self, f: &mut voxel_engine::Frame, scene: &Scene) {
        let Scene { pose, camera, peers, dt, .. } = scene;
        {
            // The pose's f64 eye is the render-space origin for camera rebase:
            // TAA's translation reprojection depends on this.
            // Lighting is decided when the 3D scope opens (no post-hoc setter):
            // the composed per-frame UBO carries the lighting truth in every mode
            // (the renderer overlays the debug-flat reserved key for TerrainKey).
            let mut f3 = f.begin_3d(
                camera,
                pose.eye,
                voxel_engine::Lighting::Composed(scene.frame_uniforms),
            );
            f3.set_debug_flat(scene.debug_flat);
            if matches!(self.debug_view, DebugView::Normal) {
                let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::ListSky);
                self.sky.draw(&mut f3);
            }
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::ListWorld);
            self.world.render(&mut f3, pose.eye);
            // Other players: a six-box humanoid, head tracking their look and
            // body lazily following, limbs swinging with their gait. Poses are
            // already camera-relative (see peer_draws).
            for peer in peers {
                Pose::resolve(&peer.pose, &peer.rig).draw(&mut f3, peer.color, true);
            }
            // The player's own body — always drawn so it casts a shadow and is
            // visible when looking down. In first person the head is omitted (the
            // camera sits inside it); third person and freecam show the full body.
            {
                let feet = Feet(DVec3::new(
                    self.player.position.x,
                    self.player.feet_y(),
                    self.player.position.z,
                ));
                let v = self.player.velocity();
                let speed = ((v.x * v.x + v.z * v.z).sqrt()) as f32;
                let rp = RenderPose::new(
                    feet,
                    Eye(pose.eye),
                    self.player.orientation.yaw,
                    self.player.orientation.pitch,
                    Stance::of_player(&self.player),
                    Gait::new(self.local_gait as f32, speed),
                );
                let rig = self.local_anim.step(&rp, *dt);
                Pose::resolve(&rp, &rig).draw(
                    &mut f3,
                    peer_color(&self.save_name),
                    self.camera.shows_body(),
                );
            }
        }
    }

    /// Everything over the world: minimap, reticle, name tags, info text, the
    /// mods' HUD data (rendered by the core — mods never touch the frame), and
    /// the console on top.
    fn hud_phase(&mut self, f: &mut voxel_engine::Frame, mods: &mut Mods, scene: &Scene) {
        let screen = scene.screen;
        let _hud = voxel_engine::profile::scope(voxel_engine::profile::Meter::ListHud);
        let theme = &self.theme;

        // Minimap: informational, so Full mode only (HUD Off must blank it too).
        if theme.hud.shows_minimap() {
            let player_col = IVec2::new(
                self.player.position.x.floor() as i32,
                self.player.position.z.floor() as i32,
            );
            self.minimap.draw(f, screen, player_col, self.player.orientation.yaw);
        }

        // Reticle and world-space name tags: shown in every mode but fully-off.
        if theme.hud.shows_world_ui() {
            theme.crosshair.draw(f, screen);

            // Floating name tags over each visible player, in the peer's own
            // tint, fading with distance and dimming when terrain occludes the
            // head (instead of drawing full-strength through walls).
            for peer in &scene.peers {
                if let Some((tag, alpha)) = peer.tag {
                    let fs = theme.fs(18);
                    let tw = f.measure_text(&peer.name, fs);
                    let c = peer.color;
                    console::shadowed(
                        f,
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
            ui::label(f, theme, screen, Anchor::Top, (0, 12), 26, ui::Role::Primary.color(), &scene.coord_text);
            ui::label(f, theme, screen, Anchor::TopLeft, (10, 12), 20, ui::Role::Positive.color(), &scene.fps_text);
            if let Some(count) = scene.online {
                let text = match scene.ping_ms {
                    Some(ms) => format!("players online: {count}   {ms} ms"),
                    None => format!("players online: {count}"),
                };
                ui::label(f, theme, screen, Anchor::TopRight, (-12, 180), 20, ui::Role::Positive.color(), &text);
            }
        }

        // Enabled mods contribute their HUD as data; the core renders it over the
        // world, under the console. Mods never touch the frame themselves.
        // Gameplay UI, so it follows the reticle: hidden only when HUD is Off.
        if theme.hud.shows_mod_hud() {
            let hud = mods.hud(&self.world, screen);
            ui::render_hud(f, theme, screen, &hud);
        }
        self.console.draw(f, screen.0, screen.1);
    }

    /// Build the per-frame draw data for other players. `&mut self` because animator state advances here.
    fn peer_draws(&mut self, eng: &Engine, camera: &Camera3D, pose: &ViewPose, dt: f32) -> Vec<PeerDraw> {
        let Some(net) = &mut self.net else { return Vec::new() };
        let world = &self.world;
        let eye = pose.eye;
        let forward = pose.forward();
        let now = Instant::now();
        net.peers_mut()
            // Outside interest range there is no live pose: drawing the last
            // heard one would freeze a ghost in place.
            .filter(|peer| peer.visible())
            .map(|peer| {
                let r = peer.sample(now);
                let feet = r.pos.feet(r.stance);
                let rp = RenderPose::new(feet, Eye(eye), r.yaw, r.pitch, r.stance, Gait::new(r.phase, r.speed));
                let rig = peer.anim.step(&rp, dt);
                let head = feet.0 + DVec3::new(0.0, Pose::HEAD_TOP as f64 + 0.2, 0.0);
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

/// The world-space centre of a voxel cell (occurrence position).
fn cell_center(x: i32, y: i32, z: i32) -> DVec3 {
    DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5)
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
