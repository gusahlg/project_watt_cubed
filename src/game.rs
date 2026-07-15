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
use crate::console::{self, Console};
use crate::harness::{CameraPose, DebugView};
use crate::input::intent::{GameplayEvent, GlobalEvent, MenuEvent};
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
use crate::settings::Settings;
use crate::sim::Simulation;
use crate::sky::Sky;
use crate::ui::{self, Anchor, HudMode, Theme};
use crate::world::World;

/// Catch-up bank shared by the independently throttled game clocks. Bounding
/// it prevents a pause/debugger break from turning one render frame into an
/// unbounded burst of work.
const MAX_RATE_ACCUMULATED: f32 = 0.25;
/// Human-readable FPS text need not track every instantaneous sample. Four
/// refreshes per second stays responsive while avoiding format/measure churn.
const FPS_LABEL_INTERVAL: f32 = 0.25;

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
    sky_frame: crate::sky::SkyFrame,
    peers: Vec<PeerDraw>,
    frame_uniforms: voxel_engine::skeleton::FrameUniformsGpu,
    clear: voxel_engine::LinearRgb,
    debug_flat: Option<Color>,
    screen: (i32, i32),
    dt: f32,
}

/// Lighting/clear state for a profile whose sky and animation inputs are
/// frozen. Wrapped-water coordinates are cached independently so camera motion
/// does not force the palette and lighting packet to be recomposed.
#[derive(Clone, Copy)]
struct StaticFrameCache {
    day_bits: u64,
    uniforms: voxel_engine::skeleton::FrameUniformsGpu,
    clear: voxel_engine::LinearRgb,
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
    g_escape: bool,
    g_hud: bool,
    g_shot: bool,
    g_minimap: bool,
}

/// Edge-triggered mod intents from one render frame, retained in order when mod
/// updates run at a fixed cadence.
#[derive(Clone, Copy, Default)]
struct PendingModInput {
    place: bool,
    /// Cell selected by the edge frame's player pose. Cadence-delayed mod
    /// replay must not re-raycast from a later camera direction.
    place_target: Option<(i32, i32, i32)>,
    toggle_inventory: bool,
    toggle_crafting: bool,
    nav_up: bool,
    nav_down: bool,
    nav_confirm: bool,
}

impl PendingModInput {
    fn capture(
        input: &FrameInput,
        allow_place: bool,
        allow_ui: bool,
        place_target: Option<(i32, i32, i32)>,
    ) -> Self {
        let place = allow_place && input.do_place;
        Self {
            place,
            place_target: place.then_some(place_target).flatten(),
            toggle_inventory: allow_ui && input.toggle_inventory,
            toggle_crafting: allow_ui && input.toggle_crafting,
            nav_up: allow_ui && input.nav_up,
            nav_down: allow_ui && input.nav_down,
            nav_confirm: allow_ui && input.nav_confirm,
        }
    }

    fn any(self) -> bool {
        self.place
            || self.toggle_inventory
            || self.toggle_crafting
            || self.nav_up
            || self.nav_down
            || self.nav_confirm
    }

    fn clear_ui(&mut self) {
        self.toggle_inventory = false;
        self.toggle_crafting = false;
        self.nav_up = false;
        self.nav_down = false;
        self.nav_confirm = false;
    }
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
    /// Optional so the minimum preset pays neither construction nor tick cost.
    sim: Option<Simulation>,
    /// Outer 20 Hz admission bank. This keeps the enabled simulation driver's
    /// own accumulator and virtual dispatch off uncapped render frames.
    sim_call_accumulator: f32,
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
    /// Last displayed integer FPS and its formatted label. `None` until Full
    /// HUD is actually composed, avoiding hidden string work entirely.
    fps_cache: Option<(i32, String)>,
    fps_refresh_accumulator: f32,
    /// Cached multiplayer count/ping label, likewise built only for Full HUD.
    online_cache: Option<(usize, Option<u32>, String)>,
    /// In-world UI look and HUD visibility (see [`ui::Theme`]).
    theme: Theme,
    /// Day/night clock, atmosphere colour, weather, and the lighting edge into
    /// voxel shading (see [`crate::sky`]).
    sky: Sky,
    /// Clock sample reused while the day value is unchanged. Fixed-day and
    /// scripted/minimum modes therefore perform no steady-frame sun trig.
    sky_frame_cache: Option<(f64, crate::sky::SkyFrame)>,
    /// Full composed lighting and clear colour reused by fixed stripped
    /// profiles until the visual time actually changes.
    static_frame_cache: Option<StaticFrameCache>,
    /// Precision-preserving world anchor for still water, recomputed only when
    /// camera XZ changes rather than on every rendered frame.
    anim_uv_cache: Option<([u64; 2], [f32; 2])>,
    /// Camera orientation is independent of its rebased f64 eye. Cache the
    /// f32 engine camera until yaw/pitch/roll/FOV changes, avoiding steady-frame
    /// f64 trigonometry.
    camera_cache: Option<([u32; 4], Camera3D)>,
    /// Top-down minimap: throttled terrain raster drawn in the HUD corner.
    minimap: Option<Minimap>,
    /// Expensive subsystems that can be independently stripped out.
    mod_logic: bool,
    mod_hud: bool,
    /// Set only when a visible mod UI is hidden through settings. The next
    /// overlay phase closes any modal exactly once; stripped steady-state
    /// frames never dispatch into the mod stack just to discover no UI.
    pending_mod_overlay_close: bool,
    player_models: bool,
    name_tags: bool,
    /// World streaming cadence. Zero preserves the historical every-frame path.
    stream_hz: u32,
    stream_interval: f32,
    stream_accumulator: f32,
    force_stream: bool,
    /// Player physics cadence. Look/camera effects remain render-rate responsive.
    physics_hz: u32,
    physics_interval: f32,
    physics_accumulator: f32,
    /// Day/night clock cadence. A fixed rate lets the cached palette/lighting
    /// packet survive the render frames between visually meaningful updates.
    sky_hz: u32,
    sky_interval: f32,
    sky_accumulator: f32,
    /// Enabled mod hooks can run at their own rate. Edge intents accumulate in
    /// `pending_mod_input` so a high-FPS render loop cannot lose a click/key.
    mod_hz: u32,
    mod_interval: f32,
    mod_accumulator: f32,
    pending_mod_input: Vec<PendingModInput>,
    /// Edge input must survive render frames that do not execute a physics tick.
    pending_toggle_fly: bool,
    pending_jump: bool,
    /// Reused frame buffers: neither mod placements nor peer draw records need
    /// to allocate afresh on a stable frame.
    placement_scratch: Vec<(i32, i32, i32, crate::block::BlockId)>,
    peer_scratch: Vec<PeerDraw>,
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
            sim: None,
            sim_call_accumulator: 0.0,
            console: Console::new(),
            save_name,
            net: None,
            pending_edits: std::collections::HashMap::new(),
            local_anim: presence::Animator::default(),
            local_gait: 0.0,
            coord_cache: (i64::MIN, i64::MIN, i64::MIN, String::new()),
            fps_cache: None,
            fps_refresh_accumulator: FPS_LABEL_INTERVAL,
            online_cache: None,
            theme: Theme::new(),
            sky: Sky::new(),
            sky_frame_cache: None,
            static_frame_cache: None,
            anim_uv_cache: None,
            camera_cache: None,
            minimap: None,
            mod_logic: true,
            mod_hud: true,
            pending_mod_overlay_close: false,
            player_models: true,
            name_tags: true,
            stream_hz: 0,
            stream_interval: 0.0,
            stream_accumulator: 0.0,
            force_stream: true,
            physics_hz: 0,
            physics_interval: 0.0,
            physics_accumulator: 0.0,
            sky_hz: 0,
            sky_interval: 0.0,
            sky_accumulator: 0.0,
            mod_hz: 0,
            mod_interval: 0.0,
            mod_accumulator: 0.0,
            pending_mod_input: Vec::new(),
            pending_toggle_fly: false,
            pending_jump: false,
            placement_scratch: Vec::new(),
            peer_scratch: Vec::new(),
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
        self.static_frame_cache = None;
        self.anim_uv_cache = None;
    }

    /// Push every live-applicable setting into this game: engine values, the
    /// world's view radius and lighting lane, and the per-frame look config.
    /// THE one path — world entry and in-game `/gfx` edits both come through
    /// here, so engine flags, view/LOD state, meshing inputs, and look lanes
    /// cannot drift apart.
    pub fn apply_settings(&mut self, eng: &mut Engine, settings: &mut Settings) {
        let mod_ui_was_active = self.mod_ui_active();
        settings.apply(eng);
        let render = settings.render_config();
        self.world
            .set_view_distances(settings.render_distance, settings.vertical_distance);
        self.world.set_render_config(render, eng);
        self.world
            .set_meshing_config(settings.lighting, settings.ao, eng);
        self.render = render;
        self.static_frame_cache = None;
        self.anim_uv_cache = None;

        self.theme.scale = settings.ui_scale;
        self.theme.hud = match settings.hud_mode {
            crate::settings::HUD_OFF => HudMode::Off,
            crate::settings::HUD_MINIMAL => HudMode::Minimal,
            _ => HudMode::Full,
        };

        if self.stream_hz != settings.stream_hz {
            self.stream_hz = settings.stream_hz;
            self.stream_interval = rate_interval(settings.stream_hz);
            self.stream_accumulator = 0.0;
        }
        if self.physics_hz != settings.physics_hz {
            self.physics_hz = settings.physics_hz;
            self.physics_interval = rate_interval(settings.physics_hz);
            self.physics_accumulator = 0.0;
        }
        if self.sky_hz != settings.sky_hz {
            self.sky_hz = settings.sky_hz;
            self.sky_interval = rate_interval(settings.sky_hz);
            self.sky_accumulator = 0.0;
        }
        if self.mod_hz != settings.mod_hz {
            self.mod_hz = settings.mod_hz;
            self.mod_interval = rate_interval(settings.mod_hz);
            self.mod_accumulator = 0.0;
        }
        self.force_stream = true;

        if settings.simulation {
            if self.sim.is_none() {
                self.sim = Some(Simulation::new());
                self.sim_call_accumulator = 0.0;
            }
        } else {
            self.sim = None;
            self.sim_call_accumulator = 0.0;
        }
        if settings.minimap {
            self.minimap
                .get_or_insert_with(|| Minimap::new(MinimapConfig::DEFAULT));
        } else {
            self.minimap = None;
        }
        let mod_ui_will_be_active = settings.mod_logic
            && settings.mod_hud
            && self.theme.hud.shows_mod_hud();
        if mod_ui_will_be_active {
            // If visibility is restored before the next frame, the overlay is
            // visible again and does not need to be force-closed.
            self.pending_mod_overlay_close = false;
        } else if mod_ui_was_active {
            self.pending_mod_overlay_close = true;
            for pending in &mut self.pending_mod_input {
                pending.clear_ui();
            }
            self.pending_mod_input.retain(|pending| pending.any());
        }
        self.mod_logic = settings.mod_logic;
        if !self.mod_logic {
            self.mod_accumulator = 0.0;
            self.pending_mod_input.clear();
        }
        self.mod_hud = settings.mod_hud;
        self.player_models = settings.player_models;
        self.name_tags = settings.name_tags;
    }

    pub fn scripted(seed: u64, render: crate::render_config::RenderConfig) -> Game {
        let world = World::with_config(seed as i64, render);
        let player = Player::new(DVec3::new(0.0, 80.0, 0.0));
        let mut g = Game::new(world, player, "scripted".to_string());
        g.scripted = true;
        g.render = render;
        // Harness captures retain the historical full game presentation without
        // going through the live Settings application path.
        g.sim = Some(Simulation::new());
        g.minimap = Some(Minimap::new(MinimapConfig::DEFAULT));
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
        self.static_frame_cache = None;
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

    /// Whether a modal supplied by the mod layer can both be seen and receive
    /// input. Keeping one predicate for routing and Escape prevents invisible
    /// overlays when either the mod lane or the master HUD is disabled.
    fn mod_ui_active(&self) -> bool {
        self.mod_logic && self.mod_hud && self.theme.hud.shows_mod_hud()
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
            if let Some(sim) = &mut self.sim {
                sim.advance(&mut self.world, dt);
            }
            return Signal::Continue;
        }

        // Advance the day/night clock (singleplayer drives it locally; a server
        // sync overrides `day` on arrival). With this lane off, compose samples
        // fixed noon without mutating authoritative clock state; re-enabling
        // resumes the stored time instead of freezing a stripped profile at night.
        if self.render.day_night {
            let steps = rate_steps(&mut self.sky_accumulator, self.sky_interval, dt);
            if steps != 0 {
                let sky_dt = if self.sky_hz == 0 {
                    dt as f64
                } else {
                    steps as f64 / self.sky_hz as f64
                };
                self.sky.tick(sky_dt);
            }
        } else if self.sky_accumulator != 0.0 {
            self.sky_accumulator = 0.0;
        }

        if let Some(signal) = self.net_phase(mods) {
            return signal;
        }
        let input = self.input_phase(eng, router, dt);
        if let Some(signal) = self.overlay_phase(&input, eng, router, mods, settings) {
            return signal;
        }
        let detached = self.motion_phase(&input, dt);
        self.interact_phase(&input, detached, router.captured(), dt, eng, mods);
        self.stream_phase(eng, dt);
        Signal::Continue
    }

    /// Drain server events and send our heartbeat. `Some(ExitToMenu)` when the
    /// server dropped us. Runs before input so edits and chat keep flowing even
    /// while the console is open or the player stands still — and the move
    /// report doubles as the keepalive, so it too runs unconditionally.
    fn net_phase(&mut self, mods: &mut Mods) -> Option<Signal> {
        // The overwhelmingly common singleplayer path should not even enter a
        // profiling scope or call through the event-poll seam.
        self.net.as_ref()?;
        let net_disconnected = {
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::NetEvents);
            self.apply_net_events(mods)
        };
        if net_disconnected {
            self.console.print("* disconnected from server".to_string());
            return Some(Signal::ExitToMenu);
        }
        if let Some(net) = &mut self.net {
            net.send_move(
                self.player.position,
                self.player.yaw,
                self.player.pitch,
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
        router.set_context(if self.console.is_open() {
            Context::Text
        } else {
            Context::Gameplay
        });

        let mut f = FrameInput::default();
        let mod_ui = self.mod_ui_active();
        let input = router.frame_filtered(
            eng,
            dt,
            self.mod_logic,
            mod_ui,
            self.minimap.is_some(),
        );
        match input.view() {
            View::Gameplay(gp) => {
                let move_input = movement::MoveInput::from_view(&gp);
                f.look_delta = gp.look();
                // Same axes the player reads, reinterpreted by the freecam rig
                // when the camera is detached (the two never both consume them).
                let (forward, right, up, boost) = move_input.freecam_axes();
                f.fly_axes = FlyAxes {
                    forward,
                    right,
                    up,
                    boost,
                };
                f.do_break = gp.event(GameplayEvent::Break);
                if self.mod_logic {
                    f.do_place = gp.event(GameplayEvent::Place);
                }
                if mod_ui {
                    f.toggle_inventory = gp.event(GameplayEvent::ToggleInventory);
                    f.toggle_crafting = gp.event(GameplayEvent::ToggleCrafting);
                    f.nav_up = gp.overlay_nav(MenuEvent::Up);
                    f.nav_down = gp.overlay_nav(MenuEvent::Down);
                    f.nav_confirm = gp.overlay_nav(MenuEvent::Confirm);
                }
                f.open_console = gp.event(GameplayEvent::OpenConsole);
                f.open_chat = gp.event(GameplayEvent::OpenChat);
                f.toggle_capture = gp.event(GameplayEvent::ToggleCapture);
                f.move_input = Some(move_input);
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
    ) -> Option<Signal> {
        if std::mem::take(&mut self.pending_mod_overlay_close) {
            mods.close_overlay();
        }

        // Text context: the console owns all input; nothing else runs. Esc is
        // the game's call (the Text view has no bindable events), and here it
        // means "close the console", never "leave the world".
        if input.is_text {
            self.pending_mod_input.clear();
            self.mod_accumulator = 0.0;
            self.pending_toggle_fly = false;
            self.pending_jump = false;
            if input.g_escape {
                self.console.close();
                return Some(Signal::Continue);
            }
            if let Some(line) = self
                .console
                .handle_input(&input.text_chars, input.text_edit)
            {
                self.submit_line(line, eng, settings);
            }
            return Some(Signal::Continue);
        }

        // Esc closes an in-world mod overlay before leaving the world.
        if input.g_escape {
            self.pending_mod_input.clear();
            self.mod_accumulator = 0.0;
            self.pending_toggle_fly = false;
            self.pending_jump = false;
            if self.mod_ui_active() && mods.close_overlay() {
                return Some(Signal::Continue);
            }
            return Some(Signal::ExitToMenu);
        }

        // Open the console: `/` (OpenConsole) pre-fills a slash, `T` (OpenChat)
        // does not. Drain the char queue so the opening key isn't also typed.
        if input.open_console || input.open_chat {
            self.pending_mod_input.clear();
            self.mod_accumulator = 0.0;
            self.pending_toggle_fly = false;
            self.pending_jump = false;
            self.console.open(input.open_console);
            while eng.get_char_pressed().is_some() {}
            return Some(Signal::Continue);
        }

        if input.toggle_capture {
            // A placement edge captured before the cursor is released must not
            // fire later after a throttled mod tick.
            for pending in &mut self.pending_mod_input {
                pending.place = false;
                pending.place_target = None;
            }
            self.pending_mod_input.retain(|pending| pending.any());
            toggle_mouse(eng, router);
        }
        if input.g_hud {
            let mod_ui_was_active = self.mod_ui_active();
            self.theme.cycle_hud();
            settings.hud_mode = match self.theme.hud {
                HudMode::Off => crate::settings::HUD_OFF,
                HudMode::Minimal => crate::settings::HUD_MINIMAL,
                HudMode::Full => crate::settings::HUD_FULL,
            };
            settings.mark_custom();
            settings.save();
            if mod_ui_was_active && !self.mod_ui_active() {
                self.pending_mod_overlay_close = true;
                for pending in &mut self.pending_mod_input {
                    pending.clear_ui();
                }
                self.pending_mod_input.retain(|pending| pending.any());
            }
            self.force_stream = true;
        }
        if input.g_shot {
            match eng.screenshot() {
                Some(path) => println!("screenshot queued: {}", path.display()),
                None => eprintln!("screenshot could not be queued"),
            }
        }
        if input.g_minimap && let Some(minimap) = &mut self.minimap {
            minimap.toggle_orientation();
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
            self.camera
                .toggle_freecam(&self.player, &self.world, settings.fov);
            self.force_stream = true;
        }
        None
    }

    /// Camera effects plus movement. Exactly one thing consumes look/move per
    /// frame — the detached freecam rig (player frozen) or the player; returns
    /// whether the rig had it (see `CameraMode`).
    fn motion_phase(&mut self, input: &FrameInput, dt: f32) -> bool {
        self.camera.fx.update(dt);
        if let Some(rig) = self.camera.free_rig() {
            self.pending_toggle_fly = false;
            self.pending_jump = false;
            rig.look(input.look_delta);
            rig.fly(input.fly_axes, dt);
            true
        } else {
            // Look (inert while uncaptured — the query already zeroed the delta).
            look::apply(&mut self.player, input.look_delta);

            if let Some(mi) = &input.move_input {
                self.pending_toggle_fly |= mi.toggle_fly();
                self.pending_jump |= mi.jump();
                let steps = rate_steps(&mut self.physics_accumulator, self.physics_interval, dt);
                if steps != 0 {
                    let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::Physics);
                    let step_dt = if self.physics_hz == 0 {
                        dt
                    } else {
                        1.0 / self.physics_hz as f32
                    };
                    for step in 0..steps {
                        let mut tick_input = *mi;
                        tick_input.set_toggle_fly(step == 0 && self.pending_toggle_fly);
                        tick_input.set_jump(mi.jump() || (step == 0 && self.pending_jump));
                        movement::update_player(
                            &mut self.player,
                            &self.world,
                            &tick_input,
                            step_dt,
                        );
                        // Advance the local walk cycle from horizontal travel,
                        // mirroring how peers accumulate their snapshot phase.
                        let v = self.player.velocity();
                        self.local_gait += (v.x * v.x + v.z * v.z).sqrt()
                            * step_dt as f64
                            * presence::STRIDE_FREQ;
                    }
                    self.pending_toggle_fly = false;
                    self.pending_jump = false;
                }
            }
            false
        }
    }

    /// World edits: block breaking, then cadence-controlled mod hooks and queued
    /// placements. Edge-bearing render frames are replayed in order at the next
    /// permitted mod tick; hooks never run inside the voxel loop.
    fn interact_phase(
        &mut self,
        input: &FrameInput,
        detached: bool,
        captured: bool,
        dt: f32,
        eng: &mut Engine,
        mods: &mut Mods,
    ) {
        // Break is capture-gated in the query; freecam additionally can't act
        // on the world (the crosshair isn't where the player aims).
        if input.do_break && !detached && captured {
            self.break_block(mods);
        }

        if !self.mod_logic {
            return;
        }

        if detached || !captured {
            for pending in &mut self.pending_mod_input {
                pending.place = false;
                pending.place_target = None;
            }
            self.pending_mod_input.retain(|pending| pending.any());
        }
        let allow_ui = self.mod_ui_active();
        let allow_place = !detached && captured;
        let place_target = if allow_place && input.do_place {
            interact::raycast(
                &self.world,
                self.player.position,
                self.player.forward(),
                interact::REACH,
            )
            .map(|hit| hit.previous)
        } else {
            None
        };
        let current = PendingModInput::capture(input, allow_place, allow_ui, place_target);
        if current.any() {
            self.pending_mod_input.push(current);
        }
        if rate_steps(&mut self.mod_accumulator, self.mod_interval, dt) == 0 {
            return;
        }
        let mut pending = std::mem::take(&mut self.pending_mod_input);

        let mut placements = std::mem::take(&mut self.placement_scratch);
        placements.clear();
        let (screen_w, screen_h) = (eng.screen_width(), eng.screen_height());
        // Preserve ordering and multiplicity for edge-bearing render frames.
        // With no edge, one empty update keeps periodic work at `mod_hz`.
        for index in 0..pending.len().max(1) {
            let events = pending.get(index).copied().unwrap_or_default();
            placements = {
                let mut ctx = ModContext {
                    player: &mut self.player,
                    world: &mut self.world,
                    screen_w,
                    screen_h,
                    place: events.place,
                    place_target: events.place_target,
                    toggle_inventory: events.toggle_inventory,
                    toggle_crafting: events.toggle_crafting,
                    nav_up: events.nav_up,
                    nav_down: events.nav_down,
                    nav_confirm: events.nav_confirm,
                    placements,
                };
                mods.update(eng, &mut ctx);
                ctx.placements
            };
            // Apply after each event frame so repeated placements observe the
            // previous write and cannot spend twice against one empty cell.
            self.apply_placements(&mut placements);
            placements.clear();
        }
        placements.clear();
        self.placement_scratch = placements;
        pending.clear();
        self.pending_mod_input = pending;
    }

    /// Load/mesh/unload chunks around the camera (the player, unless the
    /// freecam rig has flown elsewhere), refresh the minimap (throttled), and
    /// step the simulation.
    fn stream_phase(&mut self, eng: &mut Engine, dt: f32) {
        if let Some(sim) = &mut self.sim {
            self.sim_call_accumulator =
                (self.sim_call_accumulator + dt).min(MAX_RATE_ACCUMULATED);
            if self.sim_call_accumulator >= crate::sim::TICK_SECONDS {
                let elapsed = std::mem::take(&mut self.sim_call_accumulator);
                sim.advance(&mut self.world, elapsed);
            }
        }

        let stream_due = if self.force_stream {
            self.force_stream = false;
            self.stream_accumulator = 0.0;
            true
        } else {
            rate_steps(&mut self.stream_accumulator, self.stream_interval, dt) != 0
        };
        if !stream_due {
            return;
        }

        let stream_center = match &self.camera.mode {
            CameraMode::Free { rig, .. } => rig.pos,
            CameraMode::Person(_) => self.player.position,
        };
        self.world.stream(stream_center, eng);

        // A hidden/minimal HUD does no minimap clock read or terrain raster
        // work. Refreshes share streaming's cadence instead of waking alone.
        if self.theme.hud.shows_minimap() && let Some(minimap) = &mut self.minimap {
            let p = self.player.position;
            let player_col = IVec2::new(p.x.floor() as i32, p.z.floor() as i32);
            minimap.refresh(eng, &self.world, player_col, Instant::now());
        }
    }

    /// Drain queued server messages: apply world edits, resolve our own edit
    /// verdicts (rolling back rejected predictions), surface chat, and report
    /// a lost connection. Returns `true` if the server dropped us.
    fn apply_net_events(&mut self, mods: &mut Mods) -> bool {
        let events = match &mut self.net {
            Some(net) => net.poll(),
            None => return false,
        };
        let mut disconnected = false;
        for event in events {
            match event {
                Incoming::Edit { x, y, z, spec } => {
                    // Resolve the portable spec against our own palette, then
                    // apply. The connection already dropped stale revisions,
                    // and our own edits come back as acks, not broadcasts.
                    let id = save::parse_block(&mut self.world, &spec);
                    self.world.set_block(x, y, z, id);
                }
                Incoming::EditAccepted { req } => {
                    // Prediction confirmed: the optimistic apply IS the truth.
                    self.pending_edits.remove(&req);
                }
                Incoming::EditRejected { req, restore } => {
                    let Some(pending) = self.pending_edits.remove(&req) else {
                        continue;
                    };
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
                    self.force_stream = true;
                }
                Incoming::Chat {
                    from_name,
                    channel,
                    text,
                } => {
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
                    self.console
                        .push(ui::Line::of(ui::Role::Positive, format!("* {name} joined")));
                }
                Incoming::Left { name } => {
                    self.console
                        .push(ui::Line::of(ui::Role::Muted, format!("* {name} left")));
                }
                Incoming::Time { day, day_secs } => {
                    // The server owns the shared clock: phase AND cycle length.
                    self.sky.clock.set_day(day as f64);
                    self.sky.day_length = crate::sky::DayLength::clamped(day_secs as f64);
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
        self.console.echo(&line);
        let before = settings.clone();
        let day_before = self.sky.clock.day();
        let day_len_before = self.sky.day_length;
        let pos_before = self.player.position;
        // Each output line already carries its role (System output vs Error
        // rejection), so there is nothing to guess — just show them.
        for out in command::execute(
            &line,
            &mut self.player,
            &mut self.world,
            settings,
            &mut self.sky,
        ) {
            self.console.push(out);
        }
        // A `/gfx` command edits settings; push the result through the one
        // application path and persist it, only when something actually changed.
        if *settings != before {
            self.apply_settings(eng, settings);
            settings.save();
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
            self.console
                .print("* day length is set by the server".to_string());
        }
        // A `/tp` is a position discontinuity: ordinary moves are envelope-
        // checked server-side, so report it as an explicit teleport (the
        // server may still snap us back if teleports are disabled).
        if self.player.position != pos_before {
            self.force_stream = true;
            if let Some(net) = &mut self.net {
                net.send_teleport(self.player.position);
            }
        }
    }

    /// Break the block the player is looking at, handing its elements to the mods.
    fn break_block(&mut self, mods: &mut Mods) {
        let Some(hit) = interact::raycast(
            &self.world,
            self.player.position,
            self.player.forward(),
            interact::REACH,
        ) else {
            return;
        };
        let (x, y, z) = hit.block;
        let id = self.world.block_at(x, y, z);
        // Snapshot the block's elements before it's removed.
        let elements = self.world.registry().block(id).composition.elements();
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
        placements: &mut Vec<(i32, i32, i32, crate::block::BlockId)>,
    ) {
        for (x, y, z, id) in placements.drain(..) {
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
                    PendingEdit {
                        cell: (x, y, z),
                        prev,
                        kind: PendingKind::Place(id),
                    },
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
        let mut scene = self.compose_phase(eng, fov, shake);
        let mut f = eng.begin_frame(scene.clear);
        self.scene_phase(&mut f, &scene);
        self.hud_phase(&mut f, mods, &scene);
        // Reclaim peer capacity after both consumers finish with the immutable
        // scene. Stable multiplayer frames allocate no new draw-record vector.
        self.peer_scratch = std::mem::take(&mut scene.peers);
        self.peer_scratch.clear();
    }

    /// Everything a frame needs decided BEFORE recording starts: the camera
    /// pose, the per-frame lighting truth (the engine UBO's single source),
    /// peer render poses, and the cached HUD strings.
    fn compose_phase(&mut self, eng: &mut Engine, fov: f32, shake: f32) -> Scene {
        let dt = eng.frame_time();
        // The one pose this frame renders from: mode observation plus effects.
        let pose = self.camera.pose(&self.player, &self.world, fov, shake);
        let camera_key = [
            pose.yaw.to_bits(),
            pose.pitch.to_bits(),
            pose.roll.to_bits(),
            pose.fovy.to_bits(),
        ];
        let camera = match self.camera_cache {
            Some((key, camera)) if key == camera_key => camera,
            _ => {
                let camera = pose.camera3d();
                self.camera_cache = Some((camera_key, camera));
                camera
            }
        };

        if self.theme.hud.shows_info() {
            let p = self.player.position;
            // 0.1-block display resolution: only re-format when a shown digit moves.
            let key = (
                (p.x * 10.0).round() as i64,
                (p.y * 10.0).round() as i64,
                (p.z * 10.0).round() as i64,
            );
            if (key.0, key.1, key.2)
                != (self.coord_cache.0, self.coord_cache.1, self.coord_cache.2)
            {
                let text = format!("X: {:.1}    Y: {:.1}    Z: {:.1}", p.x, p.y, p.z);
                self.coord_cache = (key.0, key.1, key.2, text);
            }

            // Scripted frames pin the readout so golden diffs never include a
            // nondeterministic live FPS region. Live FPS is sampled at human
            // display cadence rather than formatted on high-frequency jitter.
            if self.scripted {
                if self.fps_cache.as_ref().map(|(fps, _)| *fps) != Some(-1) {
                    self.fps_cache = Some((-1, "-- FPS".to_string()));
                }
            } else {
                let frame_dt = if dt.is_finite() { dt.max(0.0) } else { 0.0 };
                self.fps_refresh_accumulator =
                    (self.fps_refresh_accumulator + frame_dt).min(FPS_LABEL_INTERVAL);
                if self.fps_cache.is_none()
                    || self.fps_refresh_accumulator >= FPS_LABEL_INTERVAL
                {
                    let fps = eng.fps();
                    if self.fps_cache.as_ref().map(|(old, _)| *old) != Some(fps) {
                        self.fps_cache = Some((fps, format!("{fps:2} FPS")));
                    }
                    self.fps_refresh_accumulator = 0.0;
                }
            }
        } else {
            // Make the first Full-HUD frame refresh immediately after a mode change.
            self.fps_refresh_accumulator = FPS_LABEL_INTERVAL;
        }
        let screen = if matches!(self.theme.hud, HudMode::Off) && !self.console.is_open() {
            (0, 0)
        } else {
            (eng.screen_width(), eng.screen_height())
        };

        // `dt` steps each peer's animator (body-yaw follow, stance blend, swing).
        let want_tags = self.name_tags && self.theme.hud.shows_world_ui();
        let peers = self.peer_draws(eng, &camera, &pose, dt, self.player_models, want_tags);

        // Compose the single per-frame lighting truth: the source for the
        // engine's per-frame UBO for sky/fog and avatar key lighting. The UBO is
        // the only path; legacy push lanes have been retired.
        //
        // Exposure is the render thread's latest metered+smoothed value,
        // sourced through `Engine::exposure_for_compose`; temporal smoothing
        // already happened render-side, so frame delta is passed only for
        // signature symmetry (unused there).
        // Allow pinning exposure to a fixed default for stable bless/debug output.
        static EXPOSURE_ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            !matches!(std::env::var("WATT_EXPOSURE").as_deref(), Ok("0"))
        });
        let exposure = if self.render.exposure && *EXPOSURE_ON {
            eng.exposure_for_compose(dt)
        } else {
            voxel_engine::skeleton::Exposure::DEFAULT
        };
        // Scripted (harness) frames pin the dither phase: capture lands on an
        // arbitrary frame index under uncapped pacing, and a cycling blue-noise
        // phase is per-run LSB wobble on gradient/blend surfaces (water) that a
        // golden diff must never see.
        let dither_frame = if self.scripted { 0 } else { self.frame_index };
        let sky_day = if self.render.day_night {
            self.sky.clock.day()
        } else {
            0.5 // fixed noon: cheap, readable stripped-profile lighting
        };
        let sky_frame = match self.sky_frame_cache {
            Some((cached_day, frame)) if cached_day == sky_day => frame,
            _ => {
                let frame = if self.render.day_night {
                    self.sky.frame()
                } else {
                    self.sky.frame_at_day(sky_day)
                };
                self.sky_frame_cache = Some((sky_day, frame));
                frame
            }
        };
        let uv_key = [pose.eye.x.to_bits(), pose.eye.z.to_bits()];
        let anim_uv = match self.anim_uv_cache {
            Some((key, uv)) if key == uv_key => uv,
            _ => {
                let uv = crate::frame_snapshot::animation_uv(pose.eye);
                self.anim_uv_cache = Some((uv_key, uv));
                uv
            }
        };
        let cacheable_frame = !self.render.weather
            && !self.render.clouds
            && !self.render.water_anim
            && !self.render.exposure;
        let static_day = sky_day.to_bits();
        let (mut frame_uniforms, cached_clear) = if cacheable_frame {
            let cached = match self.static_frame_cache {
                Some(cached) if cached.day_bits == static_day => cached,
                _ => {
                    let snapshot = crate::frame_snapshot::compose_at_with_uv(
                        &self.sky,
                        sky_frame,
                        pose.eye,
                        anim_uv,
                        dither_frame,
                        exposure,
                        &self.render,
                    );
                    let cached = StaticFrameCache {
                        day_bits: static_day,
                        uniforms: voxel_engine::skeleton::FrameUniformsGpu::from(&snapshot),
                        clear: self.sky.clear_at(sky_frame),
                    };
                    self.static_frame_cache = Some(cached);
                    cached
                }
            };
            (cached.uniforms, Some(cached.clear))
        } else {
            self.static_frame_cache = None;
            let snapshot = crate::frame_snapshot::compose_at_with_uv(
                &self.sky,
                sky_frame,
                pose.eye,
                anim_uv,
                dither_frame,
                exposure,
                &self.render,
            );
            (
                voxel_engine::skeleton::FrameUniformsGpu::from(&snapshot),
                None,
            )
        };
        if cacheable_frame {
            // These are the only lanes that can differ while lighting is frozen.
            // Exposure and jitter are fixed by the cache predicate.
            frame_uniforms.exposure_dither[1] =
                crate::frame_snapshot::dither_at(dither_frame).0;
            frame_uniforms.anim[1] = anim_uv[0];
            frame_uniforms.anim[2] = anim_uv[1];
        }
        self.frame_index = self.frame_index.wrapping_add(1);

        // TerrainKey: flat terrain, sky/fog disabled, magenta clear for the
        // sky-hole detector. Normal: real clear, no debug flat.
        let (clear, debug_flat) = match self.debug_view {
            DebugView::Normal => (cached_clear.unwrap_or_else(|| self.sky.clear_at(sky_frame)), None),
            // Pure-magenta endpoints (255/0) decode identically under sRGB and raw
            // normalize, so the sky-hole detector's HDR key value is unchanged.
            DebugView::TerrainKey => (
                crate::harness::SKY_KEY.to_linear(),
                Some(crate::harness::TERRAIN_KEY),
            ),
        };

        Scene {
            pose,
            camera,
            sky_frame,
            peers,
            frame_uniforms,
            clear,
            debug_flat,
            screen,
            dt,
        }
    }

    /// The 3D scope: sky, world, and every humanoid, all camera-relative.
    fn scene_phase(&mut self, f: &mut voxel_engine::Frame, scene: &Scene) {
        let Scene {
            pose,
            camera,
            peers,
            dt,
            ..
        } = scene;
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
            if matches!(self.debug_view, DebugView::Normal) && self.render.sky {
                let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::ListSky);
                self.sky.draw_at(&mut f3, scene.sky_frame);
            }
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::ListWorld);
            self.world.render(&mut f3, pose.eye);
            // Other players: a six-box humanoid, head tracking their look and
            // body lazily following, limbs swinging with their gait. Poses are
            // already camera-relative (see peer_draws).
            for peer in peers {
                if let Some((pose, rig)) = &peer.model {
                    Pose::resolve(pose, rig).draw(&mut f3, peer.color);
                }
            }
            // The player's own body, whenever the camera can see it (third
            // person and freecam) — the same humanoid + animator the peers use.
            if self.player_models && self.camera.shows_body() {
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
                    self.player.yaw,
                    self.player.pitch,
                    Stance::of_player(&self.player),
                    Gait::new(self.local_gait as f32, speed),
                );
                let rig = self.local_anim.step(&rp, *dt);
                Pose::resolve(&rp, &rig).draw(&mut f3, peer_color(&self.save_name));
            }
        }
    }

    /// Everything over the world: minimap, reticle, name tags, info text, the
    /// mods' HUD data (rendered by the core — mods never touch the frame), and
    /// the console on top.
    fn hud_phase(&mut self, f: &mut voxel_engine::Frame, mods: &mut Mods, scene: &Scene) {
        let screen = scene.screen;
        if matches!(self.theme.hud, HudMode::Off) && !self.console.is_open() {
            return;
        }
        let _hud = voxel_engine::profile::scope(voxel_engine::profile::Meter::ListHud);
        let theme = &self.theme;

        // Minimap: informational, so Full mode only (HUD Off must blank it too).
        if theme.hud.shows_minimap() && let Some(minimap) = &self.minimap {
            let player_col = IVec2::new(
                self.player.position.x.floor() as i32,
                self.player.position.z.floor() as i32,
            );
            minimap.draw(f, screen, player_col, self.player.yaw);
        }

        // Reticle and world-space name tags: shown in every mode but fully-off.
        if theme.hud.shows_world_ui() {
            theme.crosshair.draw(f, screen);

            // Floating name tags over each visible player, in the peer's own
            // tint, fading with distance and dimming when terrain occludes the
            // head (instead of drawing full-strength through walls).
            for peer in &scene.peers {
                if let (Some((tag, alpha)), Some(name)) = (peer.tag, peer.name.as_deref()) {
                    let fs = theme.fs(18);
                    let tw = f.measure_text(name, fs);
                    let c = peer.color;
                    console::shadowed(
                        f,
                        name,
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
            ui::label(
                f,
                theme,
                screen,
                Anchor::Top,
                (0, 12),
                26,
                ui::Role::Primary.color(),
                &self.coord_cache.3,
            );
            if let Some((_, fps_text)) = &self.fps_cache {
                ui::label(
                    f,
                    theme,
                    screen,
                    Anchor::TopLeft,
                    (10, 12),
                    20,
                    ui::Role::Positive.color(),
                    fps_text,
                );
            }
            if let Some(net) = &self.net {
                let count = net.peers().count() + 1;
                let ping_ms = net.ping_ms();
                let changed = self
                    .online_cache
                    .as_ref()
                    .map(|(old_count, old_ping, _)| {
                        *old_count != count || *old_ping != ping_ms
                    })
                    .unwrap_or(true);
                if changed {
                    let text = match ping_ms {
                        Some(ms) => format!("players online: {count}   {ms} ms"),
                        None => format!("players online: {count}"),
                    };
                    self.online_cache = Some((count, ping_ms, text));
                }
                if let Some((_, _, text)) = &self.online_cache {
                    ui::label(
                        f,
                        theme,
                        screen,
                        Anchor::TopRight,
                        (-12, 180),
                        20,
                        ui::Role::Positive.color(),
                        text,
                    );
                }
            }
        }

        // Enabled mods contribute their HUD as data; the core renders it over the
        // world, under the console. Mods never touch the frame themselves.
        // Gameplay UI, so it follows the reticle: hidden only when HUD is Off.
        if self.mod_hud && theme.hud.shows_mod_hud() {
            let hud = mods.hud(&self.world, screen);
            ui::render_hud(f, theme, screen, &hud);
        }
        if matches!(theme.hud, HudMode::Full) || self.console.is_open() {
            self.console.draw(f, screen.0, screen.1);
        }
    }

    /// Build the per-frame draw data for other players. `&mut self` because animator state advances here.
    fn peer_draws(
        &mut self,
        eng: &Engine,
        camera: &Camera3D,
        pose: &ViewPose,
        dt: f32,
        want_models: bool,
        want_tags: bool,
    ) -> Vec<PeerDraw> {
        let mut draws = std::mem::take(&mut self.peer_scratch);
        draws.clear();
        if !want_models && !want_tags {
            return draws;
        }
        let Some(net) = &mut self.net else {
            return draws;
        };
        let world = &self.world;
        let eye = pose.eye;
        let forward = if want_tags {
            let forward = camera.target;
            DVec3::new(forward.x as f64, forward.y as f64, forward.z as f64)
        } else {
            DVec3::ZERO
        };
        let now = Instant::now();
        for peer in net.peers_mut().filter(|peer| peer.visible()) {
            let r = peer.sample(now);
            let feet = r.pos.feet(r.stance);
            let color = peer_color(&peer.name);
            let model = if want_models {
                let rp = RenderPose::new(
                    feet,
                    Eye(eye),
                    r.yaw,
                    r.pitch,
                    r.stance,
                    Gait::new(r.phase, r.speed),
                );
                let rig = peer.anim.step(&rp, dt);
                Some((rp, rig))
            } else {
                None
            };
            let tag = if want_tags {
                let head = feet.0 + DVec3::new(0.0, Pose::HEAD_TOP as f64 + 0.2, 0.0);
                let to_head = head - eye;
                let dist = to_head.length();
                if to_head.dot(forward) > 0.0 && dist > 1e-6 {
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
                }
            } else {
                None
            };
            // A tags-only peer that is entirely hidden needs no record at all.
            if model.is_some() || tag.is_some() {
                draws.push(PeerDraw {
                    model,
                    color,
                    name: tag.map(|_| peer.name.clone()),
                    tag,
                });
            }
        }
        draws
    }
}

/// Everything needed to draw one other player this frame.
struct PeerDraw {
    /// Camera-relative render pose (world minus eye, subtracted in f64, then
    /// narrowed) — safe to hand to the f32 immediate draws.
    /// Model data is absent in name-tags-only mode, so animator and humanoid
    /// composition are skipped rather than merely hidden at draw time.
    model: Option<(RenderPose, presence::RigParams)>,
    color: Color,
    name: Option<String>,
    /// Screen position + fade alpha for the name tag, or `None` when
    /// off-screen, behind us, or out of range.
    tag: Option<(Vec2, f32)>,
}

fn rate_interval(hz: u32) -> f32 {
    if hz == 0 { 0.0 } else { 1.0 / hz as f32 }
}

/// Bank render time and return how many fixed ticks are due. Interval zero is
/// the compatibility/per-frame mode. The reciprocal is cached when settings
/// change, and the overwhelmingly common not-due path returns before division.
fn rate_steps(accumulator: &mut f32, interval: f32, dt: f32) -> u32 {
    if interval == 0.0 {
        *accumulator = 0.0;
        return 1;
    }
    let dt = if dt.is_finite() {
        dt.clamp(0.0, MAX_RATE_ACCUMULATED)
    } else {
        0.0
    };
    *accumulator = (*accumulator + dt).min(MAX_RATE_ACCUMULATED);
    if *accumulator < interval {
        return 0;
    }
    let steps = (*accumulator / interval) as u32;
    *accumulator = (*accumulator - interval * steps as f32).max(0.0);
    steps
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

#[cfg(test)]
mod tests {
    use super::{FrameInput, MAX_RATE_ACCUMULATED, PendingModInput, rate_interval, rate_steps};

    #[test]
    fn zero_rate_preserves_every_frame_behavior() {
        let mut accumulator = 0.123;
        assert_eq!(rate_steps(&mut accumulator, rate_interval(0), 0.0), 1);
        assert_eq!(accumulator, 0.0);
    }

    #[test]
    fn fixed_rate_accumulates_and_keeps_remainder() {
        let mut accumulator = 0.0;
        let interval = rate_interval(10);
        assert_eq!(rate_steps(&mut accumulator, interval, 0.04), 0);
        assert_eq!(rate_steps(&mut accumulator, interval, 0.11), 1);
        assert!((accumulator - 0.05).abs() < 1e-6);
    }

    #[test]
    fn fixed_rate_catch_up_is_bounded() {
        let mut accumulator = 0.0;
        let interval = rate_interval(1_000);
        let steps = rate_steps(&mut accumulator, interval, 10.0);
        assert!(steps <= (MAX_RATE_ACCUMULATED * 1_000.0) as u32);
        assert!(accumulator <= 1.0 / 1_000.0 + f32::EPSILON);
    }

    #[test]
    fn mod_input_capture_respects_place_and_ui_boundaries() {
        let input = FrameInput {
            do_place: true,
            toggle_inventory: true,
            nav_down: true,
            ..FrameInput::default()
        };
        let target = Some((11, 12, 13));
        let all = PendingModInput::capture(&input, true, true, target);
        assert!(all.place && all.toggle_inventory && all.nav_down && all.any());
        assert_eq!(all.place_target, target);

        let ui_only = PendingModInput::capture(&input, false, true, target);
        assert!(!ui_only.place && ui_only.toggle_inventory && ui_only.nav_down);
        assert_eq!(ui_only.place_target, None);

        let none = PendingModInput::capture(&input, false, false, target);
        assert!(!none.any());
    }
}
