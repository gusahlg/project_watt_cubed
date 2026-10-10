//! game.rs owns the in-world state — world, player, physics — and runs a frame of it:
//! input, the mods' frame hook, movement, block interaction, mods, streaming, and drawing.
//! The window and the menu/play state machine live one level up in [`app`](crate::app);
//! a `Game` is handed the engine each frame and reports back whether to keep playing
//! or return to the menu.
use std::time::{Duration, Instant};

mod audio;
mod draw;
mod sync;

use voxel_engine::{Color, DVec3, Engine, Vec2};

use audio::{AudioPhase, PeerFrame};
use sync::PendingEdit;

use crate::audio::{AudioService, CueSymbols, GameEvent, PeerAudio, SoundSystem};
use crate::block::BlockId;
use crate::camera::{CameraMode, CameraPose, FlyAxes, GameCamera};
use crate::derived::Revision;
use crate::input::intent::{GameplayEvent, GlobalEvent, MenuEvent};
use crate::input::router::{Context, Router, View};
use crate::input::{look, movement};
use crate::interact;
use crate::minimap::{MapSample, Minimap, MinimapConfig};
use crate::modding::{
    ActionSet, Channel, FrameContext, GameContext, Message, ModContext, Mods, NoticeLevel, Notices, TextFrame,
};
use crate::net::client::Connection;
use crate::player::Player;
use crate::presence;
use crate::sched::{Ctx as SchedCtx, RateGate};
use crate::settings::Settings;
use crate::sim::Simulation;
use crate::sky::Sky;
use crate::ui::{HudElement, HudMode, Theme};
use crate::world::World;

/// What a game update wants the app to do next.
pub enum Signal {
    /// Keep playing.
    Continue,
    /// Leave to the start menu (the app saves on the way out).
    ExitToMenu,
}

/// Magenta clear under [`DebugView::TerrainKey`] — sky-hole detector background.
pub const SKY_KEY: Color = Color::rgb(255, 0, 255);
/// Flat terrain fill under [`DebugView::TerrainKey`].
pub const TERRAIN_KEY: Color = Color::rgb(0, 255, 0);

/// What the app renders for a capture.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum DebugView {
    #[default]
    Normal,
    /// ALL terrain flat [`TERRAIN_KEY`], sky/fog passes disabled, clear color
    /// [`SKY_KEY`]. The sky-hole detector's input.
    TerrainKey,
}

/// One frame's routed input, snapshotted into plain data by
/// [`Game::input_phase`] so the router borrow ends before later phases take
/// `&mut Engine`. Which fields are live depends on the frame's exclusive
/// context: `is_text` carries the capturing mod's typing, everything else gameplay.
#[derive(Debug, Default, PartialEq)]
struct FrameInput {
    is_text: bool,
    text_chars: Vec<char>,
    text_edit: Option<crate::input::intent::EditKey>,
    move_input: Option<movement::MoveInput>,
    look_delta: Vec2,
    fly_axes: FlyAxes,
    do_break: bool,
    do_place: bool,
    /// Mod actions that fired this frame.
    actions: ActionSet,
    /// Immediate mod actions that fired this frame, for the frame hook (mod logic on or off).
    immediate: ActionSet,
    /// Signed scroll steps this frame.
    wheel: i8,
    nav: Nav,
    toggle_capture: bool,
    g_escape: bool,
    g_hud: bool,
    g_shot: bool,
    g_minimap: bool,
    g_person: bool,
    g_freecam: bool,
}

/// Overlay-navigation edges of one frame, one bit per [`Nav::EVENTS`] entry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Nav(u8);

impl Nav {
    /// The menu events an overlay navigates by, in bit order.
    const EVENTS: [MenuEvent; 6] =
        [MenuEvent::Up, MenuEvent::Down, MenuEvent::Left, MenuEvent::Right, MenuEvent::NextTab, MenuEvent::Confirm];

    /// The edges `fired` reports this frame.
    fn read(fired: impl Fn(MenuEvent) -> bool) -> Self {
        Nav(Self::EVENTS.iter().enumerate().fold(0, |bits, (i, &e)| bits | (fired(e) as u8) << i))
    }

    fn has(self, e: MenuEvent) -> bool {
        Self::EVENTS.iter().position(|&x| x == e).is_some_and(|i| self.0 >> i & 1 != 0)
    }
}

/// Mutable adapters owned by the overlay phase. Grouping them makes the phase
/// boundary explicit and prevents its call site from becoming an untyped list
/// of unrelated mutable references.
struct OverlayPhase<'a> {
    input: &'a FrameInput,
    eng: &'a mut Engine,
    router: &'a mut Router,
    mods: &'a mut Mods,
    settings: &'a mut Settings,
    sound: &'a mut SoundSystem,
    events: &'a mut Vec<GameEvent>,
}

/// Edge-triggered mod intents from one render frame, retained in order when
/// mod updates run at a fixed cadence.
#[derive(Clone, Copy, Default)]
struct PendingModInput {
    place: bool,
    /// Cell selected by the edge frame's player pose. Cadence-delayed mod
    /// replay must not re-raycast from a later camera direction.
    place_target: Option<(i32, i32, i32)>,
    actions: ActionSet,
    wheel: i8,
    /// Whether a mod panel was allowed to open when this edge was captured.
    mod_ui: bool,
    nav: Nav,
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
            actions: input.actions,
            wheel: input.wheel,
            mod_ui: allow_ui,
            nav: if allow_ui { input.nav } else { Nav::default() },
        }
    }

    fn any(self) -> bool {
        self.place
            || !self.actions.is_empty()
            || self.wheel != 0
            || self.nav != Nav::default()
    }

    fn clear_ui(&mut self) {
        self.actions = ActionSet::NONE;
        self.wheel = 0;
        self.mod_ui = false;
        self.nav = Nav::default();
    }
}

/// The live world the player is in.
pub struct Game {
    world: World,
    player: Player,
    /// First/third person and freecam modes, plus shake effects.
    camera: GameCamera,
    /// Notices for the player (save, audio, session), handed to the mods once per frame.
    notices: Notices,
    /// Retained buffers the frame hook's context borrows: chat lines to send and notices raised.
    chat_out: Vec<(Channel, String)>,
    hook_notices: Notices,
    /// The save slot this world belongs to.
    save_name: String,
    /// Cached `presence::peer_color(save_name)` — local third-person body tint.
    local_color: Color,
    /// The live server connection when playing multiplayer; `None` in singleplayer.
    /// The player simulates locally and the server keeps everyone in sync.
    net: Option<Connection>,
    /// Rollback bookkeeping for optimistic edits awaiting a server verdict,
    /// keyed by the connection's request id: what the cell held before, and
    /// what the economy optimistically did (loot gained, item spent).
    pending_edits: std::collections::HashMap<u32, PendingEdit>,
    /// Tool uses awaiting the server's verdict: request id → the configuration sent.
    pending_tools: std::collections::HashMap<u32, BlockId>,
    /// Animation state for the player's own third-person body — the same
    /// machine each remote player carries.
    local_anim: presence::Animator,
    /// The local walk-cycle phase, accumulated from horizontal travel, same as a peer's.
    local_gait: f64,
    /// In-world UI look and HUD visibility (see [`Theme`]).
    theme: Theme,
    /// Day/night clock, atmosphere colour, weather, and the lighting edge into
    /// voxel shading (see [`crate::sky`]).
    sky: Sky,
    /// Top-down minimap: throttled terrain raster drawn in the HUD corner.
    /// `None` when the minimap lane is disabled — no raster state retained,
    /// no refresh clock read, no draw.
    minimap: Option<Minimap>,
    /// The minimap's view of the player when the stream phase already took it this frame; the
    /// HUD takes it instead of sampling again.
    map_sample: Option<MapSample>,
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
    /// Benchmarks drive the camera themselves; stray keystrokes must not steer or stall the run.
    input_locked: bool,
    /// Visual groups enabled by mods; fancy lanes strip when a group is off.
    visual_mask: crate::modding::VisualMask,
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
    /// The sim producer's id, kept so the `simulation` settings gate can
    /// enable/disable the lane on the scheduler instead of tearing it out.
    sim_source: voxel_engine::producer::SourceId,
    /// The autosave interval gate on the scheduler's frame clock
    /// (replaces the autosaver's `Instant` throttle). App reads
    /// [`Game::autosave_due`] and clears it via [`Game::mark_autosave`]; the
    /// save itself stays in `App` (its inputs — mods, meta, slot — live there).
    autosave_interval: crate::sched::IntervalHandle,
    /// The minimap throttle gate on the scheduler's frame clock
    /// (replaces the minimap's `last_refresh: Instant`). The recenter half of
    /// the gate stays a state comparison inside [`Minimap`].
    minimap_interval: crate::sched::IntervalHandle,

    // Independent bounded clocks for the throttled lanes (`stream_hz` etc.),
    // all sharing sched::RateGate's catch-up bound and zero-is-every-frame
    // convention. Each is reconfigured through `apply_settings`.
    stream_gate: RateGate,
    /// One-shot streaming override: teleports, freecam toggles, HUD/minimap
    /// re-activation, and settings changes must refresh promptly even when
    /// `stream_hz` throttles the steady cadence.
    force_stream: bool,
    physics_gate: RateGate,
    sky_gate: RateGate,
    mod_gate: RateGate,

    /// Edge input must survive render frames that do not execute a physics
    /// tick, so a tap between two 30 Hz ticks still jumps.
    pending_jump: bool,
    /// Edge-bearing frames awaiting the next permitted mod tick, in order —
    /// ordered replay preserves discrete actions across a throttled cadence.
    pending_mod_input: Vec<PendingModInput>,
    /// An open mod overlay must be force-closed once when its HUD lane is
    /// hidden, so an invisible modal cannot keep consuming input.
    pending_mod_overlay_close: bool,

    // Settings-derived work gates: each stops its lane at the owning boundary
    // instead of merely hiding output.
    mod_logic: bool,
    player_models: bool,
    name_tags: bool,

    /// Bumped whenever render config or palette changes — the single stamp
    /// every render-dependent frame cache folds into its key, so a stale packet
    /// is a key mismatch the compiler enforces rather than a forgotten
    /// `invalidate` at each mutation site.
    content_rev: Revision,
    /// Rendering caches and scratch buffers, owned by the presentation module.
    drawing: draw::DrawState,
    // Retained-capacity scratch (cleared, never shrunk): stable frames do no
    // allocator work for these.
    placement_scratch: Vec<(i32, i32, i32, crate::block::registry::BlockId)>,
    /// This frame's peers, in [`Connection::peers`] order.
    peer_frames: Vec<PeerFrame>,
    peer_pose_scratch: Vec<PeerAudio>,
    /// Up-axes parallel to [`peer_pose_scratch`]. Kept off the public peer record.
    peer_up_scratch: Vec<crate::coord::Face>,
    /// Last frame's named-phase durations, for the stall detector.
    phases: FramePhases,
    /// The frame's audio facts; empty between frames.
    events_scratch: Vec<GameEvent>,
    hud_scratch: Vec<HudElement>,
    /// Menu notice taken when a network session leaves.
    leave_notice: Option<String>,
    /// The game changed the settings (HUD hotkey, a mod's frame hook); the app writes them off
    /// the frame.
    settings_dirty: bool,
    /// A mod's frame hook changed the settings: the app applies them after it has pushed the
    /// engine's half, so this game sees the fresh render extent.
    settings_changed: bool,
}

/// Durations of `Game::update` phases, sampled every watched frame for stall logs.
#[derive(Clone, Copy, Default)]
struct FramePhases {
    net: Duration,
    input: Duration,
    overlay: Duration,
    motion: Duration,
    interact: Duration,
    stream: Duration,
    audio: Duration,
}

/// A phase stopwatch for [`FramePhases`]; reads no clock unless watching.
struct Lap(Option<Instant>);

impl Lap {
    fn start(watch: bool) -> Self {
        Lap(watch.then(Instant::now))
    }

    fn stop(self, slot: &mut Duration) {
        if let Some(start) = self.0 {
            *slot = start.elapsed();
        }
    }
}

impl Game {
    pub fn new(world: World, player: Player, save_name: String) -> Self {
        let mut sched = crate::sched::Scheduler::new();
        // The fixed-tick sim runs through the scheduler. A pure clock lane is
        // never "starved", so its forward-progress floor is effectively infinite
        // — it fires only when whole ticks are due.
        let sim_id = sched.register(
            Simulation::manifest(),
            Box::new(Simulation::with_systems(vec![Box::new(
                crate::sim::reactions::Reactions,
            )])),
            u32::MAX,
        );
        sched.set_meter(sim_id, voxel_engine::profile::Meter::Physics);
        // The autosave and minimap throttles are scheduler interval gates.
        let autosave_interval =
            sched.register_interval(crate::save::autosave::AUTOSAVE_INTERVAL.as_secs_f32());
        let minimap_interval =
            sched.register_interval(MinimapConfig::DEFAULT.refresh_every.as_secs_f32());
        Self {
            world,
            player,
            camera: GameCamera::new(),
            notices: Notices::default(),
            chat_out: Vec::new(),
            hook_notices: Notices::default(),
            local_color: crate::presence::peer_color(&save_name),
            save_name,
            net: None,
            pending_edits: std::collections::HashMap::new(),
            pending_tools: std::collections::HashMap::new(),
            local_anim: presence::Animator::default(),
            local_gait: 0.0,
            theme: Theme::new(),
            sky: Sky::new(),
            // Constructed present so scripted/harness games keep the full
            // historical presentation; `apply_settings` drops it when the
            // minimap lane is disabled.
            minimap: Some(Minimap::new(MinimapConfig::DEFAULT)),
            map_sample: None,
            debug_view: DebugView::Normal,
            scripted: false,
            input_locked: false,
            visual_mask: crate::modding::VisualMask::default(),
            render: crate::render_config::RenderConfig::default(),
            sched,
            sim_source: sim_id,
            autosave_interval,
            minimap_interval,
            stream_gate: RateGate::from_hz(0),
            force_stream: true,
            physics_gate: RateGate::from_hz(0),
            sky_gate: RateGate::from_hz(0),
            mod_gate: RateGate::from_hz(0),
            pending_jump: false,
            pending_mod_input: Vec::new(),
            pending_mod_overlay_close: false,
            mod_logic: true,
            player_models: true,
            name_tags: true,
            content_rev: Revision::default(),
            drawing: draw::DrawState::new(),
            placement_scratch: Vec::new(),
            peer_frames: Vec::new(),
            peer_pose_scratch: Vec::new(),
            peer_up_scratch: Vec::new(),
            phases: FramePhases::default(),
            events_scratch: Vec::new(),
            hud_scratch: Vec::new(),
            leave_notice: None,
            settings_dirty: false,
            settings_changed: false,
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
        self.bump_content_rev();
    }

    /// Retire every render-dependent frame cache in one stamp bump. The single
    /// place render config or palette changes announce themselves.
    fn bump_content_rev(&mut self) {
        self.content_rev.0 = self.content_rev.0.wrapping_add(1);
    }

    /// Push every live-applicable setting into this game: the world's view
    /// volume and meshing lanes, the per-frame look config, HUD mode, lane
    /// gates, and the throttled clocks. THE one path — world entry and in-game
    /// `/gfx` edits both come through here, so they can never drift apart.
    /// The engine's half (window, MSAA, scale, lane flags) is `App::push_gfx`'s,
    /// which runs first so the lanes resolve against the fresh render extent.
    /// (World-construction lanes — occlusion/lod2 — stay entry-only by design;
    /// see `App::enter_game`.)
    pub fn apply_settings(&mut self, eng: &mut Engine, settings: &Settings) {
        let render = self.visual_mask.effective_render(settings);
        // View volume BEFORE the render config: the far ladder's `unit`
        // tracks the full-res radius, so the transition detector must see the
        // new volume.
        self.world
            .set_view_distances(settings.render_distance, settings.vertical_distance);
        self.world.set_render_config(render, eng);
        self.world.set_lighting(settings.lighting, eng);
        self.world.set_ao(settings.ao, eng);
        self.adopt_gameplay_settings(settings, render);
    }

    /// Gameplay/HUD/clock half of [`apply_settings`] — no engine. Headless
    /// tests and the live path share this so they cannot drift.
    fn adopt_gameplay_settings(
        &mut self,
        settings: &Settings,
        render: crate::render_config::RenderConfig,
    ) {
        let mod_ui_was_active = self.mod_ui_active();
        self.render = render;
        self.bump_content_rev();

        self.theme.scale = settings.ui_scale;
        self.theme.hud = settings.hud_mode;

        self.stream_gate.set_hz(settings.stream_hz);
        self.physics_gate.set_hz(settings.physics_hz);
        self.sky_gate.set_hz(settings.sky_hz);
        self.mod_gate.set_hz(settings.mod_hz);
        self.force_stream = true;

        // The sim lane stays registered on the scheduler; the gate merely
        // stops it being ticked (no hidden periodic work while disabled).
        self.sched.set_enabled(self.sim_source, settings.simulation);
        if settings.minimap {
            if self.minimap.is_none() {
                self.minimap = Some(Minimap::new(MinimapConfig::DEFAULT));
            }
        } else {
            self.minimap = None;
        }

        let mod_ui_will_be_active = mod_ui_active(settings.mod_logic, self.theme.hud);
        if mod_ui_will_be_active {
            self.pending_mod_overlay_close = false;
        } else if mod_ui_was_active {
            self.on_mod_ui_hidden();
        }
        self.mod_logic = settings.mod_logic;
        if !self.mod_logic {
            self.mod_gate.reset();
            self.pending_mod_input.clear();
        }
        self.player_models = settings.player_models;
        self.name_tags = settings.name_tags;
    }

    /// Whether a modal supplied by the mod layer can both be seen and receive
    /// input. Keeping one predicate for routing and Escape prevents invisible
    /// overlays when either the mod lane or the master HUD is disabled.
    fn mod_ui_active(&self) -> bool {
        mod_ui_active(self.mod_logic, self.theme.hud)
    }

    /// The mod UI just became invisible: force-close any open overlay once and
    /// drop queued UI edges (an invisible modal must not consume or replay
    /// input). The one implementation both the settings path and the HUD
    /// hotkey use.
    fn on_mod_ui_hidden(&mut self) {
        self.pending_mod_overlay_close = true;
        for pending in &mut self.pending_mod_input {
            pending.clear_ui();
        }
        self.pending_mod_input.retain(|pending| pending.any());
    }

    /// A scripted game over the real InfiniteDiffusion terrain (benchmarks, goldens), the
    /// player standing on the surface at the origin.
    pub fn scripted(seed: u64, render: crate::render_config::RenderConfig) -> Game {
        let world = World::with_kind(seed as i64, render, crate::world::generation::WorldgenKind::Diffusion, true);
        let pos = world.chart_spawn().unwrap_or_else(|| {
            let ground = world.surface_y(0, 0);
            DVec3::new(0.5, ground as f64 + 3.0, 0.5)
        });
        let player = Player::standing(pos, world.gravity_at(pos).accel);
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

    /// Attach a server connection, turning this into a multiplayer session.
    pub fn with_net(mut self, net: Connection) -> Self {
        self.net = Some(net);
        self.world.set_reactions_authority(false);
        self
    }

    pub fn save_name(&self) -> &str {
        &self.save_name
    }

    /// Report a status line to the player (save and load notices). It reaches the mods' message
    /// hook on the next frame; with no mod showing it, stderr.
    pub fn notify(&mut self, line: impl Into<String>) {
        self.notices.push(NoticeLevel::Info, line);
    }

    /// Whether a mod holds the keyboard (see [`Mods::text_captured`]).
    pub fn text_captured(&self, mods: &Mods) -> bool {
        mods.text_captured()
    }

    /// Whether this is a networked session (its world is a server mirror, not a
    /// local save, so the app does not autosave it).
    pub fn is_multiplayer(&self) -> bool {
        self.net.is_some()
    }

    /// Whether the game changed the settings since the last take, so the app should save them.
    pub fn take_settings_dirty(&mut self) -> bool {
        std::mem::take(&mut self.settings_dirty)
    }

    /// Whether a mod's frame hook changed the settings since the last take, so the app should
    /// [`apply_settings`](Self::apply_settings) once it has pushed the engine's half.
    pub fn take_settings_changed(&mut self) -> bool {
        std::mem::take(&mut self.settings_changed)
    }

    /// Why a network session left, for the menu. Cleared by the take.
    pub fn take_leave_notice(&mut self) -> Option<String> {
        self.leave_notice.take()
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

    pub fn set_input_locked(&mut self, locked: bool) {
        self.input_locked = locked;
    }

    pub fn set_visual_mask(&mut self, mask: crate::modding::VisualMask) {
        self.visual_mask = mask;
    }

    /// Capture the cursor when (re)entering play.
    pub fn on_enter(&mut self, eng: &mut Engine, router: &mut Router) {
        router.set_captured(true);
        eng.disable_cursor();
    }

    /// Return the world's GPU meshes and drop its far map. Called before the game
    /// is dropped when leaving to the menu, so the next world cannot show this one.
    pub fn free_gpu(&mut self, eng: &mut Engine) {
        self.sky.release_far_map(eng);
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
        audio: &mut AudioService,
        cues: &CueSymbols,
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
            self.world.stream(self.player.position, Some(&mut *eng), mods.appearance());
            let clocks = self.sched.clocks(dt);
            let mut sched_ctx = SchedCtx::new(&mut self.world);
            self.sched.tick(&mut sched_ctx, &clocks);
            return Signal::Continue;
        }

        self.tick_sky(dt);

        // This frame's audio facts, accumulated across the phases. Footsteps are
        // added at the audio commit; mods choose the cues and the voice sessions.
        let mut events = std::mem::take(&mut self.events_scratch);

        // Phase timings feed the stall log, which only debug builds and
        // benchmarks (the input-locked mode) print.
        let watch = cfg!(debug_assertions) || self.input_locked;
        if watch {
            self.phases = FramePhases::default();
        }
        let t = Lap::start(watch);
        if let Some(signal) = self.net_phase(mods, &mut events) {
            return signal;
        }
        t.stop(&mut self.phases.net);
        let t = Lap::start(watch);
        let input = self.input_phase(eng, router, mods, dt);
        t.stop(&mut self.phases.input);
        // The overlay may consume the frame (a mod typing, a capture starting): movement
        // and interaction run only on an unconsumed frame. Streaming still runs
        // while a spawn/teleport slab is outstanding so loading progresses while a
        // mod types. A still singleplayer frame skips the mixer when
        // nothing is sounding; only a real exit short-circuits the rest of the frame.
        let t = Lap::start(watch);
        let overlay = self.overlay_phase(OverlayPhase {
            input: &input,
            eng,
            router,
            mods,
            settings,
            sound,
            events: &mut events,
        });
        t.stop(&mut self.phases.overlay);
        let consumed = match overlay {
            Some(Signal::ExitToMenu) => return Signal::ExitToMenu,
            Some(Signal::Continue) => true,
            None => false,
        };
        let ready = self.world.spawn_ready();
        if !consumed && ready {
            let t = Lap::start(watch);
            let detached = self.motion_phase(&input, dt);
            t.stop(&mut self.phases.motion);
            let t = Lap::start(watch);
            self.interact_phase(&input, detached, dt, eng, mods, &mut events, router.action_ids());
            t.stop(&mut self.phases.interact);
        }
        if !consumed || !ready {
            let t = Lap::start(watch);
            self.stream_phase(eng, dt, mods);
            t.stop(&mut self.phases.stream);
        }
        let active = !consumed;
        let t = Lap::start(watch);
        self.sample_peers();
        let ids = router.action_ids();
        self.commit_audio(AudioPhase {
            dt,
            input: &input,
            sound,
            audio,
            cues,
            settings,
            events,
            active,
            mods,
            ids,
        });
        t.stop(&mut self.phases.audio);
        Signal::Continue
    }

    /// Last frame's named-phase timings, for the stall detector.
    pub(crate) fn phase_debug(&self) -> String {
        let p = &self.phases;
        format!(
            "net={:.1}ms input={:.1}ms overlay={:.1}ms motion={:.1}ms interact={:.1}ms stream={:.1}ms audio={:.1}ms",
            p.net.as_secs_f64() * 1000.0,
            p.input.as_secs_f64() * 1000.0,
            p.overlay.as_secs_f64() * 1000.0,
            p.motion.as_secs_f64() * 1000.0,
            p.interact.as_secs_f64() * 1000.0,
            p.stream.as_secs_f64() * 1000.0,
            p.audio.as_secs_f64() * 1000.0,
        )
    }

    /// The once-per-frame router transition: pick the exclusive context (Text
    /// while a mod holds the keyboard) and snapshot every intent into plain
    /// data, so the router borrow ends before any `&mut Engine` side effects
    /// (screenshot, cursor grab, a capture's char drain) run in later phases.
    fn input_phase(&mut self, eng: &mut Engine, router: &mut Router, mods: &Mods, dt: f32) -> FrameInput {
        router.sync_actions(mods);
        router.set_context(if mods.text_captured() {
            Context::Text
        } else {
            Context::Gameplay
        });

        if self.input_locked {
            router.drain_frame();
            return FrameInput::default();
        }

        let mut f = FrameInput::default();
        let mod_ui = self.mod_ui_active();
        let input = router.frame_filtered(eng, dt, self.mod_logic, self.minimap.is_some());
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
                f.actions = gp.actions();
                // Immediate actions reach the frame hook whatever the mod cadence.
                f.immediate = gp.immediate_actions();
                f.wheel = gp.wheel();
                if self.mod_logic {
                    f.do_place = gp.event(GameplayEvent::Place);
                }
                if mod_ui {
                    f.nav = Nav::read(|e| gp.overlay_nav(e));
                }
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
        f.g_person = global.event(GlobalEvent::CyclePerson);
        f.g_freecam = global.event(GlobalEvent::ToggleFreecam);
        f
    }

    /// The mods' frame hook, escape routing, and the global toggles (mouse capture, HUD
    /// cycle, screenshot, minimap, camera modes). `Some` consumes the frame: while a mod
    /// holds the keyboard, and on the frame its capture starts, nothing below runs.
    fn overlay_phase(&mut self, phase: OverlayPhase<'_>) -> Option<Signal> {
        let OverlayPhase {
            input,
            eng,
            router,
            mods,
            settings,
            sound,
            events,
        } = phase;
        if std::mem::take(&mut self.pending_mod_overlay_close) {
            mods.close_overlay();
        }

        let screen = (eng.screen_width(), eng.screen_height());
        self.frame_hook(input, screen, router.action_ids(), mods, settings, sound, events);

        if self.input_locked {
            return None;
        }

        // Text context: the capturing mod owns all input; nothing else runs. Esc went to that
        // mod and ended its capture (the frame hook), so it never means "leave the world" here.
        if input.is_text {
            self.drop_pending_edges();
            return Some(Signal::Continue);
        }

        // A capture started this frame: drain the char queue so the key that opened it isn't
        // also typed.
        if mods.text_captured() {
            self.drop_pending_edges();
            while eng.get_char_pressed().is_some() {}
            return Some(Signal::Continue);
        }

        // Esc closes an in-world mod overlay before leaving the world.
        if input.g_escape {
            self.drop_pending_edges();
            if self.mod_ui_active() && mods.close_overlay() {
                return Some(Signal::Continue);
            }
            return Some(Signal::ExitToMenu);
        }

        if input.toggle_capture {
            // A placement edge captured before the cursor is released must not
            // fire later after a throttled mod tick.
            self.drop_pending_places();
            toggle_mouse(eng, router);
        }
        if input.g_hud {
            let mod_ui_was_active = self.mod_ui_active();
            self.theme.cycle_hud();
            // The HUD hotkey and the settings row edit the same state; sync it
            // back (marking Custom) so the menu, `/gfx`, and persistence agree.
            settings.hud_mode = self.theme.hud;
            settings.mark_custom();
            self.settings_dirty = true;
            if mod_ui_was_active && !self.mod_ui_active() {
                self.on_mod_ui_hidden();
            }
            // Minimap/HUD re-activation must repaint promptly even at 15 Hz.
            self.force_stream = true;
        }
        if input.g_shot {
            match eng.screenshot() {
                Some(path) => println!("screenshot queued: {}", path.display()),
                None => eprintln!("screenshot could not be queued"),
            }
        }
        if input.g_minimap
            && let Some(minimap) = &mut self.minimap
        {
            minimap.toggle_rotation();
        }

        if input.g_person {
            self.camera.cycle_person();
        }
        if input.g_freecam {
            // Reattaching after the rig flew far away resumes physics at the
            // frozen player, whose chunks may have streamed out (the centre
            // followed the camera). Request the collision slab and freeze
            // physics until it lands so the first reattached step never runs
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

    /// Hand the queued notices to the mods' message hook; a notice no mod showed goes to stderr.
    /// An empty queue costs one length check.
    fn deliver_notices(&mut self, mods: &mut Mods) {
        if self.notices.is_empty() {
            return;
        }
        for notice in self.notices.drain() {
            if !mods.on_message(&Message::Notice(&notice)) {
                eprintln!("{}", notice.text);
            }
        }
    }

    /// The mods' frame hook ([`Mods::on_frame`]), then the core's follow-up on what it changed:
    /// changed settings are marked for the app to apply and save and re-mix the audio; a changed
    /// clock is shared with the server, whose day length wins; a moved player streams its
    /// surroundings at once and is reported as a teleport; queued chat goes to the server (and
    /// nowhere in single player). Built from retained buffers, so a frame with no input
    /// allocates nothing.
    #[allow(clippy::too_many_arguments)] // the hook's whole-game context, assembled in one place
    fn frame_hook(
        &mut self,
        input: &FrameInput,
        screen: (i32, i32),
        ids: &[&'static str],
        mods: &mut Mods,
        settings: &mut Settings,
        sound: &mut SoundSystem,
        events: &mut Vec<GameEvent>,
    ) {
        self.deliver_notices(mods);
        let day_before = self.sky.clock.day();
        let day_len_before = self.sky.day_length;
        let pos_before = self.player.position;
        let networked = self.net.is_some();
        let text = input.is_text.then_some(TextFrame {
            chars: &input.text_chars,
            edit: input.text_edit,
            escape: input.g_escape,
        });
        let mut game = GameContext::new(&mut self.player, &mut self.world, settings, &mut self.sky).with_queues(
            std::mem::take(events),
            std::mem::take(&mut self.chat_out),
            std::mem::take(&mut self.hook_notices),
        );
        game.networked = networked;
        game.detached = matches!(self.camera.mode, CameraMode::Free { .. });
        game.visuals = self.visual_mask;
        let mut ctx = FrameContext::frame(game, screen, input.immediate, ids, text);
        mods.on_frame(&mut ctx);
        let (queued, mut chat_out, mut raised, settings_changed) = ctx.game.into_queues();
        *events = queued;
        self.notices.append(&mut raised);
        self.hook_notices = raised;

        if settings_changed {
            self.settings_changed = true;
            self.settings_dirty = true;
            sound.set_mix(settings.mix_change());
        }
        // A `/time` change is shared: tell the server so every client's clock follows (the
        // server relays it and hands it to future joiners).
        if self.sky.clock.day() != day_before
            && let Some(net) = &mut self.net
        {
            net.send_set_time(self.sky.clock.day() as f32);
        }
        // The cycle LENGTH is server-owned in multiplayer: a local change would silently
        // desync every clock's advance rate.
        if self.sky.day_length != day_len_before && networked {
            self.sky.day_length = day_len_before;
            self.notices.push(NoticeLevel::Warning, "* day length is set by the server");
        }
        // A moved player is a position discontinuity: ordinary moves are envelope-checked
        // server-side, so report it as an explicit teleport (the server may still snap us back
        // if teleports are disabled) — and stream out of band so the destination doesn't wait
        // on `stream_hz`.
        if self.player.position != pos_before {
            self.force_stream = true;
            if let Some(net) = &mut self.net {
                net.send_teleport(self.player.position);
            }
        }
        // The server echoes chat back, so a sent line is shown when it returns.
        if let Some(net) = &mut self.net {
            for (channel, text) in chat_out.drain(..) {
                net.send_chat(channel.wire(), &text);
            }
        }
        chat_out.clear();
        self.chat_out = chat_out;
    }

    /// Drop every latched input edge — called when a modal (a mod's text capture,
    /// menu exit) takes over the frame, so stale edges can't fire after it closes.
    fn drop_pending_edges(&mut self) {
        self.pending_mod_input.clear();
        self.mod_gate.reset();
        self.pending_jump = false;
    }

    /// Drop only latched placement edges (aim is no longer meaningful), while
    /// UI navigation edges stay queued.
    fn drop_pending_places(&mut self) {
        for pending in &mut self.pending_mod_input {
            pending.place = false;
            pending.place_target = None;
        }
        self.pending_mod_input.retain(|pending| pending.any());
    }

    /// Camera effects plus movement. Exactly one thing consumes look/move per
    /// frame — the detached freecam rig (player frozen) or the player; returns
    /// whether the rig had it (see `CameraMode`). Mouse look and camera
    /// effects stay render-rate responsive; only the collision/velocity
    /// integration runs at `physics_hz`, with edge input latched until a
    /// physics tick consumes it.
    fn motion_phase(&mut self, input: &FrameInput, dt: f32) -> bool {
        self.camera.fx.update(dt);
        if let Some(rig) = self.camera.free_rig() {
            self.pending_jump = false;
            rig.look(input.look_delta);
            rig.fly(input.fly_axes, dt);
            true
        } else {
            // Look (inert while uncaptured — the query already zeroed the delta).
            look::apply(&mut self.player, input.look_delta);
            // The body frame follows the local up at render rate, from the gravity the last
            // physics step applied; it holds in weak gravity (zero-g keeps its orientation).
            align_body(&mut self.player, dt);

            if let Some(mi) = &input.move_input {
                self.pending_jump |= mi.jump();
                let steps = self.physics_gate.steps(dt);
                if steps != 0 {
                    let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::Physics);
                    let step_dt = self.physics_gate.step_dt(dt);
                    for step in 0..steps {
                        let mut tick_input = *mi;
                        tick_input.set_jump(mi.jump() || (step == 0 && self.pending_jump));
                        // A cruise feels no gravity and moves physically (its steps outrun any patch).
                        let cruising = self.player.cruising();
                        let gravity = if cruising { DVec3::ZERO } else { self.gravity_at(self.player.position) };
                        // On a round world the step runs in the storage frame of the patch underfoot.
                        let patch = if cruising { None } else { self.world.atlas_at(self.player.position) };
                        let trauma = match patch {
                            Some(atlas) => movement::update_player_in(
                                &mut self.player,
                                &self.world,
                                atlas,
                                &tick_input,
                                step_dt,
                                gravity,
                            ),
                            None => movement::update_player(
                                &mut self.player,
                                &self.world,
                                &tick_input,
                                step_dt,
                                gravity,
                            ),
                        };
                        if trauma > 0.0 {
                            self.camera.fx.add_trauma(trauma);
                        }
                        // Advance the local walk cycle from horizontal travel so the
                        // third-person body animates. The audio gait (footstep
                        // phase-crossings) is a separate integrator in the audio service;
                        // this one drives rendering only.
                        let mut v = self.player.velocity();
                        v[self.player.up_axis.axis()] = 0.0;
                        self.local_gait += v.length() * step_dt as f64 * presence::STRIDE_FREQ;
                    }
                    self.pending_jump = false;
                }
            }
            false
        }
    }

    /// The gravity at `p`, from the world's matter.
    fn gravity_at(&self, p: DVec3) -> DVec3 {
        self.world.gravity_at(p).accel
    }

    /// World edits: block breaking, then cadence-controlled mod hooks and
    /// queued placements. Edge-bearing render frames are replayed in order at
    /// the next permitted mod tick; hooks never run inside the voxel loop.

    fn interact_phase(
        &mut self,
        input: &FrameInput,
        detached: bool,
        dt: f32,
        eng: &mut Engine,
        mods: &mut Mods,
        events: &mut Vec<GameEvent>,
        action_ids: &[&'static str],
    ) {
        // Break is capture-gated in the query; freecam additionally can't act
        // on the world (the crosshair isn't where the player aims).
        if input.do_break && !detached {
            self.primary_action(mods, events);
        }

        // Disabled mod logic performs no probe, no queueing, no dispatch.
        if !self.mod_logic {
            return;
        }

        // A detached camera cannot perform player-origin actions: its
        // crosshair no longer represents the frozen player's aim — and any
        // earlier latched placement aim is stale the moment it detaches.
        if detached {
            self.drop_pending_places();
        }
        let allow_ui = self.mod_ui_active();
        let allow_place = !detached;
        // Resolve the placement cell AT THE EDGE, from this exact frame's
        // pose: cadence-delayed replay must not retarget from a later aim.
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
        if self.mod_gate.steps(dt) == 0 {
            return;
        }
        self.mod_tick((eng.screen_width(), eng.screen_height()), mods, events, action_ids);
    }

    /// One mod tick: replay the latched edge frames in order (one empty update when there are
    /// none), applying each frame's queued placements before the next.
    fn mod_tick(
        &mut self,
        (screen_w, screen_h): (i32, i32),
        mods: &mut Mods,
        events: &mut Vec<GameEvent>,
        action_ids: &[&'static str],
    ) {
        let mut pending = std::mem::take(&mut self.pending_mod_input);
        let mut placements = std::mem::take(&mut self.placement_scratch);
        placements.clear();
        // Preserve ordering and multiplicity for edge-bearing render frames.
        // With no edge, one empty update keeps periodic work at `mod_hz`.
        let idle = PendingModInput { mod_ui: self.mod_ui_active(), ..PendingModInput::default() };
        for index in 0..pending.len().max(1) {
            let edges = pending.get(index).copied().unwrap_or(idle);
            let networked = self.net.is_some();
            placements = {
                let mut ctx = ModContext {
                    player: &mut self.player,
                    world: &mut self.world,
                    screen_w,
                    screen_h,
                    place: edges.place,
                    place_target: edges.place_target,
                    nav_up: edges.nav.has(MenuEvent::Up),
                    nav_down: edges.nav.has(MenuEvent::Down),
                    nav_left: edges.nav.has(MenuEvent::Left),
                    nav_right: edges.nav.has(MenuEvent::Right),
                    nav_tab: edges.nav.has(MenuEvent::NextTab),
                    nav_confirm: edges.nav.has(MenuEvent::Confirm),
                    wheel: edges.wheel,
                    mod_ui: edges.mod_ui,
                    networked,
                    placements,
                    fired: edges.actions,
                    ids: action_ids,
                    extra: Vec::new(),
                };
                mods.update(&mut ctx);
                ctx.placements
            };
            // Apply after each event frame so repeated placements observe the
            // previous write and cannot spend twice against one empty cell.
            self.apply_placements(&mut placements, events, mods);
        }
        placements.clear();
        self.placement_scratch = placements;
        pending.clear();
        self.pending_mod_input = pending;
    }

    /// Load/mesh/unload chunks around the camera (the player, unless the
    /// freecam rig has flown elsewhere), refresh the minimap (throttled), and
    /// step the simulation.
    fn stream_phase(&mut self, eng: &mut Engine, dt: f32, mods: &Mods) {
        // Name configurations interned since the last frame (presentation only; O(new)).
        self.world.registry_mut().refresh_names(mods.namer());
        // The scheduler drives the fixed-tick sim lane. Its clock
        // (fixed-tick accumulator + catch-up cap) is derived once per frame
        // here; other lanes still run directly below until they migrate in.
        let clocks = self.sched.clocks(dt);
        let mut sched_ctx = SchedCtx::new(&mut self.world);
        self.sched.tick(&mut sched_ctx, &clocks);

        let due = self.take_stream_due(dt);
        if !self.stream_or_pump(Some(&mut *eng), due, mods) {
            return;
        }

        // A hidden/minimal HUD does no minimap clock read or terrain raster
        // work; refreshes share streaming's cadence instead of waking alone.
        if self.theme.hud.shows_minimap()
            && let Some(minimap) = &mut self.minimap
        {
            // The map follows the body, not the freecam. The throttle rides the
            // scheduler's interval gate (advanced in clocks() above); the recenter
            // half stays inside Minimap::due.
            let sample = MapSample::of(&self.world, &self.player);
            self.map_sample = Some(sample);
            let due = self.sched.interval_due(self.minimap_interval);
            if minimap.refresh(eng, &self.world, sample, due) {
                self.sched.interval_reset(self.minimap_interval);
            }
        }
    }

    /// Advance the day/night clock (singleplayer drives it locally; a server
    /// sync overrides `day` on arrival), throttled by `sky_hz`. With the
    /// lane off, compose samples fixed noon without mutating authoritative
    /// clock state; re-enabling resumes the stored time instead of freezing
    /// a stripped profile at night.
    fn tick_sky(&mut self, dt: f32) {
        if self.render.day_night {
            let steps = self.sky_gate.steps(dt);
            if steps != 0 {
                self.sky
                    .tick(steps as f64 * self.sky_gate.step_dt(dt) as f64);
            }
        } else {
            self.sky_gate.reset();
        }
    }

    /// Whether the topology pass (selection/admission/unload) runs this frame. It
    /// rides `stream_hz`; forced refreshes (teleports, freecam/HUD changes,
    /// settings) run out of band so correctness never waits on a 15 Hz clock.
    fn take_stream_due(&mut self, dt: f32) -> bool {
        if std::mem::take(&mut self.force_stream) {
            self.stream_gate.reset();
            true
        } else {
            self.stream_gate.steps(dt) != 0
        }
    }

    /// Where the world streams around: the camera, which is the player unless
    /// the freecam rig has flown elsewhere.
    fn stream_center(&self) -> DVec3 {
        match &self.camera.mode {
            CameraMode::Free { rig, .. } => rig.pos,
            CameraMode::Person(_) => self.player.position,
        }
    }

    /// Stream around the camera when a pass is `due` (`stream` owns the result
    /// pump on those frames), else only pump finished work. A cruise holds the
    /// world still: in-flight work lands, nothing new streams around the
    /// player; ending it streams the destination like a teleport. Returns
    /// whether it streamed.
    fn stream_or_pump(&mut self, eng: Option<&mut Engine>, due: bool, mods: &Mods) -> bool {
        if !due || self.player.cruising() {
            self.world.pump(eng, mods.appearance());
            return false;
        }
        let center = self.stream_center();
        self.world.stream(center, eng, mods.appearance());
        true
    }

    /// Headless quiet frame: input drain, the mods' frame hook, motion (inert), scheduler,
    /// stream/pump, silent audio, HUD/lighting caches — the pieces `update` + `draw` run, in
    /// order, through the same helpers, without an Engine.
    #[cfg(test)]
    fn tick_quiet(
        &mut self,
        dt: f32,
        router: &mut Router,
        sound: &mut SoundSystem,
        audio: &mut AudioService,
        cues: &CueSymbols,
        settings: &mut Settings,
        mods: &mut Mods,
    ) {
        self.tick_sky(dt);
        let mut events = std::mem::take(&mut self.events_scratch);
        if self.input_locked {
            router.drain_frame();
        }
        let input = FrameInput::default();
        self.frame_hook(&input, (1280, 720), router.action_ids(), mods, settings, sound, &mut events);
        if self.world.spawn_ready() {
            let _ = self.motion_phase(&input, dt);
        }
        let clocks = self.sched.clocks(dt);
        let mut sched_ctx = SchedCtx::new(&mut self.world);
        self.sched.tick(&mut sched_ctx, &clocks);
        let due = self.take_stream_due(dt);
        self.stream_or_pump(None, due, mods);
        self.sample_peers();
        let ids = router.action_ids();
        self.commit_audio(AudioPhase {
            dt,
            input: &input,
            sound,
            audio,
            cues,
            settings,
            events,
            active: true,
            mods,
            ids,
        });
        self.compose_quiet(mods);
    }

    /// Where this world's atmosphere ends: a body of the cosmos has air up to `AIR_TOP`, a flat
    /// world's space realm starts a few hundred blocks up.
    pub fn space_fade(&self) -> crate::frame_snapshot::SpaceFade {
        match self.world.terrain().cosmos() {
            Some(_) => crate::frame_snapshot::SpaceFade::BODY,
            None => crate::frame_snapshot::SpaceFade::FLAT,
        }
    }

    /// Altitude above the datum of the nearest body (the sky fades to space with it); far from every
    /// body, or over an airless one (a moon), effectively infinite. Inside a Hollow's cavity the
    /// shell is the sky, so this is the same sentinel and the blue atmosphere stays outside. A flat
    /// world's datum is `y = 0`.
    pub fn sky_altitude(&self, eye: DVec3) -> f64 {
        use crate::world::terrain::cosmos::Kind;
        match self.world.terrain().cosmos() {
            Some(cosmos) => {
                if cosmos.hollow_cavity(eye).is_some() {
                    return 1.0e9;
                }
                match cosmos.body_at(eye) {
                    Some(b) if b.kind != Kind::Moon => cosmos.altitude(b, eye),
                    _ => 1.0e9,
                }
            }
            None => eye.y,
        }
    }
}

/// Whether a mod-supplied modal can both be seen and receive input. Free over
/// its inputs so the live predicate and the would-be-applied check in
/// `apply_settings` share one rule instead of restating it.
fn mod_ui_active(mod_logic: bool, hud: HudMode) -> bool {
    mod_logic && hud.shows_mod_hud()
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


/// Turn the body frame toward −gravity, weighted so weak gravity barely turns it and near-zero
/// gravity leaves it alone (never normalising a vanishing vector).
fn align_body(player: &mut Player, dt: f32) {
    let g0 = crate::player::STANDARD_GRAVITY;
    let g = player.gravity.length();
    let weight = crate::math::smooth_between(0.02 * g0 as f32, 0.10 * g0 as f32, g as f32) as f64;
    if weight > 0.0 {
        let amount = 1.0 - (-crate::camera::ALIGN_RATE * weight * dt as f64).exp();
        player.orientation.align(-player.gravity / g, amount);
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameInput, Game, PendingModInput};
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    use crate::audio::{GameEvent, SoundSystem};
    use crate::input::intent::Chord;
    use crate::input::router::{Press, Router};
    use crate::modding::{Action, ActionSet, Channel, FrameContext, Message, Mod, ModContext, Mods, NoticeLevel};
    use crate::player::Player;
    use crate::render_config::RenderConfig;
    use crate::settings::Settings;
    use crate::world::World;
    use material::{Configuration, Element};
    use voxel_engine::{DVec3, Key, Vec2};

    /// The probe's immediate actions, in the router's bit order.
    const PROBE_IDS: &[&str] = &["probe.fly", "probe.type"];

    /// A mod built on the frame hook: F flies (unless the camera is detached), T takes the
    /// keyboard, and a typed word edits whatever it names. It shows every notice.
    struct Probe {
        log: Rc<RefCell<Vec<String>>>,
    }

    impl Mod for Probe {
        fn name(&self) -> &str {
            "Probe"
        }
        fn id(&self) -> &'static str {
            "probe"
        }
        fn actions(&self) -> &[Action] {
            const FLY: &[Chord] = &[Chord::key(Key::F)];
            const TYPE: &[Chord] = &[Chord::key(Key::T)];
            const ACTIONS: &[Action] = &[
                Action { id: "probe.fly", label: "Fly", default: FLY, repeat: false, held: false, immediate: true },
                Action { id: "probe.type", label: "Type", default: TYPE, repeat: false, held: false, immediate: true },
            ];
            ACTIONS
        }
        fn on_frame(&mut self, ctx: &mut FrameContext) {
            if ctx.action("probe.fly") && !ctx.game.detached {
                ctx.game.player.toggle_fly();
            }
            if ctx.action("probe.type") {
                ctx.capture_text(true);
            }
            let Some(text) = ctx.text() else { return };
            let word: String = text.chars.iter().collect();
            match word.as_str() {
                "move" => ctx.game.player.position.x += 10.0,
                "fov" => ctx.game.settings_mut().fov += 5.0,
                "same" => {
                    let fov = ctx.game.settings().fov;
                    ctx.game.settings_mut().fov = fov;
                }
                "intern" => {
                    ctx.game.world.registry_mut().intern(&Configuration::single(Element::new([1, 2, 3, 4])));
                }
                "voice" => ctx.game.events.push(GameEvent::VoiceTest),
                "chat" => ctx.game.send_chat(Channel::Local, "hello"),
                "warn" => ctx.game.notice(NoticeLevel::Warning, "careful"),
                "done" => {
                    ctx.capture_text(false);
                }
                _ => {}
            }
            self.log.borrow_mut().push(format!("typed {word} esc={}", text.escape));
        }
        fn on_message(&mut self, msg: &Message) -> bool {
            let line = match msg {
                Message::Notice(notice) => format!("notice {}", notice.text),
                Message::Chat { from, channel, text } => format!("chat {from} {channel:?} {text}"),
                Message::Joined { name } => format!("joined {name}"),
                Message::Left { name } => format!("left {name}"),
            };
            self.log.borrow_mut().push(line);
            true
        }
    }

    pub(super) fn game() -> Game {
        let world = World::with_config_lazy(1, RenderConfig::default());
        Game::new(world, Player::new(DVec3::new(0.5, 80.0, 0.5)), "probe".into())
    }

    pub(super) fn probe_mods() -> (Mods, Rc<RefCell<Vec<String>>>) {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut mods = Mods::empty();
        mods.install(Box::new(Probe { log: log.clone() }), true);
        (mods, log)
    }

    /// The immediate actions named `ids` (from [`PROBE_IDS`]) fired.
    fn fired(ids: &[&str]) -> ActionSet {
        let mut set = ActionSet::NONE;
        for id in ids {
            set.insert(PROBE_IDS.iter().position(|p| p == id).expect("a probe action"));
        }
        set
    }

    /// Run the frame hook on `input`; the audio facts it queued.
    fn hook(game: &mut Game, mods: &mut Mods, settings: &mut Settings, input: &FrameInput) -> Vec<GameEvent> {
        let (mut sound, _) = SoundSystem::mute();
        let mut events = Vec::new();
        game.frame_hook(input, (800, 600), PROBE_IDS, mods, settings, &mut sound, &mut events);
        events
    }

    /// One typed frame, as the router reports it while a mod holds the keyboard.
    fn typing(word: &str, escape: bool) -> FrameInput {
        FrameInput { is_text: true, text_chars: word.chars().collect(), g_escape: escape, ..FrameInput::default() }
    }

    /// Take the keyboard with T, then type `word`.
    pub(super) fn type_word(game: &mut Game, mods: &mut Mods, settings: &mut Settings, word: &str) -> Vec<GameEvent> {
        hook(game, mods, settings, &FrameInput { immediate: fired(&["probe.type"]), ..FrameInput::default() });
        assert!(mods.text_captured());
        hook(game, mods, settings, &typing(word, false))
    }

    /// Flight is a mod's immediate action: it reaches the frame hook on the frame of the press with
    /// mod logic off, once, and the mod sees a detached camera. The core never toggles flight.
    #[test]
    fn an_immediate_action_reaches_the_frame_hook_with_mod_logic_off() {
        let mut game = game();
        let mut settings = Settings::default();
        game.mod_logic = false;
        let fly = FrameInput { immediate: fired(&["probe.fly"]), move_input: Some(Default::default()), ..FrameInput::default() };
        game.motion_phase(&fly, 1.0 / 60.0);
        assert!(!game.player.flying(), "the core never toggles flight");
        assert!(!PendingModInput::capture(&fly, true, true, None).any(), "an immediate action is not a cadence edge");

        hook(&mut game, &mut Mods::empty(), &mut settings, &fly);
        assert!(!game.player.flying(), "without a mod that flies, F does nothing");
        let (mut mods, _) = probe_mods();
        hook(&mut game, &mut mods, &mut settings, &fly);
        assert!(game.player.flying(), "the mod flies on this frame, mod logic off");
        hook(&mut game, &mut mods, &mut settings, &FrameInput::default());
        assert!(game.player.flying(), "no key, no toggle");
        game.camera.toggle_freecam(&game.player, &game.world, 90.0);
        hook(&mut game, &mut mods, &mut settings, &fly);
        assert!(game.player.flying(), "a detached camera does not fly the frozen player");
    }

    /// A capture holds the keyboard across frames; Escape reaches the holder once and ends it, so
    /// it never leaves the world.
    #[test]
    fn a_text_capture_holds_the_keyboard_until_escape() {
        let (mut game, mut settings) = (game(), Settings::default());
        let (mut mods, log) = probe_mods();
        type_word(&mut game, &mut mods, &mut settings, "hi");
        assert!(mods.text_captured() && game.text_captured(&mods));
        hook(&mut game, &mut mods, &mut settings, &typing("", true));
        assert!(!mods.text_captured(), "Escape ends the capture");
        assert_eq!(log.take(), ["typed hi esc=false", "typed  esc=true"]);
        type_word(&mut game, &mut mods, &mut settings, "done");
        assert!(!mods.text_captured(), "the mod gave the keyboard back");
    }

    #[test]
    fn the_frame_hook_edits_the_player_world_and_settings() {
        let (mut game, mut settings) = (game(), Settings::default());
        let (mut mods, _) = probe_mods();
        let blocks = game.world.registry().block_count();
        type_word(&mut game, &mut mods, &mut settings, "intern");
        assert_eq!(game.world.registry().block_count(), blocks + 1);
        assert!(!game.take_settings_changed() && !game.take_settings_dirty());
        let fov = settings.fov;
        type_word(&mut game, &mut mods, &mut settings, "fov");
        assert_eq!(settings.fov, fov + 5.0);
        assert!(game.take_settings_changed() && game.take_settings_dirty(), "changed settings are applied and saved by the app");
        assert!(!game.take_settings_changed(), "taken once");
        type_word(&mut game, &mut mods, &mut settings, "same");
        assert!(!game.take_settings_changed(), "writing an unchanged value marks nothing");
        game.force_stream = false;
        type_word(&mut game, &mut mods, &mut settings, "move");
        assert_eq!(game.player.position.x, 10.5);
        assert!(game.force_stream, "a moved player streams its destination at once (and is reported as a teleport)");
    }

    #[test]
    fn the_frame_hook_queues_the_voice_test_chat_and_notices() {
        let (mut game, mut settings) = (game(), Settings::default());
        let (mut mods, log) = probe_mods();
        assert!(type_word(&mut game, &mut mods, &mut settings, "nothing").is_empty());
        let events = type_word(&mut game, &mut mods, &mut settings, "voice");
        assert!(matches!(events.as_slice(), [GameEvent::VoiceTest]));
        type_word(&mut game, &mut mods, &mut settings, "chat");
        assert!(game.chat_out.is_empty(), "single player has no wire: queued chat is dropped");
        log.take();
        type_word(&mut game, &mut mods, &mut settings, "warn");
        game.notify("* saved");
        assert!(log.take().iter().all(|l| !l.starts_with("notice")), "notices wait for the next frame");
        hook(&mut game, &mut mods, &mut settings, &FrameInput::default());
        assert_eq!(log.take(), ["notice careful", "notice * saved"], "a hook's notice first, then the app's");
        assert!(game.notices.is_empty());
    }

    #[test]
    fn default_frame_input_carries_no_edges() {
        let inert = FrameInput::default();
        assert!(inert.move_input.is_none());
        assert_eq!(inert.look_delta, Vec2::ZERO);
        assert!(!inert.is_text);
        assert!(inert.immediate.is_empty());
        assert!(!inert.g_escape);
        assert!(!inert.g_hud);
        assert!(!inert.g_shot);
        assert!(!inert.g_minimap);
        assert!(!inert.g_person);
        assert!(!inert.g_freecam);
        assert!(!inert.toggle_capture);
        assert!(!inert.do_break);
        assert!(!inert.do_place);
        assert!(inert.actions.is_empty());
        assert!(!PendingModInput::capture(&inert, true, true, Some((1, 2, 3))).any());
    }

    /// An edge captured while the mod tick is skipped is replayed on the next tick, once.
    #[test]
    fn actions_queue_across_a_skipped_mod_tick() {
        struct Watch {
            seen: Rc<Cell<u32>>,
        }
        impl Mod for Watch {
            fn name(&self) -> &str {
                "Watch"
            }
            fn id(&self) -> &'static str {
                "watch"
            }
            fn actions(&self) -> &[Action] {
                const CHORDS: &[Chord] = &[Chord::key(Key::Num3)];
                const ACTIONS: &[Action] = &[Action {
                    id: "probe.ping",
                    label: "Ping",
                    default: CHORDS,
                    repeat: false,
                    held: false,
                    immediate: false,
                }];
                ACTIONS
            }
            fn update(&mut self, ctx: &mut ModContext) {
                if ctx.action("probe.ping") {
                    self.seen.set(self.seen.get() + 1);
                }
            }
        }
        let seen = Rc::new(Cell::new(0));
        let mut mods = Mods::empty();
        mods.install(Box::new(Watch { seen: seen.clone() }), true);
        let mut router = Router::new();
        router.sync_actions(&mods);
        let sample = router.sample(&Press::key(Key::Num3), 1.0 / 60.0, true, true);
        let input = FrameInput { actions: sample.actions, ..FrameInput::default() };
        let pending = PendingModInput::capture(&input, true, true, None);
        assert!(pending.any());
        let mut game = game();
        game.pending_mod_input.push(pending);
        assert_eq!(seen.get(), 0);
        assert_eq!(game.pending_mod_input.len(), 1);
        let mut events = Vec::new();
        game.mod_tick((800, 600), &mut mods, &mut events, router.action_ids());
        assert_eq!(seen.get(), 1);
        assert!(game.pending_mod_input.is_empty());
    }

    #[test]
    fn sky_altitude_is_above_the_nearest_body() {
        let world = World::with_config_lazy(1, RenderConfig::default());
        let game = Game::new(world, Player::new(DVec3::new(0.5, 80.0, 0.5)), "alt".into());
        // A flat world's datum is y = 0.
        assert_eq!(game.sky_altitude(DVec3::new(12.0, 40.0, -3.0)), 40.0);
        let world = World::with_kind(1, RenderConfig::default(), crate::world::generation::WorldgenKind::Diffusion, false);
        let game = Game::new(world, Player::new(DVec3::new(0.5, 80.0, 0.5)), "alt".into());
        // Altitude above the start world's datum (a sphere; the old cube face was y = 0).
        let cosmos = game.world.terrain().cosmos().expect("a cosmos");
        let eye = DVec3::new(12.0, 40.0, -3.0);
        let home = cosmos.home();
        let cosmos = game.world.terrain().cosmos().expect("a cosmos");
        assert!((game.sky_altitude(eye) - cosmos.altitude(&home, eye)).abs() < 1e-6);
        let verdance = cosmos
            .bodies()
            .iter()
            .find(|b| b.kind == crate::world::terrain::cosmos::Kind::Verdant)
            .expect("Verdance");
        let crate::world::terrain::cosmos::Shape::Ball { r } = verdance.shape else { panic!("round") };
        let above = verdance.centre_f() + DVec3::new(0.0, 0.0, r as f64 + 300.0);
        assert!((game.sky_altitude(above) - 300.0).abs() < 1e-6, "over Verdance");
        assert!(game.sky_altitude(DVec3::new(9.0e8, 9.0e8, 9.0e8)) > 1.0e8, "deep space");

        let hollow = cosmos
            .bodies()
            .iter()
            .find(|b| b.kind == crate::world::terrain::cosmos::Kind::Hollow)
            .expect("the Hollow");
        let crate::world::terrain::cosmos::Shape::Shell { outer, inner } = hollow.shape else {
            panic!("the Hollow is a shell");
        };
        let halfway = hollow.centre_f() + DVec3::new(0.0, inner as f64 * 0.5, 0.0);
        let lip = hollow.centre_f() + DVec3::new(0.0, inner as f64 - 1.0, 0.0);
        assert_eq!(game.sky_altitude(halfway), 1.0e9, "cavity air");
        assert_eq!(game.sky_altitude(lip), 1.0e9, "just inside the inner surface");
        let outside = hollow.centre_f() + DVec3::new(0.0, outer as f64 + 300.0, 0.0);
        assert!(
            (game.sky_altitude(outside) - 300.0).abs() < 1e-6,
            "outside the Hollow"
        );
        let shell = hollow.centre_f() + DVec3::new(0.0, inner as f64 + 1.0, 0.0);
        let shell_alt = game.sky_altitude(shell);
        assert!(shell_alt.abs() < 10.0, "inside the shell rock, altitude {shell_alt}");
        assert!((shell_alt - hollow.altitude(shell)).abs() < 1e-6);
    }

    /// Quiet-frame micro-benchmark: the three remaining fixed costs at
    /// Minimum/Fast. Reports ns/call for the idle predicates vs the work they
    /// skip. Ignored: a timing run, not a correctness gate.
    #[test]
    #[ignore]
    fn quiet_frame_fixed_costs() {
        use crate::audio::{AudioService, SoundSystem};
        use crate::input::router::Router;
        use crate::modding::Notices;
        use std::hint::black_box;
        use std::time::Instant;

        const N: u32 = 50_000;

        let game = Game::scripted(1, RenderConfig::default());
        let t0 = Instant::now();
        for _ in 0..N {
            black_box(game.world().anything_in_flight());
        }
        let in_flight_ns = t0.elapsed().as_nanos() as f64 / f64::from(N);

        let mut router = Router::new();
        let t0 = Instant::now();
        for _ in 0..N {
            router.drain_frame();
        }
        let drain_ns = t0.elapsed().as_nanos() as f64 / f64::from(N);

        let (mut sound, _symbols) = SoundSystem::mute();
        let mut audio = AudioService::new();
        let world = World::generate();
        let pos = DVec3::new(0.5, 80.0, 0.5);
        let mut notices = Notices::default();
        let listener = crate::audio::Listener {
            pos,
            yaw: 0.0,
            pitch: 0.0,
            frame: glam::DQuat::IDENTITY,
        };
        audio.finish(&mut sound, &world, listener, 1.0 / 60.0, &mut notices);

        let t0 = Instant::now();
        for _ in 0..N {
            black_box(audio.can_skip(&sound, true, pos));
        }
        let skip_ns = t0.elapsed().as_nanos() as f64 / f64::from(N);

        println!("quiet_frame_fixed_costs ({N} iters):");
        println!("  anything_in_flight:     {in_flight_ns:.1} ns");
        println!("  Router::drain_frame:    {drain_ns:.1} ns");
        println!("  can_skip (idle):        {skip_ns:.1} ns");
        assert!(audio.can_skip(&sound, true, pos), "the skip predicate must hold on the idle pose used above");
        assert!(!game.world().anything_in_flight());
    }

    /// A settled scripted game at the Minimum preset with input locked, its settings adopted as
    /// world entry adopts them.
    pub(super) fn quiet_minimum_game() -> (Game, Settings) {
        let mut settings = Settings::default();
        assert!(settings.select_preset("minimum"));
        let render = settings.render_config();
        let mut game = Game::scripted(1, render);
        game.scripted = false;
        game.set_input_locked(true);
        game.world_mut()
            .set_view_distances(settings.render_distance, settings.vertical_distance);
        game.world_mut().transition_lighting(settings.lighting);
        game.world_mut().set_ao_flag(settings.ao);
        game.world_mut()
            .set_render_lanes(settings.occlusion, settings.lod2);
        game.adopt_gameplay_settings(&settings, render);
        let pos = game.player().position;
        game.world_mut().settle_around(pos);
        assert!(
            game.world().entry_complete(),
            "settled: {}",
            game.world().entry_debug()
        );
        (game, settings)
    }

    #[test]
    fn quiet_minimum_frame_allocates_nothing_and_reads_the_clock_once() {
        use crate::alloc_count;
        use crate::audio::{AudioService, SoundSystem};
        use crate::input::router::Router;
        use crate::ui::HudMode;

        let (mut game, mut settings) = quiet_minimum_game();

        let (mut sound, symbols) = SoundSystem::mute();
        let mut audio = AudioService::new();
        let mut router = Router::new();
        let mut mods = crate::modding::testing::standard();
        const DT: f32 = 1.0 / 60.0;

        let mut last_allocs = u64::MAX;
        let mut last_bytes = u64::MAX;
        let mut last_clocks = u32::MAX;
        let mut last_calls = alloc_count::EngineCalls {
            uniforms: 0,
            settings_apply: 0,
            tex_layers: 0,
        };
        for i in 0..10 {
            alloc_count::reset();
            crate::sched::reset_clock();
            game.tick_quiet(DT, &mut router, &mut sound, &mut audio, &symbols, &mut settings, &mut mods);
            if i >= 5 {
                last_allocs = alloc_count::alloc_count();
                last_bytes = alloc_count::alloc_bytes();
                last_clocks = crate::sched::clock();
                last_calls = alloc_count::engine_calls();
                assert_eq!(
                    last_bytes, 0,
                    "quiet frame {i} allocated {last_allocs} times / {last_bytes} bytes; engine={last_calls:?}"
                );
                assert!(
                    last_clocks <= 1,
                    "quiet frame {i} read the clock {last_clocks} times"
                );
            }
        }
        println!(
            "quiet minimum frame (last of 10): allocs={last_allocs} bytes={last_bytes} clocks={last_clocks} engine={last_calls:?}"
        );

        // HUD strings: an unchanged snapshot must not allocate after the cache fills.
        game.theme.hud = HudMode::Full;
        for i in 0..4 {
            alloc_count::reset();
            crate::sched::reset_clock();
            game.tick_quiet(DT, &mut router, &mut sound, &mut audio, &symbols, &mut settings, &mut mods);
            if i >= 2 {
                assert_eq!(
                    alloc_count::alloc_bytes(),
                    0,
                    "HUD snapshot frame {i} allocated {} bytes",
                    alloc_count::alloc_bytes()
                );
            }
        }
    }

    #[test]
    fn quiet_minimum_frame_after_draining_the_scheduler_allocates_nothing() {
        use crate::alloc_count;
        use crate::audio::{AudioService, SoundSystem};
        use crate::input::router::Router;

        let (mut game, mut settings) = quiet_minimum_game();
        let pos = game.player().position;
        let (x, y, z) = (
            pos.x.floor() as i32,
            pos.y.floor() as i32,
            pos.z.floor() as i32,
        );
        game.world_mut().note_cell_changed(x, y, z);
        game.world_mut().note_cell_changed(x + 1, y, z);
        assert!(
            game.world().reactions().pending() > 0,
            "draining test needs a non-empty pending set"
        );
        while game.world().reactions().pending() > 0 {
            let _ = game.world_mut().tick_reactions();
        }
        game.world_mut().settle_around(pos);
        assert!(
            game.world().entry_complete(),
            "settled after drain: {}",
            game.world().entry_debug()
        );

        let (mut sound, symbols) = SoundSystem::mute();
        let mut audio = AudioService::new();
        let mut router = Router::new();
        let mut mods = crate::modding::testing::standard();
        const DT: f32 = 1.0 / 60.0;
        for i in 0..10 {
            alloc_count::reset();
            crate::sched::reset_clock();
            game.tick_quiet(DT, &mut router, &mut sound, &mut audio, &symbols, &mut settings, &mut mods);
            if i >= 5 {
                assert_eq!(
                    alloc_count::alloc_bytes(),
                    0,
                    "quiet frame {i} after drain allocated {} times / {} bytes",
                    alloc_count::alloc_count(),
                    alloc_count::alloc_bytes()
                );
            }
        }
    }
}
