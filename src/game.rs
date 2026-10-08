//! game.rs owns the in-world state — world, player, physics, console — and runs a
//! frame of it: input, movement, block interaction, mods, streaming, and drawing.
//! The window and the menu/play state machine live one level up in [`app`](crate::app);
//! a `Game` is handed the engine each frame and reports back whether to keep playing
//! or return to the menu.
use std::time::{Duration, Instant};

mod draw;

use voxel_engine::{Color, DVec3, Engine, Vec2};

use crate::audio::{AudioService, AudioView, CueSymbols, GameEvent, ModLink, PeerAudio, SoundSystem, StepPose};
use crate::block::{BlockId, AIR};
use crate::camera::{CameraMode, CameraPose, FlyAxes, GameCamera};
use crate::console::{self, Console};
use crate::derived::Revision;
use crate::input::intent::{GameplayEvent, GlobalEvent, MenuEvent};
use crate::input::router::{Context, Router, View};
use crate::input::{look, movement};
use crate::interact;
use crate::math::{Aabb, Bounded};
use crate::minimap::{MapSample, Minimap, MinimapConfig};
use crate::modding::{ActionSet, Command, CommandContext, ModContext, Mods, ToolUse};
use crate::net::chat;
use crate::net::client::{Connection, Incoming};
use crate::player::Player;
use crate::presence::{self, Stance, WireAction};
use crate::save;
use crate::sched::{Ctx as SchedCtx, RateGate};
use crate::settings::Settings;
use crate::sim::Simulation;
use crate::sky::Sky;
use crate::ui::{self, HudElement, HudMode, Theme};
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
/// context: `is_text` carries the console's typing, everything else gameplay.
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
    /// The flight key: an intent for the mods (the core has no flight toggle).
    toggle_fly: bool,
    /// Mod actions that fired this frame.
    actions: ActionSet,
    /// Signed scroll steps this frame.
    wheel: i8,
    nav_up: bool,
    nav_down: bool,
    nav_left: bool,
    nav_right: bool,
    nav_tab: bool,
    nav_confirm: bool,
    open_console: bool,
    open_chat: bool,
    toggle_capture: bool,
    g_escape: bool,
    g_hud: bool,
    g_shot: bool,
    g_minimap: bool,
    g_person: bool,
    g_freecam: bool,
}

impl FrameInput {
    fn inert() -> Self {
        Self::default()
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

/// Facts for the audio hook. `events` is what this frame already knows; footsteps are added inside.
struct AudioPhase<'a> {
    dt: f32,
    input: &'a FrameInput,
    sound: &'a mut SoundSystem,
    audio: &'a mut AudioService,
    cues: &'a CueSymbols,
    settings: &'a Settings,
    events: Vec<GameEvent>,
    active: bool,
    mods: &'a mut Mods,
    ids: &'a [&'static str],
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
    nav_up: bool,
    nav_down: bool,
    nav_left: bool,
    nav_right: bool,
    nav_tab: bool,
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
            actions: input.actions,
            wheel: input.wheel,
            mod_ui: allow_ui,
            nav_up: allow_ui && input.nav_up,
            nav_down: allow_ui && input.nav_down,
            nav_left: allow_ui && input.nav_left,
            nav_right: allow_ui && input.nav_right,
            nav_tab: allow_ui && input.nav_tab,
            nav_confirm: allow_ui && input.nav_confirm,
        }
    }

    fn any(self) -> bool {
        self.place
            || !self.actions.is_empty()
            || self.wheel != 0
            || self.nav_up
            || self.nav_down
            || self.nav_left
            || self.nav_right
            || self.nav_tab
            || self.nav_confirm
    }

    fn clear_ui(&mut self) {
        self.actions = ActionSet::NONE;
        self.wheel = 0;
        self.mod_ui = false;
        self.nav_up = false;
        self.nav_down = false;
        self.nav_left = false;
        self.nav_right = false;
        self.nav_tab = false;
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
    /// Breaking awarded this configuration; a rejection revokes it.
    Break(crate::block::BlockId),
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
    /// Cached `peer_color(save_name)` — local third-person body tint.
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
    /// In-world UI look and HUD visibility (see [`ui::Theme`]).
    theme: Theme,
    /// Day/night clock, atmosphere colour, weather, and the lighting edge into
    /// voxel shading (see [`crate::sky`]).
    sky: Sky,
    /// Top-down minimap: throttled terrain raster drawn in the HUD corner.
    /// `None` when the minimap lane is disabled — no raster state retained,
    /// no refresh clock read, no draw.
    minimap: Option<Minimap>,
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
    mod_hud: bool,
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
    peer_pose_scratch: Vec<PeerAudio>,
    /// Up-axes parallel to [`peer_pose_scratch`]. Kept off the public peer record.
    peer_up_scratch: Vec<crate::coord::Face>,
    /// Last frame's named-phase durations, for the stall detector.
    phases: FramePhases,
    hud_scratch: Vec<HudElement>,
    /// Menu notice taken when a network session leaves. The console goes with the game.
    leave_notice: Option<String>,
}

/// Durations of `Game::update` phases, sampled every frame for stall logs.
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

impl Game {
    pub fn new(mut world: World, player: Player, save_name: String) -> Self {
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
            local_color: draw::peer_color(&save_name),
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
            mod_hud: true,
            player_models: true,
            name_tags: true,
            content_rev: Revision::default(),
            drawing: draw::DrawState::new(),
            placement_scratch: Vec::new(),
            peer_pose_scratch: Vec::new(),
            peer_up_scratch: Vec::new(),
            phases: FramePhases::default(),
            hud_scratch: Vec::new(),
            leave_notice: None,
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

    /// Push every live-applicable setting into this game: engine values, the
    /// world's view volume and meshing lanes, the per-frame look config, HUD
    /// mode, lane gates, and the throttled clocks. THE one path — world entry
    /// and in-game `/gfx` edits both come through here, so they can never
    /// drift apart. (World-construction lanes — occlusion/lod2 — stay
    /// entry-only by design; see `App::enter_game`.)
    pub fn apply_settings(&mut self, eng: &mut Engine, settings: &mut Settings) {
        settings.apply(eng);
        let render = self.visual_mask.effective_render(settings);
        eng.set_flags(render.engine_flags());
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

        let mod_ui_will_be_active =
            mod_ui_active(settings.mod_logic, settings.mod_hud, self.theme.hud);
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
        self.mod_hud = settings.mod_hud;
        self.player_models = settings.player_models;
        self.name_tags = settings.name_tags;
    }

    /// Whether a modal supplied by the mod layer can both be seen and receive
    /// input. Keeping one predicate for routing and Escape prevents invisible
    /// overlays when either the mod lane or the master HUD is disabled.
    fn mod_ui_active(&self) -> bool {
        mod_ui_active(self.mod_logic, self.mod_hud, self.theme.hud)
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
        let mut player = Player::new(pos);
        player.stand_in(world.gravity_at(player.position).accel);
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

    /// Surface a status line in the in-world console (save/load notices).
    pub fn notify(&mut self, line: impl Into<String>) {
        self.console.print(line);
    }

    /// Whether this is a networked session (its world is a server mirror, not a
    /// local save, so the app does not autosave it).
    pub fn is_multiplayer(&self) -> bool {
        self.net.is_some()
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
            self.world.stream(
                self.player.position,
                Some(&mut *eng),
                &mut self.sched,
                mods.appearance(),
            );
            let clocks = self.sched.clocks(dt);
            let mut sched_ctx = SchedCtx::new(&mut self.world, Some(&mut *eng));
            self.sched.tick(&mut sched_ctx, &clocks);
            return Signal::Continue;
        }

        // Advance the day/night clock (singleplayer drives it locally; a server
        // sync overrides `day` on arrival), throttled by `sky_hz`. With the
        // lane off, compose samples fixed noon without mutating authoritative
        // clock state; re-enabling resumes the stored time instead of freezing
        // a stripped profile at night.
        if self.render.day_night {
            let steps = self.sky_gate.steps(dt);
            if steps != 0 {
                self.sky
                    .tick(steps as f64 * self.sky_gate.step_dt(dt) as f64);
            }
        } else {
            self.sky_gate.reset();
        }

        // This frame's audio facts, accumulated across the phases. Footsteps are
        // added at the audio commit; mods choose the cues and the voice sessions.
        let mut events: Vec<GameEvent> = Vec::new();

        self.phases = FramePhases::default();
        let t = Instant::now();
        if let Some(signal) = self.net_phase(mods, &mut events) {
            return signal;
        }
        self.phases.net = t.elapsed();
        let t = Instant::now();
        let input = self.input_phase(eng, router, mods, dt);
        self.phases.input = t.elapsed();
        // The overlay may consume the frame (console typing, opening chat): movement
        // and interaction run only on an unconsumed frame. Streaming still runs
        // while a spawn/teleport slab is outstanding so loading progresses with
        // the console open. A still singleplayer frame skips the mixer when
        // nothing is sounding; only a real exit short-circuits the rest of the frame.
        let t = Instant::now();
        let overlay = self.overlay_phase(OverlayPhase {
            input: &input,
            eng,
            router,
            mods,
            settings,
            sound,
            events: &mut events,
        });
        self.phases.overlay = t.elapsed();
        let consumed = match overlay {
            Some(Signal::ExitToMenu) => return Signal::ExitToMenu,
            Some(Signal::Continue) => true,
            None => false,
        };
        let ready = self.world.spawn_ready();
        if !consumed && ready {
            let t = Instant::now();
            let detached = self.motion_phase(&input, dt);
            self.phases.motion = t.elapsed();
            let t = Instant::now();
            self.interact_phase(&input, detached, dt, eng, mods, &mut events, router.action_ids());
            self.phases.interact = t.elapsed();
        }
        if !consumed || !ready {
            let t = Instant::now();
            self.stream_phase(eng, dt, mods);
            self.phases.stream = t.elapsed();
        }
        let active = !consumed;
        let t = Instant::now();
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
        self.phases.audio = t.elapsed();
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

    /// Drain server events and send our heartbeat. `Some(ExitToMenu)` when the
    /// server dropped us. Runs before input so edits and chat keep flowing even
    /// while the console is open or the player stands still — and the move
    /// report doubles as the keepalive, so it too runs unconditionally.
    fn net_phase(&mut self, mods: &mut Mods, events: &mut Vec<GameEvent>) -> Option<Signal> {
        // The overwhelmingly common singleplayer path should not even enter a
        // profiling scope or call through the event-poll seam.
        self.net.as_ref()?;
        let net_disconnected = {
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::NetEvents);
            self.apply_net_events(mods, events)
        };
        if let Some(reason) = net_disconnected {
            let interrupted = reason == crate::net::client::INTERRUPTED;
            let line = if interrupted {
                crate::net::client::INTERRUPTED.to_string()
            } else if reason.to_ascii_lowercase().contains("shutting down") {
                "* server shutting down".to_string()
            } else if reason.is_empty() {
                "* disconnected from server".to_string()
            } else {
                format!("* disconnected: {reason}")
            };
            self.console.print(line.clone());
            self.leave_notice = Some(line);
            return Some(Signal::ExitToMenu);
        }
        if let Some(net) = &mut self.net {
            net.sync_cruise(self.player.cruise.map(|c| c.speed));
            net.send_move(
                self.player.position,
                self.player.orientation.yaw,
                self.player.orientation.pitch,
                self.player.orientation.frame,
                self.player.velocity().as_vec3(),
                self.player.up_axis,
                Stance::of_player(&self.player),
            );
        }
        None
    }

    /// The once-per-frame router transition: pick the exclusive context (Text
    /// while the console captures typing) and snapshot every intent into plain
    /// data, so the router borrow ends before any `&mut Engine` side effects
    /// (screenshot, cursor grab, console open) run in later phases.
    fn input_phase(&mut self, eng: &mut Engine, router: &mut Router, mods: &Mods, dt: f32) -> FrameInput {
        router.sync_actions(mods);
        router.set_context(if self.console.is_open() {
            Context::Text
        } else {
            Context::Gameplay
        });

        if self.input_locked {
            router.drain_frame();
            return FrameInput::inert();
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
                // The flight key reaches the mods whatever the mod cadence (see `fly_key`).
                f.toggle_fly = gp.event(GameplayEvent::ToggleFly);
                f.actions = gp.actions();
                f.wheel = gp.wheel();
                if self.mod_logic {
                    f.do_place = gp.event(GameplayEvent::Place);
                }
                if mod_ui {
                    f.nav_up = gp.overlay_nav(MenuEvent::Up);
                    f.nav_down = gp.overlay_nav(MenuEvent::Down);
                    f.nav_left = gp.overlay_nav(MenuEvent::Left);
                    f.nav_right = gp.overlay_nav(MenuEvent::Right);
                    f.nav_tab = gp.overlay_nav(MenuEvent::NextTab);
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
        f.g_person = global.event(GlobalEvent::CyclePerson);
        f.g_freecam = global.event(GlobalEvent::ToggleFreecam);
        f
    }

    /// Console, escape routing, and the global toggles (mouse capture, HUD
    /// cycle, screenshot, minimap, camera modes). `Some` consumes the frame:
    /// while typing, nothing below the console runs.
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

        if self.input_locked {
            return None;
        }

        // Text context: the console owns all input; nothing else runs. Esc is
        // the game's call (the Text view has no bindable events), and here it
        // means "close the console", never "leave the world".
        if input.is_text {
            self.drop_pending_edges();
            if input.g_escape {
                self.console.close();
                return Some(Signal::Continue);
            }
            if let Some(line) = self
                .console
                .handle_input(&input.text_chars, input.text_edit, mods.commands())
            {
                self.submit_line(line, eng, settings, sound, events, mods);
            }
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

        // Open the console: `/` (OpenConsole) pre-fills a slash, `T` (OpenChat)
        // does not. Drain the char queue so the opening key isn't also typed.
        if input.open_console || input.open_chat {
            self.drop_pending_edges();
            self.console.open(input.open_console);
            while eng.get_char_pressed().is_some() {}
            return Some(Signal::Continue);
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
            settings.save();
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

    /// Drop every latched input edge — called when a modal (console, menu
    /// exit) takes over the frame, so stale edges can't fire after it closes.
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
                                &atlas.clone(),
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
    /// The flight key: the first mod that offers flight takes it on this frame, whatever the mod
    /// cadence or the mod-logic setting. A detached camera never flies the frozen player.
    fn fly_key(&mut self, input: &FrameInput, detached: bool, mods: &mut Mods) {
        if input.toggle_fly && !detached {
            mods.on_toggle_fly(&mut self.player, &self.world);
        }
    }

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
        self.fly_key(input, detached, mods);

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
                    nav_up: edges.nav_up,
                    nav_down: edges.nav_down,
                    nav_left: edges.nav_left,
                    nav_right: edges.nav_right,
                    nav_tab: edges.nav_tab,
                    nav_confirm: edges.nav_confirm,
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
        let mut sched_ctx = SchedCtx::new(&mut self.world, Some(&mut *eng));
        self.sched.tick(&mut sched_ctx, &clocks);

        // The topology pass (selection/admission/unload) rides `stream_hz`;
        // forced refreshes (teleports, freecam/HUD changes, settings) run out
        // of band so correctness never waits on a 15 Hz clock. `stream` owns
        // the result pump on those frames, so skip the extra pump here.
        let stream_due = if std::mem::take(&mut self.force_stream) {
            self.stream_gate.reset();
            true
        } else {
            self.stream_gate.steps(dt) != 0
        };
        // A cruise holds the world still: in-flight work lands, nothing new streams around the
        // player. Ending it streams the destination like a teleport.
        if !stream_due || self.player.cruising() {
            self.world
                .pump(Some(&mut *eng), &mut self.sched, mods.appearance());
            return;
        }

        let stream_center = match &self.camera.mode {
            CameraMode::Free { rig, .. } => rig.pos,
            CameraMode::Person(_) => self.player.position,
        };
        self.world.stream(
            stream_center,
            Some(&mut *eng),
            &mut self.sched,
            mods.appearance(),
        );

        // A hidden/minimal HUD does no minimap clock read or terrain raster
        // work; refreshes share streaming's cadence instead of waking alone.
        if self.theme.hud.shows_minimap()
            && let Some(minimap) = &mut self.minimap
        {
            // The map follows the body, not the freecam. The throttle rides the
            // scheduler's interval gate (advanced in clocks() above); the recenter
            // half stays inside Minimap::due.
            let sample = MapSample::from_player(
                &self.world,
                self.player.position,
                self.player.up_axis,
                self.player.orientation.frame,
                self.player.orientation.yaw,
            );
            let due = self.sched.interval_due(self.minimap_interval);
            if minimap.refresh(eng, &self.world, sample, due) {
                self.sched.interval_reset(self.minimap_interval);
            }
        }
    }

    /// Headless quiet frame: input drain, motion (inert), scheduler, stream/pump,
    /// silent audio, HUD/lighting caches — the pieces `update` + `draw` run, in
    /// order, without an Engine.
    #[cfg(test)]
    fn tick_quiet(
        &mut self,
        dt: f32,
        router: &mut Router,
        sound: &mut SoundSystem,
        audio: &mut AudioService,
        cues: &CueSymbols,
        settings: &Settings,
        mods: &mut Mods,
    ) {
        if self.render.day_night {
            let steps = self.sky_gate.steps(dt);
            if steps != 0 {
                self.sky
                    .tick(steps as f64 * self.sky_gate.step_dt(dt) as f64);
            }
        } else {
            self.sky_gate.reset();
        }
        let events: Vec<GameEvent> = Vec::new();
        if self.input_locked {
            router.drain_frame();
        }
        let input = FrameInput::inert();
        if self.world.spawn_ready() {
            let _ = self.motion_phase(&input, dt);
        }
        let clocks = self.sched.clocks(dt);
        let mut sched_ctx = SchedCtx::new(&mut self.world, None);
        self.sched.tick(&mut sched_ctx, &clocks);
        let stream_due = if std::mem::take(&mut self.force_stream) {
            self.stream_gate.reset();
            true
        } else {
            self.stream_gate.steps(dt) != 0
        };
        let stream_center = match &self.camera.mode {
            CameraMode::Free { rig, .. } => rig.pos,
            CameraMode::Person(_) => self.player.position,
        };
        if stream_due && !self.player.cruising() {
            self.world
                .stream(stream_center, None, &mut self.sched, mods.appearance());
        } else {
            self.world
                .pump(None, &mut self.sched, mods.appearance());
        }
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

    /// Hand this frame to the mods. A still singleplayer frame still runs the hook,
    /// on stack data, and skips the mixer unless the hook queued work.
    fn commit_audio(&mut self, phase: AudioPhase<'_>) {
        let AudioPhase {
            dt,
            input,
            sound,
            audio,
            cues,
            settings,
            mut events,
            active,
            mods,
            ids,
        } = phase;
        let idle = events.is_empty() && input.actions.is_empty();
        let skip = self.net.is_none() && audio.can_skip(sound, idle, self.player.position);
        if skip {
            self.dispatch_audio(dt, input, sound, audio, cues, settings, mods, ids, &[], &events);
            if !audio.dirty() {
                sound.poll_starvation();
                return;
            }
            self.submit_audio(dt, sound, audio);
            return;
        }

        let now = crate::sched::now();
        let mut peers = std::mem::take(&mut self.peer_pose_scratch);
        let mut ups = std::mem::take(&mut self.peer_up_scratch);
        peers.clear();
        ups.clear();
        if let Some(net) = &self.net {
            for peer in net.peers() {
                let rendered = peer.sample(now);
                peers.push(PeerAudio {
                    id: peer.id(),
                    at: rendered.pos.0,
                    feet: rendered.pos.feet(rendered.stance, rendered.up).0,
                    visible: peer.visible(),
                    gait: rendered.phase,
                    speed: rendered.speed,
                });
                ups.push(rendered.up);
            }
        }
        // A console-owned frame does not step the player, so a stale walk speed
        // must not fire a footstep.
        let velocity = if active { self.player.velocity() } else { DVec3::ZERO };
        audio.footsteps(
            StepPose {
                feet: self.player.feet(),
                velocity,
                on_ground: self.player.on_ground(),
                up: self.player.up_axis,
            },
            &self.world,
            dt,
            &peers,
            &ups,
            &mut events,
        );
        self.dispatch_audio(dt, input, sound, audio, cues, settings, mods, ids, &peers, &events);
        self.submit_audio(dt, sound, audio);
        self.peer_pose_scratch = peers;
        self.peer_up_scratch = ups;
    }

    fn dispatch_audio(
        &mut self,
        dt: f32,
        input: &FrameInput,
        sound: &mut SoundSystem,
        audio: &mut AudioService,
        cues: &CueSymbols,
        settings: &Settings,
        mods: &mut Mods,
        ids: &[&'static str],
        peers: &[PeerAudio],
        events: &[GameEvent],
    ) {
        let pos = self.player.position;
        let view = AudioView {
            dt,
            pos,
            peers,
            in_world: true,
            voice_enabled: settings.voice_enabled,
            hear_voice: settings.voice_incoming,
            actions: input.actions,
            ids,
        };
        let mut link = ModLink::new(self.net.as_mut());
        let mut api = audio.api(sound, cues, Some(&self.world), Some(&mut self.console));
        for event in events {
            mods.on_game_event(event, &mut api);
        }
        mods.on_audio(&view, &mut api, &mut link);
    }

    fn submit_audio(&mut self, dt: f32, sound: &mut SoundSystem, audio: &mut AudioService) {
        let listener = crate::audio::Listener {
            pos: self.player.position,
            yaw: self.player.orientation.yaw,
            pitch: self.player.orientation.pitch,
            frame: self.player.orientation.frame,
        };
        audio.finish(sound, &self.world, listener, dt, &mut self.console);
    }

    /// Drain queued server messages: apply world edits, resolve our own edit
    /// verdicts (rolling back rejected predictions), surface chat, and report
    /// a lost connection. `Some(reason)` if the server dropped us.
    fn apply_net_events(&mut self, mods: &mut Mods, events: &mut Vec<GameEvent>) -> Option<String> {
        let incoming = match &mut self.net {
            Some(net) => net.poll(),
            None => return None,
        };
        let mut disconnected = None;
        for event in incoming {
            match event {
                Incoming::Edit { x, y, z, spec } => {
                    // Resolve the portable spec against our own palette, then
                    // apply. The connection already dropped stale revisions,
                    // and our own edits come back as acks, not broadcasts.
                    // Snapshot the cell first so a remote break names the block that
                    // WAS there (its sound class), not a generic default.
                    let prev = self.world.block_at(x, y, z);
                    let id = save::parse_block(self.world.registry_mut(), &spec);
                    self.world.set_block(x, y, z, id);
                    self.world.note_cell_changed(x, y, z);
                    let at = sound_at(&self.world, x, y, z);
                    events.push(if id == AIR {
                        GameEvent::BlockBroken { at, block: prev, local: false }
                    } else {
                        GameEvent::BlockPlaced { at, block: id, local: false }
                    });
                }
                Incoming::Mutation { x, y, z, spec } => {
                    // Snapshot content: apply silently. A client is never the reaction
                    // authority, so there is nothing to note; a cascade of a thousand
                    // cells must not play a thousand block cues.
                    let id = save::parse_block(self.world.registry_mut(), &spec);
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
                        PendingKind::Break(id) => {
                            self.player.inventory.revoke(id, 1);
                            mods.on_break_rejected(id);
                        }
                        PendingKind::Place(id) => {
                            self.player.inventory.add(id, 1);
                            mods.on_place_rejected(id, &self.world);
                        }
                    }
                }
                Incoming::Position { pos, frame, up } => {
                    // Authoritative snap-back (refused teleport or implausible
                    // move): request the collision slab and freeze until it
                    // lands, exactly like a local teleport. The server's
                    // MOVE_WINDOW_CAP_SECS envelope tolerates a brief pause.
                    self.world.prepare_around(pos);
                    self.player.position = pos;
                    self.player.orientation.frame = frame;
                    self.player.up_axis = up;
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
                    self.console
                        .push(line.then(ui::Role::Muted, text.to_string()));
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
                Incoming::Disconnected { reason } => disconnected = Some(reason),
                Incoming::Interrupted => {
                    self.console
                        .push(ui::Line::of(ui::Role::Warning, crate::net::client::INTERRUPTED));
                }
                Incoming::ToolResult { req, reacted, cell, cell_spec, tool_spec } => {
                    let Some(tool) = self.pending_tools.remove(&req) else { continue };
                    if !reacted {
                        mods.on_tool_used(ToolUse::NoReaction);
                        continue;
                    }
                    let (x, y, z) = cell;
                    let target = self.world.block_at(x, y, z);
                    let new_cell = save::parse_block(self.world.registry_mut(), &cell_spec);
                    let new_tool = save::parse_block(self.world.registry_mut(), &tool_spec);
                    self.world.set_block(x, y, z, new_cell);
                    self.finish_tool_change(tool, new_tool, target, new_cell, mods);
                    events.push(GameEvent::ToolReacted { at: sound_at(&self.world, x, y, z), block: target });
                }
                Incoming::PeerSwing { id } => {
                    // The swing edge → a whoosh at the peer's current position. The
                    // local animator update already happened in `Connection::apply`.
                    if let Some(peer) = self
                        .net
                        .as_ref()
                        .and_then(|net| net.peers().find(|p| p.id() == id))
                    {
                        events.push(GameEvent::PeerSwing {
                            at: peer.sample(Instant::now()).pos.0,
                            peer: id,
                        });
                    }
                }
            }
        }
        disconnected
    }

    /// Handle one submitted console line (see [`run_line`](Self::run_line)). A command that edits
    /// settings (`/gfx`, the audio rows) goes through the one application path, is persisted and
    /// re-mixes the audio, only when something actually changed.
    fn submit_line(
        &mut self,
        line: String,
        eng: &mut Engine,
        settings: &mut Settings,
        sound: &mut SoundSystem,
        events: &mut Vec<GameEvent>,
        mods: &mut Mods,
    ) {
        if self.run_line(line, settings, events, mods) {
            self.apply_settings(eng, settings);
            settings.save();
            sound.set_mix(settings.mix_change());
        }
    }

    /// A submitted console line, short of the engine. A leading `/` is a command, except that in
    /// multiplayer `/op <secret>` is the server's operator login, sent as chat and not echoed; in
    /// multiplayer any other line is chat (a leading `!` sends it to global chat), while in
    /// singleplayer it stays a command. The first enabled mod that knows the command runs it, then
    /// the core follows up on the state it changed. Returns whether it changed the settings.
    fn run_line(&mut self, line: String, settings: &mut Settings, events: &mut Vec<GameEvent>, mods: &mut Mods) -> bool {
        if (line == "/op" || line.starts_with("/op "))
            && let Some(net) = &mut self.net
        {
            net.send_chat(chat::GLOBAL, &line);
            return false;
        }
        if !line.starts_with('/')
            && let Some(net) = &mut self.net
        {
            let (channel, text) = match line.strip_prefix('!') {
                Some(rest) => (chat::GLOBAL, rest.trim().to_string()),
                None => (chat::LOCAL, line),
            };
            if !text.is_empty() {
                // The server echoes chat back to us, so we don't print it here.
                net.send_chat(channel, &text);
            }
            return false;
        }
        self.console.echo(&line);
        let mut parts = line.strip_prefix('/').unwrap_or(line.as_str()).split_whitespace();
        let Some(cmd) = parts.next() else {
            return false;
        };
        let args: Vec<&str> = parts.collect();
        let commands: Vec<Command> = mods.commands().copied().collect();
        let before = settings.clone();
        let day_before = self.sky.clock.day();
        let day_len_before = self.sky.day_length;
        let pos_before = self.player.position;
        let mut ctx = CommandContext::new(&mut self.player, &mut self.world, settings, &mut self.sky);
        ctx.visuals = self.visual_mask;
        ctx.networked = self.net.is_some();
        ctx.commands = &commands;
        let out = mods.run_command(&mut ctx, cmd, &args);
        // The test cue is a fact for the sounds mod (it runs this frame even though the console
        // owns input).
        if ctx.voice_test {
            events.push(GameEvent::VoiceTest);
        }
        // Each output line already carries its role (output vs rejection): just show them.
        for line in out.unwrap_or_else(|| vec![console::unknown_command(cmd, &commands)]) {
            self.console.push(line);
        }
        // A `/time` change is shared: tell the server so every client's clock
        // follows (the server relays it and hands it to future joiners).
        if self.sky.clock.day() != day_before
            && let Some(net) = &mut self.net
        {
            net.send_set_time(self.sky.clock.day() as f32);
        }
        // The cycle LENGTH is server-owned in multiplayer: a local change
        // would silently desync every clock's advance rate.
        if self.sky.day_length != day_len_before && self.net.is_some() {
            self.sky.day_length = day_len_before;
            self.console
                .print("* day length is set by the server".to_string());
        }
        // A moved player is a position discontinuity: ordinary moves are envelope-
        // checked server-side, so report it as an explicit teleport (the
        // server may still snap us back if teleports are disabled) — and
        // stream out of band so the destination doesn't wait on `stream_hz`.
        if self.player.position != pos_before {
            self.force_stream = true;
            if let Some(net) = &mut self.net {
                net.send_teleport(self.player.position);
            }
        }
        *settings != before
    }

    /// Left click: with a tool, a reaction between the tool and the targeted block; with
    /// no tool, breaking the block into the inventory.
    fn primary_action(&mut self, mods: &mut Mods, events: &mut Vec<GameEvent>) {
        let Some(hit) = interact::raycast(&self.world, self.player.position, self.player.forward(), interact::REACH)
        else {
            return;
        };
        match mods.tool(&self.player) {
            Some(tool) => self.use_tool(tool, hit.block, mods, events),
            None => self.break_block(hit.block, mods, events),
        }
    }

    /// Use the held configuration `tool` on the block at `cell`: ONE operation of the law between
    /// the block (A, the world cell) and the tool (B). Elements move between them; the block may
    /// empty (the tool has absorbed it) and the tool may grow, shrink or change entirely. The
    /// changed cell wakes its contacts, so a disturbed block can start a cascade. On a server the
    /// law is the server's to run: the request goes out and [`Incoming::ToolResult`] applies it.
    fn use_tool(&mut self, tool: BlockId, cell: (i32, i32, i32), mods: &mut Mods, events: &mut Vec<GameEvent>) {
        let (x, y, z) = cell;
        let target = self.world.block_at(x, y, z);
        self.local_anim.on_action(WireAction::Swing);
        let at = sound_at(&self.world, x, y, z);
        events.push(GameEvent::Swing { at });
        if let Some(net) = &mut self.net {
            let spec = save::block_spec(self.world.registry(), tool);
            if let Some(req) = net.send_tool_use(x, y, z, spec.into()) {
                self.pending_tools.insert(req, tool);
            }
            net.send_swing();
            return;
        }
        match self.world.registry_mut().react(target, tool) {
            Some((_, new_cell, new_tool)) => {
                self.world.set_block(x, y, z, new_cell);
                self.world.note_cell_changed(x, y, z);
                self.finish_tool_change(tool, new_tool, target, new_cell, mods);
                events.push(GameEvent::ToolReacted { at, block: target });
                self.camera.fx.add_trauma(0.08);
            }
            None => mods.on_tool_used(ToolUse::NoReaction),
        }
    }

    /// One unit of `tool` became `new_tool` (the cell went `target` → `new_cell`): update the
    /// inventory and tell the mods.
    fn finish_tool_change(&mut self, tool: BlockId, new_tool: BlockId, target: BlockId, new_cell: BlockId, mods: &mut Mods) {
        if self.player.inventory.consume(tool, 1) && new_tool != AIR {
            self.player.inventory.add(new_tool, 1);
        }
        mods.on_tool_changed(tool, new_tool);
        let reg = self.world.registry();
        let before = reg.configuration(tool).len();
        let after = reg.configuration(new_tool).len();
        let outcome = if new_cell == AIR {
            ToolUse::CellDissolved { cell: target }
        } else if new_tool == AIR {
            ToolUse::ToolDissolved { cell: target }
        } else if after > before {
            ToolUse::Drew { cell: target }
        } else if after < before {
            ToolUse::Gave { cell: target }
        } else {
            ToolUse::Exchanged { cell: target }
        };
        mods.on_tool_used(outcome);
    }

    /// No tool: break the block at `cell` into the inventory.
    fn break_block(&mut self, cell: (i32, i32, i32), mods: &mut Mods, events: &mut Vec<GameEvent>) {
        let (x, y, z) = cell;
        let id = self.world.block_at(x, y, z);
        let at = sound_at(&self.world, x, y, z);
        events.push(GameEvent::Swing { at });
        events.push(GameEvent::BlockBroken { at, block: id, local: true });
        self.world.set_block(x, y, z, AIR);
        self.world.note_cell_changed(x, y, z);
        let overflow = !self.player.inventory.add(id, 1);
        mods.on_block_break(id, &self.world, overflow);
        self.camera.fx.add_trauma(0.15);
        self.local_anim.on_action(WireAction::Swing);
        // Tell the server (it validates and relays to everyone else). The
        // apply above is a PREDICTION for responsiveness: the ack rolls it
        // back — cell and loot both — if we lose the race for this cell.
        if let Some(net) = &mut self.net {
            if let Some(req) = net.send_edit(x, y, z, "air".into()) {
                self.pending_edits.insert(
                    req,
                    PendingEdit {
                        cell: (x, y, z),
                        prev: id,
                        kind: PendingKind::Break(id),
                    },
                );
                net.send_swing();
            } else {
                self.world.set_block(x, y, z, id);
                if !overflow {
                    self.player.inventory.revoke(id, 1);
                }
                mods.on_break_rejected(id);
            }
        }
    }

    /// Apply the block placements mods queued this tick, draining the buffer in
    /// place so its capacity is retained. A placement lands only in a
    /// non-obstacle cell (air, or a liquid it replaces) that doesn't overlap
    /// the player. Well-behaved mods (the crafting mod) ran an equivalent check
    /// before queueing — and before spending a block on it — so within one tick
    /// the two always agree; re-checking here is a cheap guard against a mod that
    /// queues without validating.
    fn apply_placements(
        &mut self,
        placements: &mut Vec<(i32, i32, i32, crate::block::BlockId)>,
        events: &mut Vec<GameEvent>,
        mods: &mut Mods,
    ) {
        for (x, y, z, id) in placements.drain(..) {
            // Overlap check in f64: at far coordinates an f32 cell centre
            // would land whole blocks away from the real cell.
            let cell = Aabb::new(
                DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5),
                DVec3::splat(0.5),
            );
            // Lands only in empty space clear of the player; a refused placement gives the
            // spent unit back.
            if self.world.is_solid(x, y, z) || cell.intersects(&self.player.aabb()) {
                self.player.inventory.add(id, 1);
                continue;
            }
            let prev = self.world.block_at(x, y, z);
            let at = sound_at(&self.world, x, y, z);
            events.push(GameEvent::Swing { at });
            events.push(GameEvent::BlockPlaced { at, block: id, local: true });
            self.world.set_block(x, y, z, id);
            self.world.note_cell_changed(x, y, z);
            self.local_anim.on_action(WireAction::Swing);
            // Tell the server in the same portable spec form saves use; it
            // validates and relays, exactly like breaking does with "air".
            // The spent unit is refunded if the server says no.
            if let Some(net) = &mut self.net {
                let spec = save::block_spec(self.world.registry(), id);
                if let Some(req) = net.send_edit(x, y, z, spec.into()) {
                    self.pending_edits.insert(
                        req,
                        PendingEdit {
                            cell: (x, y, z),
                            prev,
                            kind: PendingKind::Place(id),
                        },
                    );
                    net.send_swing();
                } else {
                    self.world.set_block(x, y, z, prev);
                    self.player.inventory.add(id, 1);
                    mods.on_place_rejected(id, &self.world);
                }
            }
        }
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
fn mod_ui_active(mod_logic: bool, mod_hud: bool, hud: HudMode) -> bool {
    mod_logic && mod_hud && hud.shows_mod_hud()
}

/// The world-space centre of a voxel cell (occurrence position).
fn cell_center(x: i32, y: i32, z: i32) -> DVec3 {
    DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5)
}

/// Where a sound at cell `(x, y, z)` plays: the cell's physical centre. A round world's storage cell
/// (x past a billion) sits where its chart embeds it, next to the listener.
fn sound_at(world: &World, x: i32, y: i32, z: i32) -> DVec3 {
    crate::space::atlas::embed_cell(world.atlases(), (x, y, z)).unwrap_or_else(|| cell_center(x, y, z))
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
    use std::cell::Cell;
    use std::rc::Rc;

    use crate::audio::GameEvent;
    use crate::input::intent::Chord;
    use crate::input::router::{Press, Router};
    use crate::modding::{Action, Command, CommandContext, Mod, ModContext, Mods};
    use crate::player::Player;
    use crate::render_config::RenderConfig;
    use crate::settings::Settings;
    use crate::ui::{Line, Role};
    use crate::world::World;
    use material::{Configuration, Element};
    use voxel_engine::{DVec3, Key, Vec2};

    /// A mod whose one command edits whatever its argument names, and that flies on the flight key.
    struct Probe;

    impl Mod for Probe {
        fn name(&self) -> &str {
            "Probe"
        }
        fn id(&self) -> &'static str {
            "probe"
        }
        fn commands(&self) -> &[Command] {
            &[Command { name: "probe", args: "<what>", help: "edit the game" }]
        }
        fn run_command(&mut self, ctx: &mut CommandContext<'_>, cmd: &str, args: &[&str]) -> Option<Vec<Line>> {
            if cmd != "probe" {
                return None;
            }
            match args {
                ["move"] => ctx.player.position.x += 10.0,
                ["fov"] => ctx.settings.fov += 5.0,
                ["intern"] => {
                    ctx.world.registry_mut().intern(&Configuration::single(Element::new([1, 2, 3, 4])));
                }
                ["voice"] => ctx.voice_test = true,
                _ => {}
            }
            Some(vec![Line::of(Role::Dim, format!("{} command(s)", ctx.commands.len()))])
        }
        fn on_toggle_fly(&mut self, player: &mut Player, _world: &World) -> bool {
            player.toggle_fly();
            true
        }
    }

    fn game() -> Game {
        let world = World::with_config_lazy(1, RenderConfig::default());
        Game::new(world, Player::new(DVec3::new(0.5, 80.0, 0.5)), "probe".into())
    }

    fn probe_mods() -> Mods {
        let mut mods = Mods::empty();
        mods.install(Box::new(Probe), true);
        mods
    }

    /// Run `line` as the console would; the scrollback's newest line and whether settings changed.
    fn run(game: &mut Game, mods: &mut Mods, settings: &mut Settings, line: &str) -> (String, Role, bool, Vec<GameEvent>) {
        let mut events = Vec::new();
        let changed = game.run_line(line.to_string(), settings, &mut events, mods);
        let last = game.console.last().expect("the console printed a line");
        let role = last.spans().next().expect("a line has a span").role;
        (last.text().to_string(), role, changed, events)
    }

    #[test]
    fn a_command_no_mod_handles_prints_the_hint() {
        let mut game = game();
        let mut settings = Settings::default();
        let (text, role, changed, _) = run(&mut game, &mut Mods::empty(), &mut settings, "/tp 1 2 3");
        assert_eq!(text, "unknown command 'tp' - commands come from mods such as the Developer Toolkit");
        assert_eq!(role, Role::Danger);
        assert!(!changed);
        assert_eq!(game.player.position, DVec3::new(0.5, 80.0, 0.5), "the base game has no /tp");
        let (text, ..) = run(&mut game, &mut probe_mods(), &mut settings, "/nope");
        assert!(text.starts_with("unknown command 'nope'"), "{text}");
        // With a mod that offers /help, the hint points there.
        struct Helper;
        impl Mod for Helper {
            fn name(&self) -> &str {
                "helper"
            }
            fn id(&self) -> &'static str {
                "helper"
            }
            fn commands(&self) -> &[Command] {
                &[Command { name: "help", args: "", help: "list commands" }]
            }
        }
        let mut mods = Mods::empty();
        mods.install(Box::new(Helper), true);
        let (text, ..) = run(&mut game, &mut mods, &mut settings, "/nope");
        assert_eq!(text, "unknown command 'nope' - type '/help'");
    }

    #[test]
    fn a_mod_command_edits_the_player_world_and_settings() {
        let (mut game, mut mods, mut settings) = (game(), probe_mods(), Settings::default());
        let blocks = game.world.registry().block_count();
        let (text, role, changed, _) = run(&mut game, &mut mods, &mut settings, "/probe intern");
        assert_eq!((text.as_str(), role), ("1 command(s)", Role::Dim), "the context lists every enabled command");
        assert!(!changed);
        assert_eq!(game.world.registry().block_count(), blocks + 1);
        let fov = settings.fov;
        let (.., changed, _) = run(&mut game, &mut mods, &mut settings, "probe fov");
        assert_eq!(settings.fov, fov + 5.0);
        assert!(changed, "changed settings are applied, saved and re-mixed by the caller");
        game.force_stream = false;
        let (.., changed, _) = run(&mut game, &mut mods, &mut settings, "/probe move");
        assert_eq!(game.player.position.x, 10.5);
        assert!(!changed);
        assert!(game.force_stream, "a moved player streams its destination at once (and is reported as a teleport)");
    }

    #[test]
    fn a_command_asking_for_the_voice_test_plays_the_cue() {
        let (mut game, mut mods, mut settings) = (game(), probe_mods(), Settings::default());
        let (.., events) = run(&mut game, &mut mods, &mut settings, "/probe");
        assert!(events.is_empty());
        let (.., events) = run(&mut game, &mut mods, &mut settings, "/probe voice");
        assert!(matches!(events.as_slice(), [GameEvent::VoiceTest]));
    }

    /// F is not a core toggle: the first mod that offers flight takes it on the frame of the press
    /// (mod logic on or off), once, and never while a detached camera holds the player.
    #[test]
    fn the_flight_key_goes_to_the_first_flight_mod() {
        let mut game = game();
        game.mod_logic = false;
        let input = FrameInput { toggle_fly: true, move_input: Some(Default::default()), ..FrameInput::default() };
        game.motion_phase(&input, 1.0 / 60.0);
        assert!(!game.player.flying(), "the core never toggles flight");
        assert!(!PendingModInput::capture(&input, true, true, None).any(), "F is not a cadence edge");

        game.fly_key(&input, false, &mut Mods::empty());
        assert!(!game.player.flying(), "without a mod that flies, F does nothing");
        game.fly_key(&input, true, &mut probe_mods());
        assert!(!game.player.flying(), "a detached camera does not fly the frozen player");
        // Two flight mods: the first takes the key, so it toggles once.
        let mut two = probe_mods();
        two.install(Box::new(Probe), true);
        game.fly_key(&input, false, &mut two);
        assert!(game.player.flying(), "the first flight mod toggles, on this frame, mod logic off");
        game.fly_key(&FrameInput::default(), false, &mut two);
        assert!(game.player.flying(), "no key, no toggle");
    }

    #[test]
    fn inert_frame_input_matches_default_and_carries_no_edges() {
        let inert = FrameInput::inert();
        assert_eq!(inert, FrameInput::default());
        assert!(inert.move_input.is_none());
        assert_eq!(inert.look_delta, Vec2::ZERO);
        assert!(!inert.is_text);
        assert!(!inert.open_console);
        assert!(!inert.open_chat);
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
    fn overlay_consume_flags_are_off_on_inert_input() {
        // overlay_phase returns Some only for text, Escape, or console-open edges.
        let inert = FrameInput::inert();
        assert!(!inert.is_text && !inert.g_escape && !inert.open_console && !inert.open_chat);
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

    #[test]
    fn game_gates_physics_on_spawn_ready() {
        let world = World::with_config_lazy(1, RenderConfig::default());
        let pos = DVec3::new(0.5, 80.0, 0.5);
        let mut game = Game::new(world, Player::new(pos), "gate".to_string());
        game.world_mut().prepare_around(pos);
        assert!(
            !game.world().spawn_ready(),
            "Game::update must not run motion/interact until the slab lands"
        );
        let before = game.player().position;
        if game.world().spawn_ready() {
            game.player_mut().position.y -= 1.0;
        }
        assert_eq!(game.player().position, before);
    }

    /// Quiet-frame micro-benchmark: the three remaining fixed costs at
    /// Minimum/Fast. Reports ns/call for the idle predicates vs the work they
    /// skip. Ignored: a timing run, not a correctness gate.
    #[test]
    #[ignore]
    fn quiet_frame_fixed_costs() {
        use crate::audio::{AudioService, SoundSystem};
        use crate::console::Console;
        use crate::input::router::Router;
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
        let mut console = Console::new();
        let listener = crate::audio::Listener {
            pos,
            yaw: 0.0,
            pitch: 0.0,
            frame: glam::DQuat::IDENTITY,
        };
        audio.finish(&mut sound, &world, listener, 1.0 / 60.0, &mut console);

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

    #[test]
    fn quiet_minimum_frame_allocates_nothing_and_reads_the_clock_once() {
        use crate::alloc_count;
        use crate::audio::{AudioService, SoundSystem};
        use crate::input::router::Router;
        use crate::settings::Settings;
        use crate::ui::HudMode;

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
            game.tick_quiet(DT, &mut router, &mut sound, &mut audio, &symbols, &settings, &mut mods);
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
            game.tick_quiet(DT, &mut router, &mut sound, &mut audio, &symbols, &settings, &mut mods);
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
        use crate::settings::Settings;

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
            game.tick_quiet(DT, &mut router, &mut sound, &mut audio, &symbols, &settings, &mut mods);
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
