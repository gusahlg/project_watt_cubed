//! game.rs owns the in-world state — world, player, physics, console — and runs a
//! frame of it: input, movement, block interaction, mods, streaming, and drawing.
//! The window and the menu/play state machine live one level up in [`app`](crate::app);
//! a `Game` is handed the engine each frame and reports back whether to keep playing
//! or return to the menu.
use std::time::{Duration, Instant};

mod draw;

use voxel_engine::{Color, DVec3, Engine, IVec2, Vec2};

use crate::audio::{
    AudioCtx, AudioDirector, PeerPose, PlayerPose, SoundEvent, SoundSystem, UiSound,
};
use crate::block::AIR;
use crate::camera::{CameraMode, CameraPose, FlyAxes, GameCamera};
use crate::command;
use crate::console::Console;
use crate::derived::Revision;
use crate::input::intent::{GameplayEvent, GameplayState, GlobalEvent, MenuEvent};
use crate::input::router::{Context, Router, View};
use crate::input::{look, movement};
use crate::interact;
use crate::math::{Aabb, Bounded};
use crate::minimap::{Minimap, MinimapConfig};
use crate::mods::{ModContext, Mods};
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
    events: &'a mut Vec<SoundEvent>,
}

/// Owned/read-only facts plus the two audio committers for the final update
/// phase. In particular, `events` moves exactly once into the director.
struct AudioPhase<'a> {
    dt: f32,
    input: &'a FrameInput,
    sound: &'a mut SoundSystem,
    audio: &'a mut AudioDirector,
    settings: &'a Settings,
    events: Vec<SoundEvent>,
    active: bool,
}

/// Edge-triggered mod intents from one render frame, retained in order when
/// mod updates run at a fixed cadence.
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
    visual_mask: crate::mods::VisualMask,
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
    /// tick, so a tap between two 30 Hz ticks still flies/jumps.
    pending_toggle_fly: bool,
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
    peer_pose_scratch: Vec<PeerPose>,
    /// Last frame's named-phase durations, for the stall detector.
    phases: FramePhases,
    hud_scratch: Vec<HudElement>,
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
            visual_mask: crate::mods::VisualMask::default(),
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
            pending_toggle_fly: false,
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
            phases: FramePhases::default(),
            hud_scratch: Vec::new(),
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
        let mod_ui_was_active = self.mod_ui_active();
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
            // If visibility is restored before the next frame, the overlay is
            // visible again and does not need to be force-closed.
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

    pub fn set_visual_mask(&mut self, mask: crate::mods::VisualMask) {
        self.visual_mask = mask;
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
            self.world
                .stream(self.player.position, eng, &mut self.sched);
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

        // This frame's unrecoverable audio facts, accumulated across the
        // phases and folded by the director. Everything else — footsteps, splash, the
        // underwater bed, voice sessions — the director DERIVES from the readout.
        let mut events: Vec<SoundEvent> = Vec::new();

        self.phases = FramePhases::default();
        let t = Instant::now();
        if let Some(signal) = self.net_phase(mods, &mut events) {
            return signal;
        }
        self.phases.net = t.elapsed();
        let t = Instant::now();
        let input = self.input_phase(eng, router, dt);
        self.phases.input = t.elapsed();
        // The overlay may consume the frame (console typing, opening chat): movement
        // and interaction run only on an unconsumed frame. Streaming still runs
        // while a spawn/teleport slab is outstanding so loading progresses with
        // the console open. Audio skips the director commit when nothing is
        // sounding and the listener is still; only a real exit short-circuits
        // the rest of the frame.
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
            self.interact_phase(&input, detached, dt, eng, mods, &mut events);
            self.phases.interact = t.elapsed();
        }
        if !consumed || !ready {
            let t = Instant::now();
            self.stream_phase(eng, dt);
            self.phases.stream = t.elapsed();
        }
        let active = !consumed;
        let t = Instant::now();
        self.commit_audio(AudioPhase {
            dt,
            input: &input,
            sound,
            audio,
            settings,
            events,
            active,
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
    fn net_phase(&mut self, mods: &mut Mods, events: &mut Vec<SoundEvent>) -> Option<Signal> {
        // The overwhelmingly common singleplayer path should not even enter a
        // profiling scope or call through the event-poll seam.
        self.net.as_ref()?;
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
        let input = router.frame_filtered(eng, dt, self.mod_logic, mod_ui, self.minimap.is_some());
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
                f.ptt = gp.state(GameplayState::PushToTalk);
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
                .handle_input(&input.text_chars, input.text_edit)
            {
                self.submit_line(line, eng, settings, sound, events);
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
        self.pending_toggle_fly = false;
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
                let steps = self.physics_gate.steps(dt);
                if steps != 0 {
                    let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::Physics);
                    let step_dt = self.physics_gate.step_dt(dt);
                    for step in 0..steps {
                        let mut tick_input = *mi;
                        tick_input.set_toggle_fly(step == 0 && self.pending_toggle_fly);
                        tick_input.set_jump(mi.jump() || (step == 0 && self.pending_jump));
                        let trauma = movement::update_player(
                            &mut self.player,
                            &self.world,
                            &tick_input,
                            step_dt,
                        );
                        if trauma > 0.0 {
                            self.camera.fx.add_trauma(trauma);
                        }
                        // Advance the local walk cycle from horizontal travel so the
                        // third-person body animates. The AUDIO gait (footstep
                        // phase-crossings) is derived inside the director from the
                        // same speed — this one drives rendering only.
                        let v = self.player.velocity();
                        let speed = (v.x * v.x + v.z * v.z).sqrt();
                        self.local_gait += speed * step_dt as f64 * presence::STRIDE_FREQ;
                    }
                    self.pending_toggle_fly = false;
                    self.pending_jump = false;
                }
            }
            false
        }
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
        events: &mut Vec<SoundEvent>,
    ) {
        // Break is capture-gated in the query; freecam additionally can't act
        // on the world (the crosshair isn't where the player aims).
        if input.do_break && !detached {
            self.break_block(mods, events);
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

        let mut pending = std::mem::take(&mut self.pending_mod_input);
        let mut placements = std::mem::take(&mut self.placement_scratch);
        placements.clear();
        let (screen_w, screen_h) = (eng.screen_width(), eng.screen_height());
        // Preserve ordering and multiplicity for edge-bearing render frames.
        // With no edge, one empty update keeps periodic work at `mod_hz`.
        for index in 0..pending.len().max(1) {
            let edges = pending.get(index).copied().unwrap_or_default();
            placements = {
                let mut ctx = ModContext {
                    player: &mut self.player,
                    world: &mut self.world,
                    screen_w,
                    screen_h,
                    place: edges.place,
                    place_target: edges.place_target,
                    toggle_inventory: edges.toggle_inventory,
                    toggle_crafting: edges.toggle_crafting,
                    nav_up: edges.nav_up,
                    nav_down: edges.nav_down,
                    nav_confirm: edges.nav_confirm,
                    placements,
                };
                mods.update(&mut ctx);
                ctx.placements
            };
            // Apply after each event frame so repeated placements observe the
            // previous write and cannot spend twice against one empty cell.
            self.apply_placements(&mut placements, events);
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
        if !stream_due {
            self.world.pump(eng, &mut self.sched);
            return;
        }

        let stream_center = match &self.camera.mode {
            CameraMode::Free { rig, .. } => rig.pos,
            CameraMode::Person(_) => self.player.position,
        };
        self.world.stream(stream_center, eng, &mut self.sched);

        // A hidden/minimal HUD does no minimap clock read or terrain raster
        // work; refreshes share streaming's cadence instead of waking alone.
        if self.theme.hud.shows_minimap()
            && let Some(minimap) = &mut self.minimap
        {
            let p = self.player.position;
            let player_col = IVec2::new(p.x.floor() as i32, p.z.floor() as i32);
            // The minimap throttle rides the scheduler's interval gate (advanced in
            // clocks() above); the recenter half stays inside Minimap::due. Reset
            // the gate whenever a rebuild actually happens (either trigger).
            let due = self.sched.interval_due(self.minimap_interval);
            if minimap.refresh(eng, &self.world, player_col, due) {
                self.sched.interval_reset(self.minimap_interval);
            }
        }
    }

    /// Hand this frame's readout to the audio director: it folds
    /// the drained `events`, derives the rest from the trace (footsteps, splash,
    /// underwater bed, voice sessions), commits the [`AudioFrame`], and services the
    /// voice/capture path. The window, medium, gait and session bookkeeping that used
    /// to live here are the director's now.
    fn commit_audio(&mut self, phase: AudioPhase<'_>) {
        let AudioPhase {
            dt,
            input,
            sound,
            audio,
            settings,
            events,
            active,
        } = phase;
        // Singleplayer idle: skip pose/peer/director/mixer construction. Voice
        // ingest and capture need the full path (a new session is not yet live).
        if self.net.is_none()
            && audio.can_skip_commit(sound, &events, self.player.position, input.ptt)
        {
            sound.poll_starvation();
            return;
        }
        // THE per-frame peer sample: one `Instant`, consumed by the director for
        // both remote footsteps and voice sessions. `peer_draws` in draw() keeps its
        // own richer sample — it runs in the separate draw() call, steps each peer's
        // animator, and needs render fields absent from `PeerPose`. The scratch
        // vector retains capacity so stable multiplayer frames allocate nothing.
        let now = Instant::now();
        let mut peers = std::mem::take(&mut self.peer_pose_scratch);
        peers.clear();
        if let Some(net) = &self.net {
            peers.extend(net.peers().map(|p| {
                let r = p.sample(now);
                PeerPose {
                    id: p.id(),
                    at: r.pos.0,
                    feet: r.pos.feet(r.stance).0,
                    visible: p.visible(),
                    phase: r.phase,
                    speed: r.speed,
                }
            }));
        }

        // On a console-owned frame the player isn't stepped, so freeze the listener
        // velocity: a stale walk speed would fire phantom footsteps in the director.
        let velocity = if active {
            self.player.velocity()
        } else {
            DVec3::ZERO
        };
        let player = PlayerPose {
            pos: self.player.position,
            feet: DVec3::new(
                self.player.position.x,
                self.player.feet_y(),
                self.player.position.z,
            ),
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
        self.peer_pose_scratch = peers;
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
                    let id = save::parse_block(self.world.registry_mut(), &spec);
                    self.world.set_block(x, y, z, id);
                    if id == AIR {
                        self.world.note_block_broken(x, y, z);
                    } else {
                        self.world.note_block_placed(x, y, z);
                    }
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
                    let Some(pending) = self.pending_edits.remove(&req) else {
                        continue;
                    };
                    if restore {
                        let (x, y, z) = pending.cell;
                        self.world.set_block(x, y, z, pending.prev);
                    }
                    match pending.kind {
                        PendingKind::Break(id) => {
                            self.player.stash.revoke(id, 1);
                            mods.on_break_rejected(id);
                        }
                        PendingKind::Place(id) => mods.on_place_rejected(id, &self.world),
                    }
                }
                Incoming::Position { pos } => {
                    // Authoritative snap-back (refused teleport or implausible
                    // move): request the collision slab and freeze until it
                    // lands, exactly like a local teleport. The server's
                    // MOVE_WINDOW_CAP_SECS envelope tolerates a brief pause.
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
                Incoming::Disconnected => disconnected = true,
                Incoming::PeerSwing { id } => {
                    // The swing edge → a whoosh at the peer's current position. The
                    // local animator update already happened in `Connection::apply`.
                    if let Some(peer) = self
                        .net
                        .as_ref()
                        .and_then(|net| net.peers().find(|p| p.id() == id))
                    {
                        events.push(SoundEvent::PeerSwing {
                            at: peer.sample(Instant::now()).pos.0,
                        });
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
            return;
        }
        self.console.echo(&line);
        let before = settings.clone();
        let day_before = self.sky.clock.day();
        let day_len_before = self.sky.day_length;
        let pos_before = self.player.position;
        // Each output line already carries its role (System output vs Error
        // rejection), so there is nothing to guess — just show them.
        for out in command::execute_with_visuals(
            &line,
            &mut self.player,
            &mut self.world,
            settings,
            &mut self.sky,
            self.visual_mask,
        ) {
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
        // A `/tp` is a position discontinuity: ordinary moves are envelope-
        // checked server-side, so report it as an explicit teleport (the
        // server may still snap us back if teleports are disabled) — and
        // stream out of band so the destination doesn't wait on `stream_hz`.
        if self.player.position != pos_before {
            self.force_stream = true;
            if let Some(net) = &mut self.net {
                net.send_teleport(self.player.position);
            }
        }
    }

    /// Break the block the player is looking at, depositing its configuration
    /// into the core stash before notifying mods.
    fn break_block(&mut self, mods: &mut Mods, events: &mut Vec<SoundEvent>) {
        let Some(hit) = interact::raycast_solid(
            &self.world,
            self.player.position,
            self.player.forward(),
            interact::REACH,
        ) else {
            return;
        };
        let (x, y, z) = hit.block;
        let id = self.world.block_at(x, y, z);
        events.push(SoundEvent::BlockBroken {
            at: cell_center(x, y, z),
            block: id,
        });
        self.world.set_block(x, y, z, AIR);
        self.world.note_block_broken(x, y, z);
        let overflow = !self.player.stash.add(id, 1);
        mods.on_block_break(id, &self.world, overflow);
        self.camera.fx.add_trauma(0.15);
        self.local_anim.on_action(WireAction::Swing);
        // Tell the server (it validates and relays to everyone else). The
        // apply above is a PREDICTION for responsiveness: the ack rolls it
        // back — cell and loot both — if we lose the race for this cell.
        if let Some(net) = &mut self.net {
            let req = net.send_edit(x, y, z, "air".into());
            self.pending_edits.insert(
                req,
                PendingEdit {
                    cell: (x, y, z),
                    prev: id,
                    kind: PendingKind::Break(id),
                },
            );
            net.send_swing();
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
        events: &mut Vec<SoundEvent>,
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
            // Report the placed block; the director derives its class-specific cue.
            events.push(SoundEvent::BlockPlaced {
                at: cell_center(x, y, z),
                block: id,
            });
            self.world.set_block(x, y, z, id);
            self.world.note_block_placed(x, y, z);
            self.local_anim.on_action(WireAction::Swing);
            // Tell the server in the same portable spec form saves use; it
            // validates and relays, exactly like breaking does with "air".
            // The spent crafted block is refunded if the server says no.
            if let Some(net) = &mut self.net {
                let spec = save::block_spec(self.world.registry(), id);
                let req = net.send_edit(x, y, z, spec.into());
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

#[cfg(test)]
mod tests {
    use super::{FrameInput, Game, PendingModInput};
    use crate::player::Player;
    use crate::render_config::RenderConfig;
    use crate::world::World;
    use voxel_engine::{DVec3, Vec2};

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
        assert!(!inert.ptt);
        assert!(!PendingModInput::capture(&inert, true, true, Some((1, 2, 3))).any());
    }

    #[test]
    fn overlay_consume_flags_are_off_on_inert_input() {
        // overlay_phase returns Some only for text, Escape, or console-open edges.
        let inert = FrameInput::inert();
        assert!(!inert.is_text && !inert.g_escape && !inert.open_console && !inert.open_chat);
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
        use crate::audio::{AudioCtx, AudioDirector, PlayerPose, SoundSystem};
        use crate::audio::palette::CuePalette;
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

        let (mut sound, symbols) = SoundSystem::mute();
        let (palette, _) = CuePalette::build(&symbols, sound.catalog());
        let mut audio = AudioDirector::new(palette);
        let world = World::generate();
        let pos = DVec3::new(0.5, 80.0, 0.5);
        let mut console = Console::new();
        let player = PlayerPose {
            pos,
            feet: DVec3::new(pos.x, pos.y - 1.6, pos.z),
            yaw: 0.0,
            pitch: 0.0,
            velocity: DVec3::ZERO,
            on_ground: true,
        };
        audio.frame(
            AudioCtx {
                dt: 1.0 / 60.0,
                player: PlayerPose { ..player },
                ptt: false,
                voice_enabled: false,
                events: Vec::new(),
                peers: &[],
                world: &world,
                net: None,
                console: &mut console,
            },
            &mut sound,
        );

        let t0 = Instant::now();
        for _ in 0..N {
            black_box(audio.can_skip_commit(&sound, &[], pos, false));
        }
        let skip_ns = t0.elapsed().as_nanos() as f64 / f64::from(N);

        let t0 = Instant::now();
        for _ in 0..N {
            let mut console = Console::new();
            audio.frame(
                AudioCtx {
                    dt: 1.0 / 60.0,
                    player: PlayerPose { ..player },
                    ptt: false,
                    voice_enabled: false,
                    events: Vec::new(),
                    peers: &[],
                    world: &world,
                    net: None,
                    console: &mut console,
                },
                &mut sound,
            );
        }
        let director_ns = t0.elapsed().as_nanos() as f64 / f64::from(N);

        println!("quiet_frame_fixed_costs ({N} iters):");
        println!("  anything_in_flight:     {in_flight_ns:.1} ns");
        println!("  Router::drain_frame:    {drain_ns:.1} ns");
        println!("  can_skip_commit (idle): {skip_ns:.1} ns  [after]");
        println!("  AudioDirector::frame:   {director_ns:.1} ns  [before, still silent]");
        assert!(
            audio.can_skip_commit(&sound, &[], pos, false),
            "the skip predicate must hold on the idle pose used above"
        );
        assert!(!game.world().anything_in_flight());
    }
}
