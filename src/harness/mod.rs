//! Golden-shot harness — the executable acceptance criteria for the voxel engine.
//!
//! Two halves:
//!
//! - The PURE detectors — [`diff`] and [`sky_hole_count`] — are ordinary
//!   functions over a decoded [`Screenshot`]. They have no engine dependency
//!   and their unit tests (below) are this package's acceptance contract.
//! - The LIVE half — [`check`] and [`time_to_first_full_render`] — needs a
//!   scripted `Game` and deterministic frame capture, which are other packages'
//!   work in flight. Every path that would touch the engine funnels through the
//!   single [`capture`] seam, which degrades with a clear error until that work
//!   lands. Nothing here panics on the degraded path.
//!

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{Duration, Instant};

use voxel_engine::skeleton::Screenshot;
use voxel_engine::{Camera3D, Color, DVec3};

use crate::camera::ViewPose;
use crate::game::Game;
use crate::mods::Mods;
use crate::settings::Settings;

/// The one seed every golden shot and metric uses.
pub const GOLDEN_SEED: u64 = 0xC0FFEE;

/// The day fraction a scripted `Game`'s clock starts at — the `SkyClock::default`
/// value (`sky/clock.rs`). The pre-existing blessed shots pin `GoldenShot::day`
/// to THIS, making the runner's `set_day` a provable no-op for them (their look
/// must not change). `scripted_default_day_matches_clock` guards the equality so
/// a change to the clock default can't silently drift the goldens.
pub const SCRIPTED_DEFAULT_DAY: f64 = 0.3;

/// Blessed goldens live here, one PNG per [`GoldenShot::name`].
pub const GOLDEN_DIR: &str = "tests/golden";

/// Clear/background key under [`DebugView::TerrainKey`] — anything showing this
/// through the terrain silhouette is a hole.
pub const SKY_KEY: Color = Color::rgb(255, 0, 255);
/// Flat fill every terrain surface renders as under [`DebugView::TerrainKey`].
pub const TERRAIN_KEY: Color = Color::rgb(0, 255, 0);

/// Per-channel absolute delta a pixel must exceed to count as "changed" — the
/// 8-bit noise floor. Dithered/tonemapped output wobbles by a couple of codes
/// frame to frame even when nothing moved.
const NOISE_FLOOR: u8 = 4;

/// Per-channel tolerance for classifying a captured pixel as one of the two
/// keys. The keys are maximally separated (0 vs 255 in every channel), so any
/// tolerance well below 128 keeps them unambiguous; 24 absorbs MSAA-resolve and
/// tonemap drift on the flat key fills without ever letting a terrain pixel read
/// as sky or vice-versa. Synthetic test images sit exactly on the keys, so the
/// tolerance is a no-op there.
const KEY_TOL: u8 = 24;

// ============================================================================
// Camera pose / golden shot
// ============================================================================

/// A camera pose mirroring `Player` (position `DVec3`, yaw/pitch `f32`); the
/// harness derives `Camera3D` through the SAME `ViewPose::camera3d` path the
/// game uses.
#[derive(Clone, Copy, Debug)]
pub struct CameraPose {
    pub pos: DVec3,
    pub yaw: f32,
    pub pitch: f32,
}

impl CameraPose {
    /// Delegates camera derivation to the game's ViewPose path to avoid reimplementing projection math.
    pub fn camera(&self, fovy: f32) -> Camera3D {
        ViewPose { eye: self.pos, yaw: self.yaw, pitch: self.pitch, roll: 0.0, fovy }.camera3d()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GoldenShot {
    pub seed: u64,
    pub cam: CameraPose,
    pub name: &'static str,
    /// Day/night clock fraction the runner pins before capture (0.5 = noon,
    /// 0.0 = midnight); see [`SkyClock`](crate::sky::SkyClock). The two
    /// pre-existing blessed goldens are pinned to [`SCRIPTED_DEFAULT_DAY`] —
    /// the clock's scripted default — so `set_day(day)` is a provable no-op for
    /// them and their captured look must not change. Only the new shots use
    /// other days.
    pub day: f64,
    /// One-shot world edit applied after `teleport`, before streaming (e.g. the
    /// cave shot carves its room). `None` for shots that read the plain seed.
    pub setup: Option<fn(&mut Game)>,
}

/// What the app renders for a capture.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum DebugView {
    #[default]
    Normal,
    /// ALL terrain flat [`TERRAIN_KEY`], sky/fog passes disabled, clear color
    /// [`SKY_KEY`]. The sky-hole detector's input.
    TerrainKey,
}

// ============================================================================
// Pure detectors
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiffStats {
    /// Largest per-channel absolute delta anywhere in the image.
    pub max_channel_delta: u8,
    /// Percent of pixels with any channel delta > [`NOISE_FLOOR`].
    pub pct_changed: f32,
}

/// Compare two same-sized screenshots. Pure.
///
/// Mismatched dimensions can't be diffed pixel-for-pixel; rather than panic we
/// report a total mismatch so callers surface it as a plain failure.
pub fn diff(a: &Screenshot, b: &Screenshot) -> DiffStats {
    if a.width != b.width || a.height != b.height || a.rgba.len() != b.rgba.len() {
        return DiffStats { max_channel_delta: u8::MAX, pct_changed: 100.0 };
    }
    let total_px = (a.width as usize) * (a.height as usize);
    if total_px == 0 {
        return DiffStats { max_channel_delta: 0, pct_changed: 0.0 };
    }

    let mut max_delta = 0u8;
    let mut changed = 0usize;
    for (pa, pb) in a.rgba.chunks_exact(4).zip(b.rgba.chunks_exact(4)) {
        let mut px_changed = false;
        for c in 0..4 {
            let d = pa[c].abs_diff(pb[c]);
            if d > max_delta {
                max_delta = d;
            }
            if d > NOISE_FLOOR {
                px_changed = true;
            }
        }
        if px_changed {
            changed += 1;
        }
    }

    DiffStats {
        max_channel_delta: max_delta,
        pct_changed: (changed as f32 / total_px as f32) * 100.0,
    }
}

/// True if every pixel of `shot` is byte-identical (a flat fill). A real
/// rendered scene is never uniform; a uniform capture means the frame was
/// presented without being drawn. Used to refuse blessing a degenerate golden
/// (the black-screenshot failure class). An empty image counts as uniform.
pub fn is_uniform(shot: &Screenshot) -> bool {
    let mut px = shot.rgba.chunks_exact(4);
    match px.next() {
        None => true,
        Some(first) => px.all(|p| p == first),
    }
}

/// True if `px` (RGBA slice, alpha ignored) is within [`KEY_TOL`] of `key`.
fn is_key(px: &[u8], key: Color) -> bool {
    px[0].abs_diff(key.r) <= KEY_TOL
        && px[1].abs_diff(key.g) <= KEY_TOL
        && px[2].abs_diff(key.b) <= KEY_TOL
}

/// Executable Phase-D criterion: on a [`DebugView::TerrainKey`] capture,
/// for each pixel column find the topmost [`TERRAIN_KEY`] pixel; count
/// [`SKY_KEY`] pixels BELOW it (sky showing through the terrain silhouette = a
/// hole). Pure.
///
/// Rows run top-to-bottom (present orientation, per `Screenshot`), so "topmost"
/// is the smallest `y` and "below" is a larger `y`. A column with no terrain
/// pixel has no silhouette and so contributes no holes.
pub fn sky_hole_count(shot: &Screenshot) -> u32 {
    let w = shot.width as usize;
    let h = shot.height as usize;
    if w == 0 || h == 0 {
        return 0;
    }
    let pixel = |x: usize, y: usize| -> &[u8] {
        let i = (y * w + x) * 4;
        &shot.rgba[i..i + 4]
    };

    let mut holes = 0u32;
    for x in 0..w {
        // Topmost terrain pixel in this column, if any.
        let mut top = None;
        for y in 0..h {
            if is_key(pixel(x, y), TERRAIN_KEY) {
                top = Some(y);
                break;
            }
        }
        if let Some(top) = top {
            for y in (top + 1)..h {
                if is_key(pixel(x, y), SKY_KEY) {
                    holes += 1;
                }
            }
        }
    }
    holes
}

// ============================================================================
// Executable acceptance criteria
// ============================================================================

/// A phase of the rewrite ladder, ordered. `Display`/[`Phase::marker`] yields
/// the literal that appears inside `PROVISIONAL(<marker>)` comments. The `Ord`
/// derive follows declaration order (`PreA < A < … < E`), which is what makes
/// the cumulative sweep (`p <= through`) meaningful.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Phase {
    PreA,
    A,
    B,
    C,
    D,
    E,
}

impl Phase {
    /// Every phase in ascending order — the cumulative sweep filters this by
    /// `<= through`.
    pub const ALL: [Phase; 6] = [Phase::PreA, Phase::A, Phase::B, Phase::C, Phase::D, Phase::E];

    /// The literal inside the marker: "pre-A", "A", "B", "C", "D", "E".
    pub fn marker(self) -> &'static str {
        match self {
            Phase::PreA => "pre-A",
            Phase::A => "A",
            Phase::B => "B",
            Phase::C => "C",
            Phase::D => "D",
            Phase::E => "E",
        }
    }

    /// Parse "pre-A" | "A" | … | "E" (the `golden --phase` argument).
    pub fn parse(s: &str) -> Option<Phase> {
        Phase::ALL.iter().copied().find(|p| p.marker() == s)
    }
}

impl std::fmt::Display for Phase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.marker())
    }
}

/// An acceptance criterion that EXECUTES (strings don't).
pub enum Criterion {
    /// Capture `shot`, compare against its blessed golden.
    ImageMatch { shot: GoldenShot, max_pct_changed: f32 },
    /// Capture `shot` under [`DebugView::TerrainKey`]; `sky_hole_count ≤ max`.
    SkyHoleCount { shot: GoldenShot, max: u32 },
    /// `time_to_first_full_render(seed) ≤ max`.
    EntryTime { seed: u64, max: Duration },
    /// Mean frame time at `shot` over 120 frames ≤ `max_ms`.
    FrameTime { shot: GoldenShot, max_ms: f32 },
    /// Walks `src/` + `voxel-engine/src` with `std::fs`; fails if ANY
    /// `PROVISIONAL(p)` marker survives for a phase `p <= through`. The
    /// sweep is cumulative: closing phase E must also clear every pre-A..D
    /// marker left behind, not just the E ones.
    NoProvisional { through: Phase },
}

pub struct Acceptance {
    pub phase: Phase,
    pub criteria: Vec<Criterion>,
}

#[derive(Clone, Debug)]
pub struct Failure {
    pub what: String,
    pub detail: String,
}

/// Fixed offscreen-capture window config: a real winit window still
/// backs it (the engine has no headless path), but size/pacing are pinned so
/// captures are reproducible across machines.
fn scripted_config() -> voxel_engine::Config {
    voxel_engine::Config {
        title: "golden-harness".into(),
        width: 1280,
        height: 720,
        vsync: false,
        // Engine-side lanes for every capture (blocklight ON for cave_interior's
        // emitter; a no-op for the emitter-free shots). Process-global — one
        // event loop drives all stages — so it matches `golden()`'s engine flags.
        flags: crate::render_config::RenderConfig::golden().engine_flags(),
        ..Default::default()
    }
}

/// A single unit of engine work in the golden sequence: build a scripted
/// `Game` at `seed`, teleport to `cam`, wait for `World::entry_complete`, then
/// perform `kind`. Every stage runs inside ONE shared `voxel_engine::run` —
/// winit permits a single `EventLoop` per process — so the whole shot list is
/// driven by one event loop and the pure evaluation happens after it returns.
struct Stage {
    seed: u64,
    cam: Option<CameraPose>,
    /// Clock fraction pinned before capture (mirrors [`GoldenShot::day`]);
    /// entry-time stages use [`SCRIPTED_DEFAULT_DAY`] so their `set_day` is a
    /// no-op.
    day: f64,
    view: DebugView,
    /// One-shot world edit run once after teleport (mirrors [`GoldenShot::setup`]).
    setup: Option<fn(&mut Game)>,
    /// The render lanes this stage's world is built with — part of the shot's
    /// identity, so e.g. `tile_boundary` is *defined* as tiles-on and cannot be
    /// captured with the wrong far-field filler. Threaded into [`Game::scripted`].
    render: crate::render_config::RenderConfig,
    kind: StageKind,
}

enum StageKind {
    /// Record elapsed from `Game` creation to the first `entry_complete` frame
    /// under `seed`. One frame, then done.
    EntryTime,
    /// Screenshot the first ready frame straight to `path` (bless → golden,
    /// else a scratch file the evaluator decodes back).
    Capture { path: PathBuf },
    /// Accumulate frame time over [`SAMPLE_FRAMES`] ready frames; record the mean (ms) under `name`.
    FrameSample { name: String },
}

/// Sample width for `FrameSample`'s mean (120 frames).
const SAMPLE_FRAMES: u32 = 120;

/// Wall-clock hold AFTER `entry_complete` before a `Capture` screenshot.
/// `entry_complete` only means the CPU worklists (mesh/light) have drained — but
/// with frames in flight the freshly-lit mesh built on that frame hasn't reached
/// the offscreen `screenshot_to` re-presents until a frame or two later (a carved,
/// emitter-lit cave is black on the entry frame and lights up ~2 frames on). This
/// hold also lets the render-thread exposure smoother (`k=exp(-dt·1.25)`, τ=0.8s)
/// walk off its default seed to the metered target, so the blessed PNG is exposed
/// as it will be in-game.
///
/// Gated on WALL-CLOCK, not a frame count: capture pacing is uncapped
/// (`target_fps=0`, vsync off), so a fixed frame count is an unknown,
/// machine-dependent slice of the decay. ~6τ drives the residual
/// `exp(-1.25·t)` below the 8-bit floor (`exp(-6.25)≈0.0019 < 1/256`), making
/// the blessed exposure reproducible across machines.
const CAPTURE_SETTLE: Duration = Duration::from_secs(5);

/// Everything the single event-loop pass produces for the post-loop pure
/// evaluation: per-`path` capture outcomes, per-seed entry times, per-shot mean
/// frame times. No engine handle escapes here.
#[derive(Default)]
struct Outcomes {
    captures: std::collections::HashMap<PathBuf, Result<(), String>>,
    entry_times: std::collections::HashMap<u64, Duration>,
    frame_times: std::collections::HashMap<String, f32>,
}

/// Scratch capture path for a shot/view (decoded back by the evaluator).
fn scratch_path(name: &str, tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("watt_golden_{name}_{tag}.png"))
}

/// The Normal-view capture path for an `ImageMatch` shot: the golden itself
/// when blessing (captured straight in), else a scratch file to diff.
fn image_capture_path(name: &str, bless: bool) -> PathBuf {
    if bless { golden_path(name) } else { scratch_path(name, "normal") }
}

/// Derive the engine work-list from the acceptance set (pure). One `EntryTime`
/// stage per distinct seed any metric needs (always incl. `GOLDEN_SEED` for the
/// printed number), plus one capture/sample stage per image/sky/frame-time
/// criterion. Ordering is irrelevant — each stage rebuilds its own `Game`.
fn plan_stages(acc: &Acceptance, bless: bool) -> Vec<Stage> {
    use std::collections::BTreeSet;
    let mut entry_seeds = BTreeSet::new();
    entry_seeds.insert(GOLDEN_SEED);
    for c in &acc.criteria {
        if let Criterion::EntryTime { seed, .. } = c {
            entry_seeds.insert(*seed);
        }
    }
    let mut stages: Vec<Stage> = entry_seeds
        .into_iter()
        .map(|seed| Stage {
            seed,
            cam: None,
            day: SCRIPTED_DEFAULT_DAY,
            view: DebugView::Normal,
            setup: None,
            render: crate::render_config::RenderConfig::golden(),
            kind: StageKind::EntryTime,
        })
        .collect();
    for c in &acc.criteria {
        match c {
            Criterion::ImageMatch { shot, .. } => stages.push(Stage {
                seed: shot.seed,
                cam: Some(shot.cam),
                day: shot.day,
                view: DebugView::Normal,
                setup: shot.setup,
                render: crate::render_config::RenderConfig::golden(),
                kind: StageKind::Capture { path: image_capture_path(shot.name, bless) },
            }),
            Criterion::SkyHoleCount { shot, .. } => stages.push(Stage {
                seed: shot.seed,
                cam: Some(shot.cam),
                day: shot.day,
                view: DebugView::TerrainKey,
                setup: shot.setup,
                render: crate::render_config::RenderConfig::golden(),
                kind: StageKind::Capture { path: scratch_path(shot.name, "terrainkey") },
            }),
            Criterion::FrameTime { shot, .. } => stages.push(Stage {
                seed: shot.seed,
                cam: Some(shot.cam),
                day: shot.day,
                view: DebugView::Normal,
                setup: shot.setup,
                render: crate::render_config::RenderConfig::golden(),
                kind: StageKind::FrameSample { name: shot.name.to_string() },
            }),
            Criterion::EntryTime { .. } | Criterion::NoProvisional { .. } => {}
        }
    }
    stages
}

/// Drive the WHOLE stage list inside ONE `voxel_engine::run` — the single
/// `EventLoop` this process is allowed. A state machine walks the list: build a
/// scripted `Game` for the current stage, tick it (`Game::update`, the same
/// call the real app makes) until `entry_complete`, run the stage's action,
/// then advance — rebuilding the `Game` for the next seed/pose. The callback
/// returns `false` only after the last stage; that `false` is `run`'s existing
/// exit affordance (no engine change needed).
///
/// No second projection/camera path here: `teleport` puts the pose on the
/// real `Player` the game already carries.
fn execute(stages: Vec<Stage>) -> Outcomes {
    if stages.is_empty() {
        return Outcomes::default();
    }
    let mut mods = Mods::with_defaults();
    let mut settings = Settings::default();
    // Update requires a router even though scripted games consume no live input.
    let mut router = crate::input::router::Router::new();
    let outcomes = Rc::new(RefCell::new(Outcomes::default()));
    let sink = Rc::clone(&outcomes);

    // Watchdog (per stage): a genuinely slow stream keeps changing its
    // stuck-clause signature (= progress); a true STALL freezes it. Fail loud
    // on a stall or a hard ceiling — naming the stuck clause — instead of
    // hanging. Progress is logged live so a slow stream is visibly moving.
    const HARD_TIMEOUT: Duration = Duration::from_secs(180);
    const STALL_TIMEOUT: Duration = Duration::from_secs(8);

    let mut idx = 0usize;
    let mut game: Option<Game> = None;
    let mut stage_start = Instant::now();
    let mut last_sig = String::new();
    let mut last_change = Instant::now();
    let mut frame: u64 = 0;
    let mut settle_start: Option<Instant> = None;
    let mut sample_total_ms = 0f32;
    let mut sample_n = 0u32;

    voxel_engine::run(scripted_config(), move |eng| {
        let stage = &stages[idx];
        // First frame of a stage: (re)build its scripted game and reset the
        // per-stage watchdog / sample accumulators.
        let g = game.get_or_insert_with(|| {
            let mut g = Game::scripted(stage.seed, stage.render);
            g.set_debug_view(stage.view);
            if let Some(cam) = stage.cam {
                g.teleport(cam);
            }
            // Pin the shot's lighting, then run its one-shot world edit (cave
            // carve) — both before the first `update`/stream so the edit is in
            // the overlay when the carved chunks generate.
            g.set_day(stage.day);
            if let Some(setup) = stage.setup {
                setup(&mut g);
            }
            stage_start = Instant::now();
            last_sig = String::new();
            last_change = Instant::now();
            frame = 0;
            settle_start = None;
            sample_total_ms = 0.0;
            sample_n = 0;
            g
        });
        // Advance ONE frame exactly as the app does: update THEN draw. Drawing
        // is part of advancing a frame — never a per-stage action — so no stage
        // (present or future) can capture/sample a frame that was never drawn.
        // `Game::draw` fills + finishes the `Frame`, populating the engine's
        // `last_lists` that `screenshot_to` re-presents; without it every capture
        // was a blank (uniform black) frame that passed ImageMatch trivially.
        // Streaming frames draw too, matching the real app (which renders while
        // chunks load) and warming pipelines before the captured frame.
        g.update(eng, &mut router, &mut mods, &mut settings);
        // Shake 0: captures must be deterministic (no live trauma exists in the
        // scripted path anyway).
        g.draw(eng, &mut mods, settings.fov, 0.0);

        if !g.world().entry_complete() {
            frame += 1;
            // Sample a few times a second (entry_debug scans the box — not per frame).
            if frame.is_multiple_of(20) {
                let sig = g.world().entry_debug();
                if sig != last_sig {
                    eprintln!("harness: streaming… {sig}");
                    last_sig = sig;
                    last_change = Instant::now();
                }
            }
            let stalled = last_change.elapsed() > STALL_TIMEOUT;
            if stalled || stage_start.elapsed() > HARD_TIMEOUT {
                eprintln!(
                    "harness: entry {} — {}",
                    if stalled { "STALLED (no progress in 8s)" } else { "timed out (180s)" },
                    g.world().entry_debug()
                );
                std::process::exit(2);
            }
            return true;
        }

        // A `Capture` holds (drawing) past `entry_complete` so the freshly-settled
        // mesh/light AND the render-thread exposure smoother reach the captured
        // target — see `CAPTURE_SETTLE`. Other kinds record at the entry frame
        // (EntryTime would be inflated by a wait) or sample across frames
        // themselves (FrameSample).
        if let StageKind::Capture { .. } = &stage.kind {
            let started = *settle_start.get_or_insert_with(Instant::now);
            if started.elapsed() < CAPTURE_SETTLE {
                return true;
            }
        }

        let done = match &stage.kind {
            StageKind::EntryTime => {
                sink.borrow_mut().entry_times.insert(stage.seed, stage_start.elapsed());
                true
            }
            StageKind::Capture { path } => {
                // The frame is already drawn (single advance point above); capture
                // re-presents `last_lists` straight to `path`. A fresh checkout has
                // no golden dir yet (goldens ship unblessed), so create the parent
                // here — the first-ever `bless` must not fail on ENOENT.
                let r = path
                    .parent()
                    .map_or(Ok(()), std::fs::create_dir_all)
                    .and_then(|()| voxel_engine::skeleton::screenshot_to(eng, path))
                    .map_err(|e| e.to_string());
                sink.borrow_mut().captures.insert(path.clone(), r);
                true
            }
            StageKind::FrameSample { name } => {
                sample_total_ms += eng.frame_time() * 1000.0;
                sample_n += 1;
                let full = sample_n >= SAMPLE_FRAMES;
                if full {
                    let mean = sample_total_ms / sample_n as f32;
                    sink.borrow_mut().frame_times.insert(name.clone(), mean);
                }
                full
            }
        };

        if done {
            idx += 1;
            game = None; // rebuild for the next stage's seed/pose
            if idx >= stages.len() {
                return false; // last stage complete — exit the single event loop
            }
        }
        true
    });

    Rc::try_unwrap(outcomes).map(RefCell::into_inner).unwrap_or_default()
}

/// Path of the blessed golden PNG for a shot name.
fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(GOLDEN_DIR).join(format!("{name}.png"))
}

/// The golden run's product: the printed golden-seed entry time and the
/// aggregated pass/fail — both from the SAME single event-loop pass.
pub struct Report {
    pub golden_entry_time: Duration,
    pub result: Result<(), Vec<Failure>>,
}

/// THE single entry point `golden` calls: plan the stages, drive them ALL in
/// one `voxel_engine::run`, then evaluate the (pure) criteria over the captured
/// artifacts. Exactly one `run` per process.
pub fn run_acceptance(acc: &Acceptance, bless: bool) -> Report {
    let outcomes = execute(plan_stages(acc, bless));
    let golden_entry_time = outcomes.entry_times.get(&GOLDEN_SEED).copied().unwrap_or_default();
    let result = evaluate(acc, bless, &outcomes);
    Report { golden_entry_time, result }
}

/// Frozen-contract wrapper (`check`): run the whole
/// acceptance set and return only the pass/fail. `golden` uses
/// [`run_acceptance`] directly so it gets the entry-time number from the SAME
/// pass; standalone callers can use this. One `run` per call.
pub fn check(a: &Acceptance, bless: bool) -> Result<(), Vec<Failure>> {
    run_acceptance(a, bless).result
}

/// Evaluate every criterion over the captured artifacts (PURE — no engine).
/// Aggregates all failures rather than stopping at the first.
fn evaluate(acc: &Acceptance, bless: bool, out: &Outcomes) -> Result<(), Vec<Failure>> {
    let mut failures = Vec::new();
    for c in &acc.criteria {
        if let Err(f) = eval_criterion(c, bless, out) {
            failures.push(f);
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

/// Look up a capture's write outcome; a missing entry means the planner and
/// evaluator disagree on the path (a bug), surfaced as a failure not a panic.
fn capture_result(out: &Outcomes, path: &Path, what: &str) -> Result<(), Failure> {
    match out.captures.get(path) {
        Some(Ok(())) => Ok(()),
        Some(Err(e)) => {
            Err(Failure { what: what.to_string(), detail: format!("screenshot_to: {e}") })
        }
        None => Err(Failure {
            what: what.to_string(),
            detail: format!("no capture recorded for {}", path.display()),
        }),
    }
}

fn eval_criterion(c: &Criterion, bless: bool, out: &Outcomes) -> Result<(), Failure> {
    match c {
        Criterion::ImageMatch { shot, max_pct_changed } => {
            let path = image_capture_path(shot.name, bless);
            capture_result(out, &path, &format!("image_match {}", shot.name))?;
            if bless {
                // A blessed golden IS the captured frame written straight to
                // `path`. There is nothing to COMPARE, but there is something to
                // VALIDATE: a uniform frame (every pixel identical) is never a
                // real scene — it is the signature of a capture-before-draw
                // plumbing failure. Blessing one silently poisons the whole
                // harness (black-vs-black ImageMatch passes trivially), which is
                // exactly the regression this guard makes impossible.
                let golden = voxel_engine::skeleton::load_png(&path).map_err(|e| Failure {
                    what: format!("image_match {}", shot.name),
                    detail: format!("load blessed {}: {e}", path.display()),
                })?;
                if is_uniform(&golden) {
                    return Err(Failure {
                        what: format!("image_match {}", shot.name),
                        detail: format!(
                            "refusing to bless a UNIFORM frame ({}×{}, every pixel identical) — \
                             the scene was not drawn before capture",
                            golden.width, golden.height
                        ),
                    });
                }
                return Ok(());
            }
            let got = voxel_engine::skeleton::load_png(&path).map_err(|e| Failure {
                what: format!("image_match {}", shot.name),
                detail: format!("load capture {}: {e}", path.display()),
            })?;
            let golden = golden_path(shot.name);
            let want = voxel_engine::skeleton::load_png(&golden).map_err(|e| Failure {
                what: format!("image_match {}", shot.name),
                detail: format!("load golden {}: {e}", golden.display()),
            })?;
            let stats = diff(&got, &want);
            if stats.pct_changed > *max_pct_changed {
                return Err(Failure {
                    what: format!("image_match {}", shot.name),
                    detail: format!(
                        "{:.3}% changed (max {:.3}%), peak channel delta {}",
                        stats.pct_changed, max_pct_changed, stats.max_channel_delta
                    ),
                });
            }
            Ok(())
        }

        Criterion::SkyHoleCount { shot, max } => {
            let path = scratch_path(shot.name, "terrainkey");
            capture_result(out, &path, &format!("sky_holes {}", shot.name))?;
            let got = voxel_engine::skeleton::load_png(&path).map_err(|e| Failure {
                what: format!("sky_holes {}", shot.name),
                detail: format!("load capture {}: {e}", path.display()),
            })?;
            let n = sky_hole_count(&got);
            if n > *max {
                return Err(Failure {
                    what: format!("sky_holes {}", shot.name),
                    detail: format!("{n} sky-holes (max {max})"),
                });
            }
            Ok(())
        }

        Criterion::EntryTime { seed, max } => {
            let elapsed = out.entry_times.get(seed).copied().ok_or_else(|| Failure {
                what: format!("entry_time seed={seed:#x}"),
                detail: "no entry time recorded".into(),
            })?;
            if elapsed > *max {
                return Err(Failure {
                    what: format!("entry_time seed={seed:#x}"),
                    detail: format!("{elapsed:?} (max {max:?})"),
                });
            }
            Ok(())
        }

        Criterion::FrameTime { shot, max_ms } => {
            let mean_ms = out.frame_times.get(shot.name).copied().ok_or_else(|| Failure {
                what: format!("frame_time {}", shot.name),
                detail: "no frame-time sample recorded".into(),
            })?;
            if mean_ms > *max_ms {
                return Err(Failure {
                    what: format!("frame_time {}", shot.name),
                    detail: format!(
                        "{mean_ms:.3} ms mean over {SAMPLE_FRAMES} frames (max {max_ms})"
                    ),
                });
            }
            Ok(())
        }

        Criterion::NoProvisional { through } => no_provisional(*through),
    }
}

/// Walk `src/` and `voxel-engine/src` and fail if ANY `PROVISIONAL(p)` marker
/// survives for a phase `p <= through` — cumulative, so a late phase also
/// clears every earlier marker. Fully live (pure filesystem, no engine).
fn no_provisional(through: Phase) -> Result<(), Failure> {
    let targets: Vec<Phase> = Phase::ALL.iter().copied().filter(|p| *p <= through).collect();
    let mut hits: Vec<(Phase, String)> = Vec::new();
    for root in ["src", "voxel-engine/src"] {
        scan_dir(std::path::Path::new(root), &targets, &mut hits);
    }
    if hits.is_empty() {
        return Ok(());
    }
    // Actionable detail: per-phase counts plus a handful of file:line examples,
    // ordered by phase so the earliest un-cleared markers read first.
    let mut detail = String::new();
    for p in &targets {
        let count = hits.iter().filter(|(hp, _)| hp == p).count();
        if count == 0 {
            continue;
        }
        let examples: Vec<&str> =
            hits.iter().filter(|(hp, _)| hp == p).take(4).map(|(_, loc)| loc.as_str()).collect();
        detail.push_str(&format!("PROVISIONAL({}): {count} [{}]; ", p.marker(), examples.join(", ")));
    }
    Err(Failure {
        what: format!("no_provisional(through {through})"),
        detail: format!("{} surviving marker(s): {}", hits.len(), detail.trim_end()),
    })
}

/// True for the two files this harness owns — `src/harness/mod.rs` and
/// `src/bin/golden.rs` — which both spell `PROVISIONAL(...)` in their own doc
/// comments (they DEFINE/illustrate the marker syntax). Skip them so the sweep
/// never trips over its own contract text.
fn is_owned_self_file(path: &std::path::Path) -> bool {
    let matches = |dir: &str, file: &str| {
        path.file_name().is_some_and(|n| n == file)
            && path.parent().is_some_and(|p| p.ends_with(dir))
    };
    matches("harness", "mod.rs") || matches("bin", "golden.rs")
}

fn scan_dir(dir: &std::path::Path, targets: &[Phase], hits: &mut Vec<(Phase, String)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan_dir(&path, targets, hits);
        } else if path.extension().is_some_and(|e| e == "rs") {
            if is_owned_self_file(&path) {
                continue;
            }
            if let Ok(text) = std::fs::read_to_string(&path) {
                for (i, line) in text.lines().enumerate() {
                    scan_line(line, targets, &path, i + 1, hits);
                }
            }
        }
    }
}

/// Record every `PROVISIONAL(<marker>)` on `line` whose phase is in `targets`.
/// Anchors on the FULL token: extract the text between `PROVISIONAL(` and the
/// next `)` and match it EXACTLY, so `PROVISIONAL(A)` never matches inside
/// `PROVISIONAL(pre-A)`.
fn scan_line(
    line: &str,
    targets: &[Phase],
    path: &std::path::Path,
    lineno: usize,
    hits: &mut Vec<(Phase, String)>,
) {
    const OPEN: &str = "PROVISIONAL(";
    let mut rest = line;
    while let Some(start) = rest.find(OPEN) {
        let after = &rest[start + OPEN.len()..];
        let Some(end) = after.find(')') else { break };
        let marker = &after[..end];
        if let Some(p) = targets.iter().copied().find(|p| p.marker() == marker) {
            hits.push((p, format!("{}:{}", path.display(), lineno)));
        }
        rest = &after[end + 1..];
    }
}

/// Elapsed from world creation until the first frame where
/// `World::entry_complete()` is true (no camera pose needed — entry completion
/// is a property of streaming around the player's spawn, not of any particular shot).
pub fn time_to_first_full_render(seed: u64) -> Duration {
    // Standalone/contract path: drives its own single-`EntryTime`-stage `run`.
    // `golden` does NOT call this — it reads the number from `run_acceptance`'s
    // single pass — so the golden process still opens exactly one event loop.
    let out = execute(vec![Stage {
        seed,
        cam: None,
        day: SCRIPTED_DEFAULT_DAY,
        view: DebugView::Normal,
        setup: None,
        render: crate::render_config::RenderConfig::golden(),
        kind: StageKind::EntryTime,
    }]);
    out.entry_times.get(&seed).copied().unwrap_or_default()
}

/// The canonical golden-shot list (fixed seed, a spread of poses). Extend as
/// phases add shots; every name maps to one PNG in [`GOLDEN_DIR`].
///
/// The first three shots are blessed (or bless candidates) and pinned to
/// [`SCRIPTED_DEFAULT_DAY`] so day control cannot change their look. The last
/// four exercise the new day/setup lighting cases; their poses are pending
/// first-capture review and they are NOT blessed yet — `bless` must run once
/// before they can pass an ImageMatch.
pub fn golden_shots() -> Vec<GoldenShot> {
    vec![
        GoldenShot {
            seed: GOLDEN_SEED,
            cam: CameraPose { pos: DVec3::new(0.0, 80.0, 0.0), yaw: 0.0, pitch: -0.2 },
            name: "spawn_forward",
            day: SCRIPTED_DEFAULT_DAY,
            setup: None,
        },
        GoldenShot {
            seed: GOLDEN_SEED,
            // Looking out at the horizon — the shot where LOD/skin holes show.
            cam: CameraPose { pos: DVec3::new(0.0, 90.0, 0.0), yaw: 2.35, pitch: 0.0 },
            name: "horizon",
            day: SCRIPTED_DEFAULT_DAY,
            setup: None,
        },
        GoldenShot {
            seed: GOLDEN_SEED,
            // Looking down-forward so the full-res box edge — view_radius = 6
            // chunks = 96 m — sits mid-frame: this is where the chunk→far-tile
            // handoff must be seamless.
            cam: CameraPose { pos: DVec3::new(0.0, 110.0, 0.0), yaw: 0.0, pitch: -0.35 },
            name: "tile_boundary",
            day: SCRIPTED_DEFAULT_DAY,
            setup: None,
        },
        GoldenShot {
            seed: GOLDEN_SEED,
            // PROVISIONAL(pre-A): pose reviewed at first capture. Primary-day pose (mirrors spawn_forward) at midnight.
            cam: CameraPose { pos: DVec3::new(0.0, 80.0, 0.0), yaw: 0.0, pitch: -0.2 },
            name: "night_field",
            day: 0.0,
            setup: None,
        },
        GoldenShot {
            seed: GOLDEN_SEED,
            // Pose reviewed at first capture (2026-07-11): the terrain surface at
            // origin is ~64, so the original y=68 sat ABOVE ground — an open pit,
            // not a cave. Placed at the CENTRE of chunk (0,2,0) (CHUNK_SIZE=16) so
            // the whole 10×6×10 carved room + stone shell fits inside ONE chunk —
            // no chunk boundary runs through it, so no cross-chunk light re-settle
            // can strand the emitter's blocklight. Genuinely underground (surface
            // ~64); `carve_cave` seals the shell dark, facing the +x emitter.
            cam: CameraPose { pos: DVec3::new(8.0, 40.0, 8.0), yaw: 0.0, pitch: 0.0 },
            name: "cave_interior",
            day: 0.5,
            setup: Some(carve_cave),
        },
        GoldenShot {
            seed: GOLDEN_SEED,
            // PROVISIONAL(pre-A): pose reviewed at first capture. Golden pos
            // +30 m, grazing down (~ −8°), yaw along the day-0.35 sun azimuth so
            // the 64/256 m splits + fade band sit mid-frame.
            cam: CameraPose { pos: DVec3::new(0.0, 110.0, 0.0), yaw: 0.24, pitch: -0.14 },
            name: "shadow_boundary",
            day: 0.35,
            setup: None,
        },
        GoldenShot {
            seed: GOLDEN_SEED,
            // PROVISIONAL(pre-A): pose reviewed at first capture. Near-surface,
            // near-level (~ −1°), yaw toward the day-0.30 sun azimuth so pure sky
            // sits over fog→1 terrain. `pos.y` is a stand-in for surface_y(0,0)+2,
            // to be re-pinned at first capture.
            cam: CameraPose { pos: DVec3::new(0.0, 66.0, 0.0), yaw: 0.20, pitch: -0.017 },
            name: "horizon_fog_vs_sky",
            day: 0.30,
            setup: None,
        },
        GoldenShot {
            seed: GOLDEN_SEED,
            // PROVISIONAL(pre-A): pose reviewed at first capture. Water shot: a
            // low camera (y≈72, a few blocks over the ~64 water surface) grazing
            // ACROSS the lakes that fill the origin basin (see the blessed
            // spawn_forward/tile_boundary captures) at a moderate down angle so a
            // wide sheet of water fills mid-frame — the grazing view maximizes
            // fresnel/sky reflection. Yaw is pinned along the day-0.3 sun azimuth
            // (as shadow_boundary reasons) so the sun's specular GLINT lands on the
            // water. Pinned day ⇒ pinned `anim` phase ⇒ deterministic wave
            // geometry. This is the golden that protects every WATER_* tunable.
            cam: CameraPose { pos: DVec3::new(0.0, 72.0, 0.0), yaw: 0.24, pitch: -0.12 },
            name: "water",
            day: SCRIPTED_DEFAULT_DAY,
            setup: None,
        },
    ]
}

/// `cave_interior`'s setup ([`GoldenShot::setup`]): carve an 8×4×8 air room
/// around the (already-teleported) camera, wrap it in a one-block stone shell,
/// and seat one emitter block on the +x wall it faces (yaw 0). The shell seals
/// the room so it stays dark even where the terrain's cave field would otherwise
/// open it to skylight — the whole point of the shot is an enclosed emitter-lit
/// interior (the `L_cave` exposure anchor). Centring on the live player position
/// keeps the room in step with the shot's pose with no duplicated coordinates.
/// The edits land in the world overlay before streaming, so the carved chunks
/// generate already-carved.
fn carve_cave(game: &mut Game) {
    let registry = game.world().registry();
    // The one built-in light-emitting block (`El::Lumin`, block/registry).
    let emitter = registry.id_by_name("LuminVein").expect("LuminVein is a built-in block");
    let stone = registry.id_by_name("Stone").expect("Stone is a built-in block");
    let air = crate::block::registry::AIR;
    let p = game.player().position;
    let (cx, cy, cz) = (p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32);
    let world = game.world_mut();
    // Interior air pocket [cx-4,cx+3] × [cy-2,cy+1] × [cz-4,cz+3]; the enclosing
    // shell is the same box grown by one block on every side. Setting the shell
    // solid guarantees a sealed, lightless room regardless of natural terrain.
    for x in (cx - 5)..=(cx + 4) {
        for y in (cy - 3)..=(cy + 2) {
            for z in (cz - 5)..=(cz + 4) {
                let interior = (cx - 4..cx + 4).contains(&x)
                    && (cy - 2..cy + 2).contains(&y)
                    && (cz - 4..cz + 4).contains(&z);
                world.set_block(x, y, z, if interior { air } else { stone });
            }
        }
    }
    // Emitter at eye level on the wall the camera looks at (+x, yaw 0).
    world.set_block(cx + 3, cy, cz, emitter);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a solid-color RGBA screenshot.
    fn filled(w: u32, h: u32, c: Color) -> Screenshot {
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            rgba.extend_from_slice(&[c.r, c.g, c.b, c.a]);
        }
        Screenshot { width: w, height: h, rgba }
    }

    fn set_px(shot: &mut Screenshot, x: u32, y: u32, c: Color) {
        let i = ((y * shot.width + x) * 4) as usize;
        shot.rgba[i..i + 4].copy_from_slice(&[c.r, c.g, c.b, c.a]);
    }

    #[test]
    fn uniform_frame_is_rejected_but_one_pixel_saves_it() {
        // The bless guard: a flat fill (the black-screenshot failure) is
        // uniform; a single differing pixel (any real scene) is not.
        let flat = filled(8, 6, Color::rgb(0, 0, 0));
        assert!(is_uniform(&flat), "an all-black frame is uniform → unblessable");

        let mut scene = a_clone(&flat);
        set_px(&mut scene, 3, 2, Color::rgb(1, 0, 0));
        assert!(!is_uniform(&scene), "one differing pixel makes it a real frame");
    }

    #[test]
    fn diff_reflexive_zero() {
        let img = filled(8, 6, Color::rgb(30, 60, 90));
        let d = diff(&img, &img);
        assert_eq!(d.max_channel_delta, 0);
        assert_eq!(d.pct_changed, 0.0);
    }

    #[test]
    fn diff_detects_single_pixel_change() {
        let a = filled(10, 10, Color::rgb(0, 0, 0));
        let mut b = a_clone(&a);
        set_px(&mut b, 3, 4, Color::rgb(200, 0, 0));
        let d = diff(&a, &b);
        assert_eq!(d.max_channel_delta, 200);
        // Exactly one of 100 pixels changed.
        assert!((d.pct_changed - 1.0).abs() < 1e-4, "pct_changed = {}", d.pct_changed);
    }

    #[test]
    fn diff_ignores_noise_floor() {
        let a = filled(4, 4, Color::rgb(100, 100, 100));
        let mut b = a_clone(&a);
        // A single-code wobble is below NOISE_FLOOR: registered in the peak
        // delta but not counted as a changed pixel.
        set_px(&mut b, 0, 0, Color::rgb(103, 100, 100));
        let d = diff(&a, &b);
        assert_eq!(d.max_channel_delta, 3);
        assert_eq!(d.pct_changed, 0.0);
    }

    #[test]
    fn diff_mismatched_dims_is_total() {
        let a = filled(4, 4, Color::rgb(0, 0, 0));
        let b = filled(5, 4, Color::rgb(0, 0, 0));
        let d = diff(&a, &b);
        assert_eq!(d.max_channel_delta, u8::MAX);
        assert_eq!(d.pct_changed, 100.0);
    }

    fn a_clone(s: &Screenshot) -> Screenshot {
        Screenshot { width: s.width, height: s.height, rgba: s.rgba.clone() }
    }

    #[test]
    fn sky_hole_one_below_silhouette() {
        // A column of sky with a terrain pixel partway down, then a sky pixel
        // BELOW it: exactly one hole.
        let mut img = filled(1, 5, SKY_KEY);
        set_px(&mut img, 0, 2, TERRAIN_KEY); // silhouette top at y=2
        set_px(&mut img, 0, 3, SKY_KEY); // hole below
        set_px(&mut img, 0, 4, TERRAIN_KEY); // solid again, not a hole
        assert_eq!(sky_hole_count(&img), 1);
    }

    #[test]
    fn sky_above_silhouette_is_not_a_hole() {
        // Sky above the topmost terrain pixel is the ordinary skyline, never a
        // hole (nothing below the silhouette here).
        let mut img = filled(1, 5, SKY_KEY);
        set_px(&mut img, 0, 3, TERRAIN_KEY);
        set_px(&mut img, 0, 4, TERRAIN_KEY);
        assert_eq!(sky_hole_count(&img), 0);
    }

    #[test]
    fn sky_hole_counts_per_column() {
        // Two columns, each with a terrain top and two sky pixels beneath.
        let mut img = filled(2, 4, TERRAIN_KEY);
        for x in 0..2 {
            set_px(&mut img, x, 0, TERRAIN_KEY);
            set_px(&mut img, x, 2, SKY_KEY);
            set_px(&mut img, x, 3, SKY_KEY);
        }
        assert_eq!(sky_hole_count(&img), 4);
    }

    #[test]
    fn sky_hole_column_without_terrain_is_zero() {
        // No terrain silhouette anywhere → no holes, even though it's all sky.
        let img = filled(3, 3, SKY_KEY);
        assert_eq!(sky_hole_count(&img), 0);
    }

    #[test]
    fn key_tolerance_absorbs_small_drift() {
        // A terrain pixel drifted a few codes still classifies as terrain, and a
        // drifted sky pixel below it still counts as a hole.
        let mut img = filled(1, 2, TERRAIN_KEY);
        set_px(&mut img, 0, 0, Color::rgb(8, 248, 8)); // ~TERRAIN_KEY silhouette
        set_px(&mut img, 0, 1, Color::rgb(248, 8, 248)); // ~SKY_KEY, below
        assert_eq!(sky_hole_count(&img), 1);
    }

    // Live-capture criteria (ImageMatch/SkyHoleCount/EntryTime/FrameTime) now
    // drive a real windowed `Engine` via `drive_ready_frames` — no longer
    // unit-testable headlessly here. `golden.rs` (`cargo run --bin golden`)
    // is their acceptance run; `NoProvisional` and the pure detectors above
    // stay covered by these `cargo test`-safe unit tests.

    #[test]
    fn scripted_default_day_matches_clock() {
        // The pre-existing shots pin `day` to SCRIPTED_DEFAULT_DAY so `set_day`
        // is a no-op for them; that only holds while it equals the scripted
        // clock's start. Guards note-1's by-construction no-op.
        assert_eq!(crate::sky::SkyClock::default().day(), SCRIPTED_DEFAULT_DAY);
    }

    #[test]
    fn phase_order_and_roundtrip() {
        // Declaration order is the ladder order, and marker/parse round-trips.
        assert!(Phase::PreA < Phase::A && Phase::A < Phase::E);
        for p in Phase::ALL {
            assert_eq!(Phase::parse(p.marker()), Some(p));
        }
        assert_eq!(Phase::parse("nonsense"), None);
    }

    #[test]
    fn scan_line_matches_exact_marker_only() {
        // The `through = B` sweep targets pre-A, A, B — but NOT C.
        let targets = [Phase::PreA, Phase::A, Phase::B];
        let path = std::path::Path::new("x.rs");
        let mut hits = Vec::new();
        scan_line("// PROVISIONAL(A): a", &targets, path, 1, &mut hits);
        scan_line("// PROVISIONAL(pre-A): b", &targets, path, 2, &mut hits);
        scan_line("// PROVISIONAL(C): out of range", &targets, path, 3, &mut hits);
        let phases: Vec<Phase> = hits.iter().map(|(p, _)| *p).collect();
        assert_eq!(phases, vec![Phase::A, Phase::PreA]);
    }

    #[test]
    fn scan_line_a_does_not_match_inside_pre_a() {
        // The `A` literal must anchor on the full token: `PROVISIONAL(pre-A)`
        // yields ONLY a pre-A hit, never a spurious A hit.
        let targets = Phase::ALL;
        let path = std::path::Path::new("x.rs");
        let mut hits = Vec::new();
        scan_line("PROVISIONAL(pre-A)", &targets, path, 1, &mut hits);
        assert_eq!(hits.iter().map(|(p, _)| *p).collect::<Vec<_>>(), vec![Phase::PreA]);
    }
}
