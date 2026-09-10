//! Golden-shot harness — the executable acceptance criteria for the voxel engine.
//!
//! The pure detectors — [`diff`] and [`sky_hole_count`] — operate on decoded
//! [`Screenshot`]s. The live runner drives scripted [`Game`] instances through
//! one event loop, captures deterministic frames, and evaluates those detectors
//! together with entry- and frame-time criteria.
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
        ViewPose {
            eye: self.pos,
            yaw: self.yaw,
            pitch: self.pitch,
            roll: 0.0,
            fovy,
        }
        .camera3d()
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
        return DiffStats {
            max_channel_delta: u8::MAX,
            pct_changed: 100.0,
        };
    }
    let total_px = (a.width as usize) * (a.height as usize);
    if total_px == 0 {
        return DiffStats {
            max_channel_delta: 0,
            pct_changed: 0.0,
        };
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

/// On a [`DebugView::TerrainKey`] capture, for each pixel column find the
/// topmost [`TERRAIN_KEY`] pixel; count
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

/// An acceptance criterion that EXECUTES (strings don't).
pub enum Criterion {
    /// Capture `shot`, compare against its blessed golden.
    ImageMatch {
        shot: GoldenShot,
        max_pct_changed: f32,
    },
    /// Capture `shot` under [`DebugView::TerrainKey`]; `sky_hole_count ≤ max`.
    SkyHoleCount { shot: GoldenShot, max: u32 },
    /// Time from world creation to the first fully streamed frame is at most
    /// `max`.
    EntryTime { seed: u64, max: Duration },
    /// Mean frame time at `shot` over 120 frames ≤ `max_ms`.
    FrameTime { shot: GoldenShot, max_ms: f32 },
}

pub struct Acceptance {
    pub criteria: Vec<Criterion>,
}

#[derive(Clone, Debug)]
pub struct Failure {
    pub what: String,
    pub detail: String,
}

/// Fixed offscreen-capture window config: a real winit window still backs the
/// surface (the engine has no fully headless path). Size and pacing are pinned
/// for reproducibility across machines; the dimension check in `evaluate`
/// reports any WM-forced resize as "window resized", rather than pixel drift.
fn scripted_config() -> voxel_engine::Config {
    voxel_engine::Config {
        title: "golden-harness".into(),
        width: 1280,
        height: 720,
        vsync: false,
        resizable: false,
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
    /// Horizontal view radius override (chunks), applied straight through
    /// `World::set_view_distances` — the scripted `Game::update` never applies
    /// `Settings`, so this is the only way a stage can stream a non-default
    /// radius. `None` keeps the world's default.
    radius: Option<i32>,
    /// The render lanes this stage's world is built with — part of the shot's
    /// identity, so e.g. `tile_boundary` is *defined* as tiles-on and cannot be
    /// captured with the wrong far-field filler. Threaded into [`Game::scripted`].
    render: crate::render_config::RenderConfig,
    kind: StageKind,
}

impl Stage {
    /// A stage at the world's default streaming radius — every site but the
    /// stress-flight builder wants this.
    fn new(
        seed: u64,
        cam: Option<CameraPose>,
        day: f64,
        view: DebugView,
        setup: Option<fn(&mut Game)>,
        render: crate::render_config::RenderConfig,
        kind: StageKind,
    ) -> Self {
        Stage {
            seed,
            cam,
            day,
            view,
            setup,
            radius: None,
            render,
            kind,
        }
    }
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
    /// After entry completes, fly the player at `speed_mps` along +X for
    /// `secs`, then stop and measure how long streaming takes to fully
    /// re-settle — the fast-flight lag scenario, reproduced. Records a
    /// [`StressOutcome`] under `name`. `pace_hz` throttles the loop (see
    /// [`StressSpec::pace_hz`]).
    StressFlight {
        name: String,
        speed_mps: f64,
        secs: f64,
        pace_hz: f64,
    },
}

/// Sample width for `FrameSample`'s mean (120 frames).
const SAMPLE_FRAMES: u32 = 120;

// ============================================================================
// Stress-flight scenario (fast flight then stop — the streaming-lag repro)
// ============================================================================

/// One stress-flight scenario for [`run_stress`].
#[derive(Clone, Copy, Debug)]
pub struct StressSpec {
    pub name: &'static str,
    /// Flight speed (m/s). Keep below the streamer's teleport threshold
    /// (`MAX_PREDICT_SPEED` = 512) so this exercises TRAVEL, not the purge path.
    pub speed_mps: f64,
    /// Flight duration after entry completes.
    pub secs: f64,
    /// Horizontal view radius (chunks) — see [`Stage::radius`].
    pub radius: i32,
    /// Frame pacing (Hz; 0 = uncapped). The capture window runs vsync-off at
    /// thousands of FPS, which hands the per-frame-budgeted streaming lanes
    /// ~30× more wall-time per second than a real session and hides every
    /// backlog symptom. Pacing to 60 restores a real game's budget rate; the
    /// recorded frame times cover only the WORK half (update+draw), never the
    /// pacing sleep.
    pub pace_hz: f64,
}

/// Frame-time distribution over one phase of a stress run (milliseconds).
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    pub frames: usize,
    pub p50: f32,
    pub p95: f32,
    pub p99: f32,
    pub max: f32,
}

fn mean_p95_u32(samples: &[u32]) -> (f32, f32) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let n = samples.len();
    let sum: u64 = samples.iter().map(|&v| u64::from(v)).sum();
    let mean = sum as f32 / n as f32;
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let p95 = sorted[((n - 1) as f32 * 0.95).round() as usize] as f32;
    (mean, p95)
}

impl FrameStats {
    /// Sorts `ms` in place and reads the nearest-rank percentiles.
    fn from_ms(ms: &mut [f32]) -> FrameStats {
        if ms.is_empty() {
            return FrameStats::default();
        }
        ms.sort_by(f32::total_cmp);
        let at = |q: f64| ms[((ms.len() - 1) as f64 * q).round() as usize];
        FrameStats {
            frames: ms.len(),
            p50: at(0.50),
            p95: at(0.95),
            p99: at(0.99),
            max: *ms.last().expect("non-empty"),
        }
    }
}

/// A finished stress run's numbers. First landings are report-only; thresholds
/// get pinned (like the throughput tests' doc-comment numbers) once a few runs
/// establish the machine's baseline.
#[derive(Clone, Debug)]
pub struct StressOutcome {
    /// Frame times while flying.
    pub flight: FrameStats,
    /// Frame times between stopping and settling (or the cap).
    pub settle: FrameStats,
    /// Stop → first `entry_complete` frame. `None` = still unsettled at
    /// [`STRESS_SETTLE_CAP`] — the permanent-wedge symptom the claim-release
    /// fixes target.
    pub settle_time: Option<Duration>,
    /// `entry_debug()` at the cap (empty when settled) — names the stuck stage.
    pub stuck: String,
    /// Peak queue depths observed across the whole run.
    pub max_upload_queue: usize,
    pub max_light_apply: usize,
    pub max_mesh_worklist: usize,
    pub max_chunks: usize,
    pub max_worker_near_queue: usize,
    pub max_worker_far_queue: usize,
    /// Peak per-frame section (LOD tile) upload vertex bytes.
    pub max_section_upload_bytes: usize,
    pub min_active_workers: usize,
    pub min_stream_effort: f32,
    /// Light worklist size and loaded-chunk count at the moment of stop.
    pub light_worklist_at_stop: usize,
    pub chunks_at_stop: usize,
    /// Cumulative light-worklist inserts at stop, and inserts / chunks.
    pub light_seed_inserts_at_stop: u64,
    pub seeds_per_chunk: f32,
    /// Insert attempts by source at stop.
    pub light_seed_split_at_stop: crate::world::LightSeedSplit,
    /// Mean light jobs admitted per second between stop and settle (or cap).
    pub settle_light_admit_per_s: f32,
    /// Per-second snapshots after stop: admit rate, workers, effort, worklist.
    pub settle_samples: Vec<SettleSample>,
    /// `remesh_async` calls per coord between uploads (whole run).
    pub remesh_between_upload_mean: f32,
    pub remesh_between_upload_p95: f32,
    pub remesh_between_upload_n: u64,
    /// Mesh jobs claimed per chunk before its 27-neighbourhood light fixpoint.
    pub mesh_jobs_before_fixpoint_mean: f32,
    pub mesh_jobs_before_fixpoint_p95: f32,
    pub mesh_jobs_before_fixpoint_n: u64,
    /// `drop_stale_upload` hits per flight frame.
    pub drop_stale_per_frame_mean: f32,
    pub drop_stale_per_frame_p95: f32,
    pub remesh_async_calls: u64,
    pub drop_stale_uploads: u64,
    /// Worker mesh jobs that wrote into the staging ring vs the Vec fallback.
    pub mesh_staged: u64,
    pub mesh_fallback: u64,
    pub mesh_ring_full: u64,
    pub section_staged: u64,
    pub section_fallback: u64,
    pub section_ring_full: u64,
    /// Peak per-frame chunk+section upload vertex bytes (drain).
    pub max_drain_upload_bytes: usize,
}

/// One second of post-stop streaming (the light-drain counters).
#[derive(Clone, Copy, Debug)]
pub struct SettleSample {
    pub sec: u32,
    pub light_admit_per_s: f32,
    pub active_workers: usize,
    pub effort: f32,
    pub light_worklist: usize,
}

/// How long after stopping a stress run waits for `entry_complete` before
/// recording the world as wedged. Generous: a healthy settle is seconds.
const STRESS_SETTLE_CAP: Duration = Duration::from_secs(60);

/// Live state of the one in-flight stress run (stage-local, like the
/// capture/sample accumulators).
struct StressRun {
    flight_start: Instant,
    stopped: Option<Instant>,
    flight_ms: Vec<f32>,
    settle_ms: Vec<f32>,
    max_upload: usize,
    max_apply: usize,
    max_worklist: usize,
    max_chunks: usize,
    max_worker_near: usize,
    max_worker_far: usize,
    max_section_upload_bytes: usize,
    min_active_workers: usize,
    min_effort: f32,
    stop_admitted: u64,
    stop_worklist: usize,
    stop_chunks: usize,
    stop_seeds: u64,
    stop_split: crate::world::LightSeedSplit,
    sample_sec: u32,
    sample_admitted: u64,
    end_admitted: u64,
    samples: Vec<SettleSample>,
    drop_stale_flight: Vec<u32>,
    remesh_between_upload_mean: f32,
    remesh_between_upload_p95: f32,
    remesh_between_upload_n: u64,
    mesh_jobs_before_fixpoint_mean: f32,
    mesh_jobs_before_fixpoint_p95: f32,
    mesh_jobs_before_fixpoint_n: u64,
    remesh_async_calls: u64,
    drop_stale_uploads: u64,
    mesh_staged: u64,
    mesh_fallback: u64,
    mesh_ring_full: u64,
    section_staged: u64,
    section_fallback: u64,
    section_ring_full: u64,
    max_drain_upload_bytes: usize,
}

impl StressRun {
    fn new() -> StressRun {
        StressRun {
            flight_start: Instant::now(),
            stopped: None,
            flight_ms: Vec::new(),
            settle_ms: Vec::new(),
            max_upload: 0,
            max_apply: 0,
            max_worklist: 0,
            max_chunks: 0,
            max_worker_near: 0,
            max_worker_far: 0,
            max_section_upload_bytes: 0,
            min_active_workers: usize::MAX,
            min_effort: 1.0,
            stop_admitted: 0,
            stop_worklist: 0,
            stop_chunks: 0,
            stop_seeds: 0,
            stop_split: crate::world::LightSeedSplit::default(),
            sample_sec: 0,
            sample_admitted: 0,
            end_admitted: 0,
            samples: Vec::new(),
            drop_stale_flight: Vec::new(),
            remesh_between_upload_mean: 0.0,
            remesh_between_upload_p95: 0.0,
            remesh_between_upload_n: 0,
            mesh_jobs_before_fixpoint_mean: 0.0,
            mesh_jobs_before_fixpoint_p95: 0.0,
            mesh_jobs_before_fixpoint_n: 0,
            remesh_async_calls: 0,
            drop_stale_uploads: 0,
            mesh_staged: 0,
            mesh_fallback: 0,
            mesh_ring_full: 0,
            section_staged: 0,
            section_fallback: 0,
            section_ring_full: 0,
            max_drain_upload_bytes: 0,
        }
    }

    fn finish(mut self, settle_time: Option<Duration>, stuck: String) -> StressOutcome {
        let (drop_mean, drop_p95) = mean_p95_u32(&self.drop_stale_flight);
        StressOutcome {
            flight: FrameStats::from_ms(&mut self.flight_ms),
            settle: FrameStats::from_ms(&mut self.settle_ms),
            settle_time,
            stuck,
            max_upload_queue: self.max_upload,
            max_light_apply: self.max_apply,
            max_mesh_worklist: self.max_worklist,
            max_chunks: self.max_chunks,
            max_worker_near_queue: self.max_worker_near,
            max_worker_far_queue: self.max_worker_far,
            max_section_upload_bytes: self.max_section_upload_bytes,
            min_active_workers: if self.min_active_workers == usize::MAX {
                0
            } else {
                self.min_active_workers
            },
            min_stream_effort: self.min_effort,
            light_worklist_at_stop: self.stop_worklist,
            chunks_at_stop: self.stop_chunks,
            light_seed_inserts_at_stop: self.stop_seeds,
            seeds_per_chunk: if self.stop_chunks == 0 {
                0.0
            } else {
                self.stop_seeds as f32 / self.stop_chunks as f32
            },
            light_seed_split_at_stop: self.stop_split,
            settle_light_admit_per_s: {
                let dt = settle_time.unwrap_or(STRESS_SETTLE_CAP).as_secs_f32();
                if dt <= 0.0 {
                    0.0
                } else {
                    (self.end_admitted.saturating_sub(self.stop_admitted)) as f32 / dt
                }
            },
            settle_samples: self.samples,
            remesh_between_upload_mean: self.remesh_between_upload_mean,
            remesh_between_upload_p95: self.remesh_between_upload_p95,
            remesh_between_upload_n: self.remesh_between_upload_n,
            mesh_jobs_before_fixpoint_mean: self.mesh_jobs_before_fixpoint_mean,
            mesh_jobs_before_fixpoint_p95: self.mesh_jobs_before_fixpoint_p95,
            mesh_jobs_before_fixpoint_n: self.mesh_jobs_before_fixpoint_n,
            drop_stale_per_frame_mean: drop_mean,
            drop_stale_per_frame_p95: drop_p95,
            remesh_async_calls: self.remesh_async_calls,
            drop_stale_uploads: self.drop_stale_uploads,
            mesh_staged: self.mesh_staged,
            mesh_fallback: self.mesh_fallback,
            mesh_ring_full: self.mesh_ring_full,
            section_staged: self.section_staged,
            section_fallback: self.section_fallback,
            section_ring_full: self.section_ring_full,
            max_drain_upload_bytes: self.max_drain_upload_bytes,
        }
    }
}

/// Run each stress scenario in its own stage (fresh `Game` per spec) inside the
/// process's single event loop and return the outcomes in spec order.
pub fn run_stress(specs: &[StressSpec]) -> Vec<(String, StressOutcome)> {
    let stages = specs
        .iter()
        .map(|s| Stage {
            seed: GOLDEN_SEED,
            // Above the Terrain band (amplitude 20 around ~64) so a straight
            // +X flight stays airborne; scripted games run no physics, so the
            // height only affects which chunk layers stream.
            cam: Some(CameraPose {
                pos: DVec3::new(0.0, 96.0, 0.0),
                yaw: 0.0,
                pitch: -0.15,
            }),
            day: SCRIPTED_DEFAULT_DAY,
            view: DebugView::Normal,
            setup: None,
            // The one site that needs a non-default radius, so it stays a
            // plain literal rather than `Stage::new` + an override.
            radius: Some(s.radius),
            render: crate::render_config::RenderConfig::golden(),
            kind: StageKind::StressFlight {
                name: s.name.to_string(),
                speed_mps: s.speed_mps,
                secs: s.secs,
                pace_hz: s.pace_hz,
            },
        })
        .collect();
    let out = execute(stages);
    specs
        .iter()
        .filter_map(|s| {
            out.stress
                .get(s.name)
                .map(|o| (s.name.to_string(), o.clone()))
        })
        .collect()
}

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
    stress: std::collections::HashMap<String, StressOutcome>,
}

/// Scratch capture path for a shot/view (decoded back by the evaluator).
fn scratch_path(name: &str, tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("watt_golden_{name}_{tag}.png"))
}

/// The Normal-view capture path for an `ImageMatch` shot: the golden itself
/// when blessing (captured straight in), else a scratch file to diff.
fn image_capture_path(name: &str, bless: bool) -> PathBuf {
    if bless {
        golden_path(name)
    } else {
        scratch_path(name, "normal")
    }
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
        .map(|seed| {
            Stage::new(
                seed,
                None,
                SCRIPTED_DEFAULT_DAY,
                DebugView::Normal,
                None,
                crate::render_config::RenderConfig::golden(),
                StageKind::EntryTime,
            )
        })
        .collect();
    for c in &acc.criteria {
        match c {
            Criterion::ImageMatch { shot, .. } => stages.push(Stage::new(
                shot.seed,
                Some(shot.cam),
                shot.day,
                DebugView::Normal,
                shot.setup,
                crate::render_config::RenderConfig::golden(),
                StageKind::Capture {
                    path: image_capture_path(shot.name, bless),
                },
            )),
            Criterion::SkyHoleCount { shot, .. } => stages.push(Stage::new(
                shot.seed,
                Some(shot.cam),
                shot.day,
                DebugView::TerrainKey,
                shot.setup,
                crate::render_config::RenderConfig::golden(),
                StageKind::Capture {
                    path: scratch_path(shot.name, "terrainkey"),
                },
            )),
            Criterion::FrameTime { shot, .. } => stages.push(Stage::new(
                shot.seed,
                Some(shot.cam),
                shot.day,
                DebugView::Normal,
                shot.setup,
                crate::render_config::RenderConfig::golden(),
                StageKind::FrameSample {
                    name: shot.name.to_string(),
                },
            )),
            Criterion::EntryTime { .. } => {}
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
    // `Game::update` requires the audio seam, but the scripted path returns before
    // touching it — a muted SoundSystem (NullBackend + empty catalog) satisfies the
    // signature with no audio device and no `assets/sounds` dependency.
    let (mut sound, cues) = crate::audio::SoundSystem::mute();
    // Scripted `Game::update` returns before the audio seam, but its signature needs a
    // director; build one from the loaded catalog (warnings discarded — headless).
    let (palette, _) = crate::audio::CuePalette::build(&cues, sound.catalog());
    let mut audio = crate::audio::AudioDirector::new(palette);
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
    let mut stress: Option<StressRun> = None;
    let mut floated = false;

    voxel_engine::run(scripted_config(), move |eng| {
        // Work-half timestamp for the stress stages' frame samples (their
        // pacing sleep must not count as frame cost).
        let frame_started = Instant::now();
        // First frame of the whole run: pull our just-opened (and therefore
        // focused) window out of the tiling layout. gharial ignores the
        // fixed-size hint and re-splits the column the moment another window
        // on the tag is touched mid-run — which resizes the swapchain and
        // breaks every later capture. Floating keeps the requested size for
        // the whole run. Best-effort: no gharialctl (other WMs, CI) → no-op;
        // a mis-timed focus is caught by the capture-dimension guard anyway.
        if !floated {
            floated = true;
            let _ = std::process::Command::new("gharialctl")
                .arg("toggle-float")
                .status();
        }
        let stage = &stages[idx];
        // First frame of a stage: (re)build its scripted game and reset the
        // per-stage watchdog / sample accumulators.
        let g = game.get_or_insert_with(|| {
            let mut g = Game::scripted(stage.seed, stage.render);
            g.set_debug_view(stage.view);
            if let Some(cam) = stage.cam {
                g.teleport(cam);
            }
            // Non-default streaming radius (stress stages): scripted updates
            // never apply `Settings`, so push it straight into the world.
            // Vertical stays at the settings default (3 layers).
            if let Some(radius) = stage.radius {
                g.world_mut()
                    .set_view_distances(radius, Settings::default().vertical_distance);
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
            stress = None;
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
        g.update(
            eng,
            &mut router,
            &mut mods,
            &mut settings,
            &mut sound,
            &mut audio,
        );
        // Shake 0: captures must be deterministic (no live trauma exists in the
        // scripted path anyway).
        g.draw(eng, &mut mods, settings.fov, 0.0);

        // A live stress run owns its stage's frames from here: it deliberately
        // flies THROUGH the un-entry states the gate below waits out, so it
        // bypasses that gate and its watchdog (the run keeps its own settle cap).
        if let StageKind::StressFlight {
            name,
            speed_mps,
            secs,
            pace_hz,
        } = &stage.kind
            && let Some(run) = stress.as_mut()
        {
            // The WORK half of this frame (update+draw, everything above);
            // the pacing sleep below is deliberately excluded.
            let work = frame_started.elapsed();
            let ms = work.as_secs_f32() * 1000.0;
            let gauges = g.world().stream_gauges();
            run.max_upload = run.max_upload.max(gauges.upload_queue);
            run.max_apply = run.max_apply.max(gauges.light_apply_queue);
            run.max_worklist = run.max_worklist.max(gauges.mesh_worklist);
            run.max_chunks = run.max_chunks.max(gauges.chunks);
            run.max_worker_near = run.max_worker_near.max(gauges.worker_near_queue);
            run.max_worker_far = run.max_worker_far.max(gauges.worker_far_queue);
            run.max_section_upload_bytes =
                run.max_section_upload_bytes.max(gauges.section_upload_bytes);
            if gauges.worker_capacity != 0 {
                run.min_active_workers = run.min_active_workers.min(gauges.active_workers);
            }
            run.min_effort = run.min_effort.min(gauges.effort);
            run.remesh_between_upload_mean = gauges.remesh_between_upload_mean;
            run.remesh_between_upload_p95 = gauges.remesh_between_upload_p95;
            run.remesh_between_upload_n = gauges.remesh_between_upload_n;
            run.mesh_jobs_before_fixpoint_mean = gauges.mesh_jobs_before_fixpoint_mean;
            run.mesh_jobs_before_fixpoint_p95 = gauges.mesh_jobs_before_fixpoint_p95;
            run.mesh_jobs_before_fixpoint_n = gauges.mesh_jobs_before_fixpoint_n;
            run.remesh_async_calls = gauges.remesh_async_calls;
            run.drop_stale_uploads = gauges.drop_stale_uploads;
            run.mesh_staged = gauges.mesh_staged;
            run.mesh_fallback = gauges.mesh_fallback;
            run.mesh_ring_full = gauges.mesh_ring_full;
            run.section_staged = gauges.section_staged;
            run.section_fallback = gauges.section_fallback;
            run.section_ring_full = gauges.section_ring_full;
            run.max_drain_upload_bytes = run.max_drain_upload_bytes.max(gauges.drain_upload_bytes);
            let finished = match run.stopped {
                None => {
                    run.flight_ms.push(ms);
                    run.drop_stale_flight.push(gauges.drop_stale_this_frame);
                    // dt-based advance: constant speed at any frame rate. The
                    // clamp keeps one hitch from a teleport-sized jump (the
                    // streamer treats >0.5 s gaps as discontinuities).
                    let dt = f64::from(eng.frame_time()).min(0.1);
                    g.player_mut().position.x += speed_mps * dt;
                    if run.flight_start.elapsed().as_secs_f64() >= *secs {
                        run.stopped = Some(Instant::now());
                        run.stop_admitted = gauges.light_admitted;
                        run.stop_worklist = gauges.light_worklist;
                        run.stop_chunks = gauges.chunks;
                        run.stop_seeds = gauges.light_seed_inserts;
                        run.stop_split = gauges.light_seed_split;
                        run.sample_admitted = gauges.light_admitted;
                        run.end_admitted = gauges.light_admitted;
                        eprintln!(
                            "stress {name}: flight over — settling… light_worklist={} chunks={} seeds/chunk={:.2}",
                            gauges.light_worklist,
                            gauges.chunks,
                            if gauges.chunks == 0 {
                                0.0
                            } else {
                                gauges.light_seed_inserts as f32 / gauges.chunks as f32
                            }
                        );
                    }
                    None
                }
                Some(stopped) => {
                    run.settle_ms.push(ms);
                    let sec = stopped.elapsed().as_secs() as u32;
                    if sec > run.sample_sec {
                        let dt = (sec - run.sample_sec) as f32;
                        let delta = gauges.light_admitted.saturating_sub(run.sample_admitted);
                        let admit_per_s = delta as f32 / dt;
                        run.samples.push(SettleSample {
                            sec,
                            light_admit_per_s: admit_per_s,
                            active_workers: gauges.active_workers,
                            effort: gauges.effort,
                            light_worklist: gauges.light_worklist,
                        });
                        run.sample_admitted = gauges.light_admitted;
                        run.sample_sec = sec;
                        eprintln!(
                            "stress {name}: t={sec}s admit/s={admit_per_s:.0} workers={} effort={:.2} light_worklist={}",
                            gauges.active_workers,
                            gauges.effort,
                            gauges.light_worklist
                        );
                    }
                    run.end_admitted = gauges.light_admitted;
                    if g.world().entry_complete() {
                        Some((Some(stopped.elapsed()), String::new()))
                    } else if stopped.elapsed() >= STRESS_SETTLE_CAP {
                        Some((None, g.world().entry_debug()))
                    } else {
                        if frame.is_multiple_of(120) {
                            eprintln!("stress {name}: settling… {}", g.world().entry_debug());
                        }
                        None
                    }
                }
            };
            frame += 1;
            if let Some((settle_time, stuck)) = finished {
                let outcome = stress
                    .take()
                    .expect("run is live")
                    .finish(settle_time, stuck);
                sink.borrow_mut().stress.insert(name.clone(), outcome);
                idx += 1;
                game = None;
                if idx >= stages.len() {
                    return false;
                }
            } else if *pace_hz > 0.0 {
                // Pace to the target cadence so per-frame lane budgets fire
                // at a real session's rate, not the capture window's
                // uncapped thousands of FPS.
                let target = Duration::from_secs_f64(1.0 / pace_hz);
                if let Some(rest) = target.checked_sub(frame_started.elapsed()) {
                    std::thread::sleep(rest);
                }
            }
            return true;
        }

        // Captures additionally wait for the fully-refined far field: a coarse
        // ancestor cover is entry-playable, but refinement landing later moves
        // horizon pixels — and, through the exposure meter, the whole frame's
        // brightness — which made blessed shots run-to-run flaky.
        let entry = g.world().entry_complete();
        let refined =
            !matches!(stage.kind, StageKind::Capture { .. }) || g.world().far_field_refined();
        if !entry || !refined {
            frame += 1;
            // Sample a few times a second (entry_debug scans the box — not per frame).
            if frame.is_multiple_of(20) {
                let sig = if entry {
                    format!(
                        "far field refining… sections pending: {}",
                        g.world().far_field_pending()
                    )
                } else {
                    g.world().entry_debug()
                };
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
                    if stalled {
                        "STALLED (no progress in 8s)"
                    } else {
                        "timed out (180s)"
                    },
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
                sink.borrow_mut()
                    .entry_times
                    .insert(stage.seed, stage_start.elapsed());
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
            StageKind::StressFlight {
                name,
                speed_mps,
                pace_hz,
                ..
            } => {
                // Entry complete: begin the flight. The branch above the entry
                // gate owns every subsequent frame of this stage.
                eprintln!(
                    "stress {name}: entry complete — flying +X at {speed_mps} m/s ({pace_hz} Hz pace)"
                );
                stress = Some(StressRun::new());
                false
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

    Rc::try_unwrap(outcomes)
        .map(RefCell::into_inner)
        .unwrap_or_default()
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
    let golden_entry_time = outcomes
        .entry_times
        .get(&GOLDEN_SEED)
        .copied()
        .unwrap_or_default();
    let result = evaluate(acc, bless, &outcomes);
    Report {
        golden_entry_time,
        result,
    }
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
        Some(Err(e)) => Err(Failure {
            what: what.to_string(),
            detail: format!("screenshot_to: {e}"),
        }),
        None => Err(Failure {
            what: what.to_string(),
            detail: format!("no capture recorded for {}", path.display()),
        }),
    }
}

fn eval_criterion(c: &Criterion, bless: bool, out: &Outcomes) -> Result<(), Failure> {
    match c {
        Criterion::ImageMatch {
            shot,
            max_pct_changed,
        } => {
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
            // A dimension mismatch is never pixel drift: the WM resized the
            // window mid-run (tiling WMs may ignore the fixed-size hint). Name
            // the real problem instead of reporting a meaningless 100%.
            if (got.width, got.height) != (want.width, want.height) {
                return Err(Failure {
                    what: format!("image_match {}", shot.name),
                    detail: format!(
                        "capture is {}×{} but the golden is {}×{} — the harness window was \
                         resized during the run (keep it floating/untouched and re-run)",
                        got.width, got.height, want.width, want.height
                    ),
                });
            }
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
            let mean_ms = out
                .frame_times
                .get(shot.name)
                .copied()
                .ok_or_else(|| Failure {
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
    }
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
            cam: CameraPose {
                pos: DVec3::new(0.0, 80.0, 0.0),
                yaw: 0.0,
                pitch: -0.2,
            },
            name: "spawn_forward",
            day: SCRIPTED_DEFAULT_DAY,
            setup: None,
        },
        GoldenShot {
            seed: GOLDEN_SEED,
            // Looking out at the horizon — the shot where LOD/skin holes show.
            cam: CameraPose {
                pos: DVec3::new(0.0, 90.0, 0.0),
                yaw: 2.35,
                pitch: 0.0,
            },
            name: "horizon",
            day: SCRIPTED_DEFAULT_DAY,
            setup: None,
        },
        GoldenShot {
            seed: GOLDEN_SEED,
            // Looking down-forward so the full-res box edge — view_radius = 6
            // chunks = 96 m — sits mid-frame: this is where the chunk→far-tile
            // handoff must be seamless.
            cam: CameraPose {
                pos: DVec3::new(0.0, 110.0, 0.0),
                yaw: 0.0,
                pitch: -0.35,
            },
            name: "tile_boundary",
            day: SCRIPTED_DEFAULT_DAY,
            setup: None,
        },
        GoldenShot {
            seed: GOLDEN_SEED,
            // Pose reviewed at first capture. Primary-day pose (mirrors spawn_forward) at midnight.
            cam: CameraPose {
                pos: DVec3::new(0.0, 80.0, 0.0),
                yaw: 0.0,
                pitch: -0.2,
            },
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
            cam: CameraPose {
                pos: DVec3::new(8.0, 40.0, 8.0),
                yaw: 0.0,
                pitch: 0.0,
            },
            name: "cave_interior",
            day: 0.5,
            setup: Some(carve_cave),
        },
        GoldenShot {
            seed: GOLDEN_SEED,
            // Pose reviewed at first capture. Golden pos
            // +30 m, grazing down (~ −8°), yaw along the day-0.35 sun azimuth so
            // the 64/256 m splits + fade band sit mid-frame.
            cam: CameraPose {
                pos: DVec3::new(0.0, 110.0, 0.0),
                yaw: 0.24,
                pitch: -0.14,
            },
            name: "shadow_boundary",
            day: 0.35,
            setup: None,
        },
        GoldenShot {
            seed: GOLDEN_SEED,
            // Pose reviewed at first capture. Near-surface,
            // near-level (~ −1°), yaw toward the day-0.30 sun azimuth so pure sky
            // sits over fog→1 terrain. `pos.y` is a stand-in for surface_y(0,0)+2,
            // to be re-pinned at first capture.
            cam: CameraPose {
                pos: DVec3::new(0.0, 66.0, 0.0),
                yaw: 0.20,
                pitch: -0.017,
            },
            name: "horizon_fog_vs_sky",
            day: 0.30,
            setup: None,
        },
        GoldenShot {
            seed: GOLDEN_SEED,
            // Pose reviewed at first capture. Water shot: a
            // low camera (y≈72, a few blocks over the ~64 water surface) grazing
            // ACROSS the lakes that fill the origin basin (see the blessed
            // spawn_forward/tile_boundary captures) at a moderate down angle so a
            // wide sheet of water fills mid-frame — the grazing view maximizes
            // fresnel/sky reflection. Yaw is pinned along the day-0.3 sun azimuth
            // (as shadow_boundary reasons) so the sun's specular GLINT lands on the
            // water. Pinned day ⇒ pinned `anim` phase ⇒ deterministic wave
            // geometry. This is the golden that protects every WATER_* tunable.
            cam: CameraPose {
                pos: DVec3::new(0.0, 72.0, 0.0),
                yaw: 0.24,
                pitch: -0.12,
            },
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
    // The ordinary element-derived stone/lumin composition.
    let emitter = registry
        .id_by_name("Stone+Lumin")
        .expect("worldgen registers the Stone+Lumin composition");
    let stone = registry
        .id_by_name("Stone")
        .expect("Stone is a built-in block");
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
        Screenshot {
            width: w,
            height: h,
            rgba,
        }
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
        assert!(
            is_uniform(&flat),
            "an all-black frame is uniform → unblessable"
        );

        let mut scene = a_clone(&flat);
        set_px(&mut scene, 3, 2, Color::rgb(1, 0, 0));
        assert!(
            !is_uniform(&scene),
            "one differing pixel makes it a real frame"
        );
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
        assert!(
            (d.pct_changed - 1.0).abs() < 1e-4,
            "pct_changed = {}",
            d.pct_changed
        );
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
        Screenshot {
            width: s.width,
            height: s.height,
            rgba: s.rgba.clone(),
        }
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

    // Live-capture criteria (ImageMatch/SkyHoleCount/EntryTime/FrameTime) drive
    // a real windowed `Engine` and are exercised by `cargo run --bin golden`.

    #[test]
    fn scripted_default_day_matches_clock() {
        // The pre-existing shots pin `day` to SCRIPTED_DEFAULT_DAY so `set_day`
        // is a no-op for them; that only holds while it equals the scripted
        // clock's start. Guards note-1's by-construction no-op.
        assert_eq!(crate::sky::SkyClock::default().day(), SCRIPTED_DEFAULT_DAY);
    }
}
