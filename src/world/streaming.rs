//! Streaming: the [`World::stream`] pass, worker result draining, job queueing,
//! budgeted uploads, chunk unloading, and view-radius bookkeeping.
//! These are `World` methods (struct lives in `mod.rs`).

use std::time::{Duration, Instant};

use voxel_engine::producer::{Budget, Progress};
use voxel_engine::{DVec3, Engine, FadeStyle};

use crate::block::appearance::{fill_layer, BlockAppearance, LAYER_BYTES, TEXTURE_SIZE};

use crate::coord::{ByPass, ChunkBox, Face};
use crate::space::FaceFrame;
use crate::derived::Revision;
use crate::math::block_coord;

use super::chunk::{CHUNK_SIZE, Chunk, ChunkData};
use super::generation::{Classify, ColumnHeights};
use crate::block::registry::{HotTables, AIR};
use super::brick::ChunkPayload;
use super::heightmip::{BakeExtent, HeightMip};
use super::metric::{DyCap, EyeDist, EyeMetric, HeightEnvelope};
use super::section::SectionPos;
use super::summary::{CellError, CellSummary, SseBudget};
use super::{
    ColumnKey, Coord, DIRTY_BUDGET, FastMap, FastSet, LightLane, Loaded, MeshLane, MeshState,
    SECTION_UPLOAD_BUDGET, SectionFrontierKey, SectionLane, SectionState, Sky, StreamLane,
    UPLOAD_BUDGET_BYTES, UPLOAD_QUEUE_MAX, UPLOAD_SCAN_MAX, World, light, mesh, pipeline, pyramid,
    quadtree,
};

/// Exact upload placement for a chunk mesh: integer chunk origin, full detail.
/// Pinned at upload so the GPU record is written once; draws only mark
/// visibility.
fn chunk_placement(coord: Coord) -> voxel_engine::MeshPlacement {
    voxel_engine::MeshPlacement::terrain(
        voxel_engine::IVec3::new(coord.x, coord.y, coord.z) * CHUNK_SIZE as i32,
        crate::ident::Detail::FULL,
    )
}

/// The GPU bytes a finished mesh will stage on upload (direction-major
/// vertices across every pass) — what the byte-based upload budget charges.
/// Counts both staged regions and the `Vec` fallback.
pub(in crate::world) fn mesh_output_bytes(data: &pipeline::MeshPayload) -> usize {
    data.vertex_bytes()
}

/// Vertex bytes a finished section mesh will stage (every slab × pass).
#[cfg(test)]
pub(in crate::world) fn section_output_bytes(data: &super::SectionMeshData) -> usize {
    data.vertex_bytes()
}

/// Edits whose chunk falls inside `pos`'s footprint and height domain. Free
/// function (not a `World` method) so callers needing only `&self.edits` — the
/// heightmip overlay refresh among them — don't have to borrow the rest of `World`.
#[cfg(test)]
pub(in crate::world) fn edits_in_footprint(
    edits: &FastMap<Coord, FastMap<usize, crate::block::registry::BlockId>>,
    pos: SectionPos,
) -> Vec<(Coord, Vec<(usize, crate::block::registry::BlockId)>)> {
    let cs = CHUNK_SIZE as i32;
    let span = pos.span();
    let (cx0, cz0) = (pos.min_x().div_euclid(cs), pos.min_z().div_euclid(cs));
    let cn = span / cs; // chunk columns per section side
    let cy_hi = super::section::DOMAIN_H / cs; // vertical chunk-layer count
    edits
        .iter()
        .filter(|(c, _)| {
            (cx0..cx0 + cn).contains(&c.x)
                && (cz0..cz0 + cn).contains(&c.z)
                && (0..cy_hi).contains(&c.y)
        })
        .map(|(&c, cells)| (c, cells.iter().map(|(&i, &b)| (i, b)).collect()))
        .collect()
}

/// Velocity prediction horizon: pre-loads sections ahead of eye motion so they're
/// ready by the time the eye reaches them. Conservative; tuning it larger preloads more
/// (safer for fast motion, low cost) but never shrinks the view.
const TAU_STREAM: f64 = 1.0;

/// Chart prediction lands on whole chunks. A flight's velocity jitters frame to frame; an exact
/// lookahead would move the predicted eye, and so recompute the frontier, every frame. Adding
/// zero folds a rounded `-0` into `+0`, so equal deltas key equal bits.
fn chart_delta(vel: DVec3) -> DVec3 {
    let step = CHUNK_SIZE as f64;
    (vel * TAU_STREAM / step).round() * step + DVec3::ZERO
}

/// Above this sample gap, treat eye motion as pause/teleport; discard velocity
/// to zero prediction. Generous (streaming may legitimately run at 15 Hz under
/// `stream_hz`); the speed cap below catches genuine discontinuities.
const MAX_PREDICT_SAMPLE_GAP: f64 = 0.5;

/// Above this apparent speed (m/s), treat the motion as a teleport rather than
/// travel: prediction is zeroed AND queued far work is purged, because its
/// admission-time priorities are stale where the eye is now.
const MAX_PREDICT_SPEED: f64 = 512.0;

/// Travel up to this speed gets the full streaming budget. It is comfortably
/// above ordinary walking/sprinting, so normal play and world entry retain
/// maximum convergence speed. Above it, useful chunk lifetime falls roughly
/// inversely with velocity, and effort follows the same curve.
const FULL_EFFORT_SPEED_MPS: f64 = 24.0;

/// Keep a small progress floor even during extreme travel. Stopping is never
/// required for the centre/collision neighbourhood to advance, while the cap
/// leaves most CPU and transfer time to the frame loop.
const MIN_STREAM_EFFORT: f32 = 0.15;

/// Once travel stops, restore background capacity over this time constant.
/// Shedding is immediate (a hitch should stop now); recovery is deliberately
/// damped so the first stationary frame cannot release a catch-up avalanche.
const STREAM_RECOVERY_SECS: f64 = 0.75;

/// Last topology pass cheaper than this: leftover light/mesh work at rest may
/// run at full worker/admission capacity. Half a 60 Hz frame — the post-flight
/// frames on this branch sit well below it, while an already-expensive pass
/// keeps travel shedding.
const STREAM_HEADROOM_SECS: f64 = 0.008;

/// Hysteresis: once rest-boosted, stay boosted until a pass exceeds this so a
/// slightly heavier first stationary admit cannot immediately re-shed.
const STREAM_HEADROOM_EXIT_SECS: f64 = 0.014;

/// Near-queue cap at rest when leftover light/mesh work remains. Travel keeps
/// `active * 4` because queued jobs go stale; at rest they will still be wanted,
/// and cheap light jobs otherwise idle the pool for the rest of a 16 ms frame.
/// Same order as [`pipeline::FAR_QUEUE_CAP`].
const NEAR_REST_QUEUE_CAP: usize = 256;

/// A real mesh upload always makes progress even when the scaled byte budget is
/// tiny. Most chunks fit below this; an unusually large first mesh is allowed
/// to overrun it once, just as it may overrun the normal byte budget once.
const MIN_UPLOAD_BUDGET_BYTES: usize = 256 << 10;

/// Minimum completed results integrated per drain before its time budget may
/// stop it. This releases claims promptly without reverting to the old
/// unbounded channel drain.
const RESULT_INTEGRATE_FLOOR: usize = 8;

/// Blocks past a chart's stored top that still stream on that chart. The band ends
/// `RELIEF` above the datum and the crust tops out at `MAX_GROUND`, so this clears
/// three thousand blocks of flight over the highest crust.
const CHART_FLIGHT: f64 = 2_048.0;

/// Blocks above a round world's stored top within which the far field still stands on its chart.
/// Higher up the body's impostor alone draws it.
const FAR_FLIGHT: f64 = 262_144.0;

/// Hysteresis of the far field's thresholds: crossing one back takes this factor more height than
/// crossing it did (the far reach is left at `FAR_FLIGHT · FAR_HOLD`, a ring doubling dropped at
/// `1 / FAR_HOLD` of the height that took it, and the chart stood on is left for another round
/// world's only when that top is this factor nearer).
const FAR_HOLD: f64 = 1.25;

/// The chart rings reach this many times the eye's height above the ground.
const FAR_VIEW: f64 = 3.0;

/// Candidates the coarsest chart ring may sweep (its square of sections), twice the section floor.
const FAR_CANDIDATES: f64 = (2 * super::SECTION_SLOT_FLOOR) as f64;

/// Velocity-aware streaming load controller. `effort` is the one normalized
/// signal shared by worker concurrency, queue lookahead, admission deadlines,
/// result integration, and GPU uploads, so those stages cannot fight each
/// other by independently trying to catch up. `boost` is the rest-time override:
/// leftover light/mesh work on a frame with headroom runs at full capacity so
/// the travel floor cannot idle the pool while tens of thousands of jobs wait.
#[derive(Clone, Copy, Debug)]
pub(in crate::world) struct StreamPacer {
    speed_mps: f64,
    effort: f32,
    boost: bool,
}

impl Default for StreamPacer {
    fn default() -> Self {
        Self {
            speed_mps: 0.0,
            effort: 1.0,
            boost: false,
        }
    }
}

impl StreamPacer {
    /// The useful-work fraction at `speed_mps`. Inverse scaling models the
    /// shrinking time a chunk remains in view; it is continuous at the full
    /// effort threshold and bounded away from zero for forward progress.
    fn target_effort(speed_mps: f64) -> f32 {
        if speed_mps.is_nan() || speed_mps <= FULL_EFFORT_SPEED_MPS {
            return 1.0;
        }
        if speed_mps.is_infinite() {
            return MIN_STREAM_EFFORT;
        }
        (FULL_EFFORT_SPEED_MPS / speed_mps).max(f64::from(MIN_STREAM_EFFORT)) as f32
    }

    fn update(&mut self, velocity: DVec3, sample_dt: f64) {
        // `hypot(x, 0) = |x|` and `hypot` is even, so `vy == 0` matches the
        // old horizontal speed bit for bit.
        self.speed_mps = velocity.x.hypot(velocity.y).hypot(velocity.z);
        let target = Self::target_effort(self.speed_mps);
        if target <= self.effort {
            // Load shedding has to beat the next expensive frame.
            self.effort = target;
            return;
        }
        // Recovery is a time-based exponential, independent of stream_hz.
        let dt = sample_dt.clamp(0.0, MAX_PREDICT_SAMPLE_GAP);
        let alpha = 1.0 - (-dt / STREAM_RECOVERY_SECS).exp();
        self.effort += (target - self.effort) * alpha as f32;
        if (target - self.effort).abs() < 0.001 {
            self.effort = target;
        }
    }

    /// Rest-time override: full workers and admission while light/mesh work
    /// remains, the eye is at or below walking speed, and the last topology
    /// pass had frame-time headroom. Travel still sheds — boosting during
    /// flight would spend the frame-time win on stale work.
    fn set_boost(&mut self, queued_near: bool, last_stream_secs: f64) {
        let at_rest = self.speed_mps <= FULL_EFFORT_SPEED_MPS;
        if !queued_near || !at_rest {
            self.boost = false;
            return;
        }
        if self.boost {
            self.boost = last_stream_secs < STREAM_HEADROOM_EXIT_SECS;
        } else {
            self.boost = last_stream_secs < STREAM_HEADROOM_SECS;
        }
    }

    pub(in crate::world) fn boosting(self) -> bool {
        self.boost
    }

    /// Effort applied to workers, admission, drain, and uploads this pass.
    fn applied_effort(self) -> f32 {
        if self.boost { 1.0 } else { self.effort }
    }

    pub(in crate::world) fn effort(self) -> f32 {
        self.effort
    }

    pub(in crate::world) fn speed_mps(self) -> f64 {
        self.speed_mps
    }

    pub(in crate::world) fn duration(self, base: Duration) -> Duration {
        base.mul_f32(self.applied_effort())
    }

    pub(in crate::world) fn floor(self, base: usize) -> usize {
        ((base as f32 * self.applied_effort()).ceil() as usize).clamp(1, base.max(1))
    }

    fn upload_bytes(self) -> usize {
        ((UPLOAD_BUDGET_BYTES as f32 * self.applied_effort()) as usize).max(MIN_UPLOAD_BUDGET_BYTES)
    }

    fn section_uploads(self) -> usize {
        ((SECTION_UPLOAD_BUDGET as f32 * self.applied_effort()).round() as usize)
            .clamp(1, SECTION_UPLOAD_BUDGET)
    }

    fn active_workers(self, capacity: usize) -> usize {
        ((capacity as f32 * self.applied_effort()).ceil() as usize).clamp(1, capacity.max(1))
    }

    /// Near-queue lookahead. Travel keeps a short cap so queued jobs do not go
    /// stale; at rest the deeper cap keeps cheap light jobs from idling the pool.
    fn near_queue_cap(self, capacity: usize) -> usize {
        let active = self.active_workers(capacity);
        let travel = (active * 4).max(8);
        if self.boost {
            travel.max(NEAR_REST_QUEUE_CAP)
        } else {
            travel
        }
    }
}

/// Timeout before meshing a chunk with missing neighbour light as degraded.
/// Degraded chunks remesh once real light arrives.
///
// Wait-time gating avoids a remesh storm at cold-world entry: most chunks
// receive light within this window and mesh once with final light. Only
// stragglers degrade. Without this, the worker pool remeshes every chunk twice.
const LIGHT_WAIT_DEGRADE: Duration = Duration::from_millis(150);

/// Tracks degraded meshes waiting for neighbour light to settle.
/// `blocked_since`: per-chunk timer for when it became light-blocked.
/// `degraded`: set of chunks currently drawing a degraded mesh, owed a remesh.
/// `dirty`: changed-light chunks waiting for a 27-neighbourhood fixpoint (or
/// the degrade timer) before their next mesh job.
#[derive(Default)]
pub(in crate::world) struct LightGate {
    pub(in crate::world) blocked_since: FastMap<Coord, Instant>,
    pub(in crate::world) degraded: FastSet<Coord>,
    pub(in crate::world) dirty: FastMap<Coord, Instant>,
}

impl LightGate {
    /// Start the wait timer for a light-blocked chunk (keeps an existing
    /// timer — re-eviction must not push the degrade horizon out).
    pub(in crate::world) fn note_blocked(&mut self, coord: Coord) {
        self.blocked_since.entry(coord).or_insert_with(crate::sched::now);
    }

    /// Mark a changed-light chunk; the first mark starts the degrade clock.
    fn mark_dirty(&mut self, coord: Coord) {
        self.dirty.entry(coord).or_insert_with(crate::sched::now);
    }
}

/// Cap on per-chunk remesh/job samples kept for the stress mean/p95 gauges.
const REMESH_SAMPLE_CAP: usize = 1 << 16;

/// Flight counters for light-convergence remeshes (stress C3).
#[derive(Default)]
pub(in crate::world) struct RemeshStats {
    pub remesh_async_calls: u64,
    pub drop_stale_uploads: u64,
    pub drop_stale_this_frame: u32,
    remesh_since_upload: FastMap<Coord, u16>,
    remesh_between_upload_samples: Vec<u16>,
    mesh_jobs_until_fixpoint: FastMap<Coord, u16>,
    mesh_jobs_fixpoint_done: FastSet<Coord>,
    mesh_jobs_before_fixpoint_samples: Vec<u16>,
}

impl RemeshStats {
    fn note_remesh(&mut self, coord: Coord) {
        self.remesh_async_calls += 1;
        let n = self.remesh_since_upload.entry(coord).or_insert(0);
        *n = n.saturating_add(1);
    }

    fn note_upload(&mut self, coord: Coord) {
        let n = self.remesh_since_upload.remove(&coord).unwrap_or(0);
        if self.remesh_between_upload_samples.len() < REMESH_SAMPLE_CAP {
            self.remesh_between_upload_samples.push(n);
        }
    }

    fn note_drop_stale(&mut self) {
        self.drop_stale_uploads += 1;
        self.drop_stale_this_frame = self.drop_stale_this_frame.saturating_add(1);
    }

    pub(in crate::world) fn note_mesh_job(&mut self, coord: Coord, nhood_quiet: bool) {
        if self.mesh_jobs_fixpoint_done.contains(&coord) {
            return;
        }
        if nhood_quiet {
            let n = self.mesh_jobs_until_fixpoint.remove(&coord).unwrap_or(0);
            if self.mesh_jobs_before_fixpoint_samples.len() < REMESH_SAMPLE_CAP {
                self.mesh_jobs_before_fixpoint_samples.push(n);
            }
            self.mesh_jobs_fixpoint_done.insert(coord);
        } else {
            let n = self.mesh_jobs_until_fixpoint.entry(coord).or_insert(0);
            *n = n.saturating_add(1);
        }
    }

    fn forget(&mut self, coord: Coord) {
        self.remesh_since_upload.remove(&coord);
        self.mesh_jobs_until_fixpoint.remove(&coord);
        self.mesh_jobs_fixpoint_done.remove(&coord);
    }

    fn between_upload_mean_p95(&self) -> (f32, f32, u64) {
        sample_mean_p95(&self.remesh_between_upload_samples)
    }

    fn jobs_before_fixpoint_mean_p95(&self) -> (f32, f32, u64) {
        sample_mean_p95(&self.mesh_jobs_before_fixpoint_samples)
    }
}

fn sample_mean_p95(samples: &[u16]) -> (f32, f32, u64) {
    let n = samples.len() as u64;
    if samples.is_empty() {
        return (0.0, 0.0, 0);
    }
    let sum: u64 = samples.iter().map(|&v| u64::from(v)).sum();
    let mean = sum as f32 / n as f32;
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let p95 = sorted[((n - 1) as f32 * 0.95).round() as usize] as f32;
    (mean, p95, n)
}

/// The strike/quarantine identity of a panicked job — the per-lane key
/// [`World::fail_job`] counts strikes against. A generate failure is keyed by
/// its whole column: the failing chunk inside a column job is unknown, and the
/// span requested for a column varies with the view, so per-span keys would
/// never accumulate strikes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(in crate::world) enum FailKey {
    Column { key: ColumnKey },
    /// One `Open` chunk. Not a [`Column`](Self::Column): a PosY column at the
    /// same `(cx, cz)` must keep its own strike count.
    Open { coord: Coord },
    Mesh { coord: Coord },
    Light { coord: Coord },
    Section { pos: SectionPos },
}

impl FailKey {
    fn of(key: &pipeline::JobKey) -> FailKey {
        match key {
            pipeline::JobKey::Column { key, .. } => FailKey::Column { key: *key },
            pipeline::JobKey::Open { coord } => FailKey::Open { coord: *coord },
            pipeline::JobKey::Mesh { coord } => FailKey::Mesh { coord: *coord },
            pipeline::JobKey::Light { coord } => FailKey::Light { coord: *coord },
            pipeline::JobKey::Section { pos, .. } => FailKey::Section { pos: *pos },
        }
    }
}

/// One generate admission: a face column, or a single `Open` chunk.
/// `Open` is not encoded as a one-chunk PosY column — that key collided with
/// a real PosY column's [`FailKey`] at the same `(cx, cz)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::world) enum GenRun {
    Column { key: ColumnKey, lo: i32, hi: i32 },
    Open { coord: Coord },
}

impl GenRun {
    /// Chunk the run is ordered from. A column uses its low end: every layer
    /// shares the tangent coordinates, and +Y ordering ignores altitude.
    fn anchor(self) -> Coord {
        match self {
            GenRun::Column { key, lo, .. } => key.chunk(lo),
            GenRun::Open { coord } => coord,
        }
    }
}

/// Panics tolerated per claim before it is quarantined. A panic is a real bug
/// in job code, usually deterministic for one input — retrying a couple of
/// times absorbs flukes (allocation pressure, a racing palette snapshot)
/// without looping forever on poison.
const MAX_JOB_STRIKES: u8 = 3;

/// Why a claim is being resolved WITHOUT a payload — see
/// [`World::resolve_claim`]. Cancelled: descheduled at the pool, no strike.
/// Failed: the job panicked; strikes accumulate toward quarantine.
enum ClaimOutcome {
    Cancelled,
    Failed,
}

/// Forward-progress floor for the generation lane: admit at least this many
/// columns before the deadline can stop it, so a boundary-cross flood still
/// makes strict progress each frame under a tight budget (the same floor role
/// [`super::StreamLane::MIN_ADMIT`] plays for the per-chunk lanes).
const GEN_MIN_ADMIT: usize = 8;

/// Group chunk coords into generate runs.
///
/// `Axis(f)` chunks that share a [`ColumnKey`] become one inclusive altitude
/// run. With `span_loaded` (the spawn slab) every coord of that column in the
/// input is recorded, loaded or not, and a run is emitted only when something
/// in a contiguous group is missing — a uniform PosY slab therefore submits
/// its full altitude span. Without it (`request_region_data`) only missing
/// coords are recorded, and a gap stays inside the run when every chunk
/// between the ends has the same sky (a loaded hole of the same face). A gap
/// whose sky differs splits the run, so a PosY job never generates an Open coord.
///
/// `Open` is one job per missing chunk ([`GenRun::Open`]), not merged with
/// its `(cx, cz)` neighbours and not encoded as a PosY column. Voxel
/// generation still calls `generate_column` with that PosY encoding so
/// `key.chunk(cy)` round-trips; only the claim and quarantine key differ.
/// `accept_column` does not install a ceiling and `store_chunk` does not
/// record the chunk when `sky` is `Open`.
///
/// `skip_quarantine` drops a quarantined run before it can take a slot.
/// The slab path leaves it false so the submit rejects the run and the
/// caller keeps `pending_gen` set.
fn gather_column_runs(
    coords: impl IntoIterator<Item = Coord>,
    mut sky_of: impl FnMut(Coord) -> Sky,
    mut present: impl FnMut(Coord) -> bool,
    mut quarantined: impl FnMut(FailKey) -> bool,
    span_loaded: bool,
    skip_quarantine: bool,
) -> Vec<GenRun> {
    let mut axis: FastMap<ColumnKey, Vec<(i32, bool)>> = FastMap::default();
    let mut open_seen: FastSet<Coord> = FastSet::default();
    let mut runs: Vec<GenRun> = Vec::new();
    for coord in coords {
        let missing = !present(coord);
        // Without `span_loaded` a present chunk records nothing, whatever its sky.
        if !span_loaded && !missing {
            continue;
        }
        match sky_of(coord) {
            Sky::Open => {
                if skip_quarantine && quarantined(FailKey::Open { coord }) {
                    continue;
                }
                if missing && open_seen.insert(coord) {
                    runs.push(GenRun::Open { coord });
                }
            }
            Sky::Axis(face) => {
                let (key, alt) = ColumnKey::of(face, coord);
                if skip_quarantine && quarantined(FailKey::Column { key }) {
                    continue;
                }
                axis.entry(key).or_default().push((alt, missing));
            }
        }
    }
    for (key, mut alts) in axis {
        alts.sort_unstable_by_key(|p| p.0);
        let mut deduped: Vec<(i32, bool)> = Vec::new();
        for (alt, missing) in alts {
            if let Some(last) = deduped.last_mut() {
                if last.0 == alt {
                    last.1 |= missing;
                    continue;
                }
            }
            deduped.push((alt, missing));
        }
        if span_loaded {
            let mut i = 0;
            while i < deduped.len() {
                let mut j = i;
                while j + 1 < deduped.len() && deduped[j + 1].0 == deduped[j].0 + 1 {
                    j += 1;
                }
                if deduped[i..=j].iter().any(|p| p.1) {
                    runs.push(GenRun::Column { key, lo: deduped[i].0, hi: deduped[j].0 });
                }
                i = j + 1;
            }
        } else {
            let mut start = 0;
            for i in 0..deduped.len() {
                let split = i + 1 == deduped.len()
                    || !gap_same_sky(key, deduped[i].0, deduped[i + 1].0, &mut sky_of);
                if split {
                    runs.push(GenRun::Column {
                        key,
                        lo: deduped[start].0,
                        hi: deduped[i].0,
                    });
                    start = i + 1;
                }
            }
        }
    }
    // Open occupies the slot the old PosY one-chunk encoding sorted into, so
    // a +Y gather keeps today's order.
    runs.sort_unstable_by_key(|run| match run {
        GenRun::Column { key, lo, .. } => (key.face.index(), key.a, key.b, *lo),
        GenRun::Open { coord } => (Face::PosY.index(), coord.x, coord.z, coord.y),
    });
    runs
}

fn gap_same_sky(key: ColumnKey, lo: i32, hi: i32, sky_of: &mut impl FnMut(Coord) -> Sky) -> bool {
    let want = Sky::Axis(key.face);
    (lo + 1..hi).all(|alt| sky_of(key.chunk(alt)) == want)
}

/// Column priority: tangent chess (not the ×2 along-axis weight — a column
/// job is the whole run). `None` is 3-D chess of the anchor. +Y reads only
/// XZ, matching the old `(cx, cz)` key.
fn column_order(center: Coord, vel: DVec3, anchor: Coord, up: Option<Face>) -> u64 {
    let across = match up {
        None => anchor.chess3(center) as u64,
        Some(face) => anchor.across(center, face) as u64,
    };
    let scale = CHUNK_SIZE as f64;
    super::bias_order(
        across.saturating_mul(across).saturating_mul(1024),
        vel,
        f64::from(anchor.x - center.x) * scale,
        f64::from(anchor.y - center.y) * scale,
        f64::from(anchor.z - center.z) * scale,
        up,
    )
}

/// The chunk holding point `p`.
fn eye_chunk(p: DVec3) -> Coord {
    World::chunk_of(block_coord(p.x), block_coord(p.y), block_coord(p.z))
}

/// Sections from the eye's out to the edge of the square the coarsest ring sweeps, for a ring
/// reaching `outer` metres in sections of `span` (`quadtree::desired_sections`).
fn ring_reach(outer: f64, span: f64) -> f64 {
    (outer / span).ceil() + 1.0
}

/// Storage block the chart frontier treats as the eye: the centre chunk's middle, plus the
/// prediction delta. Y is the streamed altitude, not the chunk layer.
fn storage_eye_block(center: Coord, eye_y: f64, delta: DVec3) -> (i64, i64, i64) {
    let cs = CHUNK_SIZE as i64;
    let x = center.x as i64 * cs + cs / 2 + delta.x.round() as i64;
    let z = center.z as i64 * cs + cs / 2 + delta.z.round() as i64;
    let y = (eye_y + delta.y).round() as i64;
    (x, y, z)
}

/// Generator surface bounds of the storage rects the near-window punch tested, stamped with the
/// frontier sweep that last read them. The terrain is immutable, so an answer never changes; a
/// sweep keeps only the rects it read.
#[derive(Default)]
pub(in crate::world) struct NearBounds {
    rects: FastMap<(u16, [i32; 4]), (Option<(i32, i32)>, u32)>,
    pass: u32,
}

impl NearBounds {
    /// Run one frontier sweep over the memo, then drop the rects it did not read.
    fn sweep<R>(&mut self, select: impl FnOnce(&mut Self) -> R) -> R {
        self.pass = self.pass.wrapping_add(1);
        let out = select(self);
        let pass = self.pass;
        self.rects.retain(|_, e| e.1 == pass);
        out
    }

    /// The bounds of `rect`, read through `surface` the first time.
    fn get(&mut self, rect: (u16, [i32; 4]), surface: impl FnOnce() -> Option<(i32, i32)>) -> Option<(i32, i32)> {
        let pass = self.pass;
        let e = self.rects.entry(rect).or_insert_with(|| (surface(), pass));
        e.1 = pass;
        e.0
    }
}

/// Sections of a chart seat, already filtered. Distance rings chose the detail: collapsing every
/// complete quad would flatten those rings onto the chord cap, so a quad merges only while the
/// frontier is over `budget`, and only when the parent passes `keep`.
fn coarsen_chart(
    sections: Vec<SectionPos>,
    max_detail: i8,
    budget: usize,
    mut keep: impl FnMut(SectionPos) -> bool,
) -> Vec<SectionPos> {
    let mut set: FastSet<SectionPos> = sections.into_iter().filter(|s| s.detail.0 <= max_detail).collect();
    if set.len() <= budget || set.is_empty() {
        return set.into_iter().collect();
    }
    let finest = set.iter().map(|s| s.detail.0).min().unwrap();
    for child_d in (finest..max_detail).rev() {
        if set.len() <= budget {
            break;
        }
        let mut kids: FastMap<SectionPos, u8> = FastMap::default();
        for &c in &set {
            if c.detail.0 == child_d {
                *kids.entry(c.parent()).or_insert(0) += 1;
            }
        }
        let mut merges: Vec<SectionPos> = kids.into_iter().filter(|&(p, n)| n == 4 && keep(p)).map(|(p, _)| p).collect();
        merges.sort_unstable_by_key(section_key);
        for p in merges {
            if set.len() <= budget {
                break;
            }
            for q in super::section::Quadrant::ALL {
                set.remove(&p.child(q));
            }
            set.insert(p);
        }
    }
    set.into_iter().collect()
}

/// Home-chart footprint of `s` (`hi` exclusive). A neighbour section unfolds across the seam.
fn home_rect(s: SectionPos, across: Option<&super::seam::SeamAcross>) -> (i64, i64, i64, i64) {
    let span = s.span() as i64;
    let (x0, z0) = (s.min_x() as i64, s.min_z() as i64);
    if let Some(m) = across {
        let (a, c) = m.home_xz(x0, z0);
        let (b, d) = m.home_xz(x0 + span, z0 + span);
        (a.min(b), c.min(d), a.max(b), c.max(d))
    } else {
        (x0, z0, x0 + span, z0 + span)
    }
}

/// Whether `s` meets the full-res chunk box. A neighbour section is tested in the home chart,
/// unfolded past the seam.
fn covers_near(s: SectionPos, near: (i64, i64, i64, i64), across: Option<&super::seam::SeamAcross>) -> bool {
    let (x0, z0, x1, z1) = home_rect(s, across);
    x0 < near.1 && x1 > near.0 && z0 < near.3 && z1 > near.2
}

/// Whether `s` lies wholly inside the full-res chunk box.
fn inside_near(s: SectionPos, near: (i64, i64, i64, i64), across: Option<&super::seam::SeamAcross>) -> bool {
    let (x0, z0, x1, z1) = home_rect(s, across);
    x0 >= near.0 && x1 <= near.1 && z0 >= near.2 && z1 <= near.3
}

/// Storage rectangle of `s` inside the full-res box, `(u0, v0, u1, v1)` exclusive.
/// `None` when that overlap is empty or does not land back inside `s`.
fn overlap_storage(
    s: SectionPos,
    near: (i64, i64, i64, i64),
    across: Option<&super::seam::SeamAcross>,
) -> Option<(i32, i32, i32, i32)> {
    let span = s.span() as i64;
    let (sx0, sz0) = (s.min_x() as i64, s.min_z() as i64);
    let (sx1, sz1) = (sx0 + span, sz0 + span);
    let (hx0, hz0, hx1, hz1) = if let Some(m) = across {
        let (a, c) = m.home_xz(sx0, sz0);
        let (b, d) = m.home_xz(sx1, sz1);
        (a.min(b), c.min(d), a.max(b), c.max(d))
    } else {
        (sx0, sz0, sx1, sz1)
    };
    let ix0 = hx0.max(near.0);
    let iz0 = hz0.max(near.2);
    let ix1 = hx1.min(near.1);
    let iz1 = hz1.min(near.3);
    if ix0 >= ix1 || iz0 >= iz1 {
        return None;
    }
    let (u0, v0, u1, v1) = if let Some(m) = across {
        // Inclusive corners: a reflected exclusive edge is off by one.
        let (a, c) = m.storage_xz(ix0, iz0);
        let (b, d) = m.storage_xz(ix1 - 1, iz1 - 1);
        (
            a.min(b).max(sx0),
            c.min(d).max(sz0),
            (a.max(b) + 1).min(sx1),
            (c.max(d) + 1).min(sz1),
        )
    } else {
        (ix0, iz0, ix1, iz1)
    };
    if u0 >= u1 || v0 >= v1 {
        return None;
    }
    Some((
        i32::try_from(u0).ok()?,
        i32::try_from(v0).ok()?,
        i32::try_from(u1).ok()?,
        i32::try_from(v1).ok()?,
    ))
}

/// The section's storage square lies wholly inside the chart box (`hi` exclusive).
fn inside_xz(s: SectionPos, lo: [i64; 3], hi: [i64; 3]) -> bool {
    let span = s.span() as i64;
    let (x, z) = (s.min_x() as i64, s.min_z() as i64);
    x >= lo[0] && x + span <= hi[0] && z >= lo[2] && z + span <= hi[2]
}

/// The section's storage square meets the chart box.
fn overlaps_xz(s: SectionPos, lo: [i64; 3], hi: [i64; 3]) -> bool {
    let span = s.span() as i64;
    let (x, z) = (s.min_x() as i64, s.min_z() as i64);
    x < hi[0] && x + span > lo[0] && z < hi[2] && z + span > lo[2]
}

/// Wholly-inside pieces of `s`. A tile that crosses an edge is replaced by the largest descendants
/// that do not: the edge is not a multiple of the coarser spans, so dropping the straddler leaves
/// a strip of the chart with nothing drawn. Finest tiles meet a 128-aligned edge and are not split.
fn cover_chart(s: SectionPos, lo: [i64; 3], hi: [i64; 3], out: &mut Vec<SectionPos>) {
    if !overlaps_xz(s, lo, hi) {
        return;
    }
    if inside_xz(s, lo, hi) {
        out.push(s);
        return;
    }
    if s.detail.0 <= super::section::FINEST_DETAIL.0 {
        return;
    }
    for q in super::section::Quadrant::ALL {
        cover_chart(s.child(q), lo, hi, out);
    }
}

/// Pieces of `s` against the full-res box. A tile that crosses the edge is replaced by the largest
/// descendants that do not. Wholly outside pieces stay. Wholly inside pieces stay for the punch.
/// Descent continues past the finest far level: that span is wider than the full-res box, so a tile
/// kept there redraws the whole near field. It stops at detail 0 (span 32), the last grid coarser
/// than a chunk. A tile still crossing the edge stays, so the sliver beside the box is drawn.
fn cover_near(
    s: SectionPos,
    near: (i64, i64, i64, i64),
    across: Option<&super::seam::SeamAcross>,
    out: &mut Vec<SectionPos>,
) {
    let crosses = covers_near(s, near, across) && !inside_near(s, near, across);
    if !crosses || s.detail.0 <= 0 {
        out.push(s);
        return;
    }
    for q in super::section::Quadrant::ALL {
        cover_near(s.child(q), near, across, out);
    }
}

fn section_dist2(s: SectionPos, ex: f64, ez: f64) -> f64 {
    let span = s.span() as f64;
    let dx = s.min_x() as f64 + span * 0.5 - ex;
    let dz = s.min_z() as f64 + span * 0.5 - ez;
    dx * dx + dz * dz
}

fn section_key(s: &SectionPos) -> (u16, u8, crate::ident::Detail, i32, i32) {
    (s.body, s.face as u8, s.detail, s.x, s.z)
}

impl World {
    /// Up face of the streaming centre. +Y until the first resolve, so
    /// pre-stream orders match the historical volume.
    pub(in crate::world) fn live_up(&self) -> Option<Face> {
        if self.stream_up_set { self.stream_up } else { Some(Face::PosY) }
    }

    /// Up face for `center`. An `Open` centre keeps the previous face while it
    /// is within one mesh-box radius (3-D chess of [`ViewVolume`]'s horizontal
    /// radius) of a chunk with that face, so an edge band does not flip the
    /// box every chunk. The first centre, with nothing committed, does not
    /// inherit +Y: `Open` there is isotropic.
    pub(in crate::world) fn resolve_stream_up(&self, center: Coord) -> Option<Face> {
        match self.generator.sky(center) {
            Sky::Axis(face) => Some(face),
            Sky::Open => {
                if !self.stream_up_set {
                    return None;
                }
                let Some(prev) = self.stream_up else {
                    return None;
                };
                let r = self.view.horizontal;
                if r > 0 && self.open_near_face(center, prev, r) {
                    Some(prev)
                } else {
                    None
                }
            }
        }
    }

    fn open_near_face(&self, center: Coord, face: Face, r: i32) -> bool {
        let want = Sky::Axis(face);
        for dx in -r..=r {
            for dy in -r..=r {
                for dz in -r..=r {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let c = Coord::new(center.x + dx, center.y + dy, center.z + dz);
                    if self.generator.sky(c) == want {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// The mesh box: chunks meshed and drawn around `center`.
    fn mesh_box(&self, center: Coord) -> ChunkBox {
        self.view.mesh(center, self.live_up())
    }

    /// The data box: the mesh box plus one [`DATA_MARGIN`] shell of voxel data,
    /// so edge chunks can cull against neighbours that are loaded but unmeshed.
    fn data_box(&self, center: Coord) -> ChunkBox {
        self.view.data(center, self.live_up())
    }

    /// The unload box: the mesh box plus the unload hysteresis, past which
    /// chunks are freed.
    pub(in crate::world) fn unload_box(&self, center: Coord) -> ChunkBox {
        self.view.unload(center, self.live_up())
    }

    /// Whether `coord` is inside the current mesh box. The single mesh-view
    /// check: the enqueue gate (the [`MeshLane`] ready predicate) and the
    /// apply gate ([`mesh_result_applies`](Self::mesh_result_applies)) both call
    /// this, so a chunk is enqueued only if its result would be accepted.
    /// `false` before the first stream (no centre yet).
    pub(in crate::world) fn in_mesh_box(&self, coord: Coord) -> bool {
        self.center
            .is_some_and(|c| self.view_contains(self.mesh_box(c), coord))
    }

    /// Whether `coord` lies in view box `b` once the charts around a storage centre are unfolded
    /// into one net (SPACE-ARCHITECTURE §7); plain containment elsewhere.
    #[inline]
    pub(in crate::world) fn view_contains(&self, b: ChunkBox, coord: Coord) -> bool {
        b.contains(self.fold.fold(coord))
    }

    /// The real chunks of view box `b`: across the unfolded chart net around a storage centre
    /// (storage that holds nothing is skipped), `b` itself elsewhere.
    pub(in crate::world) fn view_coords(&self, b: ChunkBox) -> impl Iterator<Item = Coord> + use<> {
        let fold = self.fold;
        b.coords().filter_map(move |v| fold.unfold(v))
    }

    /// The real chunks of view box `b` that view box `other` does not hold. Both boxes are in the
    /// current net, and a real chunk folds back to the net cell it unfolded from.
    pub(in crate::world) fn view_shell(&self, b: ChunkBox, other: ChunkBox) -> impl Iterator<Item = Coord> + use<> {
        let fold = self.fold;
        b.coords_outside(other).filter_map(move |v| {
            let c = fold.unfold(v)?;
            // The walk skips `other` by net cell: a chunk folding elsewhere would be misjudged.
            debug_assert_eq!(fold.fold(c), v, "net cell {v:?} unfolds to {c:?}, which folds elsewhere");
            Some(c)
        })
    }

    /// Upload placement of chunk `coord`'s meshes: a storage chunk of a round world is drawn bent
    /// through its chart cage (made once per loaded chunk; corners relative to an anchor block, so
    /// they stay precise in `f32`), every other chunk at its integer origin.
    fn placement_of(&mut self, coord: Coord, eng: &mut Engine) -> voxel_engine::MeshPlacement {
        if let Some(&cage) = self.cages.get(&coord) {
            return voxel_engine::MeshPlacement::caged(cage, crate::ident::Detail::FULL);
        }
        let Some(corners) = self.seams.cage(coord) else { return chunk_placement(coord) };
        let a = corners[0].floor();
        let anchor = voxel_engine::IVec3::new(a.x as i32, a.y as i32, a.z as i32);
        let rel = corners.map(|c| (c - a).as_vec3());
        match eng.create_cage(anchor, rel) {
            Some(cage) => {
                self.cages.insert(coord, cage);
                voxel_engine::MeshPlacement::caged(cage, crate::ident::Detail::FULL)
            }
            None => chunk_placement(coord),
        }
    }

    /// Storage position of an eye on or above a round body, including flight past the stored
    /// top ([`CHART_FLIGHT`], wider than the stream window). `None` on a flat world and away
    /// from every chart.
    pub(crate) fn chart_eye(&self, eye: DVec3) -> Option<DVec3> {
        if self.seams.is_empty() {
            return None;
        }
        let window = (self.view.horizontal.max(self.view.vertical) + super::DATA_MARGIN + 2) as f64
            * CHUNK_SIZE as f64;
        self.seams.storage_eye(eye, window.max(CHART_FLIGHT))
    }

    /// The point streaming stands on: the eye's storage position on (or above) a round world's
    /// chart, else the eye itself.
    pub(crate) fn stream_eye(&self, eye: DVec3) -> DVec3 {
        self.chart_eye(eye).unwrap_or(eye)
    }

    /// The points streaming stands on for physical eye `eye`: the near window's
    /// ([`stream_eye`](Self::stream_eye)) and the far field's, whose altitude it captures.
    pub(in crate::world) fn place_eyes(&mut self, eye: DVec3) -> (DVec3, DVec3) {
        let chart = self.chart_eye(eye);
        let near = chart.unwrap_or(eye);
        let far = if self.lod2 { self.adopt_far_eye(eye, chart) } else { near };
        self.section_eye_y = far.y;
        (near, far)
    }

    /// The far field's eye: the near window's chart eye `chart`, else the chart column under the
    /// nearest round world within [`FAR_FLIGHT`] of its top, else `eye`. Commits the atlas it
    /// stands on and the altitude scale of its rings.
    fn adopt_far_eye(&mut self, eye: DVec3, chart: Option<DVec3>) -> DVec3 {
        let far = chart.or_else(|| self.seams.far_eye(eye, FAR_FLIGHT, self.far_atlas, FAR_HOLD));
        let seat = far.and_then(|p| self.seams.chart_seat(eye_chunk(p)));
        self.far_atlas = seat.map(|s| s.index);
        self.far_scale = match (far, seat) {
            (Some(p), Some(s)) => {
                let ground = self.far_ground(p);
                let h = if ground == i32::MIN { 0.0 } else { (p.y - f64::from(ground)).max(0.0) };
                self.far_scale_at(h, s.radius)
            }
            _ => 0,
        };
        far.unwrap_or(eye)
    }

    /// The generated ground of storage point `p`'s column, kept until the column changes.
    fn far_ground(&mut self, p: DVec3) -> i32 {
        let column = (block_coord(p.x), block_coord(p.z));
        if let Some((c, ground)) = self.far_ground
            && c == column
        {
            return ground;
        }
        let ground = self.generator.surface(Face::PosY, column.0, column.1);
        self.far_ground = Some((column, ground));
        ground
    }

    /// Unit doublings of the chart rings at height `h` over the ground on a chart of datum radius
    /// `radius`: the rings reach [`FAR_VIEW`] times the height, a doubling is dropped only
    /// [`FAR_HOLD`] lower than it was taken, and the coarsest ring's square stays within
    /// [`FAR_CANDIDATES`] (its detail is capped by the chord rule, so only its reach can grow).
    fn far_scale_at(&self, h: f64, radius: i64) -> u8 {
        let Some((cfg, max_d)) = self.chart_pyramid(radius) else { return 0 };
        let outer = f64::from(cfg.outer_m()) / f64::from(1u32 << self.far_scale);
        let span = f64::from(super::section::section_span(crate::ident::Detail(max_d)));
        let rows = |s: u8| 2.0 * ring_reach(outer * f64::from(1u32 << s), span) + 1.0;
        let mut cap = 0u8;
        while cap < 16 && rows(cap + 1).powi(2) <= FAR_CANDIDATES {
            cap += 1;
        }
        let level = |reach: f64| (0..cap).find(|&s| outer * f64::from(1u32 << s) >= reach).unwrap_or(cap);
        self.far_scale.clamp(level(FAR_VIEW * h), level(FAR_VIEW * FAR_HOLD * h))
    }

    /// The section ladder with its unit doubled [`far_scale`](World::far_scale) times (the
    /// ladder itself off charts, where the scale is 0).
    fn far_pyramid(&self) -> pyramid::PyramidCfg {
        let src = &self.section_pyramid;
        let unit = src.unit * (1u32 << self.far_scale) as f32;
        pyramid::PyramidCfg::sections_with(unit, src.levels.get(), src.finest.0 as u8)
    }

    /// The far field's descheduling horizon in metres from its eye: the ladder's outer edge, and on
    /// a chart the half-diagonal of the coarsest ring's square of sections, whose corners reach
    /// far past that edge. The selection is untouched; only the worker gate reads this.
    fn far_horizon(&self) -> f64 {
        let outer = f64::from(self.far_pyramid().outer_m());
        let chart = self.far_atlas.and_then(|i| self.chart_pyramid(self.seams.atlases()[i].radius));
        let Some((cfg, max_d)) = chart else { return outer };
        let span = f64::from(super::section::section_span(crate::ident::Detail(max_d)));
        outer.max(ring_reach(f64::from(cfg.outer_m()), span) * span * std::f64::consts::SQRT_2)
    }

    /// The far field's centre chunk (the streaming centre unless it stands on a chart the near
    /// window has left). `None` before the first stream.
    pub(in crate::world) fn section_center(&self) -> Option<Coord> {
        self.center.map(|c| self.far_center.unwrap_or(c))
    }

    /// Move the far field's centre to `centre`, with the chart net the gate measures far work in.
    /// Returns whether it moved.
    pub(in crate::world) fn set_far_center(&mut self, centre: Coord) -> bool {
        if self.far_center.replace(centre) == Some(centre) {
            return false;
        }
        self.far_fold = self.far_unfold(centre);
        true
    }

    /// The chart net around far-field centre `centre`: the identity off charts.
    fn far_unfold(&self, centre: Coord) -> super::seam::Unfold {
        if self.section_on_chart(centre) { self.seams.unfold_at(centre) } else { super::seam::Unfold::IDENTITY }
    }

    /// The far field standing on chunk `centre`, as the job gate is told it (the net is the cached
    /// one when that is the far centre).
    fn far_view(&self, centre: Coord) -> pipeline::FarView {
        let fold = if self.far_center == Some(centre) { self.far_fold } else { self.far_unfold(centre) };
        pipeline::FarView { x: centre.x, z: centre.z, fold }
    }

    /// Whether far-field centre `center` stands on a round world's chart: in or above a storage
    /// box that is not a warped cube's.
    fn section_on_chart(&self, center: Coord) -> bool {
        let (_, _, in_cube) = self.lod_place(center);
        !in_cube && self.seams.in_column(center)
    }

    /// Adopt the chart net around streaming centre `centre`; returns whether it changed. A new net
    /// drops the previous boxes' diffs (they were measured in the old net) and re-buckets the
    /// worklists.
    pub(in crate::world) fn adopt_fold(&mut self, centre: Coord) -> bool {
        let fold = self.seams.unfold_at(centre);
        if fold == self.fold {
            return false;
        }
        self.fold = fold;
        self.prev_mesh_box = None;
        self.prev_unload_box = None;
        self.mesh_worklist.set_fold(fold);
        self.light_worklist.set_fold(fold);
        if let Some(workers) = self.workers.as_ref() {
            workers.set_fold(fold);
        }
        true
    }

    /// Peek the dirty-remesh hint without consuming it.
    pub(in crate::world) fn dirty_pending(&self) -> bool {
        self.pending_dirty.get()
    }

    /// The every-frame half of streaming: land finished worker results, run
    /// the budgeted uploads, and remesh edited chunks — everything whose
    /// LATENCY the player sees directly. The game runs this every frame no
    /// matter how `stream_hz` throttles [`stream`](Self::stream), so a 15 Hz
    /// Minimum profile still publishes finished terrain the frame it lands
    /// and a mined block still vanishes the same frame it was clicked.
    /// Steady-state cost is a channel poll and two sticky-flag checks.
    /// `eng` is `None` only in headless tests; GPU work panics without it.
    pub fn pump(
        &mut self,
        mut eng: Option<&mut Engine>,
        sched: &mut crate::sched::Scheduler,
        appearance: &dyn BlockAppearance,
    ) {
        self.remesh_stats.drop_stale_this_frame = 0;
        // Palette growth appends new block texture layers before any upload
        // this frame references a new layer.
        if let Some(eng) = eng.as_deref_mut() {
            self.refresh_textures(eng, appearance);
        }
        // Idle: no claim can produce a `Done`, so skip try_recv and the
        // deadline Instant. Dirty-remesh and lod-clip stay (flag checks).
        if self.anything_in_flight() {
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::StreamDrain);
            let drain_lane = self.lanes().drain;
            sched.run_manual(drain_lane, self, eng.as_deref_mut());
        }
        // The synchronous edit remesh: self-gates on `pending_dirty`, so an
        // editless frame pays one flag check and does not need the engine.
        let dirty_lane = self.lanes().dirty_remesh;
        sched.run_manual(dirty_lane, self, eng.as_deref_mut());
        // Fold any settle events into the LOD clip the moment they land.
        self.refresh_lod_clip();
    }

    /// Land worker results, queue generation/meshing, free distant chunks —
    /// the topology half, run at `stream_hz` (or out of band on forced
    /// refreshes). [`pump`](Self::pump) covers the every-frame latency half;
    /// the drain/dirty lanes here are second-run no-ops on a pumped frame.
    /// Steady-state zero cost: one channel poll, lazy unload/generate on boundary cross.
    pub fn stream(
        &mut self,
        center: DVec3,
        mut eng: Option<&mut Engine>,
        sched: &mut crate::sched::Scheduler,
        appearance: &dyn BlockAppearance,
    ) {
        if let Some(eng) = eng.as_deref() {
            let stats = eng.mesh_stats();
            self.gpu_live_slots = stats.live_slots;
            self.slot_ceiling = stats.cpu_cull_max.max(1);
        }
        let stager = eng.as_ref().map(|e| e.mesh_stager());
        let (center_chunk, far_chunk, full_pass, far_moved) = self.begin_stream(center, stager);
        // Each lane creates its own budget window, not shared: lanes run
        // sequentially, so a single frame-start snapshot would starve lanes
        // after the first.
        // Textures, worker results, and edit remeshes land through the ONE
        // owner of that trio — after the centre update above, so a
        // boundary-cross frame's results drain against the live centre, not
        // the one they'd be discarded by. `stream_phase` pumps only on frames
        // the topology pass does not run; this is the pump on stream-due frames.
        self.pump(eng.as_deref_mut(), sched, appearance);
        if full_pass {
            self.unload_far(
                center_chunk,
                eng.as_deref_mut()
                    .expect("unload on a boundary cross needs the engine"),
            );
            self.cross_boundary(center_chunk);
        }
        // Runs every frame to drain a boundary-cross flood across frames;
        // self-gates on `pending_gen` so a settled world pays one flag check.
        // Placed after unload so freed slots can regenerate.
        {
            let gen_lane = self.lanes().generate;
            sched.run_manual(gen_lane, self, None);
        }
        if self.radius_shrunk.take() {
            // Meshes are about to be freed: the settled scan must restart.
            self.lod_clip_shrunk.set();
            // Free meshes between new radius and unload ring (data stays).
            // Air/NeedsMesh own no handle.
            // Ready chunks drop to NeedsMesh; Dirty chunks stay dirty
            // (prev: None) so same-frame dirty pass still remeshes them.
            let keep = self.mesh_box(center_chunk);
            let fold = self.fold;
            for (&coord, loaded) in self.chunks.iter_mut() {
                if keep.contains(fold.fold(coord)) {
                    continue;
                }
                // Ready becomes NeedsMesh; Dirty stays Dirty (prev: None); a
                // NeedsMesh carrying a rebuild's old mesh drops it (keeping
                // its claim truthful — the in-flight result resolves as
                // stale). Same-frame dirty pass remeshes Dirty chunks.
                // Retire frees the old mesh.
                let next = match loaded.state {
                    MeshState::Ready(_) => MeshState::needs_mesh(),
                    MeshState::Dirty { prev: Some(_) } => MeshState::Dirty { prev: None },
                    MeshState::NeedsMesh {
                        building,
                        prev: Some(_),
                    } => MeshState::NeedsMesh {
                        building,
                        prev: None,
                    },
                    _ => continue,
                };
                let stays_dirty = matches!(next, MeshState::Dirty { .. });
                loaded.retire(
                    next,
                    eng.as_deref_mut()
                        .expect("radius shrink frees GPU meshes"),
                );
                // A retired `Dirty` chunk (its drawn mesh just freed) still needs
                // the same-frame dirty pass to remesh it — which only runs when
                // `pending_dirty` is set. Set it explicitly here rather than
                // hoping some other path already did.
                if stays_dirty {
                    self.pending_dirty.set();
                }
            }
        }
        // Light settling: worklist lane. Trivial grids publish synchronously;
        // only the dense surface band reaches the worker pool.
        {
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::StreamLight);
            if self.lighting {
                let light_lane = self.lanes().light_admit;
                sched.run_manual(light_lane, self, None);
                // No level-triggered mesh-lane forcing here: every event that
                // can flip a chunk's mesh-readiness arms `pending_fresh` WITH
                // a seed (`settle_light` seeds self + moved-border neighbours,
                // `store_chunk` seeds self + 6, the light gate re-seeds timed
                // chunks, fail/cancel re-seed). The old unconditional forcing
                // while any light work existed papered over wedged
                // `light_inflight` claims — fixed at the root in `accept_light`
                // — at the cost of a full admission pass every frame of a flood.
            } else {
                // No flood. Edit seeds stay dormant so re-enabling lighting only
                // settles chunks changed while it was off; `light_ready` bypasses
                // this worklist while disabled.
            }
        }
        // Sync dirty remesh (edited chunks, budgeted) then the fresh mesh lane
        // (worklist seeded on load/light-move, O(shell) not a whole-map rescan).
        {
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::StreamMesh);
            let dirty_lane = self.lanes().dirty_remesh;
            sched.run_manual(dirty_lane, self, eng.as_deref_mut());
            // Advance the light-gate degrade timers and keep still-waiting chunks on
            // the worklist (their degrade fires on the clock, which raises no re-seed
            // event) BEFORE the mesh lane reads them.
            self.tick_light_gate();
            // The mesh lane evicts blocked/stale seeds itself (see `admit`),
            // so the worklist stays O(fresh work) with no separate prune here.
            // A deep upload queue pauses the RUN (never `ready` — that would
            // evict the whole worklist with no re-seed event): `pending_fresh`
            // stays raised and admission self-resumes as uploads drain.
            if !self.upload_backlogged() {
                let mesh_lane = self.lanes().mesh_admit;
                sched.run_manual(mesh_lane, self, None);
            }
            // Level-triggered backstop to the edge-triggered degraded clear: once
            // ALL light work is quiescent, any chunk still degraded is owed a
            // remesh that no future light-arrival event will ever deliver (its
            // missing neighbour is already terminal). Promote it to final now.
            self.flush_degraded_terminal();
        }
        // LOD2 section far field: skipped entirely when disabled (zero cost). Visible
        // set rebuilt every pass because sections become Ready asynchronously.
        if self.lod2 {
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::StreamTiles);
            // Update pyramid unit to track the current view distance.
            self.section_pyramid.unit = self.view.lod_unit();
            self.update_lod_face(far_chunk);
            // Until the bake lands, selection uses the worst-case ladder;
            // mip only coarsens, no upward pops during bake.
            let mip_lane = self.lanes().mip;
            sched.run_manual(mip_lane, self, None);
            // Section overlay lane: refresh the edit overlay BEFORE
            // selection/occlusion/material read it (the frontier's error
            // coarsening below already consults it).
            let overlay_lane = self.lanes().section_overlay;
            sched.run_manual(overlay_lane, self, None);
            self.refresh_frontier(far_chunk);
            if full_pass || far_moved {
                self.unload_sections(
                    far_chunk,
                    eng.as_deref_mut()
                        .expect("section unload on a boundary cross needs the engine"),
                );
                self.pending_sections.set();
            }
            // Section dirty-remesh lane: free GPU meshes of edited sections so
            // they re-extract from the updated generator overlay.
            let section_remesh_lane = self.lanes().section_remesh;
            sched.run_manual(section_remesh_lane, self, eng.as_deref_mut());
            // The floor may be full of sections this frontier no longer draws.
            // Unload runs only on a boundary cross, so a still camera never
            // drops them and admission stays refused.
            self.reclaim_blocked_sections(far_chunk, eng.as_deref_mut());
            let section_lane = self.lanes().section_admit;
            sched.run_manual(section_lane, self, None);
            // Section visible-set lane: re-resolve the covering only when an
            // event moved it (upload/unload/free/claim release/frontier or
            // ladder change) or while admission is still pending — the
            // level-triggered backstop that keeps holes re-arming the lane.
            // A converged, still far field pays a flag check, no covering walk.
            if self.section_cover_dirty.take() || self.pending_sections.get() {
                let visible_lane = self.lanes().section_visible;
                sched.run_manual(visible_lane, self, eng.as_deref_mut());
            }
        }
        // Occlusion is derived state, rebuilt here at the `&mut` sync point (never
        // in the `&self` render). The lane self-gates: it rebuilds only when the
        // adaptive gate is active AND an input changed (or it was just
        // activated), and always updates `occlusion_active` (so turning the gate
        // off lets render draw everything). A CPU-bound world pays nothing.
        {
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::StreamOcclusion);
            let occ_lane = self.lanes().occlusion;
            sched.run_manual(occ_lane, self, eng.as_deref_mut());
        }
        // Unloads/boundary crossings above may have shrunk the settled rings;
        // fold them in before this frame renders.
        self.refresh_lod_clip();
        #[cfg(debug_assertions)]
        self.debug_assert_liveness();
    }

    /// The prologue of [`stream`](Self::stream): the near and far eyes, velocity and pacing, the
    /// chart net and up face, the worklist rings, and the worker view. Returns the centre chunk,
    /// the far field's centre chunk, whether this is a full pass (centre, up face or chart net
    /// moved) and whether the far centre moved.
    pub(in crate::world) fn begin_stream(
        &mut self,
        center: DVec3,
        stager: Option<voxel_engine::MeshStager>,
    ) -> (Coord, Coord, bool, bool) {
        // On a round world streaming stands in the chart's storage cells, and the far field on the
        // chart under the eye (also above the near window's reach).
        let (center, far) = self.place_eyes(center);
        // Eye velocity for prediction. Resets to zero on non-finite values, non-positive dt,
        // or teleport-sized gaps, so prediction never fires on garbage input.
        let now = crate::sched::now();
        let (section_vel, pacing_vel, sample_dt) = match self.section_eye_prev {
            Some((prev, t)) => {
                let dt = now.duration_since(t).as_secs_f64();
                let v = (far - prev) / dt;
                let sane =
                    far.is_finite() && dt > 0.0 && dt <= MAX_PREDICT_SAMPLE_GAP && v.is_finite();
                if sane {
                    // Prediction treats >512 m/s as a discontinuity, but the
                    // pacer still sees that finite motion. Sustained extreme
                    // flight therefore sheds load instead of masquerading as
                    // rest; a one-off teleport gets the same safe one-frame
                    // shedding and then a gradual recovery.
                    let prediction = if v.length() <= MAX_PREDICT_SPEED {
                        v
                    } else {
                        DVec3::ZERO
                    };
                    (prediction, v, dt)
                } else {
                    // Implausible motion (teleport, pause, or faster than
                    // MAX_PREDICT_SPEED): zero prediction. Far jobs left behind
                    // by the jump are re-keyed and descheduled by the pool's
                    // per-epoch sync (the boundary cross bumps the view epoch),
                    // so no separate purge is needed here.
                    (DVec3::ZERO, DVec3::ZERO, dt.min(MAX_PREDICT_SAMPLE_GAP))
                }
            }
            None => (DVec3::ZERO, DVec3::ZERO, 0.0),
        };
        self.section_vel = section_vel;
        self.stream_pacer.update(pacing_vel, sample_dt);
        let queued_near = !self.light_worklist.is_empty()
            || !self.mesh_worklist.is_empty()
            || !self.light_inflight.is_empty()
            || !self.light_apply_queue.is_empty();
        self.stream_pacer.set_boost(queued_near, self.last_stream_secs);
        self.last_stream_secs = sample_dt;
        self.light_admitted_last = 0;
        self.section_eye_prev = far.is_finite().then_some((far, now));
        let center_chunk = eye_chunk(center);
        let far_chunk = eye_chunk(far);
        let far_moved = self.set_far_center(far_chunk);
        // Update centre before draining: old centre may be a sentinel, so draining
        // against it would discard all results and regenerate them immediately.
        // An up-face change is the same kind of pass: the box changed shape.
        let prev_center = self.center;
        let center_moved = Some(center_chunk) != self.center;
        let fold_changed = center_moved && self.adopt_fold(center_chunk);
        let up_changed = if center_moved || !self.stream_up_set {
            let up = self.resolve_stream_up(center_chunk);
            let changed = if self.stream_up_set {
                up != self.stream_up
            } else {
                // Pre-stream orders assume +Y. A different first face reshapes.
                up != Some(Face::PosY)
            };
            self.stream_up = up;
            self.stream_up_set = true;
            changed
        } else {
            false
        };
        let full_pass = center_moved || up_changed || fold_changed;
        self.center = Some(center_chunk);
        // Re-bucket worklists around the live centre before any lane (or pump
        // insert) runs. O(n) once per boundary cross; a no-op when the rings,
        // centre, and up face already match.
        if full_pass {
            let up = self.live_up();
            let rings = self.view.worklist_rings(up);
            self.mesh_worklist.fit(center_chunk, rings, up);
            self.light_worklist.fit(center_chunk, rings, up);
        }
        // Publish the live view to the worker pool: queued jobs re-key toward
        // the player's CURRENT position on every view change, and entries left
        // behind by fast movement — far sections included — are descheduled
        // instead of run. The far horizon covers the whole frontier plus the
        // velocity lookahead, so prediction-desired sections survive it.
        let speed3 = self.section_vel.x.hypot(self.section_vel.y).hypot(self.section_vel.z);
        let far_m = self.far_horizon() + speed3 * TAU_STREAM;
        let far_view = self.far_view(far_chunk);
        // Configure the pool before any lane can submit this frame. On the
        // first stream this avoids one permissive/full-capacity burst from a
        // lazily spawned pool before the pacer catches it on the next pass.
        let pacer = self.stream_pacer;
        let velocity = self.section_vel;
        let view_radius = self.view.horizontal;
        let up = self.live_up();
        let workers = self.worker_pool();
        if let Some(stager) = stager {
            workers.set_stager(stager);
        }
        workers.set_view(
            center_chunk.x,
            center_chunk.y,
            center_chunk.z,
            far_view,
            view_radius,
            far_m,
            velocity.x,
            velocity.y,
            velocity.z,
            up,
        );
        let capacity = workers.worker_capacity();
        workers.set_pacing(
            pacer.active_workers(capacity),
            pacer.near_queue_cap(capacity),
        );
        // Crossing a chunk boundary moves the BFS root, so the visible set is stale.
        self.occlusion_dirty.raise(full_pass);
        // The ring geometry is centred on the eye: a boundary cross SHIFTS the
        // settled rings by the move's chess distance across the up axis (a
        // move along that axis, an up-face change, or the first pass restarts
        // the scan) — see `shift_lod_clip`.
        if full_pass {
            if up_changed {
                self.lod_clip_shrunk.set();
            } else {
                self.shift_lod_clip(prev_center, center_chunk);
            }
        }
        (center_chunk, far_chunk, full_pass, far_moved)
    }

    /// The engine-free rest of a full pass, after [`unload_far`](Self::unload_far).
    pub(in crate::world) fn cross_boundary(&mut self, center: Coord) {
        // Stale queued uploads (the trailing edge of fast movement) release
        // in ONE pass here instead of trickling through the drain budget.
        self.prune_upload_queue();
        // Sync-generate the centre only when it is missing and not already
        // claimed: a claimed job is imminent and the previous centre's
        // collision halo still exists.
        self.ensure_data(center);
        self.pending_gen.set();
        // Mesh box moved: re-seed loaded chunks awaiting a mesh that JUST
        // entered it (a build still in flight lands, or re-seeds when its
        // result is stale). Only the shell (new ∖ old) needs probing — a chunk
        // in old ∩ new was either already seeded, or was evicted as
        // blocked, and blocked evictions re-seed through their own events
        // (data arrival, light settle, degrade expiry). O(|shell|) probes
        // instead of the old all-chunks iteration per cross.
        let new_box = self.mesh_box(center);
        let fresh: Vec<Coord> = match self.prev_mesh_box {
            Some(prev) => self.view_shell(new_box, prev).filter(|&c| self.awaits_mesh(c)).collect(),
            None => self.view_coords(new_box).filter(|&c| self.awaits_mesh(c)).collect(),
        };
        self.mesh_worklist.extend(fresh);
        self.pending_fresh.set();
        self.prev_mesh_box = Some(new_box);
    }

    /// Recompute the far-field selection when its inputs moved (see [`SectionFrontierKey`]).
    pub(in crate::world) fn refresh_frontier(&mut self, center: Coord) {
        // ONE selection sweep, retained across passes: unloading, the load
        // lane, and the covering rebuild below all read this cache. The
        // frontier is a pure function of the key's inputs (eye, velocity,
        // ladder, relief-mip readiness), so while they are bit-identical —
        // a still camera — the sweep (grid walk + relief coarsening) is
        // skipped entirely. Edits force a recompute: relief coarsening
        // consults the edit overlay, which the key cannot cheaply cover.
        let (body, face_u8, cu, cv) = match self.section_lod_face {
            Some((b, f)) => {
                let (cu, _, cv) = FaceFrame::new(f).chunk_to_local(center);
                (b, f as u8, cu, cv)
            }
            // A chart has no cube face. The storage centre still has to invalidate the frontier.
            None if self.section_on_chart(center) => (u16::MAX, u8::MAX, center.x, center.z),
            None => (u16::MAX, u8::MAX, 0, 0),
        };
        // A chart reads whole blocks (`storage_eye_block`), so its key is exact on them: an
        // eye that moves within one block keeps the frontier. A cube face reads the exact eye;
        // its velocity is quantised to 0.25 m/s so a continuously changing flight velocity does
        // not recompute the frontier every pass.
        let (eye_y, velocity) = if self.section_on_chart(center) {
            let d = chart_delta(self.section_vel);
            let y = self.section_eye_y;
            (y.round().to_bits(), [d.x.to_bits(), (y + d.y).round().to_bits(), d.z.to_bits()])
        } else {
            let v = (self.section_vel * 4.0).round();
            (self.section_eye_y.to_bits(), [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()])
        };
        let frontier_key = SectionFrontierKey {
            center_xz: [cu, cv],
            center_y: center.y,
            body,
            face: face_u8,
            eye_y,
            velocity,
            vertical: self.view.vertical,
            up: self.live_up(),
            unit: self.far_pyramid().unit.to_bits(),
            finest: self.section_pyramid.finest.0,
            levels: self.section_pyramid.levels.get(),
            step: self.section_pyramid.step(),
            mip_ready: self.section_mip.is_some(),
            allowed: self.sections_allowed() as u32,
        };
        if self.section_frontier_key != Some(frontier_key) || !self.dirty_sections.is_empty() {
            let mut memo = std::mem::take(&mut self.near_bounds);
            self.section_desired = memo.sweep(|memo| self.desired_sections_with(center, memo));
            self.near_bounds = memo;
            self.section_frontier_key = Some(frontier_key);
            self.section_cover_dirty.set();
        }
    }

    /// Land finished worker results (non-blocking). Generate results clear
    /// `generating`; stale results release their exact claims. Result
    /// integration used to drain the unbounded channel in one frame, making a
    /// productive worker burst a main-thread hitch. It now shares the adaptive
    /// effort signal and keeps a small forward-progress floor.
    pub(in crate::world) fn drain_results(&mut self, eng: &mut Engine, result_budget: Duration) {
        self.integrate_results(result_budget);
        self.section_upload_bytes = 0;
        self.drain_upload_bytes = 0;
        if self.upload_queue.is_empty()
            && self.section_upload_queue.is_empty()
            && self.light_apply_queue.is_empty()
        {
            return;
        }

        // Budgeted uploads, charged in BYTES (the actual staging cost — see
        // `UPLOAD_BUDGET_BYTES`). A stale entry costs nothing but a bounded
        // pop (`UPLOAD_SCAN_MAX`), so a post-flight queue of stale entries no
        // longer starves real uploads for dozens of frames. The byte check
        // sits at the loop head, so at least one real upload always lands —
        // the same forward-progress floor the admission lanes keep.
        // Re-validate at the moment of upload: an entry may have sat queued
        // across frames while an edit bumped the chunk's rev.
        let pacer = self.stream_pacer;
        let upload_budget = pacer.upload_bytes();
        let mut upload_bytes = 0usize;
        let mut uploads = 0usize;
        let mut pops = 0usize;
        while (uploads == 0 || upload_bytes < upload_budget) && pops < UPLOAD_SCAN_MAX {
            let Some((coord, rev, data)) = self.upload_queue.pop_front() else {
                break;
            };
            pops += 1;
            if !self.mesh_result_applies(coord, rev) {
                // Stale while queued: edit made it Dirty or it left the box.
                data.release_staging(eng);
                self.drop_stale_upload(coord);
                continue;
            }
            upload_bytes += mesh_output_bytes(&data);
            uploads += 1;
            // Both passes upload together under one budget charge (same rev).
            // Staged payloads install through the worker-written ring; the
            // Vec fallback uses the existing main-thread copy. (The rev
            // check above guarantees the state is NeedsMesh { building: true }.)
            self.upload_chunk_payload(coord, data, eng);
            // A newly drawn chunk may complete a settled ring.
            self.lod_clip_grow.set();
        }

        self.apply_light_queue();

        // Section uploads share the chunk byte counter. A section is binary
        // (`SectionState` Ready-or-not), so the byte gate sits before the pop:
        // the next whole tile lands only while the counter has room. The count
        // cap stays as a secondary ceiling. Re-validated by claim token at the
        // moment of upload: an entry that sat queued across an unload or a
        // re-admission must not capture the replacement claim.
        let section_budget = pacer.section_uploads();
        let mut section_uploads = 0;
        while section_uploads < section_budget {
            let Some((_, _, _, _)) = self.section_upload_queue.front() else {
                break;
            };
            if upload_bytes >= upload_budget {
                break;
            }
            let (pos, token, bytes, meshes) = self
                .section_upload_queue
                .pop_front()
                .expect("front was Some");
            section_uploads += 1;
            // `section_material` borrows all of `self`, so it must run before
            // `self.sections.get_mut` below takes an overlapping mutable borrow.
            let (flat_color, flat_rgba) = self.section_material(pos);
            let bend = self.chart_bend(pos);
            if let Some(state @ SectionState::Meshing { .. }) = self.sections.get_mut(&pos)
                && matches!(state, SectionState::Meshing { token: t } if *t == token)
            {
                super::adjust_count(&mut self.meshing_sections, true, false);
                upload_bytes += bytes;
                self.section_upload_bytes += bytes;
                *state = SectionState::from_upload_payload(pos, meshes, eng, bend.as_ref());
                // Slots are born visible (residency implies it for everything but the
                // far field), so a section that Coverage does not draw — or draws only
                // in part — must be corrected here, at the transition that gave it slots
                // to correct. No frame intervenes: patches flush at submit.
                state.set_visible(eng, self.section_fade.drawn_mask(pos));
                // Push the section's far-material style so it doesn't draw one frame at
                // the engine's post-upload default.
                state.push_style(eng, FadeStyle { flat_color }, flat_rgba);
                // A new Ready section moves the covering: re-arm the lane so
                // any refinement it exposes loads immediately.
                self.pending_sections.set();
                self.section_cover_dirty.set();
            } else {
                meshes.release_staging(eng);
            }
        }
        self.drain_upload_bytes = upload_bytes;
    }

    /// The first block of [`drain_results`](Self::drain_results): integrate finished worker
    /// results within the paced `budget`, past a small forward-progress floor.
    pub(in crate::world) fn integrate_results(&mut self, budget: Duration) {
        let pacer = self.stream_pacer;
        let deadline = pipeline::Deadline::from_budget(pacer.duration(budget));
        let floor = pacer.floor(RESULT_INTEGRATE_FLOOR);
        let mut integrated = 0usize;
        while integrated < floor || !deadline.expired() {
            let Some(result) = self.workers.as_ref().and_then(pipeline::Workers::try_recv) else {
                break;
            };
            self.integrate_worker_result(result);
            integrated += 1;
        }
    }

    /// Budgeted light application, after the chunk uploads of
    /// [`drain_results`](Self::drain_results). Order-independent: each grid is absolute,
    /// leftovers apply next frame with no seam.
    pub(in crate::world) fn apply_light_queue(&mut self) {
        let deadline =
            pipeline::Deadline::from_budget(self.stream_pacer.duration(pipeline::LIGHT_APPLY_BUDGET));
        let mut applied = 0usize;
        while applied == 0 || !deadline.expired() {
            let Some((coord, grid)) = self.light_apply_queue.pop_front() else {
                break;
            };
            self.settle_light(coord, grid);
            applied += 1;
        }
    }

    /// Route one completed worker payload through its owning lane. This is the
    /// claim-resolution chokepoint: every accepted claim is owed exactly one
    /// payload, cancellation, or failure, and consuming it must release or
    /// transfer that claim even when the result became stale in flight.
    fn integrate_worker_result(&mut self, result: pipeline::Done) {
        #[cfg(debug_assertions)]
        let light_audit = match &result {
            pipeline::Done::Light { coord, epoch, light_gen, .. } => {
                Some((*coord, *epoch, *light_gen))
            }
            _ => None,
        };
        #[cfg(debug_assertions)]
        let section_audit = match &result {
            pipeline::Done::Section {
                pos, epoch, token, ..
            } => Some((*pos, *epoch, *token)),
            _ => None,
        };
        match result {
            pipeline::Done::Column { key, chunks, heights } => {
                self.accept_column(key, chunks, heights)
            }
            m @ pipeline::Done::Mesh { .. } => MeshLane::integrate(self, m),
            l @ pipeline::Done::Light { .. } => LightLane::integrate(self, l),
            sc @ pipeline::Done::Section { .. } => SectionLane::integrate(self, sc),
            pipeline::Done::Failed(key) => self.fail_job(*key),
            pipeline::Done::Cancelled(keys) => {
                for key in keys {
                    self.cancel_job(key);
                }
            }
        }
        // A consumed CURRENT-epoch light result must have released its claim
        // or transferred it into the apply queue.
        #[cfg(debug_assertions)]
        if let Some((coord, epoch, light_gen)) = light_audit {
            let live = self.chunks.get(&coord).map(|l| l.light_gen) == Some(light_gen);
            debug_assert!(
                epoch != self.light_epoch
                    || !live
                    || !self.light_inflight.contains(&coord)
                    || self.light_apply_queue.iter().any(|(c, _)| *c == coord),
                "light Done for {coord:?} left its claim neither released nor transferred"
            );
        }
        // A current section result matching the live token must likewise have
        // transferred to the upload queue.
        #[cfg(debug_assertions)]
        if let Some((pos, epoch, token)) = section_audit {
            debug_assert!(
                epoch != self.section_epoch
                    || !matches!(self.sections.get(&pos),
                        Some(SectionState::Meshing { token: t }) if *t == token)
                    || self
                        .section_upload_queue
                        .iter()
                        .any(|(p, t, _, _)| *p == pos && *t == token),
                "section Done for {pos:?} matched the live claim but was not transferred"
            );
        }
    }

    /// Generated chunk result: discard if out-of-range/already loaded; else store (replays edits).
    pub(in crate::world) fn accept_chunk(&mut self, coord: Coord, chunk: Chunk) {
        if !self.will_accept_chunk(coord) {
            return;
        }
        self.store_chunk(coord, chunk);
    }

    /// Mesh result at `rev`: queue for upload if still applies; else drop and re-arm scan.
    /// Staged regions release on drop of a rejected payload.
    pub(in crate::world) fn accept_mesh(
        &mut self,
        coord: Coord,
        rev: u32,
        data: impl Into<pipeline::MeshPayload>,
    ) {
        let data = data.into();
        if self.mesh_result_applies(coord, rev) {
            self.upload_queue.push_back((coord, rev, data));
        } else {
            // Stale: chunk edited (Dirty) or left box. Staging Drop releases.
            self.drop_stale_upload(coord);
        }
    }

    /// Release a stale mesh result's build claim and re-seed the coord so it
    /// can mesh again later — the one stale-drop path, shared by the accept
    /// site, the pop-time re-validation, and the boundary-cross prune. A chunk
    /// an edit made `Dirty` belongs to the dirty lane and takes no seed.
    fn drop_stale_upload(&mut self, coord: Coord) {
        self.remesh_stats.note_drop_stale();
        // An unloaded chunk is not re-seeded: its next load seeds it.
        let Some(loaded) = self.chunks.get_mut(&coord) else { return };
        if loaded.state.release_build() {
            super::adjust_count(&mut self.building_meshes, true, false);
        }
        self.pending_fresh.set();
        self.seed_mesh(coord);
    }

    /// One-pass prune of stale upload entries (boundary cross): each is
    /// released and re-seeded exactly like the pop-time stale path — without
    /// letting a deep post-flight backlog of left-behind meshes trickle out
    /// at drain speed while real uploads wait behind it.
    pub(in crate::world) fn prune_upload_queue(&mut self) {
        if self.upload_queue.is_empty() {
            return;
        }
        let mut queue = std::mem::take(&mut self.upload_queue);
        let mut stale: Vec<Coord> = Vec::new();
        queue.retain(|(coord, rev, _)| {
            let live = self.mesh_result_applies(*coord, *rev);
            if !live {
                stale.push(*coord);
            }
            live
        });
        self.upload_queue = queue;
        for coord in stale {
            self.drop_stale_upload(coord);
        }
    }

    /// Whether mesh admission should pause this pass (see [`UPLOAD_QUEUE_MAX`]).
    pub(in crate::world) fn upload_backlogged(&self) -> bool {
        self.upload_queue.len() >= UPLOAD_QUEUE_MAX
    }

    /// Upload one built chunk mesh (every pass) and retire the chunk's state
    /// to the fresh `Ready`/`Air` — the one upload+install step shared by the
    /// async drain, the sync edit remesh, and the terminal degraded promotion.
    /// `retire` frees whatever the old state carried, exactly once.
    /// `hash` is `Some` only on the sync edit path; async passes `None` and
    /// clears `mesh_hash` (no `content_hash` on the main thread).
    fn upload_chunk(
        &mut self,
        coord: Coord,
        data: &mesh::ChunkMeshData,
        hash: Option<u64>,
        eng: &mut Engine,
    ) {
        self.upload_chunk_inner(coord, Some((data, eng)), hash);
    }

    /// Worker payload: staged regions install through the ring; the `Vec`
    /// fallback uses the existing main-thread copy. Async, so `mesh_hash` is
    /// cleared (`None`).
    fn upload_chunk_payload(
        &mut self,
        coord: Coord,
        data: pipeline::MeshPayload,
        eng: &mut Engine,
    ) {
        match data {
            pipeline::MeshPayload::Cpu(data) => {
                self.upload_chunk(coord, &data, None, eng);
            }
            pipeline::MeshPayload::Staged(mut staged) => {
                let placement = self.placement_of(coord, eng);
                let handles = ByPass::from_fn(|p| {
                    staged.passes[p].take().and_then(|pass| {
                        eng.upload_mesh_staged(pass.staging, pass.quad_counts, p, placement)
                    })
                });
                self.install_chunk_handles(coord, handles, None, eng);
            }
        }
    }

    fn upload_chunk_inner(
        &mut self,
        coord: Coord,
        gpu: Option<(&mesh::ChunkMeshData, &mut Engine)>,
        hash: Option<u64>,
    ) {
        if let Some(h) = hash
            && self.chunks.get(&coord).is_some_and(|l| l.mesh_hash == Some(h))
        {
            self.keep_resident_mesh(coord, gpu.map(|(_, eng)| eng));
            return;
        }
        if let Some((data, eng)) = gpu {
            let placement = self.placement_of(coord, eng);
            let handles = ByPass::from_fn(|p| eng.upload_mesh_placed(&data[p], placement));
            self.install_chunk_handles(coord, handles, hash, eng);
            return;
        }
        if let Some(loaded) = self.chunks.get_mut(&coord) {
            loaded.mesh_hash = hash;
        }
    }

    fn install_chunk_handles(
        &mut self,
        coord: Coord,
        handles: ByPass<Option<voxel_engine::MeshHandle>>,
        hash: Option<u64>,
        eng: &mut Engine,
    ) {
        let vis = !self.occlusion_active || self.occlusion.is_visible(coord);
        if let Some(loaded) = self.chunks.get_mut(&coord) {
            let was = loaded.state.is_building();
            loaded.retire(MeshState::from_upload(handles), eng);
            loaded.mesh_hash = hash;
            super::adjust_count(&mut self.building_meshes, was, false);
            loaded.visible = vis;
            if !vis && let Some(meshes) = loaded.state.live_meshes() {
                meshes.set_visible(eng, false);
            }
        }
    }

    /// An edit remesh whose vertex bytes match the resident GPU mesh: keep the
    /// existing handles and drop the Dirty/NeedsMesh claim, no upload.
    fn keep_resident_mesh(&mut self, coord: Coord, eng: Option<&mut Engine>) {
        let vis = !self.occlusion_active || self.occlusion.is_visible(coord);
        if let Some(loaded) = self.chunks.get_mut(&coord) {
            let was = loaded.state.is_building();
            let next = match std::mem::replace(&mut loaded.state, MeshState::Air) {
                MeshState::Ready(m)
                | MeshState::Dirty { prev: Some(m) }
                | MeshState::NeedsMesh { prev: Some(m), .. } => MeshState::Ready(m),
                MeshState::Dirty { prev: None }
                | MeshState::NeedsMesh { prev: None, .. }
                | MeshState::Air => MeshState::Air,
            };
            loaded.state = next;
            super::adjust_count(&mut self.building_meshes, was, false);
            self.remesh_stats.note_upload(coord);
            if vis != loaded.visible {
                loaded.visible = vis;
                if let Some(meshes) = loaded.state.live_meshes() {
                    #[cfg(test)]
                    super::vis_log::record(meshes.handles(), vis);
                    if let Some(eng) = eng {
                        meshes.set_visible(eng, vis);
                    }
                }
            }
        }
    }

    #[cfg(test)]
    fn upload_chunk_without_gpu(&mut self, coord: Coord, hash: Option<u64>) {
        self.upload_chunk_inner(coord, None, hash);
    }

    /// Light result at `epoch`: release-or-transfer the claim, then queue the
    /// grid if it still applies. The claim rule at this consumption site: a
    /// claimed key is owed exactly one `Done`, and consuming that `Done` must
    /// release or transfer the claim — silently dropping a result used to
    /// wedge its coord's `light_inflight` entry forever (a chunk re-loaded at
    /// that coord could never settle light again: `light_ready` read the
    /// stale claim as still-in-flight, the mesh lane skipped it as in-flight,
    /// and quiescence — degraded promotion, `entry_complete` — never came).
    ///
    /// Epoch and generation reasoning (what makes the release sound):
    /// [`transition_lighting`](Self::transition_lighting) is the only
    /// `light_epoch` bump and it clears `light_inflight` in the same breath, so
    /// - a CURRENT-epoch result is the unique owner of any in-flight entry at
    ///   its coord (releasing can never steal a newer claim), while
    /// - a STALE-epoch result's claim was already wiped at the bump — an entry
    ///   present now belongs to a post-bump job and must not be touched.
    /// `light_gen` is the per-`Loaded` stamp: unload then regenerate at the same
    /// coord does not bump the epoch, so a current-epoch result for the *old*
    /// resident must not publish onto the new voxels (and store skips trivial
    /// settle while the old claim is still in flight, so this Done remains
    /// that claim's unique owner).
    pub(in crate::world) fn accept_light(
        &mut self,
        coord: Coord,
        epoch: u32,
        light_gen: u32,
        grid: light::LightGrid,
    ) {
        if epoch != self.light_epoch {
            return;
        }
        let live_gen = self.chunks.get(&coord).map(|l| l.light_gen);
        if live_gen != Some(light_gen) {
            // Unusable: unloaded, or a later Loaded at this coord. Release
            // the leftover claim and re-seed the new resident so it can
            // settle against its own voxels.
            self.light_inflight.remove(&coord);
            if live_gen.is_some() && self.lighting {
                self.seed_light(coord, super::LightSeed::Store);
                self.light_pending.set();
            }
            return;
        }
        if !self.lighting {
            self.light_inflight.remove(&coord);
            return;
        }
        // TRANSFER: the claim stays held through the apply queue (so
        // `light_ready` keeps treating the chunk as unsettled);
        // [`settle_light`](Self::settle_light) is the release point.
        self.light_apply_queue.push_back((coord, grid));
    }

    /// Mesh result still applies: chunk loaded, in view range, rev not bumped.
    pub(in crate::world) fn mesh_result_applies(&self, coord: Coord, rev: u32) -> bool {
        self.in_mesh_box(coord) && self.chunks.get(&coord).is_some_and(|l| l.rev == rev)
    }

    /// Streaming priority. `Some(face)`: tangent chess, distance along the
    /// face ×2 (terrain before sky). `None`: plain 3-D chess. +Y is
    /// `ring.max(2 * updown)`.
    pub(in crate::world) fn order(a: Coord, b: Coord, up: Option<Face>) -> i32 {
        match up {
            None => a.chess3(b),
            Some(face) => a.across(b, face).max(2 * a.along(b, face)),
        }
    }

    /// The [`GenerateLane`](lanes::GenerateLane) producer's body: queue worker
    /// jobs for missing chunks in the data box, grouped into vertical columns so
    /// the `cy`-invariant column profile is sampled once per column, nearest
    /// column first, up to `deadline`. Self-gates on `pending_gen` (raised on a
    /// boundary cross and by a generate strike-out re-request). Column
    /// granularity means it keeps its own gather/claim rather than the per-chunk
    /// [`admit`](super::admit) loop, but the forward-progress floor + time budget
    /// are the one shared rule ([`admission_exhausted`](super::admission_exhausted)).
    /// The centre chunk is generated synchronously in `stream` only when it is
    /// missing and not already claimed; `accept_column` lands these results.
    ///
    /// The run list is gathered once per data box and drained across frames.
    /// Walking every coord each pass (classify, then sky) blows the 2 ms budget
    /// before any job is submitted, so an open asteroid admits only the floor
    /// and spends the frame on the walk.
    pub(in crate::world) fn request_region_data(
        &mut self,
        center: Coord,
        budget: Budget,
    ) -> Progress {
        if !self.pending_gen.take() {
            return Progress::Idle;
        }
        if !self.gen_cursor_matches(center) {
            self.rebuild_gen_cursor(center);
        }
        if self.gen_columns.is_empty() {
            return Progress::Idle;
        }
        let slots = match self.workers.as_ref() {
            Some(w) => w.near_slots_free(),
            None => usize::MAX,
        };
        if slots == 0 {
            self.pending_gen.set();
            return Progress::Partial {
                remaining: self.gen_columns.len() as u32,
            };
        }
        let deadline = super::lanes::paced_deadline(self, budget);
        let vel = self.section_vel;
        if !self.gen_cursor_ranked || self.gen_cursor_vel != vel {
            let up = self.live_up();
            let fold = self.fold;
            for entry in &mut self.gen_columns {
                let anchor = entry.1.anchor();
                entry.0 = column_order(center, vel, fold.fold(anchor), up);
            }
            self.gen_columns.sort_by_key(|e| e.0);
            self.gen_cursor_vel = vel;
            self.gen_cursor_ranked = true;
        }
        let min_admit = self.stream_pacer.floor(GEN_MIN_ADMIT);
        let mut admitted = 0usize;
        let mut consumed = 0usize;
        while consumed < self.gen_columns.len() {
            if super::admission_exhausted(admitted, min_admit, deadline) {
                break;
            }
            let run = self.gen_columns[consumed].1;
            if self.run_quarantined(run) || self.run_covered(run) {
                consumed += 1;
                continue;
            }
            let accepted = match run {
                GenRun::Column { key, lo, hi } => self.try_submit_column(key, lo, hi),
                GenRun::Open { coord } => self.try_submit_open(coord),
            };
            if accepted {
                admitted += 1;
                consumed += 1;
            } else {
                break;
            }
        }
        self.gen_columns.drain(..consumed);
        if self.gen_columns.is_empty() {
            Progress::Idle
        } else {
            self.pending_gen.set();
            Progress::Partial {
                remaining: self.gen_columns.len() as u32,
            }
        }
    }

    /// Whether `gen_columns` is still the run list for `center`'s data box.
    fn gen_cursor_matches(&self, center: Coord) -> bool {
        if self.gen_cursor_dirty || self.gen_cursor_center != Some(center) {
            return false;
        }
        if self.gen_cursor_up != self.live_up()
            || self.gen_cursor_h != self.view.horizontal
            || self.gen_cursor_v != self.view.vertical
        {
            return false;
        }
        match (self.gen_cursor_slab, self.spawn_slab) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                a.center == b.center && a.rh == b.rh && a.rv == b.rv && a.up == b.up
            }
            _ => false,
        }
    }

    /// Classify the data box once, store uniform chunks, and queue the rest.
    fn rebuild_gen_cursor(&mut self, center: Coord) {
        let mut coords: Vec<Coord> = self.view_coords(self.data_box(center)).collect();
        if let Some(slab) = self.spawn_slab {
            coords.extend(self.view_coords(slab));
        }
        let mut stored = false;
        coords.retain(|c| {
            if self.store_if_free(*c) {
                stored = true;
                false
            } else {
                true
            }
        });
        if stored {
            self.refresh_spawn_slab();
        }
        let runs = gather_column_runs(
            coords,
            |c| self.generator.sky(c),
            |c| self.chunks.contains_key(&c) || self.generating.contains(&c),
            |fail| self.quarantined.contains(&fail),
            false,
            true,
        );
        self.gen_columns.clear();
        self.gen_columns.extend(runs.into_iter().map(|run| (0, run)));
        self.gen_cursor_center = Some(center);
        self.gen_cursor_up = self.live_up();
        self.gen_cursor_h = self.view.horizontal;
        self.gen_cursor_v = self.view.vertical;
        self.gen_cursor_slab = self.spawn_slab;
        self.gen_cursor_dirty = false;
        self.gen_cursor_ranked = false;
    }

    /// A quarantined run is dropped, matching `gather_column_runs`'s
    /// `skip_quarantine`. Pool backpressure is a different `false` from submit
    /// and must leave the run queued.
    fn run_quarantined(&self, run: GenRun) -> bool {
        let key = match run {
            GenRun::Open { coord } => FailKey::Open { coord },
            GenRun::Column { key, .. } => FailKey::Column { key },
        };
        self.quarantined.contains(&key)
    }

    /// Every chunk of `run` is loaded or already claimed.
    fn run_covered(&self, run: GenRun) -> bool {
        match run {
            GenRun::Open { coord } => {
                self.chunks.contains_key(&coord) || self.generating.contains(&coord)
            }
            GenRun::Column { key, lo, hi } => (lo..=hi).all(|alt| {
                let coord = key.chunk(alt);
                self.chunks.contains_key(&coord) || self.generating.contains(&coord)
            }),
        }
    }

    /// Land a generated column: install the skylight ceiling from the worker's
    /// heights (plus any edited-roof raise) before storing, so `trivial_light`
    /// hits the cache instead of sampling the generator on this thread.
    pub(in crate::world) fn accept_column(
        &mut self,
        key: ColumnKey,
        chunks: Vec<(Coord, Chunk)>,
        heights: Box<ColumnHeights>,
    ) {
        // Only cache when at least one axis chunk will actually land — an install
        // with no `column_chunks` bump would leak in `ceilings` forever. Open
        // runs encode a PosY key and must not install a ceiling.
        if chunks.iter().any(|(coord, _)| {
            self.will_accept_chunk(*coord) && matches!(self.generator.sky(*coord), Sky::Axis(_))
        }) {
            self.install_ceiling(key, &heights);
        }
        for (coord, chunk) in chunks {
            self.generating.remove(&coord);
            self.accept_chunk(coord, chunk);
        }
        self.refresh_spawn_slab();
    }

    /// `accept_chunk`'s store predicate: in the live data box or the requested
    /// spawn slab, and not yet loaded.
    fn will_accept_chunk(&self, coord: Coord) -> bool {
        if self.chunks.contains_key(&coord) {
            return false;
        }
        self.in_data_or_slab(coord)
    }

    fn in_data_or_slab(&self, coord: Coord) -> bool {
        self.spawn_slab.is_some_and(|slab| self.view_contains(slab, coord))
            || self
                .center
                .is_some_and(|center| self.view_contains(self.data_box(center), coord))
    }

    /// A queued job was DESCHEDULED at the pool: its region left the live view
    /// while it waited (fast movement). Release the exact claim with no strike
    /// and no requeue — the work is unwanted where the player is now, and the
    /// boundary-cross scans re-request it if the player ever returns. (The one
    /// exception: a still-loaded chunk is OWED its light settle, so light
    /// claims re-seed — the cancel ring sits outside the unload ring, so this
    /// is rare.)
    pub(in crate::world) fn cancel_job(&mut self, key: pipeline::JobKey) {
        self.resolve_claim(key, ClaimOutcome::Cancelled);
    }

    /// A worker job PANICKED: release its exact claim so streaming can
    /// converge, then retry (the normal scans re-request freed work) up to
    /// [`MAX_JOB_STRIKES`] times. Past that the claim is quarantined — a
    /// bounded hole instead of an infinite panic loop — and every enqueue path
    /// skips it via `quarantined`.
    pub(in crate::world) fn fail_job(&mut self, key: pipeline::JobKey) {
        self.resolve_claim(key, ClaimOutcome::Failed);
    }

    /// The ONE payload-less claim-resolution path (cancel and fail shared the
    /// whole per-kind release; only the strike/re-arm policy differed).
    /// RELEASE is unconditional per kind; RE-ARM follows `outcome`: a
    /// cancellation re-arms only the light settle it still owes, a
    /// non-quarantined failure re-arms its lane for the retry.
    fn resolve_claim(&mut self, key: pipeline::JobKey, outcome: ClaimOutcome) {
        let rearm = match outcome {
            ClaimOutcome::Cancelled => matches!(key, pipeline::JobKey::Light { .. }),
            ClaimOutcome::Failed => {
                let fail_key = FailKey::of(&key);
                let strikes = self.job_strikes.entry(fail_key).or_insert(0);
                *strikes = strikes.saturating_add(1);
                let quarantine = *strikes >= MAX_JOB_STRIKES;
                if quarantine {
                    self.quarantined.insert(fail_key);
                    eprintln!(
                        "streaming: {fail_key:?} panicked {MAX_JOB_STRIKES} times — quarantined"
                    );
                }
                !quarantine
            }
        };
        match key {
            pipeline::JobKey::Column { key, range } => {
                // A cancelled spawn-slab column must be re-requested even when
                // the pool dropped it as out-of-view: physics is frozen on it.
                let in_slab = self.spawn_slab.is_some_and(|slab| {
                    range.clone().any(|alt| slab.contains(key.chunk(alt)))
                });
                for alt in range {
                    self.generating.remove(&key.chunk(alt));
                }
                // Freed generate claims are otherwise only re-requested on a
                // boundary cross; a retryable failure re-arms the lane so a
                // standing-still player still converges.
                if rearm || in_slab {
                    self.pending_gen.set();
                    // The run already left the queue when it was submitted.
                    self.gen_cursor_dirty = true;
                }
            }
            pipeline::JobKey::Open { coord } => {
                let in_slab = self.spawn_slab.is_some_and(|slab| slab.contains(coord));
                self.generating.remove(&coord);
                if rearm || in_slab {
                    self.pending_gen.set();
                    self.gen_cursor_dirty = true;
                }
            }
            pipeline::JobKey::Mesh { coord } => {
                if let Some(loaded) = self.chunks.get_mut(&coord) {
                    if loaded.state.release_build() {
                        super::adjust_count(&mut self.building_meshes, true, false);
                    }
                }
                if rearm {
                    self.seed_mesh(coord);
                    self.pending_fresh.set();
                }
            }
            pipeline::JobKey::Light { coord } => {
                self.light_inflight.remove(&coord);
                if rearm {
                    // An unloaded chunk's seed is dropped by the lane's submit.
                    self.seed_light(coord, super::LightSeed::Store);
                    self.light_pending.set();
                }
                // Quarantined light: the chunk never settles, so the mesh
                // lane's degrade timeout takes over and the terminal flush
                // promotes it — the world converges on fallback light.
            }
            pipeline::JobKey::Section { pos, epoch, token } => {
                // Release only the exact claim: a same-position replacement
                // minted after this job was queued keeps its own claim.
                let held = epoch == self.section_epoch
                    && matches!(self.sections.get(&pos),
                        Some(SectionState::Meshing { token: t }) if *t == token);
                if held {
                    self.sections.remove(&pos);
                    super::adjust_count(&mut self.meshing_sections, true, false);
                    self.section_cover_dirty.set();
                }
                if rearm {
                    self.pending_sections.set();
                }
            }
        }
    }

    /// Ensure every chunk within the data box of `center` exists (voxel data
    /// only). Cheap and GPU-free, so it also seeds headless queries.
    pub(in crate::world) fn ensure_region_data(&mut self, center: Coord) {
        let coords: Vec<Coord> = self.view_coords(self.data_box(center)).collect();
        for coord in coords {
            self.ensure_data(coord);
        }
    }

    /// Collision halo around an eye chunk. `Some(face)`: ±1 across the face,
    /// ±2 along it (+Y is the old 3×3 columns, two layers below through two
    /// above). `None`: ±2 on every axis.
    pub(in crate::world) fn collision_slab(center: Coord, up: Option<Face>) -> ChunkBox {
        match up {
            None => ChunkBox::with_up(center, 2, 2, None),
            Some(face) => ChunkBox::with_up(center, 1, 2, Some(face)),
        }
    }

    fn slab_up(&self, center: Coord) -> Option<Face> {
        match self.generator.sky(center) {
            Sky::Axis(face) => Some(face),
            Sky::Open => None,
        }
    }

    /// Request the collision slab around `pos` from the worker pool. Does not
    /// generate on this thread — [`spawn_ready`](Self::spawn_ready) is true
    /// once every chunk of the box has loaded. Teleports and net snaps use
    /// the same request (physics freezes until it lands).
    pub fn prepare_around(&mut self, pos: DVec3) {
        let (near, far) = self.place_eyes(pos);
        let (c, f) = (eye_chunk(near), eye_chunk(far));
        self.adopt_fold(c);
        let up = self.slab_up(c);
        let slab = Self::collision_slab(c, up);
        let (far_m, far_view) = (self.far_horizon(), self.far_view(f));
        let view_r = self.view.horizontal;
        self.worker_pool()
            .set_view(c.x, c.y, c.z, far_view, view_r, far_m, 0.0, 0.0, 0.0, up);
        self.submit_slab_columns(slab);
        self.pending_gen.set();
        if self.view_coords(slab).all(|coord| self.chunks.contains_key(&coord)) {
            self.spawn_slab = None;
        } else {
            self.spawn_slab = Some(slab);
        }
    }

    /// Synchronously generate the collision slab. Headless callers (tests,
    /// anything that queries voxels before a stream pass).
    pub fn ensure_around(&mut self, pos: DVec3) {
        let c = eye_chunk(self.stream_eye(pos));
        self.adopt_fold(c);
        let coords: Vec<Coord> = self.view_coords(Self::collision_slab(c, self.slab_up(c))).collect();
        for coord in coords {
            self.ensure_data(coord);
        }
    }

    /// True once every chunk of the requested spawn/teleport slab is loaded,
    /// or no slab is outstanding.
    pub fn spawn_ready(&self) -> bool {
        match self.spawn_slab {
            None => true,
            Some(slab) => self.view_coords(slab).all(|c| self.chunks.contains_key(&c)),
        }
    }

    /// Drive in-flight generate jobs until the spawn slab is loaded. Tests
    /// only — the live path drains through [`stream`](Self::stream).
    #[cfg(test)]
    pub fn drive_spawn_ready(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while !self.spawn_ready() {
            assert!(
                Instant::now() < deadline,
                "spawn slab did not land: {}",
                self.entry_debug()
            );
            if let Some(slab) = self.spawn_slab {
                self.submit_slab_columns(slab);
            }
            let mut got = false;
            while let Some(done) = self.workers.as_ref().and_then(pipeline::Workers::try_recv) {
                self.integrate_worker_result(done);
                got = true;
            }
            self.refresh_spawn_slab();
            if !got {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }

    fn refresh_spawn_slab(&mut self) {
        let Some(slab) = self.spawn_slab else {
            return;
        };
        if self.view_coords(slab).all(|c| self.chunks.contains_key(&c)) {
            self.spawn_slab = None;
        }
    }

    fn submit_slab_columns(&mut self, slab: ChunkBox) {
        let coords: Vec<Coord> = self.view_coords(slab).collect();
        let runs = gather_column_runs(
            coords,
            |c| self.generator.sky(c),
            |c| self.chunks.contains_key(&c) || self.generating.contains(&c),
            |fail| self.quarantined.contains(&fail),
            true,
            false,
        );
        let mut remaining = false;
        for run in runs {
            let accepted = match run {
                GenRun::Column { key, lo, hi } => self.try_submit_column(key, lo, hi),
                GenRun::Open { coord } => self.try_submit_open(coord),
            };
            if !accepted {
                remaining = true;
            }
        }
        if remaining {
            self.pending_gen.set();
        }
    }

    /// Submit one column job and claim its missing coords. `false` means the
    /// pool rejected it (backpressure, shutdown, or quarantine) so the caller
    /// must retry.
    fn try_submit_column(&mut self, key: ColumnKey, lo: i32, hi: i32) -> bool {
        if self.quarantined.contains(&FailKey::Column { key }) {
            return false;
        }
        let edits: Vec<(Coord, Vec<(usize, crate::block::registry::BlockId)>)> = (lo..=hi)
            .filter_map(|alt| {
                let coord = key.chunk(alt);
                self.edits
                    .get(&coord)
                    .map(|cells| (coord, cells.iter().map(|(&i, &id)| (i, id)).collect()))
            })
            .collect();
        let job = pipeline::Job::GenerateColumn {
            key,
            range: lo..=hi,
            generator: self.generator.clone(),
            edits,
        };
        let accepted = self.worker_pool().submit(job);
        if accepted {
            for alt in lo..=hi {
                let coord = key.chunk(alt);
                if !self.chunks.contains_key(&coord) {
                    self.generating.insert(coord);
                }
            }
        }
        accepted
    }

    /// Submit one `Open` chunk. The worker still fills it through the PosY
    /// one-chunk `generate_column` encoding; the claim key does not.
    fn try_submit_open(&mut self, coord: Coord) -> bool {
        if self.quarantined.contains(&FailKey::Open { coord }) {
            return false;
        }
        let edits = self
            .edits
            .get(&coord)
            .map(|cells| cells.iter().map(|(&i, &id)| (i, id)).collect())
            .unwrap_or_default();
        let job = pipeline::Job::GenerateOpen {
            coord,
            generator: self.generator.clone(),
            edits,
        };
        let accepted = self.worker_pool().submit(job);
        if accepted && !self.chunks.contains_key(&coord) {
            self.generating.insert(coord);
        }
        accepted
    }

    /// Air and uniform bulk never take a worker slot: same `Chunk`, same edit replay,
    /// same light fast path as a generated uniform chunk.
    fn store_if_free(&mut self, coord: Coord) -> bool {
        if self.chunks.contains_key(&coord) || self.generating.contains(&coord) {
            return false;
        }
        let id = match self.generator.classify(coord) {
            Classify::Mixed => return false,
            Classify::Air => AIR,
            Classify::Uniform(id) => id,
        };
        let chunk = Chunk::from_data(coord.x, coord.y, coord.z, ChunkData::Uniform(id));
        self.store_chunk(coord, chunk);
        true
    }

    /// Generate a chunk's data if it isn't loaded, replaying any saved edits on it.
    /// Uses `generate_column` so the ceiling heights come from the same sample
    /// the voxels did — never a second `height()` walk on this thread.
    /// Skips coords already claimed in `generating`: the async result is imminent.
    pub(in crate::world) fn ensure_data(&mut self, coord: Coord) {
        if self.chunks.contains_key(&coord) || self.generating.contains(&coord) {
            return;
        }
        if self.store_if_free(coord) {
            self.refresh_spawn_slab();
            return;
        }
        let sky = self.generator.sky(coord);
        let (key, alt) = match sky {
            Sky::Axis(face) => ColumnKey::of(face, coord),
            // Same PosY encoding as `gather_column_runs`: one layer, no ceiling.
            Sky::Open => (ColumnKey { face: Face::PosY, a: coord.x, b: coord.z }, coord.y),
        };
        let (chunks, heights) = self.generator.generate_column(key, alt..=alt);
        if matches!(sky, Sky::Axis(_)) {
            self.install_ceiling(key, &heights);
        }
        let data = chunks
            .into_iter()
            .next()
            .map(|(_, data)| data)
            .expect("generate_column emits the requested layer");
        let placed = key.chunk(alt);
        let chunk = Chunk::from_data(placed.x, placed.y, placed.z, data);
        self.store_chunk(coord, chunk);
        self.refresh_spawn_slab();
    }

    /// Insert freshly generated data: replay the edit overlay, then register
    /// the chunk. A uniform-air chunk (after replay) can never produce
    /// geometry, so it is born `meshed` with no mesh — no worker job, no
    /// upload, nothing drawn.
    fn store_chunk(&mut self, coord: Coord, mut chunk: Chunk) {
        if let Some(edits) = self.edits.get(&coord) {
            for (&index, &id) in edits {
                chunk.set_index(index, id);
            }
        }
        // Uniform non-solid chunks produce no geometry, so start Air.
        // Check solidity, not AIR id, for future non-solid blocks.
        let born_air = chunk
            .uniform()
            .is_some_and(|id| !self.registry.is_solid(id));
        let state = if born_air {
            MeshState::Air
        } else {
            MeshState::needs_mesh()
        };
        // Born-air is already settled; a sky ring can complete without a
        // single upload.
        self.lod_clip_grow.raise(born_air);
        // No flood-fill here; occlusion rebuild computes connectivity lazily.
        let chunk = std::sync::Arc::new(chunk);
        // Liveness check: coord must not be claimed in generating (would shadow data).
        debug_assert!(
            !self.generating.contains(&coord),
            "storing {coord:?} still claimed in generating — a stuck generate claim"
        );
        self.light_claim_seq = self.light_claim_seq.wrapping_add(1);
        let light_gen = self.light_claim_seq;
        self.chunks.insert(
            coord,
            Loaded {
                chunk: std::sync::Arc::clone(&chunk),
                state,
                rev: 0,
                connectivity: None,
                visible: true,
                light: None,
                has_blocklight: false,
                light_reseed: false,
                light_gen,
                mesh_hash: None,
            },
        );
        // Ceiling-cache lifetime: the column's last layer out drops the entry.
        // Open chunks have no ceiling and are not recorded here.
        if let Sky::Axis(face) = self.generator.sky(coord) {
            let (key, alt) = ColumnKey::of(face, coord);
            let ys = self.column_chunks.entry(key).or_default();
            if !ys.contains(&alt) {
                ys.push(alt);
                ys.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        // Occlusion learns of the new chunk through the fill queue (bounded
        // drain per rebuild) — no per-rebuild missing-connectivity scan.
        if self.occlusion_enabled() {
            self.conn_fill_queue.push_back(coord);
        }
        // Light: try the analytic fast path first — a uniform-opaque chunk settles
        // to all-dark and an above-surface uniform-air chunk to full sky with no
        // flood. A trivial grid publishes synchronously (which fans the border to
        // its neighbours); only the residual Dense band seeds the settle worklist.
        if self.lighting {
            if self.light_inflight.contains(&coord) {
                // A previous Loaded at this coord still owns the inflight
                // claim. Skip trivial publish (it would steal that claim via
                // settle_light) and seed so we resettle after the stale Done
                // is consumed against the old generation.
                self.seed_light(coord, super::LightSeed::Store);
                self.light_pending.set();
            } else {
                match self.trivial_light(coord, &chunk) {
                    Some(grid) => self.settle_light(coord, grid),
                    None => {
                        self.seed_light(coord, super::LightSeed::Store);
                        self.light_pending.set();
                    }
                }
            }
        }
        // A new chunk changes what the BFS can reach — topology class:
        // debounced (an unclassified fresh chunk is over-draw, never a hole).
        self.occlusion_topo_dirty.set();
        // And it changes the near-field coverage picture the section skip
        // reads: re-arm the far-field lane so LOD reacts to ANY chunk
        // creation instead of waiting for a boundary crossing.
        if self.lod2 {
            self.pending_sections.set();
        }
        // Seed this chunk and 6 neighbours; a neighbour may have been
        // blocked waiting on this data even if itself uniform air.
        // A buried solid is `Air` with no mesh. New neighbour voxels (edit
        // replay on load) can open a face, so that chunk meshes again.
        // Uniform non-solid `Air` is born with nothing to draw.
        self.seed_mesh(coord);
        for face in Face::ALL {
            let n = self.neighbour(coord, face);
            let (is_air, fill) = match self.chunks.get(&n) {
                Some(l) if matches!(l.state, MeshState::Air) => (true, l.chunk.uniform()),
                _ => (false, None),
            };
            let buried = is_air && !fill.is_some_and(|id| !self.registry.is_solid(id));
            if buried {
                if let Some(loaded) = self.chunks.get_mut(&n) {
                    loaded.state = MeshState::needs_mesh();
                }
            }
            let ready = matches!(
                self.chunks.get(&n).map(|l| &l.state),
                Some(MeshState::Ready(_))
            );
            self.seed_mesh(n);
            // A Ready neighbour meshed without this chunk (terminal promotion
            // at a load-set edge, or the neighbour unloaded after the mesh).
            // Rebuild so the final look picks up the new border; worklist
            // seeding alone cannot, since Ready fails `is_needs_mesh`.
            if ready {
                self.remesh_async(n);
            }
        }
        self.pending_fresh.set();
    }

    /// Coords in the previous unload box that have left `new_box`, or every
    /// loaded chunk past `new_box` when there is no previous box (first pass
    /// or a radius change). Spawn-slab chunks stay.
    pub(in crate::world) fn unload_leaving(&self, new_box: ChunkBox) -> Vec<Coord> {
        let keep_spawn = |coord| self.spawn_slab.is_some_and(|slab| self.view_contains(slab, coord));
        match self.prev_unload_box {
            Some(prev) => self
                .view_shell(prev, new_box)
                .filter(|&coord| !keep_spawn(coord) && self.chunks.contains_key(&coord))
                .collect(),
            None => self
                .chunks
                .keys()
                .copied()
                .filter(|&coord| !self.view_contains(new_box, coord) && !keep_spawn(coord))
                .collect(),
        }
    }

    /// Free chunks past the unload box, releasing their GPU meshes.
    fn unload_far(&mut self, center: Coord, eng: &mut Engine) {
        self.unload_far_with(center, |state, cage| {
            state.free_owned(eng);
            if let Some(cage) = cage {
                eng.free_cage(cage);
            }
        });
    }

    /// [`unload_far`](Self::unload_far) with the GPU release passed in: `free` gets each removed
    /// chunk's mesh state and cage.
    fn unload_far_with(
        &mut self,
        center: Coord,
        mut free: impl FnMut(MeshState, Option<voxel_engine::CageHandle>),
    ) {
        let unload = self.unload_box(center);
        // Collect-then-remove instead of `retain`: freeing borrows the caller's
        // engine, which can't be borrowed inside a retain closure over `self.chunks`.
        let far = self.unload_leaving(unload);
        self.prev_unload_box = Some(unload);
        // A removed chunk changes what the BFS can reach — topology class.
        self.occlusion_topo_dirty.raise(!far.is_empty());
        for &coord in &far {
            // Free the mesh handle (Ready or Dirty); Air/NeedsMesh own none.
            // `far` holds loaded chunks only (see `unload_leaving`).
            if let Some(loaded) = self.chunks.remove(&coord) {
                super::adjust_count(&mut self.building_meshes, loaded.state.is_building(), false);
                free(loaded.state, self.cages.remove(&coord));
            }
            self.dirty_worklist.remove(&coord);
            // Seeds of a chunk that is gone are garbage: its next load seeds afresh. Left in, they
            // pile up in the clamped last ring during flight (never visited, re-bucketed on every
            // centre move).
            self.light_worklist.remove(&coord);
            self.mesh_worklist.remove(&coord);
            self.light_terminal.remove(&coord);
            self.remesh_stats.forget(coord);
            // Column layers: the last chunk out drops the cached ceiling.
            if let Sky::Axis(face) = self.generator.sky(coord) {
                let (key, alt) = ColumnKey::of(face, coord);
                if let Some(ys) = self.column_chunks.get_mut(&key) {
                    if let Some(i) = ys.iter().position(|&y| y == alt) {
                        ys.remove(i);
                    }
                    if ys.is_empty() {
                        self.column_chunks.remove(&key);
                        self.ceilings.remove(&key);
                    }
                }
            }
        }
        // Settled grids still queued for removed chunks describe the world
        // being unloaded: applying one to a LATER re-generated chunk would
        // publish stale light past every epoch check. Drop them and release
        // the claims they were carrying (one retain pass, not per-coord scans).
        if !self.light_apply_queue.is_empty() && !far.is_empty() {
            let removed: FastSet<Coord> = far.iter().copied().collect();
            let inflight = &mut self.light_inflight;
            self.light_apply_queue.retain(|(c, _)| {
                let gone = removed.contains(c);
                if gone {
                    inflight.remove(c);
                }
                !gone
            });
        }
        // (Ceilings for fully-unloaded columns dropped by the refcount above;
        // the heightmap is pure, so a re-entered column simply recomputes once.)
    }

    /// Remesh edited (`Dirty`) chunks synchronously, budgeted, nearest first —
    /// carved out from the async mesh lane so a broken block never lags a frame.
    /// Reports `Progress::Partial { remaining }` when more `Dirty` chunks are
    /// queued than the per-frame [`DIRTY_BUDGET`] (they stay `Dirty`, drained
    /// next frame).
    pub(in crate::world) fn remesh_dirty(&mut self, eng: &mut Engine) -> Progress {
        // Gated by pending_dirty so idle frames pay one flag check.
        if !self.pending_dirty.take() {
            return Progress::Idle;
        }
        let Some(center) = self.center else {
            return Progress::Idle;
        };
        // Drain the MAINTAINED membership set (entries whose chunk moved on —
        // unloaded, or resolved by another path — drop right here), instead of
        // filtering every loaded chunk each frame the hint is up: during a
        // light flood that was an O(world) iteration per frame.
        let chunks = &self.chunks;
        self.dirty_worklist
            .retain(|c| chunks.get(c).is_some_and(|l| l.state.is_dirty()));
        let mut dirty: Vec<Coord> = self.dirty_worklist.iter().copied().collect();
        let (up, fold) = (self.live_up(), self.fold);
        dirty.sort_by_key(|&coord| Self::order(fold.fold(coord), center, up));
        // Leftovers past the budget stay `Dirty` (still in the fiber); re-arm
        // the hint so the next frame drains them.
        let remaining = dirty.len().saturating_sub(DIRTY_BUDGET);
        if remaining > 0 {
            self.pending_dirty.set();
        }
        for coord in dirty.into_iter().take(DIRTY_BUDGET) {
            self.dirty_worklist.remove(&coord);
            // No neighbour-data gate: edited chunks remesh even with missing
            // neighbour data (mesher reads them as air).
            self.mesh_chunk(coord, eng);
        }
        if remaining > 0 {
            Progress::Partial {
                remaining: remaining as u32,
            }
        } else {
            Progress::Idle
        }
    }

    /// Snapshot for mesh job: chunk storage, neighbour shell, solidity table, and rev.
    pub(in crate::world) fn snapshot(
        &self,
        coord: Coord,
        degraded: bool,
    ) -> (u32, pipeline::ChunkSnapshot) {
        let loaded = &self.chunks[&coord];
        // One 3×3×3 lookup feeds both the voxel shell and the light shell.
        let nhood = self.loaded_neighbourhood(coord);
        let fallback = (self.lighting && degraded).then(light::LightGrid::open_sky);
        let mut padded = mesh::Padded::capture(|dx, dy, dz| {
            Self::nhood_at(&nhood, dx, dy, dz).map(|l| &*l.chunk)
        });
        self.seam_halo(coord, &mut padded);
        (
            loaded.rev,
            pipeline::ChunkSnapshot {
                padded,
                uniform: loaded.chunk.uniform(),
                // Lighting off omits the 18³ shell entirely — the mesher's
                // unlit path reads constant full light instead. `open_sky` is
                // Uniform (task 05): the degraded fallback does not allocate.
                light: self.lighting.then(|| {
                    let mut shell = light::PaddedLight::capture(|dx, dy, dz| {
                        Self::nhood_at(&nhood, dx, dy, dz)
                            .and_then(|l| l.light.as_ref())
                            .or(fallback.as_ref())
                    });
                    self.seam_light_halo(coord, &mut shell, fallback.as_ref());
                    shell
                }),
                tables: self.tables.get(),
            },
        )
    }

    /// The 3×3×3 neighbourhood of `coord`, dx-fast then dy then dz.
    fn loaded_neighbourhood(&self, coord: Coord) -> [Option<&Loaded>; 27] {
        let mut nhood = [None; 27];
        let mut i = 0;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    nhood[i] = self
                        .chunks
                        .get(&Coord::new(coord.x + dx, coord.y + dy, coord.z + dz));
                    i += 1;
                }
            }
        }
        nhood
    }

    /// Settled light shell for chunk and 26 neighbours (18³). A missing grid reads
    /// dark for a normal mesh; for a `degraded` mesh it stands in as fully-lit
    /// open-sky, so an unsettled neighbourhood fails toward visible-and-plausible.
    /// Only called with lighting enabled (the disabled path captures nothing).
    fn capture_padded_light(&self, coord: Coord, degraded: bool) -> light::PaddedLight {
        debug_assert!(self.lighting, "unlit meshes take the no-shell path");
        let fallback = degraded.then(light::LightGrid::open_sky);
        let mut shell = light::PaddedLight::capture(|dx, dy, dz| {
            self.chunks
                .get(&Coord::new(coord.x + dx, coord.y + dy, coord.z + dz))
                .and_then(|l| l.light.as_ref())
                .or(fallback.as_ref())
        });
        self.seam_light_halo(coord, &mut shell, fallback.as_ref());
        shell
    }

    /// Near-face light from six neighbours (input for light flood seeds).
    pub(in crate::world) fn capture_face_shell(&self, coord: Coord) -> light::FaceShell {
        if !self.lighting {
            return light::FaceShell::dark();
        }
        let mut shell = light::FaceShell::capture(|face| {
            self.chunks
                .get(&coord.step(face))
                .and_then(|l| l.light.as_ref())
        });
        self.seam_face_shell(coord, &mut shell);
        shell
    }

    /// Skylight ceiling: ground altitude per face-local column (caves dark
    /// consistently) RAISED by edited opaque roofs, so a player-built ceiling
    /// shadows the chunks below it. Keyed by [`ColumnKey`] and cached. `Open`
    /// chunks have no ceiling; this returns the ignored window and does not
    /// cache it.
    ///
    /// Async columns install the window in [`accept_column`](Self::accept_column)
    /// before store; [`ensure_data`](Self::ensure_data) does the same from the
    /// synchronous `generate_column`. This miss path is the remainder (edit
    /// invalidation, tests) and still reads heights from `generate_column`,
    /// never `height()`.
    ///
    /// Generated volumetrics (overhang shelves, flying islands) are still NOT
    /// part of the ceiling: `height()` deliberately describes ground only, so
    /// they don't shadow the columns beneath them — a known model limit that
    /// needs a generator-side occupancy summary to lift.
    pub(in crate::world) fn capture_ceiling(
        &mut self,
        coord: Coord,
    ) -> std::sync::Arc<light::CeilingWindow> {
        let Sky::Axis(face) = self.generator.sky(coord) else {
            return light::ignored_ceiling();
        };
        let (key, _) = ColumnKey::of(face, coord);
        if let Some(ceiling) = self.ceilings.get(&key) {
            return std::sync::Arc::clone(ceiling);
        }
        // Empty altitude range: both generators sample the 256 column profiles
        // before iterating the chunk layers, so this is the height field
        // without a voxel fill.
        let heights = self.generator.generate_column(key, 1..=0).1;
        let ceiling = std::sync::Arc::new(self.ceiling_from_heights(key, &heights));
        self.ceilings.insert(key, std::sync::Arc::clone(&ceiling));
        ceiling
    }

    /// Rebuild the PosY ceiling from `height()` plus edited roofs, ignoring the
    /// cache — equality check against production `generate_column` heights.
    #[cfg(test)]
    pub(in crate::world) fn capture_ceiling_slow(&self, coord: Coord) -> light::CeilingWindow {
        let x0 = coord.x * CHUNK_SIZE as i32;
        let z0 = coord.z * CHUNK_SIZE as i32;
        let generator = &self.generator;
        let mut ceiling = light::CeilingWindow::from_heights(Face::PosY, |lx, lz| {
            generator.height(x0 + lx as i32, z0 + lz as i32)
        });
        let key = ColumnKey { face: Face::PosY, a: coord.x, b: coord.z };
        self.raise_edited_roofs(key, &mut ceiling);
        ceiling
    }

    fn install_ceiling(&mut self, key: ColumnKey, heights: &ColumnHeights) {
        if self.ceilings.contains_key(&key) {
            return;
        }
        let ceiling = std::sync::Arc::new(self.ceiling_from_heights(key, heights));
        self.ceilings.insert(key, ceiling);
    }

    fn ceiling_from_heights(
        &self,
        key: ColumnKey,
        heights: &ColumnHeights,
    ) -> light::CeilingWindow {
        let mut ceiling =
            light::CeilingWindow::from_heights(key.face, |lu, lv| heights[lu + lv * CHUNK_SIZE]);
        self.raise_edited_roofs(key, &mut ceiling);
        ceiling
    }

    fn raise_edited_roofs(&self, key: ColumnKey, ceiling: &mut light::CeilingWindow) {
        let frame = FaceFrame::new(key.face);
        let s = CHUNK_SIZE as i32;
        for (&c, cells) in &self.edits {
            if !matches!(self.generator.sky(c), Sky::Axis(face) if face == key.face) {
                continue;
            }
            if ColumnKey::of(key.face, c).0 != key {
                continue;
            }
            for (&index, &id) in cells {
                if !self.registry.is_opaque(id) {
                    continue;
                }
                let (lx, ly, lz) = Chunk::local_of(index);
                let (lu, _, lv) = frame.index_to_local(lx, ly, lz);
                let world = (c.x * s + lx as i32, c.y * s + ly as i32, c.z * s + lz as i32);
                let alt = frame.cell_to_local(world).1;
                ceiling.raise(lu, lv, alt + 1);
            }
        }
    }

    /// The analytic light grid for a chunk whose settled light is provable
    /// without a flood, or `None` if it must go through the worker settle. The
    /// two trivial cases collapse the load-time light-job burst to the thin
    /// Dense surface band (see [`store_chunk`](Self::store_chunk)):
    /// - a uniform opaque, non-emissive chunk settles to all-dark (no light enters);
    /// - a uniform-*air* chunk fully above every column's surface, with no near
    ///   blocklight from a loaded neighbour, settles to full sky / dark block.
    ///   `Open` air with no near blocklight is full sky with no ceiling.
    ///
    /// Correctness anchor: the returned grid equals `propagate(uniform, dark
    /// shell, ceiling, sky, alt0, tables)`.
    ///
    /// `&mut self` so it can warm the `ceilings` column cache and the hot tables
    /// while probing — it mutates no lane state.
    fn trivial_light(&mut self, coord: Coord, chunk: &Chunk) -> Option<light::LightGrid> {
        if !self.lighting {
            return None;
        }
        self.refresh_tables();
        let tables = self.tables.get();
        // A full block of inert opaque rock settles to all-dark: no skylight
        // column stays open through it and no neighbour light can relax into an
        // opaque cell. Emissive opaque blocks must take the flood path so they
        // can seed their own blocklight.
        if chunk.is_uniform_opaque(&tables) {
            return Some(light::LightGrid::dark());
        }
        // A uniform-air chunk that sits fully above every column's surface is
        // full sky — *if* no loaded neighbour has near-border blocklight that
        // would bleed in (skylight can't exceed FULL, so only blocklight breaks
        // the analytic result). `propagate` with a dark shell yields exactly
        // `open_sky()` here; the neighbour check is what makes the dark shell sound.
        if chunk.uniform() == Some(crate::block::registry::AIR) {
            match self.generator.sky(coord) {
                Sky::Open => {
                    if !self.neighbour_blocklight_near(coord) {
                        return Some(light::LightGrid::open_sky());
                    }
                }
                Sky::Axis(face) => {
                    let alt0 = FaceFrame::new(face).chunk_alt0(coord);
                    let ceiling = self.capture_ceiling(coord);
                    if alt0 >= ceiling.min_surface() && !self.neighbour_blocklight_near(coord) {
                        return Some(light::LightGrid::open_sky());
                    }
                }
            }
        }
        None
    }

    /// Whether any loaded face-neighbour carries near-border blocklight `> 1`
    /// (light level 1 attenuates to 0 crossing in, so it can't seed). Used by
    /// [`trivial_light`](Self::trivial_light) to reject the dark-shell fast path
    /// when a torch next door would actually bleed across the border.
    fn neighbour_blocklight_near(&self, coord: Coord) -> bool {
        Face::ALL.iter().any(|&face| {
            self.chunks
                .get(&self.neighbour(coord, face))
                .is_some_and(|l| l.has_blocklight)
        })
    }

    /// Schedule an ASYNC rebuild for a chunk whose mesh inputs changed off the
    /// edit path (a light grid landed, a degraded mesh's real light arrived).
    /// The chunk keeps drawing its current mesh — carried as `NeedsMesh.prev`
    /// — until the fresh worker result uploads, and the rev bump both strands
    /// any in-flight build against the old inputs and stales any queued
    /// upload. This replaces the old routing of light arrivals through the
    /// SYNC `Dirty` machinery, which built up to `DIRTY_BUDGET` full greedy
    /// meshes per frame ON THE MAIN THREAD during load floods (nearly every
    /// chunk meshes degraded first under the 150 ms light gate, then relights)
    /// — the "still laggy seconds after stopping" stall. The sync path stays
    /// for player edits only, where same-frame response is the point.
    fn remesh_async(&mut self, coord: Coord) {
        let skip = match self.chunks.get(&coord).map(|l| &l.state) {
            None => return,
            Some(MeshState::Dirty { .. } | MeshState::Air) => true,
            Some(_) => false,
        };
        if skip {
            self.light_gate.dirty.remove(&coord);
            return;
        }
        {
            let Some(loaded) = self.chunks.get_mut(&coord) else {
                return;
            };
            if let MeshState::Ready(_) = &loaded.state {
                // Carry the drawn mesh into the rebuild. A NeedsMesh already
                // awaiting/mid-build just takes the rev bump, which strands the
                // in-flight result.
                let prev = std::mem::replace(&mut loaded.state, MeshState::needs_mesh()).into_owned();
                loaded.state = MeshState::NeedsMesh {
                    building: false,
                    prev,
                };
            }
            loaded.rev = loaded.rev.wrapping_add(1);
        }
        self.seed_mesh(coord);
        self.pending_fresh.set();
        self.light_gate.dirty.remove(&coord);
        self.remesh_stats.note_remesh(coord);
    }

    /// Faces whose border lumels differ. A first publish compares against dark
    /// (the shell missing neighbours already assumed).
    fn face_moves(prev: Option<&light::LightGrid>, grid: &light::LightGrid) -> u8 {
        let dark = light::LightGrid::dark();
        let old = prev.unwrap_or(&dark);
        let mut bits = 0u8;
        for &face in &Face::ALL {
            if light::border_changed(old, grid, face) {
                bits |= 1 << (face as u8);
            }
        }
        bits
    }

    /// Count a light-worklist insert (the stress harness's seeds-per-chunk signal).
    pub(in crate::world) fn seed_light(&mut self, coord: Coord, source: super::LightSeed) {
        self.light_seed_inserts += 1;
        self.light_seed_split.add(source);
        if let Some(loaded) = self.chunks.get_mut(&coord) {
            loaded.light_reseed = false;
        }
        self.light_worklist.insert(coord);
    }

    /// Publish settled light, re-arm mesh readiness, and seed neighbours to re-settle.
    /// Shared by sync (trivial) and async settle paths.
    pub(in crate::world) fn settle_light(&mut self, coord: Coord, grid: light::LightGrid) {
        // Release the claim first. No-op on sync path (trivial never enters inflight);
        // absorbs async removal, keeping settled/inflight state consistent.
        self.light_inflight.remove(&coord);
        // Unloaded while the flood flew (or before a trivial publish): drop it.
        let reseed = match self.chunks.get_mut(&coord) {
            Some(loaded) => {
                let r = loaded.light_reseed;
                loaded.light_reseed = false;
                r
            }
            None => return,
        };
        let self_changed = self.chunks[&coord]
            .light
            .as_ref()
            .is_none_or(|old| *old != grid);
        // Re-arm mesh readiness unconditionally, even on identical grids.
        // A fixpoint re-settle that skipped this seed would strand the chunk
        // off the worklist forever (ready but unreachable, idle stall).
        self.pending_fresh.set();
        self.seed_mesh(coord);
        if !self_changed {
            if reseed {
                self.seed_light(coord, super::LightSeed::Border);
                self.light_pending.set();
            }
            return;
        }
        // Face bitmask (bit = `Face` discriminant). First publish compares
        // against dark — missing neighbours already assumed that shell for the
        // flood. Neighbours still need a *mesh* seed: first publish is what
        // makes `light_ready` true for them.
        let first = self.chunks[&coord].light.is_none();
        let moved = Self::face_moves(self.chunks[&coord].light.as_ref(), &grid);
        let has_blocklight = grid.has_border_blocklight();
        let open_sky = grid == light::LightGrid::open_sky();
        {
            let loaded = self.chunks.get_mut(&coord).unwrap();
            loaded.light = Some(grid);
            loaded.has_blocklight = has_blocklight;
        }
        // Light-seed only neighbours whose shared border moved and that already
        // have data. An in-flight neighbour is marked, not re-inserted: at most
        // one extra flood when its result integrates. Mesh-seed on first publish
        // too (waiting neighbours become ready). Mark dirty instead of remeshing
        // immediately: `tick_light_gate` promotes once the 27-neighbourhood
        // has no pending light work (or the degrade timer expires).
        self.light_gate.mark_dirty(coord);
        for &face in &Face::ALL {
            let face_moved = moved & (1 << (face as u8)) != 0;
            if !face_moved && !first {
                continue;
            }
            let n = self.neighbour(coord, face);
            if !self.chunks.contains_key(&n) {
                continue;
            }
            if face_moved {
                // Open-sky next to open-sky cannot change. Mesh-seed only:
                // this publish can complete their light_ready.
                let n_sky = self.chunks[&n].light.as_ref() == Some(&light::LightGrid::open_sky());
                if open_sky && n_sky {
                    self.seed_mesh(n);
                    continue;
                }
                if self.light_inflight.contains(&n) {
                    self.chunks.get_mut(&n).unwrap().light_reseed = true;
                } else if !self.light_worklist.contains(&n) {
                    // Pending floods read live neighbour grids at admit.
                    self.seed_light(n, super::LightSeed::Border);
                }
            }
            self.seed_mesh(n);
            self.light_gate.mark_dirty(n);
        }
        if reseed {
            self.seed_light(coord, super::LightSeed::Border);
        }
        if !self.light_worklist.is_empty() {
            self.light_pending.set();
        }
    }

    fn nhood_at<'a>(nhood: &[Option<&'a Loaded>; 27], dx: i32, dy: i32, dz: i32) -> Option<&'a Loaded> {
        nhood[((dz + 1) * 9 + (dy + 1) * 3 + (dx + 1)) as usize]
    }

    /// Chunk + 1-voxel neighbour shell for mesh build (shared by worker and sync paths).
    fn capture_padded(&self, coord: Coord) -> mesh::Padded {
        let mut padded = mesh::Padded::capture(|dx, dy, dz| {
            self.chunks
                .get(&Coord::new(coord.x + dx, coord.y + dy, coord.z + dz))
                .map(|l| &*l.chunk)
        });
        self.seam_halo(coord, &mut padded);
        padded
    }

    // Column-LOD section selection and streaming.

    /// Per-frame selection metric: chunk-centre tangents, `dy` from eye altitude to the LOD envelope.
    /// Tangents stay on the chunk centre (not the raw eye) so PosY `dy=0` stays bit-identical.
    /// Altitude is relative to the face datum, so the envelope stays `[0, 512]` on every face.
    /// Streaming centre in a warped-cube box, mapped back to the reference cube. Outside every
    /// cube box this is the centre unchanged, so an identity fold stays bit-identical. A reference
    /// centre is not inside a box, so a second call does not translate again.
    fn lod_place(&self, center: Coord) -> (Coord, f64, bool) {
        let cs = CHUNK_SIZE as i64;
        let cell = [center.x as i64 * cs, center.y as i64 * cs, center.z as i64 * cs];
        for atlas in self.generator.atlases() {
            let Some(g) = atlas.grid else { continue };
            let inside = (0..3).all(|a| cell[a] >= g.origin[a] && cell[a] < g.origin[a] + g.size[a]);
            if !inside {
                continue;
            }
            let d = [
                (g.ref_min[0] - g.origin[0]) / cs,
                (g.ref_min[1] - g.origin[1]) / cs,
                (g.ref_min[2] - g.origin[2]) / cs,
            ];
            let reference = Coord::new(
                (center.x as i64 + d[0]) as i32,
                (center.y as i64 + d[1]) as i32,
                (center.z as i64 + d[2]) as i32,
            );
            let eye_y = self.section_eye_y + (g.ref_min[1] - g.origin[1]) as f64;
            return (reference, eye_y, true);
        }
        (center, self.section_eye_y, false)
    }

    fn section_metric_on(&self, center: Coord, delta: DVec3, face: Face, datum: i32) -> EyeMetric {
        let (center, eye_y, _) = self.lod_place(center);
        let cs = CHUNK_SIZE as i32;
        let cfg = &self.section_pyramid;
        let env = HeightEnvelope::new(super::section::LOD_FLOOR_Y as f32, super::section::LOD_CEIL_Y as f32);
        let cap = DyCap::new(cfg.outer_m(), cfg.base);
        if face == Face::PosY && datum == 0 {
            let (pcx, pcz) = (center.x * cs + cs / 2, center.z * cs + cs / 2);
            return EyeMetric::new(
                DVec3::new(pcx as f64 + delta.x, eye_y + delta.y, pcz as f64 + delta.z),
                env,
                cap,
            );
        }
        let frame = FaceFrame::new(face);
        let (cu, _, cv) = frame.chunk_to_local(center);
        let (u, v) = (cu * cs + cs / 2, cv * cs + cs / 2);
        let d = frame.point_to_local(delta);
        let eye = DVec3::new((center.x * cs + cs / 2) as f64, eye_y, (center.z * cs + cs / 2) as f64);
        let rel = frame.point_to_local(eye).y + d.y - datum as f64;
        EyeMetric::new(DVec3::new(u as f64 + d.x, rel, v as f64 + d.z), env, cap)
    }

    /// Desired frontier at one metric, stamped with `body`/`face` before coarsening
    /// so a parent keeps the frame and the mip lookup hits the right bake.
    fn frontier(&self, metric: &EyeMetric, body: u16, face: Face) -> Vec<SectionPos> {
        let cfg = &self.section_pyramid;
        let mut radial = quadtree::desired_sections(metric, cfg);
        for s in &mut radial {
            s.body = body;
            s.face = face;
        }
        let anchor = self.section_mip_anchor;
        let selected = match &self.section_mip {
            Some(mip) => {
                let summary_at = |c: SectionPos| {
                    if let Some((b, f, _, _)) = anchor
                        && (c.body != b || c.face != f)
                    {
                        return CellSummary {
                            env: HeightEnvelope::new(0.0, super::section::DOMAIN_H as f32),
                            err: CellError::worst_case(c.detail),
                        };
                    }
                    match self.section_overlay.get(&c) {
                        Some(ov) => CellSummary {
                            env: HeightEnvelope::new(ov.lo, ov.hi),
                            err: CellError::from_metres(ov.hi - ov.lo),
                        },
                        None => mip.summary(c),
                    }
                };
                quadtree::coarsen_by_error(radial, metric, cfg, &summary_at, &self.sse_budget())
            }
            None => radial,
        };
        // Prefer coarser tiles over dropping coverage when the far field
        // would exceed its section-slot budget (outermost ring first).
        quadtree::coarsen_to_budget(selected, self.sections_allowed(), cfg)
    }

    /// Ladder-pinned SSE budget for current view radius. Rebuilt per query
    /// as `unit` tracks view distance.
    fn sse_budget(&self) -> SseBudget {
        SseBudget::ladder(self.section_pyramid.unit, self.section_pyramid.finest.0)
    }

    /// Spawn background max-mip bake around the eye's face. Re-bakes when the eye
    /// leaves the inner half of the baked square, or the body/face changes.
    /// A still camera (anchor held, bake landed or in flight) allocates nothing.
    pub(in crate::world) fn ensure_mip_bake(&mut self) {
        if self.section_face_set && self.section_lod_face.is_none() {
            return;
        }
        let (body, face) = self.section_lod_face.unwrap_or((0, Face::PosY));
        let (au, av) = match (self.section_face_set, self.center) {
            (true, Some(c)) => self.face_tangent_centre(c, face),
            _ => (0, 0),
        };
        let cfg = &self.section_pyramid;
        let extent = BakeExtent::new(cfg.outer_m() as i32, cfg.coarsest());
        let half = extent.half_m() / 2;
        let face_changed = self.section_mip_anchor.is_some_and(|(b, f, _, _)| b != body || f != face);
        if face_changed {
            self.section_mip = None;
            self.section_mip_rx = None;
            self.section_frontier_key = None;
        }
        let moved = self.section_mip_anchor.is_some_and(|(_, _, u, v)| (au - u).abs() > half || (av - v).abs() > half);
        if !face_changed && !moved && (self.section_mip.is_some() || self.section_mip_rx.is_some()) {
            return;
        }
        if self.section_mip_rx.is_some() {
            return;
        }
        self.section_mip_anchor = Some((body, face, au, av));
        let generator = self.generator.clone();
        // The generator stores resolved IDs for every element-worldgen
        // composition registered during `World::new`. A fresh builtin registry
        // is too short for those IDs; snapshot the matching color table instead.
        let colors = self.registry.color_snapshot();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(HeightMip::bake_at(&*generator, &colors, extent, au, av, face, body));
        });
        self.section_mip_rx = Some(rx);
    }

    /// Install background bake if landed. Newly-arrived mip only coarsens
    /// far field, so just re-arm section pass to re-select.
    pub(in crate::world) fn poll_mip(&mut self) {
        if let Some(rx) = &self.section_mip_rx
            && let Ok(mip) = rx.try_recv()
        {
            self.section_mip = Some(mip);
            self.section_mip_rx = None;
            self.pending_sections.set();
        }
    }

    /// Re-derive the edit-folded cell for every section touched by a live edit,
    /// fixing the immutable bake's edit-staleness (a mined-out feature would
    /// otherwise keep occluding/colouring/measuring error as if still solid).
    /// The `&mut` sync point `render`'s `&self` readers may never recompute
    /// (occlusion-class derived state is rebuilt here, not in render).
    ///
    /// Cost is bounded by `section_edit_rev`'s size (sections an edit has EVER
    /// touched), not by view distance or total edit count: untouched cells never
    /// enter the loop, so an unedited world pays nothing (`section_overlay` stays
    /// empty and every reader falls back to the pure bake, bit-identical to
    /// before this cache existed).
    /// A quiet frame is one set-emptiness check: only the exact positions
    /// edits touched since the last refresh (`section_overlay_dirty`) are
    /// re-derived, and the resolved map is maintained incrementally instead
    /// of cleared and rebuilt every pass.
    ///
    /// Each position costs about one far section's extract, and one edit dirties a position per
    /// active detail, so the work is spread over frames: at least one position per pass, more while
    /// `budget` lasts. [`Self::remesh_dirty_sections`] holds a square back until its overlay is in.
    pub(in crate::world) fn refresh_section_overlay(&mut self, budget: Budget) -> Progress {
        if self.section_overlay_dirty.is_empty() {
            return Progress::Idle;
        }
        let deadline = super::lanes::paced_deadline(self, budget);
        let positions: Vec<SectionPos> = self.section_overlay_dirty.iter().copied().collect();
        for (done, pos) in positions.into_iter().enumerate() {
            if done > 0 && deadline.expired() {
                break;
            }
            self.section_overlay_dirty.remove(&pos);
            let rev = voxel_engine::Rev(self.section_edit_rev.get(&pos).copied().unwrap_or(0));
            let touched = self.edits_for_section(pos);
            if touched.is_empty() {
                // Reverted back to what the generator would produce (overlay
                // compaction, edits.rs): no override, the pure bake applies.
                self.section_overlay.remove(&pos);
                continue;
            }
            let cell = *self.section_overlay_cache.get_or_recompute(pos, rev, || {
                let colors = self.registry.color_snapshot();
                super::heightmip::resample_cell(pos, &*self.generator, &touched, &colors)
            });
            match cell {
                Some(cell) => {
                    self.section_overlay.insert(pos, cell);
                }
                None => {
                    self.section_overlay.remove(&pos);
                }
            }
        }
        match self.section_overlay_dirty.len() {
            0 => Progress::Idle,
            n => Progress::Partial { remaining: n as u32 },
        }
    }

    /// The atlas patch a chart section is bent through. `None` for a cube section or a square
    /// that is not inside a storage box.
    fn chart_bend(&self, pos: SectionPos) -> Option<super::ChartBend> {
        if pos.body < super::section::CHART_BODY_BASE {
            let atlas = self.seams.atlases().iter().find(|a| {
                a.grid.as_ref().is_some_and(|g| g.body == pos.body) && a.warp.is_some()
            })?;
            return Some(super::ChartBend { atlas: atlas.clone(), patch: crate::space::atlas::Patch::Grid });
        }
        let index = (pos.body - super::section::CHART_BODY_BASE) as usize;
        let atlas = self.seams.atlases().get(index)?.clone();
        let (patch, _) = atlas.locate([pos.min_x() as i64, 0, pos.min_z() as i64])?;
        Some(super::ChartBend { atlas, patch })
    }

    /// Far sections of the home chart and, where the far field reaches a side, its neighbours.
    /// Storage +Y is the chart's up, so the sections are [`Face::PosY`] over storage `(x, z)`.
    fn chart_sections(&self, center: Coord, memo: &mut NearBounds) -> Vec<SectionPos> {
        let Some(seat) = self.seams.chart_seat(center) else { return Vec::new() };
        let Some((cfg, max_d)) = self.chart_pyramid(seat.radius) else { return Vec::new() };
        let base = self.chart_pick(center, DVec3::ZERO, &seat, &cfg, max_d, memo);
        let delta = chart_delta(self.section_vel);
        if delta == DVec3::ZERO {
            return base;
        }
        quadtree::union_frontiers(base, self.chart_pick(center, delta, &seat, &cfg, max_d, memo))
    }

    /// Pyramid stopped at the largest detail whose span still satisfies `L² ≤ 8R`, on the
    /// altitude-scaled unit.
    fn chart_pyramid(&self, radius: i64) -> Option<(pyramid::PyramidCfg, i8)> {
        let src = self.far_pyramid();
        let mut levels = 0u8;
        let mut max_d = src.finest.0;
        for ring in 0..src.levels.get() {
            let detail = src.finest.0 + ring as i8 * src.step() as i8;
            let span = super::section::section_span(crate::ident::Detail(detail));
            if !super::section::section_fits(span, radius) {
                break;
            }
            levels += 1;
            max_d = detail;
        }
        (levels > 0).then(|| (pyramid::PyramidCfg::sections_with(src.unit, levels, src.finest.0 as u8), max_d))
    }

    /// Eye metric in the storage frame. Altitude is height above the column under the eye, so
    /// standing on a mountain still selects the finest ring (the cube envelope is `[0, 512]`).
    fn chart_metric(&self, center: Coord, delta: DVec3, cfg: &pyramid::PyramidCfg) -> EyeMetric {
        let (ex, ey, ez) = storage_eye_block(center, self.section_eye_y, delta);
        let ground = self.generator.surface(Face::PosY, ex as i32, ez as i32);
        let rel = if ground == i32::MIN { 0.0 } else { (ey as f64 - ground as f64).clamp(0.0, 1.0e7) };
        let env = HeightEnvelope::new(super::section::LOD_FLOOR_Y as f32, super::section::LOD_CEIL_Y as f32);
        EyeMetric::new(DVec3::new(ex as f64, rel, ez as f64), env, DyCap::new(cfg.outer_m(), cfg.base))
    }

    fn chart_pick(
        &self,
        center: Coord,
        delta: DVec3,
        seat: &super::seam::ChartSeat,
        cfg: &pyramid::PyramidCfg,
        max_d: i8,
        memo: &mut NearBounds,
    ) -> Vec<SectionPos> {
        let (ex, ey, ez) = storage_eye_block(center, self.section_eye_y, delta);
        let ground = self.generator.surface(Face::PosY, ex as i32, ez as i32);
        if ground == i32::MIN {
            return Vec::new();
        }
        let rel = (ey as f64 - ground as f64).clamp(0.0, 1.0e7);
        let body = super::section::CHART_BODY_BASE + seat.index as u16;
        let (y0, y1) = self.near_y_range(center);
        let near = self.chart_near(center, body, y0);
        let mut tagged =
            self.seat_sections(seat, ex as f64, ez as f64, rel, body, cfg, max_d, near, y0, y1, None, memo);
        // The neighbour is visible as far as the coarsest ring's square reaches, not merely the
        // two finest sections. Its rings run on from the eye unfolded beyond its edge.
        let span = super::section::section_span(crate::ident::Detail(max_d)) as f64;
        let band = (ring_reach(f64::from(cfg.outer_m()), span) * span) as i64;
        for across in self.seams.seam_across(*seat, [ex, ey, ez], band) {
            let (ix, iz) = across.storage_xz(ex, ez);
            tagged.extend(self.seat_sections(
                &across.seat,
                ix as f64,
                iz as f64,
                rel,
                body,
                cfg,
                max_d,
                near,
                y0,
                y1,
                Some(&across),
                memo,
            ));
        }
        let budget = self.sections_allowed();
        if tagged.len() > budget {
            // Nearest first, so the cap drops the far rim rather than the ground beside the eye.
            tagged.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| section_key(&a.0).cmp(&section_key(&b.0))));
            tagged.truncate(budget);
        }
        let mut out: Vec<_> = tagged.into_iter().map(|(s, _)| s).collect();
        out.sort_unstable_by_key(section_key);
        out
    }

    fn seat_sections(
        &self,
        seat: &super::seam::ChartSeat,
        ex: f64,
        ez: f64,
        rel: f64,
        body: u16,
        cfg: &pyramid::PyramidCfg,
        max_d: i8,
        near: (i64, i64, i64, i64),
        y0: i64,
        y1: i64,
        across: Option<&super::seam::SeamAcross>,
        memo: &mut NearBounds,
    ) -> Vec<(SectionPos, f64)> {
        let env = HeightEnvelope::new(super::section::LOD_FLOOR_Y as f32, super::section::LOD_CEIL_Y as f32);
        let metric = EyeMetric::new(DVec3::new(ex, rel, ez), env, DyCap::new(cfg.outer_m(), cfg.base));
        let mut radial = quadtree::desired_sections(&metric, cfg);
        for s in &mut radial {
            s.body = body;
            s.face = Face::PosY;
        }
        let mut clipped = Vec::new();
        for s in radial {
            cover_chart(s, seat.lo, seat.hi, &mut clipped);
        }
        // A straddler is replaced before the punch. Dropping it whole would leave the part
        // outside the box — up to one ring of span — drawn by nothing.
        let mut edged = Vec::new();
        for s in clipped {
            cover_near(s, near, across, &mut edged);
        }
        // Charts have no shader clip. A section wholly inside the near square is dropped only
        // when that square's surface sits inside the full-res window; a valley or a hilltop
        // outside it stays, so the far field draws what the window misses.
        let keep = |s: SectionPos, memo: &mut NearBounds| {
            inside_xz(s, seat.lo, seat.hi)
                && super::section::section_fits(s.span(), seat.radius)
                && !self.near_window_holds(s, near, y0, y1, across, memo)
        };
        let kept: Vec<SectionPos> = edged.into_iter().filter(|&s| keep(s, memo)).collect();
        // A straddling parent must not merge back: that tile is what the descent just replaced.
        let merge = |p: SectionPos| keep(p, memo) && !(covers_near(p, near, across) && !inside_near(p, near, across));
        coarsen_chart(kept, max_d, self.sections_allowed(), merge)
            .into_iter()
            .map(|s| (s, section_dist2(s, ex, ez)))
            .collect()
    }

    /// The full-res window already draws every solid top of `s`, and `s` lies wholly inside the
    /// near square. A section that only crosses the edge is not punched: the part outside the
    /// square would be drawn by nothing. Storage altitude: `surface` is the first open cell, so
    /// the solid top is the block below it. A bound we cannot place is kept (punched nowhere)
    /// so a missed column is not a sky hole.
    fn near_window_holds(
        &self,
        s: SectionPos,
        near: (i64, i64, i64, i64),
        y0: i64,
        y1: i64,
        across: Option<&super::seam::SeamAcross>,
        memo: &mut NearBounds,
    ) -> bool {
        if !inside_near(s, near, across) {
            return false;
        }
        let Some((u0, v0, u1, v1)) = overlap_storage(s, near, across) else {
            return false;
        };
        let surface = || self.generator.surface_rect(s.body, Face::PosY, u0, v0, u1, v1);
        let Some((lo, hi)) = memo.get((s.body, [u0, v0, u1, v1]), surface) else {
            return false;
        };
        i64::from(lo) - 1 >= y0 && i64::from(hi) - 1 < y1
    }

    /// The full-res square the punch tests chart sections against. Empty when the near window
    /// stands in physical space, or when its floor `y0` is above every surface of the square: it
    /// draws no ground there, so splitting the far field around it would punch nothing.
    fn chart_near(&self, center: Coord, body: u16, y0: i64) -> (i64, i64, i64, i64) {
        const NONE: (i64, i64, i64, i64) = (0, 0, 0, 0);
        if self.fold.is_identity() {
            return NONE;
        }
        let near = self.near_block_box(center);
        let span = i32::try_from((near.1 - near.0).max(near.3 - near.2)).unwrap_or(i32::MAX);
        match self.generator.surface_bounds(body, Face::PosY, near.0 as i32, near.2 as i32, span) {
            Some((_, hi)) if i64::from(hi) - 1 < y0 => NONE,
            _ => near,
        }
    }

    /// Full-res vertical block range (`y1` exclusive) of the mesh box. Up is storage Y.
    fn near_y_range(&self, center: Coord) -> (i64, i64) {
        let cs = CHUNK_SIZE as i64;
        let b = self.mesh_box(center);
        let y0 = i64::from(b.min().y) * cs;
        (y0, y0 + i64::from(b.size().1) * cs)
    }

    /// Full-res chunk box in storage blocks (`hi` exclusive), wide on x/z. Up is storage Y, so this
    /// is the footprint a chart section overlaps when it meets the near square.
    fn near_block_box(&self, center: Coord) -> (i64, i64, i64, i64) {
        let cs = CHUNK_SIZE as i64;
        let h = self.view.horizontal as i64;
        (
            (center.x as i64 - h) * cs,
            (center.x as i64 + h + 1) * cs,
            (center.z as i64 - h) * cs,
            (center.z as i64 + h + 1) * cs,
        )
    }

    /// [`desired_sections_with`](Self::desired_sections_with) reading every surface afresh.
    #[cfg(test)]
    pub(in crate::world) fn desired_sections(&self, center: Coord) -> Vec<SectionPos> {
        self.desired_sections_with(center, &mut NearBounds::default())
    }

    /// Desired frontier: union of static eye and velocity-predicted eye position.
    /// Pulls sections ahead of player motion. At rest, velocity is zero so returns
    /// static frontier bit-for-bit. Open space and a round body seen from past the far reach
    /// select nothing; a far-field centre on a chart (in or above its box) selects that chart's
    /// sections, reading chart surfaces through `memo`.
    fn desired_sections_with(&self, center: Coord, memo: &mut NearBounds) -> Vec<SectionPos> {
        if self.section_on_chart(center) {
            return self.chart_sections(center, memo);
        }
        let focus = if self.section_face_set { self.section_lod_face } else { self.dominant_lod_face(center) };
        let Some((body, face)) = focus else { return Vec::new() };
        let mut out = self.frontier_union(center, body, face);
        for nface in self.edge_faces(center, body, face) {
            out = quadtree::union_frontiers(out, self.frontier_union(center, body, nface));
        }
        out
    }

    fn frontier_union(&self, center: Coord, body: u16, face: Face) -> Vec<SectionPos> {
        let datum = self.generator.face_datum(body, face);
        let base = self.frontier(&self.section_metric_on(center, DVec3::ZERO, face, datum), body, face);
        let delta = self.section_vel * TAU_STREAM;
        if delta == DVec3::ZERO {
            return base;
        }
        let predicted = self.frontier(&self.section_metric_on(center, delta, face, datum), body, face);
        quadtree::union_frontiers(base, predicted)
    }

    /// Chunk-centre sample the far field treats as the eye (tangents quantised, altitude exact on Y).
    fn lod_eye_point(&self, center: Coord) -> DVec3 {
        let (center, eye_y, _) = self.lod_place(center);
        let cs = CHUNK_SIZE as i32;
        DVec3::new((center.x * cs + cs / 2) as f64, eye_y, (center.z * cs + cs / 2) as f64)
    }

    /// Face-local chunk-centre tangents.
    fn face_tangent_centre(&self, center: Coord, face: Face) -> (i32, i32) {
        let (center, _, _) = self.lod_place(center);
        let cs = CHUNK_SIZE as i32;
        let (cu, _, cv) = FaceFrame::new(face).chunk_to_local(center);
        (cu * cs + cs / 2, cv * cs + cs / 2)
    }

    /// The cube face under the camera. `None` in a chart's storage, in open space, or over a round
    /// body. A streaming centre inside a warped cube still names that cube's face.
    fn dominant_lod_face(&self, center: Coord) -> Option<(u16, Face)> {
        if self.section_on_chart(center) {
            return None;
        }
        let Some(cosmos) = self.generator.cosmos() else {
            return Some((0, Face::PosY));
        };
        let eye = self.lod_eye_point(center);
        let body = cosmos.body_at(eye)?;
        if !matches!(body.shape, super::terrain::cosmos::Shape::Cube { .. }) {
            return None;
        }
        Some((body.id, Face::from_dominant(eye - body.centre_f())))
    }

    /// Commit the face for this pass. The previous face sticks while its component
    /// is within two finest sections of the dominant one (the edge).
    fn update_lod_face(&mut self, center: Coord) {
        let dominant = self.dominant_lod_face(center);
        self.section_lod_face = match (self.section_lod_face, dominant) {
            (Some((id, prev)), Some((bid, _))) if id == bid && self.face_holds(center, id, prev) => Some((id, prev)),
            _ => dominant,
        };
        self.section_face_set = true;
    }

    fn face_holds(&self, center: Coord, body_id: u16, prev: Face) -> bool {
        let Some(cosmos) = self.generator.cosmos() else { return prev == Face::PosY };
        let Some(body) = cosmos.bodies().iter().find(|b| b.id == body_id) else { return false };
        let rel = self.lod_eye_point(center) - body.centre_f();
        let comps = [rel.x.abs(), rel.y.abs(), rel.z.abs()];
        let max = comps[0].max(comps[1]).max(comps[2]);
        let band = (super::section::section_span(super::section::FINEST_DETAIL) * 2) as f64;
        max - comps[prev.axis()] <= band
    }

    /// Neighbouring faces whose squares are within two finest sections of the eye.
    fn edge_faces(&self, center: Coord, body_id: u16, face: Face) -> Vec<Face> {
        let Some(cosmos) = self.generator.cosmos() else { return Vec::new() };
        let Some(body) = cosmos.bodies().iter().find(|b| b.id == body_id) else { return Vec::new() };
        let super::terrain::cosmos::Shape::Cube { half } = body.shape else { return Vec::new() };
        let frame = FaceFrame::new(face);
        let local = frame.point_to_local(self.lod_eye_point(center) - body.centre_f());
        let band = (super::section::section_span(super::section::FINEST_DETAIL) * 2) as f64;
        let mut out = Vec::new();
        let mut push = |du: i32, dv: i32| {
            let (x, y, z) = frame.cell_to_world((du, 0, dv));
            let n = Face::from_dominant(DVec3::new(x as f64, y as f64, z as f64));
            if n != face && !out.contains(&n) {
                out.push(n);
            }
        };
        if half as f64 - local.x.abs() < band && local.x != 0.0 {
            push(local.x.signum() as i32, 0);
        }
        if half as f64 - local.z.abs() < band && local.z != 0.0 {
            push(0, local.z.signum() as i32);
        }
        out
    }

    /// True if the cell or a Ready ancestor covers it.
    pub(in crate::world) fn section_covered(&self, cell: SectionPos) -> bool {
        let max = crate::ident::Detail(crate::render_config::LOD_COARSEST_DETAIL as i8);
        let ready = |p: SectionPos| self.sections.get(&p).is_some_and(|s| s.is_ready());
        quadtree::drawable_cover(cell, max, &ready).is_some()
    }

    /// Edits affecting this section: chunks within its footprint and height domain.
    /// Used when re-extracting after an edit. Reads the `section_edit_chunks`
    /// index — O(this section's edited chunks), not a scan of every edit in
    /// the world (the reference scan survives as `edits_in_footprint`, pinned
    /// equivalent by test). Compacted-away chunks fall out at the lookup.
    pub(in crate::world) fn edits_for_section(
        &self,
        pos: SectionPos,
    ) -> Vec<(Coord, Vec<(usize, crate::block::registry::BlockId)>)> {
        let Some(chunks) = self.section_edit_chunks.get(&pos) else {
            return Vec::new();
        };
        chunks
            .iter()
            .filter_map(|&c| {
                let cells = self.edits.get(&c)?;
                if cells.is_empty() {
                    return None;
                }
                Some((c, cells.iter().map(|(&i, &b)| (i, b)).collect()))
            })
            .collect()
    }

    /// Rebuild the visible set every frame; as sections become Ready, the covering
    /// changes and stale entries would draw incorrectly.
    ///
    /// Also the LEVEL-TRIGGERED load arming (the fast-movement staleness fix):
    /// while ANY desired cell is unloaded, uncovered by a Ready self/ancestor,
    /// and not skipped as chunk-covered near field, the section lane stays
    /// armed. The old edge-triggered arming (boundary crossings and a few
    /// events) could go quiet with holes still open — flying far up left the
    /// covering permanently behind the live frontier, drawing a couple of
    /// stale coarse cubes over an otherwise missing far field.
    pub(in crate::world) fn rebuild_section_visible(&mut self, eng: Option<&mut Engine>) {
        let Some(center) = self.section_center() else {
            return;
        };
        let desired = std::mem::take(&mut self.section_desired);
        let max = crate::ident::Detail(crate::render_config::LOD_COARSEST_DETAIL as i8);
        let ready = |p: SectionPos| self.sections.get(&p).is_some_and(|s| s.is_ready());
        let cut = quadtree::resolve_covering(&desired, max, &ready);
        let backlog = desired.iter().any(|&c| {
            !self.sections.contains_key(&c)
                && quadtree::drawable_cover(c, max, &ready).is_none()
                && !self.coverage_skips(center, c)
        });
        self.section_desired = desired;
        if backlog {
            self.pending_sections.set();
        }
        self.section_visible = cut.iter().copied().collect();
        // Adopt the new cut (hard pop); it decides what actually draws.
        let changed = self.section_fade.update_now(&self.section_visible);
        // The mask is a projection of that decision, so it follows the same diff —
        // a region whose slots have not landed yet is caught by the upload site instead.
        if let Some(eng) = eng {
            for (pos, mask) in changed {
                if let Some(state) = self.sections.get(&pos) {
                    state.set_visible(eng, mask);
                }
            }
        }
    }

    /// Free Ready sections that no desired cell draws, when the section floor is
    /// full and a desired cell is still unloaded. A still camera does not unload,
    /// so those extras would block covering for good.
    fn reclaim_blocked_sections(&mut self, center: Coord, mut eng: Option<&mut Engine>) {
        let allowed = self.sections_allowed();
        let used = self.section_budget_used();
        if used < allowed {
            return;
        }
        // A still, converged floor has neither flag. Holes keep admission
        // pending (a full budget no longer clears it), and a frontier change
        // raises the cover flag before this runs, so the scan below stays off
        // the quiet path.
        if !self.pending_sections.get() && !self.section_cover_dirty.get() {
            return;
        }
        let holes = self
            .section_desired
            .iter()
            .filter(|&&c| {
                !self.sections.contains_key(&c)
                    && !self.section_covered(c)
                    && !self.coverage_skips(center, c)
                    && !self.quarantined.contains(&FailKey::Section { pos: c })
            })
            .count();
        if holes == 0 {
            return;
        }
        let need = holes + (used - allowed);
        let max = crate::ident::Detail(crate::render_config::LOD_COARSEST_DETAIL as i8);
        let ready = |p: SectionPos| self.sections.get(&p).is_some_and(|s| s.is_ready());
        let desired: FastSet<SectionPos> = self.section_desired.iter().copied().collect();
        let mut covers: FastSet<SectionPos> = FastSet::default();
        for &c in &self.section_desired {
            if let Some(p) = quadtree::drawable_cover(c, max, &ready) {
                covers.insert(p);
            }
        }
        let spare: Vec<SectionPos> = self
            .sections
            .iter()
            .filter_map(|(&s, state)| {
                if !matches!(state, SectionState::Ready { .. }) {
                    return None;
                }
                (!desired.contains(&s) && !covers.contains(&s)).then_some(s)
            })
            .collect();
        let mut victims: Vec<(u64, SectionPos)> = spare
            .into_iter()
            .map(|s| {
                let rank = <SectionLane as StreamLane>::dist2(self, center, s).unwrap_or(0);
                (rank, s)
            })
            .collect();
        if victims.is_empty() {
            return;
        }
        victims.sort_unstable_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| section_key(&a.1).cmp(&section_key(&b.1)))
        });
        let mut freed = 0usize;
        for (_, s) in victims.into_iter().take(need) {
            let gpu = match self.sections.get(&s) {
                Some(SectionState::Ready { meshes, cages, .. }) => !meshes.is_empty() || !cages.is_empty(),
                _ => false,
            };
            if gpu && eng.is_none() {
                continue;
            }
            if let Some(state) = self.sections.remove(&s) {
                if let Some(eng) = eng.as_deref_mut() {
                    state.free(eng);
                }
                freed += 1;
            }
        }
        if freed > 0 {
            self.pending_sections.set();
            self.section_cover_dirty.set();
        }
    }

    /// Unload sections outside desired, visible, and hysteresis bands (boundary cross).
    /// Hysteresis prevents thrashing at view edges.
    fn unload_sections(&mut self, center: Coord, eng: &mut Engine) {
        self.unload_sections_with(center, |state| state.free(eng));
    }

    /// [`unload_sections`](Self::unload_sections) with the GPU release passed in.
    fn unload_sections_with(&mut self, center: Coord, mut free: impl FnMut(SectionState)) {
        // KEEP reads the frame's cached frontier (already velocity-unioned),
        // so sections stay kept even as a fast-moving eye passes.
        let desired: FastSet<SectionPos> = self.section_desired.iter().copied().collect();
        let visible: FastSet<SectionPos> = self.section_visible.iter().map(|(p, _)| *p).collect();
        // Fading sections still draw this frame. Keep meshes until fade completes
        // or outgoing section vanishes mid-fade.
        let fading: FastSet<SectionPos> = self.section_fade.tracked().collect();
        let cfg = &self.far_pyramid();
        let (metric_body, metric_face, metric) = if self.section_on_chart(center) {
            let body = self
                .seams
                .chart_seat(center)
                .map(|s| super::section::CHART_BODY_BASE + s.index as u16)
                .unwrap_or(u16::MAX);
            (body, Face::PosY, self.chart_metric(center, DVec3::ZERO, cfg))
        } else {
            let metric_face = self.section_lod_face.map(|(_, f)| f).unwrap_or(Face::PosY);
            let metric_body = self.section_lod_face.map(|(b, _)| b).unwrap_or(0);
            let metric = self.section_metric_on(
                center,
                DVec3::ZERO,
                metric_face,
                self.generator.face_datum(metric_body, metric_face),
            );
            (metric_body, metric_face, metric)
        };
        let stale: Vec<SectionPos> = self
            .sections
            .keys()
            .copied()
            .filter(|s| {
                if desired.contains(s) || visible.contains(s) || fading.contains(s) {
                    return false;
                }
                let span = s.span();
                let (cx, cz) = (s.x * span + span / 2, s.z * span + span / 2);
                // A section on another face is kept only while it is still desired.
                let dist = if s.body == metric_body && s.face == metric_face {
                    metric.point(cx as f64, cz as f64)
                } else {
                    EyeDist::new(f32::MAX)
                };
                !pyramid::acceptable(dist, s.detail, cfg)
            })
            .collect();
        for s in &stale {
            if let Some(state) = self.sections.remove(s) {
                super::adjust_count(
                    &mut self.meshing_sections,
                    matches!(state, SectionState::Meshing { .. }),
                    false,
                );
                free(state);
            }
        }
        // Removals move the covering (a freed cell may re-expose an ancestor).
        self.section_cover_dirty.raise(!stale.is_empty());
    }

    /// Free GPU meshes so edited sections re-extract from the updated overlay.
    pub(in crate::world) fn remesh_dirty_sections(&mut self, eng: &mut Engine) {
        if self.dirty_sections.is_empty() {
            return;
        }
        let dirty: Vec<SectionPos> = self.dirty_sections.iter().copied().collect();
        let mut freed = false;
        for s in dirty {
            // Re-extract only once the square's overlay is in: its upload reads the overlay colour.
            if self.section_overlay_dirty.contains(&s) {
                continue;
            }
            match self.sections.get(&s) {
                Some(SectionState::Ready { .. }) => {
                    if let Some(state) = self.sections.remove(&s) {
                        state.free(eng);
                    }
                    self.dirty_sections.remove(&s);
                    freed = true;
                }
                Some(SectionState::Meshing { .. }) => {} // in flight: free once it lands Ready
                None => {
                    self.dirty_sections.remove(&s);
                }
            }
        }
        if freed {
            self.pending_sections.set();
            self.section_cover_dirty.set();
        }
    }

    /// Every cell is opaque. A paletted chunk's entries are exactly the ids in
    /// use, so the palette decides it without a cell walk.
    fn chunk_all_opaque(chunk: &Chunk, tables: &HotTables) -> bool {
        match &chunk.data().payload {
            ChunkPayload::Uniform(v) => tables.opaque(v.id),
            ChunkPayload::Paletted { palette, .. } => palette.iter().all(|p| tables.opaque(p.id)),
            ChunkPayload::Dense(cells) => cells.iter().all(|c| tables.opaque(c.id)),
        }
    }

    /// The face that touches `face` is solid opaque. Uniform and all-opaque
    /// payloads answer without walking the face.
    fn chunk_face_opaque(chunk: &Chunk, face: Face, tables: &HotTables) -> bool {
        if let Some(id) = chunk.uniform() {
            return tables.opaque(id);
        }
        if Self::chunk_all_opaque(chunk, tables) {
            return true;
        }
        let edge = if face.sign() > 0 { CHUNK_SIZE - 1 } else { 0 };
        for a in 0..CHUNK_SIZE {
            for b in 0..CHUNK_SIZE {
                let (x, y, z) = match face.axis() {
                    0 => (edge, a, b),
                    1 => (a, edge, b),
                    _ => (a, b, edge),
                };
                if !tables.opaque(chunk.get_local(x, y, z)) {
                    return false;
                }
            }
        }
        true
    }

    /// A fresh mesh whose chunk is fully opaque and whose six neighbour faces
    /// are too has nothing to draw. Settle it `Air` (the empty-mesh result)
    /// and skip the snapshot. A carried GPU mesh still goes through the worker
    /// so the old handle is freed, and a missing neighbour reads as air and
    /// would emit faces.
    pub(in crate::world) fn bury_solid_mesh(&mut self, coord: Coord) -> bool {
        let fresh = matches!(
            self.chunks.get(&coord).map(|l| &l.state),
            Some(MeshState::NeedsMesh {
                building: false,
                prev: None,
            })
        );
        if !fresh || self.light_terminal.contains(&coord) || !self.neighbours_have_data(coord) {
            return false;
        }
        self.refresh_tables();
        let tables = self.tables.get();
        if !self
            .chunks
            .get(&coord)
            .is_some_and(|l| Self::chunk_all_opaque(&l.chunk, &tables))
        {
            return false;
        }
        for &face in &Face::ALL {
            let ncoord = self.neighbour(coord, face);
            let covered = self.chunks.get(&ncoord).is_some_and(|l| {
                Self::chunk_face_opaque(&l.chunk, face.opposite(), &tables)
            });
            if !covered {
                return false;
            }
        }
        if let Some(loaded) = self.chunks.get_mut(&coord) {
            loaded.state = MeshState::Air;
        }
        self.lod_clip_grow.set();
        true
    }

    /// Put `coord` on the mesh worklist, or settle it `Air` when it is already
    /// walled in. Only a chunk the mesh lane could admit is seeded: an unloaded
    /// coord seeds itself when it loads, an in-flight build re-seeds if its
    /// result goes stale (`drop_stale_upload`), and a `Ready`, `Dirty` or `Air`
    /// chunk is never admitted (a relight remeshes through the light gate, an
    /// edit through the dirty lane). Seeding those parked them in rings the
    /// admission walk never reached, re-bucketed on every centre move.
    fn seed_mesh(&mut self, coord: Coord) {
        if self.awaits_mesh(coord) && !self.bury_solid_mesh(coord) {
            self.mesh_worklist.insert(coord);
        }
    }

    /// A loaded chunk awaiting a mesh with no build in flight: the only state the mesh lane admits.
    fn awaits_mesh(&self, coord: Coord) -> bool {
        matches!(
            self.chunks.get(&coord).map(|l| &l.state),
            Some(MeshState::NeedsMesh { building: false, .. })
        )
    }

    /// Six orthogonal neighbours have data loaded.
    pub(in crate::world) fn neighbours_have_data(&self, coord: Coord) -> bool {
        Face::ALL
            .iter()
            .all(|&f| self.chunks.contains_key(&self.neighbour(coord, f)))
    }

    /// Light settled enough to mesh: chunk and face neighbours have grids, and the
    /// chunk is neither seeded nor being settled (in-flight counts as not-yet-final,
    /// so a chunk never meshes against a flood that's still running for it).
    pub(in crate::world) fn light_ready(&self, coord: Coord) -> bool {
        if !self.lighting {
            // Nothing to settle: gate meshing on data alone (checked separately).
            return self.chunks.contains_key(&coord);
        }
        !self.light_worklist.contains(&coord)
            && !self.light_inflight.contains(&coord)
            && self.chunks.get(&coord).is_some_and(|l| l.light.is_some())
            && Face::ALL.iter().all(|&f| {
                self.chunks
                    .get(&self.neighbour(coord, f))
                    .is_some_and(|l| l.light.is_some())
            })
    }

    /// True when the 27-neighbourhood has no pending light work. Apply-queue
    /// coords keep their inflight claim until `settle_light`, so inflight
    /// covers the queue; the empty-queue check is the cheap global fast path.
    pub(in crate::world) fn light_nhood_quiet(&self, coord: Coord) -> bool {
        if self.light_worklist.is_empty()
            && self.light_inflight.is_empty()
            && self.light_apply_queue.is_empty()
        {
            return true;
        }
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let n = Coord::new(coord.x + dx, coord.y + dy, coord.z + dz);
                    if self.light_worklist.contains(&n) || self.light_inflight.contains(&n) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Turn a `light_dirty` mark into a mesh job. Returns true if the mark
    /// can drop. Skips an in-flight first mesh (no rev bump) so the early
    /// degraded mesh still appears on the same clock as before.
    fn promote_dirty_mesh(&mut self, coord: Coord) -> bool {
        let Some(loaded) = self.chunks.get(&coord) else {
            return true;
        };
        let action = match &loaded.state {
            MeshState::Air | MeshState::Dirty { .. } => 0u8,
            MeshState::NeedsMesh {
                building: true, ..
            } => 1,
            MeshState::NeedsMesh {
                building: false,
                prev: None,
            } => 2,
            MeshState::Ready(_)
            | MeshState::NeedsMesh {
                building: false,
                prev: Some(_),
            } => 3,
        };
        match action {
            0 => true,
            1 => false,
            2 => self.light_nhood_quiet(coord),
            _ => {
                self.remesh_async(coord);
                true
            }
        }
    }

    /// A chunk waiting purely on neighbour light: it has data and is in view and
    /// awaiting a fresh mesh, but its neighbourhood light has not settled. The
    /// [`LightGate`] times exactly these chunks.
    pub(in crate::world) fn chunk_light_blocked(&self, coord: Coord) -> bool {
        self.is_needs_mesh(coord)
            && self.in_mesh_box(coord)
            && self.neighbours_have_data(coord)
            && !self.light_ready(coord)
    }

    /// Whether `coord` has waited on neighbour light past [`LIGHT_WAIT_DEGRADE`] —
    /// the mesh-lane predicate that admits a DEGRADED mesh.
    pub(in crate::world) fn light_wait_expired(&self, coord: Coord) -> bool {
        self.light_gate
            .blocked_since
            .get(&coord)
            .is_some_and(|t| t.elapsed() >= LIGHT_WAIT_DEGRADE)
    }

    /// Record (or clear) that `coord` is currently drawing a degraded, known-not-
    /// final mesh. The set is queryable by [`entry_complete`](Self::entry_complete)
    /// ("none pending").
    pub(in crate::world) fn mark_degraded(&mut self, coord: Coord, degraded: bool) {
        if degraded {
            self.light_gate.degraded.insert(coord);
        } else {
            self.light_gate.degraded.remove(&coord);
            self.light_terminal.remove(&coord);
        }
    }

    /// Advance the light-gate before the mesh lane runs: reap timers whose
    /// chunk stopped waiting, drop degraded/dirty entries for unloaded chunks,
    /// promote `light_dirty` (and relit-degraded) chunks whose 27-neighbourhood
    /// has no pending light work or whose degrade timer expired, and re-seed
    /// exactly the chunks whose DEGRADE TIMER expired — expiry raises no event
    /// of its own, so this sweep (over ONLY the timed/dirty maps, never the
    /// world) is what un-strands them. Timers START at the admit loop's
    /// blocked-eviction event ([`MeshLane::on_blocked`]); every pre-expiry
    /// re-seed comes from a real event (a grid landing via `settle_light`,
    /// neighbour data via `store_chunk`).
    pub(in crate::world) fn tick_light_gate(&mut self) {
        // `LightGate` is `Default`, so move it out to break the self-borrow while
        // the predicates below read the chunk map. Empty maps skip `retain`
        // (it still walks capacity); a drained flood `shrink_to_fit`s once.
        let mut gate = std::mem::take(&mut self.light_gate);
        if !gate.degraded.is_empty() {
            gate.degraded.retain(|c| self.chunks.contains_key(c));
            if gate.degraded.is_empty() {
                gate.degraded.shrink_to_fit();
            }
        }
        if !gate.dirty.is_empty() {
            gate.dirty.retain(|c, _| self.chunks.contains_key(c));
            if gate.dirty.is_empty() {
                gate.dirty.shrink_to_fit();
            }
        }
        if !self.light_terminal.is_empty() {
            self.light_terminal.retain(|c| self.chunks.contains_key(c));
            if self.light_terminal.is_empty() {
                self.light_terminal.shrink_to_fit();
            }
        }
        if !gate.blocked_since.is_empty() {
            gate.blocked_since
                .retain(|c, _| self.chunk_light_blocked(*c));
            if gate.blocked_since.is_empty() {
                gate.blocked_since.shrink_to_fit();
            }
        }
        // One clock read for the sweep, and only when a timer exists to compare.
        // Timers are tested before the 27-neighbourhood scan: an expired entry
        // promotes whatever the neighbours say.
        let now = (!gate.dirty.is_empty() || !gate.blocked_since.is_empty()).then(crate::sched::now);
        let waited = |t: &Instant| now.is_some_and(|n| n.duration_since(*t) >= LIGHT_WAIT_DEGRADE);
        // Promote dirty (and relit-degraded) chunks once the 27-neighbourhood
        // has no pending light work, or the degrade timer has expired. One
        // pass over the dirty/degraded sets, never the world.
        let mut promote: Vec<Coord> = gate
            .dirty
            .iter()
            .filter(|(c, t)| waited(t) || self.light_nhood_quiet(**c))
            .map(|(c, _)| *c)
            .collect();
        for &c in &gate.degraded {
            if gate.dirty.contains_key(&c) {
                continue;
            }
            if self.light_ready(c)
                && (gate.blocked_since.get(&c).is_some_and(waited) || self.light_nhood_quiet(c))
            {
                promote.push(c);
            }
        }
        for c in promote {
            if self.promote_dirty_mesh(c) {
                gate.dirty.remove(&c);
            } else {
                gate.dirty.entry(c).or_insert_with(crate::sched::now);
            }
        }
        // The expiry sweep: a chunk past LIGHT_WAIT_DEGRADE is mesh-ready via
        // `light_wait_expired` but was evicted from the worklist when it
        // blocked — re-seed it now that the clock (not an event) unblocked it.
        // A timed chunk whose build is in flight takes no seed.
        let mut expired = false;
        for (&c, t) in &gate.blocked_since {
            if waited(t) {
                self.seed_mesh(c);
                expired = true;
            }
        }
        if expired {
            self.pending_fresh.set();
        }
        self.light_gate = gate;
    }

    /// Level-triggered backstop for the degraded set: per-coord, once that
    /// chunk's 27-neighbourhood has no pending light work. Event-driven paths
    /// miss degraded chunks whose missing neighbour settled without moving the
    /// shared border; this sweep promotes them to final so entry_complete
    /// doesn't hang. A still-building/Dirty chunk is left for a later flush.
    pub(in crate::world) fn flush_degraded_terminal(&mut self) {
        if self.light_gate.degraded.is_empty() {
            return;
        }
        let stuck: Vec<Coord> = self.light_gate.degraded.iter().copied().collect();
        for coord in stuck {
            match self.chunks.get(&coord).map(|l| &l.state) {
                // Still building (in-flight result pending) or Dirty (a sync
                // remesh owns it): another path is about to resolve it.
                Some(MeshState::NeedsMesh { building: true, .. } | MeshState::Dirty { .. }) => {}
                // Seeded in the box: the mesh lane admits it.
                Some(MeshState::NeedsMesh { .. })
                    if self.in_mesh_box(coord) && self.mesh_worklist.contains(&coord) => {}
                // Every arm below acts, and only once the 27-neighbourhood has
                // no pending light work.
                _ if !self.light_nhood_quiet(coord) => {}
                // Unloaded out from under the set between marking and here, or
                // nothing to draw: drop the degraded flag.
                None | Some(MeshState::Air) => self.mark_degraded(coord, false),
                // Past the mesh box (unload hysteresis): not drawn, and a
                // rebuild would fail `in_mesh_box` / be dropped at apply. Drop
                // the flag so quiescence is not wedged.
                Some(_) if !self.in_mesh_box(coord) => self.mark_degraded(coord, false),
                // Settled on a degraded mesh — the stuck case. Rebuild async;
                // if neighbour light is still missing it will never arrive, so
                // the terminal set makes the snapshot read missing planes dark.
                Some(MeshState::Ready(_)) => {
                    if !self.light_ready(coord) {
                        self.light_terminal.insert(coord);
                    }
                    self.remesh_async(coord);
                }
                // Admit evicted the seed (a missing neighbour used to fail
                // `ready`, or this flush ran after that pass's admit). Re-seed
                // it; mark terminal if neighbour light will not arrive.
                Some(MeshState::NeedsMesh { .. }) => {
                    if !self.light_ready(coord) {
                        self.light_terminal.insert(coord);
                    }
                    self.mesh_worklist.insert(coord);
                    self.pending_fresh.set();
                }
            }
        }
    }

    /// World-entry completeness predicate: true once, within the view
    /// radius, every chunk shows a FINAL-light mesh (`Ready`/`Air`, none degraded
    /// and none still waiting on light), no near generate/mesh/light work is queued
    /// or in flight, and (under lod2) the section far field is covering-complete.
    /// Reads private streaming state — its home here.
    pub fn entry_complete(&self) -> bool {
        let Some(center) = self.center else {
            return false;
        };
        // No near work queued or in flight, and nothing owed a final-light remesh.
        if !self.near_quiescent()
            || !self.upload_queue.is_empty()
            || !self.light_gate.degraded.is_empty()
            || !self.light_gate.dirty.is_empty()
            || !self.light_gate.blocked_since.is_empty()
        {
            return false;
        }
        // Every in-view chunk has a final mesh (data loaded, not building/dirty).
        for coord in self.view_coords(self.mesh_box(center)) {
            match self.chunks.get(&coord).map(|l| &l.state) {
                Some(MeshState::Air | MeshState::Ready(_)) => {}
                _ => return false,
            }
        }
        // LOD2 far field: all desired cells covered and no uploads pending. Skipped
        // when disabled (no far field in near-only mode).
        if self.desired_unrefined().is_some() {
            if !self.section_upload_queue.is_empty() {
                return false;
            }
            if self
                .section_desired
                .iter()
                .any(|&c| !self.section_covered(c))
            {
                return false;
            }
        }
        true
    }

    /// Near generate/mesh/light queues empty — the five-queue rest predicate.
    fn near_quiescent(&self) -> bool {
        self.generating.is_empty()
            && self.mesh_worklist.is_empty()
            && self.light_worklist.is_empty()
            && self.light_inflight.is_empty()
            && self.light_apply_queue.is_empty()
    }

    /// `None` when LOD2 is off. `Some(n)` is how many desired far cells lack
    /// their own Ready mesh.
    fn desired_unrefined(&self) -> Option<usize> {
        if !self.lod2 {
            return None;
        }
        Some(
            self.section_desired
                .iter()
                .filter(|c| !self.sections.get(c).is_some_and(|s| s.is_ready()))
                .count(),
        )
    }

    /// Every desired far-field section is itself Ready — the strongest far-field
    /// state. `entry_complete` accepts a Ready *ancestor* as covering (right for
    /// playability), but a coarse cover moves the horizon's pixels — and through
    /// the exposure meter, the whole frame's brightness — as refinement lands.
    /// The golden harness gates captures on this so blessed shots are the
    /// converged frame; gameplay never waits on it.
    pub fn far_field_refined(&self) -> bool {
        if self.center.is_none() {
            return false;
        }
        match self.desired_unrefined() {
            None => true,
            Some(n) => self.section_upload_queue.is_empty() && n == 0,
        }
    }

    /// How many desired far-field sections still lack their own mesh — the
    /// harness's progress signal while it waits on
    /// [`far_field_refined`](Self::far_field_refined).
    pub fn far_field_pending(&self) -> usize {
        if self.center.is_none() {
            return 0;
        }
        self.desired_unrefined().unwrap_or(0)
    }

    /// Snapshot the streaming-queue depths (see [`super::StreamGauges`]).
    pub fn stream_gauges(&self) -> super::StreamGauges {
        let (worker_near_queue, worker_far_queue, active_workers, worker_capacity) = self
            .workers
            .as_ref()
            .map(|workers| {
                let (near, far) = workers.queue_depths();
                (
                    near,
                    far,
                    workers.active_workers(),
                    workers.worker_capacity(),
                )
            })
            .unwrap_or_default();
        let staging = self
            .workers
            .as_ref()
            .map(pipeline::Workers::staging_snapshot)
            .unwrap_or_default();
        let (ru_mean, ru_p95, ru_n) = self.remesh_stats.between_upload_mean_p95();
        let (jf_mean, jf_p95, jf_n) = self.remesh_stats.jobs_before_fixpoint_mean_p95();
        super::StreamGauges {
            chunks: self.chunks.len(),
            generating: self.generating.len(),
            mesh_worklist: self.mesh_worklist.len(),
            upload_queue: self.upload_queue.len(),
            light_worklist: self.light_worklist.len(),
            light_inflight: self.light_inflight.len(),
            light_apply_queue: self.light_apply_queue.len(),
            worker_near_queue,
            worker_far_queue,
            active_workers,
            worker_capacity,
            travel_speed_mps: self.stream_pacer.speed_mps(),
            effort: self.stream_pacer.effort(),
            light_admitted: self.light_admitted,
            light_admitted_last: self.light_admitted_last,
            light_seed_inserts: self.light_seed_inserts,
            mesh_slots: if self.gpu_live_slots != 0 {
                self.gpu_live_slots as usize
            } else {
                self.local_mesh_slots()
            },
            slot_ceiling: self.slot_ceiling as usize,
            section_ready: self.sections.values().filter(|s| s.is_ready()).count(),
            light_seed_split: self.light_seed_split,
            remesh_async_calls: self.remesh_stats.remesh_async_calls,
            drop_stale_uploads: self.remesh_stats.drop_stale_uploads,
            drop_stale_this_frame: self.remesh_stats.drop_stale_this_frame,
            remesh_between_upload_mean: ru_mean,
            remesh_between_upload_p95: ru_p95,
            remesh_between_upload_n: ru_n,
            mesh_jobs_before_fixpoint_mean: jf_mean,
            mesh_jobs_before_fixpoint_p95: jf_p95,
            mesh_jobs_before_fixpoint_n: jf_n,
            section_upload_bytes: self.section_upload_bytes,
            drain_upload_bytes: self.drain_upload_bytes,
            mesh_staged: staging.chunk_staged,
            mesh_fallback: staging.chunk_fallback,
            mesh_ring_full: staging.chunk_ring_full,
            section_staged: staging.section_staged,
            section_fallback: staging.section_fallback,
            section_ring_full: staging.section_ring_full,
            reactions_pending: self.reactions.pending(),
            reactions_mutations: self.reactions.operations,
        }
    }

    /// Human-readable reason `entry_complete` is not yet true — the first
    /// unsatisfied clause with a count, so a stalled bless/harness run says WHICH
    /// streaming stage is stuck instead of hanging silently. Clause order mirrors
    /// [`entry_complete`](Self::entry_complete).
    pub fn entry_debug(&self) -> String {
        if let Some(slab) = self.spawn_slab {
            let missing = self
                .view_coords(slab)
                .filter(|c| !self.chunks.contains_key(c))
                .count();
            if missing > 0 {
                return format!(
                    "spawn slab loading: {missing} chunks, generating={}",
                    self.generating.len()
                );
            }
        }
        let Some(center) = self.center else {
            return "no stream centre yet".into();
        };
        if !self.quarantined.is_empty() {
            return format!(
                "{} claim(s) quarantined after repeated worker panics: {:?}",
                self.quarantined.len(),
                self.quarantined.iter().take(4).collect::<Vec<_>>()
            );
        }
        // Share the one queue-depth source with the harness gauge, so the two
        // can never drift; the gate counters have no gauge field, so stay local.
        let g = self.stream_gauges();
        let near: [(&str, usize); 10] = [
            ("generating", g.generating),
            ("mesh_worklist", g.mesh_worklist),
            ("upload_queue", g.upload_queue),
            ("light_worklist", g.light_worklist),
            ("light_inflight", g.light_inflight),
            ("light_apply_queue", g.light_apply_queue),
            ("degraded", self.light_gate.degraded.len()),
            ("light_dirty", self.light_gate.dirty.len()),
            ("terminal", self.light_terminal.len()),
            ("light_blocked", self.light_gate.blocked_since.len()),
        ];
        let pending: Vec<String> = near
            .iter()
            .filter(|(_, n)| *n != 0)
            .map(|(k, n)| format!("{k}={n}"))
            .collect();
        if !pending.is_empty() {
            let mut msg = format!("near work pending: {}", pending.join(", "));
            // If the fresh-mesh lane is the blocker, tally WHICH ready()-predicate the
            // stuck chunks fail — the four gates from `MeshLane::ready`.
            if !self.mesh_worklist.is_empty() {
                let (mut not_needs, mut out_box, mut no_neigh, mut lit_or_expired) = (0, 0, 0, 0);
                for &c in self.mesh_worklist.iter() {
                    if !self.is_needs_mesh(c) {
                        not_needs += 1;
                    } else if !self.in_mesh_box(c) {
                        out_box += 1;
                    } else if !self.neighbours_have_data(c) && !self.light_terminal.contains(&c) {
                        no_neigh += 1;
                    } else if self.light_ready(c)
                        || self.light_wait_expired(c)
                        || self.light_terminal.contains(&c)
                    {
                        lit_or_expired += 1;
                    }
                }
                msg.push_str(&format!(
                    " | mesh_worklist stuck-on: not_needs_mesh={not_needs} out_of_box={out_box} \
                     no_neighbour_data={no_neigh} ready_but_unclaimed={lit_or_expired} \
                     (lighting={})",
                    self.lighting
                ));
            }
            return msg;
        }
        // Terminal wedge: near-work queues empty but some in-box chunk not final.
        // Tally by state and (for idle NeedsMesh) by which gate would block.
        let (mut missing, mut idle, mut building, mut dirty, mut queued) = (0, 0, 0, 0, 0);
        let (mut idle_no_neigh, mut idle_unlit) = (0, 0);
        for c in self.view_coords(self.mesh_box(center)) {
            let in_wl = self.mesh_worklist.contains(&c);
            match self.chunks.get(&c).map(|l| &l.state) {
                Some(MeshState::Air | MeshState::Ready(_)) => {}
                None => missing += 1,
                Some(MeshState::Dirty { .. }) => dirty += 1,
                Some(MeshState::NeedsMesh { building: true, .. }) => building += 1,
                Some(MeshState::NeedsMesh {
                    building: false, ..
                }) => {
                    if in_wl {
                        queued += 1;
                    } else {
                        idle += 1;
                        if !self.neighbours_have_data(c) && !self.light_terminal.contains(&c) {
                            idle_no_neigh += 1;
                        } else if !(self.light_ready(c)
                            || self.light_wait_expired(c)
                            || self.light_terminal.contains(&c))
                        {
                            idle_unlit += 1;
                        }
                    }
                }
            }
        }
        let unmeshed = missing + idle + building + dirty + queued;
        if unmeshed != 0 {
            return format!(
                "chunks without a final mesh: {unmeshed} \
                 [missing={missing} idle={idle} building={building} dirty={dirty} \
                 queued_but_worklist_drained={queued}] \
                 idle stuck-on: no_neighbour_data={idle_no_neigh} unlit={idle_unlit}"
            );
        }
        if self.lod2 {
            if !self.section_upload_queue.is_empty() {
                return format!("section_upload_queue = {}", self.section_upload_queue.len());
            }
            let uncovered = self
                .section_desired
                .iter()
                .filter(|&&c| !self.section_covered(c))
                .count();
            if uncovered != 0 {
                return format!(
                    "column sections uncovered: {uncovered} of {} desired",
                    self.section_desired.len()
                );
            }
        }
        "entry complete".into()
    }

    /// Build chunk GPU mesh (sync dirty-remesh). Frees old handle exactly once.
    fn mesh_chunk(&mut self, coord: Coord, eng: &mut Engine) {
        self.refresh_tables();
        // Move the scratch out so the build can borrow `self.chunks` shared
        // (for cross-chunk neighbour culling) while filling it. `MeshData` has no
        // `Default` (it carries a `Pass`), so swap in a fresh opaque scratch
        // rather than `mem::take`; `build_chunk_mesh` clears it first anyway.
        let mut scratch = std::mem::replace(&mut self.scratch, mesh::new_chunk_mesh_data());
        let tables = self.tables.get();
        let uniform = self.chunks[&coord].chunk.uniform();
        let padded = self.capture_padded(coord);
        // Use currently-published light (may be stale after edits). Geometry updates
        // this frame for responsiveness; relit result lands later when light reconverges.
        let degraded = !self.light_ready(coord);
        self.mark_degraded(coord, degraded);
        let light = self.capture_padded_light(coord, degraded);
        mesh::build_chunk_mesh(&padded, uniform, &tables, &light, &mut scratch);
        debug_assert!(
            self.chunks.get(&coord).is_some_and(|l| l.state.is_dirty()),
            "sync remesh of non-Dirty {coord:?}"
        );
        // `upload_chunk`'s retire frees the edited-Ready chunk's old mesh
        // (`Dirty.prev`) exactly once and installs the fresh `Ready`/`Air`.
        let hash = mesh::content_hash(&scratch);
        self.upload_chunk(coord, &scratch, Some(hash), eng);
        self.scratch = scratch;
    }

    /// Re-snapshot hot solidity array if palette grew (append-only, new Arc, old jobs unaffected)
    /// or a stamped meshing input (AO) flipped — the epoch folds into the revision's high bits
    /// (block count stays far below 2^32, so the two never collide).
    pub(in crate::world) fn refresh_tables(&mut self) {
        // Split the borrow: `sync`'s rebuild closure needs `&self.registry`
        // while `&mut self.tables` is held, so bind `registry` separately.
        let count = self.registry.block_count();
        let registry = &self.registry;
        let layer_cap = self.texture_layer_cap;
        let ao = self.ao;
        let rev = Revision::from_count(count | (self.tables_epoch as usize) << 32);
        self.tables.sync(rev, || {
            let mut tables = registry.hot_tables();
            tables.layer_cap = layer_cap;
            tables.ao = ao;
            tables
        });
    }

    /// Rebuild/upload the block texture array when configurations gain layers (world entry, a newly
    /// interned configuration) or the appearance `revision` changes. Existing layers never change at
    /// one revision (a pure function of the configuration; layer ids are append-only), so only the
    /// first upload / a revision rebuild uses `set_block_textures`; later growth appends.
    fn refresh_textures(&mut self, eng: &mut Engine, appearance: &dyn BlockAppearance) {
        // Never zero and never past the vertex field's 14 bits. Construction caches `u16::MAX`; the
        // device cap is read once.
        if !self.texture_cap_from_device {
            let device = eng.max_texture_array_layers().clamp(1, u16::MAX as u32) as u16;
            self.texture_layer_cap = device.min(crate::block::MAX_DESCRIPTORS as u16);
            self.texture_cap_from_device = true;
            self.registry.set_descriptor_cap(self.texture_layer_cap);
            #[cfg(test)]
            crate::alloc_count::note_engine(crate::alloc_count::EngineCall::TexLayers);
        }
        let count = self.registry.descriptor_count();
        let rev = appearance.revision();
        if self.appearance_revision != rev {
            self.texture_cache.clear();
            self.uploaded_len = 0;
            self.textures_built = 0;
            self.appearance_revision = rev;
        }
        if self.textures_built == count {
            return;
        }
        for i in self.texture_cache.len()..count {
            let mut buf = [0u8; LAYER_BYTES];
            fill_layer(appearance, &self.registry, i as u16, &mut buf);
            self.texture_cache.push(buf.to_vec());
        }
        let visible = count.min(self.texture_layer_cap as usize);
        match plan_texture_upload(&self.texture_cache, self.uploaded_len, visible) {
            Some(TextureUpload::Set(layers)) => eng.set_block_textures(TEXTURE_SIZE, layers),
            Some(TextureUpload::Append(layers)) => eng.append_block_textures(layers),
            None => {}
        }
        self.uploaded_len = visible;
        self.textures_built = count;
    }

    /// Start-world sites the far field is checked from high above the ground: the direction from
    /// the centre and the heights. A face centre, a highland, a seam (77 km from the +X/+Y edge)
    /// and a cube corner (46 km from both edges of the +X chart).
    #[cfg(test)]
    pub(in crate::world) const FAR_SITES: [(&'static str, DVec3, &'static [f64]); 4] = [
        ("plus-y", DVec3::new(0.0, 1.0, 0.0), &[10_000.0, 50_000.0, 150_000.0]),
        ("highland", DVec3::new(1.0, 0.9, 0.8), &[10_000.0, 50_000.0, 150_000.0]),
        ("seam", DVec3::new(1.0, 0.995, 0.3), &[10_000.0, 50_000.0, 150_000.0]),
        ("corner", DVec3::new(1.0, 0.997, 0.997), &[50_000.0]),
    ];

    /// A physical eye `above` blocks out along the local up (the radial) from the start world's
    /// ground in direction `dir` from its centre.
    #[cfg(test)]
    pub(in crate::world) fn home_eye(&self, dir: DVec3, above: f64) -> DVec3 {
        use crate::space::atlas::Patch;
        use crate::space::chart::{self, Map};
        let centre = self.generator.cosmos().expect("cosmos").home().centre_f();
        let atlas = self
            .generator
            .atlases()
            .iter()
            .find(|a| (a.centre - centre).length() < 1.0)
            .expect("the start world is charted");
        let dir = dir.normalize();
        let face = Face::from_dominant(dir);
        let (tu, nn, tv) = chart::basis(face);
        let (xi, eta) = Map::Equiangular.inverse(DVec3::new(dir.dot(tu), dir.dot(nn), dir.dot(tv)));
        let n = atlas.bands[0].n;
        let step = 2.0 / n as f64;
        let (i, j) = (((xi + 1.0) / step).floor() as i64, ((eta + 1.0) / step).floor() as i64);
        assert!((0..n).contains(&i) && (0..n).contains(&j), "({i},{j}) leaves the {face:?} chart");
        let patch = Patch::Shell { band: 0, face };
        let (origin, _) = atlas.storage_box(patch);
        let stored = atlas.storage(patch, [i, 0, j]);
        let ground = self.generator.surface(Face::PosY, stored[0] as i32, stored[2] as i32);
        assert_ne!(ground, i32::MIN, "{face:?} column has no surface");
        let local_y = ground as f64 - origin[1] as f64;
        let surf = atlas.embed(patch, DVec3::new(i as f64 + 0.5, local_y, j as f64 + 0.5));
        surf + (surf - atlas.centre).normalize() * above
    }

    /// Headless settle: centre on `pos`, fill the mesh box with data, mark every
    /// in-view chunk as a final Air mesh, and drain worklists so
    /// [`entry_complete`](Self::entry_complete) holds without an Engine.
    #[cfg(test)]
    pub fn settle_around(&mut self, pos: DVec3) {
        // Same eyes as `stream`: a charted body stands in storage, so a later
        // quiet frame does not see a boundary cross and demand an engine.
        let (pos, far) = self.place_eyes(pos);
        let center = eye_chunk(pos);
        self.center = Some(center);
        self.set_far_center(eye_chunk(far));
        let _ = self.adopt_fold(center);
        for coord in self.mesh_box(center).coords() {
            self.ensure_data(coord);
            if let Some(loaded) = self.chunks.get_mut(&coord)
                && !matches!(loaded.state, MeshState::Air | MeshState::Ready(_))
            {
                loaded.state = MeshState::Air;
            }
        }
        self.mesh_worklist.clear();
        self.light_worklist.clear();
        self.generating.clear();
        self.light_inflight.clear();
        self.upload_queue.clear();
        self.light_apply_queue.clear();
        self.section_upload_queue.clear();
        self.section_desired.clear();
        self.light_gate.degraded.clear();
        self.light_gate.dirty.clear();
        self.light_gate.blocked_since.clear();
        self.pending_fresh.take();
        self.pending_gen.take();
        self.pending_dirty.take();
        self.pending_sections.take();
        let _ = self.worker_pool();
    }
}

/// GPU write for a palette-growth step. `Set` is the initial bind; `Append`
/// is every later growth (ids above `uploaded_len`, in id order).
#[derive(Debug)]
enum TextureUpload<'a> {
    Set(&'a [Vec<u8>]),
    Append(&'a [Vec<u8>]),
}

/// Layers to send for the current cache vs last uploaded count, clamped to
/// the device layer cap (`visible`). Prefix layers are never rewritten.
fn plan_texture_upload(
    cache: &[Vec<u8>],
    uploaded_len: usize,
    visible: usize,
) -> Option<TextureUpload<'_>> {
    if visible == 0 {
        return None;
    }
    if uploaded_len == 0 {
        return Some(TextureUpload::Set(&cache[..visible]));
    }
    if visible > uploaded_len {
        return Some(TextureUpload::Append(&cache[uploaded_len..visible]));
    }
    None
}

#[cfg(test)]
mod flight_bench;

#[cfg(test)]
mod tests {
    use super::super::StreamLane;
    use super::*;
    use crate::coord::ChunkCoord;

    #[test]
    fn texture_growth_appends_only_new_layers_in_id_order() {
        let layer = |id: u8| vec![id; 4];
        let mut cache = vec![layer(0), layer(1), layer(2)];
        let mut uploaded_len = 0usize;
        let cap = 8usize;

        let visible = cache.len().min(cap);
        match plan_texture_upload(&cache, uploaded_len, visible) {
            Some(TextureUpload::Set(layers)) => {
                assert_eq!(layers.len(), 3);
                assert_eq!(layers[0], layer(0));
                assert_eq!(layers[1], layer(1));
                assert_eq!(layers[2], layer(2));
            }
            other => panic!("initial upload must set, got {other:?}"),
        }
        uploaded_len = visible;
        assert_eq!(uploaded_len, 3);

        cache.push(layer(3));
        cache.push(layer(4));
        let visible = cache.len().min(cap);
        match plan_texture_upload(&cache, uploaded_len, visible) {
            Some(TextureUpload::Append(layers)) => {
                assert_eq!(layers, &[layer(3), layer(4)]);
                assert_eq!(
                    uploaded_len + layers.len(),
                    visible,
                    "append is exactly the ids above uploaded_len"
                );
            }
            other => panic!("growth must append, got {other:?}"),
        }
        uploaded_len = visible;
        assert_eq!(uploaded_len, 5);

        let cap = 5usize;
        cache.push(layer(5));
        let visible = cache.len().min(cap);
        assert!(
            plan_texture_upload(&cache, uploaded_len, visible).is_none(),
            "past the layer cap, nothing is re-sent"
        );
        assert_eq!(uploaded_len, 5);
    }

    #[test]
    fn revision_rebuild_resets_to_a_set() {
        let cache = vec![vec![1u8; 4], vec![2; 4], vec![3; 4]];
        match plan_texture_upload(&cache, 0, cache.len()) {
            Some(TextureUpload::Set(layers)) => assert_eq!(layers.len(), 3),
            other => panic!("revision rebuild must set, got {other:?}"),
        }
    }

    #[test]
    fn stream_pacer_scales_with_useful_chunk_lifetime_and_recovers_gradually() {
        assert_eq!(StreamPacer::target_effort(0.0), 1.0);
        assert_eq!(StreamPacer::target_effort(FULL_EFFORT_SPEED_MPS), 1.0);
        assert!((StreamPacer::target_effort(48.0) - 0.5).abs() < f32::EPSILON);
        assert_eq!(StreamPacer::target_effort(10_000.0), MIN_STREAM_EFFORT);
        assert_eq!(StreamPacer::target_effort(f64::INFINITY), MIN_STREAM_EFFORT);
        assert_eq!(StreamPacer::target_effort(f64::NAN), 1.0);

        let mut pacer = StreamPacer::default();
        pacer.update(DVec3::new(200.0, 0.0, 0.0), 1.0 / 60.0);
        assert_eq!(pacer.effort(), MIN_STREAM_EFFORT, "shedding is immediate");
        assert_eq!(pacer.active_workers(12), 2);
        assert_eq!(pacer.floor(32), 5);
        assert_eq!(pacer.section_uploads(), 1);

        pacer.update(DVec3::ZERO, 0.1);
        assert!(
            pacer.effort() > MIN_STREAM_EFFORT && pacer.effort() < 1.0,
            "recovery ramps instead of releasing a one-frame catch-up burst"
        );
        for _ in 0..100 {
            pacer.update(DVec3::ZERO, 0.1);
        }
        assert_eq!(pacer.effort(), 1.0);
    }

    #[test]
    fn stream_pacer_runs_full_workers_for_queued_work_at_rest() {
        let mut pacer = StreamPacer::default();
        pacer.update(DVec3::new(200.0, 0.0, 0.0), 1.0 / 60.0);
        assert_eq!(pacer.active_workers(12), 2, "travel still sheds");
        pacer.set_boost(true, 0.001);
        assert!(
            !pacer.boosting(),
            "queued work during travel must not lift the floor"
        );
        assert_eq!(pacer.active_workers(12), 2);
        assert_eq!(pacer.near_queue_cap(12), 8);

        pacer.update(DVec3::ZERO, 1.0 / 60.0);
        pacer.set_boost(true, 0.001);
        assert!(pacer.boosting(), "cheap rest frame with leftover work");
        assert_eq!(pacer.active_workers(12), 12);
        assert_eq!(pacer.near_queue_cap(12), NEAR_REST_QUEUE_CAP);
        assert_eq!(pacer.floor(32), 32);
        assert!(
            pacer.duration(Duration::from_millis(1)) >= Duration::from_millis(1),
            "rest boost restores the full admission window"
        );

        pacer.set_boost(true, 0.020);
        assert!(
            !pacer.boosting(),
            "an already-expensive pass keeps the travel floor"
        );
        let expected = ((12.0 * pacer.effort()).ceil() as usize).clamp(1, 12);
        assert_eq!(pacer.active_workers(12), expected);
    }

    /// A degraded drawn chunk whose neighbourhood becomes light-ready WITHOUT
    /// a border event (nothing re-seeds it) is promoted by the gate's relit
    /// sweep — through the ASYNC rebuild path: old mesh carried and drawing,
    /// no sync `Dirty` involvement.
    #[test]
    fn relit_degraded_chunk_promotes_through_the_async_path() {
        let mut world = World::generate();
        let c = ChunkCoord::new(0, 0, 0);
        world.center = Some(c);
        for n in std::iter::once(c).chain(Face::ALL.iter().map(|&f| c.step(f))) {
            world.chunks.get_mut(&n).expect("pregenerated").light = Some(light::LightGrid::dark());
        }
        let h = voxel_engine::MeshHandle::from_raw_parts(21, 1);
        let meshes = super::super::ChunkMeshes::from_upload_handles(ByPass::from_fn(|p| {
            (p == voxel_engine::Pass::Opaque).then_some(h)
        }))
        .expect("one pass present");
        world.chunks.get_mut(&c).unwrap().state = MeshState::Ready(meshes);
        world.mark_degraded(c, true);
        world.light_worklist.clear();
        world.light_inflight.clear();
        world.light_apply_queue.clear();
        world.pending_dirty.take();

        world.tick_light_gate();

        let state = &world.chunks[&c].state;
        assert!(
            matches!(
                state,
                MeshState::NeedsMesh {
                    building: false,
                    prev: Some(_)
                }
            ),
            "promoted through the async rebuild: {state:?}"
        );
        assert!(world.mesh_worklist.contains(&c), "seeded for the rebuild");
        assert!(
            !world.pending_dirty.get(),
            "the sync dirty path is not involved"
        );
    }

    /// The admit loop evicts a light-blocked seed from `mesh_worklist` and
    /// starts its degrade timer at that EVENT (`MeshLane::on_blocked`); expiry
    /// raises no event of its own, so `tick_light_gate`'s sweep over the timed
    /// map — NOT worklist membership — is what re-seeds the chunk once
    /// `light_wait_expired` makes it mesh-ready. This pins both halves: an
    /// un-expired evicted chunk is NOT re-seeded by a tick (its re-seed must
    /// come from a real event), and an expired one always is.
    #[test]
    fn light_wait_expiry_reseeds_an_evicted_chunk_without_stranding_it() {
        let mut world = World::generate();
        let c = ChunkCoord::new(0, 0, 0);
        world.center = Some(c);

        // C and its 6 face neighbours must have data (generate() pregenerates
        // near spawn); C's own light stays unset so `light_ready(c)` is false
        // and `chunk_light_blocked(c)` holds without touching the light worklist.
        for n in std::iter::once(c).chain(crate::coord::Face::ALL.iter().map(|&f| c.step(f))) {
            world
                .chunks
                .get_mut(&n)
                .expect("neighbourhood pregenerated near spawn");
        }
        world.chunks.get_mut(&c).unwrap().light = None;
        world.chunks.get_mut(&c).unwrap().state = MeshState::needs_mesh();
        assert!(world.neighbours_have_data(c));
        assert!(world.in_mesh_box(c));
        assert!(
            world.chunk_light_blocked(c),
            "no published light: c is light-blocked"
        );

        // Seed C and run the REAL admission pass: it must evict the blocked
        // seed and start its wait timer through the `on_blocked` event.
        world.mesh_worklist.insert(c);
        world.pending_fresh.set();
        super::super::admit::<MeshLane>(
            &mut world,
            c,
            voxel_engine::producer::Budget::Millis(5.0),
        );
        assert!(
            !world.mesh_worklist.contains(&c),
            "the blocked seed is evicted"
        );
        assert!(
            world.light_gate.blocked_since.contains_key(&c),
            "eviction must start the wait timer"
        );
        assert!(!world.light_wait_expired(c), "not yet past the wait budget");

        // Before expiry, a tick must NOT re-seed it — pre-expiry re-seeds come
        // from real events, never from the per-pass sweep.
        world.tick_light_gate();
        assert!(
            !world.mesh_worklist.contains(&c),
            "an un-expired evicted chunk is not re-seeded by the sweep"
        );

        // Back-date the timer past LIGHT_WAIT_DEGRADE without sleeping — the
        // gate's expiry is wall-clock, so this is the only deterministic way to
        // reach the expired state.
        world.light_gate.blocked_since.insert(
            c,
            Instant::now() - LIGHT_WAIT_DEGRADE - Duration::from_millis(1),
        );
        assert!(
            world.light_wait_expired(c),
            "back-dated timer must read as expired"
        );

        // The next `tick_light_gate` must re-seed it purely from the expired
        // timer, with no dependency on `c` already being in the worklist.
        world.tick_light_gate();
        assert!(
            world.mesh_worklist.contains(&c),
            "eviction-stall regression: an expired-but-evicted chunk must be re-seeded"
        );
        assert!(
            world.pending_fresh.get(),
            "re-armed: the fresh scan will pick it up"
        );
        assert!(
            <MeshLane as StreamLane>::ready(&world, c),
            "expired wait admits a degraded mesh even though light never settled"
        );
    }

    /// A degraded `Ready` chunk whose neighbour light is permanently missing
    /// is, at quiescence, rebuilt asynchronously: the drawn mesh is carried,
    /// the terminal set records that missing planes are settled dark, and
    /// `MeshLane::submit` snapshots non-degraded. Claim (not submit) drops
    /// the degraded and terminal marks.
    #[test]
    fn degraded_ready_chunk_promotes_through_terminal_async_path() {
        let mut world = World::generate();
        let c = ChunkCoord::new(0, 0, 0);
        world.center = Some(c);
        let missing = c.step(Face::PosX);
        for n in std::iter::once(c).chain(Face::ALL.iter().map(|&f| c.step(f))) {
            let loaded = world.chunks.get_mut(&n).expect("pregenerated");
            loaded.light = if n == missing {
                None
            } else {
                Some(light::LightGrid::dark())
            };
        }
        let h = voxel_engine::MeshHandle::from_raw_parts(21, 1);
        let meshes = super::super::ChunkMeshes::from_upload_handles(ByPass::from_fn(|p| {
            (p == voxel_engine::Pass::Opaque).then_some(h)
        }))
        .expect("one pass present");
        world.chunks.get_mut(&c).unwrap().state = MeshState::Ready(meshes);
        world.mark_degraded(c, true);
        world.generating.clear();
        world.mesh_worklist.clear();
        world.light_worklist.clear();
        world.light_inflight.clear();
        world.light_apply_queue.clear();
        world.pending_dirty.take();
        assert!(!world.light_ready(c), "one neighbour grid is permanently missing");

        world.flush_degraded_terminal();

        let state = &world.chunks[&c].state;
        assert!(
            matches!(
                state,
                MeshState::NeedsMesh {
                    building: false,
                    prev: Some(_)
                }
            ),
            "promoted through the async rebuild: {state:?}"
        );
        assert!(world.mesh_worklist.contains(&c), "seeded for the rebuild");
        assert!(
            world.light_terminal.contains(&c),
            "missing neighbour light is terminal"
        );
        assert!(
            world.light_gate.degraded.contains(&c),
            "degraded flag stays until the rebuild is claimed"
        );
        assert!(
            <MeshLane as StreamLane>::ready(&world, c),
            "terminal membership admits the rebuild without another light wait"
        );
        assert!(
            !world.pending_dirty.get(),
            "the sync dirty path is not involved"
        );

        let job = <MeshLane as StreamLane>::submit(&mut world, c).expect("terminal mesh job");
        let pipeline::Job::Mesh { snapshot, .. } = job else {
            panic!("expected a mesh job");
        };
        let shell = snapshot.light.expect("lighting on");
        assert_eq!(
            shell.at(CHUNK_SIZE as i32, 8, 8),
            light::Lumel::DARK,
            "terminal snapshot reads the missing +X neighbour as settled dark"
        );
        assert!(
            world.light_gate.degraded.contains(&c),
            "submit must not mutate the degraded set"
        );
        assert!(
            world.light_terminal.contains(&c),
            "submit must not drop the terminal mark (a rejected submit retries)"
        );

        <MeshLane as StreamLane>::claim(&mut world, c);
        assert!(
            !world.light_gate.degraded.contains(&c),
            "claim marks the snapshot non-degraded"
        );
        assert!(
            world.light_terminal.is_empty(),
            "claim consumes the terminal mark"
        );
    }

    /// One GPU-free stream pass: light-gate, mesh admit (submit + claim +
    /// install the carried mesh as Ready — tests have no Engine), terminal
    /// flush. Matches the live `stream` order so promotion completes in one
    /// quiescence round after the seed.
    fn pump_terminal_mesh(world: &mut World) {
        world.tick_light_gate();
        let seeds: Vec<Coord> = world.mesh_worklist.iter().copied().collect();
        for key in seeds {
            if <MeshLane as StreamLane>::in_flight(world, key) {
                continue;
            }
            if !<MeshLane as StreamLane>::ready(world, key) {
                world.mesh_worklist.remove(&key);
                <MeshLane as StreamLane>::on_blocked(world, key);
                continue;
            }
            let _job = <MeshLane as StreamLane>::submit(world, key).expect("mesh job");
            <MeshLane as StreamLane>::claim(world, key);
            if let Some(loaded) = world.chunks.get_mut(&key) {
                if let MeshState::NeedsMesh {
                    building: true,
                    prev,
                } = &mut loaded.state
                {
                    let next = match prev.take() {
                        Some(m) => MeshState::Ready(m),
                        None => MeshState::Air,
                    };
                    loaded.retire_logged(next);
                    super::super::adjust_count(&mut world.building_meshes, true, false);
                }
            }
        }
        world.flush_degraded_terminal();
    }

    /// A degraded mesh whose face neighbour has unloaded (trailing-edge /
    /// load-set-edge) still promotes at quiescence: the terminal mark admits
    /// the rebuild without neighbour data, a stranded `NeedsMesh` is re-seeded,
    /// and `entry_complete` becomes true with the centre set.
    #[test]
    fn degraded_chunk_promotes_when_a_neighbour_is_missing() {
        let mut world = World::generate();
        world.lod2 = false;
        world.set_view_distances(2, 2);
        let cy = world.generator.height(0, 0).div_euclid(CHUNK_SIZE as i32);
        let center = ChunkCoord::new(0, cy, 0);
        world.center = Some(center);
        let edge = ChunkCoord::new(2, cy, 0);
        let missing = edge.step(Face::PosX);
        assert!(world.in_mesh_box(edge), "edge chunk is drawn");
        assert!(
            !world.in_mesh_box(missing),
            "the unloaded neighbour sits outside the mesh box"
        );

        for coord in world.mesh_box(center).coords() {
            if !world.chunks.contains_key(&coord) {
                world.ensure_data(coord);
            }
            let loaded = world.chunks.get_mut(&coord).expect("in-box data");
            loaded.state = MeshState::Air;
            if loaded.light.is_none() {
                loaded.light = Some(light::LightGrid::dark());
            }
        }

        let h = voxel_engine::MeshHandle::from_raw_parts(77, 1);
        let meshes = super::super::ChunkMeshes::from_upload_handles(ByPass::from_fn(|p| {
            (p == voxel_engine::Pass::Opaque).then_some(h)
        }))
        .expect("one pass present");
        world.chunks.get_mut(&edge).unwrap().state = MeshState::Ready(meshes);
        world.mark_degraded(edge, true);
        world.chunks.remove(&missing);
        world.generating.clear();
        world.mesh_worklist.clear();
        world.light_worklist.clear();
        world.light_inflight.clear();
        world.light_apply_queue.clear();
        world.light_gate.blocked_since.clear();
        world.upload_queue.clear();
        assert!(
            !world.neighbours_have_data(edge),
            "the face neighbour is gone"
        );
        assert!(!world.light_ready(edge));

        world.flush_degraded_terminal();
        assert!(
            matches!(
                world.chunks[&edge].state,
                MeshState::NeedsMesh {
                    building: false,
                    prev: Some(_)
                }
            ),
            "Ready degraded promotes through the async path"
        );
        assert!(world.light_terminal.contains(&edge));
        assert!(
            <MeshLane as StreamLane>::ready(&world, edge),
            "terminal admits without neighbour data"
        );

        // The live admit loop evicts a blocked seed; a later flush must
        // re-seed the stranded NeedsMesh instead of assuming another path
        // will resolve it.
        world.mesh_worklist.remove(&edge);
        world.pending_fresh.take();
        world.flush_degraded_terminal();
        assert!(
            world.mesh_worklist.contains(&edge),
            "stuck NeedsMesh is re-seeded"
        );
        assert!(world.pending_fresh.get());
        assert!(world.light_terminal.contains(&edge));

        let deadline = Instant::now() + Duration::from_secs(5);
        while !world.entry_complete() {
            assert!(
                Instant::now() < deadline,
                "promotion did not settle: {}",
                world.entry_debug()
            );
            pump_terminal_mesh(&mut world);
        }
        assert!(
            !world.light_gate.degraded.contains(&edge),
            "the rebuild claim cleared the degraded flag"
        );
        assert!(
            matches!(world.chunks[&edge].state, MeshState::Ready(_)),
            "the chunk shows a Ready mesh"
        );
        assert!(world.entry_complete(), "centre is set and the box is final");
        assert!(world.chunks[&edge].state.live_meshes().unwrap().draws(h));

        // A later real neighbour arrival must rebuild the promoted chunk so
        // the dark-plane snapshot is not permanent.
        world.ensure_data(missing);
        assert!(
            matches!(
                world.chunks[&edge].state,
                MeshState::NeedsMesh {
                    building: false,
                    prev: Some(_)
                }
            ),
            "storing the missing neighbour rebuilds the terminal-promoted chunk"
        );
        assert!(world.mesh_worklist.contains(&edge));
    }

    /// A degraded chunk that has left the mesh box (still loaded in the unload
    /// hysteresis) cannot be admitted, so flush drops the flag instead of
    /// re-seeding a seed admit will just evict.
    #[test]
    fn flush_drops_degraded_outside_the_mesh_box() {
        let mut world = World::generate();
        world.lod2 = false;
        world.set_view_distances(2, 2);
        let cy = world.generator.height(0, 0).div_euclid(CHUNK_SIZE as i32);
        let center = ChunkCoord::new(0, cy, 0);
        world.center = Some(center);
        let outside = ChunkCoord::new(3, cy, 0);
        assert!(!world.in_mesh_box(outside));
        if !world.chunks.contains_key(&outside) {
            world.ensure_data(outside);
        }
        for coord in world.mesh_box(center).coords() {
            if !world.chunks.contains_key(&coord) {
                world.ensure_data(coord);
            }
            world.chunks.get_mut(&coord).unwrap().state = MeshState::Air;
        }
        let h = voxel_engine::MeshHandle::from_raw_parts(78, 1);
        let meshes = super::super::ChunkMeshes::from_upload_handles(ByPass::from_fn(|p| {
            (p == voxel_engine::Pass::Opaque).then_some(h)
        }))
        .expect("one pass present");
        world.chunks.get_mut(&outside).unwrap().state = MeshState::Ready(meshes);
        world.mark_degraded(outside, true);
        world.generating.clear();
        world.mesh_worklist.clear();
        world.light_worklist.clear();
        world.light_inflight.clear();
        world.light_apply_queue.clear();
        world.light_gate.blocked_since.clear();

        world.flush_degraded_terminal();
        assert!(
            !world.light_gate.degraded.contains(&outside),
            "out-of-box degraded is not owed a remesh"
        );
        assert!(
            !world.mesh_worklist.contains(&outside),
            "must not re-seed a seed admit will evict"
        );
        assert!(
            matches!(world.chunks[&outside].state, MeshState::Ready(_)),
            "the drawn mesh is left in place"
        );

        world.chunks.get_mut(&outside).unwrap().state = MeshState::NeedsMesh {
            building: false,
            prev: Some(
                super::super::ChunkMeshes::from_upload_handles(ByPass::from_fn(|p| {
                    (p == voxel_engine::Pass::Opaque).then_some(h)
                }))
                .expect("one pass present"),
            ),
        };
        world.mark_degraded(outside, true);
        world.mesh_worklist.clear();
        world.flush_degraded_terminal();
        assert!(!world.light_gate.degraded.contains(&outside));
        assert!(!world.mesh_worklist.contains(&outside));
    }

    fn ready_handle(id: u32) -> super::super::ChunkMeshes {
        let h = voxel_engine::MeshHandle::from_raw_parts(id, 1);
        super::super::ChunkMeshes::from_upload_handles(ByPass::from_fn(|p| {
            (p == voxel_engine::Pass::Opaque).then_some(h)
        }))
        .expect("one pass present")
    }

    /// Changed settles mark `light_dirty` and do not remesh while the
    /// 27-neighbourhood still has pending light work; one tick after the
    /// neighbourhood quiets issues a single async rebuild.
    #[test]
    fn changed_settle_remeshes_once_at_nhood_fixpoint() {
        let mut world = World::generate();
        let c = Coord::new(0, 0, 0);
        world.center = Some(c);
        world.chunks.get_mut(&c).unwrap().state = MeshState::Ready(ready_handle(91));
        world.chunks.get_mut(&c).unwrap().light = Some(light::LightGrid::open_sky());
        world.light_worklist.clear();
        world.light_inflight.clear();
        world.light_apply_queue.clear();
        world.pending_dirty.take();
        let pending = c.step(Face::PosX);
        world.light_worklist.insert(pending);
        let rev = world.chunks[&c].rev;

        world.settle_light(c, light::LightGrid::dark());
        world.settle_light(c, light::LightGrid::full());
        world.settle_light(c, light::LightGrid::dark());
        assert!(
            matches!(world.chunks[&c].state, MeshState::Ready(_)),
            "pending nhood light must not remesh"
        );
        assert_eq!(world.chunks[&c].rev, rev, "no rev bump while waiting");
        assert!(world.light_gate.dirty.contains_key(&c));
        assert_eq!(world.remesh_stats.remesh_async_calls, 0);

        world.light_worklist.clear();
        world.light_gate.dirty.retain(|&k, _| k == c);
        world.tick_light_gate();
        assert!(
            matches!(
                world.chunks[&c].state,
                MeshState::NeedsMesh {
                    building: false,
                    prev: Some(_)
                }
            ),
            "quiet nhood promotes one async rebuild"
        );
        assert_eq!(world.remesh_stats.remesh_async_calls, 1);
        assert!(!world.light_gate.dirty.contains_key(&c));
    }

    /// The degrade timer still promotes a dirty Ready chunk while its
    /// neighbourhood has pending light work, so the first update is not
    /// delayed past LIGHT_WAIT_DEGRADE.
    #[test]
    fn dirty_chunk_promotes_when_degrade_timer_expires() {
        let mut world = World::generate();
        let c = Coord::new(0, 0, 0);
        world.center = Some(c);
        world.chunks.get_mut(&c).unwrap().state = MeshState::Ready(ready_handle(92));
        world.chunks.get_mut(&c).unwrap().light = Some(light::LightGrid::open_sky());
        world.light_worklist.clear();
        world.light_inflight.clear();
        world.light_apply_queue.clear();
        world.settle_light(c, light::LightGrid::dark());
        world.light_worklist.insert(c.step(Face::PosY));
        assert!(matches!(world.chunks[&c].state, MeshState::Ready(_)));

        world.light_gate.dirty.insert(
            c,
            Instant::now() - LIGHT_WAIT_DEGRADE - Duration::from_millis(1),
        );
        world.tick_light_gate();
        assert!(
            matches!(
                world.chunks[&c].state,
                MeshState::NeedsMesh {
                    building: false,
                    prev: Some(_)
                }
            ),
            "expired dirty mark remeshes even with pending nhood light"
        );
    }

    /// First mesh of a never-drawn chunk is not delayed: settle keeps it on
    /// the worklist without a rev-bumping remesh_async.
    #[test]
    fn first_mesh_is_not_delayed_by_light_dirty() {
        let mut world = World::generate();
        let c = Coord::new(0, 0, 0);
        world.center = Some(c);
        world.chunks.get_mut(&c).unwrap().state = MeshState::needs_mesh();
        world.chunks.get_mut(&c).unwrap().light = Some(light::LightGrid::open_sky());
        world.light_worklist.clear();
        world.light_inflight.clear();
        world.light_apply_queue.clear();
        world.mesh_worklist.clear();
        let rev = world.chunks[&c].rev;
        world.light_worklist.insert(c.step(Face::NegZ));
        world.settle_light(c, light::LightGrid::dark());
        assert_eq!(world.chunks[&c].rev, rev, "first mesh must not take a rev bump");
        assert!(world.mesh_worklist.contains(&c), "still seeded for the first mesh");
        assert!(
            matches!(
                world.chunks[&c].state,
                MeshState::NeedsMesh {
                    building: false,
                    prev: None
                }
            )
        );
        world.tick_light_gate();
        assert_eq!(world.chunks[&c].rev, rev);
        assert_eq!(world.remesh_stats.remesh_async_calls, 0);
    }

    /// `flush_degraded_terminal` promotes a per-coord-quiet degraded chunk
    /// even while unrelated light work is still queued elsewhere.
    #[test]
    fn flush_degraded_is_per_coord_not_global() {
        let mut world = World::generate();
        let c = Coord::new(0, 0, 0);
        world.center = Some(c);
        let missing = c.step(Face::PosX);
        for n in std::iter::once(c).chain(Face::ALL.iter().map(|&f| c.step(f))) {
            let loaded = world.chunks.get_mut(&n).expect("pregenerated");
            loaded.light = if n == missing {
                None
            } else {
                Some(light::LightGrid::dark())
            };
        }
        world.chunks.get_mut(&c).unwrap().state = MeshState::Ready(ready_handle(93));
        world.mark_degraded(c, true);
        world.generating.clear();
        world.mesh_worklist.clear();
        world.light_worklist.clear();
        world.light_inflight.clear();
        world.light_apply_queue.clear();
        world.pending_dirty.take();
        let far = Coord::new(8, 0, 8);
        world.light_worklist.insert(far);
        assert!(!world.near_quiescent(), "global light work is still queued");
        assert!(!world.light_ready(c));
        assert!(world.light_nhood_quiet(c), "this coord's 27-nhood is idle");

        world.flush_degraded_terminal();
        assert!(
            matches!(
                world.chunks[&c].state,
                MeshState::NeedsMesh {
                    building: false,
                    prev: Some(_)
                }
            ),
            "per-coord quiet promotes without waiting for global quiescence"
        );
        assert!(world.light_terminal.contains(&c));
    }

    /// `LightLane::submit` must not skip a job just because `trivial_light`
    /// would succeed — that decision was made at store time.
    #[test]
    fn light_lane_submit_does_not_recheck_trivial_light() {
        let mut world = World::generate();
        let coord = world
            .chunks
            .iter()
            .find_map(|(&c, l)| l.light.as_ref().map(|_| c))
            .expect("generate publishes at least one grid");
        assert!(
            <LightLane as StreamLane>::submit(&mut world, coord).is_some(),
            "submit must not re-check trivial_light"
        );
    }

    fn assert_ceilings_eq(got: &light::CeilingWindow, slow: &light::CeilingWindow) {
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                assert_eq!(
                    got.surface_at(lx, lz),
                    slow.surface_at(lx, lz),
                    "ceiling lx={lx} lz={lz}"
                );
            }
        }
    }

    /// `accept_column` caches the skylight ceiling from worker heights (not
    /// `height()`) and `trivial_light` publishes the same grid the slow path
    /// would. Covers the flat world, diffusion, and an edited-roof raise.
    #[test]
    fn accept_column_caches_ceiling_and_trivial_light_matches_slow() {
        use crate::render_config::RenderConfig;
        use crate::world::generation::WorldgenKind;

        for kind in [WorldgenKind::Flat, WorldgenKind::Diffusion] {
            let mut world = World::with_kind(7, RenderConfig::default(), kind, false);
            // Flat: a chunk above the hills. Diffusion: the start world is charted, so the
            // air chunk and the roof sit on its +Y storage column, not the physical origin.
            let (coord, roof_y) = if kind == WorldgenKind::Flat {
                (Coord::new(1, 31, -2), 200)
            } else {
                use crate::space::atlas::Patch;
                let home = world.generator.cosmos().expect("cosmos").home();
                let atlas = world
                    .generator
                    .atlases()
                    .iter()
                    .find(|a| (a.centre - home.centre_f()).length() < 1.0)
                    .expect("the start world is charted");
                let n = atlas.bands[0].n;
                let s = atlas.storage(Patch::Shell { band: 0, face: Face::PosY }, [n / 2, 0, n / 2]);
                let h = world.generator.height(s[0] as i32 + 8, s[2] as i32 + 8);
                assert_ne!(h, i32::MIN, "chart column has no surface");
                let roof_y = h + 8;
                let cs = CHUNK_SIZE as i32;
                let cx = (s[0] as i32).div_euclid(cs);
                let cz = (s[2] as i32).div_euclid(cs);
                let cy = roof_y.div_euclid(cs) + 2;
                (Coord::new(cx, cy, cz), roof_y)
            };
            world.center = Some(coord);

            let stone = world.registry.id_by_label("rock").expect("builtin Stone");
            // Roof in this column, below the stored chunk: raise before store.
            world.set_block(
                coord.x * CHUNK_SIZE as i32 + 3,
                roof_y,
                coord.z * CHUNK_SIZE as i32 + 5,
                stone,
            );

            let slow = world.capture_ceiling_slow(coord);
            let key = ColumnKey { face: Face::PosY, a: coord.x, b: coord.z };
            assert!(
                !world.ceilings.contains_key(&key),
                "slow helper must not warm the cache"
            );

            let (datas, heights) = world.generator.generate_column(key, coord.y..=coord.y);
            let chunks: Vec<_> = datas
                .into_iter()
                .map(|(alt, data)| {
                    let placed = key.chunk(alt);
                    (placed, Chunk::from_data(placed.x, placed.y, placed.z, data))
                })
                .collect();
            world.accept_column(key, chunks, Box::new(heights));

            let cached = world
                .ceilings
                .get(&key)
                .expect("accept_column installs the ceiling before store");
            assert_ceilings_eq(cached.as_ref(), &slow);
            assert!(
                cached.surface_at(3, 5) >= roof_y + 1,
                "edited roof must raise the cached ceiling"
            );

            let grid = world.chunks[&coord]
                .light
                .as_ref()
                .expect("uniform-air above the surface publishes trivial light");
            let world_y0 = coord.y * CHUNK_SIZE as i32;
            let all_open = (0..CHUNK_SIZE)
                .all(|lz| (0..CHUNK_SIZE).all(|lx| slow.open_above(lx, lz, world_y0)));
            assert!(all_open, "fixture sits fully above the (raised) ceiling");
            assert!(
                grid == &light::LightGrid::open_sky(),
                "trivial light must be open_sky"
            );
        }
    }

    #[test]
    fn neighbour_blocklight_near_reads_the_settled_flag() {
        let mut world = World::generate();
        let cy = world.generator.height(0, 0).div_euclid(CHUNK_SIZE as i32);
        let c = Coord::new(0, cy, 0);
        let n = c.step(Face::PosX);
        world.chunks.get(&c).expect("generate preloads the origin");
        world
            .chunks
            .get(&n)
            .expect("generate preloads the face neighbour");
        world.settle_light(n, light::LightGrid::dark());
        assert!(!world.chunks[&n].has_blocklight);
        assert!(!world.neighbour_blocklight_near(c));
        world.settle_light(n, light::LightGrid::full());
        assert!(world.chunks[&n].has_blocklight);
        assert!(world.neighbour_blocklight_near(c));
        world.settle_light(n, light::LightGrid::full());
        assert!(
            world.chunks[&n].has_blocklight,
            "identical re-settle keeps the flag"
        );
    }

    /// First publish of a dark grid matches the missing-neighbour shell, so
    /// no face moved and no neighbour is seeded.
    #[test]
    fn dark_first_publish_seeds_no_neighbour() {
        let mut world = World::generate();
        let c = Coord::new(0, 0, 0);
        world.chunks.get_mut(&c).unwrap().light = None;
        world.chunks.get_mut(&c.step(Face::PosX)).unwrap().state = MeshState::needs_mesh();
        world.light_worklist.clear();
        world.light_inflight.clear();
        world.mesh_worklist.clear();
        world.settle_light(c, light::LightGrid::dark());
        for &face in &Face::ALL {
            assert!(
                !world.light_worklist.contains(&c.step(face)),
                "dark first publish must not seed {face:?}"
            );
        }
        let n = c.step(Face::PosX);
        assert!(
            world.mesh_worklist.contains(&n),
            "first publish still re-seeds a waiting neighbour's mesh"
        );
    }

    /// A moved face seeds only neighbours that already have data; missing
    /// neighbours are not inserted (store_chunk / first-publish constraint).
    #[test]
    fn settle_seeds_only_neighbours_that_have_data() {
        let mut world = World::generate();
        let c = Coord::new(0, 0, 0);
        let missing = c.step(Face::PosY);
        let present = c.step(Face::PosX);
        world.chunks.remove(&missing);
        world.chunks.get_mut(&c).unwrap().light = None;
        assert!(
            world.chunks.contains_key(&present),
            "generate preloads a lateral neighbour"
        );
        world.chunks.get_mut(&present).unwrap().light = Some(light::LightGrid::dark());
        world.light_worklist.clear();
        world.light_inflight.clear();
        world.settle_light(c, light::LightGrid::open_sky());
        assert!(
            !world.light_worklist.contains(&missing),
            "must not seed a neighbour without data"
        );
        assert!(
            world.light_worklist.contains(&present),
            "a loaded neighbour whose shared face moved must be seeded"
        );
    }

    /// An in-flight neighbour is marked, not re-inserted; the seed lands when
    /// its result integrates — even if that grid equals the one it already had.
    #[test]
    fn inflight_neighbour_reseeds_after_landing() {
        let mut world = World::generate();
        let c = Coord::new(0, 0, 0);
        let n = c.step(Face::PosX);
        world.chunks.get_mut(&c).unwrap().light = Some(light::LightGrid::dark());
        world.chunks.get_mut(&n).unwrap().light = Some(light::LightGrid::dark());
        world.light_worklist.clear();
        world.light_inflight.clear();
        world.light_inflight.insert(n);
        world.settle_light(c, light::LightGrid::open_sky());
        assert!(
            !world.light_worklist.contains(&n),
            "in-flight neighbour must not be re-inserted immediately"
        );
        assert!(world.chunks[&n].light_reseed);
        world.settle_light(n, light::LightGrid::dark());
        assert!(!world.chunks[&n].light_reseed);
        assert!(
            world.light_worklist.contains(&n),
            "re-seed after landing so the wave costs one extra flood"
        );
    }

    /// Interior-only change: faces match the previous grid, so no neighbour
    /// is light-seeded (the `border_changed` cut).
    #[test]
    fn settle_unchanged_faces_seeds_no_neighbour() {
        let mut world = World::generate();
        let c = Coord::new(0, 0, 0);
        world.chunks.get_mut(&c).unwrap().light = Some(light::LightGrid::dark());
        world.light_worklist.clear();
        world.light_inflight.clear();
        world.mesh_worklist.clear();
        let mut interior = light::LightGrid::dark();
        interior.set(
            Chunk::index(8, 8, 8),
            light::Lumel {
                sky: light::LightLevel::FULL,
                block: light::LightLevel::DARK,
            },
        );
        world.settle_light(c, interior);
        for &face in &Face::ALL {
            assert!(
                !world.light_worklist.contains(&c.step(face)),
                "unchanged face {face:?} must not seed its neighbour"
            );
        }
        assert!(
            world.mesh_worklist.contains(&c),
            "self is still mesh-seeded on a changed grid"
        );
    }

    /// A neighbour already on the worklist is not counted again: the pending
    /// flood reads live neighbour grids at admit.
    #[test]
    fn settle_does_not_recount_already_queued_neighbour() {
        let mut world = World::generate();
        let c = Coord::new(0, 0, 0);
        let n = c.step(Face::PosX);
        world.chunks.get_mut(&c).unwrap().light = Some(light::LightGrid::dark());
        for &face in &Face::ALL {
            if let Some(loaded) = world.chunks.get_mut(&c.step(face)) {
                loaded.light = Some(light::LightGrid::dark());
            }
        }
        world.light_worklist.clear();
        world.light_inflight.clear();
        world.light_worklist.insert(n);
        world.light_seed_inserts = 0;
        world.light_seed_split = super::super::LightSeedSplit::default();
        world.settle_light(c, light::LightGrid::open_sky());
        assert!(world.light_worklist.contains(&n));
        let other_loaded = Face::ALL
            .iter()
            .filter(|&&f| {
                let n2 = c.step(f);
                n2 != n && world.chunks.contains_key(&n2)
            })
            .count() as u64;
        assert_eq!(
            world.light_seed_split.border, other_loaded,
            "already-queued neighbour is not a counted insert"
        );
    }

    /// Two open-sky grids: the neighbour is already at the analytic result, so
    /// a first-publish face move against dark is a no-op flood.
    #[test]
    fn open_sky_does_not_reseed_open_sky_neighbour() {
        let mut world = World::generate();
        let c = Coord::new(0, 0, 0);
        let n = c.step(Face::PosX);
        world.chunks.get_mut(&c).unwrap().light = None;
        world.chunks.get_mut(&n).unwrap().light = Some(light::LightGrid::open_sky());
        world.light_worklist.clear();
        world.light_inflight.clear();
        world.settle_light(c, light::LightGrid::open_sky());
        assert!(
            !world.light_worklist.contains(&n),
            "open-sky neighbour cannot change when this chunk publishes open sky"
        );
    }

    /// `store_chunk` (via `ensure_data`) must not seed neighbours that have
    /// no data, even when the stored chunk publishes a non-dark first grid.
    #[test]
    fn store_chunk_does_not_seed_neighbours_without_data() {
        use crate::render_config::RenderConfig;
        let mut world = World::with_config_lazy(1, RenderConfig::default());
        let c = Coord::new(2, 25, -3);
        world.center = Some(c);
        world.ensure_data(c);
        assert!(world.chunks.contains_key(&c));
        for &face in &Face::ALL {
            let n = c.step(face);
            assert!(
                !world.light_worklist.contains(&n),
                "store must not seed neighbour {face:?} that has no data"
            );
        }
    }

    #[test]
    fn seed_light_counts_each_source() {
        let mut world = World::generate();
        let c = Coord::new(0, 0, 0);
        world.light_worklist.clear();
        world.light_seed_inserts = 0;
        world.light_seed_split = super::super::LightSeedSplit::default();
        world.seed_light(c, super::super::LightSeed::Store);
        world.seed_light(c, super::super::LightSeed::Border);
        world.seed_light(c, super::super::LightSeed::Edit);
        world.seed_light(c, super::super::LightSeed::Degrade);
        world.seed_light(c, super::super::LightSeed::Terminal);
        world.seed_light(c, super::super::LightSeed::Remesh);
        assert_eq!(world.light_seed_inserts, 6);
        let s = world.light_seed_split;
        assert_eq!(
            (s.store, s.border, s.edit, s.degrade, s.terminal, s.remesh),
            (1, 1, 1, 1, 1, 1)
        );
    }

    #[test]
    fn unload_leaving_is_the_old_minus_new_shell() {
        let mut world = World::generate();
        let cy = world.generator.height(0, 0).div_euclid(CHUNK_SIZE as i32);
        let a = Coord::new(0, cy, 0);
        world.center = Some(a);
        world.prev_unload_box = Some(world.unload_box(a));
        let b = Coord::new(2, 0, 0);
        let leaving = world.unload_leaving(world.unload_box(b));
        let old = world.unload_box(a);
        let new = world.unload_box(b);
        for &c in &leaving {
            assert!(old.contains(c) && !new.contains(c), "{c:?} not in old∖new");
        }
        for c in old.coords() {
            if new.contains(c)
                || !world.chunks.contains_key(&c)
                || world.spawn_slab.is_some_and(|s| s.contains(c))
            {
                continue;
            }
            assert!(leaving.contains(&c), "{c:?} loaded in old∖new must leave");
        }
    }

    /// Sync `ensure_data` (headless region, unclaimed boundary-cross centre)
    /// also installs from `generate_column` heights, so `trivial_light` never
    /// calls `height()`.
    #[test]
    fn ensure_data_caches_ceiling_matching_slow() {
        use crate::render_config::RenderConfig;

        let mut world = World::with_config_lazy(11, RenderConfig::default());
        let coord = Coord::new(2, 25, 1);
        world.ensure_data(coord);
        let cached = world
            .ceilings
            .get(&ColumnKey { face: Face::PosY, a: coord.x, b: coord.z })
            .expect("ensure_data installs the ceiling before store");
        let slow = world.capture_ceiling_slow(coord);
        assert_ceilings_eq(cached.as_ref(), &slow);
        assert!(
            world.chunks[&coord].light.as_ref() == Some(&light::LightGrid::open_sky()),
            "trivial light must be open_sky"
        );
    }

    #[test]
    fn async_upload_does_not_hash_and_identical_edit_skips_remesh() {
        mesh::reset_content_hash_calls();
        let data = mesh::new_chunk_mesh_data();
        let hash = mesh::content_hash(&data);
        assert_eq!(mesh::content_hash_calls(), 1);

        let mut world = World::generate();
        let c = *world.chunks.keys().next().expect("spawn chunks");
        let h = voxel_engine::MeshHandle::from_raw_parts(91, 1);
        let meshes = super::super::ChunkMeshes::from_upload_handles(ByPass::from_fn(|p| {
            (p == voxel_engine::Pass::Opaque).then_some(h)
        }))
        .expect("one pass present");
        {
            let loaded = world.chunks.get_mut(&c).unwrap();
            loaded.state = MeshState::Dirty {
                prev: Some(meshes),
            };
            loaded.mesh_hash = Some(hash);
            loaded.visible = true;
        }

        mesh::reset_content_hash_calls();
        world.upload_chunk_without_gpu(c, Some(hash));
        assert_eq!(mesh::content_hash_calls(), 0, "upload_chunk never hashes");
        assert!(
            matches!(world.chunks[&c].state, MeshState::Ready(_)),
            "identical edit remesh keeps the resident mesh"
        );
        assert_eq!(world.chunks[&c].mesh_hash, Some(hash));

        mesh::reset_content_hash_calls();
        world.upload_chunk_without_gpu(c, None);
        assert_eq!(mesh::content_hash_calls(), 0, "async drain passes None, no hash");
        assert_eq!(world.chunks[&c].mesh_hash, None);
    }

    #[test]
    fn skipped_remesh_pushes_visibility_when_the_chunk_was_hidden() {
        let mut world = World::generate();
        let c = *world.chunks.keys().next().expect("spawn chunks");
        let h = voxel_engine::MeshHandle::from_raw_parts(92, 1);
        let meshes = super::super::ChunkMeshes::from_upload_handles(ByPass::from_fn(|p| {
            (p == voxel_engine::Pass::Opaque).then_some(h)
        }))
        .expect("one pass present");
        {
            let loaded = world.chunks.get_mut(&c).unwrap();
            loaded.state = MeshState::Dirty {
                prev: Some(meshes),
            };
            loaded.visible = false;
        }
        world.occlusion_active = false;
        super::super::vis_log::take();
        world.keep_resident_mesh(c, None);
        assert_eq!(
            super::super::vis_log::take(),
            vec![(h, true)],
            "a hidden resident mesh must be shown when vis becomes true"
        );
        assert!(world.chunks[&c].visible);
        assert!(matches!(world.chunks[&c].state, MeshState::Ready(_)));

        super::super::vis_log::take();
        world.keep_resident_mesh(c, None);
        assert!(
            super::super::vis_log::take().is_empty(),
            "unchanged vis must not push set_visible"
        );
    }

    /// Same-sky holes stay one run. A foreign sky splits. Open chunks that share
    /// `(cx, cz)` stay separate jobs. A slab span includes the loaded layers.
    #[test]
    fn gather_column_runs_splits_foreign_sky_and_keeps_open_separate() {
        let sky = |c: Coord| -> Sky {
            if c.x == 0 && c.z == 0 && (c.y == 1 || c.y == 3) {
                Sky::Open
            } else if c.x >= 4 {
                Sky::Axis(Face::PosX)
            } else {
                Sky::Axis(Face::PosY)
            }
        };
        let missing = [
            Coord::new(1, 0, 0),
            Coord::new(1, 2, 0),
            Coord::new(0, 0, 0),
            Coord::new(0, 2, 0),
            Coord::new(0, 1, 0),
            Coord::new(0, 3, 0),
            Coord::new(4, 0, 0),
            Coord::new(6, 0, 0),
        ];
        let runs = gather_column_runs(missing, sky, |_| false, |_| false, false, false);
        let hole = ColumnKey { face: Face::PosY, a: 1, b: 0 };
        assert!(
            runs.contains(&GenRun::Column { key: hole, lo: 0, hi: 2 }),
            "a loaded PosY hole stays one run: {runs:?}"
        );
        let split = ColumnKey { face: Face::PosY, a: 0, b: 0 };
        assert!(runs.contains(&GenRun::Column { key: split, lo: 0, hi: 0 }), "{runs:?}");
        assert!(runs.contains(&GenRun::Column { key: split, lo: 2, hi: 2 }), "{runs:?}");
        assert!(
            runs.contains(&GenRun::Open { coord: Coord::new(0, 1, 0) }),
            "the Open layer is its own run: {runs:?}"
        );
        assert!(
            !runs.contains(&GenRun::Column { key: split, lo: 0, hi: 2 }),
            "Open in the gap splits the PosY run"
        );
        assert!(
            runs.contains(&GenRun::Open { coord: Coord::new(0, 3, 0) }),
            "Open chunks that share xz are not merged"
        );
        let posx = ColumnKey { face: Face::PosX, a: -1, b: 0 };
        assert!(
            runs.contains(&GenRun::Column { key: posx, lo: 4, hi: 6 }),
            "a same-sky gap along +X merges: {runs:?}"
        );

        let slab = [Coord::new(2, 0, 3), Coord::new(2, 1, 3), Coord::new(2, 2, 3)];
        let slab_runs = gather_column_runs(slab, sky, |c| c.y == 1, |_| false, true, false);
        assert_eq!(
            slab_runs,
            vec![GenRun::Column { key: ColumnKey { face: Face::PosY, a: 2, b: 3 }, lo: 0, hi: 2 }],
            "a slab span includes the loaded middle"
        );

        let gapped = [Coord::new(2, 0, 4), Coord::new(2, 2, 4)];
        let gapped_runs = gather_column_runs(gapped, sky, |_| false, |_| false, true, false);
        let gk = ColumnKey { face: Face::PosY, a: 2, b: 4 };
        assert_eq!(
            gapped_runs,
            vec![
                GenRun::Column { key: gk, lo: 0, hi: 0 },
                GenRun::Column { key: gk, lo: 2, hi: 2 },
            ]
        );
    }

    #[test]
    fn open_fail_key_does_not_collide_with_a_pos_y_column() {
        let coord = Coord::new(3, 1, 4);
        let open = FailKey::Open { coord };
        let column = FailKey::Column {
            key: ColumnKey { face: Face::PosY, a: coord.x, b: coord.z },
        };
        assert_ne!(open, column);
        let mut set = FastSet::default();
        set.insert(column);
        assert!(!set.contains(&open));
        let runs = gather_column_runs(
            [coord],
            |_| Sky::Open,
            |_| false,
            |k| k == column,
            false,
            true,
        );
        assert_eq!(runs, vec![GenRun::Open { coord }]);
    }

    /// Valley or hilltop columns inside the near square, below or above the full-res window,
    /// that neither a full-res chunk nor a far section draws. Seed 42, diffusion, the bench
    /// camera on the start world's +Z chart: ground level, that eye, and 300 above it.
    #[test]
    fn far_chart_strip_hole_is_closed() {
        use crate::render_config::RenderConfig;
        use crate::world::generation::WorldgenKind;

        let mut world = World::with_kind(42, RenderConfig::default(), WorldgenKind::Diffusion, false);
        let eye = world.chart_eye(DVec3::new(0.0, -8_640_801.0, 22_107_307.0)).expect("symptom chart eye");
        let ground = world.terrain().surface(Face::PosY, eye.x as i32, eye.z as i32);
        assert_ne!(ground, i32::MIN, "symptom column has no surface");
        // `expect_outside`: the window must miss some column, or a punch of the whole square
        // would still report zero holes.
        let sites = [
            ("ground", DVec3::new(eye.x, ground as f64, eye.z), false),
            ("eye", eye, true),
            ("+300", DVec3::new(eye.x, ground as f64 + 300.0, eye.z), true),
        ];
        for (name, storage, expect_outside) in sites {
            let center = Coord::new(
                (storage.x / 16.0).floor() as i32,
                (storage.y / 16.0).floor() as i32,
                (storage.z / 16.0).floor() as i32,
            );
            world.section_eye_y = storage.y;
            world.adopt_fold(center);
            let seat = world.seams.chart_seat(center).unwrap_or_else(|| panic!("{name}: no chart seat"));
            let near = world.near_block_box(center);
            assert!(
                near.0 >= seat.lo[0] && near.1 <= seat.hi[0] && near.2 >= seat.lo[2] && near.3 <= seat.hi[2],
                "{name}: the near square meets a seam"
            );
            let (y0, y1) = world.near_y_range(center);
            let desired = world.desired_sections(center);
            let mut rects = Vec::new();
            for s in &desired {
                let span = s.span() as i64;
                let (x, z) = (s.min_x() as i64, s.min_z() as i64);
                if inside_xz(*s, seat.lo, seat.hi) {
                    rects.push((x, z, x + span, z + span));
                }
            }
            let covered = |x: i64, z: i64| {
                rects.iter().any(|&(x0, z0, x1, z1)| x >= x0 && x < x1 && z >= z0 && z < z1)
            };
            let mut holes = 0i32;
            let mut outside = 0i32;
            let mut x = near.0 + 8;
            while x < near.1 {
                let mut z = near.2 + 8;
                while z < near.3 {
                    let surf = world.terrain().surface(Face::PosY, x as i32, z as i32);
                    if surf != i32::MIN {
                        let solid = i64::from(surf) - 1;
                        if solid < y0 || solid >= y1 {
                            outside += 1;
                            if !covered(x, z) {
                                holes += 1;
                            }
                        }
                    }
                    z += 16;
                }
                x += 16;
            }
            assert_eq!(
                holes, 0,
                "{name}: {holes} strip holes of {outside} columns outside the window, desired {}",
                desired.len()
            );
            if expect_outside {
                assert!(outside > 0, "{name}: no column sits outside the full-res window");
            }
        }
    }

    /// Columns of the far-field disk that draw neither a chart section nor a full-res chunk.
    /// Across a seam the neighbour chart is in the disk, and the near window's chunks there are
    /// real storage chunks (or a section covers them). A section wholly inside the near square is
    /// dropped only when that square's surface sits inside the full-res window. The band just
    /// outside the box, out to one coarse-ring span, is part of the same count: a straddler is
    /// replaced by its descendants, so that band is drawn.
    #[test]
    fn far_chart_seam_has_no_hole() {
        use crate::render_config::RenderConfig;
        use crate::space::atlas::Patch;
        use crate::space::chart::{self, Map};
        use crate::world::generation::WorldgenKind;
        use crate::ident::Detail;
        use crate::world::section::{section_span, FINEST_DETAIL};

        let mut world = World::with_kind(42, RenderConfig::default(), WorldgenKind::Diffusion, false);
        let centre = world.generator.cosmos().expect("cosmos").home().centre_f();
        let atlas = world
            .generator
            .atlases()
            .iter()
            .find(|a| (a.centre - centre).length() < 1.0)
            .expect("charted")
            .clone();

        let storage_from_dir = |world: &World, dir: DVec3, above: f64| -> DVec3 {
            let dir = dir.normalize();
            let face = Face::from_dominant(dir);
            let (tu, nn, tv) = chart::basis(face);
            let (xi, eta) = Map::Equiangular.inverse(DVec3::new(dir.dot(tu), dir.dot(nn), dir.dot(tv)));
            let n = atlas.bands[0].n;
            let step = 2.0 / n as f64;
            // An exact seam parameter floors to n, one cell past the box. Stand on the last cell.
            let i = (((xi + 1.0) / step).floor() as i64).clamp(0, n - 1);
            let j = (((eta + 1.0) / step).floor() as i64).clamp(0, n - 1);
            let patch = Patch::Shell { band: 0, face };
            let (origin, _) = atlas.storage_box(patch);
            let stored = atlas.storage(patch, [i, 0, j]);
            let ground = world.terrain().surface(Face::PosY, stored[0] as i32, stored[2] as i32);
            let local_y = ground as f64 - origin[1] as f64;
            let surf = atlas.embed(patch, DVec3::new(i as f64 + 0.5, local_y, j as f64 + 0.5));
            let up = (surf - atlas.centre).normalize();
            let fallback = DVec3::new(stored[0] as f64 + 0.5, ground as f64 + above, stored[2] as f64 + 0.5);
            world.chart_eye(surf + up * above).unwrap_or(fallback)
        };

        let mut sites: Vec<(String, DVec3)> = Vec::new();
        let symptom = world.chart_eye(DVec3::new(0.0, -8_640_801.0, 22_107_307.0)).expect("symptom chart eye");
        sites.push(("symptom".to_string(), symptom));
        let sym_ground = world.terrain().surface(Face::PosY, symptom.x as i32, symptom.z as i32);
        sites.push(("symptom-ground".to_string(), DVec3::new(symptom.x, sym_ground as f64, symptom.z)));
        sites.push(("symptom+300".to_string(), DVec3::new(symptom.x, sym_ground as f64 + 300.0, symptom.z)));
        for above in [0.0_f64, 300.0] {
            let tag = if above == 0.0 { "ground" } else { "+300" };
            sites.push((format!("seam-yz-{tag}"), storage_from_dir(&world, DVec3::new(0.0, 1.0, 1.0), above)));
            sites.push((format!("seam-yx-{tag}"), storage_from_dir(&world, DVec3::new(1.0, 1.0, 0.0), above)));
            sites.push((format!("corner-{tag}"), storage_from_dir(&world, DVec3::new(1.0, 0.985, 0.97), above)));
            sites.push((format!("face-{tag}"), storage_from_dir(&world, DVec3::new(0.0, 1.0, 0.0), above)));
        }
        let seam = storage_from_dir(&world, DVec3::new(0.0, 1.0, 1.0), 0.0);
        let seam_c = Coord::new((seam.x / 16.0).floor() as i32, 0, (seam.z / 16.0).floor() as i32);
        let seam_seat = world.seams.chart_seat(seam_c).expect("seam seat");
        let inset = if (seam.z as i64 - seam_seat.lo[2]).abs() < (seam_seat.hi[2] - seam.z as i64).abs() {
            100 * 16
        } else {
            -100 * 16
        };
        let (ix, iz) = (seam.x, seam.z + inset as f64);
        let ig = world.terrain().surface(Face::PosY, ix as i32, iz as i32);
        sites.push(("inset100-ground".to_string(), DVec3::new(ix, ig as f64, iz)));
        sites.push(("inset100+300".to_string(), DVec3::new(ix, ig as f64 + 300.0, iz)));

        let cs = 16i64;
        // The ring outside the finest one. A punched section of that span leaves a wider overhang
        // than a finest tile when the finest annulus does not reach past the full-res box.
        let sliver = section_span(Detail(FINEST_DETAIL.0 + 1)) as i64;
        let outer = world.section_pyramid.outer_m() as i64;
        for (name, storage) in sites {
            let center = Coord::new(
                (storage.x / 16.0).floor() as i32,
                (storage.y / 16.0).floor() as i32,
                (storage.z / 16.0).floor() as i32,
            );
            world.section_eye_y = storage.y;
            world.adopt_fold(center);
            let seat = world.seams.chart_seat(center).unwrap_or_else(|| panic!("{name}: no chart seat"));
            let desired = world.desired_sections(center);
            let (ex, _, ez) = storage_eye_block(center, storage.y, DVec3::ZERO);
            let across = world.seams.seam_across(seat, [ex, storage.y.round() as i64, ez], outer);
            let near = world.near_block_box(center);
            let v = world.view.vertical as i64;
            let mut rects: Vec<(i64, i64, i64, i64)> = Vec::new();
            for s in &desired {
                let span = s.span() as i64;
                let (x, z) = (s.min_x() as i64, s.min_z() as i64);
                let (x0, z0, x1, z1) = if inside_xz(*s, seat.lo, seat.hi) {
                    (x, z, x + span, z + span)
                } else if let Some(m) = across.iter().find(|m| inside_xz(*s, m.seat.lo, m.seat.hi)) {
                    let (a, c) = m.home_xz(x, z);
                    let (b, d) = m.home_xz(x + span, z + span);
                    (a.min(b), c.min(d), a.max(b), c.max(d))
                } else {
                    continue;
                };
                rects.push((x0, x1, z0, z1));
            }
            let covered = |x: i64, z: i64| rects.iter().any(|&(x0, x1, z0, z1)| x >= x0 && x < x1 && z >= z0 && z < z1);
            let mut holes = 0i32;
            let mut hole_ex = String::new();
            let mut across_cols = 0i32;
            let mut across_bad = 0i32;
            let mut x = ex - outer;
            let step = 64i64;
            while x <= ex + outer {
                let mut z = ez - outer;
                while z <= ez + outer {
                    let (dx, dz) = (x - ex, z - ez);
                    if dx * dx + dz * dz > outer * outer {
                        z += step;
                        continue;
                    }
                    let virt = Coord::new(x.div_euclid(cs) as i32, center.y, z.div_euclid(cs) as i32);
                    let real = world.fold.unfold(virt);
                    let in_near = x >= near.0 && x < near.1 && z >= near.2 && z < near.3;
                    let section_hit = covered(x, z);
                    let mut chunk_hit = false;
                    if in_near {
                        if let Some(rc) = real {
                            let surf = world.terrain().surface(Face::PosY, rc.x * 16 + 8, rc.z * 16 + 8);
                            if rc != virt {
                                across_cols += 1;
                                if surf == i32::MIN || world.seams.chart_seat(rc).is_none() {
                                    across_bad += 1;
                                }
                            }
                            let gy = if surf == i32::MIN { i64::MIN } else { (surf as i64 - 1).div_euclid(cs) };
                            chunk_hit = gy >= center.y as i64 - v && gy <= center.y as i64 + v;
                        }
                    }
                    if !section_hit && !chunk_hit {
                        let past_home = x < seat.lo[0] || x >= seat.hi[0] || z < seat.lo[2] || z >= seat.hi[2];
                        let ox = if x < near.0 { near.0 - x } else if x >= near.1 { x - (near.1 - 1) } else { 0 };
                        let oz = if z < near.2 { near.2 - z } else if z >= near.3 { z - (near.3 - 1) } else { 0 };
                        holes += 1;
                        if holes <= 6 {
                            hole_ex.push_str(&format!(
                                " at ({x},{z}) past_home {past_home} in_near {in_near} ox {ox} oz {oz} real {real:?};"
                            ));
                        }
                    }
                    z += step;
                }
                x += step;
            }
            // Chunk centres in the overhang band. Step 64 misses a sliver narrower than the stride.
            let mut fine = 0i32;
            let mut fine_n = 0i32;
            let mut fx = near.0 - sliver + 8;
            while fx < near.1 + sliver {
                let mut fz = near.2 - sliver + 8;
                while fz < near.3 + sliver {
                    let ox = if fx < near.0 { near.0 - fx } else if fx >= near.1 { fx - (near.1 - 1) } else { 0 };
                    let oz = if fz < near.2 { near.2 - fz } else if fz >= near.3 { fz - (near.3 - 1) } else { 0 };
                    let outside = ox.max(oz) > 0 && ox.max(oz) < sliver;
                    if outside {
                        fine_n += 1;
                        if !covered(fx, fz) {
                            fine += 1;
                        }
                    }
                    fz += 16;
                }
                fx += 16;
            }
            let edge = [seat.hi[0] - ex, ex - (seat.lo[0] - 1), seat.hi[2] - ez, ez - (seat.lo[2] - 1)];
            let reaches_seam = edge.iter().any(|&d| d < (near.1 - near.0) / 2);
            assert!(fine_n > 0, "{name}: the overhang band was not sampled");
            assert_eq!(
                fine, 0,
                "{name}: {fine} overhang columns of {fine_n} within one coarse span, desired {} holes {holes}{hole_ex}",
                desired.len()
            );
            assert_eq!(
                holes, 0,
                "{name}: {holes} uncovered columns, desired {} near {near:?} eye ({ex},{ez}) center {center:?} edges {edge:?};{hole_ex}",
                desired.len()
            );
            assert!(
                desired.len() <= world.sections_allowed(),
                "{name}: {} sections over the slot budget {}",
                desired.len(),
                world.sections_allowed()
            );
            if reaches_seam {
                assert!(across_cols > 0, "{name}: the near window does not cross the seam");
                assert_eq!(across_bad, 0, "{name}: {across_bad} near columns across the seam are not real chunks");
            }
        }
    }

    /// A full section floor must not disarm the lane, and must drop Ready sections
    /// nothing desired draws so the open cell can be admitted.
    /// One edit dirties an overlay position per active detail, each about a far section's extract:
    /// a pass past its budget stops after one position, the rest follow on later passes.
    #[test]
    fn the_edit_overlay_refresh_spreads_over_passes() {
        use crate::world::section::SectionPos;
        let mut world = World::new(7);
        for x in 0..3 {
            let pos = SectionPos { body: 0, face: Face::PosY, detail: crate::world::section::FINEST_DETAIL, x, z: 0 };
            world.section_overlay_dirty.insert(pos);
        }
        let pass = |world: &mut World| match world.refresh_section_overlay(Budget::Millis(0.0)) {
            Progress::Partial { remaining } => Some(remaining),
            Progress::Idle => None,
            _ => panic!("unexpected progress"),
        };
        assert_eq!(pass(&mut world), Some(2));
        assert_eq!(pass(&mut world), Some(1));
        assert_eq!(pass(&mut world), None);
        assert!(world.section_overlay_dirty.is_empty());
        assert_eq!(pass(&mut world), None);
    }

    #[test]
    fn full_section_floor_keeps_the_lane_armed_and_frees_a_slot() {
        let mut world = World::generate();
        let center = Coord::new(0, 4, 0);
        world.center = Some(center);
        world.slot_ceiling = 1024;
        world.gpu_live_slots = 6000;
        let hole = SectionPos {
            body: 0,
            face: Face::PosY,
            detail: super::super::section::FINEST_DETAIL,
            x: 0,
            z: 0,
        };
        let filler = |i: usize| SectionPos {
            body: 0,
            face: Face::PosY,
            detail: super::super::section::FINEST_DETAIL,
            x: 10_000 + i as i32,
            z: -3,
        };
        world.section_desired = vec![hole];
        let empty = || SectionState::Ready { meshes: Vec::new(), cages: Vec::new(), last_style: None };
        for i in 0..super::super::SECTION_SLOT_FLOOR {
            world.sections.insert(filler(i), empty());
        }
        assert!(!world.section_covered(hole), "the hole has no resident cover");
        assert!(!<SectionLane as StreamLane>::ready(&world, hole), "the floor is full");
        world.pending_sections.set();
        super::super::admit::<SectionLane>(&mut world, center, Budget::Millis(8.0));
        assert!(
            world.pending_sections.get(),
            "a refused budget is not a drained backlog"
        );
        assert!(!world.sections.contains_key(&hole), "nothing was admitted");
        world.reclaim_blocked_sections(center, None);
        assert!(
            world.section_budget_used() < world.sections_allowed(),
            "one unwanted Ready section makes room, used {} allowed {}",
            world.section_budget_used(),
            world.sections_allowed()
        );
        assert!(world.pending_sections.get(), "freeing a slot re-arms admission");
        assert!(<SectionLane as StreamLane>::ready(&world, hole));
    }

    /// Only a chunk the mesh lane could admit takes a seed: an unloaded coord, a drawn chunk and
    /// an in-flight build stay off the worklist, whichever path seeds (an expired light wait, a
    /// failed build), and a stale build re-seeds itself unless an edit made it the dirty lane's.
    #[test]
    fn mesh_seeds_only_admissible_chunks() {
        let mut world = World::generate();
        let c = Coord::new(0, 0, 0);
        let gone = Coord::new(0, 40, 0);
        assert!(!world.chunks.contains_key(&gone), "the probe coord is not loaded");
        world.mesh_worklist.clear();
        world.seed_mesh(gone);
        assert!(!world.mesh_worklist.contains(&gone), "an unloaded coord seeds itself on load");
        world.fail_job(pipeline::JobKey::Mesh { coord: gone });
        assert!(!world.mesh_worklist.contains(&gone), "a failed build of one seeds nothing");
        world.chunks.get_mut(&c).unwrap().state = MeshState::Ready(ready_handle(5));
        world.seed_mesh(c);
        assert!(!world.mesh_worklist.contains(&c), "a drawn chunk is never admitted");
        let claim = |world: &mut World, prev| {
            world.chunks.get_mut(&c).unwrap().state = MeshState::NeedsMesh { building: true, prev };
            world.building_meshes = 1;
        };
        claim(&mut world, None);
        world.seed_mesh(c);
        assert!(!world.mesh_worklist.contains(&c), "an in-flight build is never admitted");
        world.center = Some(c);
        world.chunks.get_mut(&c).unwrap().light = None;
        assert!(world.chunk_light_blocked(c), "the build waits on light");
        world
            .light_gate
            .blocked_since
            .insert(c, Instant::now() - LIGHT_WAIT_DEGRADE - Duration::from_millis(1));
        world.tick_light_gate();
        assert!(world.light_gate.blocked_since.contains_key(&c), "the wait is still timed");
        assert!(!world.mesh_worklist.contains(&c), "an expired wait seeds no in-flight build");
        world.invalidate_mesh(c);
        world.drop_stale_upload(c);
        world.fail_job(pipeline::JobKey::Mesh { coord: c });
        assert!(!world.mesh_worklist.contains(&c), "an edited build's stale or failed result seeds nothing");
        claim(&mut world, Some(ready_handle(6)));
        world.drop_stale_upload(c);
        assert_eq!(world.building_meshes, 0, "the stale build released its claim");
        assert!(world.mesh_worklist.contains(&c), "a stale build re-seeds");
        world.mesh_worklist.clear();
        world.chunks.get_mut(&c).unwrap().state = MeshState::needs_mesh();
        world.seed_mesh(c);
        assert!(
            world.mesh_worklist.contains(&c) || matches!(world.chunks[&c].state, MeshState::Air),
            "a fresh chunk is seeded unless it is walled in"
        );
    }

    /// On a chart the frontier is a function of whole blocks and whole-chunk prediction: an eye
    /// that moves inside one block, or a velocity that jitters inside one chunk of lookahead,
    /// keeps the cached selection, and that selection is what a fresh sweep returns. The surface
    /// memo keeps only the rects the last sweep read.
    #[test]
    fn chart_frontier_holds_within_a_block() {
        use crate::render_config::RenderConfig;
        use crate::world::generation::WorldgenKind;

        let mut world = World::with_kind(42, RenderConfig::default(), WorldgenKind::Diffusion, false);
        let spawn = world.chart_spawn().expect("the start world is charted");
        let eye = world.chart_eye(spawn).expect("spawn stands on a chart");
        let center = Coord::new(
            (eye.x / 16.0).floor() as i32,
            (eye.y / 16.0).floor() as i32,
            (eye.z / 16.0).floor() as i32,
        );
        world.adopt_fold(center);
        world.center = Some(center);
        world.update_lod_face(center);
        assert!(world.section_on_chart(center), "spawn streams from a chart's storage");
        let y = eye.y.floor() + 0.25;
        let refresh = |world: &mut World, eye_y: f64, vel: DVec3| {
            world.section_eye_y = eye_y;
            world.section_vel = vel;
            let key = world.section_frontier_key;
            world.refresh_frontier(center);
            let fresh = world.desired_sections(center);
            assert_eq!(world.section_desired, fresh, "the cached frontier is the fresh one");
            let memo = &world.near_bounds;
            assert!(memo.rects.values().all(|e| e.1 == memo.pass), "stale rects kept");
            world.section_frontier_key != key
        };
        assert!(refresh(&mut world, y, DVec3::ZERO), "first pass selects");
        assert!(!world.section_desired.is_empty());
        assert!(!refresh(&mut world, y + 0.2, DVec3::ZERO), "same block");
        assert!(refresh(&mut world, y + 1.0, DVec3::ZERO), "next block");
        assert!(refresh(&mut world, y + 1.0, DVec3::new(100.3, 0.0, 0.0)), "prediction starts");
        assert!(!refresh(&mut world, y + 1.0, DVec3::new(100.6, 0.0, -0.4)), "jitter in one chunk");
        assert!(refresh(&mut world, y + 1.0, DVec3::new(120.0, 0.0, 0.0)), "next chunk of lookahead");
    }

    fn empty_ready() -> SectionState {
        SectionState::Ready { meshes: Vec::new(), cages: Vec::new(), last_style: None }
    }

    /// Desired sections with no Ready self or ancestor.
    fn uncovered(world: &World) -> usize {
        world.section_desired.iter().filter(|&&c| !world.section_covered(c)).count()
    }

    /// What the section passes landed: cancelled section jobs, those of them the frontier still
    /// wanted, and failed jobs.
    #[derive(Default)]
    struct Landed {
        cancels: usize,
        wanted: usize,
        fails: usize,
    }

    /// Land worker results and turn queued section uploads into empty Ready meshes. Returns whether
    /// anything landed.
    fn pump_sections(world: &mut World, landed: &mut Landed) -> bool {
        let mut got = false;
        while let Some(done) = world.workers.as_ref().and_then(pipeline::Workers::try_recv) {
            got = true;
            match &done {
                pipeline::Done::Cancelled(keys) => {
                    landed.cancels += keys.len();
                    landed.wanted += keys
                        .iter()
                        .filter(|k| matches!(k, pipeline::JobKey::Section { pos, .. } if world.section_desired.contains(pos)))
                        .count();
                }
                pipeline::Done::Failed(_) => landed.fails += 1,
                _ => {}
            }
            world.integrate_worker_result(done);
        }
        while let Some((pos, token, _, _)) = world.section_upload_queue.pop_front() {
            if let Some(state @ SectionState::Meshing { .. }) = world.sections.get_mut(&pos)
                && matches!(state, SectionState::Meshing { token: t } if *t == token)
            {
                world.meshing_sections = world.meshing_sections.saturating_sub(1);
                *state = empty_ready();
            }
        }
        got
    }

    /// One pass of the section lanes around the far-field centre, in `stream`'s order: land,
    /// reclaim, admit, then the visible rebuild that re-arms holes. Pending is not forced on from
    /// outside.
    fn section_pass(world: &mut World, landed: &mut Landed) {
        let center = world.section_center().expect("a far-field centre");
        let got = pump_sections(world, landed);
        world.reclaim_blocked_sections(center, None);
        super::super::admit::<SectionLane>(world, center, Budget::Millis(8.0));
        if world.section_cover_dirty.take() || world.pending_sections.get() {
            world.rebuild_section_visible(None);
        }
        if !got {
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Run section passes until every desired section is covered and nothing is in flight.
    /// Returns the passes taken.
    fn drive_sections(world: &mut World, name: &str, deadline: Instant, landed: &mut Landed) -> usize {
        let mut passes = 0usize;
        loop {
            let unc = uncovered(world);
            let queued = world.workers.as_ref().map(pipeline::Workers::queue_depths).unwrap_or((0, 0)).1;
            if unc == 0 && world.meshing_sections == 0 && world.section_upload_queue.is_empty() && queued == 0 {
                assert_eq!(landed.fails, 0, "{name}: section jobs failed");
                return passes;
            }
            passes += 1;
            assert!(
                Instant::now() < deadline && passes < 20_000,
                "{name}: uncovered {unc} of {} after {passes} passes, cancels {} fails {} meshing {} queued {queued}",
                world.section_desired.len(),
                landed.cancels,
                landed.fails,
                world.meshing_sections
            );
            section_pass(world, landed);
        }
    }

    /// The chart cap, once the near field has filled the CPU-cull knob, keeps the
    /// sections nearest in the chart frame. Sections loaded earlier are the
    /// storage-nearest of the wider frontier, which is not that set: neighbours
    /// and the far rim disagree. Readiness still reaches an empty uncovered count.
    #[test]
    fn far_chart_seam_readiness_converges() {
        use crate::render_config::RenderConfig;
        use crate::space::atlas::Patch;
        use crate::space::chart::{self, Map};
        use crate::world::generation::WorldgenKind;

        let mut world = World::with_kind(42, RenderConfig::default(), WorldgenKind::Diffusion, false);
        let centre = world.generator.cosmos().expect("cosmos").home().centre_f();
        let atlas = world
            .generator
            .atlases()
            .iter()
            .find(|a| (a.centre - centre).length() < 1.0)
            .expect("charted")
            .clone();
        let storage_from_dir = |world: &World, dir: DVec3, above: f64| -> DVec3 {
            let dir = dir.normalize();
            let face = Face::from_dominant(dir);
            let (tu, nn, tv) = chart::basis(face);
            let (xi, eta) = Map::Equiangular.inverse(DVec3::new(dir.dot(tu), dir.dot(nn), dir.dot(tv)));
            let n = atlas.bands[0].n;
            let step = 2.0 / n as f64;
            let i = (((xi + 1.0) / step).floor() as i64).clamp(0, n - 1);
            let j = (((eta + 1.0) / step).floor() as i64).clamp(0, n - 1);
            let patch = Patch::Shell { band: 0, face };
            let (origin, _) = atlas.storage_box(patch);
            let stored = atlas.storage(patch, [i, 0, j]);
            let ground = world.terrain().surface(Face::PosY, stored[0] as i32, stored[2] as i32);
            let local_y = ground as f64 - origin[1] as f64;
            let surf = atlas.embed(patch, DVec3::new(i as f64 + 0.5, local_y, j as f64 + 0.5));
            let up = (surf - atlas.centre).normalize();
            let fallback = DVec3::new(stored[0] as f64 + 0.5, ground as f64 + above, stored[2] as f64 + 0.5);
            world.chart_eye(surf + up * above).unwrap_or(fallback)
        };

        let mut sites: Vec<(String, DVec3)> = Vec::new();
        let symptom = world.chart_eye(DVec3::new(0.0, -8_640_801.0, 22_107_307.0)).expect("symptom chart eye");
        sites.push(("symptom".into(), symptom));
        let sym_ground = world.terrain().surface(Face::PosY, symptom.x as i32, symptom.z as i32);
        sites.push(("symptom-ground".into(), DVec3::new(symptom.x, sym_ground as f64, symptom.z)));
        sites.push(("symptom+300".into(), DVec3::new(symptom.x, sym_ground as f64 + 300.0, symptom.z)));
        sites.push(("seam-yz".into(), storage_from_dir(&world, DVec3::new(0.0, 1.0, 1.0), 0.0)));
        sites.push(("seam-yz+300".into(), storage_from_dir(&world, DVec3::new(0.0, 1.0, 1.0), 300.0)));
        sites.push(("seam-yx".into(), storage_from_dir(&world, DVec3::new(1.0, 1.0, 0.0), 0.0)));
        sites.push(("corner".into(), storage_from_dir(&world, DVec3::new(1.0, 0.985, 0.97), 0.0)));
        sites.push(("face".into(), storage_from_dir(&world, DVec3::new(0.0, 1.0, 0.0), 0.0)));

        let deadline = Instant::now() + Duration::from_secs(90);
        for (name, storage) in sites {
            let center = Coord::new(
                (storage.x / 16.0).floor() as i32,
                (storage.y / 16.0).floor() as i32,
                (storage.z / 16.0).floor() as i32,
            );
            world.sections.clear();
            world.section_eye_y = storage.y;
            world.center = Some(center);
            world.stream_up = Some(Face::PosY);
            world.stream_up_set = true;
            world.adopt_fold(center);
            world.slot_ceiling = 1024;
            world.gpu_live_slots = 0;
            let wide = world.desired_sections(center);
            world.gpu_live_slots = 512;
            let tight_allowed = world.sections_allowed();
            let tight = world.desired_sections(center);
            let far_m = f64::from(world.section_pyramid.outer_m());
            let radius = world.view.horizontal;
            let (fold, far_view) = (world.fold, world.far_view(center));
            {
                let workers = world.worker_pool();
                workers.set_view(center.x, center.y, center.z, far_view, radius, far_m, 0.0, 0.0, 0.0, Some(Face::PosY));
                workers.set_fold(fold);
            }
            // Cold start: the bench once the near field has already taken the
            // surplus above the section floor, and nothing coarser is resident.
            if name == "symptom" {
                world.section_desired = tight.clone();
                world.pending_sections.set();
                world.section_cover_dirty.set();
                drive_sections(&mut world, "symptom cold", deadline, &mut Landed::default());
                assert_eq!(uncovered(&world), 0, "symptom cold start left sections uncovered");
                world.sections.clear();
                world.meshing_sections = 0;
                world.section_upload_queue.clear();
                while world.workers.as_ref().and_then(pipeline::Workers::try_recv).is_some() {}
            }
            // The wider frontier's storage-nearest floor is what admission loads
            // first. After the cap drops, that set is not the chart-nearest one.
            let mut prefix = wide;
            prefix.sort_by_key(|s| <SectionLane as StreamLane>::order(&world, center, *s));
            prefix.truncate(tight_allowed.min(prefix.len()));
            for &s in &prefix {
                world.sections.insert(s, empty_ready());
            }
            world.section_desired = tight;
            world.pending_sections.set();
            world.section_cover_dirty.set();
            let planted = uncovered(&world);
            drive_sections(&mut world, &name, deadline, &mut Landed::default());
            assert_eq!(
                uncovered(&world),
                0,
                "{name}: planted {planted} uncovered sections of {} and the floor never cleared",
                world.section_desired.len()
            );
        }
    }

    /// The far field's thresholds hold while hovering. The chart rings' scale rises with the height
    /// over the ground and falls back only well below where it rose; the chart under the eye is kept
    /// past the far reach once the far field stands on it; and just below the near window's reach
    /// the far field is the frontier it stays just above it.
    #[test]
    fn far_eye_thresholds_hold() {
        use crate::render_config::RenderConfig;
        use crate::world::generation::WorldgenKind;

        let mut world = World::with_kind(42, RenderConfig::default(), WorldgenKind::Diffusion, false);
        let up = DVec3::new(0.0, 1.0, 0.0);
        let climb = |world: &mut World, above: f64| {
            world.place_eyes(world.home_eye(up, above));
            (world.far_scale, world.far_atlas.is_some())
        };
        // Default ladder: the rings reach 12,288 blocks, so a doubling is taken past 4,096 · 2^s of
        // height and dropped below 4,096 · 2^s / 1.25.
        let path = [
            (0.0, 0, "ground"),
            (4_600.0, 1, "past the first step"),
            (3_700.0, 1, "held below the first step"),
            (2_800.0, 0, "dropped well below it"),
            (9_000.0, 2, "past the second step"),
            (7_200.0, 2, "held below the second step"),
            (6_000.0, 1, "dropped one step"),
            (20_000.0, 3, "at the candidate cap"),
            (200_000.0, 3, "held at the cap"),
        ];
        for (above, scale, what) in path {
            assert_eq!(climb(&mut world, above), (scale, true), "+{above}: {what}");
        }
        // Past the far reach (262,144 above the stored top, ~2,000 over the ground): kept once
        // stood on, left past the hold, and not taken again until back under the reach.
        assert!(climb(&mut world, 280_000.0).1, "held past the far reach");
        assert!(!climb(&mut world, 340_000.0).1, "left past the hold");
        assert!(!climb(&mut world, 280_000.0).1, "not re-taken above the reach");
        assert!(climb(&mut world, 250_000.0).1, "re-taken under the reach");

        // The highest eye the near window still streams on the chart, and the far field there with
        // the near window on the chart and in physical space.
        let mut above = 1_000.0;
        while world.chart_eye(world.home_eye(up, above + 16.0)).is_some() {
            above += 16.0;
        }
        let eye = world.home_eye(up, above);
        let (near, far) = world.place_eyes(eye);
        assert_eq!(near, far, "the near window stands on the chart at +{above}");
        let far_c = eye_chunk(far);
        world.adopt_fold(far_c);
        let on_chart = world.desired_sections(far_c);
        world.adopt_fold(eye_chunk(eye));
        assert!(world.fold.is_identity());
        assert!(!on_chart.is_empty(), "+{above}: no chart sections");
        assert_eq!(world.desired_sections(far_c), on_chart, "+{above}: the far field changes at the near window's reach");
    }

    /// Stand the eye `above` blocks over the start world in direction `dir`, the way `stream`
    /// places it: the near window back in physical space, the far field on the chart under it,
    /// and no sections resident.
    fn hover(world: &mut World, dir: DVec3, above: f64, name: &str) {
        let (near, far) = world.place_eyes(world.home_eye(dir, above));
        let near_c = eye_chunk(near);
        world.sections.clear();
        world.center = Some(near_c);
        world.set_far_center(eye_chunk(far));
        world.stream_up = world.resolve_stream_up(near_c);
        world.stream_up_set = true;
        world.adopt_fold(near_c);
        assert!(world.fold.is_identity(), "{name}: the near window is not in physical space");
        assert!(!world.far_fold.is_identity(), "{name}: the far field does not stand on a chart");
        publish_far(world);
        assert!(!world.section_desired.is_empty(), "{name}: no chart sections");
    }

    /// Select the far frontier around the far centre and publish the view to the worker gate, as
    /// `stream` does after the far centre moves.
    fn publish_far(world: &mut World) {
        let (near_c, far_c) = (world.center.expect("a centre"), world.section_center().expect("a far centre"));
        world.section_desired = world.desired_sections(far_c);
        let (far_m, radius, up) = (world.far_horizon(), world.view.horizontal, world.stream_up);
        let (fold, far_view) = (world.fold, world.far_view(far_c));
        let workers = world.worker_pool();
        workers.set_view(near_c.x, near_c.y, near_c.z, far_view, radius, far_m, 0.0, 0.0, 0.0, up);
        workers.set_fold(fold);
        world.pending_sections.set();
        world.section_cover_dirty.set();
    }

    /// From high above the start world (the near window back in physical space, the worker gate
    /// measuring far work from the far centre, in the far field's chart net) every desired section
    /// lands, also across a seam and at a cube corner: readiness reaches an empty uncovered count in
    /// bounded passes, and the gate never deschedules a section the frontier wants.
    #[test]
    fn far_chart_altitude_readiness_converges() {
        use crate::render_config::RenderConfig;
        use crate::world::generation::WorldgenKind;

        let mut world = World::with_kind(42, RenderConfig::default(), WorldgenKind::Diffusion, false);
        let deadline = Instant::now() + Duration::from_secs(240);
        for (site, dir, heights) in World::FAR_SITES {
            for &above in heights {
                let name = &format!("{site} +{above}");
                hover(&mut world, dir, above, name);
                let mut landed = Landed::default();
                let passes = drive_sections(&mut world, name, deadline, &mut landed);
                println!(
                    "{name}: {} sections ready in {passes} passes, {} cancelled",
                    world.section_desired.len(),
                    landed.cancels
                );
                assert_eq!(uncovered(&world), 0, "{name}: sections left uncovered");
                assert_eq!(landed.wanted, 0, "{name}: the gate descheduled wanted sections");
            }
        }
    }

    /// The far centre moving while the frontier fills, 50 km over every far site: one chunk every
    /// few passes, re-publishing the view each time as `stream` does, and stopping well before the
    /// frontier has landed. Every view change re-measures the queued far work, so a horizon short
    /// of the coarsest ring's corners, or a neighbour chart measured a face box away, would
    /// deschedule wanted sections on every crossing. None is, and once the centre stops every
    /// desired section lands.
    #[test]
    fn far_chart_altitude_readiness_converges_while_moving() {
        use crate::render_config::RenderConfig;
        use crate::world::generation::WorldgenKind;

        const MOVES: usize = 48;
        const PASSES_PER_MOVE: usize = 12;
        let mut world = World::with_kind(42, RenderConfig::default(), WorldgenKind::Diffusion, false);
        let deadline = Instant::now() + Duration::from_secs(240);
        let above = 50_000.0;
        for (site, dir, _) in World::FAR_SITES {
            let name = &format!("{site} +{above} moving");
            hover(&mut world, dir, above, name);
            let mut landed = Landed::default();
            for _ in 0..MOVES {
                let c = world.section_center().expect("a far centre");
                world.set_far_center(Coord::new(c.x + 1, c.y, c.z));
                publish_far(&mut world);
                for _ in 0..PASSES_PER_MOVE {
                    section_pass(&mut world, &mut landed);
                }
            }
            let moving = landed.cancels;
            let passes = drive_sections(&mut world, name, deadline, &mut landed);
            println!(
                "{name}: {} sections ready {passes} passes after stopping, {moving} cancelled while moving",
                world.section_desired.len()
            );
            assert_eq!(landed.wanted, 0, "{name}: the gate descheduled wanted sections");
            assert_eq!(uncovered(&world), 0, "{name}: sections left uncovered");
        }
    }
}
