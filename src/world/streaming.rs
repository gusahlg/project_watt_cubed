//! Streaming: the [`World::stream`] pass, worker result draining, job queueing,
//! budgeted uploads, chunk unloading, and view-radius bookkeeping.
//! These are `World` methods (struct lives in `mod.rs`).

use std::time::{Duration, Instant};

use voxel_engine::producer::{Budget, Progress};
use voxel_engine::{DVec3, Engine, FadeStyle, MaterialDesc};

use crate::block::appearance::{
    fill_descriptor_layer, placeholder_layer, procedural_material_desc, BlockAppearance,
    LAYER_BYTES, TEXTURE_SIZE,
};

use crate::coord::{ByPass, ChunkBox, ChunkCoord, Face};
use crate::derived::Revision;
use crate::math::block_coord;

use super::chunk::{CHUNK_SIZE, Chunk};
use super::generation::ColumnHeights;
use super::heightmip::{BakeExtent, HeightMip};
use super::metric::{DyCap, EyeMetric, HeightEnvelope};
use super::section::SectionPos;
use super::summary::{CellError, CellSummary, SseBudget};
use super::{
    Coord, DIRTY_BUDGET, FastMap, FastSet, LightLane, Loaded, MeshLane, MeshState,
    SECTION_UPLOAD_BUDGET, SectionFrontierKey, SectionLane, SectionState, StreamLane,
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

/// Vertex bytes a finished section mesh will stage (one packed mesh × passes).
#[cfg(test)]
pub(in crate::world) fn section_output_bytes(data: &super::SectionMeshData) -> usize {
    voxel_engine::Pass::ALL
        .iter()
        .map(|&p| data.data[p].vertex_bytes())
        .sum()
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
        self.speed_mps = velocity.x.hypot(velocity.z);
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
    Column { col: (i32, i32) },
    Mesh { coord: Coord },
    Light { coord: Coord },
    Section { pos: SectionPos },
}

impl FailKey {
    fn of(key: &pipeline::JobKey) -> FailKey {
        match key {
            pipeline::JobKey::Column { col, .. } => FailKey::Column { col: *col },
            pipeline::JobKey::Mesh { coord } => FailKey::Mesh { coord: *coord },
            pipeline::JobKey::Light { coord } => FailKey::Light { coord: *coord },
            pipeline::JobKey::Section { pos, .. } => FailKey::Section { pos: *pos },
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

fn column_order(center: Coord, vel: DVec3, cx: i32, cz: i32) -> u64 {
    let dx = cx - center.x;
    let dz = cz - center.z;
    let ring = dx.abs().max(dz.abs()) as u64;
    super::motion_biased_dist2(
        ring.saturating_mul(ring).saturating_mul(1024),
        vel,
        f64::from(dx) * CHUNK_SIZE as f64,
        f64::from(dz) * CHUNK_SIZE as f64,
    )
}

impl World {
    /// The mesh box: chunks meshed and drawn around `center`.
    fn mesh_box(&self, center: Coord) -> ChunkBox {
        self.view.mesh(center)
    }

    /// The data box: the mesh box plus one [`DATA_MARGIN`] shell of voxel data,
    /// so edge chunks can cull against neighbours that are loaded but unmeshed.
    fn data_box(&self, center: Coord) -> ChunkBox {
        self.view.data(center)
    }

    /// The unload box: the mesh box plus the unload hysteresis, past which
    /// chunks are freed.
    pub(in crate::world) fn unload_box(&self, center: Coord) -> ChunkBox {
        self.view.unload(center)
    }

    /// Whether `coord` is inside the current mesh box. The single mesh-view
    /// check: the enqueue gate (the [`MeshLane`] ready predicate) and the
    /// apply gate ([`mesh_result_applies`](Self::mesh_result_applies)) both call
    /// this, so a chunk is enqueued only if its result would be accepted.
    /// `false` before the first stream (no centre yet).
    pub(in crate::world) fn in_mesh_box(&self, coord: Coord) -> bool {
        self.center
            .is_some_and(|c| self.mesh_box(c).contains(coord))
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
        // Capture eye altitude; section metric measures dy from it.
        self.section_eye_y = center.y;
        // Eye velocity for prediction. Resets to zero on non-finite values, non-positive dt,
        // or teleport-sized gaps, so prediction never fires on garbage input.
        let now = crate::sched::now();
        let (section_vel, pacing_vel, sample_dt) = match self.section_eye_prev {
            Some((prev, t)) => {
                let dt = now.duration_since(t).as_secs_f64();
                let v = (center - prev) / dt;
                let sane =
                    center.is_finite() && dt > 0.0 && dt <= MAX_PREDICT_SAMPLE_GAP && v.is_finite();
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
        self.light_admitted_last = 0;
        self.section_eye_prev = center.is_finite().then_some((center, now));
        let s = CHUNK_SIZE as i32;
        let center_chunk = ChunkCoord::new(
            block_coord(center.x).div_euclid(s),
            block_coord(center.y).div_euclid(s),
            block_coord(center.z).div_euclid(s),
        );
        // Update centre before draining: old centre may be a sentinel, so draining
        // against it would discard all results and regenerate them immediately.
        let prev_center = self.center;
        let full_pass = Some(center_chunk) != self.center;
        self.center = Some(center_chunk);
        // Re-bucket worklists around the live centre before any lane (or pump
        // insert) runs. O(n) once per boundary cross; a no-op when the rings
        // and centre already match.
        if full_pass {
            let rings = self.view.worklist_rings();
            self.mesh_worklist.resize(rings);
            self.mesh_worklist.recenter(center_chunk);
            self.light_worklist.resize(rings);
            self.light_worklist.recenter(center_chunk);
        }
        // Publish the live view to the worker pool: queued jobs re-key toward
        // the player's CURRENT position on every view change, and entries left
        // behind by fast movement — far sections included — are descheduled
        // instead of run. The far horizon is the outer ladder radius plus the
        // velocity lookahead, so prediction-desired sections survive it.
        let vxz = self.section_vel.x.hypot(self.section_vel.z);
        let far_m = f64::from(self.section_pyramid.outer_m()) + vxz * TAU_STREAM;
        // Configure the pool before any lane can submit this frame. On the
        // first stream this avoids one permissive/full-capacity burst from a
        // lazily spawned pool before the pacer catches it on the next pass.
        let pacer = self.stream_pacer;
        let velocity = self.section_vel;
        let view_radius = self.view.horizontal;
        let stager = eng.as_ref().map(|e| e.mesh_stager());
        let workers = self.worker_pool();
        if let Some(stager) = stager {
            workers.set_stager(stager);
        }
        workers.set_view(
            center_chunk.x,
            center_chunk.z,
            view_radius,
            far_m,
            velocity.x,
            velocity.z,
        );
        let capacity = workers.worker_capacity();
        workers.set_pacing(
            pacer.active_workers(capacity),
            pacer.near_queue_cap(capacity),
        );
        // Crossing a chunk boundary moves the BFS root, so the visible set is stale.
        self.occlusion_dirty.raise(full_pass);
        // The ring geometry is centred on the eye: a boundary cross SHIFTS the
        // settled rings by the move's chess distance (only a vertical move or
        // the first pass restarts the scan) — see `shift_lod_clip`.
        if full_pass {
            self.shift_lod_clip(prev_center, center_chunk);
        }
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
            // Stale queued uploads (the trailing edge of fast movement) release
            // in ONE pass here instead of trickling through the drain budget.
            self.prune_upload_queue();
            // Sync-generate the centre only when it is missing and not already
            // claimed: a claimed job is imminent and the previous centre's
            // collision halo still exists.
            self.ensure_data(center_chunk);
            self.pending_gen.set();
            // Mesh box moved: re-seed loaded NeedsMesh chunks that JUST
            // entered it. Only the shell (new ∖ old) needs probing — a chunk
            // in old ∩ new was either already seeded, or was evicted as
            // blocked, and blocked evictions re-seed through their own events
            // (data arrival, light settle, degrade expiry). O(|shell|) probes
            // instead of the old all-chunks iteration per cross.
            let new_box = self.mesh_box(center_chunk);
            let prev_box = self.prev_mesh_box;
            let fresh: Vec<Coord> = new_box
                .coords()
                .filter(|&c| prev_box.is_none_or(|p| !p.contains(c)))
                .filter(|&c| self.is_needs_mesh(c))
                .collect();
            self.mesh_worklist.extend(fresh);
            self.pending_fresh.set();
            self.prev_mesh_box = Some(new_box);
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
            for (&coord, loaded) in self.chunks.iter_mut() {
                if keep.contains(coord) {
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
            // Until the bake lands, selection uses the worst-case ladder;
            // mip only coarsens, no upward pops during bake.
            let mip_lane = self.lanes().mip;
            sched.run_manual(mip_lane, self, None);
            // Section overlay lane: refresh the edit overlay BEFORE
            // selection/occlusion/material read it (the frontier's error
            // coarsening below already consults it).
            let overlay_lane = self.lanes().section_overlay;
            sched.run_manual(overlay_lane, self, None);
            // ONE selection sweep, retained across passes: unloading, the load
            // lane, and the covering rebuild below all read this cache. The
            // frontier is a pure function of the key's inputs (eye, velocity,
            // ladder, relief-mip readiness), so while they are bit-identical —
            // a still camera — the sweep (grid walk + relief coarsening) is
            // skipped entirely. Edits force a recompute: relief coarsening
            // consults the edit overlay, which the key cannot cheaply cover.
            let frontier_key = SectionFrontierKey {
                center_xz: [center_chunk.x, center_chunk.z],
                eye_y: self.section_eye_y.to_bits(),
                // Quantise to 0.25 m/s so a continuously changing flight
                // velocity does not recompute the frontier every pass.
                velocity: [
                    (self.section_vel.x * 4.0).round().to_bits(),
                    (self.section_vel.y * 4.0).round().to_bits(),
                    (self.section_vel.z * 4.0).round().to_bits(),
                ],
                unit: self.section_pyramid.unit.to_bits(),
                finest: self.section_pyramid.finest.0,
                levels: self.section_pyramid.levels.get(),
                step: self.section_pyramid.step(),
                mip_ready: self.section_mip.is_some(),
                allowed: self.sections_allowed() as u32,
            };
            if self.section_frontier_key != Some(frontier_key) || !self.dirty_sections.is_empty() {
                self.section_desired = self.desired_sections(center_chunk);
                self.section_frontier_key = Some(frontier_key);
                self.section_cover_dirty.set();
            }
            if full_pass {
                self.unload_sections(
                    center_chunk,
                    eng.as_deref_mut()
                        .expect("section unload on a boundary cross needs the engine"),
                );
                self.pending_sections.set();
            }
            // Section dirty-remesh lane: free GPU meshes of edited sections so
            // they re-extract from the updated generator overlay.
            let section_remesh_lane = self.lanes().section_remesh;
            sched.run_manual(section_remesh_lane, self, eng.as_deref_mut());
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
        self.last_stream_secs = sample_dt;
    }

    /// Land finished worker results (non-blocking). Generate results clear
    /// `generating`; stale results release their exact claims. Result
    /// integration used to drain the unbounded channel in one frame, making a
    /// productive worker burst a main-thread hitch. It now shares the adaptive
    /// effort signal and keeps a small forward-progress floor.
    pub(in crate::world) fn drain_results(&mut self, eng: &mut Engine, result_budget: Duration) {
        let pacer = self.stream_pacer;
        let result_deadline = pipeline::Deadline::from_budget(pacer.duration(result_budget));
        let result_floor = pacer.floor(RESULT_INTEGRATE_FLOOR);
        let mut integrated = 0usize;
        while integrated < result_floor || !result_deadline.expired() {
            let Some(result) = self.workers.as_ref().and_then(pipeline::Workers::try_recv) else {
                break;
            };
            self.integrate_worker_result(result);
            integrated += 1;
        }

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

        // Budgeted light application. Order-independent: each grid is absolute,
        // leftovers apply next frame with no seam.
        let light_apply =
            pipeline::Deadline::from_budget(pacer.duration(pipeline::LIGHT_APPLY_BUDGET));
        let mut light_applied = 0usize;
        while light_applied == 0 || !light_apply.expired() {
            let Some((coord, grid)) = self.light_apply_queue.pop_front() else {
                break;
            };
            self.settle_light(coord, grid);
            light_applied += 1;
        }

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
            if let Some(state @ SectionState::Meshing { .. }) = self.sections.get_mut(&pos)
                && matches!(state, SectionState::Meshing { token: t } if *t == token)
            {
                super::adjust_count(&mut self.meshing_sections, true, false);
                upload_bytes += bytes;
                self.section_upload_bytes += bytes;
                *state = SectionState::from_upload_payload(pos, meshes, eng);
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
            pipeline::Done::Column { col, chunks, heights } => {
                self.accept_column(col, chunks, heights)
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
    /// site, the pop-time re-validation, and the boundary-cross prune.
    fn drop_stale_upload(&mut self, coord: Coord) {
        self.remesh_stats.note_drop_stale();
        if let Some(loaded) = self.chunks.get_mut(&coord) {
            if loaded.state.release_build() {
                super::adjust_count(&mut self.building_meshes, true, false);
            }
        }
        self.pending_fresh.set();
        self.mesh_worklist.insert(coord);
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
                let placement = chunk_placement(coord);
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
            let handles =
                ByPass::from_fn(|p| eng.upload_mesh_placed(&data[p], chunk_placement(coord)));
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

    /// Streaming priority: chessboard distance with vertical axis weighted 2x (terrain before sky).
    pub(in crate::world) fn order(a: Coord, b: Coord) -> i32 {
        a.ring(b).max(2 * a.updown(b))
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
    /// Leftover columns re-arm the gate.
    pub(in crate::world) fn request_region_data(
        &mut self,
        center: Coord,
        budget: Budget,
    ) -> Progress {
        if !self.pending_gen.take() {
            return Progress::Idle;
        }
        let deadline = super::lanes::paced_deadline(self, budget);
        let mut columns: super::FastMap<(i32, i32), (i32, i32)> = super::FastMap::default();
        let mut consider = |coord: Coord| {
            if self.chunks.contains_key(&coord)
                || self.generating.contains(&coord)
                || self.quarantined.contains(&FailKey::Column {
                    col: (coord.x, coord.z),
                })
            {
                return;
            }
            let entry = columns
                .entry((coord.x, coord.z))
                .or_insert((coord.y, coord.y));
            entry.0 = entry.0.min(coord.y);
            entry.1 = entry.1.max(coord.y);
        };
        for coord in self.data_box(center).coords() {
            consider(coord);
        }
        if let Some(slab) = self.spawn_slab {
            for coord in slab.coords() {
                consider(coord);
            }
        }
        if columns.is_empty() {
            return Progress::Idle;
        }
        let slots = match self.workers.as_ref() {
            Some(w) => w.near_slots_free(),
            None => usize::MAX,
        };
        if slots == 0 {
            self.pending_gen.set();
            return Progress::Partial {
                remaining: columns.len() as u32,
            };
        }
        self.gen_columns.clear();
        self.gen_columns
            .extend(columns.into_iter().map(|((cx, cz), range)| {
                (column_order(center, self.section_vel, cx, cz), (cx, cz), range)
            }));
        let n = self.gen_columns.len();
        let min_admit = self.stream_pacer.floor(GEN_MIN_ADMIT);
        let want = n.min(slots.max(min_admit));
        if want < n {
            self.gen_columns
                .select_nth_unstable_by_key(want - 1, |e| e.0);
            self.gen_columns[..want].sort_by_key(|e| e.0);
        } else {
            self.gen_columns.sort_by_key(|e| e.0);
        }
        let mut admitted = 0usize;
        for i in 0..want {
            if super::admission_exhausted(admitted, min_admit, deadline) {
                break;
            }
            let ((cx, cz), (cy_lo, cy_hi)) = (self.gen_columns[i].1, self.gen_columns[i].2);
            if self.try_submit_column(cx, cz, cy_lo, cy_hi) {
                admitted += 1;
            } else {
                break;
            }
        }
        let remaining = n - admitted;
        if remaining > 0 {
            self.pending_gen.set();
            Progress::Partial {
                remaining: remaining as u32,
            }
        } else {
            Progress::Idle
        }
    }

    /// Land a generated column: install the skylight ceiling from the worker's
    /// heights (plus any edited-roof raise) before storing, so `trivial_light`
    /// hits the cache instead of sampling the generator on this thread.
    pub(in crate::world) fn accept_column(
        &mut self,
        col: (i32, i32),
        chunks: Vec<(Coord, Chunk)>,
        heights: Box<ColumnHeights>,
    ) {
        // Only cache when at least one chunk will actually land — an install
        // with no `column_chunks` bump would leak in `ceilings` forever.
        if chunks.iter().any(|(coord, _)| self.will_accept_chunk(*coord)) {
            self.install_ceiling(col, &heights);
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
        self.spawn_slab.is_some_and(|slab| slab.contains(coord))
            || self
                .center
                .is_some_and(|center| self.data_box(center).contains(coord))
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
            pipeline::JobKey::Column { col: (cx, cz), cy } => {
                // A cancelled spawn-slab column must be re-requested even when
                // the pool dropped it as out-of-view: physics is frozen on it.
                let in_slab = self.spawn_slab.is_some_and(|slab| {
                    cy.clone()
                        .any(|y| slab.contains(Coord::new(cx, y, cz)))
                });
                for cyy in cy {
                    self.generating.remove(&Coord::new(cx, cyy, cz));
                }
                // Freed generate claims are otherwise only re-requested on a
                // boundary cross; a retryable failure re-arms the lane so a
                // standing-still player still converges.
                if rearm || in_slab {
                    self.pending_gen.set();
                }
            }
            pipeline::JobKey::Mesh { coord } => {
                if let Some(loaded) = self.chunks.get_mut(&coord) {
                    if loaded.state.release_build() {
                        super::adjust_count(&mut self.building_meshes, true, false);
                    }
                }
                if rearm {
                    self.mesh_worklist.insert(coord);
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
        for coord in self.data_box(center).coords() {
            self.ensure_data(coord);
        }
    }

    /// Collision halo around an eye chunk: 3×3 columns, two layers below through two above.
    fn collision_slab(center: Coord) -> ChunkBox {
        ChunkBox::new(center, 1, 2)
    }

    /// Request the collision slab around `pos` from the worker pool. Does not
    /// generate on this thread — [`spawn_ready`](Self::spawn_ready) is true
    /// once every chunk of the box has loaded. Teleports and net snaps use
    /// the same request (physics freezes until it lands).
    pub fn prepare_around(&mut self, pos: DVec3) {
        let c = Self::chunk_of(block_coord(pos.x), block_coord(pos.y), block_coord(pos.z));
        let slab = Self::collision_slab(c);
        let far_m = f64::from(self.section_pyramid.outer_m());
        let view_r = self.view.horizontal;
        self.worker_pool()
            .set_view(c.x, c.z, view_r, far_m, 0.0, 0.0);
        self.submit_slab_columns(slab);
        self.pending_gen.set();
        if slab.coords().all(|coord| self.chunks.contains_key(&coord)) {
            self.spawn_slab = None;
        } else {
            self.spawn_slab = Some(slab);
        }
    }

    /// Synchronously generate the collision slab. Headless callers (tests,
    /// anything that queries voxels before a stream pass).
    pub fn ensure_around(&mut self, pos: DVec3) {
        let c = Self::chunk_of(block_coord(pos.x), block_coord(pos.y), block_coord(pos.z));
        for coord in Self::collision_slab(c).coords() {
            self.ensure_data(coord);
        }
    }

    /// True once every chunk of the requested spawn/teleport slab is loaded,
    /// or no slab is outstanding.
    pub fn spawn_ready(&self) -> bool {
        match self.spawn_slab {
            None => true,
            Some(slab) => slab.coords().all(|c| self.chunks.contains_key(&c)),
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
        if slab.coords().all(|c| self.chunks.contains_key(&c)) {
            self.spawn_slab = None;
        }
    }

    fn submit_slab_columns(&mut self, slab: ChunkBox) {
        let min = slab.min();
        let (sx, sy, sz) = slab.size();
        let cy_lo = min.y;
        let cy_hi = min.y + sy - 1;
        let mut remaining = false;
        for cx in min.x..min.x + sx {
            for cz in min.z..min.z + sz {
                let mut need = false;
                for cy in cy_lo..=cy_hi {
                    let coord = ChunkCoord::new(cx, cy, cz);
                    if !self.chunks.contains_key(&coord) && !self.generating.contains(&coord) {
                        need = true;
                        break;
                    }
                }
                if !need {
                    continue;
                }
                if !self.try_submit_column(cx, cz, cy_lo, cy_hi) {
                    remaining = true;
                }
            }
        }
        if remaining {
            self.pending_gen.set();
        }
    }

    /// Submit one column job and claim its missing coords. `false` means the
    /// pool rejected it (backpressure or shutdown) so the caller must retry.
    fn try_submit_column(&mut self, cx: i32, cz: i32, cy_lo: i32, cy_hi: i32) -> bool {
        if self.quarantined.contains(&FailKey::Column { col: (cx, cz) }) {
            return false;
        }
        let edits: Vec<(Coord, Vec<(usize, crate::block::registry::BlockId)>)> = (cy_lo
            ..=cy_hi)
            .filter_map(|cy| {
                let coord = ChunkCoord::new(cx, cy, cz);
                self.edits
                    .get(&coord)
                    .map(|cells| (coord, cells.iter().map(|(&i, &id)| (i, id)).collect()))
            })
            .collect();
        let job = pipeline::Job::GenerateColumn {
            col: (cx, cz),
            cy: cy_lo..=cy_hi,
            generator: self.generator.clone(),
            edits,
        };
        let accepted = self.worker_pool().submit(job);
        if accepted {
            for cy in cy_lo..=cy_hi {
                let coord = ChunkCoord::new(cx, cy, cz);
                if !self.chunks.contains_key(&coord) {
                    self.generating.insert(coord);
                }
            }
        }
        accepted
    }

    /// Generate a chunk's data if it isn't loaded, replaying any saved edits on it.
    /// Uses `generate_column` so the ceiling heights come from the same sample
    /// the voxels did — never a second `height()` walk on this thread.
    /// Skips coords already claimed in `generating`: the async result is imminent.
    pub(in crate::world) fn ensure_data(&mut self, coord: Coord) {
        if self.chunks.contains_key(&coord) || self.generating.contains(&coord) {
            return;
        }
        let (chunks, heights) =
            self.generator
                .generate_column(coord.x, coord.z, coord.y..=coord.y);
        self.install_ceiling((coord.x, coord.z), &heights);
        let data = chunks
            .into_iter()
            .next()
            .map(|(_, data)| data)
            .expect("generate_column emits the requested layer");
        let chunk = Chunk::from_data(coord.x, coord.y, coord.z, data);
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
        let ys = self.column_chunks.entry((coord.x, coord.z)).or_default();
        if !ys.contains(&coord.y) {
            ys.push(coord.y);
            ys.sort_unstable_by(|a, b| b.cmp(a));
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
        self.mesh_worklist.insert(coord);
        for face in Face::ALL {
            let n = coord.step(face);
            self.mesh_worklist.insert(n);
            // A Ready neighbour meshed without this chunk (terminal promotion
            // at a load-set edge, or the neighbour unloaded after the mesh).
            // Rebuild so the final look picks up the new border; worklist
            // seeding alone cannot, since Ready fails `is_needs_mesh`.
            if matches!(
                self.chunks.get(&n).map(|l| &l.state),
                Some(MeshState::Ready(_))
            ) {
                self.remesh_async(n);
            }
        }
        self.pending_fresh.set();
    }

    /// Coords in the previous unload box that have left `new_box`, or every
    /// loaded chunk past `new_box` when there is no previous box (first pass
    /// or a radius change). Spawn-slab chunks stay.
    fn unload_leaving(&self, new_box: ChunkBox) -> Vec<Coord> {
        let keep_spawn = |coord| self.spawn_slab.is_some_and(|slab| slab.contains(coord));
        match self.prev_unload_box {
            Some(prev) => prev
                .coords()
                .filter(|&coord| !new_box.contains(coord) && !keep_spawn(coord))
                .filter(|&coord| self.chunks.contains_key(&coord))
                .collect(),
            None => self
                .chunks
                .keys()
                .copied()
                .filter(|&coord| !new_box.contains(coord) && !keep_spawn(coord))
                .collect(),
        }
    }

    /// Free chunks past the unload box, releasing their GPU meshes.
    fn unload_far(&mut self, center: Coord, eng: &mut Engine) {
        let unload = self.unload_box(center);
        // Collect-then-remove instead of `retain`: freeing needs `&mut eng`,
        // which can't be borrowed inside a retain closure over `self.chunks`.
        let far = self.unload_leaving(unload);
        self.prev_unload_box = Some(unload);
        // A removed chunk changes what the BFS can reach — topology class.
        self.occlusion_topo_dirty.raise(!far.is_empty());
        for &coord in &far {
            // Free the mesh handle (Ready or Dirty); Air/NeedsMesh own none.
            if let Some(loaded) = self.chunks.remove(&coord) {
                super::adjust_count(&mut self.building_meshes, loaded.state.is_building(), false);
                loaded.state.free_owned(eng);
            }
            self.dirty_worklist.remove(&coord);
            self.light_terminal.remove(&coord);
            self.remesh_stats.forget(coord);
            // Column layers: the last chunk out drops the cached ceiling.
            if let Some(ys) = self.column_chunks.get_mut(&(coord.x, coord.z)) {
                if let Some(i) = ys.iter().position(|&y| y == coord.y) {
                    ys.remove(i);
                }
                if ys.is_empty() {
                    self.column_chunks.remove(&(coord.x, coord.z));
                    self.ceilings.remove(&(coord.x, coord.z));
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
        dirty.sort_by_key(|&coord| Self::order(coord, center));
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
        (
            loaded.rev,
            pipeline::ChunkSnapshot {
                padded: mesh::Padded::capture(|dx, dy, dz| {
                    Self::nhood_at(&nhood, dx, dy, dz).map(|l| &*l.chunk)
                }),
                uniform: loaded.chunk.uniform(),
                // Lighting off omits the 18³ shell entirely — the mesher's
                // unlit path reads constant full light instead. `open_sky` is
                // Uniform (task 05): the degraded fallback does not allocate.
                light: self.lighting.then(|| {
                    light::PaddedLight::capture(|dx, dy, dz| {
                        Self::nhood_at(&nhood, dx, dy, dz)
                            .and_then(|l| l.light.as_ref())
                            .or(fallback.as_ref())
                    })
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
        light::PaddedLight::capture(|dx, dy, dz| {
            self.chunks
                .get(&Coord::new(coord.x + dx, coord.y + dy, coord.z + dz))
                .and_then(|l| l.light.as_ref())
                .or(fallback.as_ref())
        })
    }

    /// Near-face light from six neighbours (input for light flood seeds).
    pub(in crate::world) fn capture_face_shell(&self, coord: Coord) -> light::FaceShell {
        if !self.lighting {
            return light::FaceShell::dark();
        }
        light::FaceShell::capture(|face| {
            self.chunks
                .get(&coord.step(face))
                .and_then(|l| l.light.as_ref())
        })
    }

    /// Skylight ceiling: ground height per column (caves dark consistently)
    /// RAISED by edited opaque roofs, so a player-built ceiling shadows the
    /// chunks below it. Keyed by `(x, z)` chunk column and cached.
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
        if let Some(ceiling) = self.ceilings.get(&(coord.x, coord.z)) {
            return std::sync::Arc::clone(ceiling);
        }
        // Empty `cy` range: both generators sample the 256 column profiles
        // before iterating the chunk layers, so this is the height field
        // without a voxel fill.
        let heights = self.generator.generate_column(coord.x, coord.z, 1..=0).1;
        let ceiling = std::sync::Arc::new(self.ceiling_from_heights((coord.x, coord.z), &heights));
        self.ceilings
            .insert((coord.x, coord.z), std::sync::Arc::clone(&ceiling));
        ceiling
    }

    /// Rebuild the ceiling from `height()` plus edited roofs, ignoring the
    /// cache — equality check against production `generate_column` heights.
    #[cfg(test)]
    pub(in crate::world) fn capture_ceiling_slow(&self, coord: Coord) -> light::CeilingWindow {
        let x0 = coord.x * CHUNK_SIZE as i32;
        let z0 = coord.z * CHUNK_SIZE as i32;
        let generator = &self.generator;
        let mut ceiling = light::CeilingWindow::from_heights(|lx, lz| {
            generator.height(x0 + lx as i32, z0 + lz as i32)
        });
        self.raise_edited_roofs((coord.x, coord.z), &mut ceiling);
        ceiling
    }

    fn install_ceiling(&mut self, col: (i32, i32), heights: &ColumnHeights) {
        if self.ceilings.contains_key(&col) {
            return;
        }
        let ceiling = std::sync::Arc::new(self.ceiling_from_heights(col, heights));
        self.ceilings.insert(col, ceiling);
    }

    fn ceiling_from_heights(
        &self,
        col: (i32, i32),
        heights: &ColumnHeights,
    ) -> light::CeilingWindow {
        let mut ceiling =
            light::CeilingWindow::from_heights(|lx, lz| heights[lx + lz * CHUNK_SIZE]);
        self.raise_edited_roofs(col, &mut ceiling);
        ceiling
    }

    fn raise_edited_roofs(&self, col: (i32, i32), ceiling: &mut light::CeilingWindow) {
        for (&c, cells) in &self.edits {
            if c.x != col.0 || c.z != col.1 {
                continue;
            }
            for (&index, &id) in cells {
                if !self.registry.is_opaque(id) {
                    continue;
                }
                let (lx, ly, lz) = Chunk::local_of(index);
                ceiling.raise(lx, lz, c.y * CHUNK_SIZE as i32 + ly as i32 + 1);
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
    ///
    /// Correctness anchor: the returned grid equals `propagate(uniform, dark
    /// shell, ceiling, world_y0, tables)`.
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
            let world_y0 = coord.y * CHUNK_SIZE as i32;
            let ceiling = self.capture_ceiling(coord);
            if world_y0 >= ceiling.min_surface() && !self.neighbour_blocklight_near(coord) {
                return Some(light::LightGrid::open_sky());
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
                .get(&coord.step(face))
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
        self.mesh_worklist.insert(coord);
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
        self.mesh_worklist.insert(coord);
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
            let n = coord.step(face);
            if !self.chunks.contains_key(&n) {
                continue;
            }
            if face_moved {
                // Open-sky next to open-sky cannot change. Mesh-seed only:
                // this publish can complete their light_ready.
                let n_sky = self.chunks[&n].light.as_ref() == Some(&light::LightGrid::open_sky());
                if open_sky && n_sky {
                    self.mesh_worklist.insert(n);
                    continue;
                }
                if self.light_inflight.contains(&n) {
                    self.chunks.get_mut(&n).unwrap().light_reseed = true;
                } else if !self.light_worklist.contains(&n) {
                    // Pending floods read live neighbour grids at admit.
                    self.seed_light(n, super::LightSeed::Border);
                }
            }
            self.mesh_worklist.insert(n);
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
        mesh::Padded::capture(|dx, dy, dz| {
            self.chunks
                .get(&Coord::new(coord.x + dx, coord.y + dy, coord.z + dz))
                .map(|l| &*l.chunk)
        })
    }

    // Column-LOD section selection and streaming.

    /// Per-frame selection metric: chunk-centre XZ, `dy` from eye altitude to LOD envelope.
    /// XZ stays on chunk centre (not eye) to keep `dy=0` bit-identical to shipped LOD2.
    /// `delta` is prediction offset applied as inflation off the static anchor.
    fn section_metric(&self, center: Coord, delta: DVec3) -> EyeMetric {
        let cs = CHUNK_SIZE as i32;
        let (pcx, pcz) = (center.x * cs + cs / 2, center.z * cs + cs / 2);
        let cfg = &self.section_pyramid;
        EyeMetric::new(
            DVec3::new(
                pcx as f64 + delta.x,
                self.section_eye_y + delta.y,
                pcz as f64 + delta.z,
            ),
            HeightEnvelope::new(
                super::section::LOD_FLOOR_Y as f32,
                super::section::LOD_CEIL_Y as f32,
            ),
            DyCap::new(cfg.outer_m(), cfg.base),
        )
    }

    /// Desired frontier at one metric: radial ladder, coarsened by per-cell relief
    /// once max-mip bakes.
    fn frontier(&self, metric: &EyeMetric) -> Vec<SectionPos> {
        let cfg = &self.section_pyramid;
        let radial = quadtree::desired_sections(metric, cfg);
        let selected = match &self.section_mip {
            Some(mip) => {
                let summary_at = |c: SectionPos| match self.section_overlay.get(&c) {
                    Some(ov) => CellSummary {
                        env: HeightEnvelope::new(ov.lo, ov.hi),
                        err: CellError::from_metres(ov.hi - ov.lo),
                    },
                    None => mip.summary(c),
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

    /// Spawn background max-mip bake (idempotent: no-op if spawned or landed).
    /// Runs off main thread; worst-case ladder streams meanwhile.
    pub(in crate::world) fn ensure_mip_bake(&mut self) {
        if self.section_mip.is_some() || self.section_mip_rx.is_some() {
            return;
        }
        let generator = self.generator.clone();
        // The generator stores resolved IDs for every element-worldgen
        // composition registered during `World::new`. A fresh builtin registry
        // is too short for those IDs; snapshot the matching color table instead.
        let colors = self.registry.color_snapshot();
        let cfg = &self.section_pyramid;
        let extent = BakeExtent::new(cfg.outer_m() as i32, cfg.coarsest());
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(HeightMip::bake(&*generator, &colors, extent));
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
    pub(in crate::world) fn refresh_section_overlay(&mut self) {
        if self.section_overlay_dirty.is_empty() {
            return;
        }
        let positions: Vec<SectionPos> = self.section_overlay_dirty.drain().collect();
        for pos in positions {
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
                Some(super::heightmip::resample_cell(
                    pos,
                    &*self.generator,
                    &touched,
                    &colors,
                ))
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
    }

    /// Desired frontier: union of static eye and velocity-predicted eye position.
    /// Pulls sections ahead of player motion. At rest, velocity is zero so returns
    /// static frontier bit-for-bit.
    pub(in crate::world) fn desired_sections(&self, center: Coord) -> Vec<SectionPos> {
        let base = self.frontier(&self.section_metric(center, DVec3::ZERO));
        let delta = self.section_vel * TAU_STREAM;
        if delta == DVec3::ZERO {
            return base;
        }
        let predicted = self.frontier(&self.section_metric(center, delta));
        quadtree::union_frontiers(base, predicted)
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
        let Some(center) = self.center else {
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

    /// Unload sections outside desired, visible, and hysteresis bands (boundary cross).
    /// Hysteresis prevents thrashing at view edges.
    fn unload_sections(&mut self, center: Coord, eng: &mut Engine) {
        // KEEP reads the frame's cached frontier (already velocity-unioned),
        // so sections stay kept even as a fast-moving eye passes.
        let desired: FastSet<SectionPos> = self.section_desired.iter().copied().collect();
        let visible: FastSet<SectionPos> = self.section_visible.iter().map(|(p, _)| *p).collect();
        // Fading sections still draw this frame. Keep meshes until fade completes
        // or outgoing section vanishes mid-fade.
        let fading: FastSet<SectionPos> = self.section_fade.tracked().collect();
        let metric = self.section_metric(center, DVec3::ZERO);
        let cfg = &self.section_pyramid;
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
                let dist = metric.point(cx as f64, cz as f64);
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
                state.free(eng);
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

    /// Six orthogonal neighbours have data loaded.
    pub(in crate::world) fn neighbours_have_data(&self, coord: Coord) -> bool {
        Face::ALL
            .iter()
            .all(|&f| self.chunks.contains_key(&coord.step(f)))
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
                    .get(&coord.step(f))
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
        // Promote dirty (and relit-degraded) chunks once the 27-neighbourhood
        // has no pending light work, or the degrade timer has expired. One
        // pass over the dirty/degraded sets, never the world.
        let mut promote: Vec<Coord> = gate
            .dirty
            .iter()
            .filter(|(c, t)| self.light_nhood_quiet(**c) || t.elapsed() >= LIGHT_WAIT_DEGRADE)
            .map(|(c, _)| *c)
            .collect();
        for &c in &gate.degraded {
            if gate.dirty.contains_key(&c) {
                continue;
            }
            if self.light_ready(c)
                && (self.light_nhood_quiet(c)
                    || gate
                        .blocked_since
                        .get(&c)
                        .is_some_and(|t| t.elapsed() >= LIGHT_WAIT_DEGRADE))
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
        let mut expired = false;
        for (&c, t) in &gate.blocked_since {
            if t.elapsed() >= LIGHT_WAIT_DEGRADE {
                self.mesh_worklist.insert(c);
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
            if !self.light_nhood_quiet(coord) {
                continue;
            }
            match self.chunks.get(&coord).map(|l| &l.state) {
                // Nothing to draw: drop the degraded flag.
                Some(MeshState::Air) => self.mark_degraded(coord, false),
                // Settled on a degraded mesh — the stuck case. Rebuild async;
                // if neighbour light is still missing it will never arrive, so
                // the terminal set makes the snapshot read missing planes dark.
                Some(MeshState::Ready(_)) => {
                    if !self.in_mesh_box(coord) {
                        // Past the mesh box (unload hysteresis): not drawn,
                        // and a rebuild would fail `in_mesh_box` / be dropped
                        // at apply. Drop the flag so quiescence is not wedged.
                        self.mark_degraded(coord, false);
                    } else {
                        if !self.light_ready(coord) {
                            self.light_terminal.insert(coord);
                        }
                        self.remesh_async(coord);
                    }
                }
                // Unloaded out from under the set between marking and here.
                None => self.mark_degraded(coord, false),
                // Admit evicted the seed (a missing neighbour used to fail
                // `ready`, or this flush ran after that pass's admit). Re-seed
                // in-box chunks; mark terminal if neighbour light will not
                // arrive. Out-of-box: same as Ready — drop the flag.
                Some(MeshState::NeedsMesh {
                    building: false, ..
                }) => {
                    if !self.in_mesh_box(coord) {
                        self.mark_degraded(coord, false);
                    } else if !self.mesh_worklist.contains(&coord) {
                        if !self.light_ready(coord) {
                            self.light_terminal.insert(coord);
                        }
                        self.mesh_worklist.insert(coord);
                        self.pending_fresh.set();
                    }
                }
                // Still building (in-flight result pending) or Dirty (a sync
                // remesh owns it): another path is about to resolve it.
                Some(MeshState::NeedsMesh { building: true, .. } | MeshState::Dirty { .. }) => {}
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
        for coord in self.mesh_box(center).coords() {
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
            reactions_mutations: self.reactions.mutations,
        }
    }

    /// Human-readable reason `entry_complete` is not yet true — the first
    /// unsatisfied clause with a count, so a stalled bless/harness run says WHICH
    /// streaming stage is stuck instead of hanging silently. Clause order mirrors
    /// [`entry_complete`](Self::entry_complete).
    pub fn entry_debug(&self) -> String {
        if let Some(slab) = self.spawn_slab {
            let missing = slab
                .coords()
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
        for c in self.mesh_box(center).coords() {
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

    /// Rebuild/upload block texture array on descriptor growth (rare: world entry
    /// or a newly interned look) or an appearance `revision` change. Existing
    /// layers never change at one revision (pure function of the visual;
    /// descriptor ids are append-only), so only the first upload / a revision
    /// rebuild uses `set_block_textures`; later growth appends.
    fn refresh_textures(&mut self, eng: &mut Engine, appearance: &dyn BlockAppearance) {
        // Never zero (modulo divisor) and never past the vertex field's u16.
        // Construction caches `u16::MAX`; the device cap is read once.
        if !self.texture_cap_from_device {
            self.texture_layer_cap = eng.max_texture_array_layers().clamp(1, u16::MAX as u32) as u16;
            self.texture_cap_from_device = true;
            #[cfg(test)]
            crate::alloc_count::note_engine(crate::alloc_count::EngineCall::TexLayers);
        }
        let count = self.registry.descriptor_count();
        let rev = appearance.revision();
        let gpu = appearance.wants_gpu_descriptors();
        if self.appearance_revision != rev || self.appearance_gpu != gpu {
            self.texture_cache.clear();
            self.uploaded_len = 0;
            self.textures_built = 0;
            self.appearance_revision = rev;
            self.appearance_gpu = gpu;
        }
        if self.textures_built == count {
            return;
        }
        if gpu {
            for i in self.texture_cache.len()..count {
                let vis = self.registry.descriptor(i as u16);
                self.texture_cache
                    .push(placeholder_layer(&vis, i as u16));
            }
        } else {
            for i in self.texture_cache.len()..count {
                let mut buf = [0u8; LAYER_BYTES];
                fill_descriptor_layer(appearance, &self.registry, i as u16, &mut buf);
                self.texture_cache.push(buf.to_vec());
            }
        }
        let visible = count.min(self.texture_layer_cap as usize);
        if count > visible && self.uploaded_len < visible {
            eprintln!(
                "render descriptors ({count}) exceed the device texture-layer cap \
                 ({visible}); further textures wrap onto existing layers"
            );
        }
        let texel = if gpu { 1 } else { TEXTURE_SIZE };
        let upload = plan_texture_upload(&self.texture_cache, self.uploaded_len, visible);
        let desc_range = match &upload {
            Some(TextureUpload::Set(_)) => Some((0, visible)),
            Some(TextureUpload::Append(_)) => Some((self.uploaded_len, visible)),
            None => None,
        };
        let descs: Option<Vec<MaterialDesc>> = if gpu {
            desc_range.map(|(lo, hi)| {
                (lo..hi)
                    .map(|i| procedural_material_desc(&self.registry.descriptor(i as u16)))
                    .collect()
            })
        } else {
            None
        };
        match upload {
            Some(TextureUpload::Set(layers)) => {
                eng.set_block_textures(texel, layers);
            }
            Some(TextureUpload::Append(layers)) => {
                eng.append_block_textures(layers);
            }
            None => {}
        }
        if let Some(descs) = descs {
            match desc_range {
                Some((0, _)) => eng.set_material_descs(&descs),
                Some(_) => eng.append_material_descs(&descs),
                None => {}
            }
            self.gpu_descs_uploaded = true;
        } else if self.gpu_descs_uploaded {
            // Engine default is ARRAY_LAYER per slot; an empty set restores it.
            eng.set_material_descs(&[]);
            self.gpu_descs_uploaded = false;
        }
        self.uploaded_len = visible;
        self.textures_built = count;
    }

    /// Headless settle: centre on `pos`, fill the mesh box with data, mark every
    /// in-view chunk as a final Air mesh, and drain worklists so
    /// [`entry_complete`](Self::entry_complete) holds without an Engine.
    #[cfg(test)]
    pub fn settle_around(&mut self, pos: DVec3) {
        let s = CHUNK_SIZE as i32;
        let center = ChunkCoord::new(
            block_coord(pos.x).div_euclid(s),
            block_coord(pos.y).div_euclid(s),
            block_coord(pos.z).div_euclid(s),
        );
        self.center = Some(center);
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
mod tests {
    use super::super::StreamLane;
    use super::*;

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
    /// would. Covers classic, diffusion, and an edited-roof raise.
    #[test]
    fn accept_column_caches_ceiling_and_trivial_light_matches_slow() {
        use crate::render_config::RenderConfig;
        use crate::world::generation::WorldgenKind;

        for kind in [WorldgenKind::Classic, WorldgenKind::Diffusion] {
            let mut world = World::with_kind(7, RenderConfig::default(), kind, false);
            // Above terrain and the flying-island band: uniform air, open sky.
            let coord = Coord::new(1, 25, -2);
            world.center = Some(coord);

            let stone = world.registry.id_by_label("rock").expect("builtin Stone");
            // Roof in this column, below the stored chunk: raise before store.
            const ROOF_Y: i32 = 200;
            world.set_block(
                coord.x * CHUNK_SIZE as i32 + 3,
                ROOF_Y,
                coord.z * CHUNK_SIZE as i32 + 5,
                stone,
            );

            let slow = world.capture_ceiling_slow(coord);
            assert!(
                !world.ceilings.contains_key(&(coord.x, coord.z)),
                "slow helper must not warm the cache"
            );

            let (datas, heights) =
                world
                    .generator
                    .generate_column(coord.x, coord.z, coord.y..=coord.y);
            let chunks: Vec<_> = datas
                .into_iter()
                .map(|(cy, data)| {
                    (
                        Coord::new(coord.x, cy, coord.z),
                        Chunk::from_data(coord.x, cy, coord.z, data),
                    )
                })
                .collect();
            world.accept_column((coord.x, coord.z), chunks, Box::new(heights));

            let cached = world
                .ceilings
                .get(&(coord.x, coord.z))
                .expect("accept_column installs the ceiling before store");
            assert_ceilings_eq(cached.as_ref(), &slow);
            assert!(
                cached.surface_at(3, 5) >= ROOF_Y + 1,
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
            .get(&(coord.x, coord.z))
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
}
