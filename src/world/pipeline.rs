//! pipeline.rs runs chunk generation and meshing on background threads so the
//! render thread never pays for either — a tiny std-only worker pool (zero
//! dependencies) fed and drained by [`World::stream`](super::World::stream).
//!
//! Threading model:
//! - `min(3, cores - 1).max(1)` worker threads share ONE [`JobQueue`] behind a
//!   `Mutex` + `Condvar`. A worker holds the lock only while dequeuing (or
//!   waiting for work); every job runs unlocked.
//! - Jobs carry owned value data only (a shared immutable generator, a voxel snapshot,
//!   border planes, an `Arc`'d solidity table). Workers never touch the GPU,
//!   the `World`, or the live chunk map, so there is nothing to contend on
//!   and nothing that can deadlock against the render thread.
//! - Priority is two-class, near-preferred: near work (generate/mesh/light)
//!   dequeues before far LOD work (tile/skin), so a burst of slow tile jobs
//!   can never make the chunk under the player wait behind them. Within a class
//!   the order is FIFO, and the world sorts each batch nearest-first first.
//! - Results come back on a plain `mpsc` channel, drained non-blockingly once
//!   per frame. The main thread re-validates every result on arrival (the
//!   chunk may have unloaded, edits may have landed while the job flew).
//! - Shutdown: dropping [`Workers`] closes the job sender, so a blocked
//!   `recv()` errors out and each loop exits; `Drop` then joins the handles.
//!   Bounded even mid-burst (a worker finishes at most its current job), and
//!   a stuck GPU can never block it — workers never issue GPU calls.
use std::ops::{Deref, DerefMut, RangeInclusive};
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::Coord;
use super::chunk::Chunk;
use super::generation::{SineHills, TerrainGenerator};
use super::light::{self, CeilingWindow, FaceShell, LightGrid, PaddedLight};
use super::mesh::{self, ChunkMeshData, Padded, new_chunk_mesh_data};
use super::section::{self, SectionMeshData, SectionPos};
use crate::block::registry::{BlockId, HotTables};

/// Mesh job snapshot: pure mesher state (light pre-settled, no live chunk map sharing).
pub struct ChunkSnapshot {
    /// The chunk's voxels plus a one-voxel shell from its 26 neighbours — the sole
    /// voxel source for the mesh (it subsumes the chunk clone).
    pub padded: Padded,
    /// The chunk's uniform block id, if uniform — drives the mesher fast paths
    /// (uniform air ⇒ empty, uniform solid ⇒ border slices only).
    pub uniform: Option<BlockId>,
    /// The settled light shell sampled per vertex. `None` is the allocation-free
    /// constant-full-light path used when voxel lighting is disabled.
    pub light: Option<PaddedLight>,
    /// The hot tables (solid/opaque/emission), shared by refcount. Palette growth
    /// swaps the world's `Arc` for a new one while in-flight jobs keep the old —
    /// harmless because the palette is append-only and every result is
    /// re-validated on arrival anyway.
    pub tables: Arc<HotTables>,
}

/// Cross-thread pool for greedy-mesh output. Geometry vectors retain their
/// capacities after upload/stale rejection, so traversal can refill them
/// without allocating vertices and index buckets from scratch for every job.
// The box is intentional: besides being recycled with the geometry, it keeps
// `Done::Mesh` pointer-sized instead of inflating every result-channel message.
#[allow(clippy::vec_box)]
static MESH_OUTPUT_POOL: Mutex<Vec<Box<ChunkMeshData>>> = Mutex::new(Vec::new());
const MESH_OUTPUT_POOL_CAP: usize = 32;

pub(in crate::world) struct MeshOutput(Option<Box<ChunkMeshData>>);

impl MeshOutput {
    pub(in crate::world) fn new() -> Self {
        let data = MESH_OUTPUT_POOL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop()
            .unwrap_or_else(|| Box::new(new_chunk_mesh_data()));
        Self(Some(data))
    }
}

impl Deref for MeshOutput {
    type Target = ChunkMeshData;

    fn deref(&self) -> &Self::Target {
        self.0.as_deref().expect("live pooled mesh output")
    }
}

impl DerefMut for MeshOutput {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_deref_mut().expect("live pooled mesh output")
    }
}

impl Drop for MeshOutput {
    fn drop(&mut self) {
        let Some(data) = self.0.take() else { return };
        let mut pool = MESH_OUTPUT_POOL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if pool.len() < MESH_OUTPUT_POOL_CAP {
            pool.push(data);
        }
    }
}

/// Light-settle job snapshot: the pure inputs [`light::propagate`] reads. All
/// owned/refcounted, so the worker touches neither the live chunk map nor the
/// neighbour grids — it recomputes this chunk's grid from a frozen neighbourhood.
pub struct LightSnapshot {
    /// Chunk voxels (opacity/emission source); shared by refcount.
    pub chunk: Arc<Chunk>,
    /// Near-face light of the 6 neighbour faces (snapshot at enqueue time).
    pub shell: FaceShell,
    /// The skylight ceiling (surface heightmap) for the chunk's column.
    pub ceiling: CeilingWindow,
    /// World-space Y of the chunk's bottom cell — seeds the open-sky column test.
    pub world_y0: i32,
    /// Hot tables (opaque/emission), shared by refcount like a mesh snapshot's.
    pub tables: Arc<HotTables>,
}

/// Work sent to the pool.
pub(in crate::world) enum Job {
    /// Generate a whole vertical *column* of chunks at horizontal `col = (cx,
    /// cz)` over the chunk-layer range `cy`, from one shared generator. The
    /// column profile (`profile(wx, wz)`) is `cy`-invariant, so generating the
    /// run together samples it once instead of R times. `edits` carries the
    /// per-chunk edit overlay (`(coord, [(flat index, block)])`) replayed after
    /// each chunk's fill — voxel-identical to per-chunk generation.
    GenerateColumn {
        col: (i32, i32),
        cy: RangeInclusive<i32>,
        generator: Arc<SineHills>,
        edits: Vec<(Coord, Vec<(usize, BlockId)>)>,
    },
    /// Greedy-mesh a snapshot taken at chunk revision `rev`.
    Mesh {
        coord: Coord,
        rev: u32,
        snapshot: ChunkSnapshot,
    },
    /// Relax the light grid for `coord` from a frozen neighbourhood snapshot.
    Light {
        coord: Coord,
        epoch: u32,
        snapshot: Box<LightSnapshot>,
    },
    /// Extract and mesh a section's columns at its detail level. Pure computation
    /// with no live snapshots or neighbor lookups.
    Section {
        pos: SectionPos,
        epoch: u32,
        token: u64,
        generator: Arc<SineHills>,
        edits: Vec<(Coord, Vec<(usize, BlockId)>)>,
        tables: Arc<HotTables>,
    },
    /// Test-only: panics inside `run`, reporting the given claim — the injector
    /// for the worker-panic → `Done::Failed` → claim-release path.
    #[cfg(test)]
    Panic(Box<JobKey>),
}

/// The claim a job holds while in flight, extractable from the job itself.
/// A panicking job returns this in [`Done::Failed`] so the main thread can
/// release the EXACT claim instead of leaving it stranded forever.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::world) enum JobKey {
    Column { col: (i32, i32), cy: RangeInclusive<i32> },
    Mesh { coord: Coord },
    Light { coord: Coord },
    Section {
        pos: SectionPos,
        epoch: u32,
        token: u64,
    },
}

impl JobKey {
    /// The claim identity of a job, captured before the job runs.
    fn of(job: &Job) -> JobKey {
        match job {
            Job::GenerateColumn { col, cy, .. } => JobKey::Column { col: *col, cy: cy.clone() },
            Job::Mesh { coord, .. } => JobKey::Mesh { coord: *coord },
            Job::Light { coord, .. } => JobKey::Light { coord: *coord },
            Job::Section {
                pos, epoch, token, ..
            } => JobKey::Section {
                pos: *pos,
                epoch: *epoch,
                token: *token,
            },
            #[cfg(test)]
            Job::Panic(key) => (**key).clone(),
        }
    }
}

/// Finished work returned to the main thread.
pub(in crate::world) enum Done {
    /// A generated column: every chunk built for the requested `cy` range,
    /// paired with its coord. Landed together and stored in one drain step.
    Column { col: (i32, i32), chunks: Vec<(Coord, Chunk)> },
    /// Pointer-sized pooled geometry output. It rides untouched into the upload
    /// queue, then its retained Vec capacities return to the worker pool.
    Mesh { coord: Coord, rev: u32, data: MeshOutput },
    Light { coord: Coord, epoch: u32, grid: LightGrid },
    Section {
        pos: SectionPos,
        epoch: u32,
        token: u64,
        meshes: [SectionMeshData; 4],
    },
    /// The job PANICKED. Carries its claim so `World::fail_job` can release it
    /// and apply the bounded retry/quarantine policy — without this, a single
    /// bad job left `generating`/`light_inflight`/`building`/`Meshing` claimed
    /// forever and streaming never converged.
    Failed(Box<JobKey>),
    /// The job was DESCHEDULED at the pool before running: its region left
    /// the live view while it sat queued (fast movement). `World::cancel_job`
    /// releases the claim — no strike, no requeue; the normal boundary-cross
    /// scans re-request the work if the player ever comes back.
    Cancelled(Box<JobKey>),
}

// Keep the result channel payload small: the largest variant should be the
// `Section` mesh array, not an inlined per-chunk mesh (see structural
// opportunity #8 — this was 544 B with `ChunkMeshData` inline).
const _: () = assert!(size_of::<Done>() <= 128);

/// A job's scheduling class. Derived from its kind — near work outranks far LOD
/// work — so it never rides along on the wire as a redundant field.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Priority {
    /// Chunks near the player: generate, mesh, light. Dequeued first.
    Near,
    /// Far LOD geometry: column sections. Dequeued only when no near work waits.
    Far,
}

fn priority(job: &Job) -> Priority {
    match job {
        Job::Section { .. } => Priority::Far,
        _ => Priority::Near,
    }
}

impl Job {
    /// The horizontal chunk column a NEAR job serves, for live-view distance
    /// ordering and descheduling. `None` for far/section work (the far queue
    /// carries its own distance keys) and test-only jobs.
    fn col(&self) -> Option<(i32, i32)> {
        match self {
            Job::GenerateColumn { col, .. } => Some(*col),
            Job::Mesh { coord, .. } => Some((coord.x, coord.z)),
            Job::Light { coord, .. } => Some((coord.x, coord.z)),
            Job::Section { .. } => None,
            #[cfg(test)]
            Job::Panic(_) => None,
        }
    }
}

/// Chunks past the view radius a queued job survives before it is descheduled
/// at the pool. Strictly wider than [`UNLOAD_MARGIN`](super::World)'s unload
/// ring, so a job whose result would still be integrated is never cancelled —
/// only genuinely left-behind work is.
const CANCEL_MARGIN: i32 = 4;

/// The live view, shared with the worker pool and consulted at DEQUEUE time.
/// The world stores the streaming centre and radius here every frame; workers
/// then (a) pop the near job CLOSEST to where the player is NOW — not where
/// they were when it was enqueued — and (b) deschedule queued jobs whose
/// region fell out of range entirely. Fast movement therefore reorders the
/// backlog every pop and sheds it instead of grinding through stale regions.
///
/// Two relaxed atomics: centre and radius may briefly disagree mid-update;
/// [`CANCEL_MARGIN`] absorbs the tear (it can only mis-order or briefly spare
/// a job, never cancel wanted work — the margin exceeds any one-frame move).
pub(in crate::world) struct ViewGate {
    /// `(cx as u32) << 32 | (cz as u32)`.
    center: AtomicU64,
    /// Horizontal view radius in chunks; `i32::MAX` (permissive) until set.
    radius: AtomicI32,
}

impl ViewGate {
    fn new() -> Self {
        Self { center: AtomicU64::new(0), radius: AtomicI32::new(i32::MAX) }
    }

    fn set(&self, cx: i32, cz: i32, radius: i32) {
        self.center.store(((cx as u32 as u64) << 32) | (cz as u32 as u64), Ordering::Relaxed);
        self.radius.store(radius, Ordering::Relaxed);
    }

    fn center(&self) -> (i32, i32) {
        let packed = self.center.load(Ordering::Relaxed);
        ((packed >> 32) as u32 as i32, packed as u32 as i32)
    }

    /// Chessboard chunk distance from the live centre, `0` while permissive.
    fn dist(&self, cx: i32, cz: i32) -> i32 {
        if self.radius.load(Ordering::Relaxed) == i32::MAX {
            return 0;
        }
        let (px, pz) = self.center();
        (cx - px).abs().max((cz - pz).abs())
    }

    /// Whether a job at this column is still worth running.
    fn wanted(&self, cx: i32, cz: i32) -> bool {
        let radius = self.radius.load(Ordering::Relaxed);
        radius == i32::MAX || self.dist(cx, cz) <= radius + CANCEL_MARGIN
    }
}

/// Admission-control deadline: enqueue/apply loops check it *between* items and
/// never abort an item already admitted. Time, not counts — so a burst of cheap
/// items and a burst of expensive ones no longer share one integer "budget".
#[derive(Clone, Copy, Debug)]
pub struct Deadline(Instant);

impl Deadline {
    /// A deadline `budget` from now.
    pub fn from_budget(budget: Duration) -> Deadline {
        Deadline(Instant::now() + budget)
    }
    #[must_use]
    pub fn expired(self) -> bool {
        Instant::now() >= self.0
    }
}

// All three tuned against time_to_first_full_render — the
// values are a first cut, not measured optima. Chunk streaming gets the lion's
// share; the far LOD ring and the light settle enqueue each get a slim slice so
// a world-entry flood of either can't stall the chunk under the player.
/// Per-frame admission budget for fresh chunk meshing (the [`MeshLane`] enqueue).
pub const STREAM_BUDGET: Duration = Duration::from_millis(2);
/// Per-frame admission budget for the cross-chunk light settle work. The
/// *apply* drain in `streaming.rs` and the [`LightLane`] enqueue each mint their
/// OWN window from this value (two loops, two windows — the per-frame light cost
/// is their sum).
pub const LIGHT_APPLY_BUDGET: Duration = Duration::from_millis(1);
/// Per-frame admission budget for a far LOD lane (tiles and skins each mint one).
pub const LOD_ENQUEUE_BUDGET: Duration = Duration::from_millis(1);

// Each loop mints a FRESH `Deadline::from_budget(...)` at the instant it starts —
// never one frame-start snapshot shared across lanes. The lanes run sequentially
// (drain → light → mesh → LOD), so a single anchored instant would leave every
// lane after the first ~1 ms pre-expired and admitting nothing (world-entry
// starvation: `MeshLane`/`LightLane` never drain). Budgets are admission caps,
// so idle lanes still return immediately.

/// Far-queue cap. At the cap [`Workers::submit_far`] REJECTS
/// the submit (returns `false`) and the lane simply does not claim the key, so
/// it retries naturally on a later frame — retry-not-drop lives at the
/// requester. Rejection at admission, never eviction after acceptance: an
/// accepted far job has already been claimed by its lane (`Meshing` state), and
/// a claimed key is owed exactly one `Done` — evicting it would strand the claim
/// forever (a permanent hole + an `entry_complete` hang).
pub const FAR_QUEUE_CAP: usize = 256;

/// One far-queue entry's ordering key: distance first, then a monotone sequence
/// number so equal-distance jobs keep FIFO order.
#[derive(Clone, Copy, Debug)]
struct FarEntry {
    dist2: u64,
    seq: u64,
}

/// The far scheduling class: nearest-first pop, FIFO tie-break on equal `dist2`
/// via the monotone `seq`. Replaces the far `VecDeque` in [`JobQueue`] (the near
/// class keeps its FIFO — its batches already arrive nearest-sorted).
/// Representation is a flat `Vec` scanned on pop: the queue is small (capped at
/// [`FAR_QUEUE_CAP`] by admission) and pop runs only a handful of times per
/// frame, so the scan beats a heap's constant factor and keeps the FIFO
/// tie-break trivial.
#[derive(Default)]
pub struct FarQueue {
    entries: Vec<(FarEntry, Job)>,
    next_seq: u64,
}

impl FarQueue {
    /// Push `job` keyed by `dist2` (squared euclidean METRES from the job's
    /// world-space centre to the player, computed at submit).
    pub(in crate::world) fn push(&mut self, job: Job, dist2: u64) {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.entries.push((FarEntry { dist2, seq }, job));
    }

    /// Pop the nearest job: lowest `dist2`, earliest `seq` breaking ties.
    pub(in crate::world) fn pop_nearest(&mut self) -> Option<Job> {
        let idx = self
            .entries
            .iter()
            .enumerate()
            .min_by_key(|(_, (e, _))| (e.dist2, e.seq))
            .map(|(i, _)| i)?;
        Some(self.entries.swap_remove(idx).1)
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn clear_claims(&mut self) -> Vec<JobKey> {
        self.entries
            .drain(..)
            .map(|(_, job)| JobKey::of(&job))
            .collect()
    }
}

/// Two-class queue shared by the pool. `pop` drains `near` fully before `far`,
/// so far LOD jobs fill idle workers without ever starving the chunk under the
/// player. Near is LIVE-distance-ordered against the [`ViewGate`] (nearest to
/// where the player is NOW pops first, and left-behind entries are descheduled
/// at pop); far is distance-ordered at admission (see [`FarQueue`]). `closed`
/// is the shutdown flag a blocked `pop` wakes on.
#[derive(Default)]
struct JobQueue {
    near: Vec<(u64, Job)>,
    near_seq: u64,
    far: FarQueue,
    closed: bool,
}

impl JobQueue {
    fn clear_far(&mut self) -> Vec<JobKey> {
        self.far.clear_claims()
    }

    /// Push at the job's scheduling class. A far job pushed here (the legacy
    /// [`Workers::submit`] path and headless tests) carries no distance, so it
    /// sorts at `dist2 = 0` and equal-distance far jobs fall back to FIFO by
    /// `seq` — the old `VecDeque` order. This legacy path is uncapped (its only
    /// producers are tests); the streaming lanes go through the cap-checked
    /// [`Workers::submit_far`].
    fn push(&mut self, job: Job) {
        match priority(&job) {
            Priority::Near => {
                let seq = self.near_seq;
                self.near_seq += 1;
                self.near.push((seq, job));
            }
            Priority::Far => self.far.push(job, 0),
        }
    }

    /// Admit a far job keyed by `dist2`, or REJECT it at [`FAR_QUEUE_CAP`]
    /// (returns whether it was admitted). Rejection is the whole cap mechanism:
    /// the lane never claims a rejected key, so it retries on a later frame.
    #[must_use]
    fn push_far(&mut self, job: Job, dist2: u64) -> bool {
        debug_assert!(matches!(priority(&job), Priority::Far), "push_far on a near job");
        if self.far.len() >= FAR_QUEUE_CAP {
            return false;
        }
        self.far.push(job, dist2);
        true
    }

    /// The next job to run: the near job closest to the LIVE view centre
    /// (FIFO by seq on ties, and the whole class before any far job), then
    /// the nearest far job. Near entries whose region left the view are
    /// drained into `cancelled` — the caller reports each as
    /// [`Done::Cancelled`] so its claim is released instead of stranded.
    ///
    /// The linear scan re-keys every entry against the CURRENT centre, which
    /// is what makes fast movement re-prioritize the backlog for free; the
    /// near queue is bounded by the admission budgets, so the scan stays tiny
    /// next to the job that follows it.
    fn pop(&mut self, gate: &ViewGate, cancelled: &mut Vec<JobKey>) -> Option<Job> {
        // Deschedule first, then select — two passes so the winning index
        // can't be invalidated by a removal.
        self.near.retain(|(_, job)| match job.col() {
            Some((cx, cz)) if !gate.wanted(cx, cz) => {
                cancelled.push(JobKey::of(job));
                false
            }
            _ => true,
        });
        let best = self
            .near
            .iter()
            .enumerate()
            .min_by_key(|(_, (seq, job))| {
                let d = job.col().map_or(0, |(cx, cz)| gate.dist(cx, cz));
                (d, *seq)
            })
            .map(|(i, _)| i);
        match best {
            Some(i) => Some(self.near.swap_remove(i).1),
            None => self.far.pop_nearest(),
        }
    }
}

/// The worker pool. Owned by the `World` and spawned lazily on the first
/// `stream()`, so headless worlds (dedicated server, tests) never start threads.
pub struct Workers {
    /// The shared job queue + its wait condition; `Drop` sets `closed` and wakes
    /// every worker to join.
    gate: Arc<(Mutex<JobQueue>, Condvar)>,
    /// The live view snapshot the queue re-prioritizes and descheduled against.
    view: Arc<ViewGate>,
    results: Receiver<Done>,
    handles: Vec<JoinHandle<()>>,
}

impl Workers {
    /// The pool size for this machine: leave a core for the render thread,
    /// never more than 3 (chunk work is bursty, not sustained), at least 1.
    pub fn default_threads() -> usize {
        let cores = thread::available_parallelism().map_or(1, |n| n.get());
        cores.saturating_sub(1).clamp(1, 3)
    }

    /// Spawn `threads` workers (at least 1) sharing one job queue.
    pub fn spawn(threads: usize) -> Self {
        let (done, results) = mpsc::channel::<Done>();
        let gate = Arc::new((Mutex::new(JobQueue::default()), Condvar::new()));
        let view = Arc::new(ViewGate::new());
        let handles = (0..threads.max(1))
            .map(|_| {
                let gate = Arc::clone(&gate);
                let view = Arc::clone(&view);
                let done = done.clone();
                thread::spawn(move || worker_loop(&gate, &view, &done))
            })
            .collect();
        Self {
            gate,
            view,
            results,
            handles,
        }
    }

    /// Publish the live streaming centre and horizontal radius (chunks). The
    /// queue re-prioritizes near work against it at every pop and descheduled
    /// left-behind entries — the fast-movement fix.
    pub(in crate::world) fn set_view(&self, cx: i32, cz: i32, radius: i32) {
        self.view.set(cx, cz, radius);
    }

    /// Queue a job at its scheduling class; returns whether it was accepted.
    /// `false` only once the pool is shutting down (`closed`), so the caller
    /// must not mark the coord in flight and the normal scans simply retry it.
    pub(in crate::world) fn submit(&self, job: Job) -> bool {
        let (lock, cvar) = &*self.gate;
        let mut queue = lock.lock().unwrap_or_else(|p| p.into_inner());
        if queue.closed {
            return false;
        }
        queue.push(job);
        drop(queue);
        cvar.notify_one();
        true
    }

    /// Queue a far LOD job keyed by `dist2` (squared metres to the player).
    /// Unlike [`submit`](Self::submit), the far class is distance-ordered, so
    /// the nearest LOD work drains first. Returns whether it was admitted:
    /// `false` when the pool is shutting down OR the far queue is at
    /// [`FAR_QUEUE_CAP`] — the caller must NOT claim a rejected key (accepted ⇒
    /// claimed ⇒ owed exactly one `Done`), it just retries on a later frame.
    #[must_use]
    pub(in crate::world) fn submit_far(&self, job: Job, dist2: u64) -> bool {
        let (lock, cvar) = &*self.gate;
        let mut queue = lock.lock().unwrap_or_else(|p| p.into_inner());
        if queue.closed {
            return false;
        }
        let admitted = queue.push_far(job, dist2);
        drop(queue);
        if admitted {
            cvar.notify_one();
        }
        admitted
    }

    /// Non-blocking poll for one finished result.
    pub(in crate::world) fn try_recv(&self) -> Option<Done> {
        self.results.try_recv().ok()
    }

    /// Drop queued far-section work and return every exact claim. Callers
    /// either cancel those claims in-place (camera discontinuity) or retire the
    /// whole section lane (configuration change). A worker already executing a
    /// job is unaffected and remains protected by epoch/token validation.
    pub(in crate::world) fn clear_far(&self) -> Vec<JobKey> {
        let (lock, _) = &*self.gate;
        lock.lock()
            .unwrap_or_else(|p| p.into_inner())
            .clear_far()
    }
}

impl Drop for Workers {
    /// Flag `closed` and wake every worker first, then join: each worker is
    /// either waiting on the condvar (returns at once) or finishing one job, so
    /// the join is bounded and GPU-independent.
    fn drop(&mut self) {
        let (lock, cvar) = &*self.gate;
        lock.lock().unwrap_or_else(|p| p.into_inner()).closed = true;
        cvar.notify_all();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

/// The profiler meter for a job kind. Workers run off the main thread, but
/// [`voxel_engine::profile`] is an atomic global, so they feed the same unified
/// report as the CPU and GPU tiers.
fn job_meter(job: &Job) -> voxel_engine::profile::Meter {
    use voxel_engine::profile::Meter;
    match job {
        Job::GenerateColumn { .. } => Meter::WorkGenerate,
        Job::Mesh { .. } => Meter::WorkMesh,
        Job::Light { .. } => Meter::WorkLight,
        // Reuse WorkTile meter: new variant would touch profile.rs (outside scope).
        Job::Section { .. } => Meter::WorkTile,
        #[cfg(test)]
        Job::Panic(_) => Meter::WorkMesh,
    }
}

fn worker_loop(gate: &(Mutex<JobQueue>, Condvar), view: &ViewGate, done: &Sender<Done>) {
    let (lock, cvar) = gate;
    loop {
        // Lock only around the dequeue; the job itself runs unlocked. Poisoned
        // mutexes (a sibling panicked) still yield a usable queue.
        let mut cancelled: Vec<JobKey> = Vec::new();
        let job = {
            let mut queue = lock.lock().unwrap_or_else(|p| p.into_inner());
            loop {
                let job = queue.pop(view, &mut cancelled);
                if job.is_some() || !cancelled.is_empty() {
                    break job;
                }
                if queue.closed {
                    return; // pool shutting down and drained
                }
                queue = cvar.wait(queue).unwrap_or_else(|p| p.into_inner());
            }
        };
        // Report descheduled entries so their claims release; then run the
        // popped job (if the pop found only cancellations, just loop back).
        for key in cancelled {
            if done.send(Done::Cancelled(Box::new(key))).is_err() {
                return;
            }
        }
        let Some(job) = job else { continue };
        // Headline runs keep profiling disabled. Avoid label allocation and
        // worker clock reads in that mode; the claim key identifies a panic.
        let profile_start = voxel_engine::profile::is_enabled()
            .then(|| (job_meter(&job), std::time::Instant::now()));
        let key = JobKey::of(&job);
        // Guard the job body: a panic in `run` (bad generator sample, light/mesh
        // index, edit replay) used to unwind straight out of `worker_loop` and
        // KILL this thread — as workers died one by one, the whole pool went
        // silent and every streaming counter froze (the entry STALL). Catching it
        // keeps the thread alive, and `Done::Failed` hands the job's claim back
        // to the main thread so `fail_job` can release it and retry/quarantine —
        // a claimed key is owed exactly one `Done`, panic or not. The label
        // names the culprit so it stops being invisible.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(job)));
        if let Some((meter, start)) = profile_start {
            voxel_engine::profile::add(meter, start.elapsed());
        }
        let produced = match result {
            Ok(produced) => produced,
            Err(_) => {
                eprintln!("worker: job PANICKED (thread survives, claim released): {key:?}");
                Done::Failed(Box::new(key))
            }
        };
        if done.send(produced).is_err() {
            return; // result channel closed mid-shutdown: stop early
        }
    }
}

/// Pure CPU on owned data (same code as sync paths for determinism).
fn run(job: Job) -> Done {
    match job {
        Job::GenerateColumn {
            col,
            cy,
            generator,
            edits,
        } => {
            let (cx, cz) = col;
            // Share the column profile across the whole run, then replay each
            // chunk's edit overlay — voxel-identical to per-chunk generation.
            let chunks = generator
                .generate_column(cx, cz, cy)
                .into_iter()
                .map(|(cyy, data)| {
                    let coord = Coord::new(cx, cyy, cz);
                    let mut chunk = Chunk::from_data(cx, cyy, cz, data);
                    if let Some((_, cells)) = edits.iter().find(|(c, _)| *c == coord) {
                        for &(index, id) in cells {
                            chunk.set_index(index, id);
                        }
                    }
                    (coord, chunk)
                })
                .collect();
            Done::Column { col, chunks }
        }
        Job::Mesh {
            coord,
            rev,
            snapshot,
        } => {
            // Pure meshing: light was settled on the main thread and travels in
            // the snapshot as a ready shell, so the worker only greedy-meshes.
            let mut data = MeshOutput::new();
            match snapshot.light.as_ref() {
                Some(light) => mesh::build_chunk_mesh(
                    &snapshot.padded,
                    snapshot.uniform,
                    &snapshot.tables,
                    light,
                    &mut data,
                ),
                None => mesh::build_chunk_mesh_unlit(
                    &snapshot.padded,
                    snapshot.uniform,
                    &snapshot.tables,
                    &mut data,
                ),
            }
            Done::Mesh { coord, rev, data }
        }
        Job::Light { coord, epoch, snapshot } => {
            // Pure flood: same `propagate` the sync path called, now on an owned
            // neighbourhood snapshot instead of live neighbour grids.
            let mut grid = LightGrid::dark();
            light::propagate(
                &snapshot.chunk,
                &snapshot.shell,
                &snapshot.ceiling,
                snapshot.world_y0,
                &snapshot.tables,
                &mut grid,
            );
            Done::Light { coord, epoch, grid }
        }
        Job::Section {
            pos,
            epoch,
            token,
            generator,
            edits,
            tables,
        } => {
            // Pure CPU on owned data. Extraction writes straight into the
            // dense meshing scratch instead of allocating 1024 temporary RLE
            // columns and expanding them immediately afterward.
            let meshes = section::extract_section_mesh(pos, generator.as_ref(), &edits, &tables);
            Done::Section {
                pos,
                epoch,
                token,
                meshes,
            }
        }
        #[cfg(test)]
        Job::Panic(key) => panic!("injected worker panic for {key:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::{AIR, BlockRegistry};
    use std::time::Duration;
    use voxel_engine::Pass;

    /// Mirrors `World::new`'s generator construction.
    fn generator(seed: i64) -> Arc<SineHills> {
        Arc::new(SineHills::new(
            &mut BlockRegistry::with_builtins(),
            20.0,
            seed,
        ))
    }

    /// Create a far section job tagged by id for scheduler tests.
    fn section_job(terrain: &Arc<SineHills>, id: i32) -> Job {
        Job::Section {
            pos: SectionPos { detail: 2, x: id, z: 0 },
            epoch: 0,
            token: id as u64,
            generator: Arc::clone(terrain),
            edits: Vec::new(),
            tables: Arc::new(BlockRegistry::with_builtins().hot_tables()),
        }
    }
    /// The `x` id a `section_job` was tagged with.
    fn section_id(j: &Job) -> i32 {
        match j {
            Job::Section { pos, .. } => pos.x,
            _ => panic!("expected a Section job"),
        }
    }

    #[test]
    fn terrain_jobs_share_one_generator_allocation() {
        let terrain = generator(17);
        let column = Job::GenerateColumn {
            col: (0, 0),
            cy: 0..=0,
            generator: Arc::clone(&terrain),
            edits: Vec::new(),
        };
        let section = section_job(&terrain, 0);

        let Job::GenerateColumn {
            generator: column_generator,
            ..
        } = &column
        else {
            unreachable!()
        };
        let Job::Section {
            generator: section_generator,
            ..
        } = &section
        else {
            unreachable!()
        };
        assert!(Arc::ptr_eq(&terrain, column_generator));
        assert!(Arc::ptr_eq(&terrain, section_generator));
        assert_eq!(Arc::strong_count(&terrain), 3);
    }

    #[test]
    fn worker_generation_matches_the_sync_path() {
        let generator = generator(42);
        let coord = Coord::new(3, 1, -2); // a ground chunk: y 16..=31 crosses the surface
        let edits = vec![
            (Chunk::index(1, 3, 2), AIR),         // dig a hole
            (Chunk::index(5, 14, 5), BlockId(1)), // place high in the chunk
        ];
        let mut expected = Chunk::new(coord.x, coord.y, coord.z, generator.as_ref());
        for &(index, id) in &edits {
            expected.set_index(index, id);
        }

        let workers = Workers::spawn(2);
        assert!(workers.submit(Job::GenerateColumn {
            col: (coord.x, coord.z),
            cy: coord.y..=coord.y,
            generator: Arc::clone(&generator),
            edits: vec![(coord, edits)],
        }));
        let done = workers
            .results
            .recv_timeout(Duration::from_secs(10))
            .expect("worker finished");
        let Done::Column { col, chunks } = done else {
            panic!("expected a column result");
        };
        assert_eq!(col, (coord.x, coord.z));
        let chunk = &chunks.iter().find(|(c, _)| *c == coord).expect("coord in column").1;
        assert_eq!(chunk.data(), expected.data(), "voxel-identical to sync");
    }

    #[test]
    fn worker_meshing_matches_the_sync_mesher() {
        let mut registry = BlockRegistry::with_builtins();
        let generator = SineHills::new(&mut registry, 20.0, 5);
        // The chunk holding the surface at the origin, with all six neighbours
        // (below: solid ground, above: sky, sides: more surface).
        let chunk = Chunk::new(0, 1, 0, &generator);
        // The 3x3x3 neighbourhood around it (only the 6 face-neighbours are
        // interesting terrain here; the rest read as air, which is fine).
        let neighbourhood = |dx: i32, dy: i32, dz: i32| Chunk::new(dx, 1 + dy, dz, &generator);
        let neigh: Vec<Chunk> = (0..27)
            .map(|k| neighbourhood(k % 3 - 1, k / 9 - 1, k / 3 % 3 - 1))
            .collect();
        let at = |dx: i32, dy: i32, dz: i32| -> Option<&Chunk> {
            Some(&neigh[((dx + 1) + (dz + 1) * 3 + (dy + 1) * 9) as usize])
        };
        let tables = Arc::new(registry.hot_tables());
        let padded = Padded::capture(at);
        let light = PaddedLight::full();

        let mut expected = new_chunk_mesh_data();
        mesh::build_chunk_mesh(&padded, None, &tables, &light, &mut expected);
        // The neighbours must actually matter, or equality proves nothing.
        let mut unculled = new_chunk_mesh_data();
        mesh::build_chunk_mesh(&Padded::capture(|dx, dy, dz| (dx == 0 && dy == 0 && dz == 0).then_some(&chunk)), None, &tables, &light, &mut unculled);
        let index_count =
            |d: &ChunkMeshData| d[Pass::Opaque].buckets().iter().map(|b| b.len()).sum::<usize>();
        assert_ne!(index_count(&unculled), index_count(&expected), "border culling engaged");

        let snapshot = ChunkSnapshot {
            padded: Padded::capture(at),
            uniform: chunk.uniform(),
            light: Some(PaddedLight::full()),
            tables: Arc::clone(&tables),
        };
        let workers = Workers::spawn(1);
        assert!(workers.submit(Job::Mesh { coord: Coord::new(0, 1, 0), rev: 7, snapshot }));
        let done = workers
            .results
            .recv_timeout(Duration::from_secs(10))
            .expect("worker finished");
        let Done::Mesh { coord, rev, data } = done else {
            panic!("expected a mesh result");
        };
        assert_eq!((coord, rev), (Coord::new(0, 1, 0), 7));
        for p in Pass::ALL {
            assert_eq!(data[p].buckets(), expected[p].buckets());
            assert_eq!(data[p].vertices(), expected[p].vertices(), "worker mesh matches sync");
        }
    }

    /// A permissive gate (no view published yet): near keeps FIFO order.
    fn open_gate() -> ViewGate {
        ViewGate::new()
    }

    /// `pop` with cancellation plumbing asserted empty — for tests where no
    /// descheduling is expected.
    fn pop_clean(q: &mut JobQueue, gate: &ViewGate) -> Option<Job> {
        let mut cancelled = Vec::new();
        let job = q.pop(gate, &mut cancelled);
        assert!(cancelled.is_empty(), "unexpected descheduling: {cancelled:?}");
        job
    }

    #[test]
    fn near_jobs_dequeue_before_far_regardless_of_insertion_order() {
        let terrain = generator(0);
        let near = |c: i32| Job::GenerateColumn {
            col: (c, c),
            cy: 0..=0,
            generator: Arc::clone(&terrain),
            edits: Vec::new(),
        };
        let far = |c: i32| section_job(&terrain, c);

        // Interleave far/near so a FIFO alone would not reproduce the order.
        let mut q = JobQueue::default();
        let gate = open_gate();
        q.push(far(0));
        q.push(near(0));
        q.push(far(1));
        q.push(near(1));

        // All near first (FIFO within class), then all far (FIFO within class).
        assert!(matches!(pop_clean(&mut q, &gate), Some(Job::GenerateColumn { col: (0, 0), .. })));
        assert!(matches!(pop_clean(&mut q, &gate), Some(Job::GenerateColumn { col: (1, 1), .. })));
        assert_eq!(section_id(&pop_clean(&mut q, &gate).unwrap()), 0);
        assert_eq!(section_id(&pop_clean(&mut q, &gate).unwrap()), 1);
        assert!(pop_clean(&mut q, &gate).is_none());
    }

    /// The fast-movement fix: near pops re-key against the LIVE centre (the
    /// closest chunk to the player NOW runs first, whatever the enqueue
    /// order), and entries left outside the cancel ring are descheduled with
    /// their claims reported instead of silently ground through.
    #[test]
    fn near_queue_reprioritizes_live_and_deschedules_left_behind_work() {
        let terrain = generator(0);
        let near = |cx: i32, cz: i32| Job::GenerateColumn {
            col: (cx, cz),
            cy: 0..=0,
            generator: Arc::clone(&terrain),
            edits: Vec::new(),
        };

        let mut q = JobQueue::default();
        let gate = open_gate();
        q.push(near(26, 26)); // enqueued first, but no longer the closest
        q.push(near(0, 1)); // right next to the ORIGINAL centre
        q.push(near(6, 6)); // a few chunks out from the original centre
        q.push(near(28, 29)); // right next to where the player ends up

        // The player sprints to (28, 28) with radius 3: priorities flip, and
        // everything left more than radius + CANCEL_MARGIN chunks behind is
        // descheduled with its claim reported.
        gate.set(28, 28, 3);
        let mut cancelled = Vec::new();
        let first = q.pop(&gate, &mut cancelled).expect("work remains");
        assert!(
            matches!(first, Job::GenerateColumn { col: (28, 29), .. }),
            "the job nearest the LIVE centre must pop first, not the oldest"
        );
        assert_eq!(
            cancelled,
            vec![
                JobKey::Column { col: (0, 1), cy: 0..=0 },
                JobKey::Column { col: (6, 6), cy: 0..=0 },
            ],
            "left-behind work is descheduled with its claims"
        );

        // The surviving (26, 26) — inside the ring at distance 2 — runs next.
        let mut cancelled = Vec::new();
        let second = q.pop(&gate, &mut cancelled).expect("one survivor");
        assert!(matches!(second, Job::GenerateColumn { col: (26, 26), .. }));
        assert!(cancelled.is_empty());
        assert!(q.pop(&gate, &mut cancelled).is_none(), "queue drained");
    }

    #[test]
    fn far_queue_contract() {
        // Tag each far job with a distinct column id so pops are identifiable.
        let terrain = generator(0);
        let job = |id: i32| section_job(&terrain, id);
        let id_of = section_id;

        // push dist2 {9, 1, 4, 1}: pops must see 1(first-pushed), 1, 4, 9.
        let mut q = FarQueue::default();
        q.push(job(0), 9); // seq 0
        q.push(job(1), 1); // seq 1 — first-pushed of the two dist2 = 1
        q.push(job(2), 4); // seq 2
        q.push(job(3), 1); // seq 3
        assert_eq!(id_of(&q.pop_nearest().unwrap()), 1, "nearest, first-pushed tie");
        assert_eq!(id_of(&q.pop_nearest().unwrap()), 3, "nearest, second tie (FIFO)");
        assert_eq!(id_of(&q.pop_nearest().unwrap()), 2, "dist2 = 4 next");
        assert_eq!(id_of(&q.pop_nearest().unwrap()), 0, "dist2 = 9 last");
        assert!(q.pop_nearest().is_none(), "drained");

        // At the cap, admission REJECTS (never evicts an accepted job: accepted
        // ⇒ claimed ⇒ owed a Done); everything already admitted survives.
        let mut q = JobQueue::default();
        for i in 0..FAR_QUEUE_CAP {
            assert!(q.push_far(job(i as i32), i as u64), "under the cap admits");
        }
        assert!(!q.push_far(job(-1), 0), "at the cap rejects — even a nearer job");
        assert_eq!(q.far.len(), FAR_QUEUE_CAP, "rejection leaves the queue intact");
        // Popping frees a slot, so the next submit admits again (lane retry).
        assert_eq!(id_of(&pop_clean(&mut q, &open_gate()).unwrap()), 0, "nearest still pops first");
        assert!(q.push_far(job(-1), 0), "below the cap admits again");
    }

    #[test]
    fn clearing_far_returns_exact_claims_and_preserves_near_work() {
        let terrain = generator(0);
        let mut q = JobQueue::default();
        q.push(section_job(&terrain, 3));
        q.push(Job::GenerateColumn {
            col: (7, 8),
            cy: 1..=2,
            generator: Arc::clone(&terrain),
            edits: Vec::new(),
        });
        q.push(section_job(&terrain, -4));

        let claims = q.clear_far();
        assert_eq!(
            claims,
            vec![
                JobKey::Section {
                    pos: SectionPos { detail: 2, x: 3, z: 0 },
                    epoch: 0,
                    token: 3,
                },
                JobKey::Section {
                    pos: SectionPos { detail: 2, x: -4, z: 0 },
                    epoch: 0,
                    token: (-4i32) as u64,
                },
            ]
        );
        assert!(matches!(
            pop_clean(&mut q, &open_gate()),
            Some(Job::GenerateColumn { col: (7, 8), .. })
        ));
        assert!(pop_clean(&mut q, &open_gate()).is_none());
    }

    /// Worker→main channel throughput (structural-opportunities #8): floods the
    /// pool with real mesh jobs and reports jobs/second plus `size_of::<Done>()`.
    /// Ignored: a timing benchmark, not a correctness gate. Run with
    /// `cargo test --release mesh_result_channel_throughput -- --ignored --nocapture`.
    /// 2026-07-13 (RTX 3070 box, 4 workers): 544 B inline ≈ 28.6k jobs/s;
    /// boxed 112 B ≈ 29.2k jobs/s — throughput is meshing-bound, the boxing is
    /// a payload/regression guard rather than a measured speedup.
    #[test]
    #[ignore]
    fn mesh_result_channel_throughput() {
        let mut registry = BlockRegistry::with_builtins();
        let generator = SineHills::new(&mut registry, 20.0, 5);
        let neigh: Vec<Chunk> = (0..27)
            .map(|k| Chunk::new(k % 3 - 1, 1 + k / 9 - 1, k / 3 % 3 - 1, &generator))
            .collect();
        let at = |dx: i32, dy: i32, dz: i32| -> Option<&Chunk> {
            Some(&neigh[((dx + 1) + (dz + 1) * 3 + (dy + 1) * 9) as usize])
        };
        let tables = Arc::new(registry.hot_tables());
        let snapshot = || ChunkSnapshot {
            padded: Padded::capture(at),
            uniform: None,
            light: Some(PaddedLight::full()),
            tables: Arc::clone(&tables),
        };

        const JOBS: u32 = 4000;
        let workers = Workers::spawn(4);
        let start = std::time::Instant::now();
        for i in 0..JOBS {
            assert!(workers.submit(Job::Mesh {
                coord: Coord::new(i as i32, 1, 0),
                rev: 1,
                snapshot: snapshot(),
            }));
        }
        let mut got = 0;
        while got < JOBS {
            let done = workers.results.recv_timeout(Duration::from_secs(30)).expect("drained");
            assert!(matches!(done, Done::Mesh { .. }));
            got += 1;
        }
        let dt = start.elapsed();
        println!(
            "size_of::<Done>() = {} B; {} mesh jobs in {:.3}s = {:.0} jobs/s",
            std::mem::size_of::<Done>(),
            JOBS,
            dt.as_secs_f64(),
            JOBS as f64 / dt.as_secs_f64()
        );
    }

    #[test]
    fn panicking_jobs_report_failed_with_their_claim_and_leave_the_pool_alive() {
        let workers = Workers::spawn(1);
        let keys = [
            JobKey::Column { col: (3, -2), cy: 0..=2 },
            JobKey::Mesh { coord: Coord::new(1, 2, 3) },
            JobKey::Light { coord: Coord::new(-1, 0, 1) },
            JobKey::Section {
                pos: SectionPos { detail: 2, x: 5, z: -5 },
                epoch: 7,
                token: 11,
            },
        ];
        for key in keys.clone() {
            assert!(workers.submit(Job::Panic(Box::new(key))));
        }
        let mut got = Vec::new();
        for _ in 0..keys.len() {
            match workers.results.recv_timeout(Duration::from_secs(10)).expect("failure lands") {
                Done::Failed(k) => got.push(*k),
                _ => panic!("expected Done::Failed for an injected panic"),
            }
        }
        for key in &keys {
            assert!(got.contains(key), "missing failure for {key:?}");
        }

        // The single worker thread survived every panic: real work still runs.
        let terrain = generator(9);
        assert!(workers.submit(Job::GenerateColumn {
            col: (0, 0),
            cy: 0..=0,
            generator: terrain,
            edits: Vec::new(),
        }));
        let done = workers.results.recv_timeout(Duration::from_secs(10)).expect("pool alive");
        assert!(matches!(done, Done::Column { .. }));
    }

    #[test]
    fn spawn_then_drop_terminates_even_with_queued_jobs() {
        // Run the drop on a helper thread so a regression hangs this test's
        // timeout instead of the whole suite.
        let (finished, check) = mpsc::channel();
        let generator = generator(1);
        thread::spawn(move || {
            let workers = Workers::spawn(2);
            for i in 0..6 {
                workers.submit(Job::GenerateColumn {
                    col: (i, i),
                    cy: 0..=0,
                    generator: Arc::clone(&generator),
                    edits: Vec::new(),
                });
                // A far job too, so drop must drain/close both classes.
                workers.submit(section_job(&generator, i));
            }
            drop(workers); // flags closed, wakes workers, then joins — must be bounded
            let _ = finished.send(());
        });
        check
            .recv_timeout(Duration::from_secs(10))
            .expect("Workers::drop hung");
    }
}
