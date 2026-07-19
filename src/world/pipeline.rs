//! pipeline.rs runs chunk generation and meshing on background threads so the
//! render thread never pays for either — a tiny std-only worker pool (zero
//! dependencies) fed and drained by [`World::stream`](super::World::stream).
//!
//! Threading model:
//! - `min(3, cores - 1).max(1)` worker threads share ONE [`JobQueue`] behind a
//!   `Mutex` + `Condvar`. A worker holds the lock only while dequeuing (or
//!   waiting for work); every job runs unlocked.
//! - Jobs carry owned value data only (a generator clone, a voxel snapshot,
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
use std::ops::RangeInclusive;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::Coord;
use super::chunk::{CHUNK_SIZE, Chunk};
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
    /// The settled light shell (this chunk's grid + its neighbours'), sampled per
    /// vertex for smooth light across interior, border, and diagonal cells.
    /// `None` is the allocation-free constant-full-light path used when voxel
    /// lighting is disabled — the 18³ shell is never captured at all.
    pub light: Option<PaddedLight>,
    /// The hot tables (solid/opaque/emission), shared by refcount. Palette growth
    /// swaps the world's `Arc` for a new one while in-flight jobs keep the old —
    /// harmless because the palette is append-only and every result is
    /// re-validated on arrival anyway.
    pub tables: Arc<HotTables>,
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
    /// cz)` over the chunk-layer range `cy`, from one generator clone. The
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
    /// with no live snapshots or neighbor lookups. `epoch`/`token` are the
    /// asynchronous claim identity: a result from a retired ladder epoch, or
    /// from a claim replaced after unload/re-admission, must never capture the
    /// live section at the same position.
    Section {
        pos: SectionPos,
        epoch: u32,
        token: ClaimToken,
        generator: Arc<SineHills>,
        edits: Vec<(Coord, Vec<(usize, BlockId)>)>,
        tables: Arc<HotTables>,
    },
    /// Test-only: panics inside `run`, reporting the given claim — the injector
    /// for the worker-panic → `Done::Failed` → claim-release path.
    #[cfg(test)]
    Panic(Box<JobKey>),
}

/// The unique identity of one asynchronous section claim, minted by the
/// section lane's submit (a wrapping counter on `World`). A result,
/// cancellation, or failure presents its token back; anything that doesn't
/// match the live `Meshing` claim belongs to a superseded job and must not
/// touch the replacement. A plain `u64` would let any counter impersonate a
/// claim — the newtype confines minting to the one owner.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct ClaimToken(pub(in crate::world) u64);

/// The claim a job holds while in flight, extractable from the job itself.
/// A panicking job returns this in [`Done::Failed`] so the main thread can
/// release the EXACT claim instead of leaving it stranded forever.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum JobKey {
    Column { col: (i32, i32), cy: RangeInclusive<i32> },
    Mesh { coord: Coord },
    Light { coord: Coord },
    Section { pos: SectionPos, epoch: u32, token: ClaimToken },
}

impl JobKey {
    /// The claim identity of a job, captured before the job runs.
    fn of(job: &Job) -> JobKey {
        match job {
            Job::GenerateColumn { col, cy, .. } => JobKey::Column { col: *col, cy: cy.clone() },
            Job::Mesh { coord, .. } => JobKey::Mesh { coord: *coord },
            Job::Light { coord, .. } => JobKey::Light { coord: *coord },
            Job::Section { pos, epoch, token, .. } => {
                JobKey::Section { pos: *pos, epoch: *epoch, token: *token }
            }
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
    /// Boxed: `ChunkMeshData` is ~530 B inline (three passes × Vec headers ×
    /// six index buckets), and it dominated the whole enum — every channel
    /// send/recv and match memcpy'd it. One box per mesh job is noise next to
    /// the meshing itself; the Box rides untouched into `upload_queue`.
    Mesh { coord: Coord, rev: u32, data: MeshOutput },
    Light { coord: Coord, epoch: u32, grid: LightGrid },
    Section { pos: SectionPos, epoch: u32, token: ClaimToken, meshes: [SectionMeshData; 4] },
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

/// Cross-thread pool for greedy-mesh output. Geometry vectors retain their
/// capacities after upload/stale rejection, so traversal refills them without
/// allocating vertex and index buckets from scratch for every job.
// The box is intentional: besides being recycled with the geometry, it keeps
// `Done::Mesh` pointer-sized instead of inflating every result-channel message.
#[allow(clippy::vec_box)]
static MESH_OUTPUT_POOL: Mutex<Vec<Box<ChunkMeshData>>> = Mutex::new(Vec::new());
const MESH_OUTPUT_POOL_CAP: usize = 32;

/// A pooled `Box<ChunkMeshData>`: taken from [`MESH_OUTPUT_POOL`] at job start
/// (workers), returned on drop wherever the result dies (upload or stale
/// rejection on the main thread) — completing the cross-thread round trip.
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

impl std::ops::Deref for MeshOutput {
    type Target = ChunkMeshData;

    fn deref(&self) -> &Self::Target {
        self.0.as_deref().expect("live pooled mesh output")
    }
}

impl std::ops::DerefMut for MeshOutput {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_deref_mut().expect("live pooled mesh output")
    }
}

impl Drop for MeshOutput {
    fn drop(&mut self) {
        let Some(data) = self.0.take() else { return };
        let mut pool =
            MESH_OUTPUT_POOL.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if pool.len() < MESH_OUTPUT_POOL_CAP {
            pool.push(data);
        }
    }
}

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

/// The live view, shared with the worker pool. The world stores the streaming
/// centre/radius (and the far-field horizon) here every stream pass; the
/// queues then re-key their backlogs toward where the player is NOW and
/// deschedule entries left behind — once per CHANGE ([`ViewGate::epoch`]),
/// not per pop. Fast movement therefore reorders the backlog and sheds it
/// instead of grinding through stale regions.
///
/// Relaxed atomics: centre and radius may briefly disagree mid-update;
/// [`CANCEL_MARGIN`] absorbs the tear (it can only mis-order or briefly spare
/// a job, never cancel wanted work — the margin exceeds any one-frame move).
pub(in crate::world) struct ViewGate {
    /// `(cx as u32) << 32 | (cz as u32)`.
    center: AtomicU64,
    /// Horizontal view radius in chunks; `i32::MAX` (permissive) until set.
    radius: AtomicI32,
    /// Monotone stamp of the `(centre, radius)` pair: bumped only when one
    /// actually changes, so the queues' O(n) re-key/deschedule rebuild runs
    /// once per boundary cross instead of once per pop.
    epoch: AtomicU64,
    /// Far-field descheduling horizon in METRES (`f64` bits; +∞ until set):
    /// the outer ladder radius plus the velocity lookahead, refreshed every
    /// stream pass. Deliberately NOT folded into `epoch` — it wobbles with
    /// velocity every pass, and the wanted checks read it LIVE rather than
    /// baking it into keys.
    far_m: AtomicU64,
}

impl ViewGate {
    fn new() -> Self {
        Self {
            center: AtomicU64::new(0),
            radius: AtomicI32::new(i32::MAX),
            epoch: AtomicU64::new(0),
            far_m: AtomicU64::new(f64::INFINITY.to_bits()),
        }
    }

    fn set(&self, cx: i32, cz: i32, radius: i32) {
        let packed = ((cx as u32 as u64) << 32) | (cz as u32 as u64);
        let prev_center = self.center.swap(packed, Ordering::Relaxed);
        let prev_radius = self.radius.swap(radius, Ordering::Relaxed);
        if prev_center != packed || prev_radius != radius {
            self.epoch.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Publish the far-field horizon (metres from the eye).
    fn set_far(&self, metres: f64) {
        self.far_m.store(metres.to_bits(), Ordering::Relaxed);
    }

    fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Relaxed)
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

    /// The eye position in metres — the centre chunk's centre, matching
    /// [`player_dist2`](super::player_dist2)'s convention.
    fn eye_m(&self) -> (f64, f64) {
        let (cx, cz) = self.center();
        let s = CHUNK_SIZE as f64;
        (cx as f64 * s + s / 2.0, cz as f64 * s + s / 2.0)
    }

    /// Live squared horizontal distance (m²) from the eye to a world point —
    /// the far class's re-key metric (2-D, like its admission metric).
    fn far_dist2_m(&self, wx: i64, wz: i64) -> u64 {
        let (ex, ez) = self.eye_m();
        let (dx, dz) = (wx as f64 - ex, wz as f64 - ez);
        (dx * dx + dz * dz) as u64
    }

    /// Whether a far entry with world-centre `(wx, wz)` and footprint `span`
    /// is still inside the live horizon. The entry's own span is the
    /// hysteresis margin — sections are large, so the chunk-sized
    /// [`CANCEL_MARGIN`] would be meaningless here. Permissive until both a
    /// view and a horizon have been published.
    fn far_wanted(&self, wx: i64, wz: i64, span: i64) -> bool {
        let far = f64::from_bits(self.far_m.load(Ordering::Relaxed));
        if !far.is_finite() || self.radius.load(Ordering::Relaxed) == i32::MAX {
            return true;
        }
        let limit = far + span as f64;
        (self.far_dist2_m(wx, wz) as f64) <= limit * limit
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

/// Per-frame budget for the *apply* half of the light settle work — the
/// `drain_results` loop that folds settled grids into chunks (`DrainLane`'s
/// internal window). The *admit* half's budget now lives in the `light_admit`
/// producer's manifest (one budget locus per lane); this remains its own window
/// because it is a distinct loop in a distinct pass. 2 ms: each apply is cheap
/// bookkeeping, and a deep apply queue HOLDS light claims — chunks read as
/// unsettled, mesh degraded, and remesh again — so draining it fast is worth
/// twice the old 1 ms window (the flood peaks near 1.8k queued grids).
pub const LIGHT_APPLY_BUDGET: Duration = Duration::from_millis(2);

// Each admission producer mints a FRESH `Deadline::from_budget(...)` from its
// scheduler-provided budget at the instant its `run()` starts — never one
// frame-start snapshot shared across lanes. The lanes run sequentially
// (drain → light → mesh → LOD), so a single anchored instant would leave every
// lane after the first ~1 ms pre-expired and admitting nothing (world-entry
// starvation). Budgets are admission caps, so idle lanes still return immediately.

/// Far-queue cap. At the cap [`Workers::submit_far`] REJECTS
/// the submit (returns `false`) and the lane simply does not claim the key, so
/// it retries naturally on a later frame — retry-not-drop lives at the
/// requester. Rejection at admission, never eviction after acceptance: an
/// accepted far job has already been claimed by its lane (`Meshing` state), and
/// a claimed key is owed exactly one `Done` — evicting it would strand the claim
/// forever (a permanent hole + an `entry_complete` hang).
pub const FAR_QUEUE_CAP: usize = 256;

/// One queued job with its scheduling key. `d` is the class metric — near:
/// live chess distance in chunks; far: squared metres (admission-keyed,
/// re-keyed live on epoch changes). `(wx, wz)` is the entry's location for
/// re-keying — the chunk column for near jobs, the world-space section centre
/// for far — and `span` the far footprint (the descheduling margin; 0 near).
struct Keyed {
    d: u64,
    seq: u64,
    wx: i64,
    wz: i64,
    span: i64,
    job: Job,
}

// Ordered by (d, seq) only — the FIFO tie-break on equal distance. The job
// payload never participates.
impl PartialEq for Keyed {
    fn eq(&self, other: &Self) -> bool {
        (self.d, self.seq) == (other.d, other.seq)
    }
}
impl Eq for Keyed {}
impl PartialOrd for Keyed {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Keyed {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.d, self.seq).cmp(&(other.d, other.seq))
    }
}

/// A nearest-first queue whose keys go stale as the player moves. The whole
/// backlog is SYNCED — every entry re-keyed against the live view, the
/// left-behind descheduled into `cancelled` — at most once per [`ViewGate`]
/// epoch (a real centre/radius change), then pops are plain O(log n) heap
/// pops. The old representation paid an O(n) retain + full re-key min-scan
/// under the queue mutex on EVERY pop, which is the lock-hold hazard a
/// 10-worker pool multiplies.
#[derive(Default)]
struct EpochHeap {
    heap: std::collections::BinaryHeap<std::cmp::Reverse<Keyed>>,
    keyed_at: u64,
    next_seq: u64,
}

impl EpochHeap {
    fn push(&mut self, d: u64, wx: i64, wz: i64, span: i64, job: Job) {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.heap.push(std::cmp::Reverse(Keyed { d, seq, wx, wz, span, job }));
    }

    /// Re-key every entry and deschedule the unwanted, once per epoch.
    fn sync(
        &mut self,
        epoch: u64,
        key: impl Fn(&Keyed) -> u64,
        wanted: impl Fn(&Keyed) -> bool,
        cancelled: &mut Vec<JobKey>,
    ) {
        if self.keyed_at == epoch {
            return;
        }
        self.keyed_at = epoch;
        if self.heap.is_empty() {
            return;
        }
        let mut kept = Vec::with_capacity(self.heap.len());
        for std::cmp::Reverse(mut entry) in std::mem::take(&mut self.heap).into_vec() {
            if wanted(&entry) {
                entry.d = key(&entry);
                kept.push(std::cmp::Reverse(entry));
            } else {
                cancelled.push(JobKey::of(&entry.job));
            }
        }
        self.heap = std::collections::BinaryHeap::from(kept); // O(n) heapify
    }

    fn pop(&mut self) -> Option<Job> {
        self.heap.pop().map(|std::cmp::Reverse(k)| k.job)
    }

    fn len(&self) -> usize {
        self.heap.len()
    }

    /// Drop every queued entry, returning the exact claims so the caller can
    /// release them (a claimed key is owed exactly one resolution).
    fn drain_claims(&mut self) -> Vec<JobKey> {
        std::mem::take(&mut self.heap)
            .into_vec()
            .into_iter()
            .map(|std::cmp::Reverse(k)| JobKey::of(&k.job))
            .collect()
    }
}

/// A far job's world-space centre and footprint span (metres), for live
/// re-keying and horizon descheduling without any callback into the `World`.
fn far_center_span(job: &Job) -> (i64, i64, i64) {
    match job {
        Job::Section { pos, .. } => {
            let span = pos.span() as i64;
            (pos.min_x() as i64 + span / 2, pos.min_z() as i64 + span / 2, span)
        }
        _ => (0, 0, 0),
    }
}

/// Two-class queue shared by the pool. `pop` drains `near` fully before `far`,
/// so far LOD jobs fill idle workers without ever starving the chunk under the
/// player. Both classes live in [`EpochHeap`]s: near keys by live chess
/// distance, far by admission dist² re-keyed (and horizon-descheduled) on
/// every view change — sustained fast flight sheds far work it has left
/// behind instead of grinding it (the old far class re-keyed NEVER: only the
/// >512 m/s teleport purge touched it). `closed` is the shutdown flag a
/// blocked `pop` wakes on.
#[derive(Default)]
struct JobQueue {
    near: EpochHeap,
    far: EpochHeap,
    closed: bool,
}

impl JobQueue {
    /// Push at the job's scheduling class. A far job pushed here (the legacy
    /// [`Workers::submit`] path and headless tests) carries no distance, so it
    /// sorts at `d = 0` and equal-distance far jobs fall back to FIFO by `seq`.
    /// This legacy path is uncapped (its only producers are tests); the
    /// streaming lanes go through the cap-checked [`Workers::submit_far`].
    fn push(&mut self, job: Job, gate: &ViewGate) {
        match priority(&job) {
            Priority::Near => {
                let (cx, cz) = job.col().unwrap_or((0, 0));
                let d = gate.dist(cx, cz) as u64;
                self.near.push(d, cx as i64, cz as i64, 0, job);
            }
            Priority::Far => {
                let (wx, wz, span) = far_center_span(&job);
                self.far.push(0, wx, wz, span, job);
            }
        }
    }

    fn clear_far(&mut self) -> Vec<JobKey> {
        self.far.drain_claims()
    }

    /// Admit a far job keyed by `dist2` (squared metres, motion-biased at the
    /// lane), or REJECT it at [`FAR_QUEUE_CAP`] (returns whether it was
    /// admitted). Rejection is the whole cap mechanism: the lane never claims
    /// a rejected key, so it retries on a later frame.
    #[must_use]
    fn push_far(&mut self, job: Job, dist2: u64) -> bool {
        debug_assert!(matches!(priority(&job), Priority::Far), "push_far on a near job");
        if self.far.len() >= FAR_QUEUE_CAP {
            return false;
        }
        let (wx, wz, span) = far_center_span(&job);
        self.far.push(dist2, wx, wz, span, job);
        true
    }

    /// The next job to run: the near job closest to the LIVE view centre
    /// (FIFO by seq on ties, and the whole class before any far job), then
    /// the nearest far job. On a view change (and only then), both heaps
    /// re-key and drain their left-behind entries into `cancelled` — the
    /// caller reports each as [`Done::Cancelled`] so its claim is released
    /// instead of stranded.
    fn pop(&mut self, gate: &ViewGate, cancelled: &mut Vec<JobKey>) -> Option<Job> {
        let epoch = gate.epoch();
        self.near.sync(
            epoch,
            |e| gate.dist(e.wx as i32, e.wz as i32) as u64,
            |e| gate.wanted(e.wx as i32, e.wz as i32),
            cancelled,
        );
        // Far syncs on the same trigger even while near work exists: a flood
        // keeps workers in the near class for a long time, and stale far
        // claims must release promptly, not once the near backlog drains.
        self.far.sync(
            epoch,
            |e| gate.far_dist2_m(e.wx, e.wz),
            |e| gate.far_wanted(e.wx, e.wz, e.span),
            cancelled,
        );
        self.near.pop().or_else(|| self.far.pop())
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
    /// The pool size for `cores` logical CPUs: reserve two (the main/render
    /// thread plus OS/audio/driver headroom), use the rest, cap at 12. The
    /// old `min(3)` cap starved loading on big machines — a fast-flight
    /// generate+light+mesh flood is sustained, not bursty, and three workers
    /// simply cannot keep up. Past ~12 the producers outrun the main-thread
    /// consumers instead; the backpressure there is the upload-queue
    /// admission gate, the bounded data box, and [`FAR_QUEUE_CAP`].
    /// Pure so the mapping is pinnable by test.
    fn threads_for(cores: usize) -> usize {
        cores.saturating_sub(2).clamp(1, 12)
    }

    /// [`threads_for`](Self::threads_for) at this machine's core count.
    pub fn default_threads() -> usize {
        Self::threads_for(thread::available_parallelism().map_or(1, |n| n.get()))
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

    /// Publish the live streaming centre, horizontal radius (chunks), and the
    /// far-field horizon (metres). The queues re-key their backlogs against it
    /// and deschedule left-behind entries — once per change, at the pool.
    pub(in crate::world) fn set_view(&self, cx: i32, cz: i32, radius: i32, far_m: f64) {
        self.view.set(cx, cz, radius);
        self.view.set_far(far_m);
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
        queue.push(job, &self.view);
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
    /// either cancel those claims in place (camera discontinuity) or retire the
    /// whole section lane (configuration change). A worker already executing a
    /// job is unaffected and remains protected by epoch/token validation.
    pub(in crate::world) fn clear_far(&self) -> Vec<JobKey> {
        let (lock, _) = &*self.gate;
        lock.lock().unwrap_or_else(|p| p.into_inner()).clear_far()
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
        // Headline runs keep profiling disabled: avoid the per-job label
        // allocation and worker clock reads in that mode. The claim key still
        // identifies a panicking job in the report.
        let profile_start =
            voxel_engine::profile::is_enabled().then(|| (job_meter(&job), std::time::Instant::now()));
        let key = JobKey::of(&job);
        // Guard the job body: a panic in `run` (bad generator sample, light/mesh
        // index, edit replay) used to unwind straight out of `worker_loop` and
        // KILL this thread — as workers died one by one, the whole pool went
        // silent and every streaming counter froze (the entry STALL). Catching it
        // keeps the thread alive, and `Done::Failed` hands the job's claim back
        // to the main thread so `fail_job` can release it and retry/quarantine —
        // a claimed key is owed exactly one `Done`, panic or not. The key
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
            match &snapshot.light {
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
        Job::Section { pos, epoch, token, generator, edits, tables } => {
            // Fused extract+mesh on owned data: the generator samples straight
            // into the dense quadrant grid — no RLE brick storage is built for
            // a result whose Section would be dropped after meshing anyway.
            // Pinned byte-identical to the storage path by the parity test in
            // `section::mesh`.
            let meshes = section::extract_section_mesh(pos, &*generator, &edits, &tables);
            Done::Section { pos, epoch, token, meshes }
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
        Arc::new(SineHills::new(&mut BlockRegistry::with_builtins(), 20.0, seed))
    }

    /// Create a far section job tagged by id for scheduler tests.
    fn section_job(terrain: &Arc<SineHills>, id: i32) -> Job {
        Job::Section {
            pos: SectionPos { detail: voxel_engine::Detail(2), x: id, z: 0 },
            epoch: 0,
            token: ClaimToken(id as u64),
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
    fn worker_generation_matches_the_sync_path() {
        let generator = generator(42);
        let coord = Coord::new(3, 1, -2); // a ground chunk: y 16..=31 crosses the surface
        let edits = vec![
            (Chunk::index(1, 3, 2), AIR),         // dig a hole
            (Chunk::index(5, 14, 5), BlockId(1)), // place high in the chunk
        ];
        let mut expected = Chunk::new(coord.x, coord.y, coord.z, &*generator);
        for &(index, id) in &edits {
            expected.set_index(index, id);
        }

        let workers = Workers::spawn(2);
        assert!(workers.submit(Job::GenerateColumn {
            col: (coord.x, coord.z),
            cy: coord.y..=coord.y,
            generator: generator.clone(),
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

        let mut expected = new_chunk_mesh_data();
        mesh::build_chunk_mesh(
            &padded,
            None,
            &tables,
            &PaddedLight::full(),
            &mut expected,
        );
        // The neighbours must actually matter, or equality proves nothing.
        let mut unculled = new_chunk_mesh_data();
        mesh::build_chunk_mesh(
            &Padded::capture(|dx, dy, dz| {
                (dx == 0 && dy == 0 && dz == 0).then_some(&chunk)
            }),
            None,
            &tables,
            &PaddedLight::full(),
            &mut unculled,
        );
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
            generator: terrain.clone(),
            edits: Vec::new(),
        };
        let far = |c: i32| section_job(&terrain, c);

        // Interleave far/near so a FIFO alone would not reproduce the order.
        let mut q = JobQueue::default();
        let gate = open_gate();
        q.push(far(0), &gate);
        q.push(near(0), &gate);
        q.push(far(1), &gate);
        q.push(near(1), &gate);

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
            generator: terrain.clone(),
            edits: Vec::new(),
        };

        let mut q = JobQueue::default();
        let gate = open_gate();
        q.push(near(26, 26), &gate); // enqueued first, but no longer the closest
        q.push(near(0, 1), &gate); // right next to the ORIGINAL centre
        q.push(near(6, 6), &gate); // a few chunks out from the original centre
        q.push(near(28, 29), &gate); // right next to where the player ends up

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
        // Cancellation ORDER was never load-bearing (the epoch rebuild drains
        // in heap layout order); the SET of descheduled claims is the contract.
        cancelled.sort_by_key(|k| match k {
            JobKey::Column { col, .. } => *col,
            _ => (i32::MAX, i32::MAX),
        });
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
        // A permissive gate never bumps its epoch, so the admission keys hold.
        let mut q = JobQueue::default();
        let gate = open_gate();
        assert!(q.push_far(job(0), 9)); // seq 0
        assert!(q.push_far(job(1), 1)); // seq 1 — first-pushed of the two dist2 = 1
        assert!(q.push_far(job(2), 4)); // seq 2
        assert!(q.push_far(job(3), 1)); // seq 3
        assert_eq!(id_of(&pop_clean(&mut q, &gate).unwrap()), 1, "nearest, first-pushed tie");
        assert_eq!(id_of(&pop_clean(&mut q, &gate).unwrap()), 3, "nearest, second tie (FIFO)");
        assert_eq!(id_of(&pop_clean(&mut q, &gate).unwrap()), 2, "dist2 = 4 next");
        assert_eq!(id_of(&pop_clean(&mut q, &gate).unwrap()), 0, "dist2 = 9 last");
        assert!(pop_clean(&mut q, &gate).is_none(), "drained");

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

    /// Sustained fast movement (below the teleport threshold) must DESCHEDULE
    /// far work left beyond the live horizon and re-key the survivors to the
    /// live eye: admission-time priorities go stale in metres per frame, and
    /// the old far class never revisited them, so workers ground through
    /// sections the player had left kilometres behind.
    #[test]
    fn far_queue_rekeys_live_and_deschedules_beyond_the_horizon() {
        let terrain = generator(0);
        let job = |id: i32| section_job(&terrain, id);
        let (wx0, wz0, span0) = far_center_span(&job(0));
        let (wx50, wz50, _) = far_center_span(&job(50));

        let mut q = JobQueue::default();
        let gate = ViewGate::new();
        // Admission claims job 50 is NEAREST (dist2 = 1 vs 100) — stale lies.
        assert!(q.push_far(job(0), 100));
        assert!(q.push_far(job(50), 1));

        // The player appears at the origin; the horizon covers section 0 but
        // falls short of section 50 (midpoint of their true eye distances).
        gate.set(0, 0, 8);
        let eye = |wx: i64, wz: i64| {
            let (ex, ez) = (8.0f64, 8.0f64);
            ((wx as f64 - ex).powi(2) + (wz as f64 - ez).powi(2)).sqrt()
        };
        gate.set_far((eye(wx0, wz0) + eye(wx50, wz50)) / 2.0 - span0 as f64);

        let mut cancelled = Vec::new();
        let popped = q.pop(&gate, &mut cancelled).expect("the in-horizon section survives");
        assert_eq!(section_id(&popped), 0, "re-keyed to the LIVE eye, not admission dist");
        assert_eq!(cancelled.len(), 1, "the beyond-horizon section is descheduled");
        assert!(
            matches!(&cancelled[0], JobKey::Section { pos, .. } if pos.x == 50),
            "with its exact claim reported: {cancelled:?}"
        );
        assert!(q.pop(&gate, &mut cancelled).is_none(), "drained");
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

    /// The pool-size policy: reserve two cores, cap at 12, floor at 1.
    #[test]
    fn thread_policy_reserves_two_and_caps() {
        for (cores, want) in [(1, 1), (2, 1), (3, 1), (4, 2), (8, 6), (12, 10), (14, 12), (24, 12)] {
            assert_eq!(Workers::threads_for(cores), want, "cores = {cores}");
        }
    }

    /// Queue dequeue cost under a live moving view — the gauge for the queue
    /// redesign (today's near class runs a retain + full re-key scan on every
    /// pop). Ignored: a timing benchmark, not a correctness gate. Run with
    /// `cargo test --release queue_pop_throughput -- --ignored --nocapture`.
    /// 2026-07-19 (12-core box), retain+scan Vec: ~529k pops/s at depth 512;
    /// epoch-synced BinaryHeap: ~3.66M pops/s (the mutex hold per pop is the
    /// number that matters under a 10-worker pool).
    #[test]
    #[ignore]
    fn queue_pop_throughput() {
        let terrain = generator(7);
        // Columns within ±7 of the origin: always wanted at radius 16, so the
        // benchmark measures the scan, never descheduling churn.
        let near = |i: i32| Job::GenerateColumn {
            col: (i % 15 - 7, (i / 15) % 15 - 7),
            cy: 0..=0,
            generator: terrain.clone(),
            edits: Vec::new(),
        };
        const QUEUE: usize = 512;
        const POPS: usize = 100_000;
        let gate = ViewGate::new();
        gate.set(0, 0, 16);
        let mut q = JobQueue::default();
        for i in 0..QUEUE {
            q.push(near(i as i32), &gate);
        }
        let mut cancelled = Vec::new();
        let start = Instant::now();
        for n in 0..POPS {
            // Boundary-cross cadence: nudge the live centre every 64 pops so
            // the re-key path sees a moving view (never far enough to shed).
            if n % 64 == 0 {
                gate.set(((n / 64) % 3) as i32, 0, 16);
            }
            let job = q.pop(&gate, &mut cancelled);
            assert!(cancelled.is_empty(), "benchmark jobs must stay in view");
            q.push(job.expect("queue kept full"), &gate);
        }
        let dt = start.elapsed();
        println!(
            "{POPS} pops (queue depth {QUEUE}, moving gate) in {:.3}s = {:.0} pops/s",
            dt.as_secs_f64(),
            POPS as f64 / dt.as_secs_f64()
        );
    }

    #[test]
    fn panicking_jobs_report_failed_with_their_claim_and_leave_the_pool_alive() {
        let workers = Workers::spawn(1);
        let keys = [
            JobKey::Column { col: (3, -2), cy: 0..=2 },
            JobKey::Mesh { coord: Coord::new(1, 2, 3) },
            JobKey::Light { coord: Coord::new(-1, 0, 1) },
            JobKey::Section { pos: SectionPos { detail: voxel_engine::Detail(2), x: 5, z: -5 }, epoch: 0, token: ClaimToken(0) },
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
                    generator: generator.clone(),
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
