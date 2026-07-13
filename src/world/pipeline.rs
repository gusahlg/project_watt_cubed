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
use std::collections::VecDeque;
use std::ops::RangeInclusive;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::Coord;
use super::chunk::Chunk;
use super::generation::{SineHills, TerrainGenerator};
use super::light::{self, CeilingWindow, FaceShell, LightGrid, PaddedLight};
use super::mesh::{self, ChunkMeshData, Padded, new_chunk_mesh_data};
use super::section::{self, Section, SectionMeshData, SectionPos};
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
    pub light: PaddedLight,
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
        generator: SineHills,
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
        generator: SineHills,
        edits: Vec<(Coord, Vec<(usize, BlockId)>)>,
        tables: Arc<HotTables>,
    },
}

/// Finished work returned to the main thread.
pub(in crate::world) enum Done {
    /// A generated column: every chunk built for the requested `cy` range,
    /// paired with its coord. Landed together and stored in one drain step.
    Column { col: (i32, i32), chunks: Vec<(Coord, Chunk)> },
    Mesh { coord: Coord, rev: u32, data: ChunkMeshData },
    Light { coord: Coord, epoch: u32, grid: LightGrid },
    Section { pos: SectionPos, meshes: [SectionMeshData; 4] },
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
}

/// Two-class queue shared by the pool. `pop` drains `near` fully before `far`,
/// so far LOD jobs fill idle workers without ever starving the chunk under the
/// player. Near is FIFO; far is distance-ordered (see [`FarQueue`]). `closed` is
/// the shutdown flag a blocked `pop` wakes on.
#[derive(Default)]
struct JobQueue {
    near: VecDeque<Job>,
    far: FarQueue,
    closed: bool,
}

impl JobQueue {
    /// Push at the job's scheduling class. A far job pushed here (the legacy
    /// [`Workers::submit`] path and headless tests) carries no distance, so it
    /// sorts at `dist2 = 0` and equal-distance far jobs fall back to FIFO by
    /// `seq` — the old `VecDeque` order. This legacy path is uncapped (its only
    /// producers are tests); the streaming lanes go through the cap-checked
    /// [`Workers::submit_far`].
    fn push(&mut self, job: Job) {
        match priority(&job) {
            Priority::Near => self.near.push_back(job),
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

    /// The next job to run: near-first (FIFO), then the nearest far job.
    fn pop(&mut self) -> Option<Job> {
        self.near.pop_front().or_else(|| self.far.pop_nearest())
    }
}

/// The worker pool. Owned by the `World` and spawned lazily on the first
/// `stream()`, so headless worlds (dedicated server, tests) never start threads.
pub struct Workers {
    /// The shared job queue + its wait condition; `Drop` sets `closed` and wakes
    /// every worker to join.
    gate: Arc<(Mutex<JobQueue>, Condvar)>,
    results: Receiver<Done>,
    handles: Vec<JoinHandle<()>>,
}

impl Workers {
    /// The pool size for this machine: leave a core for the render thread,
    /// never more than 3 (chunk work is bursty, not sustained), at least 1.
    pub fn default_threads() -> usize {
        let cores = thread::available_parallelism().map_or(1, |n| n.get());
        cores.saturating_sub(1).min(3).max(1)
    }

    /// Spawn `threads` workers (at least 1) sharing one job queue.
    pub fn spawn(threads: usize) -> Self {
        let (done, results) = mpsc::channel::<Done>();
        let gate = Arc::new((Mutex::new(JobQueue::default()), Condvar::new()));
        let handles = (0..threads.max(1))
            .map(|_| {
                let gate = Arc::clone(&gate);
                let done = done.clone();
                thread::spawn(move || worker_loop(&gate, &done))
            })
            .collect();
        Self {
            gate,
            results,
            handles,
        }
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
    }
}

fn worker_loop(gate: &(Mutex<JobQueue>, Condvar), done: &Sender<Done>) {
    let (lock, cvar) = gate;
    loop {
        // Lock only around the dequeue; the job itself runs unlocked. Poisoned
        // mutexes (a sibling panicked) still yield a usable queue.
        let job = {
            let mut queue = lock.lock().unwrap_or_else(|p| p.into_inner());
            loop {
                if let Some(job) = queue.pop() {
                    break job;
                }
                if queue.closed {
                    return; // pool shutting down and drained
                }
                queue = cvar.wait(queue).unwrap_or_else(|p| p.into_inner());
            }
        };
        let meter = job_meter(&job);
        let label = job_label(&job);
        let start = std::time::Instant::now();
        // Guard the job body: a panic in `run` (bad generator sample, light/mesh
        // index, edit replay) used to unwind straight out of `worker_loop` and
        // KILL this thread — its claimed coord (`generating`/`light_inflight`)
        // never cleared and, as workers died one by one, the whole pool went
        // silent and every streaming counter froze (the entry STALL). Catching it
        // keeps the thread alive: the panicking job simply produces no `Done` (its
        // claim is refereed out by the usual staleness/rev paths on a later scan),
        // so one poison chunk degrades to a single missing result instead of a
        // dead pool. The label names the culprit so it stops being invisible.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(job)));
        voxel_engine::profile::add(meter, start.elapsed());
        match result {
            Ok(produced) => {
                if done.send(produced).is_err() {
                    return; // result channel closed mid-shutdown: stop early
                }
            }
            Err(_) => {
                eprintln!(
                    "worker: job PANICKED and was dropped (thread survives): {label}"
                );
            }
        }
    }
}

/// A short, allocation-cheap identifier for a job, captured before `run`
/// consumes it — so a panic report names the exact culprit (kind + coord).
fn job_label(job: &Job) -> String {
    match job {
        Job::GenerateColumn { col: (cx, cz), cy, .. } => {
            format!("GenerateColumn col=({cx},{cz}) cy={}..={}", cy.start(), cy.end())
        }
        Job::Mesh { coord, rev, .. } => format!("Mesh {coord:?} rev={rev}"),
        Job::Light { coord, epoch, .. } => format!("Light {coord:?} epoch={epoch}"),
        Job::Section { pos, .. } => format!("Section {pos:?}"),
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
            let mut data = new_chunk_mesh_data();
            mesh::build_chunk_mesh(
                &snapshot.padded,
                snapshot.uniform,
                &snapshot.tables,
                &snapshot.light,
                &mut data,
            );
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
        Job::Section { pos, generator, edits, tables } => {
            // Pure CPU on owned data; reuses sync path.
            let sec = Section::extract(pos, &generator, &edits);
            let meshes = section::build_section_mesh(&sec, &tables);
            Done::Section { pos, meshes }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::{AIR, BlockRegistry};
    use std::time::Duration;
    use voxel_engine::Pass;

    /// Mirrors `World::new`'s generator construction.
    fn generator(seed: i64) -> SineHills {
        SineHills::new(&BlockRegistry::with_builtins(), 20.0, seed)
    }

    /// Create a far section job tagged by id for scheduler tests.
    fn section_job(terrain: &SineHills, id: i32) -> Job {
        Job::Section {
            pos: SectionPos { detail: 2, x: id, z: 0 },
            generator: terrain.clone(),
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
        let mut expected = Chunk::new(coord.x, coord.y, coord.z, &generator);
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
        let registry = BlockRegistry::with_builtins();
        let generator = SineHills::new(&registry, 20.0, 5);
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
            light: PaddedLight::full(),
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
        q.push(far(0));
        q.push(near(0));
        q.push(far(1));
        q.push(near(1));

        // All near first (FIFO within class), then all far (FIFO within class).
        assert!(matches!(q.pop(), Some(Job::GenerateColumn { col: (0, 0), .. })));
        assert!(matches!(q.pop(), Some(Job::GenerateColumn { col: (1, 1), .. })));
        assert_eq!(section_id(&q.pop().unwrap()), 0);
        assert_eq!(section_id(&q.pop().unwrap()), 1);
        assert!(q.pop().is_none());
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
        assert_eq!(id_of(&q.pop().unwrap()), 0, "nearest still pops first");
        assert!(q.push_far(job(-1), 0), "below the cap admits again");
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
