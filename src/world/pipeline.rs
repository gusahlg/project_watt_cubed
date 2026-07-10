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

use voxel_engine::SurfaceData;

use super::Coord;
use super::chunk::Chunk;
use super::generation::{SineHills, TerrainGenerator};
use super::light::{self, CeilingWindow, FaceShell, LightGrid, PaddedLight};
use super::lod::{self, Tile};
use super::mesh::{self, ChunkMeshData, Padded, new_chunk_mesh_data};
use super::skin::{self, SkinColumn};
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
pub enum Job {
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
        snapshot: LightSnapshot,
    },
    /// Build a far LOD tile's coarse mesh from its own generator clone (pure fn
    /// of seed+coords — no snapshot, no rev). Carries the hot tables the greedy
    /// mesher reads (solid/opaque), shared by refcount like a chunk snapshot's.
    Tile { tile: Tile, generator: SineHills, tables: Arc<HotTables> },
    /// Build a far-skin column's grey surface mesh from its own generator clone
    /// (pure fn of seed+coords — no snapshot, no rev, no tables: height only).
    Skin { col: SkinColumn, generator: SineHills },
}

/// Finished work returned to the main thread.
pub enum Done {
    /// A generated column: every chunk built for the requested `cy` range,
    /// paired with its coord. Landed together and stored in one drain step.
    Column { col: (i32, i32), chunks: Vec<(Coord, Chunk)> },
    Mesh { coord: Coord, rev: u32, data: ChunkMeshData },
    Light { coord: Coord, grid: LightGrid },
    Tile { tile: Tile, data: ChunkMeshData },
    Skin { col: SkinColumn, data: SurfaceData },
}

/// A job's scheduling class. Derived from its kind — near work outranks far LOD
/// work — so it never rides along on the wire as a redundant field.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Priority {
    /// Chunks near the player: generate, mesh, light. Dequeued first.
    Near,
    /// Far LOD geometry: tiles and skins. Dequeued only when no near work waits.
    Far,
}

fn priority(job: &Job) -> Priority {
    match job {
        Job::Tile { .. } | Job::Skin { .. } => Priority::Far,
        _ => Priority::Near,
    }
}

/// Two-class FIFO shared by the pool. `pop` drains `near` fully before `far`, so
/// far LOD jobs fill idle workers without ever starving the chunk under the
/// player. `closed` is the shutdown flag a blocked `pop` wakes on.
#[derive(Default)]
struct JobQueue {
    near: VecDeque<Job>,
    far: VecDeque<Job>,
    closed: bool,
}

impl JobQueue {
    fn push(&mut self, job: Job) {
        match priority(&job) {
            Priority::Near => self.near.push_back(job),
            Priority::Far => self.far.push_back(job),
        }
    }

    /// The next job to run: near-first, FIFO within a class.
    fn pop(&mut self) -> Option<Job> {
        self.near.pop_front().or_else(|| self.far.pop_front())
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
    pub fn submit(&self, job: Job) -> bool {
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

    /// Non-blocking poll for one finished result.
    pub fn try_recv(&self) -> Option<Done> {
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
        Job::Tile { .. } => Meter::WorkTile,
        // Reuse the tile worker meter: the skin lane is the same off-thread
        // "sample the generator + build a surface" shape, and adding a Meter
        // variant would touch profile.rs (outside this lane's file set).
        Job::Skin { .. } => Meter::WorkTile,
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
        let start = std::time::Instant::now();
        let result = run(job);
        voxel_engine::profile::add(meter, start.elapsed());
        if done.send(result).is_err() {
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
        Job::Light { coord, snapshot } => {
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
            Done::Light { coord, grid }
        }
        Job::Tile { tile, generator, tables } => {
            let data = lod::build_tile_mesh(tile, &generator, &tables);
            Done::Tile { tile, data }
        }
        Job::Skin { col, generator } => {
            let data = skin::build_skin_mesh(col, &generator);
            Done::Skin { col, data }
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
        let far = |c: i32| Job::Skin { col: SkinColumn { x: c, z: c }, generator: terrain.clone() };

        // Interleave far/near so a FIFO alone would not reproduce the order.
        let mut q = JobQueue::default();
        q.push(far(0));
        q.push(near(0));
        q.push(far(1));
        q.push(near(1));

        // All near first (FIFO within class), then all far (FIFO within class).
        assert!(matches!(q.pop(), Some(Job::GenerateColumn { col: (0, 0), .. })));
        assert!(matches!(q.pop(), Some(Job::GenerateColumn { col: (1, 1), .. })));
        assert!(matches!(q.pop(), Some(Job::Skin { col: SkinColumn { x: 0, .. }, .. })));
        assert!(matches!(q.pop(), Some(Job::Skin { col: SkinColumn { x: 1, .. }, .. })));
        assert!(q.pop().is_none());
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
                workers.submit(Job::Skin { col: SkinColumn { x: i, z: i }, generator: generator.clone() });
            }
            drop(workers); // flags closed, wakes workers, then joins — must be bounded
            let _ = finished.send(());
        });
        check
            .recv_timeout(Duration::from_secs(10))
            .expect("Workers::drop hung");
    }
}
