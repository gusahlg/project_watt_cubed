//! pipeline.rs runs chunk generation and meshing on background threads so the
//! render thread never pays for either — a tiny std-only worker pool (zero
//! dependencies) fed and drained by [`World::stream`](super::World::stream).
//!
//! Threading model:
//! - `min(3, cores - 1).max(1)` worker threads share ONE job queue: an
//!   `mpsc::Receiver<Job>` behind a `Mutex`. A worker holds the lock only
//!   while blocked in `recv()` — exactly one worker waits on the channel, the
//!   rest wait on the mutex, and every job runs unlocked.
//! - Jobs carry owned value data only (a generator clone, a voxel snapshot,
//!   border planes, an `Arc`'d solidity table). Workers never touch the GPU,
//!   the `World`, or the live chunk map, so there is nothing to contend on
//!   and nothing that can deadlock against the render thread.
//! - Priority is enqueue order: the world sorts each batch nearest-the-player
//!   first before submitting, and the channel is FIFO — good enough.
//! - Results come back on a plain `mpsc` channel, drained non-blockingly once
//!   per frame. The main thread re-validates every result on arrival (the
//!   chunk may have unloaded, edits may have landed while the job flew).
//! - Shutdown: dropping [`Workers`] closes the job sender, so a blocked
//!   `recv()` errors out and each loop exits; `Drop` then joins the handles.
//!   Bounded even mid-burst (a worker finishes at most its current job), and
//!   a stuck GPU can never block it — workers never issue GPU calls.
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use voxel_engine::MeshData;

use super::Coord;
use super::chunk::Chunk;
use super::generation::SineHills;
use super::light::PaddedLight;
use super::lod::{self, Tile};
use super::mesh::{self, ChunkMeshData, Padded, new_chunk_mesh_data};
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

/// Work sent to the pool.
pub enum Job {
    /// Generate the chunk at `coord` from its own copy of the generator, then
    /// replay `edits` (flat voxel index -> block) — the exact synchronous
    /// recipe, so the result is voxel-identical to inline generation.
    Generate {
        coord: Coord,
        generator: SineHills,
        edits: Vec<(usize, BlockId)>,
    },
    /// Greedy-mesh a snapshot taken at chunk revision `rev`.
    Mesh {
        coord: Coord,
        rev: u32,
        snapshot: ChunkSnapshot,
    },
    /// Build a far LOD tile's coarse mesh from its own generator clone (pure fn
    /// of seed+coords — no snapshot, no rev). Carries the hot tables the greedy
    /// mesher reads (solid/opaque), shared by refcount like a chunk snapshot's.
    Tile { tile: Tile, generator: SineHills, tables: Arc<HotTables> },
}

/// Finished work returned to the main thread.
pub enum Done {
    Chunk { coord: Coord, chunk: Chunk },
    Mesh { coord: Coord, rev: u32, data: ChunkMeshData },
    Tile { tile: Tile, data: MeshData },
}

/// The worker pool. Owned by the `World` and spawned lazily on the first
/// `stream()`, so headless worlds (dedicated server, tests) never start threads.
pub struct Workers {
    /// `Some` while running; taken in `Drop` to close the queue before joining.
    jobs: Option<Sender<Job>>,
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
        let (jobs, queue) = mpsc::channel::<Job>();
        let (done, results) = mpsc::channel::<Done>();
        let queue = Arc::new(Mutex::new(queue));
        let handles = (0..threads.max(1))
            .map(|_| {
                let queue = Arc::clone(&queue);
                let done = done.clone();
                thread::spawn(move || worker_loop(&queue, &done))
            })
            .collect();
        Self {
            jobs: Some(jobs),
            results,
            handles,
        }
    }

    /// Queue a job; returns whether it was accepted. `false` means every
    /// worker died (a worker panic — a bug), and the caller must not mark the
    /// coord in flight, so the normal scans simply retry it.
    pub fn submit(&self, job: Job) -> bool {
        self.jobs.as_ref().is_some_and(|tx| tx.send(job).is_ok())
    }

    /// Non-blocking poll for one finished result.
    pub fn try_recv(&self) -> Option<Done> {
        self.results.try_recv().ok()
    }
}

impl Drop for Workers {
    /// Close the queue first, then join: each worker is either blocked in
    /// `recv()` (errors out at once) or finishing one job, so the join is
    /// bounded and GPU-independent.
    fn drop(&mut self) {
        self.jobs = None;
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
        Job::Generate { .. } => Meter::WorkGenerate,
        Job::Mesh { .. } => Meter::WorkMesh,
        Job::Tile { .. } => Meter::WorkTile,
    }
}

fn worker_loop(queue: &Mutex<Receiver<Job>>, done: &Sender<Done>) {
    loop {
        // Lock only around `recv`; the job itself runs unlocked. A poisoned
        // mutex (a sibling panicked mid-recv) still yields a usable receiver.
        let job = match queue.lock() {
            Ok(guard) => guard.recv(),
            Err(poisoned) => poisoned.into_inner().recv(),
        };
        let Ok(job) = job else {
            return; // queue closed: the world is shutting the pool down
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
        Job::Generate {
            coord,
            generator,
            edits,
        } => {
            let mut chunk = Chunk::new(coord.x, coord.y, coord.z, &generator);
            for (index, id) in edits {
                chunk.set_index(index, id);
            }
            Done::Chunk { coord, chunk }
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
        Job::Tile { tile, generator, tables } => {
            let data = lod::build_tile_mesh(tile, &generator, &tables);
            Done::Tile { tile, data }
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
        assert!(workers.submit(Job::Generate {
            coord,
            generator: generator.clone(),
            edits,
        }));
        let done = workers
            .results
            .recv_timeout(Duration::from_secs(10))
            .expect("worker finished");
        let Done::Chunk { coord: got, chunk } = done else {
            panic!("expected a chunk result");
        };
        assert_eq!(got, coord);
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
        for p in [Pass::Opaque, Pass::Transparent] {
            assert_eq!(data[p].buckets(), expected[p].buckets());
            assert_eq!(data[p].vertices(), expected[p].vertices(), "worker mesh matches sync");
        }
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
                workers.submit(Job::Generate {
                    coord: Coord::new(i, 0, i),
                    generator: generator.clone(),
                    edits: Vec::new(),
                });
            }
            drop(workers); // closes the queue, then joins — must be bounded
            let _ = finished.send(());
        });
        check
            .recv_timeout(Duration::from_secs(10))
            .expect("Workers::drop hung");
    }
}
