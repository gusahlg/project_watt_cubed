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
use super::mesh::{self, BorderPlanes};
use crate::block::registry::BlockId;

/// Everything a mesh job needs, copied out of the world at enqueue time
/// (at most ~7 KiB: a dense chunk's 4 KiB cells plus up to six 0.5 KiB border
/// planes — a uniform chunk's clone is just its enum) so the worker shares no
/// state with the live chunk map.
pub struct ChunkSnapshot {
    /// A clone of the chunk: its storage (uniform id or dense cells) and coords.
    pub chunk: Chunk,
    /// The six neighbour facing planes; a missing one reads as air.
    pub borders: BorderPlanes,
    /// The solidity table, shared by refcount. Palette growth swaps the
    /// world's `Arc` for a new one while in-flight jobs keep the old — that is
    /// harmless because the palette is append-only and every result is
    /// re-validated on arrival anyway.
    pub solid: Arc<Vec<bool>>,
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
}

/// Finished work returned to the main thread.
pub enum Done {
    Chunk { coord: Coord, chunk: Chunk },
    Mesh { coord: Coord, rev: u32, data: MeshData },
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
        if done.send(run(job)).is_err() {
            return; // result channel closed mid-shutdown: stop early
        }
    }
}

/// Execute one job. Pure CPU on owned data; determinism with the synchronous
/// paths is guaranteed by running the exact same code on the same inputs
/// (`Chunk::new` + edit replay, `build_chunk_mesh_with`).
fn run(job: Job) -> Done {
    match job {
        Job::Generate {
            coord,
            generator,
            edits,
        } => {
            let mut chunk = Chunk::new(coord.0, coord.1, coord.2, &generator);
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
            let mut data = MeshData::default();
            mesh::build_chunk_mesh_with(&snapshot.chunk, &snapshot.borders, &snapshot.solid, &mut data);
            Done::Mesh { coord, rev, data }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::{AIR, BlockRegistry};
    use crate::world::mesh::Neighbours;
    use std::time::Duration;

    /// Mirrors `World::new`'s generator construction.
    fn generator(seed: i64) -> SineHills {
        SineHills::new(&BlockRegistry::with_builtins(), 20.0, seed)
    }

    #[test]
    fn worker_generation_matches_the_sync_path() {
        let generator = generator(42);
        let coord = (3, 1, -2); // a ground chunk: y 16..=31 crosses the surface
        let edits = vec![
            (Chunk::index(1, 3, 2), AIR),         // dig a hole
            (Chunk::index(5, 14, 5), BlockId(1)), // place high in the chunk
        ];
        let mut expected = Chunk::new(coord.0, coord.1, coord.2, &generator);
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
        let (nx, px) = (Chunk::new(-1, 1, 0, &generator), Chunk::new(1, 1, 0, &generator));
        let (nz, pz) = (Chunk::new(0, 1, -1, &generator), Chunk::new(0, 1, 1, &generator));
        let (ny, py) = (Chunk::new(0, 0, 0, &generator), Chunk::new(0, 2, 0, &generator));
        let neighbours = Neighbours {
            neg_x: Some(&nx),
            pos_x: Some(&px),
            neg_z: Some(&nz),
            pos_z: Some(&pz),
            neg_y: Some(&ny),
            pos_y: Some(&py),
        };
        let solid: Arc<Vec<bool>> = Arc::new(
            (0..registry.block_count())
                .map(|i| registry.is_solid(BlockId(i as u16)))
                .collect(),
        );

        let mut expected = MeshData::default();
        mesh::build_chunk_mesh(&chunk, &neighbours, &solid, &mut expected);
        // The neighbours must actually matter, or equality proves nothing.
        let alone = Neighbours {
            neg_x: None,
            pos_x: None,
            neg_z: None,
            pos_z: None,
            neg_y: None,
            pos_y: None,
        };
        let mut unculled = MeshData::default();
        mesh::build_chunk_mesh(&chunk, &alone, &solid, &mut unculled);
        assert_ne!(unculled.indices.len(), expected.indices.len(), "border culling engaged");

        let snapshot = ChunkSnapshot {
            chunk: chunk.clone(),
            borders: BorderPlanes::capture(&neighbours),
            solid: Arc::clone(&solid),
        };
        let workers = Workers::spawn(1);
        assert!(workers.submit(Job::Mesh { coord: (0, 1, 0), rev: 7, snapshot }));
        let done = workers
            .results
            .recv_timeout(Duration::from_secs(10))
            .expect("worker finished");
        let Done::Mesh { coord, rev, data } = done else {
            panic!("expected a mesh result");
        };
        assert_eq!((coord, rev), ((0, 1, 0), 7));
        assert_eq!(data.indices, expected.indices);
        assert_eq!(data.vertices.len(), expected.vertices.len());
        for (a, b) in data.vertices.iter().zip(expected.vertices.iter()) {
            assert_eq!((a.pos, a.uv, a.color), (b.pos, b.uv, b.color));
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
                    coord: (i, 0, i),
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
