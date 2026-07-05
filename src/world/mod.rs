//! The world owns the block palette and an *infinite*, streamed field of
//! chunks — infinite along all three axes: 16-cube chunks stack upward through
//! the flying-island band and downward through bottomless stone. It keeps the
//! chunks near the player loaded (generated and meshed), discards distant
//! ones, and answers what block is at a position, whether a box collides with
//! terrain, and how to draw the visible surface.
//!
//! Two design choices serve the "optimisation ahead of readability" mandate:
//! chunks live in a `HashMap` behind a tiny multiplicative hasher (the default
//! SipHash is far too slow for a per-frame collision hot path), and player edits
//! live in a compact overlay so a chunk can be regenerated identically after it
//! streams out and back in. The third is inherited from the storage layer:
//! most of the 3D streaming volume is uniform air or stone
//! ([`ChunkData::Uniform`](chunk::ChunkData)), which costs no voxel array and
//! — for air — no mesh job at all.
//!
//! Heavy chunk work is off the render thread: generation and fresh meshing run
//! on a small worker pool (see [`pipeline`]), while *edited* chunks keep a
//! synchronous remesh so a broken block never lags a frame.
//!
//! [`World`] is ONE struct, but its methods are grouped by concern across
//! sibling files (multiple `impl World` blocks — pure code motion):
//!
//! * `mod.rs` (this file) — constants, the fast hash maps, [`Loaded`], the
//!   `World` struct itself, construction, and rendering.
//! * `streaming.rs` — the per-frame [`stream`](World::stream) pass: worker
//!   result draining, generation/mesh job queueing, budgeted uploads,
//!   unloading, and the radius/centre bookkeeping.
//! * `query.rs` — read-only queries: block lookup, solidity, collision,
//!   surface height, coordinate mapping, registry/seed accessors.
//! * `edits.rs` — player edits: block placement, the edit overlay, dirty
//!   marking, mesh freeing, and the render-distance setting.
pub mod chunk;
pub mod generation;
pub mod mesh;
pub mod pipeline;

mod edits;
mod query;
mod streaming;

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;

use voxel_engine::{Frame3D, MeshData, MeshHandle};

use crate::block::registry::{BlockId, BlockRegistry};
use crate::render::Render;
use chunk::{CHUNK_SIZE, Chunk};
use generation::{SineHills, TerrainGenerator};

/// Default number of chunk rings meshed and drawn around the player.
const DEFAULT_VIEW_RADIUS: i32 = 6;
/// The range a runtime render-distance change is clamped to.
const VIEW_RADIUS_RANGE: std::ops::RangeInclusive<i32> = 3..=10;
/// One extra shell of *data* (not meshed) in all three axes so edge chunks can
/// cull faces against their neighbours without re-meshing when those
/// neighbours later load.
const DATA_MARGIN: i32 = 1;
/// How far past the view radius chunks survive horizontally before they are
/// freed, so walking back and forth across the boundary doesn't thrash.
const UNLOAD_MARGIN: i32 = 3;
/// The vertical unload hysteresis (vertical radii are smaller, so is this).
const UNLOAD_MARGIN_V: i32 = 2;
/// How many fresh-chunk *mesh jobs* may be handed to the worker pool per
/// stream. Bounds the enqueue-time snapshot cost (a few KiB copy each) and
/// keeps the queue from flooding when a world is entered.
const MESH_ENQUEUE_BUDGET: usize = 8;
/// How many finished worker meshes may be uploaded to the GPU per stream —
/// the upload is the only part of the async path the render thread still pays.
const UPLOAD_BUDGET: usize = 4;
/// How many *dirty* (edited) chunks may remesh per frame. Processed nearest
/// first, so a locally broken block still vanishes the same frame while a
/// multiplayer join snapshot flood spreads over a few frames instead of one hitch.
const DIRTY_BUDGET: usize = 8;
/// The seed a default (`generate`) world uses when none is chosen.
pub const DEFAULT_SEED: i64 = 1;

/// A chunk coordinate: `(cx, cy, cz)` where world X = `cx * CHUNK_SIZE +
/// local x`, and likewise for Y and Z.
type Coord = (i32, i32, i32);

/// The "no centre yet" sentinel that forces the next stream to run a full pass.
const NO_CENTER: Coord = (i32::MIN, i32::MIN, i32::MIN);

/// Fast identity-ish hasher for the small integer keys the chunk/edit maps use.
/// The keys are already well-distributed grid coordinates, so a couple of
/// multiplies beat a general-purpose hash by a wide margin on the hot path.
#[derive(Default)]
struct FastHasher(u64);

impl Hasher for FastHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x0100_0000_01b3);
        }
    }
    fn write_i32(&mut self, i: i32) {
        self.0 = (self.0 ^ i as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    fn write_usize(&mut self, i: usize) {
        self.0 = (self.0 ^ i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;
type FastSet<K> = HashSet<K, BuildHasherDefault<FastHasher>>;

/// A loaded chunk: its voxel data plus the GPU mesh built from it. `meshed`
/// distinguishes a chunk that only has data (a margin chunk, or one awaiting its
/// turn in the mesh budget) from one ready to draw; a meshed all-air chunk has
/// `meshed == true` with `mesh == None` (uniform-air chunks are *born* that
/// way, skipping the worker round-trip entirely).
struct Loaded {
    chunk: Chunk,
    mesh: Option<MeshHandle>,
    meshed: bool,
    /// Mesh-input revision: bumped whenever this chunk's mesh inputs change —
    /// a direct edit, or an edit on a neighbour's touching border (which flips
    /// this chunk's exposed faces). A worker mesh result carries the rev its
    /// snapshot was taken at; a result whose rev no longer matches is stale
    /// and dropped (the chunk is in `dirty` or gets re-scanned anyway).
    rev: u32,
}

/// The streamed world: the block palette, the terrain generator, the currently
/// loaded chunks, and the overlay of player edits that outlive chunk unloads.
pub struct World {
    /// The block palette every voxel indexes into. Built once and read-only; the
    /// hot solidity/colour arrays it owns are what meshing and collision read.
    registry: BlockRegistry,
    generator: SineHills,
    chunks: FastMap<Coord, Loaded>,
    /// Player edits, grouped by chunk so regenerating a chunk can replay just its
    /// own. Inner key is the flat voxel index within the chunk. (Saves store
    /// absolute coordinates; only this in-memory keying is per-chunk.)
    edits: FastMap<Coord, FastMap<usize, BlockId>>,
    /// Chunks whose mesh is stale (an edit changed them) and must rebuild,
    /// nearest first, ahead of any fresh meshing.
    dirty: FastSet<Coord>,
    /// The chunk the player was last centred on, so streaming only reacts to
    /// crossing a chunk boundary. Invalidated to force a full pass.
    center: Coord,
    /// Runtime render distance in chunk rings (clamped to [`VIEW_RADIUS_RANGE`]).
    /// The vertical streaming radius is derived from it — see
    /// [`vertical_radius`](Self::vertical_radius).
    view_radius: i32,
    /// Whether a fresh-mesh scan might still find unmeshed chunks. Cleared when
    /// a scan finishes with nothing left, set again by anything that could
    /// create work (centre/radius change, edits, new chunk data).
    pending_fresh: bool,
    /// Set when the render distance shrank; the next stream frees meshes
    /// beyond the new radius instead of leaving them drawn until movement.
    radius_shrunk: bool,
    /// Reusable CPU-side mesh scratch; `upload_mesh` copies out of it, so one
    /// buffer serves every chunk build without per-chunk allocations.
    scratch: MeshData,
    /// Snapshot of the registry's hot solidity array for the mesher, behind an
    /// `Arc` so worker mesh jobs share it without copying. Refreshed when the
    /// palette grows (it is append-only): a refresh builds a *new* Arc, and
    /// in-flight jobs keep the old one harmlessly.
    solid_table: Arc<Vec<bool>>,
    /// Background generate/mesh workers, spawned lazily on the first
    /// [`stream`](Self::stream) so headless worlds (server, tests) never start
    /// threads. Dropped with the world: closing the job queue makes every
    /// worker exit, then the handles are joined (bounded — workers never touch
    /// the GPU, so nothing can wedge the join).
    workers: Option<pipeline::Workers>,
    /// Coords with a worker job in flight — either kind, one entry per coord
    /// (a generate job implies no data, a mesh job requires data, so the two
    /// never coexist). Blocks the centre-change scan from re-enqueueing a
    /// generate and the fresh scan from re-enqueueing a mesh; cleared per
    /// coord when its result drains, whatever becomes of the result.
    in_flight: FastSet<Coord>,
    /// Finished worker meshes awaiting their turn in the per-frame upload
    /// budget. Every entry re-validates its rev at upload time — it may have
    /// gone stale while queued.
    upload_queue: VecDeque<(Coord, u32, MeshData)>,
    /// Reusable buffer for draining worker results, so the drain neither
    /// borrows the channel across the processing loop nor allocates per frame.
    done_scratch: Vec<pipeline::Done>,
    /// How many blocks the last uploaded block-texture array covered. When the
    /// palette outgrows it (0 -> N on the first stream, +1 when crafting mints
    /// a new block type), the next stream rebuilds and re-uploads the array.
    textures_built: usize,
}

impl World {
    /// A fresh world for `seed`, with the region around the origin pre-generated
    /// (data only — no GPU) so spawning and headless queries work before the first
    /// [`stream`](Self::stream).
    pub fn new(seed: i64) -> Self {
        let registry = BlockRegistry::with_builtins();
        let generator = SineHills::new(&registry, 20.0, seed);
        let mut world = Self {
            registry,
            generator,
            chunks: FastMap::default(),
            edits: FastMap::default(),
            dirty: FastSet::default(),
            center: NO_CENTER,
            view_radius: DEFAULT_VIEW_RADIUS,
            pending_fresh: true,
            radius_shrunk: false,
            scratch: MeshData::default(),
            solid_table: Arc::new(Vec::new()),
            workers: None,
            in_flight: FastSet::default(),
            upload_queue: VecDeque::new(),
            done_scratch: Vec::new(),
            textures_built: 0,
        };
        // Centre the pre-generated box on the origin's surface chunk, the
        // spawn point's own layer.
        let cy = world.generator.height(0, 0).div_euclid(CHUNK_SIZE as i32);
        world.ensure_region_data((0, cy, 0));
        world
    }

    /// The default world (seed [`DEFAULT_SEED`]).
    pub fn generate() -> Self {
        Self::new(DEFAULT_SEED)
    }

    /// Draw the meshed chunks. All per-voxel work happened when each chunk was
    /// built; a frame is one `draw_mesh` per chunk (the engine frustum-culls
    /// each against its AABB internally).
    pub fn render(&self, f: &mut Frame3D) {
        for loaded in self.chunks.values() {
            if let Some(handle) = loaded.mesh {
                f.draw_mesh(handle);
            }
        }
    }
}

impl Render for World {
    fn render(&self, f: &mut Frame3D) {
        World::render(self, f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::AIR;
    use crate::math::Aabb;
    use voxel_engine::Vec3;

    #[test]
    fn ground_is_solid_and_sky_is_air() {
        let world = World::generate();
        assert!(world.is_solid(8, 0, 8), "surface-band ground should be solid");
        assert!(
            world.is_solid(8, -200, 8) || world.block_at(8, -200, 8) == AIR,
            "deep query must not panic"
        );
        // Deep rock is stone forever down (within the pre-generated region).
        assert!(world.is_solid(8, -40, 8), "no world floor: stone all the way down");
        // Above the hills and below the island band: air.
        assert!(!world.is_solid(8, 40, 8), "sky between terrain and islands is air");
    }

    #[test]
    fn collision_agrees_with_solidity() {
        let world = World::generate();
        let in_ground = Aabb::new(Vec3::new(8.5, 0.5, 8.5), Vec3::new(0.3, 0.3, 0.3));
        let in_sky = Aabb::new(Vec3::new(8.5, 40.0, 8.5), Vec3::new(0.3, 0.3, 0.3));
        let in_deep = Aabb::new(Vec3::new(8.5, -30.0, 8.5), Vec3::new(0.3, 0.3, 0.3));
        assert!(world.collides(&in_ground));
        assert!(!world.collides(&in_sky));
        assert!(world.collides(&in_deep), "uniform stone chunks collide");
    }

    #[test]
    fn collision_grouped_lookup_matches_per_cell_path() {
        // Boxes straddling chunk boundaries exercise the multi-chunk grouping
        // (including vertical boundaries now); the grouped fast path must
        // agree with a per-cell `is_solid` sweep.
        let world = World::generate();
        for center in [
            Vec3::new(15.9, 18.0, 15.9), // corner of four chunks
            Vec3::new(0.1, 21.5, 8.0),   // one X boundary
            Vec3::new(-3.2, 19.0, -16.4),
            Vec3::new(4.0, -1.0, 4.0),  // below the surface band: solid now
            Vec3::new(4.0, 15.9, 4.0),  // straddles a vertical chunk boundary
            Vec3::new(4.0, 200.0, 4.0), // unloaded high sky: air on both paths
        ] {
            let aabb = Aabb::new(center, Vec3::new(0.4, 0.9, 0.4));
            let reference = aabb.voxel_cells().any(|(x, y, z)| world.is_solid(x, y, z));
            assert_eq!(world.collides(&aabb), reference, "at {center:?}");
        }
    }

    #[test]
    fn column_is_layered_grass_dirt_stone() {
        let mut world = World::generate();
        let reg = world.registry();
        let (grass, dirt, stone) = (
            reg.id_by_name("Grass").unwrap(),
            reg.id_by_name("Dirt").unwrap(),
            reg.id_by_name("Stone").unwrap(),
        );

        // Lowland columns (height <= base - 6) surface as sand now, and band
        // stone can carry ore flecks — those rules have their own tests in
        // generation.rs. Here we pick a column tall enough for grass whose
        // shallow stone rolled clean, and check the canonical layering.
        let z = 8;
        let (x, h) = (0..64)
            .filter_map(|x| {
                let h = (0..64).rev().find(|&y| world.is_solid(x, y, z))?;
                // Above the beach line (base 20 - 6), with unmineralised
                // shallow stone.
                (world.surface_y(x, z) > 14 && world.block_at(x, h - 3, z) == stone)
                    .then_some((x, h))
            })
            .next()
            .expect("a tall column with clean shallow stone near spawn");

        assert_eq!(world.block_at(x, h + 1, z), AIR);
        assert_eq!(world.block_at(x, h, z), grass);
        assert_eq!(world.block_at(x, h - 1, z), dirt);
        assert_eq!(world.block_at(x, h - 3, z), stone);
        // And no bottom anymore: the deep layer continues below y = 0 —
        // checked below the ore band (depth > 64), where stone is provably
        // pure. That chunk sits outside the pregenerated region, so load it.
        world.ensure_data(World::chunk_of(x, h - 70, z));
        assert_eq!(world.block_at(x, h - 70, z), stone);
    }

    #[test]
    fn edits_persist_across_unload() {
        let mut world = World::generate();
        // Break the surface block, then regenerate the chunk from scratch —
        // the edit must replay. Also place a block above the old ceiling
        // (y >= 64 is legal now) and expect the same.
        let (x, z) = (8, 8);
        let h = (0..64)
            .rev()
            .find(|&y| world.is_solid(x, y, z))
            .unwrap();
        world.set_block(x, h, z, AIR);
        assert_eq!(world.block_at(x, h, z), AIR);
        let stone = world.registry().id_by_name("Stone").unwrap();
        world.set_block(x, 70, z, stone);

        // Drop the chunks and regenerate; the recorded edits should return.
        world.chunks.clear();
        world.ensure_data(World::chunk_of(x, h, z));
        world.ensure_data(World::chunk_of(x, 70, z));
        assert_eq!(world.block_at(x, h, z), AIR, "edit survived reload");
        assert_eq!(world.block_at(x, 70, z), stone, "high edit survived reload");
    }

    #[test]
    fn distinct_seeds_differ() {
        let a = World::new(1);
        let b = World::new(9_999);
        let ha: Vec<i32> = (0..16).map(|x| a.surface_y(x, 0)).collect();
        let hb: Vec<i32> = (0..16).map(|x| b.surface_y(x, 0)).collect();
        assert_ne!(ha, hb, "different seeds should sculpt different terrain");
    }

    #[test]
    fn stale_rev_mesh_results_are_dropped() {
        let mut world = World::generate();
        world.center = (0, 0, 0); // pretend the player streamed here
        let coord = (0, 0, 0);
        let rev = world.chunks[&coord].rev;
        assert!(world.mesh_result_applies(coord, rev));

        // An edit bumps the rev: the snapshot a worker holds is now stale.
        world.set_block(3, 3, 3, AIR);
        assert!(!world.mesh_result_applies(coord, rev));

        // A stale landing is dropped and re-arms the fresh scan.
        world.pending_fresh = false;
        world.accept_mesh(coord, rev, MeshData::default());
        assert!(world.upload_queue.is_empty(), "stale result never queues");
        assert!(world.pending_fresh, "drop re-arms the scan");

        // A current-rev landing queues for upload.
        let rev = world.chunks[&coord].rev;
        world.accept_mesh(coord, rev, MeshData::default());
        assert_eq!(world.upload_queue.len(), 1);
        world.upload_queue.clear();

        // Unloaded / out-of-range coords are rejected too — horizontally and
        // vertically (the vertical radius is tighter).
        assert!(!world.mesh_result_applies((99, 0, 99), 0));
        assert!(!world.mesh_result_applies((0, 99, 0), 0));
        let rv = world.vertical_radius();
        assert!(!world.mesh_result_applies((0, rv + 1, 0), 0), "just past vertical range");
    }

    #[test]
    fn neighbour_edits_bump_the_bordering_chunks_rev() {
        let mut world = World::generate();
        // An edit at x == 0 of chunk (0, 0, 0) touches chunk (-1, 0, 0)'s border.
        let before = world.chunks[&(-1, 0, 0)].rev;
        world.set_block(0, 5, 8, AIR);
        assert_eq!(world.chunks[&(-1, 0, 0)].rev, before + 1, "border neighbour");
        assert_eq!(world.chunks[&(0, 0, 0)].rev, 1, "edited chunk itself");
        assert_eq!(world.chunks[&(1, 0, 0)].rev, 0, "far side untouched");

        // Vertical borders count too: an edit at y == 16 (bottom of chunk
        // layer 1) touches the chunk below.
        let below = world.chunks[&(0, 0, 0)].rev;
        world.set_block(8, 16, 8, AIR);
        assert_eq!(world.chunks[&(0, 0, 0)].rev, below + 1, "chunk below bumped");
        assert_eq!(world.chunks[&(0, 1, 0)].rev, 1, "edited vertical chunk");
    }

    #[test]
    fn landed_chunks_replay_edits_that_arrived_mid_flight() {
        let mut world = World::generate();
        world.center = (0, 0, 0);
        let coord = (2, 0, 2);
        let (x, z) = (coord.0 * CHUNK_SIZE as i32 + 3, coord.2 * CHUNK_SIZE as i32 + 4);
        // Simulate the coord being in flight: no data yet, edit lands meanwhile
        // (recorded in the overlay only).
        world.chunks.remove(&coord);
        world.set_block(x, 5, z, AIR);
        // The worker's result was built before that edit existed.
        let raw = Chunk::new(coord.0, coord.1, coord.2, &world.generator);
        assert_ne!(raw.get_local(3, 5, 4), AIR, "terrain is solid there");
        world.pending_fresh = false;
        world.accept_chunk(coord, raw);
        assert_eq!(world.block_at(x, 5, z), AIR, "overlay replayed on landing");
        assert!(world.pending_fresh, "new data re-arms the fresh scan");

        // Results for coords the world has moved past are discarded —
        // horizontally or vertically.
        let far = (100, 0, 100);
        world.accept_chunk(far, Chunk::new(far.0, far.1, far.2, &world.generator));
        assert!(!world.chunks.contains_key(&far), "out-of-range chunk dropped");
        let high = (0, 100, 0);
        world.accept_chunk(high, Chunk::new(high.0, high.1, high.2, &world.generator));
        assert!(!world.chunks.contains_key(&high), "out-of-height chunk dropped");
    }

    #[test]
    fn uniform_air_chunks_are_born_meshed() {
        let world = World::generate();
        // A sky chunk between the hills and the island band: uniform air,
        // meshed on arrival with no mesh and no worker job ever queued.
        let sky = &world.chunks[&(0, 3, 0)];
        assert_eq!(sky.chunk.uniform(), Some(AIR));
        assert!(sky.meshed, "uniform air needs no mesh job");
        assert!(sky.mesh.is_none());
        // A ground chunk still goes through the normal mesh path.
        let ground = &world.chunks[&(0, 0, 0)];
        assert!(!ground.meshed, "dense terrain waits for a real mesh");
    }

    #[test]
    fn vertical_radius_derives_from_view_radius() {
        let mut world = World::generate();
        for (view, vertical) in [(3, 2), (4, 2), (6, 3), (8, 4), (10, 5)] {
            world.set_view_radius(view);
            assert_eq!(world.vertical_radius(), vertical, "view {view}");
        }
    }

    #[test]
    fn streaming_order_weights_vertical_double() {
        let c = (0, 0, 0);
        assert_eq!(World::order((4, 0, 0), c), 4);
        assert_eq!(World::order((0, 2, 0), c), 4, "2 layers up ranks like 4 rings out");
        assert!(
            World::order((0, 3, 0), c) > World::order((5, 0, 0), c),
            "lateral terrain streams before the sky"
        );
    }

    #[test]
    fn view_radius_clamps_and_flags_streaming() {
        let mut world = World::generate();
        assert_eq!(world.view_radius(), DEFAULT_VIEW_RADIUS);
        world.set_view_radius(99);
        assert_eq!(world.view_radius(), 10);
        world.set_view_radius(1);
        assert_eq!(world.view_radius(), 3);
        // The change must force the next stream to rescan.
        assert!(world.pending_fresh);
        assert_eq!(world.center, NO_CENTER);
    }
}
