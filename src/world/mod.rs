//! The world owns the block palette and an *infinite*, streamed field of chunks:
//! it keeps the chunks near the player loaded (generated and meshed), discards
//! distant ones, and answers what block is at a position, whether a box collides
//! with terrain, and how to draw the visible surface.
//!
//! Two design choices serve the "optimisation ahead of readability" mandate:
//! chunks live in a `HashMap` behind a tiny multiplicative hasher (the default
//! SipHash is far too slow for a per-frame collision hot path), and player edits
//! live in a compact overlay so a chunk can be regenerated identically after it
//! streams out and back in.
pub mod chunk;
pub mod generation;
pub mod mesh;

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

use raylib::prelude::*;

use crate::block::registry::{AIR, BlockId, BlockRegistry};
use crate::math::Aabb;
use crate::render::Render;
use chunk::{CHUNK_DEPTH, CHUNK_HEIGHT, CHUNK_WIDTH, Chunk};
use generation::{SineHills, TerrainGenerator};

/// Chunks meshed and drawn around the player, measured in chunks along each axis.
pub const VIEW_RADIUS: i32 = 6;
/// One extra ring of *data* (not meshed) so edge chunks can cull faces against
/// their neighbours without re-meshing when those neighbours later load.
const DATA_MARGIN: i32 = 1;
/// Chunks beyond this Chebyshev distance are freed. A little past the view radius
/// so walking back and forth across the boundary doesn't thrash.
const UNLOAD_RADIUS: i32 = VIEW_RADIUS + 3;
/// How many fresh chunk meshes to build per stream so entering a world grows the
/// terrain in over a few frames instead of freezing on one.
const MESH_BUDGET: usize = 6;
/// The seed a default (`generate`) world uses when none is chosen.
pub const DEFAULT_SEED: i64 = 1;

/// A chunk coordinate: `(cx, cz)` where world X = `cx * CHUNK_WIDTH + local x`.
type Coord = (i32, i32);

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

/// A loaded chunk: its voxel data plus the GPU model(s) built from it. `meshed`
/// distinguishes a chunk that only has data (a margin chunk, or one awaiting its
/// turn in the mesh budget) from one ready to draw.
struct Loaded {
    chunk: Chunk,
    models: Vec<Model>,
    meshed: bool,
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
    /// own. Inner key is the flat voxel index within the chunk.
    edits: FastMap<Coord, FastMap<usize, BlockId>>,
    /// Chunks whose mesh is stale (an edit changed them) and must rebuild now,
    /// bypassing the per-frame mesh budget.
    dirty: FastSet<Coord>,
    /// The chunk the player was last centred on, so streaming only reacts to
    /// crossing a chunk boundary.
    center: Coord,
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
            center: (i32::MIN, i32::MIN),
        };
        world.ensure_region_data((0, 0));
        world
    }

    /// The default world (seed [`DEFAULT_SEED`]).
    pub fn generate() -> Self {
        Self::new(DEFAULT_SEED)
    }

    /// The seed this world was generated from.
    pub fn seed(&self) -> i64 {
        self.generator.seed
    }

    /// The block palette, for resolving ids to names, properties, and the hot
    /// solidity/colour arrays.
    pub fn registry(&self) -> &BlockRegistry {
        &self.registry
    }

    /// Mutable access to the palette, so crafting and mods can register new blocks.
    pub fn registry_mut(&mut self) -> &mut BlockRegistry {
        &mut self.registry
    }

    /// Surface height of a column, for placing the player on spawn.
    pub fn surface_y(&self, x: i32, z: i32) -> i32 {
        self.generator.height(x, z)
    }

    /// Bring the world up to date around `center` (the player's position): load and
    /// mesh nearby chunks, free distant ones. Requires a live window (it uploads
    /// meshes), so it runs from the game update, not from headless logic.
    pub fn stream(&mut self, center: Vector3, rl: &mut RaylibHandle, thread: &RaylibThread) {
        let center_chunk = (
            (center.x.floor() as i32).div_euclid(CHUNK_WIDTH as i32),
            (center.z.floor() as i32).div_euclid(CHUNK_DEPTH as i32),
        );
        self.center = center_chunk;

        self.unload_far(center_chunk);
        self.ensure_region_data(center_chunk);
        self.build_meshes(center_chunk, rl, thread);
    }

    /// Ensure every chunk within the data radius of `center` exists (voxel data
    /// only). Cheap and GPU-free, so it also seeds headless queries.
    fn ensure_region_data(&mut self, center: Coord) {
        let radius = VIEW_RADIUS + DATA_MARGIN;
        for cx in (center.0 - radius)..=(center.0 + radius) {
            for cz in (center.1 - radius)..=(center.1 + radius) {
                self.ensure_data((cx, cz));
            }
        }
    }

    /// Generate a chunk's data if it isn't loaded, replaying any saved edits on it.
    fn ensure_data(&mut self, coord: Coord) {
        if self.chunks.contains_key(&coord) {
            return;
        }
        let mut chunk = Chunk::new(coord.0, coord.1, &self.generator);
        if let Some(edits) = self.edits.get(&coord) {
            for (&index, &id) in edits {
                chunk.set_index(index, id);
            }
        }
        self.chunks.insert(
            coord,
            Loaded {
                chunk,
                models: Vec::new(),
                meshed: false,
            },
        );
    }

    /// Free chunks past the unload radius, releasing their GPU models.
    fn unload_far(&mut self, center: Coord) {
        self.chunks.retain(|&(cx, cz), _| {
            (cx - center.0).abs() <= UNLOAD_RADIUS && (cz - center.1).abs() <= UNLOAD_RADIUS
        });
    }

    /// Mesh dirty chunks immediately, then up to [`MESH_BUDGET`] fresh chunks in the
    /// view radius, nearest first. A chunk only meshes once its four orthogonal
    /// neighbours have data, so border faces are culled correctly the first time.
    fn build_meshes(&mut self, center: Coord, rl: &mut RaylibHandle, thread: &RaylibThread) {
        // Edited chunks rebuild now — the player expects a broken block to vanish
        // this frame, not whenever the budget reaches it.
        let dirty: Vec<Coord> = self.dirty.drain().collect();
        for coord in dirty {
            if self.chunks.contains_key(&coord) && self.neighbours_have_data(coord) {
                self.mesh_chunk(coord, rl, thread);
            }
        }

        // Fresh chunks, nearest first, capped by the frame budget.
        let mut pending: Vec<Coord> = self
            .chunks
            .iter()
            .filter(|(_, loaded)| !loaded.meshed)
            .map(|(&coord, _)| coord)
            .filter(|&(cx, cz)| {
                (cx - center.0).abs() <= VIEW_RADIUS && (cz - center.1).abs() <= VIEW_RADIUS
            })
            .filter(|&coord| self.neighbours_have_data(coord))
            .collect();
        pending.sort_by_key(|&(cx, cz)| (cx - center.0).abs().max((cz - center.1).abs()));

        for coord in pending.into_iter().take(MESH_BUDGET) {
            self.mesh_chunk(coord, rl, thread);
        }
    }

    /// Whether the four orthogonal neighbours of a chunk have voxel data loaded.
    fn neighbours_have_data(&self, coord: Coord) -> bool {
        let (cx, cz) = coord;
        self.chunks.contains_key(&(cx - 1, cz))
            && self.chunks.contains_key(&(cx + 1, cz))
            && self.chunks.contains_key(&(cx, cz - 1))
            && self.chunks.contains_key(&(cx, cz + 1))
    }

    /// Build (or rebuild) one chunk's GPU models and mark it drawable.
    fn mesh_chunk(&mut self, coord: Coord, rl: &mut RaylibHandle, thread: &RaylibThread) {
        // Build against a shared borrow of the world (for cross-chunk neighbour
        // culling), then store the owned models under a fresh mutable borrow.
        let models = {
            let loaded = &self.chunks[&coord];
            mesh::build_chunk_models(&loaded.chunk, self, rl, thread)
        };
        if let Some(loaded) = self.chunks.get_mut(&coord) {
            loaded.models = models;
            loaded.meshed = true;
        }
    }

    /// Look up the block id at an absolute world voxel coordinate. Anything outside
    /// the loaded region (or above/below the world) reads as [`AIR`].
    pub fn block_at(&self, x: i32, y: i32, z: i32) -> BlockId {
        if y < 0 || y >= CHUNK_HEIGHT as i32 {
            return AIR;
        }
        let coord = Self::chunk_of(x, z);
        match self.chunks.get(&coord) {
            Some(loaded) => {
                let lx = x.rem_euclid(CHUNK_WIDTH as i32) as usize;
                let lz = z.rem_euclid(CHUNK_DEPTH as i32) as usize;
                loaded.chunk.get_local(lx, y as usize, lz)
            }
            None => AIR,
        }
    }

    /// Whether the block at a world voxel coordinate is solid. The per-frame
    /// collision hot path: a fast-hashed chunk lookup plus one registry array load.
    pub fn is_solid(&self, x: i32, y: i32, z: i32) -> bool {
        self.registry.is_solid(self.block_at(x, y, z))
    }

    /// Replace the block at a world coordinate, recording the change in the edit
    /// overlay (so it survives streaming and can be saved) and marking the affected
    /// chunk — and any neighbour across a shared face — for remeshing. Returns the
    /// block that was there.
    pub fn set_block(&mut self, x: i32, y: i32, z: i32, id: BlockId) -> BlockId {
        if y < 0 || y >= CHUNK_HEIGHT as i32 {
            return AIR;
        }
        let coord = Self::chunk_of(x, z);
        let lx = x.rem_euclid(CHUNK_WIDTH as i32) as usize;
        let lz = z.rem_euclid(CHUNK_DEPTH as i32) as usize;
        let ly = y as usize;
        let index = Chunk::index(lx, ly, lz);

        let previous = self.block_at(x, y, z);
        self.edits.entry(coord).or_default().insert(index, id);

        if let Some(loaded) = self.chunks.get_mut(&coord) {
            loaded.chunk.set_index(index, id);
            loaded.meshed = false;
            self.dirty.insert(coord);
            // A block on a chunk edge also changes the neighbour's exposed faces.
            if lx == 0 {
                self.mark_dirty((coord.0 - 1, coord.1));
            }
            if lx == CHUNK_WIDTH - 1 {
                self.mark_dirty((coord.0 + 1, coord.1));
            }
            if lz == 0 {
                self.mark_dirty((coord.0, coord.1 - 1));
            }
            if lz == CHUNK_DEPTH - 1 {
                self.mark_dirty((coord.0, coord.1 + 1));
            }
        }
        previous
    }

    /// Mark a loaded chunk stale so the next stream remeshes it.
    fn mark_dirty(&mut self, coord: Coord) {
        if let Some(loaded) = self.chunks.get_mut(&coord) {
            loaded.meshed = false;
            self.dirty.insert(coord);
        }
    }

    /// Collision test: does the given box overlap any solid voxel?
    pub fn collides(&self, aabb: &Aabb) -> bool {
        aabb.voxel_cells().any(|(x, y, z)| self.is_solid(x, y, z))
    }

    /// Every recorded edit as `((x, y, z), block)`, for saving.
    pub fn edits(&self) -> impl Iterator<Item = ((i32, i32, i32), BlockId)> + '_ {
        self.edits.iter().flat_map(|(&(cx, cz), cells)| {
            cells.iter().map(move |(&index, &id)| {
                let (lx, ly, lz) = Chunk::local_of(index);
                let x = cx * CHUNK_WIDTH as i32 + lx as i32;
                let z = cz * CHUNK_DEPTH as i32 + lz as i32;
                ((x, ly as i32, z), id)
            })
        })
    }

    /// The chunk coordinate an absolute world `(x, z)` falls in.
    fn chunk_of(x: i32, z: i32) -> Coord {
        (
            x.div_euclid(CHUNK_WIDTH as i32),
            z.div_euclid(CHUNK_DEPTH as i32),
        )
    }
}

impl Render for World {
    /// Draw the meshed chunks. All per-voxel work happened when each chunk was
    /// built; a frame is one `draw_model` per chunk model.
    fn render<D: RaylibDraw3D>(&self, d: &mut D) {
        for loaded in self.chunks.values() {
            for model in &loaded.models {
                d.draw_model(model, Vector3::zero(), 1.0, Color::WHITE);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ground_is_solid_and_sky_is_air() {
        let world = World::generate();
        assert!(world.is_solid(8, 0, 8), "deep ground should be solid");
        assert!(
            !world.is_solid(8, CHUNK_HEIGHT as i32 - 1, 8),
            "top of the world should be air"
        );
    }

    #[test]
    fn collision_agrees_with_solidity() {
        let world = World::generate();
        let in_ground = Aabb::new(Vector3::new(8.5, 0.5, 8.5), Vector3::new(0.3, 0.3, 0.3));
        let in_sky = Aabb::new(
            Vector3::new(8.5, CHUNK_HEIGHT as f32 - 0.5, 8.5),
            Vector3::new(0.3, 0.3, 0.3),
        );
        assert!(world.collides(&in_ground));
        assert!(!world.collides(&in_sky));
    }

    #[test]
    fn column_is_layered_grass_dirt_stone() {
        let world = World::generate();
        let reg = world.registry();
        let (grass, dirt, stone) = (
            reg.id_by_name("Grass").unwrap(),
            reg.id_by_name("Dirt").unwrap(),
            reg.id_by_name("Stone").unwrap(),
        );

        let (x, z) = (8, 8);
        let h = (0..CHUNK_HEIGHT as i32)
            .rev()
            .find(|&y| world.is_solid(x, y, z))
            .expect("the column has solid ground");

        assert_eq!(world.block_at(x, h + 1, z), AIR);
        assert_eq!(world.block_at(x, h, z), grass);
        assert_eq!(world.block_at(x, h - 1, z), dirt);
        assert_eq!(world.block_at(x, h - 3, z), stone);
    }

    #[test]
    fn edits_persist_across_unload() {
        let mut world = World::generate();
        // Break the surface block far enough out that it will stream away, then be
        // regenerated when we ask again — the edit must replay.
        let (x, z) = (8, 8);
        let h = (0..CHUNK_HEIGHT as i32)
            .rev()
            .find(|&y| world.is_solid(x, y, z))
            .unwrap();
        world.set_block(x, h, z, AIR);
        assert_eq!(world.block_at(x, h, z), AIR);

        // Drop the chunk and regenerate its data; the recorded edit should return.
        world.chunks.clear();
        world.ensure_data(World::chunk_of(x, z));
        assert_eq!(world.block_at(x, h, z), AIR, "edit survived reload");
    }

    #[test]
    fn distinct_seeds_differ() {
        let a = World::new(1);
        let b = World::new(9_999);
        let ha: Vec<i32> = (0..16).map(|x| a.surface_y(x, 0)).collect();
        let hb: Vec<i32> = (0..16).map(|x| b.surface_y(x, 0)).collect();
        assert_ne!(ha, hb, "different seeds should sculpt different terrain");
    }
}
