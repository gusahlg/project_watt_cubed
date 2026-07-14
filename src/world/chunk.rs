//! chunk.rs stores one 16x16x16 cube of voxels. It owns no generation logic of
//! its own — it asks a [`TerrainGenerator`] to fill itself.
//!
//! A cell is a [`BlockId`] — a compact index into the world's
//! [`BlockRegistry`](crate::block::BlockRegistry), not a block itself. Storage
//! is the memory backbone of the infinite-Y world: most chunks are all air or
//! all stone, so [`ChunkData::Uniform`] stores those as one id (~a dozen bytes)
//! instead of a cell array. Mixed chunks store one *palette index* per cell
//! ([`ChunkData::Paletted`]): a chunk holds a handful of distinct blocks, so the
//! cell array stays one byte per cell no matter how wide the global palette
//! grows. Past [`PALETTE_MAX`] distinct blocks in one chunk (a museum wall of
//! crafted blocks — legal play, never a crash) storage falls back to one full
//! [`BlockId`] per cell ([`ChunkData::Dense`]).
use crate::block::registry::{AIR, BlockId, HotTables};
use crate::world::generation::TerrainGenerator;

/// Chunk edge length along every world axis (chunks are cubes).
pub const CHUNK_SIZE: usize = 16;
/// Cells per chunk.
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
/// Distinct blocks a paletted chunk can hold — the `u8` cell index space.
const PALETTE_MAX: usize = 256;

/// A chunk's voxel storage. `Uniform` is what makes an infinite-Y world
/// affordable: sky and deep rock cost no array. `Paletted` is the mixed-chunk
/// workhorse; `Dense` the >[`PALETTE_MAX`]-distinct escape hatch.
#[derive(Clone, PartialEq, Debug)]
pub enum ChunkData {
    /// Every cell is this block.
    Uniform(BlockId),
    /// One `u8` palette index per cell, flat-indexed by [`Chunk::index`];
    /// `palette[cells[i]]` is the block. Cells are valid palette indices by
    /// construction.
    Paletted { palette: Vec<BlockId>, cells: Box<[u8; CHUNK_VOLUME]> },
    /// One full block id per cell — only reachable once more than
    /// [`PALETTE_MAX`] distinct blocks share one chunk.
    Dense(Box<[BlockId; CHUNK_VOLUME]>),
}

impl ChunkData {
    /// Build storage from a dense fill, choosing the cheapest representation:
    /// `Uniform` when every cell agrees, else palette-indexed cells, else the
    /// full-width fallback. The generator's correctness backstop — every fill
    /// path funnels through here.
    pub fn from_cells(cells: Box<[BlockId; CHUNK_VOLUME]>) -> ChunkData {
        let first = cells[0];
        if cells.iter().all(|&c| c == first) {
            return ChunkData::Uniform(first);
        }
        let mut palette: Vec<BlockId> = Vec::with_capacity(8);
        let mut out = Box::new([0u8; CHUNK_VOLUME]);
        // Terrain is long same-id runs: memoize the last (id, slot) pair so the
        // palette scan runs only on run boundaries, not per cell.
        palette.push(first);
        let mut last = (first, 0u8);
        for i in 0..CHUNK_VOLUME {
            let id = cells[i];
            if id != last.0 {
                let slot = match palette.iter().position(|&p| p == id) {
                    Some(k) => k as u8,
                    None => {
                        if palette.len() == PALETTE_MAX {
                            return ChunkData::Dense(cells);
                        }
                        palette.push(id);
                        (palette.len() - 1) as u8
                    }
                };
                last = (id, slot);
            }
            out[i] = last.1;
        }
        ChunkData::Paletted { palette, cells: out }
    }
}

/// A 16-cube region of the world. `Clone` copies at most the cell array plus a
/// small palette (uniform chunks clone for free) — used to snapshot a chunk for
/// a worker-thread mesh job (see [`pipeline`](super::pipeline)), never on a
/// per-frame hot path.
#[derive(Clone)]
pub struct Chunk {
    /// Chunk coordinate on the X axis (world X = cx * CHUNK_SIZE + local x).
    pub cx: i32,
    /// Chunk coordinate on the Y axis (world Y = cy * CHUNK_SIZE + local y).
    pub cy: i32,
    /// Chunk coordinate on the Z axis (world Z = cz * CHUNK_SIZE + local z).
    pub cz: i32,
    data: ChunkData,
}

impl Chunk {
    /// Create a chunk at the given chunk coordinate and fill it using `generator`.
    pub fn new<G: TerrainGenerator>(cx: i32, cy: i32, cz: i32, generator: &G) -> Self {
        Self {
            cx,
            cy,
            cz,
            data: generator.generate(cx, cy, cz),
        }
    }

    /// Build a uniform chunk of one block — for tests that place voxels by hand.
    #[cfg(test)]
    pub fn from_uniform(cx: i32, cy: i32, cz: i32, id: BlockId) -> Self {
        Self { cx, cy, cz, data: ChunkData::Uniform(id) }
    }

    /// Build a chunk from a raw cell array — for tests. Palettizes through the
    /// real [`ChunkData::from_cells`] path, so fixtures exercise the shipped
    /// representation.
    #[cfg(test)]
    pub fn from_cells(cx: i32, cy: i32, cz: i32, cells: Box<[BlockId; CHUNK_VOLUME]>) -> Self {
        Self { cx, cy, cz, data: ChunkData::from_cells(cells) }
    }

    /// Wrap pre-generated storage at a chunk coordinate — used by the column
    /// generation worker, which produces [`ChunkData`] from `generate_column`
    /// and pairs it with its coord.
    pub fn from_data(cx: i32, cy: i32, cz: i32, data: ChunkData) -> Self {
        Self { cx, cy, cz, data }
    }

    /// Flat index from chunk-local coordinates.
    pub const fn index(x: usize, y: usize, z: usize) -> usize {
        x + z * CHUNK_SIZE + y * CHUNK_SIZE * CHUNK_SIZE
    }

    /// Chunk-local coord from flat index (inverse of index()).
    pub const fn local_of(index: usize) -> (usize, usize, usize) {
        let x = index % CHUNK_SIZE;
        let z = (index / CHUNK_SIZE) % CHUNK_SIZE;
        let y = index / (CHUNK_SIZE * CHUNK_SIZE);
        (x, y, z)
    }

    /// The raw storage, for the mesher's uniform fast paths and flat reads.
    #[inline]
    pub fn data(&self) -> &ChunkData {
        &self.data
    }

    /// The single block filling this chunk, if it is uniform.
    #[inline]
    pub fn uniform(&self) -> Option<BlockId> {
        match self.data {
            ChunkData::Uniform(id) => Some(id),
            _ => None,
        }
    }

    /// Whether this chunk is uniform, opaque, and non-emissive — the analytic
    /// light fast path's "a full block of inert rock, so its settled grid is all
    /// dark" test. Opaque emitters still need propagation to seed blocklight.
    #[inline]
    pub fn is_uniform_opaque(&self, tables: &HotTables) -> bool {
        self.uniform().is_some_and(|id| {
            let index = id.0 as usize;
            tables.opaque[index] && tables.emission[index] == 0
        })
    }

    /// Read a voxel by flat index. For paletted chunks this is one extra
    /// dependent load from a palette that is at most 512 B and L1-resident
    /// wherever reads cluster (meshing, collision, light).
    #[inline]
    pub fn get_index(&self, index: usize) -> BlockId {
        match &self.data {
            ChunkData::Uniform(id) => *id,
            ChunkData::Paletted { palette, cells } => palette[cells[index] as usize],
            ChunkData::Dense(cells) => cells[index],
        }
    }

    /// Read a voxel using chunk-local coordinates.
    #[inline]
    pub fn get_local(&self, x: usize, y: usize, z: usize) -> BlockId {
        self.get_index(Self::index(x, y, z))
    }

    /// Write a voxel using chunk-local coordinates.
    pub fn set_local(&mut self, x: usize, y: usize, z: usize, v: BlockId) {
        self.set_index(Self::index(x, y, z), v);
    }

    /// Write voxel by flat index (replay saved/broken-block edits).
    /// First differing write promotes uniform → paletted; matching write stays
    /// uniform. Storage holding a single id collapses back to Uniform
    /// (edit-revert), reclaiming the cell array.
    pub fn set_index(&mut self, index: usize, v: BlockId) {
        match &mut self.data {
            ChunkData::Uniform(id) => {
                if *id == v {
                    return;
                }
                let mut cells = Box::new([0u8; CHUNK_VOLUME]);
                cells[index] = 1;
                self.data = ChunkData::Paletted { palette: vec![*id, v], cells };
            }
            ChunkData::Paletted { palette, cells } => {
                let slot = match palette.iter().position(|&p| p == v) {
                    Some(k) => k as u8,
                    None => {
                        if palette.len() == PALETTE_MAX {
                            // A long edit history strands dead entries: garbage-
                            // collect from live cells first; only a genuinely
                            // 256-distinct chunk pays the full-width fallback.
                            if !gc_palette(palette, cells) {
                                let mut dense = Box::new([AIR; CHUNK_VOLUME]);
                                for (d, &c) in dense.iter_mut().zip(cells.iter()) {
                                    *d = palette[c as usize];
                                }
                                dense[index] = v;
                                self.data = ChunkData::Dense(dense);
                                return;
                            }
                        }
                        palette.push(v);
                        (palette.len() - 1) as u8
                    }
                };
                cells[index] = slot;
                // Only the just-written value can be the new uniform fill, so
                // scan only when it could have unified the chunk — the common
                // edit keeps a chunk mixed and never scans.
                if slot == cells[0] && cells.iter().all(|&c| c == slot) {
                    self.data = ChunkData::Uniform(v);
                }
            }
            ChunkData::Dense(cells) => {
                cells[index] = v;
                if v == cells[0] && cells.iter().all(|&c| c == v) {
                    self.data = ChunkData::Uniform(v);
                }
            }
        }
    }
}

/// Compact a full palette down to its live entries, remapping cells. Returns
/// `false` when every entry is genuinely in use (nothing to reclaim).
fn gc_palette(palette: &mut Vec<BlockId>, cells: &mut Box<[u8; CHUNK_VOLUME]>) -> bool {
    let mut used = [false; PALETTE_MAX];
    for &c in cells.iter() {
        used[c as usize] = true;
    }
    if used.iter().take(palette.len()).all(|&u| u) {
        return false;
    }
    let mut remap = [0u8; PALETTE_MAX];
    let mut live: Vec<BlockId> = Vec::with_capacity(palette.len());
    for (i, &id) in palette.iter().enumerate() {
        if used[i] {
            remap[i] = live.len() as u8;
            live.push(id);
        }
    }
    for c in cells.iter_mut() {
        *c = remap[*c as usize];
    }
    *palette = live;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::{AIR, BlockRegistry};
    use crate::world::generation::SineHills;

    /// The generator plus the registry-resolved ids its terrain is made of.
    fn hills(seed: i64) -> (SineHills, BlockId, BlockId) {
        let mut registry = BlockRegistry::with_builtins();
        let stone = registry.id_by_name("Stone").unwrap();
        let dirt = registry.id_by_name("Dirt").unwrap();
        (SineHills::new(&mut registry, 20.0, seed), stone, dirt)
    }

    #[test]
    fn index_and_local_of_are_inverses() {
        for index in [0, 1, 255, 256, 4095] {
            let (x, y, z) = Chunk::local_of(index);
            assert_eq!(Chunk::index(x, y, z), index);
        }
        assert_eq!(Chunk::index(1, 2, 3), 1 + 3 * 16 + 2 * 256, "x + z*16 + y*256");
    }

    #[test]
    fn get_set_roundtrip_on_mixed() {
        let (g, _, dirt) = hills(7);
        let mut chunk = Chunk::new(0, 0, 0, &g); // ground chunk: mixed cells
        assert!(chunk.uniform().is_none(), "surface chunks hold mixed cells");
        chunk.set_local(3, 4, 5, dirt);
        assert_eq!(chunk.get_local(3, 4, 5), dirt);
        chunk.set_local(3, 4, 5, AIR);
        assert_eq!(chunk.get_local(3, 4, 5), AIR);
    }

    #[test]
    fn uniform_promotes_to_paletted_on_first_differing_write() {
        let (g, stone, _) = hills(7);
        // Deep rock — but caves can hollow deep chunks now, so scan along +z
        // for one the generator still proves (or collapses) to uniform stone.
        let mut chunk = (0..64)
            .map(|cz| Chunk::new(0, -10, cz, &g))
            .find(|c| c.uniform() == Some(stone))
            .expect("a cave-free deep chunk within 64 along +z");

        // Writing the same block keeps the cheap representation.
        chunk.set_local(0, 0, 0, stone);
        assert_eq!(chunk.uniform(), Some(stone), "matching write stays uniform");

        // The first differing write promotes, preserving every other cell.
        chunk.set_local(8, 8, 8, AIR);
        assert!(chunk.uniform().is_none(), "differing write goes paletted");
        assert!(
            matches!(chunk.data(), ChunkData::Paletted { palette, .. } if palette.len() == 2),
            "promotion palettizes, never pays full width"
        );
        assert_eq!(chunk.get_local(8, 8, 8), AIR);
        assert_eq!(chunk.get_local(0, 0, 0), stone);
        assert_eq!(chunk.get_local(15, 15, 15), stone);
    }

    #[test]
    fn mixed_chunk_recompacts_to_uniform_when_edited_back() {
        let (g, stone, _) = hills(7);
        let mut chunk = (0..64)
            .map(|cz| Chunk::new(0, -10, cz, &g))
            .find(|c| c.uniform() == Some(stone))
            .expect("a cave-free deep chunk within 64 along +z");

        // Dig a hole: promotes to paletted.
        chunk.set_local(8, 8, 8, AIR);
        assert!(chunk.uniform().is_none(), "hole makes it mixed");

        // Fill it back with the original id: the chunk is one id again and
        // must reclaim the array instead of staying mixed forever.
        chunk.set_local(8, 8, 8, stone);
        assert_eq!(chunk.uniform(), Some(stone), "edit-and-revert collapses to uniform");

        // A chunk left genuinely mixed must NOT collapse.
        chunk.set_local(1, 1, 1, AIR);
        chunk.set_local(2, 2, 2, AIR);
        chunk.set_local(1, 1, 1, stone); // one hole remains at (2,2,2)
        assert!(chunk.uniform().is_none(), "still-mixed chunk stays mixed");
        assert_eq!(chunk.get_local(2, 2, 2), AIR);
    }

    #[test]
    fn generated_sky_chunk_is_uniform_air() {
        let (g, _, _) = hills(7);
        // No fixed sky band exists (hills reach ~88 here and islands can sit
        // higher): find the first generated layer that IS uniform air. The
        // point stays "generation yields the compact representation", without
        // hardcoding where the generator puts terrain.
        let sky = (2..64)
            .map(|cy| Chunk::new(0, cy, 0, &g))
            .find(|c| c.uniform() == Some(AIR))
            .expect("a chunk layer above the terrain is uniform air, stored as one id");
        assert_eq!(sky.uniform(), Some(AIR), "sky chunk stores one id, not a cell array");
        // The uniform representation really is tiny: the enum is a few pointers
        // wide (the Paletted variant's Vec + Box), nowhere near CHUNK_VOLUME.
        assert!(std::mem::size_of::<ChunkData>() <= 40);
    }

    #[test]
    fn from_cells_picks_the_cheapest_representation() {
        let (_, stone, dirt) = hills(7);
        // All-same collapses to Uniform.
        assert_eq!(
            ChunkData::from_cells(Box::new([stone; CHUNK_VOLUME])),
            ChunkData::Uniform(stone)
        );
        // Mixed palettizes: cells index a two-entry palette, roundtripping ids.
        let mut cells = Box::new([stone; CHUNK_VOLUME]);
        cells[7] = dirt;
        let data = ChunkData::from_cells(cells);
        match &data {
            ChunkData::Paletted { palette, cells } => {
                assert_eq!(palette.len(), 2);
                assert_eq!(palette[cells[7] as usize], dirt);
                assert_eq!(palette[cells[0] as usize], stone);
            }
            other => panic!("expected Paletted, got {other:?}"),
        }
    }

    #[test]
    fn palette_gc_reclaims_dead_entries() {
        // Saturate the palette with distinct ids, then leave only two alive:
        // the next NEW id must GC the strays instead of paying full width.
        let mut chunk = Chunk::from_uniform(0, 0, 0, BlockId(0));
        for i in 0..PALETTE_MAX {
            chunk.set_index(i, BlockId(i as u16));
        }
        // Overwrite all but the last cell (an all-equal write would collapse to
        // Uniform — the other reclaim path, tested elsewhere).
        for i in 0..CHUNK_VOLUME - 1 {
            chunk.set_index(i, BlockId(1));
        }
        assert!(matches!(chunk.data(), ChunkData::Paletted { palette, .. } if palette.len() == PALETTE_MAX));

        chunk.set_index(0, BlockId(999)); // palette full — must GC, not promote
        match chunk.data() {
            ChunkData::Paletted { palette, .. } => {
                assert_eq!(palette.len(), 3, "GC kept only the live ids plus the new one")
            }
            other => panic!("expected GC'd Paletted, got a {other:?} variant"),
        }
        assert_eq!(chunk.get_index(0), BlockId(999));
        assert_eq!(chunk.get_index(CHUNK_VOLUME - 1), BlockId(0));
        assert_eq!(chunk.get_index(100), BlockId(1));
    }

    #[test]
    fn past_256_distinct_blocks_one_chunk_pays_full_width() {
        // A museum wall: more distinct blocks than the u8 index space. Storage
        // must promote to Dense (one full id per cell) and stay correct, never
        // panic — this is legal play under the widened global palette.
        let mut chunk = Chunk::from_uniform(0, 0, 0, BlockId(0));
        for i in 0..300 {
            chunk.set_index(i, BlockId(1000 + i as u16));
        }
        assert!(matches!(chunk.data(), ChunkData::Dense(_)), "257th distinct id promotes");
        for i in 0..300 {
            assert_eq!(chunk.get_index(i), BlockId(1000 + i as u16));
        }
        assert_eq!(chunk.get_index(400), BlockId(0), "untouched cells keep the old fill");
    }

    #[test]
    fn edit_replay_on_uniform_chunk_promotes_correctly() {
        let (_, stone, dirt) = hills(7);
        // Constructed uniform-air chunk: this test is about edit replay and
        // promotion, not about where the generator happens to put terrain.
        let mut chunk = Chunk::from_uniform(2, 3, 2, AIR);
        assert_eq!(chunk.uniform(), Some(AIR));
        // Replaying an edit overlay (flat index -> id) like the world does.
        for (index, id) in [(Chunk::index(1, 2, 3), stone), (Chunk::index(0, 0, 0), dirt)] {
            chunk.set_index(index, id);
        }
        assert_eq!(chunk.get_local(1, 2, 3), stone);
        assert_eq!(chunk.get_local(0, 0, 0), dirt);
        assert_eq!(chunk.get_local(5, 5, 5), AIR, "untouched cells keep the old fill");
    }
}
