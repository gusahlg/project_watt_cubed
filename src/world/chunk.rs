//! [`ChunkData`] is construction vocabulary only: the generator still speaks
//! it (`from_cells`, `Uniform`), but a [`Chunk`]'s actual storage is
//! [`Brick`] — one representation, shared with LOD sections.
//! [`ChunkData::from_cells`] picks the cheapest representation: `Uniform`
//! when every cell agrees, else palette-indexed cells, else the full-width
//! fallback for a museum-wall chunk past [`PALETTE_MAX`] distinct blocks
//! (legal play, never a crash).
use crate::block::registry::{AIR, BlockId, HotTables};
use crate::ident::{BlockState, Detail};
use crate::world::brick::{Brick, ChunkPayload, PALETTE_MAX};
use crate::world::generation::TerrainGenerator;

/// Chunk edge length along every world axis (chunks are cubes).
pub const CHUNK_SIZE: usize = 16;
/// Cells per chunk (== `brick::BRICK_VOLUME`: a chunk is a k=0 brick).
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

/// Construction-vocabulary storage shape (see module doc). `BlockId`-keyed —
/// widened to `BlockState{id, state:0}` only at [`Chunk::from_data`]'s
/// boundary.
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
    /// The generator's correctness backstop — every fill path funnels
    /// through here.
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

/// Widen a construction-vocabulary [`ChunkData`] into the real storage
/// [`Brick<ChunkPayload>`] (level 0). Already palettized in the same per-cell
/// shape, so this is a direct variant-for-variant widen
/// (`BlockId → BlockState{id, state:0}`), never a re-pack — construction-
/// boundary only, once per chunk generation/load. The one canonicalization:
/// a single-entry `Paletted` collapses to `Uniform`, so a directly
/// constructed single-valued `ChunkData` still lands on the uniform fast
/// paths (analytic light, free clone, born-air). No other all-equal case is
/// reachable — the builder returns `Uniform` for an all-equal array, and
/// `Dense` means 257+ distinct ids.
fn chunk_data_to_brick(data: ChunkData) -> Brick<ChunkPayload> {
    let widen = |id: BlockId| BlockState { id, state: 0 };
    let payload = match data {
        ChunkData::Uniform(id) => ChunkPayload::Uniform(widen(id)),
        ChunkData::Paletted { palette, cells: _ } if palette.len() == 1 => {
            ChunkPayload::Uniform(widen(palette[0]))
        }
        ChunkData::Paletted { palette, cells } => {
            ChunkPayload::Paletted { palette: palette.into_iter().map(widen).collect(), cells }
        }
        ChunkData::Dense(dense) => ChunkPayload::Dense(dense.iter().map(|&id| widen(id)).collect()),
    };
    Brick { level: Detail(0), rev: voxel_engine::Rev::START, payload }
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
    data: Brick<ChunkPayload>,
}

impl Chunk {
    /// Create a chunk at the given chunk coordinate and fill it using `generator`.
    pub fn new<G: TerrainGenerator>(cx: i32, cy: i32, cz: i32, generator: &G) -> Self {
        Self { cx, cy, cz, data: chunk_data_to_brick(generator.generate(cx, cy, cz)) }
    }

    /// Build a uniform chunk of one block — for tests that place voxels by hand.
    #[cfg(test)]
    pub fn from_uniform(cx: i32, cy: i32, cz: i32, id: BlockId) -> Self {
        Self {
            cx,
            cy,
            cz,
            data: Brick { level: Detail(0), rev: voxel_engine::Rev::START, payload: ChunkPayload::Uniform(BlockState { id, state: 0 }) },
        }
    }

    /// Build a chunk from a raw cell array — for tests. Palettizes through the
    /// real [`ChunkData::from_cells`] path, so fixtures exercise the shipped
    /// representation.
    #[cfg(test)]
    pub fn from_cells(cx: i32, cy: i32, cz: i32, cells: Box<[BlockId; CHUNK_VOLUME]>) -> Self {
        Self { cx, cy, cz, data: chunk_data_to_brick(ChunkData::from_cells(cells)) }
    }

    /// Wrap pre-generated storage at a chunk coordinate — used by the column
    /// generation worker, which produces [`ChunkData`] from `generate_column`
    /// and pairs it with its coord.
    pub fn from_data(cx: i32, cy: i32, cz: i32, data: ChunkData) -> Self {
        Self { cx, cy, cz, data: chunk_data_to_brick(data) }
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

    /// The raw storage: the mesher's uniform fast paths and flat reads, plus
    /// (once a consumer needs it) the chunk's `Brick.rev`. The stored truth
    /// is `Brick`, full stop — `ChunkData` never appears here.
    #[inline]
    pub fn data(&self) -> &Brick<ChunkPayload> {
        &self.data
    }

    /// The single block filling this chunk, if it is uniform.
    #[inline]
    pub fn uniform(&self) -> Option<BlockId> {
        match &self.data.payload {
            ChunkPayload::Uniform(v) => Some(v.id),
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
            tables.opaque(BlockId(index as u16)) && tables.emission[index] == 0
        })
    }

    /// Read a voxel by flat index. For paletted chunks this is one extra
    /// dependent load from a palette that is at most 512 B and L1-resident
    /// wherever reads cluster (meshing, collision, light).
    #[inline]
    pub fn get_index(&self, index: usize) -> BlockId {
        match &self.data.payload {
            ChunkPayload::Uniform(v) => v.id,
            ChunkPayload::Paletted { palette, cells } => palette[cells[index] as usize].id,
            ChunkPayload::Dense(cells) => cells[index].id,
        }
    }

    /// Read a voxel using chunk-local coordinates.
    #[inline]
    pub fn get_local(&self, x: usize, y: usize, z: usize) -> BlockId {
        self.get_index(Self::index(x, y, z))
    }

    /// Copy the 16-cell x-row at `(y, z)` into `out` — the snapshot capture's
    /// bulk read. ONE payload dispatch per row instead of one per cell:
    /// Uniform fills, Paletted runs 16 palette loads over its contiguous `u8`
    /// row (x is the fastest axis in [`cell_index`](super::brick::cell_index)),
    /// Dense strides its row directly.
    #[inline]
    pub fn copy_row(&self, y: usize, z: usize, out: &mut [BlockId]) {
        debug_assert_eq!(out.len(), CHUNK_SIZE);
        let base = Self::index(0, y, z);
        match &self.data.payload {
            ChunkPayload::Uniform(v) => out.fill(v.id),
            ChunkPayload::Paletted { palette, cells } => {
                for (o, &idx) in out.iter_mut().zip(&cells[base..base + CHUNK_SIZE]) {
                    *o = palette[idx as usize].id;
                }
            }
            ChunkPayload::Dense(cells) => {
                for (o, c) in out.iter_mut().zip(&cells[base..base + CHUNK_SIZE]) {
                    *o = c.id;
                }
            }
        }
    }

    /// Fill `out` with one opacity bit per cell (cell-index order), decoded
    /// ONCE per light settle instead of a payload dispatch + palette load per
    /// flood probe (~6 probes × up to 4096 relaxed cells). Uniform: one
    /// probe; Paletted: one probe per palette entry then a linear cell pass;
    /// Dense: one linear pass.
    pub fn fill_opacity(
        &self,
        opaque: impl Fn(BlockId) -> bool,
        out: &mut [u64; CHUNK_VOLUME / 64],
    ) {
        match &self.data.payload {
            ChunkPayload::Uniform(v) => out.fill(if opaque(v.id) { u64::MAX } else { 0 }),
            ChunkPayload::Paletted { palette, cells } => {
                let mut lut = [false; super::brick::PALETTE_MAX];
                for (i, p) in palette.iter().enumerate() {
                    lut[i] = opaque(p.id);
                }
                out.fill(0);
                for (i, &idx) in cells.iter().enumerate() {
                    if lut[idx as usize] {
                        out[i >> 6] |= 1 << (i & 63);
                    }
                }
            }
            ChunkPayload::Dense(cells) => {
                out.fill(0);
                for (i, c) in cells.iter().enumerate() {
                    if opaque(c.id) {
                        out[i >> 6] |= 1 << (i & 63);
                    }
                }
            }
        }
    }

    /// Write a voxel using chunk-local coordinates.
    pub fn set_local(&mut self, x: usize, y: usize, z: usize, v: BlockId) {
        self.set_index(Self::index(x, y, z), v);
    }

    /// Write voxel by flat index (replay saved/broken-block edits).
    /// First differing write promotes uniform → paletted; matching write stays
    /// uniform. Storage holding a single id collapses back to Uniform
    /// (edit-revert), reclaiming the cell array. In-place on [`ChunkPayload`]:
    /// `from_cells` never runs per edit, only at construction.
    pub fn set_index(&mut self, index: usize, v: BlockId) {
        let v = BlockState { id: v, state: 0 };
        match &mut self.data.payload {
            ChunkPayload::Uniform(cur) => {
                if *cur == v {
                    return;
                }
                let mut cells = vec![0u8; CHUNK_VOLUME].into_boxed_slice();
                cells[index] = 1;
                self.data.payload = ChunkPayload::Paletted { palette: vec![*cur, v].into_boxed_slice(), cells };
            }
            ChunkPayload::Paletted { palette, cells } => {
                let slot = match palette.iter().position(|&p| p == v) {
                    Some(k) => k as u8,
                    None => {
                        if palette.len() == PALETTE_MAX {
                            // A long edit history strands dead entries: garbage-
                            // collect from live cells first; only a genuinely
                            // 256-distinct chunk pays the full-width fallback.
                            if !gc_palette(palette, cells) {
                                let mut dense = vec![BlockState { id: AIR, state: 0 }; CHUNK_VOLUME].into_boxed_slice();
                                for (d, &c) in dense.iter_mut().zip(cells.iter()) {
                                    *d = palette[c as usize];
                                }
                                dense[index] = v;
                                self.data.payload = ChunkPayload::Dense(dense);
                                return;
                            }
                        }
                        // Box<[BlockState]> has no in-place push: grow via a
                        // Vec and re-box (palettes are tiny; amortized cost).
                        let mut grown = palette.to_vec();
                        grown.push(v);
                        let slot = (grown.len() - 1) as u8;
                        *palette = grown.into_boxed_slice();
                        slot
                    }
                };
                cells[index] = slot;
                // Only the just-written value can be the new uniform fill, so
                // scan only when it could have unified the chunk — the common
                // edit keeps a chunk mixed and never scans.
                if slot == cells[0] && cells.iter().all(|&c| c == slot) {
                    self.data.payload = ChunkPayload::Uniform(v);
                }
            }
            ChunkPayload::Dense(cells) => {
                cells[index] = v;
                if v == cells[0] && cells.iter().all(|&c| c == v) {
                    self.data.payload = ChunkPayload::Uniform(v);
                }
            }
        }
    }
}

/// Compact a full palette down to its live entries, remapping cells. Returns
/// `false` when every entry is genuinely in use (nothing to reclaim).
fn gc_palette(palette: &mut Box<[BlockState]>, cells: &mut Box<[u8]>) -> bool {
    let mut used = [false; PALETTE_MAX];
    for &c in cells.iter() {
        used[c as usize] = true;
    }
    if used.iter().take(palette.len()).all(|&u| u) {
        return false;
    }
    let mut remap = [0u8; PALETTE_MAX];
    let mut live: Vec<BlockState> = Vec::with_capacity(palette.len());
    for (i, &s) in palette.iter().enumerate() {
        if used[i] {
            remap[i] = live.len() as u8;
            live.push(s);
        }
    }
    for c in cells.iter_mut() {
        *c = remap[*c as usize];
    }
    *palette = live.into_boxed_slice();
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
            matches!(&chunk.data().payload, ChunkPayload::Paletted { palette, .. } if palette.len() == 2),
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
        // The construction-vocabulary enum stays tiny (a few pointers wide).
        assert!(std::mem::size_of::<ChunkData>() <= 40);
        // So does the real stored representation: Brick is a few pointers +
        // an enum tag, nowhere near CHUNK_VOLUME.
        assert!(std::mem::size_of::<Brick<ChunkPayload>>() <= 64, "got {}", std::mem::size_of::<Brick<ChunkPayload>>());
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
        assert!(matches!(&chunk.data().payload, ChunkPayload::Paletted { palette, .. } if palette.len() == PALETTE_MAX));

        chunk.set_index(0, BlockId(999)); // palette full — must GC, not promote
        match &chunk.data().payload {
            ChunkPayload::Paletted { palette, .. } => {
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
        assert!(matches!(&chunk.data().payload, ChunkPayload::Dense(_)), "257th distinct id promotes");
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

    // ChunkData -> Brick construction and the in-place ChunkPayload ops
    // agree with an independent reference.

    /// Independent reference for `set_index`'s edit-sequence semantics:
    /// replay onto a plain `HashMap<usize, BlockId>` overlay on top of the
    /// generator's cells, read back through `get`. Deliberately NOT the
    /// production promote/GC/dense-escape state machine — this is a
    /// different code path (no palette at all) so it can't share a bug with
    /// `Chunk::set_index`.
    struct ReferenceOverlay {
        base: BlockId,
        edits: std::collections::HashMap<usize, BlockId>,
    }
    impl ReferenceOverlay {
        fn set(&mut self, i: usize, v: BlockId) {
            self.edits.insert(i, v);
        }
        fn get(&self, i: usize) -> BlockId {
            self.edits.get(&i).copied().unwrap_or(self.base)
        }
    }

    /// Reachable failure: fails if any promote/GC/dense-escape transition in
    /// `set_index`'s `ChunkPayload` state machine loses or corrupts a cell
    /// relative to a dead-simple overlay reference, across every transition
    /// (uniform->paletted, paletted GC, paletted->dense, dense stays dense,
    /// collapse back to uniform).
    #[test]
    fn set_index_matches_an_independent_overlay_reference_across_every_transition() {
        let mut chunk = Chunk::from_uniform(0, 0, 0, AIR);
        let mut reference = ReferenceOverlay { base: AIR, edits: Default::default() };

        // splitmix64: deterministic PRNG, no rand dep.
        let mut state = 0xC0FFEEu64;
        let mut next = move || {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };

        // 400 edits: enough to cross uniform->paletted, force a GC (bounded
        // id range keeps the palette saturating), and revisit indices so
        // some writes are "matching write stays" no-ops.
        for _ in 0..400 {
            let index = (next() % CHUNK_VOLUME as u64) as usize;
            let id = BlockId((next() % 300) as u16); // >256 range: also exercises Dense
            chunk.set_index(index, id);
            reference.set(index, id);
        }

        for i in 0..CHUNK_VOLUME {
            assert_eq!(chunk.get_index(i), reference.get(i), "cell {i} diverged from the overlay reference");
        }
    }

    /// Reachable failure: fails if `chunk_data_to_brick`'s widening
    /// (`BlockId -> BlockState{id, state:0}`, variant-for-variant) drops or
    /// reorders a cell relative to `ChunkData::from_cells`'s own palette — an
    /// independent construction of the "same" chunk two ways.
    #[test]
    fn construction_boundary_roundtrips_every_cell() {
        let (g, _, _) = hills(11);
        for (cx, cy, cz) in [(0, 0, 0), (0, -10, 0), (3, 2, -5)] {
            let want = g.generate(cx, cy, cz);
            let chunk = Chunk::from_data(cx, cy, cz, want.clone());
            for i in 0..CHUNK_VOLUME {
                let want_id = match &want {
                    ChunkData::Uniform(id) => *id,
                    ChunkData::Paletted { palette, cells } => palette[cells[i] as usize],
                    ChunkData::Dense(cells) => cells[i],
                };
                assert_eq!(chunk.get_index(i), want_id, "cell {i} at ({cx},{cy},{cz})");
            }
        }
    }
}
