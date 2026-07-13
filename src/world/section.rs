//! Run-length-encoded vertical columns for far LOD instead of dense tiles.
//!
//! Terrain is vertically homogeneous, so far columns compress as stacks of
//! [`LodRun`]s (block id, height, skylight) rather than cell arrays. Each
//! [`LodColumn`] is a top-down run stack over the fixed world-Y domain
//! `[LOD_FLOOR_Y, LOD_CEIL_Y)`. Run bottoms are implicit (derived from runs
//! above), which is a struct invariant enforced by [`LodColumn::validate`].
//! This invariant is load-bearing: the downsampler and mesher depend on it.
//!
//! A [`Section`] is an N×N grid of columns at one detail level, with a
//! per-section [`Palette`] mapping run ids to real [`BlockId`]s (sections
//! typically contain only a few block types). Finest sections are extracted
//! once from the generator; coarser levels are pure 4-to-1 integer merges
//! via [`Section::downsample`] (no re-sampling). Conservative merge rules
//! (air loses ties, skylight by vote winners' mean) prevent hollow silhouettes
//! from developing holes as detail drops.
//!
//! Skylight is baked per run from the generator's surface height: cells at or
//! above the surface are fully lit (15), buried cells are dark (0). No neighbor
//! diffusion. This matches the existing far-tile rendering and the chunk mesher's
//! treatment of the topmost air layer.
//!
//! These types compile and test in isolation; they are not yet wired into
//! World, streaming lanes, or rendering. Module-wide dead_code allowance below.
#![allow(dead_code)]

use crate::block::registry::{AIR, BlockId};
use crate::coord::ChunkCoord;

use super::chunk::{CHUNK_SIZE, Chunk};
use super::generation::TerrainGenerator;

/// Submodule keeps mesh representation and consumer together.
mod mesh;
pub(in crate::world) use mesh::{SectionMeshData, build_section_mesh};

/// 32 not 64: reduces remesh cost under frequent edits.
pub(in crate::world) const SECTION_N: usize = 32;

/// Finest LOD detail level: cell size is 2^detail metres. Zone-1 full-res chunks are finer.
pub(in crate::world) const FINEST_DETAIL: u8 = 2;

/// Fixed world-Y domain all columns tile. Floor at y=0 (below is solid ground).
/// Ceiling at 512 (covers terrain and islands). Domain height is a power of two.
pub(in crate::world) const LOD_FLOOR_Y: i32 = 0;
pub(in crate::world) const LOD_CEIL_Y: i32 = 512;
/// Vertical extent every column's runs sum to.
pub(in crate::world) const DOMAIN_H: i32 = LOD_CEIL_Y - LOD_FLOOR_Y;

/// Chunk size as a signed coordinate (edit flattening).
const CS: i32 = CHUNK_SIZE as i32;

// Quadrant: 2-bit child convention (bit 0 = +X, bit 1 = +Z).

/// One of four quadrants: bit 0 = +X, bit 1 = +Z.
/// All code routes through this definition; bit meaning is canonical here.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(in crate::world) struct Quadrant(u8);

impl Quadrant {
    /// All four quadrants, ascending — the canonical iteration order.
    pub const ALL: [Quadrant; 4] = [Quadrant(0), Quadrant(1), Quadrant(2), Quadrant(3)];

    pub fn new(q: u8) -> Quadrant {
        debug_assert!(q < 4, "quadrant is a 2-bit index, got {q}");
        Quadrant(q & 0b11)
    }
    /// The quadrant a `(x, z)` grid coord occupies within its parent.
    pub fn of_coords(x: i32, z: i32) -> Quadrant {
        Quadrant((x.rem_euclid(2) | (z.rem_euclid(2) << 1)) as u8)
    }
    pub const fn get(self) -> u8 {
        self.0
    }
    pub const fn index(self) -> usize {
        self.0 as usize
    }
    /// The +X bit as an offset (0 or 1).
    pub const fn dx(self) -> i32 {
        (self.0 & 1) as i32
    }
    /// The +Z bit as an offset (0 or 1).
    pub const fn dz(self) -> i32 {
        (self.0 >> 1) as i32
    }
}

// SectionPos: canonical position for every node of the hierarchy.

/// A section's place in the quadtree: detail level and grid coords.
/// Uses floor division (div_euclid) only to avoid the classic quadtree bug of rounding toward zero.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(in crate::world) struct SectionPos {
    pub detail: u8,
    pub x: i32,
    pub z: i32,
}

impl SectionPos {
    /// Metres per cell (2^detail).
    pub const fn cell_size(self) -> i32 {
        1 << self.detail
    }

    /// Metres per section side (N * 2^detail).
    pub const fn span(self) -> i32 {
        (SECTION_N as i32) << self.detail
    }

    /// World min-corner X of this section.
    pub const fn min_x(self) -> i32 {
        self.x * self.span()
    }

    /// World min-corner Z of this section.
    pub const fn min_z(self) -> i32 {
        self.z * self.span()
    }

    /// Coarser section containing this one, computed with floor division.
    pub fn parent(self) -> SectionPos {
        SectionPos { detail: self.detail + 1, x: self.x.div_euclid(2), z: self.z.div_euclid(2) }
    }

    /// Finer child at quadrant q; inverse of parent().
    pub fn child(self, quadrant: Quadrant) -> SectionPos {
        debug_assert!(self.detail > 0, "finest level has no children");
        SectionPos {
            detail: self.detail - 1,
            x: self.x * 2 + quadrant.dx(),
            z: self.z * 2 + quadrant.dz(),
        }
    }

    /// Which quadrant of its parent this section occupies.
    pub fn quadrant(self) -> Quadrant {
        Quadrant::of_coords(self.x, self.z)
    }
}

// LodRun: bit-packed datapoint.

const ID_SHIFT: u64 = 0;
const HEIGHT_SHIFT: u64 = 16;
const SKYLIGHT_SHIFT: u64 = 28;
const ID_MASK: u64 = (1 << 16) - 1;
const HEIGHT_MASK: u64 = (1 << 12) - 1;
const SKYLIGHT_MASK: u64 = (1 << 4) - 1;

/// Largest height a single run can encode (12 bits).
pub(in crate::world) const MAX_RUN_HEIGHT: u16 = HEIGHT_MASK as u16;
/// Fully-lit skylight (4 bits, max value).
pub(in crate::world) const FULL_SKYLIGHT: u8 = SKYLIGHT_MASK as u8;

/// One vertical run: palette id (u16), height in metres (u12), skylight (u4).
/// Top 32 bits reserved for block-light and flags. repr(transparent) + const
/// accessors preserve exact-integer equality for mesher and coalescing.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(in crate::world) struct LodRun(u64);

impl LodRun {
    /// Pack a run. Height and skylight are validated by debug_assert.
    pub const fn new(id: u16, height: u16, skylight: u8) -> Self {
        debug_assert!(height >= 1, "an empty run is not representable");
        debug_assert!(height as u64 <= HEIGHT_MASK, "run height overflows 12 bits");
        debug_assert!(skylight as u64 <= SKYLIGHT_MASK, "skylight overflows 4 bits");
        LodRun(
            ((id as u64) << ID_SHIFT)
                | ((height as u64) << HEIGHT_SHIFT)
                | ((skylight as u64) << SKYLIGHT_SHIFT),
        )
    }

    pub const fn id(self) -> u16 {
        ((self.0 >> ID_SHIFT) & ID_MASK) as u16
    }

    pub const fn height(self) -> u16 {
        ((self.0 >> HEIGHT_SHIFT) & HEIGHT_MASK) as u16
    }

    pub const fn skylight(self) -> u8 {
        ((self.0 >> SKYLIGHT_SHIFT) & SKYLIGHT_MASK) as u8
    }
}

// Palette: per-section id-map.

/// Per-section id to BlockId map. Runs store dense indices; real block ids
/// are looked up here, so a section stores one palette instead of BlockId per run.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(in crate::world) struct Palette(Vec<BlockId>);

impl Palette {
    /// Palette id 0 is reserved for AIR. Skylight and empty columns always use a fixed id.
    pub fn new() -> Self {
        Self(vec![AIR])
    }

    /// Index for b, inserting if new. Linear scan: palettes are tiny (few block types).
    pub fn intern(&mut self, b: BlockId) -> u16 {
        match self.0.iter().position(|&x| x == b) {
            Some(i) => i as u16,
            None => {
                let i = self.0.len();
                self.0.push(b);
                i as u16
            }
        }
    }

    pub fn get(&self, i: u16) -> BlockId {
        self.0[i as usize]
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

// LodColumn: top-down run stack over the fixed domain.

/// A column's runs, top-down (index 0 is the topmost). Contiguous, non-overlapping,
/// tiling the domain exactly. Typical terrain has 1-4 runs; stored as a Vec.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(in crate::world) struct LodColumn {
    runs: Vec<LodRun>,
}

impl LodColumn {
    /// Wrap top-down runs, enforcing the tiling invariant (debug builds).
    pub fn from_top_down(runs: Vec<LodRun>) -> Self {
        let col = Self { runs };
        col.validate();
        col
    }

    pub fn runs(&self) -> &[LodRun] {
        &self.runs
    }

    /// Canonical-form invariant: non-empty, heights tile domain exactly,
    /// no adjacent runs share (id, skylight) — they would have been coalesced.
    fn validate(&self) {
        debug_assert!(!self.runs.is_empty(), "a column has at least one run");
        let mut sum: i64 = 0;
        for (i, run) in self.runs.iter().enumerate() {
            debug_assert!(run.height() >= 1, "run {i} is empty");
            sum += run.height() as i64;
            if i > 0 {
                let prev = self.runs[i - 1];
                debug_assert!(
                    prev.id() != run.id() || prev.skylight() != run.skylight(),
                    "runs {} and {i} share (id, skylight) and must be one run",
                    i - 1
                );
            }
        }
        debug_assert_eq!(sum, DOMAIN_H as i64, "column runs must tile the domain");
    }

    /// The run spanning world-Y `y` (assumed within the domain).
    fn at(&self, y: i32) -> LodRun {
        let mut top = LOD_CEIL_Y;
        for &run in &self.runs {
            let bottom = top - run.height() as i32;
            if y >= bottom && y < top {
                return run;
            }
            top = bottom;
        }
        if y >= LOD_CEIL_Y { self.runs[0] } else { *self.runs.last().unwrap() }
    }

    /// Topmost solid run's top-Y and block, or None for all-air columns.
    pub fn topmost_solid(&self, palette: &Palette) -> Option<(i32, BlockId)> {
        let mut top = LOD_CEIL_Y;
        for &run in &self.runs {
            let b = palette.get(run.id());
            if b != AIR {
                return Some((top, b));
            }
            top -= run.height() as i32;
        }
        None
    }

    /// Expand runs back to one BlockId per cell-tall slice, bottom-up.
    /// Valid only when run heights are whole cell multiples (true for extracted finest columns).
    pub fn cell_ids(&self, palette: &Palette, cell: i32) -> Vec<BlockId> {
        let mut out = Vec::with_capacity((DOMAIN_H / cell) as usize);
        for &run in self.runs.iter().rev() {
            debug_assert_eq!(run.height() as i32 % cell, 0, "run not a whole number of cells");
            let b = palette.get(run.id());
            for _ in 0..(run.height() as i32 / cell) {
                out.push(b);
            }
        }
        out
    }
}

// Section: N×N columns at one detail level.

/// N×N grid of LodColumns at one detail level, with a per-section Palette.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(in crate::world) struct Section {
    pos: SectionPos,
    palette: Palette,
    cols: Box<[LodColumn]>,
}

impl Section {
    pub fn pos(&self) -> SectionPos {
        self.pos
    }

    pub fn palette(&self) -> &Palette {
        &self.palette
    }

    pub fn column(&self, ix: usize, iz: usize) -> &LodColumn {
        debug_assert!(ix < SECTION_N && iz < SECTION_N);
        &self.cols[ix + iz * SECTION_N]
    }

    /// Extract finest-level section from the generator, applying player edits.
    /// Each column is sampled once, then RLE-compacted. Skylight is baked from surface height.
    pub fn extract<G: TerrainGenerator>(
        pos: SectionPos,
        r#gen: &G,
        edits: &[(ChunkCoord, Vec<(usize, BlockId)>)],
    ) -> Section {
        let cell = pos.cell_size();
        debug_assert_eq!(DOMAIN_H % cell, 0, "cell size must divide the domain");
        let n = (DOMAIN_H / cell) as usize;
        let half = cell / 2;
        let ys: Vec<i32> = (0..n).map(|j| LOD_FLOOR_Y + j as i32 * cell + half).collect();
        let flat = flatten_edits(edits);

        let mut palette = Palette::new();
        let mut cols: Vec<LodColumn> = Vec::with_capacity(SECTION_N * SECTION_N);
        let mut scratch = vec![AIR; n];
        for iz in 0..SECTION_N as i32 {
            for ix in 0..SECTION_N as i32 {
                let (fx, fz) = (pos.min_x() + ix * cell, pos.min_z() + iz * cell);
                let (wx, wz) = (fx + half, fz + half);
                r#gen.lod_column(wx, wz, &ys, &mut scratch);
                apply_edits(&mut scratch, &flat, fx, fz, cell);
                let height = r#gen.height(wx, wz);
                cols.push(rle_column(&scratch, &ys, height, cell, &mut palette));
            }
        }
        Section { pos, palette, cols: cols.into_boxed_slice() }
    }

    /// Merge four children into their coarser parent via 4-to-1 voting.
    /// Each parent column votes on its 2x2 child block (air loses ties, skylight by winners' mean).
    pub fn downsample(children: [Section; 4]) -> Section {
        let parent_pos = children[0].pos.parent();
        for (q, child) in children.iter().enumerate() {
            debug_assert_eq!(child.pos, parent_pos.child(Quadrant::new(q as u8)), "child {q} misplaced");
        }

        let mut palette = Palette::new();
        let mut cols: Vec<LodColumn> = Vec::with_capacity(SECTION_N * SECTION_N);
        for pz in 0..SECTION_N {
            for px in 0..SECTION_N {
                // Gather the 2x2 child columns under this parent position.
                let sources: [(&LodColumn, &Palette); 4] = std::array::from_fn(|q| {
                    let qd = Quadrant::new(q as u8);
                    let gx = 2 * px + qd.dx() as usize;
                    let gz = 2 * pz + qd.dz() as usize;
                    let sect = &children[(gx / SECTION_N) + (gz / SECTION_N) * 2];
                    (sect.column(gx % SECTION_N, gz % SECTION_N), &sect.palette)
                });
                cols.push(downsample_column(sources, &mut palette));
            }
        }
        Section { pos: parent_pos, palette, cols: cols.into_boxed_slice() }
    }
}

// Extraction and downsample helpers.

/// Flatten tile edits (per-chunk flat indices) to absolute world coordinates.
fn flatten_edits(edits: &[(ChunkCoord, Vec<(usize, BlockId)>)]) -> Vec<(i32, i32, i32, BlockId)> {
    let mut out = Vec::new();
    for (coord, cells) in edits {
        for &(index, id) in cells {
            let (lx, ly, lz) = Chunk::local_of(index);
            out.push((coord.x * CS + lx as i32, coord.y * CS + ly as i32, coord.z * CS + lz as i32, id));
        }
    }
    out
}

/// Apply edits to one column's coarse cells: solid edits overwrite the cell,
/// air edits only clear at the cell's exact center sample point.
fn apply_edits(cells: &mut [BlockId], flat: &[(i32, i32, i32, BlockId)], fx: i32, fz: i32, cell: i32) {
    if flat.is_empty() {
        return;
    }
    let half = cell / 2;
    for &(ewx, ewy, ewz, id) in flat {
        if ewx < fx || ewx >= fx + cell || ewz < fz || ewz >= fz + cell {
            continue;
        }
        let j = (ewy - LOD_FLOOR_Y).div_euclid(cell);
        let Some(slot) = usize::try_from(j).ok().and_then(|j| cells.get_mut(j)) else {
            continue;
        };
        if id != AIR {
            *slot = id;
        } else {
            let (sx, sy, sz) = (fx + half, LOD_FLOOR_Y + j * cell + half, fz + half);
            if ewx == sx && ewy == sy && ewz == sz {
                *slot = AIR;
            }
        }
    }
}

/// RLE a cell array into a top-down LodColumn, baking skylight per cell.
fn rle_column(cells: &[BlockId], ys: &[i32], height: i32, cell: i32, palette: &mut Palette) -> LodColumn {
    let sky = |y: i32| if y >= height { FULL_SKYLIGHT } else { 0 };
    let mut runs: Vec<LodRun> = Vec::new();
    let mut j = 0;
    while j < cells.len() {
        let (id, light) = (cells[j], sky(ys[j]));
        let mut k = j + 1;
        while k < cells.len() && cells[k] == id && sky(ys[k]) == light {
            k += 1;
        }
        let h = ((k - j) as i32 * cell) as u16;
        runs.push(LodRun::new(palette.intern(id), h, light));
        j = k;
    }
    runs.reverse(); // built bottom-up; a column is stored top-down
    // Ceiling-window law at the source (see downsample_column): ceiling air is lit.
    debug_assert!(
        runs.first().is_none_or(|r| palette.get(r.id()) != AIR || r.skylight() == FULL_SKYLIGHT),
        "bake produced unlit ceiling air (height/lod_column disagree above the domain)"
    );
    LodColumn::from_top_down(runs)
}

// Downsample helpers.

/// Merge four child columns into one parent column via voting.
/// Sweeps Y-transitions, votes each slice (air loses ties, skylight by winners' mean),
/// then coalesces adjacent identical runs. Order-independent for determinism.
fn downsample_column(children: [(&LodColumn, &Palette); 4], out_palette: &mut Palette) -> LodColumn {
    // Collect all Y-transitions from the four stacks.
    let mut bounds: Vec<i32> = Vec::with_capacity(16);
    for (col, _) in children {
        let mut top = LOD_CEIL_Y;
        bounds.push(top);
        for run in col.runs() {
            top -= run.height() as i32;
            bounds.push(top);
        }
    }
    bounds.sort_unstable_by(|a, b| b.cmp(a));
    bounds.dedup();

    // Vote each maximal slice and coalesce adjacent identical runs into the parent.
    let mut merged: Vec<(BlockId, i32, u8)> = Vec::new();
    for w in bounds.windows(2) {
        let (hi, lo) = (w[0], w[1]);
        let mid = lo + (hi - lo) / 2;
        let mut ids = [AIR; 4];
        let mut skys = [0u8; 4];
        for (i, (col, pal)) in children.iter().enumerate() {
            let run = col.at(mid);
            ids[i] = pal.get(run.id());
            skys[i] = run.skylight();
        }
        let (block, light) = vote_slice(ids, skys);
        match merged.last_mut() {
            Some(last) if last.0 == block && last.2 == light => last.1 += hi - lo,
            _ => merged.push((block, hi - lo, light)),
        }
    }

    // Ceiling-window law: air at the domain ceiling must stay fully lit; preserve during merge.
    debug_assert!(
        merged.first().is_none_or(|&(b, _, l)| b != AIR || l == FULL_SKYLIGHT),
        "downsample produced unlit ceiling air"
    );
    let runs = merged
        .into_iter()
        .map(|(block, h, light)| LodRun::new(out_palette.intern(block), h as u16, light))
        .collect();
    LodColumn::from_top_down(runs)
}

/// Vote one slice: block by mode_air_loses, skylight by mean of winning block's values only.
fn vote_slice(ids: [BlockId; 4], skys: [u8; 4]) -> (BlockId, u8) {
    let block = mode_air_loses(ids);
    let (mut sum, mut n) = (0u32, 0u32);
    for i in 0..4 {
        if ids[i] == block {
            sum += skys[i] as u32;
            n += 1;
        }
    }
    (block, ((sum + n / 2) / n) as u8)
}

/// Mode of four blocks: air only wins with strict plurality (never ties).
/// Solid ties break to smallest BlockId for determinism.
fn mode_air_loses(ids: [BlockId; 4]) -> BlockId {
    if ids.iter().all(|&b| b == AIR) {
        return AIR;
    }
    let mut best = AIR;
    let mut best_count = 0usize;
    for &cand in &ids {
        let count = ids.iter().filter(|&&b| b == cand).count();
        let wins = count > best_count
            || (count == best_count && best == AIR && cand != AIR)
            || (count == best_count && best != AIR && cand != AIR && cand.0 < best.0);
        if wins {
            best = cand;
            best_count = count;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::BlockRegistry;
    use crate::world::generation::SineHills;

    // -- fixtures ----------------------------------------------------------

    /// Test generator: h = surface height, b = block at cell. These two closures fully determine every column.
    struct FnGen<H, B> {
        h: H,
        b: B,
        surf: BlockId,
        deep: BlockId,
    }
    impl<H: Fn(i32, i32) -> i32, B: Fn(i32, i32, i32) -> BlockId> TerrainGenerator for FnGen<H, B> {
        fn height(&self, wx: i32, wz: i32) -> i32 {
            (self.h)(wx, wz)
        }
        fn surface_at(&self, _: i32, _: i32) -> BlockId {
            self.surf
        }
        fn deep(&self) -> BlockId {
            self.deep
        }
        fn lod_block_at(&self, wx: i32, wy: i32, wz: i32) -> BlockId {
            (self.b)(wx, wy, wz)
        }
    }

    struct Blocks {
        air: BlockId,
        grass: BlockId,
        dirt: BlockId,
        stone: BlockId,
        sand: BlockId,
        water: BlockId,
    }
    fn blocks() -> Blocks {
        let r = BlockRegistry::with_builtins();
        let id = |n: &str| r.id_by_name(n).unwrap();
        Blocks {
            air: AIR,
            grass: id("Grass"),
            dirt: id("Dirt"),
            stone: id("Stone"),
            sand: id("Sand"),
            water: id("Water"),
        }
    }

    fn sine(seed: i64) -> SineHills {
        SineHills::new(&mut BlockRegistry::with_builtins(), 20.0, seed)
    }

    const FINEST: SectionPos = SectionPos { detail: FINEST_DETAIL, x: 0, z: 0 };
    const CELL: i32 = 1 << FINEST_DETAIL;
    const NCELLS: usize = (DOMAIN_H / CELL) as usize;

    /// Test generator: ground + surface + air, with optional water table and solid shelf.
    fn terrain_gen(
        b: &Blocks,
        h: i32,
        water: i32,
        shelf: Option<(i32, i32)>,
    ) -> FnGen<impl Fn(i32, i32) -> i32, impl Fn(i32, i32, i32) -> BlockId> {
        let (grass, dirt, stone, sand, water_id, air) =
            (b.grass, b.dirt, b.stone, b.sand, b.water, b.air);
        FnGen {
            h: move |_, _| h,
            b: move |_, y, _| {
                if y < h - 3 {
                    stone
                } else if y < h - 1 {
                    dirt
                } else if y < h {
                    if h <= water { sand } else { grass }
                } else if y < water {
                    water_id
                } else if shelf.is_some_and(|(lo, hi)| y >= lo && y < hi) {
                    stone
                } else {
                    air
                }
            },
            surf: grass,
            deep: stone,
        }
    }

    // -- reference (independent of the code under test) --------------------

    /// Reference: coarse-cell sweep per column from generator contract (parity oracle).
    fn reference_cells<G: TerrainGenerator>(r#gen: &G, wx: i32, wz: i32, cell: i32) -> Vec<BlockId> {
        let half = cell / 2;
        let ys: Vec<i32> = (0..DOMAIN_H / cell).map(|j| LOD_FLOOR_Y + j * cell + half).collect();
        let mut out = vec![AIR; ys.len()];
        r#gen.lod_column(wx, wz, &ys, &mut out);
        out
    }

    /// Reference implementation of apply_edits for testing.
    fn reference_reduce(cells: &mut [BlockId], flat: &[(i32, i32, i32, BlockId)], fx: i32, fz: i32, cell: i32) {
        let half = cell / 2;
        for &(ewx, ewy, ewz, id) in flat {
            if ewx < fx || ewx >= fx + cell || ewz < fz || ewz >= fz + cell {
                continue;
            }
            let i = (ewy - LOD_FLOOR_Y).div_euclid(cell);
            let Some(slot) = usize::try_from(i).ok().and_then(|i| cells.get_mut(i)) else {
                continue;
            };
            if id != AIR {
                *slot = id;
            } else if (ewx, ewy, ewz) == (fx + half, LOD_FLOOR_Y + i * cell + half, fz + half) {
                *slot = AIR;
            }
        }
    }

    // SectionPos / LodRun packing tests.

    #[test]
    fn child_parent_is_an_involution_across_the_sign_boundary() {
        for &(x, z) in &[(0, 0), (1, 1), (-1, -1), (-1, 0), (5, -7), (-4, 3), (i32::MIN / 4, 9)] {
            let p = SectionPos { detail: 5, x, z };
            for q in Quadrant::ALL {
                let c = p.child(q);
                assert_eq!(c.parent(), p, "child({q:?}).parent() != self at {x},{z}");
                assert_eq!(c.quadrant(), q, "quadrant disagrees with child index");
                assert_eq!(c.detail, 4);
            }
        }
    }

    #[test]
    fn parent_uses_floor_division_not_toward_zero() {
        // Ensure -1 divides to -1 (floor), not 0 (toward zero).
        let p = SectionPos { detail: 2, x: -1, z: -3 };
        assert_eq!(p.parent(), SectionPos { detail: 3, x: -1, z: -2 });
    }

    #[test]
    fn lodrun_packs_and_unpacks_each_field() {
        for &(id, h, sky) in &[(0u16, 1u16, 0u8), (513, DOMAIN_H as u16, 15), (65535, MAX_RUN_HEIGHT, 9)] {
            let r = LodRun::new(id, h, sky);
            assert_eq!((r.id(), r.height(), r.skylight()), (id, h, sky));
        }
    }

    // Extraction: column classes tests.

    /// Every terrain class extracts to a canonical, domain-tiling column whose cells
    /// and exposed surface match the generator's own coarse sweep.
    #[test]
    fn extraction_matches_the_generator_sweep_for_every_class() {
        let b = blocks();
        let cases: [(&str, i32, i32, Option<(i32, i32)>); 6] = [
            ("uniform-air", 0, 0, None),      // surface at the floor: all air
            ("deep-ground", 400, 0, None),    // ground fills almost the whole domain
            ("water-covered", 30, 60, None),  // ocean column
            ("shore", 40, 40, None),          // height == water table: sand beach
            ("overhang", 100, 0, Some((108, 112))), // solid shelf above the surface
            ("island", 64, 0, Some((120, 140))),    // solid body high in the sky
        ];
        for (name, h, water, shelf) in cases {
            let r#gen = terrain_gen(&b, h, water, shelf);
            let sec = Section::extract(FINEST, &r#gen, &[]);
            let col = sec.column(0, 0);
            // Cells round-trip through the RLE and equal the reference sweep.
            let want = reference_cells(&r#gen, FINEST.min_x() + CELL / 2, FINEST.min_z() + CELL / 2, CELL);
            assert_eq!(col.cell_ids(sec.palette(), CELL), want, "{name}: cells");
            // Exposed surface agrees with the reference topmost solid cell.
            let ref_top = want.iter().rposition(|&c| c != AIR);
            match (col.topmost_solid(sec.palette()), ref_top) {
                (Some((top_y, id)), Some(j)) => {
                    assert_eq!(top_y, LOD_FLOOR_Y + (j as i32 + 1) * CELL, "{name}: surface Y");
                    assert_eq!(id, want[j], "{name}: surface block");
                }
                (None, None) => {}
                (got, r) => panic!("{name}: topmost solid {got:?} vs reference cell {r:?}"),
            }
        }
    }

    #[test]
    fn skylight_is_lit_at_and_above_the_surface_and_dark_below() {
        let b = blocks();
        let r#gen = terrain_gen(&b, 100, 0, None);
        let sec = Section::extract(FINEST, &r#gen, &[]);
        let col = sec.column(0, 0);
        let mut top = LOD_CEIL_Y;
        for &run in col.runs() {
            let bottom = top - run.height() as i32;
            // A run is uniform in skylight; sample its bottom cell's centre.
            let lit = bottom + CELL / 2 >= 100;
            assert_eq!(run.skylight() == FULL_SKYLIGHT, lit, "run [{bottom},{top}) skylight");
            assert!(run.skylight() == 0 || run.skylight() == FULL_SKYLIGHT);
            top = bottom;
        }
    }

    // Extraction: edits tests.

    #[test]
    fn edit_reduction_matches_the_reference_reducer() {
        let b = blocks();
        let r#gen = terrain_gen(&b, 100, 0, None);
        // Solid placement above surface (overwrites), air digs on/off sample point.
        let cell = CELL;
        let (fx, fz) = (FINEST.min_x(), FINEST.min_z());
        let (half, j) = (cell / 2, 20i32);
        let sample_y = LOD_FLOOR_Y + j * cell + half;
        let edits_world = [
            (fx + half, sample_y, fz + half, b.stone),      // solid: overwrites the cell
            (fx + half, LOD_FLOOR_Y + 5 * cell + half, fz + half, b.air), // air on sample point: clears
            (fx + 1, LOD_FLOOR_Y + 6 * cell + half, fz + half, b.air),    // air off sample point: ignored
        ];
        let overlay = overlay_from_world(&edits_world);

        let sec = Section::extract(FINEST, &r#gen, &overlay);
        let got = sec.column(0, 0).cell_ids(sec.palette(), cell);

        let mut want = reference_cells(&r#gen, fx + half, fz + half, cell);
        let flat = flatten_edits(&overlay);
        reference_reduce(&mut want, &flat, fx, fz, cell);
        assert_eq!(got, want);
        assert_eq!(got[j as usize], b.stone, "solid edit landed");
        assert_eq!(got[5], b.air, "air edit on the sample point cleared the cell");
    }

    /// Convert world edits to chunk-overlay format for extraction.
    fn overlay_from_world(edits: &[(i32, i32, i32, BlockId)]) -> Vec<(ChunkCoord, Vec<(usize, BlockId)>)> {
        use crate::coord::BlockCoord;
        let mut map: std::collections::HashMap<ChunkCoord, Vec<(usize, BlockId)>> = Default::default();
        for &(x, y, z, id) in edits {
            let (c, l) = BlockCoord::new(x, y, z).split();
            map.entry(c).or_default().push((Chunk::index(l.lx(), l.ly(), l.lz()), id));
        }
        map.into_iter().collect()
    }

    // Determinism tests.

    #[test]
    fn extraction_is_deterministic() {
        let r#gen = sine(0xBEEF);
        let a = Section::extract(FINEST, &r#gen, &[]);
        let b = Section::extract(FINEST, &r#gen, &[]);
        assert_eq!(a, b, "same seed + pos must extract bit-identically");
    }

    #[test]
    fn downsample_is_deterministic() {
        let r#gen = sine(0x1234);
        let parent = SectionPos { detail: FINEST_DETAIL + 1, x: 0, z: 0 };
        let kids = || std::array::from_fn(|q| Section::extract(parent.child(Quadrant::new(q as u8)), &r#gen, &[]));
        assert_eq!(Section::downsample(kids()), Section::downsample(kids()));
    }

    // -- downsample reducers ----------------------------------------------

    /// Build a test column from (block, height) slices, bottom-up.
    fn column_of(pal: &mut Palette, slices: &[(BlockId, i32)]) -> LodColumn {
        let mut runs: Vec<LodRun> = slices
            .iter()
            .map(|&(b, h)| LodRun::new(pal.intern(b), h as u16, if b == AIR { FULL_SKYLIGHT } else { 0 }))
            .collect();
        runs.reverse();
        LodColumn::from_top_down(runs)
    }

    #[test]
    fn air_never_wins_a_two_two_tie() {
        let b = blocks();
        let mut pal = Palette::new();
        // Two solid children, two all-air children.
        let solid = column_of(&mut pal, &[(b.stone, 200), (b.air, DOMAIN_H - 200)]);
        let air = column_of(&mut pal, &[(b.air, DOMAIN_H)]);
        let mut out = Palette::new();
        let parent = downsample_column(
            [(&solid, &pal), (&solid, &pal), (&air, &pal), (&air, &pal)],
            &mut out,
        );
        // Solid survives the 2/2 tie.
        assert_eq!(out.get(parent.at(100).id()), b.stone, "solid survives the tie");
        assert_eq!(out.get(parent.at(400).id()), b.air, "the all-air band above stays air");
    }

    #[test]
    fn mode_prefers_a_strict_plurality_and_averages_light() {
        let b = blocks();
        // 3 air / 1 stone: air wins.
        assert_eq!(mode_air_loses([b.air, b.air, b.air, b.stone]), b.air);
        // 2 stone / 2 dirt: solids tie, smallest id wins.
        let lo = if b.stone.0 < b.dirt.0 { b.stone } else { b.dirt };
        assert_eq!(mode_air_loses([b.stone, b.stone, b.dirt, b.dirt]), lo);
        // Skylight averages winners only; buried winner keeps its dark value.
        let mut pal = Palette::new();
        let lit = column_of(&mut pal, &[(b.air, DOMAIN_H)]);
        let dark = column_of(&mut pal, &[(b.stone, DOMAIN_H)]);
        let mut out = Palette::new();
        let parent = downsample_column([(&lit, &pal), (&lit, &pal), (&dark, &pal), (&dark, &pal)], &mut out);
        assert_eq!(parent.at(256).skylight(), 0, "buried winner keeps its own light");
    }

    // Parity and island survival on real generator.

    #[test]
    fn exposed_surface_matches_sample_coarse_on_sinehills() {
        for &seed in &[1i64, 7, 42, 0xABCD] {
            let r#gen = sine(seed);
            let sec = Section::extract(FINEST, &r#gen, &[]);
            for &(ix, iz) in &[(0usize, 0usize), (5, 9), (17, 3), (31, 31)] {
                let (wx, wz) = (FINEST.min_x() + ix as i32 * CELL + CELL / 2, FINEST.min_z() + iz as i32 * CELL + CELL / 2);
                let want = reference_cells(&r#gen, wx, wz, CELL);
                let col = sec.column(ix, iz);
                assert_eq!(col.cell_ids(sec.palette(), CELL), want, "seed {seed} col {ix},{iz}");
            }
        }
    }

    #[test]
    fn islands_survive_two_merges() {
        let seed = island_bearing_seed();
        let r#gen = sine(seed);
        let grand = SectionPos { detail: FINEST_DETAIL + 2, x: 0, z: 0 };
        // Build 4x4 finest, merge to 2x2, merge to 1.
        let level1: [Section; 4] = std::array::from_fn(|q| {
            let mid = grand.child(Quadrant::new(q as u8));
            Section::downsample(std::array::from_fn(|r| Section::extract(mid.child(Quadrant::new(r as u8)), &r#gen, &[])))
        });
        let top = Section::downsample(level1);
        assert!(
            has_island_block(&top),
            "islands present at finest vanished entirely after two merges (seed {seed})"
        );
    }

    /// Check if any column has a solid run in the island band.
    fn has_island_block(sec: &Section) -> bool {
        (0..SECTION_N).any(|iz| {
            (0..SECTION_N).any(|ix| {
                let col = sec.column(ix, iz);
                let mut top = LOD_CEIL_Y;
                for &run in col.runs() {
                    let bottom = top - run.height() as i32;
                    let solid = sec.palette().get(run.id()) != AIR;
                    if solid && bottom >= crate::world::generation::ISLAND_MIN_Y {
                        return true;
                    }
                    top = bottom;
                }
                false
            })
        })
    }

    /// Find a seed with island geometry in finest extraction.
    fn island_bearing_seed() -> i64 {
        for seed in 0i64..64 {
            let r#gen = sine(seed);
            let sec = Section::extract(SectionPos { detail: FINEST_DETAIL, x: 0, z: 0 }, &r#gen, &[]);
            if has_island_block(&sec) {
                return seed;
            }
        }
        panic!("no island-bearing seed found in 0..64");
    }
}
