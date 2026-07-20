//! Section: a quadtree node backed by four brick stacks.
//!
//! A section is `pos` (detail + grid coords) plus four [`BrickStack`]s, one
//! per intra-section mesh quadrant (bit 0 = +X, bit 1 = +Z — same convention
//! [`Quadrant`] uses for the UNRELATED inter-section quadtree-child concept;
//! the two "quadrant" notions share a bit layout by coincidence, not by
//! design, so this file never reuses [`Quadrant`] for the intra-section
//! split). Each stack is a vertical run of [`Brick`]s (16³ cells) built
//! directly from RLE column data.
//!
//! Each detail level is independently re-sampled from the generator; there is
//! no 4-to-1 merge between levels.
//!
//! Lighting is NOT baked here: no per-cell skylight in any payload. Far
//! sections are shaded by the same light-volume sample the chunk mesher
//! relies on (or the neutral fallback), so a relight is a texture write,
//! never a section re-extract.

use crate::block::registry::{AIR, BlockId};
use crate::coord::ChunkCoord;
#[cfg(test)]
use crate::ident::BlockState;
use crate::ident::Detail;

use super::brick::BRICK_DIM;
#[cfg(test)]
use super::brick::{BRICK_VOLUME, Brick, BrickPayload, PackStrategy, cell_index};
use super::chunk::{CHUNK_SIZE, Chunk};
#[cfg(test)]
use super::generation::TerrainGenerator;
use super::lod;

/// Submodule keeps mesh representation and consumer together.
mod mesh;
pub(in crate::world) use mesh::{SectionMeshData, extract_section_mesh};

/// 32 not 64: reduces remesh cost under frequent edits.
pub(in crate::world) const SECTION_N: usize = 32;

/// Finest LOD detail level: cell size is 2^k metres. Zone-1 full-res chunks are finer.
pub(in crate::world) const FINEST_DETAIL: Detail = Detail(2);

/// Metres per section side at detail `d` (`32·2^k`) — the section-grid span, the
/// one place [`SECTION_N`] scales the shared cell size. Free function so callers
/// holding just a [`Detail`] (heightmip, edit dirtying) don't fabricate a
/// [`SectionPos`].
pub(in crate::world) fn section_span(d: Detail) -> i32 {
    SECTION_N as i32 * lod::cell(d)
}

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

    #[cfg(test)]
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
pub(crate) struct SectionPos {
    pub detail: Detail,
    pub x: i32,
    pub z: i32,
}

impl SectionPos {
    /// Metres per cell (2^k).
    pub fn cell_size(self) -> i32 {
        lod::cell(self.detail)
    }

    /// Metres per section side (N * 2^k).
    pub fn span(self) -> i32 {
        section_span(self.detail)
    }

    /// World min-corner X of this section.
    pub fn min_x(self) -> i32 {
        self.x * self.span()
    }

    /// World min-corner Z of this section.
    pub fn min_z(self) -> i32 {
        self.z * self.span()
    }

    /// Coarser section containing this one, computed with floor division.
    pub fn parent(self) -> SectionPos {
        SectionPos { detail: Detail(self.detail.0 + 1), x: self.x.div_euclid(2), z: self.z.div_euclid(2) }
    }

    /// Finer child at quadrant q; inverse of parent().
    ///
    /// `pub(in crate::world)`, not `pub`: widening `SectionPos` crate-wide
    /// must not drag the unrelated intra-section [`Quadrant`] type with it.
    pub(in crate::world) fn child(self, quadrant: Quadrant) -> SectionPos {
        debug_assert!(self.detail.0 > 0, "finest level has no children");
        SectionPos {
            detail: Detail(self.detail.0 - 1),
            x: self.x * 2 + quadrant.dx(),
            z: self.z * 2 + quadrant.dz(),
        }
    }

    /// Which quadrant of its parent this section occupies.
    pub(in crate::world) fn quadrant(self) -> Quadrant {
        Quadrant::of_coords(self.x, self.z)
    }

    /// Cells tiling the domain at this position's detail (`DOMAIN_H / cell_size`).
    fn n_cells(self) -> i32 {
        DOMAIN_H / self.cell_size()
    }

    /// Bricks per quadrant stack (`max(1, n_cells.div_ceil(16))`, derived
    /// from `n_cells` rather than a hardcoded `2^(5-k)` so it stays correct
    /// at the coarse rings where `n_cells < 16` (single padded brick)).
    #[cfg(test)]
    fn num_bricks(self) -> usize {
        (self.n_cells() as usize).div_ceil(BRICK_DIM).max(1)
    }
}

// BrickStack: one quadrant's vertical run of bricks.

/// One resolved (BlockId, cell-count) run, bottom-up, cell units — the
/// shared decode the mesher uses.
#[cfg(test)]
#[derive(Clone, Copy)]
pub(in crate::world) struct DecodedRun {
    pub block: BlockId,
    pub count: i32,
}

/// One quadrant's vertical stack of bricks. Stack index b covers cells
/// `[16b, 16b+16)` above [`LOD_FLOOR_Y`]. A brick's cells beyond this
/// section's `n_cells` (only possible at the coarse rings, where
/// `n_cells < 16`) are padded AIR — invisible to every reader bounded by
/// `n_cells` (the mesher) or by the stack's own brick count (`column_runs`,
/// self-bounding).
#[cfg(test)]
pub(in crate::world) struct BrickStack(Box<[Brick]>);

#[cfg(test)]
impl BrickStack {
    fn from_bricks(bricks: Vec<Brick>) -> Self {
        BrickStack(bricks.into_boxed_slice())
    }

    /// Decode one brick's column (x, z) into resolved (BlockId, count) runs.
    fn brick_column(brick: &Brick, x: usize, z: usize) -> Vec<(BlockId, u8)> {
        match &brick.payload {
            BrickPayload::Uniform(v) => vec![(v.id, BRICK_DIM as u8)],
            BrickPayload::Rle { palette, columns } => {
                columns.runs_for_column(x, z).iter().map(|r| (palette[r.palette_index() as usize].id, r.count())).collect()
            }
            BrickPayload::Paletted { palette, cells } => decode_dense_column(x, z, |y| palette[cells[cell_index(x, y, z)] as usize].id),
            BrickPayload::Dense(cells) => decode_dense_column(x, z, |y| cells[cell_index(x, y, z)].id),
        }
    }

    /// Fused bottom-up runs for local column (x, z) across the whole stack:
    /// decode each brick's column, then fuse adjacent equal-BlockId runs
    /// across brick boundaries (originals are maximal within a brick, so
    /// fusion exactly reconstructs the continuous run stream).
    pub fn column_runs(&self, x: usize, z: usize) -> Vec<DecodedRun> {
        let mut out: Vec<DecodedRun> = Vec::new();
        for brick in self.0.iter() {
            for (id, count) in Self::brick_column(brick, x, z) {
                match out.last_mut() {
                    Some(last) if last.block == id => last.count += count as i32,
                    _ => out.push(DecodedRun { block: id, count: count as i32 }),
                }
            }
        }
        out
    }

}

/// Decode a Paletted/Dense brick's column via a per-cell lookup closure,
/// coalescing adjacent equal ids into runs (shared by both variants).
#[cfg(test)]
fn decode_dense_column(_x: usize, _z: usize, cell: impl Fn(usize) -> BlockId) -> Vec<(BlockId, u8)> {
    let mut out: Vec<(BlockId, u8)> = Vec::new();
    for y in 0..BRICK_DIM {
        let id = cell(y);
        match out.last_mut() {
            Some((last_id, count)) if *last_id == id => *count += 1,
            _ => out.push((id, 1)),
        }
    }
    out
}

// Section: pos + four brick stacks.

/// A section: position plus four quadrant brick stacks (bit 0 = +X, bit 1 = +Z).
///
/// TEST-ONLY since the fused paths landed: production far jobs
/// ([`mesh::extract_section_mesh`]) and the edit overlay
/// (`heightmip::resample_cell`) sample the generator directly — this stored
/// form survives as the INDEPENDENT implementation their byte-parity oracles
/// are pinned against. (Chunk storage's bricks in `brick.rs` remain
/// production; only the section-level stacking is oracle machinery now.)
#[cfg(test)]
pub(in crate::world) struct Section {
    pos: SectionPos,
    quadrants: [BrickStack; 4],
}

#[cfg(test)]
impl Section {
    pub fn pos(&self) -> SectionPos {
        self.pos
    }

    /// Global-column (0..SECTION_N each) to (quadrant, local x, local z).
    fn locate(ix: usize, iz: usize) -> (usize, usize, usize) {
        debug_assert!(ix < SECTION_N && iz < SECTION_N);
        (ix / BRICK_DIM + (iz / BRICK_DIM) * 2, ix % BRICK_DIM, iz % BRICK_DIM)
    }

    /// Fused bottom-up runs for global column (ix, iz) (0..SECTION_N each).
    pub fn column_runs(&self, ix: usize, iz: usize) -> Vec<DecodedRun> {
        let (q, lx, lz) = Self::locate(ix, iz);
        self.quadrants[q].column_runs(lx, lz)
    }

    /// Topmost solid cell's absolute Y (top-exclusive) and block over global
    /// column (ix, iz), or `None` for an all-air column. Convenience for
    /// heightmip's `resample_cell` seam.
    pub fn topmost_solid(&self, ix: usize, iz: usize) -> Option<(i32, BlockId)> {
        let cell = self.pos.cell_size();
        let mut y = LOD_FLOOR_Y;
        let mut top = None;
        for run in self.column_runs(ix, iz) {
            y += run.count * cell; // run.count is CELL units; Y is metres
            if run.block != AIR {
                top = Some((y, run.block));
            }
        }
        top
    }

    /// Extract finest-level section from the generator, applying player edits.
    /// Each column is sampled once into its cell's slot in the owning brick's
    /// flat array; [`BrickPayload::from_cells`] (the same packer chunk
    /// storage uses) then picks the payload variant. `rev` stamps every
    /// extracted brick: transient section bricks carry the edit_generation
    /// observed at extract time.
    pub fn extract<G: TerrainGenerator>(
        pos: SectionPos,
        r#gen: &G,
        edits: &[(ChunkCoord, Vec<(usize, BlockId)>)],
        rev: voxel_engine::Rev,
    ) -> Section {
        let cell = pos.cell_size();
        debug_assert_eq!(DOMAIN_H % cell, 0, "cell size must divide the domain");
        let n = pos.n_cells() as usize;
        let num_bricks = pos.num_bricks();
        let half = cell / 2;
        let ys = cell_centers(pos);
        let flat = flatten_edits(edits);

        let mut scratch = vec![AIR; n];
        let quadrants: [BrickStack; 4] = std::array::from_fn(|q| {
            let (qx, qz) = (q & 1, q >> 1);
            // Cells beyond `n` (only possible at the coarse rings, where
            // `n < num_bricks * BRICK_DIM`) stay AIR — unobserved by any
            // n_cells-bounded reader.
            let mut per_brick = vec![[BlockState { id: AIR, state: 0 }; BRICK_VOLUME]; num_bricks];
            for lz in 0..BRICK_DIM {
                for lx in 0..BRICK_DIM {
                    let (ix, iz) = (qx * BRICK_DIM + lx, qz * BRICK_DIM + lz);
                    let (fx, fz) = (pos.min_x() + ix as i32 * cell, pos.min_z() + iz as i32 * cell);
                    let (wx, wz) = (fx + half, fz + half);
                    r#gen.lod_column(wx, wz, &ys, &mut scratch);
                    apply_edits(&mut scratch, &flat, fx, fz, cell);
                    for (y, &id) in scratch.iter().enumerate() {
                        let (b, ly) = (y / BRICK_DIM, y % BRICK_DIM);
                        per_brick[b][cell_index(lx, ly, lz)] = BlockState { id, state: 0 };
                    }
                }
            }
            BrickStack::from_bricks(
                per_brick
                    .into_iter()
                    .map(|cells| Brick { level: pos.detail, rev, payload: BrickPayload::from_cells(&cells, PackStrategy::Rle) })
                    .collect(),
            )
        });
        Section { pos, quadrants }
    }

}

// Extraction helpers — shared by [`Section::extract`] (storage path) and the
// fused [`mesh::extract_section_mesh`] production path, so the two can never
// disagree on sample coordinates or edit folding.

/// The world-Y centre of every vertical cell of a section at `pos`'s detail —
/// the exact generator sample heights both extraction paths use.
pub(in crate::world) fn cell_centers(pos: SectionPos) -> Vec<i32> {
    let cell = pos.cell_size();
    let half = cell / 2;
    (0..pos.n_cells()).map(|j| LOD_FLOOR_Y + j * cell + half).collect()
}

/// Flatten tile edits (per-chunk flat indices) to absolute world coordinates,
/// sorted by ascending `(y, x, z)`. The sort is the determinism contract
/// [`apply_edits`] relies on: edits originate in hash maps whose iteration
/// order is arbitrary, but position is session-independent truth — so live
/// edit history and an unordered join snapshot flatten identically.
pub(in crate::world) fn flatten_edits(
    edits: &[(ChunkCoord, Vec<(usize, BlockId)>)],
) -> Vec<(i32, i32, i32, BlockId)> {
    let mut out = Vec::new();
    for (coord, cells) in edits {
        for &(index, id) in cells {
            let (lx, ly, lz) = Chunk::local_of(index);
            out.push((coord.x * CS + lx as i32, coord.y * CS + ly as i32, coord.z * CS + lz as i32, id));
        }
    }
    out.sort_unstable_by_key(|&(x, y, z, _)| (y, x, z));
    out
}

/// Apply edits to one column's coarse cells — an ORDER-INDEPENDENT reduction.
/// Many fine edits can land in one coarse cell; the winner is decided by
/// POSITION, never input order:
/// - an edit exactly at the cell's centre sample point wins outright (it IS
///   the cell's sample — and it is the only way an air edit clears a cell);
/// - otherwise the topmost solid edit (greatest `(y, x, z)`) wins — `flat` is
///   sorted ascending by [`flatten_edits`], so last-write-wins realizes that
///   tie-break in one sweep;
/// - air edits off the sample point never affect the coarse cell.
pub(in crate::world) fn apply_edits(
    cells: &mut [BlockId],
    flat: &[(i32, i32, i32, BlockId)],
    fx: i32,
    fz: i32,
    cell: i32,
) {
    if flat.is_empty() {
        return;
    }
    debug_assert!(
        flat.windows(2).all(|w| (w[0].1, w[0].0, w[0].2) <= (w[1].1, w[1].0, w[1].2)),
        "flat edits must arrive (y, x, z)-sorted (see flatten_edits)"
    );
    let half = cell / 2;
    let n = cells.len();
    let slot_j = move |ewx: i32, ewy: i32, ewz: i32| -> Option<usize> {
        if ewx < fx || ewx >= fx + cell || ewz < fz || ewz >= fz + cell {
            return None;
        }
        let j = (ewy - LOD_FLOOR_Y).div_euclid(cell);
        usize::try_from(j).ok().filter(|&j| j < n)
    };
    let is_centre = |ewx: i32, ewy: i32, ewz: i32, j: usize| {
        ewx == fx + half && ewz == fz + half && ewy == LOD_FLOOR_Y + j as i32 * cell + half
    };
    // Pass 1: solid, non-centre edits (ascending order makes topmost win).
    for &(ewx, ewy, ewz, id) in flat {
        let Some(j) = slot_j(ewx, ewy, ewz) else { continue };
        if id != AIR && !is_centre(ewx, ewy, ewz, j) {
            cells[j] = id;
        }
    }
    // Pass 2: centre-sample edits override everything, air included.
    for &(ewx, ewy, ewz, id) in flat {
        let Some(j) = slot_j(ewx, ewy, ewz) else { continue };
        if is_centre(ewx, ewy, ewz, j) {
            cells[j] = id;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::BlockRegistry;
    use crate::coord::BlockCoord;
    use crate::world::generation::SineHills;

    // Test fixtures

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
        Blocks { air: AIR, grass: id("Grass"), dirt: id("Dirt"), stone: id("Stone"), sand: id("Sand"), water: id("Water") }
    }

    fn sine(seed: i64) -> SineHills {
        SineHills::new(&mut BlockRegistry::with_builtins(), 20.0, seed)
    }

    const FINEST: SectionPos = SectionPos { detail: FINEST_DETAIL, x: 0, z: 0 };
    const CELL: i32 = 1 << FINEST_DETAIL.0;

    fn extract<G: TerrainGenerator>(pos: SectionPos, g: &G, edits: &[(ChunkCoord, Vec<(usize, BlockId)>)]) -> Section {
        Section::extract(pos, g, edits, voxel_engine::Rev::START)
    }

    fn terrain_gen(
        b: &Blocks,
        h: i32,
        water: i32,
        shelf: Option<(i32, i32)>,
    ) -> FnGen<impl Fn(i32, i32) -> i32, impl Fn(i32, i32, i32) -> BlockId> {
        let (grass, dirt, stone, sand, water_id, air) = (b.grass, b.dirt, b.stone, b.sand, b.water, b.air);
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

    /// Reference: coarse-cell sweep per column from generator contract (parity oracle).
    fn reference_cells<G: TerrainGenerator>(r#gen: &G, wx: i32, wz: i32, cell: i32) -> Vec<BlockId> {
        let half = cell / 2;
        let ys: Vec<i32> = (0..DOMAIN_H / cell).map(|j| LOD_FLOOR_Y + j * cell + half).collect();
        let mut out = vec![AIR; ys.len()];
        r#gen.lod_column(wx, wz, &ys, &mut out);
        out
    }

    /// Reference implementation of the apply_edits POSITION rule, written
    /// per-cell (independent of input order by construction).
    fn reference_reduce(cells: &mut [BlockId], flat: &[(i32, i32, i32, BlockId)], fx: i32, fz: i32, cell: i32) {
        let half = cell / 2;
        for (j, slot) in cells.iter_mut().enumerate() {
            let centre = (fx + half, LOD_FLOOR_Y + j as i32 * cell + half, fz + half);
            if let Some(&(_, _, _, id)) = flat.iter().find(|&&(x, y, z, _)| (x, y, z) == centre) {
                *slot = id;
                continue;
            }
            let mut best: Option<((i32, i32, i32), BlockId)> = None;
            for &(x, y, z, id) in flat {
                if id == AIR || x < fx || x >= fx + cell || z < fz || z >= fz + cell || (y - LOD_FLOOR_Y).div_euclid(cell) != j as i32 {
                    continue;
                }
                if best.is_none_or(|(key, _)| (y, x, z) > key) {
                    best = Some(((y, x, z), id));
                }
            }
            if let Some((_, id)) = best {
                *slot = id;
            }
        }
    }

    /// Bottom-up flat cell list for global column (ix, iz), from the fused
    /// [`Section::column_runs`].
    fn column_ids(section: &Section, ix: usize, iz: usize) -> Vec<BlockId> {
        let mut out = Vec::new();
        for run in section.column_runs(ix, iz) {
            out.extend(std::iter::repeat_n(run.block, run.count as usize));
        }
        out
    }

    fn overlay_from_world(edits: &[(i32, i32, i32, BlockId)]) -> Vec<(ChunkCoord, Vec<(usize, BlockId)>)> {
        let mut map: std::collections::HashMap<ChunkCoord, Vec<(usize, BlockId)>> = Default::default();
        for &(x, y, z, id) in edits {
            let (c, l) = BlockCoord::new(x, y, z).split();
            map.entry(c).or_default().push((Chunk::index(l.lx(), l.ly(), l.lz()), id));
        }
        map.into_iter().collect()
    }

    // SectionPos / Quadrant tests.

    #[test]
    fn child_parent_is_an_involution_across_the_sign_boundary() {
        for &(x, z) in &[(0, 0), (1, 1), (-1, -1), (-1, 0), (5, -7), (-4, 3), (i32::MIN / 4, 9)] {
            let p = SectionPos { detail: Detail(5), x, z };
            for q in Quadrant::ALL {
                let c = p.child(q);
                assert_eq!(c.parent(), p, "child({q:?}).parent() != self at {x},{z}");
                assert_eq!(c.quadrant(), q, "quadrant disagrees with child index");
                assert_eq!(c.detail, Detail(4));
            }
        }
    }

    #[test]
    fn parent_uses_floor_division_not_toward_zero() {
        // Ensure -1 divides to -1 (floor), not 0 (toward zero).
        let p = SectionPos { detail: Detail(2), x: -1, z: -3 };
        assert_eq!(p.parent(), SectionPos { detail: Detail(3), x: -1, z: -2 });
    }

    // Extraction: column classes tests.

    /// Every terrain class extracts to a canonical column whose cells and
    /// exposed surface match the generator's own coarse sweep.
    #[test]
    fn extraction_matches_the_generator_sweep_for_every_class() {
        let b = blocks();
        let cases: [(&str, i32, i32, Option<(i32, i32)>); 6] = [
            ("uniform-air", 0, 0, None),
            ("deep-ground", 400, 0, None),
            ("water-covered", 30, 60, None),
            ("shore", 40, 40, None),
            ("overhang", 100, 0, Some((108, 112))),
            ("island", 64, 0, Some((120, 140))),
        ];
        for (name, h, water, shelf) in cases {
            let r#gen = terrain_gen(&b, h, water, shelf);
            let sec = extract(FINEST, &r#gen, &[]);
            let want = reference_cells(&r#gen, FINEST.min_x() + CELL / 2, FINEST.min_z() + CELL / 2, CELL);
            assert_eq!(column_ids(&sec, 0, 0), want, "{name}: cells");
            let ref_top = want.iter().rposition(|&c| c != AIR);
            match (sec.topmost_solid(0, 0), ref_top) {
                (Some((top_y, id)), Some(j)) => {
                    assert_eq!(top_y, LOD_FLOOR_Y + (j as i32 + 1) * CELL, "{name}: surface Y");
                    assert_eq!(id, want[j], "{name}: surface block");
                }
                (None, None) => {}
                (got, r) => panic!("{name}: topmost solid {got:?} vs reference cell {r:?}"),
            }
        }
    }

    // Extraction: edits tests.

    #[test]
    fn edit_reduction_matches_the_reference_reducer() {
        let b = blocks();
        let r#gen = terrain_gen(&b, 100, 0, None);
        let cell = CELL;
        let (fx, fz) = (FINEST.min_x(), FINEST.min_z());
        let (half, j) = (cell / 2, 20i32);
        let sample_y = LOD_FLOOR_Y + j * cell + half;
        let edits_world = [
            (fx + half, sample_y, fz + half, b.stone),
            (fx + half, LOD_FLOOR_Y + 5 * cell + half, fz + half, b.air),
            (fx + 1, LOD_FLOOR_Y + 6 * cell + half, fz + half, b.air),
        ];
        let overlay = overlay_from_world(&edits_world);

        let sec = extract(FINEST, &r#gen, &overlay);
        let got = column_ids(&sec, 0, 0);

        let mut want = reference_cells(&r#gen, fx + half, fz + half, cell);
        let flat = flatten_edits(&overlay);
        reference_reduce(&mut want, &flat, fx, fz, cell);
        assert_eq!(got, want);
        assert_eq!(got[j as usize], b.stone, "solid edit landed");
        assert_eq!(got[5], b.air, "air edit on the sample point cleared the cell");
    }

    /// Edits originate in hash maps with no ordering contract, so the
    /// coarse reduction must give ONE answer for every input permutation.
    #[test]
    fn edit_reduction_is_independent_of_input_permutation() {
        let b = blocks();
        let r#gen = terrain_gen(&b, 100, 0, None);
        let cell = CELL;
        let (fx, fz) = (FINEST.min_x(), FINEST.min_z());
        let (half, j) = (cell / 2, 20i32);
        let cell_y = LOD_FLOOR_Y + j * cell;
        let edits_world = [
            (fx, cell_y, fz, b.stone),
            (fx + 1, cell_y + 1, fz, b.grass),
            (fx, cell_y + 2, fz + 1, b.sand),
            (fx + 1, cell_y, fz + 1, b.air),
            (fx + half, LOD_FLOOR_Y + 5 * cell + half, fz + half, b.air),
        ];

        let mut want = reference_cells(&r#gen, fx + half, fz + half, cell);
        reference_reduce(&mut want, &flatten_edits(&overlay_from_world(&edits_world)), fx, fz, cell);
        assert_eq!(want[j as usize], b.sand, "the topmost (y,x,z) solid must win");
        assert_eq!(want[5], b.air, "the centre-sample air clears its cell");

        let mut order: Vec<usize> = (0..edits_world.len()).collect();
        permute(&mut order, 0, &mut |order| {
            let permuted: Vec<_> = order.iter().map(|&i| edits_world[i]).collect();
            let overlay: Vec<_> = permuted.iter().flat_map(|e| overlay_from_world(std::slice::from_ref(e))).collect();
            let sec = extract(FINEST, &r#gen, &overlay);
            assert_eq!(column_ids(&sec, 0, 0), want, "order {order:?}");
        });
    }

    fn permute(order: &mut Vec<usize>, k: usize, visit: &mut impl FnMut(&[usize])) {
        if k == order.len() {
            visit(order);
            return;
        }
        for i in k..order.len() {
            order.swap(k, i);
            permute(order, k + 1, visit);
            order.swap(k, i);
        }
    }

    // Determinism tests.

    #[test]
    fn extraction_is_deterministic() {
        let r#gen = sine(0xBEEF);
        let a = extract(FINEST, &r#gen, &[]);
        let b = extract(FINEST, &r#gen, &[]);
        for iz in 0..SECTION_N {
            for ix in 0..SECTION_N {
                assert_eq!(column_ids(&a, ix, iz), column_ids(&b, ix, iz), "same seed + pos must extract bit-identically");
            }
        }
    }

    #[test]
    fn exposed_surface_matches_sample_coarse_on_sinehills() {
        for &seed in &[1i64, 7, 42, 0xABCD] {
            let r#gen = sine(seed);
            let sec = extract(FINEST, &r#gen, &[]);
            for &(ix, iz) in &[(0usize, 0usize), (5, 9), (17, 3), (31, 31)] {
                let (wx, wz) = (FINEST.min_x() + ix as i32 * CELL + CELL / 2, FINEST.min_z() + iz as i32 * CELL + CELL / 2);
                let want = reference_cells(&r#gen, wx, wz, CELL);
                assert_eq!(column_ids(&sec, ix, iz), want, "seed {seed} col {ix},{iz}");
            }
        }
    }

    // Roundtrip tests: column_ids(extract(...)) must match the unchanged
    // sampling stage (gen.lod_column + apply_edits) at every ring.

    /// Reachable failure: fails if per-brick RLE construction or the
    /// column_runs/topmost_solid accessor layer drops, duplicates, or
    /// mis-orders a cell relative to the untouched sampling stage, at any
    /// ring including the padded coarse rings.
    #[test]
    fn roundtrip_every_column_matches_the_sampling_stage_at_every_ring() {
        let b = blocks();
        let r#gen = terrain_gen(&b, 200, 40, Some((260, 280)));
        let k = FINEST_DETAIL.0;
        for dk in 0..=6 {
            let pos = SectionPos { detail: Detail(k + dk), x: 0, z: 0 };
            let cell = pos.cell_size();
            let sec = extract(pos, &r#gen, &[]);
            for &(ix, iz) in &[(0usize, 0usize), (5, 9), (15, 15), (16, 0), (31, 31)] {
                let (wx, wz) = (pos.min_x() + ix as i32 * cell + cell / 2, pos.min_z() + iz as i32 * cell + cell / 2);
                let want = reference_cells(&r#gen, wx, wz, cell);
                // At the coarse rings (n_cells < 16) the single brick pads
                // its tail with AIR; compare only the real cells.
                let got = column_ids(&sec, ix, iz);
                assert_eq!(&got[..want.len()], want.as_slice(), "detail {dk} col {ix},{iz}");
                assert!(got[want.len()..].iter().all(|&c| c == AIR), "detail {dk} col {ix},{iz}: brick padding must be AIR");
            }
        }
    }

    /// Reachable failure: fails if any horizontal-axis term in `cell_index`,
    /// `Section::extract`'s quadrant/local-column mapping, or `column_runs`
    /// swaps x and z — a marker only at local column (3, 11) would then
    /// surface at the detectably different (11, 3). A height-only generator
    /// (as used by every other fixture here) cannot catch this: it is
    /// column-position-blind by construction.
    #[test]
    fn roundtrip_x_z_asymmetric_marker_column_stays_put() {
        let b = blocks();
        let cell = CELL;
        let r#gen = FnGen {
            h: |_, _| 0,
            b: move |wx: i32, y: i32, wz: i32| {
                if !(200..204).contains(&y) {
                    return AIR;
                }
                let (lx, lz) = (wx.div_euclid(cell).rem_euclid(32), wz.div_euclid(cell).rem_euclid(32));
                if lx == 3 && lz == 11 { b.stone } else { AIR }
            },
            surf: b.air,
            deep: b.air,
        };
        let sec = extract(FINEST, &r#gen, &[]);
        assert!(column_ids(&sec, 3, 11).iter().any(|&c| c == b.stone), "marker missing at its true column (3, 11)");
        assert!(column_ids(&sec, 11, 3).iter().all(|&c| c == AIR), "marker leaked to the transposed column (11, 3)");
    }

    /// Reachable failure: fails if the RLE slicer or per-brick construction
    /// mishandles a run that spans exactly the boundary between two bricks
    /// (cell 16), e.g. double-counting or dropping a cell at the seam.
    #[test]
    fn roundtrip_edit_straddling_a_brick_y_boundary() {
        let b = blocks();
        let r#gen = terrain_gen(&b, 0, 0, None); // all-air baseline
        let cell = CELL;
        // Cell index 16 is the first brick's ceiling / second brick's floor.
        let wy = LOD_FLOOR_Y + 16 * cell + cell / 2;
        let edits_world = [(FINEST.min_x() + cell / 2, wy, FINEST.min_z() + cell / 2, b.stone)];
        let overlay = overlay_from_world(&edits_world);
        let sec = extract(FINEST, &r#gen, &overlay);
        let mut want = reference_cells(&r#gen, FINEST.min_x() + cell / 2, FINEST.min_z() + cell / 2, cell);
        reference_reduce(&mut want, &flatten_edits(&overlay), FINEST.min_x(), FINEST.min_z(), cell);
        assert_eq!(column_ids(&sec, 0, 0), want);
        assert_eq!(want[16], b.stone, "edit landed at the brick boundary cell");
    }

}
