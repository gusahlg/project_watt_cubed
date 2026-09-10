//! Per-seed height pyramid: a sparse mipmap of height envelopes (min/max bounds per cell)
//! used for LOD selection and occlusion culling. For each detail level from finest to coarsest,
//! records the ground height range, relief error (max - min), and surface colour per cell.
//!
//! Correctness by construction: the finest level samples terrain at the same cell centres
//! that the LOD extractor uses, so recorded bounds are exact. Each coarser level is formed
//! by taking min(lo) and max(hi) of its four children (pure reduction), ensuring parent
//! bounds always contain child bounds and error monotonically increases up the pyramid.
//! No resampling per level — children fully determine parents where coverage overlaps.
//!
//! The bake is immutable once computed: edits are applied elsewhere as a runtime overlay.
use voxel_engine::{Color, DVec3};

use crate::block::registry::AIR;
use crate::coord::ChunkCoord;
use crate::ident::Detail;
use super::generation::TerrainGenerator;
use super::lod;
use super::metric::HeightEnvelope;
use super::section::{FINEST_DETAIL, LOD_FLOOR_Y, SECTION_N, SectionPos, section_span};
use super::summary::{CellError, CellSummary};

/// The world region and detail band a bake covers. The finest level is always
/// [`FINEST_DETAIL`] (the LOD extractor's own granularity); `coarsest` matches the
/// pyramid's outermost ring. `half_m` is the half-side of the baked square in
/// metres — cells outside it fall back to worst case ([`HeightMip::summary`]).
#[derive(Clone, Copy, Debug)]
pub(in crate::world) struct BakeExtent {
    half_m: i32,
    finest: Detail,
    coarsest: Detail,
}

impl BakeExtent {
    pub fn new(half_m: i32, coarsest: Detail) -> BakeExtent {
        debug_assert!(half_m > 0, "extent half-side must be positive");
        debug_assert!(FINEST_DETAIL <= coarsest, "finest detail exceeds coarsest");
        BakeExtent { half_m, finest: FINEST_DETAIL, coarsest }
    }
}

/// One baked cell: ground-height bounds (lo, hi), relief (hi - lo), and palette-average colour.
/// Also the shape [`resample_cell`] produces for the edit overlay (`World::section_overlay`),
/// so a cache hit there is a drop-in replacement for a baked cell at the same position.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(in crate::world) struct MipCell {
    pub(in crate::world) lo: f32,
    pub(in crate::world) hi: f32,
    pub(in crate::world) color: Color,
}

impl MipCell {
    /// The cell's relief (hi - lo). Zero for flat terrain, allowing coarse drawing at any distance.
    fn err(&self) -> f32 {
        self.hi - self.lo
    }
}

/// One detail level's dense grid of cells, row-major over `[x0, x0+nx) × [z0, z0+nz)`
/// in section-grid coords at this `detail`.
#[derive(Clone, PartialEq, Debug)]
struct MipLevel {
    detail: Detail,
    x0: i32,
    z0: i32,
    nx: usize,
    nz: usize,
    cells: Vec<MipCell>,
}

impl MipLevel {
    fn get(&self, x: i32, z: i32) -> Option<&MipCell> {
        let (lx, lz) = (x - self.x0, z - self.z0);
        if lx < 0 || lz < 0 || lx as usize >= self.nx || lz as usize >= self.nz {
            return None;
        }
        Some(&self.cells[lz as usize * self.nx + lx as usize])
    }
}

/// The max-mip: one [`MipLevel`] per detail from `finest` to `coarsest`, indexed by (detail - finest).
#[derive(Clone, PartialEq, Debug)]
pub(in crate::world) struct HeightMip {
    finest: Detail,
    coarsest: Detail,
    levels: Vec<MipLevel>,
}

impl HeightMip {
    /// Build a sparse pyramid: coverage halves per level (geometric base-2), so cell count
    /// per level stays roughly constant. Finest detail covers only the inner rings; coarser
    /// levels extend outward. This avoids quadratic cost growth.
    ///
    /// Each level is built by reducing four finer children (min/max bounds) where they exist,
    /// or sampling the generator at this level's stride for areas the finer level doesn't cover.
    /// Finer children on the boundary are included in the parent to preserve containment.
    pub fn bake<G: TerrainGenerator + ?Sized>(terra: &G, colors: &[Color], extent: BakeExtent) -> HeightMip {
        let (finest, coarsest) = (extent.finest, extent.coarsest);
        let mut levels: Vec<MipLevel> = Vec::with_capacity((coarsest.0 - finest.0 + 1) as usize);
        for k in finest.0..=coarsest.0 {
            let detail = Detail(k);
            // Coverage radius halves per level below the coarsest (which spans the
            // whole extent). Aligned to the absolute section grid so a parent's
            // children map by index doubling.
            let radius = extent.half_m >> (coarsest.0 - k);
            let span = section_span(detail);
            let x0 = (-radius).div_euclid(span);
            let z0 = (-radius).div_euclid(span);
            let nx = (radius.div_euclid(span) - x0 + 1) as usize;
            let nz = (radius.div_euclid(span) - z0 + 1) as usize;
            let child = levels.last();
            levels.push(build_level(terra, colors, detail, x0, z0, nx, nz, child));
        }
        HeightMip { finest, coarsest, levels }
    }

    /// The summary for a cell: the baked envelope + recorded ε when the cell sits in
    /// the extent, else the worst-case fallback (full domain envelope, `2^detail`
    /// error) — a section beyond the bake is treated as maximally uncertain.
    pub fn summary(&self, cell: SectionPos) -> CellSummary {
        if let Some(c) = self.cell(cell) {
            CellSummary { env: HeightEnvelope::new(c.lo, c.hi), err: CellError::from_metres(c.err()) }
        } else {
            CellSummary {
                env: HeightEnvelope::new(0.0, super::section::DOMAIN_H as f32),
                err: CellError::worst_case(cell.detail),
            }
        }
    }

    pub(in crate::world) fn allocated_bytes(&self) -> usize {
        self.levels
            .iter()
            .map(|lvl| {
                std::mem::size_of::<MipLevel>() + std::mem::size_of_val(lvl.cells.as_slice())
            })
            .sum()
    }

    /// The palette-average colour of a baked cell, if inside the extent; `None` beyond the bake.
    pub fn color(&self, cell: SectionPos) -> Option<Color> {
        self.cell(cell).map(|c| c.color)
    }

    fn cell(&self, pos: SectionPos) -> Option<&MipCell> {
        if pos.detail < self.finest || pos.detail > self.coarsest {
            return None;
        }
        self.levels[(pos.detail.0 - self.finest.0) as usize].get(pos.x, pos.z)
    }

    /// The baked ground-height envelope `(lo, hi)` of a cell, if inside the extent.
    /// `None` beyond the bake means the caller cannot skip.
    pub fn relief_band(&self, cell: SectionPos) -> Option<(f32, f32)> {
        self.cell(cell).map(|c| (c.lo, c.hi))
    }

    /// Returns `true` if `cell` is definitely occluded from `eye` by nearer terrain.
    /// Safe to use only for draw-time culling (false positive = hole). Never gate
    /// selection or streaming on this result.
    ///
    /// Uses a conservative approach: march from eye toward cell top, checking if any
    /// blocker's floor (`lo` height) intersects the line-of-sight. Reads blockers at
    /// the coarsest level (widest coverage), so error favors under-culling (safe but misses
    /// some occlusion). Finer-level blocking is possible future refinement.
    #[allow(dead_code)] // draw-time occlusion culling, not yet wired to the renderer
    pub fn occludes(&self, eye: DVec3, cell: SectionPos) -> bool {
        let Some(target) = self.cell(cell) else { return false }; // unbaked, can't prove occlusion
        let span = cell.span() as f64;
        let (cx, cz) = (cell.min_x() as f64 + span * 0.5, cell.min_z() as f64 + span * 0.5);
        let (dx, dz) = (cx - eye.x, cz - eye.z);
        let horiz = (dx * dx + dz * dz).sqrt();
        // Nothing between eye and cell (adjacent or overhead) means it can't be occluded.
        if horiz < span {
            return false;
        }
        // March XZ from eye toward the cell's TOP; if any blocker `lo` rises above
        // the sightline to that top, every point of the cell is at or below it and
        // the whole cell is hidden. Step ~one coarsest cell — finer is pointless
        // when blockers are read at the coarsest level.
        let top = target.hi as f64;
        let coarse_span = section_span(self.coarsest) as f64;
        let steps = ((horiz / coarse_span).ceil() as usize).clamp(2, OCCLUDE_MARCH_CAP);
        for i in 1..steps {
            let t = i as f64 / steps as f64;
            let (sx, sz) = (eye.x + dx * t, eye.z + dz * t);
            let y = eye.y + (top - eye.y) * t;
            if let Some(lo) = self.coarse_lo(sx, sz) && lo as f64 >= y {
                return true;
            }
        }
        false
    }

    /// Blocker floor at a world XZ, read at the coarsest level.
    /// `None` outside the baked grid (safe: missing data never causes false occlusion).
    fn coarse_lo(&self, wx: f64, wz: f64) -> Option<f32> {
        let lvl = self.levels.last()?;
        let span = section_span(lvl.detail);
        let gx = (wx.floor() as i32).div_euclid(span);
        let gz = (wz.floor() as i32).div_euclid(span);
        lvl.get(gx, gz).map(|c| c.lo)
    }

}

/// Max march samples to prevent unbounded loops on very distant cells.
const OCCLUDE_MARCH_CAP: usize = 64;

/// Sample colour every 4th cell (reduces cost without visible loss for coarse tints).
const COLOR_STRIDE: i32 = 4;

/// Build one level over its grid. Cells with all four children take min/max from them;
/// cells without full coverage are sampled from the generator and merged with any existing children.
#[allow(clippy::too_many_arguments)] // mip level build takes grid bounds and optional child separately
fn build_level<G: TerrainGenerator + ?Sized>(
    terra: &G,
    colors: &[Color],
    detail: Detail,
    x0: i32,
    z0: i32,
    nx: usize,
    nz: usize,
    child: Option<&MipLevel>,
) -> MipLevel {
    let mut cells = Vec::with_capacity(nx * nz);
    for sz in 0..nz {
        for sx in 0..nx {
            let (ax, az) = (x0 + sx as i32, z0 + sz as i32);
            // The four finer children covering this section (index doubling), each
            // present only if the finer level's coverage reaches here.
            let kids: [Option<&MipCell>; 4] = std::array::from_fn(|q| {
                child.and_then(|c| c.get(2 * ax + (q & 1) as i32, 2 * az + (q >> 1) as i32))
            });
            let cell = if kids.iter().all(Option::is_some) {
                merge(kids.map(|k| *k.unwrap()))
            } else {
                let mut c = sample_section(terra, colors, detail, ax, az);
                for k in kids.iter().flatten() {
                    c.lo = c.lo.min(k.lo);
                    c.hi = c.hi.max(k.hi);
                }
                c
            };
            cells.push(cell);
        }
    }
    MipLevel { detail, x0, z0, nx, nz, cells }
}

/// Sample one section at absolute grid `(ax, az)`: max/min height and palette-average colour
/// at the same points [`Section::extract`] samples, ensuring `hi` bounds drawable terrain.
fn sample_section<G: TerrainGenerator + ?Sized>(
    terra: &G,
    colors: &[Color],
    detail: Detail,
    ax: i32,
    az: i32,
) -> MipCell {
    let cell = lod::cell(detail);
    let half = cell / 2;
    let span = section_span(detail);
    let (min_x, min_z) = (ax * span, az * span);
    let mut fold = MipFold::new();
    for iz in 0..SECTION_N as i32 {
        for ix in 0..SECTION_N as i32 {
            let wx = min_x + ix * cell + half;
            let wz = min_z + iz * cell + half;
            // Height must sample every cell; colour is an average, sampled on a stride to save cost.
            fold.height(terra.height(wx, wz) as f32);
            if ix % COLOR_STRIDE == 0 && iz % COLOR_STRIDE == 0 {
                fold.color(colors[terra.surface_at(wx, wz).0 as usize]);
            }
        }
    }
    fold.finish()
}

/// The one lo/hi/mean-colour fold both the pure bake ([`sample_section`]) and
/// the edit-folded resample ([`resample_cell`]) reduce their columns through,
/// so the two can never round or accumulate differently.
struct MipFold {
    lo: f32,
    hi: f32,
    r: u64,
    g: u64,
    b: u64,
    n: u64,
}

impl MipFold {
    fn new() -> Self {
        Self { lo: f32::INFINITY, hi: f32::NEG_INFINITY, r: 0, g: 0, b: 0, n: 0 }
    }
    fn height(&mut self, h: f32) {
        self.lo = self.lo.min(h);
        self.hi = self.hi.max(h);
    }
    fn color(&mut self, c: Color) {
        self.r += c.r as u64;
        self.g += c.g as u64;
        self.b += c.b as u64;
        self.n += 1;
    }
    fn finish(self) -> MipCell {
        let Self { lo, hi, r, g, b, n } = self;
        MipCell { lo, hi, color: Color::new((r / n) as u8, (g / n) as u8, (b / n) as u8, 255) }
    }
}

/// Merge four cells: hi = max, lo = min, colour = mean.
fn merge(kids: [MipCell; 4]) -> MipCell {
    let lo = kids.iter().fold(f32::INFINITY, |m, c| m.min(c.lo));
    let hi = kids.iter().fold(f32::NEG_INFINITY, |m, c| m.max(c.hi));
    let mean = |f: fn(&Color) -> u8| (kids.iter().map(|c| f(&c.color) as u32).sum::<u32>() / 4) as u8;
    MipCell { lo, hi, color: Color::new(mean(|c| c.r), mean(|c| c.g), mean(|c| c.b), 255) }
}

/// Edit-folded cell for a section whose footprint has a live edit: derived via
/// [`Section::extract`], the SAME generator-plus-edits fold the mesh path itself
/// trusts, so the overlay can never disagree with what's actually drawn. Correct
/// even when a dig removed the topmost block the bake recorded — unlike a bake
/// patch (which only has the aggregate lo/hi, not per-column detail, and so can
/// only safely grow, never shrink), this resamples every column fresh.
///
/// Only called for edited cells (bounded, rare); the immutable bake covers
/// everything else at zero cost.
pub(in crate::world) fn resample_cell<G: TerrainGenerator + ?Sized>(
    pos: SectionPos,
    terra: &G,
    edits: &[(ChunkCoord, Vec<(usize, crate::block::registry::BlockId)>)],
    colors: &[Color],
) -> MipCell {
    // Sample columns directly (generator + edit fold), with no RLE Section
    // built in between: this used to run a full `Section::extract` — four
    // brick stacks and thousands of transient run vectors — per edited cell,
    // only to read one topmost-solid value per column back out. The column
    // fold below is pinned equivalent to the extract-based reference by test.
    let cell = pos.cell_size();
    let half = cell / 2;
    let ys = crate::world::section::cell_centers(pos);
    let flat = crate::world::section::flatten_edits(edits);
    let mut column = vec![AIR; ys.len()];
    let mut fold = MipFold::new();
    for iz in 0..SECTION_N as i32 {
        for ix in 0..SECTION_N as i32 {
            let (fx, fz) = (pos.min_x() + ix * cell, pos.min_z() + iz * cell);
            terra.lod_column(fx + half, fz + half, &ys, &mut column);
            crate::world::section::apply_edits(&mut column, &flat, fx, fz, cell);
            // Topmost non-air cell → the world-space TOP of its run (exclusive),
            // exactly what `Section::topmost_solid` reports from stored runs.
            let (h, block) = match column.iter().rposition(|&id| id != AIR) {
                Some(j) => (LOD_FLOOR_Y as f32 + ((j as i32 + 1) * cell) as f32, column[j]),
                None => (LOD_FLOOR_Y as f32, AIR), // dug through to the floor: rare, conservative
            };
            fold.height(h);
            fold.color(colors[block.0 as usize]);
        }
    }
    fold.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::BlockRegistry;
    use crate::world::generation::Terrain;
    use crate::world::section::Section;

    /// The extract-based reference `resample_cell` replaced: build the full
    /// RLE section and read `topmost_solid` per column. The fused direct
    /// sampler must match it exactly for every detail and edit pattern.
    fn resample_reference(
        pos: SectionPos,
        terra: &Terrain,
        edits: &[(crate::coord::ChunkCoord, Vec<(usize, crate::block::registry::BlockId)>)],
        colors: &[Color],
    ) -> MipCell {
        use crate::block::registry::AIR;
        let section = Section::extract(pos, terra, edits, voxel_engine::Rev::START);
        let mut fold = MipFold::new();
        for iz in 0..SECTION_N {
            for ix in 0..SECTION_N {
                let (h, block) = match section.topmost_solid(ix, iz) {
                    Some((top, block)) => (top as f32, block),
                    None => (LOD_FLOOR_Y as f32, AIR),
                };
                fold.height(h);
                fold.color(colors[block.0 as usize]);
            }
        }
        fold.finish()
    }

    #[test]
    fn fused_resample_matches_the_extract_reference() {
        let (reg, g) = terra(0xC0FFEE);
        let colors = reg.color_snapshot();
        let cs = 16usize;
        let idx = |x: usize, y: usize, z: usize| x + z * cs + y * cs * cs;
        let stone = reg.id_by_label("rock").unwrap();
        let edits = [
            (crate::coord::ChunkCoord::new(0, 5, 0), vec![(idx(2, 6, 2), stone), (idx(3, 1, 5), crate::block::registry::AIR)]),
            (crate::coord::ChunkCoord::new(1, 8, 1), vec![(idx(9, 9, 9), stone)]),
        ];
        use crate::ident::Detail;
        for detail in [FINEST_DETAIL, Detail(FINEST_DETAIL.0 + 3), Detail(FINEST_DETAIL.0 + 7)] {
            for pos in [
                SectionPos { detail, x: 0, z: 0 },
                SectionPos { detail, x: -1, z: 2 },
            ] {
                for edit_set in [&[][..], &edits[..]] {
                    let want = resample_reference(pos, &g, edit_set, &colors);
                    let got = resample_cell(pos, &g, edit_set, &colors);
                    assert_eq!((want.lo, want.hi, want.color), (got.lo, got.hi, got.color),
                        "fused resample diverged at {pos:?} (edits: {})", !edit_set.is_empty());
                }
            }
        }
    }

    fn terra(seed: i64) -> (BlockRegistry, Terrain) {
        let mut registry = BlockRegistry::with_builtins();
        let terrain = Terrain::new(&mut registry, 20.0, seed);
        (registry, terrain)
    }

    /// A small extent keeps the property tests fast; the timing test uses the real one.
    fn small(reg: &BlockRegistry, g: &Terrain) -> HeightMip {
        HeightMip::bake(g, &reg.color_snapshot(), BakeExtent::new(2048, Detail(FINEST_DETAIL.0 + 3)))
    }

    /// Recorded `hi` bounds every ground height sampled by the LOD extractor.
    #[test]
    fn e1_finest_hi_bounds_every_sampled_height() {
        let (reg, g) = terra(7);
        let mip = small(&reg, &g);
        let lvl = &mip.levels[0];
        let cell = 1i32 << lvl.detail.0;
        let half = cell / 2;
        let span = section_span(lvl.detail);
        for sz in 0..lvl.nz.min(4) {
            for sx in 0..lvl.nx.min(4) {
                let c = lvl.get(lvl.x0 + sx as i32, lvl.z0 + sz as i32).unwrap();
                let (min_x, min_z) = ((lvl.x0 + sx as i32) * span, (lvl.z0 + sz as i32) * span);
                for iz in 0..SECTION_N as i32 {
                    for ix in 0..SECTION_N as i32 {
                        let h = g.height(min_x + ix * cell + half, min_z + iz * cell + half) as f32;
                        assert!(c.hi >= h && c.lo <= h, "cell env [{},{}] excludes {h}", c.lo, c.hi);
                    }
                }
            }
        }
    }

    /// Parent bounds contain child bounds and parent error is at least the child error.
    #[test]
    fn m2_parent_env_and_error_dominate_children() {
        let (reg, g) = terra(11);
        let mip = small(&reg, &g);
        let mut checked = 0;
        for w in mip.levels.windows(2) {
            let (child, parent) = (&w[0], &w[1]);
            for sz in 0..parent.nz {
                for sx in 0..parent.nx {
                    let (ax, az) = (parent.x0 + sx as i32, parent.z0 + sz as i32);
                    let p = parent.get(ax, az).unwrap();
                    for q in 0..4 {
                        let Some(c) = child.get(2 * ax + (q & 1), 2 * az + (q >> 1)) else { continue };
                        assert!(p.lo <= c.lo && p.hi >= c.hi, "parent env does not contain child");
                        assert!(p.err() >= c.err(), "parent ε {} < child ε {}", p.err(), c.err());
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 0, "no parent/child pairs exercised");
    }

    /// Coverage shrinks per level, keeping cell count bounded. Sections beyond the finest
    /// coverage use worst-case error, but their coarser ancestors are baked.
    #[test]
    fn sparse_pyramid_shape() {
        let (reg, g) = terra(5);
        let mip = small(&reg, &g);
        // Each finer level's coverage radius is smaller: its cell count does not
        // exceed the coarser level's (dense would be 4× larger).
        for w in mip.levels.windows(2) {
            assert!(w[0].cells.len() <= w[1].cells.len(), "finer level not sparser");
        }
        // Section beyond finest coverage uses worst_case at finest detail, but coarser ancestor is baked.
        let finest = &mip.levels[0];
        let past = SectionPos { detail: FINEST_DETAIL, x: finest.x0 + finest.nx as i32 + 1, z: 0 };
        assert_eq!(
            mip.summary(past).err.get(),
            CellError::worst_case(FINEST_DETAIL).get(),
            "finest section past coverage should be worst_case"
        );
        // Its coarsest ancestor (same ground, largest coverage) is inside the bake.
        let anc = SectionPos { detail: mip.coarsest, x: past.x >> (mip.coarsest.0 - FINEST_DETAIL.0), z: 0 };
        assert!(mip.color(anc).is_some(), "coarse ancestor should be baked");
    }

    /// Determinism: a bake is a pure function of the seed — two are identical.
    #[test]
    fn deterministic_in_seed() {
        let (reg, g) = terra(19);
        let a = small(&reg, &g);
        let b = small(&reg, &g);
        assert!(a == b, "two bakes of one seed differ");
    }

    /// The worst-case fallback covers cells outside the baked extent.
    #[test]
    fn summary_falls_back_beyond_extent() {
        let (reg, g) = terra(3);
        let mip = small(&reg, &g);
        let far = SectionPos { detail: FINEST_DETAIL, x: 1_000_000, z: 0 };
        let s = mip.summary(far);
        assert_eq!(s.err.get(), CellError::worst_case(FINEST_DETAIL).get());
    }

    /// Build a single-level mip from a per-cell `(lo, hi)` closure — full control
    /// over the terrain silhouette for the occlusion property tests. `occludes`
    /// reads blockers at the coarsest level (here, the only one) and targets a
    /// cell's own level, so one level exercises the whole march.
    fn hand_mip(
        detail: Detail,
        x0: i32,
        z0: i32,
        nx: usize,
        nz: usize,
        f: impl Fn(i32, i32) -> (f32, f32),
    ) -> HeightMip {
        let mut cells = Vec::with_capacity(nx * nz);
        for sz in 0..nz {
            for sx in 0..nx {
                let (lo, hi) = f(x0 + sx as i32, z0 + sz as i32);
                cells.push(MipCell { lo, hi, color: Color::new(0, 0, 0, 255) });
            }
        }
        HeightMip { finest: detail, coarsest: detail, levels: vec![MipLevel { detail, x0, z0, nx, nz, cells }] }
    }

    const OD: Detail = FINEST_DETAIL; // a modest detail so a few cells span a wide march

    /// Flat terrain above the eye occludes nothing.
    #[test]
    fn occlude_flat_world_culls_nothing() {
        let mip = hand_mip(OD, -8, -8, 16, 16, |_, _| (0.0, 0.0));
        for gx in -8..8 {
            let cell = SectionPos { detail: OD, x: gx, z: 0 };
            assert!(
                !mip.occludes(DVec3::new(0.0, 50.0, 0.0), cell),
                "flat world culled cell x={gx}"
            );
        }
    }

    /// A wall between eye and target occludes the target.
    #[test]
    fn occlude_wall_hides_cell_behind() {
        let span = section_span(OD);
        // Wall at grid x==4 rises to 100; everything else is ground level 0.
        let mip = hand_mip(OD, 0, -4, 12, 8, |x, _| if x == 4 { (100.0, 100.0) } else { (0.0, 0.0) });
        let behind = SectionPos { detail: OD, x: 8, z: 0 };
        // Eye just above the ground on the near side of the wall.
        let eye = DVec3::new(0.5 * span as f64, 5.0, 0.5 * span as f64);
        assert!(mip.occludes(eye, behind), "cell behind the wall not culled");
    }

    /// A cell whose top clears all blockers is never culled (no holes).
    #[test]
    fn occlude_no_false_positive_when_top_clears_ridge() {
        let span = section_span(OD);
        // Ridge floor 100 at x==4; a tall target at x==8 reaching 200.
        let mip = hand_mip(OD, 0, -4, 12, 8, |x, _| match x {
            4 => (100.0, 100.0),
            8 => (0.0, 200.0),
            _ => (0.0, 0.0),
        });
        let tower = SectionPos { detail: OD, x: 8, z: 0 };
        // Eye high enough that the sightline to the tower top stays above the ridge.
        let eye = DVec3::new(0.5 * span as f64, 150.0, 0.5 * span as f64);
        assert!(!mip.occludes(eye, tower), "tower clearing the ridge was culled (hole)");
    }

    /// Occlusion only triggers when a blocker actually blocks the sightline to the target.
    #[test]
    fn occlude_never_culls_without_a_real_blocker() {
        let span = section_span(OD) as f64;
        // A diagonal ridge so blockers vary with position.
        let mip = hand_mip(OD, -16, -16, 32, 32, |x, z| {
            let h = if (x + z).rem_euclid(7) == 0 { 80.0 } else { 5.0 };
            (h, h)
        });
        for &ey in &[0.0f64, 20.0, 90.0, 200.0] {
            for tx in [-10, -3, 4, 11] {
                for tz in [-8, 0, 9] {
                    let cell = SectionPos { detail: OD, x: tx, z: tz };
                    let Some((_, hi)) = mip.relief_band(cell) else { continue };
                    let eye = DVec3::new(0.0, ey, 0.0);
                    if mip.occludes(eye, cell) {
                        // Re-derive: verify a blocker actually crosses the sightline.
                        let (cx, cz) =
                            (cell.min_x() as f64 + span * 0.5, cell.min_z() as f64 + span * 0.5);
                        let horiz = (cx * cx + cz * cz).sqrt();
                        let steps = ((horiz / span).ceil() as usize).clamp(2, 64);
                        let blocked = (1..steps).any(|i| {
                            let t = i as f64 / steps as f64;
                            let y = ey + (hi as f64 - ey) * t;
                            mip.coarse_lo(cx * t, cz * t).is_some_and(|lo| lo as f64 >= y)
                        });
                        assert!(blocked, "occludes fired with no blocker (eye_y={ey}, {tx},{tz})");
                    }
                }
            }
        }
    }

    /// Determinism: `occludes` is a pure function of `(eye, cell)`.
    #[test]
    fn occlude_deterministic() {
        let mip = hand_mip(OD, -8, -8, 16, 16, |x, _| if x == 2 { (60.0, 60.0) } else { (0.0, 0.0) });
        let eye = DVec3::new(3.0, 8.0, 1.0);
        let cell = SectionPos { detail: OD, x: 6, z: 0 };
        assert_eq!(mip.occludes(eye, cell), mip.occludes(eye, cell));
    }

    /// `resample_cell`'s lo/hi is exactly the min/max of `topmost_solid` over every
    /// column `Section::extract` produces — the aggregation itself is correct,
    /// independent of whether any edit is present.
    #[test]
    fn resample_cell_matches_an_independent_reduction_over_extracted_columns() {
        let (reg, g) = terra(23);
        let colors = reg.color_snapshot();
        let pos = SectionPos { detail: FINEST_DETAIL, x: 3, z: -2 };
        let got = resample_cell(pos, &g, &[], &colors);

        let section = Section::extract(pos, &g, &[], voxel_engine::Rev::START);
        let (mut want_lo, mut want_hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for iz in 0..SECTION_N {
            for ix in 0..SECTION_N {
                let h = section.topmost_solid(ix, iz).map_or(LOD_FLOOR_Y as f32, |(top, _)| top as f32);
                want_lo = want_lo.min(h);
                want_hi = want_hi.max(h);
            }
        }
        assert_eq!(got.lo, want_lo);
        assert_eq!(got.hi, want_hi);
    }

    /// An edit changes what `resample_cell` reports, which an
    /// immutable bake — sampled once at startup — can never reflect. Building a
    /// column up past the unedited height is the deterministic direction (digging
    /// down depends on what the generator puts underneath); the defect is
    /// symmetric, so this direction is sufficient to prove the overlay is live.
    #[test]
    fn resample_cell_folds_edits_the_bake_never_sees() {
        use crate::coord::BlockCoord;
        use crate::world::chunk::Chunk;

        let (reg, g) = terra(23);
        let stone = reg.id_by_label("rock").unwrap();
        let colors = reg.color_snapshot();
        let pos = SectionPos { detail: FINEST_DETAIL, x: 100, z: -50 };
        let cell = pos.cell_size();
        let (wx, wz) = (pos.min_x() + cell / 2, pos.min_z() + cell / 2); // column (0,0)'s sample point

        let unedited = resample_cell(pos, &g, &[], &colors);
        let built_y = unedited.hi as i32 + 40;
        let (coord, local) = BlockCoord::new(wx, built_y, wz).split();
        let index = Chunk::index(local.lx(), local.ly(), local.lz());
        let edits = vec![(coord, vec![(index, stone)])];

        let edited = resample_cell(pos, &g, &edits, &colors);
        assert!(
            edited.hi >= built_y as f32,
            "a built-up column must raise the cell's recorded height to at least where it was built"
        );
        assert!(
            edited.hi > unedited.hi,
            "the edit must actually change the result — proving the overlay, not the bake, saw it"
        );
    }

    /// Benchmark 5-level and 7-level bakes to verify performance stays within budget.
    /// Run: `cargo test --release -- --ignored --nocapture bake_timing`.
    #[test]
    #[ignore]
    fn bake_timing_at_target_extent() {
        let (reg, g) = terra(1);
        // (half_m = outer_m, coarsest = FINEST_DETAIL + (levels-1)*step, step=1).
        for (name, half_m, coarsest) in [
            ("5-level", 3072, Detail(FINEST_DETAIL.0 + 4)),
            ("7-level", 12288, Detail(FINEST_DETAIL.0 + 6)),
        ] {
            let t = std::time::Instant::now();
            let mip = HeightMip::bake(&g, &reg.color_snapshot(), BakeExtent::new(half_m, coarsest));
            let dt = t.elapsed();
            let total: usize = mip.levels.iter().map(|l| l.cells.len()).sum();
            let per: Vec<usize> = mip.levels.iter().map(|l| l.cells.len()).collect();
            eprintln!("{name}: {dt:?}  levels={} total_cells={total} per_level={per:?}", mip.levels.len());
        }
    }
}
