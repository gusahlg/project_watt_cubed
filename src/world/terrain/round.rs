//! Round worlds on curved charts: Verdance's giant forests and the Hollow's two surfaces (an icy
//! crust outside, crystal forests hanging toward the centre inside). Each is painted in its atlas's
//! storage cells, where storage `+Y` is the chart's up, so terrain is a height field per chart
//! column. Heights come from 3-D noise at the physical point on the datum sphere above the column,
//! so the surface is continuous across chart seams. Pure in `(seed, cell)`.

use std::sync::Arc;

use glam::DVec3;

use super::noise::{hash2, perlin3, unit};
use super::Materials;
use crate::block::registry::{AIR, BlockId};
use crate::space::atlas::{Atlas, Patch};
use crate::world::chunk::{CHUNK_SIZE, CHUNK_VOLUME, Chunk, ChunkData};

/// What a round world looks like.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Style {
    /// Rolling forested hills, ranges and deep green valleys; giant trees.
    Verdant,
    /// The Hollow's outer crust: frost and ice, crags and ice spires.
    HollowOuter,
    /// The Hollow's inner surface: violet ground and crystal spires reaching for the centre.
    HollowInner,
}

/// One round world's painter.
pub struct Round {
    pub atlas: Atlas,
    seed: u32,
    style: Style,
    m: Arc<Materials>,
}

/// Grid spacing of tree / spire sites, in cells.
const SITE: i64 = 24;
/// Sites stay this far inside a chart's box so nothing they paint crosses a seam.
const SITE_MARGIN: i64 = 14;

/// One column of a chart: where its surface is (storage y of the first open cell) and how it is dressed.
#[derive(Clone, Copy, Debug)]
struct Column {
    surface: i64,
    top: BlockId,
    sub: BlockId,
    sub_depth: i64,
}

/// 3-D fractal noise on a physical point (octaves rotated through a fixed matrix), ≈ −1..1.
fn fbm3(seed: u32, p: DVec3, octaves: u32) -> f32 {
    let (mut q, mut sum, mut amp, mut norm) = (p, 0.0f32, 1.0f32, 0.0f32);
    for o in 0..octaves {
        sum += amp * perlin3(seed.wrapping_add(o.wrapping_mul(0x632B_E5AB)), q.x, q.y, q.z);
        norm += amp;
        amp *= 0.5;
        q = DVec3::new(0.8 * q.x - 0.6 * q.z + 17.3, q.y * 1.0 + 5.1, 0.6 * q.x + 0.8 * q.z - 41.1) * 2.0;
    }
    sum / norm
}

/// Ridged 3-D noise (sharp crests), ≈ 0..1.
fn ridged3(seed: u32, p: DVec3, octaves: u32) -> f32 {
    let (mut q, mut sum, mut amp, mut norm, mut weight) = (p, 0.0f32, 1.0f32, 0.0f32, 1.0f32);
    for o in 0..octaves {
        let mut r = 1.0 - perlin3(seed.wrapping_add(o.wrapping_mul(0x5851_F42D)), q.x, q.y, q.z).abs();
        r *= r * weight;
        weight = (r * 1.8).clamp(0.0, 1.0);
        sum += r * amp;
        norm += amp;
        amp *= 0.5;
        q = DVec3::new(0.8 * q.x - 0.6 * q.z - 7.9, q.y + 3.3, 0.6 * q.x + 0.8 * q.z + 13.3) * 2.0;
    }
    sum / norm
}

fn smoothstep(a: f32, b: f32, t: f32) -> f32 {
    let t = ((t - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl Round {
    pub fn new(atlas: Atlas, seed: u32, style: Style, m: Arc<Materials>) -> Self {
        Self { atlas, seed, style, m }
    }

    fn salt(&self, k: u32) -> u32 {
        self.seed.wrapping_mul(0x9E37_79B9).wrapping_add(k.wrapping_mul(0x85EB_CA6B))
    }

    /// Height of the terrain above (outward) or into the cavity from (inward) the datum radius at
    /// physical point `p` on the datum sphere, in blocks.
    fn relief(&self, p: DVec3) -> f32 {
        let s = |k| self.salt(k);
        match self.style {
            Style::Verdant => {
                let cont = 0.5 + 0.7 * fbm3(s(1), p / 9000.0, 4);
                let ranges = smoothstep(0.55, 0.85, cont);
                let peaks = if ranges > 0.0 { ridged3(s(2), p / 1700.0, 5) } else { 0.0 };
                let hills = 34.0 * fbm3(s(3), p / 420.0, 4);
                let valleys = smoothstep(0.0, 0.12, fbm3(s(4), p / 2600.0, 3).abs());
                20.0 + 60.0 * cont + hills * (0.4 + 0.6 * valleys) + 900.0 * peaks * peaks * ranges * valleys - 30.0 * (1.0 - valleys)
            }
            Style::HollowOuter => {
                let crags = ridged3(s(5), p / 900.0, 5);
                24.0 + 260.0 * crags * crags + 18.0 * fbm3(s(6), p / 150.0, 3)
            }
            Style::HollowInner => {
                // The inner surface hangs toward the centre: positive relief reaches into the cavity.
                16.0 + 120.0 * smoothstep(0.3, 0.9, ridged3(s(7), p / 700.0, 4)) + 10.0 * fbm3(s(8), p / 90.0, 3)
            }
        }
    }

    /// The column at chart cell `(i, j)` of a shell patch.
    fn column(&self, patch: Patch, i: i64, j: i64) -> Column {
        let b = match patch {
            Patch::Shell { band, .. } => self.atlas.bands[band as usize],
            _ => unreachable!("columns belong to shell charts"),
        };
        let centre = self.atlas.centre;
        let dir = (self.atlas.embed(patch, DVec3::new(i as f64 + 0.5, 0.5, j as f64 + 0.5)) - centre).normalize();
        let datum = self.atlas.radius as f64;
        let h = self.relief(centre + dir * datum) as f64;
        // Outward charts count storage y up from r_lo; inward ones from r_hi toward the centre.
        let surface = if self.atlas.inward { b.r_hi as f64 - (datum - h) } else { datum + h - b.r_lo as f64 };
        let m = &self.m;
        let wet = fbm3(self.salt(9), dir * datum / 700.0, 2);
        let (top, sub, sub_depth) = match self.style {
            Style::Verdant if h > 620.0 => (m.snow, m.gravel, 2),
            Style::Verdant if h > 380.0 => (m.gravel, m.rock[1], 2),
            Style::Verdant => (if wet > 0.2 { m.moss } else if wet < -0.3 { m.meadow } else { m.grass }, m.soil, 4),
            Style::HollowOuter => (if h > 180.0 { m.snow } else { m.frost }, m.ice, 6),
            Style::HollowInner => (if wet > 0.1 { m.glowcap } else { m.violet }, m.crystal, 3),
        };
        Column { surface: surface.floor() as i64, top, sub, sub_depth }
    }

    /// The ground at depth `d` (1 = the top cell) below a column's surface.
    fn ground(&self, col: &Column, d: i64) -> BlockId {
        if d == 1 {
            col.top
        } else if d <= col.sub_depth + 1 {
            col.sub
        } else if d < 220 {
            self.m.rock[((d / 9) % 4) as usize]
        } else if self.style == Style::HollowInner || self.style == Style::HollowOuter {
            self.m.basalt
        } else {
            self.m.deeprock
        }
    }

    /// Whether a cave carves the cell at physical point `p`, `d` cells under the surface.
    fn cave(&self, p: DVec3, d: i64) -> bool {
        if !(5..400).contains(&d) {
            return false;
        }
        let a = perlin3(self.salt(10), p.x / 48.0, p.y / 48.0, p.z / 48.0);
        let b = perlin3(self.salt(11), p.x / 48.0, p.y / 48.0, p.z / 48.0);
        let r = 0.07 + 0.05 * (d as f32 / 400.0);
        a * a + b * b < r * r
    }

    /// The tree or spire of site `(si, sj)`, if any.
    fn site(&self, patch: Patch, size: i64, si: i64, sj: i64) -> Option<Plant> {
        let h = hash2(self.salt(12) ^ hash2(patch_tag(patch), si as i32, sj as i32), si as i32, sj as i32);
        let chance = match self.style {
            Style::Verdant => 0.55,
            Style::HollowOuter => 0.12,
            Style::HollowInner => 0.35,
        };
        if unit(h) >= chance {
            return None;
        }
        let (bi, bj) = (si * SITE + 4 + (h % 16) as i64, sj * SITE + 4 + ((h >> 4) % 16) as i64);
        if bi < SITE_MARGIN || bj < SITE_MARGIN || bi >= size - SITE_MARGIN || bj >= size - SITE_MARGIN {
            return None;
        }
        let tall = (h >> 8) % 100;
        let (half, height, crown) = match self.style {
            Style::Verdant => (if tall > 70 { 2 } else { 1 }, 28 + tall as i64 / 2, 6 + (tall % 5) as i64),
            Style::HollowOuter => (1, 14 + tall as i64 / 4, 0),
            Style::HollowInner => (if tall > 80 { 2 } else { 1 }, 12 + tall as i64 / 3, 0),
        };
        Some(Plant { i: bi, j: bj, base: self.column(patch, bi, bj).surface, half, height, crown })
    }

    /// Every plant whose site could paint into columns `i0..=i1` × `j0..=j1`.
    fn plants_near(&self, patch: Patch, size: i64, i0: i64, i1: i64, j0: i64, j1: i64, out: &mut Vec<Plant>) {
        for si in i0.div_euclid(SITE) - 1..=i1.div_euclid(SITE) + 1 {
            for sj in j0.div_euclid(SITE) - 1..=j1.div_euclid(SITE) + 1 {
                out.extend(self.site(patch, size, si, sj));
            }
        }
    }

    /// The plant block at chart cell `(i, y, j)` among `plants`, if any.
    fn plant(&self, plants: &[Plant], i: i64, y: i64, j: i64) -> Option<BlockId> {
        let m = &self.m;
        for p in plants {
            let (di, dj, dy) = (i - p.i, j - p.j, y - p.base);
            if dy < 0 {
                continue;
            }
            match self.style {
                Style::Verdant => {
                    if di.abs() <= p.half && dj.abs() <= p.half && dy < p.height {
                        return Some(m.timber);
                    }
                    let cy = dy - p.height;
                    if di * di + dj * dj + cy * cy * 2 <= p.crown * p.crown {
                        return Some(m.leaves);
                    }
                }
                Style::HollowOuter | Style::HollowInner => {
                    // A tapering spire: its radius shrinks with height.
                    let r = (p.half + 1) as f64 * (1.0 - dy as f64 / p.height as f64);
                    if dy < p.height && ((di * di + dj * dj) as f64) <= r * r {
                        return Some(if self.style == Style::HollowOuter { m.ice } else { m.crystal });
                    }
                }
            }
        }
        None
    }

    /// The block of a shell cell given its column and the plants around it.
    fn shell_cell(&self, patch: Patch, col: &Column, plants: &[Plant], l: [i64; 3]) -> BlockId {
        let d = col.surface - l[1];
        if d >= 1 {
            let p = self.atlas.embed(patch, DVec3::new(l[0] as f64 + 0.5, l[1] as f64 + 0.5, l[2] as f64 + 0.5));
            if self.cave(p, d) { AIR } else { self.ground(col, d) }
        } else {
            self.plant(plants, l[0], l[1], l[2]).unwrap_or(AIR)
        }
    }

    /// The block in storage cell `s` (air outside every patch box).
    pub fn voxel(&self, s: [i64; 3]) -> BlockId {
        let Some((patch, l)) = self.atlas.locate(s) else { return AIR };
        match patch {
            Patch::Shell { .. } => {
                let (_, size) = self.atlas.storage_box(patch);
                let col = self.column(patch, l[0], l[2]);
                let mut plants = Vec::new();
                self.plants_near(patch, size[0], l[0], l[0], l[2], l[2], &mut plants);
                self.shell_cell(patch, &col, &plants, l)
            }
            // The deep interior: rock, then a molten heart.
            Patch::Transition { .. } => self.m.deeprock,
            Patch::Core => self.m.magma,
        }
    }

    /// The storage chunk at chunk coordinates `c`, filled column by column (equal to [`voxel`](Self::voxel)).
    pub fn fill_chunk(&self, c: [i64; 3]) -> ChunkData {
        let n = CHUNK_SIZE as i64;
        let origin = [c[0] * n, c[1] * n, c[2] * n];
        let Some((patch, l0)) = self.atlas.locate(origin) else { return ChunkData::Uniform(AIR) };
        if !matches!(patch, Patch::Shell { .. }) {
            return ChunkData::Uniform(self.voxel(origin));
        }
        let (_, size) = self.atlas.storage_box(patch);
        let mut plants = Vec::new();
        self.plants_near(patch, size[0], l0[0], l0[0] + n - 1, l0[2], l0[2] + n - 1, &mut plants);
        let mut cells = Box::new([AIR; CHUNK_VOLUME]);
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let (i, j) = (l0[0] + lx as i64, l0[2] + lz as i64);
                let col = self.column(patch, i, j);
                for ly in 0..CHUNK_SIZE {
                    cells[Chunk::index(lx, ly, lz)] = self.shell_cell(patch, &col, &plants, [i, l0[1] + ly as i64, j]);
                }
            }
        }
        ChunkData::from_cells(cells)
    }
}

/// A tree or spire: base column, base storage y, trunk half-width, height, crown radius.
#[derive(Clone, Copy, Debug)]
struct Plant {
    i: i64,
    j: i64,
    base: i64,
    half: i64,
    height: i64,
    crown: i64,
}

/// A stable small tag of a patch for hashing.
fn patch_tag(p: Patch) -> u32 {
    match p {
        Patch::Shell { band, face } => 1 + band as u32 * 8 + face as u32,
        Patch::Transition { face } => 200 + face as u32,
        Patch::Core => 300,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::BlockRegistry;
    use crate::coord::Face;

    fn round(style: Style, inward: bool) -> Round {
        let mut reg = BlockRegistry::with_builtins();
        let m = Arc::new(Materials::intern(&mut reg));
        let c = DVec3::new(5.0e8, -2.0e8, 1.0e8);
        let atlas = match style {
            Style::Verdant => Atlas::new(c, 60_000, 62_048, false, 0),
            _ => Atlas::shell(c, 60_000, 58_000, 62_048, inward, 1),
        };
        Round::new(atlas, 99, style, m)
    }

    #[test]
    fn a_column_has_ground_below_its_surface_and_air_above() {
        for (style, inward) in [(Style::Verdant, false), (Style::HollowOuter, false), (Style::HollowInner, true)] {
            let r = round(style, inward);
            let b = r.atlas.bands[0];
            let patch = Patch::Shell { band: 0, face: Face::PosZ };
            let (i, j) = (b.n / 3, b.n / 2 + 5);
            let col = r.column(patch, i, j);
            let below = r.atlas.storage(patch, [i, col.surface - 1, j]);
            let above = r.atlas.storage(patch, [i, col.surface + 120, j]);
            assert_ne!(r.voxel(below), AIR, "{style:?} top cell is ground");
            assert_eq!(r.voxel(above), AIR, "{style:?} sky above");
        }
    }

    #[test]
    fn the_surface_is_continuous_across_a_chart_seam() {
        let r = round(Style::Verdant, false);
        let b = r.atlas.bands[0];
        let top = Patch::Shell { band: 0, face: Face::PosY };
        for j in [b.n / 4, b.n / 2, 3 * b.n / 4] {
            let here = r.column(top, b.n - 1, j);
            // The neighbouring chart's first column across the seam.
            let s = r.atlas.storage(top, [b.n, here.surface, j]);
            let g = r.atlas.glue(s).unwrap();
            let (patch, l) = r.atlas.locate(g).unwrap();
            let there = r.column(patch, l[0], l[2]);
            assert!((here.surface - there.surface).abs() <= 3, "seam step at j={j}: {} vs {}", here.surface, there.surface);
        }
    }

    #[test]
    fn batch_fill_equals_the_per_cell_definition() {
        let r = round(Style::Verdant, false);
        let b = r.atlas.bands[0];
        let patch = Patch::Shell { band: 0, face: Face::NegX };
        let (i, j) = (b.n / 2, b.n / 2);
        let col = r.column(patch, i, j);
        let s = r.atlas.storage(patch, [i, col.surface, j]);
        let c = [s[0].div_euclid(16), s[1].div_euclid(16), s[2].div_euclid(16)];
        let data = r.fill_chunk(c);
        for (lx, ly, lz) in [(0, 0, 0), (5, 9, 3), (15, 15, 15), (8, 1, 12)] {
            let cell = [c[0] * 16 + lx as i64, c[1] * 16 + ly as i64, c[2] * 16 + lz as i64];
            assert_eq!(data.get(Chunk::index(lx, ly, lz)), r.voxel(cell));
        }
    }
}
