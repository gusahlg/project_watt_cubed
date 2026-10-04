//! Round worlds on curved charts: Verdance's giant forests, lakes and meadows, the Hollow's two
//! surfaces (an icy crust outside, crystal forests hanging toward the Ember inside), the molten
//! Ember and the cratered moons. Each is painted in its atlas's storage cells, where storage `+Y`
//! is the chart's up, so terrain is a height field per chart column. Heights come from 3-D noise
//! at the physical point on the datum sphere above the column, so the surface is continuous across
//! chart seams. Pure in `(seed, cell)`.

use std::cell::RefCell;
use std::sync::Arc;

use glam::DVec3;

use super::noise::{hash2, hash3, perlin3, unit};
use super::Materials;
use crate::block::registry::{AIR, BlockId};
use crate::space::atlas::{Atlas, Patch};
use crate::world::chunk::{CHUNK_SIZE, CHUNK_VOLUME, Chunk, ChunkData};

const CHUNK_AREA: usize = CHUNK_SIZE * CHUNK_SIZE;

/// What a round world looks like.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Style {
    /// Rolling forested hills, ranges and deep green valleys; giant trees.
    Verdant,
    /// The Hollow's outer crust: frost and ice, crags and ice spires.
    HollowOuter,
    /// The Hollow's inner surface: violet ground and crystal spires reaching for the centre.
    HollowInner,
    /// A moon: crater fields at three scales over regolith, dark basalt maria; `tone` gives it its
    /// character (0 grey dust, 1 frozen, 2 rust).
    Moon { tone: u8 },
    /// The Hollow's core: basalt plates over glowing magma seams, basalt spires.
    Ember,
}

/// One round world's painter.
pub struct Round {
    pub atlas: Atlas,
    seed: u32,
    style: Style,
    /// Unit direction of a moon's ice cap: from the parent toward the moon. Zero for every other style.
    pole: DVec3,
    m: Arc<Materials>,
}

/// Grid spacing of tree / spire sites, in cells.
const SITE: i64 = 24;
/// Spacing (cells) of the lattice the relief and cave fields are sampled on; cells interpolate
/// between nodes (the fields are smooth at that scale). Nodes on a chart edge are the same physical
/// points from both sides, so the surface stays continuous across seams.
const LATTICE: i64 = 4;
/// Sites stay this far inside a chart's box so nothing they paint crosses a seam.
const SITE_MARGIN: i64 = 14;
/// Verdant waterline. Lows flood toward it; the mask is a function of the physical point, so a lake
/// has no cliff and the seam stays continuous. Still water is solid frost over ice.
const SHORE: f32 = 72.0;
/// How far above a column's first open cell a plant can reach (Verdant crown included).
pub(super) const PLANT_REACH: i64 = 96;

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
    pub fn new(atlas: Atlas, seed: u32, style: Style, m: Arc<Materials>, pole: DVec3) -> Self {
        Self { atlas, seed, style, pole, m }
    }

    fn salt(&self, k: u32) -> u32 {
        self.seed.wrapping_mul(0x9E37_79B9).wrapping_add(k.wrapping_mul(0x85EB_CA6B))
    }

    /// Height of the terrain above (outward) or into the cavity from (inward) the datum radius at
    /// physical point `p` on the datum sphere, in blocks.
    fn relief(&self, p: DVec3) -> f32 {
        let s = |k| self.salt(k);
        match self.style {
            Style::Verdant => self.apply_lake(p, self.verdant_land(p)),
            Style::HollowOuter => {
                let crags = ridged3(s(5), p / 900.0, 5);
                24.0 + 260.0 * crags * crags + 18.0 * fbm3(s(6), p / 150.0, 3)
            }
            Style::HollowInner => {
                // The inner surface hangs toward the centre: positive relief reaches into the cavity.
                16.0 + 120.0 * smoothstep(0.3, 0.9, ridged3(s(7), p / 700.0, 4)) + 10.0 * fbm3(s(8), p / 90.0, 3)
            }
            Style::Moon { .. } => {
                let base = 30.0 * fbm3(s(13), p / 3000.0, 3);
                let mare = self.mare(p);
                base * (1.0 - 0.7 * mare) - 25.0 * mare + self.craters(p) * (1.0 - 0.5 * mare)
            }
            Style::Ember => {
                let plates = ridged3(s(15), p / 600.0, 4);
                let base = 18.0 + 70.0 * plates * plates + 6.0 * fbm3(s(16), p / 60.0, 2);
                base - 18.0 * self.river_mask(p)
            }
        }
    }

    /// How much of a moon's dark mare covers point `p` (0..1).
    fn mare(&self, p: DVec3) -> f32 {
        smoothstep(0.25, 0.45, fbm3(self.salt(14), p / 40_000.0, 2))
    }

    /// Crater relief at a point on the datum sphere: bowls with raised rims at three scales, central
    /// peaks in the largest. Crater sites live on a 3-D grid around the body and are projected onto
    /// the datum sphere, so craters are round on the sphere and continuous across chart seams.
    fn craters(&self, p: DVec3) -> f32 {
        // (grid cell, smallest and largest radius, chance a cell holds one); the largest radius times
        // the rim reach stays under one cell, so the 27 cells around the point see every crater.
        const SCALES: [(f64, f64, f64, f32); 3] =
            [(9_000.0, 900.0, 2_600.0, 0.35), (1_800.0, 120.0, 600.0, 0.5), (360.0, 14.0, 110.0, 0.55)];
        let rel = p - self.atlas.centre;
        let datum = self.atlas.radius as f64;
        let mut h = 0.0f64;
        for (k, &(cell, rmin, rmax, chance)) in SCALES.iter().enumerate() {
            let g = (rel / cell).floor();
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let c = g + DVec3::new(dx as f64, dy as f64, dz as f64);
                        let (ci, cj, ck) = (c.x as i32, c.y as i32, c.z as i32);
                        let hh = hash3(self.salt(20 + k as u32), ci, cj, ck);
                        if unit(hh) >= chance {
                            continue;
                        }
                        let off = DVec3::new(
                            unit(hash3(hh, 1, 0, 0)) as f64,
                            unit(hash3(hh, 2, 0, 0)) as f64,
                            unit(hash3(hh, 3, 0, 0)) as f64,
                        );
                        let site = (c + off) * cell;
                        let len = site.length();
                        if len == 0.0 {
                            continue;
                        }
                        // Many small, few large: radius ∝ u³ between the bounds.
                        let u = unit(hash3(hh, 4, 0, 0)) as f64;
                        let rad = rmin + (rmax - rmin) * u * u * u;
                        let t = (rel - site * (datum / len)).length() / rad;
                        if t >= 1.6 {
                            continue;
                        }
                        let (depth, rim) = (0.22 * rad, 0.07 * rad);
                        let t2 = t * t;
                        let bowl = if t < 1.0 {
                            -depth * (1.0 - t2) + rim * t2 * t2 * t2
                        } else {
                            let f = 1.0 - (t - 1.0) / 0.6;
                            rim * f * f
                        };
                        let peak = if rad > 800.0 { 0.18 * depth * (1.0 - t / 0.2).max(0.0) } else { 0.0 };
                        h += bowl + peak;
                    }
                }
            }
        }
        h as f32
    }

    /// Verdance before lakes: ranges, hills and valleys. A function of the physical point only.
    fn verdant_land(&self, p: DVec3) -> f32 {
        let s = |k| self.salt(k);
        let cont = 0.5 + 0.7 * fbm3(s(1), p / 9000.0, 4);
        let ranges = smoothstep(0.55, 0.85, cont);
        let peaks = if ranges > 0.0 { ridged3(s(2), p / 1700.0, 5) } else { 0.0 };
        let hills = 34.0 * fbm3(s(3), p / 420.0, 4);
        let valleys = smoothstep(0.0, 0.12, fbm3(s(4), p / 2600.0, 3).abs());
        20.0 + 60.0 * cont + hills * (0.4 + 0.6 * valleys) + 900.0 * peaks * peaks * ranges * valleys - 30.0 * (1.0 - valleys)
    }

    /// Where a low basin fills (0 on dry ground, 1 in a lake). Smooth in the physical point.
    fn flood_mask(&self, p: DVec3) -> f32 {
        smoothstep(0.18, 0.62, fbm3(self.salt(19), p / 1400.0, 3))
    }

    /// Raise land below [`SHORE`] toward the waterline. Peaks are unchanged.
    fn apply_lake(&self, p: DVec3, raw: f32) -> f32 {
        if raw >= SHORE {
            return raw;
        }
        let depth = smoothstep(SHORE, 20.0, raw);
        raw + (SHORE - raw) * self.flood_mask(p) * depth
    }

    /// Magma rivers: a thin ridge of the seam noise, 1 in the channel.
    fn river_mask(&self, p: DVec3) -> f32 {
        let n = fbm3(self.salt(17), p / 900.0, 3).abs();
        1.0 - smoothstep(0.015, 0.08, n)
    }

    /// A moon's ice cap lies where the surface faces away from its parent, never along a world axis.
    fn ice_cap(&self, dir: DVec3) -> bool {
        matches!(self.style, Style::Moon { .. })
            && self.seed % 5 == 0
            && self.pole.length_squared() > 0.5
            && dir.dot(self.pole) > 0.84
    }

    /// Glowing fissures on dusty and rust moons. Frozen moons (tone 1) have none.
    fn glowing_crack(&self, tone: u8, p: DVec3) -> bool {
        tone != 1 && fbm3(self.salt(18), p / 240.0, 3).abs() < 0.032
    }

    /// Physical point on the datum sphere above column `(i, j)`.
    fn datum(&self, patch: Patch, i: i64, j: i64) -> DVec3 {
        let centre = self.atlas.centre;
        let dir = (self.atlas.embed(patch, DVec3::new(i as f64 + 0.5, 0.5, j as f64 + 0.5)) - centre).normalize();
        centre + dir * self.atlas.radius as f64
    }

    /// The painter's style.
    pub fn style(&self) -> Style {
        self.style
    }

    /// Storage y of the first open cell of shell column `(i, j)`.
    pub fn column_surface(&self, patch: Patch, i: i64, j: i64) -> i64 {
        self.column(patch, i, j).surface
    }

    /// The fill below every cave (the uniform deep body of a shell chart).
    pub fn deep(&self) -> BlockId {
        match self.style {
            Style::HollowInner | Style::HollowOuter => self.m.basalt,
            Style::Ember => self.m.magma,
            _ => self.m.deeprock,
        }
    }

    /// The fill of the transition shell and the core.
    pub fn heart(&self) -> (BlockId, BlockId) {
        match self.style {
            Style::Moon { .. } => (self.m.deeprock, self.m.basalt),
            _ => (self.m.deeprock, self.m.magma),
        }
    }

    /// Relief at lattice node `(i, j)` (a cell corner) of a shell patch.
    fn relief_node(&self, patch: Patch, i: i64, j: i64) -> f32 {
        let centre = self.atlas.centre;
        let dir = (self.atlas.embed(patch, DVec3::new(i as f64, 0.5, j as f64)) - centre).normalize();
        self.relief(centre + dir * self.atlas.radius as f64)
    }

    /// Relief of column `(i, j)`: bilinear between its four lattice nodes.
    fn relief_at(&self, patch: Patch, i: i64, j: i64) -> f32 {
        let (i0, j0) = (i.div_euclid(LATTICE) * LATTICE, j.div_euclid(LATTICE) * LATTICE);
        let n = |a: i64, b: i64| self.relief_node(patch, i0 + a * LATTICE, j0 + b * LATTICE);
        bilerp([[n(0, 0), n(0, 1)], [n(1, 0), n(1, 1)]], i - i0, j - j0)
    }

    /// The column at chart cell `(i, j)` of a shell patch.
    fn column(&self, patch: Patch, i: i64, j: i64) -> Column {
        self.column_with(patch, i, j, self.relief_at(patch, i, j))
    }

    /// The column at chart cell `(i, j)` given its relief `h`.
    fn column_with(&self, patch: Patch, i: i64, j: i64, h: f32) -> Column {
        let b = match patch {
            Patch::Shell { band, .. } => self.atlas.bands[band as usize],
            _ => unreachable!("columns belong to shell charts"),
        };
        let centre = self.atlas.centre;
        let dir = (self.atlas.embed(patch, DVec3::new(i as f64 + 0.5, 0.5, j as f64 + 0.5)) - centre).normalize();
        let datum = self.atlas.radius as f64;
        let p = centre + dir * datum;
        let h = h as f64;
        // Outward charts count storage y up from r_lo; inward ones from r_hi toward the centre.
        let surface = if self.atlas.inward { b.r_hi as f64 - (datum - h) } else { datum + h - b.r_lo as f64 };
        let m = &self.m;
        let wet = fbm3(self.salt(9), p / 700.0, 2);
        let (top, sub, sub_depth) = match self.style {
            Style::Verdant => self.verdant_cover(p, h as f32, wet),
            Style::HollowOuter => (if h > 180.0 { m.snow } else { m.frost }, m.ice, 6),
            Style::HollowInner => (if wet > 0.1 { m.glowcap } else { m.violet }, m.crystal, 3),
            Style::Moon { tone } => self.moon_cover(tone, dir, p, h as f32),
            Style::Ember => {
                if self.river_mask(p) > 0.62 { (m.magma, m.magma, 4) } else { (m.basalt, m.basalt, 4) }
            }
        };
        Column { surface: surface.floor() as i64, top, sub, sub_depth }
    }

    /// Verdant cover. A lake is decided from the unflooded land at this column, not from a peak
    /// the interpolated waterline happens to sit under.
    fn verdant_cover(&self, p: DVec3, h: f32, wet: f32) -> (BlockId, BlockId, i64) {
        let m = &self.m;
        let land = self.verdant_land(p);
        let flooded = self.flood_mask(p) * smoothstep(SHORE, 20.0, land);
        if land < SHORE - 1.0 && flooded > 0.55 {
            let depth = ((SHORE - land).round() as i64).clamp(3, 8);
            return (m.frost, m.ice, depth);
        }
        if h > 620.0 {
            (m.snow, m.gravel, 2)
        } else if h > 380.0 {
            (m.gravel, m.rock[1], 2)
        } else if wet > 0.25 {
            (m.moss, m.soil, 4)
        } else if wet < -0.15 {
            (m.grass, m.soil, 4)
        } else {
            (m.meadow, m.soil, 4)
        }
    }

    /// Moon cover: a rare ice cap facing away from the parent, then tone-driven fissures, then dust.
    fn moon_cover(&self, tone: u8, dir: DVec3, p: DVec3, h: f32) -> (BlockId, BlockId, i64) {
        let m = &self.m;
        if self.ice_cap(dir) {
            return (m.snow, m.ice, 6);
        }
        if self.glowing_crack(tone, p) {
            let id = if tone >= 2 { m.magma } else { m.glowshroom };
            return (id, if tone >= 2 { m.magma } else { m.basalt }, 2);
        }
        match tone {
            1 => (if h > 60.0 { m.snow } else { m.frost }, m.ice, 5),
            2 => (m.redsand, m.ochre, 3),
            _ => (if self.mare(p) > 0.5 { m.basalt } else { m.regolith }, m.gravel, 3),
        }
    }

    /// The ground at depth `d` (1 = the top cell) below a column's surface.
    fn ground(&self, col: &Column, d: i64) -> BlockId {
        if d == 1 {
            col.top
        } else if d <= col.sub_depth + 1 {
            col.sub
        } else if self.style == Style::Ember {
            if d < 60 { self.m.basalt } else { self.m.magma }
        } else if d < 220 {
            self.m.rock[((d / 9) % 4) as usize]
        } else {
            self.deep()
        }
    }

    /// The two cave fields at lattice node `l` (a cell corner) of a patch.
    fn cave_node(&self, patch: Patch, l: [i64; 3]) -> [f32; 2] {
        let p = self.atlas.embed(patch, DVec3::new(l[0] as f64, l[1] as f64, l[2] as f64)) / 48.0;
        [perlin3(self.salt(10), p.x, p.y, p.z), perlin3(self.salt(11), p.x, p.y, p.z)]
    }

    /// The eight cave nodes around cell `l`, indexed `[x][y][z]` from its lattice corner.
    fn cave_cube(&self, patch: Patch, l: [i64; 3]) -> [[[[f32; 2]; 2]; 2]; 2] {
        let o = l.map(|v| v.div_euclid(LATTICE) * LATTICE);
        std::array::from_fn(|x| {
            std::array::from_fn(|y| {
                std::array::from_fn(|z| {
                    self.cave_node(patch, [o[0] + x as i64 * LATTICE, o[1] + y as i64 * LATTICE, o[2] + z as i64 * LATTICE])
                })
            })
        })
    }

    /// Whether a cave carves cell `l`, `d` cells under the surface, from the cave nodes around it.
    fn cave(&self, cube: &[[[[f32; 2]; 2]; 2]; 2], l: [i64; 3], d: i64) -> bool {
        if !(5..400).contains(&d) {
            return false;
        }
        let f = l.map(|v| v.rem_euclid(LATTICE));
        let field = |k: usize| {
            trilerp(std::array::from_fn(|x| std::array::from_fn(|y| std::array::from_fn(|z| cube[x][y][z][k]))), f)
        };
        let (a, b) = (field(0), field(1));
        let r = 0.07 + 0.05 * (d as f32 / 400.0);
        a * a + b * b < r * r
    }

    /// The tree or spire of site `(si, sj)`, if any.
    fn site(&self, patch: Patch, size: i64, si: i64, sj: i64) -> Option<Plant> {
        let h = hash2(self.salt(12) ^ hash2(patch_tag(patch), si as i32, sj as i32), si as i32, sj as i32);
        let chance = match self.style {
            Style::Verdant => 0.42,
            Style::HollowOuter => 0.12,
            Style::HollowInner => 0.28,
            // Hash first: crater rims are expensive, and most sites are bare dust.
            Style::Moon { .. } => 0.16,
            Style::Ember => 0.07,
        };
        if unit(h) >= chance {
            return None;
        }
        let (bi, bj) = (si * SITE + 4 + (h % 16) as i64, sj * SITE + 4 + ((h >> 4) % 16) as i64);
        if bi < SITE_MARGIN || bj < SITE_MARGIN || bi >= size - SITE_MARGIN || bj >= size - SITE_MARGIN {
            return None;
        }
        if matches!(self.style, Style::Moon { .. }) && self.craters(self.datum(patch, bi, bj)) < 2.5 {
            return None;
        }
        if self.style == Style::Ember && self.river_mask(self.datum(patch, bi, bj)) > 0.35 {
            return None;
        }
        let tall = (h >> 8) % 100;
        let (half, height, crown) = match self.style {
            Style::Verdant => (if tall > 70 { 2 } else { 1 }, 28 + tall as i64 / 2, 6 + (tall % 5) as i64),
            Style::HollowOuter => (1, 14 + tall as i64 / 4, 0),
            Style::HollowInner => (if tall > 80 { 2 } else { 1 }, 16 + tall as i64 / 3, 0),
            Style::Moon { .. } => (3, 3 + (tall % 3) as i64, 0),
            Style::Ember => (2, 12 + tall as i64 / 4, 0),
        };
        let col = self.column(patch, bi, bj);
        // Meadows stay open: most rolls there do not grow a tree.
        if self.style == Style::Verdant && col.top == self.m.meadow && unit(hash3(h, 9, 0, 0)) > 0.22 {
            return None;
        }
        Some(Plant { i: bi, j: bj, base: col.surface, half, height, crown })
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
        for p in plants {
            let (di, dj, dy) = (i - p.i, j - p.j, y - p.base);
            if dy < 0 {
                continue;
            }
            let id = match self.style {
                Style::Verdant => verdant_tree(&self.m, p, di, dj, dy),
                Style::HollowInner => crystal_forest(&self.m, p, di, dj, dy),
                Style::Moon { .. } => boulder(&self.m, p, di, dj, dy),
                Style::Ember => basalt_column(&self.m, p, di, dj, dy),
                Style::HollowOuter => ice_spire(self.m.ice, p, di, dj, dy),
            };
            if id.is_some() {
                return id;
            }
        }
        None
    }

    /// The block of a shell cell given its column and the plants around it.
    /// `caves` gives the cave nodes around `l` (only asked for cells under the surface).
    fn shell_cell(
        &self,
        col: &Column,
        plants: &[Plant],
        l: [i64; 3],
        caves: impl FnOnce() -> [[[[f32; 2]; 2]; 2]; 2],
    ) -> BlockId {
        let d = col.surface - l[1];
        if d >= 1 {
            if (5..400).contains(&d) && self.cave(&caves(), l, d) { AIR } else { self.ground(col, d) }
        } else {
            self.plant(plants, l[0], l[1], l[2]).unwrap_or_else(|| self.meadow_flower(col, l))
        }
    }

    /// A single flower on open meadow, one cell above the grass. It sits on its own column.
    fn meadow_flower(&self, col: &Column, l: [i64; 3]) -> BlockId {
        if self.style != Style::Verdant || col.top != self.m.meadow || l[1] != col.surface {
            return AIR;
        }
        let h = hash2(self.salt(21), l[0] as i32, l[2] as i32);
        if unit(h) >= 0.045 {
            return AIR;
        }
        match h % 4 {
            0 => self.m.flower_red,
            1 => self.m.flower_yellow,
            2 => self.m.flower_blue,
            _ => self.m.flower_white,
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
                self.shell_cell(&col, &plants, l, || self.cave_cube(patch, l))
            }
            // The deep interior: rock, then the heart.
            Patch::Transition { .. } => self.heart().0,
            Patch::Core => self.heart().1,
        }
    }

    /// The 16×16 columns of a chunk whose first column is `(i0, j0)` (chunk aligned) of a shell
    /// patch: the relief lattice once, then each column (equal to [`column`](Self::column)).
    fn chunk_columns(&self, patch: Patch, i0: i64, j0: i64) -> Vec<Column> {
        const NODES: usize = CHUNK_SIZE / LATTICE as usize + 1;
        let relief: [[f32; NODES]; NODES] = std::array::from_fn(|a| {
            std::array::from_fn(|b| self.relief_node(patch, i0 + a as i64 * LATTICE, j0 + b as i64 * LATTICE))
        });
        (0..CHUNK_AREA)
            .map(|k| {
                let (lx, lz) = (k % CHUNK_SIZE, k / CHUNK_SIZE);
                let (a, b) = (lx / LATTICE as usize, lz / LATTICE as usize);
                let h = bilerp(
                    [[relief[a][b], relief[a][b + 1]], [relief[a + 1][b], relief[a + 1][b + 1]]],
                    lx as i64 % LATTICE,
                    lz as i64 % LATTICE,
                );
                self.column_with(patch, i0 + lx as i64, j0 + lz as i64, h)
            })
            .collect()
    }

    /// The storage-y surfaces of a chunk-aligned 16×16 block of shell columns starting at `(i0, j0)`,
    /// indexed `lx + lz·16` (equal to [`column_surface`](Self::column_surface)).
    pub fn chunk_surfaces(&self, patch: Patch, i0: i64, j0: i64) -> Vec<i64> {
        self.chunk_columns(patch, i0, j0).iter().map(|c| c.surface).collect()
    }

    /// Storage y of relief `h` on a shell patch (the same formula as a column's surface).
    pub(super) fn surface_of_relief(&self, patch: Patch, h: f32) -> i64 {
        let b = match patch {
            Patch::Shell { band, .. } => self.atlas.bands[band as usize],
            _ => return 0,
        };
        let datum = self.atlas.radius as f64;
        let h = h as f64;
        let surface = if self.atlas.inward { b.r_hi as f64 - (datum - h) } else { datum + h - b.r_lo as f64 };
        surface.floor() as i64
    }

    /// Min and max relief over chart columns `[i0, i0+span) × [j0, j0+span)`.
    /// Exact over the lattice when the square is small; coarser squares stride and pad, so the
    /// range stays a superset (a far section must not clip a peak the stride stepped over).
    pub(super) fn relief_bounds(&self, patch: Patch, i0: i64, j0: i64, span: i64) -> (f32, f32) {
        let snap = |p: i64| p.div_euclid(LATTICE) * LATTICE;
        let i_lo = snap(i0);
        let j_lo = snap(j0);
        let i_hi = snap(i0 + span - 1) + LATTICE;
        let j_hi = snap(j0 + span - 1) + LATTICE;
        let ni = (i_hi - i_lo) / LATTICE + 1;
        let nj = (j_hi - j_lo) / LATTICE + 1;
        let (si, sj, pad) = if ni <= 64 && nj <= 64 { (1, 1, 0.0) } else { ((ni / 48).max(1), (nj / 48).max(1), 160.0) };
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        let mut i = i_lo;
        loop {
            let mut j = j_lo;
            loop {
                let h = self.relief_node(patch, i, j);
                lo = lo.min(h);
                hi = hi.max(h);
                if j == j_hi {
                    break;
                }
                j = (j + sj * LATTICE).min(j_hi);
            }
            if i == i_hi {
                break;
            }
            i = (i + si * LATTICE).min(i_hi);
        }
        if !lo.is_finite() {
            return (0.0, 0.0);
        }
        (lo - pad, hi + pad)
    }

    /// Far-LOD column: one relief sample, the plants on it, solid ground below. No caves —
    /// a coarse tile aliases them to noise, same as a cube face.
    pub(super) fn lod_column(&self, patch: Patch, i: i64, j: i64, origin_y: i64, ys: &[i32], out: &mut [BlockId]) {
        let col = self.column(patch, i, j);
        let (_, size) = self.atlas.storage_box(patch);
        self.with_plants(patch, size[0], i, j, |plants| {
            for (o, &sy) in out.iter_mut().zip(ys) {
                let y = sy as i64 - origin_y;
                let d = col.surface - y;
                *o = if d >= 1 { self.ground(&col, d) } else { self.plant(plants, i, y, j).unwrap_or(AIR) };
            }
        });
    }

    /// Plants that can paint column `(i, j)`, reused across the columns of one section.
    fn with_plants(&self, patch: Patch, size: i64, i: i64, j: i64, f: impl FnOnce(&[Plant])) {
        PLANTS.with(|slot| {
            let mut slot = slot.borrow_mut();
            let key = (self.atlas.centre.x.to_bits(), self.atlas.centre.y.to_bits(), self.atlas.centre.z.to_bits(), self.atlas.radius, self.atlas.inward, self.seed, patch);
            let hit = slot.as_ref().is_some_and(|w| w.key == key && i >= w.i0 && i <= w.i1 && j >= w.j0 && j <= w.j1);
            if !hit {
                let (i0, i1) = (i - 32, i + 192);
                let (j0, j1) = (j - 32, j + 192);
                let mut plants = Vec::new();
                self.plants_near(patch, size, i0, i1, j0, j1, &mut plants);
                *slot = Some(PlantWin { key, i0, i1, j0, j1, plants });
            }
            f(&slot.as_ref().expect("plant window").plants);
        });
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
        let cols = self.chunk_columns(patch, l0[0], l0[2]);
        const NODES: usize = CHUNK_SIZE / LATTICE as usize + 1;
        // The cave lattice, filled only if some cell of the chunk lies in the cave band.
        let top = l0[1] + n - 1;
        let caves_here = cols.iter().any(|c| c.surface - top < 400 && c.surface - l0[1] >= 5);
        let cave_nodes: Vec<[f32; 2]> = if caves_here {
            (0..NODES * NODES * NODES)
                .map(|k| {
                    let (x, y, z) = (k % NODES, k / NODES % NODES, k / (NODES * NODES));
                    self.cave_node(patch, [l0[0] + x as i64 * LATTICE, l0[1] + y as i64 * LATTICE, l0[2] + z as i64 * LATTICE])
                })
                .collect()
        } else {
            Vec::new()
        };
        let mut cells = Box::new([AIR; CHUNK_VOLUME]);
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let (i, j) = (l0[0] + lx as i64, l0[2] + lz as i64);
                let col = &cols[lx + lz * CHUNK_SIZE];
                for ly in 0..CHUNK_SIZE {
                    let l = [i, l0[1] + ly as i64, j];
                    let cube = || {
                        let (bx, by, bz) = (lx / LATTICE as usize, ly / LATTICE as usize, lz / LATTICE as usize);
                        std::array::from_fn(|x| {
                            std::array::from_fn(|y| {
                                std::array::from_fn(|z| cave_nodes[(bx + x) + (by + y) * NODES + (bz + z) * NODES * NODES])
                            })
                        })
                    };
                    cells[Chunk::index(lx, ly, lz)] = self.shell_cell(col, &plants, l, cube);
                }
            }
        }
        ChunkData::from_cells(cells)
    }
}

/// Giant tree: bark buttresses at the base, a timber trunk, and three leaf disks. Every radius stays
/// inside [`SITE_MARGIN`], and the crown stays under the old ellipsoid's top so the sky test holds.
fn verdant_tree(m: &Materials, p: &Plant, di: i64, dj: i64, dy: i64) -> Option<BlockId> {
    let (adi, adj) = (di.abs(), dj.abs());
    if dy < 6 && adi.max(adj) <= p.half + 5 && adi.min(adj) <= 1 && adi.max(adj) > p.half {
        return Some(m.bark);
    }
    if adi <= p.half && adj <= p.half && dy < p.height {
        return Some(if dy < 4 { m.bark } else { m.timber });
    }
    let cy = dy - p.height;
    let disks = [(0i64, p.crown), (-3, p.crown - 1), (-6, (p.crown - 2).max(3))];
    for (layer, rad) in disks {
        if (cy - layer).abs() <= 1 && di * di + dj * dj <= rad * rad {
            return Some(m.leaves);
        }
    }
    None
}

/// Crystal trunks, side arms and glowing tips. Storage +Y on an inward chart points at the Ember.
fn crystal_forest(m: &Materials, p: &Plant, di: i64, dj: i64, dy: i64) -> Option<BlockId> {
    if dy < p.height && di.abs() <= p.half && dj.abs() <= p.half {
        return Some(m.crystal);
    }
    if (p.height..p.height + 3).contains(&dy) && di.abs() <= 1 && dj.abs() <= 1 {
        return Some(m.glowshroom);
    }
    let reach = p.half + 5;
    let arm = |at: i64, along: i64, across: i64| (dy - at).abs() <= 1 && across.abs() <= 1 && along > p.half && along <= reach;
    let h1 = p.height / 3;
    let h2 = 2 * p.height / 3;
    if arm(h1, di, dj) || arm(h2, -di, dj) || arm(h2, dj, di) {
        return Some(m.crystal);
    }
    if (dy - h1).abs() <= 1 && di == reach && dj.abs() <= 1 {
        return Some(m.glowshroom);
    }
    None
}

/// A squat regolith heap on a crater rim.
fn boulder(m: &Materials, p: &Plant, di: i64, dj: i64, dy: i64) -> Option<BlockId> {
    (dy < p.height && di * di + dj * dj <= p.half * p.half).then_some(m.regolith)
}

/// A basalt prism of constant radius. Rivers are skipped before the site is kept.
fn basalt_column(m: &Materials, p: &Plant, di: i64, dj: i64, dy: i64) -> Option<BlockId> {
    (dy < p.height && di * di + dj * dj <= p.half * p.half).then_some(m.basalt)
}

/// A tapering ice spire on the Hollow's outer crust.
fn ice_spire(ice: BlockId, p: &Plant, di: i64, dj: i64, dy: i64) -> Option<BlockId> {
    let r = (p.half + 1) as f64 * (1.0 - dy as f64 / p.height as f64);
    (dy < p.height && ((di * di + dj * dj) as f64) <= r * r).then_some(ice)
}

/// Bilinear between lattice nodes `h[a][b]` at cell offset `(di, dj)` (`0..LATTICE`) from node
/// `[0][0]`, sampled at the cell's centre.
fn bilerp(h: [[f32; 2]; 2], di: i64, dj: i64) -> f32 {
    let (fx, fz) = ((di as f32 + 0.5) / LATTICE as f32, (dj as f32 + 0.5) / LATTICE as f32);
    let lo = h[0][0] + (h[1][0] - h[0][0]) * fx;
    let hi = h[0][1] + (h[1][1] - h[0][1]) * fx;
    lo + (hi - lo) * fz
}

/// Trilinear between lattice nodes `v[x][y][z]` at cell offset `f` (`0..LATTICE` per axis), sampled
/// at the cell's centre.
fn trilerp(v: [[[f32; 2]; 2]; 2], f: [i64; 3]) -> f32 {
    let t = f.map(|k| (k as f32 + 0.5) / LATTICE as f32);
    let x = |y: usize, z: usize| v[0][y][z] + (v[1][y][z] - v[0][y][z]) * t[0];
    let (y0, y1) = (x(0, 0) + (x(0, 1) - x(0, 0)) * t[2], x(1, 0) + (x(1, 1) - x(1, 0)) * t[2]);
    y0 + (y1 - y0) * t[1]
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

/// Plants covering one section-sized window, so a 32×32 extract does not rebuild every column.
struct PlantWin {
    key: (u64, u64, u64, i64, bool, u32, Patch),
    i0: i64,
    i1: i64,
    j0: i64,
    j1: i64,
    plants: Vec<Plant>,
}

thread_local! {
    static PLANTS: RefCell<Option<PlantWin>> = const { RefCell::new(None) };
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
            Style::Verdant | Style::Moon { .. } | Style::Ember => Atlas::new(c, 60_000, 62_048, false, 0),
            _ => Atlas::shell(c, 60_000, 58_000, 62_048, inward, 1),
        };
        Round::new(atlas, 99, style, m, DVec3::ZERO)
    }

    #[test]
    fn a_column_has_ground_below_its_surface_and_air_above() {
        for (style, inward) in [
            (Style::Verdant, false),
            (Style::HollowOuter, false),
            (Style::HollowInner, true),
            (Style::Moon { tone: 0 }, false),
            (Style::Moon { tone: 1 }, false),
            (Style::Moon { tone: 2 }, false),
            (Style::Ember, false),
        ] {
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

    /// Craters are bowls with rims: across the moon's surface the relief spans deep pits and raised
    /// rims, and they reach across a seam like everything else.
    #[test]
    fn moons_are_cratered() {
        let r = round(Style::Moon { tone: 0 }, false);
        let b = r.atlas.bands[0];
        let top = Patch::Shell { band: 0, face: Face::PosY };
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for i in (0..b.n).step_by(97) {
            for j in (0..b.n).step_by(89) {
                let p = r.atlas.embed(top, DVec3::new(i as f64 + 0.5, (r.atlas.radius - b.r_lo) as f64, j as f64 + 0.5));
                let c = r.craters(p);
                lo = lo.min(c);
                hi = hi.max(c);
            }
        }
        assert!(lo < -40.0 && hi > 4.0, "crater relief spans {lo}..{hi}");
    }

    #[test]
    fn batch_fill_equals_the_per_cell_definition() {
        for style in [
            Style::Verdant,
            Style::HollowOuter,
            Style::HollowInner,
            Style::Moon { tone: 0 },
            Style::Moon { tone: 1 },
            Style::Moon { tone: 2 },
            Style::Ember,
        ] {
            batch_matches(style);
        }
    }

    fn batch_matches(style: Style) {
        let r = round(style, style == Style::HollowInner);
        let b = r.atlas.bands[0];
        let patch = Patch::Shell { band: 0, face: Face::NegX };
        let (i, j) = (b.n / 2, b.n / 2);
        let col = r.column(patch, i, j);
        // The surface chunk and two in the cave band: every cell.
        for depth in [0, 40, 130] {
            let s = r.atlas.storage(patch, [i, col.surface - depth, j]);
            let c = [s[0].div_euclid(16), s[1].div_euclid(16), s[2].div_euclid(16)];
            let data = r.fill_chunk(c);
            for k in 0..CHUNK_VOLUME {
                let (lx, ly, lz) = (k % 16, k / 256, k / 16 % 16);
                let cell = [c[0] * 16 + lx as i64, c[1] * 16 + ly as i64, c[2] * 16 + lz as i64];
                assert_eq!(data.get(Chunk::index(lx, ly, lz)), r.voxel(cell), "{style:?} depth {depth} cell {lx},{ly},{lz}");
            }
        }
    }

    /// The cap faces the parent. +Y is a world axis and is not a cap when the pole is +X.
    #[test]
    fn moon_ice_caps_follow_the_parent_not_a_world_axis() {
        let mut reg = BlockRegistry::with_builtins();
        let m = Arc::new(Materials::intern(&mut reg));
        let c = DVec3::new(5.0e8, -2.0e8, 1.0e8);
        let atlas = Atlas::new(c, 60_000, 62_048, false, 0);
        let pole = DVec3::X;
        let r = Round::new(atlas, 100, Style::Moon { tone: 0 }, m, pole);
        assert!(r.ice_cap(DVec3::X));
        assert!(!r.ice_cap(DVec3::Y));
        assert!(!r.ice_cap(DVec3::new(0.0, 0.8, 0.6)));
        let bare = Round::new(r.atlas.clone(), 99, Style::Moon { tone: 0 }, r.m.clone(), pole);
        assert!(!bare.ice_cap(DVec3::X), "seed 99 grows no cap");
        let b = r.atlas.bands[0];
        let (i, j) = (b.n / 2, b.n / 2);
        let cap = r.column(Patch::Shell { band: 0, face: Face::PosX }, i, j);
        let mid = r.column(Patch::Shell { band: 0, face: Face::PosY }, i, j);
        assert_eq!(cap.top, r.m.snow, "the outward face is the cap");
        assert_ne!(mid.top, r.m.snow, "+Y is not the cap");
    }

    #[test]
    fn moons_fissures_boulders_and_the_worlds_have_character() {
        let verdant = round(Style::Verdant, false);
        let b = verdant.atlas.bands[0];
        let face = Patch::Shell { band: 0, face: Face::PosZ };
        let mut lake = false;
        let mut meadow = false;
        for i in (b.n / 5..b.n * 4 / 5).step_by(900) {
            for j in (b.n / 5..b.n * 4 / 5).step_by(900) {
                let col = verdant.column(face, i, j);
                lake |= col.top == verdant.m.frost;
                meadow |= col.top == verdant.m.meadow;
            }
        }
        assert!(lake, "a verdant basin holds a frost lake");
        assert!(meadow, "lowlands are meadow");
        let mut tree = None;
        'sites: for si in 2..80 {
            for sj in 2..80 {
                if let Some(p) = verdant.site(face, b.n, si, sj) {
                    tree = Some(p);
                    break 'sites;
                }
            }
        }
        let tree = tree.expect("a verdant tree");
        assert!(tree.half + 5 < super::SITE_MARGIN && tree.crown < super::SITE_MARGIN, "the tree stays inside the margin");
        let buttress = super::verdant_tree(&verdant.m, &tree, tree.half + 2, 0, 1);
        assert_eq!(buttress, Some(verdant.m.bark), "buttress roots");
        let canopy = super::verdant_tree(&verdant.m, &tree, 0, 0, tree.height);
        assert_eq!(canopy, Some(verdant.m.leaves), "a canopy layer");
        let above = super::verdant_tree(&verdant.m, &tree, 0, 0, tree.height + 8);
        assert_eq!(above, None, "the crown stays low");

        let dusty = round(Style::Moon { tone: 0 }, false);
        let frozen = round(Style::Moon { tone: 1 }, false);
        let rust = round(Style::Moon { tone: 2 }, false);
        let mb = dusty.atlas.bands[0];
        let mf = Patch::Shell { band: 0, face: Face::NegZ };
        let (mut glow, mut magma, mut frozen_glow) = (false, false, false);
        for i in (0..mb.n).step_by(1_100) {
            for j in (0..mb.n).step_by(1_300) {
                glow |= dusty.column(mf, i, j).top == dusty.m.glowshroom;
                magma |= rust.column(mf, i, j).top == rust.m.magma;
                let top = frozen.column(mf, i, j).top;
                frozen_glow |= top == frozen.m.glowshroom || top == frozen.m.magma;
            }
        }
        assert!(glow, "dusty moons have glowing fissures");
        assert!(magma, "rust moons have magma fissures");
        assert!(!frozen_glow, "a frozen moon has no glowing fissure");
        let mut heap = None;
        'rim: for si in 0..120 {
            for sj in 0..120 {
                if let Some(p) = dusty.site(mf, mb.n, si, sj) {
                    heap = Some(p);
                    break 'rim;
                }
            }
        }
        let heap = heap.expect("a boulder on a crater rim");
        assert!(heap.half < super::SITE_MARGIN);
        assert_eq!(super::boulder(&dusty.m, &heap, 0, 0, 0), Some(dusty.m.regolith));

        let ember = round(Style::Ember, false);
        let eb = ember.atlas.bands[0];
        let ef = Patch::Shell { band: 0, face: Face::PosX };
        let mut river = false;
        for i in (0..eb.n).step_by(800) {
            for j in (0..eb.n).step_by(800) {
                river |= ember.column(ef, i, j).top == ember.m.magma;
            }
        }
        assert!(river, "the Ember has a magma river");
        let mut column = None;
        'col: for si in 0..100 {
            for sj in 0..100 {
                if let Some(p) = ember.site(ef, eb.n, si, sj) {
                    column = Some(p);
                    break 'col;
                }
            }
        }
        let column = column.expect("a basalt column");
        assert_eq!(super::basalt_column(&ember.m, &column, 0, 0, column.height / 2), Some(ember.m.basalt));
        assert_eq!(
            super::basalt_column(&ember.m, &column, column.half, 0, column.height / 2),
            Some(ember.m.basalt),
            "the prism does not taper"
        );

        let inner = round(Style::HollowInner, true);
        let ib = inner.atlas.bands[0];
        let inf = Patch::Shell { band: 0, face: Face::PosY };
        let mut forest = None;
        'cry: for si in 0..60 {
            for sj in 0..60 {
                if let Some(p) = inner.site(inf, ib.n, si, sj) {
                    forest = Some(p);
                    break 'cry;
                }
            }
        }
        let forest = forest.expect("a crystal tree");
        assert!(forest.half + 5 < super::SITE_MARGIN);
        assert_eq!(super::crystal_forest(&inner.m, &forest, 0, 0, forest.height), Some(inner.m.glowshroom), "a glowing tip");
        assert_eq!(
            super::crystal_forest(&inner.m, &forest, forest.half + 3, 0, forest.height / 3),
            Some(inner.m.crystal),
            "an arm"
        );
        assert!(forest.height + 3 < 120, "the forest stays under the sky");
    }
}
