//! Beneath the surface: caves, ore veins, and abandoned mines.
//!
//! Caves and veins come from four smooth 3-D fields sampled on a 4-block lattice and interpolated
//! trilinearly (a chunk needs 5×7×5 samples, not 4096): two fields whose joint zero set is a web
//! of **tunnels**, one whose high crests open **caverns** that grow with depth, one for **veins**.
//! The deep is dressed: glowing fungus on cavern floors, crystal on the deepest ceilings.
//!
//! **Mines** sit on a 160 m site grid. A site has two to four levels; each level is a grid of
//! corridors (3 wide, 3 tall) between nodes 16 m apart, some edges missing, some collapsed. Timber
//! frames every four metres, a rail down the middle, a rare lamp hanging from a beam, rooms at some
//! nodes and shafts down to the next level. Mostly dark — the light is where the miners left it.

use std::sync::Arc;

use super::noise::{hash2, hash3, perlin3, unit};
use super::shape::Column;
use super::{Materials, TerrainCfg};
use crate::block::registry::{AIR, BlockId};

/// Field channels: tunnel A, tunnel B, cavern, vein.
pub const CH: usize = 4;
/// Lattice spacing of the coarse fields.
const LATTICE: i32 = 4;

/// Coarse field samples covering one chunk (x and z: 5 points; y: 7 points from `y0 − 4`).
pub struct Grid {
    vals: Vec<[f32; CH]>,
    x0: i32,
    y0: i32,
    z0: i32,
}

const GX: usize = 5;
const GY: usize = 7;

impl Grid {
    /// Trilinear interpolation of eight lattice corners at a voxel. Corner order: bit 0 = +x,
    /// bit 1 = +y, bit 2 = +z. The one interpolation both the batch and the per-voxel paths use.
    #[inline]
    pub fn interp_corners(c: &[[f32; CH]; 8], x: i32, y: i32, z: i32) -> [f32; CH] {
        let fx = x.rem_euclid(LATTICE) as f32 * 0.25;
        let fy = y.rem_euclid(LATTICE) as f32 * 0.25;
        let fz = z.rem_euclid(LATTICE) as f32 * 0.25;
        let mut out = [0.0f32; CH];
        for (k, o) in out.iter_mut().enumerate() {
            let x00 = c[0][k] + (c[1][k] - c[0][k]) * fx;
            let x10 = c[2][k] + (c[3][k] - c[2][k]) * fx;
            let x01 = c[4][k] + (c[5][k] - c[4][k]) * fx;
            let x11 = c[6][k] + (c[7][k] - c[6][k]) * fx;
            let y0 = x00 + (x10 - x00) * fy;
            let y1 = x01 + (x11 - x01) * fy;
            *o = y0 + (y1 - y0) * fz;
        }
        out
    }

    fn sample(&self, ix: usize, iy: usize, iz: usize) -> [f32; CH] {
        self.vals[ix + GX * (iy + GY * iz)]
    }

    /// The fields at chunk-local `(lx, ly, lz)`; `ly` may be −1 or 16 (one cell beyond the chunk).
    pub fn at(&self, lx: usize, ly: i32, lz: usize) -> [f32; CH] {
        let (x, y, z) = (self.x0 + lx as i32, self.y0 + ly, self.z0 + lz as i32);
        let ix = (x.div_euclid(LATTICE) - self.x0.div_euclid(LATTICE)) as usize;
        let iy = (y.div_euclid(LATTICE) - (self.y0 - LATTICE).div_euclid(LATTICE)) as usize;
        let iz = (z.div_euclid(LATTICE) - self.z0.div_euclid(LATTICE)) as usize;
        let c = std::array::from_fn(|i| self.sample(ix + (i & 1), iy + ((i >> 1) & 1), iz + ((i >> 2) & 1)));
        Self::interp_corners(&c, x, y, z)
    }
}

/// Mine site edge, in metres.
const SITE: i32 = 160;
/// Corridor node spacing.
const NODE: i32 = 16;
/// Nodes per site axis (corridors stay `MARGIN` inside the site).
const NODES: i32 = 9;
const MARGIN: i32 = 16;
/// The heights mine levels may sit at.
const LEVELS: [i32; 6] = [-14, -38, -62, -86, -110, -134];

/// What a mine puts in a cell.
enum MineCell {
    Air,
    Block(BlockId),
}

pub struct Underground {
    s: u32,
    caves: f32,
    mines: f32,
    m: Arc<Materials>,
}

impl Underground {
    pub fn new(s: u32, cfg: TerrainCfg, m: Arc<Materials>) -> Self {
        Self { s, caves: cfg.caves as f32 / 100.0, mines: cfg.mines as f32 / 100.0, m }
    }

    fn seed(&self, k: u32) -> u32 {
        self.s.wrapping_mul(0x2545_F491).wrapping_add(k.wrapping_mul(0x9E37_79B9))
    }

    /// The coarse fields at a lattice point.
    fn point(&self, x: i32, y: i32, z: i32) -> [f32; CH] {
        let (xf, yf, zf) = (x as f64, y as f64, z as f64);
        let a = perlin3(self.seed(0), xf / 56.0, yf / 34.0, zf / 56.0);
        let b = perlin3(self.seed(1), xf / 56.0, yf / 34.0, zf / 56.0);
        let c = 0.65 * perlin3(self.seed(2), xf / 110.0, yf / 64.0, zf / 110.0)
            + 0.35 * perlin3(self.seed(3), xf / 42.0, yf / 30.0, zf / 42.0);
        let v = perlin3(self.seed(4), xf / 13.0, yf / 13.0, zf / 13.0);
        [a, b, c, v]
    }

    /// The eight lattice corners around a voxel (per-voxel path).
    pub fn corners(&self, x: i32, y: i32, z: i32) -> [[f32; CH]; 8] {
        let (bx, by, bz) =
            (x.div_euclid(LATTICE) * LATTICE, y.div_euclid(LATTICE) * LATTICE, z.div_euclid(LATTICE) * LATTICE);
        std::array::from_fn(|i| {
            self.point(
                bx + (i as i32 & 1) * LATTICE,
                by + ((i as i32 >> 1) & 1) * LATTICE,
                bz + ((i as i32 >> 2) & 1) * LATTICE,
            )
        })
    }

    /// The coarse fields for a whole chunk (batch path).
    pub fn grid(&self, x0: i32, y0: i32, z0: i32) -> Grid {
        let mut vals = Vec::with_capacity(GX * GY * GX);
        let (bx, by, bz) = (x0.div_euclid(LATTICE), (y0 - LATTICE).div_euclid(LATTICE), z0.div_euclid(LATTICE));
        for iz in 0..GX as i32 {
            for iy in 0..GY as i32 {
                for ix in 0..GX as i32 {
                    vals.push(self.point((bx + ix) * LATTICE, (by + iy) * LATTICE, (bz + iz) * LATTICE));
                }
            }
        }
        Grid { vals, x0, y0, z0 }
    }

    /// Whether the fields open a cave at this depth below the surface.
    fn carved(&self, v: [f32; CH], depth: i32) -> bool {
        if self.caves <= 0.0 || depth < 3 {
            return false;
        }
        let d = depth as f32;
        // Tunnels: the joint zero set of two fields, widening a little with depth.
        let r = (0.075 + 0.045 * (d / 300.0).min(1.0)) * self.caves.sqrt();
        if v[0] * v[0] + v[1] * v[1] < r * r {
            return true;
        }
        // Caverns: crests of a slow field, opening further the deeper you go.
        if depth > 28 {
            let t = ((d - 28.0) / 320.0).min(1.0);
            let thr = 0.62 - 0.24 * t - 0.08 * (self.caves - 1.0);
            return v[2] > thr;
        }
        false
    }

    /// The finished cell below the surface: mines, caves (with their dressing), veins, ground.
    #[allow(clippy::too_many_arguments)]
    pub fn finish(
        &self,
        col: &Column,
        x: i32,
        y: i32,
        z: i32,
        ground: BlockId,
        v: [f32; CH],
        below: impl Fn() -> [f32; CH],
        above: impl Fn() -> [f32; CH],
    ) -> BlockId {
        let depth = col.height - y;
        if let Some(cell) = self.mine(col, x, y, z) {
            return match cell {
                MineCell::Air => AIR,
                MineCell::Block(id) => id,
            };
        }
        if self.carved(v, depth) {
            let m = &self.m;
            let h = hash3(self.seed(20), x, y, z);
            // Deep floors grow glowing fungus; the deepest ceilings grow crystal.
            if depth > 70 && unit(h) < 0.07 && !self.carved(below(), depth + 1) {
                return m.glowcap;
            }
            if depth > 180 && unit(h) < 0.05 && !self.carved(above(), depth - 1) {
                return m.crystal;
            }
            return AIR;
        }
        if depth > col.sub_depth + 1
            && v[3] > 0.56
            && let Some(ore) = self.vein(x, y, z, depth, ground)
        {
            return ore;
        }
        ground
    }

    /// The vein mineral at a vein cell, if this ground hosts one.
    fn vein(&self, x: i32, y: i32, z: i32, depth: i32, ground: BlockId) -> Option<BlockId> {
        let m = &self.m;
        let stratum = m.rock.contains(&ground)
            || m.sandstone.contains(&ground)
            || ground == m.deeprock
            || ground == m.abyss;
        if !stratum {
            return None;
        }
        let cell = hash3(self.seed(21), x.div_euclid(8), y.div_euclid(8), z.div_euclid(8));
        if depth > 20 && cell.is_multiple_of(5) {
            // A reagent vein, only in a stratum it lies dormant in.
            let hosted: Vec<BlockId> = m.reagents.iter().filter(|r| r.hosts.contains(&ground)).map(|r| r.id).collect();
            if !hosted.is_empty() {
                return Some(hosted[(cell / 5) as usize % hosted.len()]);
            }
        }
        let pick = (cell >> 8) % 100;
        Some(if depth < 60 {
            if pick < 60 { m.copper } else { m.azurite }
        } else if depth < 170 {
            if pick < 55 { m.azurite } else { m.gold }
        } else if pick < 50 {
            m.gold
        } else {
            m.crystal
        })
    }

    /// The mine structure at a cell, if any.
    fn mine(&self, col: &Column, x: i32, y: i32, z: i32) -> Option<MineCell> {
        if self.mines <= 0.0 || y > LEVELS[0] + 4 || y < LEVELS[LEVELS.len() - 1] - 1 {
            return None;
        }
        let (sx, sz) = (x.div_euclid(SITE), z.div_euclid(SITE));
        let site = hash2(self.seed(30), sx, sz);
        if unit(site) >= 0.45 * self.mines {
            return None;
        }
        let (u, w) = (x - sx * SITE - MARGIN, z - sz * SITE - MARGIN);
        let span = NODE * (NODES - 1);
        if !(-3..=span + 3).contains(&u) || !(-3..=span + 3).contains(&w) {
            return None;
        }
        // Two to four levels per site: a bit per level, at least one forced on.
        let mask = (site >> 8) as usize & ((1 << LEVELS.len()) - 1) | 1 << (site % 3);
        for (i, &level) in LEVELS.iter().enumerate() {
            if mask & (1 << i) == 0 || !(level - 1..=level + 4).contains(&y) && y >= level {
                continue;
            }
            if level + 8 > col.height {
                continue; // never breach the surface
            }
            let lower = (i + 1..LEVELS.len()).find(|&j| mask & (1 << j) != 0).map(|j| LEVELS[j]);
            if let Some(c) = self.level_cell(site, level, lower, u, y, w, x, z) {
                return Some(c);
            }
        }
        None
    }

    /// Shallowest mine level under `(x, z)` that a shaft can meet: inside the site, not breaching
    /// `surface`, and still in the crust. `None` when the column is not on a live site.
    pub(super) fn mine_floor(&self, x: i32, z: i32, surface: i32) -> Option<i32> {
        if self.mines <= 0.0 {
            return None;
        }
        let (sx, sz) = (x.div_euclid(SITE), z.div_euclid(SITE));
        let site = hash2(self.seed(30), sx, sz);
        if unit(site) >= 0.45 * self.mines {
            return None;
        }
        let (u, w) = (x - sx * SITE - MARGIN, z - sz * SITE - MARGIN);
        let span = NODE * (NODES - 1);
        if !(-3..=span + 3).contains(&u) || !(-3..=span + 3).contains(&w) {
            return None;
        }
        let mask = (site >> 8) as usize & ((1 << LEVELS.len()) - 1) | 1 << (site % 3);
        for (i, &level) in LEVELS.iter().enumerate() {
            if mask & (1 << i) == 0 || level + 8 > surface || surface - level >= super::cube::CRUST {
                continue;
            }
            return Some(level);
        }
        None
    }

    #[allow(clippy::too_many_arguments)]
    fn level_cell(&self, site: u32, level: i32, lower: Option<i32>, u: i32, y: i32, w: i32, x: i32, z: i32) -> Option<MineCell> {
        let m = &self.m;
        let key = site ^ (level as u32).wrapping_mul(0x9E37_79B9);
        let edge_x = |ix: i32, iz: i32| (0..NODES - 1).contains(&ix) && (0..NODES).contains(&iz) && unit(hash2(key, ix * 2, iz)) < 0.62;
        let edge_z = |ix: i32, iz: i32| (0..NODES).contains(&ix) && (0..NODES - 1).contains(&iz) && unit(hash2(key, ix * 2 + 1, iz)) < 0.62;
        let (nx, nz) = ((u + NODE / 2).div_euclid(NODE), (w + NODE / 2).div_euclid(NODE));
        let (ou, ow) = (u - nx * NODE, w - nz * NODE);
        let node_live = edge_x(nx, nz) || edge_x(nx - 1, nz) || edge_z(nx, nz) || edge_z(nx, nz - 1);
        // Shafts down to the next level.
        if let Some(lo) = lower
            && node_live
            && unit(hash2(key ^ 0x5AF7, nx, nz)) < 0.14
            && ou.abs() <= 1
            && ow.abs() <= 1
            && (lo..=level + 2).contains(&y)
        {
            return Some(if ou.abs() == 1 && ow.abs() == 1 { MineCell::Block(m.plank) } else { MineCell::Air });
        }
        // Rooms at some nodes.
        if node_live
            && unit(hash2(key ^ 0x400A, nx, nz)) < 0.16
            && ou.abs() <= 3
            && ow.abs() <= 3
            && (level..=level + 3).contains(&y)
        {
            let pillar = ou.abs() == 3 && ow.abs() == 3;
            return Some(if pillar { MineCell::Block(m.plank) } else { MineCell::Air });
        }
        // Corridors: along x on row nz (|ow| ≤ 1), along z on column nx (|ou| ≤ 1).
        let along_x = ow.abs() <= 1 && edge_x(u.div_euclid(NODE), nz) && u.rem_euclid(NODE) != 0 || ow.abs() <= 1 && ou == 0 && node_live;
        let along_z = ou.abs() <= 1 && edge_z(nx, w.div_euclid(NODE)) && w.rem_euclid(NODE) != 0 || ou.abs() <= 1 && ow == 0 && node_live;
        if !(along_x || along_z) {
            return None;
        }
        let (run, across, segment) = if along_x { (u, ow, (u.div_euclid(NODE), nz, 0)) } else { (w, ou, (nx, w.div_euclid(NODE), 1)) };
        let at_node = ou == 0 && ow == 0;
        if y == level - 1 {
            // The rail runs down the middle of the floor.
            return (across == 0 && !at_node).then_some(MineCell::Block(m.rail));
        }
        if !(level..=level + 2).contains(&y) {
            return None;
        }
        let h = hash3(key ^ 0xB0E5, x, y, z);
        // Collapsed stretches.
        if unit(hash2(key ^ 0xC011, segment.0 * 2 + segment.2, segment.1)) < 0.14 && run.rem_euclid(NODE) > 4 {
            let pile = (y == level && unit(h) < 0.6) || (y == level + 1 && unit(h) < 0.25);
            if pile {
                return Some(MineCell::Block(m.rubble));
            }
        }
        // Timber frames every four metres: posts at the walls, a beam overhead.
        if run.rem_euclid(4) == 0 && !at_node {
            if across.abs() == 1 && y < level + 2 {
                return Some(MineCell::Block(m.plank));
            }
            if y == level + 2 {
                if across == 0 && unit(hash2(key ^ 0x1A3B, run, segment.0 + segment.1 * 64)) < 0.1 {
                    return Some(MineCell::Block(m.lamp));
                }
                return Some(MineCell::Block(m.plank));
            }
        }
        if y == level && unit(h) < 0.008 {
            return Some(MineCell::Block(m.bone));
        }
        Some(MineCell::Air)
    }
}
