//! The surface: heightfield, climate, biomes and the layers under them.
//!
//! Height is built from a few large fields, all domain-warped so nothing lines up with the axes:
//! continentalness (basins ↔ highlands), mountain *ranges* inside the highlands (ridged
//! multifractal crests over derivative-eroded flanks), hills everywhere, and long **valleys** — the
//! zero set of a low-frequency field — that carve through lowlands and ranges alike. In the arid
//! belt the profile is terraced into mesas. Every cliff shows banded strata that wave across the
//! land.

use std::sync::Arc;

use super::noise::{eroded2, fbm2, hash2, perlin2, ridged2, smoothstep};
use super::{Materials, MAX_GROUND, MIN_GROUND};
use crate::block::registry::BlockId;

/// What grows and lies on a column.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Biome {
    Plains,
    Forest,
    Autumn,
    Cherry,
    Desert,
    Mesa,
    Alpine,
    Snow,
}

/// One column's description: height, biome, slope and surface layers.
#[derive(Clone, Copy, Debug)]
pub struct Column {
    /// First air cell above the ground.
    pub height: i32,
    pub biome: Biome,
    /// Steepest neighbouring height step, in blocks per block (×4, saturating).
    pub slope4: u8,
    /// The top cell.
    pub surface: BlockId,
    /// The layer under it.
    pub sub: BlockId,
    /// Cells of `sub` under the top cell.
    pub sub_depth: i32,
    /// Strata family: sandstone (true) or rock bands.
    pub mesa: bool,
    /// Vertical offset of the strata bands here (they wave across the land).
    pub warp: i32,
    /// Below this height lies deep rock.
    pub deep: i32,
}

/// The surface generator.
pub struct Shape {
    s: u32,
    relief: f32,
    m: Arc<Materials>,
}

/// Raw field values at a column.
struct Field {
    h: f32,
    arid: f32,
    cont: f32,
}

/// Soft ceiling: mountains approach [`MAX_GROUND`] without flat-topping on it.
fn soft_ceiling(h: f32) -> f32 {
    const KNEE: f32 = 380.0;
    let room = (MAX_GROUND - 2) as f32 - KNEE;
    if h <= KNEE { h } else { KNEE + (h - KNEE) * room / (room + (h - KNEE)) }
}

impl Shape {
    pub fn new(s: u32, relief: f32, m: Arc<Materials>) -> Self {
        Self { s, relief, m }
    }

    fn seed(&self, k: u32) -> u32 {
        self.s.wrapping_mul(0x9E37_79B9).wrapping_add(k.wrapping_mul(0x85EB_CA6B))
    }

    fn field(&self, x: i32, z: i32) -> Field {
        let (xf, zf) = (x as f64, z as f64);
        // Domain warp: bends ranges, valleys and coasts away from the noise lattice.
        let wx = xf + 260.0 * fbm2(self.seed(0), xf / 1100.0, zf / 1100.0, 3, 0.5) as f64;
        let wz = zf + 260.0 * fbm2(self.seed(1), xf / 1100.0 + 31.7, zf / 1100.0 - 12.9, 3, 0.5) as f64;
        let cont = (0.5 + 0.8 * fbm2(self.seed(2), wx / 3200.0, wz / 3200.0, 4, 0.5)).clamp(0.0, 1.0);
        let ranges = smoothstep(0.42, 0.72, cont);
        let erosion = eroded2(self.seed(4), wx / 520.0, wz / 520.0, 6);
        let mountain = if ranges > 0.0 {
            let r = ridged2(self.seed(3), wx / 820.0, wz / 820.0, 5);
            let peaks = r * r * 1.2 + 0.32 * (0.5 + 0.5 * erosion);
            self.relief * 280.0 * peaks * ranges
        } else {
            0.0
        };
        let hills = 24.0 * fbm2(self.seed(5), wx / 300.0, wz / 300.0, 4, 0.5) + 9.0 * erosion;
        let base = 38.0 + 72.0 * cont;
        // Valleys: where a slow field crosses zero the land is carved down toward the base, through
        // hills and ranges alike — long valleys with steep flanks.
        let vn = fbm2(self.seed(6), wx / 1400.0, wz / 1400.0, 3, 0.45).abs();
        let open = smoothstep(0.0, 0.09, vn);
        let shoulders = open * open * (3.0 - 2.0 * open);
        let mut h = base + hills * (0.3 + 0.7 * open) + mountain * shoulders - 20.0 * (1.0 - open);
        // The arid belt is terraced into mesas: flat steps with short steep risers.
        let arid = smoothstep(0.56, 0.8, 0.5 + 0.6 * fbm2(self.seed(7), xf / 2200.0, zf / 2200.0, 3, 0.5));
        if arid > 0.0 {
            let step = 13.0;
            let t = h / step;
            let fl = t.floor();
            let terraced = (fl + smoothstep(0.4, 0.62, t - fl)) * step;
            h += (terraced - h) * arid * (1.0 - 0.6 * ranges);
        }
        h += 2.2 * perlin2(self.seed(8), xf / 23.0, zf / 23.0);
        Field { h: soft_ceiling(h), arid, cont }
    }

    /// First air cell above the ground at `(x, z)`.
    pub fn height(&self, x: i32, z: i32) -> i32 {
        (self.field(x, z).h.floor() as i32).clamp(MIN_GROUND, MAX_GROUND)
    }

    /// The column at `(x, z)`, from its own field and four neighbouring heights.
    pub fn column(&self, x: i32, z: i32) -> Column {
        let f = self.field(x, z);
        let height = (f.h.floor() as i32).clamp(MIN_GROUND, MAX_GROUND);
        let dx = (self.height(x + 1, z) - self.height(x - 1, z)).unsigned_abs();
        let dz = (self.height(x, z + 1) - self.height(x, z - 1)).unsigned_abs();
        let slope4 = (dx.max(dz) * 2).min(255) as u8;
        self.describe(x, z, height, slope4, &f)
    }

    fn describe(&self, x: i32, z: i32, height: i32, slope4: u8, f: &Field) -> Column {
        let m = &self.m;
        let (xf, zf) = (x as f64, z as f64);
        let temp = 0.55 + 0.5 * fbm2(self.seed(9), xf / 3400.0, zf / 3400.0, 3, 0.5) - (height - 90) as f32 / 520.0;
        let humid = 0.5 + 0.6 * fbm2(self.seed(10), xf / 1600.0, zf / 1600.0, 3, 0.5);
        let variety = perlin2(self.seed(11), xf / 700.0, zf / 700.0);
        let snowline = 300 + (26.0 * perlin2(self.seed(12), xf / 90.0, zf / 90.0)) as i32;
        let biome = if height > snowline || temp < 0.12 {
            Biome::Snow
        } else if height > 236 {
            Biome::Alpine
        } else if f.arid > 0.35 {
            if f.cont < 0.5 { Biome::Desert } else { Biome::Mesa }
        } else if humid > 0.62 {
            if variety > 0.45 { Biome::Autumn } else { Biome::Forest }
        } else if variety < -0.5 {
            Biome::Cherry
        } else {
            Biome::Plains
        };
        let mesa = matches!(biome, Biome::Mesa | Biome::Desert) || f.arid > 0.5;
        let warp = (11.0 * perlin2(self.seed(13), xf / 170.0, zf / 170.0)) as i32;
        let deep = -40 + (14.0 * perlin2(self.seed(14), xf / 230.0, zf / 230.0)) as i32;
        let patch = perlin2(self.seed(15), xf / 19.0, zf / 19.0);
        let (mut surface, mut sub, mut sub_depth) = match biome {
            Biome::Plains => (if patch > 0.15 { m.meadow } else { m.grass }, m.soil, 3),
            Biome::Forest | Biome::Cherry => (m.grass, m.soil, 4),
            Biome::Autumn => (if patch > 0.3 { m.meadow } else { m.grass }, m.soil, 4),
            Biome::Desert => (m.sand, m.sand, 5),
            Biome::Mesa => (m.redsand, m.redsand, 2),
            Biome::Alpine => (m.gravel, m.gravel, 2),
            Biome::Snow => (m.snow, m.snow, 2),
        };
        let steep = slope4 >= 6;
        let col = Column { height, biome, slope4, surface, sub, sub_depth, mesa, warp, deep };
        if steep && !(biome == Biome::Snow && slope4 < 9) {
            // Cliffs: bare banded strata.
            surface = self.stratum(&col, height - 1);
            sub = surface;
            sub_depth = 0;
        }
        Column { surface, sub, sub_depth, ..col }
    }

    /// The banded stratum at height `y` in this column.
    pub fn stratum(&self, col: &Column, y: i32) -> BlockId {
        let m = &self.m;
        if y < col.deep - 260 {
            return m.abyss;
        }
        if y < col.deep {
            return m.deeprock;
        }
        let (thick, set) = if col.mesa { (5, &m.sandstone) } else { (9, &m.rock) };
        let band = (y + col.warp).div_euclid(thick);
        set[(hash2(self.seed(16), band, col.mesa as i32) % 4) as usize]
    }

    /// Ground at `(x, y, z)` with `y` below the column height: surface layers, then strata.
    pub fn ground(&self, col: &Column, _x: i32, y: i32, _z: i32) -> BlockId {
        let d = col.height - y;
        if d == 1 {
            col.surface
        } else if d <= col.sub_depth + 1 {
            col.sub
        } else {
            self.stratum(col, y)
        }
    }
}
