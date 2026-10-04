//! The surface: a themed heightfield and the layers under it.
//!
//! Height is the same large fields as before — domain-warped continentalness, ridged ranges,
//! eroded hills, valleys — scaled by the column's province. Terraces, dunes and flatness come
//! from the theme. Climate (geopotential plus noise) sets the snowline; cliffs still show strata.

use std::sync::Arc;

use super::noise::{eroded2, fbm2, hash2, perlin2, ridged2, smoothstep, unit};
use super::province::{self, Petals, Place, Realm, Species, Strata, Surf, ThemeId, FEAT};
use super::{Materials, MAX_GROUND, MIN_GROUND};
use crate::block::registry::{AIR, BlockId};
use crate::coord::Face;

/// One column's description: height, theme, slope and surface layers.
#[derive(Clone, Copy, Debug)]
pub struct Column {
    /// First air cell above the ground.
    pub height: i32,
    /// Steepest neighbouring height step, in blocks per block (×4, saturating).
    pub slope4: u8,
    /// The top cell.
    pub surface: BlockId,
    /// The layer under it.
    pub sub: BlockId,
    /// Cells of `sub` under the top cell.
    pub sub_depth: i32,
    /// Strata family under the soil.
    pub strata: Strata,
    /// Vertical offset of the strata bands here (they wave across the land).
    pub warp: i32,
    /// Below this height lies deep rock.
    pub deep: i32,
    /// Nearest province's theme. The map paints this.
    pub theme: ThemeId,
    pub species: Species,
    /// Planting probability. Zero when the dithered species is none, and lower on snow.
    pub trees: f32,
    /// Flower probability. Zero on snow, cliff, or a bare surface.
    pub flowers: f32,
    /// The flower block, or air when `flowers` is zero.
    pub flower: BlockId,
    /// Blended feature densities for later passes.
    pub feats: [f32; FEAT],
}

/// The surface generator.
pub struct Shape {
    s: u32,
    relief: f32,
    m: Arc<Materials>,
    provinces: province::Provinces,
}

/// Raw field values at a column. The province is part of the field, sampled once.
struct Field {
    h: f32,
    place: Place,
}

/// Soft ceiling: mountains approach [`MAX_GROUND`] without flat-topping on it.
fn soft_ceiling(h: f32) -> f32 {
    const KNEE: f32 = 380.0;
    let room = (MAX_GROUND - 2) as f32 - KNEE;
    if h <= KNEE { h } else { KNEE + (h - KNEE) * room / (room + (h - KNEE)) }
}

fn lifted(h: f32) -> i32 {
    (h.floor() as i32).clamp(MIN_GROUND, MAX_GROUND)
}

fn block_of(m: &Materials, surf: Surf) -> BlockId {
    match surf {
        Surf::Grass => m.grass,
        Surf::Meadow => m.meadow,
        Surf::Sand => m.sand,
        Surf::Redsand => m.redsand,
        Surf::Snow => m.snow,
        Surf::Gravel => m.gravel,
        Surf::Moss => m.moss,
        Surf::Tundra => m.tundra,
        Surf::Lichen => m.lichen,
        Surf::Ash => m.ash,
        Surf::Salt => m.salt,
        Surf::Clay => m.clay,
        Surf::Limestone => m.limestone,
        Surf::Basalt => m.basalt,
        Surf::Bone => m.bone,
        Surf::Marble => m.marble,
        Surf::Petrified => m.petrified,
        Surf::Regolith => m.regolith,
        Surf::Mud => m.mud,
        Surf::Ice => m.ice,
        Surf::Obsidian => m.obsidian,
        Surf::Soil => m.soil,
    }
}

impl Shape {
    /// `s` salts the height. `body_seed` salts provinces, shared by the body's six faces.
    pub fn new(
        s: u32,
        relief: f32,
        m: Arc<Materials>,
        face: Face,
        half: i64,
        realm: Realm,
        body_seed: u32,
        variety: f32,
        garden: bool,
    ) -> Self {
        Self {
            s,
            relief,
            m,
            provinces: province::Provinces::new(body_seed, variety, realm, face, half, garden),
        }
    }

    fn seed(&self, k: u32) -> u32 {
        self.s.wrapping_mul(0x9E37_79B9).wrapping_add(k.wrapping_mul(0x85EB_CA6B))
    }

    /// Blended feature densities, without the height field.
    pub(super) fn feats_at(&self, u: i32, v: i32) -> [f32; FEAT] {
        self.provinces.at(u, v).feats
    }

    /// Blocks of margin inside the face square.
    pub(super) fn inset(&self, u: i32, v: i32) -> i64 {
        self.provinces.inset(u, v)
    }

    /// Cache identity: provinces, height salt, relief and the palette pointer.
    pub(super) fn cache_key(&self) -> u64 {
        let mut k = self.provinces.cache_key();
        k = k.wrapping_mul(0x1000_0000_01B3) ^ self.s as u64;
        k = k.wrapping_mul(0x1000_0000_01B3) ^ self.relief.to_bits() as u64;
        k.wrapping_mul(0x1000_0000_01B3) ^ Arc::as_ptr(&self.m) as u64
    }

    fn field(&self, x: i32, z: i32) -> Field {
        let place = self.provinces.at(x, z);
        let (xf, zf) = (x as f64, z as f64);
        // Domain warp: bends ranges, valleys and coasts away from the noise lattice.
        let wx = xf + 260.0 * fbm2(self.seed(0), xf / 1100.0, zf / 1100.0, 3, 0.5) as f64;
        let wz = zf + 260.0 * fbm2(self.seed(1), xf / 1100.0 + 31.7, zf / 1100.0 - 12.9, 3, 0.5) as f64;
        let cont = (0.5 + 0.8 * fbm2(self.seed(2), wx / 3200.0, wz / 3200.0, 4, 0.5)).clamp(0.0, 1.0);
        let ranges = smoothstep(0.42, 0.72, cont);
        let erosion = eroded2(self.seed(4), wx / 520.0, wz / 520.0, 6);
        let amp = self.relief * place.relief;
        let mountain = if ranges > 0.0 {
            let r = ridged2(self.seed(3), wx / 820.0, wz / 820.0, 5);
            let peaks = r * r * 1.2 + 0.32 * (0.5 + 0.5 * erosion);
            amp * 280.0 * peaks * ranges
        } else {
            0.0
        };
        let hills = (24.0 * fbm2(self.seed(5), wx / 300.0, wz / 300.0, 4, 0.5) + 9.0 * erosion) * place.hills;
        let base = 38.0 + 72.0 * cont + place.base;
        // Valleys: where a slow field crosses zero the land is carved down toward the base, through
        // hills and ranges alike — long valleys with steep flanks.
        let vn = fbm2(self.seed(6), wx / 1400.0, wz / 1400.0, 3, 0.45).abs();
        let open = smoothstep(0.0, 0.09, vn);
        let shoulders = open * open * (3.0 - 2.0 * open);
        let mut h = base + hills * (0.3 + 0.7 * open) + mountain * shoulders - 20.0 * (1.0 - open);
        // Mesas and terraces: flat steps with short steep risers, weak inside mountain ranges.
        if place.terrace > 0.0 {
            let step = 13.0;
            let t = h / step;
            let fl = t.floor();
            let terraced = (fl + smoothstep(0.4, 0.62, t - fl)) * step;
            h += (terraced - h) * place.terrace * (1.0 - 0.6 * ranges);
        }
        // Dunes: ridges across a slow wind. Skipped where the theme has none.
        if place.dune > 0.4 {
            let wu = fbm2(self.seed(17), xf / 4_200.0, zf / 4_200.0, 2, 0.5);
            let wv = fbm2(self.seed(18), xf / 4_200.0 + 40.0, zf / 4_200.0 - 17.0, 2, 0.5);
            let len = (wu * wu + wv * wv).sqrt().max(0.2);
            let (dx, dz) = ((wu / len) as f64, (wv / len) as f64);
            let across = -dz * xf + dx * zf;
            let along = dx * xf + dz * zf;
            let ridge = ridged2(self.seed(19), across / 86.0, along / 380.0, 3);
            h += place.dune * (ridge - 0.38);
        }
        // Salt flats and ash wastes pull the profile toward the plateau.
        h += (base - h) * place.flat;
        h += 2.2 * perlin2(self.seed(8), xf / 23.0, zf / 23.0) * (1.0 - 0.8 * place.flat);
        Field { h: soft_ceiling(h), place }
    }

    /// First air cell above the ground at `(x, z)`.
    pub fn height(&self, x: i32, z: i32) -> i32 {
        lifted(self.field(x, z).h)
    }

    /// The column at `(x, z)`, from its own field and four neighbouring heights.
    pub fn column(&self, x: i32, z: i32) -> Column {
        let f = self.field(x, z);
        let height = lifted(f.h);
        let dx = (self.height(x + 1, z) - self.height(x - 1, z)).unsigned_abs();
        let dz = (self.height(x, z + 1) - self.height(x, z - 1)).unsigned_abs();
        let slope4 = (dx.max(dz) * 2).min(255) as u8;
        self.describe(x, z, height, slope4, &f)
    }

    /// One chunk of columns. Edge heights are sampled so slopes match [`column`](Self::column).
    pub(super) fn columns_16(&self, u0: i32, v0: i32) -> Vec<Column> {
        const N: usize = 18;
        let mut heights = [0i32; N * N];
        let mut fields = Vec::with_capacity(256);
        for j in 0..N {
            for i in 0..N {
                let u = u0.wrapping_add(i as i32 - 1);
                let v = v0.wrapping_add(j as i32 - 1);
                if (1..17).contains(&i) && (1..17).contains(&j) {
                    let f = self.field(u, v);
                    heights[i + j * N] = lifted(f.h);
                    fields.push(f);
                } else {
                    heights[i + j * N] = self.height(u, v);
                }
            }
        }
        let mut cols = Vec::with_capacity(256);
        let mut k = 0;
        for lv in 0..16 {
            for lu in 0..16 {
                let (i, j) = (lu + 1, lv + 1);
                let at = |ii: usize, jj: usize| heights[ii + jj * N];
                let dx = at(i + 1, j).wrapping_sub(at(i - 1, j)).unsigned_abs();
                let dz = at(i, j + 1).wrapping_sub(at(i, j - 1)).unsigned_abs();
                let slope4 = (dx.max(dz) * 2).min(255) as u8;
                let u = u0.wrapping_add(lu as i32);
                let v = v0.wrapping_add(lv as i32);
                cols.push(self.describe(u, v, at(i, j), slope4, &fields[k]));
                k += 1;
            }
        }
        cols
    }

    fn describe(&self, x: i32, z: i32, height: i32, slope4: u8, f: &Field) -> Column {
        let place = &f.place;
        let skin = province::skin(place.pick);
        let (xf, zf) = (x as f64, z as f64);
        let snowline = (46.0 + place.temp * 380.0 + 22.0 * perlin2(self.seed(12), xf / 90.0, zf / 90.0)) as i32;
        let snow = height > snowline || place.temp < 0.12;
        let warp = (11.0 * perlin2(self.seed(13), xf / 170.0, zf / 170.0)) as i32;
        let deep = -40 + (14.0 * perlin2(self.seed(14), xf / 230.0, zf / 230.0)) as i32;
        let (mut surface, mut sub, mut sub_depth) =
            (block_of(&self.m, skin.surf), block_of(&self.m, skin.sub), skin.depth);
        if snow {
            let ice = place.temp < 0.12 && matches!(skin.strata, Strata::Ice | Strata::Crystal);
            surface = if ice { self.m.ice } else { self.m.snow };
            sub = self.m.snow;
            sub_depth = 2;
        }
        let trees = if snow { place.trees * 0.15 } else { place.trees };
        let (mut flowers, mut flower) = (place.flowers, self.petal(skin.petals, x, z));
        if flower == AIR || !province::grassy(skin.surf) || snow {
            flowers = 0.0;
            flower = AIR;
        }
        let col = Column {
            height,
            slope4,
            surface,
            sub,
            sub_depth,
            strata: skin.strata,
            warp,
            deep,
            theme: place.theme,
            species: skin.species,
            trees,
            flowers,
            flower,
            feats: place.feats,
        };
        let steep = slope4 >= 6;
        if steep && !(snow && slope4 < 9) {
            let rock = self.stratum(&col, height - 1);
            Column { surface: rock, sub: rock, sub_depth: 0, flowers: 0.0, flower: AIR, ..col }
        } else {
            col
        }
    }

    /// A flower on the ground cell, when the column's roll lands inside its density.
    pub fn flower_at(&self, col: &Column, x: i32, z: i32) -> Option<BlockId> {
        if col.flowers <= 0.0 {
            return None;
        }
        (unit(hash2(self.seed(21), x, z)) < col.flowers).then_some(col.flower)
    }

    fn petal(&self, petals: Petals, x: i32, z: i32) -> BlockId {
        let m = &self.m;
        let set: &[BlockId] = match petals {
            Petals::None => return AIR,
            Petals::White => return m.flower_white,
            Petals::Warm => &[m.flower_red, m.flower_yellow],
            Petals::Cool => &[m.flower_blue, m.flower_white],
            Petals::Mixed => &[m.flower_red, m.flower_yellow, m.flower_blue, m.flower_white],
        };
        set[(hash2(self.seed(22), x, z) as usize) % set.len()]
    }

    /// The four blocks a strata family cycles, and the band thickness.
    fn bands(&self, strata: Strata) -> ([BlockId; 4], i32) {
        let m = &self.m;
        match strata {
            Strata::Rock => (m.rock, 9),
            Strata::Sandstone => (m.sandstone, 5),
            Strata::Basalt => ([m.basalt, m.obsidian, m.slate, m.cinder], 9),
            Strata::Limestone => ([m.limestone, m.marble, m.sandstone[0], m.gravel], 9),
            Strata::Ice => ([m.ice, m.frost, m.snow, m.marble], 9),
            Strata::Ash => ([m.ash, m.cinder, m.basalt, m.gravel], 9),
            Strata::Bone => ([m.bone, m.limestone, m.marble, m.gravel], 9),
            Strata::Crystal => ([m.crystal, m.marble, m.violet, m.frost], 9),
        }
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
        let (set, thick) = self.bands(col.strata);
        let band = (y + col.warp).div_euclid(thick);
        set[(hash2(self.seed(16), band, col.strata as i32) % 4) as usize]
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

    #[cfg(test)]
    pub(super) fn place(&self, u: i32, v: i32) -> Place {
        self.provinces.at(u, v)
    }

    #[cfg(test)]
    pub(super) fn realm(&self) -> Realm {
        self.provinces.realm()
    }
}
