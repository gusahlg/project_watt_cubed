//! Integer InfiniteDiffusion v2: plates, continents, climate, and column fill.
//!
//! Macro phases 0 (plates/continents) and 1 (climate) write eight Q16 channels.
//! Erosion, hydrology, caves, sky, and far-LOD height are later tasks; this
//! file still floods to the same sea and dresses with the existing placement
//! vocabulary. All value math is wrapping 32-bit / Q16.16.

use std::cell::Cell;

use infinite_field::inoise::{
    cellular_q16, clamp_q16, fbm_q16, lerp_q16, mul_q16, rem_floor, ridged_q16, smoothstep_q16,
    uniform_q16, value_noise_q16, warp_q16, HALF, ONE,
};
use infinite_field::{IntField, IntScore, IntSpec, Stencil};

use super::chunk::{CHUNK_SIZE, CHUNK_VOLUME, Chunk, ChunkData};
use super::diffusion::DiffusionCfg;
use super::generation::{cell_hash, ColumnHeights, TerrainGenerator};
use super::placement::{self, SurfaceKind};
use crate::block::registry::{AIR, BlockId, BlockRegistry};

const SEA: i32 = 20;
const CHANNELS: u32 = 8;
const PHASES: u32 = 2;

/// Channel indices. Values are Q16.16 unless a packing note says otherwise.
pub const CH_ELEV: u32 = 0;
pub const CH_TEMP: u32 = 1;
pub const CH_HUMID: u32 = 2;
pub const CH_GEOL: u32 = 3;
pub const CH_WATER: u32 = 4;
#[allow(dead_code)] // WORLDGEN-DIFFUSION-V2-DESIGN-2026-09-10.md §2.2 (W3/W4)
pub const CH_CAVES: u32 = 5;
#[allow(dead_code)] // WORLDGEN-DIFFUSION-V2-DESIGN-2026-09-10.md §2.2 (W3/W4)
pub const CH_SKY: u32 = 6;
#[allow(dead_code)] // WORLDGEN-DIFFUSION-V2-DESIGN-2026-09-10.md §2.2 (W3/W4)
pub const CH_DETAIL: u32 = 7;

/// Geology packing (one i32):
///   bits [7:0]   region id (`cellular` id low 8 bits)
///   bits [15:8]  hardness as Q8 (0..=255)
///   bits [31:16] 0
///
/// Water packing (one i32):
///   bits [7:0]   river width in metres (0 = none). W2 writes 0.
///   bit  8       basin (1 = 5×5 elevation local minimum)
///   bits [31:9]  water-table height as Q16 metres with the low 9 bits
///                cleared (`table_q16 & !0x1FF`). 0 = no table.
///   W2 writes bit 8 only.
///
/// Caves packing (one i32, 0 in W2):
///   bits [7:0]   floor height (u8 metres, biased)
///   bits [15:8]  ceiling height (u8 metres, biased)
///   bits [23:16] width (u8 metres)
/// Sky and detail are 0 in W2.

const WATER_BASIN: i32 = 1 << 8;

const SALT_PLATE: u32 = 0x51A7_E01D;
const SALT_WARP: u32 = 0xC0DE_A11A;
const SALT_CONT: u32 = 0xC0A1_7E11;
const SALT_BELT: u32 = 0xBE17_0001;
const SALT_VOLC: u32 = 0xB0C1_A001;
const SALT_HUMID: u32 = 0xA11D_0002;

const CELL_PLATE: i32 = 3000;
const CELL_CONT: i32 = 2048;
const CELL_BELT: i32 = 512;
const CELL_VOLC: i32 = 1500;
const WARP_AMP: i32 = 256 * ONE;
const FBM4: i32 = ONE + HALF + ONE / 4 + ONE / 8;

const C_DEEP: i32 = (ONE * 22) / 100;
const C_SHELF: i32 = (ONE * 32) / 100;
const C_COAST: i32 = (ONE * 42) / 100;
const C_LOW: i32 = (ONE * 55) / 100;
const C_UP: i32 = (ONE * 75) / 100;

const BELT_AMP: i32 = 85 * ONE;
const LAPSE_Q16: i32 = 262;
const POINT_35: i32 = (35 * ONE) / 100;

const SURFACE_SCATTER_SALT: i64 = 0x3E7A_1B96_D4C8_205Fu64 as i64;
const ORE_B_SALT: i64 = 0x9D3A_44E1_0C67_B52Bu64 as i64;

/// cos(2π k / 256) in Q16 via Bhaskara sine on the folded angle.
const COS_TABLE: [i32; 256] = {
    let mut t = [0i32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = cos_k(i);
        i += 1;
    }
    t
};

const fn sin_bhaskara_q16(deg: i32) -> i32 {
    if deg <= 0 || deg >= 180 {
        return 0;
    }
    if deg == 90 {
        return ONE;
    }
    let p = deg * (180 - deg);
    let num = (4 * p) as i64 * ONE as i64;
    let den = (40500 - p) as i64;
    (num / den) as i32
}

const fn cos_k(i: usize) -> i32 {
    let q = (i % 256) as i32;
    let s = (q + 64) & 255;
    let half = if s <= 128 { s } else { 256 - s };
    let deg = (half * 180 + 64) / 128;
    let sinv = sin_bhaskara_q16(deg);
    if s <= 128 {
        sinv
    } else {
        sinv.wrapping_neg()
    }
}

fn cos_lat(z: i32) -> i32 {
    let t = rem_floor(z, 16384);
    let idx = (t >> 6) as usize;
    let frac = (((t & 63) as i64) << 16) / 64;
    let a = COS_TABLE[idx & 255];
    let b = COS_TABLE[(idx + 1) & 255];
    lerp_q16(a, b, frac as i32)
}

fn q16_round(v: i32) -> i32 {
    ((v as i64 + 32768) >> 16) as i32
}

fn norm01(v: i32, n: i32) -> i32 {
    if n <= 0 {
        return 0;
    }
    clamp_q16((((v as i64) << 16) / n as i64) as i32, 0, ONE)
}

fn elev_spline(cont: i32) -> i32 {
    const K: [(i32, i32); 7] = [
        (0, -40 * ONE),
        (C_DEEP, -40 * ONE),
        (C_SHELF, -8 * ONE),
        (C_COAST, 2 * ONE),
        (C_LOW, 8 * ONE),
        (C_UP, 30 * ONE),
        (ONE, 45 * ONE),
    ];
    if cont <= K[0].0 {
        return K[0].1;
    }
    let mut i = 1usize;
    while i < 7 {
        if cont <= K[i].0 {
            let (c0, e0) = K[i - 1];
            let (c1, e1) = K[i];
            let span = c1.wrapping_sub(c0);
            if span <= 0 {
                return e1;
            }
            let t = (((cont.wrapping_sub(c0) as i64) << 16) / span as i64) as i32;
            return lerp_q16(e0, e1, t);
        }
        i += 1;
    }
    K[6].1
}

fn scale_300(dist_m: i32) -> i32 {
    let d = if dist_m < 0 { 0 } else { dist_m };
    let den = 300i32.wrapping_add(d);
    if den <= 0 {
        ONE
    } else {
        (((300i64) << 16) / den as i64) as i32
    }
}

fn relief_q16(r: f32) -> i32 {
    let bits = (r * 100.0).round() as i32;
    match bits {
        50 => HALF,
        100 => ONE,
        150 => ONE + HALF,
        200 => ONE * 2,
        400 => ONE * 4,
        _ => ONE,
    }
}

fn is_volcanic(id: u32) -> bool {
    (id & 0xFF) % 6 == 0
}

fn is_basin(w: i32) -> bool {
    (w & 0x1FF) >= 0x80
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Biome {
    Coast,
    Tundra,
    Taiga,
    Grassland,
    Savanna,
    Desert,
    Badlands,
    Rainforest,
    Alpine,
    Volcanic,
    Wetland,
}

#[derive(Clone, Copy)]
struct Col {
    height: i32,
    water: i32,
    dress: BlockId,
    crust: BlockId,
}

struct V2Score {
    seed: u32,
    relief: i32,
}

thread_local! {
    static LAST: Cell<Option<(u32, u32, i32, i32, [i32; 8])>> = const { Cell::new(None) };
}

impl V2Score {
    fn all(&self, st: &Stencil, x: i32, z: i32, phase: u32) -> [i32; 8] {
        if let Some((s, p, lx, lz, out)) = LAST.get() {
            if s == self.seed && p == phase && lx == x && lz == z {
                return out;
            }
        }
        let out = if phase == 0 {
            self.plates(x, z)
        } else if phase == 1 {
            self.climate(st, z)
        } else {
            let mut o = [0i32; 8];
            let mut c = 0u32;
            while c < CHANNELS {
                o[c as usize] = st.prev(c, 0, 0);
                c = c.wrapping_add(1);
            }
            o
        };
        LAST.set(Some((self.seed, phase, x, z, out)));
        out
    }

    fn plates(&self, x: i32, z: i32) -> [i32; 8] {
        let (f1, f2, gid) = cellular_q16(self.seed, x, z, CELL_PLATE, SALT_PLATE);
        let bound = f2.wrapping_sub(f1);
        let bound_m = ((bound as i64 * CELL_PLATE as i64) >> 16) as i32;
        let region = (gid & 0xFF) as i32;
        let hard = (uniform_q16(gid) >> 8) as i32;
        let geol = region | (hard << 8);

        let (dx, dz) = warp_q16(self.seed, x, z, CELL_CONT, WARP_AMP, SALT_WARP);
        let wx = x.wrapping_add(dx >> 16);
        let wz = z.wrapping_add(dz >> 16);
        let cont = clamp_q16(
            norm01(
                fbm_q16(self.seed, wx, wz, CELL_CONT, 4, HALF, SALT_CONT),
                FBM4,
            )
            .wrapping_add(ONE / 10), // lift so continents occupy ~half the map
            0,
            ONE,
        );

        let mut elev = elev_spline(cont);
        if cont < C_COAST {
            elev = lerp_q16(elev, -60 * ONE, scale_300(bound_m));
        }
        let land = smoothstep_q16({
            let span = C_COAST.wrapping_sub(C_SHELF);
            if span <= 0 {
                if cont >= C_COAST { ONE } else { 0 }
            } else {
                (((cont.wrapping_sub(C_SHELF) as i64) << 16) / span as i64) as i32
            }
        });
        let ridge = ridged_q16(value_noise_q16(self.seed, x, z, CELL_BELT, SALT_BELT));
        let belt = mul_q16(mul_q16(ridge, scale_300(bound_m)), BELT_AMP);
        elev = elev.wrapping_add(mul_q16(belt, land));

        if is_volcanic(gid) {
            let (vf1, _, _) = cellular_q16(self.seed, x, z, CELL_VOLC, SALT_VOLC);
            let dist_m = ((vf1 as i64 * CELL_VOLC as i64) >> 16) as i32;
            let cone = (90 * ONE).wrapping_sub(mul_q16(dist_m.wrapping_mul(ONE), POINT_35));
            if cone > 0 {
                elev = elev.wrapping_add(cone);
            }
            if dist_m < 12 {
                elev = elev.wrapping_sub(20 * ONE);
            }
        }
        elev = mul_q16(elev, self.relief);

        let humid = norm01(
            fbm_q16(self.seed, x, z, CELL_CONT, 4, HALF, SALT_HUMID),
            FBM4,
        );
        let mut out = [0i32; 8];
        out[CH_ELEV as usize] = elev;
        // Climate reads this as continentalness, then overwrites TEMP.
        out[CH_TEMP as usize] = cont;
        out[CH_HUMID as usize] = humid;
        out[CH_GEOL as usize] = geol;
        out
    }

    fn climate(&self, st: &Stencil, z: i32) -> [i32; 8] {
        let elev = st.prev(CH_ELEV, 0, 0);
        let cont = st.prev(CH_TEMP, 0, 0);
        let h_base = st.prev(CH_HUMID, 0, 0);
        let geol = st.prev(CH_GEOL, 0, 0);
        let h_up = st.prev(CH_HUMID, -2, 0);
        let e_up = st.prev(CH_ELEV, -2, 0);

        let lat = mul_q16(cos_lat(z), HALF).wrapping_add(HALF);
        let lapse = mul_q16(elev, LAPSE_Q16);
        let pull = mul_q16(ONE / 4, ONE.wrapping_sub(cont));
        let temp = clamp_q16(lerp_q16(lat.wrapping_sub(lapse), HALF, pull), 0, ONE);

        let mut h = mul_q16(h_base, HALF).wrapping_add(mul_q16(h_up, HALF));
        let de = e_up.wrapping_sub(elev);
        let rs = if de > 0 {
            let v = de / 40;
            if v > ONE { ONE } else { v }
        } else {
            0
        };
        h = h.wrapping_sub(rs);
        h = h.wrapping_add(mul_q16(ONE / 5, ONE.wrapping_sub(cont)));
        h = clamp_q16(h, 0, ONE);

        let mut basin = true;
        let mut dz = -2i32;
        'n: while dz <= 2 {
            let mut dx = -2i32;
            while dx <= 2 {
                if dx != 0 || dz != 0 {
                    if st.prev(CH_ELEV, dx, dz) <= elev {
                        basin = false;
                        break 'n;
                    }
                }
                dx = dx.wrapping_add(1);
            }
            dz = dz.wrapping_add(1);
        }

        let mut out = [0i32; 8];
        out[CH_ELEV as usize] = elev;
        out[CH_TEMP as usize] = temp;
        out[CH_HUMID as usize] = h;
        out[CH_GEOL as usize] = geol;
        out[CH_WATER as usize] = if basin { WATER_BASIN } else { 0 };
        out
    }
}

impl IntScore for V2Score {
    fn predict(
        &self,
        st: &Stencil,
        ch: u32,
        x: i32,
        z: i32,
        phase: u32,
        _phases: u32,
        current: i32,
    ) -> i32 {
        if ch >= CHANNELS {
            return current;
        }
        self.all(st, x, z, phase)[ch as usize]
    }
}

/// TerrainGenerator driven by the integer v2 field.
pub struct DiffusionV2 {
    seed: i64,
    sea: i32,
    relief: f32,
    field: IntField<V2Score>,
    mat: placement::Resolved,
}

impl DiffusionV2 {
    pub fn new(registry: &mut BlockRegistry, cfg: DiffusionCfg, seed: i64) -> Self {
        let cfg = cfg.clamp();
        let mat = placement::builtin().compile(registry).expect("v0 hosts the placement table");
        let iseed = seed as u32;
        let spec = IntSpec {
            seed: iseed,
            tile: cfg.tile,
            stride: cfg.stride,
            phases: PHASES,
            channels: CHANNELS,
        };
        Self {
            seed,
            sea: SEA,
            relief: cfg.relief,
            field: IntField::new(
                spec,
                V2Score {
                    seed: iseed,
                    relief: relief_q16(cfg.relief),
                },
            ),
            mat,
        }
    }

    fn sample(&self, wx: i32, wz: i32) -> [i32; 8] {
        let mut ch = [0i32; 8];
        self.field.sample_all(wx, wz, &mut ch);
        ch
    }

    fn col_from_ch(&self, ch: &[i32], wx: i32, wz: i32) -> Col {
        let elev = ch[CH_ELEV as usize];
        let height = self.sea.wrapping_add(q16_round(elev));
        let biome = classify(
            ch[CH_TEMP as usize],
            ch[CH_HUMID as usize],
            elev,
            ch[CH_GEOL as usize],
            is_basin(ch[CH_WATER as usize]),
            wx,
            height,
            wz,
            self.seed,
        );
        let (dress, crust, kind) = self.dress_of(biome, height);
        let dress = self.scatter(kind, dress, wx, height, wz);
        Col {
            height,
            water: self.sea,
            dress,
            crust,
        }
    }

    fn column(&self, wx: i32, wz: i32) -> Col {
        self.col_from_ch(&self.sample(wx, wz), wx, wz)
    }

    fn dress_of(&self, biome: Biome, height: i32) -> (BlockId, BlockId, Option<SurfaceKind>) {
        use Biome::*;
        match biome {
            Coast => (
                self.mat.dress[SurfaceKind::Shore as usize],
                self.mat.crust[SurfaceKind::Shore as usize],
                Some(SurfaceKind::Shore),
            ),
            Tundra => (
                self.mat.dress[SurfaceKind::Snowy as usize],
                self.mat.crust[SurfaceKind::Snowy as usize],
                Some(SurfaceKind::Snowy),
            ),
            Taiga | Grassland | Rainforest => (
                self.mat.dress[SurfaceKind::Grassy as usize],
                self.mat.crust[SurfaceKind::Grassy as usize],
                Some(SurfaceKind::Grassy),
            ),
            Savanna => (
                self.mat.dress[SurfaceKind::BeachEdge as usize],
                self.mat.crust[SurfaceKind::Grassy as usize],
                Some(SurfaceKind::BeachEdge),
            ),
            Desert => (
                self.mat.dress[SurfaceKind::Desert as usize],
                self.mat.crust[SurfaceKind::Desert as usize],
                Some(SurfaceKind::Desert),
            ),
            Badlands => {
                let sand = rem_floor(height, 12) < 6;
                if sand {
                    (
                        self.mat.dress[SurfaceKind::Desert as usize],
                        self.mat.crust[SurfaceKind::Desert as usize],
                        Some(SurfaceKind::Desert),
                    )
                } else {
                    (
                        self.mat.crust[SurfaceKind::Desert as usize],
                        self.mat.crust[SurfaceKind::Desert as usize],
                        None,
                    )
                }
            }
            Alpine | Volcanic => (self.mat.stone, self.mat.stone, None),
            Wetland => (
                self.mat.crust[SurfaceKind::Grassy as usize],
                self.mat.crust[SurfaceKind::Grassy as usize],
                None,
            ),
        }
    }

    fn scatter(
        &self,
        kind: Option<SurfaceKind>,
        dress: BlockId,
        wx: i32,
        height: i32,
        wz: i32,
    ) -> BlockId {
        let Some(kind) = kind else {
            return dress;
        };
        let slices = &self.mat.surface_scatter[kind as usize];
        if slices.is_empty() {
            return dress;
        }
        let roll = cell_hash(self.seed ^ SURFACE_SCATTER_SALT, wx, height, wz);
        let mut cut = 0u32;
        for slice in slices {
            cut = cut.wrapping_add(slice.width);
            if roll < cut {
                return slice.id;
            }
        }
        dress
    }

    fn ore_at(&self, wx: i32, wy: i32, wz: i32, depth: i32) -> Option<BlockId> {
        let hit = |slices: &[placement::Slice], roll: u32| -> Option<usize> {
            let mut cut = 0u32;
            for (i, slice) in slices.iter().enumerate() {
                if depth < slice.min_depth {
                    break;
                }
                cut = cut.wrapping_add(slice.width);
                if roll < cut {
                    return Some(i);
                }
            }
            None
        };
        let a = hit(&self.mat.seams, cell_hash(self.seed, wx, wy, wz));
        let b = if depth >= self.mat.seams.first().map_or(i32::MAX, |s| s.min_depth) {
            hit(
                &self.mat.seams_b,
                cell_hash(self.seed ^ ORE_B_SALT, wx, wy, wz),
            )
        } else {
            None
        };
        match (a, b) {
            (Some(i), Some(j)) if i != j => {
                let (hi, lo) = if i > j { (i, j) } else { (j, i) };
                Some(self.mat.pairs[hi][lo])
            }
            (Some(i), _) | (None, Some(i)) => Some(self.mat.seams[i].id),
            (None, None) => None,
        }
    }

    fn cell(&self, c: &Col, wx: i32, wy: i32, wz: i32) -> BlockId {
        if wy < c.height {
            if wy >= c.height - 1 {
                c.dress
            } else if wy >= c.height - 3 {
                c.crust
            } else {
                let depth = c.height - wy;
                if depth <= self.mat.max_scattered_depth {
                    if let Some(ore) = self.ore_at(wx, wy, wz, depth) {
                        return ore;
                    }
                }
                self.mat.stone
            }
        } else if wy < c.water {
            self.mat.water
        } else {
            AIR
        }
    }

    fn lod_height(&self, wx: i32, wz: i32) -> i32 {
        let n = hash_noise(self.seed as u64, wx, wz, 64);
        let elev = (n * 2.0 - 1.0) * 36.0 * self.relief.clamp(0.25, 4.0).sqrt();
        (self.sea as f32 + elev).round() as i32
    }
}

fn classify(
    temp: i32,
    humid: i32,
    elev: i32,
    geol: i32,
    basin: bool,
    wx: i32,
    height: i32,
    wz: i32,
    seed: i64,
) -> Biome {
    let roll = cell_hash(seed ^ SURFACE_SCATTER_SALT, wx, height, wz);
    let dt = ((roll & 0xFFFF) as i32 - 32768) >> 4;
    let dh = (((roll >> 16) & 0xFFFF) as i32 - 32768) >> 4;
    let temp = clamp_q16(temp.wrapping_add(dt), 0, ONE);
    let humid = clamp_q16(humid.wrapping_add(dh), 0, ONE);

    if is_volcanic(geol as u32) && elev > 8 * ONE {
        return Biome::Volcanic;
    }
    if elev >= 40 * ONE {
        return Biome::Alpine;
    }
    if elev >= 0 && elev < 2 * ONE {
        return Biome::Coast;
    }
    if basin && humid > HALF && elev >= 0 && elev < 10 * ONE {
        return Biome::Wetland;
    }
    const T_TUNDRA: i32 = ONE / 4;
    const T_TAIGA: i32 = (ONE * 2) / 5;
    const T_HOT: i32 = (ONE * 11) / 20;
    const H_DRY: i32 = (ONE * 7) / 25;
    const H_ARID: i32 = (ONE * 2) / 5;
    const H_WET: i32 = (ONE * 13) / 20;
    if temp < T_TUNDRA {
        Biome::Tundra
    } else if temp < T_TAIGA && humid > H_ARID {
        Biome::Taiga
    } else if temp > T_HOT && humid > H_WET {
        Biome::Rainforest
    } else if temp > T_HOT && humid < H_DRY {
        Biome::Desert
    } else if temp > T_HOT && humid < H_ARID && elev > 15 * ONE {
        Biome::Badlands
    } else if temp > T_HOT && humid < HALF + ONE / 10 {
        Biome::Savanna
    } else {
        Biome::Grassland
    }
}

fn hash_noise(seed: u64, x: i32, z: i32, cell: u32) -> f32 {
    let cell = cell.max(1) as i32;
    let gx = x.div_euclid(cell);
    let gz = z.div_euclid(cell);
    let fx = x.rem_euclid(cell) as f32 / cell as f32;
    let fz = z.rem_euclid(cell) as f32 / cell as f32;
    let sx = fx * fx * (3.0 - 2.0 * fx);
    let sz = fz * fz * (3.0 - 2.0 * fz);
    let n = |ix: i32, iz: i32| {
        let mut h = seed
            ^ (ix as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (iz as u32 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        h ^= h >> 30;
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 27;
        (h >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    };
    let a = n(gx, gz);
    let b = n(gx + 1, gz);
    let c = n(gx, gz + 1);
    let d = n(gx + 1, gz + 1);
    let u = a + (b - a) * sx;
    u + (c + (d - c) * sx - u) * sz
}

impl TerrainGenerator for DiffusionV2 {
    fn seed(&self) -> i64 {
        self.seed
    }
    fn sea_level(&self) -> i32 {
        self.sea
    }
    fn kind(&self) -> &'static str {
        "diffusion"
    }

    fn height(&self, wx: i32, wz: i32) -> i32 {
        self.column(wx, wz).height
    }

    fn heights_16(&self, cx: i32, cz: i32) -> ColumnHeights {
        let x0 = cx * CHUNK_SIZE as i32;
        let z0 = cz * CHUNK_SIZE as i32;
        let mut buf = [0i32; CHUNK_SIZE * CHUNK_SIZE * 8];
        self.field
            .fill_all(x0, z0, CHUNK_SIZE as u32, CHUNK_SIZE as u32, &mut buf);
        let mut heights = [0i32; CHUNK_SIZE * CHUNK_SIZE];
        for i in 0..CHUNK_SIZE * CHUNK_SIZE {
            let elev = buf[i * 8 + CH_ELEV as usize];
            heights[i] = self.sea.wrapping_add(q16_round(elev));
        }
        heights
    }

    fn surface_at(&self, wx: i32, wz: i32) -> BlockId {
        self.column(wx, wz).dress
    }

    fn deep(&self) -> BlockId {
        self.mat.stone
    }

    fn block_at(&self, wx: i32, wy: i32, wz: i32, _height: i32) -> BlockId {
        let c = self.column(wx, wz);
        self.cell(&c, wx, wy, wz)
    }

    fn lod_block_at(&self, wx: i32, wy: i32, wz: i32) -> BlockId {
        let h = self.lod_height(wx, wz);
        if wy < h {
            self.mat.stone
        } else if wy < self.sea {
            self.mat.water
        } else {
            AIR
        }
    }

    fn lod_column(&self, wx: i32, wz: i32, ys: &[i32], out: &mut [BlockId]) {
        let h = self.lod_height(wx, wz);
        for (o, &wy) in out.iter_mut().zip(ys) {
            *o = if wy < h {
                self.mat.stone
            } else if wy < self.sea {
                self.mat.water
            } else {
                AIR
            };
        }
    }

    fn generate(&self, cx: i32, cy: i32, cz: i32) -> ChunkData {
        self.generate_column(cx, cz, cy..=cy)
            .0
            .into_iter()
            .next()
            .map(|(_, data)| data)
            .unwrap_or_else(|| ChunkData::from_cells(Box::new([AIR; CHUNK_VOLUME])))
    }

    #[allow(clippy::needless_range_loop)]
    fn generate_column(
        &self,
        cx: i32,
        cz: i32,
        cy: std::ops::RangeInclusive<i32>,
    ) -> (Vec<(i32, ChunkData)>, ColumnHeights) {
        let x0 = cx * CHUNK_SIZE as i32;
        let z0 = cz * CHUNK_SIZE as i32;
        let mut cols = [[Col {
            height: 0,
            water: SEA,
            dress: AIR,
            crust: AIR,
        }; CHUNK_SIZE]; CHUNK_SIZE];
        let mut buf = [0i32; CHUNK_SIZE * CHUNK_SIZE * 8];
        self.field
            .fill_all(x0, z0, CHUNK_SIZE as u32, CHUNK_SIZE as u32, &mut buf);
        let mut heights = [0i32; CHUNK_SIZE * CHUNK_SIZE];
        let mut max_top = i32::MIN;
        let mut min_h = i32::MAX;
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let i = (lz * CHUNK_SIZE + lx) * 8;
                let wx = x0 + lx as i32;
                let wz = z0 + lz as i32;
                let c = self.col_from_ch(&buf[i..i + 8], wx, wz);
                heights[lx + lz * CHUNK_SIZE] = c.height;
                max_top = max_top.max(c.height.max(c.water));
                min_h = min_h.min(c.height);
                cols[lz][lx] = c;
            }
        }
        let deep_cut = min_h - self.mat.max_scattered_depth.max(48);
        let chunks = cy
            .map(|cyy| {
                let y0 = cyy * CHUNK_SIZE as i32;
                let y1 = y0 + CHUNK_SIZE as i32;
                if y0 >= max_top {
                    return (cyy, ChunkData::Uniform(AIR));
                }
                if y1 <= deep_cut {
                    return (cyy, ChunkData::Uniform(self.mat.stone));
                }
                let mut cells = Box::new([AIR; CHUNK_VOLUME]);
                for lz in 0..CHUNK_SIZE {
                    for lx in 0..CHUNK_SIZE {
                        let c = &cols[lz][lx];
                        let wx = x0 + lx as i32;
                        let wz = z0 + lz as i32;
                        for ly in 0..CHUNK_SIZE {
                            let wy = y0 + ly as i32;
                            cells[Chunk::index(lx, ly, lz)] = self.cell(c, wx, wy, wz);
                        }
                    }
                }
                (cyy, ChunkData::from_cells(cells))
            })
            .collect();
        (chunks, heights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::BlockRegistry;

    fn cfg_v2() -> DiffusionCfg {
        let mut c = DiffusionCfg::default();
        c.version = 2;
        c
    }

    fn v2(seed: i64) -> DiffusionV2 {
        DiffusionV2::new(&mut BlockRegistry::with_builtins(), cfg_v2(), seed)
    }

    #[test]
    fn cos_table_quadrants() {
        assert_eq!(COS_TABLE[0], ONE);
        assert!(COS_TABLE[64].abs() < ONE / 20, "cos(π/2) got {}", COS_TABLE[64]);
        assert_eq!(COS_TABLE[128], -ONE);
        assert!(COS_TABLE[192].abs() < ONE / 20, "cos(3π/2) got {}", COS_TABLE[192]);
        assert_eq!(cos_lat(0), COS_TABLE[0]);
        let c = cos_lat(4096);
        assert!(c.abs() < ONE / 10, "cos(π/2) lat got {c}");
    }

    #[test]
    fn diffusion_v2_is_seed_and_order_stable() {
        let a = v2(9);
        let b = v2(9);
        let c = v2(10);
        for z in [0, 3, 16, -8] {
            for x in [0, 7, -3] {
                assert_eq!(a.height(x, z), b.height(x, z));
                assert_eq!(
                    a.block_at(x, a.height(x, z) - 1, z, 0),
                    b.block_at(x, b.height(x, z) - 1, z, 0)
                );
            }
        }
        assert_eq!(a.generate(0, 0, 0), b.generate(0, 0, 0));
        let mut differ = false;
        for z in -8..8 {
            for x in -8..8 {
                if a.height(x, z) != c.height(x, z) {
                    differ = true;
                }
            }
        }
        assert!(differ, "different seeds must move the heightfield");
        let mut fwd = Vec::new();
        for z in 0..24 {
            for x in 0..24 {
                fwd.push(a.height(x, z));
            }
        }
        let mut rev = Vec::new();
        for z in (0..24).rev() {
            for x in (0..24).rev() {
                rev.push(b.height(x, z));
            }
        }
        rev.reverse();
        assert_eq!(fwd, rev);
    }

    #[test]
    fn heights_16_matches_height() {
        let g = v2(11);
        for &(cx, cz) in &[(0, 0), (2, -3), (-1, 7)] {
            let batch = g.heights_16(cx, cz);
            let x0 = cx * CHUNK_SIZE as i32;
            let z0 = cz * CHUNK_SIZE as i32;
            for lz in 0..CHUNK_SIZE {
                for lx in 0..CHUNK_SIZE {
                    assert_eq!(
                        batch[lx + lz * CHUNK_SIZE],
                        g.height(x0 + lx as i32, z0 + lz as i32),
                        "cx={cx} cz={cz} lx={lx} lz={lz}"
                    );
                }
            }
        }
    }

    #[test]
    fn generate_column_heights_match_height() {
        let g = v2(7);
        super::super::generation::assert_generate_column_heights_match_height(
            &g,
            &[(0, 0), (2, -3), (-1, 7), (4, 4)],
        );
    }

    #[test]
    fn generate_agrees_with_per_cell_block_at() {
        let g = v2(5);
        for &(cx, cy, cz) in &[(0, 1, 0), (2, 0, -1), (0, 4, 0)] {
            let got = g.generate(cx, cy, cz);
            let mut cells = Box::new([AIR; CHUNK_VOLUME]);
            let x0 = cx * CHUNK_SIZE as i32;
            let y0 = cy * CHUNK_SIZE as i32;
            let z0 = cz * CHUNK_SIZE as i32;
            for lz in 0..CHUNK_SIZE {
                for lx in 0..CHUNK_SIZE {
                    let wx = x0 + lx as i32;
                    let wz = z0 + lz as i32;
                    let h = g.height(wx, wz);
                    for ly in 0..CHUNK_SIZE {
                        cells[Chunk::index(lx, ly, lz)] = g.block_at(wx, y0 + ly as i32, wz, h);
                    }
                }
            }
            assert_eq!(got, ChunkData::from_cells(cells), "chunk {cx},{cy},{cz}");
        }
    }

    #[test]
    fn far_coords_are_finite_and_stable() {
        let a = v2(3);
        let b = v2(3);
        for &(x, z) in &[
            (1_000_000_000, 0),
            (-1_000_000_000, 1_000_000_000),
            (1_000_000_000, -1_000_000_000),
            (i32::MAX - 40, -1_000_000_000),
        ] {
            let ha = a.height(x, z);
            let hb = b.height(x, z);
            assert_eq!(ha, hb, "at {x},{z}");
            let _ = a.block_at(x, ha - 1, z, ha);
            let _ = a.sample(x, z);
        }
    }

    #[test]
    fn lod_uses_the_same_sea() {
        let g = v2(13);
        assert_eq!(g.sea_level(), SEA);
        let ys = [SEA - 1, SEA, SEA + 1];
        let mut out = [AIR; 3];
        g.lod_column(0, 0, &ys, &mut out);
        assert_ne!(out[0], AIR, "LOD fills below sea");
        assert!(out[1] == AIR || out[1] == g.deep());
    }

    #[test]
    fn version_2_differs_from_v1_and_world_selects_it() {
        let v1 = super::super::diffusion::DiffusionTerrain::new(
            &mut BlockRegistry::with_builtins(),
            DiffusionCfg::default(),
            42,
        );
        let g = v2(42);
        let mut differ = false;
        for z in 0..32 {
            for x in 0..32 {
                if v1.height(x, z) != g.height(x, z) {
                    differ = true;
                }
            }
        }
        assert!(differ);
        assert_eq!(g.kind(), "diffusion");
        assert_eq!(DiffusionCfg::from_text("version=2").version, 2);
        assert_eq!(DiffusionCfg::from_text("").version, 1);
        let cfg = cfg_v2().clamp();
        assert_eq!(DiffusionCfg::from_text(&cfg.to_text()), cfg);
    }

    fn climate_from_plates(s: &V2Score, x: i32, z: i32) -> (i32, i32, i32) {
        let p = s.plates(x, z);
        let u = s.plates(x.wrapping_sub(2), z);
        let elev = p[CH_ELEV as usize];
        let cont = p[CH_TEMP as usize];
        let h_base = p[CH_HUMID as usize];
        let h_up = u[CH_HUMID as usize];
        let e_up = u[CH_ELEV as usize];
        let lat = mul_q16(cos_lat(z), HALF).wrapping_add(HALF);
        let lapse = mul_q16(elev, LAPSE_Q16);
        let pull = mul_q16(ONE / 4, ONE.wrapping_sub(cont));
        let temp = clamp_q16(lerp_q16(lat.wrapping_sub(lapse), HALF, pull), 0, ONE);
        let mut h = mul_q16(h_base, HALF).wrapping_add(mul_q16(h_up, HALF));
        let de = e_up.wrapping_sub(elev);
        let rs = if de > 0 {
            let v = de / 40;
            if v > ONE { ONE } else { v }
        } else {
            0
        };
        h = h.wrapping_sub(rs);
        h = h.wrapping_add(mul_q16(ONE / 5, ONE.wrapping_sub(cont)));
        h = clamp_q16(h, 0, ONE);
        (q16_round(elev), temp, h)
    }

    fn sample_window(s: &V2Score, step: i32, n: i32) -> Vec<(i32, i32, i32, i32, i32)> {
        let mut out = Vec::with_capacity((n * n) as usize);
        let origin = -n.wrapping_mul(step) / 2;
        let mut iz = 0i32;
        while iz < n {
            let mut ix = 0i32;
            while ix < n {
                let x = origin.wrapping_add(ix.wrapping_mul(step));
                let z = origin.wrapping_add(iz.wrapping_mul(step));
                let (elev, temp, humid) = climate_from_plates(s, x, z);
                out.push((x, z, elev, temp, humid));
                ix = ix.wrapping_add(1);
            }
            iz = iz.wrapping_add(1);
        }
        out
    }

    #[test]
    fn shape_ocean_peak_lapse_and_rain_shadow() {
        let s = V2Score {
            seed: 42,
            relief: ONE,
        };
        let g = v2(42);
        for &(x, z) in &[(0, 0), (400, -200), (1600, 800), (-2400, 1200)] {
            let (elev, _, _) = climate_from_plates(&s, x, z);
            let h = g.height(x, z);
            assert!(
                (h - (SEA + elev)).abs() <= 2,
                "field height {h} vs plates {} at {x},{z}",
                SEA + elev
            );
        }
        let step = 128i32;
        let n = 48i32;
        let pts = sample_window(&s, step, n);
        let mut ocean = 0usize;
        let mut peak = false;
        let mut n_corr = 0i64;
        let mut s_e = 0i64;
        let mut s_t = 0i64;
        let mut s_ee = 0i64;
        let mut s_tt = 0i64;
        let mut s_et = 0i64;
        for &(_, _, elev, temp, _) in &pts {
            if elev < 0 {
                ocean += 1;
            }
            if elev > 80 {
                peak = true;
            }
            let e = elev as i64;
            let t = temp as i64;
            n_corr += 1;
            s_e += e;
            s_t += t;
            s_ee += e * e;
            s_tt += t * t;
            s_et += e * t;
        }
        let frac = ocean as f64 / pts.len() as f64;
        assert!(
            (0.20..0.60).contains(&frac),
            "ocean fraction {frac:.3} outside 20-60% (n={})",
            pts.len()
        );
        assert!(peak, "no cell above +80 m in the 2 km window at seed 42");
        let num = n_corr * s_et - s_e * s_t;
        let den_e = n_corr * s_ee - s_e * s_e;
        let den_t = n_corr * s_tt - s_t * s_t;
        assert!(den_e > 0 && den_t > 0, "degenerate correlation");
        assert!(
            num < 0,
            "temperature must decrease with elevation (cov={num})"
        );

        let cols = n as usize;
        let mut lee_drier = 0usize;
        let mut ridges = 0usize;
        for iz in 0..n as usize {
            for ix in 1..cols - 1 {
                let i = iz * cols + ix;
                let elev = pts[i].2;
                if elev <= 60 {
                    continue;
                }
                let west = pts[i - 1].2;
                let east = pts[i + 1].2;
                if elev <= west || elev <= east {
                    continue;
                }
                ridges += 1;
                let h_w = pts[i - 1].4;
                let h_e = pts[i + 1].4;
                if h_e < h_w {
                    lee_drier += 1;
                }
            }
        }
        assert!(ridges > 0, "no >60 m ridge cells sampled");
        assert!(
            lee_drier * 2 >= ridges,
            "rain shadow: lee drier on {lee_drier}/{ridges} ridge cells"
        );
    }

    fn biome_rgb(b: Biome) -> [u8; 3] {
        match b {
            Biome::Coast => [210, 190, 120],
            Biome::Tundra => [230, 240, 245],
            Biome::Taiga => [40, 100, 55],
            Biome::Grassland => [90, 160, 60],
            Biome::Savanna => [190, 170, 70],
            Biome::Desert => [220, 200, 90],
            Biome::Badlands => [180, 90, 40],
            Biome::Rainforest => [20, 90, 35],
            Biome::Alpine => [160, 160, 165],
            Biome::Volcanic => [70, 40, 35],
            Biome::Wetland => [50, 120, 110],
        }
    }

    fn height_rgb(elev_m: i32) -> [u8; 3] {
        if elev_m < 0 {
            let t = (-elev_m).clamp(0, 80) as u32;
            let b = 220u32.saturating_sub(t * 2);
            [10, 30 + (t / 4) as u8, b.min(255) as u8]
        } else if elev_m < 8 {
            [70, 150, 55]
        } else if elev_m < 30 {
            let t = (elev_m - 8) as u32;
            [
                (70 + t * 4).min(160) as u8,
                (150 - t * 2).min(150) as u8,
                50,
            ]
        } else if elev_m < 80 {
            let t = (elev_m - 30) as u32;
            [
                (160 + t).min(220) as u8,
                (110 + t / 2).min(180) as u8,
                (50 + t).min(160) as u8,
            ]
        } else {
            [245, 245, 250]
        }
    }

    /// 1024×1024 P6, four quadrants of a 2048 m window at 4 m/pixel, seed 42.
    /// No PNG crate is reachable without a new dependency.
    #[test]
    #[ignore]
    fn worldgen_v2_map_ppm() {
        let seed = 42i64;
        let g = v2(seed);
        let side = 512u32;
        let pix = 4i32;
        let mut img = vec![0u8; 1024 * 1024 * 3];
        let put = |img: &mut [u8], qx: u32, qz: u32, lx: u32, lz: u32, rgb: [u8; 3]| {
            let x = qx * side + lx;
            let z = qz * side + lz;
            let i = ((z * 1024 + x) * 3) as usize;
            img[i] = rgb[0];
            img[i + 1] = rgb[1];
            img[i + 2] = rgb[2];
        };
        let origin = -((side as i32) * pix) / 2;
        let mut lz = 0u32;
        while lz < side {
            let mut lx = 0u32;
            while lx < side {
                let x = origin.wrapping_add((lx as i32).wrapping_mul(pix));
                let z = origin.wrapping_add((lz as i32).wrapping_mul(pix));
                let ch = g.sample(x, z);
                let elev_m = q16_round(ch[CH_ELEV as usize]);
                let basin = is_basin(ch[CH_WATER as usize]);
                let biome = classify(
                    ch[CH_TEMP as usize],
                    ch[CH_HUMID as usize],
                    ch[CH_ELEV as usize],
                    ch[CH_GEOL as usize],
                    basin,
                    x,
                    SEA.wrapping_add(elev_m),
                    z,
                    seed,
                );
                put(&mut img, 0, 0, lx, lz, height_rgb(elev_m));
                let brgb = if elev_m < 0 {
                    [20, 60, 180]
                } else {
                    biome_rgb(biome)
                };
                put(&mut img, 1, 0, lx, lz, brgb);
                let water = if elev_m < 0 {
                    [20, 60, 180]
                } else if basin {
                    [40, 180, 200]
                } else {
                    [30, 30, 30]
                };
                put(&mut img, 0, 1, lx, lz, water);
                let (_, _, gid) = cellular_q16(seed as u32, x, z, CELL_PLATE, SALT_PLATE);
                let h = gid ^ gid.wrapping_mul(0x9E37_79B9);
                let geol = [(h >> 16) as u8, (h >> 8) as u8, h as u8];
                put(&mut img, 1, 1, lx, lz, geol);
                lx += 1;
            }
            lz += 1;
        }
        let path = format!("target/worldgen-v2-map-{seed}.ppm");
        let mut buf = format!("P6\n1024 1024\n255\n").into_bytes();
        buf.extend_from_slice(&img);
        std::fs::create_dir_all("target").expect("target dir");
        std::fs::write(&path, buf).expect("write ppm");
        eprintln!("wrote {path}");
    }

    #[test]
    #[ignore]
    fn worldgen_column_cost_v2() {
        use std::time::Instant;
        let n = 24i32;
        let classic = super::super::diffusion::classic(&mut BlockRegistry::with_builtins(), 42);
        let v1 = super::super::diffusion::diffusion(
            &mut BlockRegistry::with_builtins(),
            42,
            DiffusionCfg::default(),
        );
        let v2g = super::super::diffusion::diffusion_v2(
            &mut BlockRegistry::with_builtins(),
            42,
            cfg_v2(),
        );
        let time = |g: &super::super::diffusion::Generator| {
            let t = Instant::now();
            for cz in 0..n {
                for cx in 0..n {
                    let _ = g.generate_column(cx, cz, 0..=3);
                }
            }
            t.elapsed().as_secs_f64() * 1000.0
        };
        let classic_ms = time(&classic);
        let v1_ms = time(&v1);
        let v2_ms = time(&v2g);
        let cols = (n * n) as f64;
        eprintln!(
            "worldgen_column_cost_v2 n={n} classic={classic_ms:.1}ms ({:.3} ms/col) v1={v1_ms:.1}ms ({:.3} ms/col) v2={v2_ms:.1}ms ({:.3} ms/col) v2/classic={:.2} v2/v1={:.2}",
            classic_ms / cols,
            v1_ms / cols,
            v2_ms / cols,
            v2_ms / classic_ms.max(0.001),
            v2_ms / v1_ms.max(0.001)
        );
    }
}
