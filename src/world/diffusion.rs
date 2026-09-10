//! InfiniteDiffusion terrain: overlapping-window fields interpreted by the
//! same placement table as classic noise. The field is the substrate; this
//! file is the voxel rules.

use std::sync::Arc;

use infinite_field::{InfiniteField, Score, Spec};

use super::chunk::{CHUNK_SIZE, ChunkData};
use super::generation::{cell_hash, ColumnHeights, TerrainGenerator};
use super::placement;
use crate::block::registry::{AIR, BlockId, BlockRegistry};

const SEA: i32 = 20;
const SNOW_ABOVE: i32 = 55;
const COLD: f32 = 0.30;
const HOT: f32 = 0.72;
const DRY: f32 = 0.32;

/// Tunables the InfiniteDiffusion mod exposes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiffusionCfg {
    pub tile: u32,
    pub stride: u32,
    pub phases: u32,
    /// Extra vertical relief scale (1 = default).
    pub relief: f32,
    /// 1 = overlapping f32 field ([`DiffusionTerrain`]); 2 = integer v2
    /// ([`super::diffusion_v2::DiffusionV2`]). Knob list is unchanged; v2 is
    /// selected only through this field (W6 switches the mod over).
    pub version: u8,
}

impl Default for DiffusionCfg {
    fn default() -> Self {
        Self {
            tile: 32,
            stride: 16,
            phases: 2,
            relief: 1.0,
            version: 1,
        }
    }
}

impl DiffusionCfg {
    /// Tile sizes the knob stepper cycles. [`clamp`](Self::clamp) snaps here.
    pub const TILES: [u32; 3] = [16, 32, 64];
    pub const MIN_STRIDE: u32 = 8;
    pub const STRIDE_STEP: u32 = 8;
    pub const PHASES_MIN: u32 = 2;
    pub const PHASES_MAX: u32 = 8;
    /// Relief values the knob stepper cycles. [`clamp`](Self::clamp) snaps here.
    pub const RELIEFS: [f32; 5] = [0.5, 1.0, 1.5, 2.0, 4.0];

    pub fn clamp(mut self) -> Self {
        self.tile = snap_u32(&Self::TILES, self.tile);
        self.stride = snap_stride(self.stride, self.tile);
        self.phases = self.phases.clamp(Self::PHASES_MIN, Self::PHASES_MAX);
        self.relief = snap_f32(&Self::RELIEFS, self.relief);
        self.version = self.version.clamp(1, 2);
        self
    }

    /// Wire form of the diffusion worldgen payload (`tile=…,stride=…,…`).
    /// `version` is omitted at 1 so existing save / mod-state bytes stay put.
    pub fn to_text(self) -> String {
        if self.version == 1 {
            format!(
                "tile={},stride={},phases={},relief={:.2}",
                self.tile, self.stride, self.phases, self.relief
            )
        } else {
            format!(
                "tile={},stride={},phases={},relief={:.2},version={}",
                self.tile, self.stride, self.phases, self.relief, self.version
            )
        }
    }

    /// Parse a full or partial knob string, starting from the defaults.
    pub fn from_text(data: &str) -> Self {
        Self::default().overlay(data)
    }

    /// Overlay keys from `data` onto `self`, then clamp.
    pub fn overlay(mut self, data: &str) -> Self {
        for part in data.split(',') {
            let Some((k, v)) = part.split_once('=') else {
                continue;
            };
            match k.trim() {
                "tile" => self.tile = v.parse().unwrap_or(self.tile),
                "stride" => self.stride = v.parse().unwrap_or(self.stride),
                "phases" => self.phases = v.parse().unwrap_or(self.phases),
                "relief" => self.relief = v.parse().unwrap_or(self.relief),
                "version" => self.version = v.parse().unwrap_or(self.version),
                _ => {}
            }
        }
        self.clamp()
    }
}

fn snap_u32(list: &[u32], v: u32) -> u32 {
    list.iter()
        .copied()
        .min_by_key(|&c| c.abs_diff(v))
        .unwrap_or(v)
}

fn snap_stride(stride: u32, tile: u32) -> u32 {
    let lo = DiffusionCfg::MIN_STRIDE;
    let hi = tile.max(lo);
    let v = stride.clamp(lo, hi);
    let step = DiffusionCfg::STRIDE_STEP;
    let snapped = ((v + step / 2) / step) * step;
    snapped.clamp(lo, hi)
}

fn snap_f32(list: &[f32], v: f32) -> f32 {
    list.iter()
        .copied()
        .min_by(|a, b| (a - v).abs().total_cmp(&(b - v).abs()))
        .unwrap_or(v)
}

struct TerrainScore {
    seed: u64,
    relief: f32,
}

impl Score for TerrainScore {
    fn predict(&self, channel: u32, x: i32, z: i32, phase: u32, phases: u32, current: f32) -> f32 {
        let remaining = phases.saturating_sub(1).saturating_sub(phase);
        let cell = match channel {
            0 => 16u32 << remaining.min(6),
            1 => 32u32 << remaining.min(5),
            2 => 24u32 << remaining.min(5),
            _ => 8u32 << remaining.min(4),
        };
        let n = hash_noise(self.seed ^ channel as u64, x, z, cell);
        let target = if channel == 0 {
            let ridged = 1.0 - (n * 2.0 - 1.0).abs();
            (n * 0.65 + ridged * 0.35) * self.relief.clamp(0.25, 4.0).sqrt()
        } else {
            n
        };
        let alpha = 0.30 + 0.14 * (phase as f32 / phases.max(1) as f32);
        current + (target.clamp(0.0, 1.0) - current) * alpha
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

/// TerrainGenerator driven by an InfiniteDiffusion field.
pub struct DiffusionTerrain {
    seed: i64,
    sea: i32,
    relief: f32,
    field: InfiniteField<TerrainScore>,
    mat: placement::Resolved,
}

impl DiffusionTerrain {
    pub fn new(registry: &mut BlockRegistry, cfg: DiffusionCfg, seed: i64) -> Self {
        let cfg = cfg.clamp();
        let mat = placement::builtin().compile(registry);
        let spec = Spec {
            seed: seed as u64,
            tile: cfg.tile,
            stride: cfg.stride,
            phases: cfg.phases,
            channels: 4,
        };
        Self {
            seed,
            sea: SEA,
            relief: cfg.relief,
            field: InfiniteField::new(
                spec,
                TerrainScore {
                    seed: seed as u64,
                    relief: cfg.relief,
                },
            ),
            mat,
        }
    }

    fn column(&self, wx: i32, wz: i32) -> Col {
        let mut ch = [0.0f32; 4];
        self.field.sample_all(wx, wz, &mut ch);
        self.col_from_ch(&ch)
    }

    fn col_from_ch(&self, ch: &[f32]) -> Col {
        let elev = (ch[0] * 2.0 - 1.0) * 36.0;
        let height = (self.sea as f32 + elev).round() as i32;
        let lake = ch[2] > 0.78 && elev > 2.0 && elev < 18.0;
        Col {
            height,
            water: if lake { height.max(self.sea) + 3 } else { self.sea },
            temp: ch[1],
            humid: ch[2],
            cave: ch[3],
        }
    }

    /// Coarse silhouette for far LOD — hash height, not the overlapping field.
    /// Far sections sample kilometres of unique (x,z); running InfiniteDiffusion
    /// there grows the tile cache without bound and never settles.
    fn lod_height(&self, wx: i32, wz: i32) -> i32 {
        let n = hash_noise(self.seed as u64, wx, wz, 64);
        let elev = (n * 2.0 - 1.0) * 36.0 * self.relief.clamp(0.25, 4.0).sqrt();
        (self.sea as f32 + elev).round() as i32
    }

    fn dress(&self, c: &Col, _wx: i32, _wz: i32) -> BlockId {
        use placement::SurfaceKind::*;
        let kind = if c.height <= c.water {
            Shore
        } else if c.temp < COLD || c.height - self.sea > SNOW_ABOVE {
            Snowy
        } else if c.temp > HOT && c.humid < DRY {
            Desert
        } else {
            Grassy
        };
        self.mat.dress[kind as usize]
    }

    fn carved(&self, c: &Col, wy: i32) -> bool {
        let depth = c.height - wy;
        depth > 8 && depth < 48 && c.cave > 0.62 && wy < c.height - 3
    }

    fn cell(&self, c: &Col, wx: i32, wy: i32, wz: i32) -> BlockId {
        if wy < c.height {
            if wy >= c.height - 1 {
                self.dress(c, wx, wz)
            } else if wy >= c.height - 3 {
                self.mat.crust[0]
            } else if self.carved(c, wy) {
                AIR
            } else {
                let depth = c.height - wy;
                if depth <= self.mat.max_scattered_depth {
                    let roll = cell_hash(self.seed, wx, wy, wz);
                    let mut cut = 0u32;
                    for slice in &self.mat.seams {
                        if depth < slice.min_depth {
                            break;
                        }
                        cut += slice.width;
                        if roll < cut {
                            return slice.id;
                        }
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
}

#[derive(Clone, Copy)]
struct Col {
    height: i32,
    water: i32,
    temp: f32,
    humid: f32,
    cave: f32,
}

impl TerrainGenerator for DiffusionTerrain {
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
        let mut ch0 = [0.0f32; CHUNK_SIZE * CHUNK_SIZE];
        self.field
            .fill_ch0(x0, z0, CHUNK_SIZE as u32, CHUNK_SIZE as u32, &mut ch0);
        let mut heights = [0i32; CHUNK_SIZE * CHUNK_SIZE];
        for i in 0..CHUNK_SIZE * CHUNK_SIZE {
            let elev = (ch0[i] * 2.0 - 1.0) * 36.0;
            heights[i] = (self.sea as f32 + elev).round() as i32;
        }
        heights
    }

    fn surface_at(&self, wx: i32, wz: i32) -> BlockId {
        let c = self.column(wx, wz);
        self.dress(&c, wx, wz)
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
            .unwrap_or_else(|| ChunkData::from_cells(Box::new([AIR; super::chunk::CHUNK_VOLUME])))
    }

    #[allow(clippy::needless_range_loop)] // lx/lz are world-space offsets, not just array indices
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
            temp: 0.5,
            humid: 0.5,
            cave: 0.0,
        }; CHUNK_SIZE]; CHUNK_SIZE];
        let mut buf = [0.0f32; CHUNK_SIZE * CHUNK_SIZE * 4];
        self.field
            .fill_all(x0, z0, CHUNK_SIZE as u32, CHUNK_SIZE as u32, &mut buf);
        let mut heights = [0i32; CHUNK_SIZE * CHUNK_SIZE];
        let mut max_top = i32::MIN;
        let mut min_h = i32::MAX;
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let i = (lz * CHUNK_SIZE + lx) * 4;
                let c = self.col_from_ch(&buf[i..i + 4]);
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
                let mut cells = Box::new([AIR; super::chunk::CHUNK_VOLUME]);
                for lz in 0..CHUNK_SIZE {
                    for lx in 0..CHUNK_SIZE {
                        let c = &cols[lz][lx];
                        let wx = x0 + lx as i32;
                        let wz = z0 + lz as i32;
                        for ly in 0..CHUNK_SIZE {
                            let wy = y0 + ly as i32;
                            cells[super::chunk::Chunk::index(lx, ly, lz)] =
                                self.cell(c, wx, wy, wz);
                        }
                    }
                }
                (cyy, ChunkData::from_cells(cells))
            })
            .collect();
        (chunks, heights)
    }
}

/// Shared handle workers clone.
pub type Generator = Arc<dyn TerrainGenerator>;

pub fn classic(registry: &mut BlockRegistry, seed: i64) -> Generator {
    Arc::new(super::generation::Terrain::new(registry, 20.0, seed))
}

pub fn diffusion(registry: &mut BlockRegistry, seed: i64, cfg: DiffusionCfg) -> Generator {
    Arc::new(DiffusionTerrain::new(registry, cfg, seed))
}

pub fn diffusion_v2(registry: &mut BlockRegistry, seed: i64, cfg: DiffusionCfg) -> Generator {
    Arc::new(super::diffusion_v2::DiffusionV2::new(registry, cfg, seed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::BlockRegistry;

    #[test]
    fn diffusion_is_seed_and_order_stable() {
        let mut ra = BlockRegistry::with_builtins();
        let mut rb = BlockRegistry::with_builtins();
        let a = DiffusionTerrain::new(&mut ra, DiffusionCfg::default(), 9);
        let b = DiffusionTerrain::new(&mut rb, DiffusionCfg::default(), 9);
        for z in [0, 3, 16, -8] {
            for x in [0, 7, -3] {
                assert_eq!(a.height(x, z), b.height(x, z));
                assert_eq!(
                    a.block_at(x, a.height(x, z) - 1, z, 0),
                    b.block_at(x, b.height(x, z) - 1, z, 0)
                );
            }
        }
        let ca = a.generate(0, 0, 0);
        let cb = b.generate(0, 0, 0);
        assert_eq!(ca, cb);
    }

    #[test]
    fn from_text_round_trips_to_text() {
        let cfg = DiffusionCfg {
            tile: 64,
            stride: 16,
            phases: 4,
            relief: 1.5,
            version: 1,
        }
        .clamp();
        assert_eq!(DiffusionCfg::from_text(&cfg.to_text()), cfg);
        assert_eq!(DiffusionCfg::from_text(""), DiffusionCfg::default());
        assert_eq!(
            DiffusionCfg::from_text("tile=64").tile,
            64,
            "partial overlay on defaults"
        );
        let v2 = DiffusionCfg {
            version: 2,
            ..DiffusionCfg::default()
        }
        .clamp();
        assert_eq!(DiffusionCfg::from_text(&v2.to_text()), v2);
        assert_eq!(DiffusionCfg::from_text("version=2").version, 2);
        assert_eq!(DiffusionCfg::from_text("").version, 1);
    }

    #[test]
    fn clamp_applies_tile_before_stride_so_stride_cannot_exceed_tile() {
        let cfg = DiffusionCfg {
            tile: 100,
            stride: 80,
            phases: 1,
            relief: 9.0,
            version: 1,
        }
        .clamp();
        assert_eq!(cfg.tile, 64);
        assert!(cfg.stride <= cfg.tile, "stride={} tile={}", cfg.stride, cfg.tile);
        assert_eq!(cfg.stride, 64);
        assert_eq!(cfg.phases, 2);
        assert_eq!(cfg.relief, 4.0);
    }

    #[test]
    fn clamp_snaps_to_values_the_knob_stepper_can_display() {
        let cfg = DiffusionCfg {
            tile: 48,
            stride: 12,
            phases: 1,
            relief: 0.25,
            version: 1,
        }
        .clamp();
        assert!(
            DiffusionCfg::TILES.contains(&cfg.tile),
            "tile {} not in {:?}",
            cfg.tile,
            DiffusionCfg::TILES
        );
        assert_eq!(cfg.stride % DiffusionCfg::STRIDE_STEP, 0);
        assert!(cfg.stride >= DiffusionCfg::MIN_STRIDE && cfg.stride <= cfg.tile);
        assert!((DiffusionCfg::PHASES_MIN..=DiffusionCfg::PHASES_MAX).contains(&cfg.phases));
        assert!(
            DiffusionCfg::RELIEFS
                .iter()
                .any(|v| (*v - cfg.relief).abs() < f32::EPSILON),
            "relief {} not in {:?}",
            cfg.relief,
            DiffusionCfg::RELIEFS
        );
    }

    #[test]
    fn default_matches_spec_new_and_is_stepper_reachable() {
        let cfg = DiffusionCfg::default();
        let spec = Spec::new(0);
        assert_eq!(cfg.tile, spec.tile);
        assert_eq!(cfg.stride, spec.stride);
        assert_eq!(cfg.phases, spec.phases);
        assert!(DiffusionCfg::TILES.contains(&cfg.tile));
        assert_eq!(cfg.stride % DiffusionCfg::STRIDE_STEP, 0);
        assert!((DiffusionCfg::PHASES_MIN..=DiffusionCfg::PHASES_MAX).contains(&cfg.phases));
        assert!(
            DiffusionCfg::RELIEFS
                .iter()
                .any(|v| (*v - cfg.relief).abs() < f32::EPSILON)
        );
        assert_eq!(cfg.clamp(), cfg);
    }

    #[test]
    fn lod_and_near_flood_to_the_same_sea() {
        let g = DiffusionTerrain::new(&mut BlockRegistry::with_builtins(), DiffusionCfg::default(), 13);
        let sea = g.sea_level();
        assert_eq!(sea, 20);
        let mut ocean = 0;
        for z in -16..16 {
            for x in -16..16 {
                let h = g.height(x, z);
                if h >= sea - 2 {
                    continue;
                }
                ocean += 1;
                let near_below = g.block_at(x, sea - 1, z, h);
                let near_at = g.block_at(x, sea, z, h);
                let lod_below = g.lod_block_at(x, sea - 1, z);
                let lod_at = g.lod_block_at(x, sea, z);
                assert_ne!(near_below, AIR, "ocean column ({x},{z}) must flood to sea");
                assert_eq!(near_at, AIR, "ocean column ({x},{z}) must stop flooding at sea");
                assert!(
                    lod_at == AIR || lod_at == g.deep(),
                    "LOD must not flood the sea cell at ({x},{z})"
                );
                assert_ne!(lod_below, AIR, "LOD must fill below sea at ({x},{z})");
            }
        }
        assert!(ocean > 0, "seed 13 must have open-ocean columns in the sample");
    }

    #[test]
    fn diffusion_kind_is_distinct() {
        assert_eq!(
            DiffusionTerrain::new(&mut BlockRegistry::with_builtins(), DiffusionCfg::default(), 1)
                .kind(),
            "diffusion"
        );
    }

    #[test]
    fn generate_matches_generate_column() {
        let g = DiffusionTerrain::new(&mut BlockRegistry::with_builtins(), DiffusionCfg::default(), 3);
        let a = g.generate(1, 0, -2);
        let (b, _) = g.generate_column(1, -2, 0..=0);
        assert_eq!(a, b[0].1);
    }

    #[test]
    fn generate_column_heights_match_height() {
        let g = DiffusionTerrain::new(&mut BlockRegistry::with_builtins(), DiffusionCfg::default(), 7);
        super::super::generation::assert_generate_column_heights_match_height(
            &g,
            &[(0, 0), (2, -3), (-1, 7), (4, 4)],
        );
    }

    #[test]
    fn heights_16_matches_height() {
        let g = DiffusionTerrain::new(&mut BlockRegistry::with_builtins(), DiffusionCfg::default(), 11);
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
    fn generate_agrees_with_per_cell_block_at() {
        use crate::world::chunk::{CHUNK_SIZE, CHUNK_VOLUME, Chunk};
        let g = DiffusionTerrain::new(&mut BlockRegistry::with_builtins(), DiffusionCfg::default(), 5);
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
                        cells[Chunk::index(lx, ly, lz)] =
                            g.block_at(wx, y0 + ly as i32, wz, h);
                    }
                }
            }
            assert_eq!(got, ChunkData::from_cells(cells), "chunk {cx},{cy},{cz}");
        }
    }

    fn chunk_data_bytes(data: &ChunkData) -> Vec<u8> {
        match data {
            ChunkData::Uniform(id) => {
                let mut b = vec![0u8];
                b.extend_from_slice(&id.0.to_le_bytes());
                b
            }
            ChunkData::Paletted { palette, cells } => {
                let mut b = vec![1u8];
                b.extend_from_slice(&(palette.len() as u32).to_le_bytes());
                for p in palette {
                    b.extend_from_slice(&p.0.to_le_bytes());
                }
                b.extend_from_slice(&cells[..]);
                b
            }
            ChunkData::Dense(cells) => {
                let mut b = vec![2u8];
                for id in cells.iter() {
                    b.extend_from_slice(&id.0.to_le_bytes());
                }
                b
            }
        }
    }

    /// Pin `fnv1a_32` over six fixed seed-42 chunks. Values locked before the
    /// batch-sample pass; a mismatch means generated `ChunkData` bytes moved.
    #[test]
    fn diffusion_chunk_byte_pin() {
        use crate::hash::fnv1a_32;
        let g = DiffusionTerrain::new(
            &mut BlockRegistry::with_builtins(),
            DiffusionCfg::default(),
            42,
        );
        // surface, lake column, cave band, deep, two far coords.
        let pins: [(&str, i32, i32, i32, u32); 6] = [
            ("surface", 0, 1, 0, 0x9b39f240),
            ("lake", -22, 1, -24, 0x1b5636d6),
            ("cave", -24, -3, -24, 0x24dfb131),
            ("deep", 0, -20, 0, 0x24ae7d4e),
            ("far_a", 6_250_000, 0, 0, 0x7f5a5a13),
            ("far_b", -6_250_000, -2, 3, 0x9f993f6c),
        ];
        for (name, cx, cy, cz, want) in pins {
            assert_eq!(
                fnv1a_32(&chunk_data_bytes(&g.generate(cx, cy, cz))),
                want,
                "{name} ({cx},{cy},{cz})"
            );
        }
    }

    /// Column generation cost, n=24 columns × 4 layers. Ignored timing gauge.
    /// Before batching: classic=140.3ms diffusion=401.8ms ratio=2.86.
    /// After batching: classic=141.8ms diffusion=176.7ms ratio=1.25 (2.27× vs before).
    /// Run with `cargo test --release worldgen_column_cost -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn worldgen_column_cost() {
        use std::time::Instant;
        let classic = super::classic(&mut BlockRegistry::with_builtins(), 42);
        let diffusion = super::diffusion(
            &mut BlockRegistry::with_builtins(),
            42,
            DiffusionCfg::default(),
        );
        let n = 24i32;
        let t0 = Instant::now();
        for cz in 0..n {
            for cx in 0..n {
                let _ = classic.generate_column(cx, cz, 0..=3);
            }
        }
        let classic_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t1 = Instant::now();
        for cz in 0..n {
            for cx in 0..n {
                let _ = diffusion.generate_column(cx, cz, 0..=3);
            }
        }
        let diffusion_ms = t1.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "worldgen_column_cost n={n} classic={classic_ms:.1}ms diffusion={diffusion_ms:.1}ms ratio={:.2}",
            diffusion_ms / classic_ms.max(0.001)
        );
    }

    #[test]
    #[ignore]
    fn worldgen_column_cost_by_phases() {
        use std::time::Instant;
        let n = 16i32;
        for phases in [2u32, 3, 4, 6] {
            let cfg = DiffusionCfg { phases, ..Default::default() };
            let g = super::diffusion(&mut BlockRegistry::with_builtins(), 42, cfg);
            let t = Instant::now();
            for cz in 0..n {
                for cx in 0..n {
                    let _ = g.generate_column(cx, cz, 0..=3);
                }
            }
            eprintln!(
                "worldgen_phases phases={phases} n={n} ms={:.1}",
                t.elapsed().as_secs_f64() * 1000.0
            );
        }
    }
}
