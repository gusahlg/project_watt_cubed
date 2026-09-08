//! InfiniteDiffusion terrain: overlapping-window fields interpreted by the
//! same placement table as classic noise. The field is the substrate; this
//! file is the voxel rules.

use std::sync::Arc;

use infinite_field::{InfiniteField, Score, Spec};

use super::chunk::{CHUNK_SIZE, ChunkData};
use super::generation::{cell_hash, TerrainGenerator};
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
}

impl Default for DiffusionCfg {
    fn default() -> Self {
        Self {
            tile: 32,
            stride: 16,
            phases: 2,
            relief: 1.0,
        }
    }
}

impl DiffusionCfg {
    pub fn clamp(mut self) -> Self {
        self.tile = self.tile.clamp(16, 64);
        self.stride = self.stride.clamp(8, self.tile);
        self.phases = self.phases.clamp(2, 8);
        self.relief = self.relief.clamp(0.25, 4.0);
        self
    }
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
            .into_iter()
            .next()
            .map(|(_, data)| data)
            .unwrap_or_else(|| ChunkData::from_cells(Box::new([AIR; super::chunk::CHUNK_VOLUME])))
    }

    fn generate_column(
        &self,
        cx: i32,
        cz: i32,
        cy: std::ops::RangeInclusive<i32>,
    ) -> Vec<(i32, ChunkData)> {
        let x0 = cx * CHUNK_SIZE as i32;
        let z0 = cz * CHUNK_SIZE as i32;
        let mut cols = [[Col {
            height: 0,
            water: SEA,
            temp: 0.5,
            humid: 0.5,
            cave: 0.0,
        }; CHUNK_SIZE]; CHUNK_SIZE];
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                cols[lz][lx] = self.column(x0 + lx as i32, z0 + lz as i32);
            }
        }
        let mut max_top = i32::MIN;
        let mut min_h = i32::MAX;
        for row in &cols {
            for c in row {
                max_top = max_top.max(c.height.max(c.water));
                min_h = min_h.min(c.height);
            }
        }
        let deep_cut = min_h - self.mat.max_scattered_depth.max(48);
        cy.map(|cyy| {
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
        .collect()
    }
}

/// Shared handle workers clone.
pub type Generator = Arc<dyn TerrainGenerator>;

pub fn classic(registry: &mut BlockRegistry, seed: i64) -> Generator {
    Arc::new(super::generation::SineHills::new(registry, 20.0, seed))
}

pub fn diffusion(registry: &mut BlockRegistry, seed: i64, cfg: DiffusionCfg) -> Generator {
    Arc::new(DiffusionTerrain::new(registry, cfg, seed))
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
        let b = g.generate_column(1, -2, 0..=0);
        assert_eq!(a, b[0].1);
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
            let mut cfg = DiffusionCfg::default();
            cfg.phases = phases;
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
