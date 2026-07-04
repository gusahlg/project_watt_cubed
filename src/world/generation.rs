//! Terrain generation, decoupled from chunk storage so the algorithm can be
//! swapped without touching how voxels are stored or drawn.
//!
//! A generator works in [`BlockId`]s, not raw element compositions: it resolves the
//! handful of blocks it places against the [`BlockRegistry`] once, up front, so
//! filling a cell stays a cheap id copy with no per-voxel allocation.
//!
//! Two layers of terrain exist in the infinite-Y world:
//! - the **surface band**: a heightfield h(x, z) with grass/dirt layering on
//!   top and stone forever down;
//! - **flying islands** for `y >= 64`: a 3D value-noise field is island-solid
//!   where it exceeds a threshold that rises with altitude, so islands get
//!   fewer and smaller the higher you fly, asymptotically vanishing but never
//!   impossible (the threshold clamps below the noise maximum).
//!
//! Generation is a pure function of (seed, chunk coord) — worker threads and
//! multiplayer clients all reproduce identical chunks. Whole-chunk generation
//! proves uniformity where it can (all-stone below the terrain, all-air above
//! it and below the island band) and otherwise collapses an all-identical
//! dense fill, so sky and deep rock cost bytes, not kilobytes.
use super::chunk::{CHUNK_SIZE, CHUNK_VOLUME, Chunk, ChunkData};
use crate::block::registry::{AIR, BlockId, BlockRegistry};

/// Produces terrain for absolute world coordinates.
///
/// An implementor provides a surface [`height`](Self::height) and the three blocks
/// it layers (surface, subsoil, deep); the default [`block_at`](Self::block_at)
/// turns those into stacked terrain and the default [`generate`](Self::generate)
/// fills whole chunks from it. A richer generator (islands, caves, biomes)
/// overrides those instead.
pub trait TerrainGenerator {
    /// Surface height for a world column: the number of solid layers stacked from
    /// `y = 0` upward.
    fn height(&self, wx: i32, wz: i32) -> i32;

    /// The block placed on the surface (the topmost solid layer).
    fn surface(&self) -> BlockId;
    /// The block placed just below the surface.
    fn subsoil(&self) -> BlockId;
    /// The block placed deep underground.
    fn deep(&self) -> BlockId;

    /// The block at a world coordinate, given the column's surface `height`.
    ///
    /// Default layering: surface block on top, subsoil just below, deep block
    /// all the way down, and air above the surface.
    fn block_at(&self, _wx: i32, wy: i32, _wz: i32, height: i32) -> BlockId {
        if wy >= height {
            AIR
        } else if wy >= height - 1 {
            self.surface()
        } else if wy >= height - 3 {
            self.subsoil()
        } else {
            self.deep()
        }
    }

    /// Generate a whole 16-cube chunk's storage. The default densely evaluates
    /// [`block_at`](Self::block_at) (one [`height`](Self::height) per column)
    /// and collapses to [`ChunkData::Uniform`] when every cell agrees —
    /// correct for any `block_at`; implementors override it to *prove*
    /// uniformity without evaluating (see [`SineHills`]).
    fn generate(&self, cx: i32, cy: i32, cz: i32) -> ChunkData {
        let y0 = cy * CHUNK_SIZE as i32;
        let mut cells = Box::new([0u8; CHUNK_VOLUME]);
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let wx = cx * CHUNK_SIZE as i32 + lx as i32;
                let wz = cz * CHUNK_SIZE as i32 + lz as i32;
                let height = self.height(wx, wz);
                for ly in 0..CHUNK_SIZE {
                    let id = self.block_at(wx, y0 + ly as i32, wz, height);
                    debug_assert!(id.0 < 256, "dense cells are u8");
                    cells[Chunk::index(lx, ly, lz)] = id.0 as u8;
                }
            }
        }
        collapse(cells)
    }
}

/// Collapse a dense fill to `Uniform` when every cell came out identical —
/// one linear byte scan, trivially cheap next to the fill itself.
fn collapse(cells: Box<[u8; CHUNK_VOLUME]>) -> ChunkData {
    let first = cells[0];
    if cells.iter().all(|&c| c == first) {
        ChunkData::Uniform(BlockId(first as u16))
    } else {
        ChunkData::Dense(cells)
    }
}

// ---------------------------------------------------------------------------
// Island noise — pure functions of (seed, position), shared by the generator
// and testable standalone.
// ---------------------------------------------------------------------------

/// Islands exist only at or above this altitude.
pub const ISLAND_MIN_Y: i32 = 64;
/// Lattice cell size (blocks) of the island noise's base octave; the second
/// octave runs at half this.
const ISLAND_CELL: f32 = 24.0;
/// Base-octave frequency, and the second octave's (double).
const ISLAND_FREQ_0: f32 = 1.0 / ISLAND_CELL;
const ISLAND_FREQ_1: f32 = 2.0 / ISLAND_CELL;
/// The solidity threshold never exceeds this, so islands thin out with
/// altitude but never become impossible (noise tops out below 1.0).
const ISLAND_MAX_THRESHOLD: f32 = 0.97;

/// The noise level a cell must exceed to be island-solid at altitude `y`:
/// rises linearly from 0.60 at the island floor, clamped to 0.97.
pub fn island_threshold(y: i32) -> f32 {
    (0.60 + (y - ISLAND_MIN_Y) as f32 * 0.000_35).min(ISLAND_MAX_THRESHOLD)
}

/// The island field at a world cell: 2-octave 3D value noise in [0, 1).
pub fn island_noise(seed: i64, x: i32, y: i32, z: i32) -> f32 {
    let a = octave(
        octave_seed(seed, 0),
        x as f32 * ISLAND_FREQ_0,
        y as f32 * ISLAND_FREQ_0,
        z as f32 * ISLAND_FREQ_0,
    );
    let b = octave(
        octave_seed(seed, 1),
        x as f32 * ISLAND_FREQ_1,
        y as f32 * ISLAND_FREQ_1,
        z as f32 * ISLAND_FREQ_1,
    );
    (a + 0.5 * b) * (1.0 / 1.5)
}

/// Whether the island field makes this cell solid. Never true below
/// [`ISLAND_MIN_Y`].
pub fn island_at(seed: i64, x: i32, y: i32, z: i32) -> bool {
    y >= ISLAND_MIN_Y && island_noise(seed, x, y, z) > island_threshold(y)
}

/// Decorrelate the two octaves' lattices from the one world seed.
fn octave_seed(seed: i64, octave: u64) -> u64 {
    (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ octave.wrapping_mul(0xD1B5_4A32_D192_ED03)
}

/// Seeded hash of an integer lattice point onto a uniform value in [0, 1).
fn lattice(seed: u64, x: i32, y: i32, z: i32) -> f32 {
    let mut h = seed
        ^ (x as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u32 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ (z as u32 as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
    // splitmix64 finisher: breaks the linearity of the per-axis products.
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    (h >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Smoothstep fade, the classic value-noise interpolant.
fn fade(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Bilinear lattice blend in the XZ plane at integer lattice level `ly`
/// (`x`/`z` already in lattice units).
fn plane_value(seed: u64, x: f32, ly: i32, z: f32) -> f32 {
    let (x0, z0) = (x.floor(), z.floor());
    let (xi, zi) = (x0 as i32, z0 as i32);
    let (tx, tz) = (fade(x - x0), fade(z - z0));
    let v00 = lattice(seed, xi, ly, zi);
    let v10 = lattice(seed, xi + 1, ly, zi);
    let v01 = lattice(seed, xi, ly, zi + 1);
    let v11 = lattice(seed, xi + 1, ly, zi + 1);
    lerp(lerp(v00, v10, tx), lerp(v01, v11, tx), tz)
}

/// One value-noise octave: the two bracketing XZ plane blends, faded in Y.
/// Split this way (rather than a plain trilinear) so [`OctaveColumn`] can
/// cache the plane blends per column and stay bit-identical.
fn octave(seed: u64, x: f32, y: f32, z: f32) -> f32 {
    let y0 = y.floor();
    let yi = y0 as i32;
    let ty = fade(y - y0);
    lerp(plane_value(seed, x, yi, z), plane_value(seed, x, yi + 1, z), ty)
}

/// One octave sampled down a fixed (x, z) column: the XZ plane blends are
/// computed once per lattice level instead of once per cell, which is the
/// generator's hot path (a 20-cell column touches at most 4 levels).
struct OctaveColumn {
    freq: f32,
    /// Lowest lattice level cached.
    base: i32,
    /// `plane_value` at `base + i`. A 20-cell column at the half-cell octave
    /// spans at most ceil(19/12) + 1 = 3 level intervals, so 5 always fits.
    planes: [f32; 5],
}

impl OctaveColumn {
    fn new(seed: u64, wx: i32, wz: i32, freq: f32, y_lo: i32, y_hi: i32) -> Self {
        let x = wx as f32 * freq;
        let z = wz as f32 * freq;
        let base = (y_lo as f32 * freq).floor() as i32;
        let top = (y_hi as f32 * freq).floor() as i32 + 1;
        debug_assert!((top - base) < 5, "column spans more levels than cached");
        let mut planes = [0.0; 5];
        for (i, level) in (base..=top).enumerate() {
            planes[i] = plane_value(seed, x, level, z);
        }
        Self { freq, base, planes }
    }

    /// Bit-identical to [`octave`] at (x, y, z) for y within the built range.
    fn sample(&self, y: i32) -> f32 {
        let fy = y as f32 * self.freq;
        let y0 = fy.floor();
        let ty = fade(fy - y0);
        let i = (y0 as i32 - self.base) as usize;
        lerp(self.planes[i], self.planes[i + 1], ty)
    }
}

/// The full island field down one column, cached per lattice level —
/// bit-identical to [`island_noise`] (asserted in tests), several times
/// cheaper for whole-chunk generation.
struct IslandColumn {
    o0: OctaveColumn,
    o1: OctaveColumn,
}

impl IslandColumn {
    fn new(seed: i64, wx: i32, wz: i32, y_lo: i32, y_hi: i32) -> Self {
        Self {
            o0: OctaveColumn::new(octave_seed(seed, 0), wx, wz, ISLAND_FREQ_0, y_lo, y_hi),
            o1: OctaveColumn::new(octave_seed(seed, 1), wx, wz, ISLAND_FREQ_1, y_lo, y_hi),
        }
    }

    /// [`island_at`] for this column's (x, z), y within the built range.
    fn solid(&self, y: i32) -> bool {
        y >= ISLAND_MIN_Y
            && (self.o0.sample(y) + 0.5 * self.o1.sample(y)) * (1.0 / 1.5) > island_threshold(y)
    }
}

// ---------------------------------------------------------------------------
// SineHills — the game's generator.
// ---------------------------------------------------------------------------

/// Rolling hills built from layered sine waves, plus the flying-island field —
/// deterministic and dependency-free. A `seed` shifts the wave phases and the
/// island lattice so each world looks different while staying fully
/// reproducible from that one number. `Clone` because generation jobs run on
/// worker threads (see [`pipeline`](super::pipeline)): each job carries its own
/// copy of this handful of plain numbers and ids.
#[derive(Clone)]
pub struct SineHills {
    /// Average terrain height that the waves oscillate around.
    pub base: f32,
    /// The world seed this generator was built from (saved and restored verbatim).
    pub seed: i64,
    // Phase offsets derived from the seed, so different seeds sample different terrain.
    offset_x: f32,
    offset_z: f32,
    // Block palette, resolved once against the registry so generation never looks
    // anything up per cell.
    grass: BlockId,
    dirt: BlockId,
    stone: BlockId,
}

impl SineHills {
    /// Build the generator for a seed, resolving its grass/dirt/stone palette against
    /// the registry. Panics if those built-in blocks are missing — they are part of
    /// every [`BlockRegistry::with_builtins`].
    pub fn new(registry: &BlockRegistry, base: f32, seed: i64) -> Self {
        let resolve = |name: &str| {
            registry
                .id_by_name(name)
                .unwrap_or_else(|| panic!("SineHills needs the built-in '{name}' block"))
        };
        // Spread the seed's bits into two large, unrelated phase offsets.
        let offset_x = (seed.wrapping_mul(0x2545F491_4F6CDD1D) as u32 as f32) * 0.000_01;
        let offset_z = (seed.wrapping_mul(0x9E3779B9_7F4A7C15u64 as i64) as u32 as f32) * 0.000_01;
        Self {
            base,
            seed,
            offset_x,
            offset_z,
            grass: resolve("Grass"),
            dirt: resolve("Dirt"),
            stone: resolve("Stone"),
        }
    }

    /// Surface-band layering for a cell below its column's surface.
    fn ground_block(&self, wy: i32, height: i32) -> BlockId {
        if wy >= height - 1 {
            self.grass
        } else if wy >= height - 3 {
            self.dirt
        } else {
            self.stone
        }
    }

    /// The block for an island-solid cell, from what sits above it in the
    /// field: exposed top -> grass, within 3 below a surface cell -> dirt,
    /// buried deeper -> stone.
    fn island_block(&self, above: [bool; 4]) -> BlockId {
        if !above[0] {
            self.grass
        } else if !above[1] || !above[2] || !above[3] {
            self.dirt
        } else {
            self.stone
        }
    }
}

impl TerrainGenerator for SineHills {
    fn height(&self, wx: i32, wz: i32) -> i32 {
        let x = wx as f32 + self.offset_x;
        let z = wz as f32 + self.offset_z;

        let h = self.base
            + 6.0 * (x * 0.08).sin() * (z * 0.08).cos()
            + 3.0 * (x * 0.21 + z * 0.13).sin()
            + 2.0 * (z * 0.30).cos();

        // Floor of 1 so there is always ground; no ceiling — Y is infinite now
        // (the waves top out at base + 11 regardless).
        h.round().max(1.0) as i32
    }

    fn surface(&self) -> BlockId {
        self.grass
    }
    fn subsoil(&self) -> BlockId {
        self.dirt
    }
    fn deep(&self) -> BlockId {
        self.stone
    }

    /// Surface band below `height`, the island field above it.
    fn block_at(&self, wx: i32, wy: i32, wz: i32, height: i32) -> BlockId {
        if wy < height {
            self.ground_block(wy, height)
        } else if island_at(self.seed, wx, wy, wz) {
            self.island_block([
                island_at(self.seed, wx, wy + 1, wz),
                island_at(self.seed, wx, wy + 2, wz),
                island_at(self.seed, wx, wy + 3, wz),
                island_at(self.seed, wx, wy + 4, wz),
            ])
        } else {
            AIR
        }
    }

    /// Whole-chunk generation with cheap uniformity proofs:
    /// - entirely below every column's deep line -> `Uniform(stone)`, no fill;
    /// - entirely above every column's surface and below the island band
    ///   -> `Uniform(air)`, no fill;
    /// - otherwise a dense fill (island noise sampled per column, cached per
    ///   lattice level), collapsed to `Uniform` when all cells agree — which
    ///   is how island-free sky chunks above y=64 become uniform air.
    fn generate(&self, cx: i32, cy: i32, cz: i32) -> ChunkData {
        let y0 = cy * CHUNK_SIZE as i32;
        let y1 = y0 + CHUNK_SIZE as i32 - 1;

        // One height per column, reused by the fast paths and the fill.
        let mut heights = [0i32; CHUNK_SIZE * CHUNK_SIZE];
        let (mut h_min, mut h_max) = (i32::MAX, i32::MIN);
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let h = self.height(
                    cx * CHUNK_SIZE as i32 + lx as i32,
                    cz * CHUNK_SIZE as i32 + lz as i32,
                );
                heights[lx + lz * CHUNK_SIZE] = h;
                h_min = h_min.min(h);
                h_max = h_max.max(h);
            }
        }

        // Below the shallowest column's dirt line: stone forever down.
        if y1 < h_min - 3 {
            return ChunkData::Uniform(self.stone);
        }
        // Above the tallest column and below the island band: guaranteed air.
        // (Within the band the threshold never exceeds the noise maximum, so
        // air can't be proven — the dense fill below collapses it instead.)
        if y0 >= h_max && y1 < ISLAND_MIN_Y {
            return ChunkData::Uniform(AIR);
        }

        let mut cells = Box::new([0u8; CHUNK_VOLUME]);
        // A cell can only be island-solid at y >= 64, so chunks entirely below
        // the band skip the noise pass (their mask stays all-false).
        let islands_possible = y1 >= ISLAND_MIN_Y;
        // Island mask for one column: the chunk's 16 cells plus the 4 above
        // that grass/dirt typing looks at.
        let mut isl = [false; CHUNK_SIZE + 4];
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let wx = cx * CHUNK_SIZE as i32 + lx as i32;
                let wz = cz * CHUNK_SIZE as i32 + lz as i32;
                let height = heights[lx + lz * CHUNK_SIZE];

                if islands_possible {
                    let col = IslandColumn::new(self.seed, wx, wz, y0, y1 + 4);
                    for (k, cell) in isl.iter_mut().enumerate() {
                        *cell = col.solid(y0 + k as i32);
                    }
                }

                for ly in 0..CHUNK_SIZE {
                    let wy = y0 + ly as i32;
                    let id = if wy < height {
                        self.ground_block(wy, height)
                    } else if isl[ly] {
                        self.island_block([isl[ly + 1], isl[ly + 2], isl[ly + 3], isl[ly + 4]])
                    } else {
                        AIR
                    };
                    debug_assert!(id.0 < 256, "dense cells are u8");
                    cells[Chunk::index(lx, ly, lz)] = id.0 as u8;
                }
            }
        }
        collapse(cells)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hills(seed: i64) -> SineHills {
        SineHills::new(&BlockRegistry::with_builtins(), 20.0, seed)
    }

    #[test]
    fn island_noise_is_deterministic() {
        for &(x, y, z) in &[(0, 64, 0), (-317, 900, 512), (12_345, 70, -9_876)] {
            assert_eq!(island_noise(42, x, y, z), island_noise(42, x, y, z));
        }
        // And actually driven by the seed.
        let differing = (0..64)
            .filter(|&i| island_noise(1, i * 31, 100, -i * 17) != island_noise(2, i * 31, 100, -i * 17))
            .count();
        assert!(differing > 48, "different seeds sample a different field");
    }

    #[test]
    fn island_threshold_rises_monotonically_and_clamps() {
        let mut prev = island_threshold(ISLAND_MIN_Y);
        assert_eq!(prev, 0.60);
        for y in (ISLAND_MIN_Y + 1..3000).step_by(7) {
            let t = island_threshold(y);
            assert!(t >= prev, "threshold never falls with altitude (y={y})");
            assert!(t <= ISLAND_MAX_THRESHOLD);
            prev = t;
        }
        assert_eq!(island_threshold(5000), ISLAND_MAX_THRESHOLD, "clamped high up");
    }

    #[test]
    fn island_density_strictly_decreases_with_altitude() {
        // Census a horizontal slab of cells at three altitudes: the rising
        // threshold must thin the field out monotonically.
        let census = |y: i32| -> usize {
            let mut solid = 0;
            for x in -64..64 {
                for z in -64..64 {
                    solid += island_at(9, x, y, z) as usize;
                }
            }
            solid
        };
        let (low, mid, high) = (census(80), census(400), census(1200));
        assert!(low > mid, "y=80 denser than y=400 ({low} vs {mid})");
        assert!(mid > high, "y=400 denser than y=1200 ({mid} vs {high})");
        assert!(low > 0, "islands exist near the band floor");
    }

    #[test]
    fn islands_never_generate_below_the_band() {
        for y in [-1000, -1, 0, 32, 63] {
            for x in -32..32 {
                assert!(!island_at(9, x * 5, y, x * 3 - 7), "no islands below y=64");
            }
        }
        // Boundary inclusive: y=64 is allowed to be solid (find one somewhere).
        let any = (-200..200).any(|x| (-200..200).step_by(8).any(|z| island_at(9, x, 64, z)));
        assert!(any, "the band floor itself can hold islands");
    }

    #[test]
    fn column_cache_matches_the_pure_noise_fn() {
        // The generator's per-column fast path must be bit-identical to the
        // standalone island functions.
        for seed in [1, 42, -777] {
            for (wx, wz) in [(0, 0), (13, -27), (-1000, 999)] {
                let (y_lo, y_hi) = (96, 96 + 19);
                let col = IslandColumn::new(seed, wx, wz, y_lo, y_hi);
                for y in y_lo..=y_hi {
                    assert_eq!(col.solid(y), island_at(seed, wx, y, wz), "at y={y}");
                }
            }
        }
    }

    #[test]
    fn generated_chunks_match_per_cell_block_at() {
        // Whole-chunk generation (fast paths, column cache, collapse) must be
        // cell-identical to the naive per-cell recipe.
        let g = hills(3);
        for (cx, cy, cz) in [(0, 0, 0), (0, 1, 0), (2, 4, -3), (-1, 5, 7), (0, -2, 0), (0, 3, 0)] {
            let chunk = Chunk::new(cx, cy, cz, &g);
            for lz in 0..CHUNK_SIZE {
                for lx in 0..CHUNK_SIZE {
                    let wx = cx * CHUNK_SIZE as i32 + lx as i32;
                    let wz = cz * CHUNK_SIZE as i32 + lz as i32;
                    let h = g.height(wx, wz);
                    for ly in 0..CHUNK_SIZE {
                        let wy = cy * CHUNK_SIZE as i32 + ly as i32;
                        assert_eq!(
                            chunk.get_local(lx, ly, lz),
                            g.block_at(wx, wy, wz, h),
                            "cell ({wx}, {wy}, {wz}) of chunk ({cx}, {cy}, {cz})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn uniform_proofs_hold() {
        let g = hills(11);
        // Deep rock: provably uniform stone without a fill.
        assert_eq!(g.generate(0, -1, 0), ChunkData::Uniform(g.deep()));
        assert_eq!(g.generate(5, -100, -5), ChunkData::Uniform(g.deep()));
        // Sky below the island band: provably uniform air.
        assert_eq!(g.generate(0, 3, 0), ChunkData::Uniform(AIR));
        // Ground chunks stay dense (they mix layers and air).
        assert!(matches!(g.generate(0, 1, 0), ChunkData::Dense(_)));
    }
}
