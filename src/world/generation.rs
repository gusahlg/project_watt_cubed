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
//! The surface band's stone is seasoned with single-cell **ore veins**, but
//! only within the ore band (`depth = height - wy` in `3..=ORE_MAX_DEPTH`);
//! below it stone is provably pure, so the deep-rock uniformity proof stays
//! valid. Island stone rolls its own veins (the only Aerium source), lowland
//! surfaces turn to sand, and island tops above [`ICE_SURFACE_Y`] freeze over.
//!
//! Generation is a pure function of (seed, chunk coord) — worker threads and
//! multiplayer clients all reproduce identical chunks. Whole-chunk generation
//! proves uniformity where it can (all-stone below the ore band, all-air above
//! the terrain and below the island band) and otherwise collapses an
//! all-identical dense fill, so sky and deep rock cost bytes, not kilobytes.
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
const ISLAND_CELL: f64 = 24.0;
/// Base-octave frequency, and the second octave's (double). f64: the
/// world-coordinate -> lattice-coordinate reduction must run in f64 (see
/// [`reduce`]) — in f32 the ULP of a world coordinate reaches 32 blocks at
/// |w| = 2^28, larger than a whole lattice cell, well inside the certified
/// ±1e9 world border.
const ISLAND_FREQ_0: f64 = 1.0 / ISLAND_CELL;
const ISLAND_FREQ_1: f64 = 2.0 / ISLAND_CELL;
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
    let a = octave(octave_seed(seed, 0), x, y, z, ISLAND_FREQ_0);
    let b = octave(octave_seed(seed, 1), x, y, z, ISLAND_FREQ_1);
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

/// Reduce a world coordinate to its lattice cell and in-cell fraction, IN F64.
///
/// This is the one place world positions become lattice positions, and it must
/// not run in f32: an i32 world coordinate is exact in f64 (and the fraction
/// `t - floor(t)` is exact by Sterbenz), but in f32 the ULP of the coordinate
/// itself reaches 32 blocks at |w| = 2^28 — larger than a lattice cell — which
/// both quantized the fraction to steps (slab-shaped islands beyond ~1.5e8)
/// and inflated the apparent cell span past [`OctaveColumn`]'s plane cache
/// (index panic near y = 2.7e8). The i64 cell is fed to [`lattice`] as
/// `cell as i32` (a wrapping truncation): within the certified ±1e9 world the
/// cell never exceeds ±1e9/12 ≈ ±8.4e7, far inside i32, so the hash input —
/// and therefore every lattice value — is unchanged wherever the old f32 path
/// computed the right cell; beyond that it stays deterministic everywhere.
fn reduce(w: i32, freq: f64) -> (i64, f32) {
    let t = w as f64 * freq;
    let cell = t.floor() as i64;
    (cell, (t - cell as f64) as f32)
}

/// Bilinear lattice blend in the XZ plane at integer lattice level `ly`,
/// from [`reduce`]d cells (`xi`/`zi`) and in-cell fractions (`fx`/`fz`).
fn plane_value(seed: u64, xi: i32, fx: f32, ly: i32, zi: i32, fz: f32) -> f32 {
    let (tx, tz) = (fade(fx), fade(fz));
    let v00 = lattice(seed, xi, ly, zi);
    let v10 = lattice(seed, xi + 1, ly, zi);
    let v01 = lattice(seed, xi, ly, zi + 1);
    let v11 = lattice(seed, xi + 1, ly, zi + 1);
    lerp(lerp(v00, v10, tx), lerp(v01, v11, tx), tz)
}

/// One value-noise octave at a world cell: the two bracketing XZ plane blends,
/// faded in Y. Split this way (rather than a plain trilinear) so
/// [`OctaveColumn`] can cache the plane blends per column and stay
/// bit-identical.
fn octave(seed: u64, wx: i32, wy: i32, wz: i32, freq: f64) -> f32 {
    let (xi, fx) = reduce(wx, freq);
    let (zi, fz) = reduce(wz, freq);
    let (yi, fy) = reduce(wy, freq);
    let ty = fade(fy);
    let (xi, zi, yi) = (xi as i32, zi as i32, yi as i32);
    lerp(
        plane_value(seed, xi, fx, yi, zi, fz),
        plane_value(seed, xi, fx, yi + 1, zi, fz),
        ty,
    )
}

/// One octave sampled down a fixed (x, z) column: the XZ plane blends are
/// computed once per lattice level instead of once per cell, which is the
/// generator's hot path (a 20-cell column touches at most 4 levels).
struct OctaveColumn {
    freq: f64,
    /// Lowest lattice cell cached (i64: the exact [`reduce`] cell).
    base: i64,
    /// `plane_value` at cell `base + i`. Span proof: cells come from the
    /// monotone f64 map `y -> floor(fl(y * freq))`, so over a 20-cell column
    /// (y_hi - y_lo = 19) at the half-cell octave (freq = 1/12) the cell
    /// difference is at most floor(19/12) + 1 = 2 (the f64 rounding slack on
    /// |y| <= 2^31 is ~1e-7, far below the next integer). With the +1 top
    /// plane that is 4 entries, so 5 always fits — exactly, at any world
    /// coordinate, which the old f32 path could not guarantee past |y| = 2^28.
    planes: [f32; 5],
}

impl OctaveColumn {
    fn new(seed: u64, wx: i32, wz: i32, freq: f64, y_lo: i32, y_hi: i32) -> Self {
        let (xi, fx) = reduce(wx, freq);
        let (zi, fz) = reduce(wz, freq);
        let base = (y_lo as f64 * freq).floor() as i64;
        let top = (y_hi as f64 * freq).floor() as i64 + 1;
        debug_assert!(top - base < 5, "column spans more levels than cached");
        let mut planes = [0.0; 5];
        for (i, cell) in (base..=top).enumerate() {
            planes[i] = plane_value(seed, xi as i32, fx, cell as i32, zi as i32, fz);
        }
        Self { freq, base, planes }
    }

    /// Bit-identical to [`octave`] at (x, y, z) for y within the built range:
    /// same [`reduce`], same plane cells, same fade.
    fn sample(&self, y: i32) -> f32 {
        let (cell, fy) = reduce(y, self.freq);
        let ty = fade(fy);
        let i = (cell - self.base) as usize;
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
// Ore scattering — pure functions of (seed, world cell), one cheap hash per
// candidate stone cell. Everything here must stay bit-deterministic: workers
// and multiplayer clients re-derive the same veins from the same seed.
// ---------------------------------------------------------------------------

/// Shallowest depth (`height - wy`) at which ore can appear. Shallower cells
/// are grass/dirt anyway; the constant is the coal tier's floor.
pub const ORE_MIN_DEPTH: i32 = 3;
/// Deepest depth at which ore can appear. Strictly below this the stone is
/// PURE by construction — the fact that lets [`SineHills::generate`] claim
/// `Uniform(stone)` for chunks entirely beneath the band without a fill.
pub const ORE_MAX_DEPTH: i32 = 64;
/// Island surface cells at or above this altitude are Ice instead of grass —
/// the game's only natural Ice source.
pub const ICE_SURFACE_Y: i32 = 220;

/// Island stone rolls AeriumVein at 1/45 — flying islands are the *only*
/// natural Aerium source (the exploration reward that explains why they fly).
const ISLAND_AERIUM_W: u32 = u32::MAX / 45;
/// ...and QuartzVein at 1/160, stacked after the Aerium slice.
const ISLAND_QUARTZ_W: u32 = u32::MAX / 160;

/// One ore type the ground can roll: eligible from `min_depth` down, hit when
/// the cell's one hash lands in a slice `width` wide. Slices are stacked
/// cumulatively, so each ore's probability is exactly `width / 2^32`.
#[derive(Clone, Copy)]
struct Seam {
    min_depth: i32,
    width: u32,
    block: BlockId,
}

/// Seeded hash of a world cell onto a uniform `u32` — the single roll a
/// candidate ore cell makes. Same splitmix64 finisher as [`lattice`], seeded
/// on a different stream so veins don't correlate with island noise.
fn cell_hash(seed: i64, x: i32, y: i32, z: i32) -> u32 {
    let mut h = (seed as u64 ^ 0x517C_C1B7_2722_0A95)
        ^ (x as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u32 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ (z as u32 as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    (h >> 32) as u32
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
    sand: BlockId,
    ice: BlockId,
    aerium_vein: BlockId,
    quartz_vein: BlockId,
    /// Ground ore table, sorted by `min_depth` so the roll can stop at the
    /// first tier this cell is too shallow for.
    seams: [Seam; 10],
    /// Columns no taller than this get sand surfaces (lowland "beaches"):
    /// `base - 6`, precomputed so the per-cell path never touches floats.
    sand_height: i32,
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
        let seam = |min_depth: i32, rarity: u32, name: &str| Seam {
            min_depth,
            width: u32::MAX / rarity,
            block: resolve(name),
        };
        Self {
            base,
            seed,
            offset_x,
            offset_z,
            grass: resolve("Grass"),
            dirt: resolve("Dirt"),
            stone: resolve("Stone"),
            sand: resolve("Sand"),
            ice: resolve("Ice"),
            aerium_vein: resolve("AeriumVein"),
            quartz_vein: resolve("QuartzVein"),
            // Depth-tiered rarities, sorted by tier: the shallow band carries
            // fuel and workhorse metals, the deep band the exotic stuff.
            seams: [
                seam(ORE_MIN_DEPTH, 90, "CoalVein"),
                seam(8, 110, "IronVein"),
                seam(8, 130, "CopperVein"),
                seam(20, 240, "SulfurVein"),
                seam(20, 200, "QuartzVein"),
                seam(20, 220, "LeadVein"),
                seam(32, 300, "GoldVein"),
                seam(32, 380, "LuminVein"),
                seam(48, 460, "TitanVein"),
                seam(48, 240, "Obsidian"),
            ],
            sand_height: base.round() as i32 - 6,
        }
    }

    /// The ore (if any) a stone cell in the band rolls: one [`cell_hash`],
    /// mapped through the cumulative rarity slices of the tiers this depth
    /// reaches. `None` (by far the common case) keeps the cell plain stone.
    fn ore_at(&self, wx: i32, wy: i32, wz: i32, depth: i32) -> Option<BlockId> {
        let roll = cell_hash(self.seed, wx, wy, wz);
        let mut cut = 0u32;
        for seam in &self.seams {
            if depth < seam.min_depth {
                break; // sorted by tier: every later seam is deeper still
            }
            cut += seam.width;
            if roll < cut {
                return Some(seam.block);
            }
        }
        None
    }

    /// Surface-band layering for a cell below its column's surface. Lowland
    /// columns surface as sand instead of grass; stone cells inside the ore
    /// band roll one hash for a vein, and below the band stay pure stone.
    fn ground_block(&self, wx: i32, wy: i32, wz: i32, height: i32) -> BlockId {
        if wy >= height - 1 {
            if height <= self.sand_height { self.sand } else { self.grass }
        } else if wy >= height - 3 {
            self.dirt
        } else {
            let depth = height - wy;
            if depth <= ORE_MAX_DEPTH {
                if let Some(ore) = self.ore_at(wx, wy, wz, depth) {
                    return ore;
                }
            }
            self.stone
        }
    }

    /// The block for an island-solid cell, from what sits above it in the
    /// field: exposed top -> grass (Ice at [`ICE_SURFACE_Y`] and up), within
    /// 3 below a surface cell -> dirt, buried deeper -> stone — which rolls
    /// the island veins: Aerium (found nowhere else) and a little quartz.
    fn island_block(&self, wx: i32, wy: i32, wz: i32, above: [bool; 4]) -> BlockId {
        if !above[0] {
            if wy >= ICE_SURFACE_Y { self.ice } else { self.grass }
        } else if !above[1] || !above[2] || !above[3] {
            self.dirt
        } else {
            let roll = cell_hash(self.seed, wx, wy, wz);
            if roll < ISLAND_AERIUM_W {
                self.aerium_vein
            } else if roll < ISLAND_AERIUM_W + ISLAND_QUARTZ_W {
                self.quartz_vein
            } else {
                self.stone
            }
        }
    }
}

impl TerrainGenerator for SineHills {
    fn height(&self, wx: i32, wz: i32) -> i32 {
        // f64 with the sine arguments reduced modulo 2π BEFORE the sin/cos:
        // at |wx| ~1e9 the raw argument is ~8e7 radians, where f32 sin() is
        // pure rounding garbage and — worse — *platform-dependent* garbage, so
        // clients and workers could disagree on terrain. Reduction keeps the
        // argument small and the f64 result bit-stable everywhere; near the
        // origin the value matches the old f32 formula (asserted in tests).
        let x = wx as f64 + self.offset_x as f64;
        let z = wz as f64 + self.offset_z as f64;
        let tau = std::f64::consts::TAU;
        let s = |arg: f64| arg.rem_euclid(tau).sin();
        let c = |arg: f64| arg.rem_euclid(tau).cos();

        let h = self.base as f64
            + 6.0 * s(x * 0.08) * c(z * 0.08)
            + 3.0 * s(x * 0.21 + z * 0.13)
            + 2.0 * c(z * 0.30);

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
            self.ground_block(wx, wy, wz, height)
        } else if island_at(self.seed, wx, wy, wz) {
            self.island_block(wx, wy, wz, [
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
    /// - entirely below every column's ore band -> `Uniform(stone)`, no fill;
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

        // Below the shallowest column's ore band: stone forever down. The
        // proof must be exactly this conservative — a cell at depth
        // `height - wy <= ORE_MAX_DEPTH` may roll a vein, so uniform stone
        // can only be claimed when even the chunk's top cell in its
        // shallowest column sits strictly below the band.
        if y1 < h_min - ORE_MAX_DEPTH {
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
                        self.ground_block(wx, wy, wz, height)
                    } else if isl[ly] {
                        self.island_block(wx, wy, wz, [isl[ly + 1], isl[ly + 2], isl[ly + 3], isl[ly + 4]])
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
    fn height_matches_the_old_f32_formula_near_the_origin() {
        // The f64 + argument-reduction rewrite must not change the terrain
        // players have already seen: recompute the ORIGINAL f32 formula
        // inline and compare across a near-origin grid, for several seeds
        // (including seed 1, the default world's).
        //
        // Bit-identity with the old output is mathematically unattainable:
        // the old f32 pipeline rounded `x * 0.08` (x carries a seed phase up
        // to ~43k) to f32 *before* the sin, injecting position-dependent
        // argument noise of ~1e-4 rad — a ~1e-3-block wobble in h that the
        // f64 path deliberately removes. Where old-h sat within that wobble
        // of an exact .5, the round now flips by one block. Measured over
        // these four seeds: 3..16 flipped columns per 16384 (<= 0.1%), never
        // by more than 1. The bounds below pin exactly that: any real
        // regression (wrong frequency, dropped term, lost phase) shifts
        // whole regions by whole blocks and fails instantly.
        for seed in [1, 42, -777, 4242] {
            let g = hills(seed);
            let old = |wx: i32, wz: i32| -> i32 {
                let x = wx as f32 + g.offset_x;
                let z = wz as f32 + g.offset_z;
                let h = g.base
                    + 6.0 * (x * 0.08).sin() * (z * 0.08).cos()
                    + 3.0 * (x * 0.21 + z * 0.13).sin()
                    + 2.0 * (z * 0.30).cos();
                h.round().max(1.0) as i32
            };
            let mut flipped = 0usize;
            for wx in -64..64 {
                for wz in -64..64 {
                    let (new, old) = (g.height(wx, wz), old(wx, wz));
                    if new != old {
                        assert_eq!(
                            (new - old).abs(),
                            1,
                            "seed {seed}, column ({wx}, {wz}): {new} vs old {old}"
                        );
                        flipped += 1;
                    }
                }
            }
            assert!(
                flipped <= 24,
                "seed {seed}: {flipped} of 16384 columns moved — more than \
                 rounding-boundary flips can explain"
            );
        }
    }

    #[test]
    fn height_is_sane_and_deterministic_far_out() {
        // At 1e8..1e9 the reduced-argument f64 path must keep producing the
        // same bounded rolling hills (base 20 ± 11, floored at 1) instead of
        // f32 trig noise. Determinism: same inputs, same heights — cheap but
        // real, since it crosses the reduction path twice.
        let g = hills(9);
        for &wx in &[100_000_000, 999_999_000, -100_000_000] {
            for wz in -8..8 {
                let h = g.height(wx, wz * 12_345_679);
                assert!((1..=31).contains(&h), "far height {h} out of the wave envelope");
                assert_eq!(h, g.height(wx, wz * 12_345_679), "bit-stable");
            }
        }
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
        // standalone island functions — including far from the origin, where
        // the old f32 reduction disagreed with itself (and panicked past
        // y = 2^28). 268_435_453 sits just below 2^28, 268_435_488 just above
        // the old crash line; 999_999_981 rides the certified +1e9 border.
        for seed in [1, 42, -777] {
            for (wx, wz) in [(0, 0), (13, -27), (-1000, 999), (300_000_000, -299_999_777)] {
                for y_lo in [96, 268_435_453, 268_435_488, 999_999_981] {
                    let y_hi = y_lo + 19;
                    let col = IslandColumn::new(seed, wx, wz, y_lo, y_hi);
                    for y in y_lo..=y_hi {
                        // Bit-identity of the raw noise, not just the solid bool
                        // (which is almost always false at high altitude).
                        let cached = (col.o0.sample(y) + 0.5 * col.o1.sample(y)) * (1.0 / 1.5);
                        assert_eq!(
                            cached,
                            island_noise(seed, wx, y, wz),
                            "noise at ({wx}, {y}, {wz})"
                        );
                        assert_eq!(col.solid(y), island_at(seed, wx, y, wz), "at y={y}");
                    }
                }
            }
        }
    }

    #[test]
    fn far_altitude_chunks_generate_without_panicking() {
        // BUG 1 regression: at |y| >= 2^28 the f32 ULP of a world coordinate
        // is >= 32 blocks, so the old f32 plane-cache span inflated past its
        // [f32; 5] cache and indexed out of bounds. cy = 16_777_218 puts the
        // chunk floor at y = 268_435_488 — the exact crash coordinate.
        let g = hills(3);
        g.generate(0, 16_777_218, 0);

        // And a deterministic pseudo-random scan of the whole affected band,
        // cy in [2.6e8/16, 1e9/16] (~45% of layers here crashed before).
        let (lo, hi) = (260_000_000i64 / 16, 1_000_000_000i64 / 16);
        let mut state = 0x243F_6A88_85A3_08D3u64;
        for _ in 0..50 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let cy = (lo + ((state >> 16) % (hi - lo) as u64) as i64) as i32;
            g.generate(-7, cy, 11);
        }
    }

    #[test]
    fn island_noise_is_not_lattice_quantized_far_out() {
        // BUG 2 regression: with the f32 reduction, at wx = 3e8 (f32 ULP 16)
        // every world x in a 16-block run collapsed to the same lattice
        // fraction, so the noise was a staircase of at most ~6 distinct values
        // over 48 blocks. Smooth value noise lerps a fresh fade fraction every
        // block, so a 48-block line must show rich per-block variation.
        let (y, z) = (200, 123);
        for seed in [7, -31] {
            let vals: Vec<f32> = (0..48)
                .map(|i| island_noise(seed, 300_000_000 + i, y, z))
                .collect();
            let mut distinct = vals.clone();
            distinct.sort_by(f32::total_cmp);
            distinct.dedup();
            assert!(
                distinct.len() >= 24,
                "seed {seed}: only {} distinct noise values over 48 blocks — \
                 lattice-quantized",
                distinct.len()
            );
            let moving = vals.windows(2).filter(|w| w[0] != w[1]).count();
            assert!(
                moving >= 40,
                "seed {seed}: only {moving} of 47 consecutive deltas nonzero — \
                 step-quantized"
            );
        }
    }

    #[test]
    fn generated_chunks_match_per_cell_block_at() {
        // Whole-chunk generation (fast paths, column cache, collapse) must be
        // cell-identical to the naive per-cell recipe.
        let g = hills(3);
        // (0, -2, 0) sits inside the ore band, (0, -4, 0) straddles its lower
        // edge, and (2, 14, 3) reaches the frozen island altitudes.
        let coords = [
            (0, 0, 0), (0, 1, 0), (2, 4, -3), (-1, 5, 7), (0, -2, 0), (0, -4, 0), (0, 3, 0), (2, 14, 3),
        ];
        for (cx, cy, cz) in coords {
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
        // Deep rock below the ore band: provably uniform stone without a fill.
        // (Terrain heights bottom out at 9, so cy = -5 — top cell y = -65 —
        // is strictly deeper than depth 64 in every column.)
        assert_eq!(g.generate(0, -5, 0), ChunkData::Uniform(g.deep()));
        assert_eq!(g.generate(5, -100, -5), ChunkData::Uniform(g.deep()));
        // Inside the ore band the proof must NOT fire: seams make it dense.
        assert!(matches!(g.generate(0, -1, 0), ChunkData::Dense(_)));
        // Sky below the island band: provably uniform air.
        assert_eq!(g.generate(0, 3, 0), ChunkData::Uniform(AIR));
        // Ground chunks stay dense (they mix layers and air).
        assert!(matches!(g.generate(0, 1, 0), ChunkData::Dense(_)));
    }

    /// Registry + generator pair, for tests that need to resolve vein ids.
    fn hills_with_registry(seed: i64) -> (BlockRegistry, SineHills) {
        let registry = BlockRegistry::with_builtins();
        let generator = SineHills::new(&registry, 20.0, seed);
        (registry, generator)
    }

    #[test]
    fn ore_rolls_are_deterministic_and_seed_driven() {
        let (a, b, other) = (hills(42), hills(42), hills(43));
        let mut differing = 0;
        for x in -32..32 {
            for z in -32..32 {
                let h = a.height(x, z);
                let wy = h - 12; // stone cell, inside the band
                assert_eq!(a.block_at(x, wy, z, h), b.block_at(x, wy, z, h), "same seed, same veins");
                if a.block_at(x, wy, z, h) != other.block_at(x, wy, z, h) {
                    differing += 1;
                }
            }
        }
        assert!(differing > 20, "different seeds lay different veins ({differing} cells differ)");
    }

    #[test]
    fn ore_frequency_is_the_right_magnitude() {
        // Census one stone cell per column at depth 12 (the coal/iron/copper
        // tier) over a 128x128 slab: each rate must land within 2x of its
        // configured rarity — loose enough for hash noise, tight enough to
        // catch a dropped or doubled slice.
        let (reg, g) = hills_with_registry(5);
        let veins = [
            (reg.id_by_name("CoalVein").unwrap(), 90u32),
            (reg.id_by_name("IronVein").unwrap(), 110),
            (reg.id_by_name("CopperVein").unwrap(), 130),
        ];
        let mut counts = [0usize; 3];
        let mut cells = 0usize;
        for x in -64..64 {
            for z in -64..64 {
                let h = g.height(x, z);
                let block = g.block_at(x, h - 12, z, h);
                cells += 1;
                if let Some(i) = veins.iter().position(|&(id, _)| id == block) {
                    counts[i] += 1;
                }
            }
        }
        for (&(id, rarity), &count) in veins.iter().zip(&counts) {
            let expected = cells / rarity as usize;
            assert!(
                count >= expected / 2 && count <= expected * 2,
                "vein {id:?}: {count} hits, expected ~{expected} of {cells}"
            );
        }
    }

    #[test]
    fn ore_tiers_respect_their_min_depth() {
        let (reg, g) = hills_with_registry(5);
        let deep_only: Vec<BlockId> = ["SulfurVein", "QuartzVein", "LeadVein", "GoldVein", "LuminVein", "TitanVein", "Obsidian"]
            .iter()
            .map(|n| reg.id_by_name(n).unwrap())
            .collect();
        let deepest: Vec<BlockId> = ["TitanVein", "Obsidian"]
            .iter()
            .map(|n| reg.id_by_name(n).unwrap())
            .collect();
        for x in -64..64 {
            for z in -64..64 {
                let h = g.height(x, z);
                // Depth 12: only the shallow tier may appear.
                let shallow = g.block_at(x, h - 12, z, h);
                assert!(!deep_only.contains(&shallow), "deep-tier ore at depth 12: {shallow:?}");
                // Depth 40: everything but the depth-48 tier is fair game.
                let mid = g.block_at(x, h - 40, z, h);
                assert!(!deepest.contains(&mid), "depth-48 ore at depth 40: {mid:?}");
            }
        }
    }

    #[test]
    fn stone_below_the_ore_band_is_pure_and_provably_uniform() {
        let (reg, g) = hills_with_registry(7);
        let stone = reg.id_by_name("Stone").unwrap();
        for x in -48..48 {
            for z in -48..48 {
                let h = g.height(x, z);
                for depth in [ORE_MAX_DEPTH + 1, 80, 200] {
                    assert_eq!(g.block_at(x, h - depth, z, h), stone, "depth {depth} is pure stone");
                }
            }
        }
        // And a whole chunk strictly below the band still takes the no-fill
        // uniform path — the memory backbone the band must not erode.
        let chunk = Chunk::new(0, -5, 0, &g);
        assert_eq!(chunk.uniform(), Some(stone), "deep chunk stays ChunkData::Uniform");
    }

    #[test]
    fn island_stone_carries_aerium_and_quartz_veins() {
        // Islands are the only Aerium source: a slab of island-band chunks
        // must actually contain some, plus the rarer quartz sprinkle.
        let (reg, g) = hills_with_registry(3);
        let aerium = reg.id_by_name("AeriumVein").unwrap();
        let quartz = reg.id_by_name("QuartzVein").unwrap();
        let (mut aerium_cells, mut quartz_cells) = (0usize, 0usize);
        for cx in -2..2 {
            for cz in -2..2 {
                for cy in 4..8 {
                    if let ChunkData::Dense(cells) = g.generate(cx, cy, cz) {
                        for &c in cells.iter() {
                            let id = BlockId(c as u16);
                            aerium_cells += (id == aerium) as usize;
                            quartz_cells += (id == quartz) as usize;
                        }
                    }
                }
            }
        }
        assert!(aerium_cells > 0, "island slab holds AeriumVein");
        assert!(quartz_cells > 0, "island slab holds QuartzVein");
        assert!(aerium_cells > quartz_cells, "1/45 outnumbers 1/160 ({aerium_cells} vs {quartz_cells})");
    }

    #[test]
    fn island_tops_freeze_at_altitude() {
        let (reg, g) = hills_with_registry(3);
        let ice = reg.id_by_name("Ice").unwrap();
        let grass = reg.id_by_name("Grass").unwrap();
        let (mut frozen, mut grassy) = (0usize, 0usize);
        for x in -80..80 {
            for z in -80..80 {
                // An island surface cell: solid with air directly above.
                for y in [ICE_SURFACE_Y + 2, 100] {
                    if island_at(g.seed, x, y, z) && !island_at(g.seed, x, y + 1, z) {
                        let h = g.height(x, z);
                        let block = g.block_at(x, y, z, h);
                        if y >= ICE_SURFACE_Y {
                            assert_eq!(block, ice, "island top at y={y} freezes over");
                            frozen += 1;
                        } else {
                            assert_eq!(block, grass, "island top at y={y} stays grass");
                            grassy += 1;
                        }
                    }
                }
            }
        }
        assert!(frozen > 0, "found frozen island tops");
        assert!(grassy > 0, "found grassy island tops");
    }

    #[test]
    fn lowland_surfaces_are_sand_beaches() {
        let (reg, g) = hills_with_registry(3);
        let sand = reg.id_by_name("Sand").unwrap();
        let grass = reg.id_by_name("Grass").unwrap();
        let beach_line = 20 - 6; // base 20.0: columns at or below base - 6
        let (mut beaches, mut lawns) = (0usize, 0usize);
        for x in -64..64 {
            for z in -64..64 {
                let h = g.height(x, z);
                let surface = g.block_at(x, h - 1, z, h);
                if h <= beach_line {
                    assert_eq!(surface, sand, "valley column (h={h}) beaches over");
                    beaches += 1;
                } else {
                    assert_eq!(surface, grass, "higher ground (h={h}) keeps its grass");
                    lawns += 1;
                }
            }
        }
        assert!(beaches > 0 && lawns > 0, "slab spans both ({beaches} beaches, {lawns} lawns)");
    }
}
