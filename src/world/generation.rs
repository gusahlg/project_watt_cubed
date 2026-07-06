//! Terrain generation, decoupled from chunk storage so the algorithm can be
//! swapped without touching how voxels are stored or drawn.
//!
//! A generator works in [`BlockId`]s, not raw element compositions: it resolves the
//! handful of blocks it places against the [`BlockRegistry`] once, up front, so
//! filling a cell stays a cheap id copy with no per-voxel allocation.
//!
//! Three layers of terrain exist in the infinite-Y world:
//! - the **surface band**: a heightfield h(x, z) with grass/dirt layering on
//!   top and stone forever down;
//! - **caves** under that surface: a second value-noise field (its own seed
//!   stream) carves stone to air where it exceeds a threshold that FALLS with
//!   depth — hairline tunnels near the crust widening into caverns, saturating
//!   at depth 550 — and never carves shallower than [`CAVE_MIN_DEPTH`], so the
//!   grass/dirt crust stays intact;
//! - **flying islands** for `y >= 64`: a 3D value-noise field is island-solid
//!   where it exceeds a threshold that FALLS with altitude, so islands start
//!   as tiny, rare islets just above the band floor and grow into large
//!   sky-masses higher up, saturating near y = 1400.
//!
//! The surface band's stone is seasoned with single-cell **ore veins**, but
//! only within the ore band (`depth = height - wy` in `3..=ORE_MAX_DEPTH`);
//! below it stone is provably ore-free. Carving wins over ore: a carved cell
//! is air even where a vein would have landed. Island stone rolls its own
//! veins (the only Aerium source), lowland surfaces turn to sand, and island
//! tops above [`ICE_SURFACE_Y`] freeze over.
//!
//! Generation is a pure function of (seed, chunk coord) — worker threads and
//! multiplayer clients all reproduce identical chunks. Whole-chunk generation
//! proves uniformity where it can (below the ore band, a lattice-corner bound
//! on the cave field — see [`octave_sup`] — proves whole chunks solid stone
//! with no per-cell work; above the terrain and below the island band the sky
//! is provably air) and otherwise collapses an all-identical dense fill, so
//! sky and deep rock cost bytes, not kilobytes.
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
/// The solidity threshold never falls below this: from the saturation
/// altitude up, island statistics are constant ("large sky-masses").
const ISLAND_MIN_THRESHOLD: f32 = 0.56;
/// How fast the threshold falls per block of altitude. Saturation solves
/// `0.86 - dy * k = 0.56` -> `dy = 0.30 / 2.25e-4 = 4000/3 ~ 1333.3`, so the
/// clamp engages between y = 1397 (the last unclamped altitude) and y = 1398
/// — "saturates near y = 1400" with a clean constant.
const ISLAND_THRESHOLD_K: f32 = 0.000_225;

/// The noise level a cell must exceed to be island-solid at altitude `y`:
/// INVERTED growth — starts at 0.86 at the band floor (tiny, rare islets)
/// and falls linearly with altitude (growing sky-masses), clamped at
/// [`ISLAND_MIN_THRESHOLD`] from y = 1398 up.
pub fn island_threshold(y: i32) -> f32 {
    (0.86 - (y - ISLAND_MIN_Y) as f32 * ISLAND_THRESHOLD_K).max(ISLAND_MIN_THRESHOLD)
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
// Cave noise — the same 2-octave value-noise machinery as the islands, on a
// seed stream decorrelated by [`CAVE_SEED_SALT`]. Pure functions of
// (seed, position), shared by the generator and testable standalone.
// ---------------------------------------------------------------------------

/// Shallowest depth (`height - wy`) at which caves may carve stone to air.
/// The soil crust is grass (depth 1) plus dirt (depths 2..=3) — see
/// [`SineHills::ground_block_carved`] — so carving from depth 6 down can
/// never touch soil, and even leaves a two-block stone roof (depths 4..=5)
/// under the dirt.
pub const CAVE_MIN_DEPTH: i32 = 6;
/// Lattice cell size (blocks) of the cave noise's base octave; the second
/// octave runs at half this (cells 24 and 12).
const CAVE_CELL: f64 = 24.0;
const CAVE_FREQ_0: f64 = 1.0 / CAVE_CELL;
const CAVE_FREQ_1: f64 = 2.0 / CAVE_CELL;
/// XORed into the island stream's octave seeds: same world seed, distinct
/// hash stream, so cave shapes never correlate with island shapes.
const CAVE_SEED_SALT: u64 = 0xA24B_AED4_963E_E407;

/// Decorrelate the cave octaves' lattices from the islands'.
fn cave_octave_seed(seed: i64, octave: u64) -> u64 {
    octave_seed(seed, octave) ^ CAVE_SEED_SALT
}

/// The noise level a stone cell must exceed to be carved at `depth` blocks
/// below its column's surface: falls linearly from 0.80 (hairline tunnels
/// near the crust) and clamps at 0.58 once `0.80 - depth * 4e-4` reaches it
/// — at depth `0.22 / 4e-4 = 550`, below which cave statistics are constant.
pub fn cave_threshold(depth: i32) -> f32 {
    (0.80 - depth as f32 * 0.000_4).max(0.58)
}

/// The cave field at a world cell: 2-octave 3D value noise in [0, 1).
pub fn cave_noise(seed: i64, x: i32, y: i32, z: i32) -> f32 {
    let a = octave(cave_octave_seed(seed, 0), x, y, z, CAVE_FREQ_0);
    let b = octave(cave_octave_seed(seed, 1), x, y, z, CAVE_FREQ_1);
    (a + 0.5 * b) * (1.0 / 1.5)
}

/// Whether the cave field carves this cell, `depth` blocks below its
/// column's surface. Never true above [`CAVE_MIN_DEPTH`] (the depth guard
/// also short-circuits the noise evaluation for crust cells).
pub fn cave_at(seed: i64, x: i32, y: i32, z: i32, depth: i32) -> bool {
    depth >= CAVE_MIN_DEPTH && cave_noise(seed, x, y, z) > cave_threshold(depth)
}

/// The cave field down one column, cached per lattice level — bit-identical
/// to [`cave_noise`] (asserted in tests), same construction as
/// [`IslandColumn`].
struct CaveColumn {
    o0: OctaveColumn,
    o1: OctaveColumn,
}

impl CaveColumn {
    fn new(seed: i64, wx: i32, wz: i32, y_lo: i32, y_hi: i32) -> Self {
        Self {
            o0: OctaveColumn::new(cave_octave_seed(seed, 0), wx, wz, CAVE_FREQ_0, y_lo, y_hi),
            o1: OctaveColumn::new(cave_octave_seed(seed, 1), wx, wz, CAVE_FREQ_1, y_lo, y_hi),
        }
    }

    /// [`cave_at`] for this column's (x, z), y within the built range.
    fn carved(&self, y: i32, depth: i32) -> bool {
        depth >= CAVE_MIN_DEPTH
            && (self.o0.sample(y) + 0.5 * self.o1.sample(y)) * (1.0 / 1.5) > cave_threshold(depth)
    }
}

// ---------------------------------------------------------------------------
// Cave-free uniformity proof — the lattice-corner bound that lets a chunk
// fully below the ore band claim Uniform(stone) with no per-cell evaluation.
// ---------------------------------------------------------------------------

/// Slack absorbing f32 rounding skew between a per-cell noise evaluation and
/// the (mathematically not-smaller) bound: both run the same ~10-step
/// lerp/fade pipeline on values in [0, 1.5], so they diverge by at most a
/// few f32 ulps (< 1e-6). 1e-5 is an order of magnitude of headroom, at the
/// price of a vanishing number of missed proofs.
const BOUND_SLACK: f32 = 1e-5;

/// Exact supremum of one value-noise octave over a chunk's world-space box
/// `[w0, w0 + 15]` per axis, computed from the span's lattice corners.
///
/// PROOF. Within one lattice cell the octave is the multilinear (trilinear)
/// interpolation of the cell's 8 corner hashes in the faded per-axis
/// fractions. A multilinear function is affine in each coordinate with the
/// others held fixed, so over any axis-aligned box it attains its maximum at
/// a box vertex (apply the affine-endpoint argument once per axis); in
/// particular it never exceeds the cell's lattice-corner max, since every
/// vertex value is a convex combination of corner values. `fade` is monotone
/// on [0, 1], so the chunk box in raw fractions maps to a box in faded
/// fractions and the argument carries over. The chunk spans at most 3
/// lattice cells per axis (16 blocks; cells are at least 12 wide), so
/// evaluating the interpolant at the <= 8 vertices of every chunk-cell
/// sub-box — all built from the span's <= 4^3 corner hashes — yields the
/// octave's exact supremum over the chunk. Octave suprema then add:
/// `sup(a + 0.5 b) <= sup(a) + 0.5 sup(b)` (see [`cave_noise_bound`]).
fn octave_sup(seed: u64, x0: i32, y0: i32, z0: i32, freq: f64) -> f32 {
    /// One chunk axis in lattice space: first cell, cell count, and the
    /// in-cell fractions where the box enters (first cell) / ends (last).
    fn axis(w0: i32, freq: f64) -> (i64, usize, f32, f32) {
        let (c0, f0) = reduce(w0, freq);
        let (c1, f1) = reduce(w0 + CHUNK_SIZE as i32 - 1, freq);
        (c0, (c1 - c0) as usize + 1, f0, f1)
    }
    let (xc, xn, xf0, xf1) = axis(x0, freq);
    let (yc, yn, yf0, yf1) = axis(y0, freq);
    let (zc, zn, zf0, zf1) = axis(z0, freq);
    debug_assert!(xn <= 3 && yn <= 3 && zn <= 3, "16 blocks cross at most 2 cell boundaries");

    // The span's lattice corners, hashed once. [`reduce`]'s i64 cell is fed
    // to [`lattice`] as i32 exactly like [`octave`] feeds it, so the bound
    // sees the same lattice everywhere the field does.
    let mut corner = [[[0.0f32; 4]; 4]; 4];
    for (i, plane) in corner.iter_mut().enumerate().take(xn + 1) {
        for (j, row) in plane.iter_mut().enumerate().take(yn + 1) {
            for (k, c) in row.iter_mut().enumerate().take(zn + 1) {
                *c = lattice(
                    seed,
                    (xc + i as i64) as i32,
                    (yc + j as i64) as i32,
                    (zc + k as i64) as i32,
                );
            }
        }
    }

    // Fraction interval of the box within cell index `i` of `n` on an axis.
    let ends = |i: usize, n: usize, f0: f32, f1: f32| -> [f32; 2] {
        [
            if i == 0 { f0 } else { 0.0 },
            if i + 1 == n { f1 } else { 1.0 },
        ]
    };
    let mut sup = 0.0f32;
    for i in 0..xn {
        for j in 0..yn {
            for k in 0..zn {
                for tx in ends(i, xn, xf0, xf1) {
                    for ty in ends(j, yn, yf0, yf1) {
                        for tz in ends(k, zn, zf0, zf1) {
                            // The same bilinear-in-XZ-then-lerp-in-Y nesting
                            // as [`octave`]/[`plane_value`], from the cache.
                            let (fx, fy, fz) = (fade(tx), fade(ty), fade(tz));
                            let plane = |j: usize| {
                                lerp(
                                    lerp(corner[i][j][k], corner[i + 1][j][k], fx),
                                    lerp(corner[i][j][k + 1], corner[i + 1][j][k + 1], fx),
                                    fz,
                                )
                            };
                            sup = sup.max(lerp(plane(j), plane(j + 1), fy));
                        }
                    }
                }
            }
        }
    }
    sup
}

/// Upper bound on [`cave_noise`] anywhere in the chunk box whose minimum
/// world corner is `(x0, y0, z0)`: per-octave exact suprema (each bounded by
/// its lattice-corner max — see [`octave_sup`]) combined with the octave
/// weights, because suprema of sums never exceed sums of suprema.
fn cave_noise_bound(seed: i64, x0: i32, y0: i32, z0: i32) -> f32 {
    let m0 = octave_sup(cave_octave_seed(seed, 0), x0, y0, z0, CAVE_FREQ_0);
    let m1 = octave_sup(cave_octave_seed(seed, 1), x0, y0, z0, CAVE_FREQ_1);
    (m0 + 0.5 * m1) * (1.0 / 1.5)
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

    /// Surface-band layering for a cell below its column's surface, with the
    /// carve decision supplied by the caller: whole-chunk fills compute it
    /// through [`CaveColumn`]'s plane cache, [`Self::ground_block`] through
    /// the pure noise fn — bit-identical, asserted in tests. Lowland columns
    /// surface as sand instead of grass; the soil crust (surface + two dirt)
    /// sits at depths 1..=3, structurally out of `carved`'s reach — and
    /// [`cave_at`] can't be true there anyway ([`CAVE_MIN_DEPTH`] = 6).
    /// Carving is checked BEFORE the ore roll: a carved cell is air even
    /// where a vein would have landed, so ore never floats in cave voids.
    fn ground_block_carved(&self, wx: i32, wy: i32, wz: i32, height: i32, carved: bool) -> BlockId {
        if wy >= height - 1 {
            if height <= self.sand_height { self.sand } else { self.grass }
        } else if wy >= height - 3 {
            self.dirt
        } else if carved {
            AIR
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

    /// [`Self::ground_block_carved`] with the carve decision taken from the
    /// standalone cave field — the per-cell (non-column-cached) path.
    fn ground_block(&self, wx: i32, wy: i32, wz: i32, height: i32) -> BlockId {
        let carved = cave_at(self.seed, wx, wy, wz, height - wy);
        self.ground_block_carved(wx, wy, wz, height, carved)
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
    /// - entirely below every column's ore band AND cleared by the cave
    ///   corner bound -> `Uniform(stone)`, no fill;
    /// - entirely above every column's surface and below the island band
    ///   -> `Uniform(air)`, no fill;
    /// - otherwise a dense fill (island and cave noise sampled per column,
    ///   cached per lattice level), collapsed to `Uniform` when all cells
    ///   agree — which is how island-free sky chunks above y=64 and cave-free
    ///   deep chunks the bound couldn't clear still become uniform.
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

        // Below the shallowest column's ore band, ore is impossible and every
        // cell is ground — only caves can spoil Uniform(stone). The ore half
        // of the proof must be exactly this conservative: a cell at depth
        // `height - wy <= ORE_MAX_DEPTH` may roll a vein, so it applies only
        // when even the chunk's top cell in its shallowest column sits
        // strictly below the band. The cave half is the CORNER-BOUND PROOF:
        // `cave_noise_bound` >= the cave field everywhere in this chunk
        // (within a lattice cell the field is a multilinear interpolation,
        // which never exceeds the max over the box vertices built from the
        // span's corner hashes — so never the per-octave lattice-corner max —
        // and per-octave suprema add; see `octave_sup`). `cave_threshold`
        // never rises with depth, so the chunk's strictest (smallest)
        // threshold rules at its deepest cell, depth `h_max - y0`; the
        // shallow end cannot govern — a bound under the larger shallow-cell
        // threshold could still exceed the deep cells'. If the bound (plus
        // slack for f32 rounding skew) stays under that strictest threshold,
        // no cell anywhere in the chunk can carve: provably solid stone, no
        // per-cell evaluation. Chunks the bound cannot clear fall through to
        // the dense fill, where `collapse` still uniforms the cave-free ones
        // — the bound is a CPU shortcut, never a correctness gate.
        if y1 < h_min - ORE_MAX_DEPTH {
            let (x0, z0) = (cx * CHUNK_SIZE as i32, cz * CHUNK_SIZE as i32);
            if cave_noise_bound(self.seed, x0, y0, z0) + BOUND_SLACK
                < cave_threshold(h_max - y0)
            {
                return ChunkData::Uniform(self.stone);
            }
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

                // Carve mask for this column's cells: only depths >=
                // CAVE_MIN_DEPTH can carve, so columns whose carvable range
                // (wy <= height - 6) misses the chunk skip the noise pass.
                let mut carved = [false; CHUNK_SIZE];
                let cave_top = height - CAVE_MIN_DEPTH;
                if y0 <= cave_top {
                    let col = CaveColumn::new(self.seed, wx, wz, y0, y1.min(cave_top));
                    for (k, cell) in carved.iter_mut().enumerate() {
                        let wy = y0 + k as i32;
                        *cell = wy <= cave_top && col.carved(wy, height - wy);
                    }
                }

                for ly in 0..CHUNK_SIZE {
                    let wy = y0 + ly as i32;
                    let id = if wy < height {
                        self.ground_block_carved(wx, wy, wz, height, carved[ly])
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
    fn island_threshold_falls_monotonically_and_saturates() {
        assert_eq!(island_threshold(ISLAND_MIN_Y), 0.86, "tiny islets at the band floor");
        let mut prev = island_threshold(ISLAND_MIN_Y);
        for y in (ISLAND_MIN_Y + 1..3000).step_by(7) {
            let t = island_threshold(y);
            assert!(t <= prev, "threshold never rises with altitude (y={y})");
            assert!(t >= ISLAND_MIN_THRESHOLD);
            prev = t;
        }
        // k = 2.25e-4 exactly: the clamp engages at dy = 0.30 / k = 4000/3 ~
        // 1333.3 above the band floor, i.e. between y = 1397 (last unclamped
        // altitude) and y = 1398 — "saturates near y = 1400".
        assert!(island_threshold(1397) > ISLAND_MIN_THRESHOLD, "still falling at 1397");
        assert_eq!(island_threshold(1398), ISLAND_MIN_THRESHOLD, "clamped from 1398 up");
        assert_eq!(island_threshold(100_000), ISLAND_MIN_THRESHOLD);
    }

    #[test]
    fn island_density_strictly_increases_with_altitude() {
        // Inverted islands: censusing a horizontal slab at three altitudes,
        // the falling threshold must thicken the field monotonically — tiny
        // rare islets near the band floor, big sky-masses high up. (256^2:
        // near the floor solids are ~0.05%, so a small slab could read zero.)
        let census = |y: i32| -> usize {
            let mut solid = 0;
            for x in -128..128 {
                for z in -128..128 {
                    solid += island_at(9, x, y, z) as usize;
                }
            }
            solid
        };
        let (low, mid, high) = (census(80), census(400), census(1200));
        assert!(low > 0, "islets exist near the band floor");
        assert!(low < mid, "y=80 sparser than y=400 ({low} vs {mid})");
        assert!(mid < high, "y=400 sparser than y=1200 ({mid} vs {high})");
        // And none below the band floor, ever.
        assert_eq!(census(63), 0, "no islands below y=64");
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
    fn cave_noise_is_deterministic_and_decorrelated_from_islands() {
        for &(x, y, z) in &[(0, -100, 0), (-317, -900, 512), (12_345, -70, -9_876)] {
            assert_eq!(cave_noise(42, x, y, z), cave_noise(42, x, y, z));
        }
        // Driven by the seed...
        let differing = (0..64)
            .filter(|&i| cave_noise(1, i * 31, -100, -i * 17) != cave_noise(2, i * 31, -100, -i * 17))
            .count();
        assert!(differing > 48, "different seeds carve different caves");
        // ...and its own stream: CAVE_SEED_SALT decorrelates the cave field
        // from the island field on the SAME seed — they disagree everywhere
        // that two independent hashes would.
        let salted = (0..64)
            .filter(|&i| cave_noise(9, i * 31, 200, -i * 17) != island_noise(9, i * 31, 200, -i * 17))
            .count();
        assert!(salted > 48, "cave stream must not mirror the island stream");
    }

    #[test]
    fn cave_threshold_falls_with_depth_and_saturates() {
        assert_eq!(cave_threshold(0), 0.80);
        let first = cave_threshold(CAVE_MIN_DEPTH);
        assert!(first > 0.79 && first < 0.80, "0.7976 at the first carvable depth");
        let mut prev = first;
        for depth in CAVE_MIN_DEPTH + 1..1200 {
            let t = cave_threshold(depth);
            assert!(t <= prev, "never rises with depth (depth={depth})");
            assert!(t >= 0.58, "never below the cavern floor");
            prev = t;
        }
        // Saturation at depth 0.22 / 4e-4 = 550: constant from there down,
        // so deep cave statistics are depth-independent.
        assert!(cave_threshold(549) > cave_threshold(551), "still falling at 549");
        assert!((cave_threshold(550) - 0.58).abs() < 1e-6);
        assert_eq!(cave_threshold(551), 0.58, "clamped past saturation");
        assert_eq!(cave_threshold(1_000_000), 0.58);
    }

    #[test]
    fn caves_never_touch_the_soil_crust() {
        // The depth >= 6 guard means carving can never reach the soil crust:
        // grass sits at depth 1 and dirt at depths 2..=3, so even a maximal
        // cave leaves a two-block stone roof (depths 4..=5) under the dirt.
        let g = hills(5);
        let mut guard_bit = 0usize;
        for x in -64..64 {
            for z in -64..64 {
                let h = g.height(x, z);
                for depth in 1..CAVE_MIN_DEPTH {
                    let wy = h - depth;
                    assert!(!cave_at(g.seed, x, wy, z, depth), "carved above CAVE_MIN_DEPTH");
                    assert_ne!(g.block_at(x, wy, z, h), AIR, "crust cell went missing");
                    // Count cells where ONLY the depth guard blocked a carve,
                    // so the census provably exercises it.
                    guard_bit += (cave_noise(g.seed, x, wy, z) > cave_threshold(depth)) as usize;
                }
            }
        }
        assert!(guard_bit > 0, "guard never exercised — census too small");
    }

    #[test]
    fn cave_fraction_grows_with_depth_then_saturates() {
        // Census the carve fraction at fixed depths below the surface over a
        // 512x512 slab: the falling threshold must widen caves monotonically
        // down to the saturation depth (550), past which the fraction is flat
        // up to sampling noise (different y-slabs of the same field).
        let g = hills(5);
        let depths = [20, 200, 800, 1600];
        let mut counts = [0usize; 4];
        for x in -256..256 {
            for z in -256..256 {
                let h = g.height(x, z);
                for (count, &depth) in counts.iter_mut().zip(&depths) {
                    *count += cave_at(g.seed, x, h - depth, z, depth) as usize;
                }
            }
        }
        let f = counts.map(|c| c as f64 / (512.0 * 512.0));
        assert!(f[0] > 0.005, "shallow tunnels exist ({})", f[0]);
        assert!(f[0] < f[1], "depth 20 airier than 200? ({} vs {})", f[0], f[1]);
        assert!(f[1] < f[2], "depth 200 airier than 800? ({} vs {})", f[1], f[2]);
        assert!(f[2] > 4.0 * f[1], "saturated caverns dwarf shallow tunnels");
        assert!(
            (f[2] - f[3]).abs() < 0.04,
            "past saturation the fraction is flat ({} vs {})",
            f[2],
            f[3]
        );
    }

    #[test]
    fn cave_column_matches_the_pure_noise_fn() {
        // The generator's per-column cave path must be bit-identical to the
        // standalone cave functions — including far out and deep down, over
        // exactly the 16-cell spans `generate` builds.
        for seed in [1, 42, -777] {
            for (wx, wz) in [(0, 0), (13, -27), (-1000, 999), (300_000_000, -299_999_777)] {
                for y_lo in [-2000, -64, 96, 999_999_966] {
                    let y_hi = y_lo + CHUNK_SIZE as i32 - 1;
                    let col = CaveColumn::new(seed, wx, wz, y_lo, y_hi);
                    for y in y_lo..=y_hi {
                        let cached = (col.o0.sample(y) + 0.5 * col.o1.sample(y)) * (1.0 / 1.5);
                        assert_eq!(cached, cave_noise(seed, wx, y, wz), "noise at ({wx}, {y}, {wz})");
                        for depth in [CAVE_MIN_DEPTH, 100, 1000] {
                            assert_eq!(col.carved(y, depth), cave_at(seed, wx, y, wz, depth));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn deep_cave_free_chunks_prove_uniform_by_the_corner_bound() {
        // Scanning a deep layer must find chunks the corner bound clears —
        // at moderate depth (threshold ~0.77) the bound fires for roughly a
        // third of below-band chunks, at saturated depth for a few percent —
        // and every cleared chunk must come out Uniform(stone) with no fill
        // (per-cell equivalence is generated_chunks_match_per_cell_block_at's
        // job; here we pin that the proof actually fires in the wild).
        let g = hills(11);
        let (cx, cy, cz) = find_proven_chunk(&g, -6);
        assert_eq!(g.generate(cx, cy, cz), ChunkData::Uniform(g.deep()));
        let (cx, cy, cz) = find_proven_chunk(&g, -40);
        assert_eq!(g.generate(cx, cy, cz), ChunkData::Uniform(g.deep()));
    }

    #[test]
    fn deep_cavern_chunks_generate_dense_with_air() {
        // Chunk (0, -40, 0) on seed 3 holds a saturated-depth cavern (found
        // by scanning, pinned so a regression that re-hides caves fails
        // loudly). Every carved cell must be plain air below the crust and
        // agree with the standalone field; every solid cell must be pure
        // stone (no ore below the band).
        let (reg, g) = hills_with_registry(3);
        let stone = reg.id_by_name("Stone").unwrap();
        let (cx, cy, cz) = (0, -40, 0);
        let ChunkData::Dense(cells) = g.generate(cx, cy, cz) else {
            panic!("cavern chunk collapsed to uniform");
        };
        let mut air_cells = 0usize;
        for (i, &cell) in cells.iter().enumerate() {
            let (lx, ly, lz) = Chunk::local_of(i);
            let wx = cx * CHUNK_SIZE as i32 + lx as i32;
            let wy = cy * CHUNK_SIZE as i32 + ly as i32;
            let wz = cz * CHUNK_SIZE as i32 + lz as i32;
            let h = g.height(wx, wz);
            let id = BlockId(cell as u16);
            if id == AIR {
                air_cells += 1;
                assert!(h - wy >= CAVE_MIN_DEPTH, "carved into the crust at ({wx},{wy},{wz})");
                assert!(cave_at(g.seed, wx, wy, wz, h - wy), "air without a carve");
            } else {
                assert_eq!(id, stone, "only stone or cave air below the band");
            }
        }
        assert!(air_cells > 0, "the cavern is really there");
    }

    #[test]
    fn ore_never_replaces_carved_cells() {
        // Carve wins over ore: wherever the cave field carves inside the ore
        // band, the ground must yield air — never a vein floating in a void.
        let g = hills(5);
        let mut carved_in_band = 0usize;
        for x in -64..64 {
            for z in -64..64 {
                let h = g.height(x, z);
                for depth in [CAVE_MIN_DEPTH, 12, 24, 40, 60] {
                    let wy = h - depth;
                    if cave_at(g.seed, x, wy, z, depth) {
                        carved_in_band += 1;
                        assert_eq!(g.block_at(x, wy, z, h), AIR, "vein/stone in a carved cell");
                    }
                }
            }
        }
        assert!(carved_in_band > 100, "census barely met caves ({carved_in_band})");
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
        // edge, (2, 14, 3) reaches the frozen island altitudes, (0, -1, 5)
        // carries ore-band cave tunnels, and (0, -40, 0) a saturated-depth
        // cavern — both cave chunks must match the per-cell recipe exactly.
        let coords = [
            (0, 0, 0), (0, 1, 0), (2, 4, -3), (-1, 5, 7), (0, -2, 0), (0, -4, 0), (0, 3, 0), (2, 14, 3),
            (0, -1, 5), (0, -40, 0),
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

    /// Column heights of chunk `(cx, cz)`'s footprint, reduced to (min, max)
    /// — the same census `generate` runs before its fast paths.
    fn chunk_heights(g: &SineHills, cx: i32, cz: i32) -> (i32, i32) {
        let (mut h_min, mut h_max) = (i32::MAX, i32::MIN);
        for lz in 0..CHUNK_SIZE as i32 {
            for lx in 0..CHUNK_SIZE as i32 {
                let h = g.height(cx * CHUNK_SIZE as i32 + lx, cz * CHUNK_SIZE as i32 + lz);
                h_min = h_min.min(h);
                h_max = h_max.max(h);
            }
        }
        (h_min, h_max)
    }

    /// Scan +z along layer `cy` (cx = 0) for the first chunk the cave corner
    /// bound proves cave-free — the way tests pick provably-solid deep
    /// chunks now that fixed coordinates may host caves.
    fn find_proven_chunk(g: &SineHills, cy: i32) -> (i32, i32, i32) {
        let y0 = cy * CHUNK_SIZE as i32;
        let y1 = y0 + CHUNK_SIZE as i32 - 1;
        for cz in 0..128 {
            let (h_min, h_max) = chunk_heights(g, 0, cz);
            if y1 >= h_min - ORE_MAX_DEPTH {
                continue; // not fully below the ore band: proof out of scope
            }
            let z0 = cz * CHUNK_SIZE as i32;
            if cave_noise_bound(g.seed, 0, y0, z0) + BOUND_SLACK < cave_threshold(h_max - y0) {
                return (0, cy, cz);
            }
        }
        panic!("no bound-proven chunk within 128 on layer cy={cy} — bound too weak");
    }

    #[test]
    fn uniform_proofs_hold() {
        let g = hills(11);
        // Deep rock below the ore band: provably uniform stone without a
        // fill — where the cave corner bound clears the chunk (caves keep
        // the other deep chunks honest), on a shallow-ish layer and on a
        // fully saturated one (threshold floor 0.58, rarer clears).
        let (cx, cy, cz) = find_proven_chunk(&g, -6);
        assert_eq!(g.generate(cx, cy, cz), ChunkData::Uniform(g.deep()));
        let (cx, cy, cz) = find_proven_chunk(&g, -40);
        assert_eq!(g.generate(cx, cy, cz), ChunkData::Uniform(g.deep()));
        // Inside the ore band the stone proof must NOT fire: seams (and
        // shallow tunnels) make it dense.
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
        // "Pure" now means ore-free: below the band a cell is stone or a
        // carved cave void — never a vein, and air exactly where the cave
        // field says so.
        let (reg, g) = hills_with_registry(7);
        let stone = reg.id_by_name("Stone").unwrap();
        for x in -48..48 {
            for z in -48..48 {
                let h = g.height(x, z);
                for depth in [ORE_MAX_DEPTH + 1, 80, 200] {
                    let expected = if cave_at(g.seed, x, h - depth, z, depth) { AIR } else { stone };
                    assert_eq!(g.block_at(x, h - depth, z, h), expected, "depth {depth}");
                }
            }
        }
        // And a bound-proven chunk below the band still takes the no-fill
        // uniform path — the memory backbone caves must not erode.
        let (cx, cy, cz) = find_proven_chunk(&g, -10);
        let chunk = Chunk::new(cx, cy, cz, &g);
        assert_eq!(chunk.uniform(), Some(stone), "deep chunk stays ChunkData::Uniform");
    }

    #[test]
    fn island_stone_carries_aerium_and_quartz_veins() {
        // Islands are the only Aerium source: a slab of island chunks must
        // actually contain some, plus the rarer quartz sprinkle. With the
        // inverted threshold the band floor holds only skinny islets with no
        // buried interior, so census high up (y ~ 880..960) where sky-masses
        // have real stone cores.
        let (reg, g) = hills_with_registry(3);
        let aerium = reg.id_by_name("AeriumVein").unwrap();
        let quartz = reg.id_by_name("QuartzVein").unwrap();
        let (mut aerium_cells, mut quartz_cells) = (0usize, 0usize);
        for cx in -2..2 {
            for cz in -2..2 {
                for cy in 55..60 {
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
