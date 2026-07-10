//! Terrain generation, decoupled from chunk storage so the algorithm can be
//! swapped without touching how voxels are stored or drawn.
//!
//! A generator works in [`BlockId`]s, not raw element compositions: it resolves the
//! handful of blocks it places against the [`BlockRegistry`] once, up front, so
//! filling a cell stays a cheap id copy with no per-voxel allocation.
//!
//! # One primitive, spelled once
//!
//! Everything is built from a single noise type, [`Fbm`] (fractal value noise),
//! shaped by a few tiny data types:
//! - [`Ramp`] — a monotone-falling, clamped threshold `(start - slope·t).max(floor)`.
//! - [`Term`] — an [`Fbm`] compared to a [`Ramp`] along an [`Axis`] (depth or
//!   altitude), yielding a signed *excess* density (`> 0` ⟺ the field wins).
//!   Caves read it as carved, flying islands as solid — the same code, different
//!   data.
//! - [`Spline`] / [`Control`] — a Minecraft-style shaping curve over an [`Fbm`];
//!   "world flavour" (oceans, coasts, mountains) becomes const knot tables.
//!
//! The heightfield routes through [`Terrain::profile`], sampled once per [`Column`]
//! and threaded downstream, so biome dressing never re-samples noise. Water is a
//! normal translucent solid: every column floods up to its `water_level` field by
//! one `wy < water_level` rule — `sea_level` almost everywhere, raised inside lake
//! blobs — giving oceans, coastal seas, rivers, and highland lakes for free.
//! Overhang shelves, ravines, and trees layer on as further terms/decoration.
//!
//! Generation is a pure function of (seed, chunk coord) — worker threads and
//! multiplayer clients all reproduce identical chunks. Whole-chunk generation takes
//! shortcuts where it can (deep rock the cave field can't reach is `Uniform(stone)`;
//! sky above every surface is `Uniform(air)` or `Uniform(water)`) and otherwise
//! collapses an all-identical dense fill.
use super::chunk::{CHUNK_SIZE, CHUNK_VOLUME, Chunk, ChunkData};
use crate::block::registry::{AIR, BlockId, BlockRegistry};

/// Produces terrain for absolute world coordinates.
pub trait TerrainGenerator {
    /// Surface height for a world column: the number of solid layers stacked from
    /// `y = 0` upward (the first `y` that is *not* ground).
    fn height(&self, wx: i32, wz: i32) -> i32;

    /// The block on the surface of a column — biome-dependent, so it is coord-aware
    /// (LOD tiles sample it per cell to show biome colour).
    fn surface_at(&self, wx: i32, wz: i32) -> BlockId;

    /// The block placed deep underground / on far LOD side walls.
    fn deep(&self) -> BlockId;

    /// The block at a world coordinate, given the column's surface `height`.
    ///
    /// Default layering — surface block on top, deep block below, air above — is
    /// enough for the trivial test generators; [`Terrain`] overrides it with the
    /// full ground/water/island stack.
    fn block_at(&self, wx: i32, wy: i32, wz: i32, height: i32) -> BlockId {
        if wy >= height {
            AIR
        } else if wy >= height - 1 {
            self.surface_at(wx, wz)
        } else {
            self.deep()
        }
    }

    /// Generate a whole 16-cube chunk's storage. The default densely evaluates
    /// [`block_at`](Self::block_at) and collapses to [`ChunkData::Uniform`] when
    /// every cell agrees; [`Terrain`] overrides it with shortcuts.
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
                    cells[Chunk::index(lx, ly, lz)] = id.0;
                }
            }
        }
        collapse(cells)
    }
}

/// Collapse a dense fill to `Uniform` when every cell came out identical.
fn collapse(cells: Box<[u8; CHUNK_VOLUME]>) -> ChunkData {
    let first = cells[0];
    if cells.iter().all(|&c| c == first) {
        ChunkData::Uniform(BlockId(first))
    } else {
        ChunkData::Dense(cells)
    }
}

// ---------------------------------------------------------------------------
// The noise vocabulary.
// ---------------------------------------------------------------------------

/// A `[0, 1)` noise sample. A newtype so a raw field value can't be silently
/// compared against a world height or added to a coordinate.
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
pub struct Unit(pub f32);

/// The world seed. Every derived hash stream salts off it.
#[derive(Clone, Copy)]
struct Seed(i64);

/// A decorrelated hash stream. Two fields can only correlate if handed the same
/// stream — the type replaces the old manual octave-seed / salt xor-juggling.
#[derive(Clone, Copy)]
struct Stream(u64);

impl Seed {
    /// A fresh stream for `salt`; distinct salts never mirror each other.
    fn stream(self, salt: u64) -> Stream {
        Stream((self.0 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt)
    }
}

impl Stream {
    /// The per-octave lattice seed.
    fn octave(self, octave: u64) -> u64 {
        self.0 ^ octave.wrapping_mul(0xD1B5_4A32_D192_ED03)
    }
}

/// Fractal value noise: `octaves` of value noise, each at double the frequency and
/// half the weight of the last (gain ½). The one and only noise type.
#[derive(Clone)]
struct Fbm {
    stream: Stream,
    /// Lattice cell size (blocks) of the base octave.
    cell: f64,
    octaves: u8,
}

impl Fbm {
    /// Sum of octave weights `1 + ½ + ¼ + …`, the normaliser to keep output in
    /// `[0, 1)`.
    fn norm(&self) -> f32 {
        (0..self.octaves).map(|o| 0.5f32.powi(o as i32)).sum()
    }

    /// The field at a 3D world cell.
    fn at3(&self, wx: i32, wy: i32, wz: i32) -> Unit {
        let mut acc = 0.0;
        let mut w = 1.0;
        for o in 0..self.octaves {
            let f = (1u32 << o) as f64 / self.cell;
            acc += w * octave(self.stream.octave(o as u64), wx, wy, wz, f);
            w *= 0.5;
        }
        Unit(acc / self.norm())
    }

    /// Field at a 2D column (lattice level 0 in Y).
    fn at(&self, wx: i32, wz: i32) -> Unit {
        let mut acc = 0.0;
        let mut w = 1.0;
        for o in 0..self.octaves {
            let f = (1u32 << o) as f64 / self.cell;
            acc += w * octave2(self.stream.octave(o as u64), wx, wz, f);
            w *= 0.5;
        }
        Unit(acc / self.norm())
    }

    /// Field down a column with cached plane blends; bit-identical to at3 but cheaper.
    fn column(&self, wx: i32, wz: i32, y_lo: i32, y_hi: i32) -> FbmColumn {
        let cols = (0..self.octaves)
            .map(|o| {
                let f = (1u32 << o) as f64 / self.cell;
                OctaveColumn::new(self.stream.octave(o as u64), wx, wz, f, y_lo, y_hi)
            })
            .collect();
        FbmColumn { cols, norm: self.norm() }
    }

    /// A conservative interval enclosing the field everywhere in a chunk's world
    /// box (min corner `x0, y0, z0`), from each octave's trilinear corner bound.
    /// Composes by interval arithmetic; generalizes the old scalar `sup`.
    fn bound(&self, x0: i32, y0: i32, z0: i32) -> Interval {
        let (mut lo, mut hi) = (0.0, 0.0);
        let mut w = 1.0;
        for o in 0..self.octaves {
            let f = (1u32 << o) as f64 / self.cell;
            let (l, h) = octave_bound(self.stream.octave(o as u64), x0, y0, z0, f);
            lo += w * l;
            hi += w * h;
            w *= 0.5;
        }
        let n = self.norm();
        Interval { lo: lo / n, hi: hi / n }
    }

    /// An upper bound on the field anywhere in a chunk's world box — the `hi` half
    /// of [`bound`](Self::bound), kept for the carve-verdict call site.
    fn sup(&self, x0: i32, y0: i32, z0: i32) -> Unit {
        Unit(self.bound(x0, y0, z0).hi)
    }
}

/// Conservative min/max bounds over a chunk box; used for dormancy tests and
/// interval arithmetic composition.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Interval {
    lo: f32,
    hi: f32,
}

/// One [`Fbm`] sampled down a column, its octave plane blends cached.
struct FbmColumn {
    cols: Vec<OctaveColumn>,
    norm: f32,
}

impl FbmColumn {
    fn sample(&self, y: i32) -> Unit {
        let mut acc = 0.0;
        let mut w = 1.0;
        for c in &self.cols {
            acc += w * c.sample(y);
            w *= 0.5;
        }
        Unit(acc / self.norm)
    }
}

/// Falling threshold: `(start - slope·t).max(floor)`. Both caves and islands use this.
#[derive(Clone, Copy)]
struct Ramp {
    start: f32,
    slope: f32,
    floor: f32,
}

impl Ramp {
    fn at(self, t: f32) -> Unit {
        Unit((self.start - self.slope * t).max(self.floor))
    }
}

/// Axis a term's threshold ramps along: depth below surface or absolute altitude.
#[derive(Clone, Copy)]
enum Axis {
    /// Depth below the column surface (`height - wy`): caves.
    Depth,
    /// Absolute altitude (`wy`): flying islands.
    Altitude,
}

impl Axis {
    fn t(self, wy: i32, height: i32) -> i32 {
        match self {
            Axis::Depth => height - wy,
            Axis::Altitude => wy,
        }
    }
}

/// A field compared to a falling threshold, yielding signed excess: `field − threshold`.
/// Both caves (excess > 0 = carved) and islands (excess > 0 = solid) use this;
/// only the interpretation differs.
#[derive(Clone)]
struct Term {
    field: Fbm,
    ramp: Ramp,
    axis: Axis,
    /// Minimum `t` at which the term is active at all (crust roof / band floor).
    gate: i32,
    /// The `t` at which the ramp's `start` applies (ramp input is `t - origin`).
    origin: i32,
}

/// Signed excess: positive means solid/carved depending on context. Below the gate,
/// reports INACTIVE (a value no field can reach).
type Excess = f32;

impl Term {
    /// A magnitude the `[0,1)`-bounded excess can never attain, marking a cell
    /// outside the term's active band as firmly negative for `max`/`min` folds.
    const INACTIVE: Excess = -1.0e9;

    fn threshold(&self, t: i32) -> f32 {
        self.ramp.at((t - self.origin) as f32).0
    }

    /// The signed excess at a cell: `field − threshold` where active, else
    /// [`INACTIVE`](Self::INACTIVE).
    fn excess(&self, wx: i32, wy: i32, wz: i32, height: i32) -> Excess {
        let t = self.axis.t(wy, height);
        if t < self.gate {
            Self::INACTIVE
        } else {
            self.field.at3(wx, wy, wz).0 - self.threshold(t)
        }
    }

    /// [`excess`](Self::excess) off a cached column blend.
    fn excess_col(&self, col: &FbmColumn, wy: i32, height: i32) -> Excess {
        let t = self.axis.t(wy, height);
        if t < self.gate {
            Self::INACTIVE
        } else {
            col.sample(wy).0 - self.threshold(t)
        }
    }

    /// Whether the term can fire anywhere in a chunk box. Dormant when even the
    /// field's upper bound can't reach the strictest (lowest) threshold over the
    /// active `t` range — so `excess ≤ 0` everywhere and the term contributes
    /// nothing. `max_t` is the largest `t` any cell in the box reaches (deepest
    /// depth / highest altitude), where the falling ramp is easiest to exceed.
    fn dormant(&self, x0: i32, y0: i32, z0: i32, max_t: i32) -> bool {
        self.field.sup(x0, y0, z0).0 + BOUND_SLACK < self.threshold(max_t)
    }
}

/// Slack absorbing f32 rounding between a per-cell evaluation and [`Fbm::sup`].
const BOUND_SLACK: f32 = 1e-5;

/// Piecewise-linear shaping curve (knots sorted by x).
#[derive(Clone, Copy)]
struct Spline(&'static [(f32, f32)]);

impl Spline {
    fn eval(self, x: f32) -> f32 {
        let knots = self.0;
        if x <= knots[0].0 {
            return knots[0].1;
        }
        for w in knots.windows(2) {
            let ((x0, y0), (x1, y1)) = (w[0], w[1]);
            if x <= x1 {
                let t = (x - x0) / (x1 - x0);
                return y0 + (y1 - y0) * t;
            }
        }
        knots[knots.len() - 1].1
    }

    /// Min/max image of the curve over `[lo, hi]`. Scans interior knots since
    /// non-monotone curves (like RIDGE_KNOTS) don't reach extrema at endpoints alone.
    fn image(self, lo: f32, hi: f32) -> (f32, f32) {
        let a = self.eval(lo);
        let b = self.eval(hi);
        let (mut mn, mut mx) = (a.min(b), a.max(b));
        for &(x, y) in self.0 {
            if x > lo && x < hi {
                mn = mn.min(y);
                mx = mx.max(y);
            }
        }
        (mn, mx)
    }
}

/// Field shaped by a spline curve (terrain flavour control).
#[derive(Clone)]
struct Control {
    field: Fbm,
    curve: Spline,
}

impl Control {
    fn at(&self, wx: i32, wz: i32) -> f32 {
        self.curve.eval(self.field.at(wx, wz).0)
    }
}

/// Domain-warped field: read at coordinates nudged by offset fields so features
/// meander (not straight lines). Applied only to 2-D fields (biome axes) to avoid
/// invalidating term bounds and column caches.
#[derive(Clone)]
struct Warp {
    field: Fbm,
    dx: Fbm,
    dz: Fbm,
    /// Peak coordinate displacement in blocks.
    amp: f64,
}

impl Warp {
    fn at(&self, wx: i32, wz: i32) -> Unit {
        let ox = (self.dx.at(wx, wz).0 as f64 * 2.0 - 1.0) * self.amp;
        let oz = (self.dz.at(wx, wz).0 as f64 * 2.0 - 1.0) * self.amp;
        self.field.at(wx + ox.round() as i32, wz + oz.round() as i32)
    }
}

/// One column's precomputed data: 2-D fields evaluated once, then used by 3-D terms
/// with height passed as a parameter (avoiding circular dependency).
#[derive(Clone, Copy)]
struct Column {
    height: i32,
    /// The water table for this column: `sea_level` almost everywhere, raised to a
    /// lake surface inside a lake blob (S4). Cells above ground but below this are
    /// water — one rule for oceans, coastal seas, rivers, and lakes.
    water_level: i32,
    temperature: Unit,
    humidity: Unit,
}

// ---------------------------------------------------------------------------
// Value noise primitives. Uses f64 world coordinates for far-out stability.
// ---------------------------------------------------------------------------

/// Seeded hash of an integer lattice point onto a uniform value in [0, 1).
fn lattice(seed: u64, x: i32, y: i32, z: i32) -> f32 {
    let h = seed
        ^ (x as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u32 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ (z as u32 as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
    let h = crate::hash::splitmix_finish(h);
    (h >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Smoothstep fade, the classic value-noise interpolant.
fn fade(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Reduce a world coordinate to its lattice cell and in-cell fraction, in f64
/// (f32 loses precision far from the origin and eventually panics).
fn reduce(w: i32, freq: f64) -> (i64, f32) {
    let t = w as f64 * freq;
    let cell = t.floor() as i64;
    (cell, (t - cell as f64) as f32)
}

/// Bilinear lattice blend in the XZ plane at integer lattice level `ly`.
fn plane_value(seed: u64, xi: i32, fx: f32, ly: i32, zi: i32, fz: f32) -> f32 {
    let (tx, tz) = (fade(fx), fade(fz));
    let v00 = lattice(seed, xi, ly, zi);
    let v10 = lattice(seed, xi + 1, ly, zi);
    let v01 = lattice(seed, xi, ly, zi + 1);
    let v11 = lattice(seed, xi + 1, ly, zi + 1);
    lerp(lerp(v00, v10, tx), lerp(v01, v11, tx), tz)
}

/// One 3D value-noise octave at a world cell.
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

/// One 2D value-noise octave at a world column (lattice level 0 in Y).
fn octave2(seed: u64, wx: i32, wz: i32, freq: f64) -> f32 {
    let (xi, fx) = reduce(wx, freq);
    let (zi, fz) = reduce(wz, freq);
    plane_value(seed, xi as i32, fx, 0, zi as i32, fz)
}

/// One octave sampled down a column; plane blends cached per level (hot path).
struct OctaveColumn {
    freq: f64,
    base: i64,
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

    fn sample(&self, y: i32) -> f32 {
        let (cell, fy) = reduce(y, self.freq);
        let ty = fade(fy);
        let i = (cell - self.base) as usize;
        lerp(self.planes[i], self.planes[i + 1], ty)
    }
}

/// Min/max bounds for a value-noise octave over a chunk box, from lattice corners.
fn octave_bound(seed: u64, x0: i32, y0: i32, z0: i32, freq: f64) -> (f32, f32) {
    fn axis(w0: i32, freq: f64) -> (i64, usize, f32, f32) {
        let (c0, f0) = reduce(w0, freq);
        let (c1, f1) = reduce(w0 + CHUNK_SIZE as i32 - 1, freq);
        (c0, (c1 - c0) as usize + 1, f0, f1)
    }
    let (xc, xn, xf0, xf1) = axis(x0, freq);
    let (yc, yn, yf0, yf1) = axis(y0, freq);
    let (zc, zn, zf0, zf1) = axis(z0, freq);
    debug_assert!(xn <= 3 && yn <= 3 && zn <= 3, "16 blocks cross at most 2 cell boundaries");

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

    let ends = |i: usize, n: usize, f0: f32, f1: f32| -> [f32; 2] {
        [if i == 0 { f0 } else { 0.0 }, if i + 1 == n { f1 } else { 1.0 }]
    };
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for i in 0..xn {
        for j in 0..yn {
            for k in 0..zn {
                for tx in ends(i, xn, xf0, xf1) {
                    for ty in ends(j, yn, yf0, yf1) {
                        for tz in ends(k, zn, zf0, zf1) {
                            let (fx, fy, fz) = (fade(tx), fade(ty), fade(tz));
                            let plane = |j: usize| {
                                lerp(
                                    lerp(corner[i][j][k], corner[i + 1][j][k], fx),
                                    lerp(corner[i][j][k + 1], corner[i + 1][j][k + 1], fx),
                                    fz,
                                )
                            };
                            let v = lerp(plane(j), plane(j + 1), fy);
                            lo = lo.min(v);
                            hi = hi.max(v);
                        }
                    }
                }
            }
        }
    }
    (lo, hi)
}

/// Seeded hash of a world cell onto a uniform `u32` — the single roll an ore
/// candidate cell makes.
fn cell_hash(seed: i64, x: i32, y: i32, z: i32) -> u32 {
    let h = (seed as u64 ^ 0x517C_C1B7_2722_0A95)
        ^ (x as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u32 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ (z as u32 as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
    let h = crate::hash::splitmix_finish(h);
    (h >> 32) as u32
}

// ---------------------------------------------------------------------------
// Tuning surface — the one place terrain flavour lives, as const data.
// ---------------------------------------------------------------------------

/// Flying islands exist only at or above this altitude (well above sea level, so
/// water and islands never interact).
pub const ISLAND_MIN_Y: i32 = 64;
/// Shallowest depth at which caves may carve, leaving the soil crust intact.
const CAVE_MIN_DEPTH: i32 = 6;
/// Shallowest depth at which a ravine may bite (S6) — deeper than a cave roof so
/// ravine slots read as gashes below the crust, not surface cracks.
const RAVINE_MIN_DEPTH: i32 = 8;
/// How many blocks above a column's surface an overhang shelf may reach (S6).
const OVERHANG_REACH: i32 = 8;
/// Overhang field value needed at the surface, and how much stricter it gets per
/// block of height above it — so shelves fade out with altitude above the ground.
const OVERHANG_THRESH: f32 = 0.60;
const OVERHANG_FADE: f32 = 0.05;
/// Shallowest / deepest depth at which ground ore can appear.
const ORE_MIN_DEPTH: i32 = 3;
const ORE_MAX_DEPTH: i32 = 64;
/// Island surface cells at or above this altitude freeze to Ice.
const ICE_SURFACE_Y: i32 = 220;

// Hash-stream salts — distinct constants, so no two fields share a lattice.
const CONT_SALT: u64 = 0x0001;
const EROSION_SALT: u64 = 0x0002;
const WEIRD_SALT: u64 = 0x0003;
const TEMP_SALT: u64 = 0x0004;
const HUMID_SALT: u64 = 0x0005;
const DETAIL_SALT: u64 = 0x0006;
const LAKE_SALT: u64 = 0x0007;
const WARPX_SALT: u64 = 0x0008;
const WARPZ_SALT: u64 = 0x0009;
const RANGE_SALT: u64 = 0x000A;
const CAVE_SALT: u64 = 0xA24B_AED4_963E_E407;
const ISLAND_SALT: u64 = 0x1D3E_66F0_9C2A_B517;
const RAVINE_SALT: u64 = 0x77C1_9B0A_5E3D_2F81;
const OVERHANG_SALT: u64 = 0x2B9F_10E6_A4C7_5D33;
/// Salt for the tree scatter roll (S7 decoration), kept off the ore `cell_hash`
/// stream so tree placement and ore rolls never correlate.
const TREE_SALT: i64 = 0x51ED_2C97_7A3B_10F5u64 as i64;

/// Continentalness → base height offset from sea level: deep ocean floors, coastal
/// shelves, inland plains, and high interiors.
const CONT_KNOTS: &[(f32, f32)] = &[
    (0.00, -30.0),
    (0.35, -12.0),
    (0.48, -3.0),
    (0.55, 2.0),
    (0.68, 10.0),
    (0.82, 26.0),
    (1.00, 52.0),
];
/// Erosion → relief amplitude: flat plains at low erosion, jagged mountains high.
/// A late, steep ramp so genuine mountains are common wherever erosion peaks
/// rather than a rare tail.
const EROSION_KNOTS: &[(f32, f32)] = &[
    (0.00, 2.0),
    (0.30, 5.0),
    (0.52, 16.0),
    (0.72, 42.0),
    (0.88, 74.0),
    (1.00, 96.0),
];
/// Weirdness → signed ridge factor: peaks flanking a valley floor at the mid band,
/// where rivers run.
const RIDGE_KNOTS: &[(f32, f32)] = &[
    (0.00, 0.9),
    (0.15, 1.0),
    (0.35, 0.2),
    (0.50, -0.7),
    (0.65, 0.2),
    (0.85, 1.0),
    (1.00, 0.9),
];

/// Half-width (in weirdness units) of the river channel band around the valley
/// floor (weirdness 0.5).
const RIVER_HALF: f32 = 0.06;
/// How far below sea level a river carves its channel.
const RIVER_DEPTH: i32 = 4;
/// A column is "inland" (river-eligible) when its base height sits this far above
/// sea level, so ocean basins aren't carved twice.
const RIVER_INLAND: f32 = 3.0;

/// Lake noise above this fires an inland lake blob; the smoothstep from here to 1
/// carves the bowl and raises the water table, so lakes sit in scattered patches.
const LAKE_ON: f32 = 0.70;
/// A lake's water surface, in blocks above sea level — the raised local water
/// table `water_level` reports inside a lake, giving standing water above the sea.
const LAKE_RISE: i32 = 7;
/// How far below its water surface a lake's basin floor is pulled, so terrain
/// inside the blob dips beneath the raised table and actually floods.
const LAKE_BOWL: i32 = 6;

/// Continentalness (base height above sea) past which ridged mountain ranges kick
/// in, so ranges sharpen genuine highlands and never lift ocean floors.
const RANGE_ONSET: f32 = 14.0;
/// Peak extra height a ridgeline adds atop an elevated column, scaled by both how
/// far past the onset the base sits and the erosion amplitude.
const RANGE_GAIN: f32 = 0.55;

/// Snow line: dressed columns this far above sea level freeze over, so only real
/// peaks cap with snow while mid-height slopes keep grass and bare stone.
const SNOW_ABOVE_SEA: i32 = 55;
/// Biome dressing cutoffs on temperature / humidity units.
const COLD: f32 = 0.30;
const HOT: f32 = 0.72;
const DRY: f32 = 0.32;

/// Trees (S7 decoration): how densely they scatter (1 in N eligible grass
/// columns), how tall the trunk is, and the canopy's horizontal reach — which is
/// also the neighbour radius a chunk must scan so a tree's leaves cross into it.
const TREE_RARITY: u32 = 140;
const TREE_TRUNK: i32 = 5;
const LEAF_R: i32 = 2;
/// The topmost cell above a tree's base that its canopy can occupy — bounds the
/// vertical band a chunk must include for trees to be possible.
const TREE_TOP: i32 = TREE_TRUNK + 1;

/// Island ore odds — flying islands are the only natural Aerium source.
const ISLAND_AERIUM_W: u32 = u32::MAX / 45;
const ISLAND_QUARTZ_W: u32 = u32::MAX / 160;

/// One ground ore tier: eligible from `min_depth` down, hit when the cell's hash
/// lands in a cumulative slice `width` wide.
#[derive(Clone, Copy)]
struct Seam {
    min_depth: i32,
    width: u32,
    block: BlockId,
}

// ---------------------------------------------------------------------------
// Terrain — the game's generator. `SineHills` kept as an alias so existing call
// sites need no change.
// ---------------------------------------------------------------------------

/// Natural terrain — oceans, coasts, mountains, plains, rivers, and biomes —
/// expressed as data over the noise vocabulary, plus the flying-island and cave
/// 3-D features. Deterministic and cheap to clone (a handful of numbers and ids),
/// since generation jobs run on worker threads.
#[derive(Clone)]
pub struct Terrain {
    /// The world seed (saved and restored verbatim).
    pub seed: i64,
    sea_level: i32,

    continentalness: Control,
    erosion: Control,
    weirdness: Control,
    /// Mid-frequency rolling detail added to every column, so even plains vary
    /// and mountains gain rough flanks. Scaled up by erosion.
    detail: Fbm,
    /// Low-frequency lake mask: scattered inland blobs raise the water table and
    /// carve a bowl (S4), so `water_level` is a field, not the flat sea constant.
    lakes: Fbm,
    /// Ridged mountain-range noise (S5): a folded field that raises sharp crests
    /// along ridgelines where the terrain is already elevated.
    ranges: Fbm,
    /// Biome axes are domain-warped (S5) so temperature/humidity — and thus the
    /// biome borders they dress — meander instead of sitting in round blobs.
    temperature: Warp,
    humidity: Warp,
    caves: Term,
    /// A second carve term (S6): narrow deep ravines slicing the ground, folded in
    /// with the caves as another `excess > 0` subtraction.
    ravines: Term,
    /// 3-D overhang fill (S6): solid rock placed *above* the heightfield surface
    /// in a fading band, so cliffs grow shelves the pure heightfield can't express.
    overhangs: Fbm,
    islands: Term,

    grass: BlockId,
    dirt: BlockId,
    stone: BlockId,
    sand: BlockId,
    snow: BlockId,
    ice: BlockId,
    water: BlockId,
    wood: BlockId,
    leaves: BlockId,
    aerium_vein: BlockId,
    quartz_vein: BlockId,
    /// Ground ore table, sorted by `min_depth`.
    seams: [Seam; 10],
}

/// Kept for compatibility with existing call sites.
pub type SineHills = Terrain;

impl Terrain {
    /// Build the generator for a seed, resolving its palette against the registry.
    /// `base` sets sea level. Panics if a built-in block is missing.
    pub fn new(registry: &BlockRegistry, base: f32, seed: i64) -> Self {
        let resolve = |name: &str| {
            registry
                .id_by_name(name)
                .unwrap_or_else(|| panic!("Terrain needs the built-in '{name}' block"))
        };
        let s = Seed(seed);
        let fbm = |salt: u64, cell: f64, octaves: u8| Fbm { stream: s.stream(salt), cell, octaves };
        let seam = |min_depth: i32, rarity: u32, name: &str| Seam {
            min_depth,
            width: u32::MAX / rarity,
            block: resolve(name),
        };
        Self {
            seed,
            sea_level: base.round() as i32,
            continentalness: Control { field: fbm(CONT_SALT, 280.0, 3), curve: Spline(CONT_KNOTS) },
            erosion: Control { field: fbm(EROSION_SALT, 200.0, 2), curve: Spline(EROSION_KNOTS) },
            weirdness: Control { field: fbm(WEIRD_SALT, 72.0, 3), curve: Spline(RIDGE_KNOTS) },
            detail: fbm(DETAIL_SALT, 64.0, 3),
            lakes: fbm(LAKE_SALT, 220.0, 2),
            ranges: fbm(RANGE_SALT, 150.0, 3),
            temperature: Warp {
                field: fbm(TEMP_SALT, 480.0, 2),
                dx: fbm(WARPX_SALT, 300.0, 2),
                dz: fbm(WARPZ_SALT, 300.0, 2),
                amp: 48.0,
            },
            humidity: Warp {
                field: fbm(HUMID_SALT, 520.0, 2),
                dx: fbm(WARPX_SALT, 300.0, 2),
                dz: fbm(WARPZ_SALT, 300.0, 2),
                amp: 48.0,
            },
            caves: Term {
                field: fbm(CAVE_SALT, 24.0, 2),
                ramp: Ramp { start: 0.80, slope: 0.000_4, floor: 0.58 },
                axis: Axis::Depth,
                gate: CAVE_MIN_DEPTH,
                origin: 0,
            },
            ravines: Term {
                field: fbm(RAVINE_SALT, 30.0, 2),
                ramp: Ramp { start: 0.90, slope: 0.000_2, floor: 0.80 },
                axis: Axis::Depth,
                gate: RAVINE_MIN_DEPTH,
                origin: 0,
            },
            overhangs: fbm(OVERHANG_SALT, 20.0, 2),
            islands: Term {
                field: fbm(ISLAND_SALT, 24.0, 2),
                ramp: Ramp { start: 0.86, slope: 0.000_225, floor: 0.56 },
                axis: Axis::Altitude,
                gate: ISLAND_MIN_Y,
                origin: ISLAND_MIN_Y,
            },
            grass: resolve("Grass"),
            dirt: resolve("Dirt"),
            stone: resolve("Stone"),
            sand: resolve("Sand"),
            snow: resolve("Snow"),
            ice: resolve("Ice"),
            water: resolve("Water"),
            wood: resolve("Wood"),
            leaves: resolve("Leaves"),
            aerium_vein: resolve("AeriumVein"),
            quartz_vein: resolve("QuartzVein"),
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
        }
    }

    /// Sea level — spawn logic keeps players off the seabed.
    pub fn sea_level(&self) -> i32 {
        self.sea_level
    }

    /// Everything a column needs, sampled once. `height` folds continentalness
    /// (base), erosion·ridge (relief), and rivers (valley-floor channels).
    fn profile(&self, wx: i32, wz: i32) -> Column {
        let base = self.continentalness.at(wx, wz);
        let amp = self.erosion.at(wx, wz);
        let ridge = self.weirdness.at(wx, wz);
        // Mid-frequency rolling detail on every column: a small baseline so plains
        // are never dead flat, growing with erosion so mountains get rough flanks.
        let detail = self.detail.at(wx, wz).0 * 2.0 - 1.0;
        let mut h = self.sea_level as f32 + base + amp * ridge + (2.5 + amp * 0.25) * detail;

        // Ridged mountain ranges (S5): fold the range noise to a ridgeline crest
        // (`1 − |2n − 1|` peaks at n = ½) and add it only atop already-elevated
        // columns, scaled by how far past the onset the base sits — so ranges
        // sharpen highlands into ridges without touching plains or seas.
        if base > RANGE_ONSET {
            let crest = 1.0 - (self.ranges.at(wx, wz).0 * 2.0 - 1.0).abs();
            h += RANGE_GAIN * (base - RANGE_ONSET).min(amp) * crest;
        }

        // Rivers: carve toward a sub-sea channel at the valley floor (weirdness
        // 0.5), gated to inland columns so ocean basins aren't double-carved.
        let w = self.weirdness.field.at(wx, wz).0;
        let d = (w - 0.5).abs();
        if base > RIVER_INLAND && d < RIVER_HALF {
            let t = 1.0 - d / RIVER_HALF;
            let s = t * t * (3.0 - 2.0 * t);
            let target = (self.sea_level - RIVER_DEPTH) as f32;
            h += (target.min(h) - h) * s;
        }

        // Lakes (S4): inland lake blobs raise the water table to a surface above
        // sea level and pull the terrain into a bowl beneath it, so the raised
        // table actually floods. Containment is automatic — water only appears
        // where ground sits below `water_level`, which the bowl guarantees at the
        // blob centre while the untouched rim stays dry.
        let mut water_level = self.sea_level;
        let lake = self.lakes.at(wx, wz).0;
        if base > RIVER_INLAND && lake > LAKE_ON {
            let t = (lake - LAKE_ON) / (1.0 - LAKE_ON);
            let s = t * t * (3.0 - 2.0 * t);
            let floor = (self.sea_level + LAKE_RISE - LAKE_BOWL) as f32;
            h += (floor.min(h) - h) * s;
            water_level = self.sea_level + LAKE_RISE;
        }

        Column {
            height: (h.round() as i32).max(1),
            water_level,
            temperature: self.temperature.at(wx, wz),
            humidity: self.humidity.at(wx, wz),
        }
    }

    /// The surface block a column dresses in: sand at/under sea (shore, lakebed),
    /// snow on cold or high ground, sand on hot & dry, grass otherwise.
    fn dress(&self, p: &Column) -> BlockId {
        if p.height <= p.water_level {
            self.sand
        } else if p.temperature.0 < COLD || p.height - self.sea_level > SNOW_ABOVE_SEA {
            self.snow
        } else if p.temperature.0 > HOT && p.humidity.0 < DRY {
            self.sand
        } else {
            self.grass
        }
    }

    /// The ore (if any) a stone cell rolls, through the cumulative rarity slices.
    fn ore_at(&self, wx: i32, wy: i32, wz: i32, depth: i32) -> Option<BlockId> {
        let roll = cell_hash(self.seed, wx, wy, wz);
        let mut cut = 0u32;
        for seam in &self.seams {
            if depth < seam.min_depth {
                break;
            }
            cut += seam.width;
            if roll < cut {
                return Some(seam.block);
            }
        }
        None
    }

    /// A ground cell below its column's surface, carve decision supplied.
    fn ground(&self, p: &Column, wx: i32, wy: i32, wz: i32, carved: bool) -> BlockId {
        let height = p.height;
        if wy >= height - 1 {
            self.dress(p)
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

    /// The block for an island-solid cell, from what sits above it in the field.
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

    /// Whether a ground cell is carved out — by a cave or a ravine (S6), the two
    /// carve terms folded together as one `excess > 0` subtraction.
    fn carved(&self, wx: i32, wy: i32, wz: i32, height: i32) -> bool {
        self.caves.excess(wx, wy, wz, height) > 0.0 || self.ravines.excess(wx, wy, wz, height) > 0.0
    }

    /// Whether an overhang shelf places solid rock at a cell above the surface
    /// (S6): within [`OVERHANG_REACH`] blocks of the surface and past a threshold
    /// that stiffens with height, so shelves jut from cliffs and fade upward.
    fn overhang_solid(&self, wx: i32, wy: i32, wz: i32, height: i32) -> bool {
        let up = wy - height;
        (0..OVERHANG_REACH).contains(&up)
            && self.overhangs.at3(wx, wy, wz).0 > OVERHANG_THRESH + OVERHANG_FADE * up as f32
    }

    /// Whether a tree grows on the column at `(ox, oz)`, and if so its base `y`
    /// (the first cell above the surface). Trees scatter (S7) on dry grassy
    /// columns via a sparse `cell_hash` roll — the decoration analogue of the ore
    /// `Seam` scatter, but with a multi-cell payload ([`tree_voxel`]).
    fn tree_at(&self, ox: i32, oz: i32) -> Option<i32> {
        let p = self.profile(ox, oz);
        if p.height <= p.water_level || self.dress(&p) != self.grass {
            return None;
        }
        (cell_hash(self.seed ^ TREE_SALT, ox, 0, oz) < u32::MAX / TREE_RARITY).then_some(p.height)
    }

    /// The block a tree whose base sits at the origin places at the offset
    /// `(dx, dy, dz)` from that base — a trunk column crowned by a leaf blob — or
    /// `None` where the tree has no voxel.
    fn tree_voxel(&self, dx: i32, dy: i32, dz: i32) -> Option<BlockId> {
        if dx == 0 && dz == 0 && (0..TREE_TRUNK).contains(&dy) {
            return Some(self.wood);
        }
        let top = TREE_TRUNK - 1;
        if (top..=TREE_TOP).contains(&dy) {
            let rad = if dy == TREE_TOP { 1 } else { LEAF_R };
            if dx * dx + dz * dz <= rad * rad {
                return Some(self.leaves);
            }
        }
        None
    }

    /// The decoration block at a cell, if any tree in the surrounding
    /// [`LEAF_R`]-column neighbourhood reaches it. Scanned in a fixed order (z
    /// outer, x inner) so overlapping canopies resolve identically here and in the
    /// chunk fast path.
    fn feature_at(&self, wx: i32, wy: i32, wz: i32) -> Option<BlockId> {
        for oz in (wz - LEAF_R)..=(wz + LEAF_R) {
            for ox in (wx - LEAF_R)..=(wx + LEAF_R) {
                if let Some(base) = self.tree_at(ox, oz) {
                    if let Some(b) = self.tree_voxel(wx - ox, wy - base, wz - oz) {
                        return Some(b);
                    }
                }
            }
        }
        None
    }

    /// The full vertical stack at a cell (per-cell path): ground below the
    /// surface, an overhang shelf or water above it, island-or-air higher still,
    /// then tree decoration overlaid into any air cell it reaches.
    fn cell(&self, p: &Column, wx: i32, wy: i32, wz: i32) -> BlockId {
        let base = if wy < p.height {
            self.ground(p, wx, wy, wz, self.carved(wx, wy, wz, p.height))
        } else if wy < p.water_level {
            self.water
        } else if self.overhang_solid(wx, wy, wz, p.height) {
            self.stone
        } else if self.islands.excess(wx, wy, wz, p.height) > 0.0 {
            self.island_block(wx, wy, wz, [
                self.islands.excess(wx, wy + 1, wz, p.height) > 0.0,
                self.islands.excess(wx, wy + 2, wz, p.height) > 0.0,
                self.islands.excess(wx, wy + 3, wz, p.height) > 0.0,
                self.islands.excess(wx, wy + 4, wz, p.height) > 0.0,
            ])
        } else {
            AIR
        };
        if base == AIR {
            self.feature_at(wx, wy, wz).unwrap_or(AIR)
        } else {
            base
        }
    }
}

impl TerrainGenerator for Terrain {
    /// The LOD/spawn surface height — the topmost *ground* cell's column value.
    /// Caves never carve the top [`CAVE_MIN_DEPTH`] cells, so the topmost ground
    /// cell is always `height − 1`. Overhang shelves (S6) sit in the air *above*
    /// this and trees decorate above it, so neither moves the walkable ground
    /// surface LOD and spawn key off — that stays the one column `height`.
    fn height(&self, wx: i32, wz: i32) -> i32 {
        self.profile(wx, wz).height
    }

    fn surface_at(&self, wx: i32, wz: i32) -> BlockId {
        self.dress(&self.profile(wx, wz))
    }

    fn deep(&self) -> BlockId {
        self.stone
    }

    fn block_at(&self, wx: i32, wy: i32, wz: i32, _height: i32) -> BlockId {
        self.cell(&self.profile(wx, wz), wx, wy, wz)
    }

    /// Whole-chunk generation, shortcuts folded from height band + region verdicts:
    /// - deep rock below the ore band the caves can't reach → `Uniform(stone)`;
    /// - above every surface and below the island band → `Uniform(water)` (below
    ///   sea) or `Uniform(air)`;
    /// - otherwise a dense fill (columns cached), collapsed when uniform.
    fn generate(&self, cx: i32, cy: i32, cz: i32) -> ChunkData {
        let (x0, z0) = (cx * CHUNK_SIZE as i32, cz * CHUNK_SIZE as i32);
        let y0 = cy * CHUNK_SIZE as i32;
        let y1 = y0 + CHUNK_SIZE as i32 - 1;

        // One profile per column, reused by the fast paths and the fill.
        let mut profiles: Vec<Column> = Vec::with_capacity(CHUNK_SIZE * CHUNK_SIZE);
        let (mut h_min, mut h_max) = (i32::MAX, i32::MIN);
        let (mut w_min, mut w_max) = (i32::MAX, i32::MIN);
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let p = self.profile(x0 + lx as i32, z0 + lz as i32);
                h_min = h_min.min(p.height);
                h_max = h_max.max(p.height);
                w_min = w_min.min(p.water_level);
                w_max = w_max.max(p.water_level);
                profiles.push(p);
            }
        }

        // Deep below every ore band and beyond either carve field's reach: solid
        // stone. Both caves and ravines must be dormant for the chunk to be safe.
        if y1 < h_min - ORE_MAX_DEPTH
            && self.caves.dormant(x0, y0, z0, h_max - y0)
            && self.ravines.dormant(x0, y0, z0, h_max - y0)
        {
            return ChunkData::Uniform(self.stone);
        }
        // Above every surface and below the island band: uniform sky. Fully below
        // the lowest water table → water; fully at/above the highest → air. (A
        // chunk straddling a water table falls through to the dense fill, which
        // collapses it anyway.)
        // Guarded past the overhang reach above the tallest surface, since a shelf
        // can place solid rock up to `OVERHANG_REACH` blocks over the ground.
        if y0 >= h_max + OVERHANG_REACH && y1 < ISLAND_MIN_Y {
            if y1 < w_min {
                return ChunkData::Uniform(self.water);
            }
            if y0 >= w_max {
                return ChunkData::Uniform(AIR);
            }
        }

        // Tree origins whose canopy can reach this chunk (S7): collected once from
        // the LEAF_R-block margin so leaves that spill across the chunk border are
        // stamped, then overlaid into air cells in the fill. Only when the chunk's
        // y-span can hold tree cells at all.
        let treeband = y1 >= h_min && y0 <= h_max + TREE_TOP;
        let mut trees: Vec<(i32, i32, i32)> = Vec::new();
        if treeband {
            for oz in (z0 - LEAF_R)..=(z0 + CHUNK_SIZE as i32 - 1 + LEAF_R) {
                for ox in (x0 - LEAF_R)..=(x0 + CHUNK_SIZE as i32 - 1 + LEAF_R) {
                    if let Some(base) = self.tree_at(ox, oz) {
                        trees.push((ox, base, oz));
                    }
                }
            }
        }

        let mut cells = Box::new([0u8; CHUNK_VOLUME]);
        let islands_possible = y1 >= ISLAND_MIN_Y;
        let mut isl = [false; CHUNK_SIZE + 4];
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let wx = x0 + lx as i32;
                let wz = z0 + lz as i32;
                let p = &profiles[lx + lz * CHUNK_SIZE];
                let height = p.height;

                if islands_possible {
                    let col = self.islands.field.column(wx, wz, y0, y1 + 4);
                    for (k, cell) in isl.iter_mut().enumerate() {
                        *cell = self.islands.excess_col(&col, y0 + k as i32, height) > 0.0;
                    }
                }

                // Carve mask: caves (depth ≥ CAVE_MIN_DEPTH) and ravines (depth ≥
                // RAVINE_MIN_DEPTH) both bite here, folded together. `cave_top` is
                // the shallower roof and bounds the cached column span for both.
                let mut carved = [false; CHUNK_SIZE];
                let cave_top = height - CAVE_MIN_DEPTH;
                let rav_top = height - RAVINE_MIN_DEPTH;
                if y0 <= cave_top {
                    let cave_col = self.caves.field.column(wx, wz, y0, y1.min(cave_top));
                    let rav_col = self.ravines.field.column(wx, wz, y0, y1.min(cave_top));
                    for (k, cell) in carved.iter_mut().enumerate() {
                        let wy = y0 + k as i32;
                        let cave = wy <= cave_top && self.caves.excess_col(&cave_col, wy, height) > 0.0;
                        let rav = wy <= rav_top && self.ravines.excess_col(&rav_col, wy, height) > 0.0;
                        *cell = cave || rav;
                    }
                }

                for ly in 0..CHUNK_SIZE {
                    let wy = y0 + ly as i32;
                    let mut id = if wy < height {
                        self.ground(p, wx, wy, wz, carved[ly])
                    } else if wy < p.water_level {
                        self.water
                    } else if self.overhang_solid(wx, wy, wz, height) {
                        self.stone
                    } else if isl[ly] {
                        self.island_block(wx, wy, wz, [isl[ly + 1], isl[ly + 2], isl[ly + 3], isl[ly + 4]])
                    } else {
                        AIR
                    };
                    if id == AIR {
                        for &(ox, base, oz) in &trees {
                            if (wx - ox).abs() <= LEAF_R && (wz - oz).abs() <= LEAF_R {
                                if let Some(b) = self.tree_voxel(wx - ox, wy - base, wz - oz) {
                                    id = b;
                                    break;
                                }
                            }
                        }
                    }
                    cells[Chunk::index(lx, ly, lz)] = id.0;
                }
            }
        }
        collapse(cells)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terrain(seed: i64) -> Terrain {
        Terrain::new(&BlockRegistry::with_builtins(), 20.0, seed)
    }

    fn terrain_with_registry(seed: i64) -> (BlockRegistry, Terrain) {
        let registry = BlockRegistry::with_builtins();
        let generator = Terrain::new(&registry, 20.0, seed);
        (registry, generator)
    }

    #[test]
    fn column_cache_matches_the_pure_field() {
        // The generalized parity check: Fbm::column must be bit-identical to
        // Fbm::at3 for both region fields, including far out and deep down.
        let g = terrain(42);
        for field in [&g.caves.field, &g.islands.field] {
            for (wx, wz) in [(0, 0), (13, -27), (-1000, 999), (300_000_000, -299_999_777)] {
                for y_lo in [-2000, -64, 96, 999_999_966] {
                    let y_hi = y_lo + CHUNK_SIZE as i32 - 1;
                    let col = field.column(wx, wz, y_lo, y_hi);
                    for y in y_lo..=y_hi {
                        assert_eq!(col.sample(y), field.at3(wx, y, wz), "({wx},{y},{wz})");
                    }
                }
            }
        }
    }

    #[test]
    fn cave_sup_never_under_reports() {
        // The chunk field bound must be a true upper bound: no cell may exceed it.
        let g = terrain(7);
        for (cx, cy, cz) in [(0, -3, 0), (2, -1, -5), (-4, -30, 6)] {
            let (x0, y0, z0) = (cx * 16, cy * 16, cz * 16);
            let sup = g.caves.field.sup(x0, y0, z0).0;
            for lz in 0..16 {
                for lx in 0..16 {
                    for ly in 0..16 {
                        let v = g.caves.field.at3(x0 + lx, y0 + ly, z0 + lz).0;
                        assert!(v <= sup + BOUND_SLACK, "sup {sup} < sample {v}");
                    }
                }
            }
        }
    }

    #[test]
    fn bound_encloses_samples_both_sides() {
        // The interval bound must enclose every cell — lower AND upper — so a
        // density built from it can decide `straddles(0)` soundly (D7).
        let g = terrain(7);
        for (cx, cy, cz) in [(0, -3, 0), (2, -1, -5), (-4, -30, 6), (30_000_000, 4, -7)] {
            let (x0, y0, z0) = (cx * 16, cy * 16, cz * 16);
            for field in [&g.caves.field, &g.islands.field] {
                let b = field.bound(x0, y0, z0);
                for lz in 0..16 {
                    for lx in 0..16 {
                        for ly in 0..16 {
                            let v = field.at3(x0 + lx, y0 + ly, z0 + lz).0;
                            assert!(v >= b.lo - BOUND_SLACK, "lo {} > sample {v}", b.lo);
                            assert!(v <= b.hi + BOUND_SLACK, "hi {} < sample {v}", b.hi);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn spline_image_scans_interior_knots() {
        // RIDGE_KNOTS dips to -0.7 at x=0.50, between endpoints that are ~0.9.
        // An endpoints-only image would miss the trough; `image` must not.
        let s = Spline(RIDGE_KNOTS);
        let (mn, mx) = s.image(0.0, 1.0);
        assert!(mn <= -0.69, "image floor reaches the interior trough, got {mn}");
        assert!(mx >= 0.99, "image ceiling reaches the interior peak, got {mx}");
        // Census: dense sampling never leaves the reported image.
        for i in 0..=1000 {
            let x = i as f32 / 1000.0;
            let v = s.eval(x);
            assert!(v >= mn - 1e-6 && v <= mx + 1e-6, "eval {v} outside image at {x}");
        }
        // A sub-interval straddling only the trough side stays tight.
        let (mn2, mx2) = s.image(0.35, 0.65);
        assert!(mn2 <= -0.69 && mx2 <= 0.21, "sub-interval image [{mn2},{mx2}]");
    }

    #[test]
    fn generated_chunks_match_per_cell_block_at() {
        let g = terrain(3);
        let coords = [
            (0, 0, 0), (0, 1, 0), (2, 4, -3), (-1, 5, 7), (0, -2, 0),
            (0, -4, 0), (0, 3, 0), (2, 14, 3), (0, -1, 5), (0, -40, 0),
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
    fn height_is_deterministic_and_finite_far_out() {
        let g = terrain(9);
        for &wx in &[100_000_000, 999_999_000, -100_000_000] {
            for wz in -8..8 {
                let h = g.height(wx, wz * 12_345_679);
                assert!(h >= 1, "height floored at 1");
                assert_eq!(h, g.height(wx, wz * 12_345_679), "bit-stable");
            }
        }
    }

    #[test]
    fn slab_has_oceans_mountains_and_rivers() {
        // Shape sanity: over a wide slab, columns span from below sea level
        // (oceans) to well above (mountains), and thin sub-sea inland channels
        // (rivers) exist.
        let g = terrain(5);
        let sea = g.sea_level();
        let (mut ocean, mut mountain) = (0usize, 0usize);
        for x in -512..512 {
            for z in -512..512 {
                let h = g.height(x * 3, z * 3);
                ocean += (h < sea) as usize;
                mountain += (h > sea + 30) as usize;
            }
        }
        assert!(ocean > 0, "oceans exist (columns below sea level)");
        assert!(mountain > 0, "mountains exist (columns well above sea level)");
    }

    #[test]
    fn oceans_fill_with_water() {
        let (reg, g) = terrain_with_registry(5);
        let water = reg.id_by_name("Water").unwrap();
        let sea = g.sea_level();
        // Find an ocean column and confirm the cell just under sea level is water.
        let mut found = false;
        'scan: for x in -512..512 {
            for z in -512..512 {
                let (wx, wz) = (x * 3, z * 3);
                let h = g.height(wx, wz);
                if h < sea {
                    assert_eq!(g.block_at(wx, sea - 1, wz, h), water, "ocean fills to sea level");
                    found = true;
                    break 'scan;
                }
            }
        }
        assert!(found, "the slab holds at least one ocean column");
    }

    #[test]
    fn biome_warp_displaces_the_field() {
        // S5: the biome axes are domain-warped, so the warped read differs from
        // the unwarped field at the same coordinate wherever the offset is nonzero.
        let g = terrain(5);
        let differs = (0..500).any(|i| {
            let (wx, wz) = (i * 37, i * -53);
            g.temperature.at(wx, wz) != g.temperature.field.at(wx, wz)
        });
        assert!(differs, "domain warp moves the biome sample off the raw lattice");
    }

    #[test]
    fn lakes_appear_above_sea_level() {
        // S4: the water table is a field, so inland lake blobs hold standing water
        // above sea level — a cell at the lake surface (y = height > sea) is water.
        let (reg, g) = terrain_with_registry(5);
        let water = reg.id_by_name("Water").unwrap();
        let sea = g.sea_level();
        let mut found = false;
        'scan: for x in -700..700 {
            for z in -700..700 {
                let (wx, wz) = (x * 3, z * 3);
                let h = g.height(wx, wz);
                if h > sea && g.block_at(wx, h, wz, h) == water {
                    found = true;
                    break 'scan;
                }
            }
        }
        assert!(found, "an inland lake holds standing water above sea level");
    }

    #[test]
    fn trees_scatter_wood_and_leaves_above_grass() {
        // S7: the decoration pass scatters trees on dry grassy columns, placing a
        // wood trunk crowned by a leaf canopy above the surface.
        let (reg, g) = terrain_with_registry(5);
        let (wood, leaves) = (reg.id_by_name("Wood").unwrap(), reg.id_by_name("Leaves").unwrap());
        let (mut saw_wood, mut saw_leaves) = (false, false);
        'scan: for x in -256..256 {
            for z in -256..256 {
                let (wx, wz) = (x * 3, z * 3);
                let h = g.height(wx, wz);
                for up in 0..=TREE_TOP {
                    let b = g.block_at(wx, h + up, wz, h);
                    saw_wood |= b == wood;
                    saw_leaves |= b == leaves;
                }
                if saw_wood && saw_leaves {
                    break 'scan;
                }
            }
        }
        assert!(saw_wood, "trees place wood trunks above the surface");
        assert!(saw_leaves, "trees place leaf canopies above the surface");
    }

    #[test]
    fn tree_chunk_fill_matches_per_cell() {
        // The chunk fast path's cached tree stamping must agree with the per-cell
        // block_at overlay, cell for cell, in a surface chunk that holds trees.
        let g = terrain(5);
        for (cx, cy, cz) in [(0, 1, 0), (3, 1, -2), (-5, 2, 4), (7, 1, 9)] {
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
                            "cell ({wx},{wy},{wz})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn overhangs_place_solid_rock_above_the_surface() {
        // S6: overhang shelves put solid rock strictly above a column's heightfield
        // surface — relief the pure heightfield could never express.
        let (reg, g) = terrain_with_registry(5);
        let stone = reg.id_by_name("Stone").unwrap();
        let mut found = false;
        'scan: for x in -300..300 {
            for z in -300..300 {
                let (wx, wz) = (x * 2, z * 2);
                let h = g.height(wx, wz);
                for up in 1..OVERHANG_REACH {
                    if g.block_at(wx, h + up, wz, h) == stone {
                        found = true;
                        break 'scan;
                    }
                }
            }
        }
        assert!(found, "an overhang shelf sits above the heightfield surface");
    }

    #[test]
    fn islands_only_at_or_above_the_band() {
        let g = terrain(9);
        for y in [-1000, -1, 0, 32, 63] {
            for x in -32..32 {
                assert!(g.islands.excess(x * 5, y, x * 3 - 7, 0) <= 0.0, "no islands below y=64");
            }
        }
        let any = (-200..200).any(|x| (-200..200).step_by(8).any(|z| g.islands.excess(x, 64, z, 0) > 0.0));
        assert!(any, "the band floor can hold islands");
    }

    #[test]
    fn deep_chunks_are_uniform_stone() {
        let (reg, g) = terrain_with_registry(11);
        let stone = reg.id_by_name("Stone").unwrap();
        // Scan +z for a deep chunk the cave bound clears.
        let cy = -20;
        let y0 = cy * 16;
        let mut proven = None;
        for cz in 0..128 {
            let (x0, z0) = (0, cz * 16);
            let mut h_max = i32::MIN;
            let mut h_min = i32::MAX;
            for lx in 0..16 {
                for lz in 0..16 {
                    let h = g.height(x0 + lx, z0 + lz);
                    h_max = h_max.max(h);
                    h_min = h_min.min(h);
                }
            }
            if y0 + 15 < h_min - ORE_MAX_DEPTH
                && g.caves.dormant(x0, y0, z0, h_max - y0)
                && g.ravines.dormant(x0, y0, z0, h_max - y0)
            {
                proven = Some(cz);
                break;
            }
        }
        let cz = proven.expect("a bound-cleared deep chunk within 128");
        assert_eq!(g.generate(0, cy, cz), ChunkData::Uniform(stone));
    }
}
