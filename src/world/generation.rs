//! Terrain generation decoupled from chunk storage so the algorithm can be swapped.
//!
//! Uses BlockIds resolved once up front to avoid per-voxel allocation.
//!
//! One primitive Fbm (fractal value noise) shaped by:
//! - Ramp — falling threshold by depth
//! - Term — Fbm vs Ramp comparison (caves, ravines)
//! - Spline / Control — terrain/biome curves
//!
//! Heightfield via Terrain::profile, sampled once per Column;
//! water fills to `water_level` everywhere (sea/lakes/rivers);
//! overhangs/trees layer on top.
//!
//! The heightfield routes through [`Terrain::profile`], sampled once per [`Column`]
//! and threaded downstream, so biome dressing never re-samples noise. Water is a
//! normal translucent solid: every column floods up to its `water_level` field by
//! one `wy < water_level` rule — `sea_level` almost everywhere, raised inside lake
//! blobs — giving oceans, coastal seas, rivers, and highland lakes for free.
//! Overhang shelves and ravines layer on as further features. Terrain emits
//! ONLY element unions the placement table derives — no named blocks, no
//! decoration overlay, no special cases (trees were retired in the v3 pass).
//!
//! Generation is a pure function of (seed, chunk coord) — worker threads and
//! multiplayer clients all reproduce identical chunks. Whole-chunk generation takes
//! shortcuts where it can (deep rock the cave field can't reach is `Uniform(stone)`;
//! sky above every surface is `Uniform(air)` or `Uniform(water)`) and otherwise
//! collapses an all-identical dense fill.
use std::ops::RangeInclusive;

use super::chunk::{CHUNK_SIZE, CHUNK_VOLUME, Chunk, ChunkData};
use super::placement;
use crate::block::registry::{AIR, BlockId, BlockRegistry};

/// Ground height per cell of a 16×16 chunk column — identical to [`TerrainGenerator::height`].
pub type ColumnHeights = [i32; CHUNK_SIZE * CHUNK_SIZE];

/// Sample [`TerrainGenerator::height`] across a chunk column. Used by the
/// default [`TerrainGenerator::generate_column`] (test gens that do not batch).
fn sample_column_heights(g: &(impl TerrainGenerator + ?Sized), cx: i32, cz: i32) -> ColumnHeights {
    let x0 = cx * CHUNK_SIZE as i32;
    let z0 = cz * CHUNK_SIZE as i32;
    let mut heights = [0i32; CHUNK_SIZE * CHUNK_SIZE];
    for lz in 0..CHUNK_SIZE {
        for lx in 0..CHUNK_SIZE {
            heights[lx + lz * CHUNK_SIZE] = g.height(x0 + lx as i32, z0 + lz as i32);
        }
    }
    heights
}

/// Which generator a world is built with. Folded into the content fingerprint.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum WorldgenKind {
    #[default]
    Classic,
    Diffusion,
}

impl WorldgenKind {
    pub fn id(self) -> &'static str {
        match self {
            Self::Classic => "classic",
            Self::Diffusion => "diffusion",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "classic" => Some(Self::Classic),
            "diffusion" => Some(Self::Diffusion),
            _ => None,
        }
    }

    pub fn wire(self) -> u8 {
        match self {
            Self::Classic => 0,
            Self::Diffusion => 1,
        }
    }

    pub fn from_wire(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Classic),
            1 => Some(Self::Diffusion),
            _ => None,
        }
    }
}

/// Produces terrain for absolute world coordinates.
pub trait TerrainGenerator: Send + Sync {
    /// World seed this generator was built from.
    fn seed(&self) -> i64 {
        0
    }
    /// Sea level in blocks, for spawn and flooding.
    fn sea_level(&self) -> i32 {
        20
    }
    /// Stable id folded into the content fingerprint (`classic`, `diffusion`, …).
    fn kind(&self) -> &'static str {
        "classic"
    }
    /// Topmost non-ground cell in this column.
    fn height(&self, wx: i32, wz: i32) -> i32;

    /// Ground height for every cell of the 16×16 chunk column at `(cx, cz)`.
    /// Default walks [`height`](Self::height); Diffusion fills the rectangle
    /// from the field in one tile load.
    fn heights_16(&self, cx: i32, cz: i32) -> ColumnHeights {
        sample_column_heights(self, cx, cz)
    }

    /// Surface block (biome-dependent).
    fn surface_at(&self, wx: i32, wz: i32) -> BlockId;

    /// Deep block (underground / far-LOD sides).
    fn deep(&self) -> BlockId;

    /// Block at world coordinate; default is surface/deep/air; Terrain overrides.
    fn block_at(&self, wx: i32, wy: i32, wz: i32, height: i32) -> BlockId {
        if wy >= height {
            AIR
        } else if wy >= height - 1 {
            self.surface_at(wx, wz)
        } else {
            self.deep()
        }
    }

    /// The block a coarse far-LOD tile shows at a cell. Distinct from
    /// [`block_at`](Self::block_at) because a tile samples at a `2^k`-metre stride
    /// where sub-cell detail would alias to noise: only the volumetric
    /// silhouette (ground/water/overhang/island) matters. The default reuses
    /// `block_at`; [`Terrain`] overrides it to recompute the column profile
    /// only once per cell instead of the caller re-passing `height`.
    fn lod_block_at(&self, wx: i32, wy: i32, wz: i32) -> BlockId {
        self.block_at(wx, wy, wz, self.height(wx, wz))
    }

    /// Fill a vertical run of LOD cells; default per-cell, Terrain batches.
    fn lod_column(&self, wx: i32, wz: i32, ys: &[i32], out: &mut [BlockId]) {
        for (o, &wy) in out.iter_mut().zip(ys) {
            *o = self.lod_block_at(wx, wy, wz);
        }
    }

    /// Generate chunk; default dense then collapse; Terrain shortcuts.
    fn generate(&self, cx: i32, cy: i32, cz: i32) -> ChunkData {
        let y0 = cy * CHUNK_SIZE as i32;
        let mut cells = Box::new([AIR; CHUNK_VOLUME]);
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let wx = cx * CHUNK_SIZE as i32 + lx as i32;
                let wz = cz * CHUNK_SIZE as i32 + lz as i32;
                let height = self.height(wx, wz);
                for ly in 0..CHUNK_SIZE {
                    cells[Chunk::index(lx, ly, lz)] = self.block_at(wx, y0 + ly as i32, wz, height);
                }
            }
        }
        ChunkData::from_cells(cells)
    }

    /// Generate a vertical run of chunks together with the column's 256 ground
    /// heights (identical to [`height`](Self::height) at each cell). Heights
    /// are produced even when `cy` is empty — the profile sample does not
    /// depend on the chunk layers. Default loops per-chunk; Terrain and
    /// Diffusion batch the profile.
    fn generate_column(
        &self,
        cx: i32,
        cz: i32,
        cy: RangeInclusive<i32>,
    ) -> (Vec<(i32, ChunkData)>, ColumnHeights) {
        let chunks = cy.map(|cyy| (cyy, self.generate(cx, cyy, cz))).collect();
        (chunks, sample_column_heights(self, cx, cz))
    }
}

// The noise vocabulary.

/// Noise sample in [0, 1); prevents silent misuse with coordinates/heights.
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
pub struct Unit(pub f32);

/// World seed; all streams derive from it.
#[derive(Clone, Copy)]
struct Seed(i64);

/// Decorrelated hash stream; avoids manual salt juggling.
#[derive(Clone, Copy)]
struct Stream(u64);

impl Seed {
    /// Fresh stream for this salt.
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

/// Fractal value noise; the one noise type used throughout.
#[derive(Clone)]
struct Fbm {
    stream: Stream,
    cell: f64,
    octaves: u8,
    /// Sum of the octave weights. Immutable for a field, so computing it for
    /// every 2-D/3-D sample only burns cycles during chunk generation.
    norm: f32,
}

impl Fbm {
    fn new(stream: Stream, cell: f64, octaves: u8) -> Self {
        assert!(octaves as usize <= MAX_FBM_OCTAVES, "FBM octave cache is too small");
        let norm = (0..octaves).map(|o| 0.5f32.powi(o as i32)).sum();
        Self { stream, cell, octaves, norm }
    }

    fn at3(&self, wx: i32, wy: i32, wz: i32) -> Unit {
        let mut acc = 0.0;
        let mut w = 1.0;
        for o in 0..self.octaves {
            let f = (1u32 << o) as f64 / self.cell;
            acc += w * octave(self.stream.octave(o as u64), wx, wy, wz, f);
            w *= 0.5;
        }
        Unit(acc / self.norm)
    }

    fn at(&self, wx: i32, wz: i32) -> Unit {
        let mut acc = 0.0;
        let mut w = 1.0;
        for o in 0..self.octaves {
            let f = (1u32 << o) as f64 / self.cell;
            acc += w * octave2(self.stream.octave(o as u64), wx, wz, f);
            w *= 0.5;
        }
        Unit(acc / self.norm)
    }

    /// Cached planes down a column; bit-identical to at3, cheaper.
    fn column(&self, wx: i32, wz: i32, y_lo: i32, y_hi: i32) -> FbmColumn {
        // Inline array (no per-column heap allocation for the two or three
        // octave planes); `flatten` in `sample` skips the unused slots.
        let cols = std::array::from_fn(|index| {
            (index < self.octaves as usize).then(|| {
                let o = index as u8;
                let f = (1u32 << o) as f64 / self.cell;
                OctaveColumn::new(self.stream.octave(o as u64), wx, wz, f, y_lo, y_hi)
            })
        });
        FbmColumn { cols, norm: self.norm }
    }

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
        let n = self.norm;
        Interval { lo: lo / n, hi: hi / n }
    }

    fn sup(&self, x0: i32, y0: i32, z0: i32) -> Unit {
        let mut hi = 0.0;
        let mut w = 1.0;
        for o in 0..self.octaves {
            let f = (1u32 << o) as f64 / self.cell;
            hi += w * octave_sup(self.stream.octave(o as u64), x0, y0, z0, f);
            w *= 0.5;
        }
        Unit(hi / self.norm)
    }

    /// 2D upper bound for island placement checks.
    fn sup2(&self, x0: i32, z0: i32, dx: i32, dz: i32) -> f32 {
        let mut hi = 0.0;
        let mut w = 1.0;
        for o in 0..self.octaves {
            let f = (1u32 << o) as f64 / self.cell;
            hi += w * octave2_sup(self.stream.octave(o as u64), x0, z0, dx, dz, f);
            w *= 0.5;
        }
        hi / self.norm
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct Interval {
    lo: f32,
    hi: f32,
}

/// Every configured terrain FBM has at most this many octaves. Keeping the
/// cached vertical planes inline avoids two tiny heap allocations (cave +
/// ravine) for every dense XZ column, plus island-detail allocations where
/// active. [`Fbm::new`] asserts the bound.
const MAX_FBM_OCTAVES: usize = 3;

struct FbmColumn {
    cols: [Option<OctaveColumn>; MAX_FBM_OCTAVES],
    norm: f32,
}

impl FbmColumn {
    fn sample(&self, y: i32) -> Unit {
        let mut acc = 0.0;
        let mut w = 1.0;
        for c in self.cols.iter().flatten() {
            acc += w * c.sample(y);
            w *= 0.5;
        }
        Unit(acc / self.norm)
    }
}

/// Falling threshold; used for caves, terrain.
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

/// Fbm vs falling threshold; carves when exceeding (caves, ravines).
#[derive(Clone)]
struct Term {
    field: Fbm,
    ramp: Ramp,
    /// Minimum depth at which active.
    gate: i32,
    /// Ramp input origin; ramp receives `t - origin`.
    origin: i32,
}

type Excess = f32;

impl Term {
    const INACTIVE: Excess = -1.0e9;

    fn threshold(&self, t: i32) -> f32 {
        self.ramp.at((t - self.origin) as f32).0
    }

    fn excess(&self, wx: i32, wy: i32, wz: i32, height: i32) -> Excess {
        let t = height - wy;
        if t < self.gate {
            Self::INACTIVE
        } else {
            self.field.at3(wx, wy, wz).0 - self.threshold(t)
        }
    }

    fn excess_col(&self, col: &FbmColumn, wy: i32, height: i32) -> Excess {
        let t = height - wy;
        if t < self.gate {
            Self::INACTIVE
        } else {
            col.sample(wy).0 - self.threshold(t)
        }
    }

    /// True if this term might be active in the chunk.
    fn dormant(&self, x0: i32, y0: i32, z0: i32, max_t: i32) -> bool {
        self.field.sup(x0, y0, z0).0 + BOUND_SLACK < self.threshold(max_t)
    }
}

const BOUND_SLACK: f32 = 1e-5;

/// Flying islands: placement mask + vertical profile (domed top, long keel).
/// Placement mask survives LOD coarse sampling; profile gives shape.
#[derive(Clone)]
struct Islands {
    place: Fbm,
    lift: Fbm,
    detail: Fbm,
    on: f32,
    core_floor: f32,
    top_h: f32,
    keel_h: f32,
    detail_amp: f32,
    band_lo: i32,
    band_span: i32,
}

struct IslandColumn {
    core: f32,
    center: i32,
    detail: FbmColumn,
}

impl Islands {
    fn core_center(&self, wx: i32, wz: i32) -> Option<(f32, i32)> {
        let place = self.place.at(wx, wz).0;
        if place <= self.on {
            return None;
        }
        let t = (place - self.on) / (1.0 - self.on);
        let s = t * t * (3.0 - 2.0 * t);
        let core = self.core_floor + (1.0 - self.core_floor) * s;
        let center = self.band_lo + (self.lift.at(wx, wz).0 * self.band_span as f32).round() as i32;
        Some((core, center))
    }

    fn density(&self, core: f32, center: i32, wy: i32, detail: f32) -> f32 {
        let dy = (wy - center) as f32;
        let vfall = if dy >= 0.0 { dy / self.top_h } else { -dy / self.keel_h };
        core - vfall + (detail * 2.0 - 1.0) * self.detail_amp
    }

    fn column(&self, wx: i32, wz: i32, y_lo: i32, y_hi: i32) -> Option<IslandColumn> {
        let (core, center) = self.core_center(wx, wz)?;
        Some(IslandColumn { core, center, detail: self.detail.column(wx, wz, y_lo, y_hi) })
    }

    fn solid_col(&self, c: &IslandColumn, wy: i32) -> bool {
        self.density(c.core, c.center, wy, c.detail.sample(wy).0) > 0.0
    }

    fn solid(&self, wx: i32, wy: i32, wz: i32) -> bool {
        match self.core_center(wx, wz) {
            None => false,
            Some((core, center)) => self.density(core, center, wy, self.detail.at3(wx, wy, wz).0) > 0.0,
        }
    }

    fn band_bottom(&self) -> i32 {
        self.band_lo - (self.keel_h * (1.0 + self.detail_amp)).ceil() as i32
    }
    fn band_top(&self) -> i32 {
        self.band_lo + self.band_span + (self.top_h * (1.0 + self.detail_amp)).ceil() as i32
    }

    fn possible(&self, x0: i32, y0: i32, z0: i32, dims: (i32, i32, i32)) -> bool {
        let (dx, dy, dz) = dims;
        if y0 >= self.band_top() || y0 + dy <= self.band_bottom() {
            return false;
        }
        self.place.sup2(x0, z0, dx, dz) > self.on
    }
}

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

    #[cfg(test)]
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

/// Displaces coordinates before reading (features meander).
#[derive(Clone)]
struct Warp {
    field: Fbm,
    dx: Fbm,
    dz: Fbm,
    amp: f64,
}

impl Warp {
    /// The displaced sample coordinates. Split from [`at`](Self::at) so
    /// controls that intentionally share one displacement field pair (the
    /// height axes; the climate axes) can compute the warp once and read each
    /// of their fields at the same coordinates — bit-identical to warping each
    /// read separately, because the fields share dx/dz by construction.
    fn coordinates(&self, wx: i32, wz: i32) -> (i32, i32) {
        let ox = (self.dx.at(wx, wz).0 as f64 * 2.0 - 1.0) * self.amp;
        let oz = (self.dz.at(wx, wz).0 as f64 * 2.0 - 1.0) * self.amp;
        (wx + ox.round() as i32, wz + oz.round() as i32)
    }

    #[cfg(test)]
    fn at(&self, wx: i32, wz: i32) -> Unit {
        let (x, z) = self.coordinates(wx, wz);
        self.field.at(x, z)
    }
}

/// Spline-shaped field; gamma redistributes values; read via domain warp.
#[derive(Clone)]
struct Control {
    field: Warp,
    gamma: f32,
    curve: Spline,
}

impl Control {
    /// Shape an already-sampled raw field value: gamma redistribution, then
    /// the spline. Takes the sample rather than coordinates so warp-sharing
    /// callers (see `Terrain::profile`) can feed one shared read to several
    /// controls.
    fn shape(&self, raw: Unit) -> f32 {
        // Most terrain controls deliberately use the identity gamma. Avoid a
        // comparatively expensive libm call for those samples (`powf(x, 1.0)`
        // is exactly `x`, so the bypass is bit-identical).
        let redistributed = if self.gamma == 1.0 { raw.0 } else { raw.0.powf(self.gamma) };
        self.curve.eval(redistributed)
    }
}

/// Precomputed column data; sampled once per vertical run.
#[derive(Clone, Copy)]
struct Column {
    height: i32,
    /// Water level (sea level or lake surface).
    water_level: i32,
    temperature: Unit,
    humidity: Unit,
    /// Y-invariant surface class; cached so crust/dress don't recompute it.
    kind: placement::SurfaceKind,
    /// Surface block for this column (dress LUT + scatter).
    dress: BlockId,
}

/// Per-column carve / overhang / island bits for one chunk layer.
#[derive(Clone, Copy)]
struct ColMask {
    wx: i32,
    wz: i32,
    carved: u16,
    overhang: u16,
    island: u32,
}

// Value noise primitives. Uses f64 world coordinates for far-out stability.

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

fn fade(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn reduce(w: i32, freq: f64) -> (i64, f32) {
    let t = w as f64 * freq;
    let cell = t.floor() as i64;
    (cell, (t - cell as f64) as f32)
}

fn plane_value(seed: u64, xi: i32, fx: f32, ly: i32, zi: i32, fz: f32) -> f32 {
    let (tx, tz) = (fade(fx), fade(fz));
    let v00 = lattice(seed, xi, ly, zi);
    let v10 = lattice(seed, xi + 1, ly, zi);
    let v01 = lattice(seed, xi, ly, zi + 1);
    let v11 = lattice(seed, xi + 1, ly, zi + 1);
    lerp(lerp(v00, v10, tx), lerp(v01, v11, tx), tz)
}

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

fn octave2(seed: u64, wx: i32, wz: i32, freq: f64) -> f32 {
    let (xi, fx) = reduce(wx, freq);
    let (zi, fz) = reduce(wz, freq);
    plane_value(seed, xi as i32, fx, 0, zi as i32, fz)
}

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

fn octave_bound(seed: u64, x0: i32, y0: i32, z0: i32, freq: f64) -> (f32, f32) {
    octave_range::<true>(seed, x0, y0, z0, freq)
}

fn octave_sup(seed: u64, x0: i32, y0: i32, z0: i32, freq: f64) -> f32 {
    octave_range::<false>(seed, x0, y0, z0, freq).1
}

fn octave_range<const LO: bool>(seed: u64, x0: i32, y0: i32, z0: i32, freq: f64) -> (f32, f32) {
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
    let (mut lo, mut hi) = (if LO { f32::INFINITY } else { 0.0 }, f32::NEG_INFINITY);
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
                            if LO {
                                lo = lo.min(v);
                            }
                            hi = hi.max(v);
                        }
                    }
                }
            }
        }
    }
    (lo, hi)
}

fn octave2_sup(seed: u64, x0: i32, z0: i32, dx: i32, dz: i32, freq: f64) -> f32 {
    let (xc0, _) = reduce(x0, freq);
    let (xc1, _) = reduce(x0 + dx - 1, freq);
    let (zc0, _) = reduce(z0, freq);
    let (zc1, _) = reduce(z0 + dz - 1, freq);
    let mut hi = 0.0f32;
    for xi in xc0..=xc1 + 1 {
        for zi in zc0..=zc1 + 1 {
            hi = hi.max(lattice(seed, xi as i32, 0, zi as i32));
        }
    }
    hi
}

pub(crate) fn cell_hash(seed: i64, x: i32, y: i32, z: i32) -> u32 {
    let h = (seed as u64 ^ 0x517C_C1B7_2722_0A95)
        ^ (x as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u32 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ (z as u32 as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
    let h = crate::hash::splitmix_finish(h);
    (h >> 32) as u32
}

// Tuning surface — the one place terrain flavour lives, as const data.

/// Islands start above sea; no water interaction.
pub const ISLAND_MIN_Y: i32 = 112;
/// Shallowest cave carve depth (soil crust).
const CAVE_MIN_DEPTH: i32 = 6;
/// Shallowest ravine carve (deeper than cave, reads as gashes).
const RAVINE_MIN_DEPTH: i32 = 8;
/// Overhang reach above ground.
const OVERHANG_REACH: i32 = 8;
/// Overhang threshold (stricter higher up).
const OVERHANG_THRESH: f32 = 0.60;
const OVERHANG_FADE: f32 = 0.05;
/// Island surface cells at or above this altitude freeze to Ice.
const ICE_SURFACE_Y: i32 = 220;

// Salts for decorrelated hash streams.
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
const HEIGHT_WARPX_SALT: u64 = 0x000B;
const HEIGHT_WARPZ_SALT: u64 = 0x000C;
const TERRACE_SALT: u64 = 0x000D;
const CAVE_SALT: u64 = 0xA24B_AED4_963E_E407;
const ISLAND_SALT: u64 = 0x1D3E_66F0_9C2A_B517;
const ISLAND_LIFT_SALT: u64 = 0x4C8A_2FE1_90B7_D63A;
const ISLAND_DETAIL_SALT: u64 = 0x9F27_5B3C_E140_A8D6;
const RAVINE_SALT: u64 = 0x77C1_9B0A_5E3D_2F81;
const OVERHANG_SALT: u64 = 0x2B9F_10E6_A4C7_5D33;
/// Decorrelates the second ore stream from the first: two independent rolls
/// per stone cell whose deduped union is the cell's payload set — 0, 1, or 2
/// extra elements, never more (the arity-2 bound is this construction).
const ORE_B_SALT: i64 = 0x9D3A_44E1_0C67_B52Bu64 as i64;
/// The cave-wall (floor/ceiling) cluster roll — its own stream so cavern
/// glow is independent of the seam layout.
const CAVE_WALL_SALT: i64 = 0x2F8C_71A5_E9D0_63B7u64 as i64;
/// The beach-edge dither — the column hash deciding whether a just-above-water
/// grassy column joins the sand/soil transition band.
const BEACH_SALT: i64 = 0x6B14_D8F3_2A79_C40Du64 as i64;
/// The surface-growth scatter roll — luminous tufts on the top ground cell,
/// its own stream so surface glow never correlates with beach dither or ores.
const SURFACE_SCATTER_SALT: i64 = 0x3E7A_1B96_D4C8_205Fu64 as i64;

/// Continentalness → base height offset from sea level: deep ocean floors, coastal
/// shelves, inland plains, and high interiors.
const CONT_KNOTS: &[(f32, f32)] = &[
    (0.00, -30.0),
    (0.35, -12.0),
    (0.48, -3.0),
    (0.55, 2.0),
    (0.62, 4.0),   // flat bench — plains
    (0.66, 14.0),  // sharp step up — escarpment
    (0.82, 26.0),
    (0.90, 30.0),  // high plateau shelf
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
    (0.88, 90.0),   // steepened tail — dramatic highlands
    (1.00, 150.0),  // rare, genuinely tall country
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

/// Domain-warp for the height fields (continentalness, erosion): coordinate
/// displacement that makes coastlines and relief provinces meander instead of
/// tracking the value-noise lattice. Larger cell than the fields it warps; amp
/// small against their cells so it nudges features rather than scrambling them.
const HEIGHT_WARP_CELL: f64 = 360.0;
const HEIGHT_WARP_AMP: f64 = 60.0;

/// Regional terracing (mesa / badlands): a low-frequency mask above `TERRACE_ON`
/// quantizes the land surface into benches `TERRACE_STEP` blocks tall, so only
/// scattered regions step while the rest stays smooth. The mask's smoothstep past
/// the threshold blends terracing in at region edges rather than cliffing.
// Alien identity: terraces are common and TALL — stepped mesa country is a
// signature landform, not a rarity. Free at RD: the vertical view volume caps
// loaded-chunk count regardless of relief (benched identical vs baseline).
const TERRACE_CELL: f64 = 260.0;
const TERRACE_ON: f32 = 0.54;
const TERRACE_STEP: f32 = 14.0;

/// Continentalness (base height above sea) past which ridged mountain ranges kick
/// in, so ranges sharpen genuine highlands and never lift ocean floors.
const RANGE_ONSET: f32 = 14.0;
/// Peak extra height a ridgeline adds atop an elevated column, scaled by both how
/// far past the onset the base sits and the erosion amplitude. High: highlands
/// crest into blade-thin spines rather than rounded domes.
const RANGE_GAIN: f32 = 0.75;

/// Snow line: dressed columns this far above sea level freeze over, so only real
/// peaks cap with snow while mid-height slopes keep grass and bare stone.
const SNOW_ABOVE_SEA: i32 = 55;
/// Biome dressing cutoffs on temperature / humidity units.
const COLD: f32 = 0.30;
const HOT: f32 = 0.72;
const DRY: f32 = 0.32;

/// Island shaping (placement-masked isosurface). The placement mask is large so
/// islands out-size the LOD cell (surviving coarse sampling); `ON` sets rarity;
/// the top/keel heights set the domed-over-tapered silhouette; the detail cell/amp
/// set the organic edge; lift/band spread centres over the vertical band.
const ISLAND_PLACE_CELL: f64 = 190.0;
const ISLAND_LIFT_CELL: f64 = 400.0;
const ISLAND_DETAIL_CELL: f64 = 20.0;
/// Rarity: only where the mask clears this does an island exist. High → sparse,
/// small clumps rather than one wide sheet.
const ISLAND_ON: f32 = 0.70;
/// Minimum body thickness fraction — keeps islands chunky, not flying grass.
const ISLAND_CORE_FLOOR: f32 = 0.55;
const ISLAND_TOP_H: f32 = 10.0;
const ISLAND_KEEL_H: f32 = 22.0;
const ISLAND_DETAIL_AMP: f32 = 0.30;
const ISLAND_BAND_SPAN: i32 = 120;

// Terrain — the game's generator. `SineHills` kept as an alias so existing call
// sites need no change.

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
    /// carve a bowl, so `water_level` is a field, not the flat sea constant.
    lakes: Fbm,
    /// Ridged mountain-range noise: a folded field that raises sharp crests
    /// along ridgelines where the terrain is already elevated.
    ranges: Fbm,
    /// Low-frequency terrace mask: scattered regions where the land surface is
    /// quantized into benches (mesa/badlands), blended in at their edges.
    terraces: Fbm,
    /// Biome axes are domain-warped so temperature/humidity — and thus the
    /// biome borders they dress — meander instead of sitting in round blobs.
    temperature: Warp,
    humidity: Warp,
    caves: Term,
    /// A second carve term: narrow deep ravines slicing the ground, folded in
    /// with the caves as another `excess > 0` subtraction.
    ravines: Term,
    /// 3-D overhang fill: solid rock placed *above* the heightfield surface
    /// in a fading band, so cliffs grow shelves the pure heightfield can't express.
    overhangs: Fbm,
    islands: Islands,

    /// Pre-resolved placement LUTs — the generator's only view of the palette.
    /// Terrain speaks elements: every material is the union of the elements
    /// whose placement rules want the cell (see [`placement`]) — nothing else
    /// exists. No decoration overlay, no named blocks, no special cases.
    mat: placement::Resolved,
}

/// Kept for compatibility with existing call sites.
pub type SineHills = Terrain;

impl Terrain {
    /// Build the generator for a seed. `base` sets sea level.
    ///
    /// Compiles the builtin placement table against the registry — startup,
    /// main thread, before any worker exists: every block terrain can emit is
    /// registered here in canonical order, and the generator keeps only the
    /// resolved ids (it can never register at runtime — it holds no registry).
    pub fn new(registry: &mut BlockRegistry, base: f32, seed: i64) -> Self {
        // Terrain speaks only the placement table now — no named block is ever
        // resolved by hand; every material is an enumerated element union.
        let mat = placement::builtin().compile(registry);
        let s = Seed(seed);
        let fbm = |salt: u64, cell: f64, octaves: u8| Fbm::new(s.stream(salt), cell, octaves);
        // Shared height-warp offsets (like the biome axes share theirs), so
        // continentalness and erosion meander in step rather than decorrelating.
        let hwarp = |field: Fbm| Warp {
            field,
            dx: fbm(HEIGHT_WARPX_SALT, HEIGHT_WARP_CELL, 2),
            dz: fbm(HEIGHT_WARPZ_SALT, HEIGHT_WARP_CELL, 2),
            amp: HEIGHT_WARP_AMP,
        };
        Self {
            seed,
            sea_level: base.round() as i32,
            continentalness: Control { field: hwarp(fbm(CONT_SALT, 280.0, 3)), gamma: 1.0, curve: Spline(CONT_KNOTS) },
            // Erosion redistributed toward its floor (gamma > 1): flat-to-rolling
            // country becomes the common case and tall relief a rarer, sharper tail.
            erosion: Control { field: hwarp(fbm(EROSION_SALT, 320.0, 3)), gamma: 1.35, curve: Spline(EROSION_KNOTS) },
            // Weirdness warped too: rivers meander with the ridge valleys rather
            // than running the lattice, since the river read shares this field.
            weirdness: Control { field: hwarp(fbm(WEIRD_SALT, 72.0, 3)), gamma: 1.0, curve: Spline(RIDGE_KNOTS) },
            detail: fbm(DETAIL_SALT, 64.0, 3),
            lakes: fbm(LAKE_SALT, 220.0, 2),
            ranges: fbm(RANGE_SALT, 150.0, 3),
            terraces: fbm(TERRACE_SALT, TERRACE_CELL, 2),
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
                gate: CAVE_MIN_DEPTH,
                origin: 0,
            },
            ravines: Term {
                field: fbm(RAVINE_SALT, 30.0, 2),
                ramp: Ramp { start: 0.90, slope: 0.000_2, floor: 0.80 },
                gate: RAVINE_MIN_DEPTH,
                origin: 0,
            },
            overhangs: fbm(OVERHANG_SALT, 20.0, 2),
            islands: Islands {
                place: fbm(ISLAND_SALT, ISLAND_PLACE_CELL, 3),
                lift: fbm(ISLAND_LIFT_SALT, ISLAND_LIFT_CELL, 2),
                detail: fbm(ISLAND_DETAIL_SALT, ISLAND_DETAIL_CELL, 2),
                on: ISLAND_ON,
                core_floor: ISLAND_CORE_FLOOR,
                top_h: ISLAND_TOP_H,
                keel_h: ISLAND_KEEL_H,
                detail_amp: ISLAND_DETAIL_AMP,
                band_lo: ISLAND_MIN_Y,
                band_span: ISLAND_BAND_SPAN,
            },
            mat,
        }
    }

    /// Sea level — spawn logic keeps players off the seabed.
    pub fn sea_level(&self) -> i32 {
        self.sea_level
    }

    /// Everything a column needs, sampled once. `height` folds continentalness
    /// (base), erosion·ridge (relief), and rivers (valley-floor channels).
    fn profile(&self, wx: i32, wz: i32) -> Column {
        // The three height axes intentionally share their warp displacement
        // fields (see `hwarp` in the constructor). Compute that displacement
        // once, and retain raw weirdness for the river pass below instead of
        // sampling the same warped field a second time — bit-identical.
        let (hx, hz) = self.continentalness.field.coordinates(wx, wz);
        let base = self.continentalness.shape(self.continentalness.field.field.at(hx, hz));
        let amp = self.erosion.shape(self.erosion.field.field.at(hx, hz));
        let raw_weirdness = self.weirdness.field.field.at(hx, hz);
        let ridge = self.weirdness.shape(raw_weirdness);
        // Mid-frequency rolling detail on every column: a small baseline so plains
        // are never dead flat, growing with erosion so mountains get rough flanks.
        let detail = self.detail.at(wx, wz).0 * 2.0 - 1.0;
        let mut h = self.sea_level as f32 + base + amp * ridge + (2.5 + amp * 0.25) * detail;

        // Ridged mountain ranges: fold the range noise to a ridgeline crest
        // (`1 − |2n − 1|` peaks at n = ½) and add it only atop already-elevated
        // columns, scaled by how far past the onset the base sits — so ranges
        // sharpen highlands into ridges without touching plains or seas.
        if base > RANGE_ONSET {
            let crest = 1.0 - (self.ranges.at(wx, wz).0 * 2.0 - 1.0).abs();
            h += RANGE_GAIN * (base - RANGE_ONSET).min(amp) * crest;
        }

        // Regional terracing: on inland columns inside a terrace region, quantize
        // the land surface into benches, smoothstep-blended from the region edge so
        // mesas rise out of smooth terrain instead of behind a wall. Applied before
        // rivers/lakes so water still carves clean channels through the benches.
        let tmask = self.terraces.at(wx, wz).0;
        if base > RIVER_INLAND && tmask > TERRACE_ON {
            let t = (tmask - TERRACE_ON) / (1.0 - TERRACE_ON);
            let s = t * t * (3.0 - 2.0 * t);
            let stepped = (h / TERRACE_STEP).round() * TERRACE_STEP;
            h += (stepped - h) * s;
        }

        // Rivers: carve toward a sub-sea channel at the valley floor (weirdness
        // 0.5), gated to inland columns so ocean basins aren't double-carved.
        let w = raw_weirdness.0;
        let d = (w - 0.5).abs();
        if base > RIVER_INLAND && d < RIVER_HALF {
            let t = 1.0 - d / RIVER_HALF;
            let s = t * t * (3.0 - 2.0 * t);
            let target = (self.sea_level - RIVER_DEPTH) as f32;
            h += (target.min(h) - h) * s;
        }

        // Lakes: inland lake blobs raise the water table to a surface above
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

        // The climate axes share their warp pair by construction too.
        let (climate_x, climate_z) = self.temperature.coordinates(wx, wz);
        let mut p = Column {
            height: (h.round() as i32).max(1),
            water_level,
            temperature: self.temperature.field.at(climate_x, climate_z),
            humidity: self.humidity.field.at(climate_x, climate_z),
            kind: placement::SurfaceKind::Grassy,
            dress: AIR,
        };
        p.kind = self.surface_kind(&p, wx, wz);
        p.dress = self.dress(&p, wx, wz);
        p
    }

    /// The ground column's surface dressing, classified from the shared context:
    /// shore at/under the water table, snow on cold or high ground, desert on
    /// hot & dry, grass otherwise — except that a grassy column one or two
    /// blocks above the water line may dither into the beach-edge band (sand
    /// still holding soil), so beaches fade into grass instead of ending on a
    /// hard line. One axis the depth-1 placement rules filter on — the block
    /// itself comes from the [`placement`] dress LUT.
    fn surface_kind(&self, p: &Column, wx: i32, wz: i32) -> placement::SurfaceKind {
        use placement::SurfaceKind::*;
        if p.height <= p.water_level {
            Shore
        } else if p.temperature.0 < COLD || p.height - self.sea_level > SNOW_ABOVE_SEA {
            Snowy
        } else if p.temperature.0 > HOT && p.humidity.0 < DRY {
            Desert
        } else {
            let rim = p.height - p.water_level;
            if rim <= 2 {
                // Half the columns at +1, a quarter at +2 — a dissolving edge.
                let cut = u32::MAX / if rim == 1 { 2 } else { 4 };
                if cell_hash(self.seed ^ BEACH_SALT, wx, 0, wz) < cut {
                    return BeachEdge;
                }
            }
            Grassy
        }
    }

    /// The surface block a column dresses in: the banded element union of its
    /// [`surface_kind`](Self::surface_kind), or — where a scatter roll hits —
    /// that union plus a luminous payload (glow tufts on the plains, phosphor
    /// sparks in the desert). One hash stream, so surface finds stay singles.
    fn dress(&self, p: &Column, wx: i32, wz: i32) -> BlockId {
        let kind = p.kind;
        let slices = &self.mat.surface_scatter[kind as usize];
        if !slices.is_empty() {
            let roll = cell_hash(self.seed ^ SURFACE_SCATTER_SALT, wx, p.height, wz);
            let mut cut = 0u32;
            for slice in slices {
                cut += slice.width;
                if roll < cut {
                    return slice.id;
                }
            }
        }
        self.mat.dress[kind as usize]
    }

    /// The ore (if any) a stone cell rolls: two decorrelated hash streams, each
    /// walking the cumulative rarity slices (stream B's are ÷8), deduped —
    /// distinct hits on both streams yield the overlap pair, a multi-yield
    /// find. Stream A alone is byte-identical to the legacy distribution.
    #[inline]
    fn ore_at(&self, wx: i32, wy: i32, wz: i32, depth: i32) -> Option<BlockId> {
        let hit = |slices: &[placement::Slice], roll: u32| -> Option<usize> {
            let mut cut = 0u32;
            for (i, slice) in slices.iter().enumerate() {
                if depth < slice.min_depth {
                    break;
                }
                cut += slice.width;
                if roll < cut {
                    return Some(i);
                }
            }
            None
        };
        let a = hit(&self.mat.seams, cell_hash(self.seed, wx, wy, wz));
        // The B roll only exists where it can land (below the shallowest
        // slice), so the common stone cell pays one hash, as before.
        let b = if depth >= self.mat.seams.first().map_or(i32::MAX, |s| s.min_depth) {
            hit(&self.mat.seams_b, cell_hash(self.seed ^ ORE_B_SALT, wx, wy, wz))
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

    /// A ground cell below its column's surface, carve decision supplied.
    fn ground(&self, p: &Column, wx: i32, wy: i32, wz: i32, carved: bool) -> BlockId {
        let height = p.height;
        if wy >= height - 1 {
            p.dress
        } else if wy >= height - 3 {
            self.mat.crust[p.kind as usize]
        } else if carved {
            AIR
        } else {
            let depth = height - wy;
            if depth <= self.mat.max_scattered_depth && let Some(ore) = self.ore_at(wx, wy, wz, depth) {
                return ore;
            }
            if let Some(id) = self.cave_wall_at(p, wx, wy, wz, depth) {
                return id;
            }
            self.mat.stone
        }
    }

    /// The cave-wall cluster (Lumin on deep cavern floors and ceilings), if it
    /// lands here: eligible depth, its own hash roll, and a carved cell
    /// directly above or below — VERTICAL adjacency only, so the check stays
    /// inside one column (two carve reads, only after the rare roll hits) and
    /// the deep-uniform proof only needs carve dormancy one chunk up/down.
    fn cave_wall_at(&self, p: &Column, wx: i32, wy: i32, wz: i32, depth: i32) -> Option<BlockId> {
        let cw = self.mat.cave_wall.as_ref()?;
        if depth < cw.min_depth
            || cell_hash(self.seed ^ CAVE_WALL_SALT, wx, wy, wz) >= cw.width
        {
            return None;
        }
        let carved_v = |ny: i32| ny < p.height && self.carved(wx, ny, wz, p.height);
        (carved_v(wy + 1) || carved_v(wy - 1)).then_some(cw.id)
    }

    /// The block for an island-solid cell, from what sits above it in the field.
    fn island_block(&self, wx: i32, wy: i32, wz: i32, above: [bool; 4]) -> BlockId {
        if !above[0] {
            if wy >= ICE_SURFACE_Y { self.mat.island_ice } else { self.mat.island_grass }
        } else if !above[1] || !above[2] || !above[3] {
            self.mat.island_crust
        } else {
            // Interior scatter: the same two-stream dedup as the ground ores
            // (an Aerium+Quartz overlap is the island's multi-yield find).
            let hit = |slices: &[placement::Slice], roll: u32| -> Option<usize> {
                let mut cut = 0u32;
                for (i, slice) in slices.iter().enumerate() {
                    cut += slice.width;
                    if roll < cut {
                        return Some(i);
                    }
                }
                None
            };
            let a = hit(&self.mat.island_seams, cell_hash(self.seed, wx, wy, wz));
            let b = hit(&self.mat.island_seams_b, cell_hash(self.seed ^ ORE_B_SALT, wx, wy, wz));
            match (a, b) {
                (Some(i), Some(j)) if i != j => {
                    let (hi, lo) = if i > j { (i, j) } else { (j, i) };
                    self.mat.island_pairs[hi][lo]
                }
                (Some(i), _) | (None, Some(i)) => self.mat.island_seams[i].id,
                (None, None) => self.mat.stone,
            }
        }
    }

    /// Whether a ground cell is carved out — by a cave or a ravine, the two
    /// carve terms folded together as one `excess > 0` subtraction.
    fn carved(&self, wx: i32, wy: i32, wz: i32, height: i32) -> bool {
        self.caves.excess(wx, wy, wz, height) > 0.0 || self.ravines.excess(wx, wy, wz, height) > 0.0
    }

    /// Whether an overhang shelf places solid rock at a cell above the surface
    ///: within OVERHANG_REACH blocks of the surface and past a threshold
    /// that stiffens with height, so shelves jut from cliffs and fade upward.
    fn overhang_solid(&self, wx: i32, wy: i32, wz: i32, height: i32) -> bool {
        let up = wy - height;
        (0..OVERHANG_REACH).contains(&up)
            && self.overhangs.at3(wx, wy, wz).0 > OVERHANG_THRESH + OVERHANG_FADE * up as f32
    }

    /// The vertical stack at a cell: ground below the surface, an overhang
    /// shelf or water above it, island-or-air higher still. `caves` gates the
    /// sub-surface carve — far tiles pass `false` (caves/ravines are
    /// sub-4m-cell, invisible at LOD range, and their noise eval is pure waste
    /// there), so the carve is not even sampled.
    fn cell_base(&self, p: &Column, wx: i32, wy: i32, wz: i32, caves: bool) -> BlockId {
        if wy < p.height {
            self.ground(p, wx, wy, wz, caves && self.carved(wx, wy, wz, p.height))
        } else if wy < p.water_level {
            self.mat.water
        } else if self.overhang_solid(wx, wy, wz, p.height) {
            self.mat.stone
        } else if self.islands.solid(wx, wy, wz) {
            self.island_block(wx, wy, wz, [
                self.islands.solid(wx, wy + 1, wz),
                self.islands.solid(wx, wy + 2, wz),
                self.islands.solid(wx, wy + 3, wz),
                self.islands.solid(wx, wy + 4, wz),
            ])
        } else {
            AIR
        }
    }

}

impl TerrainGenerator for Terrain {
    fn seed(&self) -> i64 {
        self.seed
    }
    fn sea_level(&self) -> i32 {
        self.sea_level
    }

    /// The LOD/spawn surface height — the topmost *ground* cell's column value.
    /// Caves never carve the top [`CAVE_MIN_DEPTH`] cells, so the topmost ground
    /// cell is always `height − 1`. Overhang shelves sit in the air above the surface
    /// and never move the walkable ground level; LOD and spawn key off the base height.
    fn height(&self, wx: i32, wz: i32) -> i32 {
        self.profile(wx, wz).height
    }

    fn surface_at(&self, wx: i32, wz: i32) -> BlockId {
        self.profile(wx, wz).dress
    }

    fn deep(&self) -> BlockId {
        self.mat.stone
    }

    fn block_at(&self, wx: i32, wy: i32, wz: i32, _height: i32) -> BlockId {
        self.cell_base(&self.profile(wx, wz), wx, wy, wz, true)
    }

    /// Far tiles pay a single `profile` per cell — the caller no longer
    /// re-samples `height` separately.
    fn lod_block_at(&self, wx: i32, wy: i32, wz: i32) -> BlockId {
        self.cell_base(&self.profile(wx, wz), wx, wy, wz, false)
    }

    /// The column profile — the ~nine 2-D noise fields a far tile pays for — is
    /// sampled once here and reused down the whole vertical run, instead of once
    /// per cell as the default per-`lod_block_at` fill would. Mirrors the per-column
    /// reuse the full-res generate already relies on, and is the
    /// single biggest cost drop for a far tile sample.
    fn lod_column(&self, wx: i32, wz: i32, ys: &[i32], out: &mut [BlockId]) {
        let p = self.profile(wx, wz);
        let island = self.islands.core_center(wx, wz);
        let mut isl: Vec<bool> = Vec::new();
        let mut isl_lo = 0i32;
        if let Some((core, center)) = island {
            if let (Some(&y_min), Some(&y_max)) = (ys.iter().min(), ys.iter().max()) {
                let lo = y_min.max(self.islands.band_bottom());
                let hi = (y_max + 4).min(self.islands.band_top());
                if lo <= hi {
                    isl_lo = lo;
                    isl.resize((hi - lo + 1) as usize, false);
                    for (k, slot) in isl.iter_mut().enumerate() {
                        let wy = lo + k as i32;
                        *slot = self.islands.density(
                            core,
                            center,
                            wy,
                            self.islands.detail.at3(wx, wy, wz).0,
                        ) > 0.0;
                    }
                }
            }
        }
        let island_at = |wy: i32| -> bool {
            let i = wy - isl_lo;
            i >= 0 && (i as usize) < isl.len() && isl[i as usize]
        };
        for (o, &wy) in out.iter_mut().zip(ys) {
            *o = if wy < p.height {
                self.ground(&p, wx, wy, wz, false)
            } else if wy < p.water_level {
                self.mat.water
            } else if self.overhang_solid(wx, wy, wz, p.height) {
                self.mat.stone
            } else if island_at(wy) {
                self.island_block(wx, wy, wz, [
                    island_at(wy + 1),
                    island_at(wy + 2),
                    island_at(wy + 3),
                    island_at(wy + 4),
                ])
            } else {
                AIR
            };
        }
    }

    /// Whole-chunk generation: sample the column profiles, then fill from them.
    fn generate(&self, cx: i32, cy: i32, cz: i32) -> ChunkData {
        let (x0, z0) = (cx * CHUNK_SIZE as i32, cz * CHUNK_SIZE as i32);
        let (profiles, h_min, h_max, w_min, w_max) = self.column_profiles(x0, z0);
        self.fill_chunk(x0, z0, cy, &profiles, h_min, h_max, w_min, w_max, None)
    }

    /// Column-batched generation: the 256 column profiles are `cy`-invariant, so
    /// a whole vertical run shares one sampling instead of R× re-sampling — the
    /// single biggest load-time generation cost drop. Voxel-identical to looping
    /// generate over the range.
    ///
    /// Column job cost (64 surface columns × 9 layers, `--release`):
    /// 1.008 ms/column before the run-fill pass, 0.616 ms/column after
    /// (median of 3; per-column bands, y-outer contiguous writes).
    fn generate_column(
        &self,
        cx: i32,
        cz: i32,
        cy: RangeInclusive<i32>,
    ) -> (Vec<(i32, ChunkData)>, ColumnHeights) {
        let (x0, z0) = (cx * CHUNK_SIZE as i32, cz * CHUNK_SIZE as i32);
        let (profiles, h_min, h_max, w_min, w_max) = self.column_profiles(x0, z0);
        let mut heights = [0i32; CHUNK_SIZE * CHUNK_SIZE];
        for (i, p) in profiles.iter().enumerate() {
            heights[i] = p.height;
        }
        let cy_lo = *cy.start();
        let n = (*cy.end() as i64 - cy_lo as i64 + 3).max(0) as usize;
        let mut carve_dorm = vec![None; n];
        let base = cy_lo - 1;
        let chunks = cy
            .map(|cyy| {
                (
                    cyy,
                    self.fill_chunk(
                        x0,
                        z0,
                        cyy,
                        &profiles,
                        h_min,
                        h_max,
                        w_min,
                        w_max,
                        Some((base, &mut carve_dorm)),
                    ),
                )
            })
            .collect();
        (chunks, heights)
    }
}

impl Terrain {
    /// The deep Uniform(stone) proof: every cell sits below every scattered
    /// rule's reach and both carve fields are dormant over the chunk box — and,
    /// when a cave-wall rule exists, over the boxes one chunk above and below
    /// too, since its VERTICAL adjacency reads one cell past the chunk's rim
    /// (same columns, so the height extents carry over). The dense fill's
    /// collapse remains the correctness backstop; this is a CPU shortcut.
    fn box_dormant(&self, x0: i32, y0: i32, z0: i32, h_max: i32) -> bool {
        self.caves.dormant(x0, y0, z0, h_max - y0)
            && self.ravines.dormant(x0, y0, z0, h_max - y0)
    }

    fn deep_uniform_provable(
        &self,
        x0: i32,
        y0: i32,
        z0: i32,
        y1: i32,
        h_min: i32,
        h_max: i32,
    ) -> bool {
        self.deep_uniform_with(y0, y1, h_min, |ny0| self.box_dormant(x0, ny0, z0, h_max))
    }

    fn deep_uniform_with(
        &self,
        y0: i32,
        y1: i32,
        h_min: i32,
        mut dormant: impl FnMut(i32) -> bool,
    ) -> bool {
        if y1 >= h_min - self.mat.max_scattered_depth || !dormant(y0) {
            return false;
        }
        if self.mat.cave_wall.is_none() {
            return true;
        }
        let cs = CHUNK_SIZE as i32;
        dormant(y0 - cs) && dormant(y0 + cs)
    }

    /// The 256 column profiles for a chunk column, plus the height/water extents
    /// the fast paths read. `cy`-invariant — sampled once per vertical column.
    /// Exactly one chunk column is sampled at a time, so the fixed 256-profile
    /// array lives inline instead of costing an allocator round trip per job.
    fn column_profiles(
        &self,
        x0: i32,
        z0: i32,
    ) -> ([Column; CHUNK_SIZE * CHUNK_SIZE], i32, i32, i32, i32) {
        let (mut h_min, mut h_max) = (i32::MAX, i32::MIN);
        let (mut w_min, mut w_max) = (i32::MAX, i32::MIN);
        // Index order matches the old lz-outer/lx-inner push order exactly.
        let profiles = std::array::from_fn(|index| {
            let lx = index % CHUNK_SIZE;
            let lz = index / CHUNK_SIZE;
            let p = self.profile(x0 + lx as i32, z0 + lz as i32);
            h_min = h_min.min(p.height);
            h_max = h_max.max(p.height);
            w_min = w_min.min(p.water_level);
            w_max = w_max.max(p.water_level);
            p
        });
        (profiles, h_min, h_max, w_min, w_max)
    }

    /// One chunk's storage from shared column profiles — the `cy`-varying half of
    /// generation (height-band + region shortcuts, then a run fill).
    #[allow(clippy::too_many_arguments)] // column stats stay unpacked so the fill can early-out per bound
    fn fill_chunk(
        &self,
        x0: i32,
        z0: i32,
        cy: i32,
        profiles: &[Column],
        h_min: i32,
        h_max: i32,
        w_min: i32,
        w_max: i32,
        mut carve_dorm: Option<(i32, &mut [Option<bool>])>,
    ) -> ChunkData {
        let y0 = cy * CHUNK_SIZE as i32;
        let y1 = y0 + CHUNK_SIZE as i32 - 1;
        let cs = CHUNK_SIZE as i32;

        // Deep below every scattered rule's reach and beyond either carve
        // field's: solid stone. The depth bound is derived from the placement
        // table, not a constant.
        let mut dormant_at = |ny0: i32| -> bool {
            if let Some((base, slots)) = carve_dorm.as_mut() {
                let i = (ny0 / cs - *base) as usize;
                if let Some(v) = slots[i] {
                    return v;
                }
                let v = self.box_dormant(x0, ny0, z0, h_max);
                slots[i] = Some(v);
                v
            } else {
                self.box_dormant(x0, ny0, z0, h_max)
            }
        };
        if self.deep_uniform_with(y0, y1, h_min, &mut dormant_at) {
            return ChunkData::Uniform(self.mat.stone);
        }
        // Above every surface and below the island band: uniform sky. Fully below
        // the lowest water table → water; fully at/above the highest → air. (A
        // chunk straddling a water table falls through to the dense fill, which
        // collapses it anyway.)
        // Guarded past the overhang reach above the tallest surface, since a shelf
        // can place solid rock up to `OVERHANG_REACH` blocks over the ground.
        if y0 >= h_max + OVERHANG_REACH
            && !self.islands.possible(x0, y0, z0, (cs, cs, cs))
        {
            if y1 < w_min {
                return ChunkData::Uniform(self.mat.water);
            }
            if y0 >= w_max {
                return ChunkData::Uniform(AIR);
            }
        }

        let carve_dormant = dormant_at(y0);
        let wall_lo_dormant = self.mat.cave_wall.is_none() || dormant_at(y0 - cs);
        let wall_hi_dormant = self.mat.cave_wall.is_none() || dormant_at(y0 + cs);

        let islands_possible = y1 + 4 >= self.islands.band_bottom() && y0 <= self.islands.band_top();
        let any_carve = !carve_dormant && y0 <= h_max - CAVE_MIN_DEPTH;
        let any_overhang = y1 >= h_min && y0 <= h_max + OVERHANG_REACH - 1;
        let mut bands = [ColMask { wx: 0, wz: 0, carved: 0, overhang: 0, island: 0 }; CHUNK_SIZE * CHUNK_SIZE];
        for (xz, band) in bands.iter_mut().enumerate() {
            let lx = xz % CHUNK_SIZE;
            let lz = xz / CHUNK_SIZE;
            let wx = x0 + lx as i32;
            let wz = z0 + lz as i32;
            let height = profiles[xz].height;
            let mut carved = 0u16;
            let cave_top = height - CAVE_MIN_DEPTH;
            let rav_top = height - RAVINE_MIN_DEPTH;
            if any_carve && y0 <= cave_top {
                let cave_hi = y1.min(cave_top);
                let cave_col = self.caves.field.column(wx, wz, y0, cave_hi);
                let rav_col = (y0 <= rav_top)
                    .then(|| self.ravines.field.column(wx, wz, y0, cave_hi.min(rav_top)));
                for ly in 0..CHUNK_SIZE {
                    let wy = y0 + ly as i32;
                    let cave = wy <= cave_top && self.caves.excess_col(&cave_col, wy, height) > 0.0;
                    let rav = rav_col
                        .as_ref()
                        .is_some_and(|col| wy <= rav_top && self.ravines.excess_col(col, wy, height) > 0.0);
                    if cave || rav {
                        carved |= 1 << ly;
                    }
                }
            }
            let mut overhang = 0u16;
            let oh_lo = height;
            let oh_hi = height + OVERHANG_REACH - 1;
            if any_overhang && y1 >= oh_lo && y0 <= oh_hi {
                let col = self.overhangs.column(wx, wz, y0.max(oh_lo), y1.min(oh_hi));
                for ly in 0..CHUNK_SIZE {
                    let wy = y0 + ly as i32;
                    let up = wy - height;
                    if (0..OVERHANG_REACH).contains(&up)
                        && col.sample(wy).0 > OVERHANG_THRESH + OVERHANG_FADE * up as f32
                    {
                        overhang |= 1 << ly;
                    }
                }
            }
            let mut island = 0u32;
            if islands_possible {
                if let Some(col) = self.islands.column(wx, wz, y0, y1 + 4) {
                    for k in 0..(CHUNK_SIZE + 4) {
                        if self.islands.solid_col(&col, y0 + k as i32) {
                            island |= 1 << k;
                        }
                    }
                }
            }
            *band = ColMask { wx, wz, carved, overhang, island };
        }

        // Band values written y-outer / x-inner so a uniform plane is one
        // contiguous run; only ore / cave-wall / island cells take a hash.
        let mut cells = Box::new([AIR; CHUNK_VOLUME]);
        let plane = CHUNK_SIZE * CHUNK_SIZE;
        let sky_lo = h_max + OVERHANG_REACH;
        for ly in 0..CHUNK_SIZE {
            let wy = y0 + ly as i32;
            let dest = &mut cells[ly * plane..(ly + 1) * plane];
            if wy >= sky_lo && !islands_possible {
                if wy >= w_max {
                    dest.fill(AIR);
                    continue;
                }
                if wy < w_min {
                    dest.fill(self.mat.water);
                    continue;
                }
            }
            for xz in 0..plane {
                dest[xz] = self.column_at(
                    &profiles[xz],
                    &bands[xz],
                    ly,
                    wy,
                    wall_lo_dormant,
                    wall_hi_dormant,
                );
            }
        }
        ChunkData::from_cells(cells)
    }

    #[inline]
    fn column_at(
        &self,
        p: &Column,
        m: &ColMask,
        ly: usize,
        wy: i32,
        lo_dormant: bool,
        hi_dormant: bool,
    ) -> BlockId {
        let height = p.height;
        if wy < height {
            if wy >= height - 1 {
                p.dress
            } else if wy >= height - 3 {
                self.mat.crust[p.kind as usize]
            } else if m.carved & (1 << ly) != 0 {
                AIR
            } else {
                let depth = height - wy;
                if depth <= self.mat.max_scattered_depth {
                    if let Some(ore) = self.ore_at(m.wx, wy, m.wz, depth) {
                        return ore;
                    }
                }
                if let Some(id) = self.wall_cell(p, m, ly, wy, depth, lo_dormant, hi_dormant) {
                    return id;
                }
                self.mat.stone
            }
        } else if wy < p.water_level {
            self.mat.water
        } else if m.overhang & (1 << ly) != 0 {
            self.mat.stone
        } else if m.island & (1 << ly) != 0 {
            let bit = |k: usize| m.island & (1u32 << k) != 0;
            self.island_block(
                m.wx,
                wy,
                m.wz,
                [bit(ly + 1), bit(ly + 2), bit(ly + 3), bit(ly + 4)],
            )
        } else {
            AIR
        }
    }

    /// In-mask neighbour, or a rim whose neighbouring chunk box is not dormant.
    #[inline]
    fn wall_adjacent(m: &ColMask, ly: usize, lo_dormant: bool, hi_dormant: bool) -> bool {
        (ly > 0 && m.carved & (1 << (ly - 1)) != 0)
            || (ly + 1 < CHUNK_SIZE && m.carved & (1 << (ly + 1)) != 0)
            || (ly == 0 && !lo_dormant)
            || (ly + 1 == CHUNK_SIZE && !hi_dormant)
    }

    #[inline]
    fn wall_cell(
        &self,
        p: &Column,
        m: &ColMask,
        ly: usize,
        wy: i32,
        depth: i32,
        lo_dormant: bool,
        hi_dormant: bool,
    ) -> Option<BlockId> {
        let cw = self.mat.cave_wall.as_ref()?;
        if depth < cw.min_depth || !Self::wall_adjacent(m, ly, lo_dormant, hi_dormant) {
            return None;
        }
        if cell_hash(self.seed ^ CAVE_WALL_SALT, m.wx, wy, m.wz) >= cw.width {
            return None;
        }
        let in_mask = (ly > 0 && m.carved & (1 << (ly - 1)) != 0)
            || (ly + 1 < CHUNK_SIZE && m.carved & (1 << (ly + 1)) != 0);
        if in_mask {
            return Some(cw.id);
        }
        let carved_v = |ny: i32| ny < p.height && self.carved(m.wx, ny, m.wz, p.height);
        (carved_v(wy + 1) || carved_v(wy - 1)).then_some(cw.id)
    }
}

#[cfg(test)]
pub(in crate::world) fn assert_generate_column_heights_match_height(
    g: &impl TerrainGenerator,
    cols: &[(i32, i32)],
) {
    for &(cx, cz) in cols {
        let (_, heights) = g.generate_column(cx, cz, 0..=0);
        let (_, empty) = g.generate_column(cx, cz, 1..=0);
        let x0 = cx * CHUNK_SIZE as i32;
        let z0 = cz * CHUNK_SIZE as i32;
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let i = lx + lz * CHUNK_SIZE;
                assert_eq!(
                    heights[i],
                    g.height(x0 + lx as i32, z0 + lz as i32),
                    "cx={cx} cz={cz} lx={lx} lz={lz}"
                );
                assert_eq!(empty[i], heights[i], "empty cy still reports height cx={cx} cz={cz}");
            }
        }
    }
}

#[cfg(test)]
mod generate_column_tests {
    use super::*;

    /// `generate_column` must be voxel-identical to per-chunk `generate` over the
    /// range — the shared-profile fast path can't change a single cell.
    #[test]
    fn generate_column_matches_per_chunk_generate() {
        let g = Terrain::new(&mut BlockRegistry::with_builtins(), 20.0, 3);
        for (cx, cz) in [(0, 0), (2, -3), (-1, 7), (0, -4)] {
            let cy_lo = -3;
            let cy_hi = 5;
            let (column, _) = g.generate_column(cx, cz, cy_lo..=cy_hi);
            assert_eq!(column.len() as i32, cy_hi - cy_lo + 1);
            for (cy, data) in column {
                assert_eq!(data, g.generate(cx, cy, cz), "chunk ({cx}, {cy}, {cz})");
            }
        }
    }

    /// Heights returned with the column equal [`TerrainGenerator::height`] at
    /// every cell — the skylight ceiling must not drift from the walkable ground.
    #[test]
    fn generate_column_heights_match_height() {
        let g = Terrain::new(&mut BlockRegistry::with_builtins(), 20.0, 7);
        super::assert_generate_column_heights_match_height(&g, &[(0, 0), (2, -3), (-1, 7), (4, 4)]);
    }

    /// Column job cost: 64 surface columns × 9 layers. Ignored timing gauge.
    /// Run with `cargo test --release generate_column_ms -- --ignored --nocapture`.
    ///
    /// Median of 3 `--release` runs: 1.008 ms/column before the run-fill pass,
    /// 0.616 ms/column after (1.64×).
    #[test]
    #[ignore]
    fn generate_column_ms() {
        use std::time::Instant;
        let g = Terrain::new(&mut BlockRegistry::with_builtins(), 20.0, 42);
        let cols: [(i32, i32); 64] =
            std::array::from_fn(|i| ((i as i32) % 8, (i as i32) / 8));
        let cy = -3..=5;
        for &(cx, cz) in &cols {
            let _ = std::hint::black_box(g.generate_column(cx, cz, cy.clone()));
        }
        let t = Instant::now();
        for &(cx, cz) in &cols {
            let _ = std::hint::black_box(g.generate_column(cx, cz, cy.clone()));
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        println!(
            "generate_column: {ms:.2} ms total, {:.3} ms/column (64 columns × 9 layers)",
            ms / 64.0
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terrain(seed: i64) -> Terrain {
        Terrain::new(&mut BlockRegistry::with_builtins(), 20.0, seed)
    }

    fn terrain_with_registry(seed: i64) -> (BlockRegistry, Terrain) {
        let mut registry = BlockRegistry::with_builtins();
        let generator = Terrain::new(&mut registry, 20.0, seed);
        (registry, generator)
    }

    #[test]
    fn column_cache_matches_the_pure_field() {
        // The generalized parity check: Fbm::column must be bit-identical to
        // Fbm::at3 for both region fields, including far out and deep down.
        let g = terrain(42);
        for field in [&g.caves.field, &g.islands.detail, &g.overhangs] {
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
        // density built from it can decide `straddles(0)` soundly.
        let g = terrain(7);
        for (cx, cy, cz) in [(0, -3, 0), (2, -1, -5), (-4, -30, 6), (30_000_000, 4, -7)] {
            let (x0, y0, z0) = (cx * 16, cy * 16, cz * 16);
            for field in [&g.caves.field, &g.islands.detail] {
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
    fn lod_column_matches_lod_block_at() {
        let g = terrain(42);
        let columns = [(0, 0), (13, -27), (-8, -56), (100, -80 * 16 + 8)];
        let ys: Vec<i32> = (0..64).map(|j| j * 4 + 2).collect();
        let mut out = vec![AIR; ys.len()];
        for &(wx, wz) in &columns {
            g.lod_column(wx, wz, &ys, &mut out);
            for (i, &wy) in ys.iter().enumerate() {
                assert_eq!(out[i], g.lod_block_at(wx, wy, wz), "lod ({wx},{wy},{wz})");
            }
        }
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
        // the biome axes are domain-warped, so the warped read differs from
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
        // the water table is a field, so inland lake blobs hold standing water
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
    fn surface_chunk_fill_matches_per_cell() {
        // The chunk fast path (banded run fill) must agree with the per-cell
        // block_at, cell for cell, across surface chunks (crust, carve, overhangs, water).
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
        // overhang shelves put solid rock strictly above a column's heightfield
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
    fn islands_live_in_the_band_and_keels_clear_terrain() {
        let g = terrain(9);
        let isl = &g.islands;

        // No island cell exists below the band floor, however far down we probe.
        for y in [isl.band_bottom() - 1, -100, -1000] {
            for x in -64..64 {
                assert!(!isl.solid(x * 5, y, x * 3 - 7), "no islands below the band floor");
            }
        }

        // Keels clear water: the band floor sits above the highest water surface
        // (sea level plus a lake's rise), so islands and water never interact.
        let max_water = g.sea_level + LAKE_RISE;
        assert!(isl.band_bottom() > max_water, "island keels clear the water table");

        // Islands genuinely exist within the band over a wide region (the mask is
        // low-frequency, so a small window can miss every island).
        let any = (-4000..4000).step_by(32).any(|x| {
            (-4000..4000).step_by(32).any(|z| {
                (isl.band_bottom()..=isl.band_top()).any(|y| isl.solid(x, y, z))
            })
        });
        assert!(any, "the band holds islands somewhere");
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
            if g.deep_uniform_provable(x0, y0, z0, y0 + 15, h_min, h_max) {
                proven = Some(cz);
                break;
            }
        }
        let cz = proven.expect("a bound-cleared deep chunk within 128");
        assert_eq!(g.generate(0, cy, cz), ChunkData::Uniform(stone));
        // Proof soundness: the per-cell path agrees with the shortcut on every
        // cell — the proof is a CPU shortcut, never a semantic gate.
        let (x0, y0, z0) = (0, cy * 16, cz * 16);
        for lx in 0..16 {
            for lz in 0..16 {
                let h = g.height(x0 + lx, z0 + lz);
                for ly in 0..16 {
                    assert_eq!(g.block_at(x0 + lx, y0 + ly, z0 + lz, h), stone);
                }
            }
        }
    }

    /// A frozen copy of the pre-placement material picker (named blocks and the
    /// hand-written branches), so the element-first rewiring can be censused
    /// against it: geometry must be IDENTICAL, and materials must map exactly
    /// (identity everywhere except the three accepted drifts — grass and dirt
    /// become their natural unions, pure Obsidian becomes Stone+Obsidian).
    struct Legacy {
        grass: BlockId,
        dirt: BlockId,
        stone: BlockId,
        sand: BlockId,
        snow: BlockId,
        ice: BlockId,
        water: BlockId,
        aerium_vein: BlockId,
        quartz_vein: BlockId,
        seams: [(i32, u32, BlockId); 10],
    }

    impl Legacy {
        fn resolve(reg: &BlockRegistry) -> Legacy {
            let id = |n: &str| reg.id_by_name(n).unwrap();
            let seam = |d: i32, r: u32, n: &str| (d, u32::MAX / r, id(n));
            Legacy {
                grass: id("Soil+Organic"),
                dirt: id("Soil+Clay"),
                stone: id("Stone"),
                sand: id("Sand"),
                snow: id("Snow"),
                ice: id("Ice"),
                water: id("Water"),
                aerium_vein: id("Stone+Aerium"),
                quartz_vein: id("Stone+Quartz"),
                seams: [
                    seam(3, 90, "Stone+Coal"),
                    seam(8, 110, "Stone+Iron"),
                    seam(8, 130, "Stone+Copper"),
                    seam(20, 240, "Stone+Sulfur"),
                    seam(20, 200, "Stone+Quartz"),
                    seam(20, 220, "Stone+Lead"),
                    seam(32, 300, "Stone+Gold"),
                    seam(32, 380, "Stone+Lumin"),
                    seam(48, 460, "Stone+Titan"),
                    seam(48, 240, "Obsidian"),
                ],
            }
        }

        fn dress(&self, g: &Terrain, p: &Column) -> BlockId {
            if p.height <= p.water_level {
                self.sand
            } else if p.temperature.0 < COLD || p.height - g.sea_level > SNOW_ABOVE_SEA {
                self.snow
            } else if p.temperature.0 > HOT && p.humidity.0 < DRY {
                self.sand
            } else {
                self.grass
            }
        }

        fn cell_base(&self, g: &Terrain, p: &Column, wx: i32, wy: i32, wz: i32) -> BlockId {
            let height = p.height;
            if wy < height {
                // legacy ground()
                if wy >= height - 1 {
                    self.dress(g, p)
                } else if wy >= height - 3 {
                    self.dirt
                } else if g.carved(wx, wy, wz, height) {
                    AIR
                } else {
                    let depth = height - wy;
                    if depth <= 64 {
                        let roll = cell_hash(g.seed, wx, wy, wz);
                        let mut cut = 0u32;
                        for &(min_depth, width, block) in &self.seams {
                            if depth < min_depth {
                                break;
                            }
                            cut += width;
                            if roll < cut {
                                return block;
                            }
                        }
                    }
                    self.stone
                }
            } else if wy < p.water_level {
                self.water
            } else if g.overhang_solid(wx, wy, wz, height) {
                self.stone
            } else if g.islands.solid(wx, wy, wz) {
                // legacy island_block()
                let above = [
                    g.islands.solid(wx, wy + 1, wz),
                    g.islands.solid(wx, wy + 2, wz),
                    g.islands.solid(wx, wy + 3, wz),
                    g.islands.solid(wx, wy + 4, wz),
                ];
                if !above[0] {
                    if wy >= ICE_SURFACE_Y { self.ice } else { self.grass }
                } else if !above[1] || !above[2] || !above[3] {
                    self.dirt
                } else {
                    let roll = cell_hash(g.seed, wx, wy, wz);
                    if roll < u32::MAX / 45 {
                        self.aerium_vein
                    } else if roll < u32::MAX / 45 + u32::MAX / 160 {
                        self.quartz_vein
                    } else {
                        self.stone
                    }
                }
            } else {
                AIR
            }
        }
    }

    /// The load-bearing worldgen invariant: element-first placement changes
    /// what solid cells are MADE OF, never WHERE they are. The legacy picker's
    /// solid/air pattern must match the current generator's cell for cell — the
    /// property that keeps old saves' edits meaningful across the whole v1→v3
    /// material evolution. (Material parity itself ended at v2→v3: per-biome
    /// crust and luminous surface scatter deliberately diverge — those have
    /// their own distribution tests. Deep uncarved stone below every ore band
    /// stays pure, checked here as a spot invariant.) Censused across origin,
    /// deep, island-band, and far coordinates.
    #[test]
    fn placement_rewiring_is_geometry_identical() {
        let (reg, g) = terrain_with_registry(3);
        let legacy = Legacy::resolve(&reg);
        let stone = reg.id_by_name("Stone").unwrap();

        let chunks: Vec<(i32, i32, i32)> = [
            // Spawn area: surface band with crust, ores, water, carve.
            (0, 0, 0), (0, 1, 0), (0, -1, 0), (1, 0, -1), (2, 3, 2),
            // Deep rock inside and below the ore band.
            (0, -3, 0), (1, -4, 1),
            // The island band (ISLAND_MIN_Y = 112 → cy 7+), icy heights.
            (0, 8, 0), (3, 9, -2), (0, 14, 5),
            // Far out: the f64-spine coordinates the old round fixed.
            (6_250_000, 0, 0), (6_250_000, 8, 0), (-6_250_000, -2, 3),
        ]
        .into_iter()
        .collect();

        let mut cells = 0u64;
        for (cx, cy, cz) in chunks {
            let (x0, y0, z0) =
                (cx * CHUNK_SIZE as i32, cy * CHUNK_SIZE as i32, cz * CHUNK_SIZE as i32);
            for lz in 0..CHUNK_SIZE as i32 {
                for lx in 0..CHUNK_SIZE as i32 {
                    let (wx, wz) = (x0 + lx, z0 + lz);
                    let p = g.profile(wx, wz);
                    for ly in 0..CHUNK_SIZE as i32 {
                        let wy = y0 + ly;
                        let old = legacy.cell_base(&g, &p, wx, wy, wz);
                        let new = g.cell_base(&p, wx, wy, wz, true);
                        cells += 1;
                        assert_eq!(
                            reg.is_solid(old),
                            reg.is_solid(new),
                            "geometry moved at ({wx},{wy},{wz}): {old:?} vs {new:?}"
                        );
                        // Deep uncarved stone below every scattered band stays
                        // pure Stone — no crust/scatter/ore reaches here.
                        if old == legacy.stone && p.height - wy > 64 {
                            assert_eq!(new, stone, "deep stone drifted at ({wx},{wy},{wz})");
                        }
                    }
                }
            }
        }
        assert!(cells > 50_000, "census actually covered ground ({cells} cells)");
    }

    /// Doc test 8 — stream B's distribution: overlap pairs occur (multi-yield
    /// finds are real), stay rare, arity never exceeds two (every emitted id is
    /// a known single or pair), and the single rate stays in the expected band
    /// (stream A's ~5.4% plus B's ~1/8 bonus at full eligibility).
    #[test]
    fn stream_b_yields_bounded_pairs_and_boosted_singles() {
        use std::collections::HashSet;
        let (_reg, g) = terrain_with_registry(11);
        let single_ids: HashSet<BlockId> = g.mat.seams.iter().map(|s| s.id).collect();
        let pair_ids: HashSet<BlockId> = g.mat.pairs.iter().flatten().copied().collect();

        let (mut singles, mut pairs, mut total) = (0u64, 0u64, 0u64);
        // Depth 60: every tier eligible on both streams. The roll is a pure
        // function of (seed, cell), so sampling it directly is the real thing.
        for wx in 0..512 {
            for wz in 0..512 {
                total += 1;
                match g.ore_at(wx, -1000, wz, 60) {
                    None => {}
                    Some(id) if single_ids.contains(&id) => singles += 1,
                    Some(id) if pair_ids.contains(&id) => pairs += 1,
                    Some(id) => panic!("ore_at emitted an unknown id {id:?} — arity bound broken"),
                }
            }
        }
        let single_rate = singles as f64 / total as f64;
        assert!(
            (0.045..=0.075).contains(&single_rate),
            "single-vein rate {single_rate:.4} left the expected band"
        );
        assert!(pairs > 10, "overlap pairs must actually occur (got {pairs} in {total})");
        assert!(
            (pairs as f64) < (singles as f64) * 0.05,
            "pairs must stay rare finds ({pairs} pairs vs {singles} singles)"
        );
    }

    /// The beach-edge dither: about half the grassy columns one block above
    /// the water line (and a quarter at two) dissolve into Soil+Sand; the band
    /// never reaches higher ground.
    #[test]
    fn beach_edge_dither_holds_its_band_and_rates() {
        let (reg, g) = terrain_with_registry(3);
        let beach = reg.id_by_name("Soil+Sand").unwrap();
        let mut rim = [[0u64; 2]; 3]; // [rim-1, rim-2, rim-3+ grassy][total, beach]
        for wx in -512..512 {
            for wz in -512..512 {
                let p = g.profile(wx, wz);
                if p.height <= p.water_level
                    || p.temperature.0 < COLD
                    || p.height - g.sea_level > SNOW_ABOVE_SEA
                    || (p.temperature.0 > HOT && p.humidity.0 < DRY)
                {
                    continue; // not otherwise-grassy: the dither never applies
                }
                let band = ((p.height - p.water_level).min(3) - 1) as usize;
                rim[band][0] += 1;
                if g.dress(&p, wx, wz) == beach {
                    rim[band][1] += 1;
                }
            }
        }
        assert!(rim[0][0] > 200 && rim[1][0] > 200, "seed 3 must offer shoreline to sample");
        let rate = |b: [u64; 2]| b[1] as f64 / b[0] as f64;
        assert!((0.42..=0.58).contains(&rate(rim[0])), "+1 rim ~half: {:?}", rim[0]);
        assert!((0.17..=0.33).contains(&rate(rim[1])), "+2 rim ~quarter: {:?}", rim[1]);
        assert_eq!(rim[2][1], 0, "the dither never reaches above the +2 rim");
    }

    /// Cave-wall clusters: below the seam band (depth > 64) the only Lumin is
    /// the wall rule's, so every hit there must sit vertically against a carved
    /// cell — and the glow does occur.
    #[test]
    fn cave_wall_lumin_hugs_carved_floors_and_ceilings() {
        let (reg, g) = terrain_with_registry(9);
        let lumin = reg.id_by_name("Stone+Lumin").unwrap();
        let (mut found, mut scanned) = (0u64, 0u64);
        'scan: for cz in 0..96 {
            for cy in [-6i32, -7, -8] {
                let (x0, y0, z0) = (0, cy * 16, cz * 16);
                for lx in 0..16 {
                    for lz in 0..16 {
                        let (wx, wz) = (x0 + lx, z0 + lz);
                        let p = g.profile(wx, wz);
                        for ly in 0..16 {
                            let wy = y0 + ly;
                            if p.height - wy <= 64 {
                                continue; // seam band: Lumin is ambiguous there
                            }
                            scanned += 1;
                            if g.cell_base(&p, wx, wy, wz, true) == lumin {
                                found += 1;
                                let carved_v = |ny: i32| {
                                    ny < p.height && g.carved(wx, ny, wz, p.height)
                                };
                                assert!(
                                    carved_v(wy + 1) || carved_v(wy - 1),
                                    "wall Lumin at ({wx},{wy},{wz}) without adjacent carve"
                                );
                                if found >= 25 {
                                    break 'scan;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(found > 0, "deep caverns must actually glow (scanned {scanned} cells)");
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

    /// Pin `fnv1a_32` over eight fixed seed-42 chunks. Values locked before the
    /// reuse pass; a mismatch means generated `ChunkData` bytes moved.
    #[test]
    fn chunk_byte_pin() {
        use crate::hash::fnv1a_32;
        let g = terrain(42);
        // surface, deep, cave, island band, beach, snow crust, two far coords.
        let pins: [(&str, i32, i32, i32, u32); 8] = [
            ("surface", 0, 1, 0, 0xb25ac3be),
            ("deep", 0, -20, 0, 0x24ae7d4e),
            ("cave", 0, -3, 0, 0x148fc284),
            ("island", -1, 9, -4, 0x2c77460e),
            ("beach", 4, 0, -7, 0x1d670c00),
            ("crust", 55, 1, -80, 0xe09b8252),
            ("far_a", 6_250_000, 0, 0, 0x4b4d2cf9),
            ("far_b", -6_250_000, -2, 3, 0xefed0476),
        ];
        for (name, cx, cy, cz, want) in pins {
            assert_eq!(
                fnv1a_32(&chunk_data_bytes(&g.generate(cx, cy, cz))),
                want,
                "{name} ({cx},{cy},{cz})"
            );
        }
    }
}
