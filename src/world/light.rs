//! Cross-chunk lighting (v2.1). A [`LightGrid`] holds skylight and blocklight
//! (each `0..=15`) for every cell of one chunk. It is computed by [`propagate`]
//! as a function of the chunk's own voxels, its six neighbour face light layers
//! ([`FaceShell`]), and the column ceiling ([`CeilingWindow`], the skylight
//! source). Settling is *decoupled* from meshing: a cheap main-thread
//! Gauss-Seidel pass (the `LightLane` worklist lane) relaxes the field,
//! reading the latest neighbour grids directly and enqueuing a neighbour only
//! when their shared border moves ([`border_changed`]) — so convergence costs no
//! GPU work. The mesher later samples the settled [`PaddedLight`] per vertex
//! (interior, border, and face-diagonal alike) for seamless *smooth* light.
//!
//! - **Skylight:** seeded `15` at every cell above its column's terrain surface
//!   (the ceiling), then flooded — losing one level per step *except straight
//!   down at full strength*, so an open column stays lit to the ground. A cave
//!   below the surface receives no seed and only brightens via flood from an
//!   opening, so it is dark and consistent regardless of chunk alignment.
//! - **Blocklight:** a BFS seeded from every emissive cell (and from the
//!   neighbour boundaries), attenuating by 1 per step, stopping at opaque cells.
//!
//! Propagation is a pure function of a [`Chunk`], [`FaceShell`] (neighbour
//! light layers), and [`CeilingWindow`], exactly matching what [`propagate`]
//! reads (interior voxels and face borders only).
use std::cell::RefCell;
use std::collections::VecDeque;

use crate::block::registry::HotTables;
use crate::coord::Face;

use super::chunk::{CHUNK_SIZE, CHUNK_VOLUME, Chunk};
use super::neighborhood::Neighborhood;

/// Maximum light level; the 4-bit domain the packed vertex stores.
pub const MAX_LIGHT: u8 = 15;
/// Cells in one chunk face.
pub const CHUNK_AREA: usize = CHUNK_SIZE * CHUNK_SIZE;
/// Chunk size as a signed coordinate, for the `-1..=16` padded range.
const CS: i32 = CHUNK_SIZE as i32;

/// Light value: 4-bit clamped to 0..=15. Every constructor clamps or is const-checked.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct LightLevel(u8);

impl LightLevel {
    pub const DARK: Self = Self(0);
    pub const FULL: Self = Self(MAX_LIGHT);

    #[inline]
    pub const fn new(v: u8) -> Self {
        Self(if v > MAX_LIGHT { MAX_LIGHT } else { v })
    }
    #[inline]
    pub const fn get(self) -> u8 {
        self.0
    }
    /// One flood step of attenuation; saturates at [`DARK`](Self::DARK).
    #[inline]
    pub const fn attenuated(self) -> Self {
        Self(self.0.saturating_sub(1))
    }
    /// Returns the brighter of the two values.
    #[inline]
    pub fn brighter(self, o: Self) -> Self {
        Self(self.0.max(o.0))
    }
}

/// Skylight and blocklight paired; prevents channel desyncs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Lumel {
    pub sky: LightLevel,
    pub block: LightLevel,
}

impl Lumel {
    pub const DARK: Self = Self { sky: LightLevel::DARK, block: LightLevel::DARK };
    pub const FULL: Self = Self { sky: LightLevel::FULL, block: LightLevel::FULL };
}

/// Per-cell light for one chunk. Compared for equality to detect settlement fixpoint.
#[derive(PartialEq, Eq)]
pub struct LightGrid {
    cells: Box<[Lumel]>,
}

impl LightGrid {
    /// An all-dark grid (also the reusable scratch the settle pass refills).
    pub fn dark() -> Self {
        Self { cells: vec![Lumel::DARK; CHUNK_VOLUME].into() }
    }

    /// An all-full-bright grid, for tests and the neutral mesher path.
    pub fn full() -> Self {
        Self { cells: vec![Lumel::FULL; CHUNK_VOLUME].into() }
    }

    /// Full skylight, no blocklight — the settled light of a chunk fully open to
    /// the sky with no emitters. This is exactly `propagate(uniform_air, dark
    /// shell, open ceiling, …)`'s result, so the analytic light fast path
    /// ([`World::trivial_light`](crate::world::World)) can publish it without a
    /// flood.
    pub fn open_sky() -> Self {
        Self {
            cells: vec![Lumel { sky: LightLevel::FULL, block: LightLevel::DARK }; CHUNK_VOLUME]
                .into(),
        }
    }

    #[inline]
    pub fn at(&self, idx: usize) -> Lumel {
        self.cells[idx]
    }
    #[inline]
    fn set(&mut self, idx: usize, v: Lumel) {
        self.cells[idx] = v;
    }
    /// Copy the 16-cell x-row at `(y, z)` — cells are x-fastest, so this is
    /// one contiguous slice copy (the shell capture's bulk read).
    #[inline]
    pub fn copy_row(&self, y: usize, z: usize, out: &mut [Lumel]) {
        let base = Chunk::index(0, y, z);
        out.copy_from_slice(&self.cells[base..base + CHUNK_SIZE]);
    }
}

/// Light grid plus one-cell shell from 26 neighbours (coords -1..=16). Serves
/// interior, border, and diagonal cells for smooth light across chunk borders.
/// Settling reads only the six face layers; missing neighbours are dark. The
/// light instantiation of [`Neighborhood`]: capture/index/pooling live there,
/// shared with the mesh pass's [`Padded`](super::mesh::Padded).
pub struct PaddedLight {
    inner: Neighborhood<Lumel>,
}

impl PaddedLight {
    /// Light at signed coord (x, y, z) in -1..=16.
    #[cfg(test)]
    #[inline]
    pub(in crate::world) fn at(&self, x: i32, y: i32, z: i32) -> Lumel {
        self.inner.at(x, y, z)
    }

    /// Flat-index read (same [`padded_index`](super::neighborhood::padded_index)
    /// layout as [`Padded`](super::mesh::Padded)) — the sweep's stride walk.
    #[inline]
    pub(in crate::world) fn at_flat(&self, i: usize) -> Lumel {
        self.inner.at_flat(i)
    }

    /// An all-dark shell (no neighbour light anywhere) — the neutral settle path.
    pub fn dark() -> Self {
        Self { inner: Neighborhood::filled(Lumel::DARK) }
    }

    /// An all-full-bright shell — the neutral mesher path (tests).
    pub fn full() -> Self {
        Self { inner: Neighborhood::filled(Lumel::FULL) }
    }

    /// Full skylight, no blocklight — the shell equivalent of
    /// [`LightGrid::open_sky`]. Used to mesh coarse LOD tiles, which are top-down
    /// surface approximations open to the sky with no emitters, so their shading
    /// tracks day/night via skylight instead of clamping to a fake full emitter.
    pub fn open_sky() -> Self {
        Self { inner: Neighborhood::filled(Lumel { sky: LightLevel::FULL, block: LightLevel::DARK }) }
    }

    /// A shell filled from a per-cell closure over signed coords `-1..=16` — for
    /// exercising the mesher's smooth-light sampling with a known field.
    #[cfg(test)]
    pub fn from_fn(f: impl Fn(i32, i32, i32) -> Lumel) -> Self {
        Self { inner: Neighborhood::from_fn(f, Lumel::DARK) }
    }

    /// Copy the chunk and its shell out of the light field. `grid_at(dx, dy, dz)`
    /// yields the [`LightGrid`] at chunk-offset `(dx, dy, dz)` (each `∈ -1..=1`,
    /// `(0,0,0)` is the chunk itself), or `None` (→ dark). Mirrors
    /// [`Padded::capture`](super::mesh::Padded::capture) cell-for-cell; the
    /// bulk fills through [`LightGrid::copy_row`]'s contiguous slice copies.
    pub fn capture<'a>(grid_at: impl Fn(i32, i32, i32) -> Option<&'a LightGrid>) -> Self {
        Self {
            inner: Neighborhood::capture_rows(
                Lumel::DARK,
                grid_at,
                |g: &LightGrid, lx, ly, lz| g.at(Chunk::index(lx, ly, lz)),
                |g: &LightGrid, ly, lz, out| g.copy_row(ly, lz, out),
            ),
        }
    }

}

/// Six neighbour-light face layers (16x16 each) that settle reads. Interior
/// floods locally; borders come from here.
pub struct FaceShell {
    faces: [[Lumel; CHUNK_AREA]; 6], // indexed by Face as usize; near layer of each face neighbour
}

impl FaceShell {
    /// Near border layer of the neighbour across `face` (0 for Pos, 15 for Neg).
    #[inline]
    fn near_layer(face: Face) -> usize {
        match face {
            Face::PosX | Face::PosY | Face::PosZ => 0,
            _ => CHUNK_SIZE - 1,
        }
    }

    /// Capture light grids from neighbours (or None for dark). Reads the near
    /// border layer of each neighbour.
    pub fn capture<'a>(grid_at: impl Fn(Face) -> Option<&'a LightGrid>) -> Self {
        let mut faces = [[Lumel::DARK; CHUNK_AREA]; 6];
        for face in Face::ALL {
            let Some(g) = grid_at(face) else { continue };
            let na = normal_axis(face);
            let (au, av) = plane_axes(face);
            let n = Self::near_layer(face);
            let layer = &mut faces[face as usize];
            for b in 0..CHUNK_SIZE {
                for a in 0..CHUNK_SIZE {
                    let mut lc = [0usize; 3];
                    lc[na] = n;
                    lc[au] = a;
                    lc[av] = b;
                    layer[a + b * CHUNK_SIZE] = g.at(Chunk::index(lc[0], lc[1], lc[2]));
                }
            }
        }
        Self { faces }
    }

    /// Light value from neighbour across `face` at coords `(a, b)`.
    #[inline]
    pub(in crate::world) fn at(&self, face: Face, a: usize, b: usize) -> Lumel {
        self.faces[face as usize][a + b * CHUNK_SIZE]
    }

    /// All-dark shell (no neighbours).
    pub fn dark() -> Self {
        Self { faces: [[Lumel::DARK; CHUNK_AREA]; 6] }
    }
}

/// The skylight ceiling per column: the Y at and above which a column is open
/// sky. Seeded from the generator's ground height (a pure function, so caves
/// stay consistently dark regardless of chunk load order), then RAISED by
/// edited opaque roofs ([`raise`](Self::raise)) so a player-built ceiling
/// shadows every chunk below it instead of leaking full skylight.
#[derive(Clone)]
pub struct CeilingWindow {
    surface: [i32; CHUNK_AREA],
}

impl CeilingWindow {
    /// Compute surface height per column via generator callback.
    pub fn from_heights(mut height: impl FnMut(usize, usize) -> i32) -> Self {
        let mut surface = [0i32; CHUNK_AREA];
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                surface[lx + lz * CHUNK_SIZE] = height(lx, lz);
            }
        }
        Self { surface }
    }

    /// Everything open to the sky — for tests and the neutral path.
    pub fn open() -> Self {
        Self { surface: [i32::MIN; CHUNK_AREA] }
    }

    #[inline]
    pub(in crate::world) fn open_above(&self, lx: usize, lz: usize, world_y: i32) -> bool {
        world_y >= self.surface[lx + lz * CHUNK_SIZE]
    }

    /// Raise one column's ceiling to at least `surface` (a constructed opaque
    /// roof: open sky begins at the cell ABOVE it). Never lowers — the
    /// generator ground below stays the floor of the value.
    pub(in crate::world) fn raise(&mut self, lx: usize, lz: usize, surface: i32) {
        let cell = &mut self.surface[lx + lz * CHUNK_SIZE];
        *cell = (*cell).max(surface);
    }

    /// The Y at which this column becomes open sky (see [`open_above`](Self::open_above)).
    pub(in crate::world) fn surface_at(&self, lx: usize, lz: usize) -> i32 {
        self.surface[lx + lz * CHUNK_SIZE]
    }
}

/// Reusable flood scratch for [`propagate`]: the two per-channel level grids and
/// the BFS frontier. Held thread-local so a worker's repeated `propagate` calls
/// reuse one allocation each instead of allocating two `CHUNK_VOLUME` boxes and a
/// growing `VecDeque` per job (the light-settle churn). Never shared or sent
/// across threads (borrowed only for the duration of one `propagate` call), so it
/// is sound to key on the calling worker.
struct FloodScratch {
    sky: Vec<LightLevel>,
    block: Vec<LightLevel>,
    queue: VecDeque<usize>,
}

thread_local! {
    static FLOOD: RefCell<FloodScratch> = const {
        RefCell::new(FloodScratch { sky: Vec::new(), block: Vec::new(), queue: VecDeque::new() })
    };
}

/// Reset a channel grid to the all-dark initial state (every unseeded cell must
/// read `DARK`), sizing it on first use. `fill` reuses the existing allocation
/// when the length already matches — the common (post-warmup) case.
fn reset_dark(v: &mut Vec<LightLevel>) {
    if v.len() != CHUNK_VOLUME {
        v.clear();
        v.resize(CHUNK_VOLUME, LightLevel::DARK);
    } else {
        v.fill(LightLevel::DARK);
    }
}

/// Recompute chunk light from scratch. Light removal needs no second pass:
/// breaking emitters or placing blocks just lowers the grid. `world_y0` is chunk's Y origin.
#[allow(clippy::needless_range_loop)] // `i` is the flood-queue key and the `block[]` slot
pub fn propagate(
    chunk: &Chunk,
    shell: &FaceShell,
    ceiling: &CeilingWindow,
    world_y0: i32,
    tables: &HotTables,
    out: &mut LightGrid,
) {
    out.cells.fill(Lumel::DARK);
    let cs = CHUNK_SIZE as i32;
    // Decode the opacity field ONCE (payload-specialized, ~a palette pass)
    // into an L1-resident bitset: the flood probes it ~6 times per relaxed
    // cell, and each probe used to be a payload dispatch + palette load.
    let mut opaque_bits = [0u64; CHUNK_VOLUME / 64];
    chunk.fill_opacity(|id| tables.opaque(id), &mut opaque_bits);
    let opaque_at = |x: i32, y: i32, z: i32| {
        let i = Chunk::index(x as usize, y as usize, z as usize);
        (opaque_bits[i >> 6] >> (i & 63)) & 1 != 0
    };

    // Skylight: borrow thread-local scratch, reset dark, seed and flood.
    // No stale flood state from a prior job survives.
    let (mut sky, mut block, mut queue) = FLOOD.with_borrow_mut(|s| {
        (std::mem::take(&mut s.sky), std::mem::take(&mut s.block), std::mem::take(&mut s.queue))
    });
    reset_dark(&mut sky);
    reset_dark(&mut block);
    queue.clear();
    // Seed 1: open sky floods down each column until the first opaque voxel
    // (classic heightmap seed, gated by ceiling). Deep chunks seed nothing here;
    // their light arrives from the +Y halo.
    let top_y = world_y0 + CHUNK_SIZE as i32;
    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            if !ceiling.open_above(x, z, top_y) {
                continue;
            }
            for y in (0..CHUNK_SIZE).rev() {
                if opaque_at(x as i32, y as i32, z as i32) {
                    break; // shadowed below the first opaque cell
                }
                let i = Chunk::index(x, y, z);
                sky[i] = LightLevel::FULL;
                queue.push_back(i);
            }
        }
    }
    // Seed 2: the six neighbour boundaries (light crossing in loses one step).
    seed_from_shell(shell, |i, lum| {
        if lum.sky > sky[i] {
            sky[i] = lum.sky;
            queue.push_back(i);
        }
    });
    // Flood: -1 per step, except full skylight passes straight down (open columns stay lit).
    while let Some(i) = queue.pop_front() {
        let level = sky[i];
        let (x, y, z) = Chunk::local_of(i);
        let (x, y, z) = (x as i32, y as i32, z as i32);
        let mut relax = |nx: i32, ny: i32, nz: i32, down: bool| {
            if opaque_at(nx, ny, nz) {
                return;
            }
            let cand = if down && level == LightLevel::FULL { LightLevel::FULL } else { level.attenuated() };
            let ni = Chunk::index(nx as usize, ny as usize, nz as usize);
            if cand > sky[ni] {
                sky[ni] = cand;
                queue.push_back(ni);
            }
        };
        if x > 0 { relax(x - 1, y, z, false); }
        if x + 1 < cs { relax(x + 1, y, z, false); }
        if y > 0 { relax(x, y - 1, z, true); }
        if y + 1 < cs { relax(x, y + 1, z, false); }
        if z > 0 { relax(x, y, z - 1, false); }
        if z + 1 < cs { relax(x, y, z + 1, false); }
    }

    // Blocklight: block was reset to all-dark; clear queue defensively, seed emitters, flood.
    queue.clear();
    for i in 0..CHUNK_VOLUME {
        let (x, y, z) = Chunk::local_of(i);
        let em = tables.emission[chunk.get_local(x, y, z).0 as usize];
        if em > 0 {
            block[i] = LightLevel::new(em);
            queue.push_back(i);
        }
    }
    seed_from_shell(shell, |i, lum| {
        if lum.block > block[i] {
            block[i] = lum.block;
            queue.push_back(i);
        }
    });
    while let Some(i) = queue.pop_front() {
        let level = block[i];
        if level <= LightLevel::new(1) {
            continue;
        }
        let cand = level.attenuated();
        let (x, y, z) = Chunk::local_of(i);
        let (x, y, z) = (x as i32, y as i32, z as i32);
        let mut relax = |nx: i32, ny: i32, nz: i32| {
            if opaque_at(nx, ny, nz) {
                return;
            }
            let ni = Chunk::index(nx as usize, ny as usize, nz as usize);
            if cand > block[ni] {
                block[ni] = cand;
                queue.push_back(ni);
            }
        };
        if x > 0 { relax(x - 1, y, z); }
        if x + 1 < cs { relax(x + 1, y, z); }
        if y > 0 { relax(x, y - 1, z); }
        if y + 1 < cs { relax(x, y + 1, z); }
        if z > 0 { relax(x, y, z - 1); }
        if z + 1 < cs { relax(x, y, z + 1); }
    }

    for i in 0..CHUNK_VOLUME {
        out.set(i, Lumel { sky: sky[i], block: block[i] });
    }

    // Return the scratch buffers (with their capacity) for the next call.
    FLOOD.with_borrow_mut(|s| {
        s.sky = sky;
        s.block = block;
        s.queue = queue;
    });
}

/// Seed border cells from neighbour shell faces (skylight full-strength from +Y).
/// Dark shell cells (missing neighbours) don't seed.
fn seed_from_shell(shell: &FaceShell, mut seed: impl FnMut(usize, Lumel)) {
    for face in Face::ALL {
        let na = normal_axis(face);
        let (au, av) = plane_axes(face);
        let inner = match face {
            Face::PosX | Face::PosY | Face::PosZ => CS - 1,
            _ => 0,
        };
        for b in 0..CHUNK_SIZE {
            for a in 0..CHUNK_SIZE {
                let src = shell.at(face, a, b);
                let sky = if face == Face::PosY && src.sky == LightLevel::FULL {
                    LightLevel::FULL
                } else {
                    src.sky.attenuated()
                };
                let seeded = Lumel { sky, block: src.block.attenuated() };
                if seeded == Lumel::DARK {
                    continue;
                }
                let mut ci = [0i32; 3];
                ci[na] = inner;
                ci[au] = a as i32;
                ci[av] = b as i32;
                seed(Chunk::index(ci[0] as usize, ci[1] as usize, ci[2] as usize), seeded);
            }
        }
    }
}

/// Whether border changed on a face. Used to enqueue neighbours only when their
/// shared boundary moves (settling convergence detection).
pub(in crate::world) fn border_changed(a: &LightGrid, b: &LightGrid, face: Face) -> bool {
    face_cells(face).any(|(ci, _)| {
        let i = Chunk::index(ci[0] as usize, ci[1] as usize, ci[2] as usize);
        a.at(i) != b.at(i)
    })
}

#[inline]
fn normal_axis(face: Face) -> usize {
    match face {
        Face::NegX | Face::PosX => 0,
        Face::NegY | Face::PosY => 1,
        Face::NegZ | Face::PosZ => 2,
    }
}

/// The two in-face axes of `face`, ascending.
#[inline]
fn plane_axes(face: Face) -> (usize, usize) {
    match normal_axis(face) {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    }
}

/// Map face's border cells to (interior, exterior) coords. Used by both seeding and diffing.
fn face_cells(face: Face) -> impl Iterator<Item = ([i32; 3], [i32; 3])> {
    let na = normal_axis(face);
    let (au, av) = plane_axes(face);
    let (inner, outer) = match face {
        Face::PosX | Face::PosY | Face::PosZ => (CS - 1, CS),
        _ => (0, -1),
    };
    (0..CHUNK_SIZE).flat_map(move |a| {
        (0..CHUNK_SIZE).map(move |b| {
            let mut ci = [0i32; 3];
            ci[na] = inner;
            ci[au] = a as i32;
            ci[av] = b as i32;
            let mut co = ci;
            co[na] = outer;
            (ci, co)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::BlockId;
    use voxel_engine::Pass;

    fn tables() -> HotTables {
        HotTables::from_parts(
            &[false, true, true, true],
            &[false, true, false, true], // id 1 stone, id 3 opaque emitter
            &[false, false, false, false],
            vec![Pass::Opaque, Pass::Opaque, Pass::Blend, Pass::Opaque].into(),
            vec![0, 0, 15, 15].into(), // ids 2 and 3 emit 15
            vec![0, 0, 0, 0].into(),
        )
    }

    fn lit(chunk: &Chunk) -> LightGrid {
        let mut grid = LightGrid::dark();
        propagate(chunk, &FaceShell::dark(), &CeilingWindow::open(), 0, &tables(), &mut grid);
        grid
    }

    /// Full settle-flood cost for a surface-band chunk — the gauge for the
    /// propagate opacity-bitset redesign. Ignored: a timing benchmark, not a
    /// correctness gate. Run with
    /// `cargo test --release light_propagate_throughput -- --ignored --nocapture`.
    /// 2026-07-19 (12-core box), per-probe `get_local`: ~18.1k settles/s;
    /// decoded opacity bitset: ~32.6k settles/s (1.8×).
    #[test]
    #[ignore]
    fn light_propagate_throughput() {
        use crate::block::registry::BlockRegistry;
        use crate::world::generation::{SineHills, TerrainGenerator};

        let mut registry = BlockRegistry::with_builtins();
        let generator = SineHills::new(&mut registry, 20.0, 5);
        // The surface chunk at the origin: the Dense band every load floods
        // (deep/sky chunks take the analytic fast paths and never get here).
        let cy = generator.height(0, 0).div_euclid(CHUNK_SIZE as i32);
        let chunk = Chunk::new(0, cy, 0, &generator);
        let tables = registry.hot_tables();
        let shell = FaceShell::dark();
        let ceiling = CeilingWindow::from_heights(|lx, lz| generator.height(lx as i32, lz as i32));
        let mut out = LightGrid::dark();

        const N: usize = 4000;
        let start = std::time::Instant::now();
        for _ in 0..N {
            propagate(&chunk, &shell, &ceiling, cy * CHUNK_SIZE as i32, &tables, &mut out);
            std::hint::black_box(&out);
        }
        let dt = start.elapsed();
        println!(
            "{N} propagates in {:.3}s = {:.0} settles/s",
            dt.as_secs_f64(),
            N as f64 / dt.as_secs_f64()
        );
    }

    #[test]
    fn analytic_grids_match_propagate() {
        // The anchor for `World::trivial_light`: the analytic grids it publishes
        // without a flood must equal what `propagate` computes with a dark shell.
        let tables = tables();
        // Uniform opaque (id 1) → all dark, regardless of ceiling.
        let opaque = Chunk::from_uniform(0, -10, 0, BlockId(1));
        let mut got = LightGrid::dark();
        propagate(&opaque, &FaceShell::dark(), &CeilingWindow::from_heights(|_, _| 100), -160, &tables, &mut got);
        assert!(got == LightGrid::dark(), "uniform opaque == dark()");
        // Opaque does not imply dark: an opaque emitter must bypass the analytic
        // shortcut and seed blocklight in the regular propagation path.
        let emissive = Chunk::from_uniform(0, -10, 0, BlockId(3));
        assert!(!emissive.is_uniform_opaque(&tables));
        let mut got = LightGrid::dark();
        propagate(
            &emissive,
            &FaceShell::dark(),
            &CeilingWindow::from_heights(|_, _| 100),
            -160,
            &tables,
            &mut got,
        );
        assert_eq!(got.at(Chunk::index(8, 8, 8)).block, LightLevel::FULL);
        // Uniform air fully open to the sky → full sky, no blocklight.
        let air = Chunk::from_uniform(0, 10, 0, BlockId(0));
        let mut got = LightGrid::dark();
        propagate(&air, &FaceShell::dark(), &CeilingWindow::open(), 160, &tables, &mut got);
        assert!(got == LightGrid::open_sky(), "open-sky air == open_sky()");
    }

    #[test]
    fn open_column_lit_to_floor_and_sealed_layer_shadows_below() {
        // A full opaque layer at y=5 seals the lower half: with no gap for the
        // horizontal skylight flood to leak through, everything below is dark,
        // while the open cells above are lit to the layer.
        let mut cells = [BlockId(0); CHUNK_VOLUME];
        for z in 0..16 {
            for x in 0..16 {
                cells[Chunk::index(x, 5, z)] = BlockId(1); // opaque floor across the chunk
            }
        }
        let chunk = Chunk::from_cells(0, 0, 0, Box::new(cells));
        let grid = lit(&chunk);

        assert_eq!(grid.at(Chunk::index(4, 15, 4)).sky, LightLevel::FULL, "top lit");
        assert_eq!(grid.at(Chunk::index(4, 6, 4)).sky, LightLevel::FULL, "just above the layer lit");
        assert_eq!(grid.at(Chunk::index(4, 4, 4)).sky, LightLevel::DARK, "sealed below the layer");
        assert_eq!(grid.at(Chunk::index(0, 0, 0)).sky, LightLevel::DARK, "floor sealed dark");
    }

    /// A player-built roof in the chunk above must stop the analytic per-column
    /// skylight seed in the chunk below: the ceiling window is raised by edited
    /// opaque cells and the edit invalidates the cached column.
    #[test]
    fn constructed_roof_in_upper_chunk_shadows_lower_chunk() {
        use crate::coord::ChunkCoord;
        use crate::world::World;

        // Safely above terrain and the flying-island band, exactly on a chunk
        // boundary so the roof occupies local y=0 of the upper chunk.
        const ROOF_Y: i32 = 400;
        let lower_cy = ROOF_Y.div_euclid(CS) - 1;
        let lower_y0 = lower_cy * CS;
        let lower_coord = ChunkCoord::new(0, lower_cy, 0);

        let mut world = World::new(0x5EED);
        let before = world.capture_ceiling(lower_coord);
        assert!(before.open_above(8, 8, ROOF_Y), "fixture starts open to sky");

        let stone = world.registry().id_by_name("Stone").expect("builtin Stone");
        for z in 0..CS {
            for x in 0..CS {
                world.set_block(x, ROOF_Y, z, stone);
            }
        }

        // The ceiling moved, so every LOADED chunk below the roof in this
        // column is owed a re-settle (the roof chunk itself is unloaded here,
        // so any worklist entry in the column proves the cascade fired).
        assert!(
            world.light_worklist.iter().any(|c| c.x == 0 && c.z == 0),
            "raising a column's ceiling must re-seed the loaded chunks below it"
        );

        let ceiling = world.capture_ceiling(lower_coord);

        // Settle the real upper neighbour containing the opaque roof.
        let mut roof_cells = [BlockId(0); CHUNK_VOLUME];
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                roof_cells[Chunk::index(x, 0, z)] = BlockId(1);
            }
        }
        let roof = Chunk::from_cells(0, lower_cy + 1, 0, Box::new(roof_cells));
        let mut roof_light = LightGrid::dark();
        propagate(
            &roof,
            &FaceShell::dark(),
            &ceiling,
            ROOF_Y,
            &tables(),
            &mut roof_light,
        );
        assert_eq!(
            roof_light.at(Chunk::index(8, 0, 8)).sky,
            LightLevel::DARK,
            "the roof's lower face is dark",
        );

        let upper_shell =
            FaceShell::capture(|face| (face == Face::PosY).then_some(&roof_light));
        let lower = Chunk::from_uniform(0, lower_cy, 0, BlockId(0));
        let mut lower_light = LightGrid::dark();
        propagate(
            &lower,
            &upper_shell,
            &ceiling,
            lower_y0,
            &tables(),
            &mut lower_light,
        );

        assert_eq!(
            lower_light.at(Chunk::index(8, CHUNK_SIZE - 1, 8)).sky,
            LightLevel::DARK,
            "the constructed roof must shadow the chunk directly below it",
        );
    }

    #[test]
    fn cave_in_a_deep_chunk_is_dark() {
        // A hollow chunk whose top is far below the terrain surface: the ceiling
        // reports the top as closed, so no skylight is seeded and the cavern is
        // dark — consistently, regardless of the 16-cell chunk alignment.
        let chunk = Chunk::from_cells(0, -8, 0, Box::new([BlockId(0); CHUNK_VOLUME]));
        let ceiling = CeilingWindow::from_heights(|_, _| 40); // surface well above this chunk
        let mut grid = LightGrid::dark();
        propagate(&chunk, &FaceShell::dark(), &ceiling, -128, &tables(), &mut grid);

        assert_eq!(grid.at(Chunk::index(4, 8, 4)).sky, LightLevel::DARK, "cavern dark");
        assert_eq!(grid.at(Chunk::index(0, 0, 0)).sky, LightLevel::DARK, "cavern floor dark");
    }

    #[test]
    fn blocklight_falls_off_by_one_per_step() {
        let mut chunk = Chunk::from_cells(0, 0, 0, Box::new([BlockId(0); CHUNK_VOLUME]));
        chunk.set_local(8, 8, 8, BlockId(2)); // emitter, level 15
        let ceiling = CeilingWindow::from_heights(|_, _| 100); // fully underground: isolate blocklight
        let mut grid = LightGrid::dark();
        propagate(&chunk, &FaceShell::dark(), &ceiling, 0, &tables(), &mut grid);

        assert_eq!(grid.at(Chunk::index(8, 8, 8)).block.get(), 15, "the emitter");
        assert_eq!(grid.at(Chunk::index(9, 8, 8)).block.get(), 14, "one step");
        assert_eq!(grid.at(Chunk::index(11, 8, 8)).block.get(), 12, "three steps");
    }

    /// FaceShell reads neighbour near-layer correctly for all faces.
    #[test]
    fn face_shell_captures_the_neighbour_near_layer() {
        let mut grid = LightGrid::dark();
        for i in 0..CHUNK_VOLUME {
            let (x, y, z) = Chunk::local_of(i);
            grid.set(i, Lumel {
                sky: LightLevel::new((x + y) as u8 % 16),
                block: LightLevel::new((z + y) as u8 % 16),
            });
        }
        for face in Face::ALL {
            let (dx, dy, dz) = face.delta();
            let padded = PaddedLight::capture(|nx, ny, nz| {
                (nx == dx && ny == dy && nz == dz).then_some(&grid)
            });
            let shell = FaceShell::capture(|f| (f == face).then_some(&grid));
            let na = normal_axis(face);
            let (au, av) = plane_axes(face);
            let outer = match face {
                Face::PosX | Face::PosY | Face::PosZ => CS,
                _ => -1,
            };
            for b in 0..CHUNK_SIZE {
                for a in 0..CHUNK_SIZE {
                    let mut co = [0i32; 3];
                    co[na] = outer;
                    co[au] = a as i32;
                    co[av] = b as i32;
                    assert_eq!(
                        shell.at(face, a, b),
                        padded.at(co[0], co[1], co[2]),
                        "face {face:?} at ({a}, {b})"
                    );
                }
            }
        }
    }

    /// Two adjacent chunks relaxed by hand — a torch in the left chunk floods
    /// across the shared border into the right. Iterating "capture the shell,
    /// propagate" reaches a fixpoint (no border moves) within a bounded number of
    /// passes: the convergence the decoupled settle pass relies on.
    #[test]
    fn settle_reaches_a_fixpoint_across_a_border() {
        let tables = tables();
        let mut left_c = Chunk::from_cells(0, 0, 0, Box::new([BlockId(0); CHUNK_VOLUME]));
        left_c.set_local(14, 8, 8, BlockId(2)); // emitter near the +X border
        let right_c = Chunk::from_cells(1, 0, 0, Box::new([BlockId(0); CHUNK_VOLUME]));
        let ceiling = CeilingWindow::from_heights(|_, _| 100); // underground: isolate blocklight

        // Shell with one neighbour across face (dark elsewhere).
        fn shell(face: Face, nbr: &LightGrid) -> FaceShell {
            FaceShell::capture(|f| (f == face).then_some(nbr))
        }

        let mut left = LightGrid::dark();
        let mut right = LightGrid::dark();
        let mut passes = 0;
        loop {
            passes += 1;
            let mut new_left = LightGrid::dark();
            propagate(&left_c, &shell(Face::PosX, &right), &ceiling, 0, &tables, &mut new_left);
            let mut new_right = LightGrid::dark();
            propagate(&right_c, &shell(Face::NegX, &new_left), &ceiling, 0, &tables, &mut new_right);
            let stable = !border_changed(&left, &new_left, Face::PosX)
                && !border_changed(&right, &new_right, Face::NegX);
            left = new_left;
            right = new_right;
            if stable {
                break;
            }
            assert!(passes < 20, "settle did not converge");
        }
        // The torch light actually crossed the border (right chunk's near cell lit).
        assert!(right.at(Chunk::index(0, 8, 8)).block.get() > 0, "light crossed the seam");
    }

}
