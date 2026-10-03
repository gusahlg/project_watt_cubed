//! Cross-chunk lighting (v2.1). A [`LightGrid`] holds skylight and blocklight
//! (each `0..=15`) for every cell of one chunk: `Uniform` when every cell
//! agrees (open sky, solid rock), else a dense 16³ box of packed bytes
//! (`sky << 4 | block`, 4 KiB). It is computed by [`propagate`]
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
use std::sync::{Arc, OnceLock};

use crate::block::registry::HotTables;
use crate::coord::Face;
use crate::space::FaceFrame;

use super::chunk::{CHUNK_SIZE, CHUNK_VOLUME, Chunk};
use super::neighborhood::Neighborhood;
use super::Sky;

/// Maximum light level; the 4-bit domain the packed vertex stores.
pub const MAX_LIGHT: u8 = 15;
/// Cells in one chunk face.
pub const CHUNK_AREA: usize = CHUNK_SIZE * CHUNK_SIZE;
/// Chunk size as a signed coordinate, for the `-1..=16` padded range.
#[cfg(test)]
const CS: i32 = CHUNK_SIZE as i32;
/// Flat-index strides matching [`Chunk::index`]: x fastest, then z, then y.
const STRIDE_Z: usize = CHUNK_SIZE;
const STRIDE_Y: usize = CHUNK_SIZE * CHUNK_SIZE;

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
}

/// Skylight and blocklight paired; prevents channel desyncs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Lumel {
    pub sky: LightLevel,
    pub block: LightLevel,
}

impl Lumel {
    pub const DARK: Self = Self { sky: LightLevel::DARK, block: LightLevel::DARK };
    #[cfg(test)]
    pub const FULL: Self = Self { sky: LightLevel::FULL, block: LightLevel::FULL };

    /// Dense-grid storage byte: `sky << 4 | block`.
    #[inline]
    pub const fn pack(self) -> u8 {
        (self.sky.get() << 4) | self.block.get()
    }

    /// Inverse of [`pack`](Self::pack).
    #[inline]
    pub const fn unpack(b: u8) -> Self {
        Self { sky: LightLevel::new(b >> 4), block: LightLevel::new(b & 0x0f) }
    }
}

/// Packed lumel byte (`sky << 4 | block`). Dense [`LightGrid`], [`FaceShell`],
/// and [`PaddedLight`] storage; [`Lumel`] is decoded at the public read boundary.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::world) struct PackedLumel(u8);

impl PackedLumel {
    pub(in crate::world) const DARK: Self = Self(0);
    #[cfg(test)]
    const FULL: Self = Self((MAX_LIGHT << 4) | MAX_LIGHT);
    const OPEN_SKY: Self = Self(MAX_LIGHT << 4);

    #[inline]
    const fn pack(l: Lumel) -> Self {
        Self(l.pack())
    }

    #[inline]
    const fn unpack(self) -> Lumel {
        Lumel::unpack(self.0)
    }

    #[inline]
    const fn sky(self) -> u8 {
        self.0 >> 4
    }

    #[inline]
    const fn block(self) -> u8 {
        self.0 & 0x0f
    }

    #[inline]
    const fn with_sky(self, sky: u8) -> Self {
        Self((sky << 4) | (self.0 & 0x0f))
    }

    #[inline]
    const fn with_block(self, block: u8) -> Self {
        Self((self.0 & 0xf0) | block)
    }
}

/// Per-cell light for one chunk. Compared for equality to detect settlement fixpoint.
/// Two representations: one lumel when every cell agrees, else a dense packed 16³ box.
pub struct LightGrid(Repr);

enum Repr {
    Uniform(Lumel),
    Cells(Box<[PackedLumel; CHUNK_VOLUME]>),
}

const _: () = assert!(std::mem::size_of::<LightGrid>() <= 16);

/// Heap-allocate a filled cell box without staging the 4 KiB array on the stack.
fn alloc_cells(fill: PackedLumel) -> Box<[PackedLumel; CHUNK_VOLUME]> {
    vec![fill; CHUNK_VOLUME].into_boxed_slice().try_into().unwrap_or_else(|_| unreachable!())
}

impl LightGrid {
    /// An all-dark grid (also the reusable scratch the settle pass refills).
    pub const fn dark() -> Self {
        Self(Repr::Uniform(Lumel::DARK))
    }

    /// An all-full-bright grid, for tests and the neutral mesher path.
    #[cfg(test)]
    pub const fn full() -> Self {
        Self(Repr::Uniform(Lumel::FULL))
    }

    /// Full skylight, no blocklight — the settled light of a chunk fully open to
    /// the sky with no emitters. This is exactly `propagate(uniform_air, dark
    /// shell, open ceiling, …)`'s result, so the analytic light fast path
    /// ([`World::trivial_light`](crate::world::World)) can publish it without a
    /// flood.
    pub const fn open_sky() -> Self {
        Self(Repr::Uniform(Lumel { sky: LightLevel::FULL, block: LightLevel::DARK }))
    }

    /// Any face-border lumel has blocklight `> 1` (level 1 attenuates to 0
    /// crossing in, so it cannot seed a neighbour). Uniform is one compare;
    /// dense scans the six faces.
    pub(in crate::world) fn has_border_blocklight(&self) -> bool {
        match &self.0 {
            Repr::Uniform(v) => v.block.get() > 1,
            Repr::Cells(cells) => Face::ALL.iter().any(|&face| {
                FACE_INDEX[face as usize].iter().any(|&i| cells[i].block() > 1)
            }),
        }
    }

    #[inline]
    pub(in crate::world) fn packed_at(&self, idx: usize) -> PackedLumel {
        match &self.0 {
            Repr::Uniform(v) => PackedLumel::pack(*v),
            Repr::Cells(c) => c[idx],
        }
    }

    #[cfg(test)]
    #[inline]
    pub fn at(&self, idx: usize) -> Lumel {
        self.packed_at(idx).unpack()
    }
    #[cfg(test)]
    #[inline]
    pub(in crate::world) fn set(&mut self, idx: usize, v: Lumel) {
        match &self.0 {
            Repr::Cells(_) => {}
            Repr::Uniform(u) if *u == v => return,
            Repr::Uniform(_) => {
                self.make_dense();
            }
        }
        match &mut self.0 {
            Repr::Cells(c) => c[idx] = PackedLumel::pack(v),
            Repr::Uniform(_) => unreachable!(),
        }
    }
    /// Copy the 16-cell x-row at `(y, z)` — cells are x-fastest, so this is
    /// one contiguous slice copy (the shell capture's bulk read). Uniform
    /// grids fill the row without a cell loop.
    #[inline]
    pub(in crate::world) fn copy_row(&self, y: usize, z: usize, out: &mut [PackedLumel]) {
        debug_assert_eq!(out.len(), CHUNK_SIZE);
        match &self.0 {
            Repr::Uniform(v) => out.fill(PackedLumel::pack(*v)),
            Repr::Cells(cells) => {
                let base = Chunk::index(0, y, z);
                out.copy_from_slice(&cells[base..base + CHUNK_SIZE]);
            }
        }
    }

    /// Near-border layer of `face` (0 for Pos, 15 for Neg) into a 16×16 face buffer.
    /// Uniform fills once; Y/Z faces copy 16 x-rows; X is strided.
    fn copy_face(&self, face: Face, out: &mut [PackedLumel]) {
        debug_assert_eq!(out.len(), CHUNK_AREA);
        match &self.0 {
            Repr::Uniform(v) => out.fill(PackedLumel::pack(*v)),
            Repr::Cells(cells) => {
                let n = FaceShell::near_layer(face);
                match face {
                    Face::PosY | Face::NegY => {
                        for z in 0..CHUNK_SIZE {
                            let base = Chunk::index(0, n, z);
                            out[z * CHUNK_SIZE..z * CHUNK_SIZE + CHUNK_SIZE]
                                .copy_from_slice(&cells[base..base + CHUNK_SIZE]);
                        }
                    }
                    Face::PosZ | Face::NegZ => {
                        for y in 0..CHUNK_SIZE {
                            let base = Chunk::index(0, y, n);
                            out[y * CHUNK_SIZE..y * CHUNK_SIZE + CHUNK_SIZE]
                                .copy_from_slice(&cells[base..base + CHUNK_SIZE]);
                        }
                    }
                    Face::PosX | Face::NegX => {
                        for z in 0..CHUNK_SIZE {
                            for y in 0..CHUNK_SIZE {
                                out[y + z * CHUNK_SIZE] = cells[Chunk::index(n, y, z)];
                            }
                        }
                    }
                }
            }
        }
    }

    /// Densify in place and return the cell slice. Uniform expands to a filled box.
    #[cfg(test)]
    fn make_dense(&mut self) -> &mut [PackedLumel] {
        if let Repr::Uniform(v) = &self.0 {
            let p = PackedLumel::pack(*v);
            self.0 = Repr::Cells(alloc_cells(p));
        }
        match &mut self.0 {
            Repr::Cells(c) => &mut c[..],
            Repr::Uniform(_) => unreachable!(),
        }
    }

    /// Dense expansion of this grid — test helper so a Uniform grid and its
    /// cell-wise equivalent can be meshed side by side.
    #[cfg(test)]
    pub fn to_dense(&self) -> Self {
        match &self.0 {
            Repr::Cells(c) => Self(Repr::Cells(c.clone())),
            Repr::Uniform(v) => Self(Repr::Cells(alloc_cells(PackedLumel::pack(*v)))),
        }
    }

    /// Heap bytes of the dense cell box; a Uniform grid holds no heap array.
    pub fn allocated_bytes(&self) -> usize {
        match &self.0 {
            Repr::Uniform(_) => 0,
            Repr::Cells(c) => std::mem::size_of_val(c.as_ref()),
        }
    }

    /// `true` when every lumel is stored as one value (no 16³ box).
    pub(in crate::world) fn is_uniform(&self) -> bool {
        matches!(self.0, Repr::Uniform(_))
    }
}

/// Value equality: Uniform and dense-all-equal grids holding the same lumel compare equal.
impl PartialEq for LightGrid {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Repr::Uniform(a), Repr::Uniform(b)) => a == b,
            (Repr::Cells(a), Repr::Cells(b)) => **a == **b,
            (Repr::Uniform(v), Repr::Cells(c)) | (Repr::Cells(c), Repr::Uniform(v)) => {
                let p = PackedLumel::pack(*v);
                c.iter().all(|&cell| cell == p)
            }
        }
    }
}
impl Eq for LightGrid {}

/// Light grid plus one-cell shell from 26 neighbours (coords -1..=16). Serves
/// interior, border, and diagonal cells for smooth light across chunk borders.
/// Settling reads only the six face layers; missing neighbours are dark. The
/// light instantiation of [`Neighborhood`]: capture/index/pooling live there,
/// shared with the mesh pass's [`Padded`](super::mesh::Padded).
pub struct PaddedLight {
    inner: Neighborhood<PackedLumel>,
}

impl PaddedLight {
    /// Light at signed coord (x, y, z) in -1..=16.
    #[cfg(test)]
    #[inline]
    pub(in crate::world) fn at(&self, x: i32, y: i32, z: i32) -> Lumel {
        self.inner.at(x, y, z).unpack()
    }

    /// Flat-index read (same [`padded_index`](super::neighborhood::padded_index)
    /// layout as [`Padded`](super::mesh::Padded)) — the sweep's stride walk.
    #[inline]
    pub(in crate::world) fn at_flat(&self, i: usize) -> Lumel {
        self.inner.at_flat(i).unpack()
    }

    /// An all-full-bright shell — the neutral mesher path (tests).
    #[cfg(test)]
    pub fn full() -> Self {
        Self { inner: Neighborhood::filled(PackedLumel::FULL) }
    }

    /// A shell filled from a per-cell closure over signed coords `-1..=16` — for
    /// exercising the mesher's smooth-light sampling with a known field.
    #[cfg(test)]
    pub fn from_fn(f: impl Fn(i32, i32, i32) -> Lumel) -> Self {
        Self {
            inner: Neighborhood::from_fn(|x, y, z| PackedLumel::pack(f(x, y, z)), PackedLumel::DARK),
        }
    }

    /// Overwrite one shell cell (signed coords): a seam patch after the capture.
    #[inline]
    pub(in crate::world) fn set(&mut self, p: [i32; 3], v: PackedLumel) {
        self.inner.set(p, v);
    }

    /// Copy the chunk and its shell out of the light field. `grid_at(dx, dy, dz)`
    /// yields the [`LightGrid`] at chunk-offset `(dx, dy, dz)` (each `∈ -1..=1`,
    /// `(0,0,0)` is the chunk itself), or `None` (→ dark). Mirrors
    /// [`Padded::capture`](super::mesh::Padded::capture) cell-for-cell; the
    /// bulk fills through [`LightGrid::copy_row`]'s contiguous packed-byte copies.
    pub fn capture<'a>(grid_at: impl Fn(i32, i32, i32) -> Option<&'a LightGrid>) -> Self {
        Self {
            inner: Neighborhood::capture_rows(
                PackedLumel::DARK,
                grid_at,
                |g: &LightGrid, lx, ly, lz| g.packed_at(Chunk::index(lx, ly, lz)),
                |g: &LightGrid, ly, lz, out| g.copy_row(ly, lz, out),
            ),
        }
    }

}

/// Six neighbour-light face layers (16x16 each) that settle reads. Interior
/// floods locally; borders come from here.
pub struct FaceShell {
    faces: [[PackedLumel; CHUNK_AREA]; 6], // indexed by Face as usize; near layer of each face neighbour
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
        let mut faces = [[PackedLumel::DARK; CHUNK_AREA]; 6];
        for face in Face::ALL {
            let Some(g) = grid_at(face) else { continue };
            g.copy_face(face, &mut faces[face as usize]);
        }
        Self { faces }
    }

    /// Overwrite one cell of the layer across `face` (`i = a + b·16`): a seam patch.
    #[inline]
    pub(in crate::world) fn set(&mut self, face: Face, i: usize, v: PackedLumel) {
        self.faces[face as usize][i] = v;
    }

    /// Light value from neighbour across `face` at coords `(a, b)`.
    #[cfg(test)]
    #[inline]
    pub(in crate::world) fn at(&self, face: Face, a: usize, b: usize) -> Lumel {
        self.faces[face as usize][a + b * CHUNK_SIZE].unpack()
    }

    /// All-dark shell (no neighbours).
    pub fn dark() -> Self {
        Self { faces: [[PackedLumel::DARK; CHUNK_AREA]; 6] }
    }
}

/// The skylight ceiling per face-local column: the altitude at and above which
/// a column is open sky. Seeded from the generator's ground altitude (a pure
/// function, so caves stay consistently dark regardless of chunk load order),
/// then RAISED by edited opaque roofs ([`raise`](Self::raise)) so a player-built
/// ceiling shadows every chunk below it instead of leaking full skylight.
/// Columns are indexed `lu + lv * 16` in face-local order. PosY's `(lu, lv)` is
/// `(x, z)`.
#[derive(Clone)]
pub struct CeilingWindow {
    /// Which face's `(u, v)` orders `surface`. The flood takes that face as a const.
    #[allow(dead_code)]
    face: Face,
    surface: [i32; CHUNK_AREA],
    /// Lowest altitude at which every column is open sky — `max` of `surface`.
    /// A chunk at `alt0` is all-open when `alt0 >= min_surface`.
    min_surface: i32,
}

impl CeilingWindow {
    /// Surface altitude per face-local column via generator callback.
    pub fn from_heights(face: Face, mut height: impl FnMut(usize, usize) -> i32) -> Self {
        let mut surface = [0i32; CHUNK_AREA];
        let mut min_surface = i32::MIN;
        for lv in 0..CHUNK_SIZE {
            for lu in 0..CHUNK_SIZE {
                let h = height(lu, lv);
                surface[lu + lv * CHUNK_SIZE] = h;
                min_surface = min_surface.max(h);
            }
        }
        Self { face, surface, min_surface }
    }

    /// Everything open to the sky — for tests and the neutral path.
    #[cfg(test)]
    pub fn open() -> Self {
        Self { face: Face::PosY, surface: [i32::MIN; CHUNK_AREA], min_surface: i32::MIN }
    }

    #[inline]
    pub(in crate::world) fn open_above(&self, lu: usize, lv: usize, alt: i32) -> bool {
        alt >= self.surface[lu + lv * CHUNK_SIZE]
    }

    /// Lowest altitude at which every column is open sky.
    #[inline]
    pub(in crate::world) fn min_surface(&self) -> i32 {
        self.min_surface
    }

    /// Raise one column's ceiling to at least `surface` (a constructed opaque
    /// roof: open sky begins at the cell ABOVE it). Never lowers — the
    /// generator ground below stays the floor of the value. The all-open
    /// altitude can only stay or rise.
    pub(in crate::world) fn raise(&mut self, lu: usize, lv: usize, surface: i32) {
        let cell = &mut self.surface[lu + lv * CHUNK_SIZE];
        *cell = (*cell).max(surface);
        self.min_surface = self.min_surface.max(*cell);
    }

    /// The altitude at which this column becomes open sky (see [`open_above`](Self::open_above)).
    pub(in crate::world) fn surface_at(&self, lu: usize, lv: usize) -> i32 {
        self.surface[lu + lv * CHUNK_SIZE]
    }
}

/// Ceiling the Open kernel ignores. Not on the PosY path.
pub(in crate::world) fn ignored_ceiling() -> Arc<CeilingWindow> {
    static CEILING: OnceLock<Arc<CeilingWindow>> = OnceLock::new();
    Arc::clone(CEILING.get_or_init(|| {
        Arc::new(CeilingWindow::from_heights(Face::PosY, |_, _| i32::MIN))
    }))
}

/// Reusable flood scratch for [`propagate`]: the packed working grid and the
/// BFS frontier. Held thread-local so a worker's repeated `propagate` calls
/// reuse one allocation each instead of allocating a `CHUNK_VOLUME` box and a
/// growing `VecDeque` per job (the light-settle churn). Never shared or sent
/// across threads (borrowed only for the duration of one `propagate` call), so it
/// is sound to key on the calling worker.
struct FloodScratch {
    queue: VecDeque<usize>,
    /// Dense packed box recycled across `propagate` calls on this thread.
    /// Moved into the output grid when the flood stays dense.
    cells: Option<Box<[PackedLumel; CHUNK_VOLUME]>>,
    /// Blocklight shell seeds applied after emitters so the queue is
    /// emitters then faces.
    shell_block: Vec<(usize, u8)>,
}

thread_local! {
    static FLOOD: RefCell<FloodScratch> = const {
        RefCell::new(FloodScratch {
            queue: VecDeque::new(),
            cells: None,
            shell_block: Vec::new(),
        })
    };
}

/// Recompute chunk light from scratch. Light removal needs no second pass:
/// breaking emitters or placing blocks just lowers the grid. `alt0` is the
/// chunk's minimum altitude along `sky` (for PosY, the chunk's world Y origin).
#[inline]
pub fn propagate(
    chunk: &Chunk,
    shell: &FaceShell,
    ceiling: &CeilingWindow,
    sky: Sky,
    alt0: i32,
    tables: &HotTables,
    out: &mut LightGrid,
) {
    // Heights are stored in this face's `(u, v)` order. Open ignores the window.
    debug_assert!(match sky {
        Sky::Open => true,
        Sky::Axis(face) => face == ceiling.face,
    });
    match sky {
        Sky::Axis(Face::NegX) => propagate_sky::<0>(chunk, shell, ceiling, alt0, tables, out),
        Sky::Axis(Face::PosX) => propagate_sky::<1>(chunk, shell, ceiling, alt0, tables, out),
        Sky::Axis(Face::NegZ) => propagate_sky::<2>(chunk, shell, ceiling, alt0, tables, out),
        Sky::Axis(Face::PosZ) => propagate_sky::<3>(chunk, shell, ceiling, alt0, tables, out),
        Sky::Axis(Face::NegY) => propagate_sky::<4>(chunk, shell, ceiling, alt0, tables, out),
        Sky::Axis(Face::PosY) => propagate_sky::<5>(chunk, shell, ceiling, alt0, tables, out),
        Sky::Open => propagate_sky::<{ Sky::OPEN_CODE }>(chunk, shell, ceiling, alt0, tables, out),
    }
}

/// `SKY` is a [`Face`] discriminant, or [`Sky::OPEN_CODE`]. PosY (`5`) keeps the
/// original column-mask seed and the −Y full-strength step.
#[inline]
fn propagate_sky<const SKY: u8>(
    chunk: &Chunk,
    shell: &FaceShell,
    ceiling: &CeilingWindow,
    alt0: i32,
    tables: &HotTables,
    out: &mut LightGrid,
) {
    // Decode the opacity field ONCE (payload-specialized, ~a palette pass)
    // into an L1-resident bitset plus per-column occupancy: the flood probes
    // the bitset ~6 times per relaxed cell, and the sky seed is one
    // `leading_zeros` per column.
    let mut opaque_bits = [0u64; CHUNK_VOLUME / 64];
    let mut col = [0u16; CHUNK_AREA];
    chunk.fill_opacity(|id| tables.opaque(id), &mut opaque_bits, &mut col);
    let opaque_at = |i: usize| (opaque_bits[i >> 6] >> (i & 63)) & 1 != 0;

    // Borrow thread-local scratch. Recycled cells are filled dark below so no
    // stale flood state from a prior job survives.
    let (mut queue, tls_cells, mut shell_block) = FLOOD.with_borrow_mut(|s| {
        (std::mem::take(&mut s.queue), s.cells.take(), std::mem::take(&mut s.shell_block))
    });
    let (mut cells, leftover) = match std::mem::replace(&mut out.0, Repr::Uniform(Lumel::DARK)) {
        Repr::Cells(c) => (c, tls_cells),
        Repr::Uniform(_) => (tls_cells.unwrap_or_else(|| alloc_cells(PackedLumel::DARK)), None),
    };
    cells.fill(PackedLumel::DARK);
    queue.clear();
    shell_block.clear();
    // Seed 1: open sky from the chunk's high-altitude end along −n until the
    // first opaque voxel. Open has no ceiling seed. PosY uses the Y occupancy
    // mask's `leading_zeros` (empty run from y=15); push order is y=15,14,..
    // per column, x-inner z-outer.
    if SKY == 5 {
        let top_y = alt0 + CHUNK_SIZE as i32;
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                if !ceiling.open_above(x, z, top_y) {
                    continue;
                }
                let n = col[x + z * CHUNK_SIZE].leading_zeros() as usize;
                for k in 0..n {
                    let y = CHUNK_SIZE - 1 - k;
                    let i = x + z * STRIDE_Z + y * STRIDE_Y;
                    cells[i] = PackedLumel::OPEN_SKY;
                    queue.push_back(i);
                }
            }
        }
    } else if SKY != Sky::OPEN_CODE {
        seed_axis_ceiling::<SKY>(ceiling, alt0, &opaque_bits, &col, &mut cells, &mut queue);
    }
    // Seed 2: one 6×256 walk writes sky now and stashes blocklight for after
    // the emitter scan, so both channels share the face-index table.
    seed_from_shell::<SKY>(shell, |i, s| {
        if s > cells[i].sky() {
            cells[i] = cells[i].with_sky(s);
            queue.push_back(i);
        }
    }, |i, b| {
        shell_block.push((i, b));
    });
    // Flood: -1 per step, except full skylight passes straight along −n
    // (open columns stay lit). Open keeps full sky in every direction.
    while let Some(i) = queue.pop_front() {
        let level = cells[i].sky();
        let (x, y, z) = Chunk::local_of(i);
        let mut relax = |ni: usize, down: bool| {
            if opaque_at(ni) {
                return;
            }
            let cand = if down && level == MAX_LIGHT { MAX_LIGHT } else { level.saturating_sub(1) };
            if cand > cells[ni].sky() {
                cells[ni] = cells[ni].with_sky(cand);
                queue.push_back(ni);
            }
        };
        if SKY == 5 {
            if x > 0 { relax(i - 1, false); }
            if x + 1 < CHUNK_SIZE { relax(i + 1, false); }
            if y > 0 { relax(i - STRIDE_Y, true); }
            if y + 1 < CHUNK_SIZE { relax(i + STRIDE_Y, false); }
            if z > 0 { relax(i - STRIDE_Z, false); }
            if z + 1 < CHUNK_SIZE { relax(i + STRIDE_Z, false); }
        } else if SKY == Sky::OPEN_CODE {
            if x > 0 { relax(i - 1, true); }
            if x + 1 < CHUNK_SIZE { relax(i + 1, true); }
            if y > 0 { relax(i - STRIDE_Y, true); }
            if y + 1 < CHUNK_SIZE { relax(i + STRIDE_Y, true); }
            if z > 0 { relax(i - STRIDE_Z, true); }
            if z + 1 < CHUNK_SIZE { relax(i + STRIDE_Z, true); }
        } else {
            // Full-strength step along −n. Codes: NegX +X, PosX −X, NegZ +Z, PosZ −Z, NegY +Y.
            let down_xm = SKY == 1;
            let down_xp = SKY == 0;
            let down_yp = SKY == 4;
            let down_zm = SKY == 3;
            let down_zp = SKY == 2;
            if x > 0 { relax(i - 1, down_xm); }
            if x + 1 < CHUNK_SIZE { relax(i + 1, down_xp); }
            if y > 0 { relax(i - STRIDE_Y, false); }
            if y + 1 < CHUNK_SIZE { relax(i + STRIDE_Y, down_yp); }
            if z > 0 { relax(i - STRIDE_Z, down_zm); }
            if z + 1 < CHUNK_SIZE { relax(i + STRIDE_Z, down_zp); }
        }
    }

    // Blocklight: low nibble is still dark except where we write emitters, then
    // the stashed shell (emitters then faces).
    queue.clear();
    chunk.for_each_emission(&tables.emission, |i, em| {
        let em = if em > MAX_LIGHT { MAX_LIGHT } else { em };
        cells[i] = cells[i].with_block(em);
        queue.push_back(i);
    });
    for &(i, lvl) in &shell_block {
        if lvl > cells[i].block() {
            cells[i] = cells[i].with_block(lvl);
            queue.push_back(i);
        }
    }
    while let Some(i) = queue.pop_front() {
        let level = cells[i].block();
        if level <= 1 {
            continue;
        }
        let cand = level - 1;
        let (x, y, z) = Chunk::local_of(i);
        let mut relax = |ni: usize| {
            if opaque_at(ni) {
                return;
            }
            if cand > cells[ni].block() {
                cells[ni] = cells[ni].with_block(cand);
                queue.push_back(ni);
            }
        };
        if x > 0 { relax(i - 1); }
        if x + 1 < CHUNK_SIZE { relax(i + 1); }
        if y > 0 { relax(i - STRIDE_Y); }
        if y + 1 < CHUNK_SIZE { relax(i + STRIDE_Y); }
        if z > 0 { relax(i - STRIDE_Z); }
        if z + 1 < CHUNK_SIZE { relax(i + STRIDE_Z); }
    }

    // One linear compare after the flood — not per frame. Deep rock and open
    // sky both land here even when the voxel payload is paletted (a cave of
    // air under a closed ceiling is uniformly dark).
    let first = cells[0];
    let recycle = if cells.chunks_exact(64).all(|row| row.iter().all(|&c| c == first)) {
        *out = LightGrid(Repr::Uniform(first.unpack()));
        leftover.or(Some(cells))
    } else {
        *out = LightGrid(Repr::Cells(cells));
        leftover
    };

    // Return the scratch buffers (with their capacity) for the next call.
    FLOOD.with_borrow_mut(|s| {
        s.queue = queue;
        s.cells = recycle;
        s.shell_block = shell_block;
    });
}

/// Interior cell index of face slot `a + b*16` (a inner, b outer — the
/// `seed_from_shell` walk). Built with the same (na, au, av, inner) as the
/// old dynamic-axis loop.
const fn face_index_table() -> [[usize; CHUNK_AREA]; 6] {
    let mut t = [[0usize; CHUNK_AREA]; 6];
    let mut f = 0;
    while f < 6 {
        let (na, au, av, inner): (usize, usize, usize, usize) = match f {
            0 => (0, 1, 2, 0),
            1 => (0, 1, 2, CHUNK_SIZE - 1),
            2 => (2, 0, 1, 0),
            3 => (2, 0, 1, CHUNK_SIZE - 1),
            4 => (1, 0, 2, 0),
            _ => (1, 0, 2, CHUNK_SIZE - 1),
        };
        let mut b = 0;
        while b < CHUNK_SIZE {
            let mut a = 0;
            while a < CHUNK_SIZE {
                let mut c = [0usize; 3];
                c[na] = inner;
                c[au] = a;
                c[av] = b;
                t[f][a + b * CHUNK_SIZE] = c[0] + c[2] * CHUNK_SIZE + c[1] * CHUNK_SIZE * CHUNK_SIZE;
                a += 1;
            }
            b += 1;
        }
        f += 1;
    }
    t
}

const FACE_INDEX: [[usize; CHUNK_AREA]; 6] = face_index_table();

/// 16 occupancy bits along X at fixed `(y, z)`. They sit in one `u64` (the
/// base is a multiple of 16 and `y * 256` is a multiple of 64).
#[inline]
fn mask_x(bits: &[u64; CHUNK_VOLUME / 64], y: usize, z: usize) -> u16 {
    let base = z * STRIDE_Z + y * STRIDE_Y;
    ((bits[base >> 6] >> (base & 63)) & 0xFFFF) as u16
}

/// 16 occupancy bits along Z at fixed `(x, y)`. Stride 16, so this is a gather.
#[inline]
fn mask_z(bits: &[u64; CHUNK_VOLUME / 64], x: usize, y: usize) -> u16 {
    let mut mask = 0u16;
    let mut z = 0;
    while z < CHUNK_SIZE {
        let i = x + z * STRIDE_Z + y * STRIDE_Y;
        if (bits[i >> 6] >> (i & 63)) & 1 != 0 {
            mask |= 1 << z;
        }
        z += 1;
    }
    mask
}

/// Ceiling seed for every axis except +Y (that path is inlined in [`propagate_sky`]).
/// Positive faces walk from local 15 with `leading_zeros`; negative faces walk
/// from local 0 with `trailing_zeros`. Push order is lv-outer, lu-inner.
fn seed_axis_ceiling<const SKY: u8>(
    ceiling: &CeilingWindow,
    alt0: i32,
    opaque_bits: &[u64; CHUNK_VOLUME / 64],
    col: &[u16; CHUNK_AREA],
    cells: &mut [PackedLumel; CHUNK_VOLUME],
    queue: &mut VecDeque<usize>,
) {
    let face = Face::ALL[SKY as usize];
    let frame = FaceFrame::new(face);
    let top = alt0 + CHUNK_SIZE as i32;
    let positive = face.sign() > 0;
    let axis = face.axis();
    for lv in 0..CHUNK_SIZE {
        for lu in 0..CHUNK_SIZE {
            if !ceiling.open_above(lu, lv, top) {
                continue;
            }
            let (mut lx, mut ly, mut lz) = frame.index_to_world(lu, 0, lv);
            let mask = match axis {
                0 => mask_x(opaque_bits, ly, lz),
                1 => col[lx + lz * CHUNK_SIZE],
                _ => mask_z(opaque_bits, lx, ly),
            };
            let n = if positive {
                mask.leading_zeros() as usize
            } else {
                mask.trailing_zeros() as usize
            };
            for k in 0..n {
                let along = if positive { CHUNK_SIZE - 1 - k } else { k };
                match axis {
                    0 => lx = along,
                    1 => ly = along,
                    _ => lz = along,
                }
                let i = Chunk::index(lx, ly, lz);
                cells[i] = PackedLumel::OPEN_SKY;
                queue.push_back(i);
            }
        }
    }
}

/// Seed border cells from neighbour shell faces. Full skylight is kept only
/// across the face whose normal is `+n` (every face when `SKY` is open).
/// Dark shell cells (missing neighbours) don't seed. One 6×256 walk; callers
/// split sky (applied now) from block (stashed until after emitters).
fn seed_from_shell<const SKY: u8>(
    shell: &FaceShell,
    mut sky: impl FnMut(usize, u8),
    mut block: impl FnMut(usize, u8),
) {
    for face in Face::ALL {
        let layer = &shell.faces[face as usize];
        let idx = &FACE_INDEX[face as usize];
        let keep_full_sky = if SKY == 5 {
            face == Face::PosY
        } else {
            SKY == Sky::OPEN_CODE || face as u8 == SKY
        };
        for slot in 0..CHUNK_AREA {
            let src = layer[slot];
            let sky_raw = src.sky();
            let sky_l = if keep_full_sky && sky_raw == MAX_LIGHT {
                MAX_LIGHT
            } else {
                sky_raw.saturating_sub(1)
            };
            let block_l = src.block().saturating_sub(1);
            if sky_l == 0 && block_l == 0 {
                continue;
            }
            let i = idx[slot];
            if sky_l != 0 {
                sky(i, sky_l);
            }
            if block_l != 0 {
                block(i, block_l);
            }
        }
    }
}

/// Whether border changed on a face. Used to enqueue neighbours only when their
/// shared boundary moves (settling convergence detection).
pub(in crate::world) fn border_changed(a: &LightGrid, b: &LightGrid, face: Face) -> bool {
    let idx = &FACE_INDEX[face as usize];
    match (&a.0, &b.0) {
        (Repr::Uniform(x), Repr::Uniform(y)) => x != y,
        (Repr::Uniform(x), Repr::Cells(c)) => {
            let p = PackedLumel::pack(*x);
            idx.iter().any(|&i| c[i] != p)
        }
        (Repr::Cells(c), Repr::Uniform(y)) => {
            let p = PackedLumel::pack(*y);
            idx.iter().any(|&i| c[i] != p)
        }
        (Repr::Cells(ca), Repr::Cells(cb)) => idx.iter().any(|&i| ca[i] != cb[i]),
    }
}

/// The two in-face axes of `face`, ascending.
#[cfg(test)]
#[inline]
fn plane_axes(face: Face) -> (usize, usize) {
    match face.axis() {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    }
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
            vec![Pass::Opaque, Pass::Opaque, Pass::Blend, Pass::Opaque].into(),
            vec![0, 0, 15, 15].into(), // ids 2 and 3 emit 15
            vec![0, 0, 0, 0].into(),
            vec![0, 1, 2, 3].into(),
        )
    }

    fn lit(chunk: &Chunk) -> LightGrid {
        let mut grid = LightGrid::dark();
        propagate(
            chunk,
            &FaceShell::dark(),
            &CeilingWindow::open(),
            Sky::Axis(Face::PosY),
            0,
            &tables(),
            &mut grid,
        );
        grid
    }

    fn grid_hash(grid: &LightGrid) -> u32 {
        use crate::hash::fnv1a_32;
        let mut bytes = [0u8; CHUNK_VOLUME * 2];
        for i in 0..CHUNK_VOLUME {
            let l = grid.at(i);
            bytes[i * 2] = l.sky.get();
            bytes[i * 2 + 1] = l.block.get();
        }
        fnv1a_32(&bytes)
    }

    #[test]
    fn dense_grid_is_one_byte_per_cell() {
        assert_eq!(std::mem::size_of::<PackedLumel>(), 1);
        assert_eq!(LightGrid::dark().allocated_bytes(), 0);
        let dense = LightGrid::open_sky().to_dense();
        assert_eq!(dense.allocated_bytes(), CHUNK_VOLUME);
        assert_eq!(
            dense.at(0),
            Lumel { sky: LightLevel::FULL, block: LightLevel::DARK }
        );
        for sky in 0..=MAX_LIGHT {
            for block in 0..=MAX_LIGHT {
                let l = Lumel { sky: LightLevel::new(sky), block: LightLevel::new(block) };
                assert_eq!(Lumel::unpack(l.pack()), l);
            }
        }
    }

    /// Pin `fnv1a_32` over lumel bytes of four fixed seed-42 `propagate` results.
    /// Values locked before the flood-path rewrite; a mismatch means settled
    /// light bytes moved.
    #[test]
    fn light_byte_pin() {
        use crate::block::registry::BlockRegistry;
        use crate::world::generation::TerrainGenerator;
        use crate::world::terrain::Terrain;

        let mut registry = BlockRegistry::with_builtins();
        let generator = Terrain::new(&mut registry, 42);
        // The pin is the blocklight field of a cell that actually emits — not
        // whatever the "lamp" label happens to intern (a rest-stable centre can
        // still observe as dark). Intern a configuration and assert emission.
        let lumin = {
            use material::Configuration;
            let mut found = None;
            if let Some(lamp) = registry.id_by_label("lamp").filter(|&id| registry.emission(id) >= 8) {
                found = Some(lamp);
            }
            if found.is_none() {
                for n in 0u32..40_000 {
                    let e = material::Element::new([
                        n as u8,
                        (n >> 8) as u8,
                        (n >> 16) as u8,
                        (n >> 24) as u8,
                    ]);
                    let id = registry.intern(&Configuration::single(e)).unwrap();
                    if registry.emission(id) >= 8 {
                        found = Some(id);
                        break;
                    }
                }
            }
            found.expect("observe must reach glow_min for some solid element")
        };
        assert!(
            registry.emission(lumin) >= 8,
            "emissive pin block emission {}",
            registry.emission(lumin)
        );
        let tables = registry.hot_tables();

        let pin = |chunk: &Chunk, shell: &FaceShell, ceiling: &CeilingWindow, world_y0: i32| {
            let mut grid = LightGrid::dark();
            propagate(chunk, shell, ceiling, Sky::Axis(Face::PosY), world_y0, &tables, &mut grid);
            grid_hash(&grid)
        };
        let ceiling_at = |cx: i32, cz: i32| {
            let x0 = cx * CHUNK_SIZE as i32;
            let z0 = cz * CHUNK_SIZE as i32;
            CeilingWindow::from_heights(Face::PosY, |lx, lz| generator.height(x0 + lx as i32, z0 + lz as i32))
        };

        // Surface chunk at the origin column, real ceiling, dark neighbours.
        let cy = generator.height(8, 8).div_euclid(CHUNK_SIZE as i32);
        let surface = Chunk::new(0, cy, 0, &generator);
        let surface_hash = pin(&surface, &FaceShell::dark(), &ceiling_at(0, 0), cy * CHUNK_SIZE as i32);

        // Cave-band chunk, real ceiling (surface well above), dark neighbours.
        let cave = Chunk::new(0, -3, 0, &generator);
        let cave_hash = pin(&cave, &FaceShell::dark(), &ceiling_at(0, 0), -3 * CHUNK_SIZE as i32);

        // Same cave chunk with a Lumin cell via `set_index`, closed ceiling so
        // the pin is the blocklight field.
        let mut emissive = Chunk::new(0, -3, 0, &generator);
        emissive.set_index(Chunk::index(8, 8, 8), lumin);
        let closed = CeilingWindow::from_heights(Face::PosY, |_, _| 1000);
        let emissive_hash = pin(&emissive, &FaceShell::dark(), &closed, -3 * CHUNK_SIZE as i32);

        // All-air under a checkerboard ceiling, plus a patterned neighbour
        // shell so the pin covers `seed_from_shell`.
        let air = Chunk::from_uniform(0, 2, 0, BlockId(0));
        let partial = CeilingWindow::from_heights(Face::PosY, |lx, lz| {
            if (lx + lz) % 2 == 0 { 100 } else { i32::MIN }
        });
        let mut nbr = LightGrid::dark();
        for i in 0..CHUNK_VOLUME {
            let (x, y, z) = Chunk::local_of(i);
            nbr.set(
                i,
                Lumel {
                    sky: LightLevel::new(((x + y) % 16) as u8),
                    block: LightLevel::new(((z * 3) % 16) as u8),
                },
            );
        }
        let shell = FaceShell::capture(|_| Some(&nbr));
        let air_hash = pin(&air, &shell, &partial, 2 * CHUNK_SIZE as i32);

        let pins: [(&str, u32, u32); 4] = [
            ("surface", surface_hash, 0xf5c30be2),
            ("cave", cave_hash, 0xb4ffde98),
            ("emissive", emissive_hash, 0x40686e77),
            ("air", air_hash, 0x19839265),
        ];
        for (name, got, _) in pins {
            println!("light_byte_pin {name}=0x{got:08x}");
        }
        for (name, got, want) in pins {
            assert_eq!(got, want, "{name}");
        }
    }

    #[test]
    fn face_index_table_matches_axis_walk() {
        for face in Face::ALL {
            let na = face.axis();
            let (au, av) = plane_axes(face);
            let inner = match face {
                Face::PosX | Face::PosY | Face::PosZ => CHUNK_SIZE - 1,
                _ => 0,
            };
            for b in 0..CHUNK_SIZE {
                for a in 0..CHUNK_SIZE {
                    let mut ci = [0usize; 3];
                    ci[na] = inner;
                    ci[au] = a;
                    ci[av] = b;
                    assert_eq!(
                        FACE_INDEX[face as usize][a + b * CHUNK_SIZE],
                        Chunk::index(ci[0], ci[1], ci[2]),
                        "{face:?} slot ({a},{b})"
                    );
                }
            }
        }
    }

    /// Full settle-flood cost for a surface-band chunk — the gauge for the
    /// propagate opacity-bitset redesign. Ignored: a timing benchmark, not a
    /// correctness gate. Run with
    /// `cargo test --release light_propagate_throughput -- --ignored --nocapture`.
    /// 2026-07-19 (12-core box), per-probe `get_local`: ~18.1k settles/s;
    /// decoded opacity bitset: ~32.6k settles/s (1.8×).
    /// 2026-09-09, flood-path rewrite (skip empty emitter scan, column-mask
    /// sky seed, flat-index relax, one-pass shell seed): before 34.3k
    /// settles/s (median of 3: 33.4k / 34.3k / 34.8k); after 37.6k
    /// settles/s (median of 3: 35.5k / 37.6k / 38.0k).
    #[test]
    #[ignore]
    fn light_propagate_throughput() {
        use crate::block::registry::BlockRegistry;
        use crate::world::generation::TerrainGenerator;
        use crate::world::terrain::Terrain;

        let mut registry = BlockRegistry::with_builtins();
        let generator = Terrain::new(&mut registry, 5);
        // The surface chunk at the origin: the Dense band every load floods
        // (deep/sky chunks take the analytic fast paths and never get here).
        let cy = generator.height(0, 0).div_euclid(CHUNK_SIZE as i32);
        let chunk = Chunk::new(0, cy, 0, &generator);
        let tables = registry.hot_tables();
        let shell = FaceShell::dark();
        let ceiling = CeilingWindow::from_heights(Face::PosY, |lx, lz| generator.height(lx as i32, lz as i32));
        let mut out = LightGrid::dark();

        const N: usize = 4000;
        let start = std::time::Instant::now();
        for _ in 0..N {
            propagate(
                &chunk,
                &shell,
                &ceiling,
                Sky::Axis(Face::PosY),
                cy * CHUNK_SIZE as i32,
                &tables,
                &mut out,
            );
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
    fn min_surface_is_the_all_open_y() {
        let c = CeilingWindow::from_heights(Face::PosY, |lx, lz| 10 + (lx + lz) as i32);
        assert_eq!(c.min_surface(), 10 + 2 * (CHUNK_SIZE as i32 - 1));
        let y0 = c.min_surface();
        assert!(
            (0..CHUNK_SIZE).all(|lz| (0..CHUNK_SIZE).all(|lx| c.open_above(lx, lz, y0))),
            "y0 >= min_surface ⇒ every column is open"
        );
        assert!(
            !(0..CHUNK_SIZE).all(|lz| (0..CHUNK_SIZE).all(|lx| c.open_above(lx, lz, y0 - 1))),
            "one below min_surface is not all-open"
        );
        let mut raised = CeilingWindow::from_heights(Face::PosY, |_, _| 10);
        assert_eq!(raised.min_surface(), 10);
        raised.raise(0, 0, 40);
        assert_eq!(raised.min_surface(), 40);
        raised.raise(1, 1, 20);
        assert_eq!(raised.min_surface(), 40, "a lower raise must not drop the all-open Y");
    }

    #[test]
    fn border_blocklight_ignores_interior_and_level_one() {
        assert!(!LightGrid::dark().has_border_blocklight());
        assert!(!LightGrid::open_sky().has_border_blocklight());
        assert!(LightGrid::full().has_border_blocklight());
        let mut g = LightGrid::dark();
        g.set(
            Chunk::index(0, 8, 8),
            Lumel {
                sky: LightLevel::DARK,
                block: LightLevel::new(15),
            },
        );
        assert!(g.has_border_blocklight());
        let mut g = LightGrid::dark();
        g.set(
            Chunk::index(8, 8, 8),
            Lumel {
                sky: LightLevel::DARK,
                block: LightLevel::new(15),
            },
        );
        assert!(
            !g.has_border_blocklight(),
            "interior-only blocklight cannot seed a neighbour"
        );
        let mut g = LightGrid::dark();
        g.set(
            Chunk::index(0, 8, 8),
            Lumel {
                sky: LightLevel::DARK,
                block: LightLevel::new(1),
            },
        );
        assert!(
            !g.has_border_blocklight(),
            "level 1 attenuates to 0 crossing the border"
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
        propagate(
            &opaque,
            &FaceShell::dark(),
            &CeilingWindow::from_heights(Face::PosY, |_, _| 100),
            Sky::Axis(Face::PosY),
            -160,
            &tables,
            &mut got,
        );
        assert!(got == LightGrid::dark(), "uniform opaque == dark()");
        assert!(
            matches!(got.0, Repr::Uniform(v) if v == Lumel::DARK),
            "uniform opaque collapses to Uniform(dark)"
        );
        // Opaque does not imply dark: an opaque emitter must bypass the analytic
        // shortcut and seed blocklight in the regular propagation path.
        let emissive = Chunk::from_uniform(0, -10, 0, BlockId(3));
        assert!(!emissive.is_uniform_opaque(&tables));
        let mut got = LightGrid::dark();
        propagate(
            &emissive,
            &FaceShell::dark(),
            &CeilingWindow::from_heights(Face::PosY, |_, _| 100),
            Sky::Axis(Face::PosY),
            -160,
            &tables,
            &mut got,
        );
        assert_eq!(got.at(Chunk::index(8, 8, 8)).block, LightLevel::FULL);
        // Uniform air fully open to the sky → full sky, no blocklight.
        let air = Chunk::from_uniform(0, 10, 0, BlockId(0));
        let mut got = LightGrid::dark();
        propagate(
            &air,
            &FaceShell::dark(),
            &CeilingWindow::open(),
            Sky::Axis(Face::PosY),
            160,
            &tables,
            &mut got,
        );
        assert!(got == LightGrid::open_sky(), "open-sky air == open_sky()");
        assert!(
            matches!(
                got.0,
                Repr::Uniform(v) if v == Lumel { sky: LightLevel::FULL, block: LightLevel::DARK }
            ),
            "open-sky air collapses to Uniform(open sky lumel)"
        );
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

        let stone = world.registry().id_by_label("rock").expect("builtin Stone");
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
            Sky::Axis(Face::PosY),
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
            Sky::Axis(Face::PosY),
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
        let ceiling = CeilingWindow::from_heights(Face::PosY, |_, _| 40); // surface well above this chunk
        let mut grid = LightGrid::dark();
        propagate(
            &chunk,
            &FaceShell::dark(),
            &ceiling,
            Sky::Axis(Face::PosY),
            -128,
            &tables(),
            &mut grid,
        );

        assert_eq!(grid.at(Chunk::index(4, 8, 4)).sky, LightLevel::DARK, "cavern dark");
        assert_eq!(grid.at(Chunk::index(0, 0, 0)).sky, LightLevel::DARK, "cavern floor dark");
        assert!(
            grid.is_uniform(),
            "all-dark mixed-voxel (air) chunk collapses to Uniform"
        );
        assert!(grid == LightGrid::dark());
    }

    #[test]
    fn blocklight_falls_off_by_one_per_step() {
        let mut chunk = Chunk::from_cells(0, 0, 0, Box::new([BlockId(0); CHUNK_VOLUME]));
        chunk.set_local(8, 8, 8, BlockId(2)); // emitter, level 15
        let ceiling = CeilingWindow::from_heights(Face::PosY, |_, _| 100); // fully underground: isolate blocklight
        let mut grid = LightGrid::dark();
        propagate(
            &chunk,
            &FaceShell::dark(),
            &ceiling,
            Sky::Axis(Face::PosY),
            0,
            &tables(),
            &mut grid,
        );

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
            let na = face.axis();
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
        let ceiling = CeilingWindow::from_heights(Face::PosY, |_, _| 100); // underground: isolate blocklight

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
            propagate(
                &left_c,
                &shell(Face::PosX, &right),
                &ceiling,
                Sky::Axis(Face::PosY),
                0,
                &tables,
                &mut new_left,
            );
            let mut new_right = LightGrid::dark();
            propagate(
                &right_c,
                &shell(Face::NegX, &new_left),
                &ceiling,
                Sky::Axis(Face::PosY),
                0,
                &tables,
                &mut new_right,
            );
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

    #[test]
    fn uniform_and_dense_grids_compare_by_value() {
        let uni = LightGrid::open_sky();
        let dense = uni.to_dense();
        assert!(std::mem::size_of::<LightGrid>() <= 16);
        assert!(uni == dense, "Uniform equals its dense expansion");
        assert!(dense == uni);
        assert!(LightGrid::dark() == LightGrid::dark().to_dense());
        assert!(LightGrid::full() == LightGrid::full().to_dense());
        assert!(uni != LightGrid::dark());
        assert!(uni != LightGrid::dark().to_dense());
        assert!(uni.to_dense() != LightGrid::dark());

        let mut mixed = LightGrid::open_sky().to_dense();
        mixed.set(0, Lumel::DARK);
        assert!(mixed != uni);
        assert!(uni != mixed);

        let mut written = LightGrid::dark();
        assert!(written.is_uniform());
        written.set(Chunk::index(8, 8, 8), Lumel::FULL);
        assert!(!written.is_uniform(), "set densifies a Uniform grid");
        assert_eq!(written.at(Chunk::index(8, 8, 8)), Lumel::FULL);
        assert_eq!(written.at(0), Lumel::DARK);

        for face in Face::ALL {
            assert!(!border_changed(&uni, &dense, face), "{face:?} same values");
            assert!(!border_changed(&dense, &uni, face), "{face:?} same values swapped");
            assert!(
                border_changed(&uni, &LightGrid::dark(), face),
                "{face:?} uniform vs different uniform"
            );
            assert!(
                border_changed(&uni, &LightGrid::dark().to_dense(), face),
                "{face:?} uniform vs different dense"
            );
            assert!(!border_changed(
                &LightGrid::dark(),
                &LightGrid::dark().to_dense(),
                face
            ));
        }
        // Cell 0 sits on NegX/NegY/NegZ; the opposite faces stay equal.
        assert!(border_changed(&mixed, &uni, Face::NegX));
        assert!(!border_changed(&mixed, &uni, Face::PosX));

        let shell_u = FaceShell::capture(|_| Some(&uni));
        let shell_d = FaceShell::capture(|_| Some(&dense));
        for face in Face::ALL {
            for b in 0..CHUNK_SIZE {
                for a in 0..CHUNK_SIZE {
                    assert_eq!(shell_u.at(face, a, b), shell_d.at(face, a, b), "face {face:?}");
                }
            }
        }
    }

    #[test]
    fn uniform_and_dense_padded_light_meshes_byte_identically() {
        use crate::world::mesh::{self, new_chunk_mesh_data};

        let mut chunk = Chunk::from_uniform(0, 0, 0, BlockId(0));
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                chunk.set_local(x, 0, z, BlockId(1));
            }
        }
        let padded = mesh::Padded::capture(|dx, dy, dz| {
            (dx == 0 && dy == 0 && dz == 0).then_some(&chunk)
        });

        let uniform_grids: [LightGrid; 27] = std::array::from_fn(|k| match k % 3 {
            0 => LightGrid::open_sky(),
            1 => LightGrid::dark(),
            _ => LightGrid::full(),
        });
        let dense_grids: [LightGrid; 27] = std::array::from_fn(|k| uniform_grids[k].to_dense());
        let idx = |dx: i32, dy: i32, dz: i32| ((dx + 1) + (dz + 1) * 3 + (dy + 1) * 9) as usize;
        let uni_pad = PaddedLight::capture(|dx, dy, dz| Some(&uniform_grids[idx(dx, dy, dz)]));
        let dense_pad = PaddedLight::capture(|dx, dy, dz| Some(&dense_grids[idx(dx, dy, dz)]));
        for y in -1..=CS {
            for z in -1..=CS {
                for x in -1..=CS {
                    assert_eq!(uni_pad.at(x, y, z), dense_pad.at(x, y, z), "padded ({x},{y},{z})");
                }
            }
        }

        let mut a = new_chunk_mesh_data();
        let mut b = new_chunk_mesh_data();
        mesh::build_chunk_mesh(&padded, chunk.uniform(), &tables(), &uni_pad, &mut a);
        mesh::build_chunk_mesh(&padded, chunk.uniform(), &tables(), &dense_pad, &mut b);
        for (pass, va) in a.iter() {
            let vb = &b[pass];
            assert_eq!(va.vertices(), vb.vertices(), "vertices {pass:?}");
            assert_eq!(va.quad_counts(), vb.quad_counts(), "quad_counts {pass:?}");
        }
    }

    /// Rotating a chunk, its shell and its ceiling onto face `f` and propagating
    /// with `Sky::Axis(f)` matches the rotated PosY grid. The fixpoint is unique,
    /// so a different BFS order still has to land on the same bytes.
    #[test]
    fn light_is_equivariant_under_every_face_frame() {
        use voxel_engine::DVec3;

        fn xs(s: &mut u32) -> u32 {
            *s ^= s.wrapping_shl(13);
            *s ^= s.wrapping_shr(17);
            *s ^= s.wrapping_shl(5);
            *s
        }

        fn rotate_shell(frame: FaceFrame, src: &FaceShell) -> FaceShell {
            let mut faces = [[PackedLumel::DARK; CHUNK_AREA]; 6];
            let mut seen = [[false; CHUNK_AREA]; 6];
            for src_face in Face::ALL {
                let (dx, dy, dz) = src_face.delta();
                let (wx, wy, wz) = frame.cell_to_world((dx, dy, dz));
                let dst_face = Face::from_dominant(DVec3::new(wx as f64, wy as f64, wz as f64));
                for slot in 0..CHUNK_AREA {
                    let src_i = FACE_INDEX[src_face as usize][slot];
                    let (x, y, z) = Chunk::local_of(src_i);
                    let (ix, iy, iz) = frame.index_to_world(x, y, z);
                    let dst_i = Chunk::index(ix, iy, iz);
                    let dst_slot = FACE_INDEX[dst_face as usize]
                        .iter()
                        .position(|&i| i == dst_i)
                        .expect("rotated border cell stays on the image face");
                    assert!(!seen[dst_face as usize][dst_slot], "shell slot collision");
                    seen[dst_face as usize][dst_slot] = true;
                    faces[dst_face as usize][dst_slot] = src.faces[src_face as usize][slot];
                }
            }
            assert!(seen.iter().all(|row| row.iter().all(|&s| s)));
            FaceShell { faces }
        }

        let tables = tables();
        let alt0 = 100;
        for seed in [0x51_u32, 0xA5] {
            let mut state = seed;
            let mut src_cells = [BlockId(0); CHUNK_VOLUME];
            for cell in &mut src_cells {
                *cell = match xs(&mut state) % 8 {
                    0 => BlockId(1),
                    1 => BlockId(3),
                    _ => BlockId(0),
                };
            }
            let mut shell_faces = [[PackedLumel::DARK; CHUNK_AREA]; 6];
            for face in &mut shell_faces {
                for slot in face.iter_mut() {
                    let sky = (xs(&mut state) % 16) as u8;
                    let block = (xs(&mut state) % 16) as u8;
                    *slot = PackedLumel::pack(Lumel {
                        sky: LightLevel::new(sky),
                        block: LightLevel::new(block),
                    });
                }
            }
            let src_shell = FaceShell { faces: shell_faces };
            let mut heights = [0i32; CHUNK_AREA];
            for h in &mut heights {
                *h = match xs(&mut state) % 5 {
                    0 => i32::MIN,
                    1 => alt0 - 5,
                    2 => alt0 + 8,
                    3 => alt0 + 16,
                    _ => alt0 + 20,
                };
            }
            let src_chunk = Chunk::from_cells(0, 0, 0, Box::new(src_cells));
            let src_ceiling = CeilingWindow::from_heights(Face::PosY, |u, v| heights[u + v * 16]);
            let mut src_grid = LightGrid::dark();
            propagate(
                &src_chunk,
                &src_shell,
                &src_ceiling,
                Sky::Axis(Face::PosY),
                alt0,
                &tables,
                &mut src_grid,
            );

            for face in Face::ALL {
                let frame = FaceFrame::new(face);
                let mut image = [BlockId(0); CHUNK_VOLUME];
                for i in 0..CHUNK_VOLUME {
                    let (u, a, v) = Chunk::local_of(i);
                    let id = src_chunk.get_local(u, a, v);
                    let (x, y, z) = frame.index_to_world(u, a, v);
                    image[Chunk::index(x, y, z)] = id;
                }
                let image_chunk = Chunk::from_cells(0, 0, 0, Box::new(image));
                let image_shell = rotate_shell(frame, &src_shell);
                let image_ceiling = CeilingWindow::from_heights(face, |u, v| heights[u + v * 16]);
                let mut image_grid = LightGrid::dark();
                propagate(
                    &image_chunk,
                    &image_shell,
                    &image_ceiling,
                    Sky::Axis(face),
                    alt0,
                    &tables,
                    &mut image_grid,
                );
                for i in 0..CHUNK_VOLUME {
                    let (u, a, v) = Chunk::local_of(i);
                    let (x, y, z) = frame.index_to_world(u, a, v);
                    assert_eq!(
                        image_grid.at(Chunk::index(x, y, z)),
                        src_grid.at(i),
                        "{face:?} seed {seed:#x} cell ({u},{a},{v})"
                    );
                }
            }
        }
    }
}
