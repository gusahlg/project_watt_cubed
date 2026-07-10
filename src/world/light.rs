//! Cross-chunk lighting (v2.1). A [`LightGrid`] holds skylight and blocklight
//! (each `0..=15`) for every cell of one chunk. It is computed by [`propagate`]
//! as a function of the chunk's own voxels, its neighbour light shell
//! ([`PaddedLight`]), and the column ceiling ([`CeilingWindow`], the skylight
//! source). Settling is *decoupled* from meshing: a cheap main-thread
//! Gauss-Seidel pass (`World::settle_light`) relaxes the field over a worklist,
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
//! Propagation is a pure function of an owned snapshot ([`Padded`] voxels +
//! [`PaddedLight`] shell + [`CeilingWindow`]), so the settle pass can run it on
//! the main thread against the latest neighbour grids without any snapshot copy.
use std::collections::VecDeque;

use crate::block::registry::HotTables;
use crate::coord::Face;

use super::chunk::{CHUNK_SIZE, CHUNK_VOLUME, Chunk};
use super::mesh::Padded;

/// Maximum light level; the 4-bit domain the packed vertex stores.
pub const MAX_LIGHT: u8 = 15;
/// Cells in one chunk face.
pub const CHUNK_AREA: usize = CHUNK_SIZE * CHUNK_SIZE;
/// Chunk size as a signed coordinate, for the `-1..=16` padded range.
const CS: i32 = CHUNK_SIZE as i32;
/// Padded light shell edge: the 16 chunk cells plus one shell cell each side.
const PADL: usize = CHUNK_SIZE + 2;

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
    /// The lattice join used by relaxation — the brighter of the two.
    #[inline]
    pub fn brighter(self, o: Self) -> Self {
        Self(self.0.max(o.0))
    }
}

/// Skylight and blocklight paired; prevents channel desyncs.
#[derive(Clone, Copy, PartialEq, Eq)]
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

    #[inline]
    pub fn at(&self, idx: usize) -> Lumel {
        self.cells[idx]
    }
    #[inline]
    fn set(&mut self, idx: usize, v: Lumel) {
        self.cells[idx] = v;
    }
}

/// Light grid plus one-cell shell from 26 neighbours (coords -1..=16). Serves
/// interior, border, and diagonal cells for smooth light across chunk borders.
/// Settling reads only the six face layers; missing neighbours are dark.
pub struct PaddedLight {
    cells: Box<[Lumel]>, // PADL^3
}

impl PaddedLight {
    #[inline]
    fn index(x: i32, y: i32, z: i32) -> usize {
        (x + 1) as usize + (z + 1) as usize * PADL + (y + 1) as usize * PADL * PADL
    }

    /// Light at signed coord (x, y, z) in -1..=16.
    #[inline]
    pub(in crate::world) fn at(&self, x: i32, y: i32, z: i32) -> Lumel {
        self.cells[Self::index(x, y, z)]
    }

    /// An all-dark shell (no neighbour light anywhere) — the neutral settle path.
    pub fn dark() -> Self {
        Self { cells: vec![Lumel::DARK; PADL * PADL * PADL].into() }
    }

    /// An all-full-bright shell — the neutral mesher path (tests).
    pub fn full() -> Self {
        Self { cells: vec![Lumel::FULL; PADL * PADL * PADL].into() }
    }

    /// A shell filled from a per-cell closure over signed coords `-1..=16` — for
    /// exercising the mesher's smooth-light sampling with a known field.
    #[cfg(test)]
    pub fn from_fn(f: impl Fn(i32, i32, i32) -> Lumel) -> Self {
        let mut cells = vec![Lumel::DARK; PADL * PADL * PADL];
        for y in -1..=CS {
            for z in -1..=CS {
                for x in -1..=CS {
                    cells[Self::index(x, y, z)] = f(x, y, z);
                }
            }
        }
        Self { cells: cells.into() }
    }

    /// Copy the chunk and its shell out of the light field. `grid_at(dx, dy, dz)`
    /// yields the [`LightGrid`] at chunk-offset `(dx, dy, dz)` (each `∈ -1..=1`,
    /// `(0,0,0)` is the chunk itself), or `None` (→ dark). Mirrors
    /// [`Padded::capture`] cell-for-cell.
    pub fn capture<'a>(grid_at: impl Fn(i32, i32, i32) -> Option<&'a LightGrid>) -> Self {
        let neigh: [Option<&LightGrid>; 27] =
            std::array::from_fn(|k| grid_at(k as i32 % 3 - 1, k as i32 / 9 - 1, k as i32 / 3 % 3 - 1));
        let get = |dx: i32, dy: i32, dz: i32| neigh[((dx + 1) + (dz + 1) * 3 + (dy + 1) * 9) as usize];
        let split = |c: i32| -> (i32, usize) {
            if c < 0 {
                (-1, CHUNK_SIZE - 1)
            } else if c >= CS {
                (1, 0)
            } else {
                (0, c as usize)
            }
        };
        let mut cells = vec![Lumel::DARK; PADL * PADL * PADL];
        for y in -1..=CS {
            for z in -1..=CS {
                for x in -1..=CS {
                    let (dx, lx) = split(x);
                    let (dy, ly) = split(y);
                    let (dz, lz) = split(z);
                    if let Some(g) = get(dx, dy, dz) {
                        cells[Self::index(x, y, z)] = g.at(Chunk::index(lx, ly, lz));
                    }
                }
            }
        }
        Self { cells: cells.into() }
    }
}

/// Terrain surface height per column; determines skylight seeding. Pure function
/// of generator (independent of chunk load order), so caves stay consistently dark.
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
    fn open_above(&self, lx: usize, lz: usize, world_y: i32) -> bool {
        world_y >= self.surface[lx + lz * CHUNK_SIZE]
    }
}

/// Recompute chunk light from scratch. Light removal needs no second pass:
/// breaking emitters or placing blocks just lowers the grid. `world_y0` is chunk's Y origin.
pub fn propagate(
    padded: &Padded,
    shell: &PaddedLight,
    ceiling: &CeilingWindow,
    world_y0: i32,
    tables: &HotTables,
    out: &mut LightGrid,
) {
    out.cells.fill(Lumel::DARK);
    let cs = CHUNK_SIZE as i32;
    let opaque_at = |x: i32, y: i32, z: i32| tables.opaque[padded.at(x, y, z).0 as usize];

    // --- Skylight ---------------------------------------------------------
    let mut sky: Box<[LightLevel]> = vec![LightLevel::DARK; CHUNK_VOLUME].into();
    let mut queue: VecDeque<usize> = VecDeque::new();
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

    // --- Blocklight -------------------------------------------------------
    let mut block: Box<[LightLevel]> = vec![LightLevel::DARK; CHUNK_VOLUME].into();
    queue.clear();
    for i in 0..CHUNK_VOLUME {
        let (x, y, z) = Chunk::local_of(i);
        let em = tables.emission[padded.at(x as i32, y as i32, z as i32).0 as usize];
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
}

/// Seed border cells from neighbour shell faces (skylight full-strength from +Y).
/// Dark shell cells (missing neighbours) don't seed.
fn seed_from_shell(shell: &PaddedLight, mut seed: impl FnMut(usize, Lumel)) {
    for face in Face::ALL {
        for (ci, co) in face_cells(face) {
            let src = shell.at(co[0], co[1], co[2]);
            let sky = if face == Face::PosY && src.sky == LightLevel::FULL {
                LightLevel::FULL
            } else {
                src.sky.attenuated()
            };
            let seeded = Lumel { sky, block: src.block.attenuated() };
            if seeded == Lumel::DARK {
                continue;
            }
            seed(Chunk::index(ci[0] as usize, ci[1] as usize, ci[2] as usize), seeded);
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

    fn tables() -> HotTables {
        HotTables {
            solid: vec![false, true, true].into(),
            opaque: vec![false, true, false].into(), // id 1 opaque (stone), id 2 clear
            emission: vec![0, 0, 15].into(),         // id 2 emits 15
        }
    }

    fn solo(chunk: &Chunk) -> Padded {
        Padded::capture(|dx, dy, dz| (dx == 0 && dy == 0 && dz == 0).then_some(chunk))
    }

    fn lit(chunk: &Chunk) -> LightGrid {
        let mut grid = LightGrid::dark();
        propagate(&solo(chunk), &PaddedLight::dark(), &CeilingWindow::open(), 0, &tables(), &mut grid);
        grid
    }

    #[test]
    fn open_column_lit_to_floor_and_sealed_layer_shadows_below() {
        // A full opaque layer at y=5 seals the lower half: with no gap for the
        // horizontal skylight flood to leak through, everything below is dark,
        // while the open cells above are lit to the layer.
        let mut cells = [0u8; CHUNK_VOLUME];
        for z in 0..16 {
            for x in 0..16 {
                cells[Chunk::index(x, 5, z)] = 1; // opaque floor across the chunk
            }
        }
        let chunk = Chunk::from_dense(0, 0, 0, Box::new(cells));
        let grid = lit(&chunk);

        assert_eq!(grid.at(Chunk::index(4, 15, 4)).sky, LightLevel::FULL, "top lit");
        assert_eq!(grid.at(Chunk::index(4, 6, 4)).sky, LightLevel::FULL, "just above the layer lit");
        assert_eq!(grid.at(Chunk::index(4, 4, 4)).sky, LightLevel::DARK, "sealed below the layer");
        assert_eq!(grid.at(Chunk::index(0, 0, 0)).sky, LightLevel::DARK, "floor sealed dark");
    }

    #[test]
    fn cave_in_a_deep_chunk_is_dark() {
        // A hollow chunk whose top is far below the terrain surface: the ceiling
        // reports the top as closed, so no skylight is seeded and the cavern is
        // dark — consistently, regardless of the 16-cell chunk alignment.
        let chunk = Chunk::from_dense(0, -8, 0, Box::new([0u8; CHUNK_VOLUME]));
        let ceiling = CeilingWindow::from_heights(|_, _| 40); // surface well above this chunk
        let mut grid = LightGrid::dark();
        propagate(&solo(&chunk), &PaddedLight::dark(), &ceiling, -128, &tables(), &mut grid);

        assert_eq!(grid.at(Chunk::index(4, 8, 4)).sky, LightLevel::DARK, "cavern dark");
        assert_eq!(grid.at(Chunk::index(0, 0, 0)).sky, LightLevel::DARK, "cavern floor dark");
    }

    #[test]
    fn blocklight_falls_off_by_one_per_step() {
        let mut chunk = Chunk::from_dense(0, 0, 0, Box::new([0u8; CHUNK_VOLUME]));
        chunk.set_local(8, 8, 8, BlockId(2)); // emitter, level 15
        let ceiling = CeilingWindow::from_heights(|_, _| 100); // fully underground: isolate blocklight
        let mut grid = LightGrid::dark();
        propagate(&solo(&chunk), &PaddedLight::dark(), &ceiling, 0, &tables(), &mut grid);

        assert_eq!(grid.at(Chunk::index(8, 8, 8)).block.get(), 15, "the emitter");
        assert_eq!(grid.at(Chunk::index(9, 8, 8)).block.get(), 14, "one step");
        assert_eq!(grid.at(Chunk::index(11, 8, 8)).block.get(), 12, "three steps");
    }

    /// Two adjacent chunks relaxed by hand — a torch in the left chunk floods
    /// across the shared border into the right. Iterating "capture the shell,
    /// propagate" reaches a fixpoint (no border moves) within a bounded number of
    /// passes: the convergence the decoupled settle pass relies on.
    #[test]
    fn settle_reaches_a_fixpoint_across_a_border() {
        let tables = tables();
        let mut left_c = Chunk::from_dense(0, 0, 0, Box::new([0u8; CHUNK_VOLUME]));
        left_c.set_local(14, 8, 8, BlockId(2)); // emitter near the +X border
        let right_c = Chunk::from_dense(1, 0, 0, Box::new([0u8; CHUNK_VOLUME]));
        let ceiling = CeilingWindow::from_heights(|_, _| 100); // underground: isolate blocklight

        // A shell holding one neighbour at chunk offset `nbr_dx` (dark elsewhere)
        // — the settle pass's `capture` in miniature.
        fn shell(nbr_dx: i32, nbr: &LightGrid) -> PaddedLight {
            PaddedLight::capture(|dx, _, _| (dx == nbr_dx).then_some(nbr))
        }

        let mut left = LightGrid::dark();
        let mut right = LightGrid::dark();
        let mut passes = 0;
        loop {
            passes += 1;
            let mut new_left = LightGrid::dark();
            propagate(&solo(&left_c), &shell(1, &right), &ceiling, 0, &tables, &mut new_left);
            let mut new_right = LightGrid::dark();
            propagate(&solo(&right_c), &shell(-1, &new_left), &ceiling, 0, &tables, &mut new_right);
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
