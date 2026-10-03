//! Chunks at curved-chart seams (SPACE-ARCHITECTURE §7).
//!
//! A round body's cells live in storage boxes (`space::atlas`), one box per patch, with storage
//! `+Y` along the chart's up. A storage chunk on the side of its box has its real neighbour across
//! that side in another patch's box, reached through the atlas glue: [`Seams::across`] names that
//! chunk and the signed map from this side's (virtual) neighbour cells onto its cells. The world's
//! neighbour reads (the mesher's halo, both light shells) read through it, and its neighbour
//! triggers (arrival, border light, edits, the mesh gate) use [`Seams::neighbour`], so a chart edge
//! is as invisible as a chunk border. Chunks outside the storage region never reach the atlas:
//! one compare and out.
//!
//! Exact across chart edges (cells conform face to face); approximate across band interfaces
//! (1 : 2) and at the eight valence-3 corners, where diagonal halo cells may read as air — the
//! declared exceptional regions of the stage-0 report.

use std::sync::{Arc, Mutex, PoisonError};

use super::chunk::{CHUNK_SIZE, Chunk};
use super::{Coord, FastMap};
use crate::coord::{BlockCoord, Face};
use crate::space::atlas::{Atlas, GLUE, Remap};

const CS: i64 = CHUNK_SIZE as i64;
/// Seam answers kept before the cache starts over (seam chunks are a thin set of the loaded ones).
const CACHE_CAP: usize = 1 << 15;

/// A neighbour across a seam: the real chunk and the map of virtual-neighbour cells onto it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Across {
    pub chunk: Coord,
    pub remap: Remap,
}

impl Across {
    /// The storage cell holding cell `l` of the virtual neighbour (local to it; a cell or two
    /// outside it lands in a chunk next to [`chunk`](Self::chunk)).
    #[inline]
    pub fn cell(&self, l: [i64; 3]) -> [i64; 3] {
        let r = self.remap.apply(l);
        [self.chunk.x as i64 * CS + r[0], self.chunk.y as i64 * CS + r[1], self.chunk.z as i64 * CS + r[2]]
    }
}

/// One storage box in chunk coordinates (`hi` exclusive) and the atlas it belongs to.
#[derive(Clone, Copy, Debug)]
struct Region {
    lo: [i32; 3],
    hi: [i32; 3],
    atlas: usize,
}

impl Region {
    #[inline]
    fn contains(&self, c: Coord) -> bool {
        let c = [c.x, c.y, c.z];
        (0..3).all(|a| c[a] >= self.lo[a] && c[a] < self.hi[a])
    }

    /// Whether `c` (inside) touches a side of the box.
    #[inline]
    fn on_side(&self, c: Coord) -> bool {
        let c = [c.x, c.y, c.z];
        (0..3).any(|a| c[a] == self.lo[a] || c[a] == self.hi[a] - 1)
    }
}

/// The seams of every atlas in a world.
pub struct Seams {
    atlases: Vec<Arc<Atlas>>,
    regions: Vec<Region>,
    /// Smallest storage chunk x of any box: everything below is physical space.
    min_cx: i32,
    cache: Mutex<FastMap<(Coord, Face), Option<Across>>>,
}

impl Seams {
    pub fn new(atlases: Vec<Arc<Atlas>>) -> Self {
        let mut regions = Vec::new();
        for (i, atlas) in atlases.iter().enumerate() {
            for p in atlas.patches() {
                let (o, size) = atlas.storage_box(p);
                debug_assert!((0..3).all(|a| o[a] % CS == 0 && size[a] % CS == 0), "storage boxes are chunk aligned");
                let lo = std::array::from_fn(|a| (o[a] / CS) as i32);
                let hi = std::array::from_fn(|a| ((o[a] + size[a]) / CS) as i32);
                regions.push(Region { lo, hi, atlas: i });
            }
        }
        let min_cx = regions.iter().map(|r| r.lo[0]).min().unwrap_or(i32::MAX);
        Self { atlases, regions, min_cx, cache: Mutex::new(FastMap::default()) }
    }

    /// No atlases: every answer is the plain grid.
    #[cfg(test)]
    pub fn none() -> Self {
        Self::new(Vec::new())
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// The atlases (the generator's round bodies).
    pub fn atlases(&self) -> &[Arc<Atlas>] {
        &self.atlases
    }

    #[inline]
    fn region_of(&self, c: Coord) -> Option<&Region> {
        if c.x < self.min_cx {
            return None;
        }
        self.regions.iter().find(|r| r.contains(c))
    }

    /// Whether `c` is a storage chunk inside some box.
    #[cfg(test)]
    pub fn in_storage(&self, c: Coord) -> bool {
        self.region_of(c).is_some()
    }

    /// Whether `c` is a storage-region chunk outside every box but within a chunk of one (its cells
    /// may read through the glue).
    #[inline]
    pub fn beside_storage(&self, c: Coord) -> bool {
        if c.x < self.min_cx - 1 {
            return false;
        }
        let near = |r: &Region| (0..3).all(|a| {
            let v = [c.x, c.y, c.z][a];
            v >= r.lo[a] - 1 && v <= r.hi[a]
        });
        self.region_of(c).is_none() && self.regions.iter().any(near)
    }

    /// The chunk across `face` of `c` when that side is a seam (the plain neighbour lies outside
    /// every box and the glue finds a patch there). `None` for physical chunks, interior storage
    /// chunks, the top of a band and the eight corners.
    pub fn across(&self, c: Coord, face: Face) -> Option<Across> {
        let r = self.region_of(c)?;
        if r.contains(c.step(face)) {
            return None;
        }
        let mut cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(hit) = cache.get(&(c, face)) {
            return *hit;
        }
        let got = self.atlases[r.atlas]
            .chunk_across([c.x as i64, c.y as i64, c.z as i64], face.axis(), face.sign() as i64)
            .map(|(k, remap)| Across { chunk: Coord::new(k[0] as i32, k[1] as i32, k[2] as i32), remap });
        if cache.len() >= CACHE_CAP {
            cache.clear();
        }
        cache.insert((c, face), got);
        got
    }

    /// The chunk that borders `c` across `face`: across a seam the glued chunk, else the plain step.
    #[inline]
    pub fn neighbour(&self, c: Coord, face: Face) -> Coord {
        if c.x < self.min_cx {
            return c.step(face);
        }
        self.across(c, face).map_or(c.step(face), |a| a.chunk)
    }

    /// Every halo cell of `c` (padded coordinates `-1..=16`, outside the chunk) whose plain chunk
    /// lies outside `c`'s box and across a seam, with the storage cell that holds it. Regions with
    /// no seam (above a band's top, the corners) are not visited and keep what the plain capture
    /// read.
    pub fn for_each_glued_halo(&self, c: Coord, mut visit: impl FnMut([i32; 3], [i64; 3])) {
        let Some(r) = self.region_of(c) else { return };
        if !r.on_side(c) {
            return;
        }
        let r = *r;
        let span = |d: i32| match d {
            -1 => -1..=-1,
            0 => 0..=CS as i32 - 1,
            _ => CS as i32..=CS as i32,
        };
        for dy in -1..=1 {
            for dz in -1..=1 {
                for dx in -1..=1 {
                    let d = [dx, dy, dz];
                    if d == [0, 0, 0] || r.contains(Coord::new(c.x + dx, c.y + dy, c.z + dz)) {
                        continue;
                    }
                    // The seam this region hangs across: the first axis whose single step leaves
                    // the box through a seam.
                    let seam = (0..3).filter(|&a| d[a] != 0).find_map(|a| {
                        let face = face_of(a, d[a]);
                        self.across(c, face).map(|x| (a, x))
                    });
                    let Some((a, x)) = seam else { continue };
                    for py in span(dy) {
                        for pz in span(dz) {
                            for px in span(dx) {
                                let p = [px, py, pz];
                                let mut l = p.map(i64::from);
                                l[a] -= d[a] as i64 * CS;
                                visit(p, x.cell(l));
                            }
                        }
                    }
                }
            }
        }
    }

    /// The storage cells of the near layer of the neighbour across `face` when that side is a
    /// seam, in [`FaceShell`](super::light::FaceShell) order (`i = a + b·16` over the face's two
    /// other axes in increasing order). Returns whether `face` is a seam.
    pub fn for_each_glued_face(&self, c: Coord, face: Face, mut visit: impl FnMut(usize, [i64; 3])) -> bool {
        let Some(x) = self.across(c, face) else { return false };
        let a = face.axis();
        let (u, v) = match a {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let near = if face.sign() > 0 { 0 } else { CS - 1 };
        for j in 0..CS {
            for i in 0..CS {
                let mut l = [0; 3];
                l[a] = near;
                l[u] = i;
                l[v] = j;
                visit((i + j * CS) as usize, x.cell(l));
            }
        }
        true
    }

    /// The physical cell holding storage cell `cell` (the cell of its centre's embedding), for
    /// anything that must act where the matter really is — the gravity ledger. `None` for physical
    /// cells and storage outside every box.
    pub fn physical_cell(&self, cell: BlockCoord) -> Option<(i32, i32, i32)> {
        let c = Coord::new(cell.x.div_euclid(CS as i32), cell.y.div_euclid(CS as i32), cell.z.div_euclid(CS as i32));
        self.region_of(c)?;
        let p = crate::space::atlas::embed_cell(&self.atlases, (cell.x, cell.y, cell.z))?;
        Some((p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32))
    }

    /// For a storage cell just outside every box (within the atlas glue reach of one), the cell of
    /// the neighbouring patch at the same physical point. `None` for physical cells, cells inside a
    /// box and cells beyond the glue (open space around a body's storage).
    pub fn glue_cell(&self, cell: BlockCoord) -> Option<BlockCoord> {
        let c = Coord::new(cell.x.div_euclid(CS as i32), cell.y.div_euclid(CS as i32), cell.z.div_euclid(CS as i32));
        if c.x < self.min_cx - 1 {
            return None;
        }
        if self.region_of(c).is_some() {
            return None;
        }
        let s = [cell.x as i64, cell.y as i64, cell.z as i64];
        let near = |r: &Region| {
            (0..3).all(|a| s[a] >= r.lo[a] as i64 * CS - GLUE && s[a] < r.hi[a] as i64 * CS + GLUE)
        };
        let r = self.regions.iter().find(|r| near(r))?;
        let g = self.atlases[r.atlas].glue(s)?;
        Some(BlockCoord::new(g[0] as i32, g[1] as i32, g[2] as i32))
    }
}

/// One side of a home chart whose neighbour lies across a seam: the neighbour's box (chunks, `hi`
/// exclusive) and the affine chunk map between virtual chunks beyond the side and its real chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SideMap {
    lo: [i32; 3],
    hi: [i32; 3],
    /// The virtual chunk just beyond the side ...
    virt: [i32; 3],
    /// ... is this real chunk of the neighbour.
    real: [i32; 3],
    /// The real step of a unit virtual step along each axis (signed unit vectors: a rotation).
    cols: [[i32; 3]; 3],
}

#[inline]
fn inside(k: [i32; 3], lo: [i32; 3], hi: [i32; 3]) -> bool {
    (0..3).all(|a| k[a] >= lo[a] && k[a] < hi[a])
}

/// How streaming sees the charts around a centre in storage: the centre's own box (home) and its
/// neighbours across seams unfolded beyond its four sides, so a view box around the centre covers
/// the neighbouring charts' chunks as if the charts were one flat net. The identity everywhere
/// else (a physical centre, or no atlas).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Unfold {
    home: Option<([i32; 3], [i32; 3])>,
    sides: [Option<SideMap>; 4],
    /// First storage chunk x: everything below is physical space, which no net touches.
    storage_cx0: i32,
}

impl Unfold {
    pub const IDENTITY: Self = Self { home: None, sides: [None; 4], storage_cx0: i32::MAX };

    /// Whether this is the identity (a physical centre).
    #[inline]
    pub fn is_identity(&self) -> bool {
        self.home.is_none()
    }

    /// Where real chunk `c` sits in the net: home chunks are themselves, a neighbour chart's chunks
    /// the virtual chunks beyond the side they unfold from, physical chunks themselves, and storage
    /// outside the net (other bands, faces and bodies) far away from everything. Only ever fold real
    /// chunks (a virtual chunk may coincide with some other box's address).
    #[inline]
    pub fn fold(&self, c: Coord) -> Coord {
        let Some((lo, hi)) = self.home else { return c };
        let k = [c.x, c.y, c.z];
        if inside(k, lo, hi) {
            return c;
        }
        for s in self.sides.iter().flatten() {
            if inside(k, s.lo, s.hi) {
                // The inverse of a rotation is its transpose.
                let d = [k[0] - s.real[0], k[1] - s.real[1], k[2] - s.real[2]];
                let v: [i32; 3] =
                    std::array::from_fn(|a| s.virt[a] + (0..3).map(|b| s.cols[a][b] * d[b]).sum::<i32>());
                return Coord::new(v[0], v[1], v[2]);
            }
        }
        if c.x >= self.storage_cx0 {
            return Coord::new(i32::MIN / 4, i32::MIN / 4, i32::MIN / 4);
        }
        c
    }

    /// The real chunk standing for virtual chunk `v` of a view box: `v` in the home chart (and
    /// everywhere for the identity), the neighbour's chunk beyond a seam side, `None` for storage
    /// that holds nothing (beyond the relief top or the bottom of the band, past the corners).
    #[inline]
    pub fn unfold(&self, v: Coord) -> Option<Coord> {
        let Some((lo, hi)) = self.home else { return Some(v) };
        let k = [v.x, v.y, v.z];
        if inside(k, lo, hi) || v.x < self.storage_cx0 {
            return Some(v);
        }
        let out = |a: usize| (k[a] >= hi[a]) as i32 - (k[a] < lo[a]) as i32;
        let (ox, oy, oz) = (out(0), out(1), out(2));
        if oy != 0 || (ox != 0) == (oz != 0) {
            return None;
        }
        let side = match (ox, oz) {
            (-1, _) => 0,
            (1, _) => 1,
            (_, -1) => 2,
            _ => 3,
        };
        let s = self.sides[side]?;
        let d = [k[0] - s.virt[0], k[1] - s.virt[1], k[2] - s.virt[2]];
        let r: [i32; 3] = std::array::from_fn(|b| s.real[b] + (0..3).map(|a| s.cols[a][b] * d[a]).sum::<i32>());
        inside(r, s.lo, s.hi).then(|| Coord::new(r[0], r[1], r[2]))
    }
}

impl Seams {
    /// The unfolding around streaming centre `centre` (a storage chunk inside a box, or above one).
    pub fn unfold_at(&self, centre: Coord) -> Unfold {
        if centre.x < self.min_cx {
            return Unfold::IDENTITY;
        }
        let column = |r: &&Region| centre.x >= r.lo[0] && centre.x < r.hi[0] && centre.z >= r.lo[2] && centre.z < r.hi[2];
        let Some(r) = self.region_of(centre).or_else(|| self.regions.iter().find(column)) else {
            return Unfold::IDENTITY;
        };
        let mut u = Unfold { home: Some((r.lo, r.hi)), sides: [None; 4], storage_cx0: self.min_cx };
        for (i, face) in [Face::NegX, Face::PosX, Face::NegZ, Face::PosZ].into_iter().enumerate() {
            let a = face.axis();
            let mut p = [centre.x, centre.y, centre.z];
            for k in 0..3 {
                p[k] = p[k].clamp(r.lo[k], r.hi[k] - 1);
            }
            p[a] = if face.sign() > 0 { r.hi[a] - 1 } else { r.lo[a] };
            let Some(x) = self.across(Coord::new(p[0], p[1], p[2]), face) else { continue };
            let cols = x.remap.cols.map(|c| c.map(|v| v as i32));
            let unit = cols.iter().all(|c| c.iter().map(|v| v.abs()).sum::<i32>() == 1 && c.iter().any(|v| v.abs() == 1));
            let Some(nr) = self.region_of(x.chunk) else { continue };
            if !unit {
                continue;
            }
            let mut virt = p;
            virt[a] += face.sign();
            u.sides[i] = Some(SideMap { lo: nr.lo, hi: nr.hi, virt, real: [x.chunk.x, x.chunk.y, x.chunk.z], cols });
        }
        u
    }

    /// Where streaming should stand for a physical eye at `p`: its storage position in the patch
    /// under it, also when it flies above a chart's relief top (up to `reach` blocks, projected
    /// straight down onto the chart). `None` away from every round body.
    pub fn storage_eye(&self, p: glam::DVec3, reach: f64) -> Option<glam::DVec3> {
        for a in &self.atlases {
            if let Some((patch, l)) = a.find(p) {
                let (o, _) = a.storage_box(patch);
                return Some(l + glam::DVec3::new(o[0] as f64, o[1] as f64, o[2] as f64));
            }
            let rel = p - a.centre;
            let r = rel.length();
            let b = a.bands[0];
            // How far above the top (outward charts: beyond r_hi; inward ones: inside r_lo).
            let (top, above) = if a.inward { (b.r_lo as f64 + 0.5, b.r_lo as f64 + 0.5 - r) } else { (b.r_hi as f64 - 0.5, r - (b.r_hi as f64 - 0.5)) };
            if !(above > 0.0 && above < reach) || r == 0.0 {
                continue;
            }
            if let Some((patch, l)) = a.find(a.centre + rel * (top / r)) {
                let (o, _) = a.storage_box(patch);
                return Some(glam::DVec3::new(l.x + o[0] as f64, l.y + o[1] as f64 + above, l.z + o[2] as f64));
            }
        }
        None
    }
}

fn face_of(axis: usize, d: i32) -> Face {
    match (axis, d > 0) {
        (0, true) => Face::PosX,
        (0, false) => Face::NegX,
        (1, true) => Face::PosY,
        (1, false) => Face::NegY,
        (_, true) => Face::PosZ,
        (_, false) => Face::NegZ,
    }
}

/// Chunk and local flat index of a storage cell.
#[inline]
pub fn split(s: [i64; 3]) -> (Coord, usize) {
    let c = Coord::new(s[0].div_euclid(CS) as i32, s[1].div_euclid(CS) as i32, s[2].div_euclid(CS) as i32);
    let l = s.map(|v| v.rem_euclid(CS) as usize);
    (c, Chunk::index(l[0], l[1], l[2]))
}

/// A one-entry memo over chunk lookups: halo patches read runs of cells from the same chunk.
pub struct LastChunk<S> {
    key: Option<(Coord, Option<S>)>,
}

impl<S: Copy> LastChunk<S> {
    pub fn new() -> Self {
        Self { key: None }
    }

    #[inline]
    pub fn get(&mut self, c: Coord, lookup: impl FnOnce(Coord) -> Option<S>) -> Option<S> {
        match self.key {
            Some((k, v)) if k == c => v,
            _ => {
                let v = lookup(c);
                self.key = Some((c, v));
                v
            }
        }
    }
}

impl super::World {
    /// Read a captured voxel halo's seam regions through the glue (missing glued chunks read as air,
    /// like any missing neighbour).
    pub(in crate::world) fn seam_halo(&self, coord: Coord, padded: &mut super::mesh::Padded) {
        if self.seams.is_empty() {
            return;
        }
        let mut last = LastChunk::new();
        self.seams.for_each_glued_halo(coord, |p, s| {
            let (c, i) = split(s);
            let chunk = last.get(c, |c| self.chunks.get(&c).map(|l| &*l.chunk));
            padded.set(p, chunk.map_or(crate::block::registry::AIR, |ch| ch.get_index(i)));
        });
    }

    /// Read a captured light halo's seam regions through the glue (a missing grid reads as
    /// `fallback`, else dark — as the plain capture does).
    pub(in crate::world) fn seam_light_halo(
        &self,
        coord: Coord,
        light: &mut super::light::PaddedLight,
        fallback: Option<&super::light::LightGrid>,
    ) {
        if self.seams.is_empty() {
            return;
        }
        let mut last = LastChunk::new();
        self.seams.for_each_glued_halo(coord, |p, s| {
            let (c, i) = split(s);
            let grid = last.get(c, |c| self.chunks.get(&c).and_then(|l| l.light.as_ref())).or(fallback);
            light.set(p, grid.map_or(super::light::PackedLumel::DARK, |g| g.packed_at(i)));
        });
    }

    /// Read a face shell's seam layers through the glue (a missing grid reads dark).
    pub(in crate::world) fn seam_face_shell(&self, coord: Coord, shell: &mut super::light::FaceShell) {
        if self.seams.is_empty() {
            return;
        }
        for face in Face::ALL {
            let mut last = LastChunk::new();
            self.seams.for_each_glued_face(coord, face, |i, s| {
                let (c, idx) = split(s);
                let grid = last.get(c, |c| self.chunks.get(&c).and_then(|l| l.light.as_ref()));
                shell.set(face, i, grid.map_or(super::light::PackedLumel::DARK, |g| g.packed_at(idx)));
            });
        }
    }

    /// The round body whose atlas covers physical point `p` (its bands, transition or core): there
    /// motion and picking run in that patch's storage frame.
    pub fn atlas_at(&self, p: glam::DVec3) -> Option<&Arc<Atlas>> {
        self.seams.atlases().iter().find(|a| a.find(p).is_some())
    }

    /// The generator's round bodies.
    pub fn atlases(&self) -> &[Arc<Atlas>] {
        self.seams.atlases()
    }

    /// The cell that really holds storage cell `c`: the neighbouring chart's cell for a cell just
    /// outside a box, else `c` itself.
    pub fn glued(&self, c: (i32, i32, i32)) -> (i32, i32, i32) {
        self.seams.glue_cell(BlockCoord::new(c.0, c.1, c.2)).map_or(c, |g| (g.x, g.y, g.z))
    }

    /// Test worlds: hand round bodies to a world whose generator has none (cells placed by edits).
    #[cfg(test)]
    pub fn set_atlases(&mut self, atlases: Vec<Arc<Atlas>>) {
        self.seams = Seams::new(atlases);
    }

    /// The chunk bordering `coord` across `face` (through a seam when there is one).
    #[inline]
    pub(in crate::world) fn neighbour(&self, coord: Coord, face: Face) -> Coord {
        self.seams.neighbour(coord, face)
    }
}

#[cfg(test)]
mod tests {
    use glam::DVec3;

    use super::*;
    use crate::space::atlas::Patch;

    fn atlas() -> Arc<Atlas> {
        Arc::new(Atlas::new(DVec3::new(3.0e8, -2.0e8, 1.0e8), 4096, 4096 + 128, false, 0))
    }

    fn chunk_of(s: [i64; 3]) -> Coord {
        split(s).0
    }

    /// A storage chunk on the +u side of the +Y chart's outer band, at the surface.
    fn seam_chunk(a: &Atlas) -> Coord {
        let p = Patch::Shell { band: 0, face: Face::PosY };
        let (o, size) = a.storage_box(p);
        chunk_of([o[0] + size[0] - 1, o[1] + size[1] - 64, o[2] + size[2] / 2])
    }

    #[test]
    fn physical_and_interior_chunks_have_no_seams() {
        let seams = Seams::new(vec![atlas()]);
        let a = atlas();
        for f in Face::ALL {
            assert_eq!(seams.across(Coord::new(0, 0, 0), f), None);
            assert_eq!(seams.neighbour(Coord::new(5, -3, 2), f), Coord::new(5, -3, 2).step(f));
        }
        let p = Patch::Shell { band: 0, face: Face::PosY };
        let (o, size) = a.storage_box(p);
        let mid = chunk_of([o[0] + size[0] / 2, o[1] + size[1] / 2, o[2] + size[2] / 2]);
        for f in Face::ALL {
            assert_eq!(seams.across(mid, f), None, "{f:?}");
        }
        assert!(Seams::none().is_empty());
    }

    #[test]
    fn a_seam_is_symmetric() {
        let a = atlas();
        let seams = Seams::new(vec![a.clone()]);
        let c = seam_chunk(&a);
        let x = seams.across(c, Face::PosX).expect("the +u side of a chart is a seam");
        assert!(seams.in_storage(x.chunk));
        assert_ne!(seams.region_of(x.chunk).map(|r| r.lo), seams.region_of(c).map(|r| r.lo), "another patch");
        let back = Face::ALL.iter().filter_map(|&f| seams.across(x.chunk, f)).any(|y| y.chunk == c);
        assert!(back, "the glued chunk sees this one across one of its sides");
        assert_eq!(seams.neighbour(c, Face::PosX), x.chunk);
        // The top of the band is not a seam (open space above the relief).
        let top = {
            let (o, size) = a.storage_box(Patch::Shell { band: 0, face: Face::PosY });
            chunk_of([o[0] + size[0] - 1, o[1] + size[1] - 1, o[2] + size[2] / 2])
        };
        assert_eq!(seams.across(top, Face::PosY), None);
    }

    /// The remapped halo equals the atlas's per-cell glue (the slow, exact reference) on every face
    /// region and the edge regions along the seam.
    #[test]
    fn glued_halo_cells_are_the_glue_of_each_cell() {
        let a = atlas();
        let seams = Seams::new(vec![a.clone()]);
        let c = seam_chunk(&a);
        let mut visited = 0;
        let mut mismatched = Vec::new();
        seams.for_each_glued_halo(c, |p, s| {
            visited += 1;
            let raw = [c.x as i64 * CS + p[0] as i64, c.y as i64 * CS + p[1] as i64, c.z as i64 * CS + p[2] as i64];
            if let Some(g) = a.glue(raw) {
                if g != s {
                    mismatched.push((p, s, g));
                }
            }
        });
        assert!(visited >= 16 * 16, "the whole +x face region is glued ({visited})");
        assert!(mismatched.is_empty(), "{} of {visited} differ, e.g. {:?}", mismatched.len(), &mismatched[..mismatched.len().min(4)]);
    }

    #[test]
    fn glued_face_layer_is_the_glue_of_each_cell() {
        let a = atlas();
        let seams = Seams::new(vec![a.clone()]);
        let c = seam_chunk(&a);
        let mut n = 0;
        assert!(seams.for_each_glued_face(c, Face::PosX, |i, s| {
            n += 1;
            let (y, z) = ((i % 16) as i64, (i / 16) as i64);
            let raw = [c.x as i64 * CS + 16, c.y as i64 * CS + y, c.z as i64 * CS + z];
            assert_eq!(a.glue(raw), Some(s), "cell {i}");
        }));
        assert_eq!(n, 256);
        assert!(!seams.for_each_glued_face(c, Face::NegX, |_, _| {}), "the inner side is plain");
    }

    #[test]
    fn glue_cell_reads_through_the_seam_and_leaves_box_cells_alone() {
        let a = atlas();
        let seams = Seams::new(vec![a.clone()]);
        let c = seam_chunk(&a);
        let inside = BlockCoord::new(c.x * 16 + 15, c.y * 16 + 3, c.z * 16 + 3);
        assert_eq!(seams.glue_cell(inside), None);
        let outside = BlockCoord::new(c.x * 16 + 16, c.y * 16 + 3, c.z * 16 + 3);
        let g = seams.glue_cell(outside).expect("one cell outside a chart's side");
        assert_eq!(Some([g.x as i64, g.y as i64, g.z as i64]), a.glue([outside.x as i64, outside.y as i64, outside.z as i64]));
        assert_eq!(seams.glue_cell(BlockCoord::new(10, 20, 30)), None, "physical cells");
    }

    #[test]
    fn the_net_unfolds_and_folds_back() {
        let a = atlas();
        let seams = Seams::new(vec![a.clone()]);
        let c = seam_chunk(&a);
        let u = seams.unfold_at(c);
        assert!(!u.is_identity());
        assert!(Seams::new(vec![a.clone()]).unfold_at(Coord::new(1, 2, 3)).is_identity(), "physical centres");
        // Every chunk along the +u side: the virtual chunk beyond it is the glued neighbour.
        let r = *seams.region_of(c).unwrap();
        for z in (r.lo[2]..r.hi[2]).step_by(7) {
            let side = Coord::new(r.hi[0] - 1, c.y, z);
            let x = seams.across(side, Face::PosX).expect("a seam all along the side");
            let v = side.step(Face::PosX);
            assert_eq!(u.unfold(v), Some(x.chunk), "z {z}");
            assert_eq!(u.fold(x.chunk), v);
        }
        // Further out the net keeps going into the neighbour, and folds back exactly.
        for k in 1..6 {
            let v = Coord::new(r.hi[0] - 1 + k, c.y + 1, c.z - 3);
            let real = u.unfold(v).expect("inside the neighbour chart");
            assert!(seams.in_storage(real));
            assert_eq!(u.fold(real), v);
        }
        // Home chunks are themselves; beyond the top and past a corner hold nothing.
        assert_eq!(u.unfold(c), Some(c));
        assert_eq!(u.unfold(Coord::new(c.x, r.hi[1] + 2, c.z)), None);
        assert_eq!(u.unfold(Coord::new(r.hi[0] + 1, c.y, r.hi[2] + 1)), None);
    }

    #[test]
    fn the_storage_eye_sits_on_the_cell_holding_it_and_above_the_top() {
        let a = atlas();
        let seams = Seams::new(vec![a.clone()]);
        let top = Patch::Shell { band: 0, face: Face::PosY };
        let b = a.bands[0];
        let l = DVec3::new(b.n as f64 * 0.3, (a.radius - b.r_lo) as f64 + 1.5, b.n as f64 * 0.6);
        let p = a.embed(top, l);
        let (o, _) = a.storage_box(top);
        let s = seams.storage_eye(p, 1000.0).expect("on the surface");
        assert!((s - (l + DVec3::new(o[0] as f64, o[1] as f64, o[2] as f64))).length() < 1e-6);
        // Fly 300 blocks above the relief top: straight above the same column, 300 cells above.
        let dir = (p - a.centre).normalize();
        let high = a.centre + dir * (b.r_hi as f64 - 0.5 + 300.0);
        let sh = seams.storage_eye(high, 1000.0).expect("above the top within reach");
        assert!((sh.x - s.x).abs() < 0.01 && (sh.z - s.z).abs() < 0.01, "{sh:?} vs {s:?}");
        assert!((sh.y - (o[1] as f64 + (b.r_hi - b.r_lo) as f64 - 0.5 + 300.0)).abs() < 0.01);
        assert_eq!(seams.storage_eye(a.centre + dir * (b.r_hi as f64 + 5000.0), 1000.0), None, "beyond reach");
    }

    #[test]
    fn seam_answers_are_cheap_once_cached() {
        let a = atlas();
        let seams = Seams::new(vec![a.clone()]);
        let c = seam_chunk(&a);
        let t = std::time::Instant::now();
        let first = seams.across(c, Face::PosX);
        let cold = t.elapsed();
        let t = std::time::Instant::now();
        for _ in 0..1000 {
            assert_eq!(seams.across(c, Face::PosX), first);
        }
        let warm = t.elapsed() / 1000;
        assert!(warm < cold.max(std::time::Duration::from_micros(5)), "cold {cold:?}, warm {warm:?}");
    }
}
