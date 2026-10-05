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
use crate::space::atlas::{Atlas, GLUE, Patch, Remap};

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

    /// The eight physical corners of storage chunk `c` (a chunk inside a box): the cage its mesh
    /// is drawn through. `None` for physical chunks.
    pub fn cage(&self, c: Coord) -> Option<[glam::DVec3; 8]> {
        let r = self.region_of(c)?;
        self.atlases[r.atlas].chunk_cage([c.x as i64 * CS, c.y as i64 * CS, c.z as i64 * CS])
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

impl SideMap {
    /// The virtual chunk beyond the side standing for real chunk `k` of the neighbour. The inverse
    /// of a rotation is its transpose.
    #[inline]
    fn to_virtual(&self, k: [i32; 3]) -> [i32; 3] {
        let d = [k[0] - self.real[0], k[1] - self.real[1], k[2] - self.real[2]];
        std::array::from_fn(|a| self.virt[a] + (0..3).map(|b| self.cols[a][b] * d[b]).sum::<i32>())
    }
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
                let v = s.to_virtual(k);
                return Coord::new(v[0], v[1], v[2]);
            }
        }
        if c.x >= self.storage_cx0 {
            return Coord::new(i32::MIN / 4, i32::MIN / 4, i32::MIN / 4);
        }
        c
    }

    /// [`fold`](Self::fold) for storage block column `(x, z)`, which the far field's sections are:
    /// home columns are themselves, a neighbour chart's the centre of the virtual chunk column
    /// beyond its side, physical columns themselves, and `None` for storage outside the net.
    pub fn fold_column(&self, x: i64, z: i64) -> Option<(i64, i64)> {
        let Some((lo, hi)) = self.home else { return Some((x, z)) };
        let (cx, cz) = (x.div_euclid(CS) as i32, z.div_euclid(CS) as i32);
        let column = |lo: [i32; 3], hi: [i32; 3]| cx >= lo[0] && cx < hi[0] && cz >= lo[2] && cz < hi[2];
        if column(lo, hi) {
            return Some((x, z));
        }
        if let Some(s) = self.sides.iter().flatten().find(|s| column(s.lo, s.hi)) {
            let v = s.to_virtual([cx, s.lo[1], cz]);
            return Some((i64::from(v[0]) * CS + CS / 2, i64::from(v[2]) * CS + CS / 2));
        }
        (cx < self.storage_cx0).then_some((x, z))
    }

    /// The real chunk standing for virtual chunk `v` of a view box: `v` in the home chart (and
    /// everywhere for the identity), the neighbour's chunk beyond a seam side, `None` for storage
    /// that holds nothing (beyond the relief top or the bottom of the band, past the corners).
    #[inline]
    pub fn unfold(&self, v: Coord) -> Option<Coord> {
        let Some((lo, hi)) = self.home else { return Some(v) };
        let k = [v.x, v.y, v.z];
        if inside(k, lo, hi) {
            return Some(v);
        }
        // Chunk x below every storage box is physical space, except the virtual chunks just past
        // a chart's −X side: the first box shares the storage origin, so that neighbour sits
        // below `storage_cx0`.
        let beside_neg_x = k[0] < lo[0] && k[1] >= lo[1] && k[1] < hi[1] && k[2] >= lo[2] && k[2] < hi[2];
        if v.x < self.storage_cx0 && !beside_neg_x {
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
        let Some(r) = self.column_of(centre) else {
            return Unfold::IDENTITY;
        };
        let mut u = Unfold { home: Some((r.lo, r.hi)), sides: [None; 4], storage_cx0: self.min_cx };
        for (i, face) in [Face::NegX, Face::PosX, Face::NegZ, Face::PosZ].into_iter().enumerate() {
            let a = face.axis();
            // Anchor each side map at the middle of the region's side, not at the centre's
            // projection: a conforming seam's map is one affine step along its whole length, and a
            // fixed anchor keeps the unfold (and every worklist bucketed by it) unchanged while the
            // centre moves within the region.
            let mut p: [i32; 3] = std::array::from_fn(|k| r.lo[k] + (r.hi[k] - r.lo[k]) / 2);
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
    /// straight down onto the chart). `None` away from every chart and every warped cube.
    pub fn storage_eye(&self, p: glam::DVec3, reach: f64) -> Option<glam::DVec3> {
        for a in &self.atlases {
            if let Some((patch, l)) = a.find(p) {
                let (o, _) = a.storage_box(patch);
                return Some(l + glam::DVec3::new(o[0] as f64, o[1] as f64, o[2] as f64));
            }
            let Some((above, top)) = over_top(a, p) else { continue };
            if above >= reach {
                continue;
            }
            if let Some(s) = lifted(a, top, above) {
                return Some(s);
            }
        }
        None
    }

    /// Where the far field should stand for a physical eye at `p` above a round world: the chart
    /// column straight under it, lifted by its height above the stored top, on the nearest top
    /// within `reach` blocks. The `held` atlas counts `hold` times nearer than it is: it is kept
    /// past the reach, and until another top is that much closer. `None` when no top is close
    /// enough.
    pub fn far_eye(&self, p: glam::DVec3, reach: f64, held: Option<usize>, hold: f64) -> Option<glam::DVec3> {
        let mut best: Option<(f64, f64, usize, glam::DVec3)> = None;
        for (i, a) in self.atlases.iter().enumerate() {
            let Some((above, top)) = over_top(a, p) else { continue };
            let rank = if held == Some(i) { above / hold } else { above };
            if rank < reach && best.is_none_or(|(b, ..)| rank < b) {
                best = Some((rank, above, i, top));
            }
        }
        let (_, above, i, top) = best?;
        lifted(&self.atlases[i], top, above)
    }

    /// The box `centre` is in, else the box whose column it stands in (above its top or below its
    /// bottom). `None` in physical space.
    fn column_of(&self, centre: Coord) -> Option<&Region> {
        if centre.x < self.min_cx {
            return None;
        }
        let column = |r: &&Region| centre.x >= r.lo[0] && centre.x < r.hi[0] && centre.z >= r.lo[2] && centre.z < r.hi[2];
        self.region_of(centre).or_else(|| self.regions.iter().find(column))
    }

    /// Whether `centre` is a storage chunk in a box or in the column of one.
    pub(in crate::world) fn in_column(&self, centre: Coord) -> bool {
        self.column_of(centre).is_some()
    }

    /// The band-0 shell under a storage streaming centre (including flight just above its box).
    /// Deeper bands and the core select nothing: the player is inside the body.
    pub(in crate::world) fn chart_seat(&self, centre: Coord) -> Option<ChartSeat> {
        self.seat_of(self.column_of(centre)?)
    }

    fn seat_of(&self, r: &Region) -> Option<ChartSeat> {
        let atlas = &self.atlases[r.atlas];
        let cell = [r.lo[0] as i64 * CS + CS / 2, r.lo[1] as i64 * CS, r.lo[2] as i64 * CS + CS / 2];
        let (patch, _) = atlas.locate(cell)?;
        if !matches!(patch, Patch::Shell { band: 0, .. }) {
            return None;
        }
        let (o, size) = atlas.storage_box(patch);
        Some(ChartSeat {
            index: r.atlas,
            patch,
            lo: o,
            hi: [o[0] + size[0], o[1] + size[1], o[2] + size[2]],
            radius: atlas.radius,
        })
    }

    /// Neighbour band-0 charts when `eye` (storage blocks) is within `band` blocks of a home side.
    /// Only unit seams. A side whose cells are not a rotation of the home edge is skipped.
    pub(in crate::world) fn seam_across(&self, home: ChartSeat, eye: [i64; 3], band: i64) -> Vec<SeamAcross> {
        let centre = Coord::new(eye[0].div_euclid(CS) as i32, eye[1].div_euclid(CS) as i32, eye[2].div_euclid(CS) as i32);
        let unfold = self.unfold_at(centre);
        let atlas = &self.atlases[home.index];
        let mut out = Vec::new();
        // Same side order as [`Unfold`]: NegX, PosX, NegZ, PosZ.
        let sides: [(usize, [i64; 3]); 4] = [(0, [-1, 0, 0]), (0, [1, 0, 0]), (2, [0, 0, -1]), (2, [0, 0, 1])];
        for (i, (axis, side)) in sides.into_iter().enumerate() {
            if unfold.sides[i].is_none() {
                continue;
            }
            let steps = if side[axis] > 0 { home.hi[axis] - eye[axis] } else { eye[axis] - (home.lo[axis] - 1) };
            if steps <= 0 || steps > band {
                continue;
            }
            let tan = if axis == 0 { 2 } else { 0 };
            let mut probe = eye;
            probe[1] = probe[1].clamp(home.lo[1], home.hi[1] - 1);
            probe[tan] = probe[tan].clamp(home.lo[tan] + 1, home.hi[tan] - 2);
            probe[axis] = if side[axis] > 0 { home.hi[axis] } else { home.lo[axis] - 1 };
            let Some(g) = atlas.glue(probe) else { continue };
            if (g[1] - probe[1]).abs() > 4 {
                continue;
            }
            let Some((npatch, _)) = atlas.locate(g) else { continue };
            if npatch == home.patch || !matches!(npatch, Patch::Shell { band: 0, .. }) {
                continue;
            }
            let (o, size) = atlas.storage_box(npatch);
            let nlo = o;
            let nhi = [o[0] + size[0], o[1] + size[1], o[2] + size[2]];
            let Some(inward) = edge_inward(g, nlo, nhi) else { continue };
            let mut g2 = None;
            let mut home_step = 0i64;
            for sign in [16i64, -16] {
                let t = probe[tan] + sign;
                if t < home.lo[tan] || t >= home.hi[tan] {
                    continue;
                }
                let mut p2 = probe;
                p2[tan] = t;
                if let Some(gg) = atlas.glue(p2) {
                    home_step = sign;
                    g2 = Some(gg);
                    break;
                }
            }
            let Some(g2) = g2 else { continue };
            let delta = [g2[0] - g[0], g2[1] - g[1], g2[2] - g[2]];
            if delta[1].abs() > 2 {
                continue;
            }
            let Some(tan_n) = unit_step(delta, home_step) else { continue };
            let mut tan_h = [0i64; 3];
            tan_h[tan] = 1;
            out.push(SeamAcross {
                seat: ChartSeat { index: home.index, patch: npatch, lo: nlo, hi: nhi, radius: home.radius },
                g,
                inward,
                tan_n,
                probe,
                side,
                tan_h,
            });
        }
        out
    }
}

/// How far `p` is above `a`'s stored top (outward charts: beyond `r_hi`; inward ones: inside
/// `r_lo`), on the unlifted grid, and the top's physical point along `p`'s ray, which carries its
/// column's lift. `None` at or below the top.
fn over_top(a: &Atlas, p: glam::DVec3) -> Option<(f64, glam::DVec3)> {
    let b = *a.bands.first()?;
    let rel = p - a.centre;
    let r = rel.length();
    let ru = a.unlifted_radius(p);
    let (top, above) = if a.inward { (b.r_lo as f64 + 0.5, b.r_lo as f64 + 0.5 - ru) } else { (b.r_hi as f64 - 0.5, ru - (b.r_hi as f64 - 0.5)) };
    (above > 0.0 && r != 0.0).then(|| (above, a.centre + rel * ((top + (r - ru)) / r)))
}

/// The storage point `above` blocks over the top at physical point `top`.
fn lifted(a: &Atlas, top: glam::DVec3, above: f64) -> Option<glam::DVec3> {
    let (patch, l) = a.find(top)?;
    let (o, _) = a.storage_box(patch);
    Some(glam::DVec3::new(l.x + o[0] as f64, l.y + o[1] as f64 + above, l.z + o[2] as f64))
}

/// A band-0 shell chart: atlas index, patch, storage box (`hi` exclusive) and datum radius.
#[derive(Clone, Copy, Debug)]
pub(in crate::world) struct ChartSeat {
    pub index: usize,
    pub patch: Patch,
    pub lo: [i64; 3],
    pub hi: [i64; 3],
    pub radius: i64,
}

/// A neighbour chart joined across a unit seam, and the map back into the home storage frame.
#[derive(Clone, Copy, Debug)]
pub(in crate::world) struct SeamAcross {
    pub seat: ChartSeat,
    g: [i64; 3],
    inward: [i64; 3],
    tan_n: [i64; 3],
    probe: [i64; 3],
    side: [i64; 3],
    tan_h: [i64; 3],
}

impl SeamAcross {
    /// Home storage `(x, z)` of a neighbour storage `(x, z)`. The neighbour's interior continues
    /// past the home edge (the full-res box only overlaps the first blocks beyond the seam).
    /// The seam is a signed permutation.
    pub(in crate::world) fn home_xz(self, x: i64, z: i64) -> (i64, i64) {
        let d_in = (x - self.g[0]) * self.inward[0] + (z - self.g[2]) * self.inward[2];
        let d_tan = (x - self.g[0]) * self.tan_n[0] + (z - self.g[2]) * self.tan_n[2];
        let hx = self.probe[0] + self.side[0] * d_in + self.tan_h[0] * d_tan;
        let hz = self.probe[2] + self.side[2] * d_in + self.tan_h[2] * d_tan;
        (hx, hz)
    }

    /// Neighbour storage `(x, z)` of a home storage `(hx, hz)`. Inverse of [`home_xz`](Self::home_xz).
    pub(in crate::world) fn storage_xz(self, hx: i64, hz: i64) -> (i64, i64) {
        let d_in = (hx - self.probe[0]) * self.side[0] + (hz - self.probe[2]) * self.side[2];
        let d_tan = (hx - self.probe[0]) * self.tan_h[0] + (hz - self.probe[2]) * self.tan_h[2];
        let x = self.g[0] + self.inward[0] * d_in + self.tan_n[0] * d_tan;
        let z = self.g[2] + self.inward[2] * d_in + self.tan_n[2] * d_tan;
        (x, z)
    }
}

fn edge_inward(g: [i64; 3], lo: [i64; 3], hi: [i64; 3]) -> Option<[i64; 3]> {
    let mut found = None;
    for a in [0usize, 2] {
        let on_lo = g[a] - lo[a] <= 4;
        let on_hi = hi[a] - 1 - g[a] <= 4;
        if on_lo == on_hi {
            continue;
        }
        if found.is_some() {
            return None;
        }
        let mut u = [0i64; 3];
        u[a] = if on_lo { 1 } else { -1 };
        found = Some(u);
    }
    found
}

/// Neighbour units per one positive step of the home tangent. `home_step` is the probe offset.
fn unit_step(delta: [i64; 3], home_step: i64) -> Option<[i64; 3]> {
    let s = home_step.abs();
    if s == 0 {
        return None;
    }
    let mut axis = None;
    for a in [0usize, 2] {
        if delta[a].abs() <= 2 {
            continue;
        }
        if (delta[a].abs() - s).abs() > 2 {
            return None;
        }
        if axis.is_some() {
            return None;
        }
        let mut u = [0i64; 3];
        u[a] = delta[a].signum() * home_step.signum();
        axis = Some(u);
    }
    axis
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

    /// The round body or warped cube whose atlas covers physical point `p`: there motion and
    /// picking run in that patch's storage frame.
    pub fn atlas_at(&self, p: glam::DVec3) -> Option<&Arc<Atlas>> {
        self.seams.atlases().iter().find(|a| a.find(p).is_some())
    }

    /// `p`'s place in the chart under it: storage position (storage +Y is up) and the local Jacobian.
    pub(crate) fn chart_local(&self, p: glam::DVec3) -> Option<crate::space::atlas::Local> {
        self.seams.atlases().iter().find_map(|a| a.local(p))
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
        Arc::new(Atlas::new(DVec3::new(3.0e8, -2.0e8, 1.0e8), 4096, 4096 + 128, false, crate::space::atlas::STORAGE_X0))
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
        // The net is a property of the region, not of where in it the centre stands: moving the
        // centre keeps the unfold (worklists bucketed by it are not re-bucketed).
        for (dx, dy, dz) in [(-3, 0, 5), (-40, 2, -11), (-200, -1, 90)] {
            let elsewhere = Coord::new(c.x + dx, c.y + dy, c.z + dz);
            if seams.region_of(elsewhere).is_some_and(|q| q.lo == r.lo && q.hi == r.hi) {
                assert_eq!(seams.unfold_at(elsewhere), u, "centre moved by ({dx},{dy},{dz})");
            }
        }
        // Home chunks are themselves; beyond the top and past a corner hold nothing.
        assert_eq!(u.unfold(c), Some(c));
        assert_eq!(u.unfold(Coord::new(c.x, r.hi[1] + 2, c.z)), None);
        assert_eq!(u.unfold(Coord::new(r.hi[0] + 1, c.y, r.hi[2] + 1)), None);
    }

    /// The first storage box's −X neighbour has chunk x below every box. It still unfolds.
    #[test]
    fn the_neg_x_side_unfolds_into_the_neighbour() {
        let a = atlas();
        let seams = Seams::new(vec![a.clone()]);
        let (o, size) = a.storage_box(Patch::Shell { band: 0, face: Face::PosY });
        let c = chunk_of([o[0], o[1] + size[1] - 64, o[2] + size[2] / 2]);
        let u = seams.unfold_at(c);
        let v = c.step(Face::NegX);
        let real = u.unfold(v).expect("past −x");
        assert_ne!(real, v, "the virtual chunk is not physical space");
        assert!(seams.in_storage(real), "lands in a box");
        assert_eq!(u.fold(real), v);
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
