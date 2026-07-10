//! Coordinate newtypes and the split/join conversion between an absolute
//! world block coordinate and a `(chunk, local)` pair.
//!
//! This module owns the coordinate split and join logic, chunk-space regions
//! ([`ChunkBox`]), and chunk face helpers ([`Face`], [`ByFace`]) used by
//! streaming and meshing. [`Local`]'s fields are private; the only way to
//! create a valid local coordinate is through split or [`Local::new`].

use std::ops::{Index, IndexMut};

use crate::world::chunk::CHUNK_SIZE;

/// An absolute world voxel coordinate (unbounded integer range).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlockCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

/// A chunk coordinate; combined with a local coordinate to form a world block position.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ChunkCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

/// A chunk-local coordinate; each component is `< CHUNK_SIZE`. `u8` holds
/// `CHUNK_SIZE - 1` (= 15) with room to spare; `lx/ly/lz` expose the `usize`
/// form `Chunk::index` wants. Fields are PRIVATE so an out-of-range `Local`
/// can never be constructed (see the checked `new` constructor below).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Local {
    x: u8,
    y: u8,
    z: u8,
}

impl Local {
    /// Checked constructor for a component triple arriving from outside the
    /// split (e.g. a saved edit index). `None` if any component `>= CHUNK_SIZE`.
    #[inline]
    pub fn new(x: u8, y: u8, z: u8) -> Option<Self> {
        let n = CHUNK_SIZE as u8;
        (x < n && y < n && z < n).then_some(Self { x, y, z })
    }

    /// Local X as `usize`, for [`Chunk::index`](crate::world::chunk::Chunk::index).
    #[inline]
    pub fn lx(self) -> usize {
        self.x as usize
    }
    /// Local Y as `usize`.
    #[inline]
    pub fn ly(self) -> usize {
        self.y as usize
    }
    /// Local Z as `usize`.
    #[inline]
    pub fn lz(self) -> usize {
        self.z as usize
    }
}

impl BlockCoord {
    #[inline]
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Splits a world block coordinate into a chunk and its local position within that chunk.
    #[inline]
    pub fn split(self) -> (ChunkCoord, Local) {
        let s = CHUNK_SIZE as i32;
        let chunk = ChunkCoord {
            x: self.x.div_euclid(s),
            y: self.y.div_euclid(s),
            z: self.z.div_euclid(s),
        };
        let local = Local {
            x: self.x.rem_euclid(s) as u8,
            y: self.y.rem_euclid(s) as u8,
            z: self.z.rem_euclid(s) as u8,
        };
        (chunk, local)
    }

    /// Reconstructs a world block coordinate from a chunk and local position.
    #[inline]
    pub fn join(c: ChunkCoord, l: Local) -> Self {
        let s = CHUNK_SIZE as i32;
        Self {
            x: c.x * s + l.x as i32,
            y: c.y * s + l.y as i32,
            z: c.z * s + l.z as i32,
        }
    }

    #[inline]
    pub fn to_tuple(self) -> (i32, i32, i32) {
        (self.x, self.y, self.z)
    }
}

impl From<(i32, i32, i32)> for BlockCoord {
    #[inline]
    fn from((x, y, z): (i32, i32, i32)) -> Self {
        Self { x, y, z }
    }
}

impl ChunkCoord {
    #[inline]
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn to_tuple(self) -> (i32, i32, i32) {
        (self.x, self.y, self.z)
    }

    /// The neighbouring chunk across `face`: `(cx+dx, cy+dy, cz+dz)`.
    #[inline]
    pub fn step(self, face: Face) -> Self {
        let (dx, dy, dz) = face.delta();
        Self { x: self.x + dx, y: self.y + dy, z: self.z + dz }
    }

    /// Horizontal distance to chunk `o` (max of x and z difference).
    #[inline]
    pub fn ring(self, o: ChunkCoord) -> i32 {
        (self.x - o.x).abs().max((self.z - o.z).abs())
    }

    /// Vertical (chunk-layer) distance to `o`.
    #[inline]
    pub fn updown(self, o: ChunkCoord) -> i32 {
        (self.y - o.y).abs()
    }
}

impl From<(i32, i32, i32)> for ChunkCoord {
    #[inline]
    fn from((x, y, z): (i32, i32, i32)) -> Self {
        Self { x, y, z }
    }
}

/// The six axis-aligned chunk-face directions. The discriminant doubles as
/// the index into the mesher's border planes ([`ByFace`]), so a face's
/// neighbour offset and its border slice always line up without a separate
/// index to keep in sync.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(usize)]
pub enum Face {
    NegX = 0,
    PosX = 1,
    NegZ = 2,
    PosZ = 3,
    NegY = 4,
    PosY = 5,
}

impl Face {
    /// Every face in discriminant order.
    pub const ALL: [Face; 6] = [
        Face::NegX,
        Face::PosX,
        Face::NegZ,
        Face::PosZ,
        Face::NegY,
        Face::PosY,
    ];

    /// The offset `(dx, dy, dz)` for stepping to an adjacent chunk in this face direction.
    #[inline]
    pub const fn delta(self) -> (i32, i32, i32) {
        match self {
            Face::NegX => (-1, 0, 0),
            Face::PosX => (1, 0, 0),
            Face::NegZ => (0, 0, -1),
            Face::PosZ => (0, 0, 1),
            Face::NegY => (0, -1, 0),
            Face::PosY => (0, 1, 0),
        }
    }

    /// The opposing face.
    #[inline]
    pub const fn opposite(self) -> Face {
        Face::ALL[(self as usize) ^ 1]
    }

    /// Whether a cell at local coord `l` sits on this face — i.e. an edit there
    /// also changes the neighbour across this face.
    #[inline]
    pub fn touches(self, l: Local) -> bool {
        let edge = (CHUNK_SIZE - 1) as u8;
        match self {
            Face::NegX => l.x == 0,
            Face::PosX => l.x == edge,
            Face::NegZ => l.z == 0,
            Face::PosZ => l.z == edge,
            Face::NegY => l.y == 0,
            Face::PosY => l.y == edge,
        }
    }
}

/// A value per chunk face, keyed by [`Face`] instead of a loose plane index.
/// `[T; 6]` indexed through `Index<Face>`, so slot `k` always belongs to
/// `Face::ALL[k]`.
pub struct ByFace<T>([T; 6]);

impl<T> ByFace<T> {
    /// Build one value per face, in [`Face::ALL`] order.
    #[inline]
    pub fn from_fn(f: impl FnMut(Face) -> T) -> Self {
        ByFace(Face::ALL.map(f))
    }
}

impl<T> Index<Face> for ByFace<T> {
    type Output = T;
    #[inline]
    fn index(&self, f: Face) -> &T {
        &self.0[f as usize]
    }
}

impl<T> IndexMut<Face> for ByFace<T> {
    #[inline]
    fn index_mut(&mut self, f: Face) -> &mut T {
        &mut self.0[f as usize]
    }
}

/// A value per draw [`Pass`], keyed by `Pass` instead of a loose index — the
/// per-pass analogue of [`ByFace`]. Slot `k` always belongs to `Pass` with
/// discriminant `k`, and the width is [`Pass::COUNT`](voxel_engine::Pass::COUNT),
/// so the container follows the enum by construction — adding a pass never widens
/// this type by hand. Used for the per-technique mesh product on both its CPU side
/// (`ByPass<MeshData>`) and its GPU side (`ByPass<Option<OwnedMesh>>`).
#[derive(Debug, PartialEq, Eq)]
pub struct ByPass<T>([T; voxel_engine::Pass::COUNT]);

impl<T> ByPass<T> {
    /// Build one value per pass, in discriminant (= draw) order.
    #[inline]
    pub fn from_fn(mut f: impl FnMut(voxel_engine::Pass) -> T) -> Self {
        ByPass(voxel_engine::Pass::ALL.map(&mut f))
    }
    /// Iterate `(pass, &value)` in pass order.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (voxel_engine::Pass, &T)> {
        voxel_engine::Pass::ALL.into_iter().zip(self.0.iter())
    }
    /// Mutably iterate `(pass, &mut value)` in pass order.
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (voxel_engine::Pass, &mut T)> {
        voxel_engine::Pass::ALL.into_iter().zip(self.0.iter_mut())
    }
    /// Consume into `(pass, value)` pairs in pass order.
    #[inline]
    pub fn into_iter_passes(self) -> impl Iterator<Item = (voxel_engine::Pass, T)> {
        voxel_engine::Pass::ALL.into_iter().zip(self.0)
    }
    /// Consume into the raw per-pass slots, in discriminant order.
    #[inline]
    pub fn into_slots(self) -> [T; voxel_engine::Pass::COUNT] {
        self.0
    }
}

impl<T> Index<voxel_engine::Pass> for ByPass<T> {
    type Output = T;
    #[inline]
    fn index(&self, p: voxel_engine::Pass) -> &T {
        &self.0[p as usize]
    }
}

impl<T> IndexMut<voxel_engine::Pass> for ByPass<T> {
    #[inline]
    fn index_mut(&mut self, p: voxel_engine::Pass) -> &mut T {
        &mut self.0[p as usize]
    }
}

/// A chunk-space box around `center` with horizontal radius `rh` and vertical radius `rv`.
/// Represents "the region around the player" for membership tests and iteration.
#[derive(Clone, Copy, Debug)]
pub struct ChunkBox {
    pub center: ChunkCoord,
    pub rh: i32,
    pub rv: i32,
}

impl ChunkBox {
    #[inline]
    pub fn new(center: ChunkCoord, rh: i32, rv: i32) -> Self {
        Self { center, rh, rv }
    }

    /// Whether `c` lies within the box (inclusive on every axis).
    #[inline]
    pub fn contains(self, c: ChunkCoord) -> bool {
        c.ring(self.center) <= self.rh && c.updown(self.center) <= self.rv
    }

    /// Every chunk coord in the box, iterated x → z → y (matching the triple
    /// loops this replaced, so the enqueue order is unchanged).
    pub fn coords(self) -> impl Iterator<Item = ChunkCoord> {
        let ChunkBox { center: c, rh, rv } = self;
        (c.x - rh..=c.x + rh).flat_map(move |x| {
            (c.z - rh..=c.z + rh)
                .flat_map(move |z| (c.y - rv..=c.y + rv).map(move |y| ChunkCoord::new(x, y, z)))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The old split behaviour, reimplemented independently here so the test
    /// checks `split()` against plain `div_euclid`/`rem_euclid` rather than
    /// against itself.
    fn reference_split(x: i32, y: i32, z: i32) -> ((i32, i32, i32), (usize, usize, usize)) {
        let s = CHUNK_SIZE as i32;
        (
            (x.div_euclid(s), y.div_euclid(s), z.div_euclid(s)),
            (
                x.rem_euclid(s) as usize,
                y.rem_euclid(s) as usize,
                z.rem_euclid(s) as usize,
            ),
        )
    }

    #[test]
    fn split_join_roundtrip_including_negatives() {
        let s = CHUNK_SIZE as i32;
        let samples = [-2 * s, -s - 1, -s, -1, 0, 1, s - 1, s, s + 3, 2 * s, 100, -100];
        for &x in &samples {
            for &y in &samples {
                for &z in &samples {
                    let b = BlockCoord::new(x, y, z);
                    let (c, l) = b.split();
                    assert_eq!(BlockCoord::join(c, l), b, "roundtrip at {:?}", (x, y, z));
                    assert!(l.x < s as u8 && l.y < s as u8 && l.z < s as u8, "local < CHUNK_SIZE");
                }
            }
        }
    }

    #[test]
    fn split_matches_existing_div_rem_euclid() {
        for &(x, y, z) in &[(0, 0, 0), (15, 16, 17), (-1, -16, -17), (-33, 5, -5), (100, -100, 200)] {
            let (c, l) = BlockCoord::new(x, y, z).split();
            let (rc, rl) = reference_split(x, y, z);
            assert_eq!(c.to_tuple(), rc, "chunk at {:?}", (x, y, z));
            assert_eq!((l.lx(), l.ly(), l.lz()), rl, "local at {:?}", (x, y, z));
        }
    }

    #[test]
    fn local_new_rejects_out_of_range() {
        let n = CHUNK_SIZE as u8;
        assert!(Local::new(0, 0, 0).is_some());
        assert!(Local::new(n - 1, n - 1, n - 1).is_some(), "the max valid cell");
        assert!(Local::new(n, 0, 0).is_none(), "== CHUNK_SIZE is out of range");
        assert!(Local::new(0, n, 0).is_none());
        assert!(Local::new(0, 0, 200).is_none());
    }

    #[test]
    fn byface_index_matches_all_order() {
        // A `ByFace` built by `from_fn` should read back each slot as the
        // face that produced it.
        let by = ByFace::from_fn(|f| f);
        for (k, &f) in Face::ALL.iter().enumerate() {
            assert_eq!(f as usize, k, "Face::ALL not in discriminant order");
            assert_eq!(by[f], f, "ByFace slot disagrees with its Face key");
        }
    }

    #[test]
    fn face_delta_touches_match_old_branches() {
        // `delta` reproduces the old NEIGHBOR_OFFSETS set (order doesn't matter).
        let old_offsets = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)];
        let mut got: Vec<_> = Face::ALL.iter().map(|f| f.delta()).collect();
        got.sort_unstable();
        let mut want = old_offsets.to_vec();
        want.sort_unstable();
        assert_eq!(got, want, "Face deltas != NEIGHBOR_OFFSETS set");

        // `touches` reproduces the old per-face `local[axis] == 0 / == last` test
        // for every face at a spread of locals: interior, single-face, corner.
        let edge = (CHUNK_SIZE - 1) as u8;
        for &(lx, ly, lz) in &[(0u8, 0u8, 0u8), (edge, edge, edge), (5, 0, 9), (0, 7, edge), (8, 8, 8)] {
            let l = Local::new(lx, ly, lz).unwrap();
            let want = |dx: i32, dy: i32, dz: i32| {
                let local = [lx as i32, ly as i32, lz as i32];
                let axis = [dx, dy, dz].iter().position(|&c| c != 0).unwrap();
                let delta = [dx, dy, dz][axis];
                if delta < 0 { local[axis] == 0 } else { local[axis] == edge as i32 }
            };
            for f in Face::ALL {
                let (dx, dy, dz) = f.delta();
                assert_eq!(f.touches(l), want(dx, dy, dz), "touches {f:?} at {:?}", (lx, ly, lz));
            }
        }
    }

    #[test]
    fn opposite_negates_the_delta_and_is_an_involution() {
        for f in Face::ALL {
            let (dx, dy, dz) = f.delta();
            let (ox, oy, oz) = f.opposite().delta();
            assert_eq!((ox, oy, oz), (-dx, -dy, -dz), "opposite of {f:?} must negate its normal");
            assert_eq!(f.opposite().opposite(), f, "opposite is its own inverse");
            assert_ne!(f.opposite(), f);
        }
    }

    #[test]
    fn chunkbox_coords_matches_old_triple_loop() {
        for &(rh, rv) in &[(0, 0), (1, 1), (2, 3), (6, 4)] {
            for &center in &[ChunkCoord::new(0, 0, 0), ChunkCoord::new(-3, 5, 2)] {
                let got: Vec<ChunkCoord> = ChunkBox::new(center, rh, rv).coords().collect();
                // The exact x → z → y order the old streaming loops used.
                let mut want = Vec::new();
                for x in (center.x - rh)..=(center.x + rh) {
                    for z in (center.z - rh)..=(center.z + rh) {
                        for y in (center.y - rv)..=(center.y + rv) {
                            want.push(ChunkCoord::new(x, y, z));
                        }
                    }
                }
                assert_eq!(got, want, "box coords differ at rh={rh} rv={rv} center={center:?}");
            }
        }
    }
}
