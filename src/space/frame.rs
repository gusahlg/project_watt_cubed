//! Right-handed face frame: world ↔ face-local `(u, a, v)`.
//!
//! `a` is altitude along the face normal. PosY is the identity.

use voxel_engine::DVec3;

use crate::coord::{ChunkCoord, Face};

/// In-chunk edge. Matches [`crate::world::chunk::CHUNK_SIZE`].
const CHUNK: usize = 16;

/// One signed world axis. `axis` is 0=X, 1=Y, 2=Z; `sign` is ±1.
#[derive(Clone, Copy)]
struct Axis {
    axis: u8,
    sign: i8,
}

const fn ax(axis: u8, sign: i8) -> Axis {
    Axis { axis, sign }
}

/// `(t_u, n, t_v)` indexed by [`Face`]'s discriminant. Each is a signed permutation
/// and the basis is right-handed (`det = +1`).
const BASIS: [[Axis; 3]; 6] = [
    [ax(1, 1), ax(0, -1), ax(2, 1)], // NegX: (+Y, −X, +Z)
    [ax(1, -1), ax(0, 1), ax(2, 1)], // PosX: (−Y, +X, +Z)
    [ax(0, 1), ax(2, -1), ax(1, 1)], // NegZ: (+X, −Z, +Y)
    [ax(0, 1), ax(2, 1), ax(1, -1)], // PosZ: (+X, +Z, −Y)
    [ax(0, 1), ax(1, -1), ax(2, -1)], // NegY: (+X, −Y, −Z)
    [ax(0, 1), ax(1, 1), ax(2, 1)],  // PosY: (+X, +Y, +Z)
];

/// Maps between world coordinates and the face-local frame of one [`Face`].
#[derive(Clone, Copy, Debug)]
pub struct FaceFrame {
    face: Face,
}

impl FaceFrame {
    #[inline]
    pub const fn new(face: Face) -> Self {
        Self { face }
    }

    #[inline]
    pub const fn face(self) -> Face {
        self.face
    }

    #[inline]
    fn basis(self) -> [Axis; 3] {
        BASIS[self.face as usize]
    }

    /// `(u, a, v) = (t_u·p, n·p, t_v·p)`.
    #[inline]
    pub fn cell_to_local(self, p: (i32, i32, i32)) -> (i32, i32, i32) {
        let [tu, n, tv] = self.basis();
        (dot(tu, p), dot(n, p), dot(tv, p))
    }

    /// `p = t_u·u + n·a + t_v·v`.
    #[inline]
    pub fn cell_to_world(self, (u, a, v): (i32, i32, i32)) -> (i32, i32, i32) {
        let [tu, n, tv] = self.basis();
        let mut w = [0i32; 3];
        w[tu.axis as usize] = u * tu.sign as i32;
        w[n.axis as usize] = a * n.sign as i32;
        w[tv.axis as usize] = v * tv.sign as i32;
        (w[0], w[1], w[2])
    }

    #[inline]
    pub fn point_to_local(self, p: DVec3) -> DVec3 {
        let [tu, n, tv] = self.basis();
        DVec3::new(dot_f(tu, p), dot_f(n, p), dot_f(tv, p))
    }

    #[inline]
    pub fn point_to_world(self, p: DVec3) -> DVec3 {
        let [tu, n, tv] = self.basis();
        let mut w = [0.0; 3];
        w[tu.axis as usize] = p.x * tu.sign as f64;
        w[n.axis as usize] = p.y * n.sign as f64;
        w[tv.axis as usize] = p.z * tv.sign as f64;
        DVec3::new(w[0], w[1], w[2])
    }

    /// Chunk coord in face-local order `(cu, ca, cv)`. A flipped axis maps `c` to `−c−1`.
    #[inline]
    pub fn chunk_to_local(self, c: ChunkCoord) -> (i32, i32, i32) {
        let comps = [c.x, c.y, c.z];
        let [tu, n, tv] = self.basis();
        (
            map_chunk(tu.sign, comps[tu.axis as usize]),
            map_chunk(n.sign, comps[n.axis as usize]),
            map_chunk(tv.sign, comps[tv.axis as usize]),
        )
    }

    /// Inverse of [`chunk_to_local`](Self::chunk_to_local). The chunk map is an involution per axis.
    #[inline]
    pub fn chunk_to_world(self, (cu, ca, cv): (i32, i32, i32)) -> ChunkCoord {
        let [tu, n, tv] = self.basis();
        let mut w = [0i32; 3];
        w[tu.axis as usize] = map_chunk(tu.sign, cu);
        w[n.axis as usize] = map_chunk(n.sign, ca);
        w[tv.axis as usize] = map_chunk(tv.sign, cv);
        ChunkCoord::new(w[0], w[1], w[2])
    }

    /// In-chunk index in face-local order `(lu, la, lv)`. A flipped axis maps `l` to `15−l`.
    #[inline]
    pub fn index_to_local(self, lx: usize, ly: usize, lz: usize) -> (usize, usize, usize) {
        let comps = [lx, ly, lz];
        let [tu, n, tv] = self.basis();
        (
            map_index(tu.sign, comps[tu.axis as usize]),
            map_index(n.sign, comps[n.axis as usize]),
            map_index(tv.sign, comps[tv.axis as usize]),
        )
    }

    /// Inverse of [`index_to_local`](Self::index_to_local).
    #[inline]
    pub fn index_to_world(self, lu: usize, la: usize, lv: usize) -> (usize, usize, usize) {
        let [tu, n, tv] = self.basis();
        let mut w = [0usize; 3];
        w[tu.axis as usize] = map_index(tu.sign, lu);
        w[n.axis as usize] = map_index(n.sign, la);
        w[tv.axis as usize] = map_index(tv.sign, lv);
        (w[0], w[1], w[2])
    }

    /// World flat index → face-local flat index (`u + v*16 + a*256`).
    #[inline]
    pub fn local_chunk_index(self, world_index: usize) -> usize {
        let (x, y, z) = world_local(world_index);
        let (u, a, v) = self.index_to_local(x, y, z);
        u + v * CHUNK + a * CHUNK * CHUNK
    }

    /// Lowest altitude of any cell in `coord` (the chunk's `alt0`).
    /// Positive axes: `c * 16`. Flipped axes: `−(c * 16 + 15)`. The open plane
    /// just above the chunk is `alt0 + 16` either way.
    #[inline]
    pub fn chunk_alt0(self, coord: ChunkCoord) -> i32 {
        let n = self.basis()[1];
        let c = match n.axis {
            0 => coord.x,
            1 => coord.y,
            _ => coord.z,
        };
        let s = CHUNK as i32;
        if n.sign > 0 { c * s } else { -(c * s + (s - 1)) }
    }
}

#[inline]
fn dot(axis: Axis, (x, y, z): (i32, i32, i32)) -> i32 {
    let c = match axis.axis {
        0 => x,
        1 => y,
        _ => z,
    };
    c * axis.sign as i32
}

#[inline]
fn dot_f(axis: Axis, p: DVec3) -> f64 {
    let c = match axis.axis {
        0 => p.x,
        1 => p.y,
        _ => p.z,
    };
    c * axis.sign as f64
}

/// Flipped chunk axes send `c` to `−c−1` so the chunk still covers the same cells.
#[inline]
fn map_chunk(sign: i8, c: i32) -> i32 {
    if sign > 0 { c } else { -c - 1 }
}

#[inline]
fn map_index(sign: i8, l: usize) -> usize {
    if sign > 0 { l } else { CHUNK - 1 - l }
}

fn world_local(index: usize) -> (usize, usize, usize) {
    let x = index % CHUNK;
    let z = (index / CHUNK) % CHUNK;
    let y = index / (CHUNK * CHUNK);
    (x, y, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(s: &mut u32) -> i32 {
        *s ^= s.wrapping_shl(13);
        *s ^= s.wrapping_shr(17);
        *s ^= s.wrapping_shl(5);
        *s as i32
    }

    fn basis_det(face: Face) -> i32 {
        let frame = FaceFrame::new(face);
        let (ux, uy, uz) = frame.cell_to_world((1, 0, 0));
        let (nx, ny, nz) = frame.cell_to_world((0, 1, 0));
        let (vx, vy, vz) = frame.cell_to_world((0, 0, 1));
        // t_u · (n × t_v)
        ux * (ny * vz - nz * vy) - uy * (nx * vz - nz * vx) + uz * (nx * vy - ny * vx)
    }

    #[test]
    fn every_basis_is_right_handed_and_pos_y_is_identity() {
        assert_eq!(CHUNK, crate::world::chunk::CHUNK_SIZE);
        for face in Face::ALL {
            assert_eq!(basis_det(face), 1, "{face:?} det");
            let (nx, ny, nz) = FaceFrame::new(face).cell_to_world((0, 1, 0));
            assert_eq!((nx, ny, nz), face.normal(), "{face:?} normal column");
        }
        let id = FaceFrame::new(Face::PosY);
        assert_eq!(id.cell_to_local((3, -5, 9)), (3, -5, 9));
        assert_eq!(id.cell_to_world((3, -5, 9)), (3, -5, 9));
        assert_eq!(id.index_to_local(1, 2, 3), (1, 2, 3));
        assert_eq!(id.index_to_world(1, 2, 3), (1, 2, 3));
        let c = ChunkCoord::new(-4, 7, 2);
        assert_eq!(id.chunk_to_local(c), (-4, 7, 2));
        assert_eq!(id.chunk_to_world((-4, 7, 2)), c);
        assert_eq!(id.chunk_alt0(c), 7 * 16);
    }

    #[test]
    fn cells_chunks_and_indices_round_trip_on_every_face() {
        let mut state = 0xC0FFEE_u32;
        let mut samples = vec![-100, -17, -16, -1, 0, 1, 15, 16, 31, 40, 100];
        for _ in 0..24 {
            samples.push(xorshift(&mut state));
        }
        for face in Face::ALL {
            let frame = FaceFrame::new(face);
            for &x in &samples {
                for &y in &samples {
                    for &z in &samples {
                        let p = (x, y, z);
                        assert_eq!(frame.cell_to_world(frame.cell_to_local(p)), p, "{face:?} cell {p:?}");
                        let back = frame.cell_to_local(frame.cell_to_world(p));
                        assert_eq!(back, p, "{face:?} cell inverse {p:?}");
                    }
                }
            }
            for &x in &samples {
                for &y in &samples {
                    for &z in &samples {
                        let c = ChunkCoord::new(x, y, z);
                        assert_eq!(frame.chunk_to_world(frame.chunk_to_local(c)), c, "{face:?} chunk");
                        let (cu, ca, cv) = (x, y, z);
                        assert_eq!(
                            frame.chunk_to_local(frame.chunk_to_world((cu, ca, cv))),
                            (cu, ca, cv),
                            "{face:?} chunk inverse"
                        );
                    }
                }
            }
            for l in 0..CHUNK {
                for m in 0..CHUNK {
                    for n in 0..CHUNK {
                        let (u, a, v) = frame.index_to_local(l, m, n);
                        assert_eq!(frame.index_to_world(u, a, v), (l, m, n));
                        let (x, y, z) = frame.index_to_world(l, m, n);
                        assert_eq!(frame.index_to_local(x, y, z), (l, m, n));
                    }
                }
            }
            for i in 0..CHUNK * CHUNK * CHUNK {
                let (x, y, z) = world_local(i);
                let local = frame.local_chunk_index(i);
                let (u, a, v) = frame.index_to_local(x, y, z);
                assert_eq!(local, u + v * CHUNK + a * CHUNK * CHUNK);
                let (wx, wy, wz) = frame.index_to_world(u, a, v);
                assert_eq!((wx, wy, wz), (x, y, z));
            }
        }
    }

    #[test]
    fn points_round_trip_and_alt0_is_the_minimum_altitude() {
        for face in Face::ALL {
            let frame = FaceFrame::new(face);
            for p in [
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(1.5, -2.25, 3.0),
                DVec3::new(-40.0, 16.0, -0.5),
            ] {
                let back = frame.point_to_world(frame.point_to_local(p));
                assert!((back - p).length() < 1e-9, "{face:?} point {p:?} -> {back:?}");
            }
            for c in [ChunkCoord::new(0, 0, 0), ChunkCoord::new(-3, 2, -1), ChunkCoord::new(4, -5, 6)] {
                let mut min_a = i32::MAX;
                for i in 0..CHUNK {
                    let mut local = [0usize; 3];
                    local[face.axis()] = i;
                    let world = (
                        c.x * CHUNK as i32 + local[0] as i32,
                        c.y * CHUNK as i32 + local[1] as i32,
                        c.z * CHUNK as i32 + local[2] as i32,
                    );
                    min_a = min_a.min(frame.cell_to_local(world).1);
                }
                assert_eq!(frame.chunk_alt0(c), min_a, "{face:?} alt0 at {c:?}");
            }
        }
    }
}
