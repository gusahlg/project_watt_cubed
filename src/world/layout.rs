//! Per-chunk sky and the column a chunk belongs to.

use std::hash::{Hash, Hasher};

use crate::coord::{ChunkCoord, Face};
use crate::space::FaceFrame;

use super::chunk::CHUNK_SIZE;

/// How skylight enters a chunk. From the generator, not from gravity.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sky {
    /// Skylight falls along `−face.normal()`. Columns run along the normal.
    Axis(Face),
    /// No ceiling. Full sky is kept across every face.
    Open,
}

impl Sky {
    /// Const code for the light kernel: face discriminant, or 6 for [`Open`](Self::Open).
    pub const OPEN_CODE: u8 = 6;

    #[inline]
    pub const fn code(self) -> u8 {
        match self {
            Sky::Axis(face) => face as u8,
            Sky::Open => Self::OPEN_CODE,
        }
    }
}

/// A run of chunks along `face`'s normal at local chunk tangents `(a, b) = (cu, cv)`.
///
/// `ColumnKey { face: PosY, a: cx, b: cz }` is today's `(cx, cz)` column.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ColumnKey {
    pub face: Face,
    pub a: i32,
    pub b: i32,
}

impl ColumnKey {
    /// The column `coord` belongs to, and its altitude chunk index.
    #[inline]
    pub fn of(face: Face, coord: ChunkCoord) -> (Self, i32) {
        let (cu, ca, cv) = FaceFrame::new(face).chunk_to_local(coord);
        (Self { face, a: cu, b: cv }, ca)
    }

    /// World chunk at altitude `alt` in this column.
    #[inline]
    pub fn chunk(self, alt: i32) -> ChunkCoord {
        FaceFrame::new(self.face).chunk_to_world((self.a, alt, self.b))
    }

    /// World-cell tangents `(u, v)` of face-local column `(lu, lv)`.
    /// PosY is `(cx * 16 + lu, cz * 16 + lv)`.
    #[inline]
    pub fn column_cell_uv(self, lu: i32, lv: i32) -> (i32, i32) {
        let frame = FaceFrame::new(self.face);
        let chunk = self.chunk(0);
        let (lx, ly, lz) = frame.index_to_world(lu as usize, 0, lv as usize);
        let s = CHUNK_SIZE as i32;
        let world = (chunk.x * s + lx as i32, chunk.y * s + ly as i32, chunk.z * s + lz as i32);
        let (u, _, v) = frame.cell_to_local(world);
        (u, v)
    }
}

impl Hash for ColumnKey {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.face.index());
        state.write_i32(self.a);
        state.write_i32(self.b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::FastMap;

    fn xorshift(s: &mut u32) -> i32 {
        *s ^= s.wrapping_shl(13);
        *s ^= s.wrapping_shr(17);
        *s ^= s.wrapping_shl(5);
        *s as i32
    }

    #[test]
    fn pos_y_column_is_todays_cx_cz() {
        let coord = ChunkCoord::new(4, -2, 9);
        let (key, alt) = ColumnKey::of(Face::PosY, coord);
        assert_eq!(key, ColumnKey { face: Face::PosY, a: 4, b: 9 });
        assert_eq!(alt, -2);
        assert_eq!(key.chunk(alt), coord);
        assert_eq!(key.column_cell_uv(3, 5), (4 * 16 + 3, 9 * 16 + 5));
    }

    #[test]
    fn columns_round_trip_for_every_face() {
        let mut state = 0xA11CEu32;
        let mut samples = vec![-8, -1, 0, 1, 3, 15];
        for _ in 0..12 {
            samples.push(xorshift(&mut state));
        }
        for face in Face::ALL {
            for &x in &samples {
                for &y in &samples {
                    for &z in &samples {
                        let coord = ChunkCoord::new(x, y, z);
                        let (key, alt) = ColumnKey::of(face, coord);
                        assert_eq!(key.face, face);
                        assert_eq!(key.chunk(alt), coord, "{face:?} {coord:?}");
                        let (again, alt2) = ColumnKey::of(face, key.chunk(alt + 3));
                        assert_eq!(again, key);
                        assert_eq!(alt2, alt + 3);
                    }
                }
            }
        }
    }

    #[test]
    fn column_key_keys_a_fast_map() {
        let mut map: FastMap<ColumnKey, i32> = FastMap::default();
        let key = ColumnKey { face: Face::NegX, a: -3, b: 9 };
        map.insert(key, 7);
        assert_eq!(map.get(&key), Some(&7));
        assert_eq!(Sky::Axis(Face::PosY).code(), Face::PosY as u8);
        assert_eq!(Sky::Open.code(), Sky::OPEN_CODE);
    }
}
