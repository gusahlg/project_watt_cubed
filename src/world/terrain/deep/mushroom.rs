//! Giant mushrooms on a cavern or chamber floor, and the same plant hanging from the ceiling.
//!
//! Sites are a flat grid in the face's tangent plane. A column asks only the nine sites that can
//! reach it (the cap is smaller than the cell).

use super::super::noise::hash2;
use super::{Up, idx_i32, roll, tangent};
use crate::block::registry::BlockId;

const CELL: i64 = 28;
/// Tallest stem plus cap. Callers skip the grid when a voxel is farther inside than this.
pub(super) const REACH: i64 = 22;

/// A mushroom block at `rel`, if a floor or ceiling site covers it.
pub(super) fn occupy(
    seed: u32,
    scale: f32,
    rel: [i64; 3],
    up: Up,
    m: &super::super::Materials,
    floor_at: impl Fn(i64, i64) -> Option<i64>,
    ceil_at: impl Fn(i64, i64) -> Option<i64>,
) -> Option<BlockId> {
    let (t0, t1) = tangent(rel, up.axis);
    let o = up.outward(rel);
    let i0 = t0.div_euclid(CELL);
    let i1 = t1.div_euclid(CELL);
    for d1 in -1..=1 {
        for d0 in -1..=1 {
            let (Some(a0), Some(a1)) = (idx_i32(i0 + d0), idx_i32(i1 + d1)) else { continue };
            let h = hash2(seed ^ 0x3455_4001, a0, a1);
            if !roll(h, 0.6 * scale) {
                continue;
            }
            let jitter = |shift: u32| i64::from((h >> shift) % 7) - 3;
            let p0 = (i0 + d0) * CELL + CELL / 2 + jitter(8);
            let p1 = (i1 + d1) * CELL + CELL / 2 + jitter(12);
            let height = 12 + i64::from((h >> 16) % 11);
            let stem_r = 1 + i64::from((h >> 20) % 2);
            let cap_r = 4 + i64::from((h >> 21) % 4);
            let horiz = {
                let (dx, dz) = (t0 - p0, t1 - p1);
                dx * dx + dz * dz
            };
            if let Some(id) = plant(m, h, o, floor_at(p0, p1), 1, height, stem_r, cap_r, horiz, true) {
                return Some(id);
            }
            if let Some(id) = plant(m, h, o, ceil_at(p0, p1), -1, height, stem_r, cap_r, horiz, false) {
                return Some(id);
            }
        }
    }
    None
}

/// Farther inside a sphere than any mushroom can reach (one sqrt, then the grid is skipped).
#[inline]
pub(super) fn out_of_reach(radius: i64, dist_sq: i64) -> bool {
    radius as f64 - (dist_sq as f64).sqrt() > REACH as f64
}

/// `dir` is +1 growing out from a floor, −1 hanging down from a ceiling.
fn plant(
    m: &super::super::Materials,
    h: u32,
    o: i64,
    anchor: Option<i64>,
    dir: i64,
    height: i64,
    stem_r: i64,
    cap_r: i64,
    horiz: i64,
    floor: bool,
) -> Option<BlockId> {
    let anchor = anchor?;
    let rise = (o - anchor) * dir;
    if rise <= 0 || rise > height {
        return None;
    }
    let cap_t = 3;
    if rise > height - cap_t {
        if horiz > cap_r * cap_r {
            return None;
        }
        // The underside of a floor cap is the glow.
        if floor && rise == height - cap_t + 1 && horiz > stem_r * stem_r {
            return Some(m.glowshroom);
        }
        return Some(if floor {
            if h & 1 == 0 { m.cap_red } else { m.cap_brown }
        } else {
            m.bark
        });
    }
    (horiz <= stem_r * stem_r).then_some(if floor { m.stem } else { m.darkwood })
}
