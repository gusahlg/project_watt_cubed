//! Patchy light on an interior floor, and short glowing tips under a ceiling.
//!
//! Sites are a tangent grid. A cell asks the nine sites that can reach it. The `deep` scale
//! multiplies the chance; at 0 the cells and tips are gone. A site places one block: a wider
//! patch would re-flood the same floor chunk, and those chunks are on the block-light budget.
//! The cell stays on the shell, inside the margin `all_air` already requires.

use super::super::noise::hash2;
use super::{idx_i32, roll};
use crate::block::registry::BlockId;

/// Tangent spacing of a floor light. One block on this grid leaves most of the floor bare.
pub(super) const GAP: i64 = 16;

/// One `block`, jittered by a cell so neighbouring sites do not line up.
pub(super) fn at(salt: u32, scale: f32, t0: i64, t1: i64, chance: f32, block: BlockId) -> Option<BlockId> {
    if scale <= 0.0 || chance <= 0.0 {
        return None;
    }
    let p = chance * scale;
    let i0 = t0.div_euclid(GAP);
    let i1 = t1.div_euclid(GAP);
    for d1 in -1..=1 {
        for d0 in -1..=1 {
            let (Some(a0), Some(a1)) = (idx_i32(i0 + d0), idx_i32(i1 + d1)) else { continue };
            let h = hash2(salt, a0, a1);
            if !roll(h, p) {
                continue;
            }
            let p0 = (i0 + d0) * GAP + GAP / 2 + i64::from((h >> 8) % 3) - 1;
            let p1 = (i1 + d1) * GAP + GAP / 2 + i64::from((h >> 12) % 3) - 1;
            if t0 == p0 && t1 == p1 {
                return Some(block);
            }
        }
    }
    None
}

/// One column hanging inward from a ceiling, 3 to 6 blocks, on a wider grid than the floor.
pub(super) fn tip(salt: u32, scale: f32, t0: i64, t1: i64, drop: i64, block: BlockId) -> Option<BlockId> {
    const GAP: i64 = 22;
    const LEN: i64 = 6;
    if drop <= 0 || drop > LEN || scale <= 0.0 {
        return None;
    }
    let i0 = t0.div_euclid(GAP);
    let i1 = t1.div_euclid(GAP);
    for d1 in -1..=1 {
        for d0 in -1..=1 {
            let (Some(a0), Some(a1)) = (idx_i32(i0 + d0), idx_i32(i1 + d1)) else { continue };
            let h = hash2(salt, a0, a1);
            if !roll(h, 0.65 * scale) {
                continue;
            }
            let len = 3 + i64::from((h >> 16) % 4);
            if drop > len {
                continue;
            }
            let p0 = (i0 + d0) * GAP + GAP / 2 + i64::from((h >> 8) % 3) - 1;
            let p1 = (i1 + d1) * GAP + GAP / 2 + i64::from((h >> 12) % 3) - 1;
            if t0 == p0 && t1 == p1 {
                return Some(block);
            }
        }
    }
    None
}
