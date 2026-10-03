//! Trees: one candidate per 7 m cell, kept by the theme's density on gentle ground. Broadleaf
//! (green, autumn or blossom crowns on timber) and conifers (stacked pine tiers).

use std::sync::Arc;

use super::noise::{hash2, unit};
use super::province::Species;
use super::shape::Shape;
use super::Materials;
use crate::block::registry::BlockId;

/// Tallest tree above its base, crown included.
pub const MAX_TREE_HEIGHT: i32 = 16;
const CELL: i32 = 7;

#[derive(Clone, Copy)]
enum Kind {
    Broad { crown: BlockId },
    Conifer,
}

#[derive(Clone, Copy)]
struct Tree {
    x: i32,
    z: i32,
    base: i32,
    trunk: i32,
    radius: i32,
    kind: Kind,
    trim: u32,
}

pub struct Trees {
    s: u32,
    m: Arc<Materials>,
}

impl Trees {
    pub fn new(s: u32, m: Arc<Materials>) -> Self {
        Self { s, m }
    }

    fn in_cell(&self, shape: &Shape, cx: i32, cz: i32) -> Option<Tree> {
        let h = hash2(self.s, cx, cz);
        let x = cx * CELL + 1 + (h % 5) as i32;
        let z = cz * CELL + 1 + ((h >> 3) % 5) as i32;
        let roll = unit(hash2(self.s ^ 0x7111, cx, cz));
        let col = shape.column(x, z);
        if col.slope4 >= 4 || col.height > 300 {
            return None;
        }
        let m = &self.m;
        let kind = match col.species {
            Species::None => return None,
            Species::Broadleaf => Kind::Broad { crown: m.leaves },
            Species::Conifer => Kind::Conifer,
            Species::Autumn => Kind::Broad { crown: m.autumn },
            Species::Blossom => Kind::Broad { crown: m.blossom },
        };
        if roll >= col.trees {
            return None;
        }
        let (trunk, radius) = match kind {
            Kind::Broad { .. } => (4 + (h >> 6) as i32 % 4, 2 + (h >> 9) as i32 % 2),
            Kind::Conifer => (6 + (h >> 6) as i32 % 6, 2 + (h >> 9) as i32 % 2),
        };
        Some(Tree { x, z, base: col.height, trunk, radius, kind, trim: h })
    }

    fn block(&self, t: &Tree, x: i32, y: i32, z: i32) -> Option<BlockId> {
        let (dx, dy, dz) = (x - t.x, y - t.base, z - t.z);
        if dy < 0 {
            return None;
        }
        if dx == 0 && dz == 0 && dy < t.trunk {
            return Some(self.m.timber);
        }
        match t.kind {
            Kind::Broad { crown } => {
                let cy = dy - (t.trunk - 1);
                let r = t.radius;
                let d2 = dx * dx + dz * dz + cy * cy * 2;
                if d2 <= r * r + 1 && cy >= -1 && cy <= r {
                    // Ragged edge: drop some outermost leaves.
                    let edge = d2 >= r * r - 1;
                    let cut = edge && (t.trim.rotate_left(((dx * 7 + dz * 3 + cy * 5) & 31) as u32) & 3) == 0;
                    return (!cut).then_some(crown);
                }
                None
            }
            Kind::Conifer => {
                if dx == 0 && dz == 0 && dy <= t.trunk + 1 {
                    return Some(self.m.pine);
                }
                if dy < 2 || dy > t.trunk {
                    return None;
                }
                // Tiers shrink toward the top; every other ring is pulled in, so the crown reads
                // as stacked boughs.
                let left = t.trunk + 1 - dy;
                let r = (left * (t.radius + 1) / t.trunk).min(t.radius) - dy % 2;
                (r > 0 && dx.abs() + dz.abs() <= r).then_some(self.m.pine)
            }
        }
    }

    /// The tree block at a cell above the ground, if any (per-voxel path). Cells are visited in the
    /// same order as [`blocks_in`](Self::blocks_in), so overlaps resolve identically.
    pub fn block_at(&self, shape: &Shape, x: i32, y: i32, z: i32) -> Option<BlockId> {
        let (cx, cz) = (x.div_euclid(CELL), z.div_euclid(CELL));
        for tz in cz - 1..=cz + 1 {
            for tx in cx - 1..=cx + 1 {
                if let Some(t) = self.in_cell(shape, tx, tz) {
                    if y - t.base > MAX_TREE_HEIGHT {
                        continue;
                    }
                    if let Some(id) = self.block(&t, x, y, z) {
                        return Some(id);
                    }
                }
            }
        }
        None
    }

    /// Every tree block in the `n`-wide square at `(x0, z0)` (batch path), in cell order.
    pub fn blocks_in(&self, shape: &Shape, x0: i32, z0: i32, n: i32) -> Vec<(i32, i32, i32, BlockId)> {
        let mut out = Vec::new();
        let (c0x, c0z) = ((x0 - 4).div_euclid(CELL), (z0 - 4).div_euclid(CELL));
        let (c1x, c1z) = ((x0 + n + 4).div_euclid(CELL), (z0 + n + 4).div_euclid(CELL));
        // Visit cells so that, for any cell in the square, its 3×3 neighbourhood comes in the same
        // relative order as `block_at` visits it: row-major by (tz, tx).
        let mut trees = Vec::new();
        for tz in c0z..=c1z {
            for tx in c0x..=c1x {
                if let Some(t) = self.in_cell(shape, tx, tz) {
                    trees.push(t);
                }
            }
        }
        // Resolve overlaps per cell exactly like `block_at`: the first tree in (tz, tx) order wins.
        for z in z0..z0 + n {
            for x in x0..x0 + n {
                let (cx, cz) = (x.div_euclid(CELL), z.div_euclid(CELL));
                let near: Vec<&Tree> = trees
                    .iter()
                    .filter(|t| (t.x.div_euclid(CELL) - cx).abs() <= 1 && (t.z.div_euclid(CELL) - cz).abs() <= 1)
                    .collect();
                if near.is_empty() {
                    continue;
                }
                let lo = near.iter().map(|t| t.base).min().unwrap_or(0);
                let hi = near.iter().map(|t| t.base).max().unwrap_or(0) + MAX_TREE_HEIGHT;
                for y in lo..=hi {
                    for t in &near {
                        if y - t.base > MAX_TREE_HEIGHT {
                            continue;
                        }
                        if let Some(id) = self.block(t, x, y, z) {
                            out.push((x, y, z, id));
                            break;
                        }
                    }
                }
            }
        }
        out
    }
}
