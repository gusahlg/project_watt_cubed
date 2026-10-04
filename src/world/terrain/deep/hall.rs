//! Dwarf halls in the Deep: a rare pillared room on a wide grid, joined to its neighbours by a
//! straight tunnel. A hall stays inside its cell; only the tunnel crosses the boundary.

use super::super::noise::hash3;
use super::{Ctx, DEEP_HI, Up, box_hits, for_cells, hash_site, plane_depth, roll};
use crate::block::registry::{AIR, BlockId};

const CELL: i64 = 2_560;
const P: f32 = 0.045;
const SALT: u32 = 0x0A11_0001;
const WIDTH: i64 = 22;
const AIR_H: i64 = 12;

struct Hall {
    center: [i64; 3],
    up: Up,
    long_axis: usize,
    length: i64,
    floor_o: i64,
}

fn may_host(ctx: &Ctx, idx: [i64; 3]) -> bool {
    let mid = std::array::from_fn(|a| idx[a] * CELL + CELL / 2);
    let pd = plane_depth(ctx.half, mid);
    let slack = CELL / 2;
    pd + slack > i64::from(super::super::cube::CRUST) && pd - slack <= i64::from(DEEP_HI)
}

fn hall_at(ctx: &Ctx, idx: [i64; 3]) -> Option<Hall> {
    if ctx.scale <= 0.0 || !may_host(ctx, idx) {
        return None;
    }
    let h = hash_site(ctx.seed ^ SALT, idx)?;
    if !roll(h, P * ctx.scale) {
        return None;
    }
    let origin: [i64; 3] = std::array::from_fn(|a| idx[a] * CELL);
    let center = std::array::from_fn(|a| origin[a] + CELL / 2 + (super::super::noise::unit(hash3(h, a as i32, 2, 3)) * 80.0 - 40.0) as i64);
    let pd = plane_depth(ctx.half, center);
    if pd <= i64::from(super::super::cube::CRUST) || pd > i64::from(DEEP_HI) {
        return None;
    }
    let up = Up::of(center);
    let tangents = match up.axis {
        0 => [1usize, 2],
        1 => [0, 2],
        _ => [0, 1],
    };
    let long_axis = tangents[(h >> 8) as usize % 2];
    let length = 48 + i64::from((h >> 12) % 32);
    let floor_o = up.outward(center) - AIR_H / 2;
    Some(Hall { center, up, long_axis, length, floor_o })
}

fn across_axis(h: &Hall) -> usize {
    3 - h.up.axis - h.long_axis
}

fn extent(h: &Hall, axis: usize) -> i64 {
    if h.long_axis == axis { h.length / 2 } else { WIDTH / 2 }
}

fn aabb(h: &Hall) -> ([i64; 3], [i64; 3]) {
    let mut lo = h.center;
    let mut hi = h.center;
    let along = h.long_axis;
    let across = across_axis(h);
    lo[along] -= h.length / 2;
    hi[along] += h.length / 2;
    lo[across] -= WIDTH / 2;
    hi[across] += WIDTH / 2;
    let c0 = h.floor_o * i64::from(h.up.sign);
    let c1 = (h.floor_o + AIR_H) * i64::from(h.up.sign);
    lo[h.up.axis] = c0.min(c1);
    hi[h.up.axis] = c0.max(c1);
    (lo, hi)
}

fn inside(h: &Hall, rel: [i64; 3], m: &super::super::Materials) -> Option<BlockId> {
    let along = rel[h.long_axis] - h.center[h.long_axis];
    let across = rel[across_axis(h)] - h.center[across_axis(h)];
    let o = h.up.outward(rel);
    if along.abs() > h.length / 2 || across.abs() > WIDTH / 2 || o < h.floor_o || o > h.floor_o + AIR_H {
        return None;
    }
    if o == h.floor_o {
        return Some(if across == 0 { m.rail } else { m.marble });
    }
    // Stairs climb the low end of the long axis.
    let step = along + h.length / 2;
    if (0..AIR_H).contains(&step) && o <= h.floor_o + step {
        return Some(m.marble);
    }
    if along.rem_euclid(8) == 4 && across.abs() == WIDTH / 2 - 2 {
        return Some(if o == h.floor_o + AIR_H { m.lamp } else { m.marble });
    }
    Some(AIR)
}

fn tunnel(a: &Hall, b: &Hall, axis: usize, rel: [i64; 3], m: &super::super::Materials) -> Option<BlockId> {
    if a.up.axis != b.up.axis || a.up.sign != b.up.sign || axis == a.up.axis {
        return None;
    }
    let floor = a.floor_o.min(b.floor_o);
    let across = 3 - a.up.axis - axis;
    let mid = (a.center[across] + b.center[across]) / 2;
    let o = a.up.outward(rel);
    if o < floor || o > floor + 3 || (rel[across] - mid).abs() > 1 {
        return None;
    }
    let delta = b.center[axis] - a.center[axis];
    if delta == 0 {
        return None;
    }
    let a_face = a.center[axis] + delta.signum() * extent(a, axis);
    let b_face = b.center[axis] - delta.signum() * extent(b, axis);
    let (lo, hi) = (a_face.min(b_face), a_face.max(b_face));
    if rel[axis] <= lo || rel[axis] >= hi {
        return None;
    }
    Some(if o == floor {
        if rel[across] == mid { m.rail } else { m.marble }
    } else {
        AIR
    })
}

fn tunnels_from(ctx: &Ctx, idx: [i64; 3], rel: [i64; 3]) -> Option<BlockId> {
    let a = hall_at(ctx, idx)?;
    for axis in 0..3 {
        if axis == a.up.axis {
            continue;
        }
        for dir in [-1, 1] {
            let mut nb = idx;
            nb[axis] += dir;
            let Some(b) = hall_at(ctx, nb) else { continue };
            if let Some(id) = tunnel(&a, &b, axis, rel, ctx.m) {
                return Some(id);
            }
        }
    }
    None
}

fn tunnel_box(a: &Hall, b: &Hall, axis: usize) -> Option<([i64; 3], [i64; 3])> {
    if a.up.axis != b.up.axis || a.up.sign != b.up.sign || axis == a.up.axis {
        return None;
    }
    let floor = a.floor_o.min(b.floor_o);
    let across = 3 - a.up.axis - axis;
    let mid = (a.center[across] + b.center[across]) / 2;
    let delta = b.center[axis] - a.center[axis];
    if delta == 0 {
        return None;
    }
    let a_face = a.center[axis] + delta.signum() * extent(a, axis);
    let b_face = b.center[axis] - delta.signum() * extent(b, axis);
    let mut lo = [0i64; 3];
    let mut hi = [0i64; 3];
    lo[axis] = a_face.min(b_face);
    hi[axis] = a_face.max(b_face);
    lo[across] = mid - 1;
    hi[across] = mid + 1;
    let c0 = floor * i64::from(a.up.sign);
    let c1 = (floor + 3) * i64::from(a.up.sign);
    lo[a.up.axis] = c0.min(c1);
    hi[a.up.axis] = c0.max(c1);
    Some((lo, hi))
}

/// Mean air fraction a hall grid adds to the deep shell. Tunnels sit in the error bound.
pub(super) fn porosity(scale: f32) -> f64 {
    let s = f64::from(scale.clamp(0.0, 2.0));
    if s == 0.0 {
        return 0.0;
    }
    let length = 63.5;
    let air = length * WIDTH as f64 * AIR_H as f64;
    let cell = CELL as f64;
    air / (cell * cell * cell) * f64::from(P) * s
}

pub(super) fn hits(ctx: &Ctx, lo: [i64; 3], hi: [i64; 3]) -> bool {
    for_cells(lo, hi, CELL, |idx| {
        let Some(h) = hall_at(ctx, idx) else { return false };
        let (a, b) = aabb(&h);
        if box_hits(a, b, lo, hi) {
            return true;
        }
        for axis in 0..3 {
            if axis == h.up.axis {
                continue;
            }
            for dir in [-1, 1] {
                let mut nb = idx;
                nb[axis] += dir;
                let Some(other) = hall_at(ctx, nb) else { continue };
                if let Some((p, q)) = tunnel_box(&h, &other, axis)
                    && box_hits(p, q, lo, hi)
                {
                    return true;
                }
            }
        }
        false
    })
}

pub(super) fn block(ctx: &Ctx, rel: [i64; 3]) -> Option<BlockId> {
    let idx = std::array::from_fn(|a| rel[a].div_euclid(CELL));
    if let Some(h) = hall_at(ctx, idx)
        && let Some(id) = inside(&h, rel, ctx.m)
    {
        return Some(id);
    }
    tunnels_from(ctx, idx, rel)
}

#[cfg(test)]
pub(super) fn locate(ctx: &Ctx) -> Option<[i64; 3]> {
    let y0 = (ctx.half - i64::from(DEEP_HI)).div_euclid(CELL);
    let y1 = (ctx.half - i64::from(super::super::cube::CRUST)).div_euclid(CELL);
    for y in y0..=y1 {
        for z in -8..8 {
            for x in -8..8 {
                if let Some(h) = hall_at(ctx, [x, y, z]) {
                    return Some(h.center);
                }
            }
        }
    }
    None
}
