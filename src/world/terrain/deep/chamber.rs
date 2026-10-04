//! The Underdark: sparse chambers a few kilometres across, each with a floor, and a shaft up into
//! the Deep. The hollow is an exact ball (one coarse radius per cell). The landscape is the floor
//! field, mushrooms, crystal spires and lanterns; those stay near the floor or the ceiling, which
//! is what `all_air` relies on.

use super::mushroom;
use super::super::noise::{hash2, hash3, perlin2, unit};
use super::{
    Ctx, DEEP_HI, UNDER_HI, Up, band_certain, box_hits, confine, dist2, field_at, for_cells, for_shafts, hash_rel,
    hash_site, idx_i32, min_outward, plane_depth, roll, sphere_hits, tangent,
};
use crate::block::registry::{AIR, BlockId};

const CELL: i64 = 16_384;
const R_LO: i64 = 500;
const R_HI: i64 = 2_500;
const P: f32 = 0.12;
const SALT: u32 = 0x0D0A_4A01;
const SHAFT_W: i64 = 3;
const STUB: i64 = 64;
/// Cells a shaft can climb. 4 × cell covers the underdark's thickness.
const SHAFT_PAD: i64 = 4;
const FILL: f64 = 0.60;
/// Hills (`perlin2` can exceed 1) plus the tallest spire. A box above this, under the ceiling margin, is air.
const FLOOR_CLEAR: i64 = 120 + 90;
const CEIL_MARGIN: i64 = 26;

struct Site {
    center: [i64; 3],
    r: i64,
    up: Up,
    salt: u32,
}

fn may_host(ctx: &Ctx, idx: [i64; 3]) -> bool {
    let mid = std::array::from_fn(|a| idx[a] * CELL + CELL / 2);
    let pd = plane_depth(ctx.half, mid);
    let slack = CELL / 2;
    pd + slack > i64::from(DEEP_HI) && pd - slack <= i64::from(UNDER_HI)
}

fn site(ctx: &Ctx, idx: [i64; 3]) -> Option<Site> {
    if ctx.scale <= 0.0 || !may_host(ctx, idx) {
        return None;
    }
    let h = hash_site(ctx.seed ^ SALT, idx)?;
    if !roll(h, P * ctx.scale) {
        return None;
    }
    let origin: [i64; 3] = std::array::from_fn(|a| idx[a] * CELL);
    let mid = std::array::from_fn(|a| origin[a] + CELL / 2);
    let t = ((field_at(ctx.seed ^ SALT, mid) + 1.0) * 0.5).clamp(0.0, 1.0);
    let r = (R_LO + (t * (R_HI - R_LO) as f32) as i64).clamp(R_LO, R_HI);
    let center = std::array::from_fn(|a| confine(origin[a], CELL, r, unit(hash3(h, a as i32, 1, 2))));
    let pd = plane_depth(ctx.half, center);
    if pd <= i64::from(DEEP_HI) || pd > i64::from(UNDER_HI) {
        return None;
    }
    Some(Site { center, r, up: Up::of(center), salt: h })
}

fn contains(s: &Site, p: [i64; 3]) -> bool {
    dist2(p, s.center) <= s.r * s.r
}

fn column_half(s: &Site, t0: i64, t1: i64) -> Option<i64> {
    let (c0, c1) = tangent(s.center, s.up.axis);
    let (d0, d1) = (t0 - c0, t1 - c1);
    let h2 = s.r * s.r - d0 * d0 - d1 * d1;
    (h2 >= 0).then_some((h2 as f64).sqrt().floor() as i64)
}

fn floor_of(s: &Site, t0: i64, t1: i64) -> Option<i64> {
    let half = column_half(s, t0, t1)?;
    let co = s.up.outward(s.center);
    let hill = (perlin2(s.salt, t0 as f64 / 180.0, t1 as f64 / 180.0) * 40.0).floor() as i64;
    Some((co - s.r / 5 + hill).max(co - half))
}

fn ceil_of(s: &Site, t0: i64, t1: i64) -> Option<i64> {
    let half = column_half(s, t0, t1)?;
    Some(s.up.outward(s.center) + half)
}

/// Highest outward the shaft reaches: the sphere's top, or a stub into the Deep, whichever is higher.
fn shaft_top(ctx: &Ctx, s: &Site) -> i64 {
    let co = s.up.outward(s.center);
    (co + s.r).max(ctx.half - i64::from(DEEP_HI) + STUB)
}

fn on_shaft(ctx: &Ctx, s: &Site, rel: [i64; 3]) -> bool {
    let (t0, t1) = tangent(rel, s.up.axis);
    let (c0, c1) = tangent(s.center, s.up.axis);
    if (t0 - c0).abs() > SHAFT_W || (t1 - c1).abs() > SHAFT_W {
        return false;
    }
    let o = s.up.outward(rel);
    let co = s.up.outward(s.center);
    o >= co - s.r && o <= shaft_top(ctx, s)
}

fn shaft_aabb(ctx: &Ctx, s: &Site) -> ([i64; 3], [i64; 3]) {
    let (c0, c1) = tangent(s.center, s.up.axis);
    let co = s.up.outward(s.center);
    let foot = co - s.r;
    let top = shaft_top(ctx, s);
    let sign = i64::from(s.up.sign);
    let mut lo = s.center;
    let mut hi = s.center;
    let (a, b) = match s.up.axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    lo[a] = c0 - SHAFT_W;
    hi[a] = c0 + SHAFT_W;
    lo[b] = c1 - SHAFT_W;
    hi[b] = c1 + SHAFT_W;
    let c_foot = foot * sign;
    let c_top = top * sign;
    lo[s.up.axis] = c_foot.min(c_top);
    hi[s.up.axis] = c_foot.max(c_top);
    (lo, hi)
}

fn sum_cubes(n: i64) -> f64 {
    if n <= 0 {
        0.0
    } else {
        let t = n as f64 * (n + 1) as f64 * 0.5;
        t * t
    }
}

pub(super) fn porosity(scale: f32) -> f64 {
    let s = f64::from(scale.clamp(0.0, 2.0));
    if s == 0.0 {
        return 0.0;
    }
    let mean = (sum_cubes(R_HI) - sum_cubes(R_LO - 1)) / (R_HI - R_LO + 1) as f64;
    let vol = 4.0 / 3.0 * std::f64::consts::PI * mean * FILL;
    let cell = CELL as f64;
    vol / (cell * cell * cell) * f64::from(P) * s
}

pub(super) fn hits(ctx: &Ctx, lo: [i64; 3], hi: [i64; 3]) -> bool {
    for_cells(lo, hi, CELL, |idx| site(ctx, idx).is_some_and(|s| sphere_hits(s.center, s.r, lo, hi)))
        || for_shafts(lo, hi, CELL, SHAFT_PAD, |idx| {
            site(ctx, idx).is_some_and(|s| {
                let (a, b) = shaft_aabb(ctx, &s);
                box_hits(a, b, lo, hi)
            })
        })
}

pub(super) fn shaft_hits(ctx: &Ctx, lo: [i64; 3], hi: [i64; 3]) -> bool {
    for_shafts(lo, hi, CELL, SHAFT_PAD, |idx| {
        site(ctx, idx).is_some_and(|s| {
            let (a, b) = shaft_aabb(ctx, &s);
            box_hits(a, b, lo, hi)
        })
    })
}

pub(super) fn all_air(ctx: &Ctx, lo: [i64; 3], hi: [i64; 3]) -> bool {
    if !band_certain(ctx.half, lo, hi, DEEP_HI, UNDER_HI) {
        return false;
    }
    for_cells(lo, hi, CELL, |idx| {
        let Some(s) = site(ctx, idx) else { return false };
        if !corners_inside(&s, lo, hi, CEIL_MARGIN) {
            return false;
        }
        // Above the hills and the spires, and not on the shaft (the shaft is near the centre column).
        let co = s.up.outward(s.center);
        min_outward(s.up, lo, hi) > co - s.r / 5 + FLOOR_CLEAR && !shaft_box_hits(ctx, &s, lo, hi)
    })
}

fn corners_inside(s: &Site, lo: [i64; 3], hi: [i64; 3], margin: i64) -> bool {
    if s.r <= margin {
        return false;
    }
    let rr = (s.r - margin) * (s.r - margin);
    (0..8).all(|i| {
        let p = [
            if i & 1 == 0 { lo[0] } else { hi[0] },
            if i & 2 == 0 { lo[1] } else { hi[1] },
            if i & 4 == 0 { lo[2] } else { hi[2] },
        ];
        dist2(p, s.center) <= rr
    })
}

fn shaft_box_hits(ctx: &Ctx, s: &Site, lo: [i64; 3], hi: [i64; 3]) -> bool {
    let (a, b) = shaft_aabb(ctx, s);
    box_hits(a, b, lo, hi)
}

pub(super) fn block(ctx: &Ctx, rel: [i64; 3]) -> Option<BlockId> {
    // The ball never leaves its cell. The shaft does, so a miss still checks the column.
    let idx = std::array::from_fn(|a| rel[a].div_euclid(CELL));
    if let Some(s) = site(ctx, idx)
        && contains(&s, rel)
    {
        return Some(interior(ctx, &s, rel));
    }
    on_any_shaft(ctx, rel).then_some(AIR)
}

pub(super) fn shaft_block(ctx: &Ctx, rel: [i64; 3]) -> Option<BlockId> {
    on_any_shaft(ctx, rel).then_some(AIR)
}

fn on_any_shaft(ctx: &Ctx, rel: [i64; 3]) -> bool {
    for_shafts(rel, rel, CELL, SHAFT_PAD, |idx| site(ctx, idx).is_some_and(|s| on_shaft(ctx, &s, rel)))
}

fn interior(ctx: &Ctx, s: &Site, rel: [i64; 3]) -> BlockId {
    if on_shaft(ctx, s, rel) {
        return AIR;
    }
    let (t0, t1) = tangent(rel, s.up.axis);
    let o = s.up.outward(rel);
    let Some(floor) = floor_of(s, t0, t1) else { return AIR };
    let m = ctx.m;
    if o < floor {
        let h = hash_rel(s.salt, rel);
        return if h % 3 == 0 { m.limestone } else { m.rock[(h as usize) % 4] };
    }
    if o == floor {
        return match hash_rel(s.salt ^ 0xF100, rel) % 5 {
            0 => m.moss,
            1 => m.lichen,
            _ => m.soil,
        };
    }
    if let Some(id) = spire(s, rel, m) {
        return id;
    }
    if let Some(id) = lantern(s, rel, m) {
        return id;
    }
    let d2 = dist2(rel, s.center);
    if !mushroom::out_of_reach(s.r, d2)
        && let Some(id) = mushroom::occupy(
            ctx.seed ^ s.salt,
            ctx.scale.max(0.25),
            rel,
            s.up,
            m,
            |a, b| floor_of(s, a, b),
            |a, b| ceil_of(s, a, b),
        )
    {
        return id;
    }
    AIR
}

fn spire(s: &Site, rel: [i64; 3], m: &super::super::Materials) -> Option<BlockId> {
    const G: i64 = 48;
    let (t0, t1) = tangent(rel, s.up.axis);
    let o = s.up.outward(rel);
    let (i0, i1) = (t0.div_euclid(G), t1.div_euclid(G));
    for d1 in -1..=1 {
        for d0 in -1..=1 {
            let (Some(a0), Some(a1)) = (idx_i32(i0 + d0), idx_i32(i1 + d1)) else { continue };
            let h = hash2(s.salt ^ 0x5912, a0, a1);
            if unit(h) > 0.22 {
                continue;
            }
            let p0 = (i0 + d0) * G + G / 2;
            let p1 = (i1 + d1) * G + G / 2;
            let Some(base) = floor_of(s, p0, p1) else { continue };
            let height = 30 + i64::from((h >> 16) % 61);
            let rise = o - base;
            if rise <= 0 || rise > height {
                continue;
            }
            let (dx, dz) = (t0 - p0, t1 - p1);
            if dx * dx + dz * dz > 4 {
                continue;
            }
            return Some(if rise == height { m.star } else { m.crystal });
        }
    }
    None
}

fn lantern(s: &Site, rel: [i64; 3], m: &super::super::Materials) -> Option<BlockId> {
    const G: i64 = 600;
    let (t0, t1) = tangent(rel, s.up.axis);
    let o = s.up.outward(rel);
    let (i0, i1) = (t0.div_euclid(G), t1.div_euclid(G));
    for d1 in -1..=1 {
        for d0 in -1..=1 {
            let (Some(a0), Some(a1)) = (idx_i32(i0 + d0), idx_i32(i1 + d1)) else { continue };
            let h = hash2(s.salt ^ 0x1A47, a0, a1);
            if unit(h) > 0.35 {
                continue;
            }
            let p0 = (i0 + d0) * G + G / 2;
            let p1 = (i1 + d1) * G + G / 2;
            let Some(ceil) = ceil_of(s, p0, p1) else { continue };
            let drop = ceil - o;
            if drop < 0 || drop > 24 {
                continue;
            }
            let (dx, dz) = (t0 - p0, t1 - p1);
            let horiz = dx * dx + dz * dz;
            if horiz > 16 {
                continue;
            }
            return Some(if horiz <= 4 { m.star } else { m.crystal });
        }
    }
    None
}

#[cfg(test)]
pub(super) fn locate(ctx: &Ctx) -> Option<[i64; 3]> {
    let y0 = (ctx.half - i64::from(UNDER_HI)).div_euclid(CELL);
    let y1 = (ctx.half - i64::from(DEEP_HI)).div_euclid(CELL);
    for y in y0..=y1 {
        for z in -6..6 {
            for x in -6..6 {
                if let Some(s) = site(ctx, [x, y, z]) {
                    return Some(s.center);
                }
            }
        }
    }
    None
}
