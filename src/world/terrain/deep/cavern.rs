//! The Deep: caverns a few hundred blocks across, one per cell of a coarse grid.
//!
//! The radius and the biome come from one coarse sample at the cell, not from a per-voxel field,
//! so the hollow is an exact ball. `confine` keeps that ball inside its cell, and a shaft is the
//! only part that leaves it. Dressing (mushrooms, crystals, a lake, roots, limestone, glow) stays
//! within [`MARGIN`] of the shell or of the geode's core, which is what `all_air` relies on.

use super::mushroom::{self, REACH};
use super::super::noise::hash2;
use super::{
    Ctx, DEEP_HI, Up, band_certain, confine, dist2, field_at, for_cells, for_shafts, hash_rel, hash_site, idx_i32,
    min_outward, place, plane_depth, roll, sphere_hits, tangent,
};
use crate::block::registry::{AIR, BlockId};

const CELL: i64 = 1_024;
const R_LO: i64 = 50;
const R_HI: i64 = 300;
const P: f32 = 0.42;
const SHAFT_P: f32 = 0.28;
const SALT: u32 = 0x0CA7_E401;
/// Dressing never reaches farther in than this, so a box inset by it is air.
const MARGIN: i64 = REACH + 4;
/// How many cells a shaft can climb from its cavern toward the crust.
const SHAFT_PAD: i64 = 6;
const FILL: f64 = 0.85;

struct Site {
    center: [i64; 3],
    r: i64,
    up: Up,
    kind: u32,
    shaft: bool,
    salt: u32,
}

fn may_host(ctx: &Ctx, idx: [i64; 3]) -> bool {
    let mid = std::array::from_fn(|a| idx[a] * CELL + CELL / 2);
    let pd = plane_depth(ctx.half, mid);
    let slack = CELL / 2;
    pd + slack > i64::from(super::super::cube::CRUST) && pd - slack <= i64::from(DEEP_HI)
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
    let center = std::array::from_fn(|a| confine(origin[a], CELL, r, super::super::noise::unit(hash3_axis(h, a))));
    let pd = plane_depth(ctx.half, center);
    if pd <= i64::from(super::super::cube::CRUST) || pd > i64::from(DEEP_HI) {
        return None;
    }
    let up = Up::of(center);
    let kind = hash3_axis(h, 3).wrapping_add((pd / 600) as u32) % 6;
    let shaft = roll(hash3_axis(h, 6), SHAFT_P);
    Some(Site { center, r, up, kind, shaft, salt: h })
}

fn hash3_axis(h: u32, a: usize) -> u32 {
    super::super::noise::hash3(h, a as i32, 1, 2)
}

fn contains(s: &Site, p: [i64; 3]) -> bool {
    dist2(p, s.center) <= s.r * s.r
}

fn core_r(s: &Site) -> i64 {
    12 + i64::from((s.salt >> 16) % 5)
}

fn lake_top(s: &Site) -> i64 {
    s.up.outward(s.center) - (s.r * 2) / 5
}

fn column_half(s: &Site, t0: i64, t1: i64) -> Option<i64> {
    let (c0, c1) = tangent(s.center, s.up.axis);
    let (d0, d1) = (t0 - c0, t1 - c1);
    let h2 = s.r * s.r - d0 * d0 - d1 * d1;
    (h2 >= 0).then_some((h2 as f64).sqrt().floor() as i64)
}

fn add_out(up: Up, rel: [i64; 3], steps: i64) -> [i64; 3] {
    let (t0, t1) = tangent(rel, up.axis);
    place(up.axis, up.sign, (t0, t1), up.outward(rel) + steps)
}

/// Mean air fraction of the deep shell at this `deep` scale.
pub(super) fn porosity(scale: f32) -> f64 {
    let s = f64::from(scale.clamp(0.0, 2.0));
    if s == 0.0 {
        return 0.0;
    }
    let sum = |n: i64| {
        if n <= 0 {
            0.0
        } else {
            let t = n as f64 * (n + 1) as f64 * 0.5;
            t * t
        }
    };
    let mean = (sum(R_HI) - sum(R_LO - 1)) / (R_HI - R_LO + 1) as f64;
    let vol = 4.0 / 3.0 * std::f64::consts::PI * mean * FILL;
    let cell = CELL as f64;
    vol / (cell * cell * cell) * f64::from(P) * s
}

pub(super) fn hits(ctx: &Ctx, lo: [i64; 3], hi: [i64; 3]) -> bool {
    for_cells(lo, hi, CELL, |idx| site(ctx, idx).is_some_and(|s| sphere_hits(s.center, s.r, lo, hi)))
        || for_shafts(lo, hi, CELL, SHAFT_PAD, |idx| site(ctx, idx).is_some_and(|s| shaft_meets(&s, ctx, lo, hi)))
}

pub(super) fn all_air(ctx: &Ctx, lo: [i64; 3], hi: [i64; 3]) -> bool {
    if !band_certain(ctx.half, lo, hi, super::super::cube::CRUST, DEEP_HI) {
        return false;
    }
    for_cells(lo, hi, CELL, |idx| {
        let Some(s) = site(ctx, idx) else { return false };
        box_inside(&s, lo, hi, MARGIN) && clear(&s, lo, hi)
    })
}

fn box_inside(s: &Site, lo: [i64; 3], hi: [i64; 3], margin: i64) -> bool {
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

fn clear(s: &Site, lo: [i64; 3], hi: [i64; 3]) -> bool {
    match s.kind {
        1 => !sphere_hits(s.center, core_r(s) + 1, lo, hi),
        2 => min_outward(s.up, lo, hi) > lake_top(s) + 6,
        _ => true,
    }
}

pub(super) fn block(ctx: &Ctx, rel: [i64; 3]) -> Option<BlockId> {
    let idx = std::array::from_fn(|a| rel[a].div_euclid(CELL));
    if let Some(s) = site(ctx, idx)
        && contains(&s, rel)
    {
        return Some(dress(ctx, &s, rel));
    }
    in_shaft(ctx, rel).then_some(AIR)
}

fn in_shaft(ctx: &Ctx, rel: [i64; 3]) -> bool {
    for_shafts(rel, rel, CELL, SHAFT_PAD, |idx| site(ctx, idx).is_some_and(|s| point_on_shaft(&s, ctx, rel)))
}

fn point_on_shaft(s: &Site, ctx: &Ctx, rel: [i64; 3]) -> bool {
    if !s.shaft || contains(s, rel) {
        return false;
    }
    let (t0, t1) = tangent(rel, s.up.axis);
    let (c0, c1) = tangent(s.center, s.up.axis);
    if (t0 - c0).abs() > 1 || (t1 - c1).abs() > 1 {
        return false;
    }
    let o = s.up.outward(rel);
    let co = s.up.outward(s.center);
    // From just above the cavern up to where the crust begins (the caller is already in the Deep).
    o > co && o <= ctx.half - i64::from(super::super::cube::CRUST)
}

fn shaft_meets(s: &Site, ctx: &Ctx, lo: [i64; 3], hi: [i64; 3]) -> bool {
    if !s.shaft {
        return false;
    }
    let (c0, c1) = tangent(s.center, s.up.axis);
    let co = s.up.outward(s.center);
    let cap = ctx.half - i64::from(super::super::cube::CRUST);
    let a = place(s.up.axis, s.up.sign, (c0 - 1, c1 - 1), co);
    let b = place(s.up.axis, s.up.sign, (c0 + 1, c1 + 1), cap.max(co));
    let lo_s = std::array::from_fn(|i| a[i].min(b[i]));
    let hi_s = std::array::from_fn(|i| a[i].max(b[i]));
    super::box_hits(lo_s, hi_s, lo, hi)
}

fn dress(ctx: &Ctx, s: &Site, rel: [i64; 3]) -> BlockId {
    let m = ctx.m;
    match s.kind {
        0 => fungal(ctx, s, rel).unwrap_or(AIR),
        1 => geode(s, rel, m),
        2 => magma(s, rel, m),
        3 => hung(s, rel, m).unwrap_or(AIR),
        4 => cones(s, rel, m).unwrap_or(AIR),
        _ => glow(s, rel, m).unwrap_or(AIR),
    }
}

fn fungal(ctx: &Ctx, s: &Site, rel: [i64; 3]) -> Option<BlockId> {
    let d2 = dist2(rel, s.center);
    if mushroom::out_of_reach(s.r, d2) {
        return None;
    }
    let shell = |t0: i64, t1: i64| column_half(s, t0, t1);
    let co = s.up.outward(s.center);
    mushroom::occupy(ctx.seed ^ s.salt, ctx.scale.max(0.25), rel, s.up, ctx.m, |t0, t1| {
        shell(t0, t1).map(|h| co - h)
    }, |t0, t1| shell(t0, t1).map(|h| co + h))
}

fn geode(s: &Site, rel: [i64; 3], m: &super::super::Materials) -> BlockId {
    let d2 = dist2(rel, s.center);
    let cr = core_r(s);
    if d2 <= cr * cr {
        return match hash_rel(s.salt ^ 0x6E0D, rel) % 5 {
            0 => m.star,
            1 => m.glowshroom,
            _ => m.crystal,
        };
    }
    let lining = s.r - 4;
    if lining > 0 && d2 > lining * lining {
        return if super::super::noise::unit(hash_rel(s.salt, rel)) < 0.1 { m.glowshroom } else { m.crystal };
    }
    AIR
}

fn magma(s: &Site, rel: [i64; 3], m: &super::super::Materials) -> BlockId {
    let o = s.up.outward(rel);
    let top = lake_top(s);
    let lining = s.r - 5;
    let near_wall = lining > 0 && dist2(rel, s.center) > lining * lining;
    if o < top {
        if super::super::noise::unit(hash_rel(s.salt, rel)) < 0.07 { m.obsidian } else { m.magma }
    } else if o == top {
        m.obsidian
    } else if o <= top + 5 && near_wall {
        m.basalt
    } else {
        AIR
    }
}

/// Roots hanging from the ceiling: darkwood inside, bark on the outside.
fn hung(s: &Site, rel: [i64; 3], m: &super::super::Materials) -> Option<BlockId> {
    const G: i64 = 12;
    let (t0, t1) = tangent(rel, s.up.axis);
    let o = s.up.outward(rel);
    let (i0, i1) = (t0.div_euclid(G), t1.div_euclid(G));
    let co = s.up.outward(s.center);
    for d1 in -1..=1 {
        for d0 in -1..=1 {
            let (Some(a0), Some(a1)) = (idx_i32(i0 + d0), idx_i32(i1 + d1)) else { continue };
            let h = hash2(s.salt ^ 0x4007, a0, a1);
            if super::super::noise::unit(h) > 0.55 {
                continue;
            }
            let p0 = (i0 + d0) * G + G / 2 + i64::from((h >> 8) % 5) - 2;
            let p1 = (i1 + d1) * G + G / 2 + i64::from((h >> 12) % 5) - 2;
            let Some(half) = column_half(s, p0, p1) else { continue };
            let len = 8 + i64::from((h >> 16) % 17);
            let drop = (co + half) - o;
            if drop < 0 || drop > len {
                continue;
            }
            let (dx, dz) = (t0 - p0, t1 - p1);
            let horiz = dx * dx + dz * dz;
            let rad: i64 = if drop < 2 { 2 } else { 1 };
            if horiz > rad * rad {
                continue;
            }
            return Some(if horiz <= 1 { m.darkwood } else { m.bark });
        }
    }
    None
}

fn cones(s: &Site, rel: [i64; 3], m: &super::super::Materials) -> Option<BlockId> {
    const G: i64 = 8;
    let (t0, t1) = tangent(rel, s.up.axis);
    let o = s.up.outward(rel);
    let (i0, i1) = (t0.div_euclid(G), t1.div_euclid(G));
    let co = s.up.outward(s.center);
    for d1 in -1..=1 {
        for d0 in -1..=1 {
            let (Some(a0), Some(a1)) = (idx_i32(i0 + d0), idx_i32(i1 + d1)) else { continue };
            let h = hash2(s.salt ^ 0x57A1, a0, a1);
            if super::super::noise::unit(h) > 0.7 {
                continue;
            }
            let p0 = (i0 + d0) * G + G / 2;
            let p1 = (i1 + d1) * G + G / 2;
            let Some(half) = column_half(s, p0, p1) else { continue };
            let len = 6 + i64::from((h >> 16) % 11);
            let (dx, dz) = (t0 - p0, t1 - p1);
            let horiz = dx * dx + dz * dz;
            let cover = |gap: i64| {
                if !(0..len).contains(&gap) {
                    return false;
                }
                let rad = 1 + (len - gap) * 2 / len;
                horiz <= rad * rad
            };
            if cover((co + half) - o) || cover(o - (co - half)) {
                return Some(m.limestone);
            }
        }
    }
    None
}

fn glow(s: &Site, rel: [i64; 3], m: &super::super::Materials) -> Option<BlockId> {
    if contains(s, add_out(s.up, rel, 1)) {
        return None;
    }
    (super::super::noise::unit(hash_rel(s.salt ^ 0x6100, rel)) < 0.12).then_some(m.glowcap)
}

#[cfg(test)]
pub(super) fn locate(ctx: &Ctx) -> Option<[i64; 3]> {
    let y0 = (ctx.half - i64::from(DEEP_HI)).div_euclid(CELL);
    let y1 = (ctx.half - i64::from(super::super::cube::CRUST)).div_euclid(CELL);
    for y in y0..=y1 {
        for z in -16..16 {
            for x in -16..16 {
                if let Some(s) = site(ctx, [x, y, z]) {
                    return Some(s.center);
                }
            }
        }
    }
    None
}
