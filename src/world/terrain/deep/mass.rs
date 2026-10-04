//! The Heart and the mantle bubbles, as voxels and as the analytic masses gravity subtracts.
//!
//! Both are exact: a cube of half-size [`HEART`] at the centre, and rare balls confined to the
//! mantle. The Deep and the Underdark are not exact — their mean air fraction is a shell of
//! negative density, with an error bound for the caverns' scatter.

use glam::DVec3;

use super::super::noise::{hash3, unit};
use super::{Ctx, DEEP_HI, UNDER_HI, confine, dist2, field_at, for_cells, hash_rel, hash_site, roll, sphere_hits};
use crate::block::registry::{AIR, BlockId};
use crate::gravity::{Primitive, Shape as MassShape, Visitor};

/// Half-size of the hollow cube at a planet's centre.
pub(super) const HEART: i64 = 50_000;
const CORE: i64 = 6;
const SPIKE: i64 = 11;
const SHAFT_W: i64 = 4;

const C_CELL: i64 = 320;
const C_P: f32 = 0.28;
const C_R_LO: i64 = 6;
const C_R_HI: i64 = 18;
const C_SALT: u32 = 0xC1A5_7E21;

const B_CELL: i64 = 8_000_000;
const B_R_LO: i64 = 10_000;
const B_R_HI: i64 = 50_000;
const B_P: f32 = 0.09;
const B_SALT: u32 = 0xB0BB_1E00;
const CORE_P: f32 = 0.22;
const CORE_R: i64 = 14;

struct Cluster {
    center: [i64; 3],
    r: i64,
    salt: u32,
}

struct Bubble {
    center: [i64; 3],
    r: i64,
    core: bool,
    salt: u32,
}

fn linf(p: [i64; 3]) -> i64 {
    p.iter().copied().map(i64::abs).max().unwrap_or(0)
}

fn cluster_at(ctx: &Ctx, idx: [i64; 3]) -> Option<Cluster> {
    if ctx.scale <= 0.0 {
        return None;
    }
    let h = hash_site(ctx.seed ^ C_SALT, idx)?;
    if !roll(h, C_P * ctx.scale) {
        return None;
    }
    let origin: [i64; 3] = std::array::from_fn(|a| idx[a] * C_CELL);
    let r = C_R_LO + (unit(hash3(h, 1, 2, 3)) * (C_R_HI - C_R_LO) as f32) as i64;
    let r = r.clamp(C_R_LO, C_R_HI);
    let center = std::array::from_fn(|a| confine(origin[a], C_CELL, r, unit(hash3(h, a as i32, 4, 5))));
    // Clear of the core and of the heart's wall, so the cluster floats in the chamber.
    if linf(center) <= SPIKE + r || linf(center) + r >= HEART - 1 {
        return None;
    }
    Some(Cluster { center, r, salt: h })
}

fn in_mantle(half: i64, c: [i64; 3], r: i64) -> bool {
    let n = linf(c);
    n - r >= HEART && n + r <= half - i64::from(UNDER_HI)
}

fn bubble_at(seed: u32, half: i64, scale: f32, idx: [i64; 3]) -> Option<Bubble> {
    if scale <= 0.0 {
        return None;
    }
    let h = hash_site(seed ^ B_SALT, idx)?;
    if !roll(h, B_P * scale) {
        return None;
    }
    let origin: [i64; 3] = std::array::from_fn(|a| idx[a] * B_CELL);
    let mid = std::array::from_fn(|a| origin[a] + B_CELL / 2);
    let t = ((field_at(seed ^ B_SALT, mid) + 1.0) * 0.5).clamp(0.0, 1.0);
    let r = (B_R_LO + (t * (B_R_HI - B_R_LO) as f32) as i64).clamp(B_R_LO, B_R_HI);
    let center = std::array::from_fn(|a| confine(origin[a], B_CELL, r, unit(hash3(h, a as i32, 1, 2))));
    if !in_mantle(half, center, r) {
        return None;
    }
    let core = roll(hash3(h, 4, 5, 6), CORE_P);
    Some(Bubble { center, r, core, salt: h })
}

fn on_spike(rel: [i64; 3]) -> bool {
    let a = [rel[0].abs(), rel[1].abs(), rel[2].abs()];
    (a[0] <= 1 && a[1] <= 1) || (a[0] <= 1 && a[2] <= 1) || (a[1] <= 1 && a[2] <= 1)
}

fn in_heart_shaft(ctx: &Ctx, rel: [i64; 3]) -> bool {
    let gap = ctx.half - i64::from(UNDER_HI);
    if gap <= HEART {
        return false;
    }
    let a = [rel[0].abs(), rel[1].abs(), rel[2].abs()];
    let axis = if a[1] <= SHAFT_W && a[2] <= SHAFT_W {
        0
    } else if a[0] <= SHAFT_W && a[2] <= SHAFT_W {
        1
    } else if a[0] <= SHAFT_W && a[1] <= SHAFT_W {
        2
    } else {
        return false;
    };
    (HEART..gap).contains(&a[axis])
}

fn heart_open(lo: [i64; 3], hi: [i64; 3]) -> bool {
    let p = std::array::from_fn(|a| 0i64.clamp(lo[a], hi[a]));
    linf(p) < HEART
}

fn shaft_meets(ctx: &Ctx, lo: [i64; 3], hi: [i64; 3]) -> bool {
    let gap = ctx.half - i64::from(UNDER_HI);
    if gap <= HEART {
        return false;
    }
    for axis in 0..3 {
        let mut a0 = [-SHAFT_W, -SHAFT_W, -SHAFT_W];
        let mut a1 = [SHAFT_W, SHAFT_W, SHAFT_W];
        a0[axis] = HEART;
        a1[axis] = gap - 1;
        // Both directions of the axis (the box above is the positive ray; mirror it).
        if super::box_hits(a0, a1, lo, hi) {
            return true;
        }
        let b0 = std::array::from_fn(|i| -a1[i]);
        let b1 = std::array::from_fn(|i| -a0[i]);
        if super::box_hits(b0, b1, lo, hi) {
            return true;
        }
    }
    false
}

fn cluster_here(ctx: &Ctx, rel: [i64; 3]) -> Option<Cluster> {
    let idx = std::array::from_fn(|a| rel[a].div_euclid(C_CELL));
    // The ball stays in its cell, so the voxel's own cell is enough.
    let c = cluster_at(ctx, idx)?;
    (dist2(rel, c.center) <= c.r * c.r).then_some(c)
}

pub(super) fn heart_block(ctx: &Ctx, rel: [i64; 3]) -> Option<BlockId> {
    if linf(rel) < HEART {
        if linf(rel) <= CORE {
            return Some(ctx.m.core);
        }
        if linf(rel) <= SPIKE && on_spike(rel) {
            return Some(ctx.m.star);
        }
        if let Some(c) = cluster_here(ctx, rel) {
            return Some(match hash_rel(c.salt, rel) % 4 {
                0 => ctx.m.star,
                1 => ctx.m.glowshroom,
                _ => ctx.m.crystal,
            });
        }
        return Some(AIR);
    }
    in_heart_shaft(ctx, rel).then_some(AIR)
}

pub(super) fn heart_hits(ctx: &Ctx, lo: [i64; 3], hi: [i64; 3]) -> bool {
    heart_open(lo, hi) || shaft_meets(ctx, lo, hi)
}

pub(super) fn heart_all_air(ctx: &Ctx, lo: [i64; 3], hi: [i64; 3]) -> bool {
    if super::max_linf(lo, hi) >= HEART {
        return false;
    }
    let close = std::array::from_fn(|a| 0i64.clamp(lo[a], hi[a]));
    if linf(close) <= SPIKE {
        return false;
    }
    !for_cells(lo, hi, C_CELL, |idx| cluster_at(ctx, idx).is_some_and(|c| sphere_hits(c.center, c.r, lo, hi)))
}

fn bubble_here(ctx: &Ctx, rel: [i64; 3]) -> Option<Bubble> {
    let idx = std::array::from_fn(|a| rel[a].div_euclid(B_CELL));
    let b = bubble_at(ctx.seed, ctx.half, ctx.scale, idx)?;
    (dist2(rel, b.center) <= b.r * b.r).then_some(b)
}

pub(super) fn bubble_block(ctx: &Ctx, rel: [i64; 3]) -> Option<BlockId> {
    let b = bubble_here(ctx, rel)?;
    let d2 = dist2(rel, b.center);
    if b.core && d2 <= CORE_R * CORE_R {
        return Some(if d2 <= 16 {
            ctx.m.star
        } else if hash_rel(b.salt, rel) % 5 == 0 {
            ctx.m.glowshroom
        } else {
            ctx.m.crystal
        });
    }
    Some(AIR)
}

pub(super) fn bubble_hits(ctx: &Ctx, lo: [i64; 3], hi: [i64; 3]) -> bool {
    for_cells(lo, hi, B_CELL, |idx| {
        bubble_at(ctx.seed, ctx.half, ctx.scale, idx).is_some_and(|b| sphere_hits(b.center, b.r, lo, hi))
    })
}

pub(super) fn bubble_all_air(ctx: &Ctx, lo: [i64; 3], hi: [i64; 3]) -> bool {
    for_cells(lo, hi, B_CELL, |idx| {
        let Some(b) = bubble_at(ctx.seed, ctx.half, ctx.scale, idx) else { return false };
        let inside = (0..8).all(|i| {
            let p = [
                if i & 1 == 0 { lo[0] } else { hi[0] },
                if i & 2 == 0 { lo[1] } else { hi[1] },
                if i & 4 == 0 { lo[2] } else { hi[2] },
            ];
            dist2(p, b.center) <= b.r * b.r
        });
        inside && !(b.core && sphere_hits(b.center, CORE_R + 1, lo, hi))
    })
}

fn box_prim(centre: [i64; 3], half: f64, density: f64) -> Primitive {
    let c = DVec3::new(centre[0] as f64, centre[1] as f64, centre[2] as f64);
    let h = DVec3::splat(half);
    Primitive::new(MassShape::Box { lo: c - h, hi: c + h }, density)
}

fn ball_prim(centre: [i64; 3], rel: [i64; 3], r: f64, density: f64) -> Primitive {
    let c = DVec3::new((centre[0] + rel[0]) as f64, (centre[1] + rel[1]) as f64, (centre[2] + rel[2]) as f64);
    Primitive::new(MassShape::Ball { c, r }, density)
}

/// Negative-density shells, the Heart, and the bubbles. `density` is the bulk amount.
pub(in super::super) fn apply(centre: [i64; 3], half: i64, seed: u32, density: f64, scale: f32, v: &mut dyn Visitor) {
    let (phi_d, phi_u) = super::porosity(scale);
    if phi_d > 0.0 && half > i64::from(DEEP_HI) {
        let outer = (half - i64::from(super::super::cube::CRUST)) as f64;
        let inner = (half - i64::from(DEEP_HI)) as f64;
        v.primitive(&box_prim(centre, outer, -density * phi_d));
        v.primitive(&box_prim(centre, inner, density * phi_d));
        let thick = (DEEP_HI - super::super::cube::CRUST) as f64;
        v.error(2.0 * std::f64::consts::PI * density * phi_d * thick * 0.35);
        // Floor lights replace void this shell already removed. Bounded by one bulk-density cell
        // per light. Hall lamps replace marble inside that same air volume, so they stay in the
        // 0.35 scatter and do not add a term.
        let phi_l = super::cavern::light_phi(scale);
        if phi_l > 0.0 {
            v.error(2.0 * std::f64::consts::PI * density * phi_l * thick);
        }
    }
    if phi_u > 0.0 && half > i64::from(UNDER_HI) {
        let outer = (half - i64::from(DEEP_HI)) as f64;
        let inner = (half - i64::from(UNDER_HI)) as f64;
        v.primitive(&box_prim(centre, outer, -density * phi_u));
        v.primitive(&box_prim(centre, inner, density * phi_u));
        let thick = (UNDER_HI - DEEP_HI) as f64;
        v.error(2.0 * std::f64::consts::PI * density * phi_u * thick * 0.35);
        let phi_l = super::chamber::light_phi(scale);
        if phi_l > 0.0 {
            v.error(2.0 * std::f64::consts::PI * density * phi_l * thick);
        }
    }
    if half > HEART {
        v.primitive(&box_prim(centre, HEART as f64, -density));
    }
    let scale = scale.clamp(0.0, 2.0);
    if scale <= 0.0 {
        return;
    }
    // Sites live in body-relative cells. The ball stays inside its cell, so the cube's own cells suffice.
    let reach = half + B_R_HI;
    for_cells([-reach, -reach, -reach], [reach, reach, reach], B_CELL, |idx| {
        if let Some(b) = bubble_at(seed, half, scale, idx) {
            v.primitive(&ball_prim(centre, b.center, b.r as f64, -density));
            if b.core {
                v.primitive(&ball_prim(centre, b.center, CORE_R as f64, density));
            }
        }
        false
    });
}

#[cfg(test)]
pub(super) fn locate_bubble(ctx: &Ctx) -> Option<([i64; 3], i64)> {
    let s = |v: i64| v.div_euclid(B_CELL);
    for z in s(-ctx.half)..=s(ctx.half) {
        for y in s(-ctx.half)..=s(ctx.half) {
            for x in s(-ctx.half)..=s(ctx.half) {
                if let Some(b) = bubble_at(ctx.seed, ctx.half, ctx.scale, [x, y, z]) {
                    return Some((b.center, b.r));
                }
            }
        }
    }
    None
}
