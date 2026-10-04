//! Mine heads: a plank headframe over a shaft. The shaft meets a mine level when one is below.

use super::super::noise::hash2;
use super::super::Materials;
use super::{carved, foundation, keep, site_anchor, site_key, solid, Ctx, Stamp};

pub(super) const CELL: i32 = 2432;
pub(super) const REACH: i32 = 6;
const SALT: u32 = 0xA1E0_0007;
const DENS: f32 = 0.16;

#[derive(Clone, Copy)]
pub(super) struct Spec {
    x: i32,
    z: i32,
    base: i32,
    bottom: i32,
    y0: i32,
    y1: i32,
}

pub(super) fn spawn(ctx: &Ctx, cx: i32, cz: i32) -> Option<Spec> {
    super::cache!(Spec, site_key(ctx, SALT), cx, cz, build(ctx, cx, cz))
}

fn build(ctx: &Ctx, cx: i32, cz: i32) -> Option<Spec> {
    if !keep(ctx.scale, DENS, hash2(ctx.s ^ SALT, cx, cz)) {
        return None;
    }
    let (x, z, _salt) = site_anchor(ctx.s ^ SALT, CELL, cx, cz);
    if !ctx.inland(x, z, REACH) {
        return None;
    }
    let col = ctx.column(x, z);
    if col.slope4 >= 8 {
        return None;
    }
    let samples = [(2, 2), (2, -2), (-2, 2), (-2, -2), (2, 0), (-2, 0), (0, 2), (0, -2)];
    if !ctx.flat(x, z, col.height, &samples) {
        return None;
    }
    let mut bottom = col.height - 6;
    if let Some(level) = ctx.mine_floor(x, z, col.height) {
        if level < bottom {
            bottom = level;
        }
    }
    Some(Spec { x, z, base: col.height, bottom, y0: bottom, y1: col.height + 14 })
}

pub(super) fn bounds(s: &Spec) -> (i32, i32) {
    (s.y0, s.y1)
}

pub(super) fn anchor(s: &Spec) -> (i32, i32) {
    (s.x, s.z)
}

pub(super) fn covers(s: &Spec, x: i32, z: i32) -> bool {
    (x - s.x).abs() <= 2 && (z - s.z).abs() <= 2
}

pub(super) fn paint(s: &Spec, m: &Materials, x: i32, y: i32, z: i32, ground: i32) -> Option<Stamp> {
    let dx = x - s.x;
    let dz = z - s.z;
    if dx.abs() > REACH || dz.abs() > REACH {
        return None;
    }
    let shaft = dx.abs() <= 1 && dz.abs() <= 1;
    if shaft && (s.bottom..=s.base).contains(&y) {
        return Some(carved());
    }
    let ring = dx.abs() <= 2 && dz.abs() <= 2 && (dx.abs() == 2 || dz.abs() == 2);
    let corner = dx.abs() == 2 && dz.abs() == 2;
    if corner && (s.base..=s.base + 10).contains(&y) {
        return Some(solid(m.plank, y, ground));
    }
    if ring && (y == s.base + 5 || y == s.base + 10 || (s.base - 2..=s.base).contains(&y)) {
        return Some(solid(m.plank, y, ground));
    }
    if y == s.base + 12 {
        let d2 = dx * dx + dz * dz;
        if (1..=4).contains(&d2) {
            return Some(solid(m.plank, y, ground));
        }
    }
    if dx.abs() <= 2 && dz.abs() <= 2 && !shaft {
        return foundation(s.base, ground, y, m.plank, true);
    }
    None
}

#[cfg(test)]
pub(super) fn describe(s: &Spec) -> super::Desc {
    super::Desc { x: s.x, z: s.z, pad: s.base, a: s.bottom, b: 0, qu: 0, qa: 0, qv: 0 }
}
