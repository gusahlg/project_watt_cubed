//! Step pyramids: terraces, a stair up one side, and a lamp on the chamber floor.

use super::super::noise::hash2;
use super::super::Materials;
use super::{carved, foundation, keep, masonry, site_anchor, site_key, solid, Ctx, Stamp};
use crate::block::registry::BlockId;

pub(super) const CELL: i32 = 2304;
pub(super) const REACH: i32 = 22;
const SALT: u32 = 0xB1A4_0002;
const DENS: f32 = 0.15;

#[derive(Clone, Copy)]
pub(super) struct Spec {
    x: i32,
    z: i32,
    base: i32,
    n: i32,
    stone: BlockId,
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
    let (x, z, salt) = site_anchor(ctx.s ^ SALT, CELL, cx, cz);
    if !ctx.inland(x, z, REACH) {
        return None;
    }
    let col = ctx.column(x, z);
    if col.slope4 >= 8 {
        return None;
    }
    let n = 4 + (salt % 6) as i32;
    let foot = n * 2;
    let samples = [
        (foot, foot),
        (foot, -foot),
        (-foot, foot),
        (-foot, -foot),
        (foot, 0),
        (-foot, 0),
        (0, foot),
        (0, -foot),
    ];
    if !ctx.flat(x, z, col.height, &samples) {
        return None;
    }
    Some(Spec {
        x,
        z,
        base: col.height,
        n,
        stone: masonry(ctx.m, col.theme, salt),
        y0: col.height - 12,
        y1: col.height + n * 2 + 1,
    })
}

pub(super) fn bounds(s: &Spec) -> (i32, i32) {
    (s.y0, s.y1)
}

pub(super) fn anchor(s: &Spec) -> (i32, i32) {
    (s.x, s.z)
}

pub(super) fn covers(s: &Spec, x: i32, z: i32) -> bool {
    let foot = s.n * 2;
    (x - s.x).abs() <= foot && (z - s.z).abs() <= foot
}

pub(super) fn paint(s: &Spec, m: &Materials, x: i32, y: i32, z: i32, ground: i32) -> Option<Stamp> {
    let dx = x - s.x;
    let dz = z - s.z;
    let dy = y - s.base;
    let n = s.n;
    let base = n * 2;
    if dx.abs() > base + 1 || dz.abs() > base + 1 {
        return None;
    }
    if dz == 1 && (0..base).contains(&dy) && dx == base - dy {
        return Some(solid(s.stone, y, ground));
    }
    if dz == 0 && (1..=2).contains(&dy) && dx > 1 {
        let half = (n - dy / 2) * 2;
        if dx <= half {
            return Some(carved());
        }
    }
    if dx.abs() <= 1 && dz.abs() <= 1 && (1..=3).contains(&dy) {
        if dx == 0 && dz == 0 && dy == 1 {
            return Some(solid(m.lamp, y, ground));
        }
        return Some(carved());
    }
    if (0..base).contains(&dy) {
        let half = (n - dy / 2) * 2;
        if dx.abs() <= half && dz.abs() <= half {
            return Some(solid(s.stone, y, ground));
        }
    }
    foundation(s.base, ground, y, s.stone, dx.abs() <= base && dz.abs() <= base)
}

#[cfg(test)]
pub(super) fn describe(s: &Spec) -> super::Desc {
    super::Desc { x: s.x, z: s.z, pad: s.base, a: s.n, b: 0, qu: 0, qa: 0, qv: 0 }
}
