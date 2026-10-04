//! Stone circles: a ring of standing stones around a flat altar.

use super::super::noise::hash2;
use super::super::Materials;
use super::{disk, foundation, keep, masonry, site_anchor, site_key, solid, Ctx, Stamp};
use crate::block::registry::BlockId;

pub(super) const CELL: i32 = 1792;
pub(super) const REACH: i32 = 16;
const SALT: u32 = 0xC1EC_0003;
const DENS: f32 = 0.15;

#[derive(Clone, Copy)]
pub(super) struct Spec {
    x: i32,
    z: i32,
    base: i32,
    r: i32,
    h: i32,
    gap: u8,
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
    let r = 8 + ((salt >> 2) % 3) as i32 * 2;
    let foot = r + 1;
    let samples = [
        (foot, 0),
        (-foot, 0),
        (0, foot),
        (0, -foot),
        (foot, foot),
        (foot, -foot),
        (-foot, foot),
        (-foot, -foot),
    ];
    if !ctx.flat(x, z, col.height, &samples) {
        return None;
    }
    Some(Spec {
        x,
        z,
        base: col.height,
        r,
        h: 3 + ((salt >> 8) % 3) as i32,
        gap: (salt % 8) as u8,
        stone: masonry(ctx.m, col.theme, salt),
        y0: col.height - 12,
        y1: col.height + 6,
    })
}

pub(super) fn bounds(s: &Spec) -> (i32, i32) {
    (s.y0, s.y1)
}

pub(super) fn anchor(s: &Spec) -> (i32, i32) {
    (s.x, s.z)
}

pub(super) fn covers(s: &Spec, x: i32, z: i32) -> bool {
    disk(x - s.x, z - s.z, s.r + 1)
}

fn stone_at(r: i32, i: i32) -> (i32, i32) {
    const RING: [(i32, i32); 8] = [(4, 0), (3, 3), (0, 4), (-3, 3), (-4, 0), (-3, -3), (0, -4), (3, -3)];
    let (bx, bz) = RING[i.rem_euclid(8) as usize];
    let scale = r / 4;
    let (mut x, mut z) = (bx * scale, bz * scale);
    if r == 10 {
        if bx.abs() == 4 {
            x = bx.signum() * 10;
        }
        if bz.abs() == 4 {
            z = bz.signum() * 10;
        }
    }
    (x, z)
}

pub(super) fn paint(s: &Spec, _m: &Materials, x: i32, y: i32, z: i32, ground: i32) -> Option<Stamp> {
    let dx = x - s.x;
    let dz = z - s.z;
    let dy = y - s.base;
    if dx.abs() > REACH || dz.abs() > REACH {
        return None;
    }
    for i in 0..8 {
        if i == i32::from(s.gap) {
            continue;
        }
        let (px, pz) = stone_at(s.r, i);
        if dx == px && dz == pz && (0..s.h).contains(&dy) {
            return Some(solid(s.stone, y, ground));
        }
    }
    if dx.abs() <= 1 && dz.abs() <= 1 && dy == 0 {
        return Some(solid(s.stone, y, ground));
    }
    foundation(s.base, ground, y, s.stone, disk(dx, dz, s.r + 1))
}

#[cfg(test)]
pub(super) fn describe(s: &Spec) -> super::Desc {
    super::Desc { x: s.x, z: s.z, pad: s.base, a: s.r, b: s.h, qu: i32::from(s.gap), qa: 0, qv: 0 }
}
