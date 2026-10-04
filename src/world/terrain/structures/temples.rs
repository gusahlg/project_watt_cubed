//! Temple courts: a walled yard, pillars, a raised dais, and a broken roof.

use super::super::noise::hash2;
use super::super::Materials;
use super::{carved, foundation, keep, masonry, site_anchor, site_key, solid, Ctx, Stamp};
use crate::block::registry::BlockId;

pub(super) const CELL: i32 = 1920;
pub(super) const REACH: i32 = 12;
const SALT: u32 = 0x7EAB_0005;
const DENS: f32 = 0.15;

#[derive(Clone, Copy)]
pub(super) struct Spec {
    x: i32,
    z: i32,
    base: i32,
    half: i32,
    wall: i32,
    stone: BlockId,
    salt: u32,
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
    let half = 6 + (salt % 5) as i32;
    let samples = [
        (half, half),
        (half, -half),
        (-half, half),
        (-half, -half),
        (half, 0),
        (-half, 0),
        (0, half),
        (0, -half),
    ];
    if !ctx.flat(x, z, col.height, &samples) {
        return None;
    }
    let wall = 5 + ((salt >> 4) % 3) as i32;
    Some(Spec {
        x,
        z,
        base: col.height,
        half,
        wall,
        stone: masonry(ctx.m, col.theme, salt),
        salt,
        y0: col.height - 12,
        y1: col.height + wall + 2,
    })
}

pub(super) fn bounds(s: &Spec) -> (i32, i32) {
    (s.y0, s.y1)
}

pub(super) fn anchor(s: &Spec) -> (i32, i32) {
    (s.x, s.z)
}

pub(super) fn covers(s: &Spec, x: i32, z: i32) -> bool {
    (x - s.x).abs() <= s.half && (z - s.z).abs() <= s.half
}

pub(super) fn paint(s: &Spec, _m: &Materials, x: i32, y: i32, z: i32, ground: i32) -> Option<Stamp> {
    let dx = x - s.x;
    let dz = z - s.z;
    let dy = y - s.base;
    if dx.abs() > s.half || dz.abs() > s.half {
        return None;
    }
    let edge = dx.abs() == s.half || dz.abs() == s.half;
    let gate = dx == s.half && dz.abs() <= 1 && (1..=3).contains(&dy);
    if gate {
        return Some(carved());
    }
    if edge && (0..s.wall).contains(&dy) {
        return Some(solid(s.stone, y, ground));
    }
    let pillar = dx.abs() == s.half / 2 && dz.abs() == s.half / 2 && (1..s.wall).contains(&dy);
    if pillar {
        return Some(solid(s.stone, y, ground));
    }
    let beam = dx.rem_euclid(3) == 0 && dz.rem_euclid(3) == 0;
    let roof = dy == s.wall && (edge || beam) && hash2(s.salt, dx, dz) % 4 != 0;
    if roof {
        return Some(solid(s.stone, y, ground));
    }
    if dx.abs() <= 1 && dz.abs() <= 1 && dy == 0 {
        return Some(solid(s.stone, y, ground));
    }
    if (0..s.wall).contains(&dy) {
        return Some(carved());
    }
    foundation(s.base, ground, y, s.stone, true)
}

#[cfg(test)]
pub(super) fn describe(s: &Spec) -> super::Desc {
    super::Desc { x: s.x, z: s.z, pad: s.base, a: s.half, b: s.wall, qu: 0, qa: 0, qv: 0 }
}
