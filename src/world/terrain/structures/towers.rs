//! Ruined towers: round or square, a broken top, one collapsed side, a rubble skirt, and a stair.

use super::super::noise::hash2;
use super::super::Materials;
use super::{carved, disk, foundation, keep, masonry, site_anchor, site_key, solid, Ctx, Stamp};
use crate::block::registry::BlockId;

pub(super) const CELL: i32 = 2048;
pub(super) const REACH: i32 = 12;
const SALT: u32 = 0x70E2_0001;
const DENS: f32 = 0.16;

#[derive(Clone, Copy)]
pub(super) struct Spec {
    x: i32,
    z: i32,
    base: i32,
    r: i32,
    h: i32,
    round: bool,
    broken: i32,
    side: u8,
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
    let r = 3 + ((salt >> 3) % 3) as i32;
    let h = 15 + ((salt >> 6) % 46) as i32;
    let foot = r + 3;
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
        r,
        h,
        round: salt & 1 == 1,
        broken: 2 + ((salt >> 12) % 5) as i32,
        side: ((salt >> 16) % 4) as u8,
        stone: masonry(ctx.m, col.theme, salt),
        salt,
        y0: col.height - 12,
        y1: col.height + h + 1,
    })
}

pub(super) fn bounds(s: &Spec) -> (i32, i32) {
    (s.y0, s.y1)
}

pub(super) fn anchor(s: &Spec) -> (i32, i32) {
    (s.x, s.z)
}

pub(super) fn covers(s: &Spec, x: i32, z: i32) -> bool {
    foot(s, x - s.x, z - s.z)
}

fn foot(s: &Spec, dx: i32, dz: i32) -> bool {
    let rad = s.r + 3;
    if s.round { disk(dx, dz, rad) } else { dx.abs().max(dz.abs()) <= rad }
}

fn wall(s: &Spec, dx: i32, dz: i32) -> bool {
    if s.round { disk(dx, dz, s.r) && !disk(dx, dz, s.r - 1) } else { dx.abs().max(dz.abs()) == s.r }
}

fn hollow(s: &Spec, dx: i32, dz: i32) -> bool {
    if s.round { disk(dx, dz, s.r - 1) } else { dx.abs().max(dz.abs()) < s.r }
}

fn round_step(r: i32, dy: i32) -> (i32, i32) {
    let k = (r - 1).max(1);
    let d = (k * 2 / 3).max(1);
    let pts = [(k, 0), (d, d), (0, k), (-d, d), (-k, 0), (-d, -d), (0, -k), (d, -d)];
    pts[dy.rem_euclid(8) as usize]
}

fn square_step(k: i32, dy: i32) -> (i32, i32) {
    if k <= 0 {
        return (0, 0);
    }
    let t = dy.rem_euclid(8 * k);
    let side = t / (2 * k);
    let i = t % (2 * k);
    match side {
        0 => (k, -k + i),
        1 => (k - i, k),
        2 => (-k, k - i),
        _ => (-k + i, -k),
    }
}

fn stair_at(s: &Spec, dx: i32, dz: i32, dy: i32) -> bool {
    if !(0..s.h).contains(&dy) {
        return false;
    }
    let (sx, sz) = if s.round { round_step(s.r, dy) } else { square_step(s.r - 1, dy) };
    dx == sx && dz == sz
}

fn door(s: &Spec, dx: i32, dz: i32, dy: i32) -> bool {
    dz == 0 && dx > 0 && (0..=2).contains(&dy) && wall(s, dx, dz)
}

fn collapsed(s: &Spec, dx: i32, dz: i32, dy: i32) -> bool {
    if dy <= s.h / 3 || dy >= s.h * 2 / 3 {
        return false;
    }
    match s.side {
        0 => dx > s.r / 2,
        1 => dx < -(s.r / 2),
        2 => dz > s.r / 2,
        _ => dz < -(s.r / 2),
    }
}

fn rubble(s: &Spec, dx: i32, dz: i32, dy: i32) -> bool {
    if !(0..=2).contains(&dy) || wall(s, dx, dz) || hollow(s, dx, dz) {
        return false;
    }
    let on = match s.side {
        0 => dx > s.r && dx <= s.r + 3 && dz.abs() <= s.r,
        1 => dx < -s.r && dx >= -s.r - 3 && dz.abs() <= s.r,
        2 => dz > s.r && dz <= s.r + 3 && dx.abs() <= s.r,
        _ => dz < -s.r && dz >= -s.r - 3 && dx.abs() <= s.r,
    };
    on && hash2(s.salt ^ 0x5B1E, dx, dz) % 3 != 0
}

pub(super) fn paint(s: &Spec, m: &Materials, x: i32, y: i32, z: i32, ground: i32) -> Option<Stamp> {
    let dx = x - s.x;
    let dz = z - s.z;
    let dy = y - s.base;
    if dx.abs() > REACH || dz.abs() > REACH {
        return None;
    }
    if rubble(s, dx, dz, dy) {
        return Some(solid(m.rubble, y, ground));
    }
    if hollow(s, dx, dz) && (0..s.h).contains(&dy) {
        if stair_at(s, dx, dz, dy) {
            return Some(solid(s.stone, y, ground));
        }
        return Some(carved());
    }
    if wall(s, dx, dz) && (0..s.h).contains(&dy) {
        if door(s, dx, dz, dy) || collapsed(s, dx, dz, dy) {
            return Some(carved());
        }
        if dy >= s.h - s.broken && hash2(s.salt ^ 0xB10C, dx, dz) % 3 == 0 {
            return Some(carved());
        }
        return Some(solid(s.stone, y, ground));
    }
    foundation(s.base, ground, y, s.stone, foot(s, dx, dz))
}

#[cfg(test)]
pub(super) fn describe(s: &Spec) -> super::Desc {
    super::Desc { x: s.x, z: s.z, pad: s.base, a: s.h, b: s.r, qu: if s.round { 1 } else { 0 }, qa: 0, qv: 0 }
}
