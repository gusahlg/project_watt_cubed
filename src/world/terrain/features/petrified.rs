//! Petrified trunks, standing or fallen. Spends the giant-tree slot on petrified ground.

use super::noise::hash2;
use super::province::{Species, ThemeId, GIANTS};
use super::{keep, site_anchor, Ctx, Stamp, Materials};

pub(super) const CELL: i32 = 28;
pub(super) const REACH: i32 = 14;
const SALT: u32 = 0x9E72_1F1E;

#[derive(Clone, Copy)]
pub(super) struct Spec {
    x: i32,
    z: i32,
    base: i32,
    kind: u8,
    a: i32,
    b: i32,
    salt: u32,
    y0: i32,
    y1: i32,
}

pub(super) fn spawn(ctx: &Ctx, cx: i32, cz: i32) -> Option<Spec> {
    let (x, z, h) = site_anchor(ctx.s ^ SALT, CELL, cx, cz);
    if !ctx.inland(x, z, REACH) {
        return None;
    }
    if ctx.feats(x, z)[GIANTS] <= 0.0 {
        return None;
    }
    let col = ctx.column(x, z);
    let here = col.surface == ctx.m.petrified || (col.theme == ThemeId::Petrified && col.species == Species::None);
    if !here || !keep(ctx.scale, col.feats[GIANTS], hash2(ctx.s ^ SALT ^ 0xCD, cx, cz)) {
        return None;
    }
    if col.slope4 >= 10 {
        return None;
    }
    let base = col.height;
    let (kind, a, b, top) = if h % 10 < 7 {
        let height = 6 + ((h >> 4) % 13) as i32;
        let rad = 1 + ((h >> 10) % 2) as i32;
        (0u8, height, rad, height + 1)
    } else {
        let len = 6 + ((h >> 4) % 9) as i32;
        (1, len, 1, 3)
    };
    Some(Spec { x, z, base, kind, a, b, salt: h, y0: base - 2, y1: base + top })
}

pub(super) fn bounds(s: &Spec) -> (i32, i32) {
    (s.y0, s.y1)
}

pub(super) fn paint(s: &Spec, m: &Materials, x: i32, y: i32, z: i32, _ground: i32) -> Option<Stamp> {
    let dx = x - s.x;
    let dz = z - s.z;
    let dy = y - s.base;
    if dx.abs().max(dz.abs()) > REACH {
        return None;
    }
    if s.kind == 0 {
        standing(s, m, dx, dy, dz)
    } else {
        fallen(s, m, dx, dy, dz)
    }
}

fn standing(s: &Spec, m: &Materials, dx: i32, dy: i32, dz: i32) -> Option<Stamp> {
    if dy < -1 || dy >= s.a {
        return None;
    }
    let mut r = s.b;
    if dy > s.a * 2 / 3 {
        r = 0;
    }
    let shift = if dy % 5 == 4 { 1 } else { 0 };
    if (dx - shift).abs().max(dz.abs()) <= r {
        return Some(Stamp { id: m.petrified, dig: dy < 0 });
    }
    None
}

fn fallen(s: &Spec, m: &Materials, dx: i32, dy: i32, dz: i32) -> Option<Stamp> {
    let (along, perp) = if s.salt & 1 == 0 { (dx, dz) } else { (dz, dx) };
    if along < 0 || along >= s.a || perp.abs() > s.b || !(-1..2).contains(&dy) {
        return None;
    }
    Some(Stamp { id: m.petrified, dig: dy < 0 })
}
