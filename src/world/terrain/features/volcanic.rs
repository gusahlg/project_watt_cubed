//! Cinder cones with summit magma, basalt prism fields, and obsidian flows split by a fissure.

use super::noise::hash2;
use super::province::CONES;
use super::{disk, keep, site_anchor, Ctx, Stamp, Materials};

pub(super) const CELL: i32 = 112;
pub(super) const REACH: i32 = 64;
const SALT: u32 = 0xC01E_5A11;

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
    if !keep(ctx.scale, ctx.feats(x, z)[CONES], hash2(ctx.s ^ SALT ^ 0x44, cx, cz)) {
        return None;
    }
    let col = ctx.column(x, z);
    if col.slope4 >= 12 {
        return None;
    }
    let base = col.height;
    let (kind, a, b, y0, y1) = match h % 20 {
        0..=10 => {
            let height = 20 + ((h >> 8) % 101) as i32;
            let radius = (height / 2).clamp(8, REACH - 4);
            (0, height, radius, base, base + height)
        }
        11..=15 => (1, 14, 0, base, base + 17),
        _ => {
            let len = 10 + ((h >> 8) % 16) as i32;
            (2, len, 0, base - 28, base + 26)
        }
    };
    Some(Spec { x, z, base, kind, a, b, salt: h, y0, y1 })
}

pub(super) fn bounds(s: &Spec) -> (i32, i32) {
    (s.y0, s.y1)
}

pub(super) fn paint(s: &Spec, m: &Materials, x: i32, y: i32, z: i32, ground: i32) -> Option<Stamp> {
    let dx = x - s.x;
    let dz = z - s.z;
    if dx.abs().max(dz.abs()) > REACH {
        return None;
    }
    match s.kind {
        0 => cone(s, m, dx, y - s.base, dz),
        1 => columns(s, m, dx, y - s.base, dz),
        _ => flow(s, m, dx, y, dz, ground),
    }
}

fn cone(s: &Spec, m: &Materials, dx: i32, dy: i32, dz: i32) -> Option<Stamp> {
    if dy < 0 || dy >= s.a {
        return None;
    }
    let rad = s.b * (s.a - dy) / s.a;
    let crater_d = (s.a / 6).clamp(4, 12);
    let crater_r = (s.b / 4).clamp(2, 10);
    if dy >= s.a - crater_d && disk(dx, dz, crater_r) {
        return (dy < s.a - crater_d + 2).then_some(Stamp { id: m.magma, dig: false });
    }
    let on = if rad <= 0 { dx == 0 && dz == 0 } else { disk(dx, dz, rad) };
    if !on {
        return None;
    }
    let shell = rad <= 1 || !disk(dx, dz, rad * 2 / 3);
    Some(Stamp { id: if shell { m.basalt } else { m.ash }, dig: false })
}

fn columns(s: &Spec, m: &Materials, dx: i32, dy: i32, dz: i32) -> Option<Stamp> {
    if dy < 0 || dy > 16 {
        return None;
    }
    for iz in -4..=4 {
        for ix in -4..=4 {
            let ox = if iz % 2 == 0 { 0 } else { 1 };
            let px = ix * 3 + ox;
            let pz = iz * 3;
            if dx != px || dz != pz {
                continue;
            }
            let h = 4 + (hash2(s.salt, ix, iz) % 13) as i32;
            if hash2(s.salt ^ 0x51, ix, iz) % 5 == 0 {
                return None;
            }
            if dy < h {
                return Some(Stamp { id: m.basalt, dig: false });
            }
        }
    }
    None
}

fn flow(s: &Spec, m: &Materials, dx: i32, y: i32, dz: i32, ground: i32) -> Option<Stamp> {
    if (ground - s.base).abs() > 24 {
        return None;
    }
    let along = if s.salt & 1 == 0 { dx } else { dz };
    let perp = if s.salt & 1 == 0 { dz } else { dx };
    if along < 0 || along >= s.a || perp.abs() > 2 {
        return None;
    }
    if perp == 0 && y < ground && y >= ground - 3 {
        return Some(Stamp { id: m.magma, dig: true });
    }
    if y == ground {
        let id = if perp == 0 { m.magma } else { m.obsidian };
        return Some(Stamp { id, dig: false });
    }
    None
}
