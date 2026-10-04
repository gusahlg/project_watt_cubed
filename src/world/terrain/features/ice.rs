//! Ice spires and seracs, crevasse cuts, and frozen waterfalls. Uses the spire slot on ice strata.

use super::noise::hash2;
use super::province::{Strata, SPIRES};
use super::{keep, site_anchor, Ctx, Stamp, DIRS, Materials};

pub(super) const CELL: i32 = 40;
pub(super) const REACH: i32 = 18;
const SALT: u32 = 0x1CE0_0A11;

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
    if ctx.feats(x, z)[SPIRES] <= 0.0 {
        return None;
    }
    let col = ctx.column(x, z);
    if col.strata != Strata::Ice || !keep(ctx.scale, col.feats[SPIRES], hash2(ctx.s ^ SALT ^ 0x66, cx, cz)) {
        return None;
    }
    let base = col.height;
    let (kind, a, b, y0, y1) = if col.slope4 >= 6 {
        let drop = 10 + (h % 11) as i32;
        (2, drop, 0, base - drop, base + 1)
    } else if h % 3 == 0 {
        let half = 8 + ((h >> 4) % 8) as i32;
        let depth = 12 + ((h >> 8) % 13) as i32;
        (1, half, depth, base - 24 - depth, base + 26)
    } else {
        let height = 12 + ((h >> 4) % 25) as i32;
        (0, height, 0, base - 2, base + height + 1)
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
        1 => crevasse(s, m, dx, y, dz, ground),
        2 => fall(s, m, dx, y - s.base, dz),
        _ => spire(s, m, dx, y - s.base, dz),
    }
}

fn spire(s: &Spec, m: &Materials, dx: i32, dy: i32, dz: i32) -> Option<Stamp> {
    let r = if dy > s.a * 2 / 3 { 0 } else { 1 };
    let main = (-1..s.a).contains(&dy) && dx.abs().max(dz.abs()) <= r;
    let spur = dx == 3 && dz == 1 && (0..s.a * 2 / 3).contains(&dy);
    if !main && !spur {
        return None;
    }
    let id = if dy >= s.a - 2 { m.snow } else if dy % 5 == 0 { m.frost } else { m.ice };
    Some(Stamp { id, dig: dy < 0 })
}

fn crevasse(s: &Spec, m: &Materials, dx: i32, y: i32, dz: i32, ground: i32) -> Option<Stamp> {
    if (ground - s.base).abs() > 24 {
        return None;
    }
    let (along, perp) = if s.salt & 1 == 0 { (dx, dz) } else { (dz, dx) };
    if along.abs() > s.a {
        return None;
    }
    let mouth = if ground - y <= 2 { 1 } else { 0 };
    if perp.abs() <= mouth && y < ground && y >= ground - s.b {
        return Some(Stamp { id: super::AIR, dig: true });
    }
    if perp.abs() == mouth + 1 && y == ground {
        return Some(Stamp { id: m.snow, dig: false });
    }
    None
}

fn fall(s: &Spec, m: &Materials, dx: i32, dy: i32, dz: i32) -> Option<Stamp> {
    let dir = DIRS[(s.salt as usize) % 8];
    let sheet = (dx == 0 && dz == 0) || (dx == dir.0 && dz == dir.1) || (dx == dir.0 * 2 && dz == dir.1 * 2);
    if !sheet || dy > 0 || dy <= -s.a {
        return None;
    }
    let id = if dy == 0 { m.snow } else if dy % 4 == 0 { m.frost } else { m.ice };
    Some(Stamp { id, dig: false })
}
