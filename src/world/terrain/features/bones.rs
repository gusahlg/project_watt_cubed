//! Bone lands: ribcage arches, half-buried skulls, and spines.

use super::noise::hash2;
use super::province::BONES;
use super::{keep, site_anchor, Ctx, Stamp, Materials};

pub(super) const CELL: i32 = 48;
pub(super) const REACH: i32 = 18;
const SALT: u32 = 0xB0E0_0A11;

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
    if !keep(ctx.scale, ctx.feats(x, z)[BONES], hash2(ctx.s ^ SALT ^ 0xAB, cx, cz)) {
        return None;
    }
    let col = ctx.column(x, z);
    if col.slope4 >= 10 {
        return None;
    }
    let base = col.height;
    let kind = (h % 3) as u8;
    let (a, b, top) = match kind {
        0 => {
            let half = 6 + (h % 3) as i32;
            let rise = 8 + ((h >> 4) % 5) as i32;
            (half, rise, rise + 4)
        }
        1 => (5, 4, 8),
        _ => {
            let len = 8 + ((h >> 4) % 7) as i32;
            (len, 0, 3)
        }
    };
    Some(Spec { x, z, base, kind, a, b, salt: h, y0: base - 4, y1: base + top })
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
    match s.kind {
        0 => ribs(s, m, dx, dy, dz),
        1 => skull(s, m, dx, dy, dz),
        _ => spine(s, m, dx, dy, dz),
    }
}

fn ribs(s: &Spec, m: &Materials, dx: i32, dy: i32, dz: i32) -> Option<Stamp> {
    let ribs = 3 + (s.salt % 3) as i32;
    let half = s.a;
    let rise = s.b;
    for i in 0..ribs {
        let oz = (i - ribs / 2) * 2;
        if dz != oz || dx.abs() > half {
            continue;
        }
        let h2 = half * half;
        let ay = rise * (h2 - dx * dx) / h2.max(1);
        if dy >= ay - 1 && dy <= ay + 1 && dy >= -2 {
            return Some(Stamp { id: m.bone, dig: dy < 2 });
        }
    }
    None
}

fn skull(_s: &Spec, m: &Materials, dx: i32, dy: i32, dz: i32) -> Option<Stamp> {
    let (rx2, ry2, rz2) = (25i64, 16i64, 16i64);
    let cy = dy - 1;
    let inside = i64::from(dx * dx) * ry2 * rz2 + i64::from(cy * cy) * rx2 * rz2 + i64::from(dz * dz) * rx2 * ry2
        <= rx2 * ry2 * rz2;
    if !inside {
        return None;
    }
    let eye = cy == 1 && dz == 2 && dx.abs() == 2;
    let nose = cy == 0 && dx == 0 && dz == 3;
    if eye || nose {
        return Some(Stamp { id: super::AIR, dig: true });
    }
    Some(Stamp { id: m.bone, dig: dy < 2 })
}

fn spine(s: &Spec, m: &Materials, dx: i32, dy: i32, dz: i32) -> Option<Stamp> {
    let (along, perp) = if s.salt & 1 == 0 { (dx, dz) } else { (dz, dx) };
    if along < 0 || along >= s.a || perp.abs() > 0 || !(-2..=1).contains(&dy) {
        return None;
    }
    Some(Stamp { id: m.bone, dig: dy < 1 })
}
