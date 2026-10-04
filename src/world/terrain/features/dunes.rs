//! Wind ripples on sand, and flat-capped buttes where the mesa theme spends the hoodoo slot.

use super::noise::hash2;
use super::province::{ThemeId, HOODOOS};
use super::{disk, keep, Ctx, Stamp, Materials};

pub(super) const CELL: i32 = 24;
pub(super) const REACH: i32 = 16;
const SALT: u32 = 0xD0E5_0A11;

#[derive(Clone, Copy)]
pub(super) struct Spec {
    x: i32,
    z: i32,
    cx: i32,
    cz: i32,
    base: i32,
    kind: u8,
    a: i32,
    b: i32,
    salt: u32,
    y0: i32,
    y1: i32,
}

pub(super) fn spawn(ctx: &Ctx, cx: i32, cz: i32) -> Option<Spec> {
    let (x, z, h) = centered(ctx.s ^ SALT, cx, cz);
    if !ctx.inland(x, z, REACH) {
        return None;
    }
    if ctx.feats(x, z)[HOODOOS] <= 0.0 {
        return None;
    }
    let col = ctx.column(x, z);
    let sand = col.surface == ctx.m.sand;
    let mesa = col.theme == ThemeId::Mesa;
    if (!sand && !mesa) || !keep(ctx.scale, col.feats[HOODOOS], hash2(ctx.s ^ SALT ^ 0x77, cx, cz)) {
        return None;
    }
    if mesa && col.slope4 >= 8 && !sand {
        return None;
    }
    let kind = if sand && !mesa { 0 } else if mesa && !sand { 1 } else if h % 2 == 0 { 0 } else { 1 };
    let base = col.height;
    let (a, b, y0, y1) = if kind == 0 {
        let period = 4 + (h % 4) as i32;
        (period, ((h >> 4) % period as u32) as i32, base - 25, base + 25)
    } else {
        let radius = 6 + ((h >> 2) % 4) as i32;
        let height = 8 + ((h >> 6) % 11) as i32;
        (radius, height, base, base + height + 1)
    };
    Some(Spec { x, z, cx, cz, base, kind, a, b, salt: h, y0, y1 })
}

fn centered(salt: u32, cx: i32, cz: i32) -> (i32, i32, u32) {
    let h = hash2(salt, cx, cz);
    let mid = CELL / 2;
    let x = cx.wrapping_mul(CELL) + mid - 2 + (h % 5) as i32;
    let z = cz.wrapping_mul(CELL) + mid - 2 + ((h >> 8) % 5) as i32;
    (x, z, h)
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
    if s.kind == 0 {
        ripple(s, m, x, y, z, ground)
    } else {
        butte(s, m, dx, y, dz)
    }
}

fn ripple(s: &Spec, m: &Materials, x: i32, y: i32, z: i32, ground: i32) -> Option<Stamp> {
    if x.div_euclid(CELL) != s.cx || z.div_euclid(CELL) != s.cz || (ground - s.base).abs() > 24 {
        return None;
    }
    let coord = if s.salt & 1 == 0 { x } else { z };
    let w = (coord + s.b).rem_euclid(s.a);
    if w == 0 && y == ground {
        return Some(Stamp { id: m.sand, dig: false });
    }
    if w == s.a / 2 && y == ground - 1 {
        return Some(Stamp { id: super::AIR, dig: true });
    }
    None
}

fn butte(s: &Spec, m: &Materials, dx: i32, y: i32, dz: i32) -> Option<Stamp> {
    let dy = y - s.base;
    if dy < 0 || dy > s.b {
        return None;
    }
    let cap = dy >= s.b - 1;
    let r = if cap { s.a + 1 } else { s.a };
    if !disk(dx, dz, r) {
        return None;
    }
    let id = if cap { m.redsand } else { m.sandstone[(dy as usize) % 4] };
    Some(Stamp { id, dig: false })
}
