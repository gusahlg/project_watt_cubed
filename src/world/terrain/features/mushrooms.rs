//! Giant mushrooms: a stem, a domed or flat cap with gills, and a cluster of small ones.

use crate::block::registry::BlockId;

use super::noise::hash2;
use super::province::{ThemeId, MUSHROOMS};
use super::{disk, keep, site_anchor, Ctx, Stamp, DIRS};

pub(super) const CELL: i32 = 40;
pub(super) const REACH: i32 = 18;
const SALT: u32 = 0x5E3D_0A11;

#[derive(Clone, Copy)]
pub(super) struct Spec {
    x: i32,
    z: i32,
    base: i32,
    stem: i32,
    stem_r: i32,
    cap_r: i32,
    cap_h: i32,
    flat: bool,
    salt: u32,
    cap: BlockId,
    gill: BlockId,
    y0: i32,
    y1: i32,
}

pub(super) fn spawn(ctx: &Ctx, cx: i32, cz: i32) -> Option<Spec> {
    let (x, z, h) = site_anchor(ctx.s ^ SALT, CELL, cx, cz);
    if !ctx.inland(x, z, REACH) {
        return None;
    }
    if !keep(ctx.scale, ctx.feats(x, z)[MUSHROOMS], hash2(ctx.s ^ SALT ^ 0x22, cx, cz)) {
        return None;
    }
    let col = ctx.column(x, z);
    if col.slope4 >= 8 {
        return None;
    }
    let m = ctx.m;
    let cap = if col.theme == ThemeId::GlowMoss || h % 5 == 0 {
        m.glowshroom
    } else if h % 2 == 0 {
        m.cap_red
    } else {
        m.cap_brown
    };
    let gill = if cap == m.glowshroom { m.glowshroom } else { m.stem };
    let stem = 8 + ((h >> 6) % 33) as i32;
    let stem_r = if stem >= 24 { 2 } else { 1 };
    let cap_r = 3 + ((h >> 12) % 4) as i32;
    let cap_h = 3 + ((h >> 16) % 3) as i32;
    let base = col.height;
    Some(Spec {
        x,
        z,
        base,
        stem,
        stem_r,
        cap_r,
        cap_h,
        flat: h % 2 == 0,
        salt: h,
        cap,
        gill,
        y0: base - 1,
        y1: base + stem + cap_h + 1,
    })
}

pub(super) fn bounds(s: &Spec) -> (i32, i32) {
    (s.y0, s.y1)
}

pub(super) fn paint(s: &Spec, m: &super::Materials, x: i32, y: i32, z: i32, _ground: i32) -> Option<Stamp> {
    let dx = x - s.x;
    let dz = z - s.z;
    if dx.abs().max(dz.abs()) > REACH {
        return None;
    }
    if let Some(st) = one(m, dx, y - s.base, dz, s.stem, s.stem_r, s.cap_r, s.cap_h, s.flat, s.cap, s.gill) {
        return Some(st);
    }
    for i in 0..4 {
        let dir = DIRS[((s.salt >> i) % 8) as usize];
        let dist = 5 + ((s.salt >> (4 + i)) % 5) as i32;
        let sh = 4 + ((s.salt >> (10 + i)) % 5) as i32;
        let sx = dir.0 * dist;
        let sz = dir.1 * dist;
        if let Some(st) = one(m, dx - sx, y - s.base, dz - sz, sh, 0, 2, 2, true, s.cap, s.gill) {
            return Some(st);
        }
    }
    None
}

fn one(
    m: &super::Materials,
    dx: i32,
    dy: i32,
    dz: i32,
    stem: i32,
    stem_r: i32,
    cap_r: i32,
    cap_h: i32,
    flat: bool,
    cap: BlockId,
    gill: BlockId,
) -> Option<Stamp> {
    if dy >= 0 && dy < stem && ((stem_r == 0 && dx == 0 && dz == 0) || disk(dx, dz, stem_r)) {
        return Some(Stamp { id: m.stem, dig: false });
    }
    if flat {
        if dy == stem && disk(dx, dz, cap_r) {
            let id = if disk(dx, dz, stem_r) { cap } else { gill };
            return Some(Stamp { id, dig: false });
        }
        if dy == stem + 1 && disk(dx, dz, cap_r) {
            return Some(Stamp { id: cap, dig: false });
        }
        return None;
    }
    if dy >= stem && dy < stem + cap_h {
        let k = dy - stem;
        let r = cap_r - k * cap_r / cap_h;
        if r >= 0 && ((r == 0 && dx == 0 && dz == 0) || disk(dx, dz, r)) {
            let id = if k == 0 && !disk(dx, dz, stem_r.max(1)) { gill } else { cap };
            return Some(Stamp { id, dig: false });
        }
    }
    None
}
