//! Clusters of prismatic crystals leaning out from a shared root.

use crate::block::registry::BlockId;

use super::noise::hash2;
use super::province::CRYSTALS;
use super::{keep, site_anchor, Ctx, Stamp, DIRS, Materials};

pub(super) const CELL: i32 = 24;
pub(super) const REACH: i32 = 16;
const SALT: u32 = 0xC295_7A11;

#[derive(Clone, Copy)]
pub(super) struct Spec {
    x: i32,
    z: i32,
    base: i32,
    salt: u32,
    y0: i32,
    y1: i32,
}

pub(super) fn spawn(ctx: &Ctx, cx: i32, cz: i32) -> Option<Spec> {
    let (x, z, h) = site_anchor(ctx.s ^ SALT, CELL, cx, cz);
    if !ctx.inland(x, z, REACH) {
        return None;
    }
    if !keep(ctx.scale, ctx.feats(x, z)[CRYSTALS], hash2(ctx.s ^ SALT ^ 0x55, cx, cz)) {
        return None;
    }
    let col = ctx.column(x, z);
    if col.slope4 >= 12 {
        return None;
    }
    let base = col.height;
    Some(Spec { x, z, base, salt: h, y0: base - 2, y1: base + 31 })
}

pub(super) fn bounds(s: &Spec) -> (i32, i32) {
    (s.y0, s.y1)
}

pub(super) fn paint(s: &Spec, m: &Materials, x: i32, y: i32, z: i32, _ground: i32) -> Option<Stamp> {
    let dx = x - s.x;
    let dz = z - s.z;
    let dy = y - s.base;
    if dx.abs().max(dz.abs()) > REACH || dy < -1 || dy > 30 {
        return None;
    }
    let count = 5 + (s.salt % 5) as usize;
    for i in 0..count {
        let mat = crystal(m, i);
        let vertical = i == 0;
        let len = if vertical { 16 + ((s.salt >> 8) % 15) as i32 } else { 4 + ((s.salt >> (i as u32)) % 21) as i32 };
        let dir = if vertical { (0, 0) } else { DIRS[(i + (s.salt as usize % 8)) % 8] };
        if dy == -1 && dx == 0 && dz == 0 {
            return Some(Stamp { id: mat, dig: true });
        }
        if dy < 0 || dy >= len {
            continue;
        }
        let shift = if vertical { 0 } else { dy / 2 };
        let ox = dir.0 * shift;
        let oz = dir.1 * shift;
        if ox.abs().max(oz.abs()) > REACH - 1 {
            continue;
        }
        let on = dx == ox && dz == oz;
        let thick = dy < len / 3 && (dx - ox).abs() + (dz - oz).abs() <= 1;
        if on || thick {
            return Some(Stamp { id: mat, dig: false });
        }
    }
    None
}

fn crystal(m: &Materials, i: usize) -> BlockId {
    match i % 4 {
        0 => m.crystal,
        1 => m.violet,
        2 => m.glowcap,
        _ => m.glowshroom,
    }
}
