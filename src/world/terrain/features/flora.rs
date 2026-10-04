//! Boulders, bushes, lichen-capped stones, and single-cell tufts. Not a province slot: a fixed
//! chance on gentle vegetated or stony ground, still scaled by the features knob.

use crate::block::registry::BlockId;

use super::{keep, site_anchor, Ctx, Stamp, Materials};

pub(super) const CELL: i32 = 8;
pub(super) const REACH: i32 = 3;
const SALT: u32 = 0xF102_A000;
const DENS: f32 = 0.40;

#[derive(Clone, Copy)]
pub(super) struct Spec {
    x: i32,
    z: i32,
    base: i32,
    kind: u8,
    salt: u32,
    y0: i32,
    y1: i32,
}

pub(super) fn spawn(ctx: &Ctx, cx: i32, cz: i32) -> Option<Spec> {
    let (x, z, h) = site_anchor(ctx.s ^ SALT, CELL, cx, cz);
    if !ctx.inland(x, z, REACH) || !keep(ctx.scale, DENS, h) {
        return None;
    }
    let col = ctx.column(x, z);
    if col.slope4 >= 6 {
        return None;
    }
    let m = ctx.m;
    let kind = if col.surface == m.gravel || col.surface == m.regolith {
        2
    } else if col.surface == m.lichen || col.surface == m.tundra || col.surface == m.moss {
        3
    } else if col.surface == m.grass || col.surface == m.meadow || col.surface == m.soil {
        if h % 2 == 0 { 0 } else { 1 }
    } else {
        return None;
    };
    let base = col.height;
    Some(Spec { x, z, base, kind, salt: h, y0: base - 26, y1: base + 28 })
}

pub(super) fn bounds(s: &Spec) -> (i32, i32) {
    (s.y0, s.y1)
}

pub(super) fn paint(s: &Spec, m: &Materials, x: i32, y: i32, z: i32, ground: i32) -> Option<Stamp> {
    let dx = x - s.x;
    let dz = z - s.z;
    if dx.abs().max(dz.abs()) > REACH || (ground - s.base).abs() > 24 {
        return None;
    }
    let id = match s.kind {
        0 => tuft(s, m, dx, y, dz, ground),
        1 => bush(s, m, dx, y, dz, ground),
        2 => boulder(s, m, dx, y, dz, ground),
        _ => lichen_stone(s, m, dx, y, dz, ground),
    }?;
    Some(Stamp { id, dig: false })
}

fn tuft(s: &Spec, m: &Materials, dx: i32, y: i32, dz: i32, ground: i32) -> Option<BlockId> {
    if dx == 0 && dz == 0 && y == ground {
        Some(if s.salt % 2 == 0 { m.meadow } else { m.lichen })
    } else {
        None
    }
}

fn bush(s: &Spec, m: &Materials, dx: i32, y: i32, dz: i32, ground: i32) -> Option<BlockId> {
    if dx == 0 && dz == 0 && y == ground {
        return Some(m.darkwood);
    }
    if dx.abs().max(dz.abs()) <= 1 && y == ground + 1 {
        return Some(m.leaves);
    }
    if dx == 0 && dz == 0 && y == ground + 2 && s.salt % 4 == 0 {
        return Some(if s.salt % 8 == 0 { m.flower_white } else { m.flower_yellow });
    }
    None
}

fn boulder(s: &Spec, m: &Materials, dx: i32, y: i32, dz: i32, ground: i32) -> Option<BlockId> {
    let r = if s.salt % 3 == 0 { 1 } else { 0 };
    if dx.abs().max(dz.abs()) <= r && y == ground {
        return Some(m.rock[(s.salt as usize) % 4]);
    }
    if dx == 0 && dz == 0 && y == ground + 1 && s.salt % 2 == 0 {
        return Some(m.lichen);
    }
    None
}

fn lichen_stone(s: &Spec, m: &Materials, dx: i32, y: i32, dz: i32, ground: i32) -> Option<BlockId> {
    if dx.abs() + dz.abs() <= 1 && y == ground {
        return Some(m.rock[(s.salt as usize) % 4]);
    }
    if dx == 0 && dz == 0 && y == ground + 1 {
        return Some(m.lichen);
    }
    None
}
