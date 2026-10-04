//! Round karst shafts. Spends the crater slot on limestone, and opens into the caves below.

use crate::block::registry::AIR;

use super::noise::hash2;
use super::province::{Strata, CRATERS};
use super::{disk, keep, site_anchor, Ctx, Stamp, Materials};

pub(super) const CELL: i32 = 64;
pub(super) const REACH: i32 = 12;
const SALT: u32 = 0x51A0_0E11;

#[derive(Clone, Copy)]
pub(super) struct Spec {
    x: i32,
    z: i32,
    base: i32,
    radius: i32,
    depth: i32,
    salt: u32,
    y0: i32,
    y1: i32,
}

pub(super) fn spawn(ctx: &Ctx, cx: i32, cz: i32) -> Option<Spec> {
    let (x, z, h) = site_anchor(ctx.s ^ SALT, CELL, cx, cz);
    if !ctx.inland(x, z, REACH) {
        return None;
    }
    if ctx.feats(x, z)[CRATERS] <= 0.0 {
        return None;
    }
    let col = ctx.column(x, z);
    if col.strata != Strata::Limestone || col.slope4 >= 16 {
        return None;
    }
    if !keep(ctx.scale, col.feats[CRATERS], hash2(ctx.s ^ SALT ^ 0xEF, cx, cz)) {
        return None;
    }
    let radius = 3 + ((h >> 4) % 6) as i32;
    let depth = 18 + ((h >> 8) % 47) as i32;
    let base = col.height;
    Some(Spec { x, z, base, radius, depth, salt: h, y0: base - 24 - depth, y1: base + 26 })
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
    let mouth = if y > ground - 4 { s.radius + 1 } else { s.radius };
    if y < ground && y >= ground - s.depth && disk(dx, dz, mouth) {
        return Some(Stamp { id: AIR, dig: true });
    }
    if y == ground && disk(dx, dz, s.radius + 2) && !disk(dx, dz, s.radius) {
        let id = if s.salt % 2 == 0 { m.limestone } else { m.rubble };
        return Some(Stamp { id, dig: false });
    }
    None
}
