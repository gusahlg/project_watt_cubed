//! Impact craters: a raised rim, a parabolic bowl, and a meteorite core. Limestone keeps this
//! slot for sinkholes.

use crate::block::registry::{AIR, BlockId};

use super::noise::hash2;
use super::province::{Strata, CRATERS};
use super::{keep, site_anchor, Ctx, Stamp, Materials};

pub(super) const CELL: i32 = 360;
pub(super) const REACH: i32 = 168;
const SALT: u32 = 0xC2A7_E200;

#[derive(Clone, Copy)]
pub(super) struct Spec {
    x: i32,
    z: i32,
    base: i32,
    radius: i32,
    rim_w: i32,
    rim_h: i32,
    depth: i32,
    core: i32,
    core_id: BlockId,
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
    if col.strata == Strata::Limestone || !keep(ctx.scale, col.feats[CRATERS], hash2(ctx.s ^ SALT ^ 0x99, cx, cz)) {
        return None;
    }
    let radius = 10 + ((h >> 4) % 141) as i32;
    let rim_w = 3 + (h % 4) as i32;
    let rim_h = 2 + (radius / 30).min(6);
    let depth = (radius / 3).clamp(8, 40);
    let core = 3 + ((h >> 12) % 3) as i32;
    let core_id = match h % 3 {
        0 => ctx.m.gold,
        1 => ctx.m.copper,
        _ => ctx.m.basalt,
    };
    let base = col.height;
    Some(Spec {
        x,
        z,
        base,
        radius,
        rim_w,
        rim_h,
        depth,
        core,
        core_id,
        y0: base - 32 - depth - core - 2,
        y1: base + 32 + rim_h + 2,
    })
}

pub(super) fn bounds(s: &Spec) -> (i32, i32) {
    (s.y0, s.y1)
}

pub(super) fn paint(s: &Spec, m: &Materials, x: i32, y: i32, z: i32, ground: i32) -> Option<Stamp> {
    let dx = x - s.x;
    let dz = z - s.z;
    if dx.abs().max(dz.abs()) > REACH || (ground - s.base).abs() > 32 {
        return None;
    }
    let d2 = i64::from(dx) * i64::from(dx) + i64::from(dz) * i64::from(dz);
    let r2 = i64::from(s.radius) * i64::from(s.radius);
    let outer = i64::from(s.radius + s.rim_w);
    let o2 = outer * outer;
    if d2 > o2 {
        return None;
    }
    if d2 <= r2 {
        let dig = (i64::from(s.depth) * (r2 - d2) / r2.max(1)) as i32;
        let floor = ground - dig;
        if dig > 0 && y == floor {
            return Some(Stamp { id: m.regolith, dig: true });
        }
        if y < floor && y >= floor - s.core && d2 <= 9 {
            return Some(Stamp { id: s.core_id, dig: true });
        }
        if dig > 0 && y > floor && y < ground {
            return Some(Stamp { id: AIR, dig: true });
        }
        return None;
    }
    let span = (o2 - r2).max(1);
    let t = ((o2 - d2) * i64::from(s.rim_h) / span) as i32;
    if t > 0 && y >= ground && y < ground + t {
        let id = if (y + s.radius) % 3 == 0 { m.gravel } else { m.regolith };
        return Some(Stamp { id, dig: false });
    }
    None
}
