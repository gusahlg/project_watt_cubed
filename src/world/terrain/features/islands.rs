//! Sky islands: an inverted cone of rock, a grass cap, small trees, dangling roots, a rare glow stream.

use super::noise::hash2;
use super::province::ISLANDS;
use super::{disk, keep, site_anchor, Ctx, Stamp, DIRS, Materials};

pub(super) const CELL: i32 = 176;
pub(super) const REACH: i32 = 44;
const SALT: u32 = 0x151A_4D00;

#[derive(Clone, Copy)]
pub(super) struct Spec {
    x: i32,
    z: i32,
    top: i32,
    radius: i32,
    vines: i32,
    trees: i32,
    salt: u32,
    stream: bool,
    y0: i32,
    y1: i32,
}

pub(super) fn spawn(ctx: &Ctx, cx: i32, cz: i32) -> Option<Spec> {
    let (x, z, h) = site_anchor(ctx.s ^ SALT, CELL, cx, cz);
    if !ctx.inland(x, z, REACH) {
        return None;
    }
    if !keep(ctx.scale, ctx.feats(x, z)[ISLANDS], hash2(ctx.s ^ SALT ^ 0x88, cx, cz)) {
        return None;
    }
    let col = ctx.column(x, z);
    let radius = 5 + ((h >> 8) % 36) as i32;
    let lift = 100 + (h % 201) as i32;
    let vines = 4 + ((h >> 16) % 7) as i32;
    let trees = ((h >> 20) % 4) as i32;
    let top = col.height + lift;
    Some(Spec {
        x,
        z,
        top,
        radius,
        vines,
        trees,
        salt: h,
        stream: h % 11 == 0,
        y0: top - radius - vines - 20,
        y1: top + 8,
    })
}

pub(super) fn bounds(s: &Spec) -> (i32, i32) {
    (s.y0, s.y1)
}

pub(super) fn paint(s: &Spec, m: &Materials, x: i32, y: i32, z: i32, _ground: i32) -> Option<Stamp> {
    let dx = x - s.x;
    let dz = z - s.z;
    let dy = y - s.top;
    if dx.abs().max(dz.abs()) > REACH {
        return None;
    }
    if s.stream && dx == s.radius + 1 && dz == 0 && dy <= 0 && dy > -14 {
        return Some(Stamp { id: m.glowcap, dig: false });
    }
    for i in 0..s.trees {
        let dir = DIRS[(s.salt as usize + i as usize) % 8];
        let dist = (s.radius / 3).max(2);
        let lx = dx - dir.0 * dist;
        let lz = dz - dir.1 * dist;
        if lx == 0 && lz == 0 && (1..=4).contains(&dy) {
            return Some(Stamp { id: m.timber, dig: false });
        }
        let ey = dy - 4;
        if lx * lx + lz * lz + ey * ey <= 4 && (3..=6).contains(&dy) && !(lx == 0 && lz == 0 && dy <= 4) {
            return Some(Stamp { id: m.leaves, dig: false });
        }
    }
    if dy == 0 && disk(dx, dz, s.radius) {
        return Some(Stamp { id: m.grass, dig: false });
    }
    if dy == -1 && disk(dx, dz, s.radius) {
        return Some(Stamp { id: m.soil, dig: false });
    }
    if dy <= -2 {
        let rad = s.radius - (-dy - 1);
        if rad > 0 && disk(dx, dz, rad) || rad == 0 && dx == 0 && dz == 0 {
            let depth = -dy;
            return Some(Stamp { id: m.rock[(depth as usize) % 4], dig: false });
        }
    }
    let tip = -(s.radius + 1);
    if dx.abs() <= 1 && dz.abs() <= 1 && dy < tip && dy >= tip - s.vines {
        return Some(Stamp { id: m.darkwood, dig: false });
    }
    for i in 0..4 {
        let dir = DIRS[i];
        if dx == dir.0 * 2 && dz == dir.1 * 2 && dy < tip + 2 && dy >= tip + 2 - s.vines {
            return Some(Stamp { id: m.bark, dig: false });
        }
    }
    None
}
