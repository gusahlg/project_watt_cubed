//! Canyon bridges: a plank-and-rope span or a stone arch, where both rims are close.

use super::super::cube;
use super::super::noise::hash2;
use super::super::province::ThemeId;
use super::super::Materials;
use super::{keep, masonry, site_anchor, site_key, solid, Ctx, Stamp};
use crate::block::registry::BlockId;

pub(super) const CELL: i32 = 1408;
/// Half a span, plus the rope margin. The deck can run ~80 from its midpoint.
pub(super) const REACH: i32 = 96;
const SALT: u32 = 0xB81D_0008;
const DENS: f32 = 0.55;

#[derive(Clone, Copy)]
pub(super) struct Spec {
    x: i32,
    z: i32,
    base: i32,
    left: i32,
    right: i32,
    rise: i32,
    axis: u8,
    arch: bool,
    stone: BlockId,
    y0: i32,
    y1: i32,
}

pub(super) fn spawn(ctx: &Ctx, cx: i32, cz: i32) -> Option<Spec> {
    super::cache!(Spec, site_key(ctx, SALT), cx, cz, build(ctx, cx, cz))
}

fn canyon(theme: ThemeId) -> bool {
    matches!(theme, ThemeId::Canyon | ThemeId::Mesa | ThemeId::Badlands | ThemeId::Karst)
}

struct Gap {
    axis: u8,
    mid: i32,
    left: i32,
    right: i32,
    deck: i32,
    dip: i32,
}

/// Valleys in this height field are about a hundred blocks across, so the probe is the whole site
/// line, not a short cross at the anchor.
fn transect(ctx: &Ctx, x: i32, z: i32, axis: u8, lo: i32, hi: i32) -> Option<Gap> {
    const STEP: i32 = 16;
    const MAX: usize = 96;
    let mut off = [0i32; MAX];
    let mut ht = [0i32; MAX];
    let mut n = 0;
    let mut p = lo;
    while p < hi && n < MAX {
        let (sx, sz) = if axis == 0 { (p, z) } else { (x, p) };
        off[n] = if axis == 0 { p - x } else { p - z };
        ht[n] = ctx.ground(sx, sz);
        n += 1;
        p += STEP;
    }
    let mut best: Option<Gap> = None;
    for a in 0..n {
        for b in a + 1..n {
            let width = off[b] - off[a];
            if width < 32 {
                continue;
            }
            if width > 160 {
                break;
            }
            if (ht[a] - ht[b]).abs() > 8 {
                continue;
            }
            let deck = ht[a].min(ht[b]);
            let mut coarse = deck;
            let mut ok = true;
            for k in a + 1..b {
                if ht[k] > deck + 2 {
                    ok = false;
                    break;
                }
                coarse = coarse.min(ht[k]);
            }
            if !ok || coarse > deck - 8 {
                continue;
            }
            if let Some(cur) = &best {
                let cw = cur.right - cur.left;
                if width > cw || (width == cw && deck - coarse <= cur.dip) {
                    continue;
                }
            }
            let (ax, az) = if axis == 0 { (x + off[a], z) } else { (x, z + off[a]) };
            let (bx, bz) = if axis == 0 { (x + off[b], z) } else { (x, z + off[b]) };
            if ctx.inset(ax, az) < i64::from(cube::RIM) || ctx.inset(bx, bz) < i64::from(cube::RIM) {
                continue;
            }
            let Some(dip) = clear_span(ctx, x, z, axis, off[a], off[b], deck) else { continue };
            let mid = off[a] + width / 2;
            best = Some(Gap {
                axis,
                mid,
                left: off[a] - mid,
                right: off[b] - mid,
                deck,
                dip,
            });
        }
    }
    best
}

fn clear_span(ctx: &Ctx, x: i32, z: i32, axis: u8, a: i32, b: i32, deck: i32) -> Option<i32> {
    let mut floor = deck;
    let mut t = a + 2;
    while t < b {
        let (sx, sz) = if axis == 0 { (x + t, z) } else { (x, z + t) };
        if ctx.inset(sx, sz) < i64::from(cube::RIM) {
            return None;
        }
        let h = ctx.ground(sx, sz);
        if h > deck + 1 {
            return None;
        }
        floor = floor.min(h);
        t += 2;
    }
    if floor > deck - 8 { None } else { Some(deck - floor) }
}

fn prefer(a: Gap, b: Gap) -> Gap {
    let w = a.right - a.left;
    let wb = b.right - b.left;
    if w < wb || (w == wb && a.dip >= b.dip) { a } else { b }
}

fn build(ctx: &Ctx, cx: i32, cz: i32) -> Option<Spec> {
    if !keep(ctx.scale, DENS, hash2(ctx.s ^ SALT, cx, cz)) {
        return None;
    }
    let (x, z, salt) = site_anchor(ctx.s ^ SALT, CELL, cx, cz);
    if !ctx.inland(x, z, REACH) {
        return None;
    }
    let col = ctx.column(x, z);
    if !canyon(col.theme) {
        return None;
    }
    let x0 = cx.wrapping_mul(CELL);
    let z0 = cz.wrapping_mul(CELL);
    let along = transect(ctx, x, z, 0, x0, x0.wrapping_add(CELL));
    let across = transect(ctx, x, z, 1, z0, z0.wrapping_add(CELL));
    let gap = match (along, across) {
        (Some(a), Some(b)) => prefer(a, b),
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => return None,
    };
    let (sx, sz) = if gap.axis == 0 { (x + gap.mid, z) } else { (x, z + gap.mid) };
    let arch = salt & 1 == 1;
    let width = gap.right - gap.left;
    let rise = if arch { (width / 4).clamp(3, 8) } else { 2 };
    let base = gap.deck - 1;
    Some(Spec {
        x: sx,
        z: sz,
        base,
        left: gap.left,
        right: gap.right,
        rise,
        axis: gap.axis,
        arch,
        stone: masonry(ctx.m, col.theme, salt),
        y0: base - 4,
        y1: base + rise + 3,
    })
}

pub(super) fn bounds(s: &Spec) -> (i32, i32) {
    (s.y0, s.y1)
}

pub(super) fn anchor(s: &Spec) -> (i32, i32) {
    (s.x, s.z)
}

fn axes(s: &Spec, dx: i32, dz: i32) -> (i32, i32) {
    if s.axis == 0 { (dx, dz) } else { (dz, dx) }
}

pub(super) fn covers(s: &Spec, x: i32, z: i32) -> bool {
    let (along, perp) = axes(s, x - s.x, z - s.z);
    (s.left - 1..=s.right + 1).contains(&along) && perp.abs() <= 2
}

pub(super) fn paint(s: &Spec, m: &Materials, x: i32, y: i32, z: i32, ground: i32) -> Option<Stamp> {
    let (along, perp) = axes(s, x - s.x, z - s.z);
    if !(s.left - 1..=s.right + 1).contains(&along) || perp.abs() > 2 {
        return None;
    }
    let on_span = (s.left..=s.right).contains(&along);
    if s.arch {
        let half = ((s.right - s.left) / 2).max(1);
        let mid = (s.left + s.right) / 2;
        let rel = along - mid;
        if on_span && perp.abs() <= 1 && rel.abs() <= half {
            let ay = s.rise * (half * half - rel * rel) / (half * half);
            if y == s.base + ay || y == s.base + ay + 1 {
                return Some(solid(s.stone, y, ground));
            }
        }
        if (along == s.left || along == s.right) && perp.abs() <= 1 && (s.base - 4..=s.base).contains(&y) {
            return Some(solid(s.stone, y, ground));
        }
        return None;
    }
    if on_span && perp.abs() <= 1 && y == s.base {
        return Some(solid(m.plank, y, ground));
    }
    if on_span && perp.abs() == 2 && along.rem_euclid(4) == 0 && (s.base..=s.base + 2).contains(&y) {
        return Some(solid(m.plank, y, ground));
    }
    if on_span && perp.abs() == 2 && y == s.base + 1 && along.rem_euclid(2) == 0 {
        return Some(solid(m.darkwood, y, ground));
    }
    if (along == s.left || along == s.right) && perp.abs() <= 1 && (s.base - 4..=s.base).contains(&y) {
        return Some(solid(m.plank, y, ground));
    }
    None
}

#[cfg(test)]
pub(super) fn describe(s: &Spec) -> super::Desc {
    super::Desc {
        x: s.x,
        z: s.z,
        pad: s.base,
        a: s.left,
        b: s.right,
        qu: i32::from(s.axis),
        qa: if s.arch { 1 } else { 0 },
        qv: s.rise,
    }
}
