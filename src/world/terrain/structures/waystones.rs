//! Waystones: marked stones on the straight line between two nearby structures.

use super::super::Materials;
use super::{masonry, site_key, solid, Ctx, Stamp};
use crate::block::registry::BlockId;

pub(super) const CELL: i32 = 16;
pub(super) const REACH: i32 = 2;
const SALT: u32 = 0xA5A5_0009;

#[derive(Clone, Copy)]
pub(super) struct Spec {
    x: i32,
    z: i32,
    ground: i32,
    stone: BlockId,
    mark: BlockId,
    y0: i32,
    y1: i32,
}

pub(super) fn spawn(ctx: &Ctx, cx: i32, cz: i32) -> Option<Spec> {
    super::cache!(Spec, site_key(ctx, SALT), cx, cz, place(ctx, cx, cz))
}

pub(super) fn bounds(s: &Spec) -> (i32, i32) {
    (s.y0, s.y1)
}

pub(super) fn anchor(s: &Spec) -> (i32, i32) {
    (s.x, s.z)
}

pub(super) fn covers(s: &Spec, x: i32, z: i32) -> bool {
    x == s.x && z == s.z
}

fn consider(
    cell: i32,
    x: i32,
    z: i32,
    spawn: impl Fn(i32, i32) -> Option<(i32, i32)>,
    anchors: &mut [(i32, i32); 32],
    n: &mut usize,
) {
    let cx = x.div_euclid(cell);
    let cz = z.div_euclid(cell);
    for dz in -1..=1 {
        for dx in -1..=1 {
            if *n >= anchors.len() {
                return;
            }
            let Some(p) = spawn(cx + dx, cz + dz) else { continue };
            if !anchors[..*n].contains(&p) {
                anchors[*n] = p;
                *n += 1;
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Roads {
    anchors: [(i32, i32); 32],
    n: u8,
}

/// Points in the same structure cells see the same anchors. Chunk fill asks once per 16-block cell.
fn region_key(x: i32, z: i32) -> u64 {
    let cells = [
        super::towers::CELL,
        super::pyramids::CELL,
        super::circles::CELL,
        super::monoliths::CELL,
        super::temples::CELL,
        super::observatories::CELL,
        super::mines::CELL,
        super::bridges::CELL,
    ];
    let mut k = 0xA5A5_0009u64;
    for cell in cells {
        k = k.wrapping_mul(0x1000_0000_01B3) ^ x.div_euclid(cell) as u32 as u64;
        k = k.wrapping_mul(0x1000_0000_01B3) ^ z.div_euclid(cell) as u32 as u64;
    }
    k
}

fn roads(ctx: &Ctx, x: i32, z: i32) -> Roads {
    let key = region_key(x, z) ^ site_key(ctx, SALT);
    super::cache!(Roads, key, 0, 0, {
        let mut anchors = [(0i32, 0i32); 32];
        let n = collect(ctx, x, z, &mut anchors);
        Some(Roads { anchors, n: n as u8 })
    })
    .unwrap_or(Roads { anchors: [(0, 0); 32], n: 0 })
}

fn collect(ctx: &Ctx, x: i32, z: i32, anchors: &mut [(i32, i32); 32]) -> usize {
    let mut n = 0;
    consider(super::towers::CELL, x, z, |cx, cz| super::towers::spawn(ctx, cx, cz).map(|s| super::towers::anchor(&s)), anchors, &mut n);
    consider(super::pyramids::CELL, x, z, |cx, cz| super::pyramids::spawn(ctx, cx, cz).map(|s| super::pyramids::anchor(&s)), anchors, &mut n);
    consider(super::circles::CELL, x, z, |cx, cz| super::circles::spawn(ctx, cx, cz).map(|s| super::circles::anchor(&s)), anchors, &mut n);
    consider(super::monoliths::CELL, x, z, |cx, cz| super::monoliths::spawn(ctx, cx, cz).map(|s| super::monoliths::anchor(&s)), anchors, &mut n);
    consider(super::temples::CELL, x, z, |cx, cz| super::temples::spawn(ctx, cx, cz).map(|s| super::temples::anchor(&s)), anchors, &mut n);
    consider(super::observatories::CELL, x, z, |cx, cz| super::observatories::spawn(ctx, cx, cz).map(|s| super::observatories::anchor(&s)), anchors, &mut n);
    consider(super::mines::CELL, x, z, |cx, cz| super::mines::spawn(ctx, cx, cz).map(|s| super::mines::anchor(&s)), anchors, &mut n);
    consider(super::bridges::CELL, x, z, |cx, cz| super::bridges::spawn(ctx, cx, cz).map(|s| super::bridges::anchor(&s)), anchors, &mut n);
    n
}

fn nearest(anchors: &[(i32, i32)], i: usize) -> Option<(i32, i32)> {
    let (ax, az) = anchors[i];
    let mut best: Option<(i32, i32, i32)> = None;
    for (j, &(bx, bz)) in anchors.iter().enumerate() {
        if j == i {
            continue;
        }
        let d = (ax - bx).abs().max((az - bz).abs());
        if !(48..=4096).contains(&d) {
            continue;
        }
        let better = match best {
            None => true,
            Some((bd, x, z)) => d < bd || (d == bd && (bx, bz) < (x, z)),
        };
        if better {
            best = Some((d, bx, bz));
        }
    }
    best.map(|(_, x, z)| (x, z))
}

fn stone_in_cell(ax: i32, az: i32, bx: i32, bz: i32, cx: i32, cz: i32) -> Option<(i32, i32)> {
    let dx = bx - ax;
    let dz = bz - az;
    let steps = dx.abs().max(dz.abs());
    if !(48..=4096).contains(&steps) {
        return None;
    }
    let mut t = 48;
    while t < steps - 40 {
        let sx = ax + dx * t / steps;
        let sz = az + dz * t / steps;
        if sx.div_euclid(CELL) == cx && sz.div_euclid(CELL) == cz {
            return Some((sx, sz));
        }
        t += 16;
    }
    None
}

fn place(ctx: &Ctx, cx: i32, cz: i32) -> Option<Spec> {
    if ctx.scale <= 0.0 {
        return None;
    }
    let x = cx * CELL + CELL / 2;
    let z = cz * CELL + CELL / 2;
    let roads = roads(ctx, x, z);
    let n = roads.n as usize;
    if n < 2 {
        return None;
    }
    let anchors = &roads.anchors;
    for i in 0..n {
        let (ax, az) = anchors[i];
        let Some((bx, bz)) = nearest(&anchors[..n], i) else { continue };
        if (ax, az) >= (bx, bz) {
            continue;
        }
        let Some((sx, sz)) = stone_in_cell(ax, az, bx, bz, cx, cz) else { continue };
        if !ctx.inland(sx, sz, REACH) {
            continue;
        }
        let col = ctx.column(sx, sz);
        if col.slope4 >= 8 {
            continue;
        }
        let stone = masonry(ctx.m, col.theme, 0);
        let mark = if stone == ctx.m.obsidian { ctx.m.marble } else { ctx.m.obsidian };
        return Some(Spec {
            x: sx,
            z: sz,
            ground: col.height,
            stone,
            mark,
            y0: col.height,
            y1: col.height + 2,
        });
    }
    None
}

pub(super) fn paint(s: &Spec, _m: &Materials, x: i32, y: i32, z: i32, ground: i32) -> Option<Stamp> {
    if x != s.x || z != s.z || (y != s.ground && y != s.ground + 1) {
        return None;
    }
    let id = if y == s.ground { s.stone } else { s.mark };
    Some(solid(id, y, ground))
}

#[cfg(test)]
pub(super) fn describe(s: &Spec) -> super::Desc {
    super::Desc { x: s.x, z: s.z, pad: s.ground, a: 0, b: 0, qu: 0, qa: 0, qv: 0 }
}
