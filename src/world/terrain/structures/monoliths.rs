//! Monoliths: an obsidian or marble cube, standing on a face or on a corner.

use super::super::noise::hash2;
use super::super::province::ThemeId;
use super::super::Materials;
use super::{foundation, keep, site_anchor, site_key, solid, Ctx, Stamp};
use crate::block::registry::BlockId;

pub(super) const CELL: i32 = 2560;
pub(super) const REACH: i32 = 24;
const SALT: u32 = 0xA0A0_0004;
const DENS: f32 = 0.14;

#[derive(Clone, Copy)]
pub(super) struct Spec {
    x: i32,
    z: i32,
    base: i32,
    side: i32,
    corner: bool,
    stone: BlockId,
    y0: i32,
    y1: i32,
}

pub(super) fn spawn(ctx: &Ctx, cx: i32, cz: i32) -> Option<Spec> {
    super::cache!(Spec, site_key(ctx, SALT), cx, cz, build(ctx, cx, cz))
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
    if col.slope4 >= 8 {
        return None;
    }
    let side = 8 + ((salt >> 3) % 5) as i32 * 4;
    let corner = salt & 1 == 1;
    let samples: [(i32, i32); 8] = if corner {
        let s = side - 1;
        let m = s / 2;
        [(0, 0), (s, 0), (0, s), (s, s), (m, 0), (0, m), (s, m), (m, s)]
    } else {
        let lo = -(side / 2);
        let hi = side / 2 - 1;
        [(lo, lo), (lo, hi), (hi, lo), (hi, hi), (lo, 0), (hi, 0), (0, lo), (0, hi)]
    };
    if !ctx.flat(x, z, col.height, &samples) {
        return None;
    }
    Some(Spec {
        x,
        z,
        base: col.height,
        side,
        corner,
        stone: block(col.theme, ctx.m),
        y0: col.height - 12,
        y1: col.height + side,
    })
}

fn block(theme: ThemeId, m: &Materials) -> BlockId {
    match theme {
        ThemeId::Volcanic | ThemeId::Ash | ThemeId::Crater => m.obsidian,
        _ => m.marble,
    }
}

pub(super) fn bounds(s: &Spec) -> (i32, i32) {
    (s.y0, s.y1)
}

pub(super) fn anchor(s: &Spec) -> (i32, i32) {
    (s.x, s.z)
}

pub(super) fn covers(s: &Spec, x: i32, z: i32) -> bool {
    let dx = x - s.x;
    let dz = z - s.z;
    if s.corner {
        (0..s.side).contains(&dx) && (0..s.side).contains(&dz)
    } else {
        let u = dx + s.side / 2;
        let v = dz + s.side / 2;
        (0..s.side).contains(&u) && (0..s.side).contains(&v)
    }
}

/// Cube on a vertex. The cell centre `(2x+1, 2y+1, 2z+1)` lies in the corner-origin cube of side `s`.
fn in_corner(dx: i32, dy: i32, dz: i32, s: i32) -> bool {
    let x = dx * 2 + 1;
    let y = dy * 2 + 1;
    let z = dz * 2 + 1;
    let a = x + y - z;
    let b = y + z - x;
    let c = x + z - y;
    let lim = s * 2;
    a >= 0 && b >= 0 && c >= 0 && a < lim && b < lim && c < lim
}

pub(super) fn paint(s: &Spec, _m: &Materials, x: i32, y: i32, z: i32, ground: i32) -> Option<Stamp> {
    let dx = x - s.x;
    let dz = z - s.z;
    let dy = y - s.base;
    if dx.abs() > REACH || dz.abs() > REACH || dy.abs() > REACH {
        return None;
    }
    let inside = if s.corner {
        in_corner(dx, dy, dz, s.side)
    } else {
        let u = dx + s.side / 2;
        let v = dz + s.side / 2;
        (0..s.side).contains(&u) && (0..s.side).contains(&v) && (0..s.side).contains(&dy)
    };
    if inside {
        return Some(solid(s.stone, y, ground));
    }
    foundation(s.base, ground, y, s.stone, covers(s, x, z))
}

#[cfg(test)]
pub(super) fn describe(s: &Spec) -> super::Desc {
    super::Desc {
        x: s.x,
        z: s.z,
        pad: s.base,
        a: s.side,
        b: if s.corner { 1 } else { 0 },
        qu: 0,
        qa: 0,
        qv: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::in_corner;

    #[test]
    fn corner_cube_stays_in_the_positive_octant() {
        for s in [8, 16, 24] {
            assert!(in_corner(0, 0, 0, s));
            assert!(in_corner(s - 1, s - 1, s - 1, s));
            assert!(!in_corner(-1, 0, 0, s));
            assert!(!in_corner(s, 0, 0, s));
            assert!(!in_corner(0, -1, 0, s));
            for dx in 0..s {
                for dz in 0..s {
                    for dy in 0..s {
                        if in_corner(dx, dy, dz, s) {
                            assert!((0..s).contains(&dx) && (0..s).contains(&dy) && (0..s).contains(&dz));
                        }
                    }
                }
            }
        }
    }
}
