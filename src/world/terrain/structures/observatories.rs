//! Observatories: a domed hut and a rust-and-plank frame aimed at the nearest big body.

use super::super::noise::hash2;
use super::super::Materials;
use super::{carved, disk, foundation, keep, masonry, site_anchor, site_key, solid, Ctx, Stamp};
use crate::block::registry::BlockId;
use crate::coord::Face;
use crate::space::FaceFrame;

pub(super) const CELL: i32 = 2176;
pub(super) const REACH: i32 = 24;
const SALT: u32 = 0x0B5E_0006;
const DENS: f32 = 0.14;

#[derive(Clone, Copy)]
pub(super) struct Spec {
    x: i32,
    z: i32,
    base: i32,
    radius: i32,
    span: i32,
    qu: i32,
    qa: i32,
    qv: i32,
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
    let radius = 4 + (salt % 2) as i32;
    let span = 10 + ((salt >> 3) % 5) as i32;
    let foot = radius + 1;
    let samples = [(foot, 0), (-foot, 0), (0, foot), (0, -foot), (foot, foot), (-foot, foot), (foot, -foot), (-foot, -foot)];
    if !ctx.flat(x, z, col.height, &samples) {
        return None;
    }
    let (qu, qa, qv) = aim(ctx.face, ctx.half, ctx.centre, ctx.aims, x, col.height, z, span);
    Some(Spec {
        x,
        z,
        base: col.height,
        radius,
        span,
        qu,
        qa,
        qv,
        stone: masonry(ctx.m, col.theme, salt),
        y0: col.height - span - 8,
        y1: col.height + radius + span + 6,
    })
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
    if disk(dx, dz, s.radius + 1) {
        return true;
    }
    let span = s.span.max(1);
    for t in 0..=s.span {
        let fx = s.qu * t / span;
        let fz = s.qv * t / span;
        if (dx - fx).abs() <= 3 && (dz - fz).abs() <= 3 {
            return true;
        }
    }
    false
}

/// Direction from the hut to the nearest other big body's centre, in face-local block steps.
pub(super) fn aim(
    face: Face,
    half: i64,
    centre: [i64; 3],
    targets: &[[i64; 3]],
    u: i32,
    h: i32,
    v: i32,
    span: i32,
) -> (i32, i32, i32) {
    let straight = (0, span.max(0), 0);
    if span <= 0 || targets.is_empty() {
        return straight;
    }
    let Ok(a) = i32::try_from(half + i64::from(h)) else { return straight };
    let (wx, wy, wz) = FaceFrame::new(face).cell_to_world((u, a, v));
    let here = [centre[0] + i64::from(wx), centre[1] + i64::from(wy), centre[2] + i64::from(wz)];
    let mut best: Option<(i128, [i64; 3])> = None;
    for t in targets {
        let d0 = i128::from(t[0] - here[0]);
        let d1 = i128::from(t[1] - here[1]);
        let d2 = i128::from(t[2] - here[2]);
        let d = d0 * d0 + d1 * d1 + d2 * d2;
        let nearer = match best {
            None => true,
            Some((bd, _)) => d < bd,
        };
        if nearer {
            best = Some((d, *t));
        }
    }
    let Some((_, target)) = best else { return straight };
    let delta = [target[0] - here[0], target[1] - here[1], target[2] - here[2]];
    let (Ok(dx), Ok(dy), Ok(dz)) = (i32::try_from(delta[0]), i32::try_from(delta[1]), i32::try_from(delta[2])) else {
        return straight;
    };
    let (lu, la, lv) = FaceFrame::new(face).cell_to_local((dx, dy, dz));
    quantise(i64::from(lu), i64::from(la), i64::from(lv), span)
}

pub(super) fn quantise(du: i64, da: i64, dv: i64, span: i32) -> (i32, i32, i32) {
    let span_i = i64::from(span);
    let m = du.abs().max(da.abs()).max(dv.abs());
    if m == 0 || span_i <= 0 {
        return (0, span.max(0), 0);
    }
    ((du * span_i / m) as i32, (da * span_i / m) as i32, (dv * span_i / m) as i32)
}

fn perp(qu: i32, qa: i32, qv: i32) -> (i32, i32, i32) {
    if qa.abs() >= qu.abs() && qa.abs() >= qv.abs() { (1, 0, 0) } else { (0, 1, 0) }
}

fn orth(qu: i32, qa: i32, qv: i32, px: i32, py: i32, pz: i32) -> (i32, i32, i32) {
    let cx = qa * pz - qv * py;
    let cy = qv * px - qu * pz;
    let cz = qu * py - qa * px;
    let m = cx.abs().max(cy.abs()).max(cz.abs()).max(1);
    (cx / m, cy / m, cz / m)
}

fn frame(s: &Spec, m: &Materials, dx: i32, dy: i32, dz: i32, y: i32, ground: i32) -> Option<Stamp> {
    let span = s.span.max(1);
    let (px, py, pz) = perp(s.qu, s.qa, s.qv);
    let (sx, sy, sz) = orth(s.qu, s.qa, s.qv, px, py, pz);
    const DISH: [(i32, i32); 8] = [(2, 0), (-2, 0), (0, 2), (0, -2), (1, 1), (1, -1), (-1, 1), (-1, -1)];
    for t in 0..=s.span {
        let fx = s.qu * t / span;
        let fy = s.radius + s.qa * t / span;
        let fz = s.qv * t / span;
        if dx == fx && dy == fy && dz == fz {
            return Some(solid(m.plank, y, ground));
        }
        if dx == fx + px && dy == fy + py && dz == fz + pz {
            return Some(solid(m.rust, y, ground));
        }
        if dx == fx - px && dy == fy - py && dz == fz - pz {
            return Some(solid(m.rust, y, ground));
        }
        if t == s.span {
            for (a, b) in DISH {
                let rx = fx + px * a + sx * b;
                let ry = fy + py * a + sy * b;
                let rz = fz + pz * a + sz * b;
                if dx == rx && dy == ry && dz == rz {
                    return Some(solid(m.rust, y, ground));
                }
            }
        }
    }
    None
}

pub(super) fn paint(s: &Spec, m: &Materials, x: i32, y: i32, z: i32, ground: i32) -> Option<Stamp> {
    let dx = x - s.x;
    let dz = z - s.z;
    let dy = y - s.base;
    if dx.abs() > REACH || dz.abs() > REACH {
        return None;
    }
    if let Some(st) = frame(s, m, dx, dy, dz, y, ground) {
        return Some(st);
    }
    let r = s.radius;
    let d2 = dx * dx + dy * dy + dz * dz;
    let door = dz == 0 && dx > 0 && (0..=2).contains(&dy) && dx >= r - 1 && dx <= r;
    if door {
        return Some(carved());
    }
    let shell = dy >= 0 && d2 >= (r - 1) * (r - 1) && d2 <= r * r;
    if shell {
        return Some(solid(s.stone, y, ground));
    }
    if dy >= 0 && d2 < (r - 1) * (r - 1) {
        return Some(carved());
    }
    foundation(s.base, ground, y, s.stone, disk(dx, dz, r + 1))
}

#[cfg(test)]
pub(super) fn describe(s: &Spec) -> super::Desc {
    super::Desc { x: s.x, z: s.z, pad: s.base, a: s.span, b: s.radius, qu: s.qu, qa: s.qa, qv: s.qv }
}

#[cfg(test)]
mod tests {
    use super::quantise;

    #[test]
    fn aim_quantises_to_the_dominant_axis() {
        assert_eq!(quantise(0, 1000, 0, 10), (0, 10, 0));
        assert_eq!(quantise(300, 100, 0, 12), (12, 4, 0));
        assert_eq!(quantise(-50, 10, 0, 10), (-10, 2, 0));
        assert_eq!(quantise(0, 0, 0, 8), (0, 8, 0));
    }
}
