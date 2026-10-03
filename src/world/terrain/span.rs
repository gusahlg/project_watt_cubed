//! The twins' facing faces: tall spires and arches reaching across the weightless canyon.
//!
//! Both faces use the cube painter. This only dresses the face that looks at the other twin, and
//! only inside the face (clear of the rim blend). Everything it paints sits below [`CLEAR`], which
//! is inside the body's relief bound, so the 1.5e6-block gap past that bound stays empty. The
//! lower-id twin is lush (timber, leaves); the other is crystalline.

use super::cosmos::{Body, Cosmos, Kind};
use super::noise::{hash2, unit};
use super::{Materials, MAX_GROUND};
use super::cube::RIM;
use crate::block::registry::BlockId;
use crate::coord::Face;

/// First face-local altitude that is above every spire and arch.
pub(super) const CLEAR: i32 = 624;

const SITE: i32 = 80;
const TOP_LO: i32 = MAX_GROUND + 64;
const TOP_HI: i32 = 616;

struct Site {
    u: i32,
    v: i32,
    top: i32,
}

/// The face of `body` whose normal points at the other twin.
pub(super) fn facing_face(cosmos: &Cosmos, body: &Body) -> Option<Face> {
    if body.kind != Kind::Twin {
        return None;
    }
    let other = cosmos.bodies().iter().find(|b| b.kind == Kind::Twin && b.id != body.id)?;
    let axis = (0..3).max_by_key(|&a| (other.centre[a] - body.centre[a]).abs()).unwrap();
    let pos = other.centre[axis] > body.centre[axis];
    Some(match (axis, pos) {
        (0, true) => Face::PosX,
        (0, false) => Face::NegX,
        (1, true) => Face::PosY,
        (1, false) => Face::NegY,
        (2, true) => Face::PosZ,
        _ => Face::NegZ,
    })
}

/// The lush twin is the one with the smaller id; its partner is crystalline.
pub(super) fn lush(cosmos: &Cosmos, body: &Body) -> bool {
    cosmos.bodies().iter().filter(|b| b.kind == Kind::Twin).map(|b| b.id).min() == Some(body.id)
}

/// Salt for the facing face. One per twin, shared by the batch fill and the per-cell query.
pub(super) fn face_seed(body: &Body) -> u32 {
    body.seed ^ 0x5A11_0000
}

fn inset(half: i64, u: i32, v: i32) -> bool {
    half as i32 - u.abs().max(v.abs()) >= RIM + 96
}

fn site(seed: u32, su: i32, sv: i32) -> Option<Site> {
    let h = hash2(seed ^ 0x5A11_5EED, su, sv);
    if unit(h) > 0.62 {
        return None;
    }
    let u = su * SITE + SITE / 2 + (h % 21) as i32 - 10;
    let v = sv * SITE + SITE / 2 + ((h >> 5) % 21) as i32 - 10;
    let top = TOP_LO + ((h >> 12) % (TOP_HI - TOP_LO) as u32) as i32;
    Some(Site { u, v, top })
}

/// A spire or arch block at face-local `(u, h, v)`, if this facing face paints one.
pub(super) fn block(m: &Materials, lush: bool, seed: u32, half: i64, u: i32, h: i32, v: i32) -> Option<BlockId> {
    if !(0..CLEAR).contains(&h) || !inset(half, u, v) {
        return None;
    }
    let (su, sv) = (u.div_euclid(SITE), v.div_euclid(SITE));
    for dv in -1..=1 {
        for du in -1..=1 {
            let Some(s) = site(seed, su + du, sv + dv) else { continue };
            if let Some(id) = spire(m, lush, &s, u, h, v) {
                return Some(id);
            }
            if let Some(id) = arch_between(m, lush, seed, &s, su + du, sv + dv, 1, 0, u, h, v) {
                return Some(id);
            }
            if let Some(id) = arch_between(m, lush, seed, &s, su + du, sv + dv, 0, 1, u, h, v) {
                return Some(id);
            }
        }
    }
    None
}

fn spire(m: &Materials, lush: bool, s: &Site, u: i32, h: i32, v: i32) -> Option<BlockId> {
    if h >= s.top {
        return None;
    }
    let du = u - s.u;
    let dv = v - s.v;
    let d2 = du * du + dv * dv;
    if h + 8 >= s.top && d2 <= 10 {
        return Some(if lush { m.leaves } else { m.glowshroom });
    }
    if d2 > 4 {
        return None;
    }
    if lush {
        Some(if h % 23 == 0 { m.jade } else { m.timber })
    } else if d2 <= 1 {
        Some(m.marble)
    } else {
        Some(m.crystal)
    }
}

fn arch_between(
    m: &Materials,
    lush: bool,
    seed: u32,
    a: &Site,
    su: i32,
    sv: i32,
    du: i32,
    dv: i32,
    u: i32,
    h: i32,
    v: i32,
) -> Option<BlockId> {
    let b = site(seed, su + du, sv + dv)?;
    on_arch(m, lush, a, &b, u, h, v)
}

/// The ribbon between two spire tops. It springs from just inside each shaft and sags, staying
/// above [`MAX_GROUND`] so it clears every column of the face.
fn on_arch(m: &Materials, lush: bool, a: &Site, b: &Site, u: i32, h: i32, v: i32) -> Option<BlockId> {
    let du = b.u - a.u;
    let dv = b.v - a.v;
    let len2 = du * du + dv * dv;
    if len2 < 64 {
        return None;
    }
    let len = (len2 as f32).sqrt();
    let t = ((u - a.u) as f32 * du as f32 + (v - a.v) as f32 * dv as f32) / (len * len);
    if !(0.0..1.0).contains(&t) {
        return None;
    }
    let dist = t * len;
    if dist < 2.0 || dist > len - 2.0 {
        return None;
    }
    let cu = a.u as f32 + du as f32 * t;
    let cv = a.v as f32 + dv as f32 * t;
    let off2 = (u as f32 - cu) * (u as f32 - cu) + (v as f32 - cv) * (v as f32 - cv);
    if off2 > 2.25 {
        return None;
    }
    let span = (dist - 2.0) / (len - 4.0);
    let dip = 4.0 * span * (1.0 - span);
    let arch_h = a.top.min(b.top) - 4 - (24.0 * dip) as i32;
    let dh = h - arch_h;
    if !(-1..=1).contains(&dh) {
        return None;
    }
    let accent = if lush { m.leaves } else { m.glowshroom };
    let beam = if lush { m.timber } else { m.crystal };
    Some(if dh > 0 { accent } else { beam })
}

/// A shaft cell of some spire on this face, in face-local `(u, h, v)`.
#[cfg(test)]
pub(super) fn example(seed: u32, half: i64) -> Option<(i32, i32, i32)> {
    for su in -40..40 {
        for sv in -40..40 {
            let Some(s) = site(seed, su, sv) else { continue };
            if inset(half, s.u, s.v) {
                return Some((s.u, s.top - 24, s.v));
            }
        }
    }
    None
}

/// An arch cell between two spires, above the highest ground.
#[cfg(test)]
pub(super) fn an_arch(m: &Materials, lush_face: bool, seed: u32, half: i64) -> Option<(i32, i32, i32)> {
    for su in -30..30 {
        for sv in -30..30 {
            let Some(a) = site(seed, su, sv) else { continue };
            for (du, dv) in [(1, 0), (0, 1)] {
                let Some(b) = site(seed, su + du, sv + dv) else { continue };
                let u = (a.u + b.u) / 2;
                let v = (a.v + b.v) / 2;
                if !inset(half, u, v) {
                    continue;
                }
                let top = a.top.min(b.top);
                for h in (top - 40)..top {
                    if h <= MAX_GROUND {
                        continue;
                    }
                    if on_arch(m, lush_face, &a, &b, u, h, v).is_some() && block(m, lush_face, seed, half, u, h, v).is_some()
                    {
                        return Some((u, h, v));
                    }
                }
            }
        }
    }
    None
}
