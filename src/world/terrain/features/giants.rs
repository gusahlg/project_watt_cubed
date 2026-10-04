//! Giant trees: bark trunks with a darkwood heart, root flares, and limb-borne leaf clusters.

use crate::block::registry::BlockId;

use super::noise::hash2;
use super::province::{Species, ThemeId, GIANTS};
use super::{keep, site_anchor, Ctx, Stamp, DIRS};

pub(super) const CELL: i32 = 52;
pub(super) const REACH: i32 = 16;
const SALT: u32 = 0x61A0_7501;

#[derive(Clone, Copy)]
pub(super) struct Spec {
    x: i32,
    z: i32,
    base: i32,
    trunk: i32,
    rad: i32,
    salt: u32,
    leaf: BlockId,
    y0: i32,
    y1: i32,
}

pub(super) fn spawn(ctx: &Ctx, cx: i32, cz: i32) -> Option<Spec> {
    let (x, z, h) = site_anchor(ctx.s ^ SALT, CELL, cx, cz);
    if !ctx.inland(x, z, REACH) {
        return None;
    }
    if !keep(ctx.scale, ctx.feats(x, z)[GIANTS], hash2(ctx.s ^ SALT ^ 0x11, cx, cz)) {
        return None;
    }
    let col = ctx.column(x, z);
    if col.slope4 >= 10 || col.theme == ThemeId::Petrified || col.surface == ctx.m.petrified {
        return None;
    }
    let leaf = match col.species {
        Species::Broadleaf => ctx.m.leaves,
        Species::Autumn => ctx.m.autumn,
        Species::Blossom => ctx.m.blossom,
        Species::None | Species::Conifer => return None,
    };
    let trunk = 30 + ((h >> 6) % 41) as i32;
    let rad = 1 + ((h >> 12) % 2) as i32;
    let base = col.height;
    Some(Spec { x, z, base, trunk, rad, salt: h, leaf, y0: base - 2, y1: base + trunk + 4 })
}

pub(super) fn bounds(s: &Spec) -> (i32, i32) {
    (s.y0, s.y1)
}

pub(super) fn paint(s: &Spec, m: &super::Materials, x: i32, y: i32, z: i32, _ground: i32) -> Option<Stamp> {
    let dx = x - s.x;
    let dz = z - s.z;
    let dy = y - s.base;
    if dx.abs().max(dz.abs()) > REACH || dy < -2 || dy >= s.trunk + 4 {
        return None;
    }
    if let Some(id) = wood(s, m, dx, dy, dz) {
        return Some(Stamp { id, dig: dy < 0 });
    }
    if leaves(s, dx, dy, dz) {
        return Some(Stamp { id: s.leaf, dig: false });
    }
    None
}

fn wood(s: &Spec, m: &super::Materials, dx: i32, dy: i32, dz: i32) -> Option<BlockId> {
    if ( -2..s.trunk).contains(&dy) {
        let r = if dy < 3 { s.rad + 2 } else if dy < 6 { s.rad + 1 } else { s.rad };
        if dx.abs().max(dz.abs()) <= r {
            return Some(if dx == 0 && dz == 0 { m.darkwood } else { m.bark });
        }
    }
    let n = limbs(s.salt);
    for i in 0..n {
        let limb = limb_at(s, i, n);
        for step in 0..=limb.len {
            let px = limb.ox * (s.rad + step);
            let pz = limb.oz * (s.rad + step);
            if dx == px && dz == pz && dy == limb.dy {
                return Some(m.bark);
            }
        }
    }
    None
}

fn leaves(s: &Spec, dx: i32, dy: i32, dz: i32) -> bool {
    let top = s.trunk - 1;
    let cy = dy - top;
    let crown = s.rad + 3;
    if (-2..=3).contains(&cy) && dx * dx + dz * dz + cy * cy <= crown * crown {
        return true;
    }
    let n = limbs(s.salt);
    for i in 0..n {
        let limb = limb_at(s, i, n);
        let ex = dx - limb.ox * (s.rad + limb.len);
        let ez = dz - limb.oz * (s.rad + limb.len);
        let ey = dy - limb.dy;
        if ex * ex + ey * ey + ez * ez <= limb.leaf * limb.leaf {
            return true;
        }
    }
    false
}

struct Limb {
    dy: i32,
    ox: i32,
    oz: i32,
    len: i32,
    leaf: i32,
}

fn limbs(salt: u32) -> i32 {
    3 + ((salt >> 4) % 4) as i32
}

fn limb_at(s: &Spec, i: i32, n: i32) -> Limb {
    let dir = DIRS[((s.salt >> (8 + i as u32)) % 8) as usize];
    let mut len = 4 + ((s.salt >> (16 + (i as u32) * 2)) % 5) as i32;
    let leaf = 2 + ((s.salt >> (24 + i as u32)) % 2) as i32;
    while s.rad + len + leaf > REACH && len > 2 {
        len -= 1;
    }
    Limb { dy: s.trunk * (i + 1) / (n + 1), ox: dir.0, oz: dir.1, len, leaf }
}
