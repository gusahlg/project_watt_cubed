//! Rock spires, mushroom-capped hoodoos, and natural arches. Ice strata and sand or mesa
//! columns leave this slot to the ice and dune families.

use super::noise::{hash2, unit};
use super::province::{Strata, ThemeId, HOODOOS, SPIRES};
use super::{disk, keep, site_anchor, Ctx, Stamp, DIRS, Materials};

pub(super) const CELL: i32 = 48;
pub(super) const REACH: i32 = 22;
const SALT: u32 = 0x5912_E001;

#[derive(Clone, Copy)]
pub(super) struct Spec {
    x: i32,
    z: i32,
    base: i32,
    kind: u8,
    a: i32,
    b: i32,
    salt: u32,
    y0: i32,
    y1: i32,
}

pub(super) fn spawn(ctx: &Ctx, cx: i32, cz: i32) -> Option<Spec> {
    let (x, z, h) = site_anchor(ctx.s ^ SALT, CELL, cx, cz);
    if !ctx.inland(x, z, REACH) {
        return None;
    }
    let feats = ctx.feats(x, z);
    if feats[SPIRES] <= 0.0 && feats[HOODOOS] <= 0.0 {
        return None;
    }
    let col = ctx.column(x, z);
    let ds = if col.strata == Strata::Ice { 0.0 } else { col.feats[SPIRES] };
    let sand = col.surface == ctx.m.sand;
    let mesa = col.theme == ThemeId::Mesa;
    let dh = if sand || mesa { 0.0 } else { col.feats[HOODOOS] };
    let sum = ds + dh;
    if sum <= 0.0 || !keep(ctx.scale, sum.min(1.0), hash2(ctx.s ^ SALT ^ 0x33, cx, cz)) {
        return None;
    }
    let hoodoo = unit_pick(h, ds, sum);
    let arch = !hoodoo
        && h % 5 == 0
        && matches!(col.theme, ThemeId::Karst | ThemeId::Canyon | ThemeId::Badlands);
    let (kind, a, b, y1) = if arch {
        let half = 5 + (h % 16) as i32;
        (2, half, (half / 2).clamp(4, 14), half / 2 + 6)
    } else if hoodoo {
        let shaft = 12 + ((h >> 4) % 21) as i32;
        (1, shaft, 3, shaft + 3)
    } else {
        let height = 18 + ((h >> 4) % 35) as i32;
        (0, height, 0, height + 2)
    };
    let base = col.height;
    Some(Spec { x, z, base, kind, a, b, salt: h, y0: base - 2, y1: base + y1 })
}

fn unit_pick(h: u32, ds: f32, sum: f32) -> bool {
    if ds <= 0.0 {
        return true;
    }
    if sum <= ds {
        return false;
    }
    unit(hash2(h, 3, 5)) >= ds / sum
}

pub(super) fn bounds(s: &Spec) -> (i32, i32) {
    (s.y0, s.y1)
}

pub(super) fn paint(s: &Spec, m: &Materials, x: i32, y: i32, z: i32, _ground: i32) -> Option<Stamp> {
    let dx = x - s.x;
    let dz = z - s.z;
    let dy = y - s.base;
    if dx.abs().max(dz.abs()) > REACH {
        return None;
    }
    let hit = match s.kind {
        2 => arch(s, dx, dy, dz),
        1 => hoodoo(s, dx, dy, dz),
        _ => spire(s, dx, dy, dz),
    };
    hit.map(|dig| Stamp { id: band(m, y, s.salt), dig })
}

fn spire(s: &Spec, dx: i32, dy: i32, dz: i32) -> Option<bool> {
    if (-2..4).contains(&dy) && dx.abs().max(dz.abs()) <= 1 {
        return Some(dy < 0);
    }
    let dir = DIRS[((s.salt >> 8) % 8) as usize];
    let shift = if dy > 0 { dy / 8 } else { 0 };
    if dx == dir.0 * shift && dz == dir.1 * shift && dy >= 0 && dy < s.a {
        return Some(false);
    }
    None
}

fn hoodoo(s: &Spec, dx: i32, dy: i32, dz: i32) -> Option<bool> {
    let neck = s.a - 3;
    if dy >= -2 && dy < neck && dx.abs().max(dz.abs()) <= 1 {
        return Some(dy < 0);
    }
    if dy >= neck && dy < s.a && disk(dx, dz, s.b) {
        return Some(false);
    }
    if dy >= s.a && dy < s.a + 2 && disk(dx, dz, s.b - 1) {
        return Some(false);
    }
    None
}

fn arch(s: &Spec, dx: i32, dy: i32, dz: i32) -> Option<bool> {
    let (along, perp) = if s.salt & 1 == 0 { (dx, dz) } else { (dz, dx) };
    let half = s.a;
    if along.abs() > half || perp.abs() > 1 {
        return None;
    }
    let h2 = half * half;
    let ay = s.b * (h2 - along * along) / h2.max(1);
    if dy >= ay && dy < ay + 2 {
        return Some(dy < 1);
    }
    if along.abs() >= half - 1 && (-2..2).contains(&dy) {
        return Some(true);
    }
    None
}

fn band(m: &Materials, y: i32, salt: u32) -> super::BlockId {
    match (y + (salt as i32 & 7)).rem_euclid(6) {
        0 | 1 => m.limestone,
        2 | 3 => m.sandstone[(salt as usize + y.unsigned_abs() as usize) % 4],
        _ => m.slate,
    }
}
