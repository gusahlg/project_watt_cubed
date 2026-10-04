//! Landmark shapes on a round chart. Placement (where, and how tall) lives in [`super::round`];
//! this file only answers the block at an offset from a placed mark, or how deep a dig goes.
//! Integer arithmetic only, so a mark is the same on every peer.

use super::noise::{hash2, hash3};
use super::round::Style;
use super::Materials;
use crate::block::registry::BlockId;

/// How far a mark may reach from its anchor, in cells. Anchors sit at least this far inside a chart.
pub(super) const REACH: i64 = 48;
/// Tallest cell a mark paints above its anchor, exclusive. The far field keeps a little more.
pub(super) const TALL: i64 = 120;
/// Cave band starts at depth 5, so a dig stops here.
pub(super) const DIG: i64 = 4;

pub(super) const V_GROVE: u8 = 0;
pub(super) const V_ARCH: u8 = 1;
pub(super) const V_TEPUI: u8 = 2;
pub(super) const V_BOULDER: u8 = 3;

pub(super) const I_ARCH: u8 = 0;
pub(super) const I_CLUSTER: u8 = 1;
pub(super) const I_CHANDELIER: u8 = 2;

pub(super) const O_CREVASSE: u8 = 0;
pub(super) const O_GEYSER: u8 = 1;
pub(super) const O_ARCH: u8 = 2;
pub(super) const O_RIDGE: u8 = 3;

pub(super) const E_FOUNTAIN: u8 = 0;
pub(super) const E_CONE: u8 = 1;
pub(super) const E_SHARD: u8 = 2;

pub(super) const M_PEAK: u8 = 0;
pub(super) const M_EJECTA: u8 = 1;
pub(super) const M_COLUMN: u8 = 2;
pub(super) const M_CRACK: u8 = 3;
pub(super) const M_DUNE: u8 = 4;
pub(super) const M_MESA: u8 = 5;
pub(super) const M_RILLE: u8 = 6;

/// One placed landmark. `a`, `b`, `c` are that family's sizes (height, radius, span).
#[derive(Clone, Copy, Debug)]
pub(super) struct Mark {
    pub i: i64,
    pub j: i64,
    pub base: i64,
    pub kind: u8,
    pub a: i64,
    pub b: i64,
    pub c: i64,
    pub salt: u32,
    /// Drawn by the far field. Boulder fields, shards, ejecta and cracks stay near.
    pub far: bool,
    /// Cells from the anchor the shape can touch. Copied from [`reach`] so a column can reject it once.
    pub reach: i64,
}

/// Families a style actually places, in kind order. Names are for tests.
#[cfg(test)]
pub(super) fn families(style: Style) -> &'static [(u8, &'static str)] {
    match style {
        Style::Verdant => &[(V_GROVE, "grove"), (V_ARCH, "arch"), (V_TEPUI, "tepui"), (V_BOULDER, "boulders")],
        Style::HollowInner => &[(I_ARCH, "crystal-arch"), (I_CLUSTER, "cluster"), (I_CHANDELIER, "chandelier")],
        Style::HollowOuter => &[(O_CREVASSE, "crevasse"), (O_GEYSER, "geyser"), (O_ARCH, "ice-arch"), (O_RIDGE, "ridge")],
        Style::Ember => &[(E_FOUNTAIN, "fountain"), (E_CONE, "cone"), (E_SHARD, "shards")],
        Style::Moon { tone: 1 } => &[(M_PEAK, "peak"), (M_EJECTA, "ejecta"), (M_COLUMN, "ice-column"), (M_CRACK, "cracks")],
        Style::Moon { tone: 2 } => &[(M_PEAK, "peak"), (M_EJECTA, "ejecta"), (M_DUNE, "dune"), (M_MESA, "mesa")],
        Style::Moon { .. } => &[(M_PEAK, "peak"), (M_EJECTA, "ejecta"), (M_RILLE, "rille")],
    }
}

/// The far field draws this family.
pub(super) fn far(style: Style, kind: u8) -> bool {
    match style {
        Style::Verdant => kind != V_BOULDER,
        Style::Ember => kind != E_SHARD,
        Style::Moon { .. } => kind != M_EJECTA && kind != M_CRACK,
        _ => true,
    }
}

/// Cells from the anchor the shape can touch. Stays ≤ [`REACH`] for every size we place.
pub(super) fn reach(style: Style, kind: u8, a: i64, b: i64, c: i64) -> i64 {
    match (style, kind) {
        (Style::Verdant, V_GROVE) => c.max(b / 2 + 4),
        (Style::Verdant, V_ARCH) | (Style::HollowInner, I_ARCH) | (Style::HollowOuter, O_ARCH) => a + 1,
        (Style::Verdant, V_TEPUI) => b + 3,
        (Style::Verdant, V_BOULDER) | (Style::Moon { .. }, M_EJECTA) | (Style::Moon { .. }, M_CRACK) => b,
        // A shard leans a couple of cells past the base it grows from.
        (Style::Ember, E_SHARD) => b + 2,
        (Style::HollowInner, I_CLUSTER) | (Style::HollowInner, I_CHANDELIER) => b + 1,
        (Style::HollowOuter, O_CREVASSE) => a.max(10),
        (Style::HollowOuter, O_GEYSER) | (Style::Ember, E_CONE) | (Style::Moon { .. }, M_PEAK) | (Style::Moon { .. }, M_MESA) => b,
        (Style::HollowOuter, O_RIDGE) => a.max(c),
        (Style::Ember, E_FOUNTAIN) => b + 5,
        (Style::Moon { .. }, M_COLUMN) => b + 4,
        (Style::Moon { .. }, M_DUNE) => a.max(c),
        (Style::Moon { .. }, M_RILLE) => a.max(6),
        _ => 0,
    }
}

/// Block at chart `(anchor + di, y, anchor + dj)`, or nothing. `local` is that column's surface.
pub(super) fn paint(style: Style, m: &Materials, k: &Mark, di: i64, dj: i64, y: i64, local: i64) -> Option<BlockId> {
    if di.abs() > REACH || dj.abs() > REACH {
        return None;
    }
    match (style, k.kind) {
        (Style::Verdant, V_GROVE) => tree(m, k, di, dj, y),
        (Style::Verdant, V_ARCH) => arch(m.rock[1], m.rock[0], None, k, di, dj, y),
        (Style::Verdant, V_TEPUI) => table(m, k, di, dj, y, local, true),
        (Style::Verdant, V_BOULDER) => lumps(k, di, dj, y, local, m.moss, m.rock[2]),
        (Style::HollowInner, I_ARCH) => arch(m.crystal, m.crystal, Some(m.glowshroom), k, di, dj, y),
        (Style::HollowInner, I_CLUSTER) => cluster(m, k, di, dj, y),
        (Style::HollowInner, I_CHANDELIER) => chandelier(m, k, di, dj, y),
        (Style::HollowOuter, O_GEYSER) => cone(k, di, dj, y, local, m.ice, m.frost, m.snow, true),
        (Style::HollowOuter, O_ARCH) => arch(m.ice, m.snow, None, k, di, dj, y),
        (Style::HollowOuter, O_RIDGE) => ridge(m, k, di, dj, y, local),
        (Style::Ember, E_FOUNTAIN) => fountain(m, k, di, dj, y),
        (Style::Ember, E_CONE) => cone(k, di, dj, y, local, m.cinder, m.ash, m.magma, false),
        (Style::Ember, E_SHARD) => shards(m, k, di, dj, y, local),
        (Style::Moon { tone }, M_PEAK) => peak(m, tone, k, di, dj, y),
        (Style::Moon { .. }, M_EJECTA) => lumps(k, di, dj, y, local, m.regolith, m.basalt),
        (Style::Moon { .. }, M_COLUMN) => ice_column(m, k, di, dj, y),
        (Style::Moon { tone: 2 }, M_DUNE) => dune(m, k, di, dj, y, local),
        (Style::Moon { .. }, M_MESA) => table(m, k, di, dj, y, local, false),
        _ => None,
    }
}

/// How many ground cells this column loses (0..=[`DIG`]). Crevasses, rilles and cracked plains only.
pub(super) fn digs(style: Style, k: &Mark, di: i64, dj: i64) -> i64 {
    if di.abs() > REACH || dj.abs() > REACH {
        return 0;
    }
    match (style, k.kind) {
        (Style::HollowOuter, O_CREVASSE) => crevasse(k, di, dj),
        (Style::Moon { tone: 1 }, M_CRACK) => cracks(k, di, dj),
        (Style::Moon { tone }, M_RILLE) if tone != 1 && tone != 2 => rille(k, di, dj),
        _ => 0,
    }
}

fn disk(di: i64, dj: i64, r: i64) -> bool {
    r >= 0 && di * di + dj * dj <= r * r
}

/// `w` cells wide, centred on the anchor (3, 4, 5 and 6 all come out that wide).
fn width_box(di: i64, dj: i64, w: i64) -> bool {
    let lo = -(w / 2);
    di >= lo && di < lo + w && dj >= lo && dj < lo + w
}

fn orient(di: i64, dj: i64, salt: u32) -> (i64, i64) {
    if salt & 1 == 0 { (di, dj) } else { (dj, di) }
}

fn arch_h(along: i64, half: i64, rise: i64) -> i64 {
    let aa = half * half;
    if aa == 0 || along.abs() > half {
        return 0;
    }
    rise * (aa - along * along) / aa
}

/// Centreline wander of a trench, ±2 cells, lerped every 8 so it stays continuous.
fn wobble(salt: u32, along: i64, lane: i64) -> i64 {
    let q = along.div_euclid(8);
    let f = along.rem_euclid(8);
    let at = |q: i64| (hash2(salt, q as i32, lane as i32) % 5) as i64 - 2;
    let (h0, h1) = (at(q), at(q + 1));
    h0 + (h1 - h0) * f / 8
}

fn tree(m: &Materials, k: &Mark, di: i64, dj: i64, y: i64) -> Option<BlockId> {
    let dy = y - k.base;
    if dy < 0 || dy >= k.a {
        return None;
    }
    if width_box(di, dj, k.b) {
        return Some(if dy < 6 { m.bark } else { m.timber });
    }
    if dy < 8 {
        let (adi, adj) = (di.abs(), dj.abs());
        let flare = k.b / 2 + 4;
        if adi.max(adj) <= flare && adi.min(adj) <= 1 && adi.max(adj) > k.b / 2 {
            return Some(m.bark);
        }
    }
    let layers = [(k.a * 55 / 100, k.c), (k.a * 74 / 100, (k.c - 3).max(6)), (k.a * 90 / 100, (k.c / 2).max(4))];
    for (at, rad) in layers {
        if (dy - at).abs() <= 1 && disk(di, dj, rad) {
            return Some(m.leaves);
        }
    }
    None
}

fn arch(body: BlockId, cap: BlockId, glow: Option<BlockId>, k: &Mark, di: i64, dj: i64, y: i64) -> Option<BlockId> {
    let dy = y - k.base;
    let (half, thick, rise) = (k.a, k.b, k.c);
    if dy < 0 || dy >= rise + thick {
        return None;
    }
    let (along, across) = orient(di, dj, k.salt);
    if along.abs() > half + 1 || across.abs() > thick {
        return None;
    }
    let curve = arch_h(along, half, rise);
    let ring = along.abs() <= half && dy >= curve && dy < curve + thick;
    let foot = along.abs() >= half - thick && along.abs() <= half && dy < thick + 3;
    if !ring && !foot {
        return None;
    }
    if glow.is_some() && along.abs() <= 1 && across.abs() <= 1 && dy + 1 >= rise + thick {
        return glow;
    }
    Some(if dy + 1 >= curve + thick { cap } else { body })
}

fn table(m: &Materials, k: &Mark, di: i64, dj: i64, y: i64, local: i64, forest: bool) -> Option<BlockId> {
    let top = k.base + k.a;
    let rad = k.b;
    if y < local || y >= top + if forest { 12 } else { 0 } {
        return None;
    }
    if y < k.base && k.base - y > 24 {
        return None;
    }
    if y >= top {
        return forest.then(|| cap_tree(m, k, di, dj, y - top, rad)).flatten();
    }
    if !disk(di, dj, rad) {
        return None;
    }
    if y >= top - 3 {
        return Some(if forest { m.moss } else { m.redsand });
    }
    if y >= top - 7 {
        return Some(if forest { m.soil } else { m.ochre });
    }
    Some(if forest { m.rock[0] } else { m.rock[2] })
}

fn cap_tree(m: &Materials, k: &Mark, di: i64, dj: i64, dy: i64, rad: i64) -> Option<BlockId> {
    if dy < 0 || dy >= 12 || di * di + dj * dj > (rad - 2) * (rad - 2) {
        return None;
    }
    let (qi, qj) = (di.div_euclid(5), dj.div_euclid(5));
    let (ci, cj) = (qi * 5 + 2, qj * 5 + 2);
    let (ex, ez) = (di - ci, dj - cj);
    let h = hash2(k.salt, qi as i32, qj as i32);
    if h % 3 != 0 {
        return None;
    }
    let th = 6 + (h % 5) as i64;
    if ex == 0 && ez == 0 && dy < th {
        return Some(m.timber);
    }
    if dy >= th - 3 && dy <= th && ex * ex + ez * ez <= 4 {
        return Some(m.leaves);
    }
    None
}

/// Squat mossy (or regolith) lumps on a 5-cell lattice inside the field.
fn lumps(k: &Mark, di: i64, dj: i64, y: i64, local: i64, soft: BlockId, hard: BlockId) -> Option<BlockId> {
    if !disk(di, dj, k.b) || y < local {
        return None;
    }
    let (qx, qz) = (di.div_euclid(5), dj.div_euclid(5));
    for oz in -1..=0 {
        for ox in -1..=0 {
            let (cx, cz) = (qx + ox, qz + oz);
            let h = hash3(k.salt, cx as i32, cz as i32, 7);
            if h % 3 != 0 {
                continue;
            }
            let (ex, ez) = (di - (cx * 5 + 2), dj - (cz * 5 + 2));
            let rad = 1 + (h % 2) as i64;
            let dy = y - local;
            if dy > rad {
                continue;
            }
            let rr = rad - dy / 2;
            if ex * ex + ez * ez <= rr * rr {
                return Some(if h % 2 == 0 { soft } else { hard });
            }
        }
    }
    None
}

fn cluster(m: &Materials, k: &Mark, di: i64, dj: i64, y: i64) -> Option<BlockId> {
    let dy = y - k.base;
    if dy < 0 || dy >= k.a {
        return None;
    }
    if disk(di, dj, 1) && dy >= k.a * 2 / 3 {
        return Some(m.glowcap);
    }
    if disk(di, dj, 2) && dy < k.a * 2 / 3 {
        return Some(m.glowshroom);
    }
    let s = k.b;
    let arms = [(s, 0i64), (-s, 0), (0, s), (0, -s), (s * 2 / 3, s * 2 / 3), (-s * 2 / 3, s * 2 / 3)];
    let spike = k.a - 4;
    for (ox, oz) in arms {
        let (ex, ez) = (di - ox, dj - oz);
        if ex.abs() <= 1 && ez.abs() <= 1 && dy < spike {
            return Some(m.crystal);
        }
        if ex == 0 && ez == 0 && dy >= spike - 2 && dy < k.a {
            return Some(m.glowcap);
        }
    }
    None
}

/// Arms step inward as they rise, so the spire leans toward its axis — storage +Y, the centre.
fn chandelier(m: &Materials, k: &Mark, di: i64, dj: i64, y: i64) -> Option<BlockId> {
    let dy = y - k.base;
    if dy < 0 || dy >= k.a {
        return None;
    }
    let ring = k.b * (k.a - dy) / k.a;
    if di.abs() <= 1 && dj.abs() <= 1 && dy >= k.a / 5 {
        return Some(if dy + 3 >= k.a { m.glowcap } else { m.crystal });
    }
    for sign in [ring, -ring] {
        let arm = |across: i64| across.abs() <= 1;
        let on = (arm(dj) && (di - sign).abs() <= 1) || (arm(di) && (dj - sign).abs() <= 1);
        if on {
            return Some(if dy + 4 >= k.a { m.glowshroom } else { m.crystal });
        }
    }
    None
}

fn cone(k: &Mark, di: i64, dj: i64, y: i64, local: i64, shell: BlockId, fill: BlockId, vent: BlockId, frozen: bool) -> Option<BlockId> {
    let (height, base_r) = (k.a, k.b);
    if y < local || y >= k.base + height {
        return None;
    }
    if y < k.base {
        return (k.base - y <= 16 && disk(di, dj, base_r)).then_some(fill);
    }
    let dy = y - k.base;
    let rad = (base_r * (height - dy) + height - 1) / height.max(1);
    if !disk(di, dj, rad) {
        return None;
    }
    let vent_r = (base_r / 5).max(2);
    let vent_d = (height / 5).max(3);
    if dy >= height - vent_d && disk(di, dj, vent_r) {
        if frozen && disk(di, dj, 1) {
            return Some(vent);
        }
        if !frozen && dy + 2 >= height {
            return Some(vent);
        }
        return None;
    }
    let inner = (rad - 2).max(0);
    Some(if disk(di, dj, inner) { fill } else { shell })
}

fn fountain(m: &Materials, k: &Mark, di: i64, dj: i64, y: i64) -> Option<BlockId> {
    let dy = y - k.base;
    if dy < 0 || dy >= k.a {
        return None;
    }
    let r = k.b;
    if disk(di, dj, r) || (dy + 3 >= k.a && disk(di, dj, r + 2)) {
        return Some(m.magma);
    }
    if (dy == k.a / 3 || dy == 2 * k.a / 3) && disk(di, dj, r + 2) {
        return Some(m.magma);
    }
    if dy < 5 && disk(di, dj, r + 5) {
        return Some(m.basalt);
    }
    None
}

// Leaning obsidian needles on a 4-cell lattice. The lean stays inside the extra cells of `reach`.
fn shards(m: &Materials, k: &Mark, di: i64, dj: i64, y: i64, local: i64) -> Option<BlockId> {
    if y < local {
        return None;
    }
    let dy = y - local;
    if dy >= 12 {
        return None;
    }
    let (qx, qz) = (di.div_euclid(4), dj.div_euclid(4));
    for oz in -1..=1 {
        for ox in -1..=1 {
            let (cx, cz) = (qx + ox, qz + oz);
            let (bx, bz) = (cx * 4 + 1, cz * 4 + 1);
            if !disk(bx, bz, k.b) {
                continue;
            }
            let h = hash3(k.salt, cx as i32, cz as i32, 4);
            if h % 3 != 0 {
                continue;
            }
            let sh = 4 + (h % 8) as i64;
            if dy >= sh {
                continue;
            }
            let lean = (h % 5) as i64 - 2;
            if di == bx + lean * dy / 4 && dj == bz {
                return Some(m.obsidian);
            }
        }
    }
    None
}

fn peak(m: &Materials, tone: u8, k: &Mark, di: i64, dj: i64, y: i64) -> Option<BlockId> {
    let dy = y - k.base;
    if dy < 0 || dy >= k.a {
        return None;
    }
    let rad = ((k.b * (k.a - dy) + k.a - 1) / k.a.max(1)).max(1);
    if !disk(di, dj, rad) {
        return None;
    }
    Some(match tone {
        1 if dy + 3 >= k.a => m.snow,
        1 => m.ice,
        2 => m.ochre,
        _ => {
            if dy + 2 >= k.a {
                m.regolith
            } else {
                m.rock[0]
            }
        }
    })
}

fn ice_column(m: &Materials, k: &Mark, di: i64, dj: i64, y: i64) -> Option<BlockId> {
    let dy = y - k.base;
    if dy < 0 || dy >= k.a {
        return None;
    }
    if dy < 5 && disk(di, dj, k.b + 4) {
        return Some(m.frost);
    }
    let rad = if dy > k.a * 3 / 4 { 1 } else { k.b };
    if !disk(di, dj, rad) {
        return None;
    }
    Some(if dy + 3 >= k.a { m.snow } else { m.ice })
}

fn dune(m: &Materials, k: &Mark, di: i64, dj: i64, y: i64, local: i64) -> Option<BlockId> {
    let (along, across) = orient(di, dj, k.salt);
    let (a, width, height) = (k.a, k.c, k.b);
    if along.abs() > a || across.abs() > width || y < local {
        return None;
    }
    let denom = a * a * width * width;
    if denom == 0 {
        return None;
    }
    let mut hf = height * (a * a - along * along) * (width * width - across * across) / denom;
    if across > width / 3 {
        hf /= 2;
    }
    if hf <= 0 || y >= k.base + hf || (y < k.base && k.base - y > 12) {
        return None;
    }
    Some(if y + 1 >= k.base + hf { m.redsand } else { m.ochre })
}

fn ridge(m: &Materials, k: &Mark, di: i64, dj: i64, y: i64, local: i64) -> Option<BlockId> {
    let (along, across) = orient(di, dj, k.salt);
    let (a, width, height) = (k.a, k.c, k.b);
    if along.abs() > a || y < local {
        return None;
    }
    let aa = a * a;
    if aa == 0 {
        return None;
    }
    let spine = height * (aa - along * along) / aa;
    if spine <= 0 {
        return None;
    }
    let half = (width * spine / height.max(1)).max(1);
    if across.abs() > half || y >= k.base + spine || (y < k.base && k.base - y > 16) {
        return None;
    }
    Some(if y + 1 >= k.base + spine { m.snow } else { m.ice })
}

fn crevasse(k: &Mark, di: i64, dj: i64) -> i64 {
    let (along, across) = orient(di, dj, k.salt);
    if along.abs() > k.a {
        return 0;
    }
    let lane = across.div_euclid(7);
    if !(-1..=1).contains(&lane) {
        return 0;
    }
    let pos = across - lane * 7 - wobble(k.salt, along, lane);
    // Five wide, so a far-field step of four still lands in a lane.
    if pos.abs() <= 2 { DIG } else { 0 }
}

fn cracks(k: &Mark, di: i64, dj: i64) -> i64 {
    if !disk(di, dj, k.b) {
        return 0;
    }
    let (q, r) = (di.div_euclid(6), dj.div_euclid(6));
    let h = hash2(k.salt, q as i32, r as i32);
    let (lx, lz) = (di.rem_euclid(6), dj.rem_euclid(6));
    if (h % 3 != 0 && lx == 0) || ((h >> 4) % 3 != 0 && lz == 0) { 2 } else { 0 }
}

fn rille(k: &Mark, di: i64, dj: i64) -> i64 {
    let (along, across) = orient(di, dj, k.salt);
    if along.abs() > k.a {
        return 0;
    }
    let center = wobble(k.salt ^ 0x51, along, 0);
    if (across - center).abs() <= k.b { DIG } else { 0 }
}
