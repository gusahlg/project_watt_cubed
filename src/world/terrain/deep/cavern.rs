//! The Deep: caverns a few hundred blocks across, one per cell of a coarse grid.
//!
//! The radius and the biome come from one coarse sample at the cell, not from a per-voxel field,
//! so the hollow is an exact ball. `confine` keeps that ball inside its cell, and a shaft is the
//! only part that leaves it. Dressing (mushrooms, crystals, a lake, roots, limestone, glow) stays
//! within [`MARGIN`] of the shell or of the geode's core, which is what `all_air` relies on.

use super::mushroom::{self, REACH};
use super::super::noise::hash2;
use super::{
    Ctx, DEEP_HI, Up, band_certain, confine, dist2, field_at, for_cells, for_shafts, hash_rel, hash_site, idx_i32,
    min_outward, place, plane_depth, roll, sphere_hits, tangent,
};
use crate::block::registry::{AIR, BlockId};

const CELL: i64 = 1_024;
const R_LO: i64 = 50;
const R_HI: i64 = 300;
const P: f32 = 0.42;
const SHAFT_P: f32 = 0.28;
const SALT: u32 = 0x0CA7_E401;
/// Dressing never reaches farther in than this, so a box inset by it is air.
const MARGIN: i64 = REACH + 4;
/// How many cells a shaft can climb from its cavern toward the crust.
const SHAFT_PAD: i64 = 6;
const FILL: f64 = 0.85;

struct Site {
    center: [i64; 3],
    r: i64,
    up: Up,
    kind: u32,
    shaft: bool,
    salt: u32,
}

fn may_host(ctx: &Ctx, idx: [i64; 3]) -> bool {
    let mid = std::array::from_fn(|a| idx[a] * CELL + CELL / 2);
    let pd = plane_depth(ctx.half, mid);
    let slack = CELL / 2;
    pd + slack > i64::from(super::super::cube::CRUST) && pd - slack <= i64::from(DEEP_HI)
}

fn site(ctx: &Ctx, idx: [i64; 3]) -> Option<Site> {
    if ctx.scale <= 0.0 || !may_host(ctx, idx) {
        return None;
    }
    let h = hash_site(ctx.seed ^ SALT, idx)?;
    if !roll(h, P * ctx.scale) {
        return None;
    }
    let origin: [i64; 3] = std::array::from_fn(|a| idx[a] * CELL);
    let mid = std::array::from_fn(|a| origin[a] + CELL / 2);
    let t = ((field_at(ctx.seed ^ SALT, mid) + 1.0) * 0.5).clamp(0.0, 1.0);
    let r = (R_LO + (t * (R_HI - R_LO) as f32) as i64).clamp(R_LO, R_HI);
    let center = std::array::from_fn(|a| confine(origin[a], CELL, r, super::super::noise::unit(hash3_axis(h, a))));
    let pd = plane_depth(ctx.half, center);
    if pd <= i64::from(super::super::cube::CRUST) || pd > i64::from(DEEP_HI) {
        return None;
    }
    let up = Up::of(center);
    let kind = hash3_axis(h, 3).wrapping_add((pd / 600) as u32) % 6;
    let shaft = roll(hash3_axis(h, 6), SHAFT_P);
    Some(Site { center, r, up, kind, shaft, salt: h })
}

fn hash3_axis(h: u32, a: usize) -> u32 {
    super::super::noise::hash3(h, a as i32, 1, 2)
}

fn contains(s: &Site, p: [i64; 3]) -> bool {
    dist2(p, s.center) <= s.r * s.r
}

fn core_r(s: &Site) -> i64 {
    12 + i64::from((s.salt >> 16) % 5)
}

fn lake_top(s: &Site) -> i64 {
    s.up.outward(s.center) - (s.r * 2) / 5
}

fn column_half(s: &Site, t0: i64, t1: i64) -> Option<i64> {
    let (c0, c1) = tangent(s.center, s.up.axis);
    let (d0, d1) = (t0 - c0, t1 - c1);
    let h2 = s.r * s.r - d0 * d0 - d1 * d1;
    (h2 >= 0).then_some((h2 as f64).sqrt().floor() as i64)
}

/// Mean air fraction of the deep shell at this `deep` scale.
pub(super) fn porosity(scale: f32) -> f64 {
    let s = f64::from(scale.clamp(0.0, 2.0));
    if s == 0.0 {
        return 0.0;
    }
    let sum = |n: i64| {
        if n <= 0 {
            0.0
        } else {
            let t = n as f64 * (n + 1) as f64 * 0.5;
            t * t
        }
    };
    let mean = (sum(R_HI) - sum(R_LO - 1)) / (R_HI - R_LO + 1) as f64;
    let vol = 4.0 / 3.0 * std::f64::consts::PI * mean * FILL;
    let cell = CELL as f64;
    vol / (cell * cell * cell) * f64::from(P) * s
}

/// Volume fraction of the deep shell occupied by floor lights.
///
/// The porosity above already counts those cells as void. Each site of [`super::glow::at`]
/// places one block, and the chance saturates at 1.
pub(super) fn light_phi(scale: f32) -> f64 {
    let s = f64::from(scale.clamp(0.0, 2.0));
    if s == 0.0 {
        return 0.0;
    }
    let sum_sq = |n: i64| {
        if n <= 0 {
            0.0
        } else {
            let n = n as f64;
            n * (n + 1.0) * (2.0 * n + 1.0) / 6.0
        }
    };
    let mean_r2 = (sum_sq(R_HI) - sum_sq(R_LO - 1)) / (R_HI - R_LO + 1) as f64;
    let area = std::f64::consts::PI * mean_r2;
    let cells = area / (super::glow::GAP as f64 * super::glow::GAP as f64);
    let dress = f64::from(s.min(1.0));
    cells / (CELL as f64 * CELL as f64 * CELL as f64) * f64::from(P) * s * dress
}

pub(super) fn hits(ctx: &Ctx, lo: [i64; 3], hi: [i64; 3]) -> bool {
    for_cells(lo, hi, CELL, |idx| site(ctx, idx).is_some_and(|s| sphere_hits(s.center, s.r, lo, hi)))
        || for_shafts(lo, hi, CELL, SHAFT_PAD, |idx| site(ctx, idx).is_some_and(|s| shaft_meets(&s, ctx, lo, hi)))
}

pub(super) fn all_air(ctx: &Ctx, lo: [i64; 3], hi: [i64; 3]) -> bool {
    if !band_certain(ctx.half, lo, hi, super::super::cube::CRUST, DEEP_HI) {
        return false;
    }
    for_cells(lo, hi, CELL, |idx| {
        let Some(s) = site(ctx, idx) else { return false };
        box_inside(&s, lo, hi, MARGIN) && clear(&s, lo, hi)
    })
}

fn box_inside(s: &Site, lo: [i64; 3], hi: [i64; 3], margin: i64) -> bool {
    if s.r <= margin {
        return false;
    }
    let rr = (s.r - margin) * (s.r - margin);
    (0..8).all(|i| {
        let p = [
            if i & 1 == 0 { lo[0] } else { hi[0] },
            if i & 2 == 0 { lo[1] } else { hi[1] },
            if i & 4 == 0 { lo[2] } else { hi[2] },
        ];
        dist2(p, s.center) <= rr
    })
}

fn clear(s: &Site, lo: [i64; 3], hi: [i64; 3]) -> bool {
    match s.kind {
        1 => !sphere_hits(s.center, core_r(s) + 1, lo, hi),
        2 => min_outward(s.up, lo, hi) > lake_top(s) + 6,
        _ => true,
    }
}

pub(super) fn block(ctx: &Ctx, rel: [i64; 3]) -> Option<BlockId> {
    let idx = std::array::from_fn(|a| rel[a].div_euclid(CELL));
    if let Some(s) = site(ctx, idx)
        && contains(&s, rel)
    {
        return Some(dress(ctx, &s, rel));
    }
    in_shaft(ctx, rel).then_some(AIR)
}

fn in_shaft(ctx: &Ctx, rel: [i64; 3]) -> bool {
    for_shafts(rel, rel, CELL, SHAFT_PAD, |idx| site(ctx, idx).is_some_and(|s| point_on_shaft(&s, ctx, rel)))
}

fn point_on_shaft(s: &Site, ctx: &Ctx, rel: [i64; 3]) -> bool {
    if !s.shaft || contains(s, rel) {
        return false;
    }
    let (t0, t1) = tangent(rel, s.up.axis);
    let (c0, c1) = tangent(s.center, s.up.axis);
    if (t0 - c0).abs() > 1 || (t1 - c1).abs() > 1 {
        return false;
    }
    let o = s.up.outward(rel);
    let co = s.up.outward(s.center);
    // From just above the cavern up to where the crust begins (the caller is already in the Deep).
    o > co && o <= ctx.half - i64::from(super::super::cube::CRUST)
}

fn shaft_meets(s: &Site, ctx: &Ctx, lo: [i64; 3], hi: [i64; 3]) -> bool {
    if !s.shaft {
        return false;
    }
    let (c0, c1) = tangent(s.center, s.up.axis);
    let co = s.up.outward(s.center);
    let cap = ctx.half - i64::from(super::super::cube::CRUST);
    let a = place(s.up.axis, s.up.sign, (c0 - 1, c1 - 1), co);
    let b = place(s.up.axis, s.up.sign, (c0 + 1, c1 + 1), cap.max(co));
    let lo_s = std::array::from_fn(|i| a[i].min(b[i]));
    let hi_s = std::array::from_fn(|i| a[i].max(b[i]));
    super::box_hits(lo_s, hi_s, lo, hi)
}

fn dress(ctx: &Ctx, s: &Site, rel: [i64; 3]) -> BlockId {
    let m = ctx.m;
    match s.kind {
        0 => fungal(ctx, s, rel).unwrap_or(AIR),
        1 => geode(ctx, s, rel),
        2 => magma(ctx, s, rel),
        3 => hung(s, rel, m).or_else(|| floor_light(ctx, s, rel, 0x51A7, m.glowcap)).unwrap_or(AIR),
        4 => cones(s, rel, m).or_else(|| floor_light(ctx, s, rel, 0xC04E, m.glowcap)).unwrap_or(AIR),
        _ => glow_biome(ctx, s, rel).unwrap_or(AIR),
    }
}

/// Radial gap from the shell is at most `reach` (no sqrt: `r − √d² ≤ reach`).
fn within(r: i64, d2: i64, reach: i64) -> bool {
    if d2 > r * r {
        return false;
    }
    if r <= reach {
        return true;
    }
    let inner = r - reach;
    d2 >= inner * inner
}

/// Glowcap (or magma) on the shell cell of this column: the floor, not a carpet inward of it.
fn floor_light(ctx: &Ctx, s: &Site, rel: [i64; 3], salt: u32, block: BlockId) -> Option<BlockId> {
    let d2 = dist2(rel, s.center);
    if !within(s.r, d2, 2) {
        return None;
    }
    let (t0, t1) = tangent(rel, s.up.axis);
    let half = column_half(s, t0, t1)?;
    let rise = s.up.outward(rel) - (s.up.outward(s.center) - half);
    if rise != 0 {
        return None;
    }
    super::glow::at(s.salt ^ salt, ctx.scale, t0, t1, 1.0, block)
}

fn fungal(ctx: &Ctx, s: &Site, rel: [i64; 3]) -> Option<BlockId> {
    let d2 = dist2(rel, s.center);
    if mushroom::out_of_reach(s.r, d2) {
        return None;
    }
    let shell = |t0: i64, t1: i64| column_half(s, t0, t1);
    let co = s.up.outward(s.center);
    if let Some(id) = mushroom::occupy(ctx.seed ^ s.salt, ctx.scale.max(0.25), rel, s.up, ctx.m, |t0, t1| {
        shell(t0, t1).map(|h| co - h)
    }, |t0, t1| shell(t0, t1).map(|h| co + h))
    {
        return Some(id);
    }
    // Mycelium on the shell. The mushroom early-out already proved this cell is near it.
    if !within(s.r, d2, 2) {
        return None;
    }
    let (t0, t1) = tangent(rel, s.up.axis);
    let half = column_half(s, t0, t1)?;
    let rise = s.up.outward(rel) - (co - half);
    if rise != 0 {
        return None;
    }
    super::glow::at(s.salt ^ 0x61F0, ctx.scale, t0, t1, 1.0, ctx.m.glowcap)
}

fn geode(ctx: &Ctx, s: &Site, rel: [i64; 3]) -> BlockId {
    let m = ctx.m;
    let d2 = dist2(rel, s.center);
    let cr = core_r(s);
    if d2 <= cr * cr {
        return match hash_rel(s.salt ^ 0x6E0D, rel) % 5 {
            0 => m.star,
            1 => m.glowshroom,
            _ => m.crystal,
        };
    }
    let lining = s.r - 4;
    if lining > 0 && d2 > lining * lining {
        // Crystal wall. The face that touches the hollow carries a glowshroom on the floor grid.
        if inner_face(s, rel, lining) {
            let (t0, t1) = tangent(rel, s.up.axis);
            if let Some(id) = super::glow::at(s.salt ^ 0x6E1D, ctx.scale, t0, t1, 1.0, m.glowshroom) {
                return id;
            }
        }
        return m.crystal;
    }
    AIR
}

/// Lining cell whose step toward the centre leaves the crystal and enters the hollow.
fn inner_face(s: &Site, rel: [i64; 3], lining: i64) -> bool {
    let mut axis = 0usize;
    let mut best = 0i64;
    for a in 0..3 {
        let d = (s.center[a] - rel[a]).abs();
        if d > best {
            best = d;
            axis = a;
        }
    }
    if best == 0 {
        return false;
    }
    let mut inward = rel;
    inward[axis] += (s.center[axis] - rel[axis]).signum();
    dist2(inward, s.center) <= lining * lining
}

fn magma(ctx: &Ctx, s: &Site, rel: [i64; 3]) -> BlockId {
    let m = ctx.m;
    let o = s.up.outward(rel);
    let top = lake_top(s);
    let lining = s.r - 5;
    let near_wall = lining > 0 && dist2(rel, s.center) > lining * lining;
    if o < top {
        if super::super::noise::unit(hash_rel(s.salt, rel)) < 0.07 { m.obsidian } else { m.magma }
    } else if o == top {
        // Open pools in the lid, so the lake's light reaches the air above it.
        let (t0, t1) = tangent(rel, s.up.axis);
        if super::glow::at(s.salt ^ 0x1A6A, ctx.scale, t0, t1, 0.45, m.magma).is_some() {
            m.magma
        } else {
            m.obsidian
        }
    } else if o <= top + 5 && near_wall {
        m.basalt
    } else {
        floor_light(ctx, s, rel, 0x5A0E, m.magma).unwrap_or(AIR)
    }
}

/// Roots hanging from the ceiling: darkwood inside, bark on the outside.
fn hung(s: &Site, rel: [i64; 3], m: &super::super::Materials) -> Option<BlockId> {
    const G: i64 = 12;
    let (t0, t1) = tangent(rel, s.up.axis);
    let o = s.up.outward(rel);
    let (i0, i1) = (t0.div_euclid(G), t1.div_euclid(G));
    let co = s.up.outward(s.center);
    for d1 in -1..=1 {
        for d0 in -1..=1 {
            let (Some(a0), Some(a1)) = (idx_i32(i0 + d0), idx_i32(i1 + d1)) else { continue };
            let h = hash2(s.salt ^ 0x4007, a0, a1);
            if super::super::noise::unit(h) > 0.55 {
                continue;
            }
            let p0 = (i0 + d0) * G + G / 2 + i64::from((h >> 8) % 5) - 2;
            let p1 = (i1 + d1) * G + G / 2 + i64::from((h >> 12) % 5) - 2;
            let Some(half) = column_half(s, p0, p1) else { continue };
            let len = 8 + i64::from((h >> 16) % 17);
            let drop = (co + half) - o;
            if drop < 0 || drop > len {
                continue;
            }
            let (dx, dz) = (t0 - p0, t1 - p1);
            let horiz = dx * dx + dz * dz;
            let rad: i64 = if drop < 2 { 2 } else { 1 };
            if horiz > rad * rad {
                continue;
            }
            // The free end glows; the rest of the root stays wood. Length is unchanged.
            if drop == len && horiz <= 1 {
                return Some(m.glowcap);
            }
            return Some(if horiz <= 1 { m.darkwood } else { m.bark });
        }
    }
    None
}

fn cones(s: &Site, rel: [i64; 3], m: &super::super::Materials) -> Option<BlockId> {
    const G: i64 = 8;
    let (t0, t1) = tangent(rel, s.up.axis);
    let o = s.up.outward(rel);
    let (i0, i1) = (t0.div_euclid(G), t1.div_euclid(G));
    let co = s.up.outward(s.center);
    for d1 in -1..=1 {
        for d0 in -1..=1 {
            let (Some(a0), Some(a1)) = (idx_i32(i0 + d0), idx_i32(i1 + d1)) else { continue };
            let h = hash2(s.salt ^ 0x57A1, a0, a1);
            if super::super::noise::unit(h) > 0.7 {
                continue;
            }
            let p0 = (i0 + d0) * G + G / 2;
            let p1 = (i1 + d1) * G + G / 2;
            let Some(half) = column_half(s, p0, p1) else { continue };
            let len = 6 + i64::from((h >> 16) % 11);
            let (dx, dz) = (t0 - p0, t1 - p1);
            let horiz = dx * dx + dz * dz;
            let cover = |gap: i64| {
                if !(0..len).contains(&gap) {
                    return false;
                }
                let rad = 1 + (len - gap) * 2 / len;
                horiz <= rad * rad
            };
            if cover((co + half) - o) || cover(o - (co - half)) {
                return Some(m.limestone);
            }
        }
    }
    None
}

fn glow_biome(ctx: &Ctx, s: &Site, rel: [i64; 3]) -> Option<BlockId> {
    let d2 = dist2(rel, s.center);
    // Floor disks sit on the shell; tips hang at most 6 from the ceiling. Deeper cells are air.
    if !within(s.r, d2, 6) {
        return None;
    }
    let (t0, t1) = tangent(rel, s.up.axis);
    let half = column_half(s, t0, t1)?;
    let co = s.up.outward(s.center);
    let o = s.up.outward(rel);
    let m = ctx.m;
    if o == co - half
        && let Some(id) = super::glow::at(s.salt ^ 0x6101, ctx.scale, t0, t1, 1.0, m.glowcap)
    {
        return Some(id);
    }
    let drop = (co + half) - o;
    if drop == 0 && super::super::noise::unit(hash_rel(s.salt ^ 0x6100, rel)) < 0.12 {
        return Some(m.glowcap);
    }
    super::glow::tip(s.salt ^ 0x6102, ctx.scale, t0, t1, drop, m.glowcap)
}

#[cfg(test)]
fn merge_ranges(mut rs: Vec<(i64, i64)>) -> Vec<(i64, i64)> {
    rs.retain(|(a, b)| a <= b);
    rs.sort_unstable();
    let mut out: Vec<(i64, i64)> = Vec::new();
    for (a, b) in rs {
        if let Some(last) = out.last_mut()
            && a <= last.1 + 1
        {
            last.1 = last.1.max(b);
            continue;
        }
        out.push((a, b));
    }
    out
}

/// Floor cells (air above a solid) and emissive blocks in the bands dressing can reach.
#[cfg(test)]
fn measure(ctx: &Ctx, s: &Site) -> super::Cover {
    // The whole disk. Lights outside a smaller floor sample would still reach it, so the sample
    // and the light scan share this radius.
    measure_span(ctx, s, s.r)
}

/// Like [`measure`], but floor cells only inside tangent radius `cap` (lights out to `cap + 12`).
#[cfg(test)]
fn measure_span(ctx: &Ctx, s: &Site, cap: i64) -> super::Cover {
    let (c0, c1) = tangent(s.center, s.up.axis);
    let co = s.up.outward(s.center);
    let mut floors = Vec::new();
    let mut lights = Vec::new();
    let r = s.r;
    let span = r.min(cap + 12);
    let cap2 = cap * cap;
    for t1 in (c1 - span)..=(c1 + span) {
        for t0 in (c0 - span)..=(c0 + span) {
            let (d0, d1) = (t0 - c0, t1 - c1);
            let dt = d0 * d0 + d1 * d1;
            if dt > span * span {
                continue;
            }
            let Some(half) = column_half(s, t0, t1) else { continue };
            let lo = co - half;
            let hi = co + half;
            // The shell band holds the floor, mushrooms, cones and the geode lining. A short
            // chord also holds the ceiling. The magma lid and the geode core sit deeper.
            let mut ranges = vec![(lo, (lo + 36).min(hi))];
            if hi - lo <= 48 {
                ranges.push((lo, hi));
            }
            if s.kind == 2 {
                let top = lake_top(s);
                ranges.push(((top - 14).max(lo), (top + 8).min(hi)));
            }
            if s.kind == 1 {
                let cr = core_r(s) + 2;
                let (d0, d1) = (t0 - c0, t1 - c1);
                if d0 * d0 + d1 * d1 <= cr * cr {
                    ranges.push(((co - cr).max(lo), (co + cr).min(hi)));
                }
            }
            for (a, b) in merge_ranges(ranges) {
                let below = cell_of(s, t0, t1, a - 1);
                let mut below_solid = !contains(s, below) || dress(ctx, s, below) != AIR;
                for o in a..=b {
                    let rel = cell_of(s, t0, t1, o);
                    if !contains(s, rel) {
                        below_solid = true;
                        continue;
                    }
                    let id = dress(ctx, s, rel);
                    if super::emits_light(ctx.m, id) {
                        lights.push(rel);
                    }
                    if id == AIR && below_solid && dt <= cap2 {
                        floors.push(rel);
                    }
                    below_solid = id != AIR;
                }
            }
        }
    }
    let near = super::NearLights::new(lights).count(&floors);
    super::Cover {
        kind: s.kind,
        r: s.r,
        center: s.center,
        up_axis: s.up.axis,
        up_sign: s.up.sign,
        floor: floors.len() as u32,
        near,
        step: 1,
    }
}

#[cfg(test)]
fn cell_of(s: &Site, t0: i64, t1: i64, o: i64) -> [i64; 3] {
    place(s.up.axis, s.up.sign, (t0, t1), o)
}

#[cfg(test)]
pub(super) fn survey(ctx: &Ctx, n: usize) -> Vec<super::Cover> {
    let mut out = Vec::new();
    let y0 = (ctx.half - i64::from(DEEP_HI)).div_euclid(CELL);
    let y1 = (ctx.half - i64::from(super::super::cube::CRUST)).div_euclid(CELL);
    for y in y0..=y1 {
        for z in -24..24 {
            for x in -24..24 {
                if let Some(s) = site(ctx, [x, y, z]) {
                    out.push(measure(ctx, &s));
                    if out.len() == n {
                        return out;
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
pub(super) fn survey_at(ctx: &Ctx, rel: [i64; 3]) -> Option<super::Cover> {
    let base = std::array::from_fn(|a| rel[a].div_euclid(CELL));
    if let Some(s) = site(ctx, base)
        && contains(&s, rel)
    {
        return Some(measure(ctx, &s));
    }
    let mut best: Option<(i64, Site)> = None;
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let idx = [base[0] + dx, base[1] + dy, base[2] + dz];
                let Some(s) = site(ctx, idx) else { continue };
                let d = dist2(rel, s.center);
                if d > (s.r + 48) * (s.r + 48) {
                    continue;
                }
                if best.as_ref().is_none_or(|(bd, _)| d < *bd) {
                    best = Some((d, s));
                }
            }
        }
    }
    best.map(|(_, s)| measure(ctx, &s))
}

/// The smallest cavern of each biome in the spawn window, measured cell by cell.
#[cfg(test)]
pub(super) fn measure_each_kind(ctx: &Ctx) -> Vec<super::Cover> {
    let mut best: [Option<([i64; 3], i64)>; 6] = [None; 6];
    let y0 = (ctx.half - i64::from(DEEP_HI)).div_euclid(CELL);
    let y1 = (ctx.half - i64::from(super::super::cube::CRUST)).div_euclid(CELL);
    for y in y0..=y1 {
        for z in -24..24 {
            for x in -24..24 {
                let idx = [x, y, z];
                let Some(s) = site(ctx, idx) else { continue };
                // True depth is plane depth plus the column height. A site whose tallest column
                // leaves the deep band is bulk on the chart; measure one the painter carves.
                let pd = plane_depth(ctx.half, s.center);
                if pd + i64::from(super::super::MAX_GROUND) > i64::from(DEEP_HI) {
                    continue;
                }
                let k = s.kind as usize;
                if best[k].is_none_or(|(_, r)| s.r < r) {
                    best[k] = Some((idx, s.r));
                }
            }
        }
    }
    // A 32-block cap is the open floor. The grid does not change across the disk, and the span
    // keeps lights that sit just outside the cap.
    best.into_iter().flatten().filter_map(|(idx, _)| site(ctx, idx).map(|s| measure_span(ctx, &s, 32))).collect()
}

#[cfg(test)]
pub(super) fn locate(ctx: &Ctx) -> Option<[i64; 3]> {
    let y0 = (ctx.half - i64::from(DEEP_HI)).div_euclid(CELL);
    let y1 = (ctx.half - i64::from(super::super::cube::CRUST)).div_euclid(CELL);
    for y in y0..=y1 {
        for z in -16..16 {
            for x in -16..16 {
                if let Some(s) = site(ctx, [x, y, z]) {
                    return Some(s.center);
                }
            }
        }
    }
    None
}
