//! Cube bodies: which face owns a cell, the shared edge rim, and the deep bulk.
//!
//! Face-local `(u, a, v)` comes from [`FaceFrame`](crate::space::FaceFrame). Altitude `h = a − half`
//! is 0 on the face plane. The rim height is 3-D noise at one surface point per body, so the two
//! faces of an edge agree.

use super::cosmos::{Body, Shape, BULK_DENSITY};
use super::noise::{hash3, perlin3, smoothstep};
use super::trees::MAX_TREE_HEIGHT;
use super::{Materials, MAX_GROUND, MIN_GROUND};
use crate::block::registry::{BlockId, BlockRegistry};
use crate::coord::{ChunkCoord, Face};
use crate::space::FaceFrame;

/// Crust thickness: today's caves and mines, then the bulk.
pub(super) const CRUST: i32 = 350;
/// Blend the column into the rim within this many blocks of an edge.
pub(super) const RIM: i32 = 4_000;
/// `sky` is open inside this many blocks of an edge.
pub(super) const SKY_EDGE: i64 = 2_000;
/// Edge of a deep-mix cell. A 16³ chunk sits in one cell when the body centre is 64-aligned.
pub(super) const MIX: i64 = 64;

/// First face-local altitude that is above every tree (`MAX_GROUND + 16 + 1`).
pub(super) const TREE_CLEAR: i32 = MAX_GROUND + MAX_TREE_HEIGHT + 1;

/// Dominant axis of a body-relative vector. Ties break X, then Y, then Z; + wins.
/// The zero vector is [`Face::PosY`], matching [`Face::from_dominant`](Face::from_dominant).
#[inline]
pub(super) fn face_of(rel: [i64; 3]) -> Face {
    let (ax, ay, az) = (rel[0].abs(), rel[1].abs(), rel[2].abs());
    if ax == 0 && ay == 0 && az == 0 {
        return Face::PosY;
    }
    if ax >= ay && ax >= az {
        if rel[0] > 0 { Face::PosX } else { Face::NegX }
    } else if ay >= az {
        if rel[1] > 0 { Face::PosY } else { Face::NegY }
    } else if rel[2] > 0 {
        Face::PosZ
    } else {
        Face::NegZ
    }
}

#[inline]
pub(super) fn half_of(body: &Body) -> i64 {
    match body.shape {
        Shape::Cube { half } => half,
        _ => 0,
    }
}

/// `||rel||∞` at or below this is deeper than [`CRUST`] even on the lowest column.
#[inline]
pub(super) fn deep_limit(half: i64) -> i64 {
    half + MIN_GROUND as i64 - CRUST as i64 - 1
}

#[inline]
pub(super) fn in_deep(rel: [i64; 3], half: i64) -> bool {
    let lim = deep_limit(half);
    rel.iter().all(|c| c.abs() <= lim)
}

/// One noise seed per body, shared by every face so an edge reconstructs one rim.
#[inline]
pub(super) fn rim_seed(body: &Body) -> u32 {
    body.seed ^ 0x51D0_E3D6
}

/// Minimum `||p − centre||∞` over the inclusive box. This is the face altitude `a` of the
/// closest cell, so `a − half` is the lowest face-local height in the box.
pub(super) fn min_reach(centre: [i64; 3], lo: [i64; 3], hi: [i64; 3]) -> i64 {
    let mut m = 0i64;
    for a in 0..3 {
        let d = if centre[a] < lo[a] {
            lo[a] - centre[a]
        } else if centre[a] > hi[a] {
            centre[a] - hi[a]
        } else {
            0
        };
        m = m.max(d);
    }
    m
}

/// Inclusive cell box of a chunk.
pub(super) fn chunk_bounds(c: ChunkCoord) -> ([i64; 3], [i64; 3]) {
    let s = 16i64;
    let lo = [c.x as i64 * s, c.y as i64 * s, c.z as i64 * s];
    (lo, [lo[0] + 15, lo[1] + 15, lo[2] + 15])
}

pub(super) fn corners(lo: [i64; 3], hi: [i64; 3]) -> [[i64; 3]; 8] {
    std::array::from_fn(|i| {
        [
            if i & 1 == 0 { lo[0] } else { hi[0] },
            if i & 2 == 0 { lo[1] } else { hi[1] },
            if i & 4 == 0 { lo[2] } else { hi[2] },
        ]
    })
}

#[inline]
pub(super) fn normal_dot(centre: [i64; 3], face: Face) -> i64 {
    let (x, y, z) = face.normal();
    centre[0] * x as i64 + centre[1] * y as i64 + centre[2] * z as i64
}

pub(super) fn centre_i32(c: [i64; 3]) -> Option<(i32, i32, i32)> {
    Some((i32::try_from(c[0]).ok()?, i32::try_from(c[1]).ok()?, i32::try_from(c[2]).ok()?))
}

/// Body-frame tangents of world-frame tangents `(u, v)`.
pub(super) fn tangents(face: Face, centre: (i32, i32, i32), u: i32, v: i32) -> (i64, i64) {
    let (cu, _, cv) = FaceFrame::new(face).cell_to_local(centre);
    (i64::from(u) - i64::from(cu), i64::from(v) - i64::from(cv))
}

/// Body-relative vector of face-local `(u, a, v)`.
pub(super) fn local_to_rel(face: Face, u: i32, a: i32, v: i32) -> [i64; 3] {
    let (x, y, z) = FaceFrame::new(face).cell_to_world((u, a, v));
    [i64::from(x), i64::from(y), i64::from(z)]
}

/// World altitude of face-local height `h`, if it fits in `i32`.
pub(super) fn world_a(half: i64, h: i32, n_dot: i64) -> Option<i32> {
    i32::try_from(half + i64::from(h) + n_dot).ok()
}

/// Face-local height of a world altitude.
#[inline]
pub(super) fn face_h(half: i64, n_dot: i64, world_a: i32) -> i32 {
    (i64::from(world_a) - n_dot - half) as i32
}

/// Distance from the face edge, negative past the face square. `half − max(|u|, |v|)`.
#[inline]
pub(super) fn edge_inside(half: i64, ub: i64, vb: i64) -> i64 {
    half - ub.abs().max(vb.abs())
}

/// The point on the cube surface this column blends toward. Tangents clamp into the face square,
/// so both faces of an edge (and every wedge beside it) rebuild the same point.
pub(super) fn surface_point(face: Face, half: i64, ub: i64, vb: i64) -> [i64; 3] {
    let uc = ub.clamp(-half, half) as i32;
    let vc = vb.clamp(-half, half) as i32;
    let (x, y, z) = FaceFrame::new(face).cell_to_world((uc, half as i32, vc));
    [i64::from(x), i64::from(y), i64::from(z)]
}

fn rim_height(seed: u32, s: [i64; 3]) -> i32 {
    let n = perlin3(seed, s[0] as f64 / 900.0, s[1] as f64 / 900.0, s[2] as f64 / 900.0);
    let n2 = perlin3(seed ^ 0x0A11_CE00, s[0] as f64 / 280.0, s[1] as f64 / 280.0, s[2] as f64 / 280.0);
    let h = 64.0 + 48.0 * n + 18.0 * n2;
    (h.floor() as i32).clamp(MIN_GROUND, MAX_GROUND)
}

/// Column height: the face's own terrain, the rim past the face square, or a smoothstep blend
/// in the outer [`RIM`] blocks. `terrain` is returned unchanged at `inside ≥ RIM` so the home
/// +Y field is not re-rounded.
pub(super) fn blend_height(terrain: i32, seed: u32, face: Face, half: i64, ub: i64, vb: i64) -> i32 {
    let inside = edge_inside(half, ub, vb);
    if inside >= i64::from(RIM) {
        return terrain;
    }
    let rim = rim_height(seed, surface_point(face, half, ub, vb));
    if inside <= 0 {
        return rim;
    }
    let t = smoothstep(0.0, RIM as f32, inside as f32);
    let h = rim as f32 + (terrain as f32 - rim as f32) * t;
    (h.floor() as i32).clamp(MIN_GROUND, MAX_GROUND)
}

/// Two deep materials mixed so the mean amount is [`BULK_DENSITY`].
pub(super) struct Bulk {
    lo: BlockId,
    hi: BlockId,
    /// `hi` when `(hash >> 8) < cut` (24-bit threshold).
    cut: u32,
    seed: u32,
}

pub(super) fn choose_bulk(reg: &BlockRegistry, m: &Materials, seed: u32) -> Bulk {
    let ids = [
        m.rock[0], m.rock[1], m.rock[2], m.rock[3], m.sandstone[0], m.sandstone[1], m.sandstone[2],
        m.sandstone[3], m.deeprock, m.abyss, m.basalt, m.gravel, m.ochre,
    ];
    if let Some(&id) = ids.iter().find(|&&id| reg.amount(id) == BULK_DENSITY as u8) {
        return Bulk { lo: id, hi: id, cut: 0, seed };
    }
    let mut quiet: Option<(u8, BlockId, BlockId)> = None;
    let mut any: Option<(u8, BlockId, BlockId)> = None;
    for &lo in &ids {
        for &hi in &ids {
            let (a, b) = (reg.amount(lo), reg.amount(hi));
            if a >= BULK_DENSITY as u8 || b <= BULK_DENSITY as u8 {
                continue;
            }
            let gap = b - a;
            if any.map(|(g, _, _)| gap < g).unwrap_or(true) {
                any = Some((gap, lo, hi));
            }
            if reg.quiescent(lo, hi) && reg.quiescent(hi, lo) && quiet.map(|(g, _, _)| gap < g).unwrap_or(true) {
                quiet = Some((gap, lo, hi));
            }
        }
    }
    let Some((_, lo, hi)) = quiet.or(any) else {
        let id = ids.into_iter().min_by_key(|&id| (reg.amount(id) as i16 - BULK_DENSITY as i16).unsigned_abs()).unwrap();
        return Bulk { lo: id, hi: id, cut: 0, seed };
    };
    let (a, b) = (reg.amount(lo) as f64, reg.amount(hi) as f64);
    let frac = (BULK_DENSITY - a) / (b - a);
    let cut = (frac * 16_777_216.0).round() as u32;
    Bulk { lo, hi, cut, seed }
}

#[inline]
pub(super) fn bulk_id(bulk: &Bulk, body: &Body, rel: [i64; 3]) -> BlockId {
    if bulk.lo == bulk.hi || bulk.cut == 0 {
        return bulk.lo;
    }
    if bulk.cut >= 1 << 24 {
        return bulk.hi;
    }
    let h = hash3(
        bulk.seed ^ body.seed,
        rel[0].div_euclid(MIX) as i32,
        rel[1].div_euclid(MIX) as i32,
        rel[2].div_euclid(MIX) as i32,
    );
    if (h >> 8) < bulk.cut { bulk.hi } else { bulk.lo }
}

/// The one material of a deep chunk, when every coarse cell it touches agrees.
pub(super) fn bulk_uniform(bulk: &Bulk, body: &Body, cells: &[[i64; 3]]) -> Option<BlockId> {
    let id = bulk_id(bulk, body, cells[0]);
    cells[1..].iter().all(|&r| bulk_id(bulk, body, r) == id).then_some(id)
}
