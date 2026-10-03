//! Round bodies until curved charts: a noisy sphere, a 3-block crust, a kind's materials.
//!
//! Surface bumps are 3-D noise on the unit direction, clamped inside the body's relief so a chunk
//! the catalog calls empty stays empty. Integer coordinates, squared distances.

use super::cosmos::{Body, Kind, Shape, RELIEF};
use super::noise::perlin3;
use super::Materials;
use crate::block::registry::{AIR, BlockId};
use crate::world::generation::Classify;

const CRUST: f64 = 3.0;

/// `±0.3%` of the radius, never past the catalog's relief (minus one, so `dist > r + relief` is air).
fn amp(r: i64) -> f64 {
    (r as f64 * 0.003).min((RELIEF - 1) as f64)
}

fn shell(shape: Shape) -> (i64, Option<i64>) {
    match shape {
        Shape::Ball { r } => (r, None),
        Shape::Shell { outer, inner } => (outer, Some(inner)),
        Shape::Cube { .. } => (0, None),
    }
}

fn interior(kind: Kind, m: &Materials) -> BlockId {
    match kind {
        Kind::Ember => m.magma,
        Kind::Verdant => m.rock[0],
        Kind::Moon | Kind::Hollow => m.basalt,
        _ => m.basalt,
    }
}

/// Crust over the interior. Hollow lines both surfaces with frost then ice.
fn crust_block(kind: Kind, m: &Materials, from_out: f64, from_in: Option<f64>) -> BlockId {
    let skin = |d: f64| d <= CRUST;
    match kind {
        Kind::Verdant => {
            if from_out <= 1.0 {
                m.moss
            } else if skin(from_out) {
                m.soil
            } else {
                m.rock[0]
            }
        }
        Kind::Hollow => {
            let inner = from_in.unwrap_or(f64::MAX);
            if from_out <= 1.0 || inner <= 1.0 {
                m.frost
            } else if skin(from_out) || skin(inner) {
                m.ice
            } else {
                m.basalt
            }
        }
        Kind::Ember => {
            if skin(from_out) { m.basalt } else { m.magma }
        }
        _ => {
            if skin(from_out) { m.regolith } else { m.basalt }
        }
    }
}

/// Noise on the unit direction. The centre has no direction and no bump.
fn bump(body: &Body, rel: [i64; 3], radius: i64) -> (f64, f64) {
    let d2 = rel[0] as f64 * rel[0] as f64 + rel[1] as f64 * rel[1] as f64 + rel[2] as f64 * rel[2] as f64;
    if d2 == 0.0 {
        return (0.0, 0.0);
    }
    let dist = d2.sqrt();
    let n = perlin3(
        body.seed ^ 0xBA11_0000,
        rel[0] as f64 / dist,
        rel[1] as f64 / dist,
        rel[2] as f64 / dist,
    );
    (dist, amp(radius) * (n.clamp(-1.0, 1.0) as f64))
}

/// The cell at world `p`, or air outside the noisy surface (and inside a shell's cavity).
pub(super) fn block(body: &Body, p: [i64; 3], m: &Materials) -> BlockId {
    let (outer, inner) = shell(body.shape);
    if outer == 0 {
        return AIR;
    }
    let rel = [p[0] - body.centre[0], p[1] - body.centre[1], p[2] - body.centre[2]];
    let (dist, bump) = bump(body, rel, outer);
    let outer_r = outer as f64 + bump;
    if dist > outer_r {
        return AIR;
    }
    let inner_r = inner.map(|i| i as f64 + bump);
    if let Some(inn) = inner_r {
        if dist < inn {
            return AIR;
        }
    }
    // Centre of a ball: interior. `dist == 0` never reaches a shell's crust.
    if dist == 0.0 {
        return interior(body.kind, m);
    }
    let from_in = inner_r.map(|inn| dist - inn);
    crust_block(body.kind, m, outer_r - dist, from_in)
}

fn near_far2(centre: [i64; 3], lo: [i64; 3], hi: [i64; 3]) -> (f64, f64) {
    let mut near2 = 0.0;
    let mut far2 = 0.0;
    for a in 0..3 {
        let c = centre[a] as f64;
        let l = lo[a] as f64;
        let h = hi[a] as f64;
        let d = c - c.clamp(l, h);
        near2 += d * d;
        let far = (c - l).abs().max((c - h).abs());
        far2 += far * far;
    }
    (near2, far2)
}

/// `Uniform` when the chunk is wholly air or wholly interior. The crust is [`Classify::Mixed`].
pub(super) fn classify(body: &Body, lo: [i64; 3], hi: [i64; 3], m: &Materials) -> Classify {
    let (outer, inner) = shell(body.shape);
    if outer == 0 {
        return Classify::Mixed;
    }
    let a = amp(outer);
    let (near2, far2) = near_far2(body.centre, lo, hi);
    // One block of slack so a rounding error cannot call a crust chunk uniform.
    let out_air = outer as f64 + a + 1.0;
    if near2 > out_air * out_air {
        return Classify::Uniform(AIR);
    }
    if let Some(inn) = inner {
        let cav = inn as f64 - a - 1.0;
        if cav > 0.0 && far2 < cav * cav {
            return Classify::Uniform(AIR);
        }
    }
    let deep = outer as f64 - a - CRUST - 1.0;
    let buried = deep > 0.0 && far2 < deep * deep;
    let clear_inner = match inner {
        None => true,
        Some(inn) => {
            let lim = inn as f64 + a + CRUST + 1.0;
            near2 > lim * lim
        }
    };
    if buried && clear_inner {
        Classify::Uniform(interior(body.kind, m))
    } else {
        Classify::Mixed
    }
}
