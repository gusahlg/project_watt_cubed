//! Creep in world time (guide §7.7, stage 6a): how a body keeps relaxing while people play.
//!
//! Generation solves a body straight to its rate-independent equilibrium (stresses at or under
//! yield everywhere). Afterwards matter moves (mining, building, a growing player-made body) and
//! the body answers at the pace of its material: Perzyna viscoplasticity, the overstress above yield
//! relaxing with the material's creep time `τ`. One world step of length `dt`:
//!
//! 1. elastic equilibrium with the plastic state frozen (dynamic relaxation without return, so the
//!    stress may stand above yield);
//! 2. every Gauss point's overstress relaxes by `1 − e^{−dt/τ}` (radial return of that fraction).
//!
//! The next step's equilibrium carries the stress the return shed into deformation. A body whose
//! stresses sit under yield does not move at all.

use super::solver::{Body, Relax, Report};

/// The outcome of one creep step.
#[derive(Clone, Copy, Debug, Default)]
pub struct Step {
    /// The equilibrium solve.
    pub report: Report,
    /// Largest overstress before the return, relative to yield (0: nothing flows).
    pub overstress: f64,
}

/// Advance `body` by `dt` world seconds (see the module notes). `opts.plastic_relax` is ignored.
pub fn step(body: &mut Body, dt: f64, opts: &Relax) -> Step {
    let report = body.relax(&Relax { plastic_relax: 0.0, ..*opts });
    let mut overstress = 0.0f64;
    let gp = body.stress.len() / body.params.len().max(1);
    for e in 0..body.params.len() {
        let p = body.params[e];
        if p.density <= 0.0 || !p.yield_stress.is_finite() {
            continue;
        }
        let fraction = 1.0 - (-dt / p.creep_time).exp();
        for g in e * gp..(e + 1) * gp {
            let eq = equivalent(&body.stress[g]);
            if eq <= p.yield_stress {
                continue;
            }
            overstress = overstress.max((eq - p.yield_stress) / p.yield_stress);
            let target = eq - fraction * (eq - p.yield_stress);
            let k = target / eq;
            for s in body.stress[g].iter_mut() {
                *s *= k;
            }
            body.plastic[g] += (eq - target) / (3.0 * p.shear.max(1e-300));
        }
    }
    Step { report, overstress }
}

/// von Mises equivalent of a deviator in Voigt order (xx, yy, zz, xy, yz, zx).
fn equivalent(s: &[f64; 6]) -> f64 {
    let dd = s[0] * s[0] + s[1] * s[1] + s[2] * s[2] + 2.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]);
    (1.5 * dd).sqrt()
}

#[cfg(test)]
mod tests {
    use glam::DVec3;

    use super::*;
    use crate::mechanics::lattice::Lattice;
    use crate::mechanics::material::Params;

    /// A weak cube of 6 matter elements in one element of void, under its own gravity.
    fn cube(yield_stress: f64) -> Body {
        let (n, cell) = (6usize, 1i64 << 16);
        let dims = n + 2;
        let matter = Params::from_yield(5.0, yield_stress);
        let void = Params::void(&matter);
        let params = (0..dims * dims * dims)
            .map(|e| {
                let ijk = [e % dims, (e / dims) % dims, e / (dims * dims)];
                if ijk.iter().any(|&q| q == 0 || q == dims - 1) { void } else { matter }
            })
            .collect();
        let lattice = Lattice::undeformed([0, 0, 0], cell, [dims; 3], DVec3::splat(-(dims as f64) * cell as f64 / 2.0));
        Body::new(lattice, params)
    }

    fn corner_drop(body: &Body) -> f64 {
        let l = &body.lattice;
        let (top, mid) = (l.dims[0] - 1, l.dims[0] / 2);
        let c = l.nodes[l.node(mid, mid, mid)];
        let start = (l.node(top, top, top), l.node(mid, top, mid));
        (l.nodes[start.0] - c).length() / (l.nodes[start.1] - c).length()
    }

    #[test]
    fn creep_flows_at_the_material_pace_and_stops_under_yield() {
        let opts = Relax { max_iterations: 4_000, tolerance: 1e-4, ..Relax::default() };
        // Π_g ≈ 20: yield well below the self-gravity stress.
        let pi = 20.0;
        let half = 3.0 * 65_536.0;
        let y = crate::gravity::G * 25.0 * half * half / pi;
        let mut coarse = cube(y);
        let mut fine = cube(y);
        let start = corner_drop(&coarse);
        // The same 120 world seconds in two steps and in four: the flow agrees to first order.
        for _ in 0..2 {
            step(&mut coarse, 60.0, &opts);
        }
        for _ in 0..4 {
            step(&mut fine, 30.0, &opts);
        }
        let (a, b) = (start - corner_drop(&coarse), start - corner_drop(&fine));
        assert!(a > 0.0 && b > 0.0, "the corners sink: {a} {b}");
        assert!((a - b).abs() < 0.35 * a.max(b), "step size changes the flow too much: {a} vs {b}");
        // A strong cube does not creep.
        let mut strong = cube(y * 1e4);
        let before = strong.lattice.nodes.clone();
        let s = step(&mut strong, 60.0, &opts);
        assert_eq!(s.overstress, 0.0);
        let moved = strong.lattice.nodes.iter().zip(&before).map(|(p, q)| (*p - *q).length()).fold(0.0, f64::max);
        assert!(moved < 1e-3 * half, "a strong cube only settles elastically: {moved}");
    }
}
