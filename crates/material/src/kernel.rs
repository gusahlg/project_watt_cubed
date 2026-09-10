//! The universal interaction law: F(δ) → Δ for two elements, and its bounded pairwise aggregate for
//! two configurations. No names, no recipes; only lattice geometry and the law's numbers.

use crate::configuration::Configuration;
use crate::element::{Element, D};
use crate::law::{Boundary, EventKind, Kernel, Law};

/// The change one interaction applies to one element's coordinates.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Delta(pub [i16; D]);

/// The outcome of one interaction between neighbouring configurations.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ReactionResult {
    /// The target's configuration after the event.
    pub target: Configuration,
    /// The origin's configuration after the event; `None` while the law leaves origins untouched (v0).
    pub origin: Option<Configuration>,
    /// True when `target` differs from the input.
    pub changed: bool,
    /// Sum of absolute coordinate changes over all target elements and axes.
    pub magnitude: u32,
}

/// Signed per-axis separation `a - b` under the law's boundary rule (Wrap takes the short way).
pub(crate) fn axis_delta(boundary: Boundary, a: u8, b: u8) -> i32 {
    let d = a as i32 - b as i32;
    match boundary {
        Boundary::Clamp => d,
        Boundary::Wrap => {
            if d > 127 {
                d - 256
            } else if d < -128 {
                d + 256
            } else {
                d
            }
        }
    }
}

/// Odd response curve: piecewise linear through the kernel's knots over |δ|, sign of δ.
pub(crate) fn response(kernel: &Kernel, delta: i32) -> i32 {
    let mag = delta.unsigned_abs().min(255) as i32;
    let mut r = kernel.knots[kernel.knots.len() - 1].1 as i32;
    for w in kernel.knots.windows(2) {
        let (d0, r0) = (w[0].0 as i32, w[0].1 as i32);
        let (d1, r1) = (w[1].0 as i32, w[1].1 as i32);
        if mag <= d1 {
            // d1 > d0 by Law::validate; integer interpolation, truncating toward zero.
            r = r0 + (r1 - r0) * (mag - d0) / (d1 - d0);
            break;
        }
    }
    if delta < 0 {
        -r
    } else {
        r
    }
}

/// Unscaled influence of element `a` on element `b` (before event strength and step bound), Q0.
fn influence_q0(law: &Law, a: Element, b: Element) -> [i32; D] {
    let mut r = [0i32; D];
    for i in 0..D {
        r[i] = response(&law.kernel, axis_delta(law.boundary, a.0[i], b.0[i]));
    }
    let mut out = [0i32; D];
    for i in 0..D {
        let mut acc = 0i32;
        for j in 0..D {
            acc = acc.wrapping_add(law.kernel.mixing[i][j] as i32 * r[j]);
        }
        out[i] = acc / 16;
    }
    out
}

/// F(δ) → Δ: the elementary law for one origin element acting on one target element, bounded by the
/// kernel's `max_step` (unit event strength).
pub fn element_influence(law: &Law, a: Element, b: Element) -> Delta {
    let raw = influence_q0(law, a, b);
    let s = law.kernel.max_step as i32;
    let mut d = [0i16; D];
    for i in 0..D {
        d[i] = raw[i].clamp(-s, s) as i16;
    }
    Delta(d)
}

/// Mean unscaled influence of every origin element on one target element; `None` for a void origin.
pub(crate) fn raw_influence(law: &Law, origin: &Configuration, target: Element) -> Option<[i32; D]> {
    let n = origin.len() as i32;
    if n == 0 {
        return None;
    }
    let mut acc = [0i32; D];
    for a in origin.elements() {
        let f = influence_q0(law, *a, target);
        for i in 0..D {
            acc[i] = acc[i].wrapping_add(f[i]);
        }
    }
    Some(acc.map(|v| v / n))
}

fn apply_axis(law: &Law, b: u8, step: i32) -> u8 {
    let v = b as i32 + step;
    let v = match law.boundary {
        Boundary::Clamp => v.clamp(0, 255),
        Boundary::Wrap => v.rem_euclid(256),
    };
    let q = law.quantum as i32;
    if q > 1 {
        ((v + q / 2) / q * q).clamp(0, 255) as u8
    } else {
        v as u8
    }
}

/// The reaction: every origin element influences every target element (pairwise influence), the mean
/// influence is scaled by the event's strength and bounded per axis by `max_step`, then applied under
/// the boundary rule (and the quantum, if any). Void on either side changes nothing.
pub fn interact(law: &Law, origin: &Configuration, target: &Configuration, event: EventKind) -> ReactionResult {
    interact_many(law, &[(origin, event)], target)
}

/// Several origins acting on one target at once (one generation of the scheduler): each origin's mean
/// influence is scaled by its own event strength, the origins are averaged (so the result does not
/// depend on their order), then bounded and applied exactly as [`interact`]. Void origins are skipped;
/// no origins or a void target changes nothing.
pub fn interact_many(law: &Law, origins: &[(&Configuration, EventKind)], target: &Configuration) -> ReactionResult {
    let max_step = law.kernel.max_step as i32;
    let live: Vec<(&Configuration, i32)> = origins
        .iter()
        .filter(|(o, _)| !o.is_void())
        .map(|(o, e)| (*o, law.events.0[*e as usize] as i32))
        .collect();
    let mut out = Vec::with_capacity(target.len());
    let mut magnitude = 0u32;
    let mut changed = false;
    for b in target.elements() {
        let mut nb = *b;
        if !live.is_empty() {
            let mut acc = [0i32; D];
            for (origin, strength) in &live {
                if let Some(raw) = raw_influence(law, origin, *b) {
                    for i in 0..D {
                        acc[i] = acc[i].wrapping_add(raw[i] * strength / 256);
                    }
                }
            }
            for i in 0..D {
                let step = (acc[i] / live.len() as i32).clamp(-max_step, max_step);
                let v = apply_axis(law, b.0[i], step);
                magnitude += (v as i32 - b.0[i] as i32).unsigned_abs();
                if v != b.0[i] {
                    changed = true;
                }
                nb.0[i] = v;
            }
        }
        out.push(nb);
    }
    let target = Configuration::new(out).expect("target length unchanged");
    ReactionResult { target, origin: None, changed, magnitude }
}
