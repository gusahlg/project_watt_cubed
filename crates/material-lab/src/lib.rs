//! Empirical search framework for the universal material law.
//!
//! Runs many local interactions of a [`material::Law`] and reports a scorecard
//! (similarity, determinism, fixed points, cascades, proliferation, families,
//! observations); searches resource space for stable worldgen regions; mutates
//! the v0 knots to explore the rule space. Simulation math is 32-bit integer;
//! f64 appears only in printed statistics.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod cascade;
mod regions;
mod rng;
mod score;
mod sweep;

pub use cascade::{run_one as run_cascade, CascadeRun, Grid};
pub use regions::{compatible, find_regions, Region, LABELS};
pub use rng::Rng;
pub use score::{
    measure_cascades, measure_determinism, measure_families, measure_fixed_points,
    measure_observations, measure_proliferation, measure_similarity, render, run_scorecard,
    Cascades, Determinism, Families, FixedPoints, Growth, Observations, Proliferation, Scale,
    Scorecard, Similarity, Verdict,
};
pub use sweep::{mutate_law, sweep, SweepHit};

use material::{Element, Law, LawError};

/// Chebyshev radius of a region's variants.
pub const SPREAD: u8 = 8;

/// Six variants of `centre` inside the Chebyshev ball of radius `spread`.
/// Determined by the centre coordinates alone (not the search seed).
pub fn variants(centre: Element, spread: u8) -> [Element; 6] {
    let mut h: u64 = 0xC0FF_EE00_0000_0001;
    for &b in &centre.0 {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    let mut rng = Rng::new(h);
    let s = spread as i32;
    let mut out = [centre; 6];
    for slot in out.iter_mut() {
        for _try in 0..16 {
            let mut e = centre;
            for axis in 0..material::D {
                let off = rng.inc(-s, s);
                e.0[axis] = (centre.0[axis] as i32 + off).clamp(0, 255) as u8;
            }
            if e != centre || spread == 0 {
                *slot = e;
                break;
            }
        }
    }
    out
}

/// Lowercase hex of `law.stamp()`, no `0x` prefix.
pub fn stamp_hex(law: &Law) -> String {
    let s = law.stamp();
    let mut out = String::with_capacity(s.len() * 2);
    for b in s {
        let _ = core::fmt::Write::write_fmt(&mut out, format_args!("{b:02x}"));
    }
    out
}

/// Inverse of [`stamp_hex`].
pub fn law_from_hex(hex: &str) -> Result<Law, String> {
    let hex = hex.trim();
    if hex.len() % 2 != 0 {
        return Err(format!("odd hex length {}", hex.len()));
    }
    let mut bytes = Vec::with_capacity(hex.len() / 2);
    let h = hex.as_bytes();
    let mut i = 0;
    while i < h.len() {
        let pair = std::str::from_utf8(&h[i..i + 2]).map_err(|_| "non-utf8 hex".to_string())?;
        let b = u8::from_str_radix(pair, 16).map_err(|_| format!("bad hex at {i}"))?;
        bytes.push(b);
        i += 2;
    }
    Law::from_stamp(&bytes).map_err(|e| match e {
        LawError::Length(n) => format!("stamp length {n}"),
        LawError::Version(v) => format!("stamp version {v}"),
        LawError::Invalid(s) => format!("invalid stamp: {s}"),
    })
}

#[cfg(test)]
mod tests;
