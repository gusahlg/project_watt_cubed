//! Human-readable readout of a law stamp: the response curve and pair-regime fractions.

use std::fmt::Write;

use material::{response, Boundary, Law, D, KNOTS};

use crate::rng::Rng;
use crate::stamp_hex;

/// Default pair count for the `explain` command.
pub const EXPLAIN_PAIRS: u32 = 10_000;
/// Seed of the pair sample (fixed so a stamp prints the same twice).
pub const EXPLAIN_SEED: u64 = 0xE8A1_0001;

/// Per-distance regime of g(|δ|).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Regime {
    /// g = 0 outside the reactive well (dead zone or far).
    Inert,
    /// g < 0.
    Repulsive,
    /// g = 0 between the first and last non-zero response (the rest well).
    Rest,
    /// g > 0.
    Attractive,
}

impl Regime {
    fn name(self) -> &'static str {
        match self {
            Regime::Inert => "inert",
            Regime::Repulsive => "repulsive",
            Regime::Rest => "rest",
            Regime::Attractive => "attractive",
        }
    }
}

/// Curve, regimes and random-pair occupancy of one law.
#[derive(Clone, Debug)]
pub struct Explain {
    /// g(|δ|) for δ = 0..=255.
    pub curve: [i32; 256],
    /// Regime of each distance.
    pub regime: [Regime; 256],
    /// Element pairs sampled.
    pub pairs: u32,
    /// Axes counted (`pairs * D`).
    pub axes: u32,
    /// Axes in [`Regime::Inert`].
    pub inert: u32,
    /// Axes in [`Regime::Repulsive`].
    pub repulsive: u32,
    /// Axes in [`Regime::Rest`].
    pub rest: u32,
    /// Axes in [`Regime::Attractive`].
    pub attractive: u32,
}

/// Classify g(|δ|) for every distance. Interior zeros between the first and last
/// non-zero response are the rest well; zeros outside that span are inert.
pub fn regimes_of(curve: &[i32; 256]) -> [Regime; 256] {
    let first_nz = curve.iter().position(|&x| x != 0);
    let last_nz = curve.iter().rposition(|&x| x != 0);
    let mut out = [Regime::Inert; 256];
    for d in 0..256 {
        if curve[d] < 0 {
            out[d] = Regime::Repulsive;
        } else if curve[d] > 0 {
            out[d] = Regime::Attractive;
        } else if let (Some(a), Some(b)) = (first_nz, last_nz) {
            if d > a && d < b {
                out[d] = Regime::Rest;
            }
        }
    }
    out
}

fn axis_mag(boundary: Boundary, a: u8, b: u8) -> u8 {
    let d = a as i32 - b as i32;
    let w = match boundary {
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
    };
    w.unsigned_abs() as u8
}

/// Evaluate the curve, classify distances, count random element-pair axes.
pub fn explain(law: &Law, seed: u64, pairs: u32) -> Explain {
    let mut curve = [0i32; 256];
    for d in 0..=255 {
        curve[d] = response(&law.kernel, d as i32);
    }
    let regime = regimes_of(&curve);
    let mut rng = Rng::new(seed);
    let mut inert = 0u32;
    let mut repulsive = 0u32;
    let mut rest = 0u32;
    let mut attractive = 0u32;
    for _ in 0..pairs {
        let a = rng.element();
        let b = rng.element();
        for i in 0..D {
            match regime[axis_mag(law.boundary, a.0[i], b.0[i]) as usize] {
                Regime::Inert => inert += 1,
                Regime::Repulsive => repulsive += 1,
                Regime::Rest => rest += 1,
                Regime::Attractive => attractive += 1,
            }
        }
    }
    Explain {
        curve,
        regime,
        pairs,
        axes: pairs.saturating_mul(D as u32),
        inert,
        repulsive,
        rest,
        attractive,
    }
}

fn runs(regime: &[Regime; 256]) -> Vec<(Regime, u8, u8)> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < 256 {
        let r = regime[i];
        let start = i;
        i += 1;
        while i < 256 && regime[i] == r {
            i += 1;
        }
        out.push((r, start as u8, (i - 1) as u8));
    }
    out
}

fn ascii_curve(curve: &[i32; 256], knots: &[(u8, i16); KNOTS]) -> String {
    const XMAX: usize = 64;
    let mut lo = 0i32;
    let mut hi = 0i32;
    for d in 0..=XMAX {
        lo = lo.min(curve[d]);
        hi = hi.max(curve[d]);
    }
    if lo == hi {
        hi += 1;
    }
    let span = hi - lo;
    let height: usize = if span <= 24 {
        span as usize + 1
    } else {
        17
    };
    let height = height.max(2);
    let y_of = |v: i32| -> usize {
        let t = (hi - v) as i64 * (height as i64 - 1) / span as i64;
        (t as usize).min(height - 1)
    };
    let mut grid = vec![vec![b' '; XMAX + 1]; height];
    let y0 = y_of(0);
    for x in 0..=XMAX {
        grid[y0][x] = b'-';
    }
    for d in 0..=XMAX {
        grid[y_of(curve[d])][d] = b'*';
    }
    for &(kd, _) in knots {
        let x = kd as usize;
        if x <= XMAX {
            grid[y_of(curve[x])][x] = b'K';
        }
    }
    let mut s = String::new();
    s.push_str("g(|δ|)  * = value  K = knot  - = 0\n");
    for row in 0..height {
        let v = hi - (row as i32 * span) / (height as i32 - 1);
        let _ = write!(s, "{v:4} |");
        for &c in &grid[row] {
            s.push(c as char);
        }
        s.push('\n');
    }
    s.push_str("     +");
    for x in 0..=XMAX {
        s.push(if x % 8 == 0 { '+' } else { '-' });
    }
    s.push('\n');
    s.push_str("      ");
    let mut x = 0usize;
    while x <= XMAX {
        let lab = format!("{x}");
        s.push_str(&lab);
        x += 8;
        if x <= XMAX {
            for _ in lab.len()..8 {
                s.push(' ');
            }
        }
    }
    s.push('\n');
    let tail_lo = curve[65..=255].iter().copied().min().unwrap_or(0);
    let tail_hi = curve[65..=255].iter().copied().max().unwrap_or(0);
    if tail_lo == 0 && tail_hi == 0 {
        s.push_str("δ>64: g=0 (inert)\n");
    } else {
        let _ = writeln!(s, "δ>64: g in [{tail_lo}, {tail_hi}]");
    }
    s
}

fn pct(num: u32, den: u32) -> f64 {
    if den == 0 {
        0.0
    } else {
        100.0 * num as f64 / den as f64
    }
}

/// Print the curve, named knobs, regime spans and pair fractions.
pub fn render_explain(law: &Law, e: &Explain) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "======== material-lab explain ========");
    let _ = writeln!(s, "stamp {}", stamp_hex(law));
    let _ = writeln!(
        s,
        "version={}  boundary={:?}  max_step={}  quantum={}",
        law.version, law.boundary, law.kernel.max_step, law.quantum
    );
    s.push_str("mixing (Q4, 16=1.0):\n");
    for row in law.kernel.mixing {
        s.push_str("  ");
        for (j, m) in row.iter().enumerate() {
            if j > 0 {
                s.push(' ');
            }
            let _ = write!(s, "{m:3}");
        }
        s.push('\n');
    }
    let _ = writeln!(s, "events: {:?}", law.events.0);
    s.push_str("knots:\n");
    for (d, r) in law.kernel.knots {
        let _ = writeln!(s, "  {d:3}  {r:4}");
    }
    s.push('\n');
    s.push_str(&ascii_curve(&e.curve, &law.kernel.knots));
    s.push('\n');
    s.push_str("regimes of g(|δ|):\n");
    for (r, lo, hi) in runs(&e.regime) {
        let _ = writeln!(s, "  {:10} {lo}..{hi}", r.name());
    }
    let _ = writeln!(
        s,
        "random pairs: {} pairs × {D} axes = {} axes",
        e.pairs, e.axes
    );
    for (name, n) in [
        ("inert", e.inert),
        ("repulsive", e.repulsive),
        ("rest", e.rest),
        ("attractive", e.attractive),
    ] {
        let _ = writeln!(s, "  {name:10} {n:6}  {:5.1}%", pct(n, e.axes));
    }
    s.push_str("======================================");
    s
}

/// `explain` at the command's default seed and pair count, rendered.
pub fn explain_text(law: &Law) -> String {
    render_explain(law, &explain(law, EXPLAIN_SEED, EXPLAIN_PAIRS))
}
