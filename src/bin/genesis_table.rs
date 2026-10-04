//! Relaxed shapes of a solid cube of matter by `Π_g` (guide §17, stage 5).
//!
//! A homogeneous cube's equilibrium under its own gravity depends only on `Π_g = Gρ²L²/Y`, so
//! generation reads its shape from a table this binary computes with `mechanics::genesis` — the
//! same solver and law the runtime uses, run once at a resolution and patience world creation
//! cannot afford.
//!
//! `cargo run --release --bin genesis_table -- --pi <Π> [elements]` relaxes one cube and reports;
//! `cargo run --release --bin genesis_table -- --write <file>` writes the table source.

use std::time::Instant;

use glam::DVec3;
use project_watt_cubed::mechanics::genesis::{canonical_nodes, canonical_of, choose, pi_g, solve, Layout, Patience, Spec, TABLE_ELEMENTS, TABLE_HALF, TABLE_VERSION};
use project_watt_cubed::mechanics::material::Params;

const HALF: f64 = TABLE_HALF;
const DENSITY: f64 = 5.0;
/// Table entries: `Π_g = 2^(k/2)` for `k` in this range (¼ to 4096).
const STEPS: std::ops::RangeInclusive<i32> = -4..=24;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--pi") => {
            let pi: f64 = args.get(1).and_then(|a| a.parse().ok()).expect("--pi <value>");
            let n: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(16);
            let trace = std::env::var("TRACE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            one(pi, n, trace);
        }
        Some("--write") => write(args.get(1).expect("--write <file>")),
        _ => eprintln!("usage: genesis_table --pi <Π> [elements] | --write <file>"),
    }
}

/// Yield stress giving `Π_g = pi` for the table's cube.
fn yield_for(pi: f64) -> f64 {
    let probe = Params::from_yield(DENSITY, 1.0);
    pi_g(&probe, HALF) / pi
}

fn write(path: &str) {
    let n = TABLE_ELEMENTS;
    let dims = n + 2;
    let m = dims / 2;
    let canon = canonical_nodes(m);
    let mut out = Vec::new();
    out.extend_from_slice(b"PWCG");
    out.extend_from_slice(&TABLE_VERSION.to_le_bytes());
    out.extend_from_slice(&(n as u32).to_le_bytes());
    out.extend_from_slice(&(STEPS.count() as u32).to_le_bytes());
    for k in STEPS {
        let pi = 2f64.powf(k as f64 / 2.0);
        let start = Instant::now();
        let spec = Spec { half: HALF, cavity_half: None, matter: Params::from_yield(DENSITY, yield_for(pi)), elements: n, trace: 0, patience: Patience::TABLE };
        let solved = solve(&spec);
        let lat = &solved.lattice;
        let half_c = n as f64 * lat.cell as f64 / 2.0;
        // Average each canonical node over its symmetry images (the solve is symmetric up to
        // rounding), in units of the half-size.
        let mut sum = vec![DVec3::ZERO; canon.len()];
        let mut count = vec![0.0f64; canon.len()];
        let index: std::collections::HashMap<[usize; 3], usize> = canon.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let mut worst = 0.0f64;
        for kk in 0..=dims {
            for j in 0..=dims {
                for i in 0..=dims {
                    let o = [i as i64 - m as i64, j as i64 - m as i64, kk as i64 - m as i64];
                    let (c, slot, sign) = canonical_of(o);
                    let x = lat.nodes[lat.node(i, j, kk)] / half_c;
                    let mut canonical = DVec3::ZERO;
                    for a in 0..3 {
                        canonical[slot[a]] = sign[a] * x[a];
                    }
                    let id = index[&c];
                    if count[id] > 0.0 {
                        worst = worst.max((canonical - sum[id] / count[id]).length());
                    }
                    sum[id] += canonical;
                    count[id] += 1.0;
                }
            }
        }
        let out_of = choose(&solved, HALF, None);
        println!(
            "Π 2^{:>5.1} = {pi:>9.3}: roundness {:.4} tilt {:4.1}° converged {} folded {} stages {:>2} its {:>6} asymmetry {worst:.1e} ({:.0} s)",
            k as f64 / 2.0,
            out_of.roundness,
            out_of.max_tilt,
            solved.report.converged,
            solved.folded,
            solved.stages,
            solved.report.iterations,
            start.elapsed().as_secs_f64()
        );
        out.extend_from_slice(&((k as f32) / 2.0).to_le_bytes());
        out.extend_from_slice(&u32::from(solved.report.converged).to_le_bytes());
        for (s, c) in sum.iter().zip(&count) {
            let p = *s / *c;
            for a in 0..3 {
                out.extend_from_slice(&(p[a] as f32).to_le_bytes());
            }
        }
    }
    std::fs::write(path, out).expect("write the table");
    println!("wrote {path}");
}

fn one(pi: f64, n: usize, trace: usize) {
    let start = Instant::now();
    let patience = if std::env::var_os("TABLE").is_some() { Patience::TABLE } else { Patience::QUICK };
    let spec = Spec { half: HALF, cavity_half: None, matter: Params::from_yield(DENSITY, yield_for(pi)), elements: n, trace, patience };
    let solved = solve(&spec);
    let out = choose(&solved, HALF, None);
    let layout = match &out.layout {
        Layout::Cube => "cube".to_string(),
        Layout::Round { radius, datum, .. } => {
            let (lo, hi) = datum.range();
            format!("round r/half {:.4} relief {:+.4}..{:+.4}", radius / HALF, lo / HALF, hi / HALF)
        }
    };
    println!(
        "Π {pi:.3e}: {layout}  roundness {:.4} tilt {:.1}° converged {} folded {} hydrostatic {} its {} stages {} residual {:.2e} quality {:.3} ({:.1} s)",
        out.roundness,
        out.max_tilt,
        out.report.converged,
        out.folded,
        out.hydrostatic,
        out.report.iterations,
        solved.stages,
        out.report.residual,
        out.report.min_quality,
        start.elapsed().as_secs_f64()
    );
}
