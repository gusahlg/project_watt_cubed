//! Relaxed shapes of a solid cube of matter by `Π_g` (guide §17, stage 5).
//!
//! A homogeneous cube's equilibrium under its own gravity depends only on `Π_g = Gρ²L²/Y`, so
//! generation reads its shape from a table this binary computes with `mechanics::genesis` — the
//! same solver and law the runtime uses, run once at a resolution and patience world creation
//! cannot afford.
//!
//! `cargo run --release --bin genesis_table -- --pi <Π> [elements]` relaxes one cube and reports;
//! `cargo run --release --bin genesis_table -- --write <file>` computes and writes the table;
//! `--upgrade <from> <to>` re-derives the layout figures of an existing table from its nodes.

use std::time::Instant;

use glam::DVec3;
use project_watt_cubed::mechanics::genesis::{
    canonical_nodes, canonical_of, choose, entry_solved, pi_g, read_table, solve, write_table, Layout, Patience, Spec, Table, TableEntry, DATUM_SAMPLES,
    TABLE_ELEMENTS, TABLE_HALF,
};
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
        Some("--upgrade") => upgrade(args.get(1).expect("--upgrade <from> <to>"), args.get(2).expect("--upgrade <from> <to>")),
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
    let mut entries = Vec::new();
    for k in STEPS {
        let pi = 2f64.powf(k as f64 / 2.0);
        let start = Instant::now();
        let spec = Spec { half: HALF, cavity_half: None, matter: Params::from_yield(DENSITY, yield_for(pi)), elements: n, trace: 0, patience: Patience::TABLE };
        let solved = solve(&spec);
        let (nodes, worst) = symmetrised(&solved);
        let entry = shaped(n, k as f64 / 2.0, solved.report.converged, nodes);
        println!(
            "Π 2^{:>5.1} = {pi:>9.3}: roundness {:.4} tilt {:4.1}° radius {:.4} converged {} folded {} stages {:>2} its {:>6} asymmetry {worst:.1e} ({:.0} s)",
            k as f64 / 2.0,
            entry.roundness,
            entry.max_tilt,
            entry.radius,
            solved.report.converged,
            solved.folded,
            solved.stages,
            solved.report.iterations,
            start.elapsed().as_secs_f64()
        );
        entries.push(entry);
    }
    save(path, entries);
}

/// Re-derive every entry's layout figures from its nodes (a version-1 table, or a changed rule).
fn upgrade(from: &str, to: &str) {
    let bytes = std::fs::read(from).expect("read the table");
    let old = read_table(&bytes).expect("a table");
    let entries = old.entries.into_iter().map(|e| shaped(old.elements, e.log2_pi, e.converged, e.nodes)).collect();
    save(to, entries);
}

fn save(path: &str, entries: Vec<TableEntry>) {
    let table = Table { elements: TABLE_ELEMENTS, samples: DATUM_SAMPLES, entries };
    std::fs::write(path, write_table(&table)).expect("write the table");
    println!("wrote {path}");
}

/// Each canonical node averaged over its symmetry images (the solve is symmetric up to rounding),
/// in units of the half-size, and the largest deviation seen.
fn symmetrised(solved: &project_watt_cubed::mechanics::genesis::Solved) -> (Vec<DVec3>, f64) {
    let lat = &solved.lattice;
    let dims = solved.elements + 2;
    let m = dims / 2;
    let canon = canonical_nodes(m);
    let half_c = solved.elements as f64 * lat.cell as f64 / 2.0;
    let index: std::collections::HashMap<[usize; 3], usize> = canon.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let mut sum = vec![DVec3::ZERO; canon.len()];
    let mut count = vec![0.0f64; canon.len()];
    let mut worst = 0.0f64;
    for k in 0..=dims {
        for j in 0..=dims {
            for i in 0..=dims {
                let o = [i as i64 - m as i64, j as i64 - m as i64, k as i64 - m as i64];
                let (c, slot, sign) = canonical_of(o);
                let x = lat.nodes[lat.node(i, j, k)] / half_c;
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
    (sum.iter().zip(&count).map(|(s, c)| *s / *c).collect(), worst)
}

/// An entry with its layout figures, from symmetric canonical nodes.
fn shaped(n: usize, log2_pi: f64, converged: bool, nodes: Vec<DVec3>) -> TableEntry {
    let pi = 2f64.powf(log2_pi);
    let solved = entry_solved(n, &nodes, converged, pi, HALF);
    let out = choose(&solved, HALF, None);
    let (radius, offsets) = match &out.layout {
        Layout::Round { radius, datum, .. } => (radius / HALF, datum.offsets[..DATUM_SAMPLES * DATUM_SAMPLES].iter().map(|&o| (o as f64 / HALF) as f32).collect()),
        Layout::Cube => (0.0, vec![0.0; DATUM_SAMPLES * DATUM_SAMPLES]),
    };
    TableEntry { log2_pi, converged, max_tilt: out.max_tilt, roundness: out.roundness, radius, offsets, nodes }
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
