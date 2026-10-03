//! The planet lab (guide §17 stage 0): measure cube-sphere charts where they hurt — face centre,
//! edges, corners, depth, seams and the core transition — and print the report as Markdown.
//! `cargo run --release --bin planet_lab`.

use glam::DVec3;
use project_watt_cubed::space::Face;
use project_watt_cubed::space::chart::{cells_per_face, quality, Chart, Map, Quality};

const MAPS: [Map; 3] = [Map::Gnomonic, Map::Equiangular, Map::Spherified];

fn cell(chart: &Chart, i: u32, j: u32, k: u32) -> Quality {
    quality(&chart.corners(i, j, k), chart.jacobian(i as f64 + 0.5, j as f64 + 0.5, k as f64 + 0.5))
}

fn row(name: &str, q: Quality) -> String {
    format!(
        "| {name} | {:.4} | {:.3} | {:.4} | {:.4} / {:.4} / {:.4} |",
        q.edge_ratio, q.skew_deg, q.volume, q.singular[0], q.singular[1], q.singular[2]
    )
}

/// Fraction of surface cells over the ordinary-building thresholds (guide §9.2: edge ratio 1.05, 3°).
fn exceed(chart: &Chart, samples: u32) -> (f64, f64) {
    let (mut over_ratio, mut over_skew) = (0u32, 0u32);
    for a in 0..samples {
        for b in 0..samples {
            let i = ((a as f64 + 0.5) / samples as f64 * (chart.n - 1) as f64) as u32;
            let j = ((b as f64 + 0.5) / samples as f64 * (chart.n - 1) as f64) as u32;
            let q = cell(chart, i, j, 0);
            over_ratio += (q.edge_ratio > 1.05) as u32;
            over_skew += (q.skew_deg > 3.0) as u32;
        }
    }
    let n = (samples * samples) as f64;
    (over_ratio as f64 / n, over_skew as f64 / n)
}

/// The angle (degrees) between a grid line leaving face A across the +u edge and its continuation
/// on face B, at edge parameter `j`.
fn seam_kink(map: Map, n: u32, r: f64, j: f64) -> f64 {
    let a = Chart { centre: DVec3::ZERO, face: Face::PosY, map, n, r0: r, layers: 1 };
    let b = Chart { centre: DVec3::ZERO, face: Face::PosX, map, n, r0: r, layers: 1 };
    let p = a.point(n as f64, j, 0.0);
    let before = p - a.point(n as f64 - 1.0, j, 0.0);
    let q = b.cell_of(p).expect("edge point lies on the neighbour");
    // Step away from the shared edge on B along its coordinate that is not the edge's.
    let along_edge_is_i = (q.x - 0.0).abs() > 1e-6 && (q.x - n as f64).abs() > 1e-6;
    let next = if along_edge_is_i {
        let s = if q.y < 1.0 { 1.0 } else { -1.0 };
        b.point(q.x, q.y + s, 0.0)
    } else {
        let s = if q.x < 1.0 { 1.0 } else { -1.0 };
        b.point(q.x + s, q.y, 0.0)
    };
    before.angle_between(next - p).to_degrees()
}

fn main() {
    println!("# Planet lab — cube-sphere charts\n");
    println!("Metrics per cell (guide §9.2): edge ratio (longest/shortest of 12 edges), skew (largest corner-angle");
    println!("deviation from 90°), volume (blocks³), singular values of the centre Jacobian. Surface cells are sized for");
    println!("one block of arc on average (`n = π/2 · R` cells per face edge).\n");
    for r in [1.0e5, 2.0e6, 8.0e6] {
        let n = cells_per_face(r);
        println!("## R = {r:.0e} (n = {n})\n");
        for map in MAPS {
            println!("### {map:?}\n");
            println!("| where | edge ratio | skew ° | volume | σ1 / σ2 / σ3 |\n|---|---|---|---|---|");
            for (label, depth) in [("surface", 0.0), ("5 % deep", 0.05), ("25 % deep", 0.25), ("half depth", 0.499)] {
                let chart = Chart { centre: DVec3::ZERO, face: Face::PosY, map, n, r0: r * (1.0 - depth) - 1.0, layers: 2 };
                let at = |fi: f64, fj: f64| ((fi * (n - 1) as f64) as u32, (fj * (n - 1) as f64) as u32);
                for (spot, (fi, fj)) in [("centre", (0.5, 0.5)), ("half to edge", (0.75, 0.5)), ("edge middle", (1.0, 0.5)), ("near corner", (0.97, 0.97)), ("corner", (1.0, 1.0))] {
                    let (i, j) = at(fi, fj);
                    println!("{}", row(&format!("{label}, {spot}"), cell(&chart, i, j, 0)));
                }
            }
            let chart = Chart { centre: DVec3::ZERO, face: Face::PosY, map, n, r0: r - 1.0, layers: 2 };
            let (fr, fs) = exceed(&chart, 120);
            println!("\nSurface cells over 1.05 edge ratio: **{:.1} %**; over 3° skew: **{:.1} %**.", fr * 100.0, fs * 100.0);
            println!(
                "Seam kink (grid-line bend across a chart edge): edge middle {:.4}°, quarter {:.3}°, near corner {:.3}°.\n",
                seam_kink(map, n, r, n as f64 * 0.5),
                seam_kink(map, n, r, n as f64 * 0.75),
                seam_kink(map, n, r, n as f64 * 0.98)
            );
        }
    }

    println!("## Core transition (guide §10.5)\n");
    println!("A Cartesian core cube of half-size `a = r_in / 2` joined to the innermost spherical band (radius `r_in`)");
    println!("by six mapped blocks whose cells interpolate linearly from the cube face to the sphere; the shell is");
    println!("`r_in − a` thick at a face centre but only `r_in − a√3` at the cube corners.\n");
    let r_in = 4096.0;
    let n = cells_per_face(r_in);
    let a = r_in * 0.5;
    // One radial cell per block where the shell is thickest (at a face centre).
    let layers = (r_in - a) as u32;
    let map = Map::Equiangular;
    let point = |i: f64, j: f64, k: f64| {
        let step = 2.0 / n as f64;
        let (xi, eta) = (-1.0 + i * step, -1.0 + j * step);
        let sphere = map.dir(xi, eta) * r_in;
        let q = std::f64::consts::FRAC_PI_4;
        let cube = DVec3::new((xi * q).tan(), 1.0, (eta * q).tan()) * a;
        cube + (sphere - cube) * (k / layers as f64)
    };
    println!("| where | edge ratio | skew ° | volume |\n|---|---|---|---|");
    for (spot, fi, fj) in [("centre", 0.5, 0.5), ("edge middle", 0.999, 0.5), ("corner", 0.999, 0.999)] {
        for (lk, k) in [("inner", 0u32), ("middle", layers / 2), ("outer", layers - 1)] {
            let (i, j) = ((fi * (n - 1) as f64) as u32, (fj * (n - 1) as f64) as u32);
            let c: [DVec3; 8] = std::array::from_fn(|c| point((i + (c as u32 & 1)) as f64, (j + (c as u32 >> 1 & 1)) as f64, (k + (c as u32 >> 2 & 1)) as f64));
            let q = quality(&c, glam::DMat3::IDENTITY);
            println!("| {spot}, {lk} | {:.3} | {:.2} | {:.3} |", q.edge_ratio, q.skew_deg, q.volume);
        }
    }
}
