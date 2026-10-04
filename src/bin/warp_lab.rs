//! Warp laboratory (guide §17, stage 5): relax a cube of matter under its own gravity and report the
//! shape and the cell geometry a builder would meet — at a face centre, halfway to an edge, on an
//! edge and at a corner — before any of it reaches the game.
//!
//! `cargo run --release --bin warp_lab -- [half] [elements] [yield] [density]`
//! (defaults: the start cube, 25,000,000 half size, 24 matter elements per axis, yield 1e5,
//! density 5).

use std::time::Instant;

use glam::{DMat3, DVec3};
use project_watt_cubed::mechanics::lattice::{local_jacobian, Lattice};
use project_watt_cubed::mechanics::material::Params;
use project_watt_cubed::mechanics::solver::{Body, Relax};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let half: f64 = args.first().and_then(|a| a.parse().ok()).unwrap_or(25_000_000.0);
    let n: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(24);
    let yield_stress: f64 = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(1e5);
    let density: f64 = args.get(3).and_then(|a| a.parse().ok()).unwrap_or(5.0);

    // Element edge: the power of two nearest the requested matter size per element.
    let want = 2.0 * half / n as f64;
    let cell = 1i64 << (want.log2().round() as u32);
    let dims = n + 2;
    let matter = Params::from_yield(density, yield_stress);
    let void = Params::void(&matter);
    let params: Vec<Params> = (0..dims * dims * dims)
        .map(|e| {
            let ijk = [e % dims, (e / dims) % dims, e / (dims * dims)];
            if ijk.iter().any(|&q| q == 0 || q == dims - 1) { void } else { matter }
        })
        .collect();
    let lattice = Lattice::undeformed([0, 0, 0], cell, [dims; 3], DVec3::splat(-(dims as f64) * cell as f64 / 2.0));
    let g = 4.349e-8f64;
    let l = n as f64 * cell as f64 / 2.0;
    println!(
        "cube: half {:.3e} ({} elements of {} blocks), yield {:.2e}, density {}, Π_g = Gρ²L²/Y = {:.3e}",
        l,
        n,
        cell,
        yield_stress,
        density,
        g * density * density * l * l / yield_stress
    );
    let mut body = Body::new(lattice, params);
    let start = Instant::now();
    let trace = std::env::var("WARP_TRACE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let iters = std::env::var("WARP_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(60_000);
    let report = body.relax(&Relax { max_iterations: iters, tolerance: 2e-4, trace, ..Relax::default() });
    let secs = start.elapsed().as_secs_f64();
    println!("relax: {report:?} in {secs:.1} s ({:.2} ms/iteration)", 1e3 * secs / report.iterations.max(1) as f64);

    let lat = &body.lattice;
    let c = lat.nodes[lat.node(dims / 2, dims / 2, dims / 2)];
    let top = dims - 1;
    let r = |i: usize, j: usize, k: usize| (lat.nodes[lat.node(i, j, k)] - c).length();
    let (face, edge, corner) = (r(dims / 2, top, dims / 2), r(dims / 2, top, top), r(top, top, top));
    println!(
        "shape: face {:.4e}, edge {:.4e}, corner {:.4e}  (corner/face {:.3}; a cube is 1.732, a ball 1.000)",
        face,
        edge,
        corner,
        corner / face
    );
    // Surface cells of the +Y face (the top matter layer, element row j = dims − 2).
    let j = dims - 2;
    println!("surface cells (+Y top layer): edge ratio (longest/shortest image of a unit cell), worst face angle off 90°, up tilt off gravity");
    for (label, i, k) in [
        ("face centre", dims / 2, dims / 2),
        ("halfway to an edge", dims / 2, (dims / 2 + top) / 2),
        ("edge", dims / 2, top - 1),
        ("corner", top - 1, top - 1),
    ] {
        let e = lat.element(i, j, k);
        let jac = local_jacobian(&lat.corners(e), DVec3::new(0.5, 1.0, 0.5)) * (1.0 / cell as f64);
        let (ratio, angle) = cell_shape(&jac);
        let up = jac.col(1).normalize();
        let radial = (lat.embed(DVec3::new((i as f64 + 0.5) * cell as f64, (j as f64 + 1.0) * cell as f64, (k as f64 + 0.5) * cell as f64)) - c).normalize();
        let tilt = up.dot(radial).clamp(-1.0, 1.0).acos().to_degrees();
        println!(
            "  {label:<20} ratio {ratio:.3}  angle {angle:5.1}°  up tilt {tilt:5.1}°  quality {:.3}",
            lat.certify(e)
        );
    }
    // Distribution over the whole top layer.
    let (mut ok, mut total) = (0usize, 0usize);
    for i in 1..top {
        for k in 1..top {
            let e = lat.element(i, j, k);
            let jac = local_jacobian(&lat.corners(e), DVec3::new(0.5, 1.0, 0.5)) * (1.0 / cell as f64);
            let (ratio, angle) = cell_shape(&jac);
            total += 1;
            if ratio <= 1.05 && angle <= 3.0 {
                ok += 1;
            }
        }
    }
    println!("top layer within the guide's prototype thresholds (ratio ≤ 1.05, angle ≤ 3°): {ok}/{total}");
    let p_centre = body.pressure(lat.element(dims / 2, dims / 2, dims / 2));
    let rho_g = 2.0 * std::f64::consts::PI / 3.0 * g * density * density * face * face;
    println!("centre pressure {p_centre:.3e} (uniform ball of the same radius: {rho_g:.3e})");
}

/// Longest over shortest image of the unit reference edges, and the worst angle (degrees) between
/// two images off 90°.
fn cell_shape(j: &DMat3) -> (f64, f64) {
    let cols = [j.col(0), j.col(1), j.col(2)];
    let lens = cols.map(|c| c.length());
    let ratio = lens.iter().copied().fold(0.0f64, f64::max) / lens.iter().copied().fold(f64::INFINITY, f64::min);
    let mut worst = 0.0f64;
    for (a, b) in [(0, 1), (1, 2), (0, 2)] {
        let cos = (cols[a].dot(cols[b]) / (lens[a] * lens[b])).clamp(-1.0, 1.0);
        worst = worst.max((cos.acos().to_degrees() - 90.0).abs());
    }
    (ratio, worst)
}
