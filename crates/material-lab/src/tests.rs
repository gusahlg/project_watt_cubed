use material::{interact, observe, Configuration, Element, EventKind, Law};

use crate::*;

fn tiny_rng(seed: u64) -> Rng {
    Rng::new(seed)
}

#[test]
fn similarity_on_tiny_input() {
    let law = Law::v0();
    let s = measure_similarity(&law, &mut tiny_rng(1), 40);
    assert_eq!(s.n, 40);
    assert!(s.p99 <= s.max);
    assert!(s.mean >= 0.0);
    assert!(s.max <= 2 * law.kernel.max_step as u32);
}

#[test]
fn determinism_on_tiny_input() {
    let d = measure_determinism(&Law::v0(), &mut tiny_rng(2), 20);
    assert_eq!(d.n, 20);
    assert_eq!(d.matched, 20);
    assert_eq!(d.verdict, Verdict::Pass);
}

#[test]
fn fixed_points_on_tiny_input() {
    let f = measure_fixed_points(&Law::v0(), &mut tiny_rng(3), 20);
    assert_eq!(f.n, 20);
    assert!(f.self_fixed <= 20);
    assert!(f.both_fixed <= f.self_fixed);
    assert!(f.both_fixed <= f.neighbour_fixed);
}

#[test]
fn observations_on_tiny_input() {
    let o = measure_observations(&Law::v0(), &mut tiny_rng(4), 40);
    assert_eq!(o.n, 40);
    assert!(o.liquid + o.soft <= 40);
    assert!(o.glowing <= 40);
    assert!(o.transparent <= 40);
}

#[test]
fn proliferation_on_tiny_input() {
    let p = measure_proliferation(&Law::v0(), 5, 8, 40);
    assert!(p.native.final_count >= 1);
    assert!(p.native.at_1k <= p.native.final_count);
    assert!(p.quantum4.final_count >= 1);
}

#[test]
fn families_on_tiny_input() {
    let f = measure_families(&Law::v0(), &mut tiny_rng(6), 40);
    assert_eq!(f.sample, 40);
    assert!(f.stable <= 40);
    if f.stable == 0 {
        assert_eq!(f.families, 0);
        assert_eq!(f.largest, 0);
        assert_eq!(f.mean, 0.0);
    } else {
        assert!(f.families >= 1);
        assert!(f.largest >= 1);
        assert!(f.largest <= f.stable);
        assert!((f.mean - f.stable as f64 / f.families as f64).abs() < 1e-9);
    }
}

#[test]
fn cascades_on_tiny_input() {
    let c = measure_cascades(&Law::v0(), &mut tiny_rng(7), 1, 4, 8);
    assert_eq!(c.runs, 1);
    assert_eq!(c.cells, 64);
    assert!(c.gens_max <= 8);
    assert!(c.cells_max <= 64);
}

#[test]
fn cascade_generation_is_jacobi() {
    let law = Law::v0();
    let p = Configuration::single(Element([10, 20, 30, 40]));
    let q = Configuration::single(Element([200, 10, 50, 90]));
    let r = Configuration::single(Element([40, 180, 20, 200]));
    let mut grid = Grid::new(4, Configuration::void());
    grid.set(0, 0, 0, p.clone());
    grid.set(1, 0, 0, q.clone());
    grid.set(2, 0, 0, r.clone());

    let q_from_p = interact(&law, &p, &q, EventKind::NewContact);
    let p_from_q = interact(&law, &q, &p, EventKind::NewContact);
    let r_from_q = interact(&law, &q, &r, EventKind::NewContact);
    assert!(q_from_p.changed, "test setup: P must mutate Q");
    let q_prime = q_from_p.target.clone();
    let gauss_p = interact(&law, &q_prime, &p, EventKind::NewContact).target;
    let gauss_r = interact(&law, &q_prime, &r, EventKind::NewContact).target;

    let changed = vec![grid.idx(0, 0, 0), grid.idx(1, 0, 0)];
    let _next = grid.generation(&law, &changed);

    assert_eq!(grid.cells[grid.idx(1, 0, 0)], q_from_p.target);
    assert_eq!(grid.cells[grid.idx(0, 0, 0)], p_from_q.target);
    assert_eq!(grid.cells[grid.idx(2, 0, 0)], r_from_q.target);
    // Gauss-Seidel would feed Q' back into the same generation.
    if gauss_p != p_from_q.target {
        assert_ne!(grid.cells[grid.idx(0, 0, 0)], gauss_p);
    }
    if gauss_r != r_from_q.target {
        assert_ne!(grid.cells[grid.idx(2, 0, 0)], gauss_r);
    }
}

#[test]
fn cascade_last_write_is_position_order() {
    let law = Law::v0();
    // Reactive-band distances (10..64) so NewContact mutates T under the v0 curve.
    let p = Configuration::single(Element([10, 20, 30, 40]));
    let t = Configuration::single(Element([200, 10, 50, 90]));
    let r = Configuration::single(Element([200, 10, 50, 50]));
    let mut grid = Grid::new(4, Configuration::void());
    grid.set(1, 0, 0, p.clone());
    grid.set(1, 0, 1, t.clone());
    grid.set(1, 0, 2, r.clone());

    let from_p = interact(&law, &p, &t, EventKind::NewContact);
    let from_r = interact(&law, &r, &t, EventKind::NewContact);
    assert!(from_p.changed && from_r.changed, "test setup: both origins mutate T");
    assert_ne!(from_p.target, from_r.target, "test setup: the two writes must differ");

    let changed = vec![grid.idx(1, 0, 0), grid.idx(1, 0, 2)];
    let _ = grid.generation(&law, &changed);
    // R has higher z, so a higher index: last write wins.
    assert_eq!(grid.cells[grid.idx(1, 0, 1)], from_r.target);
}

#[test]
fn variants_stay_inside_spread() {
    let mut rng = tiny_rng(9);
    for _ in 0..50 {
        let c = rng.element();
        for v in variants(c, SPREAD) {
            assert!(c.max_axis_distance(v) <= SPREAD as u32);
        }
    }
}

#[test]
fn stamp_hex_round_trips_v0() {
    let law = Law::v0();
    let hex = stamp_hex(&law);
    assert_eq!(hex.len(), law.stamp().len() * 2);
    assert_eq!(law_from_hex(&hex).unwrap(), law);
}

#[test]
fn find_regions_seed_42_v0() {
    let law = Law::v0();
    let found = find_regions(&law, 42, 1);
    assert!(!found.is_empty(), "search returned no regions");
    let labels: Vec<&str> = found.iter().map(|r| r.label).collect();
    for want in LABELS {
        if !labels.contains(&want) {
            eprintln!("FINDING: Law::v0() seed 42 produced no rest-stable {want} region");
        }
    }
    for r in &found {
        assert_eq!(r.spread, SPREAD);
        let line = format!("{r}");
        assert!(line.starts_with(r.label));
        assert!(line.contains("centre=["));
        assert!(line.ends_with("spread=8"));
        let o = observe(&law, &Configuration::single(r.centre));
        match r.label {
            "rock-like" => assert!(o.solid && o.transparency == 0 && o.hardness > 160),
            "soil-like" => assert!(o.solid && o.transparency == 0 && (90..=150).contains(&o.hardness)),
            "water-like" => assert!(o.liquid && o.transparency > 120),
            "glass-like" => assert!(o.solid && o.transparency > 160),
            "lamp-like" => assert!(o.emission >= 8),
            other => panic!("unknown label {other}"),
        }
        for other in &found {
            if other.centre == r.centre {
                continue;
            }
            assert!(
                compatible(&law, r, other),
                "{} centre {:?} reacts at rest with {} centre {:?}",
                r.label,
                r.centre,
                other.label,
                other.centre
            );
        }
    }
}

#[test]
fn tiny_scorecard_renders() {
    let card = run_scorecard(&Law::v0(), 11, Scale::tiny());
    let text = render(&card);
    assert!(text.contains("similarity"));
    assert!(text.contains("determinism"));
    assert!(text.contains("fixed points"));
    assert!(text.contains("cascades"));
    assert!(text.contains("proliferation"));
    assert!(text.contains("families"));
    assert!(text.contains("mean="));
    assert!(text.contains("observations"));
    assert!(text.contains("VERDICT"));
    assert!(card.pass_count() <= 7);
}

#[test]
fn mutate_law_searches_named_dimensions() {
    let base = Law::v0();
    let v0 = named(&base);
    assert_eq!(v0.rest_lo, 22);
    assert_eq!(v0.rest_hi, 26);
    assert_eq!(v0.peak_d, 40);
    assert_eq!(v0.fade, 64);
    assert_eq!(v0.coupling, 4);
    assert_eq!(v0.max_step, 6);

    let mut rng = Rng::new(1);
    let mut rest = std::collections::HashSet::new();
    let mut peaks = std::collections::HashSet::new();
    let mut fades = std::collections::HashSet::new();
    let mut couplings = std::collections::HashSet::new();
    let mut steps = std::collections::HashSet::new();
    for _ in 0..80 {
        let law = mutate_law(&base, &mut rng);
        law.validate().unwrap();
        let n = named(&law);
        assert!(n.rest_lo >= 20, "rest_lo {}", n.rest_lo);
        assert!(n.rest_hi <= 30, "rest_hi {}", n.rest_hi);
        assert!(n.rest_lo + 1 < n.rest_hi);
        assert_eq!(law.kernel.knots[3].1, 0);
        assert_eq!(law.kernel.knots[4].1, 0);
        assert!(n.peak_d > n.rest_hi);
        assert!(n.fade > n.peak_d);
        assert!(n.peak_r > 0);
        assert!(law.kernel.knots[2].1 < 0);
        assert_eq!(law.kernel.knots[7].1, 0);
        assert!((0..=8).contains(&n.coupling), "coupling {}", n.coupling);
        for i in 0..material::D {
            assert_eq!(law.kernel.mixing[i][i], 16);
            assert_eq!(law.kernel.mixing[i][(i + 1) % material::D], n.coupling);
            for j in 0..material::D {
                if j != i && j != (i + 1) % material::D {
                    assert_eq!(law.kernel.mixing[i][j], 0);
                }
            }
        }
        assert!((4..=8).contains(&n.max_step), "max_step {}", n.max_step);
        rest.insert((n.rest_lo, n.rest_hi));
        peaks.insert(n.peak_d);
        fades.insert(n.fade);
        couplings.insert(n.coupling);
        steps.insert(n.max_step);
        assert_eq!(law_from_hex(&stamp_hex(&law)).unwrap(), law);
        assert_ne!(law, base, "a mutation must differ from v0");
    }
    assert!(rest.len() >= 5, "rest-band variety {}", rest.len());
    assert!(peaks.len() >= 3, "peak variety {}", peaks.len());
    assert!(fades.len() >= 3, "fade variety {}", fades.len());
    assert!(couplings.len() >= 3, "coupling variety {}", couplings.len());
    assert!(steps.len() >= 3, "max_step variety {}", steps.len());
}

#[test]
fn sweep_rank_orders_named_goals() {
    let a = rank_tuple(4, 5, 10, 300, true, 3);
    let b = rank_tuple(3, 5, 10, 300, true, 7);
    assert!(a > b, "quiescent % outranks PASS count");

    let in_band = rank_tuple(4, 5, 10, 300, true, 3);
    let out_band = rank_tuple(4, 5, 5_000, 5_000, true, 3);
    assert!(in_band > out_band, "family count in 8-200 outranks a swarm");

    let mean_ok = rank_tuple(4, 5, 10, 300, true, 3);
    let mean_small = rank_tuple(4, 5, 10, 100, true, 3);
    assert!(mean_ok > mean_small, "mean size >= 20 outranks tiny families");
    assert!(family_target(10, 300));
    assert!(!family_target(10, 100));
    assert!(!family_target(5_000, 5_000));

    let native = rank_tuple(4, 5, 10, 300, true, 3);
    let quantum = rank_tuple(4, 5, 10, 300, false, 3);
    assert!(native > quantum, "native sub-linear outranks quantum-only");

    let more_pass = rank_tuple(4, 5, 10, 300, true, 5);
    let fewer_pass = rank_tuple(4, 5, 10, 300, true, 4);
    assert!(more_pass > fewer_pass);
}

#[test]
fn sweep_returns_top_five_sorted_by_new_rank() {
    let hits = sweep_at(123, 6, Scale::tiny());
    assert!(hits.len() <= 5);
    assert!(!hits.is_empty());
    for w in hits.windows(2) {
        assert!(rank_key(&w[0].card) >= rank_key(&w[1].card));
    }
    for h in &hits {
        assert_eq!(h.stamp_hex, stamp_hex(&h.law));
        assert_eq!(h.passes, h.card.pass_count());
        assert_eq!(h.families, h.card.families.families);
        h.law.validate().unwrap();
        let n = named(&h.law);
        assert!((4..=8).contains(&n.max_step));
        assert!((0..=8).contains(&n.coupling));
        assert!(n.rest_lo >= 20 && n.rest_hi <= 30);
        let text = render_hit(1, h);
        assert!(text.contains("quiescent="));
        assert!(text.contains("family-target="));
        assert!(text.contains("native-sublinear="));
        assert!(text.contains("stamp="));
    }
}

#[test]
fn explain_v0_curve_has_four_regimes() {
    let law = Law::v0();
    let e = explain(&law, 1, 4_000);
    assert_eq!(e.pairs, 4_000);
    assert_eq!(e.axes, 4_000 * material::D as u32);
    assert_eq!(e.inert + e.repulsive + e.rest + e.attractive, e.axes);
    assert!(e.inert > 0 && e.repulsive > 0 && e.rest > 0 && e.attractive > 0);
    assert_eq!(e.regime[0], Regime::Inert);
    assert_eq!(e.regime[5], Regime::Inert);
    assert_eq!(e.regime[16], Regime::Repulsive);
    assert_eq!(e.regime[24], Regime::Rest);
    assert_eq!(e.regime[40], Regime::Attractive);
    assert_eq!(e.regime[100], Regime::Inert);
    assert_eq!(e.curve[0], 0);
    assert!(e.curve[16] < 0);
    assert_eq!(e.curve[24], 0);
    assert!(e.curve[40] > 0);
    assert_eq!(e.curve[100], 0);

    let text = render_explain(&law, &e);
    assert!(text.contains("inert"));
    assert!(text.contains("repulsive"));
    assert!(text.contains("rest"));
    assert!(text.contains("attractive"));
    assert!(text.contains("g(|δ|)"));
    assert!(text.contains("rest"));
    assert!(text.contains(&stamp_hex(&law)));
    assert!(text.contains("K"));
    let full = explain_text(&law);
    assert!(full.contains("random pairs"));
    assert!(full.contains("mixing"));
}

#[test]
fn explain_regimes_mark_interior_zeros_as_rest() {
    let mut curve = [0i32; 256];
    curve[10] = -4;
    curve[40] = 6;
    let r = regimes_of(&curve);
    assert_eq!(r[0], Regime::Inert);
    assert_eq!(r[10], Regime::Repulsive);
    assert_eq!(r[20], Regime::Rest);
    assert_eq!(r[40], Regime::Attractive);
    assert_eq!(r[80], Regime::Inert);
}
