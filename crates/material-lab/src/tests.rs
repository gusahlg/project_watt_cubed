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
    } else {
        assert!(f.families >= 1);
        assert!(f.largest >= 1);
        assert!(f.largest <= f.stable);
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
    let p = Configuration::single(Element([8, 16, 24, 32]));
    let t = Configuration::single(Element([200, 180, 40, 90]));
    let r = Configuration::single(Element([70, 210, 15, 250]));
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
    assert!(text.contains("observations"));
    assert!(text.contains("VERDICT"));
    assert!(card.pass_count() <= 7);
}
