//! The reference suite of selective transfer v1, ported: exact geometry, cached-score algebra,
//! conservation, capacity, termination, permutation invariance, exhaustive swap optimality and the
//! behavioural regressions (the destructive exception and the held-out resistance panel).

use super::*;

fn el(c: [u8; 4]) -> Element {
    Element(c)
}

fn block(c: &[[u8; 4]]) -> Block {
    Block::new(&c.iter().map(|&c| el(c)).collect::<Vec<_>>()).unwrap()
}

fn potential(s: &Contact) -> i64 {
    let mut total = 0;
    for i in 0..s.len {
        for j in (i + 1)..s.len {
            if s.owner[i] == s.owner[j] {
                total += i64::from(fit_raw(s.elements[i], s.elements[j]));
            }
        }
    }
    total
}

fn check_gains(s: &Contact) {
    for i in 0..s.len {
        let mut holding = 0;
        let mut external = 0;
        for j in 0..s.len {
            if i == j {
                continue;
            }
            if s.owner[i] == s.owner[j] {
                holding += fit_raw(s.elements[i], s.elements[j]);
            } else {
                external += fit_raw(s.elements[i], s.elements[j]);
            }
        }
        assert_eq!(s.gain[i], external - holding);
        assert_eq!(s.holding[i], holding);
    }
}

fn canonical(s: &Contact) -> [Vec<Element>; 2] {
    let blocks = s.blocks();
    let mut out = [blocks[0].elements().to_vec(), blocks[1].elements().to_vec()];
    out[0].sort_unstable();
    out[1].sort_unstable();
    out
}

fn settle(mut s: Contact) -> Contact {
    for _ in 0..1550 {
        check_gains(&s);
        let [mut a, mut b] = s.blocks();
        a.canonicalize();
        b.canonicalize();
        let rebuilt = Contact::new(&a, &b);
        assert_eq!(s.peek(), rebuilt.peek());
        if s.counts.contains(&CAPACITY) {
            // Independent enumeration without the bound-based pruning.
            let mut unpruned = None;
            for i in 0..s.len {
                if s.counts[1 - s.owner[i] as usize] < CAPACITY {
                    s.consider(&mut unpruned, Decision { movement: Move::Transfer(i), gain: s.gain[i] });
                }
                if s.owner[i] != 0 {
                    continue;
                }
                for j in 0..s.len {
                    if s.owner[j] != 1 {
                        continue;
                    }
                    s.consider(
                        &mut unpruned,
                        Decision {
                            movement: Move::Swap(i, j),
                            gain: s.gain[i] + s.gain[j] - 2 * fit_raw(s.elements[i], s.elements[j]),
                        },
                    );
                }
            }
            assert_eq!(s.peek(), unpruned.map(|d| s.describe(d)));
        }
        let before = potential(&s);
        match s.step() {
            None => return s,
            Some(op) => {
                assert_eq!(potential(&s) - before, i64::from(op.raw_gain));
                assert!(op.raw_gain > s.threshold());
                assert!(s.counts.iter().all(|&n| n <= CAPACITY));
            }
        }
    }
    panic!("Exceeded the conservative fixed-pair operation bound")
}

struct Rng(u64);
impl Rng {
    fn byte(&mut self) -> u8 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u8
    }
    fn block(&mut self, n: usize) -> Block {
        let v: Vec<Element> = (0..n).map(|_| el([self.byte(), self.byte(), self.byte(), self.byte()])).collect();
        Block::new(&v).unwrap()
    }
}

#[test]
fn pair_geometry_is_exact_including_wrap_and_translation() {
    assert_eq!(fit_raw(el([0; 4]), el([0; 4])), 0);
    assert_eq!(fit_raw(el([0; 4]), el([64; 4])), 4 * QUANTUM);
    assert_eq!(fit_raw(el([0; 4]), el([128; 4])), -8 * QUANTUM);
    for d in 0..=255u8 {
        for t in 0..=255u8 {
            let a = el([t; 4]);
            let b = el([t.wrapping_add(d); 4]);
            assert_eq!(fit_raw(a, b), fit_raw(el([0; 4]), el([d; 4])));
            assert_eq!(fit_raw(a, b), fit_raw(b, a));
        }
        let x = f64::from(d) * std::f64::consts::TAU / 256.0;
        let exact = x.cos() - (2.0 * x).cos();
        let actual = f64::from(fit_raw(el([0; 4]), el([d; 4]))) / (4.0 * f64::from(QUANTUM));
        assert!((actual - exact).abs() <= 0.5 / f64::from(QUANTUM) + 1e-14);
    }
}

#[test]
fn construction_capacity_empty_and_identical_occurrences() {
    assert!(Block::new(&[el([0; 4]); 33]).is_none());
    let empty = Block::default();
    assert!(Contact::new(&empty, &empty).step().is_none());
    assert!(Contact::new(&block(&[[3; 4]]), &empty).step().is_none());
    let a = block(&[[19; 4]; 7]);
    let b = block(&[[19; 4]; 9]);
    let s = settle(Contact::new(&a, &b));
    assert_eq!(s.counts(), [7, 9]);
}

#[test]
fn exact_tie_does_not_trap_repulsive_two_element_mixture() {
    let a = block(&[[0; 4], [128; 4]]);
    let mut s = Contact::new(&a, &Block::default());
    let op = s.step().unwrap();
    assert_eq!(op.change, Change::Transfer { from: 0, element: el([0; 4]) });
    assert_eq!(op.raw_gain, 8 * QUANTUM);
    assert_eq!(s.counts(), [1, 1]);
    assert!(s.step().is_none());
}

#[test]
fn threshold_is_strict_and_gain_dominates_all_tie_keys() {
    let a = block(&[[0; 4]]);
    let s = Contact::new(&a, &a);
    let mut best = None;
    s.consider(&mut best, Decision { movement: Move::Transfer(0), gain: s.threshold() });
    assert!(best.is_none());
    s.consider(&mut best, Decision { movement: Move::Transfer(1), gain: s.threshold() + 1 });
    s.consider(&mut best, Decision { movement: Move::Transfer(0), gain: s.threshold() + 1 });
    assert!(matches!(best.unwrap().movement, Move::Transfer(0)));
    s.consider(&mut best, Decision { movement: Move::Transfer(1), gain: s.threshold() + 2 });
    assert!(matches!(best.unwrap().movement, Move::Transfer(1)));
}

#[test]
fn incremental_scores_conservation_termination_and_input_permutation() {
    let mut rng = Rng(0x20161002decaf);
    let mut tested = 0;
    for (na, nb) in [(0, 4), (1, 1), (2, 3), (4, 4), (8, 8), (16, 16), (1, 32), (32, 1), (16, 32), (32, 32)] {
        for _ in 0..32 {
            let a = rng.block(na);
            let b = rng.block(nb);
            let initial = Contact::new(&a, &b);
            let end = settle(initial.clone());
            let mut originals = initial.elements[..initial.len].to_vec();
            originals.sort_unstable();
            let blocks = end.blocks();
            let mut final_elements = [blocks[0].elements(), blocks[1].elements()].concat();
            final_elements.sort_unstable();
            assert_eq!(originals, final_elements);
            let mut ar = a.elements().to_vec();
            ar.reverse();
            let mut br = b.elements().to_vec();
            br.reverse();
            let reversed = settle(Contact::new(&Block::new(&ar).unwrap(), &Block::new(&br).unwrap()));
            assert_eq!(canonical(&end), canonical(&reversed));
            tested += 1;
        }
    }
    assert_eq!(tested, 320);
}

#[test]
fn full_capacity_choice_matches_exhaustive_potential_differences() {
    let mut rng = Rng(0xabcabc1234567);
    let mut s = Contact::new(&rng.block(32), &rng.block(32));
    for _ in 0..4 {
        let before = potential(&s);
        let mut largest = 0;
        for i in 0..s.len {
            if s.owner[i] != 0 {
                continue;
            }
            for j in 0..s.len {
                if s.owner[j] != 1 {
                    continue;
                }
                let mut candidate = s.clone();
                candidate.owner[i] = 1;
                candidate.owner[j] = 0;
                largest = largest.max(potential(&candidate) - before);
            }
        }
        let op = s.peek();
        if largest > i64::from(s.threshold()) {
            assert_eq!(i64::from(op.unwrap().raw_gain), largest);
            assert!(matches!(op.unwrap().change, Change::Swap { .. }));
            s.step().unwrap();
            check_gains(&s);
        } else {
            assert!(op.is_none());
            break;
        }
    }
}

pub(crate) const A: [[u8; 4]; 4] = [[73, 145, 162, 161], [71, 77, 157, 208], [34, 125, 217, 144], [8, 85, 210, 206]];
pub(crate) const E: [[u8; 4]; 4] = [[83, 135, 211, 195], [51, 125, 144, 147], [11, 72, 167, 145], [25, 80, 211, 204]];

#[test]
fn selective_counterexample_and_its_local_neighbourhood_survive_refinement() {
    let mut s = Contact::new(&block(&A), &block(&E));
    for step in 1..=4 {
        assert!(matches!(s.step().unwrap().change, Change::Transfer { from: 0, .. }));
        assert_eq!(s.counts(), [4 - step, 4 + step]);
    }
    assert!(s.step().is_none());
    for axis in 0..3 {
        let mut shifted = E;
        for e in &mut shifted {
            e[axis] = e[axis].wrapping_add(64);
        }
        assert!(Contact::new(&block(&A), &block(&shifted)).step().is_none());
    }
    for index in 0..4 {
        for axis in 0..4 {
            for offset in [1, 255] {
                let mut changed_a = A;
                changed_a[index][axis] = changed_a[index][axis].wrapping_add(offset);
                let mut changed_e = E;
                changed_e[index][axis] = changed_e[index][axis].wrapping_add(offset);
                assert_eq!(settle(Contact::new(&block(&changed_a), &block(&E))).counts()[0], 0);
                assert_eq!(settle(Contact::new(&block(&A), &block(&changed_e))).counts()[0], 0);
            }
        }
    }
}

#[test]
fn independent_held_out_cohesive_panel_preserves_resistance_profile() {
    let bytes = include_bytes!("../fixtures/cohesive_holdout.bin");
    assert_eq!(bytes.len(), 512 * 4 * 4);
    let a = block(&A);
    let mut unchanged = 0;
    for row in bytes.chunks_exact(16) {
        let mut e = [[0u8; 4]; 4];
        for (dst, src) in e.iter_mut().zip(row.chunks_exact(4)) {
            dst.copy_from_slice(src);
        }
        let initial = Contact::new(&a, &block(&e));
        if initial.peek().is_none() {
            unchanged += 1;
        }
        let end = settle(initial);
        assert!(end.owner[..4].iter().all(|&owner| owner == 0));
    }
    assert_eq!(unchanged, 508);
}

#[test]
fn react_once_matches_one_contact_step_and_canonicalizes() {
    let mut rng = Rng(0x5eed_0002);
    for n in [1, 3, 6, 12, 32] {
        for _ in 0..40 {
            let (a, b) = (rng.block(n), rng.block(n.min(31)));
            let mut s = Contact::new(&a, &b);
            let expect = s.step();
            let (mut a2, mut b2) = (a.clone(), b.clone());
            let got = react_once(&mut a2, &mut b2);
            assert_eq!(got, expect);
            if got.is_none() {
                assert_eq!(a2.elements(), a.elements());
                continue;
            }
            let [ea, eb] = canonical(&s);
            assert_eq!(a2.elements(), &ea[..]);
            assert_eq!(b2.elements(), &eb[..]);
            // The returned caches are exactly a fresh build of the same occurrences.
            assert_eq!(a2.holding(), Block::new(a2.elements()).unwrap().holding());
            assert_eq!(b2.holding(), Block::new(b2.elements()).unwrap().holding());
        }
    }
}

/// `cargo test -p material --release -- --ignored --nocapture measure` — the reference
/// microbenchmark's two kernel columns on this machine.
#[test]
#[ignore]
fn measure() {
    use std::hint::black_box;
    use std::time::Instant;
    let mut rng = Rng(0xdecaf20161002);
    for n in [1, 4, 8, 16, 32] {
        let blocks: Vec<_> = (0..128).map(|_| [rng.block(n), rng.block(n)]).collect();
        let start = Instant::now();
        let mut count = 0usize;
        while start.elapsed().as_secs_f64() < 0.2 {
            for [a, b] in &blocks {
                let (mut a, mut b) = (a.clone(), b.clone());
                black_box(react_once(black_box(&mut a), black_box(&mut b)));
                count += 1;
            }
        }
        let us = start.elapsed().as_secs_f64() * 1e6 / count as f64;
        println!("n={n:2}: react_once from cached blocks {:.3} ms / 1000 calls", us);
    }
}
