//! Search resource space for stable single-element worldgen regions.

use material::{interact, observe, Configuration, Element, EventKind, Law, Observation};

use crate::rng::Rng;
use crate::{variants, SPREAD};

/// Worldgen labels the lab searches for. Printed as the first token of each region line.
pub const LABELS: [&str; 5] = ["rock-like", "soil-like", "water-like", "glass-like", "lamp-like"];

const ROCK: usize = 0;
const SOIL: usize = 1;
const WATER: usize = 2;
const GLASS: usize = 3;
const LAMP: usize = 4;

/// A stable single-element region: a centre, six spread-8 variants, and a label.
#[derive(Clone, Debug)]
pub struct Region {
    /// One of [`LABELS`].
    pub label: &'static str,
    /// Centre element.
    pub centre: Element,
    /// Chebyshev radius of the variants (always 8).
    pub spread: u8,
    /// Six variants of the centre, a function of the centre coordinates.
    pub variants: [Element; 6],
}

impl Region {
    /// Centre plus variants.
    pub fn members(&self) -> impl Iterator<Item = Element> + '_ {
        std::iter::once(self.centre).chain(self.variants)
    }
}

impl std::fmt::Display for Region {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} centre=[{},{},{},{}] spread={}",
            self.label, self.centre.0[0], self.centre.0[1], self.centre.0[2], self.centre.0[3], self.spread
        )
    }
}

fn matches(label: usize, o: &Observation) -> bool {
    match label {
        ROCK => o.solid && o.transparency == 0 && o.hardness > 160,
        SOIL => o.solid && o.transparency == 0 && o.hardness >= 90 && o.hardness <= 150,
        WATER => o.liquid && o.transparency > 120,
        GLASS => o.solid && o.transparency > 160,
        LAMP => o.emission >= 8,
        _ => false,
    }
}

fn cfg(e: Element) -> Configuration {
    Configuration::single(e)
}

fn reacts(law: &Law, a: Element, b: Element) -> bool {
    interact(law, &cfg(a), &cfg(b), EventKind::NewContact).changed
}

/// Two regions are rest-compatible: every member of each is a fixed point under contact
/// with the other region's centre (both directions).
pub fn compatible(law: &Law, a: &Region, b: &Region) -> bool {
    for m in a.members() {
        if reacts(law, b.centre, m) || reacts(law, m, b.centre) {
            return false;
        }
    }
    for m in b.members() {
        if reacts(law, a.centre, m) || reacts(law, m, a.centre) {
            return false;
        }
    }
    true
}

fn make_region(label: usize, centre: Element) -> Region {
    Region {
        label: LABELS[label],
        centre,
        spread: SPREAD,
        variants: variants(centre, SPREAD),
    }
}

const CANDIDATES_PER_LABEL: usize = 48;
const SEARCH_BUDGET: u32 = 300_000;
const BACKTRACK_NODES: u32 = 24_000;
const BACKTRACK_FANOUT: usize = 10;

/// Kernel-active radii: repulsive rim, attractive peak, fade.
const RING: [i32; 7] = [12, 24, 36, 48, 64, 80, 96];

fn consider(law: &Law, buckets: &mut [Vec<Region>; 5], e: Element) {
    let o = observe(law, &cfg(e));
    for label in 0..5 {
        if buckets[label].len() >= CANDIDATES_PER_LABEL {
            continue;
        }
        if !matches(label, &o) {
            continue;
        }
        if buckets[label].iter().any(|r| r.centre == e) {
            continue;
        }
        buckets[label].push(make_region(label, e));
    }
}

fn ring_point(probe: Element, radius: i32, signs: u8) -> Element {
    let mut e = probe;
    for i in 0..4 {
        let sign = if (signs >> i) & 1 == 1 { 1 } else { -1 };
        e.0[i] = (probe.0[i] as i32 + sign * radius).clamp(0, 255) as u8;
    }
    e
}

fn around_probe(probe: Element, rng: &mut Rng) -> Element {
    let mut e = probe;
    for i in 0..4 {
        let r = RING[rng.index(RING.len())];
        let sign = if rng.next_u32() % 2 == 0 { 1 } else { -1 };
        let jitter = rng.inc(-10, 10);
        e.0[i] = (probe.0[i] as i32 + sign * r + jitter).clamp(0, 255) as u8;
    }
    e
}

/// Per-axis coordinates where the v0 response is ~0: identical, the 12-unit knot, or the far edge.
fn null_axis(c: u8) -> impl Iterator<Item = u8> {
    [c, c.saturating_add(12), c.saturating_sub(12), 0, 255]
        .into_iter()
        .filter(move |&x| {
            let d = (c as i32 - x as i32).unsigned_abs();
            d == 0 || d == 12 || d == 255
        })
}

fn null_shell(centre: Element) -> Vec<Element> {
    let ax: [Vec<u8>; 4] = std::array::from_fn(|i| null_axis(centre.0[i]).collect());
    let mut out = Vec::new();
    for &a in &ax[0] {
        for &b in &ax[1] {
            for &c in &ax[2] {
                for &d in &ax[3] {
                    let e = Element([a, b, c, d]);
                    if e != centre {
                        out.push(e);
                    }
                }
            }
        }
    }
    out
}

/// Search for up to `count` mutually rest-stable regions of each label.
///
/// `count` is the number of regions per label to return (best effort). Missing labels
/// mean the law produced no matching stable centre in the search budget.
pub fn find_regions(law: &Law, seed: u64, count: usize) -> Vec<Region> {
    let count = count.max(1);
    let mut rng = Rng::new(seed ^ 0x5245_4749_4F4E_0001);
    let mut buckets: [Vec<Region>; 5] = Default::default();
    let probes = [
        law.probes.contact,
        law.probes.light,
        law.probes.flow,
        law.probes.glow,
        law.probes.friction,
    ];
    let budget = SEARCH_BUDGET.saturating_mul(count as u32).min(1_500_000);
    for _ in 0..budget {
        if buckets.iter().all(|b| b.len() >= CANDIDATES_PER_LABEL) {
            break;
        }
        consider(law, &mut buckets, rng.element());
    }
    if buckets.iter().any(|b| b.len() < 4) {
        for probe in probes {
            for &r in &RING {
                for signs in 0..16u8 {
                    consider(law, &mut buckets, ring_point(probe, r, signs));
                }
            }
        }
        for _ in 0..budget / 2 {
            if buckets.iter().all(|b| b.len() >= CANDIDATES_PER_LABEL) {
                break;
            }
            consider(law, &mut buckets, around_probe(probes[rng.index(probes.len())], &mut rng));
        }
    }
    let seeds: Vec<Element> = buckets.iter().flatten().map(|r| r.centre).take(80).collect();
    for s in seeds {
        for e in null_shell(s) {
            consider(law, &mut buckets, e);
        }
    }
    eprintln!(
        "find-regions candidates: {}={} {}={} {}={} {}={} {}={}",
        LABELS[0],
        buckets[0].len(),
        LABELS[1],
        buckets[1].len(),
        LABELS[2],
        buckets[2].len(),
        LABELS[3],
        buckets[3].len(),
        LABELS[4],
        buckets[4].len()
    );
    pick(law, &buckets, count)
}

fn pick(law: &Law, buckets: &[Vec<Region>; 5], count: usize) -> Vec<Region> {
    let mut order: Vec<usize> = (0..5).collect();
    order.sort_by_key(|&i| buckets[i].len());
    let mut best: Vec<Region> = Vec::new();
    let mut picked: Vec<Region> = Vec::new();
    let mut used = 0u32;
    fn rec(
        law: &Law,
        buckets: &[Vec<Region>; 5],
        order: &[usize],
        count: usize,
        picked: &mut Vec<Region>,
        best: &mut Vec<Region>,
        used: &mut u32,
        skipped: u8,
    ) -> bool {
        *used += 1;
        if *used > BACKTRACK_NODES {
            return false;
        }
        if score(picked) > score(best) {
            *best = picked.clone();
        }
        let mut have = [0usize; 5];
        for r in picked.iter() {
            if let Some(i) = LABELS.iter().position(|&l| l == r.label) {
                have[i] += 1;
            }
        }
        let next = order
            .iter()
            .copied()
            .find(|&i| have[i] < count && (skipped & (1 << i)) == 0);
        let Some(label) = next else {
            return have.iter().all(|&h| h >= count);
        };
        for cand in buckets[label].iter().take(BACKTRACK_FANOUT) {
            if picked.iter().any(|p| p.centre == cand.centre) {
                continue;
            }
            if !picked.iter().all(|p| compatible(law, p, cand)) {
                continue;
            }
            picked.push(cand.clone());
            if rec(law, buckets, order, count, picked, best, used, skipped) {
                return true;
            }
            picked.pop();
            if *used > BACKTRACK_NODES {
                return false;
            }
        }
        rec(law, buckets, order, count, picked, best, used, skipped | (1 << label))
    }
    let all_five = rec(law, buckets, &order, count, &mut picked, &mut best, &mut used, 0);
    let mut out = if all_five && picked.len() >= 5 * count {
        picked
    } else {
        let greedy_std = greedy_restarts(law, buckets, count, &[0, 1, 2, 3, 4]);
        let greedy_rare = greedy_restarts(law, buckets, count, &order);
        let mut choice = best;
        if score(&greedy_std) > score(&choice) {
            choice = greedy_std;
        }
        if score(&greedy_rare) > score(&choice) {
            choice = greedy_rare;
        }
        choice
    };
    out.sort_by_key(|r| LABELS.iter().position(|&l| l == r.label).unwrap_or(99));
    out
}

fn greedy_restarts(law: &Law, buckets: &[Vec<Region>; 5], count: usize, order: &[usize]) -> Vec<Region> {
    let Some(&start) = order.iter().find(|&&i| !buckets[i].is_empty()) else {
        return Vec::new();
    };
    let mut best = Vec::new();
    for seed in &buckets[start] {
        let mut greedy = vec![seed.clone()];
        for &label in order {
            if label == start {
                continue;
            }
            let mut added = 0usize;
            for cand in &buckets[label] {
                if added >= count {
                    break;
                }
                if greedy.iter().any(|p: &Region| p.centre == cand.centre) {
                    continue;
                }
                if greedy.iter().all(|p| compatible(law, p, cand)) {
                    greedy.push(cand.clone());
                    added += 1;
                }
            }
        }
        if score(&greedy) > score(&best) {
            best = greedy;
            if best.len() >= 5 * count {
                break;
            }
        }
    }
    best
}

fn score(picked: &[Region]) -> (u32, u32) {
    let mut have = [false; 5];
    for r in picked {
        if let Some(i) = LABELS.iter().position(|&l| l == r.label) {
            have[i] = true;
        }
    }
    (have.iter().filter(|h| **h).count() as u32, picked.len() as u32)
}
