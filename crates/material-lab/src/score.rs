//! Scorecard statistics and the provisional PASS/WARN bands.

use std::collections::HashSet;
use std::fmt::{self, Write};

use material::{
    element_influence, interact, observe, Acoustic, Boundary, Configuration, Element, EventKind, Law,
    D,
};

use crate::cascade::{self, CascadeRun};
use crate::rng::Rng;
use crate::stamp_hex;

/// Sample sizes for a scorecard.
#[derive(Clone, Copy, Debug)]
pub struct Scale {
    /// Similarity triples.
    pub similarity: u32,
    /// Determinism pairs.
    pub determinism: u32,
    /// Fixed-point configurations.
    pub fixed_points: u32,
    /// Cascade runs.
    pub cascade_runs: u32,
    /// Cascade grid edge.
    pub cascade_n: usize,
    /// Cascade generation cap.
    pub cascade_max_gen: u32,
    /// Starting configurations for proliferation.
    pub proliferation_seeds: u32,
    /// Pairwise events for proliferation.
    pub proliferation_events: u32,
    /// Single-element samples for family clustering.
    pub families: u32,
    /// Random configurations for the observation census.
    pub observations: u32,
}

impl Scale {
    /// The full scorecard the `scorecard` command prints.
    pub fn full() -> Self {
        Self {
            similarity: 200_000,
            determinism: 10_000,
            fixed_points: 10_000,
            cascade_runs: 50,
            cascade_n: 16,
            cascade_max_gen: 64,
            proliferation_seeds: 64,
            proliferation_events: 100_000,
            families: 20_000,
            observations: 50_000,
        }
    }

    /// Smaller samples for `sweep`.
    pub fn reduced() -> Self {
        Self {
            similarity: 2_000,
            determinism: 200,
            fixed_points: 200,
            cascade_runs: 5,
            cascade_n: 8,
            cascade_max_gen: 32,
            proliferation_seeds: 16,
            proliferation_events: 2_000,
            families: 400,
            observations: 500,
        }
    }

    /// Tiny samples for unit tests.
    pub fn tiny() -> Self {
        Self {
            similarity: 40,
            determinism: 20,
            fixed_points: 20,
            cascade_runs: 1,
            cascade_n: 4,
            cascade_max_gen: 8,
            proliferation_seeds: 8,
            proliferation_events: 40,
            families: 40,
            observations: 40,
        }
    }
}

/// PASS or WARN against a provisional band.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Inside the band.
    Pass,
    /// Outside the band.
    Warn,
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Verdict::Pass => f.write_str("PASS"),
            Verdict::Warn => f.write_str("WARN"),
        }
    }
}

/// ‖F(a,b) − F(a′,b)‖∞ over L1-neighbours.
#[derive(Clone, Copy, Debug)]
pub struct Similarity {
    /// Samples taken.
    pub n: u32,
    /// Maximum inf-norm.
    pub max: u32,
    /// 99th percentile (integer index).
    pub p99: u32,
    /// Mean inf-norm (reporting).
    pub mean: f64,
    /// Verdict: max ≤ kernel.max_step.
    pub verdict: Verdict,
}

/// Bit-for-bit replay of `interact`.
#[derive(Clone, Copy, Debug)]
pub struct Determinism {
    /// Pairs compared.
    pub n: u32,
    /// How many matched.
    pub matched: u32,
    /// Verdict: all matched.
    pub verdict: Verdict,
}

/// Fixed-point fractions under self-contact and neighbour contact.
#[derive(Clone, Copy, Debug)]
pub struct FixedPoints {
    /// Samples taken.
    pub n: u32,
    /// Unchanged by `interact(c,c,NewContact)`.
    pub self_fixed: u32,
    /// Unchanged by `interact(neighbour, c, NewContact)`.
    pub neighbour_fixed: u32,
    /// Unchanged by both.
    pub both_fixed: u32,
    /// Verdict: self-fixed fraction in 20–80%.
    pub verdict: Verdict,
}

/// Cascade-size distribution over many filled grids.
#[derive(Clone, Debug)]
pub struct Cascades {
    /// Runs performed.
    pub runs: u32,
    /// Grid cell count.
    pub cells: u32,
    /// Runs that went quiet within the generation cap.
    pub quiescent: u32,
    /// Mean generations to quiescence (capped runs count as the cap).
    pub gens_mean: f64,
    /// Maximum generations observed.
    pub gens_max: u32,
    /// Mean cells mutated per generation (over all generations of all runs).
    pub cells_mean: f64,
    /// Largest cells-mutated in any generation of any run.
    pub cells_max: u32,
    /// Verdict: ≥95% quiescent and cells_max ≤ 30% of the grid.
    pub verdict: Verdict,
}

/// Distinct-configuration growth, one quantum setting.
#[derive(Clone, Copy, Debug)]
pub struct Growth {
    /// Distinct after 1k events (or all events if the run is shorter).
    pub at_1k: u32,
    /// Distinct after 10k events.
    pub at_10k: u32,
    /// Distinct after 100k events.
    pub at_100k: u32,
    /// Distinct at the end of the run.
    pub final_count: u32,
    /// Growth rate after 10k is no higher than 1k–10k.
    pub sublinear: bool,
}

/// Proliferation with the law's quantum and with quantum 4.
#[derive(Clone, Copy, Debug)]
pub struct Proliferation {
    /// Law as given.
    pub native: Growth,
    /// Same samples with `quantum = 4`.
    pub quantum4: Growth,
    /// Verdict: native sublinear, or quantum-4 sublinear (stated as required).
    pub verdict: Verdict,
    /// True when only quantum 4 is sublinear.
    pub quantum4_required: bool,
}

/// Single-linkage families of stable single-element configurations.
#[derive(Clone, Copy, Debug)]
pub struct Families {
    /// Lattice points sampled.
    pub sample: u32,
    /// How many were stable under self and a random neighbour.
    pub stable: u32,
    /// Number of clusters (L1 ≤ 24, single linkage).
    pub families: u32,
    /// Size of the largest cluster.
    pub largest: u32,
    /// `stable / sample` as a percentage of the sampled lattice.
    pub coverage: f64,
    /// Verdict: family count in 8–200.
    pub verdict: Verdict,
}

/// Observation census.
#[derive(Clone, Copy, Debug)]
pub struct Observations {
    /// Samples taken.
    pub n: u32,
    /// `Observation.liquid`.
    pub liquid: u32,
    /// `emission > 0`.
    pub glowing: u32,
    /// `transparency > 0`.
    pub transparent: u32,
    /// `Acoustic::Soft`.
    pub soft: u32,
    /// Verdict: liquid fraction in 3–25%.
    pub verdict: Verdict,
}

/// One complete scorecard.
#[derive(Clone, Debug)]
pub struct Scorecard {
    /// Seed the streams were derived from.
    pub seed: u64,
    /// Law under test.
    pub law: Law,
    /// Sample sizes used.
    pub scale: Scale,
    /// Similarity.
    pub similarity: Similarity,
    /// Determinism.
    pub determinism: Determinism,
    /// Fixed points.
    pub fixed_points: FixedPoints,
    /// Cascades.
    pub cascades: Cascades,
    /// Proliferation.
    pub proliferation: Proliferation,
    /// Families.
    pub families: Families,
    /// Observations.
    pub observations: Observations,
}

impl Scorecard {
    /// How many of the seven bands passed.
    pub fn pass_count(&self) -> u32 {
        [
            self.similarity.verdict,
            self.determinism.verdict,
            self.fixed_points.verdict,
            self.cascades.verdict,
            self.proliferation.verdict,
            self.families.verdict,
            self.observations.verdict,
        ]
        .iter()
        .filter(|v| **v == Verdict::Pass)
        .count() as u32
    }
}

const TAG_SIM: u64 = 1;
const TAG_DET: u64 = 2;
const TAG_FIX: u64 = 3;
const TAG_CAS: u64 = 4;
const TAG_PRO: u64 = 5;
const TAG_FAM: u64 = 6;
const TAG_OBS: u64 = 7;

fn stream(seed: u64, tag: u64) -> Rng {
    Rng::new(seed ^ tag.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

fn pct(num: u32, den: u32) -> f64 {
    if den == 0 {
        0.0
    } else {
        100.0 * num as f64 / den as f64
    }
}

fn in_pct_band(num: u32, den: u32, lo: u32, hi: u32) -> bool {
    den > 0 && num * 100 >= lo * den && num * 100 <= hi * den
}

fn pass(ok: bool) -> Verdict {
    if ok {
        Verdict::Pass
    } else {
        Verdict::Warn
    }
}

/// Run every statistic at `scale`.
pub fn run_scorecard(law: &Law, seed: u64, scale: Scale) -> Scorecard {
    let similarity = measure_similarity(law, &mut stream(seed, TAG_SIM), scale.similarity);
    let determinism = measure_determinism(law, &mut stream(seed, TAG_DET), scale.determinism);
    let fixed_points = measure_fixed_points(law, &mut stream(seed, TAG_FIX), scale.fixed_points);
    let cascades = measure_cascades(
        law,
        &mut stream(seed, TAG_CAS),
        scale.cascade_runs,
        scale.cascade_n,
        scale.cascade_max_gen,
    );
    let proliferation = measure_proliferation(
        law,
        seed,
        scale.proliferation_seeds,
        scale.proliferation_events,
    );
    let families = measure_families(law, &mut stream(seed, TAG_FAM), scale.families);
    let observations = measure_observations(law, &mut stream(seed, TAG_OBS), scale.observations);
    Scorecard {
        seed,
        law: *law,
        scale,
        similarity,
        determinism,
        fixed_points,
        cascades,
        proliferation,
        observations,
        families,
    }
}

/// Similarity of `element_influence` under a one-unit L1 step that does not wrap 255→0 under Clamp.
pub fn measure_similarity(law: &Law, rng: &mut Rng, n: u32) -> Similarity {
    let mut max = 0u32;
    let mut sum = 0u64;
    let mut hist = [0u32; 33];
    for _ in 0..n {
        let a = rng.element();
        let b = rng.element();
        let axis = rng.index(D);
        let mut a2 = a;
        match law.boundary {
            Boundary::Clamp => {
                if a.0[axis] == 255 {
                    a2.0[axis] = 254;
                } else {
                    a2.0[axis] = a.0[axis] + 1;
                }
            }
            Boundary::Wrap => {
                a2.0[axis] = a.0[axis].wrapping_add(1);
            }
        }
        let f1 = element_influence(law, a, b);
        let f2 = element_influence(law, a2, b);
        let mut inf = 0u32;
        for i in 0..D {
            inf = inf.max((f1.0[i] as i32 - f2.0[i] as i32).unsigned_abs());
        }
        max = max.max(inf);
        sum += inf as u64;
        hist[inf.min(32) as usize] += 1;
    }
    let p99 = percentile(&hist, n, 99);
    let mean = if n == 0 { 0.0 } else { sum as f64 / n as f64 };
    Similarity {
        n,
        max,
        p99,
        mean,
        verdict: pass(max <= law.kernel.max_step as u32),
    }
}

fn percentile(hist: &[u32; 33], n: u32, p: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    let want = (n as u64 * p as u64) / 100;
    let want = want.max(1);
    let mut acc = 0u64;
    for (v, &c) in hist.iter().enumerate() {
        acc += c as u64;
        if acc >= want {
            return v as u32;
        }
    }
    32
}

/// Replay `n` random interactions and count bit-identical repeats.
pub fn measure_determinism(law: &Law, rng: &mut Rng, n: u32) -> Determinism {
    let mut matched = 0u32;
    for _ in 0..n {
        let a = rng.config(6);
        let b = rng.config(6);
        let kind = EventKind::ALL[rng.index(EventKind::ALL.len())];
        let r1 = interact(law, &a, &b, kind);
        let r2 = interact(law, &a, &b, kind);
        if r1 == r2 {
            matched += 1;
        }
    }
    Determinism {
        n,
        matched,
        verdict: pass(matched == n),
    }
}

/// Fraction of random 1..=6-element configurations that are fixed points.
pub fn measure_fixed_points(law: &Law, rng: &mut Rng, n: u32) -> FixedPoints {
    let mut self_fixed = 0u32;
    let mut neighbour_fixed = 0u32;
    let mut both_fixed = 0u32;
    for _ in 0..n {
        let c = rng.config(6);
        let nbor = rng.config(6);
        let s = !interact(law, &c, &c, EventKind::NewContact).changed;
        let k = !interact(law, &nbor, &c, EventKind::NewContact).changed;
        if s {
            self_fixed += 1;
        }
        if k {
            neighbour_fixed += 1;
        }
        if s && k {
            both_fixed += 1;
        }
    }
    FixedPoints {
        n,
        self_fixed,
        neighbour_fixed,
        both_fixed,
        verdict: pass(in_pct_band(self_fixed, n, 20, 80)),
    }
}

/// Cascade runs on an `n³` grid.
pub fn measure_cascades(law: &Law, rng: &mut Rng, runs: u32, n: usize, max_gen: u32) -> Cascades {
    let cells = (n * n * n) as u32;
    let mut quiescent = 0u32;
    let mut gens_sum = 0u64;
    let mut gens_max = 0u32;
    let mut cells_sum = 0u64;
    let mut cells_n = 0u64;
    let mut cells_max = 0u32;
    for _ in 0..runs {
        let r: CascadeRun = cascade::run_one(law, rng, n, max_gen);
        if r.quiescent {
            quiescent += 1;
        }
        gens_sum += r.generations as u64;
        gens_max = gens_max.max(r.generations);
        cells_sum += r.initial_changed as u64;
        cells_n += 1;
        for c in &r.cells_per_gen {
            cells_sum += *c as u64;
            cells_n += 1;
        }
        cells_max = cells_max.max(r.max_changed);
    }
    let gens_mean = if runs == 0 {
        0.0
    } else {
        gens_sum as f64 / runs as f64
    };
    let cells_mean = if cells_n == 0 {
        0.0
    } else {
        cells_sum as f64 / cells_n as f64
    };
    let q_ok = runs == 0 || quiescent * 100 >= 95 * runs;
    let burst_ok = cells == 0 || cells_max * 100 <= 30 * cells;
    Cascades {
        runs,
        cells,
        quiescent,
        gens_mean,
        gens_max,
        cells_mean,
        cells_max,
        verdict: pass(q_ok && burst_ok),
    }
}

fn growth_at(counts: &[(u32, u32)], events: u32) -> u32 {
    let mut last = 0;
    for &(at, n) in counts {
        if at <= events {
            last = n;
        }
    }
    last
}

fn is_sublinear(g: &Growth, events: u32) -> bool {
    // Compare 1k→10k against 10k→100k when the run is long enough; otherwise
    // compare the first tenth of the run against the rest.
    let (a, b, c) = if events >= 100_000 {
        (g.at_1k, g.at_10k, g.at_100k)
    } else if events >= 10_000 {
        (g.at_1k, g.at_10k, g.final_count)
    } else {
        return true;
    };
    let inc1 = b.saturating_sub(a);
    let inc2 = c.saturating_sub(b);
    inc2 <= inc1.saturating_mul(10)
}

fn one_proliferation(law: &Law, rng: &mut Rng, n_seeds: u32, n_events: u32) -> Growth {
    let mut pool: Vec<Configuration> = (0..n_seeds).map(|_| rng.config(6)).collect();
    let mut distinct: HashSet<Configuration> = pool.iter().cloned().collect();
    let mut marks: Vec<(u32, u32)> = vec![(0, distinct.len() as u32)];
    for i in 1..=n_events {
        let a = rng.index(pool.len());
        let b = rng.index(pool.len());
        let kind = EventKind::ALL[rng.index(EventKind::ALL.len())];
        let r = interact(law, &pool[a], &pool[b], kind);
        if distinct.insert(r.target.clone()) {
            pool.push(r.target);
        }
        if i == 1_000 || i == 10_000 || i == 100_000 || i == n_events {
            marks.push((i, distinct.len() as u32));
        }
    }
    let at_1k = growth_at(&marks, 1_000.min(n_events));
    let at_10k = growth_at(&marks, 10_000.min(n_events));
    let at_100k = growth_at(&marks, 100_000.min(n_events));
    let mut g = Growth {
        at_1k,
        at_10k,
        at_100k,
        final_count: distinct.len() as u32,
        sublinear: false,
    };
    g.sublinear = is_sublinear(&g, n_events);
    g
}

/// Distinct configurations after many pairwise events, native quantum and quantum 4.
pub fn measure_proliferation(law: &Law, seed: u64, n_seeds: u32, n_events: u32) -> Proliferation {
    let native = one_proliferation(law, &mut stream(seed, TAG_PRO), n_seeds, n_events);
    let mut q4 = *law;
    q4.quantum = 4;
    let quantum4 = one_proliferation(&q4, &mut stream(seed, TAG_PRO), n_seeds, n_events);
    let quantum4_required = !native.sublinear && quantum4.sublinear;
    Proliferation {
        native,
        quantum4,
        verdict: pass(native.sublinear || quantum4.sublinear),
        quantum4_required,
    }
}

/// Single-linkage clustering of stable single-element samples, L1 threshold 24.
pub fn measure_families(law: &Law, rng: &mut Rng, n: u32) -> Families {
    let mut stable: Vec<Element> = Vec::new();
    for _ in 0..n {
        let e = rng.element();
        let c = Configuration::single(e);
        if interact(law, &c, &c, EventKind::NewContact).changed {
            continue;
        }
        let nbor = Configuration::single(rng.element());
        if interact(law, &nbor, &c, EventKind::NewContact).changed {
            continue;
        }
        stable.push(e);
    }
    let ns = stable.len();
    let (families, largest) = if ns == 0 {
        (0, 0)
    } else {
        cluster_l1(&mut stable, 24)
    };
    let coverage = pct(ns as u32, n);
    Families {
        sample: n,
        stable: ns as u32,
        families,
        largest,
        coverage,
        verdict: pass((8..=200).contains(&families)),
    }
}

fn cluster_l1(points: &mut [Element], thresh: u32) -> (u32, u32) {
    points.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let n = points.len();
    let mut uf = Uf::new(n);
    for i in 0..n {
        let x0 = points[i].0[0] as u32;
        for j in i + 1..n {
            if points[j].0[0] as u32 > x0 + thresh {
                break;
            }
            if points[i].distance(points[j]) <= thresh {
                uf.union(i, j);
            }
        }
    }
    let mut roots = 0u32;
    let mut largest = 0u32;
    for i in 0..n {
        if uf.find(i) == i {
            roots += 1;
            largest = largest.max(uf.sz[i]);
        }
    }
    (roots, largest)
}

struct Uf {
    p: Vec<usize>,
    sz: Vec<u32>,
}

impl Uf {
    fn new(n: usize) -> Self {
        Self {
            p: (0..n).collect(),
            sz: vec![1; n],
        }
    }
    fn find(&mut self, mut x: usize) -> usize {
        while self.p[x] != x {
            self.p[x] = self.p[self.p[x]];
            x = self.p[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) {
        let mut a = self.find(a);
        let mut b = self.find(b);
        if a == b {
            return;
        }
        if self.sz[a] < self.sz[b] {
            std::mem::swap(&mut a, &mut b);
        }
        self.p[b] = a;
        self.sz[a] += self.sz[b];
    }
}

/// Liquid / glowing / transparent / soft fractions over random configurations.
pub fn measure_observations(law: &Law, rng: &mut Rng, n: u32) -> Observations {
    let mut liquid = 0u32;
    let mut glowing = 0u32;
    let mut transparent = 0u32;
    let mut soft = 0u32;
    for _ in 0..n {
        let o = observe(law, &rng.config(6));
        if o.liquid {
            liquid += 1;
        }
        if o.emission > 0 {
            glowing += 1;
        }
        if o.transparency > 0 {
            transparent += 1;
        }
        if o.acoustic == Acoustic::Soft {
            soft += 1;
        }
    }
    Observations {
        n,
        liquid,
        glowing,
        transparent,
        soft,
        verdict: pass(in_pct_band(liquid, n, 3, 25)),
    }
}

impl fmt::Display for Scorecard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let law_tag = if self.law == Law::v0() { "v0" } else { "custom" };
        let hex = stamp_hex(&self.law);
        writeln!(f, "======== material-lab scorecard ========")?;
        writeln!(
            f,
            "seed {}  law {}  stamp {}  fingerprint {:016x}",
            self.seed,
            law_tag,
            hex,
            self.law.fingerprint()
        )?;
        writeln!(
            f,
            "similarity   n={}  max={}  p99={}  mean={:.3}  [{}] max<=max_step={}",
            self.similarity.n,
            self.similarity.max,
            self.similarity.p99,
            self.similarity.mean,
            self.similarity.verdict,
            self.law.kernel.max_step
        )?;
        writeln!(
            f,
            "determinism  {}/{} bit-identical  [{}]",
            self.determinism.matched, self.determinism.n, self.determinism.verdict
        )?;
        writeln!(
            f,
            "fixed points n={}  self={:.1}%  neighbour={:.1}%  both={:.1}%  [{}] self 20-80%",
            self.fixed_points.n,
            pct(self.fixed_points.self_fixed, self.fixed_points.n),
            pct(self.fixed_points.neighbour_fixed, self.fixed_points.n),
            pct(self.fixed_points.both_fixed, self.fixed_points.n),
            self.fixed_points.verdict
        )?;
        writeln!(
            f,
            "cascades     {} runs, {n}³ ({cells} cells)",
            self.cascades.runs,
            n = self.scale.cascade_n,
            cells = self.cascades.cells
        )?;
        writeln!(
            f,
            "             quiescent {}/{} ({:.1}%)  gens mean={:.2} max={}  [{}]",
            self.cascades.quiescent,
            self.cascades.runs,
            pct(self.cascades.quiescent, self.cascades.runs),
            self.cascades.gens_mean,
            self.cascades.gens_max,
            self.cascades.verdict
        )?;
        writeln!(
            f,
            "             cells/gen mean={:.2} max={}/{} ({:.1}%)  cap 30%",
            self.cascades.cells_mean,
            self.cascades.cells_max,
            self.cascades.cells,
            pct(self.cascades.cells_max, self.cascades.cells)
        )?;
        let qnote = if self.proliferation.quantum4_required {
            "quantum 4 is required"
        } else if self.proliferation.native.sublinear {
            "native sublinear"
        } else {
            "not sublinear"
        };
        writeln!(
            f,
            "proliferation {} seeds, {} events  [{}] {}",
            self.scale.proliferation_seeds,
            self.scale.proliferation_events,
            self.proliferation.verdict,
            qnote
        )?;
        writeln!(
            f,
            "             q={}  1k={}  10k={}  100k={}  sublinear={}",
            self.law.quantum,
            self.proliferation.native.at_1k,
            self.proliferation.native.at_10k,
            self.proliferation.native.at_100k,
            self.proliferation.native.sublinear
        )?;
        writeln!(
            f,
            "             q=4  1k={}  10k={}  100k={}  sublinear={}",
            self.proliferation.quantum4.at_1k,
            self.proliferation.quantum4.at_10k,
            self.proliferation.quantum4.at_100k,
            self.proliferation.quantum4.sublinear
        )?;
        writeln!(
            f,
            "families     sample={}  stable={}  families={}  largest={}  coverage={:.1}%  [{}] 8-200",
            self.families.sample,
            self.families.stable,
            self.families.families,
            self.families.largest,
            self.families.coverage,
            self.families.verdict
        )?;
        writeln!(
            f,
            "observations n={}  liquid={:.1}%  glowing={:.1}%  transparent={:.1}%  soft={:.1}%  [{}] liquid 3-25%",
            self.observations.n,
            pct(self.observations.liquid, self.observations.n),
            pct(self.observations.glowing, self.observations.n),
            pct(self.observations.transparent, self.observations.n),
            pct(self.observations.soft, self.observations.n),
            self.observations.verdict
        )?;
        writeln!(f, "----------------------------------------")?;
        let overall = if self.pass_count() == 7 {
            Verdict::Pass
        } else {
            Verdict::Warn
        };
        writeln!(f, "VERDICT  {}  {}/7 bands", overall, self.pass_count())?;
        write!(f, "========================================")
    }
}

/// Render the scorecard to a String (same as Display).
pub fn render(card: &Scorecard) -> String {
    let mut s = String::new();
    let _ = write!(s, "{card}");
    s
}
