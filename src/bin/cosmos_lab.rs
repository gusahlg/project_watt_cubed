//! The cosmos lab (EMERGENT-WORLDGEN-DESIGN P1): grow N universes and print what they hold
//! (bodies, layouts, gravity, heat, air, suites, fallbacks, storage, creation time) with the P1
//! acceptance gates, and write PPMs (nebula slices, suite swatches, G=32 globe previews).
//!
//! `cosmos_lab <out_dir> [--seeds N] [--first S] [--threads T] [--images K] [--set name=value]...
//! [--sweep name=v1,v2,...]`. A sweep prints the gates for each value instead of the full report.

use std::fmt::Write as _;
use std::time::Instant;

use project_watt_cubed::world::terrain::emergent::globe::{self, Input, Node};
use project_watt_cubed::world::terrain::emergent::nebula::{Nebula, N};
use project_watt_cubed::world::terrain::emergent::{nebula_seed, Form, Params, Rank, Universe};
use project_watt_cubed::world::terrain::TerrainCfg;

/// Proposed thresholds for the suites gate (the design leaves them to be agreed).
const AMOUNT_SPREAD_MIN: u8 = 2;
const COLOUR_SPREAD_MIN: u16 = 120;

struct Args {
    out: String,
    seeds: u64,
    first: u64,
    threads: usize,
    images: u64,
    params: Params,
    sweep: Option<(String, Vec<String>)>,
    bench: u32,
}

fn args() -> Args {
    let mut a = Args {
        out: String::new(),
        seeds: 1000,
        first: 1,
        threads: 4,
        images: 6,
        params: Params::default(),
        sweep: None,
        bench: 0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--seeds" => a.seeds = value().parse().expect("--seeds N"),
            "--first" => a.first = value().parse().expect("--first S"),
            "--threads" => a.threads = value().parse().expect("--threads T"),
            "--images" => a.images = value().parse().expect("--images K"),
            "--bench" => a.bench = value().parse().expect("--bench R"),
            "--set" => {
                let v = value();
                let (k, x) = v.split_once('=').expect("--set name=value");
                a.params.set(k, x).unwrap_or_else(|e| panic!("{e}"));
            }
            "--sweep" => {
                let v = value();
                let (k, xs) = v.split_once('=').expect("--sweep name=v1,v2");
                a.sweep = Some((k.to_string(), xs.split(',').map(str::to_string).collect()));
            }
            _ if a.out.is_empty() && !arg.starts_with("--") => a.out = arg.clone(),
            _ => panic!("unknown argument {arg}"),
        }
    }
    assert!(!a.out.is_empty(), "usage: cosmos_lab <out_dir> [--seeds N] [--first S] [--threads T] [--images K]");
    a
}

/// Values of one quantity across the run.
#[derive(Default)]
struct Series(Vec<f64>);

impl Series {
    fn push(&mut self, v: f64) {
        self.0.push(v);
    }

    fn sorted(&self) -> Vec<f64> {
        let mut v = self.0.clone();
        v.sort_by(f64::total_cmp);
        v
    }

    fn quantile(&self, q: f64) -> f64 {
        let v = self.sorted();
        if v.is_empty() { f64::NAN } else { v[((v.len() - 1) as f64 * q).round() as usize] }
    }

    fn summary(&self) -> String {
        let v = self.sorted();
        if v.is_empty() {
            return "(none)".into();
        }
        let q = |x: f64| v[((v.len() - 1) as f64 * x).round() as usize];
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        format!(
            "n={} min={} p10={} p50={} p90={} p99={} max={} mean={}",
            v.len(),
            fmt(v[0]),
            fmt(q(0.1)),
            fmt(q(0.5)),
            fmt(q(0.9)),
            fmt(q(0.99)),
            fmt(v[v.len() - 1]),
            fmt(mean)
        )
    }

    /// A text histogram over the given bin edges (values past the last edge go in the last bin).
    fn histogram(&self, title: &str, edges: &[f64]) -> String {
        let mut counts = vec![0usize; edges.len()];
        for &x in &self.0 {
            let k = edges.iter().rposition(|&e| x >= e).unwrap_or(0);
            counts[k] += 1;
        }
        let most = counts.iter().copied().max().unwrap_or(1).max(1);
        let mut s = format!("{title}: {}\n", self.summary());
        for (k, &c) in counts.iter().enumerate() {
            let hi = edges.get(k + 1).map_or("+".to_string(), |e| fmt(*e));
            let _ = writeln!(s, "  [{:>8} .. {:>8}) {:>6} {}", fmt(edges[k]), hi, c, "#".repeat(c * 50 / most));
        }
        s
    }
}

fn fmt(v: f64) -> String {
    if v == 0.0 || (v.abs() >= 0.01 && v.abs() < 1e5) {
        format!("{v:.2}").trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        format!("{v:.2e}")
    }
}

/// Category counts.
#[derive(Default)]
struct Counts(Vec<(String, usize)>);

impl Counts {
    fn add(&mut self, k: impl Into<String>) {
        let k = k.into();
        match self.0.iter_mut().find(|(x, _)| *x == k) {
            Some(e) => e.1 += 1,
            None => self.0.push((k, 1)),
        }
    }

    fn line(&self, title: &str) -> String {
        let total: usize = self.0.iter().map(|e| e.1).sum();
        let mut v = self.0.clone();
        v.sort();
        let parts: Vec<String> =
            v.iter().map(|(k, c)| format!("{k} {c} ({:.1}%)", 100.0 * *c as f64 / total.max(1) as f64)).collect();
        format!("{title}: {}\n", parts.join(", "))
    }
}

/// Everything the report and the gates read.
#[derive(Default)]
struct Stats {
    seeds: usize,
    bodies: Series,
    others: Series,
    systems: Series,
    basins: Series,
    half: Series,
    radius_round: Series,
    form: Counts,
    rank: Counts,
    both_layouts: usize,
    gravity: Series,
    heat: Series,
    temp: Series,
    air: Counts,
    air_top: Series,
    life: Series,
    glow: usize,
    pi: Series,
    pi_clamped: usize,
    suites: Series,
    suite_amount: Series,
    suite_amount_spread: Series,
    suite_cohesion: Series,
    suite_colour_spread: Series,
    suite_density: Series,
    suite_yield: Series,
    suite_distinct: Series,
    suite_sweeps: Series,
    suite_replaced: Series,
    suite_dropped: Series,
    suite_fallback: usize,
    suite_count: usize,
    causes: Counts,
    own_amount: Series,
    own_cohesion: Series,
    own_amount_spread: Series,
    own_colour_spread: Series,
    own_restless: Series,
    own_restless_of: usize,
    suite_spread_ok: usize,
    seeds_all_own: usize,
    start_own: usize,
    suite_ms: Series,
    start_density: Series,
    start_radius: Series,
    start_tries: Series,
    start_fallback: usize,
    start_valid: usize,
    resalts: Series,
    debris: Series,
    rings: Series,
    binaries: Series,
    fell: Series,
    out_of_bounds: Series,
    capped: Series,
    storage_rows: Series,
    storage_row0: Series,
    storage_cube_y: Series,
    storage_demoted: Series,
    t_total: Series,
    t_nebula: Series,
    t_accrete: Series,
    t_suites: Series,
    t_settle: Series,
    t_storage: Series,
    t_globe: Series,
    t_globe128: Series,
    t_globe64: Series,
    t_create: Series,
    globe_g: Counts,
    globe_phase: [Series; 8],
}

impl Stats {
    fn add(&mut self, u: &Universe, globe_ms: f64, globe_g: u32, g128: &globe::Globe, g64_ms: f64) {
        self.seeds += 1;
        self.bodies.push(u.bodies.len() as f64);
        self.others.push(u.bodies.len() as f64 - 1.0);
        self.systems.push(u.systems as f64);
        self.basins.push(u.basins as f64);
        let mut forms = [false; 3];
        for b in &u.bodies {
            let t = &b.traits;
            self.half.push(b.half as f64);
            if t.form == Form::Round {
                self.radius_round.push(b.radius as f64);
            }
            self.form.add(format!("{:?}", t.form));
            self.rank.add(format!("{:?}", t.rank));
            forms[t.form as usize] = true;
            self.gravity.push(t.gravity as f64);
            self.heat.push(t.heat as f64 / 256.0);
            self.temp.push(t.temp as f64);
            self.air.add(if t.air_top.is_some() { "air" } else { "airless" });
            if let Some(top) = t.air_top {
                self.air_top.push(top as f64);
            }
            self.life.push(t.life as f64);
            self.glow += t.glow as usize;
            self.pi.push(t.pi_g);
            self.pi_clamped += t.pi_clamped as usize;
        }
        self.both_layouts += ((forms[0] || forms[1]) && forms[2]) as usize;
        self.suites.push(u.suites.len() as f64);
        let mut all_own = true;
        for (s, &ms) in u.suites.iter().zip(&u.suite_ms) {
            self.suite_count += 1;
            // Suites the law was asked for (the start world's last resort is a fixed role).
            if ms > 0.0 {
                self.suite_ms.push(ms);
                self.suite_distinct.push(s.distinct as f64);
                self.suite_sweeps.push(s.sweeps as f64);
                self.suite_dropped.push(s.dropped as f64);
                self.suite_replaced.push(s.own.iter().filter(|m| m.replaced).count() as f64);
                for m in &s.own {
                    self.own_amount.push(m.amount as f64);
                    self.own_cohesion.push(m.cohesion as f64);
                }
                let amounts = s.own.iter().map(|m| m.amount);
                self.own_amount_spread.push((amounts.clone().max().unwrap_or(0) - amounts.min().unwrap_or(0)) as f64);
                self.own_colour_spread.push(colour_spread(&s.own));
                if self.seeds <= 100 {
                    let law = project_watt_cubed::material::Law::current();
                    for m in &s.own {
                        let (n, of) = project_watt_cubed::world::terrain::emergent::minerals::restless_against(&law, m);
                        self.own_restless.push(n as f64);
                        self.own_restless_of = of;
                    }
                }
                let names = ["few distinct", "colour", "restless"];
                let why: Vec<&str> = (0..3).filter(|k| s.causes >> k & 1 == 1).map(|k| names[k]).collect();
                self.causes.add(if why.is_empty() { "own".to_string() } else { why.join("+") });
            }
            if s.fallback {
                self.suite_fallback += 1;
                all_own = false;
                continue;
            }
            for m in &s.minerals {
                self.suite_amount.push(m.amount as f64);
                self.suite_cohesion.push(m.cohesion as f64);
            }
            self.suite_amount_spread.push(s.amount_spread as f64);
            self.suite_colour_spread.push(s.colour_spread as f64);
            self.suite_density.push(s.density);
            self.suite_yield.push(s.yield_stress);
            self.suite_spread_ok += (s.amount_spread >= AMOUNT_SPREAD_MIN && s.colour_spread >= COLOUR_SPREAD_MIN) as usize;
        }
        self.seeds_all_own += all_own as usize;
        let start = &u.bodies[0];
        self.start_own += !u.suites[start.traits.suite as usize].fallback as usize;
        self.start_density.push(start.density);
        self.start_radius.push(start.radius as f64);
        self.start_tries.push(u.start_tries as f64);
        self.start_fallback += u.start_fallback as usize;
        let contract = (start.density * start.radius as f64 / project_watt_cubed::world::terrain::emergent::RHO_R - 1.0).abs() < 1e-6;
        let valid = start.traits.rank == Rank::Start && start.traits.form == Form::Round && contract;
        self.start_valid += valid as usize;
        self.resalts.push(u.resalts as f64);
        self.debris.push(u.debris);
        self.rings.push(u.rings as f64);
        self.binaries.push(u.binaries as f64);
        self.fell.push(u.fell as f64);
        self.out_of_bounds.push(u.out_of_bounds as f64);
        self.capped.push(u.capped as f64);
        self.storage_rows.push(u.storage.rows as f64);
        self.storage_row0.push(u.storage.row0);
        self.storage_cube_y.push(u.storage.cube_y);
        self.storage_demoted.push(u.storage.demoted as f64);
        let t = &u.time;
        self.t_total.push(t.total);
        self.t_nebula.push(t.nebula);
        self.t_accrete.push(t.accrete);
        self.t_suites.push(t.suites);
        self.t_settle.push(t.settle);
        self.t_storage.push(t.storage);
        self.t_globe.push(globe_ms);
        self.t_create.push(t.total + globe_ms);
        self.globe_g.add(format!("G={globe_g}"));
        self.t_globe128.push(g128.ms.iter().sum());
        self.t_globe64.push(g64_ms);
        for (k, &ms) in g128.ms.iter().enumerate() {
            self.globe_phase[k].push(ms);
        }
    }

    /// The P1 gates, one line each.
    fn gates(&self) -> String {
        let pct = |a: usize, b: usize| 100.0 * a as f64 / b.max(1) as f64;
        let own = pct(self.seeds_all_own, self.seeds);
        let spread_ok = pct(self.suite_spread_ok, self.suite_count - self.suite_fallback);
        let suite_med = self.suite_ms.quantile(0.5);
        let g128 = self.t_globe128.quantile(0.5);
        let g128_p99 = self.t_globe128.quantile(0.99);
        let g64 = self.t_globe64.quantile(0.5);
        let g64_p99 = self.t_globe64.quantile(0.99);
        let valid = pct(self.start_valid, self.seeds);
        let verdict = |ok: bool| if ok { "PASS" } else { "FAIL" };
        let mut s = String::new();
        let _ = writeln!(
            s,
            "GATE suites: {own:.1}% of seeds have every suite their own (non-fallback, >= 3 cohesive minerals); need >= 90% -> {}",
            verdict(own >= 90.0)
        );
        let _ = writeln!(
            s,
            "  suite fallback rate {:.1}% of {} suites; start-world suite own in {:.1}% of seeds; own suites with amount spread >= {AMOUNT_SPREAD_MIN} and colour spread >= {COLOUR_SPREAD_MIN}: {spread_ok:.1}% (thresholds proposed, not agreed)",
            pct(self.suite_fallback, self.suite_count),
            self.suite_count,
            pct(self.start_own, self.seeds)
        );
        let _ = writeln!(
            s,
            "GATE speed: suites median {:.3} ms per suite (need <= 0.5) -> {}; start globe at G=128 on the run's threads p50 {g128:.1} ms, p99 {g128_p99:.1} ms (need <= 25) -> {}; the design's fallback G=64: p50 {g64:.1} ms, p99 {g64_p99:.1} ms -> {}",
            suite_med,
            verdict(suite_med <= 0.5),
            verdict(g128_p99 <= 25.0),
            verdict(g64_p99 <= 25.0)
        );
        let _ = writeln!(
            s,
            "GATE start world: valid (start rank, round, ρ·R pinned) in {valid:.1}% of seeds, {:.1}% through the `rock` fallback; need 100% -> {}",
            pct(self.start_fallback, self.seeds),
            verdict(self.start_valid == self.seeds)
        );
        s
    }
}

/// Largest pairwise colour distance among minerals (sum of channel differences).
fn colour_spread(m: &[project_watt_cubed::world::terrain::emergent::minerals::Mineral]) -> f64 {
    let mut best = 0;
    for (i, a) in m.iter().enumerate() {
        for b in &m[i + 1..] {
            best = best.max((0..3).map(|k| (a.rgb[k] as i32 - b.rgb[k] as i32).unsigned_abs()).sum::<u32>());
        }
    }
    best as f64
}

fn write_ppm(path: &str, w: usize, h: usize, rgb: &[u8]) {
    let mut data = format!("P6\n{w} {h}\n255\n").into_bytes();
    data.extend_from_slice(rgb);
    std::fs::write(path, data).unwrap_or_else(|e| panic!("writing {path}: {e}"));
}

fn hue(k: u32) -> [u8; 3] {
    let h = k.wrapping_mul(0x9E37_79B9).rotate_left(7).to_le_bytes();
    [64 + h[0] / 2, 64 + h[1] / 2, 64 + h[2] / 2]
}

/// Three panels per seed: mass projected along z (log grey), the z = 16 slice of mass, and the
/// z = 16 slice coloured by system (grey: basins that became debris).
fn nebula_image(out: &str, seed: u64, u: &Universe, cfg: &TerrainCfg, p: &Params) {
    const S: usize = 4;
    let n = N as usize;
    let neb = Nebula::new(nebula_seed(seed, u.resalts), cfg, p, 4);
    let mut system = vec![u32::MAX; n * n * n];
    for (k, s) in neb.systems.iter().enumerate() {
        for &(c, _) in &s.cells {
            system[c as usize] = k as u32;
        }
    }
    let lg = |m: u64| (m.max(1) as f64).log2();
    let (lo, hi) = (lg(1 << 16), neb.cells.iter().map(|c| lg(c.m)).fold(0.0, f64::max));
    let grey = |m: u64| (((lg(m) - lo) / (hi - lo).max(1e-9)).clamp(0.0, 1.0) * 255.0) as u8;
    let w = 3 * n * S;
    let mut img = vec![0u8; w * n * S * 3];
    for y in 0..n {
        for x in 0..n {
            let col = (0..n).map(|z| neb.cells[x + n * (y + n * z)].m).max().unwrap_or(0);
            let mid = x + n * (y + n * (n / 2));
            let panels = [[grey(col); 3], [grey(neb.cells[mid].m); 3], match system[mid] {
                u32::MAX => [40, 40, 40],
                k => hue(k + 1),
            }];
            for (p_i, c) in panels.iter().enumerate() {
                for dy in 0..S {
                    for dx in 0..S {
                        let px = p_i * n * S + x * S + dx;
                        let py = (n - 1 - y) * S + dy;
                        img[(py * w + px) * 3..][..3].copy_from_slice(c);
                    }
                }
            }
        }
    }
    write_ppm(&format!("{out}/nebula_{seed}.ppm"), w, n * S, &img);
}

/// One row per suite: the law's own minerals (densest left; a red corner marks one the rest check
/// rejected), then the layers the body uses (a red bar marks a fallback suite). Bar height is amount.
fn suite_image(out: &str, seed: u64, u: &Universe) {
    const CELL: usize = 10;
    const RED: [u8; 3] = [255, 0, 0];
    let w = 26 * CELL;
    let h = u.suites.len().max(1) * CELL;
    let mut img = vec![16u8; w * h * 3];
    let mut put = |x: usize, y: usize, c: &[u8; 3]| img[(y * w + x) * 3..][..3].copy_from_slice(c);
    for (r, s) in u.suites.iter().enumerate() {
        let rows = [(0usize, &s.own), (13, &s.minerals)];
        for (x0, minerals) in rows {
            for (k, m) in minerals.iter().take(12).enumerate() {
                let tall = (m.amount as usize * CELL / 32).clamp(2, CELL);
                for dy in CELL - tall..CELL {
                    for dx in 0..CELL - 1 {
                        put((x0 + k) * CELL + dx, r * CELL + dy, &m.rgb);
                    }
                }
                if x0 == 0 && m.replaced {
                    for d in 0..3 {
                        put((x0 + k) * CELL + d, r * CELL, &RED);
                        put((x0 + k) * CELL, r * CELL + d, &RED);
                    }
                }
            }
        }
        if s.fallback {
            for dy in 0..CELL - 1 {
                put(25 * CELL + 3, r * CELL + dy, &RED);
                put(25 * CELL + 4, r * CELL + dy, &RED);
            }
        }
    }
    write_ppm(&format!("{out}/suites_{seed}.ppm"), w, h, &img);
}

/// A G=32 globe as an unfolded cube: albedo shaded by elevation.
fn globe_image(path: &str, g: &globe::Globe) {
    let s = field::Sphere::get(g.g);
    let side = (g.g + 1) as usize;
    let (w, h) = (4 * side, 3 * side);
    let mut img = vec![0u8; w * h * 3];
    let (lo, hi) = g.nodes.iter().fold((i16::MAX, i16::MIN), |(a, b), n| (a.min(n.elev), b.max(n.elev)));
    // (face, column, row) of the cross: +Y on top, then −X +Z +X −Z, −Y below.
    let layout = [(2usize, 1usize, 0usize), (1, 0, 1), (4, 1, 1), (0, 2, 1), (5, 3, 1), (3, 1, 2)];
    for &(face, cx, cy) in &layout {
        for j in 0..side {
            for i in 0..side {
                let node: &Node = &g.nodes[s.node(face, i as u32, j as u32)];
                let a = node.albedo;
                let rgb = [((a >> 11) & 31) << 3, ((a >> 5) & 63) << 2, (a & 31) << 3];
                let shade = 0.6 + 0.4 * (node.elev - lo) as f64 / (hi - lo).max(1) as f64;
                let px = cx * side + i;
                let py = cy * side + (side - 1 - j);
                for k in 0..3 {
                    img[(py * w + px) * 3 + k] = (rgb[k] as f64 * shade).min(255.0) as u8;
                }
            }
        }
    }
    write_ppm(path, w, h, &img);
}

/// The gates for every value of one parameter.
fn sweep(a: &Args, cfg: &TerrainCfg, name: &str, values: &[String]) {
    println!("# sweep {name} over {} seeds from {}", a.seeds, a.first);
    for v in values {
        let mut p = a.params.clone();
        p.set(name, v).unwrap_or_else(|e| panic!("{e}"));
        let mut st = Stats::default();
        for seed in a.first..a.first + a.seeds {
            let u = Universe::new(seed, cfg, &p, a.threads);
            let g = globe::grow(&Input::of(&u, 0, 32, 1.0), a.threads);
            st.add(&u, 0.0, 32, &g, 0.0);
        }
        println!("## {name}={v}");
        println!("bodies {}", st.bodies.summary());
        println!("{}", st.form.line("forms").trim_end());
        println!("law amount {}", st.own_amount.summary());
        println!("law cohesion {}", st.own_cohesion.summary());
        println!("law colour spread {}", st.own_colour_spread.summary());
        println!("law distinct {}", st.suite_distinct.summary());
        print!("{}", st.causes.line("verdict"));
        println!("law minerals' reacting universals (of {}) {}", st.own_restless_of, st.own_restless.summary());
        println!("start density {}", st.start_density.summary());
        print!("{}", st.gates());
    }
}

/// The start globe of the first seed grown `a.bench` times per resolution and thread count: the
/// fastest run per phase (the build box is shared, so the minimum is the cost).
fn bench(a: &Args, cfg: &TerrainCfg) {
    let u = Universe::new(a.first, cfg, &a.params, a.threads);
    println!("# globe bench: seed {}, best of {} runs, ms per phase ({})", a.first, a.bench, globe::PHASES.join(", "));
    for g in [32u32, 64, 128] {
        for threads in [1, a.threads] {
            let input = Input::of(&u, 0, g, 1.0);
            let mut best = [f64::INFINITY; 8];
            for _ in 0..a.bench {
                let r = globe::grow(&input, threads);
                for k in 0..8 {
                    best[k] = best[k].min(r.ms[k]);
                }
            }
            let parts: Vec<String> = best.iter().map(|v| format!("{v:.2}")).collect();
            println!("G={g:<3} threads={threads}: total {:.2} [{}]", best.iter().sum::<f64>(), parts.join(", "));
        }
    }
    let mut best = f64::INFINITY;
    for _ in 0..a.bench {
        best = best.min(Universe::new(a.first, cfg, &a.params, a.threads).time.total);
    }
    println!("universe (no globe), best: {best:.2} ms");
}

fn main() {
    let a = args();
    let cfg = TerrainCfg::default();
    std::fs::create_dir_all(&a.out).expect("output directory");
    if a.bench > 0 {
        bench(&a, &cfg);
        return;
    }
    if let Some((name, values)) = &a.sweep {
        sweep(&a, &cfg, name, values);
        return;
    }
    println!("# Cosmos lab: {} seeds from {}, {} threads", a.seeds, a.first, a.threads);
    println!("params: {}\n", a.params.describe());
    let mut st = Stats::default();
    let mut examples = String::new();
    let wall = Instant::now();
    for seed in a.first..a.first + a.seeds {
        let u = Universe::new(seed, &cfg, &a.params, a.threads);
        let start = &u.bodies[0];
        let g = Input::resolution(start.radius as f64);
        let t = Instant::now();
        let own = globe::grow(&Input::of(&u, 0, g, 1.0), a.threads);
        let globe_ms = t.elapsed().as_secs_f64() * 1e3;
        let g128 = if g == 128 { own } else { globe::grow(&Input::of(&u, 0, 128, 1.0), a.threads) };
        let g64: f64 = globe::grow(&Input::of(&u, 0, 64, 1.0), a.threads).ms.iter().sum();
        st.add(&u, globe_ms, g, &g128, g64);
        if seed < a.first + a.images {
            nebula_image(&a.out, seed, &u, &cfg, &a.params);
            suite_image(&a.out, seed, &u);
            for id in 0..u.bodies.len().min(4) {
                let gl = globe::grow(&Input::of(&u, id, 32, 1.0), a.threads);
                globe_image(&format!("{}/globe_{seed}_{id}.ppm", a.out), &gl);
            }
            let _ = writeln!(examples, "seed {seed}: {} systems, {} bodies", u.systems, u.bodies.len());
            for b in u.bodies.iter().take(12) {
                let t = &b.traits;
                let s = &u.suites[t.suite as usize];
                let _ = writeln!(
                    examples,
                    "  #{:<2} {:<12} {:?}/{:?} half {:.2e} r {:.2e} ρ {:.2} Y {:.1e} Π {:.1e} g {:.2} heat {:.2} T {} air {:?} life {} glow {} suite {}{} [{}]",
                    b.id,
                    b.name(),
                    t.rank,
                    t.form,
                    b.half as f64,
                    b.radius as f64,
                    b.density,
                    t.yield_stress,
                    t.pi_g,
                    t.gravity,
                    t.heat as f64 / 256.0,
                    t.temp,
                    t.air_top,
                    t.life,
                    t.glow,
                    t.suite,
                    if s.fallback { " (fallback)" } else { "" },
                    s.minerals.iter().map(|m| format!("{}:{}", m.amount, m.cohesion)).collect::<Vec<_>>().join(" ")
                );
            }
        }
        if (seed - a.first + 1) % 100 == 0 {
            eprintln!("{} seeds, {:.0} s", seed - a.first + 1, wall.elapsed().as_secs_f64());
        }
    }
    let log_edges = |lo: i32, hi: i32| (lo..=hi).map(|k| 10f64.powi(k)).collect::<Vec<_>>();
    let lin = |lo: f64, step: f64, n: usize| (0..n).map(|k| lo + step * k as f64).collect::<Vec<_>>();
    println!("## Examples\n{examples}");
    println!("## Bodies");
    print!("{}", st.bodies.histogram("bodies per seed (start included)", &lin(0.0, 4.0, 17)));
    print!("{}", st.systems.histogram("systems per seed", &lin(0.0, 2.0, 14)));
    println!("basins per seed: {}", st.basins.summary());
    print!("{}", st.half.histogram("half-size of every body (blocks)", &log_edges(3, 9)));
    print!("{}", st.radius_round.histogram("datum radius of round bodies (blocks)", &log_edges(3, 9)));
    print!("{}", st.form.line("forms"));
    print!("{}", st.rank.line("ranks"));
    println!("seeds with both a cube layout and a round one: {:.1}%", 100.0 * st.both_layouts as f64 / st.seeds as f64);
    print!("{}", st.pi.histogram("Π_g of every body", &log_edges(-2, 6)));
    println!("Π outside the genesis table (clamped): {} bodies", st.pi_clamped);
    println!("\n## Physics");
    print!("{}", st.gravity.histogram("surface gravity (m/s²)", &[0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 24.0, 30.0, 100.0]));
    print!("{}", st.heat.histogram("heat (start = 1)", &[0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 1.5, 3.0, 10.0]));
    print!("{}", st.temp.histogram("temperature (K)", &lin(0.0, 50.0, 12)));
    print!("{}", st.air.line("air"));
    println!("air top (blocks): {}", st.air_top.summary());
    print!("{}", st.life.histogram("life (0..255)", &[0.0, 1.0, 8.0, 16.0, 32.0, 64.0, 128.0]));
    println!("glowing bodies: {}", st.glow);
    println!("\n## Suites");
    print!("{}", st.suites.histogram("suites per seed", &lin(0.0, 4.0, 12)));
    println!("### The law's minerals (every suite the law was asked for, before any fallback)");
    print!("{}", st.own_amount.histogram("amount", &lin(1.0, 3.0, 11)));
    print!("{}", st.own_cohesion.histogram("cohesion (Q8; yield rises from 384 to 768)", &lin(-200.0, 100.0, 12)));
    print!("{}", st.own_amount_spread.histogram("amount spread per suite", &lin(0.0, 2.0, 16)));
    print!("{}", st.own_colour_spread.histogram("colour spread per suite (0..765)", &lin(0.0, 40.0, 16)));
    print!("{}", st.causes.line("verdict (fallback causes)"));
    print!(
        "{}",
        st.own_restless.histogram(
            &format!("universal materials (of {}) each mineral reacts with, first 100 seeds", st.own_restless_of),
            &[0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0, 32.0]
        )
    );
    println!("### Suites kept (not fallback)");
    print!("{}", st.suite_amount.histogram("mineral amount", &lin(1.0, 3.0, 11)));
    print!("{}", st.suite_amount_spread.histogram("amount spread per suite", &lin(0.0, 2.0, 16)));
    print!("{}", st.suite_cohesion.histogram("mineral cohesion", &lin(-200.0, 100.0, 12)));
    print!("{}", st.suite_colour_spread.histogram("colour spread per suite", &lin(0.0, 40.0, 16)));
    print!("{}", st.suite_density.histogram("suite density (mantle mean amount)", &lin(1.0, 3.0, 11)));
    print!("{}", st.suite_yield.histogram("suite yield", &log_edges(4, 8)));
    println!("distinct minerals after repair: {}", st.suite_distinct.summary());
    println!("differentiation sweeps: {}", st.suite_sweeps.summary());
    println!("occurrences dropped by repair: {}", st.suite_dropped.summary());
    println!("layers the rest check replaced: {}", st.suite_replaced.summary());
    println!(
        "fallback suites: {} of {} ({:.1}%)",
        st.suite_fallback,
        st.suite_count,
        100.0 * st.suite_fallback as f64 / st.suite_count.max(1) as f64
    );
    println!("\n## Start world");
    println!("density: {}", st.start_density.summary());
    println!("datum radius: {}", st.start_radius.summary());
    println!("composition re-salts: {}", st.start_tries.summary());
    println!("`rock` fallback: {} seeds", st.start_fallback);
    println!("interest re-salts: {}", st.resalts.summary());
    println!("\n## Leftovers");
    println!("debris share of nebula mass: {}", st.debris.summary());
    println!("rings: {}", st.rings.summary());
    println!("contact binaries: {}", st.binaries.summary());
    println!("bodies fallen onto their parent: {}", st.fell.summary());
    println!("bodies out of bounds: {}", st.out_of_bounds.summary());
    println!("bodies over the cap of {}: {}", a.params.bodies_max, st.capped.summary());
    println!("\n## Storage");
    println!("shelf rows: {}", st.storage_rows.summary());
    println!("row 0 x fill: {}", st.storage_row0.summary());
    println!("warped boxes' y fill: {}", st.storage_cube_y.summary());
    println!("demoted (did not fit): {}", st.storage_demoted.summary());
    println!("\n## Creation time (ms, {} threads)", a.threads);
    println!("universe total: {}", st.t_total.summary());
    println!("  nebula: {}", st.t_nebula.summary());
    println!("  accretion: {}", st.t_accrete.summary());
    println!("  suites: {}", st.t_suites.summary());
    println!("  settle (shape, hierarchy, traits): {}", st.t_settle.summary());
    println!("  storage: {}", st.t_storage.summary());
    print!("{}", st.globe_g.line("start globe resolution"));
    println!("start globe at its resolution: {}", st.t_globe.summary());
    println!("creation incl. start globe: {}", st.t_create.summary());
    println!("per suite: {}", st.suite_ms.summary());
    println!("start globe forced to G=64: {}", st.t_globe64.summary());
    println!("start globe forced to G=128: {}", st.t_globe128.summary());
    for (k, name) in globe::PHASES.iter().enumerate() {
        println!("  {name}: {}", st.globe_phase[k].summary());
    }
    println!("\n## Gates");
    print!("{}", st.gates());
    eprintln!("done in {:.0} s", wall.elapsed().as_secs_f64());
}
