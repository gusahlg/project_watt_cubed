//! The lab's guards: generation code calls no libm, its helpers match what they replace, the
//! nebula conserves mass for any thread count, universes are sane, and every stage's bytes are
//! pinned per seed (the same pins on every machine are the cross-machine check).

use super::nebula::Nebula;
use super::*;

/// Seeds the digests are pinned for: GOLDEN_SEED, 42 and the owner's seed.
#[cfg(feature = "dev-tools")]
const SEEDS: [u64; 3] = [0xC0FFEE, 42, 1_791_184_794_939_118_871];

/// `(nebula, catalog, suites, start globe at G=32)` per seed.
#[cfg(feature = "dev-tools")]
const PINS: [[u64; 4]; 3] = [
    [0xc9eb33927aa562fa, 0x6b1c643a854df8f1, 0x6935160005a0a303, 0xb35ca704b2e41828],
    [0x14276b40ffcd9ae0, 0xdcd25efe3aa85d89, 0x5b4086c57a5864d2, 0xc0777c8deca31ed6],
    [0xdcdcba074d536dbe, 0xe01207b66de43499, 0xdb9ecabd77c12c21, 0xbd95e746bde93ca5],
];

/// Calls whose results may differ between math libraries.
const BANNED: &[&str] = &[
    "exp", "exp2", "exp_m1", "ln", "ln_1p", "log", "log2", "log10", "pow", "powi", "powf", "cbrt", "sin", "cos", "tan",
    "sin_cos", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "hypot", "mul_add",
];

#[test]
fn generation_code_calls_no_libm() {
    let root = env!("CARGO_MANIFEST_DIR");
    let mut checked = 0;
    for dir in ["src/world/terrain/emergent", "crates/field/src"] {
        for entry in std::fs::read_dir(format!("{root}/{dir}")).expect("source dir") {
            let path = entry.expect("entry").path();
            if path.extension().is_none_or(|e| e != "rs") || path.file_name().is_some_and(|f| f == "tests.rs") {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("source");
            for (n, line) in text.lines().enumerate() {
                let code = line.split("//").next().unwrap_or("");
                for name in BANNED {
                    for call in [format!(".{name}("), format!("::{name}(")] {
                        assert!(!code.contains(&call), "{}:{}: `{call}` in generation code", path.display(), n + 1);
                    }
                }
            }
            checked += 1;
        }
    }
    assert!(checked >= 8, "scanned {checked} files");
}

#[test]
fn cube_root_and_log2_are_exact_enough() {
    for k in -40..60 {
        for m in [1.0, 1.37, 1.414, 1.415, 1.7, 1.999_999] {
            let x = m * 2f64.powi(k);
            assert!((cbrt(x) / x.cbrt() - 1.0).abs() < 1e-15, "cbrt {x}");
            assert!((log2(x) - x.log2()).abs() < 1e-13, "log2 {x}");
        }
    }
}

#[test]
fn layouts_match_the_genesis_table() {
    for k in -30..140 {
        let pi = 2f64.powf(k as f64 * 0.11);
        for half in [3.0e4, 6.0e6, 2.5e7] {
            let ours = layout(pi, half);
            let (theirs, tilt) = genesis::tabulated_layout(pi, half).expect("a table with layouts");
            assert!((ours.tilt - tilt).abs() < 1e-9, "Π {pi}: tilt {} vs {tilt}", ours.tilt);
            match theirs {
                genesis::Layout::Round { radius, .. } => {
                    assert_eq!(ours.form, Form::Round, "Π {pi}");
                    assert!((ours.radius / radius - 1.0).abs() < 1e-9, "Π {pi}");
                }
                genesis::Layout::Cube => assert_ne!(ours.form, Form::Round, "Π {pi}"),
            }
        }
    }
}

#[test]
fn the_nebula_conserves_mass_for_any_thread_count() {
    let cfg = TerrainCfg::default();
    let p = Params::default();
    let still = Nebula::new(7, &cfg, &Params { collapse: 0, ..p.clone() }, 1);
    let one = Nebula::new(7, &cfg, &p, 1);
    let four = Nebula::new(7, &cfg, &p, 4);
    assert_eq!(one.cells, four.cells);
    let sum = |n: &Nebula| n.cells.iter().map(|c| c.m).sum::<u64>();
    assert_eq!(sum(&still), sum(&one));
    assert!(one.systems[0].origin && one.systems.iter().skip(1).all(|s| !s.origin));
}

#[cfg(feature = "dev-tools")]
fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

#[cfg(feature = "dev-tools")]
fn digests(seed: u64) -> [u64; 4] {
    use super::globe::{self, Input};
    let cfg = TerrainCfg::default();
    let p = Params::default();
    let u = Universe::new(seed, &cfg, &p, 2);
    let s = nebula_seed(seed, u.resalts);
    let mut d = [0xcbf2_9ce4_8422_2325u64; 4];
    for c in Nebula::new(s, &cfg, &p, 3).cells {
        fnv(&mut d[0], &c.m.to_le_bytes());
        fnv(&mut d[0], &c.comp.map(|v| v as u8));
    }
    for b in &u.bodies {
        let t = &b.traits;
        for v in [b.id as i64, b.system as i64, b.half, b.radius, b.centre[0], b.centre[1], b.centre[2], t.rank as i64] {
            fnv(&mut d[1], &v.to_le_bytes());
        }
        for v in [b.density, t.mass, t.yield_stress, t.pi_g, t.gravity as f64, b.history.heat_in] {
            fnv(&mut d[1], &v.to_bits().to_le_bytes());
        }
        fnv(&mut d[1], &[t.form as u8, t.glow as u8, t.wet, t.life]);
        fnv(&mut d[1], &t.heat.to_le_bytes());
        fnv(&mut d[1], &t.temp.to_le_bytes());
        fnv(&mut d[1], &t.air_top.unwrap_or(-1).to_le_bytes());
        fnv(&mut d[1], &t.suite.to_le_bytes());
        fnv(&mut d[1], &t.name);
    }
    for s in &u.suites {
        for m in &s.minerals {
            fnv(&mut d[2], m.config.encode().as_bytes());
        }
        fnv(&mut d[2], &[s.fallback as u8, s.glow as u8]);
    }
    let g = globe::grow(&Input::of(&u, 0, 32, 1.0), 2);
    for n in &g.nodes {
        fnv(&mut d[3], &n.elev.to_le_bytes());
        fnv(&mut d[3], &n.albedo.to_le_bytes());
        fnv(&mut d[3], &[n.temp as u8, n.wet, n.flow, n.rock, n.fill, n.life, n.plate]);
    }
    d
}

#[cfg(feature = "dev-tools")]
#[test]
fn stage_digests_are_pinned() {
    let got: Vec<[u64; 4]> = SEEDS.iter().map(|&s| digests(s)).collect();
    if got != PINS {
        let rows: Vec<String> = got.iter().map(|d| format!("    [{:#018x}, {:#018x}, {:#018x}, {:#018x}],", d[0], d[1], d[2], d[3])).collect();
        panic!("stage digests changed; if intended, pin:\nconst PINS: [[u64; 4]; 3] = [\n{}\n];", rows.join("\n"));
    }
}

#[test]
fn a_universe_is_the_same_twice_and_sane() {
    let cfg = TerrainCfg::default();
    let p = Params::default();
    for seed in [3u64, 99] {
        let u = Universe::new(seed, &cfg, &p, 1);
        let v = Universe::new(seed, &cfg, &p, 4);
        assert_eq!(u.bodies, v.bodies, "seed {seed}");
        assert_eq!(u.suites, v.suites, "seed {seed}");
        let start = &u.bodies[0];
        assert_eq!((start.traits.rank, start.traits.form), (Rank::Start, Form::Round));
        assert!(start.traits.air_top.is_some());
        assert!((start.density * start.radius as f64 - RHO_R).abs() <= 8.0 * start.density, "the spawn contract (to half a snap)");
        assert!((start.traits.gravity as f64 / start_pull() - 1.0).abs() < 1e-3, "the spawn pull: {}", start.traits.gravity);
        assert!(u.bodies.len() > OTHERS_MIN || u.resalts == INTEREST_TRIES);
        for (i, a) in u.bodies.iter().enumerate() {
            for b in &u.bodies[i + 1..] {
                let d = dist(a.centre.map(|v| v as f64), b.centre.map(|v| v as f64));
                let touch = if a.traits.partner == Some(b.id) { (a.half + b.half) as f64 } else { a.reach() + b.reach() };
                assert!(d > touch, "{} and {} overlap", a.id, b.id);
                if a.system != b.system {
                    assert!(d > R_G + a.reach() + b.reach(), "systems {} and {} pull on each other", a.system, b.system);
                }
            }
        }
    }
}

/// Grow `seeds` on `threads` threads (each universe on one), in seed order.
fn grow_all(seeds: std::ops::Range<u64>, threads: u64) -> Vec<Universe> {
    let cfg = TerrainCfg::default();
    let p = Params::default();
    let chunk = (seeds.end - seeds.start).div_ceil(threads);
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|k| {
                let lo = seeds.start + k * chunk;
                let hi = (lo + chunk).min(seeds.end);
                let (cfg, p) = (&cfg, &p);
                s.spawn(move || (lo..hi).map(|seed| Universe::new(seed, cfg, p, 1)).collect::<Vec<_>>())
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().expect("a universe thread")).collect()
    })
}

/// The properties every universe keeps, whatever its seed: a valid start world that pulls as
/// designed, company, no overlap, systems that never pull on each other, and satellites that
/// dominate their own ground.
fn check_universe(u: &Universe, p: &Params) {
    let start = &u.bodies[0];
    let t = &start.traits;
    let seed = u.seed;
    assert_eq!((t.rank, t.form), (Rank::Start, Form::Round), "seed {seed}: the start world is round");
    assert!(t.air_top.is_some(), "seed {seed}: the start world has air");
    assert!(start_fallback_or(u, |temp| (TEMPERATE.0..=TEMPERATE.1).contains(&temp)), "seed {seed}: temperate start ({})", t.temp);
    assert!((start.density * start.radius as f64 - RHO_R).abs() <= 8.0 * start.density, "seed {seed}: the spawn contract");
    assert!((t.gravity as f64 / start_pull() - 1.0).abs() < 1e-3, "seed {seed}: the spawn pull {}", t.gravity);
    assert!(u.bodies.len() > OTHERS_MIN || u.resalts == INTEREST_TRIES, "seed {seed}: company");
    assert!(u.bodies.len() <= p.bodies_max, "seed {seed}: the cap");
    let surface = |b: &Body| match b.traits.form {
        Form::Round => G * b.traits.mass / (b.radius as f64 * b.radius as f64),
        _ => KAPPA_FACE * G * b.density * b.half as f64,
    };
    let radius = |b: &Body| if b.traits.form == Form::Round { b.radius as f64 } else { b.half as f64 };
    for (i, a) in u.bodies.iter().enumerate() {
        assert_eq!(a.id as usize, i, "seed {seed}: ids are indices");
        for b in &u.bodies[i + 1..] {
            let d = dist(a.centre.map(|v| v as f64), b.centre.map(|v| v as f64));
            let touch = if a.traits.partner == Some(b.id) { (a.half + b.half) as f64 } else { a.reach() + b.reach() };
            assert!(d > touch, "seed {seed}: {} and {} overlap", a.id, b.id);
            if a.system != b.system {
                assert!(d > R_G + a.reach() + b.reach(), "seed {seed}: systems {} and {} pull on each other", a.system, b.system);
            }
        }
        if let Some(parent) = a.traits.parent.filter(|_| a.traits.rank == Rank::Satellite) {
            let up = &u.bodies[parent as usize];
            let d = (dist(a.centre.map(|v| v as f64), up.centre.map(|v| v as f64)) - radius(a)).max(1.0);
            let theirs = pull_at(up.traits.mass, d);
            assert!(surface(a) >= p.k_dom * theirs * (1.0 - 1e-6), "seed {seed}: satellite {} pulls {} under {} × its parent's {theirs}", a.id, surface(a), p.k_dom);
        }
        if let Some(partner) = a.traits.partner {
            let b = &u.bodies[partner as usize];
            assert_eq!(b.traits.partner, Some(a.id), "seed {seed}: partners point at each other");
            assert!(a.traits.form != Form::Round && b.traits.form != Form::Round, "seed {seed}: a binary is two cubes");
        }
    }
}

/// `pred` of the start world's temperature, or the last-resort `rock` start world.
fn start_fallback_or(u: &Universe, pred: impl Fn(f64) -> bool) -> bool {
    u.start_fallback || pred(u.bodies[0].traits.temp as f64)
}

/// The P2 property suite over 1000 seeds (`EMERGENT_SEEDS` overrides).
#[test]
fn a_thousand_universes_keep_their_properties() {
    let n: u64 = std::env::var("EMERGENT_SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(1000);
    let p = Params::default();
    for u in grow_all(1..n + 1, 4) {
        check_universe(&u, &p);
    }
}

/// `cargo test --release --lib ten_thousand_universes -- --ignored`: no seed panics.
#[test]
#[ignore]
fn ten_thousand_universes_grow_without_panic() {
    let p = Params::default();
    let all = grow_all(0..10_000, 4);
    assert_eq!(all.len(), 10_000);
    for u in &all {
        check_universe(u, &p);
    }
}

/// The same seed grows the same universe on one thread and on four, and twice in a row: what a
/// server and every joiner build is one universe.
#[test]
fn two_growths_of_a_seed_are_equal() {
    let cfg = TerrainCfg::default();
    let p = Params::default();
    for seed in [0u64, 157, 240, 1_791_184_794_939_118_871, u64::MAX] {
        let a = Universe::new(seed, &cfg, &p, 1);
        let b = Universe::new(seed, &cfg, &p, 4);
        assert_eq!(a.bodies, b.bodies, "seed {seed}");
        assert_eq!(a.suites, b.suites, "seed {seed}");
        assert_eq!(a.resalts, b.resalts, "seed {seed}");
    }
    // All 64 bits count.
    let low = Universe::new(5, &cfg, &p, 2);
    let high = Universe::new(5 | (1 << 40), &cfg, &p, 2);
    assert_ne!(low.bodies, high.bodies, "the high half of the seed grows another universe");
}
