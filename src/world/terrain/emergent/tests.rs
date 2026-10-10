//! The lab's guards: generation code calls no libm, its helpers match what they replace, the
//! nebula conserves mass for any thread count, universes are sane, and every stage's bytes are
//! pinned per seed (the same pins on every machine are the cross-machine check).

use super::globe::{self, Input};
use super::nebula::Nebula;
use super::*;
use crate::mechanics::material::{cohesion01, mechanical_response};

/// Seeds the digests are pinned for: GOLDEN_SEED, 42 and the owner's seed.
const SEEDS: [u64; 3] = [0xC0FFEE, 42, 1_791_184_794_939_118_871];

/// `(nebula, catalog, suites, start globe at G=32)` per seed.
const PINS: [[u64; 4]; 3] = [
    [0xc9eb33927aa562fa, 0x06841ffe54f7bfae, 0xf02949df449ea31b, 0x78d5dea9c373a504],
    [0x14276b40ffcd9ae0, 0x3e1f91ec5e582658, 0x8d1e4c780a06985c, 0xc0777c8deca31ed6],
    [0xdcdcba074d536dbe, 0x7a4702cb079c0af8, 0x89ba8e8f7a7c92a0, 0xbd95e746bde93ca5],
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
fn yield_matches_the_prototype_response() {
    for c in 300..800 {
        let want = mechanical_response(5, cohesion01(c)).yield_stress;
        assert!((minerals::yield_of(c) / want - 1.0).abs() < 1e-12, "cohesion {c}");
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

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn digests(seed: u64) -> [u64; 4] {
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
