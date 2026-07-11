//! `golden` — run the golden-shot acceptance harness.
//!
//!   cargo run --bin golden            # check the current acceptance set
//!   cargo run --bin golden -- bless   # regenerate blessed goldens
//!
//! The four shots (`night_field`, `cave_interior`, `shadow_boundary`,
//! `horizon_fog_vs_sky`) ship WITHOUT blessed PNGs, so `bless` must run once
//! before a plain check can pass their ImageMatch — until then each fails LOUD
//! (a "load golden … No such file" `Failure`, never a panic).
//!
//! A thin runner over `harness::check`: builds the phase-A acceptance set,
//! prints the entry-time number on the golden seed, then runs every
//! criterion (all live now — image match, sky-holes, entry/frame time, and
//! the `NoProvisional` filesystem sweep).

use std::time::Duration;

use project_watt_cubed::harness::{
    Acceptance, Criterion, GOLDEN_SEED, Phase, golden_shots, run_acceptance,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    // The harness's render lanes are a typed property of each stage
    // (`RenderConfig::golden`): tiles ON (the far-field filler the `SkyHoleCount`
    // detector needs — without it the chunk→tile handoff band reads as bare sky)
    // and blocklight ON (`cave_interior`'s emitter; a no-op for the emitter-free
    // shots). Threaded through `Game::scripted` / `scripted_config`, not env vars.
    let bless = args.iter().any(|a| a == "bless");
    // Phase advances as the project does; default to the latest implemented (E).
    // `--phase <pre-A|A|B|C|D|E>` overrides — the sweep is cumulative, so it
    // fails on any `PROVISIONAL(p)` marker for p at or below the given phase.
    let phase = args
        .iter()
        .position(|a| a == "--phase")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| Phase::parse(s))
        .unwrap_or(Phase::E);
    let acc = default_acceptance(phase);

    println!(
        "golden: phase {}, {} criteria{}",
        acc.phase,
        acc.criteria.len(),
        if bless { " (bless)" } else { "" }
    );

    // ONE event loop for the whole process: `run_acceptance` drives every shot
    // and the golden-seed entry time inside a single `voxel_engine::run`, then
    // evaluates the pure criteria over the captured PNGs.
    let report = run_acceptance(&acc, bless);

    println!(
        "golden: entry_time(seed={GOLDEN_SEED:#x}) = {:?}",
        report.golden_entry_time
    );

    match report.result {
        Ok(()) => println!("golden: all criteria passed"),
        Err(failures) => {
            eprintln!("golden: {} criteria failed:", failures.len());
            for f in &failures {
                eprintln!("  - {}: {}", f.what, f.detail);
            }
            std::process::exit(1);
        }
    }
}

/// The acceptance set for `phase`: an image match + sky-hole check on each golden
/// shot, the entry-time ceiling, a frame-time ceiling on `spawn_forward`, and
/// that phase's cumulative provisional-marker sweep.
fn default_acceptance(phase: Phase) -> Acceptance {
    let shots = golden_shots();
    let mut criteria = Vec::new();
    for shot in &shots {
        criteria.push(Criterion::ImageMatch { shot: *shot, max_pct_changed: 0.5 });
        criteria.push(Criterion::SkyHoleCount { shot: *shot, max: 0 });
    }
    // Frame-time ceiling on the primary shot. 16.7 ms is the 60 fps budget.
    let spawn_forward = shots.iter().find(|s| s.name == "spawn_forward").copied().expect("golden_shots always defines spawn_forward");
    criteria.push(Criterion::FrameTime {
        shot: spawn_forward,
        // First-cut ceiling; tighten after the first measured report.
        max_ms: 16.7,
    });
    criteria.push(Criterion::EntryTime { seed: GOLDEN_SEED, max: Duration::from_secs(5) });
    criteria.push(Criterion::NoProvisional { through: phase });
    Acceptance { phase, criteria }
}
