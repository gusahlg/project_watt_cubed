//! `golden_compare` — compare captured frames against blessed references (PNG files only).
//!
//! The engine has no headless capture mode, so frames are captured manually
//! (F2 screenshot) and dropped into captures dir. This tool reads PNGs:
//!
//!   cargo run --bin golden_compare              # compare captures/ vs blessed refs
//!   cargo run --bin golden_compare -- bless     # promote captures/ -> blessed refs
//!   cargo run --bin golden_compare -- poses     # print the capture instructions
//!
//! Uses the same pose registry, threshold, and diff as the harness — stays
//! in sync without duplication.

use std::path::PathBuf;

use project_watt_cubed::harness::{diff, golden_shots, is_uniform, GoldenShot, GOLDEN_DIR};
use voxel_engine::load_png;

const CAPTURES_DIR: &str = "tests/golden/captures";

const MAX_PCT_CHANGED: f32 = 0.5;

fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(GOLDEN_DIR).join(format!("{name}.png"))
}

fn capture_path(name: &str) -> PathBuf {
    PathBuf::from(CAPTURES_DIR).join(format!("{name}.png"))
}

fn main() {
    let mode = std::env::args().nth(1);
    let code = match mode.as_deref() {
        None => compare(),
        Some("bless") => bless(),
        Some("poses") => {
            print_poses();
            0
        }
        Some(other) => {
            eprintln!("unknown mode '{other}' — use (nothing) | bless | poses");
            2
        }
    };
    std::process::exit(code);
}

fn compare() -> i32 {
    let mut failures = 0;
    for shot in golden_shots() {
        let cap = capture_path(shot.name);
        let git = golden_path(shot.name);
        let got = match load_png(&cap) {
            Ok(s) => s,
            Err(e) => {
                println!("FAIL {:<20} no capture: {} ({e})", shot.name, cap.display());
                failures += 1;
                continue;
            }
        };
        let want = match load_png(&git) {
            Ok(s) => s,
            Err(e) => {
                println!("FAIL {:<20} no golden: {} ({e})", shot.name, git.display());
                failures += 1;
                continue;
            }
        };
        let stats = diff(&got, &want);
        if stats.pct_changed > MAX_PCT_CHANGED {
            println!(
                "FAIL {:<20} {:.3}% changed (max {MAX_PCT_CHANGED}%), peak delta {}",
                shot.name, stats.pct_changed, stats.max_channel_delta
            );
            failures += 1;
        } else {
            println!("ok   {:<20} {:.3}% changed", shot.name, stats.pct_changed);
        }
    }
    if failures == 0 {
        println!("golden_compare: all {} poses clean", golden_shots().len());
        0
    } else {
        eprintln!("golden_compare: {failures} pose(s) failed");
        1
    }
}

/// Rejects uniform frames (black-screenshot failure) to match harness guard.
fn bless() -> i32 {
    let mut errors = 0;
    for shot in golden_shots() {
        let cap = capture_path(shot.name);
        let got = match load_png(&cap) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skip {:<20} no capture: {} ({e})", shot.name, cap.display());
                errors += 1;
                continue;
            }
        };
        if is_uniform(&got) {
            eprintln!(
                "skip {:<20} UNIFORM frame ({}x{}) — not a real scene, refusing to bless",
                shot.name, got.width, got.height
            );
            errors += 1;
            continue;
        }
        let dst = golden_path(shot.name);
        if let Err(e) = std::fs::copy(&cap, &dst) {
            eprintln!("skip {:<20} copy failed: {e}", shot.name);
            errors += 1;
            continue;
        }
        println!("blessed {:<20} -> {}", shot.name, dst.display());
    }
    if errors == 0 {
        0
    } else {
        1
    }
}

fn print_poses() {
    println!("Manual capture — run the app, reach each pose, press F2, then move the");
    println!("resulting PNG from screenshots/ to {CAPTURES_DIR}/<name>.png\n");
    println!("Fixed for every pose: seed, resolution 1280x720, RenderConfig::golden().\n");
    println!(
        "{:<20} {:>10} {:>10} {:>10} {:>7} {:>7} {:>5}",
        "name", "x", "y", "z", "yaw", "pitch", "day"
    );
    for GoldenShot { name, cam, day, seed, .. } in golden_shots() {
        println!(
            "{name:<20} {:>10.2} {:>10.2} {:>10.2} {:>7.3} {:>7.3} {day:>5.2}  seed={seed:#x}{}",
            cam.pos.x,
            cam.pos.y,
            cam.pos.z,
            cam.yaw,
            cam.pitch,
            if name == "cave_interior" { "  (setup: carve_cave)" } else { "" }
        );
        println!("    capture -> {}", capture_path(name).display());
    }
}
