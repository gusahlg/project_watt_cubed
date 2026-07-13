//! shadow_probe — manual diagnostic (not part of the acceptance suite).
//!
//! Verifies view-independence of the shadow pass end-to-end on the live
//! renderer: capture pose A, swing the camera away and back, recapture A —
//! the two captures must be pixel-identical up to the dither/exposure noise
//! floor (`pct_changed: 0.0`). Regression guard for the 2026-07-13 pair of
//! shadow bugs (shadow_depth.vert stride mismatch scattering occluders as the
//! draw sort reordered; camera-anchored cascade fit). Also captures A with
//! shadows off so the shadow mask itself is visible in a manual diff.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use project_watt_cubed::game::Game;
use project_watt_cubed::harness::{diff, CameraPose, GOLDEN_SEED};
use project_watt_cubed::input::router::Router;
use project_watt_cubed::mods::Mods;
use project_watt_cubed::render_config::RenderConfig;
use project_watt_cubed::settings::Settings;
use voxel_engine::DVec3;

const POSE_A: CameraPose = CameraPose { pos: DVec3::new(0.0, 110.0, 0.0), yaw: 0.24, pitch: -0.35 };
const POSE_B: CameraPose = CameraPose { pos: DVec3::new(0.0, 110.0, 0.0), yaw: 1.24, pitch: -0.1 };

/// (capture name, pose, shadows on)
const STEPS: [(&str, CameraPose, bool); 4] = [
    ("a_before", POSE_A, true),
    ("b_swing", POSE_B, true),
    ("a_after", POSE_A, true),
    ("a_noshadow", POSE_A, false),
];

fn shot_path(name: &str) -> PathBuf {
    PathBuf::from(format!("/tmp/watt-shadow/{name}.png"))
}

fn main() {
    let mut mods = Mods::with_defaults();
    let mut settings = Settings::default();
    let mut router = Router::new();

    let render = RenderConfig { shadows: true, ..RenderConfig::golden() };
    let config = voxel_engine::Config {
        title: "shadow-probe".into(),
        width: 1280,
        height: 720,
        vsync: false,
        resizable: false,
        flags: render.engine_flags(),
        ..Default::default()
    };

    let mut game: Option<Game> = None;
    let mut settle: Option<Instant> = None;
    let mut step = 0usize;

    voxel_engine::run(config, move |eng| {
        let g = game.get_or_insert_with(|| {
            let mut g = Game::scripted(GOLDEN_SEED, render);
            g.teleport(STEPS[0].1);
            g.set_day(0.35);
            g
        });
        g.update(eng, &mut router, &mut mods, &mut settings);
        g.draw(eng, &mut mods, settings.fov, 0.0);

        if !g.world().entry_complete() || !g.world().far_field_refined() {
            return true;
        }
        let started = *settle.get_or_insert_with(Instant::now);
        // Long first settle (exposure smoother), short re-settles after pose/flag flips.
        let need = if step == 0 { Duration::from_secs(4) } else { Duration::from_secs(1) };
        if started.elapsed() < need {
            return true;
        }
        let (name, _, _) = STEPS[step];
        voxel_engine::skeleton::screenshot_to(eng, &shot_path(name)).expect("capture failed");
        eprintln!("captured {name}");
        step += 1;
        if step >= STEPS.len() {
            return false;
        }
        let (_, pose, shadows_on) = STEPS[step];
        g.teleport(pose);
        eng.set_flags(RenderConfig { shadows: shadows_on, ..render }.engine_flags());
        settle = Some(Instant::now());
        true
    });

    // Post-run: decode and report. a_before vs a_after must be ~identical.
    let load = |n: &str| voxel_engine::skeleton::load_png(&shot_path(n)).expect("decode");
    let before = load("a_before");
    let after = load("a_after");
    let noshadow = load("a_noshadow");
    let stable = diff(&before, &after);
    let mask = diff(&before, &noshadow);
    println!("A-before vs A-after   : {stable:?}");
    println!("A-shadow vs A-noshadow: {mask:?}");
}
