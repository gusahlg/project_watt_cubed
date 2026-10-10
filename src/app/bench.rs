//! The self-describing benchmark (`WATT_BENCH=<seconds>`): enter a reproducible world, wait for
//! both the warmup floor and streaming readiness, rotate the camera, and hand every measured
//! frame to the recorder.
use std::time::Duration;

use voxel_engine::{DVec3, Engine};

use super::{App, Screen};
use crate::benchmark::{Benchmark, Step};
use crate::game::Game;
use crate::settings::Settings;

impl App {
    /// Drive one benchmark frame. False once the report is out and the app should exit.
    pub(super) fn bench_frame(&mut self, eng: &mut Engine) -> bool {
        let dt = eng.frame_time();
        let Some(bench) = self.bench.as_mut() else { return true };
        if !bench.has_started() {
            bench.begin();
            let start = (bench.position(), bench.look(), bench.day());
            self.enter_bench_world(eng, start);
            return true;
        }
        let Self { bench: Some(bench), screen, settings, .. } = self else { return true };
        let Screen::Playing(game) = screen else { return true };
        // One unsampled frame after Complete has presented; capture that image
        // (blocking) before the report so the readback is outside the samples.
        if bench.measurement_complete() {
            return finish(bench, settings, eng, game);
        }
        // A slow spin (`WATT_BENCH_YAW`, default 0.4 rad/s; 0 = static) sweeps
        // the frustum; an optional flight along +X (`WATT_BENCH_MOVE`)
        // exercises paths a parked camera never touches.
        game.player_mut().orientation.yaw += bench.yaw_rate() as f32 * dt;
        bench.apply_move(game.player_mut(), dt);
        let (ready, gauges) = bench.poll_world(game.world());
        match bench.step(dt, ready, gauges, eng.frames_rendered(), eng.frames_coalesced()) {
            Step::ReadyTimeout => {
                eprintln!("{}", game.world().entry_debug());
                true
            }
            Step::Warming => {
                if !ready && bench.wait_log_due() {
                    eprintln!("benchmark: waiting for world ({})", game.world().entry_debug());
                }
                true
            }
            Step::Measuring => true,
            Step::Complete if bench.screenshot_path().is_some() => true,
            Step::Complete => finish(bench, settings, eng, game),
        }
    }

    /// The bench's world: uncapped, a pinned seed, input locked, and the requested day, view and
    /// far position (`start`).
    fn enter_bench_world(&mut self, eng: &mut Engine, start: (Option<DVec3>, Option<(f32, f32)>, Option<f64>)) {
        let (pos, look, day) = start;
        // Uncapped and unsynced, or the bench measures the throttle (the report states both).
        // Entering the world pushes the graphics settings.
        self.settings.vsync = false;
        self.settings.max_fps = 0;
        self.start_new_world(eng);
        let Screen::Playing(game) = &mut self.screen else { return };
        game.set_input_locked(true);
        if let Some(day) = day {
            game.set_day(day);
        }
        // Far-coordinate bench: park the player at the requested position
        // with the ground under them made real, and give streaming a
        // little extra warmup to catch up before sampling starts.
        if let Some((yaw, pitch)) = look {
            game.player_mut().orientation.yaw = yaw;
            game.player_mut().orientation.pitch = pitch;
        }
        if let Some(pos) = pos {
            game.player_mut().position = pos;
            game.player_mut().set_flying(true);
            game.world_mut().prepare_around(pos);
            // Stand up along the local pull there, as a teleport does (any face of any body).
            let weightless = 0.02 * crate::player::STANDARD_GRAVITY;
            let pull = game.world().gravity_at(pos);
            game.player_mut().gravity = pull.accel;
            if let Some(up) = pull.up(weightless) {
                game.player_mut().snap_up(up);
            }
            if let Some(bench) = &mut self.bench {
                bench.add_warmup(Duration::from_secs(2));
            }
        }
    }
}

/// Capture the last presented frame if requested, then emit the report.
fn finish(bench: &mut Benchmark, settings: &Settings, eng: &mut Engine, game: &Game) -> bool {
    bench.capture_screenshot(eng);
    bench.finish(settings, eng, game.world(), game.player().position).emit();
    false
}
