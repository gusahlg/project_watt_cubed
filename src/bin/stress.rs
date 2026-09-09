//! `stress` — the fast-flight streaming stress scenario (the user-visible
//! "fly fast, stop, and the game stays laggy" symptom, reproduced and
//! measured).
//!
//!   cargo run --release --bin stress
//!
//! Each scenario flies the scripted player at a fixed speed for a fixed
//! duration after world entry completes, then stops and measures how long
//! streaming takes to fully re-settle (`entry_complete`), plus frame-time
//! percentiles during flight and settle and the peak streaming-queue depths.
//! Report-only: the numbers are the before/after gauge for every streaming
//! phase; pin thresholds here once a few runs establish the machine baseline.
//!
//! 2026-07-19 pre-fix baseline (12-core box, 3 workers, 60 Hz pace):
//!   r6_64mps    settle 0.10s  flight p50 0.43 / p95  7.98 / p99 14.75 ms
//!   r6_200mps   settle 0.21s  flight p50 4.20 / p95 12.04 / p99 13.08 ms
//!   r20_200mps  settle 0.32s  flight p50 12.47 / p95 18.25 / p99 20.02 ms
//!                             (peaks: upload_queue 45, light_apply 1793)

use project_watt_cubed::harness::{StressSpec, run_stress};

fn main() {
    // Two speeds at the default radius plus one heavy-world variant, all paced
    // to 60 Hz so streaming budgets fire at a real session's rate. All well
    // below the streamer's 512 m/s teleport threshold: this exercises travel.
    let specs = [
        StressSpec {
            name: "r6_64mps",
            speed_mps: 64.0,
            secs: 10.0,
            radius: 6,
            pace_hz: 60.0,
        },
        StressSpec {
            name: "r6_200mps",
            speed_mps: 200.0,
            secs: 10.0,
            radius: 6,
            pace_hz: 60.0,
        },
        StressSpec {
            name: "r20_200mps",
            speed_mps: 200.0,
            secs: 10.0,
            radius: 20,
            pace_hz: 60.0,
        },
    ];

    let reports = run_stress(&specs);

    println!();
    println!("stress: {} scenario(s)", reports.len());
    for (name, o) in &reports {
        let settle = match o.settle_time {
            Some(t) => format!("{:.2}s", t.as_secs_f64()),
            None => "NEVER (capped)".to_string(),
        };
        println!("== {name} ==");
        println!("  settle after stop: {settle}");
        println!(
            "  flight frames: n={} p50={:.2}ms p95={:.2}ms p99={:.2}ms max={:.2}ms",
            o.flight.frames, o.flight.p50, o.flight.p95, o.flight.p99, o.flight.max
        );
        println!(
            "  settle frames: n={} p50={:.2}ms p95={:.2}ms p99={:.2}ms max={:.2}ms",
            o.settle.frames, o.settle.p50, o.settle.p95, o.settle.p99, o.settle.max
        );
        println!(
            "  peaks: upload_queue={} light_apply={} mesh_worklist={} worker_near={} worker_far={} chunks={}",
            o.max_upload_queue,
            o.max_light_apply,
            o.max_mesh_worklist,
            o.max_worker_near_queue,
            o.max_worker_far_queue,
            o.max_chunks
        );
        println!(
            "  adaptive floor: effort={:.2} active_workers={}",
            o.min_stream_effort, o.min_active_workers
        );
        println!(
            "  at stop: light_worklist={} chunks={} seeds={} seeds/chunk={:.2}",
            o.light_worklist_at_stop,
            o.chunks_at_stop,
            o.light_seed_inserts_at_stop,
            o.seeds_per_chunk
        );
        println!(
            "  settle light admit/s={:.0}",
            o.settle_light_admit_per_s
        );
        for s in &o.settle_samples {
            println!(
                "  t={}s admit/s={:.0} workers={} effort={:.2} light_worklist={}",
                s.sec, s.light_admit_per_s, s.active_workers, s.effort, s.light_worklist
            );
        }
        if !o.stuck.is_empty() {
            println!("  STUCK: {}", o.stuck);
        }
    }
}
