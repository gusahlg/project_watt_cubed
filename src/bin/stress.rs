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
//!
//! 2026-10-08: frame p50/p95/p99 and the drop_stale p95 are the bench's nearest
//! rank, the value at rank ⌈n·q⌉; they were the one at index round((n−1)·q).
//! The two can differ by one sample at small n. The remesh p95s keep the old
//! index.

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

    let reports = run_stress(&specs, &project_watt_cubed::modding::GameBuild::vanilla());

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
        let (p, stop, end, r) = (&o.peaks, &o.at_stop, &o.at_end, &o.remesh);
        println!(
            "  peaks: upload_queue={} light_apply={} mesh_worklist={} worker_near={} worker_far={} chunks={} section_upload_bytes={}",
            p.max_upload_queue,
            p.max_light_apply_queue,
            p.max_mesh_worklist,
            p.max_worker_near_queue,
            p.max_worker_far_queue,
            p.max_chunks,
            p.max_section_upload_bytes
        );
        println!("  adaptive floor: effort={:.2}", p.min_effort);
        println!(
            "  at stop: light_worklist={} chunks={} seeds={} seeds/chunk={:.2}",
            stop.light_worklist,
            stop.chunks,
            stop.light_seed_inserts,
            o.seeds_per_chunk()
        );
        let s = stop.light_seed_split;
        println!(
            "  seeds: store={} border={} edit={} degrade={} terminal={} remesh={}",
            s.store, s.border, s.edit, s.degrade, s.terminal, s.remesh
        );
        println!("  settle light admit/s={:.0}", o.settle_light_admit_per_s());
        println!(
            "  remesh_async/coord between uploads: mean={:.2} p95={:.2} n={}",
            r.remesh_between_upload_mean, r.remesh_between_upload_p95, r.remesh_between_upload_n
        );
        println!(
            "  mesh jobs/chunk before light fixpoint: mean={:.2} p95={:.2} n={}",
            r.mesh_jobs_before_fixpoint_mean, r.mesh_jobs_before_fixpoint_p95, r.mesh_jobs_before_fixpoint_n
        );
        println!(
            "  drop_stale/frame (flight): mean={:.2} p95={:.2}  totals: remesh_async={} drop_stale={}",
            o.drop_stale_per_frame_mean,
            o.drop_stale_per_frame_p95,
            end.remesh_async_calls,
            end.drop_stale_uploads
        );
        let share = |part: u64, whole: u64| {
            if whole == 0 { 0.0 } else { 100.0 * part as f64 / whole as f64 }
        };
        println!(
            "  mesh staging: staged={} fallback={} ring_full={} ({:.2}%)",
            end.mesh_staged,
            end.mesh_fallback,
            end.mesh_ring_full,
            share(end.mesh_ring_full, end.mesh_staged + end.mesh_fallback)
        );
        println!(
            "  section staging: staged={} fallback={} ring_full={} ({:.2}%)",
            end.section_staged,
            end.section_fallback,
            end.section_ring_full,
            share(end.section_ring_full, end.section_staged + end.section_fallback)
        );
        println!("  peak drain upload bytes/frame: {}", p.max_drain_upload_bytes);
        for s in &o.settle_samples {
            println!(
                "  t={}s admit/s={:.0} effort={:.2} light_worklist={}",
                s.sec, s.light_admit_per_s, s.effort, s.light_worklist
            );
        }
        if !o.stuck.is_empty() {
            println!("  STUCK: {}", o.stuck);
        }
    }
}
