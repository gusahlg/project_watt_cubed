//! Headless fast flight over the round start world: an ignored main-thread timing bench.
//! Each frame mirrors [`World::stream`] (and the pump inside it) with the GPU stood in:
//! uploads land as fake handles, frees go to the test log, occlusion masks touch no engine.
//! Lane budgets match `lanes.rs`; frames are paced like an uncapped game whose render
//! thread keeps up (`FLIGHT_HZ`).
//!
//! `cargo test --release --lib round_flight_breakdown -- --ignored --nocapture`
//! Env: `FLIGHT_SPEEDS` (m/s, default `100,600`), `FLIGHT_SECS` (default 20),
//! `FLIGHT_SETTLE` (rest seconds before flight, default 60), `FLIGHT_VIEW` (`h,v`,
//! default `6,3`), `FLIGHT_LOD2` (`0` turns the far field off), `FLIGHT_HZ` (default 240),
//! `FLIGHT_STOP` (seconds held still after the flight, default 0), `FLIGHT_ASSERT`
//! (`1` checks the RD16/V5 acceptance numbers).

use std::time::{Duration, Instant};

use voxel_engine::{MeshHandle, Pass};

use super::*;
use crate::render_config::RenderConfig;
use crate::world::generation::WorldgenKind;
use crate::world::{adjust_count, admit, mesh_free_log};

const PHASES: [&str; 14] = [
    "begin",
    "drain",
    "unload",
    "cross",
    "generate",
    "light",
    "mesh",
    "lod_face",
    "frontier",
    "sec_unload",
    "reclaim",
    "sec_admit",
    "sec_visible",
    "occlusion",
];

struct Laps {
    at: Instant,
    sum: [Duration; PHASES.len()],
    /// The same, over full-pass frames only.
    full: [Duration; PHASES.len()],
    in_full: bool,
}

impl Laps {
    fn lap(&mut self, phase: usize) {
        let now = Instant::now();
        self.sum[phase] += now - self.at;
        if self.in_full {
            self.full[phase] += now - self.at;
        }
        self.at = now;
    }
}

/// [`World::install_chunk_handles`] without an engine: one fake opaque handle, or `Air`.
fn fake_install(w: &mut World, coord: Coord, drawn: bool) {
    let vis = !w.occlusion_active || w.occlusion.is_visible(coord);
    let handles = ByPass::from_fn(|p| (drawn && p == Pass::Opaque).then(|| MeshHandle::from_raw_parts(1, 1)));
    if let Some(loaded) = w.chunks.get_mut(&coord) {
        let was = loaded.state.is_building();
        loaded.retire_logged(MeshState::from_upload(handles));
        loaded.mesh_hash = None;
        adjust_count(&mut w.building_meshes, was, false);
        loaded.visible = vis;
    }
}

/// [`World::drain_results`] without an engine, at the drain lane's 1 ms. The two upload loops
/// are copies of its own; the engine-free blocks are shared.
fn fake_drain(w: &mut World) {
    w.integrate_results(Duration::from_millis(1));
    let pacer = w.stream_pacer;
    // The chunk-upload loop, installing through `fake_install` for `upload_chunk_payload`.
    let budget = pacer.upload_bytes();
    let (mut bytes, mut uploads, mut pops) = (0usize, 0usize, 0usize);
    while (uploads == 0 || bytes < budget) && pops < UPLOAD_SCAN_MAX {
        let Some((coord, rev, data)) = w.upload_queue.pop_front() else {
            break;
        };
        pops += 1;
        if !w.mesh_result_applies(coord, rev) {
            w.drop_stale_upload(coord);
            continue;
        }
        let b = mesh_output_bytes(&data);
        bytes += b;
        uploads += 1;
        fake_install(w, coord, b > 0);
        w.note_settled();
    }
    w.apply_light_queue();
    // The section-upload loop: a section lands `Ready` with no slabs.
    let mut sections = 0;
    while sections < pacer.section_uploads() && bytes < budget {
        let Some((pos, token, b, _)) = w.section_upload_queue.pop_front() else {
            break;
        };
        sections += 1;
        if let Some(state @ SectionState::Meshing { .. }) = w.sections.get_mut(&pos)
            && matches!(state, SectionState::Meshing { token: t } if *t == token)
        {
            adjust_count(&mut w.meshing_sections, true, false);
            bytes += b;
            *state = SectionState::Ready { meshes: Vec::new(), cages: Vec::new(), last_style: None };
            w.pending_sections.set();
            w.section_cover_dirty.set();
        }
    }
}

/// One `stream` frame at `eye`. Returns whether it was a full pass.
fn frame(w: &mut World, eye: DVec3, laps: &mut Laps) -> bool {
    laps.at = Instant::now();
    let (center, far, full_pass, far_moved) = w.begin_stream(eye, None);
    laps.in_full = full_pass;
    laps.lap(0);
    if w.anything_in_flight() {
        fake_drain(w);
    }
    assert!(!w.dirty_pending(), "the flight edits nothing");
    w.refresh_lod_clip();
    laps.lap(1);
    let load_follow = !full_pass && (w.load_moved || w.heading_changed);
    if full_pass {
        w.unload_far_with(center, |state, _| state.free_logged());
        laps.lap(2);
        w.cross_boundary(center);
        w.finish_load_window(center, true);
        laps.lap(3);
    } else if load_follow {
        laps.lap(2);
        w.finish_load_window(center, false);
        laps.lap(3);
    }
    w.request_region_data(center, Budget::Millis(2.0));
    laps.lap(4);
    if w.lighting {
        admit::<LightLane>(w, center, Budget::Millis(1.0));
    }
    laps.lap(5);
    w.tick_light_gate();
    if !w.upload_backlogged() {
        admit::<MeshLane>(w, center, Budget::Millis(2.0));
    }
    w.flush_degraded_terminal();
    laps.lap(6);
    if w.lod2 {
        w.section_pyramid.unit = w.view.lod_unit();
        w.update_lod_face(far);
        w.poll_mip();
        w.ensure_mip_bake();
        w.refresh_section_overlay(Budget::Millis(1.0));
        laps.lap(7);
        w.refresh_frontier(far);
        laps.lap(8);
        if full_pass || far_moved {
            w.unload_sections_with(far, |_| {});
            w.pending_sections.set();
        }
        laps.lap(9);
        w.reclaim_blocked_sections(far, None);
        laps.lap(10);
        admit::<SectionLane>(w, far, Budget::Millis(1.0));
        laps.lap(11);
        if w.section_cover_dirty.take() || w.pending_sections.get() {
            w.rebuild_section_visible(None);
        }
        laps.lap(12);
    }
    w.rebuild_occlusion(None, Budget::Millis(0.5));
    w.refresh_lod_clip();
    laps.lap(13);
    full_pass
}

/// One untimed `stream` frame at `eye`, for the streaming convergence tests, the engine's slot
/// report stood in.
pub(super) fn step(w: &mut World, eye: DVec3) -> bool {
    w.gpu_live_slots = w.local_mesh_slots() as u32;
    mesh_free_log::take();
    let zero = [Duration::ZERO; PHASES.len()];
    let full = frame(w, eye, &mut Laps { at: Instant::now(), sum: zero, full: zero, in_full: false });
    mesh_free_log::take();
    full
}

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

/// One per-second look at the near field: the loading window's Ready share and the full draw
/// box's, with the loading radii.
struct Sample {
    t: f64,
    load: (usize, usize),
    full: (usize, usize),
    lh: i32,
    lv: i32,
}

impl Sample {
    fn take(w: &World, t: f64) -> Self {
        Self { t, load: load_window_ready(w), full: full_window_ready(w), lh: w.load_h, lv: w.load_v }
    }

    fn print(&self, tag: &str) {
        let pct = |(ready, want): (usize, usize)| if want == 0 { 100.0 } else { ready as f64 * 100.0 / want as f64 };
        println!(
            "  {tag}t={:.0}s load window {}/{} ({:.1}%) full window {}/{} ({:.1}%) load=({},{})",
            self.t,
            self.load.0,
            self.load.1,
            pct(self.load),
            self.full.0,
            self.full.1,
            pct(self.full),
            self.lh,
            self.lv,
        );
    }
}

/// Mean, p50, p95 and max of frame times (sorts `v`).
fn frame_stats(v: &mut [f32]) -> (f32, f32, f32, f32) {
    if v.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    v.sort_by(f32::total_cmp);
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let at = |q: f64| v[((v.len() - 1) as f64 * q) as usize];
    (mean, at(0.5), at(0.95), v[v.len() - 1])
}

/// Settle at rest, then fly +X at `speed` for `secs`, frames paced to `hz`.
fn fly(speed: f64, secs: f64, settle: f64, view: (i32, i32), lod2: bool, hz: f64) {
    let mut render = RenderConfig::default();
    render.lod2 = lod2;
    let mut w = World::with_kind(42, render, WorldgenKind::Diffusion, false);
    w.set_view_distances(view.0, view.1);
    let spawn = w.chart_spawn().expect("the start world is charted");
    w.prepare_around(spawn);
    w.drive_spawn_ready();
    let period = Duration::from_secs_f64(1.0 / hz);
    let zero = [Duration::ZERO; PHASES.len()];
    let mut laps = Laps { at: Instant::now(), sum: zero, full: zero, in_full: false };
    let pace = |w: &mut World, start: Instant| {
        // A real engine reports its live slots each frame; sections here carry none.
        w.gpu_live_slots = w.local_mesh_slots() as u32;
        mesh_free_log::take();
        if let Some(rest) = period.checked_sub(start.elapsed()) {
            std::thread::sleep(rest);
        }
    };
    let t0 = Instant::now();
    while t0.elapsed().as_secs_f64() < settle && !w.entry_complete() {
        let start = Instant::now();
        frame(&mut w, spawn, &mut laps);
        pace(&mut w, start);
    }
    println!(
        "flight {speed} m/s: settled={} after {:.1}s view={view:?} lod2={lod2} hz={hz} chunks={} sections={}",
        w.entry_complete(),
        t0.elapsed().as_secs_f64(),
        w.chunks.len(),
        w.sections.len()
    );
    laps.sum = zero;
    laps.full = zero;
    let mut ms: Vec<f32> = Vec::new();
    let mut cross_ms: Vec<f32> = Vec::new();
    let (mut frontiers, mut occlusions) = (0u32, 0u32);
    let jobs_done0 = w.counters.jobs_completed;
    let jobs_cancel0 = w.counters.jobs_cancelled;
    let gen0 = w.counters.gen_landed;
    let behind0 = w.counters.gen_landed_behind;
    let discarded0 = w.counters.gen_discarded;
    let rebuilds0 = w.counters.gen_cursor_rebuilds;
    let (mut near_sum, mut far_sum) = (0u64, 0u64);
    let (mut near_max, mut far_max) = (0usize, 0usize);
    let (mut workers_min, mut workers_max) = (usize::MAX, 0usize);
    let (mut mesh_max, mut light_max, mut gen_max) = (0usize, 0usize, 0usize);
    let mut effort_min = f32::MAX;
    let mut next_sample = 1.0f64;
    let mut window_samples: Vec<Sample> = Vec::new();
    let mut last_eye = spawn;
    let t0 = Instant::now();
    loop {
        let flown = t0.elapsed().as_secs_f64();
        if flown >= secs {
            break;
        }
        let start = Instant::now();
        let key = w.section_frontier_key;
        let occ = w.last_occlusion_rebuild;
        last_eye = spawn + DVec3::X * (speed * flown);
        let full = frame(&mut w, last_eye, &mut laps);
        let took = start.elapsed().as_secs_f32() * 1e3;
        ms.push(took);
        if full {
            cross_ms.push(took);
        }
        frontiers += u32::from(w.section_frontier_key != key);
        occlusions += u32::from(w.last_occlusion_rebuild != occ);
        let pool = w.worker_pool();
        let (nq, fq) = pool.queue_depths();
        let active = pool.active_workers();
        near_sum += nq as u64;
        far_sum += fq as u64;
        near_max = near_max.max(nq);
        far_max = far_max.max(fq);
        workers_min = workers_min.min(active);
        workers_max = workers_max.max(active);
        mesh_max = mesh_max.max(w.mesh_worklist.len());
        light_max = light_max.max(w.light_worklist.len());
        gen_max = gen_max.max(w.generating.len());
        effort_min = effort_min.min(w.stream_pacer.effort());
        if flown >= next_sample {
            window_samples.push(Sample::take(&w, flown));
            let [absent, generating, building, waiting] = load_window_misses(&w);
            println!(
                "  t={flown:.0}s miss absent={absent} generating={generating} building={building} waiting={waiting} \
                 | discarded so far={}",
                w.counters.gen_discarded - discarded0,
            );
            next_sample += 1.0;
        }
        pace(&mut w, start);
    }
    // Counters and the end window are the flight, not the standstill afterwards.
    let flown = t0.elapsed().as_secs_f64();
    let end = Sample::take(&w, flown);
    let end_chunks = w.chunks.len();
    let end_mesh = w.mesh_worklist.len();
    let end_not_mesh = w.mesh_worklist.iter().filter(|c| !w.is_needs_mesh(**c)).count();
    let end_unloaded = w.mesh_worklist.iter().filter(|c| !w.chunks.contains_key(c)).count();
    let end_light = w.light_worklist.len();
    let end_upload = w.upload_queue.len();
    let end_sections = w.sections.len();
    let end_desired = w.section_desired.len();
    let end_generating = w.generating.len();
    let jobs_done = w.counters.jobs_completed - jobs_done0;
    let jobs_cancel = w.counters.jobs_cancelled - jobs_cancel0;
    let generated = w.counters.gen_landed - gen0;
    let behind = w.counters.gen_landed_behind - behind0;
    let discarded = w.counters.gen_discarded - discarded0;
    let rebuilds = w.counters.gen_cursor_rebuilds - rebuilds0;
    let mut stop = hold_still(&mut w, last_eye, &pace);
    let frames = ms.len();
    let total: f32 = ms.iter().sum();
    let (mean, p50, p95, max) = frame_stats(&mut ms);
    let (cmean, _, cp95, cmax) = frame_stats(&mut cross_ms);
    println!(
        "flight {speed} m/s: frames={frames} ({:.0}/s) main ms/frame mean={mean:.3} p50={p50:.3} \
         p95={p95:.3} max={max:.2} busy={:.1}% | full passes={} mean={cmean:.3} p95={cp95:.3} \
         max={cmax:.2} | frontier recomputes={frontiers} occlusion rebuilds={occlusions}",
        frames as f64 / flown,
        f64::from(total) / (flown * 10.0),
        cross_ms.len(),
    );
    let mut line = String::from("  ms/frame by phase:");
    for (name, d) in PHASES.iter().zip(laps.sum) {
        line.push_str(&format!(" {name}={:.3}", d.as_secs_f64() * 1e3 / frames.max(1) as f64));
    }
    println!("{line}");
    let mut line = String::from("  ms/full pass by phase:");
    for (name, d) in PHASES.iter().zip(laps.full) {
        line.push_str(&format!(" {name}={:.3}", d.as_secs_f64() * 1e3 / cross_ms.len().max(1) as f64));
    }
    println!("{line}");
    println!(
        "  end: chunks={end_chunks} mesh_worklist={end_mesh} (not NeedsMesh {end_not_mesh}, unloaded {end_unloaded}) \
         light_worklist={end_light} upload_queue={end_upload} sections={end_sections} desired={end_desired} generating={end_generating}",
    );
    let n = frames.max(1) as f64;
    println!(
        "  queues: near mean={:.1} max={near_max} far mean={:.1} max={far_max} | \
         workers {workers_min}..={workers_max} effort_min={effort_min:.3} | \
         peaks mesh={mesh_max} light={light_max} generating={gen_max} | \
         jobs completed={jobs_done} cancelled={jobs_cancel} | generated={generated} \
         wasted={} (landed_behind={behind} discarded={discarded}) | gen cursor rebuilds={rebuilds}",
        near_sum as f64 / n,
        far_sum as f64 / n,
        behind + discarded,
    );
    for sample in &window_samples {
        sample.print("");
    }
    end.print("end ");
    if !stop.ms.is_empty() {
        let stop_frames = stop.ms.len();
        let (_, gp50, gp95, gmax) = frame_stats(&mut stop.ms[..stop.regrow_frames]);
        let (smean, sp50, sp95, smax) = frame_stats(&mut stop.ms);
        let full_at = stop
            .samples
            .iter()
            .find(|s| s.full.1 > 0 && s.full.0 * 100 >= s.full.1 * 99)
            .map_or_else(|| "never".to_string(), |s| format!("{:.0}s", s.t));
        println!(
            "  stop: frames={stop_frames} main ms/frame mean={smean:.3} p50={sp50:.3} p95={sp95:.3} max={smax:.2} | \
             regrow frames={} p50={gp50:.3} p95={gp95:.3} max={gmax:.2} gen cursor rebuilds={} | \
             full window 99% Ready at {full_at}",
            stop.regrow_frames,
            stop.regrow_rebuilds,
        );
    }
    for sample in &stop.samples {
        sample.print("stop ");
    }
    if env_or("FLIGHT_ASSERT", 0) == 1 && view == (16, 5) {
        assert_flight(speed, p95, workers_min, mesh_max, light_max, &window_samples, &stop.samples);
    }
}

/// The standstill after a flight: per-second samples, every frame's main-thread time, how
/// many of those frames ran before the loading window was whole again, and the
/// generation-cursor rebuilds in them.
struct Stop {
    samples: Vec<Sample>,
    ms: Vec<f32>,
    regrow_frames: usize,
    regrow_rebuilds: u64,
}

/// Hold `eye` still for `FLIGHT_STOP` seconds so the loading window can grow back to the
/// full view.
fn hold_still(w: &mut World, eye: DVec3, pace: &impl Fn(&mut World, Instant)) -> Stop {
    let secs = env_or("FLIGHT_STOP", 0.0);
    let mut stop = Stop { samples: Vec::new(), ms: Vec::new(), regrow_frames: 0, regrow_rebuilds: 0 };
    if secs <= 0.0 {
        return stop;
    }
    let zero = [Duration::ZERO; PHASES.len()];
    let mut laps = Laps { at: Instant::now(), sum: zero, full: zero, in_full: false };
    let rebuilds0 = w.counters.gen_cursor_rebuilds;
    let mut regrowing = true;
    let mut next = 1.0f64;
    let t0 = Instant::now();
    loop {
        let held = t0.elapsed().as_secs_f64();
        if held >= secs {
            break;
        }
        let start = Instant::now();
        frame(w, eye, &mut laps);
        stop.ms.push(start.elapsed().as_secs_f32() * 1e3);
        if regrowing && w.loading_full() {
            regrowing = false;
            stop.regrow_frames = stop.ms.len();
            stop.regrow_rebuilds = w.counters.gen_cursor_rebuilds - rebuilds0;
        }
        if held >= next {
            stop.samples.push(Sample::take(w, held));
            next += 1.0;
        }
        pace(w, start);
    }
    if regrowing {
        stop.regrow_frames = stop.ms.len();
        stop.regrow_rebuilds = w.counters.gen_cursor_rebuilds - rebuilds0;
    }
    stop.samples.push(Sample::take(w, t0.elapsed().as_secs_f64()));
    stop
}

/// Acceptance for the owner's view: the reduced window stays ready at 100 m/s,
/// fast flight stays under ~2 ms p95, and stopping brings the full window back.
fn assert_flight(
    speed: f64,
    p95: f32,
    workers_min: usize,
    mesh_max: usize,
    light_max: usize,
    window: &[Sample],
    stop: &[Sample],
) {
    assert!(workers_min > 2, "{speed} m/s parked workers at {workers_min}");
    assert!(mesh_max < 8_000 && light_max < 8_000, "{speed} m/s worklists mesh={mesh_max} light={light_max}");
    if speed >= 600.0 {
        assert!(p95 < 2.25, "{speed} m/s p95 {p95:.3} ms");
    }
    let share = |(ready, want): (usize, usize)| ready as f64 / want.max(1) as f64;
    if (speed - 100.0).abs() < 1.0 {
        for s in window.iter().filter(|s| s.t >= 3.0) {
            assert!(share(s.load) > 0.90, "t={:.0}s loading window {:?}", s.t, s.load);
        }
    }
    if let Some(s) = stop.last() {
        assert!(s.lh >= 16 && s.lv >= 5, "stopping left the window at ({},{})", s.lh, s.lv);
        assert!(share(s.full) > 0.90, "after stopping the full window is {:?}", s.full);
    }
}

/// Chunks the loading window is willing to admit, and how many of those are drawn
/// (`Air` or `Ready`). Behind the direction of travel does not count.
fn load_window_ready(w: &World) -> (usize, usize) {
    let Some(center) = w.center else { return (0, 0) };
    let mut want = 0usize;
    let mut ready = 0usize;
    for coord in w.view_coords(w.load_mesh_box(center)) {
        if !w.admits_mesh(coord) {
            continue;
        }
        want += 1;
        ready += usize::from(drawn(w, coord));
    }
    (ready, want)
}

/// The whole draw box (the full view radius, behind the player included) and how much of it
/// is drawn.
fn full_window_ready(w: &World) -> (usize, usize) {
    let Some(center) = w.center else { return (0, 0) };
    let (mut ready, mut want) = (0usize, 0usize);
    for coord in w.view_coords(w.mesh_box(center)) {
        want += 1;
        ready += usize::from(drawn(w, coord));
    }
    (ready, want)
}

fn drawn(w: &World, coord: Coord) -> bool {
    w.chunks
        .get(&coord)
        .is_some_and(|loaded| matches!(loaded.state, MeshState::Air | MeshState::Ready(_)))
}

/// Unready chunks inside the loading window: not stored, claimed by generation,
/// mesh in flight, or stored and still waiting (light, neighbours, dirty).
fn load_window_misses(w: &World) -> [usize; 4] {
    let Some(center) = w.center else { return [0; 4] };
    let mut miss = [0usize; 4];
    for coord in w.view_coords(w.load_mesh_box(center)) {
        if !w.admits_mesh(coord) {
            continue;
        }
        match w.chunks.get(&coord).map(|loaded| &loaded.state) {
            Some(MeshState::Air | MeshState::Ready(_)) => {}
            None if w.generating.contains(&coord) => miss[1] += 1,
            None => miss[0] += 1,
            Some(MeshState::NeedsMesh { building: true, .. }) => miss[2] += 1,
            Some(_) => miss[3] += 1,
        }
    }
    miss
}

#[test]
#[ignore]
fn round_flight_breakdown() {
    let speeds: Vec<f64> = std::env::var("FLIGHT_SPEEDS")
        .unwrap_or_else(|_| "100,600".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let secs = env_or("FLIGHT_SECS", 20.0);
    let settle = env_or("FLIGHT_SETTLE", 60.0);
    let view = std::env::var("FLIGHT_VIEW")
        .ok()
        .and_then(|v| {
            let (h, v) = v.split_once(',')?;
            Some((h.trim().parse().ok()?, v.trim().parse().ok()?))
        })
        .unwrap_or((6, 3));
    let lod2 = env_or("FLIGHT_LOD2", 1) != 0;
    let hz = env_or("FLIGHT_HZ", 240.0);
    for speed in speeds {
        fly(speed, secs, settle, view, lod2, hz);
    }
}
