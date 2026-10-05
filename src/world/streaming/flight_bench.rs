//! Headless fast flight over the round start world: an ignored main-thread timing bench.
//! Each frame mirrors [`World::stream`] (and the pump inside it) with the GPU stood in:
//! uploads land as fake handles, frees go to the test log, occlusion masks touch no engine.
//! Lane budgets match `lanes.rs`; frames are paced like an uncapped game whose render
//! thread keeps up (`FLIGHT_HZ`).
//!
//! `cargo test --release --lib round_flight_breakdown -- --ignored --nocapture`
//! Env: `FLIGHT_SPEEDS` (m/s, default `100,600`), `FLIGHT_SECS` (default 20),
//! `FLIGHT_SETTLE` (rest seconds before flight, default 60), `FLIGHT_VIEW` (`h,v`,
//! default `6,3`), `FLIGHT_LOD2` (`0` turns the far field off), `FLIGHT_HZ` (default 240).

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
        w.lod_clip_grow.set();
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
    let (center, full_pass) = w.begin_stream(eye, None);
    laps.in_full = full_pass;
    laps.lap(0);
    if w.anything_in_flight() {
        fake_drain(w);
    }
    assert!(!w.dirty_pending(), "the flight edits nothing");
    w.refresh_lod_clip();
    laps.lap(1);
    if full_pass {
        w.unload_far_with(center, |state, _| state.free_logged());
        laps.lap(2);
        w.cross_boundary(center);
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
        w.update_lod_face(center);
        w.poll_mip();
        w.ensure_mip_bake();
        w.refresh_section_overlay(Budget::Millis(1.0));
        laps.lap(7);
        w.refresh_frontier(center);
        laps.lap(8);
        if full_pass {
            w.unload_sections_with(center, |_| {});
            w.pending_sections.set();
        }
        laps.lap(9);
        w.reclaim_blocked_sections(center, None);
        laps.lap(10);
        admit::<SectionLane>(w, center, Budget::Millis(1.0));
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

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
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
    let t0 = Instant::now();
    loop {
        let flown = t0.elapsed().as_secs_f64();
        if flown >= secs {
            break;
        }
        let start = Instant::now();
        let key = w.section_frontier_key;
        let occ = w.last_occlusion_rebuild;
        let full = frame(&mut w, spawn + DVec3::X * (speed * flown), &mut laps);
        let took = start.elapsed().as_secs_f32() * 1e3;
        ms.push(took);
        if full {
            cross_ms.push(took);
        }
        frontiers += u32::from(w.section_frontier_key != key);
        occlusions += u32::from(w.last_occlusion_rebuild != occ);
        pace(&mut w, start);
    }
    let frames = ms.len();
    let flown = t0.elapsed().as_secs_f64();
    let stats = |v: &mut Vec<f32>| -> (f32, f32, f32, f32) {
        if v.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }
        v.sort_by(f32::total_cmp);
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let at = |q: f64| v[((v.len() - 1) as f64 * q) as usize];
        (mean, at(0.5), at(0.95), v[v.len() - 1])
    };
    let total: f32 = ms.iter().sum();
    let (mean, p50, p95, max) = stats(&mut ms);
    let (cmean, _, cp95, cmax) = stats(&mut cross_ms);
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
    let not_needs_mesh = w.mesh_worklist.iter().filter(|c| !w.is_needs_mesh(**c)).count();
    let unloaded = w.mesh_worklist.iter().filter(|c| !w.chunks.contains_key(c)).count();
    println!(
        "  end: chunks={} mesh_worklist={} (not NeedsMesh {not_needs_mesh}, unloaded {unloaded}) \
         light_worklist={} upload_queue={} sections={} desired={} generating={}",
        w.chunks.len(),
        w.mesh_worklist.len(),
        w.light_worklist.len(),
        w.upload_queue.len(),
        w.sections.len(),
        w.section_desired.len(),
        w.generating.len(),
    );
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
