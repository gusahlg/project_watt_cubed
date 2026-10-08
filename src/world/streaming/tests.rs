//! Streaming tests.

use super::super::StreamLane;
use super::*;
use crate::coord::ChunkCoord;
use crate::world::{light, mesh};
use super::far::{inside_xz, storage_eye_block};
use super::light::LIGHT_WAIT_DEGRADE;

#[test]
fn new_chunks_follow_the_loading_radius_and_not_the_trail() {
    use crate::render_config::RenderConfig;
    let mut world = World::with_config_lazy(1, RenderConfig::default());
    world.set_view_distances(16, 5);
    world.begin_stream(DVec3::new(8.0, 64.0, 8.0), None);
    let center = world.center.expect("stream publishes a centre");
    let edge = Coord::new(center.x + 16, center.y, center.z);
    let above = Coord::new(center.x, center.y + 5, center.z);
    assert!(world.will_accept_chunk(edge), "rest loads the view edge");
    assert!(world.will_accept_chunk(above), "rest loads the vertical edge");

    world.stream_pacer.update(DVec3::new(600.0, 0.0, 0.0), 0.0);
    world.apply_loading_radius();
    let lh = world.load_h;
    assert!((1..=3).contains(&lh), "600 m/s keeps a few chunks, got {lh}");
    assert!(!world.will_accept_chunk(edge), "the far edge is not admitted");
    assert!(!world.will_accept_chunk(above), "the vertical edge is not admitted");
    assert!(
        !world.will_accept_chunk(Coord::new(center.x - 1, center.y, center.z)),
        "nothing behind the player"
    );
    assert!(world.will_accept_chunk(Coord::new(center.x + lh, center.y, center.z)));
    let outside = Coord::new(center.x + lh + 2, center.y, center.z);
    assert!(!world.will_accept_chunk(outside), "past the data shell is not admitted");
    assert!(
        world.view_contains(world.mesh_box(center), outside),
        "the draw radius is still the full view"
    );
    assert!(world.view_contains(world.unload_box(center), outside));
    world.ensure_data(outside);
    assert!(world.chunks.contains_key(&outside));
    world.unload_far_with(center, |state, _| {
        state.free_logged();
    });
    assert!(
        world.chunks.contains_key(&outside),
        "a loaded chunk outside the loading radius stays drawn"
    );
}

/// A reduced window lights its data box, so a mesh-box edge chunk waits for
/// its data-shell neighbour. Only a neighbour behind the player, which the
/// window will not light, counts as settled. The full window always waits.
#[test]
fn loading_window_waits_on_its_data_shell_but_not_on_the_trail() {
    let mut world = World::generate();
    let cy = world.generator.height(0, 0).div_euclid(CHUNK_SIZE as i32);
    let center = Coord::new(0, cy, 0);
    let ahead = world.neighbour(center, Face::PosX);
    let behind = world.neighbour(center, Face::NegX);
    assert!(world.chunks.contains_key(&ahead) && world.chunks.contains_key(&behind));
    for coord in Face::ALL.map(|f| world.neighbour(center, f)).into_iter().chain([center]) {
        if let Some(loaded) = world.chunks.get_mut(&coord) {
            loaded.light = Some(light::LightGrid::dark());
        }
    }
    world.light_worklist.clear();
    world.light_inflight.clear();
    world.center = Some(center);
    world.load_h = 0;
    world.load_v = 0;
    world.load_heading = 1;
    world.stream_pacer.update(DVec3::new(100.0, 0.0, 0.0), 0.0);
    world.chunks.get_mut(&behind).unwrap().light = None;
    assert!(!world.admits_light(behind));
    assert!(world.light_ready(center), "an unlit neighbour behind the player does not hold the surface");
    world.chunks.get_mut(&ahead).unwrap().light = None;
    assert!(world.admits_light(ahead), "the window lights its data shell");
    assert!(!world.light_ready(center), "the edge waits for that shell's light");
    world.chunks.get_mut(&ahead).unwrap().light = Some(light::LightGrid::dark());
    world.load_h = world.view.horizontal;
    world.load_v = world.view.vertical;
    world.load_heading = 0;
    assert!(world.loading_full());
    assert!(!world.light_ready(center), "the full window waits for every neighbour");
}

/// A light seed the reduced window drops is owed, not lost. An already-lit
/// chunk's border re-settle, dropped by a prune, by the lane's eviction, or
/// as a cancelled claim, comes back once the window covers it again.
#[test]
fn dropped_light_seed_survives_prune_and_regrow() {
    let mut world = World::generate();
    let cy = world.generator.height(0, 0).div_euclid(CHUNK_SIZE as i32);
    let center = Coord::new(0, cy, 0);
    let far = Coord::new(3, cy, 0);
    assert!(world.chunks.contains_key(&far));
    world.chunks.get_mut(&far).unwrap().light = Some(light::LightGrid::dark());
    world.light_worklist.clear();
    world.light_inflight.clear();
    world.center = Some(center);
    let shrink = |world: &mut World| {
        world.load_h = 0;
        world.load_v = 0;
        assert!(!world.admits_light(far), "the reduced window does not cover it");
    };
    let regrow = |world: &mut World| {
        world.load_h = world.view.horizontal;
        world.load_v = world.view.vertical;
        world.seed_load_window(center);
        assert!(world.light_worklist.contains(&far), "the regrown window re-seeds it");
        assert!(world.light_owed.is_empty());
    };

    world.seed_light(far, super::super::LightSeed::Border);
    shrink(&mut world);
    world.prune_admission_worklists();
    assert!(!world.light_worklist.contains(&far), "the prune drops the seed");
    assert!(world.light_owed.contains(&far), "and owes the settle");
    regrow(&mut world);

    shrink(&mut world);
    world.light_pending.set();
    super::super::admit::<LightLane>(&mut world, center, Budget::Millis(1.0));
    assert!(world.light_owed.contains(&far), "the lane's eviction owes it");
    regrow(&mut world);

    shrink(&mut world);
    world.light_worklist.remove(&far);
    world.light_inflight.insert(far);
    world.cancel_job(pipeline::JobKey::Light { coord: far });
    assert!(world.light_owed.contains(&far), "a cancelled claim outside the window is owed");
    regrow(&mut world);
}

/// A flight along a diagonal keeps one heading: samples alternating either
/// side of 45 degrees do not flip it, so the generation cursor is not
/// rebuilt every pass. A clear turn still switches.
#[test]
fn near_diagonal_flight_keeps_its_heading() {
    use crate::render_config::RenderConfig;
    let mut world = World::with_config_lazy(1, RenderConfig::default());
    world.set_view_distances(8, 3);
    world.begin_stream(DVec3::new(8.0, 64.0, 8.0), None);
    let center = world.center.expect("stream publishes a centre");
    let (a, b) = (DVec3::new(100.0, 0.0, 96.0), DVec3::new(96.0, 0.0, 100.0));
    let pass = |world: &mut World, vel: DVec3| {
        world.stream_pacer.update(vel, 1.0 / 60.0);
        world.apply_loading_radius();
        world.pending_gen.set();
        world.request_region_data(center, Budget::Millis(0.0));
    };
    pass(&mut world, a);
    let heading = world.load_heading;
    assert_ne!(heading, 0, "a reduced window aims");
    let rebuilds = world.counters.gen_cursor_rebuilds;
    for i in 0..20 {
        pass(&mut world, if i % 2 == 0 { b } else { a });
        assert_eq!(world.load_heading, heading, "pass {i} flipped the heading");
    }
    assert_eq!(world.counters.gen_cursor_rebuilds, rebuilds, "the cursor holds across the diagonal");
    pass(&mut world, DVec3::new(0.0, 0.0, 140.0));
    assert_ne!(world.load_heading, heading, "a clear turn switches");
}

/// Whether `coord` is drawn: a final mesh or nothing to draw.
fn drawn(world: &World, coord: Coord) -> bool {
    world
        .chunks
        .get(&coord)
        .is_some_and(|l| matches!(l.state, MeshState::Air | MeshState::Ready(_)))
}

/// The round start world at view `(h, v)`, far field off, spawn slab landed.
fn spawned_round_world(h: i32, v: i32) -> (World, DVec3) {
    use crate::render_config::RenderConfig;
    use crate::world::generation::WorldgenKind;
    let render = RenderConfig { lod2: false, ..RenderConfig::default() };
    let mut world = World::with_kind(42, render, WorldgenKind::Diffusion, false);
    world.set_view_distances(h, v);
    let spawn = world.chart_spawn().expect("the start world is charted");
    world.prepare_around(spawn);
    world.drive_spawn_ready();
    (world, spawn)
}

/// Every loaded chunk of the data box has a settled grid.
fn assert_data_box_lit(world: &World) {
    let center = world.center.expect("streamed");
    for coord in world.view_coords(world.data_box(center)) {
        if let Some(loaded) = world.chunks.get(&coord) {
            assert!(loaded.light.is_some(), "{coord:?} in the data box has no light");
        }
    }
}

/// At rest the loading window is the whole view and light covers the data
/// box: every loaded chunk there settles, so no mesh-box edge chunk is
/// promoted on dark planes for a neighbour that never lights.
#[test]
fn rest_lights_the_data_box_and_promotes_nothing_on_dark_planes() {
    let (mut world, spawn) = spawned_round_world(4, 2);
    let deadline = Instant::now() + Duration::from_secs(60);
    while !world.entry_complete() {
        assert!(Instant::now() < deadline, "rest did not converge: {}", world.entry_debug());
        super::flight_bench::step(&mut world, spawn);
        assert!(world.light_terminal.is_empty(), "a chunk was promoted on dark planes");
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(world.loading_full());
    assert_data_box_lit(&world);
    assert!(world.light_owed.is_empty());
}

/// Flying fast, then holding just above walking speed until the window is
/// whole again, then stopping: the whole draw box ends drawn, including the
/// trail the flight skipped, and no light settle stays owed.
#[test]
fn flight_then_stop_fills_the_whole_view() {
    let (mut world, spawn) = spawned_round_world(6, 3);
    let mut eye = spawn;
    let fly = |world: &mut World, eye: &mut DVec3, speed: f64, secs: f64| {
        let start = Instant::now();
        let mut last = start;
        while start.elapsed().as_secs_f64() < secs {
            let now = Instant::now();
            *eye += DVec3::X * (speed * (now - last).as_secs_f64());
            last = now;
            super::flight_bench::step(world, *eye);
            std::thread::sleep(Duration::from_millis(8));
        }
    };
    fly(&mut world, &mut eye, 100.0, 2.0);
    assert!(world.load_h < world.view.horizontal, "100 m/s reduces the window");
    fly(&mut world, &mut eye, 25.0, 3.5);
    assert!(world.loading_full(), "25 m/s grows the window back to the view");
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        super::flight_bench::step(&mut world, eye);
        let center = world.center.expect("streamed");
        let whole = world.view_coords(world.mesh_box(center)).all(|c| drawn(&world, c));
        if whole && world.light_owed.is_empty() && world.entry_complete() {
            break;
        }
        assert!(Instant::now() < deadline, "the view did not fill after stopping: {}", world.entry_debug());
        std::thread::sleep(Duration::from_millis(4));
    }
    assert_data_box_lit(&world);
}

/// A degraded drawn chunk whose neighbourhood becomes light-ready WITHOUT
/// a border event (nothing re-seeds it) is promoted by the gate's relit
/// sweep — through the ASYNC rebuild path: old mesh carried and drawing,
/// no sync `Dirty` involvement.
#[test]
fn relit_degraded_chunk_promotes_through_the_async_path() {
    let mut world = World::generate();
    let c = ChunkCoord::new(0, 0, 0);
    world.center = Some(c);
    for n in std::iter::once(c).chain(Face::ALL.iter().map(|&f| c.step(f))) {
        world.chunks.get_mut(&n).expect("pregenerated").light = Some(light::LightGrid::dark());
    }
    let h = voxel_engine::MeshHandle::from_raw_parts(21, 1);
    let meshes = super::super::ChunkMeshes::from_upload_handles(ByPass::from_fn(|p| {
        (p == voxel_engine::Pass::Opaque).then_some(h)
    }))
    .expect("one pass present");
    world.chunks.get_mut(&c).unwrap().state = MeshState::Ready(meshes);
    world.mark_degraded(c, true);
    world.light_worklist.clear();
    world.light_inflight.clear();
    world.light_apply_queue.clear();
    world.pending_dirty.take();

    world.tick_light_gate();

    let state = &world.chunks[&c].state;
    assert!(
        matches!(
            state,
            MeshState::NeedsMesh {
                building: false,
                prev: Some(_)
            }
        ),
        "promoted through the async rebuild: {state:?}"
    );
    assert!(world.mesh_worklist.contains(&c), "seeded for the rebuild");
    assert!(
        !world.pending_dirty.get(),
        "the sync dirty path is not involved"
    );
}

/// The admit loop evicts a light-blocked seed from `mesh_worklist` and
/// starts its degrade timer at that EVENT (`MeshLane::on_blocked`); expiry
/// raises no event of its own, so `tick_light_gate`'s sweep over the timed
/// map — NOT worklist membership — is what re-seeds the chunk once
/// `light_wait_expired` makes it mesh-ready. This pins both halves: an
/// un-expired evicted chunk is NOT re-seeded by a tick (its re-seed must
/// come from a real event), and an expired one always is.
#[test]
fn light_wait_expiry_reseeds_an_evicted_chunk_without_stranding_it() {
    let mut world = World::generate();
    let c = ChunkCoord::new(0, 0, 0);
    world.center = Some(c);

    // C and its 6 face neighbours must have data (generate() pregenerates
    // near spawn); C's own light stays unset so `light_ready(c)` is false
    // and `chunk_light_blocked(c)` holds without touching the light worklist.
    for n in std::iter::once(c).chain(crate::coord::Face::ALL.iter().map(|&f| c.step(f))) {
        world
            .chunks
            .get_mut(&n)
            .expect("neighbourhood pregenerated near spawn");
    }
    world.chunks.get_mut(&c).unwrap().light = None;
    world.chunks.get_mut(&c).unwrap().state = MeshState::needs_mesh();
    assert!(world.neighbours_have_data(c));
    assert!(world.in_mesh_box(c));
    assert!(
        world.chunk_light_blocked(c),
        "no published light: c is light-blocked"
    );

    // Seed C and run the REAL admission pass: it must evict the blocked
    // seed and start its wait timer through the `on_blocked` event.
    world.mesh_worklist.insert(c);
    world.pending_fresh.set();
    super::super::admit::<MeshLane>(
        &mut world,
        c,
        voxel_engine::producer::Budget::Millis(5.0),
    );
    assert!(
        !world.mesh_worklist.contains(&c),
        "the blocked seed is evicted"
    );
    assert!(
        world.light_gate.blocked_since.contains_key(&c),
        "eviction must start the wait timer"
    );
    assert!(!world.light_wait_expired(c), "not yet past the wait budget");

    // Before expiry, a tick must NOT re-seed it — pre-expiry re-seeds come
    // from real events, never from the per-pass sweep.
    world.tick_light_gate();
    assert!(
        !world.mesh_worklist.contains(&c),
        "an un-expired evicted chunk is not re-seeded by the sweep"
    );

    // Back-date the timer past LIGHT_WAIT_DEGRADE without sleeping — the
    // gate's expiry is wall-clock, so this is the only deterministic way to
    // reach the expired state.
    world.light_gate.blocked_since.insert(
        c,
        Instant::now() - LIGHT_WAIT_DEGRADE - Duration::from_millis(1),
    );
    assert!(
        world.light_wait_expired(c),
        "back-dated timer must read as expired"
    );

    // The next `tick_light_gate` must re-seed it purely from the expired
    // timer, with no dependency on `c` already being in the worklist.
    world.tick_light_gate();
    assert!(
        world.mesh_worklist.contains(&c),
        "eviction-stall regression: an expired-but-evicted chunk must be re-seeded"
    );
    assert!(
        world.pending_fresh.get(),
        "re-armed: the fresh scan will pick it up"
    );
    assert!(
        <MeshLane as StreamLane>::ready(&world, c),
        "expired wait admits a degraded mesh even though light never settled"
    );
}

/// A degraded `Ready` chunk whose neighbour light is permanently missing
/// is, at quiescence, rebuilt asynchronously: the drawn mesh is carried,
/// the terminal set records that missing planes are settled dark, and
/// `MeshLane::submit` snapshots non-degraded. Claim (not submit) drops
/// the degraded and terminal marks.
#[test]
fn degraded_ready_chunk_promotes_through_terminal_async_path() {
    let mut world = World::generate();
    let c = ChunkCoord::new(0, 0, 0);
    world.center = Some(c);
    let missing = c.step(Face::PosX);
    for n in std::iter::once(c).chain(Face::ALL.iter().map(|&f| c.step(f))) {
        let loaded = world.chunks.get_mut(&n).expect("pregenerated");
        loaded.light = if n == missing {
            None
        } else {
            Some(light::LightGrid::dark())
        };
    }
    let h = voxel_engine::MeshHandle::from_raw_parts(21, 1);
    let meshes = super::super::ChunkMeshes::from_upload_handles(ByPass::from_fn(|p| {
        (p == voxel_engine::Pass::Opaque).then_some(h)
    }))
    .expect("one pass present");
    world.chunks.get_mut(&c).unwrap().state = MeshState::Ready(meshes);
    world.mark_degraded(c, true);
    world.generating.clear();
    world.mesh_worklist.clear();
    world.light_worklist.clear();
    world.light_inflight.clear();
    world.light_apply_queue.clear();
    world.pending_dirty.take();
    assert!(!world.light_ready(c), "one neighbour grid is permanently missing");

    world.flush_degraded_terminal();

    let state = &world.chunks[&c].state;
    assert!(
        matches!(
            state,
            MeshState::NeedsMesh {
                building: false,
                prev: Some(_)
            }
        ),
        "promoted through the async rebuild: {state:?}"
    );
    assert!(world.mesh_worklist.contains(&c), "seeded for the rebuild");
    assert!(
        world.light_terminal.contains(&c),
        "missing neighbour light is terminal"
    );
    assert!(
        world.light_gate.degraded.contains(&c),
        "degraded flag stays until the rebuild is claimed"
    );
    assert!(
        <MeshLane as StreamLane>::ready(&world, c),
        "terminal membership admits the rebuild without another light wait"
    );
    assert!(
        !world.pending_dirty.get(),
        "the sync dirty path is not involved"
    );

    let job = <MeshLane as StreamLane>::submit(&mut world, c).expect("terminal mesh job");
    let pipeline::Job::Mesh { snapshot, .. } = job else {
        panic!("expected a mesh job");
    };
    let shell = snapshot.light.expect("lighting on");
    assert_eq!(
        shell.at(CHUNK_SIZE as i32, 8, 8),
        light::Lumel::DARK,
        "terminal snapshot reads the missing +X neighbour as settled dark"
    );
    assert!(
        world.light_gate.degraded.contains(&c),
        "submit must not mutate the degraded set"
    );
    assert!(
        world.light_terminal.contains(&c),
        "submit must not drop the terminal mark (a rejected submit retries)"
    );

    <MeshLane as StreamLane>::claim(&mut world, c);
    assert!(
        !world.light_gate.degraded.contains(&c),
        "claim marks the snapshot non-degraded"
    );
    assert!(
        world.light_terminal.is_empty(),
        "claim consumes the terminal mark"
    );
}

/// One GPU-free stream pass: light-gate, mesh admit (submit + claim +
/// install the carried mesh as Ready — tests have no Engine), terminal
/// flush. Matches the live `stream` order so promotion completes in one
/// quiescence round after the seed.
fn pump_terminal_mesh(world: &mut World) {
    world.tick_light_gate();
    let seeds: Vec<Coord> = world.mesh_worklist.iter().copied().collect();
    for key in seeds {
        if <MeshLane as StreamLane>::in_flight(world, key) {
            continue;
        }
        if !<MeshLane as StreamLane>::ready(world, key) {
            world.mesh_worklist.remove(&key);
            <MeshLane as StreamLane>::on_blocked(world, key);
            continue;
        }
        let _job = <MeshLane as StreamLane>::submit(world, key).expect("mesh job");
        <MeshLane as StreamLane>::claim(world, key);
        if let Some(loaded) = world.chunks.get_mut(&key) {
            if let MeshState::NeedsMesh {
                building: true,
                prev,
            } = &mut loaded.state
            {
                let next = match prev.take() {
                    Some(m) => MeshState::Ready(m),
                    None => MeshState::Air,
                };
                loaded.retire_logged(next);
                super::super::adjust_count(&mut world.building_meshes, true, false);
            }
        }
    }
    world.flush_degraded_terminal();
}

/// A degraded mesh whose face neighbour has unloaded (trailing-edge /
/// load-set-edge) still promotes at quiescence: the terminal mark admits
/// the rebuild without neighbour data, a stranded `NeedsMesh` is re-seeded,
/// and `entry_complete` becomes true with the centre set.
#[test]
fn degraded_chunk_promotes_when_a_neighbour_is_missing() {
    let mut world = World::generate();
    world.lod2 = false;
    world.set_view_distances(2, 2);
    let cy = world.generator.height(0, 0).div_euclid(CHUNK_SIZE as i32);
    let center = ChunkCoord::new(0, cy, 0);
    world.center = Some(center);
    let edge = ChunkCoord::new(2, cy, 0);
    let missing = edge.step(Face::PosX);
    assert!(world.in_mesh_box(edge), "edge chunk is drawn");
    assert!(
        !world.in_mesh_box(missing),
        "the unloaded neighbour sits outside the mesh box"
    );

    for coord in world.mesh_box(center).coords() {
        if !world.chunks.contains_key(&coord) {
            world.ensure_data(coord);
        }
        let loaded = world.chunks.get_mut(&coord).expect("in-box data");
        loaded.state = MeshState::Air;
        if loaded.light.is_none() {
            loaded.light = Some(light::LightGrid::dark());
        }
    }

    let h = voxel_engine::MeshHandle::from_raw_parts(77, 1);
    let meshes = super::super::ChunkMeshes::from_upload_handles(ByPass::from_fn(|p| {
        (p == voxel_engine::Pass::Opaque).then_some(h)
    }))
    .expect("one pass present");
    world.chunks.get_mut(&edge).unwrap().state = MeshState::Ready(meshes);
    world.mark_degraded(edge, true);
    world.forget_chunk(missing);
    world.generating.clear();
    world.mesh_worklist.clear();
    world.light_worklist.clear();
    world.light_inflight.clear();
    world.light_apply_queue.clear();
    world.light_gate.blocked_since.clear();
    world.upload_queue.clear();
    assert!(
        !world.neighbours_have_data(edge),
        "the face neighbour is gone"
    );
    assert!(!world.light_ready(edge));

    world.flush_degraded_terminal();
    assert!(
        matches!(
            world.chunks[&edge].state,
            MeshState::NeedsMesh {
                building: false,
                prev: Some(_)
            }
        ),
        "Ready degraded promotes through the async path"
    );
    assert!(world.light_terminal.contains(&edge));
    assert!(
        <MeshLane as StreamLane>::ready(&world, edge),
        "terminal admits without neighbour data"
    );

    // The live admit loop evicts a blocked seed; a later flush must
    // re-seed the stranded NeedsMesh instead of assuming another path
    // will resolve it.
    world.mesh_worklist.remove(&edge);
    world.pending_fresh.take();
    world.flush_degraded_terminal();
    assert!(
        world.mesh_worklist.contains(&edge),
        "stuck NeedsMesh is re-seeded"
    );
    assert!(world.pending_fresh.get());
    assert!(world.light_terminal.contains(&edge));

    let deadline = Instant::now() + Duration::from_secs(5);
    while !world.entry_complete() {
        assert!(
            Instant::now() < deadline,
            "promotion did not settle: {}",
            world.entry_debug()
        );
        pump_terminal_mesh(&mut world);
    }
    assert!(
        !world.light_gate.degraded.contains(&edge),
        "the rebuild claim cleared the degraded flag"
    );
    assert!(
        matches!(world.chunks[&edge].state, MeshState::Ready(_)),
        "the chunk shows a Ready mesh"
    );
    assert!(world.entry_complete(), "centre is set and the box is final");
    assert!(world.chunks[&edge].state.live_meshes().unwrap().draws(h));

    // A later real neighbour arrival must rebuild the promoted chunk so
    // the dark-plane snapshot is not permanent.
    world.ensure_data(missing);
    assert!(
        matches!(
            world.chunks[&edge].state,
            MeshState::NeedsMesh {
                building: false,
                prev: Some(_)
            }
        ),
        "storing the missing neighbour rebuilds the terminal-promoted chunk"
    );
    assert!(world.mesh_worklist.contains(&edge));
}

/// A degraded chunk that has left the mesh box (still loaded in the unload
/// hysteresis) cannot be admitted, so flush drops the flag instead of
/// re-seeding a seed admit will just evict.
#[test]
fn flush_drops_degraded_outside_the_mesh_box() {
    let mut world = World::generate();
    world.lod2 = false;
    world.set_view_distances(2, 2);
    let cy = world.generator.height(0, 0).div_euclid(CHUNK_SIZE as i32);
    let center = ChunkCoord::new(0, cy, 0);
    world.center = Some(center);
    let outside = ChunkCoord::new(3, cy, 0);
    assert!(!world.in_mesh_box(outside));
    if !world.chunks.contains_key(&outside) {
        world.ensure_data(outside);
    }
    for coord in world.mesh_box(center).coords() {
        if !world.chunks.contains_key(&coord) {
            world.ensure_data(coord);
        }
        world.chunks.get_mut(&coord).unwrap().state = MeshState::Air;
    }
    let h = voxel_engine::MeshHandle::from_raw_parts(78, 1);
    let meshes = super::super::ChunkMeshes::from_upload_handles(ByPass::from_fn(|p| {
        (p == voxel_engine::Pass::Opaque).then_some(h)
    }))
    .expect("one pass present");
    world.chunks.get_mut(&outside).unwrap().state = MeshState::Ready(meshes);
    world.mark_degraded(outside, true);
    world.generating.clear();
    world.mesh_worklist.clear();
    world.light_worklist.clear();
    world.light_inflight.clear();
    world.light_apply_queue.clear();
    world.light_gate.blocked_since.clear();

    world.flush_degraded_terminal();
    assert!(
        !world.light_gate.degraded.contains(&outside),
        "out-of-box degraded is not owed a remesh"
    );
    assert!(
        !world.mesh_worklist.contains(&outside),
        "must not re-seed a seed admit will evict"
    );
    assert!(
        matches!(world.chunks[&outside].state, MeshState::Ready(_)),
        "the drawn mesh is left in place"
    );

    world.chunks.get_mut(&outside).unwrap().state = MeshState::NeedsMesh {
        building: false,
        prev: Some(
            super::super::ChunkMeshes::from_upload_handles(ByPass::from_fn(|p| {
                (p == voxel_engine::Pass::Opaque).then_some(h)
            }))
            .expect("one pass present"),
        ),
    };
    world.mark_degraded(outside, true);
    world.mesh_worklist.clear();
    world.flush_degraded_terminal();
    assert!(!world.light_gate.degraded.contains(&outside));
    assert!(!world.mesh_worklist.contains(&outside));
}

fn ready_handle(id: u32) -> super::super::ChunkMeshes {
    let h = voxel_engine::MeshHandle::from_raw_parts(id, 1);
    super::super::ChunkMeshes::from_upload_handles(ByPass::from_fn(|p| {
        (p == voxel_engine::Pass::Opaque).then_some(h)
    }))
    .expect("one pass present")
}

/// Changed settles mark `light_dirty` and do not remesh while the
/// 27-neighbourhood still has pending light work; one tick after the
/// neighbourhood quiets issues a single async rebuild.
#[test]
fn changed_settle_remeshes_once_at_nhood_fixpoint() {
    let mut world = World::generate();
    let c = Coord::new(0, 0, 0);
    world.center = Some(c);
    world.chunks.get_mut(&c).unwrap().state = MeshState::Ready(ready_handle(91));
    world.chunks.get_mut(&c).unwrap().light = Some(light::LightGrid::open_sky());
    world.light_worklist.clear();
    world.light_inflight.clear();
    world.light_apply_queue.clear();
    world.pending_dirty.take();
    let pending = c.step(Face::PosX);
    world.light_worklist.insert(pending);
    let rev = world.chunks[&c].rev;

    world.settle_light(c, light::LightGrid::dark());
    world.settle_light(c, light::LightGrid::full());
    world.settle_light(c, light::LightGrid::dark());
    assert!(
        matches!(world.chunks[&c].state, MeshState::Ready(_)),
        "pending nhood light must not remesh"
    );
    assert_eq!(world.chunks[&c].rev, rev, "no rev bump while waiting");
    assert!(world.light_gate.dirty.contains_key(&c));
    assert_eq!(world.remesh_stats.remesh_async_calls, 0);

    world.light_worklist.clear();
    world.light_gate.dirty.retain(|&k, _| k == c);
    world.tick_light_gate();
    assert!(
        matches!(
            world.chunks[&c].state,
            MeshState::NeedsMesh {
                building: false,
                prev: Some(_)
            }
        ),
        "quiet nhood promotes one async rebuild"
    );
    assert_eq!(world.remesh_stats.remesh_async_calls, 1);
    assert!(!world.light_gate.dirty.contains_key(&c));
}

/// The degrade timer still promotes a dirty Ready chunk while its
/// neighbourhood has pending light work, so the first update is not
/// delayed past LIGHT_WAIT_DEGRADE.
#[test]
fn dirty_chunk_promotes_when_degrade_timer_expires() {
    let mut world = World::generate();
    let c = Coord::new(0, 0, 0);
    world.center = Some(c);
    world.chunks.get_mut(&c).unwrap().state = MeshState::Ready(ready_handle(92));
    world.chunks.get_mut(&c).unwrap().light = Some(light::LightGrid::open_sky());
    world.light_worklist.clear();
    world.light_inflight.clear();
    world.light_apply_queue.clear();
    world.settle_light(c, light::LightGrid::dark());
    world.light_worklist.insert(c.step(Face::PosY));
    assert!(matches!(world.chunks[&c].state, MeshState::Ready(_)));

    world.light_gate.dirty.insert(
        c,
        Instant::now() - LIGHT_WAIT_DEGRADE - Duration::from_millis(1),
    );
    world.tick_light_gate();
    assert!(
        matches!(
            world.chunks[&c].state,
            MeshState::NeedsMesh {
                building: false,
                prev: Some(_)
            }
        ),
        "expired dirty mark remeshes even with pending nhood light"
    );
}

/// First mesh of a never-drawn chunk is not delayed: settle keeps it on
/// the worklist without a rev-bumping remesh_async.
#[test]
fn first_mesh_is_not_delayed_by_light_dirty() {
    let mut world = World::generate();
    let c = Coord::new(0, 0, 0);
    world.center = Some(c);
    world.chunks.get_mut(&c).unwrap().state = MeshState::needs_mesh();
    world.chunks.get_mut(&c).unwrap().light = Some(light::LightGrid::open_sky());
    world.light_worklist.clear();
    world.light_inflight.clear();
    world.light_apply_queue.clear();
    world.mesh_worklist.clear();
    let rev = world.chunks[&c].rev;
    world.light_worklist.insert(c.step(Face::NegZ));
    world.settle_light(c, light::LightGrid::dark());
    assert_eq!(world.chunks[&c].rev, rev, "first mesh must not take a rev bump");
    assert!(world.mesh_worklist.contains(&c), "still seeded for the first mesh");
    assert!(
        matches!(
            world.chunks[&c].state,
            MeshState::NeedsMesh {
                building: false,
                prev: None
            }
        )
    );
    world.tick_light_gate();
    assert_eq!(world.chunks[&c].rev, rev);
    assert_eq!(world.remesh_stats.remesh_async_calls, 0);
}

/// `flush_degraded_terminal` promotes a per-coord-quiet degraded chunk
/// even while unrelated light work is still queued elsewhere.
#[test]
fn flush_degraded_is_per_coord_not_global() {
    let mut world = World::generate();
    let c = Coord::new(0, 0, 0);
    world.center = Some(c);
    let missing = c.step(Face::PosX);
    for n in std::iter::once(c).chain(Face::ALL.iter().map(|&f| c.step(f))) {
        let loaded = world.chunks.get_mut(&n).expect("pregenerated");
        loaded.light = if n == missing {
            None
        } else {
            Some(light::LightGrid::dark())
        };
    }
    world.chunks.get_mut(&c).unwrap().state = MeshState::Ready(ready_handle(93));
    world.mark_degraded(c, true);
    world.generating.clear();
    world.mesh_worklist.clear();
    world.light_worklist.clear();
    world.light_inflight.clear();
    world.light_apply_queue.clear();
    world.pending_dirty.take();
    let far = Coord::new(8, 0, 8);
    world.light_worklist.insert(far);
    assert!(!world.near_quiescent(), "global light work is still queued");
    assert!(!world.light_ready(c));
    assert!(world.light_nhood_quiet(c), "this coord's 27-nhood is idle");

    world.flush_degraded_terminal();
    assert!(
        matches!(
            world.chunks[&c].state,
            MeshState::NeedsMesh {
                building: false,
                prev: Some(_)
            }
        ),
        "per-coord quiet promotes without waiting for global quiescence"
    );
    assert!(world.light_terminal.contains(&c));
}

/// `LightLane::submit` must not skip a job just because `trivial_light`
/// would succeed — that decision was made at store time.
#[test]
fn light_lane_submit_does_not_recheck_trivial_light() {
    let mut world = World::generate();
    let coord = world
        .chunks
        .iter()
        .find_map(|(&c, l)| l.light.as_ref().map(|_| c))
        .expect("generate publishes at least one grid");
    assert!(
        <LightLane as StreamLane>::submit(&mut world, coord).is_some(),
        "submit must not re-check trivial_light"
    );
}

fn assert_ceilings_eq(got: &light::CeilingWindow, slow: &light::CeilingWindow) {
    for lz in 0..CHUNK_SIZE {
        for lx in 0..CHUNK_SIZE {
            assert_eq!(
                got.surface_at(lx, lz),
                slow.surface_at(lx, lz),
                "ceiling lx={lx} lz={lz}"
            );
        }
    }
}

/// `accept_column` caches the skylight ceiling from worker heights (not
/// `height()`) and `trivial_light` publishes the same grid the slow path
/// would. Covers the flat world, diffusion, and an edited-roof raise.
#[test]
fn accept_column_caches_ceiling_and_trivial_light_matches_slow() {
    use crate::render_config::RenderConfig;
    use crate::world::generation::WorldgenKind;

    for kind in [WorldgenKind::Flat, WorldgenKind::Diffusion] {
        let mut world = World::with_kind(7, RenderConfig::default(), kind, false);
        // Flat: a chunk above the hills. Diffusion: the start world is charted, so the
        // air chunk and the roof sit on its +Y storage column, not the physical origin.
        let (coord, roof_y) = if kind == WorldgenKind::Flat {
            (Coord::new(1, 31, -2), 200)
        } else {
            use crate::space::atlas::Patch;
            let home = world.generator.cosmos().expect("cosmos").home();
            let atlas = world
                .generator
                .atlases()
                .iter()
                .find(|a| (a.centre - home.centre_f()).length() < 1.0)
                .expect("the start world is charted");
            let n = atlas.bands[0].n;
            let s = atlas.storage(Patch::Shell { band: 0, face: Face::PosY }, [n / 2, 0, n / 2]);
            let h = world.generator.height(s[0] as i32 + 8, s[2] as i32 + 8);
            assert_ne!(h, i32::MIN, "chart column has no surface");
            let roof_y = h + 8;
            let cs = CHUNK_SIZE as i32;
            let cx = (s[0] as i32).div_euclid(cs);
            let cz = (s[2] as i32).div_euclid(cs);
            let cy = roof_y.div_euclid(cs) + 2;
            (Coord::new(cx, cy, cz), roof_y)
        };
        world.center = Some(coord);

        let stone = world.registry.id_by_label("rock").expect("builtin Stone");
        // Roof in this column, below the stored chunk: raise before store.
        world.set_block(
            coord.x * CHUNK_SIZE as i32 + 3,
            roof_y,
            coord.z * CHUNK_SIZE as i32 + 5,
            stone,
        );

        let slow = world.capture_ceiling_slow(coord);
        let key = ColumnKey { face: Face::PosY, a: coord.x, b: coord.z };
        assert!(
            !world.ceilings.contains_key(&key),
            "slow helper must not warm the cache"
        );

        let (datas, heights) = world.generator.generate_column(key, coord.y..=coord.y);
        let chunks: Vec<_> = datas
            .into_iter()
            .map(|(alt, data)| {
                let placed = key.chunk(alt);
                (placed, Chunk::from_data(placed.x, placed.y, placed.z, data))
            })
            .collect();
        world.accept_column(key, chunks, Box::new(heights));

        let cached = world
            .ceilings
            .get(&key)
            .expect("accept_column installs the ceiling before store");
        assert_ceilings_eq(cached.as_ref(), &slow);
        assert!(
            cached.surface_at(3, 5) >= roof_y + 1,
            "edited roof must raise the cached ceiling"
        );

        let grid = world.chunks[&coord]
            .light
            .as_ref()
            .expect("uniform-air above the surface publishes trivial light");
        let world_y0 = coord.y * CHUNK_SIZE as i32;
        let all_open = (0..CHUNK_SIZE)
            .all(|lz| (0..CHUNK_SIZE).all(|lx| slow.open_above(lx, lz, world_y0)));
        assert!(all_open, "fixture sits fully above the (raised) ceiling");
        assert!(
            grid == &light::LightGrid::open_sky(),
            "trivial light must be open_sky"
        );
    }
}

#[test]
fn neighbour_blocklight_near_reads_the_settled_flag() {
    let mut world = World::generate();
    let cy = world.generator.height(0, 0).div_euclid(CHUNK_SIZE as i32);
    let c = Coord::new(0, cy, 0);
    let n = c.step(Face::PosX);
    world.chunks.get(&c).expect("generate preloads the origin");
    world
        .chunks
        .get(&n)
        .expect("generate preloads the face neighbour");
    world.settle_light(n, light::LightGrid::dark());
    assert!(!world.chunks[&n].has_blocklight);
    assert!(!world.neighbour_blocklight_near(c));
    world.settle_light(n, light::LightGrid::full());
    assert!(world.chunks[&n].has_blocklight);
    assert!(world.neighbour_blocklight_near(c));
    world.settle_light(n, light::LightGrid::full());
    assert!(
        world.chunks[&n].has_blocklight,
        "identical re-settle keeps the flag"
    );
}

/// First publish of a dark grid matches the missing-neighbour shell, so
/// no face moved and no neighbour is seeded.
#[test]
fn dark_first_publish_seeds_no_neighbour() {
    let mut world = World::generate();
    let c = Coord::new(0, 0, 0);
    world.chunks.get_mut(&c).unwrap().light = None;
    world.chunks.get_mut(&c.step(Face::PosX)).unwrap().state = MeshState::needs_mesh();
    world.light_worklist.clear();
    world.light_inflight.clear();
    world.mesh_worklist.clear();
    world.settle_light(c, light::LightGrid::dark());
    for &face in &Face::ALL {
        assert!(
            !world.light_worklist.contains(&c.step(face)),
            "dark first publish must not seed {face:?}"
        );
    }
    let n = c.step(Face::PosX);
    assert!(
        world.mesh_worklist.contains(&n),
        "first publish still re-seeds a waiting neighbour's mesh"
    );
}

/// A moved face seeds only neighbours that already have data; missing
/// neighbours are not inserted (store_chunk / first-publish constraint).
#[test]
fn settle_seeds_only_neighbours_that_have_data() {
    let mut world = World::generate();
    let c = Coord::new(0, 0, 0);
    let missing = c.step(Face::PosY);
    let present = c.step(Face::PosX);
    world.forget_chunk(missing);
    world.chunks.get_mut(&c).unwrap().light = None;
    assert!(
        world.chunks.contains_key(&present),
        "generate preloads a lateral neighbour"
    );
    world.chunks.get_mut(&present).unwrap().light = Some(light::LightGrid::dark());
    world.light_worklist.clear();
    world.light_inflight.clear();
    world.settle_light(c, light::LightGrid::open_sky());
    assert!(
        !world.light_worklist.contains(&missing),
        "must not seed a neighbour without data"
    );
    assert!(
        world.light_worklist.contains(&present),
        "a loaded neighbour whose shared face moved must be seeded"
    );
}

/// An in-flight neighbour is marked, not re-inserted; the seed lands when
/// its result integrates — even if that grid equals the one it already had.
#[test]
fn inflight_neighbour_reseeds_after_landing() {
    let mut world = World::generate();
    let c = Coord::new(0, 0, 0);
    let n = c.step(Face::PosX);
    world.chunks.get_mut(&c).unwrap().light = Some(light::LightGrid::dark());
    world.chunks.get_mut(&n).unwrap().light = Some(light::LightGrid::dark());
    world.light_worklist.clear();
    world.light_inflight.clear();
    world.light_inflight.insert(n);
    world.settle_light(c, light::LightGrid::open_sky());
    assert!(
        !world.light_worklist.contains(&n),
        "in-flight neighbour must not be re-inserted immediately"
    );
    assert!(world.chunks[&n].light_reseed);
    world.settle_light(n, light::LightGrid::dark());
    assert!(!world.chunks[&n].light_reseed);
    assert!(
        world.light_worklist.contains(&n),
        "re-seed after landing so the wave costs one extra flood"
    );
}

/// Interior-only change: faces match the previous grid, so no neighbour
/// is light-seeded (the `border_changed` cut).
#[test]
fn settle_unchanged_faces_seeds_no_neighbour() {
    let mut world = World::generate();
    let c = Coord::new(0, 0, 0);
    world.chunks.get_mut(&c).unwrap().light = Some(light::LightGrid::dark());
    world.light_worklist.clear();
    world.light_inflight.clear();
    world.mesh_worklist.clear();
    let mut interior = light::LightGrid::dark();
    interior.set(
        Chunk::index(8, 8, 8),
        light::Lumel {
            sky: light::LightLevel::FULL,
            block: light::LightLevel::DARK,
        },
    );
    world.settle_light(c, interior);
    for &face in &Face::ALL {
        assert!(
            !world.light_worklist.contains(&c.step(face)),
            "unchanged face {face:?} must not seed its neighbour"
        );
    }
    assert!(
        world.mesh_worklist.contains(&c),
        "self is still mesh-seeded on a changed grid"
    );
}

/// A neighbour already on the worklist is not counted again: the pending
/// flood reads live neighbour grids at admit.
#[test]
fn settle_does_not_recount_already_queued_neighbour() {
    let mut world = World::generate();
    let c = Coord::new(0, 0, 0);
    let n = c.step(Face::PosX);
    world.chunks.get_mut(&c).unwrap().light = Some(light::LightGrid::dark());
    for &face in &Face::ALL {
        if let Some(loaded) = world.chunks.get_mut(&c.step(face)) {
            loaded.light = Some(light::LightGrid::dark());
        }
    }
    world.light_worklist.clear();
    world.light_inflight.clear();
    world.light_worklist.insert(n);
    world.counters.light_seed_inserts = 0;
    world.counters.light_seed_split = super::super::LightSeedSplit::default();
    world.settle_light(c, light::LightGrid::open_sky());
    assert!(world.light_worklist.contains(&n));
    let other_loaded = Face::ALL
        .iter()
        .filter(|&&f| {
            let n2 = c.step(f);
            n2 != n && world.chunks.contains_key(&n2)
        })
        .count() as u64;
    assert_eq!(
        world.counters.light_seed_split.border, other_loaded,
        "already-queued neighbour is not a counted insert"
    );
}

/// Two open-sky grids: the neighbour is already at the analytic result, so
/// a first-publish face move against dark is a no-op flood.
#[test]
fn open_sky_does_not_reseed_open_sky_neighbour() {
    let mut world = World::generate();
    let c = Coord::new(0, 0, 0);
    let n = c.step(Face::PosX);
    world.chunks.get_mut(&c).unwrap().light = None;
    world.chunks.get_mut(&n).unwrap().light = Some(light::LightGrid::open_sky());
    world.light_worklist.clear();
    world.light_inflight.clear();
    world.settle_light(c, light::LightGrid::open_sky());
    assert!(
        !world.light_worklist.contains(&n),
        "open-sky neighbour cannot change when this chunk publishes open sky"
    );
}

/// `store_chunk` (via `ensure_data`) must not seed neighbours that have
/// no data, even when the stored chunk publishes a non-dark first grid.
#[test]
fn store_chunk_does_not_seed_neighbours_without_data() {
    use crate::render_config::RenderConfig;
    let mut world = World::with_config_lazy(1, RenderConfig::default());
    let c = Coord::new(2, 25, -3);
    world.center = Some(c);
    world.ensure_data(c);
    assert!(world.chunks.contains_key(&c));
    for &face in &Face::ALL {
        let n = c.step(face);
        assert!(
            !world.light_worklist.contains(&n),
            "store must not seed neighbour {face:?} that has no data"
        );
    }
}

#[test]
fn seed_light_counts_each_source() {
    let mut world = World::generate();
    let c = Coord::new(0, 0, 0);
    world.light_worklist.clear();
    world.counters.light_seed_inserts = 0;
    world.counters.light_seed_split = super::super::LightSeedSplit::default();
    world.seed_light(c, super::super::LightSeed::Store);
    world.seed_light(c, super::super::LightSeed::Border);
    world.seed_light(c, super::super::LightSeed::Edit);
    world.seed_light(c, super::super::LightSeed::Degrade);
    world.seed_light(c, super::super::LightSeed::Terminal);
    world.seed_light(c, super::super::LightSeed::Remesh);
    assert_eq!(world.counters.light_seed_inserts, 6);
    let s = world.counters.light_seed_split;
    assert_eq!(
        (s.store, s.border, s.edit, s.degrade, s.terminal, s.remesh),
        (1, 1, 1, 1, 1, 1)
    );
}

#[test]
fn unload_leaving_is_the_old_minus_new_shell() {
    let mut world = World::generate();
    let cy = world.generator.height(0, 0).div_euclid(CHUNK_SIZE as i32);
    let a = Coord::new(0, cy, 0);
    world.center = Some(a);
    world.prev_unload_box = Some(world.unload_box(a));
    let b = Coord::new(2, 0, 0);
    let leaving = world.unload_leaving(world.unload_box(b));
    let old = world.unload_box(a);
    let new = world.unload_box(b);
    for &c in &leaving {
        assert!(old.contains(c) && !new.contains(c), "{c:?} not in old∖new");
    }
    for c in old.coords() {
        if new.contains(c)
            || !world.chunks.contains_key(&c)
            || world.spawn_slab.is_some_and(|s| s.contains(c))
        {
            continue;
        }
        assert!(leaving.contains(&c), "{c:?} loaded in old∖new must leave");
    }
}

/// Far from every seam a trailing chunk unloads with the box, as on any world. Near a seam,
/// settled chunks past the unload box (home or across the seam) wait for a turn-back out to
/// the skirt, also while a section is on screen over them, and leave past it.
#[test]
fn chunks_wait_for_a_turn_back_only_near_a_seam() {
    use super::super::quadtree::QuadrantMask;
    use crate::ident::Detail;
    use crate::render_config::{RenderConfig, lod_for};
    use crate::world::generation::WorldgenKind;

    let (lod_levels, lod_detail) = lod_for(6);
    let render = RenderConfig { lod2: true, occlusion: true, lod_levels, lod_detail, ..RenderConfig::default() };
    let world_at = |dir: DVec3| {
        let mut world = World::with_kind(42, render, WorldgenKind::Diffusion, false);
        world.set_view_distances(6, 3);
        let eye = world.home_eye(dir, 100.0);
        let (home, _, _, _) = world.begin_stream(eye, None);
        assert!(!world.fold.is_identity(), "the eye is on a chart");
        (world, home)
    };
    // Settled and drawing: a fake one-pass mesh.
    let settle = |world: &mut World, c: Coord| {
        world.ensure_data(c);
        let h = voxel_engine::MeshHandle::from_raw_parts(1, 1);
        let meshes = super::super::ChunkMeshes::from_upload_handles(ByPass::from_fn(|p| {
            (p == voxel_engine::Pass::Opaque).then_some(h)
        }))
        .expect("one pass present");
        world.chunks.get_mut(&c).expect("ensure_data stores the chunk").state = MeshState::Ready(meshes);
    };
    let walk = |world: &mut World, from: Coord, to: Coord| {
        world.prev_unload_box = Some(world.unload_box(from));
        world.center = Some(to);
        world.unload_far_with(to, |state, _| drop(state));
    };

    // A face's middle: a straight walk holds nothing past the unload box.
    let (mut world, home) = world_at(DVec3::new(0.0, 1.0, 0.0));
    assert!(!world.near_a_seam(home), "the face middle is far from every seam");
    settle(&mut world, home);
    walk(&mut world, home, Coord::new(home.x, home.y, home.z - 11));
    assert!(!world.chunks.contains_key(&home), "a trailing chunk unloads with the box");
    assert!(world.far_wait.is_empty(), "nothing waits away from a seam");

    // Beside the +Z seam: the home chunk and the chunk just across the seam both wait.
    let (mut world, home) = world_at(DVec3::new(0.0, 1.0, 1.0 - 6.5e-6));
    assert!(world.near_a_seam(home));
    let seat = world.seams.chart_seat(home).expect("a chart seat");
    let edge = Coord::new(home.x, home.y, (seat.hi[2] / 16) as i32 - 1);
    let across = world.seams.across(edge, Face::PosZ).expect("a neighbour across +Z").chunk;
    assert_ne!(world.fold.fold(across), across, "the neighbour chunk folds beyond the seam");
    settle(&mut world, home);
    settle(&mut world, across);
    let folded = world.fold.fold(across);
    let skirt = world.skirt();
    let inland = Coord::new(home.x, home.y, home.z - 11);
    assert!(folded.across(inland, Face::PosY) <= skirt);
    walk(&mut world, home, inland);
    assert!(world.chunks.contains_key(&home), "a home chunk near the seam waits");
    assert!(world.chunks.contains_key(&across), "the chunk across the seam waits");
    assert_eq!(world.far_wait.len(), 2);

    let span = 32;
    let (bx, bz) = (home.x * 16 + 8, home.z * 16 + 8);
    let pos = SectionPos { detail: Detail(0), body: 0, face: Face::PosY, x: bx.div_euclid(span), z: bz.div_euclid(span) };
    world.sections.insert(pos, SectionState::Ready { meshes: Vec::new(), cages: Vec::new(), last_style: None });
    world.section_visible.push((pos, QuadrantMask::ALL));
    world.unload_far_with(inland, |state, _| drop(state));
    assert!(world.chunks.contains_key(&home), "a section on screen does not drop a waiting chunk");

    let gone = Coord::new(home.x, home.y, home.z - skirt - 2);
    world.center = Some(gone);
    world.unload_far_with(gone, |state, _| drop(state));
    assert!(!world.chunks.contains_key(&home), "past the skirt the home chunk unloads");
    assert!(!world.chunks.contains_key(&across), "past the skirt the neighbour chunk unloads");
    assert!(world.far_wait.is_empty());
}

/// The skirt is sideways only: near a seam, a layer the player climbs or digs away from unloads
/// as anywhere else. And with the far field turned off, nothing keeps waiting.
#[test]
fn waiting_chunks_respect_the_layers_and_the_far_field_switch() {
    use crate::render_config::{RenderConfig, lod_for};
    use crate::world::generation::WorldgenKind;

    let (lod_levels, lod_detail) = lod_for(6);
    let render = RenderConfig { lod2: true, occlusion: true, lod_levels, lod_detail, ..RenderConfig::default() };
    let mut world = World::with_kind(42, render, WorldgenKind::Diffusion, false);
    world.set_view_distances(6, 3);
    let eye = world.home_eye(DVec3::new(0.0, 1.0, 1.0 - 6.5e-6), 100.0);
    let (home, _, _, _) = world.begin_stream(eye, None);
    assert!(world.near_a_seam(home));
    // Settled and drawing: a fake one-pass mesh.
    let settle = |world: &mut World, c: Coord| {
        world.ensure_data(c);
        let h = voxel_engine::MeshHandle::from_raw_parts(1, 1);
        let meshes = super::super::ChunkMeshes::from_upload_handles(ByPass::from_fn(|p| {
            (p == voxel_engine::Pass::Opaque).then_some(h)
        }))
        .expect("one pass present");
        world.chunks.get_mut(&c).expect("ensure_data stores the chunk").state = MeshState::Ready(meshes);
    };
    settle(&mut world, home);

    // Dig past the skirt straight down: the drawing chunk left above unloads.
    world.window = Window::default();
    let down = Coord::new(home.x, home.y - world.skirt() - 2, home.z);
    world.prev_unload_box = Some(world.unload_box(home));
    world.center = Some(down);
    world.unload_far_with(down, |state, _| drop(state));
    assert!(!world.chunks.contains_key(&home), "a layer past the skirt vertically unloads near a seam");
    assert!(world.far_wait.is_empty());

    // Air draws nothing, so it never waits, even inside the skirt near a seam.
    world.ensure_data(home);
    world.chunks.get_mut(&home).expect("stored").state = MeshState::Air;
    let inland = Coord::new(home.x, home.y, home.z - 11);
    world.prev_unload_box = Some(world.unload_box(home));
    world.center = Some(inland);
    world.unload_far_with(inland, |state, _| drop(state));
    assert!(!world.chunks.contains_key(&home), "an air chunk does not wait");
    assert!(world.far_wait.is_empty());

    // Walk inland so the home chunk waits, then turn the far field off: it unloads on the next pass.
    settle(&mut world, home);
    let inland = Coord::new(home.x, home.y, home.z - 11);
    world.prev_unload_box = Some(world.unload_box(home));
    world.center = Some(inland);
    world.unload_far_with(inland, |state, _| drop(state));
    assert!(world.chunks.contains_key(&home), "the home chunk waits near the seam");
    world.lod2 = false;
    world.unload_far_with(inland, |state, _| drop(state));
    assert!(!world.chunks.contains_key(&home), "with the far field off a waiting chunk unloads");
    assert!(world.far_wait.is_empty());
}

/// Sync `ensure_data` (headless region, unclaimed boundary-cross centre)
/// also installs from `generate_column` heights, so `trivial_light` never
/// calls `height()`.
#[test]
fn ensure_data_caches_ceiling_matching_slow() {
    use crate::render_config::RenderConfig;

    let mut world = World::with_config_lazy(11, RenderConfig::default());
    let coord = Coord::new(2, 25, 1);
    world.ensure_data(coord);
    let cached = world
        .ceilings
        .get(&ColumnKey { face: Face::PosY, a: coord.x, b: coord.z })
        .expect("ensure_data installs the ceiling before store");
    let slow = world.capture_ceiling_slow(coord);
    assert_ceilings_eq(cached.as_ref(), &slow);
    assert!(
        world.chunks[&coord].light.as_ref() == Some(&light::LightGrid::open_sky()),
        "trivial light must be open_sky"
    );
}

#[test]
fn async_upload_does_not_hash_and_identical_edit_skips_remesh() {
    mesh::reset_content_hash_calls();
    let data = mesh::new_chunk_mesh_data();
    let hash = mesh::content_hash(&data);
    assert_eq!(mesh::content_hash_calls(), 1);

    let mut world = World::generate();
    let c = *world.chunks.keys().next().expect("spawn chunks");
    let h = voxel_engine::MeshHandle::from_raw_parts(91, 1);
    let meshes = super::super::ChunkMeshes::from_upload_handles(ByPass::from_fn(|p| {
        (p == voxel_engine::Pass::Opaque).then_some(h)
    }))
    .expect("one pass present");
    {
        let loaded = world.chunks.get_mut(&c).unwrap();
        loaded.state = MeshState::Dirty {
            prev: Some(meshes),
        };
        loaded.mesh_hash = Some(hash);
        loaded.visible = true;
    }

    mesh::reset_content_hash_calls();
    world.upload_chunk_without_gpu(c, Some(hash));
    assert_eq!(mesh::content_hash_calls(), 0, "upload_chunk never hashes");
    assert!(
        matches!(world.chunks[&c].state, MeshState::Ready(_)),
        "identical edit remesh keeps the resident mesh"
    );
    assert_eq!(world.chunks[&c].mesh_hash, Some(hash));

    mesh::reset_content_hash_calls();
    world.upload_chunk_without_gpu(c, None);
    assert_eq!(mesh::content_hash_calls(), 0, "async drain passes None, no hash");
    assert_eq!(world.chunks[&c].mesh_hash, None);
}

#[test]
fn skipped_remesh_pushes_visibility_when_the_chunk_was_hidden() {
    let mut world = World::generate();
    let c = *world.chunks.keys().next().expect("spawn chunks");
    let h = voxel_engine::MeshHandle::from_raw_parts(92, 1);
    let meshes = super::super::ChunkMeshes::from_upload_handles(ByPass::from_fn(|p| {
        (p == voxel_engine::Pass::Opaque).then_some(h)
    }))
    .expect("one pass present");
    {
        let loaded = world.chunks.get_mut(&c).unwrap();
        loaded.state = MeshState::Dirty {
            prev: Some(meshes),
        };
        loaded.visible = false;
    }
    world.occlusion_active = false;
    super::super::vis_log::take();
    world.keep_resident_mesh(c, None);
    assert_eq!(
        super::super::vis_log::take(),
        vec![(h, true)],
        "a hidden resident mesh must be shown when vis becomes true"
    );
    assert!(world.chunks[&c].visible);
    assert!(matches!(world.chunks[&c].state, MeshState::Ready(_)));

    super::super::vis_log::take();
    world.keep_resident_mesh(c, None);
    assert!(
        super::super::vis_log::take().is_empty(),
        "unchanged vis must not push set_visible"
    );
}

/// Valley or hilltop columns inside the near square, below or above the full-res window,
/// that neither a full-res chunk nor a far section draws. Seed 42, diffusion, the bench
/// camera on the start world's +Z chart: ground level, that eye, and 300 above it.
#[test]
fn far_chart_strip_hole_is_closed() {
    use crate::render_config::RenderConfig;
    use crate::world::generation::WorldgenKind;

    let mut world = World::with_kind(42, RenderConfig::default(), WorldgenKind::Diffusion, false);
    let eye = world.chart_eye(DVec3::new(0.0, -8_640_801.0, 22_107_307.0)).expect("symptom chart eye");
    let ground = world.terrain().surface(Face::PosY, eye.x as i32, eye.z as i32);
    assert_ne!(ground, i32::MIN, "symptom column has no surface");
    // `expect_outside`: the window must miss some column, or a punch of the whole square
    // would still report zero holes.
    let sites = [
        ("ground", DVec3::new(eye.x, ground as f64, eye.z), false),
        ("eye", eye, true),
        ("+300", DVec3::new(eye.x, ground as f64 + 300.0, eye.z), true),
    ];
    for (name, storage, expect_outside) in sites {
        let center = Coord::new(
            (storage.x / 16.0).floor() as i32,
            (storage.y / 16.0).floor() as i32,
            (storage.z / 16.0).floor() as i32,
        );
        world.section_eye_y = storage.y;
        world.adopt_fold(center);
        let seat = world.seams.chart_seat(center).unwrap_or_else(|| panic!("{name}: no chart seat"));
        let near = world.near_block_box(center);
        assert!(
            near.0 >= seat.lo[0] && near.1 <= seat.hi[0] && near.2 >= seat.lo[2] && near.3 <= seat.hi[2],
            "{name}: the near square meets a seam"
        );
        let (y0, y1) = world.near_y_range(center);
        let desired = world.desired_sections(center);
        let mut rects = Vec::new();
        for s in &desired {
            let span = s.span() as i64;
            let (x, z) = (s.min_x() as i64, s.min_z() as i64);
            if inside_xz(*s, seat.lo, seat.hi) {
                rects.push((x, z, x + span, z + span));
            }
        }
        let covered = |x: i64, z: i64| {
            rects.iter().any(|&(x0, z0, x1, z1)| x >= x0 && x < x1 && z >= z0 && z < z1)
        };
        let mut holes = 0i32;
        let mut outside = 0i32;
        let mut x = near.0 + 8;
        while x < near.1 {
            let mut z = near.2 + 8;
            while z < near.3 {
                let surf = world.terrain().surface(Face::PosY, x as i32, z as i32);
                if surf != i32::MIN {
                    let solid = i64::from(surf) - 1;
                    if solid < y0 || solid >= y1 {
                        outside += 1;
                        if !covered(x, z) {
                            holes += 1;
                        }
                    }
                }
                z += 16;
            }
            x += 16;
        }
        assert_eq!(
            holes, 0,
            "{name}: {holes} strip holes of {outside} columns outside the window, desired {}",
            desired.len()
        );
        if expect_outside {
            assert!(outside > 0, "{name}: no column sits outside the full-res window");
        }
    }
}

/// Columns of the far-field disk that draw neither a chart section nor a full-res chunk.
/// Across a seam the neighbour chart is in the disk, and the near window's chunks there are
/// real storage chunks (or a section covers them). A section wholly inside the near square is
/// dropped only when that square's surface sits inside the full-res window. The band just
/// outside the box, out to one coarse-ring span, is part of the same count: a straddler is
/// replaced by its descendants, so that band is drawn.
#[test]
fn far_chart_seam_has_no_hole() {
    use crate::render_config::RenderConfig;
    use crate::space::atlas::Patch;
    use crate::space::chart::{self, Map};
    use crate::world::generation::WorldgenKind;
    use crate::ident::Detail;
    use crate::world::section::{section_span, FINEST_DETAIL};

    let mut world = World::with_kind(42, RenderConfig::default(), WorldgenKind::Diffusion, false);
    let centre = world.generator.cosmos().expect("cosmos").home().centre_f();
    let atlas = world
        .generator
        .atlases()
        .iter()
        .find(|a| (a.centre - centre).length() < 1.0)
        .expect("charted")
        .clone();

    let storage_from_dir = |world: &World, dir: DVec3, above: f64| -> DVec3 {
        let dir = dir.normalize();
        let face = Face::from_dominant(dir);
        let (tu, nn, tv) = chart::basis(face);
        let (xi, eta) = Map::Equiangular.inverse(DVec3::new(dir.dot(tu), dir.dot(nn), dir.dot(tv)));
        let n = atlas.bands[0].n;
        let step = 2.0 / n as f64;
        // An exact seam parameter floors to n, one cell past the box. Stand on the last cell.
        let i = (((xi + 1.0) / step).floor() as i64).clamp(0, n - 1);
        let j = (((eta + 1.0) / step).floor() as i64).clamp(0, n - 1);
        let patch = Patch::Shell { band: 0, face };
        let (origin, _) = atlas.storage_box(patch);
        let stored = atlas.storage(patch, [i, 0, j]);
        let ground = world.terrain().surface(Face::PosY, stored[0] as i32, stored[2] as i32);
        let local_y = ground as f64 - origin[1] as f64;
        let surf = atlas.embed(patch, DVec3::new(i as f64 + 0.5, local_y, j as f64 + 0.5));
        let up = (surf - atlas.centre).normalize();
        let fallback = DVec3::new(stored[0] as f64 + 0.5, ground as f64 + above, stored[2] as f64 + 0.5);
        world.chart_eye(surf + up * above).unwrap_or(fallback)
    };

    let mut sites: Vec<(String, DVec3)> = Vec::new();
    let symptom = world.chart_eye(DVec3::new(0.0, -8_640_801.0, 22_107_307.0)).expect("symptom chart eye");
    sites.push(("symptom".to_string(), symptom));
    let sym_ground = world.terrain().surface(Face::PosY, symptom.x as i32, symptom.z as i32);
    sites.push(("symptom-ground".to_string(), DVec3::new(symptom.x, sym_ground as f64, symptom.z)));
    sites.push(("symptom+300".to_string(), DVec3::new(symptom.x, sym_ground as f64 + 300.0, symptom.z)));
    for above in [0.0_f64, 300.0] {
        let tag = if above == 0.0 { "ground" } else { "+300" };
        sites.push((format!("seam-yz-{tag}"), storage_from_dir(&world, DVec3::new(0.0, 1.0, 1.0), above)));
        sites.push((format!("seam-yx-{tag}"), storage_from_dir(&world, DVec3::new(1.0, 1.0, 0.0), above)));
        sites.push((format!("corner-{tag}"), storage_from_dir(&world, DVec3::new(1.0, 0.985, 0.97), above)));
        sites.push((format!("face-{tag}"), storage_from_dir(&world, DVec3::new(0.0, 1.0, 0.0), above)));
    }
    let seam = storage_from_dir(&world, DVec3::new(0.0, 1.0, 1.0), 0.0);
    let seam_c = Coord::new((seam.x / 16.0).floor() as i32, 0, (seam.z / 16.0).floor() as i32);
    let seam_seat = world.seams.chart_seat(seam_c).expect("seam seat");
    let inset = if (seam.z as i64 - seam_seat.lo[2]).abs() < (seam_seat.hi[2] - seam.z as i64).abs() {
        100 * 16
    } else {
        -100 * 16
    };
    let (ix, iz) = (seam.x, seam.z + inset as f64);
    let ig = world.terrain().surface(Face::PosY, ix as i32, iz as i32);
    sites.push(("inset100-ground".to_string(), DVec3::new(ix, ig as f64, iz)));
    sites.push(("inset100+300".to_string(), DVec3::new(ix, ig as f64 + 300.0, iz)));

    let cs = 16i64;
    // The ring outside the finest one. A punched section of that span leaves a wider overhang
    // than a finest tile when the finest annulus does not reach past the full-res box.
    let sliver = section_span(Detail(FINEST_DETAIL.0 + 1)) as i64;
    let outer = world.section_pyramid.outer_m() as i64;
    for (name, storage) in sites {
        let center = Coord::new(
            (storage.x / 16.0).floor() as i32,
            (storage.y / 16.0).floor() as i32,
            (storage.z / 16.0).floor() as i32,
        );
        world.section_eye_y = storage.y;
        world.adopt_fold(center);
        let seat = world.seams.chart_seat(center).unwrap_or_else(|| panic!("{name}: no chart seat"));
        let desired = world.desired_sections(center);
        let (ex, _, ez) = storage_eye_block(center, storage.y, DVec3::ZERO);
        let across = world.seams.seam_across(seat, [ex, storage.y.round() as i64, ez], outer);
        let near = world.near_block_box(center);
        let v = world.view.vertical as i64;
        let mut rects: Vec<(i64, i64, i64, i64)> = Vec::new();
        for s in &desired {
            let span = s.span() as i64;
            let (x, z) = (s.min_x() as i64, s.min_z() as i64);
            let (x0, z0, x1, z1) = if inside_xz(*s, seat.lo, seat.hi) {
                (x, z, x + span, z + span)
            } else if let Some(m) = across.iter().find(|m| inside_xz(*s, m.seat.lo, m.seat.hi)) {
                let (a, c) = m.home_xz(x, z);
                let (b, d) = m.home_xz(x + span, z + span);
                (a.min(b), c.min(d), a.max(b), c.max(d))
            } else {
                continue;
            };
            rects.push((x0, x1, z0, z1));
        }
        let covered = |x: i64, z: i64| rects.iter().any(|&(x0, x1, z0, z1)| x >= x0 && x < x1 && z >= z0 && z < z1);
        let mut holes = 0i32;
        let mut hole_ex = String::new();
        let mut across_cols = 0i32;
        let mut across_bad = 0i32;
        let mut x = ex - outer;
        let step = 64i64;
        while x <= ex + outer {
            let mut z = ez - outer;
            while z <= ez + outer {
                let (dx, dz) = (x - ex, z - ez);
                if dx * dx + dz * dz > outer * outer {
                    z += step;
                    continue;
                }
                let virt = Coord::new(x.div_euclid(cs) as i32, center.y, z.div_euclid(cs) as i32);
                let real = world.fold.unfold(virt);
                let in_near = x >= near.0 && x < near.1 && z >= near.2 && z < near.3;
                let section_hit = covered(x, z);
                let mut chunk_hit = false;
                if in_near {
                    if let Some(rc) = real {
                        let surf = world.terrain().surface(Face::PosY, rc.x * 16 + 8, rc.z * 16 + 8);
                        if rc != virt {
                            across_cols += 1;
                            if surf == i32::MIN || world.seams.chart_seat(rc).is_none() {
                                across_bad += 1;
                            }
                        }
                        let gy = if surf == i32::MIN { i64::MIN } else { (surf as i64 - 1).div_euclid(cs) };
                        chunk_hit = gy >= center.y as i64 - v && gy <= center.y as i64 + v;
                    }
                }
                if !section_hit && !chunk_hit {
                    let past_home = x < seat.lo[0] || x >= seat.hi[0] || z < seat.lo[2] || z >= seat.hi[2];
                    let ox = if x < near.0 { near.0 - x } else if x >= near.1 { x - (near.1 - 1) } else { 0 };
                    let oz = if z < near.2 { near.2 - z } else if z >= near.3 { z - (near.3 - 1) } else { 0 };
                    holes += 1;
                    if holes <= 6 {
                        hole_ex.push_str(&format!(
                            " at ({x},{z}) past_home {past_home} in_near {in_near} ox {ox} oz {oz} real {real:?};"
                        ));
                    }
                }
                z += step;
            }
            x += step;
        }
        // Chunk centres in the overhang band. Step 64 misses a sliver narrower than the stride.
        let mut fine = 0i32;
        let mut fine_n = 0i32;
        let mut fx = near.0 - sliver + 8;
        while fx < near.1 + sliver {
            let mut fz = near.2 - sliver + 8;
            while fz < near.3 + sliver {
                let ox = if fx < near.0 { near.0 - fx } else if fx >= near.1 { fx - (near.1 - 1) } else { 0 };
                let oz = if fz < near.2 { near.2 - fz } else if fz >= near.3 { fz - (near.3 - 1) } else { 0 };
                let outside = ox.max(oz) > 0 && ox.max(oz) < sliver;
                if outside {
                    fine_n += 1;
                    if !covered(fx, fz) {
                        fine += 1;
                    }
                }
                fz += 16;
            }
            fx += 16;
        }
        let edge = [seat.hi[0] - ex, ex - (seat.lo[0] - 1), seat.hi[2] - ez, ez - (seat.lo[2] - 1)];
        let reaches_seam = edge.iter().any(|&d| d < (near.1 - near.0) / 2);
        assert!(fine_n > 0, "{name}: the overhang band was not sampled");
        assert_eq!(
            fine, 0,
            "{name}: {fine} overhang columns of {fine_n} within one coarse span, desired {} holes {holes}{hole_ex}",
            desired.len()
        );
        assert_eq!(
            holes, 0,
            "{name}: {holes} uncovered columns, desired {} near {near:?} eye ({ex},{ez}) center {center:?} edges {edge:?};{hole_ex}",
            desired.len()
        );
        assert!(
            desired.len() <= world.sections_allowed(),
            "{name}: {} sections over the slot budget {}",
            desired.len(),
            world.sections_allowed()
        );
        if reaches_seam {
            assert!(across_cols > 0, "{name}: the near window does not cross the seam");
            assert_eq!(across_bad, 0, "{name}: {across_bad} near columns across the seam are not real chunks");
        }
    }
}

/// A full section floor must not disarm the lane, and must drop Ready sections
/// nothing desired draws so the open cell can be admitted.
/// One edit dirties an overlay position per active detail, each about a far section's extract:
/// a pass past its budget stops after one position, the rest follow on later passes.
#[test]
fn the_edit_overlay_refresh_spreads_over_passes() {
    use crate::world::section::SectionPos;
    let mut world = World::new(7);
    for x in 0..3 {
        let pos = SectionPos { body: 0, face: Face::PosY, detail: crate::world::section::FINEST_DETAIL, x, z: 0 };
        world.section_overlay_dirty.insert(pos);
    }
    let pass = |world: &mut World| match world.refresh_section_overlay(Budget::Millis(0.0)) {
        Progress::Partial { remaining } => Some(remaining),
        Progress::Idle => None,
        _ => panic!("unexpected progress"),
    };
    assert_eq!(pass(&mut world), Some(2));
    assert_eq!(pass(&mut world), Some(1));
    assert_eq!(pass(&mut world), None);
    assert!(world.section_overlay_dirty.is_empty());
    assert_eq!(pass(&mut world), None);
}

#[test]
fn full_section_floor_keeps_the_lane_armed_and_frees_a_slot() {
    let mut world = World::generate();
    let center = Coord::new(0, 4, 0);
    world.center = Some(center);
    world.slot_ceiling = 1024;
    world.gpu_live_slots = 6000;
    let hole = SectionPos {
        body: 0,
        face: Face::PosY,
        detail: super::super::section::FINEST_DETAIL,
        x: 0,
        z: 0,
    };
    let filler = |i: usize| SectionPos {
        body: 0,
        face: Face::PosY,
        detail: super::super::section::FINEST_DETAIL,
        x: 10_000 + i as i32,
        z: -3,
    };
    world.section_desired = vec![hole];
    let empty = || SectionState::Ready { meshes: Vec::new(), cages: Vec::new(), last_style: None };
    for i in 0..super::super::SECTION_SLOT_FLOOR {
        world.sections.insert(filler(i), empty());
    }
    assert!(!world.section_covered(hole), "the hole has no resident cover");
    assert!(!<SectionLane as StreamLane>::ready(&world, hole), "the floor is full");
    world.pending_sections.set();
    super::super::admit::<SectionLane>(&mut world, center, Budget::Millis(8.0));
    assert!(
        world.pending_sections.get(),
        "a refused budget is not a drained backlog"
    );
    assert!(!world.sections.contains_key(&hole), "nothing was admitted");
    world.reclaim_blocked_sections(center, None);
    assert!(
        world.section_budget_used() < world.sections_allowed(),
        "one unwanted Ready section makes room, used {} allowed {}",
        world.section_budget_used(),
        world.sections_allowed()
    );
    assert!(world.pending_sections.get(), "freeing a slot re-arms admission");
    assert!(<SectionLane as StreamLane>::ready(&world, hole));
}

/// A Ready child under an uncovered desired parent fills a floor slot and the parent
/// is still admitted. The child stays: freeing it to make the slot is the bare-ground pop.
#[test]
fn standin_under_a_full_floor_still_admits_its_parent() {
    use super::super::section::Quadrant;

    let mut world = World::generate();
    let center = Coord::new(0, 4, 0);
    world.center = Some(center);
    world.slot_ceiling = 1024;
    world.gpu_live_slots = 6000;
    let parent = SectionPos {
        body: 0,
        face: Face::PosY,
        detail: crate::ident::Detail(super::super::section::FINEST_DETAIL.0 + 1),
        x: 80,
        z: -40,
    };
    let child = parent.child(Quadrant::ALL[0]);
    let filler = |i: usize| SectionPos {
        body: 0,
        face: Face::PosY,
        detail: super::super::section::FINEST_DETAIL,
        x: 10_000 + i as i32,
        z: -3,
    };
    world.section_desired = vec![parent];
    let empty = || SectionState::Ready { meshes: Vec::new(), cages: Vec::new(), last_style: None };
    for i in 0..super::super::SECTION_SLOT_FLOOR - 1 {
        world.sections.insert(filler(i), empty());
    }
    world.sections.insert(child, empty());
    assert_eq!(world.section_budget_used(), world.sections_allowed(), "the floor is full");
    let desired: FastSet<SectionPos> = world.section_desired.iter().copied().collect();
    assert!(world.stands_under_hole(child, &desired), "the child covers the parent's hole");
    assert!(!world.section_covered(parent), "nothing Ready draws the parent");
    world.pending_sections.set();
    super::super::admit::<SectionLane>(&mut world, center, Budget::Millis(8.0));
    assert!(
        matches!(world.sections.get(&parent), Some(SectionState::Meshing { .. })),
        "the parent is admitted while its child holds a floor slot"
    );
    assert!(
        world.sections.get(&child).is_some_and(|s| s.is_ready()),
        "the child keeps drawing until the parent can"
    );
}

/// Only a chunk the mesh lane could admit takes a seed: an unloaded coord, a drawn chunk and
/// an in-flight build stay off the worklist, whichever path seeds (an expired light wait, a
/// failed build), and a stale build re-seeds itself unless an edit made it the dirty lane's.
#[test]
fn mesh_seeds_only_admissible_chunks() {
    let mut world = World::generate();
    let c = Coord::new(0, 0, 0);
    let gone = Coord::new(0, 40, 0);
    assert!(!world.chunks.contains_key(&gone), "the probe coord is not loaded");
    world.mesh_worklist.clear();
    world.seed_mesh(gone);
    assert!(!world.mesh_worklist.contains(&gone), "an unloaded coord seeds itself on load");
    world.fail_job(pipeline::JobKey::Mesh { coord: gone });
    assert!(!world.mesh_worklist.contains(&gone), "a failed build of one seeds nothing");
    world.chunks.get_mut(&c).unwrap().state = MeshState::Ready(ready_handle(5));
    world.seed_mesh(c);
    assert!(!world.mesh_worklist.contains(&c), "a drawn chunk is never admitted");
    let claim = |world: &mut World, prev| {
        world.chunks.get_mut(&c).unwrap().state = MeshState::NeedsMesh { building: true, prev };
        world.building_meshes = 1;
    };
    claim(&mut world, None);
    world.seed_mesh(c);
    assert!(!world.mesh_worklist.contains(&c), "an in-flight build is never admitted");
    world.center = Some(c);
    world.chunks.get_mut(&c).unwrap().light = None;
    assert!(world.chunk_light_blocked(c), "the build waits on light");
    world
        .light_gate
        .blocked_since
        .insert(c, Instant::now() - LIGHT_WAIT_DEGRADE - Duration::from_millis(1));
    world.tick_light_gate();
    assert!(world.light_gate.blocked_since.contains_key(&c), "the wait is still timed");
    assert!(!world.mesh_worklist.contains(&c), "an expired wait seeds no in-flight build");
    world.invalidate_mesh(c);
    world.drop_stale_upload(c);
    world.fail_job(pipeline::JobKey::Mesh { coord: c });
    assert!(!world.mesh_worklist.contains(&c), "an edited build's stale or failed result seeds nothing");
    claim(&mut world, Some(ready_handle(6)));
    world.drop_stale_upload(c);
    assert_eq!(world.building_meshes, 0, "the stale build released its claim");
    assert!(world.mesh_worklist.contains(&c), "a stale build re-seeds");
    world.mesh_worklist.clear();
    world.chunks.get_mut(&c).unwrap().state = MeshState::needs_mesh();
    world.seed_mesh(c);
    assert!(
        world.mesh_worklist.contains(&c) || matches!(world.chunks[&c].state, MeshState::Air),
        "a fresh chunk is seeded unless it is walled in"
    );
}

/// On a chart the frontier is a function of whole blocks and whole-chunk prediction: an eye
/// that moves inside one block, or a velocity that jitters inside one chunk of lookahead,
/// keeps the cached selection, and that selection is what a fresh sweep returns. The surface
/// memo keeps only the rects the last sweep read.
#[test]
fn chart_frontier_holds_within_a_block() {
    use crate::render_config::RenderConfig;
    use crate::world::generation::WorldgenKind;

    let mut world = World::with_kind(42, RenderConfig::default(), WorldgenKind::Diffusion, false);
    let spawn = world.chart_spawn().expect("the start world is charted");
    let eye = world.chart_eye(spawn).expect("spawn stands on a chart");
    let center = Coord::new(
        (eye.x / 16.0).floor() as i32,
        (eye.y / 16.0).floor() as i32,
        (eye.z / 16.0).floor() as i32,
    );
    world.adopt_fold(center);
    world.center = Some(center);
    world.update_lod_face(center);
    assert!(world.section_on_chart(center), "spawn streams from a chart's storage");
    let y = eye.y.floor() + 0.25;
    let refresh = |world: &mut World, eye_y: f64, vel: DVec3| {
        world.section_eye_y = eye_y;
        world.section_vel = vel;
        let key = world.section_frontier_key;
        world.refresh_frontier(center);
        let fresh = world.desired_sections(center);
        // Held sections still wait on chunks no test loads; the rest is the selection.
        let selected: Vec<_> = world.section_desired.iter().copied().filter(|s| !world.section_held.contains_key(s)).collect();
        assert_eq!(selected, fresh, "the cached frontier is the fresh one");
        let memo = &world.near_bounds;
        assert!(memo.rects.values().all(|e| e.1 == memo.pass), "stale rects kept");
        world.section_frontier_key != key
    };
    assert!(refresh(&mut world, y, DVec3::ZERO), "first pass selects");
    assert!(!world.section_desired.is_empty());
    assert!(!refresh(&mut world, y + 0.2, DVec3::ZERO), "same block");
    assert!(refresh(&mut world, y + 1.0, DVec3::ZERO), "next block");
    assert!(refresh(&mut world, y + 1.0, DVec3::new(100.3, 0.0, 0.0)), "prediction starts");
    assert!(!refresh(&mut world, y + 1.0, DVec3::new(100.6, 0.0, -0.4)), "jitter in one chunk");
    assert!(refresh(&mut world, y + 1.0, DVec3::new(120.0, 0.0, 0.0)), "next chunk of lookahead");
}

/// Several km/s keeps the far-field sweep on a fixed grid. One chunk and a
/// few metres of altitude do not rebuild it. The same chunk step below that
/// speed does, and so does a step onto the next grid cell.
#[test]
fn fast_frontier_holds_on_a_fixed_grid() {
    use crate::render_config::RenderConfig;
    use crate::world::generation::WorldgenKind;

    let mut world = World::with_kind(42, RenderConfig::default(), WorldgenKind::Diffusion, false);
    let spawn = world.chart_spawn().expect("the start world is charted");
    let eye = world.chart_eye(spawn).expect("spawn stands on a chart");
    let center = Coord::new(
        (eye.x / 16.0).floor() as i32,
        (eye.y / 16.0).floor() as i32,
        (eye.z / 16.0).floor() as i32,
    );
    world.adopt_fold(center);
    world.center = Some(center);
    world.update_lod_face(center);
    assert!(world.section_on_chart(center), "spawn streams from a chart's storage");
    let inside = if center.x.rem_euclid(FRONTIER_CHUNK_QUANTUM) + 1 < FRONTIER_CHUNK_QUANTUM {
        Coord::new(center.x + 1, center.y, center.z)
    } else {
        Coord::new(center.x - 1, center.y, center.z)
    };
    assert!(world.section_on_chart(inside), "one chunk stays on the chart");

    // Mid-bucket altitude, so a few metres cannot cross the 512 m grid.
    let y = 256.0;
    world.stream_pacer.update(DVec3::new(2000.0, 0.0, 0.0), 0.0);
    assert_eq!(world.frontier_bucket(inside.x), world.frontier_bucket(center.x));
    world.section_eye_y = y;
    world.section_vel = DVec3::ZERO;
    world.refresh_frontier(center);
    let held = world.section_frontier_key;
    let desired = world.section_desired.clone();
    world.section_eye_y = y + 20.0;
    world.refresh_frontier(inside);
    assert_eq!(world.section_frontier_key, held, "inside the grid the key holds");
    assert_eq!(world.section_desired, desired, "the sweep is not redone");

    world.stream_pacer.update(DVec3::new(100.0, 0.0, 0.0), 0.0);
    world.section_eye_y = y;
    world.refresh_frontier(center);
    let slow = world.section_frontier_key;
    world.refresh_frontier(inside);
    assert_ne!(world.section_frontier_key, slow, "below the coarse speed one chunk rebuilds");

    world.stream_pacer.update(DVec3::new(2000.0, 0.0, 0.0), 0.0);
    world.section_eye_y = y;
    world.refresh_frontier(center);
    let again = world.section_frontier_key;
    let jumped = Coord::new(center.x + FRONTIER_CHUNK_QUANTUM, center.y, center.z);
    world.refresh_frontier(jumped);
    assert_ne!(world.section_frontier_key, again, "a chunk-grid step rebuilds the sweep");
    world.section_eye_y = y + FRONTIER_EYE_QUANTUM;
    world.refresh_frontier(center);
    assert_ne!(world.section_frontier_key, again, "an altitude-grid step rebuilds the sweep");
}

fn empty_ready() -> SectionState {
    SectionState::Ready { meshes: Vec::new(), cages: Vec::new(), last_style: None }
}

/// Desired sections with no Ready self or ancestor.
fn uncovered(world: &World) -> usize {
    world.section_desired.iter().filter(|&&c| !world.section_covered(c)).count()
}

/// What the section passes landed: cancelled section jobs, those of them the frontier still
/// wanted, and failed jobs.
#[derive(Default)]
struct Landed {
    cancels: usize,
    wanted: usize,
    fails: usize,
}

/// Land worker results and turn queued section uploads into empty Ready meshes. Returns whether
/// anything landed.
fn pump_sections(world: &mut World, landed: &mut Landed) -> bool {
    let mut got = false;
    while let Some(done) = world.workers.as_ref().and_then(pipeline::Workers::try_recv) {
        got = true;
        match &done {
            pipeline::Done::Cancelled(keys) => {
                landed.cancels += keys.len();
                landed.wanted += keys
                    .iter()
                    .filter(|k| matches!(k, pipeline::JobKey::Section { pos, .. } if world.section_desired.contains(pos)))
                    .count();
            }
            pipeline::Done::Failed(_) => landed.fails += 1,
            _ => {}
        }
        world.integrate_worker_result(done);
    }
    while let Some((pos, token, _, _)) = world.section_upload_queue.pop_front() {
        if let Some(state @ SectionState::Meshing { .. }) = world.sections.get_mut(&pos)
            && matches!(state, SectionState::Meshing { token: t } if *t == token)
        {
            world.meshing_sections = world.meshing_sections.saturating_sub(1);
            *state = empty_ready();
        }
    }
    got
}

/// One pass of the section lanes around the far-field centre, in `stream`'s order: land,
/// reclaim, admit, then the visible rebuild that re-arms holes. Pending is not forced on from
/// outside.
fn section_pass(world: &mut World, landed: &mut Landed) {
    let center = world.section_center().expect("a far-field centre");
    let got = pump_sections(world, landed);
    world.reclaim_blocked_sections(center, None);
    super::super::admit::<SectionLane>(world, center, Budget::Millis(8.0));
    if world.section_cover_dirty.take() || world.pending_sections.get() {
        world.rebuild_section_visible(None);
    }
    if !got {
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Run section passes until every desired section is covered and nothing is in flight.
/// Returns the passes taken.
fn drive_sections(world: &mut World, name: &str, deadline: Instant, landed: &mut Landed) -> usize {
    let mut passes = 0usize;
    loop {
        let unc = uncovered(world);
        let queued = world.workers.as_ref().map(pipeline::Workers::queue_depths).unwrap_or((0, 0)).1;
        if unc == 0 && world.meshing_sections == 0 && world.section_upload_queue.is_empty() && queued == 0 {
            assert_eq!(landed.fails, 0, "{name}: section jobs failed");
            return passes;
        }
        passes += 1;
        assert!(
            Instant::now() < deadline && passes < 20_000,
            "{name}: uncovered {unc} of {} after {passes} passes, cancels {} fails {} meshing {} queued {queued}",
            world.section_desired.len(),
            landed.cancels,
            landed.fails,
            world.meshing_sections
        );
        section_pass(world, landed);
    }
}

/// The chart cap, once the near field has filled the CPU-cull knob, keeps the
/// sections nearest in the chart frame. Sections loaded earlier are the
/// storage-nearest of the wider frontier, which is not that set: neighbours
/// and the far rim disagree. Readiness still reaches an empty uncovered count.
#[test]
fn far_chart_seam_readiness_converges() {
    use crate::render_config::RenderConfig;
    use crate::space::atlas::Patch;
    use crate::space::chart::{self, Map};
    use crate::world::generation::WorldgenKind;

    let mut world = World::with_kind(42, RenderConfig::default(), WorldgenKind::Diffusion, false);
    let centre = world.generator.cosmos().expect("cosmos").home().centre_f();
    let atlas = world
        .generator
        .atlases()
        .iter()
        .find(|a| (a.centre - centre).length() < 1.0)
        .expect("charted")
        .clone();
    let storage_from_dir = |world: &World, dir: DVec3, above: f64| -> DVec3 {
        let dir = dir.normalize();
        let face = Face::from_dominant(dir);
        let (tu, nn, tv) = chart::basis(face);
        let (xi, eta) = Map::Equiangular.inverse(DVec3::new(dir.dot(tu), dir.dot(nn), dir.dot(tv)));
        let n = atlas.bands[0].n;
        let step = 2.0 / n as f64;
        let i = (((xi + 1.0) / step).floor() as i64).clamp(0, n - 1);
        let j = (((eta + 1.0) / step).floor() as i64).clamp(0, n - 1);
        let patch = Patch::Shell { band: 0, face };
        let (origin, _) = atlas.storage_box(patch);
        let stored = atlas.storage(patch, [i, 0, j]);
        let ground = world.terrain().surface(Face::PosY, stored[0] as i32, stored[2] as i32);
        let local_y = ground as f64 - origin[1] as f64;
        let surf = atlas.embed(patch, DVec3::new(i as f64 + 0.5, local_y, j as f64 + 0.5));
        let up = (surf - atlas.centre).normalize();
        let fallback = DVec3::new(stored[0] as f64 + 0.5, ground as f64 + above, stored[2] as f64 + 0.5);
        world.chart_eye(surf + up * above).unwrap_or(fallback)
    };

    let mut sites: Vec<(String, DVec3)> = Vec::new();
    let symptom = world.chart_eye(DVec3::new(0.0, -8_640_801.0, 22_107_307.0)).expect("symptom chart eye");
    sites.push(("symptom".into(), symptom));
    let sym_ground = world.terrain().surface(Face::PosY, symptom.x as i32, symptom.z as i32);
    sites.push(("symptom-ground".into(), DVec3::new(symptom.x, sym_ground as f64, symptom.z)));
    sites.push(("symptom+300".into(), DVec3::new(symptom.x, sym_ground as f64 + 300.0, symptom.z)));
    sites.push(("seam-yz".into(), storage_from_dir(&world, DVec3::new(0.0, 1.0, 1.0), 0.0)));
    sites.push(("seam-yz+300".into(), storage_from_dir(&world, DVec3::new(0.0, 1.0, 1.0), 300.0)));
    sites.push(("seam-yx".into(), storage_from_dir(&world, DVec3::new(1.0, 1.0, 0.0), 0.0)));
    sites.push(("corner".into(), storage_from_dir(&world, DVec3::new(1.0, 0.985, 0.97), 0.0)));
    sites.push(("face".into(), storage_from_dir(&world, DVec3::new(0.0, 1.0, 0.0), 0.0)));

    let deadline = Instant::now() + Duration::from_secs(90);
    for (name, storage) in sites {
        let center = Coord::new(
            (storage.x / 16.0).floor() as i32,
            (storage.y / 16.0).floor() as i32,
            (storage.z / 16.0).floor() as i32,
        );
        world.sections.clear();
        world.section_eye_y = storage.y;
        world.center = Some(center);
        world.stream_up = Some(Face::PosY);
        world.stream_up_set = true;
        world.adopt_fold(center);
        world.slot_ceiling = 1024;
        world.gpu_live_slots = 0;
        let wide = world.desired_sections(center);
        world.gpu_live_slots = 512;
        let tight_allowed = world.sections_allowed();
        let tight = world.desired_sections(center);
        let far_m = f64::from(world.section_pyramid.outer_m());
        let radius = world.view.horizontal;
        let (fold, far_view) = (world.fold, world.far_view(center));
        world.publish_rest_view(center, far_view, radius, far_m, Some(Face::PosY), fold);
        // Cold start: the bench once the near field has already taken the
        // surplus above the section floor, and nothing coarser is resident.
        if name == "symptom" {
            world.section_desired = tight.clone();
            world.pending_sections.set();
            world.section_cover_dirty.set();
            drive_sections(&mut world, "symptom cold", deadline, &mut Landed::default());
            assert_eq!(uncovered(&world), 0, "symptom cold start left sections uncovered");
            world.sections.clear();
            world.meshing_sections = 0;
            world.section_upload_queue.clear();
            while world.workers.as_ref().and_then(pipeline::Workers::try_recv).is_some() {}
        }
        // Admission orders sections in the chart net, so this prefix is the
        // ground beside the eye. Planting it must not leave the capped frontier open.
        let mut prefix = wide;
        prefix.sort_by_key(|s| <SectionLane as StreamLane>::order(&world, center, *s));
        prefix.truncate(tight_allowed.min(prefix.len()));
        for &s in &prefix {
            world.sections.insert(s, empty_ready());
        }
        world.section_desired = tight;
        world.pending_sections.set();
        world.section_cover_dirty.set();
        let planted = uncovered(&world);
        drive_sections(&mut world, &name, deadline, &mut Landed::default());
        assert_eq!(
            uncovered(&world),
            0,
            "{name}: planted {planted} uncovered sections of {} and the floor never cleared",
            world.section_desired.len()
        );
    }
}

/// The far field's thresholds hold while hovering. The chart rings' scale rises with the height
/// over the ground and falls back only well below where it rose; the chart under the eye is kept
/// past the far reach once the far field stands on it; and just below the near window's reach
/// the far field is the frontier it stays just above it.
#[test]
fn far_eye_thresholds_hold() {
    use crate::render_config::RenderConfig;
    use crate::world::generation::WorldgenKind;

    let mut world = World::with_kind(42, RenderConfig::default(), WorldgenKind::Diffusion, false);
    let up = DVec3::new(0.0, 1.0, 0.0);
    let climb = |world: &mut World, above: f64| {
        world.place_eyes(world.home_eye(up, above));
        (world.far_scale, world.far_atlas.is_some())
    };
    // Default ladder: the rings reach 12,288 blocks, so a doubling is taken past 4,096 · 2^s of
    // height and dropped below 4,096 · 2^s / 1.25.
    let path = [
        (0.0, 0, "ground"),
        (4_600.0, 1, "past the first step"),
        (3_700.0, 1, "held below the first step"),
        (2_800.0, 0, "dropped well below it"),
        (9_000.0, 2, "past the second step"),
        (7_200.0, 2, "held below the second step"),
        (6_000.0, 1, "dropped one step"),
        (20_000.0, 3, "at the candidate cap"),
        (200_000.0, 3, "held at the cap"),
    ];
    for (above, scale, what) in path {
        assert_eq!(climb(&mut world, above), (scale, true), "+{above}: {what}");
    }
    // Past the far reach (262,144 above the stored top, ~2,000 over the ground): kept once
    // stood on, left past the hold, and not taken again until back under the reach.
    assert!(climb(&mut world, 280_000.0).1, "held past the far reach");
    assert!(!climb(&mut world, 340_000.0).1, "left past the hold");
    assert!(!climb(&mut world, 280_000.0).1, "not re-taken above the reach");
    assert!(climb(&mut world, 250_000.0).1, "re-taken under the reach");

    // The highest eye the near window still streams on the chart, and the far field there with
    // the near window on the chart and in physical space.
    let mut above = 1_000.0;
    while world.chart_eye(world.home_eye(up, above + 16.0)).is_some() {
        above += 16.0;
    }
    let eye = world.home_eye(up, above);
    let (near, far) = world.place_eyes(eye);
    assert_eq!(near, far, "the near window stands on the chart at +{above}");
    let far_c = eye_chunk(far);
    world.adopt_fold(far_c);
    let on_chart = world.desired_sections(far_c);
    world.adopt_fold(eye_chunk(eye));
    assert!(world.fold.is_identity());
    assert!(!on_chart.is_empty(), "+{above}: no chart sections");
    assert_eq!(world.desired_sections(far_c), on_chart, "+{above}: the far field changes at the near window's reach");
}

/// Stand the eye `above` blocks over the start world in direction `dir`, the way `stream`
/// places it: the near window back in physical space, the far field on the chart under it,
/// and no sections resident.
fn hover(world: &mut World, dir: DVec3, above: f64, name: &str) {
    let (near, far) = world.place_eyes(world.home_eye(dir, above));
    let near_c = eye_chunk(near);
    world.sections.clear();
    world.center = Some(near_c);
    world.set_far_center(eye_chunk(far));
    world.stream_up = world.resolve_stream_up(near_c);
    world.stream_up_set = true;
    world.adopt_fold(near_c);
    assert!(world.fold.is_identity(), "{name}: the near window is not in physical space");
    assert!(!world.far_fold.is_identity(), "{name}: the far field does not stand on a chart");
    publish_far(world);
    assert!(!world.section_desired.is_empty(), "{name}: no chart sections");
}

/// Select the far frontier around the far centre and publish the view to the worker gate, as
/// `stream` does after the far centre moves.
fn publish_far(world: &mut World) {
    let (near_c, far_c) = (world.center.expect("a centre"), world.section_center().expect("a far centre"));
    world.section_desired = world.desired_sections(far_c);
    let (far_m, radius, up) = (world.far_horizon(), world.view.horizontal, world.stream_up);
    let (fold, far_view) = (world.fold, world.far_view(far_c));
    world.publish_rest_view(near_c, far_view, radius, far_m, up, fold);
    world.pending_sections.set();
    world.section_cover_dirty.set();
}

/// From high above the start world (the near window back in physical space, the worker gate
/// measuring far work from the far centre, in the far field's chart net) every desired section
/// lands, also across a seam and at a cube corner: readiness reaches an empty uncovered count in
/// bounded passes, and the gate never deschedules a section the frontier wants.
#[test]
fn far_chart_altitude_readiness_converges() {
    use crate::render_config::RenderConfig;
    use crate::world::generation::WorldgenKind;

    let mut world = World::with_kind(42, RenderConfig::default(), WorldgenKind::Diffusion, false);
    let deadline = Instant::now() + Duration::from_secs(240);
    for (site, dir, heights) in World::FAR_SITES {
        for &above in heights {
            let name = &format!("{site} +{above}");
            hover(&mut world, dir, above, name);
            let mut landed = Landed::default();
            let passes = drive_sections(&mut world, name, deadline, &mut landed);
            println!(
                "{name}: {} sections ready in {passes} passes, {} cancelled",
                world.section_desired.len(),
                landed.cancels
            );
            assert_eq!(uncovered(&world), 0, "{name}: sections left uncovered");
            assert_eq!(landed.wanted, 0, "{name}: the gate descheduled wanted sections");
        }
    }
}

/// The far centre moving while the frontier fills, 50 km over every far site: one chunk every
/// few passes, re-publishing the view each time as `stream` does, and stopping well before the
/// frontier has landed. Every view change re-measures the queued far work, so a horizon short
/// of the coarsest ring's corners, or a neighbour chart measured a face box away, would
/// deschedule wanted sections on every crossing. None is, and once the centre stops every
/// desired section lands.
#[test]
fn far_chart_altitude_readiness_converges_while_moving() {
    use crate::render_config::RenderConfig;
    use crate::world::generation::WorldgenKind;

    const MOVES: usize = 48;
    const PASSES_PER_MOVE: usize = 12;
    let mut world = World::with_kind(42, RenderConfig::default(), WorldgenKind::Diffusion, false);
    let deadline = Instant::now() + Duration::from_secs(240);
    let above = 50_000.0;
    for (site, dir, _) in World::FAR_SITES {
        let name = &format!("{site} +{above} moving");
        hover(&mut world, dir, above, name);
        let mut landed = Landed::default();
        for _ in 0..MOVES {
            let c = world.section_center().expect("a far centre");
            world.set_far_center(Coord::new(c.x + 1, c.y, c.z));
            publish_far(&mut world);
            for _ in 0..PASSES_PER_MOVE {
                section_pass(&mut world, &mut landed);
            }
        }
        let moving = landed.cancels;
        let passes = drive_sections(&mut world, name, deadline, &mut landed);
        println!(
            "{name}: {} sections ready {passes} passes after stopping, {moving} cancelled while moving",
            world.section_desired.len()
        );
        assert_eq!(landed.wanted, 0, "{name}: the gate descheduled wanted sections");
        assert_eq!(uncovered(&world), 0, "{name}: sections left uncovered");
    }
}

/// Far tiles: block distances in the millions, whose squares overflow i32.
#[test]
fn span_reach_holds_far_tiles_without_overflow() {
    use super::super::section::SectionPos;
    let s = SectionPos { body: 2, face: Face::PosY, detail: crate::ident::Detail(6), x: 1000, z: -1000 };
    let (far, near) = span_reach(s, 0, 0);
    let (x0, z1) = (f64::from(s.min_x()), f64::from(s.min_z() + s.span()));
    let near_want = x0.hypot(z1) as f32;
    assert!(near > 1.0e6 && (near - near_want).abs() < 1.0, "near {near} vs {near_want}");
    assert!(far > near);
    assert_eq!(span_reach(s, s.min_x() + 1, s.min_z() + 1).1, 0.0, "inside the tile");
}
