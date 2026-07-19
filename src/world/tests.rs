//! `World` unit tests, split from `mod.rs` purely for file size — same
//! module tree (`world::tests`), same `super::*` access to private state.

use super::*;
use crate::block::registry::AIR;
use crate::math::Aabb;
use crate::render_config::RenderConfig;
use voxel_engine::{DVec3, Pass};

/// The settled LOD clip: full-res radius only where chunks are actually
/// drawn (or born-air), ring by ring — the far field covers everything
/// beyond, so a loading edge shows coarse terrain, never a hole.
#[test]
fn lod_clip_tracks_the_settled_rings() {
    let mut world = World::generate();
    world.set_view_radius(3);
    let center = ChunkCoord::new(0, 2, 0);
    world.center = Some(center);
    world.ensure_region_data(center);

    // Nothing is drawn yet: the whole far field must stay visible.
    world.refresh_lod_clip();
    assert_eq!(world.lod_clip().radius, 0.0, "unmeshed centre keeps the clip closed");

    // Settle every chunk in the mesh box (Air is settled by definition).
    let coords: Vec<Coord> = world.chunks.keys().copied().collect();
    for coord in coords {
        world.chunks.get_mut(&coord).unwrap().state = MeshState::Air;
    }
    world.lod_clip_grow.set();
    world.refresh_lod_clip();
    assert_eq!(
        world.lod_clip().radius,
        world.view.coverage().radius,
        "fully settled must be bit-identical to the full-res clip"
    );
    assert_eq!(world.lod_clip().half_height, world.view.coverage().half_height);

    // A drawn mesh counts settled the same as Air (edited chunks keep their
    // previous mesh on screen, so they must not reopen the clip).
    let h = MeshHandle::from_raw_parts(7, 1);
    let probe = ChunkCoord::new(1, center.y, 0);
    world.chunks.get_mut(&probe).unwrap().state = MeshState::Dirty { prev: Some(meshes(h)) };
    world.lod_clip_shrunk.set();
    world.refresh_lod_clip();
    assert_eq!(world.lod_clip().radius, world.view.coverage().radius);

    // Unsettle one column at ring 2: the clip retreats to one ring inside it
    // (the nearest face of ring 2 can be 16 m from an off-centre eye).
    world.chunks.get_mut(&ChunkCoord::new(2, center.y, -1)).unwrap().state =
        MeshState::NeedsMesh { building: false, prev: None };
    world.lod_clip_shrunk.set();
    world.refresh_lod_clip();
    assert_eq!(world.lod_clip().radius, CHUNK_SIZE as f32, "rings 0..=1 settled, ring 2 open");

    // Events are the only triggers: without a flag the cached value stands,
    // and growth resumes from the frontier ring once the column settles.
    world.chunks.get_mut(&ChunkCoord::new(2, center.y, -1)).unwrap().state = MeshState::Air;
    world.refresh_lod_clip();
    assert_eq!(world.lod_clip().radius, CHUNK_SIZE as f32, "no event, no rescan");
    world.lod_clip_grow.set();
    world.refresh_lod_clip();
    assert_eq!(world.lod_clip().radius, world.view.coverage().radius);
}

fn lod2_world() -> World {
    World::with_config(DEFAULT_SEED, RenderConfig::default())
}

/// The `section_edit_chunks` index must return exactly what the reference
/// footprint scan returns, for every active detail — including after
/// compaction empties a chunk's edits (the stale index coord must filter
/// out) and for edits above the LOD Y-domain (indexed by neither).
#[test]
fn indexed_section_edits_match_the_footprint_scan() {
    let mut world = lod2_world();
    let stone = world.registry.id_by_name("Stone").unwrap();
    // Edits across several chunk columns and heights, one out of domain.
    world.set_block(3, 50, 4, stone);
    world.set_block(200, 90, -150, stone);
    world.set_block(-40, 10, 700, stone);
    world.set_block(5, section::LOD_CEIL_Y + 5, 5, stone); // outside the domain
    world.set_block(70, 30, 70, stone);
    // Compact one away again: restore what generation would produce.
    let restore = world.generator.block_at(70, 30, 70, world.generator.height(70, 70));
    world.set_block(70, 30, 70, restore);

    let mut positions: Vec<section::SectionPos> = world.section_edit_chunks.keys().copied().collect();
    // Also probe positions with no edits at all.
    positions.push(section::SectionPos { detail: section::FINEST_DETAIL, x: 1000, z: 1000 });
    let sort = |mut v: Vec<(Coord, Vec<(usize, crate::block::registry::BlockId)>)>| {
        for (_, cells) in &mut v {
            cells.sort_unstable();
        }
        v.sort_unstable_by_key(|(c, _)| (c.x, c.y, c.z));
        v
    };
    for pos in positions {
        assert_eq!(
            sort(world.edits_for_section(pos)),
            sort(streaming::edits_in_footprint(&world.edits, pos)),
            "index and scan disagree at {pos:?}",
        );
    }
    // The out-of-domain edit must not have been indexed anywhere.
    for chunks in world.section_edit_chunks.values() {
        assert!(
            chunks.iter().all(|c| c.y < section::DOMAIN_H / CHUNK_SIZE as i32),
            "an out-of-domain edit leaked into the section index"
        );
    }
}

#[test]
fn lod2_is_the_default_and_near_only_leaves_sections_dormant() {
    assert!(World::generate().lod2, "lod2 far field on by default");
    let d = World::with_config(DEFAULT_SEED, RenderConfig { lod2: false, ..RenderConfig::default() });
    assert!(!d.lod2 && d.sections.is_empty() && d.section_visible.is_empty());
}

#[test]
fn section_lane_claim_and_integrate_parity() {
    let mut world = lod2_world();
    let center = ChunkCoord::new(0, 0, 0);
    world.center = Some(center);
    let pos = world.desired_sections(center)[0];
    assert!(!<SectionLane as StreamLane>::in_flight(&world, pos));
    // Claim without its paired submit would trip the debug assertion, so
    // mirror the lane's real order: mint the pending token first.
    world.section_pending_claim = Some((pos, pipeline::ClaimToken(7)));
    <SectionLane as StreamLane>::claim(&mut world, pos);
    assert!(matches!(world.sections.get(&pos), Some(SectionState::Meshing { token: pipeline::ClaimToken(7) })));
    assert!(<SectionLane as StreamLane>::in_flight(&world, pos), "claim marks in-flight");
    <SectionLane as StreamLane>::integrate(
        &mut world,
        pipeline::Done::Section { pos, epoch: 0, token: pipeline::ClaimToken(7), meshes: Default::default() },
    );
    assert_eq!(world.section_upload_queue.len(), 1, "landing queued for upload");

    // A result from a superseded claim (stale token) or a retired epoch
    // must be dropped, never queued.
    <SectionLane as StreamLane>::integrate(
        &mut world,
        pipeline::Done::Section { pos, epoch: 0, token: pipeline::ClaimToken(6), meshes: Default::default() },
    );
    <SectionLane as StreamLane>::integrate(
        &mut world,
        pipeline::Done::Section { pos, epoch: 1, token: pipeline::ClaimToken(7), meshes: Default::default() },
    );
    assert_eq!(world.section_upload_queue.len(), 1, "stale results are dropped");
}

#[test]
fn section_covering_gates_on_a_ready_ancestor_or_self() {
    // A cell is covered by a Ready self or by a Ready ancestor.
    let mut world = lod2_world();
    let center = ChunkCoord::new(0, 0, 0);
    let cell = world.desired_sections(center)[0];
    assert!(!world.section_covered(cell), "nothing loaded means uncovered");
    let empty_ready = || SectionState::Ready { quadrants: Default::default(), last_style: None };
    world.sections.insert(cell, empty_ready());
    assert!(world.section_covered(cell), "a Ready self covers");
    world.sections.remove(&cell);
    world.sections.insert(cell.parent(), empty_ready());
    assert!(world.section_covered(cell), "a Ready ancestor covers the finer cell");
}

/// The fast-movement staleness fix (user report: flying far up left a
/// couple of stale LOD cubes floating over a missing far field): the load
/// lane is LEVEL-triggered — armed for as long as any desired cell is
/// unloaded, uncovered, and unskipped — instead of relying on boundary-
/// crossing events that can go quiet with holes still open.
#[test]
fn section_lane_stays_armed_while_desired_cells_are_uncovered() {
    let mut world = lod2_world();
    let center = ChunkCoord::new(0, 0, 0);
    world.center = Some(center);
    world.pending_sections.take();

    // Fresh world: everything desired is missing — the rebuild must arm.
    world.section_desired = world.desired_sections(center);
    world.rebuild_section_visible(None);
    assert!(world.pending_sections.get(), "open holes must keep the lane armed");

    // Everything in flight (Meshing): no hole is unclaimed — no re-arm.
    world.pending_sections.take();
    for &cell in &world.section_desired.clone() {
        world.sections.insert(cell, SectionState::Meshing { token: pipeline::ClaimToken(0) });
    }
    world.section_desired = world.desired_sections(center);
    world.rebuild_section_visible(None);
    assert!(!world.pending_sections.get(), "in-flight cells are not holes");

    // Everything Ready: converged — still no re-arm.
    world.pending_sections.take();
    for &cell in &world.section_desired.clone() {
        world.sections.insert(cell, SectionState::Ready { quadrants: Default::default(), last_style: None });
    }
    world.section_desired = world.desired_sections(center);
    world.rebuild_section_visible(None);
    assert!(!world.pending_sections.get(), "a converged covering leaves the lane idle");
}

/// A LOD aux lane driven end-to-end through the scheduler's `run_manual`
/// (the call-point path `World::stream` uses) has the same effect as the
/// direct method — here the visible-set lane arming the section lane
/// while the covering has open holes.
#[test]
fn section_visible_lane_drives_through_run_manual() {
    let mut world = lod2_world();
    let center = ChunkCoord::new(0, 0, 0);
    world.center = Some(center);
    world.pending_sections.take();
    world.section_desired = world.desired_sections(center);

    let mut sched = crate::sched::Scheduler::new();
    let handle = sched.register_manual(
        lanes::SectionVisibleLane::manifest(),
        Box::new(lanes::SectionVisibleLane),
    );
    sched.run_manual(handle, &mut world, None);

    assert!(
        world.pending_sections.get(),
        "the visible lane, driven via run_manual, arms the section lane on open holes"
    );
}

/// The mesh admit lane IS the `MeshLane` producer now (no shim): driven
/// end-to-end through the scheduler's `run_manual` it has the same effect as
/// calling `admit` directly — here clearing `pending_fresh` once its
/// worklist is empty (drained).
#[test]
fn mesh_admit_lane_drives_through_run_manual() {
    let mut world = lod2_world();
    let center = ChunkCoord::new(0, 0, 0);
    world.center = Some(center);
    world.mesh_worklist.clear();
    world.pending_fresh.set();

    let mut sched = crate::sched::Scheduler::new();
    let handle = sched.register_manual(MeshLane::manifest(), Box::new(MeshLane));
    sched.run_manual(handle, &mut world, None);

    assert!(
        !world.pending_fresh.get(),
        "the mesh-admit lane, driven via run_manual, drains an empty worklist"
    );
}

/// ANY chunk creation re-arms the far-field lane (it changes the coverage
/// picture the section skip reads), so LOD reacts to streaming activity
/// instead of waiting for the next boundary crossing.
#[test]
fn chunk_creation_arms_the_section_lane() {
    let mut world = lod2_world();
    world.pending_sections.take();
    world.ensure_data(ChunkCoord::new(40, 0, 40));
    assert!(world.pending_sections.get(), "a stored chunk must re-arm the section lane");
}

/// The desired frontier tracks eye ALTITUDE continuously: far above the
/// LOD slab the near rings vanish (they sit wholly overhead) and coarse
/// rings take over, so selection must differ from the ground frontier.
#[test]
fn desired_frontier_responds_to_altitude() {
    let mut world = lod2_world();
    let center = ChunkCoord::new(0, 0, 0);
    world.section_eye_y = 40.0;
    let ground: FastSet<_> = world.desired_sections(center).into_iter().collect();
    world.section_eye_y = 4000.0;
    let sky: FastSet<_> = world.desired_sections(center).into_iter().collect();
    assert_ne!(ground, sky, "altitude must reshape the desired frontier");
    assert!(!sky.is_empty(), "high altitude still selects a (coarser) far field");
    let finest = world.section_pyramid.finest;
    assert!(
        sky.iter().all(|s| s.detail > finest),
        "wholly-overhead near rings must drop out at altitude"
    );
}

/// A settled born-air chunk for testing coverage skip logic.
fn air_chunk(cx: i32, cy: i32, cz: i32) -> Loaded {
    Loaded {
        chunk: std::sync::Arc::new(Chunk::from_uniform(cx, cy, cz, AIR)),
        state: MeshState::Air,
        rev: 0,
        connectivity: None,
        light: None,
    }
}

/// Verify coverage_skips skips only provably-covered sections inside the slab,
/// and only when backed by settled chunks.
#[test]
fn coverage_skip_is_sound_and_backed() {
    use super::heightmip::BakeExtent;
    let mut world = lod2_world();
    let center = ChunkCoord::new(0, 0, 0);
    world.center = Some(center);
    world.view = ViewVolume::view(20);
    world.section_mip = Some(HeightMip::bake(
        &*world.generator,
        &world.registry.color_snapshot(),
        BakeExtent::new(2048, Detail(section::FINEST_DETAIL.0 + 3)),
    ));
    let mip = world.section_mip.clone().unwrap();

    let cell = SectionPos { detail: section::FINEST_DETAIL, x: 0, z: 0 };
    let (lo, hi) = mip.relief_band(cell).expect("near cell is baked");
    // Centre the eye on the cell's relief so its terrain sits inside the slab.
    world.section_eye_y = ((lo + hi) * 0.5) as f64;

    // Selection stays total: skip is invisible to covering.
    assert!(world.desired_sections(center).contains(&cell), "cell still desired");

    // With nothing loaded, footprint is unbacked, so don't skip.
    assert!(!world.coverage_skips(center, cell), "unbacked near disc must not be skipped");

    // Back the footprint with settled chunks.
    let cs = CHUNK_SIZE as i32;
    let nchunks = cell.span() / cs;
    let (cy_lo, cy_hi) = ((lo.floor() as i32).div_euclid(cs), (hi.floor() as i32).div_euclid(cs));
    for cy in cy_lo..=cy_hi {
        for cz in 0..nchunks {
            for cx in 0..nchunks {
                world.chunks.insert(ChunkCoord::new(cx, cy, cz), air_chunk(cx, cy, cz));
            }
        }
    }
    assert!(world.coverage_skips(center, cell), "backed near disc inside the slab is skipped");

    // If any chunk is in-flight, don't skip (fast-descent guard).
    world.chunks.insert(ChunkCoord::new(0, cy_lo, 0), Loaded {
        state: MeshState::NeedsMesh { building: true, prev: None },
        ..air_chunk(0, cy_lo, 0)
    });
    assert!(!world.coverage_skips(center, cell), "an in-flight covering chunk blocks the skip");
}

/// Verify sections outside the skipped core are never skipped.
#[test]
fn coverage_skip_never_skips_outside_the_core() {
    use super::heightmip::BakeExtent;
    let mut world = lod2_world();
    let center = ChunkCoord::new(0, 0, 0);
    world.center = Some(center);
    world.view = ViewVolume::view(20);
    world.section_mip = Some(HeightMip::bake(
        &*world.generator,
        &world.registry.color_snapshot(),
        BakeExtent::new(2048, Detail(section::FINEST_DETAIL.0 + 3)),
    ));
    let mip = world.section_mip.clone().unwrap();

    // Far section: clip draws it, so don't skip it.
    let far = SectionPos { detail: section::FINEST_DETAIL, x: 5, z: 0 };
    if let Some((lo, hi)) = mip.relief_band(far) {
        world.section_eye_y = ((lo + hi) * 0.5) as f64;
    }
    assert!(!world.coverage_skips(center, far), "a far section is never skipped");

    // Near section but eye is high above it: terrain pokes out of slab, so don't skip.
    let near = SectionPos { detail: section::FINEST_DETAIL, x: 0, z: 0 };
    world.section_eye_y = 5000.0;
    assert!(!world.coverage_skips(center, near), "high eye over low ground is not skipped");
}

/// A `Ready` state drawing a single opaque mesh `h` (the common test shape).
fn ready(h: MeshHandle) -> MeshState {
    MeshState::from_upload(ByPass::from_fn(|p| (p == Pass::Opaque).then_some(h)))
}
/// The `ChunkMeshes` for a single opaque mesh `h`.
fn meshes(h: MeshHandle) -> ChunkMeshes {
    match ready(h) {
        MeshState::Ready(m) => m,
        _ => unreachable!("opaque handle yields Ready"),
    }
}

#[test]
fn ground_is_solid_and_sky_is_air() {
    let world = World::generate();
    assert!(world.is_solid(8, 0, 8), "surface-band ground should be solid");
    assert!(
        world.is_solid(8, -200, 8) || world.block_at(8, -200, 8) == AIR,
        "deep query must not panic"
    );
    assert!(world.is_solid(8, -40, 8), "no world floor: stone all the way down");
    assert!(!world.is_solid(8, 40, 8), "sky between terrain and islands is air");
}

#[test]
fn collision_agrees_with_solidity() {
    let world = World::generate();
    let in_ground = Aabb::new(DVec3::new(8.5, 0.5, 8.5), DVec3::new(0.3, 0.3, 0.3));
    let in_sky = Aabb::new(DVec3::new(8.5, 40.0, 8.5), DVec3::new(0.3, 0.3, 0.3));
    let in_deep = Aabb::new(DVec3::new(8.5, -30.0, 8.5), DVec3::new(0.3, 0.3, 0.3));
    assert!(world.collides(&in_ground));
    assert!(!world.collides(&in_sky));
    assert!(world.collides(&in_deep), "uniform stone chunks collide");
}

#[test]
fn collision_grouped_lookup_matches_per_cell_path() {
    let world = World::generate();
    for center in [
        DVec3::new(15.9, 18.0, 15.9),
        DVec3::new(0.1, 21.5, 8.0),
        DVec3::new(-3.2, 19.0, -16.4),
        DVec3::new(4.0, -1.0, 4.0),
        DVec3::new(4.0, 15.9, 4.0),
        DVec3::new(4.0, 200.0, 4.0),
    ] {
        let aabb = Aabb::new(center, DVec3::new(0.4, 0.9, 0.4));
        // Collision skips liquids (solid to mesher, passable to collision).
        let reference = aabb.voxel_cells().any(|(x, y, z)| world.is_obstacle(x, y, z));
        assert_eq!(world.collides(&aabb), reference, "at {center:?}");
    }
}

#[test]
fn column_is_layered_grass_dirt_stone() {
    let mut world = World::generate();
    let reg = world.registry();
    // Terrain speaks elements now: the crust blocks are the natural unions
    // the placement table derives, not the authored Grass/Dirt mixtures
    // (which remain registered for crafting and old saves).
    let (grass, dirt, stone) = (
        reg.id_by_name("Soil+Organic").unwrap(),
        reg.id_by_name("Soil+Clay").unwrap(),
        reg.id_by_name("Stone").unwrap(),
    );

    let (x, z, h) = (0..64)
        .flat_map(|x| (0..64).map(move |z| (x, z)))
        .find_map(|(x, z)| {
            let h = (0..96).rev().find(|&y| world.is_solid(x, y, z))?;
            (world.block_at(x, h, z) == grass && world.block_at(x, h - 3, z) == stone)
                .then_some((x, z, h))
        })
        .expect("a grass-topped column with clean shallow stone near spawn");

    assert_eq!(world.block_at(x, h + 1, z), AIR);
    assert_eq!(world.block_at(x, h, z), grass);
    assert_eq!(world.block_at(x, h - 1, z), dirt);
    assert_eq!(world.block_at(x, h - 3, z), stone);
    // Deep stone persists below y = 0.
    world.ensure_data(World::chunk_of(x, h - 70, z));
    assert_eq!(world.block_at(x, h - 70, z), stone);
}

#[test]
fn restoring_the_generated_block_compacts_the_overlay() {
    // Edit-and-revert must leave NO overlay weight: regeneration produces
    // the reverted block anyway, so saves and join transfers stay
    // proportional to the world's real difference from its seed.
    let mut world = World::generate();
    let (x, z) = (8, 8);
    let h = (0..96).rev().find(|&y| world.is_solid(x, y, z)).unwrap();
    let original = world.block_at(x, h, z);
    assert_eq!(world.edits().count(), 0);
    world.set_block(x, h, z, AIR);
    assert_eq!(world.edits().count(), 1, "a real edit is recorded");
    world.set_block(x, h, z, original);
    assert_eq!(world.edits().count(), 0, "restoring generation drops the entry");
}

#[test]
fn edits_persist_across_unload() {
    let mut world = World::generate();
    let (x, z) = (8, 8);
    let h = (0..64)
        .rev()
        .find(|&y| world.is_solid(x, y, z))
        .unwrap();
    world.set_block(x, h, z, AIR);
    assert_eq!(world.block_at(x, h, z), AIR);
    let stone = world.registry().id_by_name("Stone").unwrap();
    world.set_block(x, 70, z, stone);

    world.chunks.clear();
    world.ensure_data(World::chunk_of(x, h, z));
    world.ensure_data(World::chunk_of(x, 70, z));
    assert_eq!(world.block_at(x, h, z), AIR, "edit survived reload");
    assert_eq!(world.block_at(x, 70, z), stone, "high edit survived reload");
}

#[test]
fn distinct_seeds_differ() {
    let a = World::new(1);
    let b = World::new(9_999);
    let ha: Vec<i32> = (0..16).map(|x| a.surface_y(x, 0)).collect();
    let hb: Vec<i32> = (0..16).map(|x| b.surface_y(x, 0)).collect();
    assert_ne!(ha, hb, "different seeds should sculpt different terrain");
}

#[test]
fn lighting_toggle_reseeds_only_stale_work_and_rejects_old_results() {
    let mut world = World::generate();
    let coord = ChunkCoord::new(0, 0, 0);
    let missing = ChunkCoord::new(1, 0, 0);

    world.light_worklist.clear();
    world.chunks.get_mut(&coord).unwrap().light = Some(light::LightGrid::dark());
    world.light_inflight.insert(coord);
    assert!(world.transition_lighting(false));
    let off_epoch = world.light_epoch;
    assert!(!world.lighting());
    assert!(world.light_worklist.is_empty());
    assert!(world.light_inflight.is_empty());
    assert!(world.chunks[&coord].light.is_none());

    let old = world.block_at(3, 3, 3);
    let stone = world.registry().id_by_name("Stone").unwrap();
    world.set_block(3, 3, 3, if old == AIR { stone } else { AIR });
    world.chunks.get_mut(&missing).unwrap().light = None;
    assert!(world.light_worklist.contains(&coord));

    assert!(world.transition_lighting(true));
    assert_eq!(world.light_epoch, off_epoch.wrapping_add(1));
    assert!(world.light_worklist.contains(&coord));
    assert!(world.light_worklist.contains(&missing));
    assert!(world.light_pending.get());

    // Results from old generation don't publish after re-enable.
    world.light_inflight.insert(coord);
    <LightLane as StreamLane>::integrate(
        &mut world,
        pipeline::Done::Light { coord, epoch: off_epoch, grid: light::LightGrid::dark() },
    );
    assert!(world.light_apply_queue.is_empty());
    let current_epoch = world.light_epoch;
    <LightLane as StreamLane>::integrate(
        &mut world,
        pipeline::Done::Light {
            coord,
            epoch: current_epoch,
            grid: light::LightGrid::dark(),
        },
    );
    assert_eq!(world.light_apply_queue.len(), 1);
    assert!(!world.transition_lighting(true), "same value is a no-op");
}

#[test]
fn failed_jobs_release_claims_then_quarantine_after_repeated_strikes() {
    let mut world = World::generate();
    let coord = *world.chunks.keys().next().unwrap();

    // Mesh lane: a panicked build releases the claim and re-seeds the worklist.
    world.chunks.get_mut(&coord).unwrap().state = MeshState::NeedsMesh { building: true, prev: None };
    world.mesh_worklist.remove(&coord);
    world.fail_job(pipeline::JobKey::Mesh { coord });
    assert!(
        matches!(world.chunks[&coord].state, MeshState::NeedsMesh { building: false, prev: None }),
        "the build claim must be released"
    );
    assert!(world.mesh_worklist.contains(&coord), "released work is re-seeded");

    // Strike out: the third failure quarantines and stops re-seeding.
    world.fail_job(pipeline::JobKey::Mesh { coord });
    world.mesh_worklist.remove(&coord);
    world.fail_job(pipeline::JobKey::Mesh { coord });
    assert!(world.quarantined.contains(&streaming::FailKey::Mesh { coord }));
    assert!(!world.mesh_worklist.contains(&coord), "quarantined claims are not re-seeded");
    assert!(
        !<MeshLane as StreamLane>::ready(&world, coord),
        "the mesh lane skips a quarantined coord"
    );

    // Light lane: claim released and re-seeded, then quarantined likewise.
    world.light_inflight.insert(coord);
    world.light_worklist.remove(&coord);
    world.fail_job(pipeline::JobKey::Light { coord });
    assert!(!world.light_inflight.contains(&coord));
    assert!(world.light_worklist.contains(&coord));
    for _ in 0..2 {
        world.light_inflight.insert(coord);
        world.fail_job(pipeline::JobKey::Light { coord });
    }
    assert!(!world.light_inflight.contains(&coord), "claim always releases");
    assert!(world.quarantined.contains(&streaming::FailKey::Light { coord }));
    assert!(
        <LightLane as StreamLane>::submit(&mut world, coord).is_none(),
        "a quarantined light claim never resubmits"
    );

    // Generate lane: every claimed coord in the failed column span clears.
    let (cx, cz) = (100, 100);
    for cy in 0..=2 {
        world.generating.insert(ChunkCoord::new(cx, cy, cz));
    }
    world.fail_job(pipeline::JobKey::Column { col: (cx, cz), cy: 0..=2 });
    assert!((0..=2).all(|cy| !world.generating.contains(&ChunkCoord::new(cx, cy, cz))));

    // Section lane: a panicked Meshing claim is dropped so selection retries.
    let pos = SectionPos { detail: Detail(2), x: 9, z: 9 };
    world.sections.insert(pos, SectionState::Meshing { token: pipeline::ClaimToken(3) });
    // A stale token must NOT clear the live claim...
    world.fail_job(pipeline::JobKey::Section { pos, epoch: 0, token: pipeline::ClaimToken(2) });
    assert!(world.sections.contains_key(&pos), "a superseded failure leaves the live claim");
    // ...while the exact claim clears it so selection retries.
    world.fail_job(pipeline::JobKey::Section { pos, epoch: 0, token: pipeline::ClaimToken(3) });
    assert!(!world.sections.contains_key(&pos), "the Meshing claim must clear");
}

/// Descheduled (left-behind) jobs release their claims like panics do,
/// but with NO strike, NO quarantine, and no forced requeue — coming back
/// later must re-request the work as if it had never been claimed.
#[test]
fn cancelled_jobs_release_claims_without_strikes() {
    let mut world = World::generate();
    let coord = *world.chunks.keys().next().unwrap();

    world.chunks.get_mut(&coord).unwrap().state = MeshState::NeedsMesh { building: true, prev: None };
    world.cancel_job(pipeline::JobKey::Mesh { coord });
    assert!(matches!(world.chunks[&coord].state, MeshState::NeedsMesh { building: false, prev: None }));

    world.light_inflight.insert(coord);
    world.cancel_job(pipeline::JobKey::Light { coord });
    assert!(!world.light_inflight.contains(&coord));

    let (cx, cz) = (200, 200);
    for cy in 0..=1 {
        world.generating.insert(ChunkCoord::new(cx, cy, cz));
    }
    world.cancel_job(pipeline::JobKey::Column { col: (cx, cz), cy: 0..=1 });
    assert!((0..=1).all(|cy| !world.generating.contains(&ChunkCoord::new(cx, cy, cz))));

    assert!(world.quarantined.is_empty(), "cancellation is not a failure");
    assert!(world.job_strikes.is_empty(), "cancellation earns no strikes");
}

/// The claim rule at the light-result consumption site: an unusable result
/// (its chunk unloaded mid-flight — the ROUTINE fast-flight case) must RELEASE
/// its `light_inflight` claim, never drop silently. A leaked claim wedged the
/// coord forever: a chunk re-loaded there read as perpetually in-flight, so
/// its light never settled, its mesh stayed degraded, and quiescence
/// (`flush_degraded_terminal`, `entry_complete`) never came.
#[test]
fn unloaded_light_result_releases_the_claim() {
    let mut world = World::generate();
    let coord = ChunkCoord::new(0, 0, 0);
    let epoch = world.light_epoch;

    // A light job was in flight when the chunk unloaded (pre-mesh states own
    // no GPU handle, so dropping the entry engine-free is sound).
    world.light_inflight.insert(coord);
    world.chunks.remove(&coord).expect("origin pregenerated");

    <LightLane as StreamLane>::integrate(
        &mut world,
        pipeline::Done::Light { coord, epoch, grid: light::LightGrid::dark() },
    );
    assert!(world.light_apply_queue.is_empty(), "an unusable result never queues");
    assert!(!world.light_inflight.contains(&coord), "the claim must release");

    // The coord is usable again: a re-loaded chunk settles normally instead of
    // reading as forever in-flight.
    world.ensure_data(coord);
    world.settle_light(coord, light::LightGrid::dark());
    assert!(world.chunks[&coord].light.is_some(), "a re-loaded chunk settles");
    assert!(!world.light_inflight.contains(&coord));
}

/// A light grid landing on an already-drawn chunk schedules an ASYNC rebuild:
/// the old mesh keeps drawing (carried as `NeedsMesh.prev`, still `settled()`
/// so the LOD clip holds), the mesh worklist is seeded, and the rev bump
/// strands stale work — while the SYNC dirty path stays untouched (its
/// main-thread build storm during light floods was the post-flight lag).
#[test]
fn light_arrival_schedules_async_rebuild_and_keeps_drawing() {
    let mut world = World::generate();
    world.center = Some(ChunkCoord::new(0, 0, 0));
    let coord = ChunkCoord::new(0, 0, 0);
    let h = MeshHandle::from_raw_parts(11, 1);
    world.chunks.get_mut(&coord).unwrap().state = ready(h);
    world.chunks.get_mut(&coord).unwrap().light = Some(light::LightGrid::open_sky());
    let rev = world.chunks[&coord].rev;

    world.pending_dirty.take();
    world.settle_light(coord, light::LightGrid::dark()); // a CHANGED grid

    let state = &world.chunks[&coord].state;
    assert!(
        matches!(state, MeshState::NeedsMesh { building: false, prev: Some(_) }),
        "async rebuild scheduled with the old mesh carried: {state:?}"
    );
    assert!(state.settled(), "a carried mesh still counts settled (the LOD clip holds)");
    assert!(state.live_meshes().unwrap().draws(h), "the old mesh keeps drawing");
    assert_ne!(world.chunks[&coord].rev, rev, "the rev bump strands stale work");
    assert!(world.mesh_worklist.contains(&coord));
    assert!(world.pending_fresh.get());
    assert!(!world.pending_dirty.get(), "the sync dirty path is NOT involved");
}

/// Claiming an async rebuild flips `building` IN PLACE — a whole-state
/// overwrite would blank the chunk and leak the carried handle — and a stale
/// result's release keeps the carried mesh drawing while re-seeding the retry.
#[test]
fn async_rebuild_claim_and_stale_release_preserve_the_drawn_mesh() {
    let mut world = World::generate();
    world.center = Some(ChunkCoord::new(0, 0, 0));
    let coord = ChunkCoord::new(0, 0, 0);
    let h = MeshHandle::from_raw_parts(12, 1);
    world.chunks.get_mut(&coord).unwrap().state =
        MeshState::NeedsMesh { building: false, prev: Some(meshes(h)) };

    <MeshLane as StreamLane>::claim(&mut world, coord);
    let state = &world.chunks[&coord].state;
    assert!(matches!(state, MeshState::NeedsMesh { building: true, prev: Some(_) }));
    assert!(state.live_meshes().unwrap().draws(h), "still drawing through the claim");

    // A stale result (rev moved on) releases the claim and keeps `prev`.
    let rev = world.chunks[&coord].rev;
    world.chunks.get_mut(&coord).unwrap().rev = rev.wrapping_add(1);
    world.pending_fresh.take();
    world.accept_mesh(coord, rev, pipeline::MeshOutput::new());
    let state = &world.chunks[&coord].state;
    assert!(matches!(state, MeshState::NeedsMesh { building: false, prev: Some(_) }));
    assert!(state.live_meshes().unwrap().draws(h), "a stale drop keeps drawing");
    assert!(world.mesh_worklist.contains(&coord), "re-seeded for the retry");
    assert!(world.pending_fresh.get());
}

/// A boundary cross prunes STALE upload entries in one pass — releasing each
/// claim and re-seeding, exactly like the pop-time stale path — instead of
/// letting a deep post-flight backlog trickle out at drain speed while real
/// uploads wait behind it.
#[test]
fn boundary_cross_prunes_stale_uploads_in_one_pass() {
    let mut world = World::generate();
    world.center = Some(ChunkCoord::new(0, 0, 0));
    let stale_coord = ChunkCoord::new(0, 0, 0);
    let valid_coord = ChunkCoord::new(1, 0, 0);
    for &c in &[stale_coord, valid_coord] {
        world.chunks.get_mut(&c).unwrap().state =
            MeshState::NeedsMesh { building: true, prev: None };
        let rev = world.chunks[&c].rev;
        world.upload_queue.push_back((c, rev, pipeline::MeshOutput::new()));
    }
    // An edit landed while the first entry sat queued.
    world.chunks.get_mut(&stale_coord).unwrap().rev =
        world.chunks[&stale_coord].rev.wrapping_add(1);

    world.pending_fresh.take();
    world.mesh_worklist.clear();
    world.prune_upload_queue();

    assert_eq!(world.upload_queue.len(), 1, "only the stale entry is pruned");
    assert_eq!(world.upload_queue[0].0, valid_coord);
    assert!(
        matches!(world.chunks[&stale_coord].state, MeshState::NeedsMesh { building: false, .. }),
        "the pruned entry's claim is released"
    );
    assert!(world.mesh_worklist.contains(&stale_coord), "and its coord re-seeded");
    assert!(world.pending_fresh.get());
    assert!(
        matches!(world.chunks[&valid_coord].state, MeshState::NeedsMesh { building: true, .. }),
        "the live entry's claim is untouched"
    );
}

/// The byte-based upload budget charges exactly what staging costs: vertex
/// bytes plus index bytes across every pass — pinning the stride assumptions
/// (packed 8 B vertices, u32 indices) the budget's sizing was derived from.
#[test]
fn upload_byte_accounting_matches_vertex_and_index_sizes() {
    let mut world = World::generate();
    world.refresh_tables();
    let coord = ChunkCoord::new(0, 0, 0); // surface chunk: non-empty mesh
    let (_, snapshot) = world.snapshot(coord, true);
    let mut out = pipeline::MeshOutput::new();
    mesh::build_chunk_mesh(
        &snapshot.padded,
        snapshot.uniform,
        &snapshot.tables,
        &snapshot.light.expect("lighting on by default"),
        &mut out,
    );
    let expected: usize = Pass::ALL
        .iter()
        .map(|&p| {
            out[p].vertices().len() * 8
                + out[p].buckets().iter().map(|b| b.len() * 4).sum::<usize>()
        })
        .sum();
    assert!(expected > 0, "a surface chunk yields geometry");
    assert_eq!(streaming::mesh_output_bytes(&out), expected);
}

/// Mesh admission pauses at the upload-queue cap and resumes below it.
#[test]
fn upload_backlog_pauses_mesh_admission_at_the_cap() {
    let mut world = World::generate();
    assert!(!world.upload_backlogged());
    for i in 0..96 {
        world.upload_queue.push_back((ChunkCoord::new(i, 0, 0), 0, pipeline::MeshOutput::new()));
    }
    assert!(world.upload_backlogged(), "at the cap admission pauses");
    world.upload_queue.pop_front();
    assert!(!world.upload_backlogged(), "below the cap it resumes");
}

/// A STALE-epoch light result's claim was wiped by the toggle that bumped the
/// epoch, so consuming it must never touch `light_inflight` — an entry present
/// at its coord belongs to a NEWER job (the epoch-soundness half of
/// `accept_light`'s unconditional release).
#[test]
fn stale_epoch_light_result_never_touches_a_newer_claim() {
    let mut world = World::generate();
    let coord = ChunkCoord::new(0, 0, 0);
    let stale = world.light_epoch;
    assert!(world.transition_lighting(false));
    assert!(world.transition_lighting(true));

    // A post-bump job holds the live claim.
    world.light_inflight.insert(coord);
    <LightLane as StreamLane>::integrate(
        &mut world,
        pipeline::Done::Light { coord, epoch: stale, grid: light::LightGrid::dark() },
    );
    assert!(world.light_inflight.contains(&coord), "the newer claim survives");
    assert!(world.light_apply_queue.is_empty(), "the stale grid never publishes");
}

#[test]
fn stale_rev_mesh_results_are_dropped() {
    let mut world = World::generate();
    world.center = Some(ChunkCoord::new(0, 0, 0));
    let coord = ChunkCoord::new(0, 0, 0);
    let rev = world.chunks[&coord].rev;
    assert!(world.mesh_result_applies(coord, rev));

    world.set_block(3, 3, 3, AIR);
    assert!(!world.mesh_result_applies(coord, rev));

    world.pending_fresh.take();
    world.accept_mesh(coord, rev, pipeline::MeshOutput::new());
    assert!(world.upload_queue.is_empty(), "stale result never queues");
    assert!(world.pending_fresh.get(), "drop re-arms the scan");

    let rev = world.chunks[&coord].rev;
    world.accept_mesh(coord, rev, pipeline::MeshOutput::new());
    assert_eq!(world.upload_queue.len(), 1);
    world.upload_queue.clear();

    // Unloaded / out-of-range coords are rejected too — horizontally and
    // vertically (the vertical bound is the tighter, derived vertical radius).
    assert!(!world.mesh_result_applies(ChunkCoord::new(99, 0, 99), 0));
    assert!(!world.mesh_result_applies(ChunkCoord::new(0, 99, 0), 0));
    let rv = world.view.vertical;
    assert!(!world.mesh_result_applies(ChunkCoord::new(0, rv + 1, 0), 0), "just past vertical range");
}

#[test]
fn neighbour_edits_bump_the_bordering_chunks_rev() {
    let mut world = World::generate();
    let before = world.chunks[&ChunkCoord::new(-1, 0, 0)].rev;
    world.set_block(0, 5, 8, AIR);
    assert_eq!(world.chunks[&ChunkCoord::new(-1, 0, 0)].rev, before + 1, "border neighbour");
    assert_eq!(world.chunks[&ChunkCoord::new(0, 0, 0)].rev, 1, "edited chunk itself");
    assert_eq!(world.chunks[&ChunkCoord::new(1, 0, 0)].rev, 0, "far side untouched");

    // Vertical borders also bump rev on the neighbour below.
    let stone = world.registry().id_by_name("Stone").unwrap();
    let other =
        if world.block_at(8, 16, 8) == crate::block::AIR { stone } else { crate::block::AIR };
    let below = world.chunks[&ChunkCoord::new(0, 0, 0)].rev;
    world.set_block(8, 16, 8, other);
    assert_eq!(world.chunks[&ChunkCoord::new(0, 0, 0)].rev, below + 1, "chunk below bumped");
    assert_eq!(world.chunks[&ChunkCoord::new(0, 1, 0)].rev, 1, "edited vertical chunk");
}

#[test]
fn landed_chunks_replay_edits_that_arrived_mid_flight() {
    let mut world = World::generate();
    world.center = Some(ChunkCoord::new(0, 0, 0));
    let coord = ChunkCoord::new(2, 0, 2);
    let (x, z) = (coord.x * CHUNK_SIZE as i32 + 3, coord.z * CHUNK_SIZE as i32 + 4);
    world.chunks.remove(&coord);
    world.set_block(x, 5, z, AIR);
    let raw = Chunk::new(coord.x, coord.y, coord.z, &*world.generator);
    assert_ne!(raw.get_local(3, 5, 4), AIR, "terrain is solid there");
    world.pending_fresh.take();
    world.accept_chunk(coord, raw);
    assert_eq!(world.block_at(x, 5, z), AIR, "overlay replayed on landing");
    assert!(world.pending_fresh.get(), "new data re-arms the fresh scan");

    let far = ChunkCoord::new(100, 0, 100);
    world.accept_chunk(far, Chunk::new(far.x, far.y, far.z, &*world.generator));
    assert!(!world.chunks.contains_key(&far), "out-of-range chunk dropped");
    let high = ChunkCoord::new(0, 100, 0);
    world.accept_chunk(high, Chunk::new(high.x, high.y, high.z, &*world.generator));
    assert!(!world.chunks.contains_key(&high), "out-of-height chunk dropped");
}

#[test]
fn uniform_air_chunks_are_born_meshed() {
    let world = World::generate();
    let sky = &world.chunks[&ChunkCoord::new(0, 3, 0)];
    assert_eq!(sky.chunk.uniform(), Some(AIR));
    assert_eq!(sky.state, MeshState::Air, "uniform air is born Air");
    assert!(sky.state.live_meshes().is_none());
    let ground = &world.chunks[&ChunkCoord::new(0, 0, 0)];
    assert_eq!(
        ground.state,
        MeshState::NeedsMesh { building: false, prev: None },
        "dense terrain waits for a real mesh"
    );
}

#[test]
fn handle_is_tracked_exactly_once_across_edits() {
    let mut world = World::generate();
    let coord = ChunkCoord::new(0, 0, 0);
    let h = MeshHandle::from_raw_parts(42, 3);
    world.chunks.get_mut(&coord).unwrap().state = ready(h);

    world.set_block(3, 3, 3, AIR);
    assert_eq!(world.chunks[&coord].state, MeshState::Dirty { prev: Some(meshes(h)) });

    world.set_block(4, 4, 4, AIR);
    assert_eq!(
        world.chunks[&coord].state,
        MeshState::Dirty { prev: Some(meshes(h)) },
        "re-edit preserves the single handle"
    );

    assert!(world.chunks[&coord].state.live_meshes().unwrap().draws(h));
    for s in [
        MeshState::Air,
        MeshState::NeedsMesh { building: false, prev: None },
        MeshState::NeedsMesh { building: true, prev: None },
        MeshState::Dirty { prev: None },
    ] {
        world.chunks.get_mut(&coord).unwrap().state = s;
        let state = &world.chunks[&coord].state;
        assert!(state.live_meshes().is_none(), "nothing to free for {state:?}");
    }
}

#[test]
fn edit_during_meshing_drops_the_stale_async_result() {
    let mut world = World::generate();
    world.center = Some(ChunkCoord::new(0, 0, 0));
    let coord = ChunkCoord::new(0, 0, 0);
    world.chunks.get_mut(&coord).unwrap().state = MeshState::NeedsMesh { building: true, prev: None };
    let rev = world.chunks[&coord].rev;

    world.set_block(2, 2, 2, AIR);
    assert!(world.chunks[&coord].state.is_dirty(), "edit turns a building chunk into Dirty");
    assert_ne!(world.chunks[&coord].rev, rev, "edit bumps rev");

    world.pending_fresh.take();
    world.accept_mesh(coord, rev, pipeline::MeshOutput::new());
    assert!(world.upload_queue.is_empty(), "stale mesh result never queues");
    assert!(world.chunks[&coord].state.is_dirty(), "chunk stays Dirty for the sync remesh");
    assert!(world.pending_fresh.get(), "drop re-arms the fresh scan");
}

#[test]
fn mesh_result_stale_by_box_exit_releases_the_claim() {
    // Stale result when chunk left the mesh box (no edit, no rev bump)
    // must release the in-flight claim so the chunk is re-meshable.
    let mut world = World::generate();
    let coord = ChunkCoord::new(0, 0, 0);
    world.center = Some(coord);
    let rev = world.chunks[&coord].rev;
    world.chunks.get_mut(&coord).unwrap().state = MeshState::NeedsMesh { building: true, prev: None };
    world.center = Some(ChunkCoord::new(1000, 0, 0));
    assert!(!world.mesh_result_applies(coord, rev), "out-of-box result is stale");

    world.pending_fresh.take();
    world.accept_mesh(coord, rev, pipeline::MeshOutput::new());
    assert!(world.upload_queue.is_empty(), "stale result never queues");
    assert_eq!(
        world.chunks[&coord].state,
        MeshState::NeedsMesh { building: false, prev: None },
        "claim released, chunk is re-meshable"
    );
    assert!(world.mesh_worklist.contains(&coord), "re-seeded for a later mesh");
    assert!(world.pending_fresh.get(), "drop re-arms the fresh scan");
}

#[test]
fn view_volume_vertical_is_derived_and_flatter() {
    let mut world = World::generate();
    for (view, vertical) in [(1, 2), (3, 2), (4, 2), (6, 3), (8, 4), (10, 5), (20, 5)] {
        world.set_view_radius(view);
        assert_eq!(world.view.horizontal, view, "view {view}");
        assert_eq!(world.view.vertical, vertical, "vertical at view {view}");
    }
}

#[test]
fn explicit_view_distances_clamp_axes_independently() {
    let mut world = World::generate();
    world.center = Some(ChunkCoord::new(1, 2, 3));
    world.set_view_distances(0, 0);
    assert_eq!((world.view_radius(), world.vertical_radius()), (0, 1));
    assert_eq!(world.center, None);
    assert!(world.radius_shrunk.get());

    world.set_view_distances(i32::MAX, i32::MAX);
    assert_eq!((world.view_radius(), world.vertical_radius()), (20, 10));
    assert!(world.radius_shrunk.get(), "a later grow preserves the queued shrink");
}

#[test]
fn horizontal_distance_change_invalidates_the_lod_range() {
    let render = RenderConfig::default();
    let mut world = World::with_config_lazy(DEFAULT_SEED, render);
    assert!(!world.section_config_changed(render));
    world.set_view_distances(3, 2);
    assert!(
        world.section_config_changed(render),
        "LOD unit must follow the full-resolution radius before streaming"
    );
}

#[test]
fn pure_lod_toggle_preserves_the_height_mip_contract() {
    let render = RenderConfig::default();
    let world = World::with_config_lazy(DEFAULT_SEED, render);
    let toggled = RenderConfig { lod2: !render.lod2, ..render };
    assert!(world.section_config_changed(toggled));
    assert!(
        !world.section_pyramid_changed(toggled),
        "on/off does not change the generator-only mip extent"
    );

    let changed = RenderConfig { lod_levels: render.lod_levels + 1, ..render };
    assert!(world.section_pyramid_changed(changed));
}

#[test]
fn streaming_order_weights_vertical_double() {
    let c = ChunkCoord::new(0, 0, 0);
    assert_eq!(World::order(ChunkCoord::new(4, 0, 0), c), 4);
    assert_eq!(World::order(ChunkCoord::new(0, 2, 0), c), 4, "2 layers up ranks like 4 rings out");
    assert!(
        World::order(ChunkCoord::new(0, 3, 0), c) > World::order(ChunkCoord::new(5, 0, 0), c),
        "lateral terrain streams before the sky"
    );
}

#[test]
fn motion_bias_reorders_toward_heading_and_is_identity_at_rest() {
    let base = 1_000_000u64;
    let vx = DVec3::new(1.0, 0.0, 0.0);
    let ahead = motion_biased_dist2(base, vx, 500.0, 0.0);
    let behind = motion_biased_dist2(base, vx, -500.0, 0.0);
    assert!(ahead < base, "cell ahead of motion sorts sooner");
    assert!(behind > base, "cell behind motion sorts later");
    assert_eq!(base - ahead, behind - base, "symmetric about the base key");
    assert_eq!(motion_biased_dist2(base, vx, 0.0, 500.0), base, "perpendicular is unbiased");
    assert_eq!(motion_biased_dist2(base, DVec3::ZERO, 500.0, 0.0), base, "identity at rest");
    assert_eq!(
        motion_biased_dist2(base, DVec3::new(0.1, 0.0, 0.0), 500.0, 0.0),
        base,
        "below speed floor is identity"
    );
}

#[test]
fn view_radius_clamps_and_flags_streaming() {
    let mut world = World::generate();
    assert_eq!(world.view_radius(), DEFAULT_VIEW_RADIUS);
    world.set_view_radius(99);
    assert_eq!(world.view_radius(), 20);
    world.set_view_radius(1);
    assert_eq!(world.view_radius(), 1);
    assert!(world.pending_fresh.get());
    assert_eq!(world.center, None);
}

#[test]
fn queued_shrink_survives_an_intervening_grow() {
    let mut world = World::generate();
    world.set_view_radius(4);
    world.set_view_radius(8);
    assert!(world.radius_shrunk.get(), "grow must not clobber a queued shrink");
    assert!(world.radius_shrunk.take());
    assert!(!world.radius_shrunk.get(), "take consumes it");
}

#[test]
fn invalidate_carries_or_clears_the_owned_mesh() {
    let h = MeshHandle::from_raw_parts(5, 2);
    let mut s = ready(h);
    s.invalidate();
    assert_eq!(s, MeshState::Dirty { prev: Some(meshes(h)) });
    s.invalidate();
    assert_eq!(s, MeshState::Dirty { prev: Some(meshes(h)) });
    for empty in [
        MeshState::Air,
        MeshState::NeedsMesh { building: false, prev: None },
        MeshState::NeedsMesh { building: true, prev: None },
        MeshState::Dirty { prev: None },
    ] {
        let mut s = empty;
        s.invalidate();
        assert_eq!(s, MeshState::Dirty { prev: None });
    }

    // An edit landing mid-async-rebuild carries the still-drawn mesh into
    // `Dirty` — the chunk keeps drawing through both machineries.
    let h2 = MeshHandle::from_raw_parts(6, 1);
    let mut s = MeshState::NeedsMesh { building: true, prev: Some(meshes(h2)) };
    s.invalidate();
    assert_eq!(s, MeshState::Dirty { prev: Some(meshes(h2)) });
}

#[test]
fn from_upload_and_into_owned_agree_on_handle_ownership() {
    let h = MeshHandle::from_raw_parts(9, 1);
    let state = ready(h);
    assert!(state.live_meshes().unwrap().draws(h));
    assert!(matches!(state.into_owned(), Some(m) if m.draws(h)));
    let air = MeshState::from_upload(ByPass::from_fn(|_| None));
    assert_eq!(air, MeshState::Air);
    assert!(air.live_meshes().is_none());
    assert!(air.into_owned().is_none());
}

#[test]
fn edit_raises_pending_dirty_and_enters_the_dirty_fiber() {
    let mut world = World::generate();
    assert!(!world.pending_dirty.get());
    let coord = ChunkCoord::new(0, 0, 0);
    world.set_block(1, 1, 1, AIR);
    assert!(world.pending_dirty.get(), "an edit raises the dirty hint");
    let in_fiber = world.chunks.iter().any(|(&c, l)| c == coord && l.state.is_dirty());
    assert!(in_fiber, "the edited chunk is in the Dirty fiber");
}

#[test]
fn generating_never_shadows_loaded_data() {
    // A coord claimed as a live generate job must be absent — a loaded
    // coord left in `generating` would block its own regeneration.
    let mut world = World::generate();
    let absent = ChunkCoord::new(500, 0, 500);
    assert!(!world.chunks.contains_key(&absent));
    world.generating.insert(absent);
    world.debug_assert_liveness(); // holds: the claim shadows no data
}

#[test]
fn lod2_far_field_drives_to_covering_complete() {
    // Run the real section lane for the whole desired frontier on real
    // terrain, simulating GPU upload as an empty `Ready`, and assert
    // covering resolution terminates (every cell covered by an ancestor or self).
    let mut world = lod2_world();
    let center = ChunkCoord::new(0, 0, 0);
    world.center = Some(center);
    let desired = world.desired_sections(center);
    assert!(!desired.is_empty(), "a lod2 world wants a far frontier at spawn");

    let workers = pipeline::Workers::spawn(2);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while world.desired_sections(center).iter().any(|&c| !world.section_covered(c)) {
        assert!(std::time::Instant::now() < deadline, "covering did not converge: {}", world.entry_debug());
        // Submit every desired-but-unloaded section (the real lane path:
        // submit mints the claim token, acceptance claims it).
        for pos in world.desired_sections(center) {
            if <SectionLane as StreamLane>::in_flight(&world, pos) {
                continue;
            }
            if let Some(job) = <SectionLane as StreamLane>::submit(&mut world, pos) {
                assert!(workers.submit(job), "worker pool admits the section job");
                <SectionLane as StreamLane>::claim(&mut world, pos);
            }
        }
        // Drain finished jobs and yield briefly when the pool is empty.
        let mut got = false;
        while let Some(done) = workers.try_recv() {
            <SectionLane as StreamLane>::integrate(&mut world, done);
            got = true;
        }
        if !got {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        while let Some((pos, token, _meshes)) = world.section_upload_queue.pop_front() {
            if let Some(s @ SectionState::Meshing { .. }) = world.sections.get_mut(&pos)
                && matches!(s, SectionState::Meshing { token: t } if *t == token)
            {
                *s = SectionState::Ready { quadrants: Default::default(), last_style: None };
            }
        }
    }
    assert!(world.sections.values().any(|s| s.is_ready()), "at least one Ready section");
    assert!(
        world.desired_sections(center).iter().all(|&c| world.section_covered(c)),
        "terminal state: every desired cell has a drawable ancestor-or-self",
    );
}

#[test]
fn light_settle_to_identical_grid_reseeds_evicted_mesh_seed() {
    // `admit::<MeshLane>` evicts a light-blocked seed from `mesh_worklist`,
    // trusting `settle_light` to re-seed it once the block clears — even
    // when the settled grid is IDENTICAL to the old one, which would
    // otherwise strand a ready-but-unreachable chunk.
    let mut world = World::generate();
    let c = ChunkCoord::new(0, 0, 0);
    world.center = Some(c);

    // C and its 6 face neighbours must have data (generate() pregenerates
    // near spawn) and each must carry a published light grid so `light_ready`
    // is gateable purely by `light_inflight`/`light_worklist` membership.
    for n in std::iter::once(c).chain(crate::coord::Face::ALL.iter().map(|&f| c.step(f))) {
        world.chunks.get_mut(&n).expect("neighbourhood pregenerated near spawn").light =
            Some(light::LightGrid::dark());
    }
    world.chunks.get_mut(&c).unwrap().state = MeshState::NeedsMesh { building: false, prev: None };
    assert!(world.neighbours_have_data(c));
    assert!(world.in_mesh_box(c));

    // Block C on light: still awaiting its own settle.
    world.light_worklist.clear();
    world.light_inflight.insert(c);
    assert!(!world.light_ready(c), "in-flight light blocks readiness");
    assert!(!<MeshLane as StreamLane>::ready(&world, c), "blocked: not mesh-ready yet");

    // Seed C, then run the real eviction path: admit partitions
    // candidates into ready/blocked and evicts every blocked worklist seed.
    world.mesh_worklist.insert(c);
    world.pending_fresh.set();
    admit::<MeshLane>(
        &mut world,
        c,
        pipeline::Deadline::from_budget(std::time::Duration::from_secs(1)),
    );
    assert!(!world.mesh_worklist.contains(&c), "admit evicted the blocked seed");

    // C's light settles — but to the SAME grid it already had. `settle_light`
    // is the atomic terminal transition: it releases the in-flight claim
    // itself, publishes, and re-arms mesh.
    world.settle_light(c, light::LightGrid::dark());

    assert!(world.light_ready(c), "light_inflight cleared, grid published: ready now");
    assert!(<MeshLane as StreamLane>::ready(&world, c), "every MeshLane::ready predicate holds");
    assert!(world.mesh_worklist.contains(&c), "re-seeded: ready chunk restored to the seed set");
    assert!(world.pending_fresh.get(), "re-armed: the fresh scan will pick it up");
}
