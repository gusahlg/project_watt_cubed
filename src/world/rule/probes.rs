//! The owner's three LOD symptoms as probes (LOD-RULE step 0). Each streams the owner's view
//! (render distance 16, vertical 5) at a site and measures what is drawn against the rule: inside
//! the render distance every ground column is full-res, and beyond it each far tile is the
//! detail its distance gives, measured from the eye's cell as the rule measures. They failed on
//! the selection the rule replaced and are run by hand:
//! `cargo test --release --lib lod_rule_symptom -- --ignored --nocapture --test-threads 1`

use voxel_engine::DVec3;

use super::*;
use crate::coord::{ChunkCoord as Coord, Face};
use crate::render_config::{LOD_COARSEST_DETAIL, RenderConfig, lod_for};
use crate::world::chunk::CHUNK_SIZE;
use crate::world::fixtures::round_world;
use crate::world::streaming::headless::{step, step_finished};

const OWNER_SEED: i64 = 1_791_184_794_939_118_871;
const OWNER_EYE: DVec3 = DVec3::new(-19.1, 390.8, -0.6);
/// Twin 2's +Y stand on seed 42.
const TWIN_EYE: DVec3 = DVec3::new(296_989_696.0, -308_421_885.0, -370_431_824.0);

/// The start world of `seed` at the owner's view.
fn owner_view(seed: i64) -> World {
    let (lod_levels, lod_detail) = lod_for(16);
    let render = RenderConfig { lod2: true, occlusion: true, lod_levels, lod_detail, ..RenderConfig::default() };
    let mut world = round_world(seed, render);
    world.set_view_distances(16, 5);
    world.section_pyramid.unit = world.view.lod_unit();
    world
}

/// The rule for `world`'s view, on body `body`: a chart stops at its chord rule.
fn rule_of(world: &World, body: u16) -> Rule {
    let ground = match body.checked_sub(crate::world::section::CHART_BODY_BASE) {
        Some(i) => Ground::Chart { index: usize::from(i) },
        None if world.generator.cosmos().is_none() => Ground::Flat,
        None => Ground::Cube { body },
    };
    world.rule_on(ground)
}

/// Stream at `eye` until the whole view is loaded and drawn. Returns the passes taken.
fn settle(world: &mut World, eye: DVec3) -> usize {
    for pass in 0.. {
        if world.loading_full() && world.entry_complete() {
            return pass;
        }
        assert!(pass < 100_000, "did not settle: {} {}", world.entry_debug(), world.clips_debug());
        step_finished(world, eye);
    }
    unreachable!()
}

/// The far tiles drawn now. The engine draws a section whole for any quadrant the cut gives it.
fn drawn(world: &World) -> Vec<SectionPos> {
    world.section_visible.clone()
}

/// The columns tile `s`'s drawn mesh leaves to the full-res chunks.
fn drawn_clip(world: &World, s: SectionPos) -> crate::world::section::ColumnClip {
    match world.sections.get(&s) {
        Some(crate::world::resident::SectionState::Ready { clip, .. }) => clip.drawn,
        _ => crate::world::section::ColumnClip::NONE,
    }
}

/// The drawn far field against the rule.
#[derive(Debug, Default)]
struct CutAudit {
    tiles: usize,
    /// Finer than the rule: a tile whose parent the rule would not split, or finer than the
    /// finest far level outside the render distance.
    finer: usize,
    /// Coarser than the rule: a tile the rule would split.
    coarser: usize,
    /// A tile two or more levels coarser than one farther away.
    inverted: usize,
    /// Drawn past the horizon.
    past: usize,
    worst: Option<(SectionPos, f64)>,
}

fn audit_cut(world: &World, eye: DVec3) -> CutAudit {
    let cell = PhysBox::eye_cell(eye);
    let mut tiles: Vec<(SectionPos, f64)> = drawn(world).into_iter().filter_map(|s| Some((s, world.node_box(s)?.dist_box(&cell)))).collect();
    let mut audit = CutAudit { tiles: tiles.len(), ..CutAudit::default() };
    for &(s, d) in &tiles {
        let rule = rule_of(world, s.body);
        if d >= rule.horizon() {
            audit.past += 1;
            continue;
        }
        let finer = if s.detail.0 < rule.finest {
            d >= rule.r
        } else if s.detail.0 < rule.coarsest {
            world.node_box(s.parent()).is_some_and(|p| !rule.splits(s.detail.0 + 1, p.dist_box(&cell), p.span))
        } else {
            false
        };
        if finer {
            audit.finer += 1;
            if audit.worst.is_none_or(|(_, w)| d > w) {
                audit.worst = Some((s, d));
            }
        }
        audit.coarser += usize::from(world.node_box(s).is_some_and(|b| rule.splits(s.detail.0, d, b.span)));
    }
    tiles.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut farther_min = i8::MAX;
    for &(s, _) in tiles.iter().rev() {
        audit.inverted += usize::from(i32::from(s.detail.0) > i32::from(farther_min) + 1);
        farther_min = farther_min.min(s.detail.0);
    }
    audit
}

/// How the ground columns whose top lies inside the render distance (less `margin` blocks) are
/// drawn: for each, whether a settled chunk holds its top, and whether a far tile draws over it (a
/// tile whose drawn mesh leaves the column out does not). Columns are chunk centres of the near
/// square: charts read storage through the net, a cube face its reference frame, a flat world its
/// plane.
pub(in crate::world) fn ground_columns(world: &World, eye: DVec3, margin: f64) -> Vec<(bool, bool)> {
    let cs = CHUNK_SIZE as i32;
    let center = world.center.expect("a centre");
    let h = world.view.horizontal;
    let reach = f64::from(h * cs) - margin;
    let tiles: Vec<(SectionPos, [i64; 4], crate::world::section::ColumnClip)> = drawn(world).into_iter().map(|s| (s, world.tile_rect(s), drawn_clip(world, s))).collect();
    // A chart's tiles are read in storage, whatever patch they count from.
    let covers = |body: u16, face: Option<Face>, u: i32, v: i32| {
        let (u, v) = (i64::from(u), i64::from(v));
        tiles.iter().any(|&(s, [x0, z0, x1, z1], clip)| {
            s.body == body
                && face.is_none_or(|f| s.face == f)
                && (x0..x1).contains(&u)
                && (z0..z1).contains(&v)
                && !clip.masks(((u - x0) / 16) as i32, ((v - z0) / 16) as i32)
        })
    };
    let settled = |c: Coord| world.chunks.get(&c).is_some_and(|l| l.state.settled());
    let mut out = Vec::new();
    let (reference, _, in_cube) = world.lod_place(center);
    if in_cube {
        let face = world.live_up().expect("an up face");
        let (body, _) = world.section_lod_face.expect("a cube face");
        let warp = world.seams.atlases().iter().find(|a| a.grid.as_ref().is_some_and(|g| g.body == body)).and_then(|a| a.warp.clone());
        let frame = FaceFrame::new(face);
        let (cu, _, cv) = frame.chunk_to_local(reference);
        let d = [reference.x - center.x, reference.y - center.y, reference.z - center.z];
        for dv in -h..=h {
            for du in -h..=h {
                let (u, v) = ((cu + du) * cs + cs / 2, (cv + dv) * cs + cs / 2);
                let open = world.generator.surface(face, u, v);
                if open == i32::MIN {
                    continue;
                }
                let (x, y, z) = frame.cell_to_world((u, open - 1, v));
                let p = DVec3::new(f64::from(x), f64::from(y), f64::from(z));
                let p = warp.as_ref().map_or(p, |w| w.apply(p));
                if p.distance(eye) >= reach {
                    continue;
                }
                let rc = World::chunk_of(x, y, z);
                let c = Coord::new(rc.x - d[0], rc.y - d[1], rc.z - d[2]);
                out.push((settled(c), covers(body, Some(face), u, v)));
            }
        }
        return out;
    }
    let Some(seat) = world.seams.chart_seat(center) else {
        // A flat world's plane.
        for cz in center.z - h..=center.z + h {
            for cx in center.x - h..=center.x + h {
                let (x, z) = (cx * cs + cs / 2, cz * cs + cs / 2);
                let top = world.generator.height(x, z) - 1;
                if DVec3::new(f64::from(x), f64::from(top), f64::from(z)).distance(eye) >= reach {
                    continue;
                }
                out.push((settled(Coord::new(cx, top.div_euclid(cs), cz)), covers(0, Some(Face::PosY), x, z)));
            }
        }
        return out;
    };
    let floor = seat.lo[1].div_euclid(i64::from(cs)) as i32;
    let body = crate::world::section::CHART_BODY_BASE + seat.index as u16;
    for cz in center.z - h..=center.z + h {
        for cx in center.x - h..=center.x + h {
            let Some(c) = world.fold.unfold(Coord::new(cx, floor, cz)) else { continue };
            let (x, z) = (c.x * cs + cs / 2, c.z * cs + cs / 2);
            let top = world.generator.surface(Face::PosY, x, z) - 1;
            let s = DVec3::new(f64::from(x), f64::from(top), f64::from(z));
            let cell = [x as i64, top as i64, z as i64];
            let Some(p) = world.seams.atlases().iter().find_map(|a| a.locate(cell).map(|(patch, _)| a.embed_storage(patch, s))) else { continue };
            if p.distance(eye) >= reach {
                continue;
            }
            out.push((settled(Coord::new(c.x, top.div_euclid(cs), c.z)), covers(body, None, x, z)));
        }
    }
    out
}

/// Ground columns inside the render distance (half a chunk of margin), and how many of them are
/// not drawn by settled full-res chunks alone.
fn not_full_res(world: &World, eye: DVec3) -> (usize, usize) {
    let cols = ground_columns(world, eye, f64::from(CHUNK_SIZE as i32) * 0.75);
    (cols.len(), cols.iter().filter(|&&(settled, far)| !settled || far).count())
}

/// Drawn far tiles that overlap another drawn tile of their tree.
fn overlapping(world: &World) -> usize {
    let tiles = drawn(world);
    let set: crate::world::FastSet<SectionPos> = tiles.iter().copied().collect();
    tiles
        .iter()
        .filter(|&&s| std::iter::successors(Some(s), |p| (p.detail.0 < LOD_COARSEST_DETAIL as i8).then(|| p.parent())).skip(1).any(|p| set.contains(&p)))
        .count()
}

/// Ground columns inside the render distance that nothing draws: no settled chunk and no far tile.
pub(in crate::world) fn bare(world: &World, eye: DVec3) -> usize {
    ground_columns(world, eye, 0.0).iter().filter(|&&(settled, far)| !settled && !far).count()
}

/// A physical eye on the start world `above` blocks over the ground at angle `t` (radians) from
/// direction `from` toward `toward`.
fn arc_eye(world: &World, from: DVec3, toward: DVec3, t: f64, above: f64) -> DVec3 {
    let a = from.normalize();
    let b = (toward - a * toward.dot(a)).normalize();
    world.home_eye(a * t.cos() + b * t.sin(), above)
}

/// The site on arcs from spawn whose near square's ground spans the most: more than the near
/// window's 32 layers can hold beside its eye band.
fn steep_site(world: &World) -> DVec3 {
    let centre = world.generator.cosmos().expect("cosmos").home().centre_f();
    let from = world.chart_spawn().expect("charted") - centre;
    let half = world.view.horizontal * CHUNK_SIZE as i32 + CHUNK_SIZE as i32 / 2;
    let mut best = (i32::MIN, from);
    for toward in [DVec3::X, DVec3::Z, -DVec3::X, -DVec3::Z] {
        for k in 0..1_500 {
            let eye = arc_eye(world, from, toward, f64::from(k) * 2.0e-4, 1.7);
            let Some(storage) = world.chart_eye(eye) else { continue };
            let (x, z) = (storage.x.floor() as i32, storage.z.floor() as i32);
            let Some(seat) = world.seams.chart_seat(World::chunk_of(x, storage.y.floor() as i32, z)) else { continue };
            let body = crate::world::section::CHART_BODY_BASE + seat.index as u16;
            if let Some((lo, hi)) = world.storage_rect(body, x - half, z - half, x + half, z + half)
                && hi - lo > best.0
            {
                best = (hi - lo, eye);
            }
        }
    }
    println!("steepest site: eye {:?}, near-square relief {}", best.1, best.0);
    best.1
}

/// A twin 2 eye `dx` blocks along +X from its +Y stand, as high over the ground as the stand is.
fn twin_walk_eye(world: &World, dx: f64) -> DVec3 {
    let ground = |p: DVec3| f64::from(world.generator.surface(Face::PosY, p.x.floor() as i32, p.z.floor() as i32));
    let over = TWIN_EYE.y - ground(TWIN_EYE);
    let p = TWIN_EYE + DVec3::new(dx, 0.0, 0.0);
    DVec3::new(p.x, ground(p) + over, p.z)
}

/// (a) Coarse LOD that will not leave a hillside. At a chart site whose near square holds more
/// relief than the near window's cap, and on twin 2 after walking 272 blocks from where its relief
/// was baked, ground columns inside the render distance are drawn by far tiles or have no chunk.
#[test]
#[ignore]
fn lod_rule_symptom_a_hillsides_inside_r_are_full_res() {
    let mut failures = Vec::new();
    let mut world = owner_view(42);
    let eye = steep_site(&world);
    world.prepare_around(eye);
    world.drive_spawn_ready();
    let passes = settle(&mut world, eye);
    let (inside, bad) = not_full_res(&world, eye);
    println!("steep chart site: settled in {passes} passes, {bad} of {inside} columns inside R not full-res");
    if bad > 0 {
        failures.push(format!("steep chart site: {bad} of {inside}"));
    }

    let mut world = owner_view(42);
    world.prepare_around(TWIN_EYE);
    world.drive_spawn_ready();
    settle(&mut world, TWIN_EYE);
    let mut eye = TWIN_EYE;
    for k in 1..=17 {
        eye = twin_walk_eye(&world, 16.0 * f64::from(k));
        world.near_eye_prev = None;
        step_finished(&mut world, eye);
    }
    let passes = settle(&mut world, eye);
    let (inside, bad) = not_full_res(&world, eye);
    println!("twin 2 after a 272-block walk: settled in {passes} passes, {bad} of {inside} columns inside R not full-res");
    if bad > 0 {
        failures.push(format!("twin walk: {bad} of {inside}"));
    }
    assert!(failures.is_empty(), "ground inside the render distance not full-res: {failures:?}");
}

/// (b) Detail that does not spread out from the player. Settled at the owner's site, on twin 2
/// and on a flat world, a drawn tile is coarser than its distance gives, or two or more levels
/// coarser than a tile farther away.
#[test]
#[ignore]
fn lod_rule_symptom_b_detail_is_concentric() {
    use crate::world::generation::{FLAT_HEIGHT, WorldgenKind};
    let mut failures = Vec::new();
    let mut check = |name: &str, world: &World, eye: DVec3| {
        let a = audit_cut(world, eye);
        println!("{name}: {a:?}");
        if a.coarser > 0 || a.inverted > 0 {
            failures.push(format!("{name}: {} coarser, {} inverted of {}", a.coarser, a.inverted, a.tiles));
        }
    };
    let mut world = owner_view(OWNER_SEED);
    world.prepare_around(OWNER_EYE);
    world.drive_spawn_ready();
    settle(&mut world, OWNER_EYE);
    check("owner site", &world, OWNER_EYE);

    let mut world = owner_view(42);
    world.prepare_around(TWIN_EYE);
    world.drive_spawn_ready();
    settle(&mut world, TWIN_EYE);
    check("twin 2", &world, TWIN_EYE);

    let (lod_levels, lod_detail) = lod_for(16);
    let render = RenderConfig { lod2: true, occlusion: true, lod_levels, lod_detail, ..RenderConfig::default() };
    let mut world = World::with_kind(1, render, WorldgenKind::Flat, false);
    world.set_view_distances(16, 5);
    let eye = DVec3::new(8.5, f64::from(FLAT_HEIGHT) + 2.0, 8.5);
    world.prepare_around(eye);
    world.drive_spawn_ready();
    settle(&mut world, eye);
    check("flat world", &world, eye);
    assert!(failures.is_empty(), "detail not concentric: {failures:?}");
}

/// (c) Far things in more detail than nearer ones. Settled on a chart seam, flying at 300 m/s,
/// and a few passes after stopping from that flight, a drawn tile is finer than its distance gives.
#[test]
#[ignore]
fn lod_rule_symptom_c_far_tiles_are_never_finer() {
    let mut failures = Vec::new();
    let mut check = |name: &str, world: &World, eye: DVec3| {
        let a = audit_cut(world, eye);
        println!("{name}: {a:?}");
        if a.finer > 0 {
            failures.push(format!("{name}: {} finer of {} (farthest {:?})", a.finer, a.tiles, a.worst));
        }
    };
    let mut world = owner_view(42);
    let seam = world.home_eye(DVec3::new(1.0, 0.995, 0.3), 1.7);
    world.prepare_around(seam);
    world.drive_spawn_ready();
    settle(&mut world, seam);
    check("chart seam", &world, seam);

    let mut world = owner_view(OWNER_SEED);
    world.prepare_around(OWNER_EYE);
    world.drive_spawn_ready();
    settle(&mut world, OWNER_EYE);
    let centre = world.generator.cosmos().expect("cosmos").home().centre_f();
    let from = OWNER_EYE - centre;
    let radius = from.length();
    // 300 m/s at sixty passes a second: five blocks a pass, for 2 km.
    let mut eye = OWNER_EYE;
    for k in 1..=400 {
        eye = arc_eye(&world, from, DVec3::X, 5.0 * f64::from(k) / radius, 120.0);
        step(&mut world, eye);
    }
    check("flying at 300 m/s", &world, eye);
    for _ in 0..30 {
        step(&mut world, eye);
    }
    check("half a second after stopping", &world, eye);
    assert!(failures.is_empty(), "far tiles finer than their distance: {failures:?}");
}

/// The rule's cut at the owner's site, in counts per detail, next to the cut drawn once settled.
#[test]
#[ignore]
fn lod_rule_cut_at_the_owner_site() {
    for seed in [OWNER_SEED, 42] {
        let mut world = owner_view(seed);
        world.prepare_around(OWNER_EYE);
        world.drive_spawn_ready();
        settle(&mut world, OWNER_EYE);
        let mut drawn_per = [0usize; 10];
        for s in drawn(&world) {
            drawn_per[s.detail.0 as usize] += 1;
        }
        let storage = world.chart_eye(OWNER_EYE).expect("a chart eye");
        let seat = world.seams.chart_seat(World::chunk_of(storage.x.floor() as i32, storage.y.floor() as i32, storage.z.floor() as i32)).expect("a seat");
        let ground = Ground::Chart { index: seat.index };
        let (mut memo, mut leaves) = (RuleMemo::default(), Vec::new());
        let t = std::time::Instant::now();
        let rule = world.rule_cut(ground, OWNER_EYE, &mut memo, &mut leaves);
        let us = t.elapsed().as_micros();
        let mut per = [0usize; 10];
        for l in &leaves {
            per[l.pos.detail.0 as usize] += 1;
        }
        let near = leaves.iter().filter(|l| l.dist < rule.r).count();
        println!(
            "seed {seed}: drawn {} tiles {drawn_per:?}; the rule {} tiles {per:?} ({near} within R) in {us} µs; horizon {} finest d{} coarsest d{}",
            drawn_per.iter().sum::<usize>(),
            leaves.len(),
            rule.horizon(),
            rule.finest,
            rule.coarsest
        );
    }
}

/// The start world of `seed` at render distance `h` and vertical distance `v`.
fn view_of(seed: i64, h: i32, v: i32) -> World {
    let (lod_levels, lod_detail) = lod_for(h);
    let render = RenderConfig { lod2: true, occlusion: true, lod_levels, lod_detail, ..RenderConfig::default() };
    let mut world = round_world(seed, render);
    world.set_view_distances(h, v);
    world
}

/// Settled at spawn and on the steepest nearby hillside (render distance 6), every ground column
/// inside the render distance is drawn by settled full-res chunks and by no far tile, and the
/// eye's own column is near.
#[test]
fn near_columns_settle_full_res_at_spawn_and_on_a_hillside() {
    let mut world = view_of(42, 6, 3);
    let spawn = world.chart_spawn().expect("charted");
    let centre = world.generator.cosmos().expect("cosmos").home().centre_f();
    let from = spawn - centre;
    let hill = (0..400)
        .map(|k| arc_eye(&world, from, DVec3::X, f64::from(k) * 1.0e-4, 1.7))
        .max_by_key(|&eye| {
            let s = world.chart_eye(eye).expect("a chart eye");
            let (x, z) = (s.x.floor() as i32, s.z.floor() as i32);
            let ground = |dx: i32, dz: i32| world.generator.surface(Face::PosY, x + dx, z + dz);
            let rim = [(-96, 0), (96, 0), (0, -96), (0, 96)].map(|(dx, dz)| ground(dx, dz));
            rim.iter().max().unwrap() - rim.iter().min().unwrap()
        })
        .expect("sites");
    for (name, eye) in [("spawn", world.home_eye(from, 1.62)), ("hillside", hill)] {
        world.prepare_around(eye);
        world.drive_spawn_ready();
        let passes = settle(&mut world, eye);
        let (inside, bad) = not_full_res(&world, eye);
        println!("{name}: settled in {passes} passes, {bad} of {inside} columns inside R not full-res, {} chunks", world.chunks.len());
        assert!(inside > 0, "{name}: no ground inside R");
        assert_eq!(bad, 0, "{name}: ground inside R not full-res");
    }
}

/// Walking 400 blocks at eight blocks a pass, then descending from 300 over the ground at render
/// distance 6: on every pass every ground column inside the render distance is drawn by a settled
/// chunk or a far tile, and settled again it is drawn by the chunks alone.
#[test]
fn walking_and_descending_never_bare_the_ground() {
    let mut world = view_of(42, 6, 3);
    let spawn = world.chart_spawn().expect("charted");
    let centre = world.generator.cosmos().expect("cosmos").home().centre_f();
    let from = spawn - centre;
    let radius = from.length();
    let start = world.home_eye(from, 1.62);
    world.prepare_around(start);
    world.drive_spawn_ready();
    settle(&mut world, start);
    for k in 1..=50 {
        let eye = arc_eye(&world, from, DVec3::Z, 8.0 * f64::from(k) / radius, 1.62);
        world.near_eye_prev = None;
        step(&mut world, eye);
        assert_eq!(bare(&world, eye), 0, "walk pass {k}: bare ground");
        assert_eq!(overlapping(&world), 0, "walk pass {k}: far tiles drawn over one another");
    }
    let descent: Vec<DVec3> = (0..=38).map(|k| arc_eye(&world, from, DVec3::Z, 400.0 / radius, (300.0 - 8.0 * f64::from(k)).max(1.62))).collect();
    for (k, &eye) in descent.iter().enumerate() {
        world.near_eye_prev = None;
        step(&mut world, eye);
        assert_eq!(bare(&world, eye), 0, "descent pass {k}: bare ground");
        assert_eq!(overlapping(&world), 0, "descent pass {k}: far tiles drawn over one another");
    }
    let eye = *descent.last().expect("a descent");
    settle(&mut world, eye);
    let (inside, bad) = not_full_res(&world, eye);
    assert_eq!(bad, 0, "settled: {bad} of {inside} columns inside R not full-res");
}

/// On a flat world at render distance 6, hovering 200 up (no ground within R) and standing on the
/// ground: no column inside R is ever bare, and settled on the ground every column inside R is
/// drawn by the chunks alone.
#[test]
fn flat_world_near_columns_are_full_res() {
    use crate::world::generation::{FLAT_HEIGHT, WorldgenKind};
    let render = RenderConfig { lod2: true, occlusion: true, ..RenderConfig::default() };
    let mut world = World::with_kind(1, render, WorldgenKind::Flat, false);
    world.set_view_distances(6, 3);
    let at = |above: f64| DVec3::new(8.5, f64::from(FLAT_HEIGHT) + above, 8.5);
    world.prepare_around(at(200.0));
    world.drive_spawn_ready();
    settle(&mut world, at(200.0));
    for k in 0..=25 {
        let eye = at(200.0 - 8.0 * f64::from(k));
        world.near_eye_prev = None;
        step(&mut world, eye);
        assert_eq!(bare(&world, eye), 0, "descent pass {k}: bare ground");
    }
    let eye = at(1.62);
    settle(&mut world, eye);
    let (inside, bad) = not_full_res(&world, eye);
    assert!(inside > 0);
    assert_eq!(bad, 0, "{bad} of {inside} columns inside R not full-res");
}

/// What the near field holds on twin 2 after a walk: a report when (a) fails there.
#[test]
#[ignore]
fn lod_rule_twin_near_report() {
    let mut world = owner_view(42);
    world.prepare_around(TWIN_EYE);
    world.drive_spawn_ready();
    settle(&mut world, TWIN_EYE);
    let cols = ground_columns(&world, TWIN_EYE, 12.0);
    let center = world.center.expect("a centre");
    println!(
        "frame {:?} up {:?} lod_place {:?} near columns {} of {}; settled {} far {}",
        world.near.frame(),
        world.near.up(),
        world.lod_place(center),
        world.near.near_columns().count(),
        cols.len(),
        cols.iter().filter(|c| c.0).count(),
        cols.iter().filter(|c| c.1).count()
    );
    let far_c = world.section_center().expect("far centre");
    let (desired, clips) = world.desired_and_clips(far_c);
    let near_tiles: Vec<_> = drawn(&world).into_iter().filter(|&s| world.node_box(s).is_some_and(|b| b.dist_box(&PhysBox::eye_cell(TWIN_EYE)) < 256.0)).collect();
    println!("desired {} wanted clips {}; drawn tiles within R {}", desired.len(), clips.len(), near_tiles.len());
    let nc: Vec<_> = world.near.near_columns().take(3).collect();
    println!("near columns e.g. {nc:?}; fold identity {}", world.fold.is_identity());
    if let Some(&s) = near_tiles.first() {
        let keys: Vec<_> = (0..8).flat_map(|cz| (0..8).map(move |cx| (cx, cz))).filter_map(|(cx, cz)| world.tile_column_chunks(s, cx, cz)).map(|c| (c[0].x, c[0].z, world.fold.fold(c[0]))).collect();
        println!("tile {s:?} columns {:?} .. {:?}", keys.first(), keys.last());
    }
    for s in near_tiles.iter().take(4) {
        let mut col = None;
        let side = crate::world::section::ColumnClip::side(*s);
        if side.is_some() {
            col = world.tile_column_chunks(*s, 0, 0);
        }
        println!("  {s:?} drawn clip {:#x} target {:#x} wanted {:?} col00 {:?} near {:?}", drawn_clip(&world, *s).0, world.clip_target(*s).0, clips.get(s), col, col.as_ref().and_then(|c| world.near.near_column(world.fold.fold(c[0]))));
    }
}

/// Flying 1.2 km at 100 m/s (a pass a frame) at render distance 6 and stopping: no two drawn far
/// tiles ever overlap, nothing inside the render distance is bare, no drawn tile is finer than its
/// distance gives, and stopped, the view settles full-res inside R.
#[test]
fn flying_never_overlaps_or_bares_and_settles() {
    let mut world = view_of(42, 6, 3);
    let spawn = world.chart_spawn().expect("charted");
    let centre = world.generator.cosmos().expect("cosmos").home().centre_f();
    let from = spawn - centre;
    let radius = from.length();
    let start = world.home_eye(from, 30.0);
    world.prepare_around(start);
    world.drive_spawn_ready();
    settle(&mut world, start);
    let mut eye = start;
    for k in 1..=720 {
        eye = arc_eye(&world, from, DVec3::X, 100.0 / 60.0 * f64::from(k) / radius, 30.0);
        step(&mut world, eye);
        assert_eq!(overlapping(&world), 0, "pass {k}: far tiles drawn over one another");
        let a = audit_cut(&world, eye);
        assert_eq!(a.finer, 0, "pass {k}: {a:?}");
    }
    settle(&mut world, eye);
    let (inside, bad) = not_full_res(&world, eye);
    assert_eq!(bad, 0, "stopped: {bad} of {inside} columns inside R not full-res");
    assert_eq!(audit_cut(&world, eye).coarser, 0, "stopped: tiles coarser than their distance");
}

/// Settled at spawn (render distance 6) and on a flat world, every level from two above the finest
/// to the next-to-coarsest morphs over a band that starts past where the finer level reaches and
/// ends short of where the coarser one begins; the coarsest level does not morph and is meshed
/// flat.
#[test]
fn morph_bands_hold_from_the_cut() {
    use crate::world::generation::{FLAT_HEIGHT, WorldgenKind};
    let mut world = view_of(42, 6, 3);
    let spawn = world.chart_spawn().expect("charted");
    let centre = world.generator.cosmos().expect("cosmos").home().centre_f();
    let eye = world.home_eye(spawn - centre, 1.62);
    world.prepare_around(eye);
    world.drive_spawn_ready();
    settle(&mut world, eye);
    let render = RenderConfig { lod2: true, occlusion: true, ..RenderConfig::default() };
    let mut flat = World::with_kind(1, render, WorldgenKind::Flat, false);
    flat.set_view_distances(6, 3);
    let flat_eye = DVec3::new(8.5, f64::from(FLAT_HEIGHT) + 1.62, 8.5);
    flat.prepare_around(flat_eye);
    flat.drive_spawn_ready();
    settle(&mut flat, flat_eye);
    for (name, w) in [("spawn", &world), ("flat world", &flat)] {
        let ground = w.far_ground(w.section_center().expect("a far centre")).expect("ground");
        let rule = w.rule_on(ground);
        let bands = w.lod_morph;
        println!("{name}: {:?}", &bands[..10]);
        // The two finest levels may not: the eye's cell and their relief fill their whole band. The
        // coarsest level drawn has no coarser one to meet.
        let top_drawn = w.section_desired.iter().map(|s| s.detail.0).max().expect("a cut");
        for d in rule.finest + 2..top_drawn {
            let b = bands[d as usize + 1];
            assert!(b.end > b.start && b.start >= 0.0, "{name}: level d{d} does not morph: {b:?}");
            assert!(f64::from(b.start) >= rule.range(d) * 0.5, "{name}: level d{d} starts before its level does: {b:?}");
        }
        let top = bands[top_drawn as usize + 1];
        assert!(!(top.end > top.start), "{name}: the coarsest level drawn morphs");
        let coarsest = SectionPos { detail: crate::ident::Detail(rule.coarsest), ..w.section_desired[0] };
        assert!(!w.section_morphs(coarsest), "{name}: the coarsest level is meshed with dy");
    }
}
