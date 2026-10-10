//! The owner's three LOD symptoms as probes (LOD-RULE step 0). Each streams the owner's view
//! (render distance 16, vertical 5) at a site and measures what is drawn against the rule: inside
//! the render distance every ground column is full-res, and beyond it each far tile is the
//! detail its distance gives. They fail on the selection the rule replaces and are run by hand:
//! `cargo test --release --lib lod_rule_symptom -- --ignored --nocapture --test-threads 1`

use voxel_engine::DVec3;

use super::select::Surface;
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
    let rule = Rule::new(f64::from(world.view.horizontal * CHUNK_SIZE as i32), LOD_COARSEST_DETAIL as i8);
    let Some(atlas) = body.checked_sub(crate::world::section::CHART_BODY_BASE).and_then(|i| world.seams.atlases().get(usize::from(i))) else {
        return rule;
    };
    let cap = (rule.finest..=rule.coarsest).take_while(|&d| crate::world::section::section_fits(crate::world::section::section_span(crate::ident::Detail(d)), atlas.radius)).last();
    rule.capped(cap.unwrap_or(rule.finest))
}

/// Stream at `eye` until the whole view is loaded and drawn. Returns the passes taken.
fn settle(world: &mut World, eye: DVec3) -> usize {
    for pass in 0.. {
        if world.loading_full() && world.entry_complete() {
            return pass;
        }
        assert!(pass < 100_000, "did not settle: {}", world.entry_debug());
        step_finished(world, eye);
    }
    unreachable!()
}

/// The far tiles drawn now. The engine draws a section whole for any quadrant the cut gives it.
fn drawn(world: &World) -> Vec<SectionPos> {
    world.section_visible.iter().filter(|(_, m)| !m.is_empty()).map(|&(s, _)| s).collect()
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
    let mut tiles: Vec<(SectionPos, f64)> = drawn(world).into_iter().filter_map(|s| Some((s, world.node_box(s)?.dist(eye)))).collect();
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
            world.node_box(s.parent()).is_some_and(|p| !rule.splits(s.detail.0 + 1, p.dist(eye)))
        } else {
            false
        };
        if finer {
            audit.finer += 1;
            if audit.worst.is_none_or(|(_, w)| d > w) {
                audit.worst = Some((s, d));
            }
        }
        audit.coarser += usize::from(rule.splits(s.detail.0, d));
    }
    tiles.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut farther_min = i8::MAX;
    for &(s, _) in tiles.iter().rev() {
        audit.inverted += usize::from(i32::from(s.detail.0) > i32::from(farther_min) + 1);
        farther_min = farther_min.min(s.detail.0);
    }
    audit
}

/// Ground columns whose top lies inside the render distance (half a chunk of margin) and are not
/// drawn by settled full-res chunks alone: no settled chunk holds the top, or a far tile draws
/// over it. Columns are chunk centres of the near square: charts read storage through the net,
/// a cube face its reference frame.
fn not_full_res(world: &World, eye: DVec3) -> (usize, usize) {
    let cs = CHUNK_SIZE as i32;
    let center = world.center.expect("a centre");
    let h = world.view.horizontal;
    let reach = f64::from(h * cs) - f64::from(cs) * 0.75;
    let tiles: Vec<SectionPos> = drawn(world).into_iter().map(|s| world.storage_tile(s)).collect();
    let covers = |body: u16, face: Face, u: i32, v: i32| {
        tiles.iter().any(|s| s.body == body && s.face == face && (s.min_x()..s.min_x() + s.span()).contains(&u) && (s.min_z()..s.min_z() + s.span()).contains(&v))
    };
    let settled = |c: Coord| world.chunks.get(&c).is_some_and(|l| l.state.settled());
    let (mut inside, mut bad) = (0, 0);
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
                inside += 1;
                let rc = World::chunk_of(x, y, z);
                let c = Coord::new(rc.x - d[0], rc.y - d[1], rc.z - d[2]);
                bad += usize::from(!settled(c) || covers(body, face, u, v));
            }
        }
        return (inside, bad);
    }
    let seat = world.seams.chart_seat(center).expect("a chart under the eye");
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
            inside += 1;
            let top_chunk = Coord::new(c.x, top.div_euclid(cs), c.z);
            bad += usize::from(!settled(top_chunk) || covers(body, Face::PosY, x, z));
        }
    }
    (inside, bad)
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

/// The start world's band-0 charts as the rule's trees, measured where the rule would draw them:
/// every chart's tiles of its chord-rule detail near the eye, a tile across a chart's edge cut to
/// the part on the chart.
struct Charts<'a> {
    world: &'a World,
    index: usize,
    /// Each chart's patch, storage box and the eye's local position on it.
    seats: Vec<(crate::space::atlas::Patch, [i64; 3], [i64; 3], (f64, f64))>,
}

impl<'a> Charts<'a> {
    fn new(world: &'a World, eye: DVec3) -> Self {
        use crate::space::atlas::Patch;
        use crate::space::chart::{Map, basis};
        let storage = world.chart_eye(eye).expect("a chart eye");
        let seat = world.seams.chart_seat(World::chunk_of(storage.x.floor() as i32, storage.y.floor() as i32, storage.z.floor() as i32)).expect("a seat");
        let atlas = &world.seams.atlases()[seat.index];
        let n = atlas.bands[0].n;
        let dir = (eye - atlas.centre).normalize();
        let seats = Face::ALL
            .into_iter()
            .filter_map(|face| {
                let (tu, nn, tv) = basis(face);
                let d = DVec3::new(dir.dot(tu), dir.dot(nn), dir.dot(tv));
                if d.y <= 0.05 {
                    return None;
                }
                let (xi, eta) = Map::Equiangular.inverse(d);
                let patch = Patch::Shell { band: 0, face };
                let (o, size) = atlas.storage_box(patch);
                let local = ((xi + 1.0) * 0.5 * n as f64, (eta + 1.0) * 0.5 * n as f64);
                Some((patch, o, [o[0] + size[0], o[1] + size[1], o[2] + size[2]], local))
            })
            .collect();
        Charts { world, index: seat.index, seats }
    }

    fn seat_of(&self, s: SectionPos) -> Option<&(crate::space::atlas::Patch, [i64; 3], [i64; 3], (f64, f64))> {
        let (x, z, w) = (i64::from(s.min_x()), i64::from(s.min_z()), i64::from(s.span()));
        self.seats.iter().find(|(_, lo, hi, _)| x < hi[0] && x + w > lo[0] && z < hi[2] && z + w > lo[2])
    }
}

impl Surface for Charts<'_> {
    fn roots(&self, rule: &Rule, _eye: DVec3, out: &mut Vec<SectionPos>) {
        let body = crate::world::section::CHART_BODY_BASE + self.index as u16;
        let span = i64::from(crate::world::section::section_span(crate::ident::Detail(rule.coarsest)));
        let reach = (rule.horizon() * 1.5) as i64;
        for (_, lo, hi, (lx, lz)) in &self.seats {
            let (ex, ez) = (lo[0] + *lx as i64, lo[2] + *lz as i64);
            let (x0, x1) = ((ex - reach).max(lo[0]), (ex + reach).min(hi[0]));
            let (z0, z1) = ((ez - reach).max(lo[2]), (ez + reach).min(hi[2]));
            if x0 >= x1 || z0 >= z1 {
                continue;
            }
            for z in z0.div_euclid(span)..=(z1 - 1).div_euclid(span) {
                for x in x0.div_euclid(span)..=(x1 - 1).div_euclid(span) {
                    out.push(SectionPos { detail: crate::ident::Detail(rule.coarsest), body, face: Face::PosY, x: x as i32, z: z as i32 });
                }
            }
        }
    }

    fn node_box(&self, s: SectionPos) -> Option<PhysBox> {
        let (patch, lo, hi, _) = self.seat_of(s)?;
        let w = i64::from(s.span());
        let (x0, z0) = (i64::from(s.min_x()).max(lo[0]), i64::from(s.min_z()).max(lo[2]));
        let (x1, z1) = ((i64::from(s.min_x()) + w).min(hi[0]), (i64::from(s.min_z()) + w).min(hi[2]));
        let (a, b) = if (x1 - x0, z1 - z0) == (w, w) {
            self.world.storage_bounds(s.body, x0 as i32, z0 as i32, w as i32)?
        } else {
            self.world.storage_rect(s.body, x0 as i32, z0 as i32, x1 as i32, z1 as i32)?
        };
        let atlas = &self.world.seams.atlases()[self.index];
        let pts = (0..8).map(|c| {
            DVec3::new(
                if c & 1 == 0 { x0 } else { x1 } as f64,
                f64::from(if c & 2 == 0 { a } else { b }),
                if c & 4 == 0 { z0 } else { z1 } as f64,
            )
        });
        let span = w as f64;
        PhysBox::around(pts.map(|p| atlas.embed_storage(*patch, p)), span * span / (8.0 * atlas.radius as f64) + 1.0 + DRESS)
    }
}

/// The rule's cut at the owner's site, in counts per detail, next to main's drawn cut there.
#[test]
#[ignore]
fn lod_rule_cut_at_the_owner_site() {
    for seed in [OWNER_SEED, 42] {
        let mut world = owner_view(seed);
        world.prepare_around(OWNER_EYE);
        world.drive_spawn_ready();
        settle(&mut world, OWNER_EYE);
        let mut main_per = [0usize; 10];
        for s in drawn(&world) {
            main_per[s.detail.0 as usize] += 1;
        }
        let charts = Charts::new(&world, OWNER_EYE);
        let rule = rule_of(&world, crate::world::section::CHART_BODY_BASE + charts.index as u16);
        let (mut stack, mut leaves) = (Vec::new(), Vec::new());
        let t = std::time::Instant::now();
        rule.select(OWNER_EYE, &charts, &mut stack, &mut leaves);
        let us = t.elapsed().as_micros();
        let mut per = [0usize; 10];
        for l in &leaves {
            per[l.pos.detail.0 as usize] += 1;
        }
        let near = leaves.iter().filter(|l| l.dist < rule.r).count();
        let cut = leaves.iter().filter(|l| charts.seat_of(l.pos).is_some_and(|(_, lo, hi, _)| {
            let (x, z, w) = (i64::from(l.pos.min_x()), i64::from(l.pos.min_z()), i64::from(l.pos.span()));
            x < lo[0] || z < lo[2] || x + w > hi[0] || z + w > hi[2]
        })).count();
        println!(
            "seed {seed}: main draws {} tiles {main_per:?}; the rule {} tiles {per:?} ({near} within R, {cut} cut by a chart edge, {} charts) in {us} µs; horizon {} finest d{} coarsest d{}",
            main_per.iter().sum::<usize>(),
            leaves.len(),
            charts.seats.len(),
            rule.horizon(),
            rule.finest,
            rule.coarsest
        );
    }
}
