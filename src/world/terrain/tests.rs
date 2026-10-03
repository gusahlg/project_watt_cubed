//! Generator contract and character checks.

use super::*;
use super::cube;
use super::{space, span};
use crate::coord::{ChunkCoord, Face};
use crate::space::FaceFrame;
use crate::world::chunk::Chunk;
use crate::world::generation::Classify;
use crate::world::layout::{ColumnKey, Sky};

fn make(seed: i64) -> (BlockRegistry, Terrain) {
    let mut reg = BlockRegistry::with_builtins();
    let t = Terrain::new(&mut reg, seed);
    (reg, t)
}

/// Every cell of a batch-generated chunk equals the per-voxel query: the two paths share one
/// definition, so a worker, a client and a reaction reading an unloaded cell all agree.
fn assert_chunk_matches(t: &Terrain, cx: i32, cy: i32, cz: i32) {
    let data = t.generate(cx, cy, cz);
    let n = CHUNK_SIZE as i32;
    for ly in 0..CHUNK_SIZE {
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let (x, y, z) = (cx * n + lx as i32, cy * n + ly as i32, cz * n + lz as i32);
                let batch = data.get(Chunk::index(lx, ly, lz));
                let single = t.voxel_at(x, y, z);
                assert_eq!(batch, single, "chunk ({cx},{cy},{cz}) cell ({x},{y},{z})");
            }
        }
    }
}

#[test]
fn batch_generation_equals_the_per_voxel_definition_in_every_realm() {
    let (_reg, t) = make(42);
    // The surface layer of a few columns (trees included), the ground below it, the mine band,
    // the deep, and space.
    for (cx, cz) in [(0, 0), (3, -2), (-7, 11), (40, 40)] {
        let h = t.height(cx * 16 + 8, cz * 16 + 8);
        let top = h.div_euclid(16);
        for cy in [top - 1, top, top + 1] {
            assert_chunk_matches(&t, cx, cy, cz);
        }
    }
    for cy in [-1, -3, -5, -9, -20] {
        assert_chunk_matches(&t, 2, cy, 5);
        assert_chunk_matches(&t, -9, cy, 4);
    }
}

#[test]
#[allow(clippy::reversed_empty_ranges)] // `1..=0`: the height field with no chunk layers
fn column_heights_agree_on_every_path() {
    let (_reg, t) = make(7);
    for (cx, cz) in [(0, 0), (-3, 9), (100, -40)] {
        let (_, hs) = t.generate_column(ColumnKey { face: Face::PosY, a: cx, b: cz }, 1..=0);
        let h16 = t.heights_16(cx, cz);
        for lz in 0..16 {
            for lx in 0..16 {
                let (x, z) = (cx * 16 + lx as i32, cz * 16 + lz as i32);
                assert_eq!(hs[lx + lz * 16], t.height(x, z));
                assert_eq!(h16[lx + lz * 16], t.height(x, z));
                assert!((MIN_GROUND..=MAX_GROUND).contains(&t.height(x, z)));
            }
        }
    }
}

#[test]
fn far_coordinates_generate_without_panic() {
    let (_reg, t) = make(3);
    // Still on the +Y face, far from spawn.
    let h = t.height(1_000_000, -1_000_000);
    assert!((MIN_GROUND..=MAX_GROUND).contains(&h), "on-face height {h}");
    let _ = t.voxel_at(1_000_000, h - 1, -1_000_000);
    // Off every cube: no height, and the cell is air.
    for &(x, z) in &[(1_000_000_000, -1_000_000_000), (i32::MAX - 40, i32::MIN + 40)] {
        assert_eq!(t.height(x, z), i32::MIN);
        assert_eq!(t.voxel_at(x, -500, z), AIR);
        assert_eq!(t.voxel_at(x, SPACE_FLOOR + 900, z), AIR);
        let c = ChunkCoord::new(x.div_euclid(16), 0, z.div_euclid(16));
        assert_eq!(t.classify(c), Classify::Air);
        assert_eq!(t.generate(c.x, c.y, c.z).uniform(), Some(AIR));
    }
}

#[test]
fn the_three_realms_have_their_features() {
    let (reg, t) = make(42);
    let m = t.materials().clone();
    // Mountains and valleys: a wide survey finds both high peaks and low valley floors.
    let (mut lo, mut hi) = (i32::MAX, i32::MIN);
    for i in -60..60 {
        for j in -60..60 {
            let h = t.height(i * 40, j * 40);
            lo = lo.min(h);
            hi = hi.max(h);
        }
    }
    assert!(hi >= 300, "no mountain reaches 300 m (highest {hi})");
    assert!(lo <= 50, "no valley floor below 50 m (lowest {lo})");
    // Trees on the surface somewhere near the origin.
    let mut trees = 0;
    for cx in -6..6 {
        for cz in -6..6 {
            let h = t.height(cx * 16 + 8, cz * 16 + 8);
            for cy in [h.div_euclid(16), h.div_euclid(16) + 1] {
                let d = t.generate(cx, cy, cz);
                for i in 0..CHUNK_VOLUME {
                    trees += (d.get(i) == m.timber) as u32;
                }
            }
        }
    }
    assert!(trees > 0, "no tree trunk in 144 surface columns");
    // Mines: rails, planks and lamps at the mine levels; caves: air underground.
    let (mut rails, mut planks, mut lamps, mut cave_air) = (0, 0, 0, 0);
    for cx in -20..20 {
        for cz in -20..20 {
            for cy in [-1, -3, -5, -7] {
                let d = t.generate(cx, cy, cz);
                if d.uniform().is_some() {
                    continue;
                }
                for i in 0..CHUNK_VOLUME {
                    let id = d.get(i);
                    rails += (id == m.rail) as u32;
                    planks += (id == m.plank) as u32;
                    lamps += (id == m.lamp) as u32;
                    cave_air += (id == AIR) as u32;
                }
            }
        }
    }
    assert!(rails > 50 && planks > 50 && lamps > 0, "mines: {rails} rail, {planks} plank, {lamps} lamp cells");
    assert!(cave_air > 1000, "caves: only {cave_air} open underground cells");
    // The old space realm is air. A moon is charted: nothing of it in physical space, its ground in
    // its chart's storage cells.
    assert_eq!(t.voxel_at(0, SPACE_FLOOR + 10, 0), AIR);
    let moon = t.cosmos.bodies().iter().copied().find(|b| b.kind == cosmos::Kind::Moon).expect("a moon");
    let cosmos::Shape::Ball { r } = moon.shape else { panic!("moon is a ball") };
    for d in [r / 2, r - 8, r - 1] {
        assert_eq!(t.voxel_at((moon.centre[0] + d) as i32, moon.centre[1] as i32, moon.centre[2] as i32), AIR);
    }
    let (x, y, z) = moon_surface_cell(&t, &moon);
    assert_ne!(t.voxel_at(x, y - 1, z), AIR, "the moon's ground, in storage");
    assert_eq!(t.voxel_at(x, y + 40, z), AIR, "its sky");
    let _ = reg;
}

#[test]
fn generated_surface_matter_lies_at_rest_when_disturbed() {
    use crate::sim::reactions::{Budget, CellStore, Pos, ReactionScheduler};
    struct Gen<'a> {
        t: &'a Terrain,
        reg: BlockRegistry,
        edits: std::collections::HashMap<Pos, BlockId>,
    }
    impl CellStore for Gen<'_> {
        fn block_at(&self, p: Pos) -> Option<BlockId> {
            Some(self.edits.get(&p).copied().unwrap_or_else(|| self.t.voxel_at(p.0, p.1, p.2)))
        }
        fn set_block(&mut self, p: Pos, id: BlockId) -> Option<BlockId> {
            let prev = self.block_at(p)?;
            self.edits.insert(p, id);
            Some(prev)
        }
        fn registry(&self) -> &BlockRegistry {
            &self.reg
        }
        fn registry_mut(&mut self) -> &mut BlockRegistry {
            &mut self.reg
        }
    }
    let mut reg = BlockRegistry::with_builtins();
    let t = Terrain::new(&mut reg, 11);
    let mut store = Gen { t: &t, reg, edits: Default::default() };
    let mut s = ReactionScheduler::new();
    // Wake every contact around a slab of surface and subsurface cells.
    for x in -24..24 {
        for z in -24..24 {
            let h = t.height(x, z);
            for y in h - 6..h + 2 {
                s.wake_cell((x, y, z));
            }
        }
    }
    let mut ops = 0;
    for _ in 0..200 {
        ops += s.tick(&mut store, Budget::DEFAULT).len();
        if s.pending() == 0 {
            break;
        }
    }
    assert_eq!(ops, 0, "generated surface matter reacted ({ops} cell changes)");
}

#[test]
fn v4_materials_intern_under_their_role_labels() {
    let (reg, t) = make(1);
    let m = t.materials();
    let pairs = [
        (m.flower_red, "flower_red"),
        (m.flower_yellow, "flower_yellow"),
        (m.flower_blue, "flower_blue"),
        (m.flower_white, "flower_white"),
        (m.cap_red, "cap_red"),
        (m.cap_brown, "cap_brown"),
        (m.stem, "stem"),
        (m.ash, "ash"),
        (m.obsidian, "obsidian"),
        (m.salt, "salt"),
        (m.clay, "clay"),
        (m.limestone, "limestone"),
        (m.marble, "marble"),
        (m.jade, "jade"),
        (m.rust, "rust"),
        (m.mud, "mud"),
        (m.lichen, "lichen"),
        (m.darkwood, "darkwood"),
        (m.bark, "bark"),
        (m.amber, "amber"),
        (m.slate, "slate"),
        (m.cinder, "cinder"),
        (m.petrified, "petrified"),
        (m.tundra, "tundra"),
        (m.glowshroom, "glowshroom"),
        (m.star, "star"),
    ];
    for (id, label) in pairs {
        assert_eq!(reg.label(id), Some(label));
    }
}

#[test]
fn every_reagent_has_a_host_and_empties_its_target() {
    let (mut reg, t) = make(1);
    let m = t.materials();
    for r in &m.reagents {
        assert!(!r.hosts.is_empty(), "reagent {:?} has no dormant host stratum", r.id);
        let (mut cell, mut tool) = (r.target, r.id);
        let mut steps = 0;
        while let Some((_, c, k)) = reg.react(cell, tool) {
            (cell, tool) = (c, k);
            steps += 1;
            assert!(steps < 64);
        }
        assert!(steps > 0, "reagent {:?} does nothing to its target", r.id);
    }
}

#[test]
#[ignore]
fn worldgen_column_cost() {
    let (_reg, t) = make(42);
    let start = std::time::Instant::now();
    let mut chunks = 0;
    for cx in 0..12 {
        for cz in 0..12 {
            let h = t.height(cx * 16, cz * 16).div_euclid(16);
            let (c, _) = t.generate_column(ColumnKey { face: Face::PosY, a: cx, b: cz }, h - 4..=h + 2);
            chunks += c.len();
        }
    }
    let ms = start.elapsed().as_secs_f64() * 1e3;
    println!("{chunks} chunks in {ms:.1} ms: {:.1} µs per chunk, {:.2} ms per column", ms * 1e3 / chunks as f64, ms / 144.0);
}

fn chunk_of(p: [i64; 3]) -> (i32, i32, i32) {
    (
        p[0].div_euclid(16) as i32,
        p[1].div_euclid(16) as i32,
        p[2].div_euclid(16) as i32,
    )
}

/// Chunk containing the top solid cell at the centre of `body`'s `face`.
fn face_centre_chunk(t: &Terrain, body: &cosmos::Body, face: Face) -> (i32, i32, i32) {
    let half = cube::half_of(body);
    let centre = cube::centre_i32(body.centre).expect("cube centre fits i32");
    let (cu, _, cv) = FaceFrame::new(face).cell_to_local(centre);
    let world_a = t.surface(face, cu, cv);
    assert_ne!(world_a, i32::MIN, "body {} {face:?} has no surface", body.id);
    let h = cube::face_h(half, cube::normal_dot(body.centre, face), world_a);
    let a = (half + i64::from(h - 1)) as i32;
    let (rx, ry, rz) = FaceFrame::new(face).cell_to_world((0, a, 0));
    chunk_of([
        i64::from(rx) + body.centre[0],
        i64::from(ry) + body.centre[1],
        i64::from(rz) + body.centre[2],
    ])
}

/// Worker path (`generate_column`), including chunks `generate` stores from `classify`.
fn assert_worker_matches(t: &Terrain, cx: i32, cy: i32, cz: i32) {
    let coord = ChunkCoord::new(cx, cy, cz);
    let (key, alt) = match t.sky(coord) {
        Sky::Axis(face) => ColumnKey::of(face, coord),
        Sky::Open => (ColumnKey { face: Face::PosY, a: cx, b: cz }, cy),
    };
    let (chunks, _) = t.generate_column(key, alt..=alt);
    let data = &chunks.iter().find(|(a, _)| *a == alt).expect("requested layer").1;
    let n = CHUNK_SIZE as i32;
    for ly in 0..CHUNK_SIZE {
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let (x, y, z) = (cx * n + lx as i32, cy * n + ly as i32, cz * n + lz as i32);
                assert_eq!(
                    data.get(Chunk::index(lx, ly, lz)),
                    t.voxel_at(x, y, z),
                    "worker chunk ({cx},{cy},{cz}) cell ({x},{y},{z})"
                );
            }
        }
    }
}

#[test]
fn home_plus_y_keeps_the_v3_field_near_spawn() {
    let (_reg, t) = make(42);
    let paint = t.paints[Face::PosY.index()].as_ref().expect("home +Y");
    for x in [-4000, -80, 0, 8, 40, 1000, 9000] {
        for z in [-9000, -15, 0, 8, 77, 2500] {
            assert_eq!(t.height(x, z), paint.shape.height(x, z), "({x},{z})");
            assert!((MIN_GROUND..=MAX_GROUND).contains(&t.height(x, z)));
        }
    }
}

#[test]
fn faces_edges_bulk_twin_and_moon_match_the_voxel() {
    let (_reg, t) = make(42);
    let home = *t.cosmos.home();
    for face in Face::ALL {
        let (cx, cy, cz) = face_centre_chunk(&t, &home, face);
        let coord = ChunkCoord::new(cx, cy, cz);
        assert_eq!(t.sky(coord), Sky::Axis(face), "home {face:?} centre sky");
        assert_eq!(t.classify(coord), Classify::Mixed, "home {face:?} centre is crust");
        assert_chunk_matches(&t, cx, cy, cz);
    }
    // Inside the rim blend, outside the sky's edge band: batched and blended.
    let x = cosmos::HOME_HALF as i32 - 3_000;
    let h = t.height(x, 0);
    let (cx, cy, cz) = (x.div_euclid(16), h.div_euclid(16), 0);
    assert_eq!(t.sky(ChunkCoord::new(cx, cy, cz)), Sky::Axis(Face::PosY));
    assert_chunk_matches(&t, cx, cy, cz);
    let (_, hs) = t.generate_column(ColumnKey { face: Face::PosY, a: cx, b: 0 }, 1..=0);
    let lu = x.rem_euclid(16) as usize;
    assert_eq!(hs[lu], t.height(x, 0));

    // The seam and the three-face corner are open (edge band) and still match.
    let seam = cosmos::HOME_HALF as i32;
    let hs = t.height(seam, 0);
    assert_chunk_matches(&t, seam.div_euclid(16), hs.div_euclid(16), 0);
    let hc = t.height(seam, seam);
    assert_chunk_matches(&t, seam.div_euclid(16), hc.div_euclid(16), seam.div_euclid(16));

    // Deep bulk, both the classify short-circuit and the column fill.
    let deep_y: i32 = -1_000;
    let (dx, dy, dz) = (0, deep_y.div_euclid(16), 0);
    let deep = ChunkCoord::new(dx, dy, dz);
    let id = match t.classify(deep) {
        Classify::Uniform(id) => id,
        other => panic!("deep chunk should be uniform, got {other:?}"),
    };
    assert_ne!(id, AIR);
    assert_eq!(t.generate(dx, dy, dz).uniform(), Some(id));
    assert_worker_matches(&t, dx, dy, dz);

    // Above every tree on +Y: uniform air, and the worker agrees.
    assert_eq!(t.classify(ChunkCoord::new(0, 31, 0)), Classify::Uniform(AIR));
    assert_eq!(t.generate(0, 31, 0).uniform(), Some(AIR));
    assert_worker_matches(&t, 0, 31, 0);

    // The outward face of the lower twin (the inner faces see each other).
    let mut twins: Vec<_> = t.cosmos.bodies().iter().copied().filter(|b| b.kind == cosmos::Kind::Twin).collect();
    assert_eq!(twins.len(), 2);
    let axis = (0..3).max_by_key(|&a| (twins[0].centre[a] - twins[1].centre[a]).abs()).unwrap();
    twins.sort_by_key(|b| b.centre[axis]);
    let away = [Face::NegX, Face::NegY, Face::NegZ][axis];
    let (cx, cy, cz) = face_centre_chunk(&t, &twins[0], away);
    assert_eq!(t.sky(ChunkCoord::new(cx, cy, cz)), Sky::Axis(away));
    assert_chunk_matches(&t, cx, cy, cz);

    // A moon: physical space holds none of it; its surface chunk in storage matches cell for cell.
    let moon = t.cosmos.bodies().iter().copied().find(|b| b.kind == cosmos::Kind::Moon).unwrap();
    let (cx, cy, cz) = chunk_of(moon.centre);
    assert_eq!(t.classify(ChunkCoord::new(cx, cy, cz)), Classify::Air);
    let (x, y, z) = moon_surface_cell(&t, &moon);
    let (cx, cy, cz) = (x.div_euclid(16), (y - 1).div_euclid(16), z.div_euclid(16));
    assert_eq!(t.sky(ChunkCoord::new(cx, cy, cz)), Sky::Axis(Face::PosY), "storage +Y is the chart's up");
    assert_chunk_matches(&t, cx, cy, cz);
    assert_worker_matches(&t, cx, cy, cz);
}

/// A storage cell just above a moon's ground: the first open cell of the +Y chart's middle column.
fn moon_surface_cell(t: &Terrain, moon: &cosmos::Body) -> (i32, i32, i32) {
    use crate::space::atlas::Patch;
    let atlas = t.atlases().iter().find(|a| (a.centre - moon.centre_f()).length() < 1.0).expect("the moon is charted");
    let b = atlas.bands[0];
    let top = Patch::Shell { band: 0, face: Face::PosY };
    let s = atlas.storage(top, [b.n / 2, 0, b.n / 2]);
    let y = t.surface(Face::PosY, s[0] as i32, s[2] as i32);
    (s[0] as i32, y, s[2] as i32)
}

#[test]
fn cube_edges_share_one_rim_and_the_ridge_is_solid() {
    let (_reg, t) = make(42);
    let h = cosmos::HOME_HALF as i32;
    for z in [0, 1_000, -8_000, h - 10] {
        let dy = t.surface(Face::PosY, h, z);
        let dx = t.surface(Face::PosX, 0, z);
        assert_eq!(dy, dx - h, "edge z={z}");
        assert!((MIN_GROUND..=MAX_GROUND).contains(&dy));
        // One block past the square still belongs to this face (the rim is at least 6 tall)
        // and rebuilds the same surface point.
        assert_eq!(t.surface(Face::PosY, h + 1, z), dy, "wedge z={z}");
        assert_ne!(t.voxel_at(h + dy - 1, dy - 1, z), AIR, "ridge solid z={z}");
        assert_eq!(t.voxel_at(h + dy, dy + 30, z), AIR, "above the ridge z={z}");
    }
    let dy = t.surface(Face::PosY, h, h);
    let dx = t.surface(Face::PosX, 0, h);
    let dz = t.surface(Face::PosZ, h, 0);
    assert_eq!(dy, dx - h, "corner +X");
    assert_eq!(dy, dz - h, "corner +Z");
    assert_ne!(t.voxel_at(h + dy - 1, dy - 1, h + dy - 1), AIR);
    assert_eq!(t.voxel_at(h + dy, dy + 30, h + dy), AIR);
}

#[test]
fn deep_mix_mean_amount_is_the_bulk_density() {
    let (reg, t) = make(1);
    let home = *t.cosmos.home();
    let mut sum = 0.0f64;
    let mut n = 0u32;
    for i in 0..40 {
        for j in 0..40 {
            for k in 0..12 {
                let rel = [i as i64 * 64, j as i64 * 64, k as i64 * 64];
                debug_assert!(cube::in_deep(rel, cube::half_of(&home)));
                sum += f64::from(reg.amount(cube::bulk_id(&t.bulk, &home, rel)));
                n += 1;
            }
        }
    }
    let mean = sum / f64::from(n);
    assert!(
        (mean - cosmos::BULK_DENSITY).abs() <= 0.05,
        "deep mean {mean} over {n} cells"
    );
}

#[test]
fn classify_matches_what_generation_stores() {
    let (_reg, t) = make(42);
    let samples = [
        ChunkCoord::new(0, 1_000, 0),
        ChunkCoord::new(0, 31, 0),
        ChunkCoord::new(0, (-1_000i32).div_euclid(16), 0),
        ChunkCoord::new(0, cosmos::HOME_CENTRE[1].div_euclid(16) as i32, 0),
    ];
    let moon = t.cosmos.bodies().iter().copied().find(|b| b.kind == cosmos::Kind::Moon).unwrap();
    let (mx, my, mz) = chunk_of(moon.centre);
    let mut samples = samples.to_vec();
    samples.push(ChunkCoord::new(mx, my, mz));
    for c in samples {
        let data = t.generate(c.x, c.y, c.z);
        match t.classify(c) {
            Classify::Air => assert_eq!(data.uniform(), Some(AIR), "air {c:?}"),
            Classify::Uniform(id) => assert_eq!(data.uniform(), Some(id), "uniform {c:?}"),
            Classify::Mixed => assert!(data.uniform().is_none() || data.uniform() == Some(AIR), "mixed {c:?}"),
        }
    }
    // Face interior, edge band, empty space, a moon.
    let y = t.height(8, 8).div_euclid(16);
    assert_eq!(t.sky(ChunkCoord::new(0, y, 0)), Sky::Axis(Face::PosY));
    let edge = cosmos::HOME_HALF as i32 - 100;
    assert_eq!(t.sky(ChunkCoord::new(edge.div_euclid(16), 0, 0)), Sky::Open);
    assert_eq!(t.sky(ChunkCoord::new(0, 1_000, 0)), Sky::Open);
    assert_eq!(t.classify(ChunkCoord::new(0, 1_000, 0)), Classify::Air);
    assert_eq!(t.sky(ChunkCoord::new(mx, my, mz)), Sky::Open);
}

/// `cargo test --release -- --ignored --nocapture landmarks`: coordinates worth a screenshot.
#[test]
#[ignore]
fn landmarks() {
    let (_reg, t) = make(42);
    let m = t.materials().clone();
    let mut peak = (0, 0, i32::MIN);
    for i in -80..80 {
        for j in -80..80 {
            let (x, z) = (i * 24, j * 24);
            let h = t.height(x, z);
            if h > peak.2 {
                peak = (x, z, h);
            }
        }
    }
    println!("peak: {peak:?}");
    'mine: for cx in -40..40 {
        for cz in -40..40 {
            for cy in [-1, -2, -3, -4, -5] {
                let d = t.generate(cx, cy, cz);
                if d.uniform().is_some() {
                    continue;
                }
                for i in 0..CHUNK_VOLUME {
                    if d.get(i) == m.lamp {
                        let (lx, ly, lz) = Chunk::local_of(i);
                        println!("mine lamp: ({}, {}, {})", cx * 16 + lx as i32, cy * 16 + ly as i32, cz * 16 + lz as i32);
                        break 'mine;
                    }
                }
            }
        }
    }
    for b in t.cosmos.bodies() {
        println!("{:?} {} centre {:?} {:?}", b.kind, b.id, b.centre, b.shape);
    }
    let mut near: Vec<_> = t.cosmos.clusters().iter().collect();
    near.sort_by(|a, b| a.centre.length().total_cmp(&b.centre.length()));
    for c in near.iter().take(5) {
        println!("cluster {:?} centre {:?} r {:.0} rocks {:.0}", c.form, c.centre, c.radius, c.count);
    }
}

#[test]
fn home_spawn_is_a_temperate_meadow_and_the_twins_differ() {
    let (_reg, t) = make(42);
    let home = t.cosmos.home();
    let expect = [
        (Face::PosY, province::Realm::Green),
        (Face::NegY, province::Realm::Ashen),
        (Face::PosX, province::Realm::Dune),
        (Face::NegX, province::Realm::Shattered),
        (Face::PosZ, province::Realm::Glass),
        (Face::NegZ, province::Realm::Fungal),
    ];
    for (face, realm) in expect {
        assert_eq!(t.paint(home, face).shape.realm(), realm, "{face:?}");
    }
    let here = t.paint(home, Face::PosY).shape.place(0, 0);
    assert_eq!(here.theme, province::ThemeId::Meadow);
    assert!(here.temp > 0.35 && here.temp < 0.98, "temp {}", here.temp);
    let mut themes = Vec::new();
    for z in -15..=15 {
        for x in -15..=15 {
            let (dx, dz) = (x * 200, z * 200);
            if i64::from(dx) * i64::from(dx) + i64::from(dz) * i64::from(dz) > 3_000 * 3_000 {
                continue;
            }
            let theme = t.paint(home, Face::PosY).shape.place(dx, dz).theme;
            if !themes.contains(&theme) {
                themes.push(theme);
            }
        }
    }
    assert!(themes.len() >= 3, "themes within 3 km: {themes:?}");

    let twins: Vec<_> = t.cosmos.bodies().iter().copied().filter(|b| b.kind == cosmos::Kind::Twin).collect();
    assert_eq!(twins.len(), 2);
    let mut realms = Vec::new();
    for b in &twins {
        let realm = t.paint(b, Face::PosY).shape.realm();
        for face in Face::ALL {
            assert_eq!(t.paint(b, face).shape.realm(), realm, "twin {} {face:?}", b.id);
        }
        realms.push(realm);
    }
    realms.sort_by_key(|r| *r as u8);
    assert_eq!(realms, vec![province::Realm::Lush, province::Realm::Crystal]);
}

#[test]
fn batched_columns_match_the_single_column() {
    let (_reg, t) = make(42);
    let shape = &t.paints[Face::PosY.index()].as_ref().expect("+Y").shape;
    for (u0, v0) in [(0, 0), (-80, 64)] {
        let cols = shape.columns_16(u0, v0);
        for lv in 0..16 {
            for lu in 0..16 {
                let (u, v) = (u0 + lu, v0 + lv);
                let one = shape.column(u, v);
                let got = &cols[lu as usize + lv as usize * 16];
                assert_eq!(got.height, one.height, "({u},{v})");
                assert_eq!(got.slope4, one.slope4, "({u},{v})");
                assert_eq!(got.surface, one.surface, "({u},{v})");
                assert_eq!(got.sub, one.sub, "({u},{v})");
                assert_eq!(got.theme, one.theme, "({u},{v})");
                assert_eq!(got.species, one.species, "({u},{v})");
                assert_eq!(got.flower, one.flower, "({u},{v})");
                assert_eq!(got.flowers.to_bits(), one.flowers.to_bits(), "({u},{v})");
                assert_eq!(got.trees.to_bits(), one.trees.to_bits(), "({u},{v})");
            }
        }
    }
}

#[test]
fn flowers_sit_on_the_meadow() {
    let (_reg, t) = make(42);
    let shape = &t.paints[Face::PosY.index()].as_ref().expect("+Y").shape;
    let m = t.materials();
    let mut n = 0;
    let mut checked = false;
    for z in (-48..48).step_by(2) {
        for x in (-48..48).step_by(2) {
            let col = shape.column(x, z);
            let Some(id) = shape.flower_at(&col, x, z) else { continue };
            n += 1;
            assert!(
                id == m.flower_red || id == m.flower_yellow || id == m.flower_blue || id == m.flower_white,
                "flower {id:?}"
            );
            if !checked && t.voxel_at(x, col.height, z) == id {
                checked = true;
            }
        }
    }
    assert!(n > 0, "no flowers within 48 m of spawn");
    assert!(checked, "every flower sat under a tree");
}

/// Face window for tuning. `WATT_MAP_SEED` (i64), `WATT_MAP_FACE` (py/ny/px/nx/pz/nz),
/// `WATT_MAP_SIZE` (pixels, 8..=768), `WATT_MAP_SCALE` (blocks per pixel), `WATT_MAP_U0`,
/// `WATT_MAP_V0`. Writes height and province-colour PPM files under `CARGO_TARGET_DIR`.
#[test]
#[ignore]
fn worldgen_map() {
    fn env_i64(key: &str, default: i64) -> i64 {
        std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
    }
    fn env_i32(key: &str, default: i32) -> i32 {
        std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
    }
    let seed = env_i64("WATT_MAP_SEED", 42);
    let face = match std::env::var("WATT_MAP_FACE").unwrap_or_else(|_| "py".into()).as_str() {
        "ny" => Face::NegY,
        "px" => Face::PosX,
        "nx" => Face::NegX,
        "pz" => Face::PosZ,
        "nz" => Face::NegZ,
        _ => Face::PosY,
    };
    let size = env_i32("WATT_MAP_SIZE", 192).clamp(8, 768);
    let scale = env_i32("WATT_MAP_SCALE", 64).max(1);
    let span = size.saturating_mul(scale);
    let u0 = env_i32("WATT_MAP_U0", -span / 2);
    let v0 = env_i32("WATT_MAP_V0", -span / 2);
    let (_reg, t) = make(seed);
    let shape = &t.paints[face.index()].as_ref().expect("home face").shape;
    let mut height = vec![0u8; (size as usize) * (size as usize) * 3];
    let mut province_px = vec![0u8; height.len()];
    let mut lo = i32::MAX;
    let mut hi = i32::MIN;
    for row in 0..size {
        for col in 0..size {
            let u = u0.saturating_add(col.saturating_mul(scale));
            let v = v0.saturating_add(row.saturating_mul(scale));
            let sample = shape.column(u, v);
            let h = sample.height;
            lo = lo.min(h);
            hi = hi.max(h);
            let rgb = hypo(h);
            let i = ((row as usize) * (size as usize) + col as usize) * 3;
            height[i..i + 3].copy_from_slice(&rgb);
            province_px[i..i + 3].copy_from_slice(&province::theme_rgb(sample.theme));
        }
    }
    let tag = match face {
        Face::PosY => "py",
        Face::NegY => "ny",
        Face::PosX => "px",
        Face::NegX => "nx",
        Face::PosZ => "pz",
        Face::NegZ => "nz",
    };
    let dir = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".into());
    let stem = format!("{dir}/worldgen-map-{seed}-{tag}");
    write_ppm(&format!("{stem}.ppm"), size, &height);
    write_ppm(&format!("{stem}-province.ppm"), size, &province_px);
    println!("worldgen_map {stem}.ppm height {lo}..={hi} ({size}px, {scale} blocks)");
}

fn hypo(h: i32) -> [u8; 3] {
    let t = ((h - MIN_GROUND) as f32 / (MAX_GROUND - MIN_GROUND) as f32).clamp(0.0, 1.0);
    if t < 0.35 {
        let u = t / 0.35;
        [(40.0 + 50.0 * u) as u8, (110.0 + 70.0 * u) as u8, (150.0 - 70.0 * u) as u8]
    } else if t < 0.72 {
        let u = (t - 0.35) / 0.37;
        [(90.0 + 70.0 * u) as u8, (170.0 - 50.0 * u) as u8, (70.0 - 20.0 * u) as u8]
    } else {
        let u = (t - 0.72) / 0.28;
        [(160.0 + 90.0 * u) as u8, (120.0 + 120.0 * u) as u8, (60.0 + 180.0 * u) as u8]
    }
}

fn write_ppm(path: &str, n: i32, rgb: &[u8]) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).expect(path);
    write!(f, "P6\n{n} {n}\n255\n").unwrap();
    f.write_all(rgb).unwrap();
}

fn rock_index(kind: cosmos::RockKind) -> usize {
    match kind {
        cosmos::RockKind::Rocky => 0,
        cosmos::RockKind::Carbon => 1,
        cosmos::RockKind::Metallic => 2,
        cosmos::RockKind::Icy => 3,
        cosmos::RockKind::Geode => 4,
        cosmos::RockKind::Derelict => 5,
    }
}

/// One rock of each kind, taken from cluster centres (class-2 sub-cells there are usually full).
fn one_of_each_kind(t: &Terrain) -> Vec<cosmos::Rock> {
    let mut have = [false; 6];
    let mut out = Vec::new();
    for c in t.cosmos.clusters() {
        let p = [c.centre.x as i64, c.centre.y as i64, c.centre.z as i64];
        let mut buf = Vec::new();
        t.cosmos.rocks_touching(p, p, &mut buf);
        for r in buf {
            let i = rock_index(r.kind);
            if !have[i] {
                have[i] = true;
                out.push(r);
            }
        }
        if have.iter().all(|h| *h) {
            break;
        }
    }
    assert!(have.iter().all(|h| *h), "the catalog sample missed a rock kind: {have:?}");
    out
}

fn solid_chunk(t: &Terrain, rock: &cosmos::Rock) -> (i32, i32, i32, [i64; 3]) {
    let c = [rock.centre[0] as i64, rock.centre[1] as i64, rock.centre[2] as i64];
    let r = rock.r.max(1.0) as i64;
    for dist in [r * 3 / 5, r * 7 / 10, r / 2, r * 4 / 5, r / 3, 0] {
        for axis in 0..3 {
            let mut p = c;
            p[axis] += dist;
            if space::paint(rock, t.materials(), p).is_some_and(|id| id != AIR) {
                return (chunk_of(p).0, chunk_of(p).1, chunk_of(p).2, p);
            }
        }
    }
    panic!("no solid cell in rock at {:?} r={}", rock.centre, rock.r);
}

fn face_world(body: &cosmos::Body, face: Face, u: i32, h: i32, v: i32) -> [i32; 3] {
    let half = cube::half_of(body) as i32;
    let (x, y, z) = FaceFrame::new(face).cell_to_world((u, half + h, v));
    let c = cube::centre_i32(body.centre).expect("twin centre fits i32");
    [x + c.0, y + c.1, z + c.2]
}

#[test]
fn asteroids_round_worlds_and_the_twin_canyon() {
    let (_reg, t) = make(42);
    let m = t.materials();
    let rocks = one_of_each_kind(&t);
    for rock in &rocks {
        let (cx, cy, cz, p) = solid_chunk(&t, rock);
        let coord = ChunkCoord::new(cx, cy, cz);
        assert_ne!(t.classify(coord), Classify::Air, "{:?} chunk classified empty", rock.kind);
        assert_ne!(t.voxel_at(p[0] as i32, p[1] as i32, p[2] as i32), AIR);
        assert_chunk_matches(&t, cx, cy, cz);
        assert_worker_matches(&t, cx, cy, cz);
        let reach = space::reach(rock);
        let c = [rock.centre[0] as i64, rock.centre[1] as i64, rock.centre[2] as i64];
        for axis in 0..3 {
            for sign in [-1, 1] {
                let mut q = c;
                q[axis] += sign * (reach + 1);
                assert!(space::paint(rock, m, q).is_none(), "{:?} leaks on axis {axis}", rock.kind);
            }
        }
    }

    // A small rock's whole sub-cell, and the cell just outside it, stay inside the reserved box.
    let mut small = None;
    'find: for c in t.cosmos.clusters() {
        let p = [c.centre.x as i64, c.centre.y as i64, c.centre.z as i64];
        let mut buf = Vec::new();
        t.cosmos.rocks_touching(p, p, &mut buf);
        for r in buf {
            if r.r < 22.0 && space::reach(&r) >= 16 {
                small = Some(r);
                break 'find;
            }
        }
    }
    let small = small.expect("a class-0 rock");
    let edge = 64i64;
    let s0 = [
        (small.centre[0] as i64).div_euclid(edge) * edge,
        (small.centre[1] as i64).div_euclid(edge) * edge,
        (small.centre[2] as i64).div_euclid(edge) * edge,
    ];
    let reach = space::reach(&small);
    let centre = [small.centre[0] as i64, small.centre[1] as i64, small.centre[2] as i64];
    let mut painted = 0u32;
    for x in 0..edge {
        for y in 0..edge {
            for z in 0..edge {
                let p = [s0[0] + x, s0[1] + y, s0[2] + z];
                if space::paint(&small, m, p).is_some() {
                    painted += 1;
                    for a in 0..3 {
                        assert!((p[a] - centre[a]).abs() <= reach, "class-0 rock leaves its box");
                    }
                }
            }
        }
    }
    assert!(painted > 0, "the small rock paints something");
    for a in 0..3 {
        for side in [s0[a] - 1, s0[a] + edge] {
            for u in 0..edge {
                for v in 0..edge {
                    let mut p = s0;
                    p[a] = side;
                    p[(a + 1) % 3] += u;
                    p[(a + 2) % 3] += v;
                    assert!(space::paint(&small, m, p).is_none(), "paint outside the sub-cell");
                }
            }
        }
    }
    let mut tight = false;
    for corner in 0..8 {
        let origin: [i64; 3] = std::array::from_fn(|a| s0[a] + if corner & (1 << a) == 0 { 0 } else { edge - 16 });
        let coord = ChunkCoord::new((origin[0] / 16) as i32, (origin[1] / 16) as i32, (origin[2] / 16) as i32);
        let (lo, hi) = cube::chunk_bounds(coord);
        if space::overlaps(&small, lo, hi) {
            continue;
        }
        if t.cosmos.bodies_touching(lo, hi).next().is_some() || space::any_overlap(&t.cosmos, lo, hi) {
            continue;
        }
        assert_eq!(t.classify(coord), Classify::Air, "a sub-cell corner far from every rock is air");
        assert_eq!(t.generate(coord.x, coord.y, coord.z).uniform(), Some(AIR));
        tight = true;
        break;
    }
    assert!(tight, "no empty corner of the small rock's sub-cell");

    let mut empty = None;
    for i in 0..80 {
        let x = 30_000_000i32 + i * 250_000;
        let c = ChunkCoord::new(x.div_euclid(16), 0, i * 3);
        let (lo, hi) = cube::chunk_bounds(c);
        if !t.cosmos.may_hold(lo, hi) {
            empty = Some(c);
            break;
        }
    }
    let empty = empty.expect("an empty space chunk");
    assert_eq!(t.classify(empty), Classify::Air);
    assert_eq!(t.generate(empty.x, empty.y, empty.z).uniform(), Some(AIR));

    for b in t.cosmos.bodies() {
        let c = b.centre;
        let air = |p: [i64; 3]| {
            assert_eq!(t.voxel_at(p[0] as i32, p[1] as i32, p[2] as i32), AIR, "{:?} physical cell {p:?}", b.kind);
        };
        match b.shape {
            cosmos::Shape::Cube { .. } => {}
            cosmos::Shape::Ball { r } => {
                air(c);
                air([c[0] + r / 2, c[1], c[2]]);
                air([c[0] + r - 10, c[1], c[2]]);
                air([c[0] + r + 100, c[1], c[2]]);
            }
            cosmos::Shape::Shell { outer, inner } => {
                air(c);
                air([c[0] + (outer + inner) / 2, c[1], c[2]]);
                air([c[0] + outer - 10, c[1], c[2]]);
            }
        }
    }

    let twins: Vec<_> = t.cosmos.bodies().iter().copied().filter(|b| b.kind == cosmos::Kind::Twin).collect();
    assert_eq!(twins.len(), 2);
    let lush_id = twins.iter().map(|b| b.id).min().unwrap();
    for body in &twins {
        let face = span::facing_face(&t.cosmos, body).unwrap();
        let half = cube::half_of(body);
        let lush = body.id == lush_id;
        let (u, h, v) = span::example(span::face_seed(body), half).expect("a spire");
        assert!(h > MAX_GROUND && (h as i64) < cosmos::RELIEF, "spire altitude {h}");
        let w = face_world(body, face, u, h, v);
        let id = t.voxel_at(w[0], w[1], w[2]);
        if lush {
            assert!(id == m.timber || id == m.jade, "lush spire is {id:?}");
        } else {
            assert!(id == m.marble || id == m.crystal, "crystal spire is {id:?}");
        }
        let (cx, cy, cz) = chunk_of([w[0] as i64, w[1] as i64, w[2] as i64]);
        assert_ne!(t.classify(ChunkCoord::new(cx, cy, cz)), Classify::Air);
        assert_chunk_matches(&t, cx, cy, cz);
        assert_worker_matches(&t, cx, cy, cz);
        let (au, ah, av) = span::an_arch(m, lush, span::face_seed(body), half).expect("an arch");
        assert!(ah > MAX_GROUND && (ah as i64) < cosmos::RELIEF);
        let aw = face_world(body, face, au, ah, av);
        let arch = t.voxel_at(aw[0], aw[1], aw[2]);
        let ok = if lush { arch == m.timber || arch == m.leaves } else { arch == m.crystal || arch == m.glowshroom };
        assert!(ok, "arch block {arch:?}");
        let clear = face_world(body, face, 0, span::CLEAR + 16, 0);
        assert_eq!(t.voxel_at(clear[0], clear[1], clear[2]), AIR);
        let cc = chunk_of([clear[0] as i64, clear[1] as i64, clear[2] as i64]);
        // The twin still owns this chunk, so air above the spires is uniform air.
        let above = t.classify(ChunkCoord::new(cc.0, cc.1, cc.2));
        assert!(above == Classify::Air || above == Classify::Uniform(AIR), "above the spires: {above:?}");
        let (nx, ny, nz) = face.normal();
        let out = [
            body.centre[0] + nx as i64 * (half + 3_000),
            body.centre[1] + ny as i64 * (half + 3_000),
            body.centre[2] + nz as i64 * (half + 3_000),
        ];
        assert_eq!(t.voxel_at(out[0] as i32, out[1] as i32, out[2] as i32), AIR, "the canyon beyond relief is empty");
    }
    let mid = [
        (twins[0].centre[0] + twins[1].centre[0]) / 2,
        (twins[0].centre[1] + twins[1].centre[1]) / 2,
        (twins[0].centre[2] + twins[1].centre[2]) / 2,
    ];
    assert_eq!(t.voxel_at(mid[0] as i32, mid[1] as i32, mid[2] as i32), AIR, "the canyon midpoint is empty");
    let mc = chunk_of(mid);
    assert_eq!(t.classify(ChunkCoord::new(mc.0, mc.1, mc.2)), Classify::Air);
}

/// `cargo test --lib space_chunk_costs -- --ignored --nocapture`
#[test]
#[ignore]
fn space_chunk_costs() {
    let (_reg, t) = make(42);
    let rocks = one_of_each_kind(&t);
    let small = rocks.iter().min_by(|a, b| a.r.total_cmp(&b.r)).unwrap();
    let (cx, cy, cz, _) = solid_chunk(&t, small);
    let mut empty = ChunkCoord::new(0, 0, 0);
    for i in 0..80 {
        let x = 30_000_000i32 + i * 250_000;
        let c = ChunkCoord::new(x.div_euclid(16), 0, i * 3);
        let (lo, hi) = cube::chunk_bounds(c);
        if !t.cosmos.may_hold(lo, hi) {
            empty = c;
            break;
        }
    }
    let n = 30;
    let start = std::time::Instant::now();
    for _ in 0..n {
        let _ = t.generate(cx, cy, cz);
    }
    let rock_us = start.elapsed().as_secs_f64() * 1e6 / n as f64;
    let start = std::time::Instant::now();
    for _ in 0..n {
        let _ = t.generate(empty.x, empty.y, empty.z);
    }
    let empty_us = start.elapsed().as_secs_f64() * 1e6 / n as f64;
    println!(
        "cluster chunk ({cx},{cy},{cz}) kind {:?} r {:.0}: {rock_us:.1} µs; empty space ({}, {}, {}): {empty_us:.1} µs",
        small.kind, small.r, empty.x, empty.y, empty.z
    );
}
