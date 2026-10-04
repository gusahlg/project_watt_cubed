//! Generator contract and character checks.

use super::*;
use super::cube;
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

    // Above every landmark on +Y. TREE_CLEAR rose so sky islands are still painted;
    // world y=496 (chunk 31) is no longer above them.
    let above = (cube::TREE_CLEAR + 15) / 16;
    assert_eq!(t.classify(ChunkCoord::new(0, above, 0)), Classify::Uniform(AIR));
    assert_eq!(t.generate(0, above, 0).uniform(), Some(AIR));
    assert_worker_matches(&t, 0, above, 0);

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
        ChunkCoord::new(0, (cube::TREE_CLEAR + 15) / 16, 0),
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

/// World cell of face-local `(u, h, v)`. `h` is altitude above the face plane.
fn face_world(body: &cosmos::Body, face: Face, u: i32, h: i32, v: i32) -> (i32, i32, i32) {
    let half = cube::half_of(body);
    let a = i32::try_from(half + i64::from(h)).expect("altitude");
    let (x, y, z) = FaceFrame::new(face).cell_to_world((u, a, v));
    (
        i32::try_from(i64::from(x) + body.centre[0]).expect("x"),
        i32::try_from(i64::from(y) + body.centre[1]).expect("y"),
        i32::try_from(i64::from(z) + body.centre[2]).expect("z"),
    )
}

fn agree_cell(t: &Terrain, body: &cosmos::Body, face: Face, u: i32, h: i32, v: i32) {
    let (x, y, z) = face_world(body, face, u, h, v);
    let coord = ChunkCoord::new(x.div_euclid(16), y.div_euclid(16), z.div_euclid(16));
    assert_eq!(t.sky(coord), Sky::Axis(face), "landmark chunk {coord:?} on {face:?} uses the batch path");
    assert_chunk_matches(t, coord.x, coord.y, coord.z);
}

struct ThemeHit {
    theme: province::ThemeId,
    weight: f32,
    u: i32,
    v: i32,
}

/// Deepest sample of each theme on one face, inside the inland band.
fn survey(t: &Terrain, body: &cosmos::Body, face: Face) -> Vec<ThemeHit> {
    let paint = t.paint(body, face);
    let mut hits: Vec<ThemeHit> = Vec::new();
    for i in 0..36 {
        for j in 0..36 {
            let u = 4_000 + i * 1_200;
            let v = 4_000 + j * 1_200;
            if paint.shape.inset(u, v) < 20_000 {
                continue;
            }
            let p = paint.shape.place(u, v);
            if let Some(hit) = hits.iter_mut().find(|h| h.theme == p.theme) {
                if p.weight > hit.weight {
                    *hit = ThemeHit { theme: p.theme, weight: p.weight, u, v };
                }
            } else {
                hits.push(ThemeHit { theme: p.theme, weight: p.weight, u, v });
            }
        }
    }
    hits
}

fn origin(hits: &[ThemeHit], theme: province::ThemeId) -> (i32, i32) {
    let hit = hits.iter().find(|h| h.theme == theme).unwrap_or_else(|| panic!("no {theme:?}"));
    assert!(hit.weight >= 0.5, "{theme:?} weight {}", hit.weight);
    (hit.u, hit.v)
}

fn probe(
    paint: &FacePaint,
    origin: (i32, i32),
    span: i32,
    step: i32,
    dys: &[i32],
    mut pred: impl FnMut(&Column, i32, Option<features::Stamp>) -> bool,
) -> Option<(i32, i32, i32)> {
    let (u0, v0) = origin;
    let mut v = v0 - span;
    while v < v0 + span {
        let mut u = u0 - span;
        while u < u0 + span {
            let col = paint.shape.column(u, v);
            for &dy in dys {
                let y = col.height + dy;
                let st = paint.features.block_at(&paint.shape, u, y, v, col.height);
                if pred(&col, y, st) {
                    return Some((u, v, y));
                }
            }
            u += step;
        }
        v += step;
    }
    None
}

fn must(
    paint: &FacePaint,
    origin: (i32, i32),
    span: i32,
    step: i32,
    dys: &[i32],
    what: &str,
    pred: impl FnMut(&Column, i32, Option<features::Stamp>) -> bool,
) -> (i32, i32, i32) {
    probe(paint, origin, span, step, dys, pred).unwrap_or_else(|| panic!("no {what} near {origin:?}"))
}

#[test]
fn landmarks_stay_under_relief_and_ground_under_the_lod_ceiling() {
    assert_eq!(cube::TREE_CLEAR, MAX_GROUND + features::MAX_ABOVE + 1);
    assert!(features::MAX_BELOW < cube::CRUST);
    assert!(i64::from(MAX_GROUND + features::MAX_ABOVE) < cosmos::RELIEF);
    // The far field is ground only: landmarks may rise past its window, the ground may not.
    assert!(MAX_GROUND < crate::world::section::LOD_CEIL_Y);
}

#[test]
fn features_knob_at_zero_plants_nothing() {
    let (_reg, on) = make(42);
    let mut reg = BlockRegistry::with_builtins();
    let mut cfg = TerrainCfg::default();
    cfg.features = 0;
    let off = Terrain::with_cfg(&mut reg, 42, cfg);
    let home = *on.cosmos.home();
    let mut planted = false;
    for z in -40..40 {
        for x in -40..40 {
            let col = on.paint(&home, Face::PosY).shape.column(x, z);
            let st = on.paint(&home, Face::PosY).features.block_at(&on.paint(&home, Face::PosY).shape, x, col.height, z, col.height);
            if let Some(st) = st {
                if st.id == on.materials().meadow || st.id == on.materials().lichen || st.id == on.materials().darkwood {
                    planted = true;
                }
            }
            let quiet = off.paint(&home, Face::PosY);
            for y in col.height - 2..col.height + 6 {
                assert!(
                    quiet.features.block_at(&quiet.shape, x, y, z, col.height).is_none(),
                    "features=0 still painted ({x},{y},{z})"
                );
            }
        }
    }
    assert!(planted, "default features knob planted no flora near spawn");
    let again = on.paint(&home, Face::PosY).features.block_at(
        &on.paint(&home, Face::PosY).shape,
        0,
        on.paint(&home, Face::PosY).shape.height(0, 0),
        0,
        on.paint(&home, Face::PosY).shape.height(0, 0),
    );
    let twice = on.paint(&home, Face::PosY).features.block_at(
        &on.paint(&home, Face::PosY).shape,
        0,
        on.paint(&home, Face::PosY).shape.height(0, 0),
        0,
        on.paint(&home, Face::PosY).shape.height(0, 0),
    );
    assert_eq!(again, twice);
}

#[test]
fn landmarks_skip_the_cube_edge_band() {
    let (_reg, t) = make(42);
    let m = t.materials();
    let x0 = cosmos::HOME_HALF as i32 - 80;
    let allowed = [
        AIR, m.timber, m.leaves, m.pine, m.autumn, m.blossom, m.flower_red, m.flower_yellow, m.flower_blue,
        m.flower_white,
    ];
    for z in (-24..24).step_by(2) {
        let h = t.height(x0, z);
        for y in h..h + 40 {
            let id = t.voxel_at(x0, y, z);
            assert!(allowed.contains(&id), "edge cell ({x0},{y},{z}) is not a tree or flower");
        }
    }
}

#[test]
fn every_landmark_family_matches_on_its_face() {
    let (_reg, t) = make(42);
    let home = *t.cosmos.home();
    let m = t.materials();
    let py = survey(&t, &home, Face::PosY);
    let ny = survey(&t, &home, Face::NegY);
    let px = survey(&t, &home, Face::PosX);
    let nx = survey(&t, &home, Face::NegX);
    let pz = survey(&t, &home, Face::PosZ);
    let nz = survey(&t, &home, Face::NegZ);

    let stone = |id: BlockId| {
        id == m.limestone || id == m.slate || id == m.sandstone[0] || id == m.sandstone[1] || id == m.sandstone[2]
            || id == m.sandstone[3]
    };
    let wood = |id: BlockId| id == m.bark || id == m.darkwood;
    let cap = |id: BlockId| id == m.cap_red || id == m.cap_brown || id == m.glowshroom;
    let crystal = |id: BlockId| id == m.crystal || id == m.violet || id == m.glowcap || id == m.glowshroom;
    let ice = |id: BlockId| id == m.ice || id == m.frost || id == m.snow;
    let fire = |id: BlockId| id == m.cinder || id == m.ash || id == m.magma || id == m.basalt || id == m.obsidian;

    // Giant trees, including the occasional broadleaf one.
    for (face, hits, theme, span) in [
        (Face::PosY, &py, province::ThemeId::Giant, 48),
        (Face::PosY, &py, province::ThemeId::Broadleaf, 96),
    ] {
        let paint = t.paint(&home, face);
        let (u, v, y) = must(paint, origin(hits, theme), span, 1, &[12, 24, 40], "giant trunk", |col, y, st| {
            y >= col.height + 8 && st.is_some_and(|s| s.id == m.darkwood)
        });
        let col = paint.shape.column(u, v);
        let mut run = 0;
        let mut yy = col.height;
        while yy < col.height + 80 {
            let st = paint.features.block_at(&paint.shape, u, yy, v, col.height);
            if st.is_some_and(|s| wood(s.id)) {
                run += 1;
                yy += 1;
            } else {
                break;
            }
        }
        assert!((30..=70).contains(&run), "{theme:?} trunk {run} at ({u},{v})");
        let mut leaf = false;
        for dz in -4..5 {
            for dx in -4..5 {
                for dy in (run - 2)..(run + 5) {
                    let st = paint.features.block_at(&paint.shape, u + dx, col.height + dy, v + dz, col.height);
                    if st.is_some_and(|s| s.id == m.leaves || s.id == m.autumn || s.id == m.blossom) {
                        leaf = true;
                    }
                }
            }
        }
        assert!(leaf, "{theme:?} trunk at ({u},{v}) has no crown");
        agree_cell(&t, &home, face, u, y, v);
        agree_cell(&t, &home, face, u, col.height + run - 1, v);
    }

    // Mushrooms on the fungal face.
    {
        let paint = t.paint(&home, Face::NegZ);
        let (u, v, y) = must(paint, origin(&nz, province::ThemeId::Fungal), 64, 1, &[4, 10, 20], "mushroom", |col, y, st| {
            y > col.height && st.is_some_and(|s| s.id == m.stem)
        });
        let col = paint.shape.column(u, v);
        let mut found_cap = false;
        for dy in 0..48 {
            let st = paint.features.block_at(&paint.shape, u, col.height + dy, v, col.height);
            if st.is_some_and(|s| cap(s.id)) {
                found_cap = true;
            }
        }
        assert!(found_cap, "mushroom stem without a cap at ({u},{v})");
        agree_cell(&t, &home, Face::NegZ, u, y, v);
        agree_cell(&t, &home, Face::NegZ, u, col.height + 16, v);
    }

    // Spires on karst.
    {
        let paint = t.paint(&home, Face::NegX);
        let (u, v, y) = must(paint, origin(&nx, province::ThemeId::Karst), 64, 1, &[8, 16, 28], "spire", |col, y, st| {
            y > col.height + 6 && st.is_some_and(|s| stone(s.id))
        });
        agree_cell(&t, &home, Face::NegX, u, y, v);
    }

    // Volcanic cones and ash fissures.
    {
        let paint = t.paint(&home, Face::NegY);
        let (u, v, y) = must(paint, origin(&ny, province::ThemeId::Volcanic), 80, 2, &[6, 16, 40], "volcano", |col, y, st| {
            y >= col.height && st.is_some_and(|s| fire(s.id))
        });
        agree_cell(&t, &home, Face::NegY, u, y, v);
    }

    // Crystals on the glass face.
    {
        let paint = t.paint(&home, Face::PosZ);
        let (u, v, y) = must(paint, origin(&pz, province::ThemeId::Crystal), 48, 1, &[4, 12, 22], "crystal", |col, y, st| {
            y > col.height && st.is_some_and(|s| crystal(s.id))
        });
        let col = paint.shape.column(u, v);
        let mut run = 0;
        for dz in -2..3 {
            for dx in -2..3 {
                let mut n = 0;
                for dy in 1..36 {
                    let st = paint.features.block_at(&paint.shape, u + dx, col.height + dy, v + dz, col.height);
                    if st.is_some_and(|s| crystal(s.id)) {
                        n += 1;
                    }
                }
                run = run.max(n);
            }
        }
        assert!((4..=30).contains(&run), "crystal run {run}");
        agree_cell(&t, &home, Face::PosZ, u, y, v);
    }

    // Ice on a glacier.
    {
        let paint = t.paint(&home, Face::PosZ);
        let (u, v, y) = must(paint, origin(&pz, province::ThemeId::Glacier), 56, 1, &[6, 14, -4, -10], "ice", |col, y, st| {
            let Some(st) = st else { return false };
            (y > col.height && ice(st.id)) || (y < col.height && st.dig && st.id == AIR)
        });
        agree_cell(&t, &home, Face::PosZ, u, y, v);
    }

    // Sand ripples and mesa buttes.
    {
        let paint = t.paint(&home, Face::PosX);
        let (u, v, y) = must(paint, origin(&px, province::ThemeId::Dune), 40, 1, &[0], "ripple", |col, y, st| {
            y == col.height && st.is_some_and(|s| s.id == m.sand)
        });
        agree_cell(&t, &home, Face::PosX, u, y, v);
        let (u, v, y) = must(paint, origin(&px, province::ThemeId::Mesa), 48, 1, &[4, 10], "butte", |col, y, st| {
            y > col.height && st.is_some_and(|s| s.id == m.redsand)
        });
        agree_cell(&t, &home, Face::PosX, u, y, v);
    }

    // Sky islands, well above the ground.
    {
        let paint = t.paint(&home, Face::NegX);
        let mut high = Vec::new();
        let mut d = 100;
        while d <= 312 {
            high.push(d);
            d += 4;
        }
        let (u, v, y) = must(paint, origin(&nx, province::ThemeId::Islands), 180, 6, &high, "sky island", |col, y, st| {
            y >= col.height + 100 && st.is_some_and(|s| s.id == m.grass || s.id == m.rock[0] || s.id == m.rock[1] || s.id == m.soil)
        });
        let col = paint.shape.column(u, v);
        assert!((100..=320).contains(&(y - col.height)), "island altitude {}", y - col.height);
        agree_cell(&t, &home, Face::NegX, u, y, v);
    }

    // Impact craters.
    {
        let paint = t.paint(&home, Face::NegY);
        let (u, v, y) = must(
            paint,
            origin(&ny, province::ThemeId::Crater),
            200,
            4,
            &[0, 2, 4, -6, -12],
            "crater",
            |col, y, st| {
                let Some(st) = st else { return false };
                (y >= col.height && (st.id == m.regolith || st.id == m.gravel))
                    || (y < col.height && st.dig && (st.id == m.gold || st.id == m.copper || st.id == m.basalt))
            },
        );
        agree_cell(&t, &home, Face::NegY, u, y, v);
    }

    // Bone, petrified trunks, karst sinkholes.
    {
        let paint = t.paint(&home, Face::NegZ);
        let (u, v, y) = must(paint, origin(&nz, province::ThemeId::Bone), 56, 1, &[2, 6, 10], "bone", |col, y, st| {
            y >= col.height && st.is_some_and(|s| s.id == m.bone)
        });
        agree_cell(&t, &home, Face::NegZ, u, y, v);
    }
    {
        let paint = t.paint(&home, Face::NegX);
        let (u, v, y) = must(paint, origin(&nx, province::ThemeId::Petrified), 48, 1, &[1, 4, 8], "petrified", |col, y, st| {
            y >= col.height && st.is_some_and(|s| s.id == m.petrified)
        });
        agree_cell(&t, &home, Face::NegX, u, y, v);
        let (u, v, y) = must(paint, origin(&nx, province::ThemeId::Karst), 80, 1, &[-8, -16], "sinkhole", |col, y, st| {
            col.strata == province::Strata::Limestone && y < col.height - 4 && st.is_some_and(|s| s.dig && s.id == AIR)
        });
        agree_cell(&t, &home, Face::NegX, u, y, v);
    }

    // A lush twin grows the same giants, on a face the other twin does not cover.
    let twins: Vec<_> = t.cosmos.bodies().iter().copied().filter(|b| b.kind == cosmos::Kind::Twin).collect();
    let lush = *twins.iter().min_by_key(|b| (b.seed, b.id)).unwrap();
    let other = *twins.iter().find(|b| b.id != lush.id).unwrap();
    let axis = (0..3).max_by_key(|&a| (lush.centre[a] - other.centre[a]).abs()).unwrap();
    let away = if lush.centre[axis] < other.centre[axis] {
        [Face::NegX, Face::NegY, Face::NegZ][axis]
    } else {
        [Face::PosX, Face::PosY, Face::PosZ][axis]
    };
    let hits = survey(&t, &lush, away);
    let paint = t.paint(&lush, away);
    let (u, v, y) = must(paint, origin(&hits, province::ThemeId::Giant), 64, 1, &[12, 28], "twin giant", |col, y, st| {
        y >= col.height + 8 && st.is_some_and(|s| s.id == m.darkwood)
    });
    agree_cell(&t, &lush, away, u, y, v);
}
