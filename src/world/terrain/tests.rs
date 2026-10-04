//! Generator contract and character checks.

use super::*;
use super::cube;
use super::{space, span};
use super::deep;
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
    let mut near: Vec<_> = t.cosmos.clusters().iter().collect();
    near.sort_by(|a, b| a.centre.length().total_cmp(&b.centre.length()));
    for c in near.iter().take(5) {
        println!("cluster {:?} centre {:?} r {:.0} rocks {:.0}", c.form, c.centre, c.radius, c.count);
        // The biggest rocks within 3 000 blocks of its centre.
        let (p, d) = ([c.centre.x as i64, c.centre.y as i64, c.centre.z as i64], 3_000i64);
        let mut rocks = Vec::new();
        t.cosmos.rocks_touching([p[0] - d, p[1] - d, p[2] - d], [p[0] + d, p[1] + d, p[2] + d], &mut rocks);
        rocks.sort_by(|a, b| b.r.total_cmp(&a.r));
        for r in rocks.iter().take(3) {
            println!("  rock {:?} r {:.0} at {:?}", r.kind, r.r, r.centre);
        }
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
        // Above the spires and every landmark (sky islands reach higher than the spires).
        let clear = face_world(body, face, 0, span::CLEAR.max(cube::TREE_CLEAR) + 16, 0);
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

/// Classify cost of a view-sized box of space chunks inside a big swarm (release:
/// `cargo test --release --lib cluster_classify_cost -- --ignored --nocapture`).
#[test]
#[ignore]
fn cluster_classify_cost() {
    let (_reg, t) = make(42);
    let eye = [27310502i64, 49569080, -35460644];
    let c = eye.map(|v| v.div_euclid(16) as i32);
    let mut n = 0;
    let mut mixed = 0;
    let start = std::time::Instant::now();
    for dx in -12..=12 {
        for dy in -6..=6 {
            for dz in -12..=12 {
                n += 1;
                mixed += (t.classify(ChunkCoord::new(c[0] + dx, c[1] + dy, c[2] + dz)) == Classify::Mixed) as usize;
            }
        }
    }
    let per = start.elapsed().as_secs_f64() * 1e6 / n as f64;
    println!("{per:.1} µs per classify over {n} chunks ({mixed} mixed)");
}

fn world_of(body: &cosmos::Body, rel: [i64; 3]) -> (i32, i32, i32) {
    (
        i32::try_from(body.centre[0] + rel[0]).unwrap(),
        i32::try_from(body.centre[1] + rel[1]).unwrap(),
        i32::try_from(body.centre[2] + rel[2]).unwrap(),
    )
}

fn chunk_matches_at(t: &Terrain, body: &cosmos::Body, rel: [i64; 3]) {
    let (x, y, z) = world_of(body, rel);
    assert_chunk_matches(t, x.div_euclid(16), y.div_euclid(16), z.div_euclid(16));
}

/// An air cell near `at` (features dress the shell, so the middle of a hollow is open).
fn find_air(t: &Terrain, body: &cosmos::Body, at: [i64; 3]) -> [i64; 3] {
    for dz in -2..=2 {
        for dy in -2..=2 {
            for dx in -2..=2 {
                let rel = [at[0] + dx * 8, at[1] + dy * 8, at[2] + dz * 8];
                let (x, y, z) = world_of(body, rel);
                if t.voxel_at(x, y, z) == AIR {
                    return rel;
                }
            }
        }
    }
    panic!("no air near {at:?}");
}

#[test]
fn the_interior_is_batch_exact_and_the_heart_stays_when_deep_is_off() {
    let (_reg, t) = make(42);
    let home = *t.cosmos.home();
    let cavern = t.deep.locate_cavern(&home).expect("a deep cavern");
    let air = find_air(&t, &home, cavern);
    let (x, y, z) = world_of(&home, air);
    assert_eq!(t.voxel_at(x, y, z), AIR, "cavern air at {air:?}");
    chunk_matches_at(&t, &home, cavern);

    let hall = t.deep.locate_hall(&home).expect("a dwarf hall");
    chunk_matches_at(&t, &home, hall);
    let (hx, hy, hz) = world_of(&home, hall);
    assert_eq!(t.voxel_at(hx, hy, hz), AIR, "hall centre is the room");

    let chamber = t.deep.locate_chamber(&home).expect("an underdark chamber");
    let chamber_air = find_air(&t, &home, chamber);
    chunk_matches_at(&t, &home, chamber);
    let (cx, cy, cz) = world_of(&home, chamber_air);
    assert_eq!(t.voxel_at(cx, cy, cz), AIR, "chamber air");

    let (bubble, r) = t.deep.locate_bubble(&home).expect("a mantle bubble");
    let mut inside = bubble;
    inside[0] += r / 2;
    let (bx, by, bz) = world_of(&home, inside);
    assert_eq!(t.voxel_at(bx, by, bz), AIR, "bubble interior r={r}");
    chunk_matches_at(&t, &home, inside);
    let mut bulk_outside = false;
    for axis in 0..3 {
        for sign in [-1i64, 1] {
            let mut rel = bubble;
            rel[axis] += sign * (r + 3);
            let (ox, oy, oz) = world_of(&home, rel);
            if t.voxel_at(ox, oy, oz) == cube::bulk_id(&t.bulk, &home, rel) {
                bulk_outside = true;
            }
        }
    }
    assert!(bulk_outside, "a block just outside the bubble is still bulk");

    // The core, a floating cell of the Heart, and a face-centre shaft.
    let (kx, ky, kz) = world_of(&home, [0, 0, 0]);
    assert_eq!(t.voxel_at(kx, ky, kz), t.materials().core);
    chunk_matches_at(&t, &home, [0, 0, 0]);
    let shaft = [0, 50_100, 0];
    let (sx, sy, sz) = world_of(&home, shaft);
    assert_eq!(t.voxel_at(sx, sy, sz), AIR, "heart shaft");
    let beside = [20, 50_100, 0];
    let (px, py, pz) = world_of(&home, beside);
    assert_eq!(t.voxel_at(px, py, pz), cube::bulk_id(&t.bulk, &home, beside));
    chunk_matches_at(&t, &home, shaft);

    // `deep = 0` removes the density-scaled hollows. The Heart does not scale.
    let mut reg = BlockRegistry::with_builtins();
    let off = Terrain::with_cfg(&mut reg, 42, TerrainCfg { deep: 0, ..TerrainCfg::default() });
    assert_ne!(off.voxel_at(x, y, z), AIR, "deep=0 left the cavern hollow");
    assert_eq!(off.voxel_at(x, y, z), cube::bulk_id(&off.bulk, &home, air));
    let (hx0, hy0, hz0) = world_of(&home, [0, 1_000, 0]);
    assert_eq!(off.voxel_at(hx0, hy0, hz0), AIR, "the Heart stays hollow at deep=0");
    assert_eq!(off.voxel_at(kx, ky, kz), off.materials().core);
}

#[test]
fn interior_porosity_barely_moves_spawn_gravity() {
    use crate::gravity::{Field, Primitive, Shape};
    use crate::math::BLOCK_METERS;
    let cosmos = std::sync::Arc::new(cosmos::Cosmos::with_deep(7, 1.0, 1.0));
    let field = Field::new(cosmos);
    let at = glam::DVec3::new(0.5, 70.0, 0.5);
    let after = field.sample(at).accel.length() * BLOCK_METERS;
    let c = cosmos::HOME_CENTRE;
    let h = cosmos::HOME_HALF as f64;
    let solid = Primitive::new(
        Shape::Box {
            lo: glam::DVec3::new(c[0] as f64 - h, c[1] as f64 - h, c[2] as f64 - h),
            hi: glam::DVec3::new(c[0] as f64 + h, c[1] as f64 + h, c[2] as f64 + h),
        },
        cosmos::BULK_DENSITY,
    );
    let before = (solid.field(at).0 * crate::gravity::G).length() * BLOCK_METERS;
    let (phi_d, phi_u) = deep::porosity(1.0);
    println!("spawn gravity before {before:.6} m/s² after {after:.6} m/s² (deep porosity {phi_d:.6}, under {phi_u:.6})");
    assert!((before - 24.0).abs() < 0.01 * 24.0, "solid spawn {before}");
    assert!((after - 24.0).abs() < 0.01 * 24.0, "interior spawn {after}");
    assert!((after - before).abs() < 0.01 * 24.0, "voids moved spawn by {}", after - before);

    let centre = glam::DVec3::new(c[0] as f64, c[1] as f64, c[2] as f64);
    let centre_pull = field.sample(centre).accel.length() * BLOCK_METERS;
    println!("planet centre pull {centre_pull:.6} m/s²");
    assert!(centre_pull < 0.05, "centre is not weightless: {centre_pull}");

    // Inside a bubble the removed ball cancels the cube's linear gradient.
    let (_reg, t) = make(42);
    let home = *t.cosmos.home();
    let (rel, r) = t.deep.locate_bubble(&home).expect("bubble");
    let p0 = glam::DVec3::new(
        (home.centre[0] + rel[0]) as f64,
        (home.centre[1] + rel[1]) as f64,
        (home.centre[2] + rel[2]) as f64,
    );
    // A negative ball cancels the cube's divergence inside the cavity, so the field there is the
    // cube's tide: nearly constant across a bubble that is small next to the planet.
    let carved = Field::new(t.mass());
    let g0 = carved.sample(p0).accel;
    let mut trace_void = 0.0;
    let mut trace_solid = 0.0;
    let mut worst = 0.0f64;
    for axis in 0..3 {
        let mut step = glam::DVec3::ZERO;
        step[axis] = 0.4 * r as f64;
        let p1 = p0 + step;
        let d_void = carved.sample(p1).accel - g0;
        let d_solid = (solid.field(p1).0 - solid.field(p0).0) * crate::gravity::G;
        trace_void += d_void[axis];
        trace_solid += d_solid[axis];
        worst = worst.max(d_void.length());
    }
    let g_ms = g0.length() * BLOCK_METERS;
    let worst_ms = worst * BLOCK_METERS;
    println!(
        "bubble r={r} field {g_ms:.4} m/s², tide across 0.4r {worst_ms:.4} m/s² (div void {trace_void:.6} solid {trace_solid:.6})"
    );
    assert!(trace_solid.abs() > 1.0e-4, "solid divergence {trace_solid}");
    assert!(trace_void.abs() < trace_solid.abs() * 0.05, "cavity divergence {trace_void} vs {trace_solid}");
    assert!(worst_ms < 0.05, "bubble tide {worst_ms} m/s²");
}

#[test]
fn quiet_deep_chunks_stay_uniform() {
    let (_reg, t) = make(42);
    let home = *t.cosmos.home();
    let half = cube::half_of(&home);
    let mut uniform = 0u32;
    let mut n = 0u32;
    for k in 0..12 {
        let depth = 500 + (half - 80_000) * k / 11;
        let rel_y = half - depth;
        for (i, xz) in [64i64, 200, 1_500, 8_000, 40_000, 200_000, 1_000_000, 4_000_000].into_iter().enumerate() {
            if xz + 32 >= rel_y {
                continue;
            }
            let rel = [xz, rel_y, xz / 3 + 20 + i as i64 * 17];
            if !cube::in_deep(rel, half) {
                continue;
            }
            let (x, y, z) = world_of(&home, rel);
            let class = t.classify(ChunkCoord::new(x.div_euclid(16), y.div_euclid(16), z.div_euclid(16)));
            n += 1;
            if matches!(class, Classify::Uniform(_)) {
                uniform += 1;
            }
        }
    }
    let rate = f64::from(uniform) / f64::from(n);
    println!("interior classify uniform {uniform}/{n} = {rate:.4}");
    assert!(n >= 40, "sampled {n} deep chunks");
    assert!(rate >= 0.95, "uniform hit rate {rate} ({uniform}/{n})");
}

/// World cell of face-local `(u, h, v)`. `h` is altitude above the face plane.
fn face_cell(body: &cosmos::Body, face: Face, u: i32, h: i32, v: i32) -> (i32, i32, i32) {
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
    let (x, y, z) = face_cell(body, face, u, h, v);
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

fn structure_stamp(paint: &FacePaint, x: i32, y: i32, z: i32, ground: i32) -> Option<features::Stamp> {
    paint.structures.block_at(&paint.shape, &paint.under, x, y, z, ground)
}

#[test]
fn structures_knob_at_zero_builds_nothing() {
    let mut reg = BlockRegistry::with_builtins();
    let mut cfg = TerrainCfg::default();
    cfg.structures = 0;
    let off = Terrain::with_cfg(&mut reg, 42, cfg);
    let home = *off.cosmos.home();
    let paint = off.paint(&home, Face::PosY);
    assert!(structures::survey(&paint.structures, &paint.shape, &paint.under, 0, 0, 2).is_empty());
    assert!(structures::survey_roads(&paint.structures, &paint.shape, &paint.under, 0, 0, 4).is_empty());
    for z in (-32..32).step_by(8) {
        for x in (-32..32).step_by(8) {
            let h = paint.shape.column(x, z).height;
            assert!(structure_stamp(paint, x, h, z, h).is_none(), "structures=0 painted ({x},{z})");
            assert!(!paint.structures.owns(&paint.shape, &paint.under, x, z));
        }
    }
}

#[test]
fn structures_own_their_footprint_and_match_the_batch() {
    assert!(structures::DIG_LIMIT < cube::CRUST);
    assert!(structures::DIG_LIMIT > features::MAX_BELOW);
    let (_reg, t) = make(42);
    let home = *t.cosmos.home();
    let m = t.materials();
    let paint = t.paint(&home, Face::PosY);
    let found = structures::survey(&paint.structures, &paint.shape, &paint.under, 0, 0, 3);
    let mut counts = [0u32; 9];
    for f in &found {
        counts[f.kind as usize] += 1;
        let col = paint.shape.column(f.x, f.z);
        assert!(paint.shape.inset(f.x, f.z) >= i64::from(cube::RIM), "structure on the rim");
        if f.kind != structures::KIND_BRIDGE {
            assert!(col.slope4 < 8, "kind {} slope {}", f.kind, col.slope4);
        }
        let once = structure_stamp(paint, f.x, f.pad, f.z, col.height);
        let twice = structure_stamp(paint, f.x, f.pad, f.z, col.height);
        assert_eq!(once, twice);
    }
    let names = ["tower", "pyramid", "circle", "monolith", "temple", "observatory", "mine"];
    for (i, name) in names.iter().enumerate() {
        assert!(counts[i] > 0, "no {name} near spawn {counts:?}");
    }

    let stone = |id: BlockId| {
        id == m.limestone || id == m.slate || id == m.marble || id == m.obsidian || id == m.plank
            || m.sandstone.contains(&id)
    };
    let tower = found.iter().find(|f| f.kind == structures::KIND_TOWER).unwrap();
    assert!((15..=60).contains(&tower.a), "tower height {}", tower.a);
    assert!((3..=5).contains(&tower.b));
    let tg = paint.shape.column(tower.x, tower.z).height;
    assert!(structure_stamp(paint, tower.x, tower.pad + 2, tower.z, tg).is_some_and(|s| s.id == AIR && s.dig));
    assert!(stone(structure_stamp(paint, tower.x + tower.b, tower.pad + 4, tower.z, tg).unwrap().id));
    assert!(structure_stamp(paint, tower.x + tower.b + 5, tower.pad, tower.z, tg).is_none());

    let pyramid = found.iter().find(|f| f.kind == structures::KIND_PYRAMID).unwrap();
    assert!((4..=9).contains(&pyramid.a));
    let pg = paint.shape.column(pyramid.x, pyramid.z).height;
    assert_eq!(structure_stamp(paint, pyramid.x, pyramid.pad + 1, pyramid.z, pg).unwrap().id, m.lamp);
    assert!(structure_stamp(paint, pyramid.x + pyramid.a * 2 + 3, pyramid.pad, pyramid.z, pg).is_none());

    let circle = found.iter().find(|f| f.kind == structures::KIND_CIRCLE).unwrap();
    assert!(matches!(circle.a, 8 | 10 | 12));
    assert!((3..=5).contains(&circle.b));
    let cg = paint.shape.column(circle.x, circle.z).height;
    assert!(stone(structure_stamp(paint, circle.x, circle.pad, circle.z, cg).unwrap().id));

    let mono = found.iter().find(|f| f.kind == structures::KIND_MONOLITH).unwrap();
    assert!(matches!(mono.a, 8 | 12 | 16 | 20 | 24));
    let mg = paint.shape.column(mono.x, mono.z).height;
    let mid = structure_stamp(paint, mono.x, mono.pad, mono.z, mg).unwrap().id;
    assert!(mid == m.obsidian || mid == m.marble, "monolith {mid:?}");

    let temple = found.iter().find(|f| f.kind == structures::KIND_TEMPLE).unwrap();
    assert!((6..=10).contains(&temple.a));
    assert!((5..=7).contains(&temple.b));
    let eg = paint.shape.column(temple.x + temple.a, temple.z + 2).height;
    assert!(stone(structure_stamp(paint, temple.x + temple.a, temple.pad + 1, temple.z + 2, eg).unwrap().id));

    let hut = found.iter().find(|f| f.kind == structures::KIND_OBSERVATORY).unwrap();
    assert!((10..=14).contains(&hut.a));
    assert!(matches!(hut.b, 4 | 5));
    assert_eq!(
        (hut.qu, hut.qa, hut.qv),
        structures::aim_at(&paint.structures, hut.x, hut.pad, hut.z, hut.a)
    );
    let span = hut.a.max(1);
    let (fx, fy, fz) = (hut.x + hut.qu * hut.a / span, hut.pad + hut.b + hut.qa * hut.a / span, hut.z + hut.qv * hut.a / span);
    let fg = paint.shape.column(fx, fz).height;
    let frame = structure_stamp(paint, fx, fy, fz, fg).unwrap().id;
    assert!(frame == m.plank || frame == m.rust, "telescope {frame:?}");

    let mut connected = 0;
    for mine in found.iter().filter(|f| f.kind == structures::KIND_MINE) {
        assert!(mine.pad - mine.a < cube::CRUST);
        assert!(mine.a <= mine.pad - 6);
        let g = paint.shape.column(mine.x, mine.z).height;
        assert!(structure_stamp(paint, mine.x, mine.pad, mine.z, g).is_some_and(|s| s.id == AIR && s.dig));
        assert_eq!(structure_stamp(paint, mine.x + 2, mine.pad + 4, mine.z + 2, g).unwrap().id, m.plank);
        if mine.a < mine.pad - 6 {
            connected += 1;
            assert!(structure_stamp(paint, mine.x, mine.a, mine.z, g).is_some_and(|s| s.id == AIR && s.dig));
            assert!(structure_stamp(paint, mine.x, mine.a - 1, mine.z, g).is_none(), "shaft dug the rail");
        }
    }
    assert!(connected > 0, "no shaft met a mine ({counts:?})");

    let mut checked = 0;
    for f in &found {
        if checked >= 3 {
            break;
        }
        agree_cell(&t, &home, Face::PosY, f.x, f.pad, f.z);
        let (wx, wy, wz) = face_cell(&home, Face::PosY, f.x, f.pad, f.z);
        let id = t.voxel_at(wx, wy, wz);
        let g = paint.shape.column(f.x, f.z).height;
        if let Some(st) = structure_stamp(paint, f.x, f.pad, f.z, g) {
            if st.id != AIR {
                assert_eq!(id, st.id, "voxel ignored structure at ({},{})", f.x, f.z);
            }
        }
        checked += 1;
    }

    let roads = structures::survey_roads(&paint.structures, &paint.shape, &paint.under, tower.x, tower.z, 8);
    assert!(!roads.is_empty(), "no waystone near a tower");
    let road = &roads[0];
    let rg = paint.shape.column(road.x, road.z).height;
    let base = structure_stamp(paint, road.x, road.pad, road.z, rg).unwrap().id;
    let cap = structure_stamp(paint, road.x, road.pad + 1, road.z, rg).unwrap().id;
    assert!(stone(base) || base == m.obsidian || base == m.marble);
    assert!(cap == m.obsidian || cap == m.marble);
    assert_ne!(base, cap);

    let twin = t.cosmos.bodies().iter().find(|b| b.kind == cosmos::Kind::Twin).copied().unwrap();
    let face = span::facing_face(&t.cosmos, &twin).unwrap();
    let facing = t.paint(&twin, face);
    assert!(structures::survey(&facing.structures, &facing.shape, &facing.under, 0, 0, 2).is_empty());
}

#[test]
fn canyon_bridges_span_both_rims() {
    let (_reg, t) = make(42);
    let home = *t.cosmos.home();
    let m = t.materials();
    let hits = survey(&t, &home, Face::PosX);
    let (u, v) = origin(&hits, province::ThemeId::Canyon);
    let paint = t.paint(&home, Face::PosX);
    let found = structures::survey(&paint.structures, &paint.shape, &paint.under, u, v, 2);
    let bridge = found.iter().find(|f| f.kind == structures::KIND_BRIDGE).unwrap_or_else(|| {
        panic!("no bridge near canyon ({u},{v}), {} sites", found.len())
    });
    let g = paint.shape.column(bridge.x, bridge.z).height;
    if bridge.qa == 0 {
        assert_eq!(structure_stamp(paint, bridge.x, bridge.pad, bridge.z, g).unwrap().id, m.plank);
        assert!(structure_stamp(paint, bridge.x, bridge.pad - 2, bridge.z, g).is_none());
    } else {
        let id = structure_stamp(paint, bridge.x, bridge.pad + bridge.qv, bridge.z, g).unwrap().id;
        assert!(id == m.limestone || id == m.slate || id == m.marble || id == m.obsidian || m.sandstone.contains(&id) || id == m.plank);
    }
    agree_cell(&t, &home, Face::PosX, bridge.x, bridge.pad, bridge.z);
}
