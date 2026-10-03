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
        if b.kind == cosmos::Kind::Moon {
            println!("moon {} centre {:?} r {:?}", b.id, b.centre, b.shape);
        }
    }
}
