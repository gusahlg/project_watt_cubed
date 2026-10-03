//! Generator contract and character checks.

use super::*;
use crate::coord::Face;
use crate::world::chunk::Chunk;
use crate::world::layout::ColumnKey;

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
    // A planet: find one and check the chunk through its centre.
    let planet = (0..64)
        .flat_map(|i| (0..4).map(move |j| (i, j)))
        .find_map(|(i, j)| t.space.planet(i, j, 0).map(|p| p.c));
    let c = planet.expect("some planet among 256 cells");
    assert_chunk_matches(&t, c[0].div_euclid(16), c[1].div_euclid(16), c[2].div_euclid(16));
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
    for &(x, z) in &[(1_000_000_000, -1_000_000_000), (i32::MAX - 40, i32::MIN + 40)] {
        let h = t.height(x, z);
        assert!((MIN_GROUND..=MAX_GROUND).contains(&h));
        let _ = t.generate(x.div_euclid(16), h.div_euclid(16), z.div_euclid(16));
        let _ = t.voxel_at(x, -500, z);
        let _ = t.voxel_at(x, SPACE_FLOOR + 900, z);
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
    // Space: planets with glowing cores.
    let mut cores = 0;
    for i in 0..12 {
        for j in 0..12 {
            if let Some(p) = t.space.planet(i, 0, j) {
                cores += (t.voxel_at(p.c[0], p.c[1], p.c[2]) == m.core || t.voxel_at(p.c[0], p.c[1], p.c[2]) == m.magma)
                    as u32;
            }
        }
    }
    assert!(cores > 10, "only {cores} planet cores in 144 cells");
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
    for i in 0..6 {
        if let Some(p) = t.space.planet(i, 0, 0) {
            println!("planet: {p:?}");
        }
    }
}
