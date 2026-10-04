//! Generator contract and character checks.

use super::*;
use super::cube;
use super::{space, span};
use super::deep;
use crate::coord::{ChunkCoord, Face};
use crate::space::atlas::Patch;
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
    // Surface layers of a few +Y chart columns (trees included), then mine levels and a deep
    // band-0 layer. Face-local chunk offsets land on storage chunks (the rise is chunk-aligned).
    let (_, _, rise) = home_column(&t, Face::PosY, 0, 0);
    let base = (rise / 16) as i32;
    for (cu, cv) in [(0, 0), (3, -2), (-7, 11), (40, 40)] {
        let (sx, sz, _) = home_column(&t, Face::PosY, cu * 16 + 8, cv * 16 + 8);
        let h = t.height(sx, sz);
        let (cx, cz) = (sx.div_euclid(16), sz.div_euclid(16));
        let top = h.div_euclid(16);
        for cy in [top - 1, top, top + 1] {
            assert_chunk_matches(&t, cx, cy, cz);
        }
    }
    for (cu, cv) in [(2, 5), (-9, 4)] {
        let (sx, sz, _) = home_column(&t, Face::PosY, cu * 16 + 8, cv * 16 + 8);
        let (cx, cz) = (sx.div_euclid(16), sz.div_euclid(16));
        for cy in [-1, -3, -5, -9, -20] {
            assert_chunk_matches(&t, cx, base + cy, cz);
        }
    }
}

#[test]
#[allow(clippy::reversed_empty_ranges)] // `1..=0`: the height field with no chunk layers
fn column_heights_agree_on_every_path() {
    let (_reg, t) = make(7);
    for (cu, cv) in [(0, 0), (-3, 9), (100, -40)] {
        let (sx, sz, rise) = home_column(&t, Face::PosY, cu * 16, cv * 16);
        let (cx, cz) = (sx.div_euclid(16), sz.div_euclid(16));
        let (_, hs) = t.generate_column(ColumnKey { face: Face::PosY, a: cx, b: cz }, 1..=0);
        let h16 = t.heights_16(cx, cz);
        for lz in 0..16 {
            for lx in 0..16 {
                let (x, z) = (cx * 16 + lx as i32, cz * 16 + lz as i32);
                assert_eq!(hs[lx + lz * 16], t.height(x, z));
                assert_eq!(h16[lx + lz * 16], t.height(x, z));
                let face = (i64::from(t.height(x, z)) - rise) as i32;
                assert!((MIN_GROUND..=MAX_GROUND).contains(&face), "face altitude {face}");
            }
        }
    }
}

#[test]
fn far_coordinates_generate_without_panic() {
    let (_reg, t) = make(3);
    // Still on the +Y chart, far from the centre column.
    let (sx, sz, rise) = home_column(&t, Face::PosY, 1_000_000, -1_000_000);
    let h = t.height(sx, sz);
    let face = (i64::from(h) - rise) as i32;
    assert!((MIN_GROUND..=MAX_GROUND).contains(&face), "on-face height {face}");
    let _ = t.voxel_at(sx, h - 1, sz);
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
    // Mountains and valleys on the +Y chart: a wide survey finds both high peaks and low floors.
    let (_, _, rise) = home_column(&t, Face::PosY, 0, 0);
    let (mut lo, mut hi) = (i32::MAX, i32::MIN);
    for i in -60..60 {
        for j in -60..60 {
            let (sx, sz, _) = home_column(&t, Face::PosY, i * 40, j * 40);
            let face = (i64::from(t.height(sx, sz)) - rise) as i32;
            lo = lo.min(face);
            hi = hi.max(face);
        }
    }
    assert!(hi >= 300, "no mountain reaches 300 m (highest {hi})");
    assert!(lo <= 50, "no valley floor below 50 m (lowest {lo})");
    // Trees on the chart surface near the centre column.
    let mut trees = 0;
    for cu in -6..6 {
        for cv in -6..6 {
            let (sx, sz, _) = home_column(&t, Face::PosY, cu * 16 + 8, cv * 16 + 8);
            let h = t.height(sx, sz);
            let (cx, cz) = (sx.div_euclid(16), sz.div_euclid(16));
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
    // Face chunks −1/−3/−5/−7 sit `rise/16` storage chunks up, because the rise is chunk-aligned.
    let (sx0, sz0, _) = home_column(&t, Face::PosY, 0, 0);
    let (cx0, cz0) = (sx0.div_euclid(16), sz0.div_euclid(16));
    let mine_base = (rise / 16) as i32;
    let (mut rails, mut planks, mut lamps, mut cave_air) = (0, 0, 0, 0);
    for cu in -20..20 {
        for cv in -20..20 {
            for face_cy in [-1, -3, -5, -7] {
                let d = t.generate(cx0 + cu, mine_base + face_cy, cz0 + cv);
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
            let (sx, sz, _) = home_column(&t, Face::PosY, x, z);
            let h = t.height(sx, sz);
            for y in h - 6..h + 2 {
                s.wake_cell((sx, y, sz));
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

/// Storage `(x, z)` of face-local `(u, v)` on the start world's band 0, and `radius − r_lo`.
fn home_column(t: &Terrain, face: Face, u: i32, v: i32) -> (i32, i32, i64) {
    let atlas = t.storage.home_atlas().expect("the start world is charted");
    let b = atlas.bands[0];
    let half = b.n / 2;
    let (i, j) = (half + i64::from(u), half + i64::from(v));
    assert!((0..b.n).contains(&i) && (0..b.n).contains(&j), "({u},{v}) leaves the {face:?} chart");
    let s = atlas.storage(Patch::Shell { band: 0, face }, [i, 0, j]);
    (s[0] as i32, s[2] as i32, atlas.radius - b.r_lo)
}

/// Storage cell of a virtual-cube relative position, when it lands in band 0.
fn home_rel_storage(t: &Terrain, rel: [i64; 3]) -> Option<(i32, i32, i32)> {
    let atlas = t.storage.home_atlas()?;
    let b = atlas.bands[0];
    let half = b.n / 2;
    let p = (i32::try_from(rel[0]).ok()?, i32::try_from(rel[1]).ok()?, i32::try_from(rel[2]).ok()?);
    let face = cube::face_of(rel);
    let (u, a, v) = FaceFrame::new(face).cell_to_local(p);
    let h = i64::from(a) - half;
    let i = half + i64::from(u);
    let j = half + i64::from(v);
    if !(0..b.n).contains(&i) || !(0..b.n).contains(&j) {
        return None;
    }
    let y = h + atlas.radius - b.r_lo;
    if !(0..b.r_hi - b.r_lo).contains(&y) {
        return None;
    }
    let s = atlas.storage(Patch::Shell { band: 0, face }, [i, y, j]);
    Some((s[0] as i32, s[1] as i32, s[2] as i32))
}

/// Virtual-cube relative position of a start-world band-0 storage cell.
fn storage_to_rel(t: &Terrain, x: i32, y: i32, z: i32) -> Option<[i64; 3]> {
    let atlas = t.storage.home_atlas()?;
    let (patch, local) = atlas.locate([i64::from(x), i64::from(y), i64::from(z)])?;
    let Patch::Shell { band: 0, face } = patch else { return None };
    let b = atlas.bands[0];
    let half = b.n / 2;
    let h = b.r_lo + local[1] - atlas.radius;
    let a = i32::try_from(half + h).ok()?;
    let u = i32::try_from(local[0] - half).ok()?;
    let v = i32::try_from(local[2] - half).ok()?;
    Some(cube::local_to_rel(face, u, a, v))
}

/// Chunk containing the top solid cell at the centre of `body`'s `face`.
fn face_centre_chunk(t: &Terrain, body: &cosmos::Body, face: Face) -> (i32, i32, i32) {
    if body.kind == cosmos::Kind::Home {
        let (sx, sz, _) = home_column(t, face, 0, 0);
        let h = t.height(sx, sz);
        assert_ne!(h, i32::MIN, "home {face:?} has no chart surface");
        return (sx.div_euclid(16), (h - 1).div_euclid(16), sz.div_euclid(16));
    }
    let half = cube::half_of(body);
    // This body's face centre. `surface` is the highest cube on the world tangents, and the other
    // twin shares them, so the height comes from this painter.
    let h = cube::blend_height(t.paint(body, face).shape.height(0, 0), cube::rim_seed(body), face, half, 0, 0);
    let a = (half + i64::from(h - 1)) as i32;
    let (rx, ry, rz) = FaceFrame::new(face).cell_to_world((0, a, 0));
    let world = [
        i64::from(rx) + body.centre[0],
        i64::from(ry) + body.centre[1],
        i64::from(rz) + body.centre[2],
    ];
    chunk_of(t.storage.cube_storage(body.id, world).unwrap_or(world))
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
    let (_, _, rise) = home_column(&t, Face::PosY, 0, 0);
    for x in [-4000, -80, 0, 8, 40, 1000, 9000] {
        for z in [-9000, -15, 0, 8, 77, 2500] {
            let (sx, sz, _) = home_column(&t, Face::PosY, x, z);
            let face = (i64::from(t.height(sx, sz)) - rise) as i32;
            assert_eq!(face, paint.shape.height(x, z), "({x},{z})");
            assert!((MIN_GROUND..=MAX_GROUND).contains(&face));
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
        // A chart's up is storage +Y, on every face.
        assert_eq!(t.sky(coord), Sky::Axis(Face::PosY), "home {face:?} centre sky");
        assert_eq!(t.classify(coord), Classify::Mixed, "home {face:?} centre is crust");
        assert_chunk_matches(&t, cx, cy, cz);
    }
    // Inside the rim blend: batched and blended. The last in-chart column is the seam.
    let half = cube::half_of(&home) as i32;
    let (sx, sz, _) = home_column(&t, Face::PosY, half - 3_000, 0);
    let h = t.height(sx, sz);
    let (cx, cy, cz) = (sx.div_euclid(16), h.div_euclid(16), sz.div_euclid(16));
    assert_eq!(t.sky(ChunkCoord::new(cx, cy, cz)), Sky::Axis(Face::PosY));
    assert_chunk_matches(&t, cx, cy, cz);
    let (_, hs) = t.generate_column(ColumnKey { face: Face::PosY, a: cx, b: cz }, 1..=0);
    let lu = sx.rem_euclid(16) as usize;
    assert_eq!(hs[lu], t.height(sx, sz));

    let (sx, sz, _) = home_column(&t, Face::PosY, half - 1, 0);
    let hs = t.height(sx, sz);
    assert_chunk_matches(&t, sx.div_euclid(16), hs.div_euclid(16), sz.div_euclid(16));
    let (sx, sz, _) = home_column(&t, Face::PosY, half - 1, half - 1);
    let hc = t.height(sx, sz);
    assert_chunk_matches(&t, sx.div_euclid(16), hc.div_euclid(16), sz.div_euclid(16));

    // Deep bulk in band 0, both the classify short-circuit and the column fill.
    let (sx, sz, rise) = home_column(&t, Face::PosY, 0, 0);
    let deep_y = (rise - 2_000) as i32;
    let mut found = None;
    for (du, dv) in [(0, 0), (64, 0), (0, 80), (128, -40), (200, 200), (400, -90)] {
        let (x, z, _) = home_column(&t, Face::PosY, du, dv);
        let c = ChunkCoord::new(x.div_euclid(16), deep_y.div_euclid(16), z.div_euclid(16));
        if let Classify::Uniform(id) = t.classify(c) {
            if id != AIR {
                found = Some((c, id));
                break;
            }
        }
    }
    let (deep, id) = found.expect("a quiet deep band-0 chunk");
    assert_eq!(t.generate(deep.x, deep.y, deep.z).uniform(), Some(id));
    assert_worker_matches(&t, deep.x, deep.y, deep.z);

    // Above every landmark on +Y. The face altitude sits `rise` storage cells up.
    let above = (rise / 16) as i32 + (cube::TREE_CLEAR + 15) / 16;
    let (cx, cz) = (sx.div_euclid(16), sz.div_euclid(16));
    assert_eq!(t.classify(ChunkCoord::new(cx, above, cz)), Classify::Uniform(AIR));
    assert_eq!(t.generate(cx, above, cz).uniform(), Some(AIR));
    assert_worker_matches(&t, cx, above, cz);

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
    let twin = t.cosmos.bodies().iter().copied().find(|b| b.kind == cosmos::Kind::Twin).expect("a twin");
    let half = cube::half_of(&twin);
    let hi = half as i32;
    let centre = cube::centre_i32(twin.centre).expect("twin centre");
    let (cu, _, cv) = FaceFrame::new(Face::PosY).cell_to_local(centre);
    let (xu, _, xv) = FaceFrame::new(Face::PosX).cell_to_local(centre);
    let ny = cube::normal_dot(twin.centre, Face::PosY);
    let nx = cube::normal_dot(twin.centre, Face::PosX);
    for v in [0, 1_000, -8_000, hi - 10] {
        let dy = t.surface(Face::PosY, cu + hi, cv + v);
        let dx = t.surface(Face::PosX, xu - hi, xv + v);
        let hy = cube::face_h(half, ny, dy);
        let hx = cube::face_h(half, nx, dx);
        assert_eq!(hy, hx, "edge v={v}");
        assert!((MIN_GROUND..=MAX_GROUND).contains(&hy));
        // One block past the square still belongs to this face and rebuilds the same rim.
        assert_eq!(t.surface(Face::PosY, cu + hi + 1, cv + v), dy, "wedge v={v}");
        let solid = occupy(&t, &twin, face_world(&twin, Face::PosY, hi, hy - 1, v));
        assert_ne!(t.voxel_at(solid[0], solid[1], solid[2]), AIR, "ridge solid v={v}");
        let above = occupy(&t, &twin, face_world(&twin, Face::PosY, hi, hy + 30, v));
        assert_eq!(t.voxel_at(above[0], above[1], above[2]), AIR, "above the ridge v={v}");
    }
    let dy = t.surface(Face::PosY, cu + hi, cv + hi);
    let dx = t.surface(Face::PosX, xu - hi, xv + hi);
    let hy = cube::face_h(half, ny, dy);
    let hx = cube::face_h(half, nx, dx);
    assert_eq!(hy, hx, "corner +X");
    // The exact three-face corner has one owner. The other faces' `surface()` is empty.
    // At inside = 0 every face blends to that shared rim.
    for (face, ub, vb) in [(Face::PosY, half, half), (Face::PosX, -half, half), (Face::PosZ, half, -half)] {
        let rim = cube::blend_height(0, cube::rim_seed(&twin), face, half, ub, vb);
        assert_eq!(rim, hy, "{face:?} corner rim");
    }
    let solid = occupy(&t, &twin, face_world(&twin, Face::PosY, hi, hy - 1, hi));
    assert_ne!(t.voxel_at(solid[0], solid[1], solid[2]), AIR);
    let above = occupy(&t, &twin, face_world(&twin, Face::PosY, hi, hy + 30, hi));
    assert_eq!(t.voxel_at(above[0], above[1], above[2]), AIR);
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
        ChunkCoord::new(0, t.cosmos.home().centre[1].div_euclid(16) as i32, 0),
    ];
    let moon = t.cosmos.bodies().iter().copied().find(|b| b.kind == cosmos::Kind::Moon).unwrap();
    let (mx, my, mz) = chunk_of(moon.centre);
    let mut samples = samples.to_vec();
    samples.push(ChunkCoord::new(mx, my, mz));
    let (sx, sz, rise) = home_column(&t, Face::PosY, 0, 0);
    let h = t.height(sx, sz);
    samples.push(ChunkCoord::new(sx.div_euclid(16), (h - 1).div_euclid(16), sz.div_euclid(16)));
    let deep_y = (rise - 2_000) as i32;
    samples.push(ChunkCoord::new(sx.div_euclid(16), deep_y.div_euclid(16), sz.div_euclid(16)));
    for c in samples {
        let data = t.generate(c.x, c.y, c.z);
        match t.classify(c) {
            Classify::Air => assert_eq!(data.uniform(), Some(AIR), "air {c:?}"),
            Classify::Uniform(id) => assert_eq!(data.uniform(), Some(id), "uniform {c:?}"),
            Classify::Mixed => assert!(data.uniform().is_none() || data.uniform() == Some(AIR), "mixed {c:?}"),
        }
    }
    // Chart surface, a twin's edge band, empty space, a moon.
    let y = h.div_euclid(16);
    assert_eq!(t.sky(ChunkCoord::new(sx.div_euclid(16), y, sz.div_euclid(16))), Sky::Axis(Face::PosY));
    let twin = t.cosmos.bodies().iter().copied().find(|b| b.kind == cosmos::Kind::Twin).unwrap();
    let cosmos::Shape::Cube { half } = twin.shape else { panic!("a twin is a cube") };
    let edge = occupy(&t, &twin, face_world(&twin, Face::PosY, half as i32 - 100, 0, 0));
    assert_eq!(
        t.sky(ChunkCoord::new(edge[0].div_euclid(16), edge[1].div_euclid(16), edge[2].div_euclid(16))),
        Sky::Open
    );
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
            if !checked {
                let (sx, sz, rise) = home_column(&t, Face::PosY, x, z);
                let y = (i64::from(col.height) + rise) as i32;
                if t.voxel_at(sx, y, sz) == id {
                    checked = true;
                }
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

/// Storage cell of a warped cube's reference cell, or the reference cell when the cube is not stored.
fn occupy(t: &Terrain, body: &cosmos::Body, world: [i32; 3]) -> [i32; 3] {
    let p = [i64::from(world[0]), i64::from(world[1]), i64::from(world[2])];
    match t.storage.cube_storage(body.id, p) {
        Some(s) => [s[0] as i32, s[1] as i32, s[2] as i32],
        None => world,
    }
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
            cosmos::Shape::Cube { half } => {
                air(c);
                air([c[0] + half / 2, c[1], c[2]]);
                air([c[0] + half - 10, c[1], c[2]]);
            }
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
        let w = occupy(&t, body, face_world(body, face, u, h, v));
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
        let aw = occupy(&t, body, face_world(body, face, au, ah, av));
        let arch = t.voxel_at(aw[0], aw[1], aw[2]);
        let ok = if lush { arch == m.timber || arch == m.leaves } else { arch == m.crystal || arch == m.glowshroom };
        assert!(ok, "arch block {arch:?}");
        // Above the spires and every landmark (sky islands reach higher than the spires).
        let clear = occupy(&t, body, face_world(body, face, 0, span::CLEAR.max(cube::TREE_CLEAR) + 16, 0));
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
    let rock = t.cosmos.great_rock().expect("a great rock");
    let eye = [rock.centre[0] as i64, rock.centre[1] as i64 + rock.r as i64, rock.centre[2] as i64];
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

fn world_of(t: &Terrain, body: &cosmos::Body, rel: [i64; 3]) -> (i32, i32, i32) {
    if body.kind == cosmos::Kind::Home {
        return home_rel_storage(t, rel).unwrap_or_else(|| panic!("home cell {rel:?} is outside band 0"));
    }
    (
        i32::try_from(body.centre[0] + rel[0]).unwrap(),
        i32::try_from(body.centre[1] + rel[1]).unwrap(),
        i32::try_from(body.centre[2] + rel[2]).unwrap(),
    )
}

fn chunk_matches_at(t: &Terrain, body: &cosmos::Body, rel: [i64; 3]) {
    let (x, y, z) = world_of(t, body, rel);
    assert_chunk_matches(t, x.div_euclid(16), y.div_euclid(16), z.div_euclid(16));
}

/// An air cell near `at` (features dress the shell, so the middle of a hollow is open).
fn find_air(t: &Terrain, body: &cosmos::Body, at: [i64; 3]) -> [i64; 3] {
    for dz in -2..=2 {
        for dy in -2..=2 {
            for dx in -2..=2 {
                let rel = [at[0] + dx * 8, at[1] + dy * 8, at[2] + dz * 8];
                let (x, y, z) = world_of(t, body, rel);
                if t.voxel_at(x, y, z) == AIR {
                    return rel;
                }
            }
        }
    }
    panic!("no air near {at:?}");
}

#[test]
fn the_interior_is_batch_exact_and_the_heart_is_below_band_0() {
    let (_reg, t) = make(42);
    let home = *t.cosmos.home();
    let cavern = t.deep.locate_cavern(&home).expect("a deep cavern");
    let air = find_air(&t, &home, cavern);
    let (x, y, z) = world_of(&t, &home, air);
    assert_eq!(t.voxel_at(x, y, z), AIR, "cavern air at {air:?}");
    chunk_matches_at(&t, &home, cavern);

    let hall = t.deep.locate_hall(&home).expect("a dwarf hall");
    chunk_matches_at(&t, &home, hall);
    let (hx, hy, hz) = world_of(&t, &home, hall);
    assert_eq!(t.voxel_at(hx, hy, hz), AIR, "hall centre is the room");

    let chamber = t.deep.locate_chamber(&home).expect("an underdark chamber");
    let chamber_air = find_air(&t, &home, chamber);
    chunk_matches_at(&t, &home, chamber);
    let (cx, cy, cz) = world_of(&t, &home, chamber_air);
    assert_eq!(t.voxel_at(cx, cy, cz), AIR, "chamber air");

    let (bubble, r) = t.deep.locate_bubble(&home).expect("a mantle bubble");
    let mut inside = bubble;
    inside[0] += r / 2;
    let (bx, by, bz) = world_of(&t, &home, inside);
    assert_eq!(t.voxel_at(bx, by, bz), AIR, "bubble interior r={r}");
    chunk_matches_at(&t, &home, inside);
    let mut bulk_outside = false;
    for axis in 0..3 {
        for sign in [-1i64, 1] {
            let mut rel = bubble;
            rel[axis] += sign * (r + 3);
            let (ox, oy, oz) = world_of(&t, &home, rel);
            if t.voxel_at(ox, oy, oz) == cube::bulk_id(&t.bulk, &home, rel) {
                bulk_outside = true;
            }
        }
    }
    assert!(bulk_outside, "a block just outside the bubble is still bulk");

    // The Heart and its shaft sit below band 0 (the virtual cube's centre is not on a chart).
    // The core box is the uniform home fill, with or without the interior.
    assert!(home_rel_storage(&t, [0, 0, 0]).is_none(), "the Heart is below band 0");
    let atlas = t.storage.home_atlas().expect("charted");
    let inner = atlas.inner.expect("a core");
    let core = atlas.storage(Patch::Core, [inner.core_half, inner.core_half, inner.core_half]);
    assert_eq!(t.voxel_at(core[0] as i32, core[1] as i32, core[2] as i32), t.storage.home_fill());

    let mut reg = BlockRegistry::with_builtins();
    let off = Terrain::with_cfg(&mut reg, 42, TerrainCfg { deep: 0, ..TerrainCfg::default() });
    assert_ne!(off.voxel_at(x, y, z), AIR, "deep=0 left the cavern hollow");
    assert_eq!(off.voxel_at(x, y, z), cube::bulk_id(&off.bulk, &home, air));
    assert_eq!(off.voxel_at(core[0] as i32, core[1] as i32, core[2] as i32), off.storage.home_fill());
}

fn emissive(t: &Terrain, id: BlockId) -> bool {
    let m = t.materials();
    id == m.lamp || id == m.glowcap || id == m.glowshroom || id == m.magma || id == m.star || id == m.core
}

/// An air cell of the feature. A site centre can sit just past its depth band, where the
/// painter leaves bulk; the carved cap is a short walk along the face normal.
fn carved_air(t: &Terrain, body: &cosmos::Body, center: [i64; 3]) -> [i64; 3] {
    let up = deep::Up::of(center);
    let half = cube::half_of(body);
    for delta in 0..400 {
        for sign in [1i64, -1] {
            if delta == 0 && sign < 0 {
                continue;
            }
            let mut rel = center;
            rel[up.axis] += sign * delta * i64::from(up.sign);
            let Some((x, y, z)) = home_rel_storage(t, rel) else { continue };
            if t.voxel_at(x, y, z) == AIR && cube::in_deep(rel, half) {
                return rel;
            }
        }
    }
    panic!("no carved air near {center:?}");
}

/// First solid inward of `center` after an air cell, and that air cell.
///
/// Once the walk is in open air it strides, then finishes the last stride one cell at a time so a
/// thin lining is not stepped over.
fn inward_shell(t: &Terrain, body: &cosmos::Body, center: [i64; 3]) -> ([i64; 3], [i64; 3]) {
    let up = deep::Up::of(center);
    let mut rel = center;
    let mut prev = center;
    let mut saw_air = false;
    let mut step = 1i64;
    for _ in 0..4_000 {
        let (x, y, z) = world_of(t, body, rel);
        let id = t.voxel_at(x, y, z);
        if id == AIR {
            saw_air = true;
            step = 8;
        } else if saw_air {
            if step == 1 {
                return (rel, prev);
            }
            let mut fine = prev;
            for _ in 0..step {
                let mut next = fine;
                next[up.axis] -= i64::from(up.sign);
                let (x, y, z) = world_of(t, body, next);
                if t.voxel_at(x, y, z) != AIR {
                    return (next, fine);
                }
                fine = next;
            }
            return (rel, prev);
        }
        prev = rel;
        rel[up.axis] -= i64::from(up.sign) * step;
    }
    panic!("no dressed shell inward of {center:?}");
}

fn match_cells(t: &Terrain, body: &cosmos::Body, cells: &[[i64; 3]]) {
    let mut seen = Vec::new();
    for &rel in cells {
        let (x, y, z) = world_of(t, body, rel);
        let key = (x.div_euclid(16), y.div_euclid(16), z.div_euclid(16));
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        chunk_matches_at(t, body, rel);
    }
}

/// An emissive cell in the chunk, if the shell dressing landed in this 16-block box.
fn light_in_chunk(t: &Terrain, body: &cosmos::Body, rel: [i64; 3]) -> Option<[i64; 3]> {
    let (x, y, z) = world_of(t, body, rel);
    let n = CHUNK_SIZE as i32;
    let (cx, cy, cz) = (x.div_euclid(n), y.div_euclid(n), z.div_euclid(n));
    let data = t.generate(cx, cy, cz);
    for ly in 0..CHUNK_SIZE {
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                if !emissive(t, data.get(Chunk::index(lx, ly, lz))) {
                    continue;
                }
                let wx = cx * n + lx as i32;
                let wy = cy * n + ly as i32;
                let wz = cz * n + lz as i32;
                if body.kind == cosmos::Kind::Home {
                    return storage_to_rel(t, wx, wy, wz);
                }
                let c = body.centre;
                return Some([i64::from(wx) - c[0], i64::from(wy) - c[1], i64::from(wz) - c[2]]);
            }
        }
    }
    None
}

fn cover_frac(c: &deep::Cover) -> f64 {
    assert!(c.floor > 0, "no floor cells");
    f64::from(c.near) / f64::from(c.floor)
}

/// Each cavern biome, the smallest chamber and the halls light 60% of their floor, and the
/// dressed shell matches the per-voxel query. `deep = 0` removes the light with the hollow.
#[test]
fn interior_lights_cover_the_floor_and_match_per_voxel() {
    let (_reg, t) = make(42);
    let home = *t.cosmos.home();
    let kinds = t.deep.cover_kinds(&home);
    assert_eq!(kinds.len(), 6, "six cavern biomes");
    let mut light = None;
    for c in &kinds {
        let frac = cover_frac(c);
        println!("cavern kind {} r {} floor {} near {} frac {frac:.3}", c.kind, c.r, c.floor, c.near);
        assert!(frac >= 0.60, "kind {} coverage {frac}", c.kind);
        let (solid, air) = inward_shell(&t, &home, carved_air(&t, &home, c.center));
        match_cells(&t, &home, &[solid, air]);
        let (a, b) = match c.up_axis {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        for (da, db) in [(0i64, 0), (1, 0), (-1, 0), (0, 1), (0, -1)] {
            let mut rel = solid;
            rel[a] += da * 16;
            rel[b] += db * 16;
            if let Some(found) = light_in_chunk(&t, &home, rel) {
                if (da, db) != (0, 0) {
                    chunk_matches_at(&t, &home, found);
                }
                light = Some(found);
                break;
            }
        }
    }
    let rel = light.expect("a floor light in a dressed shell chunk");

    let chamber = t.deep.cover_chamber(&home).expect("underdark chamber");
    let frac = cover_frac(&chamber);
    println!("chamber r {} floor {} near {} frac {frac:.3}", chamber.r, chamber.floor, chamber.near);
    assert!(frac >= 0.60, "chamber coverage {frac}");
    let (solid, air) = inward_shell(&t, &home, carved_air(&t, &home, chamber.center));
    match_cells(&t, &home, &[solid, air]);

    let halls = t.deep.survey_halls(&home, 4);
    assert!(halls.len() >= 4, "halls {}", halls.len());
    for h in &halls {
        let frac = cover_frac(h);
        println!("hall len {} floor {} near {} frac {frac:.3}", h.r, h.floor, h.near);
        assert!(frac >= 0.60, "hall coverage {frac}");
    }
    let hall = t.deep.locate_hall(&home).expect("hall");
    let (solid, air) = inward_shell(&t, &home, carved_air(&t, &home, hall));
    match_cells(&t, &home, &[solid, air]);

    let mut reg = BlockRegistry::with_builtins();
    let off = Terrain::with_cfg(&mut reg, 42, TerrainCfg { deep: 0, ..TerrainCfg::default() });
    let (x, y, z) = world_of(&t, &home, rel);
    assert_eq!(off.voxel_at(x, y, z), cube::bulk_id(&off.bulk, &home, rel), "deep=0 left a floor light");
}

#[test]
fn interior_porosity_barely_moves_spawn_gravity() {
    use crate::gravity::{Field, Primitive};
    use crate::math::BLOCK_METERS;
    // Home is a ball. Mantle bubbles live in the virtual cube and only the part inside band 0 is
    // carved, so the oracle stays the solid ball (plus the relief error near the surface).
    let cosmos = std::sync::Arc::new(cosmos::Cosmos::with_deep(7, 1.0, 1.0));
    let field = Field::new(cosmos.clone());
    let home = cosmos.home();
    let cosmos::Shape::Ball { r } = home.shape else { panic!("home is a ball") };
    let at = glam::DVec3::new(0.0, 2.0, 0.0);
    let sample = field.sample(at);
    let solid = Primitive::new(
        crate::gravity::Shape::Ball { c: home.centre_f(), r: r as f64 },
        cosmos::BULK_DENSITY,
    );
    let want = solid.field(at).0 * crate::gravity::G;
    let after = sample.accel.length() * BLOCK_METERS;
    let before = want.length() * BLOCK_METERS;
    let to_centre = home.centre_f() - at;
    let dot = sample.accel.dot(to_centre) / (sample.accel.length() * to_centre.length());
    println!("spawn gravity solid {before:.4} m/s² charted {after:.4} m/s² dot {dot:.6}");
    assert!(dot > 0.999, "pull points at the centre: {dot}");
    assert!((after - before).abs() < 0.05 * before, "interior moved spawn: {after} vs {before}");
    let centre_pull = field.sample(home.centre_f()).accel.length() * BLOCK_METERS;
    println!("planet centre pull {centre_pull:.6} m/s²");
    assert!(centre_pull < 0.05, "centre is not weightless: {centre_pull}");
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
            // Deeper than band 0 the chart does not paint this cell.
            let Some((x, y, z)) = home_rel_storage(&t, rel) else { continue };
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

/// Storage cell of a chart column, or the world cell of a cube face column.
/// `h` is altitude above the face plane.
fn face_cell(t: &Terrain, body: &cosmos::Body, face: Face, u: i32, h: i32, v: i32) -> (i32, i32, i32) {
    if body.kind == cosmos::Kind::Home {
        let (sx, sz, rise) = home_column(t, face, u, v);
        let atlas = t.storage.home_atlas().expect("the start world is charted");
        let _ = atlas;
        let y = i64::from(h) + rise;
        return (sx, y as i32, sz);
    }
    let half = cube::half_of(body);
    let a = i32::try_from(half + i64::from(h)).expect("altitude");
    let (x, y, z) = FaceFrame::new(face).cell_to_world((u, a, v));
    let world = [
        i64::from(x) + body.centre[0],
        i64::from(y) + body.centre[1],
        i64::from(z) + body.centre[2],
    ];
    let s = t.storage.cube_storage(body.id, world).unwrap_or(world);
    (
        i32::try_from(s[0]).expect("x"),
        i32::try_from(s[1]).expect("y"),
        i32::try_from(s[2]).expect("z"),
    )
}

fn agree_cell(t: &Terrain, body: &cosmos::Body, face: Face, u: i32, h: i32, v: i32) {
    let (x, y, z) = face_cell(t, body, face, u, h, v);
    let coord = ChunkCoord::new(x.div_euclid(16), y.div_euclid(16), z.div_euclid(16));
    // A chart's up is storage +Y on every face. A cube keeps the face's own sky.
    let sky = if body.kind == cosmos::Kind::Home { Face::PosY } else { face };
    assert_eq!(t.sky(coord), Sky::Axis(sky), "landmark chunk {coord:?} on {face:?} uses the batch path");
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
    let home = *t.cosmos.home();
    let u = cube::half_of(&home) as i32 - 80;
    let allowed = [
        AIR, m.timber, m.leaves, m.pine, m.autumn, m.blossom, m.flower_red, m.flower_yellow, m.flower_blue,
        m.flower_white,
    ];
    for v in (-24..24).step_by(2) {
        let (sx, sz, _) = home_column(&t, Face::PosY, u, v);
        let h = t.height(sx, sz);
        assert_ne!(h, i32::MIN, "rim column ({u},{v}) has no surface");
        for y in h..h + 40 {
            let id = t.voxel_at(sx, y, sz);
            assert!(allowed.contains(&id), "edge cell ({sx},{y},{sz}) is not a tree or flower");
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
        let (wx, wy, wz) = face_cell(&t, &home, Face::PosY, f.x, f.pad, f.z);
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

fn cover_line(label: &str, rows: &[deep::Cover]) {
    let mut cells = 0u64;
    let mut near = 0u64;
    let mut min = 1.0f64;
    for c in rows {
        let f = if c.floor == 0 { 0.0 } else { f64::from(c.near) / f64::from(c.floor) };
        min = min.min(f);
        cells += u64::from(c.floor);
        near += u64::from(c.near);
        let name = if label == "cavern" || label == "named" {
            match c.kind {
                0 => "fungal",
                1 => "geode",
                2 => "magma",
                3 => "roots",
                4 => "cones",
                5 => "glow",
                _ => "other",
            }
        } else {
            label
        };
        println!(
            "{label} {name} r={} step={} floor={} near={} frac={f:.3} center={:?}",
            c.r, c.step, c.floor, c.near, c.center
        );
    }
    let mean = if cells == 0 { 0.0 } else { near as f64 / cells as f64 };
    let n = rows.len();
    println!("{label} n={n} floor_cells={cells} near={near} mean={mean:.3} min={min:.3}");
    let _ = std::io::Write::flush(&mut std::io::stdout());
}

fn shell_bottom(c: &deep::Cover) -> [i64; 3] {
    let co = c.center[c.up_axis] * i64::from(c.up_sign);
    let (t0, t1) = deep::tangent(c.center, c.up_axis);
    deep::place(c.up_axis, c.up_sign, (t0, t1), co - c.r)
}

fn world_chunk(body: &cosmos::Body, rel: [i64; 3]) -> [i32; 3] {
    std::array::from_fn(|i| i32::try_from((body.centre[i] + rel[i]).div_euclid(16)).unwrap())
}

/// Chunks on the cavern floor (and one layer above it) whose box meets the ball.
fn floor_chunks(body: &cosmos::Body, c: &deep::Cover, n: usize) -> Vec<[i32; 3]> {
    let origin = world_chunk(body, shell_bottom(c));
    let tang = match c.up_axis {
        0 => [1usize, 2],
        1 => [0, 2],
        _ => [0, 1],
    };
    let mut out = Vec::new();
    for dout in 0..3 {
        for dv in -4..=4 {
            for du in -4..=4 {
                let mut coord = origin;
                coord[tang[0]] += du;
                coord[tang[1]] += dv;
                coord[c.up_axis] += dout * c.up_sign;
                let lo = std::array::from_fn(|i| i64::from(coord[i]) * 16 - body.centre[i]);
                let hi = std::array::from_fn(|i| lo[i] + 15);
                if deep::sphere_hits(c.center, c.r, lo, hi) {
                    out.push(coord);
                    if out.len() == n {
                        return out;
                    }
                }
            }
        }
    }
    out
}

fn fill_face_us(t: &Terrain, coord: ChunkCoord, reps: usize) -> Option<f64> {
    let Sky::Axis(face) = t.sky(coord) else { return None };
    let (key, _) = ColumnKey::of(face, coord);
    let body = t.face_column_body_in(key, true)?;
    let half = cube::half_of(&body);
    let centre = cube::centre_i32(body.centre)?;
    let (wu, wv) = key.column_cell_uv(0, 0);
    let (u0, v0) = cube::tangents(face, centre, wu, wv);
    let (u0, v0) = (u0 as i32, v0 as i32);
    let paint = t.paint(&body, face);
    let seed = cube::rim_seed(&body);
    let n_dot = cube::normal_dot(body.centre, face);
    let raw = paint.shape.columns_16(u0, v0);
    let mut max_terrain = i32::MIN;
    let mut min_h = i32::MAX;
    let mut cols = Vec::with_capacity(CHUNK_SIZE * CHUNK_SIZE);
    for (i, mut col) in raw.into_iter().enumerate() {
        let lu = (i % CHUNK_SIZE) as i32;
        let lv = (i / CHUNK_SIZE) as i32;
        let (ub, vb) = (i64::from(u0) + i64::from(lu), i64::from(v0) + i64::from(lv));
        max_terrain = max_terrain.max(col.height);
        col.height = cube::blend_height(col.height, seed, face, half, ub, vb);
        min_h = min_h.min(col.height);
        cols.push(col);
    }
    let tree_blocks = paint.trees.blocks_in(&paint.shape, u0, v0, CHUNK_SIZE as i32);
    let h0 = cube::face_h(half, n_dot, FaceFrame::new(face).chunk_alt0(coord));
    let _ = t.fill_face::<false>(&body, face, &cols, u0, v0, h0, max_terrain, min_h, &tree_blocks);
    let start = std::time::Instant::now();
    for _ in 0..reps {
        std::hint::black_box(t.fill_face::<false>(&body, face, &cols, u0, v0, h0, max_terrain, min_h, &tree_blocks));
    }
    Some(start.elapsed().as_secs_f64() * 1e6 / reps as f64)
}

/// Coverage of cavern, chamber and hall floors, plus block-light and `fill_deep` cost on one cavern.
///
/// `cargo test --release --lib interior_light_report -- --ignored --nocapture`
#[test]
#[ignore]
fn interior_light_report() {
    use crate::world::chunk::Chunk;
    use crate::world::light::{self, CeilingWindow, FaceShell, LightGrid};

    let (reg, t) = make(42);
    let home = *t.cosmos.home();
    let caverns = t.deep.survey_caverns(&home, 20);
    cover_line("cavern", &caverns);
    let chambers = t.deep.survey_chambers(&home, 10);
    cover_line("chamber", &chambers);
    let halls = t.deep.survey_halls(&home, 10);
    cover_line("hall", &halls);

    let named_rel = [-3783 - home.centre[0], -3796 - home.centre[1], 291 - home.centre[2]];
    if let Some(named) = t.deep.survey_cavern_at(&home, named_rel) {
        cover_line("named", std::slice::from_ref(&named));
    } else {
        println!("named cavern at {named_rel:?} not found");
    }

    let subject = t.deep.survey_cavern_at(&home, named_rel).or_else(|| caverns.into_iter().next());
    let Some(subject) = subject else {
        println!("no cavern to time");
        return;
    };
    let coords = floor_chunks(&home, &subject, 50);
    println!(
        "timing {} floor chunks of kind {} r {} at {:?}",
        coords.len(),
        subject.kind,
        subject.r,
        subject.center
    );
    let tables = reg.hot_tables();
    let ceiling = CeilingWindow::from_heights(Face::PosY, |_, _| i32::MAX);
    let shell = FaceShell::dark();
    let mut built = Vec::with_capacity(coords.len());
    for c in &coords {
        let data = t.generate(c[0], c[1], c[2]);
        built.push(Chunk::from_data(c[0], c[1], c[2], data));
    }
    let mut emitters = 0u32;
    for ch in &built {
        ch.for_each_emission(&tables.emission, |_, _| emitters += 1);
    }
    let mut grid = LightGrid::dark();
    if let Some(ch) = built.first() {
        light::propagate(ch, &shell, &ceiling, Sky::Axis(Face::PosY), 0, &tables, &mut grid);
    }
    let reps = 30;
    let start = std::time::Instant::now();
    for _ in 0..reps {
        for ch in &built {
            light::propagate(ch, &shell, &ceiling, Sky::Axis(Face::PosY), 0, &tables, &mut grid);
        }
    }
    let light_us = start.elapsed().as_secs_f64() * 1e6 / (reps * built.len().max(1)) as f64;
    println!("blocklight {light_us:.2} µs/chunk over {} chunks x {reps}, emitters {emitters}", built.len());

    let mut fill_sum = 0.0;
    let mut fill_n = 0u32;
    for c in coords.iter().take(12) {
        let coord = ChunkCoord::new(c[0], c[1], c[2]);
        if let Some(us) = fill_face_us(&t, coord, 6) {
            fill_sum += us;
            fill_n += 1;
            println!("fill_face chunk {c:?} {us:.1} µs");
        }
    }
    if fill_n > 0 {
        println!("fill_face mean {:.1} µs/chunk over {fill_n}", fill_sum / f64::from(fill_n));
    }
}

/// The face painter's rim is one height from both sides of a chart edge, and the painted
/// columns just inside the edge stay within a block of each other.
#[test]
fn home_chart_terrain_is_continuous_across_a_seam() {
    let (_reg, t) = make(42);
    let home = *t.cosmos.home();
    let half = cube::half_of(&home);
    let seed = cube::rim_seed(&home);
    for v in [0i64, 1_000, -8_000, half / 3] {
        let y_rim = cube::blend_height(0, seed, Face::PosY, half, half, v);
        let x_rim = cube::blend_height(400, seed, Face::PosX, half, -half, v);
        assert_eq!(y_rim, x_rim, "exact rim at v={v}");
    }
    let atlas = t.storage.home_atlas().expect("charted");
    let b = &atlas.bands[0];
    let top = Patch::Shell { band: 0, face: Face::PosY };
    let rise = atlas.radius - b.r_lo;
    for j in [b.n / 4, b.n / 2, 3 * b.n / 4] {
        let here_s = atlas.storage(top, [b.n - 1, 0, j]);
        let here = t.height(here_s[0] as i32, here_s[2] as i32);
        assert_ne!(here, i32::MIN, "last +Y column j={j}");
        let outside = atlas.storage(top, [b.n, here as i64, j]);
        let glued = atlas.glue(outside).expect("seam glue");
        let (patch, _) = atlas.locate(glued).expect("neighbour chart");
        assert!(matches!(patch, Patch::Shell { band: 0, face: Face::PosX }), "{patch:?} at j={j}");
        let there = t.height(glued[0] as i32, glued[2] as i32);
        let step = (here as i64 - rise) - (there as i64 - rise);
        assert!(step.abs() <= 1, "seam step at j={j}: {here} vs {there} (rise {rise})");
    }
}

/// Physical spawn is the +Y chart's centre column, two blocks above its first open cell,
/// and gravity there points at the planet's centre.
#[test]
fn home_spawn_stands_on_the_plus_y_chart() {
    use crate::gravity::Field;
    let (_reg, t) = make(42);
    let home = *t.cosmos.home();
    let cosmos::Shape::Ball { .. } = home.shape else { panic!("home is a ball") };
    let spawn = t.home_spawn().expect("chart spawn");
    assert!(spawn.x.abs() < 2.0 && spawn.z.abs() < 2.0, "spawn xz {spawn:?}");
    assert!(spawn.y > 2.0 && spawn.y < 600.0, "spawn y {}", spawn.y);
    let field = Field::new(t.mass());
    let pull = field.sample(spawn);
    let to_centre = home.centre_f() - spawn;
    let dot = pull.accel.dot(to_centre) / (pull.accel.length() * to_centre.length());
    assert!(dot > 0.999, "gravity at spawn points at the centre: {dot}");
    let atlas = t.storage.home_atlas().expect("charted");
    let cell = atlas.storage_of(spawn).expect("spawn embeds onto the chart");
    let (patch, local) = atlas.locate(cell).expect("located");
    assert!(matches!(patch, Patch::Shell { band: 0, face: Face::PosY }), "{patch:?}");
    let open = i64::from(t.height(cell[0] as i32, cell[2] as i32));
    assert!((1..=2).contains(&(local[1] - open)), "feet at local {} open {open}", local[1]);
    assert_ne!(t.voxel_at(cell[0] as i32, (open - 1) as i32, cell[2] as i32), AIR, "standing on air");
    assert_eq!(t.voxel_at(cell[0] as i32, open as i32, cell[2] as i32), AIR, "the first open cell is solid");
}

/// A datum lifts the start world's band-0 cells, not its painting: storage heights stay where the
/// face painter put them and the surface cell's physical point moves out by the offset (fitted
/// grid), with `find` inverting the lifted embedding.
#[test]
fn a_datum_lifts_the_home_grid_and_leaves_its_painting() {
    use crate::space::datum::DatumField;
    let (_reg, mut t) = make(42);
    let mut atlas = t.storage.home_atlas().expect("charted").clone();
    atlas.datum = None;
    let b = atlas.bands[0];
    let half = b.n / 2;
    let patch = Patch::Shell { band: 0, face: Face::PosY };
    let samples = [(0i32, 0i32), (8_000, -3_000), ((half - 5_000) as i32, 1_200), (-4_000, (5_000 - half) as i32)];
    let before: Vec<i32> = samples.iter().map(|&(u, v)| {
        let (sx, sz, _) = home_column(&t, Face::PosY, u, v);
        t.height(sx, sz)
    }).collect();
    let radius = atlas.radius as f64;
    t.set_home_datum(std::sync::Arc::new(DatumField::sample(17, radius, |d| radius + 2_000.0 + 800.0 * d.x)));
    let lifted = t.storage.home_atlas().expect("charted").clone();
    for (k, &(u, v)) in samples.iter().enumerate() {
        let (sx, sz, _) = home_column(&t, Face::PosY, u, v);
        let y = t.height(sx, sz);
        assert_eq!(y, before[k], "painting moved at ({u},{v})");
        let (i, j) = ((half + i64::from(u)) as f64 + 0.5, (half + i64::from(v)) as f64 + 0.5);
        let l = glam::DVec3::new(i, f64::from(y) - lifted.storage_box(patch).0[1] as f64, j);
        let flat = (atlas.embed(patch, l) - atlas.centre).length();
        let up = (lifted.embed(patch, l) - lifted.centre).length();
        let off = lifted.datum_offset(patch, i, j);
        assert!((up - flat - off).abs() < 1.0, "lift {} vs offset {off} at ({u},{v})", up - flat);
        let (p, back) = lifted.find(lifted.embed(patch, l)).expect("found");
        assert_eq!(p, patch);
        assert!((back - l).length() < 1e-6, "find {back} vs {l}");
        // Deep in the band the grid stays nearly spherical (the lift tapers to nothing at r_lo).
        let deep = glam::DVec3::new(i, 16.0, j);
        assert!(((lifted.embed(patch, deep) - atlas.embed(patch, deep)).length()) < 0.01 * off.abs() + 1.0);
    }
}

/// A sagging twin's storage cells are the old physical painter, the corner sits about a tenth
/// lower, the face centre bows out, and gravity is that shape rather than a sag error bound.
#[test]
fn warped_twins_keep_the_cube_painter_and_sag() {
    let (_reg, t) = make(42);
    let twins: Vec<_> = t.cosmos.bodies().iter().copied().filter(|b| b.kind == cosmos::Kind::Twin).collect();
    assert_eq!(twins.len(), 2);
    let cubes: Vec<_> = t.atlases().iter().filter(|a| a.grid.is_some()).cloned().collect();
    assert_eq!(cubes.len(), 2, "both twins sag into storage");
    let g0 = cubes[0].grid.unwrap();
    let g1 = cubes[1].grid.unwrap();
    let y_apart = g0.origin[1] + g0.size[1] <= g1.origin[1] || g1.origin[1] + g1.size[1] <= g0.origin[1];
    assert!(y_apart, "cube boxes share a PosX column");
    for atlas in &cubes {
        let g = atlas.grid.unwrap();
        assert!(g.origin[0] + g.size[0] < crate::math::CELL_LIMIT as i64, "cube box leaves i32");
        assert!(g.origin[1] >= crate::space::atlas::STORAGE_X0);
        let outside = [g.origin[0] + g.size[0], g.origin[1] + 16, g.origin[2] + 16];
        assert!(atlas.glue(outside).is_none(), "a cell just outside the box glues");
        let last = [(g.origin[0] + g.size[0]) / 16 - 1, g.origin[1] / 16, g.origin[2] / 16];
        assert!(atlas.chunk_across(last, 0, 1).is_none(), "the chunk past +x is a seam");
    }

    let mut samples = Vec::new();
    for body in &twins {
        let half = cube::half_of(body);
        for face in Face::ALL {
            let (cx, cy, cz) = face_centre_chunk(&t, body, face);
            samples.push(ChunkCoord::new(cx, cy, cz));
        }
        let hi = half as i32;
        for (u, v) in [(hi, 0), (hi, hi)] {
            let cell = occupy(&t, body, face_world(body, Face::PosY, u, 0, v));
            samples.push(ChunkCoord::new(cell[0].div_euclid(16), cell[1].div_euclid(16), cell[2].div_euclid(16)));
        }
        let deep = occupy(&t, body, [body.centre[0] as i32, body.centre[1] as i32, body.centre[2] as i32]);
        samples.push(ChunkCoord::new(deep[0].div_euclid(16), deep[1].div_euclid(16), deep[2].div_euclid(16)));
        let face = span::facing_face(&t.cosmos, body).unwrap();
        let (u, h, v) = span::example(span::face_seed(body), half).expect("a spire");
        let spire = occupy(&t, body, face_world(body, face, u, h, v));
        samples.push(ChunkCoord::new(spire[0].div_euclid(16), spire[1].div_euclid(16), spire[2].div_euclid(16)));
    }
    let n = CHUNK_SIZE as i32;
    for coord in samples {
        let data = t.generate(coord.x, coord.y, coord.z);
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                for ly in 0..CHUNK_SIZE {
                    let (x, y, z) = (coord.x * n + lx as i32, coord.y * n + ly as i32, coord.z * n + lz as i32);
                    let (id, reference) = t.storage.cube_ref_cell(x, y, z).expect("storage cell of a sampled chunk");
                    let body = t.cosmos.bodies().iter().find(|b| b.id == id).expect("cube body");
                    assert_eq!(data.get(Chunk::index(lx, ly, lz)), t.cube_cell(body, reference), "({x},{y},{z})");
                    let (Ok(px), Ok(py), Ok(pz)) =
                        (i32::try_from(reference[0]), i32::try_from(reference[1]), i32::try_from(reference[2]))
                    else {
                        continue;
                    };
                    assert_eq!(t.voxel_at(px, py, pz), AIR, "physical ({px},{py},{pz}) still holds the twin");
                }
            }
        }
    }

    let body = twins[0];
    let cosmos::Shape::Cube { half } = body.shape else { panic!("a twin is a cube") };
    let warp = cubes.iter().find(|a| a.grid.unwrap().body == body.id).unwrap().warp.as_ref().unwrap();
    let c = body.centre_f();
    let h = half as f64;
    let radial = (warp.apply(c + glam::DVec3::new(h, h, h)) - c).length();
    let undeformed = h * 3.0f64.sqrt();
    let ratio = radial / undeformed;
    let face_len = (warp.apply(c + glam::DVec3::new(h, 0.0, 0.0)) - c).length();
    // Corner/face is the roundness: a cube is √3 ≈ 1.73, the twins' sag is about 1.56.
    let roundness = radial / face_len;
    assert!(
        (roundness - 1.56).abs() < 0.03 && ratio < 0.98,
        "corner/face {roundness} radial ratio {ratio}"
    );
    assert!(face_len > h && face_len - h < 0.10 * h, "face centre {face_len} vs half {h}");

    let sag = warp.max_displacement();
    let field = crate::gravity::Field::new(t.cosmos.clone());
    let other = twins[1];
    let delta = [
        body.centre[0] - other.centre[0],
        body.centre[1] - other.centre[1],
        body.centre[2] - other.centre[2],
    ];
    let axis = (0..3).max_by_key(|&a| delta[a].abs()).unwrap();
    let sign = if delta[axis] >= 0 { 1.0 } else { -1.0 };
    // On the bowed surface (the warp of the reference face centre), and well beyond it.
    let mut face = c;
    face[axis] += sign * h;
    let near = warp.apply(face);
    let mut far = c;
    far[axis] += sign * (h + sag.max(cosmos::RELIEF as f64) * 8.0);
    let near_s = field.sample(near);
    let far_s = field.sample(far);
    // Terrain relief is still an error slab. The sag is the polyhedron, so it is not.
    let relief = crate::gravity::G * 2.0 * std::f64::consts::PI * body.density * cosmos::RELIEF as f64;
    assert!(near_s.error + 1e-6 >= relief, "near error {} < relief {relief}", near_s.error);
    let sag_term = crate::gravity::G * 2.0 * std::f64::consts::PI * body.density * sag * 0.5;
    assert!(
        near_s.error - far_s.error < sag_term,
        "sag still declared as error: near {} far {} sag term {sag_term}",
        near_s.error,
        far_s.error
    );
    assert!(near_s.accel.is_finite() && far_s.accel.is_finite());
}

/// Release cost of painting one home chart chunk. Debug skips it.
///
/// `cargo test --release --lib home_chart_chunk_cost -- --ignored --nocapture`
#[test]
#[ignore]
fn home_chart_chunk_cost() {
    let (_reg, t) = make(42);
    let home = *t.cosmos.home();
    let half = cube::half_of(&home) as i32;
    let spots = [(Face::PosY, 0, 0), (Face::PosX, 0, 0), (Face::PosY, half - 8, 0)];
    for (face, u, v) in spots {
        let (sx, sz, _) = home_column(&t, face, u, v);
        let h = t.height(sx, sz);
        let (cx, cy, cz) = (sx.div_euclid(16), (h - 1).div_euclid(16), sz.div_euclid(16));
        let _ = t.generate(cx, cy, cz);
        let reps = 6;
        let start = std::time::Instant::now();
        for _ in 0..reps {
            std::hint::black_box(t.generate(cx, cy, cz));
        }
        let us = start.elapsed().as_secs_f64() * 1e6 / reps as f64;
        println!("home chart {face:?} chunk ({cx},{cy},{cz}) {us:.1} µs");
    }
}

/// The start world relaxes from its cube of bulk matter: physics rounds it (Π_g of the rock mix is
/// well above the yield threshold) and its datum carries highlands toward the old cube corners.
#[test]
fn the_start_world_is_relaxed_and_its_grid_fits_the_shape() {
    let (_reg, t) = make(42);
    let atlas = t.storage.home_atlas().expect("charted");
    let datum = atlas.datum.as_ref().expect("the start world has a relaxed datum");
    let (lo, hi) = datum.range();
    println!("home datum relief {lo:.0} .. {hi:.0} blocks about r {}", atlas.radius);
    // Corners high, face centres low: a cube that rounded, not a ball.
    assert!(hi > 2e5 && hi < 3e6, "highlands {hi}");
    assert!(lo < -5e4 && lo > -1.5e6, "lowlands {lo}");
    let g = datum.g;
    let centre = datum.offset(2, 0.0, 0.0);
    let corner = datum.offset(2, 1.0, 1.0);
    assert!(corner > centre + 3e5, "corner {corner} over centre {centre}");
    let _ = g;
}

/// `cargo test --release --lib twin_gravity_sample_cost -- --ignored --nocapture`: one player
/// gravity sample just above a twin's warped face centre.
#[test]
#[ignore]
fn twin_gravity_sample_cost() {
    let (_reg, t) = make(42);
    let body = t.cosmos.bodies().iter().copied().find(|b| b.kind == cosmos::Kind::Twin).expect("twin");
    let warp = t
        .atlases()
        .iter()
        .find_map(|a| a.grid.as_ref().is_some_and(|g| g.body == body.id).then(|| a.warp.clone()).flatten())
        .expect("warp");
    let half = cube::half_of(&body) as f64;
    let c = body.centre_f();
    let face = warp.apply(c + glam::DVec3::new(half, 0.0, 0.0));
    let p = face + (face - c).normalize() * 2.0;
    let field = crate::gravity::Field::new(t.mass());
    let _ = field.sample(p);
    let n = 4_000;
    let start = std::time::Instant::now();
    let mut acc = 0.0;
    for i in 0..n {
        acc += field.sample(p + glam::DVec3::new((i % 7) as f64 * 0.01, 0.0, 0.0)).accel.length();
    }
    println!("{:.2} µs per sample on a twin face [{acc:.3}]", start.elapsed().as_secs_f64() * 1e6 / n as f64);
}

/// `cargo test --release --lib home_gravity_sample_cost -- --ignored --nocapture`: one player
/// gravity sample on the relaxed start world (ball, relief layer, cosmos).
#[test]
#[ignore]
fn home_gravity_sample_cost() {
    let (_reg, t) = make(42);
    let field = crate::gravity::Field::new(t.mass());
    let spawn = t.home_spawn().expect("spawn");
    let n = 20_000;
    let start = std::time::Instant::now();
    let mut acc = 0.0;
    for i in 0..n {
        acc += field.sample(spawn + glam::DVec3::new((i % 7) as f64 * 0.01, 0.0, 0.0)).accel.y;
    }
    println!("{:.2} µs per sample at spawn [{acc:.3}]", start.elapsed().as_secs_f64() * 1e6 / n as f64);
}

/// On a warped twin the air follows the bent grid: the bowed face centre sits ~7 % above the
/// reference cube, and standing there is still in air with a small altitude.
#[test]
fn a_warped_twin_has_air_on_its_bowed_face() {
    let (_reg, t) = make(42);
    let twin = *t.cosmos.bodies().iter().find(|b| b.kind == cosmos::Kind::Twin).unwrap();
    let cosmos::Shape::Cube { half } = twin.shape else { panic!("a twin is a cube") };
    let atlas = t.storage.atlases().iter().find(|a| a.grid.as_ref().is_some_and(|g| g.body == twin.id)).expect("warped");
    let warp = atlas.warp.as_ref().expect("a warp");
    let top = warp.apply(twin.centre_f() + glam::DVec3::new(0.0, half as f64 + 100.0, 0.0));
    assert!(top.y - twin.centre_f().y > half as f64 * 1.03, "the face bows out");
    let alt = t.cosmos.altitude(&twin, top);
    assert!((alt - 100.0).abs() < 1.0, "altitude on the bent grid {alt}");
    assert!(t.cosmos.in_air(top));
}

/// `cargo test --release --lib column_generation_costs -- --ignored --nocapture`: one storage
/// column of four surface chunks on the start world's chart and on a warped twin's face.
#[test]
#[ignore]
fn column_generation_costs() {
    let (_reg, t) = make(42);
    let mut columns = Vec::new();
    let (sx, sz, _) = home_column(&t, Face::PosY, 1_000, 2_000);
    columns.push(("home chart", sx, sz));
    let twin = *t.cosmos.bodies().iter().find(|b| b.kind == cosmos::Kind::Twin).unwrap();
    let cosmos::Shape::Cube { half } = twin.shape else { panic!() };
    let atlas = t.storage.atlases().iter().find(|a| a.grid.as_ref().is_some_and(|g| g.body == twin.id)).unwrap();
    let warp = atlas.warp.as_ref().unwrap();
    let p = warp.apply(twin.centre_f() + glam::DVec3::new(1_000.0, half as f64 + 50.0, 2_000.0));
    let (patch, l) = atlas.find(p).unwrap();
    let (o, _) = atlas.storage_box(patch);
    columns.push(("twin face", (o[0] + l.x as i64) as i32, (o[2] + l.z as i64) as i32));
    for (name, x, z) in columns {
        let h = t.height(x, z);
        let key = ColumnKey { face: Face::PosY, a: x.div_euclid(16), b: z.div_euclid(16) };
        let alt = h.div_euclid(16);
        let _ = t.generate_column(key, alt - 2..=alt + 1);
        let reps: usize = std::env::var("REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        let start = std::time::Instant::now();
        for _ in 0..reps {
            std::hint::black_box(t.generate_column(key, alt - 2..=alt + 1));
        }
        println!("{name}: {:.0} µs per 4-chunk column", start.elapsed().as_secs_f64() * 1e6 / reps as f64);
    }
}
