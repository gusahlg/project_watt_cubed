//! Asteroids. Every rock from the catalog is an ellipsoid: its three axes are assigned to the
//! world axes by a hash, the surface is bumped by 3-D noise by at most 30 % of the radius (the
//! margin the catalog reserved), bigger rocks carry small craters, and the interior follows the
//! kind. A rock never paints outside that reserved box, so it stays in its sub-cell.

use super::cosmos::{Cosmos, Rock, RockKind};
use super::noise::{hash3, perlin3, unit};
use super::Materials;
use crate::block::registry::{AIR, BlockId};
use crate::world::chunk::{CHUNK_SIZE, CHUNK_VOLUME, Chunk, ChunkData};

/// The six ways to assign the rock's three stretch axes to the world axes.
const PERMS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

/// Chebyshev reach the catalog reserved: `(r · 1.3)` rounded up, plus one cell.
/// `r` stays `f32` so this matches the placement test exactly.
pub(super) fn reach(rock: &Rock) -> i64 {
    (rock.r * 1.3).ceil() as i64 + 1
}

/// Whether any rock's reserved box meets the inclusive cell box.
pub(super) fn any_overlap(cosmos: &Cosmos, lo: [i64; 3], hi: [i64; 3]) -> bool {
    let mut rocks = Vec::new();
    cosmos.rocks_touching(lo, hi, &mut rocks);
    rocks.iter().any(|r| overlaps(r, lo, hi))
}

/// Whether `rock` may paint a cell of the inclusive cell box: its reserved box meets the box and,
/// but for a derelict's ruin, the box's nearest cell lies within the bumped ellipsoid's outer
/// bound (no cell beyond `ell = 1.30` is ever painted).
pub(super) fn overlaps(rock: &Rock, lo: [i64; 3], hi: [i64; 3]) -> bool {
    let reach = reach(rock);
    let in_box = (0..3).all(|a| {
        let c = i64::from(rock.centre[a]);
        hi[a] >= c - reach && lo[a] <= c + reach
    });
    if !in_box || rock.kind == RockKind::Derelict {
        return in_box;
    }
    let near: [f64; 3] = std::array::from_fn(|a| {
        let c = i64::from(rock.centre[a]);
        0i64.clamp(lo[a] - c, hi[a] - c) as f64
    });
    ell(near, semis(rock)) <= 1.30 + 1e-9
}

/// A space chunk (no body reaches it) whose minimum cell is `o`: every rock that may paint it,
/// smallest class first, then cell by cell the first rock that paints (equal to [`block`]).
pub(super) fn fill(cosmos: &Cosmos, m: &Materials, o: [i64; 3]) -> ChunkData {
    let n = CHUNK_SIZE as i64;
    let hi = [o[0] + n - 1, o[1] + n - 1, o[2] + n - 1];
    let mut rocks = Vec::new();
    cosmos.rocks_touching(o, hi, &mut rocks);
    rocks.retain(|r| r.r >= 1.0 && overlaps(r, o, hi));
    if rocks.is_empty() {
        return ChunkData::Uniform(AIR);
    }
    let mut cells = Box::new([AIR; CHUNK_VOLUME]);
    for lz in 0..CHUNK_SIZE {
        for ly in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let p = [o[0] + lx as i64, o[1] + ly as i64, o[2] + lz as i64];
                if let Some(id) = rocks.iter().find_map(|r| paint(r, m, p)) {
                    cells[Chunk::index(lx, ly, lz)] = id;
                }
            }
        }
    }
    ChunkData::from_cells(cells)
}

/// The block at cell `p` if `rock` paints it (`None` when `p` is outside the shape).
/// `Some(AIR)` is a cavity, a crater bowl or a cave — the rock still owns the cell.
pub(super) fn paint(rock: &Rock, m: &Materials, p: [i64; 3]) -> Option<BlockId> {
    if rock.r < 1.0 {
        return None;
    }
    let q = [
        p[0] - i64::from(rock.centre[0]),
        p[1] - i64::from(rock.centre[1]),
        p[2] - i64::from(rock.centre[2]),
    ];
    if q.iter().any(|v| v.abs() > reach(rock)) {
        return None;
    }
    let semi = semis(rock);
    let qf = [q[0] as f64, q[1] as f64, q[2] as f64];
    let e = ell(qf, semi);
    let mut limit = 1.0 + bump(rock, qf);
    craters(rock, semi, qf, &mut limit);
    limit = limit.clamp(0.42, 1.30);
    let spike = on_spike(rock, semi, qf, e);
    if let Some(id) = ruin(rock, m, q, semi) {
        return Some(id);
    }
    if e > limit && !spike {
        return None;
    }
    let f = e / limit.max(0.25);
    if cave(rock, qf, f) {
        return Some(AIR);
    }
    if spike && e > limit * 0.92 {
        return Some(m.ice);
    }
    Some(interior(rock, m, q, qf, f))
}

/// The block a rock paints at `p`, or air when no rock contains it.
pub(super) fn block(cosmos: &Cosmos, m: &Materials, p: [i64; 3]) -> BlockId {
    let mut hit = AIR;
    let mut any = false;
    cosmos.for_rocks_at(p, |rock| {
        if let Some(id) = paint(rock, m, p) {
            hit = id;
            any = true;
            true
        } else {
            false
        }
    });
    if any { hit } else { AIR }
}

fn semis(rock: &Rock) -> [f64; 3] {
    let perm = PERMS[(hash3(rock.seed, 0xA715, 0, 1) % 6) as usize];
    let r = rock.r as f64;
    [r * f64::from(rock.axes[perm[0]]), r * f64::from(rock.axes[perm[1]]), r * f64::from(rock.axes[perm[2]])]
}

fn ell(q: [f64; 3], semi: [f64; 3]) -> f64 {
    let e = q[0] / semi[0];
    let f = q[1] / semi[1];
    let g = q[2] / semi[2];
    (e * e + f * f + g * g).sqrt()
}

/// Surface displacement in ellipsoid radii, clamped to ±30 % so the rock stays in its box.
fn bump(rock: &Rock, q: [f64; 3]) -> f64 {
    let s = (rock.r as f64 * 0.34).max(2.0);
    let n = f64::from(perlin3(rock.seed ^ 0x00B4_B000, q[0] / s, q[1] / s, q[2] / s));
    let m = f64::from(perlin3(rock.seed ^ 0x00B4_C000, q[0] / (s * 0.41), q[1] / (s * 0.41), q[2] / (s * 0.41)));
    ((0.72 * n + 0.28 * m) * 0.30).clamp(-0.30, 0.30)
}

fn craters(rock: &Rock, semi: [f64; 3], q: [f64; 3], limit: &mut f64) {
    if rock.r < 48.0 {
        return;
    }
    let n = (2 + (rock.r as u32) / 180).min(7);
    for i in 0..n {
        let d = dir_from(rock.seed ^ 0xC7A7_E200, i);
        let c = [d[0] * semi[0], d[1] * semi[1], d[2] * semi[2]];
        let u = f64::from(unit(hash3(rock.seed, i as i32, 4, 0xC)));
        let rad = rock.r as f64 * (0.06 + 0.09 * u);
        let dx = q[0] - c[0];
        let dy = q[1] - c[1];
        let dz = q[2] - c[2];
        let t2 = (dx * dx + dy * dy + dz * dz) / (rad * rad);
        if t2 < 1.0 {
            *limit -= 0.14 * (1.0 - t2);
        } else if t2 < 1.96 {
            let t = t2.sqrt();
            let f = (1.4 - t) / 0.4;
            *limit += 0.045 * f * f;
        }
    }
}

fn on_spike(rock: &Rock, semi: [f64; 3], q: [f64; 3], e: f64) -> bool {
    if rock.kind != RockKind::Icy || e < 0.78 || e > 1.22 {
        return false;
    }
    let n = (3 + (rock.r as u32) / 50).min(7);
    for i in 0..n {
        let d = dir_from(rock.seed ^ 0x51CE_0000, i);
        let qn = [q[0] / semi[0], q[1] / semi[1], q[2] / semi[2]];
        let dot = (qn[0] * d[0] + qn[1] * d[1] + qn[2] * d[2]) / e;
        if dot > 0.986 {
            return true;
        }
    }
    false
}

/// A unit direction from the rock's hash. Falls back to an axis when the draw misses the sphere.
fn dir_from(seed: u32, i: u32) -> [f64; 3] {
    let h = |k: i32| f64::from(unit(hash3(seed, i as i32, k, 0xD1A))) * 2.0 - 1.0;
    let v = [h(1), h(2), h(3)];
    let l2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if l2 > 0.08 && l2 <= 1.0 {
        let inv = 1.0 / l2.sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        const AX: [[f64; 3]; 6] = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        AX[(i as usize) % 6]
    }
}

fn cave(rock: &Rock, q: [f64; 3], f: f64) -> bool {
    if rock.r <= 1000.0 || rock.kind == RockKind::Geode || !(0.32..0.70).contains(&f) {
        return false;
    }
    let s = 56.0;
    let a = f64::from(perlin3(rock.seed ^ 0xCA7E_0001, q[0] / s, q[1] / s, q[2] / s));
    let b = f64::from(perlin3(rock.seed ^ 0xCA7E_0002, q[0] / s, q[1] / s, q[2] / s));
    a * a + b * b < 0.05
}

fn noise(rock: &Rock, salt: u32, q: [f64; 3], scale: f64) -> f32 {
    perlin3(rock.seed ^ salt, q[0] / scale, q[1] / scale, q[2] / scale)
}

fn interior(rock: &Rock, m: &Materials, q: [i64; 3], qf: [f64; 3], f: f64) -> BlockId {
    let band = (hash3(rock.seed, q[0] as i32, q[1] as i32, 3) % 4) as usize;
    match rock.kind {
        RockKind::Rocky | RockKind::Derelict => {
            if f > 0.84 {
                m.regolith
            } else if f > 0.48 {
                m.rock[band]
            } else {
                m.basalt
            }
        }
        RockKind::Carbon => {
            if (0.36..0.78).contains(&f) && noise(rock, 0xC2, qf, 5.0) > 0.52 {
                return AIR;
            }
            if f > 0.74 {
                if noise(rock, 0xA5, qf, 7.0) > 0.35 { m.ash } else { m.cinder }
            } else if f < 0.32 {
                m.obsidian
            } else {
                m.basalt
            }
        }
        RockKind::Metallic => {
            if f > 0.83 {
                m.rust
            } else if (0.12..0.86).contains(&f) && noise(rock, 0xA11, qf, 16.0).abs() < 0.12 {
                if noise(rock, 0x60D, qf, 48.0) > 0.2 { m.gold } else { m.copper }
            } else if f > 0.5 {
                m.rock[1]
            } else {
                m.basalt
            }
        }
        RockKind::Icy => {
            if f > 0.80 { m.frost } else { m.ice }
        }
        RockKind::Geode => {
            if f < 0.42 {
                AIR
            } else if f < 0.50 {
                m.glowshroom
            } else if f < 0.63 {
                m.crystal
            } else if f > 0.86 {
                m.regolith
            } else {
                m.rock[band]
            }
        }
    }
}

/// A broken monolith, or a plank-and-rust frame with a lamp, sitting on the surface and staying
/// inside the reserved box (the posts are a fraction of the radius).
fn ruin(rock: &Rock, m: &Materials, q: [i64; 3], semi: [f64; 3]) -> Option<BlockId> {
    if rock.kind != RockKind::Derelict {
        return None;
    }
    let dir = dir_from(rock.seed ^ 0x0D1C_0000, 0);
    let anchor = [
        (dir[0] * semi[0]).round() as i64,
        (dir[1] * semi[1]).round() as i64,
        (dir[2] * semi[2]).round() as i64,
    ];
    let axis = (0..3).max_by(|&a, &b| dir[a].abs().total_cmp(&dir[b].abs())).unwrap();
    let sign: i64 = if dir[axis] >= 0.0 { 1 } else { -1 };
    let rel = [q[0] - anchor[0], q[1] - anchor[1], q[2] - anchor[2]];
    let along = rel[axis] * sign;
    let (t0, t1) = match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let (u, v) = (rel[t0], rel[t1]);
    let tall = ((rock.r as f64 * 0.18).round() as i64).clamp(2, 9);
    if !(0..=tall).contains(&along) {
        return None;
    }
    if rock.seed % 2 == 0 {
        if u.abs() > 1 || v.abs() > 1 || (along >= tall - 1 && u == 1 && v == 1) {
            return None;
        }
        return Some(if (along + u.abs()) % 2 == 0 { m.obsidian } else { m.slate });
    }
    let corner = u.abs() == 2 && v.abs() == 2;
    let edge = (u.abs() == 2 && v.abs() <= 2) || (v.abs() == 2 && u.abs() <= 2);
    if corner {
        return Some(m.rust);
    }
    if edge && (along == 0 || along == tall || along % 3 == 0) {
        return Some(m.plank);
    }
    if u == 0 && v == 0 && along == tall / 2 {
        return Some(m.lamp);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{Materials, Rock, RockKind, dir_from, ell, on_spike, paint, reach, semis};
    use crate::block::registry::{AIR, BlockRegistry};

    fn mats() -> Materials {
        let mut reg = BlockRegistry::with_builtins();
        Materials::intern(&mut reg)
    }

    fn rock(kind: RockKind, r: f32, seed: u32, axes: [f32; 3]) -> Rock {
        Rock { centre: [0, 0, 0], r, axes, kind, seed }
    }

    fn count(rock: &Rock, m: &Materials, step: i64) -> [u32; 8] {
        // air, regolith, rock/basalt, cinder/ash/obsidian, rust, copper/gold, frost, ice, plus crystal/glow via the last slots
        let mut n = [0u32; 8];
        let lim = reach(rock);
        let mut x = -lim;
        while x <= lim {
            let mut y = -lim;
            while y <= lim {
                let mut z = -lim;
                while z <= lim {
                    if let Some(id) = paint(rock, m, [x, y, z]) {
                        let slot = if id == AIR {
                            0
                        } else if id == m.regolith {
                            1
                        } else if id == m.basalt || m.rock.contains(&id) {
                            2
                        } else if id == m.cinder || id == m.ash || id == m.obsidian {
                            3
                        } else if id == m.rust {
                            4
                        } else if id == m.copper || id == m.gold {
                            5
                        } else if id == m.frost {
                            6
                        } else if id == m.ice || id == m.crystal || id == m.glowshroom {
                            7
                        } else {
                            2
                        };
                        n[slot] += 1;
                    }
                    z += step;
                }
                y += step;
            }
            x += step;
        }
        n
    }

    #[test]
    fn every_kind_has_its_interior_and_a_great_rock_has_caves() {
        let m = mats();
        let axes = [0.8, 0.95, 0.7];
        let rocky = count(&rock(RockKind::Rocky, 80.0, 3, axes), &m, 8);
        assert!(rocky[1] > 0 && rocky[2] > 0, "rocky crust and core {rocky:?}");

        let carbon = count(&rock(RockKind::Carbon, 80.0, 5, axes), &m, 6);
        assert!(carbon[0] > 0, "carbon crumbles");
        assert!(carbon[2] > 0 && carbon[3] > 0, "carbon core and crust {carbon:?}");

        let metal = count(&rock(RockKind::Metallic, 90.0, 8, axes), &m, 5);
        assert!(metal[4] > 0, "rust crust");
        assert!(metal[5] > 0, "copper or gold veins {metal:?}");

        let icy = rock(RockKind::Icy, 70.0, 4, [0.9, 0.75, 0.85]);
        let ice_n = count(&icy, &m, 8);
        assert!(ice_n[6] > 0 && ice_n[7] > 0, "frost crust and ice core {ice_n:?}");
        let n = (3 + (icy.r as u32) / 50).min(7);
        let mut spike = false;
        for i in 0..n {
            let d = dir_from(icy.seed ^ 0x51CE_0000, i);
            let semi = semis(&icy);
            for scale in [900, 1000, 1080, 1160] {
                let q = [
                    (d[0] * semi[0] * f64::from(scale) / 1000.0).round() as i64,
                    (d[1] * semi[1] * f64::from(scale) / 1000.0).round() as i64,
                    (d[2] * semi[2] * f64::from(scale) / 1000.0).round() as i64,
                ];
                if paint(&icy, &m, q) == Some(m.ice) && on_spike(&icy, semi, [q[0] as f64, q[1] as f64, q[2] as f64], ell([q[0] as f64, q[1] as f64, q[2] as f64], semi)) {
                    spike = true;
                }
            }
        }
        assert!(spike, "an ice spike");

        let geode = rock(RockKind::Geode, 120.0, 6, axes);
        assert_eq!(paint(&geode, &m, [0, 0, 0]), Some(AIR), "the geode is hollow");
        let (mut crystal, mut glow) = (false, false);
        for x in 0..=reach(&geode) {
            match paint(&geode, &m, [x, 0, 0]) {
                Some(id) if id == m.crystal => crystal = true,
                Some(id) if id == m.glowshroom => glow = true,
                _ => {}
            }
        }
        assert!(crystal && glow, "crystal lining and glowing tips");

        for seed in [2u32, 3] {
            let d = rock(RockKind::Derelict, 55.0, seed, axes);
            let anchor_dir = dir_from(d.seed ^ 0x0D1C_0000, 0);
            let semi = semis(&d);
            let anchor = [
                (anchor_dir[0] * semi[0]).round() as i64,
                (anchor_dir[1] * semi[1]).round() as i64,
                (anchor_dir[2] * semi[2]).round() as i64,
            ];
            let (mut obsidian, mut slate, mut plank, mut rust, mut lamp) = (false, false, false, false, false);
            for x in -10..=10 {
                for y in -10..=10 {
                    for z in -10..=10 {
                        match paint(&d, &m, [anchor[0] + x, anchor[1] + y, anchor[2] + z]) {
                            Some(id) if id == m.obsidian => obsidian = true,
                            Some(id) if id == m.slate => slate = true,
                            Some(id) if id == m.plank => plank = true,
                            Some(id) if id == m.rust => rust = true,
                            Some(id) if id == m.lamp => lamp = true,
                            _ => {}
                        }
                    }
                }
            }
            if seed % 2 == 0 {
                assert!(obsidian && slate, "a broken monolith");
            } else {
                assert!(plank && rust && lamp, "a plank-and-rust frame with a lamp");
            }
        }

        let great = rock(RockKind::Rocky, 1_200.0, 11, [0.85, 0.9, 0.8]);
        let (mut cave_air, mut cave_stone) = (0u32, 0u32);
        for x in (-700..=700).step_by(40) {
            for y in (-700..=700).step_by(40) {
                for z in (-700..=700).step_by(40) {
                    match paint(&great, &m, [x, y, z]) {
                        Some(AIR) => cave_air += 1,
                        Some(_) => cave_stone += 1,
                        None => {}
                    }
                }
            }
        }
        assert!(cave_air > 0 && cave_stone > 0, "a great rock has caves and stone ({cave_air} air, {cave_stone} stone)");

        // The long axis is assigned to one world axis by the hash.
        let long = rock(RockKind::Rocky, 40.0, 1, [1.0, 0.6, 0.62]);
        let mut far = [0i64; 3];
        for a in 0..3 {
            for d in (0..=reach(&long)).rev() {
                let mut p = [0i64; 3];
                p[a] = d;
                if paint(&long, &m, p).is_some() {
                    far[a] = d;
                    break;
                }
            }
        }
        let max = *far.iter().max().unwrap();
        let min = *far.iter().min().unwrap();
        assert!(max > min + min / 5, "axes are permuted, extents {far:?}");
    }
}
