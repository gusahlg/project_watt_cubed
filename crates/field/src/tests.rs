//! The engine's guarantees: thread count, tiling and query order never change a byte, and the
//! cube-sphere's seams are exact.

use super::*;

/// A non-linear test cell: a value and a salt that changes every pass.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Cell {
    v: i32,
    salt: u32,
}

/// Pull toward the neighbour mean, a quarter of the way up to the largest neighbour, plus a hash kick.
struct Mix;

impl Rule<Cell> for Mix {
    const RADIUS: u8 = 1;
    fn apply(&self, prev: &[Cell], i: usize, nb: &[u32]) -> Cell {
        let c = prev[i];
        let (mut sum, mut max) = (c.v as i64 * 2, c.v);
        for &n in nb {
            let v = prev[n as usize].v;
            sum += v as i64;
            max = max.max(v);
        }
        let mean = (sum / (nb.len() as i64 + 2)) as i32;
        let kick = (hash32_3(c.salt, mean, 0, 0, 1) & 0xFF) as i32 - 128;
        Cell { v: mean + (max - mean) / 4 + kick, salt: c.salt.wrapping_mul(0x9E37_79B9).wrapping_add(1) }
    }
}

fn prior(p: [i64; 3]) -> Cell {
    let h = hash32_3(7, p[0] as i32, p[1] as i32, p[2] as i32, 0);
    Cell { v: uniform_q16(h), salt: h }
}

fn box_prior(top: &Box3, lo: [i64; 3]) -> Vec<Cell> {
    (0..top.len()).map(|i| prior(std::array::from_fn(|a| lo[a] + top.coords(i)[a] as i64))).collect()
}

#[test]
fn thread_count_does_not_change_the_bytes() {
    let top = Box3::new([40, 36, 33], true);
    let start = box_prior(&top, [0, 0, 0]);
    let mut one = start.clone();
    run(&top, &mut one, &Mix, 6, 1);
    for threads in [2, 3, 4, 7, 16] {
        let mut many = start.clone();
        run(&top, &mut many, &Mix, 6, threads);
        assert_eq!(one, many, "{threads} threads");
    }
    let sphere = Sphere::get(64);
    let start: Vec<Cell> = (0..sphere.len()).map(|i| prior([sphere.owner(i) as i64, 0, 0])).collect();
    let mut one = start.clone();
    run(sphere, &mut one, &Mix, 5, 1);
    for threads in [2, 5] {
        let mut many = start.clone();
        run(sphere, &mut many, &Mix, 5, threads);
        assert_eq!(one, many, "sphere, {threads} threads");
    }
}

#[test]
fn an_apron_tile_equals_the_whole_grid() {
    let passes = 5;
    let lo = [-20i64, -20, -20];
    let top = Box3::new([40, 40, 40], true);
    let mut whole = box_prior(&top, lo);
    run(&top, &mut whole, &Mix, passes, 4);
    let at = |p: [i64; 3]| whole[top.index(std::array::from_fn(|a| (p[a] - lo[a]) as u32))];
    for (t_lo, size) in [([-3i64, 2, -7], [6u32, 5, 4]), ([-15, -15, -15], [30, 30, 30]), ([4, -9, 0], [1, 1, 1])] {
        let t = tile(&Mix, prior, t_lo, size, passes, 2);
        let mut k = 0;
        for z in 0..size[2] as i64 {
            for y in 0..size[1] as i64 {
                for x in 0..size[0] as i64 {
                    assert_eq!(t[k], at([t_lo[0] + x, t_lo[1] + y, t_lo[2] + z]), "tile {t_lo:?} cell {x},{y},{z}");
                    k += 1;
                }
            }
        }
    }
}

#[test]
fn query_order_does_not_change_values() {
    let nodes: Vec<[i64; 3]> = (0..40).map(|k| [k * 7 % 23 - 11, k * 5 % 17 - 8, k % 9 - 4]).collect();
    let forward: Vec<Cell> = nodes.iter().map(|&p| point(&Mix, prior, p, 4)).collect();
    let mut backward: Vec<Cell> = nodes.iter().rev().map(|&p| point(&Mix, prior, p, 4)).collect();
    backward.reverse();
    assert_eq!(forward, backward);
    // Overlapping tiles, built in either order, agree on their overlap and with the points.
    let a = tile(&Mix, prior, [0, 0, 0], [8, 8, 8], 4, 1);
    let b = tile(&Mix, prior, [4, 4, 4], [8, 8, 8], 4, 3);
    for z in 4..8u32 {
        for y in 4..8u32 {
            for x in 4..8u32 {
                let ia = (x + 8 * (y + 8 * z)) as usize;
                let ib = (x - 4 + 8 * (y - 4 + 8 * (z - 4))) as usize;
                assert_eq!(a[ia], b[ib]);
            }
        }
    }
    assert_eq!(a[(1 + 8 * (2 + 8 * 3)) as usize], point(&Mix, prior, [1, 2, 3], 4));
}

#[test]
fn sphere_seams_are_exact() {
    for g in [4u32, 32] {
        let s = Sphere::get(g);
        let owners: Vec<usize> = (0..s.len()).filter(|&i| s.owns(i)).collect();
        assert_eq!(owners.len(), (6 * g * g + 2) as usize, "G={g}");
        let mut nb = [0u32; MAX_NEIGHBOURS];
        let mut corners = 0;
        let lists: Vec<Vec<u32>> = owners
            .iter()
            .map(|&i| {
                let n = s.neighbours(i, &mut nb);
                nb[..n].to_vec()
            })
            .collect();
        for (k, &i) in owners.iter().enumerate() {
            let list = &lists[k];
            assert!(list.len() == 8 || list.len() == 6, "node {i} has {} neighbours", list.len());
            corners += (list.len() == 6) as u32;
            for &j in list {
                assert!(s.owns(j as usize), "neighbours are owners");
                let back = &lists[owners.binary_search(&(j as usize)).unwrap()];
                assert!(back.contains(&(i as u32)), "{i} -> {j} is one-way");
            }
        }
        assert_eq!(corners, 8);
        // Every node's direction locates back onto a node at the same point.
        for i in 0..s.len() {
            let (f, a, b) = s.locate(s.dirs[i]);
            let n = s.node(f, a.round() as u32, b.round() as u32);
            assert_eq!(s.owner(n), s.owner(i), "node {i}");
        }
        // After a run, every glued copy holds its owner's value.
        let mut cells: Vec<Cell> = (0..s.len()).map(|i| prior([s.owner(i) as i64, 1, 2])).collect();
        run(s, &mut cells, &Mix, 3, 2);
        for i in 0..s.len() {
            assert_eq!(cells[i], cells[s.owner(i)]);
        }
    }
}
