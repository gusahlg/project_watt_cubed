//! The cosmos: the seeded catalog of every body in the universe — the start cube, the few big
//! worlds far away, their moons, and the sparse asteroid clusters between them. It is the single
//! source the generator paints from, gravity sums (as an analytic mass oracle) and the sky draws
//! distant bodies from. Everything is a pure function of the seed; empty space answers in O(1).
//!
//! Layout rules: big bodies sit on a shell 4.5e8–8e8 blocks from the start cube, pairwise ≥ 4e8
//! apart, inside ±9.2e8; clusters live one per `CELL` (with a small probability) and paint only
//! inside their cell; a cluster's rocks live one per sub-cell of their size class and paint only
//! inside their sub-cell, so any box finds its rocks by looking at the sub-cells it overlaps.

use std::collections::HashMap;

use glam::DVec3;

use super::noise::{hash3, unit};
use crate::gravity::{MassOracle, Primitive, Shape as MassShape, Summary, Visitor};

/// Half-size of the start cube (50,000,000 blocks on a side).
pub const HOME_HALF: i64 = 25_000_000;
/// Centre of the start cube: its +Y face plane is `y = 0`.
pub const HOME_CENTRE: [i64; 3] = [0, -HOME_HALF, 0];
/// Design mean amount per cell of a body's bulk (the generator's bulk mix honours it).
pub const BULK_DENSITY: f64 = 5.0;
/// Most a generated surface rises above (or sinks below) its datum.
pub const RELIEF: i64 = 2_048;

/// Edge of a cluster cell (2^24 blocks).
const CELL: i64 = 1 << 24;
/// Cells per axis searched for clusters (covers ±1e9).
const CELL_SPAN: i64 = 60;
/// Edge of a super-cell grouping clusters for gravity queries (8³ cells).
const SUPER: i64 = CELL * 8;

/// What a body is made like. Content flavour only: physics never reads it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Kind {
    /// The start cube.
    Home,
    /// One of two facing cubes.
    Twin,
    /// A round world of giant forests.
    Verdant,
    /// A hollow round shell around a small glowing core.
    Hollow,
    /// The Hollow's core.
    Ember,
    /// A moon.
    Moon,
}

/// A body's solid geometry (its datum surface; relief rides on top).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Shape {
    /// Axis-aligned cube of half-size `half`.
    Cube { half: i64 },
    /// Ball of radius `r`.
    Ball { r: i64 },
    /// Hollow ball between `inner` and `outer`.
    Shell { outer: i64, inner: i64 },
}

/// One big body (start cube, worlds, moons).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Body {
    /// Stable index into the catalog.
    pub id: u16,
    pub kind: Kind,
    pub centre: [i64; 3],
    pub shape: Shape,
    /// Mean amount per cell of its bulk.
    pub density: f64,
    pub seed: u32,
}

impl Body {
    pub fn centre_f(&self) -> DVec3 {
        DVec3::new(self.centre[0] as f64, self.centre[1] as f64, self.centre[2] as f64)
    }

    /// Bounding radius including relief.
    pub fn reach(&self) -> f64 {
        match self.shape {
            Shape::Cube { half } => (half + RELIEF) as f64 * 3f64.sqrt(),
            Shape::Ball { r } => (r + RELIEF) as f64,
            Shape::Shell { outer, .. } => (outer + RELIEF) as f64,
        }
    }

    /// Altitude of `p` above the nearest datum surface (negative inside the matter).
    pub fn altitude(&self, p: DVec3) -> f64 {
        let d = p - self.centre_f();
        match self.shape {
            Shape::Cube { half } => d.abs().max_element() - half as f64,
            Shape::Ball { r } => d.length() - r as f64,
            Shape::Shell { outer, inner } => {
                let l = d.length();
                if l * 2.0 > (outer + inner) as f64 { l - outer as f64 } else { inner as f64 - l }
            }
        }
    }

    /// Whether any cell of the box `[lo, hi]` (inclusive cells) may hold this body's matter.
    pub fn touches(&self, lo: [i64; 3], hi: [i64; 3]) -> bool {
        let c = self.centre;
        match self.shape {
            Shape::Cube { half } => (0..3).all(|a| hi[a] >= c[a] - half - RELIEF && lo[a] < c[a] + half + RELIEF),
            Shape::Ball { r } | Shape::Shell { outer: r, .. } => {
                let mut d2 = 0i128;
                for a in 0..3 {
                    let d = (c[a] - hi[a].min(c[a]).max(lo[a])) as i128;
                    d2 += d * d;
                }
                let reach = (r + RELIEF) as i128;
                if d2 > reach * reach {
                    return false;
                }
                // A shell's inner cavity holds nothing: skip boxes wholly inside it.
                if let Shape::Shell { inner, .. } = self.shape {
                    let far = (0..3).map(|a| {
                        let d = (c[a] - lo[a]).abs().max((hi[a] + 1 - c[a]).abs()) as i128;
                        d * d
                    });
                    let in2 = (inner - RELIEF) as i128;
                    if far.sum::<i128>() < in2 * in2 {
                        return false;
                    }
                }
                true
            }
        }
    }

    /// The analytic mass primitives of this body (a shell is a ball minus a ball; the rest are one).
    pub fn primitives(&self) -> impl Iterator<Item = Primitive> {
        let c = self.centre_f();
        let (a, b) = match self.shape {
            Shape::Cube { half } => {
                let h = DVec3::splat(half as f64);
                (Primitive::new(MassShape::Box { lo: c - h, hi: c + h }, self.density), None)
            }
            Shape::Ball { r } => (Primitive::new(MassShape::Ball { c, r: r as f64 }, self.density), None),
            Shape::Shell { outer, inner } => (
                Primitive::new(MassShape::Ball { c, r: outer as f64 }, self.density),
                Some(Primitive::new(MassShape::Ball { c, r: inner as f64 }, -self.density)),
            ),
        };
        std::iter::once(a).chain(b)
    }

    /// Bound (per unit G) on the field error from unmodelled relief and caves near the surface:
    /// an infinite slab of the relief's thickness.
    fn relief_error(&self) -> f64 {
        2.0 * std::f64::consts::PI * self.density * RELIEF as f64
    }
}

/// How a cluster's rocks are spread.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Form {
    /// A round swarm, dense in the middle.
    Swarm,
    /// A flattened ring.
    Belt,
    /// An elongated stream.
    Stream,
}

/// One asteroid cluster.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Cluster {
    pub cell: [i32; 3],
    pub centre: DVec3,
    /// Outer radius of its density profile.
    pub radius: f64,
    pub form: Form,
    /// Unit axis (belt normal / stream direction).
    pub axis: DVec3,
    /// Expected number of rocks.
    pub count: f64,
    pub seed: u32,
}

/// What a rock is made of.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum RockKind {
    Rocky,
    Carbon,
    Metallic,
    Icy,
    /// Hollow, crystal-lined.
    Geode,
    /// Carries a small ruin.
    Derelict,
}

/// One asteroid.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Rock {
    pub centre: [i32; 3],
    /// Mean radius.
    pub r: f32,
    /// Per-axis stretch (0.6..1.0) of an ellipsoid.
    pub axes: [f32; 3],
    pub kind: RockKind,
    pub seed: u32,
}

impl Rock {
    /// Mass (amount) of the rock as a uniform ellipsoid of the bulk density.
    pub fn mass(&self) -> f64 {
        let r = self.r as f64;
        4.0 / 3.0 * std::f64::consts::PI * r * r * r * (self.axes[0] * self.axes[1] * self.axes[2]) as f64 * BULK_DENSITY
    }
}

/// Size classes of rocks: sub-cell edge and radius range. A rock (with its 30 % surface bumps) fits
/// inside its sub-cell.
const CLASSES: [(i64, f32, f32, f64); 4] = [
    // (sub-cell edge, min radius, max radius, share of the cluster's rocks)
    (64, 3.0, 22.0, 0.85),
    (512, 22.0, 170.0, 0.13),
    (4096, 170.0, 1400.0, 0.019),
    (32768, 1400.0, 3000.0, 0.001),
];

/// The catalog.
pub struct Cosmos {
    seed: u32,
    bodies: Vec<Body>,
    clusters: Vec<Cluster>,
    by_cell: HashMap<[i32; 3], u32>,
    /// Clusters grouped by super-cell, with their summed expected mass.
    groups: Vec<Group>,
    /// Group index by super-cell (a dense grid over the bounded universe; `u32::MAX` = none), so a
    /// query visits only the super-cells within the law's range.
    group_at: Vec<u32>,
}

/// Super-cells per axis on each side of the origin (covers ±1e9 with a margin).
const GRID_HALF: i64 = 9;
const GRID_SIDE: i64 = 2 * GRID_HALF;

/// The dense-grid slot of super-cell `key`, if it lies in the universe.
fn grid_slot(key: [i64; 3]) -> Option<usize> {
    let k = [key[0] + GRID_HALF, key[1] + GRID_HALF, key[2] + GRID_HALF];
    k.iter().all(|&v| (0..GRID_SIDE).contains(&v)).then(|| (k[0] + GRID_SIDE * (k[1] + GRID_SIDE * k[2])) as usize)
}

/// Clusters of one super-cell.
struct Group {
    summary: Summary,
    members: Vec<u32>,
}

/// A hash-derived unit vector (no trigonometry: normalised integer offsets, rejection sampled).
fn direction(seed: u32, k: i32) -> DVec3 {
    for t in 0..64 {
        let h = |a: i32| unit(hash3(seed, k, t, a)) as f64 * 2.0 - 1.0;
        let v = DVec3::new(h(0), h(1), h(2));
        let l2 = v.length_squared();
        if l2 > 0.01 && l2 <= 1.0 {
            return v / l2.sqrt();
        }
    }
    DVec3::Y
}

/// A position snapped to the chunk grid (multiples of 16), so a body's faces and columns align
/// with chunks.
fn to_i64(v: DVec3) -> [i64; 3] {
    let snap = |x: f64| (x / 16.0).round() as i64 * 16;
    [snap(v.x), snap(v.y), snap(v.z)]
}

impl Cosmos {
    /// The catalog for `seed`, with the asteroid-cluster density scaled by `space` (1 = designed).
    pub fn new(seed: u32, space: f32) -> Self {
        let mut bodies = vec![Body {
            id: 0,
            kind: Kind::Home,
            centre: HOME_CENTRE,
            shape: Shape::Cube { half: HOME_HALF },
            density: BULK_DENSITY,
            seed: hash3(seed, 0, 0, 0),
        }];
        let home = bodies[0].centre_f();
        // Big worlds: (kind, shape) on the far shell.
        let worlds = [
            (Kind::Twin, 6_000_000i64),
            (Kind::Verdant, 8_000_000),
            (Kind::Hollow, 6_000_000),
        ];
        let mut placed: Vec<DVec3> = vec![home];
        let mut k = 1;
        for (i, &(kind, size)) in worlds.iter().enumerate() {
            let mut at = None;
            for attempt in 0..256 {
                let dir = direction(seed ^ 0x51ED_C0DE, (i * 256 + attempt) as i32);
                let dist = 4.5e8 + unit(hash3(seed, 7, i as i32, attempt as i32)) as f64 * 3.5e8;
                let c = home + dir * dist;
                if c.abs().max_element() > 9.2e8 || placed.iter().any(|p| (*p - c).length() < 4.0e8) {
                    continue;
                }
                at = Some(c);
                break;
            }
            let Some(c) = at else { continue };
            placed.push(c);
            let s = hash3(seed, 11, i as i32, 0);
            match kind {
                Kind::Twin => {
                    // Two cubes face to face across a gap along one grid axis.
                    let axis = (s % 3) as usize;
                    let gap = 1_500_000i64;
                    for side in [-1i64, 1] {
                        let mut ctr = to_i64(c);
                        ctr[axis] += side * (gap / 2 + size);
                        bodies.push(Body {
                            id: k,
                            kind: Kind::Twin,
                            centre: ctr,
                            shape: Shape::Cube { half: size },
                            density: BULK_DENSITY,
                            seed: hash3(s, side as i32, 0, 0),
                        });
                        k += 1;
                    }
                }
                Kind::Hollow => {
                    let ctr = to_i64(c);
                    bodies.push(Body {
                        id: k,
                        kind,
                        centre: ctr,
                        shape: Shape::Shell { outer: size, inner: size - 250_000 },
                        density: BULK_DENSITY,
                        seed: s,
                    });
                    k += 1;
                    bodies.push(Body {
                        id: k,
                        kind: Kind::Ember,
                        centre: ctr,
                        shape: Shape::Ball { r: 400_000 },
                        density: BULK_DENSITY,
                        seed: s ^ 0xE3BE,
                    });
                    k += 1;
                }
                _ => {
                    bodies.push(Body {
                        id: k,
                        kind,
                        centre: to_i64(c),
                        shape: Shape::Ball { r: size },
                        density: BULK_DENSITY,
                        seed: s,
                    });
                    k += 1;
                }
            }
        }
        // Moons: two around home, one or two around each world. They sit 5–7 parent reaches out, where
        // the law's finite range has tapered the parent's pull below the moon's own (bodies are
        // anchored, so a moon close to a big world would be a rock its surface slides off).
        let parents: Vec<Body> = bodies.iter().filter(|b| b.kind != Kind::Ember).copied().collect();
        for (pi, parent) in parents.iter().enumerate() {
            let n = if parent.kind == Kind::Home { 2 } else { 1 + (parent.seed % 2) as usize };
            for m in 0..n {
                let s = hash3(seed ^ 0x300E, pi as i32, m as i32, 1);
                let r = 600_000 + (s % 900_000) as i64;
                for attempt in 0..64 {
                    let dir = direction(s, attempt);
                    let dist = parent.reach() * (5.0 + unit(hash3(s, attempt, 2, 3)) as f64 * 2.0) + r as f64;
                    let c = parent.centre_f() + dir * dist;
                    let clear = bodies.iter().all(|b| (b.centre_f() - c).length() > b.reach() + r as f64 * 2.0);
                    if clear && c.abs().max_element() < 9.5e8 {
                        bodies.push(Body { id: k, kind: Kind::Moon, centre: to_i64(c), shape: Shape::Ball { r }, density: BULK_DENSITY, seed: s });
                        k += 1;
                        break;
                    }
                }
            }
        }
        let mut cosmos = Self {
            seed,
            bodies,
            clusters: Vec::new(),
            by_cell: HashMap::new(),
            groups: Vec::new(),
            group_at: vec![u32::MAX; (GRID_SIDE * GRID_SIDE * GRID_SIDE) as usize],
        };
        cosmos.place_clusters(space);
        cosmos
    }

    /// Every big body.
    pub fn bodies(&self) -> &[Body] {
        &self.bodies
    }

    /// The start cube.
    pub fn home(&self) -> &Body {
        &self.bodies[0]
    }

    /// Every asteroid cluster.
    pub fn clusters(&self) -> &[Cluster] {
        &self.clusters
    }

    /// The body whose surface is nearest to `p` among those whose reach holds it.
    pub fn body_at(&self, p: DVec3) -> Option<&Body> {
        self.bodies
            .iter()
            .filter(|b| (b.centre_f() - p).length() <= b.reach() * 1.5)
            .min_by(|a, b| a.altitude(p).abs().total_cmp(&b.altitude(p).abs()))
    }

    fn place_clusters(&mut self, space: f32) {
        if space <= 0.0 {
            return;
        }
        let p = 0.015 * space.min(2.0);
        for x in -CELL_SPAN..CELL_SPAN {
            for y in -CELL_SPAN..CELL_SPAN {
                for z in -CELL_SPAN..CELL_SPAN {
                    let cell = [x as i32, y as i32, z as i32];
                    let h = hash3(self.seed ^ 0xC1A5_7E25, cell[0], cell[1], cell[2]);
                    if unit(h) >= p {
                        continue;
                    }
                    let g = |k: i32| hash3(h, k, 0x5EED, cell[1]);
                    // Radius 1.5e3..4e4 mostly, rare great clusters to 2e5 (cubic bias, no powf).
                    let u = unit(g(1)) as f64;
                    let radius = if unit(g(2)) < 0.06 { 4e4 + u * 1.6e5 } else { 1.5e3 + u * u * u * 3.85e4 };
                    let margin = radius * 1.25 + 4.0 * CLASSES[3].0 as f64;
                    let span = CELL as f64 - 2.0 * margin;
                    if span <= 0.0 {
                        continue;
                    }
                    let base = DVec3::new((x * CELL) as f64, (y * CELL) as f64, (z * CELL) as f64) + DVec3::splat(margin);
                    let centre = base + DVec3::new(unit(g(3)) as f64, unit(g(4)) as f64, unit(g(5)) as f64) * span;
                    if self.bodies.iter().any(|b| (b.centre_f() - centre).length() < b.reach() * 1.5 + radius) {
                        continue;
                    }
                    let form = match g(6) % 7 {
                        0 | 1 => Form::Belt,
                        2 => Form::Stream,
                        _ => Form::Swarm,
                    };
                    let count = 100.0 + unit(g(7)) as f64 * 3900.0;
                    let cluster = Cluster { cell, centre, radius, form, axis: direction(g(8), 0), count, seed: g(9) };
                    self.by_cell.insert(cell, self.clusters.len() as u32);
                    self.clusters.push(cluster);
                }
            }
        }
        // Group clusters by super-cell for gravity.
        let mut by_super: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
        for (i, c) in self.clusters.iter().enumerate() {
            let key = [(c.centre.x as i64).div_euclid(SUPER), (c.centre.y as i64).div_euclid(SUPER), (c.centre.z as i64).div_euclid(SUPER)];
            by_super.entry(key).or_default().push(i as u32);
        }
        let mut keys: Vec<_> = by_super.keys().copied().collect();
        keys.sort_unstable();
        for key in keys {
            let members = by_super.remove(&key).unwrap();
            let mass: f64 = members.iter().map(|&i| self.expected_mass(&self.clusters[i as usize])).sum();
            let com = members
                .iter()
                .map(|&i| self.clusters[i as usize].centre * self.expected_mass(&self.clusters[i as usize]))
                .sum::<DVec3>()
                / mass;
            let centre = DVec3::new(key[0] as f64 + 0.5, key[1] as f64 + 0.5, key[2] as f64 + 0.5) * SUPER as f64;
            let radius = SUPER as f64 * 0.5 * 3f64.sqrt();
            let slot = grid_slot(key).expect("clusters lie inside the universe grid");
            self.group_at[slot] = self.groups.len() as u32;
            self.groups.push(Group { summary: Summary { centre, radius, mass, com }, members });
        }
    }

    /// The density profile of a cluster at `p` (0..1, before normalisation).
    fn profile(c: &Cluster, p: DVec3) -> f64 {
        let d = p - c.centre;
        let r = c.radius;
        match c.form {
            Form::Swarm => (1.0 - d.length_squared() / (r * r)).max(0.0),
            Form::Belt => {
                let h = d.dot(c.axis);
                let rho = (d - c.axis * h).length();
                let (ring, width, thick) = (r * 0.75, r * 0.25, r * 0.06);
                let (a, b) = ((rho - ring) / width, h / thick);
                (1.0 - a * a - b * b).max(0.0)
            }
            Form::Stream => {
                let along = d.dot(c.axis);
                let across = (d - c.axis * along).length();
                let (a, b) = (along / r, across / (r * 0.15));
                (1.0 - a * a - b * b).max(0.0)
            }
        }
    }

    /// Volume integral of the profile (closed forms of the paraboloid ellipsoids).
    fn profile_volume(c: &Cluster) -> f64 {
        let r = c.radius;
        let unit_ball = 8.0 / 15.0 * std::f64::consts::PI; // ∫(1 − |x|²) over the unit ball
        match c.form {
            Form::Swarm => unit_ball * r * r * r,
            // Torus-like: the paraboloid cross-section (π/2·w·t) swept around the ring.
            Form::Belt => std::f64::consts::PI / 2.0 * (r * 0.25) * (r * 0.06) * 2.0 * std::f64::consts::PI * r * 0.75,
            Form::Stream => unit_ball * r * (r * 0.15) * (r * 0.15),
        }
    }

    /// Expected total mass of a cluster.
    fn expected_mass(&self, c: &Cluster) -> f64 {
        CLASSES
            .iter()
            .map(|&(_, lo, hi, share)| {
                // E[r³] for r = lo + (hi − lo)·u³: the cubic bias keeps most rocks small.
                let (lo, hi) = (lo as f64, hi as f64);
                let w = hi - lo;
                let e_r3 = lo * lo * lo + 3.0 * lo * lo * w / 4.0 + 3.0 * lo * w * w / 7.0 + w * w * w / 10.0;
                c.count * share * 4.0 / 3.0 * std::f64::consts::PI * e_r3 * 0.85 * BULK_DENSITY
            })
            .sum()
    }

    /// The rock of class `k` in sub-cell `sub` of cluster `c`, if any.
    fn rock(&self, c: &Cluster, k: usize, sub: [i64; 3]) -> Option<Rock> {
        let (edge, lo, hi, share) = CLASSES[k];
        let mid = DVec3::new(sub[0] as f64 + 0.5, sub[1] as f64 + 0.5, sub[2] as f64 + 0.5) * edge as f64;
        let f = Self::profile(c, mid);
        if f <= 0.0 {
            return None;
        }
        let cell_vol = (edge * edge * edge) as f64;
        let p = c.count * share * f * cell_vol / Self::profile_volume(c);
        let h = hash3(c.seed ^ k as u32, sub[0] as i32, sub[1] as i32, sub[2] as i32);
        if unit(h) as f64 >= p {
            return None;
        }
        let g = |a: i32| hash3(h, a, k as i32, 0x0C);
        let u = unit(g(1));
        let r = lo + (hi - lo) * u * u * u;
        let reach = (r * 1.3).ceil() as i64 + 1;
        let free = (edge - 2 * reach).max(1);
        let centre = std::array::from_fn(|a| (sub[a] * edge + reach + (g(2 + a as i32) as i64 % free)) as i32);
        let axes = std::array::from_fn(|a| 0.6 + 0.4 * unit(g(5 + a as i32)));
        let kind = match g(8) % 100 {
            0..55 => RockKind::Rocky,
            55..70 => RockKind::Carbon,
            70..82 => RockKind::Metallic,
            82..94 => RockKind::Icy,
            94..98 => RockKind::Geode,
            _ => RockKind::Derelict,
        };
        Some(Rock { centre, r, axes, kind, seed: g(9) })
    }

    /// Every rock whose sub-cell overlaps the cell box `[lo, hi]` (inclusive), pushed to `out`.
    pub fn rocks_touching(&self, lo: [i64; 3], hi: [i64; 3], out: &mut Vec<Rock>) {
        let cl = |v: i64| v.div_euclid(CELL) as i32;
        for x in cl(lo[0])..=cl(hi[0]) {
            for y in cl(lo[1])..=cl(hi[1]) {
                for z in cl(lo[2])..=cl(hi[2]) {
                    let Some(&i) = self.by_cell.get(&[x, y, z]) else { continue };
                    let c = &self.clusters[i as usize];
                    let reach = c.radius + CLASSES[3].0 as f64;
                    let near = (0..3).all(|a| hi[a] as f64 >= c.centre[a] - reach && lo[a] as f64 <= c.centre[a] + reach);
                    if !near {
                        continue;
                    }
                    for k in 0..CLASSES.len() {
                        let edge = CLASSES[k].0;
                        let s = |v: i64| v.div_euclid(edge);
                        for sx in s(lo[0])..=s(hi[0]) {
                            for sy in s(lo[1])..=s(hi[1]) {
                                for sz in s(lo[2])..=s(hi[2]) {
                                    if let Some(r) = self.rock(c, k, [sx, sy, sz]) {
                                        out.push(r);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// The rock of each size class whose sub-cell holds `p`, class 0 first.
    /// Stops when `f` returns true. A point lies in one sub-cell per class, and a rock paints
    /// only inside that sub-cell, so this is every rock that could contain `p`.
    pub fn for_rocks_at(&self, p: [i64; 3], mut f: impl FnMut(&Rock) -> bool) {
        let key = [p[0].div_euclid(CELL) as i32, p[1].div_euclid(CELL) as i32, p[2].div_euclid(CELL) as i32];
        let Some(&i) = self.by_cell.get(&key) else { return };
        let c = &self.clusters[i as usize];
        let reach = c.radius + CLASSES[3].0 as f64;
        let near = (0..3).all(|a| {
            let x = p[a] as f64;
            x >= c.centre[a] - reach && x <= c.centre[a] + reach
        });
        if !near {
            return;
        }
        for k in 0..CLASSES.len() {
            let edge = CLASSES[k].0;
            let sub = [p[0].div_euclid(edge), p[1].div_euclid(edge), p[2].div_euclid(edge)];
            if let Some(r) = self.rock(c, k, sub) {
                if f(&r) {
                    return;
                }
            }
        }
    }

    /// Whether the cell box `[lo, hi]` may hold any matter at all (false ⇒ certainly empty).
    pub fn may_hold(&self, lo: [i64; 3], hi: [i64; 3]) -> bool {
        if self.bodies.iter().any(|b| b.touches(lo, hi)) {
            return true;
        }
        let mut rocks = Vec::new();
        self.rocks_touching(lo, hi, &mut rocks);
        !rocks.is_empty()
    }

    /// The bodies whose matter may reach into the cell box `[lo, hi]`.
    pub fn bodies_touching(&self, lo: [i64; 3], hi: [i64; 3]) -> impl Iterator<Item = &Body> {
        self.bodies.iter().filter(move |b| b.touches(lo, hi))
    }
}

/// Radius around a query inside which a cluster's real rocks replace its expected mass.
const OPEN_RADIUS: f64 = 4096.0;

impl MassOracle for Cosmos {
    fn visit(&self, centre: DVec3, reach: f64, v: &mut dyn Visitor) {
        for b in &self.bodies {
            let dist = (b.centre_f() - centre).length();
            if dist - b.reach() >= reach {
                continue;
            }
            for p in b.primitives() {
                v.primitive(&p);
            }
            if b.altitude(centre).abs() < RELIEF as f64 * 4.0 {
                v.error(b.relief_error());
            }
        }
        // Only super-cells within `reach` (plus a cell's half-diagonal) can hold a group in range.
        let span = ((reach + SUPER as f64) / SUPER as f64).ceil() as i64;
        let home = [
            (centre.x / SUPER as f64).floor() as i64,
            (centre.y / SUPER as f64).floor() as i64,
            (centre.z / SUPER as f64).floor() as i64,
        ];
        let near = (-span..=span).flat_map(|x| (-span..=span).flat_map(move |y| (-span..=span).map(move |z| [home[0] + x, home[1] + y, home[2] + z])));
        for key in near {
            let Some(slot) = grid_slot(key) else { continue };
            let gi = self.group_at[slot];
            if gi == u32::MAX {
                continue;
            }
            let g = &self.groups[gi as usize];
            if (g.summary.centre - centre).length() - g.summary.radius >= reach || !v.group(&g.summary) {
                continue;
            }
            for &i in &g.members {
                let c = &self.clusters[i as usize];
                let mass = self.expected_mass(c);
                let summary = Summary { centre: c.centre, radius: c.radius * 1.1, mass, com: c.centre };
                if !v.group(&summary) {
                    continue;
                }
                // Opened: the real rocks near the query, the rest as the expected remainder.
                let lo = std::array::from_fn(|a| (centre[a] - OPEN_RADIUS) as i64);
                let hi = std::array::from_fn(|a| (centre[a] + OPEN_RADIUS) as i64);
                let mut rocks = Vec::new();
                self.rocks_touching(lo, hi, &mut rocks);
                let local = Self::profile(c, centre) / Self::profile_volume(c)
                    * (2.0 * OPEN_RADIUS).powi(3)
                    * mass;
                // The remainder spread over the cluster's ball (never a point: a query near the centre
                // would feel the whole cluster's mass as a singular pull).
                let ball = 4.0 / 3.0 * std::f64::consts::PI * c.radius * c.radius * c.radius;
                v.primitive(&Primitive::new(MassShape::Ball { c: c.centre, r: c.radius }, (mass - local).max(0.0) / ball));
                for r in &rocks {
                    let at = DVec3::new(r.centre[0] as f64, r.centre[1] as f64, r.centre[2] as f64) + DVec3::splat(0.5);
                    v.primitive(&Primitive::new(MassShape::Ball { c: at, r: r.r as f64 }, BULK_DENSITY * 0.85));
                }
                v.error(local * 0.5 / (OPEN_RADIUS * OPEN_RADIUS));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gravity::Field;
    use std::sync::Arc;

    #[test]
    fn the_catalog_is_deterministic_and_well_separated() {
        let a = Cosmos::new(42, 1.0);
        let b = Cosmos::new(42, 1.0);
        assert_eq!(a.bodies, b.bodies);
        assert_eq!(a.clusters.len(), b.clusters.len());
        let worlds: Vec<&Body> = a.bodies.iter().filter(|b| matches!(b.kind, Kind::Twin | Kind::Verdant | Kind::Hollow)).collect();
        assert!(worlds.len() >= 3, "home has company: {:?}", a.bodies.iter().map(|b| b.kind).collect::<Vec<_>>());
        for b in &a.bodies {
            assert!(b.centre.iter().all(|&c| c.abs() < 980_000_000), "{b:?} inside the border");
            assert!(b.centre.iter().all(|&c| c % 16 == 0), "{b:?} on the chunk grid");
        }
        // Moons clear every body.
        for m in a.bodies.iter().filter(|b| b.kind == Kind::Moon) {
            for o in a.bodies.iter().filter(|o| o.id != m.id) {
                assert!((m.centre_f() - o.centre_f()).length() > o.reach(), "moon {m:?} inside {o:?}");
            }
        }
        assert!(!a.clusters.is_empty());
    }

    #[test]
    fn home_spawn_pull_is_the_designed_gravity() {
        let cosmos = Arc::new(Cosmos::new(7, 1.0));
        let field = Field::new(cosmos);
        let s = field.sample(DVec3::new(0.5, 70.0, 0.5));
        let want = crate::player::STANDARD_GRAVITY;
        assert!((s.accel.length() - want).abs() < 0.02 * want, "{} vs {want}", s.accel.length());
        assert!(s.accel.y < 0.0 && (s.accel.x.abs() + s.accel.z.abs()) < 1e-3 * want);
    }

    #[test]
    fn empty_space_is_empty_and_rocks_stay_in_their_cluster() {
        let cosmos = Cosmos::new(42, 1.0);
        let c = cosmos.clusters[0];
        let mut rocks = Vec::new();
        let r = c.radius as i64;
        let ctr = [c.centre.x as i64, c.centre.y as i64, c.centre.z as i64];
        cosmos.rocks_touching([ctr[0] - 256, ctr[1] - 256, ctr[2] - 256], [ctr[0] + 256, ctr[1] + 256, ctr[2] + 256], &mut rocks);
        for rock in &rocks {
            let d = DVec3::new(rock.centre[0] as f64, rock.centre[1] as f64, rock.centre[2] as f64) - c.centre;
            assert!(d.length() < c.radius * 1.3 + 4096.0, "{rock:?} strays from {c:?}");
        }
        // Far from every body and cluster, a box holds nothing.
        let far = [ctr[0] + 4 * r + 9_000_000, ctr[1], ctr[2]];
        let empty_cell = cosmos.by_cell.get(&[(far[0].div_euclid(CELL)) as i32, (far[1].div_euclid(CELL)) as i32, (far[2].div_euclid(CELL)) as i32]).is_none();
        if empty_cell && cosmos.bodies.iter().all(|b| !b.touches(far, [far[0] + 15, far[1] + 15, far[2] + 15])) {
            assert!(!cosmos.may_hold(far, [far[0] + 15, far[1] + 15, far[2] + 15]));
        }
    }

    #[test]
    fn a_rock_paints_only_inside_its_sub_cell() {
        let cosmos = Cosmos::new(3, 2.0);
        let c = cosmos.clusters[0];
        let mut found = 0;
        for k in 0..CLASSES.len() {
            let edge = CLASSES[k].0;
            let base = [(c.centre.x as i64).div_euclid(edge), (c.centre.y as i64).div_euclid(edge), (c.centre.z as i64).div_euclid(edge)];
            for dx in -6..=6 {
                for dz in -6..=6 {
                    let sub = [base[0] + dx, base[1], base[2] + dz];
                    if let Some(r) = cosmos.rock(&c, k, sub) {
                        found += 1;
                        let reach = (r.r * 1.3).ceil() as i64 + 1;
                        for a in 0..3 {
                            let lo = sub[a] * edge;
                            assert!(r.centre[a] as i64 - reach >= lo && r.centre[a] as i64 + reach <= lo + edge, "{r:?} leaves its sub-cell");
                        }
                    }
                }
            }
        }
        let _ = found;
    }

    /// `cargo test --lib cosmos_report -- --ignored --nocapture`: the catalog of a seed.
    #[test]
    #[ignore]
    fn cosmos_report() {
        let seed = std::env::var("WATT_COSMOS_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(42);
        let cosmos = Cosmos::new(seed, 1.0);
        for b in cosmos.bodies() {
            let d = (b.centre_f() - cosmos.home().centre_f()).length();
            println!("{:>2} {:?} at {:?} {:?} — {:.3e} blocks from home", b.id, b.kind, b.centre, b.shape, d);
        }
        let n = cosmos.clusters().len();
        let near = cosmos.clusters().iter().filter(|c| (c.centre - cosmos.home().centre_f()).length() < 2.0e8).count();
        let nearest = cosmos.clusters().iter().map(|c| (c.centre - DVec3::new(0.0, 0.0, 0.0)).length()).fold(f64::MAX, f64::min);
        println!("{n} clusters, {near} within 2e8 of home, nearest to spawn {nearest:.3e}");
    }


    /// `cargo test --release --lib gravity_tour -- --ignored --nocapture`: the field at the places where
    /// matter-derived gravity gets interesting.
    #[test]
    #[ignore]
    fn gravity_tour() {
        let cosmos = Arc::new(Cosmos::new(42, 1.0));
        let field = Field::new(cosmos.clone());
        let g0 = crate::player::STANDARD_GRAVITY;
        let home = cosmos.home().centre_f();
        let h = HOME_HALF as f64;
        let mut stops: Vec<(String, DVec3)> = vec![
            ("home +Y face centre (spawn)".into(), home + DVec3::new(0.0, h + 70.0, 0.0)),
            ("1,000,000 blocks from spawn".into(), home + DVec3::new(1.0e6, h + 70.0, 0.0)),
            ("halfway to an edge".into(), home + DVec3::new(0.5 * h, h + 70.0, 0.0)),
            ("an edge midpoint".into(), home + DVec3::new(h, h + 70.0, 0.0)),
            ("a corner".into(), home + DVec3::splat(h + 70.0)),
            ("1,000,000 below spawn".into(), home + DVec3::new(0.0, h - 1.0e6, 0.0)),
            ("the cube's centre".into(), home),
            ("10,000,000 above spawn".into(), home + DVec3::new(0.0, h + 1.0e7, 0.0)),
        ];
        let twins: Vec<&Body> = cosmos.bodies().iter().filter(|b| b.kind == Kind::Twin).collect();
        if let [a, b] = twins[..] {
            let mid = (a.centre_f() + b.centre_f()) * 0.5;
            stops.push(("the Twins' canyon midpoint".into(), mid));
            let d = (b.centre_f() - a.centre_f()).normalize();
            stops.push(("a Twin's facing surface".into(), mid - d * (750_000.0 - 70.0)));
        }
        for b in cosmos.bodies() {
            match (b.kind, b.shape) {
                (Kind::Verdant, Shape::Ball { r }) => stops.push(("Verdance's surface".into(), b.centre_f() + DVec3::new(0.0, r as f64 + 70.0, 0.0))),
                (Kind::Hollow, Shape::Shell { outer, inner }) => {
                    stops.push(("the Hollow's outer surface".into(), b.centre_f() + DVec3::new(0.0, outer as f64 + 70.0, 0.0)));
                    stops.push(("the Hollow's inner surface".into(), b.centre_f() + DVec3::new(0.0, inner as f64 - 70.0, 0.0)));
                    stops.push(("halfway to the Hollow's core".into(), b.centre_f() + DVec3::new(0.0, inner as f64 * 0.5, 0.0)));
                }
                (Kind::Moon, Shape::Ball { r }) if (b.centre_f() - home).length() < 4.0e8 => {
                    stops.push((format!("moon {} (r {r})", b.id), b.centre_f() + DVec3::new(0.0, r as f64 + 70.0, 0.0)))
                }
                _ => {}
            }
        }
        println!("| where | pull (% of spawn) | tilt from local vertical |\n|---|---|---|");
        for (name, p) in stops {
            let s = field.sample(p);
            let g = s.accel.length();
            // Local vertical: a cube face's normal, a ball's radius.
            let up = cosmos.body_at(p).map(|b| match b.shape {
                Shape::Cube { .. } => crate::coord::Face::from_dominant(p - b.centre_f()).dvec(),
                _ => (p - b.centre_f()).normalize(),
            });
            let up = up.unwrap_or(DVec3::Y);
            let tilt = if g > 1e-9 { format!("{:.2}°", (-s.accel / g).dot(up).clamp(-1.0, 1.0).acos().to_degrees()) } else { "—".into() };
            println!("| {name} | {:.4} % | {tilt} |", 100.0 * g / g0);
        }
    }


    #[test]
    fn a_cluster_never_pulls_like_a_point_at_its_centre() {
        let cosmos = Arc::new(Cosmos::new(42, 1.0));
        let field = Field::new(cosmos.clone());
        let c = cosmos.clusters()[0];
        // Close to the centre the pull stays tiny compared with a planet's surface pull.
        let s = field.sample(c.centre + DVec3::new(7.0, 3.0, -5.0));
        assert!(s.accel.length() < 0.05 * crate::player::STANDARD_GRAVITY, "{}", s.accel.length());
    }


    /// `cargo test --release --lib gravity_sample_cost -- --ignored --nocapture`: what one player
    /// gravity sample costs against the whole cosmos (bodies + cluster groups).
    #[test]
    #[ignore]
    fn gravity_sample_cost() {
        let cosmos = Arc::new(Cosmos::new(42, 1.0));
        let field = Field::new(cosmos.clone());
        let mut p = DVec3::new(0.5, 70.0, 0.5);
        let n = 20_000;
        let t = std::time::Instant::now();
        let mut acc = 0.0;
        for i in 0..n {
            p.x += (i % 7) as f64 * 0.01;
            acc += field.sample(p).accel.y;
        }
        let per = t.elapsed().as_secs_f64() / n as f64;
        println!("{:.2} µs per sample at spawn ({} cluster groups) [{acc:.3}]", per * 1e6, cosmos.groups.len());
        let moon = cosmos.bodies().iter().find(|b| b.kind == Kind::Moon).unwrap();
        let q = moon.centre_f() + DVec3::new(0.0, 2.0e6, 0.0);
        let t = std::time::Instant::now();
        for _ in 0..2_000 {
            acc += field.sample(q).accel.y;
        }
        println!("{:.2} µs per sample above a moon [{acc:.3}]", t.elapsed().as_secs_f64() / 2_000.0 * 1e6);
    }

}
