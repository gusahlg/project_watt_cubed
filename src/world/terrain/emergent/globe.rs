//! Stage 8 (lab prototype): a body's history on a cube-sphere of its surface. Plates from the
//! body's heat, relief from plate motion inside its strength budget, basins from its recorded
//! impacts and a late flux, climate from the real sun path, erosion, seas and life, then a colour
//! per node and a min/max mip. Local phases are Jacobi rules on the field engine; the global ones
//! (distance to plate boundaries, priority flood, sea level) are exact single-threaded passes in
//! index order. Channels are separate f32 arrays with IEEE basic operations only.

use std::collections::VecDeque;
use std::time::Instant;

use field::{hash32_3, map, Fbm, Field, NoiseBox, Rule, Sphere, Topology, HALF, MAX_NEIGHBOURS, ONE};

use super::{cbrt, Body, Form, Universe};
use crate::gravity::G;

/// The sun turns about this axis on every body (`sky/clock.rs`): `(1, 1, 1)/√3`.
const SUN_AXIS: f64 = 0.577_350_269_189_625_8;
/// Integer lattice units per body radius for noise.
const UNIT: i32 = 1 << 20;
/// Strongest relief any body gets from plates, blocks.
const RELIEF_CAP: f64 = 8_000.0;
/// Passes of moisture advection, thermal erosion and stream power.
const MOIST_PASSES: u32 = 24;
const THERMAL_PASSES: u32 = 8;
const STREAM_PASSES: u32 = 4;
/// Late impacts on a body that never resurfaces.
const LATE_FLUX: f64 = 32.0;
/// Fill kinds.
pub const FROST: u8 = 1;
pub const SALT: u8 = 2;
pub const LAVA: u8 = 3;
/// Phases timed separately.
pub const PHASES: [&str; 8] = ["prior", "plates", "relief", "impacts", "climate", "erosion", "fill+life", "albedo+mip"];

/// One node of a globe: 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Node {
    /// Elevation above the datum, blocks.
    pub elev: i16,
    /// Plate uplift, blocks.
    pub uplift: i16,
    /// Temperature, °C.
    pub temp: i8,
    pub wet: u8,
    /// log₂ of drainage area in nodes.
    pub flow: u8,
    /// The suite layer that outcrops (0 = crust).
    pub rock: u8,
    pub fill: u8,
    pub life: u8,
    pub plate: u8,
    /// Convergence at the nearest plate boundary and distance to it (nodes).
    pub stress: [i8; 2],
    /// rgb565.
    pub albedo: u16,
    /// Surface age: 255 never resurfaced.
    pub age: u8,
}

/// A suite layer as the globe reads it.
#[derive(Clone, Copy, Debug)]
pub struct Layer {
    pub amount: u8,
    pub rgb: [u8; 3],
    pub friction: u8,
    pub cohesion: i16,
}

/// What a globe grows from.
#[derive(Clone, Debug)]
pub struct Input {
    pub g: u32,
    pub seed: u32,
    pub radius: f64,
    pub gravity: f64,
    pub density: f64,
    pub yield_stress: f64,
    pub mass: f64,
    /// 1.0 = the start world.
    pub heat: f64,
    pub air: bool,
    /// Body temperature, K.
    pub temp: f64,
    pub volatile: f64,
    /// Organic share: the suite's carbon-like layers, or the palette's organic ground roles on a
    /// temperate world with air.
    pub carbon: f64,
    pub glow: bool,
    /// Relief knob, 1.0 = designed.
    pub relief: f64,
    /// Recorded impacts: direction and energy.
    pub impacts: Vec<([f64; 3], f64)>,
    /// Suite layers, crust first.
    pub layers: Vec<Layer>,
}

impl Input {
    /// The globe input of body `id` at resolution `g`.
    pub fn of(u: &Universe, id: usize, g: u32, relief: f64) -> Input {
        let b: &Body = &u.bodies[id];
        let s = &u.suites[b.traits.suite as usize];
        let radius = if b.traits.form == Form::Round { b.radius as f64 } else { b.half as f64 };
        Input {
            g,
            seed: b.seed,
            radius,
            gravity: b.traits.gravity as f64,
            density: b.density,
            yield_stress: b.traits.yield_stress,
            mass: b.traits.mass,
            heat: b.traits.heat as f64 / 256.0,
            air: b.traits.air_top.is_some(),
            temp: b.traits.temp as f64,
            volatile: s.classes[3],
            carbon: if b.traits.air_top.is_some() && (super::TEMPERATE.0..=super::TEMPERATE.1).contains(&(b.traits.temp as f64)) {
                s.classes[2].max(super::ORGANICS)
            } else {
                s.classes[2]
            },
            glow: b.traits.glow,
            relief,
            impacts: b.history.impacts.iter().map(|i| (i.dir, i.energy)).collect(),
            layers: s
                .minerals
                .iter()
                .rev()
                .map(|m| Layer { amount: m.amount, rgb: m.rgb, friction: m.friction, cohesion: m.cohesion })
                .collect(),
        }
    }

    /// The resolution the design gives a body of radius `r`.
    pub fn resolution(r: f64) -> u32 {
        if r >= 1.0e7 {
            128
        } else if r >= 1.0e6 {
            64
        } else {
            32
        }
    }
}

/// A grown globe.
pub struct Globe {
    pub g: u32,
    pub nodes: Vec<Node>,
    /// Per face, levels of (min, max) elevation over squares of 2^k nodes, finest first.
    pub mip: Vec<(i16, i16)>,
    /// Milliseconds per phase ([`PHASES`]).
    pub ms: [f64; 8],
    /// Sea level, blocks.
    pub sea: f32,
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn smooth(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A hashed unit vector, rejection sampled.
fn direction(seed: u32, k: i32, salt: u32) -> [f64; 3] {
    for t in 0..64 {
        let h = |a: i32| hash32_3(seed, k, t, a, salt) as f64 / 2_147_483_648.0 - 1.0;
        let v = [h(0), h(1), h(2)];
        let l2 = dot(v, v);
        if l2 > 0.01 && l2 <= 1.0 {
            let l = l2.sqrt();
            return v.map(|x| x / l);
        }
    }
    [0.0, 1.0, 0.0]
}

/// Every node's upwind neighbours (wind blows from cold to warm, and temperature does not change
/// while moisture moves), as compact lists.
struct Upwind {
    start: Vec<u32>,
    list: Vec<u32>,
}

impl Upwind {
    fn new(s: &Sphere, temp: &[f32]) -> Upwind {
        let mut start = Vec::with_capacity(s.len() + 1);
        let mut list = Vec::with_capacity(s.len() * 4);
        let mut nb = [0u32; MAX_NEIGHBOURS];
        for i in 0..s.len() {
            start.push(list.len() as u32);
            if s.owns(i) {
                let n = s.neighbours(i, &mut nb);
                list.extend(nb[..n].iter().filter(|&&j| temp[j as usize] < temp[i]));
            }
        }
        start.push(list.len() as u32);
        Upwind { start, list }
    }

    #[inline]
    fn of(&self, i: usize) -> &[u32] {
        &self.list[self.start[i] as usize..self.start[i + 1] as usize]
    }

    /// The moisture (Q16) each node loses climbing from its upwind neighbours' mean height.
    fn rain(&self, h: &[f32], relief: f32, threads: usize) -> Vec<u16> {
        map(
            h.len(),
            |i| {
                let up = self.of(i);
                if up.is_empty() {
                    return 0;
                }
                let mean = up.iter().map(|&j| h[j as usize]).sum::<f32>() / up.len() as f32;
                ((0.5 * ((h[i] - mean) / relief).max(0.0)).min(1.0) * 65_535.0) as u16
            },
            threads,
        )
    }
}

/// Moisture (Q16) blows from colder neighbours to warmer ones, keeps 92% a step and rains out
/// where it climbs; seas and other sources stay saturated.
struct Moist<'a> {
    up: &'a Upwind,
    rain: &'a [u16],
    src: &'a [u16],
}

/// 0.92 in Q16.
const CARRY: u32 = 60_293;

impl Rule<u16> for Moist<'_> {
    const RADIUS: u8 = 1;
    const NEIGHBOURS: bool = false;
    fn apply(&self, prev: &[u16], i: usize, _nb: &[u32]) -> u16 {
        let up = self.up.of(i);
        let carried = if up.is_empty() {
            (prev[i] as u32 * CARRY) >> 16
        } else {
            let sum: u32 = up.iter().map(|&j| prev[j as usize] as u32).sum();
            (((sum as u64 * CARRY as u64) / (up.len() as u64 * 65_536)) as u32).saturating_sub(self.rain[i] as u32)
        };
        self.src[i].max(carried as u16)
    }
}

/// Matter steeper than the talus slope slides to lower neighbours (symmetric, so mass is kept).
struct Thermal<'a> {
    talus: f32,
    sphere: &'a Sphere,
    radius: f32,
}

impl Thermal<'_> {
    /// Whether any edge of node `i` is steeper than the talus slope (else a pass is the identity).
    fn steep(&self, h: &[f32], i: usize) -> bool {
        let mut nb = [0u32; MAX_NEIGHBOURS];
        let n = self.sphere.neighbours(i, &mut nb);
        let edge = self.sphere.edge(i);
        nb[..n].iter().enumerate().any(|(k, &j)| (h[i] - h[j as usize]).abs() > self.talus * self.sphere.chord[edge + k] * self.radius)
    }
}

impl Rule<f32> for Thermal<'_> {
    const RADIUS: u8 = 1;
    fn apply(&self, prev: &[f32], i: usize, nb: &[u32]) -> f32 {
        let mut dh = 0.0f32;
        let edge = self.sphere.edge(i);
        for (k, &j) in nb.iter().enumerate() {
            let limit = self.talus * self.sphere.chord[edge + k] * self.radius;
            let drop = prev[i] - prev[j as usize];
            if drop > limit {
                dh -= 0.125 * (drop - limit);
            } else if -drop > limit {
                dh += 0.125 * (-drop - limit);
            }
        }
        prev[i] + dh
    }
}

/// Stream power: a node cuts toward its receiver in proportion to √area times the drop.
struct Stream<'a> {
    recv: &'a [u32],
    area: &'a [f32],
    k: f32,
}

impl Rule<f32> for Stream<'_> {
    const RADIUS: u8 = 1;
    const NEIGHBOURS: bool = false;
    fn apply(&self, prev: &[f32], i: usize, _nb: &[u32]) -> f32 {
        let r = self.recv[i] as usize;
        if r == i {
            return prev[i];
        }
        let drop = (prev[i] - prev[r]).max(0.0);
        prev[i] - (self.k * self.area[i].sqrt() * drop).min(drop)
    }
}

/// Wall-clock laps per phase.
struct Laps {
    t: Instant,
    ms: [f64; 8],
}

impl Laps {
    fn lap(&mut self, k: usize) {
        self.ms[k] += self.t.elapsed().as_secs_f64() * 1e3;
        self.t = Instant::now();
    }
}

fn rgb565(c: [f32; 3]) -> u16 {
    let q = |v: f32, bits: u32| ((v.clamp(0.0, 255.0) as u32) >> (8 - bits)) as u16;
    (q(c[0], 5) << 11) | (q(c[1], 6) << 5) | q(c[2], 5)
}

/// Grow the globe of `input` on up to `threads` threads.
pub fn grow(input: &Input, threads: usize) -> Globe {
    let s = Sphere::get(input.g);
    let n = s.len();
    let mut laps = Laps { t: Instant::now(), ms: [0.0; 8] };
    let seed = input.seed;
    let owners: Vec<usize> = (0..n).filter(|&i| s.owns(i)).collect();
    let budget = (input.yield_stress / (input.density * input.gravity.max(1e-6))).min(RELIEF_CAP) * input.relief;
    let relief = budget.max(1.0) as f32;
    let layers = input.layers.len().max(1);

    // Prior: continent noise and a warp for plate boundaries.
    let (lo, hi) = ([-UNIT - 2; 3], [UNIT + 2; 3]);
    let lumps = Fbm::new(seed, UNIT / 2, 4, HALF, 1, lo, hi);
    let warp: [NoiseBox; 3] = std::array::from_fn(|k| NoiseBox::new(seed, UNIT / 3, 10 + k as u32, lo, hi));
    let point = |i: usize| s.dirs[i].map(|v| (v * UNIT as f64).round() as i32);
    let prior: Vec<[f32; 4]> = map(
        n,
        |i| {
            if !s.owns(i) {
                return [0.0; 4];
            }
            let p = point(i);
            let q = |v: i32| (v - HALF) as f32 / ONE as f32;
            [q(lumps.at(p)), q(warp[0].at(p)), q(warp[1].at(p)), q(warp[2].at(p))]
        },
        threads,
    );
    laps.lap(0);

    // Plates: seeds by heat, each node joins the plate its warped direction points to most.
    let plates = (1.0 + 12.0 * input.heat).clamp(1.0, 16.0) as usize;
    let seeds: Vec<[f64; 3]> = (0..plates as i32).map(|k| direction(seed, k, 0x9A7E)).collect();
    let velocity: Vec<[f64; 3]> = (0..plates as i32)
        .map(|k| {
            let v = direction(seed, k, 0x7E10);
            let d = dot(v, seeds[k as usize]);
            let speed = 0.3 + 0.7 * hash32_3(seed, k, 0, 0, 0x5BED) as f64 / 4_294_967_296.0;
            std::array::from_fn(|a| (v[a] - d * seeds[k as usize][a]) * speed)
        })
        .collect();
    // Light crust floats: each plate's crust is one of the top layers.
    let mean_amount = input.layers.iter().map(|l| l.amount as f64).sum::<f64>() / layers as f64;
    let crust: Vec<usize> = (0..plates as i32).map(|k| hash32_3(seed, k, 1, 0, 0xC0A5) as usize % layers.min(3)).collect();
    let buoyancy: Vec<f64> = crust
        .iter()
        .map(|&l| input.layers.get(l).map_or(0.0, |x| (mean_amount - x.amount as f64) / (mean_amount + 1.0)))
        .collect();
    let mean_buoyancy = buoyancy.iter().sum::<f64>() / plates as f64;
    let plate: Vec<u8> = map(
        n,
        |i| {
            let d = s.dirs[i];
            let w = prior[s.owner(i)];
            let p = [d[0] + 0.6 * w[1] as f64, d[1] + 0.6 * w[2] as f64, d[2] + 0.6 * w[3] as f64];
            let mut best = (f64::NEG_INFINITY, 0u8);
            for (k, sd) in seeds.iter().enumerate() {
                let v = dot(p, *sd);
                if v > best.0 {
                    best = (v, k as u8);
                }
            }
            best.1
        },
        threads,
    );
    // Convergence at boundary nodes: the closing speed of the two plates across the edge.
    let conv: Vec<f32> = map(
        n,
        |i| {
            if !s.owns(i) {
                return 0.0;
            }
            let mut nb = [0u32; MAX_NEIGHBOURS];
            let k = s.neighbours(i, &mut nb);
            let (pi, mut best) = (plate[i] as usize, 0.0f32);
            for &j in &nb[..k] {
                let pj = plate[j as usize] as usize;
                if pj == pi {
                    continue;
                }
                let e: [f64; 3] = std::array::from_fn(|a| s.dirs[j as usize][a] - s.dirs[i][a]);
                let l = dot(e, e).sqrt();
                let c = (0..3).map(|a| (velocity[pi][a] - velocity[pj][a]) * e[a]).sum::<f64>() / l;
                if best == 0.0 || c.abs() as f32 > best.abs() {
                    best = c as f32;
                }
            }
            best
        },
        threads,
    );
    laps.lap(1);

    // Relief: distance to the nearest boundary (multi-source BFS in index order), then uplift at
    // convergent boundaries, trenches on the denser side, rifts where plates part, hotspot domes.
    let mut hops = vec![u16::MAX; n];
    let mut source = vec![0u32; n];
    let mut queue = VecDeque::new();
    for &i in &owners {
        if plates > 1 && conv[i] != 0.0 {
            hops[i] = 0;
            source[i] = i as u32;
            queue.push_back(i);
        }
    }
    let mut nb = [0u32; MAX_NEIGHBOURS];
    while let Some(i) = queue.pop_front() {
        let k = s.neighbours(i, &mut nb);
        for &j in &nb[..k] {
            let j = j as usize;
            if hops[j] == u16::MAX {
                hops[j] = hops[i] + 1;
                source[j] = source[i];
                queue.push_back(j);
            }
        }
    }
    let width = (input.g as f64 / 16.0).max(2.0);
    let hotspots: Vec<[f64; 3]> = (0..(input.heat * 4.0).round() as i32).map(|k| direction(seed, k, 0x4075)).collect();
    let h = relief as f64;
    let elev: Vec<f32> = map(
        n,
        |i| {
            if !s.owns(i) {
                return 0.0;
            }
            let mut e = 0.5 * h * buoyancy[plate[i] as usize] + 0.25 * h * prior[i][0] as f64 * 2.0;
            if hops[i] != u16::MAX {
                let d = hops[i] as f64;
                let src = source[i] as usize;
                let c = conv[src] as f64;
                if c > 0.0 {
                    e += h * c * smooth(1.0 - d / width);
                    if buoyancy[plate[i] as usize] < mean_buoyancy {
                        e -= 0.6 * h * c * smooth(1.0 - 3.0 * d / width);
                    }
                } else {
                    e += 0.3 * h * c * smooth(1.0 - 2.0 * d / width);
                }
            }
            for spot in &hotspots {
                let ang = (2.0 - 2.0 * dot(s.dirs[i], *spot)).max(0.0).sqrt();
                e += 0.4 * h * smooth(1.0 - ang / 0.15);
            }
            e as f32
        },
        threads,
    );
    laps.lap(2);

    // Impacts: every recorded collision and a late flux that only old surfaces keep.
    let age = 1.0 / (1.0 + input.heat + if input.air { 1.0 } else { 0.0 });
    let binding = G * input.mass * input.mass / input.radius;
    let mut craters: Vec<([f64; 3], f64)> = input
        .impacts
        .iter()
        .map(|&(d, e)| (d, (0.3 * cbrt(e / binding)).clamp(0.01, 0.8)))
        .collect();
    for k in 0..(LATE_FLUX * age).round() as i32 {
        let u = hash32_3(seed, k, 0, 0, 0x1A7E) as f64 / 4_294_967_296.0;
        craters.push((direction(seed, k, 0xC4A7), 0.02 + 0.1 * u * u * u));
    }
    let radius = input.radius;
    let cratered: Vec<f32> = map(
        n,
        |i| {
            if !s.owns(i) {
                return 0.0;
            }
            let mut e = elev[i] as f64;
            for &(at, theta) in &craters {
                let c = dot(s.dirs[i], at);
                if c < 1.0 - 1.28 * theta * theta {
                    continue;
                }
                let t = (2.0 - 2.0 * c).max(0.0).sqrt() / theta;
                let depth = (0.2 * theta * radius).min(2.0 * h);
                let (dep, rim) = (depth, 0.3 * depth);
                let t2 = t * t;
                e += if t < 1.0 {
                    -dep * (1.0 - t2) + rim * t2 * t2 * t2
                } else {
                    let f = 1.0 - (t - 1.0) / 0.6;
                    rim * f.max(0.0) * f.max(0.0)
                };
            }
            e as f32
        },
        threads,
    );
    laps.lap(3);

    // Sea level: the elevation under which the volatile budget fills the surface, by solid angle
    // (a weighted histogram, so no sort).
    let weight = |i: usize| {
        let (_, a, b) = s.locate(s.dirs[i]);
        let half = input.g as f64 * 0.5;
        let (x, y) = ((a - half) / half, (b - half) / half);
        let r2 = 1.0 + x * x + y * y;
        1.0 / (r2 * r2.sqrt())
    };
    let weights: Vec<f64> = map(n, |i| if s.owns(i) { weight(i) } else { 0.0 }, threads);
    let total: f64 = owners.iter().map(|&i| weights[i]).sum();
    // Hot glowing worlds flood their basins with melt instead.
    let hot = input.glow && input.heat > 1.5;
    let budget_fill = if hot { (0.1 * (input.heat - 1.5)).min(0.3) } else if input.air { input.volatile } else { 0.0 };
    let sea = sea_level(&cratered, &weights, &owners, budget_fill * total);

    // Climate: daily mean light from the sun's path, lapse with height, greenhouse in the body
    // temperature already. Channels are separate arrays, so a pass moves only what it writes.
    let lapse = 0.0065 * input.gravity / 24.0;
    let temp: Vec<f32> = map(
        n,
        |i| {
            let d = s.dirs[i];
            let along = (d[0] + d[1] + d[2]) * SUN_AXIS;
            let light = (1.0 - along * along).max(0.0).sqrt();
            let hgt = (cratered[i] as f64).max(sea as f64);
            (input.temp + 40.0 * (light.sqrt().sqrt() - 0.85) - lapse * hgt) as f32
        },
        threads,
    );
    let seas = |h: &[f32]| -> Vec<u16> { h.iter().map(|&v| if input.air && !hot && v < sea { u16::MAX } else { 0 }).collect() };
    let src = seas(&cratered);
    let mut wet = Field::new(src.clone());
    let up = if input.air { Upwind::new(s, &temp) } else { Upwind { start: vec![0; n + 1], list: Vec::new() } };
    if input.air {
        let rain = up.rain(&cratered, relief, threads);
        wet.run(s, &Moist { up: &up, rain: &rain, src: &src }, MOIST_PASSES, threads);
    }
    laps.lap(4);

    // Erosion: thermal slides (skipped exactly when no edge is steeper than the talus slope, which
    // at globe scale is the rule), then on worlds with air a priority flood to the sea, drainage
    // area down the flood's tree and stream-power cutting.
    let talus = input.layers.first().map_or(0.6, |x| 0.3 + 0.7 * x.friction as f32 / 255.0);
    let thermal = Thermal { talus, sphere: s, radius: radius as f32 };
    let mut h = Field::new(cratered.clone());
    if map(n, |i| s.owns(i) && thermal.steep(&cratered, i), threads).contains(&true) {
        h.run(s, &thermal, THERMAL_PASSES, threads);
    }
    let mut flow = vec![0u8; n];
    if input.air {
        let mut recv: Vec<u32> = (0..n as u32).collect();
        let order = flood(s, h.cells_mut(), &mut recv, &owners, sea);
        let mut area: Vec<f32> = weights.iter().map(|&w| w as f32).collect();
        for &i in order.iter().rev() {
            let r = recv[i] as usize;
            if r != i {
                area[r] += area[i];
            }
        }
        let unit_area = (total / owners.len() as f64) as f32;
        for &i in &owners {
            flow[i] = log2_floor((area[i] / unit_area).max(1.0));
        }
        h.run(s, &Stream { recv: &recv, area: &area, k: 0.02 / unit_area.sqrt() }, STREAM_PASSES, threads);
    }
    let h = h.into_cells();
    laps.lap(5);

    // Fill and life: seas of the climate's kind, one more advection pass from them, then life.
    let src = seas(&h);
    if input.air {
        let rain = up.rain(&h, relief, threads);
        wet.run(s, &Moist { up: &up, rain: &rain, src: &src }, 1, threads);
    }
    let wet = wet.into_cells();
    laps.lap(6);

    let fill_rgb = |k: u8| match k {
        FROST => [204.0, 226.0, 246.0],
        SALT => [240.0, 238.0, 230.0],
        _ => [255.0, 112.0, 36.0],
    };
    let mut nodes: Vec<Node> = map(
        n,
        |i| {
            let o = s.owner(i);
            let (height, t, moist) = (h[o], temp[o], wet[o] as f32 / 65_535.0);
            let depth = (cratered[o] - height).max(0.0) + (-height).max(0.0);
            let rock = ((depth / (relief / layers as f32)) as usize).min(layers - 1);
            let fill = if height >= sea {
                0
            } else if hot {
                LAVA
            } else if t > 320.0 {
                SALT
            } else {
                FROST
            };
            let comfort = (1.0 - ((t - 288.0) / 40.0) * ((t - 288.0) / 40.0)).max(0.0);
            let life = if input.air && fill == 0 { comfort * moist * input.carbon as f32 } else { 0.0 };
            let base = if fill != 0 {
                fill_rgb(fill)
            } else {
                input.layers.get(rock).map_or([128.0; 3], |l| l.rgb.map(|v| v as f32))
            };
            let green = [56.0, 122.0, 44.0];
            let t = (life * 4.0).min(0.8);
            let rgb: [f32; 3] = std::array::from_fn(|k| base[k] + (green[k] - base[k]) * t);
            Node {
                elev: height.clamp(-32_768.0, 32_767.0) as i16,
                uplift: elev[o].clamp(-32_768.0, 32_767.0) as i16,
                temp: (t - 273.0).clamp(-128.0, 127.0) as i8,
                wet: (moist * 255.0) as u8,
                flow: flow[o],
                rock: rock as u8,
                fill,
                life: (life * 255.0).min(255.0) as u8,
                plate: plate[o],
                stress: [(conv[source[o] as usize] * 127.0).clamp(-127.0, 127.0) as i8, hops[o].min(127) as i8],
                albedo: rgb565(rgb),
                age: (age * 255.0) as u8,
            }
        },
        threads,
    );
    s.glue_cells(&mut nodes);
    let mip = mip(s, &nodes);
    laps.lap(7);
    Globe { g: input.g, nodes, mip, ms: laps.ms, sea }
}

/// ⌊log₂⌋ of a number ≥ 1 from its exponent bits.
fn log2_floor(x: f32) -> u8 {
    (((x.to_bits() >> 23) & 0xFF) as i32 - 127).clamp(0, 255) as u8
}

/// The elevation under which the owners' weights sum to `want`.
fn sea_level(h: &[f32], w: &[f64], owners: &[usize], want: f64) -> f32 {
    if want <= 0.0 {
        return f32::NEG_INFINITY;
    }
    let mut bins = vec![0.0f64; 1 << 16];
    for &i in owners {
        bins[(h[i].clamp(-32_768.0, 32_767.0) as i32 + 32_768) as usize] += w[i];
    }
    let mut sum = 0.0;
    for (k, &b) in bins.iter().enumerate() {
        sum += b;
        if sum >= want {
            return (k as i32 - 32_768) as f32 + 1.0;
        }
    }
    f32::INFINITY
}

/// Priority flood from the sea (or the lowest node): every node drains to the neighbour it was
/// reached from, so drainage is a tree with no pits (pits are filled in `h`). Keys only grow
/// during a flood, so the queue is monotone buckets of whole blocks, each a stack threaded through
/// one array (last in, first out). Returns the order nodes were reached.
fn flood(s: &Sphere, h: &mut [f32], recv: &mut [u32], owners: &[usize], sea: f32) -> Vec<usize> {
    const NONE: u32 = u32::MAX;
    const BUCKETS: usize = 1 << 16;
    let bucket = |v: f32| (v.floor().clamp(-32_768.0, 32_767.0) as i32 + 32_768) as usize;
    let mut head = vec![NONE; BUCKETS];
    let mut next = vec![NONE; h.len()];
    let mut seen = vec![false; h.len()];
    let push = |head: &mut [u32], next: &mut [u32], j: usize, b: usize| {
        next[j] = head[b];
        head[b] = j as u32;
    };
    let mut at = BUCKETS;
    for &i in owners {
        if h[i] < sea {
            push(&mut head, &mut next, i, bucket(h[i]));
            seen[i] = true;
            at = at.min(bucket(h[i]));
        }
    }
    if at == BUCKETS {
        let low = owners.iter().copied().min_by(|&a, &b| h[a].total_cmp(&h[b]).then(a.cmp(&b))).unwrap_or(0);
        push(&mut head, &mut next, low, bucket(h[low]));
        seen[low] = true;
        at = bucket(h[low]);
    }
    let mut order = Vec::with_capacity(owners.len());
    let mut nb = [0u32; MAX_NEIGHBOURS];
    while at < BUCKETS {
        if head[at] == NONE {
            at += 1;
            continue;
        }
        let i = head[at] as usize;
        head[at] = next[i];
        order.push(i);
        let k = s.neighbours(i, &mut nb);
        for &j in &nb[..k] {
            let j = j as usize;
            if !seen[j] {
                seen[j] = true;
                recv[j] = i as u32;
                h[j] = h[j].max(h[i]);
                push(&mut head, &mut next, j, bucket(h[j]));
            }
        }
    }
    order
}

/// Per face, min/max elevation over 2×2 node squares, then halving to one square.
fn mip(s: &Sphere, nodes: &[Node]) -> Vec<(i16, i16)> {
    let g = s.g;
    let mut out = Vec::new();
    for face in 0..6 {
        let mut level: Vec<(i16, i16)> = (0..g * g)
            .map(|k| {
                let (i, j) = (k % g, k / g);
                let e = [(0, 0), (1, 0), (0, 1), (1, 1)].map(|(a, b)| nodes[s.node(face, i + a, j + b)].elev);
                (*e.iter().min().unwrap_or(&0), *e.iter().max().unwrap_or(&0))
            })
            .collect();
        let mut side = g;
        out.extend_from_slice(&level);
        while side > 1 {
            let half = side / 2;
            level = (0..half * half)
                .map(|k| {
                    let (i, j) = (k % half * 2, k / half * 2);
                    let q = [(0, 0), (1, 0), (0, 1), (1, 1)].map(|(a, b)| level[((j + b) * side + i + a) as usize]);
                    (q.iter().map(|v| v.0).min().unwrap_or(0), q.iter().map(|v| v.1).max().unwrap_or(0))
                })
                .collect();
            side = half;
            out.extend_from_slice(&level);
        }
    }
    out
}
