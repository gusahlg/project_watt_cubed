//! Stage 8 (lab prototype): a body's history on a cube-sphere of its surface. Plates from the
//! body's heat, relief from plate motion inside its strength budget, basins from its recorded
//! impacts and a late flux, climate from the real sun path, erosion, seas and life, then a colour
//! per node and a min/max mip. Local phases are Jacobi rules on the field engine; the global ones
//! (distance to plate boundaries, priority flood, sea level) are exact single-threaded passes in
//! index order. Channels are f32 with IEEE basic operations only.

use std::collections::{BinaryHeap, VecDeque};
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
            carbon: s.classes[2],
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

/// The working channels of a node.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Cell {
    h: f32,
    temp: f32,
    wet: f32,
    src: f32,
    /// Drainage receiver and area (solid-angle units), from the priority flood.
    recv: u32,
    area: f32,
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

/// Moisture blows from colder neighbours to warmer ones and rains out where it climbs.
struct Moist {
    relief: f32,
}

impl Rule<Cell> for Moist {
    const RADIUS: u8 = 1;
    fn apply(&self, prev: &[Cell], i: usize, nb: &[u32]) -> Cell {
        let c = prev[i];
        let (mut sum, mut h_up, mut n) = (0.0f32, 0.0f32, 0.0f32);
        for &j in nb {
            let o = &prev[j as usize];
            if o.temp < c.temp {
                sum += o.wet;
                h_up += o.h;
                n += 1.0;
            }
        }
        let carried = if n > 0.0 {
            let rain = 0.5 * ((c.h - h_up / n) / self.relief).max(0.0);
            (0.92 * sum / n - rain).max(0.0)
        } else {
            0.92 * c.wet
        };
        Cell { wet: c.src.max(carried).min(1.0), ..c }
    }
}

/// Matter steeper than the talus slope slides to lower neighbours (symmetric, so mass is kept).
struct Thermal<'a> {
    talus: f32,
    sphere: &'a Sphere,
    radius: f32,
}

impl Rule<Cell> for Thermal<'_> {
    const RADIUS: u8 = 1;
    fn apply(&self, prev: &[Cell], i: usize, nb: &[u32]) -> Cell {
        let c = prev[i];
        let mut dh = 0.0f32;
        let edge = self.sphere.edge(i);
        for (k, &j) in nb.iter().enumerate() {
            let limit = self.talus * self.sphere.chord[edge + k] * self.radius;
            let drop = c.h - prev[j as usize].h;
            if drop > limit {
                dh -= 0.125 * (drop - limit);
            } else if -drop > limit {
                dh += 0.125 * (-drop - limit);
            }
        }
        Cell { h: c.h + dh, ..c }
    }
}

/// Stream power: a node cuts toward its receiver in proportion to √area times the drop.
struct Stream {
    k: f32,
}

impl Rule<Cell> for Stream {
    const RADIUS: u8 = 1;
    fn apply(&self, prev: &[Cell], i: usize, _nb: &[u32]) -> Cell {
        let c = prev[i];
        let r = c.recv as usize;
        if r == i {
            return c;
        }
        let floor = prev[r].h;
        let drop = (c.h - floor).max(0.0);
        Cell { h: c.h - (self.k * c.area.sqrt() * drop).min(drop), ..c }
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
    // temperature already.
    let lapse = 0.0065 * input.gravity / 24.0;
    let mut f = Field::new(map(
        n,
        |i| {
            let d = s.dirs[i];
            let along = (d[0] + d[1] + d[2]) * SUN_AXIS;
            let light = (1.0 - along * along).max(0.0).sqrt();
            let hgt = (cratered[i] as f64).max(sea as f64);
            let temp = input.temp + 40.0 * (light.sqrt().sqrt() - 0.85) - lapse * hgt;
            let src = if input.air && !hot && (cratered[i] as f64) < sea as f64 { 1.0 } else { 0.0 };
            Cell { h: cratered[i], temp: temp as f32, wet: src, src, recv: i as u32, area: weights[i] as f32 }
        },
        threads,
    ));
    if input.air {
        f.run(s, &Moist { relief }, MOIST_PASSES, threads);
    }
    laps.lap(4);

    // Erosion: thermal slides, then on worlds with air a priority flood to the sea, drainage area
    // down the flood's tree and stream-power cutting.
    let talus = input.layers.first().map_or(0.6, |x| 0.3 + 0.7 * x.friction as f32 / 255.0);
    f.run(s, &Thermal { talus, sphere: s, radius: radius as f32 }, THERMAL_PASSES, threads);
    let mut flow = vec![0u8; n];
    if input.air {
        let cells = f.cells_mut();
        let order = flood(s, cells, &owners, sea);
        for &i in order.iter().rev() {
            let r = cells[i].recv as usize;
            if r != i {
                cells[r].area += cells[i].area;
            }
        }
        let unit_area = (total / owners.len() as f64) as f32;
        for &i in &owners {
            flow[i] = log2_floor((cells[i].area / unit_area).max(1.0));
        }
        f.run(s, &Stream { k: 0.02 / unit_area.sqrt() }, STREAM_PASSES, threads);
    }
    laps.lap(5);

    // Fill and life: seas of the climate's kind, one more advection pass from them, then life.
    {
        let cells = f.cells_mut();
        for c in cells.iter_mut() {
            c.src = if input.air && !hot && c.h < sea { 1.0 } else { 0.0 };
        }
    }
    if input.air {
        f.run(s, &Moist { relief }, 1, threads);
    }
    let cells = f.into_cells();
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
            let c = cells[o];
            let depth = (cratered[o] - c.h).max(0.0) + (-c.h).max(0.0);
            let rock = ((depth / (relief / layers as f32)) as usize).min(layers - 1);
            let fill = if c.h >= sea {
                0
            } else if hot {
                LAVA
            } else if c.temp > 320.0 {
                SALT
            } else {
                FROST
            };
            let comfort = (1.0 - ((c.temp - 288.0) / 40.0) * ((c.temp - 288.0) / 40.0)).max(0.0);
            let life = if input.air && fill == 0 { comfort * c.wet * input.carbon as f32 } else { 0.0 };
            let base = if fill != 0 {
                fill_rgb(fill)
            } else {
                input.layers.get(rock).map_or([128.0; 3], |l| l.rgb.map(|v| v as f32))
            };
            let green = [56.0, 122.0, 44.0];
            let t = (life * 4.0).min(0.8);
            let rgb: [f32; 3] = std::array::from_fn(|k| base[k] + (green[k] - base[k]) * t);
            Node {
                elev: c.h.clamp(-32_768.0, 32_767.0) as i16,
                uplift: elev[o].clamp(-32_768.0, 32_767.0) as i16,
                temp: (c.temp - 273.0).clamp(-128.0, 127.0) as i8,
                wet: (c.wet * 255.0) as u8,
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

/// An integer that orders like the float (no NaNs).
fn ordered(h: f32) -> i32 {
    let b = h.to_bits() as i32;
    b ^ (((b >> 31) as u32) >> 1) as i32
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
/// reached from, so drainage is a tree with no pits. Returns the order nodes were reached.
fn flood(s: &Sphere, cells: &mut [Cell], owners: &[usize], sea: f32) -> Vec<usize> {
    let key = |h: f32, i: usize| std::cmp::Reverse((ordered(h), i));
    let mut heap = BinaryHeap::new();
    let mut seen = vec![false; cells.len()];
    for &i in owners {
        if cells[i].h < sea {
            heap.push(key(cells[i].h, i));
            seen[i] = true;
        }
    }
    if heap.is_empty() {
        let low = owners.iter().copied().min_by(|&a, &b| cells[a].h.total_cmp(&cells[b].h).then(a.cmp(&b))).unwrap_or(0);
        heap.push(key(cells[low].h, low));
        seen[low] = true;
    }
    let mut order = Vec::with_capacity(owners.len());
    let mut nb = [0u32; MAX_NEIGHBOURS];
    while let Some(std::cmp::Reverse((_, i))) = heap.pop() {
        order.push(i);
        let k = s.neighbours(i, &mut nb);
        for &j in &nb[..k] {
            let j = j as usize;
            if !seen[j] {
                seen[j] = true;
                cells[j].recv = i as u32;
                cells[j].h = cells[j].h.max(cells[i].h);
                heap.push(key(cells[j].h, j));
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
