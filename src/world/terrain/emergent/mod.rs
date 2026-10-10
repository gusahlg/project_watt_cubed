//! The emergent cosmos (EMERGENT-WORLDGEN-DESIGN v5, stages 1 to 8): worlds grown from the law
//! instead of a written catalog. A seed's nebula collapses into systems, each system's matter
//! accretes into bodies, each body's minerals come from the law's contact process, and their
//! density and strength decide its shape through the genesis table; traits, the start world and
//! storage follow. This is the P1 lab: the game's generator reads none of it yet.
//!
//! Creation arithmetic is integer or IEEE-basic f64 (`+ − × ÷`, `sqrt`, `floor`, `round`) in a
//! fixed order: no libm. Cube roots are Newton steps from a bit-hack seed; `log₂` is the exponent
//! bits plus an atanh series.

pub mod accrete;
pub mod globe;
pub mod minerals;
pub mod nebula;

use std::sync::OnceLock;
use std::time::Instant;

use field::hash32_3;
use material::{Element, Law};

use self::accrete::{fuse, half_of, Impact, Proto};
use self::minerals::{Found, Ground, Suite};
use self::nebula::Nebula;
use super::cosmos::{HOME_CUBE_HALF, HOME_RADIUS, RELIEF};
use super::TerrainCfg;
use crate::gravity::{window, G, R_G};
use crate::mechanics::genesis::{self, MAX_TILT_DEG};
use crate::space::atlas::{Atlas, STORAGE_X0};
use crate::space::warp::STORAGE_MOVE;

/// Today's start world's matter: amount 5 in the old 5e7-block cube.
pub const M_HOME: f64 = 5.0 * 5.0e7 * 5.0e7 * 5.0e7;
/// The spawn contract: density times datum radius of the start world (spawn pull is ∝ ρR).
pub const RHO_R: f64 = 5.0 * HOME_RADIUS as f64;
/// Cube half-size over datum radius of the start world (25e6 over 31,017,520).
const HALF_PER_RADIUS: f64 = HOME_CUBE_HALF as f64 / HOME_RADIUS as f64;
/// Face-centre pull of a uniform cube over `G·ρ·half`.
pub const KAPPA_FACE: f64 = 5.193_793_156_516_389;
const SQRT3: f64 = 1.732_050_807_568_877_2;
/// Whole-nebula re-salts allowed by the interest filter, and the bodies it wants besides the start.
const INTEREST_TRIES: u32 = 8;
const OTHERS_MIN: usize = 3;
/// Re-salts of the start world's composition before the palette fallback.
const START_TRIES: u32 = 16;
/// Rounds of calibration and hierarchy (a satellite falling on the start world moves the scale).
const CALIBRATIONS: u32 = 4;
/// Satellites smaller than this half-size take their parent's suite.
const OWN_SUITE_HALF: f64 = 1.0e6;
/// Bodies beyond these coordinates after placement become debris.
const BOUND: f64 = 9.2e8;
const BOUND_SATELLITE: f64 = 9.5e8;
/// Temperature model (a modelling choice): base, per unit heat, per unit irradiance, greenhouse.
const T_BASE: f64 = 100.0;
const T_HEAT: f64 = 173.0;
const T_IRR: f64 = 2_000.0;
const GREENHOUSE: f64 = 15.0;
/// The start world's temperature band, and the width of comfort for life.
const TEMPERATE: (f64, f64) = (250.0, 330.0);
const COMFORT: f64 = 40.0;
const T_HOME: f64 = 288.0;
/// `air_top = AIR_SCALE · T / g`: the start world keeps today's 20 km.
const AIR_SCALE: f64 = 20_000.0 * 24.0 / T_HOME;
const AIR_MAX: f64 = 40_000.0;
/// Radiogenic heat per unit mass of a fully emissive suite (about the start world's G·M/R).
const E_RAD: f64 = 8.8e8;
/// Share of the surface the palette's organic ground roles cover on a temperate world with air.
const ORGANICS: f64 = 0.5;

macro_rules! params {
    ($($(#[$doc:meta])* $name:ident: $ty:ty = $default:expr,)*) => {
        /// The process constants (design §4.18 and the mineral process), tuned in the lab.
        #[derive(Clone, Debug, PartialEq)]
        pub struct Params {
            $($(#[$doc])* pub $name: $ty,)*
        }

        impl Default for Params {
            fn default() -> Self {
                Self { $($name: $default,)* }
            }
        }

        impl Params {
            /// Set one constant by name (lab command lines).
            pub fn set(&mut self, name: &str, value: &str) -> Result<(), String> {
                match name {
                    $(stringify!($name) => {
                        self.$name = value.parse().map_err(|_| format!("bad value {value} for {name}"))?
                    })*
                    _ => return Err(format!("no parameter {name}")),
                }
                Ok(())
            }

            /// Every constant as `name=value`.
            pub fn describe(&self) -> String {
                [$(format!("{}={:?}", stringify!($name), self.$name)),*].join(" ")
            }
        }
    };
}

params! {
    /// Nebula prior amplitude over the unit floor (scaled by the variety knob), and its octaves
    /// (from 16 cells down).
    prior_amp: i32 = 24,
    octaves: u32 = 3,
    /// Origin well mass at its centre cell (units of the floor), and its radius in cells.
    well: i32 = 64,
    well_r: i32 = 3,
    /// Collapse phases, the share a cell sends per phase (`m >> send`), and the smoothing sweeps
    /// the watershed reads.
    collapse: u32 = 8,
    send: u32 = 2,
    blur: u32 = 2,
    /// System separation, and the box systems must lie in.
    d_sep: f64 = 4.0e8,
    system_bound: f64 = 7.7e8,
    /// Least mass of a system of its own, as a share of the start system's (lighter basins join a
    /// near system or become debris).
    system_min: f64 = 0.25,
    /// The start system's mass over today's home mass (calibrates the nebula before accretion).
    origin_share: f64 = 1.5,
    /// Parcels per system and the ratio of the largest parcel mass to the smallest.
    parcels: u32 = 64,
    parcel_range: f64 = 1000.0,
    /// When above 0, parcels start around the basin sinks with this scatter (cells); at 0 on cells
    /// picked by cell mass.
    sink_spread: f64 = 0.6,
    /// A merged basin lighter than this share of its system seeds no cloud of its own: its mass
    /// joins the system's first sink.
    sink_min: f64 = 1.0,
    /// Merge strength A = R_IN³ / M_home.
    merge_a: f64 = 1.6e8 * 1.6e8 * 1.6e8 / M_HOME,
    /// Contact binary: least mass ratio, and strength (impact energy per mass under K_BIN·Y/ρ).
    q_bin: f64 = 0.4,
    k_bin: f64 = 30.0,
    /// Least mass of a binary's lighter part, as a share of today's home mass.
    bin_min: f64 = 1.0e-4,
    /// A satellite's own surface pull over its parent's pull there.
    k_dom: f64 = 4.0,
    /// Debris mass that rings a body.
    ring_min: f64 = 1.0e17,
    /// Reservoirs in a body's column, occurrences per reservoir, and their ring distance from the
    /// body's composition.
    reservoirs: usize = 12,
    occ_min: u32 = 2,
    occ_max: u32 = 6,
    spread: u32 = 96,
    /// Keep every drawn occurrence the palette's gap from every palette element and the others.
    gap: bool = true,
    /// Before the rest check a restless mineral may leach (1: drop the occurrence the contact moves)
    /// or weather (2: trade occurrences with an endless bath of the role) until it rests; 0: neither.
    leach: u8 = 2,
    /// Check suites against what touches the bulk they become in P2 (true), or against the
    /// painter's whole ground vocabulary (false; slower, about 4 points more suites of their own).
    bulk: bool = true,
    /// Differentiation sweeps at most.
    sweeps: u32 = 48,
    /// Least colour spread of a suite before it falls back to the palette.
    colour_min: u16 = 60,
    /// Heat at which a glowing suite lights its body.
    h_glow: f64 = 1.5,
    /// Heat at which a crust melts into the palette's emissive magma.
    h_melt: f64 = 2.0,
    /// Air retention: air stays when g·R ≥ β·T·(1 + loss).
    beta: f64 = 2.6e4,
    /// Most bodies a universe keeps.
    bodies_max: usize = 64,
}

/// Cube root of a non-negative number: six Newton steps from a bit-hack seed.
pub fn cbrt(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut y = f64::from_bits(x.to_bits() / 3 + 0x2A9F_7893_782D_A1CE);
    for _ in 0..6 {
        y = (2.0 * y + x / (y * y)) / 3.0;
    }
    y
}

/// `log₂` of a positive normal number: exponent bits plus an atanh series of the mantissa, taken
/// in `[√½, √2)` so the series converges fast.
pub fn log2(x: f64) -> f64 {
    let bits = x.to_bits();
    let mut e = ((bits >> 52) & 0x7FF) as i64 - 1023;
    let mut m = f64::from_bits((bits & ((1 << 52) - 1)) | (1023 << 52));
    if m > std::f64::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let mut series = 1.0 / 17.0;
    for k in [15.0, 13.0, 11.0, 9.0, 7.0, 5.0, 3.0, 1.0] {
        series = 1.0 / k + s2 * series;
    }
    e as f64 + 2.0 * s * series * std::f64::consts::LOG2_E
}

/// What a body's shape asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Form {
    /// Keeps its cube grid, sagging at most half a block.
    Cube,
    /// Keeps its cube grid but sags: a warp and a storage box.
    Warped,
    /// Charted on a cube-sphere.
    Round,
}

/// A body's layout from the genesis table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Layout {
    pub form: Form,
    /// Datum radius of a round body (0 for cubes).
    pub radius: f64,
    pub tilt: f64,
    /// Π fell outside the table and was clamped.
    pub clamped: bool,
}

/// One genesis table entry, reduced to what a layout needs.
struct Entry {
    log2_pi: f64,
    tilt: f64,
    radius: f64,
    /// Largest displacement of a matter node, in units of the half-size.
    sag: f64,
}

/// The genesis table, parsed once.
fn table() -> &'static [Entry] {
    static TABLE: OnceLock<Vec<Entry>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let t = genesis::read_table(include_bytes!("../../../mechanics/genesis_table.bin")).expect("the genesis table");
        let m = (t.elements + 2) / 2;
        let nodes = genesis::canonical_nodes(m);
        t.entries
            .iter()
            .map(|e| {
                let sag = nodes
                    .iter()
                    .zip(&e.nodes)
                    .filter(|(c, _)| c[2] <= t.elements / 2)
                    .map(|(c, p)| {
                        let d = [0, 1, 2].map(|a| p[a] - 2.0 * c[a] as f64 / t.elements as f64);
                        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
                    })
                    .fold(0.0, f64::max);
                Entry { log2_pi: e.log2_pi, tilt: e.max_tilt, radius: e.radius, sag }
            })
            .collect()
    })
}

/// The layout of a cube of matter of half-size `half` at `Π_g = pi`: the table's tilt rule
/// (`genesis::tabulated_layout`) without its `log2`, plus the sag that decides a warp.
pub fn layout(pi: f64, half: f64) -> Layout {
    let t = table();
    let (first, last) = (t[0].log2_pi, t[t.len() - 1].log2_pi);
    let raw = log2(pi.max(f64::MIN_POSITIVE));
    let x = raw.clamp(first, last);
    let hi = t.iter().position(|e| e.log2_pi >= x).unwrap_or(t.len() - 1).clamp(1, t.len() - 1);
    let (a, b) = (&t[hi - 1], &t[hi]);
    let w = ((x - a.log2_pi) / (b.log2_pi - a.log2_pi)).clamp(0.0, 1.0);
    let lerp = |p: f64, q: f64| p * (1.0 - w) + q * w;
    let tilt = lerp(a.tilt, b.tilt);
    let clamped = raw < first || raw > last;
    if tilt > MAX_TILT_DEG || a.radius <= 0.0 || b.radius <= 0.0 {
        let form = if lerp(a.sag, b.sag) * half > STORAGE_MOVE { Form::Warped } else { Form::Cube };
        return Layout { form, radius: 0.0, tilt, clamped };
    }
    Layout { form: Form::Round, radius: lerp(a.radius, b.radius) * half, tilt, clamped }
}

/// `Π_g = G ρ² L² / Y` of a suite's matter at half-size `half`.
fn pi_g(s: &Suite, half: f64) -> f64 {
    G * s.density * s.density * half * half / s.yield_stress
}

/// Where a body sits in its system.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rank {
    Start,
    Primary,
    Partner,
    Satellite,
}

/// What a body is, derived after the fact (design §3.4).
#[derive(Clone, Debug, PartialEq)]
pub struct Traits {
    pub rank: Rank,
    pub form: Form,
    pub parent: Option<u16>,
    pub partner: Option<u16>,
    pub mass: f64,
    pub yield_stress: f64,
    pub pi_g: f64,
    pub pi_clamped: bool,
    /// Surface pull, m/s².
    pub gravity: f32,
    /// Q8.8, 1.0 = the start world.
    pub heat: u16,
    pub glow: bool,
    pub air_top: Option<i32>,
    pub temp: i16,
    pub wet: u8,
    pub life: u8,
    pub suite: u16,
    pub name: [u8; 12],
}

/// A body's past: what hit it and how hot that left it.
#[derive(Clone, Debug, PartialEq)]
pub struct History {
    pub impacts: Vec<Impact>,
    /// Collision heat per unit mass.
    pub heat_in: f64,
}

/// One body of the universe.
#[derive(Clone, Debug, PartialEq)]
pub struct Body {
    pub id: u16,
    pub system: u16,
    /// Snapped to 16.
    pub centre: [i64; 3],
    /// Half-size of its cube of matter.
    pub half: i64,
    /// Datum radius when round, 0 otherwise.
    pub radius: i64,
    pub density: f64,
    pub seed: u32,
    pub comp: [i8; 4],
    /// The painter ground its suite was checked against.
    pub ground: Ground,
    pub traits: Traits,
    pub history: History,
}

impl Body {
    /// Bounding radius including relief.
    pub fn reach(&self) -> f64 {
        match self.traits.form {
            Form::Round => (self.radius + RELIEF) as f64,
            _ => (self.half + RELIEF) as f64 * SQRT3,
        }
    }

    /// The name as text.
    pub fn name(&self) -> &str {
        let end = self.traits.name.iter().position(|&b| b == 0).unwrap_or(12);
        std::str::from_utf8(&self.traits.name[..end]).unwrap_or("?")
    }
}

/// How the bodies' charts and boxes fit the storage region.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Storage {
    /// Shelf rows of round atlases and warped boxes (rows stack in z).
    pub rows: u32,
    /// Share of row 0's x budget used.
    pub row0: f64,
    /// Share of the z budget the rows use.
    pub z_used: f64,
    /// Share of the y budget the warped boxes use.
    pub cube_y: f64,
    /// Bodies that did not fit (demoted to debris).
    pub demoted: u32,
}

/// Creation time by stage, milliseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Timings {
    pub nebula: f64,
    pub accrete: f64,
    pub suites: f64,
    pub settle: f64,
    pub storage: f64,
    pub total: f64,
}

/// A grown universe.
pub struct Universe {
    pub seed: u64,
    pub base: Element,
    /// Nebula re-salts the interest filter needed.
    pub resalts: u32,
    /// Re-salts of the start world's composition, and whether it fell back to `rock`.
    pub start_tries: u32,
    pub start_fallback: bool,
    pub basins: u32,
    pub systems: u16,
    /// The start world is `bodies[0]`.
    pub bodies: Vec<Body>,
    pub suites: Vec<Suite>,
    /// Milliseconds each suite took.
    pub suite_ms: Vec<f64>,
    /// Share of the nebula's mass that ended as debris.
    pub debris: f64,
    pub rings: u32,
    pub binaries: u32,
    /// Satellites that failed dominance and fell onto their parent, bodies past the bounds, and
    /// bodies past the cap.
    pub fell: u32,
    pub out_of_bounds: u32,
    pub capped: u32,
    /// Bodies whose final traits pick another painter than the ground their suite was checked in.
    pub unsettled: u32,
    pub storage: Storage,
    pub time: Timings,
}

/// Suites by composition, computed once each.
struct Suites {
    base: Element,
    seed: u32,
    /// What each composition is made of, and the milliseconds that took.
    found: Vec<(Found, f64)>,
    keys: Vec<([i8; 4], Ground)>,
    list: Vec<Suite>,
    /// Milliseconds each suite took (its composition's [`Found`] counted once, with the first).
    ms: Vec<f64>,
    /// Wall-clock milliseconds spent finding suites.
    wall: f64,
}

/// One suite to find: its composition, whether its [`Found`] is already known, and its ground.
type Job<'a> = ([i8; 4], Option<&'a Found>, Vec<Ground>);

impl Suites {
    fn new(base: Element, seed: u32) -> Self {
        Suites { base, seed, found: Vec::new(), keys: Vec::new(), list: Vec::new(), ms: Vec::new(), wall: 0.0 }
    }

    fn of(&mut self, law: &Law, comp: [i8; 4], ground: Ground, p: &Params) -> usize {
        match self.keys.iter().position(|&k| k == (comp, ground)) {
            Some(k) => k,
            None => {
                self.prefetch(law, &[(comp, ground)], p, 1);
                self.keys.len() - 1
            }
        }
    }

    /// Find the suites of `keys` that are missing, on up to `threads` threads (each is a pure
    /// function of its key, so the order they are found in changes nothing).
    fn prefetch(&mut self, law: &Law, keys: &[([i8; 4], Ground)], p: &Params, threads: usize) {
        let mut todo: Vec<([i8; 4], Vec<Ground>)> = Vec::new();
        for &(comp, ground) in keys {
            if self.keys.contains(&(comp, ground)) {
                continue;
            }
            match todo.iter_mut().find(|(c, _)| *c == comp) {
                Some((_, grounds)) if !grounds.contains(&ground) => grounds.push(ground),
                Some(_) => {}
                None => todo.push((comp, vec![ground])),
            }
        }
        if todo.is_empty() {
            return;
        }
        let t = Instant::now();
        let (base, seed) = (self.base, self.seed);
        let jobs: Vec<Job> = todo
            .into_iter()
            .map(|(comp, grounds)| (comp, self.found.iter().find(|(f, _)| f.comp == comp).map(|(f, _)| f), grounds))
            .collect();
        let chunk = jobs.len().div_ceil(threads.max(1));
        type Done = ([i8; 4], Option<(Found, f64)>, Vec<(Ground, Suite, f64)>);
        let done: Vec<Done> = std::thread::scope(|s| {
            let handles: Vec<_> = jobs
                .chunks(chunk)
                .map(|part| {
                    s.spawn(move || {
                        part.iter()
                            .map(|(comp, known, grounds)| {
                                let t = Instant::now();
                                let new = known.is_none().then(|| minerals::found(law, base, *comp, suite_seed(seed, *comp), p));
                                let first = ms(t);
                                let f = (*known).or(new.as_ref()).expect("found");
                                let suites = grounds
                                    .iter()
                                    .map(|&g| {
                                        let t = Instant::now();
                                        (g, minerals::suite(law, f, g, p), ms(t))
                                    })
                                    .collect();
                                (*comp, new.map(|f| (f, first)), suites)
                            })
                            .collect::<Vec<Done>>()
                    })
                })
                .collect();
            handles.into_iter().flat_map(|h| h.join().expect("a suite thread")).collect()
        });
        for (comp, new, suites) in done {
            let mut extra = 0.0;
            if let Some((f, first)) = new {
                extra = first;
                self.found.push((f, first));
            }
            for (g, suite, each) in suites {
                self.keys.push((comp, g));
                self.list.push(suite);
                self.ms.push(each + extra);
                extra = 0.0;
            }
        }
        self.wall += ms(t);
    }

    /// Wall-clock milliseconds spent on suites so far.
    fn spent(&self) -> f64 {
        self.wall
    }

    /// A suite that is not keyed by composition (fallbacks).
    fn push(&mut self, s: Suite) -> usize {
        self.list.push(s);
        self.ms.push(0.0);
        self.keys.push(([i8::MIN; 4], Ground::Face));
        self.list.len() - 1
    }
}

/// The seed of the suite of composition `comp` in a universe of nebula seed `seed`.
fn suite_seed(seed: u32, comp: [i8; 4]) -> u32 {
    hash32_3(seed, i32::from_le_bytes(comp.map(|v| v as u8)), 0, 0, 0x5017E)
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// A composition rounded to its suite key.
fn key(comp: [f64; 4]) -> [i8; 4] {
    comp.map(|v| v.round().clamp(-96.0, 96.0) as i8)
}

/// A body while the universe settles.
#[derive(Clone)]
struct Work {
    b: Proto,
    system: u16,
    partner: Option<usize>,
    parent: Option<usize>,
    alive: bool,
    suite: usize,
    half: f64,
    layout: Layout,
    pi: f64,
}

impl Work {
    fn new(b: Proto, system: u16) -> Self {
        let layout = Layout { form: Form::Cube, radius: 0.0, tilt: 0.0, clamped: false };
        Work { b, system, partner: None, parent: None, alive: true, suite: 0, half: 0.0, layout, pi: 0.0 }
    }

    /// Size, Π and layout from the body's suite.
    fn shape(&mut self, s: &Suite) {
        self.half = half_of(self.b.mass, s.density);
        self.pi = pi_g(s, self.half);
        self.layout = layout(self.pi, self.half);
    }

    /// Radius of the surface (datum radius, or a cube's half-size).
    fn radius(&self) -> f64 {
        if self.layout.form == Form::Round { self.layout.radius } else { self.half }
    }

    fn reach(&self) -> f64 {
        if self.layout.form == Form::Round { self.layout.radius } else { self.half * SQRT3 }
    }

    /// Pull at the surface (a cube's face centre).
    fn surface_pull(&self, s: &Suite) -> f64 {
        match self.layout.form {
            Form::Round => G * self.b.mass / (self.layout.radius * self.layout.radius),
            _ => KAPPA_FACE * G * s.density * self.half,
        }
    }
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|k| (a[k] - b[k]) * (a[k] - b[k])).sum::<f64>().sqrt()
}

/// Windowed pull of mass `m` at distance `d`.
fn pull_at(m: f64, d: f64) -> f64 {
    let d = d.max(1.0);
    G * m * window(d) / (d * d)
}

/// Gauss–Legendre nodes and weights on [−1, 1] (8 points).
const GL: [(f64, f64); 8] = [
    (-0.960_289_856_497_536_3, 0.101_228_536_290_376_3),
    (-0.796_666_477_413_626_7, 0.222_381_034_453_374_5),
    (-0.525_532_409_916_329_0, 0.313_706_645_877_887_3),
    (-0.183_434_642_495_649_8, 0.362_683_783_378_362_0),
    (0.183_434_642_495_649_8, 0.362_683_783_378_362_0),
    (0.525_532_409_916_329_0, 0.313_706_645_877_887_3),
    (0.796_666_477_413_626_7, 0.222_381_034_453_374_5),
    (0.960_289_856_497_536_3, 0.101_228_536_290_376_3),
];

/// Pull of a uniform cube (density `rho`, half `h`) on its axis at distance `d` from its centre.
fn cube_axis_pull(rho: f64, h: f64, d: f64) -> f64 {
    let mut sum = 0.0;
    for &(x, wx) in &GL {
        for &(y, wy) in &GL {
            for &(z, wz) in &GL {
                let dz = d - h * z;
                let r2 = h * h * (x * x + y * y) + dz * dz;
                sum += wx * wy * wz * dz / (r2 * r2.sqrt());
            }
        }
    }
    G * rho * h * h * h * sum
}

/// The smallest gap (a multiple of 16) at which each part of a contact binary pulls its own face
/// at least `k_dom` times harder than its partner does there.
fn binary_gap(rho: [f64; 2], half: [f64; 2], k_dom: f64) -> f64 {
    let ok = |gap: f64| {
        (0..2).all(|k| KAPPA_FACE * G * rho[k] * half[k] >= k_dom * cube_axis_pull(rho[1 - k], half[1 - k], half[1 - k] + gap))
    };
    let mut hi = 16.0;
    while !ok(hi) {
        hi *= 2.0;
    }
    let (mut lo, mut hi) = (0u64, (hi / 16.0) as u64);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if ok(mid as f64 * 16.0) { hi = mid } else { lo = mid }
    }
    hi as f64 * 16.0
}

/// Whether a qualifying pair binds: both keep their cube grid at their own size, and the impact
/// energy per mass is under `k_bin` times `Y/ρ` (a modelling choice). Accretion runs before any
/// body has minerals, so the matter is provisional: today's `rock` bulk.
fn binds(rock: &Suite, p: &Params, a: &Proto, b: &Proto, per_mass: f64) -> bool {
    let cube = |x: &Proto| {
        let half = half_of(x.mass, rock.density);
        layout(pi_g(rock, half), half).form != Form::Round
    };
    cube(a) && cube(b) && per_mass < p.k_bin * rock.yield_stress / rock.density
}

/// Unpack contact binaries into two bodies face to face, keeping their centre of mass.
fn expand(ws: &mut Vec<Work>, suites: &mut Suites, law: &Law, p: &Params) {
    for i in 0..ws.len() {
        let Some(pair) = ws[i].b.pair.take() else { continue };
        let com = ws[i].b.pos;
        let mut parts = [ws[i].clone(), ws[i].clone()];
        for k in 0..2 {
            parts[k].b.mass = pair.masses[k];
            parts[k].b.comp = pair.comps[k];
            parts[k].suite = suites.of(law, key(pair.comps[k]), Ground::Face, p);
            parts[k].shape(&suites.list[parts[k].suite]);
        }
        let rho = [0, 1].map(|k| suites.list[parts[k].suite].density);
        let half = [0, 1].map(|k| parts[k].half);
        let gap = binary_gap(rho, half, p.k_dom);
        let total = pair.masses[0] + pair.masses[1];
        let sep = half[0] + half[1] + gap;
        for k in 0..2 {
            let side = (if k == 0 { -pair.masses[1] } else { pair.masses[0] }) / total;
            parts[k].b.pos[pair.axis] = com[pair.axis] + pair.sign * sep * side;
        }
        let j = ws.len();
        let [a, mut b] = parts;
        b.partner = Some(i);
        ws[i] = a;
        ws[i].partner = Some(j);
        ws.push(b);
    }
}

/// Satellites must dominate their own surface: each body (lightest first) is a satellite of the
/// heavier body pulling hardest at its centre if its own surface pull is at least `k_dom` times
/// that body's pull there; otherwise it falls onto it. Repeats until stable. Returns the falls.
fn hierarchy(ws: &mut [Work], suites: &mut Suites, law: &Law, p: &Params, start: usize, grounds: &[Ground]) -> u32 {
    let mut falls = 0;
    loop {
        let mut order: Vec<usize> = (0..ws.len()).filter(|&i| ws[i].alive).collect();
        order.sort_by(|&a, &b| ws[a].b.mass.total_cmp(&ws[b].b.mass).then(a.cmp(&b)));
        let before = falls;
        for (rank, &i) in order.iter().enumerate() {
            let mut parent: Option<(f64, usize)> = None;
            for &c in &order[rank + 1..] {
                if !ws[c].alive || Some(c) == ws[i].partner {
                    continue;
                }
                let pull = pull_at(ws[c].b.mass, dist(ws[i].b.pos, ws[c].b.pos));
                if pull > 0.0 && parent.is_none_or(|(best, _)| pull > best) {
                    parent = Some((pull, c));
                }
            }
            ws[i].parent = parent.map(|(_, c)| c);
            let Some((_, c)) = parent else { continue };
            if i != start {
                let own = suites.of(law, key(ws[i].b.comp), grounds[i], p);
                ws[i].suite = own;
                ws[i].shape(&suites.list[own]);
                if ws[i].half < OWN_SUITE_HALF && ws[i].partner.is_none() {
                    let k = ws[c].suite;
                    ws[i].suite = k;
                    ws[i].shape(&suites.list[k]);
                }
            }
            let own = ws[i].surface_pull(&suites.list[ws[i].suite]);
            let d = (dist(ws[i].b.pos, ws[c].b.pos) - ws[i].radius()).max(1.0);
            if own < p.k_dom * pull_at(ws[c].b.mass, d) && i != start && ws[i].partner.is_none() {
                let from = ws[i].b.clone();
                fuse(&mut ws[c].b, &from, true);
                ws[i].alive = false;
                falls += 1;
                if c != start {
                    let k = suites.of(law, key(ws[c].b.comp), grounds[c], p);
                    ws[c].suite = k;
                    ws[c].shape(&suites.list[k]);
                }
            }
        }
        if falls == before {
            return falls;
        }
    }
}

/// The start world's suite: its own composition re-salted `tries` times, or `rock` past the last.
fn start_suite(law: &Law, suites: &mut Suites, p: &Params, comp: [f64; 4], tries: u32, seed: u32) -> usize {
    if tries > START_TRIES {
        return suites.push(minerals::role_suite(law, "rock"));
    }
    let salted: [f64; 4] = std::array::from_fn(|a| {
        let h = hash32_3(seed, tries as i32, a as i32, 0, 0x5A17);
        comp[a] + if tries == 0 { 0.0 } else { (h % 65) as f64 - 32.0 }
    });
    suites.of(law, key(salted), Ground::Face, p)
}

/// Three syllables or so from a seed, at most 12 letters.
fn name(seed: u32) -> [u8; 12] {
    const ONSET: [&str; 16] = ["", "b", "d", "k", "l", "m", "n", "r", "s", "t", "v", "z", "th", "sh", "kr", "dr"];
    const VOWEL: [&str; 8] = ["a", "e", "i", "o", "u", "ae", "io", "au"];
    const CODA: [&str; 8] = ["", "", "n", "r", "s", "l", "th", "x"];
    let mut s = String::new();
    for k in 0..2 + seed % 3 {
        let h = hash32_3(seed, k as i32, 0, 0, 0x9A3E);
        s.push_str(ONSET[(h & 15) as usize]);
        s.push_str(VOWEL[(h >> 4 & 7) as usize]);
        s.push_str(CODA[(h >> 7 & 7) as usize]);
    }
    let mut out = [0u8; 12];
    for (o, b) in out.iter_mut().zip(s.bytes()) {
        *o = b;
    }
    out[0] = out[0].to_ascii_uppercase();
    out
}

/// Pack the bodies' storage: round atlases and warped boxes first-fit-decreasing into shelf rows
/// along x (from the storage border to `i32::MAX`), rows stacked in z below 1e9; warped boxes also
/// need unique y ranges. What does not fit is demoted.
fn storage(bodies: &[Body]) -> Storage {
    let x_budget = i32::MAX as i64 - STORAGE_X0;
    let z_budget = 1_000_000_000i64;
    let mut items: Vec<(i64, i64, bool)> = Vec::new();
    let mut demoted = 0;
    let mut cube_y = 0i64;
    for b in bodies {
        match b.traits.form {
            Form::Round => {
                // An atlas spans about π·R in x; past the budget it cannot be stored at all.
                if b.radius as f64 * 3.3 > x_budget as f64 {
                    demoted += 1;
                    continue;
                }
                let a = Atlas::new(glam::DVec3::ZERO, b.radius, b.radius + RELIEF, false, STORAGE_X0);
                let z = a.patches().map(|p| a.storage_box(p)).map(|(o, s)| o[2] + s[2]).max().unwrap_or(0);
                items.push((a.next_x() - STORAGE_X0, z, false));
            }
            Form::Warped => {
                let size = 2 * (b.half + RELIEF);
                items.push((size, size, true));
            }
            Form::Cube => {}
        }
    }
    items.sort_by(|a, b| b.1.cmp(&a.1).then(b.0.cmp(&a.0)));
    let mut rows: Vec<(i64, i64)> = Vec::new();
    let mut z_used = 0i64;
    for (x, z, cube) in items {
        if cube && cube_y + z > x_budget {
            demoted += 1;
            continue;
        }
        if let Some(row) = rows.iter_mut().find(|r| x_budget - r.0 >= x && r.1 >= z) {
            row.0 += x;
        } else if z_used + z <= z_budget && x <= x_budget {
            rows.push((x, z));
            z_used += z;
        } else {
            demoted += 1;
            continue;
        }
        if cube {
            cube_y += z;
        }
    }
    Storage {
        rows: rows.len() as u32,
        row0: rows.first().map_or(0.0, |r| r.0 as f64 / x_budget as f64),
        z_used: z_used as f64 / z_budget as f64,
        cube_y: cube_y as f64 / x_budget as f64,
        demoted,
    }
}

/// What settling a universe gives.
struct Settled {
    bodies: Vec<Body>,
    start_tries: u32,
    start_fallback: bool,
    fell: u32,
    out_of_bounds: u32,
    capped: u32,
    unsettled: u32,
    rings: u32,
    debris: f64,
}

/// Calibrate to the start world, unpack binaries, settle the hierarchy, place, bound and give
/// traits. `None` when the start world misses its requirements with this suite.
fn settle(
    protos: &[(u16, Proto)],
    debris: &[([f64; 3], f64)],
    suites: &mut Suites,
    law: &Law,
    p: &Params,
    seed: u32,
    tries: u32,
    threads: usize,
) -> Option<Settled> {
    // The start world: the heaviest single body of the start system (system 0).
    let start = (0..protos.len())
        .filter(|&i| protos[i].0 == 0)
        .max_by(|&a, &b| {
            let single = |i: usize| protos[i].1.pair.is_none();
            single(a).cmp(&single(b)).then(protos[a].1.mass.total_cmp(&protos[b].1.mass)).then(b.cmp(&a))
        })?;
    let s_suite = start_suite(law, suites, p, protos[start].1.comp, tries, seed);
    let rho = suites.list[s_suite].density;
    let r_s = RHO_R / rho;
    let half_s = HALF_PER_RADIUS * r_s;
    let m_s = rho * 8.0 * half_s * half_s * half_s;
    let mut ws: Vec<Work> = protos.iter().map(|(sys, b)| Work::new(b.clone(), *sys)).collect();
    // Place: the start world's centre at (0, −R, 0).
    let shift: [f64; 3] = std::array::from_fn(|a| (if a == 1 { -r_s } else { 0.0 }) - ws[start].b.pos[a]);
    for w in ws.iter_mut() {
        w.b.pos = std::array::from_fn(|a| w.b.pos[a] + shift[a]);
    }
    let mut scale = m_s / ws[start].b.mass;
    let mut calibration = 1.0;
    let (mut fell, mut out_of_bounds, mut capped, mut lost) = (0, 0, 0, 0.0);
    // Each body's suite is checked against the ground of the painter its traits pick, and the
    // traits read the suite: start from the face painter's ground and settle both together.
    let mut grounds = vec![Ground::Face; ws.len()];
    for round in 0..CALIBRATIONS {
        if scale != 1.0 {
            calibration *= scale;
            for w in ws.iter_mut() {
                w.b.mass *= scale;
                if let Some(pair) = &mut w.b.pair {
                    pair.masses = pair.masses.map(|m| m * scale);
                }
            }
        }
        if round == 0 {
            // Far bodies (past the satellites' bound: ranks are not known yet) become debris and
            // only the heaviest stay (a binary counts twice), so suites are found only for bodies
            // that are kept, in parallel.
            for (i, w) in ws.iter_mut().enumerate() {
                if i != start && w.b.pos.iter().any(|v| v.abs() > BOUND_SATELLITE) {
                    w.alive = false;
                    out_of_bounds += 1;
                    lost += w.b.mass;
                }
            }
            let mut heavy: Vec<usize> = (0..ws.len()).filter(|&i| ws[i].alive).collect();
            heavy.sort_by(|&a, &b| (b == start).cmp(&(a == start)).then(ws[b].b.mass.total_cmp(&ws[a].b.mass)).then(a.cmp(&b)));
            let mut count = 0;
            for &i in &heavy {
                count += if ws[i].b.pair.is_some() { 2 } else { 1 };
                if count > p.bodies_max {
                    ws[i].alive = false;
                    capped += 1;
                    lost += ws[i].b.mass;
                }
            }
            let keys: Vec<([i8; 4], Ground)> = (0..ws.len())
                .filter(|&i| ws[i].alive && i != start)
                .flat_map(|i| match &ws[i].b.pair {
                    Some(pair) => vec![key(pair.comps[0]), key(pair.comps[1])],
                    None => vec![key(ws[i].b.comp)],
                })
                .map(|k| (k, Ground::Face))
                .collect();
            suites.prefetch(law, &keys, p, threads);
            expand(&mut ws, suites, law, p);
            grounds.resize(ws.len(), Ground::Face);
        }
        assign(&mut ws, suites, law, p, start, s_suite, &grounds);
        let before = ws[start].b.mass;
        fell += hierarchy(&mut ws, suites, law, p, start, &grounds);
        let next = grounds_of(&ws, start, &lite(&ws, start, r_s, suites, p));
        let changed = next != grounds;
        if changed {
            let keys: Vec<([i8; 4], Ground)> =
                (0..ws.len()).filter(|&i| ws[i].alive && i != start).map(|i| (key(ws[i].b.comp), next[i])).collect();
            suites.prefetch(law, &keys, p, threads);
            grounds = next;
        }
        scale = if ws[start].b.mass == before { 1.0 } else { m_s / ws[start].b.mass };
        if !changed && scale == 1.0 {
            break;
        }
    }
    assign(&mut ws, suites, law, p, start, s_suite, &grounds);
    // The last resort (`rock`) is accepted as it is.
    let last = tries > START_TRIES;
    if ws[start].layout.form != Form::Round && !last {
        return None;
    }
    // Primaries past the tighter bound become debris too.
    for (i, w) in ws.iter_mut().enumerate() {
        if w.alive && i != start && w.parent.is_none() && w.b.pos.iter().any(|v| v.abs() > BOUND) {
            w.alive = false;
            out_of_bounds += 1;
            lost += w.b.mass;
        }
    }
    // Systems never pull on each other: a body too near another system's falls onto its primary.
    loop {
        let alive: Vec<usize> = (0..ws.len()).filter(|&i| ws[i].alive).collect();
        let bad = alive.iter().flat_map(|&a| alive.iter().map(move |&b| (a, b))).find(|&(a, b)| {
            a < b && ws[a].system != ws[b].system && dist(ws[a].b.pos, ws[b].b.pos) <= R_G + ws[a].reach() + ws[b].reach()
        });
        let Some((a, b)) = bad else { break };
        let light = if ws[a].b.mass <= ws[b].b.mass && a != start { a } else { b };
        let mut root = light;
        while let Some(up) = ws[root].parent.filter(|&u| ws[u].alive) {
            root = up;
        }
        if root == light {
            root = if light == a { b } else { a };
        }
        let from = ws[light].b.clone();
        fuse(&mut ws[root].b, &from, true);
        ws[light].alive = false;
        fell += 1;
        if root != start {
            let k = suites.of(law, key(ws[root].b.comp), grounds[root], p);
            ws[root].suite = k;
            ws[root].shape(&suites.list[k]);
        }
    }
    let traits = lite(&ws, start, r_s, suites, p);
    let settled_grounds = grounds_of(&ws, start, &traits);
    let unsettled = (0..ws.len()).filter(|&i| ws[i].alive && settled_grounds[i] != grounds[i]).count() as u32;
    let mut keep: Vec<usize> = (0..ws.len()).filter(|&i| ws[i].alive).collect();
    // Ids by (rank, distance to the start world).
    let rank = |i: usize| {
        if i == start {
            Rank::Start
        } else if ws[i].partner.is_some() {
            Rank::Partner
        } else if ws[i].parent.is_some() {
            Rank::Satellite
        } else {
            Rank::Primary
        }
    };
    let origin = [0.0, -r_s, 0.0];
    keep.sort_by(|&a, &b| {
        rank(a).cmp(&rank(b)).then(dist(ws[a].b.pos, origin).total_cmp(&dist(ws[b].b.pos, origin))).then(a.cmp(&b))
    });
    let id_of = |i: usize| keep.iter().position(|&k| k == i).map(|k| k as u16);
    let mut bodies = Vec::with_capacity(keep.len());
    for (k, &i) in keep.iter().enumerate() {
        let (w, t) = (&ws[i], &traits[i]);
        let s = &suites.list[w.suite];
        let seed_b = hash32_3(seed, w.system as i32, i as i32, 0, 0xB0D1);
        let form = if i == start { Form::Round } else { w.layout.form };
        bodies.push(Body {
            id: k as u16,
            system: w.system,
            centre: w.b.pos.map(|v| (v / 16.0).round() as i64 * 16),
            half: (w.half / 16.0).round() as i64 * 16,
            radius: if form == Form::Round { (t.radius / 16.0).round() as i64 * 16 } else { 0 },
            density: s.density,
            seed: seed_b,
            comp: key(w.b.comp),
            ground: grounds[i],
            traits: Traits {
                rank: rank(i),
                form,
                parent: w.parent.and_then(id_of),
                partner: w.partner.and_then(id_of),
                mass: w.b.mass,
                yield_stress: s.yield_stress,
                pi_g: w.pi,
                pi_clamped: w.layout.clamped,
                gravity: t.g as f32,
                heat: (t.heat * 256.0).clamp(0.0, 65_535.0) as u16,
                glow: t.glow,
                air_top: t.air.then(|| (AIR_SCALE * t.temp / t.g).min(AIR_MAX) as i32),
                temp: t.temp.clamp(-32_768.0, 32_767.0) as i16,
                wet: (t.wet * 255.0) as u8,
                life: (t.life * 255.0) as u8,
                suite: w.suite as u16,
                name: name(seed_b),
            },
            history: History { impacts: w.b.impacts.clone(), heat_in: w.b.heat_in },
        });
    }
    let t = &bodies[0].traits;
    if (t.air_top.is_none() || !(TEMPERATE.0..=TEMPERATE.1).contains(&(t.temp as f64))) && !last {
        return None;
    }
    // Rings: debris between 3 and 5 reaches of a body, heavier than `ring_min` in all.
    let mut rings = 0;
    for b in &bodies {
        let c = b.centre.map(|v| v as f64);
        let ring: f64 = debris
            .iter()
            .map(|(at, m)| (std::array::from_fn::<f64, 3, _>(|a| at[a] + shift[a]), m))
            .filter(|(at, _)| (3.0 * b.reach()..=5.0 * b.reach()).contains(&dist(*at, c)))
            .map(|(_, m)| *m)
            .sum();
        rings += (ring > p.ring_min) as u32;
    }
    // Debris in the nebula's calibration (the bodies were rescaled since).
    let debris_mass = debris.iter().map(|(_, m)| m).sum::<f64>() + lost / calibration;
    Some(Settled {
        bodies,
        start_tries: tries,
        start_fallback: last,
        fell,
        out_of_bounds,
        capped,
        unsettled,
        rings,
        debris: debris_mass,
    })
}

/// Give every live body the suite of its composition in its ground (the start world keeps its own)
/// and the shape that suite asks for.
fn assign(ws: &mut [Work], suites: &mut Suites, law: &Law, p: &Params, start: usize, s_suite: usize, grounds: &[Ground]) {
    for i in 0..ws.len() {
        if ws[i].alive {
            let k = if i == start { s_suite } else { suites.of(law, key(ws[i].b.comp), grounds[i], p) };
            ws[i].suite = k;
            ws[i].shape(&suites.list[k]);
        }
    }
}

/// What the traits read, per body.
#[derive(Clone, Copy, Debug, Default)]
struct Lite {
    radius: f64,
    g: f64,
    heat: f64,
    glow: bool,
    air: bool,
    temp: f64,
    wet: f64,
    life: f64,
}

/// The traits of every live body (indexed like `ws`).
///
/// Life and glow (a modelling choice): a body glows when its heat reaches `h_glow` and either its
/// suite glows or the heat reaches `h_melt`, where its crust melts into the palette's emissive
/// magma. Life is comfort × wetness × organics; a body with air in the temperate band carries the
/// palette's organic ground roles (soil, moss, timber), [`ORGANICS`] of its surface, whatever its
/// suite holds.
fn lite(ws: &[Work], start: usize, r_s: f64, suites: &Suites, p: &Params) -> Vec<Lite> {
    let radius = |i: usize| if i == start { r_s } else { ws[i].radius() };
    let heat_raw = |i: usize| {
        let s = &suites.list[ws[i].suite];
        let emissive = s.minerals.iter().filter(|m| m.emission > 0).count() as f64 / s.minerals.len() as f64;
        G * ws[i].b.mass / radius(i) + ws[i].b.heat_in + E_RAD * emissive
    };
    let heat0 = heat_raw(start);
    let mut out = vec![Lite::default(); ws.len()];
    for i in (0..ws.len()).filter(|&i| ws[i].alive) {
        let heat = heat_raw(i) / heat0;
        out[i].heat = heat;
        out[i].glow = heat >= p.h_glow && (suites.list[ws[i].suite].glow || heat >= p.h_melt);
        out[i].radius = radius(i);
    }
    for i in (0..ws.len()).filter(|&i| ws[i].alive) {
        let (w, s) = (&ws[i], &suites.list[ws[i].suite]);
        let r = out[i].radius;
        let g = if i == start { G * w.b.mass / (r * r) } else { w.surface_pull(s) };
        let irr: f64 = (0..ws.len())
            .filter(|&o| ws[o].alive && out[o].glow && o != i && ws[o].system == w.system)
            .map(|o| {
                let d = dist(w.b.pos, ws[o].b.pos);
                out[o].heat * out[o].radius * out[o].radius * window(d) / (d * d)
            })
            .sum();
        let heat = out[i].heat;
        let bare = T_BASE + T_HEAT * heat + T_IRR * irr;
        let air = g * r >= p.beta * bare * (1.0 + (heat - 1.0).max(0.0));
        let temp = bare + if air { GREENHOUSE } else { 0.0 };
        let wet = if air { s.classes[3] } else { 0.0 };
        let comfort = (1.0 - ((temp - T_HOME) / COMFORT) * ((temp - T_HOME) / COMFORT)).max(0.0);
        let temperate = air && (TEMPERATE.0..=TEMPERATE.1).contains(&temp);
        let organics = if temperate { s.classes[2].max(ORGANICS) } else { s.classes[2] };
        out[i] = Lite { g, air, temp, wet, life: comfort * wet * organics, ..out[i] };
    }
    out
}

/// The painter each live body gets (design P2): the start world and cubes the face painter, round
/// bodies Ember when they glow, Verdant when they carry life, Moon otherwise.
fn grounds_of(ws: &[Work], start: usize, traits: &[Lite]) -> Vec<Ground> {
    (0..ws.len())
        .map(|i| {
            let t = &traits[i];
            if i == start || !ws[i].alive || ws[i].layout.form != Form::Round {
                Ground::Face
            } else if t.glow {
                Ground::Ember
            } else if t.air && t.life > 0.0 {
                Ground::Verdant
            } else {
                Ground::Moon
            }
        })
        .collect()
}

/// The nebula seed of a universe seed after `resalt` interest re-salts (all 64 bits count).
pub fn nebula_seed(seed: u64, resalt: u32) -> u32 {
    hash32_3(0x5EED, seed as i32, (seed >> 32) as i32, resalt as i32, 0)
}

impl Universe {
    /// Grow the universe of `seed` under the knobs `cfg`. `threads` bounds the nebula's passes.
    pub fn new(seed: u64, cfg: &TerrainCfg, p: &Params, threads: usize) -> Universe {
        let t_all = Instant::now();
        let law = Law::current();
        let mut time = Timings::default();
        let mut resalts = 0;
        loop {
            let s = nebula_seed(seed, resalts);
            let t = Instant::now();
            let neb = Nebula::new(s, cfg, p, threads);
            time.nebula += ms(t);
            let t = Instant::now();
            let mut suites = Suites::new(neb.base, s);
            let rock = minerals::role_suite(&law, "rock");
            let scale = p.origin_share * M_HOME / neb.systems[0].mass;
            let mut protos: Vec<(u16, Proto)> = Vec::new();
            let mut debris: Vec<([f64; 3], f64)> = Vec::new();
            let mut binaries = 0;
            for (k, sys) in neb.systems.iter().enumerate() {
                let mut bind = |a: &Proto, b: &Proto, e: f64| binds(&rock, p, a, b, e);
                let acc = accrete::accrete(&neb, sys, scale, hash32_3(s, k as i32, 0, 0, 0xACC), p, &mut bind);
                binaries += acc.binaries;
                protos.extend(acc.bodies.into_iter().map(|b| (k as u16, b)));
                debris.extend(acc.debris);
            }
            let spent = suites.spent();
            time.accrete += ms(t) - spent;
            let t = Instant::now();
            let settled = (0..=START_TRIES + 1)
                .find_map(|tries| settle(&protos, &debris, &mut suites, &law, p, s, tries, threads))
                .expect("the start system has bodies");
            time.settle += ms(t) - (suites.spent() - spent);
            time.suites += suites.spent();
            let others = settled.bodies.len() - 1;
            if others >= OTHERS_MIN || resalts == INTEREST_TRIES {
                let t = Instant::now();
                let storage = storage(&settled.bodies);
                time.storage += ms(t);
                time.total = ms(t_all);
                let total = neb.total * scale;
                return Universe {
                    seed,
                    base: neb.base,
                    resalts,
                    start_tries: settled.start_tries,
                    start_fallback: settled.start_fallback,
                    basins: neb.basins,
                    systems: neb.systems.len() as u16,
                    bodies: settled.bodies,
                    suites: suites.list,
                    suite_ms: suites.ms,
                    debris: (settled.debris + neb.residual * scale) / total,
                    rings: settled.rings,
                    binaries,
                    fell: settled.fell,
                    out_of_bounds: settled.out_of_bounds,
                    capped: settled.capped,
                    unsettled: settled.unsettled,
                    storage,
                    time,
                };
            }
            resalts += 1;
        }
    }
}

#[cfg(test)]
mod tests;
