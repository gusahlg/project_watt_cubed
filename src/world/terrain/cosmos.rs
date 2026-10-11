//! The cosmos: the seeded catalog of every body in the universe — the start world, the few big
//! worlds far away, their moons, and the sparse asteroid clusters between them. It is the single
//! source the generator paints from, gravity sums (as an analytic mass oracle) and the sky draws
//! distant bodies from. Everything is a pure function of the seed; empty space answers in O(1).
//!
//! Layout rules: big bodies sit on a shell 4.5e8–8e8 blocks from the start world, pairwise ≥ 4e8
//! apart, inside ±9.2e8; clusters live one per `CELL` (with a small probability) and paint only
//! inside their cell; a cluster's rocks live one per sub-cell of their size class and paint only
//! inside their sub-cell, so any box finds its rocks by looking at the sub-cells it overlaps.

use std::collections::HashMap;

use glam::DVec3;

use super::emergent::minerals::Suite;
use super::noise::{hash3, unit};
use crate::gravity::{MassOracle, Primitive, Shape as MassShape, Summary, Visitor, G};

/// Radius of the start world. The sphere has the volume of the old 50_000_000-block cube:
/// `round16(50_000_000 · (3/(4π))^(1/3))`.
pub const HOME_RADIUS: i64 = 31_017_520;
/// Centre of the start world: the top of its +Y chart sits near `y = 0`.
pub const HOME_CENTRE: [i64; 3] = [0, -HOME_RADIUS, 0];
/// Half-size of the start world's cube of matter (generation places a cube; physics rounds it to
/// the ball of [`HOME_RADIUS`] plus its relaxed relief).
pub const HOME_CUBE_HALF: i64 = 25_000_000;
/// Design mean amount per cell of a body's bulk (the generator's bulk mix honours it).
pub const BULK_DENSITY: f64 = 5.0;
/// Most a generated surface rises above (or sinks below) its datum.
pub const RELIEF: i64 = 2_048;
/// Air extends this far above a body's datum. Higher, and between bodies, is vacuum.
pub const AIR_TOP: f64 = 20_000.0;

/// Edge of a cluster cell (2^24 blocks).
const CELL: i64 = 1 << 24;
/// Cells per axis searched for clusters (covers ±1e9).
const CELL_SPAN: i64 = 60;
/// Edge of a super-cell grouping clusters for gravity queries (8³ cells).
const SUPER: i64 = CELL * 8;

/// What a body is made like. Content flavour only: physics never reads it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Kind {
    /// The start world, a sphere-ish ball painted on cube-sphere charts.
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

impl Kind {
    /// Catalog name, lower case.
    pub fn name(self) -> &'static str {
        match self {
            Kind::Home => "home",
            Kind::Twin => "twin",
            Kind::Verdant => "verdant",
            Kind::Hollow => "hollow",
            Kind::Ember => "ember",
            Kind::Moon => "moon",
        }
    }
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

/// Where a body sits in its system.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Rank {
    /// The start world (always id 0).
    Start,
    /// A body no heavier body holds.
    Primary,
    /// One half of a contact binary.
    Partner,
    /// A body held by a heavier one (its `parent`).
    Satellite,
}

/// What a body's shape asks for.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Form {
    /// Keeps its cube grid, sagging at most half a block.
    Cube,
    /// Keeps its cube grid but sags: a warp and a storage box.
    Warped,
    /// Charted on a cube-sphere.
    Round,
    /// A hollow ball.
    Shell,
}

/// What a body is, derived from its matter and its place (design v5 §3.4). Painters, the sky and
/// mods read these; physics reads only the shape and density.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Traits {
    pub rank: Rank,
    pub form: Form,
    /// The body holding a satellite.
    pub parent: Option<u16>,
    /// The other half of a contact binary.
    pub partner: Option<u16>,
    /// The shell a free core sits inside.
    pub core_of: Option<u16>,
    /// Matter, in amount.
    pub mass: f64,
    /// Yield stress of its bulk, amount/(block·s²).
    pub yield_stress: f64,
    /// `Π_g` of its cube of matter.
    pub pi_g: f64,
    /// Surface pull, m/s².
    pub gravity: f32,
    /// Q8.8, 256 = the start world.
    pub heat: u16,
    pub glow: bool,
    /// How far above the datum air reaches; `None` for an airless body.
    pub air_top: Option<i32>,
    /// Surface temperature, kelvin.
    pub temp: i16,
    /// Wetness and life, 0..=255.
    pub wet: u8,
    pub life: u8,
    /// Index into the cosmos's mineral suites.
    pub suite: u16,
    /// ASCII, zero-padded.
    pub name: [u8; 12],
}

impl Traits {
    /// The traits of an authored catalog entry: `kind` and `shape` say everything; the links
    /// (parent, partner, core) are filled once the catalog is known.
    #[cfg(test)]
    pub fn of(kind: Kind, shape: Shape) -> Self {
        let form = match shape {
            Shape::Cube { .. } => Form::Cube,
            Shape::Ball { .. } => Form::Round,
            Shape::Shell { .. } => Form::Shell,
        };
        let rank = match kind {
            Kind::Home => Rank::Start,
            Kind::Twin => Rank::Partner,
            Kind::Moon => Rank::Satellite,
            Kind::Verdant | Kind::Hollow | Kind::Ember => Rank::Primary,
        };
        let living = matches!(kind, Kind::Home | Kind::Verdant);
        let mut name = [0u8; 12];
        for (o, b) in name.iter_mut().zip(kind.name().bytes()) {
            *o = b;
        }
        name[0] = name[0].to_ascii_uppercase();
        Self {
            rank,
            form,
            parent: None,
            partner: None,
            core_of: None,
            mass: 0.0,
            yield_stress: 0.0,
            pi_g: 0.0,
            gravity: 0.0,
            heat: 256,
            glow: kind == Kind::Ember,
            air_top: Some(AIR_TOP as i32),
            temp: 288,
            wet: if living { 128 } else { 0 },
            life: if living { 128 } else { 0 },
            suite: 0,
            name,
        }
    }

    /// Today's kind of a body with these traits, kept one release for mods and tests: id 0 is
    /// home, a free core the Ember, a satellite a moon, a cube partner a twin, a shell the Hollow,
    /// anything else Verdant.
    pub fn legacy_kind(&self, id: u16) -> Kind {
        if id == 0 || self.rank == Rank::Start {
            Kind::Home
        } else if self.core_of.is_some() {
            Kind::Ember
        } else if self.rank == Rank::Satellite {
            Kind::Moon
        } else if self.partner.is_some() && matches!(self.form, Form::Cube | Form::Warped) {
            Kind::Twin
        } else if self.form == Form::Shell {
            Kind::Hollow
        } else {
            Kind::Verdant
        }
    }

    /// A label of what the body is, for the UI, mods and tests: "start world", "binary cube",
    /// "cube moon", "glowing world", "living moon", "airless world", ...
    pub fn class(&self) -> &'static str {
        let moon = self.rank == Rank::Satellite;
        let pick = |m: &'static str, w: &'static str| if moon { m } else { w };
        match (self.rank, self.form) {
            (Rank::Start, _) => "start world",
            _ if self.core_of.is_some() => "core",
            (Rank::Partner, _) => "binary cube",
            (_, Form::Shell) => "hollow world",
            (_, Form::Cube | Form::Warped) => pick("cube moon", "cube world"),
            _ if self.glow => pick("glowing moon", "glowing world"),
            _ if self.life > 0 => pick("living moon", "living world"),
            _ if self.air_top.is_none() => pick("airless moon", "airless world"),
            _ => pick("barren moon", "barren world"),
        }
    }

    /// The name as text.
    pub fn name(&self) -> &str {
        let end = self.name.iter().position(|&b| b == 0).unwrap_or(self.name.len());
        std::str::from_utf8(&self.name[..end]).unwrap_or("?")
    }
}

/// One big body (start world, worlds, moons).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Body {
    /// Stable index into the catalog: `bodies()[id]`.
    pub id: u16,
    /// Today's kind, derived from `traits` ([`Traits::legacy_kind`]); kept one release for mods.
    pub kind: Kind,
    pub centre: [i64; 3],
    pub shape: Shape,
    /// Mean amount per cell of its bulk.
    pub density: f64,
    pub seed: u32,
    pub traits: Traits,
}

impl Body {
    /// An entry of the authored catalog: its traits follow from `kind` and `shape`.
    #[cfg(test)]
    fn authored(id: u16, kind: Kind, centre: [i64; 3], shape: Shape, seed: u32) -> Self {
        Body { id, kind, centre, shape, density: BULK_DENSITY, seed, traits: Traits::of(kind, shape) }
    }

    /// The body's name.
    pub fn name(&self) -> &str {
        self.traits.name()
    }

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
pub enum Spread {
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
    pub form: Spread,
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
    /// What the bodies are made of, indexed by [`Traits::suite`].
    suites: Vec<Suite>,
    clusters: Vec<Cluster>,
    by_cell: HashMap<[i32; 3], u32>,
    /// Clusters grouped by super-cell, with their summed expected mass.
    groups: Vec<Group>,
    /// Group index by super-cell (a dense grid over the bounded universe; `u32::MAX` = none), so a
    /// query visits only the super-cells within the law's range.
    group_at: Vec<u32>,
    /// Interior-feature density as a fraction of the design (1 = as designed). The Heart stays.
    deep: f32,
    /// Each round body's relaxed shape by body id: its datum and that relief as a gravity source.
    reliefs: Vec<Option<Relaxed>>,
    /// A warped cube's true shape for gravity (body id). Replaces that body's box primitive.
    shapes: Vec<(u16, std::sync::Arc<crate::gravity::Polyhedron>)>,
    /// Warped cubes' displacement fields (altitude and air are measured on the bent grid).
    warps: Vec<(u16, std::sync::Arc<crate::space::warp::Warp>)>,
    /// Bodies by super-cell, so a box or point query visits only the bodies that can reach it.
    index: BodyIndex,
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

/// Take body `id` out of the catalog (its matter becomes debris): later ids move down one, and what
/// it held or was bound to stands alone.
fn demote(bodies: &mut Vec<Body>, id: u16) {
    bodies.remove(usize::from(id));
    let fix = |link: Option<u16>| link.filter(|&l| l != id).map(|l| if l > id { l - 1 } else { l });
    for b in bodies.iter_mut() {
        if b.id > id {
            b.id -= 1;
        }
        let t = &mut b.traits;
        let (parent, partner) = (fix(t.parent), fix(t.partner));
        if (t.parent.is_some() && parent.is_none()) || (t.partner.is_some() && partner.is_none()) {
            t.rank = Rank::Primary;
        }
        (t.parent, t.partner, t.core_of) = (parent, partner, fix(t.core_of));
        b.kind = t.legacy_kind(b.id);
    }
}

#[cfg(test)]
/// Fill the links between authored bodies: the twins are each other's partners, a moon's parent is
/// the nearest body that is neither a moon nor the Ember, and the Ember is the Hollow's core.
fn link(bodies: &mut [Body]) {
    let all = bodies.to_vec();
    for b in bodies.iter_mut() {
        let c = b.centre_f();
        let t = &mut b.traits;
        match b.kind {
            Kind::Twin => t.partner = all.iter().find(|o| o.kind == Kind::Twin && o.id != b.id).map(|o| o.id),
            Kind::Moon => {
                t.parent = all
                    .iter()
                    .filter(|o| o.id != b.id && o.kind != Kind::Moon && o.kind != Kind::Ember)
                    .min_by(|x, y| (x.centre_f() - c).length_squared().total_cmp(&(y.centre_f() - c).length_squared()))
                    .map(|o| o.id);
            }
            Kind::Ember => t.core_of = all.iter().find(|o| o.kind == Kind::Hollow && o.centre == b.centre).map(|o| o.id),
            _ => {}
        }
    }
    // The lush twin is the one with the smaller (seed, id); its partner is crystalline.
    let lush = all.iter().filter(|o| o.kind == Kind::Twin).min_by_key(|o| (o.seed, o.id)).map(|o| o.id);
    for b in bodies.iter_mut().filter(|b| b.kind == Kind::Twin) {
        let living = Some(b.id) == lush;
        b.traits.life = if living { 128 } else { 0 };
        b.traits.wet = b.traits.life;
    }
}

#[cfg(test)]
/// A position snapped to the chunk grid (multiples of 16), so a body's faces and columns align
/// with chunks.
fn to_i64(v: DVec3) -> [i64; 3] {
    let snap = |x: f64| (x / 16.0).round() as i64 * 16;
    [snap(v.x), snap(v.y), snap(v.z)]
}

impl Cosmos {
    /// The universe of `seed`, with the asteroid-cluster density scaled by `space` (1 = designed).
    pub fn new(seed: u64, space: f32) -> Self {
        Self::with_deep(seed, space, 1.0)
    }

    /// [`new`](Self::new), with the interior-feature density scaled by `deep` (1 = designed).
    pub fn with_deep(seed: u64, space: f32, deep: f32) -> Self {
        let percent = |v: f32| (v * 100.0).round().clamp(0.0, 200.0) as u16;
        Self::grow(seed, &super::TerrainCfg { space: percent(space), deep: percent(deep), ..super::TerrainCfg::default() })
    }

    /// The universe grown from `seed` (all 64 bits) under the knobs `cfg` (design v5 stages 1-7):
    /// a nebula collapses into systems, their matter accretes into bodies, each body's minerals
    /// come from the law, and its matter decides its shape. Bodies that do not fit storage become
    /// debris, smallest first.
    pub fn grow(seed: u64, cfg: &super::TerrainCfg) -> Self {
        let (space, deep) = (cfg.space as f32 / 100.0, cfg.deep as f32 / 100.0);
        let seed32 = (seed ^ (seed >> 32)) as u32;
        #[cfg(test)]
        if fixture::AUTHORED.with(std::cell::Cell::get) {
            return Self::authored(seed32 ^ 0x1D1F_F051, space, deep);
        }
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(4);
        let universe = super::emergent::Universe::new(seed, cfg, &super::emergent::Params::default(), threads);
        let mut bodies: Vec<Body> = universe
            .bodies
            .iter()
            .map(|b| {
                let shape = if b.traits.form == Form::Round { Shape::Ball { r: b.radius } } else { Shape::Cube { half: b.half } };
                Body { id: b.id, kind: b.traits.legacy_kind(b.id), centre: b.centre, shape, density: b.density, seed: b.seed, traits: b.traits }
            })
            .collect();
        loop {
            let warped: Vec<u16> = bodies.iter().filter(|b| b.traits.form == Form::Warped).map(|b| b.id).collect();
            if super::storage::plan(&bodies, &warped).is_ok() {
                break;
            }
            let smallest = bodies
                .iter()
                .filter(|b| b.traits.rank != Rank::Start && matches!(b.traits.form, Form::Round | Form::Warped))
                .min_by(|a, b| a.traits.mass.total_cmp(&b.traits.mass).then(a.id.cmp(&b.id)))
                .map(|b| b.id)
                .expect("the start world alone fits storage");
            demote(&mut bodies, smallest);
        }
        Self::assemble(seed32, bodies, universe.suites, space, deep)
    }

    /// The catalog with its index and clusters.
    fn assemble(seed: u32, bodies: Vec<Body>, suites: Vec<Suite>, space: f32, deep: f32) -> Self {
        let mut cosmos = Self {
            seed,
            bodies,
            suites,
            clusters: Vec::new(),
            by_cell: HashMap::new(),
            groups: Vec::new(),
            group_at: vec![u32::MAX; (GRID_SIDE * GRID_SIDE * GRID_SIDE) as usize],
            deep: deep.clamp(0.0, 2.0),
            reliefs: Vec::new(),
            shapes: Vec::new(),
            warps: Vec::new(),
            index: BodyIndex::default(),
        };
        cosmos.index = BodyIndex::new(&cosmos.bodies);
        cosmos.place_clusters(space);
        cosmos
    }

    /// Today's authored catalog (home, the twins, Verdance, the Hollow and its Ember, moons), every
    /// body of `rock`: the shells no seed grows yet, for tests of the machinery that draws them.
    #[cfg(test)]
    fn authored(seed: u32, space: f32, deep: f32) -> Self {
        let mut bodies = vec![Body::authored(0, Kind::Home, HOME_CENTRE, Shape::Ball { r: HOME_RADIUS }, hash3(seed, 0, 0, 0))];
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
                        bodies.push(Body::authored(k, Kind::Twin, ctr, Shape::Cube { half: size }, hash3(s, side as i32, 0, 0)));
                        k += 1;
                    }
                }
                Kind::Hollow => {
                    let ctr = to_i64(c);
                    bodies.push(Body::authored(k, kind, ctr, Shape::Shell { outer: size, inner: size - 250_000 }, s));
                    k += 1;
                    bodies.push(Body::authored(k, Kind::Ember, ctr, Shape::Ball { r: 400_000 }, s ^ 0xE3BE));
                    k += 1;
                }
                _ => {
                    bodies.push(Body::authored(k, kind, to_i64(c), Shape::Ball { r: size }, s));
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
                        bodies.push(Body::authored(k, Kind::Moon, to_i64(c), Shape::Ball { r }, s));
                        k += 1;
                        break;
                    }
                }
            }
        }
        link(&mut bodies);
        let rock = super::emergent::minerals::role_suite(&material::Law::current(), "rock");
        Self::assemble(seed, bodies, vec![rock], space, deep)
    }

    /// The mineral suites, indexed by [`Traits::suite`].
    pub fn suites(&self) -> &[Suite] {
        &self.suites
    }

    /// The suite body `b` is made of.
    pub fn suite(&self, b: &Body) -> &Suite {
        &self.suites[usize::from(b.traits.suite)]
    }

    /// Every big body.
    pub fn bodies(&self) -> &[Body] {
        &self.bodies
    }

    /// The start world.
    pub fn home(&self) -> &Body {
        &self.bodies[0]
    }

    /// Every asteroid cluster.
    pub fn clusters(&self) -> &[Cluster] {
        &self.clusters
    }

    /// The body whose surface is nearest to `p` among those whose reach holds it.
    pub fn body_at(&self, p: DVec3) -> Option<&Body> {
        let cell = [p.x.floor() as i64, p.y.floor() as i64, p.z.floor() as i64];
        self.index
            .near(cell, cell)
            .iter()
            .map(|&i| &self.bodies[usize::from(i)])
            .filter(|b| (b.centre_f() - p).length() <= b.reach() * 1.5)
            .min_by(|a, b| self.altitude(a, p).abs().total_cmp(&self.altitude(b, p).abs()))
    }

    /// Whether `p` is in air: a nearest body exists, it has air, and `p` is at most its air top
    /// above its surface.
    pub fn in_air(&self, p: DVec3) -> bool {
        self.body_at(p).is_some_and(|body| body.traits.air_top.is_some_and(|top| self.altitude(body, p) <= f64::from(top)))
    }

    /// Open air inside a Hollow: within the inner surface and outside the core it holds.
    /// The shell's rock and the core's interior are not the cavity.
    pub fn hollow_cavity(&self, eye: DVec3) -> Option<(&Body, &Body)> {
        for hollow in self.bodies.iter().filter(|b| b.traits.form == Form::Shell) {
            let Shape::Shell { inner, .. } = hollow.shape else {
                continue;
            };
            let dist = (hollow.centre_f() - eye).length();
            if !(dist < inner as f64) {
                continue;
            }
            let Some(ember) = self.bodies.iter().find(|b| {
                b.traits.core_of == Some(hollow.id) && b.centre == hollow.centre && matches!(b.shape, Shape::Ball { .. })
            }) else {
                continue;
            };
            let Shape::Ball { r } = ember.shape else {
                continue;
            };
            if dist > r as f64 {
                return Some((hollow, ember));
            }
        }
        None
    }

    fn place_clusters(&mut self, space: f32) {
        if space <= 0.0 {
            return;
        }
        // Space is mostly empty: about one cell in 2,500 holds a cluster, so swarms lie ~2e8 apart
        // (the same order as the worlds) and finding one is an event.
        let p = 0.0004 * space.min(2.0);
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
                        0 | 1 => Spread::Belt,
                        2 => Spread::Stream,
                        _ => Spread::Swarm,
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
            Spread::Swarm => (1.0 - d.length_squared() / (r * r)).max(0.0),
            Spread::Belt => {
                let h = d.dot(c.axis);
                let rho = (d - c.axis * h).length();
                let (ring, width, thick) = (r * 0.75, r * 0.25, r * 0.06);
                let (a, b) = ((rho - ring) / width, h / thick);
                (1.0 - a * a - b * b).max(0.0)
            }
            Spread::Stream => {
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
            Spread::Swarm => unit_ball * r * r * r,
            // Torus-like: the paraboloid cross-section (π/2·w·t) swept around the ring.
            Spread::Belt => std::f64::consts::PI / 2.0 * (r * 0.25) * (r * 0.06) * 2.0 * std::f64::consts::PI * r * 0.75,
            Spread::Stream => unit_ball * r * (r * 0.15) * (r * 0.15),
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
        self.visit_rocks(lo, hi, 0..CLASSES.len(), |r| out.push(*r));
    }

    /// Rocks of size class `class` (0 = pebbles, 3 = the largest) whose sub-cell overlaps the
    /// inclusive cell box `[lo, hi]`. Empty cluster cells are skipped, so open space is a few
    /// cell hashes; a hit scans only that class's sub-cells (O(box / edge³)).
    /// The largest class-3 rock of any cluster (class 2 when none has one), for tests and benches
    /// that need a great rock without pinning coordinates to one catalog.
    #[cfg(test)]
    pub(crate) fn great_rock(&self) -> Option<Rock> {
        let mut best: Option<Rock> = None;
        for class in [3usize, 2] {
            for c in &self.clusters {
                let reach = c.radius + CLASSES[class].0 as f64;
                let lo = [c.centre.x, c.centre.y, c.centre.z].map(|v| (v - reach).floor() as i64);
                let hi = [c.centre.x, c.centre.y, c.centre.z].map(|v| (v + reach).ceil() as i64);
                self.for_class_rocks(class, lo, hi, |r| {
                    if best.is_none_or(|b| r.r > b.r) {
                        best = Some(*r);
                    }
                });
                if class == 2 && best.is_some() {
                    break;
                }
            }
            if best.is_some() {
                break;
            }
        }
        best
    }

    pub fn for_class_rocks(&self, class: usize, lo: [i64; 3], hi: [i64; 3], f: impl FnMut(&Rock)) {
        if class < CLASSES.len() {
            self.visit_rocks(lo, hi, class..class + 1, f);
        }
    }

    /// Rocks of `classes` whose sub-cell overlaps `[lo, hi]`. Cluster cells gate the scan:
    /// a cell with no cluster is one hash, and a cluster that cannot reach the box is skipped.
    fn visit_rocks(&self, lo: [i64; 3], hi: [i64; 3], classes: std::ops::Range<usize>, mut f: impl FnMut(&Rock)) {
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
                    for k in classes.clone() {
                        let edge = CLASSES[k].0;
                        let s = |v: i64| v.div_euclid(edge);
                        for sx in s(lo[0])..=s(hi[0]) {
                            for sy in s(lo[1])..=s(hi[1]) {
                                for sz in s(lo[2])..=s(hi[2]) {
                                    if let Some(r) = self.rock(c, k, [sx, sy, sz]) {
                                        f(&r);
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
        if self.bodies_touching(lo, hi).next().is_some() {
            return true;
        }
        let mut rocks = Vec::new();
        self.rocks_touching(lo, hi, &mut rocks);
        !rocks.is_empty()
    }

    /// The bodies whose matter may reach into the cell box `[lo, hi]`.
    pub fn bodies_touching(&self, lo: [i64; 3], hi: [i64; 3]) -> impl Iterator<Item = &Body> {
        self.index.near(lo, hi).iter().map(|&i| &self.bodies[usize::from(i)]).filter(move |b| b.touches(lo, hi))
    }
}

/// Body ids by super-cell of the dense grid ([`grid_slot`]): each cell lists, in id order, every
/// body whose 1.5-reach box overlaps it (the widest test any query makes: `body_at`).
#[derive(Default)]
struct BodyIndex {
    /// Start of each slot's ids in `ids`, plus one end.
    start: Vec<u32>,
    ids: Vec<u16>,
    /// Every id: the answer for a box that spans super-cells or leaves the grid.
    all: Vec<u16>,
}

impl BodyIndex {
    fn new(bodies: &[Body]) -> Self {
        let slots = (GRID_SIDE * GRID_SIDE * GRID_SIDE) as usize;
        let cells = |b: &Body| {
            let r = (b.reach() * 1.5).ceil() as i64 + 1;
            let lo = b.centre.map(|c| (c - r).div_euclid(SUPER).clamp(-GRID_HALF, GRID_HALF - 1));
            let hi = b.centre.map(|c| (c + r).div_euclid(SUPER).clamp(-GRID_HALF, GRID_HALF - 1));
            (lo[2]..=hi[2]).flat_map(move |z| (lo[1]..=hi[1]).flat_map(move |y| (lo[0]..=hi[0]).filter_map(move |x| grid_slot([x, y, z]))))
        };
        let mut count = vec![0u32; slots + 1];
        for b in bodies {
            for slot in cells(b) {
                count[slot + 1] += 1;
            }
        }
        for k in 0..slots {
            count[k + 1] += count[k];
        }
        let mut fill = count.clone();
        let mut ids = vec![0u16; count[slots] as usize];
        for b in bodies {
            for slot in cells(b) {
                ids[fill[slot] as usize] = b.id;
                fill[slot] += 1;
            }
        }
        Self { start: count, ids, all: bodies.iter().map(|b| b.id).collect() }
    }

    /// The bodies that may reach the inclusive cell box `[lo, hi]`, in id order.
    fn near(&self, lo: [i64; 3], hi: [i64; 3]) -> &[u16] {
        let key = lo.map(|v| v.div_euclid(SUPER));
        if hi.map(|v| v.div_euclid(SUPER)) != key {
            return &self.all;
        }
        match grid_slot(key) {
            Some(slot) if !self.start.is_empty() => &self.ids[self.start[slot] as usize..self.start[slot + 1] as usize],
            _ => &self.all,
        }
    }
}

/// Test fixtures over the catalog.
#[cfg(test)]
pub(crate) mod fixture {
    use std::cell::Cell;

    use super::{Body, Cosmos, Form, Rank};

    thread_local! {
        pub(super) static AUTHORED: Cell<bool> = const { Cell::new(false) };
    }

    /// Run `f` with every universe it grows on this thread replaced by the authored catalog: the
    /// Hollow and its Ember, which no seed grows before shells are emergent.
    pub(crate) fn authored<T>(f: impl FnOnce() -> T) -> T {
        AUTHORED.with(|a| a.set(true));
        let out = f();
        AUTHORED.with(|a| a.set(false));
        out
    }

    /// A body a test needs, by what it is rather than by a catalog entry.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum Want {
        /// One half of a contact binary of cubes: the canyon between them.
        Pair,
        /// A cube that sags into a storage box.
        Warped,
        /// A cube that keeps its physical grid.
        Cube,
        /// A round body the living (Verdant) painter grows.
        Living,
        /// A round body the glowing (Ember) painter burns.
        Glowing,
        /// A round body painted as a cratered moon.
        Moonlike,
        /// A body held by a heavier one.
        Satellite,
    }

    impl Want {
        pub(crate) const ALL: [Want; 7] =
            [Want::Pair, Want::Warped, Want::Cube, Want::Living, Want::Glowing, Want::Moonlike, Want::Satellite];

        pub(crate) fn matches(self, b: &Body) -> bool {
            let t = &b.traits;
            let round = t.form == Form::Round && t.rank != Rank::Start;
            match self {
                Want::Pair => t.partner.is_some(),
                Want::Warped => t.form == Form::Warped,
                Want::Cube => t.form == Form::Cube,
                Want::Living => round && !t.glow && t.life > 0,
                Want::Glowing => round && t.glow,
                Want::Moonlike => round && !t.glow && t.life == 0,
                Want::Satellite => t.rank == Rank::Satellite,
            }
        }
    }

    /// Seeds whose universes hold what tests want, found by `census` (default knobs) and checked
    /// by `fixture_seeds_hold_what_they_promise`.
    pub(crate) const SEEDS: [(i64, &[Want]); 3] = [
        (240, &[Want::Pair, Want::Warped, Want::Glowing, Want::Moonlike, Want::Satellite]),
        (139, &[Want::Warped, Want::Living, Want::Glowing, Want::Moonlike, Want::Satellite]),
        (157, &[Want::Pair, Want::Warped, Want::Living, Want::Glowing, Want::Moonlike, Want::Satellite]),
    ];

    /// The first pinned seed whose universe holds a body of every kind in `wants`.
    pub(crate) fn seed(wants: &[Want]) -> i64 {
        SEEDS.iter().find(|(_, has)| wants.iter().all(|w| has.contains(w))).map(|(s, _)| *s).expect("a pinned seed holds it")
    }

    /// The first body of `cosmos` that is `want`.
    pub(crate) fn first(cosmos: &Cosmos, want: Want) -> Option<Body> {
        cosmos.bodies().iter().copied().find(|b| want.matches(b))
    }
}

/// A body's relaxed shape as the catalog knows it.
#[derive(Clone, Debug)]
struct Relaxed {
    datum: std::sync::Arc<crate::space::datum::DatumField>,
    layer: std::sync::Arc<crate::gravity::relief::Relief>,
}

impl Cosmos {
    /// Settle the start world on its relaxed shape: its centre moves so the surface at the top of
    /// its +Y face (datum offset `top_offset` above [`HOME_RADIUS`]) stays at `y = 0`, where spawn,
    /// the bench and the travel targets expect the ground.
    pub fn settle_home(&mut self, top_offset: i64) {
        let home = &mut self.bodies[0];
        let Shape::Ball { r } = home.shape else { return };
        home.centre = [0, -r - top_offset, 0];
        self.index = BodyIndex::new(&self.bodies);
    }

    /// Give body `id` its relaxed datum and that relief as a gravity source (generation, before
    /// the cosmos is shared).
    pub fn set_relief(&mut self, id: u16, datum: std::sync::Arc<crate::space::datum::DatumField>, layer: std::sync::Arc<crate::gravity::relief::Relief>) {
        let slot = usize::from(id);
        if self.reliefs.len() <= slot {
            self.reliefs.resize(slot + 1, None);
        }
        self.reliefs[slot] = Some(Relaxed { datum, layer });
    }

    fn relief_of(&self, b: &Body) -> Option<&Relaxed> {
        self.reliefs.get(usize::from(b.id)).and_then(Option::as_ref)
    }

    /// How far body `b`'s relaxed surface sits above its datum sphere toward `p` (0 for a body
    /// without a relaxed datum).
    pub fn surface_offset(&self, b: &Body, p: DVec3) -> f64 {
        match self.relief_of(b) {
            Some(r) => r.datum.at((p - b.centre_f()).normalize_or_zero()),
            None => 0.0,
        }
    }

    /// Relaxed datum offset at a face centre. Zero without a relaxed datum.
    pub fn face_offset(&self, b: &Body) -> f64 {
        self.surface_offset(b, b.centre_f() + DVec3::Y)
    }

    /// Relaxed datum offset toward a corner. Zero without a relaxed datum.
    pub fn corner_offset(&self, b: &Body) -> f64 {
        self.surface_offset(b, b.centre_f() + DVec3::ONE)
    }

    /// Altitude of `p` above body `b`'s relaxed surface: above its datum for a relaxed round
    /// body, above its reference cube at the point the warp maps onto `p` for a warped cube, else
    /// [`Body::altitude`].
    pub fn altitude(&self, b: &Body, p: DVec3) -> f64 {
        if let Some((_, warp)) = self.warps.iter().find(|(id, _)| *id == b.id) {
            if let Some(x) = warp.invert(p) {
                return b.altitude(x);
            }
        }
        b.altitude(p) - self.surface_offset(b, p)
    }

    /// Give a warped cube its displacement field (generation, before the cosmos is shared).
    pub fn set_warp(&mut self, id: u16, warp: std::sync::Arc<crate::space::warp::Warp>) {
        self.warps.retain(|(i, _)| *i != id);
        self.warps.push((id, warp));
    }

    /// Lowest and highest relaxed datum offsets of body `b` (zero without a relaxed datum).
    pub fn relief_range(&self, b: &Body) -> (f64, f64) {
        self.relief_of(b).map_or((0.0, 0.0), |r| r.datum.range())
    }

    /// Give a warped cube its polyhedron (generation, before the cosmos is shared). The box
    /// primitive is no longer visited for that body.
    pub fn set_shape(&mut self, id: u16, shape: std::sync::Arc<crate::gravity::Polyhedron>) {
        self.shapes.retain(|(i, _)| *i != id);
        self.shapes.push((id, shape));
    }

    fn shape_of(&self, id: u16) -> Option<&crate::gravity::Polyhedron> {
        self.shapes.iter().find(|(i, _)| *i == id).map(|(_, s)| s.as_ref())
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
            if let Some(shape) = self.shape_of(b.id) {
                // The matter as it is. The undeformed cube is only the carrier of the higher
                // multipoles once the query is many radii out (see `Polyhedron::field_over_cube`).
                let (lo, hi) = match b.shape {
                    Shape::Cube { half } => {
                        let h = DVec3::splat(half as f64);
                        (b.centre_f() - h, b.centre_f() + h)
                    }
                    _ => (b.centre_f(), b.centre_f()),
                };
                let (accel, potential) = shape.field_over_cube(lo, hi, b.density, centre);
                v.analytic(accel, potential);
            } else {
                for p in b.primitives() {
                    v.primitive(&p);
                }
            }
            // Cube porosity shells, the Heart and mantle bubbles are the twin cubes' interior.
            // Home is a ball: those voids live in the face painter's virtual cube, and only the
            // part that overlaps band 0 is carved. The oracle stays the ball plus the relief error.
            if let Shape::Cube { half } = b.shape {
                super::deep::apply(b.centre, half, b.seed, b.density, self.deep, v);
            }
            if let Some(r) = self.relief_of(b) {
                let (accel, potential) = r.layer.field(centre);
                v.analytic(accel / G, potential / G);
            }
            if self.altitude(b, centre).abs() < RELIEF as f64 * 4.0 {
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
        assert!(a.bodies.len() > 3, "the start world has company: {}", a.bodies.len());
        assert_eq!((a.home().traits.rank, a.home().traits.form, a.home().kind), (Rank::Start, Form::Round, Kind::Home));
        for (i, b) in a.bodies.iter().enumerate() {
            assert_eq!(usize::from(b.id), i, "ids are catalog indices");
            assert_eq!(b.kind, b.traits.legacy_kind(b.id));
            assert!(b.centre.iter().all(|&c| c.abs() < 980_000_000), "{b:?} inside the border");
            assert!(b.centre.iter().all(|&c| c % 16 == 0), "{b:?} on the chunk grid");
            for o in &a.bodies[i + 1..] {
                let gap = (b.centre_f() - o.centre_f()).length();
                assert!(gap > b.reach().min(o.reach()), "{} and {} overlap", b.id, o.id);
            }
        }
        assert!(!a.clusters.is_empty());
    }

    #[test]
    fn air_stops_at_air_top_and_open_space_is_vacuum() {
        let cosmos = Cosmos::new(1, 1.0);
        let home = cosmos.home().centre_f();
        let Shape::Ball { r } = cosmos.home().shape else { panic!("the start world is round") };
        let top = r as f64;
        let air = f64::from(cosmos.home().traits.air_top.expect("the start world has air"));
        assert!(cosmos.in_air(home + DVec3::new(0.0, top + 80.0, 0.0)));
        assert!(!cosmos.in_air(home + DVec3::new(0.0, top + air + 1.0, 0.0)));
        let empty = DVec3::new(0.0, 0.0, 9.95e8);
        assert!(cosmos.body_at(empty).is_none());
        assert!(!cosmos.in_air(empty));
    }

    #[test]
    fn a_cube_polyhedron_replaces_the_box_and_adds_no_sag_error() {
        use crate::gravity::{Field, Polyhedron, Primitive, Shape as MassShape};
        use std::sync::Arc;
        let seed = fixture::seed(&[fixture::Want::Warped]) as u64;
        let base = Cosmos::new(seed, 1.0);
        let twin = fixture::first(&base, fixture::Want::Warped).expect("a cube");
        let Shape::Cube { half } = twin.shape else { panic!("a cube") };
        let c = twin.centre_f();
        let h = DVec3::splat(half as f64);
        let p = c + DVec3::new(0.0, half as f64 + 1_000.0, 0.0);
        let mut same = Cosmos::new(seed, 1.0);
        same.set_shape(twin.id, Arc::new(Polyhedron::from_box(c - h, c + h, twin.density)));
        let mut none = Cosmos::new(seed, 1.0);
        none.set_shape(twin.id, Arc::new(Polyhedron::from_box(c - h, c + h, 0.0)));
        let a = Field::new(Arc::new(base)).sample(p);
        let b = Field::new(Arc::new(same)).sample(p);
        let z = Field::new(Arc::new(none)).sample(p);
        assert!((a.accel - b.accel).length() <= 1e-8 * a.accel.length(), "{} vs {}", a.accel, b.accel);
        assert!((a.error - b.error).abs() <= 1e-9, "error {} vs {}", a.error, b.error);
        let box_a = Primitive::new(MassShape::Box { lo: c - h, hi: c + h }, twin.density).field(p).0 * G;
        assert!((a.accel - z.accel - box_a).length() <= 1e-8 * box_a.length(), "removed box {} residual {}", box_a, a.accel - z.accel);
    }

    #[test]
    fn home_spawn_pull_is_the_designed_gravity() {
        let cosmos = Arc::new(Cosmos::new(7, 1.0));
        let field = Field::new(cosmos.clone());
        let home = cosmos.home();
        let Shape::Ball { r } = home.shape else { panic!("home is a ball") };
        let at = DVec3::new(0.0, 2.0, 0.0);
        let s = field.sample(at);
        let solid = crate::gravity::Primitive::new(crate::gravity::Shape::Ball { c: home.centre_f(), r: r as f64 }, home.density);
        let want = solid.field(at).0 * crate::gravity::G;
        let to_centre = home.centre_f() - at;
        let dot = s.accel.dot(to_centre) / (s.accel.length() * to_centre.length());
        println!(
            "home spawn pull {:.4} blocks/s², solid ball {:.4}, dot {dot:.6}",
            s.accel.length(),
            want.length()
        );
        assert!(dot > 0.999, "pull points at the centre: {dot}");
        assert!(
            (s.accel.length() - want.length()).abs() < 0.05 * want.length(),
            "{} vs {}",
            s.accel.length(),
            want.length()
        );
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

    #[test]
    fn for_class_rocks_is_the_class_slice_of_the_box() {
        let cosmos = Cosmos::new(42, 1.0);
        let c = cosmos.clusters[0];
        let ctr = [c.centre.x as i64, c.centre.y as i64, c.centre.z as i64];
        let lo = [ctr[0] - 512, ctr[1] - 512, ctr[2] - 512];
        let hi = [ctr[0] + 512, ctr[1] + 512, ctr[2] + 512];
        let mut all = Vec::new();
        cosmos.rocks_touching(lo, hi, &mut all);
        let mut split = Vec::new();
        for k in 0..CLASSES.len() {
            cosmos.for_class_rocks(k, lo, hi, |r| split.push(*r));
        }
        assert_eq!(split, all);
        let mut none = 0;
        cosmos.for_class_rocks(9, lo, hi, |_| none += 1);
        assert_eq!(none, 0);
    }

    /// `cargo test --release --lib census -- --ignored --nocapture`: which seeds hold which bodies
    /// (`CENSUS_SEEDS`, default 400), for [`fixture::SEEDS`].
    #[test]
    #[ignore]
    fn census() {
        use fixture::Want;
        let n: i64 = std::env::var("CENSUS_SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(400);
        for seed in 0..n {
            let cosmos = Cosmos::grow(seed as u64, &super::super::TerrainCfg::default());
            let has: Vec<Want> = Want::ALL.into_iter().filter(|w| cosmos.bodies().iter().any(|b| w.matches(b))).collect();
            println!("{seed}: {} bodies {has:?}", cosmos.bodies().len());
        }
    }

    #[test]
    fn fixture_seeds_hold_what_they_promise() {
        for (seed, wants) in fixture::SEEDS {
            let cosmos = Cosmos::grow(seed as u64, &super::super::TerrainCfg::default());
            for &w in wants {
                assert!(fixture::first(&cosmos, w).is_some(), "seed {seed} lacks {w:?}");
            }
        }
    }

    /// `cargo test --lib cosmos_report -- --ignored --nocapture`: the catalog of a seed.
    #[test]
    #[ignore]
    fn cosmos_report() {
        let seed = std::env::var("WATT_COSMOS_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(42);
        let cosmos = Cosmos::new(seed, 1.0);
        for b in cosmos.bodies() {
            let d = (b.centre_f() - cosmos.home().centre_f()).length();
            let t = &b.traits;
            println!(
                "{:>2} {:<12} {:?}/{:?} {:?} ρ {} g {:.1} T {} air {:?} life {} glow {} ({}) — {:.3e} blocks from home",
                b.id, b.name(), t.rank, t.form, b.shape, b.density, t.gravity, t.temp, t.air_top, t.life, t.glow, t.class(), d
            );
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
        let h = match cosmos.home().shape {
            Shape::Ball { r } => r as f64,
            Shape::Cube { half } => half as f64,
            Shape::Shell { outer, .. } => outer as f64,
        };
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
        for a in cosmos.bodies() {
            let Some(b) = a.traits.partner.map(|p| cosmos.bodies()[usize::from(p)]).filter(|b| b.id > a.id) else { continue };
            let (Shape::Cube { half: ha }, Shape::Cube { half: hb }) = (a.shape, b.shape) else { continue };
            let d = (b.centre_f() - a.centre_f()).normalize();
            let gap = (b.centre_f() - a.centre_f()).length() - (ha + hb) as f64;
            stops.push((format!("{}'s canyon midpoint", a.name()), a.centre_f() + d * (ha as f64 + gap * 0.5)));
            stops.push((format!("{}'s facing surface", a.name()), a.centre_f() + d * (ha as f64 + 70.0)));
        }
        for b in cosmos.bodies().iter().filter(|b| b.id != 0 && (b.centre_f() - home).length() < 4.0e8) {
            let top = match b.shape {
                Shape::Ball { r } => r,
                Shape::Cube { half } => half,
                Shape::Shell { outer, .. } => outer,
            };
            stops.push((format!("{} ({}, {top})", b.name(), b.traits.class()), b.centre_f() + DVec3::new(0.0, top as f64 + 70.0, 0.0)));
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
        let moon = cosmos.bodies().iter().find(|b| b.id != 0 && b.traits.form == Form::Round).unwrap();
        let q = moon.centre_f() + DVec3::new(0.0, 2.0e6, 0.0);
        let t = std::time::Instant::now();
        for _ in 0..2_000 {
            acc += field.sample(q).accel.y;
        }
        println!("{:.2} µs per sample above a moon [{acc:.3}]", t.elapsed().as_secs_f64() / 2_000.0 * 1e6);
    }

}
