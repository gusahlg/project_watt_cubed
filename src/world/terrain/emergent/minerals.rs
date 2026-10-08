//! Stage 4: a body's minerals, found by the law. A radial column of reservoirs is drawn around the
//! body's composition and differentiates under the law's own contact process (even pairs, then odd
//! pairs, until every contact rests). What survives repair (every pair of occurrences holds) and
//! the rest check (dormant against its neighbours and against every universal material) is the
//! body's suite, densest first: core to crust. A degenerate result falls back to a suite of
//! palette roles.

use std::sync::OnceLock;

use field::hash32_3;
use material::{observe, visual, Block, Configuration, Contact, Element, Law, CAPACITY};

use super::Params;
use crate::mechanics::material::{Params as Matter, YIELD_FLOOR};
use crate::world::terrain::palette::{self, cohesive, dormant, Need};

/// Contact operations one pair may take per sweep.
const MAX_OPS: u32 = 4 * CAPACITY as u32;
/// Yield per unit of cohesion above the bottom of the span: `10^(4 decades / 384)`.
const YIELD_STEP: f64 = 1.024_275_221_381_592_2;
/// Cohesion (Q8) where yield leaves its floor, and the span of its four decades.
const COHESION_FLOOR: i32 = 384;
const COHESION_SPAN: i32 = 384;
/// Share of the crust that must glow for the body to glow.
const GLOW_CRUST: f64 = 0.3;
/// The bulk candidates of `choose_bulk` and the rest of the palette's rock-like roles, per class
/// (refractory, rock, carbon, volatile): what a fallback suite is built from.
const CLASS_ROLES: [&[&str]; 4] = [
    &["deeprock", "abyss", "basalt", "obsidian", "slate"],
    &["rock", "rock1", "rock2", "rock3", "gravel", "limestone", "marble"],
    &["ochre", "sandstone", "sandstone1", "sandstone2", "sandstone3", "clay", "rust"],
    &["ice", "snow", "salt", "frost", "regolith"],
];
/// Universal materials besides the underground set that generated matter may touch: surface
/// dressing and timber.
const SURFACE: &[&str] = &["grass", "meadow", "snow", "ice", "timber"];

/// One layer of a suite.
#[derive(Clone, Debug, PartialEq)]
pub struct Mineral {
    pub config: Configuration,
    /// Occurrences per block: the layer's density.
    pub amount: u8,
    pub cohesion: i16,
    pub hardness: u8,
    pub friction: u8,
    pub emission: u8,
    pub transparency: u8,
    pub rgb: [u8; 3],
    /// Did not rest: in a suite's layers, a palette role standing in; in its own minerals, the
    /// mineral the rest check rejected.
    pub replaced: bool,
}

/// What a body is made of, densest first.
#[derive(Clone, Debug, PartialEq)]
pub struct Suite {
    pub minerals: Vec<Mineral>,
    /// Built from palette roles because differentiation gave too little.
    pub fallback: bool,
    /// Volume mean amount of the mantle layers.
    pub density: f64,
    /// Harmonic yield of the mantle layers, amount/(block·s²).
    pub yield_stress: f64,
    pub glow: bool,
    /// Distinct minerals after repair, before any fallback.
    pub distinct: u8,
    /// Layers the rest check replaced.
    pub replaced: u8,
    /// Occurrences repair dropped.
    pub dropped: u16,
    /// Differentiation sweeps run.
    pub sweeps: u8,
    /// Largest pairwise colour distance (sum of channel differences, 0..765).
    pub colour_spread: u16,
    /// Densest amount minus lightest.
    pub amount_spread: u8,
    /// Shares of refractory, rock, carbon and volatile layers.
    pub classes: [f64; 4],
    /// (mineral, reagent) pairs where a palette reagent can sit in the mineral as a vein.
    pub hosts: u8,
    /// Why the suite fell back: 1 fewer than three distinct minerals, 2 too little colour spread,
    /// 4 the rest check replaced more than half.
    pub causes: u8,
    /// The law's own minerals after repair, densest first; `replaced` marks the ones the rest check
    /// rejected.
    pub own: Vec<Mineral>,
}

/// Palette materials the suite is checked and repaired against, built once per process for the
/// current law.
struct Reference {
    palette: Vec<palette::Entry>,
    /// Universal materials: anything generated matter may touch.
    universal: Vec<Block>,
    /// Plain roles a failed mineral may be replaced by: (block, colour, configuration).
    plain: Vec<(Block, [u8; 3], Configuration)>,
    reagents: Vec<Block>,
}

fn reference(law: &Law) -> &'static Reference {
    static REF: OnceLock<Reference> = OnceLock::new();
    REF.get_or_init(|| {
        let p = palette::of(law);
        let universal = palette::UNDERGROUND.iter().chain(SURFACE).map(|l| role_block(&p, l)).collect();
        let mut plain = Vec::new();
        let mut reagents = Vec::new();
        for (role, e) in palette::ROLES.iter().zip(&p) {
            let b = Block::of(&e.config);
            match role.need {
                Need::Plain => plain.push((b.clone(), visual(law, &b).rgb, e.config.clone())),
                Need::Reagent(_) => reagents.push(b),
                _ => {}
            }
        }
        Reference { palette: p, universal, plain, reagents }
    })
}

fn role_block(p: &[palette::Entry], label: &str) -> Block {
    Block::of(&p.iter().find(|e| e.label == label).expect("a palette role").config)
}

/// How many universal materials a mineral would react with, out of how many (a lab diagnostic of
/// the rest check).
pub fn restless_against(law: &Law, m: &Mineral) -> (usize, usize) {
    let r = reference(law);
    let b = Block::of(&m.config);
    (r.universal.iter().filter(|u| !dormant(&b, u)).count(), r.universal.len())
}

/// Yield stress of matter of cohesion `cohesion` (Q8): the prototype response's four decades,
/// as integer powers of [`YIELD_STEP`] (no libm).
pub fn yield_of(cohesion: i32) -> f64 {
    let mut k = (cohesion - COHESION_FLOOR).clamp(0, COHESION_SPAN) as u32;
    let (mut y, mut base) = (YIELD_FLOOR, YIELD_STEP);
    while k > 0 {
        if k & 1 == 1 {
            y *= base;
        }
        base *= base;
        k >>= 1;
    }
    y
}

fn mineral(law: &Law, block: &Block, replaced: bool) -> Mineral {
    let o = observe(law, block);
    Mineral {
        config: block.configuration(),
        amount: block.len() as u8,
        cohesion: o.cohesion,
        hardness: o.hardness,
        friction: o.friction,
        emission: o.emission,
        transparency: o.transparency,
        rgb: visual(law, block).rgb,
        replaced,
    }
}

/// One occurrence within ring distance `spread` of `centre`, by rejection.
fn draw(centre: Element, spread: u32, seed: u32, r: i32, k: i32) -> Element {
    let span = 2 * spread + 1;
    for t in 0..64 {
        let h = hash32_3(seed, r, k, t, 0x0CC0).to_le_bytes();
        let off: [i32; 4] = std::array::from_fn(|a| (h[a] as u32 * span / 256) as i32 - spread as i32);
        if off.iter().map(|v| v.unsigned_abs()).sum::<u32>() <= spread {
            return Element::new(std::array::from_fn(|a| centre.0[a].wrapping_add(off[a] as u8)));
        }
    }
    centre
}

/// The reservoirs of the column, before differentiation.
fn column(centre: Element, seed: u32, p: &Params) -> Vec<Block> {
    (0..p.reservoirs as i32)
        .map(|r| {
            let span = p.occ_max.max(p.occ_min) - p.occ_min + 1;
            let n = p.occ_min + hash32_3(seed, r, -1, 0, 0x5EED) % span;
            let elems: Vec<Element> = (0..n.min(CAPACITY as u32) as i32).map(|k| draw(centre, p.spread, seed, r, k)).collect();
            Block::new(&elems).expect("within capacity")
        })
        .collect()
}

/// Run the law between vertically adjacent reservoirs, even pairs then odd pairs, until every
/// contact rests or the sweeps run out. Returns the sweeps used.
fn differentiate(blocks: &mut [Block], sweeps: u32) -> u32 {
    for sweep in 0..sweeps {
        let mut moved = false;
        for parity in 0..2 {
            for r in (parity..blocks.len().saturating_sub(1)).step_by(2) {
                let mut c = Contact::new(&blocks[r], &blocks[r + 1]);
                let mut ops = 0;
                while ops < MAX_OPS && c.step().is_some() {
                    ops += 1;
                }
                if ops > 0 {
                    moved = true;
                    let [mut a, mut b] = c.blocks();
                    a.canonicalize();
                    b.canonicalize();
                    blocks[r] = a;
                    blocks[r + 1] = b;
                }
            }
        }
        if !moved {
            return sweep;
        }
    }
    sweeps
}

/// Drop the least-held occurrence until every pair holds. Returns how many were dropped.
fn repair(block: &mut Block) -> u16 {
    let mut dropped = 0;
    while !cohesive(block.elements()) {
        let (k, _) = block
            .holding()
            .iter()
            .zip(block.elements())
            .enumerate()
            .min_by_key(|&(_, (&h, &e))| (h, e))
            .expect("a block with a repelling pair is not empty");
        let rest: Vec<Element> = block.elements().iter().enumerate().filter(|&(i, _)| i != k).map(|(_, &e)| e).collect();
        *block = Block::new(&rest).expect("smaller");
        dropped += 1;
    }
    dropped
}

/// The nearest plain palette role by colour.
fn nearest_plain(r: &Reference, rgb: [u8; 3]) -> &Block {
    let dist = |c: [u8; 3]| (0..3).map(|k| (c[k] as i32 - rgb[k] as i32).unsigned_abs()).sum::<u32>();
    &r.plain.iter().min_by_key(|(_, c, cfg)| (dist(*c), cfg.clone())).expect("plain roles").0
}

fn colour_spread(m: &[Mineral]) -> u16 {
    let mut best = 0u32;
    for i in 0..m.len() {
        for j in i + 1..m.len() {
            best = best.max((0..3).map(|k| (m[i].rgb[k] as i32 - m[j].rgb[k] as i32).unsigned_abs()).sum());
        }
    }
    best as u16
}

/// The class of a layer (refractory, rock, carbon, volatile), read from what the law observes:
/// clear or sparse matter is volatile, dense matter refractory, weakly held matter carbon-like.
/// A modelling choice, labelled as such.
fn class(m: &Mineral) -> usize {
    if m.transparency > 0 || m.amount <= 3 {
        3
    } else if m.amount >= 7 {
        0
    } else if (m.cohesion as i32) < COHESION_FLOOR {
        2
    } else {
        1
    }
}

/// Fill the derived fields of a suite from its layers.
fn finish(r: &Reference, minerals: Vec<Mineral>, fallback: bool) -> Suite {
    let n = minerals.len();
    // Mantle: every layer between the core and the crust (all of them for one or two layers).
    let mantle = if n >= 3 { &minerals[1..n - 1] } else { &minerals[..] };
    let density = mantle.iter().map(|m| m.amount as f64).sum::<f64>() / mantle.len() as f64;
    let share = 1.0 / mantle.len() as f64;
    let parts: Vec<(f64, Matter)> =
        mantle.iter().map(|m| (share, Matter::from_yield(m.amount as f64, yield_of(m.cohesion as i32)))).collect();
    let yield_stress = Matter::mix(&parts).yield_stress;
    let crust = &minerals[n - n.div_ceil(3)..];
    let glowing = crust.iter().filter(|m| m.emission > 0).count();
    let mut classes = [0.0; 4];
    for m in &minerals {
        classes[class(m)] += 1.0 / n as f64;
    }
    let blocks: Vec<Block> = minerals.iter().map(|m| Block::of(&m.config)).collect();
    let hosts = blocks.iter().map(|b| r.reagents.iter().filter(|g| dormant(b, g)).count()).sum::<usize>();
    let amounts = minerals.iter().map(|m| m.amount);
    Suite {
        density,
        yield_stress,
        glow: glowing as f64 >= GLOW_CRUST * crust.len() as f64,
        distinct: 0,
        replaced: minerals.iter().filter(|m| m.replaced).count() as u8,
        dropped: 0,
        sweeps: 0,
        colour_spread: colour_spread(&minerals),
        amount_spread: amounts.clone().max().unwrap_or(0) - amounts.min().unwrap_or(0),
        classes,
        hosts: hosts.min(255) as u8,
        causes: 0,
        own: Vec::new(),
        minerals,
        fallback,
    }
}

/// A suite of palette roles: composition axes weight the four classes, a hash picks the roles.
pub fn palette_suite(law: &Law, comp: [i8; 4], seed: u32, layers: usize) -> Suite {
    let r = reference(law);
    let weights: [u32; 4] = std::array::from_fn(|k| (comp[k] as i32 + 97) as u32);
    let total: u32 = weights.iter().sum();
    let mut picked: Vec<Block> = Vec::new();
    for slot in 0..layers as i32 {
        let h = hash32_3(seed, slot, 0, 0, 0xFA11);
        let mut t = h % total;
        let class = weights.iter().position(|&w| if t < w { true } else { t -= w; false }).expect("in range");
        let roles = CLASS_ROLES[class];
        let label = roles[(h >> 16) as usize % roles.len()];
        let block = role_block(&r.palette, label);
        if !picked.iter().any(|b| b.elements() == block.elements()) {
            picked.push(block);
        }
    }
    let mut minerals: Vec<Mineral> = picked.iter().map(|b| mineral(law, b, false)).collect();
    minerals.sort_by(|a, b| b.amount.cmp(&a.amount).then_with(|| a.config.cmp(&b.config)));
    finish(r, minerals, true)
}

/// A suite of one palette role (the start world's last resort: today's `rock`).
pub fn role_suite(law: &Law, label: &str) -> Suite {
    let r = reference(law);
    finish(r, vec![mineral(law, &role_block(&r.palette, label), false)], true)
}

/// The suite of a body of composition `comp` (offsets from `base`).
pub fn suite(law: &Law, base: Element, comp: [i8; 4], seed: u32, p: &Params) -> Suite {
    let r = reference(law);
    let centre = Element::new(std::array::from_fn(|a| base.0[a].wrapping_add(comp[a] as u8)));
    let mut blocks = column(centre, seed, p);
    let sweeps = differentiate(&mut blocks, p.sweeps);
    let mut dropped = 0u16;
    for b in &mut blocks {
        dropped += repair(b);
    }
    blocks.retain(|b| !b.is_empty());
    let mut minerals: Vec<Mineral> = Vec::with_capacity(blocks.len());
    for b in &blocks {
        let m = mineral(law, b, false);
        if !minerals.iter().any(|x| x.config == m.config) {
            minerals.push(m);
        }
    }
    minerals.sort_by(|a, b| b.amount.cmp(&a.amount).then_with(|| a.config.cmp(&b.config)));
    let distinct = minerals.len();
    // Rest check: dormant against the layers beside it and against every universal material.
    // A restless layer is marked; the suite takes the nearest plain palette role in its place.
    let blocks: Vec<Block> = minerals.iter().map(|m| Block::of(&m.config)).collect();
    for i in 0..minerals.len() {
        let beside = |j: usize| j < blocks.len() && j != i && !dormant(&blocks[i], &blocks[j]);
        minerals[i].replaced =
            beside(i.wrapping_sub(1)) || beside(i + 1) || r.universal.iter().any(|u| !dormant(&blocks[i], u));
    }
    let mut layers: Vec<Mineral> = minerals
        .iter()
        .map(|m| if m.replaced { mineral(law, nearest_plain(r, m.rgb), true) } else { m.clone() })
        .collect();
    layers.sort_by(|a, b| b.amount.cmp(&a.amount).then_with(|| a.config.cmp(&b.config)));
    let replaced = minerals.iter().filter(|m| m.replaced).count();
    let causes = (distinct < 3) as u8
        | ((colour_spread(&layers) < p.colour_min) as u8) << 1
        | ((replaced * 2 > minerals.len()) as u8) << 2;
    let mut s = if causes != 0 { palette_suite(law, comp, seed, 6) } else { finish(r, layers, false) };
    (s.distinct, s.dropped, s.sweeps, s.causes) = (distinct as u8, dropped, sweeps as u8, causes);
    s.own = minerals;
    s
}
