//! Stage 3: accretion. A system's mass is split into parcels (a truncated power law of masses,
//! placed on its cells by cell mass) that merge whenever they lie within the law's reach of each
//! other, `d³ < A·(m_i + m_j)` (a home-mass body clears `R_IN`), round after round until no pair
//! qualifies. A merge keeps its history: the collision's heat and the impact's direction and
//! energy. A similar, strong and gentle pair binds as a contact binary instead of fusing.

use field::hash32_3;

use super::nebula::{Nebula, System, CELL};
use super::{cbrt, Params};
use crate::gravity::G;

/// Density sizes are measured with before any body has minerals.
pub const PROVISIONAL_DENSITY: f64 = 5.0;
/// Impacts kept per body (the most energetic).
pub const IMPACTS: usize = 16;
/// Bodies smaller than this half-size are debris (the largest rock class's radius).
pub const DEBRIS_HALF: f64 = 3_000.0;
/// Composition jitter of a parcel around its cell, per axis.
const JITTER: f64 = 8.0;

/// One collision a body remembers: direction (unit, body frame) and energy (G·m_i·m_j/(r_i + r_j)).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Impact {
    pub dir: [f64; 3],
    pub energy: f64,
}

/// A contact binary's two parts, side by side along `axis` (part 0 on the `−sign` side).
#[derive(Clone, Debug, PartialEq)]
pub struct Pair {
    pub masses: [f64; 2],
    pub comps: [[f64; 4]; 2],
    pub axis: usize,
    pub sign: f64,
}

/// A body during accretion.
#[derive(Clone, Debug, PartialEq)]
pub struct Proto {
    pub mass: f64,
    pub pos: [f64; 3],
    /// Mass-weighted composition offsets.
    pub comp: [f64; 4],
    /// Collision heat per unit mass.
    pub heat_in: f64,
    pub impacts: Vec<Impact>,
    pub pair: Option<Pair>,
}

/// What accretion leaves in one system.
pub struct Accreted {
    pub bodies: Vec<Proto>,
    /// Bodies below [`DEBRIS_HALF`]: position and mass.
    pub debris: Vec<([f64; 3], f64)>,
    pub rounds: u32,
    pub binaries: u32,
}

/// Half-size of a cube of `mass` at `density`.
pub fn half_of(mass: f64, density: f64) -> f64 {
    cbrt(mass / density) * 0.5
}

fn unit(h: u32) -> f64 {
    h as f64 / 4_294_967_296.0
}

/// The system's parcels: `p.parcels` of them, a truncated α=2 power law of masses scaled to the
/// system's mass (raw mass times `scale`), each on a cell picked by cell mass.
fn parcels(neb: &Nebula, sys: &System, scale: f64, seed: u32, p: &Params) -> Vec<Proto> {
    let mut cum = Vec::with_capacity(sys.cells.len());
    let mut acc = 0u64;
    for &(_, m) in &sys.cells {
        acc += m;
        cum.push(acc);
    }
    let (lo, hi) = (1.0, p.parcel_range);
    let mut out: Vec<Proto> = (0..p.parcels as i32)
        .map(|k| {
            let h = |salt: u32| hash32_3(seed, k, 0, 0, salt);
            let pick = ((h(1) as u128 * acc as u128) >> 32) as u64;
            let cell = sys.cells[cum.partition_point(|&v| v <= pick)].0;
            let c = Nebula::cell_centre(cell);
            let comp = neb.cells[cell as usize].comp;
            Proto {
                mass: lo / (1.0 - unit(h(5)) * (1.0 - lo / hi)),
                pos: std::array::from_fn(|a| c[a] + (unit(h(2 + a as u32)) - 0.5) * CELL),
                comp: std::array::from_fn(|a| comp[a] as f64 + (unit(h(6 + a as u32)) - 0.5) * 2.0 * JITTER),
                heat_in: 0.0,
                impacts: Vec::new(),
                pair: None,
            }
        })
        .collect();
    let total: f64 = out.iter().map(|b| b.mass).sum();
    for b in &mut out {
        b.mass *= sys.mass * scale / total;
    }
    out
}

fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|k| (a[k] - b[k]) * (a[k] - b[k])).sum()
}

/// Keep the most energetic impacts (ties by direction bits, so the order is total).
pub fn keep_impacts(list: &mut Vec<Impact>) {
    list.sort_by(|a, b| b.energy.total_cmp(&a.energy).then_with(|| a.dir.map(f64::to_bits).cmp(&b.dir.map(f64::to_bits))));
    list.truncate(IMPACTS);
}

/// Fuse `from` into `into`: masses add, composition and heat are mass-weighted, the collision
/// heats the result and is remembered as an impact in `into`'s frame. `into` keeps its place when
/// `keep_place`, else the pair moves to its centre of mass.
pub fn fuse(into: &mut Proto, from: &Proto, keep_place: bool) {
    let m = into.mass + from.mass;
    let r = half_of(into.mass, PROVISIONAL_DENSITY) + half_of(from.mass, PROVISIONAL_DENSITY);
    let energy = G * into.mass * from.mass / r;
    let d = std::array::from_fn::<f64, 3, _>(|a| from.pos[a] - into.pos[a]);
    let len = dist2(from.pos, into.pos).sqrt();
    let dir = if len > 0.0 { d.map(|v| v / len) } else { [0.0, 1.0, 0.0] };
    let w = from.mass / m;
    if !keep_place {
        into.pos = std::array::from_fn(|a| into.pos[a] + d[a] * w);
    }
    into.comp = std::array::from_fn(|a| into.comp[a] + (from.comp[a] - into.comp[a]) * w);
    into.heat_in = into.heat_in + (from.heat_in - into.heat_in) * w + energy / m;
    if let Some(pair) = &mut into.pair {
        // A binary takes the hit on its heavier part.
        let k = if pair.masses[0] >= pair.masses[1] { 0 } else { 1 };
        let wk = from.mass / (pair.masses[k] + from.mass);
        pair.comps[k] = std::array::from_fn(|a| pair.comps[k][a] + (from.comp[a] - pair.comps[k][a]) * wk);
        pair.masses[k] += from.mass;
    }
    into.mass = m;
    into.impacts.push(Impact { dir, energy });
    keep_impacts(&mut into.impacts);
}

/// Accrete one system. `binds(a, b, energy_per_mass)` says whether a qualifying pair of similar
/// masses binds as a contact binary instead of fusing.
pub fn accrete(
    neb: &Nebula,
    sys: &System,
    scale: f64,
    seed: u32,
    p: &Params,
    binds: &mut dyn FnMut(&Proto, &Proto, f64) -> bool,
) -> Accreted {
    let mut bodies = parcels(neb, sys, scale, seed, p);
    let mut alive = vec![true; bodies.len()];
    let (mut rounds, mut binaries) = (0, 0);
    loop {
        rounds += 1;
        // Sort and sweep along x: a qualifying pair lies within `cbrt(2·A·m)` of its heavier body
        // (by mass, then index), so each pair is found once, from that side.
        let mut order: Vec<usize> = (0..bodies.len()).filter(|&i| alive[i]).collect();
        order.sort_by(|&a, &b| bodies[a].pos[0].total_cmp(&bodies[b].pos[0]).then(a.cmp(&b)));
        let heavier = |a: usize, b: usize| (bodies[a].mass, a) > (bodies[b].mass, b);
        let mut pairs: Vec<(u64, usize, usize)> = Vec::new();
        for (k, &h) in order.iter().enumerate() {
            let reach = cbrt(p.merge_a * 2.0 * bodies[h].mass) * (1.0 + 1e-9);
            let x = bodies[h].pos[0];
            let near = order[k + 1..]
                .iter()
                .take_while(|&&j| bodies[j].pos[0] - x < reach)
                .chain(order[..k].iter().rev().take_while(|&&j| x - bodies[j].pos[0] < reach));
            for &j in near {
                if !heavier(h, j) {
                    continue;
                }
                let d2 = dist2(bodies[h].pos, bodies[j].pos);
                let limit = p.merge_a * (bodies[h].mass + bodies[j].mass);
                let d3 = d2 * d2.sqrt();
                if d3 < limit {
                    pairs.push(((d3 / limit).to_bits(), h.min(j), h.max(j)));
                }
            }
        }
        pairs.sort_unstable();
        let mut used = vec![false; bodies.len()];
        let mut merged = 0;
        for (_, i, j) in pairs {
            if used[i] || used[j] {
                continue;
            }
            used[i] = true;
            used[j] = true;
            merged += 1;
            let (a, b) = (bodies[i].clone(), bodies[j].clone());
            let (big, small) = if a.mass >= b.mass { (&a, &b) } else { (&b, &a) };
            let r = half_of(a.mass, PROVISIONAL_DENSITY) + half_of(b.mass, PROVISIONAL_DENSITY);
            let per_mass = G * a.mass * b.mass / (r * (a.mass + b.mass));
            let single = a.pair.is_none() && b.pair.is_none();
            let next = if single && small.mass >= p.q_bin * big.mass && binds(&a, &b, per_mass) {
                binaries += 1;
                let d: [f64; 3] = std::array::from_fn(|k| b.pos[k] - a.pos[k]);
                let mut axis = 0;
                for k in 1..3 {
                    if d[k].abs() > d[axis].abs() {
                        axis = k;
                    }
                }
                let sign = if d[axis] >= 0.0 { 1.0 } else { -1.0 };
                let mut joined = a.clone();
                joined.pair = Some(Pair { masses: [a.mass, b.mass], comps: [a.comp, b.comp], axis, sign });
                let w = b.mass / (a.mass + b.mass);
                joined.pos = std::array::from_fn(|k| a.pos[k] + d[k] * w);
                joined.comp = std::array::from_fn(|k| a.comp[k] + (b.comp[k] - a.comp[k]) * w);
                joined.heat_in = a.heat_in + (b.heat_in - a.heat_in) * w;
                joined.mass = a.mass + b.mass;
                joined
            } else {
                let mut into = big.clone();
                fuse(&mut into, small, false);
                into
            };
            // The smaller id survives.
            bodies[i] = next;
            alive[j] = false;
        }
        if merged == 0 {
            break;
        }
    }
    let mut out = Vec::new();
    let mut debris = Vec::new();
    for (b, keep) in bodies.into_iter().zip(alive) {
        if !keep {
            continue;
        }
        if half_of(b.mass, PROVISIONAL_DENSITY) < DEBRIS_HALF {
            debris.push((b.pos, b.mass));
        } else {
            out.push(b);
        }
    }
    Accreted { bodies: out, debris, rounds, binaries }
}
