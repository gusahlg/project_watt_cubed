//! The generator's materials, found in the law — never authored.
//!
//! Every terrain role (grass, banded sandstone, mine timber, a planet's glowing core, …) names a
//! *look and behaviour* it wants; this module searches the resource lattice for a configuration
//! that has it. Colours are the law's presentation of the elements; glow and clarity are the law's
//! probe readings. Two rules keep a generated world at rest until a player disturbs it:
//!
//! * **cohesion** — every pair of occurrences inside a common material fits non-negatively, so no
//!   occurrence ever wants to leave for empty space, and a disturbance cannot run as a front
//!   through a uniform mass (a depleted or contaminated block settles against its neighbours in
//!   one or two hops);
//! * **mutual quiescence** — every pair of common materials is a dormant contact in both
//!   orientations, so strata, ores and structures lie side by side without reacting.
//!
//! **Reagents** break the second rule on purpose: each is a counter-material that empties one
//! specific common material (selective transfer's destructive exception) while staying dormant
//! against every other. They are the natural tools of the world — veins in the mines.
//!
//! The palette is a pure function of the law (computed once per process for the current law), so
//! every peer finds the same materials.

use std::sync::OnceLock;

use material::{
    centroid_q8, colour_at, fit_raw, observe, Block, Configuration, Contact, Element, Law, QUANTUM,
};

use super::noise::hash2;

/// What a role needs from its configuration besides its colour.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Need {
    /// Opaque, dark (no glow). The common case.
    Plain,
    /// Emits at least this light level (0..15).
    Glow(u8),
    /// At least this transparency (0..255); `glow` additionally requires light.
    Clear { min: u8, glow: bool },
    /// A counter-material that empties the named role and is dormant against all others.
    Reagent(&'static str),
}

/// One terrain role.
#[derive(Clone, Copy, Debug)]
pub struct Role {
    /// Internal label (the registry annotation; never shown to players).
    pub label: &'static str,
    /// Target base colour.
    pub rgb: [u8; 3],
    /// What else it needs.
    pub need: Need,
}

const fn role(label: &'static str, rgb: [u8; 3], need: Need) -> Role {
    Role { label, rgb, need }
}

/// Every role, in palette (intern) order. Reagents come after their targets.
pub const ROLES: &[Role] = &[
    // Surface dressing.
    role("grass", [84, 142, 52], Need::Plain),
    role("meadow", [128, 160, 60], Need::Plain),
    role("soil", [112, 80, 54], Need::Plain),
    role("sand", [216, 192, 132], Need::Plain),
    role("redsand", [196, 112, 68], Need::Plain),
    role("snow", [236, 240, 246], Need::Plain),
    role("ice", [168, 206, 236], Need::Clear { min: 90, glow: false }),
    role("gravel", [128, 124, 118], Need::Plain),
    // Rock strata, banded on cliffs.
    role("rock", [124, 124, 130], Need::Plain),
    role("rock1", [150, 138, 122], Need::Plain),
    role("rock2", [100, 102, 116], Need::Plain),
    role("rock3", [168, 150, 138], Need::Plain),
    role("sandstone", [206, 140, 88], Need::Plain),
    role("sandstone1", [228, 184, 124], Need::Plain),
    role("sandstone2", [172, 92, 66], Need::Plain),
    role("sandstone3", [236, 214, 172], Need::Plain),
    role("deeprock", [62, 60, 72], Need::Plain),
    role("abyss", [38, 34, 48], Need::Plain),
    // Trees.
    role("timber", [108, 78, 50], Need::Plain),
    role("leaves", [56, 122, 44], Need::Plain),
    role("pine", [38, 92, 58], Need::Plain),
    role("blossom", [226, 150, 186], Need::Plain),
    role("autumn", [206, 112, 40], Need::Plain),
    // Mines.
    role("plank", [146, 108, 66], Need::Plain),
    role("rail", [88, 90, 102], Need::Plain),
    role("lamp", [255, 196, 118], Need::Glow(10)),
    role("rubble", [98, 94, 90], Need::Plain),
    role("bone", [232, 226, 204], Need::Plain),
    // Caves and the deep.
    role("glowcap", [80, 206, 170], Need::Glow(6)),
    role("crystal", [176, 124, 255], Need::Clear { min: 110, glow: false }),
    role("copper", [184, 108, 66], Need::Plain),
    role("azurite", [58, 88, 204], Need::Plain),
    role("gold", [226, 192, 58], Need::Plain),
    // Space.
    role("regolith", [148, 144, 140], Need::Plain),
    role("basalt", [50, 50, 60], Need::Plain),
    role("frost", [204, 226, 246], Need::Plain),
    role("moss", [70, 152, 92], Need::Plain),
    role("ochre", [204, 150, 78], Need::Plain),
    role("violet", [132, 88, 172], Need::Plain),
    role("magma", [255, 112, 36], Need::Glow(9)),
    role("core", [255, 228, 172], Need::Glow(14)),
    role("star", [250, 250, 255], Need::Glow(15)),
    // Reagents: the world's natural tools.
    role("etch_rock", [0, 0, 0], Need::Reagent("rock")),
    role("etch_rock1", [0, 0, 0], Need::Reagent("rock1")),
    role("etch_rock2", [0, 0, 0], Need::Reagent("rock2")),
    role("etch_deeprock", [0, 0, 0], Need::Reagent("deeprock")),
    role("etch_soil", [0, 0, 0], Need::Reagent("soil")),
    role("etch_sandstone", [0, 0, 0], Need::Reagent("sandstone")),
    // Flowers, fungi, strata and one extra glow for the v4 realms. Appended so every earlier plain
    // role stays ahead of them in the search; the glow is brighter than glowcap, so it is searched
    // with the other lamps.
    role("flower_red", [196, 40, 52], Need::Plain),
    role("flower_yellow", [236, 206, 58], Need::Plain),
    role("flower_blue", [72, 104, 214], Need::Plain),
    role("flower_white", [236, 236, 228], Need::Plain),
    role("cap_red", [176, 44, 40], Need::Plain),
    role("cap_brown", [150, 104, 68], Need::Plain),
    role("stem", [222, 214, 194], Need::Plain),
    role("ash", [74, 72, 70], Need::Plain),
    role("obsidian", [28, 22, 36], Need::Plain),
    role("salt", [240, 238, 230], Need::Plain),
    role("clay", [168, 106, 78], Need::Plain),
    role("limestone", [206, 200, 184], Need::Plain),
    role("marble", [228, 228, 232], Need::Plain),
    role("jade", [76, 168, 118], Need::Plain),
    role("rust", [138, 72, 42], Need::Plain),
    role("mud", [84, 66, 50], Need::Plain),
    role("lichen", [148, 168, 92], Need::Plain),
    role("darkwood", [62, 42, 30], Need::Plain),
    role("bark", [96, 74, 52], Need::Plain),
    role("amber", [218, 140, 40], Need::Plain),
    role("slate", [72, 78, 88], Need::Plain),
    // The search lands this role on a dark green configuration (seed-independent: the law and the
    // roles before it decide), so painters use basalt/obsidian instead. Kept: removing a role
    // would move every pick after it.
    role("cinder", [56, 40, 38], Need::Plain),
    role("petrified", [150, 132, 116], Need::Plain),
    role("tundra", [124, 132, 100], Need::Plain),
    role("glowshroom", [180, 90, 255], Need::Glow(9)),
];

/// The materials a reagent vein can naturally touch: it must lie dormant against all of them except
/// its target, so veins rest in their host rock until a player disturbs them. Against surface,
/// tree and space materials a reagent may react — that is gameplay, not instability.
pub const UNDERGROUND: &[&str] = &[
    "gravel", "rock", "rock1", "rock2", "rock3", "sandstone", "sandstone1", "sandstone2", "sandstone3",
    "deeprock", "abyss", "plank", "rail", "lamp", "rubble", "bone", "glowcap", "crystal", "copper",
    "azurite", "gold", "soil", "sand", "redsand", "clay", "limestone", "marble", "slate", "obsidian",
    "cinder", "petrified", "rust", "amber", "jade",
];

/// Lattice distance every element of a new material keeps from every element already in use.
const GAP: u32 = 24;
/// Underground materials a reagent may react with besides its target (veins sit only in hosts
/// they are dormant against; the rest is what a reagent tool can also eat).
const MAX_REAGENT_SPILL: usize = 6;
/// Candidates nearest the target colour considered per role.
const NEAREST: usize = 3_000;
/// Size of the reactivity panel.
const PANEL: usize = 48;
/// Dormant candidates collected per role before the panel decides between them.
const SHORTLIST: usize = 12;
/// Dormant clusters gathered for the appended plains before any of them is chosen.
const BANK: usize = 64;
/// Samples walked while gathering that bank.
const BANK_SCAN: u32 = 400_000;

/// One palette entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// The role's label.
    pub label: &'static str,
    /// The configuration found for it.
    pub config: Configuration,
}

/// The palette of the current law (computed once per process).
pub fn current() -> &'static [Entry] {
    static CURRENT: OnceLock<Vec<Entry>> = OnceLock::new();
    CURRENT.get_or_init(|| search(&Law::current()))
}

/// The palette of `law`.
pub fn of(law: &Law) -> Vec<Entry> {
    if *law == Law::current() {
        return current().to_vec();
    }
    search(law)
}

fn colour_dist(a: [u8; 3], b: [u8; 3]) -> u32 {
    (0..3).map(|k| (a[k] as i32 - b[k] as i32).unsigned_abs()).sum()
}

fn cohesive(elems: &[Element]) -> bool {
    for i in 0..elems.len() {
        for j in (i + 1)..elems.len() {
            if fit_raw(elems[i], elems[j]) < 0 {
                return false;
            }
        }
    }
    true
}

/// Both orientations are quiescent. A palette block is a handful of occurrences, far below
/// capacity, so the contact can only transfer — and a transfer's gain does not depend on which
/// block is written on the left. The gain is the cross-block fit minus the cached internal hold.
fn dormant(a: &Block, b: &Block) -> bool {
    let (ae, be) = (a.elements(), b.elements());
    let (ah, bh) = (a.holding(), b.holding());
    let (na, nb) = (ae.len(), be.len());
    if na > 8 || nb > 8 {
        return Contact::new(a, b).peek().is_none() && Contact::new(b, a).peek().is_none();
    }
    let threshold = (na + nb) as i32 * (QUANTUM / 8);
    let mut cross_b = [0i32; 8];
    for i in 0..na {
        let mut cross = 0i32;
        for j in 0..nb {
            let k = fit_raw(ae[i], be[j]);
            cross += k;
            cross_b[j] += k;
        }
        if cross - ah[i] > threshold {
            return false;
        }
    }
    for j in 0..nb {
        if cross_b[j] - bh[j] > threshold {
            return false;
        }
    }
    true
}

/// Cluster shapes: one bit per axis, set = +[`STEP`] on that axis. Every pairwise per-axis difference
/// is then 0 or ±STEP, where the fit table is near its maximum (≈ 1.125 Q per differing axis), so
/// every occurrence is held by every other: a strongly cohesive material.
const SHAPES: &[&[u8]] = &[
    &[0b0000, 0b1100, 0b0011, 0b1010, 0b0101],
    &[0b0000, 0b1110, 0b1101, 0b1011, 0b0111],
    &[0b0000, 0b0000, 0b1110, 0b0111, 0b1011],
    &[0b0000, 0b1100, 0b0011, 0b1111],
    &[0b0000, 0b1100, 0b0110, 0b0011, 0b1001, 0b1111],
    &[0b1111, 0b1111, 0b1110, 0b1101, 0b1011, 0b0111],
];
/// The per-axis separation inside a cluster (the fit curve peaks near 54).
const STEP: u8 = 54;
/// Cluster positions tried for colour roles.
const BASES: u32 = 12_288;
/// Cluster positions tried around a probe for glowing and clear roles.
const PROBE_BASES: u32 = 4_096;

/// One candidate material: its occurrences and its (centroid) colour.
struct Candidate {
    elements: Vec<Element>,
    colour: [u8; 3],
}

fn write_cluster(out: &mut Vec<Element>, base: [u8; 4], shape: &[u8], salt: u32) {
    // A per-cluster sign per axis mirrors the shape (−STEP fits exactly like +STEP), giving
    // sixteen orientation families around any base.
    out.clear();
    let signs = hash2(salt, -7, 11) as u8;
    for (k, &bits) in shape.iter().enumerate() {
        let j = hash2(salt, k as i32, bits as i32).to_le_bytes();
        let mut c = base;
        for a in 0..4 {
            let step = match (bits & (1 << a) != 0, signs & (1 << a) != 0) {
                (false, _) => 0,
                (true, false) => STEP,
                (true, true) => STEP.wrapping_neg(),
            };
            // A little jitter so materials built on one shape are not translates of each other.
            c[a] = c[a].wrapping_add(step).wrapping_add(j[a] % 7).wrapping_sub(3);
        }
        out.push(Element::new(c));
    }
}

fn cluster(base: [u8; 4], shape: &[u8], salt: u32) -> Vec<Element> {
    let mut out = Vec::with_capacity(shape.len());
    write_cluster(&mut out, base, shape, salt);
    out
}

fn candidates_at(law: &Law, bases: impl Iterator<Item = [u8; 4]>, salt: u32) -> Vec<Candidate> {
    let mut out = Vec::new();
    for (i, base) in bases.enumerate() {
        for (si, shape) in SHAPES.iter().enumerate() {
            let elements = cluster(base, shape, salt ^ (i as u32) << 3 ^ si as u32);
            let colour = colour_at(law, centroid_q8(&elements).expect("non-empty"));
            out.push(Candidate { colour, elements });
        }
    }
    out
}

/// Candidates spread over the whole lattice (shared by every colour role).
fn lattice_candidates(law: &Law) -> Vec<Candidate> {
    candidates_at(law, (0..BASES).map(|i| hash2(0x5A11_E77E, i as i32, 0).to_le_bytes()), 0xC1u32)
}

/// Candidates hugging a probe element, so the mean probe fit (glow or clarity) is high.
fn probe_candidates(law: &Law, probe: Element) -> Vec<Candidate> {
    let bases = (0..PROBE_BASES).map(move |i| {
        let j = hash2(0x9B0B_E000, i as i32, 1).to_le_bytes();
        let mut c = probe.0;
        for a in 0..4 {
            c[a] = c[a].wrapping_add(j[a] % 41).wrapping_sub(20);
        }
        c
    });
    candidates_at(law, bases, 0xC2u32 ^ u32::from_le_bytes(probe.0))
}

fn needs_met(law: &Law, block: &Block, need: Need) -> bool {
    let o = observe(law, block);
    match need {
        Need::Plain => o.transparency == 0 && o.emission == 0,
        Need::Glow(min) => o.emission >= min && o.transparency == 0,
        Need::Clear { min, glow } => o.transparency >= min && (!glow || o.emission > 0),
        Need::Reagent(_) => true,
    }
}

/// A fixed panel of random cohesive configurations (3-5 occurrences, no repelling pair) that
/// stands for "matter in general" when judging how reactive a candidate is.
fn cohesive_panel() -> Vec<Block> {
    let mut out = Vec::with_capacity(PANEL);
    let mut i = 0i32;
    while out.len() < PANEL {
        let n = 3 + (hash2(0xBA9E_1000, i, -1) % 3) as i32;
        let elems: Vec<Element> = (0..n).map(|k| Element::new(hash2(0xBA9E_1000, i, k).to_le_bytes())).collect();
        i += 1;
        if cohesive(&elems) {
            out.push(Block::new(&elems).expect("small"));
        }
    }
    out
}

/// Search the whole palette under `law`.
pub fn search(law: &Law) -> Vec<Entry> {
    let lattice = lattice_candidates(law);
    let glow = probe_candidates(law, law.probes.glow);
    let clear = probe_candidates(law, law.probes.light);
    let panel = cohesive_panel();
    // Plains appended after the reagents. Searched with the other plains, from a bank gathered
    // once: the shared lattice no longer has a dormant cluster for every new colour.
    let appended = ROLES.iter().rposition(|r| matches!(r.need, Need::Reagent(_))).expect("reagents");
    let mut bank: Option<PlainBank> = None;
    let mut appended_picked: Vec<Block> = Vec::new();
    // The probe-hugging roles have the narrowest candidate sets: search them first (the brightest
    // glow, the scarcest, before all), then colours, then reagents (which need their targets). The
    // palette keeps role order.
    let rank = |r: &Role| match r.need {
        Need::Glow(level) => (0, 15 - level as i32),
        Need::Clear { .. } => (0, 8),
        Need::Plain => (1, 0),
        Need::Reagent(_) => (2, 0),
    };
    let mut order: Vec<usize> = (0..ROLES.len()).collect();
    order.sort_by_key(|&i| (rank(&ROLES[i]), i));
    let mut found: Vec<Option<Configuration>> = vec![None; ROLES.len()];
    let mut taken: Vec<Block> = Vec::with_capacity(ROLES.len());
    let mut taken_role: Vec<usize> = Vec::with_capacity(ROLES.len());
    for &ri in &order {
        let r = &ROLES[ri];
        let config = match r.need {
            Need::Reagent(target) => {
                let t = ROLES.iter().position(|x| x.label == target).expect("reagent target is a role");
                let k = taken_role.iter().position(|&x| x == t).expect("targets are searched before reagents");
                let neighbours: Vec<bool> = taken_role.iter().map(|&x| UNDERGROUND.contains(&ROLES[x].label)).collect();
                find_reagent(law, &taken, &neighbours, k, ri as u32)
            }
            Need::Plain if ri > appended => {
                let bank = bank.get_or_insert_with(|| plain_bank(law, &taken));
                let found = take_appended(bank, &appended_picked, r.rgb);
                if let Some(ref config) = found {
                    appended_picked.push(Block::of(config));
                }
                found
            }
            Need::Plain => find_material(law, &lattice, &panel, &taken, r.rgb, r.need),
            Need::Glow(_) => find_material(law, &glow, &panel, &taken, r.rgb, r.need),
            Need::Clear { .. } => find_material(law, &clear, &panel, &taken, r.rgb, r.need),
        }
        .unwrap_or_else(|| panic!("the law hosts no configuration for role {}", r.label));
        if std::env::var_os("WATT_PALETTE_DEBUG").is_some() {
            let b = Block::of(&config);
            let reactive = panel.iter().filter(|p| !dormant(p, &b)).count();
            eprintln!(
                "palette: {:12} n={} cohesion={} panel-reactive={reactive}/{PANEL}",
                r.label,
                b.len(),
                material::cohesion(&b),
            );
        }
        taken.push(Block::of(&config));
        taken_role.push(ri);
        found[ri] = Some(config);
    }
    ROLES
        .iter()
        .zip(found)
        .map(|(r, c)| Entry { label: r.label, config: c.expect("every role searched") })
        .collect()
}

/// Clusters already dormant against everything chosen before the appended plains.
struct PlainBank {
    blocks: Vec<Block>,
    colour: Vec<[u8; 3]>,
}

/// `true` when the short ring distance is at least [`GAP`]. Stops at the first axis that settles it.
fn apart(a: Element, b: Element) -> bool {
    let mut d = 0u32;
    for i in 0..4 {
        let s = a.0[i].wrapping_sub(b.0[i]);
        d += s.min(s.wrapping_neg()) as u32;
        if d >= GAP {
            return true;
        }
    }
    false
}

/// Used elements grouped by axis 0, so a gap test only looks at the coordinates within [`GAP`].
fn by_axis0(used: &[Element]) -> (Vec<Element>, [usize; 257]) {
    let mut elems = used.to_vec();
    elems.sort_unstable_by_key(|e| e.0[0]);
    let mut start = [0usize; 257];
    let mut k = 0;
    for b in 0..256 {
        while k < elems.len() && (elems[k].0[0] as usize) < b {
            k += 1;
        }
        start[b] = k;
    }
    start[256] = elems.len();
    (elems, start)
}

fn crowded(elems: &[Element], start: &[usize; 257], e: Element) -> bool {
    let x = e.0[0];
    let hit = |u0: u8| {
        let i = u0 as usize;
        elems[start[i]..start[i + 1]].iter().any(|u| !apart(e, *u))
    };
    if hit(x) {
        return true;
    }
    for d in 1..GAP as u8 {
        if hit(x.wrapping_add(d)) || hit(x.wrapping_sub(d)) {
            return true;
        }
    }
    false
}

fn plain_bank(law: &Law, taken: &[Block]) -> PlainBank {
    let used: Vec<Element> = taken.iter().flat_map(|t| t.elements().iter().copied()).collect();
    let (elems, start) = by_axis0(&used);
    let mut blocks = Vec::with_capacity(BANK);
    let mut colour = Vec::with_capacity(BANK);
    let mut scratch = Vec::with_capacity(6);
    let mut i = 0u32;
    while blocks.len() < BANK && i < BANK_SCAN {
        let base = hash2(0xB10B_0001, i as i32, 0).to_le_bytes();
        write_cluster(&mut scratch, base, SHAPES[(i as usize) % SHAPES.len()], 0xB10Bu32 ^ i.wrapping_mul(0x9E37_79B9));
        i += 1;
        if scratch.iter().any(|e| crowded(&elems, &start, *e)) || !cohesive(&scratch) {
            continue;
        }
        let block = Block::new(&scratch).expect("small");
        if !needs_met(law, &block, Need::Plain) || taken.iter().any(|t| !dormant(t, &block)) {
            continue;
        }
        colour.push(colour_at(law, centroid_q8(&scratch).expect("non-empty")));
        blocks.push(block);
    }
    PlainBank { blocks, colour }
}

/// The closest colour in `bank` that stays gapped and dormant against the appended plains already
/// chosen. The bank is already dormant against every earlier material.
fn take_appended(bank: &PlainBank, picked: &[Block], rgb: [u8; 3]) -> Option<Configuration> {
    let used: Vec<Element> = picked.iter().flat_map(|t| t.elements().iter().copied()).collect();
    let mut ranked: Vec<(u32, usize)> = bank.colour.iter().enumerate().map(|(i, c)| (colour_dist(*c, rgb), i)).collect();
    ranked.sort_unstable();
    ranked.into_iter().find(|&(_, i)| {
        let block = &bank.blocks[i];
        block.elements().iter().all(|e| used.iter().all(|u| apart(*e, *u))) && picked.iter().all(|t| dormant(t, block))
    }).map(|(_, i)| bank.blocks[i].configuration())
}

fn find_material(
    law: &Law,
    pool: &[Candidate],
    panel: &[Block],
    taken: &[Block],
    rgb: [u8; 3],
    need: Need,
) -> Option<Configuration> {
    // Elements near a used one behave almost like it, so a near-duplicate would react with that
    // material: every new element keeps a lattice distance from every element in use.
    let used: Vec<Element> = taken.iter().flat_map(|t| t.elements().iter().copied()).collect();
    let (elems, axis0) = by_axis0(&used);
    let clear_of = |c: &Candidate| !c.elements.iter().any(|e| crowded(&elems, &axis0, *e));
    let mut ranked: Vec<(u32, usize)> =
        pool.iter().enumerate().map(|(i, c)| (colour_dist(c.colour, rgb), i)).collect();
    // Usually only the nearest few thousand are walked: partial selection, then sort those; the
    // rest only when none of them qualifies.
    let keep = NEAREST.min(ranked.len());
    if keep < ranked.len() {
        ranked.select_nth_unstable(keep);
    }
    let (near, far) = ranked.split_at_mut(keep);
    near.sort_unstable();
    if let Some(found) = pick(law, pool, panel, taken, &clear_of, near, need) {
        return Some(found);
    }
    far.sort_unstable();
    pick(law, pool, panel, taken, &clear_of, far, need)
}

/// Walk `ranked` (nearest colour first); keep the first few candidates that are cohesive, read as
/// needed and lie dormant against every material already chosen; then prefer the one least
/// reactive with a random panel of cohesive matter (common terrain should be broadly inert, not a
/// sink that drinks whatever touches it).
fn pick(
    law: &Law,
    pool: &[Candidate],
    panel: &[Block],
    taken: &[Block],
    clear_of: &dyn Fn(&Candidate) -> bool,
    ranked: &[(u32, usize)],
    need: Need,
) -> Option<Configuration> {
    let mut accepted: Vec<(i64, usize)> = Vec::new();
    let mut blame: Vec<u32> = Vec::new();
    for &(dist, i) in ranked {
        if accepted.len() >= SHORTLIST {
            break;
        }
        let c = &pool[i];
        if !cohesive(&c.elements) || !clear_of(c) {
            continue;
        }
        let block = Block::new(&c.elements).expect("small");
        if !needs_met(law, &block, need) {
            continue;
        }
        if let Some(k) = taken.iter().position(|t| !dormant(t, &block)) {
            if blame.len() <= k {
                blame.resize(k + 1, 0);
            }
            blame[k] += 1;
            continue;
        }
        let reactive = panel.iter().filter(|p| !dormant(p, &block)).count() as i64;
        accepted.push((dist as i64 * 4 + reactive * 24, i));
    }
    if accepted.is_empty() && std::env::var_os("WATT_PALETTE_DEBUG").is_some() {
        eprintln!("palette: {need:?}: nothing dormant in {} candidates; first reacting partner by index: {blame:?}", ranked.len());
    }
    accepted.into_iter().min().map(|(_, i)| Block::new(&pool[i].elements).expect("small").configuration())
}

/// Occurrences the reagent pulls out of `target` (as the world cell, A) when it is the tool (B),
/// stepping the contact to rest.
fn extracted(target: &Block, reagent: &Block) -> usize {
    let mut c = Contact::new(target, reagent);
    for _ in 0..64 {
        if c.step().is_none() {
            break;
        }
    }
    target.len() - c.counts()[0].min(target.len())
}

/// Reagent shells around one target occurrence `t*`: offsets (bit per axis = ±STEP, one sign per
/// axis for the whole shell) of weight 3 or 4. Every member fits `t*` with ≥ 3.4 Q, and members
/// differ from each other by 0 or ±STEP per axis, so the shell is itself cohesive.
const SHELLS: &[&[u8]] = &[
    &[0b1111, 0b1110, 0b1101, 0b1011, 0b0111],
    &[0b1110, 0b1101, 0b1011, 0b0111],
    &[0b1111, 0b1110, 0b1101, 0b1011],
    &[0b1111, 0b1110, 0b0111],
];

fn find_reagent(_law: &Law, taken: &[Block], neighbours: &[bool], target: usize, salt: u32) -> Option<Configuration> {
    let t = &taken[target];
    let used: Vec<Element> = taken.iter().flat_map(|b| b.elements().iter().copied()).collect();
    // Best = most occurrences extracted, then fewest underground materials it also reacts with.
    let mut best: Option<((usize, usize), Configuration)> = None;
    for (ti, &star) in t.elements().iter().enumerate() {
        for signs in 0..16u8 {
            for (si, shell) in SHELLS.iter().enumerate() {
                let h = hash2(0x7E57_0000 ^ salt, (ti * 16 + signs as usize) as i32, si as i32);
                let elems: Vec<Element> = shell
                    .iter()
                    .enumerate()
                    .map(|(k, &bits)| {
                        let j = hash2(h, k as i32, 3).to_le_bytes();
                        let mut c = star.0;
                        for a in 0..4 {
                            let step = match (bits & (1 << a) != 0, signs & (1 << a) != 0) {
                                (false, _) => 0,
                                (true, false) => STEP,
                                (true, true) => STEP.wrapping_neg(),
                            };
                            c[a] = c[a].wrapping_add(step).wrapping_add(j[a] % 5).wrapping_sub(2);
                        }
                        Element::new(c)
                    })
                    .collect();
                if !cohesive(&elems) || elems.iter().any(|e| used.iter().any(|u| u.ring_distance(*e) < GAP / 2)) {
                    continue;
                }
                let block = Block::new(&elems).expect("small");
                let got = extracted(t, &block);
                if got == 0 {
                    continue;
                }
                let others = taken
                    .iter()
                    .enumerate()
                    .filter(|&(k, other)| k != target && neighbours[k] && !dormant(other, &block))
                    .count();
                if others > MAX_REAGENT_SPILL {
                    continue;
                }
                let key = (t.len() - got, others);
                if best.as_ref().is_none_or(|(b, _)| key < *b) {
                    best = Some((key, block.configuration()));
                }
            }
        }
    }
    best.map(|(_, c)| c)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The contact the kernel would actually run, both ways around.
    fn by_contact(a: &Block, b: &Block) -> bool {
        Contact::new(a, b).peek().is_none() && Contact::new(b, a).peek().is_none()
    }

    #[test]
    fn the_palette_is_cohesive_mutually_dormant_and_reads_as_intended() {
        let law = Law::current();
        let p = current();
        assert_eq!(p.len(), ROLES.len());
        let blocks: Vec<Block> = p.iter().map(|e| Block::of(&e.config)).collect();
        let reagent = |i: usize| matches!(ROLES[i].need, Need::Reagent(_));
        for (i, (e, r)) in p.iter().zip(ROLES).enumerate() {
            assert_eq!(e.label, r.label);
            assert!(cohesive(e.config.elements()), "{} has a repelling pair", e.label);
            assert!(needs_met(&law, &blocks[i], r.need), "{} misses its need {:?}", e.label, r.need);
            if reagent(i) {
                let spill = (0..p.len())
                    .filter(|&j| {
                        if reagent(j)
                            || !UNDERGROUND.contains(&p[j].label)
                            || matches!(r.need, Need::Reagent(t) if t == p[j].label)
                        {
                            return false;
                        }
                        let rest = dormant(&blocks[i], &blocks[j]);
                        assert_eq!(rest, by_contact(&blocks[i], &blocks[j]), "{} vs {}", e.label, p[j].label);
                        !rest
                    })
                    .count();
                assert!(spill <= MAX_REAGENT_SPILL, "{} reacts with {spill} underground materials", e.label);
                continue;
            }
            for j in (i + 1)..p.len() {
                if !reagent(j) {
                    let rest = dormant(&blocks[i], &blocks[j]);
                    assert_eq!(rest, by_contact(&blocks[i], &blocks[j]), "{} vs {}", e.label, p[j].label);
                    assert!(rest, "{} reacts with {}", e.label, p[j].label);
                }
            }
        }
    }

    #[test]
    fn every_reagent_extracts_from_its_target() {
        let p = current();
        for (r, e) in ROLES.iter().zip(p) {
            if let Need::Reagent(target) = r.need {
                let t = p.iter().find(|x| x.label == target).unwrap();
                let got = extracted(&Block::of(&t.config), &Block::of(&e.config));
                assert!(got > 0, "{} extracts nothing from {}", e.label, target);
            }
        }
    }

    #[test]
    fn colours_land_near_their_targets() {
        let law = Law::current();
        let mut total = 0u32;
        let mut n = 0u32;
        for (r, e) in ROLES.iter().zip(current()) {
            if matches!(r.need, Need::Reagent(_)) {
                continue;
            }
            let v = material::visual(&law, &Block::of(&e.config));
            total += colour_dist(v.rgb, r.rgb);
            n += 1;
        }
        let mean = total / n;
        assert!(mean < 120, "mean colour error {mean}");
    }

    /// The v4 roles, their needs, and the rock-hosted strata a vein must lie dormant against.
    #[test]
    fn v4_roles_keep_their_needs_and_star_stays_the_lantern() {
        let want: &[(&str, [u8; 3], Need)] = &[
            ("flower_red", [196, 40, 52], Need::Plain),
            ("flower_yellow", [236, 206, 58], Need::Plain),
            ("flower_blue", [72, 104, 214], Need::Plain),
            ("flower_white", [236, 236, 228], Need::Plain),
            ("cap_red", [176, 44, 40], Need::Plain),
            ("cap_brown", [150, 104, 68], Need::Plain),
            ("stem", [222, 214, 194], Need::Plain),
            ("ash", [74, 72, 70], Need::Plain),
            ("obsidian", [28, 22, 36], Need::Plain),
            ("salt", [240, 238, 230], Need::Plain),
            ("clay", [168, 106, 78], Need::Plain),
            ("limestone", [206, 200, 184], Need::Plain),
            ("marble", [228, 228, 232], Need::Plain),
            ("jade", [76, 168, 118], Need::Plain),
            ("rust", [138, 72, 42], Need::Plain),
            ("mud", [84, 66, 50], Need::Plain),
            ("lichen", [148, 168, 92], Need::Plain),
            ("darkwood", [62, 42, 30], Need::Plain),
            ("bark", [96, 74, 52], Need::Plain),
            ("amber", [218, 140, 40], Need::Plain),
            ("slate", [72, 78, 88], Need::Plain),
            ("cinder", [56, 40, 38], Need::Plain),
            ("petrified", [150, 132, 116], Need::Plain),
            ("tundra", [124, 132, 100], Need::Plain),
            ("glowshroom", [180, 90, 255], Need::Glow(9)),
        ];
        for &(label, rgb, need) in want {
            let role = ROLES.iter().find(|r| r.label == label).unwrap_or_else(|| panic!("missing {label}"));
            assert_eq!((role.rgb, role.need), (rgb, need), "{label}");
        }
        let star = ROLES.iter().find(|r| r.label == "star").expect("star");
        assert_eq!((star.rgb, star.need), ([250, 250, 255], Need::Glow(15)));
        for label in ["clay", "limestone", "marble", "slate", "obsidian", "cinder", "petrified", "rust", "amber", "jade"] {
            assert!(UNDERGROUND.contains(&label), "{label} is a rock-hosted stratum");
        }
    }

    #[test]
    #[ignore]
    fn palette_report() {
        let law = Law::current();
        let t0 = std::time::Instant::now();
        let p = search(&law);
        println!("palette search: {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
        for (r, e) in ROLES.iter().zip(&p) {
            let b = Block::of(&e.config);
            let v = material::visual(&law, &b);
            let o = observe(&law, &b);
            println!(
                "{:15} n={} rgb={:?} target={:?} hard={} clear={} glow={} coh={}",
                e.label,
                e.config.len(),
                v.rgb,
                r.rgb,
                o.hardness,
                o.transparency,
                o.emission,
                o.cohesion
            );
        }
    }
}
