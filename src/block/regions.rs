//! Worldgen starting regions: a labelled centre element, a small family of
//! variants, and three geological strata, computed from the law.

use std::collections::HashSet;
use std::fmt;
use std::sync::OnceLock;

use material::{
    element_changes, element_response, interact, Configuration, Element, EventKind,
    Law, Observation,
};

/// A law that cannot host the builtin worldgen: which label failed, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionError {
    pub label: &'static str,
    pub why: String,
}

impl fmt::Display for RegionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "law cannot host region {}: {}", self.label, self.why)
    }
}

impl std::error::Error for RegionError {}

/// One worldgen family: a centre that observes as the labelled kind, six
/// one-axis jitters (failing jitters collapse to the centre), and three
/// strata sub-centres the generator picks by geology.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    /// Debug/semantic name. Never a simulation input.
    pub label: &'static str,
    /// The family's centre element.
    pub centre: Element,
    /// Jitter amplitude along one axis, in lattice units.
    pub spread: u8,
    members: [Element; 7],
    strata: [Element; 3],
}

const SPREAD: u8 = 8;
/// Tried in order when filling a family's six variants. Rest is monotone in
/// event strength, so a larger jitter is preferred when it still sits at rest.
const SPREADS: [u8; 3] = [SPREAD, 4, 1];
/// Strata try ±12 first (visibly off the centre), then the dead-zone edge, then
/// the family spread — only offsets that stay in-band and at rest are kept.
const STRATUM_DELTAS: [u8; 5] = [12, 10, 8, 4, 1];
const SEARCH_CAP: u32 = 40_000;
/// L1 separation required between two region centres (same element must not
/// serve two labels).
const DISTINCT_L1: u32 = 48;
/// Kernel dead zone (inclusive) and far inert floor: an axis is inert when
/// its absolute difference is in this set.
const DEAD_ZONE: u32 = 10;
const FAR_INERT: u32 = 64;

const LABELS: [&str; 10] = [
    "rock", "soil", "sand", "clay", "organic", "water", "ice", "snow", "glass", "lamp",
];
/// Rare observation classes first so the cross-stability filter cannot starve
/// them of the few centres they have.
const SEARCH_ORDER: [&str; 10] = [
    "water", "clay", "lamp", "glass", "sand", "snow", "ice", "soil", "organic", "rock",
];

impl Region {
    /// Centre plus the six variants, as configurations.
    pub fn family(&self, _law: &Law) -> [Configuration; 7] {
        self.members.map(Configuration::single)
    }

    /// Family member `0` is the centre; `1..=6` are the variants (a collapsed
    /// variant equals the centre).
    pub fn member(&self, i: usize) -> Element {
        self.members[i]
    }

    /// Geological sub-centre `0..=2` (a collapsed stratum equals the centre).
    pub fn stratum(&self, i: usize) -> Element {
        self.strata[i]
    }

    /// Centre, variants and strata — every element this family can emit.
    pub fn matter(&self) -> impl Iterator<Item = Element> + '_ {
        self.members.iter().copied().chain(self.strata.iter().copied())
    }
}

/// The builtin worldgen regions under `law`. `Err` if a label finds nothing
/// within [`SEARCH_CAP`] candidates — that law cannot host this generator.
pub fn builtin(law: &Law) -> Result<Vec<Region>, RegionError> {
    if *law == Law::v0() {
        static V0: OnceLock<Vec<Region>> = OnceLock::new();
        return Ok(V0.get_or_init(|| find_all(&Law::v0()).expect("v0 hosts all regions")).clone());
    }
    find_all(law)
}

/// True when no pair of family members or strata of `regions` changes under
/// `Collision` (the strongest event; rest there implies rest under every weaker
/// kind).
pub fn families_at_rest(law: &Law, regions: &[Region]) -> bool {
    let mut members: Vec<Configuration> = regions.iter().flat_map(|r| r.family(law)).collect();
    members.extend(regions.iter().flat_map(|r| r.strata.map(Configuration::single)));
    pair_rest(law, &members, EventKind::Collision)
}

/// True when no ordered pair of `members` changes under `kind`.
pub fn pair_rest(law: &Law, members: &[Configuration], kind: EventKind) -> bool {
    for (i, a) in members.iter().enumerate() {
        for b in members.iter().skip(i) {
            if interact(law, a, b, kind).changed || interact(law, b, a, kind).changed {
                return false;
            }
        }
    }
    true
}

/// Counters for the region-search cost test. Cheap increments, no clocks.
#[derive(Clone, Copy, Debug, Default)]
struct SearchCost {
    candidates: u32,
    obs_early: u32,
    obs_full: u32,
    rest: u32,
    rest_skip: u32,
}

struct Search {
    law: Law,
    /// Elements that are not at rest with themselves under Collision. A later
    /// variant of the same centre is the same element and can be skipped.
    failed: HashSet<Element>,
    cost: SearchCost,
}

fn find_all(law: &Law) -> Result<Vec<Region>, RegionError> {
    search(law).map(|(regions, _)| regions)
}

fn search(law: &Law) -> Result<(Vec<Region>, SearchCost), RegionError> {
    let fp = law.fingerprint();
    let mut s = Search {
        law: *law,
        failed: HashSet::new(),
        cost: SearchCost::default(),
    };
    let mut found: Vec<Region> = Vec::with_capacity(LABELS.len());
    let mut centres: Vec<Element> = Vec::new();
    for &label in &SEARCH_ORDER {
        let label_i = LABELS.iter().position(|&l| l == label).expect("SEARCH_ORDER ⊆ LABELS");
        let mut best: Option<(u32, u32, Element)> = None;
        for n in 0..SEARCH_CAP {
            let c = candidate(fp, label_i as u32, n);
            s.cost.candidates += 1;
            let Some(obs) = s.fits_obs(label, c) else { continue };
            if !distinct(c, &centres) || !axis_inert_with(c, &centres) {
                continue;
            }
            if !s.stable_with(c, &centres) {
                continue;
            }
            let score = band_score(label, &obs);
            match best {
                Some((bs, bn, _)) if (score, n) >= (bs, bn) => {}
                _ => best = Some((score, n, c)),
            }
            if score == 0 {
                break;
            }
        }
        let Some((_, _, c)) = best else {
            return Err(RegionError {
                label,
                why: format!("no candidate in {SEARCH_CAP}"),
            });
        };
        let region = s.family_at(label, c, &centres);
        centres.push(region.centre);
        found.push(region);
    }
    found.sort_by_key(|r| LABELS.iter().position(|&l| l == r.label).unwrap_or(99));
    s.collapse_unstable(&mut found);
    s.fill_strata(&mut found);
    Ok((found, s.cost))
}

impl Search {
    fn fits_obs(&mut self, label: &str, e: Element) -> Option<Observation> {
        observe_if_fits(&self.law, label, e, &mut self.cost)
    }

    fn fits(&mut self, label: &str, e: Element) -> bool {
        self.fits_obs(label, e).is_some()
    }

    fn stable_with(&mut self, e: Element, others: &[Element]) -> bool {
        if self.failed.contains(&e) {
            self.cost.rest_skip += 1;
            return false;
        }
        self.cost.rest += 1;
        if element_changes(&self.law, e, e, EventKind::Collision) {
            self.failed.insert(e);
            return false;
        }
        for &o in others {
            if element_changes(&self.law, e, o, EventKind::Collision)
                || element_changes(&self.law, o, e, EventKind::Collision)
            {
                return false;
            }
        }
        true
    }

    /// One-axis jitters at `spread`, collapsing any that fail observation or rest
    /// with the centre / previous centres. Prefers SPREAD, then 4, then 1, and
    /// records whichever amplitude actually filled six stable variants.
    fn family_at(&mut self, label: &'static str, c: Element, centres: &[Element]) -> Region {
        let mut fallback = None;
        for &spread in &SPREADS {
            let mut members = [c; 7];
            let mut filled = 0u8;
            for (k, v) in jitters(c, spread).iter().copied().enumerate() {
                let ok = v != c
                    && self.fits(label, v)
                    && distinct(v, centres)
                    && axis_inert_with(v, centres)
                    && axis_inert_with(v, &[c])
                    && self.stable_with(v, centres)
                    && self.stable_with(v, &[c]);
                if ok {
                    members[k + 1] = v;
                    filled += 1;
                }
            }
            let region = Region {
                label,
                centre: c,
                spread,
                members,
                strata: [c; 3],
            };
            if filled == 6 {
                return region;
            }
            fallback = Some(region);
        }
        fallback.expect("SPREADS is non-empty")
    }

    /// Three sub-centres per region, picked after variant collapse so the rest
    /// filter sees the matter the world actually emits. ±12 first, then smaller
    /// dead-zone offsets; one axis each.
    fn fill_strata(&mut self, regions: &mut [Region]) {
        for i in 0..regions.len() {
            let label = regions[i].label;
            let c = regions[i].centre;
            let mut out = [c; 3];
            let mut filled = 0usize;
            let mut used_axis = [false; 4];
            let others: Vec<Element> = regions
                .iter()
                .enumerate()
                .flat_map(|(j, r)| {
                    if j == i {
                        r.members.iter().copied().collect::<Vec<_>>()
                    } else {
                        r.matter().collect()
                    }
                })
                .collect();
            for &delta in &STRATUM_DELTAS {
                if filled == 3 {
                    break;
                }
                for axis in [3usize, 0, 1, 2] {
                    if filled == 3 {
                        break;
                    }
                    if used_axis[axis] {
                        continue;
                    }
                    for &sign in &[1i16, -1] {
                        let v = c.0[axis] as i16 + sign * delta as i16;
                        if !(0..=255).contains(&v) {
                            continue;
                        }
                        let mut e = c;
                        e.0[axis] = v as u8;
                        if e == c || (0..filled).any(|k| out[k] == e) {
                            continue;
                        }
                        if !self.fits(label, e) {
                            continue;
                        }
                        if !axis_inert(e, c) {
                            continue;
                        }
                        if !axis_inert_with(e, &out[..filled]) {
                            continue;
                        }
                        if !self.stable_with(e, &others) || !self.stable_with(e, &out[..filled]) {
                            continue;
                        }
                        out[filled] = e;
                        used_axis[axis] = true;
                        filled += 1;
                        break;
                    }
                }
            }
            regions[i].strata = out;
        }
    }

    /// Drop any variant that is not at rest with the whole family, so a world
    /// built from all members stays still under Collision. Strata are filled
    /// afterwards against this collapsed set.
    fn collapse_unstable(&mut self, regions: &mut [Region]) {
        loop {
            let members: Vec<Element> = regions.iter().flat_map(|r| r.members).collect();
            let mut changed = false;
            for r in regions.iter_mut() {
                for i in 1..7 {
                    if r.members[i] == r.centre {
                        continue;
                    }
                    if !self.stable_with(r.members[i], &members) {
                        r.members[i] = r.centre;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }
}

/// Reject on the cheapest probes the label needs. A full five-probe
/// [`observe_element`] runs only for a survivor (band_score). Same accept set
/// as `observe_element` + [`matches_label`].
fn observe_if_fits(law: &Law, label: &str, e: Element, cost: &mut SearchCost) -> Option<Observation> {
    let p = &law.probes;
    let flow = element_response(law, e, p.flow);
    let liquid = flow >= p.liquid_min;
    let o = match label {
        "water" => {
            if !liquid {
                cost.obs_early += 1;
                return None;
            }
            let light = element_response(law, e, p.light);
            Observation::from_responses(law, 0, light, flow, 0, 0)
        }
        "lamp" => {
            if liquid {
                cost.obs_early += 1;
                return None;
            }
            let glow = element_response(law, e, p.glow);
            Observation::from_responses(law, 0, 0, flow, glow, 0)
        }
        "glass" => {
            if liquid {
                cost.obs_early += 1;
                return None;
            }
            let light = element_response(law, e, p.light);
            Observation::from_responses(law, 0, light, flow, 0, 0)
        }
        "ice" => {
            if liquid {
                cost.obs_early += 1;
                return None;
            }
            let light = element_response(law, e, p.light);
            let contact = element_response(law, e, p.contact);
            Observation::from_responses(law, contact, light, flow, 0, 0)
        }
        "sand" | "clay" => {
            if liquid {
                cost.obs_early += 1;
                return None;
            }
            let contact = element_response(law, e, p.contact);
            let hardness = 255 - contact;
            if label == "sand" {
                if !(60..=120).contains(&hardness) {
                    cost.obs_early += 1;
                    return None;
                }
            } else if !(100..=160).contains(&hardness) {
                cost.obs_early += 1;
                return None;
            }
            let light = element_response(law, e, p.light);
            let friction = element_response(law, e, p.friction);
            Observation::from_responses(law, contact, light, flow, 0, friction)
        }
        _ => {
            // rock, soil, organic, snow: solid + opaque + a hardness band.
            if liquid {
                cost.obs_early += 1;
                return None;
            }
            let contact = element_response(law, e, p.contact);
            let hardness = 255 - contact;
            let hardness_ok = match label {
                "rock" => hardness >= 160,
                "soil" => (90..=150).contains(&hardness),
                "organic" => (30..=130).contains(&hardness),
                "snow" => hardness < 60,
                _ => false,
            };
            if !hardness_ok {
                cost.obs_early += 1;
                return None;
            }
            let light = element_response(law, e, p.light);
            Observation::from_responses(law, contact, light, flow, 0, 0)
        }
    };
    if !matches_label(label, &o) {
        cost.obs_early += 1;
        return None;
    }
    cost.obs_full += 1;
    Some(o)
}

fn matches_label(label: &str, o: &Observation) -> bool {
    let opaque = o.solid && o.transparency == 0;
    match label {
        "rock" => o.solid && opaque && o.hardness >= 160,
        "soil" => o.solid && opaque && (90..=150).contains(&o.hardness),
        "sand" => o.solid && opaque && (60..=120).contains(&o.hardness) && o.friction < 100,
        "clay" => o.solid && opaque && (100..=160).contains(&o.hardness) && o.friction >= 120,
        "organic" => o.solid && opaque && (30..=130).contains(&o.hardness),
        "water" => o.liquid && o.transparency >= 120,
        "ice" => o.solid && (40..=180).contains(&o.transparency) && o.hardness >= 100,
        "snow" => o.solid && opaque && o.hardness < 60,
        "glass" => o.solid && o.transparency >= 160,
        "lamp" => o.solid && o.emission >= 8,
        _ => false,
    }
}

/// Distance from the middle of the class band — lower is better.
fn band_score(label: &str, o: &Observation) -> u32 {
    let mid = |v: u8, lo: u8, hi: u8| (v as i32 - (lo as i32 + hi as i32) / 2).unsigned_abs();
    match label {
        "rock" => mid(o.hardness, 160, 255),
        "soil" => mid(o.hardness, 90, 150),
        "sand" => mid(o.hardness, 60, 120) + o.friction as u32,
        "clay" => mid(o.hardness, 100, 160) + mid(o.friction, 120, 255),
        "organic" => mid(o.hardness, 30, 130),
        "water" => mid(o.transparency, 120, 255),
        "ice" => mid(o.transparency, 40, 180),
        "snow" => mid(o.hardness, 0, 59),
        "glass" => mid(o.transparency, 160, 255),
        "lamp" => mid(o.emission, 8, 15),
        _ => 0,
    }
}

fn distinct(e: Element, others: &[Element]) -> bool {
    others.iter().all(|&o| e.distance(o) >= DISTINCT_L1)
}

/// Every axis difference is in the dead zone or past the far inert floor.
fn axis_inert(a: Element, b: Element) -> bool {
    (0..4).all(|i| {
        let d = (a.0[i] as i32 - b.0[i] as i32).unsigned_abs();
        d <= DEAD_ZONE || d >= FAR_INERT
    })
}

fn axis_inert_with(e: Element, others: &[Element]) -> bool {
    others.iter().all(|&o| axis_inert(e, o))
}

/// Six one-axis jitters: axes 0..=2 × {+spread, −spread}, clamped to the lattice.
fn jitters(centre: Element, spread: u8) -> [Element; 6] {
    let mut out = [centre; 6];
    let mut i = 0;
    for axis in 0..3 {
        for &sign in &[1i16, -1] {
            let mut e = centre;
            let v = centre.0[axis] as i16 + sign * spread as i16;
            e.0[axis] = v.clamp(0, 255) as u8;
            out[i] = e;
            i += 1;
        }
    }
    out
}

/// `inoise`-style mixer over (law fingerprint, label index, candidate n).
fn hash32(fp: u64, label: u32, n: u32) -> u32 {
    let mut h = (fp as u32) ^ (fp >> 32) as u32;
    h ^= label.wrapping_mul(0x9E37_79B1);
    h = h.rotate_left(13).wrapping_mul(5).wrapping_add(0xE654_6B64);
    h ^= n.wrapping_mul(0x85EB_CA77);
    h = h.rotate_left(13).wrapping_mul(5).wrapping_add(0xE654_6B64);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 16;
    h
}

fn candidate(fp: u64, label: u32, n: u32) -> Element {
    let h = hash32(fp, label, n);
    Element::new([h as u8, (h >> 8) as u8, (h >> 16) as u8, (h >> 24) as u8])
}

#[cfg(test)]
mod tests {
    use super::*;
    use material::observe_element;
    use material::observe;

    #[test]
    fn builtin_regions_observe_as_required_and_sit_at_rest() {
        let law = Law::v0();
        let regions = builtin(&law).expect("v0 hosts all regions");
        assert_eq!(regions.len(), LABELS.len());
        for (i, r) in regions.iter().enumerate() {
            assert_eq!(r.label, LABELS[i]);
            assert!(
                SPREADS.contains(&r.spread),
                "{} spread {} is not in {SPREADS:?}",
                r.label,
                r.spread
            );
            assert_eq!(r.centre, r.members[0]);
            let obs = observe(&law, &Configuration::single(r.centre));
            println!(
                "region {:>8} centre=[{:3},{:3},{:3},{:3}] spread={}  solid={} liquid={} t={:3} e={:2} h={:3} f={:3}",
                r.label,
                r.centre.0[0],
                r.centre.0[1],
                r.centre.0[2],
                r.centre.0[3],
                r.spread,
                obs.solid,
                obs.liquid,
                obs.transparency,
                obs.emission,
                obs.hardness,
                obs.friction,
            );
            assert!(
                matches_label(r.label, &obs),
                "{} centre does not observe as required: {:?}",
                r.label,
                obs
            );
            for m in r.matter() {
                assert!(
                    matches_label(r.label, &observe(&law, &Configuration::single(m))),
                    "{} family/stratum {:?} does not observe as required",
                    r.label,
                    m.0
                );
            }
        }
        assert!(families_at_rest(&law, &regions), "a family member pair reacted under Collision");
    }

    #[test]
    fn centres_are_distinct_and_cross_inert() {
        let law = Law::v0();
        let regions = builtin(&law).expect("v0 hosts all regions");
        for (i, a) in regions.iter().enumerate() {
            for b in regions.iter().skip(i + 1) {
                assert!(
                    a.centre.distance(b.centre) >= DISTINCT_L1,
                    "{} and {} centres are L1 {} < {DISTINCT_L1}",
                    a.label,
                    b.label,
                    a.centre.distance(b.centre)
                );
                assert!(
                    axis_inert(a.centre, b.centre),
                    "{} and {} are not axis-inert",
                    a.label,
                    b.label
                );
            }
        }
        let lamp = regions.iter().find(|r| r.label == "lamp").unwrap();
        let rock = regions.iter().find(|r| r.label == "rock").unwrap();
        assert_ne!(lamp.centre, rock.centre, "lamp must not collapse onto rock");
        assert!(lamp.centre.distance(rock.centre) >= DISTINCT_L1);
    }

    #[test]
    fn each_region_has_three_strata() {
        let law = Law::v0();
        let regions = builtin(&law).expect("v0 hosts all regions");
        let mut rock_soil_variety = 0;
        for r in &regions {
            let distinct = r.strata.iter().filter(|s| **s != r.centre).count();
            if matches!(r.label, "rock" | "soil") && distinct >= 1 {
                rock_soil_variety += 1;
            }
            let axes: Vec<_> = r
                .strata
                .iter()
                .filter(|s| **s != r.centre)
                .map(|s| {
                    let d: Vec<_> = (0..4).filter(|&a| s.0[a] != r.centre.0[a]).collect();
                    assert_eq!(d.len(), 1, "{} stratum is not a one-axis offset", r.label);
                    d[0]
                })
                .collect();
            let mut seen = [false; 4];
            for a in axes {
                assert!(!seen[a], "{} reused axis {a} for two strata", r.label);
                seen[a] = true;
            }
        }
        assert_eq!(rock_soil_variety, 2, "rock and soil must each have a geological stratum");
    }

    #[test]
    fn spread_is_the_jitter_amplitude() {
        let law = Law::v0();
        let regions = builtin(&law).expect("v0 hosts all regions");
        for r in &regions {
            for i in 1..7 {
                let v = r.members[i];
                if v == r.centre {
                    continue;
                }
                let axes: Vec<_> = (0..4).filter(|&a| v.0[a] != r.centre.0[a]).collect();
                assert_eq!(axes.len(), 1, "{}#{} is not a one-axis jitter", r.label, i);
                let a = axes[0];
                let d = (v.0[a] as i16 - r.centre.0[a] as i16).unsigned_abs() as u8;
                let unclamped = r.centre.0[a] as i16
                    + if v.0[a] > r.centre.0[a] {
                        r.spread as i16
                    } else {
                        -(r.spread as i16)
                    };
                if (0..=255).contains(&unclamped) {
                    assert_eq!(d, r.spread, "{}#{} jitter is {d}, spread is {}", r.label, i, r.spread);
                } else {
                    assert!(d <= r.spread, "{}#{} clamped jitter {d} exceeds spread {}", r.label, i, r.spread);
                }
            }
        }
    }

    #[test]
    fn two_scans_agree() {
        let a = builtin(&Law::v0()).expect("v0 hosts all regions");
        let b = builtin(&Law::v0()).expect("v0 hosts all regions");
        assert_eq!(a, b);
    }

    #[test]
    fn fresh_search_matches_cached_v0() {
        let law = Law::v0();
        let (fresh, _) = search(&law).expect("v0 hosts all regions");
        assert_eq!(fresh, builtin(&law).expect("v0 hosts all regions"));
    }

    #[test]
    fn observation_early_reject_matches_full_observe() {
        let law = Law::v0();
        let mut s = Search {
            law,
            failed: HashSet::new(),
            cost: SearchCost::default(),
        };
        let mut h = 0xC0FF_EE00_u64;
        for &label in &LABELS {
            for _ in 0..1_200 {
                h ^= h << 13;
                h ^= h >> 7;
                h ^= h << 17;
                let e = Element::new([
                    h as u8,
                    (h >> 8) as u8,
                    (h >> 16) as u8,
                    (h >> 24) as u8,
                ]);
                let full = matches_label(label, &observe_element(&law, e));
                assert_eq!(
                    s.fits_obs(label, e).is_some(),
                    full,
                    "{label} early-reject disagreed for {:?}",
                    e.0
                );
            }
        }
    }

    #[test]
    fn lamp_centre_emits_at_least_eight() {
        let law = Law::v0();
        let lamp = builtin(&law).expect("v0 hosts all regions").into_iter().find(|r| r.label == "lamp").unwrap();
        let obs = observe(&law, &Configuration::single(lamp.centre));
        assert!(
            obs.emission >= 8,
            "lamp centre {:?} emission {} (glow_min={})",
            lamp.centre.0,
            obs.emission,
            law.probes.glow_min
        );
        assert!(obs.solid, "lamp centre must be solid");
    }

    #[test]
    fn similarity_holds_on_region_families() {
        let law = Law::v0();
        let regions = builtin(&law).expect("v0 hosts all regions");
        let mut worst = 0i32;
        for r in &regions {
            for e in r.matter() {
                for axis in 0..4 {
                    if e.0[axis] == 255 {
                        continue;
                    }
                    let mut e2 = e;
                    e2.0[axis] += 1;
                    for t in r.matter() {
                        let f1 = material::element_influence(&law, e, t).0;
                        let f2 = material::element_influence(&law, e2, t).0;
                        for i in 0..4 {
                            worst = worst.max((f1[i] as i32 - f2[i] as i32).abs());
                        }
                    }
                }
            }
        }
        assert!(
            worst <= 6,
            "one lattice step on a region family changed an influence by {worst}"
        );
    }

    fn flipped_knot() -> Law {
        let mut law = Law::v0();
        law.kernel.knots[2].1 = -law.kernel.knots[2].1;
        law
    }

    #[test]
    fn perturbed_law_compiles_or_errors_without_panic() {
        let law = flipped_knot();
        assert_ne!(law, Law::v0());
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| builtin(&law)));
        assert!(outcome.is_ok(), "region search panics on a non-v0 law");
        match outcome.unwrap() {
            Ok(regions) => {
                assert_eq!(regions.len(), LABELS.len());
                let mut r = crate::block::BlockRegistry::new(law);
                assert!(r.region_error().is_none());
                crate::world::placement::builtin()
                    .compile(&mut r)
                    .expect("a law that found every label compiles");
            }
            Err(e) => {
                assert!(
                    LABELS.contains(&e.label),
                    "failed label {} is not a worldgen label",
                    e.label
                );
                assert!(!e.why.is_empty(), "error names why {e}");
                let mut r = crate::block::BlockRegistry::new(law);
                assert!(r.region_error().is_some());
                assert!(crate::world::placement::builtin().compile(&mut r).is_err());
            }
        }
    }

    #[test]
    #[ignore]
    fn region_compile_stays_under_five_ms() {
        use crate::block::BlockRegistry;
        use crate::world::placement;
        let mut times = [0u128; 8];
        for t in times.iter_mut() {
            let mut r = BlockRegistry::with_builtins();
            let t0 = std::time::Instant::now();
            placement::builtin().compile(&mut r).expect("v0 hosts the placement table");
            *t = t0.elapsed().as_micros();
        }
        let first = times[0];
        times.sort_unstable();
        let mid = times[times.len() / 2];
        println!(
            "region compile first {} µs (v0 intern, OnceLock warm) median {} µs (samples {times:?})",
            first, mid
        );
        assert!(
            mid < 5_000,
            "region compile median {mid} µs must stay under 5 ms"
        );
    }

    #[test]
    #[ignore]
    fn region_search_cold_under_forty_ms() {
        use std::time::Instant;
        let law = Law::v0();
        let fp = law.fingerprint();
        const N: u32 = 20_000;

        let t0 = Instant::now();
        for n in 0..N {
            let e = candidate(fp, 0, n);
            let _ = observe(&law, &Configuration::single(e));
        }
        let alloc_obs = t0.elapsed();

        let t0 = Instant::now();
        for n in 0..N {
            let e = candidate(fp, 0, n);
            let _ = observe_element(&law, e);
        }
        let fast_obs = t0.elapsed();

        let t0 = Instant::now();
        for n in 0..N {
            let a = candidate(fp, 1, n);
            let b = candidate(fp, 2, n);
            let _ = interact(
                &law,
                &Configuration::single(a),
                &Configuration::single(b),
                EventKind::Collision,
            )
            .changed;
        }
        let alloc_rest = t0.elapsed();

        let t0 = Instant::now();
        for n in 0..N {
            let a = candidate(fp, 1, n);
            let b = candidate(fp, 2, n);
            let _ = element_changes(&law, a, b, EventKind::Collision);
        }
        let fast_rest = t0.elapsed();

        let t0 = Instant::now();
        let (regions, cost) = search(&law).expect("v0 hosts all regions");
        let cold = t0.elapsed();

        let _ = builtin(&law);
        let t0 = Instant::now();
        let cached = builtin(&law).expect("v0 hosts all regions");
        let warm = t0.elapsed();

        println!(
            "region search {N} alloc-observe {:?}  observe_element {:?}  alloc-rest {:?}  element_changes {:?}",
            alloc_obs, fast_obs, alloc_rest, fast_rest
        );
        println!(
            "region search cold {:?} warm {:?}  candidates={} obs_early={} obs_full={} rest={} rest_skip={}",
            cold,
            warm,
            cost.candidates,
            cost.obs_early,
            cost.obs_full,
            cost.rest,
            cost.rest_skip
        );
        assert_eq!(regions, cached);
        assert!(
            cold.as_millis() < 40,
            "cold region search {:?} must stay under 40 ms",
            cold
        );
        assert!(
            warm.as_millis() < 10,
            "warm region search {:?} must stay under 10 ms",
            warm
        );
    }
}
