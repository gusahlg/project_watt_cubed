//! Worldgen starting regions: a labelled centre element and a small family of
//! variants, computed from the law rather than authored constants.

use std::sync::OnceLock;

use material::{interact, observe, Configuration, Element, EventKind, Law, Observation};

/// One worldgen family: a centre that observes as the labelled kind, plus six
/// one-axis jitters (failing jitters collapse to the centre).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    /// Debug/semantic name. Never a simulation input.
    pub label: &'static str,
    /// The family's centre element.
    pub centre: Element,
    /// Jitter amplitude along one axis, in lattice units.
    pub spread: u8,
    members: [Element; 7],
}

const SPREAD: u8 = 8;
/// Tried in order when filling a family's six variants. Rest is monotone in
/// event strength, so a larger jitter is preferred when it still sits at rest.
const SPREADS: [u8; 3] = [SPREAD, 4, 1];
const SEARCH_CAP: u32 = 100_000;

const LABELS: [&str; 10] = [
    "rock", "soil", "sand", "clay", "organic", "water", "ice", "snow", "glass", "lamp",
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
}

/// The builtin worldgen regions under `law`. Panic if a label finds nothing
/// within [`SEARCH_CAP`] candidates — that law cannot host this generator.
pub fn builtin(law: &Law) -> Vec<Region> {
    if *law == Law::v0() {
        static V0: OnceLock<Vec<Region>> = OnceLock::new();
        return V0.get_or_init(|| find_all(&Law::v0())).clone();
    }
    find_all(law)
}

/// True when no pair of family members of `regions` changes under `Collision`
/// (the strongest event; rest there implies rest under every weaker kind).
pub fn families_at_rest(law: &Law, regions: &[Region]) -> bool {
    let members: Vec<Configuration> = regions.iter().flat_map(|r| r.family(law)).collect();
    pair_rest(law, &members)
}

fn find_all(law: &Law) -> Vec<Region> {
    let fp = law.fingerprint();
    let mut found: Vec<Region> = Vec::with_capacity(LABELS.len());
    let mut centres: Vec<Element> = Vec::new();
    for (label_i, &label) in LABELS.iter().enumerate() {
        let mut chosen: Option<Region> = None;
        for n in 0..SEARCH_CAP {
            let c = candidate(fp, label_i as u32, n);
            if !fits(law, label, c) {
                continue;
            }
            if !stable_with(law, c, &centres) {
                continue;
            }
            chosen = Some(family_at(law, label, c, &centres));
            break;
        }
        let Some(region) = chosen else {
            panic!("law cannot host region {label}: no candidate in {SEARCH_CAP}");
        };
        centres.push(region.centre);
        found.push(region);
    }
    collapse_unstable(law, &mut found);
    found
}

/// One-axis jitters at `spread`, collapsing any that fail observation or rest
/// with the centre / previous centres. Prefers SPREAD, then 4, then 1, and
/// records whichever amplitude actually filled six stable variants.
fn family_at(law: &Law, label: &'static str, c: Element, centres: &[Element]) -> Region {
    let mut fallback = None;
    for &spread in &SPREADS {
        let mut members = [c; 7];
        let mut filled = 0u8;
        for (k, v) in jitters(c, spread).iter().copied().enumerate() {
            let ok = v != c
                && fits(law, label, v)
                && stable_with(law, v, centres)
                && stable_with(law, v, &[c]);
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
        };
        if filled == 6 {
            return region;
        }
        fallback = Some(region);
    }
    fallback.expect("SPREADS is non-empty")
}

/// Drop any variant that is not at rest with the whole family, so a world built
/// from all members stays still under Collision.
fn collapse_unstable(law: &Law, regions: &mut [Region]) {
    loop {
        let members: Vec<Element> = regions.iter().flat_map(|r| r.members).collect();
        let mut changed = false;
        for r in regions.iter_mut() {
            for i in 1..7 {
                if r.members[i] == r.centre {
                    continue;
                }
                if !stable_with(law, r.members[i], &members) {
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

fn fits(law: &Law, label: &str, e: Element) -> bool {
    matches_label(label, &observe(law, &Configuration::single(e)))
}

fn matches_label(label: &str, o: &Observation) -> bool {
    let opaque = o.solid && o.transparency == 0;
    match label {
        "rock" => o.solid && opaque && o.hardness >= 160,
        "soil" => o.solid && opaque && (90..=150).contains(&o.hardness),
        "sand" => o.solid && opaque && (60..=120).contains(&o.hardness) && o.friction < 100,
        "clay" => o.solid && opaque && (100..=160).contains(&o.hardness) && o.friction >= 140,
        "organic" => o.solid && opaque && (40..=110).contains(&o.hardness),
        "water" => o.liquid && o.transparency >= 120,
        "ice" => o.solid && (60..=160).contains(&o.transparency) && o.hardness >= 120,
        "snow" => o.solid && opaque && o.hardness < 60,
        "glass" => o.solid && o.transparency >= 160,
        "lamp" => o.solid && o.emission >= 8,
        _ => false,
    }
}

fn stable_with(law: &Law, e: Element, others: &[Element]) -> bool {
    let c = Configuration::single(e);
    if interact(law, &c, &c, EventKind::Collision).changed {
        return false;
    }
    for &o in others {
        let d = Configuration::single(o);
        if interact(law, &c, &d, EventKind::Collision).changed
            || interact(law, &d, &c, EventKind::Collision).changed
        {
            return false;
        }
    }
    true
}

fn pair_rest(law: &Law, members: &[Configuration]) -> bool {
    for (i, a) in members.iter().enumerate() {
        for b in members.iter().skip(i) {
            if interact(law, a, b, EventKind::Collision).changed
                || interact(law, b, a, EventKind::Collision).changed
            {
                return false;
            }
        }
    }
    true
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

    #[test]
    fn builtin_regions_observe_as_required_and_sit_at_rest() {
        let law = Law::v0();
        let regions = builtin(&law);
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
            assert!(
                matches_label(r.label, &observe(&law, &Configuration::single(r.centre))),
                "{} centre does not observe as required: {:?}",
                r.label,
                observe(&law, &Configuration::single(r.centre))
            );
            for m in r.family(&law) {
                assert!(
                    matches_label(r.label, &observe(&law, &m)),
                    "{} family member {:?} does not observe as required",
                    r.label,
                    m.elements()
                );
            }
        }
        assert!(families_at_rest(&law, &regions), "a family member pair reacted under Collision");
    }

    #[test]
    fn spread_is_the_jitter_amplitude() {
        let law = Law::v0();
        let regions = builtin(&law);
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
        let a = builtin(&Law::v0());
        let b = builtin(&Law::v0());
        assert_eq!(a, b);
    }
}
