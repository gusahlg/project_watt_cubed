//! Turns a [`Composition`] into the observable properties of a block. This is the
//! heart of the "blocks are averages of their elements" rule: every core property
//! is the weight-average of the contributing elements, the colour is the same
//! average applied to element tints, and special behaviours are summed by kind.
//!
//! Derivation runs once per distinct block at registration, never per voxel, so it
//! favours clarity over raw speed — the hot path reads the precomputed results.
use raylib::prelude::*;

use crate::block::composition::Composition;
use crate::block::element::{CoreProperties, ElementRegistry, SpecialKind};

/// The nine core fields in a fixed order, so derivation can loop over them.
const FIELD_COUNT: usize = 9;

/// Read an element's core properties into the fixed-order array used for averaging.
fn fields(c: &CoreProperties) -> [u8; FIELD_COUNT] {
    [
        c.durability,
        c.hardness,
        c.conductivity,
        c.thermal_conductivity,
        c.density,
        c.temperature_resistance,
        c.friction,
        c.light_emission,
        c.transparency,
    ]
}

/// Rebuild core properties from the fixed-order array.
fn from_fields(f: [u8; FIELD_COUNT]) -> CoreProperties {
    CoreProperties {
        durability: f[0],
        hardness: f[1],
        conductivity: f[2],
        thermal_conductivity: f[3],
        density: f[4],
        temperature_resistance: f[5],
        friction: f[6],
        light_emission: f[7],
        transparency: f[8],
    }
}

/// Each core property of a block is the weighted average of its elements'. With
/// natural weights of `1` this is the plain mean (so equal parts of `1, 2, 3`
/// derive `2`); with mixture percentages it is the percentage-weighted mean.
pub fn derive_core(els: &ElementRegistry, comp: &Composition) -> CoreProperties {
    let mut acc = [0u32; FIELD_COUNT];
    let mut total = 0u32;

    for (id, weight) in comp.weights().iter().copied() {
        let f = fields(&els.get(id).core);
        for i in 0..FIELD_COUNT {
            acc[i] += f[i] as u32 * weight as u32;
        }
        total += weight as u32;
    }

    if total == 0 {
        return CoreProperties::default();
    }
    from_fields(std::array::from_fn(|i| (acc[i] / total) as u8))
}

/// The block's colour is its element tints averaged by the same weights — a
/// 70/30 soil/clay mix looks 70% soil. Air (no elements) is transparent.
pub fn derive_color(els: &ElementRegistry, comp: &Composition) -> Color {
    let mut acc = [0u32; 4];
    let mut total = 0u32;

    for (id, weight) in comp.weights().iter().copied() {
        let c = els.get(id).color;
        let channels = [c.r, c.g, c.b, c.a];
        for i in 0..4 {
            acc[i] += channels[i] as u32 * weight as u32;
        }
        total += weight as u32;
    }

    if total == 0 {
        return Color::new(0, 0, 0, 0);
    }
    Color::new(
        (acc[0] / total) as u8,
        (acc[1] / total) as u8,
        (acc[2] / total) as u8,
        (acc[3] / total) as u8,
    )
}

/// A block is solid unless it has no material in it. Air — the empty natural
/// block — is the sole exception; it is the only thing the renderer culls and the
/// only thing the player walks through.
pub fn derive_solid(comp: &Composition) -> bool {
    !comp.is_empty()
}

/// Special behaviours a block exhibits, each scaled by how much of the carrying
/// element it contains and summed across carriers. Returned sorted by kind for a
/// stable, inspectable order.
pub fn derive_specials(els: &ElementRegistry, comp: &Composition) -> Box<[(SpecialKind, u8)]> {
    let weights = comp.weights();
    let total: u32 = weights.iter().map(|&(_, w)| w as u32).sum();
    if total == 0 {
        return Box::from([]);
    }

    // Accumulate each kind's weighted strength across every element that carries it.
    let mut sums: Vec<(SpecialKind, u32)> = Vec::new();
    for (id, weight) in weights.iter().copied() {
        for special in els.get(id).specials.iter().copied() {
            let contribution = special.strength() as u32 * weight as u32;
            match sums.iter_mut().find(|(k, _)| *k == special.kind()) {
                Some((_, acc)) => *acc += contribution,
                None => sums.push((special.kind(), contribution)),
            }
        }
    }

    sums.sort_by_key(|&(k, _)| k);
    sums.into_iter()
        .map(|(k, acc)| (k, (acc / total) as u8))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::element::El;

    fn registry() -> ElementRegistry {
        ElementRegistry::with_builtins()
    }

    #[test]
    fn documented_average_holds() {
        // A bespoke three-element registry with durabilities 1, 2, 3: equal parts
        // must derive exactly 2, the documented example.
        let mut els = ElementRegistry::with_builtins();
        let mut mk = |d: u8| {
            els.register(crate::block::element::Element {
                name: "t".into(),
                core: CoreProperties { durability: d, ..Default::default() },
                color: Color::new(0, 0, 0, 255),
                specials: Box::from([]),
            })
        };
        let (a, b, c) = (mk(1), mk(2), mk(3));
        let comp = Composition::natural(&[a, b, c]);
        assert_eq!(derive_core(&els, &comp).durability, 2);
    }

    #[test]
    fn air_is_not_solid_and_transparent() {
        let comp = Composition::natural(&[]);
        assert!(!derive_solid(&comp));
        assert_eq!(derive_color(&registry(), &comp), Color::new(0, 0, 0, 0));
        assert_eq!(derive_core(&registry(), &comp), CoreProperties::default());
    }

    #[test]
    fn single_element_natural_keeps_its_values() {
        let els = registry();
        let comp = Composition::natural(&[El::Stone.id()]);
        let stone = els.get(El::Stone.id());
        assert_eq!(derive_core(&els, &comp), stone.core);
        assert_eq!(derive_color(&els, &comp), stone.color);
    }

    #[test]
    fn mixture_blends_toward_majority() {
        let els = registry();
        // 70% soil / 30% clay: density between the two, nearer soil's 110.
        let comp = Composition::mixture(&[(El::Soil.id(), 70), (El::Clay.id(), 30)]).unwrap();
        let d = derive_core(&els, &comp).density;
        assert_eq!(d, ((110u32 * 70 + 130 * 30) / 100) as u8); // 116
    }

    #[test]
    fn specials_scale_with_share() {
        let els = registry();
        // Half sulfur: explosion strength halved from its intrinsic 160.
        let comp = Composition::mixture(&[(El::Sulfur.id(), 50), (El::Stone.id(), 50)]).unwrap();
        let specials = derive_specials(&els, &comp);
        let explosion = specials
            .iter()
            .find(|(k, _)| *k == SpecialKind::ExplosionAtBreakage)
            .map(|&(_, v)| v);
        assert_eq!(explosion, Some(80));
    }
}
