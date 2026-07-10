//! Turns a [`Composition`] into the observable properties of a block. This is the
//! heart of the "blocks are averages of their elements" rule: every core property
//! is the weight-average of the contributing elements, the colour is the same
//! average applied to element tints, and special behaviours are summed by kind.
//!
//! Derivation runs once per distinct block at registration, never per voxel, so it
//! favours clarity over raw speed — the hot path reads the precomputed results.
use voxel_engine::Color;

use crate::block::bary::{SparseSpecials, barycenter};
use crate::block::composition::{Composition, Weights};
use crate::block::element::{Core, CoreProperties, ElementRegistry, SpecialKind};

/// Each core property of a block is the weighted average of its elements'. With
/// natural weights of `1` this is the plain mean (so equal parts of `1, 2, 3`
/// derive `2`); with mixture percentages it is the percentage-weighted mean.
pub fn derive_core(els: &ElementRegistry, comp: &Composition) -> CoreProperties {
    derive_core_from(els, &comp.weights())
}

/// [`derive_core`] from a precomputed [`Weights`], so a caller registering a
/// block can reduce the composition once and share it across every derivation.
pub fn derive_core_from(els: &ElementRegistry, weights: &Weights) -> CoreProperties {
    Core::blend(
        weights
            .parts()
            .iter()
            .map(|&(id, weight)| (Core::from(els.get(id).core), weight)),
    )
    .into()
}

/// The block's colour is its element tints averaged by the same weights — a
/// 70/30 soil/clay mix looks 70% soil. Air (no elements) is transparent.
pub fn derive_color(els: &ElementRegistry, comp: &Composition) -> Color {
    derive_color_from(els, &comp.weights())
}

/// [`derive_color`] from a precomputed [`Weights`].
pub fn derive_color_from(els: &ElementRegistry, weights: &Weights) -> Color {
    barycenter(
        weights
            .parts()
            .iter()
            .map(|&(id, weight)| (els.get(id).color, weight)),
    )
}

/// A block is solid unless it has no material in it. Air — the empty natural
/// block — is the sole exception; it is the only thing the renderer culls and the
/// only thing the player walks through.
pub fn derive_solid(comp: &Composition) -> bool {
    !comp.is_empty()
}

/// Whether a block hides the faces behind it — the mesher's cull key (distinct
/// from [`derive_solid`], which is collision's key). A block is opaque when it is
/// solid *and* lets no light through (`transparency == 0`); a translucent solid
/// like glass is solid but NOT opaque, so faces behind it still draw. Air is
/// non-solid, hence non-opaque.
pub fn derive_opaque(core: &CoreProperties, solid: bool) -> bool {
    solid && core.transparency == 0
}

/// A block's blocklight output on the mesher's 0..=15 scale, rescaled from the
/// element `light_emission` (0..=255). Baked once per block; the light BFS seeds
/// from blocks whose value is > 0.
pub fn derive_emission(core: &CoreProperties) -> u8 {
    (core.light_emission as u16 * 15 / 255) as u8
}

/// Special behaviours a block exhibits, each scaled by how much of the carrying
/// element it contains and summed across carriers. Returned sorted by kind for a
/// stable, inspectable order.
pub fn derive_specials(els: &ElementRegistry, comp: &Composition) -> Box<[(SpecialKind, u8)]> {
    derive_specials_from(els, &comp.weights())
}

/// [`derive_specials`] from a precomputed [`Weights`].
pub fn derive_specials_from(els: &ElementRegistry, weights: &Weights) -> Box<[(SpecialKind, u8)]> {
    // Each element contributes its specials (kind + strength) weighted by its
    // share; barycenter merges by kind, sorts, and divides by the total.
    barycenter(weights.parts().iter().map(|&(id, weight)| {
        let specials = els
            .get(id)
            .specials
            .iter()
            .map(|s| (s.kind(), s.strength()))
            .collect();
        (SparseSpecials(specials), weight)
    }))
    .0
    .into()
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
        // Three elements with durabilities 1, 2, 3: equal parts should derive 2.
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
