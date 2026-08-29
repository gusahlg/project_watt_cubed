//! Computes the observable properties of a block from its [`Composition`]. Each
//! core property blends from its contributing elements, the colour is blended
//! similarly, and special behaviours are summed.
//!
//! All computation happens once per distinct block at registration, never per voxel.
//! The results are cached so rendering and physics read precomputed values.
use voxel_engine::{Color, Pass};

use crate::block::bary::{SparseSpecials, barycenter};
use crate::block::composition::{Composition, Weights};
use crate::block::element::{Core, CoreProperties, ElementRegistry, SpecialKind};

/// Blend each core property from its contributing elements, using their weights.
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

/// Blend the block's colour from element tints using their weights.
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
/// block — is the sole exception. Rendering and mining use this material-presence
/// property; movement separately exempts liquids.
pub fn derive_solid(comp: &Composition) -> bool {
    !comp.is_empty()
}

/// Whether a block hides the faces behind it — the mesher's cull key (distinct
/// from [`derive_solid`], which is the material-presence key). A block is opaque
/// when it is solid *and* lets no light through (`transparency == 0`); a translucent solid
/// like glass is solid but NOT opaque, so faces behind it still draw. Air is
/// non-solid, hence non-opaque.
pub fn derive_opaque(core: &CoreProperties, solid: bool) -> bool {
    solid && core.transparency == 0
}

/// The draw technique a block routes to — a pure function of the same
/// `transparency` that drives [`derive_opaque`], so `Opaque` iff `derive_opaque`.
/// A solid that lets light through is tinted see-through (water/glass) → [`Pass::Blend`].
/// [`Pass::Cutout`] is reserved for atlas blocks with binary alpha holes; none
/// source per-texel alpha yet, so nothing derives it. Air routes to `Opaque`
/// (it is never meshed, so the slot is inert).
pub fn derive_layer(core: &CoreProperties, solid: bool) -> Pass {
    if !solid || core.transparency == 0 {
        Pass::Opaque
    } else {
        Pass::Blend
    }
}

/// The texel alpha stamped into a block's texture layer, from the same
/// `transparency` (0..=100 % of light let through) that drives [`derive_layer`].
/// Opaque blocks (`transparency == 0`) get a fully opaque 255; since they draw on
/// the blend-disabled pipeline, only [`Pass::Blend`] blocks ever composite with
/// this, so the two derivations agree by construction: `alpha < 255` iff `Blend`.
/// A floor keeps a very clear block reading as glass rather than vanishing.
pub fn derive_texel_alpha(core: &CoreProperties) -> u8 {
    /// ~16 % — the minimum opacity a translucent block renders at.
    const MIN_ALPHA: u8 = 40;
    let t = core.transparency.min(100);
    if t == 0 {
        return 255;
    }
    (((100 - t) as u16 * 255 / 100) as u8).max(MIN_ALPHA)
}

/// A block's blocklight output on the mesher's 0..=15 scale, rescaled from the
/// element `light_emission` (0..=255). Baked once per block; the light BFS seeds
/// from blocks whose value is > 0.
pub fn derive_emission(core: &CoreProperties) -> u8 {
    (core.light_emission as u16 * 15 / 255) as u8
}

/// The acoustic material class of a block — the single partition that drives BOTH
/// sound-cue naming (`break_<class>`/`place_<class>`/`step_<class>`) and the
/// occlusion DDA's per-cell absorption weight, so the two can never drift. Ordered
/// hardest→softest; `Open` covers non-solids and property-derived liquids. A
/// mineable liquid uses the liquid cue stem (or its catalog fallback) while
/// remaining acoustically open.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SoundClass {
    Stone,
    Soil,
    Wood,
    Glass,
    Foliage,
    Open,
}

impl SoundClass {
    /// The cue-name stem this class resolves to in the block-cue table.
    pub fn as_str(self) -> &'static str {
        match self {
            SoundClass::Stone => "stone",
            SoundClass::Soil => "soil",
            SoundClass::Wood => "wood",
            SoundClass::Glass => "glass",
            SoundClass::Foliage => "foliage",
            // The checked-in cue catalog retains its historical `water` stem.
            SoundClass::Open => "water",
        }
    }

    /// Acoustic absorption per metre (`0..=255`) — how strongly sound is attenuated
    /// crossing one voxel (the DDA's per-cell weight). `Open` is acoustically
    /// transparent (`0`); values are distinct per class so a block's class
    /// and its absorption are mutually recoverable.
    pub fn absorption(self) -> u8 {
        match self {
            SoundClass::Stone => 200,   // dense stone / metal
            SoundClass::Soil => 140,    // soil / packed earth / sand
            SoundClass::Wood => 90,     // wood / organic aggregates / coal
            SoundClass::Glass => 60,    // glass / ice — transmit sound like light
            SoundClass::Foliage => 30,  // foliage / snow / very light matter
            SoundClass::Open => 0,      // occlusion is a property of walls
        }
    }
}

/// Classify a block acoustically from the same derived physics the rest of the
/// block uses — density is the dominant proxy for sound absorption, with translucent
/// solids (glass/ice) carved out low. Non-solids and passable liquids are `Open`;
/// the acoustics kernel handles the listener's liquid medium separately.
pub fn derive_sound_class(core: &CoreProperties, solid: bool) -> SoundClass {
    if !solid || core.buoyancy > 0 {
        return SoundClass::Open;
    }
    if core.transparency >= 50 {
        return SoundClass::Glass;
    }
    match core.density {
        170.. => SoundClass::Stone,
        90..=169 => SoundClass::Soil,
        50..=89 => SoundClass::Wood,
        _ => SoundClass::Foliage,
    }
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
    fn texel_alpha_agrees_with_layer_and_is_monotonic() {
        let core = |t: u8| CoreProperties { transparency: t, ..Default::default() };
        // Opaque iff transparency == 0 iff alpha == 255 — the shared invariant.
        assert_eq!(derive_texel_alpha(&core(0)), 255);
        assert_eq!(derive_layer(&core(0), true), Pass::Opaque);
        for t in 1..=100u8 {
            let a = derive_texel_alpha(&core(t));
            assert!(a < 255, "translucent block must not be fully opaque (t={t})");
            assert_eq!(derive_layer(&core(t), true), Pass::Blend);
        }
        // More transparent → lower alpha (weakly), floored so glass stays visible.
        assert!(derive_texel_alpha(&core(30)) > derive_texel_alpha(&core(90)));
        assert!(derive_texel_alpha(&core(100)) >= 40, "floor keeps clear blocks perceptible");
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
