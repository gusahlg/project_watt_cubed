//! Block appearance seam: a configuration becomes one texture-array layer.
//!
//! Every configuration owns a layer while the device has layers (see
//! [`BlockRegistry::render_layer`]), so an appearance mod paints a texture *per configuration* from
//! the configuration itself — its elements, readings and presentation colours. The core fallback is
//! [`FlatAppearance`] (flat base colour + alpha). The world's texture cache asks
//! [`Mods::appearance`](crate::modding::Mods::appearance) — the first enabled appearance mod, else this
//! fallback. A `revision` change rebuilds every layer; otherwise the cache is append-only.

use material::{Block, Law, Observation, Visual};

use crate::block::registry::BlockRegistry;

/// Edge length of an appearance layer, in texels.
pub const TEXTURE_SIZE: u32 = 32;

/// RGBA8 bytes in one [`TEXTURE_SIZE`] layer.
pub const LAYER_BYTES: usize = (TEXTURE_SIZE * TEXTURE_SIZE * 4) as usize;

/// Everything an appearance may read about the configuration it paints.
pub struct AppearanceSource<'a> {
    /// The world's law.
    pub law: &'a Law,
    /// The configuration's kernel record (occurrences + cached supports).
    pub block: &'a Block,
    /// Its presentation colours and parameters.
    pub visual: &'a Visual,
    /// Its readings.
    pub obs: &'a Observation,
}

/// Turns a configuration into pixels. Implementors live in mods; core only ships
/// [`FlatAppearance`]. `layer` must be a pure function of the source (and the mod's own knobs).
pub trait BlockAppearance {
    fn layer(&self, src: &AppearanceSource, out: &mut [u8; LAYER_BYTES]);
    fn revision(&self) -> u32;
}

/// Core fallback: every texel is the base colour with the visual's alpha.
pub struct FlatAppearance;

/// Process-lifetime fallback when no appearance mod is enabled.
pub static FLAT: FlatAppearance = FlatAppearance;

impl BlockAppearance for FlatAppearance {
    fn layer(&self, src: &AppearanceSource, out: &mut [u8; LAYER_BYTES]) {
        let vis = src.visual;
        for texel in out.chunks_exact_mut(4) {
            texel.copy_from_slice(&[vis.rgb[0], vis.rgb[1], vis.rgb[2], vis.alpha]);
        }
    }

    fn revision(&self) -> u32 {
        0
    }
}

/// Fill one layer. Layer 0 is all white — the engine's immediate cubes and wires always sample it.
pub fn fill_layer(
    appearance: &dyn BlockAppearance,
    registry: &BlockRegistry,
    layer: u16,
    out: &mut [u8; LAYER_BYTES],
) {
    if layer == 0 {
        out.fill(255);
        return;
    }
    let id = registry.layer_source(layer);
    let visual = registry.visual(id);
    let obs = registry.observation(id);
    let src = AppearanceSource { law: registry.law(), block: registry.block(id), visual: &visual, obs: &obs };
    appearance.layer(&src, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modding::Mods;
    use material::{Configuration, Element};

    #[test]
    fn flat_fills_the_base_colour_with_alpha() {
        let mut r = BlockRegistry::with_builtins();
        let id = r.intern(&Configuration::single(Element::new([10, 20, 30, 40]))).unwrap();
        let mut out = [0u8; LAYER_BYTES];
        fill_layer(&FLAT, &r, r.render_layer(id), &mut out);
        let v = r.visual(id);
        assert!(out.chunks_exact(4).all(|t| t == [v.rgb[0], v.rgb[1], v.rgb[2], v.alpha]));
        assert_eq!(FLAT.revision(), 0);
    }

    #[test]
    fn empty_mods_use_flat_appearance_and_layer_zero_is_white() {
        let mods = Mods::empty();
        assert_eq!(mods.appearance().revision(), 0);
        let reg = BlockRegistry::with_builtins();
        let mut out = [0u8; LAYER_BYTES];
        fill_layer(mods.appearance(), &reg, 0, &mut out);
        assert!(out.iter().all(|&b| b == 255));
    }
}
