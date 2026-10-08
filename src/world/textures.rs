//! Block texture layers: palette growth appends new layers to the GPU texture array.

use voxel_engine::Engine;

use crate::block::appearance::{fill_layer, BlockAppearance, LAYER_BYTES, TEXTURE_SIZE};

use super::World;

/// The block texture array as last built and sent to the GPU.
pub(super) struct BlockTextures {
    /// Descriptor count last processed by [`World::refresh_textures`].
    built: usize,
    /// Built texture layers by id, kept so palette growth (crafting registers
    /// one block at a time) appends new layers instead of regenerating all.
    /// Cleared when the appearance `revision` (or GPU-descriptor flag) changes.
    cache: Vec<Vec<u8>>,
    /// Layers last sent to the GPU (`set` on first upload, `append` after).
    /// Existing layers never change: a layer is a pure function of the visual
    /// at one revision, and ids are append-only, so growth never re-sends the
    /// prefix unless the appearance revision moved.
    uploaded_len: usize,
    /// Appearance revision last used to fill [`Self::cache`].
    revision: u32,
    /// Device texture-array layer ceiling, stamped into `HotTables::layer_cap`
    /// so the meshers saturate vertex layers at it. Construction uses `u16::MAX`
    /// (identity); the first engine contact overwrites it once.
    pub(super) layer_cap: u16,
    /// True after [`World::pump`] has read `Engine::max_texture_array_layers`.
    cap_from_device: bool,
}

impl BlockTextures {
    /// Nothing built or sent yet.
    pub(super) fn new() -> Self {
        Self { built: 0, cache: Vec::new(), uploaded_len: 0, revision: 0, layer_cap: u16::MAX, cap_from_device: false }
    }
}

impl World {
    /// Rebuild/upload the block texture array when configurations gain layers (world entry, a newly
    /// interned configuration) or the appearance `revision` changes. Existing layers never change at
    /// one revision (a pure function of the configuration; layer ids are append-only), so only the
    /// first upload / a revision rebuild uses `set_block_textures`; later growth appends.
    pub(super) fn refresh_textures(&mut self, eng: &mut Engine, appearance: &dyn BlockAppearance) {
        // Never zero and never past the vertex field's 14 bits. Construction caches `u16::MAX`; the
        // device cap is read once.
        if !self.textures.cap_from_device {
            let device = eng.max_texture_array_layers().clamp(1, u16::MAX as u32) as u16;
            self.textures.layer_cap = device.min(crate::block::MAX_DESCRIPTORS as u16);
            self.textures.cap_from_device = true;
            self.registry.set_descriptor_cap(self.textures.layer_cap);
            #[cfg(test)]
            crate::alloc_count::note_engine(crate::alloc_count::EngineCall::TexLayers);
        }
        let count = self.registry.descriptor_count();
        let rev = appearance.revision();
        if self.textures.revision != rev {
            self.textures.cache.clear();
            self.textures.uploaded_len = 0;
            self.textures.built = 0;
            self.textures.revision = rev;
        }
        if self.textures.built == count {
            return;
        }
        for i in self.textures.cache.len()..count {
            let mut buf = [0u8; LAYER_BYTES];
            fill_layer(appearance, &self.registry, i as u16, &mut buf);
            self.textures.cache.push(buf.to_vec());
        }
        let visible = count.min(self.textures.layer_cap as usize);
        match plan_texture_upload(&self.textures.cache, self.textures.uploaded_len, visible) {
            Some(TextureUpload::Set(layers)) => eng.set_block_textures(TEXTURE_SIZE, layers),
            Some(TextureUpload::Append(layers)) => eng.append_block_textures(layers),
            None => {}
        }
        self.textures.uploaded_len = visible;
        self.textures.built = count;
    }
}

/// GPU write for a palette-growth step. `Set` is the initial bind; `Append`
/// is every later growth (ids above `uploaded_len`, in id order).
#[derive(Debug)]
enum TextureUpload<'a> {
    Set(&'a [Vec<u8>]),
    Append(&'a [Vec<u8>]),
}

/// Layers to send for the current cache vs last uploaded count, clamped to
/// the device layer cap (`visible`). Prefix layers are never rewritten.
fn plan_texture_upload(
    cache: &[Vec<u8>],
    uploaded_len: usize,
    visible: usize,
) -> Option<TextureUpload<'_>> {
    if visible == 0 {
        return None;
    }
    if uploaded_len == 0 {
        return Some(TextureUpload::Set(&cache[..visible]));
    }
    if visible > uploaded_len {
        return Some(TextureUpload::Append(&cache[uploaded_len..visible]));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_growth_appends_only_new_layers_in_id_order() {
        let layer = |id: u8| vec![id; 4];
        let mut cache = vec![layer(0), layer(1), layer(2)];
        let mut uploaded_len = 0usize;
        let cap = 8usize;

        let visible = cache.len().min(cap);
        match plan_texture_upload(&cache, uploaded_len, visible) {
            Some(TextureUpload::Set(layers)) => {
                assert_eq!(layers.len(), 3);
                assert_eq!(layers[0], layer(0));
                assert_eq!(layers[1], layer(1));
                assert_eq!(layers[2], layer(2));
            }
            other => panic!("initial upload must set, got {other:?}"),
        }
        uploaded_len = visible;
        assert_eq!(uploaded_len, 3);

        cache.push(layer(3));
        cache.push(layer(4));
        let visible = cache.len().min(cap);
        match plan_texture_upload(&cache, uploaded_len, visible) {
            Some(TextureUpload::Append(layers)) => {
                assert_eq!(layers, &[layer(3), layer(4)]);
                assert_eq!(
                    uploaded_len + layers.len(),
                    visible,
                    "append is exactly the ids above uploaded_len"
                );
            }
            other => panic!("growth must append, got {other:?}"),
        }
        uploaded_len = visible;
        assert_eq!(uploaded_len, 5);

        let cap = 5usize;
        cache.push(layer(5));
        let visible = cache.len().min(cap);
        assert!(
            plan_texture_upload(&cache, uploaded_len, visible).is_none(),
            "past the layer cap, nothing is re-sent"
        );
        assert_eq!(uploaded_len, 5);
    }

    #[test]
    fn revision_rebuild_resets_to_a_set() {
        let cache = vec![vec![1u8; 4], vec![2; 4], vec![3; 4]];
        match plan_texture_upload(&cache, 0, cache.len()) {
            Some(TextureUpload::Set(layers)) => assert_eq!(layers.len(), 3),
            other => panic!("revision rebuild must set, got {other:?}"),
        }
    }
}
