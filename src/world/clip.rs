//! What the renderer is told each frame, and what a far tile reads of the near field: whether
//! a chunk is final, and a section's baked relief and material.

use super::*;
use super::resident::FLAT_DETAIL;

impl World {
    /// Draw the meshed chunks. All per-voxel work happened when each chunk was
    /// built; a frame is one `draw_mesh` per chunk (the engine frustum-culls
    /// each against its offset AABB internally).
    ///
    /// Terrain chunks and LOD sections are RESIDENT meshes: placement and detail
    /// pinned at upload, visibility a `set_visible` mask, style a `set_style`
    /// push (both maintained in `stream`). The engine draws every visible
    /// resident mesh itself, so `render` submits no per-mesh draws. Far tiles leave the near
    /// columns out of their meshes, so the engine's LOD clip box clips nothing.
    pub fn render(&self, f: &mut Frame3D, _cam: DVec3) {
        f.set_lod_clip_box(Vec3::ZERO, Vec3::ZERO);
        f.set_lod_morph(&self.lod_morph);
        // Set-size gauge: a spike localizes a regression to a grown set (view
        // volume / section frontier). See `profile::Gauge`.
        use voxel_engine::profile::{Gauge, gauge};
        gauge(Gauge::WorldChunks, self.chunks.len() as u64);
    }

    /// Whether chunk `c` is final for the far field's handover: settled (drawn, or nothing to
    /// draw), or parked by quarantine, a bounded hole that would otherwise hold the far field
    /// over it for good.
    pub(in crate::world) fn chunk_final(&self, c: Coord) -> bool {
        if self.chunks.get(&c).is_some_and(|l| l.state.settled()) {
            return true;
        }
        if self.quarantined.is_empty() {
            return false;
        }
        self.quarantined.contains(&streaming::FailKey::Mesh { coord: c }) || self.generate_quarantined(c)
    }

    /// Far-material style: flat palette-average past [`FLAT_DETAIL`] if available,
    /// else textured. Returns `mode` bit and packed sRGB colour.
    pub(super) fn section_material(&self, pos: SectionPos) -> (bool, u32) {
        if pos.detail < FLAT_DETAIL {
            return (false, 0);
        }
        let color = self.section_overlay.get(&pos).map(|c| c.color).or_else(|| self.baked_mip(pos)?.color(pos));
        match color {
            Some(c) => (
                true,
                c.r as u32 | (c.g as u32) << 8 | (c.b as u32) << 16 | (c.a as u32) << 24,
            ),
            None => (false, 0),
        }
    }

    /// Current vertical relief for one section footprint. Live edit-derived
    /// data takes precedence over the immutable background bake.
    pub(super) fn section_relief_band(&self, key: SectionPos) -> Option<(f32, f32)> {
        self.section_overlay.get(&key).map(|cell| (cell.lo, cell.hi)).or_else(|| self.baked_mip(key)?.relief_band(key))
    }

    /// The relief bake, when it was baked for `key`'s body and face (its cells are keyed by detail
    /// and position alone).
    fn baked_mip(&self, key: SectionPos) -> Option<&super::heightmip::HeightMip> {
        let (body, face, _, _) = self.section_mip_anchor?;
        (key.body == body && key.face == face).then_some(self.section_mip.as_ref()?)
    }

    /// Geometric altitude of one baked height. +Y with a zero datum stores world Y;
    /// every other bake stores height above the face datum.
    pub(super) fn baked_world_y(&self, key: SectionPos, h: f32) -> i32 {
        let datum = self.generator.face_datum(key.body, key.face);
        let rel = h.floor() as i32;
        if key.face == Face::PosY && datum == 0 { rel } else { datum.saturating_add(rel) }
    }
}
