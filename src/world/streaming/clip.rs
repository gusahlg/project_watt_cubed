//! The far tiles' clips. A finest far tile over near columns stays whole and is meshed without the
//! chunk columns the full-res chunks draw ([`ColumnClip`]). A column joins a tile's clip once the
//! chunks of its shell are final, and leaves it at once when it stops being near. A clip change
//! re-extracts the tile while its old mesh keeps drawing, and the chunks of a column a landed mesh
//! still leaves out keep drawing (and stay loaded) until that mesh is replaced.

use super::*;
use super::drain::Landing;

/// One wanted column bit of a tile and the real columns, with their shells, it waits for.
pub(in crate::world) type ClipBit = (ColumnClip, Vec<(ColumnKey, [i32; 2])>);

/// The far lane's clip state.
#[derive(Default)]
pub(in crate::world) struct Clips {
    /// The clip each tile should draw now.
    target: FastMap<SectionPos, ColumnClip>,
    /// Tiles whose wanted clip still has unsettled columns, with every wanted bit.
    wait: FastMap<SectionPos, Vec<ClipBit>>,
    /// Ready tiles re-extracting in place, by claim token.
    reclip: FastMap<SectionPos, pipeline::ClaimToken>,
    /// Ready tiles extracted with a clip. Entries that left that state are dropped on use.
    clipped: FastSet<SectionPos>,
}

impl World {
    /// A chunk settled: a waiting clip may be done.
    pub(in crate::world) fn note_settled(&mut self) {
        self.clip_recheck.set();
    }

    /// The clip tile `s` should draw now.
    pub(in crate::world) fn clip_target(&self, s: SectionPos) -> ColumnClip {
        self.clips.target.get(&s).copied().unwrap_or_default()
    }

    /// The bits of `bits` whose columns' chunks are all final.
    fn settled_bits(&self, bits: &[ClipBit]) -> ColumnClip {
        bits.iter()
            .filter(|(_, cols)| cols.iter().all(|&(real, shell)| self.column_settled(real, shell)))
            .fold(ColumnClip::NONE, |m, (b, _)| m.union(*b))
    }

    /// Take the clips the selection wants. A tile keeps the part of its clip still wanted and adds
    /// the wanted columns whose chunks are final; the rest wait for them.
    pub(super) fn wait_clips(&mut self, wanted: Vec<(SectionPos, Vec<ClipBit>)>) {
        let prev = std::mem::take(&mut self.clips.target);
        self.clips.wait.clear();
        for (s, bits) in wanted {
            let want = bits.iter().fold(ColumnClip::NONE, |m, (b, _)| m.union(*b));
            let had = prev.get(&s).copied().unwrap_or_default().intersect(want);
            let fresh: Vec<ClipBit> = bits.iter().filter(|(b, _)| !b.is_subset_of(had)).cloned().collect();
            let now = had.union(self.settled_bits(&fresh));
            if now != want {
                self.clips.wait.insert(s, bits);
            }
            if !now.is_empty() {
                self.clips.target.insert(s, now);
            }
        }
        if prev != self.clips.target {
            self.pending_sections.set();
        }
    }

    /// Add the waiting columns whose chunks have settled since.
    pub(super) fn settle_clips(&mut self) {
        if self.clips.wait.is_empty() {
            return;
        }
        let waits: Vec<(SectionPos, Vec<ClipBit>)> = self.clips.wait.iter().map(|(&s, b)| (s, b.clone())).collect();
        let mut grew = false;
        for (s, bits) in waits {
            let want = bits.iter().fold(ColumnClip::NONE, |m, (b, _)| m.union(*b));
            let had = self.clip_target(s);
            let fresh: Vec<ClipBit> = bits.iter().filter(|(b, _)| !b.is_subset_of(had)).cloned().collect();
            let now = had.union(self.settled_bits(&fresh));
            if now != had {
                self.clips.target.insert(s, now);
                grew = true;
            }
            if now == want {
                self.clips.wait.remove(&s);
            }
        }
        if grew {
            self.pending_sections.set();
        }
    }

    /// Whether a clip still waits on chunks.
    pub(super) fn clips_waiting(&self) -> bool {
        !self.clips.wait.is_empty()
    }

    /// Tiles whose clip waits on chunks.
    pub(super) fn clips_waiting_count(&self) -> usize {
        self.clips.wait.len()
    }

    /// The clips still waiting or re-extracting, with what they wait for, for a test's report.
    #[cfg(test)]
    pub(in crate::world) fn clips_debug(&self) -> String {
        let mut out = String::new();
        for (s, t) in &self.clips.reclip {
            let state = match self.sections.get(s) {
                Some(SectionState::Ready { clip, .. }) => format!("Ready asked {:#x} drawn {:#x}", clip.asked.0, clip.drawn.0),
                Some(SectionState::Meshing { token }) => format!("Meshing {token:?}"),
                None => "absent".into(),
            };
            out += &format!("reclip {s:?} {t:?} {state} target {:#x} in cut {}; ", self.clip_target(*s).0, self.in_cut(*s));
        }
        for (s, bits) in self.clips.wait.iter().take(3) {
            let unsettled: Vec<String> = bits
                .iter()
                .flat_map(|(_, cols)| cols.iter())
                .filter(|&&(real, shell)| !self.column_settled(real, shell))
                .take(2)
                .map(|&(real, [lo, hi])| {
                    let bad: Vec<String> = (lo..=hi)
                        .map(|a| real.chunk(a))
                        .filter(|&c| !self.chunk_final(c))
                        .map(|c| format!("{c:?} {:?}", self.chunks.get(&c).map(|l| &l.state)))
                        .collect();
                    format!("{real:?} {lo}..={hi}: {bad:?}")
                })
                .collect();
            out += &format!("wait {s:?}: {unsettled:?}; ");
        }
        out
    }

    /// Every clip settled and landed: nothing waits on chunks or is re-extracting, and every drawn
    /// Ready tile draws its target.
    pub(in crate::world) fn clips_at_target(&self) -> bool {
        self.clips.wait.is_empty()
            && self.clips.reclip.is_empty()
            && self.section_visible.iter().all(|&s| !self.section_ready(s) || !self.needs_reclip(s))
    }

    /// A Ready tile whose mesh is not for its current clip, with no re-extract in flight.
    pub(in crate::world) fn needs_reclip(&self, s: SectionPos) -> bool {
        let Some(SectionState::Ready { clip, .. }) = self.sections.get(&s) else { return false };
        clip.asked != self.clip_target(s) && !self.clips.reclip.contains_key(&s)
    }

    /// Whether `s` draws now or is about to: drawn, or selected.
    pub(in crate::world) fn in_cut(&self, s: SectionPos) -> bool {
        self.section_fade.draws(s) || self.section_desired.binary_search_by_key(&section_key(&s), section_key).is_ok()
    }

    /// Visit the tiles in the cut to re-extract in place.
    pub(in crate::world) fn for_each_reclip(&self, mut visit: impl FnMut(SectionPos)) {
        let target = self.clips.target.keys().filter(|s| !self.clips.clipped.contains(s));
        for &s in self.clips.clipped.iter().chain(target) {
            if self.needs_reclip(s) && self.in_cut(s) {
                visit(s);
            }
        }
    }

    /// Record (`Some`) or drop the in-place re-extract claim on `s`.
    pub(in crate::world) fn claim_reclip(&mut self, s: SectionPos, token: Option<pipeline::ClaimToken>) {
        match token {
            Some(t) => {
                self.clips.reclip.insert(s, t);
            }
            None => {
                self.clips.reclip.remove(&s);
            }
        }
    }

    /// A cancelled or failed job releases the re-extract claim it held.
    pub(in crate::world) fn release_reclip(&mut self, s: SectionPos, token: pipeline::ClaimToken) {
        if self.clips.reclip.get(&s) == Some(&token) {
            self.clips.reclip.remove(&s);
            self.pending_sections.set();
        }
    }

    pub(in crate::world) fn reclips_in_flight(&self) -> bool {
        !self.clips.reclip.is_empty()
    }

    /// Whether `token` is the live claim on `s`: a fresh extract or an in-place one.
    pub(in crate::world) fn section_claim_live(&self, s: SectionPos, token: pipeline::ClaimToken) -> bool {
        matches!(self.sections.get(&s), Some(SectionState::Meshing { token: t }) if *t == token) || self.clips.reclip.get(&s) == Some(&token)
    }

    /// Forget the lane's clips with the lane.
    pub(in crate::world) fn clear_clips(&mut self) {
        self.clips = Clips::default();
    }

    /// Land one finished section job. A fresh mesh installs as a new Ready section; an in-place one
    /// swaps in for the mesh it replaces, which kept drawing until now. A result whose clip leaves
    /// out a column neither its target nor the mesh it replaces does is dropped: a fresh claim is
    /// released, an in-place one is asked again. Returns whether it installed.
    pub(in crate::world) fn land_section(&mut self, up: pipeline::SectionUpload, landing: &mut impl Landing) -> bool {
        let pipeline::SectionUpload { pos, token, clip: asked, meshes, .. } = up;
        let reclip = self.clips.reclip.get(&pos) == Some(&token);
        if reclip {
            self.clips.reclip.remove(&pos);
        }
        let landed = match self.sections.get(&pos) {
            Some(SectionState::Meshing { token: t }) if *t == token => Some(ColumnClip::NONE),
            Some(SectionState::Ready { clip, .. }) if reclip => Some(clip.drawn),
            _ => None,
        };
        let drawn = meshes.clip_drawn();
        if !landed.is_some_and(|l| drawn.is_subset_of(self.clip_target(pos).union(l))) {
            landing.release_section(meshes);
            if landed.is_some() && !reclip {
                self.sections.remove(&pos);
                super::super::adjust_count(&mut self.meshing_sections, true, false);
                self.section_cover_dirty.set();
            }
            self.pending_sections.set();
            return false;
        }
        if !reclip {
            super::super::adjust_count(&mut self.meshing_sections, true, false);
        }
        let state = landing.section(self, pos, asked, meshes);
        if let Some(old) = self.sections.insert(pos, state) {
            landing.free_section(old);
        }
        if !asked.is_empty() {
            self.clips.clipped.insert(pos);
        }
        // A new Ready section moves the covering, and a new clip may let retained chunks go.
        self.pending_sections.set();
        self.section_cover_dirty.set();
        if reclip {
            self.retire_check.set();
        }
        true
    }

    /// The real chunk columns some landed clip leaves out: their chunks keep drawing and stay
    /// loaded until a mesh that draws them replaces it.
    pub(in crate::world) fn landed_columns(&mut self) -> FastSet<ColumnKey> {
        let sections = &self.sections;
        self.clips.clipped.retain(|s| matches!(sections.get(s), Some(SectionState::Ready { clip, .. }) if !clip.asked.is_empty()));
        let Some(up) = self.near.up() else { return FastSet::default() };
        let mut out = FastSet::default();
        for &s in &self.clips.clipped {
            let Some(SectionState::Ready { clip, .. }) = self.sections.get(&s) else { continue };
            for (cx, cz) in clip.drawn.offsets() {
                for c in self.tile_column_chunks(s, cx, cz).unwrap_or_default() {
                    out.insert(ColumnKey::of(up, c).0);
                }
            }
        }
        out
    }
}
