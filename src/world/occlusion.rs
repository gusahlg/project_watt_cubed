//! The occlusion gate and the visible-set rebuild.

use super::*;

impl World {
    /// Whether the occlusion gate is active this frame — the adaptive decision
    /// to spend CPU culling in order to save GPU draw time. Occlusion only pays
    /// off when the frame is GPU/overdraw-bound; that signal lives in the engine
    /// (GPU timestamps) and isn't wired yet, so this is off unless force-enabled
    /// via [`RenderConfig::occlusion`](crate::render_config::RenderConfig). When the
    /// engine exposes the signal, OR it in here and every GPU-side optimisation can
    /// share this one gate.
    pub(super) fn occlusion_enabled(&self) -> bool {
        self.occlusion_forced
    }

    /// Rebuild the occlusion visible set: lazily fill any missing per-chunk
    /// connectivity (only floods chunks not yet classified — so a world that
    /// never activates occlusion never pays it), then BFS from the camera's
    /// chunk, then patch each drawable chunk's GPU visibility mask to its
    /// occlusion bit (`apply_occlusion_masks`). Nothing filters at draw time.
    ///
    /// Fill and BFS are decoupled: a partial fill does not force a rebuild, so
    /// a load flood spends the lane budget classifying chunks and rebuilds once
    /// the queue drains (or immediately on a root move / edit). Unclassified
    /// chunks stay OPEN in the BFS — over-draw, never a hole.
    /// `eng` is `None` only in headless tests: the masks then touch no GPU.
    pub(in crate::world) fn rebuild_occlusion(
        &mut self,
        mut eng: Option<&mut Engine>,
        budget: Budget,
    ) -> Progress {
        let on = self.occlusion_enabled();
        let was_active = self.occlusion_active;
        self.occlusion_active = on;
        if !on {
            if was_active {
                self.reveal_all(eng.as_deref_mut());
            }
            return Progress::Idle;
        }
        // Immediate: root moved / edit / activation — staleness can hide a
        // visible chunk. Topology (loads/unloads) is over-draw only, so it
        // waits out the debounce. Compute the clock only when the flag is up.
        let immediate = self.occlusion_dirty.take() || !was_active;
        let topo = self.occlusion_topo_dirty.get();
        let want_rebuild = immediate
            || (topo
                && self
                    .last_occlusion_rebuild
                    .is_none_or(|t| t.elapsed() >= OCCLUSION_DEBOUNCE));

        if !was_active {
            self.conn_fill_queue.clear();
            let missing = self
                .chunks
                .iter()
                .filter(|(_, l)| l.connectivity.is_none())
                .map(|(&c, _)| c);
            self.conn_fill_queue.extend(missing);
        }

        let had_fill = !self.conn_fill_queue.is_empty();
        if had_fill {
            let ms = match budget {
                Budget::Millis(ms) => ms,
                _ => 0.5,
            };
            let deadline = pipeline::Deadline::from_budget(
                self.stream_pacer
                    .duration(Duration::from_secs_f32(ms / 1000.0)),
            );
            let registry = &self.registry;
            let mut filled = 0usize;
            while let Some(coord) = self.conn_fill_queue.pop_front() {
                let Some(loaded) = self.chunks.get_mut(&coord) else {
                    continue;
                };
                if loaded.connectivity.is_some() {
                    continue;
                }
                // Water/glass are solid but see-through: they must not seal.
                loaded.connectivity = Some(Connectivity::compute(&loaded.chunk, |id| {
                    registry.is_opaque(id)
                }));
                filled += 1;
                if filled >= OCCLUSION_FILL_FLOOR && deadline.expired() {
                    break;
                }
            }
        }
        let fill_remaining = !self.conn_fill_queue.is_empty();
        let became_empty = had_fill && !fill_remaining;
        if !(want_rebuild || became_empty) {
            return if fill_remaining {
                Progress::Partial {
                    remaining: self.conn_fill_queue.len() as u32,
                }
            } else {
                Progress::Idle
            };
        }
        self.occlusion_topo_dirty.take();
        self.last_occlusion_rebuild = Some(crate::sched::now());
        let Some(origin) = self.center else {
            return Progress::Idle;
        };
        let volume = self.unload_box(origin);
        let loaded = self
            .chunks
            .iter()
            .map(|(&c, l)| (c, l.connectivity.unwrap_or(Connectivity::OPEN)));
        self.occlusion.rebuild(volume, origin, loaded);
        self.apply_occlusion_masks(eng.as_deref_mut());
        if fill_remaining {
            Progress::Partial {
                remaining: self.conn_fill_queue.len() as u32,
            }
        } else {
            Progress::Idle
        }
    }

    /// Patch drawable meshes whose occlusion bit changed since the last push.
    fn apply_occlusion_masks(&mut self, mut eng: Option<&mut Engine>) {
        for (&coord, loaded) in self.chunks.iter_mut() {
            let vis = self.occlusion.is_visible(coord);
            if vis == loaded.visible {
                continue;
            }
            loaded.visible = vis;
            if let (Some(eng), Some(meshes)) = (eng.as_deref_mut(), loaded.state.live_meshes()) {
                meshes.set_visible(eng, vis);
            }
        }
    }

    /// Reveal every drawable chunk (set its mask visible) — the one-shot restore
    /// when the occlusion gate turns off, since only occlusion ever hides a
    /// resident chunk mesh.
    fn reveal_all(&mut self, mut eng: Option<&mut Engine>) {
        for loaded in self.chunks.values_mut() {
            if loaded.visible {
                continue;
            }
            loaded.visible = true;
            if let (Some(eng), Some(meshes)) = (eng.as_deref_mut(), loaded.state.live_meshes()) {
                meshes.set_visible(eng, true);
            }
        }
    }
}
