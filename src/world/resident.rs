//! Resident state: loaded chunks and their mesh lifecycle, the GPU meshes they own, sections,
//! and the slot counts the far lane budgets against.

use super::*;

/// A loaded chunk: its voxel data plus the [`MeshState`] tracking the GPU mesh
/// built from it. The state captures both the lifecycle stage and the handle
/// ownership: a born-air chunk is `Air` (nothing drawn, no worker job), a
/// drawable one is `Ready(handle)`, and so on — see [`MeshState`].
pub(super) struct Loaded {
    /// Shared voxel storage by refcount; edits via `Arc::make_mut`.
    pub(super) chunk: Arc<Chunk>,
    pub(super) state: MeshState,
    /// Mesh-input revision: bumped whenever this chunk's mesh inputs change —
    /// a direct edit, or an edit on a neighbour's touching border (which flips
    /// this chunk's exposed faces). A worker mesh result carries the rev its
    /// snapshot was taken at; a result whose rev no longer matches is stale
    /// and dropped (the chunk is `Dirty` or gets re-scanned anyway).
    pub(super) rev: u32,
    /// Which of this chunk's faces a sightline can pass between, for the
    /// occlusion BFS. `None` until first needed and after a direct edit — it is
    /// computed **lazily**, only when the occlusion gate is active, so a world
    /// that never runs occlusion (the common, CPU-bound case) never pays the
    /// flood-fill. Depends only on the chunk's own voxels, so a neighbour edit
    /// (which bumps `rev`) leaves it valid.
    pub(super) connectivity: Option<Connectivity>,
    /// Visibility last pushed to the engine for this chunk's meshes. Engine
    /// slots birth visible; default matches that so a first hide is a real delta.
    pub(super) visible: bool,
    /// The chunk's settled light grid, published ([`settle_light`](World::settle_light))
    /// either analytically (the trivial fast path) or when a worker-pool flood
    /// lands (the flood runs off-thread, decoupled from meshing). Read
    /// as part of the neighbour shell ([`light::PaddedLight`]) when an adjacent
    /// chunk meshes or settles, and directly queryable for gameplay (mob spawns,
    /// plant growth) with no mesh. `None` until the chunk has first settled.
    pub(super) light: Option<LightGrid>,
    /// Any border lumel has blocklight `> 1`. Set at [`settle_light`](World::settle_light)
    /// so a uniform-air neighbour can reject the analytic sky path by testing
    /// six booleans instead of capturing a 3 KB face shell.
    pub(super) has_blocklight: bool,
    /// Neighbour face moved while this chunk's flood was in flight. Re-seed
    /// when the result integrates so the wave costs at most one extra flood.
    pub(super) light_reseed: bool,
    /// Identity of this `Loaded` for light-claim matching. Bumped at store so
    /// a `Done::Light` captured against a previous resident at the same coord
    /// (unload then regenerate, same `light_epoch`) cannot publish onto the
    /// new voxels.
    pub(super) light_gen: u32,
    /// Content hash of the last *sync* remesh. Async uploads store `None` so
    /// they never hash on the main thread; an identical edit remesh then uploads.
    pub(super) mesh_hash: Option<u64>,
}

/// Keep an in-flight claim counter in step with a boolean flag, without
/// borrowing the rest of `World` (so it can run while a chunk/section is held).
pub(in crate::world) fn adjust_count(count: &mut usize, was: bool, now: bool) {
    match (was, now) {
        (false, true) => *count += 1,
        (true, false) => *count = count.saturating_sub(1),
        _ => {}
    }
}

impl Loaded {
    /// Transition to `next`, freeing the mesh this chunk was drawing unless
    /// that mesh is carried into `next`. This is the single place that frees a
    /// chunk's mesh: every GPU-freeing transition — unload, radius shrink, sync
    /// remesh, world-leave — routes through here, so there's one place to check
    /// for double frees or leaks.
    pub(super) fn retire(&mut self, next: MeshState, eng: &mut Engine) {
        let carrying = next.live_meshes().is_some();
        std::mem::replace(&mut self.state, next).free_owned(eng);
        if !carrying {
            self.mesh_hash = None;
        }
    }
    /// Engine-free retire for claim tests that count frees through the hook.
    #[cfg(test)]
    pub(super) fn retire_logged(&mut self, next: MeshState) {
        let carrying = next.live_meshes().is_some();
        std::mem::replace(&mut self.state, next).free_logged();
        if !carrying {
            self.mesh_hash = None;
        }
    }
}

/// Sole owner of one GPU mesh allocation. Deliberately NOT `Copy`/`Clone`:
/// holding an `OwnedMesh` *is* holding the live handle, so the only ways to
/// dispose of it are [`free`](OwnedMesh::free) (which releases the GPU
/// allocation) or moving it into another [`MeshState`] (carrying the same mesh
/// forward). `free` consumes `self`, so the same allocation can't be released
/// twice, and the handle can't be silently overwritten and leaked.
#[derive(Debug, PartialEq, Eq)]
pub(in crate::world) struct OwnedMesh(MeshHandle);

impl OwnedMesh {
    /// Wrap a freshly uploaded handle as its sole owner.
    fn new(handle: MeshHandle) -> Self {
        Self(handle)
    }
    /// The `Copy` handle id, for recording a draw — a borrow, so ownership stays put.
    fn id(&self) -> MeshHandle {
        self.0
    }
    /// Release the GPU allocation. Consumes `self`, so it can't be double-freed.
    fn free(self, eng: &mut Engine) {
        #[cfg(test)]
        mesh_free_log::record(self.0);
        eng.free_mesh(self.0);
    }
}

/// Test-only log of handles passing through [`OwnedMesh::free`] / the
/// engine-free `free_logged` path, so claim tests can count each free once
/// without wrapping [`Engine`].
#[cfg(test)]
pub(in crate::world) mod mesh_free_log {
    use std::cell::RefCell;
    use voxel_engine::MeshHandle;

    thread_local! {
        static FREED: RefCell<Vec<MeshHandle>> = RefCell::new(Vec::new());
    }

    pub fn record(h: MeshHandle) {
        FREED.with(|f| f.borrow_mut().push(h));
    }

    pub fn take() -> Vec<MeshHandle> {
        FREED.with(|f| std::mem::take(&mut *f.borrow_mut()))
    }
}

/// Test-only log of [`ChunkMeshes::set_visible`] calls, so visibility tests
/// can record a fake engine without constructing one.
#[cfg(test)]
pub(in crate::world) mod vis_log {
    use std::cell::RefCell;
    use voxel_engine::MeshHandle;

    thread_local! {
        static CALLS: RefCell<Vec<(MeshHandle, bool)>> = RefCell::new(Vec::new());
    }

    pub fn record(handles: impl IntoIterator<Item = MeshHandle>, on: bool) {
        CALLS.with(|c| {
            for h in handles {
                c.borrow_mut().push((h, on));
            }
        });
    }

    pub fn take() -> Vec<(MeshHandle, bool)> {
        CALLS.with(|c| std::mem::take(&mut *c.borrow_mut()))
    }
}

/// The GPU meshes of one chunk, split by draw pass — the resident dual of
/// [`ChunkMeshData`]. A chunk yields an opaque mesh, a transparent mesh, or both;
/// the smart constructor enforces **at least one present** (an all-empty chunk is
/// [`MeshState::Air`], never a `Ready` with nothing to draw). Freeing frees both,
/// so this stays the sole GPU-free chokepoint, now over a `ByPass`.
#[derive(Debug, PartialEq, Eq)]
pub(in crate::world) struct ChunkMeshes(ByPass<Option<OwnedMesh>>);

impl ChunkMeshes {
    /// `None` when no pass has a mesh (the caller uses [`MeshState::Air`]).
    fn new(passes: ByPass<Option<OwnedMesh>>) -> Option<Self> {
        let any = passes.iter().any(|(_, m)| m.is_some());
        any.then_some(Self(passes))
    }
    /// Wrap freshly uploaded per-pass handles, same "at least 1 present" rule as [`Self::new`].
    pub(in crate::world) fn from_upload_handles(
        handles: ByPass<Option<MeshHandle>>,
    ) -> Option<Self> {
        Self::new(ByPass::from_fn(|p| handles[p].map(OwnedMesh::new)))
    }
    /// Push far-material style onto each present pass's resident mesh (LOD
    /// sections). Placement and detail are pinned at upload; the engine delta-gates
    /// identical style. Full-res chunks never call it (default textured style).
    fn set_style(&self, eng: &mut Engine, style: FadeStyle, flat_rgba: u32) {
        for (_, m) in self.0.iter() {
            if let Some(mesh) = m {
                eng.set_mesh_style(mesh.id(), style, flat_rgba);
            }
        }
    }
    /// Free every present pass's GPU allocation. Consumes `self`.
    fn free(self, eng: &mut Engine) {
        for (_, m) in self.0.into_iter_passes() {
            if let Some(mesh) = m {
                mesh.free(eng);
            }
        }
    }
    /// Patch every present pass's slot in the GPU visibility mask.
    pub(super) fn set_visible(&self, eng: &mut Engine, on: bool) {
        for (_, m) in self.0.iter() {
            if let Some(mesh) = m {
                eng.set_visible(mesh.id(), on);
            }
        }
    }
    fn slot_count(&self) -> usize {
        self.0.iter().filter(|(_, m)| m.is_some()).count()
    }
    /// Whether any pass draws `handle` — for the render/ownership tests.
    #[cfg(test)]
    pub(super) fn draws(&self, handle: MeshHandle) -> bool {
        self.0
            .iter()
            .any(|(_, m)| m.as_ref().is_some_and(|o| o.id() == handle))
    }
    /// Live handle ids, for the one-free-per-handle claim tests.
    #[cfg(test)]
    pub(super) fn handles(&self) -> Vec<MeshHandle> {
        self.0
            .iter()
            .filter_map(|(_, m)| m.as_ref().map(|o| o.id()))
            .collect()
    }
}

// Per-draw style (flat palette-average colour) is the typed engine `FadeStyle`
// now — no raw mode bits.
/// Detail threshold for flat palette-average colour (far-material optimization).
/// The outer rings lose per-texel detail to sub-pixel shimmer, so average colour
/// reduces bandwidth; nearer rings keep texture. Inert until the mip bake lands.
pub(super) const FLAT_DETAIL: Detail = Detail(section::FINEST_DETAIL.0 + 4);

/// A section is either meshing or ready. Blocks are pre-positioned at upload,
/// so draw needs only camera-relative offset arithmetic.
pub(in crate::world) enum SectionState {
    /// Claimed by an in-flight worker job carrying this unique token (see
    /// [`pipeline::ClaimToken`]). A result, cancellation, or failure that
    /// presents a different token belongs to a superseded claim and must not
    /// touch this entry.
    Meshing { token: pipeline::ClaimToken },
    /// One mesh per pass for each of the section's stacked slabs (none for an
    /// empty section). Positions and packed detail (`pos.detail + shift`) are
    /// pinned at upload; visibility is one `set_visible` over all of them
    /// (partial covering draws the whole tile — overlap is depth-biased) and
    /// style a `set_style` push.
    Ready {
        meshes: Vec<ChunkMeshes>,
        /// Cages bending a chart section's slabs. Empty on a cube face. Freed with the meshes.
        cages: Vec<voxel_engine::CageHandle>,
        /// Last `(style, flat_rgba)` pushed via [`Self::push_style`], so a value
        /// re-observed next frame (the steady case) sends nothing.
        last_style: Option<(FadeStyle, u32)>,
    },
}

/// The atlas patch a chart section is bent through.
pub(in crate::world) struct ChartBend {
    pub(super) atlas: Arc<crate::space::atlas::Atlas>,
    pub(super) patch: crate::space::atlas::Patch,
}

impl SectionState {
    /// Upload each slab's mesh per pass at its packed origin and the section's detail.
    fn from_upload(
        pos: SectionPos,
        mesh: SectionMeshData,
        eng: &mut Engine,
        bend: Option<&ChartBend>,
    ) -> SectionState {
        let detail = Detail(pos.detail.0.saturating_add(mesh.shift as i8));
        let mut cages = Vec::new();
        let meshes = mesh
            .slabs
            .iter()
            .filter_map(|slab| {
                let placement = Self::slab_place(pos, slab.origin_y, detail, mesh.altitude_floor, bend, eng, &mut cages)?;
                match ChunkMeshes::from_upload_handles(ByPass::from_fn(|p| eng.upload_mesh_placed(&slab.data[p], placement))) {
                    Some(m) => Some(m),
                    None => {
                        Self::drop_failed_cage(eng, &mut cages, placement);
                        None
                    }
                }
            })
            .collect();
        SectionState::Ready { meshes, cages, last_style: None }
    }

    /// Flat placement, or a cage through the slab's eight embedded corners. `None` skips the slab
    /// (an uncaged chart mesh would draw at the origin).
    fn slab_place(
        pos: SectionPos,
        origin_y: u32,
        detail: Detail,
        floor_a: i32,
        bend: Option<&ChartBend>,
        eng: &mut Engine,
        cages: &mut Vec<voxel_engine::CageHandle>,
    ) -> Option<voxel_engine::MeshPlacement> {
        if let Some(bend) = bend
            && bend.atlas.grid.is_some()
        {
            let extent = (1i32 << detail.0).saturating_mul(16);
            let y0 = floor_a + origin_y as i32 * pos.cell_size();
            let (anchor, corners) = section::warp_slab_corners(&bend.atlas, pos.face, pos.min_x(), y0, pos.min_z(), extent)?;
            let cage = eng.create_cage(anchor, corners)?;
            cages.push(cage);
            return Some(voxel_engine::MeshPlacement::caged(cage, detail));
        }
        if pos.body < section::CHART_BODY_BASE {
            return Some(Self::slab_placement(pos, origin_y, detail, floor_a));
        }
        let bend = bend?;
        let extent = (1i32 << detail.0).saturating_mul(16);
        let y0 = floor_a + origin_y as i32 * pos.cell_size();
        let (anchor, corners) = section::chart_slab_corners(&bend.atlas, bend.patch, pos.min_x(), y0, pos.min_z(), extent)?;
        let cage = eng.create_cage(anchor, corners)?;
        cages.push(cage);
        Some(voxel_engine::MeshPlacement::caged(cage, detail))
    }

    fn drop_failed_cage(eng: &mut Engine, cages: &mut Vec<voxel_engine::CageHandle>, placement: voxel_engine::MeshPlacement) {
        if let Some(c) = placement.cage {
            if cages.last() == Some(&c) {
                cages.pop();
            }
            eng.free_cage(c);
        }
    }

    pub(super) fn slab_placement(pos: SectionPos, origin_y: u32, detail: Detail, floor_a: i32) -> voxel_engine::MeshPlacement {
        // PosY on the legacy window: the placement used before face frames existed.
        if pos.face == Face::PosY && floor_a == 0 {
            return voxel_engine::MeshPlacement::terrain(
                voxel_engine::IVec3::new(pos.min_x(), origin_y as i32 * pos.cell_size(), pos.min_z()),
                detail,
            );
        }
        // Rotate the block about its centre, then place vertex (0,0,0) at the
        // world cell that corner lands on. `half` is half a packed block.
        let scale = 1i32 << detail.0;
        let half = 8 * scale;
        let a0 = floor_a + origin_y as i32 * pos.cell_size();
        let (cx, cy, cz) = FaceFrame::new(pos.face).cell_to_world((pos.min_x() + half, a0 + half, pos.min_z() + half));
        voxel_engine::MeshPlacement::terrain(voxel_engine::IVec3::new(cx - half, cy - half, cz - half), detail)
    }

    pub(super) fn from_upload_payload(
        pos: SectionPos,
        meshes: pipeline::SectionPayload,
        eng: &mut Engine,
        bend: Option<&ChartBend>,
    ) -> SectionState {
        match meshes {
            pipeline::SectionPayload::Cpu(data) => Self::from_upload(pos, *data, eng, bend),
            pipeline::SectionPayload::Staged(staged) => Self::from_upload_staged(pos, *staged, eng, bend),
        }
    }

    fn from_upload_staged(
        pos: SectionPos,
        staged: pipeline::StagedSection,
        eng: &mut Engine,
        bend: Option<&ChartBend>,
    ) -> SectionState {
        let detail = Detail(pos.detail.0.saturating_add(staged.shift as i8));
        let mut cages = Vec::new();
        let meshes = staged
            .slabs
            .into_iter()
            .filter_map(|mut slab| {
                let placement = Self::slab_place(pos, slab.origin_y, detail, staged.altitude_floor, bend, eng, &mut cages)?;
                match ChunkMeshes::from_upload_handles(ByPass::from_fn(|p| {
                    slab.passes[p]
                        .take()
                        .and_then(|pass| eng.upload_mesh_staged(pass.staging, pass.quad_counts, p, placement))
                })) {
                    Some(m) => Some(m),
                    None => {
                        Self::drop_failed_cage(eng, &mut cages, placement);
                        None
                    }
                }
            })
            .collect();
        SectionState::Ready { meshes, cages, last_style: None }
    }

    pub(super) fn is_ready(&self) -> bool {
        matches!(self, SectionState::Ready { .. })
    }

    /// Project `mask` onto this region's slots. The slabs cover the whole
    /// section: any non-empty mask draws them (overlap with finer children is
    /// depth-biased); `None` / empty hides them.
    pub(super) fn set_visible(&self, eng: &mut Engine, mask: Option<QuadrantMask>) {
        let SectionState::Ready { meshes, .. } = self else {
            return;
        };
        let on = mask.is_some_and(|m| !m.is_empty());
        for m in meshes {
            m.set_visible(eng, on);
        }
    }
    /// Push the far-material style onto this section's mesh
    /// (visibility decides which actually draw). The engine delta-gates unchanged style.
    fn set_style(&self, eng: &mut Engine, style: FadeStyle, flat_rgba: u32) {
        let SectionState::Ready { meshes, .. } = self else {
            return;
        };
        for m in meshes {
            m.set_style(eng, style, flat_rgba);
        }
    }
    /// [`Self::set_style`], gated on the pushed tuple actually changing since last
    /// time — the DrawDyn contract ("at rest, zero writes"): the engine delta-gates
    /// per-mesh too, but a settled far field must not even attempt the walk.
    pub(super) fn push_style(&mut self, eng: &mut Engine, style: FadeStyle, flat_rgba: u32) {
        let key = (style, flat_rgba);
        let SectionState::Ready { last_style, .. } = self else {
            return;
        };
        if *last_style == Some(key) {
            return;
        }
        *last_style = Some(key);
        self.set_style(eng, style, flat_rgba);
    }
    pub(super) fn free(self, eng: &mut Engine) {
        if let SectionState::Ready { meshes, cages, .. } = self {
            for m in meshes {
                m.free(eng);
            }
            for c in cages {
                eng.free_cage(c);
            }
        }
    }

    fn slot_count(&self) -> usize {
        match self {
            SectionState::Ready { meshes, .. } => meshes.iter().map(ChunkMeshes::slot_count).sum(),
            _ => 0,
        }
    }
}

impl World {
    fn section_slot_count(&self) -> usize {
        self.sections.values().map(SectionState::slot_count).sum()
    }

    fn local_chunk_slots(&self) -> usize {
        self.chunks
            .values()
            .filter_map(|l| l.state.live_meshes())
            .map(ChunkMeshes::slot_count)
            .sum()
    }

    /// Near-field GPU slots. When the engine has sampled `live_slots`, that
    /// total minus the live section slots; otherwise the local chunk count.
    /// Chunks never count against the far lane's section budget.
    fn near_chunk_slots(&self) -> usize {
        if self.gpu_live_slots != 0 {
            (self.gpu_live_slots as usize).saturating_sub(self.section_slot_count())
        } else {
            self.local_chunk_slots()
        }
    }

    /// Section slots the far lane may occupy: at least [`SECTION_SLOT_FLOOR`],
    /// else whatever remains under the CPU-cull knob after near chunks.
    pub(in crate::world) fn sections_allowed(&self) -> usize {
        SECTION_SLOT_FLOOR.max((self.slot_ceiling as usize).saturating_sub(self.near_chunk_slots()))
    }

    /// Sections held against [`sections_allowed`](Self::sections_allowed): Ready, in flight and
    /// queued for upload (a queued result is still `Meshing` in the map). Counted in sections, the
    /// unit the frontier is coarsened in: a rugged section spans several slabs (slots), and a slot
    /// count here would stop admission short of a frontier that fits the budget.
    pub(super) fn section_budget_used(&self) -> usize {
        self.sections.len()
    }

    pub(super) fn local_mesh_slots(&self) -> usize {
        self.local_chunk_slots() + self.section_slot_count()
    }
}

/// Mesh-lifecycle state of a loaded chunk. Owns GPU mesh via [`OwnedMesh`] token.
/// Handle ownership rides the state machine: moves on edit or frees on unload.
///
/// `NeedsMesh` has a `building` flag (not a separate state) to represent in-flight
/// mesh jobs. This avoids wedging when an async result goes stale due to a view
/// move: the claim is a bool that only `Air`/`Ready` cannot carry, and clearing
/// it on any result-consumption prevents chunks from getting stuck.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum MeshState {
    /// Uniform-air, born meshed: nothing to draw, no worker job ever queued.
    Air,
    /// Dense data awaiting a fresh ASYNC mesh. `building` is the in-flight
    /// claim: `true` once a mesh job is outstanding on the worker pool (`rev`
    /// on `Loaded` referees its result), held until the budgeted upload
    /// resolves or the result is dropped. `prev` is the still-drawn old mesh
    /// of a chunk whose rebuild was requested asynchronously (a light grid
    /// landed, a degraded mesh's real light arrived — see
    /// [`World::remesh_async`]): it keeps drawing until the fresh upload
    /// retires it, so an async relight never blanks the chunk. `None` for a
    /// never-meshed chunk.
    NeedsMesh {
        building: bool,
        prev: Option<ChunkMeshes>,
    },
    /// Drawable: owns the live GPU mesh(es) (up to one per pass).
    Ready(ChunkMeshes),
    /// Edited, awaiting the synchronous remesh. `prev` is the previously-drawn
    /// mesh of an edited-`Ready` chunk — kept drawn until the remesh replaces
    /// it — or `None` if the chunk had no mesh when edited.
    Dirty { prev: Option<ChunkMeshes> },
}

impl MeshState {
    /// The mesh(es) currently drawn, if any: a `Ready` chunk, or an edited
    /// `Dirty` one still showing its previous mesh. Borrows, so ownership stays
    /// in the state (the render/draw gate).
    pub(super) fn live_meshes(&self) -> Option<&ChunkMeshes> {
        match self {
            MeshState::Ready(m)
            | MeshState::Dirty { prev: Some(m) }
            | MeshState::NeedsMesh { prev: Some(m), .. } => Some(m),
            _ => None,
        }
    }

    /// Whether this chunk contributes its final ground truth to the frame:
    /// something is drawn for it, or there is provably nothing to draw
    /// (born-air). THE handoff predicate — near-section admission skipping
    /// and the settled LOD clip both gate on it, so "the chunks cover this"
    /// can never mean two different things.
    pub(super) fn settled(&self) -> bool {
        self.live_meshes().is_some() || matches!(self, MeshState::Air)
    }
    /// Move the owned meshes out for freeing. Consumes `self`; the caller owns
    /// the token afterwards and must `free` it (or carry it on).
    #[must_use]
    pub(super) fn into_owned(self) -> Option<ChunkMeshes> {
        match self {
            MeshState::Ready(m)
            | MeshState::Dirty { prev: Some(m) }
            | MeshState::NeedsMesh { prev: Some(m), .. } => Some(m),
            _ => None,
        }
    }
    /// Free the owned meshes, if any. Consuming `self` here means a caller can't
    /// hold on to a state and free it twice. Every GPU-freeing removal —
    /// unload, world-leave, [`Loaded::retire`] — routes through this.
    pub(super) fn free_owned(self, eng: &mut Engine) {
        if let Some(meshes) = self.into_owned() {
            meshes.free(eng);
        }
    }
    /// The state a fresh upload produces: `Ready` if any pass yielded a handle,
    /// else `Air` (an all-air chunk uploads to nothing).
    pub(super) fn from_upload(handles: ByPass<Option<MeshHandle>>) -> MeshState {
        let meshes = ByPass::from_fn(|p| handles[p].map(OwnedMesh::new));
        match ChunkMeshes::new(meshes) {
            Some(m) => MeshState::Ready(m),
            None => MeshState::Air,
        }
    }
    /// A fresh never-meshed (or reset) state: not building, nothing carried.
    pub(super) fn needs_mesh() -> MeshState {
        MeshState::NeedsMesh {
            building: false,
            prev: None,
        }
    }
    /// Invalidate to `Dirty`, carrying the currently-drawn mesh forward as
    /// `prev` so it keeps drawing until the sync remesh. Nothing is freed here
    /// — the token just moves. `Ready(m) -> Dirty{Some(m)}`; an already-`Dirty`
    /// chunk keeps its `prev`, and so does a `NeedsMesh` carrying one (an edit
    /// landing mid-async-rebuild keeps drawing the old mesh); other handle-less
    /// states (incl. a bare `building` chunk, whose in-flight claim is dropped
    /// — the sync remesh takes over and the orphan async result is refereed
    /// out by `rev`) -> `Dirty{None}`.
    pub(super) fn invalidate(&mut self) {
        let prev = std::mem::replace(self, MeshState::needs_mesh()).into_owned();
        *self = MeshState::Dirty { prev };
    }
    /// Release the in-flight mesh claim if this chunk is still awaiting its
    /// build (a carried `prev` keeps drawing). A no-op once the chunk has
    /// moved on (`Dirty` via an edit, `Ready`/`Air` via a prior consume):
    /// those states carry no claim. Called at every mesh-result-consumption
    /// site whose result did NOT apply, so a stale result (view moved, chunk
    /// left the box) can never wedge the claim.
    pub(super) fn release_build(&mut self) -> bool {
        if let MeshState::NeedsMesh { building, .. } = self {
            if *building {
                *building = false;
                return true;
            }
        }
        false
    }
    pub(super) fn is_building(&self) -> bool {
        matches!(self, MeshState::NeedsMesh { building: true, .. })
    }
    pub(super) fn is_needs_mesh(&self) -> bool {
        matches!(self, MeshState::NeedsMesh { .. })
    }
    pub(super) fn is_dirty(&self) -> bool {
        matches!(self, MeshState::Dirty { .. })
    }
    /// Live GPU handles this state currently carries (at most one `ChunkMeshes`).
    #[cfg(test)]
    pub(super) fn live_handles(&self) -> Vec<MeshHandle> {
        self.live_meshes()
            .map(ChunkMeshes::handles)
            .unwrap_or_default()
    }
    /// Record-and-drop the owned meshes without an [`Engine`] — fake test
    /// handles never index GPU memory, so the engine free is skipped.
    #[cfg(test)]
    pub(super) fn free_logged(self) {
        if let Some(meshes) = self.into_owned() {
            for h in meshes.handles() {
                mesh_free_log::record(h);
            }
        }
    }
}
