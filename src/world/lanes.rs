//! The `World::stream` passes as scheduler producers (`sched::Run`). Each is
//! registered `manual` (see `sched::Scheduler::manual`) and driven at its exact
//! call point inside `World::stream`, because its order relative to the other
//! stream passes is load-bearing.
//!
//! The async admission lanes — mesh, section, light, and column generation —
//! share one loop (`world::admit` / `World::request_region_data`) over the
//! per-lane accessor surface (`world::StreamLane`). They derive their per-frame
//! `Deadline` from the `Budget::Millis` the scheduler hands `run()` — ONE budget
//! locus, the manifest, never a private `pipeline::*_BUDGET` const — and share
//! the one forward-progress floor rule (`world::admission_exhausted`). There is
//! no second scheduler here: budget and floor come from the scheduler that drives
//! the producer.

use std::time::Duration;

use voxel_engine::producer::{
    Budget, Cadence, Footprint, FootprintKey, Producer, Progress,
};

use crate::sched::{Ctx, ManualHandle, Run, Scheduler};

use super::{admit, pipeline, LightLane, MeshLane, SectionLane};

/// The scheduler handles for every `World::stream` call-point producer, kept on
/// `World` so `stream` can drive each at its position. One struct, set once by
/// `Game::new`, rather than a handle field per lane.
#[derive(Clone, Copy)]
pub struct StreamLanes {
    pub occlusion: ManualHandle,
    pub dirty_remesh: ManualHandle,
    pub mip: ManualHandle,
    pub section_overlay: ManualHandle,
    pub section_remesh: ManualHandle,
    pub section_visible: ManualHandle,
    pub drain: ManualHandle,
    pub generate: ManualHandle,
    pub light_admit: ManualHandle,
    pub mesh_admit: ManualHandle,
    pub section_admit: ManualHandle,
}

impl StreamLanes {
    /// Register every stream-lane producer on `sched` and return their handles.
    /// The single wiring point (called by `Game::new`), so the lane marker types
    /// stay private to `crate::world`.
    pub(crate) fn register(sched: &mut Scheduler) -> StreamLanes {
        StreamLanes {
            occlusion: sched.register_manual(OcclusionLane::manifest(), Box::new(OcclusionLane)),
            dirty_remesh: sched
                .register_manual(DirtyRemeshLane::manifest(), Box::new(DirtyRemeshLane)),
            mip: sched.register_manual(MipLane::manifest(), Box::new(MipLane)),
            section_overlay: sched
                .register_manual(SectionOverlayLane::manifest(), Box::new(SectionOverlayLane)),
            section_remesh: sched
                .register_manual(SectionRemeshLane::manifest(), Box::new(SectionRemeshLane)),
            section_visible: sched
                .register_manual(SectionVisibleLane::manifest(), Box::new(SectionVisibleLane)),
            drain: sched.register_manual(DrainLane::manifest(), Box::new(DrainLane)),
            generate: sched.register_manual(GenerateLane::manifest(), Box::new(GenerateLane)),
            light_admit: sched.register_manual(LightLane::manifest(), Box::new(LightLane)),
            mesh_admit: sched.register_manual(MeshLane::manifest(), Box::new(MeshLane)),
            section_admit: sched.register_manual(SectionLane::manifest(), Box::new(SectionLane)),
        }
    }
}

/// A CPU, `Cadence::Frame`, `Global`-footprint manifest — the shape every stream
/// lane shares. `name` and `budget` (the pass's per-frame item cap, declarative)
/// are all that vary.
fn stream_manifest(name: &'static str, budget: Budget) -> Producer {
    Producer {
        name,
        footprint: Footprint {
            reads: vec![FootprintKey::Global],
            writes: vec![FootprintKey::Global],
        },
        cadence: Cadence::Frame,
        budget,
    }
}

/// The per-frame admission deadline from the scheduler-provided budget — the one
/// place a lane's `Budget::Millis` becomes a [`Deadline`](pipeline::Deadline),
/// so the budget has a single definition (the manifest).
fn deadline(budget: Budget) -> pipeline::Deadline {
    let Budget::Millis(ms) = budget else {
        unreachable!("streaming admission lanes declare Budget::Millis");
    };
    pipeline::Deadline::from_budget(Duration::from_secs_f32(ms / 1000.0))
}

// Core admission lanes
//
// Mesh, section, and light admission ARE the producers (no shim indirection):
// the marker types implement both `world::StreamLane` (accessors) and `Run`
// (the producer), so registration is `Box::new(MeshLane)` and the `run()` body
// is one `admit::<Self>` over the scheduler's deadline.

impl MeshLane {
    pub fn manifest() -> Producer {
        stream_manifest("mesh_admit", Budget::Millis(2.0))
    }
}
impl Run for MeshLane {
    fn run(&mut self, ctx: &mut Ctx<'_>, budget: Budget) -> Progress {
        let center = ctx.world.center.expect("mesh-admit runs after center is set");
        admit::<MeshLane>(ctx.world, center, deadline(budget));
        Progress::Idle
    }
}

impl SectionLane {
    pub fn manifest() -> Producer {
        stream_manifest("section_admit", Budget::Millis(1.0))
    }
}
impl Run for SectionLane {
    fn run(&mut self, ctx: &mut Ctx<'_>, budget: Budget) -> Progress {
        let center = ctx.world.center.expect("section-admit runs after center is set");
        admit::<SectionLane>(ctx.world, center, deadline(budget));
        Progress::Idle
    }
}

impl LightLane {
    pub fn manifest() -> Producer {
        stream_manifest("light_admit", Budget::Millis(1.0))
    }
}
impl Run for LightLane {
    fn run(&mut self, ctx: &mut Ctx<'_>, budget: Budget) -> Progress {
        let center = ctx.world.center.expect("light-admit runs after center is set");
        admit::<LightLane>(ctx.world, center, deadline(budget));
        Progress::Idle
    }
}

/// Column granularity (one job per `(cx,cz)` span) means this keeps its own
/// gather/claim inside `World::request_region_data` rather than the per-chunk
/// `admit` loop, but shares the one budget + forward-progress floor rule.
pub struct GenerateLane;
impl GenerateLane {
    pub fn manifest() -> Producer {
        stream_manifest("generate", Budget::Millis(2.0))
    }
}
impl Run for GenerateLane {
    fn run(&mut self, ctx: &mut Ctx<'_>, budget: Budget) -> Progress {
        let center = ctx.world.center.expect("generate runs after center is set");
        ctx.world.request_region_data(center, deadline(budget))
    }
}

/// Lands finished worker results (generate/mesh/light-apply); mesh upload to
/// GPU is budgeted. CPU lane — needs the engine.
pub struct DrainLane;

impl DrainLane {
    pub fn manifest() -> Producer {
        stream_manifest("drain", Budget::Millis(1.0))
    }
}

impl Run for DrainLane {
    fn run(&mut self, ctx: &mut Ctx<'_>, _budget: Budget) -> Progress {
        let eng = ctx.eng.as_deref_mut().expect("drain is a CPU lane; eng required");
        let apply = pipeline::Deadline::from_budget(pipeline::LIGHT_APPLY_BUDGET);
        ctx.world.drain_results(eng, apply);
        Progress::Idle
    }
}

// Serial stream passes (fixed call points)

/// Patches each drawable chunk's GPU visibility mask — CPU lane, needs the
/// engine.
pub struct OcclusionLane;

impl OcclusionLane {
    pub fn manifest() -> Producer {
        stream_manifest("occlusion", Budget::Dispatches(super::OCCLUSION_FILL_BUDGET as u16))
    }
}

impl Run for OcclusionLane {
    fn run(&mut self, ctx: &mut Ctx<'_>, _budget: Budget) -> Progress {
        let eng =
            ctx.eng.as_deref_mut().expect("occlusion patches masks; eng required");
        ctx.world.rebuild_occlusion(eng)
    }
}

/// Synchronous remesh of edited (`Dirty`) chunks (`World::remesh_dirty`).
/// Uploads through the engine, so it needs `ctx.eng`.
pub struct DirtyRemeshLane;

impl DirtyRemeshLane {
    pub fn manifest() -> Producer {
        stream_manifest("dirty_remesh", Budget::Dispatches(super::DIRTY_BUDGET as u16))
    }
}

impl Run for DirtyRemeshLane {
    fn run(&mut self, ctx: &mut Ctx<'_>, _budget: Budget) -> Progress {
        let eng = ctx.eng.as_deref_mut().expect("dirty-remesh is a CPU lane; eng required");
        ctx.world.remesh_dirty(eng)
    }
}

// Aux lanes (LOD far-field)
//
// All four sit inside `stream`'s far-field block, each at its exact call point. They
// are bounded single passes (not budgeted resumable iteration), so each
// self-gates internally where applicable and reports `Progress::Idle`. They stay
// separate lanes because unmigrated passes run between them.

/// The bake runs on a spawned thread; this only polls the completion channel
/// and (idempotently) spawns a new one.
pub struct MipLane;

impl MipLane {
    pub fn manifest() -> Producer {
        stream_manifest("lod_mip", Budget::Dispatches(1))
    }
}

impl Run for MipLane {
    fn run(&mut self, ctx: &mut Ctx<'_>, _budget: Budget) -> Progress {
        ctx.world.poll_mip();
        ctx.world.ensure_mip_bake();
        Progress::Idle
    }
}

/// Section edit-overlay lane: re-materialise the edit-folded cell (the δf) for
/// every section a live edit touched. Bounded by touched sections; an unedited
/// world pays nothing.
pub struct SectionOverlayLane;

impl SectionOverlayLane {
    pub fn manifest() -> Producer {
        stream_manifest("lod_overlay", Budget::Millis(1.0))
    }
}

impl Run for SectionOverlayLane {
    fn run(&mut self, ctx: &mut Ctx<'_>, _budget: Budget) -> Progress {
        ctx.world.refresh_section_overlay();
        Progress::Idle
    }
}

/// Frees GPU meshes of edited sections so they re-extract from the updated
/// overlay. CPU lane — needs the engine.
pub struct SectionRemeshLane;

impl SectionRemeshLane {
    pub fn manifest() -> Producer {
        stream_manifest("lod_section_remesh", Budget::Millis(1.0))
    }
}

impl Run for SectionRemeshLane {
    fn run(&mut self, ctx: &mut Ctx<'_>, _budget: Budget) -> Progress {
        let eng =
            ctx.eng.as_deref_mut().expect("section-remesh is a CPU lane; eng required");
        ctx.world.remesh_dirty_sections(eng);
        Progress::Idle
    }
}

/// Rebuilds the drawn covering every frame (hard LOD cut).
/// Also the level-triggered load arming.
pub struct SectionVisibleLane;

impl SectionVisibleLane {
    pub fn manifest() -> Producer {
        stream_manifest("lod_visible", Budget::Millis(1.0))
    }
}

impl Run for SectionVisibleLane {
    fn run(&mut self, ctx: &mut Ctx<'_>, _budget: Budget) -> Progress {
        ctx.world.rebuild_section_visible(ctx.eng.as_deref_mut());
        Progress::Idle
    }
}
