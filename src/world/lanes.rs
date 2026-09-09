//! The `World::stream` passes as scheduler producers (`sched::Run`). Each is
//! registered `manual` (see `sched::Scheduler::manual`) and driven at its exact
//! call point inside `World::stream`, because its order relative to the other
//! stream passes is load-bearing.
//!
//! The async admission lanes — mesh, section, light, and column generation —
//! share one loop (`world::admit` / `World::request_region_data`) over the
//! per-lane accessor surface (`world::StreamLane`). They derive their per-frame
//! `Deadline` from the `Budget::Millis` the scheduler hands `run()` — ONE budget
//! locus, the manifest, never a private `pipeline::*_BUDGET` const — minted after
//! each lane's pending gate so an idle frame never samples the clock — and share
//! the one forward-progress floor rule (`world::admission_exhausted`). There is
//! no second scheduler here: budget and floor come from the scheduler that drives
//! the producer.
//!
//! The whole lane roster is ONE [`stream_lanes!`] table: each row declares the
//! lane's `StreamLanes` field, its marker type (declared here for `new` rows;
//! `use` rows are the `world::StreamLane` markers that already exist), its
//! manifest name + budget, and its `run` body inline — the struct/impl/register
//! scaffolding that used to be restated per lane comes from the macro.

use std::time::Duration;

use voxel_engine::producer::{Budget, Cadence, Footprint, FootprintKey, Producer, Progress};

use crate::sched::{Ctx, ManualHandle, Run, Scheduler};

use super::{Coord, LightLane, MeshLane, SectionLane, World, admit, pipeline};

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
fn duration(budget: Budget) -> Duration {
    let Budget::Millis(ms) = budget else {
        unreachable!("streaming admission lanes declare Budget::Millis");
    };
    Duration::from_secs_f32(ms / 1000.0)
}

/// Mint the admission deadline from `budget` *after* the lane's pending gate.
/// `Instant::now` lives here so an idle frame never samples the clock.
pub(in crate::world) fn paced_deadline(world: &World, budget: Budget) -> pipeline::Deadline {
    pipeline::Deadline::from_budget(world.stream_pacer.duration(duration(budget)))
}

fn stream_center(world: &World) -> Coord {
    world.center.expect("stream lanes run after center is set")
}

fn eng<'a>(
    slot: &'a mut Option<&mut voxel_engine::Engine>,
    what: &'static str,
) -> &'a mut voxel_engine::Engine {
    slot.as_deref_mut().expect(what)
}

fn admit_run<L: super::StreamLane>(ctx: &mut Ctx<'_>, b: Budget) -> Progress {
    admit::<L>(ctx.world, stream_center(ctx.world), b);
    Progress::Idle
}

/// Declares a `new` row's marker struct (docs attach to it); a `use` row's
/// marker already exists in `world::mod` (the `StreamLane` implementors).
macro_rules! declare_lane {
    (new $(#[$doc:meta])* $lane:ident) => {
        $(#[$doc])*
        pub struct $lane;
    };
    (use $(#[$doc:meta])* $lane:ident) => {};
    (admit $(#[$doc:meta])* $lane:ident) => {};
}

/// The one lane table: `field: [new|use] Marker(name, budget) => |ctx, budget| { body }`.
/// Expands the `StreamLanes` handle struct, `register`, and each marker's
/// `manifest()` + `Run` impl; the run bodies stay inline and visible below.
macro_rules! stream_lanes {
    ($( $(#[$doc:meta])* $field:ident : $kind:ident $lane:ident ($name:literal, $budget:expr)
        => |$ctx:ident, $b:ident| $body:block ),+ $(,)?) => {
        /// The scheduler handles for every `World::stream` call-point producer, kept on
        /// `World` so `stream` can drive each at its position. One struct, set once by
        /// `Game::new`, rather than a handle field per lane.
        #[derive(Clone, Copy)]
        pub struct StreamLanes {
            $(pub $field: ManualHandle,)+
        }

        impl StreamLanes {
            /// Register every stream-lane producer on `sched` and return their handles.
            /// The single wiring point (called by `Game::new`), so the lane marker types
            /// stay private to `crate::world`.
            pub(crate) fn register(sched: &mut Scheduler) -> StreamLanes {
                StreamLanes {
                    $($field: sched.register_manual($lane::manifest(), Box::new($lane)),)+
                }
            }
        }

        $(
            declare_lane!($kind $(#[$doc])* $lane);
            impl $lane {
                pub fn manifest() -> Producer {
                    stream_manifest($name, $budget)
                }
            }
            impl Run for $lane {
                fn run(&mut self, $ctx: &mut Ctx<'_>, $b: Budget) -> Progress $body
            }
        )+
    };
}

stream_lanes! {
    /// Patches each drawable chunk's GPU visibility mask — CPU lane, needs the
    /// engine.
    occlusion: new OcclusionLane("occlusion", Budget::Millis(0.5))
        => |ctx, b| {
            let eng = eng(&mut ctx.eng, "occlusion patches masks; eng required");
            ctx.world.rebuild_occlusion(eng, b)
        },
    /// Synchronous remesh of edited (`Dirty`) chunks (`World::remesh_dirty`).
    /// Uploads through the engine, so it needs `ctx.eng`.
    dirty_remesh: new DirtyRemeshLane("dirty_remesh", Budget::Dispatches(super::DIRTY_BUDGET as u16))
        => |ctx, _b| {
            let eng = eng(&mut ctx.eng, "dirty-remesh is a CPU lane; eng required");
            ctx.world.remesh_dirty(eng)
        },
    /// The far-field relief bake runs on a spawned thread; this only polls the
    /// completion channel and (idempotently) spawns a new one.
    mip: new MipLane("lod_mip", Budget::Dispatches(1))
        => |ctx, _b| {
            ctx.world.poll_mip();
            ctx.world.ensure_mip_bake();
            Progress::Idle
        },
    /// Section edit-overlay lane: re-materialise the edit-folded cell (the δf)
    /// for every section a live edit touched. Bounded by touched sections; an
    /// unedited world pays nothing.
    section_overlay: new SectionOverlayLane("lod_overlay", Budget::Millis(1.0))
        => |ctx, _b| {
            ctx.world.refresh_section_overlay();
            Progress::Idle
        },
    /// Frees GPU meshes of edited sections so they re-extract from the updated
    /// overlay. CPU lane — needs the engine.
    section_remesh: new SectionRemeshLane("lod_section_remesh", Budget::Millis(1.0))
        => |ctx, _b| {
            let eng = eng(&mut ctx.eng, "section-remesh is a CPU lane; eng required");
            ctx.world.remesh_dirty_sections(eng);
            Progress::Idle
        },
    /// Rebuilds the drawn covering every frame (hard LOD cut). Also the
    /// level-triggered load arming.
    section_visible: new SectionVisibleLane("lod_visible", Budget::Millis(1.0))
        => |ctx, _b| {
            ctx.world.rebuild_section_visible(ctx.eng.as_deref_mut());
            Progress::Idle
        },
    /// Lands finished worker results (generate/mesh/light-apply); mesh upload
    /// to GPU is budgeted. CPU lane — needs the engine.
    drain: new DrainLane("drain", Budget::Millis(1.0))
        => |ctx, b| {
            let eng = eng(&mut ctx.eng, "drain is a CPU lane; eng required");
            ctx.world.drain_results(eng, duration(b));
            Progress::Idle
        },
    /// Column granularity (one job per `(cx,cz)` span) means this keeps its own
    /// gather/claim inside `World::request_region_data` rather than the per-chunk
    /// `admit` loop, but shares the one budget + forward-progress floor rule.
    generate: new GenerateLane("generate", Budget::Millis(2.0))
        => |ctx, b| {
            let center = stream_center(ctx.world);
            ctx.world.request_region_data(center, b)
        },
    /// Cross-chunk light settling admission (the `world::LightLane` marker).
    light_admit: admit LightLane("light_admit", Budget::Millis(1.0))
        => |ctx, b| { admit_run::<LightLane>(ctx, b) },
    /// Fresh full-res chunk meshing admission (the `world::MeshLane` marker).
    mesh_admit: admit MeshLane("mesh_admit", Budget::Millis(2.0))
        => |ctx, b| { admit_run::<MeshLane>(ctx, b) },
    /// LOD2 column-section admission (the `world::SectionLane` marker).
    section_admit: admit SectionLane("section_admit", Budget::Millis(1.0))
        => |ctx, b| { admit_run::<SectionLane>(ctx, b) },
}
