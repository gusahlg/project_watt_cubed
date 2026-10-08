//! The `World::stream` lanes. One [`stream_lanes!`] table declares each lane's marker type, its
//! per-frame budget and its body; `stream` and `pump` call each lane's `run` at its fixed
//! position in the pass, because that order is load-bearing.
//!
//! The async admission lanes (mesh, section, light and column generation) share one loop
//! (`world::admit` / `World::request_region_data`) over the per-lane accessor surface
//! (`world::StreamLane`). They derive their per-frame `Deadline` from the lane's
//! `Budget::Millis`, minted after each lane's pending gate so an idle frame never samples the
//! clock, and share the one forward-progress floor rule (`world::admission_exhausted`).

use std::time::Duration;

use voxel_engine::Engine;
use voxel_engine::producer::{Budget, Progress};

use super::{Coord, LightLane, MeshLane, SectionLane, World, admit, pipeline};

/// The span of a `Budget::Millis` lane budget, the one place a budget becomes a duration.
pub(in crate::world) fn duration(budget: Budget) -> Duration {
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

fn far_center(world: &World) -> Coord {
    world.section_center().expect("stream lanes run after center is set")
}

/// Declares a `new` row's marker struct (docs attach to it); an `admit` row's marker is a
/// `world::StreamLane` implementor that already exists.
macro_rules! declare_lane {
    (new $(#[$doc:meta])* $lane:ident) => {
        $(#[$doc])*
        pub(in crate::world) struct $lane;
    };
    (admit $(#[$doc:meta])* $lane:ident) => {};
}

/// The one lane table: `[new|admit] Marker(budget) => |world, eng, budget| { body }`. Expands
/// each marker's `BUDGET` and inherent `run`.
macro_rules! stream_lanes {
    ($( $(#[$doc:meta])* $kind:ident $lane:ident ($budget:expr)
        => |$world:ident, $eng:ident, $b:ident| $body:block ),+ $(,)?) => {
        $(
            declare_lane!($kind $(#[$doc])* $lane);
            impl $lane {
                pub(in crate::world) const BUDGET: Budget = $budget;

                #[inline]
                pub(in crate::world) fn run($world: &mut World, $eng: Option<&mut Engine>) -> Progress {
                    let $b = Self::BUDGET;
                    $body
                }
            }
        )+
    };
}

stream_lanes! {
    /// Patches each drawable chunk's GPU visibility mask; idle without the engine.
    new OcclusionLane(Budget::Millis(0.5))
        => |world, eng, b| {
            let Some(eng) = eng else {
                return Progress::Idle;
            };
            world.rebuild_occlusion(Some(eng), b)
        },
    /// Synchronous remesh of edited (`Dirty`) chunks (`World::remesh_dirty`). Uploads through
    /// the engine.
    new DirtyRemeshLane(Budget::Dispatches(super::DIRTY_BUDGET as u16))
        => |world, eng, _b| {
            if !world.dirty_pending() {
                return Progress::Idle;
            }
            world.remesh_dirty(eng.expect("dirty-remesh is a CPU lane; eng required"))
        },
    /// The far-field relief bake runs on a spawned thread; this only polls the
    /// completion channel and (idempotently) spawns a new one.
    new MipLane(Budget::Dispatches(1))
        => |world, _eng, _b| {
            world.poll_mip();
            world.ensure_mip_bake();
            Progress::Idle
        },
    /// Section edit-overlay lane: re-materialise the edit-folded cell (the δf)
    /// for every section a live edit touched. Bounded by touched sections; an
    /// unedited world pays nothing.
    new SectionOverlayLane(Budget::Millis(1.0))
        => |world, _eng, b| { world.refresh_section_overlay(b) },
    /// Frees GPU meshes of edited sections so they re-extract from the updated
    /// overlay. Needs the engine.
    new SectionRemeshLane(Budget::Millis(1.0))
        => |world, eng, _b| {
            world.remesh_dirty_sections(eng.expect("section-remesh is a CPU lane; eng required"));
            Progress::Idle
        },
    /// Rebuilds the drawn covering. Also the level-triggered load arming.
    new SectionVisibleLane(Budget::Millis(1.0))
        => |world, eng, _b| {
            world.rebuild_section_visible(eng);
            Progress::Idle
        },
    /// Lands finished worker results (generate/mesh/light-apply); mesh upload
    /// to GPU is budgeted. Needs the engine.
    new DrainLane(Budget::Millis(1.0))
        => |world, eng, b| {
            world.drain_results(eng.expect("drain is a CPU lane; eng required"), duration(b));
            Progress::Idle
        },
    /// Column granularity (one job per `(cx,cz)` span) means this keeps its own
    /// gather/claim inside `World::request_region_data` rather than the per-chunk
    /// `admit` loop, but shares the one budget + forward-progress floor rule.
    new GenerateLane(Budget::Millis(2.0))
        => |world, _eng, b| {
            let center = stream_center(world);
            world.request_region_data(center, b)
        },
    /// Cross-chunk light settling admission (the `world::LightLane` marker).
    admit LightLane(Budget::Millis(1.0))
        => |world, _eng, b| {
            admit::<LightLane>(world, stream_center(world), b);
            Progress::Idle
        },
    /// Fresh full-res chunk meshing admission (the `world::MeshLane` marker).
    admit MeshLane(Budget::Millis(2.0))
        => |world, _eng, b| {
            admit::<MeshLane>(world, stream_center(world), b);
            Progress::Idle
        },
    /// LOD2 column-section admission (the `world::SectionLane` marker).
    admit SectionLane(Budget::Millis(1.0))
        => |world, _eng, b| {
            admit::<SectionLane>(world, far_center(world), b);
            Progress::Idle
        },
}
