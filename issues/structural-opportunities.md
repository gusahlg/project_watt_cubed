# Structural and improvement opportunities

These are not all immediate bugs. They are places where a relatively early architectural decision can preserve the project's stated freedom, determinism, and performance goals.

## 1. Make authority a domain boundary, not a relay detail

The current server orders opaque cell strings while inventory, loot, movement, simulation, and most mod behavior remain client-owned. That boundary will become more expensive to change as reactions, automation, protection, and trading land.

A useful target shape is:

1. clients submit typed intents with IDs and expected revisions;
2. the server validates movement/permission/resources and applies one atomic state transition;
3. accepted state and rejected corrections are broadcast in server order;
4. clients predict only operations they know how to roll back;
5. server-side mod hooks participate in validation rather than trusting client-side mods.

This also clarifies the README: “authoritative” can then mean gameplay authority, not only overlay ordering.

## 2. Move inventory/economy state into the core; let mods present it

The draft describes an inventory that exists without an interface mod. Today the element stash is created as part of the default mod set, and pickups happen through the enabled inventory mod. Disabling the UI can therefore disable or discard game state rather than merely changing presentation.

Core services should own items, crafting transactions, and persistence. Mods should contribute views and intents. A valuable invariant test is: disable inventory UI → mine → re-enable → resources remain.

## 3. Design deterministic simulation before making systems active

Simulation systems currently mutate sequentially in registration order on each client. That is harmless while they are inert, but it is not a safe base for reactions, circuits, liquids, corrosion, or multiplayer machines.

Prefer server-owned stable phases: immutable read snapshot, deterministic intent emission, stable conflict reduction, then commit. Persist sparse active state and include system/mod versions in the content fingerprint. Avoid iteration-order semantics from hash maps.

## 4. Give streaming jobs identity, cancellation, failure, and budgets

**Largely landed 2026-07-14.** Jobs now carry a claim identity (`JobKey`);
panics report `Done::Failed` with bounded retry then quarantine; queued jobs
whose region leaves the live view are descheduled at dequeue
(`Done::Cancelled`, claim released without strikes); and near work pops
nearest to the CURRENT view centre rather than FIFO. Enqueue/apply budgets
already existed.

Remaining: a stress test that teleports the stream centre repeatedly while
injecting slow and failing jobs, then requires bounded queues and convergence
around the last centre; and shutdown that cancels rather than drains
irrelevant work (drop today joins after at most one job per worker, which has
been acceptable).

## 5. Separate infinite-world truth from the current finite vertical LOD slab

The near world is vertically unbounded, but far sections are fixed to `[0, 512)` in [`section.rs`](../src/world/section.rs#L45). Deep mining or very high travel can therefore lose a meaningful far field. Treat the current slab as an explicit first implementation, not the final spatial model.

Likely successors are vertical slabs selected around the eye, or a true 3D hierarchy. Preserve the useful current properties: camera-relative transforms, exact covering partitions, deterministic downsampling, and error-driven selection.

## 6. Replace texture-layer modulo aliasing with explicit indirection

When block IDs exceed the device texture-array cap, near and far mesh vertices use `id % layer_cap` ([near](../src/world/mesh.rs#L435), [far](../src/world/section/mesh.rs#L277)). A valid material can silently display an unrelated texture. That conflicts with a game where composition identity is central.

Options include paged arrays, a material-to-resident-layer indirection table, an atlas, or a deterministic composition-aware fallback texture. Whichever is chosen, make eviction/aliasing observable and test IDs just above the device cap.

## 7. Make far-light/material policy explicit

Far water intentionally becomes static opaque and far emission currently disappears. Rather than accumulating one-off rules, define and test a material transition matrix:

| Property | Near field | Far field policy to decide |
|---|---|---|
| opaque, non-emissive | full mesh/light | current coarse path |
| transparent non-liquid | blend | blend, cutout, or opaque approximation |
| transparent liquid | animated water | static translucent/opaque approximation |
| opaque liquid | opaque | opaque |
| emissive | blocklight flood | self-emission or coarse flood |

The desired answer can favor speed; it should not be an accidental discontinuity.

## 8. Reduce result-channel payload size and measure worker throughput

**Done.** The mesh variant is boxed (`Done` ≤ 128 B, compile-time guarded) and
the ignored `mesh_result_channel_throughput` benchmark records the before/after
(28.6k → 29.2k jobs/s; throughput is meshing-bound, the boxing is a
payload/regression guard rather than a measured speedup).

## 9. Make GPU resource ownership and access state explicit

**Substantially landed 2026-07-14** with the exposure/TAA fixes: the exposure
slot parity is one pure, unit-tested function; TAA history barriers derive
their source scopes from per-image tracked layouts; exposure/TAA toggles
reset their temporal state; and the scene depth layout comes from a single
`depth_pass_layout` function for both configurations. Remaining direction:
typed `Waited<FrameSlot>` proofs before host reads, and explicit invalidation
epochs on resize (today resize recreates the resources outright, which is
equivalent but implicit).

## 10. Decouple shader generation from ordinary builds

The engine build script can regenerate tracked SPIR-V fallbacks during a normal build, making compiler-version noise look like source changes. It compiles/copies modules without validating every variant. Prefer an explicit `shader-gen` task that:

- compiles all macro variants;
- runs `spirv-val` with the shipping Vulkan target;
- writes a source/options/compiler hash manifest;
- updates tracked fallbacks only on deliberate invocation;
- lets ordinary builds verify rather than rewrite the manifest.

**Partial progress 2026-07-14:** an all-module `spirv-val --target-env
vulkan1.3` gate now runs in the engine test suite (skipping loudly outside
the dev shell), and the fallbacks were regenerated once with the dev shell's
PINNED slangc — since the flake pins the compiler, ordinary rebuilds are now
byte-stable. The explicit `shader-gen` task with a source/options/compiler
manifest remains the full answer. (The once-invalid water module that
motivated this is repaired and its pipeline live — see E-08.)

## 11. Move golden rendering toward fixed-extent offscreen capture

The current acceptance harness still depends on a desktop window manager keeping a requested size. It attempts a compositor-specific floating command, but in this environment that control socket is absent and the window is resized from `1542×700` to `1542×1406` during the run. Image comparisons then become invalid even though sky-hole/performance criteria still work.

The durable solution is fixed-extent offscreen rendering (or a hidden fixed-size surface) with explicit readback. A temporary improvement is to assert the live extent before every stage and fail immediately with one infrastructure error instead of performing the whole suite.

## 12. Evolve saves toward regioned, checksummed sparse state

Seed-plus-overlay is a strong fit for region files. Frame/checksum edit and mod-state regions, enforce aggregate budgets, compact cells restored to generated baseline, and record an authoritative edit sequence only if game semantics require chronological reduction. This improves backup recovery, join snapshots, sharding, and partial loading together.

## 13. Make HUD/menu modding a typed render-plan boundary

The current core already renders mod-provided HUD data, which is a good dependency direction. Finish it by making Full/Minimal/Off a pure policy over typed widgets/modal layers. Then default, minimal, accessibility, and streamer HUDs become data/policy swaps rather than scattered `if` statements.

## 14. Add cross-platform determinism CI before broad mod/content compatibility

For fixed seeds and coordinate sets, hash:

- resolved placement table and block compositions;
- generated voxel IDs before and after canonical ID remapping;
- coarse section extraction/downsampling;
- serialized sparse overlay replay;
- selected LOD covering.

Run on x86_64 and aarch64 at minimum. Prefer canonical composition/content hashes over raw local numeric IDs when comparing builds, because mod registration may legitimately assign different local IDs.

## 15. Keep performance claims tied to reproducible scenarios

The project has unusually useful timing hooks and a benchmark mode. Extend that discipline with named scenarios and machine metadata rather than optimizing from code shape alone:

- cold entry and steady traversal;
- high-altitude LOD selection;
- rapid teleport cancellation;
- palette growth/texture upload;
- dense lighting and emissive caves;
- multiplayer fan-out at several player densities;
- GPU passes with VRS/TAA/exposure toggles.

Record median and tail latency, memory, queue depths, and draw/section counts. This keeps “performance over readability” grounded in measured benefit and prevents correctness shortcuts from masquerading as optimization.

**2026-07-14 status:** GPU attribution now covers the whole render command
buffer (`resolve`/`post` passes). Bench numbers on the standing scenario
(RTX 3070, fullscreen, msaa 1, taa+exposure on): 1.63 ms/frame, record
0.12 ms, GPU ≈ 1.55 ms of which TAA resolve ~0.5 ms and sky ~0.33 ms.
Measured next opportunities: a shared-memory tiling rewrite of
`taa_resolve.comp` (the resolve is texture-fetch-bound: ~11 samples/px), and
extending GPU metering to the copy command buffer (tonemap present-copy is
the last unmeasured GPU segment).
