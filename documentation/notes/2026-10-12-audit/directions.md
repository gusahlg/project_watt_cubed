<!-- SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors -->
<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# Improvements that fit the philosophy

These are proposals and measurement plans, not implemented optimizations. There are no new FPS
claims in this audit. Preserve the existing native build, canonical matter identities,
deterministic generation, bounded work and resident rendering model.

## D01 — Make a generator a registered implementation, not a built-in enum choice

**Highest-value extensibility change.** The public `TerrainGenerator` trait is already a useful
contract: point/batch generation, classification, column/surface samples, sky layout, mass oracle
and atlas access. The missing step is connecting it to mod registration, world construction,
saves, server startup and content compatibility. `Mod::worldgen` currently returns only
`WorldgenKind::Flat` or `Diffusion`; implementing a trait in a mod cannot select a third generator
through normal play.

Introduce a registered descriptor with a namespaced generator id, version/content digest,
bounded canonical parameter encoding and a native factory. A saved world chooses that descriptor
and its saved parameters, while client preferences choose the descriptor for a **new** world.
The server's world descriptor must select the same shared implementation on clients. Missing
implementations should produce a clear non-destructive load/join error.

Put the shipped Diffusion content behind this seam. Keep chunk storage, classification admission,
lighting injection, meshing, gravity-oracle consumption and save/network envelopes in core.
The generator returns a neutral layout/topology description; streaming should consume that
description without depending on a Diffusion-specific planet catalog. A flat fallback remains
useful for vanilla and tests.

Performance choices can be staged. A factory/trait boundary at world creation and batch jobs is
cheap; virtual dispatch at every sampled voxel deserves a benchmark. The builder can eventually
generate a closed enum over compiled generators and select a monomorphized implementation at
the job boundary, retaining native optimization without a runtime loader. Avoid genericizing
the entire application before proving where dispatch matters.

**First experiment:** build a third tiny generator outside the runtime, with distinct mass and
surface layout; create, save, reload, host and join its world without editing a core match arm.
Compare point/batch bytes, classification cost and far-field sampling. This is a better contract
test than another stand-in which only selects Diffusion.

## D02 — Give compiled mods an authoritative simulation and transaction context

**Necessary for machines, automation and varied multiplayer gameplay.** Client `Mod::update`
has a mutable `World`; documented direct machine edits do not automatically become network
requests. `ServerMod` currently gates edit/chat and observes membership, but has no general
budgeted simulation tick. The package builder generates game/golden binaries, not a modded
server entry point. A multiplayer machine mod therefore lacks a supported complete path.

Add an authority-side registration lane to the same compiled bundle, and generate a server
executable or an explicit `run_server(build, config)` entry point. Hosting from a modded client
must instantiate the same shared/server services. Keep presentation-only hooks on clients.

The authority context should provide reads of canonical cells/topology and submit small typed
transactions: expected revisions, origin, affected cells, material/holding transfers and wakeups.
Core validates/adopts them atomically, then performs overlay, gravity, reaction, save and network
bookkeeping. Core supplies the transaction mechanism; mods choose recipes, permissions,
machine behavior, inventory policy and frequency. Batch intents let a machine move many cells
without invoking all invalidation paths repeatedly.

Use deterministic per-system budgets and cadence, with bounded queues and explicit overload
policy. A slow optional system must not stall movement/confirmed edits or make outcomes depend
on rendering frame rate. Core reaction stepping remains its own small invariant-preserving lane.
Reuse the transaction result/reservation design to fix B01/B04/B06–B08.

**First experiment:** an external two-cell pump mod runs in singleplayer and on the headless
server, saves its work queue, and reaches identical committed states on two clients. Inject a
refused destination and verify no matter loss. Change its GUI without changing its simulation.

## D03 — Host arbitrary GUI styles through a small draw/event vocabulary

**Directly addresses the owner's GUI example.** Screens and HUD are already replaceable, but
current output is rectangles, a built-in fixed-width font and core-themed panels/roles. Menu
input is oriented around keyboard navigation. A new layout can be written; a genuinely different
image/font/mouse-driven UI cannot be expressed cleanly through that vocabulary.

Extend the renderer-neutral UI command stream with asset-backed images, font handles/metrics,
scissor clips and mod-selected colors/style values; add pointer position/button/wheel/focus events
to the screen host. Preserve ordered drawing and explicit input capture ownership. The first-party
UI kit owns widgets, skins, navigation conventions and layout; core owns event delivery and
submission of validated drawing commands.

Keep the vocabulary deliberate. Start with text/image/rect/clip rather than a full browser engine
or arbitrary GPU callback API. Rich widgets and animated skins can compile into libraries and
emit the same batched primitives. Optional path/vector drawing is an extension once a consumer
requires it. Bound asset memory and command counts at submission boundaries, and cache stable
geometry/text by screen revision so customization does not force repeated per-frame allocation.

**First experiment:** two independent UI packages implement keyboard-monospace and mouse/icon
menus against exactly the same host. Test focus/capture/settings behavior headlessly, then
measure command counts, upload bytes and presented GPU/UI costs with both.

## D04 — Name providers and durable state by package ownership

The provider arbitration rule can remain first active wins, but the selected service must have
an identity/generation independent of each implementation's revision (B13). Expose the selected
provider and any shadowed provider at registration diagnostics, so adding a texture or menu mod
has understandable results. Explicit priorities could be a later package-level feature;
dependency/topological order should not accidentally become a user's only appearance choice.

Use package id plus local mod id for persistence (B18), retaining old/unclaimed payloads (B09).
State versions should be independently migrateable without touching the wire format. Per-mod
state loading needs a context capable of resolving player/authority state in the right order;
the capacity example in B10 is one symptom of the current world-only restore hook.

This keeps discovery, naming and validation at build/registration/load time. Frame paths continue
to use indexed handles and revisions, not string lookup or global searches.

## D05 — Narrow the stable API and separate headless dependencies incrementally

The mod API re-exports broad runtime modules and even the renderer. That enables experiments,
but makes streaming/player refactors potential major API breaks. Public `World` access also
lets different operations bypass the same authority/dirty-bookkeeping rules.

Introduce stable facade types for cells, observations, topology queries, transactions, player
readings, options, UI commands and asset handles. Keep old re-exports during migration, mark
implementation-level interfaces deliberately, and reserve the next major API release for a
coherent boundary rather than incidental field changes. Ensure coordinate types needed by
public signatures have clear public construction paths.

Follow [audio-and-crate-boundaries.md](../../../documentation/audio-and-crate-boundaries.md): pure
domain/content/protocol/save data flow downward; client adapters consume them. Start by removing
Vulkan/CPAL from what the dedicated server must compile. Reuse the already independent
`material`/`field` kernels. Do not split every file into a crate; extract only when it enforces
an ownership/dependency rule or serves a second consumer. The native bundle still links all
chosen implementations into one executable and retains cross-crate release optimization.

## D06 — Reconcile the long-term mod policy and declared customization limits

The current native package model fits the owner's request, but `MOD_POLICY.md` calls it a
transitional stage before an official WebAssembly ABI. Settle that direction explicitly.
If native compiled mods remain the product, registry signing/review/reproducible builds can
describe provenance and trust without promising a sandbox. Keep package-manager and game
policy docs aligned.

Also make limits deliberate: the global 128-action bitset, fixed texture size, fixed core controls,
one selected tool, fixed law and bounded material identities should each have a stated reason
and an extension strategy where appropriate. “Endless extensibility” does not require
unbounded memory or unrestricted work; it requires that new behavior can be expressed through
bounded mechanisms without editing built-in package names.

For actions, a builder/registration-time indexed table can size a small-word bitset to the actual
build, retain allocation-free frames, and diagnose collisions. For textures, a resource/appearance
descriptor could specify resolution/format at world creation while preserving the current
32×32 fast default and device-budget checks.

## Optimization experiments, in order

| Experiment | Evidence/target | Keep only if |
| --- | --- | --- |
| O01: per-hook dispatch lists | `Mods::each_active`, `on_frame`, `on_audio`, HUD and providers scan boxed mods, including default hooks | 0/10/100-mod builds reduce CPU cycles/allocations with identical arbitration and capture semantics |
| O02: bounded LOD cut before more workers | B16 violates a claimed sizing invariant; far rules query conservative/tight bounds | Face/corner/altitude flights have bounded leaves/ancestors, full coverage and better settle/tail cost |
| O03: reaction/ledger batch commits | Each reactive pair currently invalidates cells individually; singleplayer `tick_reactions` returns a fresh vector | Per-tick buffer reuse and coalesced invalidation preserve transaction order/final gravity, reducing contact/mesh/light overhead |
| O04: shorten server lock holds | Tool/reaction/collision paths can query terrain and evaluate material under the global lock | Versioned precompute/recheck or an authority event loop lowers p99 lock/tick/ping time without race regressions |
| O05: proportional edit metadata | B15 stale indexes; server retains natural revision tombstones indefinitely | Memory/join size track current differences plus a documented bounded reconciliation history |
| O06: chunk-level immutable save data | Capture clones the entire overlay and spec strings on the main thread | Measured snapshot p99 improves at 100k/1m edits without copying whole maps on each edit |
| O07: stream samplers with bounded queues | `Sampler` uses unbounded channels/asked sets; pure answers can become irrelevant after flight | Flight RSS and stale-answer work fall while fallback queries stay nonblocking and deterministic |
| O08: optional-system build/runtime footprints | Bare/headless builds still depend on broad rendering/audio/content surfaces | Disabled/default-no-mod cost, server build deps and binary/RSS improve without sacrificing available mod capabilities |

### O01: compiled dispatch should be selective before it becomes more elaborate

Construct hook-specific arrays at registration, retaining stable mod index, package and order.
Have the registrar declare which hooks/services a type provides, or generate a registration
adapter that knows the implemented hook set. This removes calls to default no-ops without
requiring every mod type to become generic application state. A generated enum/function-pointer
plan can be compared later if the remaining dispatch cost is significant.

Measure wall-clock and cycles for idle/in-world/menu loops with 0, 10 and 100 no-op and active
mods. Include code size/instruction-cache behavior: maximal inlining of hundreds of mods can
make the fastest small build slower. Preserve `on_options` delivery to suspended mods and
capture's per-entry identity; performance changes must not subtly change arbitration.

### O04: improve the shared authority bottleneck with measured scope

Generator cache misses, tool law evaluation, swept collision and reactive work can run while
holding `State`. The server's existing instrumentation records lock-hold sites and queue depth.
Use that before an event-loop rewrite. Compute immutable generator facts outside the lock when
possible; for state-dependent results, snapshot inputs/revisions, compute, then recheck and
commit atomically. Tool policy must join the same path. Never release a lock and apply an old
product without revision verification.

At 32/128/256 clients, measure ping tails, reaction-tick drift, hold-site histograms and outbox
high-water marks under move/edit/tool/join workloads. A move check scanning 512 blocks is a
different load from stationary avatars. A single authority thread with message queues may make
ordering/transactions simpler; spatial sharding should wait until cross-boundary transaction
and reaction semantics are explicit.

### O05–O06: bound history without breaking confirmations

The client revision map and server natural-cell entries accumulate history. They are necessary
under today's revision scheme: deleting a revision while an old expected revision may still
arrive can admit a stale edit. Define an epoch/generation or acknowledged retention horizon
before reclaiming them. Do not present deletion of tombstones as a safe standalone optimization.

For saves, use immutable per-chunk edit pages or a retained snapshot/journal. Avoid making the
entire overlay copy-on-write: one tiny edit while a worker holds a snapshot would then copy the
whole map. Autosave currently follows only edit generation, so walking, changing inventory or
mod-only state can remain unsaved until clean exit. A player/mod dirty revision plus a periodic
checkpoint is a durability design improvement; it can stay outside the per-frame expensive path.

## Performance evidence to require

Follow the project's existing benchmark contract. Report average and low FPS/frame-time
percentiles, ready/settle time, RSS, queue/worker and upload/coverage counts, hardware, revisions,
resolution and exact settings. Separate application-produced frames, renderer-accepted frames
and actual presentations. At the stated 20,000-FPS aspiration, CPU loops discarded by a
latest-frame mailbox must not count as presented rendering throughput.

Compare vanilla, essentials, an intentionally UI-heavy build and a shared machine-mod build.
The important extension claim is not only “vanilla is fast”; it is that the framework adds
predictable cost as users replace or compose systems. Run traversal/cold-entry tests in addition
to the rotating-camera steady benchmark. Treat optional visual differences as controlled inputs,
not speed wins from accidentally disabling the chosen experience.

## A practical sequence

1. Repair conservation, topology, authority policy, transaction replies and save recovery.
2. Restore passing LOD/interpolation checks and add a clean-checkout test job with native deps.
3. Complete one end-to-end third generator, one authority machine mod and two contrasting GUIs.
4. Make provider/state identity explicit and retain absent-package records.
5. Benchmark dispatch, lock holds, snapshots and streaming metadata; change one measured bottleneck
   at a time.
6. Extract headless domain/server dependencies and narrow the next major mod API around the proven
   contracts.

This sequence keeps the core small by consolidating guarantees into reusable boundaries, while
the experience can grow through compiled packages.
