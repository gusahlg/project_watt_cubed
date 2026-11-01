<!-- SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors -->
<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# Reaction extraction: the complete authority boundary

This review describes the code read on 2026-10-13 before the simulation extraction in this
directory's accompanying changes. It is an architecture review, not a claim that a new model
has survived gameplay, performance or multiplayer validation. No core code was changed by the
review itself. The references name integration points even when the accompanying extraction
subsequently changes their line numbers.

## What the existing model actually contains

The material representation is a canonical multiset of at most 32 four-byte elements. A world
interns each distinct configuration into a `BlockId(u16)`. The registry retains immutable
`material::Block` records with cached pair supports, presentation and physical observations;
voxels contain only the small id. Empty configurations are air. Every nonempty configuration
is currently solid, and hardness, transparency, emission and grip are immutable readings of a
configuration. These are useful optimized storage and observation choices, but they also limit
what a reaction-only replacement can demonstrate.

The passive law is [selective transfer v1](../../../guides/reaction-guide/selective-transfer-v1.md).
It exchanges occurrences when internal pair compatibility increases beyond a strict threshold.
Its bounded, monotonically increasing potential proves that a finite, closed unforced system
eventually stops. It cannot independently provide perpetual oscillation, reversible signals or
transport of an additional state through stationary matter. A larger queue or a new threshold
does not remove that mathematical limitation.

`BlockRegistry::react` is an execution policy, not merely interning: it clones two cached blocks,
calls the hardwired `material::react_once`, and interns both outputs. `quiescent` similarly
consults the same fixed law. The optimized records and numerical kernel can remain reusable
libraries while their use in gameplay becomes optional. Moving only the `Reactions` tick struct
to an external crate would leave the law selected by core and leave tools reactive in vanilla.

## Integration map

| Path | Existing responsibility | Required extraction treatment |
|---|---|---|
| `src/sim/reactions.rs` | Canonical raw-grid face contacts; age-ordered deduplicated queue; fixed quota; one material step per edge | Separate selected law from scheduler mechanism, or let a mod own the complete model instance; no model must produce no work |
| `src/world/mod.rs`, `src/world/edits.rs` | A local world owns a scheduler; material/move notifications wake it; it ticks and restores saved contacts | Carry the selected model once per world; make wake, tick and restore honor its absence |
| `src/game.rs` | `Game::new` unconditionally installs the reaction fixed-tick system; attaching a connection removes local authority | A neutral host tick may remain; it must execute only the selected authority model |
| `src/game/sync.rs` | Local held-tool actions call `registry.react` directly and update inventory after a cell commit | Use the same selected law as passive contacts; no-model tool use must be a no-reaction outcome |
| `src/net/server/state.rs` | The server separately owns a scheduler; `ServerCells` serves generator/overlay reads and fallible writes | Install the selected model into this authority too; a client-only hook cannot replace server simulation |
| `src/net/server/edit.rs` | Accepted edits and tools wake the server queue; held tools call `registry.react` directly | Route both paths through the selected model and preserve validation/admission semantics |
| `src/net/server/tick.rs` | 20 Hz thread; skips the reaction tick when `pending()==0`; coalesces committed cell snapshots | Preserve bounded mutation fanout; an oscillator or field model needs its own work/liveness condition |
| `src/net/server/mod.rs` | Creates registry/generator/state from loaded world; restores contacts before thread startup | Select the model before restoring work; do not assume `Config::default` installs a first-party mod |
| `src/app/entry.rs` | New/load/join worlds are built on a background thread from a copyable `Recipe` | Capture a native factory/descriptor; do not send the UI-bearing `Mods` collection to this thread |
| `src/app/host.rs` | Integrated hosting uses a server `Config` independently of the local `World` | Pass the same compiled model descriptor into server configuration |
| `src/save/bridge.rs` | Captures world contacts; `Restored::finish` restores mod state first and contacts second | Resolve/check the model before restoring its work; keep restore independent of chunk loading |
| `src/net/persist.rs`, `src/net/server/save.rs` | Dedicated server loads/saves contacts using the same save document and preserves other mod records | Include model identity/state in this path, not only client `Mod::save_state` |
| `src/net/mod.rs`, `src/net/protocol.rs` | Core law/palette fingerprint and fixed material stamp gate compatibility | Add selected simulation identity; existing `Law::current()` does not identify an optional new dynamics mod |
| `src/world/streaming/gauges.rs` | Reads queue counters for diagnostics | Give absent/experimental models neutral or truthful counters; never report a sleeping legacy queue as active physics |

Direct legacy-law invocations in tests, palette search and numerical workbenches are useful and
may remain explicit. They must not be confused with the game's selected simulation model.

## Smallest practical complete seam

Keep core responsible for canonical matter ids, world reads, mutation admission, overlay
bookkeeping, gravity changes, rendering invalidation, saves and network envelopes. Give a
compiled package a factory/descriptor for an authority-side model. Resolve selection at world
creation and server startup, then retain that descriptor or instance instead of searching the
mod list for each cell. There is no runtime loader, interpreter or per-voxel mod-list fanout.

The first-party legacy mod explicitly supplies the selective-transfer implementation, scheduler
constants and identity. Vanilla supplies no model. Each experimental package supplies a different
descriptor. For mutually exclusive fundamental models, report multiple selection explicitly;
silently taking the first package risks testing the wrong universe. Separately composable physics
systems require an explicit composition order and conservation/accounting contract.

An additive `Mod` factory/descriptor hook with a default `None` is a small API change. At review
time `pwc-mod-api` is 3.1.0, and existing example manifests accept `^3.0`. Removing public
`sim::reactions` types or changing their required methods/signatures can instead break compiling
mods and requires the documented major-version decision. Keeping legacy methods as explicit
library utilities while gameplay stops invoking them automatically avoids unnecessary breakage.

A function pointer for one pair operation is enough for another composition-exchange law. It is
not a universal simulation interface: a field model needs per-position state, tick evolution,
save data and possibly a held-tool interaction that cannot be represented by two configurations.
Preserve that distinction in package READMEs. A standalone experiment is still useful, but label
it as a workbench until the world, server and persistence actually host its additional state.

## Identity, saves and disabling

`material::Law::stamp` combines fit-table identity, capacity, observation probes and visual seed.
Those inputs still affect existing registry observations and the generated palette even when
passive reactions are off. Replacing this stamp with a new simulation id would conflate state
encoding/appearance with dynamics and could needlessly invalidate old material specs. Prefer a
separate namespaced model id plus an implementation/version/parameter digest. Include the logical
tick rate, quota and scheduling policy in the digest when they change outcomes.

Current saves contain pending raw-grid contacts and ages but no separate model id. Old saves
implicitly belong to the legacy dynamics. Restoring those contacts into an unrelated law changes
their meaning. A safe loading policy identifies the legacy interpretation, resolves the required
compiled implementation, and refuses a missing or incompatible model without replacing the
world. If the game supports inspecting such a world with simulation disabled, preserve its
unresolved model state and pending work; do not save away that data silently. An intentional
conversion between universes should have an explicit migration policy rather than happen on
every option change.

The current dedicated save path does not instantiate ordinary client mods, so a marker written
only by `Mod::save_state` is insufficient. The authority must validate and persist its own model
identity. A bounded reserved mod record can avoid a binary save-envelope change for an initial
implementation; the client bridge and dedicated store must both handle the same record. Unknown
model records should be retained.

Mod package listings and allow/deny policy identify installed packages, but do not prove that the
same dynamics are selected with the same parameters. Join compatibility must compare the active
model explicitly. Decide whether clients must compile the server's model for deterministic
local tooling/observations or may act as presentation-only receivers; the protocol should make
that decision, not silently rely on the unchanged core material fingerprint.

Compile-time omission already supports model-off builds. If experiments additionally expose an
option to suspend simulation in one build, freeze/retain state and check whether the option is
world configuration or session pause. Loading chunks or resuming presentation must not fabricate
activation events.

## Existing hazards worth keeping separate from model research

The prior [audit](../2026-10-12-audit/bugs.md) already has reproductions for these paths:

* **B01: non-atomic pair commit.** The scheduler writes the first cell, then the second. The
  server's real spec pool can refuse the second write, and the existing probe observes 8 to 7
  occurrences. New models need pair/transaction admission before mutation. Rollback through
  public writes is not equivalent because revisions, gravity and dirty work have already changed.
* **B03: contact topology.** Contacts use storage address `±1`, while visible chart neighbors can
  lie in a different storage patch. Reading/writing real endpoint pairs and canonicalizing seam
  identity is a host concern. Retaining old adjacency is an honest legacy baseline; claiming
  universal physical face contacts would be inaccurate.
* **B04: tool permission bypass.** Server tools do not consult ordinary edit-policy hooks. Merely
  delegating reaction math into a mod does not fix protected-cell validation.
* **Queue loss at capacity.** The bounded scheduler counts refused wakeups and drops them. A
  reversible field or machine may need retained backpressure rather than permanent dormancy.
* **Panic behavior.** Local ticking temporarily removes its queue without a recovery guard. The
  server restores the scheduler after a panic, but commits made before a panic are not necessarily
  broadcast by the interrupted turn. A general mod callback cannot assume panic isolation makes
  a partially committed tick transactional.

Generated materials are searched specifically for selective-transfer dormancy and reagent
selectivity. Testing a new law with that palette is a useful controlled comparison, not proof
that the new law's natural world is stable. Do not bake named-material exceptions into a new
model to repair this mismatch.

The material guide has historical details that need updating with the extraction: it says tools
are stepped to rest, while `src/game/sync.rs` and `src/net/server/edit.rs` perform one operation;
its save/protocol version examples also predate the current implementation. The source is the
authority for the integration map above.

## Minimal checks for this change

The user asks for little testing now. The highest-value small checks are build compilation plus
one vanilla no-reaction check, one legacy two-cell conservation step, and one authority
selection/identity save round trip. Exercise a held tool as well as a passive contact, because
they currently bypass different parts of the scheduler. Broader physics claims should wait for
model-specific measurements and gameplay experiments; no such claims follow from extraction.

## Network integration accompanying the extraction

The server now takes the same optional native descriptor through `Config::reaction_model`, with
no provider in production defaults. Its tool path delegates to that selected instance, and the
neutral 20 Hz driver ticks enabled models even when a contact queue is empty. Clients receive
committed material edits and never run a competing authority model.

Protocol 18 adds an explicit simulation fingerprint to `Hello`. The fingerprint is the selected
descriptor's namespaced id/version hash; zero means simulation is disabled. The additive
`Connection::connect_with_model` and `begin_connect_with_model` methods carry the selection.
Older constructors select no model in production. Core test fixtures explicitly retain the
historical model through a test-only default.

Dedicated world snapshots carry the same reserved `@simulation.selected` and
`@simulation.state.<id>` records as client saves. Loading an actual old document without a
selection marker assigns the legacy id/version, including a settled document with no contacts;
new worlds bypass that migration. A server refuses an unavailable/different saved selection
before starting its simulation threads. Its error is a save-load error so integrated hosting can
identify an incompatible slot while retaining its file.

`ServerCells::resolve` caches chart seams, and `set_pair` checks both current input blocks and
the final spec-pool occupancy before replacing either ledger cell. Releasing the pair's old
references as part of admission allows a name that genuinely becomes unused to make room for a
new product at the capacity boundary. Refusing admission leaves both cells and revisions intact.
The held-tool path presents its computed cell product to edit-policy hooks outside the state
lock, rechecks the input revision, and treats a lost essential `ToolResult` as a connection
failure requiring a fresh join rather than silently dropping the result.

Four focused checks were added for disabled simulation, protected tool targets, atomic pair
admission at the real spec-pool limit, and mismatched-model joins. Compilation and execution are
recorded in the work's verification notes once the complete shared change is assembled.

The experimental **dynamic fields remain authority-side state**. Material changes are visible
through the existing overlay transport; field values, perturbation inputs and field HUDs do not
yet have a multiplayer snapshot/request protocol. This is an explicit prototype limitation. A
client may compile the same law to satisfy compatibility without being able to inspect or inject
its server's auxiliary fields. A future bounded field snapshot/input seam is needed before those
experiments offer their complete multiplayer workbench. The server still trusts client holdings
as the previous tool/placement protocol did; disconnecting after a lost result does not turn this
into an authoritative inventory ledger.
