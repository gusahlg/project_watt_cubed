<!-- SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors -->
<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# Fundamental simulation investigation — 2026-10-13

The original reaction system is now an optional compiled mod. Four additional native mods
explore different kinds of fundamental state. Production vanilla has no contact or held-material
reaction law. The investigation deliberately leaves the choice of a final universe open.

The latest [focused continuation](focused-universes.md), dated October 14, adds two tiny
universe candidates: conserved spatial moments and relational frames. There are now fifteen
physics packages in the package-manager repository and a managed laboratory library at 0.2.0.
Read [China-affiliated research and fiction](chinese-fiction-research.md) for the inspiration
and criticism, and [the chemistry-to-universe response](chemistry-to-universe.md) for chemical
energy, lattice gases, stable matter and recoil.

The active `laws` instance now contains all fifteen providers and the existing visual Law Lab.
Use the [in-game testing guide](in-game-testing.md) for setup, exact controls, suggested
experiments and interpretation of the instruments.

The earlier [continuation](continuation.md) records the thirteen-package stage, eight additional
laws, Soviet/Russian research and the canonical managed build workflow. The first-stage launcher
and package location described in historical verification were replaced. The five-package table
and original implementation narrative below describe the first stage of the investigation.

Read [the research report](research.md) for eight substantially different model families,
27 primary-source references, five original directions, mathematical objections, architecture
costs and proposed experiments. [The extraction review](extraction-review.md) maps the previous
system's integration points and hazards. [Verification](verification.md) records the small checks
actually performed. The earlier [project audit](../2026-10-12-audit/README.md) remains separate.

## The main finding

Selective transfer in a finite closed system strictly raises a bounded internal-fit potential.
That means it must settle. It can rearrange matter, but cannot independently supply indefinitely
running clocks or recurrent signals through stationary matter. Changing its activation queue or
gain threshold does not address this limitation.

Additional state is one possible answer, but it is not automatically temperature or electricity.
The experiments compare displacement/momentum, directional packet occupancy, a finite local
potential reservoir, and a persistent routing orientation. Each remembers something that the
original configuration alone cannot remember. None uses named material recipes or authored
gate/block behavior. The common 4-D configurations still influence the laws algebraically.

## Implemented native mods

| Mod | Fundamental state beyond the shared configurations | Universal interaction being explored | Demonstrated or intended possibility |
|---|---|---|---|
| [Selective transfer](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-selective/README.md) | Arrival-ordered active contacts | Original gain-maximizing occurrence transfers/swaps | Baseline material transformation; eventual rest |
| [Integer waves](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-wave/README.md) | Integer displacement `q` and momentum `p` per active site | Equal/opposite rounded kicks, then invertible drift; composition sets inertia | Propagation, interference, recurrence; exact inverse on the tested fixed domain |
| [Conservative packets](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-packets/README.md) | Six directional occupancy bits | Spectator-controlled head-on scattering and material-dependent transport/reflection | Moving signals and nonlinear interactions without changing background matter |
| [Reservoir chemistry](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-reservoir/README.md) | Nonnegative local reservoir `e` and active contacts | Composition exchange may go uphill if the reservoir pays; integer diffusion | Reusable material states, activation and energy transport; conserved `e - Φ` |
| [Routing-memory matter](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-rotor/README.md) | A rotor orientation and nonnegative impulse count per active site | An impulse leaves by the site's current port, then advances its orientation | Matter remembers past traffic; a discovered eight-turn closed orbit |

These are experiments, not four features added to a single default law. Each has its own
factory, persistent state and optional package. Field experiments default off. The selective
provider defaults on when installed. If multiple enabled providers are installed, the first
active provider in package registration order wins. Explicitly turn the others off when comparing
models; independent simultaneous composition needs a conservation and ordering contract first.

Use the package manager's [managed laboratory](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-lab/README.md)
and [authoring workflow](https://github.com/gusahlg/pwc-package-manager/blob/main/docs/writing-a-mod.md). The packages live in
`~/repos/pwc-package-manager/mods`; `mod.toml` is canonical and the builder generates Cargo files.
The earlier custom Cargo launcher has been replaced by this workflow.

```sh
# From ~/repos/pwc-package-manager, in a disposable instance:
pwc instance create physics-wave-lab
pwc --instance physics-wave-lab mod add pwc.essentials ./mods/physics-wave
pwc --instance physics-wave-lab run --profile dev
```

In Settings, pause `pwc.physics-selective.enabled` and enable `pwc.physics-wave.enabled`.
Use one model per world/instance for comparisons. A package's enabled option pauses its law;
installation is still a build-time decision. The managed lab library provides short headless
runs and package registration checks from a generated workspace, without a second game launcher.

Singleplayer input at the player's current canonical storage cell:

| Model | Input | External source |
|---|---|---|
| Waves | Ctrl+F6 | Adds momentum |
| Packets | Ctrl+F2 | Adds one directional packet |
| Reservoir | Ctrl+F3 | Adds reservoir energy |
| Rotor | Ctrl+F4 | Adds one impulse |

Every injection is external input. It is excluded from closed-system conservation claims. The
laboratory HUD reads model state; fields do not yet change rendered geometry or player forces.

## What changed in the framework

`Mod::reaction_model()` supplies a `ModelSpec` with a stable id, compatibility version and
native factory. Each world/server owns one `Model`; its queue/fields, tool rule and tick law live
in that mod. Core retains cell storage, mutation admission, persistence envelopes, scheduling
and networking. A single virtual call per host turn dispatches into native compiled code; there
is no interpreted rule language or per-voxel search through installed mods. The cell adapter is
also a trait interface; zero dispatch overhead and production throughput are not claimed.

The static material crate and registry still provide configurations, fit arithmetic, immutable
observations, presentation and palette search. Explicit `BlockRegistry::react` calls remain a
reusable numerical utility. Production world contacts and held-tool actions invoke only the
selected provider. The old player collision/gravity systems remain unchanged. This is a seam
for replacing dynamics, not a completed replacement of every fundamental representation.

The API is 3.2. Protocol 18 compares the selected model's id/version fingerprint separately
from representation, palette and gravity fingerprints. Behavior-changing parameters, scheduling
or laws must change the compatibility version; it is a declared identity, not a source-code hash.
Integrated hosting and the lab's dedicated mode use the selected factory. The ordinary vanilla
`watt_server` executable has no provider; compile/configure one explicitly for a reactive server.

Save v10 stores selected identity and versioned payloads in reserved opaque mod records. Pausing
or switching preserves each model's payload. Unavailable providers remain frozen and retain their
required identity when saved. Old contact queues migrate only into selective transfer, and old
save files without a model marker are identified as legacy even when already settled. Unconsumed
legacy contacts survive while another model is active. Dedicated loading refuses a saved active
model/configuration mismatch. Singleplayer intentionally permits experimental switching, so use
separate lab worlds: resuming a frozen model over matter changed by another law is an experiment,
not a physically equivalent continuation or a validated migration.

## Issues addressed during extraction

The earlier audit's **B01** allowed a contact to keep one product when a second write was refused.
Both world and server adapters now admit a pair atomically. The server preflights the final
spec-pool occupancy, releases both old references, and commits both products under the same lock.
Refused products are retained as pending work for the next turn in the extracted model.

The earlier **B04** found that held-tool reactions bypassed ordinary edit-policy validation.
The server now validates the computed cell product through those hooks and rechecks the cell
revision at commit. A missing provider refuses reactive tool requests.

The extraction also identified and corrected several new integration hazards: inactive field
models being skipped merely because they have no legacy contact queue; evolving field state not
dirtying autosave; accidentally selecting a legacy fallback while temporarily removing a host;
losing an unavailable model's identity or old contact queue during save; partial hydration after
invalid records; and joining servers that use a different selected model.

## Remaining limits and design questions

**Topology is incomplete.** Cell reads now resolve storage identities through seams, but the
legacy scheduler still generates raw-grid face contacts. This does not prove complete physical
adjacency at bent/cube chart joins, and aliases can schedule the same physical pair more than
once. The field laboratories refuse chart-crossing channels because their vector/port rotation
is not implemented. A shared canonical neighbor/port transform should be the next core primitive;
silently pretending raw storage axes are physical axes is unsuitable for a universal universe.

**The domains are deliberately small.** Wave and rotor laboratories use a finite seven-cell-wide
domain around the first accepted injection; packets and reservoirs also impose explicit state
and queue caps. These bounds prevent accidental whole-world simulations. They are workbench
constraints, not claims that activity can propagate through a massive procedural world. A whole
wave phase can include hundreds of edges even though the quota counts nodes. Budget units differ
between models, so comparing operations counters as equal costs would be misleading.

**Multiplayer field replication is pending.** Servers can host native factories and persist
their state, but field snapshots, interest management and client impulse requests are not
implemented. Experimental input is singleplayer only and clients do not simulate competing
fields. A matching join identity is necessary, but does not by itself make these experimental
fields observable or playable online.

**Conservation claims are specific.** Wave inversion does not prove ordinary energy conservation
or long-term stability. Packet scattering conserves count and local momentum; reflection against
static matter can exchange momentum with that unmodeled background. Reservoir exchange conserves
its defined `e - Φ`, while its rounding rule has a tiny directional bias and proposals are neither
proven reversible nor a validated equilibrium sampler. Rotor impulses conserve count; they do
not represent physical energy or momentum. No demonstration establishes universal computation.

**The shared substrate still biases the comparison.** All nonempty configurations are solid;
worldgen palettes were searched for selective-transfer dormancy. Cross-fit routing, composition
inertia and reservoir reactions need experiments with a law-appropriate palette. No independent
fluid, mechanical bond, dynamic light or thermodynamic system has been smuggled in as a named
material exception. Intern-table growth and active-state memory remain unmeasured concerns.

## Future work that would earn a design decision

First compare equal opportunities: a propagating disturbance through stationary matter, a
closed orbit, a controllable nonlinear interaction, and a machine that repeatedly changes matter
while accounting for its input. Record failures as well as successes. Use constructions found
under the laws instead of adding special-case wire or gate types to make a demonstration work.

Then measure idle-state cost, active cells/second, registry growth, bytes per active cell,
save/resume identity and network fanout. Give asynchronous activation/partitioned updates a
mathematical contract before optimizing them; an attractive sleeping system can change its
physics when chunk loading or budget priority chooses a different update order.

The most distinct original follow-ups in [the report](research.md) are **frustrated resonance**
(composition chooses incompatible local phase preferences), a **loop-flux universe** (persistent
state on closed face/edge cycles rather than cell charges), and **chemical arrangement as a
mechanical program** (rearrangement redirects subsequent transport). They are proposals requiring
separate future mods and counterexamples, not additional laws silently combined with today's
experiments. Routing-memory matter is the first implemented original reinterpretation.

Finally put the promising laws in front of players with simple instruments. A law should let a
player predict an experiment, learn from a surprising result, and reuse that discovery. A pretty
pattern or a short mathematical proof does not establish that the fictional physics supports
enjoyable invention. No candidate is endorsed as the final PWC simulation model here.

The [continued investigation](continuation.md) adds eight more independent canonical packages
and [Soviet/Russian primary-source research](soviet-and-new-models.md). The first-stage verification
records are historical; use the continuation for the current package location and build workflow.
