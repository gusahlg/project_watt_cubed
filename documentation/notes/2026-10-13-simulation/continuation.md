<!-- SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors -->
<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# Eight more universes, and the corrected package workflow

Date: 2026-10-13. This continues the first investigation rather than choosing a final physics
model. Eight additional, mathematically distinct native mods now exist. Together with the five
previous packages, there are **thirteen separately selectable models** and a managed laboratory
library. Their physical claims remain deliberately narrower than their possible gameplay uses.

## Package correction

All packages now live under `~/repos/pwc-package-manager/mods/physics-*`. Their canonical source
manifest is `mod.toml`; none contains a Cargo manifest, Cargo lock, build script or target directory.
Every package has its own source, README, CHANGELOG, API requirement and full licence texts.
The manager resolves their declared dependencies, stores immutable content-addressed snapshots,
generates Cargo manifests and the native registration bundle, and builds the executable.
The old game-repository package directories and custom Cargo launcher were removed.

```mermaid
flowchart LR
    S[mod.toml and complete source] --> R[Package resolver]
    R --> T[Immutable store snapshot]
    T --> B[Generated Cargo manifests and registration bundle]
    B --> N[Native executable]
    N --> M[One selected authoritative model]
    M --> C[Core cells, atomic commits, saves and networking]
```

The five migrated packages preserve their AGPL code and CC-BY-SA prose; the eight new packages
and lab library follow the manager's Apache-2.0 OR MIT convention. Explicit REUSE annotations
preserve those existing licences instead of silently relicensing extracted source. The package
contract now permits native universal simulation providers, while retaining deterministic
authority and the prohibition on name-based material behavior.

Essentials was bumped from **2.0.0 to 2.1.0** and now depends on `pwc.physics-selective ^0.1`,
restoring the original contact/tool behavior in the shipped experience. Experiments remain opt-in
and default off. They are not Essentials dependencies. Package installation is a build-time
choice; each model's enabled option pauses or selects its law. If multiple enabled providers are
present, the first wins. That arbitration does not compose several fundamental universes.

The game still contains common configurations, fit arithmetic, observations, storage, commits,
authority and persistence. `src/sim/test_selective.rs` is a standalone **test-only fixture** so
core tests do not require the sibling repository; no production fallback is installed. The real
package was exercised through the manager's generated workspace and headless lab. Ordinary player
collision/gravity systems were not replaced by these experiments.

## What the new laws remember and do

| Package | Minimal dynamic state | Law | Distinct question |
|---|---|---|---|
| [physics-fuel](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-fuel/README.md) | Feedstock F, active A, spent W | Conservative transport; F+A→2A, exhausted A→W | Can finite resources produce discoverable ignition and fronts? |
| [physics-cyclic](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-cyclic/README.md) | One small excitation/recovery phase | Resting cells fire near firing neighbors; firing/refractory cells advance to rest | Can reusable excitability support signals and timing without recipes? |
| [physics-loops](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-loops/README.md) | Signed integer edge currents | Twelve disjoint plaquette phases apply fᵢ′=fᵢ−2⌊Σf/4⌋ | What interactions can respect a spatial constraint and retain circulation memory? |
| [physics-spin](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-spin/README.md) | Binary orientation s and local finite reservoir e | A spin flip pays its exact interaction-energy change; reservoir bins swap | Can energy manipulate order, domain boundaries and persistent stationary memory? |
| [physics-symplectic](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-symplectic/README.md) | Modular angle q and impulse p | Nonlinear frustrated equal/opposite kicks, then modular drift | Can finite-state reversible dynamics be richer than linear waves? |
| [physics-bonds](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-bonds/README.md) | Node displacement u, edge reststrain r, heat ledger h | Lower elastic potential by relaxing u and allowing overstressed r to creep | Can the same local rule support stress transport and learned geometry? |
| [physics-cooperative](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-cooperative/README.md) | Active contacts; shared configurations | Choose the best positive-fit single or two-occurrence transfer | Can cooperative rearrangement overcome traps without adding another quantity? |
| [physics-permutation](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-permutation/README.md) | Matching phase and domain center; actual world configurations | Symmetric composition-conditioned whole-configuration swaps on disjoint pairs | Can matter-valued reversible permutations create motion and nonlinear collisions? |

The finite laboratory domains and numeric bounds are experiment infrastructure. They are not
proposed laws requiring a seven-cell universe. No model adds a table of named wires, magnets,
springs, fuel blocks or gates. Several do not yet couple all of their quantities to composition:
fuel populations are an explicit independent resource experiment. That limitation is documented
instead of disguised as material generality. Composition-derived coefficients in spin and torus
models are frozen and saved for exact accounting/inversion; they do not follow later block edits.

New models use **Ctrl+F5** for their specific external input when selected in singleplayer. This
shared key is intentional: only the selected model consumes its action. Their README states
whether input supplies catalyst, energy, displacement or circulation, or merely activates existing
matter. None of that supplied state belongs in a closed-system conservation claim. Earlier wave,
packet, reservoir and rotor keys remain in those packages' READMEs.

## Soviet physics and mathematics: useful tools, not imported conclusions

The [new research report](soviet-and-new-models.md) traces original/primary sources through
Math-Net.Ru, Soviet Physics JETP, Physics–Uspekhi and author archives. It covers KPP fronts,
Belousov–Zhabotinsky chemistry, spin dynamics, FPUT recurrence, Zakharov–Shabat solitons,
Sinai billiards, Arnold/Zaslavsky/Chirikov dynamics and Novikov/topological currents. International
work is identified separately; unretrieved original papers are marked rather than claimed read.

Three lessons constrain the implementations:

- **A repeating visible state may consume a hidden resource.** The fuel model has finite conserved
  tokens, while cyclic excitability deliberately has no energy ledger. Its repeated-pulse potential
  must not be advertised as a perpetual chemical engine or a Belousov–Zhabotinsky reproduction.
- **Reversible is weaker than physically energy-conserving or integrable.** The torus model is an
  exact finite-state bijection. That does not establish physical momentum, canonical symplectic
  structure after quantization, or Zakharov–Shabat solitons. The spin and bond models define and
  account for specific discrete energies instead.
- **Local spatial constraints are weaker than protected topology.** The loop map preserves discrete
  divergence. A contractible loop in a simply connected cube is not protected winding. Exact
  continuum topology results do not certify the coarse laboratory.

## Concrete new findings

### A cooperative reaction crosses a true single-transfer trap

Use two configurations on two resource axes, with the other coordinates zero:

```text
A = {(0,0), (64,0)}
B = {(0,88), (64,88)}
```

The committed fit table gives `J(64)=1,048,576` and `J(88)=-181,285`. Every single transfer in
**both** directions gains `2*J(88) = -362,570`, so strict positive-gain singles cannot move.
Moving either complete cohesive pair gains `2*J(64)+4*J(88) = 1,372,012`. The implemented law
finds that move and retains all four occurrences. An initial three-element example was rejected
during review because its reverse single transfer was profitable; it did not prove a local trap.

For general cluster S, the cooperative correction is `Σ_i∈S g_i + 2Σ_i<j∈S J(i,j)`. Limiting S
to one or two occurrences keeps this experiment's scan quadratic rather than an exponential subset
search. This extends available transformations with one small law change. It still strictly raises
a bounded potential, so a finite closed assembly still settles. It does not solve persistent clocks.

### Loop currents retain a pointwise parity memory

Every passive plaquette change is an even integer `-2q`, so **each edge's parity J mod 2 is
invariant**. An odd injected loop seeds a persistent parity pattern; even fluctuations cannot erase
it. This is a direct derivation from the implemented update, not a gameplay discovery. It is
pointwise frozen parity, not protected winding or a topological computer. The ordinary Ctrl+F5
input seeds an even sector; the managed headless fixture uses an odd input, making the comparison
a concrete next experiment. Whether a player can read/use this memory remains open.

### Excitation and finite fuel both stop in the tested finite patch

In the 32-requested-tick managed demonstrations, the fuel seed becomes dormant after 28 accepted
turns and retains **2745 total tokens**. The cyclic pulse becomes dormant after 14 turns. Those are
useful finite behaviors, not proof of oscillators, reusable wires, or fuel covering every site.
The distinction between a traveling front, a reusable pulse and a sustained clock remains central.

### Plastic memory has an exact ledger but only partial mechanics

The bond model defines `V=Σ_edges(u_b-u_a-r)^2` and `E=V+Σh`. Downhill node movement relaxes
shared-edge strain; plastic creep modifies r and sends the exact loss in V into h. The focused
construction retained `E=1024=828 heat+196 strain` after six turns and stored reststrain memory.
The larger 32-turn fixture retained its prepared `E=6144`. This is interacting scalar mechanics,
not a complete vector rigid body, momentum law, or applied force on the player.

## Managed verification

The actual package-manager workflow was exercised in an isolated home:

```sh
export PWC_HOME=/tmp/pwc-physics-managed
pwc setup --pwc ../project_watt_cubed --repository ./mods
pwc instance create physics-lab
pwc --instance physics-lab mod add ./mods/physics-lab
pwc --instance physics-lab build --profile dev --print-dir
```

It resolved **14 packages**, stored their snapshots, generated crates with the canonical
`pwc_physics_*` names, and built the native game and golden-harness executables. The laboratory
is a `kind="library"` package whose 13 model dependencies are declared in `mod.toml`; it supplies
no separate game launcher. Its own tests compile it when selected from the generated workspace.
The ordinary PWC store/instances were not used. Native library paths were supplied by the existing
development environment and compilation was offline.

From that generated workspace, eight new packages' focused tests and two laboratory checks passed:

```sh
cargo test --offline -p pwc_physics_fuel -p pwc_physics_cyclic -p pwc_physics_loops \
  -p pwc_physics_spin -p pwc_physics_symplectic -p pwc_physics_bonds \
  -p pwc_physics_cooperative -p pwc_physics_permutation -p pwc_physics_lab \
  --lib -- --nocapture
```

The registration check enabled each provider alone, verified the selected factory, and checked
off. Every headless mode ran 32 requested ticks and checked **complete model-record equality**
after save/load. [managed-lab-smoke.jsonl](managed-lab-smoke.jsonl) is the raw instrument output.
Earlier agent work used temporary development wrappers outside all package source directories;
the generated managed workspace is the final integration evidence.

| New mode | Accepted turns | Pending state/work at tick 32 | Reserved record text bytes |
|---|---:|---:|---:|
| fuel | 28 | 0 | 4658 |
| cyclic | 14 | 0 | 3288 |
| loops | 32 | 429 edges | 5166 |
| spin | 32 | 343 sites | 4651 |
| symplectic | 32 | 343 sites | 7934 |
| bonds | 32 | 343 sites | 6767 |
| cooperative | 32 | 0 | 111 |
| permutation | 32 | 343 domain cells | 116 |

These text byte counts exclude actual material, binary save overhead and container memory.
Counter and quota units differ. Inputs differ substantially. This table is not a speed or quality
ranking. Package validation, formatting, root compilation and whitespace checks are recorded in
[the continuation verification](continuation-verification.md). No large-world, GPU gameplay,
full project suite, equilibrium sampler, universal computation or networked-field trial was run.

## Architectural scrutiny and next directions

The models dispatch natively once per authority turn; there is no interpreter or per-voxel mod
list fanout. The field prototypes nevertheless use ordered maps, full-phase scratch copies and
small-domain scans for clarity and bounded rollback. Several retain an active domain even when
its fields appear settled. Cold-region sleeping, packed storage and aggregate HUD caches need
semantic tests before production optimization. An algorithm that changes its physics with chunk
loading, quota priority or map iteration is unsuitable for this project.

Physical neighbors and direction transforms across charts remain a core missing primitive.
Experimental fields refuse unsupported seam channels. Authority factories and saved identities
exist, but field replication and network impulse requests do not. Installing the same package
on a client/server does not make these fields observable online.

The most promising combination to investigate next is **stored structural memory redirecting a
conserved disturbance**: learned bond strain, rotor routing or odd loop parity could alter packet
or wave transport under one explicit accounting law. Another is **spin order changing chemical
compatibility while chemistry pays spin energy**. These combinations should be new independent
packages with a combined invariant, not two independently ticking laws glued together. The
[research report](soviet-and-new-models.md) includes five such original directions and criticisms.

Before choosing any model, ask a player to construct a repeatable experiment, discover a reusable
interaction, explain an unexpected result, and build one useful machine. Prefer failed candidates
with clear reasons over silently adding hardcoded exceptions to make a demonstration succeed.
