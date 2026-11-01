<!-- SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors -->
<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# Two tiny fictional universes — focused continuation, 2026-10-14

Two additional native mods are installed in the canonical package-manager repository:
**dipole occupation** and **relational frames**. Each explores one unusual possibility
with a small universal law. Neither extends the default selective chemistry or combines
the earlier experiments. There are now fifteen physics packages and the managed
`pwc.physics-lab` library is version **0.2.0**. Its sixteen selectable modes include `off`.

The [China-affiliated research and fiction investigation](chinese-fiction-research.md)
records the sources, candidate selection and theoretical objections. The
[strange-universe comparison](strange-universe-candidates.md) examines relational identity
and a deferred changing-adjacency idea. The new
[chemistry-to-universe response](chemistry-to-universe.md) addresses chemical energy,
lattice gases, stable matter and recoil. These are three complementary research notes;
this report records the implemented outcome and the actual limited checks.

## 1. Mobility is a consequence of conserved spatial moments

Package: [pwc.physics-dipole](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-dipole/README.md).
Each site remembers only `q ∈ {-1,0,+1}`. Define

```text
Q = Σ q_i                    signed occupation
D = Σ r_i q_i                Cartesian first spatial moment
N = Σ q_i²                   quadratic norm / occupied-site count
```

For a local geometric vector `v` satisfying `Σv_i=0` and `Σr_i v_i=0`, apply

```text
a = -2(v·q)/(v·v)
q' = q + a v.
```

Accept the complete reflection only when `a` is integer and every new occupation remains
ternary. Otherwise the stencil rests. Collinear triples use `(1,-2,1)` and square faces
use `(1,-1,-1,1)`. These are two geometric instances of the same equation, rather than
a particle-pattern collision table. The map is an involution on its admitted sector.
Disjoint stages and a stored phase provide a deterministic invertible closed evolution.

**Why it is promising.** With `N=1`, there is exactly one occupied site. `Q` fixes its
sign and `D` fixes its position. It cannot move. A neutral pair can translate because
translation changes `D` by its total charge times the displacement, which is zero.
The implemented reflections actually realize both longitudinal and transverse pair moves.
Anchors and mobile composites therefore follow from the same law and different states.
No named immobile particle, wire, or transport recipe is introduced.

This suggests a universe in which access to companions unlocks motion and spatial moment
is a manipulable constraint. An isolated occupation might store a location; composites
might move information while retaining a geometric identity. These are opportunities to
investigate, not demonstrated technology or universal computation.

**The essential criticism.** Spatial-moment conservation alone does not immobilize every
charge. Other excitations can compensate for its movement. The demonstrated isolated-unit
claim also uses the norm and ternary capacity. Removing that capacity permits
`[3,0,0] → [2,2,-1]`, with unchanged charge, moment and norm, but different support.
Likewise, preserving a neutral pair's separation is not attractive binding or a solid.
The norm is not a validated kinetic or thermodynamic energy. Coupling it to conventional
momentum, chemistry or gravity could overconstrain the world.

The inspiration includes China-affiliated research on dipole-conserving and plaquette
systems. Its statistical and quantum results are not claimed for this integer automaton.
The bounded affine-null reflection is our independent candidate, justified by its algebra.

## 2. Matter is an operation, and encounters change internal frames

Package: [pwc.physics-frames](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-frames/README.md).
Each site stores a permutation of three abstract markers. The six states form `S₃`,
the smallest noncommutative permutation group. Composition and inverse use three indices;
there are no material-specific rules or prewritten oscillator patterns.

On an ordered neighboring pair, with `ab=a∘b`:

```text
if ab = ba:  rest
otherwise:   (a,b) → (b,b⁻¹ab).
```

An active encounter preserves its ordered product and the conjugacy-class populations.
Its inverse is `(c,d) → (cdc⁻¹,c)`. The eligibility condition is unchanged by the
encounter, so the quiet branch retains bijectivity. A fixed 1-D chain preserves its
ordered product; a global ordered product is not asserted for the 3-D laboratory.
Identity sites are fixed individually, which creates a strong support restriction.

**Why it is promising.** Every uniform frame and every commuting neighboring mixture
rests. A noncommuting disturbance can change the same otherwise stationary medium.
With three transpositions `A=(01), B=(12), C=(02)`, the same equation gives

```text
(A,B) → (B,C) → (C,A) → (A,B).
```

Two encounters restore carrier order while changing internal frames; the third closes
the orbit. Encounter history changes identity without adding a separate routing memory
or clock rule. This is meaningfully different from swapping immutable configurations.
Quiet preparations, nonlinear disturbances and small recurrences arise together from
one compatibility test and one algebraic update.

Potential player questions are whether a prepared strand carries a readable disturbance,
whether a junction records encounter order, and whether a local recurrence can drive a
repeatable output. The present demonstration establishes propagation and recurrence,
not reliable signals, gates, amplification or useful machines.

**The essential criticism.** Identity gaps cannot be crossed and nonidentity support
cannot grow or move. This is internal information dynamics in prepared stationary media.
There is no energy or ordinary momentum ledger. Ordered pairs and the six matching
phases introduce directional bias. Six states might yield useful memory or merely noise.
The commutation gate intentionally breaks the Yang–Baxter relation: **48 of 216 triples**
fail. Calling the mod a quantum anyon model, gauge theory or protected topological memory
would borrow results it has not earned.

China-affiliated braiding research motivates asking whether encounters can transform
internal information. Greg Egan's *Schild's Ladder* motivates asking which alternative
stable substrates a law can sustain. Neither validates the implemented classical rule.
Primary sources and the distinction between inspiration and proposal are in the linked
research notes and package README.

## 3. Canonical packaging and independent gameplay preparation

Both packages use `mod.toml`, native `register()` entry points, complete licence texts,
README and changelog. No package ships Cargo manifests, lockfiles or build output. The
manager snapshots source and generates the Rust workspace. Both experimental options
default **off**; the installed selective provider still defaults on. One enabled provider
is selected by registration order. Multiple enabled providers do not compose their laws.
No new mandatory quantity or law was added to core in this focused round.

Each first input prepares a bounded **7×7×7** field laboratory at the player's current
canonical cell. This is explicitly external preparation, not spontaneous creation by
the law. The experiments share Ctrl+F5; only the selected provider consumes its input.

| Model | Preparation | Instrument / cost boundary |
|---|---|---|
| Dipole | Ctrl+F5 neutral pair; Ctrl+F6 positive unit; Shift+F6 negative unit | Local occupation, charge, norm, x moment and counters; at most 161 stencil attempts per turn plus scans and 343 host reads |
| Frames | Ctrl+F5 conjugation; Ctrl+F6 cycle local permutation | Local rank, conflict edges, changed pairs and turns; 343 host reads, at most 147 pair attempts and 882 edge checks |

The quota units differ: dipole admits a complete stencil schedule, while frames charges
a whole phase by domain sites. Counters are not equal-cost benchmarks. Both use ordered
maps and bounded scans for reviewability. Frames sleeps when all edges commute. Dipole
currently scans its prepared domain even for a frozen unit; an activity cache is a future
optimization requiring proof that it preserves phase semantics.

They store field state separately from material `BlockId`s. Input, numeric HUD instruments
and save/resume exist. Field rendering, material motion, player forces, network field
replication and multiplayer preparation requests do not. Unsupported chart aliases and
unreadable cells refuse whole phases; transformed physical neighbor ports remain a core
architectural requirement. Finite patch boundaries prevent whole-world scalability claims.

For an isolated managed instance, follow the
[laboratory README](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-lab/README.md). In its generated
workspace, focus only these experiments with:

```sh
PWC_PHYSICS_MODEL=dipole,frames cargo test --lib \
  -p pwc_physics_dipole -p pwc_physics_frames -p pwc_physics_lab -- --nocapture
```

The lab filter now accepts comma-separated names and rejects unknown names, avoiding a
misspelled filter silently passing without running a demonstration. Use distinct worlds
and explicitly disable competing providers when comparing laws in gameplay.

## 4. Evidence collected, with its limits

An isolated manager home at `/tmp/pwc-physics-focused` resolved **16 packages** and built
the actual native runtime. Its generated workspace was
`/tmp/pwc-physics-focused/cache/builds/e38777c3acbba450a0d3023441c1a382`.
The game target cache was reused; no personal manager instance was modified.

Four focused tests passed:

1. Dipole: an isolated unit remains unchanged for 36 phases; a neutral pair translates
   longitudinally and transversely with exact `(Q,D,N)=(0,[-1,0,0],2)`, then reverses.
   Its separate managed demonstration below covers save reload. Corrupt-load rejection
   was reviewed, rather than exercised by this mobility test.
2. Frames: all 36 pair combinations preserve product/classes and invert; 216 triples
   establish the 48 Yang–Baxter failures; the three-encounter recurrence is checked.
   One disturbance changes **28 sites** after six phases, and reversing those phases
   exactly restores the preparation. Save reload and corrupt-state rejection are checked.
3. Managed registration: all fifteen providers and `off` select exactly the requested
   factory through package entry points and options.
4. Managed short runs: each new model runs 32 host ticks and passes exact **complete
   reserved-record** save/load equality. Ten JSON readouts were retained.

At tick 32, dipole recorded 11 changed stencils; frames recorded 856 changed pairs.
No preparation or phase refusal occurred in these two fixtures. Serialized model-record
text was 3,281 and 4,659 bytes respectively, excluding binary envelopes and map overhead.
These different counters and preparations establish activity, not comparative performance.

Evidence: [test output](focused-tests.txt), [readouts](focused-lab-smoke.jsonl), and
[canonical package checks](focused-package-checks.txt). All **35** canonical repository
packages passed validation; rustfmt and both repositories' whitespace checks passed.
No broad benchmark, whole game suite or gameplay study was run in this focused round.

An independent final read-only review of both installed implementations found no remaining
concrete issue in the scoped invariants, whole-phase admission, bounded inputs, transactional
loading or provider identities. This does not replace future architectural and gameplay tests.

Review also corrected a persistence edge case before installation: a model must pause
before committing clock `u64::MAX`, because its loader rejects that exhausted value.
Previously allowing that last transition would produce an unloadable save. Frames now
reports rank `-1` outside a prepared patch instead of pretending the absent state is a
particular frame. These are code-review corrections; the short demonstrations do not
exercise the enormous clock horizon.

## 5. The chemistry-to-universe question changes the next useful experiment

Your latest proposal asks for a stronger foundation than conserved fields alone. Internal
compatibility supplies `U=-kΦ`, but does not change when intact neighboring configurations
separate. Thus it gives reaction energetics without spatial cohesion. Reversible local
states, fixed mirrors and conserved particle count also do not automatically supply a
solid, temperature distribution or mechanical recoil.

The [response](chemistry-to-universe.md) evaluates FHP lattice gases, microcanonical
demons, reversible aggregation, conservative computation, momentum-accounted soft-sphere
models and China-affiliated energy-conserving mesoscale particles. It records exactly
which pieces each source supplies and which remain missing.

One future, separate mod could investigate the minimal ledger

```text
E = Σ e_i + ε N_packet - k Φ.
```

Generic reactions release or consume the same resource that opposite packet pairs carry;
absorption can pay for an otherwise unavailable rearrangement elsewhere. This gives a
specific conversion experiment instead of adding a familiar phenomenon by name. It still
needs an invertible event selector if reversibility is claimed, and it would not yet supply
cohesive moving matter. Recoil, position-dependent interactions and velocity states must
earn their cost through separate evidence. This bridge is documented, **not implemented
as a third new model** in this round.

The two implemented candidates illuminate different aspects of a possible fictional
universe: constraints can decide mobility, and transformations can make stationary media
dynamic. Their next validation should be player-constructible, repeatable experiments
under their existing laws. Adding bespoke machinery to rescue them would defeat the
reason for investigating these foundations.
