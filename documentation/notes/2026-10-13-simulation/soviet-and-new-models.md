<!-- SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors -->
<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# Soviet and Russian mathematics: inspiration for eight more simulation models

Date: 2026-10-13. Research and design criticism only; this note edits no runtime model and
claims no new PWC benchmark or gameplay result. The proposed families below should remain
independent native mods. Their names describe experiments, not laws that must enter core.

**The useful direction is to search different kinds of memory and change:** consumable
activation, cyclic state, oriented spin, reversible internal maps, connected geometry,
collective rearrangement, and conserved spatial constraints. Eight folders with different
coefficients in the same diffusion equation would provide much less evidence.

This extends [the broad research notebook](research.md). Statements labelled **Source result**
describe the cited work. **Derivation** is an argument given here. **PWC proposal** is our
own candidate law, not an established result of the cited physicists. Claims proved for a
continuum equation do not transfer automatically to a coarse integer implementation.

The search used Math-Net.Ru's Steklov Institute archive, the Soviet Physics JETP and
Physics–Uspekhi archives, author-maintained publication collections, and original papers.
The original 1937 KPP text and the full 1935 Landau–Lifshitz paper were not directly retrieved;
their provenance and relevant mechanisms are cross-checked through the primary archive works
identified below. Original publication dates and translation dates are distinguished.
Zabusky–Kruskal and Fermi–Pasta–Ulam–Tsingou are included as relevant international work, not
incorrectly classified as Soviet work. This is a scientific comparison, not a nationality ranking.

## 1. What the primary sources actually establish

### Kolmogorov–Petrovsky–Piskunov: a front can select its own propagation speed

**Source result.** Barenblatt and Zel'dovich discuss the KPP problem as an example of a process
that loses details of its starting shape while remaining far from equilibrium. They state
conditions on the scalar reaction term and explain selection of the minimal traveling-wave
speed. Their combustion comparison distinguishes a monostable growth front from an ignition
threshold. Galaktionov's later research examines front shifts and higher-order variants.
[Barenblatt and Zel'dovich, Intermediate asymptotics, 1971](https://www.mathnet.ru/php/getFT.phtml?jrnid=rm&paperid=5188&what=fullteng),
[Galaktionov, KPP problem and front shifts, 2013](https://www.mathnet.ru/php/archive.phtml?jrnid=tm&option_lang=eng&paperid=3500&wshow=paper),
[Kanel', combustion stability, 1962, original archive record](https://www.mathnet.ru/php/archive.phtml?jrnid=sm&option_lang=eng&paperid=4701&wshow=paper).

A representative equation is

```text
∂u/∂t = D ∇²u + r u(1-u).
```

**Derivation, not a discrete game prediction.** At the low-density edge, substitute
`u ~ exp[-λ(x-ct)]` into the linearized equation. Then `c = Dλ + r/λ`, whose minimum for
positive `λ` is `2 sqrt(Dr)`. The complete continuum selection theorem needs assumptions on
reaction and initial data; this short calculation is not its proof. A discrete threshold
automaton has a different front speed, can pin at small amplitudes, and may be anisotropic.

**PWC implication.** One simple law can erase unnecessary preparation detail and produce
repeatable macroscopic behavior. This is valuable for discoverability. But the region behind
an invasion front has changed: it is not automatically a reusable wire. Finite fuel, recovery,
and repeated transmission are separate questions.

### Belousov–Zhabotinsky: feedback plus supply, rather than a magical color cycle

**Source result.** Zhabotinsky's original 1964 paper reports liquid-phase oscillations and a
mechanism in which an accelerating reaction leads, through later reactions, to its own
inhibitor. It discusses slowly changing starting concentrations as a quasi-open regime and
external reactant supply for continuing oscillations. Zaikin and Zhabotinsky subsequently
report concentration-wave propagation in an unstirred two-dimensional medium. These are
chemical experiments, not merely simulations with cyclic labels.
[Zhabotinsky, 1964, Math-Net original paper and PDF](https://www.mathnet.ru/eng/dan29852),
[Zaikin and Zhabotinsky, concentration waves, 1970](https://www.nature.com/articles/225535b0).

**PWC implication.** A return to a previous visible color can accompany consumption of a
hidden resource. A model should distinguish state that cycles from resource that is spent.
An inhibitor or slow recovery degree of freedom earns its place by enabling repeated pulses
and an observable refractory interval. A rule `phase = phase+1 mod 3` may be an interesting
fictional universe, but it does not reproduce the above mechanism.

### Landau–Lifshitz: orientation can change while matter stays put

**Source result.** The 1935 Landau–Lifshitz paper's original bibliographic identity is recorded
by CERN. Bushey's 1970 JETP paper explicitly uses effective-field spin dynamics to derive
waves in a domain wall, assuming fixed magnetization magnitude and neglecting damping.
Ivanov and Stefanovich's 1986 work studies precessing topological magnetic solitons and finds
that seemingly small higher-gradient terms can substantially change small-radius behavior.
[Landau–Lifshitz original publication record](https://cds.cern.ch/record/437299),
[Bushey, spin waves in a Bloch wall, 1970](https://www.jetp.ras.ru/cgi-bin/dn/e_031_03_0535.pdf),
[Ivanov and Stefanovich, small-radius magnetic solitons, 1986](https://www.jetp.ras.ru/cgi-bin/dn/e_064_02_0376.pdf).

A modern normalized representative is

```text
h_i = -∂H/∂s_i,  |s_i|=1
ds_i/dt = -s_i × h_i - α s_i × (s_i × h_i).
```

**Derivation.** Both terms are perpendicular to `s_i`, so the continuous equation preserves
`|s_i|²`. The precession term is perpendicular to `h_i` too and conserves `H` in an autonomous
closed model. The damping term lowers `H` for `α>=0`; it requires a destination for dissipated
energy if a later game law insists on tracking all energy. Explicit rounding of the vector
update preserves neither statement automatically.

**PWC implication.** Spin is inspiration for an orientation variable and its dynamics, not
an obligation to implement electrons, magnetism, or electromagnetism. A compass automaton
with four or six headings is also a valid fictional model, but has different state and laws
from a unit-vector precession lattice.

### FPUT, Zabusky–Kruskal, and Zakharov–Shabat: nonlinearity can preserve useful wave identity

**Source result.** The original Los Alamos report explores nonlinear chains and reports
unexpected behavior instead of the straightforward spreading among modes its authors
anticipated. Its credits include Tsingou's work. Zabusky and Kruskal numerically observe
KdV pulses recovering their shape after interaction. Zakharov and Shabat solve the focusing
one-dimensional nonlinear Schrödinger equation through inverse scattering; asymptotic
soliton amplitudes and velocities survive interactions while positions/phases shift, and
bound states are possible under the stated conditions.
[Fermi, Pasta, Ulam, Tsingou, original LA-1940 scan, 1955](https://www.egcharalampidis.com/links/ACC0041.pdf),
[Zabusky and Kruskal, 1965](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.15.240),
[Zakharov and Shabat, Russian original 1971 / English JETP 1972](https://jetp.ras.ru/cgi-bin/dn/e_034_01_0062.pdf).

Representative equations:

```text
KdV:    u_t + u u_x + δ² u_xxx = 0
NLS:    i ψ_t + ψ_xx + κ |ψ|² ψ = 0
```

**PWC implication.** A signal could be a self-organized packet instead of a marker that is
manually routed. Yet 1-D integrable dynamics does not imply the same robust packets in a
3-D branching world. Strongly elastic collisions can also make useful switching harder:
interaction may provide timing or phase changes without a robust logical output. A bump
surviving ten ticks is not evidence of a soliton collision law. Dispersion, nonlinearity,
dimensionality, resolution, and boundary conditions must all be stated.

### Sinai: simple geometry can make conservative motion mix

**Source result.** Sinai proves ergodic/K-system behavior for a class of dispersing billiards
with precise geometric assumptions. The rule between collisions is free motion with elastic
reflection; the geometry matters to the result. A box made of flat voxel faces is not the
same dispersing geometry.
[Sinai, original 1970 Math-Net record](https://www.mathnet.ru/php/archive.phtml?jrnid=rm&option_lang=eng&paperid=5322&wshow=paper).

**PWC implication.** Very simple collision laws can make rich mixing chambers, traps, and
geometry-dependent transport. Chaos is not a substitute for signal quality: a mixer may be
excellent for processing matter and poor for retaining information. Finite lattice velocities
and quantized collision boundaries can produce periodic or locked trajectories that do not
inherit the continuum theorem. Static obstacles also receive momentum in real mechanics;
an immutable external wall is an explicitly chosen boundary approximation.

### Kolmogorov, Arnold, Zaslavsky, and Chirikov: regularity and chaos coexist

**Source result.** Arnold's work on small denominators and Hamiltonian stability develops
conditions under which regular motion persists under perturbation. Chirikov and collaborators
study stochastic behavior from deterministic nonlinear dynamics and resonance overlap,
using analysis and numerical experiments. Their basic standard-map example alternates a
kick and a drift. This research supports studying parameter regions, rather than assuming
that weak coupling always means regularity or strong coupling always means useful complexity.
[Arnold, original 1963 stability paper](https://www.mathnet.ru/php/getFT.phtml?jrnid=rm&paperid=6441&what=fullteng),
[Zaslavsky and Chirikov, UFN 1971 / translation 1972](https://www.mathnet.ru/eng/ufn10843),
[Chirikov, original 1979 review](https://www.sciencedirect.com/science/article/pii/0370157379900231),
[Chirikov, Izrailev, Shepelyansky, original 1981 text](https://www.quantware.ups-tlse.fr/chirikov/refs/chi1981a.pdf).

```text
p_next = p + K sin(q)
q_next = q + p_next          (q modulo 2π).
```

**PWC implication.** Coupled maps are cheap candidates for internal matter dynamics. But a
periodically kicked map incorporates a drive; it is not automatically an autonomous isolated
energy-conserving universe. Continuous KAM statements do not directly apply to a finite set
of quantized states. Finite exact recurrence may coexist with convincing short-term mixing.
Chirikov's later work explicitly discusses discrete-spectrum and digital-system behavior as
distinct from familiar continuous chaos.
[Chirikov, Pseudochaos in Statistical Physics, 1997](https://arxiv.org/abs/chao-dyn/9705004).

### Novikov and Kamchatnov: topology can constrain behavior, but only the right topology

**Source result.** Novikov connects Hamiltonian dynamics, multivalued actions, and the
topology of spaces of paths and closed contours. Kamchatnov constructs linked closed field
lines through a Hopf mapping and an ideal-MHD solution. His stability discussion involves
more than nonzero topology: additional conserved angular momentum prevents spreading in
the case considered. These are specific continuous systems with conditions on fields and
boundaries.
[Novikov, original 1982 paper](https://www.mathnet.ru/php/archive.phtml?jrnid=rm&option_lang=eng&paperid=3819&wshow=paper),
[Kamchatnov, original JETP 1982 paper](https://www.jetp.ras.ru/cgi-bin/dn/e_055_01_0069.pdf).

**PWC implication.** Memory can be an arrangement that cannot continuously unwind under the
available moves, rather than a named stateful component. But drawing closed lines, preserving
vertex parity, or imposing zero divergence is not the same as preserving linking, winding,
helicity, or a stable particle. The distinction is essential for the loop model below.

## 2. Eight distinct native-mod directions

These are small law sketches for independent experiments. All coupling coefficients should
come from transparent common functions of configuration or geometry. Such functions are still
design choices. An obscure hash that picks a behavior from a table is authored variety, even
if no material has a name.

### N1 — Fuel-front autocatalytic lattice

**State:** integer available units `f`, active units `a`, and spent units `w` per participating
cell. Keeping `w` explicit allows honest accounting. Store a small active domain and prepared
fuel supply; fields are instance state, not new material intern IDs.

**PWC proposal:** apply local stoichiometric events such as

```text
f + a -> 2a
a -> w
```

and transport active/available units by matched face subtractions/additions. Reaction rate
or event eligibility depends on quantities and common material coupling. The closed total
`sum(f+a+w)` is conserved by these events. It is an amount invariant, not automatically a
thermodynamic energy invariant. Recharging spent units is an external operation unless a
separate paid conversion is included.

**Potential:** ignition-like fronts, consumptive signal gain, geometric splitters, controlled
processing, and resource-fed growth. An activator can produce many descendants by consuming
prepared fuel, so gain has a clear cost.

**Limit:** the front spends its medium; a reusable wire needs replenishment and a recovery
mechanism. The continuum KPP speed formula is only inspiration. An integer event law can
require a finite seed or stop at barriers where the PDE's small tail would propagate.

**Cost:** a few counters and fixed face work. The default inactive state can sleep. A front
can still activate a surface area or whole volume; sparse storage is not a hard bound on
activity. Cap admission atomically and specify simultaneous reactions versus ordered events.

**Useful first result:** one front advances through two geometries, consumes a counted finite
supply, and stops after exhaustion. Do not call this a recurring oscillator demonstration.

### N2 — Cyclic-phase or recovery lattice

**State:** a finite phase `c_i` in `Z/k`, optionally a recovery/fuel counter. For example, a
resting site advances when enough adjacent sites are in its successor phase:

```text
c_i_next = c_i+1 mod k  if count{ j : c_j=c_i+1 mod k } >= threshold(C_i)
         = c_i         otherwise.
```

**PWC proposal:** use this as a deliberately fictional cyclic interaction law. An alternative
resting/firing/recovering rule describes excitable pulses more directly, but explicitly
authors those three roles. The two experiments should be named accurately.

**Potential:** phase waves, collisions, spirals, entrainment, and timing-dependent behavior.
These are possibilities to find, not guaranteed outputs of the threshold rule.

**Limit:** a modulo cycle is not a chemical mechanism. It generally conserves neither phase
populations nor energy. A clock permanently built into each site's advance rule demonstrates
the chosen primitive, not discovery of an oscillator from stationary nonoscillating matter.
Adding fuel and inhibition to approach BZ behavior makes a different model from this rule.

**Cost:** one small state and six neighbors per site. Sparse activity is possible only if the
chosen background is a genuine fixed point. Recovery sites remain active after a pulse.
Snapshot updates and complete phases matter; scan order can create artificial travel speed.

**Useful first result:** a disturbed finite configuration creates a sustained nontrivial
pattern from an initially stationary background, with stated supply assumptions. Then test
whether a second input changes the output in a reusable way.

### N3 — Parity loops and, later, topological memory

**State:** a signed edge flux `f` or binary occupied-edge field, with one canonical owner per
edge. A plaquette move adds a signed circulation around its four edges, or toggles those four
bits. Composition can govern admission or the local energetic cost.

**Derivation:** `div(∂plaquette)=0`, so signed circulation preserves vertex divergence; XOR
toggling preserves vertex parity. Neither identity preserves edge count, loop length, energy,
or linking. Toggling the four edges of an isolated square erases that loop in one move.

**PWC proposal:** initially test a parity-preserving loop universe. To seek protected memory,
give the accessible cell complex a real hole or periodic direction and use moves that cannot
cross it. A loop with nonzero winding then differs from a contractible loop. Matter edits
that open the hole change the topology and need defined handling.

**Potential:** circulating signals, winding labels, stored loop arrangements, and geometric
memory. These must be demonstrated under the actual transition rule.

**Limit:** a simply connected cube with arbitrary local plaquette flips has no such nontrivial
winding protection. Zero divergence is not helicity conservation. A greedy local energy
decrease merely relaxes loops and does not give motion or clocks; conjugate state or another
drive is needed. Knotted memory requires stronger noncrossing/linking rules than parity.

**Cost:** more indices and shared geometry than a scalar field, but bounded plaquette moves.
Chart seams require channel/orientation transport; raw storage-coordinate adjacency is
insufficient. Huge global linking calculations should not become a hot-path requirement.

**Useful first result:** preserve the stated local invariant while moving and interacting
loops. Claim protection only after the relevant winding/linking invariant survives a broad
set of allowed disturbances.

### N4 — Spin or orientation lattice

**State:** unit vectors `s_i` with an orientation interaction energy. One common example is

```text
H = -sum_faces J_ij s_i·s_j - sum_i D_i (s_i·n_i)²
ds_i/dt = -s_i × (-∂H/∂s_i)              [precession-only start]
```

Here `J_ij` can favor alignment or opposition. With unit vectors, the pair energy is bounded
for finite coefficients. `n_i` is an anisotropy direction; deciding how material orientation
supplies it is part of the proposed universe, not a free detail.

**Potential:** signals in fixed matter, domains, phase relationships, domain-wall motion,
frustration, and possibly stable localized textures. Materials influence dynamics through
coupling and anisotropy, rather than being declared magnets or conductors.

**Limit:** ordinary integer cross-product stepping drifts away from fixed length. Normalizing
after each step changes the numerical energy law. Signed coordinate permutations preserve
length exactly but are a different coarse compass automaton. An XY phase lattice has one
angular degree of freedom; a Heisenberg spin has two. Intrinsic precession around a permanent
field is a primitive oscillator, not an emergent energy source.

**Cost:** two effective degrees of freedom, commonly stored as three coordinates, plus local
neighbor work and a length-preserving integrator. A uniform aligned background can be at rest;
random anisotropy can make generated terrain active everywhere. Seed equilibria before
claiming giant-world sparsity. Damping needs an accounted sink if energy is tracked.

**Useful first result:** perturb an equilibrium path, observe transmitted orientation changes,
and compare two assemblies with the same amount of material. Separately measure length and
energy errors. A finite heading-state test does not validate continuous spin solitons.

### N5 — Integer shear-map lattice

**State:** two bounded residues `q_i,p_i in Z/M`. A committed table or small integer function
supplies a universal nonlinear kick; composition supplies symmetric coupling.

**PWC proposal:**

```text
p_i_next = p_i + F_i(q) mod M
q_i_next = q_i + p_i_next mod M
```

**Derivation:** recover all `q_i=q_i_next-p_i_next mod M`, then recover all
`p_i=p_i_next-F_i(q) mod M`. Thus the complete phase is exactly invertible if the same
coefficients and geometry are used. No inverse of `F` is needed. Arbitrary clipping would
break this argument.

**Potential:** bounded recurrent internal dynamics, phase mixing, resonant or chaotic-looking
regions, and exact digital restart/reversal. Nonlinearity gives a distinct experiment from
the current largely linear unbounded-displacement wave shears.

**Limit:** invertibility is the proven property. The canonical continuous kick is symplectic
only for appropriate `F`, such as a gradient force with symmetric Jacobian. A finite integer
permutation should not claim all continuous symplectic/KAM properties. Antisymmetric face
kicks preserve summed `p` modulo `M`, not ordinary unbounded momentum. Wrapping coordinates
does not conserve a real kinetic-energy interpretation. Periodic external kicks imply drive.

**Cost:** two machine words and fixed-stencil work. Ensure `F(0)=0` if zero background must
sleep. Publish a complete kick then drift, storing phase if a budget spreads it over frames.
Composition edits change the map; reversal requires the old geometry or a reversible edit log.

**Useful first result:** forward/inverse returns exact bytes, an impulse reaches another site,
and two input states have distinct output behavior. A long orbit is not proof of a usable clock.

### N6 — Elastic bonds and geometry memory

**State:** bounded bonds or existing face relationships, displacement/velocity, and reference
length or orientation retained at formation. Start with a small active assembly, not an entire
planet represented as a deformable body.

**PWC proposal:** use a common bond potential, for example

```text
V_ij = k_ij (|x_i-x_j| - l_ij)² / 2
```

where shared composition-derived functions set stiffness and binding accessibility. The
reference `l_ij` provides a meaningful record of history. Bond creation, breakage, or plastic
reference changes must exchange chemical/elastic work with another tracked state if a total
energy claim is made. Deleting a strained bond deletes stored elastic energy unless paid out.

**Potential:** structural memory, pressure-driven routing, deformable machinery, fracture,
templating by shape, and assemblies that act differently despite having the same constituents.
The numerical solver can be a small mass-spring or compliant-constraint method; the solver
itself does not choose the bond law.

**Limit:** a fixed-voxel bond-strength field does not demonstrate rotating assemblies, joints,
contact mechanics, or moving technology. A bit that toggles on after strain is designed
hysteresis until a coherent work/formation law explains it. Breaking loops or reference
updates can introduce untracked resource sinks. Instantly recomputing stress to equilibrium
removes a finite signal-propagation mechanism.

**Cost:** graph bookkeeping, collision detection, connected-component maintenance, and
multiple mechanical steps may dominate. A bounded face graph is cheaper than arbitrary
subvoxel molecules; it is also less expressive. Keep the first test spatially modest.

**Useful first result:** load, unload, and reuse an assembly while preserving a meaningful
structural difference; then show a moving assembly or explicitly defer motion. A changed
bond counter alone is not a machine.

### N7 — Cooperative composition exchange

**State:** the existing bounded element multisets; optionally later an activation reservoir.
The distinguishing primitive is a coordinated movement of several occurrences, not another
dynamic phase field. This tests whether the old model's limited moves trap useful chemistry.

**Derivation from the current pair fit.** For a subset `T` moved from `A` to `B`, define the
old single-occurrence gains `g(e)=attraction_B(e)-support_A(e)`. The exact group gain is

```text
ΔS_group = sum_e_in_T g(e) + 2 sum_{e<f in T} J(e,f).
```

The correction restores interactions between moved occurrences that stay together. For two
groups `T` and `U` exchanging sides, also add their within-group corrections and subtract
`2 sum_{e in T,f in U} J(e,f)` from the sum of their old single gains. Capacity and both
destinations must be admitted atomically.

**Potential:** strongly compatible subassemblies move together, multi-step local traps become
accessible, and extraction selectivity can differ without inventing recipes. Start with pairs
rather than all subsets: 32 occurrences already give 496 candidate pairs per side.

**Limit:** accepting only positive `ΔS` still monotonically increases a bounded score, so the
closed law still cannot sustain a cycle. Better reaction search is not independent signal
transport. A threshold scaled per occurrence versus per whole event is a different law and
should be stated. Integer reservoir-assisted unfavorable moves would add another mechanism.

**Cost:** pair/subset search and intern-table growth. Cache support and cross fits; put a
small explicit bound on candidate work. An opaque solver that finds arbitrary products
risks becoming a recipe generator with a much higher runtime cost.

**Useful first result:** find one contact trapped under single-occurrence rules that changes
under exact coordinated gain while conserving every occurrence. That demonstrates a broader
reaction neighborhood, not oscillatory technology.

### N8 — Reversible block permutation

**State:** a small discrete occupancy or occurrence assignment over disjoint local blocks,
plus the global partition phase. The local transition is a permutation that preserves chosen
invariants. Shift the partition between turns to permit transport across block boundaries.

**Source result for context.** Margolus's partitioned CA construction supports reversible
local updates, conservation, and particular computationally rich examples. Universality
belongs to specific rules and constructions, not every partition permutation.
[Margolus, Crystalline Computation, 1998](https://arxiv.org/abs/comp-gas/9811002).

**PWC proposal:** generate a simple local permutation from geometry and common interaction
scores, instead of a named material rule table. A fixed permutation of a `2×2×2` block's
eight binary sites has at most 256 input patterns; preserving popcount guarantees one amount
invariant. Permuting actual constituent assignments can conserve identities too, but has a
larger local state space.

**Potential:** discrete transport, collisions, persistent structures, and information in
arrangement. It provides a different type of reversibility from the internal two-word map:
the arrangement in spatial blocks changes rather than a stationary cell's internal field.

**Limit:** repeatedly sorting by favorable energy is many-to-one, not a permutation. State-
dependent selection of individually invertible moves can also be many-to-one. A pure spatial
rotation may just rotate things forever; rich motion and stable structures need exploration.
Occupied ambient matter cannot generally sleep unless its block pattern is a fixed point.
Matter moved as `BlockId` containers is not automatically equivalent to conserved constituent
chemistry and can produce costly edit/meshing traffic.

**Cost:** packed small states and lookups can be inexpensive; the immutable rule table is
still a law with chosen content. Partition alignment must use global physical coordinates,
and complete blocks need boundary admission. Save the phase; never silently skip a block
because it lies outside visible chunks.

**Useful first result:** reverse an actual spatial collision and conserve its declared
quantity, then find a stable structure and a traveling disturbance under the same rule.

## 3. Original strange combinations to keep in reserve

These are further ideas, not claimed discoveries or implemented guarantees. Each should
remain a separate mod until it establishes that the extra state and coupling are worthwhile.

### A — Geometry writes the law's memory

Let a passing conserved impulse rotate a bond's orientation, and let that orientation change
the next impulse's scatter. This combines routing memory with the material's assembled
geometry. A train could reshape its own path, a sorter might adapt to supply, and a repeated
operation could store a pattern in bonds. The common operation is local rotation, not a
developer-recognized switch. The critical question is whether changing a bond costs or
releases work; otherwise unlimited active rewiring is an authored nonconservative primitive.

### B — Information encoded in defects between two orders

Use a spin/phase lattice with two preferred local arrangements derived from one interaction
function. A message is a defect or boundary between them. This is different from injecting a
number into a conductor: the background's structure carries the state. Local shape and
composition could pin or move a defect. The major risk is that defects only coarsen and
disappear. A bounded Hamiltonian motion or paid drive must supply useful transport while
leaving most undisturbed matter asleep.

### C — Reaction gain buys a change in geometry

Keep the old pair-fit chemistry, but let beneficial cooperative regrouping release a counted
work unit that can stretch or reorient a bond. A loaded bond can pay for a less favorable
regrouping. The same accounting would link chemistry, information and motion without named
motors. One explicit total should include chemical, bond and transported state; independent
conservation inside three solvers does not prove conservation of their coupled whole.

### D — Worlds made of local reversible deals

Each local interaction is a reversible exchange of arrangement and two-word internal state.
Rather than choosing the best outcome, follow a canonical cycle through admissible states
that share declared invariants. This avoids a one-way compatibility objective and could
make matter a medium of both storage and motion. However, enumerating all compatible states
is expensive and a large generated permutation can be an opaque behavior table. Start with
one tiny finite interaction class and ask whether it gives two useful effects, not merely
an impressive long period.

### E — Physical topology chosen by players

Let ordinary construction create holes in the accessible graph, so prepared loop flux has
different winding classes around those holes. Players would build the protected medium,
rather than find special memory blocks. Cutting or closing an aperture would become a
defined topology-changing interaction. This requires more than a binary loop flip, but it
offers a meaningful reason for geometry to matter. Distinguish safe preservation of a
winding count from a convenient global recomputation that erases or recreates flux.

## 4. How these candidates differ, and what small tests can actually establish

| Family | Additional memory | Main changing object | Cheap invariant to check | What a smoke demonstration cannot establish |
|---|---|---|---|---|
| Fuel front | available/active/spent amounts | local populations and transport | total accounted units | reusable signaling, infinite supply, PDE front speed |
| Cyclic phases | finite phase/recovery | phase at stationary sites | fixed site count only, unless more is defined | real chemistry, conserved energy, robust clocks |
| Parity loops | oriented edges / flux | spatial circulation | vertex parity or divergence | linking, protected winding, stable particle |
| Spin | orientation | local precession / domains | continuous spin length; numerical check required | exact rounded energy, continuum solitons |
| Integer maps | two residues | internal reversible shear | exact inverse; possibly modular sum | physical energy, continuous KAM dynamics |
| Bonds | references, graph, mechanical state | geometry and motion | constituent count; explicit work ledger | a full moving-voxel mechanics engine |
| Cooperative exchange | mostly existing configuration | constituent regrouping | occurrences and exact gain | recurrent closed dynamics |
| Block permutation | occupancy plus partition phase | arrangement in physical space | inverse and chosen finite invariants | computation under an arbitrary permutation |

The distinction between a quantity and its name matters. A count labelled energy is only
energy if the chosen laws consistently give it that role. A phase labelled spin is only the
specific discrete phase law implemented. PWC can adopt fictional laws deliberately, provided
the documentation describes those laws and their consequences honestly.

### The minimal experiment contract

Each mod should state its exact state encoding, background equilibrium, geometry/domain,
update phases, capacity behavior, source/sink operations, and saved version. One smoke test
can check compilation, count/inverse accounting, a restart, and one observed behavior. That
is enough for the user's present idea-first stage; it is not enough to select the game's
fundamental model.

Later compare two genuinely different player constructions under the same rule, with a
range of inputs and no special material exceptions. Useful measurements include pulse
attenuation, recovery time, load tolerance, leakage, recurrence length, work consumed,
active volume growth, memory per active site, and time per complete logical turn. Test both
good and failed constructions. A narrowly tuned survivor should not decide the architecture.

All experiments use the native provider interface described in
[the mod architecture](../../../documentation/mods.md). Core should keep cell identity,
topology adapters, transaction admission, persistence and authority. The model chooses its
law and instance state. A server selecting the same model fingerprint does not currently
replicate arbitrary experimental fields or laboratory input. Cartesian/chart-interior labs
must document that limitation rather than infer direction transport from `resolve` alone.

The package manager's [writing-a-mod tutorial](https://github.com/gusahlg/pwc-package-manager/blob/main/docs/writing-a-mod.md)
and [first-party package layout](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/README.md)
were read locally for the delivery contract: normal complete source packages, native compilation,
manifested licensing, independent options, documented input/persistence, and deterministic hooks.
No files in that repository were changed by this research.
