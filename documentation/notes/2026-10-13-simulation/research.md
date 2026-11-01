<!-- SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors -->
<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# Fundamental simulation models: research and criticism

Date: 2026-10-13. This is a research notebook, not a settled physics design. No throughput,
gameplay quality, universal-computation capability, or stability result is claimed for a new
PWC model here. The user requested a broad search and ideas first, with little testing now.

**Recommendation:** preserve selective transfer as an independently selectable baseline, then
compare small mods built around three different kinds of memory: moving conserved packets;
displacement and momentum in stationary matter; and activation with recovery and a finite fuel
source. A reservoir-assisted reaction model is a useful bridge, but it does not solve all three
problems. Do not choose a winner from attractive screenshots or mathematical elegance.

The sources below are original papers, author-hosted papers, and official project documentation.
Sourced observations are labelled **Evidence**. Statements labelled **PWC assessment**,
**Derivation**, or **Original proposal** are analysis for this project; the cited authors have
not validated them. Bibliographic years refer to original works, not search-engine crawl dates
or dates inserted by regenerated PDF builds. Short equations describe candidate laws, not
ready-made production specifications.

## 1. What PWC already has, and what it cannot have

The existing [material model](../../../documentation/material-model.md) conserves a multiset of
four-dimensional toroidal element coordinates. A voxel holds up to 32 occurrences. Its pair
compatibility determines whether a face contact transfers an occurrence or exchanges two.
The current scheduler wakes changed contacts, evaluates bounded work, and lets unchanged
contacts sleep. This gives a particularly valuable property: a vast generated world can be
quiescent without scanning every voxel every tick.

The four resource axes are identities and interaction descriptors; they are not four spatial
dimensions. There is no need to reinterpret them as named chemical elements or electrical
properties to extend the model. Material identity and a voxel's current dynamical state should
also be separate: two pieces of the same material can carry different signals or stresses.

### A limitation stronger than a missing feature

**Derivation from the specified selective-transfer law.** Define

```text
S(world) = sum over voxels of sum over unordered internal occurrence pairs J(a,b).
```

The raw gain of an accepted transfer, and the corrected gain of an accepted swap, are exactly
the change in this score. Every accepted operation increases it by a strictly positive
threshold. In a closed system with finitely many conserved occurrences, the total number of
possible internal pairs and every pair score are bounded. Therefore `S` cannot keep increasing
forever. The specified law eventually reaches a state with no accepted operation, and cannot
return to a previously visited state.

Consequences:

- Rearrangement can carry a transient signal, but exhausts available compatibility improvement.
- A closed assembly cannot be a perpetual clock under this law alone.
- Making the fit curve richer, improving tie-breaking, or lowering a positive threshold does
  not remove the monotonicity argument.
- Adding an independently changing state, a drive, or non-gradient motion changes the premise.

This reasoning applies to the ideal conserved kernel and atomic commits. Implementation bugs
that lose matter are defects, not extra expressive power. An open machine can continue to
process a fresh supply of matter, so the argument does not prohibit useful consumptive chemistry.
It does prohibit assuming that a static finite piece of matter will become a reusable oscillator
just because its compatibility function is complicated.

**PWC assessment.** At minimum, meaningful technology seems to need ways to store a state,
transport a disturbance, interact nonlinearly, and replenish whatever gets consumed. These are
capabilities to demonstrate, not mandatory names for quantities. Charge, temperature, voltage,
and familiar circuit components are optional interpretations.

## 2. Candidate A: conservative lattice packets and reversible cellular automata

**Evidence.** Fredkin and Toffoli construct computing models compatible with reversibility and
additive conservation, including a billiard-ball model in which collisions perform logic.
Margolus shows partitioned cellular automata with invertible local updates and examples in
which stable reflecting structures obey the same rule as moving particles. This establishes
that conservation and reversibility do not inherently prevent rich computation; it does not
establish that arbitrary simple conservative rules are useful worlds.
[Conservative Logic, 1981 technical report](https://dspace.mit.edu/entities/publication/19826ace-4422-4f48-b134-5be130d5bcf5),
[Crystalline Computation, 1998](https://arxiv.org/abs/comp-gas/9811002).

**State.** A few directional occupancy bits or integer packet counts at each lattice location,
possibly packet species, and a saved global update phase. Matter may itself be packets, or
stationary matter may scatter a separate packet field. The latter is an easier first comparison
with current voxels, but is not yet a unified matter-and-motion model.

**Law.** Alternate streaming with local collisions, or apply a local permutation on disjoint
small blocks and shift the partition on the following phase. Constrain local transitions to
preserve explicitly chosen invariants such as count, species multiset, and momentum. If local
rules are permutations, the global phase-composed update can be invertible.

**PWC assessment: what might emerge.** Ballistic transport, collisions, storage in closed
trajectories, reversible logic, and geometry-dependent routing. A collision rule is preferable
to an authored `if copper then conduct` branch. Derive scattering from the same elemental
interaction formula wherever possible. Stable walls should be tested as assembled states;
declaring a privileged immutable wall hides a fundamental exception.

**What is hard.** Cohesive matter, rigid machinery, liquids, useful macroscopic dissipation,
and robust signal gain require more than conservation alone. A digital collision computer can
depend on exact timing, alignment, and large engineered mirrors. Computational universality
may be almost irrelevant to a player who cannot reliably build a repeater. A six-direction
cubic stencil also visibly privileges axes. Conserving packet number alone does not conserve
physical kinetic energy or momentum.

**Cost and architecture.** Tiny state is attractive for packed integer/SIMD work. Empty regions
can remain asleep if their rule preserves the empty state. Every occupied moving packet remains
active; a dilute ambient gas across a whole planet defeats edit-only sparsity. Shifted partitions
must use world coordinates across chunk boundaries, and the partition phase belongs in saves.
Arbitrary order-dependent processing is not a reversible parallel update. A budget must finish
one logical phase before publishing the next phase, even if it takes multiple server frames.

**First convincing demonstration.** A signal crosses two material arrangements, one reflects
it and one transmits it, then two signals interact to produce a distinguishable output under
one rule. Save midway and replay. Reverse a closed finite run only if the actual rule was built
to be invertible. Reject claims that merely retaining old states makes the underlying law
reversible.

## 3. Candidate B: microcanonical chemistry with a finite local reservoir

**Evidence.** Creutz's microcanonical algorithm samples constant-total-energy configurations
by coupling a configuration to an energy reservoir. His later deterministic Ising dynamics
uses local spin, parity, and conjugate-state variables and supports nonequilibrium studies.
D'Souza and Margolus demonstrate reversible aggregation coupled to transported heat tokens:
binding releases tokens and unbinding absorbs them; growth and annealing arise in a closed
local system.
[Microcanonical Monte Carlo Simulation, 1983](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.50.1411),
[Deterministic Ising Dynamics, 1986, author PDF](https://latticeguy.net/mypubs/pub088.pdf),
[Reversible aggregation, 1998](https://arxiv.org/abs/cond-mat/9810258).

**Original proposal for PWC.** Keep elemental multisets and take `V = -S` as configurational
potential. Add nonnegative integer reservoirs `r_i` to voxel instances, with

```text
E_total = V(matter) + sum_i r_i.
accept a proposed rearrangement only when its ΔV can be paid;
r_after = r_before - ΔV.
```

Improved compatibility releases resource into the reservoir; unfavorable rearrangement can
consume it. Redistribute reservoir units across faces through conservative exchanges. Use
the same law for every material; coefficients or barriers can be derived from composition.

**Important criticism.** Reusing the old candidate chooser, which only proposes the greatest
positive compatibility gain, still gives one-way relaxation. A reservoir with no reverse
proposals is mostly bookkeeping. Include a defined, deterministic proposal schedule that can
try energetically affordable unfavorable moves. Equilibrium sampling is also not proof of a
good physical time model: Monte Carlo steps need not represent a believable movement speed.

**What might emerge.** Binding and release, diffusion of usable resource, driven extraction,
history-dependent processing, activation thresholds, reservoirs, and limited chemical cycles
fed by differences in supply. Quantized tokens could offer a discrete alternative to a named
temperature field. Players might discover materials that store, route, or expend them.

**What remains difficult.** Passive diffusion spreads and smooths disturbances. It does not
give inertial waves, stable oscillation, regenerative signal gain, or mechanics automatically.
A closed equilibrium system has no inexhaustible useful work source. Total conserved energy
can persist while becoming unusable for a particular machine.

**Cost and architecture.** A small reservoir field is cheaper than full particle velocities,
but any nonzero diffusing field can keep spreading its active footprint. Exact integer fluxes
must subtract and add the identical amount; separately rounded updates introduce or destroy
units. A bounded reservoir must block an event that cannot store its released energy, or
explicitly transport the excess. Saturation is not an energy sink unless the model declares
one. Hash-based proposal randomness can be reproducible without being time reversible.

**First demonstration.** A closed reactor conserves its declared total while changing matter;
an externally prepared resource difference drives extraction that the baseline cannot perform;
removing the difference makes processing stop. Inspect matter and resource separately. Do not
infer oscillators from changing colors.

## 4. Candidate C: nonlinear oscillator or wave lattices

**Evidence.** Fermi–Pasta–Ulam–Tsingou models use coupled nonlinear oscillators and exhibit
rich energy exchange and recurrence. Recent work also illustrates why quasiperiodicity and
thermalization cannot be assumed from a simple formula alone; parameter regime, system size,
and resonance structure matter.
[Ganapa, quasiperiodicity revisited, 2023](https://arxiv.org/abs/2303.10297),
[Nelson, Porter, Choubey, heterogeneity and recurrence, 2018](https://arxiv.org/abs/1804.06523).

**State.** One scalar displacement `q_i` and one conjugate momentum `p_i` per participating
cell; alternatively current and previous displacement in a second-order recurrence. These
describe a disturbance in matter while its elemental identity remains unchanged.

**Original proposal.** Begin with a bounded-below local Hamiltonian:

```text
H = sum_i [ p_i²/(2m_i) + a_i q_i²/2 + b q_i⁴/4 ]
  + sum_faces(i,j) k_ij (q_i - s_ij q_j)²/2
dq_i/dt = p_i/m_i
dp_i/dt = -∂H/∂q_i
```

Use `m_i > 0`, `b > 0`, and `k_ij >= 0`. Derive `m`, `a`, `k`, and sign `s in {-1,+1}`
from common functions of configurations, rather than a material classification table. `a < 0`
creates two local minima when the positive quartic term stabilizes it; this offers memory.
The sign coupling favors aligned or opposed disturbances. Loops with incompatible sign
preferences create frustration. All are hypotheses for a PWC experiment, not discoveries.

**What might emerge.** Transmission through fixed matter, reflection at interfaces, resonance,
frequency filtering, standing modes, stored oscillation, nonlinear mixing, bistable memory,
and possibly localized excitations. Geometry becomes part of the device: a branch, cavity,
loop, or alternating composition may behave differently without a named component.

**What is hard.** Linear waves alone supply no robust digital gain. Tiny waves never settle
exactly in a lossless system, so even background numerical activity can keep a large connected
region awake. Nonlinear models can disperse pulses or thermalize. A scalar field is not a
full elastic solid and does not move voxel structures. A permanent oscillator with ideal zero
damping is possible mathematically, but a useful clock must tolerate load and interference.

**Cost and architecture.** A fixed small stencil is attractive. Prefer a stable fixed-step
integration rule and document numerical energy behavior. Symplectic stepping improves long
term behavior in suitable systems; it does not imply exact energy conservation, and choosing
an arbitrary integer recurrence does not inherit a continuous Hamiltonian proof. Negative
raw compatibility used directly as a spring stiffness can make the harmonic potential
unbounded; the squared sign-coupling form above avoids that specific mistake.

The exact quiescent state can be `q=p=0` in a stable medium. Approximate sleeping needs an
explicit error/energy budget: deleting weak waves or damping them makes a different law.
Simulating only visible chunks would make transmission depend on who is watching. Changing
composition changes the Hamiltonian; the resulting work must be accounted for if a later mod
couples reactions to this field. A separate conservative reactor and a separate conservative
wave solver are not automatically conservative when connected.

**First demonstration.** A tapped chain carries a measurable pulse while its material IDs stay
fixed; an interface reflects it; a ring sustains a mode for an observed duration. Then show
repeatable nonlinear switching or mixing. Measure attenuation, numerical drift, and active
region growth before advertising an emergent computer.

## 5. Candidate D: excitable media and reaction–diffusion

**Evidence.** FitzHugh reduces excitable and oscillatory behavior to two variables with fast
activation and slower recovery. Depending on drive and parameters, the system can rest or
oscillate. Turing demonstrates that local reaction and diffusion can destabilize a uniform
state into spatial patterns. Neither result requires representing every familiar physical
quantity.
[FitzHugh, 1961, original paper](https://jxshix.people.wm.edu/2009-harbin-course/classic/FitzHugh-1961.pdf),
[Turing, 1952, original paper scan](https://www.dna.caltech.edu/courses/cs191/paperscs191/turing.pdf).

**State and representative law.** Fast activation `u`, slow recovery `v`, and possibly a
finite fuel variable for an honest energy interpretation:

```text
du/dt = u - u³/3 - v + I + D Δu
dv/dt = ε (u + a - b v)
```

This is a standard qualitative form, not the sole sign or scaling convention of FitzHugh's
original equations. `I` is a drive; it cannot be silently equated with a free renewable energy
source. For PWC, either explicitly adopt a fictional nonconservative activation law, or build
a fuel-and-recovery model and debit the fuel used to regenerate a pulse. The latter is a new
candidate and needs fresh analysis.

**PWC assessment: what might emerge.** Triggered pulses through stationary matter, refractory
intervals, interacting wave fronts, pacemakers in driven regimes, frequency division, and
threshold-based signal restoration. The recovery variable is purposeful: it lets a location
respond, temporarily resist another pulse, then respond again. A three-state resting/firing/
recovering automaton captures some of the same capability with exact integer decisions, but
declaring its transitions directly is more explicit behavioral design.

**What is hard.** Reaction terms are already specific choices. An elegant cubic polynomial
can encode as much desired behavior as a state machine; equation count is a poor measure of
hardcoding. Regenerative pulses need a supply or a consciously fictional law. Reliable wiring,
fan-out, crossings, and storage require parameter regions with margins, not one delicately
tuned movie. Recovery consumes another field and continuing work after the visible pulse.

**Cost and architecture.** Two local fields and a short stencil are modest; explicit diffusion
and the cubic term constrain the timestep. Quantization shifts thresholds. Clamp operations
can hide instability and do not conserve a quantity. Evaluate one snapshot into the next or
use a specified split schedule; arbitrary sequential updates alter front speed. A stable
exact resting state helps sparse updates, but an oscillatory background keeps a whole region
active. Prepared fuel reserves and replenishment must survive saves and server absence.

### Gray–Scott is an inspiration, not free emergent fuel

**Evidence.** Pearson's Gray–Scott experiments produce spot division, irregular spatial
patterns, oscillatory regions, and chaos from two reacting and diffusing concentrations.
The underlying reactions are `U + 2V -> 3V` and `V -> P`, plus a feed/removal process.
The reported patterns arise after finite perturbations; the paper explicitly distinguishes
them from stable Turing patterns in the parameter region investigated.
[Complex Patterns in a Simple System, 1993](https://arxiv.org/abs/patt-sol/9304003).

**PWC assessment.** This is useful for growth or ecological mods, and possibly resource-fed
information media. Its continual feed and inert product are part of the law and must appear
in the game economy if interpreted as matter. Replacing them with finite reservoirs changes
the system and can destroy the very phenomena that made it attractive. Spontaneous spots
are evidence of complexity; they are not evidence of factories, controlled computation, or
long-lived understandable materials.

**First demonstration.** A pulse traverses a material path repeatedly with a measured recovery
interval, and two pulses have a consistent interaction. If fueled, count the spent resource
and demonstrate exhaustion. Test a range of timings before claiming a logic gate.

## 6. Candidate E: composition, bonds, and artificial chemistry

**Evidence.** Fontana's Algorithmic Chemistry treats molecules as transformations, making
objects and their actions mutually generative. The Stringmol project similarly lets encoded
molecules serve both informational and catalytic roles; its official site provides executable
experiments and technical specifications. These are examples of choosing a substrate whose
expressive space can grow, rather than writing a closed table of recognized products.
[Fontana, Algorithmic Chemistry, 1991, author-hosted paper](https://sites.santafe.edu/~walter/Papers/alchemy92.pdf),
[Stringmol official research and implementation site](https://stringmol.york.ac.uk/index.htm).

**State.** A graph or short string of constituents and relationships, not only an unordered
multiset. Include current bond arrangement, orientation, or active binding state. Spatial
adjacency can supply part of the graph; bounded explicit bond ports can provide the rest.

**Original proposal: geometry remembers chemistry.** Let each element expose a small set of
directional interaction opportunities derived by one formula from its existing torus
coordinates. Bonds lower a universal energy when neighboring orientations fit; incompatible
competition and finite valence prevent everything binding to everything. Breaking or making
bonds consumes or releases a tracked resource. Two assemblies with identical elemental
multisets can now differ structurally, and therefore react differently.

**What might emerge.** Isomer-like alternatives, templating, catalytic surfaces, chains,
selective sieves, reusable reaction sites, and structural information. Catalysis would mean
an assembly changes the availability or barrier of a transition while returning to its
previous state, rather than a developer naming a catalyst material.

**What is hard.** A purely pairwise local fit may have insufficient shape selectivity for
robust templating. The world's spatial geometry may already be expressive enough, so explicit
intravoixel bonds must earn their cost. Strings with instruction pointers are rich but
effectively add a programming machine, with an instruction set that may conflict with a
minimal coherent physical universe. Arbitrary term reduction can take unbounded time, and
unbounded product growth is incompatible with predictable frame budgets.

**Cost and architecture.** Graph canonicalization, hashing, storage, matching, and intern-table
growth can dominate arithmetic. Start with a tiny bounded port representation and local
changes, with no whole-world subgraph matching. Material identity should encode structural
state only when it is truly intrinsic and immutable; instance orientation and stress should
remain instance state. Decide whether a voxel is an elementary object or a coarse vessel
before placing 32 fully bonded subparticles inside every occupied cell.

**First demonstration.** Construct two different arrangements with the same constituents;
show a repeatable difference in an interaction. Then demonstrate a template used twice,
paying for both products without consuming the template. Replication and open-ended evolution
are ambitions far beyond this initial result.

## 7. Candidate F: conserved phase fields and pattern-forming solids

**Evidence.** Cahn–Hilliard develops a free-energy description with local composition and a
gradient-energy contribution. Phase-field crystal work shows how density fields can encode
crystalline structure and elastic/plastic effects. Stefanovic and colleagues add propagating
wave modes to distinguish fast elastic relaxation from slow diffusive evolution; a later
study explicitly criticizes mechanical-equilibrium limitations of standard diffusive PFC.
[Cahn and Hilliard, nonuniform free energy, original paper III PDF](https://www.appliedpolymertechnology.org/uploads/1/0/8/8/108867241/cahnhilliard_jchemphys_31_688.pdf),
[Phase-field Crystals with Elastic Interactions, 2006](https://arxiv.org/abs/cond-mat/0601619),
[Phase-field-crystal models and mechanical equilibrium, 2013](https://arxiv.org/abs/1311.7336).

**Representative state and law.** One conserved composition/order field `φ` and a local
potential with two preferred values:

```text
F[φ] = sum_i [ (φ_i² - 1)²/4 + κ |gradient φ_i|²/2 ]
μ = φ³ - φ - κ Δφ
dφ/dt = M Δμ
```

This conventional Cahn–Hilliard form derives evolution from energy and transports composition
through differences in chemical potential. A phase-field crystal variant introduces a preferred
wavelength; it is not the same model as this double-well example.

**PWC assessment: what might emerge.** Domains, interfaces, segregation, coarsening, wetting-like
effects with appropriate boundary terms, and at a larger cost crystalline patterns or defects.
Much material variety could emerge from common local energy functions instead of block classes.

**What is hard.** Diffusive gradient flow again relaxes toward minima; it is not a signal or
clock solution by itself. Diffuse interfaces require several samples across a feature, so a
one-voxel wall is not automatically meaningful. The fourth-order spatial dynamics imposes
stiff explicit timestep restrictions; PFC can be still more demanding. Elasticity is not
automatically correct merely because a density pattern looks crystalline.

**Cost and architecture.** Dense field updates and repeated stencils work against sparse giant
worlds. Conserved antisymmetric face fluxes offer exact integer bookkeeping, but discretizing
the variational law with integers changes the dynamics. An implicit solver can introduce
global work and unpredictable iteration counts. Keep these models as bounded region or
world-generation experiments initially. They may be excellent for creating a planet's
mineral history without being a good per-voxel runtime universe.

**First demonstration.** A finite closed mixture separates while preserving its declared
amount. Show which interface and defect behavior is repeatable at voxel scale. Then explain
what additional independent state would carry useful signals; do not mistake domain formation
for technological expressiveness.

## 8. Candidate G: lattice Boltzmann and position-based mechanics

These methods are worth searching because a successful fictional chemistry can still need
machinery that moves, fluid pressure, collisions, and work. They are primarily numerical
frameworks; adopting one does not select the world's fundamental quantities or constitutive
laws.

### Lattice gas versus lattice Boltzmann

**Evidence.** Frisch, Hasslacher, and Pomeau show that a class of Boolean lattice gases has
Navier–Stokes behavior at a macroscopic limit. Modern lattice Boltzmann implementations
replace Boolean occupancy with directional populations; the LB3D implementation uses D3Q19
for interacting complex fluids. These are distinct representations with different cost and
noise properties.
[Lattice-Gas Automata for the Navier–Stokes Equation, 1986](https://doi.org/10.1103/PhysRevLett.56.1505),
[LB3D implementation paper, 2017](https://doi.org/10.1016/j.cpc.2017.03.013),
[Lattice Boltzmann authors' official textbook page](https://www.timm-krueger.de/lattice-boltzmann-book).

**PWC assessment.** Directional populations, streamed and collided, offer fluid flow,
advection, and pressure-like disturbances. They are a plausible independent fluid mod.
Introducing density, velocity, multiphase terms, and reactions makes them far more than a
one-scalar replacement for compatibility. Static solids, moving boundaries, free surfaces,
and low-density vacuum need explicit treatment. A cubic stencil's symmetry matters; a
six-direction packet experiment should not claim the established hydrodynamic behavior of
a different stencil. Low-Mach formulations are not unrestricted compressible physics.

### Position-based dynamics and XPBD

**Evidence.** XPBD formulates compliant constraints with an explicit energy interpretation,
addressing traditional PBD stiffness dependence on timestep and solver iterations. Its
authors subsequently find that, in their tested problems, many small substeps with one
constraint iteration reduce errors and damping compared with one large step and many
iterations. This is evidence for a numerical strategy, not a benchmark of PWC.
[XPBD, 2016](https://matthias-research.github.io/pages/publications/XPBD.pdf),
[Small Steps in Physics Simulation, 2019](https://matthias-research.github.io/pages/publications/smallsteps.pdf),
[author's runnable mechanics demonstrations](https://matthias-research.github.io/pages/challenges/challenges.html).

**PWC assessment.** State consists of positions, previous positions or velocities, mass, and
constraint/bond state. This can make beams, ropes, articulated structures, and moving voxel
assemblies plausible without giving each named mechanism a hand-written animation. The
constitutive laws and connectivity still have to be chosen. Iterative projection introduces
numerical damping and convergence error; it does not automatically conserve energy or
produce deterministic output under arbitrary parallel scheduling.

Cluster-level sleeping and a material-derived bounded constraint graph are attractive. Yet
finding connected components after cuts, collisions between complex assemblies, gravity,
structural fragmentation, and voxel-to-body transitions are substantial work. Avoid choosing
the entire planetary terrain as one deformable body. An active graph of modest player-built
assemblies is a more credible first experiment. A mechanics solver should debit or report
work when coupled to chemistry; a spinning wheel must not refill a reservoir for free.

**First demonstration.** A beam or connected assembly falls or bends under one derived bond
law, then drives a local signal or interaction while transferring accounted work. A few
independently authored hinge and motor behaviors would be a different, more conventional
model and should be described as such.

## 9. Candidate H: continuous cellular life and swarm chemistry

**Evidence.** Lenia demonstrates a rich collection of autonomous moving and deforming
patterns from convolution plus a growth function. Its basic update adds growth and clips
state into `[0,1]`; it is not an additive matter-conservation rule. Chan explicitly identifies
difficulties in constructing universal computation from imprecise motion and collisions.
Swarm Chemistry explores heterogeneous agents and self-organized structures; a long-run
experiment reports that evolutionary dynamics can settle after an initial transient despite
producing many novel objects.
[Lenia, 2019](https://arxiv.org/abs/1812.05433),
[Lenia author's demonstrations](https://chakazul.github.io/lenia.html),
[Swarm Chemistry long-term analysis, 2018](https://arxiv.org/abs/1804.03304),
[Sayama's review of swarm systems, 2024](https://arxiv.org/abs/2409.01469).

**PWC assessment.** These are strong inspirations for ecosystems, self-repair, living
machinery, and searching the space of local laws. They are weaker default foundations for
predictable player materials. Lenia needs a larger convolution neighborhood, which raises
the price of sparse boundary work; an FFT implementation favors dense periodic domains.
Swarm approaches need moving agents and neighbor searches. Heterogeneous parameter sets can
quietly become a content library of distinct behaviors.

Neither autonomous motion nor a large pattern catalog proves open-endedness. A creature
demonstration does not establish material accounting, composable circuits, or accessibility
on weak hardware. Adding conservation or replacing growth with finite resource consumption
creates a new system whose attractive behaviors may disappear. Keep these as independent
growth/evolution mods, and judge whether a player can cultivate, combine, or reliably use
their structures.

**First demonstration.** A structure survives a disturbance and returns to a useful behavior
under measured resource constraints. Compare a range of initial shapes and quantify failures;
do not only keep the impressive survivor.

## 10. Original directions worth making deliberately strange

These proposals combine ideas but are not attributed to the cited researchers. None is
claimed to be novel in the mathematical literature. Their purpose is to expand PWC's search
space beyond adding a copy of electricity to its current material law.

### O1 — Routing-memory matter

Every occupied cell has a small rotor index. An arriving conserved token selects an eligible
exit using that index, then advances the index. The eligible exits come from universal
compatibility with neighboring matter. A token changes the path future tokens take, so
transport itself writes local memory. Geometry could create counters, sorting paths,
circulation, and history-dependent routing with only integer counts and tiny indices.

**Evidence for the inspiration, not the proposed physics.** Rotor-router walks are
deterministic analogues of random walks. Their ordering affects behavior, and parallel
systems can have much longer periods than simple single-walk intuition suggests.
[Doerr and Friedrich, two-dimensional rotor-router walks, 2007](https://arxiv.org/abs/math/0703453),
[Chalopin and colleagues, parallel rotor-router lock-in, 2014](https://arxiv.org/abs/1407.3200).

**Criticism.** A rotor is already a small routing machine. It must have a coherent interpretation
and cost if energy is claimed. Universal local use is preferable to a special router block,
but universality of application is not absence of design choices. A mere round-robin rule may
give diffusion and repeating tours without useful gates. The early test should seek a reused
counter or sorter, then attack it with simultaneous arrivals and changed neighboring matter.

### O2 — Frustrated resonance matter

Use the Hamiltonian in candidate C. Composition determines a preference for aligned or
opposed signal displacement, and closed geometry can make those preferences incompatible.
Players would engineer rhythm, phase relationships, defects, and bistable structures by
assembling matter. No named conductor, gate, coil, or clock is necessary.

**Criticism.** Frustration often produces trapping, noise, or complicated relaxation rather
than useful computation. Lossless oscillation does not restore noisy signals. The first
experiment should find a stable memory state and a load-tolerant resonant path under the same
formula, not tune each example separately. If it needs named per-material thresholds, its
claimed elegance has failed.

### O3 — Loop-flux universe

Put signed integer flux `f` on oriented voxel faces. Require zero divergence at ordinary
matter sites. Changing the flux around a small plaquette by the same amount preserves that
constraint: the boundary of a loop contributes no net source at a vertex. Closed conserved
loops become a different candidate for persistent information. A second conjugate plaquette
state could give motion; composition changes the local cost of storing or moving flux.

**Criticism.** Conserved loops by themselves are static memory, not propagating waves.
Adding a drive or conjugate state needs its own Hamiltonian and update law. The analogy to
gauge fields must not be advertised as reproducing electromagnetism. Shared face ownership,
plaquette updates across chunks, integer bounds, and changing topology are genuine costs.
Test whether a perturbation can travel around a player-built loop before speculating about
topological computers.

### O4 — Chemical arrangement as a mechanical program

Replace a block's indistinguishable bag by a very small bounded internal graph, or use only
the graph already formed by spatial voxel contacts. A token crossing a bond can strengthen
one available bond and weaken another, subject to the same conserved work budget. A material
can then reroute a signal by reshaping its own interaction graph. A machine could be a stable
cycle of graph changes rather than a named set of gears or gates.

**Criticism.** The rule may merely hide a finite-state machine inside bonds. If every material
has 32 elements and each gets many ports, storage explodes. Start with spatial bonds or one
small graph per active assembly. Demonstrate a returning cycle that requires supplied work,
then a second different use of the same law. A hand-picked cycle is only evidence that one
cycle exists.

### O5 — Two views of the same event

Explore a universe where a local event always exchanges two kinds of organization: material
compatibility and transported disturbance. Favorable regrouping excites a wave or emits a
token; an appropriately phased disturbance can pay to reverse a regrouping. Chemical
transformations, signals, and work would become conversions under one ledger rather than
three independently prescribed feature systems.

**Criticism.** This is the most appealing unification and the easiest source of hidden free
energy. A local exact ledger must include coupling energy, finite capacities, and the work of
changing material-dependent field coefficients. Removing matter carrying stored disturbance
must transfer that state somewhere. Start with two cells and one packet mode. Do not attach
several attractive solvers and assume the combined total is conserved.

## 11. Architectural consequences before choosing a law

These are PWC-specific recommendations inferred from the current architecture, not sourced
benchmarks or requirements imposed by external simulation packages.

1. **Keep laws in compiled mods and choose one explicitly per world.** Give each model a stable
   identifier, version, parameter bytes, and content digest. Store the selection in saves and
   advertise it to joining clients. Selection by mod installation order is unsuitable for
   authoritative simulation. Presentation mods can inspect the same dynamics in different ways.
2. **Keep persistent dynamical state out of interned material IDs.** Interned composition is
   shared by many cells; wave phase, reservoir content, rotor position, bond damage, and velocity
   differ per instance. Continuously changing values also exhaust a finite intern table.
3. **A simulation mod needs a full authority path.** Cell reads, bounded stepping, atomic
   proposals/commits, save/restore, compatibility, and replication are necessary. Client-only
   `update` hooks are useful for isolated laboratories but do not establish multiplayer support.
4. **Budget complete logical steps, not arbitrary fragments of a stencil.** Compute from a
   declared snapshot or a declared split phase. Preserve a partially evaluated phase until it
   is complete. Publishing half a diffusion or wave update biases the laws toward queue order.
5. **Use the same spatial topology as the world.** Stored chart neighbors, gravity frames,
   face contacts, and planet-separated locations need a defined relationship. Adjacent storage
   coordinates are not automatically adjacent physical space. A field mod should not bridge
   unrelated charts or treat atlas edges as physical walls by accident.
6. **Loading and rendering must not drive physics.** An active pulse may travel outside visible
   chunks. Keep model state and active work independent of resident mesh chunks. Exact dormant
   regions need no allocation; active regions still need admission, persistence, and bounded
   scheduling. A distant active machine cannot disappear because no player watches it.
7. **Own inter-model coupling explicitly.** Separate experiments can be exclusive. Additive
   models need an ordered coupling contract: which model owns matter moves, how events wake
   others, and how exchanged work is counted. Arbitrary combined mod order produces a new law.
8. **Give overload a physical meaning.** Slowing an entire authoritative model preserves a
   well-defined logical world better than skipping an arbitrary subset of contacts. Region
   clocks can be useful, but crossing their boundary needs a stated rule. A benchmark should
   include competing machines and spreading activity, not only one ideal reactor.
9. **Derive parameters with purpose.** Existing torus coordinates and compatibility can supply
   coefficients, but repeatedly picking arbitrary probes or hash bits to get desired behaviors
   is still authored content. One transparent formula with meaningful sensitivity is preferable
   to a statistically colorful collection of opaque knobs.

### Memory arithmetic, not measured performance

For one million participating sites, before allocation overhead, halos, indexing, material
storage, queues, renderer data, or replication:

| State representation | Single state buffer | Two full buffers |
|---|---:|---:|
| One packed occupancy bit | 0.125 MB | 0.25 MB |
| One `u16` reservoir | 2 MB | 4 MB |
| One pair of `i32`/`f32` fields | 8 MB | 16 MB |
| Nineteen `f32` directional populations | 76 MB | 152 MB |

MB here means decimal megabytes. Counts follow directly from element width. This table does
not predict speed: sparse maps can be much larger and slower than tightly packed active tiles,
and packed bits can require extra manipulation. An allocation-free kernel does not make its
world scheduler allocation-free.

## 12. What should be investigated next, without pretending to have validated it

The requested immediate work should remain small: compilation, smoke checks, and conservation
checks for implemented prototypes. The following are subsequent experiments, not a request
to spend this turn exhaustively proving every model.

| Priority | Candidate | Small experiment | Evidence it would provide | A useful reason to reject it |
|---|---|---|---|---|
| 1 | Baseline selective transfer | Run the same two-cell reactor to rest | Reference transformation and sleep behavior | Matter accounting or restart changes the result |
| 1 | Wave/oscillator | Tap, reflect, and filter a fixed-material path | Independent information transport | Active area spreads uncontrollably or only fragile analog output exists |
| 1 | Conservative packets | Two signals collide in one geometric assembly | Discrete interaction with explicit invariants | Every useful construction requires special immovable walls |
| 1 | Reservoir-assisted chemistry | Drive a previously blocked rearrangement, then exhaust supply | Reversible energetic accessibility and resource transport | Old greedy selection still only relaxes, or the ledger drifts |
| 2 | Fuelled excitation | Repeated pulses with recovery and counted consumption | Regeneration and timing margin | The clock depends on invisible free supply or hand-tuned components |
| 2 | Routing memory | Reuse one counter/sorter under concurrent arrivals | Transport writes useful state | Arrival order dominates or no useful composition exists |
| 2 | Bonds/geometry | Same constituents, two structures, reusable template | Structural selectivity | Matching/canonicalization dominates cost |
| 3 | Phase fields / PFC | Conserved separation and interfaces in a bounded volume | Material organization | Voxel resolution and stable step size are incompatible |
| 3 | XPBD / mechanics | A modest voxel assembly transfers work | Moving machines from generic constraints | Whole-world connectivity or collision cost is required |
| 3 | Lenia/swarm growth | Resource-bounded structure survives damage | Useful self-organization | Only decorative survivors, no player control or accounting |
| 4 | Loop flux / unified events | One conserved circulating disturbance and one conversion | A different fictional foundation | Static loops or hidden coupling energy never become useful |

Compare all candidates with the same questions: how many persistent state variables, what is
actually conserved, which states sleep exactly, how quickly activity spreads, what happens
after a restart, and whether one general law can produce two different useful devices.

Later, replace theory-first optimism with a short playable laboratory. Provide an inspector,
perturbation tool, recorder, resource ledger, and saved constructions through mods. Ask a
player to discover a signal path, memory, sorter, or oscillator using clear observations.
Record failed attempts and unintended results as well as successful constructions. A model
earns adoption when it produces useful surprises repeatedly, at a measured cost, while its
core laws stay explainable.

## 13. Implemented routing-memory experiment

The independent native [Memory Matter mod](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-rotor/README.md) now implements
O1's small routing-memory idea, with bounded integer impulse counts and a per-cell rotor. A
common weighted sum of element coordinates and cardinality selects port orientation and
chirality. It stores a sparse closed 7³ laboratory, advances complete budgeted turns, and
rejects overflow rather than removing impulses. It requires API 3.2 and defaults off.

One small native test found an eight-turn repeated state in a closed four-cell square with
one impulse, kept that impulse conserved, rejected oversized injection without changing the
count, and reproduced the continuation after save/reload. The package manager's source check
passed. These are limited observed results: no useful gate, controllable clock, performance,
general chart-topology support, physical energy model, or gameplay validation is claimed.
