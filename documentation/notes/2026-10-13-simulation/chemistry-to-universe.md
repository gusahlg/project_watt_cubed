<!-- SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors -->
<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# From chemistry to a universe: one conserved resource, stable matter, and the missing bridge

Date: 2026-10-14. Research and design criticism; no third new model is implemented in this
round. The current two additional experiments remain dipole occupation and relational
frames, discussed in [the China/fiction report](chinese-fiction-research.md). Their actual
checks and package status are in the [focused continuation](focused-universes.md).

The latest steering emphasizes a stronger goal than combining several simulations:
compatibility should contribute a potential `U = -k Φ`, released potential should become
motion, and heat and mechanical work should draw from the same conserved quantity.
Stable matter must then arise from small microscopic interactions. The visible message
ends during “attractive and rep...”; no missing continuation is assumed here.

**Main finding:** PWC already has one exact chemical-resource ledger and one transport
experiment. They are independent models with different state and authority. Their results
do not establish a microscopic universe with cohesive matter, recoil, heat, and machines.
The next useful bridge would make one exact energy account span reaction and moving
carriers. That would be evidence toward unification, while cohesive moving matter would
remain a separate unresolved requirement.

Throughout this note, **Source result** describes cited primary work, **Derivation** is an
argument given here, and **Proposal** is a future PWC experiment. A conservation proof is
not a proof of thermalization, solids, chemistry, computation, or enjoyable gameplay.

## 1. What the existing packages actually establish

This comparison was checked against the current package sources, read without editing:

| Existing package | Actual contribution | Missing part of microscopic unification |
|---|---|---|
| [physics-reservoir](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-reservoir/README.md) | Compatibility-changing transfers can consume or release an integer resource; reactions and resource redistribution preserve `E = ∑e - Φ` in closed admitted operations. | The resource has no velocity, pressure, mechanical work, or recoil. Redistribution is a chosen integer averaging law. |
| [physics-packets](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-packets/README.md) | Six directional occupancy bits transport signals. Spectator-controlled local scattering preserves count and local momentum; composition determines scattering axis and interface admission. | Packets are separate from material occurrences. Fixed terrain reflects packets without receiving momentum or kinetic energy. No chemistry-to-motion conversion exists. |
| [physics-wave](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-wave/README.md) | Integer displacement/momentum produces transport and interference in a bounded laboratory. | Its invertible kick/drift law does not automatically share the reservoir's exact chemical energy; scalar displacement is not movable cohesive matter. |
| [physics-bonds](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-bonds/README.md) | Relaxation and changing rest strain investigate stress and geometry memory with a heat ledger. | This is not a complete inertial mechanics or chemical binding law. Relaxation and imposed lattice geometry supply part of the behavior. |

The shared host chooses one active provider. Turning several options on does not create a
coupled world: the first active provider wins. A combined law needs its own state, event
order, compatibility version, and conservation proof. Shared static fit arithmetic does not
provide a shared dynamic energy account.

The reservoir's split `((pool+1)/2, pool/2)` gives an odd remainder to the lower-coordinate
cell. One quantum consequently has a directional transport bias. This is a universal
integer law choice, rather than an isotropic microscopic heat model. Exact total resource
can coexist with poor transport behavior.

The packet primitive's conservation is narrower than whole-world momentum conservation.
A reflection changes a packet's momentum from `p` to `-p`. Its terrain must receive `2p`
for a mechanical account to close. The current package stores no such recoil. An immovable
background can be a deliberate investigation boundary, but it should not be mistaken for
matter governed by the same microscopic law as the moving signal.

## 2. What `U = -k Φ` gives, and what it does not

Let `Φ` be the sum of the existing internal compatibility over configurations. For an
admitted rearrangement, write `g = Φ_after - Φ_before`. Then

```text
ΔU = -k g.
E = K + e + U = K + e - k Φ.
ΔK + Δe = k g.
```

**Derivation.** A positive compatibility gain lowers the potential and frees resource;
a negative gain needs payment. The existing reservoir is the special case `k=1`, `K=0`.
The sign is essential: adding `g` to the reservoir is consistent with a lower potential,
not an extra independent reward for a reaction.

This is a sound definition of a fictional scalar potential. It does not create a force.
In continuous mechanics a positional potential acts through `F = -∇_r U`; defining the
gradient also requires positions and a Hamiltonian kinetic term. PWC's four-dimensional
resource coordinates are material descriptors, not physical positions. Differentiating fit
in that space would not supply a spatial force between blocks.

Moreover, internal compatibility is unchanged if an intact configuration translates. Two
neighboring intact configurations can separate without changing this `Φ`. Thus internal
fit supplies rearrangement energetics but does not by itself supply intercell cohesion,
surface tension, a preferred separation, or resistance to shear. Position-dependent
interactions or an equivalent discrete neighborhood energy would need an explicit purpose.

Multiplying fit by a real-valued scale and rounding each event loses exact accounting.
Choose a common integer energy quantum and a rational scale represented without per-event
loss. If `k=a/b`, scale the entire ledger by `b`; retain any required remainder as state.
The reservoir cannot silently discard small gain because it is “below visual relevance.”

Finally, summing internal pair fit over each voxel depends on the representation's grouping.
If moving exactly the same occurrences across a voxel boundary changes energy, that can be
a chosen contact law, but it is not automatically independent of voxel size. A coarse voxel
must be identified either as a physical interaction cell or as a representation that
approximates finer interactions. Those are different commitments.

## 3. Lattice gases: the successes are real, the scope matters

### FHP establishes an important continuum bridge

**Source result.** Frisch, Hasslacher, and Pomeau's 1986 paper constructs Boolean lattice
gases whose large-scale, low-Mach behavior approaches incompressible Navier-Stokes dynamics.
Hexagonal symmetry resolves a two-dimensional isotropy defect of square-lattice HPP;
three-body collisions avoid an extra conservation restriction. The paper identifies
additional variants for nontrivial temperature and warns that a straightforward 3-D lift
has its own isotropy problem.
[Original PRL paper, 56, 1505–1508, 1986](https://link.aps.org/pdf/10.1103/PhysRevLett.56.1505).

**Derivation for the basic equal-speed gas.** If every particle has mass `m` and speed `c`,

```text
K = ½ m c² N.
```

Conserving `N` conserves `K` automatically. Independent local thermal energy cannot vary
through a speed distribution because this alphabet has only one speed. Fluctuating flow
directions can still supply microscopic disorder, pressure, and viscosity. That is useful,
but it does not mean binding gain can become higher-speed matter while conserving mass.

Adding rest states, multiple speeds, or internal resource bins changes the available
events. Appropriate collision families, lattice symmetry, and ensemble behavior then need
to be studied again. Six axial channels on a cubic PWC grid do not inherit the FHP theorem
merely because the number of directions is also six.

### Momentum and kinetic energy do not imply stable chemistry

**Derivation.** In a freely streaming gas with instantaneous elastic encounters, preserving
`N`, `P`, and `K` supplies neither a lower-energy bound separation nor a barrier to drifting
apart. Conservation constrains outcomes; cohesion needs an interaction mechanism. A
particular collision rule may admit a persistent orbit or a frozen pattern, but that is
additional structure of the rule, not a consequence of the three conserved totals alone.

The same warning runs in the other direction. Attractions without a short-distance
constraint can collapse everything into a dense clump. Site or channel exclusion is a
simple discrete repulsion; a finite capacity is another, but its density scale is a
fundamental law choice. An attractive nearest-neighbor contact energy plus exclusion may
produce clustering, yet need not produce a shearing solid, selective bonds, recoverable
machines, or a broad family of materials.

For illustration, `U_contact = -J ∑_<ij> n_i n_j`, with `n_i∈{0,1}`, rewards contact while
exclusion forbids double occupancy. Its surface energy favors compact aggregates. This is
a **proposed simple energy**, not a demonstrated PWC model. It can phase-separate into
featureless dense and dilute regions; that result would fall short of the user's ambition.

### Known interaction extensions expose the accounting challenge

**Source result.** Yepez's 1993 manuscript, published in 1996, mediates longer-range
attractive and repulsive interactions through momentum exchanges and couples them to heat
bath particles. Its discussion explicitly identifies imperfect interaction-energy tracking
for the single-speed formulation and presents more complete microcanonical coupling as
future work. The fixed-temperature construction also carries qualifications about
reversibility.
[Original manuscript hosted by Margolus](https://people.csail.mit.edu/nhm/Yepez-Waterloo93.pdf).

**PWC implication.** Clumps, a nonideal equation of state, and even apparent elastic order
are not sufficient evidence of a closed energy account. A convincing demonstration should
calculate energy before and after every kind of encounter, transport, and boundary event.
Useful inspiration should preserve the author's stated limits rather than erase them.

## 4. Reversible chemistry and computational structures offer different pieces

### Microcanonical heat is more than an arbitrary “temperature” field

**Source result.** Creutz's deterministic Ising dynamics attaches finite integer demon
variables to lattice sites. Admitted spin changes exchange energy exactly with those
variables. Checkerboard updates make the elementary move reversible; temperature is an
emergent statistical quantity rather than an externally supplied value.
[Original deterministic Ising paper, 1986](https://latticeguy.net/mypubs/pub088.pdf).

**PWC implication.** A single conserved energy can partition between configuration and
other local degrees of freedom. This supports the reservoir approach, while not establishing
translation, chemical selectivity, or physical momentum. Calling a bin “kinetic” does not
cause its contents to accelerate a voxel.

### Aggregation and evaporation can share an exact ledger

**Source result.** D'Souza and Margolus give a local reversible aggregation model in which
growth releases heat tokens and heat absorption permits evaporation. Diffusing tokens store
both resource and information needed for reversal. Their update permits these transitions
only with exactly one crystal neighbor, shaping the aggregate and preventing arbitrary
bond-number changes.
[Original preprint, 1998](https://arxiv.org/abs/cond-mat/9810258),
[published PRE paper, 1999](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.60.264).

**PWC implication.** This is an excellent small demonstration of paid phase change. Its
gas/crystal tags and single-neighbor condition also show why it is not automatically the
desired universal chemistry. A PWC version should derive gain from generic interactions
and calculate all changed contacts, instead of defining each attachment to release one
token regardless of the surrounding arrangement.

### Stable computational mirrors are not a demonstrated solid phase

**Source result.** Fredkin and Toffoli's conservative-logic construction uses elastic balls
and reflectors to realize computation. Margolus's 1984 partitioned CA supplies stationary
four-bit squares from which mirrors can be constructed under the same local rule as moving
signals. These are carefully prepared computational arrangements.
[Original MIT report, 1981; journal publication followed in 1982](https://dspace.mit.edu/entities/publication/19826ace-4422-4f48-b134-5be130d5bcf5),
[Margolus's original 1984 paper](https://people.csail.mit.edu/nhm/bbmca-physica.pdf).

**PWC implication.** The four-bit structure's stability is a property of its discrete rule,
not evidence that an attractive potential has produced a stress-bearing solid. Digital
stability could still be sufficient for a fictional game. The questions are whether players
can assemble, disturb, repair, move, and reuse it through ordinary laws, and what outside
support is needed for routing.

**Source result.** Margolus's soft-sphere models explain the earlier CA's loss of momentum
conservation and construct models in which signal streams replace infinitely immovable
mirrors. Their computational constructions maintain more of the momentum account, but
require spatial and timing arrangements.
[Original author paper, arXiv upload 2008; earlier book versions 2002/2003](https://arxiv.org/abs/0806.0127).

**PWC implication.** Signals doing the job of mirrors are an interesting fictional
technology. They also consume continuing simulation work. Computational universality and
ordinary hardware efficiency do not demonstrate natural assembly, inexpensive sleeping
matter, or a practical voxel world's throughput.

## 5. China-affiliated work that explicitly couples thermal and mechanical behavior

**Source result.** Li, Zhang, Lin, Xiao, Liu, and Chen develop many-body dissipative particle
dynamics with energy conservation and a temperature-dependent attractive interaction.
The published affiliations include Tongji, Tsinghua, Shanghai University of Engineering
Science, and Hong Kong Polytechnic University. They compare droplet, transport, and
interfacial properties with water measurements using fitted or derived parameter functions.
[Original 2022 paper and affiliations](https://doi.org/10.1007/s10483-022-2840-7).

**PWC inference.** This demonstrates that thermal transfer and cohesive particle motion
can be addressed within one mesoscopic framework. It also demonstrates the price of a
targeted physical model: velocities, neighbor interactions, thermal variables, parameter
functions, and numerical integration. It is inspiration for an energy audit, not a proposed
copy of water's behavior or a universal fictional-material law. The publication abstract
and affiliations were verified; an unavailable full PDF is not treated as evidence for
exact integer conservation or a particular implementation.

The previous [China investigation](chinese-fiction-research.md) also considers active and
odd-elastic matter. External driving in those models must enter this same ledger if coupled
to chemistry. Adding nonreciprocity cannot silently act as a perpetual reservoir refill.

## 6. One minimal future experiment: chemistry pays for moving energy carriers

**Proposal only; no new package now.** Combine generic compatibility rearrangement with
resting integer resource bins and moving directional resource quanta, in one small native
laboratory. Let each packet carry energy `ε` and an explicitly defined directional momentum
`p0 d`. Write

```text
E_total = ∑ e_i + ε N_packet - k Φ.
P_packet = p0 ∑ d_packet.
```

This is a minimal extension of the existing `E=∑e-Φ`: the stored resource now has a moving
representation. It is not yet a complete moving-matter model. In particular, the packet
momentum account excludes future voxel recoil unless that recoil is added explicitly.

Use three universally described event types within this single law:

1. An occurrence-preserving rearrangement changes `Φ` by `g` and adjusts the admitted
   local resource by `kg`, with the entire reaction committed atomically.
2. A local resting resource emits a pair of opposite packets only when two outgoing
   slots exist and `e≥2ε`; subtract exactly `2ε`. The inverse absorbs an opposite pair
   and adds exactly `2ε`, provided the receiving capacity permits it. The pair has zero
   net packet momentum; its direction is supplied by the explicit saved event phase.
3. Moving packets transport and scatter under a count- and momentum-preserving primitive
   on the closed chart-interior laboratory. Reflection from fixed terrain is excluded from
   whole-world momentum claims; a later mirror must receive recoil or use other packets.

**Derivation.** Each event has `Δe + εΔN_packet - kΔΦ = 0`. Thus an uphill reaction can
consume previously transported resource, and a favorable reaction can launch outgoing
motion through the same account. Remainders stay in `e` rather than being discarded when
`kg` is not a multiple of `2ε`. Energy creation at full slots is refused, not clipped.

For a reversible version, emission and absorption must be paired by an invertible local
selector. A rule that always absorbs a pair and always emits from an empty pair can be an
involution on the appropriate admitted sector, but only if the energy and slot conditions
identify that sector consistently in both directions. Reaction proposal selection likewise
needs an inverse or an explicit nonreversible interpretation; the existing gain-preferred
search is not automatically a microscopic bijection.

**Why this experiment is useful.** It can demonstrate the sequence
`chemical preparation → released resource → moving signal → absorbed resource → otherwise
unavailable chemical change`. That is a complete energy conversion test between two forms,
using generic fit arithmetic rather than named fuels and receivers.

**What it would not demonstrate.** Independent velocity bins for massive matter, pressure
on a movable wall, mechanical acceleration, a stable bond, a rigid rotor, elastic strain,
thermal equilibrium, or a universal chemistry. A visible outgoing packet should not be
advertised as chemical energy accelerating the same material occurrences. That stronger
claim requires occurrence velocities, recoil, and cohesive geometry.

A later moving-matter branch could add rest and multiple-speed occupation states, with
`K=∑½m|v|²`, to the same potential. Then every transport step changing neighborhood
potential needs payment, and every collision needs a legal joint solution for mass,
momentum, and energy. Free streaming cannot ignore a changed potential and hope the next
reaction repairs it. The event alphabet and its finite capacities should be chosen only
after this smallest conversion experiment provides useful evidence.

## 7. Conservation can be exact while the universe is unusable

**Energy quantization.** A small velocity alphabet can leave few states with the required
`N,P,E`. For example, equal-mass one-dimensional velocities in `{−1,0,+1}` offer no
nontrivial zero-momentum two-particle speed distribution besides the opposite moving pair
and the two rest particles, whose kinetic energies differ. A finite local reservoir can
bridge that difference; without it, many events freeze. Larger alphabets increase state,
collision families, and memory traffic.

**Collision mixing.** Count, momentum, and energy conservation do not ensure that a
collision graph connects all accessible states. Extra conserved line populations, parity,
species constraints, scheduler phases, or occupancy exclusions can trap activity. A
deterministic reversible model may have many tiny orbits; a stochastic conserving model
may still sample the wrong measure. Conserved totals are the first check, not the last.

**Temperature and work.** A local scalar `e` is an amount of energy, not necessarily a
temperature. A temperature claim needs an observed distribution and an equation connecting
it to accessible degrees of freedom. Equal total energy in directed motion and disordered
motion can support different useful work. A closed equilibrium state cannot continuously
power machines solely because its conserved energy is nonzero.

**Stable structures.** A rigid phase needs a response to displacement and shear, not just
a dense region or stable color. A bound preparation must be perturbable and recoverable;
if every impact irreversibly destroys it, players cannot maintain machinery. Conversely,
perfectly frozen states that cannot interact cannot transmit useful information. We should
test those failures before expanding a material taxonomy.

**Atomicity.** If a host refuses a material mutation, the model must also refuse all
associated resource and packet changes. A conservation proof over ideal proposals does
not cover partially committed world edits. Full slots, unrepresentable configurations,
unloaded cells, changed neighbors, and interrupted partitions belong in the event contract.

## 8. The weak-hardware constraint changes representation choices

A block configuration currently remembers an unordered multiset of at most 32 resource
elements. It does not remember each occurrence's position within the voxel, velocity,
impact parameter, or bond graph. Replacing a configuration with an actual microstate adds
information even when both have the same visible block ID.

As raw storage arithmetic, seven velocity possibilities require at least three additional
bits per independently tagged occurrence: 32 occurrences add at least 96 bits, or 12 bytes,
per full cell before positions, energy, phase, coordinates, and data-structure overhead.
This is a lower bound, not an allocation or performance estimate. One million active full
cells would add at least 12 MB for these velocity tags alone. Real implementations may need
far more because tags must remain associated with element labels and variable occupancy.

Subdividing each rendered voxel into `s×s×s` microscopic sites multiplies the site count by
`s³`. If a fixed physical distance must still be traversed in the same time and signals
advance one microsite per tick, the required tick count also grows with `s`; raw update work
can then grow like `s⁴` for a fixed volume and duration. This arithmetic is an assumption-based
cost illustration, not a measured PWC benchmark. A molecular interpretation of every block
can defeat the massive-world ambition before any collision algorithm is considered.

Vacuum can sleep if its local rule fixes it exactly. A warm bulk solid may keep many
microscopic states moving forever, even when its visible shape is stationary. Sparse maps
help only where activity is sparse. Dense active chunks should use contiguous compact
storage and bitwise batches; those are engineering candidates, not evidence that a desktop
can update an enormous thermal world in real time.

Replacing distant microscopic activity with a coarse field is possible, but must preserve
the relevant resource, momentum, memory, and response when refined. A coarse approximation
is an explicit mod policy with a measured error, not an invisible deletion of heat or waves.
Networking likewise needs dynamic payloads, event phases, inputs, and deterministic chunk
activation: model identity alone is not state replication.

## 9. How the current two new models fit this longer search

Dipole occupation explores collective mobility and exact `Q,D,∑q²`. Its norm is not the
chemical potential or a demonstrated kinetic energy. It has no heat exchange, fit-driven
binding, velocity distribution, or mechanical recoil. Its frozen single occupation provides
an algebraic kind of stationary memory, not an attractive solid. It could inform the choice
of constraints in a future universe, but combining its moments with ordinary mechanics may
overconstrain motion and should not happen by default.

Relational frames explore quiescent commuting matter and propagation through noncommuting
encounters. Exact class counts and pair products do not supply a scalar energy ledger.
Stationary carrier preparations are not energetic cohesion, and a moving conflict is not
necessarily moving mass. Its useful lesson is that a universal law can make many stationary
backgrounds responsive without an authored wire type. It does not validate quantum anyons,
gauge physics, thermal chemistry, or mechanical machines.

Keep these experiments separate. The longer ambition asks whether transport, bonding,
heat, and mechanics can come from the same conserved state and generic event laws.
Independent experiments reveal which ingredients are worth attempting to unify. Their
limits should guide that attempt rather than be hidden by calling everything “physics.”

## 10. The next small evidence, when another implementation is warranted

The immediate research outcome is the proposed energy-conversion bridge above. Its first
checks should be narrowly informative: one favorable rearrangement emitting moving
resource, transport to another site, absorption, and a previously unaffordable reverse or
unfavorable rearrangement, all with an unchanged total ledger. Refusal at full capacity must
leave every coupled component unchanged. No extensive benchmark or new mod is claimed here.

The stronger universe candidate should later survive a different test: the same microscopic
matter forms a stable preparation, receives momentum and energy from an impact, stores or
transports the disturbance, and returns a manipulable output without fixed magical walls.
If that cannot be demonstrated with a small universal event alphabet, adding familiar
temperature, electricity, machines, or named bonds will not resolve the foundational gap.
