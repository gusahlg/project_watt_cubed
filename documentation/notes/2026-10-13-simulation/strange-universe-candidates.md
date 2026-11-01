<!-- SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors -->
<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# Strange universes: changing frames and changing adjacency

Research and independent criticism, 2026-10-13. These are fictional laws motivated by
primary research and official fiction texts, not reproductions of quantum physics.
The frame candidate is suitable for a small independent mod. Changing adjacency is a
more speculative direction whose architectural and gameplay costs are unresolved.
The separate moment-conserving scalar experiment is being investigated elsewhere.

October 14 implementation status: frames and dipole are now independent canonical mods.
See the [focused continuation](focused-universes.md) for actual checks and limitations.
The changing-adjacency proposal below remains unimplemented.

## 1. A stationary medium whose internal identities change during interaction

**State.** Each site stores a permutation of three markers, `g ∈ S₃`. There are six
states: identity, three transpositions, and two three-cycles. Store a permutation
array and calculate composition and inverse; no collision table or named materials
are necessary. This is an internal frame field, independent of the existing block
configuration. A prepared region and any injected disturbance are external sources.

**One rule.** For an oriented neighboring pair `(a,b)`, do nothing when `ab = ba`.
Otherwise apply

```text
R(a,b) = (b, b⁻¹ab).
```

Composition uses a declared convention, `ab = a ∘ b`; changing that convention,
pair orientation, or matching schedule changes the law's compatibility identity.
Use deterministic disjoint pair phases so a cell participates in at most one
interaction per phase. The inverse is the same commuting test followed, for an
eligible pair `(c,d)`, by

```text
R⁻¹(c,d) = (cdc⁻¹,c).
```

The gate does not lose information: noncommutation is preserved by `R`, so the
inverse can tell which branch was taken. Reverse a complete evolution by reversing
both the phase order and each eligible pair operation.

**What actually survives a collision.** The ordered pair product is unchanged,
because `b(b⁻¹ab) = ab`. Conjugacy-class populations are unchanged: one frame is
retained and the other is conjugated. Identity sites are individually invariant,
not merely conserved in number. Consequently every connected component of
nonidentity sites retains its own conjugacy histogram. An ordered product along
a fixed one-dimensional chain survives adjacent updates; there is no comparable
global ordered-product conservation asserted for a three-dimensional voxel box.
These are exact algebraic invariants. They do not identify a work or heat quantity.

**An internal change beyond permutation transport.** Let `A=(01)`, `B=(12)`,
and `C=(02)`. Repeated eligible interactions on one pair give

```text
(A,B) → (B,C) → (C,A) → (A,B).
```

After two swaps the participant positions have returned, but their frames have
changed. A law that only permutes whole immutable configurations cannot do this.
The pair has a three-step recurrence without an authored oscillator rule. A
transposition paired with a three-cycle has a four-step orbit. These tiny local
orbits are potential clock ingredients, not proof of a useful machine.

**Stable media and candidate signals.** Homogeneous frames commute and remain
still. The two three-cycles also commute with each other. Different transpositions
do not commute, so a disturbance in a homogeneous transposition medium can change
neighbors. For the states above, `(B,A) → (A,C)` moves a disturbance to the second
site, whereas `(A,B) → (B,C)` changes both sites. History changes identity and
eligibility, potentially permitting rewrite memories, collision counters, and
signal junctions without named gates. No reusable wire, logic gate, amplifier,
or autonomous large machine has yet been demonstrated.

**Killer objections and limits.** Identity gaps cannot be crossed: no nonidentity
frame can enter an identity site. The support of matter in this field never moves,
grows, or assembles itself. Its plausible role is information dynamics in prepared
stationary media. Pair orientation and phase ordering produce directional bias;
symmetry of the marker labels does not imply spatial isotropy. Reversibility gives
finite closed systems recurrence, but does not guarantee readable signals or
enjoyable complexity. With only six states, useful circuits might prove extremely
restricted or fragile. Supplying an algebra is not yet supplying a physical reason
for choosing this update rather than another invertible update.

**Why the name should be frames, not braids.** The ungated conjugation map obeys
the braid/Yang–Baxter relation, but swaps every pair, so carrier routes are prescribed
independently of matter. The commutator gate allows stable structure, at the cost
of that relation. A small independent enumeration checked all 36 pair inverses;
the ungated map had zero Yang–Baxter failures among 216 triples, and the gated map
had 48. These checks concern algebra, not performance or gameplay. The proposed
mod is a classical local frame law; it is not an anyon model, a quantum simulator,
a gauge theory, or protected topological memory.

**Cost and next experiment.** Permutation operations have constant tiny integer
cost and no magnitude growth. The dominant costs will be field storage, address
lookups, phase scans, persistence, and visualization. A bounded seven-cubed lab
can first test pulse survival in a prepared strand, interaction at a junction,
and restoration by exact inverse. Whole phases need atomic validation before
commit. Saves must retain every frame, the matching phase, and any partial-phase
cursor. A future active-interface cache must wake neighboring interfaces when
frames change and preserve the declared global phase semantics.

**Inspiration, with the boundary kept explicit.** Xu and collaborators at Zhejiang,
Tsinghua, and Nankai demonstrated Fibonacci-anyon braiding using a superconducting
quantum processor. Their quantum string-net state and fusion measurements are
substantially richer than this six-state classical field. The transferable idea
is that interaction history can transform internal information, rather than only
move objects. Their results do not establish computational universality or fault
tolerance for the proposed law. [Primary paper](https://arxiv.org/abs/2404.00091),
[participating laboratory's account](https://cqi.tsinghua.edu.cn/en/info/1018/2930.htm).

Li, Wang, and Zhang's earlier China-affiliated work constructs Yang–Baxter solutions
from braid-group representations. It supplies mathematical context, not a derivation
of the gated `S₃` law. [Primary publisher record](https://ctp.itp.ac.cn/EN/Y1994/V21/I2/167).

Egan's *Schild's Ladder* imagines matter and motion as changing graph patterns,
with more than one stable vacuum possible under a wider law. This suggests looking
for stable substrates and disturbances governed by the same operations. The novel's
quantum graph theory is fiction; neither its physics nor its rich ecology follows
from our frame rule. [Author's official excerpt](https://gregegan.net/SCHILD/00/SchildExcerpt.html).

## 2. Make nearness itself a conserved resource

**Distinctive state.** Replace a fixed adjacency graph with a degree-bounded graph
`G=(V,E)` whose nodes also carry bounded conserved integer occupancy `q`. Distance
is shortest-path hop count. This differs from changing the length or strength of
an existing bond: interactions can change which sites are neighbors at all.

**A tiny local geometry operation.** Select four distinct nodes forming a path
`u-v-w-x`. Retain `(v,w)` and replace `(u,v),(w,x)` with `(u,w),(v,x)` when the new
edges do not already exist. Every node's degree and the total edge count survive.
Both local motifs connect all four nodes, so this specific switch also preserves
connected components. Define the contact score

```text
C(G,q) = Σ_(i,j in E) q_i q_j.
ΔC = (q_u − q_x)(q_w − q_v).
```

Accept a switch exactly when `ΔC > 0`. This makes high occupancy prefer proximity
using one algebraic rule. For fixed `q`, the score is bounded and strict ascent
must settle. Sustained adaptation therefore needs actual conserved traffic.
One minimal provisional transport step swaps neighboring `q` values, preserving
their histogram and total while changing the rewiring objective. This is a proposal
to explore, not a demonstrated coherent energy model. Swapping whole occupancies
is not yet fluid transport, and selecting paths is part of the dynamics, not free
implementation detail.

**Possible machines, not claims.** Traffic-dependent shortcuts could form learned
routes, movable bottlenecks, spatial memory, and mechanisms whose operation changes
reachability. Uniform occupancy gives no adaptation. Degree conservation prevents
unbounded hubs but does not prevent a world from forming dense pockets or unusable
bottlenecks. No global energy invariant or computational universality is known.

**Why to defer implementation.** A bounded-degree switch is cheap locally, but
graph ownership, atomic four-node changes, duplicate-edge checks, deterministic
conflict resolution, loaded/unloaded interfaces, and remote endpoints are material
costs. More seriously, graph distance would disagree with rendered voxel positions,
collision, player navigation, and existing port rotations. Treating the graph as
only an invisible network evades this difficulty but gives up the stronger proposal
that space itself changes. A separate tiny graph visualizer should precede a game
integration. This direction is genuinely different, but currently less complete
and less practical than the frame experiment.

Discrete Ricci-flow research by Lai, Bai, and Lin at Renmin, Southeast, and Tsinghua
uses transport-related graph geometry to expose structure and includes normalization
to avoid collapse. Our integer edge switch is not their curvature or flow. The
research provides both inspiration and a warning: making geometry evolve requires
explicit control of degeneracy. [Primary paper](https://www.sciencedirect.com/science/article/pii/S0378437122002242).
Related finite-graph flow analysis admits finite-time blowup in some cases; elegant
geometry dynamics alone does not guarantee a viable world.
[Hua, Lin, and Wang's primary preprint](https://arxiv.org/abs/2402.07475).

Hao Jingfang's *Folding Beijing* makes spatial access and the timing of access central
to its society. The useful fictional prompt is to let changing adjacency alter what
players can reach and communicate with, rather than merely decorate a static world.
Its city machinery is not a fundamental interaction model.
[Official publisher text](https://www.uncannymagazine.com/article/folding-beijing-2/).
