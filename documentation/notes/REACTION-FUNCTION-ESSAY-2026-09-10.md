# The reaction function: what it must be, what it is, and where to take it

*An essay for the Emergent Material Model, written after the first day of living with law v1
(2026-09-10). It argues for one change of viewpoint — the reaction function should be the descent
of an energy landscape, not a hand-shaped force — and works through what follows from that for
every open question in the spec.*

## 1. Why the function is the whole game

Everything that used to be authored is now downstream of one function. Regions (what worldgen
places) are its fixed points. Properties (solid, liquid, glow, hardness) are readings of it
against probes. Crafting is the player applying it on purpose. Reactions are the world applying
it by itself. A recipe book, in this model, is a map of paths through the function's landscape,
and a mod that adds "chemistry" adds no rules at all: it adds matter (worldgen distributions),
machines (event sources) and names.

That is a great simplification, and it puts all the pressure in one place. The function has to
carry *stability* (a world that does not dissolve), *richness* (enough distinct outcomes to be
worth discovering), *legibility* (nearby matter behaves alike, so what the player learns
transfers), and *boundedness* (no cascade that eats a server, no material table that grows
without limit). Those pull against each other, and the day's numbers show exactly where v1 sits
on that tension.

## 2. Properties before shape

Before arguing about curves, it is worth writing down what the function must *guarantee*,
because the guarantees are what the rest of the game can build on. A curve is a parameter; a
guarantee is architecture.

1. **Locality and determinism.** Integer arithmetic, no floats, no hash-map order, and one
   result for the same inputs on every machine. v1 has this. The scheduler adds one more rule:
   several origins acting on one target in the same generation must give an order-independent
   result. v1 does this by averaging the origins.
2. **Boundedness per step.** No single reaction moves an element more than `max_step` along
   any axis. v1 has this (6).
3. **Termination.** Left alone, any finite piece of matter reaches a state in which no event
   changes it, in a bounded number of generations. v1 does *not* have this: 36 % of random
   cascades are quiescent within 64 generations; the rest oscillate. This is the property the
   world needs most and the one v1 lacks.
4. **Basins, not points.** Matter at rest must stay at rest under *small* perturbations: a
   family of nearby configurations that all sit still and all react the same way. This is what
   makes "rock" a thing rather than one coordinate. v1's families settled at spread 1, i.e. each
   family is a single point; the lab counts 5 614 families of size ≤ 6 where the target band is
   8–200 members.
5. **Continuity.** Nearby elements have nearby influence (the lab's *similarity* metric, v1
   passes with a maximum jump of 2). Without it nothing the player learns generalises.
6. **Proliferation control.** The number of distinct configurations reachable by ordinary play
   must grow sub-linearly in the number of reactions. v1 achieves this only with `quantum = 4`
   (snapping to a coarser lattice); at quantum 1 every reaction can mint a new exact coordinate,
   and the registry is a `u16`.
7. **Conservation.** Something should be conserved so that nothing can be manufactured from
   nothing and no cycle can run forever. v1 conserves *count* (the target keeps its length) and
   nothing else.
8. **A reason for irreversibility.** If every craft could be undone by the same touch, discovery
   would mean little. There should be steps that are cheap one way and expensive the other.

A function with 1–8 can be handed to regions, crafting and the scheduler without further
promises. The rest of this essay is about getting 3, 4, 6, 7 and 8 from one idea instead of
four patches.

## 3. What v1 actually is

For precision, since the lab and the sweep talk about it in knots:

- Per axis, the influence of an origin element on a target element is a signed response of the
  relative position δ: a dead zone (|δ| ≤ 10 → 0), repulsion (10..22, peak −8 at 16), a flat
  rest band (22..26 → 0), attraction (26..64, peak +10 at 40, 4 at 56), and inert beyond 64.
- The four axis responses are mixed by a fixed Q4 matrix (16 on the diagonal, 4 on the next
  axis, cyclic), then divided by 16.
- A configuration acts through the *mean* of its elements' influences; several origins in one
  generation are averaged after each is scaled by its event's strength (Moved 128, NewContact 96,
  Collision 255, ExternallyChanged 64, out of 256).
- The result is clamped to ±`max_step` per axis and applied to the target under the boundary rule
  (clamp to 0..255) and the quantum. Only the target moves; the origin is unchanged.

Its history explains its shape. The first curve was Lennard-Jones-like and left no stable family
anywhere: every boundary in the world reacted. The band-limited curve fixed that. The lab then
showed cascades never settling because elements oscillated ±1 across the zero crossing, and the
flat rest band was added to give them somewhere to stop. Each fix was a good local move. Together
they describe a *force field drawn by hand*, and hand-drawn force fields have a specific failure:
nothing guarantees they are the gradient of anything.

## 4. Diagnosis

Three things in v1 are structural rather than tuning problems.

**Overshoot is built in.** The rest band is four units wide (22..26) and a Collision step is up
to six units. An element attracted from 30 toward the band can land at 24, or, from 32, jump the
band entirely to 26 and then be pushed back. The sweep's best candidates widened the band to
25..27 with peak 32 and `max_step` 7 and got 100 % quiescence on the reduced card — but with
family means of 1.0. That is the same tension from the other side: a band narrow enough to be a
point is quiet; a band wide enough to hold a family gets stepped across. Any scheme in which the
step size and the flat zone are independent parameters has this problem. A step that *shrinks as
it approaches rest* does not. In control terms v1 is bang-bang with a dead zone; what is wanted
is proportional control: near the rest distance the response should cross zero *linearly*, so the
last steps are small and the element lands rather than bounces.

**The field has curl.** The mixing matrix is cyclic (axis 0 feels axis 1, axis 1 feels axis 2, …,
axis 3 feels axis 0). A cyclic, non-symmetric coupling is the textbook way to get a rotational
field: an element can be pushed around a loop in the lattice without ever descending anything,
which is exactly a cascade that never rests. Together with target-only mutation (a acts on b this
generation, b on a the next, with the same curve but from the other side) it produces two-body
orbits. No amount of knot tuning removes curl; only the structure of the coupling does.

**Species are coordinates.** Because every reaction moves an element by an arbitrary integer
amount, a product is a fresh coordinate, hence a fresh configuration id, hence a fresh material
with its own descriptor. Proliferation is then a property of play, not of the law: a busy server
mints ids at the rate players hit things. The quantum helps by snapping, but snapping to a
uniform grid is unrelated to where matter actually rests. The natural grid is the set of *rest
positions themselves*.

## 5. The proposal: the reaction is a descent

Define the pair interaction not as a force but as a **potential** `U(δ)` over the relative
position of two elements, and let the reaction move the target *down* that potential:

```
F(δ) = −∇U(δ)          (per axis, integer, from a table or an analytic profile)
```

This one change buys most of section 2 at once.

- **Termination (3).** If every reaction lowers the total energy `Σ U` over the affected pairs
  by at least one unit and the energy is bounded below, a cascade is a descent and must stop.
  That is a Lyapunov argument, and it is checkable in the lab: assert that the summed potential
  over a random neighbourhood never rises across a generation. The scheduler's FIFO queue then
  only decides *which* minimum a cascade reaches, never *whether* it stops.
- **Basins (4).** A local minimum of U with a flat or shallow bottom is a family by definition:
  every configuration in the basin rests, and every one reacts by sliding toward the same floor.
  Family size becomes a designed quantity (basin width), not an accident of a band's width versus
  a step's length. Worldgen samples basins; ores are what sits on the saddles between them.
- **Proliferation (6).** With proportional descent, elements *converge*; with a snap-to-floor
  rule after each reaction (the quantum becomes "the nearest basin floor when within reach"),
  products are basin floors. The reachable species are then the basins, a designed, countable
  set, and the registry stops growing after the world has explored them.
- **Irreversibility (8).** Two basins separated by a ridge give exactly the asymmetry the spec
  wants: sliding down into the deeper basin is cheap (any touch), climbing back needs the event
  with enough strength to cross the ridge. Which brings us to events.

### Events as temperature

The four event strengths are already the right knob, they only need the right meaning: an
event's strength is *how far up a slope the reaction may push*. A touch (NewContact, 96)
relaxes: it can only move an element downhill. A strike (Collision, 255) has activation energy:
it may carry an element over a ridge into a neighbouring basin, from which it will relax to that
basin's floor. Moved and ExternallyChanged sit between. Crafting then reads naturally — *strike
to change, touch to settle* — and the journal records sequences of strikes and touches, which
are paths on the landscape. Repeat counts are how many pushes a path takes.

### The rest of the open questions, answered by the landscape

The spec lists questions that must stay parameters. Each has a natural answer once U exists,
and each answer *stays* a parameter of U rather than a new mechanism.

- **Boundary: wrap or clamp.** Clamp makes 0 and 255 sticky walls: an element pushed against a
  wall stays there, so the walls become artificial basins and matter piles at the corners of the
  lattice. Wrap makes the lattice a torus; distances are taken within ±128 and U is periodic. No
  corner is privileged, which matches "coordinates in an abstract lattice". Recommendation: wrap,
  with U periodic by construction. (Reflect is a third option; it keeps a boundary but without
  sticking. It is a one-line variant if wrap ever proves confusing to explain.)
- **Cross-dimensional mixing.** Replace the cyclic Q4 matrix by a symmetric positive-definite
  metric `M` and let U depend on the quadratic form `δᵀ M δ` (or apply M inside the descent as a
  metric on the gradient). Symmetric mixing has no curl. It still lets axes couple — a step along
  axis 0 can pull axis 1 — but the coupling is a geometry of the lattice, not a rotation of the
  field. Everything the mixing matrix was for (materials whose axes are not independent) survives.
- **Target-only or both.** A descent moves *both* bodies in the physical picture (equal and
  opposite), and doing so halves the effective step, which is a good thing for landing. But the
  scheduler's generation semantics (every target is evaluated once with all its origins) is
  cleaner with target-only, and it is what makes the mean-of-origins order-independent. Keep
  target-only as the default; expose "both" as a parameter of U's application, and let the lab
  say whether landing improves enough to pay for the extra bookkeeping. Note that with a
  potential, target-only *still terminates* — each half-move is a descent of the pair energy.
- **Destruction.** Give configurations an energy cap. A reaction that would leave an element
  above the cap (equivalently, that would push it out of any basin's reach with no descent path)
  removes it instead: the configuration loses an element, and a single-element configuration
  becomes void. Matter can be lost but not created — creation belongs to worldgen alone. This
  gives strikes a cost, gives "wearing out" a meaning, and is the second guard on proliferation.
- **Event whitelist.** Unchanged in shape; each event kind is a temperature. Adding an event kind
  (a machine's "heat", a mod's "press") is adding a strength, not a rule.
- **Cascades.** With descent, a cascade's length is bounded by the energy it can release, which
  is bounded by how far from a floor the world was. Generated matter sits on floors, so cascades
  only start where players (or reactions) left matter uphill, and they end. The scheduler's
  budget and capacity remain as guards against a *mis-tuned* U; they are no longer what makes
  the world stop.
- **Ordered configurations.** The order in the list can mean structure: adjacent elements in the
  list are bonded, and a configuration has an internal energy — the sum of U over its adjacent
  pairs. Then "stable material" has a second meaning (internally at rest), a strike can *split* a
  configuration where an internal bond is above its cap (decomposition), and two configurations
  whose facing elements sit in each other's basins can *fuse* under contact, up to `CONFIG_MAX`
  (composition). This is the cheapest route to molecules the model has: no new function, only
  U applied inside as well as between. It is also where multiplicity earns its keep.
- **Differently-sized interactions → pairwise influence.** v1 uses the *mean* over the origin's
  elements, so a large configuration acts exactly like a small one. A *sum* bounded by
  `max_step` makes bigger matter push harder up to a saturation, which reads as mass: a boulder
  of eight elements bullies a pebble of one. The lab can compare the two; the descent property
  holds for either as long as the step is bounded.

## 6. What to keep from v1

Not everything is up for change, and the list of what stays is what makes this a law *version*
rather than a new model.

- Integer-only, wrapping arithmetic; the law as a stamped value with a fingerprint; the probes
  and the whole observation interface (solid, liquid, transparency, emission, hardness, friction,
  flow, acoustic) — the game reads readings, it never sees U.
- The scheduler: generations, Jacobi semantics, FIFO by arrival, capacity guard, refusable
  writes. It is a correct engine for any descent.
- Regions as a pure function of the law, re-derived with the pins whenever the law changes.
- The lab as the only way a law change is judged. The scorecard needs two new rows (below) and
  the sweep should search U's parameters instead of knot heights.

## 7. What to measure

The scorecard should say, for a candidate U:

1. **Descent.** Over 10 000 random neighbourhoods, `Σ U` never rises across a generation under
   any event kind (a hard PASS/FAIL).
2. **Termination time.** Distribution of generations to quiescence for random matter; target:
   100 % within 32 generations at `Budget::DEFAULT`.
3. **Basin census.** Number of basins, their sizes (members that rest), the depth of each floor;
   target: 20–200 basins with mean size ≥ 20 members, so worldgen has variety and families are
   real.
4. **Reachability.** From worldgen matter, the set of basins reachable by sequences of strikes
   and touches (the craft graph), its size and its depth (longest shortest path); target: a few
   hundred reachable species and a depth of at least four, so discovery has chapters.
5. **Irreversibility fraction.** Share of reachable transitions whose reverse needs a stronger
   event than the forward step.
6. **Proliferation under play.** Distinct configurations after 100 000 random gameplay events on a
   generated map; target: bounded by the basin count times a small constant, with no quantum
   trick.
7. **Continuity** and the existing observation bands (liquids, glow, transparency fractions)
   unchanged.

The day's sweep infrastructure (`sweep`, `explain`) already searches knots and prints curves;
searching a potential is the same loop over different parameters: well spacing, well depth,
ridge height, basin floor width, the metric `M`, the energy cap, and the four temperatures.

## 8. Migration

Law version 2 stamps the potential (a short knot table for U rather than for F, the metric, the
boundary, the cap, the temperatures). `from_stamp` refuses v1 as it refuses everything that is not
the current version today, so no save migrates — acceptable before the game has players, and the
reason to make this change soon rather than later. Regions are re-derived; every byte pin is
re-derived once; the visual map is unchanged in form but will place families on floors, so the
world's look shifts with its chemistry. Order of work: lab first (descent and termination rows,
then a sweep over U), then the crate (`kernel` gains U and a descent step; `interact_many` keeps
its signature), then regions and pins, then the game, which changes nothing but the version.

## 9. A closing thought

The model's founding sentence is that matter has no authored properties, only coordinates and a
law. v1 kept the letter of that and lost a little of the spirit: its law is a *shape* someone
drew, and the world's stability depends on the drawing. A potential is the same idea one level
up: the designer shapes a landscape, and the law is what a landscape does to things placed on it.
Valleys are materials, ridges are recipes, temperature is what the player brings, and the
player's map of the valleys is the journal. Nothing in the game needs to know any of this;
it only needs the guarantees, and the landscape gives them for free.
