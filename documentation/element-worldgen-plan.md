# Element-first world generation — the plan

Status: **LANDED 2026-07-12** (`src/world/placement.rs` + the generator
rewiring in `generation.rs`), including the Phase-4 follow-ups.

**v3 (alien pass, same day):** Wood/Leaves blocks and Earth-style trees
retired entirely — terrain now emits ONLY element unions the table derives,
no decoration overlay, no named blocks. Elements recolored to an alien
palette (teal Organic, violet-grey Soil, slate Stone, ash Sand, teal Water,
mint Lumin). Crust varies by biome (sandy desert / frozen snowy), and the
top ground cell scatters luminous growth (Organic+Soil+Lumin tufts on the
plains, Sand+Phosphor sparks in the desert) — the night face of the planet.
Landform knobs pushed bolder (common tall terraces, blade-thin ridges).
The material-parity census gate ended here by design (v2→v3 deliberately
moves materials); geometry invariance remains the load-bearing test.

Deltas from this document as written:

- The palette cap was lifted FIRST (BlockId u16, 16,384 ids, per-chunk
  palettes — the doc's "sanctioned future lift"), so C1's arithmetic relaxed:
  `ENUM_CAP = 1024`. The builtin table enumerates 50 new naturals (45 ground
  pairs, the island pair, {Soil,Organic}, {Soil,Clay}, {Soil,Sand},
  {Stone,Obsidian}) — registry lands at 75 ids, inside the doc's estimate.
- The generator had grown past the doc's parity table (snow, deserts, water
  tables, overhangs, ravines, trees). The vocabulary adapted: one
  `SurfaceKind` axis {Grassy, Shore, Snowy, Desert, BeachEdge}, plus `Flood`
  and `Overhang` contexts (Water is an element and floods via the table).
  Trees stay a decoration overlay, as the doc intended.
- Stream B is DERIVED from the same authored rows (rarity ÷8), not a second
  stream field in the vocabulary. Arity ≤ 2 holds by the two-stream dedup.
- Cave-wall Lumin (the vision-note idea) landed with VERTICAL adjacency only
  (cavern floors/ceilings): one column, two carve reads post-roll, and the
  deep Uniform(stone) proof needs just two extra dormancy sups (chunk
  above/below — same columns). Beach edges dither at +1 (1/2) and +2 (1/4).
- Guards: save format v5 carries `worldgen_version` (v4 files load and warn);
  `PROTOCOL_VERSION` bumped to 4 (mixed peers error instead of desyncing).
- Tests: the geometry/material census (frozen legacy picker), stream-B
  statistics, dither bands/rates, cave-wall adjacency invariant, proof
  soundness vs the dense fill, v4 save compat — all in-tree and green.

Discussed 2026-07-07; supersedes nothing — it *completes* the block-hierarchy
vision by making terrain speak the same language as everything else.

## The vision

Today `generation.rs` is the last subsystem that thinks in named blocks: it
resolves Grass, Dirt, CoalVein, … against the registry and paints them from
hand-written branches (`ground_block_carved`, `island_block`). Everywhere else
in the game a block *is* its element composition and every behaviour is
derived from it.

The change: world generation places **elements**, not blocks. Every element
carries **placement preferences** — where in the world it wants to exist.
Generation asks, per cell, "which elements want to be here?" and the answer
*is* the block:

- **several elements** → the block is their natural mixture,
- **exactly one** → a pure block of that element,
- **none** → air.

Formally: `block(cell) = Composition::natural(sorted set of placed elements)`.

Nothing else is needed. Grass stops being a thing we place; it *emerges*
where the Organic and Soil preferences overlap (the top of the ground). An
iron vein is where the Iron preference's sparse roll lands inside Stone's
band. A cell where Iron *and* Coal both land is a Stone+Iron+Coal block
nobody authored — with derived colour, texture, properties, and possibly
reactions, all for free, because the derivation pipeline was built for
exactly this.

### Why this is the right move

1. **Philosophical closure.** IDEA.md line one: "The world is made of voxels
   consisting of different elements." After this change that sentence is
   literally how the generator works, not just how the registry stores its
   output.
2. **Worldgen becomes data.** The ore table, sand beaches, ice caps, soil
   crust — today branches in code — become rows in a placement table. Adding
   a material to the world becomes: one `elements!` row + one placement rule.
   That is the same moddability shape the menu work established (core owns
   the model, content is data).
3. **Emergent variety at zero authoring cost.** Overlap veins (multi-yield
   finds), transition bands (Sand+Soil beach edges — follow-up), and
   reaction-bearing terrain (a Sulfur+Coal seam that is *reactive* — see
   sidenotes) all fall out of the union rule.
4. **It deletes special cases.** The eleven `*Vein` entries in `blocks!`,
   the seams array, the sand/ice branches — all subsumed by one engine
   reading one table.

## Constraints the design must honour (agreed in discussion)

These four came out of the design discussion and are treated as hard
requirements, not preferences:

- **C1 — Palette stays bounded.** `MAX_BLOCK_TYPES = 256`; dense chunk cells
  are `u8`; the mesher stamps the block id into `Vertex::color.a` (u8); the
  texture array has one layer per id. The reachable combination set must be
  bounded *by construction* and provably ≪ 256 (arithmetic below). We are
  NOT lifting the cap for this task (memory analysis below explains why we
  could, later, if wanted).
- **C2 — Deterministic BlockIds.** Generation is a pure function of
  (seed, chunk coord); workers and multiplayer clients must reproduce
  identical chunk bytes. Therefore the generator NEVER registers blocks at
  runtime. The full reachable set is enumerated from the placement table and
  registered in canonical order at startup, before any worker exists. The
  generator keeps today's shape: it holds pre-resolved ids (now LUTs) and no
  registry access.
- **C3 — Structure, not soup.** Independent per-element noise would produce
  statistical texture, not a world. Elements do not get independent worlds;
  they get independent *opinions about a shared world*. The authored part —
  surface heightfield, cave field, island field, depth/altitude — stays
  exactly as is, as **context fields**. A preference is a cheap response
  function over that context.
- **C4 — The hot path and the uniformity proofs survive.** No new noise
  fields. Per-cell cost stays ~today's (context case lookup + one or two
  `cell_hash` rolls). The `Uniform(stone)` lattice-corner proof and the
  `Uniform(air)` sky proof carry over with only their depth constants
  re-derived from the table.

## Design

### 1. Context fields (unchanged machinery)

The shared world state a preference may consult:

| field       | source                                   | notes |
|-------------|------------------------------------------|-------|
| `height`    | `SineHills::height(wx, wz)`              | unchanged |
| `depth`     | `height - wy` (ground only)              | unchanged |
| `altitude`  | `wy`                                     | unchanged |
| `carved`    | cave field vs `cave_threshold(depth)`    | unchanged, incl. `CAVE_MIN_DEPTH` crust guard |
| `island`    | island field vs `island_threshold(y)`    | unchanged, incl. the 4-above stack for surface typing |
| `lowland`   | `height <= sand_height`                  | unchanged |
| `icy`       | `wy >= ICE_SURFACE_Y` (island surface)   | unchanged |

Carved cells and non-island sky skip placement entirely — "no element wants
a carved cell" is implemented as one context check, not nineteen per-element
checks. Same result as the pure union-of-preferences reading, at today's
cost.

### 2. The placement table

One module, `src/world/placement.rs`. An element may have **multiple rules**
(e.g. Quartz places in ground stone *and* island stone; Soil needs separate
rules for depth 1 vs depths 2–3 to exclude lowland surfaces).

The vocabulary is deliberately **closed** — no closures, no arbitrary
predicates. That is what makes the reachable set mechanically enumerable
(C2), and it keeps every rule serializable, which the "mod manifest,
eventually" vision note wants for the future wasm boundary.

```rust
/// Where one element wants to exist. An element may own several rules.
struct PlacementRule {
    element: ElementId,
    region: Region,            // Ground | Island
    kind: Kind,                // see below
    /// Surface-relative eligibility, ground region (depth = height - wy).
    depth: RangeInclusive<i32>,
    /// Absolute-Y eligibility (islands care; ground rules usually pass Full).
    altitude: RangeInclusive<i32>,
    surface: SurfaceFilter,    // Any | OnlyLowland | NotLowland | OnlyIcy | NotIcy
}

enum Kind {
    /// Present wherever the rule's predicate holds. The union of banded
    /// elements is the base material — layering emerges from band overlap.
    Banded,
    /// Present where the predicate holds AND a per-cell hash roll hits.
    /// `stream` picks one of two decorrelated hash streams (see rolls).
    /// Only places into cells whose banded union is exactly `host` —
    /// structurally bounds combinations to {host} ∪ payloads.
    Scattered { rarity: u32, stream: Stream, host: ElementId },
}
```

Today's terrain, expressed as the builtin table (this is the parity target
for Phase 1):

| element | kind | region | rule | emergent block |
|---|---|---|---|---|
| Organic | banded | ground | depth 1, NotLowland | with Soil → "grass" |
| Organic | banded | island | surface cell, NotIcy | island grass |
| Soil | banded | ground | depth 1, NotLowland | |
| Soil | banded | ground | depth 2..=3 | with Clay → "dirt" |
| Soil | banded | island | surface / within 3 below | island crust |
| Clay | banded | ground | depth 2..=3 | |
| Clay | banded | island | within 3 below surface | |
| Sand | banded | ground | depth 1, OnlyLowland | pure → beaches |
| Ice | banded | island | surface cell, OnlyIcy | pure → island frosting |
| Stone | banded | ground | depth ≥ 4 | pure → the world's rock |
| Stone | banded | island | interior (≥ 4 below surface) | |
| Coal | scattered | ground | depth 4..=64, 1/90, host Stone | Stone+Coal |
| Iron | scattered | ground | depth 8..=64, 1/110, host Stone | Stone+Iron |
| Copper | scattered | ground | depth 8..=64, 1/130, host Stone | |
| Sulfur | scattered | ground | depth 20..=64, 1/240, host Stone | |
| Quartz | scattered | ground | depth 20..=64, 1/200, host Stone | |
| Lead | scattered | ground | depth 20..=64, 1/220, host Stone | |
| Gold | scattered | ground | depth 32..=64, 1/300, host Stone | |
| Lumin | scattered | ground | depth 32..=64, 1/380, host Stone | |
| Titan | scattered | ground | depth 48..=64, 1/460, host Stone | |
| Obsidian | scattered | ground | depth 48..=64, 1/240, host Stone | Stone+Obsidian (semantics change — see sidenotes) |
| Aerium | scattered | island | interior, 1/45, host Stone | Stone+Aerium |
| Quartz | scattered | island | interior, 1/160, host Stone | Stone+Quartz |

(Exact numbers copied from today's `seams` array and island constants;
`ORE_MIN_DEPTH = 3` is listed as depth 4 because today's roll only ever runs
in the stone branch, i.e. below the dirt at depths 2..=3 — the table makes
that explicit instead of implicit in branch order.)

### 3. The union rule and composition semantics

`Composition::Natural` with equal parts, elements sorted by id — matching
`CompKey`'s canonicalisation so identical sets always dedup to one id.
Today's veins are already equal-parts naturals (CoalVein =
natural[Stone, Coal]), so ore semantics don't move. The crust changes
composition slightly: "grass" becomes Natural(Organic+Soil) = 50/50 where
the builtin Grass block is Mixture(Organic 65/Soil 35). Accepted — see
sidenotes for why we don't chase exact parity.

### 4. Scattered rolls — one hash, two streams, pairs capped at two

- **Stream A** is byte-for-byte today's scheme: one `cell_hash` per stone
  cell, mapped through cumulative rarity slices, tiers gated by depth,
  yielding 0 or 1 payload. Keeping it identical preserves today's single-ore
  distribution exactly (regression-testable).
- **Stream B** is a second, decorrelated hash stream (new salt, same
  splitmix64 shape) with the same slice layout but rarities scaled down
  (÷8 as the starting tune). It also yields 0 or 1 payload.
- Cell payloads = dedup(A, B): 0, 1, or 2 extra elements. **Overlap arity is
  structurally ≤ 2** — that is constraint C1's load-bearing bolt. Pair
  frequency at ÷8 scaling: P(pair) ≈ p·(p/8) ≈ 1.4e-6 per stone cell for the
  common ores — a handful per hundred chunks of rock, a genuine find; single
  rates move by ~+12% (÷8 bonus stream), which the rarity table can absorb.

### 5. Startup enumeration and canonical registration (C2)

At world init, after builtins and after mods have registered elements and
placement rules, **before any worker thread exists**:

1. Collect all rules. Build the finite context-case space per region: the
   sorted distinct boundary values of every rule's depth range, altitude
   range, and surface filter cut the context into cases; within a case,
   every rule's predicate is constant.
2. For each case: banded union `B`. Empty → air, skip. Otherwise
   `Natural(B)` is reachable.
3. For each case: the eligible scattered set `S` (host matches `B`).
   Reachable: `Natural(B ∪ {s})` for every `s ∈ S`, and
   `Natural(B ∪ {s1, s2})` for every pair with intersecting eligibility.
4. Sort ALL reachable compositions canonically (by sorted element-id list)
   and register in that order. Sorting makes ids independent of table
   iteration order — a mod inserting a rule cannot reshuffle ids of
   unrelated combos within one build. (Cross-build id stability is NOT a
   goal; saves and the network never speak ids — see compatibility.)
5. `assert!(count <= ENUM_CAP)` with `ENUM_CAP = 192`, leaving ≥ 64 ids of
   headroom for crafting and mods at runtime. A mod that explodes the
   enumeration fails loudly at startup, not mysteriously at chunk 40,000.

Combos identical to existing builtins (e.g. {Stone, Coal} == CoalVein's
composition) dedup to the builtin's id automatically via `CompKey` — the
enumerator doesn't need to know builtins exist.

**Palette arithmetic** for the builtin table: banded combos {Stone},
{Organic, Soil}, {Soil, Clay}, {Sand}, {Ice} = 5 (mostly deduping into
existing builtins) · scattered singles on Stone = 11 · ground pairs
C(10,2) = 45 (all reachable — every ground tier's range reaches depth 64,
so all pairs intersect in 48..=64) · island pairs {Stone, Aerium, Quartz}
= 1. With Air and the existing builtins the registry lands at **≈ 75–80
ids** — under a third of the cap, no lift needed. This arithmetic goes in a
test, not just this doc.

### 6. The generator hot path (C4)

`SineHills` keeps its name (it still describes the heightfield), its
`height`, and every noise/column structure. Its block-picking interior is
replaced by tables precomputed from the enumeration:

- **Case LUT**: context case → base BlockId (the banded union's id). Cases
  are small integers (depth-band index × surface flags × region); the
  per-cell classification is a couple of integer compares, as today.
- **Payload LUTs**: per stream, today's `seams`-style cumulative slice array
  where each slice carries the *precomputed* `Natural(B ∪ {x})` id. For
  pairs, one global symmetric `n×n` matrix of `Natural({Stone, x, y})` ids
  (payload combos don't depend on depth, only eligibility does — the
  per-tier `break` gating stays exactly today's shape).
- Per stone cell: classify case → 1–2 `cell_hash` calls → at most two LUT
  reads. That is today's cost plus one hash — the bench (`WATT_BENCH`
  fps + rss_mb) confirms; budget: no measurable fps regression at RD10.

**Uniformity proofs carry over:**
- `Uniform(stone)` deep-chunk proof: today's condition "below every column's
  ore band" becomes "below every scattered rule's max depth", derived as
  `table.max_scattered_depth()` (= 64 for the builtin table — identical
  constant, now data). The cave lattice-corner bound is untouched.
- `Uniform(air)` sky proof: unchanged (above `h_max`, below `ISLAND_MIN_Y`).
- The dense-fill `collapse` remains the correctness backstop; proofs stay
  CPU shortcuts, never gates.

## The palette cap and memory (the "how bad would lifting it be" analysis)

Not needed for this task, but here is the real answer, measured against the
current bench baseline (origin 99–103 MB RSS, RD10 144 MB):

| cost center | today | at u16 ids / >256 blocks | verdict |
|---|---|---|---|
| Dense chunk cells | `u8`, 4 KiB/chunk | 8 KiB/chunk | RD10 holds roughly 2–6k dense chunks (surface band + caved deep) ≈ 8–24 MB → doubling adds ~10–25 MB (~10–15% RSS). Clone traffic for mesh-job snapshots doubles too. Noticeable, not bad. |
| Block textures | 1 KiB/id (16×16 RGBA8) | linear in ids | Nothing until thousands of ids; Vulkan `maxImageArrayLayers` (commonly 2048) is the first real wall. |
| Registry hot arrays | 5 B/id | linear | Nothing. |
| Saves / network | spec strings + u16 indices | already id-width-agnostic | Nothing — by-name portability was built in from the start. |
| **Mesher vertex** | layer = `Vertex::color.a` (u8) | needs a wider layer attribute | **The binding constraint.** A voxel-engine change: vertex format + shader + mesher (+2 B on a ~28 B vertex ≈ +7% mesh memory). Engine work, not RSS. |

So "drastically lifting the cap" is a moderate engine-plumbing task plus
~15% RSS, not a memory catastrophe. The *better* lift, if ever needed, is
**per-chunk palettes** (dense cells stay u8, indexing a small per-chunk
`Vec<BlockId>`): unbounded global palette, zero chunk-memory growth — only
the vertex layer width remains. Decision: defer; `ENUM_CAP = 192` is a
constant we can raise the day the engine work lands.

## Compatibility

- **Saves**: chunks are never saved — the seed regenerates terrain and edits
  replay by coordinate with portable spec strings. Old saves LOAD fine, but
  their edits now float over different terrain materials (same geometry —
  see testing — so no floating-in-air surprises, but a mined-out iron vein
  may sit in what is now a coal vein). This is a worldgen-version break in
  spirit. Cheap guard, recommended: bump save format to v4 = v3 + a
  `worldgen_version: u16` field; loader warns (not rejects) on mismatch.
- **Multiplayer**: palette is a pure function of (builtins, mod set,
  placement table) — same build ⇒ same ids ⇒ identical chunk bytes,
  exactly today's contract. Mixed-version peers already desync on any
  worldgen change; fold `worldgen_version` into the join handshake alongside
  the protocol version so the mismatch is an error message instead of
  silent divergence.
- **Old named blocks**: Grass/Dirt (mixtures) stay registered — crafting and
  saved edits may reference them; terrain just stops placing them. The
  `*Vein` builtin entries become redundant *names* for compositions the
  enumeration re-derives; keep them for now (free — they dedup), retire the
  macro rows in a later cleanup if the pretty names stop mattering (see
  sidenotes on naming).

## Testing plan (house style: prove, census, bit-compare)

1. **Geometry invariance (the big one).** This rewrite changes what solid
   cells are made of, never where solid cells are: heights, cave field,
   island field, thresholds are all untouched. Census a few hundred chunks
   across representative coords (origin, far, deep, high): old vs new
   generator must produce an IDENTICAL solid/air pattern. Any diff is a bug,
   full stop. (Run the census before deleting the old paths; keep old code
   in the test module if needed.)
2. **Material parity where semantics didn't move.** With stream B disabled,
   stream A reproduces today's ore distribution exactly: every cell that was
   CoalVein must be Natural{Stone, Coal}, etc. Grass cells must be
   Natural{Organic, Soil}, dirt Natural{Soil, Clay}, beaches {Sand}, island
   ice {Ice}.
3. **No runtime registration, by type.** The generator owns LUTs of ids and
   no `&mut BlockRegistry` — the compiler enforces C2. Plus a census
   asserting every id any generated chunk emits was registered by the
   enumerator.
4. **Enumeration soundness & bound.** Every combo the census observes is in
   the enumerated set (no leaks); enumerated count ≤ ENUM_CAP with the
   arithmetic from this doc asserted (≈ 75–80 for builtins).
5. **Canonical ordering.** Permute table row order → identical id
   assignment.
6. **Determinism.** Same seed ⇒ identical chunk bytes, near and far (the
   existing far-coordinate test coords: 2^28 boundaries, ±1e9 border).
7. **Proof soundness.** Existing lattice-corner and uniform-air tests carry
   over; new: `max_scattered_depth()` derivation tested against the table;
   deep chunks below it + cleared cave bound prove Uniform(stone) with the
   dense fill agreeing (the existing 11k-chunk soundness census pattern).
8. **Pair statistics.** With stream B on: pair frequency within expected
   band; single-ore rates within tolerance of the rebalanced targets;
   arity never exceeds 2 (census).
9. **Bench.** `WATT_BENCH` before/after: fps and rss_mb at origin and RD10.
   Budget: no regression beyond noise.

## Implementation phases

- **Phase 0 — vocabulary + enumeration.** `src/world/placement.rs`: rule
  types, builtin table, case-space construction, reachable-set enumeration,
  canonical registration, ENUM_CAP assert. Tests 4, 5. No generator change;
  the enumerated combos simply join the registry (mostly deduping into
  existing builtins).
- **Phase 1 — rewire the generator.** Replace `ground_block_carved` /
  `island_block` / `seams` with the case LUT + stream-A payload LUT built
  from the enumeration. Delete nothing yet outside `SineHills`'s interior.
  Tests 1, 2, 3, 6, 7. This phase must land material-identical for ores and
  geometry-identical everywhere.
- **Phase 2 — pairs.** Stream B, the pair id matrix, rarity rebalance
  (÷8 start), reaction audit print (see sidenotes). Tests 8, re-run 1/6/7.
- **Phase 3 — cleanup + guards.** Docs (this file → updated to "landed";
  module docs in generation.rs), bench run (test 9), optional save v4 +
  handshake `worldgen_version`, night-log entry.
- **Phase 4 — follow-ups (separate tasks, unblocked by this design).**
  Boundary dithering for transition bands (Sand+Soil beach edges: a banded
  rule whose edge dithers on a hash — costs exactly one palette slot per
  authored pair); Lumin clustering on deep cavern walls via a
  surface-adjacency rule (the vision-note idea — needs a `CaveWall` region,
  which the vocabulary was shaped to accept); mod hook: mods contribute
  rules pre-enumeration (the ordering already guarantees determinism).

## Sidenotes, traps, and thoughts

- **Obsidian changes meaning.** Today it scatter-places as PURE Obsidian
  (the one seam that isn't a Stone+X vein). Under the union rule it becomes
  Stone+Obsidian. That is *more* consistent (it was the odd one out) and
  slightly nerfs it (50% obsidian per block mined). If pure pockets matter
  for game feel, the vocabulary could later grow a `replaces_host` flag —
  deliberately NOT in v1; every flag added to the vocabulary multiplies the
  enumeration cases and it must earn its place.
- **Grass drift (65/35 → 50/50).** Natural equal-parts moves the crust's
  derived properties a little (more organic → slightly less durable, more
  vivid). Chasing exact parity would mean placement rules emitting weighted
  Mixtures — more vocabulary, and it breaks the beautiful "the set IS the
  block" invariant. Accept the drift; if the crust feels wrong, tune the
  *element* properties, not the placement semantics.
- **Reactions in terrain are a feature — audit them anyway.** Enumerated
  combos run through `ReactionRegistry::active_for` at registration, so an
  overlap vein can carry an Emergent(Reactive) tag (Sulfur+Coal…). That's
  the game's promise delivered by the terrain itself — mining that seam
  *should* be exciting. But we want to KNOW: the enumerator logs every
  reachable combo whose reaction set is non-empty, so a surprising world
  behaviour is a read of startup output, not a mystery.
- **Naming.** `auto_name` yields "Stone+Iron+Coal" — fine for v1 and for
  the inspection UI. The old pretty names (CoalVein) survive as aliases on
  the deduped ids as long as the macro rows stay. A curated display-name
  pass (e.g. "Iron-Coal Seam") is cosmetic, deferred.
- **Rarity semantics shifted one notch.** Today `ORE_MIN_DEPTH = 3` is
  documented but unreachable at depth 3 (that's dirt); the table writes
  depth 4 explicitly. Verify in test 2 that this is byte-identical in
  practice (it is — the roll never ran at depth 3).
- **Keep the vocabulary closed.** The single biggest architectural risk is
  someone adding a `Box<dyn Fn(Context) -> bool>` rule "just for one
  feature". That silently breaks enumeration (C1, C2 both) — the palette
  stops being provable and ids stop being derivable. The closed vocabulary
  is the contract; extend it by adding *analyzable* variants only.
- **Two hashes, not N fields.** Equally important performance contract: an
  element preference must never grow its own noise field. New spatial
  character (clustering, dithering) comes from new *derived context* (one
  shared field, added deliberately) or hash tricks on existing streams.
- **The `saves/` directory in the repo root** contains real player saves —
  geometry invariance (test 1) is what keeps them meaningful. If a future
  change DOES move geometry, that's when the v4 worldgen_version warning
  earns its keep.
- **Far-coordinate discipline.** Nothing in this design touches world→lattice
  reduction, so the f64 lessons (slab islands past 2^28, the ±1e9 border)
  are untouched — but test 6 keeps the far coords in the loop anyway because
  the LUT classification adds new integer paths (depth-band edges) that
  should see extreme inputs at least once.
- **Memory tripwire.** rss_mb sits in the bench output precisely for changes
  like Phase 2's extra LUTs (trivial) and any future cap lift (not trivial).
  Read it every phase.

## Decisions taken (so we stop re-litigating)

1. Union of placed elements IS the block; equal-parts Natural; air = empty
   set. (The user's rule, verbatim.)
2. Context fields stay authored and shared; preferences are response
   functions over them. No per-element noise fields.
3. Scattered overlap capped at 2 via the two-stream roll; pairs are wanted
   (multi-yield finds), triples are not (palette + legibility).
4. Palette cap stays 256 with ENUM_CAP = 192 headroom; per-chunk palettes
   are the sanctioned future lift; the engine vertex layer byte is the real
   wall, not RSS.
5. Startup enumeration in canonical order; generator can't register (owns no
   registry access); mods contribute rules before enumeration.
6. Geometry must not move in Phases 0–3: same heights, caves, islands.
   Material assignment is the only thing this task changes.
