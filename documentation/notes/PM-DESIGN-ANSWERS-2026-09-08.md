# Design answers — positions on the open questions and missing features

Companion to `PM-VISION-REPORT-2026-09-08.md`. Each item names the question as
the docs pose it, takes a position, and says what the smallest coherent
implementation slice is. "A" items are architecture the code needs; "D" items
are game-design decisions; "M" items are missing features. Where I disagree
with a draft I say so; these are proposals for the maintainer to accept,
change, or reject, not decisions.

## A. Architecture

### A1. The inventory belongs to the core; mods present it
The stash is created inside `Mods::with_defaults` and shared by `Rc<RefCell>`
between the inventory and crafting mods. features.md says the inventory is
"merely a list" that exists without any interface mod. Move `ElementStash`
onto the core game state (the `Player`, or a `Economy`/`Holdings` struct next
to it), pass it to mods through `ModContext` as `&mut`, and keep both mods as
pure views/intents over it. Invariant test from the structural audit: disable
the inventory UI, mine, re-enable, resources remain. Save format: the stash
already serialises through the inventory mod's `save_state`; moving it to the
core save document is a one-time migration with the same portable element
names. Size: ~200 lines. Do it before any server-side economy, because the
server will need to validate "does this player hold X" against core state,
not against a client mod.

### A2. Server-side hook twins before more client mods
The Mod trait is client-only. The relay loop already has the event points
(`on_edit`, join, chat). Add a `ServerMod` trait with `validate_edit`,
`on_join`, `on_chat` and register the same default mods' server halves
(protection, economy later). Keep every hook signature serialisable (plain
data in, plain data out) so the eventual manifest/wasm boundary stays possible
— the vision note already asks for this and menus-as-mods honoured it.

### A3. A deterministic simulation substrate before the first real system
Today the seam is not only empty but unreachable: `Simulation::add` has no callers
and `Game::new` boxes the driver into the scheduler immediately, so no mod
could register a system even if it wanted to. Fix the reachability and the
ordering rule first (systems sorted by a stable name, never install order — a
small task already dispatched), then reserve `sim/cells.rs`: a slot-based table of *active cells* (coordinate →
packed state) fed by edit and placement events, iterated by systems instead
of walking chunks. Server-owned phases: immutable snapshot → intents →
stable conflict reduction → commit, with hash-map iteration order banned from
semantics. Every future system (Watt, thermal, reactions, corrosion) is
written against this table, and multiplayer machines come "nearly free"
because pulses are just edits on the existing channel. Design doc first,
~1 week of code when the first system needs it.

### A4. The Mod surface: change these before more mods exist
- `ModContext` exposes `&mut World` and `&mut Player`. Narrow it to a
  capability struct (read block, queue placement, read/adjust holdings,
  raycast) so a mod cannot reach the hot path and so the surface can be
  serialised. This is a mechanical change while there are three mods.
- `HudElement` is already data; finish HUD-as-mods by making crosshair,
  coordinates, FPS and help line a `HudModel` the default mod renders (the
  vision note's "streamer HUD is a mod swap").
- Replace `Rc<RefCell<ItemUiState>>` sharing between inventory and crafting
  with an explicit overlay-ownership rule in the core (one open overlay at a
  time, owned by the core, mods ask to open/close).
- Give mods a content-registration hook (elements, reactions, placement rules)
  that runs before the placement table compiles, and fold the installed mod
  set into the content fingerprint so mixed clients are refused at join
  instead of desyncing. The registries already have the `register` entry
  points; only the hook and the fingerprint fold are missing.
- Stable mod ids separate from display names, versioned state blobs, and
  persisted enable/disable choices (the code claims the last already; it does
  not) — all three are small and dispatched.

### A5. Crate boundaries: follow the note, but only when a second consumer appears
`audio-and-crate-boundaries.md` is right about direction (domain → content →
protocol/save/audio → server → client) and right that nothing should move
mechanically yet. The two dependencies it names (world constructing the
acoustic window; audio importing the block sound class) are worth breaking now
because they are cheap and they keep the door open. A `watt-server` that
builds without Vulkan/Kira is the first crate worth extracting, and only once
a headless CI runs it.

### A6. Release engineering
R-01 (publishable engine URL) and R-11 (clean-checkout CI running both suites
and the pure package) are the only items on the release roadmap that protect
against the class of failure that has already happened three times
(engine/game drift). Do them before the next content push; nothing else on
that list is urgent.

## D. Design decisions

### D1. Call it Watt. Two verbs, one noun.
Agree with the vision note: instantaneous signal = watt *pulse*, sustained
power = watt *flow*, both measured in the same unit. The name is in the title,
it is a real unit so intuition never fights it, and it gives the community a
noun. In code, `conductivity` stays the element property; `watt` names the
network state.

### D2. Block kinds: two, not four
Natural (canonical element set, equal parts, derived properties, cheap) and
Mixture (author-chosen ratios, unlocks special and reaction properties).
Drop Configuration blocks as a *kind*: a 30³ sub-grid per block is a memory
and UI abstraction the drafts themselves distrust. Keep the *goal* — directed
conductivity, layered materials — and reach it with adjacency and orientation
instead: a block may carry a face mask (which faces conduct) as a small
per-block state, which gives "insulated on five sides" without a sub-grid.
Computational blocks become a *mod-provided* block family on top of Watt
(logic gates as tiny mixtures with a gate special property), not a fourth core
kind. This keeps the core to one representation: composition + a few bytes of
per-block state.

### D3. Crafting: three tiers, no magic box
The drafts reject "put things in a box and a product pops out" and want
something fundamental and automatable. Proposal:
1. **Hand crafting of naturals** stays as it is (any distinct element set →
   natural block) — it is the tier-0 loop and it already works.
2. **Mixtures are made by *state*, not by machine.** A mixture forms when
   element-bearing inputs are *co-located under a condition*: elements placed
   into a cell in sequence (the block-break event makes elements harvestable;
   placing an element into an existing block adds it to the composition) and
   a condition that "sets" the block — the first condition is *heat* (a Watt
   heater or lava adjacency). That is smelting without a furnace UI: the same
   rule works by hand and by machine, and it is the docs' "states, not
   machines".
3. **Alloys** are mixtures whose reaction set is non-empty and whose set
   condition was met; the reaction's environment condition (temperature
   threshold, pressure later) is data on the reaction row, which the reaction
   registry already validates.
The player-facing UI for tier 2 is a mod (the existing crafting panel grows a
"pour into block" action). Automation is Watt pulses toggling the condition.
What this deliberately does not do: sub-block geometry, blueprints, stardust.
Those ideas can return as mods once the base loop is fun.

### D4. The block-break event: health, broken state, harvest
block-break-event.md stops mid-sentence; finish it like this. Blocks have
health = derived durability. Damage below hardness does nothing (the docs'
floor rule). At zero health the block enters a *broken* state (same
composition, a "broken" flag bit in per-block state, darker texture, no
collision support for things above it later). A broken block can be
harvested: right-click yields its elements into the stash, leaving air; or it
decays to air after a timeout when automation is the intended harvester. This
is the one event every crafting tier above depends on (elements must be
collectable), and it is what makes `durability`/`hardness` observable.
Smallest slice: health accumulates on the client for the targeted block only
(no per-block storage until broken), the broken flag is a per-block state bit
synced as an edit.

### D5. Temperature: local and lazy, only when smelting exists
Agree with both drafts and the vision note: no global temperature field.
Temperature is a *query* — "how hot is this cell" computed from nearby heat
sources when a condition asks — so it costs nothing until a smelt happens.
Add it with D3 tier 2, not before. The survival-cold idea stays out.

### D6. Density and friction: keep, but make them earn it through assemblies
The decision is already yes. The observable consequence should be moving
assemblies (below) and nothing else until then; do not expose them in the
inspection UI as if they mattered.

### D7. Trim the property table to what is read
Keep: durability, hardness (D4), conductivity (Watt), density, friction
(assemblies), light emission, transparency-as-opacity, buoyancy (already
drives liquids). Drop from the *core* list: thermal conductivity and
temperature resistance (D5 makes them unnecessary until a heat system
exists; keep them as *special* properties on the few elements where they
matter). The vision note's trick — move rarely-used core properties to the
special-property mechanism — halves the table without losing an element.

### D8. Obsidian and grass drift
Accept both, as the plan document already did. Consistency of the union rule
is worth more than a pure-obsidian pocket. If the crust feels wrong, tune
element colours, not placement semantics.

### D9. Planets and gravity
Defer. Keep the mirrored progression (islands grow with altitude, caves with
depth). If planets ever land, they are a *different generator* selected by
region, not a change to the lattice, so nothing done now blocks them.

## M. Missing features, ranked by how much depends on them

| # | Feature | Depends on | Unlocks | Smallest first slice |
|---|---|---|---|---|
| M1 | Block-break event (D4) | nothing | all crafting tiers, harvesting, tools | client-side health on the targeted block; broken flag as an edit |
| M2 | Stash in core (A1) | nothing | server economy, save simplification | move the struct, keep mods as views |
| M3 | Watt (D1) on the sim substrate (A3) | A3 | machines, computation, automation | conductive nets via union-find, one source block, one lamp block, pulses as edits |
| M4 | Mixture crafting by state (D3) | M1, D5-lite | alloys, smelting, most content | "pour element into block" + heat condition |
| M5 | Moving assemblies | density/friction, sim | vehicles, pistons-done-right | flood-fill membership, one rigid translation per tick, brutal collision |
| M6 | Server-side hooks (A2) | nothing | protection, economies, community servers | validate_edit + on_join + on_chat with the default mods' server halves |
| M7 | HUD as mods (A4) | nothing | accessibility/streamer HUDs | HudModel + default mod render |
| M8 | Region files / checksummed saves | nothing | mega-builds, sharding | framed regions with checksums, compaction of baseline cells |
| M9 | Certificate pinning / TOFU | nothing | trusting public servers | trust-on-first-use fingerprint file |
| M10 | Tools and mining tiers | M1 | progression | tool = held natural block whose hardness sets the damage floor |

## Flaws the code exposes in the docs

- features.md still describes four block kinds and ten core properties; the
  notes have since argued them down. The drafts should be rewritten as the
  "final feature file" the refinement draft promises, with D2/D7 as the
  content, so newcomers do not read stale promises.
- The docs never state the *edit* as the universal unit of change, yet the
  whole multiplayer, save and (future) sim design is "everything is an edit".
  Writing that sentence down would settle several open questions (signals
  are edits; broken state is an edit; assemblies are batched edits).
- crafting.md and block-break-event.md are stubs; D3/D4 above are proposed
  content for them.
- The philosophy's "optimization ahead of readability" is being applied to
  comments in reverse: the hot files carry the longest prose. Conservative
  comments are part of the same philosophy.
