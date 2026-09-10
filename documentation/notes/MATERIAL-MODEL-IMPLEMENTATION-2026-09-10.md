# Emergent Material Model — implementation design (2026-09-10)

Source: the user's "Project Watt Cubed — Emergent Material Model" document (2026-09-10; §numbers below
refer to it). This file says HOW it lands in the code today: crate layout, the
provisional law v0, the probes, the scheduler, interning/render decoupling, the migration order and what
stays deliberately open (encoded as law parameters, never as architecture).

## 0. Ground rules for every step
- The game stays playable after every task; every task ends with the full suite green.
- Determinism: all runtime material math is 32-bit integer (wrapping ops, tables, no floats); the law is
  a value (`Law`) with a version stamp; saves and the protocol carry the law stamp; peers with different
  laws refuse to share a world.
- No names in the simulation. Labels exist only in `regions.rs` (worldgen starting regions), debug HUD
  and the (future) semantic layer.
- Open questions from the spec §2 are law PARAMETERS with a provisional value, documented as such:
  boundary policy (Clamp), cross-dimensional mixing (a 4×4 matrix), target-only mutation (origin change
  returned but unused), no destruction rule, event whitelist {Moved, NewContact, Collision,
  ExternallyChanged}, cascades across generations with a budget, ordered configurations with
  multiplicity, pairwise influence with a bounded aggregate.

## 1. Crate `crates/material` (pure; the game and the lab depend on it)
```
pub const D: usize = 4;                       // lattice dimensions (law stamp records it)
pub struct Element(pub [u8; D]);              // a point of the resource lattice; identity = coordinates
pub struct Configuration(Box<[Element]>);     // ordered, multiplicity kept, len ≤ CONFIG_MAX (16); empty = void
pub struct Encoding(Vec<u8>);                 // canonical bytes: len u8 + len×D bytes (the intern key, the save/wire form)
pub struct Law { version: u16, boundary: Boundary, kernel: Kernel, events: EventStrengths, probes: Probes, visual: VisualSeed, quantum: u8 }
pub enum EventKind { Moved, NewContact, Collision, ExternallyChanged }
pub struct Delta(pub [i16; D]);
pub fn element_influence(law: &Law, a: Element, b: Element) -> Delta    // F(δ): the whole physics
pub fn interact(law: &Law, origin: &Configuration, target: &Configuration, event: EventKind) -> ReactionResult // pairwise influence, bounded aggregate
pub struct ReactionResult { target: Configuration, origin: Option<Configuration>, changed: bool, magnitude: u32 }
pub struct Observation { solid, liquid, transparency: u8, emission: u8, hardness: u8, friction: u8, acoustic: u8 } // probes
pub fn observe(law: &Law, c: &Configuration) -> Observation
pub struct Visual { rgb: [u8;3], rgb2: [u8;3], frequency: u8, roughness: u8, alpha: u8, glow: u8 }
pub fn visual(law: &Law, c: &Configuration) -> Visual                    // V(C): smooth in C, no axis semantics
pub fn distance(a: Element, b: Element) -> u32; pub fn config_distance(a: &Configuration, b: &Configuration) -> u32
```
### 1.1 Kernel v0 (`element_influence`)
δ_i = a_i − b_i (i32, −255..255 under Clamp). Per-dimension response r_i = g(δ_i) where g is an ODD
piecewise-linear curve over |δ| given by 9 integer knots `(dist u8, response i16)` in the law (v0 shape:
repulsive at short range (< 12), attractive at middle range (12..96), fading to 0 by 200 — a
Lennard-Jones-like profile so elements form families without collapsing onto each other). Then
Δ = M·r with `M: [[i8; D]; D]` in Q4 (identity plus small couplings: v0 M = I + 0.25·cyclic shift),
scaled by the event strength (`EventStrengths: [u8; 4]`, Q8), and bounded by `max_step` (v0 = 6 units
per event). Smoothness: |δ − δ'| ≤ 1 ⇒ |g(δ) − g(δ')| ≤ max knot slope (the lab measures the
actual bound). Tables are integers in the law → identical on every machine.
### 1.2 Aggregate (`interact`)
For each target element b_j: Δ_j = clamp_step( (Σ_i F(a_i, b_j)) / |A| ); b'_j = clamp(b_j + Δ_j)
(Clamp boundary; `Wrap` exists as a variant and is NOT chosen). `quantum` (v0 = 1) would round
coordinates to a grid at commit time if proliferation ever demands it (measured by the lab, not assumed).
Origin: unchanged in v0 (`origin: None`).
### 1.3 Probes (`observe`) — the second small universal law, built from the first
`Probes` = fixed reference elements (constants of the universe): contact, light, flow, glow, friction.
Each observation = the bounded response magnitude of C to that probe as origin under a fixed strength,
mapped through a law-defined threshold table. `solid` = C non-empty ∧ ¬liquid; `liquid` = flow response
above `liquid_min`; `transparency` = light response; `emission` = glow response above `glow_min`;
`hardness` = 255 − contact response; `friction` = friction response; `acoustic` from hardness/liquid.
The void configuration observes as air (passable, transparent, silent). These are cached per
configuration in the registry hot tables (once per distinct configuration, never per voxel).
### 1.4 Presentation (`visual`)
Integer value-noise over the 4-D lattice with the law's `VisualSeed` (cell 32): rgb = noise₃(mean
element), rgb2 = noise₃(mean + spread direction), frequency = spread of the elements (0 for single),
roughness = contact response, alpha = transparency, glow = emission. Locality by construction; no axis
means a colour. The texture MOD turns a `Visual` into the 16×16 layer with today's pattern generator.
### 1.5 Encoding and ids
`Encoding` is the canonical bytes; `ConfigurationId(u32)` is the intern index in a world's registry;
`RenderDescriptorId(u16)` is the interned QUANTIZED `Visual` (rgb 5-6-5, rgb2 5-6-5, frequency 3 bits,
roughness 3, alpha 4, glow 4) — many configurations share one; the texture-array layer index IS the
descriptor id (≤ 16 384), so the 14-bit vertex field stops limiting the number of materials.

## 2. Game-side shape
- `src/block/`: `registry.rs` becomes the dynamic intern table (`MaterialRegistry`: configurations,
  intern map, hot SoA tables from `observe`, `render_layer` from the descriptor table, `Arc` snapshots for
  workers), `regions.rs` (worldgen starting regions: centre element + spread + size + label, verified by
  tests to observe as intended: "water-like" is liquid+transparent, "stone-like" solid+opaque+stable),
  `appearance.rs` (the `BlockAppearance` seam: descriptor → layer; core fallback = flat colour),
  `mod.rs`. Deleted: `element.rs` (`elements!`), `composition.rs`, `derive.rs`, `reaction.rs`,
  `crafting.rs` (natural crafting), `texture.rs` (moves to `src/mods/textures/procedural.rs`).
- `BlockId(u16)` stays the voxel/palette id = `ConfigurationId` truncated space (65 535 per world);
  `BlockState { id, state }` stays.
- `src/sim/reactions.rs`: the scheduler (events → candidate pairs → `interact` on a snapshot →
  mutations committed in position order → follow-up events, budgeted per generation; runs at the sim
  tick; in multiplayer only the server runs it and broadcasts mutations).
- Events come ONLY from gameplay: place → `NewContact` for the placed block against its 6 neighbours;
  break → `ExternallyChanged` for the 6 neighbours; falling/moving blocks (none yet) → `Moved`; a machine
  mod → whatever it emits through `Mod::emit_material_event`. Chunk load/gen/mesh/save never emit.
- Stash/inventory: entries `(ConfigurationId, count)`; breaking yields the block's configuration (one
  unit); the inventory mod shows the visual swatch + an optional label (region label if the configuration
  is within a region's spread, else "unknown"); crafting mod = a workbench that applies an event
  between two held configurations through `interact` and records discovered procedures as knowledge.
- Worldgen: `placement.rs` rules resolve to REGIONS; at world start each region contributes a small
  family of sampled configurations (e.g. 6 variants around the centre, deterministic from the seed),
  registered once; columns pick a family member with the existing hash stream → bounded palette,
  geological families for free. Stability: a test asserts each region's members are fixed points under
  self-contact (`interact(c, c, NewContact)` changes nothing) — the generator only emits stable matter.
- Save v8: spec table entries = `Encoding` bytes; stamp = `Law` (version, boundary, kernel knots,
  mixing, strengths, probes, visual seed, quantum) so a world identifies its physics; reaction
  mutations are ordinary overlay edits (attributed to the scheduler, not a player).
- Protocol v10: `ConfigDefinition { id, encoding }` sent once per new configuration, then
  `CellMutation { pos, id }`; `Welcome` carries the law stamp; the content fingerprint folds the law.

## 3. Lab (`crates/material-lab`, a bin): the empirical search framework
Runs millions of random local interactions for a law: similarity invariant (‖a−a'‖≤1 ⇒ ‖F(a,b)−F(a',b)‖ ≤ K),
determinism, fixed-point fraction, cascade size distribution (generations until quiescence on a small
3-D grid), configuration proliferation (distinct configurations after N events), diversity, oscillators,
family count (clusters of stable configurations). Prints a scorecard; rejects universes that collapse,
explode, freeze or cascade unboundedly. Also `find_regions`: scan candidate centres for stability and
desired observations (used to pick the worldgen regions).

## 4. Migration order (each task green before the next)
M1 crate + lab (PM writes the crate; grok writes the lab harness from the crate's API) →
M2a registry/observe/visual + all property consumers (mesher, light, physics, audio, HUD) →
M2b texture generator becomes the `procedural_textures` mod over the appearance seam; render descriptors →
M2c worldgen placement on regions + stability test; pins re-derived once →
M2d stash/inventory/crafting on configurations →
M2e save v8 + protocol v10 + fingerprint + server-authoritative scheduler →
M2f scheduler + events + mod hook + sim system →
M2g docs rewrite → M3 testing/quality (reviews, bug hunts, stress/golden, performance).
