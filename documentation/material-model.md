# Matter

A voxel holds a **configuration**: an ordered list of **elements** (points of a 4-D resource lattice, identity = coordinates). One integer **law** turns neighbouring configurations into new ones. Nothing in the simulation has a name. All math is 32-bit integer so every peer computes the same bytes.

Code: `crates/material` (pure kernel). Game intern table: `src/block/registry.rs`. Lab: `crates/material-lab`.

## Lattice and configurations

- `Element([u8; 4])` — a point. `D = 4` is part of the law stamp.
- `Configuration` — ordered, multiplicity kept, length ≤ 16. Empty is void (air).
- `Encoding` — canonical bytes (`len` + `len×D` coordinates): intern key, save spec (`c:<hex>` / `air`), wire form.

`BlockId(u16)` is the intern index in one world's table (id 0 = air). Many configurations may share one look.

## The law as a value

`Law` is the physics: version, boundary, kernel, event strengths, probes, visual seed, quantum. A world records its stamp; peers with different stamps refuse to share a world.

**v0 does:**

- Boundary `Clamp` (coordinates saturate at 0 and 255). `Wrap` exists and is not chosen.
- Per-axis odd piecewise-linear response `g(|δ|)` through 9 integer knots: short-range repulsive, mid-range attractive, fade to 0. Then `Δ = M·r` with a Q4 mixing matrix, scaled by event strength, bounded by `max_step` (6).
- Pairwise aggregate: each target element gets the mean influence of the origin's elements, then clamp-step. Origin unchanged (`origin: None`).
- `quantum = 1` (no extra rounding unless proliferation demands it).
- Events: `{Moved, NewContact, Collision, ExternallyChanged}` only.

**Provisional (parameters, not architecture):** boundary, mixing, origin mutation, destruction, event whitelist, cascade budget, ordered configurations with multiplicity, pairwise influence with a bounded aggregate.

`element_influence` / `interact` live in `crates/material/src/kernel.rs`. Open a new universe by constructing a different `Law` and re-running the lab — do not special-case materials in game code.

## Probes and observations

`Probes` are fixed reference elements (constants of the universe): contact, light, flow, glow, friction. `observe(law, C)` is the bounded response of `C` to each probe, mapped through law thresholds:

- `solid` = non-empty and not liquid; `liquid` = flow ≥ `liquid_min`
- `transparency` = light response; `emission` = glow above `glow_min`
- `hardness` = 255 − contact; `friction`; `acoustic` from hardness/liquid

Void observes as air (passable, transparent, silent). Cached once per distinct configuration in the registry hot tables — never per voxel. Consumers: mesher, light, physics, audio (`SoundClass` is derived from the observation).

## Presentation

`visual(law, C)` → `Visual` (two RGB colours, frequency, roughness, alpha, glow): integer value-noise over the lattice, no axis means a colour. Quantized `DescriptorKey` is interned; the texture-array layer **is** the descriptor id (≤ 16 384). Many configurations share one descriptor, so the 14-bit vertex field does not cap material count.

Core fallback: `src/block/appearance.rs` (`FlatAppearance`). A texture mod paints the 16×16 layer from `Visual`.

## Scheduler

`src/sim/reactions.rs`. Gameplay events → neighbour pairs → `interact` on a snapshot → mutations commit in position order → follow-up events for the next generation. The queue is keyed by `(arrival generation, position, kind)`: a generation evaluates its budget of the **oldest** events first (ties in position order), and follow-ups are deferred, never dropped; a generation defines simultaneity (targets see the mean of every origin acting in it), so the batch size is part of the dynamics and every peer runs `Budget::DEFAULT`; a scheduler refuses new events only at its capacity (`DEFAULT_CAPACITY`, the `dropped` gauge). Cells are read through the overlay and the generator when no chunk is loaded, so results never depend on streaming state. A store that refuses a write commits nothing for that cell. Runs on the sim tick. Multiplayer: **server only**; clients receive snapshot cells (`Incoming::Mutation`, applied without place/break cues). Chunk load, gen, mesh, and save never emit.

| Gameplay | Event |
|---|---|
| Place | `NewContact` vs 6 neighbours |
| Break | `ExternallyChanged` for 6 neighbours |
| Moving block (none yet) | `Moved` |
| Machine mod | `ModContext::emit_material_event` |

## Regions (worldgen)

`src/block/regions.rs` + `src/world/placement.rs`. Starting families: centre element + variants + strata + a **label used only by placement rules and the `inspect` console command** — never shown to a player. Placement rules name regions, not materials. At world start each family is interned (deterministic from the law); columns pick a member. Generator output is stable under self-contact. `WORLDGEN_VERSION` (currently 5) folds into the content fingerprint.

## Holdings and the workbench

Stash entries are `(BlockId, count)` (`src/stash.rs`). Breaking yields that configuration. Inventory shows the visual swatch and words read off the observation (`BlockRegistry::display_name` → `describe`: phase, hardness band, clarity, glow, grip — e.g. "glowing clear hard solid"); materials have no authored names, and the names players give their own products are journal knowledge. Crafting (`src/mods/crafting.rs`) applies an event between two held configurations through `interact` (repeat 1..16); discovered procedures are journal knowledge. On a server the client sends `Craft` and never trusts its own result. The workbench acts only between two HELD units (the target is consumed, the origin must be present; slots naming a spent row are cleared). The server evaluates a `Craft` only from a ready player, at most `CRAFT_RATE_LIMIT` per second per connection, and resolves the specs without growing its table: a configuration it knows resolves by lookup, a novel one is interned only while `CLIENT_INTERN_RESERVE` ids stay free for the world's own products (`resolve_client_spec`, also the `Edit` path) — no client can exhaust the material table. A rejected placement whose refund no longer fits the pouch is counted and shown, never silently destroyed.

## Saves and protocol

Save **v8** (`src/save/format.rs`): spec table = encodings; `law_stamp` = the law. Reaction mutations are ordinary overlay edits attributed to the scheduler.

Protocol **v10**: `Welcome` carries the law stamp; fingerprint folds `WORLDGEN_VERSION`, `Law::fingerprint()`, and builtin region centres. `ConfigDefinition { id, encoding }` then `CellMutation { pos, id }`. Mixed laws do not join.

## The lab

`crates/material-lab` (`lab scorecard|find-regions|sweep`). Scorecard: similarity (nearby elements → nearby influence), determinism, fixed-point fraction, cascade size, proliferation, family count, observation census. Rejects collapse, explosion, freeze, unbounded cascade. `find_regions` searches centres that observe as intended (water-like, stone-like, …).

## Changing the law

Build a new `Law`, stamp it, run the scorecard until it PASSes, re-pick regions, bump save/protocol as needed. Game systems read observations and descriptors only — they must not grow named special cases.
