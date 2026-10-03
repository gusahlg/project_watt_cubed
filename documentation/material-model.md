# Matter

A voxel holds a **configuration**: a multiset of **elements** (points of a 4-D periodic resource
lattice, identity = coordinates). There are no block types. Grass, rock, a mine lamp, a pickaxe and
a planet's core are all just configurations; one integer **law** — *selective transfer v1* — decides
what happens when two of them touch. Nothing in the simulation has a name. All decisions are integer
arithmetic over a committed table, so every peer computes the same bytes.

Code: `crates/material` (the pure law). Game intern table: `src/block/registry.rs`. Scheduler:
`src/sim/reactions.rs`. Specification: `guides/reaction-guide/selective-transfer-v1.md`.

## Lattice and configurations

- `Element([u8; 4])` — a point on the torus `(Z/256)^4`. `ring_distance` is the short way round.
- `Configuration` — a multiset of at most `CAPACITY` (32) occurrences. Empty is void (air).
- `Encoding` — canonical bytes (sorted): the intern key, the save spec (`c:<hex>` / `air`) and the
  wire form.
- `Block` — a configuration plus each occurrence's cached internal support
  `h[i] = Σ_{j≠i} fit_raw(e_i, e_j)`. The registry keeps one per interned id.

`BlockId(u16)` is the intern index in one world's table (id 0 = air).

## The law: selective transfer v1

`fit_raw(a, b)` reads the committed 256-entry `FIT` table once per axis (byte difference) and sums:
positive favours grouping, negative favours separation, identical elements fit 0.

`react_once(A, B)` (or a `Contact` stepping one operation at a time) performs **one** operation on
a face contact:

- **transfer** one occurrence from A to B or B to A, when it raises the contact's summed internal fit
  by more than `N · QUANTUM / 8` (`N` = occurrences in the contact, i.e. a `1/32` normalized gain);
- **swap** one occurrence each way, only when the receiving block is at capacity;
- deterministic tie order (spec §8): the best gain wins, then the canonical candidate order.

Elements are conserved: matter is regrouped, never minted or destroyed. A contact at rest stays at
rest. Air is the empty configuration, so a mixture whose occurrences fit badly can shed into empty
space; a cohesive material never does.

The law is a value (`Law`): version (2 = selective transfer v1), the observation probes and the
presentation seed. Its stamp folds a digest of the fit table, so a world never silently mixes laws.

## Observations

Properties are readings of the law, never authored stats (`observe(law, block)`):

| Reading | Source |
|---|---|
| `hardness` | cohesion (mean internal pair fit), plus a little bulk |
| `transparency` | fit with the **light** probe element |
| `emission` (0..15) | fit with the **glow** probe element |
| `friction` | fit with the **grip** probe element |
| `acoustic` | hardness band |

Every non-empty configuration is solid. There are no liquids. Readings are cached once per interned
configuration in the registry hot tables (mesher, light, physics, audio read those).

## Presentation

`visual(law, block)` → `Visual`: the base colour is periodic 4-D value noise sampled at the
configuration's **centroid** on the torus (so a block that gains or loses constituents visibly
drifts toward them), the accent is the least-held occurrence's colour mixed into a darker base,
plus pattern frequency, roughness, alpha and glow. An appearance mod turns that (and the
configuration itself) into the texture layer — see `documentation/mods.md`.

## Reactions in the world

`src/sim/reactions.rs`. Reactions start **only** when contact or material state changes: a block is
placed, removed or moved, or its configuration changes (a tool use, a machine, another reaction).
The affected voxel's six face contacts are queued — empty neighbours included; for a move both the
old and the new location are. Diagonals never interact.

- The queue holds `(arrival turn, Contact { lower cell, axis })`, deduplicated, in that order.
- Each simulation turn (20 Hz) works at most `Budget::DEFAULT` (384) contacts, one law operation
  each. If both cells changed, every contact of both is queued for the **next** turn, so a cascade
  advances one hop per turn instead of completing at once.
- A contact that produces no change goes **dormant**: it leaves the queue until something wakes it.
- Chunk loading, generation, meshing and rendering never queue anything. Cells are read through the
  edit overlay and the generator, so results never depend on streaming.
- Multiplayer: the server owns the scheduler and broadcasts committed cells
  (`ServerMessage::Snapshot` / `Edit`); clients never evaluate reactions.
- The pending queue (with arrival ages) is saved with the world and restored on load.

Console: `/reactions` (`active`, `turns`, `operations`).

### Tools

A tool is a block in hand. Left-click with a held material runs the law between the held unit and
the targeted cell (`BlockRegistry::react`, stepped to rest): occurrences move between them, the cell
and the held unit both change configuration, and the cell's contacts wake. A held counter-material
can empty a resistant block one constituent at a time. On a server the client sends
`ClientMessage::ToolUse` and the server answers `ServerMessage::ToolResult`. The bare hand breaks
blocks; breaking yields that configuration into the stash.

## Worldgen materials

The generator (`src/world/terrain`) never names an element. `terrain::palette` *searches the law* for
a configuration per role (grass, banded rock strata, timber, mine lamp, a planet's glowing core, …):
nearest colour to the role's target, the needed glow or clarity, **cohesive** (every internal pair
fits non-negatively, so no occurrence wants to leave for empty space) and **mutually dormant** with
every other common material in both orientations, so the generated world is at rest until a player
disturbs it. **Reagents** break that rule on purpose: counter-materials that empty one specific
rock while staying dormant against the rest — the natural tools found as veins in the mines.

## Saves and protocol

Save **v9** (`src/save/format.rs`): spec table of encodings, the law stamp, the edit overlay, the
reaction queue. Protocol **v11** (`PROTOCOL_VERSION`): `Welcome` carries the law stamp; the content
fingerprint folds `WORLDGEN_VERSION`, the law fingerprint and the generator's palette. Mixed laws do
not join.

## Changing the law

Build a new `Law` (or a new fit table), bump `Law::version`, re-run the palette search and the
material tests (`cargo test -p material`, `terrain::palette`), bump save/protocol as needed. Game
systems read observations and visuals only — they must not grow named special cases.
