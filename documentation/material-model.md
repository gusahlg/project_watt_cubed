# Matter

A voxel holds a **configuration**: a multiset of **elements** (points of a 4-D periodic resource
lattice, identity = coordinates). Grass, rock, a lamp, a tool and a planet's core are all
configurations, with no named block types. The common material catalog, compatibility kernel,
observations and presentation remain in `crates/material` and `src/block/registry.rs`.

Material interaction models are optional native modules. The extracted
[Selective Transfer Physics](https://github.com/gusahlg/pwc-package-manager/blob/main/modules/physics-selective/README.md) supplies the original integer
contact law and queue. `src/sim/model.rs` defines its interface; `src/sim/reactions.rs` hosts the
selected model, authority and saved state. Production vanilla has no reaction provider and
performs no automatic reactions or held-material reaction. The original law's specification
is [selective transfer v1](../guides/reaction-guide/selective-transfer-v1.md).

## Lattice and configurations

- `Element([u8; 4])` — a point on the torus `(Z/256)^4`. `ring_distance` is the short way round.
- `Configuration` — a multiset of at most `CAPACITY` (32) occurrences. Empty is void (air).
- `Encoding` — canonical bytes (sorted): the intern key, the save spec (`c:<hex>` / `air`) and the
  wire form.
- `Block` — a configuration plus each occurrence's cached internal support
  `h[i] = Σ_{j≠i} fit_raw(e_i, e_j)`. The registry keeps one per interned id.

`BlockId(u16)` is the intern index in one world's table (id 0 = air).

## Common compatibility kernel and selective transfer v1

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

The catalog law is a value (`Law`): version (2 = selective transfer v1), observation probes and
presentation seed. Its stamp folds the fit table's digest. This common catalog identity is
separate from the selected simulation model's id and version; changing a simulation provider
does not replace the element encoding or intern table.

## Observations

Properties are readings of the common catalog law, never authored stats (`observe(law, block)`):

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
plus pattern frequency, roughness, alpha and glow. An appearance module turns that (and the
configuration itself) into the texture layer — see [modules](modules.md).

## Optional simulation in the world

`Module::reaction_model()` returns a `ModelSpec { id, version, create }`. The first active module
returning a provider wins. Its `Model: Send` instance owns its queue or per-cell fields,
budgeted `tick`, optional `tool`, wakeups, counters, and `save`/`load` payload. Core supplies
the `CellStore` adapter and mutation bookkeeping; pair writes require atomic admission.
Dynamic fields belong to voxel instances, not shared material IDs.

A provider can declare an enabled option and return `None` when off. Switching or disabling
it freezes its payload in the host; restoring the same id and version resumes that state.
Only one model runs at once. Alternative bounded laboratories are
[waves](https://github.com/gusahlg/pwc-package-manager/blob/main/modules/physics-wave/README.md), [packets](https://github.com/gusahlg/pwc-package-manager/blob/main/modules/physics-packets/README.md),
[reservoir chemistry](https://github.com/gusahlg/pwc-package-manager/blob/main/modules/physics-reservoir/README.md), and
[memory matter](https://github.com/gusahlg/pwc-package-manager/blob/main/modules/physics-rotor/README.md); see the
[research notes](notes/2026-10-13-simulation/research.md) for their limitations.

Eight more separately packaged laws explore fuel fronts, excitation/recovery, loop currents,
spin order, nonlinear modular maps, plastic bonds, cooperative chemistry and matter permutations.
The [continued investigation](notes/2026-10-13-simulation/continuation.md) links their canonical
package-manager sources and records the small demonstrations. None is a selected final model.

### Extracted selective-transfer scheduler

`../pwc-package-manager/modules/physics-selective/src/model.rs`. Reactions start **only** when contact or material state
changes: a block is placed, removed or moved, or its configuration changes (a tool use, a
machine, another reaction).
The affected voxel's six face contacts are queued — empty neighbours included; for a move both the
old and the new location are. Diagonals never interact.

- The queue holds `(arrival turn, Contact { lower cell, axis })`, deduplicated, in that order.
- Each simulation turn (20 Hz) works at most `Budget::DEFAULT` (384) contacts, one law operation
  each. If both cells changed, every contact of both is queued for the **next** turn, so a cascade
  advances one hop per turn instead of completing at once.
- A contact that produces no change goes **dormant**: it leaves the queue until something wakes it.
- Chunk loading, generation, meshing and rendering never queue anything. Cells are read through the
  edit overlay and the generator, so results never depend on streaming.
- Multiplayer: the server owns the selected model and broadcasts committed cells
  (`ServerMessage::Snapshot` / `Edit`); clients never evaluate reactions. Experimental per-cell
  fields and laboratory perturbations are not replicated yet.
- The pending queue (with arrival ages) is saved with the world and restored on load.

The host exposes model counters and samples through `World::simulation_stats` and
`World::simulation_sample`; command and HUD modules can present them.

### Tools

A tool is a block in hand. Left-click with a held material asks the selected model's `tool`
operation. Selective Transfer Physics runs **one** `BlockRegistry::react` operation between
the targeted cell and held unit: occurrences move, both configurations change, and the
cell's contacts wake. Repeated uses can empty a resistant block one constituent at a time.
No provider means no held-material reaction; another provider chooses its own tool behavior.
On a server the client sends
`ClientMessage::ToolUse` and the server answers `ServerMessage::ToolResult`. With no tool, a
primary action breaks the block; breaking yields that configuration into the inventory.

## Worldgen materials

The generator (`src/world/terrain`) never names an element. `terrain::palette` *searches the
catalog law* for a configuration per role (grass, rock strata, timber, lamp, glowing core, …):
nearest colour to the role's target, the needed glow or clarity, **cohesive** (every internal pair
fits non-negatively, so no occurrence wants to leave for empty space) and **mutually dormant** with
every other common material in both orientations, so the generated world is at rest until a player
disturbs it. **Reagents** break that rule on purpose: counter-materials that empty one specific
rock while staying dormant against the rest — the natural tools found as veins in the mines.
These dormancy guarantees refer to selective transfer; another simulation model can evolve
the same generated configurations differently.

## Saves and protocol

Save **v10** (`src/save/format.rs`) retains the catalog stamp, canonical specs, edits and legacy
pending-contact field. Model state uses the existing opaque module-state section:
`@simulation.selected` records `id@version` or `off`, and `@simulation.state.<id>` records a
versioned payload. Paused providers' payloads remain present. This extraction does not bump
the binary save format; the selective provider can also import legacy pending contacts.

Protocol **v18** adds a simulation fingerprint to `Hello`'s content identity alongside
worldgen, gravity, catalog law and palette. It hashes the selected model id and version;
zero means no model. Mismatches refuse a join. A server restoring a world requires the saved
selection to match its configured compiled provider. Field-state replication is separate
future work; matching model identity does not make the experimental laboratory HUD networked.

## Changing the law

To change the common catalog `Law` or fit table, bump `Law::version`, re-run palette search
and material checks, and update save/protocol compatibility as needed. To change a simulation,
implement a native `Model` in its own module and bump `ModelSpec::version` when its law or saved
state interpretation changes. Game systems consume observations and visuals without adding
named-material special cases.
