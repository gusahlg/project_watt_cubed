# Mods

A mod is a compiled-in layer on the thin core: menus, the inventory and hotbar, block textures,
material names, and the world generator. Toggle them on the Mods screen. Adding a new mod still
needs a rebuild and a restart.

Matter itself is not a mod. The law, the intern table and the reaction scheduler live in core
(`documentation/material-model.md`). Mods present matter, place it, hold it, and (for machines)
change cells, which wakes their reactions.

## Hooks

`Mod` methods default to no-ops. When several enabled mods implement a hook:

- Fan-out, install order: `update`, `on_block_break`, `on_break_rejected`, `on_place_rejected`,
  `on_tool_changed`. `hud` uses the same order as z-order (later draws on top).
- First enabled wins: `menu_theme`, `start_screen`, `close_overlay` (first `true`), `held`,
  `namer`, `appearance`, `worldgen`, `worldgen_config`.
- Compose: `visual_group` bits OR into the render mask.

`held` names the block the player holds (what a left click uses as a tool and a right click
places); with no mod answering, the hand is bare. `on_tool_changed(old, new)` tells every mod that
a tool reaction turned the held unit into another configuration (the hotbar slot follows it).

A machine changes a cell through `ModContext::world` with `set_block`, then wakes its contacts in
the reaction scheduler with `note_cell_changed` (or `note_block_moved(from, to)` for a move, which
wakes both places). Chunk load, gen, mesh and save never wake anything. On a client connected to a
server the authority runs the scheduler.

HUD output is data (`HudElement::Label` / `Panel` / `Rect`), drawn by the core theme.

## Shipped mods (group **Essentials**, all on)

| Id | What it does |
|---|---|
| `menus`, `start` | Pause/settings menus and the start screen (Worlds page). |
| `inventory` | Press I: the core stash as a list. ↑/↓ choose, 1-9 equip into that hotbar slot, Enter equips into the selected slot. |
| `hotbar` | Nine slots plus the bare hand along the bottom. 1-9 / wheel select, 0 the hand. The selected slot is the held tool; new materials drop into the first free slot. |
| `atmosphere`, `post`, `lighting` | The shipped look (sky/fog/clouds, bloom/TAA/exposure, shadows/AO/blocklight). |
| `neural_textures` | Paints every configuration's texture with a CPPN grown from it (below). |
| `material_names` | Names every configuration and its tool (below). |
| `diffusion` | InfiniteDiffusion, the world generator (below). Off: a flat world. |

## Appearance: neural textures

`BlockAppearance::layer(src, out)` paints one 32×32 RGBA8 layer from an `AppearanceSource` (the
law, the configuration's `Block`, its `Visual` and observation). The world's texture cache asks
`Mods::appearance()` — the first enabled mod that returns `Some`, else core `FlatAppearance`
(every texel the base colour). A `revision` change repaints every layer.

`neural_textures` (`src/mods/textures/neural.rs`) grows a small compositional pattern-producing
network **from the configuration**: each distinct element contributes a first-layer neuron (weights,
frequency and activation hashed from its coordinates, gain from its multiplicity); the second layer
is wired from the configuration's digest. Blocks that share elements share structure; a block that
gains a constituent grows a neuron, so every distinct configuration gets its own texture. Hardness
draws veins, weak cohesion grain, glow lights the crests, clarity sets alpha. Inputs are periodic,
so the tiles are seamless. Knobs: `detail`, `contrast`.

## Naming: material names

`MaterialNamer::names(src)` returns a block name and a tool name, computed once per interned
configuration and kept by the registry as presentation. With no namer enabled the core describes
the observation ("glowing clear hard solid").

`material_names` (`src/mods/naming.rs`) speaks through a character-level Markov model (orders 3→1
with back-off) trained on real mineral, rock and element names, driven by a generator seeded from
the configuration — every peer names everything alike. The root comes from the most abundant
element (near-twins share a family name); the suffix says how it reads (`-ine` clear, `-ium`
glowing, `-ite` hard, `-ate` firm, `-ash` soft); a qualifier names a strong second element or a
busy mixture ("Banded", "Veined", "Brittle"); the tool noun follows mass and look (shard, chisel,
pick, maul, sledge; lens or lantern). Knob: `style`. Names never reach the law, worldgen, saves or
the wire.

## Worldgen: InfiniteDiffusion

`src/world/terrain` — one pure function of `(seed, coordinate)`, three realms:

- **Surface**: continents of basins, hills and great ranges; ridged crests over eroded slopes, long
  carved valleys, mesas in arid belts, banded strata on every cliff; biomes by altitude,
  temperature and moisture (meadow, forests, desert, scree, snow).
- **Underground**: tunnels and caverns, glowing fungus and crystal in the deep, ore and reagent
  veins, and abandoned **mines** — timbered corridors on several levels, rails, rare lamps,
  collapses, rooms and shafts.
- **Space** above `SPACE_FLOOR` (640): planets (rocky, icy, verdant, desert, crystal, molten) with
  crusts, mantles and glowing cores, rings and moons, asteroids and stars. The sky turns black and
  starry as the camera climbs out of the atmosphere.

Materials come from the palette search in the law (`material-model.md`). Knobs (new worlds):
`relief`, `caves`, `mines`, `space` (percent). The batch chunk path and the per-voxel path share one
definition, so workers, clients and reactions reading unloaded cells agree.

## Groups

`Mod::group` returns a group id (`""` = ungrouped). Built-ins use `essentials`. The Mods screen
shows each group as a section, an Enable all / Disable all row, then its mods.

## Persistence

`mods.cfg` stores a `version=` marker, `id=on|off` lines and optional `id.state=` knob payloads.
Files from before version 2 recorded `diffusion=off` as the default of an old experiment; those
lines are ignored so the world generator stays on. Per-world state (`save_state`, e.g. the hotbar
slots) lives in the world save, keyed by id.

Enable/disable is runtime. Visual mods apply on the next world; worldgen on the next new world.
