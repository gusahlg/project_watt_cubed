# Mods

A mod is a compiled-in layer on the thin core: menus, inventory, crafting, the
shipped look, block appearance, and optional worldgen. Toggle them on the Mods
screen. Adding a new mod still needs a rebuild and a restart.

## Hooks

`Mod` methods default to no-ops. When several enabled mods implement a hook:

- Fan-out, install order: `update`, `on_block_break`, `on_break_rejected`,
  `on_place_rejected`. `hud` uses the same order as z-order (later draws on top).
- First enabled wins: `menu_theme`, `close_overlay` (first `true`), `worldgen`,
  `worldgen_config`, `appearance`.
- Compose: `visual_group` bits OR into the render mask.

`worldgen_config` is an opaque string. The winning worldgen kind parses it
(InfiniteDiffusion reads its own knobs). `knobs` / `step_knob` are per-mod.

## Appearance

`BlockAppearance` turns a render descriptor (`Visual`) into one 16×16 RGBA8
layer. The world's texture cache asks `Mods::appearance()` — the first enabled
mod that returns `Some`, else core `FlatAppearance` (every texel is `rgb` with
`alpha`). A `revision` change rebuilds every layer; otherwise the cache is
append-only per descriptor.

Shipped appearance mods:

- **Procedural textures** (`procedural_textures`, Essentials, on). The
  two-colour value-noise generator (frequency, roughness, alpha, glow). Knobs
  `grain` and `contrast` (0..2, default 1) scale grain and colour separation;
  stepping either bumps `revision`. Disable this to let another appearance mod
  win, or to get flat colours with no appearance mod enabled.
- **GPU materials** (`gpu_materials`, ungrouped, off). Uploads one engine
  `MaterialDesc` per layer (`Visual` bytes 1:1, `MATERIAL_FLAG_PROCEDURAL` set)
  and a 1×1 placeholder so the array index exists. The GPU paints the pattern.
  With this mod off the game never calls `set_material_descs`; the engine
  table stays the default `ARRAY_LAYER` per slot (bit-identical to a renderer
  that has no descriptor table). Enable it and disable `procedural_textures`
  so it wins the first-enabled-wins seam.

To write your own: implement `BlockAppearance` (`layer`, `revision`,
`wants_gpu_descriptors`) and return `Some(self)` from `Mod::appearance`.
`layer` must be a deterministic function of `Visual` (and your knobs). Bump
`revision` when knobs change. Set `wants_gpu_descriptors` only if you upload
engine `MaterialDesc`s; the cache will then send 1×1 placeholders instead of
16×16 CPU layers. With no appearance mod (`Mods::empty()`) the game renders
flat colours.

## Groups

`Mod::group` returns a group id (`""` = ungrouped). `Mods::GROUPS` holds id,
display name, and description. Built-ins use `essentials` (**Essentials**): the
layers that make the game playable as shipped. The Mods screen shows each group
as a section title, an Enable all / Disable all row, then its mods indented.
Ungrouped mods follow under Other.

## Persistence

`saves/mods.cfg` stores `id=on|off` and optional `id.state=` knob payloads.
A group toggle writes each member's line; there is no group-level key.
Per-world state (`save_state`) lives in the world save, keyed by id.

Enable/disable is runtime. Visual mods apply on the next world; worldgen on the
next new world. New mods are compiled in: rebuild, then restart, to see them.
