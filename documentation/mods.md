# Mods

A mod is a compiled-in layer on the thin core: menus, inventory, crafting, the
shipped look, and optional worldgen. Toggle them on the Mods screen. Adding a
new mod still needs a rebuild and a restart.

## Hooks

`Mod` methods default to no-ops. When several enabled mods implement a hook:

- Fan-out, install order: `update`, `on_block_break`, `on_break_rejected`,
  `on_place_rejected`. `hud` uses the same order as z-order (later draws on top).
- First enabled wins: `menu_theme`, `close_overlay` (first `true`), `worldgen`,
  `worldgen_config`.
- Compose: `visual_group` bits OR into the render mask.

`worldgen_config` is an opaque string. The winning worldgen kind parses it
(InfiniteDiffusion reads its own knobs). `knobs` / `step_knob` are per-mod.

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
