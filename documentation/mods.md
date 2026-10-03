# Mods

PWC mods are **compiled in**. A modded PWC is an exact build: one game version plus one exact
set of mod packages, turned into a native executable by the PWC package manager
([`pwc-package-manager`](https://github.com/gusahlg/pwc-package-manager), command `pwc`). This repository is the
runtime those builds link against; it contains no installed mods.

Responsibilities:

| Part | Answers | Where |
|---|---|---|
| PWC (this repo) | What is a mod, and how does it interact with the game? | `src/modding/`, `crates/pwc-mod-api` |
| Package manager | Which mods belong to this instance? | `pwc-package-manager` (`pwc mod add`, `pwc.lock`) |
| Builder | How do these exact mods become an executable? | `pwc-package-manager` (`pwc build`) |

Matter itself is not a mod. The law, the intern table and the reaction scheduler live in the core
(`documentation/material-model.md`). Mods present matter, place it, hold it, and (for machines)
change cells, which wakes their reactions. What mods may do is governed by
[MOD_POLICY.md](../MOD_POLICY.md) and the package manager's
[`POLICY.md`](https://github.com/gusahlg/pwc-package-manager/blob/main/POLICY.md).

## Licences

The game, including `crates/pwc-mod-api` and this runtime, is `AGPL-3.0-or-later`; the package
manager tooling is `AGPL-3.0-or-later` too. Mod packages are licensed separately: a mod's code may
use any free-software licence on the package manager's allowlist of licences compatible with the
game's `AGPL-3.0-or-later` (for example `Apache-2.0 OR MIT`, `MIT`, `MPL-2.0`,
`GPL-3.0-or-later`, `AGPL-3.0-or-later`), and its assets an approved free-content licence.
Proprietary mods are not accepted. The first-party packages and `examples/mods/hello-hud` are
`Apache-2.0 OR MIT`, which is also the recommended licence for new mods. A built instance links
its mods into the AGPL game, so distributing that executable follows the AGPL as a whole; each
mod's own source keeps its licence.

## Builds

`src/modding/build.rs`:

- `ModDescriptor { id, name, version, register }` — one compiled-in package: its `mod.toml`
  identity and its entry point `fn register(&mut ModRegistrar)`.
- `GameBuild` — the packages of one build in registration (dependency) order, plus the
  environment hash of the `pwc.lock` it came from. `GameBuild::vanilla()` has no packages.
- `project_watt_cubed::run(build)` starts the game; `harness::golden_main(build)` runs the
  golden-shot harness for that build (blessed images in `WATT_GOLDEN_DIR`, default
  `tests/golden`).

`src/main.rs` is the vanilla executable. `pwc build` generates a small crate (an *instance* crate
plus the *mod bundle*) whose `main` is

```rust
fn main() {
    project_watt_cubed::run(pwc_bundle::game_build());
}
```

where `game_build()` lists every package of the lock. Nothing scans directories or loads code at
run time; disabled mods cost nothing.

## Registration

`Mods::from_build(&build)` calls each package's `register` in order. The `ModRegistrar` it gets:

- `add(m)` / `add_disabled(m)` install a value implementing `Mod` (the player's `mods.cfg` choice
  still wins at load);
- `declare_group(group)` adds a section to the Mods screen (`ESSENTIALS` is well known);
- `provide(handle)` / `get::<T>()` share values between packages during registration — a package
  reads what its dependencies provided (the inventory reads the hotbar's shared state this way).

## The `Mod` trait

`Mod` methods default to no-ops; a mod implements only `name`, `id` and the hooks it needs. When
several enabled mods implement a hook:

- Fan-out, in installation order: `update`, `on_block_break`, `on_break_rejected`,
  `on_place_rejected`, `on_tool_changed`. `hud` uses the same order as z-order (later draws on
  top).
- First enabled wins: `menu_theme`, `start_screen`, `close_overlay` (first `true`), `held`,
  `namer`, `appearance`, `worldgen`, `worldgen_config`, `command`.
- Compose: `visual_group` bits OR into the render mask.

`held` names the block the player holds (what a left click uses as a tool and a right click
places); with no mod answering, the hand is bare. `on_tool_changed(old, new)` tells every mod a
tool reaction turned the held unit into another configuration. HUD output is data
(`HudElement::Label` / `Panel` / `Rect`) drawn by the core theme. A machine changes a cell through
`ModContext::world` with `set_block`, then wakes its contacts with `note_cell_changed` (or
`note_block_moved(from, to)`). Chunk load, generation, meshing and saving never wake anything; on
a client connected to a server the authority runs the scheduler.

`id()` is a stable key: `mods.cfg` lines (`id=on|off`, `id.state=` knob payloads) and world saves
(per-mod `save_state`) are keyed by it.

## Core fallbacks (vanilla)

With no mod answering a hook the core falls back to: the default menu theme and the fallback start
screen, `FlatAppearance` (every texel the base colour), `describe` names read off the observation
("glowing clear hard solid"), the flat world generator, the bare hand. The stash still collects
what you break; without the inventory mod it is simply not shown.

## The mod API crate

`crates/pwc-mod-api` is the one crate mods depend on. It re-exports the host types
(`Mod`, `ModContext`, `ModRegistrar`, `GameBuild`, `ModDescriptor`, `Knob`, `Group`, …), the game
modules mods may use (`block`, `world`, `player`, `ui`, `menu`, `settings`, `stash`, `render_config`,
`net`, `session`, `sim`, `input`, `derived`, `engine`, `material`) and a `prelude`. Its version is
the **mod API version** a package's `mod.toml` requires (`pwc-api = "^1.0"`); a breaking change to
what it re-exports needs a major version bump.

## First-party mods

The first-party mods live in the package manager repository under `mods/` (namespace `pwc`), each
a package with its own README, licence (`Apache-2.0 OR MIT`) and tests:

| Package | Mods (ids) |
|---|---|
| `pwc.menus` | `menus` — the standard menu look |
| `pwc.start-screen` | `start` — the start screen and its Worlds page |
| `pwc.hotbar` | `hotbar` — nine slots plus the bare hand |
| `pwc.inventory` | `inventory` — the stash as a list, equips into the hotbar (depends on `pwc.hotbar`) |
| `pwc.visuals` | `atmosphere`, `post`, `lighting` — the fancy render lanes |
| `pwc.neural-textures` | `neural_textures` — per-configuration CPPN textures |
| `pwc.material-names` | `material_names` — Markov-model names for blocks and tools |
| `pwc.infinite-diffusion` | `diffusion` — selects the InfiniteDiffusion world generator and its knobs |
| `pwc.essentials` | bundle of all of the above |

The InfiniteDiffusion generator itself (`src/world/terrain`) stays in the core: the content
fingerprint, the LOD far field and the saves reference it. The mod selects it and owns its knobs.

## Persistence

`mods.cfg` (config root) stores a `version=` marker, `id=on|off` lines and optional `id.state=`
knob payloads. Files from before version 2 recorded `diffusion=off` as the default of an old
experiment; those lines are ignored. Per-world state (`save_state`) lives in the world save, keyed
by mod id. A `pwc`-built instance has its own data directory (`WATT_DATA_DIR`), so its worlds and
choices never mix with the vanilla build's.

## Testing

The core cannot link the real mods in its own tests (they depend on it), so `src/modding/testing.rs`
provides stand-ins with the same ids, groups and the behaviours the host, menu and save tests
exercise. Each package tests its own behaviour in the package manager repository.
