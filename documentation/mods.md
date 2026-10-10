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

- `PackageInfo { id, name, version, description, kind, dependencies, register }` — one
  compiled-in package of any `PackageKind` (`Mod`, `Library`, `Bundle`): its `mod.toml` identity,
  its direct dependencies from the lock (a bundle's are its members), and for a mod its entry
  point `fn register(&mut ModRegistrar)`. Libraries and bundles have `register: None`.
- `GameBuild` — the packages of one build in registration (dependency) order, plus the
  environment hash of the `pwc.lock` it came from. Generated builds use
  `GameBuild::from_static(ENVIRONMENT, PACKAGES)`; `GameBuild::vanilla()` has no packages.
  `GameBuild::info()` is the read-only `BuildInfo` (`packages()`, `package(id)`, `environment()`).
- `ModDescriptor { id, name, version, register }` with `GameBuild::with_mod` is the 2.x shorthand
  for a mod package with no description and no dependencies; tests still use it.
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

where `game_build()` lists every package of the lock, of every kind. Nothing scans directories or
loads code at run time. The build decides which mods are in: there is no switch in the game.

## Registration

`Mods::from_build(&build)` calls the `register` of each package that has one, in order. The
`ModRegistrar` it gets:

- `add(m)` installs a value implementing `Mod`; it runs for as long as the build has it;
- `option(spec)` declares a tunable in the core's options registry and returns its `OptionId`
  (see [Options](#options));
- `screen_entry(entry)` offers a screen on the main menu and/or the pause screen (see
  [Screens](#screens));
- `provide(handle)` / `get::<T>()` share values between packages during registration — a package
  reads what its dependencies provided (the inventory reads the hotbar's shared state this way);
- `package()` is the package being registered, and `build()` the whole build as a `BuildInfo`:
  every package of every kind, including those registered later. A package that shows what is
  installed copies what it needs here (`PackageInfo` is `Copy`); `pwc_mod_api::bundles_of`
  works out bundle membership from the bundles' dependency lists. The core names no package.

## The `Mod` trait

`Mod` methods default to no-ops; a mod implements only `id` and the hooks it needs (`name`
defaults to the id and is only for logs and saves older than ids). A mod is *active* unless the
core suspended its package for the session (see [Suspension](#suspension)). When several active
mods implement a hook:

- Fan-out, in installation order: `update`, `on_toggle_fly`, `on_block_break`, `on_break_rejected`,
  `on_place_rejected`, `on_tool_changed`, `on_tool_used`, `on_game_event`, `on_audio`. `hud` uses the same order as z-order (later draws on
  top), and `commands` lists concatenate in it. `actions` are collected from every active mod.
  `on_audio` runs every frame, menus included. `on_options` reaches every mod, suspended ones too.
- First active wins: `close_overlay` (first `true`), `root_screen`, `pause_screen`, `tool`,
  `namer`, `appearance`, `worldgen`, `worldgen_config`, `run_command` (first `Some`).
- Compose: `visual_group` bits OR into the render mask: a lane is allowed when an installed,
  unsuspended package provides its group.

`tool` names the configuration a primary action applies (a left click uses it, a right click
places it). With no mod answering, there is no tool and the primary action breaks the block into
the inventory. `on_tool_changed(old, new)` tells every mod a tool reaction turned that unit into
another configuration. `on_tool_used` reports what the law did; the core draws nothing for it.
Mods declare input with `actions` (`Action { id, label, default, repeat, held }`). `held` stays on
for every frame the chord is down; menus do not sample it, and `repeat` is ignored while it is set.
The core keeps the
chord table, a core binding wins a clash, and `ModContext::action` reports which ids fired.
`ModContext::wheel` is the signed scroll. There is no controls screen: `Action::label` is what a
future one will show, and rebinding is left for later. HUD output is data
(`HudElement::Label` / `Panel` / `Rect`) drawn by the core theme. A machine changes a cell through
`ModContext::world` with `set_block`, then wakes its contacts with `note_cell_changed` (or
`note_block_moved(from, to)`). Chunk load, generation, meshing and saving never wake anything; on
a client connected to a server the authority runs the scheduler.

The core has no console commands and no flight toggle of its own. `run_command(ctx, cmd, args)`
handles a console line (its default asks the context-free `command(cmd, args)`) with the player,
world, settings and sky in a `CommandContext`; the core follows up on what changed (applies and
saves changed settings, re-mixes the audio, sends a changed clock to the server, streams a moved
player's destination and reports it as a teleport). `commands()` lists a mod's commands for
`/help` and Tab completion; a line no enabled mod handles gets a short hint. `on_toggle_fly(ctx)`
receives the flight key (`F`), latched and replayed at the mod tick like the other edges. The
Developer Toolkit package (`pwc.dev-toolkit`) provides the commands and flight.

`id()` is a stable key: world saves (per-mod `save_state`) are keyed by it.

## Suspension

The only runtime state a mod has is whether its package is suspended. A server that refuses a
package lets the client join with it suspended for that session (`Mods::suspend_packages`, undone
by `resume_packages` on leaving); benchmarks pin it with `WATT_BENCH_SUSPEND=<ids>`, and
`WATT_BENCH_VISUALS=core` suspends every package that provides a visual group. A suspended mod's
hooks do not run, its actions and screen entries are gone, and `Hello` does not report it (`Hello`
reports every package of kind mod that is not suspended). Suspension is never saved and has no UI;
screens see it as `ScreenFacts::suspended`.

## Options

One core-owned registry holds every value a player may tune. The core's own settings are one
owner, `"core"` (the `SETTINGS` table over `Settings`); a package declares more at registration:

```rust
let relief = registrar.option(
    OptionSpec::percent("relief", "Terrain Relief", Category::World, (25, 200, 25), 100).next_world(),
);
registrar.add(MyWorldgen { relief, value: 100 });
// later, in Mod::on_options(&mut self, options: &Options):
self.value = options.int(self.relief); // by index: no strings on any frame path
```

Kinds are `toggle`, `choice`, `percent` and `float`; `next_world()` marks an option that applies to
new worlds, and `legacy_key(k)` reads an older `settings.cfg` key once. The core keeps the values,
calls `Mod::on_options` after registration, after loading `settings.cfg`, and whenever
`Options::revision` moves, and persists them after its own keys as `<package-id>.<key>=`. Lines no
declared option claims are kept and written back. `OptionsView` lists core settings and package
options through one interface, by index, so a settings screen renders both without naming a
package and no package depends on a settings screen.

## Screens

Menus are mods; the core only hosts them (`src/screen.rs`). A `Screen` reads the core's raw
`MenuInput` and the `ScreenFacts` (saves, session, hosting, a notice, the waiting `Phase`,
in-world, the build's `BuildInfo`, the suspended packages, the registered entries, the visual
mask) and answers `Stay`, `Back`, `Push(screen)`, `Open(entry id)` or `Request(AppRequest)`: new
world, load, delete, host, join, cancel, resume, leave world, quit. It changes tunables only
through `ScreenContext::options_mut`, so the host applies and saves them. It draws by pushing
`UiElement`s (text and rectangles; the font is monospace, `text_width` is the metric), which the
core renders: a mod never touches the frame.

Screens reach the player through two slots and an entry registry, so no menu names another:

- `Mod::root_screen(facts)`: the screen out of a world (a start screen), built each time the
  player returns to it; it also draws the connecting and loading page from `facts.phase`.
- `Mod::pause_screen(facts)`: opened by Esc in a world after text capture and every overlay
  declined the key. The world keeps running; the game reads no input meanwhile.
- `ModRegistrar::screen_entry(ScreenEntry { id, label, places, order, open })`: a screen offered
  on `Places::MAIN`, `Places::PAUSE` or both. A root or pause screen lists
  `facts.entries_for(place)` and opens one with `ScreenOutcome::Open(id)`.

## Core fallbacks (vanilla)

With no mod answering a hook the core falls back to: no screens at all (the game enters the most
recent readable world, or a new one, on its first frame, settings live only in `settings.cfg`, and
Esc saves and quits; with a root screen but no pause screen, Esc leaves the world as it always
did), `FlatAppearance` (every texel the base colour), `describe` names read off the observation
("glowing clear hard solid"), the flat world generator, and no tool. The inventory still collects
what you break; without the inventory mod it is simply not shown. The number keys select nothing
unless the hotbar package is installed. Without the sounds mod the game plays no cues. Without the
proximity-chat mod the microphone stays closed and voice is neither sent nor played.

## The mod API crate

`crates/pwc-mod-api` is the one crate mods depend on. It re-exports the host types
(`Mod`, `ModContext`, `ModRegistrar`, `GameBuild`, `BuildInfo`, `PackageInfo`, `VisualMask`, …), the game
modules mods may use (`block`, `world`, `player`, `ui`, `screen`, `settings`, `inventory`, `render_config`,
`net`, `session`, `sim`, `input`, `derived`, `engine`, `material`, `audio`) and a `prelude`. Its version is
the **mod API version** a package's `mod.toml` requires (`pwc-api = "^2.0"`); a breaking change to
what it re-exports needs a major version bump. The 2.2.0 and 2.1.0 additions and the 2.0.0 breaks
are listed at the top of `crates/pwc-mod-api/src/lib.rs`. `audio` plays cues and voice; it does not
expose the device. Besides re-exports it has one function of its own, `bundles_of`.

## First-party mods

The first-party mods live in the package manager repository under `mods/` (namespace `pwc`), each
a package with its own README, licence (`Apache-2.0 OR MIT`) and tests:

| Package | Mods (ids) |
|---|---|
| `pwc.ui-kit` | library — the menu framework, the default look and the text widgets |
| `pwc.start-screen` | `start` — the root screen: main menu, Worlds page, forms, waiting page |
| `pwc.settings-menu` | entry "Settings" — every core setting and package option, by page |
| `pwc.mod-menu` | entry "Mods" — the build's packages, read-only |
| `pwc.pause-menu` | `pause_menu` — the pause screen: Resume, pause entries, Leave World |
| `pwc.hotbar` | `hotbar` — the selection UI along the bottom; it answers `tool` |
| `pwc.inventory` | `inventory` — the inventory as a list, equips into the hotbar (depends on `pwc.hotbar`) |
| `pwc.visuals` | `atmosphere`, `post`, `lighting` — the fancy render lanes |
| `pwc.neural-textures` | `neural_textures` — per-configuration CPPN textures |
| `pwc.material-names` | `material_names` — Markov-model names for blocks and tools |
| `pwc.infinite-diffusion` | `diffusion` — selects the InfiniteDiffusion world generator and declares its options |
| `pwc.game-ui` | `game_ui` — in-world HUD pieces, starting with the facing indicator |
| `pwc.sounds` | `sounds` — footsteps, blocks, tools, swings and menu clicks |
| `pwc.proximity-chat` | `proximity_chat` — push-to-talk voice for visible peers |
| `pwc.essentials` | bundle of all of the above |

The InfiniteDiffusion generator itself (`src/world/terrain`) stays in the core: the content
fingerprint, the LOD far field and the saves reference it. The mod selects it and declares its
options.

## Persistence

Package options live in `settings.cfg` (config root) after the core's keys, as
`<package-id>.<key>=`. There is no `mods.cfg`. Per-world state (`save_state`) lives in the world
save, keyed by mod id. A `pwc`-built instance has its own data directory (`WATT_DATA_DIR`), so its
worlds and settings never mix with the vanilla build's.

## Testing

The core cannot link the real mods in its own tests (they depend on it), so `src/modding/testing.rs`
provides stand-ins with the same ids, installed under their package ids, and the behaviours the
host and save tests exercise. Each package tests its own behaviour in the package manager
repository; `pwc_ui_kit::testing::Fixture` builds a `ScreenContext` for screen tests.
