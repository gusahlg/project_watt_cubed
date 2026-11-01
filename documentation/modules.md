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

The common material catalog, canonical configurations, observations and intern table remain
in core. Optional native simulation mods own interaction laws, queues and dynamic state;
core hosts their authority, storage adapter, commits and persistence
([material model](material-model.md)). Mods also present, place and hold matter. Their scope is governed by
[MOD_POLICY.md](../MOD_POLICY.md) and the package manager's
[`POLICY.md`](https://github.com/gusahlg/pwc-package-manager/blob/main/POLICY.md).

## Licences

The game, including `crates/pwc-mod-api` and this runtime, is `AGPL-3.0-or-later`; the package
manager tooling is `AGPL-3.0-or-later` too. Mod packages are licensed separately: a mod's code may
use any free-software licence on the package manager's allowlist of licences compatible with the
game's `AGPL-3.0-or-later` (for example `Apache-2.0 OR MIT`, `MIT`, `MPL-2.0`,
`GPL-3.0-or-later`, `AGPL-3.0-or-later`), and its assets an approved free-content licence.
Proprietary mods are not accepted. Existing first-party packages in the package manager and
`examples/mods/hello-hud` use `Apache-2.0 OR MIT`; the physics packages under this repository's
`mods/` use `AGPL-3.0-or-later` for code and `CC-BY-SA-4.0` for documentation. A built instance links
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
- `project_watt_cubed::run(build)` starts the game. Before a window or an audio device opens,
  a first argument that names a command registered with `ModRegistrar::cli` (or `help` /
  `--commands`) runs that command and exits. Any other arguments are ignored and the game starts.
  `harness::golden_main(build)` runs the golden-shot harness for that build (blessed images in
  `WATT_GOLDEN_DIR`, default `tests/golden`).

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

`Mods::from_build(&build, &mut options)` calls the `register` of each package that has one, in
order; the options the packages declare go into the app's `Options`, beside the settings. The
`ModRegistrar` it gets:

- `add(m)` installs a value implementing `Mod`; it runs for as long as the build has it;
- `option(spec)` declares a tunable in the core's options registry and returns its `OptionId`
  (see [Options](#options));
- `screen_entry(entry)` offers a screen on the main menu and/or the pause screen (see
  [Screens](#screens));
- `provide(handle)` / `get::<T>()` share values between packages during registration — a package
  reads what its dependencies provided (the inventory reads the hotbar's shared state this way);
- `cli(Cli { name, about, run })` registers a command-line entry point (see
  [Command-line entry points](#command-line-entry-points-api-33)). The first registration of a
  name wins;
- `package()` is the package being registered, and `build()` the whole build as a `BuildInfo`:
  every package of every kind, including those registered later. A package that shows what is
  installed copies what it needs here (`PackageInfo` is `Copy`); `pwc_mod_api::bundles_of`
  works out bundle membership from the bundles' dependency lists. The core names no package.

## The `Mod` trait

`Mod` methods default to no-ops; a mod implements only `id` and the hooks it needs (`name`
defaults to the id and is only for logs and saves older than ids). A mod is *active* unless the
core suspended its package for the session (see [Suspension](#suspension)). When several active
mods implement a hook:

- Fan-out, in installation order: `update`, `on_frame`, `on_message`, `on_block_break`,
  `on_break_rejected`, `on_place_rejected`, `on_tool_changed`, `on_tool_used`, `on_game_event`,
  `on_audio`. `hud` uses the same order as z-order (later draws on top). `actions` are collected
  from every active mod. `on_audio` runs every frame, menus included. `on_options` reaches every
  mod, suspended ones too.
- First active wins: `close_overlay` (first `true`), `root_screen`, `pause_screen`, `tool`,
  `namer`, `appearance`, `worldgen`, `worldgen_config`, `reaction_model` (first `Some`). The keyboard capture belongs
  to the first mod that asks until it gives it back or Escape ends it.
- Compose: `visual_group` bits OR into the render mask: a lane is allowed when an installed,
  unsuspended package provides its group.

`tool` names the configuration a primary action applies (a left click uses it, a right click
places it). With no mod answering, there is no tool and the primary action breaks the block into
the inventory. Held-material reactions are delegated to the selected simulation model; without
a provider, they do nothing. `on_tool_changed(old, new)` tells every mod a tool reaction turned that unit into
another configuration. `on_tool_used` reports what the law did; the core draws nothing for it.
Mods declare input with `actions` (`Action { id, label, default, repeat, held }`). `held` stays on
for every frame the chord is down; menus do not sample it, and `repeat` is ignored while it is set.
The core keeps the
chord table, a core binding wins a clash, and `ModContext::action` reports which ids fired.
`ModContext::wheel` is the signed scroll. There is no controls screen: `Action::label` is what a
future one will show, and rebinding is left for later. HUD output is data
(`HudElement::Label` / `Panel` / `Rect`) drawn by the core theme. A machine changes a cell through
`ModContext::world` with `set_block`, then wakes its contacts with `note_cell_changed` (or
`note_block_moved(from, to)`). These wake the selected model; they are inert without a provider.
Chunk load, generation, meshing and saving never wake anything. In multiplayer, the server
runs the model; a client frame hook is not an authoritative simulation tick.

The core has no console, no commands and no flight key of its own. `on_frame(ctx)` runs every
in-world frame with the immediate actions, the keyboard capture (`FrameContext::capture_text`) and
the game state in a `GameContext`; the core follows up on what it changed (applies and saves
changed settings, re-mixes the audio, sends a changed clock to the server, streams a moved
player's destination and reports it as a teleport). `on_message` hands over chat lines, joins,
leaves and the core's notices. `pwc.chat`, `pwc.commands` and `pwc.dev-toolkit` provide the chat,
the commands and flight on top of them.

`id()` is a stable key: world saves (per-mod `save_state`) are keyed by it.

## Native simulation models (API 3.2)

`Mod::reaction_model()` returns `sim::model::ModelSpec { id, version, create }`, a native
factory for a `Model: Send`. The first active provider wins; a provider's own enabled option
can make it return `None`. No provider is the production default, with no automatic reactions
or held-material reaction. Package installation remains a compile-time choice.

The model owns wakeups, its queue or per-cell fields, budgeted `tick`, optional `tool`,
`stats`, and `save`/`load`. Core provides `CellStore`, including atomic pair admission, and
records committed mutations through its usual world/save/network paths. Keep instance state
such as phase or reservoir content separate from shared material IDs. `World::simulation_cell`
maps a physical point to the cell address; `simulation_model_id`, `simulation_sample`, and
`simulation_stats` support inspection. `perturb_simulation` is explicit local laboratory input
and does not submit a network request.

Disabling or switching a provider preserves its versioned payload for later resumption.
Alternatives run separately; their laws are not composed by installation order. Hosting copies
the selected factory into server configuration; a headless build can set
`net::server::Config::reaction_model` explicitly. Protocol 18 carries the model id/version
fingerprint, but only the server runs the model, so it never refuses a join: a client with any
provider, or none, joins any server (allow and deny lists still apply to provider packages). A
server that cannot run a world's saved model starts with it frozen, as single-player does; an
`off` world adopts the server's model. Only committed block changes are replicated;
experimental per-cell fields and input are not networked yet. The prototype mods therefore
gate their injection actions to singleplayer.

## Command-line entry points (API 3.3)

A package registers a command from `register`:

```rust
registrar.cli(pwc_mod_api::Cli {
    name: "swarm",
    about: "run headless agents",
    run: swarm::main,
});
```

`project_watt_cubed::run` looks at the first argument before it opens a window or an audio device.
When that argument equals a registered name, it registers the build's mods and loads options (so
`on_options` has the saved values), calls `run` with a `CliContext { args, data_dir, build }`, and
exits with the returned code. `args` is everything after the command name. `data_dir` is the data
root from `Paths`. `pwc-game --commands` and `pwc-game help` print each command as
`commands:` then `  {name}  {about}` and exit 0. Those two names stay reserved even when a mod
registers a command with the same name. Any other arguments leave today's behaviour: the game
starts.

## Headless clients (API 3.3)

`headless::HeadlessClient` is one player on a `watt_server`, with the game's world mirror and
physics and none of the window, the audio device, the mods or the input devices. `connect` blocks
on the calling thread until the join snapshot is applied and the collision slab around spawn is
loaded. `step` polls the network, applies look and movement through the same physics as the
windowed game, performs at most one `Act`, and sends the pose. `Place` and `UseTool` name a
material that must be in the inventory. `Break` with no tool breaks into the inventory.
`target` is the block the eye looks at; `ray` casts the same march along any direction. On a round
world both name storage cells, and `cell_center` places a cell in world space.

`World::keep_around(center, radius)` loads that collision slab and drops chunk data farther than
`radius` from the player, so many clients can stay resident. The windowed stream does not use it.
A headless client keeps a radius of 96 around the eye.

Packages in the package-manager repository:

- [physics-selective](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-selective/README.md): extracted original selective
  transfer, including its dormant contact queue and one-operation tool use.
- [physics-wave](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-wave/README.md), [physics-packets](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-packets/README.md),
  [physics-reservoir](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-reservoir/README.md), and
  [physics-rotor](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-rotor/README.md): independently selectable bounded laboratories.
  Their documentation states conservation guarantees and unresolved limitations.

The [simulation notes](notes/2026-10-13-simulation/research.md) compare the models and
future directions; prototype demonstrations do not establish useful gameplay or performance.

## Suspension

A package can be suspended for a session. A server that refuses it lets the client join with
it suspended for that session (`Mods::suspend_packages`, undone
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
new worlds, and `legacy_key(k)` reads a value saved under an older key once: an old core
`settings.cfg` key (Voice Chat reads `voice_enabled`), or a knob an older game kept in `mods.cfg`,
flattened as `<mod-id>.state.<knob>` (InfiniteDiffusion reads `diffusion.state.relief`). The core
reads `mods.cfg` only until `settings.cfg` records `options_format=1`, and never writes or deletes
it. The app keeps the values beside the settings, calls `Mod::on_options` after registration, after
loading `settings.cfg`, and whenever `Options::revision` moves, and persists them after its own keys
as `<package-id>.<key>=`. Lines no declared option claims are kept and written back.

`OptionsView` lists core settings and package options through one interface, by index (with
`full_key`, `show`, `hint`, `step` and `parse`), so a settings screen renders both without naming a
package and no package depends on a settings screen. A frame hook gets the same view through
`GameContext::options_mut` (that is how `pwc.commands`' `/set` reaches every setting and option),
and a screen through `ScreenContext::options_mut`; the core applies and saves what changed.

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
  declined the key. The world keeps running and is drawn behind the screen (the screen paints a
  translucent backdrop; `facts.in_world` is true); the game reads no input meanwhile.
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
unless the hotbar package is installed. Without a simulation provider, automatic reactions and
held-material reactions are inert. Without the sounds mod the game plays no cues. Without the
proximity-chat mod the microphone stays closed and voice is neither sent nor played.

## The mod API crate

`crates/pwc-mod-api` is the one crate mods depend on. It re-exports the host types
(`Mod`, `ModContext`, `ModRegistrar`, `GameBuild`, `BuildInfo`, `PackageInfo`, `VisualMask`, …), the game
modules mods may use (`block`, `world`, `player`, `ui`, `screen`, `settings`, `inventory`, `render_config`,
`net`, `session`, `sim`, `input`, `derived`, `engine`, `material`, `audio`) and a `prelude`. Its version is
the **mod API version** a package's `mod.toml` requires (`pwc-api = "^3.0"`); a breaking change to
what it re-exports needs a major version bump. The 3.0.0 breaks (everything mods-v3 changed) and
the earlier versions are listed at the top of `crates/pwc-mod-api/src/lib.rs`. `audio` plays cues
and voice; it does not expose the device. Besides re-exports it has `bundles_of` and
`testing::Harness`. The host type `Mods` is not part of the API: a mod cannot reach the other mods
of a build.

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
| `pwc.game-ui` | `game_ui` — the HUD: reticle, coordinates, frame rate, players, loading and link notices, facing indicator |
| `pwc.chat` | `chat` — the text chat on § and the scrollback of the game's messages |
| `pwc.commands` | `commands` — slash commands (`/help`, `/set`, `/time`, audio, `/op`) and a registry for other mods' commands |
| `pwc.chat-commands` | bundle of `pwc.chat` and `pwc.commands` |
| `pwc.sounds` | `sounds` — footsteps, blocks, tools, swings and menu clicks |
| `pwc.proximity-chat` | `proximity_chat` — push-to-talk voice for visible peers |
| `pwc.essentials` | bundle of all of the above |

The InfiniteDiffusion generator itself (`src/world/terrain`) stays in the core: the content
fingerprint, the LOD far field and the saves reference it. The mod selects it and declares its
options.

## Persistence

Package options live in `settings.cfg` (config root) after the core's keys, as
`<package-id>.<key>=`. An old `mods.cfg` is read once for knob values (see [Options](#options)) and
left on disk. Per-world state (`save_state`) lives in the world
save, keyed by mod id. A `pwc`-built instance has its own data directory (`WATT_DATA_DIR`), so its
worlds and settings never mix with the vanilla build's. Simulation selection and frozen model
payloads live in reserved `@simulation.selected` and `@simulation.state.<id>` records in the
save's existing opaque state section. Save format remains v10. Model versions are checked
before loading a payload; disabled models' saved state is retained.

## Testing

Most core mod-hook tests use `src/modding/testing.rs` stand-ins with the same package ids and
the behaviors the host/save checks exercise. Historical reaction tests include the extracted
selective test fixture only under `cfg(test)`; production vanilla does not install it. The actual
package is tested through the package manager's generated workspace.
Each package can test its behavior through `pwc_mod_api::testing::Harness` (a build's packages
registered with their options and the hooks as the game arbitrates them);
`pwc_ui_kit::testing::Fixture` builds a `ScreenContext` for screen tests.
