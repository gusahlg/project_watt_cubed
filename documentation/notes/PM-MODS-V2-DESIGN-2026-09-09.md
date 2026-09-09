# Mods v2 — "everything that is not a core mechanic is a mod" (design, 2026-09-09)

Directive (2026-09-09): everything that is not a core mechanic becomes a mod; block textures "and such"
become mods; world generation is mod-only; mods stay compiled into the binary and as fast as core
code; the mods screen says the game must be recompiled for mod choices to apply.

## 1. The line between core and mods

**Core = mechanics and machinery**, the things every world needs to exist and be consistent between
players: the element/composition block model and its chemistry (`block/`), the world store, streaming,
LOD, light, meshing and the edit overlay (`world/`), players, physics and the break/place events, the
network protocol and server, saves, settings/presets and the effective render config, input routing,
the audio mixer/director engine, the console, the bench/harness, and the mod runtime itself plus the
minimal start interface (`StartAction`/`StartFacts`).

**Mods = content and presentation**, anything a player might want to swap: world generation methods,
block appearance (textures), sounds and music content, the start screen, menu theme, HUD, inventory
and crafting presentation, minimap, name tags/player models, atmosphere/post/lighting choices.

**Rule for the core: it must boot with zero mods.** For every mod-provided surface the core carries a
minimal fallback: a void/flat generator (one stone slab at y = 0), flat derived-colour block layers,
the default menu theme, the core start screen (new world / load most recent / settings / mods / quit),
silence. An invariant test keeps this true: `Mods::empty()` → new world → 60 frames → save → load → quit.

## 2. Build-time selection ("recompile to apply")

Mod choices are a build input, not runtime state:

- `mods.toml` at the repo root, **tracked in git** (the Nix flake builds from the git tree, so an
  untracked file under `saves/` cannot influence a build). Shape:
  ```toml
  [mods]                    # id = enabled; unknown ids are an error at build time
  start_screen = true
  classic_worldgen = true
  infinite_diffusion = false
  ```
- `build.rs` reads it (`cargo:rerun-if-changed=mods.toml`), checks ids against the mod manifest
  (one table in `build_support/manifest.rs`: id, module path, type, group, default, kind), emits
  `cargo:rustc-cfg=mod_<id>` for every enabled mod (declared with `rustc-check-cfg`) and generates
  `$OUT_DIR/mod_registry.rs` with `install_all(&mut Mods, deps)` for the enabled set and
  `ALL_MODS: &[ModInfo]` (every known mod, with its compiled-in flag) for the mods screen.
- Each mod module is `#[cfg(mod_<id>)]`: a disabled mod is **not compiled** — no code, no vtable, no
  `if enabled` checks, smaller binary, faster builds. This is what makes mods "as performant as core".
- Runtime: compiled-in mods are always active. The `enabled` flag, `on_enable`/`on_disable` and
  runtime toggling go away. Mod *knobs* (worldgen parameters, visual choices) stay runtime settings
  persisted at once in the data dir (`mods.cfg`).
- The mods screen lists `ALL_MODS`, shows "compiled in" per mod, and toggles the **selection** in
  `mods.toml`. Notice (two lines): "Choices are saved at once. The game has to be recompiled for mod
  choices to apply." / "Mods are compiled into the binary: run ./play.sh (or cargo build) and restart."
  `play.sh` exports `WATT_CHECKOUT_DIR=$PWD` so the running game knows where `mods.toml` lives; without
  it (a release binary outside a checkout) the screen shows the selection read-only with the notice
  "no checkout: edit mods.toml and rebuild".
- Later, optional: generated static fan-out (`Mods` as a struct with one field per enabled mod and
  generated `for_each_*` functions) removes the remaining `dyn` calls. Not needed for performance today
  (≈10 indirect calls per frame); the cfg gating is the part that matters.

## 3. World generation as mods

- Core keeps the `TerrainGenerator` trait (`world/generation.rs`) — the contract the streamer needs:
  `heights_16`, column fill, `lod_column`, ceilings, water level — plus `VoidGenerator` (the fallback)
  and the `Unit`/noise helpers if other mods share them (`world/noise.rs`).
- The classic generator (`Terrain` and its column/profile/fill code) moves to
  `mods/worldgen/classic/` and the InfiniteDiffusion adapter to `mods/worldgen/diffusion.rs`
  (`crates/infinite-field` stays a crate). Moves are `git mv` to keep history and follow **after** the
  perf tasks on those files (36, 18, 55) have merged.
- `WorldgenKind` (an enum in core, on the wire as a u8, in saves as a kind) becomes a **string id + cfg
  string**: the mod id. Save format bumps (v8: `worldgen=<id>` + cfg; v6/v7 kinds map to the two
  ids), protocol bumps (v10: `Welcome` carries id + cfg). A save whose worldgen mod is not compiled in
  loads read-only with a clear message ("world needs mod <id>; enable it in mods.toml and rebuild"),
  never a panic or a silently different terrain.
- With several worldgen mods compiled in, the new-world flow offers a radio choice ("Worldgen:
  Classic / InfiniteDiffusion"), default = first in `mods.toml` order; the choice persists as a knob.

## 4. Block appearance as a mod

- Core: `BlockAppearance` trait — `fn layer(&self, registry, id, out: &mut [u8; 16*16*4])` and
  `fn revision(&self) -> u32` — consulted by the world's incremental texture cache
  (`world/streaming.rs`, today `block::texture::build_block_texture`). The fallback paints the derived
  colour flat. The call happens only when a block id is first registered, so there is no hot-path cost.
- The procedural 16×16 generator (`block/texture.rs`) becomes the "Procedural textures" mod
  (Essentials). Texture packs (PNG per composition class) are then just another appearance mod.
- Same seam later for block *sounds* (material → clip) and *models* if they ever exist.

## 5. Sounds and music content as a mod

- Core keeps the mixer, the director (event derivation: footsteps, splash, block break/place) and the
  acoustic window. The **content** — which clip plays for which `SoundEvent`, the music playlist
  (`audio/content.rs`, `audio/palette.rs`, `assets/sounds`, `assets/music`) — becomes the "Sounds"
  mod: `AudioContent` trait (`clip_for(event) -> Option<ClipId>`, `music()`), silence as the fallback.

## 6. Smaller surfaces

Minimap (HUD mod), name tags + player models ("Presence visuals" mod), the `/gfx` console commands
stay core (dev tooling), day/night stays core for now (it feeds light, a mechanic).

## 7. Migration order (grok tasks; each ends with the zero-mods boot test green)

| task | what | conflicts with |
|---|---|---|
| 59a | `Paths::checkout_dir()` from `WATT_CHECKOUT_DIR`; `play.sh` exports it | none (b) |
| 59 | `mods.toml` + `build.rs` codegen + `cfg(mod_*)` + generated registry; mods screen edits the selection; runtime enable state removed | mods/, app.rs, menus — after today's merge |
| 60 | worldgen: `VoidGenerator`, classic → mod, diffusion → mod, string ids (save v8, protocol v10), radio choice | generation.rs (55) — after merge |
| 61 | block appearance trait + procedural textures mod + flat fallback | streaming.rs texture cache |
| 62 | sounds/music content mod + silent fallback | audio/ |
| 63 | minimap + presence visuals as mods | game/draw |
| 64 | optional: generated static fan-out | mods/ |

Performance rules for every mod (unchanged): no per-frame allocation, hooks batched, caches keyed on
revisions, worker-side work goes through the streamer's job system, never a thread of its own.
