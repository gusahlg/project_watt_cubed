# Handoff 2026-10-03 — mods leave the game: `.pwcmod` packages and the PWC package manager

The owner asked to move the mods out of `src/`, give every mod its own package folder (README,
assets, licence, source, `mod.toml`), define a `.pwcmod` format, and create a package manager
repository holding all current mods, following their "PWC modding, packaging and build
architecture" direction (a modded PWC is an exact build). Work was fanned out to parallel agents.

## Game repository (this one)

- `src/mods/` is gone. The mod **host** is `src/modding/` (`Mod`, `Mods`, `ModContext`, …) plus
  `src/modding/build.rs`: `GameBuild`, `ModDescriptor { id, name, version, register }`,
  `ModRegistrar` (`add`, `add_disabled`, `declare_group`, `provide`/`get` shared handles).
  `Mods::from_build` registers packages in dependency order; `Mods::with_defaults` no longer
  exists and the core installs **no** mod.
- `project_watt_cubed::run(GameBuild)` is the entry point; `src/main.rs` runs
  `GameBuild::vanilla()`. `harness::golden_main(build)` replaces the old `golden` main (the
  `golden` bin calls it with vanilla); goldens honour `WATT_GOLDEN_DIR`.
- `crates/pwc-mod-api` (version 1.0.0 = the mod API version) re-exports the host and the game
  modules mods may use; several runtime modules became `pub` for it.
- `examples/mods/hello-hud` — a minimal complete mod package (AGPL-3.0-or-later).
- Core tests use stand-ins (`src/modding/testing.rs`) with the old ids; the real mods' tests moved
  into their packages. 708 lib tests pass.
- `mods.cfg` notice on the Mods screen now points at `pwc mod add/remove`.
- `MOD_POLICY.md` gained a "Current implementation" section: native compile-in packages are the
  local pre-registry stage; the sandboxed WebAssembly ABI remains the rule for official mods.

**Consequence:** `cargo run --release`, `nix run`, `./play.sh`, the benchmark (`WATT_BENCH`) and
`golden` now exercise **vanilla** PWC (flat world, flat colours, core menus). Full-experience
benchmarks and goldens come from a `pwc`-built instance: `pwc run -- …` (same `WATT_BENCH_*`
variables) and `pwc run --golden`. Pinned baselines from before this change measured the old
built-in mods; re-pin against `pwc.essentials` builds.

## PWC package manager (`../pwc-package-manager`, new, Apache-2.0 OR MIT)

Crates: `pwc-manifest` (mod.toml / instance.toml / pwc.lock, licence policy), `pwc-package`
(.pwcmod canonical tar + zstd, sha256 package hash), `pwc-store` (content-addressed immutable
store), `pwc-resolver` (deterministic backtracking resolution), `pwc-instance` (XDG dirs, config,
repositories, locking), `pwc-builder` (generated instance crate + mod bundle, Build ID, cache),
`pwc-cli` (`pwc`). Specs in `docs/spec/`, policy in `POLICY.md`.

First-party packages in `mods/` (namespace `pwc`, AGPL-3.0-or-later code + CC-BY-SA-4.0 assets per
MOD_POLICY): `pwc.menus`, `pwc.start-screen`, `pwc.hotbar`, `pwc.inventory` (depends on
`pwc.hotbar`), `pwc.visuals`, `pwc.neural-textures`, `pwc.material-names`,
`pwc.infinite-diffusion`, and the bundle `pwc.essentials`.

## Next

- Worlds and multiplayer should record/compare the lock's environment hash (the doc's
  "worlds remember their environment"; the runtime already receives it in `GameBuild`).
- The worldgen code still lives in the core; only its selection and knobs are a mod.
- Registry (phase 11) and launcher (phase 12) are not started.
