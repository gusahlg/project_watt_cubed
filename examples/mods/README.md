# Example mods

Tiny mods that document the mod API (`crates/pwc-mod-api`) and double as integration tests for the
PWC package manager. Real mods live in their own repositories; the first-party ones are in the
[PWC package manager](https://github.com/gusahlg/pwc-package-manager) repository under `mods/`.

Each directory is a complete mod package source tree (`mod.toml`, `README.md`, licence files,
`src/lib.rs`), licensed `Apache-2.0 OR MIT` like the first-party mods, so it can be copied freely
as the start of a new mod. Mods may use any free licence that `MOD_POLICY.md` accepts (one
compatible with the game's `AGPL-3.0-or-later`); proprietary mods are not accepted.

| Directory | Package | Shows |
|---|---|---|
| `hello-hud/` | `example.hello-hud` | `register`, the `Mod` trait, an event hook and a HUD line |
