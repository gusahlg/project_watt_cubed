# Example mods

Tiny mods that document the mod API (`crates/pwc-mod-api`) and double as integration tests for the
PWC package manager. Real mods live in their own repositories; the first-party ones are in the
[PWC package manager](https://github.com/gusahlg/pwc-package-manager) repository under `mods/`.

Each directory is a complete mod package source tree (`mod.toml`, `README.md`, licence files,
`src/lib.rs`), licensed AGPL-3.0-or-later like all PWC mod code (`MOD_POLICY.md`).

| Directory | Package | Shows |
|---|---|---|
| `hello-hud/` | `example.hello-hud` | `register`, the `Mod` trait, an event hook and a HUD line |
