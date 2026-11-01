# Example modules

Tiny modules that document the module API (`crates/pwc-module-api`) and double as integration tests for the
PWC package manager. Real modules live in their own repositories; the first-party ones are in the
[PWC package manager](https://github.com/gusahlg/pwc-package-manager) repository under `modules/`.

Each directory is a complete module package source tree (`module.toml`, `README.md`, licence files,
`src/lib.rs`), licensed `Apache-2.0 OR MIT` like the first-party modules, so it can be copied freely
as the start of a new module. Modules may use any free licence that `MODULE_POLICY.md` accepts (one
compatible with the game's `AGPL-3.0-or-later`); proprietary modules are not accepted.

| Directory | Package | Shows |
|---|---|---|
| `hello-hud/` | `example.hello-hud` | `register`, the `Module` trait, an event hook and a HUD line |
