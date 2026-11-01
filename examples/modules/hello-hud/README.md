# Hello HUD

The smallest useful PWC module, kept in the game repository as a living example of the module API
(`crates/pwc-module-api`). It adds one HUD line in the top-right corner that counts the blocks you
broke this session.

- **Controls:** none. It is in a build when the instance has it (`pwc module add`), and runs there.
- **Persisted state:** none.
- **Dependencies:** none besides the module API (`pwc-api ^3.4`). Its test uses
  `pwc_module_api::testing::Harness`.

Try it with the PWC package manager:

```bash
pwc instance create hello --use
pwc module add /path/to/project_watt_cubed/examples/modules/hello-hud
pwc run
```

## Licence

Licensed under either of

- Apache License, Version 2.0 ([LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)), or
- MIT licence ([LICENSES/MIT.txt](LICENSES/MIT.txt))

at your option.

Copy it to start your own module. `Apache-2.0 OR MIT` is the recommended licence for new modules; any
other free licence accepted by `MODULE_POLICY.md` and the package manager's `POLICY.md` (for example
`MPL-2.0` or `AGPL-3.0-or-later`) works too. Proprietary modules are not accepted.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
this package by you, as defined in the Apache-2.0 licence, shall be dual licensed as above, without
any additional terms or conditions.
