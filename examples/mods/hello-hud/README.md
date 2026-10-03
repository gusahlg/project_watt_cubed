# Hello HUD

The smallest useful PWC mod, kept in the game repository as a living example of the mod API
(`crates/pwc-mod-api`). It adds one HUD line in the top-right corner that counts the blocks you
broke this session.

- **Controls:** none. Toggle it on the Mods screen.
- **Persisted state:** none.
- **Dependencies:** none.

Try it with the PWC package manager:

```bash
pwc instance create hello --use
pwc mod add /path/to/project_watt_cubed/examples/mods/hello-hud
pwc run
```

## Licence

Licensed under either of

- Apache License, Version 2.0 ([LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)), or
- MIT licence ([LICENSES/MIT.txt](LICENSES/MIT.txt))

at your option.

Copy it to start your own mod. `Apache-2.0 OR MIT` is the recommended licence for new mods; any
other free licence accepted by `MOD_POLICY.md` and the package manager's `POLICY.md` (for example
`MPL-2.0` or `AGPL-3.0-or-later`) works too. Proprietary mods are not accepted.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
this package by you, as defined in the Apache-2.0 licence, shall be dual licensed as above, without
any additional terms or conditions.
