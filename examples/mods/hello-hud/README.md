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

AGPL-3.0-or-later, like all PWC mod code (see `MOD_POLICY.md`). Copy it to start your own mod.
