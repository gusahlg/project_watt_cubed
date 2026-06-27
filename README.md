A super awesome game, trust.

## Running on NixOS

This project depends on `raylib`, which builds native C code and needs OpenGL +
X11 libraries at runtime. The included `flake.nix` wires all of that up.

```sh
# Drop into a dev shell with the Rust toolchain and native deps, then run:
nix develop
cargo run

# Or build/run the packaged binary directly:
nix run
nix build   # produces ./result/bin/project_watt_cubed
```
