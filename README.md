A super awesome game, trust.

A voxel is a **configuration** of points on a 4-D resource lattice; one integer
**law** decides how neighbouring matter changes. Names are presentation only —
see [documentation/material-model.md](documentation/material-model.md).

## Multiplayer

The world is procedural, so multiplayer stays cheap: the network never ships voxel
data — a join transfers only the world **seed** and the sparse overlay of player
**edits**, and every client regenerates the terrain locally. Live play is just small
position, edit, and chat messages, with position updates interest-managed so traffic
stays sub-quadratic as the player count grows. The server is authoritative and
headless (no window, no GPU), which is what lets it scale and run on a plain box.

### Host from the game

Pick **Host Server** on the start menu, choose a port, an optional password, and your
name, then press Enter. This starts an integrated server and drops you straight into
its world. Friends join with your machine's IP and that port.

### Join a server

Pick **Join Server**, enter the address, port, password, and your name, then Enter.

### Dedicated server

For a server that runs on its own (e.g. a VPS), use the `watt_server` binary:

```sh
cargo run --release --bin watt_server -- --port 5555 --password hunter2 --seed 42
```

All flags are optional: with no `--seed` a fresh one is chosen and printed; with no
`--password` the server is open to anyone who can reach the port. `--data-dir`
sets the data/config root (same as `WATT_DATA_DIR`).

### Chat

Press `T` to open the chat/console line. Plain text is **proximity chat** (only
nearby players hear it); prefix a message with `!` for **global chat**; a line
starting with `/` is a local command (e.g. `/tp`).

### Safety

Connections are password-gated and version/content-checked (a build whose
worldgen would produce a different world from the shared seed is refused at
join). Every wire frame is length-capped, each client is rate-limited, and
pre-auth connections are bounded. Movement is plausibility-checked server-side
(implausible jumps are snapped back; `/tp` is an explicit request the server
may refuse via `--no-teleport`), and every edit is validated against reach,
spec well-formedness, and the cell's current revision — racing edits resolve
to exactly one winner and the loser's client rolls its prediction back.
Traffic uses QUIC encrypted with TLS 1.3. The server currently generates a
fresh self-signed certificate and clients accept any certificate, so server
identity is **not authenticated**. Passive observers cannot read the traffic,
but an active man-in-the-middle can impersonate the server and capture the
application password. Use a trusted network or VPN until certificate pinning
or trust-on-first-use is implemented.

Server-side mods: install `ServerMod` values on `Config.hooks` (`validate_edit`, `on_join`/`on_leave`, `on_chat`) with plain-data arguments and no wire change.
Hooks run in order outside the roster lock; the first `Deny` wins (an edit `Deny` is the same `EditRejected` path as a lost race).
A panicking hook is logged once and treated as `Allow` so a broken mod cannot take the server down.

## Graphics

Rendering runs on [voxel_engine](../voxel-engine), our own Vulkan 1.3 renderer
(pure Rust over `ash` + `winit` — no C build step). Expect greedy-meshed
chunks, frustum culling, reversed-Z depth, and uncapped frame rates by default.

Graphics are tunable at runtime from **Settings** on the start menu or the
`/gfx` console command in game (`/gfx fullscreen on`, `/gfx vsync off`,
`/gfx msaa 4`, `/gfx fps 144`, `/gfx renderdist 8`, `/gfx fov 90`). Settings
persist in `settings.cfg` under the config root.

## State location

Worlds are stored in the data root; `settings.cfg`, `session.cfg`, and `mods.cfg` in the config root.
If the launch directory already contains `saves/`, both roots stay that folder (checkouts and existing installs).
Otherwise `$XDG_DATA_HOME/project_watt_cubed` and `$XDG_CONFIG_HOME/project_watt_cubed`. Override with `WATT_DATA_DIR` or `watt_server --data-dir <dir>`.
`WATT_CHECKOUT_DIR` is the source checkout (`play.sh` sets it to the repo root) and is ignored unless that directory contains `Cargo.toml`.

## Running on NixOS

The included `flake.nix` wires up the Vulkan loader and windowing libraries.
It expects the `voxel-engine` repo checked out as a sibling directory
(`../voxel-engine`).

```sh
# Drop into a dev shell with the Rust toolchain and native deps, then run:
nix develop
cargo run --release

# Or build/run the packaged binary directly:
./play.sh   # re-pins the ../voxel-engine flake input, then `nix run`
nix build   # produces ./result/bin/project_watt_cubed
```

A plain `nix run` also works, but note the trap `play.sh` exists to avoid: the
flake pins the sibling engine's committed `main` revision, so after a
new engine commit a bare `nix run` can build the current game against a stale
engine API. `nix flake update voxel-engine` re-pins it. Uncommitted engine edits
are intentionally visible only to the dev-shell `cargo` path; commit them before
testing the pure Nix package.

On macOS, install MoltenVK and the Vulkan loader once (`brew install
molten-vk vulkan-loader`) and use plain `cargo run --release`.

## Licensing and contributions

Project-owned software is licensed `AGPL-3.0-or-later`; project-owned art,
audio, and documentation are licensed `CC-BY-SA-4.0`. Third-party material
keeps its own compatible licence. See [LICENSE.md](LICENSE.md) for the complete
matrix, [THIRD_PARTY.md](THIRD_PARTY.md) for provenance and notice handling,
and [CONTRIBUTING.md](CONTRIBUTING.md) for DCO sign-off requirements.
