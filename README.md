A super awesome game, trust.

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
`--password` the server is open to anyone who can reach the port.

### Chat

Press `T` to open the chat/console line. Plain text is **proximity chat** (only
nearby players hear it); prefix a message with `!` for **global chat**; a line
starting with `/` is a local command (e.g. `/tp`).

### Safety

Connections are password-gated, every wire frame is length-capped, each client is
rate-limited, and every edit is bounds- and reach-validated server-side, so a client
can't reach across the map or flood the server. Traffic is **not** encrypted — host
behind a VPN or trusted network if you need confidentiality on the wire.

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
