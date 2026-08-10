# Audio and future crate boundaries

This note records useful dependency boundaries in the current codebase. The
crate names are working names, not a request to create a workspace immediately.
Extract a crate only when it removes a real build dependency, supports a second
consumer, or makes an ownership rule enforceable.

The governing rule is:

> Domain data and pure kernels may be used by adapters. They must not import the
> client, renderer, UI, device backends, or test tooling.

## The current audio seam

The audio path already has a strong snapshot boundary:

```text
game facts/events
      |
      v
AudioDirector -- policy and game adapters
      |
      v
AudioFrame -- validated, owned snapshot
      |
      v
SoundSystem -- allocation, mixing, continuation
      |
      v
backend -- Kira/device or silent fallback
```

[`audio/frame.rs`](../src/audio/frame.rs) is the most important part of this
shape. `AudioFrame::new` validates bounds, ordering, finite values, and stable
IDs once; the runtime can then consume the frame without cloning or repeating
validation. [`audio/acoustics.rs`](../src/audio/acoustics.rs) is also already a
pure kernel. Content loading, runtime voice allocation, capture, Opus sessions,
and device backends are cohesive audio concerns.

[`audio/director.rs`](../src/audio/director.rs) is different: it translates
`World`, `Connection`, `Console`, player motion, and block facts into audio
intent. That is client/game policy and should remain in the client when the
runtime becomes a crate. The audio crate should accept facts and snapshots, not
borrow the game.

Two dependencies must be corrected before extraction:

- [`world/query.rs`](../src/world/query.rs) currently constructs
  `audio::AcousticWindow`, while the director depends on `World`. Move the
  acoustic snapshot vocabulary to a neutral leaf, or let a game-side adapter
  build it from a generic world occupancy query. `world` must not depend on the
  output system consuming the snapshot.
- [`audio/palette.rs`](../src/audio/palette.rs) imports the block
  `SoundClass`. Either move that stable value type into the domain/content
  layer, or hand audio an audio-neutral material key.

Settings may still produce an audio mix at the client composition root, but a
future audio runtime should not know about `Settings`.

### Music ownership

Music should follow the same split. Game policy decides the desired musical
state, while the audio runtime owns file decoding, streaming, gain, crossfades,
looping, and device recovery. Long music tracks should have a streaming path
rather than being forced through the fully resident one-shot clip store.

An empty music asset directory is a content extension point, not an implicit
playlist. Later, a manifest can map stable track names to files and optional
loop/crossfade metadata. Missing or empty music content must remain a valid,
silent configuration.

## Recommended dependency direction

The smallest useful future shape is:

```text
watt-domain (std + glam; stable value types and primitive codec)
    |
    +--> watt-content/worldgen
    +--> watt-protocol
    +--> watt-save
    +--> watt-audio

watt-server --> domain + content/worldgen + protocol (+ save)
watt-client --> all of the above + voxel_engine
watt-tools  --> client, never the reverse
```

| Working crate | Owns | Must not own |
| --- | --- | --- |
| `watt-domain` | coordinates, stable IDs, portable pose/stance values, bounded primitive codec | renderer types, devices, app state |
| `watt-content` | elements, compositions, canonical block specs, content fingerprint, placement and deterministic world generation | GPU tables and resident meshes |
| `watt-protocol` | message schema and pure encode/decode | QUIC endpoints, client interpolation, server authority |
| `watt-save` | save document, codec, salvage, slot/store policy | `World`, `Player`, mods, renderer |
| `watt-audio` | catalog, acoustic kernel, runtime mixer, capture/voice decode, backend | `World`, network connection, console, settings |
| `watt-server` | authority state, validation, interest management, server transport | window, renderer, client UI/audio |
| `watt-client` | application composition, game adapters, UI, renderer integration | reusable wire/save formats |
| `watt-tools` | golden/stress harnesses and benchmark reporting | types imported by the shipping game |

This direction lets the dedicated server compile without Vulkan, Kira/CPAL, or
client test tooling, rather than merely relying on the final linker to discard
unused code.

## Extraction prerequisites

### Audio

1. Break the `world`/audio snapshot dependency described above.
2. Keep `AudioDirector` and the network/console/capture orchestration adapter in
   the client; move only the runtime and pure kernels.
3. Give material sound classification a domain-owned value or a narrow adapter.
4. Preserve `AudioFrame` as the only commit boundary and keep silent fallback a
   normal runtime state.

### Protocol and server

[`net/protocol.rs`](../src/net/protocol.rs) mixes pure message encoding with
Quinn-specific asynchronous framing. Move transport I/O beside the client and
server endpoints; the protocol crate should operate on bytes and standard I/O.

The wire `Stance` currently lives in [`presence.rs`](../src/presence.rs), which
also imports `Player` and owns rendering/animation values. Move only the closed
wire value and its codec downward; keep `Stance::of_player`, rigging, and tag
presentation in the client.

Content compatibility is also in the wrong layer:

- [`net/mod.rs`](../src/net/mod.rs) computes a fingerprint by reaching into
  block registration, world placement, and save helpers.
- [`save/mod.rs`](../src/save/mod.rs) owns canonical block-spec parsing used by
  both save and server code.

Canonical specs, registry compilation, worldgen version, and the fingerprint
belong together in content. The server can then validate exactly what the
client generates without depending on the client or save adapter.

### Save

The format/bridge split is already correct:
[`save/format.rs`](../src/save/format.rs) owns the document codec and
[`save/bridge.rs`](../src/save/bridge.rs) translates game types.

Before extraction, remove the format codec's incidental
`voxel_engine::DVec3` construction and make the shared primitive reader/writer
depend only on standard/domain values. Keep the bridge in the client.

### Domain and content

Do not move files mechanically until these upward dependencies are removed:

- [`coord.rs`](../src/coord.rs) imports `world::chunk::CHUNK_SIZE`; the lattice
  constant belongs with coordinates or another lower-level world vocabulary.
- [`ident`](../src/ident/mod.rs) imports both `BlockId` and the renderer's
  `Detail`.
- [`block/registry.rs`](../src/block/registry.rs) stores renderer `Color` and
  `Pass` alongside semantic and physical block facts.
- [`macros.rs`](../src/macros.rs) hardcodes both block paths and
  `voxel_engine::Color`; the built-in content macros should move with content
  and emit domain values.
- [`world/generation.rs`](../src/world/generation.rs) returns the runtime
  `ChunkData`. A neutral generated volume/cell sink, or a lower-level chunk
  value, is needed before worldgen can stand alone.

Split semantic block facts from render-derived tables. The client can derive
GPU-facing color/pass tables at its adapter boundary without making content
depend on the renderer.

### Harness and benchmark tooling

The current game imports `CameraPose`, `DebugView`, and key colors from
[`harness`](../src/harness/mod.rs), while the harness imports `Game`. Move the
small debug-control vocabulary into `game` or a client render-debug module so
the harness depends on the game in one direction only.

The recent [`benchmark`](../src/benchmark/mod.rs) subsystem still combines run
state, statistics, machine/GPU probing, display/git inspection, JSON writing,
and emission, although system probing and JSON encoding now live in focused
child modules. Before a tools-crate extraction, have the app assemble a plain
`BenchmarkSnapshot`; the benchmark should not take `Settings`, `Engine`, and
`World` directly.

## Safe module splits now

These are code-motion cleanups, not crate migrations:

- Split the remaining `audio/mod.rs` into a small facade plus
  `audio/runtime.rs`. Asset discovery and the device-less backend have already
  moved to `audio/assets.rs` and `audio/backend/null.rs`. Keep module
  declarations/re-exports in the facade and move `SoundSystem`, allocation
  state, helpers, and its backend-seam tests together.
- The benchmark's first safe split is complete:
  `benchmark/{mod.rs,system.rs,json.rs}` isolates system probes and JSON
  serialization while preserving its public API.
- Move `StreamLane`, `Candidates`, the shared admission loop, ordering helpers,
  and the mesh/section/light lane implementations from `world/mod.rs` into the
  existing [`world/lanes.rs`](../src/world/lanes.rs), which already owns lane
  registration.
- Move the draw/compose/HUD/peer portion of `game.rs` into a presentation child
  module after moving debug types out of the harness.

Each split should preserve tests and public paths; avoid combining code motion
with behavioral changes.

## Remaining audio roadmap

The current pass makes effects, layered loops, UI tracks, capture gating, asset
discovery, and headless catalog validation operational. Three larger changes
remain intentionally separate:

- Extend the new app-wide audio service tick beyond UI-track reclamation. It is
  the lifecycle hook for a future streaming `MusicDirector`, backend completion
  polling, and output-device recovery.
- Move voice payloads from the reliable ordered control stream to QUIC
  datagrams or a dedicated bounded lane. Jitter/PLC cannot undo
  head-of-line blocking introduced before packets reach the decoder.
- Distinguish the audible allocation budget from resident backend resources.
  Muted voice sessions and preserved loops have useful continuity, but they
  still occupy backend tracks and need an explicit physical cap.

## Do not split yet

- **The whole `world` subsystem.** It currently owns CPU truth, scheduling
  queues, worker claims, renderer handles, and GPU lifetime. First separate
  resident-render resources from world data and make the scheduler boundary
  explicit.
- **Scheduler and simulation.** `world::lanes` imports the scheduler while the
  scheduler names `World`; the scheduler also imports the simulation tick
  constant while simulation imports scheduler context. Crates cannot preserve
  these cycles. A generic scheduler context and a neutral clock constant are
  prerequisites.
- **Settings.** It deliberately composes renderer, world, UI, and audio policy
  for the client. It has no second independent consumer yet.
- **Mods.** Current mods borrow `World`, `Player`, UI data, and engine input.
  Define a stable capability/context API before treating them as an external
  crate or ABI.
- **The entire `ident` module.** Its edit-log vocabulary is explicitly
  forward-looking and not wired into the active save/network/world path. Move
  only proven shared value types; do not elevate an unused abstraction into a
  crate boundary.
- **Tiny utility crates.** Hash, memo, and one-off macros do not justify
  independent versioning or build graph cost. Move them with the subsystem that
  owns their contract.

## Completion checks for any extraction

An extraction is successful when:

- `watt-server` builds without renderer, window, benchmark, harness, or audio
  device dependencies;
- protocol and save codec tests run without a GPU, window, network socket, or
  audio device;
- lower crates expose no `voxel_engine::Engine`, `Frame`, GPU handle, `World`,
  `Settings`, or UI types;
- content fingerprints, worldgen output, save bytes, and protocol bytes remain
  bit-for-bit covered by compatibility tests;
- audio still degrades to silence on missing content/device and a valid
  `AudioFrame` remains sufficient for one runtime commit.

Use feature gates as a transitional measurement tool if useful, but prefer a
real crate only after the intended dependency direction can be expressed
without cycles.
