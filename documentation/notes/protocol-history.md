# Protocol version history

Moved verbatim from `src/net/mod.rs` (`PROTOCOL_VERSION`). Client and server
must match exactly, checked at join.

v2: positions are 3x f64 on the wire (far-coordinate correctness).
v4: element-first worldgen (worldgen v2) — chunk materials are a pure
function of (seed, worldgen), so any worldgen change MUST bump this: mixed
peers would silently desync on terrain contents otherwise.
v5: worldgen v3 (alien pass — trees gone, per-biome crust, surface glow).
v6: authoritative edits (request id + expected cell revision + ack),
content fingerprint in `Hello`, explicit `Teleport`, `Position`
corrections, peer visibility exits, and synchronized day length.
v7: message payloads now frame `ident::codec`'s shared little-endian
primitives instead of a hand-rolled big-endian codec — wire bytes changed,
only the frame-length prefix stays BE.
v8: voice chat — `ClientMessage::Voice` and `ServerMessage::PeerVoice`
carry opaque opus frames. New message tags change the wire, so mixed v7/v8
peers must not join.
v9: `Welcome` carries worldgen kind + diffusion knobs so a joiner adopts
the server's generator instead of its local mod state.
