# Project Watt Cubed audit — 2026-07-13 (burn-down 2026-07-14)

This folder is a forward-looking engineering audit of Project Watt Cubed and its live path dependency, `../voxel-engine`. It is not a claim that the design notes are a frozen specification. I treated the notes as a direction of travel:

- a small but powerful core with modding boundaries;
- an infinite 3D world whose materials emerge from elements;
- deterministic seed-plus-sparse-overlay multiplayer;
- camera-relative precision at extreme coordinates;
- performance work that preserves correctness and inspectability.

Those ideas are useful tests for architecture. They are not reasons to preserve a draft implementation when the implementation contradicts the desired game.

## Status after the 2026-07-14 burn-down

Every game-side confirmed bug (G-01 through G-17) is FIXED with regression
tests, except the reduced remainder of G-03 (generated overhangs/islands still
do not shadow the columns beneath them — constructed roofs now do). The wire
protocol moved to v6: authoritative edits with acks and rollback, a content
fingerprint in the handshake, an explicit permissioned teleport, position
corrections, peer visibility exits, and a synchronized day cycle. Fast-motion
chunk streaming now re-prioritizes to the live view centre and deschedules
left-behind jobs. Details and the per-issue record live in
[confirmed-bugs.md](confirmed-bugs.md).

## Highest-value next work

| Priority | Finding | Why it matters |
|---|---|---|
| P1 | [E-01: exposure reads the wrong in-flight slot](confirmed-bugs.md#e-01--exposure-readback-reads-a-slot-that-was-not-the-one-waited) | Finite but torn exposure values can cause brightness instability. |
| P1 | [E-02: TAA barriers omit actual prior accesses](confirmed-bugs.md#e-02--taa-history-barriers-do-not-describe-the-real-previous-accesses) | Temporal corruption is possible even when basic validation is quiet. |
| P1 | [R-01/R-11: publishable engine URL + clean-checkout CI](nix-and-release.md) | Reproducible releases beyond this machine. |
| P2 | [G-03 remainder: generated volumetrics don't shadow](confirmed-bugs.md#g-03--generated-overhangs-and-islands-do-not-shadow-lower-chunks) | A visible lighting simplification under islands/overhangs. |
| P2 | [E-03…E-08: engine toggle/portability/shader debt](confirmed-bugs.md#voxel-engine-and-gpu) | Correctness and portability of the renderer surface. |
| — | [structural-opportunities.md](structural-opportunities.md) | Architecture directions (server-side economy state, deterministic simulation, texture indirection, …). |

## Files in this audit

- [confirmed-bugs.md](confirmed-bugs.md) — the open engine/GPU issues, the reduced G-03 remainder, and the record of everything fixed.
- [structural-opportunities.md](structural-opportunities.md) — larger design and performance improvements that should inform future work.
- [testing-and-graphics.md](testing-and-graphics.md) — commands, outcomes, graphics observations, and a practical test roadmap.
- [branch-integration.md](branch-integration.md) — what changed on main, how conflicts were resolved, and what still needs attention.
- [nix-and-release.md](nix-and-release.md) — the reproduced `nix run` incident, package fixes, measurements, and release/CI roadmap.

Severity is impact-oriented: **P0** means a correctness/security failure worth designing around now; **P1** means material correctness or architectural risk; **P2** means a real but bounded defect; **P3** means maintainability, portability, or polish debt. “Confirmed” means a code path, failing test, validation message, or direct reproduction supports the conclusion. Performance ideas without a measurement are kept in the opportunities file rather than presented as bugs.
