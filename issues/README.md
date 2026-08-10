# Project Watt Cubed audit — 2026-07-13 (burn-down 2026-07-14)

This folder is a forward-looking engineering audit of Project Watt Cubed and its live path dependency, `../voxel-engine`. It is not a claim that the design notes are a frozen specification. I treated the notes as a direction of travel:

- a small but powerful core with modding boundaries;
- an infinite 3D world whose materials emerge from elements;
- deterministic seed-plus-sparse-overlay multiplayer;
- camera-relative precision at extreme coordinates;
- performance work that preserves correctness and inspectability.

Those ideas are useful tests for architecture. They are not reasons to preserve a draft implementation when the implementation contradicts the desired game.

## Status after the 2026-07-14 burn-downs

Every confirmed bug from the audit is FIXED with regression tests — both the
game side (G-01 through G-17, protocol v6: authoritative edits with acks and
rollback, content fingerprint, permissioned teleport, position corrections,
visibility exits, synchronized day cycle) and the engine side (E-01 through
E-08: exposure slot correctness, truthful TAA barriers, toggle resets, pure
support-checked swapchain selectors, bounded LOD draw inputs, documented
bounded curvature, and the water depth-absorption local-read path repaired
and LIVE with a zero-error validation smoke). The one remainder is the
reduced G-03 (generated overhangs/islands still don't shadow the columns
beneath them — constructed roofs now do). Streaming re-prioritizes to the
live view centre, deschedules left-behind jobs, and the far-LOD load lane is
level-triggered so the covering can never silently stall behind the live
frontier. Details and the per-issue record live in
[confirmed-bugs.md](confirmed-bugs.md).

## Highest-value next work

| Priority | Finding | Why it matters |
|---|---|---|
| P1 | [R-01/R-11: publishable engine URL + clean-checkout CI](nix-and-release.md) | Reproducible releases beyond this machine. |
| P2 | [G-03 remainder: generated volumetrics don't shadow](confirmed-bugs.md#g-03--generated-overhangs-and-islands-do-not-shadow-lower-chunks) | A visible lighting simplification under islands/overhangs. |
| P2 | [R-04…R-10: release packaging/tooling roadmap](nix-and-release.md) | Split product binaries, shader provenance, XDG state paths, golden references. |
| — | [structural-opportunities.md](structural-opportunities.md) | Architecture directions (server-side economy state, deterministic simulation, texture indirection, …). |

## Files in this audit

- [confirmed-bugs.md](confirmed-bugs.md) — the reduced G-03 remainder and the record of everything fixed.
- [structural-opportunities.md](structural-opportunities.md) — larger design and performance improvements that should inform future work.
- [testing-and-graphics.md](testing-and-graphics.md) — commands, outcomes, graphics observations, and a practical test roadmap.
- [branch-integration.md](branch-integration.md) — what changed on main, how conflicts were resolved, and what still needs attention.
- [nix-and-release.md](nix-and-release.md) — the reproduced `nix run` incident, package fixes, measurements, and release/CI roadmap.

Severity is impact-oriented: **P0** means a correctness/security failure worth designing around now; **P1** means material correctness or architectural risk; **P2** means a real but bounded defect; **P3** means maintainability, portability, or polish debt. “Confirmed” means a code path, failing test, validation message, or direct reproduction supports the conclusion. Performance ideas without a measurement are kept in the opportunities file rather than presented as bugs.
