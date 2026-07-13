# Project Watt Cubed audit — 2026-07-13

This folder is a forward-looking engineering audit of Project Watt Cubed and its live path dependency, `../voxel-engine`. It is not a claim that the design notes are a frozen specification. I treated the notes as a direction of travel:

- a small but powerful core with modding boundaries;
- an infinite 3D world whose materials emerge from elements;
- deterministic seed-plus-sparse-overlay multiplayer;
- camera-relative precision at extreme coordinates;
- performance work that preserves correctness and inspectability.

Those ideas are useful tests for architecture. They are not reasons to preserve a draft implementation when the implementation contradicts the desired game.

## Outcome at a glance

The audit started from clean `experimental` branches at game commit `90059cf` and engine commit `4e83755`. Relevant main work was integrated semantically and tested in disposable clones before the real branches were fast-forwarded:

- game merge `363dea6` brings in `origin/main` through `11a44ba`;
- engine merge `00d7803` brings in `origin/main` through `168618d`;
- engine follow-up `5efc734` disables an invalid Vulkan water-local-read path and corrects present wait stages;
- engine follow-up `6edce74` prevents the tonemap pass from binding multisampled depth to its single-sample godray descriptor. Godrays are conservatively disabled under MSAA until the renderer has a resolved depth image.

The merge preserves experimental's element-first worldgen, 14-bit texture layers, VRS/water/AO controls, allocator sizing, and initially its flake lock. A later packaging pass intentionally refreshed only the stale engine input. The merge adopts main's typed eye/feet avatar anchoring, quadrant-masked LOD coverage, winner-only LOD light voting, height/error mip, motion/altitude-aware selection, swap fade, per-slot shadows, and vignette. Main's global `target-cpu=native` Cargo configuration was intentionally excluded.

Small audit fixes and regressions were then added in the game worktree:

- reject protocol frames with trailing payload bytes;
- reject non-finite peer yaw/pitch;
- preserve camera-relative remote poses at far coordinates;
- keep opaque emissive chunks out of the all-dark light shortcut;
- use one metre-scaled reach for breaking and placement, and block freecam placement;
- give the LOD height-mip worker the compiled element palette rather than a too-short builtin palette;
- add an ignored red test for the known cross-chunk constructed-roof skylight bug.

A later pure-Nix follow-up repaired a stale engine pin that made `nix run` use
the pre-LOD API, changed the engine input from a roughly 5 GiB raw path snapshot
to a 949,443-byte Git-filtered revision, made the package version follow Cargo,
added package/source-budget flake checks, and fixed the runtime paths of every
installed binary. See [nix-and-release.md](nix-and-release.md).

## Highest-value next work

| Priority | Finding | Why it matters |
|---|---|---|
| P0 | [G-01: multiplayer economy is client-authoritative](confirmed-bugs.md#g-01--multiplayer-block-order-is-authoritative-but-gameplay-is-not) | Duplicate rewards, lost items, and no rollback on rejected edits. |
| P0 | [G-02: movement can forge edit reach](confirmed-bugs.md#g-02--client-reported-movement-can-forge-edit-reach) | The server's advertised authority/bounds guarantee can be bypassed. |
| P0 | [G-03: volumetric roofs do not shadow lower chunks](confirmed-bugs.md#g-03--overhangs-islands-and-edited-roofs-do-not-shadow-lower-chunks) | A visible lighting error and a wrong world-light model. |
| P0 | [G-04: worker panics can strand claims](confirmed-bugs.md#g-04--caught-worker-panics-can-permanently-strand-streaming-claims) | One bad job can prevent streaming convergence. |
| P1 | [G-05: teleport/freecam physics uses unloaded air](confirmed-bugs.md#g-05--teleport-and-freecam-return-run-physics-against-unloaded-air) | Players can fall through or become embedded in terrain. |
| P1 | [G-06: unbounded server spec/palette growth](confirmed-bugs.md#g-06--server-spec-interning-and-client-palette-growth-are-unbounded) | Public-server memory exhaustion and client content-cap exhaustion. |
| P1 | [E-01: exposure reads the wrong in-flight slot](confirmed-bugs.md#e-01--exposure-readback-reads-a-slot-that-was-not-the-one-waited) | Finite but torn exposure values can cause brightness instability. |
| P1 | [E-02: TAA barriers omit actual prior accesses](confirmed-bugs.md#e-02--taa-history-barriers-do-not-describe-the-real-previous-accesses) | Temporal corruption is possible even when basic validation is quiet. |
| P1 | [G-07: LOD edit reduction is order-dependent](confirmed-bugs.md#g-07--coarse-lod-edit-reduction-has-no-deterministic-ordering-contract) | Live play and a join snapshot can produce different far materials. |
| P1 | [G-08: no world/content fingerprint](confirmed-bugs.md#g-08--seed-only-multiplayer-has-no-worldgencontent-fingerprint) | Seed-only networking cannot detect cross-build world divergence. |

## Files in this audit

- [confirmed-bugs.md](confirmed-bugs.md) — reproducible or code-proven defects, including engine/GPU issues and work already fixed.
- [structural-opportunities.md](structural-opportunities.md) — larger design and performance improvements that should inform future work.
- [testing-and-graphics.md](testing-and-graphics.md) — commands, outcomes, graphics observations, and a practical test roadmap.
- [branch-integration.md](branch-integration.md) — what changed on main, how conflicts were resolved, and what still needs attention.
- [nix-and-release.md](nix-and-release.md) — the reproduced `nix run` incident, package fixes, measurements, and release/CI roadmap.

Severity is impact-oriented: **P0** means a correctness/security failure worth designing around now; **P1** means material correctness or architectural risk; **P2** means a real but bounded defect; **P3** means maintainability, portability, or polish debt. “Confirmed” means a code path, failing test, validation message, or direct reproduction supports the conclusion. Performance ideas without a measurement are kept in the opportunities file rather than presented as bugs.
