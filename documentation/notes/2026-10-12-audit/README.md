<!-- SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors -->
<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# Project exploration and audit — 2026-10-12

The most urgent findings are failures of shared invariants: a reaction can lose matter, the
server checks charted collision in the wrong coordinate space, tool actions bypass edit-policy
mods, and tool confirmations have weaker delivery/reconciliation than ordinary edits.

The strongest architectural opportunity is to make the existing seams complete. Compiling mods
into one executable is already implemented. Independently supplied world generators, authoritative
mod simulation, and arbitrary GUI presentation are still limited by runtime-owned choices.

This audit adds documentation and reproduction fixtures only. It does not fix production code,
change saves/settings, commit changes, publish anything, or modify the sibling repositories.

## Reading order

1. [Architecture and philosophy](architecture.md): what the project is becoming, what is actually
   implemented, and where the core currently owns experience choices.
2. [18 bug reports](bugs.md): impact, triggers, evidence, proposed fixes and verification criteria.
3. [Optimization and design directions](directions.md): proposals ordered by value, with concrete
   experiments and performance tradeoffs.
4. [Verification and coverage](verification.md): exact scope, commands, results and limitations.
5. [Reproduction fixtures](reproductions/): tests of observed behavior, run in a temporary copy by
   `python3 notes/2026-10-12-audit/reproductions/run.py` inside the development shell.

## Priorities

| First | Why |
| --- | --- |
| B01: atomic reaction commits | Matter conservation is the foundation of simulation and gravity. |
| B02–B03: coordinate/topology agreement | Server validation and reactions must see the same world as the player. |
| B04: tool-policy bypass | A protection mod cannot reliably protect cells today. |
| B06–B08: tool reconciliation and reservations | Successful network operations can lose state or matter. |
| B11–B12: save recovery | Recovery should not put the sole intact copy at risk. |
| B16: LOD budget violation | Existing tests reproduce cuts that exceed the runtime's asserted bound. |
| D01–D03: generator, simulation and GUI seams | These unlock the forms of customization in the owner's vision. |

P1/P2 labels in the reports express repair priority, not a measured frequency. Tests that assert
an observed bug passing mean that the defect was reproduced; they are not fixed-behavior
regression tests. One additional synthetic stale-tool ordering risk is kept separate from the
18 reports because reachable wire reordering was not established.

## Snapshot

- Game: `b2b1b5b770e284d8cec6146e4d9fa52dc91cc365`, package 0.6.1.
- Renderer sibling: `990309d3c1c87ef49273dbdddd16abe1c5c1fa13`.
- Package-manager sibling: `f9e6970ac73613553823df29f641eed6cc2b0e96`.
- Pre-existing uncommitted changes: `src/world/quadtree.rs` and `src/world/rule/probes.rs`.
  The review and tests use that working tree. In particular, the LOD findings are **not** attributed
  to a particular commit or to those edits without a separate comparison.
- Implemented identities: save format 10, protocol 17, worldgen 11, mod API 3.1.0.

Coverage is broad but not exhaustive. The repository has roughly 67,000 lines even in a shallow
Rust-file count, plus deeper terrain/streaming modules and sibling projects. No claim is made
that every line or every graphical behavior has been inspected. Findings have explicit confidence
and evidence instead of relying on that claim.
