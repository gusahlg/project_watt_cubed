<!-- SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors -->
<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# Verification after canonical packaging and eight new laws

Performed 2026-10-13. Scope remains small correctness demonstrations, not a production physics
evaluation. The first-stage verification file is historical; this file records the current
package layout and final managed build.

## Canonical packaging and integration

- **33 of 33** source packages in `pwc-package-manager/mods` passed `pwc package check`, including
  all thirteen physics mods, the lab library and updated Essentials. Raw summaries and package
  hashes are in [managed-package-checks.txt](managed-package-checks.txt).
- All fourteen `physics-*` directories were checked to contain no Cargo manifest, Cargo lock,
  target directory or custom build script. Source formatting passed `rustfmt --check`.
- An isolated `PWC_HOME=/tmp/pwc-physics-managed` was set up with the real manager CLI. Its
  `physics-lab` instance resolved fourteen packages, stored immutable snapshots, generated the
  canonical Cargo crates and registration bundle, and successfully built the game and golden
  harness executables with `pwc build --profile dev --print-dir`.
- A second disposable `physics-default-check` instance resolved **Essentials 2.1.0**, nineteen
  total packages, and the selective-transfer dependency. It included no experimental physics.
  This was a resolution check; that full experience was not separately launched or compiled.
- The game passed `cargo check --workspace --all-targets --offline` after removal of the old
  package trees and introduction of the standalone test fixture. Both repositories passed
  `git diff --check`. Native libraries came from the existing development environment; Cargo
  used cached dependencies offline and the shared target cache.

No normal manager configuration, active instance, store, save or world was used. The tests' source
snapshots and generated Cargo files live in the isolated manager home. Temporary authoring
wrappers used before integration were outside all canonical source packages. No commits or
publication were made. Existing game changes to `quadtree.rs` (55 additions/15 deletions) and
`rule/probes.rs` (116 additions/7 deletions) remain as found.

## Focused checks from the generated workspace

Eight new package tests and two lab tests passed: **10 tests total**. No full manager/game suite
was rerun; tooling code was not changed. The two lab checks exercise real package registration
with options, and fourteen short headless modes with complete reserved-state round-trips.

| Package | Focused evidence |
|---|---|
| fuel | Exact total token conservation over 96 finite turns; spent state appears; restored next state agrees; corrupt domain refuses transactionally |
| cyclic | Pulse propagation and refractory behavior; restored next state agrees; corrupt state refuses transactionally |
| loops | Twelve plaquette phases preserve discrete divergence; reversed phase order restores exact state; persistence retains currents |
| spin | Eight sweeps preserve defined energy 62; inverse sweeps restore every spin/reservoir bit |
| symplectic | Thirty-two modular steps preserve summed impulse modulo 65,536; exact inverse recovers every integer |
| bonds | Shared-node relaxation and reststrain memory; exact energy 1024 = 828 heat + 196 strain after six steps |
| cooperative | Every single move in both directions loses 362,570 fit units; a double gains 1,372,012; exact occurrence multiset/potential change; queue round-trip |
| permutation | Six admitted geometric phases reverse exactly in a sparse map; occurrence conserved; model payload round-trip |
| lab | Fourteen isolated provider selections including off; thirty-two requested host ticks per mode; complete model-record equality after reload |

The arithmetic values above describe fixed small constructions, not thermodynamic measurements.
Small-format demonstrations are not proofs of universal computation, long-term stability,
scalability, protected winding, quantum behavior, or vector mechanics.

The raw final headless samples are [managed-lab-smoke.jsonl](managed-lab-smoke.jsonl). They include
all thirteen models and off. Sampling every eight ticks produces 70 JSON records. The finite
fuel and cyclic examples stop early; continuously active models accept all requested phases.
Their `operations`/`pending` units differ, so comparing those counters as CPU costs is invalid.

## Not yet evaluated

No GPU gameplay, player usability session, field rendering, massive-world scaling, equilibrium
statistics, full save-envelope stress test, cross-chart port transformation, or multiplayer
field/impulse test was performed. The manager's complete tooling CI and REUSE lint were not
run; package checks validated declared licences and required full-text paths. Frozen coefficients,
bounded patch initialization and arbitrary lattice orientation remain experiment limitations.
Meaningful gameplay and performance measurements must precede any final model choice.
