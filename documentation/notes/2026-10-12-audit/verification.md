<!-- SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors -->
<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# Verification, coverage and limits

## Snapshot and method

Review date: 2026-10-12. Rust: **1.98.1**, x86_64-unknown-linux-gnu, LLVM 21.1.8.
The [index](README.md) records game/engine/package-manager commit identities. Both sibling
repositories had clean status at the end of the audit; the game already had uncommitted
quadtree/probe changes, which remain in place.

This is a source audit with focused CPU reproductions and existing tests. The investigation
followed architectural invariants through producers and consumers: generated versus edited
state, physical versus storage coordinates, policy versus commit, request versus confirmation,
provider versus cache identity, and snapshot versus durable recovery. Tests assert the current
bad behavior in a disposable copy rather than changing the real runtime to make tests pass.

## Test results

| Check | Result | Interpretation |
| --- | --- | --- |
| Initial `cargo test --workspace --lib --offline` | Build stopped at missing `alsa.pc` | Environment issue; no product failure inferred |
| Same command with installed ALSA/D-Bus pkg-config metadata | `field`: 7 passed; `material`: 18 passed, 1 ignored; game: 1083 passed, 149 failed, 70 ignored | Game's 149 failures split into **138 socket permission errors** and **11 LOD bound panics** |
| Existing game test binary, `net::`, loopback allowed, 4 threads | 271 passed, 1 failed, 16 ignored | Remaining failure is the peer turn-interpolation check |
| That interpolation test alone, loopback allowed | 0 passed, 1 failed | Reproduced outside the broader suite; exact packet-level cause not instrumented |
| Existing game test binary, `app::`, loopback allowed | 24 passed, 2 ignored | Earlier app host/join socket failures are resolved |
| Existing game test binary, `game::sync::`, loopback allowed | 5 passed | Reconciliation integration checks pass under the required environment |
| Existing game test binary, `game::draw::`, loopback allowed | 6 passed | Earlier peer-draw socket failure is resolved |
| `cargo test --offline -p pwc-mod-api --lib` with native metadata | 3 passed | API helper tests pass; workspace's earlier failure had prevented this final execution |
| Existing game test binary, `far_face_` group | 5 passed, 4 failed | Failures share `streaming/far.rs:416`; counts 1025, 1077, 1103, 1810 |
| Existing game test binary, exact altitude-frontier test | 0 passed, 1 failed | Same guard, 1077 leaves |
| Final isolated audit probes | **17 passed, 0 failed** | 16 defect-behavior probes and 1 explicitly synthetic stale-tool ordering risk |

The broad game suite took 733.29 seconds in the unoptimized test profile. That duration is
not a release performance result. The 17 probes executed in 4.89 seconds after compilation.
The network group took 95.79 seconds. Results should not be added together as unique-test
totals: filtered reruns intentionally overlap the broad suite.

The remaining established existing-suite failures are the **11 LOD tests** plus the separately
reproduced turn-interpolation test. Environment failures were inspected by their error text,
not inferred merely because a test involved networking. No unexplained failure was silently
reclassified as a sandbox problem.

One automatic approval review for a local test rerun timed out. Its permitted single retry
succeeded, so no verification remains blocked by that review.

## Reproduction commands

The normal reproducible environment is the project's Nix dev shell:

```sh
nix develop
cargo test --workspace --lib --offline
cargo test -p pwc-mod-api --lib --offline
python3 notes/2026-10-12-audit/reproductions/run.py
```

In this audit, the dev shell was not entered. Already-installed Nix ALSA and D-Bus `.pc` files
were supplied through `PKG_CONFIG_PATH`, allowing native dependencies to compile offline.
No dependency was downloaded. The filtered existing-suite reruns used the already built
test executable at `target/debug/deps/project_watt_cubed-b6594d53e40f78b1`, for example:

```sh
target/debug/deps/project_watt_cubed-b6594d53e40f78b1 net:: --test-threads=4
target/debug/deps/project_watt_cubed-b6594d53e40f78b1 far_face_ --nocapture
target/debug/deps/project_watt_cubed-b6594d53e40f78b1 \
  --exact world::tests::desired_frontier_responds_to_altitude --nocapture
target/debug/deps/project_watt_cubed-b6594d53e40f78b1 \
  --exact net::mp::presence::a_peer_turning_round_is_drawn_turning_the_short_way --nocapture
```

That executable hash is specific to this build. After a new build use Cargo's equivalent
filters or identify its emitted executable. Local socket tests need permission to bind loopback.

`reproductions/run.py` copies `src`, `crates`, `assets` and Cargo manifests into a fresh
`/tmp/pwc-audit-*` directory, symlinks the existing sibling engine and appends child test modules
only there. The target cache defaults to the real repo's ignored `target` directory, avoiding
fresh dependency downloads. It leaves the temporary copy available for inspection; `--stage-only`
prints that copy's path without running Cargo. The script's source fixtures are the durable
reproduction artifact. [probe-results.txt](evidence/probe-results.txt) records the compact final
observations; generated executable paths/PIDs are omitted from that archive.

The deliberate debug panic caught by the descriptor-cap probe is expected evidence, not a
failure of the final probe suite. These tests are **observed-behavior tests**, so they should
eventually fail when production defects are fixed. Convert them into desired-behavior
regression tests during each repair.

## Coverage map

| Area | Material consulted and paths followed | Depth/limit |
| --- | --- | --- |
| Vision and policies | README, philosophy, IDEA, mods, material model, performance, open issues, audio/crate boundaries, space/warp/worldgen/emergent design notes, reaction/physics guides, contribution/mod/licensing context | Architecture/specification review; historical plans separated from current code |
| Package ecosystem | Sibling architecture/build specifications, first-party package descriptions, appearance/naming implementations and builder/server entry-point inventory | Integration/ownership review; package-manager suite not run |
| Renderer | Sibling README/instructions, public renderer contracts consumed by world/streaming, packed layer cap/cage/resident integration | Boundary review, not an exhaustive Vulkan/shader audit |
| Material/field | Pure crate APIs, intern/observe/react integration, contact/budget semantics, exact-apron/field execution | Existing crate tests plus exhaustion/conservation probe |
| World | Storage/chunk/brick, generator contract, overlay/bulk ledger, query/seam, streaming/mesh/textures, rule/far-frontier/coverage, sampler and indexes | Cross-path inspection and existing tests; emphasis on correctness under changes |
| Space/gravity/mechanics | Atlas/storage/motion integration, physical coordinates, gravity ledger/oracle direction, genesis/warp design and current generator identities | Architecture and critical consumers; no independent numerical scientific audit |
| Mods/API/input/UI | Registration, package identity, arbitration/suspension, frame/update contexts, option persistence, action binding/capacity, screen/HUD vocabularies | Provider/state/capacity probes; API surface assessment |
| Save | Codec invariants, snapshot/rebuild, unknown state, autosave dirty tracking, slot discovery, atomic rotation and server store | Failure injection and restore probes; no destructive tests against user worlds |
| Network | Protocol/content identity, client request/result/interpolation paths, server state/edit/tool/movement/budgets/hooks/fanout/tick/save, integration test fixtures | Broad CPU/loopback tests and direct-handler probes |
| Audio | Service/frame/runtime/capture boundaries, realtime ownership, pure acoustic and recording-test interfaces | Targeted inspection plus included existing tests; no real microphone/output-device tests |
| Build/release | Cargo features/profiles, sibling path dependency, Nix dependency/package checks, benchmark contract, missing CI and golden distribution | No distribution/deployment or release build benchmark |

## What is not established

- This is not every-line verification of this large codebase, its histories, assets or sibling
  repositories. No subagents were used and no external web research was needed for the local
  source findings.
- No interactive game/GPU run, screenshot/golden comparison, Vulkan validation-layer run, live
  audio hardware test or cross-platform determinism run was performed.
- No new release FPS, presented-frame, VRAM or latency improvement was measured. Optimization
  proposals include experiments precisely because source inspection cannot substantiate speedups.
- The LOD failure's attribution to committed code versus the two pre-existing edits is unknown.
  A clean-HEAD comparison is an explicit follow-up, not an unstated assumption.
- The synthetic stale-tool-result order has not been reproduced through the current ordered
  transport. It stays outside the 18 confirmed/actionable grouped reports.
- The test evidence proves the reported boundary behavior. Broader claims such as an Internet
  denial-of-service magnitude, exact user frequency, or the packet cause of the live interpolation
  failure require the additional experiments identified in the reports.

## Change scope

All intentional repository additions are under `notes/2026-10-12-audit/`. Production source,
Cargo manifests/lock, shaders, package-manager code and renderer code were not edited. Existing
user source edits were preserved. Test build artifacts remain ignored; source fixtures and
documentation are the reviewable deliverable.
