# PM handoff — 2026-09-09 (day 2 of the grok-driven cleanup/perf session)

Branch to merge: `pm/cleanup-2026-09-08` (worktree `../project_watt_cubed-pm`), tip: 57ce8f8 (released as v0.3.0 and merged into main the same evening).
Everything below is on that branch after tonight's merges of `pm/w-b` and `pm/w-c` (`pm/w-a` was merged
yesterday). Merge into main with `git merge pm/cleanup-2026-09-08` (no-ff). Save format is v7, protocol
v9 — old v6 saves load (the Inventory mod line migrates into the core stash).

## What landed today (all suites green, one commit per task, numbers from each task's RESULT)
| task | commit | what | measured |
|---|---|---|---|
| 09 | e00bdbb | light flood: column masks, emission LUT, flat-index relax, one shell walk | 34.3k → 37.6k settles/s (1.10×), bit-identical |
| 24 | bcb4901 | streaming bug hunt (9 leads, property tests) | 1 real claim bug fixed: stale light result landing on a regenerated chunk |
| 24b | 87d9a7c | degraded-chunk promotion wedge (from task 04) | r20 stress no longer STUCK; settle still slower than main → 24c |
| 24c | 64314d9 | post-flight settle: the adaptive pacer sat on its travel floor (effort 0.15, 2 workers); rest-time boost | settle after stop r6/64 0.27 s (main 0.92), r6/200 1.34 s (main 2.60), r20/200 3.78 s (main 12.38); flight p50 0.13/0.41/0.86 ms (main 1.5/4.1/6.8) |
| 56 | cbf85c4 | Minimum-frame fixed costs: silent audio commit, idle pump drain, locked router | director 55.9 ns → 1.5 ns skip; drain/router ~0 |
| 66 | ce65abf | settled light grids collapse to `Uniform` when all lumels match (once per flood); census counts uniform vs cells | headless census: all 820 uniform chunks hold 2 B light; RD 20 BENCH_MEM to be re-measured (was ~350 MB dense) |
| 07 | b3540ad | element stash owned by the core `Player` (save v7, v6 migrates) | invariant: Inventory mod off → mine → on → elements kept |
| 38 | 2e5d620 | `WATT_BENCH_SCREENSHOT`, `WATT_BENCH_YAW` | validated by the engine session on the 3070 |
| 40 | 12f42c5 | VRAM guard: estimate render targets, drop MSAA then scale per session, notice | the user's 6880×2880 8×MSAA case no longer panics |
| 26 | 4458f1d | VRS Auto/On/Off (Auto = on above 8 Mpx) | |
| 39 | 2e26b7c | resolution-aware Default: scale 0.8 + TAA above 1.5 Mpx (`DEFAULT_AUTO_RENDER_SCALE`) | engine data: Default 8.2k → ~10-11k at 1080p on a 4060 |
| 57 | a3df197 | mods screen: atomic config writes (shared helper), recompile notice (user wording), ellipsized details, coalesced saves, less cloning | |
| 58 | a5a0023 | review fixes: client teleport hold expires, per-checkout test data dir, `peek_file` short read, weak tests | |
| 59a | 1831bb1 | `Paths::checkout_dir()` from `WATT_CHECKOUT_DIR` (play.sh) | groundwork for build-time mod selection |
| 65 | 3f0b971 | BENCH `ready_s=`, `BENCH_MEM` census, schema 3, null rendered/coalesced fields | RD20/V10 on louise-pc: ready 8.7 s, 413 MB tracked, light grids ~350 MB |
| 69 | 127f846 | render-boundary audit report (`documentation/notes/RENDER-BOUNDARY-AUDIT-2026-09-09.md`) | |
| 76 | 2009575 | mesh bytes histogram (ignored test) | non-empty chunk: median 5.1 KB, p95 12.6 KB, max 28 KB |
| 15 | 22b4ecb | dead code + shared helpers (`world/mesh/face.rs` for both meshers, `math::smooth`/`lerp`, `hash::splitmix_next`), test-only fns gated; 56 files, −100 lines net | behaviour note: the previously dead `shake` setting is now wired (trauma on break/landing; default unchanged) |
| 39b | 57ce8f8 | resolution-aware Default made inert (`DEFAULT_AUTO_RENDER_SCALE` = 1.0) until the engine's TAAU is in main; TAA forcing fixed in the effective render config | louise-pc showed −15% at Default with 0.8 and no TAA on engine main |

Yesterday's tasks on b/c (11, 25, 49, 54, 46, 41, 44, 06, 33, 35, 34, 21, 20, 23, 08, 03, 04, 05, 12, 13, 28, 29, 18, 36, 10) are in the same branch; see `PM-HANDOFF-2026-09-08.md`.

## Benchmarks (branch tip vs main 58ffbcd, same engine e3adfc7)
louise-pc (RTX 4060, 1920×1080 fullscreen, engine main e3adfc7, branch cf39f09 = merged, before 15/39b):
| run | main | branch |
|---|---|---|
| Minimum | 41.7k / 40.6k fps | 40.9k / 40.0k fps (−2%, p1 lower) |
| Fast | 18.8k / 19.2k | 18.8k / 18.8k |
| Default | 6396 / 6416 | 5448 / 5461 (−15%: auto scale 0.8 applied WITHOUT TAA on this engine → task 39b makes it inert) |
| custom RD20/V10 | 1506 fps, RSS 826 MB | 3892 fps, RSS 390 MB, ready 2.9 s (light 30k uniform / 12k dense) |
| stress flight p50 (64/200/200 m/s at r6/r6/r20) | 1.00 / 2.14 / 3.15 ms | 0.23 / 0.62 / 1.98 ms |
| stress settle after stop | 0.19 / 2.44 / 11.62 s | 0.27 / 1.39 / 2.27 s |

This box (RTX 3070, windowed 1542×1408 copy of the user's settings, engine main e3adfc7, final tip 57ce8f8), two passes:
| run | main | branch |
|---|---|---|
| user settings (RD 6, scale 2, 8× MSAA, all lanes) | 476 / 475 fps, RSS 198-200 MB | 474 / 480 fps, RSS 169-183 MB (GPU-bound: 2.3 ms GPU per frame) |
| Minimum | 48.6k / 48.9k | 47.5k / 48.5k |
| Fast | 23.5k / 23.8k | 23.5k / 23.4k |
| stress settle after stop (r6/64, r6/200, r20/200) | 0.79 / 2.53 / 12.56 s | 0.25 / 1.38 / 2.43 s |
Flight profiles at 30 m/s on the branch (profiler on): Fast 14.1k fps, `cpu 0.02 ms (submit 0.02) | wait
0.11 (main 0.06 render 0.05) | gpu 0.06`; Default 4.6k fps, `cpu 0.05 (list 0.01 submit 0.05) | wait 0.34
(main 0.19 render 0.16) | gpu 0.19`, submit split `record 0.02 pack 0.02 submit 0.01`.
louise-pc on the final tip (after 39b): Minimum 39.8k/40.8k → 40.5k/40.6k (equal), Fast 19.1k/19.3k →
18.9k/18.8k, Default 6426/6424 → 7798/7904 (+22%: VRS Auto is off at 1080p, less main-thread work),
RD20/V10 1513 → 3920 fps with RSS 825 → 371 MB, ready 2.85 s; stress settle 0.22/2.53/11.54 → 0.27/1.39/2.27 s.


## Decisions taken today (each a one-line revert)
- Resolution-aware Default preset (task 39) and VRS Auto (26) — from the engine session's lane ablation:
  the opaque pass is bytes-per-pixel bound, so resolution is the only lever for the > 10k target.
- Face culling stays OFF by default (−5% at 1080p measured); task 37 dropped.
- Mod choices become a build input (design below); the mods screen says "recompile to apply".
- Task 27 (ring-bucketed worklists), 55 (LOD profile cache), 15/16/17 (dedup/splits/comment trim)
  moved to tomorrow.

## New programme: Mods v2 — "everything that is not a core mechanic is a mod"
Design: `documentation/notes/PM-MODS-V2-DESIGN-2026-09-09.md`. Directives ready in the scratchpad:
59 (build-time selection: tracked `mods.toml`, `build.rs` codegen, `cfg(mod_*)`, generated registry,
runtime enabled state removed), 60 (worldgen mod-only: Void fallback, classic + diffusion → mods, string
ids, save v8, protocol v10), 61 (block appearance trait + procedural textures mod), 62 (sounds/music
content mod), 63 (minimap + presence visuals mods), 64 (optional static fan-out), 67 (server holdings
ledger). Order: 59 → 60 → 61 → 62/63 → 67.

## Render boundary (joint with the engine session; audit on b, task 69)
Engine (their branch, shas per item): direction-major MeshData (done: 0148e95/25190e3 on wt/meshdata),
worker-side staging ring, Hi-Z occlusion (PARKED: 0.06 ms/frame pyramid for ~15% culled draws in a pixel-bound pass), `gpu_caps()`/`estimate_render_targets()`,
setters no-op on equal input, minimap by value + subrect, `append_block_textures` without GPU idle.
Game tasks waiting on those: 70 (MeshData API: all `buckets()` users, `vertex_bytes()`, `vertices()`
is test-only), 71 (staging uploads, per-job fallback), 74 (caps query replaces the VRAM mirror), 77 (append_block_textures on palette growth),
65b (rendered/coalesced counters in BENCH). Each starts with a precondition
check on `../voxel-engine` main and stops cleanly if the API is not there yet.

## Tomorrow's queue (effort in `tasks/effort.txt`; "fast" = low/medium)
pm (streaming/perf, high effort): 78 (light-convergence re-meshes: measure, then fixpoint promotion —
audit A1), 24d (light seed amplification, 48 seeds/chunk at 200 m/s), 55 (LOD profile cache + audit
A4: core_center once per column, halo'd section mesher), 27 (sort_by_cached_key first, then ring
buckets), 79 (section upload bytes), 73 (nibble-packed light), 68 (zero-alloc quiet frame, on-change
pushes, name-tag cache), then 15/16/17 (dedup, splits, comment trim).
c (mods, medium/high): 59 → 60 → 61 → 62 → 63 → 67.
d (engine-branch worktree beside `pmeng/voxel-engine`, low): 70 → 74 → 77 → 71 → 65b as the engine
branches land (#20 blocktex 906fc5e, #21 staging: held by the engine session until its world-entry regression is fixed; use e3cef66 for 70/74 meanwhile).
The engine session's two-repo audit is at `/home/gusahlg/repos/.wt/boundary_audit_report_2026-09-09.md`
(their C7 question — which paths mutate voxels outside store_chunk/mesh_chunk — is still owed). Engine numbers from that audit: shadows-off lane bit lifts Fast 18.0k → 22.3k (3070) and 20.5k → 23.9k (4060); `gpu_load()` becomes opt-in.
Recreate the worktrees from the merged tip first: `reset_wts.sh`.

## Constraints and tooling (see also the memory files)
- 16 GB box shared with the engine session: ≤ 2-3 grok sessions, `-j 4`, thin LTO; `hold.sh stop|cont`
  pauses everything for their A/B runs; `~/.bench_pause` before any local game run.
- louise-pc: `rtest.sh`, `rbench.sh` (flock `~/bench/lock`, X11 `DISPLAY=:0`), `rstress.sh`,
  `verify_remote.sh`; full test build 3 min, release 2 min.
- Pipelines: `run_queue3.sh` (per-task effort), waiters with time guards, `chain_final*.sh` merges with
  grok conflict resolution (`tasks/merge-b.md`, `merge-c.md`), `commit_task.sh` independent suites.
- Pending the user's decision: `tasks/pending-ask/` (mining time, friction, block health).

## Engine-branch builds without touching the sibling's checkout
`/home/gusahlg/repos/pmeng-blocktex/voxel-engine` is a second independent clone at wt/blocktex 906fc5e (PR #20) for task 77 (its game worktree goes beside it). `/home/gusahlg/repos/pmeng/voxel-engine` is an independent clone at wt/meshdata e3cef66 — move it to wt/staging 8c911aa (PR #21: MeshData + caps + setters + minimap + staging) once the engine session confirms; wt/blocktex 906fc5e is PR #20 (40+ commits past
main's e3adfc7: direction-major MeshData, gpu_caps/estimate_render_targets, setter no-ops, minimap by
value/subrect). Put an engine-dependent game worktree BESIDE it (`pmeng/project_watt_cubed-pm-d`, so the
`../voxel-engine` path dep resolves to the clone) and run tasks 70/74 (then 71/65b as they land) with
`PM_WT=/home/gusahlg/repos/pmeng/project_watt_cubed-pm-d run_queue4.sh 70-meshdata-direction-major …`
(`run_queue4.sh`/`run_grok_task4.sh` take the explicit path; the rules file is rewritten for it).
