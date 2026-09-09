# Handoff — PM session 2026-09-08 (wrapped up at the user's request; continue tomorrow)

## Branches and worktrees (all based on main 58ffbcd + docs commit f590a3e)
| Worktree | Branch | Commits on top of f590a3e |
|---|---|---|
| `/home/gusahlg/repos/project_watt_cubed-pm` | `pm/cleanup-2026-09-08` | d6a5901 task 03 (500 tests) · 109d703 task 04 (501) · e024d4e **WIP** task 05 |
| `…-pm-a` | `pm/w-a` | bb0df19 task 10 (500) · 23451f7 task 28 crash fix (504) · 618faca **WIP** task 29 |
| `…-pm-b` | `pm/w-b` | 27ceb39 task 08 (499) · 7183446 **WIP** task 11 |
| `…-pm-c` | `pm/w-c` | 1e508ef tasks 21+20 (grok: 523 green; my re-run was killed → flagged) · b9c916c task 23 (531 green, confirms the tree) · 996db6d **WIP** task 06 |

WIP commits are unverified partial grok work; their task files carry a "Resume note".

## Task directives (scratchpad/tasks/)
Done: 00 01 02(landed in main by the user) 03 04 08 10 20 21 23 28.
Interrupted (resume first): 05 (main worktree), 29 (a), 11 (b), 06 (c).
Queued, not started: main → 12 13 09 24 27; a → 18 36; b → 25; c → 33 35 34 07 26 37 38.
Pending the user's go-ahead (game feel): tasks/pending-ask/30 mining time, 31 friction, 32 block health,
39 render-scale+TAA presets (wait for engine temporal upsampling).

## How to resume (two pipelines max on this 16 GB box — four got OOM-killed)
- `run_queue2.sh <tasks…>` in the main worktree, `PM_SUFFIX=a|b|c run_queue2.sh …` for the others.
  Each task: grok headless (`run_grok_task.sh`), then an independent `cargo test --lib` and a commit
  (`commit_task.sh`). Never edit the runner scripts while a run is active (bash reads them lazily).
- Every cargo call goes through `quiet.sh` (waits for sibling benchmarks, `nice 19`, `-j 4`, thin LTO).
- Suggested order tomorrow: 1) `PM_SUFFIX=a run_queue2.sh 29-no-entry-stalls` and `PM_SUFFIX=c
  run_queue2.sh 06-mod-seam-small 33-… 35-… 34-… 07-… 26-… 37-… 38-…` (two pipelines);
  2) then main `05 12 13 09 24 27`, a `18 36`, b `11 25`;
  3) merge a/b/c into `pm/cleanup-2026-09-08` (`merge_wt.sh a|b|c`; expect conflicts in
  `world/streaming.rs` between task 04 and 28 and in `mods/*` — a grok task with the conflict list is
  the cheapest way); 4) last on the merged branch: 15 dedup, 16 file splits, 17 comment trim;
  5) benchmark at the user's settings (RD20/V10) against a 58ffbcd baseline (game window appears).

## Measured so far
- Admission pass at 20k seeds: 2300 → 1420 µs (task 03); ring buckets (27) should take it under 150 µs.
- Near mesher: 52.8k → 62.3k jobs/s, bit-identical (08). Classic generation 0.92 → 0.88 ms/column (10):
  the fill is the cost, hence task 36.
- Baselines from the sibling session's logs (RTX 3070, user's settings RD20/V10/8×MSAA/200% scale):
  90 fps, 11.2 ms/frame GPU-bound (opaque 5.0, TAA resolve 4.3, sky 1.3); entry phase 21-24 ms/frame on
  the main thread (occlusion 13-18 → fixed in main by the user; mesh/light admission 2-3 each → 03/27;
  degraded remesh tax → 04); 816 MB RSS (light grids → 05).

## Constraints learned
- The machine hosts a second Claude session (voxel-engine); coordinate via SendMessage. It benchmarks
  with the main checkout's `target/release` binary and copies under `~/repos/.wt/`.
- Memory: 16 GB; the harness kills background tasks when the box runs low. Two pipelines at `-j 4`.
- Disk: ~42 GB free after the sibling pruned; the user's main checkout target is 19 GB (theirs).
- The user harvests results into main themselves (PR #8 contained tasks 00/01/02 verbatim).
