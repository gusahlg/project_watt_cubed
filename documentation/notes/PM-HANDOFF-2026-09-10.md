# PM handoff — 2026-09-10 (day 4): the Emergent Material Model, worldgen v2 (paused), perf, quality

Release: v1.0.0 = main a571ae7 (the emergent material model), against voxel-engine v1.0.0 (f788564);
evening small release: v1.1.0 = main 86a113b (tag v1.1.0) against voxel-engine v1.1.0 (05ba8cb; Cargo ^1.1.0, flake rev pinned).

## What changed (the model of matter)
The user's "Emergent Material Model" spec replaced the authored element/composition/reaction stack:
- `crates/material` (PM-written): `Element([u8;4])`, `Configuration` (ordered, multiplicity kept,
  canonical encoding), `Law` as a value (stamp v1: band-limited per-axis response curve — dead zone
  ≤ 10, repulsion 10..22, rest band 22..26, attraction 26..64, inert beyond; Q4 mixing; event
  strengths; probe elements + thresholds; visual seed), `interact`/`interact_many` (pairwise influence,
  bounded aggregate, order-independent), `observe` (probe responses → solid/liquid/transparency/
  emission/hardness/friction/acoustic), `visual` (4-D value noise over the lattice → render descriptor).
- `src/block/registry.rs` (PM-written): dynamic intern table, hot SoA tables from `observe`, render
  descriptors (4-4-4 colour bins; layer = descriptor id ≤ 16 384 while materials go to 65 535), labels,
  `spec`/`parse_spec` text form (`air` | `c:<hex>`).
- `src/block/regions.rs` (grok): worldgen regions derived from the law by observation predicates and
  rest-stability under Collision; families of a centre + variants within spread 8.
- `src/world/placement.rs`: rules name region labels; `WORLDGEN_VERSION` 4; ores `[rock, guest]`.
- `src/sim/reactions.rs` (PM-written) + wiring (M2f): gameplay events only (place → NewContact,
  break → ExternallyChanged ×6, mod hook); Jacobi generations at the sim tick; server authority.
- Break yields the configuration; stash/pouch hold `(BlockId, count)`; natural crafting deleted
  (workbench = task M2d); save v8 (law stamp in the header; v7 loads with legacy specs → air + one
  notice); protocol v10 (`Welcome` carries the stamp; fingerprint folds WORLDGEN_VERSION + law +
  regions); `inspect` prints elements/label/readings/descriptor.
- `crates/material-lab` (grok): scorecard (similarity, determinism, fixed points, cascades,
  proliferation, families, observations), `find-regions`, `sweep`. Law v1 scorecard: similarity max 2,
  fixed points 58% self / 29% neighbour, cascades quiescent 36% within 64 generations (WARN),
  proliferation sub-linear with quantum 4, families 5614 (WARN band 8-200), liquids 4.4%.
- Appearance (M2b): `BlockAppearance` seam in core with a flat-colour fallback; `ProceduralTexturesMod`
  (Essentials; the CPU 16×16 generator over a `Visual`: two colours, frequency, roughness, alpha, glow;
  knobs grain/contrast) and `GpuMaterialsMod` (off by default; engine `MaterialDesc` SSBO, procedural in
  the fragment shader).
- Workbench (M2d): origin/target slots, touch/strike/shake × repeat → `interact`; result interned, one
  target unit consumed; journal of discovered procedures with `/name`; multiplayer `Craft`/`CraftResult`
  evaluated by the server; inventory shows swatch + "{region}-like" display names.
- Regions (M2c): rare-first search, distinct centres (L1 ≥ 48), axis-inert pairs, 3 Collision-stable
  strata per region (geology picks a stratum per 64 m cell), ores `[rock, guest]` with guests near in
  resource space when shallow and far when deep; lamp/glass guests glow / see through. Family spread
  settled at 1 (8 and 4 failed cross-region rest).
- Review fixes: M2a2 (peek_file, Collision rest, legacy holdings notice, spread truth, compile caps,
  MAX_SPECS 65 535, restored tests, vertex pin row), M2f2 (server broadcasts every committed mutation as
  Snapshot batches; content hash only on the sync edit path over packed vertex bytes; visibility flip
  on skipped remesh; one column evaluation per server probe; restored assertions).

## Numbers
Release build of v1.0.0 (pm/day3 94e525f) at the user's settings, windowed 1542×1408, RTX 3070:
- bench seed 42: avg 2797 fps, p1 1082 fps, p99 0.92 ms, 0 hitches ≥ 33 ms, ready 0.86 s, RSS 308 MB
  (v0.3.3 main: avg ≈ 2.4-2.7k fps, ready 0.55 s — the model costs nothing per frame; ready +0.3 s is the
  first region compile, M3b item 3).
- stress (settle after stop): r6_64mps 0.08 s, r6_200mps 0.74 s, r20_200mps 2.45 s
  (v0.3.3 main: 0.92 / 2.60 / 12.38 s).
- suite: 808 tests, 19 ignored, green locally and on louise-pc (clean mirror, engine 1.0.0, 79.6 s); build warnings 68 (main had 4; task C1 sweeps them — the
  architecture audit made modules `pub(crate)`, which exposed pre-existing dead code).
- lab (law v1): similarity max 2 PASS; fixed points 58 % / 29 % PASS; cascades quiescent 36 % WARN;
  families 5614 WARN; liquids 4.4 %.
- Task 71 (staging-ring mesh uploads) vs the v1.0.0 binaries, same settings: static fps equal within
  noise (2770/2721 base vs 2756/2699 branch); flight p1 686 → 768 fps, p99 1.46 → 1.30 ms, max 8.1 → 5.0 ms;
  stress settle equal (0.08 / 0.76 / 2.52 vs 0.08 / 0.74 / 2.63 s).
- gpu_materials A/B by the engine session on louise-pc (RTX 4060, Default preset): GPU procedural
  path 8.4k fps vs CPU textures 16.4k fps (opaque pass 0.03 → 0.08 ms) — the fbm+grain pattern as
  specified is ~300 fragment instructions. Stays OFF by default; the engine cheapens it tomorrow
  (noise-texture taps, two octaves).
- M3b-lite: region compile was observation-bound (10 labels × 40 000 candidates, none reaching
  score 0); probe early-out + `element_changes` → cold 87.5 → 37.4 ms, warm 155 ns (OnceLock), no disk
  cache needed; centres bit-identical. One near-duplicate merged (`ui::hud_text`); name grep clean
  (placement names regions, `SoundClass::as_str` stems derive from observation); the quiet-frame
  zero-alloc guard holds with the scheduler installed.
- C1 dead-code sweep: 68 → 0 warnings, −209/+125 lines; every "lost feature" candidate was already
  unused or test-only on main (verified per item), so nothing was restored; `Precip::{Rain,Snow}` and the
  diffusion-v2 channel constants keep `#[allow(dead_code)]` with their named follow-ups.
- R1a (evening review cut): a non-v0 law is refused (Result, `SaveError::CannotHost`, handshake
  refusal) instead of panicking; regions stored once in `BlockRegistry`; descriptor cap = min(16 384,
  device layers) with the nearest fallback at the real cap (no more `% layer_cap` onto layer 0);
  strict spec grammar (`c:00` and `+` hex rejected).
- v1.1.0 final tree (c8d4052) at the user's settings: 829 tests green locally and on louise-pc
  (engine 1.1.0 mirror), zero build warnings, bench avg 2816 fps / p1 1108 / ready 0.56 s / 0 hitches,
  stress settle 0.08 / 0.76 / 2.71 s, spawn screenshot identical to v1.0.0. One stress run stalled
  ("entry STALLED", 4 of 512 sections) while three release builds and grok loaded the machine; two
  reruns on a quieter machine passed — noted as a watchdog flake, worth a look if it recurs quiet.

## Afternoon of 2026-09-10 (after v1.0.0): four reviews, fixes, the small release
Four read-only review agents went over v1.0.0 (crafting/net authority, placement/regions, scheduler
wiring, registry/save/protocol). Fixed the same afternoon (PM-written, branch pm/w-e): the server
resolved client specs by interning them (any client could fill the 65 535-id table; now lookup-first
with a reserve line `CLIENT_INTERN_RESERVE`, a ready gate and `CRAFT_RATE_LIMIT`); the workbench never
checked the origin was held (now both slots must be held, spent slots are pruned, a refund that no
longer fits is counted and shown); journal novelty keyed on registry novelty (now on the journal, fixed
points excluded); the scheduler truncated follow-ups by coordinate (now a FIFO-by-generation queue
with a capacity guard and a `dropped` gauge — a budget changes the pace, never the outcome); a
refused write was recorded as a commit (`CellStore::set_block -> Option`); single-player cascades
depended on which chunks were loaded (`World` now reads overlay → generator for unloaded cells);
snapshot cells replayed as player edits with a cue per cell (now `Incoming::Mutation`, silent); one
tick sent a cell twice; material names leaked worldgen labels ("rock", "rock-like") — the user asked
that everything match the model, so `display_name` is now words read off the observation ("glowing
clear hard solid") and labels stay internal. Grok task R1 (worktree b) took the registry/regions/
appearance seams: panic → refusal for a non-v0 law, regions computed once, fingerprint over every
compiled configuration, descriptor cap = the device's real cap (the `% layer_cap` aliasing), fallback
by look class, one alpha/glow convention, texture upload ordering (+ the user's chunk-border flicker
report), strict spec grammar.

Still open from the reviews (tomorrow, most need a protocol/save/worldgen version bump):
- Craft request correlation + explicit refusal (`req` on Craft/CraftResult, `CraftAck`) — protocol v11.
- A client message for mod-emitted material events; a mod/audio hook receiving reaction mutations.
- Id-based mutation wire format (`ConfigDefinition` once + `CellMutation{pos,id}`), a separate
  drop-oldest queue for reaction frames so a cascade cannot kick a slow client or a joiner.
- In-flight scheduler events are not saved (a world quit mid-cascade reloads unsettled).
- Classic height path uses `f32::powf` (libm-dependent) — integer spline; WORLDGEN_VERSION bump.
- Ore guests cluster into sibling variants; `ensure_guest` can evict its own pick; dead placement
  table rows (overhang/island interior use `stone_at`); five copies of the slice walk — all change
  generated matter → one WORLDGEN_VERSION bump for the lot.
- The law stamp is checked but never used (`Law::v0()` hardcoded at load/handshake/server).
- Autosave spec-strings the whole palette per save; the server hex-decodes per cell read in the
  reaction tick (store `BlockId` in `Cell`).
- Structure: `WorkbenchApply` value type instead of 6-14-parameter functions; one hex codec on
  `Encoding`; `CellStore` test map shared; `ElementStash` rename; the chunk-border flicker if R1 did
  not find it.

## Queue for tomorrow (directives in ~/pm-tools/tasks, effort in effort.txt)
1. **R1 remainder** (`R1-review-fixes-registry-regions.md` items 3, 5, 6, 7 — fingerprint over every
   compiled configuration, fallback by look class, one alpha/glow convention in the seam, texture
   upload ordering + the user's chunk-border flicker report). If R1a did not land tonight, run R1 whole.
2. **Law change** from the lab: M1c's proposed winner (stamp in `logs/M1c-law-tuning.log`, `lab explain`
   prints its curve) is 100 % quiescent on the reduced card but has no family target; run the FULL
   scorecard on it, decide, and if adopted re-seed regions + re-derive the byte pins (print-then-patch),
   bump the law version. Visual tuning goes with it (spread → frequency saturates for two-element
   configurations; family spread settled at 1).
3. **Protocol v11** (`Craft`/`CraftResult` carry `req`, `CraftAck`, a client `MaterialEvent` message for
   mod machines, id-based mutation frames `ConfigDefinition` + `CellMutation{pos,id}`, a drop-oldest
   reaction queue that never kicks a joiner) and **save v9** (pending scheduler events persisted).
4. **Worldgen** (one WORLDGEN_VERSION bump for the lot): integer spline instead of `f32::powf` in the
   Classic height path, ore guests by region distinctness, `ensure_guest` single pass, dead placement
   rows, one `Resolved::roll`; then W3 (erosion/hydrology/biomes on regions), W4-W6, W7 (GPU diffusion
   pass through the engine's ComputeLane).
5. **Structure**: `WorkbenchApply` value type, one hex codec on `Encoding`, `BlockId` in the server's
   `Cell` (no hex per cell read), spec strings only for ids present at autosave, shared `CellStore` test
   map, `ElementStash` rename, 16b (world splits) when `src/world` is quiet, M3b items 1-2 (multiplayer
   end-to-end and save-with-reactions test suites), the law stamp actually used at load/handshake.
6. **Engine-side** (voxel-engine session): cheap gpu_materials pattern (noise-texture taps, two
   octaves), then re-measure the A/B; Hi-Z parked; task 75 (CPU occlusion removal) waits on it.

## Lessons (added today)
- Never `git add -A && merge --continue` without checking EVERY conflicted file (a conflicted
  pipeline.rs with markers was committed once; fixed the same hour).
- Grep pipelines swallow cargo's exit status: check the result line, not `&&`.
- Regions are a pure function of the law: any law/stamp change reshuffles the world → byte pins are
  re-derived once per law change (print-then-patch script in the scratchpad).
- Law tuning by measurement: the first curve had no stable families; the second oscillated at the
  zero crossing (the lab's cascade metric caught it); the flat rest band fixed both.
