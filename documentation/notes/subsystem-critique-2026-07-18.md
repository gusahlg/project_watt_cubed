# Subsystem critique notes — 2026-07-18

> **Implementation status (same day, second pass):** items marked ✅ landed;
> ❌ = evaluated and deliberately rejected (reason inline); everything else
> remains open. The big ones that landed: fused far-section extract+mesh with
> a byte-parity oracle (and the whole RLE section storage becoming
> test-only oracle machinery as a consequence); the section edit-chunk index +
> exact-dirty incremental overlay + event-driven covering rebuild; the
> pump/topology streaming split (results/uploads/edit-remesh land every frame
> at any `stream_hz`); pressure-scaled section upload budget; direct-sampling
> `resample_cell` (no more full extract per edited cell) with a shared
> `MipFold`; `Arc<str>` names/specs/chat across the whole protocol (server
> snapshot now ships interned Arcs); the packed one-byte-per-block
> property table behind typed accessors; once-per-frame modifier sampling in
> the router; `percent_bar!`; scheduler Hz/OnRevision/floor/enable tests;
> `natural_sorted`; the engine's broken TAA test removed (its 107 tests run
> again); and `world/mod.rs` tests split out (2554 → 1566 lines).

Raw notes from a full-codebase pass, taken right after re-implementing the PR #5
performance work on post-refactor main. NOT a plan. Opinions are deliberately
allowed to disagree with each other — mark ⚔ where two takes conflict so the
eventual compromise is chosen consciously. Confidence tags: [✓]=verified in
code this session, [~]=read but not re-verified, [?]=reported by a reviewer
agent, spot-check before acting.

File-size ground truth (today): `world/mod.rs` 2488, `settings.rs` 1845,
`game.rs` 1721, `net/server.rs` 1603, `harness/mod.rs` 1218, `quadtree.rs` 897,
`section.rs` 840, `light.rs` 767, `placement.rs` 758, `generation.rs` ~1300.

---

## 0. Executive shortlist (if only five things get done)

1. ✅ **Fused far-section extract+mesh** (section.rs → section/mesh.rs): the
   biggest single CPU win left in the worker pool. Today: generator →
   per-column dense scratch → RLE runs → brick slicing (thousands of transient
   Vecs) → then section/mesh.rs decodes the RLE *back* into CellRuns to mesh.
   Generate straight into one pooled dense quadrant buffer and mesh from it;
   keep the current path as the byte-parity oracle. [✓ per_brick at
   section.rs:290 allocates num_bricks×256 Vec headers per quadrant, ×4]
2. ✅ **heightmip `resample_cell` does a full `Section::extract` per edited cell**
   → quadratic in edits-per-section during edit bursts. Batch per section or
   maintain an incremental overlay. [? heightmip.rs:297–322 — verify, then fix]
3. ✅(perf half) **Finish the scheduler migration / split streaming into result-pump vs
   topology** — `World::pump` now runs every frame (results, uploads, edit
   remeshes); the full ordering-as-data scheduler migration remains open (streaming.rs stream() is still a hand-ordered serial mega-pass
   of 10 `run_manual` lanes). This is both an architecture and a perf item —
   at `stream_hz=15` finished worker results should still land every frame.
4. ✅ **`Arc<str>` names end-to-end** (protocol-wide, snapshot specs included) (client peer, server player, chat): the one
   remaining per-frame String clone in multiplayer (game.rs:1667 name clone per
   visible tagged peer) plus per-broadcast clones server-side. [✓]
5. **Engine P0s** (see §2): mailbox admission before composition, structurally
   absent lanes, draw-list clone in finish_frame. Nothing game-side can beat
   what those cost a 0.05 ms frame budget.

---

## 1. Overall organisation

- Module split is basically sane; no crate-level chaos. The pain is **five
  god-files** (world/mod.rs 2488, settings 1845, game 1721, server 1603,
  harness 1218) that each mix 3+ concerns. Everything else is pleasantly small.
- `harness/mod.rs` (1218 lines, golden-shot machinery) is `pub mod` in lib.rs
  with no feature gate [✓ lib.rs:37]. It compiles into the release binary.
  Fat LTO probably strips what's unreachable, but `Game::scripted`/DebugView
  keep entry points alive. Opinion: feature-gate it (`#[cfg(feature =
  "harness")]` default-on for tests, off for dist) or move to a separate bin.
  ⚔ Counter-opinion: the philosophy doc says a separate minimal build is only
  worth it if measured (icache/startup); harness code never runs in a normal
  frame, so this is hygiene, not perf. Do it only when touching harness anyway.
- `documentation/` + `issues/` are good. This notes file should get folded into
  issues/ once triaged.
- Tests live mostly inline (`mod tests`) — good for invariant proximity; the
  world tests make `world/mod.rs` even fatter though. Moving *only the tests*
  into `world/tests.rs` (same crate, `#[path]` include or child module) would
  cut mod.rs by several hundred lines without any architectural risk. Cheap.

### 1b. watt_cubed vs voxel_engine responsibility map

Verified state: dependency direction is clean (game → engine only; engine has
zero knowledge of game types) [?, agent-confirmed]. Type duplication is less
bad than feared: `Detail`, `Rev`, `MeshData`, `Color` are engine singletons;
game's `ident/` mostly wraps/re-exports rather than duplicates [~ — but
pyramid.rs uses `crate::ident::Detail` while world tests use
`voxel_engine::Detail`; if those are the same type re-exported, fine, but pick
ONE import path project-wide so grep tells the truth].

Things that arguably sit on the wrong side — with both directions argued:

- **`frame_snapshot.rs` lighting composition (game) vs `gate_uniforms` (engine)**.
  The lighting *policy* is split-brained: the game composes the packet
  (palette blends, overcast desaturation, fog density, ambient floor) and the
  engine then zeroes lanes per RenderFlags (stars gain, fog…). Two owners of
  "what does lighting look like when lane X is off".
  → Move composition INTO the engine (game hands a `SkyState{sun_dir, palette
  samples, weather}` desc): one owner, engine can specialize shaders per flag
  combo, stripped modes get structurally-absent lanes for free.
  ⚔ Opposite take: composition is *look/art*, and look belongs to the game;
  the engine should stay a dumb executor of a uniforms struct. Keep it
  game-side but delete the engine's gating (game composes exactly what should
  render; engine stops second-guessing). Either direction beats the split.
- **`profile::Meter` (engine) names game concepts** — `ListSky`,
  `StreamTiles`, `Physics`, `NetEvents` inside the renderer crate [?]. The
  engine should expose N generic meter slots or string keys; the game owns the
  labels. Small, uncontroversial.
- **Producer types (engine) / Scheduler (game)**: current split is
  deliberate (sched/mod.rs header explains Ctx names the app World). Agent
  verdict and mine agree: keep the split. The alternative — generic
  `Scheduler<W>` engine-side — buys nothing since there's exactly one consumer.
  If a second consumer (server-side scheduler?) appears, revisit.
- **Camera-relative rebase + LOD clip volume**: game computes `CoverageVolume`
  and eye; engine culls/rebases. Fine. Don't move.
- **`genconst::ANIM_PERIOD`** shared const engine-side, game math depends on it
  (anim UV wrap). Acceptable as a wire constant; document it as part of the
  uniforms ABI.
- **Resident-mesh patching** (`set_visible`/`set_style` per section from world
  streaming): this is the game driving engine retained state — correct
  direction post-refactor. But the *delta-gating* is duplicated: engine
  delta-gates per mesh AND the game keeps `last_style` per section
  (world/mod.rs SectionState) to avoid the walk. Two caches, one purpose.
  Opinion: keep the game-side one (avoids the walk entirely), delete or trust
  the engine one; note in the engine API contract who gates.
- **Engine P0 list** (verified by agent in engine code):
  - `finish_frame` DOES memcpy completed draw lists (Vec reuse but O(vertex
    bytes) copy) [?, engine frame path]. On a 0.05 ms budget this can dominate.
  - transparent sort per frame `vk/mod.rs:1216` O(n log n) — fine while
    n<1000, but opaque ordering should be cached/bucketed.
  - allocator `shrink` called every frame (amortized cheap, still a lock walk)
    — move to pressure/event-driven.
  - `begin_3d` rebuilds view/proj/frustum every accepted frame — cache by
    (camera bits, viewport). Game already caches Camera3D by angle bits, so
    the remaining cost is engine-side.
  - stripped lanes not structurally absent (shadows-off still costs PCF in
    shader; bloom resources alive). Needs pipeline variants/specialization.
  - **mailbox admission**: game pays full compose before the engine can drop
    the frame. Admission token API before composition = the single biggest
    "20k FPS" enabler. Game-side prep is done (compose is now cheap-ish and
    cache-heavy); the engine API is the blocker.
- **engine tests don't compile** (`vk/taa.rs:532` calls a `manifest()` that
  was never written) [✓ symptom, ? diagnosis]. Trivial: delete the aspirational
  test or stub it. Do this first — a red test target hides real regressions.

---

## 2. game.rs (1721)

- It's the app's second god-file: frame phases + console command glue + scene
  composition + HUD drawing + peer rendering + net event application + audio
  hand-off. The *phase* structure (net→input→overlay→motion→interact→stream→
  audio; then compose→scene→hud) is good and readable; the file is just too
  much. Split candidates that don't fight the borrow checker: `hud.rs`
  (hud_phase + refresh_hud_text + caches), `peers.rs` (peer_draws + PeerDraw +
  apply_net_events), `console_glue.rs` (submit_line). Keep update() itself here.
- Per-frame state is now cache-heavy (Memo × 6, RateGate × 5, scratch × 3) —
  the struct reads like a grab bag. Opinion: group into `struct FrameCaches`
  and `struct Clocks` sub-structs; pure readability, zero cost.
  ⚔ Counter: sub-structs mean `self.caches.camera` noise everywhere and Rust
  won't let two sub-structs borrow-split any better than fields do. Meh either
  way; do it only in the split-file refactor.
- `submit_line` clones the entire `Settings` (`before = settings.clone()`)
  per command — cold path, fine, but it's the pattern used in menus every
  frame too (menus.rs `let before = self.settings.clone()` each frame [~]).
  Settings is POD-ish (~200 B), so honestly fine. Leave it. (Noting because
  someone will "optimize" it into a diff system — don't, not worth it.)
- `peer_draws` still clones `peer.name` per visible tagged peer per frame
  (game.rs:1667) [✓]. With Arc<str> this becomes an atomic bump. Do together
  with server-side names (§13).
- `apply_net_events` does `save::parse_block` per remote edit — spec string →
  registry lookup per event. Fine at human edit rates; a join snapshot flood
  goes through Connection (client) though — check where snapshot edits are
  applied and whether each one re-parses a spec that repeats (specs are highly
  repetitive; an interning map per batch would collapse it). [~]
- Idea: `update()` takes 6 &mut params (eng, router, mods, settings, sound,
  audio) — a `struct AppCtx<'a>` would tame signatures. Cosmetic.
- Physics edge-latch logic in motion_phase and the mod edge-replay in
  interact_phase are subtle and ONLY tested indirectly. Deserve direct unit
  tests (feed synthetic FrameInputs at low hz, assert single fly toggle etc.).

## 3. sched/mod.rs

- `RateGate` vs `Interval` — two accumulator types in one module doing 90% the
  same thing [✓]. Interval = sticky-due + reset-on-act; RateGate = whole-step
  counting. Opinion: implement Interval as a thin wrapper over RateGate (or
  give RateGate a `due()/reset()` sticky mode) — one clock implementation.
  ⚔ Counter: Interval's clamp-to-period semantics ("fires the instant
  allowed") is genuinely different from step-counting; merging risks subtle
  autosave cadence changes for ~20 lines saved. Low value either way.
- `Registered` is duplicated for `producers` and `manual` lanes with fields
  (floor, hz_gate, stamp) meaningless for manual lanes. Opinion: manual lanes
  deserve their own slimmer type; the shared struct is a leftover.
- The starvation floor (`ticks_skipped >= floor`) has no test. The whole
  scheduler has only RateGate tests. Add: OnRevision gating fires once per rev;
  floor force-fires; set_enabled drops banked time.
- Long-term: the migration story (manual lanes → tick lanes) is written in
  comments but there's no tracking issue. The ordering constraints
  (dirty-remesh before draw-set, occlusion after loads) could be expressed as
  explicit `after(SourceId)` deps instead of prose + call order — that's the
  enabler for moving them onto `tick()` and then for the §7 pump/topology
  split. This is the right *next* architecture step for the whole streaming
  system, more valuable than any micro-opt in this file.

## 4. settings.rs (1845) + render_config + command/menu surfaces

- The descriptor-table design is genuinely good (single source, four surfaces
  fold over it). The cost is authoring: a Choice setting is ~30 hand-written
  lines of fn pointers (preset, hud_mode, msaa, max_fps, fov, scales are all
  bespoke). `rate_setting!`/`toggle_setting!`/`volume_bar!` cover the easy
  shapes. Next macro candidates: a `choice_setting!` (list + names + parse
  aliases) would collapse preset/hud_mode/msaa; a `percent_bar!` for
  render_scale/ui_scale/menu_scale/shake (4 near-identical ~30-line blocks).
  Could take the file down ~300 lines.
- `usage` strings hardcode ranges ("renderdist <0-20>") that can drift from
  the consts — I had to hand-edit one during the port [✓]. Options: (a) accept
  drift, it's help text; (b) generate at runtime via OnceLock<String> table;
  (c) make the menu render ranges from the clamp fns and drop them from usage.
  Mild preference for (c) eventually.
- `Settings` mixes persisted, env-only (cull_faces), and transient (muted)
  fields on one struct — documented, works, but the "absent from SETTINGS =
  not persisted" rule is implicit. A `#[non_persisted]`-style marker doesn't
  exist in the table design; fine, but keep the doc comments religious.
- `Setting::step/parse_human` now clone Settings for change detection (mark
  custom) [✓ mine]. Cold path; fine. Noting so nobody perf-panics later.
- menus.rs rebuilds all row Strings every menu frame (`s.show()` per row)
  [?, plausible]. It's a menu; 60 fps × 20 rows × format! is invisible.
  ⚔ But it's also trivially memoizable with the existing `Memo` keyed on
  (category, settings-clone-hash…) — actually settings has no cheap hash; skip
  it. Verdict: leave menus alone unless someone profiles menu jank.
- render_config.rs is clean now. `RenderConfig` duplicates ~20 bools that
  also exist on Settings; render_config() copies field-by-field [✓]. A macro
  could generate the mirror, but the two structs deliberately differ (lod
  levels live in both, occlusion policy differs) — hand-written mirror is
  honest. Leave.

## 5. world/mod.rs (2488) — the World god-struct

- ~60 fields across six concerns: near chunks, lighting, far sections,
  occlusion, textures/palette, job tracking [✓]. Every subsystem method takes
  `&mut self` on the whole thing; private field groups are only enforceable by
  `pub(in crate::world)` discipline.
- Restructure opinion A: split into `Near`, `Far`, `LightSys`, `Jobs`
  sub-structs; methods move to `impl Far { fn stream(&mut self, near: &Near…)`
  — makes cross-borrows explicit and lets tests build a Far without a world.
  ⚔ Opinion B: the refactor already tried subsystem separation via lanes and
  landed here because streaming genuinely touches everything at the sync
  point; a struct split will produce 5-arg methods and no perf. Prefer
  extracting *tests* (≈700 lines?) and the `SectionState`/draw plumbing into
  `section_state.rs`, keep World flat.
- **SoA observation**: far-field state is a "struct of maps" keyed by
  SectionPos: `sections`, `section_edit_rev`, `section_overlay`,
  `section_overlay_cache`, `dirty_sections`, plus visible/desired Vecs [✓].
  Five hash lookups of the same key in one pass are common. A `SectionTable`
  (one FastMap<SectionPos, SlotId> + parallel Vecs indexed by SlotId — proper
  SoA with stable slots) would turn repeated hashing into one lookup + array
  hits, and make iteration cache-linear. This is THE ECS-style win available
  in the world code. Medium effort; do together with the fused-mesh work
  since both touch section lifecycle.
- `FastHasher` — hand-rolled multiply hasher for coord keys; fine, but the
  same 3-line write_i32 trick is duplicated conceptually in hash.rs
  (splitmix). Not worth unifying (different contracts). Leave.
- Chunk map: `FastMap<Coord, Loaded>` where Loaded = {Arc<Chunk>, state, rev,
  connectivity, light}. Iterated fully in several passes (radius_shrunk scan,
  full_pass fresh scan, unload) [✓]. At view 20 that's ~1.6–13k entries —
  iteration is fine, but the *per-frame* `pending_*` sticky-gated design
  already keeps steady state O(1). Good. Don't SoA-ify the chunk map without a
  profile; the section table is the better target.

## 6. world/streaming.rs + pipeline.rs (workers)

- stream() is a serial 10-lane pass with load-bearing comment ordering [✓].
  The pump/topology split (results+uploads every frame; selection/admission at
  stream_hz or on events) is the highest-leverage architecture change in the
  world code. Prereq: express lane ordering as data (see §3).
- drain_results: budgeted uploads fine. Section upload budget 2/frame — at a
  ladder change the whole far field re-uploads at 2/frame = seconds of pop-in.
  Opinion: budget should be bytes-based, not count-based, and burst after
  `clear_section_lane` (the user just changed a setting; they expect work).
- pipeline near-queue: `pop` re-keys every entry against the live view per
  dequeue (linear scan) [✓]. Self-healing prioritization for free; O(n) per
  pop with n bounded by admission budgets. Fine now; if budgets grow, switch
  to a bucketed ring by distance. Note the threshold, don't fix.
- `Workers::default_threads` = min(3, cores-1) [✓]. On a 16-core box that
  leaves 12 cores idle during a spawn flood. ⚔ Deliberate ("chunk work is
  bursty"), but with the fused section path making far jobs cheaper, a higher
  cap (or scaling with far-queue depth) is worth a bench. Cheap experiment.
- Done channel: plain mpsc + `done_scratch` drain — good. `Done` ≤128 B assert
  survived the token additions [✓].
- Claim/token machinery now exists (epoch+token) — the `section_pending_claim`
  submit→claim handshake is a bit smelly (temporal coupling through a field,
  debug_assert guarding it) [✓ mine]. Cleaner: `submit` returns the token in
  the Job AND `admit` passes it to `claim` explicitly. Requires widening the
  StreamLane trait (claim(world, key, token_from_submit?)) — only Section uses
  it. Low priority, but the assert firing in a refactor is likely.
- `edits_in_footprint` iterates the ENTIRE edits map per section job submit
  [✓ streaming.rs:40-56]. With a big save (100k edits) and a far frontier of
  hundreds of sections, admission does hundreds × O(edits) scans. Fix: bucket
  edits by chunk column once (FastMap<(cx,cz), …> already keyed by Coord —
  derive a per-column index) or precompute per-section edit lists on edit.
  Real scalability item for edit-heavy saves.

## 7. world data: chunk.rs / brick.rs / section.rs / section/mesh.rs

- chunk.rs: palette-compressed chunk (u8 cells + BlockId palette ≤256,
  uniform fast path, GC on palette saturation) [?/~]. Design is right.
  Uniform→paletted promotion allocs 4KB once per edited chunk — fine.
- brick.rs: canonical RLE brick with cumulative column_ends — nice, tested.
- section.rs `extract`: the transient allocation storm [✓ quantified above]:
  per section job = 4 quadrants × (num_bricks × 256 Vec headers) + 1024 ×
  (rle_slice Vec + slice_into_bricks Vec). THEN section/mesh.rs decodes those
  RLE columns back into `Vec<CellRun>` per column (another ~1k Vecs) to mesh.
  The RLE round-trip exists only because Section is the storage format —
  but the mesh job doesn't NEED storage; it needs cells. Fused path: generator
  `lod_column` → pooled dense [BlockId; 32×32×N] per quadrant → mesh directly
  with O(1) neighbour reads; build the Section (storage) ONLY if something
  else consumes it (does anything? sections map stores Meshing→Ready with GPU
  handles; the extracted Section itself is dropped after meshing! [✓
  pipeline.rs run(): `let sec = Section::extract(...); build_section_mesh(&sec…)`
  — sec is temp]). So the entire RLE encode+decode per job is pure waste
  today. Strong candidate; keep extract() as the oracle for a parity test.
- section/mesh.rs: greedy mesher per 16³ block is solid; column_cells decode
  duplicates section.rs run-decoding logic (~50 lines) [?]. The fused path
  deletes both.
- ⚔ Contrarian note: far sections re-mesh rarely (ladder change, edits);
  spawn/ladder floods are the only time this matters. If the pump/topology
  split + upload bursting land first, maybe nobody notices extraction cost.
  Measure a ladder-change stall before/after to justify the fused work.

## 8. worldgen: generation.rs + placement.rs

- Post-port state is good (norm precompute, shared warps, identity-gamma
  bypass, inline octave columns, [Column;256]). Remaining:
  - `profile()` is still ~1 full evaluation per column with 8+ field samples;
    the columns of a chunk are evaluated independently — no cross-column
    coherence exploited. Batched/SIMD evaluation is the next tier but is
    gated HARD by bit-identity (reassociation changes floats). Only with a
    byte-parity oracle and probably a worldgen-version bump. Park it.
  - `lod_column` (LOD sampling) vs `profile` (full-res) — two sampling paths
    into the same fields; divergence risk is the classic LOD-seam bug source.
    They're tested for class parity (extraction_matches_the_generator_sweep)
    [✓ test exists]. Keep that test sacred.
- placement.rs compile(): startup-only, capped enumeration (ENUM_CAP 1024),
  auditable. Leave alone. The ENUM_CAP assert should print WHICH rules blew
  the cap (user-facing modding error). Cheap kindness.
- Registry rebuild on crafting: `refresh_tables` re-clones ALL hot tables per
  palette growth [✓ streaming.rs:1689]. With 16k palette that's ~100KB churn
  per crafted-new-block. Incremental append (push one entry per table) is
  easy since palette is append-only. Low frequency, medium value; do when
  touching registry.
- Hot-table layout: 4× `Vec<bool>` + Pass + u8 arrays. NOTE: Rust Vec<bool> is
  byte-per-bool, NOT bit-packed (an agent claimed otherwise — wrong). Real
  improvement: ONE `Vec<u8>` of packed property bits per block (solid|opaque|
  water|…) → mesher reads one byte per neighbour instead of hitting 2–3
  separate arrays. Halves cache lines touched in the face loop. Medium win,
  small effort; needs mesher call-site sweep. [my take, verified layout]

## 9. LOD selection: pyramid/quadtree/metric/coverage/heightmip/summary

- pyramid.rs: clean post-port (cached step, mult/compare bands).
- quadtree.rs `desired_sections`: O(Σ reach²) ≈ 100k+ section tests per
  frontier recompute at deep ladders [?]. Frontier retention (SectionFrontierKey)
  makes this cost only-on-move now [✓ mine]. Remaining idea: incremental
  frontier update on center delta instead of full rebuild — complexity not
  worth it until profiles say the rebuild-on-move hurts. ⚔ Don't.
- `coarsen_by_error` builds a FastMap per level per recompute [?]. Same
  amortization argument. Leave.
- coverage.rs: small and correct; O(cut) diffing. Leave.
- heightmip.rs: the resample_cell quadratic-per-edit issue (§0.2). Also
  `sample_section` vs `resample_cell` duplicate ~70 lines of column-fold
  logic [?] — unify with a fold callback when fixing the batching.
- metric.rs/summary.rs: good typing (EyeDist, HeightEnvelope). Model files.

## 10. lighting (light.rs 767)

- Flood is thread-pooled scratch, O(chunk) per settle, converges bounded [?].
  No action.
- Per-frame cost when ENABLED is the padded 18³ capture per mesh job (now
  pooled cross-thread [✓ mine]). P1 idea stands: uniform-lit chunks could skip
  the capture with `all-bright/all-dark` sentinel variants (analytic trivial
  grids already exist on the settle side — extend the idea to capture side).
- Model note: skylight ceiling intentionally ignores generated overhangs/
  islands (documented) — a known lighting wrongness accepted for perf.
  Fine; keep documented.

## 11. near meshing (mesh.rs / neighborhood.rs)

- Post-port state good (unlit path + parity test, shared pools, cap 32).
- `face_sample` recomputes the corner (eu,ev) selection per corner from
  `dir.corners` floats (`> 0.5` tests) [✓] — could be precomputed per-Dir
  const tables of i32 offsets. Micro (the mesher is per-job, not per-frame);
  only as drive-by.
- MASK_CAP slice mask [Option<FaceSample>; 256] per slice — 256×~14 B on
  stack per slice, fine.
- ⚔ Big-hammer idea: merge near-chunk mesher and far-section mesher into one
  generic greedy core (they share the algorithm shape, differ in sampling and
  attribute richness). Would be beautiful; likely a genericity tax
  (branches/inlining) in the hot loop and a determinism-risk churn on goldens.
  My vote: NO for now; revisit if the fused far path ends up wanting AO too.

## 12. save/

- Post-port: lazy writer, gated polling. Format module untouched this pass.
- `to_doc` builds an index_of HashMap<String,u16> of specs per save [~] —
  fine (saves are seconds-scale). Snapshot spec dedup exists server-side too;
  see §13 for the shared-`Arc<str>` idea to align them.
- Autosave encodes on the MAIN thread (encode closure runs in start()) [✓] —
  with 100k edits that's a hitch at autosave time. Opinion: move encode to the
  writer thread (send a cheap world snapshot… except World isn't Send-shareable
  cheaply — edits map clone is O(edits) anyway). Alternative: incremental edit
  journal (append-only save log + periodic compaction), which ident/mod.rs
  EditLog already models client-side! There's a designed-but-unwired story
  here: ident::EditLog + compaction marker looks like the intended incremental
  save; today saves re-encode the whole doc. Worth a design decision rather
  than a micro-fix. [~]

## 13. net/

- protocol.rs: now table-driven (messages!/Wire) [✓ mine]. Wire impl for
  Vec<u8> hardcodes the voice cap — if a second byte-blob message ever
  appears with different bounds, introduce newtypes (VoicePayload). Note for
  future-proofing only.
- client.rs (675): poll per frame; peer sample interpolation per peer per
  frame is math-only [?]. Names are String — Arc<str> them (with server).
- server.rs (1603): overall solid (interest grid 3×3 buckets, encode-once +
  Arc frame broadcast [?]). Items:
  - names cloned per join-broadcast and per chat [? lines 470/553/894] —
    Arc<str>.
  - snapshot spec `.to_string()` per edit per joiner [? line 474] despite an
    internal spec_pool of Arc<str> — plumb the Arc through protocol? Protocol
    encode takes &str either way; the clone is just for the message struct.
    Changing ServerMessage::Snapshot to carry Arc<str> leaks Arc into the
    protocol surface. ⚔ Two takes: (a) accept join-time O(edits) clones, cold
    path; (b) make protocol messages borrow (encode from &[(…, &str)]) via a
    separate builder API. (a) unless joins get slow.
  - server.rs is the 4th god-file: connection lifecycle + auth + interest +
    edit ledger + relay + voice. Split candidates: `ledger.rs` (edit cells +
    revs + specs), `interest.rs` (grid), `relay.rs`. Architecture-only.
  - movement envelope check per Move message — cheap [?]. Fine.
- mod.rs content_fingerprint: good idea, keep.

## 14. audio/

- Director post-port is allocation-light; journal/emitters Vecs move into
  AudioFrame each frame (ownership handoff to mixer thread via rtrb SPSC
  [?]). They die on the audio side — no capacity return path. Pool idea:
  ring of reusable frames back over a return channel. Only if profiling shows
  allocator noise; empty Vec::new() frames (the common case — no events, no
  emitters) allocate NOTHING [✓ std behavior], so steady-state is already
  clean.
- acoustics WindowCache: 96³ DDA trace amortized over 0.5 s refresh [?].
  Acceptable; it IS 900k cells — if audio hitches appear on window refresh,
  time-slice the trace. Note only.
- Layering (facts → director derives → SoundSystem executes) is genuinely
  nice. No change.

## 15. input/

- Chord evaluation probes modifier keys per chord per frame (Mods::current()
  recomputed ~12–24× per frame) [? intent.rs:66]. Cache Mods once per
  router.frame_filtered call and pass down. ~50 µs/frame claimed — I'd guess
  less, but the fix is 10 lines and obviously right. Do it as a drive-by.
- Bindings arrays fixed-size per event — good. Router timer unpriming on
  disable (my port) works; add a test for "held key doesn't autofire after
  re-enable".

## 16. presence / avatar / player / interact

- presence Animator uses exp() + sin() per peer per frame [? line 247]. At
  <100 peers this is noise. ⚔ LUT suggestion from agent is over-engineering;
  skip unless a 100-player stress test says otherwise.
- avatar Pose::resolve: 6 fixed parts, no alloc — good. 6 immediate draw-box
  calls per peer → engine-side batching question, not game-side.
- player.rs Motion enum is the model citizen of typed state. `sin_cos` now
  used [✓ mine].
- interact raycast: f64 DDA, ~7 steps, bounded — fine. Called per visible
  tagged peer per frame for tag occlusion [✓] — that's the expensive consumer
  (raycast × peers × frame). With tags on and 30 peers: 30 raycasts/frame.
  Idea: cache tag occlusion per peer with a Memo keyed on (peer cell, eye
  cell). ❌ Rejected on implementation review: ~30 seven-step DDA raycasts per
  frame is trivial, and the cache introduces visible staleness after terrain
  edits between stationary players. Not worth it.

## 17. ui / console / minimap / menu

- ui.rs Line/Span model allocates per line — console is bounded ring (24
  lines) [?]; fine.
- minimap: 2×256² buffers (~512 KiB) reused, throttled refresh [?] — fine.
  Minimap raster iterates loaded chunks per refresh; at radius 20 that's a
  big scan every 500 ms. Only if minimap-on profiles badly: incremental
  repaint on chunk-load events (the event exists: store_chunk). Note only.
- menus: see §4. MenuStack architecture is fine.

## 18. sim/

- Stubs (electrical/thermal are inert proof-of-concept) [✓ ~45 lines each].
  The important thing is what they DON'T yet have: an active-cell tracking
  substrate. When real sim lands it must iterate tracked cells, not chunks.
  Design note now > code later: reserve a `sim/cells.rs` SoA table (cell key
  → packed state), fed by edit/placement events. Do NOT let a first sim
  implementation walk the chunk map per tick. This is the one place the
  user's "design around ECS/SoA" instinct has a green field — get it right
  from day one instead of retrofitting like the section table.

## 19. block/ registry+composition+crafting

- hot_tables clone is revision-gated, NOT per-frame [✓ corrected]. Remaining
  improvements: incremental append on palette growth (§8), packed property
  byte (§8), absorption precompute fold-in [?].
- `Composition::natural` double-sorts after craft_natural sorted [? crafting
  16-24 vs composition 118-123] — trivial dedup, cold path, take it as
  drive-by.
- EditLog unbounded growth ~50 B/edit [? ident] — acceptable; compaction
  marker exists (see §12 save unification idea).

## 20. coord / math / hash / ident

- coord.rs newtypes are excellent (Local privacy, ByPass). block_coord clamp
  slack is fragile-but-tested [?] — leave, it's documented.
- ident/codec Writer/Reader clean. ident vs engine identity types: pick one
  canonical import path for Detail/Rev (§1b).

## 21. harness & tests

- 464 tests, fast, inline. Missing coverage called out above: scheduler
  gating, edge latch/replay, router unpriming. Also NO test drives
  `set_render_config` against a real Engine (needs GPU) — the clear_section
  lane free-exactly-once property is only asserted structurally. A mock-engine
  seam (trait for the 6 engine calls world makes) would make GPU-ownership
  tests possible. ⚔ That trait is exactly the kind of abstraction the
  codebase has deliberately avoided (engine handle passed concretely).
  Middle ground: count frees via the existing profile/gauge hooks in a
  debug assertion.

---

## 22. Cross-cutting policies worth writing down (they're implicit today)

- **String policy**: hot paths must not clone Strings; identity strings
  (names, specs) become Arc<str> at the boundary. One PR: client peer name +
  server player name + chat + (maybe) spec pool alignment.
- **Map policy**: FastMap for coord keys everywhere is consistent. But
  "struct of maps keyed by the same key" (sections ×5) is the anti-pattern to
  burn down → slot-based SoA table (§5).
- **Allocation policy**: per-frame = zero allocs (mostly achieved now);
  per-job = pooled (achieved for snapshots/mesh outputs; NOT for section
  extract — §7); per-event = don't care.
- **Module size policy**: nothing new above ~800 lines; the five god-files
  get split opportunistically (tests out first — zero risk).
- **Determinism policy**: any change touching generation float math needs the
  parity tests + goldens + (ideally) a worldgen version bump story. Already
  culturally strong; keep it.
- **Bounded-everything**: budgets/caps exist for queues, uploads, retries.
  Two unbounded things remain: EditLog growth, edits map scans at admission
  (§6 edits_in_footprint). Both scale with player edit count — the
  "long-lived world" scaling axis is the least-tested one.
