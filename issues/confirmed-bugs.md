# Confirmed bugs

This file separates open defects from items already fixed. Suggested tests are
deliberately concrete; several exist as protocol/model tests without a GPU.

## Open

### G-03 — Generated overhangs and islands do not shadow lower chunks

**Priority: P2 (reduced 2026-07-14 — the constructed-roof half is fixed). Status: open.**

The skylight ceiling now includes edited opaque roofs with per-column cache
invalidation and re-settling of the chunks below, and the former ignored
regression test is green. What remains is the GENERATED volumetrics:
[`TerrainGenerator::height`](../src/world/generation.rs) describes ground
only, so overhang shelves and flying islands still do not shadow the columns
beneath them — every vertical chunk under an island still injects full
skylight when its top is above ground height.

Lifting this needs a generator-side column occupancy summary (topmost solid
including the island/overhang bands) or the alternative model: sky injection
only at a proven top boundary followed by downward cross-chunk propagation.
The visual symptom is bounded (island undersides read as unlit strips, noted
in the audit's visual inspection), which is why this is P2 now.

## Fixed 2026-07-14, second round (engine burn-down)

Every engine issue from the audit is fixed on the engine's `experimental`
(merged to its `main` via voxel-engine PR #2). Verified by the engine suite
(84 tests + a new all-module SPIR-V validation gate), a validation smoke with
the water absorption path LIVE (standard + sync validation, resize, autoshot —
zero errors), and the release game benchmark at its default 8× MSAA (zero
errors, 55 avg FPS, matching the audit baseline).

- **E-01 (P1)** — the exposure readback reads the fence-WAITED slot's own
  buffer (frame N−2's completed value) instead of the possibly in-flight
  `slot.other()`. The parity rule is a pure function with a timeline-model
  unit test.
- **E-02 (P1)** — TAA history barriers derive their `src` scopes from each
  shared image's tracked prior use (transfer-read for the last resolve
  source, compute-sample for the last history input) instead of claiming a
  storage write that never was, or no dependency at all for an image the
  previous frame sampled.
- **E-03 (P2)** — exposure/TAA flag transitions reset their temporal state:
  exposure off pins 1.0 structurally on both the render (tonemap) and main
  (`compose`) threads; any TAA toggle invalidates history so a stale scene
  cannot ghost into the first re-enabled frame.
- **E-04 (P2)** — swapchain format/usage/composite/present-mode selection is
  factored into pure, support-checked selectors with portability-matrix
  tests: UNORM preferred in any colorspace before a loud sRGB last resort,
  `TRANSFER_SRC` requested only when offered (screenshots refuse gracefully
  without it), composite alpha falls back to a supported bit.
- **E-05** — already fixed 2026-07-13 by the eye-centred cascade rework (see
  the shadow entries in the first-round record).
- **E-06 (P2)** — curvature is documented and bounded as presentation-only:
  the droop touches clip position alone (world-space consumers all agree on
  the flat world; culling stays conservative in the safe direction) and is
  clamped so arbitrary distances cannot fold the horizon.
- **E-07 (P2)** — public LOD draw inputs are bounded: `Detail::new` clamps to
  a shift-safe `MAX_LEVEL`, `draw_mesh_faded` sanitizes non-finite fade and
  takes a typed `FadeStyle` instead of raw mode bits (the game ported in
  lockstep), and the LOD clip shader branches explicitly on zero vertical
  coverage instead of evaluating an equal-edge `smoothstep`.
- **E-08 (P2)** — the water depth-absorption local-read path is repaired and
  ENABLED as one coherent change: the depth input attachment loads as vec4
  (the module passes `spirv-val` at vulkan1.3, enforced for the whole tracked
  inventory by a new gate test), the pipeline carries the complete
  input-attachment mapping and the render-pass instance sets/restores the
  same mapping around the blend draws, the scene pass runs depth in
  `RENDERING_LOCAL_READ` whenever the pipeline exists (one
  `depth_pass_layout` function feeds every barrier and attachment info), and
  a framebuffer-local BY_REGION barrier orders opaque depth writes before the
  water branch's reads. MSAA keeps the flat-tint fallback.

Game-side in the same round (user report): the far-LOD load lane is now
LEVEL-triggered — armed for as long as any desired cell is unloaded,
uncovered, and unskipped — instead of relying on boundary-crossing events
that could go quiet with holes still open (flying far up left stale coarse
cubes floating over a missing far field). Any chunk creation and every
section landing also re-arm it, and the selection sweep (grid walk + relief
coarsening) runs once per frame instead of up to three times.

## Fixed 2026-07-14, first round (issue burn-down)

Every game-side confirmed bug from the 2026-07-13 audit was fixed on
`experimental`, with regression tests. Protocol v6 covers the wire changes.

- **G-01 (P0)** — edits are server-authoritative: requests carry an id and the
  expected cell revision; exactly one racer wins, the sender gets an
  `EditAck`, and client prediction rolls back cell, loot, and spent items on
  rejection (`on_break_rejected`/`on_place_rejected` mod hooks). Raced breaks
  over loopback yield exactly one acceptance.
- **G-02 (P0)** — server-side movement envelope (80 m/s + latency slack) and
  world-border checks; implausible moves are not committed and the client is
  snapped back with an authoritative `Position`. `/tp` is an explicit
  `Teleport` message the server can refuse (`--no-teleport`).
- **G-03 (P0), constructed-roof half** — the skylight ceiling merges edited
  opaque roofs, `set_block` invalidates the cached column and re-seeds the
  loaded chunks below, and the once-ignored roof regression is green. The
  generated-volumetrics half remains open above.
- **G-04 (P0)** — a panicked worker job reports `Done::Failed` with its exact
  claim; `fail_job` releases it, retries up to 3 strikes, then quarantines.
  Panic injection is tested for every lane.
- **G-05 (P1)** — `/tp` and freecam reattachment prepare a synchronous
  collision halo at the destination before physics resumes.
- **G-06 (P1)** — the server parses and canonicalizes edit specs with the same
  registry rules clients use (junk is rejected, spellings collapse), releases
  pool entries no live cell references, and caps the pool at the shared
  16,384-block content cap.
- **G-07 (P1)** — coarse LOD edit reduction is position-deterministic: the
  centre-sample edit wins outright, else the topmost `(y, x, z)` solid; all
  input permutations extract identically.
- **G-08 (P1)** — `Hello` carries a platform-stable FNV-1a content fingerprint
  (worldgen version + element table + compiled placement palette); mismatched
  builds are rejected at join. Cross-architecture digest CI remains roadmap
  (structural #14).
- **G-09 (P2)** — interest exits send `PeerExited` both ways and re-entry
  re-delivers poses; clients hide rather than freeze out-of-range avatars.
- **G-10 (P2)** — the shared clock is an anchored phase advanced at a shared
  cycle length; late joiners get the CURRENT time, and `Time` carries
  `day_secs` so all clocks tick in step (`--day-secs`).
- **G-11 (P2)** — far-LOD vertices stamp the block's self-emission from the
  hot table, so luminous materials keep their glow across the LOD boundary.
- **G-12 (P2)** — save decoding rejects non-finite or out-of-border player
  state; the store's backup ladder recovers. Whole-file checksums remain
  roadmap (structural #12).
- **G-13 (P2)** — pre-auth connections are bounded by a handshake cap (64);
  connections past it are refused in the accept loop before a thread spawns.
- **G-14 (P2)** — HUD Off hides the minimap and mod HUD too, via explicit
  `HudMode` policy predicates.
- **G-15 (P2)** — naturals canonicalize to sorted, deduplicated element SETS at
  construction; a block-spec round-trip property covers every reconstructable
  registered composition.
- **G-16 (P3)** — AABB cell ranges use exclusive upper edges
  (`block_coord_end`), shared by `voxel_cells` and `collides`.
- **G-17 (P3)** — reaction registration validates reagents (non-empty, unique,
  nonzero shares summing to 100).

Additionally, fast-movement chunk streaming was reworked: the near job queue
re-prioritizes against the LIVE view centre at every dequeue, and queued jobs
whose region left the view are descheduled (`Done::Cancelled`, claim released,
no strikes) instead of ground through.

## Fixed during the 2026-07-13 audit

### F-01 — Main's progressive LOD overlap/light defects

**Fixed by merge `363dea6`.** The old whole-parent/partial-child covering could double-draw, LOD light averaged losing materials, and missing-neighbour skirts inherited buried darkness. The integrated quadrant masks, quadrant meshes, winner-only light vote, and full-skylight open borders address these as one coherent pipeline. The merged suite includes exact partition, missing-child, quadrant-bound, and downsample tests.

### F-02 — Remote avatars were anchored at eye height as feet

**Fixed by merge `363dea6`.** Typed `Eye`/`Feet`, stance-aware conversion, and `RenderPose::new` remove the eye-height offset bug. New tests cover standing/sneaking/swimming conversion and far-coordinate camera-relative precision.

### F-03 — Height-mip worker paired element IDs with a builtin-only palette

**Fixed in the audit worktree.** The first post-merge graphics run repeatedly panicked at `BlockRegistry::color`: worldgen emitted ID 67 while the background mip worker had only 19 builtin colors. The worker now snapshots the compiled world's immutable color table before spawning. `HeightMip::bake` accepts that snapshot explicitly, making the dependency visible in its API.

### F-04 — Protocol decoders accepted arbitrary trailing bytes

**Fixed in the audit worktree.** Both decoders now require the reader to consume the exact frame. A table-driven test appends a byte to every client and server message variant and requires rejection.

### F-05 — Shadow occluder pass read the placement SSBO at half stride

`shadow_depth.vert.slang` declared binding 0 as `StructuredBuffer<float4>` (16-byte stride) while the CPU/`mesh3d.vert` record is the 32-byte `DrawOffset`. Occluder draw 0 was placed correctly by luck; every odd instance read `fade/mode/flat_rgba` bits as its placement and every later even instance read the WRONG draw's offset. Because the draw list re-sorts by camera distance each frame, the garbage reshuffled with the view — the user-visible "shadows move around as the player looks around". The shader now mirrors the full `DrawOffset` record.

### F-06 — Cascade fit was camera-anchored and slice-fitted

`shadow::fit` fitted each cascade to the forward view-frustum slice (16:9 rectilinear assumed) and snapped its texel grid in CAMERA-relative space, so the grid moved with the eye and every shadow edge crawled during translation. Replaced with eye-centred spheres (radius = the receiver's distance-based selection split + bias margin) and an f64 WORLD-space texel snap: the fit reads no camera orientation, matrices are bitwise-identical under rotation, translation moves the map in exact whole-texel steps, and coverage is total for any FOV/lens/aspect. At the shipped fovy the eye-centred sphere is also ~3.5× smaller than the old slice sphere, i.e. ~3.5× finer shadow texels. Locked by five unit tests in `vk/shadow.rs` and the `shadow_probe` A/B/A capture diagnostic (swing the camera away and back: `pct_changed = 0.0`).

### F-07 — Opaque emissive chunks took the all-dark lighting shortcut

**Fixed in the audit worktree.** The shortcut now also requires zero emission. The analytic-light regression includes an opaque emitter and verifies full blocklight at its cells.

### F-08 — Break/place reach disagreed and detached freecam could place

**Fixed in the audit worktree.** Both actions use `interact::REACH`, expressed as six metres in world units, and detached mode no longer forwards the placement action.

### F-09 — Invalid water local-read runtime and incorrect present wait stages

**Fixed conservatively by engine commit `5efc734`** (and the water path itself
fully repaired and re-enabled 2026-07-14, see E-08 above). The mixed present
submission uses correctness-first `ALL_COMMANDS` waits.

### F-10 — MSAA depth was bound to a single-sample godray descriptor

**Fixed conservatively by engine commit `6edce74`.** A validation run through the real game, which defaults to 8× MSAA, reported `VUID-RuntimeSpirv-samples-08725`: tonemap binding 2 declared `Sampler2D` but received the multisampled scene-depth view. The renderer now samples depth only at one sample, binds an already-readable single-sample fallback descriptor under MSAA, and sends zero godray strength for that path. A regression test covers every Vulkan sample-count flag; depth-masked godrays remain unavailable while MSAA is active until a resolved single-sample depth target is added.

### F-11 — Pure Nix selected a stale, pre-LOD engine snapshot

**Fixed in the audit worktree and lock file.** Cargo used the live sibling
engine while `nix run` used the older content hash in `flake.lock`. The input
now pins a committed engine revision, and `checks.package` compiles and tests
the pure distributable against that locked source. See
[nix-and-release.md](nix-and-release.md).

### F-12 — Nix imported 5 GiB of build output and shipped an unlaunchable golden tool

**Fixed in the audit worktree.** The raw engine `path:` input included ignored
`target/` output (~5.0 GiB vs 949,443 bytes Git-filtered); a 16 MiB flake
check now guards the budget. All installed binaries receive mandatory
Vulkan/window-library runpaths; a five-second package smoke progressed through
world streaming instead of failing library loading.
