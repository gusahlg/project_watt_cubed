# Confirmed bugs

This file separates open defects from items fixed during the audit. Suggested tests are deliberately concrete; several can become protocol/model tests without a GPU.

## Game, world, and networking

### G-01 — Multiplayer block order is authoritative, but gameplay is not

**Priority: P0. Status: open.**

Breaking removes a block and awards its elements immediately in [`Game::break_block`](../src/game.rs#L606). Placement spends a crafted item before the server decides anything in [`CraftingMod::try_place`](../src/mods/crafting.rs), then [`Game::apply_placements`](../src/game.rs#L638) writes the cell optimistically. The server's [`on_edit`](../src/net/server.rs#L622) validates only string length and distance from the last client-reported position. It does not validate the previous block, inventory, transition, loot, or an edit revision, and a rejection has no acknowledgement.

Consequences:

- two clients can break the same block and both receive its elements;
- a rejected placement can leave a ghost local block and consume the item;
- arbitrary valid-looking portable specs can bypass intended crafting progression;
- prediction cannot distinguish “accepted” from “never answered.”

Move the authoritative cell transition and economy to the server. An edit request should carry a request ID and expected old cell/revision; the response should carry accepted/rejected plus the authoritative cell. Prediction can remain, but it needs rollback/refund bookkeeping.

Tests: race two breaks and assert exactly one reward; reject a placement and assert cell/count rollback; race two placements and assert one authoritative spend.

### G-02 — Client-reported movement can forge edit reach

**Priority: P0. Status: open; non-finite angles were fixed.**

[`on_move`](../src/net/server.rs#L534) now rejects non-finite pose fields, but still accepts any finite position without a movement envelope, collision validation, elapsed-time bound, or world-border check. [`on_edit`](../src/net/server.rs#L622) trusts that position for reach. A client can therefore send `Move(target)` followed by `Edit(target)`.

This contradicts the README's server-authority and bounds/reach claims. Decide whether movement is authoritative or plausibility-checked, add latency-aware displacement and border constraints, and make teleport a permissioned server action.

Tests: a raw authenticated client teleports and edits; ordinary motion within latency slack succeeds; outside-border movement is rejected; NaN/Infinity never reaches another client.

### G-03 — Overhangs, islands, and edited roofs do not shadow lower chunks

**Priority: P0. Status: open with an ignored failing regression.**

[`World::capture_ceiling`](../src/world/streaming.rs#L631) caches only `generator.height()`. That function explicitly represents ground and excludes overhang shelves in [`TerrainGenerator for Terrain`](../src/world/generation.rs#L1149); islands and overlay edits are absent too. Every vertical chunk independently injects full skylight when its top is above that ground height in [`light::propagate`](../src/world/light.rs#L398), even when a dark `+Y` neighbour shell contains a roof.

The ignored `constructed_roof_in_upper_chunk_shadows_lower_chunk` test in [`world/light.rs`](../src/world/light.rs) reproduces the result: the cell below an opaque upper-chunk roof is `15`, expected `0`.

A correct model needs either a highest-opaque/proven-open column that includes generated volumetrics and edits, with lower-column invalidation, or sky injection only at a proven top boundary followed by downward cross-chunk propagation.

### G-04 — Caught worker panics can permanently strand streaming claims

**Priority: P0. Status: open.**

The worker loop catches a panic in [`pipeline.rs`](../src/world/pipeline.rs#L401) but sends no result. Claims are released only while integrating a corresponding `Done::Column`, `Done::Mesh`, `Done::Light`, or `Done::Section`; [`Done`](../src/world/pipeline.rs#L110) has no failure variant. A panic can therefore leave `generating`, `light_inflight`, `NeedsMesh { building: true }`, or `SectionState::Meshing` stuck forever.

Return `Done::Failed(JobKey)` on unwind, release the exact claim centrally, and requeue with a bounded retry/quarantine policy. Inject a panicking job into every lane and require the claim to clear and streaming to converge.

### G-05 — Teleport and freecam return run physics against unloaded air

**Priority: P1. Status: open.**

Unloaded chunks are deliberately treated as air by collision in [`World::collides`](../src/world/query.rs#L172). `/tp` changes only the player in [`command::teleport`](../src/command.rs#L132), while normal motion happens before streaming prepares the new region. Returning from a far detached camera has the same ordering problem. [`World::prepare_around`](../src/world/streaming.rs#L458) already exists for collision safety but is not part of these discontinuous transitions.

Make a position discontinuity transactional: prepare a collision halo before physics resumes. While detached, either retain the player halo or synchronously restore it before reattachment.

### G-06 — Server spec interning and client palette growth are unbounded

**Priority: P1. Status: open.**

Every distinct accepted string is retained forever by `spec_pool` ([state](../src/net/server.rs#L145), [insertion](../src/net/server.rs#L636)), even after its only cell is overwritten. The server never resolves or canonicalizes specs. Clients do resolve remote compositions and can grow toward the 16,384 block cap; texture work follows palette growth. Repeated unique specs against one nearby cell are therefore both a server memory leak and a client resource-exhaustion path.

Parse and canonicalize a typed `BlockSpec` server-side, validate it against negotiated content/rules, remove baseline-restoring overlays, retain only strings referenced by live cells, and impose per-player/world quotas.

### G-07 — Coarse LOD edit reduction has no deterministic ordering contract

**Priority: P1. Status: open.**

[`section::apply_edits`](../src/world/section.rs#L401) lets the last solid input win a coarse cell, but edits originate in hash maps and no server sequence number is stored. The new reducer test proves parity with its reference algorithm, not independence from input permutation. Multiple materials edited within one coarse cell can therefore differ between live history and an unordered join snapshot.

Use an order-independent reducer with a stable material tie-break, or persist an explicit authoritative edit sequence. Test all insertion permutations and live-history versus snapshot replay.

### G-08 — Seed-only multiplayer has no worldgen/content fingerprint

**Priority: P1. Status: open.**

Compatibility is a manually bumped [`PROTOCOL_VERSION`](../src/net/mod.rs#L32). `Hello`/`Welcome` carry no placement-table, registry, mod-set, generator, or shader-independent content digest ([protocol types](../src/net/protocol.rs#L20)). Generation contains threshold-sensitive floating operations such as `powf` in [`generation.rs`](../src/world/generation.rs#L420), while the network intentionally never ships chunks to repair a mismatch.

Add a canonical worldgen/content fingerprint to the handshake and reject mismatches explicitly. Run fixed seed/coordinate voxel digests across supported CPU architectures and operating systems. This is also why main's global `target-cpu=native` setting was not integrated.

### G-09 — Interest management leaves frozen peer ghosts

**Priority: P2. Status: open.**

When a player leaves the interest radius, the server merely stops sending their moves. Clients remove a peer only on disconnect and continue drawing the retained last pose. A distant avatar therefore freezes in place until the viewer moves back into range.

Track interest enter/exit and send explicit visibility events, or introduce a timeout-backed dormant state. Test exit hides/removes the pose and re-entry restores it without duplicating the peer.

### G-10 — Multiplayer clocks diverge and late joiners receive stale time

**Priority: P2. Status: open.**

The server stores a static `day: f32` ([server state](../src/net/server.rs#L166)) and only changes it on `/time`; clients advance independently. A late join receives the last set fraction, not the current phase, and the local day-length setting is not synchronized.

Represent time as a server epoch, phase, and cycle length with occasional correction. Test delayed join and cycle-length propagation.

### G-11 — Far-LOD luminous materials lose emission

**Priority: P2. Status: open.**

Element worldgen intentionally places luminous cave material, but section vertices always use blocklight zero in [`section/mesh.rs`](../src/world/section/mesh.rs#L279). Emission therefore disappears at the full-resolution-to-LOD boundary.

At minimum, stamp a block's self-emission from the hot table; a better far-light model performs a coarse flood. Add a Lumin section fixture and require exposed vertices to retain nonzero blocklight.

### G-12 — Save decoding accepts non-finite player state and lacks whole-file integrity

**Priority: P2. Status: open.**

[`format::decode`](../src/save/format.rs#L279) accepts raw float bit patterns and [`bridge::from_doc`](../src/save/bridge.rs#L82) installs them. A corrupt save can therefore load NaN/Infinity into camera/physics state. The format also has no checksum or global byte budget; the store reads the whole file before decoding.

Reject or sanitize non-finite/border-invalid pose values. A future version should frame/checksum sections, require exact EOF for an intact file, and cap aggregate bytes. Mutation tests should corrupt each region and verify an intact backup wins.

### G-13 — Unauthenticated connections are not bounded by `MAX_PLAYERS`

**Priority: P2. Status: open.**

The accept loop spawns one OS thread per TCP connection in [`accept_loop`](../src/net/server.rs#L282). `MAX_PLAYERS` is checked only after the handshake. Timeouts limit duration, not concurrent threads/file descriptors.

Use a handshake semaphore/global connection cap or a bounded handshake pool. Stress more silent connections than the cap and assert bounded resources.

### G-14 — HUD Off still draws the minimap and mod HUD

**Priority: P2. Status: open.**

[`Game::hud_phase`](../src/game.rs#L818) draws the minimap unconditionally at line 827 and renders mod HUD data unconditionally around line 870. Only the reticle, name tags, and information text consult the HUD mode.

Create a pure HUD render policy for Full/Minimal/Off and test its planned widgets without a GPU.

### G-15 — Natural block save parsing is not an inverse for duplicate naturals

**Priority: P2. Status: open/design decision.**

The registry currently permits duplicate natural elements as distinct compositions, but parsing routes through natural crafting, which sorts/deduplicates. A duplicated natural therefore reloads as another block. The design notes describe naturals as sets, so the cleanest resolution may be to forbid/canonicalize duplicates at construction rather than preserve multiplicity.

Add a block-spec round-trip property over every supported composition tier.

### G-16 — Exact AABB face contact includes the next voxel

**Priority: P3. Status: open.**

[`Aabb::voxel_cells`](../src/math/mod.rs#L97) and [`World::collides`](../src/world/query.rs#L172) use `floor(max)` inclusively. A box ending exactly at integer `1.0` therefore visits cell `1`, although that voxel begins at the non-overlapping boundary. Use `ceil(max) - 1` semantics for the exclusive upper edge and share one range helper. Test positive and negative integer boundaries.

### G-17 — An empty mod-provided reaction can panic

**Priority: P3. Status: open.**

[`ReactionRegistry::register`](../src/block/reaction.rs#L66) performs no validation. Empty reagents make “all present” vacuously true and later divide by a zero maximum. Registration should reject empty/duplicate reagents and invalid optimal-share totals.

## Voxel-engine and GPU

### E-01 — Exposure readback reads a slot that was not the one waited

**Priority: P1. Status: open.**

The renderer waits for the current frame slot in [`wait_slot_and_reclaim`](../../voxel-engine/src/vk/mod.rs#L918). [`ExposureRing::views`](../../voxel-engine/src/vk/exposure.rs#L172), however, writes that slot and reads `slot.other()`, normally the immediately previous slot that may still be in flight. A NaN guard does not make a finite partial read safe.

Read the exact slot whose completion was waited, then overwrite that slot with the new compute result. Add a pure alternating-slot/timeline ownership test.

### E-02 — TAA history barriers do not describe the real previous accesses

**Priority: P1. Status: open.**

In [`taa.rs`](../../voxel-engine/src/vk/taa.rs#L445), the prior history source is declared as a compute storage write even though its last post-resolve use was transfer-read. The image becoming output is transitioned from `UNDEFINED` with no real dependency even though the preceding frame sampled it. Because history images are shared rather than per-frame-slot, these cross-submission dependencies matter.

Track actual layout/access state: transfer-read to compute-sampled for the resolved source, and compute-sampled to compute-storage-write for the next output. Add TAA-on synchronization validation plus static-hold and orbit goldens.

### E-03 — Exposure Off retains stale metered exposure, and TAA toggles retain old history

**Priority: P2. Status: open.**

The public flag says disabling exposure pins it to 1.0, but [`Renderer::set_flags`](../../voxel-engine/src/vk/mod.rs#L658) only replaces booleans and [`Engine::exposure_for_compose`](../../voxel-engine/src/vk/exposure.rs#L575) always returns the shared stale value. Bloom/tonemap can therefore retain the old meter. TAA has the analogous stale-history problem after an off/on toggle.

Reset/publish `Exposure::DEFAULT`, reset meter timing, and invalidate TAA history on relevant flag transitions. Add live-toggle tests and captures.

### E-04 — Swapchain format/usage/composite selection is not portable

**Priority: P2. Status: open.**

[`swapchain.rs`](../../voxel-engine/src/vk/swapchain.rs#L29) falls back to an arbitrary surface format when UNORM is absent. Choosing sRGB would double-encode the already display-encoded tonemap output. Swapchain creation also requests transfer-source usage and opaque composite alpha without checking support.

Extract pure selectors and test sRGB-only formats, missing screenshot-transfer usage, unsupported opaque alpha, and present-mode fallbacks.

### E-05 — Shadow fitting assumes 16:9

**Priority: P2. Status: open.**

[`shadow::fit`](../../voxel-engine/src/vk/shadow.rs#L239) hardcodes `16:9`. Wider windows and wide-FOV source frusta can extend outside the map and produce moving fully-lit strips. Pass the actual source aspect/extent and test all eight frustum corners at 1:1, 16:9, and 32:9.

### E-06 — Curvature is raster-only and hardcoded in a reusable engine

**Priority: P2. Status: open.**

The mesh vertex shader applies a fixed roughly-300-km visual curvature to clip position, while CPU culling, world outputs, and shadow depth remain flat. This can create horizon popping and geometry/shadow disagreement. Make curvature explicit configuration and apply it coherently, or document and tightly bound it as presentation-only.

### E-07 — New public LOD draw inputs are insufficiently bounded

**Priority: P2. Status: open.**

[`Detail::new`](../../voxel-engine/src/mesh.rs#L354) accepts any `u8`, while [`Detail::scale`](../../voxel-engine/src/mesh.rs#L360) shifts `1u32 << level`. [`draw_mesh_faded`](../../voxel-engine/src/frame.rs#L432) accepts arbitrary non-finite fade and raw mode bits. The fragment shader also evaluates equal-edge `smoothstep` when vertical clip is zero.

Use a checked/bounded `Detail`, clamp or reject non-finite fade, replace mode bits with a typed style, and make zero vertical coverage an explicit branch. Add boundary/property tests.

### E-08 — The disabled water-depth SPIR-V remains invalid

**Priority: P2. Status: runtime-safe, implementation still open.**

Commit `5efc734` sets [`WATER_DEPTH_ABSORPTION_VALIDATED`](../../voxel-engine/src/vk/pipeline.rs#L88) false, so the valid flat-tint pipeline is selected and the Vulkan smoke is clean. However, `shaders_spv/mesh3d_water.frag.spv` still fails standalone `spirv-val`, and the source path still lacks a coherent dynamic-rendering local-read attachment/layout/mapping design.

Do not re-enable it piecemeal. Repair shader type/load, color/depth input indices, dynamic-rendering local-read layouts, descriptor layout, and rendering input-attachment state as one change, then require both `spirv-val` and validation smoke to pass.

## Fixed during this audit

### F-01 — Main's progressive LOD overlap/light defects

**Fixed by merge `363dea6`.** The old whole-parent/partial-child covering could double-draw, LOD light averaged losing materials, and missing-neighbour skirts inherited buried darkness. The integrated quadrant masks, quadrant meshes, winner-only light vote, and full-skylight open borders address these as one coherent pipeline. The merged suite includes exact partition, missing-child, quadrant-bound, and downsample tests.

### F-02 — Remote avatars were anchored at eye height as feet

**Fixed by merge `363dea6`.** Typed `Eye`/`Feet`, stance-aware conversion, and `RenderPose::new` remove the eye-height offset bug. New tests cover standing/sneaking/swimming conversion and far-coordinate camera-relative precision.

### F-03 — Height-mip worker paired element IDs with a builtin-only palette

**Fixed in the audit worktree.** The first post-merge graphics run repeatedly panicked at `BlockRegistry::color`: worldgen emitted ID 67 while the background mip worker had only 19 builtin colors. The worker now snapshots the compiled world's immutable color table before spawning. `HeightMip::bake` accepts that snapshot explicitly, making the dependency visible in its API.

### F-04 — Protocol decoders accepted arbitrary trailing bytes

**Fixed in the audit worktree.** Both decoders now require the reader to consume the exact frame. A table-driven test appends a byte to every client and server message variant and requires rejection.

### F-05 — Opaque emissive chunks took the all-dark lighting shortcut

**Fixed in the audit worktree.** The shortcut now also requires zero emission. The analytic-light regression includes an opaque emitter and verifies full blocklight at its cells.

### F-06 — Break/place reach disagreed and detached freecam could place

**Fixed in the audit worktree.** Both actions use `interact::REACH`, expressed as six metres in world units, and detached mode no longer forwards the placement action.

### F-07 — Invalid water local-read runtime and incorrect present wait stages

**Fixed conservatively by engine commit `5efc734`.** The invalid water pipeline is dormant in favor of the valid flat-tint fallback, and the mixed present submission uses correctness-first `ALL_COMMANDS` waits. A validation-enabled, immediate-present, resize/autoshot smoke completed 36 frames with zero warnings/errors and none of the five previously observed VUID/hazard reports.

### F-08 — MSAA depth was bound to a single-sample godray descriptor

**Fixed conservatively by engine commit `6edce74`.** A validation run through the real game, which defaults to 8× MSAA, reported `VUID-RuntimeSpirv-samples-08725`: tonemap binding 2 declared `Sampler2D` but received the multisampled scene-depth view. The renderer now samples depth only at one sample, binds an already-readable single-sample fallback descriptor under MSAA, and sends zero godray strength for that path. A regression test covers every Vulkan sample-count flag, and the same game validation benchmark now exits without validation warnings or errors.

This is a safety fix rather than the final effect implementation: depth-masked godrays remain unavailable while MSAA is active. Add a single-sample resolved depth target before re-enabling them.

### F-09 — Pure Nix selected a stale, pre-LOD engine snapshot

**Fixed in the audit worktree and lock file.** Cargo used the live sibling
engine while `nix run` used the older content hash in `flake.lock`, producing
missing `Detail`, `RenderFlags::vignette`, and `draw_mesh_faded` errors. The
input now pins committed engine revision `6edce74`, and `checks.package` compiles
and tests the pure distributable against that locked source.

The remaining machine-local URL and cross-repository release workflow are
tracked in [nix-and-release.md](nix-and-release.md).

### F-10 — Nix imported 5 GiB of build output and shipped an unlaunchable golden tool

**Fixed in the audit worktree.** The raw engine `path:` input included ignored
`target/` output: the store snapshot measured about 5.0 GiB versus 949,443 bytes
for the current Git-filtered input. A 16 MiB flake check now guards that budget.
Separately, packaged `golden` immediately failed to load `libX11.so.6` because
only the game and server received runtime paths. All installed binaries now
receive mandatory Vulkan/window-library runpaths; a five-second package smoke
progressed through world streaming instead of failing library loading.
