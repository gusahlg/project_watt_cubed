# Performance pass and benchmark contract

This document records the July 2026 optimization pass and the contract for measuring it later. The requested goals are:

| Scenario | Future target | Frame-time equivalent |
| --- | ---: | ---: |
| Minimum, stripped but playable | above 20,000 FPS | below 0.050 ms |
| Fast, with more presentation and gameplay systems enabled | above 5,000 FPS | below 0.200 ms |

These are **unverified targets, not measured results**. They are hardware-, driver-, resolution-, scene-, and frame-definition-dependent. A future result must report average FPS, 1%-low FPS (`p1_fps` in the built-in harness), average frame time, sample count, end-of-run RSS, and the exact settings and machine. At these rates, also distinguish application frames produced from frames accepted by the render thread and images actually presented; counting work that a latest-frame mailbox discards is not rendering throughput.

## Profiles and controls

The settings menu and `/gfx` command share the persisted descriptor table in [`src/settings.rs`](../src/settings.rs). Selecting a named profile applies all profile-owned fields atomically. Editing any individual field marks the state `Custom`; it does not discard the edited values. Profiles deliberately preserve fullscreen, FOV, UI scale, menu scale, camera shake, and the environment-only face-culling choice. They remove any FPS cap and own VSync plus the rendering/gameplay cost controls.

| Profile | World and clocks | Presentation | Gameplay/background work |
| --- | --- | --- | --- |
| **Minimum** | horizontal distance 0 (current chunk column only), vertical distance 1, 25% render scale, 1x MSAA, VSync off, streaming/sky/mod rates 15 Hz, physics 30 Hz, distant LOD off | HUD off; minimap, mod HUD, player models, name tags, sky and costly post/light lanes off | simulation, mod updates, and periodic autosave off |
| **Fast** | horizontal distance 3, vertical distance 2, 50% render scale, 1x MSAA, VSync off, streaming, physics, sky, and mod rates 60 Hz; distant LOD on with 3 levels starting at detail 4 | minimal HUD, player models on; minimap, mod HUD, name tags, sky and costly post/light lanes off | simulation, mod updates, and autosave on |
| **Default** | restores the shipped mix: distance 6/3, 100% scale, every-frame streaming, physics, sky-clock, and mod advancement; distant LOD is off by default | full HUD and the shipped visual lanes | simulation, mods, minimap, models, tags, and autosave on |
| **Custom** | exact individually selected values | exact individually selected values | exact individually selected values |

“Costly lanes off” currently means occlusion, voxel/block lighting, baked AO, auto exposure, bloom, god rays, clouds, weather, stars, day/night animation, TAA, fog, ambient light, shadows, procedural sky, VRS, water animation, and vignette are disabled. Sunlight remains enabled so stripped terrain is still readable. Fast then re-enables its explicitly configured distant LOD lane.

The new independent performance controls are:

- `render_distance`: 0–20 horizontal chunk rings; zero draws only the current chunk column while retaining an unmeshed collision/data halo. `vertical_distance`: 1–10 chunk layers above and below the eye. Separating them avoids loading tall columns of invisible sky and deep rock.
- `lod2`: far field on/off; `lod_levels`: 1–8 base-2 distance rings; `lod_detail`: 2–6, where a lower value is finer and the cell width is `2^detail` metres (4, 8, 16, 32, or 64 m). The approximate outer range is `max(render_distance, 1) × 16 × 2^lod_levels` metres. The hierarchy is shortened when needed so `lod_detail + lod_levels - 1 <= 9`.
- `stream_hz`: every frame, 15, 30, 60, 120, or 240 Hz. Forced refreshes still occur after teleports, free-camera changes, HUD/minimap activation, and settings changes.
- `physics_hz`: every frame, 30, 60, 120, 240, 500, or 1000 Hz. Mouse look and camera effects remain render-rate responsive; edge-triggered flight input is latched until a physics tick consumes it.
- `sky_hz`: every frame, 15, 30, 60, 120, or 240 Hz. It throttles local day/night-clock advancement through the same bounded accumulator; `day_night=off` renders fixed noon while preserving the authoritative clock for networking and future re-enables.
- `mod_hz`: every frame, 15, 30, 60, 120, or 240 Hz. It schedules enabled mod-hook batches without losing placement, inventory, crafting, or navigation edges between ticks; one permitted batch replays every queued edge-bearing frame in order. `mod_logic=off` still removes the lane completely.
- `simulation`, `mod_logic`, and `autosave`: independently remove simulation ticks, mod hooks, and periodic save serialization. Core movement, mining, and an explicit save on clean world exit remain available. Crafted-block placement requires `mod_logic`; inventory/crafting input additionally requires `mod_hud` and a visible master HUD, preventing hidden modal input.
- `hud_mode`: Off, Minimal, or Full, plus independent `minimap`, `mod_hud`, `player_models`, and `name_tags` gates.

The existing render scale, MSAA, VSync, cap, meshing-lighting/AO, LOD, sky/weather, illumination, and post-process switches remain independently editable. Named profiles are starting points, not special renderer modes.

## Implemented in this pass

### Startup, settings, and application loop

- Interactive new, loaded, and network worlds use lazy construction. The selected view and render configuration is installed before a small collision-safe spawn slab is prepared; the previous eager default-volume generation is avoided. Headless/test constructors retain their eager contract. See [`src/world/mod.rs`](../src/world/mod.rs) and [`src/save/bridge.rs`](../src/save/bridge.rs).
- Disposable benchmark worlds do not construct autosave state. Created and loaded slots reset dirty tracking to the world's current edit generation, and the autosave writer thread is created only when a dirty periodic write is actually due. Clean worlds, disabled autosave, and synchronous exit-only saves create no background thread; disabled autosave also bypasses polling and serialization.
- Static menu frames no longer resend an identical engine settings bundle, and steady frames change VSync only when the effective state changes.
- `WATT_BENCH` no longer enables instrumentation implicitly. `WATT_BENCH_PROFILE=1` opts into attribution; headline measurements remain uninstrumented.
- The benchmark sample vector reserves for 25,000 measured frames per second up front, keeping target-rate runs out of the allocator.
- Benchmark duration, warmup, and elapsed-time accounting use `f64`, and the
  wall-time/sample boundary uses one clock. Long high-rate runs therefore avoid
  both boundary drift and material `f32` accumulation error.

### Game update and presentation

- Physics, simulation calls, world streaming, sky-clock advancement, and mod dispatch have independent, bounded accumulators. A long pause can bank at most 0.25 seconds, preventing an unbounded catch-up burst. The legacy `every` setting preserves render-rate behavior where offered; simulation's outer call is capped at 20 Hz instead of being entered on every render frame.
- Singleplayer returns before network polling/profiling work. Simulation and minimap objects are lazy optional allocations; mod logic, mod HUD, models, and tags stop at their owning boundary instead of merely hiding output.
- Input routing snapshots movement axes once and reuses them for movement/freecam. Disabled mod logic skips placement probes, disabled mod UI skips inventory/crafting/navigation probes, and an overlay is closed once when its HUD lane is hidden so an invisible modal cannot consume input.
- Mod edges retain their input snapshot and exact edge-time placement raycast.
  Ordered replay therefore preserves discrete actions without retargeting a
  delayed placement from a later player pose.
- Hidden/minimal HUD modes do not refresh the minimap. Full-HUD coordinates are reformatted only when the rounded tenth changes; FPS is sampled at 4 Hz and reformatted only when its displayed integer changes; online text changes only with count or ping. HUD Off returns at the start of HUD recording unless the console is open, and Minimal does not draw closed console scrollback.
- Peer records and mod-placement buffers retain capacity across frames. Disabling models avoids animator/rig composition; disabling tags avoids projection, occlusion raycasts, text measurement, and name cloning. Sky and local/remote avatar recording are gated before draw-list construction.
- Fixed-rate physics latches flight-toggle edges, while idle collision axes return before building an AABB or querying the world. See [`src/game.rs`](../src/game.rs) and [`src/input/movement.rs`](../src/input/movement.rs).
- Sun direction/elevation/daylight are sampled once for lighting, clear colour, and sky geometry. The sample is cached by visual clock value, so disabled day/night uses cached fixed-noon lighting and performs no steady-frame trigonometry.
- Camera orientation is cached by exact yaw, pitch, roll, and FOV bit patterns. The f64 eye remains separate for camera-relative rebasing, so translation with unchanged orientation does not rebuild the `Camera3D` basis or repeat its trigonometry.
- When weather, clouds, water animation, and exposure are disabled, the composed lighting uniform packet and clear colour are cached by clock value. Per-frame dither is patched independently, and wrapped camera-XZ animation coordinates are recomputed only when camera XZ changes. Minimum and Fast use this stripped cache path.
- Disabling weather removes every weather-derived input before uniform composition: coverage, rain palette overrides, and fog bonus all become zero. Disabled weather therefore cannot leave storm tint or weather fog active.
- The crafting mod mirrors held elements only when the shared stash revision changes and retains the mirror vector's capacity. Stable mod-update frames no longer rebuild temporary element vectors.

### World streaming, drawing, and LOD

- Horizontal and vertical view distances are independent. The minimum `0/1` mesh volume is only the current vertical column (three chunks), plus a small unmeshed data/collision halo, rather than forcing the old wider/taller minimum.
- The LOD ladder is configurable and can transition live. A change frees old GPU-owned sections, clears derived frontier/fade state, re-arms the new ladder, and rejects late worker results from a disabled or obsolete ladder. Every asynchronous claim also carries a unique token, so an old result, cancellation, failure, or queued upload cannot capture a same-position replacement after unload/re-admission.
- LOD disabled is an early return in streaming and drawing: no mip bake, section selection, clipping command, section upload, or section draw.
- A permitted streaming pass computes the desired section frontier once and shares it among unload, enqueue, and visible-cover resolution instead of walking it up to three times.
- The far-field frontier is retained across streaming passes until its exact eye, velocity, ladder, or relief-mip inputs change; worker-result covering still refreshes independently. Base-2 ladder step is precomputed, and per-section band selection uses bounded multiply/compare steps instead of logarithms.
- Settled/fading cover state is maintained incrementally, section draw events
  use O(1) settled lookups, and obsolete far ancestors/children are retired once
  their cover and fade dependencies are resolved.
- Large camera discontinuities and LOD-ladder changes explicitly purge queued
  far work and cancel its claims, so obsolete jobs cannot monopolize workers.
- The chunk draw cache now stores prepared origins and copied mesh handles. Steady rendering walks only drawable chunks without hashing each coordinate back through the loaded-chunk map or recomputing chunk origins. Revision checks rebuild the cache only when mesh ownership changes. See [`src/world/streaming.rs`](../src/world/streaming.rs).
- The far-field draw cache similarly resolves the fade cut, section-map lookup, material mode/colour, quadrant masks, block origins, detail, and GPU handles at streaming cadence. Render frames only perform occlusion checks, camera-relative rebasing, and command emission. AO/lighting remeshes retain the independent terrain-summary mip instead of rebaking it.
- Padded voxel/light snapshots and mesh outputs use shared cross-thread pools.
  Lighting-off snapshots omit the 18-cubed light buffer entirely, and disabling
  both lighting and AO selects a culling-only mesh path.
- Far-section jobs sample directly into one pooled dense section and mesh all
  quadrants with O(1) neighbour reads. The production path no longer creates
  1,024 temporary RLE columns merely to expand them again; a byte-parity test
  retains the reference extraction path as a correctness oracle. See
  [`src/world/section/mesh.rs`](../src/world/section/mesh.rs).

### Authoritative terrain generation

- Each immutable fBm field precomputes its octave normalization once.
- Height controls that intentionally share displacement fields now compute warp coordinates once; climate controls do the same. Raw warped weirdness is reused for ridge shaping and river placement, and identity-gamma controls bypass `powf`.
- A generation column uses a fixed 256-profile array instead of a heap allocation, while the existing vertical-column job continues to share those profiles across all requested Y chunks.
- World jobs share one immutable terrain generator through `Arc`, and FBM
  octave-column state is inline instead of allocating tiny vectors per column.

These are caching and redundant-work removals, not a world-generation redesign. The same authoritative seed/coordinate mapping must remain exact; see [`src/world/generation.rs`](../src/world/generation.rs).

## Future benchmark procedure

Use the release profile in [`Cargo.toml`](../Cargo.toml) (optimization level 3, fat LTO, one codegen unit). Build once so compilation is outside every sample:

```sh
cargo build --release
```

For interactive inspection, select a profile in the settings menu or run `/gfx preset minimum`, `/gfx preset fast`, or `/gfx preset default`. Benchmark runs can apply the profile without mutating the saved configuration by setting `WATT_BENCH_PRESET`; record any separately overridden Custom settings because the `preset=` marker alone does not recreate them.

Run at least five samples per profile on the same machine, native release binary, window size, compositor state, GPU power mode, driver, and temperature envelope:

```sh
for run in 1 2 3 4 5; do
  WATT_BENCH=30 WATT_BENCH_PRESET=minimum WATT_BENCH_SEED=42 \
    ./target/release/project_watt_cubed
done
```

The harness performs a three-second warmup, slowly rotates the camera, and prints `frames`, `avg_fps`, `p1_fps`, `avg_ms`, and `rss_mb`. Use these scenarios:

1. **Minimum target:** `WATT_BENCH_PRESET=minimum`, seed 42, spawn position.
2. **Fast target:** `WATT_BENCH_PRESET=fast`, otherwise identical.
3. **Default/control:** `WATT_BENCH_PRESET=default`, otherwise identical.
4. **Far-coordinate parity:** repeat Minimum and Fast with `WATT_BENCH_POS="1000000,128,-1000000"`; this path receives two extra warmup seconds.
5. **LOD stress:** Custom profile with distant LOD enabled, record distance/vertical/levels/detail explicitly, then use the same seed and position.
6. **Attribution only:** repeat a representative failure with `WATT_BENCH_PROFILE=1`. Never compare this run directly with the uninstrumented target.

Keep `VOXEL_PROFILE` unset and `WATT_BENCH_PROFILE` absent for headline numbers. VSync must be off and the FPS cap zero. Also capture CPU/GPU model, clocks/power policy, RAM, OS, window resolution, render scale, backend, driver, commit IDs for both this repository and the sibling renderer, and whether the window was visible or occluded.

The current harness is a steady rotating-camera test, not a movement/streaming benchmark. Before claiming cold-start or traversal performance, add deterministic scenarios that cross chunk boundaries at fixed velocities and separately report startup-to-playable, convergence, hitch percentiles, upload backlog, and steady state. Validate frame semantics with render-thread/GPU counters: at 20,000 producer iterations per second, a mailbox can make application FPS look excellent while presenting almost none of those frames.

## Prioritized remaining opportunities

### P0: sibling `voxel-engine`

These items were found in the renderer audit but cannot be implemented in this repository; the sibling checkout is `../voxel-engine` from the repository root.

1. **Stop cloning completed draw lists in `finish_frame`.** Move, swap, or pool frame-owned lists and scratch allocations. A full list clone and its memory traffic can dominate a 0.05 ms budget even when the scene is small.
2. **Make stripped render lanes structurally absent.** Shadows-off still executes the shader's multi-tap PCF path; bloom-off still retains clear/sample work; the pipeline still pays HDR/tonemap overhead. Use pipeline/specialization variants or truly uniform fast paths so Minimum can bypass shadow sampling, bloom resources, exposure, HDR intermediates, and tonemapping where correctness permits.
3. **Reject/coalesce before frame construction.** The latest-frame mailbox can discard a frame only after the game has paid to update, allocate, sort, and record it. Acquire a render admission token before expensive composition, or reuse the rejected frame's storage, and expose accepted/presented counts beside producer FPS.
4. **Remove steady O(n log n) draw preparation.** Cache stable opaque ordering, bucket by pipeline/material, update only changed records, and use persistent/indirect GPU buffers where beneficial. Preserve the required ordering for transparent geometry.
5. **Move allocator maintenance off every frame.** Trigger it on pressure, allocation events, or a low-frequency maintenance cadence.
6. **Replace the exposure reducer with a hierarchical/parallel reduction** and skip its resources entirely when exposure is disabled.
7. **Cache view/projection/frustum construction** by camera orientation, lens, framebuffer size, and aspect. The game now avoids rebuilding `Camera3D` orientation when its angles are unchanged, but `begin_3d` still reconstructs matrices and the frustum every accepted frame.
8. **Add a true water-effects-off path.** `water_anim=false` freezes time but the water shader still evaluates its wave normal, reflection, Fresnel, absorption, and glint. Minimum needs a flat/opaque-water specialization or a structurally skipped water pass; wrapped XZ coordinates must remain intact until that lane exists.

### P1: game and world

1. Split streaming into a cheap result/upload pump and an event/rate-driven
   topology pass. Exact frontier reuse is now in place, but result latency can be
   decoupled from topology cadence so a 15 Hz Minimum profile publishes finished
   work promptly without repeating eye/velocity and admission bookkeeping.
2. Move multiplayer transport polling/reporting behind a bounded wake or event
   path. Simulation's outer call is now fixed at 20 Hz, but enabled network
   transport still crosses its call/lock boundary at application-frame rate.
3. Broaden lighting-packet caching to mixed configurations: revision-key static
   atmosphere/weather components and isolate cloud, water, and exposure updates
   so enabling one dynamic lane does not force full palette and clear-colour
   recomposition. The stripped static path is already cached.
4. Add compact lighting-on snapshot variants (`all dark`/`all bright`). The
   unlit path now omits its light grid, but uniform lit chunks can still avoid a
   padded allocation and copy while voxel lighting remains enabled.
5. Batch GPU uploads and decay oversized pools. Cross-thread snapshot and mesh
   pools now remove steady churn, but individual uploads remain expensive during
   traversal, and rare pathological meshes should not permanently set retained
   idle capacity.
6. Batch terrain-noise evaluation across columns and reuse octave coordinate
   work. SIMD or reassociation is allowed only if an authoritative byte-parity
   test proves it does not alter generated blocks; otherwise preserve exact
   scalar evaluation order.
7. Cache name-tag text/measurement and share peer names (`Arc<str>`) for the
   models/tags-on multiplayer case. The new gates remove this work when off, but
   enabled tags can still clone and measure strings each frame.
8. Budget mod edge bursts. Idle and periodic mod admission is rate-bounded, but
   a burst of input edges intentionally replays multiple ordered updates in one
   frame. A time budget with an ordered continuation would bound third-party hook
   cost without losing or reordering actions.
9. Rekey or reprioritize accepted far jobs during continuous high-speed travel.
   Teleports and ladder changes purge obsolete work, while ordinary traversal
   can still leave valid-but-low-value jobs ahead of newly important sections.
10. Consider a build-time minimal feature set only after runtime lanes converge.
    Fat LTO already removes unreachable code, so a separate binary is worthwhile
    only if measurement shows code size, instruction-cache pressure, startup
    registration, or background initialization remains material.

## Correctness constraints

- **World generation is authoritative.** For a fixed worldgen version, seed, registry, and coordinate, generated blocks must remain identical. Caching may remove duplicate evaluation but must not change hash streams, sample coordinates, floating-point ordering, placement precedence, save replay, or server/client agreement.
- **LOD remains a consecutive base-2 hierarchy.** Detail is 2–6, levels are 1–8, coarsest detail never exceeds 9, near chunks own the clipped inner volume, and covering/fade logic may show the selected level or its one-level-finer hysteresis fallback—never a hole. Live changes must free each GPU handle exactly once and discard stale asynchronous results.
- **Optimization gates cannot change core playability.** Minimum must retain input, camera, collision, mining, authoritative edits, readable terrain, console access, and an explicit clean-exit save. Placement/inventory hooks are recoverable by enabling `mod_logic`, `mod_hud`, and a master HUD mode that exposes mod UI; disabled optional systems must perform no hidden periodic work.
- **Measure the optimized program, not the profiler.** Headline runs are release, uncapped, unsynced, fixed-scene, and uninstrumented. Profile runs diagnose a miss and are reported separately. Golden/worldgen/state-machine tests establish correctness; they do not substitute for a benchmark, and no FPS target is considered met until the later reproducible run records it.
