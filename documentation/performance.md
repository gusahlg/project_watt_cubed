# Performance pass and benchmark contract

This document records the July 2026 optimization pass — originally developed on
`experimental` (PR #5) and re-implemented on top of the post-refactor `main`
architecture (producer scheduler, resident GPU meshes, audio stack) — and the
contract for measuring it later. The requested goals are:

| Scenario | Future target | Frame-time equivalent |
| --- | ---: | ---: |
| Minimum, stripped but playable | above 20,000 FPS | below 0.050 ms |
| Fast, with more presentation and gameplay systems enabled | above 5,000 FPS | below 0.200 ms |

These are **unverified targets, not measured results**. They are hardware-, driver-, resolution-, scene-, and frame-definition-dependent. A future result must report average FPS, 1%-low FPS (`p1_fps` in the built-in harness), average frame time, sample count, end-of-run RSS, and the exact settings and machine. At these rates, also distinguish application frames produced from frames accepted by the render thread and images actually presented; counting work that a latest-frame mailbox discards is not rendering throughput.

## Profiles and controls

The settings menu and `/gfx` command share the persisted descriptor table in [`src/settings.rs`](../src/settings.rs). Selecting a named profile applies all profile-owned fields atomically. Editing any individual profile-owned field marks the state `Custom` (enforced once, in the `Setting::step`/`parse_human` wrappers — no UI surface has to remember it); it does not discard the edited values. Profiles deliberately preserve fullscreen, FOV, UI scale, menu scale, camera shake, audio, and the environment-only face-culling choice. They remove any FPS cap and own VSync plus the rendering/gameplay cost controls.

| Profile | World and clocks | Presentation | Gameplay/background work |
| --- | --- | --- | --- |
| **Minimum** | horizontal distance 0 (current chunk column only), vertical distance 1, 25% render scale, 1x MSAA, VSync off, streaming/sky/mod rates 15 Hz, physics 30 Hz, distant LOD off | HUD off; minimap, mod HUD, player models, name tags, sky and costly post/light lanes off | simulation, mod updates, and periodic autosave off |
| **Fast** | horizontal distance 3, vertical distance 2, 50% render scale, 1x MSAA, VSync off, streaming, physics, sky, and mod rates 60 Hz; distant LOD on with 3 levels starting at detail 4 | minimal HUD, player models on; minimap, mod HUD, name tags, sky and costly post/light lanes off | simulation, mod updates, and autosave on |
| **Default** | restores the shipped mix: distance 6/3, Auto render scale (`DEFAULT_AUTO_RENDER_SCALE`, currently 1.0 — the threshold rule stays in code but is inert until TAAU), TAA forced on whenever the effective scale is below 1.0, every-frame streaming, physics, sky-clock, and mod advancement; distant LOD is off by default | full HUD and the shipped visual lanes | simulation, mods, minimap, models, tags, and autosave on |
| **Custom** | exact individually selected values | exact individually selected values | exact individually selected values |

“Costly lanes off” currently means occlusion, voxel/block lighting, baked AO, auto exposure, bloom, god rays, clouds, weather, stars, day/night animation, TAA, fog, ambient light, shadows, procedural sky, VRS, water animation, and vignette are disabled. Sunlight remains enabled so stripped terrain is still readable. Fast then re-enables its explicitly configured distant LOD lane. Variable-rate shading is a three-way setting (`auto`/`on`/`off`, default `auto`): Auto turns it on only when the render extent (window × render scale) has at least `Engine::vrs_useful_above_pixels()` pixels (texel area × 32768, ~8.4 Mpx for 16×16 texels), falling back to 8,000,000 when the engine has no threshold; `/gfx` prints which floor is in effect. The classify pass plus shading-rate attachment cost more than coarse shading saves at 1080p (RTX 4060: 3361 vs 5064 fps) and 3440×1440 (RTX 3070: 4236 vs 4495 fps); it pays at 4K-class extents (3440×1440 at 200% = 19.8 Mpx). Default Auto render scale is currently **1.0** (the `window_w × window_h > 1,500,000` rule remains, but the scale constant is 1.0 so it does not downscale). TAA is forced on whenever the *effective* scale is below 1.0; the bench JSON `render_lanes.taa` field is that effective lane, not the stored `taa` setting. Raise `DEFAULT_AUTO_RENDER_SCALE` to 0.8 only after the engine's Catmull-Rom TAAU (engine `wt/round4+`) is in main **and** TAA forcing is verified on the `WATT_BENCH_PRESET=default` path.

Measured 2026-09-09 on louise-pc (RTX 4060, 1920×1080 fullscreen, engine main `e3adfc7`, two passes each):

| preset | main 58ffbcd | this branch (cf39f09, Default at 0.8 without TAAU) |
|---|---|---|
| Default | 6396 / 6416 fps, p99 0.44 ms | 5448 / 5461 fps, p99 0.50–0.54 ms |

That branch's Default rendered at 1536×864 (`render_scale_auto: true`, 0.8) but the bench JSON showed `"taa": false` — a plain 0.8 upscale that was ~15% slower than native on this GPU-light preset. Minimum and Fast are unchanged (no auto scale). Native 1.0 Default is the control until TAAU lands. Custom keeps the user's explicit scale and TAA.

## VRAM guard

Startup probes Vulkan once (`src/benchmark/system.rs`, the same path as the benchmark GPU inventory) for the largest display and a first-pass heap/MSAA snap; once the window exists, session fit asks `Engine::estimate_render_targets` and budgets from `gpu_caps()` (device-local heap × 60%, or live free × 0.85 when the probe still has `VK_EXT_memory_budget`).

The degrade ladder is MSAA 8→4→2→1, then render scale in −0.25 steps down to 0.5. Cuts are this session only (settings.cfg is not rewritten). A `graphics:` line names the request, the free/held split (or the heap budget when live data is absent), and what this session actually runs. The settings screen shows the same notice.

If even 1× MSAA at 50% scale does not fit, the game still starts at that floor and prints why. The guard cannot re-fit after the engine process has already failed to allocate: an `ERROR_OUT_OF_DEVICE_MEMORY` (or similar) at render-target creation is the real failure. Close other GPU-heavy programs, then lower MSAA and render scale in settings before starting again.

The independent performance controls are:

- `render_distance`: 0–20 horizontal chunk rings; zero draws only the current chunk column while retaining an unmeshed collision/data halo. `vertical_distance`: 1–10 chunk layers above and below the eye. Separating them avoids loading tall columns of invisible sky and deep rock.
- `lod2`: far field on/off; `lod_levels`: 1–8 base-2 distance rings; `lod_detail`: 2–6, where a lower value is finer and the cell width is `2^detail` metres. The approximate outer range is `max(render_distance, 1) × 16 × 2^lod_levels` metres. The hierarchy is shortened when needed so `lod_detail + lod_levels - 1 <= 9` (one source: `render_config::max_lod_levels`).
- `stream_hz`: every frame, 15, 30, 60, 120, or 240 Hz. Forced refreshes still occur after teleports, net snap-backs, freecam changes, HUD/minimap activation, and settings changes.
- `physics_hz`: every frame, 30, 60, 120, 240, 500, or 1000 Hz. Mouse look and camera effects remain render-rate responsive; edge-triggered flight/jump input is latched until a physics tick consumes it.
- `sky_hz`: every frame, 15, 30, 60, 120, or 240 Hz. It throttles local day/night-clock advancement; `day_night=off` renders fixed noon while preserving the authoritative clock for networking and future re-enables.
- `mod_hz`: every frame, 15, 30, 60, 120, or 240 Hz. It schedules enabled mod-hook batches without losing placement, inventory, crafting, or navigation edges between ticks; one permitted batch replays every queued edge-bearing frame in order, and a delayed placement keeps its exact edge-time raycast cell (`ModContext::place_target`).
- `simulation`, `mod_logic`, and `autosave`: independently remove simulation ticks (the producer stays registered on the scheduler; `Scheduler::set_enabled` gates it), mod hooks, and periodic save serialization. Core movement, mining, and an explicit save on clean world exit remain available. Crafted-block placement requires `mod_logic`; inventory/crafting input additionally requires `mod_hud` and a visible master HUD, preventing hidden modal input.
- `hud_mode`: Off, Minimal, or Full, plus independent `minimap`, `mod_hud`, `player_models`, and `name_tags` gates.

All four rate lanes and the scheduler's `Cadence::Hz` producers run on ONE accumulator implementation, [`sched::RateGate`](../src/sched/mod.rs): a bounded bank (0.25 s cap — a pause replays a bounded burst, never an unbounded one) with `0` as the explicit every-frame mode.

`Engine::gpu_load()` (GPU frame/gap ms) is opt-in via `enable_gpu_load(true)` (~1.5% at Minimum when on) and is left off; it is the intended input for a later GPU-aware stream-pacer headroom estimate (today the pacer's "frame has headroom" is CPU-based).

## Implemented in this pass

### Startup, settings, and application loop

- Interactive new, loaded, and network worlds use lazy construction (`World::with_config_lazy`, `save::load_with_config`). The selected view and render configuration is installed before a small collision-safe spawn slab is prepared; the previous eager default-volume generation is avoided. Headless/test constructors retain their eager contract.
- The autosave writer thread is created only when a dirty periodic write is actually due (`Autosaver` spawns lazily; a dead worker is replaced on the next due write). Clean worlds, disabled autosave, and synchronous exit-only saves create no background thread; disabled autosave also bypasses polling and serialization in `App::update_playing`.
- `WATT_BENCH` no longer enables instrumentation implicitly. `WATT_BENCH_PROFILE=1` opts into attribution; headline measurements remain uninstrumented. `WATT_BENCH_PRESET` applies a profile for the run without persisting it.
- The benchmark sample vector reserves for 25,000 measured frames per second up front; duration, warmup, and elapsed-time accounting use `f64` on one clock, so long high-rate runs avoid boundary drift and `f32` accumulation error.

### Game update and presentation

- Physics, world streaming, sky-clock advancement, and mod dispatch have independent `RateGate` clocks; the fixed-tick simulation already runs at 20 Hz on the scheduler. Singleplayer returns from the net phase before any polling or profiling scope.
- Input routing snapshots movement axes once and reuses them for movement and freecam (`MoveInput::freecam_axes`). Disabled mod logic skips placement probes at the router (`Router::frame_filtered`), disabled mod UI skips inventory/crafting/navigation probes, a disabled minimap skips its mode key, and an overlay is closed once when its HUD lane is hidden so an invisible modal cannot consume input.
- HUD Off records nothing unless the console is open; Minimal does not draw closed console scrollback. Full-HUD coordinates are reformatted only when the rounded tenth changes; FPS is sampled at 4 Hz and reformatted only when its displayed integer changes; online text changes only with count or ping. All ride one keyed-memo primitive ([`derived::Memo`](../src/derived.rs)).
- Camera orientation is cached by exact yaw/pitch/roll/FOV bit patterns (the f64 eye stays outside the key, so translation rebuilds nothing). Sun direction/elevation/daylight are sampled once per frame ([`SkyFrame`](../src/sky/clock.rs)) and cached by clock value; `day_night=off` uses cached fixed-noon lighting and performs no steady-frame trigonometry.
- With weather, clouds, water animation, and auto exposure disabled, the composed lighting packet and clear colour are cached by clock value; wrapped camera-XZ animation coordinates are recomputed only when the eye XZ changes and patched into the cached packet. Minimum and Fast use this stripped path. The exposure read itself is skipped when the lane is off.
- Disabling weather removes every weather-derived input before uniform composition: coverage, rain palette overrides, and fog bonus all become zero.
- Peer draw records, the audio peer sample, and mod-placement buffers retain capacity across frames. Disabling models avoids animator/rig composition; disabling tags avoids projection, occlusion raycasts, text measurement, and name cloning — a fully hidden peer produces no record at all.
- Idle collision axes return before building an AABB or querying the world; view-direction trig uses fused `sin_cos`.
- The crafting mod mirrors held elements only when the shared stash revision changes and retains the mirror vector's capacity.

### World streaming and LOD

- Horizontal and vertical view distances are independent (`World::set_view_distances`); the minimum `0/1` mesh volume is the current vertical column plus the unmeshed data/collision halo.
- The LOD ladder is configurable and transitions live (`World::set_render_config`): a change frees old GPU-owned sections exactly once, purges queued far jobs, clears derived frontier/cover state, re-arms the new ladder, and advances the section **epoch**. Every asynchronous section claim also carries a unique **token** (`SectionState::Meshing { token }`, `JobKey::Section { pos, epoch, token }`), validated at result integration AND at the moment of upload — an old result, cancellation, failure, or queued upload can never capture a same-position replacement after unload/re-admission.
- The far-field frontier is retained across streaming passes until its exact eye, velocity, ladder, or relief-mip inputs change (`SectionFrontierKey`); edits force a recompute because relief coarsening consults the edit overlay. Per-section band selection uses bounded multiply/compare steps instead of logarithms, with the base-2 step cached in `PyramidCfg`.
- Large camera discontinuities (implausible apparent speed, > 512 m/s) purge queued far work and cancel its exact claims, so obsolete jobs cannot monopolize workers; Ready sections stay resident.
- Sustained travel is load-shed instead of treated as a backlog to overpower. Above 24 m/s one normalized `StreamPacer` scales effort approximately with useful chunk lifetime (`24 / speed`, floor 15%): admission deadlines and progress floors shrink together, near-queue lookahead stays at four jobs per active worker, background concurrency drops immediately, completed-result integration and chunk uploads are bounded, and section uploads no longer grow with queue depth. Leading-edge chunks are motion-prioritized. Capacity recovers exponentially after stopping (0.75 s time constant), preventing a one-frame catch-up avalanche while guaranteeing every lane forward progress.
- Padded voxel/light snapshots and greedy-mesh outputs use bounded cross-thread pools (snapshots are captured on the main thread and dropped by workers — a thread-local pool stranded every buffer). Lighting-off snapshots omit the 18³ light shell entirely, and disabling both lighting and AO selects a culling-only face-sample path; a parity test pins the unlit mesher byte-identical to a full-bright shell.
- Worker jobs share one immutable terrain generator through `Arc` instead of deep-cloning the compiled terrain per job, and skip per-job profiling labels/clocks when profiling is disabled.

Note: the PR-era chunk/section **draw caches** and the temporal **swap fade** were *not* ported — the GPU refactor superseded both. Meshes are resident with placement, detail, visibility, and style pinned at upload; `World::render` submits no per-mesh draws, so there is no per-frame draw walk left to cache.

### Authoritative terrain generation

- Each immutable fBm field precomputes its octave normalization once (`Fbm::new`).
- The height axes intentionally share their warp displacement fields, as do the climate axes; warp coordinates are now computed once per column and shared (`Warp::coordinates`). Raw warped weirdness is reused for ridge shaping and river placement, and identity-gamma controls bypass `powf` (`Control::shape`).
- A generation column uses a fixed 256-profile array instead of a heap allocation, and fBm octave-column state is inline (`MAX_FBM_OCTAVES`) instead of allocating tiny vectors per column.

These are caching and redundant-work removals, not a world-generation redesign. The same authoritative seed/coordinate mapping remains exact — the per-cell/per-chunk/column parity tests and goldens pass unchanged.

### Protocol

The client/server message enums and their binary codec are generated from one `messages!` table over a per-field `Wire` trait ([`src/net/protocol.rs`](../src/net/protocol.rs)): the declaration is the wire format, so encode and decode cannot drift. The byte layout is unchanged (all round-trip, bit-exactness, trailing-byte, and forged-frame tests pass verbatim).

## Benchmark procedure

Headless `#[ignore]` unit throughput benches are driven by [`scripts/bench-unit.sh`](../scripts/bench-unit.sh): one release lib build, then each pinned bench three times with the median compared to the dated pin in its doc comment.
Pass `--json` to also write the same rows as JSON lines to `benchmarks/unit.jsonl`.

Use the release profile in [`Cargo.toml`](../Cargo.toml). Build once so compilation is outside every sample:

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

The harness performs at least a three-second warmup and then waits for `World::entry_complete` before sampling (bounded by `WATT_BENCH_READY_TIMEOUT`, 60 seconds by default). It slowly rotates the camera and emits a compact `BENCH` summary, a one-line `BENCH_MEM` census, and a schema-versioned `BENCH_JSON` record (`schema_version` 3). The record includes CPU topology and power governor, RAM/cgroup limit, renderer-compatible Vulkan GPU and driver inventory, the likely selected GPU, window/render resolution, connected display EDID data, OS/kernel/session, game and renderer revisions, every relevant graphics/streaming setting, frame-time percentiles and hitch counts, RSS start/peak/end, streaming queue/worker peaks, **time to ready** (`scenario.entry_seconds`, also `ready_s=` on `BENCH`; null if the world never settled), and a **memory census** at ready and at end (`memory.census_ready`, `memory.census_end`). Streaming peaks at ready time are the existing `streaming.peaks` object (observed continuously; the ready instant is included). `frames.rendered` and `frames.coalesced` are the deltas of `Engine::frames_rendered` / `Engine::frames_coalesced` across the measured window (monotonic, main thread). `frames.rendered_fps` is that rendered count over `wall_seconds`. `coalesced` is 0 whenever vsync is off (the harness forces vsync off). The compact `BENCH` line includes `rendered_fps=` and `coalesced=`; when `rendered_fps` is below `avg_fps` the line also says `(game frames outran rendered frames)`.

`World::memory_census` walks the loaded maps once (at most twice per run). It reports chunk payload bytes and counts split by uniform / paletted / dense, light-grid bytes (dense cell arrays today; uniform slots stay 0 until compact light storage exists), CPU mesh bytes still held on the world (scratch plus not-yet-uploaded queues — 0 after settle because upload returns `Vec`s to the mesh-output pool rather than keeping them on `World`), edit-overlay bytes, section/LOD bytes, and worklist/queue capacities. `total` is the sum of those byte fields. `scripts/bench-unit.sh` is unchanged.

Set `WATT_BENCH_OUTPUT=benchmarks/results.jsonl` to append the JSON record. `WATT_BENCH_WARMUP`, `WATT_BENCH_READY_TIMEOUT`, and `WATT_BENCH_TAG` control the minimum warmup, readiness ceiling, and run label. On a multi-GPU machine where renderer selection is ambiguous without the window surface, set `WATT_BENCH_GPU` to the observed renderer device; the report records that it was an explicit override rather than silently guessing.

- `WATT_BENCH_MOVE`: +X flight speed in m/s (default 0, static camera).
- `WATT_BENCH_YAW`: steady-rotate rate in rad/s (default 0.4; `0` holds the camera). Reported as `yaw_rate_rad_s`.
- `WATT_BENCH_SCREENSHOT`: `.png` path. After the last measured sample, one extra frame presents and the harness writes that image through the same blocking capture as the golden shots (`taa` stays whatever the run configured; goldens use `taa=false`). Failure prints `benchmark: screenshot failed: …` and the JSON still emits with `"screenshot": <path or null>`.
- `WATT_BENCH_WORLDGEN`: `classic` or `diffusion`; pins worldgen without persisting the mod menu.
- `WATT_BENCH_VISUALS`: `off`/`core` strips Atmosphere/Post/Lighting (core look); `on`/`full` leaves them on.

Use these scenarios:

1. **Minimum target:** `WATT_BENCH_PRESET=minimum`, seed 42, spawn position.
2. **Fast target:** `WATT_BENCH_PRESET=fast`, otherwise identical.
3. **Default/control:** `WATT_BENCH_PRESET=default`, otherwise identical.
4. **Far-coordinate parity:** repeat Minimum and Fast with `WATT_BENCH_POS="1000000,128,-1000000"`; this path receives two extra warmup seconds.
5. **LOD stress:** Custom profile with distant LOD enabled, record distance/vertical/levels/detail explicitly, then use the same seed and position.
6. **Attribution only:** repeat a representative failure with `WATT_BENCH_PROFILE=1`. Never compare this run directly with the uninstrumented target.

Keep `VOXEL_PROFILE` unset and `WATT_BENCH_PROFILE` absent for headline numbers. The harness forces VSync off and the FPS cap to zero, and the JSON record captures the reproducibility metadata above. Compositor occlusion and thermal state remain external facts; put them in `WATT_BENCH_TAG` when they differ between samples.

The `WATT_BENCH` harness is a steady rotating-camera test, not a traversal result. The separate `stress` binary deterministically crosses chunk boundaries at 64 and 200 m/s, reports flight/settle hitch percentiles, convergence time, upload/light/mesh and worker-queue peaks, plus the minimum adaptive effort/worker allowance. Run it with `cargo run --release --bin stress`. Validate frame semantics with render-thread/GPU counters before treating application-frame throughput as presentation throughput.

## Prioritized remaining opportunities

### P0: sibling `voxel-engine`

Unchanged from the original audit (`../voxel-engine`): stop cloning completed draw lists in `finish_frame`; make stripped render lanes structurally absent (shadow PCF, bloom, HDR/tonemap); reject/coalesce before frame construction under the latest-frame mailbox; move allocator maintenance off every frame; replace the exposure reducer with a hierarchical reduction; cache view/projection/frustum construction by orientation/lens/viewport; add a true water-effects-off path. The engine's own test target currently fails to compile (`vk/taa.rs`), independent of this repository.

### P1: game and world

1. Split streaming into a cheap result/upload pump and an event/rate-driven topology pass, so a 15 Hz Minimum profile publishes finished work promptly without repeating admission bookkeeping.
2. Move multiplayer transport polling behind a bounded wake or event path.
3. Broaden lighting-packet caching to mixed configurations (revision-key static components; isolate cloud/water/exposure updates).
4. Add compact lighting-on snapshot variants (`all dark`/`all bright`) so uniform lit chunks avoid the padded copy while voxel lighting remains enabled.
5. A fused far-section extract+mesh path that samples the generator directly into one pooled dense buffer (bypassing the per-section RLE brick build), behind a byte-parity oracle against the current `Section::extract` → `build_section_mesh` path.
6. Batch terrain-noise evaluation across columns. SIMD or reassociation is allowed only if an authoritative byte-parity test proves it does not alter generated blocks.
7. Cache name-tag text/measurement and share peer names (`Arc<str>`) for the models/tags-on multiplayer case.
8. Budget mod edge bursts: a time budget with an ordered continuation would bound third-party hook cost without losing or reordering actions.

## Correctness constraints

- **World generation is authoritative.** For a fixed worldgen version, seed, registry, and coordinate, generated blocks must remain identical. Caching may remove duplicate evaluation but must not change hash streams, sample coordinates, floating-point ordering, placement precedence, save replay, or server/client agreement.
- **LOD remains a consecutive base-2 hierarchy.** Detail is 2–6, levels are 1–8, coarsest detail never exceeds 9, near chunks own the clipped inner volume, and covering logic may show the selected level or its one-level-finer hysteresis fallback — never a hole. Live changes must free each GPU handle exactly once and discard stale asynchronous results (the epoch/token machinery).
- **Optimization gates cannot change core playability.** Minimum must retain input, camera, collision, mining, authoritative edits, readable terrain, console access, and an explicit clean-exit save. Placement/inventory hooks are recoverable by enabling `mod_logic`, `mod_hud`, and a master HUD mode that exposes mod UI; disabled optional systems must perform no hidden periodic work.
- **Measure the optimized program, not the profiler.** Headline runs are release, uncapped, unsynced, fixed-scene, and uninstrumented. Golden/worldgen/state-machine tests establish correctness; they do not substitute for a benchmark, and no FPS target is considered met until a reproducible run records it.
