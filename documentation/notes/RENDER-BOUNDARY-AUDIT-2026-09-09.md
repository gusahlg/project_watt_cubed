# Render-boundary audit (game side) — 2026-09-09

Game-side half of a joint game/engine audit: what rendering-shaped work
lives in `project_watt_cubed`, what it costs, and what should move into
`voxel-engine`, change shape, or stay. Report only. Engine cited
read-only. This worktree does **not** contain task 40's VRAM estimator
(`render_target_bytes`); that lives on sibling commit `12f42c5` and is
cited as such.

Live GPU/CPU µs were **not** sampled (game binary is fullscreen; no
`VOXEL_PROFILE` run). Numbers below are type sizes, documented
budgets, and the stress-harness peaks in `src/bin/stress.rs`.

## A. Inventory

| call / data | game | engine | freq | bytes | notes |
|---|---|---|---|---|---|
| `voxel_engine::run` + `Config` | `app.rs:154-169` | `engine.rs:24-45,527` | once | Config ~flags+window | `flags` from `Settings::render_config().engine_flags()` |
| `begin_frame(clear)` → `Frame` drop → `finish_frame` | `game/draw.rs:85-88` | `engine.rs:450-471` `frame.rs:372-380` | /frame | `Box<DrawLists>` pointer move (pool of FIF+1=`3`, `render_client.rs:147`) | **not** cloned per frame. Capacities persist (`frame.rs:140`). |
| compose lighting → `Lighting::Composed(FrameUniformsGpu)` | `game/draw.rs:122-200,294-297` `frame_snapshot.rs:140-156` | `frame.rs:238-268` `gate_uniforms` `frame.rs:76-103` | /frame (cached when weather+clouds+water_anim+exposure all off, `draw.rs:168-207`) | 128 B public UBO (`build.rs:1221-1256`, 8×`[f32;4]`) + 32 B engine tail | Engine always writes UBO; jitter/view_proj rebuilt every `begin_3d` |
| `exposure_for_compose` | `draw.rs:135-139` | `vk/exposure.rs:565-570` | /frame if `render.exposure` | 4 B `Exposure` | Skipped when lane off |
| `Sky::draw` → `set_sky(SkyDesc)` | `sky/mod.rs:81-86` `draw.rs:301-302` | `frame.rs:394-399` | /frame if sky on | 28 B (`Vec3`+`LinearRgb`+`f32`) | No engine dirty-flag; `DrawLists.sky` overwritten every frame |
| `World::render` → `set_lod_clip` only | `world/mod.rs:1110-1121` | `frame.rs:388-392` | /frame | 8 B `CoverageVolume` | **No per-mesh draws.** Terrain/LOD are resident; GPU cull emits indirect cmds (`vk/cull.rs:1-4`) |
| `upload_mesh_placed` (chunks) | `streaming.rs:847-848` `drain_results:648-670` | `engine.rs:351-357` `render_client.rs:327-348` `buffers.rs:400-509` | on change, byte-budgeted | verts: 8 B×N (`mesh.rs:3,120-126`); **indices not staged** (shared quad IBO). Game *charges* verts+indices (`streaming.rs:36-49`, pinned `world/tests.rs:820-841`) | Worker `MeshOutput` moves to queue (no extra memcpy). Main thread **permutes** verts by bucket into staging (`buffers.rs:419-434`); discrete GPU then `cmd_copy_buffer`. Unified: one host write into mapped device mem. Engine transfer window 8 MiB (`buffers.rs:23`); game half is 4 MiB (`world/mod.rs:104`) |
| section/LOD tile upload | `world/mod.rs:444-472` `streaming.rs:693-720` | same `upload_mesh_placed` | ≤2/frame (`mod.rs:134`) | same 8 B verts; 2×2 quadrants of 16³ blocks | `Light::DAY` baked, not a light field (`section/mesh.rs:25-26,394-395`) |
| `free_mesh` | `world/mod.rs:353` `edits.rs:235` | `engine.rs:360-362` | on unload/edit/config | handle 8 B | Deferred GPU-side |
| `set_visible` (occlusion / LOD mask) | `mod.rs:396-399,1438-1448` `streaming.rs:849-855,711` | `engine.rs:365-367` `render_client.rs:382-403` | on change (word-batched) | 8 B/word | GPU frustum cull is separate and unconditional |
| `set_mesh_style` | `mod.rs:430,714` | `engine.rs:370-383` | on section material change; skipped if equal (`mod.rs:435-437`) | 8 B `DrawDyn` | |
| CPU occlusion BFS | `connectivity.rs:1-15` `mod.rs:1342-1428` `lanes.rs:114` | `set_visible` only | ≤0.5 ms/pass, 100 ms topo debounce (`mod.rs:119-128`) | connectivity `u16`/chunk; mask walk of live chunks | Cave/wall visibility, **not** Hi-Z. Engine has no Hi-Z (`vk/cull.rs` is AABB+frustum). `HeightMip::occludes` is dead (`heightmip.rs:166-167`) |
| `set_flags` / MSAA / scale / vsync / fps / fullscreen / `set_cull_faces` | `settings.rs:1039-1051` `app.rs:331-332` (menus **every frame**); in-world only on entry/`/gfx` (`game.rs:391-395,1276-1279`) | `engine.rs:215-286` | menus: /frame; play: on change | `RenderFlags` ~15 bools | `RenderClient::set_flags` **always sends** (`render_client.rs:483-485`), no equality check. `set_cull_faces` is **inert** (`render_client.rs:470-477`) |
| HUD 2D immediates | `draw.rs:344-443` `ui.rs:116-123,234-255,337` `console.rs:85-116` | `frame.rs:284-369` | /frame | `Vertex2D` 20 B (`pipeline.rs:108-112`) | At rest Full HUD, no peers, empty stash: ~120 glyph quads (coords+FPS+inventory header/empty, all shadowed) + 1 panel rect + 4 crosshair lines + minimap (6 tex verts + 7 lines) ≈ 0.8k verts ≈ 16 KB. Regenerated every frame into pooled `verts_2d` |
| avatar immediates | `avatar.rs:145-154` `draw.rs:310-337` | `frame.rs:412-486` | /frame per visible body | `DebugVertex` 16 B; 6 boxes × 36 verts + 6 shadow verts ≈ 222×16 ≈ 3.5 KB | First-person local body skipped (`draw.rs:315-318`) |
| minimap CPU raster + upload | `minimap.rs:45-53,102-168` `game.rs:1042-1052` | `engine.rs:427-429` `render_client.rs:427-430` `vk/minimap.rs:10-18` `vk/mod.rs:216` | 500 ms **or** 16-block recenter; draw /frame | 256²×4 = 262144 B; `update_minimap` **clones** to `Box<[u8]>` | Engine already version-gates per FIF slot. Game always full-frame rescan, no dirty rects |
| `set_block_textures` | `streaming.rs:2272-2296` `block/texture.rs:22,36-49` | `engine.rs:422-424` `render_client.rs:420-424` `block_textures.rs:1-2,50-80` | palette growth (entry / new block id) | 16²×4 = 1024 B/layer; **all visible layers cloned** then GPU-idle full-array rebuild + CPU mips | Incremental `texture_cache` on game; engine API is replace-all |
| screenshot | `game.rs:819` (F2) `harness/mod.rs:903` | `engine.rs:400-407,487-497` | on demand | presented framebuffer | Interactive fire-and-forget; harness `screenshot_to` blocking |
| Vulkan device probe | `benchmark/system.rs:255-368` | engine has `MemoryBudget` (`vk/device.rs:4-32`) **not** on `Engine` | bench collect | heap sizes | Separate ash instance; guesses selected GPU; duplicates engine feature/extension checks |
| VRAM estimator | **not in this tree**; sibling `12f42c5` `render_config.rs` `render_target_bytes` | `targets.rs:17-29` HDR=`R16G16B16A16_SFLOAT` 8 B/px, depth `D32` 4 B, FIF=2, shadow 2048²×2, bloom mips, `SKY_CLOUD_LUT_SIZE=256` | session start (task 40) | derived | Game hard-codes 16 B/px HDR (padding), FIF, shadow, LUT — engine-format mirror |
| `world_to_screen` (name tags) | `draw.rs:498` | `engine.rs:439-446` | /frame per visible peer | 8 B `Vec2` | |
| profile scopes | `draw.rs:301,304,351` stream meters in `streaming.rs:605` `sched/mod.rs` | `profile.rs:38-117,582-624` | /frame if `VOXEL_PROFILE` | atomics | Header: `profile {n}f {ms}/f {fps}fps … \| cpu {c} (sim {s} list {l} submit {u}) wait {w} (main {m} render {r}) gpu {g} work {k}` |
| camera | `camera.rs:75-80` caches `Camera3D` by yaw/pitch/roll/FOV bits (`draw.rs:104-113`) | `frame.rs:247` `view_proj` every `begin_3d` | /frame | `Camera3D` | Game avoids trig; engine rebuilds matrices |

Default view: distance 6 / vertical 3 (`settings.rs:118,129`) → mesh box 13×13×7 = **1183** chunks. `lod2` ships **off** (`settings.rs:148`) even though `RenderConfig::default` has it on (`render_config.rs:82`).

## B. Duplication and knowledge leaks

| fact | where | keep / move / query |
|---|---|---|
| Vertex = 8 B packed; 6 index buckets | game mesher emits `MeshVertex` (`world/mesh.rs:475`); budget charges verts+**indices** (`streaming.rs:38-49`) but engine stages **verts only** (`buffers.rs:399,412`) | **Keep** packing in engine (`mesh.rs`). **Fix** game budget to vertex bytes (or engine returns staged size). Do not re-derive shifts in game. |
| Shared quad IBO / permutation | `buffers.rs:419-434` on main at upload | Engine-owned. Game must not assume indices are GPU bytes. |
| FIF=2, HDR 8 B/px, D32, 2048²×2 shadows, bloom mips, 256² cloud LUT | task 40 `render_target_bytes` (sibling `12f42c5`); engine `targets.rs:17-29` `buffers.rs:46` `genconst.rs:63` | **Move to engine query** `estimate_targets(extent, msaa, scale, flags)` + `gpu_caps()`. Game must not hard-code formats. |
| `MINIMAP_SIZE=256` | game asserts (`minimap.rs:71`); engine `vk/mod.rs:216` (not public) | **Expose** `Engine::minimap_size()`; drop the magic 256. |
| `FRAMES_IN_FLIGHT` | not in this tree's game code; task 40 copies it | Engine query. |
| MSAA list 1/2/4/8 | `settings.rs` `MSAA` | Keep as UI steps; clamp already via `set_msaa` (`engine.rs:248`). |
| Present-mode / vsync | `app.rs:204-213` single writer; engine mailbox (`engine.rs:30-36`) | Keep. Game must not assume FIFO vs mailbox. |
| GPU memory heuristics | bench probe (`system.rs:274-278`) vs engine `MemoryBudget` | **Engine** `gpu_caps().device_local_bytes` / live budget. Drop second Vulkan instance. |
| Projection / frustum | engine `Camera3D::view_proj` + GPU AABB cull | Keep in engine. Game already rebases eye to origin (`camera.rs:18-19,75`). |
| Face-run culling (`set_cull_faces`) | settings toggle still pushed (`settings.rs:1048`) | **Engine or retire.** Currently a no-op (`render_client.rs:470-477`). |
| `RenderConfig` ↔ `RenderFlags` | `render_config.rs:176-194` | Keep mapping in game (occlusion/lod2/clouds/weather/day_night are **not** engine flags). |
| Layer-0 white / 16² RGBA8 | `block/texture.rs:8-10,22` `engine.rs:419-421` | Keep contract; engine should document `TEXTURE_SIZE` if it starts caring. |
| `Pass::COUNT=3` | `coord.rs:230` `mesh.rs:277-290` | Keep; engine enum is the source. |

## C. Ranked proposals

Each: what / save / risk / owner. Yes/no with evidence.

1. **Zero-copy mesh uploads — YES (engine API, game workers).** Workers already emit engine `MeshVertex`. Main thread still permutes into staging (`buffers.rs:419-434`) and walks AABB (`468-472`). Peak flight: `upload_queue` 45 (`stress.rs:18`); budget 4 MiB/frame (`mod.rs:104`) ≈ up to ~0.5M verts/frame × 8 B. Saving the permute+AABB is the **largest remaining main-thread upload cost** (hundreds of µs on a 4 MiB drain; a few ms on a hitch-sized burst). Engine owns mapped staging; game submits handles only. Risk: medium (permutation invariant, same-frame drawability). **Engine owns API; game changes mesher output order.**

2. **Engine Hi-Z for draw visibility, game keeps CPU frontier — YES, split.** Engine already GPU-frustum-culls resident meshes (`vk/cull.rs:1-4`); game BFS is *cave connectivity* (`connectivity.rs:1-6`), patched via `set_visible`. Hi-Z would cut overdraw the BFS cannot (hills from outside, far LOD). Game BFS should stay for streaming priority / not-drawing sealed caves. `HeightMip::occludes` (`heightmip.rs:166`) is the unused CPU cousin — do not wire it if Hi-Z lands. Save: GPU fill-rate, not main-thread (occlusion lane already capped at 0.5 ms). Risk: medium (false-hide = holes). **Engine owns Hi-Z; game owns BFS.**

3. **Persistent engine-side draw sets — NO for terrain (already done); maybe HUD.** `World::render` submits no per-mesh draws (`mod.rs:1105-1109`). `finish_frame` moves a pooled `Box`, does not clone lists (`engine.rs:455-463`). Remaining rebuild is 2D/debug immediates (~16 KB HUD). Incremental mesh add/remove is the current `UploadMesh`/`FreeMesh`/`SetVisible` stream. Do not rebuild that.

4. **Minimap as engine texture + dirty rects — PARTIAL.** Engine already has a dedicated 256² texture, version-gated (`vk/minimap.rs:16-18`). The cost is the **CPU column scan** (`minimap.rs:126-164`), not the 256 KB upload (cloned once, `render_client.rs:427-430`). Dirty rects help recenter-only updates; a heading-up rotate is a draw-time UV, already free. Save: ~scan of 64k columns / 500 ms, not µs/frame. Risk: low. **Game raster stays; engine could accept a subrect to skip the clone of unchanged texels.**

5. **Retained HUD text batches — NO for in-world rest; maybe menus.** Rest HUD is ~120 glyphs (`ui.rs` shadowed × coords/FPS/inventory). Pooled `verts_2d` already keeps capacity. Menus call `settings.apply` every frame (`app.rs:331-332`) and redraw every label (`menu/theme.rs:43`). Save in-world: <10 µs. Menu: more, still small vs worldgen. Risk: low, complexity high vs gain. **Keep immediates; if anything, dirty-flag `set_flags` (see 8).**

6. **`gpu_caps()` / `estimate_targets()` — YES.** Task 40 mirrors engine formats in game (sibling `12f42c5`). Engine already knows HDR/depth/FIF/shadow/bloom/LUT (`targets.rs`) and `MemoryBudget` (`device.rs:4-32`) and `DeviceCaps` (`render_client.rs:40-44`) but exposes only `max_msaa` / `max_texture_array_layers` (`engine.rs:256-265`). Bench probe (`system.rs:255`) is a second Vulkan instance. Save: correctness (OOM), not frame time. Risk: low. **Engine owns.**

7. **Block-texture layers uploaded by the registering worker — NO.** Growth is rare (`streaming.rs:2270`: entry / new id). Cache is already incremental (`2277-2282`); the engine call still clones **all** layers and rebuilds the array while GPU idle (`engine.rs:418-419`, `render_client.rs:420-424`). The missing API is **append-layer**, not a worker. Worker would race the bind used by in-flight meshes. **Engine: incremental layer upload. Game: keep building pixels on the thread that owns the registry (main is fine).**

8. **Sky/atmosphere structs pushed only on change — PARTIAL, game already does the cheap case.** Stripped profiles cache `FrameUniformsGpu` (`draw.rs:168-207`). Engine does **not** de-duplicate: `begin_3d` always `gate_uniforms` + stores scene; `set_sky` always overwrites (`frame.rs:398-399`); `set_flags` always queues (`render_client.rs:483-485`). Remaining waste: (a) Default profile recomposes palettes every frame (`frame_snapshot.rs:68-137`) — a few µs of lerps; (b) **every menu frame** `Settings::apply` → `set_flags`. Dirty `set_flags` is the easy win. UBO write itself must happen whenever camera/jitter moves. **Engine: de-dup `SetFlags`. Game: optional compose dirty for weather/clouds.**

9. **LOD tile lighting GPU-side / heights-only — NO for now.** Far verts already store `Light::DAY` (`section/mesh.rs:394-395`) so the existing sun/ambient UBO lights them. Heights are already consumed to split vertical faces (`section/mesh.rs:3-4`). A GPU height-field light would not remove the mesher's occupancy walk. Save: none on the critical path until far tiles need per-column sky occlusion. **Keep CPU `Light::DAY`.**

10. **Other (from inventory).** (a) Charge upload budget for **staged vertex bytes only** — game currently over-counts 4 B×6 indices/quad that never hit the transfer lane. (b) Retire or implement `set_cull_faces`. (c) Expose `minimap_size`. (d) Stop the extra minimap `to_vec` if the engine can borrow until the render thread copies. (e) `World::render`'s `ListWorld` scope is a lod-clip store + a gauge — keep the meter, don't hunt a draw walk that isn't there.

**Priority order to actually do:** 6 (caps/VRAM query) → 1 (zero-copy staging) → 8's `SetFlags` de-dup → 2 (Hi-Z, engine) → 4's subrect → 7's append-layer. Skip 3 (terrain), 5 (in-world HUD), 9.

## D. Numbers

Profiler line (engine `profile.rs:582-624`), when `VOXEL_PROFILE` is set:

```
profile {frames}f {ms}/f {fps}fps worst {ms} | cpu {c} (sim {s} list {l} submit {u}) wait {w} (main {m} render {r}) gpu {g} (p50 … p95 …) work {k} | rendered {r} presented {p}
```

This session did not print one (binary not run).

Derived / documented:

- `MeshVertex` 8 B; `Vertex2D` 20 B; `DebugVertex` 16 B; `FrameUniformsGpu` 128 B; GPU UBO 160 B; `MeshHandle` 8 B.
- Chunk upload budget 4 MiB/frame (`world/mod.rs:98-104`), floor 256 KiB (`streaming.rs:114`); engine copy window 8 MiB (`buffers.rs:23`). Queue cap 96 (`mod.rs:114`).
- Stress peaks (comment, 2026-07-19, `stress.rs:14-18`): `r20_200mps` p50 12.47 / p95 18.25 ms, `upload_queue` **45**, `light_apply` 1793. Post-pacer code still uses those as the published reference; not re-run here.
- Default resident set ≤ 1183 chunk columns × ≤2 passes, plus LOD only if `lod2` on (shipped off).
- Minimap upload 262144 B × ~2/s at rest (500 ms) = ~0.5 MB/s, plus a same-size clone on the command.
- Block layer 1024 B; full replace clones `visible` layers (device cap, often 2048 → 2 MiB) then mip-chains CPU-side.
- Rest HUD ~16 KB 2D + 0 3D terrain list + optional 3.5 KB/avatar.
- Task 40 example (sibling, not this tree): 3440×1440 × scale 2 × 8×MSAA + TAA+bloom `render_target_bytes` > 4.8 GB.

`World::memory_census` (`census.rs:18-21`) confirms: after settle, CPU mesh bytes are scratch + not-yet-uploaded queues (often 0); GPU copies out of `MeshData` at upload.
