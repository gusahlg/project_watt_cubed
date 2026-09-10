# Architecture audit — 2026-09-10

Game crate `src/` only. Counts: `use crate::X` statements (not comments).
Edits excluded `src/world/**` and diffusion files; those are report-only.

## 1. Map

### Incoming `use crate::` (who is imported)

| target | n | target | n | target | n |
|---|---:|---|---:|---|---:|
| block | 79 | world | 73 | mods | 21 |
| net | 19 | render_config | 19 | menu | 17 |
| ident | 17 | input | 16 | player | 16 |
| coord | 16 | audio | 14 | ui | 13 |
| settings | 12 | math | 12 | sky | 7 |

`world` is the sink (`world/mesh/mod.rs:8` `use crate::block::…`; 33 `block` uses from `world/`). `game.rs:11-38` is the hub (input, audio, world, mods, net, …). Bins: `main.rs:3` `app`, `bin/watt_server.rs:19-21` `net`/`paths`, `bin/golden.rs:17` + `bin/stress.rs:20` `harness`.

### God files (> 1500 lines)

| file | lines |
|---|---:|
| `src/world/streaming.rs` | 3502 |
| `src/world/generation.rs` | 2482 |
| `src/net/server.rs` | 2452 |
| `src/settings.rs` | 2254 |
| `src/world/mod.rs` | 2230 |
| `src/world/tests.rs` | 2162 |
| `src/world/pipeline.rs` | 1858 |
| `src/game.rs` | 1658 |
| `src/audio/runtime.rs` | 1622 |
| `src/harness/mod.rs` | 1501 |

### Cycles (`use crate::`, remaining)

| cycle | evidence | break? |
|---|---|---|
| camera ↔ player | `camera.rs:8` `Player`; `player.rs:6` `Orientation` | methods on camera take pose fields, not `Player` |
| coord ↔ world | `coord.rs:11` `CHUNK_SIZE`; `world/chunk.rs:14` owns it | move `CHUNK_SIZE` to `coord` (needs `world/**` edit) |
| audio ↔ world | `audio/director.rs` `use crate::world`; `world/query.rs:9` `AcousticWindow` | move acoustic query to `audio` |
| menu ↔ mods | `menu/mod.rs:6` `Mods`; `mods/mod.rs:25-26` start/theme | intended plugin seam |
| mods ↔ save | `mods/mod.rs:665` `write_atomic_file`; `save/bridge.rs:9` `Mods` | persist helper outside `save` |
| sched ↔ world | `sched/mod.rs:17` `World`; `world/lanes.rs:1` `sched::Run` | keep; scheduler is world-typed |
| block ↔ macros | `block/element.rs:12` `elements!` | keep; macro expands into `block` |

**Broken this task:** `game` ↔ `harness`. `CameraPose` now `camera.rs:104`; `DebugView`/`SKY_KEY`/`TERRAIN_KEY` now `game.rs:48-59`. `game.rs` no longer `use crate::harness`.

Doc-only `crate::app` in `game.rs:3` is not a use-cycle.

### Core-vs-mod leaks (Mods v2: core must not name a specific mod type)

- `WorldgenKind::Diffusion` is a core enum (`world/generation.rs:53-57`).
- `DiffusionCfg` is a core struct used on the wire (`net/protocol.rs:81-95`), in saves (`save/bridge.rs:12-13`), and in the app (`app.rs:29`, `app.rs:903`).
- `VisualGroup::mod_name` returns `"Atmosphere"`/`"Post"`/`"Lighting"` (`render_config.rs:88-93`).
- Specific mod structs (`InventoryMod`, `InfiniteDiffusionMod`, …) stay under `src/mods/` — good.

### Engine knowledge duplicated in the game

- Minimap size 256 asserted in game (`minimap.rs:72`); engine `MINIMAP_SIZE` is not queried.
- Block layer 16² (`block/texture.rs:22`); engine layer-0-white contract in the same file (`block/texture.rs:8-10,38`).
- Bench opens a **second** Vulkan instance (`benchmark/system.rs:261-269`) instead of `Engine` caps.
- Good: VRAM estimate already asks the engine (`render_config.rs:347-355` `estimate_from_engine`).
- Mesh verts are engine `MeshVertex` (`world/mesh/mod.rs:23`); packing is not re-derived in game.

### Error handling (prod ≈ not inside `mod tests` / `tests.rs`)

| style | where |
|---|---|
| `Result` | save IO (`save/store.rs:104,120`), net (`net/server.rs:459` `run`), audio init, mods.cfg (`mods/mod.rs:660`) |
| `expect`/`unreachable` | invariant (`settings.rs:1018` Custom, `settings.rs:1396` `write!` to `String`) |
| panic in tests | `console.rs:157`, `game/draw.rs` tests |
| silent `let _ =` | net `try_send` (`net/server.rs:802`), audio queue-full, test `fs::remove_file` |
| was-swallow write/rename | now `log_fs_err` (`save/store.rs:128`; callers `session.rs:54-56`, `settings.rs:1074-1076`, `store.rs:173-177`) |

`#[allow(dead_code)]`: only `HeightMip::occludes` (`world/heightmip.rs:166-167`). Not deleted (world excluded; still the draw-cull stub).

## 2. Ranked structural changes

Each: goal, size, risk, files, verify. Risk: L/M/H. Size: S <200, M 200–800, L >800 loc.

1. **`game.rs` split by phase (M, L-risk if order slips).** Methods already exist (`game.rs:713` net, `:740` input, `:806` overlay, `:936` motion, `:987` interact, `:1075` stream, `:1126` audio; driver `game.rs:634-676`). Move each to `game/<phase>.rs`, keep `Game::update` order and `FramePhases` timing. Test: existing `game.rs` benches/tests + `--lib`. Do not change sim/stream order.

2. **`settings.rs` split, keep the descriptor table (M, L).** Table is already the source (`settings.rs:593` `[Setting; 52]`). Remaining: 2254-line file, tests from ~1396, `apply` (`settings.rs:1261`), runtime-only fields (`settings.rs:178` `muted`, `:130` cull). Extract `settings/apply.rs` + `settings/tests.rs`. Fold `muted` into `SETTINGS` or document why not. Test: `save_and_load_use_the_config_root` (`settings.rs:1637`), `/gfx` tests.

3. **`app.rs` screens as modules (M, M).** `Screen` is two arms (`app.rs:52-55`) but enter/load/host/join/bench live on `App` (`app.rs:733` `load_world`, `:760` `enter_game`). Split `app/play.rs`, `app/host.rs`, `app/bench.rs`. Keep `App::run` (`app.rs:203`) the only `voxel_engine::run` caller. Test: lib tests that construct `App` paths; do not run the game binary.

4. **Save trio: keep format pure, shrink bridge (S, H if bytes change).** `format.rs:1-3` is already `SaveDoc`↔bytes; `bridge.rs:1-2` is the only game-typed file; `store.rs` is IO. Remaining leak: bridge names `DiffusionCfg`/`WorldgenKind` (`save/bridge.rs:12-13`). Replace with opaque `worldgen: (id, cfg-text)` once mods v2 lands. **Bit-identical saves required.** Test: `save/mod.rs` roundtrips, salvage tests in `store.rs`.

5. **Net: shared connection helpers, not a second codec (M, H if wire changes).** Codec is already one table (`net/protocol.rs:25-32,207-227` `encode`/`decode`). `client.rs` (1004) and `server.rs` (2452) still duplicate QUIC lifecycle, voice rings, interest. Extract `net/link.rs` (stream read/write already in `protocol.rs:343-386`). **Protocol bytes frozen.** Test: `net/protocol.rs` codec tests, server join tests.

6. **Worldgen out of core (L, H).** `WorldgenKind`/`DiffusionCfg` in core (`world/generation.rs:53`, `net/protocol.rs:81`, `app.rs:29`) violate Mods v2. String id + opaque cfg (design `documentation/notes/PM-MODS-V2-DESIGN-2026-09-09.md`). Save/protocol bump. Needs `world/**` + diffusion edits. Test: fingerprint tests (`net/mod.rs:286-299`), save worldgen roundtrip.

7. **`CHUNK_SIZE` lives in `coord` (S, M).** `coord.rs:11` imports from `world/chunk.rs:14` — inverted. Move the const, re-export from `chunk` for one release of in-crate paths. Needs `world/**`. Test: coord split/join tests (`coord.rs` tests).

8. **`net/server.rs` split (M, M).** 2452 lines: accept, validate, broadcast, voice. `hooks.rs` is already the mod seam (`net/mod.rs:17`). Split `server/accept.rs`, `server/edits.rs`, `server/presence.rs`. Keep `run` (`net/server.rs:459`) and `Config` (`:107`) the bin API (`bin/watt_server.rs:19-20`). Test: server unit tests in that file.

9. **`audio/runtime.rs` split (M, M).** 1622 lines: mixer + tests. Extract `audio/runtime/mix.rs` vs tests. Do not change mixer timing. Test: runtime tests in that file.

10. **Engine queries for remaining magic sizes (S, L for game; needs engine API).** `minimap.rs:72` 256; `benchmark/system.rs:261` second Vulkan instance. Engine should expose `minimap_size()` and `gpu_caps()`. Game half is tiny once the engine methods exist. Do not hard-code HDR/FIF (already `estimate_from_engine`, `render_config.rs:347`). Test: minimap ctor, bench metadata tests.

## 3. Do-now (this change)

- `save::log_fs_err` (`save/store.rs:128-130`) logs write/rename `Err` outside hot paths. `NotFound` on best-effort bak follow is still silent (`store.rs:173-177`).
- Crate surface: `lib.rs:5-41` `pub` only `app`, `harness`, `net`, `paths` (bins). Nested net client/hooks/protocol/chat `pub(crate)` (`net/mod.rs:16-18,304`). Unused harness detectors `pub(crate)`. Trimmed unused `pub use` re-exports (`audio/mod.rs`, `block/mod.rs`, `menu/mod.rs`, `save/mod.rs`, `sky/mod.rs`).
- Moved `CameraPose` to `camera.rs:104` and `DebugView` to `game.rs:54` (cycle 1).
- Did not delete `HeightMip::occludes` (`world/heightmip.rs:166`) — `src/world/**` excluded. Remaining unused-item warnings (`ConfirmDialog`, `Precip::Snow`, `Face::positive`, `Render` trait, `world/light.rs` helpers) are unused-but-wired API or live under `world/**`.
