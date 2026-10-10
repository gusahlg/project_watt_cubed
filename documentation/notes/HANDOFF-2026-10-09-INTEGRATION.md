# Integration and validation — 2026-10-09

Working-tree integration based on main `139ebaa`. No commits, branch resets, pushes, or changes to the other worktrees were made. Origin/main remains at release 0.3.4 (`d8cbf49`); the local main already contained 52 subsequent commits before this work.

## What was already on main

Recent work consolidated chunk and section ownership, LOD coverage and scheduling, asynchronous world entry, bulk save loading, server throughput and protocol 17, client/server policy fixes, and allocation reductions in audio and menus. The active architecture branches are pieces of that same rewrite rather than separate product features. Historical July/September audit documents describe older experimental histories; they are not evidence about today's main.

## Source integrations

| Branch | Revision | Integrated behavior |
|---|---|---|
| arch-p3 | 594bdde | Indexed lighting checks, unload bookkeeping, lit edit remeshing, residency and window reuse |
| arch-p6 | fd24933 | Shared spec table and rotating file recovery without per-edit spec copying |
| arch-p7 | 09f660b | Coherent worker view snapshots, renderer mesh staging, report-time distributions |
| arch-p10 | 30f6de8 | One streaming pass shared by the game, headless fixtures and benchmarks |
| mp-fix | bcd4308 | Multiplayer session, world, presence and chaos regressions |
| emergent-lab | b447ce1 | Field crate, emergent cosmos prototype and laboratory, gated behind dev-tools |

The emergent prototype is available for measurement without replacing shipping terrain generation. Enabling `dev-tools` makes `cosmos_lab` available; ordinary game builds retain their generator, content and save identities. The renderer lock now records version 0.2.8 and Nix pins its committed revision `175149489441aa483808c918b46462cde5706885`.

## Corrections found during integration

- Bulk edit installation now updates the roof index introduced by arch-p3. The per-edit/bulk equivalence test also compares that index.
- Refused prediction chains wait for their unresolved requests and unwind newest first. Rollback uses the authoritative revision observed at prediction time, separately from the speculative wire revision. This prevents a refused follow-up edit from undoing a tool result or peer authority. A Fast-drafted regression reproduced the tool/edit failure before the correction; a second regression checks that edits made after observing authority still restore their current baseline.
- Accepted edits send authoritative content before their verdict, allowing reconciliation after a rival/tool overwrite or local timeout. Confirmed requests are not counted twice when computing a new speculative revision. Older delayed snapshots cannot replace newer content.
- Snapshot traffic and budgeted overlay application keep the joining link alive. Transport idle timeout uses the same interrupted-link explanation as the watchdog.
- Refused and rate-limited clock requests return the authoritative clock. Refused tool use names the current generated or edited cell.
- Enabled mods that cannot fit the join offer fail locally with an explicit explanation.
- Axis collisions advance to the voxel face instead of marking a hovering body grounded. Headless test passes use explicit simulation time while the game and timed benches retain the real clock.
- Lower cruise declarations follow the final move under the old cap and wait for deceleration. Movement credit preserves its fraction when speed changes, providing bounded slack at high speed without refilling spent credit.
- Quoted operator names support spaces without changing the existing `name secret` authentication meaning. Malformed quoted lines grant no permission.
- The audio generation tool uses a feature-gated public random-function export instead of a private library module.
- Compatible lock updates: rustls 0.23.45, rustls-webpki 0.103.15, rtrb 0.3.5, chacha20 0.10.2, wayland-scanner 0.31.11 and quick-xml 0.41.0. These address the advisory findings without disabling checks. See [rustls](https://rustsec.org/advisories/RUSTSEC-2026-0285.html), [rtrb](https://rustsec.org/advisories/RUSTSEC-2026-0274.html) and [quick-xml](https://rustsec.org/advisories/RUSTSEC-2026-0194.html).

## Local OpenCode / Qwen workflow

The existing `qwen3-coder` launcher manages the model service. Selecting an OpenCode model name alone does not reload the service: it initially accepted the Fast alias while still serving Q4. Running `qwen3-coder start fast` loaded Q3_K_S, verified through `/props` and its actual model path. Fast is used for small requests; Quality was also loaded for the larger physics/streaming investigation.

Use `qwen3-coder start quality|fast|solo` to load the profile, then select the matching `llama-rpc/qwen3-coder-30b-a3b[-fast|-solo]` in OpenCode. Verify `/v1/models` before benchmarking profile differences. The quality and fast profiles use the same model family with different quantization; solo changes GPU placement too.

For simple work, give a small, self-contained task and the exact relevant source. An isolated `patch` agent with all tools disabled reduced prompt context from thousands of tokens to a few hundred and avoided repeated repository reads and context overflow. Use `--pure` and `--format json` for repeatable subprocess requests. Provide the positional message before a variadic `--file` argument. Isolated XDG directories under /tmp avoided changing the user's global OpenCode setup.

Fast was responsive on small requests but produced incorrect code and invented an accessor. Its parser-test suggestion was usable after adapting it to the existing test module and adding malformed-input cases. Review every output for actual visibility, types, API names and edge cases before applying it. Keep integration, security and concurrency decisions with the supervising agent. Quality reviewed a 6,245-token collision/streaming prompt and returned plausible but incorrect patches (including a production minimum time delta). Those patches were rejected; the supervisor implemented contact handling and a test-only clock. Small Fast requests took 1.2–2.4 seconds after loading; the larger Quality review took 94.6 seconds. Task sizes differed, so these are observed request latencies, not a controlled throughput comparison. The service was returned to Fast afterward. Fast also drafted the final tool/edit rollback regression, which was reviewed, renamed for its behavior and adapted to the existing test module before use.

## Validation

- Final all-feature workspace tests: **1,291 passed, zero failed**. Game library: 1,265 passed, 65 ignored. Field: 7 passed. Material: 18 passed, 1 ignored. Genesis-table binary: 1 passed. One mod-API documentation example is also ignored, giving 67 ignored cases across unit tests and doctests. The ignored cases include benchmarks, slow/soak scenarios and documented limits.
- Both final tool/edit rollback regressions passed in the workspace run; the first failed before the correction. The slow transport idle-timeout regression separately passed in 13 seconds.
- Ordinary workspace/all-target checks and all-feature workspace/all-target compilation passed. Clippy completed with warnings and zero errors; the game library test target reported 251 warnings, 178 shared with the library.
- REUSE passed; cargo-deny licenses and sources passed. The compatible updates removed the vulnerability and yanked-version findings. The remaining unmaintained dependency advisory and renderer path-version ban are documented below.
- Nix checks evaluated successfully for x86_64-linux using the source export under /tmp. The REUSE and renderer source-budget builds passed after fetching a missing cached bootstrap dependency. The final pure package build passed with the rollback correction included: **1,258 release library tests**, zero failures, 65 ignored. All installed binaries and assets passed installation and runtime-library fixup. The built package is `/nix/store/v8yynhg0zyqbs88kvg973mgqz6viz6ai-project_watt_cubed-0.3.4`.
- Release game, cosmos laboratory and audio generator compiled with the repository's fat-LTO release profile. Audio tool smoke wrote all eight WAVs to /tmp.
- Cosmos laboratory smoke: 10 seeds, 4 threads, 1 preview set, process exited successfully. Start-world validity passed; suite diversity failed with 100% fallback; G=128 globe generation missed its proposed 25 ms gate, while G=64 passed. These results support keeping the prototype behind dev-tools.
- The debug build initially exhausted the host filesystem. Disposable incremental output was cleared; subsequent builds used /tmp with debug symbols and incremental compilation disabled. This did not change optimization level or production settings.

Final follow-up after the queue, wire-rounding, contact and test-clock corrections:

- Optimized game-library test code: **1,265 passed, zero failed, 65 ignored** in 101.11 seconds, within the final full workspace run. Separately, all 20 active window tests passed in 7.14 seconds and all 21 movement tests passed. All-target compilation and Clippy passed after the final rollback correction; Clippy retained its existing warnings.
- The optimized configuration was a command-line dev-profile package override (`profile.dev.package.project_watt_cubed.opt-level=3`), with debug symbols and incremental output disabled. Production profile settings were not changed. This reproduced the fast-loop failure, then verified the explicit headless clock correction.
- 100,000-edit autosave probe, median of five in the fat-LTO release test binary: snapshot 0.8482 ms, encode 1.1009 ms, write 2.2539 ms, encode+write 3.3548 ms. Encoding made 17 allocations totaling 3,001,215 bytes. This is a current measurement under concurrent activity, not a controlled before/after comparison.
- Vanilla Vulkan smoke exited normally on RTX 3070; the captured frame showed flat terrain and sky. The isolated 13-package Essentials/Dev Toolkit instance compiled, launched with 12 registered mod packages, reached readiness and produced a textured frame. Both used minimum settings, private data under /tmp and explicit Vulkan validation; no validation warning/error appeared. These are startup/compatibility checks, not full visual acceptance or representative FPS results.
- Final installed Nix package launched from `/tmp` with `LD_LIBRARY_PATH` removed, private data and Vulkan validation enabled. It exited normally, reached readiness in 0.288 seconds and reported no validation warnings/errors. This checks executable-relative assets and packaged runtime libraries without relying on the checkout.
- Final REUSE: 335/335 files carried valid copyright and license metadata.

The final network and physics source is verified in both the optimized all-feature workspace run and the pure Nix fat-LTO package build. All 279 source, asset and build files matched the exported Nix snapshot. The modded startup smoke preceded the final internal rollback bookkeeping correction; that correction changes neither the mod API nor the wire protocol.

## Remaining boundaries

- Ordinary collision-checked movement longer than 512 blocks fails closed; cruise has its existing destination-only check. The ignored 20 km/s flight test documents this policy limit. A scalable swept collision proof is needed before relaxing it.
- The novel-material reaction-front test remains ignored. The material law can propagate changes through generated terrain; a bounded two-second settling expectation is not a demonstrated law invariant.
- The prototype's worldgen quality thresholds are experimental. It is not silently promoted to the shipping generator.
- Repository-wide formatting and Clippy warnings have substantial existing debt. Deliberate empty-range errors were corrected; no blanket warning suppression was added.
- Full cargo-deny remains blocked by the renderer's unversioned voxel_slang_build path dependency and the upstream unmaintained ttf-parser chain via winit/sctk-adwaita. License and source checks are separate from these findings. The [font parser advisory](https://rustsec.org/advisories/RUSTSEC-2026-0192.html) has no safe version upgrade.
- Nix verification used an explicit source export containing newly added files. A Git-flake invocation in an unstaged worktree omits those files; include them through the normal review and staging process before invoking the Git flake. No staging or commit was performed as a workaround.
