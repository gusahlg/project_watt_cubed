# Testing and graphics report

## Environment

- Date: 2026-07-13
- OS/tooling: Nix development shells from each repository
- GPU: NVIDIA GeForce RTX 3070
- Window path used for game captures: X11 selected by unsetting Wayland variables
- Vulkan validation: standard plus synchronization validation where noted

Plain host execution could not load the dynamic Wayland/X11 libraries. The reliable form here was:

```sh
nix develop -c env -u WAYLAND_DISPLAY -u XDG_SESSION_TYPE <binary>
```

## CPU suite results

Before integration:

- game: `cargo test --all-targets` — 287 passed;
- engine: `cargo test --all-targets` — 70 passed.

The disposable semantic merge was tested before touching the real branches:

- merged game against merged engine — 335 passed, 1 ignored;
- merged engine — 70 passed.

After the audit tests/fixes, the game library suite reached 340 passed with two intentional ignores: the existing height-mip timing benchmark and the new known-bug constructed-roof skylight regression. Final all-target verification should remain the release gate after future edits.

Useful commands:

```sh
cargo test --all-targets
cargo test --lib world::light::tests::constructed_roof_in_upper_chunk_shadows_lower_chunk -- --ignored --exact
```

The second command is expected to fail today with `LightLevel(15)` versus `LightLevel(0)`. It should be made non-ignored when the ceiling model is fixed.

## New regression coverage

The audit added or materially strengthened these checks:

- every client/server protocol variant rejects a trailing payload byte;
- standing, sneaking, and swimming wire stances convert eye to feet correctly;
- far-coordinate remote poses subtract in `f64` before narrowing;
- invalid wire stance values remain rejected;
- server state refuses NaN yaw and infinite pitch;
- an opaque emitting uniform chunk bypasses the all-dark shortcut and seeds blocklight;
- an ignored two-chunk roof fixture reproduces wrong cross-chunk skylight;
- the height mip now takes an explicit color snapshot from the compiled world palette, preventing the builtin/element-ID mismatch that the graphics run exposed;
- main's integrated tests cover exact LOD partitions, all partial-child cases, quadrant mesh bounds, winner-only light downsampling, height/error mip invariants, altitude/motion selection, and swap fades.

## Vulkan validation

The first engine smoke exposed four water-local-read VUIDs plus a synchronization `WRITE_AFTER_READ` hazard from swapchain acquisition. The conservative engine fix disabled only the invalid absorption variant (retaining flat water tint) and changed the mixed present submission to correctness-first wait stages.

Post-fix build and smoke:

```sh
env CARGO_TARGET_DIR=/tmp/voxel-engine-audit-target cargo build --bin demo

nix develop /home/gusahlg/repos/voxel-engine --command \
  env -u WAYLAND_DISPLAY -u XDG_SESSION_TYPE \
  VK_LAYER_VALIDATE_SYNC=1 \
  VOXEL_ENGINE_VALIDATION=1 \
  VOXEL_AUTOSHOT=1 \
  RUST_LOG=info \
  /tmp/voxel-engine-audit-target/debug/demo
```

Result: exit 0, validation enabled, immediate/vsync-off path exercised, resize/swapchain recreation exercised, 36-frame autoshot completed, zero warnings and zero errors. The four prior water VUIDs and the acquisition synchronization hazard were absent.

Caveat: the dormant `mesh3d_water.frag.spv` still fails standalone `spirv-val`. Runtime is safe because the pipeline is never instantiated, but the module must be repaired before re-enabling the feature or enforcing all-module validation.

The engine demo used its own sample-count configuration, so a second smoke ran through the release game with its normal 8× MSAA setting:

```sh
nix develop -c env -u WAYLAND_DISPLAY -u XDG_SESSION_TYPE \
  WATT_BENCH=1 \
  VOXEL_ENGINE_VALIDATION=1 \
  VOXEL_ENGINE_TIMING=1 \
  RUST_LOG=info \
  ./target/release/project_watt_cubed
```

That broader run initially exposed `VUID-RuntimeSpirv-samples-08725`: the tonemap shader's single-sample godray descriptor was bound to 8× MSAA depth. Engine commit `6edce74` gates depth sampling to one sample and supplies a valid unused descriptor while the MSAA godray strength is zero. After rebuilding, the same benchmark exited 0 with no validation warning/error, rendered 54 frames at 54 average FPS (51 FPS 1% low), and exercised FIFO-to-immediate swapchain recreation at both 1280×720 and 1920×1080.

The deliberate tradeoff is that godrays are disabled under MSAA until a resolved single-sample depth image is added. The new unit test `godray_depth_requires_single_sample_target` locks down that safety rule for all Vulkan sample counts.

## Pure Nix and packaged-runtime verification

The initial pure build reproduced the reported pre-LOD engine API errors. After
pinning committed engine revision `6edce74` and changing the input to a
Git-filtered source, the final package gate was:

```sh
nix flake check --print-build-logs

env -u WAYLAND_DISPLAY -u XDG_SESSION_TYPE \
  WATT_BENCH=1 \
  VOXEL_ENGINE_VALIDATION=1 \
  RUST_LOG=info \
  nix run .
```

Results:

- `project_watt_cubed 0.2.0` compiled against the locked engine;
- 340 release library tests passed, zero failed, and two were intentionally ignored;
- the new engine-source budget passed at 949,443 bytes (limit 16 MiB);
- packaged `nix run` rendered 55 frames at 54 average FPS / 52 FPS 1% low and exited 0 with Vulkan validation quiet;
- packaged `watt_server --help` exited 0;
- packaged `golden_compare poses` exited 0;
- a five-second packaged `golden` startup reached terrain/lighting/section streaming and was terminated by the timeout, proving the earlier `libX11.so.6` load failure is gone.

This verifies the actual Nix-store binaries, not only Cargo artifacts from the
live sibling checkout. Full release analysis and remaining portability work are
in [nix-and-release.md](nix-and-release.md).

## Golden acceptance run

Final command:

```sh
nix develop -c cargo build --release --bin golden
nix develop -c env -u WAYLAND_DISPLAY -u XDG_SESSION_TYPE ./target/release/golden
```

Results after the height-mip palette fix:

- all eight terrain-key sky-hole criteria passed (`max = 0`);
- frame-time ceiling passed;
- entry-time ceiling passed, measured about 0.348 s for seed `0xc0ffee`;
- the provisional-marker sweep passed;
- all eight image comparisons were invalid because captures became `1542×1406` while references are `1542×700`;
- no height-mip worker panic recurred.

This means 11 of 19 criteria passed and the remaining eight were infrastructure-size failures, not pixel-difference verdicts. The harness attempted `gharialctl toggle-float`, but no `/run/user/1000/gharial.sock` existed, so it could not prevent the tiling resize. Do not bless the resized captures.

The first post-merge run was particularly valuable: it repeatedly caught a background panic, `BlockRegistry::color` indexing generated block ID 67 into a 19-entry builtin table. That led to the compiled-palette snapshot fix. Re-running the same acceptance suite produced no recurrence.

## Visual inspection

I manually inspected the normal and terrain-key captures for spawn, horizon, tile boundary, night, cave, shadow boundary, fog/sky, and water, as well as the existing references.

Observations:

- no obvious new open-sky holes appeared; the automated terrain-key detector agrees;
- the merged LOD field did not show an obvious parent/child z-fighting strip in the static tile-boundary capture;
- water remains visible with the safe flat-tint path; depth absorption is intentionally absent;
- the alien fragmented shelves/islands and high material contrast appear consistent with the element-first direction and prior references;
- the cave capture is extremely dark: Lumin is visible but barely. This may be desired danger, but it should get a deliberate readability target once exposure correctness is fixed;
- dark floating undersides/strips are present in both old references and current captures, so they are not a newly introduced merge regression. Some are plausible unlit undersides; targeted shadow/skylight fixtures are still needed to distinguish intended darkness from missing light.

## Existing quality-tool debt

`cargo fmt --all -- --check` reports a very large pre-existing formatting delta in both repositories. Running whole-project formatting during this audit would have obscured functional changes, so it was not applied.

`cargo clippy --all-targets -- -D warnings` also fails in both repositories (about 35 game diagnostics and 16 engine diagnostics at the audit baseline). Most are style/maintainability warnings. One materially interesting game warning is the large `pipeline::Done` enum payload; it is captured in the opportunities report.

Treat these as baselines to burn down or configure intentionally. A CI gate cannot help while its accepted baseline is already red.

## Recommended graphics-test matrix

Add a finite renderer smoke that turns validation callback errors into process failure. Exercise:

- VRS on/off;
- TAA on/off and live toggle;
- exposure on/off and live toggle;
- immediate/vsync-off present;
- resize and 1:1, 16:9, 32:9 aspects;
- screenshot transfer supported/unsupported paths;
- water fallback and, only when repaired, depth absorption;
- MSAA on/off;
- a frame with no 3D draws.

Add deterministic fixed-extent captures for:

- all 16 parent/child LOD readiness masks;
- horizontal and vertical full-res/LOD coverage boundaries;
- static TAA convergence and camera-orbit ghosting;
- wide-aspect cascade edges and cascade transitions;
- shallow/deep water boundary;
- curvature horizon during translation and yaw;
- open column, constructed roof, floating island, and cross-chunk overhang lighting;
- opaque emitter and far-LOD emitter continuity;
- VRS on/off difference bounds.

Finally, validate checked/generated SPIR-V variants in CI and make shader regeneration explicit rather than a side effect of ordinary builds.
