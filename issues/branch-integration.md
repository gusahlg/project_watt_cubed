# Main-to-experimental integration record

## Starting histories

The repositories were clean and on `experimental` before this audit.

### Project Watt Cubed

- experimental start: `90059cf` — element-first Worldgen v3;
- common ancestor with main: `8d07e75`;
- main additions: `d145fe5` and `11a44ba`;
- tested merge: `363dea6` (`90059cf` + `11a44ba`).

### voxel-engine

- experimental start: `4e83755`;
- common base/main predecessor: `9e751a5`;
- main addition: `168618d`;
- tested merge: `00d7803` (`4e83755` + `168618d`);
- audit GPU safety follow-ups: `5efc734` and `6edce74`.

Both remotes were fetched before comparison. Integration was first performed in disposable clones under `/tmp`, conflicts were resolved there, and both full CPU suites passed before the real experimental branches were fast-forwarded to those merge commits.

## Why this was a merge, not a cherry-pick

The LOD change is a coherent cross-repository slice. In the game it changes selection, section representation, quadrant meshes, GPU upload ownership, render masks, fade state, height/error summaries, camera metrics, and tests. In the engine it changes per-draw detail/style data, shaders, shadow ownership, and render flags. Taking only visible pieces would recreate the very parent/child overlap and layout mismatches the work fixes.

At the same time, a blind merge was unsafe because experimental had independently changed worldgen, texture-layer width, render flags, memory sizing, and graphics controls. The resolution therefore preserved experimental semantics at conflicts while accepting the coherent main architecture.

## Game conflict decisions

### Preserved from experimental

- Worldgen v3's element-derived materials and placement compiler;
- retired tree-specific generation paths;
- `u16`/14-bit block layer behavior and device-cap wrapping policy;
- VRS, water animation, and AO settings;
- current far-coordinate/camera-relative conventions;
- the then-current flake lock; the merge did not broaden dependency updates.
  A later pure-Nix check proved its engine node stale and machine-specific, so
  only that node was deliberately replaced (see [nix-and-release.md](nix-and-release.md)).

### Adopted from main

- typed `Eye` and `Feet`, stance-aware conversion, and `RenderPose::new`;
- corrected avatar body orientation while retaining experimental avatar scale;
- quadrant masks and four quadrant sub-meshes per section;
- exact covering/partial-child selection;
- winner-only LOD skylight averaging and open-border full skylight;
- altitude/relief/SSE and motion-predicted section selection;
- height/error mip, summaries, and occlusion support;
- temporal swap fade;
- vignette setting and render flag;
- the new golden comparison helper and extensive LOD tests.

### Generation conflict rule

Every direct generation conflict kept the element-first implementation. Main's LOD consumers were adapted to the generator interface rather than restoring trees or fixed block assignments. This follows the project's current philosophy without treating the planning document as mechanically authoritative.

### Post-merge correction found by graphics testing

Main's height-mip worker created a fresh builtin registry. Experimental worldgen resolves many generated natural compositions at startup, so its generator can return IDs beyond the builtin table. The first release golden run repeatedly produced ID 67 against a 19-entry color table.

The corrected API passes an immutable color snapshot from the exact compiled world palette into `HeightMip::bake`. This is a useful general lesson for future branch work: a seed-pure generator can still depend on startup-compiled content tables; recreating only builtins is not an equivalent context.

## Engine conflict decisions

### Preserved from experimental

- VRS and `water_anim` flags and their shader gates;
- runtime `Engine::set_flags` API;
- 14-bit packed texture layer and `u16` public layer API;
- 16 MiB first allocation block;
- experimental render-client flag flow;
- the working flake lock.

### Adopted from main

- per-draw LOD scale/style push data;
- quadrant-compatible draw offsets;
- vignette flag and tonemap shader path;
- per-slot shadow maps and corrected shadow orientation/bias;
- shader constants/layout updates needed by the new draw representation.

Duplicate `SetFlags` enum/method/match-arm additions were removed during semantic conflict resolution rather than retaining both branches' equivalent API additions.

Shader fallbacks affected by the merged sources were regenerated. Unrelated SPIR-V compiler-version churn was excluded from the merge.

## Deliberately excluded main change

Both main branches introduced a repository `.cargo/config.toml` with:

```toml
target-cpu = "native"
```

It was removed from the merge. Reasons:

- binaries become host-specific and less portable;
- Nix/CI cache reuse is weakened;
- seed-only worldgen is more exposed to CPU-specific floating behavior;
- reproducible cross-machine testing becomes harder;
- local developers can still opt into native tuning for a benchmark explicitly.

Native optimization should return only after deterministic generator math is isolated/proven and the release-distribution policy asks for per-host builds.

## Main fixes now present

| Main change | Integrated state |
|---|---|
| Remote eye position treated as feet | Fixed with typed eye/feet conversion and tests. |
| Parent plus partial children double-draw | Fixed by quadrant masks/exact partitions. |
| LOD light averages losing materials | Fixed by winner-only averaging. |
| Dark missing-neighbour section skirts | Fixed with full-skylight open-border assumption. |
| Height/relief-aware LOD and occlusion | Integrated; palette-context panic found and corrected. |
| Motion prediction and swap fade | Integrated with unit tests. |
| Per-slot shadows | Integrated. |
| Vignette | Integrated alongside experimental VRS/water/AO flags. |

## Validation before and after integration

- disposable merged engine: 70/70 tests passed;
- disposable merged game: 335 passed, 1 ignored;
- current engine after GPU safety follow-ups: 71/71 tests passed;
- validation-enabled immediate-present/resize smoke: zero warnings/errors after `5efc734`;
- validation-enabled release-game smoke at 8× MSAA: zero warnings/errors after `6edce74`;
- the game flake now pins exact engine revision `6edce74`; its previously stale pre-merge snapshot was the cause of a pure-`nix run` API failure;
- final `nix flake check`: package/source-budget checks passed and the pure release suite reported 340 passed, 2 ignored;
- release game acceptance after palette fix: no worker panic, all sky-hole/performance/provisional criteria passed; image comparisons remain blocked by window-size instability;
- final audit game suite: see [testing-and-graphics.md](testing-and-graphics.md).

No branch was pushed. The local experimental branches are ahead of their corresponding `origin/experimental`; audit source/docs changes in the game repository remain ordinary worktree changes for review.
