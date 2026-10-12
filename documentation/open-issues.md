# Open issues

Known gaps that are still open. The July 2026 audit that first recorded them (`issues/`) was removed in 0.6.1 because
the rest of it described code that no longer exists; it remains in git history.

## Lighting
- **Generated overhangs and floating rock do not shadow the ground under them.** The skylight ceiling counts edited
  roofs but not generated volumetrics: `TerrainGenerator::height` describes ground only, so every chunk under an
  overhang still receives full skylight when its top is above ground height. The underside reads as an unlit strip.
  Fixing it needs either a generator-side column occupancy summary (the topmost solid, including overhang bands) or
  sky injection only at a proven top boundary followed by downward propagation across chunks.

## Release engineering
- **Split product binaries from repository tools.** The Nix package installs the game, the dedicated server and the
  golden tools as one output; the headless server inherits the GUI/Vulkan runpath. Separate outputs would keep the
  server lean.
- **CI on a clean checkout.** There is no CI. A clean-checkout job should build the game and run its tests, and run
  the locked engine revision's own suite too.
- **Shader provenance.** The engine's checked-in SPIR-V fallbacks should be reproducible from the release's sources.
- **Visual references.** Golden screenshots are local, untracked artifacts; decide how reference images travel
  between machines.

## Performance (LOD, after 0.5.0)
- Flying very fast costs about 0.1 ms more main-thread time per frame than 0.4.0. Ideas: align the near field's and
  the streaming centre's triggers, make the cut incremental, run the samplers on the worker pool.
