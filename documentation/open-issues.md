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
- **The LOD cut can outgrow the section floor (audit B16).** `desired_sections_with` debug-asserts that a cut fits
  `SECTION_SLOT_FLOOR`; face-twin, cube-edge and altitude selections reach 1025–1810 tiles, so twelve `world` lib
  tests panic in debug builds (they did in 0.6.1 too). Release builds skip the assertion. The fix is a derived bound
  for multi-face cuts, enforced in selection and admission, not a bigger constant
  ([audit note](notes/2026-10-12-audit/bugs.md#b16--ordinary-lod-selections-exceed-the-asserted-section-capacity-contract)).

## Simulation models (0.7.0)
- **The server steps the selected model every tick, even when nothing is in contact.** An idle-skip needs each model
  to say when it is quiescent.
- **A tool use consults the model before the edit policy.** A stateful model keeps whatever it changed when the
  policy then denies the edit; the policy should decide first.

## Headless client (0.7.0)
- **`headless::tests::two_clients_walk_break_place_and_leave` is intermittent under a loaded full test run.** The
  breaking client sees its edit and the confirmation, but the second client reports no `WorldChanged` within 8 s;
  run alone it passes. Suspect the second client receiving the edit before its mirror holds that chunk, so the
  change is applied without an event.
