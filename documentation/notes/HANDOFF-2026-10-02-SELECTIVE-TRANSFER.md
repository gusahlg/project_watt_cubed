# Handoff 2026-10-02 — selective transfer v1, three-realm worldgen, tools (v2.0.0)

The model of the game changed again: one request asked to simplify, to implement **selective
transfer v1** (`reaction-guide/selective-transfer-v1.md`), to remove water, to make blocks usable as
tools, and to make the terrain spectacular (mountains and valleys, creepy mines, space with planets),
with mods for textures, names and a hotbar. All of it landed; game v2.0.0, engine v1.2.0.

## What landed

- **The law** (`crates/material`): selective transfer v1 — the committed fit table from the
  reference implementation (verbatim, regenerable with `tools/generate_fit_table.py`), capacity 32,
  cached internal supports, the `1/32` threshold, swaps only at capacity, the spec's tie order.
  The kernel is faster than the reference on the holdout fixture. Observations are readings of the
  same fit function (cohesion → hardness; light/glow/grip probes). No liquids. Law version 2.
- **Scheduler** (`src/sim/reactions.rs`): exactly the pasted rules — triggers on place / remove /
  move / configuration change only; six face contacts including air; move wakes both places;
  one operation per contact per turn; changed cells wake their contacts next turn; no-change
  contacts go dormant; deterministic `(turn, contact)` order; 384 contacts per 20 Hz turn; queue
  saved with ages. Chunk load / gen / mesh / render never queue.
- **Tools**: a held block is the tool. Left click runs the law between the held unit and the cell
  (stepped to rest); both change; the cell's contacts wake. Server-authoritative
  (`ToolUse` / `ToolResult`, protocol 11). Player damage from reactions is still undecided.
- **Mods**: `hotbar` (1-9, 0 = hand, wheel), `inventory` (I; 1-9 equips), `neural_textures`
  (a CPPN grown from each configuration, 32×32), `material_names` (Markov model over real mineral
  names, deterministic per configuration, block + tool names), `diffusion` (the generator).
  Removed: crafting, procedural/GPU texture mods, material-lab, infinite-field, regions/placement.
- **Worldgen** (`src/world/terrain`, `WORLDGEN_VERSION` 6): three realms from one pure function;
  every material found in the law by `terrain::palette` (cohesive, mutually dormant, reagents as
  the natural tools). Landmarks for seed 42: peak (1536, 419, 360), a mine lamp at (-616, -60, -608),
  a molten planet centred (337, 870, 71) r 37.
- **Water removed** everywhere (physics, audio cues, render flags, assets, goldens).
- **Space sky** (game `frame_snapshot::space_factor` + engine star floor in
  `exposure_dither.y`): above ~500 m the sky fades to black, the halo vanishes, stars show by day.
- **Sky pass regression fixed**: since Task 68 (2026-09-10) the game pushed `set_sky` only when
  the descriptor changed, but the engine resets its draw lists every frame — every frame after the
  first had no sky pass (the flat clear colour showed). `Sky::draw` now pushes every frame.
- **LOD stacked slabs**: a section whose relief exceeds 16 packed cells stacks up to 4 slab meshes
  at the least shift instead of coarsening every axis — mountains keep 8-block LOD cells instead of
  32-block slabs (the "giant floating cubes").
- **Palette colours**: the law's colour noise is contrast-stretched so whites, golds and vivid
  colours exist (snow, bone, stars, lamps, magma now read right). Brightest glow roles search first.
- `mods.cfg` gained a `version=2` marker; older files' `diffusion=off` lines are ignored.

## Numbers (desktop, RTX 3070)

- Palette search: ~190 ms once per process. Worldgen: 175 µs per chunk, 1.22 ms per column.
- Far section extract+mesh: 4.0 ms/section (was 2.4 on the old, flatter terrain).
- Tests: 734 game + 15 material, all green; engine 301 + slang 28 green in its own shell.
- Goldens re-blessed at the current tile size (they are gitignored local artifacts).

## Next

- Player damage from tool reactions (design open).
- Far LOD: the finest ring is still 8-block cells (`FINEST_DETAIL` 2 × shift 1); horizontal
  tiling of sections would halve it at a draw-count cost.
- Clippy has pre-existing warnings outside this change; rustfmt is not enforced in this repo.
