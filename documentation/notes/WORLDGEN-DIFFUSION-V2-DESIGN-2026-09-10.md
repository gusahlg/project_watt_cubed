# InfiniteDiffusion v2 — realistic, interesting, deterministic terrain (design, 2026-09-10)

Directive: "massively improve the infinite diffusion generation method to produce kind of realistic and
interesting results; go really creative". Constraints that stay: generation is a pure function of
(seed, coordinate, content fingerprint), bit-identical across workers and peers; the generator never
registers blocks at runtime (materials = `elements!` rows + placement rules); worker threads only; far
LOD must agree with the near field; no per-cell hashing beyond the ore roll. New constraint: **all
value math is 32-bit integer / fixed point** (Q16.16 `i32`, `u32` hashes, wrapping arithmetic, `%` as
SRem, no shifts ≥ 32), so the same tiles can later be produced by a Vulkan compute shader bit-for-bit
(agreed with the engine session: game-owned SPIR-V, engine `ComputeLane`, parity test GPU == CPU).

## 1. What the world should be like
A player walking or flying should meet, in this order of scale:
- **Continents and oceans** with shelves, deep trenches, archipelagos and an occasional inland sea;
  coastlines with bays and headlands, beaches on gentle shores, cliffs on steep ones.
- **Mountain belts** where "plates" meet (ridged, with foothills, passes and valleys), isolated
  volcanoes (basalt cones, obsidian and sulfur near the vent, a crater lake or a lava lake), plateaus
  and mesas in dry regions with coloured strata, rolling hills elsewhere.
- **Climate that follows physics-ish rules**: a latitude-like temperature cycle along z (wavelength
  ~16 km, so a walk of a few km changes the climate), altitude lapse (colder up high), coastal
  moderation, humidity from a prevailing wind with **rain shadows** behind ranges → deserts on the lee
  side, rainforest on the windward side, tundra and glaciers where cold, salt flats in closed dry basins.
- **Water that makes sense**: rivers in valleys that widen toward the coast, floodplains with gravel
  and sand bars, deltas, lakes in closed basins, wetlands in humid lowlands, ice on cold lakes,
  waterfalls where rivers cross terrace edges (emerges from terraces + rivers).
- **Biomes as continuous fields, not enums**: tundra, taiga, grassland, savanna, desert (with dunes),
  badlands, rainforest, alpine, volcanic, wetland, coastal — each a region of the (temperature,
  humidity, geology, slope) space, blended at the edges by dithering, never a hard line.
- **Underground worth exploring** (the vision doc's "depth axis"): winding tunnel systems that follow
  2-D cave "rivers", caverns that grow with depth, aquifers below sea level, crystal caves (Quartz,
  Lumin) and ore veins that follow the geology (coal in sedimentary regions, copper/iron in volcanic,
  gold/titan/obsidian in the deep).
- **Sky**: the classic archipelago of flying islands above the humid-warm zones stays (band from
  y 112), with ice skins up high.
- **Surface texture**: scattered boulders on grass, snow drifts, scree below cliffs, moss in humid
  caves, salt crusts, beaches grading from sand to soil.

## 2. Architecture
### 2.1 The integer field crate (`crates/infinite-field`, v2 API beside the f32 one)
- `IntSpec { seed, tile, stride, phases, channels }` (same tiling semantics; tile 32, stride 16).
- Values are `i32` in Q16.16; channels are named by the adapter (`enum Ch`), the crate is agnostic.
- `IntScore` gets a **stencil context**: `fn predict(&self, ctx: &Stencil, ch, x, z, phase, phases,
  current: i32) -> i32` where `Stencil` reads the PREVIOUS phase's blended value at (x+dx, z+dz) for
  |dx|,|dz| ≤ 2 (Jacobi-style: reads never see this phase's writes, so tile order does not matter). This
  is what turns "noise pulls" into erosion-like operators (slope limiting, valley carving, basin fill).
- `inoise` module: `hash32(seed, x, z, salt)`, `value_noise_q16(seed, x, z, cell, salt)` with a
  fixed-point smoothstep (cubic via i64 intermediates, results in i32), `fbm_q16(octaves ≤ 4)`,
  `ridged_q16`, `cellular_q16` (Voronoi F1/F2 + cell id: plates, geology regions), `warp_q16`
  (domain warp by two noises), `gradient_q16` (central differences on any Q16 function),
  `lerp_q16`, `smoothstep_q16`, `clamp_q16`. All `wrapping_*`, all guarded; a doc table lists the exact
  formulas so the SPIR-V mirror is a transliteration.
- Tile cache: bounded **LRU** (not clear-all), key = (spec, tx, tz); separate caches per spec so the
  far-LOD spec never evicts the near spec.
- Batch API: `fill_all(x0, z0, w, h, out: &mut [i32])` layout `(z*w + x)*channels + c` as today, and
  `tile_bytes(tx, tz) -> &[i32]` for the future GPU path (the GPU produces exactly this tile array).

### 2.2 Channels (adapter `src/world/diffusion.rs`, name `DiffusionV2`)
| ch | name | scale (cell of the coarsest phase) | meaning |
|---|---|---|---|
| 0 | elevation | 2048 m → 8 m | metres above sea, Q16; the only channel that is meshed |
| 1 | temperature | 4096 m + lapse + coast | 0..1 |
| 2 | humidity | 2048 m, wind-advected, rain shadow | 0..1 |
| 3 | geology | cellular 3000 m | region id (u8 in the low bits) + hardness (Q16 high bits) |
| 4 | water | rivers/lakes/wetness | Q16 water table height (≥ sea) or 0; river width in the low byte |
| 5 | caves | 2-D cave-river field | floor height, ceiling height, width packed |
| 6 | sky | archipelago mask + lift | Q16 lift metres or 0 |
| 7 | detail | 64 m → 8 m | slope-aware detail used by fill (boulders, drifts, dunes phase) |

### 2.3 Phases (coarse → fine), each a pure local function of the previous phase
0. **Prior**: uniform hash noise per channel.
1. **Plates & continents**: cellular field → continentalness (shelf/land/deep) + plate boundary
   distance; elevation target from a spline (deep −40 … shelf −8 … coast +2 … plateau +30); belts:
   ridged noise scaled by `1/(1 + boundary_distance)`; volcano seeds = cellular F1 minima in "volcanic"
   geology regions (cone profile added to elevation, crater at the apex).
2. **Climate**: temperature = latitude cycle(z) + lapse(elevation) + coast moderation; humidity =
   base noise advected along the wind direction (+x) by 6 stencil steps of the coarse elevation:
   `rain_shadow = max(0, elev_upwind − elev) / 40 m`; wetness lows in closed basins (basin = local
   minimum of the 5×5 coarse elevation: this is where lakes and salt flats go).
3. **Erosion** (2 passes at 128 m and 32 m): thermal slope limit (pull toward neighbour mean when the
   slope exceeds a talus angle that depends on geology hardness), valley carving toward the local
   low neighbour weighted by wetness (V profile), terraces where dry + hard (quantize to 12 m benches),
   dune ripples where desert (anisotropic ridged noise along the wind), glacial rounding where cold
   and high (U profile = smooth toward 3×3 mean).
4. **Hydrology**: rivers = cells whose wetness-weighted valley score exceeds a threshold, widened by
   accumulated wetness (proxy: the 5×5 stencil sum of upstream-side wetness), water table = elevation
   − 1 in the river cell; lakes = basin cells: water table = basin rim height; wetlands = humid low
   flat cells; beaches = |elev − sea| < 2 and slope < 0.15.
5. **Detail**: 8 m noise scaled by slope (rocky where steep), boulders/drift/scree masks, cave-river
   field (2-D winding paths: ridged noise minima; floor = elevation − depth(wetness, geology), width
   from a second noise; caverns = cellular blobs whose radius grows with depth), sky mask.

### 2.4 Column fill (worker, per 16×16 column from ONE `fill_all`)
Top to bottom: sky island (if mask) → air → water/ice → dressing by biome (continuous rules, dithered
at the edges: tundra Snow, taiga Organic+Soil, grassland Organic+Soil, savanna Soil+Sand, desert Sand
with dune stratification, badlands Clay/Sand bands by height, rainforest Organic-heavy, alpine Stone,
volcanic Basalt, wetland Clay+Organic, coast Sand→Soil gradient, salt flat Salt) → crust (Soil+Clay /
Sand+Clay / Soil+Ice / Gravel on river beds and scree) → rock by geology (Stone / Basalt / strata) with
ores by geology and depth → tunnels/caverns/aquifers → deep rock. All choices from the packed
channels plus the existing two hash streams (ore roll, surface scatter) — no new per-cell noise.

### 2.5 Materials (new `elements!` rows + placement rules; `WORLDGEN_VERSION` bump)
Basalt (dark volcanic rock, hardness above Stone), Gravel (loose, river beds/scree), Salt (dry
basins). Lava is OUT of scope for this round (a second liquid needs the property system checked).

### 2.6 Far LOD
`lod_column` samples the SAME channels through a coarse `IntSpec` (tile 32 at 8 m cells = 256 m
tiles) whose phases stop at the erosion pass; the near field's height is the coarse height plus the
detail phase, so silhouettes match within one cell. The LRU caches are separate.

## 3. Performance budget
≤ 0.35 ms per surface column in the batched fill (classic is 0.25), tiles ≤ 0.5 ms each on one worker
at 32² cells, LOD sections ≤ 3 ms/job. Integer math is faster than the f32 path on CPU too.

## 4. Verification
Byte pins (re-pinned once, with the version bump), seed/order stability, far coordinates (±1e9),
`generate == block_at`, `heights_16 == height`, LOD/near sea agreement; **shape tests** with numbers:
ocean fraction 30-50% over a 4 km² window, at least one mountain > 80 m and one river reaching the sea
in a 2 km² window at seed 42, deserts only where rain shadow or heat, lakes only in basins, caves in
50-90% of columns below −20 m with tunnels connected across ≥ 64 m; a **map export** (ignored test)
writing `target/worldgen-map-<seed>.ppm` (height + biome + water + caves as four panels) so a human
can look at 4 km² at once; plus the bench screenshot hook for in-game views.

## 5. Tasks
W1 crate integer core + stencil + inoise + LRU (`crates/infinite-field`), W2 macro phases + channels +
map export, W3 erosion + hydrology + biome fill + materials, W4 underground, W5 sky islands + surface
scatter + LOD via the coarse spec, W6 shape tests + pins + performance gauge, W7 SPIR-V mirror of the
tile pass + engine compute path (after the engine's ComputeLane lands). Mods v2 task 60 (worldgen as a
mod) follows W6.
