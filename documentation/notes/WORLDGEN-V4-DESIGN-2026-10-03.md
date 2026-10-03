# InfiniteDiffusion v4 (WORLDGEN_VERSION 7) — the cube-planet universe

Companion to `SPACE-ARCHITECTURE-2026-10-03.md` (§8). This is the content design every worldgen task
follows. Owner's brief (2026-10-03): a 50,000,000³ cube planet you start on; terrain like today but with
far more variety, themed regions with soft gradients, every place unique; the interior populated and
alive with structures; space mostly empty with clustered asteroids, meteoroids and moons; a few other big,
epic, varied planets that are rare and wonderful to find. "Spectacular" comes from the rules (gravity from
matter, geometry), never from hard-coded scenery. Everything is anchored (no motion).

## 1. Module layout (`src/world/terrain/`)

| File | Responsibility |
|---|---|
| `mod.rs` | `Terrain` (the generator), `TerrainCfg`, `Materials`, dispatch: cell → body → body generator |
| `cosmos.rs` | the seeded body catalog: bodies, clusters, lookups (`bodies_touching`, `body_at`), `MassOracle` impl |
| `cube.rs` | a cube body: face ownership, face-local frames, surface/interior dispatch, edge rim, `sky()` |
| `surface.rs` | (was `shape.rs`) the km-scale height field + biome layers, in face-local `(u, h, v)` |
| `province.rs` | regions and provinces: theme vectors with soft blends, the geopotential climate term |
| `features/*.rs` | surface feature families (one file each: trees, giant trees, mushrooms, spires, cones, crystals, dunes, mesas, islands, craters, ruins, monoliths, …) |
| `underground.rs` | caves, tunnels, mines, veins (unchanged algorithms, face-local) |
| `deep.rs` | the deep bands: caverns, geodes, halls, the underdark, bubbles, the heart |
| `space.rs` | small bodies: asteroids (shapes, materials, geodes, derelicts), meteoroid belts, moons |
| `worlds.rs` | the other big bodies (twin cubes, round curved-chart worlds, the hollow world) |
| `noise.rs` | + `perlin3` octave helpers, a cellular (F1/F2) helper on 2-D and 3-D points |
| `palette.rs` | + new roles appended (Plain first, Glow/Clear only where needed) |

Every function is pure in `(seed, cell)`; IEEE basic operations only; batch fill == per-voxel definition.

## 2. The cosmos

Coordinates are world blocks (i32 cells, f64 positions); the universe is bounded by `|x| < 1e9`.

- **Start cube** `HOME`: centre `(0, −25_000_000, 0)`, half-size `H = 25_000_000`. Its +Y face plane is
  `y = 0`; spawn is at the +Y face centre `(0, surface(0,0), 0)`.
- **Big bodies** (Level 0, 3–4 of them, seeded positions on a shell of radius 4e8–8.5e8 around HOME,
  pairwise ≥ 4e8 apart, all inside ±9.2e8):
  1. **The Twins** — two cubes, half-size 6e6 each, facing each other across a 1.5e6-block gap. Between
     them gravity cancels at the midpoint: a weightless canyon whose sky is the other world's surface.
  2. **Verdance** — a round world, radius 8e6, on curved cube-sphere charts (Phase B): giant forests,
     low gravity (≈ ¼ of home), long jumps.
  3. **The Hollow** — a round shell, outer radius 6e6, shell 2.5e5 thick, curved charts on both faces
     (Phase B). Outside: a crust world; inside: weightlessness (the shell pulls nothing) around a small
     glowing core body whose faint pull draws everything inward.
  4. (optional) **Ember** — a small molten round world.
  Phase A ships HOME + the Twins + space; round worlds arrive with the curved charts.
- **Moons** (Level 0.5): 2 around HOME and 1–2 around each big body, 5–7 parent reaches out (radius
  0.6–1.5 M). Like every round body (Verdance, the Hollow's two surfaces, the Ember) they are painted on
  curved charts in storage (`terrain::storage`, `terrain::round` styles), so their ground is level under
  their own pull everywhere — no grid stair-steps. Only small rocks (asteroids) are Cartesian.
- **Clusters** (Level 1): cells of `2^24` blocks; a cell holds a cluster with probability ~3 % (× the
  `space` knob), never within 3 radii of a big body. A cluster is a swarm (sphere), a belt (flattened
  disk) or a stream (elongated), radius 5e4–2e6, 50–2000 bodies with power-law radii (most 3–30,
  some 30–300, rare 300–3000). Bodies keep their cell invariant (everything a cluster paints stays in
  its cell). Body kinds: rocky (regolith/basalt), metallic (ore-rich), icy (frost/ice), carbon (dark),
  crystal geode (hollow, crystal-lined), derelict (a monolith or ruin on a rock).
- **Empty space** costs O(1): `classify(chunk)` answers `Air` from the catalog before any sampling.
- **Oracle:** every body is a sum of analytic primitives (cubes = layered boxes; round worlds = layered
  balls; the Hollow = ball − ball); clusters are summarised groups that open into their bodies (balls).
  Relief/caves are declared approximation errors per body.

## 3. HOME — the cube

### 3.1 Faces and frames
- Ownership of a cell: the dominant axis of `p − HOME.centre` (L∞ pyramid, tie-break X > Y > Z, + wins).
- Face-local `(u, a, v)` from `space::FaceFrame`; altitude `h = a − H` (0 at the face plane).
- `sky(chunk)`: `Axis(face)` for chunks owned by a face outside the edge band; `Open` inside the edge band
  (within 2 000 blocks of an edge or corner, measured on the surface) and for chunks outside every body.
- Edge rim: within 4 000 blocks of an edge the face height blends (smoothstep) into a rim height defined by
  3-D noise at the surface point `S` (the point on the cube surface), identical from both faces, so the
  surface is continuous across edges and corners. The wedge beyond both face planes is owned by the
  dominant face, which makes every edge a clean 90° ridge.

### 3.2 Climate and the geopotential gradient
`altitude_g(u, v)`: the cube's potential along its surface, normalised 0 (face centre) → 1 (corner), from
the closed-form box potential evaluated once on a coarse table (or the analytic approximation
`(max(|u|,|v|)/H)²·0.5 + (|u|·|v|/H²)·0.5`, documented as a fit). It is the "gravitational altitude":
- temperature falls with it (basins warm, rims cold, corners frozen),
- moisture follows a province field,
- realm character per face (below) is modulated by it.

### 3.3 Faces as realms (base character, blended toward the rim climate)
| Face | Realm | Base character |
|---|---|---|
| +Y | The Green Basin | temperate meadows, forests, hills, mountain ranges (today's terrain lives here) |
| −Y | The Ashen Face | volcanic fields, basalt plains, cinder cones, obsidian, glowing fissures |
| +X | The Dune Sea | dunes, mesas, canyonlands, salt flats, badlands |
| −X | The Shattered Face | karst spires, sinkholes, stone forests, crater fields, ruins |
| +Z | The Glass Face | ice sheets, glaciers, crystal fields, frozen spires |
| −Z | The Fungal Wilds | giant mushrooms, glowing moss, bioluminescent hollows, bone lands |

### 3.4 Regions and provinces (soft themes)
- **Regions** (~60 km cellular cells on `S`): a climate offset and a relief multiplier (0.4–2.2).
- **Provinces** (~5 km cellular cells on `S`, jittered): each picks a **theme** from its realm's weighted
  table, shifted by temperature/moisture; blend weights from `F2 − F1` over a ~600-block border, so
  every boundary is a gradient. Spawn (+Y centre) is guaranteed a rich mix within 3 km.
- A theme is data: relief multiplier, base-shape weights (hills, ranges, mesas, dunes, spires, flats),
  surface layer materials, tree/feature densities, strata family. Themes (≥ 20):
  meadow plains, flower fields, broadleaf forest, giant-tree forest, autumn wood, blossom grove, taiga,
  alpine range, glacier, canyonlands, mesa steppe, dune sea, badlands/hoodoos, salt flat, karst stone
  forest, volcanic field, ash waste, crystal field, fungal forest, glow-moss hollow, bone lands,
  sky-island archipelago, crater field, petrified forest, terraced hills.

### 3.5 Surface features (each a site grid hashed in face-local coordinates, like mines today)
Trees (existing) + giant trees (30–70 tall, 3×3–5×5 trunks, branching canopies); giant mushrooms (stems,
caps, glowing gills); stone spires and hoodoos; arches; cinder cones with craters and magma pools; basalt
columns; crystal clusters (glowing); ice spires and crevasses; dunes (ridged noise); mesas and canyons;
sinkholes; floating islands (anchored, 100–300 above ground, with their own soil, trees and dangling
roots); impact craters with raised rims and ore cores; giant ribcages and skull rocks; petrified trunks;
boulder fields; flower patches.

**Structures** (rare, landmark scale): ruined towers, step pyramids, stone circles, obsidian monolith
cubes (a nod to the planet), temple courts with stairs and pillars, observatories whose dishes point at
the nearest big body in the sky (computed from the cosmos — the world itself hints where to look), mine
entrances, bridges across canyons. Each picks materials from its province theme.

### 3.6 Interior (depth `d` below the local surface along the face normal)
| Band | Content |
|---|---|
| 0–350 | today's caves, tunnels, mines, veins, glowcaps, crystals (face-local) |
| 350–4 000 | the Deep: huge caverns (100–600) with biomes — fungal forests, crystal geodes, magma lakes, root halls, stalactite forests, glow-worm ceilings; dwarf halls (pillared rectangular halls with stairs); buried ruins |
| 4 000–60 000 | the Underdark: sparse 1–5 km chambers with inner landscapes and glowing lanterns, joined by shafts |
| deeper | solid mantle with rare colossal hollow bubbles (10–50 km; inside a spherical cavity gravity is uniform — an emergent consequence the player can feel) |
| centre | the Heart: a hollow cube chamber (half-size 50 000) at the planet's centre, weightless, with anchored floating crystal clusters and a glowing core |

Solid depth with no feature intersecting a chunk returns `Uniform(layer)` in O(1). The deep bulk is a
noise-banded mix whose mean amount is exactly 5.0 (it calibrates the spawn pull to 24 m/s² with the
compiled `G`); the oracle uses the same layer densities.

## 4. Space bodies
- Asteroids: noisy ellipsoids and multi-blob "potatoes" (integer squared distances, f64 noise), with
  kind-dependent crusts and veins; geodes are hollow with a crystal lining; derelicts carry a small ruin.
- Moons: round, cratered (craters as subtracted balls with raised rims), regolith over rock, a few with
  ice caps or glowing fissures.
- The Twins: two cubes generated by the same cube code with their own realm tables (one lush, one
  crystalline) — the gap between them is the attraction.
- Voxel stars are gone; the sky draws stars.

## 5. Knobs (`TerrainCfg`, text keys appended, wire/save widened to 8 × u16)
`relief`, `caves`, `mines`, `space` (cluster density), + `variety` (province diversity), `features`
(feature density), `structures`, `deep` (deep-band density).

## 6. Verification
- batch == per-voxel in every realm (surface each face, edge, corner, deep, cluster, moon, twin);
- surface continuity across edges and corners (no step > relief slope at the seam);
- spawn gravity within 2 % of 24 m/s²; `classify` is never wrong (Air chunks are empty);
- face-map PPM export (`worldgen_map` ignored test) for tuning: height/biome/province colour maps of a
  face window;
- performance: column cost ≤ today's 1.22 ms on the surface; empty space and solid depth O(1).
