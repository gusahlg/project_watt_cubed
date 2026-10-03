# Space architecture (2026-10-03) — matter-derived gravity, face layouts, curved patches, the cube-planet universe

This is the plan of record for the "space physics" update (guides/pwc_space_physics.md) and the
InfiniteDiffusion v4 universe. Every task directive refers back to the section numbers here.

## 0. Decisions (from the owner, 2026-10-03)

1. **Gravity is the vector sum from the actual matter.** On the 50,000,000-block cube planet each face
   behaves like a vast shallow bowl: flat near the face centre (< 1.2° tilt within 1 M blocks), 16°
   halfway to an edge, 45° at edges, corners are the "peaks" (0.65 g). No per-planet gravity rule.
2. **No classification.** The guide's mechanisms are laws, not planetary special cases. A planet a
   player builds behaves the same way because behaviour comes from matter and layout, never from a
   label. Generated layouts (face frames, curved cube-sphere charts) are *initial world state* chosen
   by the generator (guide §10.2); physics never reads what kind of body something is.
3. **Everything is anchored.** Moving bodies, orbits, velocity-driven body physics, collisions,
   deformation and fracture (guide stages 4–8) are **out of scope** for now. The architecture leaves
   seams for them (revision domains, pose fields fixed at identity, patch embeddings) but implements none.
4. **Travel stays simple.** Today's fly speed stays; `/tp` becomes a slash command (it is also a console
   command). Distant planets are found by seeing them in the sky.
5. **Planet size:** the start cube is exactly 50,000,000 blocks on a side (half-size `H = 25_000_000`).

## 1. Scope (guide stages)

| Stage | In scope | Where |
|---|---|---|
| 0 Geometry lab | yes — mappings, metrics at face centre/edge/corner/depths, core transition, report | `src/bin/planet_lab.rs`, `src/space/chart.rs` |
| 1 Identity & conservation | yes — amount = occurrence count, mass ledger, conservation tests | `src/gravity/mass.rs`, registry |
| 2 Matter-derived gravity | yes — kernel, analytic sources, oracle, tree, samples, epochs | `src/gravity/` |
| 3 Playable worlds | yes — face layouts, curved charts, any-direction player, physical raycasts, camera-relative render, save/net | everywhere |
| 4–8 Assemblies, deformation, fracture, retiling, formation | **no** (owner) — seams only | — |

## 2. Units and laws

- One block = one world unit (`BLOCK_METERS = 0.85` m). Positions are `f64` (`DVec3`), cells `i32`,
  universe bounded by `WORLD_BORDER = 1e9` (physical movement clamps there). Curved-chart *storage* lives at
  `x ∈ [1.1e9, 2.0e9]` (§7) and is never a physical position; `math::block_coord` resolves cells up to
  `CELL_LIMIT = 2.05e9` (still i32-safe for chunk math).
- **Amount** of a cell = occurrence count of its configuration (`0..=32`, exact, conserved by the law:
  selective transfer moves occurrences and never changes coordinates). **Mass = amount × 1 unit.**
  Gravitational and inertial mass agree by construction (guide §7.2 row 1). Air = 0.
- **Universal gravitational constant** `G` (in block units, per amount unit) is one compiled constant,
  chosen once so that the *designed* start planet gives `24 m/s² × PER_METER` at its +Y face centre.
  It is never computed from content at runtime.
- **Force law** (guide §14.3): softened inverse square with a smooth finite range, the same for every
  pair: `k(r) = r / (r² + ε²)^{3/2} · w(r)`, `w = 1` for `r ≤ R_IN`, smoothstep to `0` at `R_G`;
  potential `Φ(r) = −G m ∫_r^{R_G} k(s) ds` (consistent with the force). `R_IN = 1.25e8`, `R_G = 2.5e8`,
  `ε = 0.5` (half a block; a point sample is a cell). Bodies are generated ≥ 4e8 apart, so within any
  body's own extent the law is exactly Newtonian and analytic closed forms apply unchanged.
- **Gravity sample** (`gravity::Sample`): `accel: DVec3`, `potential: f64`, `tidal: Option<DMat3>`,
  `error: f64` (bound on |Δaccel|), `epoch: u64` (source epoch). Never normalise `accel` without the
  zero-g threshold (§5.4).
- Flat vanilla world: a finite slab (`FLAT_HALF = 6e7` horizontally — small enough that its whole field is the
  exact closed form — and as deep as its own pull needs to be 24 m/s² at the centre, found by bisection) — its
  gravity comes from its matter like everything else.
- `/gravity` prints the local field (strength, down, tilt off the grid axis, potential, error, epoch).

## 3. Coordinates, faces, frames

- `coord::Face` (NegX, PosX, NegZ, PosZ, NegY, PosY) is **the** signed-axis type. New helpers:
  `axis()`, `sign()`, `normal() -> (i32,i32,i32)`, `dvec()`, `from_dominant(DVec3)` (fixed tie-break
  X > Y > Z, positive wins on exact ties), `ALL`.
- `space::FaceFrame` (new, `src/space/frame.rs`): for a face `f`, the right-handed integer basis
  `(t_u, n, t_v)` with `n = f.normal()`; maps world ↔ face-local `(u, a, v)` where `a` is the altitude
  coordinate along `n` (`a = sign · coord_axis`). Exact integer maps both ways (`to_local`, `to_world`,
  for cells, chunk coords and f64 points). For `PosY` it is the identity (u = x, a = y, v = z), so all
  existing Y-up code is the `PosY` special case and keeps its bytes.
- **Layout per chunk** (`world::Sky`): `enum Sky { Axis(Face), Open }`. It is static world state from the
  generator (`TerrainGenerator::sky(ChunkCoord)`), **not** from gravity. `Axis(f)`: skylight falls along
  `−n`, generation columns run along `n`, the far LOD tile family is face `f`. `Open`: no ceiling; full
  sky through every face (space, asteroids, cube edge/corner bands).
- **Column key** (`world::ColumnKey { face: Face, a: i32, b: i32 }`): a run of chunks along `face`'s
  normal at chunk tangent coords `(a, b) = (cu, cv)`; range given as chunk altitudes. `ColumnKey { PosY,
  cx, cz }` is today's `(cx, cz)` column.

## 4. Gravity module (`src/gravity/`)

```
gravity/mod.rs      Sample, Accuracy, G, R_IN, R_G, EPS, the window, PHYSICS_VERSION
gravity/kernel.rs   softened windowed point kernel (+potential, +tidal), exact for r<=R_IN
gravity/shape.rs    analytic uniform sources: Box (closed-form prism, Nagy/Okabe), Ball, Shell,
                    Point; each: field(x) -> (accel, potential, tidal), bounds, mass, first moment
gravity/source.rs   Source = Σ primitives (with density increments) + correction nodes
gravity/tree.rs     Barnes–Hut over correction nodes (mass u/i128 exact, first moment about fixed
                    node centre, abs-mass bound), opening criterion θ, near-field descent
gravity/field.rs    the per-world query object: bodies (analytic) + edits (corrections) + cache
gravity/mass.rs     amount tables, per-chunk amount/moment summaries, the edit delta ledger
```

- **Bodies are analytic.** The generator's body catalog (§8.2) describes every body as a sum of uniform
  primitives with density increments (a cube = nested boxes for its layers; spheres = nested balls;
  hollow worlds = ball minus ball). Cave/mine porosity is folded into layer densities as a measured
  constant with its error bound recorded; surface relief is a correction (below).
- **Corrections** (signed, guide §6.7 option 2): per chunk, `Δ = amount(actual) − amount(baseline
  primitive)` for edited chunks (the edit ledger, exact integers), with first moments about the chunk
  centre and `Σ|Δ|` as the error bound. Relief: per face tile (e.g. 4096²) mean relief mass and moment
  from the generator's height field, error = max relief × area × density.
- **Query** = Σ analytic primitives of every body within `R_G` (closed form) + tree walk over
  corrections (monopole + dipole about the fixed centre; descend when `size/d > θ` or when the query is
  inside the node's bounds). Loading never changes the sources (they are seed + edits), so loading is
  never a physical cause (guide §14.4).
- **Epoch:** `Field::epoch` bumps when a committed edit changes amount; `Sample.epoch` records it.
  Player sampling reuses the last sample while the eye moved < `REUSE_DIST` (e.g. 8 blocks) and the
  epoch is unchanged (guide §6.8–6.9: the field of a 25 M cube changes over thousands of blocks).
- **Tests (guide §18):** uniform ball interior linear and zero at the centre; shell cavity zero; two
  equal masses cancel at the midpoint without NaN; cube face centre vs closed form vs brute force;
  edits as corrections never double count (generate, edit, sum = direct sum); load order invariance;
  absolute *and* relative error reported.

## 5. Player and camera (any direction)

1. `Orientation` gains `frame: DQuat` (local→world; local +Y = up). `direction()`, movement basis and
   `up()` go through the frame; yaw/pitch stay relative to it. `ViewPose` carries `up`.
2. **Alignment:** at render rate the frame rotates by the shortest arc toward `−ĝ` with rate
   `ALIGN_RATE` scaled by `weight = smoothstep(0.02 g0, 0.10 g0, |g|)`; below that, the frame holds
   (zero-g; guide §12.2). Snap (no smoothing) at spawn, `/tp`, load, net snap-back.
3. **Integration:** `v += g dt` with the full vector; the walking target acts on the component
   perpendicular to `u` (the frame up); jump adds `JUMP_SPEED` along the support normal; terminal speed
   clamps the component along `ĝ` only. With `g = (0, −G0, 0)` everything reduces bit-exactly to today.
4. **Collision:** box tall along the **snapped axis** — the dominant signed axis of the frame up *in the
   storage frame of the patch the player is in* (identity for the world grid). Sweep order: the two
   tangent axes first, the snapped axis last. Grounded = blocked moving along `−snapped`. Axis switches
   use hysteresis (switch only when the new axis leads by > 0.05 in cosine) and must find a free box.
5. **Zero-g:** existing air control applies (a declared game rule, unchanged); the frame holds.
6. **Raycast:** returns `t`, hit point, face; the DDA is unchanged for the world grid. Curved patches
   transform the ray into storage space with the local affine map (§7).
7. `feet()`, gait, footsteps, avatars, contact shadows and the minimap use the frame / snapped axis.

## 6. World generalisation (root grid)

1. **Generator trait** (`world::generation::TerrainGenerator`) additions (defaults keep FlatTerrain):
   `sky(ChunkCoord) -> Sky`; `surface(face, u, v) -> i32` (altitude of the first open cell above the
   ground along `face`); `generate_column(ColumnKey, RangeInclusive<i32>) -> (Vec<(i32, ChunkData)>,
   ColumnHeights)` where heights are altitudes along the key's face; `bodies() -> &Cosmos` (catalog);
   `classify(ChunkCoord) -> Classify { Air, Uniform(id), Mixed }` called before any column work.
2. **Light:** `propagate` is monomorphised over the sky face (`const SKY: u8`, plus `OPEN`); the PosY
   instance must reproduce `light_byte_pin` bit for bit and keep `light_propagate_throughput`.
   `CeilingWindow` gains its face; ceilings/column caches are keyed by `ColumnKey`; `trivial_light`
   compares altitude; `Open` chunks of uniform air are trivially fully lit.
3. **Generation jobs:** `Job::GenerateColumn { key: ColumnKey, range, .. }`; `request_region_data`
   groups the data box by each chunk's `Sky` face; `Open` chunks are generated per chunk.
4. **Streaming shape:** `ChunkBox` gains an up face (`Option<Face>`; `None` = isotropic, used in space);
   the streaming centre's layout picks it. `ViewGate` and the pacer measure 3-D.
5. **Space is free:** `classify` returns `Air` for chunks touching no body in O(1); uniform chunks share
   one interned `Arc<Chunk>` per id; voxel stars are removed (the sky draws stars).
6. **Far LOD:** `SectionPos` gains its face; sections sample face-local columns with a per-body datum
   and a per-section altitude window from the generator's height bounds; packed vertices are permuted
   to world orientation at upload (exact integer, det = +1); the height mip follows the eye; edits dirty
   sections through the face map. Non-heightfield bodies (asteroids, moons, other planets) get a 3-D
   volume LOD (octree of the same mesher with air boundaries) and, far away, impostors (§9).

## 7. Curved patches (round worlds; guide §§5, 9, 10, 12)

- A round body's atlas = 6 cube-sphere shell charts per depth band (equiangular map; the lab compares
  normalized / equiangular / one adjusted map and documents the choice), angular resolution halving
  per band (1:4 interfaces), and a Cartesian core joined by a 6-block transition shell (guide §10.5).
- **Storage atlas** (`space::atlas`, implemented): every chart cell has an ordinary `i32` storage address in
  a box of the reserved region (`STORAGE_X0 = 1.1e9`, one `SLOT = 2^26` of x per atlas; boxes chunk aligned
  with radii and resolutions multiples of 16, ≥ 64 cells apart); storage `+Y` is the chart's up (outward, or
  toward the centre for an inner surface — inward charts also flip x so storage stays right-handed). So
  streaming, light, meshing, edits, saves and the network work on storage cells unchanged. A chunk's
  **embedding** (identity or chart map) is world state from the generator. `Atlas::shell` builds a single
  band without a core (the Hollow's two surfaces).
- **Glue:** a storage cell within two cells outside a box reads as the neighbouring patch's cell holding the
  same physical point; `Atlas::chunk_across` gives a seam neighbour as a whole chunk plus a signed index
  remap (exact across chart edges, approximate 1:2 across band interfaces) for mesher halos and light shells.
- **Rendering:** chart chunk meshes carry an 8-corner trilinear cage (engine task E4; corners computed in f64
  relative to an anchor block, so no per-frame rewrite; neighbouring chunks share corners, so no cracks;
  chord error ≤ L²/8R, 1.6e-5 blocks for a chunk at R = 2 M).
- **Motion** (`movement::update_player_in`, implemented): the player steps in the storage frame of the patch
  under it — the ordinary axis-aligned collision — with velocity, gravity and the body frame carried through
  the local Jacobian, walking speed rescaled so physical speed is preserved (the hitbox stays in cells), and the
  position re-embedded exactly. The 8 valence-3 corners and band interfaces are the declared exceptional
  regions (stage-0 report).
- Chart geometry uses `chart::tan_quarter` (Lambert's continued fraction), never the platform `tan`, so
  generated geometry is bit-identical on every peer.
- Streaming near a round body streams storage boxes around the chart-mapped eye (one per chart within
  reach; usually 1, up to 3 at corners).

## 8. InfiniteDiffusion v4 (WORLDGEN_VERSION 7)

1. **Placement:** start cube centre `C0 = (0, −H, 0)`, so its +Y face plane is `y = 0` and spawn is the
   +Y face centre `(0, surface, 0)` — gravity there is exactly perpendicular. Faces are generated in
   face-local `(u, a, v)`; ownership of a cell is the dominant axis of `p − C0` (L∞ pyramid, fixed
   tie-break); edges/corners within the relief band are `Sky::Open`.
2. **Cosmos** (`world/terrain/cosmos.rs`): the seeded body catalog, single source for terrain, the
   gravity oracle, sky impostors and `/tp` targets. Level 0: the start cube + 3–5 big bodies (round
   curved-chart worlds, a hollow shell world, other shapes) ≥ 4e8 apart, a few moons each (anchored).
   Level 1: sparse cluster cells (asteroid swarms, meteoroid belts) with power-law sizes; space between
   is empty. Every body stays inside its cell; `bodies_touching(aabb)` is O(1) in empty space.
3. **Cube surface:** themed provinces (cellular fields on the 3-D surface point, so continuous across
   edges) with soft gradients, the current km-scale shape as the base layer, geopotential "altitude"
   as a climate term (warm basins at face centres, cold rims, frozen corners), many feature families.
4. **Interior:** today's caves/mines/veins near the surface, cavern biomes and structures deeper,
   rare huge hollow chambers, a core; O(1) uniform early-outs for solid depth.
5. Materials: palette roles appended last; scoped dormancy per body family.

## 9. Rendering (voxel-engine branch `space`)

1. Wrapping `i32` camera arithmetic (A1 in the engine audit).
2. Local frame: `Frame3D::set_local_frame(up, altitude)`; sky gradient, fog, clouds, star horizon fade
   in the local basis; stars on a world-space cube map; curvature droop removed; LOD clip an axis box;
   `draw_shadow` takes a normal; water tangent plane.
3. 5-bit detail and a bounded depth bias.
4. Far bodies: `Frame3D::set_far_bodies(&[FarBody])` analytic impostors (cube, sphere, shell) in the sky
   pass, lit by the sun, occluding stars.
5. Curved meshes: a per-mesh trilinear cage table indexed by the spare `MeshRecord` lane (task E4).

## 10. Persistence and network

- Protocol 12: `Move`/`PeerMove`/`Position` carry the frame (`DQuat` → 4×f32 quantised) and velocity;
  `Teleport`/`Position` carry a request id; `Welcome` carries the physics fingerprint.
- Save v10: player frame + velocity; widened worldgen knobs; `PHYSICS_VERSION`. v9 Diffusion saves
  (worldgen < 7) are refused with a clear message ("made before the cube-planet universe"); Flat loads.
- Fingerprint folds `PHYSICS_VERSION`, `G`, the kernel constants and the amount rule.

## 11. Invariants every task keeps

- Determinism: generation, light, meshes, saves and wire bytes are pure functions of seed + edits;
  noise uses IEEE basic operations only (no `sin`/`exp`/`powf` in generation).
- The PosY / identity case of every generalised path is bit-identical to today until a task explicitly
  re-pins (and says so).
- Quiet frames allocate nothing and read the clock once.
- New core modules are not re-exported by `pwc-mod-api` (a small facade later, additive).
