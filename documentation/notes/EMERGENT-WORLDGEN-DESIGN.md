# InfiniteDiffusion v5: worlds grown from the law

**Status:** design only. I edited no files and ran no builds or benches. Every timing below is an estimate from reading the code unless it says "measured". Line numbers are for main 14fe5bd and 029df47 (the commits between them do not touch worldgen).

**Sources.** This merges the four candidate designs and three judge reviews. The backbone is "worlds grown from the law" (nebula, then accretion, then the law's own mineral process, then genesis, then a body field, then point detail, then forms). Grafted onto it:
- design 2's spawn contract and migration;
- design 1's accretion physics, debris clusters and determinism guards;
- design 3's verified facts and its sun-driven climate.

Ideas that the reviews showed are too risky are kept as gated options. They are not part of the plan.

---

## 0. Summary for the owner

**Today.** Every seed gets the same cast:
- Home;
- two Twin cubes;
- Verdance;
- the Hollow with its Ember;
- 6 to 10 moons.

That list is written out in `Cosmos::with_deep` (`src/world/terrain/cosmos.rs:310-417`). The seed only moves and resizes these bodies. Even the matter is the same every time: `choose_bulk` returns the first candidate whose amount is 5, which is `rock` alone with cut 0 (`cube.rs:210-212`). So Π_g, the relaxed home shape and the twins' sag are identical for every seed. That is also why the gravity relief can be cached once per process (`terrain/mod.rs:416-433`).

**v5.** The written list is replaced by a chain of small, deterministic processes. Each one reads only the seed, the law and the stage before it.
- How many worlds exist, how big they are, what they are made of, whether they are cubes or balls, their moons, rings, pairs, glow, air, terrain, plants and ruins all come out of the chain.
- `Kind` stops being a cause. It becomes a set of traits derived after the fact.
- InfiniteDiffusion comes back as one small integer engine that runs on three shapes of grid. It is the deleted `crates/infinite-field` (Q16.16 fixed point, Jacobi phases) with one fix: tiles get an exact margin instead of a blended overlap.

**Decisions only you can make** (details in section 7):
1. The Twins, Verdance and the Hollow with its Ember stop being guaranteed. They appear only when the processes produce them. Hollows cannot appear before the last phase.
2. Small, strong moons become cubes, because that is what the law says. This reverses the 2026-10-04 decision to leave them "honest as-is".
3. Spawn gravity stays exactly as designed: the start world's ρ·R is pinned. The spawn garden stays as the one authored exception.
4. Every content phase bumps `WORLDGEN_VERSION`, so old saves are refused each time.
5. Each world gets its own minerals, found in the law. This is the only way colours and densities can differ between seeds, and it is also the riskiest part. It is measured in a lab before anything ships.

---

## 1. Vision

A world should be found, not written. The project already works this way below the surface: materials are "found in the law, never authored" (`palette.rs:1-20`), gravity comes from matter, and the home planet is the relaxed figure of a cube of its own matter. v5 extends that rule to everything above the materials.

A seed now defines a cloud of matter. The cloud collapses into isolated systems, and inside each system the matter accretes into bodies under the game's own gravity law. Each body's composition is run through the law's own transfer process, and the minerals that come out set its density and strength. Those two numbers decide its shape through the genesis table the project already uses for home. Then each body gets a short planetary history on a coarse grid of its surface: plates, impacts, climate from the real sun path, drainage, and life. Terrain, rock layers, landforms, plants and buildings are cheap local functions that read that history.

Nothing about a particular planet is written down. What remains authored is small and stated openly:
- the law;
- a start world that is round, habitable and pulls exactly 24 m/s²;
- the spawn garden;
- a vocabulary of shape programs (how a wall, a branch or a prism is drawn) whose parameters all come from conditions.

---

## 2. What changes for a player

### 2.1 Today

Every world has the same six-plus named bodies. Home has a fixed realm on each face: +Y Green, −Y Ashen, +X Dune, −X Shattered, +Z Glass, −Z Fungal (`province.rs:63-72`).

### 2.2 v5: two seeds

These are illustrative. The P1 lab produces the real distributions, and the targets in section 6 are tuned against it.

**Seed A**
- **Start world.** A rocky, temperate, round world. Spawn is on a meadow in the interior of a continental plate.
  - About 2,000 km away (a flight, not a walk), a mountain belt runs along a plate boundary. Beyond it is a rain-shadow desert, and further on a frost sea fills an old impact basin.
- **Its moons.** A soft icy moon, round because its matter is weak (Π_g above 16 at its size). Two hard little cubes with sagging edges, because strong matter at that size cannot relax.
- **Six other systems:**
  - a weak giant that went hydrostatic, a near-perfect sphere with a debris ring and four moons;
  - a contact binary: two strong cubes locked face to face, with a weightless canyon between them. This is the Twins, arising from physics in this seed;
  - a hot world whose crust is mostly a glowing mineral and lights its two moons;
  - several airless dwarfs, saturated with craters because nothing erases them.
- **Where cultures live.** Only the start world and the giant's largest moon have enough life for cultures. Derelicts drift in the belts near them.

**Seed B**
- **The nebula.** Volatile-rich, so the systems are mostly icy.
- **Start world.** Denser matter (amount 6), so it is smaller, with radius about 2.6e7. Spawn gravity is still exactly 24 m/s².
- **Eleven other systems:**
  - mostly soft round moons, frost seas and very few cubes;
  - no binary;
  - one large, strong, low-gravity world. Its arches and pillars stand far taller than anything at home, and its trees grow into giants because buckling under gravity is the only height limit (capped by the engine at `MAX_ABOVE` 320).
- **Rock and life.** Rock layers on each world are that world's own minerals, so canyon walls are banded in colours seed A never shows. Plants are built from per-world species genomes.

### 2.3 What is still guaranteed and what is not

**Still guaranteed on every seed:**
- a round, charted start world with air;
- spawn at its +Y chart centre, on level meadow ground near (0, y, 0);
- spawn pull within the existing 5% test (`cosmos.rs:977-1003`);
- at least 3 other bodies, enforced by the interest filter in 4.8.

**No longer guaranteed:**
- the Twins;
- Verdance;
- the Hollow and the Ember;
- a fixed number of moons;
- every realm on home;
- every structure type near spawn (unless you keep that as part of the garden exception).

---

## 3. Architecture

### 3.1 The pipeline

```
seed (64-bit) + 8 knobs + law
  │
  ├─ [1] nebula        32³ integer field, collapse phases                 creation
  ├─ [2] systems       watershed basins → systems that cannot pull each other   creation
  ├─ [3] accretion     parcels merge by the law's range → bodies, contact      creation
  │                    binaries, satellites, debris, impact history
  ├─ [4] minerals      the law's Contact process per body → mineral suite    creation
  ├─ [5] shape         Π_g → genesis table → cube / warped cube / round       creation
  ├─ [6] traits        mass, g, heat, glow, air, temperature, life, name      creation
  ├─ [7] storage       charts packed by layout, never panics                  creation
  ├─ [8] globe         cube-sphere field per body: plates … climate … life     creation (start body)
  │                                                                          lazy + prewarm (others)
  ├─ [9] point detail  globe read + province lattice + existing noise          per column
  └─ [10] forms        geology rule, grown plants, crystals, cultures          per column (site grids)
```

Stages 1 to 8 are global, coarse and deterministic, and run once per world by the server and by every joining client. Stages 9 and 10 stay what generation is today: pure functions of (seed, cell), batch output equal to per-voxel output, and point-evaluable for the far field.

### 3.2 What "InfiniteDiffusion" means in v5

The original pattern (Goslin, SIGGRAPH 2026): refine a deterministic hashed prior through phases, where each phase reads its neighbours only from the previous phase. The deleted v2 crate implemented this as Q16.16 integer math with a Jacobi stencil, so neither tile order nor query order could change any value. Its source is still readable:

```
git show ec10f71^:crates/infinite-field/src/intfield.rs
git show ec10f71^:crates/infinite-field/src/inoise.rs
```

v5 revives it as `crates/field`, with two changes:
1. **No neural denoiser.** The "scores" are local integer stencils: collapse, plate growth, erosion, advection, adjacency.
2. **Exact aprons instead of blended overlaps.** Because the stencils are local and Jacobi, a tile padded by Σ(phase radius) reproduces the infinite field bit for bit. That removes seams, the cross-tile dependency cone and the shared cache. The old crate used one LRU behind a Mutex per field, which would contend across the 1 to 12 workers (`pipeline.rs:1409-1411`).

The same engine runs on three kinds of grid:

| Topology | Where | Size |
|---|---|---|
| `Box3`, finite | nebula | 32³ |
| `Sphere`, finite cube-sphere | per-body globe | 6·(G+1)², G ≤ 128 |
| `Lattice3`, infinite, evaluated as a cone | province adjacency (k=1); caves later, gated | 27 priors per node |

The far field never needs tiles. That is the constraint that sank v2 (WORLDGEN-DIFFUSION-V2 §2.6): far LOD point-samples the exact generator at strides up to about 23 km, with no prefilter (`section.rs:12-13, 362`).

### 3.3 Baked at creation versus evaluated per column

| Work | When | Who pays |
|---|---|---|
| Nebula, systems, accretion, hierarchy | creation | server and every joiner |
| Mineral suites, interned in fixed order before any worker runs | creation | same |
| Π_g, layout, datum, warp, relief harmonics | creation, table lookups | same |
| Storage allocation, cluster catalog from debris | creation | same |
| Start-body globe | creation, synchronous | same |
| Other globes | first touch through `OnceLock`, prewarmed nearest-first on a spare thread | first worker or prewarm |
| Species archetypes, culture seats | per body, lazy, `OnceLock` | first touch |
| Height, surface, strata, caves, features, structures | per column or voxel | workers, main-thread point queries |
| Province lattice values | per column, memoised in a thread-local ring | workers |

Never done:
- simulations over a whole planet at voxel resolution;
- interning materials at runtime;
- shared Mutex caches on the hot path;
- libm calls on any path that decides cells.

### 3.4 Data model

```rust
// cosmos.rs: Body keeps its place in every API. `kind` becomes `traits`.
pub struct Body {
    pub id: u16,
    pub centre: [i64; 3],      // snapped to 16, as today (cosmos.rs:298-301)
    pub shape: Shape,          // Cube{half} | Ball{r} | Shell{outer, inner}: now from the layout
    pub density: f64,          // exactly the painted bulk's mean amount (oracle == paint)
    pub seed: u32,
    pub traits: Traits,
}

pub struct Traits {
    pub rank: Rank,                  // Start | Primary | Partner | Satellite
    pub form: Form,                  // Cube | Warped | Round | Shell (Shell from P7)
    pub parent: Option<u16>,
    pub partner: Option<u16>,        // contact binary
    pub core_of: Option<u16>,        // a free core inside a cavity (P7)
    pub mass: f64,
    pub yield_stress: f64,
    pub pi_g: f64,
    pub gravity: f32,                // surface g, m/s²
    pub heat: u16,                   // Q8.8, 1.0 = start world
    pub glow: bool,
    pub air_top: Option<i32>,        // replaces the global AIR_TOP (cosmos.rs:31)
    pub temp: i16, pub wet: u8, pub life: u8,
    pub suite: u16,                  // index into Cosmos::suites
    pub name: Name,                  // [u8; 12], syllables from the seed
}

pub struct Mineral { pub block: BlockId, pub amount: u8, pub cohesion: i16, pub hardness: u8,
                     pub friction: u8, pub emission: u8, pub transparency: u8, pub thick: u16 }
pub struct Suite { pub minerals: ArrayVec<Mineral, 12> }      // densest first: core → crust

pub struct History { pub impacts: ArrayVec<Impact, 16>, pub heat_in: f64, pub age: u16 }
pub struct Impact { pub dir: [i32; 3], pub energy: f64 }

pub struct Globe { pub g: u16, pub nodes: Box<[Node]>, pub mip: Box<[(i16, i16)]> }
#[repr(C)] pub struct Node {      // 16 bytes
    elev: i16, uplift: i16, temp: i8, wet: u8, flow: u8 /* log2 accumulation */,
    rock: u8 /* outcropping suite layer */, fill: u8, life: u8, plate: u8,
    stress: [i8; 2], albedo: u16 /* rgb565 */, age: u8,
}

pub struct Cosmos {              // existing fields kept; new ones:
    pub suites: Vec<Suite>,
    pub histories: Vec<History>,
    pub globes: Vec<OnceLock<Arc<Globe>>>,
    pub reliefs: Vec<Option<Relaxed>>,   // replaces `relief: Option` (cosmos.rs:260)
    pub index: BodyGrid,                  // per super-cell body list on the 18³ grid (cosmos.rs:268-275)
    ...
}
```

`Kind` survives for one release as `legacy_kind()`, derived from traits, for the mod API and tests:
- id 0 → Home;
- `core_of` set → Ember;
- `parent` set → Moon;
- `partner` set and Cube → Twin;
- Shell → Hollow;
- otherwise → Verdant.

### 3.5 How existing systems adapt

| System | Today | v5 |
|---|---|---|
| **Charts and storage** | Boxes are laid end to end along storage x from 1.1e9 in catalog order. `check_room` panics on overflow (`atlas.rs:275-279`). Cube boxes stack in y because PosX/NegX columns are keyed by (y,z) (`storage.rs:121-123`). | Charts are allocated by layout, not kind, into shelf rows in z. Row 0 reproduces today's layout exactly. Cube boxes keep unique x and y ranges. The allocator returns an error; on error the smallest body is demoted to debris. See 4.9. |
| **Relaxation** | Genesis is consulted for home and the twins only. The table is re-parsed 2·bodies+1 times. | `tabulated_layout` runs for every body, with the table parsed once (ARCH P13 step 4 already does this). `bracket` loses its `log2`. Table v3 stores relief harmonics per entry. A cavity axis comes in P7. |
| **Gravity** | One relief slot. The oracle loops linearly over bodies. Every body has density 5.0. | One relief slot per round body (`reliefs` Vec). Per-body density. A dominance check at creation. A cull per system: systems sit more than R_G apart, so a sample only visits its own system. Kind is still never read. |
| **Relief projection** | `Relief::new` costs about 30 ms per unique datum and is cached per process (`mod.rs:416-433`). | `Relief` is linear in the datum offsets (`relief.rs:36-73`), and a table datum is a lerp of two entries times half. So h_lm (625 f64) per entry can be baked into the table and lerped. Cost drops from about 30 ms to microseconds per body. Gravity changes slightly (the f32 cast in `tabulated_layout`), so the spawn-pull tests are re-run. |
| **Sky** | `paint` matches on Kind (`sky/bodies.rs:294-330`). The list is capped at 32 in catalog order (108-110). The Hollow and Ember are a special case (158-203). | Appearance comes from globe albedo plus mineral colour. Emission comes from `glow` (generalising the Ember `SunOverride`). Impostors are chosen nearest-first. The cavity case uses `core_of`. |
| **Planet map** | A home-only bake: 25.2M `home_far_column` calls, about 50 s cold, about 24 MiB per world. | Globe albedo gives an immediate 6×G² map for the 8 nearest bodies (`MAX_FAR_MAPS` 8). The 1024² bake stays as a background refinement for the start body only. The cache key already follows the content id. |
| **LOD and bounds** | `surface_bounds` is the constant [6, 470] band for cubes and home (`mod.rs:2271-2297`). | Per-square bounds from a globe min/max mip plus an analytic bound on point-detail amplitude, which is tighter and stays a superset. Per-body envelopes come in P6. `lod_column` remains an exact point sample. |
| **Saves** | Edits are keyed by raw i32 cells, including storage cells (`save/format.rs:215-220`). A version mismatch only warns in `net/persist.rs:272`. | Every content phase bumps `WORLDGEN_VERSION` (`mod.rs:57`) and raises the refusal floor (`format.rs:79, 476-480`). `persist.rs:268-277` must refuse rather than warn, because the storage layout moves. |
| **Multiplayer** | ContentId = {WORLDGEN_VERSION, law digest, law fingerprint, palette hash} (`net/mod.rs:178-238`). Welcome carries the seed and 8 knobs. | Same shape. Every new baked table is pinned by hash to the version (precedent: `genesis.rs:791-795`). Suites are rebuilt from the seed, and `content_id` ignores interns, so nothing new goes on the wire. The seed is widened to 64 bits at the first bump (today it is folded to 32 bits at `mod.rs:526`). |
| **Knobs** | 8 × u16. `space` only scales clusters. | Still 8 × u16, so neither the save nor the protocol format changes. `space` = nebula mass, as its doc already says (`mod.rs:81`). `variety` = prior amplitude. `relief` = relief budget scale. The pwc.infinite-diffusion mod's labels change meaning, not shape. |
| **Spawn** | +Y chart centre of `bodies[0]`, settled to y≈0 (`cosmos.rs:761-764`). | Unchanged contract. The start world is id 0, round, with ρR pinned. In P3 one of 24 exact cube rotations puts its most habitable face at +Y. |
| **Mods** | dev-toolkit imports `cosmos::{Body, Cosmos, Kind, Shape}` (`../pwc-package-manager/mods/dev-toolkit/src/travel.rs:7, 63`). | `pwc_mod_api` exports `Traits`, `name()` and a class label. `Kind` stays as a deprecated derived enum for one release. /tp takes a name or class. |

Production consumers of `Kind` and what replaces each (all moved in P0 through `Traits::of(kind, shape)`, with goldens byte-identical):

| Site | Replacement |
|---|---|
| storage style (`storage.rs:91-117`) | form and painter |
| `parent_pole` (`storage.rs:29-42`) | `parent` |
| `sky_aims` moon check (`mod.rs:482`) | `rank != Satellite` |
| lush twin (`mod.rs:558-564`), twin realms (576, 587) | traits (P2), then the globe (P3) |
| home virtual cube, home salt, garden (`mod.rs:571, 581, 591`) | `rank == Start` |
| `span.rs:29-49` | `partner` (ARCH P12 step 4 already stores facing and lush per body) |
| `sky/bodies.rs` (99-107, 119-139, 158-203, 294-360, 428-434) | traits and appearance |
| `planet_map.rs` (152, 399-429) | Start |
| `game.rs:1848-1862`, `query.rs:34-75` | `air_top` and `core_of` |
| `hollow_cavity` (`cosmos.rs:464-488`) | `core_of` and Shell |

---

## 4. Algorithms

All creation-time arithmetic is either integer (Q16.16 with i64 temporaries, wrapping) or IEEE-basic f64: `+ − × ÷`, `sqrt`, `floor`, in a fixed order. There is no libm, no `mul_add`, no HashMap iteration order, and no float reduction across threads. Sort keys are total (integers, or non-negative f64 bits) with an id tie-break. Cube roots use 6 fixed Newton steps from a bit-hack seed.

### 4.1 The field engine (`crates/field`)

```rust
pub type Q = i32;                                   // Q16.16
pub trait Topology { fn len(&self) -> usize; fn neighbours(&self, i: usize) -> &[u32]; }
pub trait Rule<C: Copy> { const RADIUS: u8; fn apply(&self, prev: &[C], i: usize, nb: &[u32]) -> C; }
pub fn run<C: Copy + Send + Sync, T: Topology + Sync, R: Rule<C> + Sync>(top: &T, cells: &mut Vec<C>, rule: &R, passes: u32, threads: usize);
```

- **Jacobi double buffer.** Each pass writes the next buffer from the previous one only. Threads write disjoint index ranges, so the bytes are identical for any thread count.
- **`Sphere(G)`.**
  - Gnomonic face coordinates (a = p_u/p_n, b = p_v/p_n), so sampling costs two divisions and no `atan`. `DatumField::at` goes through `Map::inverse` (atan, `chart.rs:74`) and must not decide cells.
  - The neighbour table is built once per G (`OnceLock`).
  - Each edge and corner node has one canonical owner (the lowest `FACES` index), and a glue step copies the owner's value to its duplicates after every pass. Seams are therefore exact. The 8 valence-3 corners are the declared exceptional regions (`seam.rs:1-13`).
- **Global passes** (priority flood, sorting) are exact single-threaded algorithms with heap keys (value, index).
- **`Lattice3`.** A node's value after k phases depends on (2k+1)³ priors and is computed as a cone, with a thread-local memo ring keyed by (node, phase), like the cellular cache in `noise.rs:226-248`. Tiles with exact aprons are available but used only by gated features.
- **Noise vocabulary.** `inoise` is revived from ec10f71^ (integer value noise, fbm, ridged).
- **Tests:**
  - query-order independence;
  - apron tile == whole-grid result, bit for bit;
  - thread count does not change the bytes;
  - globe seam equality across edges and corners.

### 4.2 Nebula (`src/world/terrain/nebula.rs`)

1. **Grid.** 32³ cells of 2^26 blocks, spanning ±1.07e9. Per cell: mass `m: u32` (Q16) and composition Δ: `[i8; 4]` in [−96, 96], an offset from a base `Element` chosen by the seed (`crates/material/src/element.rs`, 4 periodic u8 axes). Blending stays linear and needs no torus trigonometry. Size: 256 KB, freed after creation.
2. **Prior (phase 0).**
   - `m = ONE + A·n²`, where `n` is integer fbm at 4, 8 and 16 cells plus a ridged term for filaments. Squaring gives a heavy tail, so some seeds make giants and some make swarms. A scales with the `variety` knob; total mass scales with `space`.
   - **Origin well:** `+ W·max(0, 1 − d²/r²)` around the origin cell. This is the only authored mass, and it guarantees a start system.
   - Composition: four independent low-frequency fields with wavelengths of 8 to 16 cells, so neighbouring systems share a chemical family.
3. **Collapse (phases 1 to 8).**
   - φ = two 6-neighbour box-blur sweeps of m.
   - Each cell finds its largest-φ neighbour of the 26 (ties go to the lowest index). If that φ exceeds its own, the cell sends `m >> 2` to it.
   - `m' = m − sent + received`. Mass is conserved exactly.
   - Composition becomes the mass-weighted mean (i64 accumulators).
   - Filaments drain into knots and voids empty.
4. **Watershed.** Each cell points to its steepest-ascent neighbour. 15 rounds of pointer jumping give every cell its sink. Per basin, accumulated in cell-index order: mass, mean composition, mass-weighted centroid, and member cells.

Cost: about 10M integer operations, an estimated 3-5 ms.

### 4.3 Systems

- Sort basins by (mass descending, sink index). The origin basin is always accepted first.
- Accept a basin as a system if its centroid is at least `D_SEP` = 4.0e8 from every accepted system and within ±7.7e8.
- Otherwise merge it into the nearest accepted system if that system is within 2·D_SEP. Failing that, its mass becomes residual debris (4.10).
- **Invariant, checked after accretion:** bodies of different systems are more than R_G + reach_a + reach_b apart (R_G = 3.2e8, `gravity/kernel.rs:16`), so systems never pull on each other. A body that breaks the invariant is merged into its own primary.
- The number of systems emerges per seed. Expected roughly 4 to 15; packing caps it near 25.

### 4.4 Accretion, binaries, hierarchy (`accrete.rs`)

**Parcels.**
- Each system gets N_p = 256 parcels.
- A parcel picks a member cell with probability proportional to cell mass. Its position is the cell centre plus a jitter sampled by rejection over the cell.
- Mass: a truncated power law with α=2, `m = m_min / (1 − u·(1 − m_min/m_max))`, which needs division only. Masses are then normalised to the system's mass.
- Composition: the cell's composition plus a small hash jitter.

**Merge rule, tied to the law.**
- Two bodies merge when `d³ < A·(m_i + m_j)`, with `A = R_IN³ / M_home`.
  - M_home = 5·(5e7)³ = 6.25e23, today's home matter.
  - A ≈ 6.55, so a home-mass body clears about R_IN = 1.6e8.
  - The test compares cubes, so no cube root is needed.
- With ρ ≤ about 8 the merge radius is about 2·cbrt(6.55ρ) ≈ 6.4 body halves, so survivors never overlap.
- Each round:
  1. Bucket bodies on a grid whose cell is the merge radius of the largest mass.
  2. Gather candidate pairs from the 27 neighbouring cells.
  3. Sort by (d³/(A·Σm) as f64 bits, i, j).
  4. Merge greedily, at most once per body per round.
  5. Repeat until no pair qualifies (about 10-15 rounds expected).
- A merge records:
  - the mass sum, centre of mass and mass-weighted composition;
  - `heat_in += G·m_i·m_j / ((r_i + r_j)·m)`;
  - an `Impact{dir, energy}` in the larger body's frame (the 16 largest are kept).
  The smaller id survives.

**Contact binary.** A qualifying pair becomes a contact binary instead of merging when all of these hold:
- the mass ratio q ≥ `Q_BIN` (0.4);
- both bodies lay out as Cube at their own size (4.6);
- the impact energy per mass is below `K_BIN·Y/ρ`, meaning the matter is too strong to fuse.

The pair is then placed face to face along the dominant axis of its separation. The gap is the smallest multiple of 16 at which each face's own pull is at least `K_DOM` times the partner's pull at that face, computed with the oracle's cube primitives. The pair takes part in later merges as one mass at its centre of mass. Today's Twins (half 6e6, gap 1.5e6) are one possible outcome. The energy threshold is a modelling choice and is labelled as such in the code.

**Hierarchy.**
- Visit bodies by ascending mass. For each, the candidate parent is the heavier body whose windowed pull at this body's centre is largest (`gravity::kernel::window`, `kernel.rs:24-35`).
- The body stays a satellite only if its own surface pull is at least `K_DOM` (4, tuned) times the parent's windowed pull at its surface, which keeps the tilt under 14°. Otherwise it merges into the parent, and the merge is recorded as an impact.
- Repeat until stable. Satellites of satellites are allowed. A body that no heavier body reaches (beyond R_G) is a primary.

This replaces today's "5-7 parent reaches" rule, which by my estimate is not dominance-safe for small moons of home:
- At 5 reaches (about 1.55e8, inside R_IN), home's pull is about 2.718e16/(1.55e8)² ≈ 1.13 m/s².
- A moon of r = 6e5 at ρ = 5 pulls about 0.55 m/s² at its own surface.
- So the parent's pull is about twice the moon's own. Even at 7 reaches the window leaves about 0.41 m/s².

This should be verified with `gravity_tour`. If it holds, it is a current bug on its own.

**Sizes.** half = cbrt(m/ρ)/2, snapped to atlas alignment. Bodies with half < 3000 become debris rocks, because the rock classes cap at radius 3000 (`cosmos.rs:238-244`).

**Rings.** Sub-threshold parcels within 3-5 reaches of a body, if their total exceeds `RING_MIN`, become a Belt cluster around that body. Its axis is the normalised sum of the impact tangents. All other leftover parcels go to debris (4.10).

**Bounds.** Bodies outside ±9.2e8 (±9.5e8 for satellites) after the spawn translation are demoted to debris. At most 64 bodies.

### 4.5 Minerals: differentiation by the law (`minerals.rs`)

This is the step that makes a seed change what a world is made of. Today every seed has the same 73 law-found roles (`ROLES`, `palette.rs:59-146`; `BAKED` has 73 entries) and the same `rock` bulk.

**Who gets a suite.** Primaries, partners and satellites above half 1e6 each get their own. Smaller satellites take their parent's suite. That means about 5-25 suites per world.

**Process.**
1. Build a radial column of K = 12 reservoir configurations. Each has 8-24 occurrences drawn by hash within ring distance 48 of `base + Δ_body`. `CAPACITY` is 32.
2. Run `Contact::new` / `Contact::step` (`crates/material/src/kernel.rs:182, 314`) between vertically adjacent reservoirs: even pairs, then odd pairs, Jacobi style. Stop after 48 sweeps or when every contact is dormant. The law kernel is integer-only, so this is deterministic.
3. **Repair.** Drop the least-held occurrence until `cohesive` passes (`palette.rs:355`). Remove empty and duplicate reservoirs.
4. Sort densest first: the densest are the core, the lightest the crust. Read `observe` (hardness, friction, emission, transparency, cohesion) and `visual` (colour).
5. **Rest check.** Every mineral must be `dormant` (`palette.rs:369`) against:
   - its neighbouring layers (true by construction);
   - every universal material that can touch generated matter on this body: the `UNDERGROUND` set (`palette.rs:151-156`), surface dressing, timber, rails, lamps and structure stone.
   A mineral that fails is replaced by the nearest palette role.
6. Re-derive reagent hosts per suite, using the same logic as `mod.rs:278-290`.

**Interning.** Suites are interned in (body id, layer) order inside `Materials::intern`, before any worker runs. Workers never intern (`mod.rs:265`). Budget: at most 64 × 12 = 768 ids, well inside BlockId and the 16,384 render descriptors.

**Derived properties.**
- density = volume-weighted mean amount of the mantle layers;
- yield = `Params::mix` of `mechanical_response(amount, cohesion01)`, the way `bulk_matter` does it (`mod.rs:361-376`). `mix` takes the harmonic yield (`material.rs:66-80`), so a body is as weak as its weak phase;
- glow if a mineral with emission > 0 is at least 30% of the crust;
- clear minerals are available for crystals and ice.

**Fallback (palette suite).** Used when any of these holds:
- fewer than 3 distinct cohesive minerals survive;
- the colour spread is too low;
- the rest check replaces more than half the suite.

The fallback maps the composition onto existing roles by class. Refractory, rock, carbon and volatile classes map to fixed lists drawn from the 13 bulk candidates and surface roles. The bulk pair is then chosen by target cohesion, generalising `choose_bulk` to any target density, with the 24-bit cut that makes the mean amount exact. No new palette roles are added: appended Glow and Clear roles would reorder the search (`palette.rs:518-525`).

**Honest status.** Whether differentiation produces varied, cohesive, good-looking minerals is unproven. The repair step may shrink reservoirs back toward the palette's amounts of 4-6. That is why P1 is a lab that measures this before anything ships.

### 4.6 Matter to shape: genesis for every body

- Π_g = Gρ²L²/Y (`genesis.rs:122-124`), then `tabulated_layout(Π, half)` (`genesis.rs:735-749`), which gives one of:
  - **Cube.** A sag of at most `STORAGE_MOVE` (0.5 block) keeps a plain physical cube with analytic gravity. A larger sag gets a `Warp` plus a storage box, the existing `cube_warps` path (`mod.rs:391-414`).
  - **Round {radius, datum}.** Each round body gets its own relief slot from the h_lm table. Π ≥ 181 is unconverged (hydrostatic), giving a near-ideal sphere.
- Worked thresholds at ρ=5, with the table's tilt crossing 25° at Π≈16:

  | Yield Y | Smallest half that rounds |
  |---|---|
  | 1e4, the softest matter (`YIELD_FLOOR`) | about 3.8e5 |
  | about 1.5e7, today's rock (home Π≈45) | about 1.5e7 |
  | 1e8, the hardest | about 3.8e7 |

  So a soft little moon is round, a hard giant stays a cube, and composition decides which.
- Π outside the table range 0.25-4096 is clamped, which should be noted in the lab output.

### 4.7 Traits

- **Surface g:** from the oracle at the +Y-equivalent face centre.
- **Heat:** `heat = (G·M/R + heat_in + E_rad·emissive_fraction) / heat_start`. It is normalised so the start world is 1.0.
- **Glow and light:** a body glows if its suite glows and heat ≥ `H_GLOW`. Irradiance on other bodies in the same system is Σ L_j·w(d)/d², O(N²) with N ≤ 64.
- **Temperature:** `T = T0 + a·heat + b·irradiance + greenhouse(air)`.
- **Air:** a Jeans-like proxy. Air is kept if `G·M/R ≥ β·T·(1 + volatile_loss)`; then `air_top = c·T/g`, clamped to [0, 40000].
  - This replaces the global `AIR_TOP` in `in_air` (`cosmos.rs:458`).
  - It also removes today's disagreement where moons have air for movement drag but are airless in `sky_altitude` (`game.rs:1848-1862`).
- **Life potential:** `comfort(T) × wet × air × carbon_share`.
- **Names:** a syllable grammar hashed from the body seed, used by /bodies, /tp and the UI.
- **Labels:** "cube world", "ringed giant", "airless moon", "glowing world" and so on are derived, for UI, mods and tests only.

### 4.8 Start world and spawn

1. **The start system** is the origin basin, and its largest body is the start world.
2. **The spawn contract** (from design 2): ρ_s·R_s = 5 × 31,017,520 = 155,087,600, so spawn pull stays as designed (g ∝ ρR inside R_IN).
   - R_s = 1.5509e8/ρ_s and `HOME_CUBE_HALF` = 0.806·R_s, the same ratio as today's 25e6 to 31.0e6.
   - Since Π ∝ (ρL)²/Y and ρL is fixed, the start world rounds exactly when Y ≲ (45/16)·Y_today ≈ 2.8·Y_today. Start selection is therefore a yield filter on the suite.
3. **Calibration.** The nebula has no absolute mass. Every mass in the universe is scaled by M_start / m_start_raw, so everything is measured against home.
4. **Requirements:** Round, air, and T inside the temperate band.
   - On failure, re-salt only the origin well's composition prior (attempts k = 1..16), which leaves the other systems unchanged.
   - If every attempt fails, use the palette fallback with today's `rock` bulk, which is known to round at Π≈45.
5. **Placement.** Translate the universe so the start world's centre is (0, −R_top, 0) through `settle_home` (`cosmos.rs:761-764`), and give it id 0. Other ids are sorted by (rank, distance to start). `home()` == `bodies[0]` still holds, so every spawn consumer keeps working: `app.rs:1290, 1554`, `game.rs:542`, `net/server.rs:2709`, `save/mod.rs:647`, `flight_bench.rs:244`.
6. **Orientation (P3).** Geology phases run in a canonical frame. Then one of the 24 exact cube rotations is chosen so the +Y face centre has the best habitability proxy: continental plate interior, low uplift, outside basins. The arrays are permuted exactly, with no resampling. Climate then runs in the world frame.
   - This keeps the "+Y chart centre" contract, unlike translating to an off-centre cell, which tilts local up by up to about 22-27° and breaks the 25 same-y multiplayer offsets (`server.rs:2708-2713`).
7. **Interest filter.** If the result has fewer than 3 bodies besides the start world, re-salt the whole nebula (at most 8 times). This is selection, not authoring, but it is listed as an owner decision.

### 4.9 Storage allocation

- Atlas sizes are as today:
  - a ball takes about 3.11R in x and 6·(n+GAP) ≈ 9.4R in z;
  - a shell takes about (π/2)R;
  - a warped cube takes 2·(half + RELIEF) in x and in y.
- Shelf packing:
  - Round atlases are placed first-fit-decreasing into rows along x, from 1.1e9 to below `i32::MAX`.
  - Rows stack along z from 0. The first version keeps z < 1e9; going further needs an audit of section keys, heightmip and save keys.
  - Row 0 is exactly today's layout when the catalog is today's, which is a P0 test.
- Cube boxes keep the existing rule: unique x and unique y ranges, with y from `STORAGE_X0`, because PosX/NegX columns are keyed by (y,z) and PosZ/NegZ columns by (x,y).
- The allocator runs inside admission and returns `Result`. On overflow, the smallest charted body is demoted to debris, so `check_room` never fires.
- Capacity: about 3 rows for home-sized bodies and many more for small ones, against today's single x row (Σ3.11R ≤ 1.047e9).

### 4.10 Clusters from debris

- **Residual density.** Debris is leftover parcels plus rejected basin mass, deposited back onto the nebula grid as residual density.
- **Placement.** Visit only nebula cells whose residual is non-zero. Within each, hash its 64 cluster cells (2^24 each) with `p = k·space·residual`. The output keeps today's cluster structure (`by_cell`, super-cell groups). This replaces the eager scan of 1.73M cells (`cosmos.rs:490-553`), which probably accounts for most of today's roughly 10 ms creation time.
- **Form.** It comes from local geometry instead of the fixed 2/7, 1/7, 4/7 split (519-523):
  - a ring around a body becomes a Belt;
  - debris between two wells whose pulls are within 2× of each other becomes a Stream;
  - anything else becomes a Swarm.
- **Rock kinds** come from the nearest suite, instead of the fixed 55/15/12/12/4/2 split (`cosmos.rs:624-631`):
  - dense minerals → Metallic;
  - clear and volatile → Icy;
  - glow → Geode;
  - low cohesion → Carbon.
  Derelicts appear only near bodies that host a culture.

### 4.11 The globe: each body's history (`globe.rs`)

**Grid.** `Sphere(G)`, with G = 128 for R ≥ 1e7, 64 for R ≥ 1e6, and 32 below that. Cube-layout bodies use the same indexing, where gnomonic coordinates equal face-linear ones.

On home, G=128 means about 380 km per node. The globe shapes continents, belts, seas and climate zones. It does not shape a walk. That is the job of 4.12 and 4.13.

**Phases** (Q16, Jacobi unless marked):

1. **Prior.** Hash noise per channel, salted with the body seed.
2. **Plates.**
   - P = clamp(1 + 12·heat, 1, 16) seed directions, rejection-sampled like `direction` (`cosmos.rs:283-294`). Each node joins the plate with the largest dot product.
   - Buoyancy comes from the amount of the crust mineral, so light crust floats as continents.
   - Each plate gets a hashed tangent velocity.
   - P = 1 is a stagnant lid, with hotspots in proportion to heat.
3. **Relief.**
   - Isostatic base from buoyancy.
   - Convergent boundaries (signed (v_a − v_b)·n) give uplift with a smoothstep falloff over boundary distance. The distance comes from a BFS in index order (global pass).
   - A trench on the denser side; rifts at divergent boundaries; domes at hotspots.
   - Amplitude follows the body's relief budget `H = min(c·Y/(ρg), cap) × relief knob`, so weak worlds are smooth and strong ones jagged.
4. **Impacts.**
   - Each recorded accretion impact becomes a basin at its direction, with angular radius ∝ cbrt(E/m) and the bowl, rim and peak profile of `round.rs:156-207`.
   - Late flux: count = exposure × age, where age = 1/(1 + resurfacing) and resurfacing ∝ heat + air.
   - Craters smaller than 2 nodes are left to point detail, through the crater-density channel.
5. **Volcanic flood.** Hot bodies fill deep basins with their basalt-like or glowing mineral up to a level.
6. **Rotation.** Start body only (4.8 step 6).
7. **Climate.**
   - Insolation comes from the real sky. The sun turns in the plane perpendicular to a = (1,1,1)/√3 (`sky/clock.rs:72-87`). Over a day, the mean light on a surface with normal n is ∝ √(1 − (n·a)²), using basic operations only.
   - The insolation poles are the (1,1,1) and (−1,−1,−1) cube corners on every body. Home +Y sits at latitude 35.3°, which matches today's temperate spawn.
   - `T = T_body + k_I·√√I − lapse(g)·elev + greenhouse(air)`.
   - Wind: thermal flow from cold to warm along the tangent gradient of T.
   - Moisture: 24 upwind advection steps from fill and volatile sources, with precipitation ∝ upslope uplift, which gives rain shadows.
8. **Erosion.**
   - 8 thermal passes, with the talus angle from the outcropping mineral's friction and cohesion.
   - Bodies with air also get fluvial erosion:
     - priority flood (global, heap key (elev, idx));
     - D8 receivers that cross face edges;
     - accumulation in sorted order;
     - 4 stream-power iterations with incision ∝ √A·S.
   - Erosion depth decides which suite layer outcrops (`rock`). Deposition basins take the softest mineral.
   - Airless bodies skip fluvial erosion, so their craters survive.
9. **Fill.**
   - The volatile budget V = volatile share × air.
   - Nodes are sorted by elevation, each weighted by its solid angle (1+a²+b²)^(−3/2) (one sqrt), and filled from the bottom until V is used.
   - Fill kind: frost over ice when cold (still water already reads as frost over ice, `round.rs:57-63`), salt flat when hot and dry, glass when very hot.
10. **Moisture again, then life.** One more advection pass, then `life = comfort(T)·wet·air·carbon`.
11. **Albedo.** Per node, the colour of the outcrop or fill mineral, tinted by life, stored as rgb565. It feeds impostors and far maps.
12. **Min/max mip** of elevation, for bounds.

**Cost estimate:** about 100k nodes × about 60 passes × about 30 integer ops plus one 100k sort ≈ 20-60 ms on one thread. Row-split across 4 threads: 10-25 ms. Small bodies cost under 3 ms. Gate: the start globe must be at most 25 ms on 4 threads, measured in P1. Otherwise drop to G = 96 or 64.

### 4.12 Point detail: reading the globe

1. **Globe read.** From the body-space direction, take the gnomonic (face, a, b) with two divisions, then bilinear over 4 taps. The batch path (`columns_16`, `shape.rs:215-248`) and the per-voxel path read the same 4 nodes, so the parity tests (`tests.rs:38, 64, 703`) hold unchanged.
2. **Province lattice (k=1).**
   - Lattice nodes sit at the existing province sites: cellular3 on the body-space surface point at about 4.5 km (`province.rs:1-6`). Each node's prior is the globe sample there plus a hash.
   - One Jacobi phase over its 26 neighbours applies adjacency rules:
     - foothills: `uplift' = max(own, ½·neighbour uplift)`;
     - shore next to fill;
     - ash downwind of volcanic provinces;
     - ecotones: a big climate jump against a neighbour is penalised.
   - A province value costs 27 priors, about 0.3-1 µs cold (estimate), and is memoised.
   - Regions (about 60 km) use the same lattice with k=1, not k=2. That bounds cold far samples.
3. **Themes.**
   - P3 keeps today's 26 `THEMES` rows (`province.rs:270-297`) as the vocabulary. They are scored by conditions over all rows: `score = prior / (1 + ((T−T_t)/σT)² + ((W−W_t)/σW)² + affinity mismatch)`. This is rational, needs no exp, and is dithered as today.
   - Removed: `Realm::of_home`, the realm biases and tables (`province.rs:63-85, 308-430`), the lush-twin rule, and the hand-fitted geopotential climate (`province.rs:450-456, 566-568`).
   - P4 replaces row scoring with `synth(conditions) -> Theme`. It returns the same struct, with surface, sub-surface and strata from the body's minerals and organic roles, and features from affinity vectors (4.13-4.14).
4. **Height.** `Shape::field` (`shape.rs:142-190`) keeps its whole vocabulary. Its literals become functions of conditions:
   - `base` comes from globe elevation;
   - the `ranges` mask comes from globe uplift;
   - the valley field is warped along the globe's downhill direction, with depth ∝ √A;
   - the dune wind is the globe wind;
   - the terrace step is the outcrop layer's thickness;
   - mountain amplitude is multiplied by clamp((Y/ρg)/(Y/ρg)_home, 0.5, 3).
5. **The window stays until P6.** Face terrain stays clamped to [6, 470] (`mod.rs:62-64`) and charts to `RELIEF` 2048. Until P6, globe elevation is mapped into that window. Honestly, that means continents read as lowlands against highlands within about 460 blocks. Mountains kilometres high need P6.
6. **Painters.**
   - Round bodies with air and life get the face painter on a virtual cube, the path home already uses (`cube.rs:48-57`). `Charted.home: bool` becomes `Painter::Face | Painter::Round`.
   - Airless, hot or tiny round bodies keep the round painter. Its `Style` enum becomes a recipe of continuous weights over the five existing relief formulas (`round.rs:123-146`):
     - crags ∝ cold·cohesion;
     - craters ∝ (1 − air)·exposure;
     - plates and magma rivers ∝ heat;
     - land and lakes ∝ life·wet.

### 4.13 Geology: strata, intrusions, one erosion rule (P4)

This replaces about 10 landmark families: craters, sinkholes, volcanic, spires, hoodoos, mesas, ice, dunes, and monolith-like structures.

- **Strata.** A column's sequence is the body's suite layers. Thickness = reservoir share × a hash. Dip and folding follow the globe's plate `stress`. This retires the 8 authored strata families (`shape.rs:318-344`).
- **Intrusions** made of the hardest mineral, hashed in region cells:
  - dikes: planes whose strike follows stress;
  - sills: horizontal sheets;
  - plugs: vertical cylinders.
- **The rule.** It runs per column in O(layers). Walk down from the uncarved height with an erosion budget E. E comes from the globe (age × wet × slope) plus noise.
  - For each layer, the cost per block is `1/(1 − r + ε)`, where r is resistance from `hardness`.
  - Remove `min(thickness, budget/cost)` and charge the budget.
  - Hard caps stop erosion, and where E breaks through a cap, the soft rock below goes fast.
- **What emerges:** mesas, buttes and hoodoos from hard caps; walls and monoliths from dikes; volcanic necks from plugs.
- **Arches.** Overhangs exist only inside bounded sites. A span is kept while it is at most √(Y·t/(ρg)).
- **Bounds.** The rule is a point function, so `lod_column` stays exact. It lowers height and never raises it, so it never breaks the upper bound.
- **Craters.** Density comes from the globe crater channel (impact history times retention). The simple-to-complex threshold (a central peak) is ∝ Y/(ρg).
- **Volcanism.** Placed by heat at divergent boundaries and hotspots. Lava is the suite's glowing mineral, so glowing fissures exist only where the matter glows.
- **Dunes.** Today's directional dunes, gated by a loose (low-cohesion) surface, dryness and air.

### 4.14 Grown life (P4)

This replaces trees, giants, mushrooms, petrified wood, flora and bones.

- **Species**, built at body creation, only on bodies with air and temperate cells:
  - Draw 64 genomes (`[u8; 16]`). Each encodes apical dominance, branch angle and count, crown superellipsoid, leaf density, tissue and leaf roles (from organic roles and the suite), and glow tips if a glowing mineral exists.
  - Score each genome analytically per climate bin (8 temperature × 8 wetness): light capture ∝ crown area, minus cost ∝ volume.
  - Keep the best K (6 to 16).
  - "Fossil" species fit the climate before erosion but not today's. They appear as petrified or bone forests.
- **Height cap.** Greenhill buckling: h_max ≈ c·(E/(ρ_t·g))^(1/3)·d^(2/3), with cube roots by Newton. Low-gravity worlds therefore grow giants without a "giant forest" theme. Heights are still clamped to the engine caps.
- **Shapes.**
  - Space colonisation: 64-160 attractors, step 1, at most 48 iterations, pipe-model thickness.
  - The result is voxelised into an `Archetype { stamps, bounds }`.
  - 8 variants per species, built lazily in `OnceLock` slots. Reads take no lock. About 1 ms per build (estimate).
- **Instances.** Archetype + yaw + site hash on today's tree site grid. The per-voxel cost is a stamp lookup.

### 4.15 Crystals, caves, deep interior

- **Crystals** grow only where the suite has clear or glowing minerals, in voids and on cold ground.
  - Habit: 3 growth axes from the element pairs.
  - Aspect from cohesion; termination from transparency.
  - 3-12 analytic prisms per cluster.
- **Caves.** P4 keeps the tunnel and cavern machinery in `underground.rs:105-191`, now with parameters from conditions:
  - dissolution ∝ wet × solubility, where solubility is a per-suite table of `Contact` non-dormancy against the body's reagent;
  - lava tubes ∝ heat;
  - ice caves ∝ cold.
  Veins take their ore from suite composition instead of the depth table (`underground.rs:204-231`). Mines stay as authored infrastructure, with their ore taken from the suite.
- **Dissolution-pattern tiles** (3-D, exact apron) are a gated option in P7 only. A cold tile under the server's random `voxel_at` (`net/server.rs:575-585`) may cost milliseconds. Also, four Jacobi phases give sharpened noise, not true labyrinths.
- **Deep interior.**
  - The six cavern biomes (`deep/cavern.rs:62`, hash%6) are gated by conditions: magma by heat, fungal by carbon × moisture, geode by clear minerals × heat, glow by glow capability.
  - The 50 km Heart (`deep/mass.rs`) becomes the dense core layer, added to the oracle as a concentric Δρ primitive.

### 4.16 Cultures and buildings (P5)

This replaces the 9 templates (`structures/mod.rs:364-376`).

- **Seats.** Culture seats are globe nodes where habitability is a local maximum above a threshold. Territories grow by a cost BFS on the globe.
- **Settlements and roads.** Settlements are placed by weighted Poisson disc: flat ground, drainage confluences, reagent veins near the surface. Roads are Dijkstra paths on the globe over a Gabriel graph with slope cost. They are stored as polylines with a segment index per globe node, a few ms per body.
- **Culture genome.** Symmetry, proportions, column spacing, roof form (flat, stepped, corbel dome), ornament rate, and purpose weights.
- **Buildings by split grammar.**
  1. The footprint comes from today's flatness check (`structures/mod.rs:167-178`).
  2. A BSP divides it into courts and rooms.
  3. Walls, openings and the roof follow.
  4. Decay ∝ age × weathering. Airless worlds keep pristine ruins.
- **Materials and size.**
  - Stone = the hardest mineral in the top layers at the site, replacing `masonry` (211-222).
  - Wall height is capped at k·Y/(ρg).
- **Building roles come from context:**
  - towers on ridges;
  - temples on the highest flat ground;
  - observatories on peaks, aimed at the brightest visible bodies (`sky_aims`, at most 8, nearest-first);
  - mine headframes over veins within `DIG_LIMIT`;
  - bridges where a road crosses a channel;
  - waystones along roads.
- **Without culture:** no structures. A body hosts derelicts only if a culture exists in its system.

### 4.17 Bounds

- **Per square:** `surface_bounds = [mip_min − D, mip_max + D]`.
  - The mip is the globe's min/max over the square.
  - D is the analytic bound of point detail: every term in `Shape::field` and every F1 or feature term is clamped to ±M_term, and D is the sum of the M_term values times the largest multipliers.
- **Superset test:** exhaustive per body type and seed.
- **Payoff:** tighter windows than the constant [6, 470] band cut far-section sampling. That pays for the extra lookups and is the precondition for P6.

### 4.18 Process constants (tuned in the lab, global to the universe)

| Constant | Meaning | Starting value |
|---|---|---|
| A | merge strength | R_IN³/M_home ≈ 6.55 |
| K_DOM | own pull ÷ parent pull for a satellite | 4 |
| Q_BIN, K_BIN | contact-binary mass ratio and strength | 0.4, tuned |
| D_SEP | system separation | 4.0e8 |
| N_p | parcels per system | 256 |
| H_GLOW, β | glow threshold, air retention | tuned |
| W, r | origin well | tuned so the start system always exists |
| RING_MIN | ring mass threshold | tuned |

---

## 5. Performance budget

### 5.1 Targets

| Item | Today | Target | Kept by |
|---|---|---|---|
| World creation, critical path (server and each joiner) | about 10 ms per seed, plus 33 ms once per process (relief) | p99 ≤ 50 ms on 4 threads; hard test ≤ 250 ms | tabulated h_lm (−33 ms); debris clusters (−scan); parallel per-body suites; row-split globe; lazy other globes; `world_creation_cost` bench pinned |
| Nebula + systems + accretion | n/a | ≤ 6 ms | integer, single pass |
| Suites | n/a | ≤ 0.5 ms median per body, about 5-25 bodies, parallel | small bodies share their parent's suite; sweeps capped |
| Start globe | n/a | ≤ 25 ms on 4 threads | G scaled by radius; gate in P1 |
| Other globes | n/a | 1-5 ms each, lazy + prewarm | `OnceLock`; prewarmed nearest-first |
| Surface chunk, home chart | 475 µs (HANDOFF-2026-10-05) | no regression > 2% (ARCH-P11 rule) | the globe read replaces region climate noise in `provinces.at` (2 cellular3 + 6 perlin3 per field); memoised lattice; no per-voxel search |
| Main-thread `surface()` | µs | µs | globe read + existing field; no tiles |
| Far section | about 4 ms (stale, v6) | ≤ 3 ms cold | point samples; tighter bounds; k=1 cones only |
| Flight bench | p95 < 2.25 ms | unchanged | gate every phase |
| Planet map | about 50 s cold, 24 MiB | globe maps in ms; old bake as refinement | 4.17 / 3.5 |
| Memory | n/a | ≤ 8 MB of globes per world; archetypes bounded | 16 B per node |
| Gravity sample | spawn about 18 µs; twin polyhedron about 57 µs | no regression at spawn | per-system cull; small cubes stay analytic |

### 5.2 How the budget is held

1. **P0 pins the benches** before any content change: `worldgen_column_cost` (`tests.rs:279`), `column_generation_costs` (2535), `home_chart_chunk_cost` (2432), `storage_chunk_cost` (`storage.rs:602`), `space_chunk_costs`, `cluster_classify_cost`, the gravity sample costs and the flight bench. None of these has a pinned value today.
2. **Every phase passes the 2% rule.** A phase that misses it does not merge.
3. **The open-path guard.** More physical cube bodies means more `SKY_EDGE` chunks on the per-voxel path (`cube.rs:20`). Measure `space_chunk_costs` on seeds rich in dwarfs, and use ARCH P12's split of `fill_slow_in`.
4. **Indexes replace linear scans:**
   - bodies through the super-cell grid, for `may_hold`, `body_at`, `bodies_touching` and `owner_in`;
   - storage boxes through P12's `column_box` binary search;
   - `Seams::atlas_at` through a sorted index.

### 5.3 Determinism guards

- **A source test** that fails if a banned call appears in `nebula.rs`, `accrete.rs`, `minerals.rs`, `globe.rs` or `crates/field`: exp, ln, log, pow, powi, powf, cbrt, sin, cos, tan, atan, mul_add.
- **Existing leaks.** `mechanical_response` uses `exp` (`material.rs:103`) and `bracket` uses `log2` (`genesis.rs:712`).
  - Both already decide cells today, through Π → home datum → `settle_home(...round())` and the twin warp (`mod.rs:533-554`). It is a low-probability latent desync between libm implementations.
  - Replacement: an exact table over integer cohesion for the yield, and exponent bits plus a fixed rational for the bracket weight.
- **Digest tests:**
  - per stage (nebula, catalog, suites, start globe) for GOLDEN_SEED 0xC0FFEE, seed 42 and the owner seed;
  - P11's worldgen byte digest;
  - compared between this machine and louise-pc.

---

## 6. Phased implementation plan

**Sequencing.** ARCH P11 to P13 (wave 1, from 2026-10-08; `~/pm-tools/space/work/tasks/ARCH-plan-2026-10-08.md`) own `terrain/**`, `genesis.rs`, `sky/bodies.rs`, `datum.rs` and `warp.rs`. P0 starts after they land and builds on what they deliver:
- P11's digest test;
- P12's sorted `column_box`, and its per-body facing and lush;
- P13's `OnceLock` genesis table.

The 23 R1/R2 review bugs land first as well. Run `cargo check` on both repos before baselines (the voxel-engine path dependency drifts).

Each phase merges and releases through the end-of-day rule with a small version bump.

### P0: Groundwork. No content change; WORLDGEN stays 10.

**Ships:**
- `Traits::of(kind, shape)`, with every production consumer of `Kind` moved onto it (3.5);
- the `reliefs` Vec;
- the body index;
- nearest-first impostors (render only);
- a per-body `air_top` equal to `AIR_TOP`;
- table v3 with h_lm per entry (gravity only);
- the storage allocator returning `Result`, with z rows;
- the libm leak fixes;
- test fixture helpers (`first_round_primary()`, `first_partner_pair()`, ...);
- pinned benches.

**Visible:** nothing in the world. Faster first-world creation (−33 ms) and no `check_room` panic path.

**Tests:**
- P11 digest identical;
- row 0 storage equals today's;
- the relief from h_lm matches `Relief::new` within 1e-6 relative;
- spawn-pull tests green.

**Acceptance:** digest identical and benches within ±2%. If the libm fixes change the digest on either machine, they move to P2's bump.

### P1: The lab (no game change)

**Ships:**
- `crates/field`;
- `nebula.rs`, `accrete.rs` and `minerals.rs` as a library, not yet wired in;
- `src/bin/cosmos_lab.rs`, which for N seeds prints histograms of:
  - body count, sizes and layouts;
  - g, heat, air;
  - suite amounts, cohesion and colour spread;
  - fallback rate;
  - storage use;
  - creation time.
  It also writes PPMs of nebula slices, suite swatches and G=32 globe previews.

**Visible:** to you, as numbers and images. You review 1000-seed histograms before any version bump.

**Tests:**
- field order independence;
- apron == whole grid;
- thread count does not change bytes;
- the banned-call test;
- digests compared on two machines.

**Acceptance gates:**
- **Suites:** at least 90% of seeds have non-fallback suites with ≥ 3 cohesive minerals. The amount spread and colour spread pass agreed thresholds.
- **Speed:** suites ≤ 0.5 ms median per body, and the start globe ≤ 25 ms on 4 threads.
- **Start world:** a valid start world in 100% of seeds, counting fallbacks.

If the suites fail, P2 ships with the palette-suite fallback, and per-world minerals move to their own later phase.

### P2: The emergent cosmos (WORLDGEN 11; refusal floor 11; 64-bit seed)

**Ships:**
- the pipeline wired into `Cosmos::with_deep`, keeping its signature;
- genesis for every body; per-body density and relief slots; dominance check;
- start-world rules, interest filter, storage by layout, debris clusters;
- names and traits; sky tones from suites;
- nearest-first `sky_aims`;
- `legacy_kind()` and the mod API shim (`/tp` by name);
- `persist.rs` refuses on a version mismatch.

**Painters stay** as they are, selected by traits:
- Start → the home path, with home realms unchanged;
- Round with life → Verdant style;
- Round and airless → Moon style;
- glowing → Ember style;
- Cube → face painter, Lush if it has life, Crystal otherwise;
- partner pairs → the `span.rs` canyon.

The start world's surface stays on palette roles. Its deep bulk comes from its suite.

**Visible:** every seed is a different universe. Count, sizes, cubes against balls, moons, rings, binaries and glowing worlds all vary.

**Tests:**
- a property suite over 1000 seeds:
  - valid start world;
  - `home_spawn_stands_on_the_plus_y_chart` and `home_spawn_pull_is_the_designed_gravity`;
  - no cross-system pull;
  - every satellite passes K_DOM;
  - no overlap;
  - storage within budget;
- 10k seeds with no panic;
- two builds equal; digests pinned; goldens re-blessed;
- kind fixtures moved to pinned fixture seeds found by the census.

**Acceptance:**
- body count median within the agreed range (proposal: 8-20, range 4-64);
- both layouts present in at least 80% of seeds (target);
- creation p99 ≤ 50 ms;
- chunk benches within 2%.

### P3: Globes (WORLDGEN 12)

**Ships:**
- `globe.rs` for every body (start body eager, others lazy);
- the 24-rotation spawn orientation;
- climate from the sun;
- realms retired; themes scored by conditions;
- `Shape::field` constants from the globe and physics;
- the face painter for round bodies with air and life;
- round-painter recipes;
- globe albedo maps for the 8 nearest bodies;
- bounds from the globe mip.

**Visible:**
- continents, mountain belts along plates, rain-shadow deserts, frost seas in impact basins;
- every planet has its own biome mix;
- far planets are coloured by their own surface.

**Tests:**
- the globe digest;
- seam equality at globe edges and corners;
- a bounds superset test;
- numeric shape tests:
  - belts exist on active worlds;
  - crater density on airless bodies is above that on active ones;
  - fill is 10-60% where volatiles are high;
- spawn properties over 1000 seeds: level ground, flora within 48 m, ≥ 3 themes within 3 km, and the garden still in force.

**Acceptance:** column and section benches within 2%. `worldgen_map` PPMs (`tests.rs:761`) reviewed with you.

### P4: Geology and life (WORLDGEN 13)

**Ships:**
- suite strata and intrusions; the erosion rule;
- physical craters; volcanism;
- species and archetypes; crystals;
- the province lattice (k=1 adjacency);
- `synth` themes; condition-gated caves, veins and cavern biomes.

Families are migrated one per commit behind P11's site engine, and the old family is deleted in the same commit.

**Visible:** canyon walls banded in each world's own minerals; hoodoos where hard caps sit on soft rock; giant forests on low-gravity worlds; different plants on every world.

**Tests:**
- the landmark envelope tests (`MAX_ABOVE`, `MAX_BELOW`);
- `features_knob_at_zero_plants_nothing`;
- `generated_surface_matter_lies_at_rest_when_disturbed` on every body type;
- the suite quiescence test against universal roles.

**Acceptance:** chunk benches within 2%.

### P5: Cultures (WORLDGEN 14)

**Ships:**
- culture seats, settlements and roads on the globe;
- the building grammar;
- masonry from minerals;
- derelicts tied to cultures;
- observatories aimed at real bodies.

**Visible:** settlements and roads that follow terrain and drainage; architecture that differs per culture; nothing built on lifeless worlds.

**Tests:**
- `structures_own_their_footprint_and_match_the_batch`;
- no structures on culture-less bodies;
- the spawn structure test as you decide it (section 7).

**Acceptance:** benches within 2%.

### P6: Relief envelopes (WORLDGEN 15)

**Ships:** per-body `MAX_GROUND`, LOD `HeightEnvelope` (`streaming.rs:3929, 3970, 4195-4201`; `heightmip.rs:103-104`) and `RELIEF`, derived from Y/(ρg) and capped by the tight bounds. This touches voxel-engine.

**Visible:** kilometre-scale mountain belts on strong, low-gravity worlds.

**Acceptance:** far-section cost does not regress on tall-relief seeds; flight bench p95 < 2.25 ms.

### P7: Physics extras (each optional, separately versioned)

- **Hollows.** An offline cavity axis in the genesis table (`Spec.cavity_half` exists, `genesis.rs:46-59`), about 4 ratios × 29 entries, hours of offline solving, pinned by hash.
  - Formation rule (a modelling choice): a volatile core boils off, and the table decides whether the shell holds.
  - A dense glowing remnant stays inside as `core_of`, which is how the Ember pattern emerges.
- **Dissolution caves.** Gate: a cold server `voxel_at` at most 2× today's cost.
- **Refinement pyramid** (design 3). Levels interpolate and only add nodes, so far LOD equals near by construction, with drainage at 32-block scale.
  - Gate: a cold main-thread `surface()` at most 20 µs and a far section at most 3 ms.
  - Two of the three judges flagged the cold descent of about 0.5 ms as a conflict with the performance rule.
- **Ring rendering.**

---

## 7. Risks and open questions

### 7.1 Decisions for you

1. **Guaranteed wonders.** The Twins, Verdance, the Hollow and the Ember become outcomes, not fixtures.
   - Twins need strong, similar masses that collide gently.
   - Hollows cannot appear before P7, and after P7 only where the table says a shell holds.
   - Accept this, or ask for rarity tuning in the lab?
2. **Cube moons.** By the law, most small strong moons stay sagging cubes. This reverses the 2026-10-04 "honest as-is" balls.
3. **Spawn.** Recommended: keep spawn gravity exact (the ρR pin) and keep the spawn garden as the one authored exception.
   - Should the start world always host a culture near spawn, so that "every structure kind near spawn" (`tests.rs:1888-1912`) survives?
   - Or does that test become "structures if a culture exists"?
4. **Per-world minerals on the start world's surface.** Recommended: other bodies first. The start world's deep bulk uses its suite from P2, but its surface stays on palette roles until you review the P4 look.
5. **Saves.** Phases P2 to P6 bump `WORLDGEN_VERSION` five times. With the end-of-day release rule, each bump refuses every older save. The alternative is a long-lived branch that releases fewer, larger bumps.
6. **The interest filter** (≥ 3 other bodies, re-salt at most 8 times). It is selection, not authoring. Acceptable?
7. **Body count targets** for the lab to tune toward: the proposal is a median of 8-20 and a cap of 64. The engine caps stay at 32 impostors (nearest-first), 8 far maps and 8 observatory aims.
8. **Breaking the mod API.** dev-toolkit /bodies and /tp move to names and traits. `Kind` stays for one release.

### 7.2 Technical risks

| Risk | Mitigation |
|---|---|
| **Differentiation produces dull or degenerate minerals.** The repair step shrinks reservoirs toward amounts 4-6; `visual` colours are unconstrained (the palette already lands `cinder` on dark green, `palette.rs:139-141`). | P1 gates; palette-suite fallback; the rest check against universal roles. |
| **Emergent but boring seeds** (many airless cube rocks). | Heavy-tailed priors, the interest filter, lab histograms, a PPM review. |
| **Determinism leaks.** One libm call or explicit `mul_add` on a creation path silently desyncs peers, because ContentId does not cover code paths. | Banned-call test; per-stage digests compared on two machines. |
| **Creation cost on every join** (estimated 25-45 ms against about 10 ms today for a second world in a process). | Measured gates in P1; G scaling; lazy globes. |
| **A lazy globe stalls one worker** for up to about 5 ms on first touch. | Nearest-first prewarm thread. |
| **Cold province cones in far sections.** | k=1 only; memo rings; the section bench gate. |
| **Bounds bugs** leave holes in LOD. | Exhaustive superset test per body type; every term clamped analytically. |
| **The relief window** keeps continents within about 460 blocks until P6, which touches the engine. | Stated up front; P6 is its own phase. |
| **Storage z rows** assume nothing caps storage z. | Keep z < 1e9; audit section keys, heightmip and save keys in P0. |
| **Genesis coverage.** The table assumes Π_g is the only group, has no layering and no cavities. Entries at Π ≥ 181 are unconverged; values outside 0.25-4096 are clamped. | Lab flags clamped bodies; 3 offline off-table relaxations to check; cavities in P7. |
| **More warped cubes** increase polyhedron gravity cost (about 57 µs per sample) and per-voxel `SKY_EDGE` chunks. | Per-system cull; small sags stay analytic; `space_chunk_costs` on dwarf-rich seeds. |
| **Churn.** About 45 test sites (about 120 references) using kind fixtures; every golden re-blessed; GOLDEN_SEED, seed 42 and the owner seed 1791184794939118871 re-pinned. | Fixture helpers in P0; pinned fixture seeds found by the census. |
| **Physics liberties.** The binary energy threshold, the air proxy, buckling, arch span, plate count, hollow formation and circulation without rotation are modelling choices. | Labelled in code as versioned prototype assignments, as `mechanical_response` is (`material.rs:1-12`). |
| **Scope.** P0-P5 is a multi-week programme. | P2 alone delivers the asked-for variety in planet count and properties; P3-P4 deliver unpredictable terrain. Each phase ships whole. |

### 7.3 Found in today's code while surveying (independent of v5)

1. **The lush twin disagrees across three places.**
   - `mod.rs:558-564` picks the smaller (seed, id).
   - `span.rs:47-49` and `sky/bodies.rs:99-107` pick the smaller id.
   - In about half of all worlds the sky and the canyon spires treat the wrong twin as lush.
2. **Moon air disagreement.** `Cosmos::in_air` gives moons air (`cosmos.rs:458`), while `sky_altitude` treats them as airless (`game.rs:1848-1862`).
3. **Moons of home may not dominate their own surfaces.** Estimate: at 5 reaches the parent's pull is about 2× the moon's own. Check with `gravity_tour`.
4. **libm on a path that decides cells:** `material.rs:103` (exp) and `genesis.rs:712` (log2).
5. **The `space` knob** is documented as planet density (`mod.rs:81`) but only scales clusters.
6. **Silent drops.** Placement failure silently drops a world or moon (`cosmos.rs:340, 405-415`) while tests `expect()` each kind.
7. **Wrong role count.** The palette has 73 roles, not 77 (`grep -c 'role("'` on `palette.rs`; `BAKED` has 73 entries).