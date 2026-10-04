# The warp: matter that deforms under its own gravity (guide stages 5 + 6a)

Plan of record for the 2026-10-04 round. Guide: `guides/pwc_space_physics.md` §§5, 7.4–7.7, 8, 9, 10.2,
17 (stages 5, 6). Predecessor: `SPACE-ARCHITECTURE-2026-10-03.md` (law, gravity, charts).

## 0. Owner decisions (2026-10-04)

- **Already relaxed at spawn.** The generator places matter; physics solves its equilibrium under
  self-gravity; that deformed shape is the world. No settling event when you arrive (guide §10.2).
- **Background relaxation.** The server keeps re-solving as mass moves, paced by material creep; a
  player-built mass deforms by the same law once it is big and weak enough.
- **Scope:** bounded deformation (stage 5) and yield/creep (stage 6a). No moving rigid pieces (stage 4),
  no fracture (6b), no retiling (7).
- **Same physics for every body.** Every planet starts as plain matter in a cube; its shape comes from the
  relaxation law.
- **Physics shape, fitted grid** (decided after the warp_lab measurements: a cube grid bent all the way to
  a ball shears its corner cells far past the guide's thresholds). Physics relaxes each generated body's
  cube of matter and decides its shape. A body that rounds is laid out on the cube-sphere charts with a
  **datum** fitted to its relaxed surface (`space::datum::DatumField`, guide §10.2: generation may choose
  a convenient initial layout); a body that stays close to its cube keeps its cube grid. Runtime
  deformation then bends whichever grid by the same law. So the charts stay.

## 1. State (guide §5)

| Space | Here |
|---|---|
| Material configuration | block id → configuration (unchanged) |
| Patch coordinates | a body's **reference grid**: ordinary `i32` cells in its storage box |
| Reference geometry | the undeformed grid (a cube of matter + one element of air around it) |
| Current physical space | `x = φ_b(X)`: the body's **lattice** maps reference to physical |

- Every big body (start cube, twins, Verdance, the Hollow, the Ember, moons) is matter on its own
  reference grid in a **storage box** beyond the physical universe (`x ≥ 1.1·10⁹`), one box per body.
  One box means no chart seams: the reference grid is a plain Cartesian cube everywhere.
- Small bodies (asteroids) stay in physical space with the identity map: their equilibrium deformation
  under the same law is far below half a block (Π_g ≪ 1), which the representation does not resolve.
- **The lattice** (`mechanics::lattice::Lattice`): a regular hex lattice over the reference box,
  element edge `2^k` blocks (16–32 matter elements per axis, chunk aligned), storing each node's
  physical position (f64). φ is trilinear per element; this is the authoritative geometry (guide §9.3):
  rendering (a chunk inside one element is bent through its 8 embedded corners — exact for a trilinear
  map), collision and motion (local Jacobian), ray marching, edits' physical position, and gravity all
  use it. Each element is certified injective by the Bernstein coefficients of its Jacobian determinant.
- Air in the box (the building layer above the surface, a hollow world's cavity) carries no mass and
  almost no stiffness: it rides on the matter beneath it.

## 2. Mechanics (guide §8)

- **Discretisation:** the lattice elements are the mechanical elements (8-node hexes, one-point
  integration with hourglass control — mean dilatation, so no volumetric locking). Element properties
  are homogenised from the generator's matter: density = mean amount; stiffness/yield scaled by the
  matter fraction (voids do not carry load).
- **Constitutive model (one published formulation):** hypoelastic J2 viscoplasticity with the Jaumann
  rate and radial return (Wilkins), Perzyna overstress for creep: `ε̇_p = ⟨(σ_eq − Y)/Y⟩ / τ`.
  Strong bulk stiffness, slow shape relaxation (guide §7.7).
- **Parameters:** a versioned prototype `mechanical_response(configuration)` from observable material
  properties (amount, cohesion) — labelled as a prototype assignment, not a derivation (guide §7.5).
  Who rounds follows from `Π_g = Gρ²L²/Y`: blocks, buildings and asteroids are strength-dominated;
  big weak bodies relax; a strong body keeps its cube.
- **Solver:** explicit dynamic relaxation (mass-scaled, kinetic damping) to equilibrium, coarse-to-fine
  over lattice levels; self-gravity from the deformed elements by a Barnes–Hut tree, refreshed every
  few iterations (staggered, validated against tighter coupling). Deterministic: per-element results,
  fixed-order node gathers, no float reductions across threads.
- **Validity (guide §9):** each increment bounds displacement, certifies every element, and rejects
  (halves the step, then freezes the region and reports) instead of committing invalid geometry.

## 3. Lifecycle

1. **Generation:** cosmos places bodies; each body's lattice is relaxed to equilibrium (deterministic)
   and stored in the save as world state (node positions, stress, plastic strain).
2. **Runtime:** the server watches each body's mass ledger; a significant change (or residual) wakes a
   background relaxation that advances in creep time and publishes a new **geometry epoch** (node
   positions). Clients swap the embedding; chunk cages refresh (meshes unchanged).
3. **Gravity:** sources are the deformed elements (adaptive subdivision near the query) plus the edit
   ledger at the edited cells' physical positions.

## 3b. Genesis as built (2026-10-04)

- `mechanics::genesis::solve` relaxes the cube (rate-independent J2, 2×2×2 volume groups, free-surface
  pressure from a nodal field) and re-grids along equiangular directions when elements degrade;
  `choose` reads the layout from the relaxed surface: tilt over 25° keeps the cube grid, otherwise
  charts with a datum; still flowing at the representation limit with `Π_g > 100` takes the
  hydrostatic figure (reported).
- Shapes are path independent and mesh convergent; at `Π_g` 45 (the start world in rock) the cube rounds
  to corner/face 1.048 with −1.4 %…+4.9 % relief, consistent with yield-supported low-degree topography.
- A homogeneous cube's shape depends only on `Π_g`, so generation reads it from `genesis_table.bin`
  (computed offline by `genesis_table` at 16 elements, half-octave steps from ¼ to 4096) instead of
  solving at world creation (16–80 s per body live).
- Matter limits found by the law: a weak (crystal) Hollow and even a rock Hollow of today's size cannot
  hold their cavity; the moons (rock, `Π_g` < 0.1) stay cubes; Verdance in regolith (`Π_g` 3.4) and the
  twins (2.6) sag (corner/face ≈ 1.45, 1.56). The twins keep that cube grid, bent by the sag (`space::warp`): their cells live in storage and embed through the displacement. The Ember in magma (11) rounds to 1.16. Which matter the
  generator gives these bodies is an owner decision.

## 4. Phases

| Phase | What | Status (2026-10-04) |
|---|---|---|
| W0 | `mechanics`: lattice, materials, solver, gravity tree, validity; `warp_lab` | done |
| W1 | Every body is cube matter (superseded by the fitted-grid decision) | dropped |
| W2 | Layouts from physics: the start world on charts fitted to its relaxed datum (band-0 lift); sagging cubes (the twins) in storage, their grid bent by `space::warp` | done |
| W3 | Generation relaxes every body: shapes tabulated by `Π_g` (`genesis_table.bin`, pinned to the worldgen version); start world and twins wired | done for the start world and twins; the other bodies wait on the owner's matter decision |
| W4 | Gravity from the relaxed shapes: relief layer (round), polyhedron (warped cube) | done |
| W5 | Background relaxation, epochs, protocol, cage refresh (`mechanics::creep` is the step) | scope question to the owner: planet-scale relaxation never triggers at hand scale |
| W6 | Far-body impostors from the relaxed shape (engine `FarShape::Rounded`) | done |

## 5. Background relaxation (W5): what it can and cannot show

`mechanics::creep::step` advances a body in world time (elastic equilibrium with the plastic state
frozen, then the overstress relaxes by `1 − e^{−dt/τ}`). Wiring it into the server is mechanical:
per relaxed body keep its lattice and stress (rebuilt from the table geometry at world creation),
sum the edit ledger's mass changes per lattice element every few world seconds, run creep steps on
a background thread when the change can matter, and publish a geometry epoch (the new datum for a
charted body, the new warp for a cube body) that clients swap in, re-caging loaded chunks and
updating the gravity sources; saves keep the epoch geometry.

What it would show at hand scale: nothing. The start world's lattice elements are ~3.9M blocks;
moving them by half a block needs ~10¹⁶ amount of matter moved, and a player's tower on rock never
yields (`ρ g h` of a 1,000-block tower is 1.4·10⁵ against a yield of 1.5·10⁷). Creep a player can
see — a tower sinking into weak ground, an ice cliff flowing — happens at the scale of the load, and
needs **local** lattices (a few hundred blocks around a heavy load on weak matter) rather than the
planetary one. Which of the two to build is the owner's call.

