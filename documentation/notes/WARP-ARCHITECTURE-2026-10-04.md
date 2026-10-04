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
- **Same physics for every body.** Every planet is plain matter on a cube grid; its round shape comes from
  the relaxation law. The fixed cube-sphere charts are removed. Visible shear near cube corners accepted.

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

## 4. Phases

| Phase | What | Who |
|---|---|---|
| W0 | `mechanics`: lattice, materials, solver, gravity tree, validity; `warp_lab` | me |
| W1 | Every body is cube matter: per-body face realms, hollow cube with inner faces | grok |
| W2 | Embedding switch: bodies in storage on lattices; charts, seams, chart painters removed | me + grok |
| W3 | Generation relaxes every body; saves store lattice state; worldgen 9 | me |
| W4 | Gravity from deformed elements | me + grok |
| W5 | Background relaxation, epochs, protocol, cage refresh | me + grok |
| W6 | Far-body impostors from the deformed shape | grok |
