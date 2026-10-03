# Space and gravity

Gravity comes from matter. Every voxel's **amount** — the number of element occurrences in its
configuration (0 for air, up to 32) — is its mass, and the same law pulls on everything everywhere:
there is no "planet gravity", no list of bodies that attract, no special case for the start world.
A planet a player builds pulls exactly as hard as one the generator made of the same matter.

Code: `src/gravity/` (the law, analytic sources, the edit ledger, the per-world field),
`src/world/terrain/cosmos.rs` (the generated universe), `src/space/` (faces, frames, curved charts),
`src/world/seam.rs` (charts in the streamed world). Plan of record:
`documentation/notes/SPACE-ARCHITECTURE-2026-10-03.md`; the guide it implements:
`guides/pwc_space_physics.md`.

## The law

One universal constant `G` (`gravity::kernel::G`) and one softened inverse-square kernel with a
smooth finite range: exactly Newtonian out to `R_IN` (1.6·10⁸ blocks), fading smoothly to nothing at
`R_G` (3.2·10⁸), softened inside half a block so a cell never pulls on itself without bound. The
potential is the kernel's integral, so force and potential always agree. `G` was chosen once so the
start cube pulls 24 m/s² at the centre of its top face; it is never derived from content at runtime.

- **Analytic matter.** The generator describes each body as a sum of uniform boxes and balls
  (`gravity::shape`): their fields are closed forms (the Nagy prism for boxes), with monopole and
  quadrupole far fields, split where the finite range cuts through a body.
- **Edits are corrections.** Every committed edit records `amount(new) − amount(what was there)` in an
  exact integer ledger (`gravity::ledger`) — per chunk and per region, with first moments about fixed
  centres — so mining a hole or building a tower changes gravity immediately and exactly, and loading
  a chunk never does.
- **Queries** (`gravity::Field`) walk the generator's mass hierarchy (Barnes–Hut over asteroid
  clusters, closed forms for bodies) plus the ledger. A per-tick sample costs about 18 µs at spawn.

`/gravity` prints the local field: strength, the down direction, its tilt off the nearest grid axis,
the potential, the error bound and the source epoch.

## Standing anywhere

The player has a **body frame** that turns toward the local pull (smoothly, at render rate) and
holds still in weak gravity, so walking off the top face of a cube onto its side is a continuous
walk over the edge. Collision stays axis aligned: the player's box stands along the grid axis
nearest to their up, switching with hysteresis. Jumping pushes along the support normal; flying
moves in the body frame.

Because gravity is the honest vector sum, a huge cube is not "flat": each face is a vast shallow
bowl — level at its centre, tilting 16° halfway to an edge and 45° at the edge, with the corners the
highest ground of all.

## Round worlds on curved charts

A voxel grid cannot make a round world whose ground is level everywhere. Round bodies (Verdance, the
Hollow's two surfaces, the Ember and the moons) are instead painted on **curved charts**: six
cube-sphere charts per depth band (equiangular), the angular resolution halving as the radius
halves, down to a Cartesian core. Each chart's cells have ordinary `i32` addresses in a **storage
box** beyond the physical universe (`x ≥ 1.1·10⁹`), with storage `+Y` along the chart's up — so
streaming, light, meshing, edits, saves and the network handle them as plain cells. Only three
things know about curvature:

- **The embedding** maps a storage cell to where it really is (`space::atlas`). Rendering bends each
  chunk through its eight embedded corners; gravity edits and reach checks use the embedded point.
- **The seams.** A chunk on the side of a chart's box reads its neighbour across the seam through
  the atlas glue (mesher halo, light, collision), and streaming unfolds the neighbouring charts
  around the player into one flat net, so a chart edge is as invisible as a chunk border. The eight
  cube corners of each band and the band interfaces are the declared exceptional regions.
- **Motion and picking** run in the storage frame of the chart under the player: velocity, gravity
  and the body frame pass through the chart's local Jacobian, and aim rays march storage cells,
  continuing into the next chart when they leave a box.

Which bodies get charts is a choice the generator makes about the world's initial layout; physics
never asks what kind of body it is standing on.

## Travel

Flying is fast enough to cross a face; distant worlds are found by seeing them in the sky and reached
with `/tp <x> <y> <z>`, which stands the player up along the local pull on arrival. Bodies are
anchored: no orbits, velocities or collisions between bodies yet.
