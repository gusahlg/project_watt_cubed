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

## The universe (InfiniteDiffusion, worldgen 7)

One seeded catalog (`world::terrain::cosmos`) describes everything; physical space between bodies is
empty and costs nothing (chunks there classify as air without being generated).

- **The start cube** — 50,000,000 blocks on a side, centred 25,000,000 below the origin so its +Y
  face is `y = 0` and spawn sits at that face's centre. Each face is its own realm of 26 themes
  (meadows and giant forests, canyons and mesas, glaciers, volcanic and ash fields, crystal plains,
  fungal and glow-moss hollows, bone lands, crater fields, sky-island archipelagos…) laid out as
  regions and provinces with soft gradients between them; edges blend into a shared rim.
  **Landmarks** grow from each theme (`world::terrain::features`): giant trees and mushrooms,
  spires, hoodoos and arches, cinder cones and basalt prisms, crystal prisms, ice spires and frozen
  falls, dunes and mesas, sky islands, impact craters with meteorite cores, rib arches and skulls,
  petrified trunks, sinkholes, and flora on the meadows. Below a 350-block crust the bulk is a mix
  whose mean amount sets the cube's gravity, carved by **the interior** (`world::terrain::deep`):
  Deep caverns in six biomes with giant mushrooms and shafts, dwarf halls with pillars, stairs, rails
  and lamps, Underdark chambers with spires and lanterns, mantle bubbles, and the hollow Heart at the
  centre. The mass the generator reports subtracts the Heart and every bubble, so the pull inside
  them is the honest one.
- **The Twins** — two 12,000,000-block cubes facing each other across a weightless canyon.
- **Verdance** — a round world of radius 8,000,000: forested ranges and giant trees.
- **The Hollow** — a shell between 5,750,000 and 6,000,000 from its centre: an icy crust outside,
  crystal forests on the inner surface facing **the Ember**, a molten ball of radius 400,000 at its
  centre. Inside the shell the shell's own pull cancels: the cavity is nearly weightless, drifting
  toward the Ember.
- **Moons** — two around the start cube and one or two around each world, cratered (grey, frozen or
  rust), airless.
- **Asteroid swarms** — sparse clusters of rocks of every size and kind, never near a big body.

Seen from afar every body is drawn as an analytic impostor in the sky (`sky::bodies`), lit by the
sun, and the rocks of a nearby swarm as sunlit boxes beyond the streamed chunks (`sky::rocks`).
Standing on a cube face, the far field is face-local LOD sections out to the horizon; standing on
a round world it is chart sections bent through cages (detail stops where the cage's chord error
would pass a block), and the body's sphere fills the horizon beyond them. Inside the Hollow the
inner wall surrounds the sky and the Ember is the sun. Above the atmosphere there is no night: the
sun lights whatever faces it.

## Travel

Flying is fast enough to cross a face; distant worlds are found by seeing them in the sky and reached
with `/tp <x> <y> <z>`, which stands the player up along the local pull on arrival. Bodies are
anchored: no orbits, velocities or collisions between bodies yet.
