# Ideas — brainstorm grounded in the vision docs

Living file. Everything here follows from IDEA.md / features.md /
philosophy.md (minimal core, freedom, community, exploration; optimization
ahead of readability) — plus a few leaps beyond them worth considering.

## World & exploration

- **Altitude as an element gradient.** With the 3D-infinite world, make depth
  and height mean something chemically: dense elements (iron, stone) enrich
  downward; a new low-density element (working name: *Aerium*) appears only in
  flying islands, and gets rarer per altitude band. Density is already a core
  property — lore writes itself: islands fly because their blocks average
  below an air-density threshold. Aerium later becomes the airship/hover
  ingredient. Exploration reward = vertical, not just horizontal.
- **Island shapes from density math.** Flying islands generate where 3D noise
  exceeds a threshold that RISES with altitude — automatically rarer and
  smaller with height, no special-case code. The same field can carve
  occasional caves below the surface (threshold falling with depth) when we
  want underground variety, one function, two features.
- **Biome = element distribution, not decoration.** Instead of Minecraft-style
  biomes, vary the element mix of natural blocks regionally. Rendering already
  derives block textures from composition, so biomes fall out visually for
  free once generation varies compositions.

## Elements, blocks, machines

- **Electricity as a sparse graph.** Conductive blocks form nets (union-find
  over placements/breaks, incremental). The 20 Hz sim only ticks nets with a
  live source. Conductivity as per-edge resistance; voltage drop = the fun
  constraint that makes conductor CHOICE matter. Zero cost when nothing is
  powered — matches the perf philosophy.
- **Thermal as an active-set automaton.** Temperature only simulated for
  blocks in the "active set" (near heat sources/sinks); everything else sits
  at ambient. Thermoelectric elements (HeatToElectricity special property)
  bridge the two systems: lava-adjacent generators as early power.
- **The docs' open question (language vs gates for computational blocks):
  both, layered.** Core ships logic GATES (tactile, visual, moddable — the
  docs' instinct is right). A mod then provides a tiny assembler that compiles
  a textual program INTO a gate arrangement. Core stays minimal; the language
  crowd gets served by the mod layer. This is the philosophy applied to its
  own dilemma.
- **Mixture crafter as the first interactive block.** Needs one new mod hook:
  `on_block_use(block, ctx)` (RMB on a block while not placing). The crafter
  UI itself ships as a default-enabled mod, like the inventory. The same hook
  unlocks doors/switches/chests later — one seam, many features.
- **Reactions as world events.** Corrosion slowly converts neighbors;
  explosive-at-breakage makes mining risky mixtures a gameplay decision.
  Reactions are already data — the sim just needs to walk blocks with active
  reactions in the active set.

## Rendering / performance runway

- **LOD rings.** Beyond the meshed radius, render 2x2x2-merged half-res
  meshes (same greedy mesher, coarser grid). Chunks already carry AABBs and
  the engine culls; a second "far" mesh per chunk column would let render
  distance jump to 16+ cheaply.
- **Baked corner AO.** Per-vertex ambient occlusion from the 3 neighboring
  voxels at each corner, folded into the existing shade byte at mesh time.
  Zero runtime cost, large depth-perception win. Caveat: breaks the current
  "one shade per face" greedy merge key — merge on (block, 4 corner AOs) or
  accept fewer merges on edges only.
- **Render scale** (in progress): offscreen is already decoupled from the
  swapchain; rendering at 50-150% and blitting with linear filter is nearly
  free and is THE knob for weak GPUs / battery play.
- **Chunk voxels as u8.** Palette caps at 256 (texture layers), so chunk
  storage can halve to u8 with a per-world guarantee. Combined with uniform
  chunks (single-value air/stone), deep rock and open sky become ~24 bytes a
  chunk instead of 8 KiB.
- **Pipeline cache to disk** (engine): faster startups, mainly matters once
  shader count grows.

## Multiplayer & community

- **Interest grid** (deferred, designed): bucket players by INTEREST_RADIUS
  cells; move fan-out touches 9 buckets instead of the roster. Needed past
  ~200 players.
- **Proximity voice-text hybrid:** chat already has proximity + global;
  a "shout" radius multiplier and per-message ranges are trivial protocol
  extensions mods could use.
- **Server-side mods.** The Mod trait is client-side today; a matching server
  hook set (on_edit, on_join, on_chat) would let communities build economies
  and protections without forking the server. The relay loop already has the
  event points.
- **World sharding.** Seed+overlay sync means a server's world state is just
  the edit log — snapshotting/handing off regions between server processes is
  plausible without a database. Massive-multiplayer runway.

## Memory-reduction roadmap (concrete, ordered by value/effort)

Measured surfaces today: chunk voxels (~1.8 MB dense at RD 6 thanks to
Uniform + u8 cells), GPU mesh blocks (64 MiB device arenas, one usually),
edit overlay (grows with play), engine host buffers, save files (binary v2).

1. **Staging/immediate buffer decay** (engine, in flight): empty staging
   blocks and oversized immediate buffers shrink back after bursts.
2. **Edit-overlay compaction**: an edit that restores the GENERATED block at
   a coord (e.g. placing stone back where stone was) can be dropped from the
   overlay at write time — regeneration produces it anyway. Needs a cheap
   "what would generate here" query, which block_at already is.
3. **Mesh arena block size tuning**: 64 MiB is generous for ~5 MB of live
   meshes; a 16 MiB first block + 64 MiB growth would cut idle GPU reserve
   4x on small worlds. One-line constant + a growth policy.
4. **Chunk map shrink on world exit**: free_meshes keeps chunk data for
   re-entry; a world left behind (menu) could drop Dense payloads and keep
   only Uniform tags + the overlay (regenerable). Worth it once worlds are
   big; measure first.
5. **MeshData scratch high-water decay**: same policy as immediate buffers,
   world-side (the scratch holds the largest chunk ever meshed).
6. **Server**: the edit overlay is the only unbounded state (HashMap<coord,
   String>); intern spec strings (Arc<str> table, mirroring save v2's table)
   — thousands of "air" entries currently each own a String. Cheap, large.

## Housekeeping ideas

- Bench variants: `WATT_BENCH_SCENE=islands|deep|surface` to catch
  regressions per altitude band once the 3D world lands.
- A `/where` command extension showing chunk coord + chunk kind
  (uniform/dense) for debugging streaming.
- Save format versioning header (one line) BEFORE region files arrive, so old
  saves stay loadable forever.
