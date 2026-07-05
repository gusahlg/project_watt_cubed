# Vision thoughts — responding to features-2.md (2026-07-06)

A separate file, as requested: reactions to the refinement draft's open
questions plus new ideas that follow from them. features-2.md is a thought
process, not a decision log — so these are arguments, not implementations.

## The open questions, argued

### Should electricity be called something else? — Yes: call it Watt.
The name is already in the title. "Project Watt Cubed" begging for its core
energy/signal system to be *watt* is almost too clean to pass up: blocks
carry watts, conductors have watt-loss, machines are watt-hungry, the meter
mod shows "12W". It is iconic (the redstone effect), it is genuinely a unit
of power so it never fights physics intuition, and it gives the community a
noun ("a watt farm", "watt-starved"). The signal-vs-power split can stay
under one name: instantaneous signal = watt PULSE, sustained power = watt
FLOW. One word, two verbs.

### Temperature: agree with the draft — gate it on smelting, and note the
cheap middle path. Full per-block temperature simulation is the expensive
version. If smelting ever lands, there's a middle implementation that costs
almost nothing: temperature as a *derived, local* quantity (distance-decayed
from heat-source blocks, computed only when something asks — a smelter block
checks "am I hot enough?" by scanning its neighborhood once). No global
field, no per-tick diffusion, no memory. That keeps the door open without
paying the simulation tax the draft worries about.

### Multiple combining methods: agree — wait. One refinement worth writing
down now: when new methods come, make them *states, not machines* (the
draft's own instinct). E.g. "wet blocks combine differently than dry ones"
is fundamental; "put items in the magic box" is not. The state vocabulary
(wet/hot/compressed/energized) can grow one axis at a time.

### Transparency/light emission: fine to cut from the ELEMENT model, but
note: the renderer already supports per-block textures and light emission is
one of the two obvious future levers for atmosphere (the other is AO). If
they leave the core property list, consider keeping them as *special*
properties (the existing scaled-with-percentage mechanism) so Lumin stays
possible without a core slot. That halves the property table while losing
nothing iconic.

### Density + friction: keep — and here is why they're worth their cost.
The draft's "moving things richer than Minecraft's piston model" is, I
think, the single biggest differentiator available to this game. Concretely:
a *moving assembly* is a set of blocks that moves as one rigid body, with
membership decided by the friction-vs-density rule the draft already
defines. Three notes from the implementation side:
1. The rendering primitive already exists. Camera-relative rendering gives
   every mesh a per-draw offset; a moving assembly is exactly "a small mesh
   with a changing offset." No engine work needed to draw one.
2. The membership rule (friction >= neighbor density → drags it) is a flood
   fill at motion-start — cheap, cacheable, and *fun to engineer around*
   (players will lubricate contact layers with low-friction blocks — ice! —
   and anchor things with dense ones — lead!). The elements added yesterday
   accidentally built the exact palette this mechanic wants.
3. Collision of a moving assembly against the static grid can start brutal
   (assembly stops if ANY cell would overlap a solid) and get smarter later.
   Brutal is still leagues beyond pistons.

## New ideas in the vision's direction

- **Watt as the tech-tree spine.** Tier 0: break blocks by hand. Tier 1:
  thermoelectric trickle (Copper's HeatToElectricity next to something hot —
  or just sunlight-warmed stone for a day/night rhythm). Tier 2: real
  generators + storage (Sulfur/Quartz). Tier 3: computation. Each tier is
  just elements + one reaction, no new systems.
- **Signals are edits.** The multiplayer protocol already syncs block edits;
  a watt-pulse traveling is state-change on conductive blocks. If pulses are
  server-relayed *events on the existing edit channel*, multiplayer circuits
  come nearly free — worth designing the sim's data model around from day
  one (sparse conductor nets, per-net not per-block state).
- **Caves as the underground biome axis.** With caves growing by depth
  (landing tonight), depth bands gain identity: hairline crack → tunnel →
  cavern → the deep voids. Ore tiers already track depth; cave SIZE now
  does too, so "explorable space" and "reward" correlate naturally. A future
  cave-only element (bioluminescent? Lumin veins clustering on cavern walls
  via a surface-adjacency rule in generation) would make deep caving
  visually distinct at ~zero cost.
- **Sky and depth as mirrored progressions.** Islands now start small and
  grow with altitude; caves start small and grow with depth. The world has
  a shape: a habitable band in the middle, wonder expanding in both
  directions. Lore/goal candidates: the biggest islands (sky continents)
  and the deepest voids are where endgame elements live. Aerium up, dense
  exotic (unnamed — "Gravium"?) down.
- **Menus-as-mods generalizes to HUD-as-mods.** After tonight, every menu
  is a model the core owns and a mod renders. The same split fits the HUD
  (crosshair, coords, help line are hardcoded in game.rs today): a HudModel
  with typed widgets, one default mod. Then a "minimal HUD" or "streamer
  HUD" is a mod swap. Cheap follow-up once the menu pattern proves itself.
- **A mod manifest format, eventually.** All mods are compiled-in today.
  The Mod trait is already narrow enough that a text-manifest + wasm (or
  script) boundary could load community mods safely later. Not now — but
  every hook added (menus tonight) should keep serializable signatures so
  the boundary stays possible.
- **Memory guardrail idea** (draft worries about memory, rightly): a
  standing budget line in the bench output — `WATT_BENCH` printing max-RSS
  next to fps — so memory regressions get caught by the same reflex that
  guards frame rate. Trivial to add; large cultural value.

## Things deliberately NOT done tonight
Acting on the property-set trim (the draft is explicit thought-in-progress);
renaming electricity anywhere in code; smelting/states-of-matter. These need
your decisions first — the arguments above are inputs to those decisions.
