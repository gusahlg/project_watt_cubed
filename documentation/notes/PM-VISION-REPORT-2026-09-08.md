# Project Watt Cubed — how I read the vision, and where the project stands

Written 2026-09-08 by the project-management session (Claude), after reading every
document under `documentation/` and `issues/`, the README, and the core of the
code (app/game/world/streaming/pipeline/sched/mods/sim/block), plus today's
benchmark logs from the sibling engine session. This is opinion as much as
summary; the companion file `PM-DESIGN-ANSWERS-2026-09-08.md` takes positions on
the open design questions one by one.

## 1. The game in one paragraph

An infinite voxel world in all three axes whose materials are not "block types"
but *sets of elements*. Everything a block does — colour, texture, hardness,
conductivity, whether it is a liquid — is derived from the elements it holds,
and the world generator itself speaks that language: it places elements, and
the union of what wants to exist in a cell *is* the block. On top of that thin,
derived core sit three layers the docs care about more than the core itself:
people (massive multiplayer, proximity chat, economies, societies), mods
(every UI and most gameplay is a swappable layer), and machines (Watt as the
signal/power system, computational blocks, automation). The stated values are
Simplicity, Freedom, Community, Exploration, and the stated engineering rule is
that performance is not a feature but the enabler of everything else — big
worlds, many players, fast machines.

That reading is not a paraphrase of IDEA.md; it is what the code already
enforces. `Composition::natural` canonicalises a sorted element set, the
registry derives every property and texture from it, `placement.rs` compiles a
closed vocabulary of placement rules into LUTs, and the network never ships
voxels — a join is a seed plus the sparse edit overlay. The vision has, to an
unusual degree for a hobby project, already become the architecture.

## 2. The load-bearing ideas, and why they are good

**Elements as the universal language.** This is the idea that makes the game
distinct rather than "Minecraft with extra steps". Its power is combinatorial:
every mechanic added later (reactions, conductivity, density, corrosion) is
written once against element properties and automatically applies to every
natural block, every crafted mixture, every terrain union the generator emits.
The element-worldgen plan closed the loop by making terrain data-driven; adding
a material is one `elements!` row plus one placement rule. The discipline
around it — a closed placement vocabulary that keeps the reachable palette
enumerable and deterministic — is exactly right and should be defended against
"just one closure" forever.

**Seed + sparse overlay as the multiplayer model.** Because generation is a
pure function of (seed, coordinate, content fingerprint), the server's world is
only its edit log. That is what lets a plain box host a big world, lets clients
regenerate terrain locally, and later lets regions be sharded or snapshotted
without a database. The content fingerprint at join (worldgen version, element
table, compiled palette) is the correct guard for this model. This is a real
structural advantage over engines that stream chunk bytes, and the design
should keep protecting it: no feature may require the server to ship voxel
data.

**Menus (and soon HUD) as mods.** The pattern "core owns the model and the
meaning, a default-enabled mod owns look and interaction, with a core fallback
so disabling the mod never bricks navigation" is the right template for the
whole modding story. It keeps the core small without making it useless.

**Performance as a first-class value.** Not as vanity numbers, but because the
community/exploration goals are about scale: 20-ring view distances, hundreds
of players, machines ticking. The codebase takes this seriously in the right
places — resident GPU meshes with visibility masks instead of per-frame draw
lists, worker pools with claim discipline, byte-budgeted uploads, memo caches
keyed on exact bit patterns, uniform-chunk proofs so most of a 3D-infinite
world costs 32 bytes per chunk. The measured stripped profile at 36,000 fps
(0.028 ms) shows the CPU side of the frame is essentially solved; what remains
is GPU (engine) and world-loading throughput at large radii (game), covered in
§6.

**Determinism as a testing culture.** Goldens, census tests, byte-parity
oracles for the fused section path, protocol round-trip tests. This is what
makes aggressive optimisation safe, and it is rarer than it should be.

## 3. Where the vision is still soft

The docs are honest that these are unresolved, and I agree they are the hard
part. In order of how much they block other things:

1. **Crafting.** Everything from machines to economies depends on how a block
   is made. features-2.md circles the problem: signals vs conditions, pushing
   elements vs submitting configurations, blueprint blocks, "stardust". The
   docs' own instinct — "states, not machines"; no magic box — is the right
   constraint, but no proposal in the drafts satisfies it yet. Today the code
   has the simplest possible placeholder: the crafting mod lets the player
   combine any distinct elements from the stash into a natural block by hand.
2. **Block kinds.** Natural / mixture / configuration / computational is four
   rulesets. features-2.md already argues mixtures are a subset of
   configurations and that configurations may be "an abstraction too big". The
   code has exactly one kind with two composition flavours (natural set,
   weighted mixture), which is a fine place to be — the question is whether the
   3D sub-block configuration idea survives contact with memory and UI at all.
3. **The property set.** Ten core properties in features.md, a trim debated in
   features-2.md, and a code table (`elements!`) that stores durability,
   hardness, conductivity, density, friction, etc. without any mechanic
   reading most of them. Properties that nothing reads are design debt: they
   look like promises. The right move is either to wire the cheapest observable
   effect (mining time from hardness/durability) or to drop the column until a
   mechanic wants it.
4. **Movement and physics.** Density/friction were decided "yes" because moving
   assemblies are the biggest differentiator available. Velocity for blocks,
   collision damage, air resistance, pressure — all mentioned, none designed.
   This is the second-hardest problem after crafting and the two interact (a
   crafter that pushes things needs movement).
5. **Simulation determinism and authority.** Electricity/thermal/reactions are
   destined to be multiplayer machines, but the sim seam is a `Tick` trait
   ticking systems in registration order on each client. The structural audit
   already flagged it: design the deterministic, server-owned phase model
   before the first real system lands, or every system will need a rewrite.
6. **Planets and gravity** (world-gen.md). Charming, expensive, and orthogonal
   to everything above; the right call is to keep the 3D-infinite world and
   the "sky and depth as mirrored progressions" shape, and defer planets until
   there is a spaceship to fly.

## 4. What is built, honestly assessed

Read as a checklist against IDEA.md:

| Vision area | State | My assessment |
|---|---|---|
| Infinite 3D world, element-derived materials | Done, mature | The strongest part of the project; generator, placement, palettes, far-LOD ladder, lighting, occlusion all exist and are tested. |
| Streaming at scale | Mature, with measurable rough edges | Claim discipline, cancellation, budgets, pacing all exist. At the user's real settings (RD20/V10, 42k chunks) entry still takes ~1 minute and the main thread is the bottleneck (§6). |
| Multiplayer core | Done for the basics | Authoritative edits with acks and rollback, interest grid, plausibility-checked movement, chat with proximity/global, voice over QUIC, join by seed. Server identity is not authenticated (documented). |
| Modding | Pattern proven, surface tiny | Mod trait with ~10 hooks, menus-as-mods landed, HUD as data. Mods are compiled-in; no server-side hooks; the inventory lives inside a mod. |
| Elements/blocks | Model done, mechanics missing | Composition, derivation, reactions registry, crafting of naturals. No property has a gameplay consequence beyond solid/opaque/liquid/emission/absorption. |
| Machines / Watt / computation | Not started | Sim seam exists, empty. |
| Audio | Large and finished-looking | ~5k lines: director, acoustics (DDA window trace), Opus voice, Kira backend, capture. Proportionally the biggest subsystem after the world; well layered. |
| Tooling | Excellent | Self-describing benchmark, golden harness, stress harness, throughput benchmarks pinned in doc comments. |

Two observations about proportion. First, the thin core the philosophy wants is
currently a *thick* engine-side core (~50k lines, ~20k in `world/`) with a thin
*game*. That is fine at this stage — the core is infrastructure — but the next
year's value is in the game layer (crafting, machines, economy), and those need
a modding surface that is still embryonic. Second, the project's writing habit
(night logs, critiques with status markers, plan documents with decisions
"so we stop re-litigating") is a genuine asset; the drafts under
`documentation/features/` are the one place where the writing is stale relative
to the decisions the notes already record (e.g. features.md still lists ten
core properties and four block kinds).

## 5. Risks I would name

- **Scope gravity.** Audio, voice, benchmark metadata and licensing tooling
  are all excellent and none of them makes the game more of a game. The
  philosophy says "perfect thin core"; the risk is a perfect thick shell.
  Recommendation: for the next stretch, every landed PR should be able to
  answer "what does the player do with this?".
- **Mod-owned state.** The element stash (inventory) is created by the
  default mod set and shared by `Rc<RefCell>` between the inventory and
  crafting mods. The docs say the inventory exists without an interface mod.
  Disabling the mods today changes state ownership, not just presentation.
  Cheap to fix now, expensive after a save format and server economy depend
  on it (see the design answers, A1).
- **Properties without mechanics** (above). Design debt that reads as promise.
- **The god files.** `streaming.rs` 2.4k, `generation.rs` 2k, `world/mod.rs`
  1.9k, `server.rs` 1.7k, `settings.rs` 1.7k, `pipeline.rs` 1.7k. They are
  well commented but the comments have become essays; the philosophy asks for
  conservative comments. This is maintainability debt, not correctness debt,
  and it should be paid by code motion, never together with behaviour changes.
- **Cross-repo drift.** The engine is a path dependency that moves
  independently; the flake pins a commit. This has bitten three times per the
  memory notes. Release engineering (R-01/R-11 in `issues/`) is still open.
- **Benchmarks reading the real settings file.** `WATT_BENCH` measures whatever
  `saves/settings.cfg` says (today: RD20, vertical 10, 8× MSAA, 200% render
  scale, FOV 220). That is the honest "what the user plays" number, but it
  makes casual comparisons across sessions meaningless unless the config is
  pinned. Today's stalled default-preset run (a stray keypress opened the
  console during a bench) shows the harness also needs to lock out input; that
  fix is in flight.

## 6. Performance: vision versus measurement

The stated targets are "above 20,000 fps stripped, above 5,000 fps fast". From
the sibling session's runs this morning (RTX 3070, i5-12400F, seed 42, 10 s):

| Scenario | avg fps | ms/frame | Where the time goes |
|---|---:|---:|---|
| Minimum preset | 35,934 | 0.028 | CPU 0.03 of which fence 0.02: GPU-bound even here |
| Fast preset | 8,900 | 0.112 | GPU 0.10 (opaque) |
| Default-ish, fullscreen 1440p ultrawide, RD6, LOD on | 1,139 | 0.878 | GPU 0.95: sky 0.50, opaque 0.32, post 0.11 |
| Same, RD12 | 1,094 | 0.914 | GPU 0.99, draws 306 |
| Everything on, RD12 | 591 | 1.69 | GPU 1.76: resolve (TAA) 0.70, sky 0.50, opaque 0.38 |
| **User's real settings** (RD20, V10, 8×MSAA, 200% scale) | **90** | **11.2** | GPU 12: opaque 5.0, TAA resolve 4.3, sky 1.0-1.3; CPU pack 1.1 |

Two conclusions. Both stated targets are already met on the CPU side; the
frame in every scenario is GPU-bound, and the GPU work is engine territory
(the sky shader is half of a default 1440p frame; TAA resolve at 200% render
scale is a third of the user's frame). That is the sibling engine session's job
and I have deliberately not touched it.

The game-side problem is different and shows in the same log: **world entry at
the user's settings**. During loading the main thread spent 21-24 ms per
frame in streaming lanes — 13-18 ms of it in the occlusion rebuild, 2-3 ms each
in mesh and light admission, 1-3 ms in the drain — so the game ran at 40 fps
for the better part of a minute and the workers (10 threads, 48 ms of work per
frame available) were starved by the main thread. After loading, a 1.2-1.5 ms
per-frame tax lingered for tens of seconds while thousands of "degraded"
chunks were re-meshed synchronously two per frame, and the benchmark never
reported ready inside 60 s for that reason. Memory sat at 816 MB, of which the
per-chunk light grids (8 KiB each, allocated even for uniform sky and rock)
plausibly account for a third.

Every one of those is a game-side, well-bounded fix, and they are the perf
work I am dispatching (details in the task files and the final summary):
an occlusion rebuild that fills connectivity at budget without re-running the
BFS and 42k mask patches every frame, and diffs visibility instead of
re-patching everything; O(n) nearest-first selection in the admission lanes
instead of collecting and sorting a 20-47k-entry worklist every frame;
promoting stuck degraded chunks through the existing async rebuild path
instead of synchronous main-thread meshes; and a uniform variant of
`LightGrid` so sky and deep rock cost no light memory. Expected outcome at
RD20/V10: entry-phase frame time from ~24 ms to under ~5 ms, entry time cut
by a multiple, and a few hundred MB less RSS. None of it changes what is
rendered.

## 7. What I would do next, in order

1. Land the perf and robustness fixes above (in flight).
2. Move the element stash into the core (A1 in the design answers) and give
   the Mod trait its first server-side twin hook set. This is the seam every
   future gameplay feature crosses.
3. Decide crafting (A3). Until then, do not build machines; build the
   block-break event (health, broken state, harvest) because it is needed
   under every crafting model.
4. Wire one core property into one observable mechanic (mining time) and
   delete the columns nothing reads.
5. Design the deterministic sim substrate (active-cell table, server-owned
   phases) on paper, then land Watt as the first system.
6. Only then: moving assemblies.

The ordering rule behind this list: prefer the change that makes the *next*
change cheaper. Stash-in-core, the break event and the sim substrate are all
"pay now or pay triple later" items; machines and assemblies are "pay when
there is a game to put them in".
