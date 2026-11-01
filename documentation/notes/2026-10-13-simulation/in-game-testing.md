<!-- SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors -->
<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# Playing with the experimental universes

Controls and observations checked against the canonical package sources on 2026-10-14.
The fifteen `pwc.physics-*` providers are alternative fundamental laws. Installing them all
makes them available; **only one provider runs on the real world at a time**. The separate
`pwc.law-lab` package already in the `laws` instance runs copied terrain in a visual
laboratory. Both systems can be installed together.

The active **laws** instance now includes all fifteen physics providers and Physics Laboratory
0.2.0, alongside Essentials, the developer toolkit and the existing visual laboratory: **43
locked packages**. The physics packages are local path requirements, so subsequent managed
builds pick up changes to their source trees. Updating all packages found no further compatible
version upgrades. The optimized release build passed.

The prior instance manifest, lock, settings and legacy world save were backed up at
`~/.local/share/pwc/instances/laws/backups/mods-20261014-172540`. No gameplay settings or saves
were changed during installation; the HUD is already Full, selective transfer remains enabled
and the experimental providers start disabled.

## Start here

Launch the updated active instance with:

```sh
pwc --instance laws run
```

1. On the main menu choose **New World** for a fresh singleplayer experiment; existing saves
   are under **Worlds**. Most
   models preserve extra state in saves, so switching them off and on resumes a run rather
   than resetting it. A new world gives clean initial conditions.
2. Press **Esc → Settings → World**. Turn **Selective Transfer Physics** off, then enable
   exactly one experimental physics option from the table below. Selective transfer defaults
   on; the others default off. If two are on, the first enabled provider in registration order
   wins. Use Up/Down to select a row and Left/Right to change its value; Enter also activates a
   toggle. Scroll down the World page to reach the added physics options. Settings changes
   select the law when you return to gameplay.
3. Keep the HUD **Full**; if hidden, press **F1** until Full returns. Minimal also shows mod instruments, but Full adds
   the coordinates useful for finding adjacent cells. Off hides the instruments.
4. Stand still in a loaded area and press that model's input chord once. For these fifteen
   providers, input and numeric sampling use **the cell containing your player**, rather than
   the block under the crosshair. Move about one voxel to sample a neighbor. Free camera
   movement does not move the player's sampling cell.
5. Most field models prepare one **7×7×7 patch** centered on the first input. Stay within three
   voxels in X, Y and Z of that first position. Further inputs outside that patch are refused;
   they do not make another patch. Packet transport and reservoir contacts have different
   activation rules and are not restricted to this same seven-cell cube.

The **Mods** menu lists what was compiled into the executable; it does not select the physics.
Hold the modifier when using a chord: bare F5 changes perspective and bare F6 toggles free camera.
These experimental preparation hooks do not inject state on multiplayer clients.

## The most revealing first experiments

**A frozen unit versus a moving pair — Dipole-Conserving Physics.** In a fresh world, turn it on
and press **Ctrl+F6 once**. The HUD should report `Q=1`, `norm=1`; the positive occupation remains
at its original position under closed evolution. Visit neighboring cells: they should remain
empty. Start another fresh world, select the same law, and press **Ctrl+F5 once**. This prepares
a positive unit and a negative unit one cell apart along +X. You now have `Q=0`, `norm=2`, and
`Dx=-1`; the neutral pair can translate. Follow it by sampling nearby cells while those totals
remain fixed. Local `q=0` means the pair has left that sample, not that the pair disappeared.
Try **Shift+F6** at a different empty cell to add a negative companion to an isolated positive
unit. Added companions can permit rearrangements. Each preparation changes the conserved totals
externally, so compare invariants only between inputs. Occupied targets refuse new preparation.
These charges are auxiliary field occupations; visible blocks and player forces are unchanged.

**A quiet medium carrying internal changes — Relational Frame Physics.** Select it and press
**Ctrl+F5 once**. This prepares a uniform seven-cell cube and disturbs its center. Watch the
`Frame`, `conflicts`, `changes` and `turns` readouts while visiting neighbors. The disturbance can
change frames at stationary sites; it does not move visible voxels. Press **Ctrl+F6** to cycle
the current cell through its six states. Rank **0** is identity; such a cell is an inert gap
and remains identity under the closed law. Try identity gaps and compare where changes reach.
There is no dedicated clear/reset chord, and the running background continues while you prepare
cells, so this is an exploratory barrier experiment rather than an automatically verified wire.
The local algebra admits a three-encounter cycle, but arbitrary 3D preparations are not promised
to make a readable clock. `Frame=-1` means the sampled cell was not prepared or is outside the
patch.

**Persistent stress versus permanent deformation — Plastic Bond Physics.** Press **Ctrl+F5**
once and watch local displacement relax while local accumulated heat grows. Walk to neighboring
cells to see how the disturbance redistributed. Press the chord again at the same cell and
compare the response: bonds retain learned reststrain, so prior loading can change the next
response. Heat is an exact bookkeeping store here; it does not yet spread or heat the player.
Compare this with Integer Wave Physics, whose disturbance keeps exchanging displacement and
momentum rather than descending irreversibly into stored heat.

## All fifteen real-world providers

Every option below is on **Settings → World**. The short name in the first column gives the
package suffix: for example, `frames` is `pwc.physics-frames`, with option
`pwc.physics-frames.enabled`.

| Package | World option label | Input | Try and observe |
|---|---|---|---|
| `selective` | Selective Transfer Physics | Ordinary placement, breaking and held-material interaction | Place different neighboring configurations and watch actual matter rearrangements. Many contacts rest because no favorable transfer exists. Repeated activity must eventually settle in a finite closed setup; this law alone has no persistent clock. No custom field HUD is provided. |
| `wave` | Integer Wave Physics | **Ctrl+F6** adds momentum 4096 | Seed once, walk to adjacent cells and compare `q`/`p` responses and timing. More inputs drive it externally. Different occurrence counts give different inertia. A net impulse can produce a drifting mean, so a growing absolute `q` is not itself evidence of instability. |
| `packets` | Conservative Packet Physics | **Ctrl+F2** adds one packet | Inject one, then several at one location; watch total `count` stay fixed between inputs and local six-bit masks change. Place/remove matter to explore transmitted versus reflected routes. Count represents signal packets, not visible flying blocks. The bit order is +X, −X, +Y, −Y, +Z, −Z. |
| `reservoir` | Conservative Reservoir Physics | **Ctrl+F3** adds 4,194,304 stored-energy units | Inject beside a material interface. Watch local energy spread and actual configurations rearrange. Compare a cold contact with an energized one. Total stored reservoir energy alone can change during chemistry: the closed conserved quantity is reservoir energy minus internal compatibility. |
| `rotor` | Memory Matter Physics | **Ctrl+F4** adds one impulse | Follow local impulse counts and rotor memory inside the patch. Inject repeatedly at the same position and compare paths; previous traffic changes routing memory. Editing matter remaps routing. Total impulses stay fixed between injections; no impulse renderer is present. |
| `fuel` | Finite Fuel | **Ctrl+F5** adds catalyst | First input supplies eight feed tokens per patch cell and one catalyst: `all tokens=2745`. Observe local feed consumption, catalyst growth and eventual spent population. Try a later injection in a depleted cell; it does not replenish feed. Later accepted inputs raise the total by one. |
| `cyclic` | Excitable Phase | **Ctrl+F5** fires a resting cell | Watch local phase: 0 rests, 1 fires, 2–5 recover. Follow the front through neighbors. Input during firing/recovery is refused. Try two seeds or altered material interfaces and look for blocked fronts or collisions. The finite front may simply die; a sustained clock is not guaranteed. |
| `loops` | Circulation Loops | **Ctrl+F5** adds magnitude-16 XY circulation | Read `Jx`, `Jy`, `Jz` around the initial square, then inject overlapping squares and watch redistribution. The +X/+Y square must fit inside the patch. The law preserves zero divergence; the HUD does not display a general energy invariant or render loops. |
| `spin` | Digital Spin Physics | **Ctrl+F5** adds 64 energy units | Read local `spin` and `local energy` at the source and neighbors. Repeated inputs can drive orientation changes; stored bins are transported by swaps. Compare matter counts prepared before first advancement: they determine frozen coupling strengths. Changing matter afterward does not retune those strengths. |
| `symplectic` | Modular Symplectic Physics | **Ctrl+F5** adds modular momentum 2048 | Track local phase and momentum around the patch, then drive at another site. Look for recurring or mixed responses. Values wrap in 0–65535 by definition; a transition from 65535 to 0 is angular wrapping. Composition-derived offsets are frozen at initial advancement. |
| `bonds` | Plastic Bond Physics | **Ctrl+F5** adds displacement 32 | Compare first and repeated loading, relaxation and retained heat. Learned reststrain supplies plastic memory. No displacement moves visible matter or pushes the player. Frozen yield thresholds do not follow later terrain edits. |
| `cooperative` | Cooperative Chemistry | **Ctrl+F5** wakes six nearby contacts | Arrange mixed neighboring configurations, then inspect `changes` and `double transfers`. A cohesive pair can sometimes transfer where either constituent alone cannot. A zero count can simply mean the chosen material contact has no favorable move. This still settles under strict compatibility ascent. |
| `permutation` | Matter Permutations | **Ctrl+F5** activates the patch | Prepare different visible configurations before activation; watch actual whole-cell swaps and the swap counter. This law moves existing matter and creates none. A uniform patch may show no swaps. Its parity predicate is a research rule, not a claim of realistic momentum or energy. |
| `dipole` | Dipole-Conserving Physics | **Ctrl+F5** pair; **Ctrl+F6** positive; **Shift+F6** negative | Compare isolated-unit immobility with neutral-pair mobility; track `Q`, `norm` and `Dx` between inputs. Use fresh worlds for clean comparisons. |
| `frames` | Relational Frame Physics | **Ctrl+F5** conjugates/disturbs; **Ctrl+F6** cycles | Explore propagation in stationary media, identity gaps and order-sensitive frame changes. Compare local ranks with the global conflict and change counters. |

For most providers the field has **numeric instruments only**. Visible particle paths, spin
colors, deformation and charge glow have not been implemented. Selective, reservoir and
cooperative chemistry can change ordinary configurations; matter permutations swap real cells.
The other providers presently carry state alongside stationary matter rather than implementing
new character movement, rigid bodies or voxel forces.

## The existing visual Law Lab

The current `laws` instance also contains `pwc.law-lab`. It does **not** claim the
real-world physics-provider slot and does not prevent the fifteen models above from working.
Its heat map is a separate simulation over a terrain snapshot. To concentrate on one system,
type **`/law off`** before using the numeric providers.

Press **/** to open command input, type a command, then Enter. A useful visual session is:

```text
/law
/law toda 16
/poke 64
/law ledger
/law pause
/law rewind 20
/law speed 1
/law off
```

`/law` lists the exact candidate ids in this build. `/law <id> 16` copies a 16³ terrain box
centered on your feet. **P**, or `/poke [strength]`, acts on the cell you look at within twelve
blocks; when no target lies inside the captured box it uses the box center. A supplied strength
persists for later P presses. `/law pause` toggles pause; `/law speed 0` explicitly pauses;
`/law speed 1` resumes at one laboratory tick per mod tick. `/law depth 1` changes the displayed
slice relative to your feet. `/law rewind 20` asks for twenty exact inverse steps and pauses on
success; irreversible candidates refuse it. This is available for the copied candidate laws,
not a rewind control for the fifteen real-world providers.

| `/law` id | A useful first observation |
|---|---|
| `demon` | Add heat to ordinary matter and inspect conduction and chemical changes in the copied box. Compare stronger versus weaker sources; inspect the conserved energy/work ledger rather than heat alone. |
| `belousov` | Excite matter and look for firing/refractory fronts. Strongly glowing copied configurations can act as recurring pacemakers. Ask whether fronts pass, collide or die in the terrain you captured. |
| `toda` | Poke a connected piece of matter, watch vibration, pause and rewind. Void carries no mass/springs in this candidate, so choose matter rather than an empty box. |
| `margolus` | Emit particles into open space and watch wall reflection. Capture a box with void or sufficiently transparent matter; opaque matter is a wall. Pause and rewind to compare a reversible trajectory. |
| `zeldovich` | Inject quanta into bonded matter and watch transport coupled to chemistry. Compare a short pulse with repeated driving and inspect the energy/work ledger. |
| `polyakov` | Deposit charge/flux, watch the field response and compare conducting versus less conducting terrain. Use the ledger to inspect constraints and dissipative loss. This is a separate field-law candidate, not the circulation-loops provider. |

This visual lab never updates the actual terrain; real-world edits appear only after you
**recapture with `/law <id>`**. Its run is session-only and is discarded when you leave the world.
Grey means resting matter; red through white indicates activity. The shown slice may miss a
disturbance above or below it, so change depth and inspect the ledger before concluding it stopped.

## Interpreting a quiet or confusing run

- **No provider HUD:** check F1, singleplayer mode and that Selective Transfer Physics and
  competing providers are off. Selective itself has no custom HUD.
- **Counters advance but local values are zero:** move around the anchored patch. A point sample
  does not show all active sites. For frames, conflicts and changes summarize the whole patch.
- **Refusals rise:** stop injecting onto occupied/nonresting targets; check domain bounds, loaded
  terrain and chart interior. These prototypes deliberately refuse unsupported seam aliases.
- **A restarted run resumes an old disturbance:** that is saved model state. Turn-off freezes
  it; it does not clear it. Start a new world for a new first-seed patch.
- **State never evolves despite successful preparation:** check that the process was not
  launched with `WATT_SIMULATION=0`, `off` or `false`; that development override disables the
  real-world simulation scheduler.

Record the provider, starting cell/materials, input count, elapsed time, observed local values
and global counters. Separate player-driven behavior from evolution with no further input. The
interesting result may be a limitation: a signal that always dies, unreadable recurrence, a
material-independent field, or a law that cannot support a useful reusable device.

## Source references

The controls and option labels above come from the canonical
[physics package sources](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/physics-lab/README.md),
[Settings menu](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/settings-menu/README.md),
[Game UI](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/game-ui/README.md) and
[Mod menu](https://github.com/gusahlg/pwc-package-manager/blob/main/mods/mod-menu/README.md).
The visual lab behavior comes from the instance's
[Law Lab README](https://github.com/gusahlg/pwc-package-manager/blob/physics-candidates/mods/law-lab/README.md)
and its six candidate package READMEs. These source-tree links describe the two existing lab
systems; the built instance should be checked in its Mods menu after installation/rebuild.
