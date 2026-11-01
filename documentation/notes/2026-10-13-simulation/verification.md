<!-- SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors -->
<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# Small verification record

Performed on 2026-10-13. The request emphasized ideas and little testing, so these are focused
correctness demonstrations and compilation checks. No full game test suite, GPU gameplay,
large-world benchmark, long-term stability study or multiplayer field demonstration was run
for this investigation. Existing project audit results are separate.

## Compilation and package checks

Passed:

```sh
cargo check --workspace --all-targets --offline
CARGO_TARGET_DIR="$PWD/target" cargo build --offline \
  --manifest-path examples/simulation-lab/Cargo.toml
pwc package check mods/physics-selective
pwc package check mods/physics-wave
pwc package check mods/physics-packets
pwc package check mods/physics-reservoir
pwc package check mods/physics-rotor
git diff --check
```

The native ALSA and D-Bus development pkg-config directories were supplied through the local
development environment. No new dependencies were downloaded. The packages and lab use a
shared target directory to reuse compilation. Local package checks do not publish/install
packages or rebuild a user's existing game instance. Their excluded Cargo lock/target warnings
refer to development files that are intentionally outside the source package.

## Focused tests

`cargo test --lib --offline sim::reactions::tests -- --nocapture` passed **12 tests**, covering
legacy conservation, bounded cascades, deterministic contact ordering, dormancy, saved queue
order, no-provider inertness, off/resume state preservation, unavailable-selection retention,
transactional rejection of corrupt records, and the existing zero-allocation sleeping tick.
Core legacy fixtures include the actual extracted mod implementation under `cfg(test)`;
production vanilla does not install it.

One demonstration per experimental package passed:

| Package | Focused check | What it establishes |
|---|---|---|
| Wave | `impulse_propagates_and_both_shears_invert_exactly` | 24 forward and 24 inverse steps on a fixed three-cell construction recover every stored integer; total momentum stays 4096 |
| Packets | `every_local_state_is_reversible_and_conserves_number_and_momentum` | All 64 masks across three control axes conserve count/local momentum and scatter twice to the original state; a spectator changes a head-on collision |
| Reservoir | `uphill_transfer_borrows_energy_and_conserves_occurrences_and_hamiltonian` | An uphill split is refused at zero reservoir; a reservoir of 238697 pays a gain of -238680 and leaves 17, preserving occurrence multiset and `e - Φ` |
| Rotor | `closed_square_discovers_a_cycle_and_preserves_count_across_reload` | A closed four-cell construction discovers an eight-turn state cycle with one impulse; saved continuation matches |

Reproduce a package's demonstration with:

```sh
CARGO_TARGET_DIR="$PWD/target" cargo test --offline \
  --manifest-path mods/physics-wave/Cargo.toml --lib -- --nocapture
```

Four focused server checks also passed during integration:

- `disabled_simulation_refuses_tools_and_does_not_wake_contacts`
- `protected_cells_refuse_computed_tool_products`
- `reaction_pair_admission_never_commits_half_at_the_spec_pool_limit`
- `selected_simulation_identity_is_required_at_join`

The identity check used local QUIC loopback. These checks cover the authority seam, not complete
experimental field networking. Merely conserving packet scatter does not establish that the
entire transport/reflection model is invertible; the wave test does not establish ordinary energy
conservation; the reservoir test does not establish equilibrium statistics or cyclic machines.

## Short native laboratory runs

The final built launcher ran `--ticks 32` for every model with the default 384-unit budget.
Each run serialized state, loaded a fresh factory, and asserted equality of the complete
reserved state records plus the sampled readout. Raw output is [lab-smoke.jsonl](lab-smoke.jsonl).

| Mode | Accepted turns | Model operations | Pending state/work | Matter occurrences | Record text bytes |
|---|---:|---:|---:|---:|---:|
| off | 0 | 0 | 0 | 0 | 23 |
| selective | 32 | 4 | 0 | 8 | 92 |
| wave | 32 | 10976 | 343 nodes | 0 | 4790 |
| packets | 32 | 94 | 3 active sites | 0 | 128 |
| reservoir | 32 | 11384 | 5164 contacts | 8 | 97104 |
| rotor | 32 | 32 | 1 impulse site | 0 | 291 |

Byte counts are the sum of reserved record key/value string lengths, not resident memory or
the entire binary save envelope. Operation units differ: nodes, processed sites or contacts.
No elapsed-time throughput comparison can be inferred from this table.

Inputs were intentionally small but different: selective/reservoir received the same two
four-occurrence configurations; reservoir also received `64*QUANTUM` energy; wave received a
zero-total-momentum dipole; packets received one spectator and a head-on pair; rotor received one
impulse. Empty matter in the field experiments is a vacuum-background demonstration, not a
claim that background material cannot influence their laws.

The legacy pair settles after four operations, consistent with its monotonic potential.
Packets retain three packets. The rotor run retains one impulse. Wave's sampled total momentum
remains zero while local state evolves. Reservoir energy spreads to **1515 nonzero cells** and
leaves **5164 contacts** pending after 32 turns, with no dropped work in this input. Its energy
total alone is not the invariant: chemistry exchanges energy with internal fit. That active-state
growth is a concern to investigate in later experiments, not a measured scalability limit.

## Review corrections

Read-only integration review caught and corrected unavailable-model selection being rewritten
to off, legacy contacts omitted when another model was active, missing client-side legacy marker
migration, partial mutation while parsing corrupt records, and same-id version changes without
a state payload bypassing compatibility checks. The launcher now isolates `--play` from ordinary
saves unless an explicit nonempty `WATT_DATA_DIR` is supplied. No gameplay was performed to
test its visible instruments or controls.

Preexisting changes to `src/world/quadtree.rs` (55 additions/15 deletions) and
`src/world/rule/probes.rs` (116 additions/7 deletions) were left as found. No commit or package
publication was made. The important next checks are chart-neighbor/frame correctness,
activation growth, multi-model state migration, field replication, and player experiments.
