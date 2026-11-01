<!-- SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors -->
<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# Bug reports

P1 means a core guarantee or durable/shared state is at risk; P2 means a narrower behavioral or
resource defect. A source-confirmed issue has a clear reachable code path but may still need an
end-to-end UI/GPU reproduction. A probe-confirmed issue has a passing test that asserts the
current incorrect behavior. The [runner](reproductions/run.py) appends these tests only to a
temporary source copy. See [verification](verification.md) for the actual test results.

## B01 — Reaction writes are not atomic; refusing the second product loses matter

**P1 · probe-confirmed · uncommon limit trigger, fundamental impact.**

**Evidence:** [`ReactionScheduler::tick_into`](../../../src/sim/reactions.rs) writes `lo`, records its
mutation, then attempts `hi`. Failure of the second write neither rolls back the first nor
suppresses the operation counter and wakeup. [`ServerCells::set_block`](../../../src/net/server/state.rs)
can legitimately refuse a write when `State::write` cannot acquire a new spec-pool slot.

**Reproduce:** `audit_reaction_second_write_refusal_loses_matter` fills the server's real
16,384-spec pool, keeps the first product already named and its old material named elsewhere,
then runs one destructive-pair contact. The first write succeeds; the second product needs a
new slot and is refused. The probe observes **8 → 7 occurrences**, with one mutation committed.
This is not merely a partial visual update: material and hence gravity can be wrong permanently.

**Expected:** both products commit, or neither does; failed admission preserves both cells and
the conservation invariant. A blocked contact should retain/retry its work when capacity changes,
according to an explicit policy, rather than silently becoming dormant.

**Repair:** give `CellStore` a pair-commit/preflight operation. Reserve the two interned product
identities and compute spec-pool effects including releases before mutating either cell. Validate
both revisions if the computation occurs outside the lock. Rollback after public `World::set_block`
is a poor substitute because it also changes gravity, revisions and mesh/light worklists.

**Acceptance:** at every one/two-product capacity boundary, total element multiset is unchanged;
refusal emits no mutations and no committed-operation increment. Apply the same rule to future
machine moves/crafting operations.

## B02 — The server checks charted players against physical-grid cells

**P1 · probe-confirmed.**

**Evidence:** [`on_move` → `body_blocked`](../../../src/net/server/movement.rs) calls `fill_body_cells`
on the physical `Pose.pos`, then queries the generator at those cells. `Seams::glue_cell` glues
storage neighbors; it does not turn a physical body into storage. Meanwhile
[`update_player_in`](../../../src/input/movement.rs) explicitly transforms client motion into the
atlas storage frame. The terrain generator states that physical space holds none of charted
bodies' cells.

**Reproduce:** `audit_physical_collision_misses_chart_ground` embeds a body sunk into seed 42's
start-world ground. The same body is blocked when the helper receives storage coordinates and
clear when it receives its physical coordinates, as the network path does. Existing
`noclip_reads_a_chart_edge_like_the_client` tests the storage helper and therefore misses this
end-to-end coordinate mismatch.

**Impact:** a client can report movement through charted ground despite noclip being disallowed,
subject to the separate speed envelope. This affects the current start world, not just a distant
optional planet. Walls built in chart storage are also invisible to that physical-grid sweep.

**Repair:** share a physical swept-body query with the client, resolving atlas/patch changes,
storage hitbox and seams at each segment. Track previously occupied cells in the resolved storage
frame, and do not reuse that set across incompatible patches.

**Acceptance:** send a real `Move` through generated and edited chart walls on multiple faces,
warped cubes and patch transitions. It must snap back under `NoclipPolicy::Off`; legitimate
seam crossings must continue to pass.

## B03 — Reactions do not follow chart seam topology

**P1 · probe-confirmed.**

**Evidence:** [`Contact::around`/`hi`](../../../src/sim/reactions.rs) use address ±1. Both
[`CellStore for World`](../../../src/world/edits.rs) and the server store read/write those raw
addresses without seam resolution. Rendering, collision and lighting instead use real glued
neighbors through [`Seams`](../../../src/world/seam.rs).

**Reproduce:** `audit_reaction_store_does_not_glue_chart_neighbours` sets a material in the
real neighboring chart cell and reads the virtual `x+1` neighbor used by the reaction scheduler.
The two reads differ. One observed pair is virtual `(1148234496,15508833,120586368)` versus
real `(1100000000,15508833,24117248)`.

**Impact:** touching materials can stop interacting across chart edges. A reaction against
virtual empty storage can also place a product outside the actual chart instead of in the
visible neighboring cell. Address-based ordering/deduplication cannot identify an undirected
physical seam contact by itself.

**Repair:** make the contact topology an authority-owned operation that returns canonical real
cell pairs and deduplicates both orientations. Resolve writes as well as reads. Define an explicit
contact rule for nonconforming band interfaces; do not accidentally make reactions depend on
which patch woke the contact first.

**Acceptance:** a reactive pair on both sides of every conforming face seam behaves like an
interior pair, conserves occurrences, creates no ghost overlay cells, and agrees between the
singleplayer and server stores. Save/restore must preserve canonical contact identity.

## B04 — Tool uses bypass server edit-policy mods

**P1 · probe-confirmed.**

**Evidence:** [`on_edit`](../../../src/net/server/edit.rs) calls `ServerMod::validate_edit`;
`on_tool_use` does not receive the hook table and commits the cell directly. The hook's contract
in [`hooks.rs`](../../../src/net/hooks.rs) describes gating a would-be cell change.

**Reproduce:** `audit_tool_bypasses_edit_policy` installs a deny-all edit hook. An ordinary air
edit is denied, but a destructive tool use against the same cell and revision changes it.

**Impact:** a server protection, ownership or build-permission mod can stop breaking/placement
while allowing tools to alter the protected block and start reactions. This undermines the mod
seam users would naturally choose to implement protection.

**Repair:** validate a proposed tool operation after computing its products and before committing
it, outside the state lock, then recheck revisions. Consider a shared mutation-policy hook with
origin (`player edit`, `tool`, `machine`, `reaction`) and all affected cells so mods can express
their own propagation policies rather than hard-coding protection rules in core.

**Acceptance:** deny/allow tests cover tools, ordinary edits and future machine intents; a denied
tool preserves the held unit, cell, revision and reaction queue and resolves the pending request.

## B05 — Over-budget tool messages still force arbitrary generator queries

**P1 · probe-confirmed handler cost; full network denial-of-service not measured.**

**Evidence:** [`charge`](../../../src/net/server/budget.rs) returns `Answer` for excess tool traffic.
The [`client_loop`](../../../src/net/server/session.rs) dispatches each one to `refuse_tool`, which
generates its requested cell on a cache miss. The out-of-reach rejection in `on_tool_use` does
the same. These answer paths are outside the tool quota's expensive-work gate.

**Reproduce:** `audit_refused_tool_still_queries_arbitrary_terrain` executes the exact excess
dispatch with 100 distinct out-of-reach addresses against a counted generator: **100 rejected
messages still cause 100 generator calls**. Cycling addresses prevents useful cache reuse.

**Impact:** the documented per-client tool budget does not bound this generator work. Under a
costly custom/current generator it can consume CPU and contend on the shared state/cache path.
Putting generation outside the lock prevents one kind of contention but does not restore the
rate limit.

**Repair:** an explicit rejection should resolve a request without fetching inaccessible cell
content; alternatively only return cached content and put misses behind a separately bounded
reply/work budget. Admit/reach-check the target before any expensive rejection lookup. Make
the client's refusal path understand a result without authoritative cell content.

**Acceptance:** thousands of over-budget or out-of-reach requests produce bounded generator
calls, bounded replies and stable legitimate-client ping/tick latency. Keep human-rate valid
tools responsive.

## B06 — A full outbox silently loses a successful ToolResult

**P1 · probe-confirmed.**

**Evidence:** `on_tool_use` commits the cell and ignores failure of `h.out.try_send(ToolResult)`.
It broadcasts the edit to everyone **except the sender**. Ordinary `on_edit` already handles
loss of essential content by kicking the slow client so rejoin replays the world.

**Reproduce:** `audit_full_outbox_loses_tool_commit_without_kicking_sender` fills a one-slot
outbox, then uses a reactive tool. The cell changes, the only queued frame is the prior Pong,
there is no ToolResult and the sender is not marked kicked.

**Impact:** the sender's cell and held tool remain stale; expiry reports no reaction even though
the server changed the world. Other clients see the change. Occurrences transferred into the
held tool can disappear from the player experience.

**Repair:** successful tool results are essential state. Reserve reliable response capacity before
commit, or use the ordinary authoritative-content/kick policy when confirmation cannot be
queued. A world snapshot alone cannot repair the held product, so transaction identity/result
retention matters too.

**Acceptance:** saturate sender and peer outboxes around a successful operation; the sender
receives the committed cell/tool exactly once or reconnects through a result-recovery path.
Never leave a successful operation indistinguishable from a no-op.

## B07 — A tool result arriving after local expiry is discarded entirely

**P1 · probe-confirmed.**

**Evidence:** [`Session::expire`](../../../src/net/client.rs) removes tool requests after `PENDING_TTL`
and emits `reacted:false`. `Session::apply(ToolResult)` returns immediately if the request no
longer exists. The result lacks coordinates, so the client needs that request to locate the cell.
The successful edit broadcast excludes its sender.

**Reproduce:** `audit_late_tool_result_is_discarded_after_timeout` expires a tool request and
then feeds its successful revision-1 result. It emits no update and records no cell revision.
Unlike ordinary edits, there is no independent authoritative snapshot preceding the answer.

**Impact:** transient transport/server delay can leave the initiating client permanently out of
sync until another edit or a rejoin. The delayed held product is also lost. A live heartbeat
does not prove each transaction completed within its local TTL.

**Repair:** distinguish UI timeout from cancellation of an authoritative operation. Retain a
bounded result journal/request tombstone, carry target/transaction identity in results and
reconcile late content. Resolve inventory reservations only on an authoritative verdict or
recovery query, not by guessing that timeout means no reaction.

**Acceptance:** successful tools delayed past TTL converge both cell and holdings; duplicates
are idempotent; actual refusals and disconnect recovery release reservations correctly.

## B08 — Concurrent tool requests do not reserve the held unit

**P1 · probe-confirmed result handling; ordinary network latency makes overlap possible.**

**Evidence:** [`Game::use_tool`](../../../src/game/sync.rs) sends each request and remembers only a
`BlockId`. It does not reserve/consume the unit while pending. `finish_tool_change` changes the
world first, then only adds the returned product if consuming the old tool succeeds. The server
trusts the claimed held spec and has no holdings ledger.

**Reproduce:** `audit_two_tool_commits_spend_one_held_unit` starts with one unit, queues two
pending requests using it and applies successful results at two different cells. Both cells
become the reaction product, but only one held tool product survives. The second old-tool
consume fails after the world has already been changed.

**Impact:** fast clicking, or placing/spending a tool before its answer, loses transferred matter.
Two uses on one cell also send the old tool spec despite the first use changing it; adjusting
the expected cell revision does not fix the stale held input.

**Repair:** reserve an actual held unit per transaction. Serialize subsequent operations on that
unit until its product is known; expose separate available and reserved quantities. Preserve
the freedom for mods to present/equip inventory differently while the core owns transaction
bookkeeping. An authoritative holdings interface is needed if servers/mods require an economy.

**Acceptance:** overlapping tools, a placement while a tool is pending, delayed refusals and
disconnects preserve the total multiset across cells, available units and reservations.

## B09 — Loading with a mod absent erases its world state on the next save

**P1 · probe-confirmed.**

**Evidence:** [`Mods::load_state`](../../../src/modding/mod.rs) silently ignores unmatched ids.
[`Restored::finish`](../../../src/save/bridge.rs) consumes the saved records, and future snapshots
contain only current mods' `save_state` output. There is no retained unknown-state collection.
In contrast, the options registry deliberately preserves unclaimed settings lines.

**Reproduce:** save a machine/stateful mod, remove its package from the same instance, enter and
save the world, then reinstall it. `audit_missing_mod_state_is_not_retained` demonstrates the
ignore → empty-next-snapshot path directly.

**Impact:** changing a build can destroy machine state, history, counters or mod-owned holdings
even if the user intended only temporary removal. Native instance isolation separates instances;
it does not prevent changing packages within one instance.

**Repair:** retain opaque unclaimed records with the world, merge them into snapshots, and let
active mods claim/replace their own record. Version-migration refusal should also be capable of
retaining the original payload instead of overwriting it with defaults.

**Acceptance:** remove/re-add a stateful mod across multiple saves and recover its byte-identical
state. Explicit deletion is a separate operation. Duplicate-key handling must be defined (B18).

## B10 — Expanded inventories reload at default capacity and silently drop items

**P2 · probe-confirmed · affects capacity-changing mods.**

**Evidence:** [`Inventory::new`](../../../src/inventory.rs) accepts arbitrary capacity and
`Player.inventory` is public. [`SaveSnapshot`](../../../src/save/bridge.rs) saves contents but not
capacity; rebuild makes `Player::new`, whose inventory starts at 100. Restore applies `add`,
which drops excess units. The mod world-state load callback has no player argument with which
to restore capacity before this happens.

**Reproduce:** `audit_mod_extended_inventory_capacity_does_not_survive_save` saves 200 units in
capacity 250 and rebuilds: **capacity 100, total 100**. No unknown-spec problem is involved.

**Impact:** an otherwise straightforward inventory/backpack mod loses legitimate holdings on
save/load. The core state does not faithfully persist the customization its API permits.

**Repair:** persist capacity if it is world/player state, or establish capacity through a stable
inventory policy before loading contents. Preserve overflow pending policy resolution rather
than silently discarding it. Keep UI replacement independent of storage/persistence policy.

**Acceptance:** capacity-expanded inventories round-trip all units and their order; a later
capacity reduction has an explicit mod-defined overflow behavior.

## B11 — A failed first save after backup recovery can destroy the sole intact copy

**P1 · probe-confirmed store sequence.**

**Evidence:** [`store::read`](../../../src/save/store.rs) can load intact `.bak` over corrupt live
bytes. The singleplayer writer still calls `write_rotating(..., true)`. That rotates the corrupt
live over the intact backup before final rename. On final-rename failure it restores the corrupt
bytes and removes the new temporary snapshot. The dedicated-server `Store` separately tracks
whether backup rotation is safe; singleplayer does not.

**Reproduce:** `audit_failed_write_after_backup_recovery_loses_both_old_copies` creates corrupt
live plus good backup, then injects only a failure of the final temporary→live rename through
the existing test seam. Afterward, live contains corrupt bytes, backup is absent and temporary
is absent. Neither intact old nor new snapshot remains.

**Impact:** an ordinary recovery followed by an I/O failure turns a recoverable world into a
lost one. Even a crash between the two renames replaces the only discoverable intact backup
with corrupt content.

**Repair:** carry `LoadReport.source`/recovery state into `ActiveSlot` and `Autosaver`; preserve
the intact backup until a replacement live file is committed, as the server store already
attempts. Directory fsync should be considered separately for power-loss durability.

**Acceptance:** inject failures at write, file sync, each rename and backup promotion after both
normal load and backup load. At least one readable intact copy must survive every failed save.

## B12 — Backup-only slots are invisible to listing and fresh-name allocation

**P1 · source-confirmed; crash-window trigger.**

**Evidence:** the supported write sequence temporarily has live renamed to `.save.bak` and no
`.save` before promoting `.tmp`. [`store::list`](../../../src/save/store.rs) only considers files
whose extension is `save`; `unused_id` also checks only the live path. Yet `store::read(id)` is
capable of recovering such a backup when given the id.

**Reproduce:** create a valid slot, rename `world.save` to `world.save.bak` to model the documented
crash window, restart. The Worlds list does not show it, and `fresh_id()` can choose `world`.
Vanilla's automatic most-recent-readable selection uses the same incomplete list. A new world
can occupy that id; a later rotating save can replace the old world's backup.

**Expected:** recovery-capable slots remain discoverable, and fresh ids never collide with
recoverable artifacts.

**Repair:** list the union of valid live and backup slot stems, prefer the appropriate metadata
read ladder and identify a recovered slot. Include `.bak` in name reservations. Decide separately
whether a synced `.tmp` may be recovered; never silently reinterpret it as a new world.

**Acceptance:** crash-window fixtures appear in the list, can load, and cannot be reused by
`fresh_id`; unrelated corrupt files do not create endless bogus slots.

## B13 — Presentation caches key revisions without provider identity

**P2 · naming probe-confirmed; texture consequence source-confirmed.**

**Evidence:** [`BlockRegistry::refresh_names`](../../../src/block/registry.rs) keys on
`Option<namer.revision()>`. [`refresh_textures`](../../../src/world/textures.rs) keys on an
appearance's numeric revision and descriptor count. Neither key includes which provider won
the first-active-provider arbitration. Independent mods commonly start their revisions at 0
or 1; the API does not require globally unique revisions.

**Reproduce:** `audit_provider_switch_with_equal_revision_keeps_old_names` switches between
two namers both at revision 0. The block keeps the first provider's name. For appearance, build
layers with provider A, suspend it or have it stop answering, and let B/flat with the same
revision win: `built == count` permits an early return with A's cached pixels.

**Impact:** valid replacement mods can show stale names/textures, defeating the exact customization
seams the game offers. Suspensions normally occur at joins, but public `World::pump` also accepts
different providers; the cache contract should not depend on a coincidental new World.

**Repair:** key caches with a stable provider/selection generation plus its local revision. Have
the host supply the generation on provider changes; raw pointer identity is fragile and not a
save/network identity. Preserve the cheap branch-only unchanged-frame path.

**Acceptance:** A→B, A→fallback→A and A→B at equal revisions invalidate exactly once; ordinary
unchanged provider/revision frames perform no repaint/rename work.

## B14 — Lowering the GPU descriptor cap leaves already-interned layers out of range

**P2 · probe-confirmed CPU contract; real-device visual effect not exercised.**

**Evidence:** [`set_descriptor_cap`](../../../src/block/registry.rs) changes the ceiling for future
allocation but does not remap existing `render_layer`/`layer_source` entries. Worlds and loaded
save materials are interned before [`refresh_textures`](../../../src/world/textures.rs) reads the
device cap. `HotTables::render_layer` debug-asserts below-cap layers, then release-clamps to
`cap-1`. The intended overflow behavior is nearest appearance, not all excess layers aliasing
to the final uploaded texture.

**Reproduce:** `audit_lower_descriptor_cap_does_not_reassign_existing_layers` interns five
non-air materials, then sets cap 2. Material 5 keeps layer 5. The hot-table query panics in debug;
the release branch returns layer 1. A real device with a smaller array limit than a loaded
world's material count is the practical trigger.

**Repair:** establish the cap before render identity allocation, or rebuild all render mappings
when applying the first cap, keeping material identity unchanged. Include render-mapping/cap
changes in hot-table revision and mesh invalidation if existing snapshots can predate that cap.

**Acceptance:** many pre-interned configurations, cap 1/2 and representative device caps all
yield valid layers with the declared nearest-look fallback. No worker snapshot contains a
layer beyond its stamped ceiling, and existing meshes are refreshed when mapping changes.

## B15 — Overlay compaction leaves unbounded stale column indexes

**P2 · probe-confirmed.**

**Evidence:** [`World::place`](../../../src/world/edits.rs) removes the overlay cell/chunk when
restoring generated terrain, but does not remove its entries from `edit_columns` or
`edit_column_chunks`. The bulk ledger path has the same asymmetry. The indexes participate
in later column/ceiling and edited-height processing.

**Reproduce:** `audit_restored_natural_edits_leave_column_indexes_behind` places and restores air
in 100 distinct flat-world columns. The final edit overlay contains **zero** cells, while both
indexes retain **100** columns. Repeating along an arbitrarily long route keeps growing them.

**Impact:** long editing sessions retain history unrelated to current world differences and make
column scans/cache rebuilding more expensive. Saving/reloading can temporarily clear some
history but is not a runtime memory policy.

**Repair:** on the last edit leaving a chunk, remove its altitude from the appropriate face-column
index; recompute/remove the ordinary edited-column span when needed. Audit the analogous
`section_edit_chunks`/revision indexes before applying the same cleanup, since some revisions
must survive in-flight work.

**Acceptance:** repeated place→restore over new columns leaves storage proportional to current
edits, with unchanged roof shadows, bulk-install parity and section invalidation correctness.

## B16 — Ordinary LOD selections exceed the asserted section capacity contract

**P1 for debug playability; P2 for unproven release performance/coverage consequences · existing tests reproduce.**

**Evidence:** [`desired_sections_with`](../../../src/world/streaming/far.rs) asserts
`out.len() * 4 / 3 <= SECTION_SLOT_FLOOR`, currently 1280. Existing world tests panic at line 416.
The working tree contains pre-existing quadtree/probe edits, so origin is deliberately unassigned.

**Reproduce:** the `far_face_` group reports leaf counts **1077, 1103, 1810 and 1025** for twin
face, +X, cube-edge and chart-edge selections. The altitude-frontier test independently reports
1077. The group ends 5 passed/4 failed; the failures happen before its intended geometry assertions.
Additional section-lane tests fail on the same guard in the broader run.

**Impact:** valid selections terminate debug builds. Release removes the assertion, so a panic
alone does not establish a release hole; it does establish that resident/admission sizing relies
on a bound the actual rule violates. Simply raising the floor could conceal an unbounded cut
and undermine the performance goal.

**Repair:** derive a cut bound accounting for multiple joined faces, conservative height ranges
and resident ancestors; enforce it in selection/admission. Inspect leaf-count growth by face,
level and pending/tight bound state before changing constants. Compare committed HEAD against
the current working tree to attribute the regression.

**Acceptance:** the listed tests reach their actual assertions; seed/face/altitude/corner sweeps
stay within the proved resident budget or deliberately coarsen with full coverage. Measure
flight settle time, leaf/ancestor count, VRAM and frame-time tails in release.

## B17 — Duplicate peer poses collapse interpolation into a snap

**P2 · probe-confirmed; a related existing loopback test fails independently.**

**Evidence:** [`apply_pose`](../../../src/net/client.rs) always replaces `prev` with `target` and
replaces the interpolation interval with time since the previous application, even when the
new pose is identical. Multiple frames drained by one poll can be applied microseconds apart.
Visibility-entry poses and regular pose ticks are both valid sources of pose frames.

**Reproduce:** `audit_duplicate_pose_collapses_interpolation` establishes a 50 ms 0→1 rad turn.
At its midpoint the sample is 0.5 rad; feeding a duplicate target 1 µs after the new pose changes
that same sample to 1.0 rad. The existing
`net::mp::presence::a_peer_turning_round_is_drawn_turning_the_short_way` also fails its
“between poses, not snapped” assertion in the network suite and when rerun alone. Its exact
packet sequence has not been instrumented, so the two observations are not claimed to prove
the same root cause.

**Impact:** harmless duplicate/batched messages can make peers snap and incorrectly reset the
derived movement speed. Native high-frame-rate presentation makes these continuity defects visible.

**Repair:** ignore equivalent target snapshots without resetting interpolation; coalesce per-peer
updates within a poll or use a bounded history with meaningful receive/server-tick timestamps.
Preserve re-entry's deliberate snap and shortest-arc frame/yaw behavior.

**Acceptance:** duplicate targets and batched bursts preserve an ongoing interpolation; stationary
poses remain stationary; the existing loopback test passes consistently. Test timing algebra
with synthetic timestamps instead of relying solely on sleeps.

## B18 — Mod ids are global and unvalidated, so separate packages can overwrite saved state

**P2 · probe-confirmed.**

**Evidence:** [`install_from`](../../../src/modding/mod.rs) accepts any `Mod::id` without uniqueness
validation. Save state is keyed solely by that id; loading chooses the first matching entry.
`PackageInfo.id` namespaces package identity but does not namespace registered mods' save keys.
Several first-party ids are intentionally short local names.

**Reproduce:** `audit_duplicate_mod_ids_share_a_persistence_key` installs two mods from distinct
packages, both called `same`, saving different state. Reload applies both records to the first
mod; its original state is overwritten while the second is never restored.

**Impact:** independent authors can make individually valid packages that become incompatible
only when combined. The failure is silent and can destroy state. The ecosystem should not
depend on every author manually coordinating all internal names forever.

**Repair:** namespace durable mod identity by package plus local id, validate duplicate ids
within a package at registration, and migrate legacy save keys deliberately. Apply comparable
diagnostics to global action and screen-entry ids; reject ambiguity at build/registration rather
than on the hot path.

**Acceptance:** independently developed packages with the same local mod id round-trip separately;
ambiguous legacy records produce a useful migration diagnostic and remain preserved.

## Additional risk kept separate from confirmed bugs

`audit_stale_tool_result_is_emitted_after_newer_snapshot` feeds a revision-3 snapshot and then a
revision-1 ToolResult. The client emits old cell content even though its revision map stays at 3;
`Game::apply_incoming` writes that content without a freshness gate. This is a real handler
inconsistency, but server queueing under one lock and one reliable ordered stream may prevent
that particular order today. No wire-order reproduction was established. A future transport or
parallel server change should not inherit the trap: reconcile held products independently from
cell freshness, and reject stale cell content while still settling the transaction.

## Known issues carried forward rather than rediscovered

[open-issues.md](../../../documentation/open-issues.md) already records generated-overhang skylight,
server/product packaging, clean-checkout CI, shader provenance and local golden references.
The README already acknowledges unauthenticated server certificates, and the client explicitly
tracks the missing teleport request id as a TODO. These remain useful work, but are not counted
as new findings here. The direct native-mod trust model and client-reported package list are
documented limitations, not purported sandbox escapes.
