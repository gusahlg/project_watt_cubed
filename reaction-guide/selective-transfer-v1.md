# Watt Cubed: the first reaction function to implement

**Recommendation: selective transfer with an integer fit table, cached internal support, and exact handling of capacity.**

This should be the first playable material experiment. Its strongest advantage is already demonstrated behaviour: configurations can resist a broad population of materials while remaining vulnerable to a specific counterpart. The mechanism is small enough to inspect and measure. It also has explicit meanings for extraction, contamination, combining materials and emptying a voxel.

The included Rust implementation is the specification's executable reference. It has no external dependencies, allocates no heap memory in the reaction kernel, and uses no runtime trigonometry or floating-point decisions. It has been compiled and tested. The world scheduler, rendering and gameplay have not been implemented here.

## 1. What this law does

An element has four byte coordinates. Those coordinates identify the element and do not change during a reaction. A block contains a multiset of element occurrences. A reaction moves one occurrence between two neighbouring blocks, or exchanges two occurrences when capacity requires considering swaps.

Every element has a relationship with every other element, computed from the same universal formula. A proposed move compares the support the element receives inside its present block with the attraction offered by the other block. A sufficiently beneficial move happens. The changed grouping can make a different move beneficial next.

The entire configuration matters because every companion contributes to that comparison. No element is labelled intrinsically hard, corrosive or metallic. A resistant configuration can have strong internal support for all its constituents, yet one particular counterpart can fit one constituent well enough to extract it. That extraction changes the remaining support and can begin a cascade.

This is a model of regrouping existing elements. It does not manufacture new resource coordinates. A configuration can disappear because all its constituents leave its voxel; those constituents still exist elsewhere.

## 2. Fixed choices for version 1

| Choice | Definition |
|---|---|
| Element | `[u8; 4]` |
| Resource topology | Four periodic axes; 255 is adjacent to 0 |
| Configuration | Unordered multiset; multiplicity is preserved |
| Capacity | At most 32 occurrences per voxel |
| Empty configuration | Empty voxel |
| Typical initial materials | Start testing with 4–8 occurrences; 32 is a capacity limit, not a requirement |
| World contact | Two face-adjacent voxels for the first sandbox |
| One operation | One transfer or one atomic swap |
| Reaction threshold | Strictly greater than `1/32` |
| Element identity | Conserved |
| Numerical law | Committed integer lookup table, signed integer scores |
| Law identifier | `watt-selective-transfer-v1` |

The axes have no individual physical names. World position and resource coordinates are separate spaces. Array order inside a configuration is storage only; it must not change reaction outcomes.

An occurrence is a unit of constituent amount in this model. `[x,x,y]` differs from `[x,y]`. Excluding one reacting occurrence from its internal support does not exclude other occurrences with the same coordinates.

These are deliberate choices for this candidate, not claims that the project has permanently settled every material-system question.

## 3. The universal pair fit

The ideal mathematical curve is

\[
K(a,b)=\frac14\sum_{d=0}^{3}
\left[\cos\theta_d-\cos(2\theta_d)\right],
\qquad \theta_d=\frac{2\pi(a_d-b_d)}{256}.
\]

Positive fit favours grouping, negative fit favours separation. The two harmonics give the curve positive and negative regions without a recipe table. In one axis, equal coordinates contribute zero, a separation of 64 contributes +1, and a separation of 128 contributes −2. Four axes contribute to a single decision to move an entire element.

Nearby resource coordinates have similar pair relationships with a fixed third element. They do not necessarily attract each other strongly. In particular, identical elements have zero pair fit. Repeating one pure element therefore does not create positive internal cohesion between its identical occurrences. Mixed configurations are where this version's internal support comes from.

### Exact runtime definition

Use `Q = 2^20 = 1,048,576`. The supplied table defines 256 signed integers:

\[
q[r]=\operatorname{round}_{\text{half away from zero}}\left(
Q[\cos(2\pi r/256)-\cos(4\pi r/256)]\right).
\]

The file `src/fit_table.rs` is authoritative. Its values were generated once using 90-digit Decimal arithmetic. Do not generate them with platform trigonometry at game startup.

For runtime use, define

\[
J(a,b)=\sum_{d=0}^{3}q[(a_d-b_d)\bmod256].
\]

The version-1 pair fit is exactly `J / (4Q)`. The ideal cosine expression explains the design; the committed integers define the actual simulation.

```rust
pub fn fit_raw(a: [u8; 4], b: [u8; 4]) -> i32 {
    FIT[a[0].wrapping_sub(b[0]) as usize]
        + FIT[a[1].wrapping_sub(b[1]) as usize]
        + FIT[a[2].wrapping_sub(b[2]) as usize]
        + FIT[a[3].wrapping_sub(b[3]) as usize]
}
```

This costs four byte differences, four table reads and three additions. The table occupies 1 KiB. Wrapping subtraction is intentional; do not use ordinary `u8` subtraction in debug Rust.

The table gives exact pair symmetry, exact common-translation invariance of pair scores, and `J(e,e)=0`. The normalized pair fit differs from the ideal curve by at most `1/(2Q)`, approximately 0.000000477. This small score error does not guarantee identical completed reactions near a decision boundary. Version this law separately from the earlier floating-point prototype.

## 4. Support, attraction and transfer score

For an occurrence `e` in A, its raw internal support is

\[
h_A(e)=\sum_{a\in A\setminus\{e\}}J(e,a).
\]

Its raw attraction to B is

\[
p_B(e)=\sum_{b\in B}J(e,b).
\]

Its raw transfer gain is

\[
g_{A\to B}(e)=p_B(e)-h_A(e).
\]

Support is allowed to be negative. Negative support means the current companions collectively favour this element leaving. Do not clamp support, attraction or gains to zero.

With `N = len(A) + len(B)`, the normalized score is

\[
G_{A\to B}(e)=\frac{g_{A\to B}(e)}{4QN}.
\]

The acceptance condition is `G > 1/32`. Implement the exactly equivalent integer comparison:

```rust
let threshold = total_elements as i32 * (QUANTUM / 8);
let passes = raw_gain > threshold;
```

There is no runtime division. All candidates in one contact have the same denominator, so raw gains also rank them correctly. When both blocks are empty, there are no candidates and nothing divides by zero.

Use sums for the support and attraction. Do not average each block separately. Those would be different laws; earlier experiments showed that normalization choices materially change whether destructive cascades occur. The common denominator bounds the scale while preserving the effect of relative amounts.

An unchanged pair does not accumulate invisible damage. Repeating exactly the same below-threshold contact has no effect. Duration matters when several successful operations must occur in sequence, or when other interactions change the participants.

## 5. Exact procedure for one reaction

1. Read the current configurations of the two blocks in a deterministic A/B ordering.
2. Obtain each occurrence's internal support and compute its attraction to the opposite block.
3. Compute every outgoing raw gain in both directions from this same snapshot.
4. Every occurrence whose destination has fewer than 32 elements supplies a legal single-transfer candidate.
5. If either block contains 32 elements, consider every exchange of one occurrence from A with one from B. When both blocks are below capacity, swaps are not allowed in version 1.
6. Discard candidates whose raw gain does not strictly exceed `N * Q/8`.
7. Choose the highest-gain remaining candidate, using the exact tie rules below.
8. Commit that one transfer or swap atomically. Update both blocks' derived caches.
9. Return the operation. If no candidate passed, return unchanged.

Do not independently accept several favourable transfers from the original snapshot. The first transfer changes other gains. Re-evaluate the next choice from the updated state.

### Swap score

For `e` in A and `f` in B:

\[
g_{\text{swap}}(e,f)=g_{A\to B}(e)+g_{B\to A}(f)-2J(e,f).
\]

Both individual gains count attraction to an element that would itself leave the destination. Subtracting `2J(e,f)` corrects that. This is exactly the change in summed internal fit caused by the exchange.

Neither individual gain must be positive for the combined swap to succeed. Score the complete swap. It uses the same operation threshold as a single transfer; the rule compares the total improvement of either allowed operation.

### Tie rules

First prefer higher raw gain. Among candidates with exactly equal gain:

1. Prefer a single transfer over a swap.
2. Among transfers, prefer A→B over B→A.
3. Among transfers in the same direction, prefer the lexicographically smaller four-byte element coordinate.
4. Among swaps, prefer the lexicographically smaller `(element_from_A, element_from_B)` pair.

Exact duplicates are interchangeable. The storage index of the selected duplicate has no observable meaning.

This replaces the earlier rule that suppressed a block's proposal when distinct types tied. Suppression could trap even a two-element repulsive mixture beside empty space. The new rule allows it to separate.

The tradeoff is explicit: absolute coordinate order resolves exact indifference. Pair scores remain exactly translation-invariant, but a completed reaction involving a tie need not translate into the same labelled outcome after a common resource-space shift. Deterministic symmetry breaking is preferable here to silently freezing tied mixtures. It is a convention at equal scores, not an intrinsic material-strength ranking.

## 6. A fully worked destructive exception

These are discovered test configurations, not recipes embedded in the kernel:

```rust
let a = [
    [73, 145, 162, 161],
    [71,  77, 157, 208],
    [34, 125, 217, 144],
    [ 8,  85, 210, 206],
];

let e = [
    [83, 135, 211, 195],
    [51, 125, 144, 147],
    [11,  72, 167, 145],
    [25,  80, 211, 204],
];
```

For A's first four possible outgoing constituents, normalized support and attraction are:

| Constituent in A | Internal support | Attraction to E | Transfer score, difference divided by 8 |
|---|---:|---:|---:|
| `[73,145,162,161]` | 2.194644 | 2.334427 | 0.017473 |
| `[71,77,157,208]` | 2.055181 | 2.386907 | **0.041466** |
| `[34,125,217,144]` | 2.292689 | 2.038585 | −0.031763 |
| `[8,85,210,206]` | 2.233904 | 2.071368 | −0.020317 |

Only the second of these exceeds 0.03125. It is the winning operation across both directions. Once it moves, A loses one of its supporting companions and E acquires another attractive constituent. Recomputed scores allow the cascade:

| Operation | Constituent transferred A→E | A count | E count | Score |
|---:|---|---:|---:|---:|
| Initial | — | 4 | 4 | — |
| 1 | `[71,77,157,208]` | 3 | 5 | 0.041465729 |
| 2 | `[34,125,217,144]` | 2 | 6 | 0.219912171 |
| 3 | `[73,145,162,161]` | 1 | 7 | 0.304374009 |
| 4 | `[8,85,210,206]` | 0 | 8 | 0.538158894 |

A's voxel becomes empty. E now contains all eight original occurrences and is itself a changed configuration.

Translate every constituent of E by 64 along coordinate 0, 1 or 2 to obtain three different counterparts. Each has the same amount and internal fit as E. All three leave A unchanged. Thus the destructive exception depends on the relative configurations, not just greater amount or a larger standalone cohesion score.

The integer implementation also leaves A unchanged against 508 of the 512 previously held-out cohesive configurations. A retains all its original constituents against every member of that panel. E still destroys all 32 one-byte neighbours of A, and all 32 one-byte neighbours of E still destroy A.

These are regression checks against a previously selected example and synthetic population. They do not establish how common good materials are in a future generated world. Earlier tests against raw random mixtures found much more absorption into A. Resistance to extraction and resistance to contamination are different observations.

## 7. Exact caching and updates

There are two useful levels of derived state.

**Per configuration:** cache the raw internal support `h[i]` beside each element occurrence. Creating these values from raw coordinates costs one traversal of the unordered internal pairs. Identical configurations can share the same immutable record. These values are derived, not authored properties.

**Per active pair:** retain both blocks' elements, an owner bit per occurrence, their internal supports, and the outgoing gains. Constructing this contact requires one traversal of the cross-block pairs: start each gain at `-h[i]`, then add `J(a,b)` to both endpoints for each cross pair.

The provided types are `Block` and `Contact`. They own bounded arrays. Coordinates remain fixed inside a contact; owner bits change.

### Incremental transfer update

Move occurrence `p` from A to B. For every other occurrence `i`:

- If `i` was in A, subtract `J(i,p)` from its holding support and add `2J(i,p)` to its outgoing gain.
- If `i` was in B, add `J(i,p)` to its holding support and subtract `2J(i,p)` from its outgoing gain.

For `p` itself, using its old values:

\[
h'_p=h_p+g_p,\qquad g'_p=-g_p.
\]

The first expression gives the support it receives in its new block. Update its owner and the two counts. This is exact integer algebra and costs one scan across the contact.

### Incremental swap update

Swap `p` from A with `q` from B. For another occurrence `i` in A:

\[
\Delta h_i=-J(i,p)+J(i,q),\qquad \Delta g_i=-2\Delta h_i.
\]

For an occurrence in B, reverse the sign of `Δh`. For the two swapped occurrences:

\[
h'_p=h_p+g_p-J(p,q),\quad g'_p=-g_p+2J(p,q),
\]

and the same equations with `p` and `q` exchanged. Their owners swap; counts stay fixed. Use old values on the right-hand sides. Commit both changes together so no temporary over-capacity block becomes visible.

### Exact pruning of swap candidates

The integer fit satisfies `J >= -8Q`. Therefore:

\[
g_{\text{swap}}(e,f)\le g_e+g_f+16Q.
\]

Seed the current best result with the exchange of the two highest outgoing-gain occurrences. Then skip any candidate whose upper bound is strictly below the current winning gain. An entire row can be skipped using the best outgoing gain on the other side. Candidates capable of tying must remain eligible for tie-breaking.

This is an exact optimization. It does not restrict the search to one proposed swap. The tests compare it against unpruned enumeration and independently calculated potential changes.

### Cache lifetime

If a third block, player action or machine changes either participant, the old contact scores are stale. Rebuild from the current blocks. The safe `react_once` API does this automatically. Retain a `Contact` across calls only while the two configurations are still exactly its own state.

Do not allocate a permanent cache for every potential world edge. Initially, use configuration caches and short-lived contact scratch state. Add persistent active-contact caches only if profiling shows that their reuse pays for memory and invalidation.

## 8. Rust API and first integration

The main entry point is:

```rust
pub fn react_once(a: &mut Block, b: &mut Block) -> Option<Operation>;
```

It attempts one operation, updates both blocks on success, and leaves them unchanged on `None`. `Operation` identifies the transferred or swapped elements and the exact raw gain. A floating-point normalized score is available only for diagnostics.

```rust
use watt_reaction::{react_once, Block};

let mut a = Block::new(&[[0, 0, 0, 0]]).unwrap();
let mut b = Block::new(&[[64, 64, 64, 64]]).unwrap();

if let Some(operation) = react_once(&mut a, &mut b) {
    // Write BOTH cells as one authoritative world mutation.
    // Wake contacts involving either changed cell.
    println!("{operation:?}");
}
```

`Block::new` validates capacity and builds the internal-support cache. Keep the resulting block or interned configuration record. Reconstructing it from coordinates before every interaction would discard that optimization.

For a controlled two-block experiment:

```rust
let mut contact = Contact::new(&a, &b);
while let Some(operation) = contact.step() {
    // Record or display the individual step.
}
let [final_a, final_b] = contact.blocks();
```

That complete-contact loop is useful in experiments. In the world, let other scheduled contacts interleave; completing one edge to equilibrium before all others is a different scheduling policy and can produce different results.

Canonicalize coordinates when creating a configuration identity or save representation. Preserve duplicates. The supplied `canonicalize` method keeps derived supports aligned with their elements. Reactions themselves do not require sorted arrays.

## 9. World scheduling and resource accounting

Start with a single deterministic queue of eligible face contacts. Canonically order the endpoints using their world coordinates. Process one material operation each time a contact gets a turn.

A successful operation changes both participants. Increment both material revisions and wake or requeue their relevant neighbouring contacts, including the original contact. Deduplicate queued edges. An unchanged contact sleeps until one participant or its contact eligibility changes.

Meaningful placement, movement or material-manipulation events wake contacts. Chunk loading and mesh rebuilding should not independently create reactions. If a world is saved with pending material work, preserve enough queue and scheduler state to resume it; loading must restore that work without creating new fictional events.

The queue ordering and logical work quota must be deterministic. A rendering frame may defer execution of pending work, but must not change thresholds, discard contacts, or silently reorder them. If a fixed per-tick attempt quota is used, treat it as part of the simulation configuration. A no-op attempt still costs CPU and should count against the work quota.

Do not commit two reactions sharing a voxel from stale snapshots. A later parallel scheduler can process disjoint edges together. Six face-edge phases—X/Y/Z axis, each split by lower-coordinate parity—provide one way to obtain disjoint batches, but adopting that schedule can change multi-contact outcomes. Scheduling is part of the world's dynamics, not a consequence-free implementation detail.

Keep material identity independent of GPU render identity. A changed multiset may need a new simulation identity without requiring a new texture layer. Coalesce rendering and network notifications after authoritative logical work; retain correct simulation revisions between operations.

### Capacity has a visible consequence

Two full 32-element blocks contain 64 conserved occurrences. Neither can become empty while only those two blocks participate. They can exchange composition, but emptying one requires available space elsewhere. A counter-material used for extraction fills up and changes composition; it is not automatically a reusable catalyst.

This can support collection vessels, waste handling, replenishment and multi-stage machines. It can also become an annoying inventory tax. It is one of the first things to test with players.

The material law also does not move whole blocks through physical space. A single element beside air has zero gain and stays put under this law. Gravity, transport, tools and ordinary block movement remain separate world operations.

## 10. Bounds, determinism and stopping

For this table, `-8Q <= J <= (9/2)Q`. A block has at most 32 elements and a contact at most 64. Internal support magnitudes are at most `31*8Q`; outgoing gain magnitudes at most `63*8Q`. Even the conservative swap upper-bound expression fits within `2^30`, so signed 32-bit arithmetic has headroom. Keep these bounds synchronized if capacity or Q changes.

Use `i64` when summing total internal fit across pairs or the world. It can exceed `i32` even for two full blocks. This potential is a diagnostic and proof tool; the runtime does not need to sum it before each reaction.

Define

\[
\Phi=\sum_{\text{blocks }C}\sum_{i<j\in C}J(e_i,e_j).
\]

Every accepted operation increases `Φ` by exactly `raw_gain`, and that gain strictly exceeds `NQ/8`. Fixed matter and bounded block capacity give a bounded potential. Passive reactions in a closed, unforced system therefore stop. The two-block system has a conservative bound of at most 1,550 accepted operations under these constants; ordinary observed contacts take much less.

This is not a promise that every material is globally optimal or maximally separated. Below capacity, swaps are deliberately unavailable. Local decisions, thresholds and processing order create resting states.

It also rules out indefinitely repeating passive reaction cycles in a closed system. A machine that forces mixing or separation can restart reactions. That external operation is outside this passive law and must have an explicit gameplay meaning.

All runtime decisions use the same committed integers, explicit byte wrapping and total tie order. Array permutation does not affect outcomes. This specifies platform-independent integer decisions for identical ordered inputs; compiler/platform testing in this work was performed on one x86-64 Linux environment. World-level replay additionally requires identical scheduling and input events.

Smooth scores do not make final discrete reactions continuous. A one-byte change near a threshold or winning-candidate boundary can change whether a cascade starts. The goal is broad, learnable profiles with exceptions, not a claim that every neighbouring resource must have the same complete outcome.

## 11. Performance and verification

Actual measurements and their scope appear below. The integer-table implementation intentionally gives up the earlier prototype's exact real-valued Fourier factorization. Quantizing the difference table does not preserve that 16-feature identity exactly. Substituting the old floating evaluator would require a separate equivalence policy or a new numerical law version.

The costs of this implementation are:

| Work | Complexity |
|---|---|
| Build one configuration's support cache | `O(n²)` once for a new configuration |
| Build scores for a fresh A/B contact | `O(nA*nB)` from cached supports |
| Select among transfers | `O(nA+nB)` |
| Apply a transfer or swap to cached scores | `O(nA+nB)` |
| Exact capacity-swap search | Worst case `O(nA*nB)`; exact bounds often skip most candidates |
| Complete cascade | Sum of all its operations and checks |

The central performance rule is to let inactive contacts sleep. Avoid spending work on the same unchanged pair every frame. At high activity and full capacity, budget processing instead of completing every contact at once.

The tests cover 320 varied random pairs and their reversed input order, checking direct support/gain recomputation and exact potential changes throughout settling. They include empty, unequal and full-capacity inputs, conservation, capacity, cached-support preservation, strict threshold behaviour, the tie-stall regression, the 512-material resistance panel, the destructive example and its one-byte neighbours. Full-capacity choices are checked against unpruned enumeration; additional cases compare every swap to an independently recomputed potential difference.

### Recorded native timings

Measured with Rust 1.99.0, release optimization and overflow checks enabled, one thread on a shared AMD EPYC 9V74 Linux server. Five batches of at least 60 ms per measurement, using 128 uniformly random input pairs at each size. These are medians, not worst-case bounds and not measurements of your machine.

Values are **milliseconds for 1,000 calls**, scaled from measured time per call.

| Elements initially in each block | Copy cached contact + one step | Fresh contact + one step, from cached blocks | Raw configuration setup + complete cascade |
|---:|---:|---:|---:|
| 1 | 0.055 ms | 0.157 ms | 0.146 ms |
| 4 | 0.084 ms | 0.251 ms | 0.322 ms |
| 8 | 0.121 ms | 0.407 ms | 0.789 ms |
| 16 | 0.180 ms | 0.913 ms | 5.771 ms |
| 32 | 0.955 ms | 3.582 ms | 26.260 ms |

The middle column includes copying the two configuration records, rebuilding contact scores and writing updated support caches back. The first column excludes that world-record writeback and assumes the cached pair is still valid. The last includes building internal support from raw coordinates and repeatedly applying the complete law until quiescence. Mean accepted operations in the full-cascade benchmark ranged from about 0.46 at size 1 to 12.97 at size 32. Different workloads and setup costs mean the columns are not interchangeable.

On this build, a `Block` occupies 264 bytes and a `Contact` 856 bytes, in addition to the shared 1 KiB fit table. These are cache/storage implementation sizes, not save-format requirements. Ten thousand persistent Contact values would consume about 8.56 MB before scheduler overhead; allocate only where reuse is useful.

The practical conclusion is that thousands of small contact attempts are plausible for the numerical kernel. Thousands of complete dense cascades are substantial work. Configuration interning, world access, queuing, graphics, networking and synchronization are excluded and must be measured in the game.

No world-scale benchmark, networking test or human playtest has been performed.

## 12. What to refine after the first playable experiment

Start with the shipped curve, threshold, capacity and tie rule. Change one decision at a time and record a new law version. Do not add per-material exceptions to repair individual examples.

The first meaningful game experiment is a small reaction workbench with visible constituent counts, one-step execution, repeatable contact ordering and saved configurations. Give the player the demonstrated A, its three harmless counterparts and E. Show how E changes as it works. Then let the player attempt a selective extraction from an unknown mixture using placement, quantities and contact order.

Judge the result with concrete questions: Can the player predict related materials? Can they reproduce a process? Does a small impurity create a useful or legible difference? Can output and waste be separated? Does a used reagent change in an understandable way? Is the required handling enjoyable?

Record retained original constituents, acquired constituents, unchanged-contact frequency, complete emptyings, product purity, operations per useful result, active-edge counts, distinct configuration counts and whole-engine CPU time. A single scalar cohesion or hardness rating loses the counter-material exceptions this model is meant to preserve.

If almost nothing reacts, first inspect the generated material population and the threshold. Lowering the threshold increases activity but can also increase contamination and cascades. If everything collects into full voxels, the current grouping objective may be a poor fit for the desired game; do not assume more optimization will solve that behavioural problem. If useful processes exist but capacity feels arbitrary, revise the capacity rule with fresh tests. If transmutation or sustained autonomous activity is essential, this passive transfer law does not supply it.

The reason to implement this candidate first is specific: it already produces configuration-dependent resistance and selective breakdown, with explicit conservation and manageable costs. The next uncertainty is whether those behaviours become understandable, rewarding player actions.
