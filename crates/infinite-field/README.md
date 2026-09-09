# infinite-field

A small Rust implementation of the **InfiniteDiffusion** sampling pattern
(Goslin, SIGGRAPH 2026): turn a finite *denoiser* into an infinite,
seed-consistent, randomly-accessible field.

The algorithm (see `annotated_infinite_panorama.py` in
[terrain-diffusion](https://github.com/xandergos/terrain-diffusion)):

1. Sample a **deterministic tiled Gaussian** so any window of the infinite
   plane sees the same noise for a given seed.
2. Denoise overlapping tiles through several **phases**. Between phases,
   overlapping tiles are blended with a linear kernel (weighted average).
3. Cache finished tiles. A query of any region is O(tiles covering it).

This crate does **not** ship a neural UNet. You plug in a [`Score`] — a
pure function `(channel, x, z, phase, current) -> predicted_clean`. A
hash-structure score is included for tests and as a procedural stand-in;
swap it for a learned model later without changing the tiling.

Dense rectangle queries go through `fill_all`, which loads each covering
tile once and blends — bit-identical to per-cell `sample_all`.

## v2 integer field

The f32 API above stays. v2 is a parallel integer path used by later worldgen
and by a GPU compute mirror:

- **Values** are `i32` in Q16.16 (`ONE = 1 << 16` is 1.0). Hashes are `u32`.
  Every value-path op is wrapping 32-bit integer math; `i64` appears only as
  a multiply transient, truncated with an arithmetic shift
  (`((a as i64 * b as i64) >> 16) as i32`). `%` is Rust `SRem` (SPIR-V
  `OpSRem`). There are no floats in this path.
- **`IntSpec`** matches `Spec` (tile / stride / phases / channels) with a
  `u32` seed. **`IntScore::predict`** receives a **`Stencil`**:
  `st.prev(ch, dx, dz)` is the previous phase's *blended* value at
  `(x+dx, z+dz)` for `|dx|`, `|dz|` ≤ 2. Phase 0 reads the prior hash
  noise. A phase never sees its own writes (Jacobi), so tile order and
  query order cannot change values. Out-of-tile stencil reads go through
  the same tent-kernel blend of neighbouring previous-phase tiles.
- **Tile array** from `IntField::tile(tx, tz)`: length `tile * tile * channels`,
  row-major **z, then x, then channel** — index
  `(z * tile + x) * channels + c`. This is the exact array a GPU pass
  produces. `fill_all` writes `(z * w + x) * channels + c` over a rectangle
  and loads each covering tile once. Tent-kernel weights are integer
  products of a 1-D tent; Q16 weights sum to exactly `1 << 16` per cell
  (the last contributor takes the remainder).
- **LRU tile cache** per `IntField`, default 4096 tiles, true
  least-recently-used eviction (no clear-all). `cached_tiles()` and
  `cache_stats()` (hits, misses, evictions) are for tests and gauges.

`inoise` is the noise vocabulary (`hash32`, `value_noise_q16`, `fbm_q16`,
`ridged_q16`, `cellular_q16`, `warp_q16`, `gradient_q16`, …). Every formula
is a table in that module's docs so the SPIR-V mirror is a transliteration.
`IntHashScore` is a reference score (integer twin of `HashScore`) so the
crate is testable without a worldgen adapter.

### GPU mirror

One compute dispatch = one phase of one tile.

- **Push block:** `{seed: u32, tx: i32, tz: i32, phase: u32, phases: u32, tile: u32, stride: u32, channels: u32}`.
- **Input SSBO:** the 3×3 neighbouring *previous-phase* tiles (or the prior
  hash field when `phase == 0`), each laid out as the tile array above.
  The shader blends those with the tent kernel to form `current` and the
  5×5 stencil.
- **Output SSBO:** the tile array for `(tx, tz)` at this phase (after
  `IntScore::predict`).
- **Launch:** 1 invocation per `(x, z)` cell in the tile; channels run in
  a loop inside the invocation. Invocations do not read each other's
  writes.

## Properties (tested)

- Same `(seed, coordinates)` always regenerate the same samples.
- Query order does not change results (A then B equals B then A).
- Overlapping queries agree on the intersection.
- `fill_all` matches `sample_all` at every cell.

## License

MIT OR Apache-2.0.
