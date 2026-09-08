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

## Properties (tested)

- Same `(seed, coordinates)` always regenerate the same samples.
- Query order does not change results (A then B equals B then A).
- Overlapping queries agree on the intersection.
- `fill_all` matches `sample_all` at every cell.

## License

MIT OR Apache-2.0.
