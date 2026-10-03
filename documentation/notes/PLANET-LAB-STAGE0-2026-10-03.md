# Planet lab — stage 0 report (2026-10-03)

Guide §17 stage 0 asks for the geometric limit of curved voxel worlds to be measured, ugly cases included,
before anything is built on it, and for the tolerated distortion to be chosen and documented. This is that
record. Reproduce with `cargo run --release --bin planet_lab` (the full output is below the decision).

## Findings

1. **The radius cancels** (guide §3.1). Every metric is identical at R = 1e5, 2e6 and 8e6: a bigger world
   does not make cube-sphere cells more cubic.
2. **Gnomonic** (equal steps on the cube face): cells 1.27 wide at face centres and 0.64 at edge middles,
   30° skew at corners; 98 % of surface cells exceed the 1.05 edge-ratio threshold, 77 % exceed 3° skew.
3. **Equiangular**: perfect unit cells at face centres and along both face axes; one tangential direction
   compresses to 0.707 toward edge middles; skew grows toward the face diagonals and reaches 27–30° in the
   corner regions. 83 % of surface cells exceed 1.05, 70 % exceed 3°.
4. **Spherified cube**: the least skew (43 % over 3°) but no cell is square — centre cells are 0.90 wide and
   edge-middle cells are stretched 1.27 × 0.74.
5. **Seams conform but kink**: neighbouring charts share every corner along a chart edge (cells meet face
   to face), but grid lines bend across the edge: 0° at edge middles, ~31° a quarter of the way, ~58° near
   the cube corners (equiangular).
6. **Depth**: with one-block radial cells, tangential width scales with `r/R`: 0.95 at 5 % depth, 0.75 at
   25 %, 0.5 at the bottom of the first band (`R/2`). Halving the angular resolution there restores 1.0.
7. **Core transition**: a Cartesian core of half-size `r_in/2` joined by six mapped blocks is valid but
   exceptional: corner cells of the transition have volume 0.16–0.21 and 30–45° skew.

## Decision (the tolerated distortion)

- **Map: equiangular.** It is the only map with exactly square cells over a usable region (the face axes and
  the central part of every face), its inverse is closed form, and its distortion is symmetric.
- **Ordinary building region:** the central part of each chart, `|ξ|, |η| ≤ 0.5`, within the top 5 % of
  the radius: edge ratio ≤ 1.13, skew ≤ 8°. This is about a quarter of the surface; it is where generated
  worlds place their spawn points and showcase content.
- **Declared exceptional regions** (visible, documented, still valid geometry): the corner regions (skew up
  to 30°, valence-3 vertices at the eight cube corners), the band interfaces (1 : 4 cell relations every
  halving of the radius), and the core transition shell.
- **Depth bands:** halve the angular resolution whenever the radius halves; radial cells stay one block.
- This keeps "nearly cubic ordinary regions with explicit exceptional topology" (guide §19) and does not
  pretend a uniform cubic lattice exists on a sphere.

---

# Planet lab — cube-sphere charts

Metrics per cell (guide §9.2): edge ratio (longest/shortest of 12 edges), skew (largest corner-angle
deviation from 90°), volume (blocks³), singular values of the centre Jacobian. Surface cells are sized for
one block of arc on average (`n = π/2 · R` cells per face edge).

## R = 1e5 (n = 157080)

### Gnomonic

| where | edge ratio | skew ° | volume | σ1 / σ2 / σ3 |
|---|---|---|---|---|
| surface, centre | 1.2732 | 0.000 | 1.6211 | 1.0000 / 1.2732 / 1.2732 |
| surface, half to edge | 1.1388 | 0.000 | 1.1600 | 1.0000 / 1.0186 / 1.1388 |
| surface, edge middle | 1.5708 | 0.001 | 0.5732 | 0.6366 / 0.9003 / 1.0000 |
| surface, near corner | 1.5836 | 27.976 | 0.3522 | 0.4601 / 0.7654 / 1.0000 |
| surface, corner | 1.6661 | 30.000 | 0.3120 | 0.4244 / 0.7351 / 1.0000 |
| 5 % deep, centre | 1.2096 | 0.000 | 1.4631 | 1.0000 / 1.2096 / 1.2096 |
| 5 % deep, half to edge | 1.1180 | 0.000 | 1.0469 | 0.9677 / 1.0000 / 1.0819 |
| 5 % deep, edge middle | 1.6535 | 0.001 | 0.5173 | 0.6048 / 0.8553 / 1.0000 |
| 5 % deep, near corner | 1.6669 | 27.976 | 0.3178 | 0.4371 / 0.7271 / 1.0000 |
| 5 % deep, corner | 1.7538 | 30.000 | 0.2816 | 0.4032 / 0.6983 / 1.0000 |
| 25 % deep, centre | 1.0472 | 0.000 | 0.9119 | 0.9549 / 0.9549 / 1.0000 |
| 25 % deep, half to edge | 1.3090 | 0.000 | 0.6525 | 0.7639 / 0.8541 / 1.0000 |
| 25 % deep, edge middle | 2.0944 | 0.001 | 0.3224 | 0.4775 / 0.6752 / 1.0000 |
| 25 % deep, near corner | 2.1114 | 27.976 | 0.1981 | 0.3451 / 0.5741 / 1.0000 |
| 25 % deep, corner | 2.2215 | 30.000 | 0.1755 | 0.3183 / 0.5513 / 1.0000 |
| half depth, centre | 1.5677 | 0.000 | 0.4069 | 0.6379 / 0.6379 / 1.0000 |
| half depth, half to edge | 1.9596 | 0.000 | 0.2912 | 0.5103 / 0.5705 / 1.0000 |
| half depth, edge middle | 3.1354 | 0.001 | 0.1439 | 0.3189 / 0.4511 / 1.0000 |
| half depth, near corner | 3.1608 | 27.976 | 0.0884 | 0.2305 / 0.3835 / 1.0000 |
| half depth, corner | 3.3256 | 30.000 | 0.0783 | 0.2126 / 0.3683 / 1.0000 |

Surface cells over 1.05 edge ratio: **97.9 %**; over 3° skew: **76.9 %**.
Seam kink (grid-line bend across a chart edge): edge middle 0.0004°, quarter 36.870°, near corner 58.641°.

### Equiangular

| where | edge ratio | skew ° | volume | σ1 / σ2 / σ3 |
|---|---|---|---|---|
| surface, centre | 1.0000 | 0.000 | 1.0000 | 1.0000 / 1.0000 / 1.0000 |
| surface, half to edge | 1.0824 | 0.000 | 0.9239 | 0.9239 / 1.0000 / 1.0000 |
| surface, edge middle | 1.4142 | 0.000 | 0.7071 | 0.7071 / 1.0000 / 1.0000 |
| surface, near corner | 1.0747 | 26.932 | 0.7720 | 0.6883 / 1.0000 / 1.1216 |
| surface, corner | 1.0607 | 30.000 | 0.7698 | 0.6667 / 1.0000 / 1.1547 |
| 5 % deep, centre | 1.0526 | 0.000 | 0.9025 | 0.9500 / 0.9500 / 1.0000 |
| 5 % deep, half to edge | 1.1394 | 0.000 | 0.8338 | 0.8777 / 0.9500 / 1.0000 |
| 5 % deep, edge middle | 1.4887 | 0.000 | 0.6382 | 0.6717 / 0.9500 / 1.0000 |
| 5 % deep, near corner | 1.1312 | 26.932 | 0.6967 | 0.6538 / 1.0000 / 1.0656 |
| 5 % deep, corner | 1.1165 | 30.000 | 0.6947 | 0.6333 / 1.0000 / 1.0970 |
| 25 % deep, centre | 1.3334 | 0.000 | 0.5625 | 0.7500 / 0.7500 / 1.0000 |
| 25 % deep, half to edge | 1.4432 | 0.000 | 0.5197 | 0.6929 / 0.7500 / 1.0000 |
| 25 % deep, edge middle | 1.8856 | 0.000 | 0.3977 | 0.5303 / 0.7500 / 1.0000 |
| 25 % deep, near corner | 1.4329 | 26.932 | 0.4342 | 0.5162 / 0.8412 / 1.0000 |
| 25 % deep, corner | 1.4142 | 30.000 | 0.4330 | 0.5000 / 0.8660 / 1.0000 |
| half depth, centre | 1.9961 | 0.000 | 0.2510 | 0.5010 / 0.5010 / 1.0000 |
| half depth, half to edge | 2.1605 | 0.000 | 0.2319 | 0.4629 / 0.5010 / 1.0000 |
| half depth, edge middle | 2.8228 | 0.000 | 0.1775 | 0.3543 / 0.5010 / 1.0000 |
| half depth, near corner | 2.1451 | 26.932 | 0.1938 | 0.3448 / 0.5619 / 1.0000 |
| half depth, corner | 2.1171 | 30.000 | 0.1932 | 0.3340 / 0.5785 / 1.0000 |

Surface cells over 1.05 edge ratio: **83.2 %**; over 3° skew: **69.7 %**.
Seam kink (grid-line bend across a chart edge): edge middle 0.0006°, quarter 31.400°, near corner 57.900°.

### Spherified

| where | edge ratio | skew ° | volume | σ1 / σ2 / σ3 |
|---|---|---|---|---|
| surface, centre | 1.1107 | 0.000 | 0.8106 | 0.9003 / 0.9003 / 1.0000 |
| surface, half to edge | 1.1601 | 0.000 | 0.8296 | 0.8620 / 0.9625 / 1.0000 |
| surface, edge middle | 1.7321 | 0.000 | 0.9359 | 0.7351 / 1.0000 / 1.2732 |
| surface, near corner | 1.0963 | 22.388 | 0.7693 | 0.7177 / 1.0000 / 1.0719 |
| surface, corner | 1.1107 | 30.000 | 0.7020 | 0.6366 / 1.0000 / 1.1026 |
| 5 % deep, centre | 1.1692 | 0.000 | 0.7315 | 0.8553 / 0.8553 / 1.0000 |
| 5 % deep, half to edge | 1.2212 | 0.000 | 0.7487 | 0.8189 / 0.9143 / 1.0000 |
| 5 % deep, edge middle | 1.7321 | 0.000 | 0.8447 | 0.6983 / 1.0000 / 1.2096 |
| 5 % deep, near corner | 1.1540 | 22.388 | 0.6943 | 0.6819 / 1.0000 / 1.0183 |
| 5 % deep, corner | 1.1692 | 30.000 | 0.6335 | 0.6048 / 1.0000 / 1.0475 |
| 25 % deep, centre | 1.4810 | 0.000 | 0.4559 | 0.6752 / 0.6752 / 1.0000 |
| 25 % deep, half to edge | 1.5468 | 0.000 | 0.4667 | 0.6465 / 0.7219 / 1.0000 |
| 25 % deep, edge middle | 1.8138 | 0.000 | 0.5265 | 0.5513 / 0.9549 / 1.0000 |
| 25 % deep, near corner | 1.4617 | 22.388 | 0.4328 | 0.5383 / 0.8039 / 1.0000 |
| 25 % deep, corner | 1.4810 | 30.000 | 0.3949 | 0.4775 / 0.8270 / 1.0000 |
| half depth, centre | 2.2171 | 0.000 | 0.2034 | 0.4511 / 0.4511 / 1.0000 |
| half depth, half to edge | 2.3156 | 0.000 | 0.2082 | 0.4319 / 0.4822 / 1.0000 |
| half depth, edge middle | 2.7153 | 0.000 | 0.2349 | 0.3683 / 0.6379 / 1.0000 |
| half depth, near corner | 2.1882 | 22.388 | 0.1931 | 0.3596 / 0.5370 / 1.0000 |
| half depth, corner | 2.2171 | 30.000 | 0.1762 | 0.3189 / 0.5524 / 1.0000 |

Surface cells over 1.05 edge ratio: **100.0 %**; over 3° skew: **43.2 %**.
Seam kink (grid-line bend across a chart edge): edge middle 0.0007°, quarter 18.734°, near corner 54.403°.

## R = 2e6 (n = 3141593)

### Gnomonic

| where | edge ratio | skew ° | volume | σ1 / σ2 / σ3 |
|---|---|---|---|---|
| surface, centre | 1.2732 | 0.000 | 1.6211 | 1.0000 / 1.2732 / 1.2732 |
| surface, half to edge | 1.1388 | 0.000 | 1.1600 | 1.0000 / 1.0186 / 1.1388 |
| surface, edge middle | 1.5708 | 0.000 | 0.5732 | 0.6366 / 0.9003 / 1.0000 |
| surface, near corner | 1.5836 | 27.976 | 0.3522 | 0.4601 / 0.7654 / 1.0000 |
| surface, corner | 1.6661 | 30.000 | 0.3120 | 0.4244 / 0.7351 / 1.0000 |
| 5 % deep, centre | 1.2096 | 0.000 | 1.4631 | 1.0000 / 1.2096 / 1.2096 |
| 5 % deep, half to edge | 1.1180 | 0.000 | 1.0469 | 0.9677 / 1.0000 / 1.0819 |
| 5 % deep, edge middle | 1.6535 | 0.000 | 0.5173 | 0.6048 / 0.8553 / 1.0000 |
| 5 % deep, near corner | 1.6669 | 27.976 | 0.3178 | 0.4371 / 0.7271 / 1.0000 |
| 5 % deep, corner | 1.7538 | 30.000 | 0.2816 | 0.4032 / 0.6983 / 1.0000 |
| 25 % deep, centre | 1.0472 | 0.000 | 0.9119 | 0.9549 / 0.9549 / 1.0000 |
| 25 % deep, half to edge | 1.3090 | 0.000 | 0.6525 | 0.7639 / 0.8541 / 1.0000 |
| 25 % deep, edge middle | 2.0944 | 0.000 | 0.3224 | 0.4775 / 0.6752 / 1.0000 |
| 25 % deep, near corner | 2.1114 | 27.976 | 0.1981 | 0.3451 / 0.5741 / 1.0000 |
| 25 % deep, corner | 2.2214 | 30.000 | 0.1755 | 0.3183 / 0.5513 / 1.0000 |
| half depth, centre | 1.5677 | 0.000 | 0.4069 | 0.6379 / 0.6379 / 1.0000 |
| half depth, half to edge | 1.9596 | 0.000 | 0.2912 | 0.5103 / 0.5705 / 1.0000 |
| half depth, edge middle | 3.1353 | 0.000 | 0.1439 | 0.3189 / 0.4511 / 1.0000 |
| half depth, near corner | 3.1608 | 27.976 | 0.0884 | 0.2305 / 0.3835 / 1.0000 |
| half depth, corner | 3.3255 | 30.000 | 0.0783 | 0.2126 / 0.3683 / 1.0000 |

Surface cells over 1.05 edge ratio: **97.9 %**; over 3° skew: **76.9 %**.
Seam kink (grid-line bend across a chart edge): edge middle 0.0000°, quarter 36.870°, near corner 58.641°.

### Equiangular

| where | edge ratio | skew ° | volume | σ1 / σ2 / σ3 |
|---|---|---|---|---|
| surface, centre | 1.0000 | 0.000 | 1.0000 | 1.0000 / 1.0000 / 1.0000 |
| surface, half to edge | 1.0824 | 0.000 | 0.9239 | 0.9239 / 1.0000 / 1.0000 |
| surface, edge middle | 1.4142 | 0.000 | 0.7071 | 0.7071 / 1.0000 / 1.0000 |
| surface, near corner | 1.0746 | 26.933 | 0.7720 | 0.6883 / 1.0000 / 1.1217 |
| surface, corner | 1.0607 | 30.000 | 0.7698 | 0.6667 / 1.0000 / 1.1547 |
| 5 % deep, centre | 1.0526 | 0.000 | 0.9025 | 0.9500 / 0.9500 / 1.0000 |
| 5 % deep, half to edge | 1.1394 | 0.000 | 0.8338 | 0.8777 / 0.9500 / 1.0000 |
| 5 % deep, edge middle | 1.4886 | 0.000 | 0.6382 | 0.6718 / 0.9500 / 1.0000 |
| 5 % deep, near corner | 1.1312 | 26.933 | 0.6967 | 0.6538 / 1.0000 / 1.0656 |
| 5 % deep, corner | 1.1165 | 30.000 | 0.6947 | 0.6333 / 1.0000 / 1.0970 |
| 25 % deep, centre | 1.3333 | 0.000 | 0.5625 | 0.7500 / 0.7500 / 1.0000 |
| 25 % deep, half to edge | 1.4432 | 0.000 | 0.5197 | 0.6929 / 0.7500 / 1.0000 |
| 25 % deep, edge middle | 1.8856 | 0.000 | 0.3977 | 0.5303 / 0.7500 / 1.0000 |
| 25 % deep, near corner | 1.4329 | 26.933 | 0.4342 | 0.5162 / 0.8412 / 1.0000 |
| 25 % deep, corner | 1.4142 | 30.000 | 0.4330 | 0.5000 / 0.8660 / 1.0000 |
| half depth, centre | 1.9960 | 0.000 | 0.2510 | 0.5010 / 0.5010 / 1.0000 |
| half depth, half to edge | 2.1605 | 0.000 | 0.2319 | 0.4629 / 0.5010 / 1.0000 |
| half depth, edge middle | 2.8228 | 0.000 | 0.1775 | 0.3543 / 0.5010 / 1.0000 |
| half depth, near corner | 2.1450 | 26.933 | 0.1938 | 0.3448 / 0.5619 / 1.0000 |
| half depth, corner | 2.1171 | 30.000 | 0.1932 | 0.3340 / 0.5785 / 1.0000 |

Surface cells over 1.05 edge ratio: **83.1 %**; over 3° skew: **69.7 %**.
Seam kink (grid-line bend across a chart edge): edge middle 0.0000°, quarter 31.400°, near corner 57.900°.

### Spherified

| where | edge ratio | skew ° | volume | σ1 / σ2 / σ3 |
|---|---|---|---|---|
| surface, centre | 1.1107 | 0.000 | 0.8106 | 0.9003 / 0.9003 / 1.0000 |
| surface, half to edge | 1.1601 | 0.000 | 0.8296 | 0.8620 / 0.9625 / 1.0000 |
| surface, edge middle | 1.7321 | 0.000 | 0.9360 | 0.7351 / 1.0000 / 1.2732 |
| surface, near corner | 1.0963 | 22.389 | 0.7693 | 0.7177 / 1.0000 / 1.0719 |
| surface, corner | 1.1107 | 30.000 | 0.7020 | 0.6366 / 1.0000 / 1.1027 |
| 5 % deep, centre | 1.1692 | 0.000 | 0.7315 | 0.8553 / 0.8553 / 1.0000 |
| 5 % deep, half to edge | 1.2212 | 0.000 | 0.7488 | 0.8189 / 0.9144 / 1.0000 |
| 5 % deep, edge middle | 1.7321 | 0.000 | 0.8447 | 0.6983 / 1.0000 / 1.2096 |
| 5 % deep, near corner | 1.1540 | 22.389 | 0.6943 | 0.6818 / 1.0000 / 1.0183 |
| 5 % deep, corner | 1.1692 | 30.000 | 0.6335 | 0.6048 / 1.0000 / 1.0475 |
| 25 % deep, centre | 1.4810 | 0.000 | 0.4559 | 0.6752 / 0.6752 / 1.0000 |
| 25 % deep, half to edge | 1.5468 | 0.000 | 0.4667 | 0.6465 / 0.7219 / 1.0000 |
| 25 % deep, edge middle | 1.8138 | 0.000 | 0.5265 | 0.5513 / 0.9549 / 1.0000 |
| 25 % deep, near corner | 1.4617 | 22.389 | 0.4328 | 0.5383 / 0.8039 / 1.0000 |
| 25 % deep, corner | 1.4810 | 30.000 | 0.3949 | 0.4775 / 0.8270 / 1.0000 |
| half depth, centre | 2.2170 | 0.000 | 0.2035 | 0.4511 / 0.4511 / 1.0000 |
| half depth, half to edge | 2.3156 | 0.000 | 0.2082 | 0.4319 / 0.4822 / 1.0000 |
| half depth, edge middle | 2.7153 | 0.000 | 0.2349 | 0.3683 / 0.6379 / 1.0000 |
| half depth, near corner | 2.1882 | 22.389 | 0.1931 | 0.3596 / 0.5370 / 1.0000 |
| half depth, corner | 2.2170 | 30.000 | 0.1762 | 0.3189 / 0.5524 / 1.0000 |

Surface cells over 1.05 edge ratio: **100.0 %**; over 3° skew: **43.2 %**.
Seam kink (grid-line bend across a chart edge): edge middle 0.0000°, quarter 18.734°, near corner 54.403°.

## R = 8e6 (n = 12566371)

### Gnomonic

| where | edge ratio | skew ° | volume | σ1 / σ2 / σ3 |
|---|---|---|---|---|
| surface, centre | 1.2732 | 0.000 | 1.6211 | 1.0000 / 1.2732 / 1.2732 |
| surface, half to edge | 1.1388 | 0.000 | 1.1600 | 1.0000 / 1.0186 / 1.1388 |
| surface, edge middle | 1.5708 | 0.000 | 0.5732 | 0.6366 / 0.9003 / 1.0000 |
| surface, near corner | 1.5836 | 27.976 | 0.3522 | 0.4601 / 0.7654 / 1.0000 |
| surface, corner | 1.6661 | 30.000 | 0.3120 | 0.4244 / 0.7351 / 1.0000 |
| 5 % deep, centre | 1.2096 | 0.000 | 1.4631 | 1.0000 / 1.2096 / 1.2096 |
| 5 % deep, half to edge | 1.1180 | 0.000 | 1.0469 | 0.9677 / 1.0000 / 1.0819 |
| 5 % deep, edge middle | 1.6535 | 0.000 | 0.5173 | 0.6048 / 0.8553 / 1.0000 |
| 5 % deep, near corner | 1.6669 | 27.976 | 0.3178 | 0.4371 / 0.7271 / 1.0000 |
| 5 % deep, corner | 1.7538 | 30.000 | 0.2816 | 0.4032 / 0.6983 / 1.0000 |
| 25 % deep, centre | 1.0472 | 0.000 | 0.9119 | 0.9549 / 0.9549 / 1.0000 |
| 25 % deep, half to edge | 1.3090 | 0.000 | 0.6525 | 0.7639 / 0.8541 / 1.0000 |
| 25 % deep, edge middle | 2.0944 | 0.000 | 0.3224 | 0.4775 / 0.6752 / 1.0000 |
| 25 % deep, near corner | 2.1114 | 27.976 | 0.1981 | 0.3451 / 0.5741 / 1.0000 |
| 25 % deep, corner | 2.2214 | 30.000 | 0.1755 | 0.3183 / 0.5513 / 1.0000 |
| half depth, centre | 1.5677 | 0.000 | 0.4069 | 0.6379 / 0.6379 / 1.0000 |
| half depth, half to edge | 1.9596 | 0.000 | 0.2912 | 0.5103 / 0.5705 / 1.0000 |
| half depth, edge middle | 3.1353 | 0.000 | 0.1439 | 0.3189 / 0.4511 / 1.0000 |
| half depth, near corner | 3.1608 | 27.976 | 0.0884 | 0.2305 / 0.3835 / 1.0000 |
| half depth, corner | 3.3255 | 30.000 | 0.0783 | 0.2126 / 0.3683 / 1.0000 |

Surface cells over 1.05 edge ratio: **97.9 %**; over 3° skew: **76.9 %**.
Seam kink (grid-line bend across a chart edge): edge middle 0.0000°, quarter 36.870°, near corner 58.641°.

### Equiangular

| where | edge ratio | skew ° | volume | σ1 / σ2 / σ3 |
|---|---|---|---|---|
| surface, centre | 1.0000 | 0.000 | 1.0000 | 1.0000 / 1.0000 / 1.0000 |
| surface, half to edge | 1.0824 | 0.000 | 0.9239 | 0.9239 / 1.0000 / 1.0000 |
| surface, edge middle | 1.4142 | 0.000 | 0.7071 | 0.7071 / 1.0000 / 1.0000 |
| surface, near corner | 1.0746 | 26.933 | 0.7720 | 0.6883 / 1.0000 / 1.1217 |
| surface, corner | 1.0607 | 30.000 | 0.7698 | 0.6667 / 1.0000 / 1.1547 |
| 5 % deep, centre | 1.0526 | 0.000 | 0.9025 | 0.9500 / 0.9500 / 1.0000 |
| 5 % deep, half to edge | 1.1394 | 0.000 | 0.8338 | 0.8777 / 0.9500 / 1.0000 |
| 5 % deep, edge middle | 1.4886 | 0.000 | 0.6382 | 0.6718 / 0.9500 / 1.0000 |
| 5 % deep, near corner | 1.1312 | 26.933 | 0.6967 | 0.6538 / 1.0000 / 1.0656 |
| 5 % deep, corner | 1.1165 | 30.000 | 0.6947 | 0.6333 / 1.0000 / 1.0970 |
| 25 % deep, centre | 1.3333 | 0.000 | 0.5625 | 0.7500 / 0.7500 / 1.0000 |
| 25 % deep, half to edge | 1.4432 | 0.000 | 0.5197 | 0.6929 / 0.7500 / 1.0000 |
| 25 % deep, edge middle | 1.8856 | 0.000 | 0.3977 | 0.5303 / 0.7500 / 1.0000 |
| 25 % deep, near corner | 1.4329 | 26.933 | 0.4342 | 0.5162 / 0.8412 / 1.0000 |
| 25 % deep, corner | 1.4142 | 30.000 | 0.4330 | 0.5000 / 0.8660 / 1.0000 |
| half depth, centre | 1.9960 | 0.000 | 0.2510 | 0.5010 / 0.5010 / 1.0000 |
| half depth, half to edge | 2.1605 | 0.000 | 0.2319 | 0.4629 / 0.5010 / 1.0000 |
| half depth, edge middle | 2.8228 | 0.000 | 0.1775 | 0.3543 / 0.5010 / 1.0000 |
| half depth, near corner | 2.1450 | 26.933 | 0.1938 | 0.3448 / 0.5620 / 1.0000 |
| half depth, corner | 2.1171 | 30.000 | 0.1932 | 0.3340 / 0.5785 / 1.0000 |

Surface cells over 1.05 edge ratio: **83.1 %**; over 3° skew: **69.7 %**.
Seam kink (grid-line bend across a chart edge): edge middle 0.0000°, quarter 31.400°, near corner 57.900°.

### Spherified

| where | edge ratio | skew ° | volume | σ1 / σ2 / σ3 |
|---|---|---|---|---|
| surface, centre | 1.1107 | 0.000 | 0.8106 | 0.9003 / 0.9003 / 1.0000 |
| surface, half to edge | 1.1601 | 0.000 | 0.8296 | 0.8620 / 0.9625 / 1.0000 |
| surface, edge middle | 1.7321 | 0.000 | 0.9360 | 0.7351 / 1.0000 / 1.2732 |
| surface, near corner | 1.0963 | 22.389 | 0.7693 | 0.7177 / 1.0000 / 1.0719 |
| surface, corner | 1.1107 | 30.000 | 0.7020 | 0.6366 / 1.0000 / 1.1027 |
| 5 % deep, centre | 1.1692 | 0.000 | 0.7315 | 0.8553 / 0.8553 / 1.0000 |
| 5 % deep, half to edge | 1.2212 | 0.000 | 0.7488 | 0.8189 / 0.9144 / 1.0000 |
| 5 % deep, edge middle | 1.7321 | 0.000 | 0.8447 | 0.6983 / 1.0000 / 1.2096 |
| 5 % deep, near corner | 1.1540 | 22.389 | 0.6943 | 0.6818 / 1.0000 / 1.0183 |
| 5 % deep, corner | 1.1692 | 30.000 | 0.6335 | 0.6048 / 1.0000 / 1.0475 |
| 25 % deep, centre | 1.4810 | 0.000 | 0.4559 | 0.6752 / 0.6752 / 1.0000 |
| 25 % deep, half to edge | 1.5468 | 0.000 | 0.4667 | 0.6465 / 0.7219 / 1.0000 |
| 25 % deep, edge middle | 1.8138 | 0.000 | 0.5265 | 0.5513 / 0.9549 / 1.0000 |
| 25 % deep, near corner | 1.4617 | 22.389 | 0.4328 | 0.5383 / 0.8039 / 1.0000 |
| 25 % deep, corner | 1.4810 | 30.000 | 0.3949 | 0.4775 / 0.8270 / 1.0000 |
| half depth, centre | 2.2170 | 0.000 | 0.2035 | 0.4511 / 0.4511 / 1.0000 |
| half depth, half to edge | 2.3156 | 0.000 | 0.2082 | 0.4319 / 0.4822 / 1.0000 |
| half depth, edge middle | 2.7153 | 0.000 | 0.2349 | 0.3683 / 0.6379 / 1.0000 |
| half depth, near corner | 2.1882 | 22.389 | 0.1931 | 0.3596 / 0.5370 / 1.0000 |
| half depth, corner | 2.2170 | 30.000 | 0.1762 | 0.3189 / 0.5524 / 1.0000 |

Surface cells over 1.05 edge ratio: **100.0 %**; over 3° skew: **43.2 %**.
Seam kink (grid-line bend across a chart edge): edge middle 0.0000°, quarter 18.734°, near corner 54.403°.

## Core transition (guide §10.5)

A Cartesian core cube of half-size `a = r_in / 2` joined to the innermost spherical band (radius `r_in`)
by six mapped blocks whose cells interpolate linearly from the cube face to the sphere; the shell is
`r_in − a` thick at a face centre but only `r_in − a√3` at the cube corners.

| where | edge ratio | skew ° | volume |
|---|---|---|---|
| centre, inner | 2.000 | 0.01 | 0.250 |
| centre, middle | 1.333 | 0.01 | 0.563 |
| centre, outer | 1.000 | 0.01 | 1.000 |
| edge middle, inner | 1.993 | 44.90 | 0.208 |
| edge middle, middle | 1.568 | 22.41 | 0.303 |
| edge middle, outer | 1.700 | 0.03 | 0.417 |
| corner, inner | 3.665 | 35.22 | 0.156 |
| corner, middle | 3.398 | 20.71 | 0.182 |
| corner, outer | 3.466 | 29.89 | 0.210 |
