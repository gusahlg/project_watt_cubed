//! Section mesher: turns a section's cells into GPU-ready [`MeshData`].
//!
//! Each solid run emits up to six faces. Vertical faces split where column heights differ
//! (so overhangs render correctly, unlike a heightmap-only approach). Opacity rules: opaque
//! blocks hide faces; translucent (water) does not cover solid; two translucent blocks of
//! the same type hide their shared edge (no internal walls).
//!
//! Section borders (edge columns) overdraw as air with an inward micro-offset to prevent
//! z-fighting with adjacent sections at different detail levels.
//!
//! The mesher reads a DENSE column-major quadrant grid ([`DenseQuad`]): every
//! neighbour/AO probe is one O(1) stride-add off a flat index, where the old
//! run-list walk paid an O(runs) linear scan per probe (~14 probes × 24k face
//! positions per 16³ block adds up). The grid has two producers sharing one
//! pooled worker-local buffer:
//! - [`build_section_mesh`] decodes a stored [`Section`]'s brick stacks — the
//!   reference path, kept as the byte-parity oracle;
//! - [`extract_section_mesh`] samples the generator (and folds edits) straight
//!   into the grid — the production worker path, which never builds the RLE
//!   brick storage only to decode it again.
//!
//! Output: a 2×2 grid of 16³-cell blocks per section, only for the vertical range occupied
//! by solid geometry (sky/deep space cost nothing). Per-corner AO samples neighbour-column
//! occupancy within the quadrant (quadrant-border corners see no occluder, so read
//! unoccluded); light is not baked — every vertex takes neutral daylight (full sky, no
//! blocklight) so coarse tiles track day/night.
use std::cell::RefCell;

use glam::UVec3;
use voxel_engine::{Ao, Light, MeshVertex, Normal, Pass};

use super::super::generation::TerrainGenerator;
use super::super::mesh::{ChunkMeshData, new_chunk_mesh_data};
use super::{BRICK_DIM, ChunkCoord, DOMAIN_H, SECTION_N, SectionPos};
#[cfg(test)]
use super::{BrickStack, Section};
use crate::block::registry::{AIR, BlockId, HotTables};
use super::super::mesh::face::{self, covered, vertex_ao, corner_uv};

/// One block's mesh plus its origin in cells within the section. Only non-empty blocks appear.
pub(in crate::world) type SectionMeshData = Vec<(UVec3, ChunkMeshData)>;

/// Cells per mesh-block edge — the 5-bit vertex position range (`0..=16`). Fixed by design.
const BLOCK: i32 = 16;
const QUAD_N: usize = SECTION_N / 2;
#[cfg(test)]
const BLOCKS_XZ: i32 = SECTION_N as i32 / BLOCK;
const SLICE: usize = (BLOCK * BLOCK) as usize;

// One dense quadrant buffer and one mesh scratch per worker. Empty 16³
// blocks reuse the scratch; only non-empty results move out. Born and
// dropped on the same thread (unlike the main-thread-captured mesh
// snapshots), so a lock-free thread-local is right.
thread_local! {
    static DENSE_QUAD: RefCell<Vec<BlockId>> = const { RefCell::new(Vec::new()) };
    static MESH_SCRATCH: RefCell<ChunkMeshData> = RefCell::new(new_chunk_mesh_data());
}

/// One quadrant's cells as a dense column-major grid: `QUAD_N × QUAD_N`
/// columns of `n_cells` cells each. Column-major keeps a column's vertical
/// neighbours adjacent — the mesher's most frequent probe direction.
struct DenseQuad<'a> {
    cells: &'a [BlockId],
    n_cells: i32,
}

impl DenseQuad<'_> {
    /// Flat-index read — the sweep's stride walk.
    #[inline]
    fn at_flat(&self, i: usize) -> BlockId {
        self.cells[i]
    }

    /// Index delta of one step along world axis `axis` (0=X, 1=Y, 2=Z) in the
    /// column-major `QUAD_N × QUAD_N × n_cells` layout — derived from the one
    /// layout law rather than restated. The sweep walks flat indices with
    /// these strides: every probe is `base ± s` where the coordinate form
    /// paid three `[i32; 3]` writes through dynamic axis indices plus two
    /// multiplies, per neighbour/AO sample.
    #[inline]
    fn axis_stride(&self, axis: usize) -> i32 {
        let mut p = [0i32; 3];
        p[axis] = 1;
        (p[0] + p[2] * QUAD_N as i32) * self.n_cells + p[1]
    }
}

/// Merge key: block, micro, and AO must all match — an AO gradient must never
/// merge into a flat quad (mirrors the chunk mesher's `FaceSample`).
#[derive(Clone, Copy, PartialEq, Eq)]
struct FaceSample {
    block: BlockId,
    micro: [i8; 3],
    ao: [u8; 4],
}

/// Face direction with corner winding; micro-offset zero for verticals (never borders).
struct Dir {
    face: face::Dir,
    micro: [i8; 3],
}

const DIRS: [Dir; 6] = [
    Dir { face: face::DIRS[0], micro: [-1, 0, 0] },
    Dir { face: face::DIRS[1], micro: [1, 0, 0] },
    Dir { face: face::DIRS[2], micro: [0, 0, 0] },
    Dir { face: face::DIRS[3], micro: [0, 0, 0] },
    Dir { face: face::DIRS[4], micro: [0, 0, -1] },
    Dir { face: face::DIRS[5], micro: [0, 0, 1] },
];

/// Opaque-occupancy probe for AO sampling: below-floor reads solid (matches the
/// cull rule's "solid ground"), above-ceiling and outside the quadrant read air
/// (matches the border-overdraw convention) — never data this quadrant lacks.
/// `idx` is the probe's flat index; bounds are checked on `(x, y, z)` first
/// because a stride step off the quadrant can land on a *different column's*
/// in-range cell.
#[inline]
fn occluder(quad: &DenseQuad<'_>, tables: &HotTables, idx: i32, x: i32, y: i32, z: i32) -> bool {
    if y < 0 {
        return true;
    }
    if y >= quad.n_cells || x < 0 || x >= QUAD_N as i32 || z < 0 || z >= QUAD_N as i32 {
        return false;
    }
    tables.opaque(quad.at_flat(idx as usize))
}

/// Sample a face: cull if covered by neighbor; overdraw section edges as air.
/// Per-corner AO reads the two in-plane occluders plus the diagonal, in the
/// layer the face opens into — same stencil `face_sample` in `world/mesh.rs`
/// uses. `idx` is the cell's flat index; every probe is a stride add off it.
#[inline]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)] // 3×3 stencil: du/dv are both indices and signed offsets
fn face_sample(
    quad: &DenseQuad<'_>,
    tables: &HotTables,
    dir: &Dir,
    s_n: i32,
    s_u: i32,
    s_v: i32,
    idx: i32,
    x: i32,
    y: i32,
    z: i32,
    corner_uv: &[[i32; 2]; 4],
) -> Option<FaceSample> {
    if y < 0 || y >= quad.n_cells {
        return None; // above the ceiling in the top block: no cell here
    }
    let me = quad.at_flat(idx as usize);
    if me == AIR {
        return None;
    }
    let open = idx + dir.face.step * s_n;
    let mut micro = [0i8; 3];
    let nbr = if dir.face.n_axis == 1 {
        let ny = y + dir.face.step;
        if ny < 0 {
            return None; // below the floor: solid ground, never a silhouette
        } else if ny >= quad.n_cells {
            AIR // above the ceiling: open sky
        } else {
            quad.at_flat(open as usize)
        }
    } else {
        let n = if dir.face.n_axis == 0 { x } else { z };
        if n + dir.face.step < 0 || n + dir.face.step >= QUAD_N as i32 {
            // Quadrant border: overdraw as air, nudge inward.
            micro = dir.micro;
            AIR
        } else {
            quad.at_flat(open as usize)
        }
    };
    if covered(me, nbr, tables) {
        return None;
    }
    // Open-cell coordinates: one step along the normal. u/v of each Dir
    // are fixed per n_axis (X: u=Z v=Y; Y: u=X v=Z; Z: u=X v=Y).
    let (ox, oy, oz) = match dir.face.n_axis {
        0 => (x + dir.face.step, y, z),
        1 => (x, y + dir.face.step, z),
        _ => (x, y, z + dir.face.step),
    };
    // One 3×3 stencil in the OPEN layer — same layout as world/mesh.rs.
    // Bounds live in `occluder`; a stride step off the quadrant can land
    // on a different column's in-range cell.
    let mut opaque = [[false; 3]; 3];
    for dv in 0..3 {
        let ev = dv as i32 - 1;
        for du in 0..3 {
            let eu = du as i32 - 1;
            let pidx = open + eu * s_u + ev * s_v;
            opaque[du][dv] = match dir.face.n_axis {
                0 => occluder(quad, tables, pidx, ox, oy + ev, oz + eu),
                1 => occluder(quad, tables, pidx, ox + eu, oy, oz + ev),
                _ => occluder(quad, tables, pidx, ox + eu, oy + ev, oz),
            };
        }
    }
    let ao = std::array::from_fn(|i| {
        let ou = (corner_uv[i][0] + 1) as usize;
        let ov = (corner_uv[i][1] + 1) as usize;
        vertex_ao(opaque[ou][1], opaque[1][ov], opaque[ou][ov])
    });
    Some(FaceSample { block: me, micro, ao })
}

/// Greedy-mesh one block: merge adjacent quads with identical properties.
/// `y_base` is the block's Y origin in quadrant cells (X and Z are 0).
fn build_block(
    quad: &DenseQuad<'_>,
    y_base: i32,
    tables: &HotTables,
    out: &mut ChunkMeshData,
) -> bool {
    let mut mask: [Option<FaceSample>; SLICE] = [None; SLICE];
    let mut emitted = false;
    // Origin (0, y_base, 0); Y-stride is 1, so the flat base is y_base.
    let base_idx = y_base;

    for dir in &DIRS {
        let (s_n, s_u, s_v) = (
            quad.axis_stride(dir.face.n_axis),
            quad.axis_stride(dir.face.u_axis),
            quad.axis_stride(dir.face.v_axis),
        );
        let corner_uv = corner_uv(&dir.face.corners);

        for n in 0..BLOCK {
            let mut any = false;
            for v in 0..BLOCK {
                let row_idx = base_idx + n * s_n + v * s_v;
                for u in 0..BLOCK {
                    let idx = row_idx + u * s_u;
                    let (x, y, z) = match dir.face.n_axis {
                        0 => (n, y_base + v, u),
                        1 => (u, y_base + n, v),
                        _ => (u, y_base + v, n),
                    };
                    let cell = face_sample(
                        quad, tables, dir, s_n, s_u, s_v, idx, x, y, z, &corner_uv,
                    );
                    mask[(u + v * BLOCK) as usize] = cell;
                    any |= cell.is_some();
                }
            }
            if !any {
                continue;
            }
            for v0 in 0..BLOCK {
                for u0 in 0..BLOCK {
                    let key = mask[(u0 + v0 * BLOCK) as usize];
                    let Some(sample) = key else { continue };
                    let mut w = 1;
                    while u0 + w < BLOCK && mask[(u0 + w + v0 * BLOCK) as usize] == key {
                        w += 1;
                    }
                    let mut h = 1;
                    'grow: while v0 + h < BLOCK {
                        for k in 0..w {
                            if mask[(u0 + k + (v0 + h) * BLOCK) as usize] != key {
                                break 'grow;
                            }
                        }
                        h += 1;
                    }
                    for dv in 0..h {
                        let row = (v0 + dv) * BLOCK;
                        for du in 0..w {
                            mask[(u0 + du + row) as usize] = None;
                        }
                    }
                    emit(out, dir, n, u0, v0, w, h, sample, tables);
                    emitted = true;
                }
            }
        }
    }
    emitted
}

/// Append one merged rectangle as a single quad, routed to its block's pass with
/// the border micro-offset applied. Corners are block-local (`0..=16`).
#[allow(clippy::too_many_arguments)]
fn emit(
    out: &mut ChunkMeshData,
    dir: &Dir,
    nslice: i32,
    u0: i32,
    v0: i32,
    w: i32,
    h: i32,
    sample: FaceSample,
    tables: &HotTables,
) {
    let mut origin = [0u32; 3];
    origin[dir.face.n_axis] = nslice as u32;
    origin[dir.face.u_axis] = u0 as u32;
    origin[dir.face.v_axis] = v0 as u32;
    let layer = sample.block.0;
    // Route water to opaque pass (no animated texturing at LOD range).
    let is_fluid = tables.fluid_surface(BlockId(layer));
    let pass = if is_fluid { Pass::Opaque } else { tables.layer[layer as usize] };
    let mut corners: [MeshVertex; 4] = std::array::from_fn(|i| {
        let cr = dir.face.corners[i];
        let mut pos = [0u32; 3];
        pos[dir.face.n_axis] = origin[dir.face.n_axis] + cr[0] as u32;
        pos[dir.face.u_axis] = origin[dir.face.u_axis] + cr[1] as u32 * w as u32;
        pos[dir.face.v_axis] = origin[dir.face.v_axis] + cr[2] as u32 * h as u32;
        MeshVertex::new(
            [pos[0] as u8, pos[1] as u8, pos[2] as u8],
            dir.face.normal,
            // Vertex layer only (tables above index by the true id); wraps
            // past the device texture-layer cap like the chunk mesher.
            layer % tables.layer_cap,
            Ao::new(sample.ao[i]),
            // Coarse LOD tiles have no smooth-light field — neutral daylight so
            // their shading tracks day/night via skylight rather than clamping.
            Light::DAY,
            false,
        )
        .with_micro(sample.micro)
    });
    // Same anisotropy fix as the chunk mesher: rotate the quad so the fixed
    // diagonal falls on the darker corner pair, not the brighter one.
    if (sample.ao[0] as u32 + sample.ao[2] as u32) < (sample.ao[1] as u32 + sample.ao[3] as u32) {
        corners.rotate_left(1);
    }
    out[pass].quad(corners);
}

/// Mesh a whole section as four independent quadrant sub-meshes (indexed by
/// [`SectionPos::quadrant`]), each over its 16×16 column sub-grid. A quadrant's
/// outer edge is treated exactly like a section border — overdrawn as if the
/// neighbour were air, with the inward micro-nudge — because the abutting quadrant
/// may be drawn at a different detail or not at all. Deterministic: the same
/// section yields bit-identical output per quadrant (fixed iteration order,
/// dense-grid sampling with no floats).
///
/// This is the REFERENCE path (stored bricks → dense grid → mesh); production
/// far jobs take [`extract_section_mesh`], which the parity test pins against
/// this one byte for byte.
#[cfg(test)]
pub(in crate::world) fn build_section_mesh(section: &Section, tables: &HotTables) -> [SectionMeshData; 4] {
    let n_cells = (DOMAIN_H / section.pos().cell_size()) as usize;
    mesh_section_with(n_cells, tables, |dense, q| {
        fill_from_stack(dense, n_cells, &section.quadrants[q as usize])
    })
}

/// Extract AND mesh a section in one pass: sample the generator (folding the
/// edit overlay) directly into the dense quadrant grid, then mesh it — the
/// production worker path. Skips the whole RLE brick round trip
/// ([`Section::extract`]'s per-column run slicing plus this module's decode),
/// which existed only because the temporary [`Section`] was built and dropped.
pub(in crate::world) fn extract_section_mesh<G: TerrainGenerator + ?Sized>(
    pos: SectionPos,
    r#gen: &G,
    edits: &[(ChunkCoord, Vec<(usize, BlockId)>)],
    tables: &HotTables,
) -> [SectionMeshData; 4] {
    let mut out = std::array::from_fn(|_| SectionMeshData::new());
    extract_section_mesh_into(pos, r#gen, edits, tables, &mut out);
    out
}

/// Production worker path: fill a pooled `[SectionMeshData; 4]`, reusing each
/// quadrant's `Vec` and inner `ChunkMeshData` capacities.
pub(in crate::world) fn extract_section_mesh_into<G: TerrainGenerator + ?Sized>(
    pos: SectionPos,
    r#gen: &G,
    edits: &[(ChunkCoord, Vec<(usize, BlockId)>)],
    tables: &HotTables,
    out: &mut [SectionMeshData; 4],
) {
    let cell = pos.cell_size();
    let n_cells = (DOMAIN_H / cell) as usize;
    let ys = super::cell_centers(pos);
    let flat = super::flatten_edits(edits);
    mesh_section_into(n_cells, tables, out, |dense, q| {
        let (qx, qz) = ((q & 1) as usize, (q >> 1) as usize);
        for lz in 0..BRICK_DIM {
            for lx in 0..BRICK_DIM {
                let (ix, iz) = (qx * BRICK_DIM + lx, qz * BRICK_DIM + lz);
                let (fx, fz) = (pos.min_x() + ix as i32 * cell, pos.min_z() + iz as i32 * cell);
                let (wx, wz) = (fx + cell / 2, fz + cell / 2);
                // The mesher's column-major layout doubles as the
                // generator's output slice: no per-column scratch at all.
                let column = &mut dense[(lx + lz * QUAD_N) * n_cells..][..n_cells];
                r#gen.lod_column(wx, wz, &ys, column);
                super::apply_edits(column, &flat, fx, fz, cell);
            }
        }
    });
}

/// The one section-mesh driver both producers share: for each quadrant, `fill`
/// overwrites this worker's pooled dense grid (contents unspecified on entry —
/// every cell must be written), then the mesher runs over it. Producers differ
/// ONLY in how the grid is filled (stored bricks vs live generator), so their
/// meshing can never diverge.
fn mesh_section_with(
    n_cells: usize,
    tables: &HotTables,
    fill: impl FnMut(&mut [BlockId], u8),
) -> [SectionMeshData; 4] {
    let mut out = std::array::from_fn(|_| SectionMeshData::new());
    mesh_section_into(n_cells, tables, &mut out, fill);
    out
}

fn mesh_section_into(
    n_cells: usize,
    tables: &HotTables,
    out: &mut [SectionMeshData; 4],
    mut fill: impl FnMut(&mut [BlockId], u8),
) {
    DENSE_QUAD.with_borrow_mut(|dense| {
        dense.resize(QUAD_N * QUAD_N * n_cells, AIR);
        for q in 0..4 {
            fill(dense, q as u8);
            mesh_quadrant(
                &DenseQuad {
                    cells: dense,
                    n_cells: n_cells as i32,
                },
                tables,
                q as u8,
                &mut out[q],
            );
        }
    });
}

/// Decode one quadrant's [`BrickStack`] into the dense grid (column-major).
/// Stack runs can extend past `n_cells` at the coarse rings (bricks pad to 16
/// cells with AIR); the column slice bound clips them exactly as the old
/// `n_cells`-bounded reader ignored them.
#[cfg(test)]
fn fill_from_stack(dense: &mut [BlockId], n_cells: usize, stack: &BrickStack) {
    for iz in 0..QUAD_N {
        for ix in 0..QUAD_N {
            let column = &mut dense[(ix + iz * QUAD_N) * n_cells..][..n_cells];
            let mut at = 0usize;
            for run in stack.column_runs(ix, iz) {
                let end = (at + run.count as usize).min(n_cells);
                column[at..end].fill(run.block);
                at = end;
                if at == n_cells {
                    break;
                }
            }
        }
    }
}

/// Mesh one quadrant `q` (its 16×16 column sub-grid) into section-space block
/// origins. Block origins are in CELLS relative to the section min-corner, so the
/// caller positions them the same way regardless of quadrant.
fn mesh_quadrant(
    quad: &DenseQuad<'_>,
    tables: &HotTables,
    q: u8,
    out: &mut SectionMeshData,
) {
    let (qx, qz) = ((q & 1) as usize, (q >> 1) as usize);
    let n_cells = quad.n_cells;

    // The vertical slab that actually holds solid cells — sky and deep space
    // are skipped entirely, so the block count is small for thin terrain.
    let (mut ylo, mut yhi) = (n_cells, 0);
    for col in quad.cells.chunks_exact(n_cells as usize) {
        // Scan from both ends: terrain columns are solid-below/air-above, so
        // each end test stops at the first hit instead of walking the middle.
        if let Some(first) = col.iter().position(|&id| id != AIR) {
            let last = col.iter().rposition(|&id| id != AIR).expect("some cell is non-air");
            ylo = ylo.min(first as i32);
            yhi = yhi.max(last as i32 + 1);
        }
    }
    if yhi <= ylo {
        out.clear();
        return;
    }

    // Section-space cell origin of the quadrant's XZ corner (0 or 16).
    let (ox, oz) = ((qx * QUAD_N) as u32, (qz * QUAD_N) as u32);
    let mut used = 0usize;
    MESH_SCRATCH.with_borrow_mut(|scratch| {
        for by in ylo / BLOCK..=(yhi - 1) / BLOCK {
            for (_, m) in scratch.iter_mut() {
                m.clear();
            }
            if build_block(quad, by * BLOCK, tables, scratch) {
                let origin = UVec3::new(ox, (by * BLOCK) as u32, oz);
                if used < out.len() {
                    out[used].0 = origin;
                    std::mem::swap(&mut out[used].1, scratch);
                } else {
                    out.push((origin, std::mem::replace(scratch, new_chunk_mesh_data())));
                }
                used += 1;
            }
        }
    });
    out.truncate(used);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::BlockRegistry;
    use crate::world::generation::{Terrain, TerrainGenerator};
    use voxel_engine::Pass;
    use crate::world::section::{FINEST_DETAIL, SectionPos};

    // Test fixtures

    struct FnGen<H, B> {
        h: H,
        b: B,
        surf: BlockId,
        deep: BlockId,
    }
    impl<H, B> TerrainGenerator for FnGen<H, B>
    where
        H: Fn(i32, i32) -> i32 + Send + Sync,
        B: Fn(i32, i32, i32) -> BlockId + Send + Sync,
    {
        fn height(&self, wx: i32, wz: i32) -> i32 {
            (self.h)(wx, wz)
        }
        fn surface_at(&self, _: i32, _: i32) -> BlockId {
            self.surf
        }
        fn deep(&self) -> BlockId {
            self.deep
        }
        fn lod_block_at(&self, wx: i32, wy: i32, wz: i32) -> BlockId {
            (self.b)(wx, wy, wz)
        }
    }

    struct Blocks {
        grass: BlockId,
        dirt: BlockId,
        stone: BlockId,
        sand: BlockId,
        water: BlockId,
    }
    fn setup() -> (BlockRegistry, HotTables, Blocks) {
        // Compile the placement table so the hot tables cover every id the
        // real generator can emit (the fixtures below use builtin names only).
        let mut r = BlockRegistry::with_builtins();
        crate::world::placement::builtin().compile(&mut r);
        let id = |n: &str| r.id_by_name(n).unwrap();
        let blocks = Blocks {
            grass: id("Soil+Organic"),
            dirt: id("Soil+Clay"),
            stone: id("Stone"),
            sand: id("Sand"),
            water: id("Water"),
        };
        let tables = r.hot_tables();
        (r, tables, blocks)
    }

    const FINEST: SectionPos = SectionPos { detail: FINEST_DETAIL, x: 0, z: 0 };
    const CELL: i32 = 1 << FINEST_DETAIL.0;

    /// Terrain with configurable surface height, water table, and floating shelf.
    fn terrain_gen(
        b: &Blocks,
        h: i32,
        water: i32,
        shelf: Option<(i32, i32)>,
    ) -> FnGen<impl Fn(i32, i32) -> i32, impl Fn(i32, i32, i32) -> BlockId> {
        let (grass, dirt, stone, sand, water_id) = (b.grass, b.dirt, b.stone, b.sand, b.water);
        FnGen {
            h: move |_, _| h,
            b: move |_, y, _| {
                if y < h - 3 {
                    stone
                } else if y < h - 1 {
                    dirt
                } else if y < h {
                    if h <= water { sand } else { grass }
                } else if y < water {
                    water_id
                } else if shelf.is_some_and(|(lo, hi)| y >= lo && y < hi) {
                    stone
                } else {
                    AIR
                }
            },
            surf: grass,
            deep: stone,
        }
    }

    fn extract(pos: SectionPos, g: &impl TerrainGenerator) -> Section {
        Section::extract(pos, g, &[], voxel_engine::Rev::START)
    }

    fn mesh_of(section: &Section, tables: &HotTables) -> [SectionMeshData; 4] {
        build_section_mesh(section, tables)
    }

    fn all_quads(mesh: &[SectionMeshData; 4]) -> impl Iterator<Item = (UVec3, Pass, [MeshVertex; 4])> {
        mesh.iter().flatten().flat_map(|(origin, data)| {
            Pass::ALL.into_iter().flat_map(move |p| {
                data[p]
                    .vertices()
                    .chunks_exact(4)
                    .map(|q| (*origin, p, [q[0], q[1], q[2], q[3]]))
                    .collect::<Vec<_>>()
            })
        })
    }

    fn normals_present(mesh: &[SectionMeshData; 4], want: Normal) -> bool {
        all_quads(mesh).any(|(_, _, q)| q[0].normal() == want)
    }

    /// Every quad must wind counter-clockwise from outside (engine back-face-culls otherwise).
    fn assert_winds_outward(mesh: &[SectionMeshData; 4]) {
        let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let cross = |a: [f32; 3], b: [f32; 3]| {
            [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
        };
        for (origin, _, q) in all_quads(mesh) {
            let p: Vec<[f32; 3]> = q
                .iter()
                .map(|v| {
                    let l = v.local_pos();
                    [l[0] + origin.x as f32, l[1] + origin.y as f32, l[2] + origin.z as f32]
                })
                .collect::<Vec<_>>();
            let c = cross(sub(p[1], p[0]), sub(p[3], p[0]));
            let d = q[0].normal().direction();
            let dot = c[0] * d[0] as f32 + c[1] * d[1] as f32 + c[2] * d[2] as f32;
            assert!(dot > 0.0, "quad faces inward: normal {:?}", q[0].normal());
        }
    }

    // Test cases

    #[test]
    fn flat_terrain_shows_tops_and_only_border_side_walls() {
        let (_r, tables, b) = setup();
        let sec = extract(FINEST, &terrain_gen(&b, 200, 0, None));
        let mesh = mesh_of(&sec, &tables);
        assert!(!mesh.is_empty(), "flat ground has geometry");
        assert!(normals_present(&mesh, Normal::PosY), "the surface has a top");
        assert_winds_outward(&mesh);

        // Interior side faces culled; only section-edge quads have micro offset.
        for (_, _, q) in all_quads(&mesh) {
            let side = matches!(
                q[0].normal(),
                Normal::PosX | Normal::NegX | Normal::PosZ | Normal::NegZ
            );
            let micro = q[0].micro();
            if side {
                assert_ne!(micro, [0, 0, 0], "an interior side wall leaked into flat terrain");
            } else {
                assert_eq!(micro, [0, 0, 0], "a top/bottom face carries a spurious micro offset");
            }
        }
    }

    #[test]
    fn air_only_section_is_empty() {
        let (_r, tables, b) = setup();
        let sec = extract(FINEST, &terrain_gen(&b, 0, 0, None));
        assert!(mesh_of(&sec, &tables).iter().all(|q| q.is_empty()), "an all-air section meshes to nothing");
    }

    #[test]
    fn floating_shelf_emits_a_bottom_and_segment_split_sides() {
        let (_r, tables, b) = setup();
        // Ground at 100 everywhere, plus a floating stone slab in cells [108,112) for x < 48.
        let (stone, dirt, grass) = (b.stone, b.dirt, b.grass);
        let r#gen = FnGen {
            h: |_, _| 100,
            b: move |x: i32, y, _z: i32| {
                if y < 97 {
                    stone
                } else if y < 99 {
                    dirt
                } else if y < 100 {
                    grass
                } else if (108..112).contains(&y) && x < 48 {
                    stone
                } else {
                    AIR
                }
            },
            surf: grass,
            deep: stone,
        };
        let sec = extract(FINEST, &r#gen);
        let mesh = mesh_of(&sec, &tables);
        assert_winds_outward(&mesh);
        assert!(normals_present(&mesh, Normal::NegY), "the floating slab shows its underside");
        // Interior side faces should survive the segment split (not merge into overdraw).
        let interior_side = all_quads(&mesh).any(|(_, _, q)| {
            matches!(q[0].normal(), Normal::PosX | Normal::NegX | Normal::PosZ | Normal::NegZ)
                && q[0].micro() == [0, 0, 0]
        });
        assert!(interior_side, "overhang side faces should segment-split, not vanish");
    }

    #[test]
    fn lod_water_routes_to_opaque_with_no_internal_walls() {
        let (_r, tables, b) = setup();
        // Shore at 40 with water up to 80: a deep water table over sand/stone.
        let sec = extract(FINEST, &terrain_gen(&b, 40, 80, None));
        let mesh = mesh_of(&sec, &tables);
        let is_water_quad = |q: &[MeshVertex]| tables.fluid_surface(BlockId(q[0].layer()));
        let opaque_water = all_quads(&mesh).any(|(_, p, q)| p == Pass::Opaque && is_water_quad(&q));
        assert!(opaque_water, "water surface meshes into the opaque pass");
        assert!(!all_quads(&mesh).any(|(_, _, q)| q[0].is_water()), "LOD water clears the water bit");
        assert!(
            !all_quads(&mesh).any(|(_, p, _)| p == Pass::Blend),
            "no translucent geometry on a LOD section"
        );
        // Water-vs-water is suppressed; all water side faces are borders (micro != 0).
        for (_, _, q) in all_quads(&mesh) {
            if !is_water_quad(&q) {
                continue;
            }
            let side = matches!(
                q[0].normal(),
                Normal::PosX | Normal::NegX | Normal::PosZ | Normal::NegZ
            );
            if side {
                assert_ne!(q[0].micro(), [0, 0, 0], "a wall appeared inside the water body");
            }
        }
    }

    #[test]
    fn every_vertex_is_block_local_and_blocks_tile_without_overlap() {
        let (_r, tables, b) = setup();
        let sec = extract(FINEST, &terrain_gen(&b, 200, 0, Some((300, 320))));
        let mesh = mesh_of(&sec, &tables);
        let mut seen = std::collections::HashSet::new();
        for (origin, data) in mesh.iter().flatten() {
            assert!(seen.insert((origin.x, origin.y, origin.z)), "two blocks share an origin");
            assert!(origin.x % 16 == 0 && origin.y % 16 == 0 && origin.z % 16 == 0, "block origin off-grid");
            assert!(origin.x < SECTION_N as u32 && origin.z < SECTION_N as u32, "block origin outside section XZ");
            for p in Pass::ALL {
                for v in data[p].vertices() {
                    let l = v.local_pos();
                    assert!(l.iter().all(|&c| (0.0..=16.0).contains(&c)), "vertex {l:?} escapes its 16³ block");
                }
            }
        }
    }

    #[test]
    fn a_flat_top_merges_and_a_checkerboard_does_not() {
        let (_r, tables, b) = setup();
        let flat = extract(FINEST, &terrain_gen(&b, 200, 0, None));
        let tops = all_quads(&build_section_mesh(&flat, &tables))
            .filter(|(_, _, q)| q[0].normal() == Normal::PosY)
            .count();
        assert_eq!(tops, BLOCKS_XZ as usize * BLOCKS_XZ as usize, "a uniform flat top merges per block");

        let (grass, dirt) = (b.grass, b.dirt);
        let stone = b.stone;
        let checker = FnGen {
            h: |_, _| 200,
            // Checkerboard varies at cell boundaries (not single-meter bands).
            b: move |x: i32, y, z: i32| {
                if y >= 200 {
                    AIR
                } else if y >= 196 {
                    if (x.div_euclid(CELL) + z.div_euclid(CELL)).rem_euclid(2) == 0 { grass } else { dirt }
                } else {
                    stone
                }
            },
            surf: grass,
            deep: stone,
        };
        let sec = extract(FINEST, &checker);
        let mesh = build_section_mesh(&sec, &tables);
        let top_quads = all_quads(&mesh).filter(|(_, _, q)| q[0].normal() == Normal::PosY).count();
        assert!(top_quads > 100, "checkerboard tops must not merge (got {top_quads})");
    }

    /// THE parity oracle for the fused production path: for every generator
    /// shape, detail level, and edit pattern, `extract_section_mesh` (generator
    /// → dense grid → mesh) must be byte-identical to the reference storage
    /// path (generator → RLE bricks → dense grid → mesh). Any divergence means
    /// the two extraction paths disagree on cell values — a hole or seam.
    #[test]
    fn fused_extract_mesh_matches_the_storage_path_exactly() {
        let (_r, tables, b) = setup();
        let flatten = |m: &[SectionMeshData; 4]| {
            m.iter()
                .flatten()
                .flat_map(|(o, d)| {
                    Pass::ALL.into_iter().map(move |p| {
                        (o.x, o.y, o.z, p as u8, d[p].vertices(), d[p].quad_counts())
                    })
                })
                .collect::<Vec<_>>()
        };

        // Edits inside the finest section footprint, exercising every
        // apply_edits branch: a centre-sample hit (air AND solid), off-centre
        // solid edits contending for one coarse cell, and an out-of-footprint
        // edit that must be ignored identically.
        let cs = crate::world::chunk::CHUNK_SIZE;
        let edit_chunk = crate::coord::ChunkCoord::new(0, 6, 0); // world y 96..112
        let far_chunk = crate::coord::ChunkCoord::new(50, 6, 0);
        let idx = |x: usize, y: usize, z: usize| x + z * cs + y * cs * cs;
        let edits = vec![
            (
                edit_chunk,
                vec![
                    (idx(2, 6, 2), b.stone), // centre of cell (0,?,0) at CELL=4: (2, 96+6=102?, 2)
                    (idx(3, 1, 5), b.dirt),  // off-centre solid
                    (idx(3, 2, 5), AIR),     // off-centre air (must not clear)
                    (idx(2, 10, 2), AIR),    // another candidate centre hit
                    (idx(7, 3, 9), b.sand),
                ],
            ),
            (far_chunk, vec![(idx(1, 1, 1), b.stone)]),
        ];

        use crate::ident::Detail;
        let k = FINEST_DETAIL.0;
        for detail in [FINEST_DETAIL, Detail(k + 2), Detail(k + 4), Detail(k + 7)] {
            let pos = SectionPos { detail, x: 0, z: 0 };
            for (label, r#gen) in [
                ("flat", terrain_gen(&b, 200, 0, None)),
                ("water", terrain_gen(&b, 40, 80, None)),
                ("shelf", terrain_gen(&b, 200, 0, Some((300, 320)))),
            ] {
                for edit_set in [&[][..], &edits[..]] {
                    let stored =
                        Section::extract(pos, &r#gen, edit_set, voxel_engine::Rev::START);
                    let reference = build_section_mesh(&stored, &tables);
                    let fused = extract_section_mesh(pos, &r#gen, edit_set, &tables);
                    assert_eq!(
                        flatten(&reference),
                        flatten(&fused),
                        "fused path diverged: {label} at {detail:?} (edits: {})",
                        !edit_set.is_empty(),
                    );
                }
            }
            // The real terrain generator, off-origin so warps/rivers vary.
            let hills = Terrain::new(&mut BlockRegistry::with_builtins(), 20.0, 0xBEEF);
            let pos = SectionPos { detail, x: 3, z: -2 };
            let stored = Section::extract(pos, &hills, &edits, voxel_engine::Rev::START);
            assert_eq!(
                flatten(&build_section_mesh(&stored, &tables)),
                flatten(&extract_section_mesh(pos, &hills, &edits, &tables)),
                "fused path diverged on Terrain at {detail:?}",
            );
        }
    }

    #[test]
    fn meshing_is_deterministic() {
        let (_r, tables, _b) = setup();
        let r#gen = Terrain::new(&mut BlockRegistry::with_builtins(), 20.0, 0xBEEF);
        let sec = extract(FINEST, &r#gen);
        let a = build_section_mesh(&sec, &tables);
        let b = build_section_mesh(&sec, &tables);
        let flatten = |m: &[SectionMeshData; 4]| {
            m.iter()
                .flatten()
                .flat_map(|(o, d)| {
                    Pass::ALL.into_iter().map(move |p| {
                        (o.x, o.y, o.z, p as u8, d[p].vertices(), d[p].quad_counts())
                    })
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(flatten(&a), flatten(&b), "same section must mesh bit-identically");
    }

    #[test]
    fn each_quadrant_mesh_stays_within_its_xz_bounds() {
        let (_r, tables, b) = setup();
        let sec = extract(FINEST, &terrain_gen(&b, 200, 0, Some((260, 280))));
        let mesh = build_section_mesh(&sec, &tables);
        for (q, quad) in mesh.iter().enumerate() {
            let (lox, loz) = (((q & 1) * QUAD_N) as f32, ((q >> 1) * QUAD_N) as f32);
            for (origin, data) in quad {
                assert_eq!((origin.x, origin.z), (lox as u32, loz as u32), "quadrant {q} origin off its band");
                for p in Pass::ALL {
                    for v in data[p].vertices() {
                        let l = v.local_pos();
                        let (wx, wz) = (origin.x as f32 + l[0], origin.z as f32 + l[2]);
                        assert!((lox - 1.0..=lox + QUAD_N as f32 + 1.0).contains(&wx), "quadrant {q} vertex x {wx} escapes");
                        assert!((loz - 1.0..=loz + QUAD_N as f32 + 1.0).contains(&wz), "quadrant {q} vertex z {wz} escapes");
                    }
                }
            }
        }
    }

    #[test]
    fn coarser_detail_agrees_on_the_exposed_surface() {
        let (_r, tables, b) = setup();
        use crate::ident::Detail;
        let k = FINEST_DETAIL.0;
        for detail in [FINEST_DETAIL, Detail(k + 2), Detail(k + 4), Detail(k + 6)] {
            let pos = SectionPos { detail, x: 0, z: 0 };
            let sec = extract(pos, &terrain_gen(&b, 200, 0, None));
            let mesh = build_section_mesh(&sec, &tables);
            assert!(normals_present(&mesh, Normal::PosY), "detail {detail:?} lost the top surface");
            assert_winds_outward(&mesh);
        }
    }

    /// extract+mesh 16 fixed sections at detail 2, seed 42 — the gauge for the
    /// stride-walk / pooled-output rewrite. Ignored: a timing benchmark, not a
    /// correctness gate. Run with
    /// `cargo test --release far_lod_section_mesh -- --ignored --nocapture`.
    /// 2026-09-09: 9.34 ms/section (median of 3; before stride-walk/pooled-output 9.53).
    #[test]
    #[ignore]
    fn far_lod_section_mesh() {
        let mut registry = BlockRegistry::with_builtins();
        let r#gen = Terrain::new(&mut registry, 20.0, 42);
        let tables = registry.hot_tables();
        let positions: [SectionPos; 16] = std::array::from_fn(|i| SectionPos {
            detail: FINEST_DETAIL,
            x: (i % 4) as i32,
            z: (i / 4) as i32,
        });

        // Warm the worker-local dense grid and the mesh scratch.
        std::hint::black_box(extract_section_mesh(positions[0], &r#gen, &[], &tables));

        let mut times = [0.0f64; 3];
        for t in &mut times {
            let start = std::time::Instant::now();
            for &pos in &positions {
                std::hint::black_box(extract_section_mesh(pos, &r#gen, &[], &tables));
            }
            *t = start.elapsed().as_secs_f64() * 1000.0 / positions.len() as f64;
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "far_lod_section_mesh 16 sections detail=2 seed=42: {:.3} {:.3} {:.3} ms/section (median {:.3})",
            times[0], times[1], times[2], times[1]
        );

        // Fingerprint after the timed loops so a timing run also shows the
        // output did not drift. FNV-1a over origins + decoded vertex fields.
        let mut verts = 0usize;
        let mut h = 0x811c9dc5u32;
        let mix = |h: &mut u32, b: u8| {
            *h ^= b as u32;
            *h = h.wrapping_mul(0x01000193);
        };
        for &pos in &positions {
            let mesh = extract_section_mesh(pos, &r#gen, &[], &tables);
            for quad in &mesh {
                for (origin, data) in quad {
                    for c in origin.to_array() {
                        for b in c.to_le_bytes() {
                            mix(&mut h, b);
                        }
                    }
                    for p in Pass::ALL {
                        for v in data[p].vertices() {
                            verts += 1;
                            for c in v.local_pos() {
                                for b in c.to_bits().to_le_bytes() {
                                    mix(&mut h, b);
                                }
                            }
                            mix(&mut h, v.normal() as u8);
                            for b in v.layer().to_le_bytes() {
                                mix(&mut h, b);
                            }
                            mix(&mut h, (0..=3).find(|&a| v.ao() == Ao::new(a)).expect("ao 0..=3"));
                            for m in v.micro() {
                                mix(&mut h, m as u8);
                            }
                        }
                    }
                }
            }
        }
        println!("far_lod_section_mesh fingerprint verts={verts} fnv={h:#010x}");
    }
}
