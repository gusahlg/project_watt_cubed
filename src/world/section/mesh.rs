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
//! One mesh per section per pass: the engine vertex format holds local coords
//! `0..=16`, so a 32-cell section is packed by a power-of-two coarsen (`shift`)
//! that also covers the occupied Y slab. Greedy merge then runs across that
//! whole packed volume — more vertices than a 16³ split is fine; draw count is
//! the cost. Two producers share one pooled native grid:
//! - [`build_section_mesh`] decodes a stored [`Section`]'s brick stacks — the
//!   reference path, kept as the byte-parity oracle;
//! - [`extract_section_mesh`] samples the generator (and folds edits) straight
//!   into the grid — the production worker path, which never builds the RLE
//!   brick storage only to decode it again.
//!
//! Light is not baked — every vertex takes neutral daylight (full sky, no
//! blocklight) so coarse tiles track day/night.
use std::cell::RefCell;

use voxel_engine::{Ao, Light, MeshVertex, Pass};

use super::super::generation::TerrainGenerator;
use super::super::mesh::{ChunkMeshData, new_chunk_mesh_data};
use super::{ChunkCoord, DOMAIN_H, SECTION_N, SectionPos};
#[cfg(test)]
use super::Section;
use crate::block::registry::{AIR, BlockId, HotTables};
use super::super::mesh::face::{self, covered, vertex_ao, corner_uv};

/// One GPU mesh (all passes) for a whole section. `origin_y` is the native-cell
/// Y of the mesh origin; `shift` extra detail bits pack the section into the
/// `0..=16` vertex range so upload uses `pos.detail + shift`.
pub(in crate::world) struct SectionMeshData {
    pub origin_y: u32,
    pub shift: u8,
    pub data: ChunkMeshData,
}

impl Default for SectionMeshData {
    fn default() -> Self {
        Self {
            origin_y: 0,
            shift: 0,
            data: new_chunk_mesh_data(),
        }
    }
}

impl SectionMeshData {
    fn is_empty(&self) -> bool {
        Pass::ALL.iter().all(|&p| self.data[p].is_empty())
    }
}

/// Cells per mesh-block edge — the 5-bit vertex position range (`0..=16`). Fixed by design.
const BLOCK: i32 = 16;
const SLICE: usize = (BLOCK * BLOCK) as usize;

// Native 32×32×n grid, packed coarse grid, and one mesh scratch per worker.
// Born and dropped on the same thread, so a lock-free thread-local is right.
thread_local! {
    static DENSE_NATIVE: RefCell<Vec<BlockId>> = const { RefCell::new(Vec::new()) };
    static DENSE_COARSE: RefCell<Vec<BlockId>> = const { RefCell::new(Vec::new()) };
    static MESH_SCRATCH: RefCell<ChunkMeshData> = RefCell::new(new_chunk_mesh_data());
}

/// Packed section cells as a dense column-major grid: `nx × nz` columns of
/// `ny` cells. Column-major keeps a column's vertical neighbours adjacent.
struct DenseGrid<'a> {
    cells: &'a [BlockId],
    nx: i32,
    ny: i32,
    nz: i32,
}

impl DenseGrid<'_> {
    #[inline]
    fn at_flat(&self, i: usize) -> BlockId {
        self.cells[i]
    }

    /// Index delta of one step along world axis `axis` (0=X, 1=Y, 2=Z) in the
    /// column-major `nx × nz × ny` layout.
    #[inline]
    fn axis_stride(&self, axis: usize) -> i32 {
        let mut p = [0i32; 3];
        p[axis] = 1;
        (p[0] + p[2] * self.nx) * self.ny + p[1]
    }

    #[inline]
    fn size(&self, axis: usize) -> i32 {
        [self.nx, self.ny, self.nz][axis]
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
/// cull rule's "solid ground"), above-ceiling and outside the section read air
/// (matches the border-overdraw convention).
#[inline]
fn occluder(grid: &DenseGrid<'_>, tables: &HotTables, idx: i32, x: i32, y: i32, z: i32) -> bool {
    if y < 0 {
        return true;
    }
    if y >= grid.ny || x < 0 || x >= grid.nx || z < 0 || z >= grid.nz {
        return false;
    }
    tables.opaque(grid.at_flat(idx as usize))
}

/// Sample a face: cull if covered by neighbor; overdraw section edges as air.
/// Per-corner AO reads the two in-plane occluders plus the diagonal, in the
/// layer the face opens into — same stencil `face_sample` in `world/mesh.rs`
/// uses. `idx` is the cell's flat index; every probe is a stride add off it.
#[inline]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)] // 3×3 stencil: du/dv are both indices and signed offsets
fn face_sample(
    grid: &DenseGrid<'_>,
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
    if y < 0 || y >= grid.ny {
        return None;
    }
    let me = grid.at_flat(idx as usize);
    if me == AIR {
        return None;
    }
    let open = idx + dir.face.step * s_n;
    let mut micro = [0i8; 3];
    let nbr = if dir.face.n_axis == 1 {
        let ny = y + dir.face.step;
        if ny < 0 {
            return None; // below the floor: solid ground, never a silhouette
        } else if ny >= grid.ny {
            AIR // above the ceiling: open sky
        } else {
            grid.at_flat(open as usize)
        }
    } else {
        let n = if dir.face.n_axis == 0 { x } else { z };
        let lim = if dir.face.n_axis == 0 { grid.nx } else { grid.nz };
        if n + dir.face.step < 0 || n + dir.face.step >= lim {
            // Section border: overdraw as air, nudge inward.
            micro = dir.micro;
            AIR
        } else {
            grid.at_flat(open as usize)
        }
    };
    if covered(me, nbr, tables) {
        return None;
    }
    let (ox, oy, oz) = match dir.face.n_axis {
        0 => (x + dir.face.step, y, z),
        1 => (x, y + dir.face.step, z),
        _ => (x, y, z + dir.face.step),
    };
    let mut opaque = [[false; 3]; 3];
    for dv in 0..3 {
        let ev = dv as i32 - 1;
        for du in 0..3 {
            let eu = du as i32 - 1;
            let pidx = open + eu * s_u + ev * s_v;
            opaque[du][dv] = match dir.face.n_axis {
                0 => occluder(grid, tables, pidx, ox, oy + ev, oz + eu),
                1 => occluder(grid, tables, pidx, ox + eu, oy, oz + ev),
                _ => occluder(grid, tables, pidx, ox + eu, oy + ev, oz),
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

/// Greedy-mesh the packed volume: merge adjacent quads with identical properties
/// across the whole section (one mesh, vertex coords `0..=16`).
fn build_volume(grid: &DenseGrid<'_>, tables: &HotTables, out: &mut ChunkMeshData) -> bool {
    let mut mask: [Option<FaceSample>; SLICE] = [None; SLICE];
    let mut emitted = false;

    for dir in &DIRS {
        let (s_n, s_u, s_v) = (
            grid.axis_stride(dir.face.n_axis),
            grid.axis_stride(dir.face.u_axis),
            grid.axis_stride(dir.face.v_axis),
        );
        let n_n = grid.size(dir.face.n_axis);
        let n_u = grid.size(dir.face.u_axis);
        let n_v = grid.size(dir.face.v_axis);
        debug_assert!(n_u * n_v <= BLOCK * BLOCK);
        let corner_uv = corner_uv(&dir.face.corners);

        for n in 0..n_n {
            let mut any = false;
            for v in 0..n_v {
                let row_idx = n * s_n + v * s_v;
                for u in 0..n_u {
                    let idx = row_idx + u * s_u;
                    let (x, y, z) = match dir.face.n_axis {
                        0 => (n, v, u),
                        1 => (u, n, v),
                        _ => (u, v, n),
                    };
                    let cell = face_sample(
                        grid, tables, dir, s_n, s_u, s_v, idx, x, y, z, &corner_uv,
                    );
                    mask[(u + v * n_u) as usize] = cell;
                    any |= cell.is_some();
                }
            }
            if !any {
                continue;
            }
            for v0 in 0..n_v {
                for u0 in 0..n_u {
                    let key = mask[(u0 + v0 * n_u) as usize];
                    let Some(sample) = key else { continue };
                    let mut w = 1;
                    while u0 + w < n_u && mask[(u0 + w + v0 * n_u) as usize] == key {
                        w += 1;
                    }
                    let mut h = 1;
                    'grow: while v0 + h < n_v {
                        for k in 0..w {
                            if mask[(u0 + k + (v0 + h) * n_u) as usize] != key {
                                break 'grow;
                            }
                        }
                        h += 1;
                    }
                    for dv in 0..h {
                        let row = (v0 + dv) * n_u;
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
            tables.render_layer(BlockId(layer)),
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

/// Mesh a whole section as one packed mesh per pass. Section edges overdraw as
/// air with the inward micro-nudge. Deterministic: the same section yields
/// bit-identical output (fixed iteration order, dense-grid sampling, no floats).
///
/// This is the REFERENCE path (stored bricks → dense grid → mesh); production
/// far jobs take [`extract_section_mesh`], which the parity test pins against
/// this one byte for byte.
#[cfg(test)]
pub(in crate::world) fn build_section_mesh(section: &Section, tables: &HotTables) -> SectionMeshData {
    let n_cells = (DOMAIN_H / section.pos().cell_size()) as usize;
    mesh_section_with(n_cells, tables, |dense| fill_from_section(dense, n_cells, section))
}

/// Extract AND mesh a section in one pass: sample the generator (folding the
/// edit overlay) directly into the dense section grid, then mesh it — the
/// production worker path. Skips the whole RLE brick round trip
/// ([`Section::extract`]'s per-column run slicing plus this module's decode),
/// which existed only because the temporary [`Section`] was built and dropped.
pub(in crate::world) fn extract_section_mesh<G: TerrainGenerator + ?Sized>(
    pos: SectionPos,
    r#gen: &G,
    edits: &[(ChunkCoord, Vec<(usize, BlockId)>)],
    tables: &HotTables,
) -> SectionMeshData {
    let cell = pos.cell_size();
    let n_cells = (DOMAIN_H / cell) as usize;
    let ys = super::cell_centers(pos);
    let flat = super::flatten_edits(edits);
    mesh_section_with(n_cells, tables, |dense| {
        for iz in 0..SECTION_N {
            for ix in 0..SECTION_N {
                let (fx, fz) = (pos.min_x() + ix as i32 * cell, pos.min_z() + iz as i32 * cell);
                let (wx, wz) = (fx + cell / 2, fz + cell / 2);
                let column = &mut dense[(ix + iz * SECTION_N) * n_cells..][..n_cells];
                r#gen.lod_column(wx, wz, &ys, column);
                super::apply_edits(column, &flat, fx, fz, cell);
            }
        }
    })
}

/// The one section-mesh driver both producers share: `fill` overwrites this
/// worker's pooled native grid (every cell must be written), then the packer
/// and mesher run over it. Producers differ ONLY in how the grid is filled.
fn mesh_section_with(
    n_cells: usize,
    tables: &HotTables,
    fill: impl FnOnce(&mut [BlockId]),
) -> SectionMeshData {
    DENSE_NATIVE.with_borrow_mut(|native| {
        native.resize(SECTION_N * SECTION_N * n_cells, AIR);
        fill(native);
        mesh_packed(native, n_cells, tables)
    })
}

/// Decode all four brick stacks into the native 32×32 grid (column-major).
#[cfg(test)]
fn fill_from_section(dense: &mut [BlockId], n_cells: usize, section: &Section) {
    for iz in 0..SECTION_N {
        for ix in 0..SECTION_N {
            let column = &mut dense[(ix + iz * SECTION_N) * n_cells..][..n_cells];
            let mut at = 0usize;
            for run in section.column_runs(ix, iz) {
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

/// Smallest extra detail bits so a native `extent` fits in `BLOCK` packed cells.
fn pack_shift(extent: i32) -> u32 {
    let mut shift = 0u32;
    while (extent + (1 << shift) - 1) >> shift > BLOCK {
        shift += 1;
        debug_assert!(shift < 8, "section extent {extent} cannot pack into 16");
    }
    shift
}

/// Highest non-air in the native `step³` cube, preferring opaque at the same Y.
fn pick_coarse(
    native: &[BlockId],
    n_cells: usize,
    x0: i32,
    y0: i32,
    z0: i32,
    step: i32,
    tables: &HotTables,
) -> BlockId {
    let mut best = AIR;
    let mut best_y = -1i32;
    let mut best_opaque = false;
    for dz in 0..step {
        for dx in 0..step {
            for dy in 0..step {
                let x = x0 + dx;
                let y = y0 + dy;
                let z = z0 + dz;
                if x < 0 || z < 0 || y < 0 {
                    continue;
                }
                if x >= SECTION_N as i32 || z >= SECTION_N as i32 || y >= n_cells as i32 {
                    continue;
                }
                let id = native[(x as usize + z as usize * SECTION_N) * n_cells + y as usize];
                if id == AIR {
                    continue;
                }
                let opaque = tables.opaque(id);
                if y > best_y || (y == best_y && opaque && !best_opaque) {
                    best = id;
                    best_y = y;
                    best_opaque = opaque;
                }
            }
        }
    }
    best
}

/// Pack the native 32×n×32 occupancy into ≤16³ and greedy-mesh it once.
fn mesh_packed(native: &[BlockId], n_cells: usize, tables: &HotTables) -> SectionMeshData {
    let ny_n = n_cells as i32;
    let (mut ylo, mut yhi) = (ny_n, 0);
    for col in native.chunks_exact(n_cells) {
        if let Some(first) = col.iter().position(|&id| id != AIR) {
            let last = col.iter().rposition(|&id| id != AIR).expect("some cell is non-air");
            ylo = ylo.min(first as i32);
            yhi = yhi.max(last as i32 + 1);
        }
    }
    if yhi <= ylo {
        return SectionMeshData::default();
    }

    let shift = pack_shift(SECTION_N as i32).max(pack_shift(yhi - ylo));
    let step = 1i32 << shift;
    let y0 = ylo / step * step;
    let y1 = (yhi + step - 1) / step * step;
    let nx = (SECTION_N as i32 + step - 1) / step;
    let ny = (y1 - y0) / step;
    let nz = nx;
    debug_assert!(nx <= BLOCK && ny <= BLOCK && nz <= BLOCK);

    DENSE_COARSE.with_borrow_mut(|coarse| {
        coarse.resize((nx * ny * nz) as usize, AIR);
        for z in 0..nz {
            for x in 0..nx {
                for y in 0..ny {
                    coarse[(x + z * nx) as usize * ny as usize + y as usize] = pick_coarse(
                        native,
                        n_cells,
                        x * step,
                        y0 + y * step,
                        z * step,
                        step,
                        tables,
                    );
                }
            }
        }
        MESH_SCRATCH.with_borrow_mut(|scratch| {
            for (_, m) in scratch.iter_mut() {
                m.clear();
            }
            let grid = DenseGrid {
                cells: coarse,
                nx,
                ny,
                nz,
            };
            if build_volume(&grid, tables, scratch) {
                SectionMeshData {
                    origin_y: y0 as u32,
                    shift: shift as u8,
                    data: std::mem::replace(scratch, new_chunk_mesh_data()),
                }
            } else {
                SectionMeshData::default()
            }
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::BlockRegistry;
    use crate::world::generation::{Terrain, TerrainGenerator};
    use voxel_engine::{Normal, Pass};
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
        let id = |n: &str| r.id_by_label(n).unwrap();
        let blocks = Blocks {
            grass: id("organic+soil"),
            dirt: id("clay+soil"),
            stone: id("rock"),
            sand: id("sand"),
            water: id("water"),
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

    fn mesh_of(section: &Section, tables: &HotTables) -> SectionMeshData {
        build_section_mesh(section, tables)
    }

    fn all_quads(mesh: &SectionMeshData) -> impl Iterator<Item = (Pass, [MeshVertex; 4])> {
        let data = &mesh.data;
        Pass::ALL.into_iter().flat_map(move |p| {
            data[p]
                .vertices()
                .chunks_exact(4)
                .map(|q| (p, [q[0], q[1], q[2], q[3]]))
                .collect::<Vec<_>>()
        })
    }

    fn normals_present(mesh: &SectionMeshData, want: Normal) -> bool {
        all_quads(mesh).any(|(_, q)| q[0].normal() == want)
    }

    /// Every quad must wind counter-clockwise from outside (engine back-face-culls otherwise).
    fn assert_winds_outward(mesh: &SectionMeshData) {
        let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let cross = |a: [f32; 3], b: [f32; 3]| {
            [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
        };
        for (_, q) in all_quads(mesh) {
            let p: Vec<[f32; 3]> = q.iter().map(|v| v.local_pos()).collect::<Vec<_>>();
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
        for (_, q) in all_quads(&mesh) {
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
        assert!(mesh_of(&sec, &tables).is_empty(), "an all-air section meshes to nothing");
    }

    #[test]
    fn floating_shelf_emits_a_bottom_and_segment_split_sides() {
        let (_r, tables, b) = setup();
        // Ground at 100, plus a floating stone slab thick enough to survive
        // the packed-cell coarsen (a 4 m wafer would vanish into the cube below).
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
                } else if (200..240).contains(&y) && x < 48 {
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
        let interior_side = all_quads(&mesh).any(|(_, q)| {
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
        // The vertex layer is the render DESCRIPTOR of the material, not its block id.
        let water_layer = tables.render_layer(b.water);
        let is_water_quad = |q: &[MeshVertex]| q[0].layer() == water_layer;
        let opaque_water = all_quads(&mesh).any(|(p, q)| p == Pass::Opaque && is_water_quad(&q));
        assert!(opaque_water, "water surface meshes into the opaque pass");
        assert!(!all_quads(&mesh).any(|(_, q)| q[0].is_water()), "LOD water clears the water bit");
        assert!(
            !all_quads(&mesh).any(|(p, _)| p == Pass::Blend),
            "no translucent geometry on a LOD section"
        );
        // Water-vs-water is suppressed; all water side faces are borders (micro != 0).
        for (_, q) in all_quads(&mesh) {
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
    fn every_vertex_is_packed_local() {
        let (_r, tables, b) = setup();
        let sec = extract(FINEST, &terrain_gen(&b, 200, 0, Some((300, 320))));
        let mesh = mesh_of(&sec, &tables);
        assert!(!mesh.is_empty(), "occupied section emits a mesh");
        for p in Pass::ALL {
            for v in mesh.data[p].vertices() {
                let l = v.local_pos();
                assert!(l.iter().all(|&c| (0.0..=16.0).contains(&c)), "vertex {l:?} escapes 0..=16");
            }
        }
    }

    #[test]
    fn one_mesh_per_section() {
        let (_r, tables, b) = setup();
        let mesh = mesh_of(&extract(FINEST, &terrain_gen(&b, 200, 0, Some((300, 320)))), &tables);
        let passes = Pass::ALL.iter().filter(|&&p| !mesh.data[p].is_empty()).count();
        assert!(passes >= 1, "occupied section has geometry");
        assert!(passes <= Pass::COUNT, "one mesh per pass, not per block");
    }

    #[test]
    fn a_flat_top_merges_and_a_checkerboard_does_not() {
        let (_r, tables, b) = setup();
        let flat = extract(FINEST, &terrain_gen(&b, 200, 0, None));
        let tops = all_quads(&build_section_mesh(&flat, &tables))
            .filter(|(_, q)| q[0].normal() == Normal::PosY)
            .count();
        assert_eq!(tops, 1, "a uniform flat top merges across the whole section");

        let (grass, dirt) = (b.grass, b.dirt);
        let stone = b.stone;
        let checker = FnGen {
            h: |_, _| 200,
            // Checkerboard varies at cell boundaries (not single-meter bands).
            b: move |x: i32, y, z: i32| {
                if y >= 200 {
                    AIR
                } else if y >= 196 {
                    // Packed cells are 2+ native cells; checkerboard at that scale.
                    if (x.div_euclid(CELL * 8) + z.div_euclid(CELL * 8)).rem_euclid(2) == 0 {
                        grass
                    } else {
                        dirt
                    }
                } else {
                    stone
                }
            },
            surf: grass,
            deep: stone,
        };
        let sec = extract(FINEST, &checker);
        let mesh = build_section_mesh(&sec, &tables);
        let top_quads = all_quads(&mesh).filter(|(_, q)| q[0].normal() == Normal::PosY).count();
        assert!(top_quads > 4, "checkerboard tops must not merge (got {top_quads})");
    }

    /// THE parity oracle for the fused production path: for every generator
    /// shape, detail level, and edit pattern, `extract_section_mesh` (generator
    /// → dense grid → mesh) must be byte-identical to the reference storage
    /// path (generator → RLE bricks → dense grid → mesh). Any divergence means
    /// the two extraction paths disagree on cell values — a hole or seam.
    #[test]
    fn fused_extract_mesh_matches_the_storage_path_exactly() {
        let (_r, tables, b) = setup();
        let flatten = |m: &SectionMeshData| {
            Pass::ALL
                .into_iter()
                .map(|p| {
                    (
                        m.origin_y,
                        m.shift,
                        p as u8,
                        m.data[p].vertices(),
                        m.data[p].quad_counts(),
                    )
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
        let flatten = |m: &SectionMeshData| {
            Pass::ALL
                .into_iter()
                .map(|p| {
                    (
                        m.origin_y,
                        m.shift,
                        p as u8,
                        m.data[p].vertices(),
                        m.data[p].quad_counts(),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(flatten(&a), flatten(&b), "same section must mesh bit-identically");
    }

    #[test]
    fn packed_mesh_stays_within_the_vertex_cube() {
        let (_r, tables, b) = setup();
        let sec = extract(FINEST, &terrain_gen(&b, 200, 0, Some((260, 280))));
        let mesh = build_section_mesh(&sec, &tables);
        for p in Pass::ALL {
            for v in mesh.data[p].vertices() {
                let l = v.local_pos();
                assert!(
                    l.iter().all(|&c| (0.0..=16.0).contains(&c)),
                    "packed vertex {l:?} escapes 0..=16"
                );
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
    /// one-mesh-per-section pack. Ignored: a timing benchmark, not a
    /// correctness gate. Run with
    /// `cargo test --release far_lod_section_mesh -- --ignored --nocapture`.
    /// 2026-09-10: 2.440 ms/section (median of 3); fingerprint verts=9048 fnv=0xa42303c7.
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
            for b in mesh.origin_y.to_le_bytes() {
                mix(&mut h, b);
            }
            mix(&mut h, mesh.shift);
            for p in Pass::ALL {
                for v in mesh.data[p].vertices() {
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
        println!("far_lod_section_mesh fingerprint verts={verts} fnv={h:#010x}");
    }
}
