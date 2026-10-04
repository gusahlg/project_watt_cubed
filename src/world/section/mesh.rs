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
//! The engine vertex format holds local coords `0..=16`, so a 32-cell section is
//! packed by a power-of-two coarsen (`shift`, at least 1 for the 32 columns).
//! Relief taller than 16 packed cells stacks up to [`MAX_SLABS`] slabs — one mesh
//! per slab per pass, all at that shift — instead of coarsening every axis until
//! a whole mountain fits one cube: steep sections keep their horizontal detail.
//! Greedy merge runs across each slab, reading its neighbours above and below
//! so slab seams emit no faces. Two producers share one pooled native grid:
//! - [`build_section_mesh`] decodes a stored [`Section`]'s brick stacks — the
//!   reference path, kept as the byte-parity oracle;
//! - [`extract_section_mesh`] samples the generator (and folds edits) straight
//!   into the grid — the production worker path, which never builds the RLE
//!   brick storage only to decode it again.
//!
//! Light is not baked — every vertex takes neutral daylight (full sky, no
//! blocklight) so coarse tiles track day/night.
use std::cell::RefCell;

use voxel_engine::{Ao, Light, MeshVertex, Normal, Pass};

use crate::coord::Face;
use crate::space::FaceFrame;

use super::super::generation::TerrainGenerator;
use super::super::mesh::{ChunkMeshData, new_chunk_mesh_data};
use super::{ChunkCoord, FINEST_DETAIL, SECTION_N, SectionPos};
#[cfg(test)]
use super::{DOMAIN_H, Section};
use crate::block::registry::{AIR, BlockId, HotTables};
use super::super::mesh::face::{self, covered, vertex_ao, corner_uv};

/// A section's GPU meshes: one per non-empty slab (all passes), stacked bottom-up. `shift` extra
/// detail bits pack the section into the `0..=16` vertex range, so upload uses `pos.detail +
/// shift` for every slab.
#[derive(Default)]
pub(in crate::world) struct SectionMeshData {
    pub shift: u8,
    pub slabs: Vec<SlabMesh>,
    /// World altitude of native cell 0. `0` on the legacy `[0, 512)` window.
    pub altitude_floor: i32,
}

/// One packed slab: up to 16 packed cells tall, its origin at native-cell Y `origin_y`.
pub(in crate::world) struct SlabMesh {
    pub origin_y: u32,
    pub data: ChunkMeshData,
}

impl SectionMeshData {
    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.slabs.iter().all(|s| voxel_engine::Pass::ALL.iter().all(|&p| s.data[p].is_empty()))
    }

    /// Vertex bytes over every slab and pass.
    pub(in crate::world) fn vertex_bytes(&self) -> usize {
        self.slabs.iter().map(|s| voxel_engine::Pass::ALL.iter().map(|&p| s.data[p].vertex_bytes()).sum::<usize>()).sum()
    }
}

/// Cells per mesh-block edge — the 5-bit vertex position range (`0..=16`). Fixed by design.
const BLOCK: i32 = 16;
const SLICE: usize = (BLOCK * BLOCK) as usize;
/// Most stacked slabs one section may use before it coarsens instead: 64 packed cells of relief
/// at the least shift (256 native cells at the finest ring) — every mountain in one tall stack.
const MAX_SLABS: i32 = 4;

// Native 32×32×n grid, packed coarse grid, and one mesh scratch per worker.
// Born and dropped on the same thread, so a lock-free thread-local is right.
thread_local! {
    static DENSE_NATIVE: RefCell<Vec<BlockId>> = const { RefCell::new(Vec::new()) };
    static DENSE_COARSE: RefCell<Vec<BlockId>> = const { RefCell::new(Vec::new()) };
    static MESH_SCRATCH: RefCell<ChunkMeshData> = RefCell::new(new_chunk_mesh_data());
}

/// Packed section cells as a dense column-major grid: `nx × nz` columns of
/// `ny` cells. Column-major keeps a column's vertical neighbours adjacent.
/// Meshing emits only the slab `y_lo..y_hi`; the cells around it are context
/// (covering faces and AO across the seam).
struct DenseGrid<'a> {
    cells: &'a [BlockId],
    nx: i32,
    ny: i32,
    nz: i32,
    y_lo: i32,
    y_hi: i32,
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

    /// Extent of the meshed window along `axis` (the slab along Y).
    #[inline]
    fn size(&self, axis: usize) -> i32 {
        [self.nx, self.y_hi - self.y_lo, self.nz][axis]
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

/// Greedy-mesh one slab of the packed volume: merge adjacent quads with identical
/// properties across it (one mesh, vertex coords `0..=16` relative to the slab).
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

        // Y has stride 1 in the column-major layout: the slab window starts `y_lo` cells in.
        let (base, y_lo) = (grid.y_lo, grid.y_lo);
        for n in 0..n_n {
            let mut any = false;
            for v in 0..n_v {
                let row_idx = base + n * s_n + v * s_v;
                for u in 0..n_u {
                    let idx = row_idx + u * s_u;
                    let (x, y, z) = match dir.face.n_axis {
                        0 => (n, v + y_lo, u),
                        1 => (u, n + y_lo, v),
                        _ => (u, v + y_lo, n),
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
    let pass = tables.layer[layer as usize];
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

/// Mesh a whole section as packed slab meshes. Section edges overdraw as
/// air with the inward micro-nudge. Deterministic: the same section yields
/// bit-identical output (fixed iteration order, dense-grid sampling, no floats).
///
/// This is the REFERENCE path (stored bricks → dense grid → mesh); production
/// far jobs take [`extract_section_mesh`], which the parity test pins against
/// this one byte for byte.
#[cfg(test)]
pub(in crate::world) fn build_section_mesh(section: &Section, tables: &HotTables, floor: Option<i32>) -> SectionMeshData {
    let n_cells = (DOMAIN_H / section.pos().cell_size()) as usize;
    mesh_section_with(n_cells, tables, floor, |dense| fill_from_section(dense, n_cells, section))
}

/// Lowest native cell a section's border walls must reach so they meet the neighbour.
/// Finest rings take the min outside surface. Coarser rings start at the window floor:
/// their edge samples sit a cell apart and miss the valley between them.
pub(in crate::world) fn ring_floor<G: TerrainGenerator + ?Sized>(pos: SectionPos, r#gen: &G) -> i32 {
    if pos.detail > FINEST_DETAIL {
        return 0;
    }
    let cell = pos.cell_size();
    let n = SECTION_N as i32;
    let top = |ix: i32, iz: i32| {
        let (wx, wz) = (pos.min_x() + ix * cell + cell / 2, pos.min_z() + iz * cell + cell / 2);
        (r#gen.height(wx, wz) - 1 - super::LOD_FLOOR_Y).div_euclid(cell)
    };
    let mut floor = i32::MAX;
    for i in -1..=n {
        floor = floor.min(top(i, -1)).min(top(i, n)).min(top(-1, i)).min(top(n, i));
    }
    floor.max(0)
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
    let Some((alo, ahi)) = section_window(pos, r#gen) else {
        return SectionMeshData::default();
    };
    let cell = pos.cell_size();
    let n_cells = ((ahi - alo) / cell) as usize;
    let half = cell / 2;
    let ys = super::column_ys(pos.detail, alo, n_cells as i32, cell);
    let flat = super::flatten_edits(edits);
    let remapped;
    let used: &[_] = if pos.face == Face::PosY {
        &flat
    } else {
        remapped = face_edits(&flat, pos.face);
        &remapped
    };
    // Home +Y keeps the legacy column and the `[0, 512]` floor. A chart is +Y in storage
    // but its surface sits far outside that window, so it takes the face sampler.
    let chart = pos.body >= super::CHART_BODY_BASE;
    let floor = if pos.face == Face::PosY && !chart { ring_floor(pos, r#gen) } else { ring_floor_face(pos, r#gen, alo) };
    let mut mesh = mesh_section_with(n_cells, tables, Some(floor), |dense| {
        for iz in 0..SECTION_N {
            for ix in 0..SECTION_N {
                let (fx, fz) = (pos.min_x() + ix as i32 * cell, pos.min_z() + iz as i32 * cell);
                let (u, v) = (fx + half, fz + half);
                let column = &mut dense[(ix + iz * SECTION_N) * n_cells..][..n_cells];
                if pos.face == Face::PosY && !chart {
                    r#gen.lod_column(u, v, &ys, column);
                } else {
                    r#gen.lod_column_face(pos.body, pos.face, u, v, &ys, column);
                }
                super::apply_edits(column, used, fx, fz, cell, alo);
            }
        }
    });
    mesh.altitude_floor = alo;
    if pos.face != Face::PosY {
        orient_section(&mut mesh, pos.face);
    }
    mesh
}

/// `[lo, hi)` the extractor samples. `None` when the square holds no surface.
fn section_window<G: TerrainGenerator + ?Sized>(pos: SectionPos, r#gen: &G) -> Option<(i32, i32)> {
    let (lo, hi) = r#gen.surface_bounds(pos.body, pos.face, pos.min_x(), pos.min_z(), pos.span())?;
    Some(super::sample_window(lo, hi, pos.cell_size()))
}

/// World edits rewritten as face-local `(u, a, v)`, sorted the way [`super::apply_edits`] requires.
fn face_edits(flat: &[(i32, i32, i32, BlockId)], face: Face) -> Vec<(i32, i32, i32, BlockId)> {
    let frame = FaceFrame::new(face);
    let mut out: Vec<_> = flat
        .iter()
        .map(|&(x, y, z, id)| {
            let (u, a, v) = frame.cell_to_local((x, y, z));
            (u, a, v, id)
        })
        .collect();
    out.sort_unstable_by_key(|&(u, a, v, _)| (a, u, v));
    out
}

/// [`ring_floor`] measured from `alo` along `pos.face` (surface altitude, not world Y).
fn ring_floor_face<G: TerrainGenerator + ?Sized>(pos: SectionPos, r#gen: &G, alo: i32) -> i32 {
    if pos.detail > FINEST_DETAIL {
        return 0;
    }
    let cell = pos.cell_size();
    let n = SECTION_N as i32;
    let top = |ix: i32, iz: i32| {
        let (u, v) = (pos.min_x() + ix * cell + cell / 2, pos.min_z() + iz * cell + cell / 2);
        let h = r#gen.surface(pos.face, u, v);
        if h == i32::MIN {
            i32::MAX
        } else {
            (h - 1 - alo).div_euclid(cell)
        }
    };
    let mut floor = i32::MAX;
    for i in -1..=n {
        floor = floor.min(top(i, -1)).min(top(i, n)).min(top(-1, i)).min(top(n, i));
    }
    floor.max(0)
}

/// Rotate a Y-up section mesh into `face`. PosY is a no-op, so home upload bytes do not move.
/// Positions stay in `0..=16` (permutation about the block centre). Winding is kept: det = +1.
pub(in crate::world) fn orient_section(mesh: &mut SectionMeshData, face: Face) {
    if face == Face::PosY {
        return;
    }
    for slab in &mut mesh.slabs {
        orient_mesh(&mut slab.data, face);
    }
}

fn orient_mesh(data: &mut ChunkMeshData, face: Face) {
    let frame = FaceFrame::new(face);
    for pass in Pass::ALL {
        let src = data[pass].vertices();
        if src.is_empty() {
            continue;
        }
        let quads: Vec<[MeshVertex; 4]> = src
            .chunks_exact(4)
            .map(|q| [map_vert(q[0], frame), map_vert(q[1], frame), map_vert(q[2], frame), map_vert(q[3], frame)])
            .collect();
        data[pass].clear();
        for corners in quads {
            data[pass].quad(corners);
        }
    }
}

fn map_vert(v: MeshVertex, frame: FaceFrame) -> MeshVertex {
    let p = v.local_pos();
    let (x, y, z) = frame.cell_to_world((p[0] as i32 - 8, p[1] as i32 - 8, p[2] as i32 - 8));
    let d = v.normal().direction();
    let (nx, ny, nz) = frame.cell_to_world((d[0] as i32, d[1] as i32, d[2] as i32));
    let m = v.micro();
    let (mx, my, mz) = frame.cell_to_world((m[0] as i32, m[1] as i32, m[2] as i32));
    MeshVertex::new(
        [(x + 8) as u8, (y + 8) as u8, (z + 8) as u8],
        normal_from([nx, ny, nz]),
        v.layer(),
        v.ao(),
        v.light(),
        v.is_water(),
    )
    .with_micro([mx as i8, my as i8, mz as i8])
}

fn normal_from(d: [i32; 3]) -> Normal {
    match d {
        [1, 0, 0] => Normal::PosX,
        [-1, 0, 0] => Normal::NegX,
        [0, 1, 0] => Normal::PosY,
        [0, -1, 0] => Normal::NegY,
        [0, 0, 1] => Normal::PosZ,
        [0, 0, -1] => Normal::NegZ,
        _ => Normal::PosY,
    }
}

/// Eight physical corners of one chart slab's packed block, anchor at the floor of corner 0
/// (bit 0 = +x, bit 1 = +y, bit 2 = +z). `extent` is the packed block edge `16·2^(detail+shift)`.
/// Vertices occupy only the section's real span of that block; the cage is this wide so the
/// shader's `local/16` map lands the geometry on the storage square.
pub(in crate::world) fn chart_slab_corners(
    atlas: &crate::space::atlas::Atlas,
    patch: crate::space::atlas::Patch,
    x0: i32,
    y0: i32,
    z0: i32,
    extent: i32,
) -> Option<(voxel_engine::IVec3, [voxel_engine::Vec3; 8])> {
    let extent = extent as f64;
    let corners: [voxel_engine::DVec3; 8] = std::array::from_fn(|c| {
        let d = [(c & 1) as f64 * extent, ((c >> 1) & 1) as f64 * extent, ((c >> 2) & 1) as f64 * extent];
        atlas.embed_storage(patch, voxel_engine::DVec3::new(x0 as f64 + d[0], y0 as f64 + d[1], z0 as f64 + d[2]))
    });
    let a = corners[0].floor();
    let fits = |v: f64| v.is_finite() && (i32::MIN as f64..=i32::MAX as f64).contains(&v);
    if !fits(a.x) || !fits(a.y) || !fits(a.z) {
        return None;
    }
    let anchor = voxel_engine::IVec3::new(a.x as i32, a.y as i32, a.z as i32);
    Some((anchor, corners.map(|c| (c - a).as_vec3())))
}

/// Eight warped corners of one cube-face slab. The packed block is the axis-aligned world box
/// [`SectionState::slab_placement`] would have used; each corner is that point through the warp.
/// Anchor is the floor of corner 0 (bit 0 = +x, bit 1 = +y, bit 2 = +z).
pub(in crate::world) fn warp_slab_corners(
    atlas: &crate::space::atlas::Atlas,
    face: Face,
    x0: i32,
    y0: i32,
    z0: i32,
    extent: i32,
) -> Option<(voxel_engine::IVec3, [voxel_engine::Vec3; 8])> {
    let warp = atlas.warp.as_ref()?;
    let half = extent / 2;
    let (cx, cy, cz) = FaceFrame::new(face).cell_to_world((x0 + half, y0 + half, z0 + half));
    let origin = [cx - half, cy - half, cz - half];
    let extent_f = extent as f64;
    let corners: [voxel_engine::DVec3; 8] = std::array::from_fn(|c| {
        let p = voxel_engine::DVec3::new(
            origin[0] as f64 + ((c & 1) as f64) * extent_f,
            origin[1] as f64 + (((c >> 1) & 1) as f64) * extent_f,
            origin[2] as f64 + (((c >> 2) & 1) as f64) * extent_f,
        );
        warp.apply(p)
    });
    let a = corners[0].floor();
    let fits = |v: f64| v.is_finite() && (i32::MIN as f64..=i32::MAX as f64).contains(&v);
    if !fits(a.x) || !fits(a.y) || !fits(a.z) {
        return None;
    }
    let anchor = voxel_engine::IVec3::new(a.x as i32, a.y as i32, a.z as i32);
    Some((anchor, corners.map(|c| (c - a).as_vec3())))
}

/// The one section-mesh driver both producers share: `fill` overwrites this
/// worker's pooled native grid (every cell must be written), then the packer
/// and mesher run over it. Producers differ ONLY in how the grid is filled.
fn mesh_section_with(
    n_cells: usize,
    tables: &HotTables,
    floor: Option<i32>,
    fill: impl FnOnce(&mut [BlockId]),
) -> SectionMeshData {
    DENSE_NATIVE.with_borrow_mut(|native| {
        native.resize(SECTION_N * SECTION_N * n_cells, AIR);
        fill(native);
        mesh_packed(native, n_cells, tables, floor)
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

/// Pack the native 32×n×32 occupancy into ≤16-cell columns and greedy-mesh it, one
/// mesh per stacked 16-cell slab.
///
/// The packed volume starts one cell below the lowest SURFACE (the lowest column top here, or
/// `floor`, the lowest top just outside the section, whichever is lower) — not at the bottom
/// of the domain: everything below is solid ground in every column and never shows, so only
/// the relief inside the section costs slabs. The shift is the least that packs the 32 columns
/// and fits the relief in [`MAX_SLABS`] slabs.
fn mesh_packed(native: &[BlockId], n_cells: usize, tables: &HotTables, floor: Option<i32>) -> SectionMeshData {
    let ny_n = n_cells as i32;
    let (mut min_top, mut yhi) = (ny_n, 0);
    for col in native.chunks_exact(n_cells) {
        if let Some(last) = col.iter().rposition(|&id| id != AIR) {
            min_top = min_top.min(last as i32);
            yhi = yhi.max(last as i32 + 1);
        }
    }
    if yhi == 0 {
        return SectionMeshData::default();
    }
    let ylo = (min_top.min(floor.unwrap_or(min_top)) - 1).max(0);

    let mut shift = pack_shift(SECTION_N as i32);
    let (step, y0, ny) = loop {
        let step = 1i32 << shift;
        let y0 = ylo / step * step;
        let ny = ((yhi + step - 1) / step * step - y0) / step;
        if ny <= BLOCK * MAX_SLABS {
            break (step, y0, ny);
        }
        shift += 1;
    };
    let nx = (SECTION_N as i32 + step - 1) / step;
    let nz = nx;
    debug_assert!(nx <= BLOCK && nz <= BLOCK);

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
            let mut slabs = Vec::new();
            for y_lo in (0..ny).step_by(BLOCK as usize) {
                for (_, m) in scratch.iter_mut() {
                    m.clear();
                }
                let grid = DenseGrid { cells: coarse, nx, ny, nz, y_lo, y_hi: (y_lo + BLOCK).min(ny) };
                if build_volume(&grid, tables, scratch) {
                    slabs.push(SlabMesh {
                        origin_y: (y0 + y_lo * step) as u32,
                        data: std::mem::replace(scratch, new_chunk_mesh_data()),
                    });
                }
            }
            SectionMeshData { shift: if slabs.is_empty() { 0 } else { shift as u8 }, slabs, altitude_floor: 0 }
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::BlockRegistry;
    use crate::world::generation::TerrainGenerator;
    use crate::world::terrain::Terrain;
    use crate::ident::Detail;
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
        crate::world::terrain::Materials::intern(&mut r);
        let id = |n: &str| r.id_by_label(n).unwrap();
        let blocks = Blocks {
            grass: id("grass"),
            dirt: id("soil"),
            stone: id("rock"),
            sand: id("sand"),
            water: id("ice"),
        };
        let tables = r.hot_tables();
        (r, tables, blocks)
    }

    const FINEST: SectionPos = SectionPos { body: 0, face: Face::PosY, detail: FINEST_DETAIL, x: 0, z: 0 };
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
        build_section_mesh(section, tables, None)
    }

    fn all_quads(mesh: &SectionMeshData) -> impl Iterator<Item = (Pass, [MeshVertex; 4])> {
        mesh.slabs.iter().flat_map(|slab| {
            Pass::ALL.into_iter().flat_map(move |p| {
                slab.data[p]
                    .vertices()
                    .chunks_exact(4)
                    .map(|q| (p, [q[0], q[1], q[2], q[3]]))
                    .collect::<Vec<_>>()
            })
        })
    }

    /// One pass of one slab with its placement: `(origin_y, shift, pass, vertices, quad counts)`.
    type FlatPass = (u32, u8, u8, Vec<MeshVertex>, [u32; 6]);

    /// Every slab's every pass, with its placement, for byte-parity comparisons.
    fn flatten(m: &SectionMeshData) -> Vec<FlatPass> {
        m.slabs
            .iter()
            .flat_map(|slab| {
                Pass::ALL.into_iter().map(move |p| {
                    (
                        slab.origin_y,
                        m.shift,
                        p as u8,
                        slab.data[p].vertices(),
                        slab.data[p].quad_counts(),
                    )
                })
            })
            .collect()
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
    fn every_vertex_is_packed_local() {
        let (_r, tables, b) = setup();
        let sec = extract(FINEST, &terrain_gen(&b, 200, 0, Some((300, 320))));
        let mesh = mesh_of(&sec, &tables);
        assert!(!mesh.is_empty(), "occupied section emits a mesh");
        for (_, q) in all_quads(&mesh) {
            for v in q {
                let l = v.local_pos();
                assert!(l.iter().all(|&c| (0.0..=16.0).contains(&c)), "vertex {l:?} escapes 0..=16");
            }
        }
    }

    /// Relief splits into stacked slabs at the least shift instead of coarsening every axis: a
    /// shelf 100 native cells above the ground keeps the horizontal shift of flat ground, and the
    /// slabs stack contiguously with no seam faces between them.
    #[test]
    fn tall_relief_stacks_slabs_at_the_least_shift() {
        let (_r, tables, b) = setup();
        let flat = mesh_of(&extract(FINEST, &terrain_gen(&b, 200, 0, None)), &tables);
        assert_eq!(flat.slabs.len(), 1, "flat ground is one slab");
        // A peak over part of the section: columns 0..12 rise 300 m above the rest.
        let (stone, grass) = (b.stone, b.grass);
        let peak = FnGen {
            h: |x: i32, _| if x < 48 { 500 } else { 200 },
            b: move |x: i32, y, _| if y < if x < 48 { 500 } else { 200 } { stone } else { AIR },
            surf: grass,
            deep: stone,
        };
        let tall = mesh_of(&extract(FINEST, &peak), &tables);
        assert_eq!(tall.shift, flat.shift, "relief must not coarsen the horizontal packing");
        assert!(tall.slabs.len() > 1, "a tall section stacks slabs");
        assert!(tall.slabs.len() <= MAX_SLABS as usize);
        let step = 1u32 << tall.shift;
        for w in tall.slabs.windows(2) {
            assert!(w[1].origin_y > w[0].origin_y && (w[1].origin_y - w[0].origin_y) % (BLOCK as u32 * step) == 0);
        }
        // The peak's columns cross every seam: the only horizontal faces are its top and the
        // low ground's top, none at the seams.
        let mut levels = std::collections::BTreeSet::new();
        for slab in &tall.slabs {
            for p in Pass::ALL {
                for v in slab.data[p].vertices() {
                    assert_ne!(v.normal(), Normal::NegY, "an underside inside solid rock");
                    if v.normal() == Normal::PosY {
                        levels.insert(slab.origin_y + v.local_pos()[1] as u32 * step);
                    }
                }
            }
        }
        assert_eq!(levels.len(), 2, "tops only at the ground and the peak, none at seams: {levels:?}");
    }

    #[test]
    fn a_flat_top_merges_and_a_checkerboard_does_not() {
        let (_r, tables, b) = setup();
        let flat = extract(FINEST, &terrain_gen(&b, 200, 0, None));
        let tops = all_quads(&build_section_mesh(&flat, &tables, None))
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
        let mesh = build_section_mesh(&sec, &tables, None);
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
            let pos = SectionPos { body: 0, face: Face::PosY, detail, x: 0, z: 0 };
            for (label, r#gen) in [
                ("flat", terrain_gen(&b, 200, 0, None)),
                ("water", terrain_gen(&b, 40, 80, None)),
                ("shelf", terrain_gen(&b, 200, 0, Some((300, 320)))),
            ] {
                for edit_set in [&[][..], &edits[..]] {
                    let stored =
                        Section::extract(pos, &r#gen, edit_set, voxel_engine::Rev::START);
                    let reference = build_section_mesh(&stored, &tables, Some(ring_floor(pos, &r#gen)));
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
            let hills = Terrain::new(&mut BlockRegistry::with_builtins(), 0xBEEF);
            let pos = SectionPos { body: 0, face: Face::PosY, detail, x: 3, z: -2 };
            // The start world is charted, so this square is not a cube face: both paths are empty.
            // The reference extractor still samples `[0, 512]`, which no longer holds a home face.
            let floor = hills
                .surface_bounds(pos.body, pos.face, pos.min_x(), pos.min_z(), pos.span())
                .map(|_| ring_floor(pos, &hills));
            let stored = Section::extract(pos, &hills, &edits, voxel_engine::Rev::START);
            assert_eq!(
                flatten(&build_section_mesh(&stored, &tables, floor)),
                flatten(&extract_section_mesh(pos, &hills, &edits, &tables)),
                "fused path diverged on Terrain at {detail:?}",
            );
        }
    }

    #[test]
    fn meshing_is_deterministic() {
        let (_r, tables, _b) = setup();
        let r#gen = Terrain::new(&mut BlockRegistry::with_builtins(), 0xBEEF);
        let sec = extract(FINEST, &r#gen);
        let a = build_section_mesh(&sec, &tables, None);
        let b = build_section_mesh(&sec, &tables, None);
        assert_eq!(flatten(&a), flatten(&b), "same section must mesh bit-identically");
    }

    #[test]
    fn packed_mesh_stays_within_the_vertex_cube() {
        let (_r, tables, b) = setup();
        let sec = extract(FINEST, &terrain_gen(&b, 200, 0, Some((260, 280))));
        let mesh = build_section_mesh(&sec, &tables, None);
        for (_, q) in all_quads(&mesh) {
            for v in q {
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
            let pos = SectionPos { body: 0, face: Face::PosY, detail, x: 0, z: 0 };
            let sec = extract(pos, &terrain_gen(&b, 200, 0, None));
            let mesh = build_section_mesh(&sec, &tables, None);
            assert!(normals_present(&mesh, Normal::PosY), "detail {detail:?} lost the top surface");
            assert_winds_outward(&mesh);
        }
    }

    /// Meshing in the Y-up frame and rotating the packed vertices must match meshing
    /// the same voxels after the face permutation. Interior cells only: the floor-solid
    /// and border-micro rules are not part of the rotation.
    #[test]
    fn far_face_rotation_matches_meshing_rotated_voxels() {
        let (_r, tables, b) = setup();
        let n = 16i32;
        let mut cells = vec![AIR; (n * n * n) as usize];
        let at = |x: i32, y: i32, z: i32| ((x + z * n) * n + y) as usize;
        // A vertical bar (so a greedy merge is real) and one offset cell. Both sit
        // in 5..=11, so every signed permutation stays off the grid border.
        let occupied = [(6, 8, 7), (6, 9, 7), (10, 7, 9)];
        for &(x, y, z) in &occupied {
            cells[at(x, y, z)] = b.stone;
        }
        let mesh_of = |grid: &[BlockId]| {
            let mut out = new_chunk_mesh_data();
            let g = DenseGrid { cells: grid, nx: n, ny: n, nz: n, y_lo: 0, y_hi: n };
            assert!(build_volume(&g, &tables, &mut out), "the feature emits no faces");
            out
        };
        let key = |v: MeshVertex| {
            let p = v.local_pos();
            let ao = (0u8..=3).find(|&a| v.ao() == Ao::new(a)).expect("ao");
            (p[0].to_bits(), p[1].to_bits(), p[2].to_bits(), v.normal() as u8, v.layer(), ao, v.micro(), v.is_water())
        };
        let canon = |data: &ChunkMeshData| {
            let mut quads = Vec::new();
            for pass in Pass::ALL {
                let verts = data[pass].vertices();
                for q in verts.chunks_exact(4) {
                    let mut c = [q[0], q[1], q[2], q[3]];
                    let mut best = 0;
                    for i in 1..4 {
                        if key(c[i]) < key(c[best]) {
                            best = i;
                        }
                    }
                    c.rotate_left(best);
                    quads.push(c);
                }
            }
            quads.sort_by_key(|q| q.map(key));
            quads
        };
        let rotate_cell = |x: i32, y: i32, z: i32, frame: FaceFrame| {
            let mut min = (i32::MAX, i32::MAX, i32::MAX);
            for dx in 0..2 {
                for dy in 0..2 {
                    for dz in 0..2 {
                        let (wx, wy, wz) = frame.cell_to_world((x + dx - 8, y + dy - 8, z + dz - 8));
                        min.0 = min.0.min(wx + 8);
                        min.1 = min.1.min(wy + 8);
                        min.2 = min.2.min(wz + 8);
                    }
                }
            }
            min
        };
        for face in Face::ALL {
            let mut turned = mesh_of(&cells);
            orient_mesh(&mut turned, face);
            let frame = FaceFrame::new(face);
            let mut oracle = vec![AIR; cells.len()];
            for &(x, y, z) in &occupied {
                let (ox, oy, oz) = rotate_cell(x, y, z, frame);
                assert!((0..n).contains(&ox) && (0..n).contains(&oy) && (0..n).contains(&oz), "{face:?} cell left the block");
                oracle[at(ox, oy, oz)] = b.stone;
            }
            let direct = mesh_of(&oracle);
            assert_eq!(canon(&turned), canon(&direct), "{face:?}");
        }
    }

    /// A placed section: tops and the vertical quads that sit on its four edges.
    /// World positions ignore the inward micro-nudge (a uniform hairline, not a slot).
    struct Tile {
        pos: SectionPos,
        x0: i32,
        x1: i32,
        z0: i32,
        z1: i32,
        tops: Vec<[i32; 5]>,
        walls: Vec<[i32; 6]>,
    }

    fn place_tile(pos: SectionPos, mesh: &SectionMeshData) -> Tile {
        let detail = Detail(pos.detail.0.saturating_add(mesh.shift as i8));
        let scale = 1i32 << detail.0;
        let (x0, z0) = (pos.min_x(), pos.min_z());
        let (x1, z1) = (x0 + pos.span(), z0 + pos.span());
        let mut tops = Vec::new();
        let mut walls = Vec::new();
        for slab in &mesh.slabs {
            let place = crate::world::SectionState::slab_placement(pos, slab.origin_y, detail, mesh.altitude_floor);
            let w = |local: f32, origin: i32| origin + (local as i32) * scale;
            for pass in Pass::ALL {
                for q in slab.data[pass].vertices().chunks_exact(4) {
                    let n = q[0].normal();
                    let mut min = [i32::MAX; 3];
                    let mut max = [i32::MIN; 3];
                    for v in q {
                        let p = v.local_pos();
                        let g = [w(p[0], place.block.x), w(p[1], place.block.y), w(p[2], place.block.z)];
                        for a in 0..3 {
                            min[a] = min[a].min(g[a]);
                            max[a] = max[a].max(g[a]);
                        }
                    }
                    if n == Normal::PosY && min[1] == max[1] {
                        tops.push([min[0], max[0], min[2], max[2], min[1]]);
                    } else if (n == Normal::PosX || n == Normal::NegX) && min[0] == max[0]
                        && (min[0] == x0 || min[0] == x1)
                    {
                        walls.push([0, min[0], min[2], max[2], min[1], max[1]]);
                    } else if (n == Normal::PosZ || n == Normal::NegZ) && min[2] == max[2]
                        && (min[2] == z0 || min[2] == z1)
                    {
                        walls.push([2, min[2], min[0], max[0], min[1], max[1]]);
                    }
                }
            }
        }
        Tile { pos, x0, x1, z0, z1, tops, walls }
    }

    fn top_at(tile: &Tile, x: i32, z: i32) -> Option<i32> {
        tile.tops.iter().filter(|t| x >= t[0] && x <= t[1] && z >= t[2] && z <= t[3]).map(|t| t[4]).max()
    }

    /// Uncovered blocks between `lo` and `hi` after the walls that contain `along` are unioned.
    fn open_span(tile_a: &Tile, tile_b: &Tile, axis: i32, edge: i32, along: i32, lo: i32, hi: i32) -> i32 {
        if hi <= lo {
            return 0;
        }
        let mut iv = Vec::new();
        for tile in [tile_a, tile_b] {
            for w in &tile.walls {
                if w[0] == axis && w[1] == edge && along >= w[2] && along <= w[3] {
                    let (a, b) = (w[4].max(lo), w[5].min(hi));
                    if b > a {
                        iv.push((a, b));
                    }
                }
            }
        }
        iv.sort_unstable();
        let mut cursor = lo;
        let mut worst = 0;
        for (a, b) in iv {
            if a > cursor {
                worst = worst.max(a - cursor);
            }
            cursor = cursor.max(b);
        }
        if hi > cursor {
            worst = worst.max(hi - cursor);
        }
        worst
    }

    struct SeamGap {
        gap: i32,
        station: i32,
        top_lo: i32,
        top_hi: i32,
        da: i8,
        db: i8,
        a: (i32, i32),
        b: (i32, i32),
    }

    /// Vertical slots where two edge-adjacent tiles' tops differ and no wall covers the step.
    fn seam_gaps(a: &Tile, b: &Tile) -> Vec<SeamGap> {
        let mut out = Vec::new();
        let mut push = |axis: i32, edge: i32, lo: i32, hi: i32, inset_a: i32, inset_b: i32| {
            if hi - lo < 4 {
                return;
            }
            let fine = a.pos.cell_size().min(b.pos.cell_size());
            let mut s = lo + fine / 2;
            while s < hi {
                let (pa, pb) = if axis == 0 {
                    ((edge + inset_a, s), (edge + inset_b, s))
                } else {
                    ((s, edge + inset_a), (s, edge + inset_b))
                };
                let (ta, tb) = (top_at(a, pa.0, pa.1), top_at(b, pb.0, pb.1));
                if let (Some(ta), Some(tb)) = (ta, tb) {
                    let (lo_y, hi_y) = (ta.min(tb), ta.max(tb));
                    let gap = open_span(a, b, axis, edge, s, lo_y, hi_y);
                    if gap > 0 {
                        out.push(SeamGap {
                            gap,
                            station: s,
                            top_lo: lo_y,
                            top_hi: hi_y,
                            da: a.pos.detail.0,
                            db: b.pos.detail.0,
                            a: (a.pos.x, a.pos.z),
                            b: (b.pos.x, b.pos.z),
                        });
                    }
                } else if ta.is_some() || tb.is_some() {
                    out.push(SeamGap {
                        gap: i32::MAX / 4,
                        station: s,
                        top_lo: ta.unwrap_or(-1),
                        top_hi: tb.unwrap_or(-1),
                        da: a.pos.detail.0,
                        db: b.pos.detail.0,
                        a: (a.pos.x, a.pos.z),
                        b: (b.pos.x, b.pos.z),
                    });
                }
                s += fine;
            }
        };
        if a.x1 == b.x0 {
            let lo = a.z0.max(b.z0);
            let hi = a.z1.min(b.z1);
            push(0, a.x1, lo, hi, -a.pos.cell_size() / 2, b.pos.cell_size() / 2);
        } else if b.x1 == a.x0 {
            let lo = a.z0.max(b.z0);
            let hi = a.z1.min(b.z1);
            push(0, a.x0, lo, hi, a.pos.cell_size() / 2, -b.pos.cell_size() / 2);
        }
        if a.z1 == b.z0 {
            let lo = a.x0.max(b.x0);
            let hi = a.x1.min(b.x1);
            push(2, a.z1, lo, hi, -a.pos.cell_size() / 2, b.pos.cell_size() / 2);
        } else if b.z1 == a.z0 {
            let lo = a.x0.max(b.x0);
            let hi = a.x1.min(b.x1);
            push(2, a.z0, lo, hi, a.pos.cell_size() / 2, -b.pos.cell_size() / 2);
        }
        out
    }

    /// A detail-4 section beside a detail-2 neighbour whose valley sits between the coarse
    /// outside samples. The coarse wall has to reach that valley or the sky shows through.
    #[test]
    fn far_seam_coarser_wall_reaches_a_finer_valley() {
        let (_r, tables, b) = setup();
        let stone = b.stone;
        fn valley(x: i32, z: i32) -> bool {
            // Wider than a finest packed cell (8) and between the coarse centres at z = 8 and 24.
            (512..528).contains(&x) && (16..24).contains(&z)
        }
        let terra = FnGen {
            h: move |x, z| if valley(x, z) { 40 } else { 200 },
            b: move |x, y, z| if y < if valley(x, z) { 40 } else { 200 } { stone } else { AIR },
            surf: b.grass,
            deep: stone,
        };
        let coarse = SectionPos { body: 0, face: Face::PosY, detail: Detail(FINEST_DETAIL.0 + 2), x: 0, z: 0 };
        let fine = SectionPos { body: 0, face: Face::PosY, detail: FINEST_DETAIL, x: 4, z: 0 };
        assert_eq!(ring_floor(coarse, &terra), 0, "a coarse skirt starts at the window floor");
        assert!(ring_floor(fine, &terra) > 0, "the finest ring still trims to the outside surface");
        let ct = place_tile(coarse, &extract_section_mesh(coarse, &terra, &[], &tables));
        let ft = place_tile(fine, &extract_section_mesh(fine, &terra, &[], &tables));
        let gaps = seam_gaps(&ct, &ft);
        let worst = gaps.iter().map(|g| g.gap).max().unwrap_or(0);
        assert!(
            gaps.is_empty(),
            "detail {:?} vs {:?} left {worst} blocks open ({} stations, first station {} tops {}..{})",
            coarse.detail,
            fine.detail,
            gaps.len(),
            gaps.first().map(|g| g.station).unwrap_or(0),
            gaps.first().map(|g| g.top_lo).unwrap_or(0),
            gaps.first().map(|g| g.top_hi).unwrap_or(0),
        );
    }

    /// Far-field closure from altitude over a twin cube's +Y (the start world is charted;
    /// its sections are the chart frontier, not these face sections). The eye sits 8 blocks
    /// off that face's centre column.
    #[test]
    fn far_seam_altitude_frontier_borders_are_closed() {
        use crate::coord::ChunkCoord;
        use crate::render_config::RenderConfig;
        use crate::world::generation::WorldgenKind;
        use crate::world::heightmip::BakeExtent;
        use crate::world::quadtree;
        use crate::world::{World, DEFAULT_SEED};
        use std::collections::BTreeMap;

        let mut world = World::with_kind(DEFAULT_SEED, RenderConfig::default(), WorldgenKind::Diffusion, false);
        let twin = world
            .terrain()
            .cosmos()
            .expect("cosmos")
            .bodies()
            .iter()
            .copied()
            .find(|b| b.kind == crate::world::terrain::cosmos::Kind::Twin)
            .expect("a twin");
        let (au, av) = (
            i32::try_from(twin.centre[0]).expect("twin x") + 8,
            i32::try_from(twin.centre[2]).expect("twin z") + 8,
        );
        let body = twin.id;
        let colors = world.registry.color_snapshot();
        let extent = BakeExtent::new(world.section_pyramid.outer_m() as i32, world.section_pyramid.coarsest());
        let bake_at = std::time::Instant::now();
        world.section_mip = Some(crate::world::heightmip::HeightMip::bake_at(
            world.terrain(),
            &colors,
            extent,
            au,
            av,
            Face::PosY,
            body,
        ));
        world.section_mip_anchor = Some((body, Face::PosY, au, av));
        let bake_ms = bake_at.elapsed().as_secs_f64() * 1000.0;
        let outer = world.section_pyramid.outer_m();
        let clip = world.lod_clip();
        let full = world.view.coverage();
        let ground = world.terrain().height(au, av);

        let mut reports = Vec::new();
        for lift in [1500.0_f64, 3000.0] {
            let eye_y = ground as f64 + lift;
            world.section_eye_y = eye_y;
            let center = ChunkCoord::new(au.div_euclid(16), (eye_y as i32).div_euclid(16), av.div_euclid(16));
            let desired = world.desired_sections(center);
            let cut = quadtree::resolve_covering(&desired, Detail(9), &|_| true);
            let drawn: Vec<SectionPos> = cut.iter().map(|(p, _)| *p).collect();
            let tables = world.registry.hot_tables();
            let mut tiles = Vec::with_capacity(drawn.len());
            let mesh_at = std::time::Instant::now();
            for pos in &drawn {
                let mesh = extract_section_mesh(*pos, world.terrain(), &[], &tables);
                tiles.push(place_tile(*pos, &mesh));
            }
            let mesh_ms = mesh_at.elapsed().as_secs_f64() * 1000.0;
            let per = if drawn.is_empty() { 0.0 } else { mesh_ms / drawn.len() as f64 };

            let mut by = BTreeMap::new();
            for p in &drawn {
                *by.entry(p.detail.0).or_insert(0usize) += 1;
            }
            let min_span = drawn.iter().map(|p| p.span()).min().unwrap_or(1);
            let limit = outer - min_span as f32;
            let step = (min_span / 2).max(1);
            let (ex, ez) = (au, av);
            let r = limit.floor() as i32;
            let mut cover_gaps = 0i32;
            let mut cover_n = 0i32;
            let mut cover_at = Vec::new();
            let mut z = ez - r;
            while z <= ez + r {
                let mut x = ex - r;
                while x <= ex + r {
                    let dx = (x - ex) as f32;
                    let dz = (z - ez) as f32;
                    if dx.hypot(dz) <= limit {
                        cover_n += 1;
                        let hit = desired.iter().any(|p| {
                            let (x0, z0) = (p.min_x(), p.min_z());
                            x >= x0 && x < x0 + p.span() && z >= z0 && z < z0 + p.span()
                        });
                        if !hit {
                            cover_gaps += 1;
                            if cover_at.len() < 4 {
                                cover_at.push((x, z));
                            }
                        }
                    }
                    x += step;
                }
                z += step;
            }

            let cam_ground = eye_y - ground as f64;
            let cam_y = eye_y as f32;
            let eye_x = (center.x * 16 + 8) as f32;
            let eye_z = (center.z * 16 + 8) as f32;
            let culled = drawn.iter().filter(|p| {
                let (x0, x1) = (p.min_x() as f32 - eye_x, (p.min_x() + p.span()) as f32 - eye_x);
                let (lo, hi) = world
                    .terrain()
                    .surface_bounds(p.body, p.face, p.min_x(), p.min_z(), p.span())
                    .unwrap_or((0, 512));
                let (wlo, whi) = super::super::sample_window(lo, hi, p.cell_size());
                let (y0, y1) = (wlo as f32 - cam_y, whi as f32 - cam_y);
                let (z0, z1) = (p.min_z() as f32 - eye_z, (p.min_z() + p.span()) as f32 - eye_z);
                x0 > -clip.half.x && x1 < clip.half.x
                    && y0 > -clip.half.y && y1 < clip.half.y
                    && z0 > -clip.half.z && z1 < clip.half.z
            }).count();

            let mut gaps: Vec<SeamGap> = Vec::new();
            let mut seams = 0usize;
            let mut cross = 0usize;
            for i in 0..tiles.len() {
                for j in (i + 1)..tiles.len() {
                    let found = seam_gaps(&tiles[i], &tiles[j]);
                    if tiles[i].x1 == tiles[j].x0 || tiles[j].x1 == tiles[i].x0 || tiles[i].z1 == tiles[j].z0 || tiles[j].z1 == tiles[i].z0
                    {
                        let z_over = tiles[i].z0.max(tiles[j].z0) < tiles[i].z1.min(tiles[j].z1);
                        let x_over = tiles[i].x0.max(tiles[j].x0) < tiles[i].x1.min(tiles[j].x1);
                        let touch_x = (tiles[i].x1 == tiles[j].x0 || tiles[j].x1 == tiles[i].x0) && z_over;
                        let touch_z = (tiles[i].z1 == tiles[j].z0 || tiles[j].z1 == tiles[i].z0) && x_over;
                        if touch_x || touch_z {
                            seams += 1;
                            if tiles[i].pos.detail != tiles[j].pos.detail {
                                cross += 1;
                            }
                        }
                    }
                    gaps.extend(found);
                }
            }
            let missing = gaps.iter().filter(|g| g.gap > 1_000_000).count();
            let vertical: Vec<&SeamGap> = gaps.iter().filter(|g| g.gap <= 1_000_000).collect();
            let worst = vertical.iter().copied().max_by_key(|g| g.gap);
            let same = vertical.iter().filter(|g| g.da == g.db).count();
            let mut bare = 0i32;
            let mut bare_n = 0i32;
            let mut origin_top = None;
            for t in &tiles {
                let cell = t.pos.cell_size();
                let mut x = t.x0 + cell / 2;
                while x < t.x1 {
                    let mut z = t.z0 + cell / 2;
                    while z < t.z1 {
                        bare_n += 1;
                        if top_at(t, x, z).is_none() {
                            bare += 1;
                        }
                        z += cell;
                    }
                    x += cell;
                }
                if t.x0 <= 128 && 128 < t.x1 && t.z0 <= 128 && 128 < t.z1 {
                    origin_top = top_at(t, t.x0 + cell / 2, t.z0 + cell / 2);
                }
            }
            let floor_note = if let Some(g) = worst {
                let side = tiles.iter().find(|t| t.pos.detail.0 == g.da && t.pos.x == g.a.0 && t.pos.z == g.a.1);
                let other = tiles.iter().find(|t| t.pos.detail.0 == g.db && t.pos.x == g.b.0 && t.pos.z == g.b.1);
                match (side, other) {
                    (Some(a), Some(b)) => format!(
                        " ring_floor world Y {} and {}",
                        mesh_floor_y(a.pos, world.terrain()),
                        mesh_floor_y(b.pos, world.terrain()),
                    ),
                    _ => String::new(),
                }
            } else {
                String::new()
            };
            let worst_s = match worst {
                Some(g) => format!(
                    "worst gap {} blocks at station {} tops {}..{} details {}/{} sections ({},{})-({},{}){}",
                    g.gap, g.station, g.top_lo, g.top_hi, g.da, g.db, g.a.0, g.a.1, g.b.0, g.b.1, floor_note
                ),
                None => "worst gap 0".to_string(),
            };
            let dy = (eye_y - 512.0).max(0.0);
            let (desired_n, drawn_n, vert_n) = (desired.len(), drawn.len(), vertical.len());
            reports.push(format!(
                "eye_y {eye_y}: desired {desired_n} drawn {drawn_n} by {by:?} seams {seams} cross-detail {cross} \
                 vertical gaps {vert_n} (same-detail {same}) missing-top {missing} {worst_s}; \
                 bare columns {bare}/{bare_n} origin-top {origin_top:?}; coverage gaps {cover_gaps}/{cover_n} \
                 step {step} limit {limit:.0} examples {cover_at:?}; mesh {mesh_ms:.1} ms ({per:.2} ms/section); \
                 dy {dy:.0} outer {outer:.0} ground {ground} cam-ground {cam_ground:.0}; lod clip half ({cx:.0},{cy:.0},{cz:.0}) \
                 full half ({fx:.0},{fy:.0},{fz:.0}) culled {culled}",
                cx = clip.half.x,
                cy = clip.half.y,
                cz = clip.half.z,
                fx = full.half.x,
                fy = full.half.y,
                fz = full.half.z,
            ));
        }
        let report = format!("mip bake {bake_ms:.1} ms; {}", reports.join(" || "));
        println!("{report}");
        let closed = reports.iter().all(|r| {
            r.contains("vertical gaps 0 ")
                && r.contains("missing-top 0 ")
                && r.contains("bare columns 0/")
                && r.contains("coverage gaps 0/")
                && r.contains("culled 0")
        });
        assert!(closed, "{report}");
    }

    fn mesh_floor_y<G: TerrainGenerator + ?Sized>(pos: SectionPos, terra: &G) -> i32 {
        ring_floor(pos, terra) * pos.cell_size()
    }

    /// Storage column of a direction from the start world's centre, the embedded surface, and the
    /// same point moved `above` blocks out along the local up (the radial). The eye is the one
    /// [`crate::world::World::chart_eye`] would stream.
    fn home_altitude_eye(
        world: &crate::world::World,
        dir: voxel_engine::DVec3,
        above: f64,
    ) -> (crate::coord::ChunkCoord, voxel_engine::DVec3, i32) {
        use crate::space::atlas::Patch;
        use crate::space::chart::{self, Map};
        let centre = world.terrain().cosmos().expect("cosmos").home().centre_f();
        let atlas = world
            .terrain()
            .atlases()
            .iter()
            .find(|a| (a.centre - centre).length() < 1.0)
            .expect("the start world is charted")
            .clone();
        let dir = dir.normalize();
        let face = Face::from_dominant(dir);
        let (tu, nn, tv) = chart::basis(face);
        let (xi, eta) = Map::Equiangular.inverse(voxel_engine::DVec3::new(dir.dot(tu), dir.dot(nn), dir.dot(tv)));
        let n = atlas.bands[0].n;
        let step = 2.0 / n as f64;
        let (i, j) = (((xi + 1.0) / step).floor() as i64, ((eta + 1.0) / step).floor() as i64);
        assert!((0..n).contains(&i) && (0..n).contains(&j), "({i},{j}) leaves the {face:?} chart");
        let patch = Patch::Shell { band: 0, face };
        let (origin, _) = atlas.storage_box(patch);
        let stored = atlas.storage(patch, [i, 0, j]);
        let (sx, sz) = (stored[0] as i32, stored[2] as i32);
        let ground = world.terrain().surface(Face::PosY, sx, sz);
        assert_ne!(ground, i32::MIN, "{face:?} column has no surface");
        let local_y = ground as f64 - origin[1] as f64;
        let surf = atlas.embed(patch, voxel_engine::DVec3::new(i as f64 + 0.5, local_y, j as f64 + 0.5));
        let up = (surf - atlas.centre).normalize();
        let eye = surf + up * above;
        let storage = world.chart_eye(eye).unwrap_or_else(|| panic!("no chart eye at +{above} on {face:?}"));
        let cs = 16.0;
        let centre_chunk = crate::coord::ChunkCoord::new(
            (storage.x / cs).floor() as i32,
            (storage.y / cs).floor() as i32,
            (storage.z / cs).floor() as i32,
        );
        (centre_chunk, storage, ground)
    }

    /// Drawn-tile bare columns, vertical seam gaps, and the near-box samples under the eye.
    /// `deep` counts samples at least one span-32 section in from the box edge.
    struct ChartFrontier {
        bare: i32,
        gaps: usize,
        missing: usize,
        under: i32,
        under_n: i32,
        deep: i32,
        deep_n: i32,
        sections: usize,
        covered: bool,
        hash: u32,
    }

    fn frontier_counts(world: &mut crate::world::World, center: crate::coord::ChunkCoord) -> ChartFrontier {
        use crate::world::quadtree;
        world.adopt_fold(center);
        let desired = world.desired_sections(center);
        let cut = quadtree::resolve_covering(&desired, Detail(9), &|_| true);
        let drawn: Vec<SectionPos> = cut.iter().map(|(p, _)| *p).collect();
        let tables = world.registry.hot_tables();
        let mut tiles = Vec::with_capacity(drawn.len());
        let mut hash = 0x811c9dc5u32;
        let mix = |h: &mut u32, b: u8| {
            *h ^= b as u32;
            *h = h.wrapping_mul(0x01000193);
        };
        let mut order: Vec<SectionPos> = drawn.clone();
        order.sort_by_key(|p| (p.detail, p.x, p.z));
        for pos in &order {
            let mesh = extract_section_mesh(*pos, world.terrain(), &[], &tables);
            mix(&mut hash, mesh.shift);
            for b in mesh.altitude_floor.to_le_bytes() {
                mix(&mut hash, b);
            }
            for b in pos.detail.0.to_le_bytes() {
                mix(&mut hash, b);
            }
            for b in pos.x.to_le_bytes() {
                mix(&mut hash, b);
            }
            for b in pos.z.to_le_bytes() {
                mix(&mut hash, b);
            }
            for slab in &mesh.slabs {
                for b in slab.origin_y.to_le_bytes() {
                    mix(&mut hash, b);
                }
                for p in Pass::ALL {
                    for v in slab.data[p].vertices() {
                        for c in v.local_pos() {
                            for b in c.to_bits().to_le_bytes() {
                                mix(&mut hash, b);
                            }
                        }
                        mix(&mut hash, v.normal() as u8);
                        for b in v.layer().to_le_bytes() {
                            mix(&mut hash, b);
                        }
                    }
                }
            }
            tiles.push(place_tile(*pos, &mesh));
        }
        let mut bare = 0i32;
        for t in &tiles {
            let cell = t.pos.cell_size();
            let mut x = t.x0 + cell / 2;
            while x < t.x1 {
                let mut z = t.z0 + cell / 2;
                while z < t.z1 {
                    if top_at(t, x, z).is_none() {
                        bare += 1;
                    }
                    z += cell;
                }
                x += cell;
            }
        }
        let mut gaps: Vec<SeamGap> = Vec::new();
        for i in 0..tiles.len() {
            for j in (i + 1)..tiles.len() {
                gaps.extend(seam_gaps(&tiles[i], &tiles[j]));
            }
        }
        let missing = gaps.iter().filter(|g| g.gap > 1_000_000).count();
        let vertical = gaps.iter().filter(|g| g.gap <= 1_000_000).count();
        let cs = 16i32;
        let h = world.view.horizontal;
        let (x0, x1) = ((center.x - h) * cs, (center.x + h + 1) * cs);
        let (z0, z1) = ((center.z - h) * cs, (center.z + h + 1) * cs);
        let mut under = 0i32;
        let mut under_n = 0i32;
        let mut deep = 0i32;
        let mut deep_n = 0i32;
        // A detail-0 tile still crossing the box edge overlaps it by less than its span.
        let border = 32i32;
        let mut x = x0 + cs / 2;
        while x < x1 {
            let mut z = z0 + cs / 2;
            while z < z1 {
                under_n += 1;
                let hit = tiles.iter().any(|t| top_at(t, x, z).is_some());
                if !hit {
                    under += 1;
                }
                if x - x0 >= border && x1 - x > border && z - z0 >= border && z1 - z > border {
                    deep_n += 1;
                    if !hit {
                        deep += 1;
                    }
                }
                z += cs;
            }
            x += cs;
        }
        let eye = ((center.x * cs + cs / 2), (center.z * cs + cs / 2));
        let covered = desired.iter().any(|p| {
            let (a, b) = (p.min_x(), p.min_z());
            eye.0 >= a && eye.0 < a + p.span() && eye.1 >= b && eye.1 < b + p.span()
        });
        ChartFrontier {
            bare,
            gaps: vertical,
            missing,
            under,
            under_n,
            deep,
            deep_n,
            sections: desired.len(),
            covered,
            hash,
        }
    }

    fn assert_chart_closed(name: &str, f: &ChartFrontier) {
        assert!(f.under_n > 0, "{name}: the near box was not sampled");
        assert_eq!(f.bare, 0, "{name}: bare columns");
        assert_eq!(f.gaps, 0, "{name}: vertical seam gaps");
        assert_eq!(f.missing, 0, "{name}: missing tops");
        assert_eq!(f.under, 0, "{name}: under-eye holes {}/{}", f.under, f.under_n);
        assert!(f.covered, "{name}: no section covers the eye column");
    }

    /// Start-world chart far field from altitude: the square under the eye is drawn, its columns
    /// have tops, and the seams around it are closed. Spawn ground level keeps the punch on the
    /// interior of the near box. The border, within one span-32 section, is drawn: that tile is
    /// what covers the sliver outside the box. Bytes re-pinned to `0xca4adead` for that border
    /// (seed 42, diffusion, default view).
    #[test]
    fn far_chart_altitude_frontier_is_closed() {
        use crate::render_config::RenderConfig;
        use crate::world::generation::WorldgenKind;
        use crate::world::World;
        use voxel_engine::DVec3;

        let mut world = World::with_kind(42, RenderConfig::default(), WorldgenKind::Diffusion, false);
        let (center, storage, _) = home_altitude_eye(&world, DVec3::new(0.0, 1.0, 0.0), 0.0);
        world.section_eye_y = storage.y;
        let spawn = frontier_counts(&mut world, center);
        assert_eq!(spawn.hash, 0xca4adead, "spawn ground-level far-field bytes changed");
        assert_eq!(spawn.bare, 0, "spawn ground-level bare columns");
        assert_eq!(spawn.gaps, 0, "spawn ground-level seam gaps");
        assert_eq!(spawn.missing, 0, "spawn ground-level missing tops");
        assert!(spawn.deep_n > 0, "spawn interior was not sampled");
        assert_eq!(
            spawn.deep, spawn.deep_n,
            "spawn interior must keep the near-box punch ({}/{}, border under {}/{}, sections {})",
            spawn.deep, spawn.deep_n, spawn.under, spawn.under_n, spawn.sections
        );
        assert!(!spawn.covered, "spawn ground level must keep the near-box punch at the eye");

        let sites = [("plus-y", DVec3::new(0.0, 1.0, 0.0)), ("highland", DVec3::new(1.0, 0.9, 0.8))];
        for (name, dir) in sites {
            for above in [300.0_f64, 1_500.0, 3_000.0] {
                let (center, storage, _) = home_altitude_eye(&world, dir, above);
                world.section_eye_y = storage.y;
                let got = frontier_counts(&mut world, center);
                assert_chart_closed(&format!("{name} +{above}"), &got);
            }
        }
    }

    /// extract+mesh 16 fixed sections at detail 2, seed 42 — the gauge for the
    /// slab pack. Ignored: a timing benchmark, not a
    /// correctness gate. Run with
    /// `cargo test --release far_lod_section_mesh -- --ignored --nocapture`.
    /// 2026-09-10: 2.440 ms/section (median of 3); fingerprint verts=9048 fnv=0xa42303c7.
    /// 2026-10-02 (InfiniteDiffusion v3, stacked slabs): 4.026 ms/section; verts=15056 fnv=0xa2f08c4a.
    #[test]
    #[ignore]
    fn far_lod_section_mesh() {
        let mut registry = BlockRegistry::with_builtins();
        let r#gen = Terrain::new(&mut registry, 42);
        let tables = registry.hot_tables();
        let positions: [SectionPos; 16] = std::array::from_fn(|i| SectionPos { body: 0, face: Face::PosY,
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
            mix(&mut h, mesh.shift);
            for (slab, p) in mesh.slabs.iter().flat_map(|s| Pass::ALL.into_iter().map(move |p| (s, p))) {
                for b in slab.origin_y.to_le_bytes() {
                    mix(&mut h, b);
                }
                for v in slab.data[p].vertices() {
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
