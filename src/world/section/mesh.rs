//! Stack mesher: turns a [`Section`]'s brick stacks straight into GPU-ready
//! [`MeshData`] — no intermediate voxel grid.
//!
//! Each solid run emits up to six faces. Vertical faces split where column heights differ
//! (so overhangs render correctly, unlike a heightmap-only approach). Opacity rules: opaque
//! blocks hide faces; translucent (water) does not cover solid; two translucent blocks of
//! the same type hide their shared edge (no internal walls).
//!
//! Section borders (edge columns) overdraw as air with an inward micro-offset to prevent
//! z-fighting with adjacent sections at different detail levels.
//!
//! Output: a 2×2 grid of 16³-cell blocks per section, only for the vertical range occupied
//! by solid geometry (sky/deep space cost nothing). Per-corner AO samples neighbour-column
//! occupancy within the quadrant (quadrant-border corners see no occluder, so read
//! unoccluded); light is not baked — every vertex takes neutral daylight (full sky, no
//! blocklight) so coarse tiles track day/night.
use glam::UVec3;
use voxel_engine::{Ao, Light, MeshVertex, Normal, Pass};

use super::super::mesh::{ChunkMeshData, new_chunk_mesh_data};
use super::{BrickStack, DOMAIN_H, SECTION_N, Section};
use crate::block::registry::{AIR, BlockId, HotTables};

/// One block's mesh plus its origin in cells within the section. Only non-empty blocks appear.
pub(in crate::world) type SectionMeshData = Vec<(UVec3, ChunkMeshData)>;

/// Cells per mesh-block edge — the 5-bit vertex position range (`0..=16`). Fixed by design.
const BLOCK: i32 = 16;
const QUAD_N: usize = SECTION_N / 2;
#[cfg(test)]
const BLOCKS_XZ: i32 = SECTION_N as i32 / BLOCK;
const SLICE: usize = (BLOCK * BLOCK) as usize;

/// One solid-or-air run of a column expressed in CELL coordinates (`[lo, hi)`,
/// bottom-up), with its block. Air runs are kept so a neighbour lookup over the
/// whole stack is a plain scan.
#[derive(Clone, Copy)]
struct CellRun {
    lo: i32,
    hi: i32,
    block: BlockId,
}

/// Merge key: block, micro, and AO must all match — an AO gradient must never
/// merge into a flat quad (mirrors the chunk mesher's `FaceSample`).
#[derive(Clone, Copy, PartialEq, Eq)]
struct FaceSample {
    block: BlockId,
    micro: [i8; 3],
    ao: [u8; 4],
}

/// Per-vertex AO level `0..=3` (`3` = unoccluded) from its three occluders;
/// two touching sides fully occlude the corner. Same model as the chunk mesher.
fn vertex_ao(side1: bool, side2: bool, corner: bool) -> u8 {
    if side1 && side2 {
        return 0;
    }
    3 - (side1 as u8 + side2 as u8 + corner as u8)
}

/// Face direction with corner winding; micro-offset zero for verticals (never borders).
struct Dir {
    normal: Normal,
    n_axis: usize,
    u_axis: usize,
    v_axis: usize,
    corners: [[u32; 3]; 4],
    micro: [i8; 3],
}

const DIRS: [Dir; 6] = [
    Dir {
        normal: Normal::PosX,
        n_axis: 0,
        u_axis: 2,
        v_axis: 1,
        corners: [[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]],
        micro: [-1, 0, 0],
    },
    Dir {
        normal: Normal::NegX,
        n_axis: 0,
        u_axis: 2,
        v_axis: 1,
        corners: [[0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 0]],
        micro: [1, 0, 0],
    },
    Dir {
        normal: Normal::PosY,
        n_axis: 1,
        u_axis: 0,
        v_axis: 2,
        corners: [[1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0]],
        micro: [0, 0, 0],
    },
    Dir {
        normal: Normal::NegY,
        n_axis: 1,
        u_axis: 0,
        v_axis: 2,
        corners: [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]],
        micro: [0, 0, 0],
    },
    Dir {
        normal: Normal::PosZ,
        n_axis: 2,
        u_axis: 0,
        v_axis: 1,
        corners: [[1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0]],
        micro: [0, 0, -1],
    },
    Dir {
        normal: Normal::NegZ,
        n_axis: 2,
        u_axis: 0,
        v_axis: 1,
        corners: [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        micro: [0, 0, 1],
    },
];

/// Chunk mesher's cull rule: opaque or same-block faces hide.
#[inline]
fn covered(my: BlockId, nbr: BlockId, tables: &HotTables) -> bool {
    tables.opaque[nbr.0 as usize] || nbr == my
}

/// Decode one quadrant-local column (ix, iz in 0..16) into bottom-up CellRuns
/// tiling `[0, n_cells)` over the quadrant's [`BrickStack`].
fn column_cells(stack: &BrickStack, ix: usize, iz: usize) -> Vec<CellRun> {
    let mut runs = Vec::new();
    let mut lo = 0i32;
    for run in stack.column_runs(ix, iz) {
        let hi = lo + run.count;
        runs.push(CellRun { lo, hi, block: run.block });
        lo = hi;
    }
    runs
}

/// The block at cell `cy` of a column (assumed in `[0, n_cells)`).
#[inline]
fn cell_at(runs: &[CellRun], cy: i32) -> BlockId {
    for r in runs {
        if cy >= r.lo && cy < r.hi {
            return r.block;
        }
    }
    AIR
}

/// Opaque-occupancy probe for AO sampling: below-floor reads solid (matches the
/// cull rule's "solid ground"), above-ceiling and outside the quadrant read air
/// (matches the border-overdraw convention) — never data this quadrant lacks.
fn occluder(cols: &[Vec<CellRun>], n_cells: i32, tables: &HotTables, p: [i32; 3]) -> bool {
    let [px, py, pz] = p;
    if py < 0 {
        return true;
    }
    if py >= n_cells || px < 0 || px >= QUAD_N as i32 || pz < 0 || pz >= QUAD_N as i32 {
        return false;
    }
    let id = cell_at(&cols[px as usize + pz as usize * QUAD_N], py);
    tables.opaque[id.0 as usize]
}

/// Sample a face: cull if covered by neighbor; overdraw section edges as air.
/// Per-corner AO reads the two in-plane occluders plus the diagonal, in the
/// layer the face opens into — same stencil `face_sample` in `world/mesh.rs`
/// uses, just backed by column runs instead of a padded voxel grid.
fn face_sample(
    cols: &[Vec<CellRun>],
    n_cells: i32,
    tables: &HotTables,
    dir: &Dir,
    s: [i32; 3],
) -> Option<FaceSample> {
    let [sx, sy, sz] = s;
    if sy < 0 || sy >= n_cells {
        return None; // above the ceiling in the top block: no cell here
    }
    let col = &cols[sx as usize + sz as usize * QUAD_N];
    let me = cell_at(col, sy);
    if me == AIR {
        return None;
    }
    let d = dir.normal.direction();
    let (nx, ny, nz) = (sx + d[0] as i32, sy + d[1] as i32, sz + d[2] as i32);
    let mut micro = [0i8; 3];
    let nbr = if dir.n_axis == 1 {
        if ny < 0 {
            return None; // below the floor: solid ground, never a silhouette
        } else if ny >= n_cells {
            AIR // above the ceiling: open sky
        } else {
            cell_at(col, ny)
        }
    } else if nx < 0 || nx >= QUAD_N as i32 || nz < 0 || nz >= QUAD_N as i32 {
        // Quadrant border: overdraw as air, nudge inward.
        micro = dir.micro;
        AIR
    } else {
        cell_at(&cols[nx as usize + nz as usize * QUAD_N], ny)
    };
    if covered(me, nbr, tables) {
        return None;
    }
    let o = [nx, ny, nz];
    let occ = |eu: i32, ev: i32| {
        let mut p = o;
        p[dir.u_axis] += eu;
        p[dir.v_axis] += ev;
        occluder(cols, n_cells, tables, p)
    };
    let ao = std::array::from_fn(|i| {
        let eu = if dir.corners[i][1] > 0 { 1 } else { -1 };
        let ev = if dir.corners[i][2] > 0 { 1 } else { -1 };
        vertex_ao(occ(eu, 0), occ(0, ev), occ(eu, ev))
    });
    Some(FaceSample { block: me, micro, ao })
}

/// Greedy-mesh one block: merge adjacent quads with identical properties.
fn build_block(cols: &[Vec<CellRun>], n_cells: i32, base: [i32; 3], tables: &HotTables, out: &mut ChunkMeshData) -> bool {
    let mut mask: [Option<FaceSample>; SLICE] = [None; SLICE];
    let mut emitted = false;

    for dir in &DIRS {
        for nslice in 0..BLOCK {
            let mut any = false;
            for v in 0..BLOCK {
                for u in 0..BLOCK {
                    let mut local = [0i32; 3];
                    local[dir.n_axis] = nslice;
                    local[dir.u_axis] = u;
                    local[dir.v_axis] = v;
                    let s = [base[0] + local[0], base[1] + local[1], base[2] + local[2]];
                    let cell = face_sample(cols, n_cells, tables, dir, s);
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
                    emit(out, dir, nslice, u0, v0, w, h, sample, tables);
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
    origin[dir.n_axis] = nslice as u32;
    origin[dir.u_axis] = u0 as u32;
    origin[dir.v_axis] = v0 as u32;
    let layer = sample.block.0;
    // Route water to opaque pass (no animated texturing at LOD range).
    let is_water = tables.water[layer as usize];
    let pass = if is_water { Pass::Opaque } else { tables.layer[layer as usize] };
    let mut corners: [MeshVertex; 4] = std::array::from_fn(|i| {
        let cr = dir.corners[i];
        let mut pos = [0u32; 3];
        pos[dir.n_axis] = origin[dir.n_axis] + cr[0];
        pos[dir.u_axis] = origin[dir.u_axis] + cr[1] * w as u32;
        pos[dir.v_axis] = origin[dir.v_axis] + cr[2] * h as u32;
        MeshVertex::new(
            [pos[0] as u8, pos[1] as u8, pos[2] as u8],
            dir.normal,
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
/// run-based sampling with no floats).
pub(in crate::world) fn build_section_mesh(section: &Section, tables: &HotTables) -> [SectionMeshData; 4] {
    let n_cells = DOMAIN_H / section.pos().cell_size();
    std::array::from_fn(|q| build_quadrant(section, tables, q as u8, n_cells))
}

/// Mesh one quadrant `q` (its 16×16 column sub-grid) into section-space block
/// origins. Block origins are in CELLS relative to the section min-corner, so the
/// caller positions them the same way regardless of quadrant.
fn build_quadrant(section: &Section, tables: &HotTables, q: u8, n_cells: i32) -> SectionMeshData {
    let (qx, qz) = ((q & 1) as usize, (q >> 1) as usize);
    let stack = &section.quadrants[q as usize];
    let cols: Vec<Vec<CellRun>> = (0..QUAD_N * QUAD_N).map(|i| column_cells(stack, i % QUAD_N, i / QUAD_N)).collect();

    // The vertical slab that actually holds solid runs — sky and deep space are
    // skipped entirely, so K is small for thin terrain.
    let (mut ylo, mut yhi) = (n_cells, 0);
    for col in &cols {
        for r in col {
            if r.block != AIR {
                ylo = ylo.min(r.lo);
                yhi = yhi.max(r.hi);
            }
        }
    }
    let mut result = SectionMeshData::new();
    if yhi <= ylo {
        return result; // no solid geometry in this quadrant
    }

    // Section-space cell origin of the quadrant's XZ corner (0 or 16).
    let (ox, oz) = ((qx * QUAD_N) as u32, (qz * QUAD_N) as u32);
    for by in ylo / BLOCK..=(yhi - 1) / BLOCK {
        let base = [0, by * BLOCK, 0];
        let mut data = new_chunk_mesh_data();
        if build_block(&cols, n_cells, base, tables, &mut data) {
            result.push((UVec3::new(ox, (by * BLOCK) as u32, oz), data));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::BlockRegistry;
    use crate::world::generation::{SineHills, TerrainGenerator};
    use voxel_engine::Pass;
    use crate::world::section::{FINEST_DETAIL, SectionPos};

    // Test fixtures

    struct FnGen<H, B> {
        h: H,
        b: B,
        surf: BlockId,
        deep: BlockId,
    }
    impl<H: Fn(i32, i32) -> i32, B: Fn(i32, i32, i32) -> BlockId> TerrainGenerator for FnGen<H, B> {
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
            grass: id("Grass"),
            dirt: id("Dirt"),
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

    fn all_quads<'a>(mesh: &'a [SectionMeshData; 4]) -> impl Iterator<Item = (UVec3, Pass, &'a [MeshVertex])> {
        mesh.iter().flatten().flat_map(|(origin, data)| {
            Pass::ALL.into_iter().flat_map(move |p| {
                data[p].vertices().chunks_exact(4).map(move |q| (*origin, p, q))
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
                .collect();
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
                } else if y >= 108 && y < 112 && x < 48 {
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
        let is_water_quad = |q: &[MeshVertex]| tables.water[q[0].layer() as usize];
        let opaque_water = all_quads(&mesh).any(|(_, p, q)| p == Pass::Opaque && is_water_quad(q));
        assert!(opaque_water, "water surface meshes into the opaque pass");
        assert!(!all_quads(&mesh).any(|(_, _, q)| q[0].is_water()), "LOD water clears the water bit");
        assert!(
            !all_quads(&mesh).any(|(_, p, _)| p == Pass::Blend),
            "no translucent geometry on a LOD section"
        );
        // Water-vs-water is suppressed; all water side faces are borders (micro != 0).
        for (_, _, q) in all_quads(&mesh) {
            if !is_water_quad(q) {
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

    #[test]
    fn meshing_is_deterministic() {
        let (_r, tables, _b) = setup();
        let r#gen = SineHills::new(&mut BlockRegistry::with_builtins(), 20.0, 0xBEEF);
        let sec = extract(FINEST, &r#gen);
        let a = build_section_mesh(&sec, &tables);
        let b = build_section_mesh(&sec, &tables);
        let flatten = |m: &[SectionMeshData; 4]| -> Vec<(u32, u32, u32, u8, Vec<MeshVertex>, [Vec<u32>; 6])> {
            m.iter()
                .flatten()
                .flat_map(|(o, d)| {
                    Pass::ALL.into_iter().map(move |p| {
                        (o.x, o.y, o.z, p as u8, d[p].vertices().to_vec(), d[p].buckets().clone())
                    })
                })
                .collect()
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
}
