//! Stack mesher: turns a [`Section`]'s RLE columns straight into GPU-ready
//! [`MeshData`] — no intermediate voxel grid.
//!
//! A run stack is meshed as boxes: every solid run in a column emits up to six
//! faces (top/bottom capped against the vertically adjacent run in the SAME
//! column; the four sides Y-segment-split against the *whole run stack* of the
//! horizontally adjacent column, so overhangs and floating islands split
//! correctly where a heightmap comparison could not). The cull key is opacity,
//! reusing the chunk mesher's rule verbatim ([`covered`]): an opaque neighbour
//! hides a face, a translucent one (water) does not cover a solid face, and two
//! same-block translucent runs hide their shared face (no walls inside water).
//!
//! Section borders (the ±X/±Z edge columns) are unconditional overdraw per the
//! ledger — the edge face is drawn as if the neighbour were air and given a
//! one-step inward micro-offset ([`MeshVertex::with_micro`]) so coincident
//! overdraw from the abutting section cannot z-fight. Interior faces carry zero
//! offset.
//!
//! A section (32×32 columns) is emitted as a `2×2×K` grid of 16³-CELL blocks
//! (vertex positions are 5-bit, `0..=16`), each drawn by the caller at
//! `offset = section_origin + block_origin·cell`, `scale = cell` — identical to
//! today's tile draws. `K` spans only the vertical slabs the runs actually
//! reach, so sky/deep space costs nothing. There is NO ambient occlusion at LOD
//! range (a fixed `Ao::NONE`) and skylight is the per-run baked nibble, which
//! keeps meshing embarrassingly parallel.
use glam::UVec3;
use voxel_engine::{Ao, Light, MeshVertex, Normal, Pass};

use super::super::mesh::{ChunkMeshData, new_chunk_mesh_data};
use super::{DOMAIN_H, FULL_SKYLIGHT, SECTION_N, Section};
use crate::block::registry::{AIR, BlockId, HotTables};

/// One block's mesh plus its origin in CELLS within the section (drawn at
/// `section_origin + origin·cell`, `scale = cell`). Only non-empty blocks appear.
pub(in crate::world) type SectionMeshData = Vec<(UVec3, ChunkMeshData)>;

/// Cells per mesh-block edge — the 5-bit vertex position range (`0..=16`). Fixed by design.
const BLOCK: i32 = 16;
/// Mesh blocks per section side (`32 / 16 = 2`).
const BLOCKS_XZ: i32 = SECTION_N as i32 / BLOCK;
/// Cells in one block slice (`16×16`).
const SLICE: usize = (BLOCK * BLOCK) as usize;

/// One solid-or-air run of a column expressed in CELL coordinates (`[lo, hi)`,
/// bottom-up), with its block and baked skylight. Air runs are kept so a
/// neighbour lookup over the whole stack is a plain scan.
#[derive(Clone, Copy)]
struct CellRun {
    lo: i32,
    hi: i32,
    block: BlockId,
    sky: u8,
}

/// Merge key: block, skylight, micro must match; border faces never merge into interior.
#[derive(Clone, Copy, PartialEq, Eq)]
struct FaceSample {
    block: BlockId,
    sky: u8,
    micro: [i8; 3],
}

/// Face direction; corners copied from chunk mesher for winding; micro-offset zero for verticals (never borders).
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

/// Explode one column's top-down metre runs into bottom-up CELL runs tiling
/// `[0, n_cells)`. Run heights must be whole cells; the block partition and
/// vertex encoding depend on it, so we assert rather than assume.
fn column_cells(section: &Section, ix: usize, iz: usize, n_cells: i32) -> Vec<CellRun> {
    let cell = section.pos().cell_size();
    let col = section.column(ix, iz);
    let pal = section.palette();
    let mut runs = Vec::with_capacity(col.runs().len());
    let mut top = n_cells;
    for &r in col.runs() {
        debug_assert_eq!(r.height() as i32 % cell, 0, "run height is not a whole number of cells");
        let lo = top - r.height() as i32 / cell;
        runs.push(CellRun { lo, hi: top, block: pal.get(r.id()), sky: r.skylight() });
        top = lo;
    }
    debug_assert_eq!(top, 0, "cell runs must tile the domain");
    runs.reverse();
    runs
}

/// The `(block, skylight)` at cell `cy` of a column (assumed in `[0, n_cells)`).
#[inline]
fn cell_at(runs: &[CellRun], cy: i32) -> (BlockId, u8) {
    for r in runs {
        if cy >= r.lo && cy < r.hi {
            return (r.block, r.sky);
        }
    }
    (AIR, 0)
}

/// Y edges asymmetric (floor=cull, sky=lit). Section edges drawn beyond boundary (overdraw with assumed-air neighbor).
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
    let col = &cols[sx as usize + sz as usize * SECTION_N];
    let (me, my_sky) = cell_at(col, sy);
    if me == AIR {
        return None;
    }
    let d = dir.normal.direction();
    let (nx, ny, nz) = (sx + d[0] as i32, sy + d[1] as i32, sz + d[2] as i32);
    let mut micro = [0i8; 3];
    let (nbr, sky) = if dir.n_axis == 1 {
        if ny < 0 {
            return None; // below the floor: solid ground, never a silhouette
        } else if ny >= n_cells {
            (AIR, FULL_SKYLIGHT) // above the ceiling: open sky
        } else {
            cell_at(col, ny)
        }
    } else if nx < 0 || nx >= SECTION_N as i32 || nz < 0 || nz >= SECTION_N as i32 {
        micro = dir.micro; // section border: overdraw as if air, nudge inward
        (AIR, my_sky)
    } else {
        cell_at(&cols[nx as usize + nz as usize * SECTION_N], ny)
    };
    if covered(me, nbr, tables) {
        return None;
    }
    Some(FaceSample { block: me, sky, micro })
}

/// Greedy-mesh one block; sweep mirrors chunk mesher's mask/grow/sweep for segment-splitting and merging.
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
    // Opaque pass prevents z-fight on coarse LOD. Clears water bit (animated only). Distant-Horizons: no fluid on LOD.
    let is_water = tables.water[layer as usize];
    let pass = if is_water { Pass::Opaque } else { tables.layer[layer as usize] };
    let corners = std::array::from_fn(|i| {
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
            Ao::NONE,
            Light::new(sample.sky, 0),
            false,
        )
        .with_micro(sample.micro)
    });
    out[pass].quad(corners);
}

/// Mesh a whole section into per-block [`ChunkMeshData`]. Deterministic: the same
/// section yields bit-identical output (fixed block iteration order, run-based
/// sampling with no floats).
pub(in crate::world) fn build_section_mesh(section: &Section, tables: &HotTables) -> SectionMeshData {
    let n_cells = DOMAIN_H / section.pos().cell_size();
    let cols: Vec<Vec<CellRun>> =
        (0..SECTION_N * SECTION_N).map(|i| column_cells(section, i % SECTION_N, i / SECTION_N, n_cells)).collect();

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
        return result; // no solid geometry anywhere
    }

    for by in ylo / BLOCK..=(yhi - 1) / BLOCK {
        for bz in 0..BLOCKS_XZ {
            for bx in 0..BLOCKS_XZ {
                let base = [bx * BLOCK, by * BLOCK, bz * BLOCK];
                let mut data = new_chunk_mesh_data();
                if build_block(&cols, n_cells, base, tables, &mut data) {
                    result.push((UVec3::new(base[0] as u32, base[1] as u32, base[2] as u32), data));
                }
            }
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

    // -- fixtures ----------------------------------------------------------

    /// Test generator mirroring section.rs's FnGen for extraction reuse.
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
    const CELL: i32 = 1 << FINEST_DETAIL;

    /// A ground/surface/air column with an optional water table and an optional
    /// solid shelf `[shelf.0, shelf.1)` above the surface (metres) — the same
    /// class generator section.rs uses, so every terrain shape is reachable.
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

    fn mesh_of(section: &Section, tables: &HotTables) -> SectionMeshData {
        build_section_mesh(section, tables)
    }

    /// Every vertex of every quad in `data`, tagged with its block-relative
    /// position, normal, layer, skylight, micro, water and the block origin.
    fn all_quads<'a>(mesh: &'a SectionMeshData) -> impl Iterator<Item = (UVec3, Pass, &'a [MeshVertex])> {
        mesh.iter().flat_map(|(origin, data)| {
            Pass::ALL.into_iter().flat_map(move |p| {
                data[p].vertices().chunks_exact(4).map(move |q| (*origin, p, q))
            })
        })
    }

    fn normals_present(mesh: &SectionMeshData, want: Normal) -> bool {
        all_quads(mesh).any(|(_, _, q)| q[0].normal() == want)
    }

    /// Every quad winds CCW from outside (engine back-face-culls otherwise) —
    /// checked in WORLD cell space (block origin + local position).
    fn assert_winds_outward(mesh: &SectionMeshData) {
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

    // -- cases -------------------------------------------------------------

    #[test]
    fn flat_terrain_shows_tops_and_only_border_side_walls() {
        let (_r, tables, b) = setup();
        let sec = Section::extract(FINEST, &terrain_gen(&b, 200, 0, None), &[]);
        let mesh = mesh_of(&sec, &tables);
        assert!(!mesh.is_empty(), "flat ground has geometry");
        assert!(normals_present(&mesh, Normal::PosY), "the surface has a top");
        assert_winds_outward(&mesh);

        // Interior side faces are all culled (identical opaque neighbours); the
        // only side faces are the section-edge overdraw, which carry a micro
        // offset. So: every side quad has micro != 0, and every interior quad
        // (top) has micro == 0.
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
        // Surface at the floor: every column is all air over the whole domain.
        let sec = Section::extract(FINEST, &terrain_gen(&b, 0, 0, None), &[]);
        assert!(mesh_of(&sec, &tables).is_empty(), "an all-air section meshes to nothing");
    }

    #[test]
    fn floating_shelf_emits_a_bottom_and_segment_split_sides() {
        let (_r, tables, b) = setup();
        // Ground at 100 everywhere, plus a detached stone slab in cells [108,112)
        // covering only the LOW-X half of the section (world x < 64). The slab's
        // edge column sits inside the section, so its exposed side faces the air
        // gap of the neighbouring slab-free column. This is the segment-split case
        // a heightmap mesher cannot handle.
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
                } else if y >= 108 && y < 112 && x < 64 {
                    stone
                } else {
                    AIR
                }
            },
            surf: grass,
            deep: stone,
        };
        let sec = Section::extract(FINEST, &r#gen, &[]);
        let mesh = mesh_of(&sec, &tables);
        assert_winds_outward(&mesh);
        // The slab's underside is open to the gap; a downward face a pure
        // ground-only mesher would never emit.
        assert!(normals_present(&mesh, Normal::NegY), "the floating slab shows its underside");
        // The slab's inner edge faces the neighbour column's air across the gap
        // band; an interior (micro == 0) side quad survives the segment split.
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
        let sec = Section::extract(FINEST, &terrain_gen(&b, 40, 80, None), &[]);
        let mesh = mesh_of(&sec, &tables);
        // LOD water is an OPAQUE solid (Distant-Horizons rule: no fluid pass on
        // coarse sections), and it never carries the animated-water bit.
        let is_water_quad = |q: &[MeshVertex]| tables.water[q[0].layer() as usize];
        let opaque_water = all_quads(&mesh).any(|(_, p, q)| p == Pass::Opaque && is_water_quad(q));
        assert!(opaque_water, "water surface meshes into the opaque pass");
        assert!(!all_quads(&mesh).any(|(_, _, q)| q[0].is_water()), "LOD water clears the water bit");
        assert!(
            !all_quads(&mesh).any(|(_, p, _)| p == Pass::Blend),
            "no translucent geometry on a LOD section"
        );
        // No interior water wall exists (water-vs-water is suppressed); every water
        // side face is a section-border overdraw (micro != 0).
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
        let sec = Section::extract(FINEST, &terrain_gen(&b, 200, 0, Some((300, 320))), &[]);
        let mesh = mesh_of(&sec, &tables);
        let mut seen = std::collections::HashSet::new();
        for (origin, data) in &mesh {
            // Origins are distinct 16-cell-aligned block corners.
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
        // Flat grass top: each 16×16 XZ block's top plane should merge to ONE
        // quad, so the section's four XZ blocks give exactly four top quads.
        let flat = Section::extract(FINEST, &terrain_gen(&b, 200, 0, None), &[]);
        let tops = all_quads(&build_section_mesh(&flat, &tables))
            .filter(|(_, _, q)| q[0].normal() == Normal::PosY)
            .count();
        assert_eq!(tops, BLOCKS_XZ as usize * BLOCKS_XZ as usize, "a uniform flat top merges per block");

        // A checkerboard of two surface blocks cannot merge its tops.
        let (grass, dirt) = (b.grass, b.dirt);
        let stone = b.stone;
        let checker = FnGen {
            h: |_, _| 200,
            // The alternating band is a full cell thick ([196,200), centre at 198)
            // so it samples; a 1 m band would miss the sampler. Alternates at each
            // 4-meter cell boundary.
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
        let sec = Section::extract(FINEST, &checker, &[]);
        let mesh = build_section_mesh(&sec, &tables);
        let top_quads = all_quads(&mesh).filter(|(_, _, q)| q[0].normal() == Normal::PosY).count();
        // 32×32 alternating tops cannot merge across the id change: many quads.
        assert!(top_quads > 100, "checkerboard tops must not merge (got {top_quads})");
    }

    #[test]
    fn meshing_is_deterministic() {
        let (_r, tables, _b) = setup();
        let r#gen = SineHills::new(&mut BlockRegistry::with_builtins(), 20.0, 0xBEEF);
        let sec = Section::extract(FINEST, &r#gen, &[]);
        let a = build_section_mesh(&sec, &tables);
        let b = build_section_mesh(&sec, &tables);
        let flatten = |m: &SectionMeshData| -> Vec<(u32, u32, u32, u8, Vec<MeshVertex>, [Vec<u32>; 6])> {
            m.iter()
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
    fn coarser_detail_agrees_on_the_exposed_surface() {
        // Detail 2 vs detail 4 over the same world area (a coarse section spans
        // 2× the metres, so its grid-0 area overlaps the finest grid-0 area).
        // Both should carry a top surface — a sanity check that extraction and
        // meshing generalise across the stride, not an exact-equality claim.
        let (_r, tables, b) = setup();
        for detail in [FINEST_DETAIL, FINEST_DETAIL + 2] {
            let pos = SectionPos { detail, x: 0, z: 0 };
            let sec = Section::extract(pos, &terrain_gen(&b, 200, 0, None), &[]);
            let mesh = build_section_mesh(&sec, &tables);
            assert!(normals_present(&mesh, Normal::PosY), "detail {detail} lost the top surface");
            assert_winds_outward(&mesh);
        }
    }
}
