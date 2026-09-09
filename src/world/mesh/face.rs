//! Shared greedy-mesher face tables: cull, AO, axis/corner layout.
use crate::block::registry::{BlockId, HotTables};
use voxel_engine::Normal;

/// Opaque neighbour or same block hides a face (two glass blocks share a hidden internal face).
#[inline]
pub fn covered(my: BlockId, nbr: BlockId, tables: &HotTables) -> bool {
    tables.opaque(nbr) || nbr == my
}

/// Per-vertex ambient-occlusion level `0..=3` (`3` = unoccluded) from its three
/// occluders. Two touching sides fully occlude the corner (the classic clamp).
#[inline]
pub fn vertex_ao(side1: bool, side2: bool, corner: bool) -> u8 {
    if side1 && side2 {
        return 0;
    }
    3 - (side1 as u8 + side2 as u8 + corner as u8)
}

/// One face direction of the greedy sweep (axes + unit-quad corners).
#[derive(Clone, Copy)]
pub struct Dir {
    pub step: i32,
    pub n_axis: usize,
    pub u_axis: usize,
    pub v_axis: usize,
    pub corners: [[u8; 3]; 4],
    pub normal: Normal,
}

pub const DIRS: [Dir; 6] = [
    Dir {
        step: 1,
        n_axis: 0,
        u_axis: 2,
        v_axis: 1,
        corners: [[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]],
        normal: Normal::PosX,
    },
    Dir {
        step: -1,
        n_axis: 0,
        u_axis: 2,
        v_axis: 1,
        corners: [[0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 0]],
        normal: Normal::NegX,
    },
    Dir {
        step: 1,
        n_axis: 1,
        u_axis: 0,
        v_axis: 2,
        corners: [[1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0]],
        normal: Normal::PosY,
    },
    Dir {
        step: -1,
        n_axis: 1,
        u_axis: 0,
        v_axis: 2,
        corners: [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]],
        normal: Normal::NegY,
    },
    Dir {
        step: 1,
        n_axis: 2,
        u_axis: 0,
        v_axis: 1,
        corners: [[1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0]],
        normal: Normal::PosZ,
    },
    Dir {
        step: -1,
        n_axis: 2,
        u_axis: 0,
        v_axis: 1,
        corners: [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        normal: Normal::NegZ,
    },
];

/// Corner u/v signs for the AO 3×3 stencil, derived from unit-quad corners.
#[inline]
pub fn corner_uv(corners: &[[u8; 3]; 4]) -> [[i32; 2]; 4] {
    std::array::from_fn(|i| {
        [
            if corners[i][1] > 0 { 1 } else { -1 },
            if corners[i][2] > 0 { 1 } else { -1 },
        ]
    })
}
