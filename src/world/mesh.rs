//! Turns a chunk's voxels into a GPU-resident triangle mesh.
//!
//! The original renderer issued one immediate-mode `draw_cube` per visible voxel
//! *every frame* — hundreds of thousands of voxel iterations, six hashmap-backed
//! neighbour lookups each, and thousands of individual draw calls. This module
//! replaces all of that with a one-time **culled mesh** per chunk:
//!
//! - Only faces that border air are emitted; interior faces and faces shared
//!   between two solid voxels never exist, so there is no overdraw.
//! - Per-face directional shading is baked into the vertex colours, which fakes
//!   cheap lighting without needing a custom shader.
//! - The mesh is uploaded to the GPU once as a [`Model`]; thereafter a frame is
//!   just a handful of `draw_model` calls and the geometry never leaves VRAM.
//!
//! Per-frame cost drops from `O(voxels)` to `O(chunks)`.
use raylib::prelude::*;

use super::World;
use super::chunk::{CHUNK_DEPTH, CHUNK_HEIGHT, CHUNK_WIDTH, Chunk};

/// One cube face: the neighbour offset tested for culling, the outward normal,
/// the four corner offsets within the unit voxel cube, and a shade factor that
/// fakes directional lighting by darkening the baked vertex colour.
struct Face {
    /// Offset to the voxel this face borders; the face is drawn only if it is air.
    neighbour: [i32; 3],
    normal: Vector3,
    /// Corners wound counter-clockwise as seen from *outside* the cube, so the
    /// default backface-culling pipeline keeps them. Emitted as (0,1,2)+(0,2,3).
    corners: [[f32; 3]; 4],
    /// Brightness multiplier: top brightest, bottom darkest, sides in between.
    shade: f32,
}

const FACES: [Face; 6] = [
    Face {
        neighbour: [1, 0, 0],
        normal: Vector3 { x: 1.0, y: 0.0, z: 0.0 },
        corners: [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
        shade: 0.80,
    },
    Face {
        neighbour: [-1, 0, 0],
        normal: Vector3 { x: -1.0, y: 0.0, z: 0.0 },
        corners: [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        shade: 0.70,
    },
    Face {
        // Top face — full brightness.
        neighbour: [0, 1, 0],
        normal: Vector3 { x: 0.0, y: 1.0, z: 0.0 },
        corners: [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        shade: 1.00,
    },
    Face {
        // Bottom face — darkest.
        neighbour: [0, -1, 0],
        normal: Vector3 { x: 0.0, y: -1.0, z: 0.0 },
        corners: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        shade: 0.50,
    },
    Face {
        neighbour: [0, 0, 1],
        normal: Vector3 { x: 0.0, y: 0.0, z: 1.0 },
        corners: [[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        shade: 0.85,
    },
    Face {
        neighbour: [0, 0, -1],
        normal: Vector3 { x: 0.0, y: 0.0, z: -1.0 },
        corners: [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        shade: 0.65,
    },
];

/// Leave headroom below the u16 index ceiling for the four vertices a face adds.
const MAX_VERTS: usize = 65_532;

/// CPU-side vertex buffers for (part of) a chunk, ready to upload as one mesh.
#[derive(Default)]
struct MeshData {
    vertices: Vec<Vector3>,
    normals: Vec<Vector3>,
    colors: Vec<Color>,
    texcoords: Vec<Vector2>,
    indices: Vec<u16>,
}

impl MeshData {
    fn push_face(&mut self, base: [f32; 3], face: &Face, color: Color) {
        let start = self.vertices.len() as u16;
        for c in &face.corners {
            self.vertices
                .push(Vector3::new(base[0] + c[0], base[1] + c[1], base[2] + c[2]));
            self.normals.push(face.normal);
            self.colors.push(color);
            // No texture is sampled; the default material's 1x1 white texel makes
            // the final colour equal to the vertex colour at any UV.
            self.texcoords.push(Vector2::zero());
        }
        self.indices
            .extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
    }

    /// Upload to the GPU and wrap in a model. `None` if there is nothing to draw.
    fn upload(self, rl: &mut RaylibHandle, thread: &RaylibThread) -> Option<Model> {
        if self.indices.is_empty() {
            return None;
        }
        let mesh = Mesh::gen_mesh(&self.vertices, &self.texcoords)
            .normals(&self.normals)
            .colors(&self.colors)
            .indices(&self.indices)
            .build(thread)
            .expect("chunk mesh buffers are internally consistent by construction");
        // SAFETY: the mesh was just built here and is handed to exactly one model,
        // which takes over ownership (and freeing) of its GPU buffers on drop.
        let weak = unsafe { mesh.make_weak() };
        rl.load_model_from_mesh(thread, weak).ok()
    }
}

/// Build the GPU model(s) for one chunk. Returns more than one model only if the
/// chunk's exposed geometry would overflow a single u16 index buffer.
pub fn build_chunk_models(
    chunk: &Chunk,
    world: &World,
    rl: &mut RaylibHandle,
    thread: &RaylibThread,
) -> Vec<Model> {
    let mut models = Vec::new();
    let mut data = MeshData::default();

    let base_x = chunk.cx * CHUNK_WIDTH as i32;
    let base_z = chunk.cz * CHUNK_DEPTH as i32;

    for ly in 0..CHUNK_HEIGHT {
        for lz in 0..CHUNK_DEPTH {
            for lx in 0..CHUNK_WIDTH {
                let voxel = chunk.get_local(lx, ly, lz);
                if !voxel.is_solid() {
                    continue;
                }

                let wx = base_x + lx as i32;
                let wy = ly as i32;
                let wz = base_z + lz as i32;
                // World-space positions are baked straight into the vertices, so
                // every chunk's model is simply drawn at the origin.
                let base = [wx as f32, wy as f32, wz as f32];
                let color = voxel.color();

                for face in &FACES {
                    let nx = wx + face.neighbour[0];
                    let ny = wy + face.neighbour[1];
                    let nz = wz + face.neighbour[2];
                    // A face is only visible if the voxel it borders is not solid.
                    if world.is_solid(nx, ny, nz) {
                        continue;
                    }

                    if data.vertices.len() >= MAX_VERTS {
                        if let Some(model) = std::mem::take(&mut data).upload(rl, thread) {
                            models.push(model);
                        }
                    }
                    data.push_face(base, face, shade(color, face.shade));
                }
            }
        }
    }

    if let Some(model) = data.upload(rl, thread) {
        models.push(model);
    }
    models
}

/// Multiply a colour's RGB by `factor`, keeping alpha. Used to bake per-face
/// directional shading into vertex colours.
fn shade(c: Color, factor: f32) -> Color {
    Color::new(
        (c.r as f32 * factor) as u8,
        (c.g as f32 * factor) as u8,
        (c.b as f32 * factor) as u8,
        c.a,
    )
}
