//! One-pass CPU memory census of a [`World`](super::World).
//!
//! Called at most twice per benchmark run (ready + end). Walks the live maps
//! once; numbers are payload bytes, not RSS.

use super::brick::ChunkPayload;
use super::chunk::Chunk;
use super::light::LightGrid;
use super::mesh::ChunkMeshData;
use super::{Coord, SectionState, World, pipeline};
use crate::block::registry::BlockId;
use crate::ident::BlockState;
use super::section::SectionPos;
use voxel_engine::MeshData;

/// CPU-side world memory snapshot. `total` is the sum of the byte fields.
///
/// CPU mesh bytes: GPU upload copies out of [`MeshData`] and the live chunk
/// then holds only a handle. This field is the world's reusable scratch plus
/// any upload-queue payloads not yet consumed. It is often 0 after settle
/// because those `Vec`s are returned to the mesh-output pool rather than kept
/// on [`World`]. The static pool is not attributed here.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryCensus {
    pub chunk_uniform_bytes: usize,
    pub chunk_uniform_count: usize,
    pub chunk_paletted_bytes: usize,
    pub chunk_paletted_count: usize,
    pub chunk_dense_bytes: usize,
    pub chunk_dense_count: usize,
    pub light_uniform_bytes: usize,
    pub light_uniform_count: usize,
    pub light_cells_bytes: usize,
    pub light_cells_count: usize,
    pub mesh_cpu_bytes: usize,
    pub edit_overlay_bytes: usize,
    pub section_lod_bytes: usize,
    pub worklist_bytes: usize,
    pub total: usize,
}

impl MemoryCensus {
    fn finish(mut self) -> Self {
        self.total = self.chunk_uniform_bytes
            + self.chunk_paletted_bytes
            + self.chunk_dense_bytes
            + self.light_uniform_bytes
            + self.light_cells_bytes
            + self.mesh_cpu_bytes
            + self.edit_overlay_bytes
            + self.section_lod_bytes
            + self.worklist_bytes;
        self
    }
}

impl World {
    /// Walk loaded maps once and report CPU payload bytes.
    pub fn memory_census(&self) -> MemoryCensus {
        let mut c = MemoryCensus::default();
        for loaded in self.chunks.values() {
            add_chunk_payload(&mut c, &loaded.chunk);
            if let Some(grid) = &loaded.light {
                add_light(&mut c, grid);
            }
        }
        for (_, grid) in &self.light_apply_queue {
            add_light(&mut c, grid);
        }
        c.mesh_cpu_bytes = mesh_held(&self.scratch);
        for (_, _, out) in &self.upload_queue {
            c.mesh_cpu_bytes += mesh_output_held(out);
        }
        for (_, _, _, quads) in &self.section_upload_queue {
            for quad in quads.iter() {
                for (_, data) in quad {
                    c.mesh_cpu_bytes += mesh_held(data);
                }
            }
        }
        c.edit_overlay_bytes = edit_bytes(&self.edits);
        c.section_lod_bytes = section_lod_bytes(self);
        c.worklist_bytes = worklist_bytes(self);
        c.finish()
    }
}

fn add_chunk_payload(c: &mut MemoryCensus, chunk: &Chunk) {
    match &chunk.data().payload {
        ChunkPayload::Uniform(_) => {
            c.chunk_uniform_count += 1;
            c.chunk_uniform_bytes += std::mem::size_of::<BlockState>();
        }
        ChunkPayload::Paletted { palette, cells } => {
            c.chunk_paletted_count += 1;
            c.chunk_paletted_bytes += std::mem::size_of_val(palette.as_ref())
                + std::mem::size_of_val(cells.as_ref());
        }
        ChunkPayload::Dense(cells) => {
            c.chunk_dense_count += 1;
            c.chunk_dense_bytes += std::mem::size_of_val(cells.as_ref());
        }
    }
}

fn add_light(c: &mut MemoryCensus, grid: &LightGrid) {
    if grid.is_uniform() {
        c.light_uniform_count += 1;
        c.light_uniform_bytes += std::mem::size_of::<super::light::Lumel>();
    } else {
        c.light_cells_count += 1;
        c.light_cells_bytes += grid.allocated_bytes();
    }
}

fn mesh_held(data: &ChunkMeshData) -> usize {
    voxel_engine::Pass::ALL
        .iter()
        .map(|&p| mesh_data_held(&data[p]))
        .sum()
}

fn mesh_output_held(data: &pipeline::MeshOutput) -> usize {
    super::streaming::mesh_output_bytes(data)
}

fn mesh_data_held(data: &MeshData) -> usize {
    data.vertex_bytes()
}

fn edit_bytes(edits: &super::FastMap<Coord, super::FastMap<usize, BlockId>>) -> usize {
    let mut n = map_cap::<Coord, super::FastMap<usize, BlockId>>(edits.capacity());
    for inner in edits.values() {
        n += map_cap::<usize, BlockId>(inner.capacity());
        n += inner.len() * (std::mem::size_of::<usize>() + std::mem::size_of::<BlockId>());
    }
    n
}

fn section_lod_bytes(world: &World) -> usize {
    let mut n = map_cap::<SectionPos, SectionState>(world.sections.capacity());
    if let Some(mip) = &world.section_mip {
        n += mip.allocated_bytes();
    }
    n += map_cap::<SectionPos, super::heightmip::MipCell>(world.section_overlay.capacity());
    n += world.section_overlay.len() * std::mem::size_of::<super::heightmip::MipCell>();
    n += vec_cap::<SectionPos>(world.section_desired.capacity());
    n += vec_cap::<(SectionPos, super::quadtree::QuadrantMask)>(world.section_visible.capacity());
    n += map_cap::<SectionPos, u64>(world.section_edit_rev.capacity());
    n += map_cap::<SectionPos, super::FastSet<Coord>>(world.section_edit_chunks.capacity());
    for set in world.section_edit_chunks.values() {
        n += set_cap::<Coord>(set.capacity());
    }
    n
}

fn worklist_bytes(world: &World) -> usize {
    set_cap::<Coord>(world.generating.capacity())
        + set_cap::<Coord>(world.mesh_worklist.capacity())
        + set_cap::<Coord>(world.light_worklist.capacity())
        + set_cap::<Coord>(world.light_inflight.capacity())
        + set_cap::<Coord>(world.dirty_worklist.capacity())
        + map_cap::<Coord, std::time::Instant>(world.light_gate.dirty.capacity())
        + set_cap::<SectionPos>(world.dirty_sections.capacity())
        + deque_cap::<(Coord, u32, pipeline::MeshOutput)>(world.upload_queue.capacity())
        + deque_cap::<(Coord, LightGrid)>(world.light_apply_queue.capacity())
        + deque_cap::<(
            SectionPos,
            pipeline::ClaimToken,
            usize,
            pipeline::SectionMeshOutput,
        )>(world.section_upload_queue.capacity())
        + deque_cap::<Coord>(world.conn_fill_queue.capacity())
}

fn map_cap<K, V>(cap: usize) -> usize {
    cap * (std::mem::size_of::<K>() + std::mem::size_of::<V>())
}

fn set_cap<K>(cap: usize) -> usize {
    cap * std::mem::size_of::<K>()
}

fn vec_cap<T>(cap: usize) -> usize {
    cap * std::mem::size_of::<T>()
}

fn deque_cap<T>(cap: usize) -> usize {
    cap * std::mem::size_of::<T>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn census_is_stable_on_an_unchanged_world() {
        let world = World::generate();
        let a = world.memory_census();
        let b = world.memory_census();
        assert_eq!(a, b);
        assert_eq!(a.total, b.total);
        assert_eq!(
            a.total,
            a.chunk_uniform_bytes
                + a.chunk_paletted_bytes
                + a.chunk_dense_bytes
                + a.light_uniform_bytes
                + a.light_cells_bytes
                + a.mesh_cpu_bytes
                + a.edit_overlay_bytes
                + a.section_lod_bytes
                + a.worklist_bytes
        );
        assert!(a.chunk_uniform_count + a.chunk_paletted_count + a.chunk_dense_count > 0);
        assert!(
            a.light_uniform_count > 0,
            "origin box publishes trivial Uniform light (sky/rock)"
        );
        eprintln!(
            "CENSUS origin chunks=u{}/p{}/d{} light=u{}/c{} light_bytes=u{}/c{} total={}",
            a.chunk_uniform_count,
            a.chunk_paletted_count,
            a.chunk_dense_count,
            a.light_uniform_count,
            a.light_cells_count,
            a.light_uniform_bytes,
            a.light_cells_bytes,
            a.total,
        );
    }
}
