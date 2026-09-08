//! Per-chunk face connectivity + camera-rooted visibility BFS — the "cave
//! culling" occlusion pass. Frustum culling (in the engine) removes chunks
//! outside the view; this removes chunks the view *cannot reach* because opaque
//! terrain walls them off (the far side of a hill, sealed cave networks). Keys on
//! opacity, not solidity: water/glass are solid but see-through, so a sightline
//! passes through them and does not seal the chunks behind.
//!
//! Two pure, engine-free pieces, both unit-tested headless like the mesher:
//!
//! - [`Connectivity::compute`] — for one chunk, which of its six faces a
//!   straight-through sightline can pass *between*, via a connected pocket of
//!   non-solid cells. A flood-fill over the same voxel data the mesher visits.
//! - [`visible_set`] — a BFS outward from the camera's chunk over the loaded
//!   chunk map, entering a chunk only through a face its connectivity says the
//!   sightline can traverse.
//!
//! **Correctness:** the only way this pass produces a hole (visible chunk wrongly
//! culled) is by failing to reach a visible chunk. Reaching extra chunks only
//! weakens the cull. So this implementation is deliberately permissive: it reaches
//! at least every visible chunk. Tighter optimizations are deferred.
use super::brick::ChunkPayload;
use super::chunk::{CHUNK_VOLUME, Chunk};
use crate::block::registry::BlockId;
use crate::coord::{ChunkBox, ChunkCoord, Face};

/// Which of a chunk's six faces a sightline can pass through. Encodes face pairs
/// as bits in a `u16` for efficient connectivity checks.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Connectivity(u16);

/// Bit mask for an unordered pair of faces.
#[inline]
fn pair_bit(a: Face, b: Face) -> u16 {
    let (a, b) = (a as usize, b as usize);
    if a == b {
        return 0;
    }
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    let index = lo * (11 - lo) / 2 + (hi - lo - 1);
    1 << index
}

impl Connectivity {
    /// Nothing connects — a fully solid chunk. Sightlines stop at it (it is
    /// still drawn if reached; it just isn't passed *through*).
    pub const SEALED: Connectivity = Connectivity(0);
    /// Every face pair connects — a fully open (all-air) chunk.
    pub const OPEN: Connectivity = Connectivity(0x7FFF);

    /// Whether a sightline entering through `a` can leave through `b`.
    #[inline]
    pub fn connects(self, a: Face, b: Face) -> bool {
        self.0 & pair_bit(a, b) != 0
    }

    /// Record a pocket touching certain faces; mark all face pairs as connected.
    fn add_pocket(&mut self, faces: u8) {
        for a in Face::ALL {
            for b in Face::ALL {
                if (a as usize) < (b as usize)
                    && faces & (1 << a as usize) != 0
                    && faces & (1 << b as usize) != 0
                {
                    self.0 |= pair_bit(a, b);
                }
            }
        }
    }

    /// Face connectivity of one chunk: flood-fill the cells a sightline can pass
    /// through and, for each connected pocket, connect every pair of chunk faces
    /// it reaches. `blocks_sight` classifies a [`BlockId`] as opaque — a closure so
    /// the caller can back it with the registry or a snapshot table without this
    /// module knowing which. Keys on *opacity*, not solidity: water/glass are solid
    /// but see-through, so a sightline passes through them.
    pub fn compute(chunk: &Chunk, blocks_sight: impl Fn(BlockId) -> bool) -> Connectivity {
        match &chunk.data().payload {
            // Uniform chunks need no scan: opaque seals everything, see-through opens it.
            ChunkPayload::Uniform(v) => {
                if blocks_sight(v.id) { Self::SEALED } else { Self::OPEN }
            }
            // One classify per palette entry up front; the fill then reads a
            // bool per cell instead of re-classifying ids.
            ChunkPayload::Paletted { palette, cells } => {
                let sight: Vec<bool> = palette.iter().map(|s| blocks_sight(s.id)).collect();
                Self::flood(|i| !sight[cells[i] as usize])
            }
            ChunkPayload::Dense(cells) => Self::flood(|i| !blocks_sight(cells[i].id)),
        }
    }

    /// The pocket flood fill over an abstract passability predicate — shared by
    /// every dense representation.
    fn flood(passable: impl Fn(usize) -> bool) -> Connectivity {
        let mut visited = [false; CHUNK_VOLUME];
        let mut conn = Connectivity::SEALED;
        let mut stack: Vec<usize> = Vec::new();
        for start in 0..CHUNK_VOLUME {
            if visited[start] || !passable(start) {
                continue;
            }
            visited[start] = true;
            stack.push(start);
            let mut faces: u8 = 0;
            while let Some(i) = stack.pop() {
                let (x, y, z) = Chunk::local_of(i);
                faces |= boundary_faces(x, y, z);
                for (nx, ny, nz) in orthogonal_neighbours(x, y, z) {
                    let j = Chunk::index(nx, ny, nz);
                    if !visited[j] && passable(j) {
                        visited[j] = true;
                        stack.push(j);
                    }
                }
            }
            conn.add_pocket(faces);
        }
        conn
    }
}

/// Bitmask of chunk faces that a cell touches (0 for interior cells).
fn boundary_faces(x: usize, y: usize, z: usize) -> u8 {
    const EDGE: usize = super::chunk::CHUNK_SIZE - 1;
    let mut m = 0u8;
    if x == 0 {
        m |= 1 << Face::NegX as usize;
    }
    if x == EDGE {
        m |= 1 << Face::PosX as usize;
    }
    if y == 0 {
        m |= 1 << Face::NegY as usize;
    }
    if y == EDGE {
        m |= 1 << Face::PosY as usize;
    }
    if z == 0 {
        m |= 1 << Face::NegZ as usize;
    }
    if z == EDGE {
        m |= 1 << Face::PosZ as usize;
    }
    m
}

/// The in-bounds orthogonal neighbours of a chunk-local cell (2–6 of them).
fn orthogonal_neighbours(x: usize, y: usize, z: usize) -> impl Iterator<Item = (usize, usize, usize)> {
    const EDGE: usize = super::chunk::CHUNK_SIZE - 1;
    let mut out = [(0usize, 0usize, 0usize); 6];
    let mut n = 0;
    let mut push = |c: (usize, usize, usize)| {
        out[n] = c;
        n += 1;
    };
    if x > 0 {
        push((x - 1, y, z));
    }
    if x < EDGE {
        push((x + 1, y, z));
    }
    if y > 0 {
        push((x, y - 1, z));
    }
    if y < EDGE {
        push((x, y + 1, z));
    }
    if z > 0 {
        push((x, y, z - 1));
    }
    if z < EDGE {
        push((x, y, z + 1));
    }
    out.into_iter().take(n)
}

/// Dense occupancy for the occlusion BFS: bit 6 is visible, bits 0–5 are the
/// entry faces already expanded. A loaded chunk outside the current unload box
/// reports visible so it is never culled.
const VISIBLE_BIT: u8 = 1 << 6;

/// The occlusion pass: determines which chunks are visible from the camera.
/// Rebuilt into a dense byte grid covering the unload box — one array lookup
/// per query, no per-chunk hashing.
pub struct Occlusion {
    origin: ChunkCoord,
    nx: i32,
    ny: i32,
    nz: i32,
    cells: Vec<u8>,
    /// Frontier of (chunk, entry-face); the root carries no entry face.
    queue: Vec<(ChunkCoord, Option<Face>)>,
}

impl Default for Occlusion {
    fn default() -> Self {
        Self {
            origin: ChunkCoord::new(0, 0, 0),
            nx: 0,
            ny: 0,
            nz: 0,
            cells: Vec::new(),
            queue: Vec::new(),
        }
    }
}

impl Occlusion {
    #[inline]
    fn index(&self, c: ChunkCoord) -> Option<usize> {
        let dx = c.x.wrapping_sub(self.origin.x);
        let dy = c.y.wrapping_sub(self.origin.y);
        let dz = c.z.wrapping_sub(self.origin.z);
        if dx < 0 || dy < 0 || dz < 0 || dx >= self.nx || dy >= self.ny || dz >= self.nz {
            return None;
        }
        Some((dx + dz * self.nx + dy * self.nx * self.nz) as usize)
    }

    /// Whether `coord` was reached from the camera in the last [`rebuild`](Self::rebuild).
    /// Outside the current box reports visible — a loaded chunk past the unload
    /// hysteresis must never be culled.
    #[inline]
    pub fn is_visible(&self, coord: ChunkCoord) -> bool {
        match self.index(coord) {
            Some(i) => self.cells[i] & VISIBLE_BIT != 0,
            None => true,
        }
    }

    #[cfg(test)]
    fn visible_count(&self) -> usize {
        self.cells.iter().filter(|c| *c & VISIBLE_BIT != 0).count()
    }

    /// Recompute the visible set using BFS from the camera's chunk. Each chunk
    /// is entered through a face and may exit through connected faces. The camera's
    /// chunk can see out of every face; other chunks are reached progressively.
    pub fn rebuild(
        &mut self,
        volume: ChunkBox,
        origin: ChunkCoord,
        conn_of: impl Fn(ChunkCoord) -> Option<Connectivity>,
    ) {
        let (nx, ny, nz) = volume.size();
        let n = (nx * ny * nz) as usize;
        if self.nx != nx || self.ny != ny || self.nz != nz || self.cells.len() != n {
            self.cells.resize(n, 0);
            self.nx = nx;
            self.ny = ny;
            self.nz = nz;
        } else {
            self.cells.fill(0);
        }
        self.origin = volume.min();
        self.queue.clear();
        self.queue.push((origin, None));

        while let Some((coord, entry)) = self.queue.pop() {
            let Some(idx) = self.index(coord) else {
                continue;
            };
            let conn = match entry {
                // The camera's own chunk is the root: always expanded (and it is
                // generated synchronously, so it is loaded in practice).
                None => conn_of(coord).unwrap_or(Connectivity::OPEN),
                // A reached chunk that isn't loaded is the frontier: it is not
                // drawn and must NOT be propagated through, or the BFS would
                // flood outward across infinite empty space and never terminate.
                Some(face) => {
                    let Some(conn) = conn_of(coord) else { continue };
                    let seen = self.cells[idx];
                    if seen & (1 << face as usize) != 0 {
                        continue; // this entry face already expanded
                    }
                    self.cells[idx] = seen | (1 << face as usize);
                    conn
                }
            };
            self.cells[idx] |= VISIBLE_BIT;
            for exit in Face::ALL {
                let open = match entry {
                    None => true, // camera chunk sees out of every face
                    Some(entry) => conn.connects(entry, exit),
                };
                if open {
                    // NOTE: per-chunk connectivity is conservative—we don't check if
                    // the shared boundary is actually open, which can over-report
                    // visibility. This never culls a visible chunk.
                    self.queue.push((coord.step(exit), Some(exit.opposite())));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::AIR;

    const STONE: BlockId = BlockId(1);
    /// Only `STONE` is solid.
    fn is_solid(id: BlockId) -> bool {
        id == STONE
    }

    /// A mixed chunk built from a per-cell fill closure.
    fn dense(mut fill: impl FnMut(usize, usize, usize) -> BlockId) -> Chunk {
        let mut cells = Box::new([AIR; CHUNK_VOLUME]);
        for y in 0..16 {
            for z in 0..16 {
                for x in 0..16 {
                    cells[Chunk::index(x, y, z)] = fill(x, y, z);
                }
            }
        }
        Chunk::from_cells(0, 0, 0, cells)
    }

    #[test]
    fn pair_bits_are_15_distinct_values() {
        let mut bits = std::collections::HashSet::new();
        for a in Face::ALL {
            for b in Face::ALL {
                if (a as usize) < (b as usize) {
                    assert!(bits.insert(pair_bit(a, b)), "duplicate pair bit");
                }
            }
        }
        assert_eq!(bits.len(), 15);
        assert_eq!(pair_bit(Face::NegX, Face::PosX), pair_bit(Face::PosX, Face::NegX));
        assert_eq!(pair_bit(Face::NegX, Face::NegX), 0);
    }

    #[test]
    fn uniform_solid_seals_and_uniform_air_opens() {
        assert_eq!(Connectivity::compute(&Chunk::from_uniform(0, 0, 0, STONE), is_solid), Connectivity::SEALED);
        assert_eq!(Connectivity::compute(&Chunk::from_uniform(0, 0, 0, AIR), is_solid), Connectivity::OPEN);
        // OPEN connects every ordered pair; SEALED connects none.
        for a in Face::ALL {
            for b in Face::ALL {
                assert_eq!(Connectivity::OPEN.connects(a, b), a != b);
                assert!(!Connectivity::SEALED.connects(a, b));
            }
        }
    }

    #[test]
    fn solid_wall_splits_faces_into_two_groups() {
        // A solid slab at x == 8 walls the −X half off from the +X half.
        let chunk = dense(|x, _, _| if x == 8 { STONE } else { AIR });
        let c = Connectivity::compute(&chunk, is_solid);
        // −X still reaches ±Y/±Z (its own open half) but NOT +X across the wall.
        assert!(c.connects(Face::NegX, Face::NegY));
        assert!(c.connects(Face::PosX, Face::PosY));
        assert!(!c.connects(Face::NegX, Face::PosX), "wall blocks the through line");
    }

    #[test]
    fn straight_tube_connects_only_its_two_ends() {
        // Solid everywhere except a single column of air along Y at (8, *, 8):
        // the only pocket runs floor-to-ceiling, touching just ±Y.
        let chunk = dense(|x, _, z| if x == 8 && z == 8 { AIR } else { STONE });
        let c = Connectivity::compute(&chunk, is_solid);
        assert!(c.connects(Face::NegY, Face::PosY), "the tube joins top and bottom");
        for a in Face::ALL {
            for b in Face::ALL {
                let is_vertical = matches!((a, b), (Face::NegY, Face::PosY) | (Face::PosY, Face::NegY));
                if !is_vertical {
                    assert!(!c.connects(a, b), "no pocket touches {a:?}/{b:?}");
                }
            }
        }
    }

    #[test]
    fn open_field_makes_every_loaded_chunk_visible() {
        // All chunks OPEN → BFS reaches the whole loaded region.
        let loaded: Vec<ChunkCoord> = (-2..=2)
            .flat_map(|x| (-2..=2).flat_map(move |y| (-2..=2).map(move |z| ChunkCoord::new(x, y, z))))
            .collect();
        let origin = ChunkCoord::new(0, 0, 0);
        let mut occ = Occlusion::default();
        occ.rebuild(ChunkBox::new(origin, 2, 2), origin, |c| {
            loaded.contains(&c).then_some(Connectivity::OPEN)
        });
        assert_eq!(occ.visible_count(), loaded.len());
        assert!(loaded.iter().all(|&c| occ.is_visible(c)));
    }

    #[test]
    fn unloaded_neighbours_bound_the_bfs() {
        // Only the origin is loaded (OPEN). The frontier must stop at every
        // unloaded neighbour instead of flooding outward forever.
        let origin = ChunkCoord::new(5, -3, 2);
        let mut occ = Occlusion::default();
        occ.rebuild(ChunkBox::new(origin, 1, 1), origin, |c| {
            (c == origin).then_some(Connectivity::OPEN)
        });
        assert_eq!(occ.visible_count(), 1);
        assert!(occ.is_visible(origin));
    }

    #[test]
    fn sealed_neighbour_is_drawn_but_not_passed_through() {
        // origin OPEN; its +X neighbour SEALED; a chunk beyond that.
        let a = ChunkCoord::new(0, 0, 0);
        let b = ChunkCoord::new(1, 0, 0); // sealed wall
        let beyond = ChunkCoord::new(2, 0, 0);
        let mut occ = Occlusion::default();
        occ.rebuild(ChunkBox::new(a, 3, 3), a, |c| {
            if c == a {
                Some(Connectivity::OPEN)
            } else if c == b {
                Some(Connectivity::SEALED)
            } else if c == beyond {
                Some(Connectivity::OPEN)
            } else {
                None
            }
        });
        assert!(occ.is_visible(a) && occ.is_visible(b), "the wall chunk itself is still drawn");
        assert!(!occ.is_visible(beyond), "sightline can't pass through the sealed wall");
    }

    #[test]
    fn outside_the_box_reports_visible() {
        let origin = ChunkCoord::new(0, 0, 0);
        let mut occ = Occlusion::default();
        occ.rebuild(ChunkBox::new(origin, 1, 1), origin, |c| {
            (c.ring(origin) <= 1 && c.updown(origin) <= 1).then_some(Connectivity::OPEN)
        });
        assert!(occ.is_visible(ChunkCoord::new(8, 0, 0)));
    }
}
