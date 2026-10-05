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
use super::chunk::Chunk;
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
        let mut rows = [0u16; ROWS];
        match &chunk.data().payload {
            // Uniform chunks need no scan: opaque seals everything, see-through opens it.
            ChunkPayload::Uniform(v) => {
                return if blocks_sight(v.id) { Self::SEALED } else { Self::OPEN };
            }
            // One classify per palette entry up front; the cell pass then reads a
            // bit per cell instead of re-classifying ids.
            ChunkPayload::Paletted { palette, cells } => {
                let mut open = [0u16; super::brick::PALETTE_MAX];
                for (o, p) in open.iter_mut().zip(palette.iter()) {
                    *o = u16::from(!blocks_sight(p.id));
                }
                for (i, &c) in cells.iter().enumerate() {
                    rows[i >> 4] |= open[c as usize] << (i & 15);
                }
            }
            ChunkPayload::Dense(cells) => {
                for (i, c) in cells.iter().enumerate() {
                    rows[i >> 4] |= u16::from(!blocks_sight(c.id)) << (i & 15);
                }
            }
        }
        Self::flood(rows)
    }

    /// The pocket flood over passable rows (`rows[z + 16y]`, bit x). A pocket takes whole x runs
    /// and steps to the rows beside each new run, so every run is visited once.
    fn flood(mut left: [u16; ROWS]) -> Connectivity {
        let mut conn = Connectivity::SEALED;
        // A push takes at least one run out of `left`, and a row holds at most eight.
        let mut stack = [(0u8, 0u16); ROWS * 8];
        for start in 0..ROWS {
            while left[start] != 0 {
                let run = x_runs(left[start], left[start] & left[start].wrapping_neg());
                left[start] &= !run;
                stack[0] = (start as u8, run);
                let mut len = 1;
                let mut faces = 0u8;
                while len > 0 {
                    len -= 1;
                    let (r, bits) = stack[len];
                    let r = usize::from(r);
                    faces |= row_faces(r, bits);
                    let (z, y) = (r & 15, r >> 4);
                    let beside = [
                        (r.wrapping_sub(1), z > 0),
                        (r + 1, z < 15),
                        (r.wrapping_sub(16), y > 0),
                        (r + 16, y < 15),
                    ];
                    for (n, inside) in beside {
                        if !inside {
                            continue;
                        }
                        let touch = bits & left[n];
                        if touch != 0 {
                            let run = x_runs(left[n], touch);
                            left[n] &= !run;
                            stack[len] = (n as u8, run);
                            len += 1;
                        }
                    }
                }
                conn.add_pocket(faces);
            }
        }
        conn
    }
}

/// One passable mask per (y, z) row of a chunk.
const ROWS: usize = super::chunk::CHUNK_SIZE * super::chunk::CHUNK_SIZE;

/// The runs of set bits in `row` that hold a bit of `seed` (`seed ⊆ row`). A carry sweeps up each
/// run from its lowest seed, the bit-reversed row sweeps down from its highest, and seeds between
/// them are the seed bits themselves.
fn x_runs(row: u16, seed: u16) -> u16 {
    let up = |r: u16, s: u16| {
        let r = u32::from(r);
        (((r + u32::from(s)) ^ r) & r) as u16
    };
    up(row, seed) | up(row.reverse_bits(), seed.reverse_bits()).reverse_bits() | seed
}

/// Chunk faces a non-empty run of row `r` touches.
fn row_faces(r: usize, bits: u16) -> u8 {
    const EDGE: usize = super::chunk::CHUNK_SIZE - 1;
    let (z, y) = (r & EDGE, r >> 4);
    let mut m = 0u8;
    if bits & 1 != 0 {
        m |= 1 << Face::NegX as usize;
    }
    if bits >> EDGE != 0 {
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

/// Dense occupancy for the occlusion BFS: bit 6 is visible, bits 0–5 are the
/// entry faces already queued (expansion is tracked in [`Occlusion::exits`]).
/// A loaded chunk outside the current unload box reports visible so it is
/// never culled.
const VISIBLE_BIT: u8 = 1 << 6;

/// Marks a loaded cell in [`Occlusion::conn`]; the low 15 bits are its [`Connectivity`].
const LOADED_BIT: u16 = 1 << 15;

/// Every face in [`Occlusion::exits`].
const ALL_EXITS: u8 = (1 << 6) - 1;

/// The occlusion pass: determines which chunks are visible from the camera.
/// Rebuilt into a dense byte grid covering the unload box — one array lookup
/// per query, no per-chunk hashing.
pub struct Occlusion {
    origin: ChunkCoord,
    nx: i32,
    ny: i32,
    nz: i32,
    cells: Vec<u8>,
    /// Loaded chunks' connectivity over the same grid, filled once per rebuild so the BFS
    /// never hashes.
    conn: Vec<u16>,
    /// Exit faces each cell has already pushed: a second entry face re-opens the same exits.
    exits: Vec<u8>,
    /// Frontier of (chunk, entry face), each pair queued once.
    queue: Vec<(ChunkCoord, Face)>,
}

impl Default for Occlusion {
    fn default() -> Self {
        Self {
            origin: ChunkCoord::new(0, 0, 0),
            nx: 0,
            ny: 0,
            nz: 0,
            cells: Vec::new(),
            conn: Vec::new(),
            exits: Vec::new(),
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
    /// `loaded` lists every loaded chunk with its connectivity; those outside `volume`
    /// are ignored.
    pub fn rebuild(
        &mut self,
        volume: ChunkBox,
        origin: ChunkCoord,
        loaded: impl IntoIterator<Item = (ChunkCoord, Connectivity)>,
    ) {
        let (nx, ny, nz) = volume.size();
        let n = (nx * ny * nz) as usize;
        self.nx = nx;
        self.ny = ny;
        self.nz = nz;
        self.origin = volume.min();
        self.cells.clear();
        self.cells.resize(n, 0);
        self.conn.clear();
        self.conn.resize(n, 0);
        self.exits.clear();
        self.exits.resize(n, 0);
        for (coord, conn) in loaded {
            if let Some(idx) = self.index(coord) {
                self.conn[idx] = LOADED_BIT | conn.0;
            }
        }
        self.queue.clear();
        let Some(root) = self.index(origin) else {
            return;
        };
        // The camera's own chunk is the root: always expanded, and seen out of every face (it is
        // generated synchronously, so it is loaded in practice).
        self.cells[root] |= VISIBLE_BIT;
        self.exits[root] = ALL_EXITS;
        for exit in Face::ALL {
            self.enter(origin.step(exit), exit.opposite());
        }
        while let Some((coord, entry)) = self.queue.pop() {
            let idx = self.index(coord).expect("queued cells are in the box");
            let pushed = self.exits[idx];
            if pushed == ALL_EXITS {
                continue;
            }
            let conn = Connectivity(self.conn[idx] & !LOADED_BIT);
            for exit in Face::ALL {
                let bit = 1u8 << exit as usize;
                // NOTE: per-chunk connectivity is conservative—we don't check if
                // the shared boundary is actually open, which can over-report
                // visibility. This never culls a visible chunk.
                if pushed & bit == 0 && conn.connects(entry, exit) {
                    self.exits[idx] |= bit;
                    self.enter(coord.step(exit), exit.opposite());
                }
            }
        }
    }

    /// Queue `coord` entered through `face` once. A chunk that isn't loaded is the frontier: it
    /// is not drawn and must NOT be propagated through, or the BFS would flood outward across
    /// infinite empty space and never terminate.
    #[inline]
    fn enter(&mut self, coord: ChunkCoord, face: Face) {
        let Some(idx) = self.index(coord) else {
            return;
        };
        let bit = 1u8 << face as usize;
        let cell = self.cells[idx];
        if cell & bit != 0 || self.conn[idx] & LOADED_BIT == 0 {
            return;
        }
        self.cells[idx] = cell | bit | VISIBLE_BIT;
        self.queue.push((coord, face));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::AIR;
    use crate::world::chunk::CHUNK_VOLUME;

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

    /// The cell-by-cell pocket fill the row flood replaced: the reference it must match.
    fn cell_flood(passable: impl Fn(usize) -> bool) -> Connectivity {
        let mut visited = vec![false; CHUNK_VOLUME];
        let mut conn = Connectivity::SEALED;
        for start in 0..CHUNK_VOLUME {
            if visited[start] || !passable(start) {
                continue;
            }
            visited[start] = true;
            let mut stack = vec![start];
            let mut faces = 0u8;
            while let Some(i) = stack.pop() {
                let (x, y, z) = Chunk::local_of(i);
                for (axis, v) in [(0, x), (1, y), (2, z)] {
                    if v == 0 {
                        faces |= 1 << [Face::NegX, Face::NegY, Face::NegZ][axis] as usize;
                    }
                    if v == 15 {
                        faces |= 1 << [Face::PosX, Face::PosY, Face::PosZ][axis] as usize;
                    }
                }
                let steps = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)];
                for (dx, dy, dz) in steps {
                    let (nx, ny, nz) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
                    if !(0..16).contains(&nx) || !(0..16).contains(&ny) || !(0..16).contains(&nz) {
                        continue;
                    }
                    let j = Chunk::index(nx as usize, ny as usize, nz as usize);
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

    /// Random chunks of every density, a few structured ones, and the paletted and dense
    /// payloads: the row flood finds the same face pairs as the cell flood.
    #[test]
    fn row_flood_matches_the_cell_flood() {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let opaque = |id: BlockId| id.0 % 2 == 1;
        for round in 0..300u64 {
            let density = round % 10;
            let ids = if round % 7 == 0 { 400 } else { 4 };
            let structured = round % 5;
            let chunk = dense(|x, y, z| {
                let r = next();
                let solid = match structured {
                    0 => (x + y + z) % 2 == 0,
                    1 => x == 8 || (y == 3 && z != 5),
                    _ => r % 10 < density,
                };
                let base = (r >> 16) % ids * 2;
                BlockId((base + u64::from(solid)) as u16)
            });
            let data = chunk.data();
            let get = |i: usize| match &data.payload {
                ChunkPayload::Uniform(v) => v.id,
                ChunkPayload::Paletted { palette, cells } => palette[cells[i] as usize].id,
                ChunkPayload::Dense(cells) => cells[i].id,
            };
            assert_eq!(
                Connectivity::compute(&chunk, opaque),
                cell_flood(|i| !opaque(get(i))),
                "round {round}"
            );
        }
    }

    /// The carry sweep keeps exactly the runs a seed touches.
    #[test]
    fn x_runs_are_the_runs_a_seed_touches() {
        let smear = |row: u16, seed: u16| {
            let mut r = seed;
            loop {
                let n = (r | r << 1 | r >> 1) & row;
                if n == r {
                    return r;
                }
                r = n;
            }
        };
        let mut seed = 0x9E37_79B9u64;
        for row in 0..=u16::MAX {
            for _ in 0..4 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let s = row & seed as u16;
                assert_eq!(x_runs(row, s), smear(row, s), "row {row:#06x} seed {s:#06x}");
            }
            if row != 0 {
                let low = row & row.wrapping_neg();
                assert_eq!(x_runs(row, low), smear(row, low));
            }
        }
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
        occ.rebuild(ChunkBox::new(origin, 2, 2), origin, loaded.iter().map(|&c| (c, Connectivity::OPEN)));
        assert_eq!(occ.visible_count(), loaded.len());
        assert!(loaded.iter().all(|&c| occ.is_visible(c)));
    }

    #[test]
    fn unloaded_neighbours_bound_the_bfs() {
        // Only the origin is loaded (OPEN). The frontier must stop at every
        // unloaded neighbour instead of flooding outward forever.
        let origin = ChunkCoord::new(5, -3, 2);
        let mut occ = Occlusion::default();
        occ.rebuild(ChunkBox::new(origin, 1, 1), origin, [(origin, Connectivity::OPEN)]);
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
        occ.rebuild(
            ChunkBox::new(a, 3, 3),
            a,
            [(a, Connectivity::OPEN), (b, Connectivity::SEALED), (beyond, Connectivity::OPEN)],
        );
        assert!(occ.is_visible(a) && occ.is_visible(b), "the wall chunk itself is still drawn");
        assert!(!occ.is_visible(beyond), "sightline can't pass through the sealed wall");
    }

    #[test]
    fn outside_the_box_reports_visible() {
        let origin = ChunkCoord::new(0, 0, 0);
        let mut occ = Occlusion::default();
        let near = ChunkBox::with_up(origin, 1, 1, None).coords();
        occ.rebuild(ChunkBox::new(origin, 1, 1), origin, near.map(|c| (c, Connectivity::OPEN)));
        assert!(occ.is_visible(ChunkCoord::new(8, 0, 0)));
    }

    /// The grid BFS reaches exactly what a hash-set BFS over the same rules reaches. Loaded
    /// chunks outside the box are ignored, and an unloaded root still sees out of every face.
    #[test]
    fn grid_bfs_matches_a_set_bfs() {
        use std::collections::{HashMap, HashSet};
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for round in 0..40 {
            let origin = ChunkCoord::new(3, -2, 7);
            let volume = ChunkBox::new(origin, 4, 2);
            let mut loaded: HashMap<ChunkCoord, Connectivity> = HashMap::new();
            for x in -6..=6 {
                for y in -4..=4 {
                    for z in -6..=6 {
                        let c = ChunkCoord::new(origin.x + x, origin.y + y, origin.z + z);
                        let r = next();
                        if r % 5 == 0 || (round % 4 == 0 && c == origin) {
                            continue;
                        }
                        let conn = match r % 3 {
                            0 => Connectivity::OPEN,
                            1 => Connectivity::SEALED,
                            _ => Connectivity((r >> 8) as u16 & 0x7FFF),
                        };
                        loaded.insert(c, conn);
                    }
                }
            }
            let mut occ = Occlusion::default();
            occ.rebuild(volume, origin, loaded.iter().map(|(&c, &k)| (c, k)));
            let mut seen: HashSet<(ChunkCoord, Option<Face>)> = HashSet::new();
            let mut visible: HashSet<ChunkCoord> = HashSet::new();
            let mut stack = vec![(origin, None)];
            while let Some((c, entry)) = stack.pop() {
                if !volume.contains(c) || !seen.insert((c, entry)) {
                    continue;
                }
                let conn = match (entry, loaded.get(&c)) {
                    (None, k) => k.copied().unwrap_or(Connectivity::OPEN),
                    (Some(_), Some(&k)) => k,
                    (Some(_), None) => continue,
                };
                visible.insert(c);
                for exit in Face::ALL {
                    if entry.is_none_or(|e| conn.connects(e, exit)) {
                        stack.push((c.step(exit), Some(exit.opposite())));
                    }
                }
            }
            for c in volume.coords() {
                assert_eq!(occ.is_visible(c), visible.contains(&c), "round {round} at {c:?}");
            }
        }
    }
}
