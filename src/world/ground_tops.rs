//! The ground a column shows from above where none of its loaded chunks holds a solid block: the
//! generator's top of that column. The near field loads only the columns within the render
//! distance (and the eye band across its square), so the minimap's square reaches columns whose
//! surface was never loaded; those read the generator, in the near field's frame, on a sampler
//! thread. A column waits (void) until its tops land, and is repainted on the next refresh.

use voxel_engine::Color;

use super::chunk::CHUNK_SIZE;
use super::generation::TerrainGenerator;
use super::sampler::{Recent, Sampler};
use super::streaming::Frame;
use super::{ColumnKey, Coord, World};
use crate::block::registry::{AIR, BlockId};
use crate::coord::Face;
use crate::space::FaceFrame;

/// A chunk column's tops, by `[lv][lu]`: the altitude of its top solid cell along the face and
/// that cell's block, or `None` where the generator holds no ground.
type ColumnTops = [[Option<(i32, BlockId)>; CHUNK_SIZE]; CHUNK_SIZE];

/// A real chunk column of a face, in the frame it was read in.
type Key = (Frame, Face, i32, i32);

/// Columns kept: four minimap squares.
const TOPS_CAP: usize = 1 << 10;

/// Most columns one call keeps from the sampler.
const LAND_MAX: usize = 64;

/// The sampler of column tops, what it answered, and a count of the columns that landed.
pub(in crate::world) struct GroundTops {
    sampler: Option<Sampler<Key, Box<ColumnTops>>>,
    tops: Recent<Key, Box<ColumnTops>>,
    rev: u64,
}

impl Default for GroundTops {
    fn default() -> Self {
        GroundTops { sampler: None, tops: Recent::with_cap(TOPS_CAP), rev: 0 }
    }
}

impl GroundTops {
    fn land(&mut self) {
        let GroundTops { sampler, tops, rev } = self;
        if let Some(sampler) = sampler {
            sampler.land(LAND_MAX, |k, t| {
                tops.insert(k, t);
                *rev += 1;
            });
        }
    }
}

/// The generator's tops of real chunk column `(cu, cv)` of `face` in `frame`: a chart's or a flat
/// world's storage +Y column, or a warped cube's storage column read at its reference cells.
fn column_tops(generator: &dyn TerrainGenerator, frame: Frame, face: Face, cu: i32, cv: i32) -> Box<ColumnTops> {
    let mut out: Box<ColumnTops> = Box::new([[None; CHUNK_SIZE]; CHUNK_SIZE]);
    let s = CHUNK_SIZE as i32;
    match frame {
        Frame::Chart { .. } | Frame::Flat => {
            if face != Face::PosY {
                return out;
            }
            let heights = generator.heights_16(cu, cv);
            for (lv, row) in out.iter_mut().enumerate() {
                for (lu, slot) in row.iter_mut().enumerate() {
                    let (x, z) = (cu * s + lu as i32, cv * s + lv as i32);
                    let block = generator.surface_at(x, z);
                    if block != AIR {
                        *slot = Some((heights[lu + lv * CHUNK_SIZE] - 1, block));
                    }
                }
            }
        }
        Frame::Cube { body, face: up, shift } => {
            if face != up {
                return out;
            }
            let f = FaceFrame::new(face);
            let c = ColumnKey { face, a: cu, b: cv }.chunk(0);
            let (ru, ra, rv) = f.chunk_to_local(Coord::new(c.x + shift[0], c.y + shift[1], c.z + shift[2]));
            let mut block = [AIR];
            for (lv, row) in out.iter_mut().enumerate() {
                for (lu, slot) in row.iter_mut().enumerate() {
                    let (u, v) = (ru * s + lu as i32, rv * s + lv as i32);
                    let a = generator.surface(face, u, v);
                    if a == i32::MIN {
                        continue;
                    }
                    generator.lod_column_face(body, face, u, v, &[a - 1], &mut block);
                    if block[0] != AIR {
                        *slot = Some((a - 1 - ra * s, block[0]));
                    }
                }
            }
        }
    }
    out
}

impl World {
    /// Fill the slots of `out` (rows `vs`, columns `us`) that are still `None` from the generator's
    /// tops of chunk column `(cu, cv)` of `face`, once they have landed; until then ask for them.
    /// Nothing outside a near field's frame.
    pub(in crate::world) fn fill_ground_tops(
        &self,
        face: Face,
        (cu, cv): (i32, i32),
        us: std::ops::Range<usize>,
        vs: std::ops::Range<usize>,
        out: &mut [[Option<(i32, Color)>; CHUNK_SIZE]; CHUNK_SIZE],
    ) {
        let Some(frame) = self.near.frame() else { return };
        let folded = match frame {
            Frame::Chart { floor, .. } => Coord::new(cu, floor, cv),
            _ => ColumnKey { face, a: cu, b: cv }.chunk(0),
        };
        let Some(real) = self.fold.unfold(folded) else { return };
        let (real, _) = ColumnKey::of(face, real);
        let key = (frame, face, real.a, real.b);
        let mut g = self.ground_tops.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if g.sampler.is_none() {
            let generator = self.generator.clone();
            g.sampler = Sampler::start("ground-tops", move |(frame, face, a, b)| column_tops(&*generator, frame, face, a, b));
        }
        g.land();
        if let Some(tops) = g.tops.touch(&key) {
            for lv in vs {
                for lu in us.clone() {
                    if out[lv][lu].is_none()
                        && let Some((alt, id)) = tops[lv][lu]
                    {
                        out[lv][lu] = Some((alt, self.registry.color(id)));
                    }
                }
            }
            return;
        }
        if let Some(sampler) = &mut g.sampler
            && sampler.mark(key)
        {
            sampler.send(vec![key]);
        }
    }

    /// How many generator columns have landed: a change means a column read from the generator
    /// may now show where it was void.
    pub(in crate::world) fn ground_tops_rev(&self) -> u64 {
        let mut g = self.ground_tops.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        g.land();
        g.rev
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_config::RenderConfig;
    use crate::world::fixtures::spawned_round_world;
    use crate::world::streaming::headless::step_finished;

    /// Every column of the minimap's square shows ground once the generator's tops have landed,
    /// though the near field loads far less than the square, and where the ground is loaded the
    /// generator's tops stand where the loaded ones do.
    #[test]
    fn every_column_of_the_map_square_shows_ground() {
        let (mut world, spawn) = spawned_round_world(RenderConfig { lod2: true, ..RenderConfig::default() }, 4, 3);
        for pass in 0.. {
            if world.loading_full() && world.entry_complete() {
                break;
            }
            assert!(pass < 20_000, "did not settle");
            step_finished(&mut world, spawn);
        }
        let storage = world.chart_eye(spawn).expect("a chart eye");
        let s = CHUNK_SIZE as i32;
        let (cx, cz) = ((storage.x.floor() as i32).div_euclid(s), (storage.z.floor() as i32).div_euclid(s));
        let all = 0..CHUNK_SIZE;
        let mut tops = [[None; CHUNK_SIZE]; CHUNK_SIZE];
        let columns = || (cx - 8..cx + 8).flat_map(move |a| (cz - 8..cz + 8).map(move |b| (a, b)));
        let start = std::time::Instant::now();
        let mut void = usize::MAX;
        while void > 0 {
            assert!(start.elapsed().as_secs() < 60, "{void} texels still void");
            void = 0;
            for c in columns() {
                world.top_solids_on_face(Face::PosY, c, all.clone(), all.clone(), &mut tops);
                void += tops.iter().flatten().filter(|t| t.is_none()).count();
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(world.ground_tops_rev() > 0, "the near field loaded the whole square: nothing was read");
        let (mut same, mut compared) = (0, 0);
        let frame = world.near.frame().expect("a near frame");
        for (a, b) in columns() {
            let Some((real, _)) = world.near.near_column(Coord::new(a, 0, b)) else { continue };
            let read = column_tops(&*world.generator, frame, Face::PosY, real.a, real.b);
            world.top_solids_on_face(Face::PosY, (a, b), all.clone(), all.clone(), &mut tops);
            for (row, gen_row) in tops.iter().zip(read.iter()) {
                for (t, g) in row.iter().zip(gen_row) {
                    if let (Some((alt, _)), Some((g_alt, _))) = (t, g) {
                        compared += 1;
                        same += usize::from(alt == g_alt);
                    }
                }
            }
        }
        println!("{} columns read from the generator; {same} of {compared} loaded tops match its", world.ground_tops_rev());
        assert!(compared > 0 && same * 2 > compared, "{same} of {compared} loaded tops match the generator's");
    }
}
