//! The rule's trees on the world: a round world's band-0 charts (each a tree in its own patch
//! coordinates), a cube's six faces, or a flat world's endless +Y plane, every node's box placed
//! where it really is.

use voxel_engine::{DVec3, LodMorph};

use super::select::{Leaf, Surface};
use super::{PhysBox, Rule};
use crate::coord::Face;
use crate::ident::Detail;
use crate::render_config::LOD_COARSEST_DETAIL;
use crate::space::FaceFrame;
use crate::space::atlas::FACES;
use crate::space::chart::{Map, basis};
use crate::world::World;
use crate::world::chunk::CHUNK_SIZE;
use crate::world::section::{CHART_BODY_BASE, SectionPos, section_fits, section_span};
use crate::world::{FastMap, terrain};

/// The ground the far field stands on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::world) enum Ground {
    /// A round world's atlas: its six band-0 patches.
    Chart { index: usize },
    /// A cube body: its six faces.
    Cube { body: u16 },
    /// A flat world's +Y plane.
    Flat,
}

/// The generator bounds of nodes the rule measured. The generator never changes them, so they are
/// kept until the memo grows past [`MEMO_CAP`].
#[derive(Default)]
pub(in crate::world) struct RuleMemo {
    bounds: FastMap<SectionPos, Option<(f64, f64)>>,
}

const MEMO_CAP: usize = 1 << 17;

/// The world as the rule's [`Surface`] on one ground.
struct WorldSurface<'a> {
    world: &'a World,
    ground: Ground,
    memo: &'a mut RuleMemo,
}

impl World {
    /// The rule at this view on `ground`. A chart's coarsest tile is the largest that keeps its
    /// chord within a block and divides its patches, so every level lines up with every chart edge.
    pub(in crate::world) fn rule_on(&self, ground: Ground) -> Rule {
        let rule = Rule::new(f64::from(self.view.horizontal * CHUNK_SIZE as i32), LOD_COARSEST_DETAIL as i8);
        let Ground::Chart { index } = ground else { return rule };
        let Some(atlas) = self.seams.atlases().get(index) else { return rule };
        let n = atlas.bands[0].n;
        let fits = |d: i8| {
            let span = section_span(Detail(d));
            section_fits(span, atlas.radius) && n % i64::from(span) == 0
        };
        let cap = (0..=rule.coarsest).take_while(|&d| fits(d)).last().unwrap_or(0);
        rule.capped(cap)
    }

    /// The rule's cut on `ground` from physical eye `eye` (measured from the cell that holds it),
    /// into `out`. Returns the rule it took.
    pub(in crate::world) fn rule_cut(&self, ground: Ground, eye: DVec3, memo: &mut RuleMemo, out: &mut Vec<Leaf>) -> Rule {
        let rule = self.rule_on(ground);
        if memo.bounds.len() > MEMO_CAP {
            memo.bounds.clear();
        }
        let mut surface = WorldSurface { world: self, ground, memo };
        rule.select(&PhysBox::eye_cell(eye), &mut surface, &mut Vec::new(), out);
        rule
    }
}

impl WorldSurface<'_> {
    /// Roots of a chart: on each band-0 patch the coarsest tiles of the window around where the
    /// eye's direction falls, as wide as the horizon's arc (twice, for the projection's stretch).
    fn chart_roots(&self, index: usize, rule: &Rule, eye: DVec3, out: &mut Vec<SectionPos>) {
        let Some(atlas) = self.world.seams.atlases().get(index) else { return };
        let body = CHART_BODY_BASE + index as u16;
        let n = atlas.bands[0].n;
        let span = i64::from(section_span(Detail(rule.coarsest)));
        let tiles = n / span;
        let rel = eye - atlas.centre;
        let radius = atlas.radius.max(1) as f64;
        // The horizon's arc on the datum sphere, and the cells it spans on a patch (a cell is a
        // quarter turn over `n` at the patch centre, down to about 1/√2 of that at a corner).
        let theta = (rule.horizon() / radius).min(std::f64::consts::PI);
        let dir = rel.normalize_or_zero();
        for face in FACES {
            let all = || (0..tiles).flat_map(move |z| (0..tiles).map(move |x| (x, z)));
            let (tu, nn, tv) = basis(face);
            let d = DVec3::new(dir.dot(tu), dir.dot(nn), dir.dot(tv));
            let window: Box<dyn Iterator<Item = (i64, i64)>> = if theta >= std::f64::consts::FRAC_PI_4 || dir == DVec3::ZERO {
                Box::new(all())
            } else if d.y <= 0.0 {
                // Every point of a patch is at least 45° from a direction behind its normal.
                continue;
            } else {
                let (xi, eta) = Map::Equiangular.inverse(d);
                let cell = std::f64::consts::FRAC_PI_2 / n as f64;
                let reach = 2.0 * theta / cell + span as f64;
                let (lx, lz) = ((xi + 1.0) * 0.5 * n as f64, (eta + 1.0) * 0.5 * n as f64);
                let range = |c: f64| {
                    let lo = ((c - reach) / span as f64).floor().max(0.0) as i64;
                    let hi = ((c + reach) / span as f64).floor().min((tiles - 1) as f64) as i64;
                    lo..=hi
                };
                let (rx, rz) = (range(lx), range(lz));
                Box::new(rz.flat_map(move |z| rx.clone().map(move |x| (x, z))))
            };
            for (x, z) in window {
                out.push(SectionPos { detail: Detail(rule.coarsest), body, face, x: x as i32, z: z as i32 });
            }
        }
    }

    /// Roots of a cube: on each face the coarsest tiles over the face's square that the horizon
    /// can reach, on the face's own grid.
    fn cube_roots(&self, body: u16, rule: &Rule, eye: DVec3, out: &mut Vec<SectionPos>) {
        let Some(cosmos) = self.world.generator.cosmos() else { return };
        let Some(b) = cosmos.bodies().iter().find(|b| b.id == body) else { return };
        let terrain::cosmos::Shape::Cube { half } = b.shape else { return };
        let warp = self
            .world
            .seams
            .atlases()
            .iter()
            .find(|a| a.grid.as_ref().is_some_and(|g| g.body == body))
            .and_then(|a| a.warp.as_ref().map(|w| w.max_displacement()))
            .unwrap_or(0.0);
        let reach = rule.horizon() + warp;
        let span = f64::from(section_span(Detail(rule.coarsest)));
        let lim = (half + terrain::cosmos::RELIEF) as f64;
        for face in Face::ALL {
            let frame = FaceFrame::new(face);
            let c = frame.point_to_local(b.centre_f());
            let e = frame.point_to_local(eye);
            // Too far below or above the face to reach it.
            if e.y - (c.y + lim) >= reach || (c.y - lim) - e.y >= reach {
                continue;
            }
            let range = |ec: f64, cc: f64| {
                let lo = ((ec - reach).max(cc - lim) / span).floor() as i64;
                let hi = ((ec + reach).min(cc + lim) / span).floor() as i64;
                lo..=hi
            };
            let (ru, rv) = (range(e.x, c.x), range(e.z, c.z));
            for z in rv {
                for x in ru.clone() {
                    let (Ok(x), Ok(z)) = (i32::try_from(x), i32::try_from(z)) else { continue };
                    out.push(SectionPos { detail: Detail(rule.coarsest), body, face, x, z });
                }
            }
        }
    }

    /// Roots of a flat world: the coarsest tiles within the horizon of the eye.
    fn flat_roots(&self, rule: &Rule, eye: DVec3, out: &mut Vec<SectionPos>) {
        let span = f64::from(section_span(Detail(rule.coarsest)));
        let reach = rule.horizon();
        let range = |c: f64| ((c - reach) / span).floor() as i32..=((c + reach) / span).floor() as i32;
        for z in range(eye.z) {
            for x in range(eye.x) {
                out.push(SectionPos { detail: Detail(rule.coarsest), body: 0, face: Face::PosY, x, z });
            }
        }
    }
}

impl Surface for WorldSurface<'_> {
    fn roots(&self, rule: &Rule, eye: DVec3, out: &mut Vec<SectionPos>) {
        match self.ground {
            Ground::Chart { index } => self.chart_roots(index, rule, eye, out),
            Ground::Cube { body } => self.cube_roots(body, rule, eye, out),
            Ground::Flat => self.flat_roots(rule, eye, out),
        }
    }

    fn node_box(&mut self, s: SectionPos) -> Option<PhysBox> {
        let (lo, hi) = match self.world.baked_bounds(s) {
            Some(b) => b,
            None => (*self.memo.bounds.entry(s).or_insert_with(|| self.world.generator_bounds(s)))?,
        };
        self.world.place_box(s, lo, hi)
    }
}

impl World {
    /// The roots of `ground` seen from `eye`, how many have a box, and the nearest box: a test's
    /// report when a cut comes back empty.
    #[cfg(test)]
    pub(in crate::world) fn rule_roots_debug(&self, ground: Ground, eye: DVec3) -> String {
        let rule = self.rule_on(ground);
        let mut memo = RuleMemo::default();
        let mut surface = WorldSurface { world: self, ground, memo: &mut memo };
        let mut roots = Vec::new();
        surface.roots(&rule, eye, &mut roots);
        let cell = PhysBox::eye_cell(eye);
        let boxes: Vec<(SectionPos, PhysBox)> = roots.iter().filter_map(|&s| Some((s, surface.node_box(s)?))).collect();
        let nearest = boxes.iter().map(|(s, b)| (b.dist_box(&cell), *s, *b)).min_by(|a, b| a.0.total_cmp(&b.0));
        format!("{rule:?}: {} roots, {} with boxes, nearest {nearest:?}", roots.len(), boxes.len())
    }
}

/// Blocks a band keeps clear of the distances it was derived from: vertex rounding and the cage's
/// float error.
const BAND_SLACK: f64 = 1.0;

/// Mesh detail is section detail plus the nesting shift (a 32-column section packs one bit).
const MESH_SHIFT: usize = 1;

impl World {
    /// Whether tile `s` is meshed with `dy`: a level with a coarser one beside it. The rule's
    /// coarsest level has none.
    pub(in crate::world) fn section_morphs(&self, s: SectionPos) -> bool {
        let ground = if s.body >= CHART_BODY_BASE {
            Ground::Chart { index: usize::from(s.body - CHART_BODY_BASE) }
        } else if self.generator.cosmos().is_some() {
            Ground::Cube { body: s.body }
        } else {
            Ground::Flat
        };
        s.detail.0 < self.rule_on(ground).coarsest
    }

    /// The morph band of every level of the cut `leaves` seen from the eye's cell `eye`, by engine
    /// detail. A level's vertices must be unmoved where a finer level meets it and fully on their
    /// parent's surface where a coarser one does. A vertex next to a finer tile lies in the split
    /// node that tile came from, so it is no farther from the camera (anywhere in the cell) than
    /// that node's distance, its diameter and the cell's; next to the full-res chunks, no farther
    /// than the near columns reach. A vertex on a coarser tile is no nearer than that tile. A level
    /// whose two bounds cross does not morph (its skirts seal the step).
    pub(in crate::world) fn morph_bands(&self, rule: &Rule, leaves: &[Leaf], eye: &PhysBox) -> [LodMorph; 16] {
        let cell = (eye.hi - eye.lo).length();
        let (mut inner, mut outer) = ([0.0f64; 16], [f64::INFINITY; 16]);
        let mut split: Vec<SectionPos> = leaves.iter().filter(|l| l.pos.detail.0 < rule.coarsest).map(|l| l.pos.parent()).collect();
        split.sort_unstable_by_key(|s| (s.body, s.face as u8, s.detail, s.x, s.z));
        split.dedup();
        for p in split {
            let d = p.detail.0 as usize;
            if let Some(b) = self.node_box(p) {
                inner[d] = inner[d].max(b.dist_box(eye) + (b.hi - b.lo).length() + cell);
            }
        }
        for l in leaves {
            let d = l.pos.detail.0 as usize;
            if d > 0 {
                outer[d - 1] = outer[d - 1].min(l.dist);
            }
        }
        let finest = rule.finest.max(0) as usize;
        if self.near.active() {
            inner[finest] = inner[finest].max(self.near.reach + cell);
        }
        let mut bands = [LodMorph::default(); 16];
        for d in 0..16 - MESH_SHIFT {
            let (start, end) = (inner[d] + BAND_SLACK, outer[d] - BAND_SLACK);
            if end.is_finite() && end > start && (d as i8) < rule.coarsest {
                bands[d + MESH_SHIFT] = LodMorph { start: start as f32, end: end as f32 };
            }
        }
        bands
    }
}
