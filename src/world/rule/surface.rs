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
use crate::world::FastSet;
use crate::world::sampler::{Recent, Sampler};
use crate::world::terrain;

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

/// The physical boxes of nodes the rule measured, and the tight generator bounds under them. A box
/// changes when its node's bounds do (a relief bake lands, an edit's overlay is re-derived, its
/// tight bounds land), which drops it. Tight bounds are pure and stay. Both keep what the cuts used
/// lately.
///
/// The cut never samples terrain: a node up to [`super::TIGHT_SPAN`] whose tight bounds are not in
/// yet stands on its body's relief band, the tiles under it are asked of a sampler thread, and it
/// does not split until they land (like a tile not yet built, its detail arrives with its data).
/// When they land, its box goes, and [`landed`](Self::landed) tells the frontier to cut again.
#[derive(Default)]
pub(in crate::world) struct RuleMemo {
    pub(in crate::world) boxes: Recent<SectionPos, Option<PhysBox>>,
    /// Tight generator bounds per node up to [`super::TIGHT_SPAN`]: sampled up to
    /// [`super::SAMPLED_SPAN`], the union of its quarters' above.
    pub(in crate::world) ground: Recent<SectionPos, Option<(f64, f64)>>,
    sampler: Option<Sampler<SectionPos, Option<(f64, f64)>>>,
    /// Tiles marked asked and not yet sent.
    asks: Vec<SectionPos>,
    /// Nodes whose box stands on the band until the tiles under them land.
    provisional: FastSet<SectionPos>,
    /// Bumped when tight bounds land under a node whose box stood on the band.
    pub(in crate::world) landed: u64,
    /// Leaves whose children were fetched ahead (a cut mostly holds the leaves the last one did).
    pub(in crate::world) fetched: Recent<SectionPos, ()>,
    /// A chart patch's relief band, which is the same for every tile of it.
    pub(in crate::world) bands: crate::world::FastMap<(u16, Face), Option<(f64, f64)>>,
}

/// Asks sent to the sampler at once.
const ASK_BATCH: usize = 256;

impl RuleMemo {
    /// Forget the boxes; the generator's bounds stand.
    pub(in crate::world) fn clear(&mut self) {
        self.boxes.clear();
    }

    /// Ask the sampler for tile `s`'s tight bounds, unless it was asked already.
    pub(in crate::world) fn ask(&mut self, generator: &terrain::Generator, s: SectionPos) {
        if self.sampler.is_none() {
            let generator = generator.clone();
            self.sampler = Sampler::start("rule-bounds", move |s| super::sampled_bounds(&*generator, s));
        }
        let Some(sampler) = &mut self.sampler else { return };
        if sampler.mark(s) {
            self.asks.push(s);
            if self.asks.len() >= ASK_BATCH {
                self.flush();
            }
        }
    }

    /// Whether `s` was asked for and has not landed.
    #[cfg(test)]
    pub(in crate::world) fn asked(&self, s: &SectionPos) -> bool {
        self.sampler.as_ref().is_some_and(|p| p.asked(s))
    }

    /// Send the asks gathered so far.
    pub(in crate::world) fn flush(&mut self) {
        if let Some(sampler) = &mut self.sampler {
            sampler.send(std::mem::take(&mut self.asks));
        }
    }

    /// Keep up to `most` landed bounds. Each drops the boxes of its tile and of the nodes over it
    /// that read it, and a node whose box stood on the band makes the cut due again.
    pub(in crate::world) fn land(&mut self, most: usize) {
        let RuleMemo { boxes, ground, sampler, provisional, landed, .. } = self;
        let Some(sampler) = sampler else { return };
        sampler.land(most, |s, b| {
            if !ground.contains(&s) {
                ground.insert(s, b);
            }
            let mut p = s;
            loop {
                boxes.remove(&p);
                if provisional.remove(&p) {
                    *landed += 1;
                }
                if p.span() * 2 > super::TIGHT_SPAN {
                    break;
                }
                p = p.parent();
            }
        });
    }

    /// Tight bounds asked for and not yet landed, or a box still standing on the band.
    pub(in crate::world) fn pending(&self) -> bool {
        !self.asks.is_empty() || !self.provisional.is_empty() || self.sampler.as_ref().is_some_and(|p| p.waiting() > 0)
    }

    /// Whether the sampler has answered every tile sent to it.
    #[cfg(test)]
    pub(in crate::world) fn answered(&self) -> bool {
        self.sampler.as_ref().is_none_or(|p| p.answered())
    }

    /// Tiles asked for that have not landed.
    pub(in crate::world) fn waiting(&self) -> usize {
        self.asks.len() + self.sampler.as_ref().map_or(0, |p| p.waiting())
    }

    /// Mark node `s`'s box as standing on the band.
    pub(in crate::world) fn standing_on_band(&mut self, s: SectionPos) {
        self.provisional.insert(s);
    }

    /// Whether node `s`'s box stands on the band, its tight bounds pending.
    pub(in crate::world) fn stands_on_band(&self, s: &SectionPos) -> bool {
        self.provisional.contains(s)
    }
}

/// The world as the rule's [`Surface`] on one ground, holding the box memo for the walk.
struct WorldSurface<'a> {
    world: &'a World,
    ground: Ground,
    memo: std::sync::MutexGuard<'a, RuleMemo>,
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
    pub(in crate::world) fn rule_cut(&self, ground: Ground, eye: DVec3, out: &mut Vec<Leaf>) -> Rule {
        let rule = self.rule_on(ground);
        let memo = self.node_boxes.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut surface = WorldSurface { world: self, ground, memo };
        rule.select(&PhysBox::eye_cell(eye), &mut surface, &mut Vec::new(), out);
        surface.memo.flush();
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
        // Half the diagonal of a root in angle (a cell is at most a quarter turn over `n`, at the
        // patch centre), and the least radius of the eye or of any ground, for the horizon test.
        let footprint = std::f64::consts::FRAC_PI_2 / n as f64 * span as f64 * std::f64::consts::FRAC_1_SQRT_2 * 1.01;
        let low = rel.length().min((radius - terrain::cosmos::RELIEF as f64 - 1024.0).max(0.0));
        let wide = theta >= std::f64::consts::FRAC_PI_4;
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
                // An inward chart counts its local x the other way.
                let lx = if atlas.inward { n as f64 - lx } else { lx };
                let range = |c: f64| {
                    let lo = ((c - reach) / span as f64).floor().max(0.0) as i64;
                    let hi = ((c + reach) / span as f64).floor().min((tiles - 1) as f64) as i64;
                    lo..=hi
                };
                let (rx, rz) = (range(lx), range(lz));
                Box::new(rz.flat_map(move |z| rx.clone().map(move |x| (x, z))))
            };
            for (x, z) in window {
                // A root wholly past the horizon is not measured: the angle between the eye and
                // the root's centre, less the most its footprint subtends, bounds the chord from
                // the eye to any point of it.
                if dir != DVec3::ZERO && !wide {
                    let c = |t: i64| (t as f64 + 0.5) * span as f64;
                    let i = if atlas.inward { n as f64 - c(x) } else { c(x) };
                    let mid = Map::Equiangular.dir(-1.0 + i * 2.0 / n as f64, -1.0 + c(z) * 2.0 / n as f64);
                    let along = (tu * mid.x + nn * mid.y + tv * mid.z).normalize();
                    let angle = along.dot(dir).clamp(-1.0, 1.0).acos() - footprint;
                    if angle > 0.0 && 2.0 * low * (0.5 * angle.min(std::f64::consts::PI)).sin() >= rule.horizon() {
                        continue;
                    }
                }
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
        let warp = self.world.seams.atlases().iter().find(|a| a.grid.as_ref().is_some_and(|g| g.body == body)).and_then(|a| a.warp.as_ref());
        // Tiles are laid out on the reference cube, so a sagging cube's eye goes back through its
        // warp, and the horizon is padded by how much the displacement changes around that point
        // (sampled over twice the horizon: the field is smooth at that scale), never by more than
        // its largest displacement. A strong sag's whole displacement would ask for too many roots.
        let horizon = rule.horizon();
        let (eye, reach) = match warp.map(|w| (w, w.invert(eye))) {
            Some((w, Some(reference))) => {
                let u0 = w.apply(reference) - reference;
                let mut pad = 0.0_f64;
                for k in 0..=6 {
                    for j in 0..=6 {
                        for i in 0..=6 {
                            let p = reference + (DVec3::new(i as f64, j as f64, k as f64) / 3.0 - DVec3::ONE) * 2.0 * horizon;
                            pad = pad.max((w.apply(p) - p - u0).length());
                        }
                    }
                }
                (reference, horizon + (1.25 * pad).min(w.max_displacement()))
            }
            Some((w, None)) => (eye, horizon + w.max_displacement()),
            None => (eye, horizon),
        };
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
        self.world.node_box_in(&mut self.memo, s)
    }

    fn settled(&mut self, s: SectionPos) -> bool {
        !self.memo.stands_on_band(&s)
    }
}

impl World {
    /// The roots of `ground` seen from `eye`, how many have a box, and the nearest box: a test's
    /// report when a cut comes back empty.
    #[cfg(test)]
    pub(in crate::world) fn rule_roots_debug(&self, ground: Ground, eye: DVec3) -> String {
        let rule = self.rule_on(ground);
        let memo = self.node_boxes.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut surface = WorldSurface { world: self, ground, memo };
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
        s.detail.0 < self.rule_on(self.ground_of(s.body)).coarsest
    }

    /// The ground a body's tiles belong to.
    pub(in crate::world) fn ground_of(&self, body: u16) -> Ground {
        if body >= CHART_BODY_BASE {
            Ground::Chart { index: usize::from(body - CHART_BODY_BASE) }
        } else if self.generator.cosmos().is_some() {
            Ground::Cube { body }
        } else {
            Ground::Flat
        }
    }

    /// Every detail the rule may draw on `body`'s tiles at this view.
    pub(in crate::world) fn rule_details(&self, body: u16) -> impl Iterator<Item = Detail> + use<> {
        let rule = self.rule_on(self.ground_of(body));
        (rule.finest..=rule.coarsest).map(Detail)
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
        let mut memo = self.node_boxes.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        for p in split {
            let d = p.detail.0 as usize;
            if let Some(b) = self.node_box_in(&mut memo, p) {
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
