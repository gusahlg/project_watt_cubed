//! The generator's bounds of the tiles a cut is about to split into, sampled off the main thread.
//! They are pure: a bound that lands early is the bound the cut would have sampled itself, so the
//! cut never waits on one and never changes with one, it only stops paying for it in the frame
//! where a coarse node splits into a few hundred freshly sampled tiles.

use super::{Leaf, Rule, SAMPLED_SPAN, SPLIT_SPANS, TIGHT_SPAN, sampled_bounds};
use crate::world::World;
use crate::world::sampler::{Recent, Sampler};
use crate::world::section::{Quadrant, SectionPos};

/// A leaf has its children's bounds fetched once the eye is within this many of its spans: twice
/// the reach at which it splits.
const LEAD_SPANS: f64 = 2.0 * SPLIT_SPANS;

/// Most tiles waiting on the sampler; past it a cut asks for nothing more until some land.
const ASKED_MAX: usize = 1 << 14;

/// Most tiles one cut asks for, and most answers it keeps: a pass after a jump spreads the work.
const PER_CUT: usize = 1 << 12;

/// The sampler of tile bounds, and the leaves whose children it was asked for (a cut mostly holds
/// the leaves the last one did).
pub(in crate::world) struct Prefetch {
    sampler: Sampler<SectionPos, Option<(f64, f64)>>,
    fetched: Recent<SectionPos, ()>,
}

/// Push the tiles whose samples give the bounds of `s`'s children: each child up to
/// [`SAMPLED_SPAN`] itself, a wider one down to its tiles of that span.
fn sampled_under(s: SectionPos, out: &mut impl FnMut(SectionPos)) {
    for q in Quadrant::ALL {
        let c = s.child(q);
        if c.span() <= SAMPLED_SPAN {
            out(c);
        } else {
            sampled_under(c, out);
        }
    }
}

impl World {
    /// Land the sampler's bounds, then ask it for those of the children of every leaf of `leaves`
    /// (the cut `rule` just took) the eye is within [`LEAD_SPANS`] of, nearest first.
    pub(in crate::world) fn prefetch_bounds(&mut self, rule: &Rule, leaves: &[Leaf]) {
        if self.bounds_prefetch.is_none() {
            let generator = self.generator.clone();
            let sampler = Sampler::start("rule-bounds", move |s| sampled_bounds(&*generator, s));
            self.bounds_prefetch = sampler.map(|sampler| Prefetch { sampler, fetched: Recent::default() });
        }
        let Some(p) = &mut self.bounds_prefetch else { return };
        let mut memo = self.node_boxes.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        p.sampler.land(PER_CUT, |s, b| {
            if !memo.ground.contains(&s) {
                memo.ground.insert(s, b);
            }
        });
        if p.sampler.waiting() >= ASKED_MAX {
            return;
        }
        let mut near: Vec<&Leaf> = leaves
            .iter()
            .filter(|l| {
                let span = l.pos.span();
                l.pos.detail.0 > rule.finest && span / 2 <= TIGHT_SPAN && l.dist < LEAD_SPANS * f64::from(span) && !p.fetched.contains(&l.pos)
            })
            .collect();
        near.sort_unstable_by(|a, b| a.dist.total_cmp(&b.dist));
        let mut want = Vec::new();
        for l in near {
            if want.len() >= PER_CUT {
                break;
            }
            p.fetched.insert(l.pos, ());
            sampled_under(l.pos, &mut |s| {
                if !memo.ground.contains(&s) && p.sampler.mark(s) {
                    want.push(s);
                }
            });
        }
        p.sampler.send(want);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_config::RenderConfig;
    use crate::world::fixtures::spawned_round_world;
    use crate::world::rule::Ground;

    /// The sampler's bounds are the ones the cut would sample itself, bit for bit, and it is asked
    /// for the children of the leaves about to split.
    #[test]
    fn prefetched_bounds_are_the_cuts_own() {
        let (mut world, spawn) = spawned_round_world(RenderConfig { lod2: true, ..RenderConfig::default() }, 6, 3);
        let storage = world.chart_eye(spawn).expect("a chart eye");
        let seat = world
            .seams
            .chart_seat(World::chunk_of(storage.x.floor() as i32, storage.y.floor() as i32, storage.z.floor() as i32))
            .expect("a seat");
        let mut leaves = Vec::new();
        let rule = world.rule_cut(Ground::Chart { index: seat.index }, spawn, &mut leaves);
        world.forget_rule_memo();
        world.prefetch_bounds(&rule, &leaves);
        let mut asked = Vec::new();
        for l in &leaves {
            if l.pos.detail.0 > rule.finest && l.pos.span() / 2 <= TIGHT_SPAN && l.dist < LEAD_SPANS * f64::from(l.pos.span()) {
                sampled_under(l.pos, &mut |s| asked.push(s));
            }
        }
        assert!(!asked.is_empty(), "nothing fetched ahead of {} leaves", leaves.len());
        assert!(asked.iter().all(|s| world.bounds_prefetch.as_ref().unwrap().sampler.asked(s)));
        let start = std::time::Instant::now();
        while world.bounds_prefetch.as_ref().is_some_and(|p| p.sampler.waiting() > 0) {
            assert!(start.elapsed().as_secs() < 60, "the sampler never answered");
            std::thread::sleep(std::time::Duration::from_millis(1));
            world.prefetch_bounds(&rule, &leaves);
        }
        let mut memo = world.node_boxes.lock().unwrap();
        for s in asked {
            assert_eq!(memo.ground.get(&s), Some(sampled_bounds(&*world.generator, s)), "{s:?}");
        }
    }
}
