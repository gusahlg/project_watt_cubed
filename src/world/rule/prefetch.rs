//! The tight bounds of the tiles a cut is about to split into, asked of the rule's sampler before
//! the cut gets there. They are pure: a bound that lands early is the bound the cut would have
//! waited for, so the cut stays tight across a split instead of standing on the band for a pass.

use super::{Leaf, Rule, SAMPLED_SPAN, SPLIT_SPANS, TIGHT_SPAN};
use crate::world::World;
use crate::world::section::{Quadrant, SectionPos};

/// A leaf has its children's bounds fetched once the eye is within this many of its spans: twice
/// the reach at which it splits.
const LEAD_SPANS: f64 = 2.0 * SPLIT_SPANS;

/// Most tiles waiting on the sampler; past it a cut asks for nothing more ahead until some land.
const ASKED_MAX: usize = 1 << 14;

/// Most tiles one cut asks for ahead.
const PER_CUT: usize = 1 << 12;

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
    /// Ask the rule's sampler for the tight bounds of the children of every leaf of `leaves` (the
    /// cut `rule` just took) the eye is within [`LEAD_SPANS`] of, nearest first.
    pub(in crate::world) fn prefetch_bounds(&self, rule: &Rule, leaves: &[Leaf]) {
        let mut memo = self.node_boxes.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if memo.waiting() >= ASKED_MAX {
            return;
        }
        let mut near: Vec<&Leaf> = leaves
            .iter()
            .filter(|l| {
                let span = l.pos.span();
                l.pos.detail.0 > rule.finest && span / 2 <= TIGHT_SPAN && l.dist < LEAD_SPANS * f64::from(span) && !memo.fetched.contains(&l.pos)
            })
            .collect();
        near.sort_unstable_by(|a, b| a.dist.total_cmp(&b.dist));
        let asked = memo.waiting();
        for l in near {
            if memo.waiting() - asked >= PER_CUT {
                break;
            }
            memo.fetched.insert(l.pos, ());
            sampled_under(l.pos, &mut |s| {
                if !memo.ground.contains(&s) {
                    memo.ask(&self.generator, s);
                }
            });
        }
        memo.flush();
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
        let (world, spawn) = spawned_round_world(RenderConfig { lod2: true, ..RenderConfig::default() }, 6, 3);
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
        assert!(asked.iter().all(|s| world.node_boxes.lock().unwrap().asked(s)));
        let start = std::time::Instant::now();
        while world.node_boxes.lock().unwrap().waiting() > 0 {
            assert!(start.elapsed().as_secs() < 60, "the sampler never answered");
            std::thread::sleep(std::time::Duration::from_millis(1));
            world.land_bounds();
        }
        let mut memo = world.node_boxes.lock().unwrap();
        for s in asked {
            assert_eq!(memo.ground.get(&s), Some(super::super::sampled_bounds(&*world.generator, s)), "{s:?}");
        }
    }
}
