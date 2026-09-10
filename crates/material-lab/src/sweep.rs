//! Random valid mutations of the v0 knots/mixing/strengths, ranked by scorecard.

use material::{Law, EVENT_KINDS, KNOTS, D};

use crate::rng::Rng;
use crate::score::{run_scorecard, Scale, Scorecard};
use crate::stamp_hex;

/// One mutated law and its reduced scorecard.
#[derive(Clone, Debug)]
pub struct SweepHit {
    /// The mutated law.
    pub law: Law,
    /// Canonical stamp, lowercase hex.
    pub stamp_hex: String,
    /// Bands that PASSed.
    pub passes: u32,
    /// Family count (tie-break).
    pub families: u32,
    /// The reduced scorecard.
    pub card: Scorecard,
}

/// Nudge v0 knots, mixing and event strengths; retry until `validate` accepts.
pub fn mutate_law(base: &Law, rng: &mut Rng) -> Law {
    loop {
        let mut law = *base;
        for k in law.kernel.knots.iter_mut() {
            let d = rng.inc(-6, 6) as i16;
            k.1 = k.1.saturating_add(d).clamp(-64, 64);
        }
        if rng.next_u32() % 3 == 0 {
            for i in 1..KNOTS - 1 {
                let lo = law.kernel.knots[i - 1].0 as i32 + 1;
                let hi = law.kernel.knots[i + 1].0 as i32 - 1;
                if hi > lo {
                    law.kernel.knots[i].0 = rng.inc(lo, hi) as u8;
                }
            }
        }
        for i in 0..D {
            for j in 0..D {
                let d = rng.inc(-2, 2) as i8;
                law.kernel.mixing[i][j] = law.kernel.mixing[i][j].saturating_add(d);
            }
        }
        for i in 0..EVENT_KINDS {
            let d = rng.inc(-20, 20);
            let v = law.events.0[i] as i32 + d;
            law.events.0[i] = v.clamp(1, 255) as u8;
        }
        if law.validate().is_ok() {
            return law;
        }
    }
}

/// Mutate v0 `n` times, run a reduced scorecard on each, return the top 5
/// (PASS count, then family count).
pub fn sweep(seed: u64, n: u32) -> Vec<SweepHit> {
    let base = Law::v0();
    let mut rng = Rng::new(seed ^ 0x5357_4545_5000_0001);
    let mut hits: Vec<SweepHit> = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let law = mutate_law(&base, &mut rng);
        let card = run_scorecard(&law, seed, Scale::reduced());
        hits.push(SweepHit {
            stamp_hex: stamp_hex(&law),
            passes: card.pass_count(),
            families: card.families.families,
            card,
            law,
        });
    }
    hits.sort_by(|a, b| b.passes.cmp(&a.passes).then(b.families.cmp(&a.families)));
    hits.truncate(5);
    hits
}
