//! Named-dimension search of a v0-shaped law: rest band, attraction peak/fade, mixing coupling,
//! max_step and event strengths, ranked by quiescence then family target then native sub-linearity.

use material::{Law, EVENT_KINDS, D};

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
    /// Family count (reporting).
    pub families: u32,
    /// The reduced scorecard.
    pub card: Scorecard,
}

/// The v0-shaped knobs the sweep actually searches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Named {
    /// Rest-band start (knot 3 distance).
    pub rest_lo: u8,
    /// Rest-band end (knot 4 distance).
    pub rest_hi: u8,
    /// Attraction-peak distance (knot 5).
    pub peak_d: u8,
    /// Attraction-peak response (knot 5).
    pub peak_r: i16,
    /// Fade-to-zero distance (knot 7).
    pub fade: u8,
    /// Cyclic mixing coupling, Q4 (v0 = 4).
    pub coupling: i8,
    /// Per-axis step bound.
    pub max_step: u8,
    /// Event strengths, stamp order.
    pub events: [u8; EVENT_KINDS],
}

/// Read the named knobs of a v0-shaped law.
pub fn named(law: &Law) -> Named {
    Named {
        rest_lo: law.kernel.knots[3].0,
        rest_hi: law.kernel.knots[4].0,
        peak_d: law.kernel.knots[5].0,
        peak_r: law.kernel.knots[5].1,
        fade: law.kernel.knots[7].0,
        coupling: law.kernel.mixing[0][1],
        max_step: law.kernel.max_step,
        events: law.events.0,
    }
}

/// Integer ranking key: higher is better. Quiescent ‱, family-target, native sub-linear, PASS count.
pub fn rank_tuple(
    quiescent: u32,
    runs: u32,
    families: u32,
    stable: u32,
    native_sublinear: bool,
    passes: u32,
) -> (u32, u32, u32, u32) {
    let q = if runs == 0 {
        0
    } else {
        quiescent.saturating_mul(10_000) / runs
    };
    let fam_ok = (8..=200).contains(&families)
        && families > 0
        && stable >= families.saturating_mul(20);
    (q, fam_ok as u32, native_sublinear as u32, passes)
}

/// [`rank_tuple`] of a scorecard.
pub fn rank_key(card: &Scorecard) -> (u32, u32, u32, u32) {
    rank_tuple(
        card.cascades.quiescent,
        card.cascades.runs,
        card.families.families,
        card.families.stable,
        card.proliferation.native.sublinear,
        card.pass_count(),
    )
}

/// True when family count is in 8..200 and mean size is at least 20.
pub fn family_target(families: u32, stable: u32) -> bool {
    rank_tuple(0, 1, families, stable, false, 0).1 == 1
}

/// Sample a v0-shaped law over the named search dimensions; retry until `validate` accepts.
pub fn mutate_law(base: &Law, rng: &mut Rng) -> Law {
    loop {
        let mut law = *base;

        let rest_lo = rng.inc(20, 28) as u8;
        let rest_hi = rng.inc(rest_lo as i32 + 2, 30) as u8;
        let peak_d = rng.inc(32, 52) as u8;
        let fade_d = rng.inc(48, 96) as u8;
        if rest_lo <= law.kernel.knots[2].0 || rest_hi >= peak_d || peak_d >= fade_d {
            continue;
        }
        let mid_d = ((peak_d as u16 + fade_d as u16) / 2) as u8;
        if mid_d <= peak_d || mid_d >= fade_d {
            continue;
        }

        let dead_end = law.kernel.knots[1].0 as i32;
        let rep_lo = dead_end + 1;
        let rep_hi = rest_lo as i32 - 1;
        if rep_hi < rep_lo {
            continue;
        }
        law.kernel.knots[2].0 = rng.inc(rep_lo, rep_hi) as u8;
        law.kernel.knots[2].1 = rng.inc(-16, -2) as i16;
        law.kernel.knots[3] = (rest_lo, 0);
        law.kernel.knots[4] = (rest_hi, 0);
        let peak_r = rng.inc(4, 16) as i16;
        law.kernel.knots[5] = (peak_d, peak_r);
        let mid_r = rng.inc(1, peak_r as i32) as i16;
        law.kernel.knots[6] = (mid_d, mid_r);
        law.kernel.knots[7] = (fade_d, 0);

        let c = rng.inc(0, 8) as i8;
        law.kernel.mixing = [[0; D]; D];
        for i in 0..D {
            law.kernel.mixing[i][i] = 16;
            law.kernel.mixing[i][(i + 1) % D] = c;
        }
        law.kernel.max_step = rng.inc(4, 8) as u8;

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

fn hit_of(law: Law, card: Scorecard) -> SweepHit {
    SweepHit {
        stamp_hex: stamp_hex(&law),
        passes: card.pass_count(),
        families: card.families.families,
        card,
        law,
    }
}

/// Mutate v0 `n` times at `scale`, return the top 5 by [`rank_key`].
pub fn sweep_at(seed: u64, n: u32, scale: Scale) -> Vec<SweepHit> {
    let base = Law::v0();
    let mut rng = Rng::new(seed ^ 0x5357_4545_5000_0001);
    let mut hits: Vec<SweepHit> = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let law = mutate_law(&base, &mut rng);
        let card = run_scorecard(&law, seed, scale);
        hits.push(hit_of(law, card));
    }
    hits.sort_by(|a, b| rank_key(&b.card).cmp(&rank_key(&a.card)));
    hits.truncate(5);
    hits
}

/// Mutate v0 `n` times, run a reduced scorecard on each, return the top 5.
pub fn sweep(seed: u64, n: u32) -> Vec<SweepHit> {
    sweep_at(seed, n, Scale::reduced())
}

/// One-line named knobs plus the ranking key, then the scorecard.
pub fn render_hit(rank: usize, h: &SweepHit) -> String {
    let n = named(&h.law);
    let (q, fam, native, passes) = rank_key(&h.card);
    let q_pct = q as f64 / 100.0;
    let fam_s = if fam == 1 { "yes" } else { "no" };
    let native_s = if native == 1 { "yes" } else { "no" };
    format!(
        "#{}  quiescent={q_pct:.2}%  family-target={fam_s}  native-sublinear={native_s}  PASS={passes}/7\n    rest={}..{}  peak={},{}  fade={}  coupling={}  max_step={}\n    events={:?}\n    stamp={}\n{}",
        rank,
        n.rest_lo,
        n.rest_hi,
        n.peak_d,
        n.peak_r,
        n.fade,
        n.coupling,
        n.max_step,
        n.events,
        h.stamp_hex,
        crate::score::render(&h.card),
    )
}
