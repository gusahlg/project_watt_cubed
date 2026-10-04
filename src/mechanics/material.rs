//! Constitutive parameters of matter (guide §§7.4–7.5).
//!
//! The emergent material law does not (yet) say how spatial strain changes a configuration's
//! energy, so stiffness and strength cannot be *derived* from it. [`mechanical_response`] is a
//! clearly labelled, versioned **prototype assignment** from observable properties (amount and
//! cohesion); it is not presented as a consequence of the reaction mathematics. It contains no
//! named-material exceptions: every configuration goes through the same formula.
//!
//! Units: length in blocks, mass in amount, time in seconds; stresses in amount/(block·s²).

/// Version of [`mechanical_response`]. Folded into the physics fingerprint and saves: changing the
/// formula changes how worlds relax.
pub const RESPONSE_VERSION: u16 = 1;

/// Shear modulus as a multiple of the yield stress: elastic strain at yield is `1 / SHEAR_PER_YIELD`.
pub const SHEAR_PER_YIELD: f64 = 1_000.0;
/// Bulk modulus of matter, amount/(block·s²): resistance to volume change is a property of packing,
/// not of strength. The centre of the start world sits near 2·10⁹, so matter compresses by about a
/// fifth of a percent there (guide §7.7: strong volume resistance, slower shape relaxation).
pub const BULK_MODULUS: f64 = 1.0e12;
/// Yield stress of matter of zero cohesion, amount/(block·s²).
pub const YIELD_FLOOR: f64 = 1.0e4;
/// Decades of yield stress across the cohesion range.
pub const YIELD_DECADES: f64 = 4.0;
/// Stiffness of void (air inside a lattice) relative to the matter's yield stress.
pub const VOID_STIFFNESS: f64 = 1e-3;
/// Creep relaxation time of the overstress, world seconds.
pub const CREEP_TIME: f64 = 120.0;

/// Mechanical parameters of one material (or a homogenised element).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Amount per block³.
    pub density: f64,
    /// Shear modulus μ.
    pub shear: f64,
    /// Bulk modulus κ.
    pub bulk: f64,
    /// von Mises yield stress.
    pub yield_stress: f64,
    /// Perzyna relaxation time of the overstress (world seconds).
    pub creep_time: f64,
}

impl Params {
    /// Parameters from density and yield stress through the prototype ratios.
    pub fn from_yield(density: f64, yield_stress: f64) -> Self {
        let shear = SHEAR_PER_YIELD * yield_stress;
        Self { density, shear, bulk: BULK_MODULUS, yield_stress, creep_time: CREEP_TIME }
    }

    /// Empty space inside a lattice (a building layer, a cavity): massless and nearly without
    /// stiffness, so it rides on the matter around it.
    pub fn void(reference: &Params) -> Self {
        // Negligible next to the weakest stress the matter can carry (its yield), so dragging the
        // air along never brakes the matter's flow.
        let modulus = VOID_STIFFNESS * reference.yield_stress.min(reference.shear);
        Self { density: 0.0, shear: modulus, bulk: 3.0 * modulus, yield_stress: f64::INFINITY, creep_time: reference.creep_time }
    }

    /// Dilatational modulus `κ + 4μ/3` (the stiffest wave), for stability bounds.
    pub fn wave_modulus(&self) -> f64 {
        self.bulk + 4.0 / 3.0 * self.shear
    }

    /// Mix of materials by volume fraction (the rest of the element is void): density and
    /// stiffness add by volume; yield follows the weakest phase (a lower bound on creep strength,
    /// the harmonic mean over the matter).
    pub fn mix(parts: &[(f64, Params)]) -> Self {
        let filled: f64 = parts.iter().map(|(f, _)| f).sum::<f64>().clamp(0.0, 1.0);
        let reference = parts.iter().map(|(_, p)| *p).find(|p| p.density > 0.0).unwrap_or(Params::from_yield(0.0, YIELD_FLOOR));
        if filled <= 0.0 {
            return Params::void(&reference);
        }
        let density = parts.iter().map(|(f, p)| f * p.density).sum();
        let air = Params::void(&reference);
        let shear = parts.iter().map(|(f, p)| f * p.shear).sum::<f64>() + (1.0 - filled) * air.shear;
        let bulk = parts.iter().map(|(f, p)| f * p.bulk).sum::<f64>() + (1.0 - filled) * air.bulk;
        let inv: f64 = parts.iter().filter(|(f, _)| *f > 0.0).map(|(f, p)| f / p.yield_stress).sum();
        let yield_stress = filled * filled / inv.max(1e-300);
        let creep_time = parts.iter().map(|(f, p)| f * p.creep_time).sum::<f64>() / filled;
        Self { density, shear, bulk, yield_stress, creep_time }
    }
}

/// Cohesion (the material law's observation, Q8) mapped to the bottom and the top of the yield
/// decades: the span the selective-transfer palette's materials actually occupy (ice/crystal ≈ 385,
/// soils ≈ 762), so the decades are used across real matter rather than an empty range.
pub const COHESION_SPAN: (f64, f64) = (384.0, 768.0);

/// Normalised cohesion of an observed material.
pub fn cohesion01(cohesion: i32) -> f64 {
    ((cohesion as f64 - COHESION_SPAN.0) / (COHESION_SPAN.1 - COHESION_SPAN.0)).clamp(0.0, 1.0)
}

/// **Prototype** mechanical response of a configuration from its amount (element occurrences,
/// 0..=32) and cohesion (normalised with [`cohesion01`]): density is the amount; yield rises over
/// [`YIELD_DECADES`] decades with cohesion; stiffness follows yield.
pub fn mechanical_response(amount: u32, cohesion01: f64) -> Params {
    let c = cohesion01.clamp(0.0, 1.0);
    // 10^(decades · c) without powf (generation arithmetic stays basic): exp(ln 10 · decades · c)
    // is fine here because this runs once per material, not per voxel, and is never hashed.
    let yield_stress = YIELD_FLOOR * (std::f64::consts::LN_10 * YIELD_DECADES * c).exp();
    Params::from_yield(amount as f64, yield_stress)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cohesion_spans_the_yield_decades() {
        let weak = mechanical_response(5, 0.0);
        let strong = mechanical_response(5, 1.0);
        assert_eq!(weak.density, 5.0);
        assert!((weak.yield_stress - YIELD_FLOOR).abs() < 1e-6);
        assert!((strong.yield_stress / weak.yield_stress - 10f64.powf(YIELD_DECADES)).abs() < 1e-6 * 10f64.powf(YIELD_DECADES));
        assert!((weak.shear - SHEAR_PER_YIELD * weak.yield_stress).abs() < 1e-6);
    }

    #[test]
    fn a_mix_is_weak_where_its_weakest_phase_is() {
        let a = Params::from_yield(5.0, 1e4);
        let b = Params::from_yield(5.0, 1e8);
        let m = Params::mix(&[(0.5, a), (0.5, b)]);
        assert!(m.yield_stress < 3e4, "{}", m.yield_stress);
        assert_eq!(m.density, 5.0);
        let half = Params::mix(&[(0.5, a)]);
        assert_eq!(half.density, 2.5);
        assert!(Params::mix(&[]).density == 0.0);
    }
}
