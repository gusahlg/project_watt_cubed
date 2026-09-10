use crate::*;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }
    fn element(&mut self) -> Element {
        let mut c = [0u8; D];
        for x in c.iter_mut() {
            *x = self.next() as u8;
        }
        Element(c)
    }
    fn config(&mut self, max: usize) -> Configuration {
        let n = 1 + (self.next() as usize % max);
        Configuration::new((0..n).map(|_| self.element()).collect::<Vec<_>>()).unwrap()
    }
}

#[test]
fn encoding_round_trips_and_keeps_order_and_multiplicity() {
    let a = Element([1, 2, 3, 4]);
    let b = Element([9, 8, 7, 6]);
    let ab = Configuration::new(vec![a, b]).unwrap();
    let ba = Configuration::new(vec![b, a]).unwrap();
    let aab = Configuration::new(vec![a, a, b]).unwrap();
    assert_ne!(ab, ba);
    assert_ne!(aab, ab);
    for c in [&ab, &ba, &aab, &Configuration::void()] {
        assert_eq!(&Configuration::decode(c.encode().as_bytes()).unwrap(), c);
    }
    assert_eq!(Configuration::decode(&[]), Err(DecodeError::Empty));
    assert_eq!(Configuration::decode(&[2, 1, 2, 3, 4]), Err(DecodeError::Truncated));
    assert_eq!(Configuration::decode(&[1, 1, 2, 3, 4, 5]), Err(DecodeError::Trailing));
    assert_eq!(Configuration::decode(&[99]), Err(DecodeError::TooLarge(99)));
    assert_eq!(Configuration::new(vec![a; CONFIG_MAX + 1]), Err(ConfigError::TooLarge(CONFIG_MAX + 1)));
}

#[test]
fn law_stamp_round_trips_and_fingerprint_is_stable() {
    let law = Law::v0();
    law.validate().unwrap();
    let stamp = law.stamp();
    assert_eq!(stamp.len(), law::STAMP_LEN);
    assert_eq!(Law::from_stamp(&stamp).unwrap(), law);
    assert_eq!(law.fingerprint(), Law::v0().fingerprint());
    let mut other = law;
    other.kernel.max_step = 7;
    assert_ne!(other.fingerprint(), law.fingerprint());
    assert_eq!(Law::from_stamp(&stamp[..10]), Err(LawError::Length(10)));
}

#[test]
fn deltas_respect_the_boundary_rule() {
    assert_eq!(kernel::axis_delta(Boundary::Clamp, 250, 5), 245);
    assert_eq!(kernel::axis_delta(Boundary::Wrap, 250, 5), -11);
    assert_eq!(kernel::axis_delta(Boundary::Wrap, 5, 250), 11);
    assert_eq!(kernel::axis_delta(Boundary::Wrap, 100, 40), 60);
}

#[test]
fn response_curve_is_odd_continuous_and_zero_at_rest() {
    let k = Law::v0().kernel;
    assert_eq!(kernel::response(&k, 0), 0);
    for d in -255..=255 {
        assert_eq!(kernel::response(&k, d), -kernel::response(&k, -d));
        if d < 255 {
            assert!((kernel::response(&k, d) - kernel::response(&k, d + 1)).abs() <= 5, "slope at {d}");
        }
    }
    assert!(kernel::response(&k, 3) < 0, "repulsive up close");
    assert!(kernel::response(&k, 48) > 0, "attractive at middle range");
    assert_eq!(kernel::response(&k, 255), 0);
}

#[test]
fn similarity_invariant_holds_statistically() {
    // ‖a − a'‖₁ = 1 ⇒ ‖F(a,b) − F(a',b)‖∞ ≤ K for the v0 law (K measured, pinned here).
    let law = Law::v0();
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let mut worst = 0i32;
    for _ in 0..20_000 {
        let a = rng.element();
        let b = rng.element();
        let axis = (rng.next() as usize) % D;
        if a.0[axis] == 255 {
            continue; // 255 → 0 is not a one-unit step under the Clamp boundary
        }
        let mut a2 = a;
        a2.0[axis] += 1;
        let f1 = element_influence(&law, a, b).0;
        let f2 = element_influence(&law, a2, b).0;
        for i in 0..D {
            worst = worst.max((f1[i] as i32 - f2[i] as i32).abs());
        }
    }
    assert!(worst <= 6, "one lattice step changed an influence by {worst} (pinned bound for law v0)");
}

#[test]
fn interact_is_deterministic_bounded_and_void_safe() {
    let law = Law::v0();
    let mut rng = Rng(7);
    for _ in 0..5_000 {
        let a = rng.config(6);
        let b = rng.config(6);
        let r1 = interact(&law, &a, &b, EventKind::Collision);
        let r2 = interact(&law, &a, &b, EventKind::Collision);
        assert_eq!(r1, r2);
        assert_eq!(r1.target.len(), b.len());
        for (x, y) in r1.target.elements().iter().zip(b.elements()) {
            assert!(x.max_axis_distance(*y) <= law.kernel.max_step as u32);
        }
        assert_eq!(r1.changed, r1.magnitude > 0);
        assert_eq!(r1.changed, r1.target != b);
    }
    let void = Configuration::void();
    let c = rng.config(4);
    assert!(!interact(&law, &void, &c, EventKind::Collision).changed);
    assert!(interact(&law, &c, &void, EventKind::Collision).target.is_void());
    // Weaker events move less.
    let a = rng.config(5);
    let b = rng.config(5);
    let strong = interact(&law, &a, &b, EventKind::Collision).magnitude;
    let weak = interact(&law, &a, &b, EventKind::ExternallyChanged).magnitude;
    assert!(weak <= strong);
}

#[test]
fn identical_elements_are_fixed_points_under_self_contact() {
    // δ = 0 on every axis ⇒ no influence: a uniform configuration never changes by touching itself.
    let law = Law::v0();
    let mut rng = Rng(99);
    for _ in 0..500 {
        let e = rng.element();
        let c = Configuration::new(vec![e; 1 + rng.next() as usize % 4]).unwrap();
        assert!(!interact(&law, &c, &c, EventKind::Collision).changed);
    }
}

#[test]
fn observations_are_bounded_and_air_is_air() {
    let law = Law::v0();
    assert_eq!(observe(&law, &Configuration::void()), Observation::AIR);
    let mut rng = Rng(3);
    let mut liquids = 0;
    let mut glows = 0;
    for _ in 0..5_000 {
        let c = rng.config(5);
        let o = observe(&law, &c);
        assert_eq!(o.solid, !o.liquid);
        assert!(o.emission <= 15);
        assert_ne!(o.acoustic, Acoustic::Void);
        liquids += o.liquid as u32;
        glows += (o.emission > 0) as u32;
    }
    // The universe has liquids and lights, but is mostly solid and dark.
    assert!(liquids > 50 && liquids < 1_500, "liquids: {liquids} of 5000");
    assert!(glows > 25 && glows < 1_000, "glows: {glows} of 5000");
}

#[test]
fn visuals_are_local_and_quantization_round_trips() {
    let law = Law::v0();
    let mut rng = Rng(11);
    let mut worst = 0u32;
    for _ in 0..3_000 {
        let e = rng.element();
        let mut e2 = e;
        let axis = (rng.next() as usize) % D;
        e2.0[axis] = e2.0[axis].saturating_add(1);
        let v1 = visual(&law, &Configuration::single(e));
        let v2 = visual(&law, &Configuration::single(e2));
        let d: u32 = (0..3).map(|i| (v1.rgb[i] as i32 - v2.rgb[i] as i32).unsigned_abs()).sum();
        worst = worst.max(d);
        let k = v1.quantize();
        let back = Visual::dequantize(k);
        assert!((back.rgb[0] as i32 - v1.rgb[0] as i32).abs() <= 16);
        assert_eq!(back.quantize(), k, "dequantize is a fixed point of quantize");
    }
    assert!(worst <= 32, "one lattice step moved the colour by {worst} (of 765; pinned bound for law v0)");
    let void = visual(&law, &Configuration::void());
    assert_eq!(void.alpha, 0);
}

#[test]
fn spread_and_mean_behave() {
    let a = Element([0, 0, 0, 0]);
    let b = Element([255, 255, 255, 255]);
    let c = Configuration::new(vec![a, b]).unwrap();
    assert_eq!(c.mean_q8().unwrap(), [255 * 128; D]);
    assert!(c.spread_q8() > 0);
    assert_eq!(Configuration::single(a).spread_q8(), 0);
    assert_eq!(Configuration::void().mean_q8(), None);
}

#[test]
#[ignore]
fn print_probe_quantiles() {
    // Calibration aid for the law's thresholds: run with --ignored --nocapture.
    let law = Law::v0();
    let mut rng = Rng(5);
    let names = ["contact", "light", "flow", "glow", "friction"];
    let probes = [law.probes.contact, law.probes.light, law.probes.flow, law.probes.glow, law.probes.friction];
    for (name, p) in names.iter().zip(probes) {
        let mut v: Vec<u8> = (0..20_000).map(|_| observe::response(&law, &rng.config(5), p)).collect();
        v.sort_unstable();
        let q = |f: f64| v[((v.len() - 1) as f64 * f) as usize];
        println!("{name:9} p05={} p25={} p50={} p75={} p85={} p92={} p98={} max={}", q(0.05), q(0.25), q(0.5), q(0.75), q(0.85), q(0.92), q(0.98), v[v.len() - 1]);
    }
}
