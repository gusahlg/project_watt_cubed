//! Crate-level checks: canonical multisets and their bytes, the law stamp, and observations.

use crate::*;

fn el(c: [u8; 4]) -> Element {
    Element::new(c)
}

fn cfg(c: &[[u8; 4]]) -> Configuration {
    Configuration::new(c.iter().map(|&c| el(c)).collect::<Vec<_>>()).unwrap()
}

#[test]
fn configurations_are_multisets_with_canonical_bytes() {
    let a = cfg(&[[9, 9, 9, 9], [1, 2, 3, 4], [9, 9, 9, 9]]);
    let b = cfg(&[[1, 2, 3, 4], [9, 9, 9, 9], [9, 9, 9, 9]]);
    assert_eq!(a, b, "storage order is not meaning");
    assert_eq!(a.encode(), b.encode());
    assert_ne!(a, cfg(&[[1, 2, 3, 4], [9, 9, 9, 9]]), "multiplicity is meaning");
    assert_eq!(Configuration::decode(a.encode().as_bytes()).unwrap(), a);
    assert_eq!(a.counts().collect::<Vec<_>>(), vec![(el([1, 2, 3, 4]), 1), (el([9, 9, 9, 9]), 2)]);
    // Unsorted bytes (an older save) decode to the same multiset.
    assert_eq!(Configuration::decode(&[2, 9, 9, 9, 9, 1, 2, 3, 4]).unwrap(), cfg(&[[1, 2, 3, 4], [9, 9, 9, 9]]));
    assert_eq!(Configuration::decode(&[]), Err(DecodeError::Empty));
    assert_eq!(Configuration::decode(&[33]), Err(DecodeError::TooLarge(33)));
    assert_eq!(Configuration::decode(&[1, 1, 2]), Err(DecodeError::Truncated));
    assert_eq!(Configuration::decode(&[0, 1]), Err(DecodeError::Trailing));
    assert!(Configuration::new(vec![el([0; 4]); CAPACITY]).is_ok());
    assert_eq!(Configuration::new(vec![el([0; 4]); CAPACITY + 1]), Err(ConfigError::TooLarge(33)));
}

#[test]
fn block_of_a_configuration_round_trips() {
    let c = cfg(&[[200, 1, 2, 3], [4, 5, 6, 7], [4, 5, 6, 7]]);
    let b = Block::of(&c);
    assert_eq!(b.configuration(), c);
    assert_eq!(b.internal_fit() * 2, b.holding().iter().map(|&h| h as i64).sum::<i64>());
}

#[test]
fn law_stamp_round_trips_and_refuses_other_functions() {
    let law = Law::current();
    let stamp = law.stamp();
    assert_eq!(stamp.len(), STAMP_LEN);
    assert_eq!(Law::from_stamp(&stamp), Ok(law));
    let mut bad = stamp.clone();
    bad[3] ^= 1;
    assert_eq!(Law::from_stamp(&bad), Err(LawError::Function), "a different fit table");
    let mut v1 = stamp.clone();
    v1[0] = 1;
    assert_eq!(Law::from_stamp(&v1), Err(LawError::Version(1)));
    assert_eq!(Law::from_stamp(&stamp[1..]), Err(LawError::Length(STAMP_LEN - 1)));
    assert_ne!(law.fingerprint(), 0);
}

#[test]
fn observations_read_the_fit_function() {
    let law = Law::current();
    assert_eq!(observe(&law, &Block::default()), Observation::AIR);
    // The cohesive configuration of the reference suite holds together: hard.
    let a = Block::of(&cfg(&crate::kernel::tests::A));
    let o = observe(&law, &a);
    assert!(o.solid && o.cohesion > 0 && o.hardness >= 170, "{o:?}");
    // A repulsive mixture reads as weakly held.
    let r = Block::of(&cfg(&[[0; 4], [128; 4]]));
    let o = observe(&law, &r);
    assert!(o.cohesion < 0 && o.hardness < 96, "{o:?}");
    // Identical occurrences have zero pair fit.
    assert_eq!(cohesion(&Block::of(&cfg(&[[7; 4]; 5]))), 0);
}

#[test]
fn observation_census_over_random_configurations() {
    // Clarity and glow are rare properties; most matter is opaque and dark.
    let law = Law::current();
    let mut x: u64 = 0x1234_5678_9abc_def1;
    let mut byte = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 32) as u8
    };
    let (mut clear, mut glow, n) = (0, 0, 4000);
    for i in 0..n {
        let len = 2 + i % 7;
        let elems: Vec<Element> = (0..len).map(|_| el([byte(), byte(), byte(), byte()])).collect();
        let o = observe(&law, &Block::new(&elems).unwrap());
        clear += (o.transparency > 0) as u32;
        glow += (o.emission > 0) as u32;
    }
    assert!(clear * 100 < n * 15 && clear > 0, "clear {clear}/{n}");
    assert!(glow * 100 < n * 15 && glow > 0, "glow {glow}/{n}");
}

#[test]
fn visual_takes_the_centroid_colour() {
    let law = Law::current();
    let x = el([10, 20, 30, 40]);
    let pure = visual(&law, &Block::of(&Configuration::single(x)));
    assert_eq!(pure.rgb, element_colour(&law, x));
    // The centroid wraps the short way round the ring: 250 and 10 average to 2, not 130.
    let c = centroid_q8(&[el([250, 0, 0, 0]), el([10, 0, 0, 0])]).unwrap();
    assert_eq!(c[0], 2 * 256);
    let y = el([14, 24, 34, 44]);
    let mixed = visual(&law, &Block::of(&cfg(&[x.0, y.0])));
    assert_eq!(mixed.rgb, colour_at(&law, [12 * 256, 22 * 256, 32 * 256, 42 * 256]));
    assert_eq!(visual(&law, &Block::default()), Visual::VOID);
}
