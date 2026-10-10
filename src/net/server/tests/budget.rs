//! Rate windows, per-kind budgets and the mod-channel budget.
use super::super::*;
use super::support::XorShift;

/// 100 frames on one channel, then the next is dropped. A second channel still has its own budget.
#[test]
fn channel_budget_is_per_channel_and_drops_the_overflow() {
    let mut budget = ChannelBudget::new();
    let voice = protocol::Channel::parse("voice").unwrap();
    let other = protocol::Channel::parse("other").unwrap();
    let now = Instant::now();
    for _ in 0..CHANNEL_RATE_LIMIT {
        assert!(budget.allow(&voice, now));
    }
    assert!(!budget.allow(&voice, now), "the 101st frame on one channel is dropped");
    assert!(budget.allow(&other, now), "a second channel keeps its own budget");
}

#[test]
fn rate_window_resets_across_the_second_and_cannot_be_gamed_at_the_boundary() {
    let t0 = Instant::now();
    let mut w = RateWindow::new(3);
    assert!(w.allow(t0));
    assert!(w.allow(t0 + Duration::from_millis(1)));
    assert!(w.allow(t0 + Duration::from_millis(2)));
    assert!(!w.allow(t0 + Duration::from_millis(3)), "over budget inside the second");
    assert!(!w.allow(t0 + Duration::from_millis(999)), "boundary-1ms still in the window");
    assert!(w.allow(t0 + Duration::from_secs(1)), "the oldest stamp ages out at +1s");
    assert!(!w.allow(t0 + Duration::from_secs(1)), "aging one stamp frees one slot, not a full refill");
    let mut fresh = RateWindow::new(3);
    for i in 0..3 {
        assert!(fresh.allow(t0 + Duration::from_millis(i)));
    }
    let mut gained = 0u32;
    for ms in 1000..=1002 {
        if fresh.allow(t0 + Duration::from_millis(ms)) {
            gained += 1;
        }
    }
    assert_eq!(gained, 3, "a full second later the budget is whole again");
}

#[test]
fn each_message_kind_has_its_own_budget() {
    let now = Instant::now();
    let mut budgets = KindBudget::new();
    let chat = ClientMessage::Chat { channel: 0, text: "hi".into() };
    for _ in 0..CHAT_RATE {
        assert!(matches!(charge(&mut budgets, &chat, now), Charge::Pass));
    }
    assert!(matches!(charge(&mut budgets, &chat, now), Charge::Drop));
    let edit = ClientMessage::Edit { req: 1, x: 0, y: 0, z: 0, expect: 0, spec: "air".into() };
    assert!(matches!(charge(&mut budgets, &edit, now), Charge::Pass), "chat does not spend edits");
    for _ in 1..EDIT_RATE {
        assert!(matches!(charge(&mut budgets, &edit, now), Charge::Pass));
    }
    assert!(matches!(charge(&mut budgets, &edit, now), Charge::Answer), "an over-budget edit is still answered");

    let swing = ClientMessage::Swing;
    for _ in 0..SWING_RATE {
        assert!(matches!(charge(&mut budgets, &swing, now), Charge::Pass));
    }
    assert!(matches!(charge(&mut budgets, &swing, now), Charge::Drop));

    let set_time = ClientMessage::SetTime { day: 0.2 };
    assert!(matches!(charge(&mut budgets, &set_time, now), Charge::Pass));
    assert!(matches!(charge(&mut budgets, &set_time, now), Charge::Answer));

    let ping = ClientMessage::Ping { nonce: 1 };
    for _ in 0..PING_RATE {
        assert!(matches!(charge(&mut budgets, &ping, now), Charge::Pass));
    }
    assert!(matches!(charge(&mut budgets, &ping, now), Charge::Drop));

    let teleport = ClientMessage::Teleport { pos: DVec3::ZERO };
    for _ in 0..TELEPORT_RATE {
        assert!(matches!(charge(&mut budgets, &teleport, now), Charge::Pass));
    }
    assert!(matches!(charge(&mut budgets, &teleport, now), Charge::Answer));

    let hop = ClientMessage::Move {
        pos: DVec3::ZERO,
        yaw: 0.0,
        pitch: 0.0,
        frame: DQuat::IDENTITY,
        velocity: Vec3::ZERO,
        up: Face::PosY,
        stance: Stance::Standing,
    };
    for _ in 0..MOVE_RATE {
        assert!(matches!(charge(&mut budgets, &hop, now), Charge::Pass));
    }
    assert!(matches!(charge(&mut budgets, &hop, now), Charge::Drop));

    let tool = ClientMessage::ToolUse { req: 1, x: 0, y: 0, z: 0, expect: 0, tool_spec: "air".into() };
    for _ in 0..TOOL_RATE_LIMIT {
        assert!(matches!(charge(&mut budgets, &tool, now), Charge::Pass));
    }
    assert!(matches!(charge(&mut budgets, &tool, now), Charge::Answer), "an over-budget tool use is still answered");

    let cruise = ClientMessage::Cruise { speed: 1.0 };
    for _ in 0..8 {
        assert!(matches!(charge(&mut budgets, &cruise, now), Charge::Pass), "cruise spends no token");
    }
}

/// Cycling channel names cannot beat the connection's total, and new names stop at the cap.
#[test]
fn mod_channels_share_one_budget_and_a_name_cap() {
    let now = Instant::now();
    let channel = |i: usize| protocol::Channel::parse(&format!("c{i}")).unwrap();
    let mut names = ChannelBudget::new();
    for i in 0..MAX_CHANNELS {
        assert!(names.allow(&channel(i), now));
    }
    assert!(!names.allow(&channel(MAX_CHANNELS), now), "a new name past the cap is dropped");
    assert!(names.allow(&channel(0), now), "a known name keeps its window");
    let mut flood = ChannelBudget::new();
    let passed = (0..4 * MOD_DATA_RATE as usize).filter(|&i| flood.allow(&channel(i % MAX_CHANNELS), now)).count();
    assert_eq!(passed, MOD_DATA_RATE as usize, "switching channels buys no extra rate");
}

/// The ring keeps the window's sliding behaviour when it wraps many times over.
#[test]
fn rate_window_ring_matches_a_sliding_window_over_many_wraps() {
    let t0 = Instant::now();
    let mut ring = RateWindow::new(5);
    let mut kept: std::collections::VecDeque<Instant> = std::collections::VecDeque::new();
    let mut rng = XorShift::new(0x5EED);
    let mut at = t0;
    for _ in 0..4000 {
        at += Duration::from_millis(u64::from(rng.u32(400)));
        while kept.front().is_some_and(|t| at.saturating_duration_since(*t) >= Duration::from_secs(1)) {
            kept.pop_front();
        }
        let want = kept.len() < 5;
        if want {
            kept.push_back(at);
        }
        assert_eq!(ring.allow(at), want, "at {:?}", at - t0);
    }
    let mut none = RateWindow::new(0);
    assert!(!none.allow(t0), "a zero budget allows nothing");
}
