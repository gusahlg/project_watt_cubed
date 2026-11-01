// SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;

#[test]
fn audit_duplicate_pose_collapses_interpolation() {
    let mut peers = HashMap::new();
    peers.insert(1, RemotePlayer::joined(1, "peer".into(), DVec3::ZERO));
    let t = Instant::now();
    let pose = |yaw| protocol::PeerPose { id: 1, pos: DVec3::ZERO, yaw, pitch: 0.0, frame: DQuat::IDENTITY, velocity: Vec3::ZERO, up: Face::PosY, stance: Stance::Standing };
    apply_pose(&mut peers, pose(0.0), t);
    apply_pose(&mut peers, pose(1.0), t + Duration::from_millis(50));
    let mid = t + Duration::from_millis(75);
    assert!((peers[&1].sample(mid).yaw - 0.5).abs() < 1e-5);
    apply_pose(&mut peers, pose(1.0), t + Duration::from_millis(50) + Duration::from_micros(1));
    assert_eq!(peers[&1].sample(mid).yaw, 1.0);
    eprintln!("BUG: a duplicate target pose turns the interpolated midpoint 0.5 into a snap at 1.0");
}

#[test]
fn audit_late_tool_result_is_discarded_after_timeout() {
    let mut session = Session::new(DVec3::ZERO);
    let cell = (1, 80, 0);
    let (req, _) = session.track(cell, PendingKind::Tool);
    session.pending_edits[0].sent = Instant::now() - PENDING_TTL - Duration::from_secs(1);
    let mut out = Vec::new();
    session.expire(Instant::now(), &mut out);
    out.clear();
    session.apply(ServerMessage::ToolResult { req, reacted: true, rev: 1, cell_spec: "air".into(), tool_spec: "c:0101020304".into() }, &mut out);
    assert!(out.is_empty());
    assert!(!session.cell_revs.contains_key(&cell));
    eprintln!("BUG: a late committed tool result updates neither the cell nor the held tool");
}

#[test]
fn audit_stale_tool_result_is_emitted_after_newer_snapshot() {
    let mut session = Session::new(DVec3::ZERO);
    let cell = (1, 80, 0);
    let (req, _) = session.track(cell, PendingKind::Tool);
    let mut out = Vec::new();
    session.land_snapshot((cell.0, cell.1, cell.2, 3, "c:0101020304".into()), &mut out);
    out.clear();
    session.apply(ServerMessage::ToolResult { req, reacted: true, rev: 1, cell_spec: "air".into(), tool_spec: "c:0101020304".into() }, &mut out);
    assert!(matches!(&out[..], [Incoming::ToolResult { reacted: true, .. }]));
    assert_eq!(session.cell_revs[&cell], 3);
    eprintln!("RISK: stale tool content is emitted while known revision stays 3 (wire reordering not demonstrated)");
}
