//! Voice sessions — the semantic half of the runtime's voice path.
//!
//! A voice stream is a loss-tolerant journal: the playout transition is total
//! over `{fresh, dup, stale, gap, empty, abandoned}`. The decision function lives
//! in [`JitterBuffer`] and is deliberately pure — it takes packets and an `abandoned`
//! flag and yields a [`PlayoutStep`] with no reference to kira, rtrb, or a real opus
//! decoder, so the full state space is enumerable in unit tests. `backend/kira.rs`
//! wraps it with the rtrb feed and the opus decode/PLC calls.

use glam::DVec3;

use super::backend::BackendVoice;

/// Server-stamped per-player identity for a voice speaker. The server stamps
/// `id` because client generations collide across peers.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct SessionKey(pub u32);

/// Join generation for a `SessionKey` (strictly increases per key). A newtype,
/// not an alias: `Epoch` and `Seq` are both `u32` inside `VoicePacket`, so a transposition
/// at the net-seam construction site is a type error rather than a silent swap.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Epoch(pub u32);

/// Wrapping RTP-style sequence number over the u32 circle.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Seq(pub u32);

impl Seq {
    const HALF: u32 = 1 << 31;

    /// RFC 1982 serial comparison: true iff `self` is strictly ahead of `other`.
    pub fn newer_than(self, other: Seq) -> bool {
        self.0 != other.0 && self.0.wrapping_sub(other.0) < Self::HALF
    }

    /// Circular distance from `other` up to `self` going forward (0 when equal).
    pub fn ahead_of(self, other: Seq) -> u32 {
        self.0.wrapping_sub(other.0)
    }

    fn next(self) -> Seq {
        Seq(self.0.wrapping_add(1))
    }
}

pub const VOICE_SAMPLE_RATE: u32 = 48_000;
pub const VOICE_FRAME_MS: u32 = 20;
/// PCM samples in one 20 ms mono frame at 48 kHz (960).
pub const VOICE_FRAME_SAMPLES: usize =
    (VOICE_SAMPLE_RATE as usize * VOICE_FRAME_MS as usize) / 1000;
/// Opus payload ceiling for a 20 ms frame; the codec enforces it.
pub const MAX_VOICE_PAYLOAD: usize = 400;

/// One encoded 20 ms voice frame, server-stamped with speaker/epoch/seq.
#[derive(Clone, Debug)]
pub struct VoicePacket {
    pub session: SessionKey,
    pub epoch: Epoch,
    pub seq: Seq,
    pub payload: Box<[u8]>,
}

/// SPSC depth of the runtime→decoder feed, co-tuned with the jitter target.
pub(crate) const PACKET_QUEUE_DEPTH: usize = 16;

/// Packets held ahead of the play head to absorb reordering (also the initial-fill floor).
pub const JITTER_REORDER_WINDOW: u32 = 4;
/// A seq more than this far ahead of the play head is treated as a resync glitch and dropped.
pub const JITTER_STALE_WINDOW: u32 = 100;
/// Consecutive concealment frames before the buffer gives up and reports starvation.
pub const PLC_MAX_CONSECUTIVE: u32 = 5;

/// Semantic session sum (the epoch floor lives in `Tombstone`; `Closed` = map absence).
///
/// The stream voice is created (via `play_stream`, initially at `gain 0`) the instant
/// a session opens and lives exactly as long as the epoch. The `VoiceDecoder` sits
/// inside that kira sound, so presentation detach (interest-exit, deafen, allocation
/// loss) is `backend.update(Dsp{gain: 0.0, ..})` — never `backend.stop`, which fires
/// only on epoch-bump teardown and `close_session`.
pub(crate) enum Session {
    Streaming {
        epoch: Epoch,
        /// SPSC producer; the matching consumer lives inside the backend `VoiceDecoder`.
        feed: rtrb::Producer<VoicePacket>,
        /// Always live for the epoch's lifetime; detach mutes it, it is never dropped here.
        voice: BackendVoice,
        /// Whether the voice is currently presented (false = muted, decoder alive).
        present: bool,
        last_at: Option<DVec3>,
        /// Per-session coordinate smoother: sessions smooth like clips/emitters.
        smooth: super::Smoothed,
    },
    Tombstone {
        epoch: Epoch,
    },
}

/// The decoder's per-frame decision. Total over the packet journal. `payload` for
/// [`PlayoutStep::Decode`] is borrowed from the buffered packet.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PlayoutStep<'a> {
    /// Decode this in-order opus frame.
    Decode(&'a [u8]),
    /// Packet lost with lookahead or within the PLC budget: opus concealment.
    Conceal,
    /// Prolonged starvation (beyond the PLC budget): emit a silent frame.
    Silence,
    /// Producer abandoned and the buffer is drained: report finished so kira stops the
    /// sound instead of leaving the voice playing forever.
    Finished,
}

/// Pure playout state machine. Reordering is absorbed by holding a
/// pre-buffer of `target_frames`; once playout starts the decision is simply
/// "is the next seq present?" — present → decode, absent → conceal/silence/finish.
pub(crate) struct JitterBuffer {
    buffer: Vec<VoicePacket>,
    play_seq: Option<Seq>,
    started: bool,
    consecutive_conceal: u32,
    target_frames: u32,
    /// Latched once per starvation burst; the runtime may read and clear it.
    starved: bool,
}

/// Why a pushed packet was refused (observation channel for the state-space tests).
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PushOutcome {
    Buffered,
    Duplicate,
    Stale,
}

impl JitterBuffer {
    pub fn new(jitter_target_ms: u32) -> Self {
        let target = (jitter_target_ms / VOICE_FRAME_MS).max(JITTER_REORDER_WINDOW).max(1);
        Self {
            buffer: Vec::with_capacity(target as usize + JITTER_REORDER_WINDOW as usize),
            play_seq: None,
            started: false,
            consecutive_conceal: 0,
            target_frames: target,
            starved: false,
        }
    }

    /// Read-and-clear the starvation flag (local; no cross-session clock).
    pub fn take_starved(&mut self) -> bool {
        std::mem::take(&mut self.starved)
    }

    /// Classify and (if fresh) insert a packet. Behind the play head ⇒ already
    /// played/skipped ⇒ `Stale`; an absurd forward jump ⇒ `Stale`; a seq already held ⇒
    /// `Duplicate`; otherwise `Buffered`.
    pub fn push(&mut self, pkt: VoicePacket) -> PushOutcome {
        if let Some(play) = self.play_seq {
            let ahead = pkt.seq.ahead_of(play);
            if ahead >= Seq::HALF {
                return PushOutcome::Stale; // behind the play head
            }
            if ahead > JITTER_STALE_WINDOW {
                return PushOutcome::Stale; // resync glitch far in the future
            }
        }
        if self.buffer.iter().any(|p| p.seq == pkt.seq) {
            return PushOutcome::Duplicate;
        }
        self.buffer.push(pkt);
        PushOutcome::Buffered
    }

    fn oldest_seq(&self) -> Option<Seq> {
        self.buffer.iter().map(|p| p.seq).reduce(|acc, s| {
            if acc.newer_than(s) { s } else { acc }
        })
    }

    /// Advance the playout clock by one 20 ms frame. Total over the journal.
    pub fn pull(&mut self, abandoned: bool) -> PlayoutStep<'_> {
        // Reclaim packets already played/skipped (strictly behind the play head). A
        // just-decoded packet is pruned on the next pull, keeping the `Decode` borrow valid.
        if let Some(play) = self.play_seq {
            self.buffer.retain(|p| {
                let behind = play.ahead_of(p.seq);
                behind == 0 || behind >= Seq::HALF
            });
        }

        if !self.started {
            let filled = self.buffer.len() as u32 >= self.target_frames;
            if filled || (abandoned && !self.buffer.is_empty()) {
                self.play_seq = self.oldest_seq();
                self.started = true;
            } else if abandoned {
                return PlayoutStep::Finished;
            } else {
                return PlayoutStep::Silence; // pre-buffering; not a starvation
            }
        }

        let play = self.play_seq.expect("started implies a play head");
        if let Some(idx) = self.buffer.iter().position(|p| p.seq == play) {
            self.play_seq = Some(play.next());
            self.consecutive_conceal = 0;
            // Borrow the payload in place; the packet is pruned on the next pull.
            return PlayoutStep::Decode(&self.buffer[idx].payload);
        }

        // Gap: the next seq is not (yet) here.
        if self.buffer.is_empty() {
            if abandoned {
                return PlayoutStep::Finished;
            }
            if self.consecutive_conceal < PLC_MAX_CONSECUTIVE {
                self.consecutive_conceal += 1;
                self.play_seq = Some(play.next());
                return PlayoutStep::Conceal;
            }
            // Prolonged starvation with nothing buffered: reset to re-prebuffer rather
            // than conceal into the void and later stutter to catch up.
            self.started = false;
            self.play_seq = None;
            self.consecutive_conceal = 0;
            self.starved = true;
            return PlayoutStep::Silence;
        }

        // Lookahead exists (future frames buffered) ⇒ the play seq is genuinely lost.
        self.play_seq = Some(play.next());
        if self.consecutive_conceal < PLC_MAX_CONSECUTIVE {
            self.consecutive_conceal += 1;
            PlayoutStep::Conceal
        } else {
            self.starved = true;
            PlayoutStep::Silence // keep advancing to reach the buffered frames
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pkt(seq: u32) -> VoicePacket {
        VoicePacket {
            session: SessionKey(1),
            epoch: Epoch(0),
            seq: Seq(seq),
            payload: vec![seq as u8].into_boxed_slice(),
        }
    }

    /// Drive the buffer one pull and return the decision as an owned, comparable tag.
    #[derive(Debug, PartialEq, Eq)]
    enum Step {
        Decode(u8),
        Conceal,
        Silence,
        Finished,
    }
    fn pull(jb: &mut JitterBuffer, abandoned: bool) -> Step {
        match jb.pull(abandoned) {
            PlayoutStep::Decode(p) => Step::Decode(p[0]),
            PlayoutStep::Conceal => Step::Conceal,
            PlayoutStep::Silence => Step::Silence,
            PlayoutStep::Finished => Step::Finished,
        }
    }

    // Small target so tests don't have to over-fill; REORDER_WINDOW is the floor (4).
    fn jb() -> JitterBuffer {
        JitterBuffer::new(JITTER_REORDER_WINDOW * VOICE_FRAME_MS)
    }

    #[test]
    fn in_order_after_prefill() {
        let mut jb = jb();
        for s in 0..4 {
            assert_eq!(jb.push(pkt(s)), PushOutcome::Buffered);
        }
        for s in 0..4 {
            assert_eq!(pull(&mut jb, false), Step::Decode(s as u8));
        }
    }

    #[test]
    fn reorder_within_window_is_sorted() {
        let mut jb = jb();
        for s in [0u32, 1, 3, 2] {
            assert_eq!(jb.push(pkt(s)), PushOutcome::Buffered);
        }
        for s in 0..4 {
            assert_eq!(pull(&mut jb, false), Step::Decode(s as u8));
        }
    }

    #[test]
    fn duplicate_is_dropped_and_played_once() {
        let mut jb = jb();
        for s in 0..4 {
            jb.push(pkt(s));
        }
        assert_eq!(jb.push(pkt(2)), PushOutcome::Duplicate);
        for s in 0..4 {
            assert_eq!(pull(&mut jb, false), Step::Decode(s as u8));
        }
    }

    #[test]
    fn stale_behind_play_head_dropped() {
        let mut jb = jb();
        for s in 0..4 {
            jb.push(pkt(s));
        }
        assert_eq!(pull(&mut jb, false), Step::Decode(0));
        assert_eq!(pull(&mut jb, false), Step::Decode(1));
        // seq 0 arriving now is behind the play head (2) ⇒ stale.
        assert_eq!(jb.push(pkt(0)), PushOutcome::Stale);
    }

    #[test]
    fn stale_beyond_forward_window_dropped() {
        let mut jb = jb();
        for s in 0..4 {
            jb.push(pkt(s));
        }
        pull(&mut jb, false); // establish play head at 0, advance to 1
        assert_eq!(jb.push(pkt(1 + JITTER_STALE_WINDOW + 1)), PushOutcome::Stale);
    }

    #[test]
    fn gap_with_lookahead_conceals_then_decodes() {
        let mut jb = jb();
        for s in [0u32, 1, 3, 4] {
            jb.push(pkt(s)); // 2 is lost; 4 distinct packets reach the prefill floor
        }
        assert_eq!(pull(&mut jb, false), Step::Decode(0));
        assert_eq!(pull(&mut jb, false), Step::Decode(1));
        assert_eq!(pull(&mut jb, false), Step::Conceal); // seq 2 lost, 3/4 buffered
        assert_eq!(pull(&mut jb, false), Step::Decode(3));
        assert_eq!(pull(&mut jb, false), Step::Decode(4));
    }

    #[test]
    fn starvation_conceals_up_to_budget_then_silence() {
        let mut jb = jb();
        for s in 0..4 {
            jb.push(pkt(s));
        }
        for s in 0..4 {
            assert_eq!(pull(&mut jb, false), Step::Decode(s as u8));
        }
        // Buffer drained, nothing arriving: conceal PLC_MAX times, then silence.
        for _ in 0..PLC_MAX_CONSECUTIVE {
            assert_eq!(pull(&mut jb, false), Step::Conceal);
        }
        assert_eq!(pull(&mut jb, false), Step::Silence);
        assert!(jb.take_starved());
        // After reset it re-prebuffers rather than concealing forever.
        assert_eq!(pull(&mut jb, false), Step::Silence);
    }

    #[test]
    fn abandoned_and_drained_reports_finished() {
        let mut jb = jb();
        for s in 0..4 {
            jb.push(pkt(s));
        }
        for s in 0..4 {
            assert_eq!(pull(&mut jb, true), Step::Decode(s as u8));
        }
        assert_eq!(pull(&mut jb, true), Step::Finished);
    }

    #[test]
    fn abandoned_before_prefill_flushes_then_finishes() {
        let mut jb = jb();
        jb.push(pkt(0));
        jb.push(pkt(1)); // fewer than target_frames
        assert_eq!(pull(&mut jb, true), Step::Decode(0));
        assert_eq!(pull(&mut jb, true), Step::Decode(1));
        assert_eq!(pull(&mut jb, true), Step::Finished);
    }

    #[test]
    fn wrapping_seq_boundary_ordering() {
        let hi = Seq(u32::MAX);
        let lo = Seq(0);
        assert!(lo.newer_than(hi)); // 0 is one past u32::MAX
        assert!(!hi.newer_than(lo));
        assert_eq!(lo.ahead_of(hi), 1);
        assert_eq!(hi.ahead_of(lo), u32::MAX);
    }

    #[test]
    fn reorder_and_play_across_u32_wrap() {
        let mut jb = jb();
        let base = u32::MAX - 1; // seqs: MAX-1, MAX, 0, 1
        for d in [0u32, 1, 3, 2] {
            jb.push(pkt(base.wrapping_add(d)));
        }
        for d in 0..4u32 {
            let expect = base.wrapping_add(d) as u8;
            assert_eq!(pull(&mut jb, false), Step::Decode(expect));
        }
    }
}
