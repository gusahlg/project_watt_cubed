//! Getting frames out: per-player outboxes and their writer threads, broadcasts, relays and the join backlog.
use super::*;

/// A recipient and its encoded frame, gathered under the state lock for [`queue`].
pub(super) type PendingSend = (u32, Arc<[u8]>);

/// One player's outbound frames, bounded at [`OUT_CAPACITY`]. The writer thread
/// parks while nothing is queued, and [`push`](Self::push) does not wake it, so a
/// broadcast under the state lock wakes writers only after the lock is released.
/// Dropping a sender wakes the writer so it notices when the last one is gone.
#[derive(Clone)]
pub(super) struct Outbox {
    /// `None` only inside `drop`, which lets go of the sender before waking.
    pub(super) tx: Option<SyncSender<Arc<[u8]>>>,
    pub(super) writer: Arc<Writer>,
}

/// What both ends of an [`Outbox`] share. Test builds count the frames waiting.
#[derive(Default)]
pub(super) struct Writer {
    thread: OnceLock<Thread>,
    #[cfg(test)]
    depth: AtomicUsize,
}

impl Writer {
    pub(super) fn wake(&self) {
        if let Some(thread) = self.thread.get() {
            thread.unpark();
        }
    }
}

/// The writer thread's end of an [`Outbox`].
pub(super) struct Outgoing {
    rx: Receiver<Arc<[u8]>>,
    writer: Arc<Writer>,
}

pub(super) fn outbox(capacity: usize) -> (Outbox, Outgoing) {
    let (tx, rx) = sync_channel(capacity);
    let writer = Arc::new(Writer::default());
    (Outbox { tx: Some(tx), writer: writer.clone() }, Outgoing { rx, writer })
}

impl Outbox {
    /// Queue `frame` and wake the writer.
    pub(super) fn try_send(&self, frame: Arc<[u8]>) -> Result<(), TrySendError<Arc<[u8]>>> {
        self.push(frame)?;
        self.writer.wake();
        Ok(())
    }

    /// Queue `frame` without waking the writer: the caller wakes it later.
    pub(super) fn push(&self, frame: Arc<[u8]>) -> Result<(), TrySendError<Arc<[u8]>>> {
        // Counted before the push, so the writer's decrement can never run first.
        #[cfg(test)]
        let depth = self.writer.depth.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
        let sent = match &self.tx {
            Some(tx) => tx.try_send(frame),
            None => Err(TrySendError::Disconnected(frame)),
        };
        #[cfg(test)]
        match &sent {
            Ok(()) => load::QUEUE.record(depth as u64),
            Err(_) => {
                self.writer.depth.fetch_sub(1, Ordering::Relaxed);
            }
        }
        sent
    }
}

impl Drop for Outbox {
    fn drop(&mut self) {
        drop(self.tx.take());
        self.writer.wake();
    }
}

/// Writers to wake once the state lock is released. Dropping it wakes them.
pub(super) struct Wake(pub(super) Vec<Arc<Writer>>);

impl Wake {
    pub(super) fn join(&mut self, mut other: Wake) {
        self.0.append(&mut other.0);
    }
}

impl Drop for Wake {
    fn drop(&mut self) {
        for writer in &self.0 {
            writer.wake();
        }
    }
}

impl Outgoing {
    /// Frame what is queued into `batch`, up to [`protocol::WRITE_BATCH`] bytes, parking while
    /// nothing is. `Ok(false)` once every sender is gone and the queue is empty; an
    /// error for a frame past the cap.
    pub(super) fn take(&self, batch: &mut Vec<u8>) -> io::Result<bool> {
        batch.clear();
        loop {
            let mut gone = false;
            protocol::pump(batch, || match self.rx.try_recv() {
                Ok(frame) => {
                    #[cfg(test)]
                    self.writer.depth.fetch_sub(1, Ordering::Relaxed);
                    Some(frame)
                }
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => {
                    gone = true;
                    None
                }
            })?;
            if !batch.is_empty() {
                return Ok(true);
            }
            if gone {
                return Ok(false);
            }
            thread::park();
        }
    }
}

/// One frame waiting out a join, and when it was queued.
pub(super) struct Queued {
    pub(super) at: Instant,
    pub(super) frame: Arc<[u8]>,
}

pub(super) fn spawn_writer(
    writer_rt: Arc<Runtime>,
    mut send: SendStream,
    rx: Outgoing,
    kick: Arc<Notify>,
) -> thread::JoinHandle<()> {
    // A write error ends the writer and wakes the reader, so the client is
    // dropped instead of left half-open. A clean channel close (depart) does
    // not kick: the reader has already exited. QUIC has no user flush.
    thread::spawn(move || {
        drain_writer(rx, &kick, |batch| {
            writer_rt.block_on(send.write_all(batch)).map_err(io::Error::other)
        });
    })
}

/// Write queued frames, one batch per write, until every sender is gone. The
/// first error (a write, or a frame past the cap) notifies `kick` and returns.
pub(super) fn drain_writer(
    rx: Outgoing,
    kick: &Notify,
    mut write: impl FnMut(&[u8]) -> io::Result<()>,
) {
    let _ = rx.writer.thread.set(thread::current());
    let mut batch = Vec::new();
    loop {
        match rx.take(&mut batch) {
            Ok(true) => {
                if write(&batch).is_err() {
                    kick.notify_one();
                    return;
                }
            }
            Ok(false) => return,
            Err(_) => {
                kick.notify_one();
                return;
            }
        }
    }
}

/// Queue a private line from "server" for a ready player.
pub(super) fn tell(h: &PlayerHandle, id: u32, text: &str, sends: &mut Vec<PendingSend>) {
    if h.ready {
        sends.push((id, server_says(chat::GLOBAL, text.into())));
    }
}

/// A chat line from the server itself: player id [`WORLD_PLAYER`], named "server".
pub(super) fn server_says(channel: u8, text: Arc<str>) -> Arc<[u8]> {
    static NAME: LazyLock<Arc<str>> = LazyLock::new(|| Arc::from("server"));
    ServerMessage::Chat { from_id: WORLD_PLAYER, from_name: Arc::clone(&NAME), channel, text }.frame()
}

/// Queue `sends` in order under the state lock, so no frame overtakes a state change
/// queued before it. A full queue marks its owner for the kick pass. The returned
/// [`Wake`] wakes the writers once the caller has dropped the guard.
pub(super) fn queue(state: &State, sends: &mut Vec<PendingSend>) -> Wake {
    let mut slow = Vec::new();
    let mut wake = Vec::with_capacity(sends.len());
    for (pid, frame) in sends.drain(..) {
        let Some(h) = state.players.get(&pid) else { continue };
        match h.out.push(frame) {
            Ok(()) => wake.push(h.out.writer.clone()),
            Err(TrySendError::Full(_)) => {
                if !slow.contains(&pid) {
                    slow.push(pid);
                }
            }
            Err(TrySendError::Disconnected(_)) => {}
        }
    }
    kick_slow(state, &slow);
    Wake(wake)
}

/// Channel traffic is loss-tolerant: `try_send` and DROP on a full/closed queue,
/// never counted toward the slow-client kick ([`kick_slow`]/[`OUT_CAPACITY`]).
/// Relayed only to the sender's visible interest set. The sender id is stamped
/// here; the client's own message does not carry it. Size is capped by the codec.
pub(super) fn on_mod_data(
    shared: &Arc<Mutex<State>>,
    id: u32,
    channel: protocol::Channel,
    seq: u32,
    bytes: protocol::ModBytes,
) {
    relay(shared, id, ServerMessage::PeerModData { channel, sender: id, seq, bytes }.frame());
}

/// Loss-tolerant: `frame` reaches `from`'s visible, ready peers, and a full queue drops
/// it without a kick. The writers wake after the lock is released.
pub(super) fn relay(shared: &Arc<Mutex<State>>, from: u32, frame: Arc<[u8]>) {
    let wake = {
        let state = shared.lock_recover();
        let Some(speaker) = state.players.get(&from) else { return };
        let mut wake = Vec::with_capacity(speaker.visible.len());
        for &pid in &speaker.visible {
            if let Some(other) = state.players.get(&pid)
                && other.ready
                && other.out.push(Arc::clone(&frame)).is_ok()
            {
                wake.push(other.out.writer.clone());
            }
        }
        Wake(wake)
    };
    drop(wake);
}

/// Encodes `msg` just once for every recipient. Players whose queue is full
/// are force-closed (they've fallen too far behind).
/// Frames are queued under the lock, in order; the returned [`Wake`] wakes the
/// writers, after the lock is released when the caller drops the guard first.
pub(super) fn broadcast(state: &mut State, msg: &ServerMessage, want: impl Fn(u32, &PlayerHandle) -> bool) -> Wake {
    let joined = match msg {
        ServerMessage::PeerJoined { id, .. } => Some(*id),
        _ => None,
    };
    broadcast_frame(state, msg.frame(), joined, want)
}

/// [`broadcast`] of an encoded frame. `joined` is the peer a `PeerJoined` frame
/// announces, so each player hears it once.
pub(super) fn broadcast_frame(
    state: &mut State,
    frame: Arc<[u8]>,
    joined: Option<u32>,
    want: impl Fn(u32, &PlayerHandle) -> bool,
) -> Wake {
    let mut slow = Vec::new();
    let mut wake = Vec::new();
    for (&pid, h) in state.players.iter_mut() {
        if !want(pid, h) {
            continue;
        }
        // Already queued for this peer in its join roster: skip it once.
        if let Some(jid) = joined && h.announced.remove(&jid) {
            continue;
        }
        if !h.ready {
            // Bootstrapping: buffer so the frame lands AFTER the snapshot.
            // Cosmetic frames are dropped instead of kicking the joiner.
            if !enqueue_backlog(h, Arc::clone(&frame), Instant::now()) {
                slow.push(pid);
            }
            continue;
        }
        match h.out.push(frame.clone()) {
            Ok(()) => wake.push(h.out.writer.clone()),
            Err(TrySendError::Full(_)) => slow.push(pid),
            Err(TrySendError::Disconnected(_)) => {}
        }
    }
    kick_slow(state, &slow);
    Wake(wake)
}

pub(super) fn broadcast_all(shared: &Arc<Mutex<State>>, msg: &ServerMessage, except: Option<u32>) {
    let mut state = shared.lock_recover();
    let wake = broadcast(&mut state, msg, |pid, _| Some(pid) != except);
    drop(state);
    drop(wake);
}

/// Force-close clients that couldn't keep up. Their reader threads then wake, error,
/// and run the normal cleanup path (emitting `PeerLeft`).
pub(super) fn kick_slow(state: &State, ids: &[u32]) {
    for id in ids {
        if let Some(h) = state.players.get(id) {
            // `notify_one` stores a permit if the reader isn't currently
            // awaiting, so a kick is never missed. The flag is what bootstrap
            // sends watch, because that thread is not in `client_loop` yet.
            h.kicked.store(true, Ordering::Relaxed);
            h.kick.notify_one();
        }
    }
}

/// Queue `frame` for a player who is not ready yet. False when an essential frame
/// does not fit after cosmetic frames have been dropped, when an essential frame has
/// waited past [`BACKLOG_AGE`], or when the player is already kicked.
pub(super) fn enqueue_backlog(h: &mut PlayerHandle, frame: Arc<[u8]>, now: Instant) -> bool {
    if h.kicked.load(Ordering::Relaxed) || !trim_aged(h, now) {
        return false;
    }
    let cosmetic = protocol::is_cosmetic(&frame);
    while h.backlog_bytes + frame.len() > BACKLOG_BYTES {
        if !drop_oldest_cosmetic(h) {
            return cosmetic;
        }
    }
    h.backlog_bytes += frame.len();
    h.backlog.push_back(Queued { at: now, frame });
    true
}

/// Drop frames older than [`BACKLOG_AGE`]. The backlog is in time order, so the aged frames
/// are a prefix, dropped in one drain. False, dropping nothing, when an aged frame is
/// essential: losing it would desync the joiner.
pub(super) fn trim_aged(h: &mut PlayerHandle, now: Instant) -> bool {
    let aged = h.backlog.iter().take_while(|q| now.saturating_duration_since(q.at) >= BACKLOG_AGE).count();
    let mut bytes = 0;
    for queued in h.backlog.range(..aged) {
        if !protocol::is_cosmetic(&queued.frame) {
            return false;
        }
        bytes += queued.frame.len();
    }
    h.backlog_bytes = h.backlog_bytes.saturating_sub(bytes);
    h.backlog.drain(..aged);
    true
}

pub(super) fn drop_oldest_cosmetic(h: &mut PlayerHandle) -> bool {
    let Some(index) = h.backlog.iter().position(|queued| protocol::is_cosmetic(&queued.frame)) else {
        return false;
    };
    if let Some(queued) = h.backlog.remove(index) {
        h.backlog_bytes = h.backlog_bytes.saturating_sub(queued.frame.len());
    }
    true
}

/// Swing relay: the swinger's visible set, same as voice. A full queue drops
/// the frame; a swing is cosmetic and never a kick.
pub(super) fn relay_swing(shared: &Arc<Mutex<State>>, id: u32) {
    relay(shared, id, ServerMessage::PeerSwing { id }.frame());
}
