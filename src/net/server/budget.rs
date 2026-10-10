//! Per-connection rate budgets: one sliding window per message kind, and the mod-channel windows.
use super::*;

/// The connection's [`MOD_DATA_RATE`] window, then one [`RateWindow`] per channel name.
/// The first packet of a channel allocates the slot, up to [`MAX_CHANNELS`]; later
/// packets scan the small vec.
pub(super) struct ChannelBudget {
    total: RateWindow,
    windows: Vec<(Arc<str>, RateWindow)>,
}

impl ChannelBudget {
    pub(super) fn new() -> Self {
        Self { total: RateWindow::new(MOD_DATA_RATE), windows: Vec::new() }
    }

    pub(super) fn allow(&mut self, channel: &protocol::Channel, now: Instant) -> bool {
        if !self.total.allow(now) {
            return false;
        }
        let name = channel.as_str();
        if let Some((_, window)) = self.windows.iter_mut().find(|(key, _)| key.as_ref() == name) {
            return window.allow(now);
        }
        if self.windows.len() >= MAX_CHANNELS {
            return false;
        }
        let mut window = RateWindow::new(CHANNEL_RATE_LIMIT);
        let ok = window.allow(now);
        self.windows.push((channel.share(), window));
        ok
    }
}

/// Sliding 1-second window: a stamp ages out once a full second has passed, so
/// dumping a full budget on both sides of a second boundary cannot double it.
/// The stamps live in a fixed ring of `limit` slots, oldest at `head`.
pub(super) struct RateWindow {
    stamps: Box<[Instant]>,
    head: usize,
    len: usize,
}

impl RateWindow {
    pub(super) fn new(limit: u32) -> Self {
        Self { stamps: vec![Instant::now(); limit as usize].into_boxed_slice(), head: 0, len: 0 }
    }

    pub(super) fn allow(&mut self, now: Instant) -> bool {
        const PERIOD: Duration = Duration::from_secs(1);
        let slots = self.stamps.len();
        while self.len > 0 && now.saturating_duration_since(self.stamps[self.head]) >= PERIOD {
            self.head = (self.head + 1) % slots;
            self.len -= 1;
        }
        if self.len >= slots {
            return false;
        }
        self.stamps[(self.head + self.len) % slots] = now;
        self.len += 1;
        true
    }
}

/// One window per message kind. Cruise, hello and mod channels are not here: cruise is a
/// declared state, and mod channels have their own [`ChannelBudget`].
pub(super) struct KindBudget {
    chat: RateWindow,
    swing: RateWindow,
    edit: RateWindow,
    set_time: RateWindow,
    movement: RateWindow,
    ping: RateWindow,
    teleport: RateWindow,
    tool: RateWindow,
}

impl KindBudget {
    pub(super) fn new() -> Self {
        Self {
            chat: RateWindow::new(CHAT_RATE),
            swing: RateWindow::new(SWING_RATE),
            edit: RateWindow::new(EDIT_RATE),
            set_time: RateWindow::new(SET_TIME_RATE),
            movement: RateWindow::new(MOVE_RATE),
            ping: RateWindow::new(PING_RATE),
            teleport: RateWindow::new(TELEPORT_RATE),
            tool: RateWindow::new(TOOL_RATE_LIMIT),
        }
    }
}

/// What to do with one client frame against its kind's budget.
pub(super) enum Charge {
    /// Under budget, or a kind that is not counted here.
    Pass,
    /// Over budget and safe to ignore.
    Drop,
    /// Over budget, but the client is waiting on an answer.
    Answer,
}

pub(super) fn charge(budgets: &mut KindBudget, msg: &ClientMessage, now: Instant) -> Charge {
    let (window, answer) = match msg {
        ClientMessage::Cruise { .. }
        | ClientMessage::Hello { .. }
        | ClientMessage::ModData { .. } => return Charge::Pass,
        ClientMessage::Chat { .. } => (&mut budgets.chat, false),
        ClientMessage::Swing => (&mut budgets.swing, false),
        ClientMessage::Edit { .. } => (&mut budgets.edit, true),
        ClientMessage::SetTime { .. } => (&mut budgets.set_time, true),
        ClientMessage::Move { .. } => (&mut budgets.movement, false),
        ClientMessage::Ping { .. } => (&mut budgets.ping, false),
        ClientMessage::Teleport { .. } => (&mut budgets.teleport, true),
        // A refused use is still answered, so the client's swing resolves.
        ClientMessage::ToolUse { .. } => (&mut budgets.tool, true),
    };
    if window.allow(now) { Charge::Pass } else if answer { Charge::Answer } else { Charge::Drop }
}
