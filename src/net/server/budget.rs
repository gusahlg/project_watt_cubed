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
pub(super) struct RateWindow {
    stamps: VecDeque<Instant>,
    limit: u32,
}

impl RateWindow {
    pub(super) fn new(limit: u32) -> Self {
        Self { stamps: VecDeque::new(), limit }
    }

    pub(super) fn allow(&mut self, now: Instant) -> bool {
        const PERIOD: Duration = Duration::from_secs(1);
        while self.stamps.front().is_some_and(|t| now.saturating_duration_since(*t) >= PERIOD) {
            self.stamps.pop_front();
        }
        if self.stamps.len() as u32 >= self.limit {
            return false;
        }
        self.stamps.push_back(now);
        true
    }
}

/// One window per message kind. Cruise, hello, tool use, and mod channels are not here.
pub(super) struct KindBudget {
    chat: RateWindow,
    swing: RateWindow,
    edit: RateWindow,
    set_time: RateWindow,
    movement: RateWindow,
    ping: RateWindow,
    teleport: RateWindow,
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
        | ClientMessage::ToolUse { .. }
        | ClientMessage::ModData { .. } => return Charge::Pass,
        ClientMessage::Chat { .. } => (&mut budgets.chat, false),
        ClientMessage::Swing => (&mut budgets.swing, false),
        ClientMessage::Edit { .. } => (&mut budgets.edit, true),
        ClientMessage::SetTime { .. } => (&mut budgets.set_time, true),
        ClientMessage::Move { .. } => (&mut budgets.movement, false),
        ClientMessage::Ping { .. } => (&mut budgets.ping, false),
        ClientMessage::Teleport { .. } => (&mut budgets.teleport, true),
    };
    if window.allow(now) { Charge::Pass } else if answer { Charge::Answer } else { Charge::Drop }
}
