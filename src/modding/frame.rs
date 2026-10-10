//! The per-frame mod hook: [`Mod::on_frame`](super::Mod::on_frame) and the context it gets.
//!
//! The core offers three generic capabilities here and knows nothing of what a mod builds on
//! them (chat, commands, flight):
//!
//! - **Immediate actions.** An [`Action`](super::Action) with `immediate: true` is sampled every
//!   frame, even with the `mod_logic` lane off, and reaches only [`FrameContext::action`], never the
//!   cadence-controlled `update`.
//! - **Text capture.** A mod asks for the keyboard with [`FrameContext::capture_text`]. While it
//!   holds it, the router reads typed text instead of gameplay keys, the world takes no input, and
//!   only that mod sees a [`TextFrame`]. The press that started the capture does not also type its
//!   character. Escape always ends a capture: the holder sees `escape` once, then the core takes
//!   the keyboard back, so Escape never leaves the world while a mod is typing.
//! - **Game state.** [`GameContext`] is the whole-game state the core owns (player, world, sky,
//!   settings and the packages' options) plus three out-queues: audio facts, chat lines to send,
//!   and notices. The core
//!   follows up on what a hook changed: it applies and saves changed settings, re-mixes the audio,
//!   shares a changed clock with the server, keeps the server's day length, and streams a moved
//!   player's surroundings at once and reports the move as a teleport.
//!
//! **Performance:** one virtual call per mod per frame. A frame with no input builds the context
//! on the stack from retained buffers and allocates nothing.

use crate::audio::GameEvent;
use crate::input::intent::EditKey;
use crate::player::Player;
use crate::settings::{Options, OptionsRef, OptionsView, Settings};
use crate::sky::Sky;
use crate::world::World;

use super::message::{NoticeLevel, Notices};
use super::{ActionSet, VisualMask};

/// Where a chat line goes. Local chat reaches the players near you; global chat reaches everyone
/// on the server.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Channel {
    Local,
    Global,
}

impl Channel {
    /// The channel byte on the wire.
    pub(crate) fn wire(self) -> u8 {
        match self {
            Channel::Local => crate::net::chat::LOCAL,
            Channel::Global => crate::net::chat::GLOBAL,
        }
    }

    /// The channel a wire byte names. Anything but global reads as local.
    pub(crate) fn from_wire(byte: u8) -> Self {
        if byte == crate::net::chat::GLOBAL { Channel::Global } else { Channel::Local }
    }
}

/// One frame of typing, for the mod that holds the keyboard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextFrame<'a> {
    /// Characters typed this frame, layout and shift applied.
    pub chars: &'a [char],
    /// At most one editing key (arrows, Backspace, Tab, Enter, history).
    pub edit: Option<EditKey>,
    /// Escape was pressed. The core ends the capture after this frame.
    pub escape: bool,
}

/// What a frame hook may read and change: the whole-game state the core owns. Build one for a
/// test with [`GameContext::new`].
#[non_exhaustive]
pub struct GameContext<'a> {
    pub player: &'a mut Player,
    pub world: &'a mut World,
    pub sky: &'a mut Sky,
    /// True when a server owns the session: it sets the day length, may refuse a teleport, and
    /// receives chat. In single player there is no wire and [`send_chat`](Self::send_chat) is a no-op.
    pub networked: bool,
    /// True while the detached (free) camera holds the player still.
    pub detached: bool,
    /// Which visual groups this build provides.
    pub visuals: VisualMask,
    /// Audio facts for the sounds mod, such as [`GameEvent::VoiceTest`].
    pub events: Vec<GameEvent>,
    settings: &'a mut Settings,
    /// The settings as they were before the first [`settings_mut`](Self::settings_mut) this frame.
    before: Option<Box<Settings>>,
    options: OptionsSlot<'a>,
    /// [`Options::revision`] when the context was built.
    options_revision: u64,
    chat_out: Vec<(Channel, String)>,
    notices: Notices,
}

impl<'a> GameContext<'a> {
    /// A single-player context over these handles, with every visual group on and empty queues.
    pub fn new(player: &'a mut Player, world: &'a mut World, settings: &'a mut Settings, sky: &'a mut Sky) -> Self {
        Self {
            player,
            world,
            sky,
            networked: false,
            detached: false,
            visuals: VisualMask::default(),
            events: Vec::new(),
            settings,
            before: None,
            options: OptionsSlot::Owned(Options::new()),
            options_revision: 0,
            chat_out: Vec::new(),
            notices: Notices::default(),
        }
    }

    /// This context over the packages' `options` (a fresh context has none).
    pub fn with_options(mut self, options: &'a mut Options) -> Self {
        self.options_revision = options.revision();
        self.options = OptionsSlot::Borrowed(options);
        self
    }

    /// Every tunable, the core's settings and the packages' options, read-only.
    pub fn options(&self) -> OptionsRef<'_> {
        OptionsRef::new(self.settings, self.options.get())
    }

    /// Every tunable, to change (`/set`). The core applies and saves what changed after the hook.
    pub fn options_mut(&mut self) -> OptionsView<'_> {
        OptionsView::new(self.settings, self.options.get_mut())
    }

    /// The player's settings.
    pub fn settings(&self) -> &Settings {
        self.settings
    }

    /// The player's settings, to change. The core applies and saves them after the hook when
    /// they differ from before.
    pub fn settings_mut(&mut self) -> &mut Settings {
        if self.before.is_none() {
            self.before = Some(Box::new(self.settings.clone()));
        }
        self.settings
    }

    /// Whether the settings differ from how they were before the first
    /// [`settings_mut`](Self::settings_mut), or a value changed through
    /// [`options_mut`](Self::options_mut).
    pub fn settings_changed(&self) -> bool {
        self.before.as_deref().is_some_and(|before| before != self.settings)
            || self.options.get().revision() != self.options_revision
    }

    /// Queue a chat line for the server. The core sends it after the hook (truncated to the
    /// protocol's limit); in single player it is dropped, so a chat mod echoes locally instead.
    pub fn send_chat(&mut self, channel: Channel, text: impl Into<String>) {
        self.chat_out.push((channel, text.into()));
    }

    /// The chat lines queued this frame, oldest first.
    pub fn chat_out(&self) -> &[(Channel, String)] {
        &self.chat_out
    }

    /// Report something to the player. A mod that shows messages prints it; with none, it goes to
    /// stderr.
    pub fn notice(&mut self, level: NoticeLevel, text: impl Into<String>) {
        self.notices.push(level, text);
    }

    /// The notices queued this frame.
    pub fn notices(&self) -> &Notices {
        &self.notices
    }

    /// Hand the core's retained buffers to this context (no allocation).
    pub(crate) fn with_queues(
        mut self,
        events: Vec<GameEvent>,
        chat_out: Vec<(Channel, String)>,
        notices: Notices,
    ) -> Self {
        self.events = events;
        self.chat_out = chat_out;
        self.notices = notices;
        self
    }

    /// Take the buffers back, with whatever the hooks queued.
    pub(crate) fn into_queues(self) -> (Vec<GameEvent>, Vec<(Channel, String)>, Notices, bool) {
        let changed = self.settings_changed();
        (self.events, self.chat_out, self.notices, changed)
    }
}

/// The options a [`GameContext`] reads: the core's (borrowed), or an empty set of its own.
enum OptionsSlot<'a> {
    Borrowed(&'a mut Options),
    Owned(Options),
}

impl OptionsSlot<'_> {
    fn get(&self) -> &Options {
        match self {
            OptionsSlot::Borrowed(options) => options,
            OptionsSlot::Owned(options) => options,
        }
    }

    fn get_mut(&mut self) -> &mut Options {
        match self {
            OptionsSlot::Borrowed(options) => options,
            OptionsSlot::Owned(options) => options,
        }
    }
}

/// The context of [`Mod::on_frame`](super::Mod::on_frame): the game state, this frame's immediate
/// actions, and the keyboard capture. Build one for a test with [`FrameContext::new`].
#[non_exhaustive]
pub struct FrameContext<'a> {
    pub game: GameContext<'a>,
    /// The window size in pixels.
    pub screen: (i32, i32),
    text: Option<TextFrame<'a>>,
    fired: ActionSet,
    ids: &'a [&'static str],
    /// Ids from [`FrameContext::set_action`]. Empty on the game path.
    extra: Vec<&'static str>,
    /// Which mod (by install index) holds the keyboard, as the hooks run.
    holder: Option<usize>,
    /// The mod whose hook is running.
    me: usize,
}

impl<'a> FrameContext<'a> {
    /// A test context: 800×600, no action fired, no typing, nobody holding the keyboard.
    pub fn new(game: GameContext<'a>) -> Self {
        Self { game, screen: (800, 600), text: None, fired: ActionSet::NONE, ids: &[], extra: Vec::new(), holder: None, me: 0 }
    }

    /// The game path's context.
    pub(crate) fn frame(
        game: GameContext<'a>,
        screen: (i32, i32),
        fired: ActionSet,
        ids: &'a [&'static str],
        text: Option<TextFrame<'a>>,
    ) -> Self {
        Self { game, screen, text, fired, ids, extra: Vec::new(), holder: None, me: 0 }
    }

    /// Whether the immediate action `id` fired this frame (any installed mod's action with that id).
    /// A frame with no action fired answers without looking at a name.
    pub fn action(&self, id: &str) -> bool {
        if self.fired.is_empty() {
            return false;
        }
        self.ids.iter().enumerate().any(|(i, name)| *name == id && self.fired.contains(i))
            || self.extra.iter().enumerate().any(|(i, name)| *name == id && self.fired.contains(self.ids.len() + i))
    }

    /// Mark `id` fired. For tests; the game path fills the fired set instead.
    pub fn set_action(&mut self, id: &'static str) {
        if let Some(i) = self.ids.iter().position(|name| *name == id) {
            self.fired.insert(i);
            return;
        }
        if let Some(i) = self.extra.iter().position(|name| *name == id) {
            self.fired.insert(self.ids.len() + i);
            return;
        }
        let i = self.ids.len() + self.extra.len();
        if i < ActionSet::CAP {
            self.extra.push(id);
            self.fired.insert(i);
        }
    }

    /// This frame's typing, if this mod holds the keyboard and the capture was in place when the
    /// frame's input was read.
    pub fn text(&self) -> Option<TextFrame<'a>> {
        self.text.filter(|_| self.holder == Some(self.me))
    }

    /// Ask for the keyboard (`true`) or give it back (`false`). Taking it fails, returning
    /// `false`, while another mod holds it. Typing starts on the next frame; the key that asked
    /// does not also type its character.
    pub fn capture_text(&mut self, on: bool) -> bool {
        match (on, self.holder) {
            (true, None) => {
                self.holder = Some(self.me);
                true
            }
            (true, Some(holder)) => holder == self.me,
            (false, Some(holder)) if holder == self.me => {
                self.holder = None;
                true
            }
            (false, _) => true,
        }
    }

    /// Whether this mod holds the keyboard.
    pub fn capturing(&self) -> bool {
        self.holder == Some(self.me)
    }

    /// For tests: make this context's mod the keyboard holder and give it `text` this frame.
    pub fn set_text(&mut self, text: TextFrame<'a>) {
        self.holder = Some(self.me);
        self.text = Some(text);
    }

    /// The host runs mod `index`'s hook next.
    pub(crate) fn enter(&mut self, index: usize) {
        self.me = index;
    }

    pub(crate) fn holder(&self) -> Option<usize> {
        self.holder
    }

    pub(crate) fn set_holder(&mut self, holder: Option<usize>) {
        self.holder = holder;
    }

    /// Whether Escape reached the holder this frame.
    pub(crate) fn escaped(&self) -> bool {
        self.text.is_some_and(|t| t.escape)
    }
}
