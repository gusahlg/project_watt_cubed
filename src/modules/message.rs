//! What the game tells the player: network chat, joins and leaves, and the core's notices, as
//! events for [`Mod::on_message`](super::Mod::on_message). The core shows none of it itself; a
//! mod (the chat) prints what it wants, and a notice no mod showed goes to stderr.

use super::frame::Channel;

/// How serious a notice is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NoticeLevel {
    /// Status: a save restored, a server's mod list.
    Info,
    /// Something degraded but the game goes on: an interrupted link, a refused change.
    Warning,
    /// Something failed: an autosave, the audio device.
    Error,
}

/// One line the core reports to the player.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Notice {
    pub level: NoticeLevel,
    pub text: String,
}

/// One event of the message stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Message<'a> {
    /// A chat line the server relayed, our own included (the server echoes it back).
    Chat { from: &'a str, channel: Channel, text: &'a str },
    /// A player joined the server.
    Joined { name: &'a str },
    /// A player left the server.
    Left { name: &'a str },
    /// A notice from the core.
    Notice(&'a Notice),
}

/// The core's notice sink: the save code, the audio service and the session write here, and the
/// game hands each notice to the mods once per frame. Empty between frames, so a quiet frame
/// does no work.
#[derive(Debug, Default)]
pub struct Notices {
    queue: Vec<Notice>,
}

impl Notices {
    /// Queue a notice.
    pub fn push(&mut self, level: NoticeLevel, text: impl Into<String>) {
        self.queue.push(Notice { level, text: text.into() });
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// The queued notices, oldest first.
    pub fn iter(&self) -> impl Iterator<Item = &Notice> {
        self.queue.iter()
    }

    /// Move every notice of `other` to the end of this queue, keeping `other`'s capacity.
    pub fn append(&mut self, other: &mut Notices) {
        self.queue.append(&mut other.queue);
    }

    /// Remove and return the queued notices, oldest first. The capacity stays.
    pub fn drain(&mut self) -> std::vec::Drain<'_, Notice> {
        self.queue.drain(..)
    }
}
