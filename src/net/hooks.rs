//! Server-side mod seam: plain-data hooks so communities can add rules without
//! forking the server. Signatures stay serialisable (ids, names, coords, specs)
//! so a future manifest/wasm boundary remains possible.
//!
//! Hook bodies run **outside** the roster/ledger lock: the server collects facts
//! under the lock, drops it, then calls into this table. A broken hook is caught
//! and treated as [`Verdict::Allow`] so it cannot take the server down.

use std::collections::HashSet;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

/// A unit of server-side policy. Every method but [`id`](Self::id) has a
/// default, so a mod implements only the events it cares about.
pub trait ServerMod: Send {
    /// Short, stable name used in panic logs.
    fn id(&self) -> &'static str;

    /// Gate a would-be cell change before the ledger decides. A [`Verdict::Deny`]
    /// takes the same `EditAck { accepted: false }` path a lost revision race
    /// does, so the client's `EditRejected.restore` is true iff no newer
    /// confirmed revision has landed on the cell.
    fn validate_edit(&mut self, intent: &EditIntent) -> Verdict {
        let _ = intent;
        Verdict::Allow
    }

    fn on_join(&mut self, player: &JoinFacts) {
        let _ = player;
    }

    fn on_leave(&mut self, player: &JoinFacts) {
        let _ = player;
    }

    /// Gate a chat line. A [`Verdict::Deny`] drops the broadcast and delivers
    /// `reason` to the sender only (existing `Chat` message, `from_id` 0).
    fn on_chat(&mut self, msg: &ChatFacts) -> Verdict {
        let _ = msg;
        Verdict::Allow
    }
}

/// Allow proceeds; the first [`Deny`](Self::Deny) in registration order wins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Allow,
    Deny { reason: Arc<str> },
}

/// A proposed cell change, owned and socket-free.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditIntent {
    pub player: u32,
    pub name: Arc<str>,
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub spec: Arc<str>,
    pub expect: u32,
}

/// Roster facts at join or leave (block coords of spawn / last position).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinFacts {
    pub player: u32,
    pub name: Arc<str>,
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

/// A chat line after the server has sanitised it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatFacts {
    pub player: u32,
    pub name: Arc<str>,
    pub channel: u8,
    pub text: Arc<str>,
}

/// Installed mods plus the set of ids already logged for a panic.
pub(crate) struct Table {
    mods: Vec<Box<dyn ServerMod>>,
    panicked: HashSet<&'static str>,
}

impl Table {
    pub(crate) fn new(mods: Vec<Box<dyn ServerMod>>) -> Self {
        Self { mods, panicked: HashSet::new() }
    }

    pub(crate) fn validate_edit(&mut self, intent: &EditIntent) -> Verdict {
        for m in &mut self.mods {
            let id = m.id();
            match catch_unwind(AssertUnwindSafe(|| m.validate_edit(intent))) {
                Ok(Verdict::Deny { reason }) => return Verdict::Deny { reason },
                Ok(Verdict::Allow) => {}
                Err(_) => note_panic(&mut self.panicked, id),
            }
        }
        Verdict::Allow
    }

    pub(crate) fn on_join(&mut self, player: &JoinFacts) {
        for m in &mut self.mods {
            let id = m.id();
            if catch_unwind(AssertUnwindSafe(|| m.on_join(player))).is_err() {
                note_panic(&mut self.panicked, id);
            }
        }
    }

    pub(crate) fn on_leave(&mut self, player: &JoinFacts) {
        for m in &mut self.mods {
            let id = m.id();
            if catch_unwind(AssertUnwindSafe(|| m.on_leave(player))).is_err() {
                note_panic(&mut self.panicked, id);
            }
        }
    }

    pub(crate) fn on_chat(&mut self, msg: &ChatFacts) -> Verdict {
        for m in &mut self.mods {
            let id = m.id();
            match catch_unwind(AssertUnwindSafe(|| m.on_chat(msg))) {
                Ok(Verdict::Deny { reason }) => return Verdict::Deny { reason },
                Ok(Verdict::Allow) => {}
                Err(_) => note_panic(&mut self.panicked, id),
            }
        }
        Verdict::Allow
    }
}

fn note_panic(panicked: &mut HashSet<&'static str>, id: &'static str) {
    if panicked.insert(id) {
        eprintln!("server hook '{id}' panicked; treating as Allow");
    }
}

#[cfg(test)]
pub(crate) struct Recording {
    pub id: &'static str,
    pub events: Arc<std::sync::Mutex<Vec<Recorded>>>,
    pub deny_edit: bool,
    pub deny_chat: bool,
    pub panic_edit: bool,
    pub panic_chat: bool,
    pub panic_join: bool,
    pub reason: Arc<str>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Recorded {
    Edit(EditIntent),
    Join(JoinFacts),
    Leave(JoinFacts),
    Chat(ChatFacts),
}

#[cfg(test)]
impl Recording {
    pub(crate) fn new(id: &'static str) -> (Self, Arc<std::sync::Mutex<Vec<Recorded>>>) {
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        (
            Self {
                id,
                events: events.clone(),
                deny_edit: false,
                deny_chat: false,
                panic_edit: false,
                panic_chat: false,
                panic_join: false,
                reason: Arc::from("denied"),
            },
            events,
        )
    }

    fn push(&self, event: Recorded) {
        self.events.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(event);
    }
}

#[cfg(test)]
impl ServerMod for Recording {
    fn id(&self) -> &'static str {
        self.id
    }

    fn validate_edit(&mut self, intent: &EditIntent) -> Verdict {
        self.push(Recorded::Edit(intent.clone()));
        if self.panic_edit {
            panic!("test hook panic");
        }
        if self.deny_edit {
            Verdict::Deny { reason: self.reason.clone() }
        } else {
            Verdict::Allow
        }
    }

    fn on_join(&mut self, player: &JoinFacts) {
        self.push(Recorded::Join(player.clone()));
        if self.panic_join {
            panic!("test hook panic");
        }
    }

    fn on_leave(&mut self, player: &JoinFacts) {
        self.push(Recorded::Leave(player.clone()));
    }

    fn on_chat(&mut self, msg: &ChatFacts) -> Verdict {
        self.push(Recorded::Chat(msg.clone()));
        if self.panic_chat {
            panic!("test hook panic");
        }
        if self.deny_chat {
            Verdict::Deny { reason: self.reason.clone() }
        } else {
            Verdict::Allow
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intent(player: u32) -> EditIntent {
        EditIntent {
            player,
            name: "p".into(),
            x: 1,
            y: 2,
            z: 3,
            spec: "air".into(),
            expect: 0,
        }
    }

    fn join(player: u32, name: &str) -> JoinFacts {
        JoinFacts { player, name: name.into(), x: 0, y: 1, z: 2 }
    }

    #[test]
    fn first_deny_wins_and_later_hooks_do_not_run() {
        let (mut a, log_a) = Recording::new("a");
        a.deny_edit = true;
        let (b, log_b) = Recording::new("b");
        let mut table = Table::new(vec![Box::new(a), Box::new(b)]);
        match table.validate_edit(&intent(1)) {
            Verdict::Deny { reason } => assert_eq!(&*reason, "denied"),
            other => panic!("expected Deny, got {other:?}"),
        }
        assert_eq!(log_a.lock().unwrap().len(), 1);
        assert!(log_b.lock().unwrap().is_empty(), "short-circuit on first Deny");
    }

    #[test]
    fn panicking_hook_is_allow_and_logged_once() {
        let (mut boom, log_boom) = Recording::new("boom");
        boom.panic_edit = true;
        let (ok, log_ok) = Recording::new("ok");
        let mut table = Table::new(vec![Box::new(boom), Box::new(ok)]);
        assert_eq!(table.validate_edit(&intent(1)), Verdict::Allow);
        assert_eq!(table.validate_edit(&intent(2)), Verdict::Allow);
        assert_eq!(table.panicked.len(), 1, "one log entry per hook id");
        assert!(table.panicked.contains("boom"));
        assert_eq!(log_boom.lock().unwrap().len(), 2);
        assert_eq!(log_ok.lock().unwrap().len(), 2, "Allow continues the chain");
    }

    #[test]
    fn join_leave_facts_arrive_in_order() {
        let (rec, log) = Recording::new("rec");
        let mut table = Table::new(vec![Box::new(rec)]);
        let a = join(1, "alice");
        let b = join(2, "bob");
        table.on_join(&a);
        table.on_join(&b);
        table.on_leave(&a);
        let events = log.lock().unwrap().clone();
        assert_eq!(
            events,
            vec![Recorded::Join(a.clone()), Recorded::Join(b), Recorded::Leave(a)]
        );
    }

    #[test]
    fn panicking_join_is_neutralised() {
        let (mut boom, _) = Recording::new("boom");
        boom.panic_join = true;
        let (ok, log_ok) = Recording::new("ok");
        let mut table = Table::new(vec![Box::new(boom), Box::new(ok)]);
        table.on_join(&join(1, "alice"));
        assert_eq!(log_ok.lock().unwrap().len(), 1);
        assert!(table.panicked.contains("boom"));
    }
}
