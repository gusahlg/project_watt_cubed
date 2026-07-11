//! Translate menu events into intents. Bindings and key-repeat are handled
//! upstream in the input router; this stage converts menu events and chars
//! into the intent alphabet.
use crate::input::intent::{EditKey, MenuEvent};
use crate::input::router;
use crate::menu::{Dir, Intent, TextOp};

/// Gather this frame's menu intents. Order matters within kinds: edits before other intents.
pub fn gather(m: &router::Menu) -> Vec<Intent> {
    let mut out = Vec::new();

    if m.event(MenuEvent::Up) {
        out.push(Intent::Nav(Dir::Prev));
    }
    // Tab walks forward through rows/fields, like Down.
    if m.event(MenuEvent::Down) || m.event(MenuEvent::NextTab) {
        out.push(Intent::Nav(Dir::Next));
    }
    if m.event(MenuEvent::Left) {
        out.push(Intent::Adjust(Dir::Prev));
    }
    if m.event(MenuEvent::Right) {
        out.push(Intent::Adjust(Dir::Next));
    }
    if m.event(MenuEvent::Confirm) || m.event(MenuEvent::Toggle) {
        out.push(Intent::Confirm);
    }
    if m.event(MenuEvent::Back) {
        out.push(Intent::Cancel);
    }

    for c in m.chars() {
        out.push(Intent::Edit(TextOp::Char(c)));
    }
    if m.event(MenuEvent::Delete) {
        out.push(Intent::Edit(TextOp::Backspace));
    }
    match m.edit() {
        Some(EditKey::DelWord) => out.push(Intent::Edit(TextOp::DelWord)),
        Some(EditKey::Home) => out.push(Intent::Edit(TextOp::Home)),
        Some(EditKey::End) => out.push(Intent::Edit(TextOp::End)),
        _ => {}
    }

    out
}
