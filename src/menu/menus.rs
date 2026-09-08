//! The concrete screens as [`Menu`] impls. Each is a small state with a pure
//! `view` and an `update` that returns a [`Command`]; the App owns the effects.
use crate::menu::{
    apply_text_op, parse_port, AppEffect, Command, Ctx, Framed, HostInfo, JoinInfo, Menu, Msg,
    Notice, Row, Style, ValueView, View, PORT_ERROR,
};
use crate::mods::{annotate_setting, VisualMask};
use crate::net::{DEFAULT_PORT, MAX_NAME};
use crate::render_config::VisualGroup;
use crate::session::Session;
use crate::settings::{Category, MenuKind, SETTINGS};
use crate::ui::EditBuf;

fn visual_mask_from_ctx(ctx: &Ctx) -> VisualMask {
    let mut mask = VisualMask {
        atmosphere: false,
        post: false,
        lighting: false,
    };
    for row in ctx.mods {
        match row.visual_group {
            Some(VisualGroup::Atmosphere) => mask.atmosphere = row.enabled,
            Some(VisualGroup::Post) => mask.post = row.enabled,
            Some(VisualGroup::Lighting) => mask.lighting = row.enabled,
            None => {}
        }
    }
    mask
}

// Start menu.

/// The start menu: New World, one Load row per save, then Host/Join/Mods/
/// Settings/Quit.
pub struct MainMenu {
    /// A transient status line (a failed connect) shown under the title.
    pub notice: Option<String>,
}

#[derive(Clone, Copy)]
pub enum MainAction {
    NewWorld,
    Load(usize),
    Host,
    Join,
    Mods,
    Settings,
    Quit,
}

impl MainMenu {
    pub fn new() -> Self {
        Self { notice: None }
    }

    /// A start menu carrying a status line (e.g. "could not join: ...").
    pub fn with_notice(notice: Option<String>) -> Self {
        Self { notice }
    }
}

impl Default for MainMenu {
    fn default() -> Self {
        Self::new()
    }
}

/// Formats playtime as "Xh Ym" or "Ym".
fn fmt_playtime(secs: u64) -> String {
    let (h, m) = (secs / 3600, (secs % 3600) / 60);
    if h > 0 { format!("{h}h {m}m played") } else { format!("{m}m played") }
}

impl Menu for MainMenu {
    type Action = MainAction;

    fn view(&self, ctx: &Ctx) -> View<MainAction> {
        let mut rows = vec![Row::action("New World", MainAction::NewWorld)];
        for (i, slot) in ctx.saves.iter().enumerate() {
            let row = match &slot.meta {
                Ok(meta) => Row::action(format!("Load: {}", meta.name), MainAction::Load(i))
                    .detail(format!("{} · {} edits", fmt_playtime(meta.playtime_secs), meta.edit_count)),
                // Still selectable: the load ladder may rescue it from the backup.
                Err(_) => Row::action(format!("Load: {} (damaged)", slot.id), MainAction::Load(i))
                    .detail("unreadable — a backup may still load"),
            };
            rows.push(row);
        }
        rows.push(Row::action("Host Server", MainAction::Host));
        rows.push(Row::action("Join Server", MainAction::Join));
        rows.push(Row::action("Mods", MainAction::Mods));
        rows.push(Row::action("Settings", MainAction::Settings));
        rows.push(Row::action("Quit", MainAction::Quit));
        View {
            title: "PROJECT WATT CUBED".to_string(),
            style: Style::Title { subtitle: "an infinite voxel world of elements".to_string() },
            rows,
            default: None,
            hint: "Up/Down select   Enter choose".to_string(),
            notice: self.notice.clone().map(Notice::info),
        }
    }

    fn update(&mut self, msg: Msg<MainAction>, ctx: &mut Ctx) -> Command {
        // Any input dismisses the status line.
        self.notice = None;
        match msg {
            Msg::Pick(action) => match action {
                MainAction::NewWorld => Command::Effect(AppEffect::NewWorld),
                MainAction::Load(i) => match ctx.saves.get(i) {
                    Some(slot) => Command::Effect(AppEffect::Load(slot.id.as_str().to_string())),
                    None => Command::Stay,
                },
                MainAction::Host => Command::Push(Framed::boxed(HostMenu::new(ctx.session))),
                MainAction::Join => Command::Push(Framed::boxed(JoinMenu::new(ctx.session))),
                MainAction::Mods => Command::Push(Framed::boxed(ModsMenu)),
                MainAction::Settings => Command::Push(Framed::boxed(SettingsHub)),
                MainAction::Quit => Command::Effect(AppEffect::Quit),
            },
            // Back at the root is ignored by the stack.
            Msg::Back => Command::Pop,
            _ => Command::Stay,
        }
    }
}

// Mods menu.

/// One toggle row per installed mod, plus knobs for mods that have them.
pub struct ModsMenu;

#[derive(Clone, Copy)]
pub enum ModsAction {
    Toggle(usize),
    Knob { mod_index: usize, knob: usize },
}

impl Menu for ModsMenu {
    type Action = ModsAction;

    fn view(&self, ctx: &Ctx) -> View<ModsAction> {
        let mut rows = Vec::new();
        for (i, m) in ctx.mods.iter().enumerate() {
            rows.push(
                Row::value(m.name.clone(), ValueView::Toggle(m.enabled), ModsAction::Toggle(i))
                    .detail(m.description.clone()),
            );
            if m.enabled {
                for (k, (label, value)) in m.knobs.iter().enumerate() {
                    let mut row = Row::value(
                        format!("  {label}"),
                        ValueView::Choice(value.clone()),
                        ModsAction::Knob { mod_index: i, knob: k },
                    );
                    if m.worldgen {
                        row = row.detail("next new world");
                    }
                    rows.push(row);
                }
            }
        }
        View {
            title: "MODS".to_string(),
            style: Style::Panel,
            rows,
            default: None,
            hint: "Enter toggle   Left/Right tune   Esc back".to_string(),
            notice: None,
        }
    }

    fn update(&mut self, msg: Msg<ModsAction>, _ctx: &mut Ctx) -> Command {
        match msg {
            Msg::Step(ModsAction::Toggle(i), _) | Msg::Pick(ModsAction::Toggle(i)) => {
                Command::Effect(AppEffect::ToggleMod(i))
            }
            Msg::Step(ModsAction::Knob { mod_index, knob }, dir) => {
                Command::Effect(AppEffect::StepModKnob {
                    mod_index,
                    knob,
                    delta: dir.delta(),
                })
            }
            Msg::Pick(ModsAction::Knob { .. }) => Command::Stay,
            Msg::Back => Command::Pop,
            _ => Command::Stay,
        }
    }
}

// Host / Join forms.

#[derive(Clone, Copy)]
pub enum ConnectionAction {
    Address,
    Port,
    Password,
    Name,
    Submit,
}

/// Shared host/join form; the mode supplies only the extra address row and the
/// final effect while editing, validation, and common fields stay identical.
pub struct ConnectionMenu<const JOIN: bool> {
    address: EditBuf,
    port: EditBuf,
    password: EditBuf,
    name: EditBuf,
    error: Option<String>,
}

pub type HostAction = ConnectionAction;
pub type JoinAction = ConnectionAction;
pub type HostMenu = ConnectionMenu<false>;
pub type JoinMenu = ConnectionMenu<true>;

impl<const JOIN: bool> ConnectionMenu<JOIN> {
    pub fn new(session: &Session) -> Self {
        Self {
            address: EditBuf::with(prefill(&session.address, "127.0.0.1"), 64),
            port: EditBuf::with(prefill(&session.port, &DEFAULT_PORT.to_string()), 5),
            password: EditBuf::new(64),
            name: EditBuf::with(prefill(&session.name, "player"), MAX_NAME),
            error: None,
        }
    }
}

impl<const JOIN: bool> Menu for ConnectionMenu<JOIN> {
    type Action = ConnectionAction;

    fn view(&self, _ctx: &Ctx) -> View<ConnectionAction> {
        let (title, password, submit, hint) = if JOIN {
            ("JOIN SERVER", "Password", "Connect", "type to edit   Enter connect   Esc back")
        } else {
            ("HOST SERVER", "Password (optional)", "Start",
             "type to edit   Enter start   Esc back")
        };
        let mut rows = Vec::new();
        if JOIN {
            rows.push(text_row("Address", &self.address, false, ConnectionAction::Address));
        }
        rows.extend([
            text_row("Port", &self.port, false, ConnectionAction::Port),
            text_row(password, &self.password, true, ConnectionAction::Password),
            text_row("Your name", &self.name, false, ConnectionAction::Name),
            Row::action(submit, ConnectionAction::Submit),
        ]);
        View {
            title: title.to_string(),
            style: Style::Panel,
            rows,
            default: Some(ConnectionAction::Submit),
            hint: hint.to_string(),
            notice: self.error.clone().map(Notice::error),
        }
    }

    fn update(&mut self, msg: Msg<ConnectionAction>, _ctx: &mut Ctx) -> Command {
        match msg {
            Msg::Edited(field, op) => {
                self.error = None;
                match field {
                    ConnectionAction::Address => apply_text_op(&mut self.address, op),
                    ConnectionAction::Port => apply_text_op(&mut self.port, op),
                    ConnectionAction::Password => apply_text_op(&mut self.password, op),
                    ConnectionAction::Name => apply_text_op(&mut self.name, op),
                    ConnectionAction::Submit => {}
                }
                Command::Stay
            }
            Msg::Pick(ConnectionAction::Submit) => match parse_port(self.port.text()) {
                Some(port) => {
                    let password = self.password.text().to_string();
                    let name = self.name.text().to_string();
                    let effect = if JOIN {
                        AppEffect::Join(JoinInfo {
                            host: self.address.text().trim().to_string(), port, password, name,
                        })
                    } else {
                        AppEffect::Host(HostInfo { port, password, name })
                    };
                    Command::Effect(effect)
                }
                None => {
                    self.error = Some(PORT_ERROR.to_string());
                    Command::Stay
                }
            },
            Msg::Back => Command::Pop,
            _ => Command::Stay,
        }
    }
}

// Settings: a hub that pushes one page per category.

/// The settings hub: one row per [`Category`], each pushing its page.
pub struct SettingsHub;

impl Menu for SettingsHub {
    type Action = Category;

    fn view(&self, _ctx: &Ctx) -> View<Category> {
        let rows = Category::ALL.iter().map(|(c, name)| Row::action(*name, *c)).collect();
        View {
            title: "SETTINGS".to_string(),
            style: Style::Panel,
            rows,
            default: None,
            hint: "Enter open   Esc back".to_string(),
            notice: None,
        }
    }

    fn update(&mut self, msg: Msg<Category>, _ctx: &mut Ctx) -> Command {
        match msg {
            Msg::Pick(cat) => Command::Push(Framed::boxed(SettingsPage::new(cat))),
            Msg::Back => Command::Pop,
            _ => Command::Stay,
        }
    }
}

/// Settings page for a category; field semantics stay table-side.
pub struct SettingsPage {
    category: Category,
}

impl SettingsPage {
    pub fn new(category: Category) -> Self {
        Self { category }
    }
}

impl Menu for SettingsPage {
    type Action = usize;

    fn view(&self, ctx: &Ctx) -> View<usize> {
        let title = Category::ALL
            .iter()
            .find(|(c, _)| *c == self.category)
            .map_or("SETTINGS", |(_, n)| n)
            .to_uppercase();
        let mask = visual_mask_from_ctx(ctx);
        let rows = SETTINGS
            .iter()
            .enumerate()
            .filter(|(_, s)| s.category() == self.category)
            .map(|(i, s)| {
                let stored = s.show(ctx.settings);
                let shown = annotate_setting(stored.clone(), s.key(), mask);
                let value = match s.menu_kind() {
                    MenuKind::Toggle if shown == stored => ValueView::Toggle(stored == "On"),
                    MenuKind::Toggle | MenuKind::Choice => ValueView::Choice(shown),
                    MenuKind::Bar => {
                        ValueView::Bar { t: s.fraction(ctx.settings), label: shown }
                    }
                };
                Row::value(s.label(), value, i)
            })
            .collect();
        View {
            title,
            style: Style::Panel,
            rows,
            default: None,
            hint: "Left/Right or h/l change   Esc back".to_string(),
            notice: None,
        }
    }

    fn update(&mut self, msg: Msg<usize>, ctx: &mut Ctx) -> Command {
        match msg {
            Msg::Step(i, dir) => {
                SETTINGS[i].step(ctx.settings, dir.delta());
                Command::Stay
            }
            Msg::Back => Command::Pop,
            _ => Command::Stay,
        }
    }
}

// A generic yes/no dialog (provided for callers that need a confirmation).

#[derive(Clone, Copy)]
pub enum Choice {
    Yes,
    No,
}

/// Confirmation dialog with a stored effect for Yes.
pub struct ConfirmDialog {
    message: String,
    on_yes: Option<AppEffect>,
}

impl ConfirmDialog {
    pub fn new(message: impl Into<String>, on_yes: AppEffect) -> Self {
        Self { message: message.into(), on_yes: Some(on_yes) }
    }
}

impl Menu for ConfirmDialog {
    type Action = Choice;

    fn view(&self, _ctx: &Ctx) -> View<Choice> {
        View {
            title: self.message.clone(),
            style: Style::Panel,
            rows: vec![Row::action("Yes", Choice::Yes), Row::action("No", Choice::No)],
            default: None,
            hint: "Enter choose   Esc cancel".to_string(),
            notice: None,
        }
    }

    fn update(&mut self, msg: Msg<Choice>, _ctx: &mut Ctx) -> Command {
        match msg {
            Msg::Pick(Choice::Yes) => match self.on_yes.take() {
                Some(effect) => Command::Effect(effect),
                None => Command::Pop,
            },
            Msg::Pick(Choice::No) | Msg::Back => Command::Pop,
            _ => Command::Stay,
        }
    }
}

// Helpers.

/// Returns remembered or falls back to default.
fn prefill<'a>(remembered: &'a str, default: &'a str) -> &'a str {
    if remembered.is_empty() {
        default
    } else {
        remembered
    }
}

/// Constructs a text row from an [`EditBuf`].
fn text_row<A: Copy>(label: &str, buf: &EditBuf, masked: bool, tag: A) -> Row<A> {
    Row::text(label, buf.text().to_string(), buf.caret_chars(), masked, tag)
}
