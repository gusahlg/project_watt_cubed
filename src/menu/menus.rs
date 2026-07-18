//! The concrete screens as [`Menu`] impls. Each is a small state with a pure
//! `view` and an `update` that returns a [`Command`]; the App owns the effects.
use crate::menu::{
    apply_text_op, parse_port, AppEffect, Command, Ctx, Framed, HostInfo, JoinInfo, Menu, Msg,
    Notice, Row, Style, ValueView, View, PORT_ERROR,
};
use crate::net::{DEFAULT_PORT, MAX_NAME};
use crate::session::Session;
use crate::settings::{Category, MenuKind, SETTINGS};
use crate::ui::EditBuf;

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

/// One toggle row per installed mod.
pub struct ModsMenu;

impl Menu for ModsMenu {
    type Action = usize;

    fn view(&self, ctx: &Ctx) -> View<usize> {
        let rows = ctx
            .mods
            .iter()
            .enumerate()
            .map(|(i, m)| {
                Row::value(m.name.clone(), ValueView::Toggle(m.enabled), i).detail(m.description.clone())
            })
            .collect();
        View {
            title: "MODS".to_string(),
            style: Style::Panel,
            rows,
            default: None,
            hint: "Enter/Space toggle   Esc back".to_string(),
            notice: None,
        }
    }

    fn update(&mut self, msg: Msg<usize>, _ctx: &mut Ctx) -> Command {
        match msg {
            Msg::Step(i, _) | Msg::Pick(i) => Command::Effect(AppEffect::ToggleMod(i)),
            Msg::Back => Command::Pop,
            _ => Command::Stay,
        }
    }
}

// Host / Join forms.

#[derive(Clone, Copy)]
pub enum HostAction {
    Port,
    Password,
    Name,
    Submit,
}

/// Host form state.
pub struct HostMenu {
    port: EditBuf,
    password: EditBuf,
    name: EditBuf,
    error: Option<String>,
}

impl HostMenu {
    pub fn new(session: &Session) -> Self {
        Self {
            port: EditBuf::with(prefill(&session.port, &DEFAULT_PORT.to_string()), 5),
            password: EditBuf::new(64),
            name: EditBuf::with(prefill(&session.name, "player"), MAX_NAME),
            error: None,
        }
    }
}

impl Menu for HostMenu {
    type Action = HostAction;

    fn view(&self, _ctx: &Ctx) -> View<HostAction> {
        let rows = vec![
            text_row("Port", &self.port, false, HostAction::Port),
            text_row("Password (optional)", &self.password, true, HostAction::Password),
            text_row("Your name", &self.name, false, HostAction::Name),
            Row::action("Start", HostAction::Submit),
        ];
        View {
            title: "HOST SERVER".to_string(),
            style: Style::Panel,
            rows,
            default: Some(HostAction::Submit),
            hint: "type to edit   Enter start   Esc back".to_string(),
            notice: self.error.clone().map(Notice::error),
        }
    }

    fn update(&mut self, msg: Msg<HostAction>, _ctx: &mut Ctx) -> Command {
        match msg {
            Msg::Edited(field, op) => {
                self.error = None;
                match field {
                    HostAction::Port => apply_text_op(&mut self.port, op),
                    HostAction::Password => apply_text_op(&mut self.password, op),
                    HostAction::Name => apply_text_op(&mut self.name, op),
                    HostAction::Submit => {}
                }
                Command::Stay
            }
            Msg::Pick(HostAction::Submit) => match parse_port(self.port.text()) {
                Some(port) => Command::Effect(AppEffect::Host(HostInfo {
                    port,
                    password: self.password.text().to_string(),
                    name: self.name.text().to_string(),
                })),
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

#[derive(Clone, Copy)]
pub enum JoinAction {
    Address,
    Port,
    Password,
    Name,
    Submit,
}

/// Join form state.
pub struct JoinMenu {
    address: EditBuf,
    port: EditBuf,
    password: EditBuf,
    name: EditBuf,
    error: Option<String>,
}

impl JoinMenu {
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

impl Menu for JoinMenu {
    type Action = JoinAction;

    fn view(&self, _ctx: &Ctx) -> View<JoinAction> {
        let rows = vec![
            text_row("Address", &self.address, false, JoinAction::Address),
            text_row("Port", &self.port, false, JoinAction::Port),
            text_row("Password", &self.password, true, JoinAction::Password),
            text_row("Your name", &self.name, false, JoinAction::Name),
            Row::action("Connect", JoinAction::Submit),
        ];
        View {
            title: "JOIN SERVER".to_string(),
            style: Style::Panel,
            rows,
            default: Some(JoinAction::Submit),
            hint: "type to edit   Enter connect   Esc back".to_string(),
            notice: self.error.clone().map(Notice::error),
        }
    }

    fn update(&mut self, msg: Msg<JoinAction>, _ctx: &mut Ctx) -> Command {
        match msg {
            Msg::Edited(field, op) => {
                self.error = None;
                match field {
                    JoinAction::Address => apply_text_op(&mut self.address, op),
                    JoinAction::Port => apply_text_op(&mut self.port, op),
                    JoinAction::Password => apply_text_op(&mut self.password, op),
                    JoinAction::Name => apply_text_op(&mut self.name, op),
                    JoinAction::Submit => {}
                }
                Command::Stay
            }
            Msg::Pick(JoinAction::Submit) => match parse_port(self.port.text()) {
                Some(port) => Command::Effect(AppEffect::Join(JoinInfo {
                    host: self.address.text().trim().to_string(),
                    port,
                    password: self.password.text().to_string(),
                    name: self.name.text().to_string(),
                })),
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
        let rows = SETTINGS
            .iter()
            .enumerate()
            .filter(|(_, s)| s.category() == self.category)
            .map(|(i, s)| {
                let value = match s.menu_kind() {
                    MenuKind::Toggle => ValueView::Toggle(s.show(ctx.settings) == "On"),
                    MenuKind::Choice => ValueView::Choice(s.show(ctx.settings)),
                    MenuKind::Bar => {
                        ValueView::Bar { t: s.fraction(ctx.settings), label: s.show(ctx.settings) }
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
