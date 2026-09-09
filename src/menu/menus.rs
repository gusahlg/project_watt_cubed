//! The concrete screens as [`Menu`] impls. Each is a small state with a pure
//! `view` and an `update` that returns a [`Command`]; the App owns the effects.
use crate::menu::{
    apply_text_op, parse_port, AppEffect, Command, Ctx, Dir, Framed, HostInfo, JoinInfo, Menu, Msg,
    Notice, Row, Style, ValueView, View, PORT_ERROR,
};
use crate::mods::{annotate_setting, Mods, VisualMask};
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

/// Grouped toggle rows per installed mod, plus knobs for mods that have them.
pub struct ModsMenu;

/// Persistent mods-screen notice: saved now, applied on the next world (visual)
/// or the next new world (worldgen); a newly added mod still needs a rebuild.
/// Each line stays under 70 glyphs so it fits the 1280-wide menu at notice size.
const MODS_NOTICE: &str = "Saved immediately. Visual mods: next world. Worldgen: next new world.\nMods are compiled in: rebuild, then restart, for a new mod to appear.";

#[derive(Clone, Copy)]
pub enum ModsAction {
    Toggle(usize),
    Knob { mod_index: usize, knob: usize },
    SetGroup { id: &'static str, on: bool },
}

impl Menu for ModsMenu {
    type Action = ModsAction;

    fn view(&self, ctx: &Ctx) -> View<ModsAction> {
        let mut rows = Vec::new();
        let mut placed = vec![false; ctx.mods.len()];
        for g in Mods::GROUPS {
            let members: Vec<usize> = ctx
                .mods
                .iter()
                .enumerate()
                .filter(|(_, m)| m.group == g.id)
                .map(|(i, _)| i)
                .collect();
            if members.is_empty() {
                continue;
            }
            for &i in &members {
                placed[i] = true;
            }
            rows.push(Row::heading(g.name).detail(g.description));
            let all_on = members.iter().all(|&i| ctx.mods[i].enabled);
            rows.push(Row::value(
                "  Enable all / Disable all",
                ValueView::Toggle(all_on),
                ModsAction::SetGroup {
                    id: g.id,
                    on: !all_on,
                },
            ));
            for i in members {
                push_mod_rows(&mut rows, i, &ctx.mods[i]);
            }
        }
        let other: Vec<usize> = placed
            .iter()
            .enumerate()
            .filter(|(_, seen)| !**seen)
            .map(|(i, _)| i)
            .collect();
        if !other.is_empty() {
            rows.push(Row::heading("Other"));
            for i in other {
                push_mod_rows(&mut rows, i, &ctx.mods[i]);
            }
        }
        View {
            title: "MODS".to_string(),
            style: Style::Panel,
            rows,
            default: None,
            hint: "Enter toggle/cycle   Left/Right tune   Esc back".to_string(),
            notice: Some(Notice::info(MODS_NOTICE.to_string())),
        }
    }

    fn update(&mut self, msg: Msg<ModsAction>, _ctx: &mut Ctx) -> Command {
        match msg {
            Msg::Step(ModsAction::Toggle(i), _) | Msg::Pick(ModsAction::Toggle(i)) => {
                Command::Effect(AppEffect::ToggleMod(i))
            }
            Msg::Step(ModsAction::SetGroup { id, on }, _)
            | Msg::Pick(ModsAction::SetGroup { id, on }) => {
                Command::Effect(AppEffect::SetGroup { id, on })
            }
            Msg::Step(ModsAction::Knob { mod_index, knob }, dir) => {
                Command::Effect(AppEffect::StepModKnob {
                    mod_index,
                    knob,
                    delta: dir.delta(),
                })
            }
            Msg::Pick(ModsAction::Knob { mod_index, knob }) => {
                Command::Effect(AppEffect::StepModKnob {
                    mod_index,
                    knob,
                    delta: Dir::Next.delta(),
                })
            }
            Msg::Back => Command::Pop,
            _ => Command::Stay,
        }
    }
}

fn push_mod_rows(rows: &mut Vec<Row<ModsAction>>, i: usize, m: &crate::menu::ModRow) {
    rows.push(
        Row::value(
            format!("  {}", m.name),
            ValueView::Toggle(m.enabled),
            ModsAction::Toggle(i),
        )
        .detail(m.description.clone()),
    );
    if !m.enabled {
        return;
    }
    for (k, (label, value, hint)) in m.knobs.iter().enumerate() {
        let mut row = Row::value(
            format!("    {label}"),
            ValueView::Choice(value.clone()),
            ModsAction::Knob {
                mod_index: i,
                knob: k,
            },
        );
        let mut detail = hint.clone();
        if m.worldgen {
            detail = if detail.is_empty() {
                "next new world".to_string()
            } else {
                format!("{detail} · next new world")
            };
        }
        if !detail.is_empty() {
            row = row.detail(detail);
        }
        rows.push(row);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::Session;
    use crate::settings::Settings;

    fn ctx<'a>(settings: &'a mut Settings, session: &'a Session) -> Ctx<'a> {
        Ctx {
            settings,
            saves: &[],
            mods: &[],
            session,
        }
    }

    #[test]
    fn mods_menu_notice_says_when_changes_apply() {
        let mut settings = Settings::default();
        let session = Session::default();
        let ctx = ctx(&mut settings, &session);
        let view = ModsMenu.view(&ctx);
        let notice = view.notice.expect("mods screen has a persistent notice");
        assert_eq!(notice.level, crate::menu::Level::Info);
        assert_eq!(notice.text, MODS_NOTICE);
        let lines: Vec<_> = notice.text.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(
            lines.iter().all(|l| l.chars().count() <= 70),
            "notice lines must fit the 1280-wide menu at 18px glyphs: {lines:?}"
        );
    }

    #[test]
    fn enter_on_a_knob_cycles_forward_like_right() {
        let mut menu = ModsMenu;
        let mut settings = Settings::default();
        let session = Session::default();
        let mut ctx = ctx(&mut settings, &session);
        let pick = menu.update(
            Msg::Pick(ModsAction::Knob {
                mod_index: 3,
                knob: 1,
            }),
            &mut ctx,
        );
        let step = menu.update(
            Msg::Step(
                ModsAction::Knob {
                    mod_index: 3,
                    knob: 1,
                },
                Dir::Next,
            ),
            &mut ctx,
        );
        match (pick, step) {
            (
                Command::Effect(AppEffect::StepModKnob {
                    mod_index: p_m,
                    knob: p_k,
                    delta: p_d,
                }),
                Command::Effect(AppEffect::StepModKnob {
                    mod_index: s_m,
                    knob: s_k,
                    delta: s_d,
                }),
            ) => {
                assert_eq!((p_m, p_k, p_d), (s_m, s_k, s_d));
                assert_eq!(p_d, 1);
            }
            _ => panic!("expected StepModKnob from both Enter and Right"),
        }
    }

    #[test]
    fn settings_row_uses_the_same_forced_off_marker_as_gfx() {
        let mut settings = Settings::default();
        let session = Session::default();
        let mods = vec![crate::menu::ModRow {
            name: "Post".into(),
            description: String::new(),
            enabled: false,
            knobs: vec![],
            visual_group: Some(VisualGroup::Post),
            worldgen: false,
            group: String::new(),
        }];
        let ctx = Ctx {
            settings: &mut settings,
            saves: &[],
            mods: &mods,
            session: &session,
        };
        let view = SettingsPage::new(Category::Video).view(&ctx);
        let bloom = view
            .rows
            .iter()
            .find(|r| r.label == "Bloom")
            .expect("bloom row");
        let marker = crate::mods::forced_off_marker("Post");
        match &bloom.kind {
            crate::menu::RowKind::Value(ValueView::Choice(s)) => {
                assert!(s.contains(&marker), "settings value {s:?} must include {marker}");
            }
            _ => panic!("expected annotated choice for a stripped bloom row"),
        }
    }

    #[test]
    fn mods_menu_nests_essentials_under_group_header() {
        let installed = crate::mods::Mods::with_defaults();
        let snap = crate::menu::ModRow::snapshot(&installed);
        let mut settings = Settings::default();
        let session = Session::default();
        let ctx = Ctx {
            settings: &mut settings,
            saves: &[],
            mods: &snap,
            session: &session,
        };
        let view = ModsMenu.view(&ctx);
        assert!(matches!(view.rows[0].kind, crate::menu::RowKind::Heading));
        assert_eq!(view.rows[0].label, "Essentials");
        assert!(view.rows[0].tag.is_none(), "group header is not selectable");
        assert_eq!(view.rows[1].label.trim(), "Enable all / Disable all");
        assert!(
            view.rows[2].label.contains("Menus"),
            "first member is indented under the group: {:?}",
            view.rows[2].label
        );
        assert!(
            !view.rows.iter().any(|r| r.label == "Other"),
            "no ungrouped built-ins"
        );
        let names: Vec<&str> = view
            .rows
            .iter()
            .filter(|r| matches!(r.kind, crate::menu::RowKind::Value(ValueView::Toggle(_))))
            .filter(|r| r.label.contains("Menus")
                || r.label.contains("Inventory")
                || r.label.contains("Crafting")
                || r.label.contains("Atmosphere")
                || r.label.contains("Post")
                || r.label.contains("Lighting")
                || r.label.contains("InfiniteDiffusion"))
            .map(|r| r.label.trim())
            .collect();
        assert_eq!(
            names,
            [
                "Menus",
                "Inventory",
                "Crafting",
                "Atmosphere",
                "Post",
                "Lighting",
                "InfiniteDiffusion"
            ]
        );
    }

    #[test]
    fn mods_menu_lists_ungrouped_under_other() {
        let extra = crate::menu::ModRow {
            name: "Extra".into(),
            description: "future external".into(),
            enabled: true,
            knobs: vec![],
            visual_group: None,
            worldgen: false,
            group: String::new(),
        };
        let installed = crate::mods::Mods::with_defaults();
        let mut snap = crate::menu::ModRow::snapshot(&installed);
        snap.push(extra);
        let mut settings = Settings::default();
        let session = Session::default();
        let ctx = Ctx {
            settings: &mut settings,
            saves: &[],
            mods: &snap,
            session: &session,
        };
        let view = ModsMenu.view(&ctx);
        let other = view
            .rows
            .iter()
            .position(|r| r.label == "Other" && matches!(r.kind, crate::menu::RowKind::Heading))
            .expect("Other section");
        assert!(view.rows[other + 1].label.contains("Extra"));
    }

    #[test]
    fn group_toggle_row_emits_set_group() {
        let mut menu = ModsMenu;
        let mut settings = Settings::default();
        let session = Session::default();
        let mut ctx = ctx(&mut settings, &session);
        match menu.update(
            Msg::Pick(ModsAction::SetGroup {
                id: crate::mods::ESSENTIALS,
                on: false,
            }),
            &mut ctx,
        ) {
            Command::Effect(AppEffect::SetGroup { id, on }) => {
                assert_eq!(id, crate::mods::ESSENTIALS);
                assert!(!on);
            }
            _ => panic!("expected SetGroup"),
        }
    }
}
