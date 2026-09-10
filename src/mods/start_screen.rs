//! Default start screen, as a disableable mod.
//!
//! Builds the screens the player sees today — main menu, the worlds page (one
//! row per saved world), host form, join form — as [`MenuModel`]s from
//! [`StartFacts`] and returns [`StartAction`]s.
//! Settings and Mods stay core so they exist even if every mod is off. Disable
//! this mod and the core fallback (New world / Load / Settings / Mods / Quit)
//! takes over.

use crate::menu::start::{HostInfo, JoinInfo, MenuModel, StartAction, StartFacts, StartScreen};
use crate::menu::{
    apply_text_op, drive, parse_port, AppEffect, Command, Ctx, Cursor, Framed, Intent, Menu, Msg,
    Notice, Row, Screen, Style, View, PORT_ERROR,
};
use crate::mods::Mod;
use crate::net::{DEFAULT_PORT, MAX_NAME};
use crate::session::Session;
use crate::settings::Settings;
use crate::ui::EditBuf;

/// The default start screen, as a disableable, replaceable mod.
pub struct StartScreenMod;

impl StartScreenMod {
    pub fn new() -> Self {
        Self
    }
}

impl Default for StartScreenMod {
    fn default() -> Self {
        Self::new()
    }
}

impl Mod for StartScreenMod {
    fn name(&self) -> &str {
        "Start"
    }

    fn id(&self) -> &'static str {
        "start"
    }

    fn description(&self) -> &str {
        "The default start screen (main menu, load list, host and join forms)."
    }

    fn group(&self) -> &'static str {
        crate::mods::ESSENTIALS
    }

    fn start_screen(&self, facts: &StartFacts) -> Option<Box<dyn StartScreen>> {
        Some(Box::new(DefaultStart::open(facts)))
    }
}

/// The screens the player sees today: main, worlds (the saved-world list), host, join.
struct DefaultStart {
    main: MainMenu,
    cursor: Cursor,
    overlay: Option<Overlay>,
}

enum Overlay {
    Worlds(Framed<WorldsMenu>),
    Host(Framed<HostMenu>),
    Join(Framed<JoinMenu>),
}

impl DefaultStart {
    fn open(facts: &StartFacts) -> Self {
        Self {
            main: MainMenu::with_notice(facts.notice.map(str::to_string)),
            cursor: Cursor::default(),
            overlay: None,
        }
    }

    fn dummy_ctx<'a>(facts: &'a StartFacts, settings: &'a mut Settings) -> Ctx<'a> {
        Ctx {
            settings,
            saves: facts.saves,
            mods: &[],
            session: facts.session,
            mods_save_error: None,
        }
    }
}

impl StartScreen for DefaultStart {
    fn view(&self, facts: &StartFacts) -> MenuModel {
        let mut settings = Settings::default();
        let ctx = Self::dummy_ctx(facts, &mut settings);
        match &self.overlay {
            Some(Overlay::Worlds(frame)) => {
                let (view, sel) = frame.view_sel(&ctx);
                MenuModel::from_view(view, sel, |a| match a {
                    WorldsAction::Load(i) => facts.saves.get(*i).map(|slot| StartAction::Load(slot.id.clone())),
                    WorldsAction::Back => None,
                })
            }
            Some(Overlay::Host(frame)) => {
                let (view, sel) = frame.view_sel(&ctx);
                MenuModel::from_view(view, sel, |_| None)
            }
            Some(Overlay::Join(frame)) => {
                let (view, sel) = frame.view_sel(&ctx);
                MenuModel::from_view(view, sel, |_| None)
            }
            None => {
                let view = self.main.view(facts);
                let selected = self.cursor.resolved(&view);
                MenuModel::from_view(view, selected, |a| start_action(*a, facts))
            }
        }
    }

    fn update(&mut self, intents: &[Intent], facts: &StartFacts) -> Option<StartAction> {
        let mut settings = Settings::default();
        let mut ctx = Self::dummy_ctx(facts, &mut settings);
        if let Some(overlay) = &mut self.overlay {
            let cmd = match overlay {
                Overlay::Worlds(frame) => frame.update(intents, &mut ctx),
                Overlay::Host(frame) => frame.update(intents, &mut ctx),
                Overlay::Join(frame) => frame.update(intents, &mut ctx),
            };
            return match cmd {
                Command::Pop => {
                    self.overlay = None;
                    None
                }
                Command::Effect(AppEffect::Load(id)) => Some(StartAction::Load(id)),
                Command::Effect(AppEffect::Host(info)) => Some(StartAction::Host(info)),
                Command::Effect(AppEffect::Join(info)) => Some(StartAction::Join(info)),
                _ => None,
            };
        }

        let view = self.main.view(facts);
        self.cursor.normalize(&view);
        match drive(intents, &view, &mut self.cursor) {
            Some(Msg::Pick(action)) => {
                self.main.notice = None;
                match action {
                    MainAction::Worlds => {
                        self.overlay = Some(Overlay::Worlds(Framed::new(WorldsMenu)));
                        None
                    }
                    MainAction::Host => {
                        self.overlay = Some(Overlay::Host(Framed::new(HostMenu::new(facts.session))));
                        None
                    }
                    MainAction::Join => {
                        self.overlay = Some(Overlay::Join(Framed::new(JoinMenu::new(facts.session))));
                        None
                    }
                    other => start_action(other, facts),
                }
            }
            Some(Msg::Back) => {
                self.main.notice = None;
                None
            }
            _ => None,
        }
    }
}

/// The start menu: New World, Worlds (the saved-world page), then Host/Join/
/// Mods/Settings/Quit. Saves are not listed inline so the menu stays short.
struct MainMenu {
    notice: Option<String>,
}

#[derive(Clone, Copy)]
enum MainAction {
    NewWorld,
    Worlds,
    Host,
    Join,
    Mods,
    Settings,
    Quit,
}

fn start_action(action: MainAction, _facts: &StartFacts) -> Option<StartAction> {
    match action {
        MainAction::NewWorld => Some(StartAction::NewWorld),
        MainAction::Worlds | MainAction::Host | MainAction::Join => None,
        MainAction::Mods => Some(StartAction::Mods),
        MainAction::Settings => Some(StartAction::Settings),
        MainAction::Quit => Some(StartAction::Quit),
    }
}

impl MainMenu {
    fn with_notice(notice: Option<String>) -> Self {
        Self { notice }
    }

    fn view(&self, facts: &StartFacts) -> View<MainAction> {
        let worlds_detail = match facts.saves.len() {
            0 => "none saved yet".to_string(),
            1 => "1 saved".to_string(),
            n => format!("{n} saved"),
        };
        let mut rows = vec![
            Row::action("New World", MainAction::NewWorld),
            Row::action("Worlds", MainAction::Worlds).detail(worlds_detail),
        ];
        rows.push(Row::action("Host Server", MainAction::Host));
        rows.push(Row::action("Join Server", MainAction::Join));
        rows.push(Row::action("Mods", MainAction::Mods));
        rows.push(Row::action("Settings", MainAction::Settings));
        rows.push(Row::action("Quit", MainAction::Quit));
        View {
            title: "PROJECT WATT CUBED".to_string(),
            style: Style::Title {
                subtitle: "an infinite voxel world of elements".to_string(),
            },
            rows,
            default: None,
            hint: "Up/Down select   Enter choose".to_string(),
            notice: self.notice.clone().map(Notice::info),
        }
    }
}

#[derive(Clone, Copy)]
enum WorldsAction {
    Load(usize),
    Back,
}

/// The worlds page: one row per saved world (newest first, as the save store
/// lists them), then Back. Enter loads; Esc goes back to the main menu.
struct WorldsMenu;

impl Menu for WorldsMenu {
    type Action = WorldsAction;

    fn view(&self, ctx: &Ctx) -> View<WorldsAction> {
        let mut rows: Vec<Row<WorldsAction>> = Vec::with_capacity(ctx.saves.len() + 1);
        if ctx.saves.is_empty() {
            rows.push(Row::heading("No saved worlds yet — pick New World on the main menu"));
        }
        for (i, slot) in ctx.saves.iter().enumerate() {
            let row = match &slot.meta {
                Ok(meta) => Row::action(format!("Load: {}", meta.name), WorldsAction::Load(i))
                    .detail(format!(
                        "{} · {} edits",
                        fmt_playtime(meta.playtime_secs),
                        meta.edit_count
                    )),
                Err(_) => Row::action(format!("Load: {} (damaged)", slot.id), WorldsAction::Load(i))
                    .detail("unreadable — a backup may still load"),
            };
            rows.push(row);
        }
        rows.push(Row::action("Back", WorldsAction::Back));
        View {
            title: "WORLDS".to_string(),
            style: Style::Panel,
            rows,
            default: None,
            hint: "Up/Down select   Enter load   Esc back".to_string(),
            notice: None,
        }
    }

    fn update(&mut self, msg: Msg<WorldsAction>, ctx: &mut Ctx) -> Command {
        match msg {
            Msg::Pick(WorldsAction::Load(i)) => match ctx.saves.get(i) {
                Some(slot) => Command::Effect(AppEffect::Load(slot.id.clone())),
                None => Command::Stay,
            },
            Msg::Pick(WorldsAction::Back) | Msg::Back => Command::Pop,
            _ => Command::Stay,
        }
    }
}

/// Formats playtime as "Xh Ym" or "Ym".
fn fmt_playtime(secs: u64) -> String {
    let (h, m) = (secs / 3600, (secs % 3600) / 60);
    if h > 0 {
        format!("{h}h {m}m played")
    } else {
        format!("{m}m played")
    }
}

#[derive(Clone, Copy)]
enum ConnectionAction {
    Address,
    Port,
    Password,
    Name,
    Submit,
}

/// Shared host/join form; the mode supplies only the extra address row and the
/// final effect while editing, validation, and common fields stay identical.
struct ConnectionMenu<const JOIN: bool> {
    address: EditBuf,
    port: EditBuf,
    password: EditBuf,
    name: EditBuf,
    error: Option<String>,
}

type HostMenu = ConnectionMenu<false>;
type JoinMenu = ConnectionMenu<true>;

impl<const JOIN: bool> ConnectionMenu<JOIN> {
    fn new(session: &Session) -> Self {
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
            (
                "JOIN SERVER",
                "Password",
                "Connect",
                "type to edit   Enter connect   Esc back",
            )
        } else {
            (
                "HOST SERVER",
                "Password (optional)",
                "Start",
                "type to edit   Enter start   Esc back",
            )
        };
        let mut rows = Vec::new();
        if JOIN {
            rows.push(text_row(
                "Address",
                &self.address,
                false,
                ConnectionAction::Address,
            ));
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
                            host: self.address.text().trim().to_string(),
                            port,
                            password,
                            name,
                        })
                    } else {
                        AppEffect::Host(HostInfo {
                            port,
                            password,
                            name,
                        })
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
    use crate::menu::start::fallback;
    use crate::menu::{Dir, TextOp};
    use crate::mods::Mods;
    use crate::save::{SaveError, Slot, SlotId};

    fn damaged(id: &str) -> Slot {
        Slot {
            id: SlotId::new(id).expect("legal slot id"),
            meta: Err(SaveError::Corrupt("truncated")),
        }
    }

    fn remembered(address: &str, port: &str, name: &str) -> Session {
        Session {
            address: address.into(),
            port: port.into(),
            name: name.into(),
        }
    }

    fn submit_form<const JOIN: bool>(session: &Session) -> Command {
        let mut menu = ConnectionMenu::<JOIN>::new(session);
        let mut settings = Settings::default();
        let mut ctx = ctx(&mut settings, session);
        menu.update(Msg::Pick(ConnectionAction::Submit), &mut ctx)
    }

    fn ctx<'a>(settings: &'a mut Settings, session: &'a Session) -> Ctx<'a> {
        Ctx {
            settings,
            saves: &[],
            mods: &[],
            session,
            mods_save_error: None,
        }
    }

    fn open(facts: &StartFacts) -> Box<dyn StartScreen> {
        StartScreenMod::new()
            .start_screen(facts)
            .expect("start mod always returns a screen")
    }

    #[test]
    fn start_mod_main_menu_is_short_and_lists_worlds_behind_one_row() {
        let session = Session::default();
        let saves = [Slot::for_test("alpha", 3661, 7), damaged("broken")];
        let f = StartFacts::test(&saves, &session, Some("could not join: refused"));
        let screen = open(&f);
        let model = screen.view(&f);
        assert_eq!(
            model.labels(),
            ["New World", "Worlds", "Host Server", "Join Server", "Mods", "Settings", "Quit"]
        );
        assert_eq!(model.rows[1].detail.as_deref(), Some("2 saved"));
        assert_eq!(
            model.actions(),
            [StartAction::NewWorld, StartAction::Mods, StartAction::Settings, StartAction::Quit]
        );
        assert_eq!(
            model.notice.as_ref().map(|n| n.text.as_str()),
            Some("could not join: refused")
        );
        assert!(matches!(model.style, Style::Title { .. }));
        assert_eq!(model.title, "PROJECT WATT CUBED");
        let one = [Slot::for_test("alpha", 0, 0)];
        let f1 = StartFacts::test(&one, &session, None);
        assert_eq!(open(&f1).view(&f1).rows[1].detail.as_deref(), Some("1 saved"));
        let none = StartFacts::test(&[], &session, None);
        assert_eq!(open(&none).view(&none).rows[1].detail.as_deref(), Some("none saved yet"));
    }

    #[test]
    fn worlds_page_lists_saves_loads_on_enter_and_backs_out() {
        let session = Session::default();
        let saves = [Slot::for_test("alpha", 3661, 7), damaged("broken")];
        let f = StartFacts::test(&saves, &session, None);
        let mut screen = open(&f);
        // New World -> Worlds, open the page.
        screen.update(&[Intent::Nav(Dir::Next)], &f);
        assert_eq!(screen.update(&[Intent::Confirm], &f), None);
        let page = screen.view(&f);
        assert_eq!(page.title, "WORLDS");
        assert_eq!(page.labels(), ["Load: alpha", "Load: broken (damaged)", "Back"]);
        assert_eq!(page.rows[0].detail.as_deref(), Some("1h 1m played · 7 edits"));
        assert_eq!(
            page.rows[1].detail.as_deref(),
            Some("unreadable — a backup may still load")
        );
        assert_eq!(
            page.actions(),
            [
                StartAction::Load(saves[0].id.clone()),
                StartAction::Load(saves[1].id.clone())
            ]
        );
        assert!(matches!(page.style, Style::Panel));
        // Esc returns to the main menu without an action.
        assert_eq!(screen.update(&[Intent::Cancel], &f), None);
        assert_eq!(screen.view(&f).title, "PROJECT WATT CUBED");
        // Back in, Enter on the first world loads it.
        assert_eq!(screen.update(&[Intent::Confirm], &f), None);
        assert_eq!(screen.view(&f).title, "WORLDS");
        assert_eq!(
            screen.update(&[Intent::Confirm], &f),
            Some(StartAction::Load(saves[0].id.clone()))
        );
        // The second world, and the Back row.
        let mut screen = open(&f);
        screen.update(&[Intent::Nav(Dir::Next)], &f);
        screen.update(&[Intent::Confirm], &f);
        screen.update(&[Intent::Nav(Dir::Next)], &f);
        assert_eq!(
            screen.update(&[Intent::Confirm], &f),
            Some(StartAction::Load(saves[1].id.clone()))
        );
        let mut screen = open(&f);
        screen.update(&[Intent::Nav(Dir::Next)], &f);
        screen.update(&[Intent::Confirm], &f);
        screen.update(&[Intent::Nav(Dir::Next), Intent::Nav(Dir::Next)], &f);
        assert_eq!(screen.update(&[Intent::Confirm], &f), None, "Back pops the page");
        assert_eq!(screen.view(&f).title, "PROJECT WATT CUBED");
    }

    #[test]
    fn worlds_page_without_saves_explains_and_backs_out() {
        let session = Session::default();
        let f = StartFacts::test(&[], &session, None);
        let mut screen = open(&f);
        screen.update(&[Intent::Nav(Dir::Next)], &f);
        assert_eq!(screen.update(&[Intent::Confirm], &f), None);
        let page = screen.view(&f);
        assert_eq!(page.title, "WORLDS");
        assert_eq!(
            page.labels(),
            ["No saved worlds yet — pick New World on the main menu", "Back"]
        );
        assert!(!page.rows[0].selectable);
        assert!(page.actions().is_empty());
        // The cursor lands on Back; Enter pops.
        assert_eq!(screen.update(&[Intent::Confirm], &f), None);
        assert_eq!(screen.view(&f).title, "PROJECT WATT CUBED");
    }

    #[test]
    fn start_mod_picks_emit_start_actions() {
        let session = Session::default();
        let saves = [Slot::for_test("alpha", 0, 0)];
        let f = StartFacts::test(&saves, &session, None);
        let mut screen = open(&f);
        assert_eq!(
            screen.update(&[Intent::Confirm], &f),
            Some(StartAction::NewWorld)
        );
        // With or without saves: New World, Worlds, Host, Join, Mods, Settings, Quit.
        let empty = StartFacts::test(&[], &session, None);
        for (steps, expected) in [(4, StartAction::Mods), (5, StartAction::Settings), (6, StartAction::Quit)] {
            let mut screen = open(&empty);
            for _ in 0..steps {
                screen.update(&[Intent::Nav(Dir::Next)], &empty);
            }
            assert_eq!(screen.update(&[Intent::Confirm], &empty), Some(expected));
        }
    }

    #[test]
    fn host_form_round_trips_session_into_host_info() {
        let session = remembered("10.0.0.2", "7777", "Ada");
        match submit_form::<false>(&session) {
            Command::Effect(AppEffect::Host(info)) => {
                assert_eq!(
                    info,
                    HostInfo {
                        port: 7777,
                        password: String::new(),
                        name: "Ada".into(),
                    }
                );
            }
            _ => panic!("expected Host effect"),
        }
    }

    #[test]
    fn join_form_round_trips_session_into_join_info() {
        let session = remembered("10.0.0.2", "7777", "Ada");
        match submit_form::<true>(&session) {
            Command::Effect(AppEffect::Join(info)) => {
                assert_eq!(
                    info,
                    JoinInfo {
                        host: "10.0.0.2".into(),
                        port: 7777,
                        password: String::new(),
                        name: "Ada".into(),
                    }
                );
            }
            _ => panic!("expected Join effect"),
        }
    }

    #[test]
    fn host_form_via_start_screen_uses_remembered_session() {
        let session = remembered("ignored-for-host", "6000", "Sam");
        let f = StartFacts::test(&[], &session, None);
        let mut screen = open(&f);
        // New World -> Worlds -> Host Server
        screen.update(&[Intent::Nav(Dir::Next), Intent::Nav(Dir::Next)], &f);
        assert_eq!(screen.update(&[Intent::Confirm], &f), None);
        assert_eq!(screen.view(&f).title, "HOST SERVER");
        assert_eq!(
            screen.update(&[Intent::Confirm], &f),
            Some(StartAction::Host(HostInfo {
                port: 6000,
                password: String::new(),
                name: "Sam".into(),
            }))
        );
    }

    #[test]
    fn join_form_via_start_screen_uses_remembered_session() {
        let session = remembered("8.8.8.8", "6000", "Sam");
        let f = StartFacts::test(&[], &session, None);
        let mut screen = open(&f);
        // New World -> Worlds -> Host -> Join
        for _ in 0..3 {
            screen.update(&[Intent::Nav(Dir::Next)], &f);
        }
        assert_eq!(screen.update(&[Intent::Confirm], &f), None);
        assert_eq!(screen.view(&f).title, "JOIN SERVER");
        assert_eq!(
            screen.update(&[Intent::Confirm], &f),
            Some(StartAction::Join(JoinInfo {
                host: "8.8.8.8".into(),
                port: 6000,
                password: String::new(),
                name: "Sam".into(),
            }))
        );
    }

    #[test]
    fn invalid_port_stays_on_the_form() {
        let session = Session::default();
        let mut menu = HostMenu::new(&session);
        let mut settings = Settings::default();
        let mut ctx = ctx(&mut settings, &session);
        // Prefill is "5555"; delete it and type "0".
        for _ in 0..4 {
            menu.update(Msg::Edited(ConnectionAction::Port, TextOp::Backspace), &mut ctx);
        }
        menu.update(Msg::Edited(ConnectionAction::Port, TextOp::Char('0')), &mut ctx);
        match menu.update(Msg::Pick(ConnectionAction::Submit), &mut ctx) {
            Command::Stay => {}
            _ => panic!("invalid port must Stay"),
        }
        let view = menu.view(&ctx);
        assert_eq!(
            view.notice.as_ref().map(|n| n.text.as_str()),
            Some(PORT_ERROR)
        );
    }

    #[test]
    fn empty_port_means_default_port() {
        let session = remembered("", "", "");
        let mut menu = HostMenu::new(&session);
        let mut settings = Settings::default();
        let mut ctx = ctx(&mut settings, &session);
        // Prefill of empty session.port is DEFAULT_PORT text; clear it.
        for _ in 0..5 {
            menu.update(Msg::Edited(ConnectionAction::Port, TextOp::Backspace), &mut ctx);
        }
        match menu.update(Msg::Pick(ConnectionAction::Submit), &mut ctx) {
            Command::Effect(AppEffect::Host(info)) => {
                assert_eq!(info.port, DEFAULT_PORT);
                assert_eq!(info.name, "player");
            }
            _ => panic!("expected Host with default port"),
        }
    }

    #[test]
    fn first_enabled_start_screen_wins_and_disabled_falls_back() {
        let session = Session::default();
        let f = StartFacts::test(&[], &session, None);
        let mods = Mods::with_defaults();
        let screen = mods.start_screen(&f).expect("default start mod is on");
        assert_eq!(screen.view(&f).labels()[0], "New World");

        let mut off = Mods::with_defaults();
        off.set_enabled("start", false);
        assert!(off.start_screen(&f).is_none());
        let fb = fallback(&f);
        assert_eq!(fb.view(&f).labels()[0], "New world");
    }
}
