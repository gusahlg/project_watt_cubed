//! Default start screen, as a disableable mod.
//!
//! Builds the screens the player sees today — main menu, load list, host form,
//! join form — as [`MenuModel`]s from [`StartFacts`] and returns [`StartAction`]s.
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

/// The screens the player sees today: main (with inline load rows), host, join.
struct DefaultStart {
    main: MainMenu,
    cursor: Cursor,
    overlay: Option<Overlay>,
}

enum Overlay {
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
        }
    }
}

impl StartScreen for DefaultStart {
    fn view(&self, facts: &StartFacts) -> MenuModel {
        let mut settings = Settings::default();
        let ctx = Self::dummy_ctx(facts, &mut settings);
        match &self.overlay {
            Some(Overlay::Host(frame)) => {
                let (view, sel) = frame.view_sel(&ctx);
                MenuModel::from_view(&view, sel, |_| None)
            }
            Some(Overlay::Join(frame)) => {
                let (view, sel) = frame.view_sel(&ctx);
                MenuModel::from_view(&view, sel, |_| None)
            }
            None => {
                let view = self.main.view(facts);
                let selected = self.cursor.resolved(&view);
                MenuModel::from_view(&view, selected, |a| match a {
                    MainAction::NewWorld => Some(StartAction::NewWorld),
                    MainAction::Load(i) => facts
                        .saves
                        .get(*i)
                        .map(|slot| StartAction::Load(slot.id.clone())),
                    MainAction::Host | MainAction::Join => None,
                    MainAction::Mods => Some(StartAction::Mods),
                    MainAction::Settings => Some(StartAction::Settings),
                    MainAction::Quit => Some(StartAction::Quit),
                })
            }
        }
    }

    fn update(&mut self, intents: &[Intent], facts: &StartFacts) -> Option<StartAction> {
        let mut settings = Settings::default();
        let mut ctx = Self::dummy_ctx(facts, &mut settings);
        if let Some(overlay) = &mut self.overlay {
            let cmd = match overlay {
                Overlay::Host(frame) => frame.update(intents, &mut ctx),
                Overlay::Join(frame) => frame.update(intents, &mut ctx),
            };
            return match cmd {
                Command::Pop => {
                    self.overlay = None;
                    None
                }
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
                    MainAction::NewWorld => Some(StartAction::NewWorld),
                    MainAction::Load(i) => facts
                        .saves
                        .get(i)
                        .map(|slot| StartAction::Load(slot.id.clone())),
                    MainAction::Host => {
                        self.overlay = Some(Overlay::Host(Framed::new(HostMenu::new(facts.session))));
                        None
                    }
                    MainAction::Join => {
                        self.overlay = Some(Overlay::Join(Framed::new(JoinMenu::new(facts.session))));
                        None
                    }
                    MainAction::Mods => Some(StartAction::Mods),
                    MainAction::Settings => Some(StartAction::Settings),
                    MainAction::Quit => Some(StartAction::Quit),
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

/// The start menu: New World, one Load row per save, then Host/Join/Mods/
/// Settings/Quit.
struct MainMenu {
    notice: Option<String>,
}

#[derive(Clone, Copy)]
enum MainAction {
    NewWorld,
    Load(usize),
    Host,
    Join,
    Mods,
    Settings,
    Quit,
}

impl MainMenu {
    fn with_notice(notice: Option<String>) -> Self {
        Self { notice }
    }

    fn view(&self, facts: &StartFacts) -> View<MainAction> {
        let mut rows = vec![Row::action("New World", MainAction::NewWorld)];
        for (i, slot) in facts.saves.iter().enumerate() {
            let row = match &slot.meta {
                Ok(meta) => Row::action(format!("Load: {}", meta.name), MainAction::Load(i))
                    .detail(format!(
                        "{} · {} edits",
                        fmt_playtime(meta.playtime_secs),
                        meta.edit_count
                    )),
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
    use crate::menu::start::{fallback, VERSION};
    use crate::menu::{Dir, TextOp};
    use crate::mods::Mods;
    use crate::save::{SaveError, SaveMeta, Slot, SlotId};

    fn slot(name: &str, playtime_secs: u64, edit_count: u32) -> Slot {
        Slot {
            id: SlotId::new(name).expect("legal slot id"),
            meta: Ok(SaveMeta {
                name: name.to_string(),
                seed: 1,
                created: 0,
                last_played: 10,
                playtime_secs,
                edit_count,
            }),
        }
    }

    fn damaged(id: &str) -> Slot {
        Slot {
            id: SlotId::new(id).expect("legal slot id"),
            meta: Err(SaveError::Corrupt("truncated")),
        }
    }

    fn facts<'a>(saves: &'a [Slot], session: &'a Session, notice: Option<&'a str>) -> StartFacts<'a> {
        StartFacts {
            saves,
            session,
            version: VERSION,
            hosting: false,
            notice,
        }
    }

    fn ctx<'a>(settings: &'a mut Settings, session: &'a Session) -> Ctx<'a> {
        Ctx {
            settings,
            saves: &[],
            mods: &[],
            session,
        }
    }

    fn open(facts: &StartFacts) -> Box<dyn StartScreen> {
        StartScreenMod::new()
            .start_screen(facts)
            .expect("start mod always returns a screen")
    }

    #[test]
    fn start_mod_main_menu_rows_match_today() {
        let session = Session::default();
        let saves = [slot("alpha", 3661, 7), damaged("broken")];
        let f = facts(&saves, &session, Some("could not join: refused"));
        let screen = open(&f);
        let model = screen.view(&f);
        assert_eq!(
            model.labels(),
            [
                "New World",
                "Load: alpha",
                "Load: broken (damaged)",
                "Host Server",
                "Join Server",
                "Mods",
                "Settings",
                "Quit",
            ]
        );
        assert_eq!(
            model.rows[1].detail.as_deref(),
            Some("1h 1m played · 7 edits")
        );
        assert_eq!(
            model.rows[2].detail.as_deref(),
            Some("unreadable — a backup may still load")
        );
        assert_eq!(
            model.actions(),
            [
                StartAction::NewWorld,
                StartAction::Load(saves[0].id.clone()),
                StartAction::Load(saves[1].id.clone()),
                StartAction::Mods,
                StartAction::Settings,
                StartAction::Quit,
            ]
        );
        assert_eq!(
            model.notice.as_ref().map(|n| n.text.as_str()),
            Some("could not join: refused")
        );
        assert!(matches!(model.style, Style::Title { .. }));
        assert_eq!(model.title, "PROJECT WATT CUBED");
    }

    #[test]
    fn start_mod_picks_emit_start_actions() {
        let session = Session::default();
        let saves = [slot("alpha", 0, 0)];
        let f = facts(&saves, &session, None);
        let mut screen = open(&f);
        assert_eq!(
            screen.update(&[Intent::Confirm], &f),
            Some(StartAction::NewWorld)
        );
        let mut screen = open(&f);
        screen.update(&[Intent::Nav(Dir::Next)], &f);
        assert_eq!(
            screen.update(&[Intent::Confirm], &f),
            Some(StartAction::Load(saves[0].id.clone()))
        );
        // No saves: New World, Host, Join, Mods, Settings, Quit.
        let empty = facts(&[], &session, None);
        let mut screen = open(&empty);
        for _ in 0..4 {
            screen.update(&[Intent::Nav(Dir::Next)], &empty);
        }
        assert_eq!(
            screen.update(&[Intent::Confirm], &empty),
            Some(StartAction::Settings)
        );
        let mut screen = open(&empty);
        for _ in 0..3 {
            screen.update(&[Intent::Nav(Dir::Next)], &empty);
        }
        assert_eq!(
            screen.update(&[Intent::Confirm], &empty),
            Some(StartAction::Mods)
        );
        let mut screen = open(&empty);
        for _ in 0..5 {
            screen.update(&[Intent::Nav(Dir::Next)], &empty);
        }
        assert_eq!(
            screen.update(&[Intent::Confirm], &empty),
            Some(StartAction::Quit)
        );
    }

    #[test]
    fn host_form_round_trips_session_into_host_info() {
        let session = Session {
            address: "10.0.0.2".into(),
            port: "7777".into(),
            name: "Ada".into(),
        };
        let mut menu = HostMenu::new(&session);
        let mut settings = Settings::default();
        let mut ctx = ctx(&mut settings, &session);
        match menu.update(Msg::Pick(ConnectionAction::Submit), &mut ctx) {
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
        let session = Session {
            address: "10.0.0.2".into(),
            port: "7777".into(),
            name: "Ada".into(),
        };
        let mut menu = JoinMenu::new(&session);
        let mut settings = Settings::default();
        let mut ctx = ctx(&mut settings, &session);
        match menu.update(Msg::Pick(ConnectionAction::Submit), &mut ctx) {
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
        let session = Session {
            address: "ignored-for-host".into(),
            port: "6000".into(),
            name: "Sam".into(),
        };
        let f = facts(&[], &session, None);
        let mut screen = open(&f);
        // New World -> Host Server
        screen.update(&[Intent::Nav(Dir::Next)], &f);
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
        let session = Session {
            address: "8.8.8.8".into(),
            port: "6000".into(),
            name: "Sam".into(),
        };
        let f = facts(&[], &session, None);
        let mut screen = open(&f);
        // New World -> Host -> Join
        screen.update(&[Intent::Nav(Dir::Next)], &f);
        screen.update(&[Intent::Nav(Dir::Next)], &f);
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
        let session = Session {
            address: String::new(),
            port: String::new(),
            name: String::new(),
        };
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
        let f = facts(&[], &session, None);
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
