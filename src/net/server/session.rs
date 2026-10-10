//! The live message loop of one connection, and the chat and clock handlers it dispatches to.
use super::*;

pub(super) fn client_loop(
    rt: &Runtime,
    recv: &mut quinn::RecvStream,
    frame: &mut Vec<u8>,
    kick: &Notify,
    shared: &Arc<Mutex<State>>,
    ctx: &Ctx,
    id: u32,
) {
    // Each kind has its own budget, so a swing flood cannot starve edits.
    // Mod channels and tool uses keep a second, tighter window of their own.
    let mut budgets = KindBudget::new();
    let mut channels = ChannelBudget::new();
    let mut tool_rate = RateWindow::new(TOOL_RATE_LIMIT);
    loop {
        // A kick (slow client) wakes this out of the blocking read so cleanup
        // runs; the read future is only ever dropped on that teardown path, so
        // no partial frame desyncs a live stream.
        let read = rt.block_on(async {
            tokio::select! {
                r = protocol::read_frame_async(recv, frame) => Some(r),
                _ = kick.notified() => None,
            }
        });
        match read {
            Some(Ok(())) => {}
            _ => break, // EOF, a malformed length, or a kick: the client is gone.
        }

        let Some(msg) = ClientMessage::decode(frame) else {
            continue;
        };
        // Cruise is a declared state, not a flood: it is applied even when every
        // other window is spent, and it does not consume a token.
        let now = Instant::now();
        match charge(&mut budgets, &msg, now) {
            Charge::Drop => continue,
            Charge::Answer => {
                match &msg {
                    ClientMessage::Edit { req, x, y, z, .. } => reject_edit(shared, id, *req, *x, *y, *z),
                    ClientMessage::Teleport { .. } => refuse_move(shared, id),
                    ClientMessage::SetTime { .. } => answer_time(shared, ctx, id),
                    _ => {}
                }
                continue;
            }
            Charge::Pass => {}
        }
        match msg {
            ClientMessage::Move { pos, yaw, pitch, frame, velocity, up, stance } => {
                on_move(shared, ctx, id, pos, yaw, pitch, frame, velocity, up, stance)
            }
            ClientMessage::Teleport { pos } => on_teleport(shared, ctx, id, pos),
            ClientMessage::Edit { req, x, y, z, expect, spec } => {
                on_edit(shared, ctx.hooks.as_ref(), &ctx.generator, id, req, x, y, z, expect, &spec)
            }
            ClientMessage::Chat { channel, text } => match op_secret(&text) {
                Some(secret) => on_op_login(shared, ctx, id, &secret),
                None => on_chat(shared, ctx.hooks.as_ref(), id, channel, &text),
            },
            ClientMessage::SetTime { day } => on_set_time(shared, ctx, id, day),
            ClientMessage::ModData { channel, seq, bytes } => {
                if !channels.allow(&channel, now) {
                    continue; // Over this channel's budget this second — drop silently.
                }
                on_mod_data(shared, id, channel, seq, bytes);
            }
            // Only players who can see the swinger, the same audience as voice.
            ClientMessage::Swing => relay_swing(shared, id),
            ClientMessage::Ping { nonce } => {
                let state = shared.lock_recover();
                if let Some(h) = state.players.get(&id) {
                    // Best-effort: a full queue drops the probe, and the
                    // client simply re-sends on its interval.
                    let _ = h.out.try_send(ServerMessage::Pong { nonce }.encode().into());
                }
            }
            ClientMessage::Hello { .. } => {} // Already authenticated; ignore repeats.
            ClientMessage::Cruise { speed } => on_cruise(shared, id, speed),
            ClientMessage::ToolUse { req, x, y, z, expect, tool_spec } => {
                if !tool_rate.allow(now) {
                    // Over the tool budget this second: refuse, so the client's swing resolves.
                    refuse_tool(shared.lock_recover(), shared, &ctx.generator, id, req, (x, y, z), &tool_spec);
                    continue;
                }
                on_tool_use(shared, &ctx.generator, id, req, x, y, z, expect, &tool_spec)
            }
        }
    }
}

/// A [`Verdict::Deny`] drops the broadcast and delivers `reason` only to the
/// sender (existing `Chat` frame, `from_id` 0). Hook bodies run outside the
/// [`State`] lock, same rule as [`on_edit`].
pub(super) fn on_chat(
    shared: &Arc<Mutex<State>>,
    hooks: Option<&Mutex<hooks::Table>>,
    id: u32,
    channel: u8,
    text: &str,
) {
    let text = clean_chat(text);
    if text.is_empty() {
        return;
    }
    let mut state = shared.lock_recover();
    let Some(sender) = state.players.get(&id) else { return };
    let from_name = sender.name.clone();
    let origin = sender.pos;
    let channel = if channel == chat::GLOBAL { chat::GLOBAL } else { chat::LOCAL };
    if let Some(hooks) = hooks {
        let facts = ChatFacts {
            player: id,
            name: from_name.clone(),
            channel,
            text: text.clone(),
        };
        let out = sender.ready.then(|| sender.out.clone());
        drop(state);
        let verdict = hooks.lock_recover().on_chat(&facts);
        if let Verdict::Deny { reason } = verdict {
            if let Some(out) = out {
                let _ = out.try_send(
                    ServerMessage::Chat {
                        from_id: 0,
                        from_name: Arc::from("server"),
                        channel,
                        text: reason,
                    }
                    .encode()
                    .into(),
                );
            }
            return;
        }
        state = shared.lock_recover();
        if !state.players.contains_key(&id) {
            return;
        }
    }
    println!("<{from_name}> {text}");
    let msg = ServerMessage::Chat { from_id: id, from_name, channel, text };
    let wake = broadcast(&mut state, &msg, |_, h| {
        channel == chat::GLOBAL || h.pos.distance(origin) <= chat::RADIUS
    });
    drop(state);
    drop(wake);
}

/// The secret of a `/op <secret>` chat line. Such a line is never relayed, logged, or shown to hooks.
pub(super) fn op_secret(text: &str) -> Option<Arc<str>> {
    let text = clean_chat(text);
    let rest = text.strip_prefix("/op")?;
    (rest.is_empty() || rest.starts_with(char::is_whitespace)).then(|| rest.trim().into())
}

/// An operator listed with a secret proves it. Either way only the sender hears the answer.
pub(super) fn on_op_login(shared: &Arc<Mutex<State>>, ctx: &Ctx, id: u32, secret: &str) {
    let mut sends = Vec::new();
    let wake = {
        let mut state = shared.lock_recover();
        let Some(h) = state.players.get_mut(&id) else { return };
        let proved = !secret.is_empty()
            && ctx.op_secrets.iter().any(|(name, want)| name.eq_ignore_ascii_case(&h.name) && want == secret);
        h.op |= proved;
        let (reply, log) = if proved {
            ("you are now an operator", "is now an operator")
        } else {
            ("operator secret refused", "sent a wrong operator secret")
        };
        println!("[op] {} (#{id}) {log}", h.name);
        tell(h, id, reply, &mut sends);
        queue(&state, sends)
    };
    drop(wake);
}

/// Anchors the shared clock so joiners inherit the CURRENT time. A non-finite
/// value is ignored rather than poisoning the shared time. Only an operator
/// may set it.
pub(super) fn on_set_time(shared: &Arc<Mutex<State>>, ctx: &Ctx, id: u32, day: f32) {
    if !day.is_finite() {
        return;
    }
    let day = day.rem_euclid(1.0);
    let mut sends = Vec::new();
    {
        let mut state = shared.lock_recover();
        let Some(h) = state.players.get(&id) else { return };
        if !is_operator(ctx, h) {
            tell(h, id, "only an operator can set the time", &mut sends);
            sends.push((id, ServerMessage::Time { day: state.day_now(ctx.day_secs), day_secs: ctx.day_secs }.encode().into()));
            let wake = queue(&state, sends);
            drop(state);
            drop(wake);
            return;
        }
        state.day = day;
        state.day_set = Instant::now();
        let wake = broadcast(&mut state, &ServerMessage::Time { day, day_secs: ctx.day_secs }, |_, _| true);
        drop(state);
        drop(wake);
    }
}

/// A refused time change reconciles the caller's optimistic local clock.
pub(super) fn answer_time(shared: &Arc<Mutex<State>>, ctx: &Ctx, id: u32) {
    let state = shared.lock_recover();
    if let Some(h) = state.players.get(&id) {
        let _ = h.out.try_send(ServerMessage::Time { day: state.day_now(ctx.day_secs), day_secs: ctx.day_secs }.encode().into());
    }
}

pub(super) fn clean_chat(raw: &str) -> Arc<str> {
    raw.chars().filter(|c| !c.is_control()).take(MAX_CHAT).collect::<String>().trim().into()
}
