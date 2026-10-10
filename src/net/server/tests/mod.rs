// Test setup (bind/connect/spawn) may unwrap: a panic here is a loud test
// failure, which is exactly what the deny on the PRODUCTION paths exists
// to prevent (a client thread silently poisoning the shared state).
#![allow(clippy::unwrap_used)]

mod support;

mod budget;
mod edit;
mod fanout;
mod join;
mod live;
mod movement;
mod save;
