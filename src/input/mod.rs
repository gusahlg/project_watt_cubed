//! Input handling: the keyboard-driven [`movement`] controller and the
//! mouse-driven [`look`] controller. Both read raylib input and mutate the
//! [`Player`](crate::player::Player); nothing here draws.
pub mod look;
pub mod movement;
