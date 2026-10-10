//! Helpers the network tests share.
use std::thread;
use std::time::{Duration, Instant};

/// Check `cond` every 10 ms until it holds or `within` has passed; true when it held. For
/// state other threads change (server cleanup, a writer draining) rather than a fixed sleep.
pub(crate) fn eventually(within: Duration, mut cond: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + within;
    loop {
        if cond() {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        thread::sleep(Duration::from_millis(10));
    }
}
