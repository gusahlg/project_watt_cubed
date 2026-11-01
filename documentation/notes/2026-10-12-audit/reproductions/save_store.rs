// SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;

#[test]
fn audit_failed_write_after_backup_recovery_loses_both_old_copies() {
    let path = test_temp_path("audit-backup").with_extension("save");
    fs::write(&path, b"corrupt live").unwrap();
    fs::write(sibling(&path, ".bak"), b"only good previous copy").unwrap();
    let tmp = sibling(&path, ".tmp");
    let failed = write_rotating_with(&path, b"good recovered snapshot", true, |from, to| {
        if from == tmp { return Err(io::Error::other("injected final rename failure")); }
        fs::rename(from, to)
    });
    assert!(failed.is_err());
    assert_eq!(fs::read(&path).unwrap(), b"corrupt live");
    assert!(!sibling(&path, ".bak").exists());
    assert!(!tmp.exists());
    fs::remove_file(path).unwrap();
    eprintln!("BUG: failed save after backup recovery leaves only the corrupt live bytes");
}
