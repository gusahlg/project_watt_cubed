#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run observed-behavior probes in an isolated copy; never patch the real sources.

Run inside the project's dev shell, or provide its native pkg-config dependencies.
The optional --stage-only flag prints the temporary game directory without running.
"""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

here = Path(__file__).resolve().parent
repo = here.parents[2]
root = Path(tempfile.mkdtemp(prefix="pwc-audit-"))
game = root / "game"
game.mkdir()
for name in ("src", "crates", "assets"):
    shutil.copytree(repo / name, game / name)
for name in ("Cargo.toml", "Cargo.lock"):
    shutil.copy2(repo / name, game / name)
(root / "voxel-engine").symlink_to(repo.parent / "voxel-engine", target_is_directory=True)
targets = {
    "server.rs": "src/net/server/mod.rs",
    "client.rs": "src/net/client.rs",
    "game_sync.rs": "src/game/sync.rs",
    "registry.rs": "src/block/registry.rs",
    "modding.rs": "src/modding/mod.rs",
    "world.rs": "src/world/mod.rs",
    "save_bridge.rs": "src/save/bridge.rs",
    "save_store.rs": "src/save/store.rs",
}
for source, target in targets.items():
    with (game / target).open("a") as file:
        file.write(f'\n#[cfg(test)]\n#[path = "{here / source}"]\nmod audit_probes;\n')
print(game, flush=True)
if "--stage-only" in sys.argv:
    sys.exit(0)
env = os.environ.copy()
env.setdefault("CARGO_TARGET_DIR", str(repo / "target"))
sys.exit(subprocess.call(["cargo", "test", "--offline", "--lib", "audit_probes::", "--", "--nocapture", "--test-threads=1"], cwd=game, env=env))
