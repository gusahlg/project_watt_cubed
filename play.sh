#!/usr/bin/env bash
# The one command that always runs the CURRENT code.
#
# The flake pins the committed ../voxel-engine experimental revision, so after
# an engine commit a plain `nix run` can build today's game against yesterday's
# engine API and fail with phantom missing-method/field errors. Re-pinning first
# makes that impossible. Uncommitted engine edits remain a dev-shell concern.
# (`cargo run --release` inside `nix develop` never has this problem: the dev
# shell builds against the live sibling directly.)
set -euo pipefail
cd "$(dirname "$0")"
nix flake update voxel-engine
exec nix run . "$@"
