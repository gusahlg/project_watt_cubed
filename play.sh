#!/usr/bin/env bash
# The one command that always runs the CURRENT code.
#
# The flake takes ../voxel-engine as a path input, which Nix COPIES into the
# store when the lock file is written — so after any engine change, a plain
# `nix run` builds today's game against yesterday's engine snapshot and fails
# with phantom missing-API errors. Re-pinning first makes that impossible.
# (`cargo run --release` inside `nix develop` never has this problem: the dev
# shell builds against the live sibling directly.)
set -euo pipefail
cd "$(dirname "$0")"
nix flake update voxel-engine
exec nix run . "$@"
