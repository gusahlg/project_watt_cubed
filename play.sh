#!/usr/bin/env bash
# The one command that always runs the CURRENT code.
#
# The flake pins the committed ../voxel-engine main revision, so after
# an engine commit a plain `nix run` can build today's game against yesterday's
# engine API and fail with phantom missing-method/field errors. Re-pinning first
# makes that impossible. Uncommitted engine edits remain a dev-shell concern.
# (`cargo run --release` inside `nix develop` never has this problem: the dev
# shell builds against the live sibling directly.)
set -euo pipefail
cd "$(dirname "$0")"
# Flake sources intentionally exclude untracked files. Point development runs
# at the live asset tree so newly dropped-in music and replacement sounds are
# visible immediately; an explicit caller override still wins.
export WATT_ASSET_DIR="${WATT_ASSET_DIR:-$PWD/assets}"
# The mods screen writes `mods.toml` in the checkout, not the data dir.
export WATT_CHECKOUT_DIR="$PWD"
nix flake update voxel-engine
# The checkout builds vanilla PWC. With the PWC package manager installed and an instance
# selected (`pwc instance use <name>`), `PWC_PLAY=instance ./play.sh` runs that modded build.
if [ "${PWC_PLAY:-}" = instance ] && command -v pwc >/dev/null; then
  exec pwc run -- "$@"
fi
exec nix run . "$@"
