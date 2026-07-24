# Music library

This directory is intentionally independent from `sounds/catalog.toml`. Short
effects are decoded eagerly; music should be streamed and managed by a future
`MusicDirector` with its own volume bus, transitions, and per-track errors.

Put source music here (subdirectories are fine). Until the streaming director is
implemented, an empty library is valid and the game remains silent rather than
treating missing music as an initialization failure.

Project-owned music and its editable source are `CC-BY-SA-4.0`. Before adding a
track, record its author, source, licence, and cryptographic hash under
`assets/attribution/`, and add a matching `.license` sidecar beside binary audio.
Third-party music keeps its original approved free-content licence and notices;
do not add files whose provenance or redistribution rights are unknown.

`./play.sh` points `WATT_ASSET_DIR` at this live checkout, so untracked tracks
are visible during development even though Nix flake source snapshots exclude
untracked files. For a copied/non-Nix binary, install the whole `assets/` tree
beside your application data and set `WATT_ASSET_DIR` to that tree's root.

A future `playlist.toml` should own stable track IDs and metadata instead of
deriving gameplay identity from filenames. The intended shape is:

<!-- SPDX-SnippetBegin -->
<!-- SPDX-SnippetCopyrightText: 2026 Project Watt Cubed contributors -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

```toml
[[tracks]]
id = "explore_day"
file = "explore/day.ogg"
title = "Daylight"
looped = true
gain = 0.8
tags = ["explore", "day"]
```

<!-- SPDX-SnippetEnd -->

Do not add music files to the SFX catalog: that would decode long tracks fully
at startup and couple music lifetime to the effects voice allocator.
