{
  description = "project_watt_cubed — a voxel game on the voxel_engine Vulkan renderer";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    # The renderer, from its public repository. flake.lock records the exact
    # revision, so `nix develop` and the package work on any machine. Cargo
    # builds still use the sibling checkout at ../voxel-engine; run
    # `nix flake update voxel-engine` when the game starts needing a newer
    # renderer revision. (A raw `path:` input once copied the engine's ignored
    # target/ directory into the store, about 5 GiB.)
    voxel-engine = {
      url = "github:gusahlg/voxel-engine";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, voxel-engine }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Runtime libraries: the Vulkan loader (ash dlopens libvulkan.so.1) and
        # the windowing libs winit dlopens (Wayland when WAYLAND_DISPLAY is set,
        # X11 otherwise). No OpenGL, no cmake, no bindgen — the engine is pure
        # Rust over the Vulkan loader.
        runtimeLibs = with pkgs; [
          vulkan-loader
          libxkbcommon
          wayland
          libx11
          libxcursor
          libxrandr
          libxi
          # cpal talks to ALSA (including PipeWire's ALSA plugin). Kira's
          # realtime audio scheduling feature talks to the system D-Bus daemon.
          alsa-lib
          dbus
        ];

        libraryPath = pkgs.lib.makeLibraryPath runtimeLibs;

        # Shader compiler for voxel_engine's build.rs. The engine falls back to
        # its checked-in SPIR-V when slangc is missing, so this is best-effort.
        slang = pkgs.lib.optionals (pkgs ? shader-slang) [ pkgs.shader-slang ];

        # buildRustPackage needs the engine's source next to the game's, at the
        # same relative path Cargo.toml uses (../voxel-engine).
        combinedSrc = pkgs.runCommand "source" { } ''
          mkdir -p $out
          cp -r ${self} $out/game
          cp -r ${voxel-engine} $out/voxel-engine
        '';
      in
      {
        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "project_watt_cubed";
          # Keep the Nix package identity tied to Cargo instead of maintaining a
          # second version that had already drifted behind (0.1.0 vs 0.2.0).
          version =
            (builtins.fromTOML (builtins.readFile ./Cargo.toml)).package.version;

          src = combinedSrc;
          sourceRoot = "source/game";

          cargoLock.lockFile = ./Cargo.lock;

          # `CARGO_MANIFEST_DIR` points into the ephemeral Nix build tree, where
          # `.git` is intentionally absent. Bake the flake identities into the
          # benchmark binary; ordinary Cargo builds still probe their live
          # worktrees at report time.
          WATT_BUILD_GAME_REV = self.dirtyRev or (self.rev or "unknown");
          WATT_BUILD_GAME_DIRTY = if self ? dirtyRev then "1" else "0";
          WATT_BUILD_ENGINE_REV = voxel-engine.rev or "unknown";
          WATT_BUILD_ENGINE_DIRTY = if voxel-engine ? dirtyRev then "1" else "0";

          nativeBuildInputs = [ pkgs.pkg-config ] ++ slang;
          buildInputs = runtimeLibs;

          # Runtime lookup first checks this executable-relative data root, so
          # `nix run` works from any directory and never depends on the checkout.
          postInstall = ''
            mkdir -p "$out/share/project_watt_cubed"
            cp -R assets "$out/share/project_watt_cubed/assets"
            test -f "$out/share/project_watt_cubed/assets/sounds/catalog.toml"
          '';

          # Every installed binary is part of the package contract. Patch the
          # directory rather than a hand-maintained name list: Cargo discovers
          # `src/bin/*` automatically, and a newly added window/audio tool must
          # not ship with only the default glibc runpath. Do not hide missing
          # outputs or patchelf failures with `|| true`.
          postFixup = ''
            for binary in "$out"/bin/*; do
              test -f "$binary"
              test -x "$binary"
              patchelf --set-rpath "${libraryPath}" "$binary"
            done
          '';

          meta.mainProgram = "project_watt_cubed";
        };

        # `nix flake check` must compile the pure package, not merely evaluate
        # the dev shell. This is the check that catches a stale voxel-engine
        # input when the game starts using a newly merged renderer API.
        checks.package = self.packages.${system}.default;

        # A raw path input once pulled the engine's ignored `target/` directory
        # into the store, turning a ~1 MiB source tree into a 5 GiB snapshot.
        # Keep a generous ceiling so normal source growth is harmless while a
        # source-filter regression fails quickly and explains itself.
        checks.engine-source-budget = pkgs.runCommand "voxel-engine-source-budget" { } ''
          size=$(du -sb ${voxel-engine} | cut -f1)
          if test "$size" -gt 16777216; then
            echo "voxel-engine source is $size bytes; expected at most 16 MiB" >&2
            exit 1
          fi
          printf '%s\n' "$size" > "$out"
        '';

        checks.reuse = pkgs.runCommand "reuse-lint" {
          nativeBuildInputs = [ pkgs.reuse ];
        } ''
          cp -R ${self} source
          chmod -R u+w source
          cd source
          reuse lint
          touch "$out"
        '';

        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [
            pkgs.cargo-deny
            pkgs.pkg-config
            pkgs.reuse
          ];
          buildInputs = with pkgs; [ rustc cargo rustfmt clippy ]
            ++ runtimeLibs
            ++ slang
            ++ [ pkgs.vulkan-validation-layers pkgs.vulkan-tools ];

          # So `cargo run` can dlopen vulkan/wayland/x11 at runtime, and the
          # validation layer is discoverable in debug builds.
          LD_LIBRARY_PATH = libraryPath;
          VK_LAYER_PATH =
            "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";

          shellHook = ''
            echo "project_watt_cubed dev shell — run 'cargo run --release'"
          '';
        };
      });
}
