{
  description = "project_watt_cubed — a voxel game on the voxel_engine Vulkan renderer";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    # The renderer lives in a sibling checkout. Use the local Git repository,
    # not a raw `path:` snapshot: the latter copied ignored build products such
    # as voxel-engine/target into the Nix store (about 5 GiB on this machine).
    # This packages the committed experimental revision and records that exact
    # revision in flake.lock. It remains machine-specific until the engine audit
    # commits are published, at which point this should become a remote Git URL.
    voxel-engine = {
      url = "git+file:///home/gusahlg/repos/voxel-engine?ref=experimental";
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

          nativeBuildInputs = slang;

          # Every installed binary is part of the package contract. In
          # particular `golden` also creates a window, so leaving it with only
          # the default glibc runpath makes the shipped tool fail to open X11.
          # Do not hide missing outputs or patchelf failures with `|| true`.
          postFixup = ''
            for binary in project_watt_cubed watt_server golden golden_compare; do
              patchelf --set-rpath "${libraryPath}" "$out/bin/$binary"
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

        devShells.default = pkgs.mkShell {
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
