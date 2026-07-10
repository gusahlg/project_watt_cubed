{
  description = "project_watt_cubed — a voxel game on the voxel_engine Vulkan renderer";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    # The renderer lives in a sibling checkout. This MUST be an absolute path: a
    # relative `path:../voxel-engine` resolves against the flake's *store copy*
    # once `nix run` archives this git tree, landing at `…-source/../voxel-engine`
    # (nonexistent) — so it only ever worked from a dirty tree. An absolute path
    # resolves the same whether the tree is clean or dirty, and still picks up
    # local edits to the sibling. Machine-specific; switch to a git URL
    # (github:gusahlg/voxel-engine) if this flake is ever fetched from elsewhere.
    voxel-engine = {
      url = "path:/home/gusahlg/repos/voxel-engine";
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
          xorg.libX11
          xorg.libXcursor
          xorg.libXrandr
          xorg.libXi
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
          version = "0.1.0";

          src = combinedSrc;
          sourceRoot = "source/game";

          cargoLock.lockFile = ./Cargo.lock;

          nativeBuildInputs = slang;

          # Wrap the binary so it can dlopen the Vulkan loader and windowing
          # libs at runtime.
          postFixup = ''
            patchelf --set-rpath "${libraryPath}" $out/bin/project_watt_cubed || true
            patchelf --set-rpath "${libraryPath}" $out/bin/watt_server || true
          '';

          meta.mainProgram = "project_watt_cubed";
        };

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
