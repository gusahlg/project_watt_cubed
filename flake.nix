{
  description = "project_watt_cubed — a raylib + glam voxel project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Native libraries raylib needs to build (cmake/bindgen) and to run
        # (OpenGL + the X11 stack pulled in by GLFW).
        runtimeLibs = with pkgs; [
          libGL
          libx11
          libxcursor
          libxrandr
          libxinerama
          libxi
          libxext
          libxkbcommon
          wayland
        ];

        nativeBuildInputs = with pkgs; [
          cmake
          pkg-config
          # bindgen (used by raylib-sys) needs libclang at build time
          llvmPackages.libclang
        ];

        libraryPath = pkgs.lib.makeLibraryPath runtimeLibs;

        # bindgen (used by raylib-sys) drives libclang directly, which on NixOS
        # has no idea where glibc / gcc / clang headers live — so it can't find
        # <math.h> etc. Feed it the cc-wrapper's own cflags plus clang's builtin
        # header dir.
        bindgenClangArgs =
          (builtins.readFile "${pkgs.stdenv.cc}/nix-support/libc-cflags")
          + " " + (builtins.readFile "${pkgs.stdenv.cc}/nix-support/cc-cflags")
          + " -idirafter ${pkgs.llvmPackages.libclang.lib}/lib/clang/"
          + (pkgs.lib.versions.major pkgs.llvmPackages.libclang.version)
          + "/include";
      in
      {
        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "project_watt_cubed";
          version = "0.1.0";
          src = ./.;

          cargoLock.lockFile = ./Cargo.lock;

          inherit nativeBuildInputs;
          buildInputs = runtimeLibs;

          # bindgen needs to find libclang and the system headers
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          BINDGEN_EXTRA_CLANG_ARGS = bindgenClangArgs;

          # raylib-sys builds raylib via cmake; let it use the system toolchain.
          # Wrap the binary so it can find GL/X11 at runtime.
          postFixup = ''
            patchelf --set-rpath "${libraryPath}" $out/bin/project_watt_cubed || true
          '';

          meta.mainProgram = "project_watt_cubed";
        };

        devShells.default = pkgs.mkShell {
          inherit nativeBuildInputs;
          buildInputs = with pkgs; [ rustc cargo rustfmt clippy ] ++ runtimeLibs;

          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          BINDGEN_EXTRA_CLANG_ARGS = bindgenClangArgs;

          # So `cargo run` can dlopen libGL / X11 / wayland at runtime.
          LD_LIBRARY_PATH = libraryPath;

          shellHook = ''
            echo "project_watt_cubed dev shell — run 'cargo run'"
          '';
        };
      });
}
