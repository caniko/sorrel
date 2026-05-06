{
  description = "Sorrel — spike-sorting curation GUI";

  inputs = {
    rs-harbor.url = "git+https://codeberg.org/caniko/rs-harbor.git";

    nixpkgs.follows = "rs-harbor/nixpkgs";
    rust-overlay.follows = "rs-harbor/rust-overlay";
    crane.follows = "rs-harbor/crane";
    flake-utils.follows = "rs-harbor/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    rs-harbor,
    flake-utils,
    rust-overlay,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [(import rust-overlay)];
      };

      toolchain = rs-harbor.lib.mkToolchain {inherit pkgs;};
      inherit (toolchain) craneLib;
      cross = rs-harbor.lib.mkCross {inherit pkgs system;};
      cargoConfig = rs-harbor.lib.mkCargoConfig {inherit pkgs;};

      src = craneLib.cleanCargoSource ./.;

      # Native deps for the eframe + wgpu GUI on Linux. rusqlite is built with
      # the `bundled` feature, so libsqlite is compiled by cc — no system
      # sqlite needed in buildInputs.
      nativeBuildInputs = with pkgs; [
        pkg-config
      ];

      # Build-time linkable libs.
      buildInputs = with pkgs;
        [
          fontconfig
          freetype
        ]
        ++ lib.optionals stdenv.isLinux [
          # Wayland stack
          wayland
          libxkbcommon
          libdecor
          # X11 stack (winit fallback)
          libx11
          libxcursor
          libxi
          libxrandr
          libxcb
          # GPU
          libGL
          vulkan-loader
        ];

      # Libraries dlopen'd at runtime by wgpu/winit; rpath them into the
      # final binary so the package is self-contained on both Wayland and X11.
      runtimeLibs = with pkgs;
        lib.optionals stdenv.isLinux [
          vulkan-loader
          libGL
          # Wayland: winit dlopens libwayland-client + libxkbcommon, and
          # libdecor for client-side decorations on GNOME / KDE wayland.
          wayland
          libxkbcommon
          libdecor
          # X11 fallback
          libx11
          libxcursor
          libxi
          libxrandr
          libxcb
          fontconfig
          freetype
        ];

      commonArgs = {
        inherit src buildInputs nativeBuildInputs;
        strictDeps = true;
        pname = "sorrel";
        version = "0.1.0";
      };

      cargoArtifacts = craneLib.buildDepsOnly commonArgs;

      sorrel = craneLib.buildPackage (commonArgs
        // {
          inherit cargoArtifacts;
          cargoExtraArgs = "-p sorrel --locked";
          # Add an rpath so the binary finds vulkan/wayland/etc. without
          # leaking the dev shell's LD_LIBRARY_PATH.
          postFixup = pkgs.lib.optionalString pkgs.stdenv.isLinux ''
            patchelf --add-rpath "${pkgs.lib.makeLibraryPath runtimeLibs}" \
              "$out/bin/sorrel"
          '';

          meta = with pkgs.lib; {
            description = "Native, monomorphised spike-sorting curation GUI";
            homepage = "https://codeberg.org/caniko/sorrel";
            license = with licenses; [mit asl20];
            mainProgram = "sorrel";
            platforms = platforms.unix;
          };
        });
    in {
      packages = {
        default = sorrel;
        inherit sorrel;
        cargo-config = cargoConfig.configPath;
      };

      checks = {
        default = sorrel;

        clippy = craneLib.cargoClippy (commonArgs
          // {
            inherit cargoArtifacts;
            cargoClippyExtraArgs = "--workspace --all-targets -- --deny warnings";
          });

        fmt = craneLib.cargoFmt {inherit src;};
      };

      devShells = rs-harbor.lib.mkDevShells {
        inherit pkgs cross cargoConfig;
        inherit (toolchain) craneLib;
        checks = self.checks.${system};

        pkgConfigDeps = buildInputs;
        packages = with pkgs;
          [
            cargo-nextest
            rust-analyzer
          ]
          ++ buildInputs
          ++ nativeBuildInputs;

        extraEnv = {
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;
        };
      };

      apps.default = {
        type = "app";
        program = "${sorrel}/bin/sorrel";
      };
    });
}
