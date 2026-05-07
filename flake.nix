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

      # Optional libhdf5 for the NWB / Kilosort 4 rez.mat backends.
      # `hdf5-metno-sys`'s build script wants headers and lib under one
      # prefix, but nixpkgs splits libhdf5 into separate outputs (`out`
      # has lib/, `dev` has include/). Symlink-join both into a single tree
      # so the build script's HDF5_DIR detection works.
      hdf5C = pkgs.symlinkJoin {
        name = "hdf5-merged-${pkgs.hdf5.version}";
        paths = [pkgs.hdf5 pkgs.hdf5.dev];
      };
      hdf5BuildInputs = [hdf5C];

      sorrelHdf5 = craneLib.buildPackage (commonArgs
        // {
          inherit cargoArtifacts;
          buildInputs = buildInputs ++ hdf5BuildInputs;
          cargoExtraArgs = "-p sorrel --locked --features hdf5";
          postFixup = pkgs.lib.optionalString pkgs.stdenv.isLinux ''
            patchelf --add-rpath "${pkgs.lib.makeLibraryPath runtimeLibs}" \
              "$out/bin/sorrel"
          '';
        });

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

      website = pkgs.stdenv.mkDerivation {
        pname = "sorrel-website";
        version = "0.1.0";
        src = pkgs.lib.fileset.toSource {
          root = ./.;
          fileset = pkgs.lib.fileset.maybeMissing ./website;
        };
        nativeBuildInputs = [pkgs.zola];
        phases = ["buildPhase" "installPhase"];
        buildPhase = ''
          cp -r --no-preserve=mode $src/website site
          cd site && zola build
        '';
        installPhase = ''
          cp -r public $out
        '';
      };

      docs = pkgs.stdenv.mkDerivation {
        pname = "sorrel-docs";
        version = "0.1.0";
        src = pkgs.lib.fileset.toSource {
          root = ./.;
          fileset = pkgs.lib.fileset.maybeMissing ./docs;
        };
        nativeBuildInputs = [pkgs.mdbook];
        buildPhase = ''
          mdbook build docs
        '';
        installPhase = ''
          cp -r docs/book $out
        '';
      };

      site = pkgs.runCommand "sorrel-site" {} ''
        mkdir -p $out
        cp -r ${website}/* $out/
        mkdir -p $out/docs
        cp -r ${docs}/* $out/docs/
      '';
    in {
      packages = {
        default = sorrel;
        inherit sorrel sorrelHdf5 website docs site;
        sorrel-hdf5 = sorrelHdf5;
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
            mdbook
            rust-analyzer
            zola
            # libhdf5 is pulled in unconditionally so `cargo check
            # --features hdf5` Just Works inside the dev shell. The default
            # `cargo build` doesn't reference it.
            hdf5
          ]
          ++ buildInputs
          ++ nativeBuildInputs;

        extraEnv = {
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;
          # Point hdf5-metno-sys at the symlink-joined hdf5 tree (headers
          # *and* libs under one prefix) so its build script's auto-detect
          # finds both `H5pubconf.h` and `libhdf5.so`.
          HDF5_DIR = "${hdf5C}";
        };

        extraShellHook = ''
          echo "Website: cd website && zola serve"
          echo "Documentation: cd docs && mdbook serve"
        '';
      };

      apps.default = {
        type = "app";
        program = "${sorrel}/bin/sorrel";
      };
    });
}
