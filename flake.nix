{
  description = "Sorrel — spike-sorting curation GUI";

  inputs = {
    harbor-rs.url = "git+https://github.com/caniko/harbor-rs.git?ref=trunk";

    simit = {
      url = "git+https://github.com/caniko/simit.git?ref=trunk";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.rust-overlay.follows = "rust-overlay";
      inputs.flake-utils.follows = "flake-utils";
      inputs.plinth.follows = "plinth";
    };

    # Pinned SDK used by harbor-rs's reproducible osxcross builder.
    harbor-macos-sdk-pin.url = "git+https://github.com/caniko/harbor-macos-sdk-pin.git";

    nix-appimage = {
      url = "github:ralismark/nix-appimage";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    nixpkgs.follows = "harbor-rs/nixpkgs";
    rust-overlay.follows = "harbor-rs/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
    treefmt-nix.follows = "harbor-rs/treefmt-nix";
    git-hooks.follows = "harbor-rs/git-hooks";
    plinth = {
      url = "git+https://github.com/caniko/plinth.git";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    harbor-rs,
    simit,
    harbor-macos-sdk-pin,
    nix-appimage,
    flake-utils,
    rust-overlay,
    treefmt-nix,
    git-hooks,
    ...
  }:
    {
      # This is the sole simit project configuration source.  Keep CI and
      # release policy next to the flake outputs it describes; simit.toml is
      # intentionally not used.
      simitConfig = {
        flake.mode = "custom";
        ci = {
          platform = "github";
          provider = "actions";
          runtime = "nix";
          runner = "ubuntu-24.04";
          workspace = true;
          workspace_strategy = "aggregate";
          with_audit = true;
          with_deny = true;
          # Public caches only; the private macOS SDK cache is a release concern.
          cache = {
            substituters = [
              "https://attic.candee.baby/canix"
              "https://cache.nixos.org"
            ];
            trusted_public_keys = [
              "canix:lPzPzKrmYqW5Rxa5r0uQWvCqD3S5nx0h2eCy7XD5JM8="
              "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
            ];
          };
          pages = {
            repo = "caniko/sorrel";
            site_output = ".#site";
            source_branch = "trunk";
          };
        };
        # Linux closures only: the darwin build depends on the private macOS
        # SDK and must never be published to the public canix cache.
        prebuild = {
          publish_attic = true;
          attic_app = ".#push-flake-inputs";
          system_runners.x86_64-linux = "ubuntu-24.04";
          nix_builds = [
            ".#packages.x86_64-linux.sorrel"
            ".#packages.x86_64-linux.sorrel-appimage"
          ];
        };
        release = {
          publish.enforcement = "activated-remote";
          smoke.command = "nix run .#release-smoke --";
          github = {
            repo = "caniko/sorrel";
            target_branch = "trunk";
          };
          attic = {
            cache = "canix";
            url = "https://attic.candee.baby";
            token_name = "sorrel";
            token_secret = "ATTIC_TOKEN";
          };
          artifacts = {
            runner = "ubuntu-24.04";
            version_attr = "sorrel";
            prebuild_binaries = true;
            substituters = [
              "https://attic.candee.baby/canix"
              "https://attic.candee.baby/harbor-macos-sdk"
              "https://cache.nixos.org"
            ];
            trusted_public_keys = [
              "canix:lPzPzKrmYqW5Rxa5r0uQWvCqD3S5nx0h2eCy7XD5JM8="
              "harbor-macos-sdk:MLRX9qZASKwDh48UWON67cvYxfEbqvjfIZQmGwt1v1E="
              "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
            ];
            checksum_globs = ["*.tar.gz" "*.zip" "*.AppImage" "*.whl"];
            build_commands = [
              "git archive --format=tar --prefix=\"sorrel-$VERSION/\" HEAD | gzip -n > \"release/sorrel-$VERSION.tar.gz\""
            ];
          };
        };
      };
    }
    // flake-utils.lib.eachSystem ["x86_64-linux" "aarch64-linux" "aarch64-darwin"] (system: let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [(import rust-overlay)];
      };

      toolchain = harbor-rs.lib.mkToolchain {
        inherit pkgs;
        toolchainProfile = "nightly";
      };
      inherit (toolchain) craneLib rustToolchain;
      cross = harbor-rs.lib.mkCross {
        inherit pkgs system;
        macosSdkStorePath = harbor-macos-sdk-pin.storePath;
        macosSdkOutputHash = harbor-macos-sdk-pin.outputHash;
        osxSdkVersion = harbor-macos-sdk-pin.sdkVersion;
      };
      cargoConfig = harbor-rs.lib.mkCargoConfig {inherit pkgs;};

      src = pkgs.lib.cleanSourceWith {
        src = ./.;
        filter = path: type:
          (craneLib.filterCargoSources path type)
          || pkgs.lib.hasSuffix ".wgsl" path
          # BIP-39 list embedded by sorrel-remote's pairing codes.
          || pkgs.lib.hasSuffix "sorrel-remote/wordlist.txt" path
          # SpikeInterface sidecar `sorrel auto` embeds with include_str!.
          || pkgs.lib.hasSuffix "scripts/sorrel_auto.py" path;
      };
      treefmtEval = treefmt-nix.lib.evalModule pkgs (import ./nix/treefmt.nix {rustfmtPackage = rustToolchain;});
      pre-commit-check = git-hooks.lib.${system}.run {
        src = ./.;
        hooks = import ./nix/pre-commit.nix {
          inherit pkgs;
          treefmtWrapper = treefmtEval.config.build.wrapper;
          inherit rustToolchain;
        };
      };

      deps = import ./nix/deps.nix {inherit pkgs;};

      rustPackages = import ./nix/rust-packages.nix {
        inherit pkgs craneLib src deps;
      };
      inherit (rustPackages) sorrel sorrelRelay commonArgs cargoArtifacts;
      sorrelHdf5 = rustPackages.sorrelHdf5 or null;

      sorrelVersion = (builtins.fromTOML (builtins.readFile ./Cargo.toml)).workspace.package.version;
      crossSorrelPackages = harbor-rs.lib.mkCrossPackages {
        inherit pkgs craneLib cross;
        pname = "sorrel";
        targets = ["aarch64-linux" "windows" "darwin-aarch64"];
        commonArgs = {
          inherit src;
          version = sorrelVersion;
          strictDeps = true;
          inherit (deps) nativeBuildInputs;
          cargoExtraArgs = "-p sorrel --locked";
        };
        targetArgs = {
          aarch64-linux = {
            buildInputs = deps.aarch64LinuxBuildInputs;
            inherit (deps) nativeBuildInputs;
            doCheck = false;
          };
          windows = {
            buildInputs = [];
            nativeBuildInputs = deps.nativeBuildInputs ++ [cross.mingwCC cross.mingwBinutils];
            # aws-lc-sys (via libp2p-quic) compiles jitterentropy, which needs
            # winpthreads' <sched.h>; harbor-rs only puts winpthreads on the
            # link path.
            CFLAGS_x86_64_pc_windows_gnu = "-I${cross.winpthreads}/include";
            doCheck = false;
          };
          darwin-aarch64 = {
            buildInputs = [];
            inherit (deps) nativeBuildInputs;
            doCheck = false;
          };
        };
      };

      # The `_native` extension for the Python wheels (nix/wheel.nix). sorrel-py
      # is its own Cargo workspace, hence the manifest path and lockfile.
      crossSorrelPyPackages = harbor-rs.lib.mkCrossPackages {
        inherit pkgs craneLib cross;
        pname = "sorrel-py";
        targets = ["native" "aarch64-linux" "windows" "darwin-aarch64"];
        commonArgs = {
          inherit src;
          version = sorrelVersion;
          strictDeps = true;
          cargoLock = ./crates/sorrel-py/Cargo.lock;
          cargoExtraArgs = "--manifest-path crates/sorrel-py/Cargo.toml --locked";
          CARGO_TARGET_DIR = "target";
          # crane installs `cargoLock` at the source root.
          preBuild = "cp Cargo.lock crates/sorrel-py/Cargo.lock";
          doCheck = false;
          # crane's default install runs `cargo metadata` at the root, which
          # is the other workspace. Copy the cdylib for whichever target.
          doNotPostBuildInstallCargoBinaries = true;
          installPhaseCommand = ''
            mkdir -p $out/lib
            find target -maxdepth 3 -path '*/release/*' \
              \( -name libsorrel.so -o -name sorrel.dll -o -name libsorrel.dylib \) \
              -exec cp {} $out/lib/ \;
            test -n "$(ls $out/lib)"
          '';
        };
        targetArgs = {
          # pyo3 reads the interpreter config on native builds; cross builds
          # use the abi3 defaults.
          native = {
            nativeBuildInputs = [pkgs.python3];
            PYO3_PYTHON = "${pkgs.python3}/bin/python3";
          };
          windows = {
            nativeBuildInputs = [cross.mingwCC cross.mingwBinutils];
            # pyo3's import-lib generator runs dlltool from the vendored
            # (read-only) crate dir, where dlltool writes its temp files.
            PYO3_MINGW_DLLTOOL = pkgs.writeShellScript "dlltool-in-tmp" ''
              cd "''${TMPDIR:-/tmp}" && exec x86_64-w64-mingw32-dlltool "$@"
            '';
          };
        };
      };

      # glibc-2.31 binaries for the Linux tarballs and the AppImage.
      portable = import ./nix/portable.nix {
        inherit pkgs craneLib src;
        version = sorrelVersion;
        appimageRuntime = nix-appimage.packages.${system}.appimage-runtimes.appimage-type2-runtime;
      };
      sorrelAppImage = portable.sorrel-appimage;

      releaseSmoke = pkgs.writeShellApplication {
        name = "sorrel-release-smoke";
        runtimeInputs = with pkgs; [coreutils file findutils gnugrep gzip gnutar unzip];
        text = ''
          exec bash ${./release/smoke.sh} "$@"
        '';
      };

      siteOutputs = import ./nix/site.nix {
        inherit pkgs;
        root = ./.;
      };
      inherit (siteOutputs) website docs site;
    in {
      packages =
        {
          default = sorrel;
          inherit sorrel website docs site;
          sorrel-relay = sorrelRelay;
          cargo-config = cargoConfig.configPath;
          release-smoke = releaseSmoke;
        }
        // crossSorrelPackages
        // crossSorrelPyPackages
        // pkgs.lib.optionalAttrs (sorrelHdf5 != null) {inherit sorrelHdf5;}
        // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux portable
        // pkgs.lib.optionalAttrs (system == "x86_64-linux") {
          release-bundle = import ./nix/release.nix {
            inherit pkgs harbor-rs;
            version = sorrelVersion;
            packages = self.packages.${system};
          };
        };

      checks =
        (import ./nix/checks.nix {
          inherit craneLib commonArgs cargoArtifacts sorrel;
        })
        // {
          formatting = treefmtEval.config.build.check self;
        };

      devShells = import ./nix/devshell.nix {
        inherit pkgs harbor-rs craneLib cross cargoConfig deps;
        simit = simit.packages.${system}.default;
        checks = self.checks.${system};
        preCommitEnabledPackages = pre-commit-check.enabledPackages;
        inherit (pre-commit-check) shellHook;
      };

      formatter = treefmtEval.config.build.wrapper;

      apps =
        {
          default = {
            type = "app";
            program = "${sorrel}/bin/sorrel";
          };
          release-smoke = {
            type = "app";
            program = "${releaseSmoke}/bin/sorrel-release-smoke";
          };
        }
        // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
          # Simit prebuild publication on trunk (secret ATTIC_TOKEN). Linux
          # outputs only: darwin builds use the private macOS SDK, which must
          # never reach the public canix cache.
          push-flake-inputs = harbor-rs.lib.mkAtticPush {
            inherit pkgs;
            flake = ".";
            paths = [sorrel sorrelAppImage];
            adapter = harbor-rs.lib.mkAdapter {
              attic = {
                endpoint = "https://attic.candee.baby";
                cache = "canix";
                tokenEnvVar = "ATTIC_TOKEN";
              };
            };
          };
        };
    });
}
