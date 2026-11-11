{
  description = "Sorrel — spike-sorting curation GUI";

  inputs = {
    harbor-rs.url = "git+https://github.com/caniko/harbor-rs.git?ref=trunk&rev=6055d7d2b3a05d7d91d50938c7fb6f862a5eec85";

    # Pinned SDK used by harbor-rs's reproducible osxcross builder.
    harbor-macos-sdk-pin.url = "git+https://github.com/caniko/harbor-macos-sdk-pin.git?ref=trunk";

    nix-appimage = {
      url = "github:ralismark/nix-appimage";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    nixpkgs.follows = "harbor-rs/nixpkgs";
    rust-overlay.follows = "harbor-rs/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
    # harbor-rs keeps these in its dev partition, out of its public inputs.
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    git-hooks = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # Static project-site generator for the landing page (built via crane below).
    plinth = {
      url = "git+https://github.com/caniko/plinth.git?ref=trunk&rev=29874c37500a813763bc2d49bcb9c3faaa77fb49";
      flake = false;
    };
  };

  outputs = {
    self,
    nixpkgs,
    harbor-rs,
    harbor-macos-sdk-pin,
    nix-appimage,
    flake-utils,
    rust-overlay,
    treefmt-nix,
    git-hooks,
    plinth,
    ...
  }:
    {
      homeManagerModules.default = import ./nix/hm-module.nix self;

      # This is the sole simit project configuration source.  Keep CI and
      # release policy next to the flake outputs it describes; simit.toml is
      # intentionally not used.
      simitConfig = let
        qualification = map (check: ".#checks.x86_64-linux.${check}") [
          "clippy"
          "default"
          "formatting"
          "hm-module"
          "workspace-tests"
          "sorrel-hdf5"
          "hdf5-tests"
          "binstall-metadata"
          "binstall-metadata-sorrel-remote"
        ];
      in {
        flake.mode = "custom";
        ci = {
          platform = "github";
          provider = "actions";
          runtime = "nix";
          runner = "ubuntu-24.04";
          workspace = true;
          workspace_strategy = "aggregate";
          # PRs qualify branches; push CI runs on trunk only. A branch push run
          # shares the PR run's concurrency group, gets cancelled, and shows up
          # as a failed check that blocks the merge.
          push_branches = ["trunk"];
          # Publish the workspace crates in dependency order through simit's
          # trusted publisher (simit-crates-io environment, restricted to trunk).
          publish_crates = true;
          publish_strategy = "coordinated";
          with_audit = true;
          with_deny = true;
          # Run the cross-built sorrel.exe and Windows wheel on real Windows:
          # wine's shader compiler is not Microsoft's, so DX12 only proves out here.
          target_smokes = [
            {
              id = "windows";
              nix_build = ".#packages.x86_64-linux.sorrel-windows-smoke";
              runner = "windows-2025";
              shell = "pwsh";
              run = "./release/smoke-windows.ps1";
              # Includes the full `sorrel auto` run: uv installs SpikeInterface cold.
              timeout_minutes = 60;
              env.RUST_LOG = "info";
              artifacts = {
                name = "windows-smoke";
                paths = ["smoke-out/**"];
              };
            }
            # No arm64 wheel is built, so the wheel step is skipped there.
            {
              id = "windows-arm64";
              nix_build = ".#packages.x86_64-linux.sorrel-windows-aarch64-smoke";
              runner = "windows-11-arm";
              shell = "pwsh";
              run = "./release/smoke-windows.ps1 -SkipWheel";
              timeout_minutes = 60;
              env.RUST_LOG = "info";
              artifacts = {
                name = "windows-arm64-smoke";
                paths = ["smoke-out/**"];
              };
            }
          ];
          # sorrel's #[cfg(windows)] paths and the save tests need a real
          # Windows run; the Nix tests above only cover Linux.
          native_tests = [
            {
              id = "windows";
              runner = "windows-2025";
              # One failing binary must not hide the rest.
              args = "--workspace --locked --no-fail-fast";
              # The UI snapshot tests render on lavapipe: fetch the Windows
              # build of the dev shell's pinned Mesa (and the Vulkan loader the
              # runner lacks) and fail instead of skipping without it.
              setup = ''
                lava="$RUNNER_TEMP\\lavapipe"
                pwsh -NoProfile -File release/fetch-lavapipe-windows.ps1 -Dir "$lava" -Register
                echo "SORREL_LAVAPIPE_ICD=$lava\\lvp_icd.x86_64.json" >> "$GITHUB_ENV"
                echo "$lava" >> "$GITHUB_PATH"
              '';
              env.SORREL_REQUIRE_GPU = "1";
              timeout_minutes = 90;
            }
          ];
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
          # Exact-installable qualification of every x86_64-linux flake check,
          # the lavapipe UI snapshot states included (checks.workspace-tests).
          nix_builds = qualification;
          nix_build = {
            check_systems = ["x86_64-linux"];
            groups.checks = qualification;
            capture_results = true;
            push_branches = ["trunk"];
            timeout_minutes = 120;
            # The simit that generated the workflows; vars.SIMIT_CI_ENGINE_REVISION overrides.
            engine_revision = "bb75df673d9397e31e1fe8eb9108934364458ae1";
          };
        };
        # Linux closures only: the darwin build depends on the private macOS
        # SDK and must never be published to the public canix cache.
        prebuild = {
          publish_attic = true;
          push_branches = ["trunk"];
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
          # Upload the bundle's wheels (`sorrel-ephys`) after the GitHub
          # release; the PyPI trusted publisher is registered for release.yml.
          pypi.trusted_publishing = true;
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
            # Pull-only credential for the private macOS SDK cache (darwin archive).
            netrc_secret = "HARBOR_MACOS_SDK_NETRC";
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
            # Standalone sorrel.exe and sorrelw.exe (the console-free launcher)
            # on the release page, checksummed and signed like the rest;
            # winget and Chocolatey keep using the zip.
            extract = [
              {
                archive = "sorrel-{version}-x86_64-windows.zip";
                member = "sorrel.exe";
                asset = "sorrel-{version}-x86_64-windows.exe";
              }
              {
                archive = "sorrel-{version}-x86_64-windows.zip";
                member = "sorrelw.exe";
                asset = "sorrel-{version}-x86_64-windows-gui.exe";
              }
              {
                archive = "sorrel-{version}-aarch64-windows.zip";
                member = "sorrel.exe";
                asset = "sorrel-{version}-aarch64-windows.exe";
              }
              {
                archive = "sorrel-{version}-aarch64-windows.zip";
                member = "sorrelw.exe";
                asset = "sorrel-{version}-aarch64-windows-gui.exe";
              }
            ];
            build_commands = [
              "git archive --format=tar --prefix=\"sorrel-$VERSION/\" HEAD | gzip -n > \"release/sorrel-$VERSION.tar.gz\""
            ];
          };
        };
        winget = {
          package_id = "Caniko.Sorrel";
          download_repo = "caniko/sorrel";
          zip_archive = "sorrel-{version}-x86_64-windows.zip";
          arm64_zip_archive = "sorrel-{version}-aarch64-windows.zip";
          publisher = "Caniko";
          license = "MIT OR Apache-2.0";
          license_url = "https://github.com/caniko/sorrel/blob/trunk/LICENSE-MIT";
          short_description = "Native GUI for curating spike-sorted electrophysiology data";
          # `sorrel auto` runs its SpikeInterface sidecar through uv.
          dependencies = ["astral-sh.uv"];
          # sorrelw.exe is the console-free launcher; without an alias winget
          # links it under its own name.
          files = [
            {
              path = "sorrel.exe";
              alias = "sorrel";
            }
            {path = "sorrelw.exe";}
          ];
        };
        chocolatey = {
          name = "sorrel";
          download_repo = "caniko/sorrel";
          description = "Native high-performance GUI for manual curation of spike-sorted electrophysiology data.";
          project_url = "https://github.com/caniko/sorrel";
          authors = "Can H. Tartanoglu";
          license_url = "https://github.com/caniko/sorrel/blob/trunk/LICENSE-MIT";
          archive_pattern = "sorrel-{version}-{arch}-windows.zip";
          # chocolatey isn't in nixpkgs yet, and the dev shell no longer ships simit.
          nix_tool = "github:caniko/nixpkgs/add-chocolatey-scoop#chocolatey git+https://github.com/caniko/simit.git?rev=0e0a1b3153649aa22148540bf87eff3d7dbf6bb4";
          api_key_secret = "CHOCOLATEY_API_KEY";
          dependencies = [{id = "uv";}];
          # Start-menu entry for the launcher (its exe embeds the icon) and a
          # GUI shim so `sorrelw` returns at once from a terminal.
          shortcuts = [
            {
              name = "Sorrel";
              target = "sorrelw.exe";
              icon = "sorrelw.exe,0";
            }
          ];
          gui_exes = ["sorrelw.exe"];
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

      srcFilter = path: type:
        (craneLib.filterCargoSources path type)
        || pkgs.lib.hasSuffix ".wgsl" path
        # Window icon (include_bytes!) and the Windows resources build.rs embeds.
        || pkgs.lib.hasInfix "crates/sorrel/assets" path
        # BIP-39 list embedded by sorrel-remote's pairing codes.
        || pkgs.lib.hasSuffix "sorrel-remote/wordlist.txt" path
        # SpikeInterface sidecar `sorrel auto` embeds with include_str!.
        || pkgs.lib.hasSuffix "crates/sorrel/src/sorrel_auto.py" path
        # `sorrel agent-guide`'s primer, also embedded with include_str!.
        || pkgs.lib.hasSuffix "crates/sorrel/src/agent_guide.md" path;
      src = pkgs.lib.cleanSourceWith {
        src = ./.;
        filter = srcFilter;
      };
      # The tests also read the UI snapshot PNGs and the committed phy fixture.
      testSrc = pkgs.lib.cleanSourceWith {
        src = ./.;
        filter = path: type:
          srcFilter path type
          || pkgs.lib.hasInfix "/tests/snapshots/" path
          || pkgs.lib.hasInfix "/release/fixtures/" path;
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
        targets = ["aarch64-linux" "windows" "windows-aarch64" "darwin-aarch64"];
        commonArgs = {
          inherit src;
          version = sorrelVersion;
          strictDeps = true;
          inherit (deps) nativeBuildInputs;
          # sorrel-relay rides along in every archive (cargo binstall
          # sorrel-remote); crane installs every executable in the build log.
          cargoExtraArgs = "-p sorrel -p sorrel-remote --locked";
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
            # This replaces harbor-rs's value, so keep its -L. The GNU ld
            # --no-gc-sections fix for linkme lives in crates/sorrel/build.rs.
            CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUSTFLAGS = "-L ${cross.winpthreads}/lib";
            doCheck = false;
          };
          # aarch64-pc-windows-gnullvm: harbor-rs supplies llvm-mingw (cc, ar,
          # windres) through depsBuildBuild and the CC_/RC_/linker env.
          windows-aarch64 = {
            buildInputs = [];
            inherit (deps) nativeBuildInputs;
            # aws-lc-sys (via libp2p-quic) compiles jitterentropy, which needs
            # winpthreads' <sched.h>; llvm-mingw's sysroot does not carry it, and nixpkgs'
            # aarch64 winpthreads does not build with clang (winver.h). The
            # headers are arch-neutral, so borrow the x64 package's.
            CFLAGS_aarch64_pc_windows_gnullvm = "-I${cross.winpthreads}/include";
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
        runtimeInputs = with pkgs; [coreutils diffutils file findutils gnugrep gzip gnutar unzip];
        text = ''
          exec bash ${./release/smoke.sh} "$@"
        '';
      };

      plinthProjectArgs = {
        pname = "sorrel-plinth-project";
        src = plinth;
        strictDeps = true;
        cargoExtraArgs = "--locked --package plinth-project --bin plinth-project";
        doCheck = false;
      };
      plinthProject = craneLib.buildPackage (plinthProjectArgs
        // {
          cargoArtifacts = craneLib.buildDepsOnly plinthProjectArgs;
        });

      siteOutputs = import ./nix/site.nix {
        inherit pkgs plinthProject;
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
          # What release/smoke-windows.ps1 runs on a windows runner: the release
          # binary and the Windows wheel, from the release derivations.
          sorrel-windows-smoke = let
            wheel = self.packages.${system}.release-bundle.wheels.windows;
          in
            pkgs.runCommand "sorrel-windows-smoke-${sorrelVersion}" {} ''
              mkdir -p $out
              cp ${self.packages.${system}.sorrel-windows}/bin/sorrel.exe $out/
              cp ${self.packages.${system}.sorrel-windows}/bin/sorrelw.exe $out/
              cp ${wheel}/${wheel.file} $out/
            '';
          sorrel-windows-aarch64-smoke = pkgs.runCommand "sorrel-windows-aarch64-smoke-${sorrelVersion}" {} ''
            mkdir -p $out
            cp ${self.packages.${system}.sorrel-windows-aarch64}/bin/sorrel.exe $out/
            cp ${self.packages.${system}.sorrel-windows-aarch64}/bin/sorrelw.exe $out/
          '';
        };

      checks =
        (import ./nix/checks.nix {
          inherit pkgs craneLib commonArgs cargoArtifacts sorrel sorrelHdf5 testSrc;
          inherit (deps) runtimeLibs hdf5C;
          hmModule = self.homeManagerModules.default;
        })
        // {
          formatting = treefmtEval.config.build.check self;
        }
        // pkgs.lib.optionalAttrs (system == "x86_64-linux") {
          # The crate's [package.metadata.binstall] must match the release
          # bundle's archives (bin-dir included), and every archive has to
          # ship every [[bin]] without required-features.
          binstall-metadata = harbor-rs.lib.mkBinstallCheck {
            inherit pkgs;
            bundle = self.packages.${system}.release-bundle;
            cargoToml = ./crates/sorrel/Cargo.toml;
          };
          # sorrel-remote's sorrel-relay ships in the same archives.
          binstall-metadata-sorrel-remote = harbor-rs.lib.mkBinstallCheck {
            inherit pkgs;
            bundle = self.packages.${system}.release-bundle;
            cargoToml = ./crates/sorrel-remote/Cargo.toml;
          };
        };

      devShells = import ./nix/devshell.nix {
        inherit pkgs harbor-rs craneLib cross cargoConfig deps plinthProject;
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
