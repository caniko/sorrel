{
  description = "Sorrel — spike-sorting curation GUI";

  inputs = {
    harbor-rs.url = "git+https://github.com/caniko/harbor-rs.git?ref=trunk";

    # Pinned SDK used by harbor-rs's reproducible osxcross builder.
    harbor-macos-sdk-pin.url = "git+https://github.com/caniko/harbor-macos-sdk-pin.git?ref=trunk";

    nix-appimage = {
      url = "github:ralismark/nix-appimage";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    nixpkgs.follows = "harbor-rs/nixpkgs";
    rust-overlay.follows = "harbor-rs/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
    treefmt-nix.follows = "harbor-rs/treefmt-nix";
    git-hooks.follows = "harbor-rs/git-hooks";
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
              timeout_minutes = 30;
              env.RUST_LOG = "info";
              artifacts = {
                name = "windows-smoke";
                paths = ["smoke-out/**"];
              };
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
            engine_revision = "78b775803103ae960faf475ad46070d771fddc91";
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
            build_commands = [
              "git archive --format=tar --prefix=\"sorrel-$VERSION/\" HEAD | gzip -n > \"release/sorrel-$VERSION.tar.gz\""
            ];
          };
        };
        winget = {
          package_id = "Caniko.Sorrel";
          download_repo = "caniko/sorrel";
          zip_archive = "sorrel-{version}-x86_64-windows.zip";
          publisher = "Caniko";
          license = "MIT OR Apache-2.0";
          license_url = "https://github.com/caniko/sorrel/blob/trunk/LICENSE-MIT";
          short_description = "Native GUI for curating spike-sorted electrophysiology data";
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
          nix_tool = "github:caniko/nixpkgs/add-chocolatey-scoop#chocolatey git+https://github.com/caniko/simit.git?rev=42293ecd5b42df7feffd615881f34dd6c83fbd76";
          api_key_secret = "CHOCOLATEY_API_KEY";
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
        # BIP-39 list embedded by sorrel-remote's pairing codes.
        || pkgs.lib.hasSuffix "sorrel-remote/wordlist.txt" path
        # SpikeInterface sidecar `sorrel auto` embeds with include_str!.
        || pkgs.lib.hasSuffix "crates/sorrel/src/sorrel_auto.py" path;
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
          # What release/smoke-windows.ps1 runs on a windows runner: the release
          # binary and the Windows wheel, from the release derivations.
          sorrel-windows-smoke = let
            wheel = self.packages.${system}.release-bundle.wheels.windows;
          in
            pkgs.runCommand "sorrel-windows-smoke-${sorrelVersion}" {} ''
              mkdir -p $out
              cp ${self.packages.${system}.sorrel-windows}/bin/sorrel.exe $out/
              cp ${wheel}/${wheel.file} $out/
            '';
        };

      checks =
        (import ./nix/checks.nix {
          inherit pkgs craneLib commonArgs cargoArtifacts sorrel testSrc;
          inherit (deps) runtimeLibs;
          hmModule = self.homeManagerModules.default;
        })
        // {
          formatting = treefmtEval.config.build.check self;
        };

      devShells = import ./nix/devshell.nix {
        inherit pkgs harbor-rs craneLib cross cargoConfig deps;
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
