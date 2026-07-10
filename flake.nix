{
  description = "Sorrel — spike-sorting curation GUI";

  inputs = {
    rs-harbor.url = "git+https://codeberg.org/caniko/rs-harbor.git?ref=trunk";

    simit = {
      url = "git+https://codeberg.org/caniko/simit?ref=refs/tags/0.17.6";
      inputs.rs-harbor.follows = "rs-harbor";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.rust-overlay.follows = "rust-overlay";
      inputs.flake-utils.follows = "flake-utils";
    };

    # Pinned SDK used by rs-harbor's reproducible osxcross builder.
    rs-harbor-macos-sdk-pin.url = "git+ssh://git@codeberg.org/caniko/rs-harbor-macos-sdk-pin.git";

    nix-appimage = {
      url = "github:ralismark/nix-appimage";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    nixpkgs.follows = "rs-harbor/nixpkgs";
    rust-overlay.follows = "rs-harbor/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    git-hooks = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    plinth = {
      url = "git+https://codeberg.org/caniko/plinth.git";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    rs-harbor,
    simit,
    rs-harbor-macos-sdk-pin,
    nix-appimage,
    flake-utils,
    rust-overlay,
    treefmt-nix,
    git-hooks,
    plinth,
    ...
  }:
    {
      # This is the sole simit project configuration source.  Keep CI and
      # release policy next to the flake outputs it describes; simit.toml is
      # intentionally not used.
      simitConfig = {
        flake.mode = "custom";
        ci = {
          runtime = "nix";
          # Every generated step invokes the Nix development environment.  It
          # therefore needs the trusted runner; keeping the mapping explicit
          # preserves simit's per-step routing contract without dispatching
          # `nix develop` to the bare atlas image.
          runner = "atlas-nix-trusted";
          packages = [
            "sorrel"
            "sorrel-cache"
            "sorrel-compute"
            "sorrel-data"
            "sorrel-gpu"
            "sorrel-io"
            "sorrel-render"
            "sorrel-ui"
          ];
          with_audit = true;
          with_deny = true;
          pages = {
            repo = "caniko/sorrel";
            site_output = "site";
            token_secret = "CODEBERG_TOKEN";
            source_branch = "trunk";
            deploy_app = "deploy-pages";
          };
          step_runners = {
            cargo-clippy = "atlas-nix-trusted";
            cargo-doc = "atlas-nix-trusted";
            cargo-fmt = "atlas-nix-trusted";
            cargo-package = "atlas-nix-trusted";
            cargo-test = "atlas-nix-trusted";
            nix-check = "atlas-nix-trusted";
            quality-tools = "atlas-nix-trusted";
          };
        };
        release = {
          publish.enforcement = "activated-remote";
          smoke.command = "nix run .#release-smoke --";
          codeberg = {
            repo = "caniko/sorrel";
            target_branch = "trunk";
            token_secret = "CODEBERG_TOKEN";
          };
          artifacts = {
            runner = "atlas-nix-trusted";
            version_attr = "sorrel";
            substituters = [
              "https://attic.candee.baby/canix"
              "https://cache.nixos.org"
            ];
            trusted_public_keys = [
              "canix:lPzPzKrmYqW5Rxa5r0uQWvCqD3S5nx0h2eCy7XD5JM8="
              "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
            ];
            checksum_globs = ["*.tar.gz" "*.zip" "*.AppImage"];
            build_commands = ["bash release/assemble.sh"];
          };
        };
      };
    }
    // flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [(import rust-overlay)];
      };

      toolchain = rs-harbor.lib.mkToolchain {inherit pkgs;};
      inherit (toolchain) craneLib rustToolchain;
      cross = rs-harbor.lib.mkCross {
        inherit pkgs system;
        macosSdkStorePath = rs-harbor-macos-sdk-pin.storePath;
        macosSdkOutputHash = rs-harbor-macos-sdk-pin.outputHash;
        osxSdkVersion = rs-harbor-macos-sdk-pin.sdkVersion;
      };
      cargoConfig = rs-harbor.lib.mkCargoConfig {inherit pkgs;};

      src = pkgs.lib.cleanSourceWith {
        src = ./.;
        filter = path: type:
          (craneLib.filterCargoSources path type)
          || pkgs.lib.hasSuffix ".wgsl" path;
      };
      treefmtEval = treefmt-nix.lib.evalModule pkgs (import ./nix/treefmt.nix);
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
      inherit (rustPackages) sorrel commonArgs cargoArtifacts;
      sorrelHdf5 = rustPackages.sorrelHdf5 or null;

      sorrelVersion = (builtins.fromTOML (builtins.readFile ./Cargo.toml)).workspace.package.version;
      crossSorrelPackages = rs-harbor.lib.mkCrossPackages {
        inherit pkgs craneLib cross;
        pname = "sorrel";
        targets = ["aarch64-linux" "windows" "darwin-aarch64"];
        commonArgs = {
          inherit src;
          version = sorrelVersion;
          strictDeps = true;
          nativeBuildInputs = deps.nativeBuildInputs;
          cargoExtraArgs = "-p sorrel --locked";
        };
        targetArgs = {
          aarch64-linux = {
            buildInputs = deps.aarch64LinuxBuildInputs;
            nativeBuildInputs = deps.nativeBuildInputs;
            doCheck = false;
          };
          windows = {
            buildInputs = [];
            nativeBuildInputs = deps.nativeBuildInputs ++ [cross.mingwCC cross.mingwBinutils];
            doCheck = false;
          };
          darwin-aarch64 = {
            buildInputs = [];
            nativeBuildInputs = deps.nativeBuildInputs;
            doCheck = false;
          };
        };
      };

      sorrelAppImage = rs-harbor.lib.mkAppImage {
        inherit system nix-appimage;
        pname = "sorrel";
        version = sorrelVersion;
        program = "${sorrel}/bin/sorrel";
      };

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
          cargo-config = cargoConfig.configPath;
          release-smoke = releaseSmoke;
        }
        // crossSorrelPackages
        // pkgs.lib.optionalAttrs (sorrelHdf5 != null) {inherit sorrelHdf5;}
        // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {sorrel-appimage = sorrelAppImage;};

      checks =
        (import ./nix/checks.nix {
          inherit craneLib src commonArgs cargoArtifacts sorrel;
        })
        // {
          formatting = treefmtEval.config.build.check self;
        };

      devShells = import ./nix/devshell.nix {
        inherit pkgs rs-harbor craneLib cross cargoConfig deps;
        simit = simit.packages.${system}.default;
        checks = self.checks.${system};
        preCommitEnabledPackages = pre-commit-check.enabledPackages;
        shellHook = pre-commit-check.shellHook;
      };

      formatter = treefmtEval.config.build.wrapper;

      apps = {
        default = {
          type = "app";
          program = "${sorrel}/bin/sorrel";
        };
        deploy-pages = plinth.lib.${system}.mkDeployPagesApp {
          domain = "sorrel.tartanoglu.com";
        };
        release-smoke = {
          type = "app";
          program = "${releaseSmoke}/bin/sorrel-release-smoke";
        };
      };
    });
}
