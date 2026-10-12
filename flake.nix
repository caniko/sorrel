{
  description = "Sorrel — spike-sorting curation GUI";

  inputs = {
    harbor.url = "git+https://github.com/caniko/harbor.git?ref=feat/harbor-monorepo-components&rev=e8b1996580e4ecd51dd402edf4a16d9df88e3a1b";

    simit = {
      url = "git+https://github.com/caniko/simit.git?ref=trunk&rev=532a5830057506586ffc94f4dd2dc66783e6842c";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.rust-overlay.follows = "rust-overlay";
      inputs.flake-utils.follows = "flake-utils";
      inputs.plinth.follows = "plinth";
    };

    nix-appimage = {
      url = "github:ralismark/nix-appimage";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    nixpkgs.follows = "harbor/nixpkgs";
    rust-overlay.follows = "harbor/rust-overlay";
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
      url = "git+https://github.com/caniko/plinth.git";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    harbor,
    simit,
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
          provider = "actions";
          platform = "github";
          runner = "ubuntu-24.04";
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
            site_output = ".#site";
            token_secret = "GITHUB_TOKEN";
            source_branch = "trunk";
            deploy_app = ".#deploy-pages";
          };
        };
        release = {
          publish.enforcement = "activated-remote";
          smoke.command = "nix run .#release-smoke --";
          github = {
            repo = "caniko/sorrel";
            target_branch = "trunk";
            token_secret = "GITHUB_TOKEN";
          };
          artifacts = {
            runner = "ubuntu-24.04";
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
      pkgs =
        import (
          if system == "x86_64-darwin"
          then harbor.inputs.nixpkgs-darwin
          else nixpkgs
        ) {
          inherit system;
          overlays = [(import rust-overlay)];
        };

      toolchain = harbor.lib.rust.mkToolchain {
        inherit pkgs;
        toolchainProfile = "nightly";
      };
      inherit (toolchain) craneLib rustToolchain;
      cross = harbor.lib.rust.mkCross {
        inherit pkgs system;
        macosSdkStorePath = harbor.lib.pins.macosSdk.storePath;
        macosSdkOutputHash = harbor.lib.pins.macosSdk.outputHash;
        osxSdkVersion = harbor.lib.pins.macosSdk.sdkVersion;
      };
      cargoConfig = harbor.lib.rust.mkCargoConfig {inherit pkgs;};

      src = pkgs.lib.cleanSourceWith {
        src = ./.;
        filter = path: type:
          (craneLib.filterCargoSources path type)
          || pkgs.lib.hasSuffix ".wgsl" path;
      };
      treefmtEval = treefmt-nix.lib.evalModule pkgs (import ./nix/treefmt.nix);
      gitHooks =
        if system == "x86_64-darwin"
        then
          import "${git-hooks}/nix" {
            nixpkgs = harbor.inputs.nixpkgs-darwin;
            inherit system;
            isFlakes = true;
          }
        else git-hooks.lib.${system};
      pre-commit-check = gitHooks.run {
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
      crossSorrelPackages = harbor.lib.rust.mkCrossPackages {
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

      sorrelAppImage = harbor.lib.rust.mkAppImage {
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
        // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {sorrel-appimage = sorrelAppImage;};

      checks =
        (import ./nix/checks.nix {
          inherit craneLib src commonArgs cargoArtifacts sorrel;
        })
        // {
          formatting = treefmtEval.config.build.check self;
        };

      devShells = import ./nix/devshell.nix {
        inherit pkgs harbor craneLib cross cargoConfig deps;
        # The current Simit flake omits Intel macOS. Build its accepted source
        # with this platform's compatible package set for the editor shell.
        simit =
          if system == "x86_64-darwin"
          then
            craneLib.buildPackage {
              pname = "simit";
              version = (builtins.fromTOML (builtins.readFile (simit + "/Cargo.toml"))).package.version;
              src = simit;
              cargoExtraArgs = "--locked --bin simit";
              strictDeps = true;
              doCheck = false;
            }
          else simit.packages.${system}.default;
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
        deploy-pages =
          (import "${plinth}/nix/project-site.nix" {
            inherit pkgs;
            inherit (pkgs) lib;
          }).mkDeployPagesApp {
            domain = "sorrel.tartanoglu.com";
          };
        release-smoke = {
          type = "app";
          program = "${releaseSmoke}/bin/sorrel-release-smoke";
        };
      };
    });
}
