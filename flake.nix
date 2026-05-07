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

      deps = import ./nix/deps.nix {inherit pkgs;};

      rustPackages = import ./nix/rust-packages.nix {
        inherit pkgs craneLib src deps;
      };
      inherit (rustPackages) sorrel sorrelHdf5 commonArgs cargoArtifacts;

      siteOutputs = import ./nix/site.nix {
        inherit pkgs;
        root = ./.;
      };
      inherit (siteOutputs) website docs site;
    in {
      packages = {
        default = sorrel;
        inherit sorrel sorrelHdf5 website docs site;
        sorrel-hdf5 = sorrelHdf5;
        cargo-config = cargoConfig.configPath;
      };

      checks = import ./nix/checks.nix {
        inherit craneLib src commonArgs cargoArtifacts sorrel;
      };

      devShells = import ./nix/devshell.nix {
        inherit pkgs rs-harbor craneLib cross cargoConfig deps;
        checks = self.checks.${system};
      };

      apps.default = {
        type = "app";
        program = "${sorrel}/bin/sorrel";
      };
    });
}
