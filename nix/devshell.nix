{
  pkgs,
  harbor,
  craneLib,
  cross,
  cargoConfig,
  checks,
  deps,
  simit,
  preCommitEnabledPackages ? [],
  shellHook ? "",
}: let
  inherit (deps) buildInputs nativeBuildInputs runtimeLibs hdf5C;
  hdf5Packages = pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [pkgs.hdf5];
  hdf5Env =
    if pkgs.stdenv.hostPlatform.isLinux
    then {
      HDF5_DIR = "${hdf5C}";
    }
    else {};
  shells =
    (harbor.lib.rust.mkDevShells {
      inherit pkgs cross cargoConfig craneLib checks;

      pkgConfigDeps = buildInputs;
      packages = with pkgs;
        [
          cargo-nextest
          imagemagick
          mdbook
          pre-commit
          rust-analyzer
          simit
          python3
          zola
        ]
        ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [pkgs.xdotool pkgs.xorg-server]
        ++ hdf5Packages
        ++ preCommitEnabledPackages
        ++ buildInputs
        ++ nativeBuildInputs;

      extraEnv = {LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;} // hdf5Env;

      extraShellHook = ''
        ${shellHook}
        echo "Website: cd website && zola serve"
        echo "Documentation: cd docs && mdbook serve"
      '';
    })
    // {
      docs = harbor.lib.rust.mkDocsShell {
        inherit pkgs cross cargoConfig craneLib checks;
        pkgConfigDeps = buildInputs;
        packages = with pkgs;
          [
            cargo-nextest
            mdbook
            pre-commit
            rust-analyzer
            simit
          ]
          ++ hdf5Packages
          ++ preCommitEnabledPackages
          ++ buildInputs
          ++ nativeBuildInputs;
        extraEnv = {LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;} // hdf5Env;
        extraShellHook = ''
          ${shellHook}
          echo "Documentation: mdbook serve docs"
        '';
      };
    };
in
  pkgs.lib.mapAttrs (_: shell:
    shell.overrideAttrs (old: {
      # Tooling closures are build-platform tools, not host link dependencies.
      # Mixing their headers/libraries into native compiler flags overflows
      # GCC's collect2 environment in a fresh CI shell.
      strictDeps = true;
      buildInputs = pkgs.lib.unique ((old.buildInputs or []) ++ buildInputs ++ hdf5Packages);
    }))
  shells
