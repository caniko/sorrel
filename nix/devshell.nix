{
  pkgs,
  rs-harbor,
  craneLib,
  cross,
  cargoConfig,
  checks,
  deps,
  preCommitEnabledPackages ? [],
  shellHook ? "",
}: let
  inherit (deps) buildInputs nativeBuildInputs runtimeLibs hdf5C;
  hdf5Packages = pkgs.lib.optionals pkgs.stdenv.isLinux [pkgs.hdf5];
  hdf5Env =
    if pkgs.stdenv.isLinux
    then {
      HDF5_DIR = "${hdf5C}";
    }
    else {};
in
  (rs-harbor.lib.mkDevShells {
    inherit pkgs cross cargoConfig craneLib checks;

    pkgConfigDeps = buildInputs;
    packages = with pkgs;
      [
        cargo-nextest
        mdbook
        pre-commit
        rust-analyzer
        zola
      ]
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
    docs = rs-harbor.lib.mkDocsShell {
      inherit pkgs cross cargoConfig craneLib checks;
      pkgConfigDeps = buildInputs;
      packages = with pkgs;
        [
          cargo-nextest
          mdbook
          pre-commit
          rust-analyzer
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
  }
