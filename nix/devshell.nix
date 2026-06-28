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
        # libhdf5 is pulled in unconditionally so `cargo check
        # --features hdf5` Just Works inside the dev shell. The default
        # `cargo build` doesn't reference it.
        hdf5
      ]
      ++ preCommitEnabledPackages
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
          hdf5
        ]
        ++ preCommitEnabledPackages
        ++ buildInputs
        ++ nativeBuildInputs;
      extraEnv = {
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;
        HDF5_DIR = "${hdf5C}";
      };
      extraShellHook = ''
        ${shellHook}
        echo "Documentation: mdbook serve docs"
      '';
    };
  }
