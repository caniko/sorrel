{
  pkgs,
  rs-harbor,
  craneLib,
  cross,
  cargoConfig,
  checks,
  deps,
}: let
  inherit (deps) buildInputs nativeBuildInputs runtimeLibs hdf5C;
in
  rs-harbor.lib.mkDevShells {
    inherit pkgs cross cargoConfig craneLib checks;

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
  }
