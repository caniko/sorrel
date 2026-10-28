{
  pkgs,
  harbor-rs,
  craneLib,
  cross,
  cargoConfig,
  checks,
  deps,
  preCommitEnabledPackages ? [],
  shellHook ? "",
}: let
  inherit (deps) buildInputs runtimeLibs hdf5C;
  hdf5Packages = pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [pkgs.hdf5];
  # Pinned software Vulkan for the headless UI snapshot tests, so atlas and
  # GPU-less CI runners rasterise identically. Only a path: the GUI keeps
  # using the host driver.
  lavapipeEnv = pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
    SORREL_LAVAPIPE_ICD = "${pkgs.mesa}/share/vulkan/icd.d/lvp_icd.${pkgs.stdenv.hostPlatform.uname.processor}.json";
  };
  hdf5Env =
    if pkgs.stdenv.hostPlatform.isLinux
    then {
      HDF5_DIR = "${hdf5C}";
    }
    else {};
in
  (harbor-rs.lib.mkDevShells {
    inherit pkgs cross cargoConfig craneLib checks;

    pkgConfigDeps = buildInputs;
    packages = with pkgs;
      [
        cargo-nextest
        imagemagick
        mdbook
        pre-commit
        python3
        xdotool
        xorg.xorgserver
        zola
      ]
      ++ hdf5Packages
      ++ preCommitEnabledPackages
      ++ buildInputs;

    extraEnv = {LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;} // hdf5Env // lavapipeEnv;

    extraShellHook = ''
      ${shellHook}
      echo "Website: cd website && zola serve"
      echo "Documentation: cd docs && mdbook serve"
    '';
  })
  // {
    docs = harbor-rs.lib.mkDocsShell {
      inherit pkgs cross cargoConfig craneLib checks;
      pkgConfigDeps = buildInputs;
      packages = with pkgs;
        [
          cargo-nextest
          mdbook
          pre-commit
        ]
        ++ hdf5Packages
        ++ preCommitEnabledPackages
        ++ buildInputs;
      extraEnv = {LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;} // hdf5Env;
      extraShellHook = ''
        ${shellHook}
        echo "Documentation: mdbook serve docs"
      '';
    };
  }
