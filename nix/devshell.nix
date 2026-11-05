{
  pkgs,
  plinthProject,
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
      ]
      ++ [plinthProject]
      ++ hdf5Packages
      ++ preCommitEnabledPackages
      ++ buildInputs;

    extraEnv =
      {
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;
        # Match the release cross build: aws-lc's jitterentropy includes sched.h,
        # which lives in winpthreads rather than the MinGW compiler's headers.
        CFLAGS_x86_64_pc_windows_gnu = "-I${cross.winpthreads}/include";
      }
      // hdf5Env // lavapipeEnv;

    extraShellHook = ''
      ${shellHook}
      ${pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isLinux ''
        # windres invokes gcc by basename. Add only the bin directories: adding
        # the cross compiler as a shell package runs its setup hook and wrongly
        # sets CC_<linux host> to the Windows compiler during native tests.
        export PATH="${cross.mingwCC}/bin:${cross.mingwBinutils}/bin:$PATH"
      ''}
      echo "Website: plinth-project dev --config website/plinth-project.toml"
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
