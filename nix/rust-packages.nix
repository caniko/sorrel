{
  pkgs,
  craneLib,
  src,
  deps,
}: let
  inherit (deps) buildInputs nativeBuildInputs runtimeLibs hdf5C;

  commonArgs = {
    inherit src buildInputs nativeBuildInputs;
    strictDeps = true;
    pname = "sorrel";
    version = "0.1.0";
  };

  cargoArtifacts = craneLib.buildDepsOnly commonArgs;

  rpathFixup = pkgs.lib.optionalString pkgs.stdenv.isLinux ''
    patchelf --add-rpath "${pkgs.lib.makeLibraryPath runtimeLibs}" \
      "$out/bin/sorrel"
  '';

  sorrelHdf5 = craneLib.buildPackage (commonArgs
    // {
      inherit cargoArtifacts;
      buildInputs = buildInputs ++ [hdf5C];
      cargoExtraArgs = "-p sorrel --locked --features hdf5";
      postFixup = rpathFixup;
    });

  sorrel = craneLib.buildPackage (commonArgs
    // {
      inherit cargoArtifacts;
      cargoExtraArgs = "-p sorrel --locked";
      # Add an rpath so the binary finds vulkan/wayland/etc. without
      # leaking the dev shell's LD_LIBRARY_PATH.
      postFixup = rpathFixup;

      meta = with pkgs.lib; {
        description = "Native, monomorphised spike-sorting curation GUI";
        homepage = "https://codeberg.org/caniko/sorrel";
        license = with licenses; [mit asl20];
        mainProgram = "sorrel";
        platforms = platforms.unix;
      };
    });
in {
  inherit sorrel sorrelHdf5 commonArgs cargoArtifacts;
}
