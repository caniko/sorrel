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
    version = "0.2.0";
  };

  cargoArtifacts = craneLib.buildDepsOnly commonArgs;

  rpathFixup = pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isLinux ''
    patchelf --add-rpath "${pkgs.lib.makeLibraryPath runtimeLibs}" \
      "$out/bin/sorrel"
  '';

  sorrelHdf5 =
    if pkgs.stdenv.hostPlatform.isLinux
    then
      craneLib.buildPackage (commonArgs
        // {
          inherit cargoArtifacts;
          buildInputs = buildInputs ++ [hdf5C];
          cargoExtraArgs = "-p sorrel --locked --features hdf5";
          postFixup = rpathFixup;
        })
    else null;

  sorrel = craneLib.buildPackage (commonArgs
    // {
      inherit cargoArtifacts;
      cargoExtraArgs = "-p sorrel --locked";
      # Add an rpath so the binary finds vulkan/wayland/etc. without
      # leaking the dev shell's LD_LIBRARY_PATH.
      postFixup = rpathFixup;

      meta = with pkgs.lib; {
        description = "Native, monomorphised spike-sorting curation GUI";
        homepage = "https://github.com/caniko/sorrel";
        license = with licenses; [mit asl20];
        mainProgram = "sorrel";
        platforms = platforms.unix;
      };
    });
  sorrelRelay = craneLib.buildPackage (commonArgs
    // {
      inherit cargoArtifacts;
      pname = "sorrel-relay";
      cargoExtraArgs = "-p sorrel-remote --bin sorrel-relay --locked";
      doCheck = false;
      meta = with pkgs.lib; {
        description = "libp2p circuit relay for Sorrel remote work (proxy mode)";
        homepage = "https://github.com/caniko/sorrel";
        license = with licenses; [mit asl20];
        mainProgram = "sorrel-relay";
        platforms = platforms.unix;
      };
    });
in {
  inherit sorrel sorrelHdf5 sorrelRelay commonArgs cargoArtifacts;
}
