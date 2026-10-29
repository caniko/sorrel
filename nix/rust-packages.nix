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
    # One version source: simit release bumps the workspace manifest.
    version = (builtins.fromTOML (builtins.readFile ../Cargo.toml)).workspace.package.version;
  };

  cargoArtifacts = craneLib.buildDepsOnly commonArgs;

  # `sorrel completions` needs to run the binary, so cross builds skip it.
  completions = pkgs.lib.optionalString (pkgs.stdenv.buildPlatform.canExecute pkgs.stdenv.hostPlatform) ''
    installShellCompletion --cmd sorrel \
      --bash <("$out/bin/sorrel" completions bash) \
      --zsh <("$out/bin/sorrel" completions zsh) \
      --fish <("$out/bin/sorrel" completions fish)
  '';

  # Also installs the menu entry, so every Nix install (profile, Home
  # Manager, NixOS) gets one.
  rpathFixup = pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isLinux ''
    patchelf --add-rpath "${pkgs.lib.makeLibraryPath runtimeLibs}" \
      "$out/bin/sorrel"
    install -Dm644 ${import ./desktop.nix {inherit pkgs;}} \
      "$out/share/applications/sorrel.desktop"
  '';

  sorrelHdf5 =
    if pkgs.stdenv.hostPlatform.isLinux
    then
      craneLib.buildPackage (commonArgs
        // {
          inherit cargoArtifacts;
          buildInputs = buildInputs ++ [hdf5C];
          cargoExtraArgs = "-p sorrel --locked --features hdf5";
          nativeBuildInputs = nativeBuildInputs ++ [pkgs.installShellFiles];
          postInstall = completions;
          postFixup = rpathFixup;
        })
    else null;

  sorrel = craneLib.buildPackage (commonArgs
    // {
      inherit cargoArtifacts;
      cargoExtraArgs = "-p sorrel --locked";
      nativeBuildInputs = nativeBuildInputs ++ [pkgs.installShellFiles];
      postInstall = completions;
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
