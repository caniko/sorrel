{
  pkgs,
  craneLib,
  commonArgs,
  cargoArtifacts,
  runtimeLibs,
  sorrel,
  testSrc,
  hmModule,
}: let
  inherit (pkgs) lib;

  # Evaluate the Home Manager module against a stub `home.packages`.
  hmPackages = enable:
    (lib.evalModules {
      modules = [
        hmModule
        {options.home.packages = lib.mkOption {type = lib.types.listOf lib.types.package;};}
        {programs.sorrel.enable = enable;}
      ];
      specialArgs = {inherit pkgs;};
    })
    .config
    .home
    .packages;
in
  {
    default = sorrel;

    clippy = craneLib.cargoClippy (commonArgs
      // {
        inherit cargoArtifacts;
        cargoClippyExtraArgs = "--workspace --all-targets -- --deny warnings";
      });

    hm-module = assert hmPackages true == [sorrel];
    assert hmPackages false == [];
      pkgs.runCommand "sorrel-hm-module" {} "touch $out";
  }
  // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
    # Every workspace test, the UI snapshot states included: they rasterise
    # on the pinned Mesa lavapipe (as in the dev shell) and must not skip.
    workspace-tests = craneLib.cargoNextest (commonArgs
      // {
        inherit cargoArtifacts;
        src = testSrc;
        # ponytail: the sandbox has no /etc/resolv.conf and libp2p's
        # with_websocket always builds its DNS transport from it, so the
        # remote round trips run only in ci.yaml's cargo lane. Hand-building
        # the websocket transport (swarm.rs, relay.rs) would bring them here.
        cargoNextestExtraArgs = "--workspace --no-fail-fast -E 'not (package(sorrel-remote) & binary(round_trip))'";
        SORREL_LAVAPIPE_ICD = "${pkgs.mesa}/share/vulkan/icd.d/lvp_icd.${pkgs.stdenv.hostPlatform.uname.processor}.json";
        SORREL_REQUIRE_GPU = "1";
        LD_LIBRARY_PATH = lib.makeLibraryPath runtimeLibs;
        preCheck = ''
          export HOME=$(mktemp -d)
          export XDG_RUNTIME_DIR=$HOME
        '';
      });
  }
