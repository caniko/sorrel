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
        # `dev` is additive (nothing is cfg(not(feature = "dev"))), so one
        # pass with it lints the default build too.
        cargoClippyExtraArgs = "--workspace --all-targets --features sorrel/dev -- --deny warnings";
      });

    hm-module = assert hmPackages true == [sorrel];
    assert hmPackages false == [];
      pkgs.runCommand "sorrel-hm-module" {} "touch $out";
  }
  // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
    # Every workspace test, the UI snapshot states included: they rasterise
    # on the pinned Mesa lavapipe (as in the dev shell) and must not skip.
    # hdf5 is a default feature, so this also runs the NWB and rez.mat
    # readers against release/fixtures (a separate HDF5 check would rebuild
    # every test binary and run the runner out of memory in flake check).
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

    # `sorrel dev ...` (bench, learn) and the bench results DB live behind the
    # optional `dev` feature, which workspace-tests does not enable.
    dev-tests = craneLib.cargoNextest (commonArgs
      // {
        inherit cargoArtifacts;
        src = testSrc;
        cargoNextestExtraArgs = "-p sorrel -p sorrel-data --features sorrel/dev --no-fail-fast";
        preCheck = ''
          export HOME=$(mktemp -d)
          export XDG_RUNTIME_DIR=$HOME
        '';
      });
  }
