{
  pkgs,
  system,
  spikeinterface,
}:
if system != "x86_64-linux"
then {
  packages = {};
  devShells = {};
}
else let
  inherit (spikeinterface.lib.${system}) mkEnv;
  mkLupin = variant:
    mkEnv {
      name = "sorrel-lupin-${variant}";
      extras = ["lupin" "nwb" variant];
    };
  mkShell = variant: env:
    pkgs.mkShellNoCC {
      name = "sorrel-lupin-${variant}";
      packages = [env];
    };
  packages = rec {
    lupin-cpu = mkLupin "cpu";
    lupin-amd = mkLupin "amd";
    lupin-nvidia = mkLupin "nvidia";
    lupin-rocm = lupin-amd;
    lupin-cuda = lupin-nvidia;
  };
in {
  inherit packages;
  devShells = rec {
    lupin-cpu = mkShell "cpu" packages.lupin-cpu;
    lupin-amd = mkShell "amd" packages.lupin-amd;
    lupin-nvidia = mkShell "nvidia" packages.lupin-nvidia;
    lupin-rocm = lupin-amd;
    lupin-cuda = lupin-nvidia;
  };
}
