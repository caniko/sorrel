# Home Manager module: `programs.sorrel.enable` installs Sorrel (and its
# menu entry, which the package ships).
self: {
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.programs.sorrel;
in {
  options.programs.sorrel = {
    enable = lib.mkEnableOption "Sorrel, the spike-sorting curation GUI";
    package = lib.mkOption {
      type = lib.types.package;
      default = self.packages.${pkgs.stdenv.hostPlatform.system}.sorrel;
      defaultText = lib.literalExpression "sorrel.packages.\${system}.sorrel";
      description = "The Sorrel package; use `sorrelHdf5` for NWB/HDF5 support.";
    };
  };

  config = lib.mkIf cfg.enable {
    home.packages = [cfg.package];
  };
}
