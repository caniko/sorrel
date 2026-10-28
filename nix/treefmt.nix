{rustfmtPackage}: {pkgs, ...}: {
  projectRootFile = "flake.nix";

  programs.rustfmt = {
    enable = true;
    edition = "2021";
    package = rustfmtPackage;
  };

  programs.alejandra.enable = true;

  programs.taplo.enable = true;

  programs.prettier = {
    enable = true;
    package = pkgs.prettier;
    # simit-generated workflows; simit's --check owns their drift.
    excludes = [
      ".crow/**"
      ".github/workflows/**"
    ];
    includes = [
      "*.md"
      "*.markdown"
      "*.yaml"
      "*.yml"
    ];
  };
}
