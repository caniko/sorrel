{
  pkgs,
  plinthProject,
  root,
}: let
  inherit ((builtins.fromTOML (builtins.readFile (root + "/Cargo.toml"))).workspace.package) version;
  website = pkgs.stdenv.mkDerivation {
    pname = "sorrel-website";
    inherit version;
    src = pkgs.lib.fileset.toSource {
      inherit root;
      fileset = pkgs.lib.fileset.unions [
        (root + "/website/plinth-project.toml")
        (root + "/website/site.css")
        (root + "/website/fonts")
        (root + "/assets/showcase")
        (root + "/assets/icon/sorrel.svg")
      ];
    };
    nativeBuildInputs = [plinthProject];
    phases = ["buildPhase" "installPhase"];
    buildPhase = ''
      plinth-project build --config $src/website/plinth-project.toml --out public
    '';
    installPhase = ''
      cp -r public $out
    '';
  };

  docs = pkgs.stdenv.mkDerivation {
    pname = "sorrel-docs";
    inherit version;
    src = pkgs.lib.fileset.toSource {
      inherit root;
      fileset = pkgs.lib.fileset.maybeMissing (root + "/docs");
    };
    nativeBuildInputs = [pkgs.mdbook];
    buildPhase = ''
      mdbook build docs
    '';
    installPhase = ''
      cp -r docs/book $out
    '';
  };

  site = pkgs.runCommand "sorrel-site" {} ''
    mkdir -p $out
    cp -r ${website}/* $out/
    mkdir -p $out/docs
    cp -r ${docs}/* $out/docs/
    printf '%s\n' sorrel.tartanoglu.com > $out/.domains
  '';
in {
  inherit website docs site;
}
