{
  pkgs,
  root,
}: let
  website = pkgs.stdenv.mkDerivation {
    pname = "sorrel-website";
    version = "0.1.0";
    src = pkgs.lib.fileset.toSource {
      inherit root;
      fileset = pkgs.lib.fileset.maybeMissing (root + "/website");
    };
    nativeBuildInputs = [pkgs.zola];
    # Zola constructs its data-loader HTTP client even for local templates.
    # Supply the sandbox's trust roots through a store-backed certificate bundle.
    SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
    phases = ["buildPhase" "installPhase"];
    buildPhase = ''
      cp -r --no-preserve=mode $src/website site
      cd site && zola build
    '';
    installPhase = ''
      cp -r public $out
    '';
  };

  docs = pkgs.stdenv.mkDerivation {
    pname = "sorrel-docs";
    version = "0.1.0";
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
