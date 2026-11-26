{
  pkgs,
  plinthProject,
  root,
}: let
  inherit ((builtins.fromTOML (builtins.readFile (root + "/Cargo.toml"))).workspace.package) version;
  inherit (pkgs) lib;
  project = builtins.fromTOML (builtins.readFile (root + "/website/plinth-project.toml"));
  releaseIndex = "https://github.com/caniko/sorrel/releases";
  releaseError = message: throw "website/plinth-project.toml: ${message}";
  isRelease = text:
    builtins.match ".*github[.]com/caniko/sorrel/releases([/#?].*)?" (lib.toLower text) != null;
  releaseUrls = value:
    if builtins.isAttrs value
    then releaseUrls (builtins.attrValues value)
    else if builtins.isList value
    then lib.concatMap releaseUrls value
    else if builtins.isString value
    then let
      text =
        builtins.replaceStrings
        ["&#47;" "&#x2f;" "&#x2F;" "&#58;" "&#x3a;" "&#x3A;"]
        ["/" "/" "/" ":" ":" ":"]
        value;
      # Inspect both quoted and unquoted URLs inside inline HTML.
      tokens = builtins.filter builtins.isString (builtins.split ''[[:space:]"'<>]+'' text);
      urls = map (token: lib.removePrefix "href=" (lib.removePrefix "src=" token)) tokens;
    in
      if isRelease text && builtins.match "[[:space:]]*(https?://|//).*" (lib.toLower text) != null
      then [text]
      else builtins.filter isRelease urls
    else [];
  versionParts = v:
    builtins.match "(0|[1-9][0-9]*)[.](0|[1-9][0-9]*)[.](0|[1-9][0-9]*)(-(alpha|beta|rc)[.](0|[1-9][0-9]*))?" v;
  compareReleaseVersions = a: b: let
    # compareVersions orders "-rc" after the final release. Rank the
    # workflow's prerelease labels explicitly, then compare numeric components.
    key = v: let
      parts = versionParts v;
      channel = builtins.elemAt parts 4;
      ranks = {
        alpha = "0";
        beta = "1";
        rc = "2";
      };
      rank =
        if channel == null
        then "3"
        else ranks.${channel};
      number =
        if channel == null
        then "0"
        else builtins.elemAt parts 5;
    in
      if parts == null
      then releaseError "Invalid release version ${v}."
      else lib.concatStringsSep "." (lib.take 3 parts ++ [rank number]);
  in
    builtins.compareVersions (key a) (key b);
  # The baseline is the published 0.4.0 asset list; newer names follow
  # nix/release.nix and the uploads in .github/workflows/release.yml.
  releaseAssets = v: let
    native =
      ["sorrel-${v}.tar.gz"]
      ++ map (suffix: "sorrel-${v}-${suffix}")
      ([
          "x86_64.AppImage"
          "x86_64-linux.tar.gz"
          "aarch64-linux.tar.gz"
          "aarch64-darwin.tar.gz"
          "x86_64-windows.zip"
          "x86_64-windows.exe"
          "x86_64-windows-gui.exe"
        ]
        ++ lib.optionals (compareReleaseVersions v "0.4.0" > 0) [
          "aarch64-windows.zip"
          "aarch64-windows.exe"
          "aarch64-windows-gui.exe"
          "third-party-notices.zip"
        ]);
  in
    ["SHA256SUMS.txt" "SHA256SUMS.txt.minisig"]
    ++ map (platform: "sorrel_ephys-${v}-cp38-abi3-${platform}.whl") [
      "manylinux_2_34_x86_64"
      "manylinux_2_34_aarch64"
      "macosx_11_0_arm64"
      "win_amd64"
    ]
    ++ lib.concatMap (name: [name "${name}.cosign.bundle" "${name}.intoto.bundle"]) native;
  linkVersion = url: let
    match = builtins.match "https://github[.]com/caniko/sorrel/releases/(tag|download)/([^/]+)(/([^/]+))?" url;
    kind = builtins.elemAt match 0;
    v = builtins.elemAt match 1;
    filename = builtins.elemAt match 3;
  in
    if match == null || versionParts v == null || (kind == "tag") != (filename == null)
    then releaseError "Invalid Sorrel release URL: ${url}"
    else if compareReleaseVersions v "0.4.0" < 0
    then releaseError "Sorrel release links target ${v}, older than the supported baseline 0.4.0."
    else if filename != null && !builtins.elem filename (releaseAssets v)
    then releaseError "Unexpected asset for Sorrel ${v}: ${filename}"
    else v;
  urls = builtins.filter (url: url != releaseIndex && url != "${releaseIndex}/") (releaseUrls project);
  versions = lib.unique (map linkVersion urls);
  websiteVersion =
    if builtins.length versions != 1
    then releaseError "Expected one version across Sorrel release links; found ${lib.concatStringsSep ", " versions}."
    else if compareReleaseVersions (builtins.head versions) version > 0
    then releaseError "Sorrel release links target ${builtins.head versions}, newer than workspace version ${version}."
    else version;
  website = pkgs.stdenv.mkDerivation {
    pname = "sorrel-website";
    version = websiteVersion;
    src = pkgs.lib.fileset.toSource {
      inherit root;
      fileset = pkgs.lib.fileset.unions [
        (root + "/website/plinth-project.toml")
        (root + "/website/site.css")
        (root + "/website/platform.js")
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
