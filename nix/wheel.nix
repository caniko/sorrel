# Python wheels for `sorrel-py`, one per release platform: the abi3 `_native`
# extension, the Python package and the `sorrel` GUI binary (installed into
# the environment's scripts dir, so `sorrel.launch` finds it on PATH).
#
# Assembled from prebuilt Nix outputs instead of `maturin build`, so the
# wheels reuse the release cross builds. `wheel pack` writes RECORD.
# ponytail: no sdist (pip would try to compile Rust on platforms without a
# wheel); add one, or musl / Intel macOS wheels, when someone asks.
{
  pkgs,
  version,
  # name -> { native = path to the cdylib; binary = path to the executable;
  #           platform = wheel platform tag; ext = extension module suffix;
  #           exe = installed binary name;
  #           elf = true to drop the /nix/store RUNPATH from a Linux cdylib; }
  targets,
}: let
  inherit (pkgs) lib;
  pyproject = builtins.fromTOML (builtins.readFile ../crates/sorrel-py/pyproject.toml);
  crateVersion = (builtins.fromTOML (builtins.readFile ../crates/sorrel-py/Cargo.toml)).package.version;
  project = pyproject.project;

  # The package sources, without tests or bytecode.
  python = lib.cleanSourceWith {
    src = ../crates/sorrel-py/python/sorrel;
    filter = path: _: baseNameOf path != "__pycache__" && !(lib.hasSuffix ".pyc" path);
  };

  metadata = pkgs.writeText "METADATA" (lib.concatStringsSep "\n" ([
      "Metadata-Version: 2.1"
      "Name: ${project.name}"
      "Version: ${version}"
      "Summary: ${project.description}"
      "Requires-Python: ${project.requires-python}"
    ]
    ++ lib.concatLists (lib.mapAttrsToList (extra: reqs:
      ["Provides-Extra: ${extra}"] ++ map (r: "Requires-Dist: ${r}; extra == \"${extra}\"") reqs)
    project.optional-dependencies)
    ++ [""]));

  # PyPI `sorrel` belongs to another project; the distribution is
  # `sorrel-ephys` (wheel names normalise `-` to `_`), the import stays `sorrel`.
  dist = builtins.replaceStrings ["-"] ["_"] project.name;

  mkWheel = name: t: let
    tag = "cp38-abi3-${t.platform}";
    file = "${dist}-${version}-${tag}.whl";
    root = "${dist}-${version}";
  in
    pkgs.runCommand "sorrel-wheel-${name}-${version}" {
      nativeBuildInputs = [pkgs.python3Packages.wheel pkgs.patchelf];
      passthru = {inherit file;};
    } ''
      root=$TMPDIR/${root}
      mkdir -p $root/sorrel $root/${root}.dist-info $root/${root}.data/scripts
      cp -r ${python}/. $root/sorrel/
      chmod -R u+w $root/sorrel
      install -m755 ${t.native} $root/sorrel/_native${t.ext}
      ${lib.optionalString (t.elf or false) "patchelf --remove-rpath $root/sorrel/_native${t.ext}"}
      install -m755 ${t.binary} $root/${root}.data/scripts/${t.exe}
      install -m644 ${metadata} $root/${root}.dist-info/METADATA
      printf 'Wheel-Version: 1.0\nGenerator: sorrel-nix\nRoot-Is-Purelib: false\nTag: ${tag}\n' \
        > $root/${root}.dist-info/WHEEL
      mkdir -p $out
      wheel pack $root -d $out
      test -f $out/${file}
    '';
in
  assert lib.assertMsg (project.version == version && crateVersion == version)
  "sorrel-py: pyproject.toml (${project.version}) and Cargo.toml (${crateVersion}) must match the workspace version ${version}";
    lib.mapAttrs mkWheel targets
