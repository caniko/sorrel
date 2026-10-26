# Flat release bundle consumed by simit's `[release.artifacts].prebuild_binaries`.
# The source tarball stays a simit `build_commands` line (it needs `git archive`).
{
  pkgs,
  harbor-rs,
  version,
  packages,
}: let
  inherit (harbor-rs.lib) mkReleaseArchive mkReleaseArtifact mkReleaseBundle;
  pname = "sorrel";

  archive = {
    package,
    system,
    rustTarget,
    format ? "tar.gz",
    binary ? "sorrel",
  }:
    mkReleaseArchive {
      inherit pkgs pname version package system rustTarget format;
      name = "sorrel-${version}-${system}.${format}";
      entries.${binary} = "bin/${binary}";
    };

  # `pip install sorrel-ephys` wheels: the `_native` extension plus the GUI binary.
  # The tag covers both files: the portable binary needs glibc 2.31, the
  # extension (plain nixpkgs toolchain) 2.34, so 2.34 binds.
  # ponytail: build the extension with zigbuild too for manylinux_2_31
  # (Ubuntu 20.04, Debian 11) if anyone needs them.
  wheels = import ./wheel.nix {
    inherit pkgs version;
    targets = {
      x86_64-linux = {
        native = "${packages.sorrel-py}/lib/libsorrel.so";
        binary = "${packages.sorrel-portable}/bin/sorrel";
        platform = "manylinux_2_34_x86_64";
        ext = ".abi3.so";
        exe = "sorrel";
        elf = true;
      };
      aarch64-linux = {
        native = "${packages.sorrel-py-aarch64-linux}/lib/libsorrel.so";
        binary = "${packages.sorrel-portable-aarch64-linux}/bin/sorrel";
        platform = "manylinux_2_34_aarch64";
        ext = ".abi3.so";
        exe = "sorrel";
        elf = true;
      };
      windows = {
        native = "${packages.sorrel-py-windows}/lib/sorrel.dll";
        binary = "${packages.sorrel-windows}/bin/sorrel.exe";
        platform = "win_amd64";
        ext = ".pyd";
        exe = "sorrel.exe";
      };
      darwin-aarch64 = {
        native = "${packages.sorrel-py-darwin-aarch64}/lib/libsorrel.dylib";
        binary = "${packages.sorrel-darwin-aarch64}/bin/sorrel";
        platform = "macosx_11_0_arm64";
        ext = ".abi3.so";
        exe = "sorrel";
      };
    };
  };
  wheelArtifact = system: wheel:
    mkReleaseArtifact {
      inherit pkgs pname version system;
      name = wheel.file;
      source = wheel;
      sourcePath = wheel.file;
    };
in
  mkReleaseBundle {
    inherit pkgs pname version;
    artifacts = {
      x86_64-linux = archive {
        package = packages.sorrel-portable;
        system = "x86_64-linux";
        rustTarget = "x86_64-unknown-linux-gnu";
      };
      aarch64-linux = archive {
        package = packages.sorrel-portable-aarch64-linux;
        system = "aarch64-linux";
        rustTarget = "aarch64-unknown-linux-gnu";
      };
      windows = archive {
        package = packages.sorrel-windows;
        system = "x86_64-windows";
        rustTarget = "x86_64-pc-windows-gnu";
        format = "zip";
        binary = "sorrel.exe";
      };
      darwin-aarch64 = archive {
        package = packages.sorrel-darwin-aarch64;
        system = "aarch64-darwin";
        rustTarget = "aarch64-apple-darwin";
      };
      appimage = mkReleaseArtifact {
        inherit pkgs pname version;
        name = "sorrel-${version}-x86_64.AppImage";
        source = packages.sorrel-appimage;
        system = "x86_64-linux";
        executable = true;
      };
      wheel-x86_64-linux = wheelArtifact "x86_64-linux" wheels.x86_64-linux;
      wheel-aarch64-linux = wheelArtifact "aarch64-linux" wheels.aarch64-linux;
      wheel-windows = wheelArtifact "x86_64-windows" wheels.windows;
      wheel-darwin-aarch64 = wheelArtifact "aarch64-darwin" wheels.darwin-aarch64;
    };
  }
  # `nix build .#release-bundle.wheels.<target>` builds one wheel.
  // {inherit wheels;}
