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
in
  mkReleaseBundle {
    inherit pkgs pname version;
    artifacts = {
      x86_64-linux = archive {
        package = packages.sorrel;
        system = "x86_64-linux";
        rustTarget = "x86_64-unknown-linux-gnu";
      };
      aarch64-linux = archive {
        package = packages.sorrel-aarch64-linux;
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
    };
  }
