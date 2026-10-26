# Distro-portable Linux builds for the release tarballs and the AppImage.
#
# A normal Nix build links nixpkgs' glibc (2.44 today) and RUNPATHs into
# /nix/store, so off NixOS it either fails to start or cannot load the host's
# GPU driver. cargo-zigbuild links against glibc 2.31 stubs instead: the
# binary uses the host's /lib64/ld-linux loader and dlopens the host's
# libvulkan/libGL/wayland/X11, which load its Mesa or NVIDIA driver. Only
# libc, libm and libgcc_s are linked directly.
{
  pkgs,
  craneLib,
  src,
  version,
  appimageRuntime,
}: let
  glibc = "2.31";

  portable = rustTarget:
    craneLib.buildPackage {
      inherit src version;
      pname = "sorrel-portable-${rustTarget}";
      strictDeps = true;
      # Dependencies must be compiled by zig too, so no shared deps derivation.
      cargoArtifacts = null;
      nativeBuildInputs = [pkgs.cargo-zigbuild pkgs.zig pkgs.cmake];
      # aws-lc-sys (via libp2p-quic) links s2n-bignum's assembly with
      # unresolvable symbols under its cc builder + zig; its CMake builder
      # links cleanly. cmake is only for that, not for configurePhase.
      AWS_LC_SYS_CMAKE_BUILDER = "1";
      dontUseCmakeConfigure = true;
      cargoBuildCommand = "cargo zigbuild --profile release --target ${rustTarget}.${glibc}";
      cargoExtraArgs = "-p sorrel --locked";
      doCheck = false;
      preBuild = ''
        export HOME=$TMPDIR ZIG_GLOBAL_CACHE_DIR=$TMPDIR/zig ZIG_LOCAL_CACHE_DIR=$TMPDIR/zig-local
      '';
      installPhaseCommand = ''
        install -Dm755 target/${rustTarget}/release/sorrel $out/bin/sorrel
      '';
    };

  x86_64 = portable "x86_64-unknown-linux-gnu";

  desktop = pkgs.writeText "sorrel.desktop" ''
    [Desktop Entry]
    Type=Application
    Name=Sorrel
    Comment=Spike-sorting curation
    Exec=sorrel
    Terminal=false
    Categories=Science;
  '';

  # A plain type-2 AppImage whose AppRun is the portable binary itself: no
  # bundled Nix closure, so the host's GPU driver stack is used as-is.
  appimage = pkgs.runCommand "sorrel-${version}.AppImage" {nativeBuildInputs = [pkgs.squashfsTools];} ''
    mkdir -p AppDir
    install -m755 ${x86_64}/bin/sorrel AppDir/AppRun
    install -m644 ${desktop} AppDir/sorrel.desktop
    mksquashfs AppDir $out -offset $(stat -L -c%s ${appimageRuntime}) -all-root -no-recovery
    dd if=${appimageRuntime} of=$out conv=notrunc status=none
    chmod 755 $out
  '';
in {
  sorrel-portable = x86_64;
  sorrel-portable-aarch64-linux = portable "aarch64-unknown-linux-gnu";
  sorrel-appimage = appimage;
}
