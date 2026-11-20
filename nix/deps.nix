{pkgs}: let
  inherit (pkgs) lib stdenv;

  # Native deps for the eframe + wgpu GUI on Linux. rusqlite is built with
  # the `bundled` feature, so libsqlite is compiled by cc — no system
  # sqlite needed in buildInputs.
  nativeBuildInputs = with pkgs; [
    clang
    cmake
    mold
    pkg-config
  ];

  # Build-time linkable libs for the native Linux package.  Keep these out of
  # the cross-target argument sets below: Windows and macOS use their native
  # window/GPU stacks and must not inherit Linux pkg-config dependencies.
  linuxBuildInputs = with pkgs; [
    fontconfig
    freetype
    # Wayland stack
    wayland
    libxkbcommon
    libdecor
    # X11 stack (winit fallback)
    libx11
    libxcursor
    libxi
    libxrandr
    libxcb
    # GPU
    libGL
    vulkan-loader
  ];

  aarch64LinuxPkgs = pkgs.pkgsCross.aarch64-multiplatform;
  aarch64LinuxBuildInputs = with aarch64LinuxPkgs; [
    fontconfig
    freetype
    wayland
    libxkbcommon
    libdecor
    libx11
    libxcursor
    libxi
    libxrandr
    libxcb
    libGL
    vulkan-loader
  ];

  # Libraries dlopen'd at runtime by wgpu/winit; rpath them into the
  # final binary so the package is self-contained on both Wayland and X11.
  runtimeLibs = with pkgs;
    lib.optionals stdenv.hostPlatform.isLinux [
      vulkan-loader
      libGL
      # Wayland: winit dlopens libwayland-client + libxkbcommon, and
      # libdecor for client-side decorations on GNOME / KDE wayland.
      wayland
      libxkbcommon
      libdecor
      # X11 fallback
      libx11
      libxcursor
      libxi
      libxrandr
      libxcb
      fontconfig
      freetype
    ];
in {
  buildInputs = lib.optionals stdenv.hostPlatform.isLinux linuxBuildInputs;
  inherit nativeBuildInputs linuxBuildInputs aarch64LinuxBuildInputs runtimeLibs;
}
