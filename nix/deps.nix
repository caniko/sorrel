{pkgs}: let
  inherit (pkgs) lib stdenv;

  # Native deps for the eframe + wgpu GUI on Linux. rusqlite is built with
  # the `bundled` feature, so libsqlite is compiled by cc — no system
  # sqlite needed in buildInputs.
  nativeBuildInputs = with pkgs; [
    clang
    mold
    pkg-config
  ];

  # Build-time linkable libs.
  buildInputs = with pkgs;
    [
      fontconfig
      freetype
    ]
    ++ lib.optionals stdenv.isLinux [
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

  # Libraries dlopen'd at runtime by wgpu/winit; rpath them into the
  # final binary so the package is self-contained on both Wayland and X11.
  runtimeLibs = with pkgs;
    lib.optionals stdenv.isLinux [
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

  # Optional libhdf5 for the NWB / Kilosort 4 rez.mat backends.
  # `hdf5-metno-sys`'s build script wants headers and lib under one
  # prefix, but nixpkgs splits libhdf5 into separate outputs (`out`
  # has lib/, `dev` has include/). Symlink-join both into a single tree
  # so the build script's HDF5_DIR detection works.
  hdf5C = pkgs.symlinkJoin {
    name = "hdf5-merged-${pkgs.hdf5.version}";
    paths = [pkgs.hdf5 pkgs.hdf5.dev];
  };
in {
  inherit nativeBuildInputs buildInputs runtimeLibs hdf5C;
}
