# The menu entry shared by the Nix packages and the AppImage.
{pkgs}:
pkgs.writeText "sorrel.desktop" ''
  [Desktop Entry]
  Type=Application
  Name=Sorrel
  Comment=Spike-sorting curation
  Exec=sorrel
  Terminal=false
  Categories=Science;
''
