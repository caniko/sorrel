#!/usr/bin/env bash
set -euo pipefail

VERSION="${VERSION:?VERSION must be set by the release workflow}"
OUT_DIR="${1:-release}"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

mkdir -p "$OUT_DIR"

copy_binary() {
  local result_dir="$1" binary="$2" destination="$3"
  if [[ -f "$result_dir/bin/.${binary}-wrapped" ]]; then
    cp "$result_dir/bin/.${binary}-wrapped" "$destination"
  else
    cp "$result_dir/bin/$binary" "$destination"
  fi
}

make_tarball() {
  local source_dir="$1" archive="$2"
  tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
    -C "$source_dir" -cf - . | gzip -n > "$archive"
}

build_binary_archive() {
  local attr="$1" platform="$2" archive="$3" binary_name="$4"
  nix build ".#$attr" --out-link "$WORK_DIR/$attr"
  mkdir -p "$WORK_DIR/$platform"
  copy_binary "$WORK_DIR/$attr" "$binary_name" "$WORK_DIR/$platform/$binary_name"
  make_tarball "$WORK_DIR/$platform" "$OUT_DIR/$archive"
}

build_binary_archive \
  sorrel \
  linux-x86_64 \
  "sorrel-${VERSION}-x86_64-linux.tar.gz" \
  sorrel

build_binary_archive \
  sorrel-aarch64-linux \
  linux-aarch64 \
  "sorrel-${VERSION}-aarch64-linux.tar.gz" \
  sorrel

nix build .#sorrel-windows --out-link "$WORK_DIR/sorrel-windows"
mkdir -p "$WORK_DIR/windows-x86_64"
cp "$WORK_DIR/sorrel-windows/bin/sorrel.exe" "$WORK_DIR/windows-x86_64/sorrel.exe"
touch -d '@0' "$WORK_DIR/windows-x86_64/sorrel.exe"
if [[ "$OUT_DIR" = /* ]]; then
  WINDOWS_ARCHIVE="$OUT_DIR/sorrel-${VERSION}-x86_64-windows.zip"
else
  WINDOWS_ARCHIVE="$PWD/$OUT_DIR/sorrel-${VERSION}-x86_64-windows.zip"
fi
if command -v zip >/dev/null 2>&1; then
  (cd "$WORK_DIR/windows-x86_64" && zip -X -q "$WINDOWS_ARCHIVE" sorrel.exe)
else
  (cd "$WORK_DIR/windows-x86_64" && nix shell nixpkgs#zip -c zip -X -q "$WINDOWS_ARCHIVE" sorrel.exe)
fi

build_binary_archive \
  sorrel-darwin-aarch64 \
  darwin-aarch64 \
  "sorrel-${VERSION}-aarch64-darwin.tar.gz" \
  sorrel

nix build .#sorrel-appimage --out-link "$WORK_DIR/sorrel-appimage"
cp "$(readlink -f "$WORK_DIR/sorrel-appimage")" "$OUT_DIR/sorrel-${VERSION}-x86_64.AppImage"

git archive --format=tar --prefix="sorrel-${VERSION}/" HEAD | \
  gzip -n > "$OUT_DIR/sorrel-${VERSION}.tar.gz"

shopt -s nullglob
artifacts=(
  "$OUT_DIR"/sorrel-"$VERSION"-*.tar.gz
  "$OUT_DIR"/sorrel-"$VERSION"-*.zip
  "$OUT_DIR"/sorrel-"$VERSION"-*.AppImage
  "$OUT_DIR"/sorrel-"$VERSION".tar.gz
)
for artifact in "${artifacts[@]}"; do
  printf '%s\n' "${artifact#"$OUT_DIR"/}"
done | LC_ALL=C sort
