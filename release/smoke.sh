#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:?usage: smoke.sh VERSION RELEASE_DIR}"
RELEASE_DIR="${2:?usage: smoke.sh VERSION RELEASE_DIR}"

cd "$RELEASE_DIR"

expected=(
  "sorrel-${VERSION}-x86_64-linux.tar.gz"
  "sorrel-${VERSION}-aarch64-linux.tar.gz"
  "sorrel-${VERSION}-x86_64-windows.zip"
  "sorrel-${VERSION}-aarch64-darwin.tar.gz"
  "sorrel-${VERSION}-x86_64.AppImage"
  "sorrel-${VERSION}.tar.gz"
)

test -s SHA256SUMS.txt
for artifact in "${expected[@]}"; do
  test -s "$artifact"
  grep -F "  $artifact" SHA256SUMS.txt >/dev/null
done

sha256sum --check SHA256SUMS.txt

tar -tzf "sorrel-${VERSION}-x86_64-linux.tar.gz" | grep -Fx './sorrel' >/dev/null
tar -tzf "sorrel-${VERSION}-aarch64-linux.tar.gz" | grep -Fx './sorrel' >/dev/null
tar -tzf "sorrel-${VERSION}-aarch64-darwin.tar.gz" | grep -Fx './sorrel' >/dev/null
unzip -Z1 "sorrel-${VERSION}-x86_64-windows.zip" | grep -Fx 'sorrel.exe' >/dev/null
tar -tzf "sorrel-${VERSION}.tar.gz" | grep -Fx "sorrel-${VERSION}/Cargo.toml" >/dev/null

file "sorrel-${VERSION}-x86_64.AppImage" | grep -Eiq 'executable|appimage|squashfs'
test -x "sorrel-${VERSION}-x86_64.AppImage"

printf 'release smoke checks passed for Sorrel %s\n' "$VERSION"
