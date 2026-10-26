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
  "sorrel-${VERSION}-cp38-abi3-manylinux_2_34_x86_64.whl"
  "sorrel-${VERSION}-cp38-abi3-manylinux_2_34_aarch64.whl"
  "sorrel-${VERSION}-cp38-abi3-win_amd64.whl"
  "sorrel-${VERSION}-cp38-abi3-macosx_11_0_arm64.whl"
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

for wheel in sorrel-"${VERSION}"-cp38-abi3-*.whl; do
  unzip -Z1 "$wheel" | grep -Ex 'sorrel/_native\.(abi3\.so|pyd)' >/dev/null
  unzip -Z1 "$wheel" | grep -Ex "sorrel-${VERSION}\.data/scripts/sorrel(\.exe)?" >/dev/null
done

file "sorrel-${VERSION}-x86_64.AppImage" | grep -Eiq 'executable|appimage|squashfs'
test -x "sorrel-${VERSION}-x86_64.AppImage"

printf 'release smoke checks passed for Sorrel %s\n' "$VERSION"
