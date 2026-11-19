#!/usr/bin/env bash
set -euo pipefail
trap 'echo "release smoke check failed at line $LINENO: $BASH_COMMAND" >&2' ERR

VERSION="${1:?usage: smoke.sh VERSION RELEASE_DIR}"
RELEASE_DIR="${2:?usage: smoke.sh VERSION RELEASE_DIR}"
NWB_FIXTURE=${SORREL_NWB_FIXTURE:-$(realpath "$(dirname "${BASH_SOURCE[0]}")/fixtures/nwb/good.nwb")}

cd "$RELEASE_DIR"

expected=(
  "sorrel-${VERSION}-x86_64-linux.tar.gz"
  "sorrel-${VERSION}-aarch64-linux.tar.gz"
  "sorrel-${VERSION}-x86_64-windows.zip"
  "sorrel-${VERSION}-x86_64-windows.exe"
  "sorrel-${VERSION}-x86_64-windows-gui.exe"
  "sorrel-${VERSION}-aarch64-windows.zip"
  "sorrel-${VERSION}-aarch64-windows.exe"
  "sorrel-${VERSION}-aarch64-windows-gui.exe"
  "sorrel-${VERSION}-aarch64-darwin.tar.gz"
  "sorrel-${VERSION}-x86_64.AppImage"
  "sorrel-${VERSION}.tar.gz"
  "sorrel_ephys-${VERSION}-cp38-abi3-manylinux_2_34_x86_64.whl"
  "sorrel_ephys-${VERSION}-cp38-abi3-manylinux_2_34_aarch64.whl"
  "sorrel_ephys-${VERSION}-cp38-abi3-win_amd64.whl"
  "sorrel_ephys-${VERSION}-cp38-abi3-macosx_11_0_arm64.whl"
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
unzip -Z1 "sorrel-${VERSION}-x86_64-windows.zip" | grep -Fx 'sorrelw.exe' >/dev/null
unzip -p "sorrel-${VERSION}-x86_64-windows.zip" sorrel.exe | cmp - "sorrel-${VERSION}-x86_64-windows.exe"
unzip -p "sorrel-${VERSION}-x86_64-windows.zip" sorrelw.exe | cmp - "sorrel-${VERSION}-x86_64-windows-gui.exe"
unzip -Z1 "sorrel-${VERSION}-aarch64-windows.zip" | grep -Fx 'sorrel.exe' >/dev/null
unzip -Z1 "sorrel-${VERSION}-aarch64-windows.zip" | grep -Fx 'sorrelw.exe' >/dev/null
unzip -p "sorrel-${VERSION}-aarch64-windows.zip" sorrel.exe | cmp - "sorrel-${VERSION}-aarch64-windows.exe"
unzip -p "sorrel-${VERSION}-aarch64-windows.zip" sorrelw.exe | cmp - "sorrel-${VERSION}-aarch64-windows-gui.exe"
tar -tzf "sorrel-${VERSION}.tar.gz" | grep -Fx "sorrel-${VERSION}/Cargo.toml" >/dev/null

for wheel in sorrel_ephys-"${VERSION}"-cp38-abi3-*.whl; do
  unzip -Z1 "$wheel" | grep -Ex 'sorrel/_native\.(abi3\.so|pyd)' >/dev/null
  unzip -Z1 "$wheel" | grep -Ex "sorrel_ephys-${VERSION}\.data/scripts/sorrel(\.exe)?" >/dev/null
done

file "sorrel-${VERSION}-x86_64.AppImage" | grep -Eiq 'executable|appimage|squashfs'
test -x "sorrel-${VERSION}-x86_64.AppImage"

# Exercise the archived executable, not a local Cargo build. `describe`
# opens the dataset and reads its units without starting a GUI or GPU.
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
describe_nwb() {
  local data
  data=$(mktemp -d "$tmp/nwb.XXXXXX")
  cp "$NWB_FIXTURE" "$data/good.nwb"
  "$@" describe "$data/good.nwb" --json > "$data/describe.json"
  grep -F '"# spikes": "5"' "$data/describe.json"
}
for arch in x86_64 aarch64; do
  tar -xOzf "sorrel-${VERSION}-${arch}-linux.tar.gz" ./sorrel > "$tmp/sorrel-$arch"
  unzip -p "sorrel_ephys-${VERSION}-cp38-abi3-manylinux_2_34_${arch}.whl" \
    "sorrel_ephys-${VERSION}.data/scripts/sorrel" | cmp - "$tmp/sorrel-$arch"
  chmod +x "$tmp/sorrel-$arch"
  if [[ $(uname -m) == "$arch" ]]; then
    describe_nwb "$tmp/sorrel-$arch"
  elif [[ $arch == aarch64 ]]; then
    describe_nwb qemu-aarch64 "${SORREL_ARM_LD:?use nix run .#release-smoke}" \
      --library-path "$SORREL_ARM_LIBS" "$tmp/sorrel-$arch"
  fi
done
unzip -p "sorrel_ephys-${VERSION}-cp38-abi3-win_amd64.whl" \
  "sorrel_ephys-${VERSION}.data/scripts/sorrel.exe" | cmp - "sorrel-${VERSION}-x86_64-windows.exe"
tar -xOzf "sorrel-${VERSION}-aarch64-darwin.tar.gz" ./sorrel > "$tmp/sorrel-darwin"
unzip -p "sorrel_ephys-${VERSION}-cp38-abi3-macosx_11_0_arm64.whl" \
  "sorrel_ephys-${VERSION}.data/scripts/sorrel" | cmp - "$tmp/sorrel-darwin"
if [[ $(uname -m) == x86_64 ]]; then
  APPIMAGE_EXTRACT_AND_RUN=1 describe_nwb "./sorrel-${VERSION}-x86_64.AppImage"
fi

printf 'release smoke checks passed for Sorrel %s\n' "$VERSION"
