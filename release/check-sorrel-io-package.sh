#!/usr/bin/env bash
# Test the published graph: cargo package strips local path overrides.
set -euo pipefail
cd "$(dirname "$0")/.."
target=$(mktemp -d)
trap 'rm -rf "$target"' EXIT
cargo package --locked --no-verify -p sorrel-io --target-dir "$target" "$@"
package=("$target"/package/sorrel-io-*.crate)
tar -xOf "${package[0]}" "$(basename "${package[0]}" .crate)/Cargo.lock" > "$target/packaged.lock"
if grep -q '^name = "libz-sys"$' "$target/packaged.lock"; then
  echo 'Published sorrel-io unexpectedly depends on libz-sys' >&2
  exit 1
fi
