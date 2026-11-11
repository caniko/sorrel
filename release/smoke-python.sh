#!/usr/bin/env bash
# Linux CI for sorrel-py: install the release wheel (downloaded to
# $SIMIT_TARGET_SMOKE_DIR) with its nwb and neuroconv extras into a fresh
# venv and run crates/sorrel-py/python/tests from the repo root.
#
#   SIMIT_TARGET_SMOKE_DIR=<dir with the .whl> release/smoke-python.sh
set -euo pipefail
dir=${SIMIT_TARGET_SMOKE_DIR:?set SIMIT_TARGET_SMOKE_DIR to the directory holding the wheel}
wheels=("$dir"/sorrel_ephys-*-manylinux*_x86_64.whl)
[ -f "${wheels[0]}" ] || { echo "no sorrel_ephys manylinux wheel in $dir" >&2; exit 1; }
command -v uv >/dev/null || pipx install uv
export PATH="$HOME/.local/bin:$PATH"
venv=$(mktemp -d)/venv
uv venv --python 3.12 "$venv"
# hdmf-zarr<0.14: neuroconv 0.10.2 still uses the zarr 2 API.
uv pip install --python "$venv/bin/python" "${wheels[0]}[nwb,neuroconv]" \
  "neuroconv==0.10.2" "hdmf-zarr<0.14" nwbinspector spikeinterface scipy pytest
"$venv/bin/python" -m pytest crates/sorrel-py/python/tests -q -rA
