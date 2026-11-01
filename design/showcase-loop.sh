#!/usr/bin/env bash
# Assemble assets/showcase/hero-loop.webp from the five "Showcase · loop N" PNG exports
# (openpencil export design/showcase.fig --page "Showcase · loop N" -f png -s 1.6 -o DIR/loopN.png).
#   usage: design/showcase-loop.sh DIR
# Order starts on loop 2 (the lasso lighting up the waveforms, the first paint) and ends fading back into it.
# Each state holds 1.8 s; each crossfade is 6 morph frames x 67 ms (~0.4 s).
# -kmax 1 makes every frame a keyframe: lossy delta frames leave dark block artifacts.
set -euo pipefail
dir=${1:?usage: showcase-loop.sh DIR_WITH_loop1-5.png}
out=$(cd "$(dirname "$0")/.." && pwd)/assets/showcase/hero-loop.webp
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
magick() { nix shell nixpkgs#imagemagick -c magick "$@" 2> >(grep -v '^warning:' >&2); }
order=(2 3 4 5 1)
for i in "${order[@]}"; do magick "$dir/loop$i.png" -alpha off -resize 1824x "$tmp/s$i.png"; done
args=()
for idx in 0 1 2 3 4; do
  a=${order[idx]} b=${order[(idx + 1) % 5]}
  magick "$tmp/s$a.png" "$tmp/s$b.png" -morph 6 "$tmp/m${idx}_%d.png" # m_0 = a, m_1..m_6 = blend, m_7 = b
  args+=(-d 1800 "$tmp/m${idx}_0.png")
  for k in 1 2 3 4 5 6; do args+=(-d 67 "$tmp/m${idx}_$k.png"); done
done
nix shell nixpkgs#libwebp -c img2webp -loop 0 -lossy -q 80 -m 4 -kmax 1 "${args[@]}" -o "$out" 2> >(grep -v '^warning:' >&2)
ls -la "$out"
# Phones get a still of the Waveforms pane alone (the full two-pane frame is unreadable at 350 px).
magick "$tmp/s2.png" -crop 912x1040+0+0 +repage -quality 85 "$(dirname "$out")/hero-mobile.webp"
