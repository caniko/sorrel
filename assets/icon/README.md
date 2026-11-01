# Sorrel app icon

A sorrel (_Rumex_) leaf whose midrib is an extracellular spike.

- `sorrel.svg` is the source; `design/icon.fig` holds the same mark for OpenPencil.
- `sorrel-<N>.png` and the multi-size `sorrel.ico` are generated from it:

```sh
for s in 16 32 48 64 128 256 512; do resvg -w $s -h $s sorrel.svg sorrel-$s.png; done
magick sorrel-512.png -define icon:auto-resize=256,64,48,40,32,24,20,16 sorrel.ico
```
