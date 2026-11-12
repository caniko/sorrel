# Sorrel app icon

Green sweeping ribbons around a terracotta pulse, on a transparent background.

- [`logo/logo-clean.svg`](../../logo/logo-clean.svg) is the source;
  `sorrel.svg` is its website copy and `design/icon.fig` is the editable OpenPencil version.
- Regenerate the SVG copy, PNG sizes, and PNG-compressed ICO with Node.js and `resvg`
  from the repository root:

```sh
node assets/icon/generate.cjs
node assets/icon/generate.cjs --check
```

`crates/sorrel/assets/` holds the two files the `sorrel` crate embeds (the 256 px
PNG and a PNG-compressed `sorrel.ico` of the 16-256 px sizes), so `cargo package`
stays self-contained. The generator updates both copies; `--check` verifies every
generated asset against the source. A unit test also checks the crate's PNG copy.
