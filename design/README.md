# Design files

- `sorrel.fig`: the OpenPencil design file.
  - **Design pages.** `Design · Lasso tool` and `Design · Feature view v2` were made before the code. `feature-view-v2.js` rebuilds the second page.
  - **Implemented pages.** `Implemented · <state>` holds the dark-theme render of each Features, cross-talk and pane UI state. These are the states with a design counterpart, so a parity check can compare the two side by side.
- `icon.fig`: the app icon.

Every other UI state is left out to keep the file small, and so is the light theme. Regenerate any of them from the UI tests. Each kittest `snap` writes a Figma-plugin script per state and theme:

```sh
nix develop . --no-update-lock-file --command sh -c \
  'SORREL_DESIGN_DIR=/tmp/ingest cargo test -p sorrel-ui --test ui_states'
# Load into an open OpenPencil tab (replaces the page's frame of that name):
openpencil eval --document-id <tab> --stdin < /tmp/ingest/<state>_<theme>.js
# Or headlessly into a copy of the file:
openpencil eval design/sorrel.fig --stdin -w < /tmp/ingest/<state>_<theme>.js
```

Rebuild the `Design · Feature view v2` page from its generator with:

```sh
openpencil eval --document-id <tab> --page-id <page> --stdin < design/feature-view-v2.js
```
