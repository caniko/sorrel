# Release Process

Every channel publishes from simit-generated GitHub Actions when a signed
`<version>` tag (for example `0.2.1`) is pushed on trunk. Nothing is
published by hand.

1. Bump the workspace version in `Cargo.toml` and match it in
   `crates/sorrel-py/Cargo.toml` and `crates/sorrel-py/pyproject.toml`
   (`nix/wheel.nix` checks all three). Move `[Unreleased]` to
   `## [<version>] - YYYY-MM-DD` in `CHANGELOG.md`, set `version` and
   `date-released` in `CITATION.cff` to match, and regenerate the BibTeX in
   `docs/src/project/citing.md` (commands there). Leave website download links
   unchanged: GitHub Pages deploys on trunk pushes, before the release assets
   are uploaded. Land the release commit on trunk with green CI, prebuild and
   both Target smokes (`windows-2025` for x64, `windows-11-arm` for ARM64).
2. Push a signed annotated tag on that commit. Release validation needs
   the CHANGELOG heading and `nix eval .#sorrel.version` to match the tag.
3. After the GitHub Release workflow finishes and all expected assets exist,
   update the tag and asset filenames in the download URLs in
   `website/plinth-project.toml` in a short follow-up commit. Check the links
   against the published release, then run
   `nix eval .#site.drvPath --no-update-lock-file`. If assets change, update the
   download rows and installation guide together. Plinth allows four primary
   routes; put additional routes under `secondary_routes`.

The tag starts:

- **`release.yml`**: builds every artifact, runs `release/smoke.sh`,
  signs and attests, publishes the GitHub release, then uploads to PyPI
  (`sorrel-ephys`, trusted publishing), Chocolatey (`sorrel`) and winget
  (`Caniko.Sorrel`, a PR to microsoft/winget-pkgs). Windows assets are
  `sorrel-<version>-x86_64-windows.zip` and
  `sorrel-<version>-aarch64-windows.zip`, each with standalone `.exe` and
  `-gui.exe` files. winget lists both architectures; Chocolatey and the
  Python wheel stay x64.
- **`publish-workspace.yaml`**: verifies and packages the workspace crates
  in dependency order. **`trusted-publish-workspace.yaml`** then uploads
  them to crates.io from the `simit-crates-io` environment (restricted to
  trunk, holding `CRATES_IO_API_TOKEN`, synced by canix) using the simit
  revision in `vars.SIMIT_PUBLISHER_REVISION`.

The ARM64 Target smoke runs `release/smoke-windows.ps1 -SkipWheel`. It covers
DX12 but not Vulkan, because the pinned lavapipe is x64-only; the x64 run
covers both.

The CLI built from trunk for archives, installers and wheels bundles static
HDF5 with gzip/deflate and LZF. It does not need a system HDF5 library. See the
[bundled library notices](crates/sorrel/THIRD-PARTY-HDF5.txt). The wheel's
Python `Session.open()` still supports phy and SortingAnalyzer only.

Regenerate these workflows with simit, never by hand:

```sh
nix run 'git+https://github.com/caniko/simit.git?rev=<rev>' -- init ci \
  --platform github --ci-provider actions --runtime nix --workspace \
  --publish-crates --coordinated-publish
nix run 'git+https://github.com/caniko/simit.git?rev=<rev>' -- init release --platform github
```

The publication order comes from `simit release plan --json`:
`sorrel-cache`, `sorrel-io`, `sorrel-compute`, `sorrel-data`, `sorrel-gpu`,
`sorrel-remote`, `sorrel-render`, `sorrel-ui`, `sorrel`.

## Toolchain bumps

A harbor-rs or nixpkgs bump can add a Rust toolchain that GitHub's runners
cannot build without exhausting their disk (the `windows-aarch64` llvm-mingw
bootstrap did). Prebuild the new toolchains on an internal builder, push them
to the canix Attic cache, and rerun the failed job; the runner then
substitutes them instead of compiling from source.
