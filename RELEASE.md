# Release Process

Every channel publishes from simit-generated GitHub Actions when a signed
`<version>` tag (for example `0.2.1`) is pushed on trunk. Nothing is
published by hand.

1. Bump `[workspace.package] version`, move `[Unreleased]` to
   `## [<version>] - YYYY-MM-DD` in `CHANGELOG.md`, and land it on trunk
   with green CI, prebuild and Target smokes (the real-Windows run).
2. Push a signed annotated tag on that commit. Release validation needs
   the CHANGELOG heading and `nix eval .#sorrel.version` to match the tag.

The tag starts:

- **`release.yml`**: builds every artifact, runs `release/smoke.sh`,
  signs and attests, publishes the GitHub release, then uploads to PyPI
  (`sorrel-ephys`, trusted publishing), Chocolatey (`sorrel`) and winget
  (`Caniko.Sorrel`, a PR to microsoft/winget-pkgs).
- **`publish-workspace.yaml`**: verifies and packages the workspace crates
  in dependency order. **`trusted-publish-workspace.yaml`** then uploads
  them to crates.io from the `simit-crates-io` environment (restricted to
  trunk, holding `CRATES_IO_API_TOKEN`, synced by canix) using the simit
  revision in `vars.SIMIT_PUBLISHER_REVISION`.

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
