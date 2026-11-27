# Contributing to Sorrel

Use [GitHub issues](https://github.com/caniko/sorrel/issues) for bugs, feature requests and data compatibility reports. Include the Sorrel version or commit, OS, backend and the smallest public or synthetic example that shows the problem. Do not upload private recordings or credentials. See [SECURITY.md](SECURITY.md) for vulnerabilities.

## Build and check a change

Run these commands from the repository root. The default Nix shell supplies the project toolchain and native dependencies:

```sh
nix develop
cargo build --release -p sorrel
cargo test --workspace
cargo clippy --workspace --all-targets -- --deny warnings
```

The [CI workflow](.github/workflows/ci.yaml) also tests and lints with default features disabled. Run these when changing feature gates or input backends:

```sh
cargo test --workspace --no-default-features
cargo clippy --workspace --all-targets --no-default-features -- --deny warnings
```

Run `nix fmt` from the repository root to apply the flake's formatter. The docs live in `docs/src/`; run `mdbook build docs` in the Nix shell, as the [docs derivation](nix/site.nix) does. Use `mdbook serve docs` to preview them locally.

Open a pull request against `trunk`. Describe the problem, what changed and which checks you ran. For behavior changes, include a regression test where possible. Keep statements about the published release separate from changes on `trunk`.
