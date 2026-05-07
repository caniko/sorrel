# Installation

Sorrel is a Rust workspace and Nix flake.

## Build With Cargo

```bash
cargo build --release -p sorrel
```

The binary is written to `target/release/sorrel`.

## Build With Nix

```bash
nix build .#sorrel
```

The default flake package also builds Sorrel:

```bash
nix build
```

## Development Shell

```bash
nix develop
```

The development shell includes the Rust toolchain, native GUI dependencies, and the static-site tools used for this documentation.
