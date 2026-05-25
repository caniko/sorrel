# Release Process

## First Publish Order

`simit release plan` is not available in the simit CLI currently resolved by
this repository, so the first-publish order is computed from `cargo metadata`
local path dependencies.

Publish the `0.1.0` crates in this order:

1. `sorrel-io`
2. `sorrel-cache`
3. `sorrel-compute`
4. `sorrel-gpu`
5. `sorrel-data`
6. `sorrel-render`
7. `sorrel-ui`
8. `sorrel`

This order is required because downstream crates depend on earlier local
workspace crates by versioned path dependency. For first publish, run
`cargo package -p <crate>` for every crate, but expect
`cargo publish --dry-run -p <downstream>` to fail until each dependency earlier
in the sequence is live on crates.io.

## Name Reservation

Before publishing, confirm each crate name is still available with:

```sh
cargo search <crate-name> --limit 1
```

Do not rename crates automatically if a name is taken; stop and choose a
project-level naming policy first.
