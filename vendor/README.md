# Temporary zarrs 0.23.14 patch

`zarrs/` is the crates.io `zarrs-0.23.14.crate` (SHA-256
`fbd6697e73e803a80f182036563038ccfaf7fca1b71fc510cce65daaa45478f8`),
without its archive lockfile, original manifest, or Cargo's VCS metadata.
The patch drops the unused `libz-sys` build dependency. Upstream
[fixed it on main](https://github.com/zarrs/zarrs/commit/dec0b26fefcf9f12a546e775f555d3b9f424720f);
remove the path override and this directory when the fix ships on crates.io.
Until then, crates.io builds of Sorrel still resolve the upstream crate, which
builds `libz-sys` even with native codecs disabled.

The two synchronous multi-chunk paths zero-initialize their output slices
instead of using `UnsafeCellSlice::new_from_vec_with_spare_capacity`. That API
is deprecated in the workspace's resolved version, and zarrs denies the lint
when built as a local crate rather than a registry dependency.
