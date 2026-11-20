fn main() {
    // ponytail: hdf5-metno-sys 0.11.4 links shlwapi only for a system HDF5,
    // but the bundled static HDF5's H5system.c needs StrStrIA from it too.
    // Drop this once hdf5-metno-sys emits it on the static path.
    #[cfg(feature = "hdf5")]
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        println!("cargo:rustc-link-lib=shlwapi");
    }
    #[cfg(all(feature = "dandi", feature = "hdf5"))]
    {
        let include = std::env::var_os("DEP_HDF5_INCLUDE")
            .expect("hdf5-metno-sys must report the include directory of the linked HDF5");
        let shim = "src/nwb/imp/http_vfd.c";
        println!("cargo:rerun-if-changed={shim}");
        println!("cargo:rerun-if-env-changed=DEP_HDF5_INCLUDE");
        cc::Build::new()
            .include(include)
            .file(shim)
            .compile("sorrel_http_vfd");
    }
}
