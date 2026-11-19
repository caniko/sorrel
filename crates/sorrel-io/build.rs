fn main() {
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
