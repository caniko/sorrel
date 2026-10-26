// macOS extension modules leave the Python symbols for the interpreter to
// resolve (`-undefined dynamic_lookup`); maturin adds this itself, the Nix
// wheel build (nix/wheel.nix) does not.
fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
