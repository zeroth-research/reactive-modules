//! Translation of reactive modules into Lean 4.

#[cfg(test)]
mod tests {

    // The translated code imports the Lean library in [`LIB_DIR`] — the static
    // part common to every translated module.
    // Path to the Lean 4 library (a Lake package) shipped with this crate.
    const LIB_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/lib");

    use std::path::Path;

    #[test]
    fn lib_dir_is_lake_package() {
        assert!(Path::new(LIB_DIR).join("lakefile.toml").is_file());
    }
}
