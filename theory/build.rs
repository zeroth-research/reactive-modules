fn main() {
    // On macOS, dyld needs the rpath baked into binaries to find libtorch*.dylib at runtime.
    // torch-sys only emits rustc-link-search (link-time), not -rpath (runtime), so test
    // binaries built from this crate would fail to load without this.
    #[cfg(all(target_os = "macos", feature = "torch"))]
    {
        let output = std::process::Command::new("python3")
            .args([
                "-c",
                "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')",
            ])
            .output();

        match output {
            Ok(o) if o.status.success() => {
                let torch_lib = String::from_utf8(o.stdout).unwrap();
                let torch_lib = torch_lib.trim();
                println!("cargo:rustc-link-arg=-Wl,-rpath,{torch_lib}");
            }
            _ => {
                println!(
                    "cargo:warning=theory build.rs: could not find torch lib via python3; libtorch may not load at runtime"
                );
            }
        }

        println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");
    }
}
