use std::path::Path;
use std::process::Command;

fn get_python_flags() {
    let status = Command::new("python3")
        .arg(Path::new("scripts/get_python_flags.py"))
        .status()
        .expect("Failed getting Python flags");

    if !status.success() {
        panic!("Failed getting Python flags")
    }
}

fn main() {
    #[cfg(feature = "pytorch")]
    {
        get_python_flags();
    }
}
