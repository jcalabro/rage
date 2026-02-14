fn main() {
    println!(
        "cargo::rustc-env=TARGET={}",
        std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string())
    );
    println!(
        "cargo::rustc-env=RUSTC_VERSION={}",
        std::env::var("RUSTC")
            .ok()
            .and_then(|rustc| {
                std::process::Command::new(rustc)
                    .arg("--version")
                    .output()
                    .ok()
                    .and_then(|o| String::from_utf8(o.stdout).ok())
            })
            .map(|v| v.trim().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    );
}
