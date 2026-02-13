# Formats, lints, and runs all tests
default: fmt lint test

# First, install rustup with: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`, then run this
install-tools:
    rustup install 1.93.1
    rustup override set 1.93.1
    rustup component add clippy rust-analyzer

# Runs the linter
lint:
    cargo clippy --all -- -D warnings

# Formats all code
fmt *ARGS:
    cargo fmt -- {{ARGS}}

# Runs fast tests (skips #[ignore] tests)
test *ARGS:
    cargo test {{ARGS}}

# Runs all tests including slow perft suites
test-long *ARGS:
    cargo test {{ARGS}} -- --include-ignored

# Runs all tests with verbose logging
testv *ARGS:
    cargo test -- --no-capture {{ARGS}}

# Builds rage
build *ARGS:
    cargo build {{ARGS}}

# Builds and runs rage
run *ARGS:
    cargo run {{ARGS}}
