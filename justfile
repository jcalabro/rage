# Formats, lints, and runs all tests
default: fmt check lint test

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

check *ARGS:
    cargo check -- {{ARGS}}

# Runs fast tests (skips #[ignore] tests)
test *ARGS:
    cargo test {{ARGS}}

# Runs all tests including slow ones
test-long *ARGS:
    PROPTEST_CASES=2048 cargo test {{ARGS}} -- --include-ignored

# Runs all tests with verbose logging
testv *ARGS:
    cargo test -- --no-capture {{ARGS}}

# Builds rage
build *ARGS:
    cargo build {{ARGS}}

# Builds and runs rage
run *ARGS:
    cargo run --bin rage {{ARGS}}

# Runs the benchmark (fixed-depth search over standard positions)
bench *ARGS:
    cargo run --release --bin rage -- bench {{ARGS}}
