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

# --- Strength testing (rage-test) ---

# SPRT test: just sprt ./new-engine ./old-engine
# Accepts optional flags: --tc "8+0.08" --depth 5 --elo0 0 --elo1 5 --concurrency 4
sprt engine-a engine-b *ARGS:
    cargo run --release --bin rage-test -- sprt --engine-a {{engine-a}} --engine-b {{engine-b}} {{ARGS}}

# Fixed-game Elo test: just elo ./new-engine ./old-engine --pairs 50
elo engine-a engine-b *ARGS:
    cargo run --release --bin rage-test -- elo --engine-a {{engine-a}} --engine-b {{engine-b}} {{ARGS}}

# Quick self-play test (same engine, depth 5, 10 pairs — sanity check)
selfplay *ARGS:
    cargo build --release --bin rage
    cargo run --release --bin rage-test -- elo --engine-a ./target/release/rage --engine-b ./target/release/rage --depth 5 --pairs 10 {{ARGS}}

# Round-robin tournament: just tournament ./engine1 ./engine2 ./engine3
tournament +ENGINES:
    cargo run --release --bin rage-test -- tournament --engine {{ENGINES}}

# Compare bench output between two binaries
bench-diff engine-a engine-b *ARGS:
    cargo run --release --bin rage-test -- bench-diff --engine-a {{engine-a}} --engine-b {{engine-b}} {{ARGS}}

# Show test result history
history *ARGS:
    cargo run --release --bin rage-test -- history {{ARGS}}
