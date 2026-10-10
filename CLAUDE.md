# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Rage is a general purpose chess library and UCI-compliant chess engine written in pure Rust targeting top-5 engine strength. It uses NNUE evaluation (HalfKAv2), alpha-beta search with Lazy SMP parallelism, and ships configurable model weights. See @DESIGN.md for the full specification.

## Build & Development Commands

All common tasks go through `just` (justfile):

- `just` — format, check, lint, and test (default)
- `just build` — build the project
- `just test` — run fast tests (skips `#[ignore]` tests)
- `just test-long` — run all tests including slow perft suites
- `just test <name>` — run a single test by name (passes args to `cargo test`)
- `just testv <name>` — run tests with output visible (`--no-capture`)
- `just lint` — run clippy with `-D warnings`
- `just fmt` — format code with rustfmt
- `just run` — build and run
- `just run` — build in check mode

Toolchain: Rust 1.99.0, pinned in `rust-toolchain.toml`. `nix develop` provides the toolchain plus just, wasm-pack, and wasmtime.

Before every commit, be sure to run the plain `just` command.

## Architecture

Cargo workspace with fine-grained crates under `crates/`:

- **rage-core** — chess fundamentals (bitboards, move generation, FEN/PGN, Zobrist). Zero non-std deps. Must compile to wasm. This is also the embeddable chess library.
- **rage-nnue** — NNUE inference with HalfKAv2 architecture. Quantized i16/i8 weights, runtime SIMD dispatch (AVX-512/AVX2/SSE4.1/NEON/wasm SIMD/scalar). Depends only on rage-core.
- **rage-search** — PVS with iterative deepening, Lazy SMP. Lock-free transposition table via atomics. Depends on rage-nnue and rage-syzygy.
- **rage-uci** — UCI protocol. Async stdin on dedicated thread, commands via channel.
- **rage-syzygy** — pure Rust Syzygy tablebase probing (WDL + DTZ). No-op stub on wasm.
- **rage-engine** — binary entry point. Also the wasm entry point via wasm-bindgen.
- **rage-train** — NNUE training pipeline using Burn framework (CUDA backend for H200s).
- **rage-test** — strength testing framework (SPRT, tournaments, Elo measurement with pentanomial model).

Dependency flow: `rage-engine -> rage-uci -> rage-search -> rage-nnue -> rage-core`, `rage-search -> rage-syzygy -> rage-core`.

## Code Conventions

Strict clippy lints are enforced project-wide (see `main.rs` attributes):
- `clippy::panic`, `clippy::todo`, `clippy::expect_used`, `clippy::unwrap_used` are all **denied**
- `clippy::undocumented_unsafe_blocks` is **denied** — every `unsafe` block needs a `// SAFETY:` comment
- `dead_code` is allowed during development
- All `unsafe` is confined to SIMD modules in rage-nnue; the rest of the codebase must be safe Rust

## Key Design Constraints

- **Pure Rust**: Zero C/C++ dependencies across the entire stack including training
- **Wasm-compatible**: rage-core, rage-nnue, and rage-search must compile to wasm32. No platform-specific code outside of cfg-gated SIMD modules.
- **No heap allocation in hot paths**: search loop and NNUE inference use stack-allocated structures
- **Lock-free parallelism**: transposition table uses AtomicU64 with Relaxed ordering; per-thread search stacks and history tables avoid contention
- **Well-tested**: we are very rigorous about unit, integration, and performance testing. We don't write meaningless tests. When doing performance analysis, we use sound statistical methods. Tests must be fast to run.
