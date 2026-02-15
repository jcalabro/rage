# rage

[![CI](https://github.com/jcalabro/rage/actions/workflows/ci.yml/badge.svg)](https://github.com/jcalabro/rage/actions/workflows/ci.yml)

Rage AI Generated Engine

A chess engine and library written in pure Rust. UCI-compliant, runs natively and in the browser via wasm.

Uses NNUE evaluation (HalfKAv2 architecture), PVS search with Lazy SMP, and includes built-in Syzygy tablebase support for up to 7-piece endgames.

## Building

Requires Rust 1.93.1 and [just](https://github.com/casey/just).

```
just install-tools
just build
just --list # see all available commands
```

## Usage

Run the engine (enters UCI mode):

```
just run
```

Run a quick benchmark:

```
just bench
```

## Development

`just` with no arguments formats, checks, lints, and tests everything.

Run a specific test:

```
just test test_name
```

Run all tests including slow perft suites:

```
just test-long
```

## Strength testing

rage ships with a built-in testing framework for engine-vs-engine matches:

```
just sprt ./new-engine ./old-engine
just elo ./new-engine ./old-engine --pairs 100
just selfplay
```

## Crate layout

```
crates/
  rage-core      chess library (bitboards, move generation, FEN, Zobrist)
  rage-nnue      NNUE inference with SIMD dispatch
  rage-search    PVS + iterative deepening + Lazy SMP
  rage-uci       UCI protocol
  rage-syzygy    Syzygy tablebase probing
  rage-engine    binary and wasm entry point
  rage-train     NNUE training pipeline (Burn + CUDA)
  rage-test      SPRT, Elo, and tournament runner
```
