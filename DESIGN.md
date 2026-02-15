# Rage Chess Engine - Design Specification

## Overview

Rage is a UCI-compliant chess engine written in pure Rust, targeting top-5 engine
strength (TCEC/CCC competitive). It uses NNUE evaluation with HalfKAv2-style
architecture, alpha-beta search with Lazy SMP parallelism, and ships configurable
model weights for different use cases (full-strength desktop, lightweight wasm).

## Goals

- **Strength**: Top-5 engine. Competitive with Stockfish, Torch, Berserk, etc.
- **Pure Rust**: Zero C/C++ dependencies. Entire stack from training to inference.
- **Performance**: Maximize NPS on modern CPUs via SIMD, cache-friendly data
  structures, and lock-free parallelism.
- **Embeddable**: First-class library API for chess primitives and engine analysis.
- **Portable**: Native binaries (x86-64, ARM64) + wasm for browser deployment.
- **Quality**: Extensively tested. Correctness is non-negotiable.

## Non-Goals (for v1)

- Chess960/Fischer Random (can be added later but not in initial scope)
- Distributed search across multiple machines
- Android/iOS native builds (wasm covers mobile browsers)
- GUI - rage is an engine, not a chess client

---

## Architecture

### Crate Layout

Cargo workspace with fine-grained crates. Each crate has a clear responsibility and
minimal coupling to others.

```
rage/
  Cargo.toml              (workspace root)
  justfile
  DESIGN.md

  crates/
    rage-core/            Chess fundamentals
    rage-nnue/            NNUE inference
    rage-search/          Search algorithm + threading
    rage-uci/             UCI protocol
    rage-syzygy/          Syzygy tablebase probing
    rage-engine/          Binary entry point
    rage-train/           GPU training pipeline (Burn)

  models/                 Trained model weights
    default.nnue          Full-strength weights
    lite.nnue             Lightweight weights for wasm/embedded

  testdata/               Test positions, EPD suites, PGN files

  crates/
    rage-test/            Strength testing framework (tournament runner + SPRT)
```

**Dependency graph** (arrows = "depends on"):

```
rage-engine
  -> rage-uci -> rage-search -> rage-nnue -> rage-core
                             -> rage-syzygy -> rage-core
  -> rage-core (for CLI utilities)

rage-train
  -> rage-core (for position/feature generation)
  -> burn (GPU compute)

rage-test
  -> rage-core (for PGN parsing, position handling)
  -> rage-engine (runs engine binaries as subprocesses)
```

### Crate Responsibilities

#### `rage-core`
The foundation. Zero non-std dependencies (except `serde` behind a feature flag for
serialization). Must compile to wasm without issue.

- **Board representation**: Bitboard-based. 8x `u64` bitboards (one per piece type +
  two for color occupancy). Mailbox redundancy array for fast piece-on-square lookups.
- **Move generation**: Legal move generation using magic bitboards for sliding pieces.
  Staged move generation (captures first, then quiets) to support move ordering in
  search. Separate bulk/perft mode for testing.
- **Position state**: Zobrist hashing (incrementally updated), castling rights,
  en passant square, halfmove clock, fullmove counter. Position uses a `Copy`-friendly
  make/unmake approach with an undo stack.
- **Move encoding**: 16-bit compact move encoding (6 bits from, 6 bits to, 4 bits
  flags for promotion piece, castling, en passant, double pawn push).
- **PGN parsing**: Full PGN reader/writer including annotations, variations, NAGs.
  Streaming parser for large PGN files.
- **FEN**: Parse and serialize FEN strings.
- **Notation**: SAN (Standard Algebraic Notation) and UCI long algebraic notation
  conversion.
- **Utilities**: Square, File, Rank, Color, Piece, CastlingRights types. All with
  exhaustive `From`/`Into` conversions.
- **Public API**: This crate IS the embeddable chess library. Its public API should be
  clean, documented, and stable.

#### `rage-nnue`
NNUE evaluation. Depends only on `rage-core`.

- **Architecture**: HalfKAv2 (Half King-All version 2).
  - Input features: 768 per perspective (64 squares x 6 piece types x 2 colors).
  - King buckets: 10 buckets mapping the king's square to a feature transformer group
    (mirrors across the vertical axis, so 32 effective king squares map to 10 buckets).
  - Feature transformer: Sparse input -> dense accumulator (size configurable,
    default 1024 for full-strength, 512 or 256 for lite).
  - Network body: Accumulator -> hidden layers -> single output.
    Full architecture: `features -> 1024 -> 8 -> 32 -> 1` (subject to tuning).
  - Activation: Clipped ReLU (CReLU) with integer quantization.
- **Incremental updates**: The accumulator is incrementally updated on make/unmake.
  Only changed features (piece additions/removals) trigger updates. King bucket
  changes trigger a full refresh of the affected perspective.
- **Quantization**: Weights are quantized to `i16` (accumulator) and `i8` (later
  layers) for fast integer SIMD inference. No floating point in the hot path.
- **SIMD dispatch**: Runtime CPU feature detection with tiered implementations:
  1. AVX-512 (512-bit, 32x i16 or 64x i8 per op)
  2. AVX2 (256-bit, 16x i16 or 32x i8)
  3. SSE4.1 (128-bit, fallback for older x86)
  4. NEON (128-bit, ARM64)
  5. wasm SIMD (128-bit)
  6. Scalar (portable fallback)

  Detection happens once at startup. Function pointers are set to the best available
  implementation. For wasm, the SIMD tier is selected at compile time.
- **Weight loading**: Custom binary format for quantized weights. Versioned header
  with architecture metadata. Weights can be embedded in the binary at compile time
  (via `include_bytes!`) or loaded from a file at runtime.
- **Weight profiles**:
  - `default.nnue`: Full-strength, ~40-80 MB. Uses 1024-wide accumulator.
  - `lite.nnue`: Lightweight, ~2-5 MB. Uses 256-wide accumulator. Targets wasm and
    resource-constrained environments. Trades ~100-200 Elo for dramatically smaller
    size and faster inference.

#### `rage-search`
The search algorithm. Core of the engine's playing strength.

- **Framework**: Principal Variation Search (PVS) with iterative deepening and
  aspiration windows.
- **Parallelism**: Lazy SMP. Multiple threads share a transposition table and search
  the same root position independently with slight depth/move-order perturbations.
  Lock-free concurrent transposition table using atomic operations.
- **Transposition table**: Lock-free hash table using `AtomicU64` entries. Each entry
  stores: hash verification bits, best move, score, depth, bound type (exact/lower/
  upper), age. Bucket-based (4 entries per bucket) with aging replacement scheme.
- **Move ordering** (critical for pruning efficiency):
  1. TT move (from transposition table)
  2. Winning captures (MVV-LVA + SEE filtering)
  3. Killer moves (2 per ply)
  4. Counter moves
  5. History heuristic (butterfly boards + continuation history)
  6. Losing captures
  7. Remaining quiet moves (sorted by history score)
- **Pruning and reductions**:
  - Null move pruning (NMP) with verification search
  - Late move reductions (LMR) with logarithmic reduction table
  - Late move pruning (LMP)
  - Futility pruning (static + extended)
  - Reverse futility pruning / static null move pruning
  - Delta pruning (in quiescence search)
  - Singular extensions (for TT moves)
  - Multi-cut pruning
  - Razoring
  - SEE (Static Exchange Evaluation) pruning for captures and quiet moves
  - Probcut
- **Quiescence search**: Searches captures (and checks at low depth) to resolve
  tactical sequences and avoid horizon effects.
- **Time management**: Adaptive time allocation based on:
  - Time control type (sudden death, increment, moves-to-go)
  - Score stability across iterations
  - Best move stability (if best move keeps changing, allocate more time)
  - Fraction of remaining time with safety margins
- **Search stack**: Per-ply stack allocated search state (killers, excluded moves,
  static eval, continuation history pointers) to avoid heap allocation in the hot
  loop.
- **Thread pool**: Fixed-size thread pool created at startup. Threads are woken via
  condition variables when a new search begins. Each thread has its own search stack
  and history tables; only the TT is shared.
- **Pondering**: Support for `go ponder` - search continues on the expected opponent
  move. If `ponderhit` arrives, seamlessly transition to normal search with
  accumulated time.
- **Reporting**: Periodic `info` output with depth, seldepth, score, nodes, nps,
  hashfull, PV line, tbhits.

#### `rage-syzygy`
Pure Rust Syzygy tablebase probing.

- **Format support**: WDL (Win/Draw/Loss) and DTZ (Distance To Zeroing) tables.
  Support for up to 7-piece tablebases.
- **Probing**: Used in search for:
  - Root filtering: At the root, filter moves to only winning/drawing moves when in
    a tablebase position.
  - Search cutoffs: In the search tree, use WDL probes to produce exact scores when
    the position is in the tablebase.
- **Decompression**: Syzygy tables use a custom compression scheme involving Huffman
  coding and permutation-based indexing. All decompression implemented in pure Rust.
- **Caching**: LRU cache for decompressed table blocks to avoid repeated disk I/O.
- **Memory mapping**: Use memory-mapped files for table access where available (not
  wasm). Fall back to buffered reads on platforms without mmap.
- **Wasm**: Syzygy probing is NOT available in the wasm build (tables are too large).
  The crate compiles to a no-op stub on wasm targets.

#### `rage-uci`
UCI (Universal Chess Interface) protocol implementation.

- **Full UCI compliance**: All standard UCI commands (uci, isready, position, go,
  stop, quit, ponderhit, setoption, ucinewgame, register, debug).
- **Custom options**:
  - `Hash` (MB): Transposition table size
  - `Threads`: Number of search threads
  - `SyzygyPath`: Path to Syzygy tablebases
  - `SyzygyProbeDepth`: Minimum depth for tablebase probing
  - `SyzygyProbeLimit`: Maximum number of pieces for probing
  - `MoveOverhead`: Time buffer for communication lag (ms)
  - `EvalFile`: Path to NNUE weight file
  - `UCI_ShowWDL`: Show win/draw/loss probabilities
- **Async I/O**: stdin reading on a dedicated thread. Commands are sent to the engine
  via a channel. The engine can be stopped asynchronously (atomic `stop` flag checked
  in the search loop).
- **Multi-PV**: Support for `MultiPV` option showing top N lines.

#### `rage-engine`
The binary. Thin wrapper that wires together all crates.

- Parses CLI arguments (if any - e.g. `rage bench`, `rage perft <depth>`)
- Initializes magic bitboard tables, Zobrist keys, NNUE weights
- Enters UCI loop
- Wasm entry point: exports a JS-friendly API (via `wasm-bindgen`) wrapping
  `rage-core` and `rage-search` with the lite NNUE model.

#### `rage-test`
Strength testing framework. Rage's equivalent of Stockfish's fishtest or OpenBench.
A self-contained tool for running engine-vs-engine tournaments with rigorous
statistical analysis.

- **Game manager**: Spawns UCI engine processes, manages concurrent games, handles
  timeouts and crashes gracefully. Plays games from a configurable book of opening
  positions (EPD/FEN file) to reduce variance.
- **SPRT (Sequential Probability Ratio Test)**: The primary testing mode. Tests
  whether a patch gains Elo with statistical significance. Supports configurable
  bounds (e.g., `H0: Elo <= 0, H1: Elo >= 5`) and confidence levels (typically
  α=0.05, β=0.05). Stops as soon as the test is conclusive - no wasted games.
- **Fixed-game tournaments**: Run N games and report Elo difference with confidence
  intervals. Useful for absolute strength measurement against reference engines.
- **Pentanomial model**: Uses the pentanomial (game-pair) model for Elo estimation
  rather than the simpler trinomial model. Pentanomial accounts for paired openings
  (each opening played once as white, once as black) and produces tighter confidence
  intervals.
- **Concurrency**: Runs many games in parallel across CPU cores. Configurable number
  of concurrent games.
- **Time controls**: Supports all standard time controls (sudden death, increment,
  moves-to-go). Default test TC: `8+0.08` (fast games for quick iteration) and
  `60+0.6` (longer TC for validation).
- **Opening books**: Ships with a curated set of balanced opening positions (drawn
  from top-level games). Openings are assigned in pairs (same opening for both
  colors) to reduce first-move advantage noise.
- **Engine management**: Can test any UCI engine, not just rage. Useful for comparing
  against Stockfish, Torch, etc. Engine paths and UCI options are configurable.
- **Results database**: Stores all test results in a local SQLite database. Query
  historical results, compare patches over time, track Elo progression.
- **CLI UX**: Clean, informative command-line interface.
  ```
  $ rage-test sprt \
      --engine-a ./rage-new --engine-b ./rage-old \
      --tc "8+0.08" --threads 4 \
      --elo0 0 --elo1 5 \
      --concurrency 16 \
      --book openings.epd

  rage-test v0.1.0 — SPRT Test
  ─────────────────────────────────────────────────────────
  Engine A: ./rage-new (options: Threads=1, Hash=16)
  Engine B: ./rage-old (options: Threads=1, Hash=16)
  Time control: 8+0.08s
  SPRT bounds: H0: Elo ≤ 0, H1: Elo ≥ 5 (α=0.05, β=0.05)
  Opening book: openings.epd (1000 positions)
  Concurrency: 16 games
  ─────────────────────────────────────────────────────────

  Games:  2048 / ???    W: 612  D: 824  L: 612
  Elo: +3.2 ± 4.1 (95%)
  LLR: 1.82 [-2.94, 2.94]  ████████████░░░░  (in progress)
  Speed: ~120 games/min

  [Live updating, Ctrl+C to abort cleanly]
  ```
- **Reports**: After completion, print a clear summary:
  ```
  ─────────────────────────────────────────────────────────
  Result: PASSED (H1 accepted)
  Games: 3,412 pairs  |  W: 1,024  D: 1,402  L: 986
  Elo: +4.8 ± 3.1 (95%)   LOS: 99.7%
  LLR: 2.95 ≥ 2.94 (upper bound reached)
  Pentanomial: [52, 284, 892, 301, 47]
  Duration: 28m 14s  |  ~121 games/min
  ─────────────────────────────────────────────────────────
  ```
- **Subcommands**:
  - `rage-test sprt`: Run SPRT test between two engines
  - `rage-test tournament`: Round-robin or gauntlet tournament
  - `rage-test elo`: Fixed-game match with Elo + confidence interval
  - `rage-test history`: Query past test results from the database
  - `rage-test bench-diff`: Compare `rage bench` output between two binaries
    (quick sanity check that search behavior changed as expected)

#### `rage-train`
NNUE training pipeline. Uses the Burn deep learning framework.

- **Data generation**: Self-play games using the engine at low-to-medium depth to
  generate training positions. Output format: packed binary with position, search
  score, and game outcome (WDL).
- **Data format**: Custom compact binary format:
  - Board state in a packed representation (~32 bytes)
  - Search score (i16)
  - Game result (WDL, 2 bits)
  - Ply number (u16)
  - Move played (u16)
- **Training loop**: Supervised learning on (position, score) pairs.
  - Loss function: Combination of MSE on eval score and cross-entropy on WDL outcome.
    `loss = λ * MSE(sigmoid(predicted), sigmoid(target)) + (1-λ) * CE(predicted_wdl, actual_wdl)`
  - Lambda scheduling: Start with higher weight on search scores, gradually shift
    toward game outcomes.
  - Learning rate: Cosine annealing with warm restarts.
  - Batch size: Large batches (16384+) to utilize H200 throughput.
- **Quantization**: After training in f32, quantize to i16/i8 for inference. Include
  quantization-aware training (QAT) in later iterations.
- **Export**: Convert trained Burn model to the custom binary weight format consumed
  by `rage-nnue`.
- **GPU backend**: Burn with CUDA backend targeting H200s. Data loading pipeline with
  parallel workers to keep GPU saturated.

---

## Data Structures

### Board Representation

```
Bitboards (8 x u64):
  white_pawns, white_knights, white_bishops, white_rooks, white_queens, white_king
  (same for black, or indexed by color)
  white_occupancy, black_occupancy (derived, kept in sync)
  all_occupancy (derived)

Mailbox (64 x Option<(Color, PieceType)>):
  Redundant square-centric lookup for fast "what piece is on square X?" queries.
  Critical for NNUE feature computation and SEE.
```

### Move Encoding (16 bits)

```
Bits 0-5:   From square (0-63)
Bits 6-11:  To square (0-63)
Bits 12-13: Promotion piece (Knight=0, Bishop=1, Rook=2, Queen=3)
Bits 14-15: Flags (0=normal, 1=promotion, 2=en passant, 3=castling)
```

### Transposition Table Entry (16 bytes per entry, 4 per bucket = 64 bytes/bucket)

```
Bits 0-15:  Verification hash (upper 16 bits of Zobrist key)
Bits 16-31: Best move (16-bit encoded)
Bits 32-47: Score (i16)
Bits 48-55: Depth (u8)
Bits 56-57: Bound type (Exact, LowerBound, UpperBound)
Bits 58-63: Age (generation counter for replacement)
```

Stored as two `AtomicU64` values per entry. Written/read with `Relaxed` ordering
(acceptable to read stale/torn data - search correctness does not depend on TT
consistency, only efficiency).

### NNUE Accumulator

```
struct Accumulator {
    white: [i16; HIDDEN_SIZE],  // e.g., [i16; 1024]
    black: [i16; HIDDEN_SIZE],
    computed: [bool; 2],        // whether each perspective is up-to-date
}
```

Stored per-ply in the search stack. On make, the accumulator is incrementally updated
(add/subtract feature weight columns). On king bucket change, marked dirty and lazily
recomputed.

---

## SIMD Strategy

All SIMD code lives in `rage-nnue` and is organized as trait implementations with
compile-time and runtime dispatch.

```rust
trait SimdOps {
    fn accumulator_add(acc: &mut [i16], weights: &[i16], offset: usize);
    fn accumulator_sub(acc: &mut [i16], weights: &[i16], offset: usize);
    fn crelu_and_flatten(white: &[i16], black: &[i16], output: &mut [i8]);
    fn linear_i8(input: &[i8], weights: &[i8], biases: &[i32], output: &mut [i32]);
    fn linear_i32(input: &[i32], weights: &[i8], biases: &[i32], output: &mut [i32]);
}
```

At startup, detect CPU features and set function pointers:

```rust
static OPS: OnceLock<&'static dyn SimdOps> = OnceLock::new();

fn init_simd() {
    let ops: &dyn SimdOps = if is_x86_feature_detected!("avx512f") {
        &Avx512Ops
    } else if is_x86_feature_detected!("avx2") {
        &Avx2Ops
    } else if is_x86_feature_detected!("sse4.1") {
        &Sse41Ops
    } else {
        &ScalarOps
    };
    OPS.set(ops).ok();
}
```

For wasm, compile-time selection via `#[cfg(target_arch = "wasm32")]`.

SIMD implementations use Rust's `core::arch` intrinsics (stable for x86 SSE/AVX,
nightly for some AVX-512 intrinsics - track stabilization). These are `unsafe` but
contained within the SIMD module with thorough testing.

---

## Threading Model

### Lazy SMP

Multiple threads search the same position with the same TT but with small
perturbations that cause them to explore different parts of the tree:

- Each thread has a unique thread ID that slightly perturbs search depth at the root
  (thread 0 searches depth D, thread 1 searches D+1, thread 2 searches D, etc.)
- All threads share the same lock-free transposition table
- Each thread has its own: search stack, move history tables, killer moves, PV table
- The main thread (thread 0) is the only one that reports info strings and makes the
  final move decision

### Thread Communication

```
Main thread:
  - Owns the UCI loop (or receives commands via channel)
  - Starts/stops searches by setting atomic flags
  - Reads best move from thread 0 when search completes

Worker threads:
  - Blocked on a condvar when idle
  - Woken when search starts
  - Check atomic `stop` flag at every node
  - Write results to shared TT
  - Thread 0 writes its best move to a shared atomic when done
```

### Data Sharing

| Data | Sharing | Synchronization |
|------|---------|-----------------|
| Transposition table | Shared | AtomicU64 (lock-free) |
| Stop flag | Shared | AtomicBool |
| Node counter | Shared | AtomicU64 (per-thread, summed on read) |
| Search stack | Per-thread | None needed |
| History tables | Per-thread | None needed |
| Killers/Counters | Per-thread | None needed |
| NNUE accumulator | Per-thread (in stack) | None needed |
| Syzygy cache | Shared | RwLock (coarse, infrequent) |

---

## Wasm Build

The wasm build is a first-class target, not an afterthought.

### What's included in wasm
- `rage-core`: Full chess library (move generation, PGN, FEN, etc.)
- `rage-nnue`: NNUE inference with wasm SIMD and lite weights
- `rage-search`: Single-threaded search (wasm threads are possible with
  `SharedArrayBuffer` but complex; start single-threaded)
- `rage-uci`: Not included (no stdin/stdout in browser)

### What's excluded from wasm
- `rage-syzygy`: Tables too large for browser
- `rage-train`: No GPU access from wasm
- Multi-threading (initially)

### Wasm API

Exposed via `wasm-bindgen`:

```rust
#[wasm_bindgen]
pub struct RageEngine { /* ... */ }

#[wasm_bindgen]
impl RageEngine {
    pub fn new(weights: &[u8]) -> RageEngine;
    pub fn set_position_fen(&mut self, fen: &str) -> Result<(), JsValue>;
    pub fn make_move_san(&mut self, san: &str) -> Result<(), JsValue>;
    pub fn search(&mut self, depth: u8) -> SearchResult;
    pub fn search_time(&mut self, ms: u32) -> SearchResult;
    pub fn get_eval(&self) -> f64;
    pub fn legal_moves(&self) -> Vec<JsValue>;
    pub fn to_fen(&self) -> String;
    pub fn is_game_over(&self) -> bool;
}
```

The lite NNUE weights file (~2-5 MB) ships as a separate `.nnue` file fetched by the
JS host and passed to `new()`.

### Size Budget

- wasm binary: Target < 500 KB (gzipped)
- Lite weights: Target < 5 MB
- Total browser payload: < 6 MB

---

## Testing Strategy

Testing is a core part of the development process, not an afterthought.

### Unit Tests (per-crate)

- **rage-core**:
  - Bitboard operations: exhaustive for single-square ops, property-based for bulk
  - Move generation: perft tests against known positions (initial position, kiwipete,
    etc.) at depth 5-6. This is THE correctness test for move generation. Perft must
    match published results exactly.
  - FEN round-tripping: parse -> serialize -> parse must be identity
  - PGN parsing: test against corpus of real games
  - Zobrist hashing: verify incremental updates match full recomputation
  - Make/unmake: verify board state is identical after make+unmake for every legal
    move in a set of test positions

- **rage-nnue**:
  - Forward pass: verify quantized inference matches f32 reference within tolerance
  - Incremental accumulator: verify incremental updates match full recomputation
  - SIMD: each SIMD tier must produce identical results to scalar reference
  - Weight loading: round-trip serialization

- **rage-search**:
  - Tactical test suites: WAC (Win At Chess), STS (Strategic Test Suite), ECM
    (Encyclopedia of Chess Middlegames) - must solve a high percentage within time/
    depth limits
  - Mate-in-N puzzles: verify engine finds forced mates
  - TT: stress test with concurrent read/write from multiple threads
  - Time management: verify engine respects time limits with margin

- **rage-syzygy**:
  - Probe known positions against expected WDL/DTZ values
  - Verify behavior when tables are missing or corrupted
  - Round-trip: probe -> verify against known results for all 3-4 piece endings

- **rage-uci**:
  - Protocol parsing: test every UCI command format
  - Option handling: test all option types and edge cases
  - Integration: mock engine, verify correct UCI output sequences

### Integration Tests

- **Full search tests**: From known FEN positions, verify the engine finds the best
  move (or an acceptable move from a set) within time/depth constraints.
- **Self-play**: Engine plays itself, verify no crashes, no illegal moves, games
  terminate properly.
- **UCI compliance**: Script that sends UCI commands and verifies responses match
  protocol spec.

### Performance Tests

- **Perft speed**: Benchmark move generation speed (target: 200M+ nodes/sec)
- **NPS benchmark**: Fixed-depth search of standard positions, track NPS over time
- **NNUE inference**: Benchmark forward pass latency per position

### Regression Tests

- **Bench command**: `rage bench` runs a fixed set of positions at fixed depth and
  outputs a total node count. This number is a signature of the search - any change
  to search behavior changes the bench number. Used to verify patches don't
  accidentally alter search behavior.
- **Elo testing**: Major changes are tested via `rage-test sprt` to verify Elo
  gain/loss with statistical significance. Every non-trivial search or eval change
  must pass an SPRT test before being merged.

### CI Pipeline

- `just test`: All unit + integration tests
- `just lint`: clippy with `-D warnings`
- `just bench`: Regression bench check
- wasm build + test
- Miri for unsafe code validation (SIMD modules)

---

## Development Phases

### Phase 1: Foundation (`rage-core`)
Build the chess library. This is the bedrock everything else depends on.

1. Types: Square, File, Rank, Color, Piece, CastlingRights, Move
2. Bitboard operations and magic bitboard initialization
3. Board state and FEN parsing
4. Legal move generation
5. Perft testing (validate against known results through depth 6+)
6. Zobrist hashing
7. Make/unmake move
8. PGN parser
9. SAN notation conversion
10. Comprehensive test suite

**Exit criteria**: Perft results match published values for all standard test
positions. All tests pass. Library API is clean and documented.

### Phase 2: Basic Search (`rage-search`, `rage-uci`)
Get the engine playing moves with a simple evaluation.

1. Basic UCI protocol (position, go, stop, quit)
2. Negamax with alpha-beta pruning
3. Iterative deepening
4. Simple material + piece-square table evaluation (placeholder for NNUE)
5. Move ordering: MVV-LVA for captures
6. Quiescence search
7. Transposition table (single-threaded first)
8. Time management (basic)

**Exit criteria**: Engine plays legal chess, beats random movers, responds to UCI
commands. Estimated strength: ~1500-1800 Elo.

### Phase 3: NNUE Evaluation (`rage-nnue`, `rage-train`)
Replace the simple eval with NNUE.

1. Define HalfKAv2 network architecture in Burn
2. Data generation: use phase 2 engine for initial self-play data, supplement with
   Leela/Stockfish data (publicly available training data)
3. Training loop with loss function, LR scheduling
4. Quantization pipeline (f32 -> i16/i8)
5. Inference in Rust: scalar implementation first
6. Wire NNUE eval into search
7. SIMD implementations: AVX2 first, then AVX-512, SSE4.1, NEON
8. Runtime SIMD dispatch
9. Incremental accumulator updates

**Exit criteria**: NNUE eval integrated and working. Significant strength gain over
simple eval. SIMD inference verified correct across all tiers.

### Phase 4: Search Improvements
Unlock the remaining Elo with advanced search techniques.

1. Principal Variation Search (PVS)
2. Aspiration windows
3. Null move pruning
4. Late move reductions (LMR)
5. Late move pruning (LMP)
6. Futility pruning (static + extended)
7. Reverse futility pruning
8. Killer moves and counter moves
9. History heuristic (butterfly + continuation)
10. Singular extensions
11. SEE (Static Exchange Evaluation)
12. Multi-cut, razoring, probcut
13. Lazy SMP (multi-threaded search)
14. Lock-free transposition table
15. Advanced time management
16. Pondering

Each technique is added incrementally and verified via self-play to confirm Elo gain.

**Exit criteria**: Engine is competitive at super-GM level (3200+ Elo). Lazy SMP
scaling is efficient to 8+ threads.

### Phase 5: Tablebases (`rage-syzygy`)
Add endgame perfection.

1. Syzygy WDL table decompression and probing
2. DTZ probing
3. Integration into search (root filtering + search cutoffs)
4. LRU cache for table blocks
5. Memory-mapped file support

**Exit criteria**: Correct probing verified against known positions. Measurable
strength gain in endgames.

### Phase 6: Strength Testing Framework (`rage-test`)
Build the infrastructure for rigorous strength measurement.

1. UCI engine subprocess management (spawn, communicate, handle crashes/timeouts)
2. Game manager: play a game between two UCI engines with clock management
3. Concurrent game execution (many games in parallel)
4. SPRT implementation with pentanomial model
5. Fixed-game Elo calculation with confidence intervals
6. Opening book loading and paired assignment
7. SQLite results database
8. Clean CLI with live-updating progress display
9. `rage-test bench-diff` for quick regression checks
10. Tournament mode (round-robin, gauntlet)

**Exit criteria**: Can run statistically valid SPRT tests. Results match cutechess-cli

### Phase 7: Wasm
Browser deployment.

1. Ensure `rage-core` compiles to wasm cleanly
2. wasm SIMD implementation for NNUE
3. Train and export lite model weights
4. `wasm-bindgen` API
5. JavaScript/TypeScript wrapper package
6. Size optimization (wasm-opt, LTO)
7. Integration tests in wasmtime

**Exit criteria**: Engine runs in the browser, plays reasonable chess with lite
weights, total payload < 6 MB.
for the same engine pair and TC. Clean, informative CLI output.

### Phase 8: Polish and Strength
Iterate on strength and quality using the testing framework.

1. Tune search parameters (LMR tables, pruning margins, etc.) via SPSA or similar
   automated tuning — use `rage-test sprt` to validate each change
2. Retrain NNUE with higher-quality self-play data (now that engine is strong)
3. Iterative training: generate data -> train -> generate better data -> retrain
4. Tournament testing against other engines via `rage-test tournament`
5. UCI compliance validation
6. Documentation and API polish
7. Multi-PV support
8. `rage bench` stability tests in CI
9. Quantization-aware training for NNUE
10. Track Elo progression over time via `rage-test history`

**Exit criteria**: Top-5 engine strength verified in testing. Clean API. Robust CI.

---

## Key Design Decisions & Rationale

### Why magic bitboards?
Magic bitboards are the fastest known method for sliding piece move generation on
modern CPUs. They trade memory (a few hundred KB of lookup tables) for speed (a
single multiply + shift + table lookup per piece). Fancy magics minimize table size.

### Why HalfKAv2?
Proven architecture used by Stockfish and most top NNUE engines. The king-bucketed
approach captures king safety patterns effectively. Starting with a known-good
architecture lets us focus on training quality and search improvements rather than
architecture research.

### Why Lazy SMP over YBWC?
Lazy SMP is dramatically simpler to implement, has excellent scaling properties (near-
linear to ~8 threads, sublinear but still beneficial to 64+), and doesn't require
complex work-stealing or synchronization. Stockfish uses Lazy SMP. The simplicity
also reduces the risk of concurrency bugs.

### Why pure Rust?
- Memory safety guarantees (critical for a heavily multi-threaded program)
- No UB risk outside of explicit `unsafe` blocks (which we confine to SIMD)
- First-class wasm support via `wasm-bindgen`
- Cargo ecosystem for build, test, docs
- No CMake/make complexity
- Performance on par with C/C++ for this workload

### Why Burn for training?
Burn is the most mature pure-Rust deep learning framework. It has CUDA backend
support, automatic differentiation, and the training pipeline stays in the Rust
ecosystem. The H200 GPUs are supported via the CUDA backend. If Burn proves
insufficient, we can evaluate alternatives, but it's the strongest starting point.

### Why custom weight format (not ONNX, etc.)?
NNUE weight files need to be as compact as possible and load with zero overhead. A
custom binary format with quantized weights, versioned header, and architecture
metadata lets us memory-map the file directly into the inference structures with no
parsing or conversion step.

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Perft speed | 200M+ nodes/sec | Single-threaded legal move generation |
| Search NPS | 2M+ nodes/sec/thread | Depends on position complexity |
| NNUE inference | < 200ns/eval | AVX2, 1024-wide accumulator |
| NNUE inference (wasm) | < 2us/eval | wasm SIMD, 256-wide accumulator |
| TT probe | < 50ns | Cache-line aligned buckets |
| Lazy SMP scaling | > 0.6x per doubling | Measured at 1/2/4/8/16 threads |
| wasm binary size | < 500 KB gzipped | With wasm-opt -Oz |
| Lite weights size | < 5 MB | 256-wide accumulator |

---

## Dependencies (Minimal)

### General
- `clap` for command line argument parsing
- `anyhow` for nicer error handling
- `tracing` for logging and instrumentation
- `tracing_tracy` for using the tracy profiler

### Engine crates (rage-core, rage-nnue, rage-search, rage-uci, rage-syzygy)
- `std` only. No external crates in the hot path.
- Optional: `serde` behind feature flag (for serialization in non-performance paths)
- `wasm-bindgen` for wasm target only (rage-engine)

### Training crate (rage-train)
- `burn` with CUDA backend
- `rand` for data shuffling
- `rayon` for parallel data loading

### Testing framework (rage-test)
- `rusqlite` for results database (SQLite)
- `crossterm` for terminal UI (live progress display)

### Dev dependencies
- `criterion` for benchmarks

---

## Open Questions / Future Work

- **Chess960**: Support can be added to `rage-core` later by generalizing castling
  logic. Search and NNUE don't need changes.
- **NNUE architecture evolution**: Once the baseline HalfKAv2 is working, experiment
  with larger networks, different bucket schemes, or attention mechanisms.
- **Wasm threading**: SharedArrayBuffer + Web Workers for multi-threaded wasm search.
  Complex but significant strength gain for browser users.
- **Opening book**: Modern top engines don't use opening books in competition. For
  analysis use, we could add Polyglot book support later.
- **Distributed Lazy SMP**: Multiple machines sharing a TT over the network. Very
  niche but technically feasible.
- **SPSA tuning framework**: Automated parameter tuning. Could be a subcommand of
  `rage-test` (`rage-test tune`) that runs SPSA by orchestrating SPRT tests with
  perturbed parameter values. This is the standard approach for tuning search
  parameters in top engines.
