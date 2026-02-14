use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use rage_core::Board;

use crate::search::Searcher;
use crate::time::SearchLimits;

/// Default bench depth (matches Stockfish convention).
pub const DEFAULT_BENCH_DEPTH: u32 = 13;

/// Default TT size for bench (MB).
const BENCH_TT_MB: usize = 16;

/// Diverse set of positions covering openings, middlegames, endgames, and tactics.
const BENCH_FENS: &[&str] = &[
    // Starting position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    // Open Sicilian
    "r1bqkb1r/pp2pppp/2np1n2/6B1/3NP3/2N5/PPP2PPP/R2QKB1R b KQkq - 0 6",
    // Kiwipete (complex middlegame)
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    // Ruy Lopez middlegame
    "r1bq1rk1/2ppbppp/p1n2n2/1p2p3/4P3/1B3N2/PPPP1PPP/RNBQR1K1 w - - 0 9",
    // Queen's Gambit Declined
    "rnbqk2r/ppp1bppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 3 4",
    // Italian Game tactical
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    // Complex middlegame with imbalance
    "r2qkb1r/pp1n1ppp/2p1pn2/3p1b2/2PP4/1QN1PN2/PP3PPP/R1B1KB1R w KQkq - 0 7",
    // Open position with tactical threats
    "r1b1k2r/ppppqppp/2n2n2/1Bb1p3/4P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
    // Endgame: rook + pawns
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    // Endgame: opposite bishops
    "8/5pk1/6p1/3Pb3/1p6/1P2B3/5PPK/8 w - - 0 1",
    // Endgame: knight vs bishop
    "8/2k5/2p3p1/p1Pp1n1p/P2P4/1K2BN2/6PP/8 w - - 0 1",
    // Double-edged middlegame
    "rnbq1rk1/pp3ppp/4pn2/2pp4/1bPP4/2NBPN2/PP3PPP/R1BQK2R w KQ - 0 7",
    // Tactical position: pins and forks
    "r2q1rk1/ppp2ppp/2n1bn2/3pp1B1/2B1P3/3P1N2/PPP2PPP/RN1QK2R w KQ - 0 7",
    // Pawn structure: isolated queen pawn
    "rnbqkb1r/pp3ppp/4pn2/2pP4/3P4/2N5/PP2PPPP/R1BQKBNR b KQkq - 0 5",
    // Closed position
    "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq c6 0 3",
    // Endgame: queen vs rook
    "6k1/5r2/8/8/8/8/1Q6/4K3 w - - 0 1",
    // Endgame: rook vs knight
    "8/8/4k3/8/3R4/8/2n5/4K3 w - - 0 1",
    // Middlegame with castling rights and tension
    "r2qkbnr/pppb1ppp/2n1p3/3pP3/3P4/5N2/PPP2PPP/RNBQKB1R w KQkq - 2 5",
    // King safety: opposite-side castling
    "r3kb1r/1pp2ppp/p1n1bn2/4p1B1/4P3/q1NB1N2/PPP2PPP/R2QK2R w KQkq - 0 9",
    // Endgame: passed pawn race
    "8/p4pk1/6p1/5p2/P4P2/6P1/5PK1/8 w - - 0 1",
];

/// Result of running the bench suite.
pub struct BenchResult {
    pub total_nodes: u64,
    pub elapsed_ms: u64,
}

/// Run the benchmark: search each position at the given depth, accumulate nodes.
/// TT is shared across positions (not cleared between them).
pub fn run_bench(depth: u32) -> BenchResult {
    let mut searcher = Searcher::new(BENCH_TT_MB);
    let stop = Arc::new(AtomicBool::new(false));
    let limits = SearchLimits {
        depth: Some(depth),
        ..Default::default()
    };

    let total_positions = BENCH_FENS.len();
    let mut total_nodes: u64 = 0;
    let start = Instant::now();

    for (i, fen) in BENCH_FENS.iter().enumerate() {
        let Ok(mut board) = Board::from_fen(fen) else {
            continue;
        };

        let result = searcher.search(&mut board, &limits, Arc::clone(&stop), None);
        total_nodes += result.nodes;

        println!(
            "Position {:>2}/{}: {} depth {}  nodes {}",
            i + 1,
            total_positions,
            fen,
            depth,
            result.nodes,
        );
    }

    let elapsed_ms = start.elapsed().as_millis() as u64;
    let nps = if elapsed_ms > 0 {
        total_nodes * 1000 / elapsed_ms
    } else {
        0
    };

    println!("===========================");
    println!("Total time : {elapsed_ms} ms");
    println!("Total nodes: {total_nodes}");
    println!("Nodes/sec  : {nps}");

    BenchResult {
        total_nodes,
        elapsed_ms,
    }
}
