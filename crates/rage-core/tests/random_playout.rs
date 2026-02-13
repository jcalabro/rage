#![allow(clippy::unwrap_used)]

mod common;

use rage_core::*;

/// Play a complete random game from the starting position, checking every
/// invariant at every move.
fn play_random_game(seed: u64) {
    let mut rng = common::Xorshift(seed);
    let mut board = Board::starting_position();
    let mut ply = 0u32;

    loop {
        // Check all structural invariants.
        common::assert_board_consistent(&board);

        let fen = board.to_fen();
        let hash = board.zobrist_hash();
        let moves = board.generate_legal_moves();

        // Verify every legal move individually.
        for &m in moves.as_slice() {
            board.make_move(m);

            // Legal moves must not leave the moving side's king in check.
            let moved_side = !board.side_to_move();
            let their_king = board.king_square(moved_side);
            assert!(
                !board.is_attacked(their_king, board.side_to_move()),
                "legal move {m} leaves king in check at ply {ply}\nFEN: {fen}"
            );

            // Zobrist must match full recomputation.
            assert_eq!(
                board.zobrist_hash(),
                board.compute_zobrist(),
                "Zobrist mismatch after {m} at ply {ply}\nFEN: {fen}"
            );

            board.unmake_move(m);

            // Make/unmake must restore exact board state.
            assert_eq!(
                board.to_fen(),
                fen,
                "FEN mismatch after make/unmake {m} at ply {ply}"
            );
            assert_eq!(
                board.zobrist_hash(),
                hash,
                "Zobrist mismatch after make/unmake {m} at ply {ply}"
            );
        }

        if moves.is_empty() {
            break; // Checkmate or stalemate.
        }

        // Pick and play a random move.
        let idx = (rng.next_u64() as usize) % moves.len();
        board.make_move(moves.as_slice()[idx]);
        ply += 1;

        // Stop on 50-move rule or excessive length.
        if board.halfmove_clock() >= 100 || ply >= 500 {
            break;
        }
    }

    // Final consistency check.
    common::assert_board_consistent(&board);
}

#[test]
fn random_playout_10_games() {
    for seed in 1u64..=10 {
        play_random_game(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    }
}

#[test]
#[ignore]
fn random_playout_1000_games() {
    for seed in 1u64..=1_000 {
        play_random_game(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    }
}
