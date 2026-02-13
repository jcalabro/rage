#![allow(clippy::unwrap_used)]

mod common;

use proptest::prelude::*;
use rage_core::*;

// ---------------------------------------------------------------------------
// Bitboard algebra
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn bb_xor_self_is_zero(bits in any::<u64>()) {
        let bb = Bitboard::new(bits);
        prop_assert_eq!(bb ^ bb, Bitboard::EMPTY);
    }

    #[test]
    fn bb_double_not_is_identity(bits in any::<u64>()) {
        let bb = Bitboard::new(bits);
        prop_assert_eq!(!!bb, bb);
    }

    #[test]
    fn bb_and_distributes_over_or(a in any::<u64>(), b in any::<u64>(), c in any::<u64>()) {
        let (a, b, c) = (Bitboard::new(a), Bitboard::new(b), Bitboard::new(c));
        // a & (b | c) == (a & b) | (a & c)
        prop_assert_eq!(a & (b | c), (a & b) | (a & c));
    }

    #[test]
    fn bb_de_morgan_and(a in any::<u64>(), b in any::<u64>()) {
        let (a, b) = (Bitboard::new(a), Bitboard::new(b));
        prop_assert_eq!(!(a & b), !a | !b);
    }

    #[test]
    fn bb_de_morgan_or(a in any::<u64>(), b in any::<u64>()) {
        let (a, b) = (Bitboard::new(a), Bitboard::new(b));
        prop_assert_eq!(!(a | b), !a & !b);
    }

    #[test]
    fn bb_popcount_matches_iter_count(bits in any::<u64>()) {
        let bb = Bitboard::new(bits);
        prop_assert_eq!(bb.popcount() as usize, bb.into_iter().count());
    }

    #[test]
    fn bb_iter_yields_contained_squares(bits in any::<u64>()) {
        let bb = Bitboard::new(bits);
        for sq in bb {
            prop_assert!(bb.contains(sq), "iterator yielded {sq} but bb doesn't contain it");
        }
    }

    #[test]
    fn bb_set_clear_roundtrip(bits in any::<u64>(), sq_idx in 0u8..64) {
        let bb = Bitboard::new(bits);
        let sq = Square::new(sq_idx);
        // Setting then clearing restores the bit to 0.
        prop_assert!(!bb.set(sq).clear(sq).contains(sq));
        // Clearing then setting always sets the bit.
        prop_assert!(bb.clear(sq).set(sq).contains(sq));
    }
}

// ---------------------------------------------------------------------------
// Move encoding roundtrip
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn move_normal_roundtrip(from_idx in 0u8..64, to_idx in 0u8..64) {
        let (from, to) = (Square::new(from_idx), Square::new(to_idx));
        let m = Move::new(from, to);
        prop_assert_eq!(m.from_sq(), from);
        prop_assert_eq!(m.to_sq(), to);
        prop_assert!(!m.is_promotion());
        prop_assert!(!m.is_en_passant());
        prop_assert!(!m.is_castling());
    }

    #[test]
    fn move_promotion_roundtrip(from_idx in 0u8..64, to_idx in 0u8..64, promo_idx in 0usize..4) {
        let (from, to) = (Square::new(from_idx), Square::new(to_idx));
        let promo = PieceType::PROMOTION[promo_idx];
        let m = Move::new_promotion(from, to, promo);
        prop_assert_eq!(m.from_sq(), from);
        prop_assert_eq!(m.to_sq(), to);
        prop_assert!(m.is_promotion());
        prop_assert_eq!(m.promotion_piece(), promo);
    }

    #[test]
    fn move_en_passant_roundtrip(from_idx in 0u8..64, to_idx in 0u8..64) {
        let (from, to) = (Square::new(from_idx), Square::new(to_idx));
        let m = Move::new_en_passant(from, to);
        prop_assert_eq!(m.from_sq(), from);
        prop_assert_eq!(m.to_sq(), to);
        prop_assert!(m.is_en_passant());
    }

    #[test]
    fn move_castling_roundtrip(from_idx in 0u8..64, to_idx in 0u8..64) {
        let (from, to) = (Square::new(from_idx), Square::new(to_idx));
        let m = Move::new_castling(from, to);
        prop_assert_eq!(m.from_sq(), from);
        prop_assert_eq!(m.to_sq(), to);
        prop_assert!(m.is_castling());
    }
}

// ---------------------------------------------------------------------------
// FEN roundtrip
// ---------------------------------------------------------------------------

const FEN_SUITE: &[&str] = &[
    Board::STARTPOS_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
    "8/8/8/8/8/8/8/4K2k w - - 0 1",
    "4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1",
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
    "4k3/8/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/8/4K3 b kq - 0 1",
    "2kr3r/pppppppp/8/8/8/8/PPPPPPPP/2KR3R w - - 0 1",
    "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",
    "r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1",
    "8/8/8/4k3/8/8/8/4K3 w - - 50 100",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
    "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
    "8/2p5/3p4/KP5r/5p1k/8/4P1P1/1R6 w - - 1 2",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "8/5k2/8/5N2/5P2/8/8/4K3 w - - 0 1",
    "8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1",
    "5k2/8/8/8/8/8/8/4K2R w K - 0 1",
    "3k4/8/8/8/8/8/8/R3K3 w Q - 0 1",
    "r3k2r/1b4bq/8/8/8/8/7B/R3K2R w KQkq - 0 1",
    "8/8/1P6/5pr1/8/4R3/7p/k2K4 w - - 0 1",
    "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
];

proptest! {
    #[test]
    fn fen_roundtrip(idx in 0..FEN_SUITE.len()) {
        let fen = FEN_SUITE[idx];
        let board = Board::from_fen(fen).unwrap();
        prop_assert_eq!(board.to_fen(), fen);
    }
}

// ---------------------------------------------------------------------------
// Make/unmake + Zobrist consistency via random playout
// ---------------------------------------------------------------------------

proptest! {
    /// Play a random game from startpos, verifying Zobrist and make/unmake at each ply.
    #[test]
    fn make_unmake_random_game(seed in any::<u64>()) {
        let mut rng = common::Xorshift(seed | 1);
        let mut board = Board::starting_position();

        for _ in 0..80 {
            common::assert_board_consistent(&board);

            let fen_before = board.to_fen();
            let hash_before = board.zobrist_hash();
            let moves = board.generate_legal_moves();

            if moves.is_empty() {
                break;
            }

            let chosen_idx = (rng.next_u64() as usize) % moves.len();
            let chosen = moves.as_slice()[chosen_idx];

            board.make_move(chosen);
            prop_assert_eq!(
                board.zobrist_hash(),
                board.compute_zobrist(),
                "Zobrist mismatch after {} from {}", chosen, fen_before
            );

            board.unmake_move(chosen);
            prop_assert_eq!(
                board.to_fen(),
                fen_before,
                "FEN mismatch after make/unmake {}", chosen
            );
            prop_assert_eq!(
                board.zobrist_hash(),
                hash_before,
                "hash mismatch after make/unmake {}", chosen
            );

            // Actually play the move.
            board.make_move(chosen);

            if board.halfmove_clock() >= 100 {
                break;
            }
        }
    }

    /// Play a short random game, testing make/unmake on EVERY legal move at each ply.
    #[test]
    fn make_unmake_exhaustive_per_ply(seed in any::<u64>()) {
        let mut rng = common::Xorshift(seed | 1);
        let mut board = Board::starting_position();

        for _ in 0..20 {
            let fen_before = board.to_fen();
            let hash_before = board.zobrist_hash();
            let moves = board.generate_legal_moves();

            if moves.is_empty() {
                break;
            }

            // Test make/unmake for EVERY legal move.
            // Use assert_eq! (borrows) instead of prop_assert_eq! (moves) so
            // fen_before isn't consumed on the first iteration.
            for &m in moves.as_slice() {
                board.make_move(m);
                assert_eq!(
                    board.zobrist_hash(),
                    board.compute_zobrist(),
                    "Zobrist mismatch after {m} from {fen_before}"
                );
                board.unmake_move(m);
                assert_eq!(
                    board.to_fen(),
                    fen_before,
                    "FEN mismatch after make/unmake {m}"
                );
                assert_eq!(
                    board.zobrist_hash(),
                    hash_before,
                    "hash mismatch after make/unmake {m}"
                );
            }

            // Pick a random move and play it.
            let chosen_idx = (rng.next_u64() as usize) % moves.len();
            board.make_move(moves.as_slice()[chosen_idx]);

            if board.halfmove_clock() >= 100 {
                break;
            }
        }
    }
}
