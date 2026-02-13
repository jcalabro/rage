//! Legal move generation.
//!
//! Generates pseudo-legal moves, then filters for legality by making the move
//! and checking if the own king is in check.

use crate::attacks;
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::moves::{Move, MoveList};
use crate::types::{Color, PieceType, Square};

impl Board {
    /// Generate all legal moves for the side to move.
    pub fn generate_legal_moves(&self) -> MoveList {
        let mut moves = MoveList::new();
        self.generate_pseudo_legal_moves(&mut moves);

        // Filter: keep only moves that don't leave own king in check
        let mut legal = MoveList::new();
        for &m in moves.as_slice() {
            let mut board = self.clone();
            board.make_move(m);
            // After make_move, side_to_move has flipped. Check if the side
            // that just moved left its king in check.
            let king_sq = board.king_square(!board.side_to_move());
            if !board.is_attacked(king_sq, board.side_to_move()) {
                legal.push(m);
            }
        }
        legal
    }

    /// Generate pseudo-legal moves (may include moves that leave the king in check).
    fn generate_pseudo_legal_moves(&self, moves: &mut MoveList) {
        let us = self.side_to_move();
        let them = !us;
        let our_pieces = self.color_bb(us);
        let their_pieces = self.color_bb(them);
        let occupied = self.occupied();

        self.generate_pawn_moves(moves, us, our_pieces, their_pieces, occupied);
        self.generate_knight_moves(moves, us, our_pieces);
        self.generate_bishop_moves(moves, us, our_pieces, occupied);
        self.generate_rook_moves(moves, us, our_pieces, occupied);
        self.generate_queen_moves(moves, us, our_pieces, occupied);
        self.generate_king_moves(moves, us, our_pieces, occupied);
        self.generate_castling_moves(moves, us, occupied);
    }

    fn generate_pawn_moves(
        &self,
        moves: &mut MoveList,
        us: Color,
        _our_pieces: Bitboard,
        their_pieces: Bitboard,
        occupied: Bitboard,
    ) {
        let pawns = self.pieces(us, PieceType::Pawn);
        let empty = !occupied;

        let (push_dir, promo_rank, _start_rank, double_rank): (
            fn(Bitboard) -> Bitboard,
            Bitboard,
            Bitboard,
            Bitboard,
        ) = match us {
            Color::White => (
                Bitboard::north,
                Bitboard::RANK_8,
                Bitboard::RANK_2,
                Bitboard::RANK_4,
            ),
            Color::Black => (
                Bitboard::south,
                Bitboard::RANK_1,
                Bitboard::RANK_7,
                Bitboard::RANK_5,
            ),
        };

        // Single pawn pushes
        let single_push = push_dir(pawns) & empty;
        let promotions = single_push & promo_rank;
        let non_promotions = single_push & !promo_rank;

        for to in non_promotions {
            let from = match us {
                Color::White => Square::new(to.index() as u8 - 8),
                Color::Black => Square::new(to.index() as u8 + 8),
            };
            moves.push(Move::new(from, to));
        }

        for to in promotions {
            let from = match us {
                Color::White => Square::new(to.index() as u8 - 8),
                Color::Black => Square::new(to.index() as u8 + 8),
            };
            for &promo in &PieceType::PROMOTION {
                moves.push(Move::new_promotion(from, to, promo));
            }
        }

        // Double pawn pushes
        let double_push = push_dir(
            single_push
                & match us {
                    Color::White => Bitboard::RANK_3,
                    Color::Black => Bitboard::RANK_6,
                },
        ) & empty
            & double_rank;

        for to in double_push {
            let from = match us {
                Color::White => Square::new(to.index() as u8 - 16),
                Color::Black => Square::new(to.index() as u8 + 16),
            };
            moves.push(Move::new(from, to));
        }

        // Pawn captures
        let capture_targets = their_pieces
            | match self.en_passant_square() {
                Some(ep) => Bitboard::from_square(ep),
                None => Bitboard::EMPTY,
            };

        for from in pawns {
            let attacks = attacks::pawn_attacks(from, us) & capture_targets;
            for to in attacks {
                if promo_rank.contains(to) {
                    for &promo in &PieceType::PROMOTION {
                        moves.push(Move::new_promotion(from, to, promo));
                    }
                } else if Some(to) == self.en_passant_square() {
                    moves.push(Move::new_en_passant(from, to));
                } else {
                    moves.push(Move::new(from, to));
                }
            }
        }
    }

    fn generate_knight_moves(&self, moves: &mut MoveList, us: Color, our_pieces: Bitboard) {
        let knights = self.pieces(us, PieceType::Knight);
        for from in knights {
            let targets = attacks::knight_attacks(from) & !our_pieces;
            for to in targets {
                moves.push(Move::new(from, to));
            }
        }
    }

    fn generate_bishop_moves(
        &self,
        moves: &mut MoveList,
        us: Color,
        our_pieces: Bitboard,
        occupied: Bitboard,
    ) {
        let bishops = self.pieces(us, PieceType::Bishop);
        for from in bishops {
            let targets = attacks::bishop_attacks(from, occupied) & !our_pieces;
            for to in targets {
                moves.push(Move::new(from, to));
            }
        }
    }

    fn generate_rook_moves(
        &self,
        moves: &mut MoveList,
        us: Color,
        our_pieces: Bitboard,
        occupied: Bitboard,
    ) {
        let rooks = self.pieces(us, PieceType::Rook);
        for from in rooks {
            let targets = attacks::rook_attacks(from, occupied) & !our_pieces;
            for to in targets {
                moves.push(Move::new(from, to));
            }
        }
    }

    fn generate_queen_moves(
        &self,
        moves: &mut MoveList,
        us: Color,
        our_pieces: Bitboard,
        occupied: Bitboard,
    ) {
        let queens = self.pieces(us, PieceType::Queen);
        for from in queens {
            let targets = attacks::queen_attacks(from, occupied) & !our_pieces;
            for to in targets {
                moves.push(Move::new(from, to));
            }
        }
    }

    fn generate_king_moves(
        &self,
        moves: &mut MoveList,
        us: Color,
        our_pieces: Bitboard,
        _occupied: Bitboard,
    ) {
        let king_sq = self.king_square(us);
        let targets = attacks::king_attacks(king_sq) & !our_pieces;
        for to in targets {
            moves.push(Move::new(king_sq, to));
        }
    }

    fn generate_castling_moves(&self, moves: &mut MoveList, us: Color, occupied: Bitboard) {
        let rights = self.castling_rights();
        let them = !us;

        let (king_sq, ks_to, qs_to, ks_path_clear, qs_path_clear, ks_check_sqs, qs_check_sqs) =
            match us {
                Color::White => (
                    Square::E1,
                    Square::G1,
                    Square::C1,
                    // Squares between king and rook must be empty (kingside: f1, g1)
                    Bitboard::from_square(Square::F1) | Bitboard::from_square(Square::G1),
                    // Queenside: b1, c1, d1
                    Bitboard::from_square(Square::B1)
                        | Bitboard::from_square(Square::C1)
                        | Bitboard::from_square(Square::D1),
                    // King must not pass through or land on attacked squares (e1, f1, g1)
                    [Square::E1, Square::F1, Square::G1],
                    // Queenside: e1, d1, c1
                    [Square::E1, Square::D1, Square::C1],
                ),
                Color::Black => (
                    Square::E8,
                    Square::G8,
                    Square::C8,
                    Bitboard::from_square(Square::F8) | Bitboard::from_square(Square::G8),
                    Bitboard::from_square(Square::B8)
                        | Bitboard::from_square(Square::C8)
                        | Bitboard::from_square(Square::D8),
                    [Square::E8, Square::F8, Square::G8],
                    [Square::E8, Square::D8, Square::C8],
                ),
            };

        // Verify king is actually on its starting square
        if self.king_square(us) != king_sq {
            return;
        }

        // Kingside
        if rights.can_castle_kingside(us)
            && (occupied & ks_path_clear).is_empty()
            && !ks_check_sqs.iter().any(|&sq| self.is_attacked(sq, them))
        {
            moves.push(Move::new_castling(king_sq, ks_to));
        }

        // Queenside
        if rights.can_castle_queenside(us)
            && (occupied & qs_path_clear).is_empty()
            && !qs_check_sqs.iter().any(|&sq| self.is_attacked(sq, them))
        {
            moves.push(Move::new_castling(king_sq, qs_to));
        }
    }
}

// ---------------------------------------------------------------------------
// Perft
// ---------------------------------------------------------------------------

/// Perft (performance test): count the number of leaf nodes at a given depth.
/// This is the fundamental correctness test for move generation.
pub fn perft(board: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let moves = board.generate_legal_moves();

    if depth == 1 {
        return moves.len() as u64;
    }

    let mut nodes = 0u64;
    for &m in moves.as_slice() {
        let mut board = board.clone();
        board.make_move(m);
        nodes += perft(&board, depth - 1);
    }
    nodes
}

/// Divide: like perft but prints the node count for each root move.
/// Useful for debugging move generation by comparing against a reference.
pub fn divide(board: &Board, depth: u32) -> u64 {
    let moves = board.generate_legal_moves();
    let mut total = 0u64;

    for &m in moves.as_slice() {
        let mut board = board.clone();
        board.make_move(m);
        let nodes = if depth <= 1 {
            1
        } else {
            perft(&board, depth - 1)
        };
        println!("{m}: {nodes}");
        total += nodes;
    }
    println!("\nTotal: {total}");
    total
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn starting_position_20_moves() {
        let board = Board::starting_position();
        let moves = board.generate_legal_moves();
        assert_eq!(moves.len(), 20);
    }

    // Known perft results from https://www.chessprogramming.org/Perft_Results

    #[test]
    fn perft_startpos_depth1() {
        let board = Board::starting_position();
        assert_eq!(perft(&board, 1), 20);
    }

    #[test]
    fn perft_startpos_depth2() {
        let board = Board::starting_position();
        assert_eq!(perft(&board, 2), 400);
    }

    #[test]
    fn perft_startpos_depth3() {
        let board = Board::starting_position();
        assert_eq!(perft(&board, 3), 8_902);
    }

    #[test]
    fn perft_startpos_depth4() {
        let board = Board::starting_position();
        assert_eq!(perft(&board, 4), 197_281);
    }

    #[test]
    fn perft_startpos_depth5() {
        let board = Board::starting_position();
        assert_eq!(perft(&board, 5), 4_865_609);
    }

    // Kiwipete: great for testing castling, en passant, promotions
    const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

    #[test]
    fn perft_kiwipete_depth1() {
        let board = Board::from_fen(KIWIPETE).unwrap();
        assert_eq!(perft(&board, 1), 48);
    }

    #[test]
    fn perft_kiwipete_depth2() {
        let board = Board::from_fen(KIWIPETE).unwrap();
        assert_eq!(perft(&board, 2), 2_039);
    }

    #[test]
    fn perft_kiwipete_depth3() {
        let board = Board::from_fen(KIWIPETE).unwrap();
        assert_eq!(perft(&board, 3), 97_862);
    }

    #[test]
    fn perft_kiwipete_depth4() {
        let board = Board::from_fen(KIWIPETE).unwrap();
        assert_eq!(perft(&board, 4), 4_085_603);
    }

    // Position 3: endgame-ish with en passant
    const POSITION3: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";

    #[test]
    fn perft_position3_depth1() {
        let board = Board::from_fen(POSITION3).unwrap();
        assert_eq!(perft(&board, 1), 14);
    }

    #[test]
    fn perft_position3_depth2() {
        let board = Board::from_fen(POSITION3).unwrap();
        assert_eq!(perft(&board, 2), 191);
    }

    #[test]
    fn perft_position3_depth3() {
        let board = Board::from_fen(POSITION3).unwrap();
        assert_eq!(perft(&board, 3), 2_812);
    }

    #[test]
    fn perft_position3_depth4() {
        let board = Board::from_fen(POSITION3).unwrap();
        assert_eq!(perft(&board, 4), 43_238);
    }

    #[test]
    fn perft_position3_depth5() {
        let board = Board::from_fen(POSITION3).unwrap();
        assert_eq!(perft(&board, 5), 674_624);
    }

    // Position 4: many promotions and checks
    const POSITION4: &str = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";

    #[test]
    fn perft_position4_depth1() {
        let board = Board::from_fen(POSITION4).unwrap();
        assert_eq!(perft(&board, 1), 6);
    }

    #[test]
    fn perft_position4_depth2() {
        let board = Board::from_fen(POSITION4).unwrap();
        assert_eq!(perft(&board, 2), 264);
    }

    #[test]
    fn perft_position4_depth3() {
        let board = Board::from_fen(POSITION4).unwrap();
        assert_eq!(perft(&board, 3), 9_467);
    }

    #[test]
    fn perft_position4_depth4() {
        let board = Board::from_fen(POSITION4).unwrap();
        assert_eq!(perft(&board, 4), 422_333);
    }

    // Position 5: another complex middlegame
    const POSITION5: &str = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";

    #[test]
    fn perft_position5_depth1() {
        let board = Board::from_fen(POSITION5).unwrap();
        assert_eq!(perft(&board, 1), 44);
    }

    #[test]
    fn perft_position5_depth2() {
        let board = Board::from_fen(POSITION5).unwrap();
        assert_eq!(perft(&board, 2), 1_486);
    }

    #[test]
    fn perft_position5_depth3() {
        let board = Board::from_fen(POSITION5).unwrap();
        assert_eq!(perft(&board, 3), 62_379);
    }

    #[test]
    fn perft_position5_depth4() {
        let board = Board::from_fen(POSITION5).unwrap();
        assert_eq!(perft(&board, 4), 2_103_487);
    }

    // Position 6: alternative starting analysis
    const POSITION6: &str =
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";

    #[test]
    fn perft_position6_depth1() {
        let board = Board::from_fen(POSITION6).unwrap();
        assert_eq!(perft(&board, 1), 46);
    }

    #[test]
    fn perft_position6_depth2() {
        let board = Board::from_fen(POSITION6).unwrap();
        assert_eq!(perft(&board, 2), 2_079);
    }

    #[test]
    fn perft_position6_depth3() {
        let board = Board::from_fen(POSITION6).unwrap();
        assert_eq!(perft(&board, 3), 89_890);
    }

    #[test]
    fn perft_position6_depth4() {
        let board = Board::from_fen(POSITION6).unwrap();
        assert_eq!(perft(&board, 4), 3_894_594);
    }

    // Verify make/unmake preserves board state
    #[test]
    fn make_unmake_preserves_state() {
        let board = Board::starting_position();
        let original_fen = board.to_fen();
        let original_hash = board.zobrist_hash();
        let moves = board.generate_legal_moves();

        for &m in moves.as_slice() {
            let mut board = board.clone();
            board.make_move(m);
            board.unmake_move(m);
            assert_eq!(
                board.to_fen(),
                original_fen,
                "FEN mismatch after make/unmake {m}"
            );
            assert_eq!(
                board.zobrist_hash(),
                original_hash,
                "Zobrist mismatch after make/unmake {m}"
            );
        }
    }

    // Verify Zobrist incremental update matches full recomputation
    #[test]
    fn zobrist_incremental_matches_full() {
        let board = Board::starting_position();
        let moves = board.generate_legal_moves();

        for &m in moves.as_slice() {
            let mut board = board.clone();
            board.make_move(m);
            let incremental = board.zobrist_hash();
            let full = board.compute_zobrist();
            assert_eq!(
                incremental, full,
                "Zobrist mismatch after {m}: incremental=0x{incremental:016x}, full=0x{full:016x}"
            );
        }
    }

    // Same test but at depth 2 (covers all first-move responses)
    #[test]
    fn zobrist_incremental_depth2() {
        let board = Board::starting_position();
        let moves = board.generate_legal_moves();

        for &m1 in moves.as_slice() {
            let mut board1 = board.clone();
            board1.make_move(m1);

            let moves2 = board1.generate_legal_moves();
            for &m2 in moves2.as_slice() {
                let mut board2 = board1.clone();
                board2.make_move(m2);
                assert_eq!(
                    board2.zobrist_hash(),
                    board2.compute_zobrist(),
                    "Zobrist mismatch after {m1} {m2}"
                );
            }
        }
    }
}
