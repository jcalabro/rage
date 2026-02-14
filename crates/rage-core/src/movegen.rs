//! Legal move generation.
//!
//! Generates pseudo-legal moves, then filters for legality by making the move
//! and checking if the own king is in check.

use crate::attacks;
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::moves::{Move, MoveList};
use crate::types::{Color, File, PieceType, Rank, Square};

impl Board {
    /// Generate all legal moves for the side to move.
    pub fn generate_legal_moves(&mut self) -> MoveList {
        let mut moves = MoveList::new();
        self.generate_pseudo_legal_moves(&mut moves);

        // Filter: keep only moves that don't leave own king in check.
        // Uses make/unmake to avoid heap-allocating clones.
        let mut legal = MoveList::new();
        for i in 0..moves.len() {
            let m = moves.as_slice()[i];
            self.make_move(m);
            // After make_move, side_to_move has flipped. Check if the side
            // that just moved left its king in check.
            let king_sq = self.king_square(!self.side_to_move());
            if !self.is_attacked(king_sq, self.side_to_move()) {
                legal.push(m);
            }
            self.unmake_move(m);
        }
        legal
    }

    /// Returns true if the side to move is in checkmate.
    pub fn is_checkmate(&mut self) -> bool {
        self.in_check() && self.generate_legal_moves().is_empty()
    }

    /// Returns true if the side to move is in stalemate.
    pub fn is_stalemate(&mut self) -> bool {
        !self.in_check() && self.generate_legal_moves().is_empty()
    }

    /// Parse a UCI long algebraic notation move (e.g. "e2e4", "e7e8q") and return
    /// the matching legal move. Returns an error if the string is malformed or
    /// the move is not legal in the current position.
    pub fn parse_uci_move(&mut self, s: &str) -> crate::Result<Move> {
        let bytes = s.as_bytes();
        if bytes.len() < 4 || bytes.len() > 5 {
            return Err(crate::Error::InvalidMove(s.to_string()));
        }

        let from_file = File::from_char(bytes[0] as char)
            .ok_or_else(|| crate::Error::InvalidMove(s.to_string()))?;
        let from_rank = Rank::from_char(bytes[1] as char)
            .ok_or_else(|| crate::Error::InvalidMove(s.to_string()))?;
        let to_file = File::from_char(bytes[2] as char)
            .ok_or_else(|| crate::Error::InvalidMove(s.to_string()))?;
        let to_rank = Rank::from_char(bytes[3] as char)
            .ok_or_else(|| crate::Error::InvalidMove(s.to_string()))?;

        let from = Square::from_file_rank(from_file, from_rank);
        let to = Square::from_file_rank(to_file, to_rank);

        let promo = if bytes.len() == 5 {
            Some(match bytes[4] {
                b'n' => PieceType::Knight,
                b'b' => PieceType::Bishop,
                b'r' => PieceType::Rook,
                b'q' => PieceType::Queen,
                _ => return Err(crate::Error::InvalidMove(s.to_string())),
            })
        } else {
            None
        };

        let moves = self.generate_legal_moves();
        for &m in moves.as_slice() {
            if m.from_sq() == from && m.to_sq() == to {
                match promo {
                    Some(pt) if m.is_promotion() && m.promotion_piece() == pt => return Ok(m),
                    None if !m.is_promotion() => return Ok(m),
                    _ => {}
                }
            }
        }

        Err(crate::Error::InvalidMove(s.to_string()))
    }

    /// Generate pseudo-legal moves (may include moves that leave the king in check).
    fn generate_pseudo_legal_moves(&self, moves: &mut MoveList) {
        let us = self.side_to_move();
        let them = !us;
        let our_pieces = self.color_bb(us);
        let their_pieces = self.color_bb(them);
        let occupied = self.occupied();

        self.generate_pawn_moves(moves, us, their_pieces, occupied);
        self.generate_knight_moves(moves, us, our_pieces);
        self.generate_bishop_moves(moves, us, our_pieces, occupied);
        self.generate_rook_moves(moves, us, our_pieces, occupied);
        self.generate_queen_moves(moves, us, our_pieces, occupied);
        self.generate_king_moves(moves, us, our_pieces);
        self.generate_castling_moves(moves, us, occupied);
    }

    fn generate_pawn_moves(
        &self,
        moves: &mut MoveList,
        us: Color,
        their_pieces: Bitboard,
        occupied: Bitboard,
    ) {
        let pawns = self.pieces(us, PieceType::Pawn);
        let empty = !occupied;

        let (push_dir, promo_rank, double_rank): (fn(Bitboard) -> Bitboard, Bitboard, Bitboard) =
            match us {
                Color::White => (Bitboard::north, Bitboard::RANK_8, Bitboard::RANK_4),
                Color::Black => (Bitboard::south, Bitboard::RANK_1, Bitboard::RANK_5),
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

    fn generate_king_moves(&self, moves: &mut MoveList, us: Color, our_pieces: Bitboard) {
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
pub fn perft(board: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let moves = board.generate_legal_moves();

    if depth == 1 {
        return moves.len() as u64;
    }

    let mut nodes = 0u64;
    for i in 0..moves.len() {
        let m = moves.as_slice()[i];
        board.make_move(m);
        nodes += perft(board, depth - 1);
        board.unmake_move(m);
    }
    nodes
}

/// Divide: like perft but prints the node count for each root move.
/// Useful for debugging move generation by comparing against a reference.
pub fn divide(board: &mut Board, depth: u32) -> u64 {
    let moves = board.generate_legal_moves();
    let mut total = 0u64;

    for i in 0..moves.len() {
        let m = moves.as_slice()[i];
        board.make_move(m);
        let nodes = if depth <= 1 {
            1
        } else {
            perft(board, depth - 1)
        };
        println!("{m}: {nodes}");
        total += nodes;
        board.unmake_move(m);
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
        let mut board = Board::starting_position();
        let moves = board.generate_legal_moves();
        assert_eq!(moves.len(), 20);
    }

    // Known perft results from https://www.chessprogramming.org/Perft_Results

    #[test]
    fn perft_startpos_depth1() {
        let mut board = Board::starting_position();
        assert_eq!(perft(&mut board, 1), 20);
    }

    #[test]
    fn perft_startpos_depth2() {
        let mut board = Board::starting_position();
        assert_eq!(perft(&mut board, 2), 400);
    }

    #[test]
    fn perft_startpos_depth3() {
        let mut board = Board::starting_position();
        assert_eq!(perft(&mut board, 3), 8_902);
    }

    #[test]
    #[ignore]
    fn perft_startpos_depth4() {
        let mut board = Board::starting_position();
        assert_eq!(perft(&mut board, 4), 197_281);
    }

    #[test]
    #[ignore]
    fn perft_startpos_depth5() {
        let mut board = Board::starting_position();
        assert_eq!(perft(&mut board, 5), 4_865_609);
    }

    // Kiwipete: great for testing castling, en passant, promotions
    const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

    #[test]
    fn perft_kiwipete_depth1() {
        let mut board = Board::from_fen(KIWIPETE).unwrap();
        assert_eq!(perft(&mut board, 1), 48);
    }

    #[test]
    fn perft_kiwipete_depth2() {
        let mut board = Board::from_fen(KIWIPETE).unwrap();
        assert_eq!(perft(&mut board, 2), 2_039);
    }

    #[test]
    fn perft_kiwipete_depth3() {
        let mut board = Board::from_fen(KIWIPETE).unwrap();
        assert_eq!(perft(&mut board, 3), 97_862);
    }

    #[test]
    #[ignore]
    fn perft_kiwipete_depth4() {
        let mut board = Board::from_fen(KIWIPETE).unwrap();
        assert_eq!(perft(&mut board, 4), 4_085_603);
    }

    // Position 3: endgame-ish with en passant
    const POSITION3: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";

    #[test]
    fn perft_position3_depth1() {
        let mut board = Board::from_fen(POSITION3).unwrap();
        assert_eq!(perft(&mut board, 1), 14);
    }

    #[test]
    fn perft_position3_depth2() {
        let mut board = Board::from_fen(POSITION3).unwrap();
        assert_eq!(perft(&mut board, 2), 191);
    }

    #[test]
    fn perft_position3_depth3() {
        let mut board = Board::from_fen(POSITION3).unwrap();
        assert_eq!(perft(&mut board, 3), 2_812);
    }

    #[test]
    #[ignore]
    fn perft_position3_depth4() {
        let mut board = Board::from_fen(POSITION3).unwrap();
        assert_eq!(perft(&mut board, 4), 43_238);
    }

    #[test]
    #[ignore]
    fn perft_position3_depth5() {
        let mut board = Board::from_fen(POSITION3).unwrap();
        assert_eq!(perft(&mut board, 5), 674_624);
    }

    // Position 4: many promotions and checks
    const POSITION4: &str = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";

    #[test]
    fn perft_position4_depth1() {
        let mut board = Board::from_fen(POSITION4).unwrap();
        assert_eq!(perft(&mut board, 1), 6);
    }

    #[test]
    fn perft_position4_depth2() {
        let mut board = Board::from_fen(POSITION4).unwrap();
        assert_eq!(perft(&mut board, 2), 264);
    }

    #[test]
    fn perft_position4_depth3() {
        let mut board = Board::from_fen(POSITION4).unwrap();
        assert_eq!(perft(&mut board, 3), 9_467);
    }

    #[test]
    #[ignore]
    fn perft_position4_depth4() {
        let mut board = Board::from_fen(POSITION4).unwrap();
        assert_eq!(perft(&mut board, 4), 422_333);
    }

    // Position 5: another complex middlegame
    const POSITION5: &str = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";

    #[test]
    fn perft_position5_depth1() {
        let mut board = Board::from_fen(POSITION5).unwrap();
        assert_eq!(perft(&mut board, 1), 44);
    }

    #[test]
    fn perft_position5_depth2() {
        let mut board = Board::from_fen(POSITION5).unwrap();
        assert_eq!(perft(&mut board, 2), 1_486);
    }

    #[test]
    fn perft_position5_depth3() {
        let mut board = Board::from_fen(POSITION5).unwrap();
        assert_eq!(perft(&mut board, 3), 62_379);
    }

    #[test]
    #[ignore]
    fn perft_position5_depth4() {
        let mut board = Board::from_fen(POSITION5).unwrap();
        assert_eq!(perft(&mut board, 4), 2_103_487);
    }

    // Position 6: alternative starting analysis
    const POSITION6: &str =
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";

    #[test]
    fn perft_position6_depth1() {
        let mut board = Board::from_fen(POSITION6).unwrap();
        assert_eq!(perft(&mut board, 1), 46);
    }

    #[test]
    fn perft_position6_depth2() {
        let mut board = Board::from_fen(POSITION6).unwrap();
        assert_eq!(perft(&mut board, 2), 2_079);
    }

    #[test]
    fn perft_position6_depth3() {
        let mut board = Board::from_fen(POSITION6).unwrap();
        assert_eq!(perft(&mut board, 3), 89_890);
    }

    #[test]
    #[ignore]
    fn perft_position6_depth4() {
        let mut board = Board::from_fen(POSITION6).unwrap();
        assert_eq!(perft(&mut board, 4), 3_894_594);
    }

    // Verify make/unmake preserves board state
    #[test]
    fn make_unmake_preserves_state() {
        let mut board = Board::starting_position();
        let original_fen = board.to_fen();
        let original_hash = board.zobrist_hash();
        let moves = board.generate_legal_moves();

        for &m in moves.as_slice() {
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
        let mut board = Board::starting_position();
        let moves = board.generate_legal_moves();

        for &m in moves.as_slice() {
            board.make_move(m);
            let incremental = board.zobrist_hash();
            let full = board.compute_zobrist();
            assert_eq!(
                incremental, full,
                "Zobrist mismatch after {m}: incremental=0x{incremental:016x}, full=0x{full:016x}"
            );
            board.unmake_move(m);
        }
    }

    // Checkmate: no legal moves, king is in check
    #[test]
    fn checkmate_has_no_legal_moves() {
        // Fool's mate final position
        let mut board =
            Board::from_fen("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
                .unwrap();
        let moves = board.generate_legal_moves();
        assert!(moves.is_empty(), "checkmate should have 0 legal moves");
        assert!(board.in_check());
    }

    // Stalemate: no legal moves, king is NOT in check
    #[test]
    fn stalemate_has_no_legal_moves() {
        // Black king trapped but not in check
        let mut board = Board::from_fen("5k2/5P2/5K2/8/8/8/8/8 b - - 0 1").unwrap();
        let moves = board.generate_legal_moves();
        assert!(moves.is_empty(), "stalemate should have 0 legal moves");
        assert!(!board.in_check());
    }

    // Make/unmake preserves state for en passant captures
    #[test]
    fn make_unmake_en_passant() {
        // After 1.e4 d5 2.e5 f5, en passant e5xf6 is available
        let mut board =
            Board::from_fen("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
                .unwrap();
        let original_fen = board.to_fen();
        let original_hash = board.zobrist_hash();
        let moves = board.generate_legal_moves();

        // Find the en passant move
        let ep_move = moves.iter().find(|m| m.is_en_passant());
        assert!(ep_move.is_some(), "should have an EP move available");

        for &m in moves.as_slice() {
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

    // Make/unmake preserves state for castling moves
    #[test]
    fn make_unmake_castling() {
        // Position where both sides can castle
        let mut board =
            Board::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1").unwrap();
        let original_fen = board.to_fen();
        let original_hash = board.zobrist_hash();
        let moves = board.generate_legal_moves();

        // Verify castling moves exist
        let castle_moves: Vec<_> = moves.iter().filter(|m| m.is_castling()).collect();
        assert_eq!(
            castle_moves.len(),
            2,
            "should have both kingside and queenside castling"
        );

        for &m in moves.as_slice() {
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

    // Make/unmake preserves state for promotion moves
    #[test]
    fn make_unmake_promotion() {
        // White pawn on 7th rank, can promote
        let mut board = Board::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let original_fen = board.to_fen();
        let original_hash = board.zobrist_hash();
        let moves = board.generate_legal_moves();

        // Should have 4 promotion moves (N, B, R, Q) plus king moves
        let promo_moves: Vec<_> = moves.iter().filter(|m| m.is_promotion()).collect();
        assert_eq!(promo_moves.len(), 4, "should have 4 promotion moves");

        for &m in moves.as_slice() {
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

    // Zobrist incremental matches full recomputation for special moves
    #[test]
    fn zobrist_incremental_special_moves() {
        let positions = [
            // EP available
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            // Castling available
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
            // Promotion available
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            // Promotion with capture
            "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        ];

        for fen in positions {
            let mut board = Board::from_fen(fen).unwrap();
            let moves = board.generate_legal_moves();
            for &m in moves.as_slice() {
                board.make_move(m);
                let incremental = board.zobrist_hash();
                let full = board.compute_zobrist();
                assert_eq!(
                    incremental, full,
                    "Zobrist mismatch in '{fen}' after {m}: incremental=0x{incremental:016x}, full=0x{full:016x}"
                );
                board.unmake_move(m);
            }
        }
    }

    #[test]
    fn is_checkmate_fools_mate() {
        let mut board =
            Board::from_fen("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
                .unwrap();
        assert!(board.is_checkmate());
        assert!(!board.is_stalemate());
    }

    #[test]
    fn is_stalemate_basic() {
        let mut board = Board::from_fen("5k2/5P2/5K2/8/8/8/8/8 b - - 0 1").unwrap();
        assert!(board.is_stalemate());
        assert!(!board.is_checkmate());
    }

    #[test]
    fn not_checkmate_or_stalemate() {
        let mut board = Board::starting_position();
        assert!(!board.is_checkmate());
        assert!(!board.is_stalemate());
    }

    #[test]
    fn parse_uci_normal_move() {
        let mut board = Board::starting_position();
        let m = board.parse_uci_move("e2e4").unwrap();
        assert_eq!(m.from_sq(), Square::E2);
        assert_eq!(m.to_sq(), Square::E4);
        assert!(!m.is_promotion());
    }

    #[test]
    fn parse_uci_promotion() {
        let mut board = Board::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let m = board.parse_uci_move("a7a8q").unwrap();
        assert!(m.is_promotion());
        assert_eq!(m.promotion_piece(), PieceType::Queen);
    }

    #[test]
    fn parse_uci_castling() {
        let mut board =
            Board::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1").unwrap();
        let m = board.parse_uci_move("e1g1").unwrap();
        assert!(m.is_castling());
    }

    #[test]
    fn parse_uci_invalid() {
        let mut board = Board::starting_position();
        assert!(board.parse_uci_move("e2e5").is_err());
        assert!(board.parse_uci_move("xx").is_err());
        assert!(board.parse_uci_move("").is_err());
    }

    // Verify divide returns same total as perft
    #[test]
    fn divide_matches_perft() {
        let mut board = Board::starting_position();
        let perft_result = perft(&mut board, 2);
        // divide prints to stdout, but returns the same total
        let divide_result = divide(&mut board, 2);
        assert_eq!(perft_result, divide_result);
    }

    // Same test but at depth 2 (covers all first-move responses)
    #[test]
    #[ignore]
    fn zobrist_incremental_depth2() {
        let mut board = Board::starting_position();
        let moves = board.generate_legal_moves();

        for &m1 in moves.as_slice() {
            board.make_move(m1);
            let moves2 = board.generate_legal_moves();
            for &m2 in moves2.as_slice() {
                board.make_move(m2);
                assert_eq!(
                    board.zobrist_hash(),
                    board.compute_zobrist(),
                    "Zobrist mismatch after {m1} {m2}"
                );
                board.unmake_move(m2);
            }
            board.unmake_move(m1);
        }
    }
}
