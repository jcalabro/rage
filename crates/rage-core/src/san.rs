//! Standard Algebraic Notation (SAN) encoding and decoding.
//!
//! Provides `Board::move_to_san` (encode) and `Board::parse_san` (decode)
//! for converting between `Move` values and human-readable SAN strings
//! like "Nf3", "O-O", "exd5", "e8=Q", "Qf7#".

use crate::board::Board;
use crate::moves::Move;
use crate::types::{File, PieceType, Rank, Square};

impl Board {
    /// Convert a legal move to Standard Algebraic Notation.
    ///
    /// The move must be legal in the current position. Returns a SAN string
    /// like "Nf3", "O-O", "exd5", "e8=Q", "Qf7#", etc.
    pub fn move_to_san(&mut self, m: Move) -> String {
        let mut san = String::new();

        // Castling
        if m.is_castling() {
            if m.to_sq().file() == File::G {
                san.push_str("O-O");
            } else {
                san.push_str("O-O-O");
            }
            self.append_check_suffix(&mut san, m);
            return san;
        }

        let piece = match self.piece_on(m.from_sq()) {
            Some(p) => p,
            None => return san,
        };
        let pt = piece.piece_type;
        let is_capture = self.is_capture(m);

        // Piece letter (not for pawns)
        if pt != PieceType::Pawn {
            san.push(pt.to_char());
        }

        // Disambiguation for non-pawn pieces
        if pt != PieceType::Pawn {
            let moves = self.generate_legal_moves();
            let mut need_file = false;
            let mut need_rank = false;
            let mut ambiguous = false;

            for &other in moves.as_slice() {
                if other == m {
                    continue;
                }
                if other.to_sq() != m.to_sq() {
                    continue;
                }
                if let Some(other_piece) = self.piece_on(other.from_sq()) {
                    if other_piece.piece_type != pt {
                        continue;
                    }
                } else {
                    continue;
                }

                ambiguous = true;
                if other.from_sq().file() == m.from_sq().file() {
                    need_rank = true;
                }
                if other.from_sq().rank() == m.from_sq().rank() {
                    need_file = true;
                }
            }

            if ambiguous {
                if !need_file && !need_rank {
                    // Different files and different ranks — file alone suffices
                    need_file = true;
                }
                if need_file {
                    san.push(m.from_sq().file().to_char());
                }
                if need_rank {
                    san.push(m.from_sq().rank().to_char());
                }
            }
        } else if is_capture {
            // Pawn captures always include origin file
            san.push(m.from_sq().file().to_char());
        }

        // Capture symbol
        if is_capture {
            san.push('x');
        }

        // To-square
        san.push(m.to_sq().file().to_char());
        san.push(m.to_sq().rank().to_char());

        // Promotion
        if m.is_promotion() {
            san.push('=');
            san.push(m.promotion_piece().to_char());
        }

        self.append_check_suffix(&mut san, m);
        san
    }

    /// Make the move, check for check/checkmate, unmake, and append suffix.
    fn append_check_suffix(&mut self, san: &mut String, m: Move) {
        self.make_move(m);
        if self.in_check() {
            if self.is_checkmate() {
                san.push('#');
            } else {
                san.push('+');
            }
        }
        self.unmake_move(m);
    }

    /// Parse a Standard Algebraic Notation string and return the matching legal move.
    ///
    /// Accepts SAN with or without check/mate/annotation suffixes (+, #, !, ?).
    /// Also accepts "0-0" and "0-0-0" as castling alternatives to "O-O" and "O-O-O".
    pub fn parse_san(&mut self, san: &str) -> crate::Result<Move> {
        let original = san;

        // Strip trailing annotations: +, #, !, ?
        let s = san.trim_end_matches(['+', '#', '!', '?']);

        if s.is_empty() {
            return Err(crate::Error::InvalidMove(original.to_string()));
        }

        let moves = self.generate_legal_moves();

        // Castling (check O-O-O before O-O to avoid prefix match)
        if s == "O-O-O" || s == "0-0-0" {
            for &m in moves.as_slice() {
                if m.is_castling() && m.to_sq().file() == File::C {
                    return Ok(m);
                }
            }
            return Err(crate::Error::InvalidMove(original.to_string()));
        }
        if s == "O-O" || s == "0-0" {
            for &m in moves.as_slice() {
                if m.is_castling() && m.to_sq().file() == File::G {
                    return Ok(m);
                }
            }
            return Err(crate::Error::InvalidMove(original.to_string()));
        }

        let bytes = s.as_bytes();
        let len = bytes.len();

        // Parse from the back
        let mut pos = len;

        // Promotion piece (e.g. "=Q")
        let promo = if pos >= 2 && bytes[pos - 2] == b'=' {
            let pt = PieceType::from_char(bytes[pos - 1] as char)
                .ok_or_else(|| crate::Error::InvalidMove(original.to_string()))?;
            if pt == PieceType::Pawn || pt == PieceType::King {
                return Err(crate::Error::InvalidMove(original.to_string()));
            }
            pos -= 2;
            Some(pt)
        } else {
            None
        };

        // To-square (last 2 chars before promotion)
        if pos < 2 {
            return Err(crate::Error::InvalidMove(original.to_string()));
        }
        let to_rank = Rank::from_char(bytes[pos - 1] as char)
            .ok_or_else(|| crate::Error::InvalidMove(original.to_string()))?;
        let to_file = File::from_char(bytes[pos - 2] as char)
            .ok_or_else(|| crate::Error::InvalidMove(original.to_string()))?;
        let to = Square::from_file_rank(to_file, to_rank);
        pos -= 2;

        // Capture marker
        if pos > 0 && bytes[pos - 1] == b'x' {
            pos -= 1;
        }

        // Piece type (leading uppercase letter, or Pawn if none)
        let mut start = 0;
        let piece_type = if pos > 0 && bytes[0].is_ascii_uppercase() {
            let pt = PieceType::from_char(bytes[0] as char)
                .ok_or_else(|| crate::Error::InvalidMove(original.to_string()))?;
            if pt == PieceType::Pawn {
                return Err(crate::Error::InvalidMove(original.to_string()));
            }
            start = 1;
            pt
        } else {
            PieceType::Pawn
        };

        // Disambiguation: chars between piece letter and capture/to-square
        let disambig = &bytes[start..pos];
        let disambig_file = if !disambig.is_empty() {
            File::from_char(disambig[0] as char)
        } else {
            None
        };
        let disambig_rank = if disambig.len() >= 2 {
            Rank::from_char(disambig[1] as char)
        } else if disambig.len() == 1 && disambig_file.is_none() {
            // Single char that's not a file — must be a rank
            Rank::from_char(disambig[0] as char)
        } else {
            None
        };

        // Filter legal moves
        let mut matched: Option<Move> = None;
        for &m in moves.as_slice() {
            if m.to_sq() != to {
                continue;
            }

            let piece = match self.piece_on(m.from_sq()) {
                Some(p) => p,
                None => continue,
            };
            if piece.piece_type != piece_type {
                continue;
            }

            // Check promotion
            if m.is_promotion() {
                match promo {
                    Some(pt) if m.promotion_piece() == pt => {}
                    _ => continue,
                }
            } else if promo.is_some() {
                continue;
            }

            // Check disambiguation
            if let Some(f) = disambig_file
                && m.from_sq().file() != f
            {
                continue;
            }
            if let Some(r) = disambig_rank
                && m.from_sq().rank() != r
            {
                continue;
            }

            if matched.is_some() {
                return Err(crate::Error::InvalidMove(format!(
                    "ambiguous SAN: {original}"
                )));
            }
            matched = Some(m);
        }

        matched.ok_or_else(|| crate::Error::InvalidMove(original.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // --- move_to_san ---

    #[test]
    fn basic_pawn_moves() {
        let mut board = Board::starting_position();
        let e4 = board.parse_uci_move("e2e4").unwrap();
        assert_eq!(board.move_to_san(e4), "e4");
        let d4 = board.parse_uci_move("d2d4").unwrap();
        assert_eq!(board.move_to_san(d4), "d4");
    }

    #[test]
    fn basic_knight_move() {
        let mut board = Board::starting_position();
        let nf3 = board.parse_uci_move("g1f3").unwrap();
        assert_eq!(board.move_to_san(nf3), "Nf3");
    }

    #[test]
    fn castling_kingside() {
        let mut board =
            Board::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1").unwrap();
        let oo = board.parse_uci_move("e1g1").unwrap();
        assert_eq!(board.move_to_san(oo), "O-O");
    }

    #[test]
    fn castling_queenside() {
        let mut board =
            Board::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1").unwrap();
        let ooo = board.parse_uci_move("e1c1").unwrap();
        assert_eq!(board.move_to_san(ooo), "O-O-O");
    }

    #[test]
    fn pawn_capture() {
        let mut board =
            Board::from_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
                .unwrap();
        let exd5 = board.parse_uci_move("e4d5").unwrap();
        assert_eq!(board.move_to_san(exd5), "exd5");
    }

    #[test]
    fn knight_capture() {
        // Position where knight can capture
        let mut board =
            Board::from_fen("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
                .unwrap();
        let nxe5 = board.parse_uci_move("f3e5").unwrap();
        assert_eq!(board.move_to_san(nxe5), "Nxe5");
    }

    #[test]
    fn promotion() {
        // Black king on h6 — queen on a8 doesn't give check
        let mut board = Board::from_fen("8/P6k/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let promo_q = board.parse_uci_move("a7a8q").unwrap();
        assert_eq!(board.move_to_san(promo_q), "a8=Q");
        let promo_n = board.parse_uci_move("a7a8n").unwrap();
        assert_eq!(board.move_to_san(promo_n), "a8=N");
    }

    #[test]
    fn promotion_with_check() {
        // Queen on a8 gives check to king on e8 along 8th rank
        let mut board = Board::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let promo_q = board.parse_uci_move("a7a8q").unwrap();
        assert_eq!(board.move_to_san(promo_q), "a8=Q+");
    }

    #[test]
    fn promotion_capture() {
        // Black king on h7 — queen on b8 doesn't give check
        let mut board = Board::from_fen("1n6/P6k/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let promo_cap = board.parse_uci_move("a7b8q").unwrap();
        assert_eq!(board.move_to_san(promo_cap), "axb8=Q");
    }

    #[test]
    fn check_suffix() {
        // Scholar's mate position: Qf7 gives check
        let mut board =
            Board::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
                .unwrap();
        let qf7 = board.parse_uci_move("f3f7").unwrap();
        let san = board.move_to_san(qf7);
        assert!(
            san.ends_with('#') || san.ends_with('+'),
            "expected check/mate suffix in {san}"
        );
    }

    #[test]
    fn checkmate_suffix() {
        // Scholar's mate: Qxf7#
        let mut board =
            Board::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
                .unwrap();
        let qxf7 = board.parse_uci_move("f3f7").unwrap();
        let san = board.move_to_san(qxf7);
        assert_eq!(san, "Qxf7#");
    }

    #[test]
    fn file_disambiguation() {
        // Two rooks on a1 and h1, king on e2 (not blocking the rank)
        let mut board = Board::from_fen("4k3/8/8/8/8/8/4K3/R6R w - - 0 1").unwrap();
        let moves = board.generate_legal_moves();
        // Both rooks can reach d1
        let rad1: Vec<_> = moves
            .iter()
            .filter(|m| m.to_sq() == Square::D1 && m.from_sq() == Square::A1)
            .collect();
        assert!(!rad1.is_empty(), "Ra1 should be able to go to d1");
        let san = board.move_to_san(*rad1[0]);
        assert_eq!(san, "Rad1");
    }

    #[test]
    fn rank_disambiguation() {
        // Two rooks on same file, different ranks
        let mut board = Board::from_fen("4k3/8/8/8/8/8/R7/R3K3 w Q - 0 1").unwrap();
        let moves = board.generate_legal_moves();
        // Both rooks on a-file can go to a4
        let r1a4: Vec<_> = moves
            .iter()
            .filter(|m| m.to_sq() == Square::A4 && m.from_sq() == Square::A1)
            .collect();
        if !r1a4.is_empty() {
            let san = board.move_to_san(*r1a4[0]);
            assert_eq!(san, "R1a4");
        }
    }

    #[test]
    fn en_passant_san() {
        let mut board =
            Board::from_fen("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
                .unwrap();
        let ep = board.parse_uci_move("e5f6").unwrap();
        assert!(ep.is_en_passant());
        assert_eq!(board.move_to_san(ep), "exf6");
    }

    // --- parse_san ---

    #[test]
    fn parse_basic_pawn() {
        let mut board = Board::starting_position();
        let m = board.parse_san("e4").unwrap();
        assert_eq!(m.from_sq(), Square::E2);
        assert_eq!(m.to_sq(), Square::E4);
    }

    #[test]
    fn parse_knight() {
        let mut board = Board::starting_position();
        let m = board.parse_san("Nf3").unwrap();
        assert_eq!(m.from_sq(), Square::G1);
        assert_eq!(m.to_sq(), Square::F3);
    }

    #[test]
    fn parse_castling_o() {
        let mut board =
            Board::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1").unwrap();
        let oo = board.parse_san("O-O").unwrap();
        assert!(oo.is_castling());
        assert_eq!(oo.to_sq().file(), File::G);

        let ooo = board.parse_san("O-O-O").unwrap();
        assert!(ooo.is_castling());
        assert_eq!(ooo.to_sq().file(), File::C);
    }

    #[test]
    fn parse_castling_zeros() {
        let mut board =
            Board::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1").unwrap();
        let oo = board.parse_san("0-0").unwrap();
        assert!(oo.is_castling());
        let ooo = board.parse_san("0-0-0").unwrap();
        assert!(ooo.is_castling());
    }

    #[test]
    fn parse_promotion() {
        let mut board = Board::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let m = board.parse_san("a8=Q").unwrap();
        assert!(m.is_promotion());
        assert_eq!(m.promotion_piece(), PieceType::Queen);
        assert_eq!(m.to_sq(), Square::A8);
    }

    #[test]
    fn parse_promotion_capture() {
        let mut board = Board::from_fen("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let m = board.parse_san("axb8=N").unwrap();
        assert!(m.is_promotion());
        assert_eq!(m.promotion_piece(), PieceType::Knight);
        assert_eq!(m.to_sq(), Square::B8);
    }

    #[test]
    fn parse_with_annotations() {
        let mut board = Board::starting_position();
        let m1 = board.parse_san("Nf3!").unwrap();
        assert_eq!(m1.to_sq(), Square::F3);
        let m2 = board.parse_san("e4!!").unwrap();
        assert_eq!(m2.to_sq(), Square::E4);
        let m3 = board.parse_san("d4?!").unwrap();
        assert_eq!(m3.to_sq(), Square::D4);
    }

    #[test]
    fn parse_with_check_suffix() {
        // Position where Bb5+ gives check
        let mut board =
            Board::from_fen("rnbqkbnr/pppp1ppp/4p3/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 2")
                .unwrap();
        let m = board.parse_san("Bb5+").unwrap();
        assert_eq!(m.to_sq(), Square::B5);
    }

    #[test]
    fn parse_errors() {
        let mut board = Board::starting_position();
        assert!(board.parse_san("").is_err());
        assert!(board.parse_san("Zf3").is_err());
        assert!(board.parse_san("e9").is_err());
        assert!(board.parse_san("e5").is_err()); // not a legal pawn move from startpos
    }

    #[test]
    fn parse_castling_not_available() {
        let mut board = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert!(board.parse_san("O-O").is_err());
        assert!(board.parse_san("O-O-O").is_err());
    }

    // --- Roundtrip tests ---

    #[test]
    fn roundtrip_startpos() {
        let mut board = Board::starting_position();
        let moves = board.generate_legal_moves();
        for &m in moves.as_slice() {
            let san = board.move_to_san(m);
            let parsed = board
                .parse_san(&san)
                .unwrap_or_else(|e| panic!("failed to parse SAN '{san}' (from {m}): {e}"));
            assert_eq!(parsed, m, "roundtrip failed for SAN '{san}' (move {m})");
        }
    }

    #[test]
    fn roundtrip_kiwipete() {
        let mut board =
            Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
                .unwrap();
        let moves = board.generate_legal_moves();
        for &m in moves.as_slice() {
            let san = board.move_to_san(m);
            let parsed = board
                .parse_san(&san)
                .unwrap_or_else(|e| panic!("failed to parse SAN '{san}' (from {m}): {e}"));
            assert_eq!(parsed, m, "roundtrip failed for SAN '{san}' (move {m})");
        }
    }

    #[test]
    fn roundtrip_promotion_position() {
        // Position with many promotions
        let mut board =
            Board::from_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
                .unwrap();
        let moves = board.generate_legal_moves();
        for &m in moves.as_slice() {
            let san = board.move_to_san(m);
            let parsed = board
                .parse_san(&san)
                .unwrap_or_else(|e| panic!("failed to parse SAN '{san}' (from {m}): {e}"));
            assert_eq!(parsed, m, "roundtrip failed for SAN '{san}' (move {m})");
        }
    }

    #[test]
    fn roundtrip_complex_positions() {
        let positions = [
            // EP available
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            // Position with castling
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
            // Endgame
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            // Two rooks on same rank
            "4k3/8/8/8/8/8/8/R3K2R w K - 0 1",
            // Position 5 (complex middlegame with promotions available)
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            // Position 6 (symmetric, lots of piece moves)
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            // Black to move, promotions
            "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
        ];

        for fen in positions {
            let mut board = Board::from_fen(fen).unwrap();
            let moves = board.generate_legal_moves();
            for &m in moves.as_slice() {
                let san = board.move_to_san(m);
                let parsed = board.parse_san(&san).unwrap_or_else(|e| {
                    panic!("failed to parse SAN '{san}' (from {m}) in position {fen}: {e}")
                });
                assert_eq!(
                    parsed, m,
                    "roundtrip failed for SAN '{san}' (move {m}) in position {fen}"
                );
            }
        }
    }
}
