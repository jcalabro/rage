//! Board state representation.
//!
//! The `Board` struct holds the complete state of a chess position:
//! bitboards, mailbox, side to move, castling rights, en passant, clocks, and Zobrist hash.

use std::fmt;

use crate::attacks;
use crate::bitboard::Bitboard;
use crate::types::{CastlingRights, Color, File, Piece, PieceType, Rank, Square};
use crate::zobrist;

/// Undo information saved before making a move, needed to unmake it.
#[derive(Clone, Copy, Debug)]
pub struct UndoInfo {
    pub captured_piece: Option<PieceType>,
    pub castling: CastlingRights,
    pub en_passant: Option<Square>,
    pub halfmove_clock: u16,
    pub zobrist: u64,
}

impl UndoInfo {
    const EMPTY: Self = Self {
        captured_piece: None,
        castling: CastlingRights::NONE,
        en_passant: None,
        halfmove_clock: 0,
        zobrist: 0,
    };
}

/// Maximum number of undo entries stored in the history.
const MAX_HISTORY: usize = 1024;

/// Complete chess position state.
#[derive(Clone)]
pub struct Board {
    /// Bitboards indexed by piece type.
    piece_bb: [Bitboard; PieceType::NUM],
    /// Bitboards indexed by color.
    color_bb: [Bitboard; Color::NUM],
    /// Mailbox: what piece is on each square (redundant with bitboards for fast lookup).
    mailbox: [Option<Piece>; Square::NUM],
    /// Side to move.
    side_to_move: Color,
    /// Castling rights.
    castling: CastlingRights,
    /// En passant target square (the square a pawn can capture to, not the pawn itself).
    en_passant: Option<Square>,
    /// Halfmove clock (for 50-move rule).
    halfmove_clock: u16,
    /// Fullmove number (starts at 1, incremented after Black moves).
    fullmove_number: u16,
    /// Zobrist hash of the position.
    zobrist: u64,
    /// Undo stack for unmake (fixed-size, no heap allocation).
    history: [UndoInfo; MAX_HISTORY],
    /// Number of entries in the undo stack.
    history_len: usize,
}

impl Board {
    // --- Construction ---

    /// Create an empty board with no pieces.
    pub fn empty() -> Self {
        Self {
            piece_bb: [Bitboard::EMPTY; PieceType::NUM],
            color_bb: [Bitboard::EMPTY; Color::NUM],
            mailbox: [None; Square::NUM],
            side_to_move: Color::White,
            castling: CastlingRights::NONE,
            en_passant: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            zobrist: 0,
            history: [UndoInfo::EMPTY; MAX_HISTORY],
            history_len: 0,
        }
    }

    /// Create the standard starting position.
    #[allow(clippy::unwrap_used)]
    pub fn starting_position() -> Self {
        // STARTPOS_FEN is a compile-time constant and always valid.
        Self::from_fen(Self::STARTPOS_FEN).unwrap()
    }

    // --- Piece manipulation ---

    /// Place a piece on a square. Does NOT update Zobrist (use in board setup only).
    pub fn put_piece(&mut self, piece: Piece, sq: Square) {
        let bb = Bitboard::from_square(sq);
        self.piece_bb[piece.piece_type.index()] |= bb;
        self.color_bb[piece.color.index()] |= bb;
        self.mailbox[sq.index()] = Some(piece);
    }

    /// Remove a piece from a square. Does NOT update Zobrist (use in board setup only).
    pub fn remove_piece(&mut self, sq: Square) {
        if let Some(piece) = self.mailbox[sq.index()] {
            let bb = Bitboard::from_square(sq);
            self.piece_bb[piece.piece_type.index()] &= !bb;
            self.color_bb[piece.color.index()] &= !bb;
            self.mailbox[sq.index()] = None;
        }
    }

    /// Compute the full Zobrist hash from scratch.
    pub fn compute_zobrist(&self) -> u64 {
        let mut hash = 0u64;

        for sq in Square::ALL {
            if let Some(piece) = self.mailbox[sq.index()] {
                hash ^= zobrist::piece_key(piece.color, piece.piece_type, sq);
            }
        }

        if self.side_to_move == Color::Black {
            hash ^= zobrist::side_key();
        }

        hash ^= zobrist::castling_key(self.castling);
        hash ^= zobrist::ep_key(self.en_passant.map(|sq| sq.file()));

        hash
    }

    /// Recompute and set the Zobrist hash. Call after manual board setup.
    pub fn refresh_zobrist(&mut self) {
        self.zobrist = self.compute_zobrist();
    }

    // --- Accessors ---

    pub fn piece_bb(&self, pt: PieceType) -> Bitboard {
        self.piece_bb[pt.index()]
    }

    pub fn color_bb(&self, color: Color) -> Bitboard {
        self.color_bb[color.index()]
    }

    pub fn occupied(&self) -> Bitboard {
        self.color_bb[0] | self.color_bb[1]
    }

    pub fn empty_squares(&self) -> Bitboard {
        !self.occupied()
    }

    pub fn pieces(&self, color: Color, pt: PieceType) -> Bitboard {
        self.piece_bb[pt.index()] & self.color_bb[color.index()]
    }

    pub fn piece_on(&self, sq: Square) -> Option<Piece> {
        self.mailbox[sq.index()]
    }

    pub fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    pub fn castling_rights(&self) -> CastlingRights {
        self.castling
    }

    pub fn en_passant_square(&self) -> Option<Square> {
        self.en_passant
    }

    pub fn halfmove_clock(&self) -> u16 {
        self.halfmove_clock
    }

    pub fn fullmove_number(&self) -> u16 {
        self.fullmove_number
    }

    pub fn zobrist_hash(&self) -> u64 {
        self.zobrist
    }

    /// Find the king square for the given color.
    pub fn king_square(&self, color: Color) -> Square {
        let king_bb = self.pieces(color, PieceType::King);
        // There must always be exactly one king per side
        debug_assert_eq!(king_bb.popcount(), 1);
        // lsb always returns Some when popcount == 1
        match king_bb.lsb() {
            Some(sq) => sq,
            None => Square::A1, // unreachable in valid positions
        }
    }

    /// Check if the given square is attacked by pieces of `by_color`.
    pub fn is_attacked(&self, sq: Square, by_color: Color) -> bool {
        let occupied = self.occupied();
        let them = self.color_bb(by_color);

        if (attacks::pawn_attacks(sq, !by_color) & self.piece_bb(PieceType::Pawn) & them)
            .is_not_empty()
        {
            return true;
        }
        if (attacks::knight_attacks(sq) & self.piece_bb(PieceType::Knight) & them).is_not_empty() {
            return true;
        }
        if (attacks::king_attacks(sq) & self.piece_bb(PieceType::King) & them).is_not_empty() {
            return true;
        }
        let bishop_queen =
            (self.piece_bb(PieceType::Bishop) | self.piece_bb(PieceType::Queen)) & them;
        if (attacks::bishop_attacks(sq, occupied) & bishop_queen).is_not_empty() {
            return true;
        }
        let rook_queen = (self.piece_bb(PieceType::Rook) | self.piece_bb(PieceType::Queen)) & them;
        if (attacks::rook_attacks(sq, occupied) & rook_queen).is_not_empty() {
            return true;
        }
        false
    }

    /// Check if the side to move is in check.
    pub fn in_check(&self) -> bool {
        let king_sq = self.king_square(self.side_to_move);
        self.is_attacked(king_sq, !self.side_to_move)
    }

    /// Check if a move is a capture in this position.
    pub fn is_capture(&self, m: crate::moves::Move) -> bool {
        m.is_en_passant() || self.piece_on(m.to_sq()).is_some()
    }

    // --- Draw detection ---

    /// Returns true if the 50-move rule threshold has been reached (100 half-moves).
    pub fn is_fifty_move_draw(&self) -> bool {
        self.halfmove_clock >= 100
    }

    /// Returns true if the position has insufficient material for either side to checkmate.
    ///
    /// Detected cases: K vs K, K+N vs K, K+B vs K, K+B vs K+B (same-color bishops).
    pub fn is_insufficient_material(&self) -> bool {
        if self.piece_bb(PieceType::Pawn).is_not_empty()
            || self.piece_bb(PieceType::Rook).is_not_empty()
            || self.piece_bb(PieceType::Queen).is_not_empty()
        {
            return false;
        }

        let white_knights = self.pieces(Color::White, PieceType::Knight).popcount();
        let black_knights = self.pieces(Color::Black, PieceType::Knight).popcount();
        let white_bishops = self.pieces(Color::White, PieceType::Bishop).popcount();
        let black_bishops = self.pieces(Color::Black, PieceType::Bishop).popcount();

        let white_minors = white_knights + white_bishops;
        let black_minors = black_knights + black_bishops;

        // K vs K
        if white_minors == 0 && black_minors == 0 {
            return true;
        }

        // K+minor vs K
        if (white_minors == 1 && black_minors == 0) || (white_minors == 0 && black_minors == 1) {
            return true;
        }

        // K+B vs K+B with same-color bishops
        if white_minors == 1
            && black_minors == 1
            && white_knights == 0
            && black_knights == 0
            && let (Some(wb), Some(bb)) = (
                self.pieces(Color::White, PieceType::Bishop).lsb(),
                self.pieces(Color::Black, PieceType::Bishop).lsb(),
            )
        {
            let wb_parity = (wb.file().index() + wb.rank().index()) % 2;
            let bb_parity = (bb.file().index() + bb.rank().index()) % 2;
            return wb_parity == bb_parity;
        }

        false
    }

    /// Returns true if the current position has occurred at least once before
    /// in the game history (twofold repetition).
    pub fn is_repetition(&self) -> bool {
        let limit = self.halfmove_clock as usize;
        let mut steps = 2;
        while steps <= limit && steps <= self.history_len {
            if self.history[self.history_len - steps].zobrist == self.zobrist {
                return true;
            }
            steps += 2;
        }
        false
    }

    // --- Setters (for FEN parsing) ---

    pub fn set_side_to_move(&mut self, color: Color) {
        self.side_to_move = color;
    }

    pub fn set_castling(&mut self, rights: CastlingRights) {
        self.castling = rights;
    }

    pub fn set_en_passant(&mut self, sq: Option<Square>) {
        self.en_passant = sq;
    }

    pub fn set_halfmove_clock(&mut self, clock: u16) {
        self.halfmove_clock = clock;
    }

    pub fn set_fullmove_number(&mut self, num: u16) {
        self.fullmove_number = num;
    }

    // --- Null Move ---

    /// Make a null move (pass the turn without moving a piece).
    /// Used by search for null move pruning. The position must not be in check.
    pub fn make_null_move(&mut self) {
        debug_assert!(self.history_len < MAX_HISTORY);
        self.history[self.history_len] = UndoInfo {
            captured_piece: None,
            castling: self.castling,
            en_passant: self.en_passant,
            halfmove_clock: self.halfmove_clock,
            zobrist: self.zobrist,
        };
        self.history_len += 1;

        // Remove old en passant from hash
        self.zobrist ^= zobrist::ep_key(self.en_passant.map(|sq| sq.file()));
        self.en_passant = None;
        self.zobrist ^= zobrist::ep_key(None);

        self.halfmove_clock += 1;

        // Flip side to move
        self.side_to_move = !self.side_to_move;
        self.zobrist ^= zobrist::side_key();

        if self.side_to_move == Color::White {
            self.fullmove_number += 1;
        }
    }

    /// Unmake a null move, restoring the board to its previous state.
    pub fn unmake_null_move(&mut self) {
        debug_assert!(self.history_len > 0, "unmake_null_move with empty history");
        self.history_len -= 1;
        let undo = self.history[self.history_len];

        self.side_to_move = !self.side_to_move;
        if self.side_to_move == Color::Black {
            self.fullmove_number -= 1;
        }

        self.en_passant = undo.en_passant;
        self.halfmove_clock = undo.halfmove_clock;
        self.zobrist = undo.zobrist;
    }

    /// Returns true if the side to move has any non-pawn, non-king material.
    /// Used by search for null move pruning eligibility.
    pub fn has_non_pawn_material(&self, color: Color) -> bool {
        let non_pk = self.pieces(color, PieceType::Knight)
            | self.pieces(color, PieceType::Bishop)
            | self.pieces(color, PieceType::Rook)
            | self.pieces(color, PieceType::Queen);
        non_pk.is_not_empty()
    }

    // --- Make / Unmake ---

    /// Make a move on the board. The move must be legal.
    pub fn make_move(&mut self, m: crate::moves::Move) {
        let from = m.from_sq();
        let to = m.to_sq();
        let us = self.side_to_move;

        // The piece on the from-square must exist for any legal move.
        debug_assert!(
            self.mailbox[from.index()].is_some(),
            "make_move: no piece on {from}"
        );
        let piece = match self.mailbox[from.index()] {
            Some(p) => p,
            None => return,
        };
        let pt = piece.piece_type;
        let captured = self.mailbox[to.index()].map(|p| p.piece_type);

        // Determine the captured piece type for undo (en passant is always a pawn).
        let undo_captured = if m.is_en_passant() {
            Some(PieceType::Pawn)
        } else {
            captured
        };

        // Save undo info with all fields fully populated.
        debug_assert!(self.history_len < MAX_HISTORY);
        self.history[self.history_len] = UndoInfo {
            captured_piece: undo_captured,
            castling: self.castling,
            en_passant: self.en_passant,
            halfmove_clock: self.halfmove_clock,
            zobrist: self.zobrist,
        };
        self.history_len += 1;

        // Update Zobrist: remove old en passant and castling
        self.zobrist ^= zobrist::ep_key(self.en_passant.map(|sq| sq.file()));
        self.zobrist ^= zobrist::castling_key(self.castling);

        // Reset en passant
        self.en_passant = None;

        // Halfmove clock: reset on pawn move or capture, else increment
        if pt == PieceType::Pawn || captured.is_some() {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }

        if m.is_en_passant() {
            // Remove the captured pawn
            let captured_sq = Square::from_file_rank(to.file(), from.rank());
            self.remove_piece_internal(captured_sq);

            // Move our pawn
            self.move_piece_internal(from, to, piece);
        } else if m.is_castling() {
            // Move the king
            self.move_piece_internal(from, to, piece);

            // Move the rook
            let (rook_from, rook_to) = if to.file() == File::G {
                (
                    Square::from_file_rank(File::H, from.rank()),
                    Square::from_file_rank(File::F, from.rank()),
                )
            } else {
                (
                    Square::from_file_rank(File::A, from.rank()),
                    Square::from_file_rank(File::D, from.rank()),
                )
            };
            let rook = Piece::new(us, PieceType::Rook);
            self.move_piece_internal(rook_from, rook_to, rook);
        } else if m.is_promotion() {
            // Remove captured piece if any
            if captured.is_some() {
                self.remove_piece_internal(to);
            }

            // Remove the pawn from `from`
            self.remove_piece_internal(from);

            // Place the promoted piece on `to`
            let promo_piece = Piece::new(us, m.promotion_piece());
            self.put_piece_internal(to, promo_piece);
        } else {
            // Normal move or capture
            if captured.is_some() {
                self.remove_piece_internal(to);
            }
            self.move_piece_internal(from, to, piece);

            // Double pawn push: set en passant square only if an enemy pawn
            // can actually capture (strict per FIDE rules / improves TT hits).
            if pt == PieceType::Pawn
                && ((from.rank() == Rank::R2 && to.rank() == Rank::R4)
                    || (from.rank() == Rank::R7 && to.rank() == Rank::R5))
            {
                let ep_rank = if us == Color::White {
                    Rank::R3
                } else {
                    Rank::R6
                };
                let ep_sq = Square::from_file_rank(from.file(), ep_rank);
                // Only record EP if an enemy pawn can actually capture on the EP square.
                let enemy_pawns = self.pieces(!us, PieceType::Pawn);
                if (attacks::pawn_attacks(ep_sq, us) & enemy_pawns).is_not_empty() {
                    self.en_passant = Some(ep_sq);
                }
            }
        }

        // Update castling rights
        self.castling = self.castling.update(from, to);

        // Update Zobrist: add new en passant and castling
        self.zobrist ^= zobrist::ep_key(self.en_passant.map(|sq| sq.file()));
        self.zobrist ^= zobrist::castling_key(self.castling);

        // Flip side to move
        self.side_to_move = !self.side_to_move;
        self.zobrist ^= zobrist::side_key();

        // Fullmove number: increment after Black moves
        if self.side_to_move == Color::White {
            self.fullmove_number += 1;
        }
    }

    /// Unmake a move, restoring the board to its previous state.
    pub fn unmake_move(&mut self, m: crate::moves::Move) {
        if self.history_len == 0 {
            return;
        }
        self.history_len -= 1;
        let undo = self.history[self.history_len];

        // Flip side back
        self.side_to_move = !self.side_to_move;
        if self.side_to_move == Color::Black {
            self.fullmove_number -= 1;
        }

        let from = m.from_sq();
        let to = m.to_sq();
        let us = self.side_to_move;

        if m.is_en_passant() {
            // Move our pawn back
            let pawn = Piece::new(us, PieceType::Pawn);
            self.move_piece_internal(to, from, pawn);

            // Restore captured pawn
            let captured_sq = Square::from_file_rank(to.file(), from.rank());
            self.put_piece_internal(captured_sq, Piece::new(!us, PieceType::Pawn));
        } else if m.is_castling() {
            // Move king back
            let king = Piece::new(us, PieceType::King);
            self.move_piece_internal(to, from, king);

            // Move rook back
            let (rook_from, rook_to) = if to.file() == File::G {
                (
                    Square::from_file_rank(File::H, from.rank()),
                    Square::from_file_rank(File::F, from.rank()),
                )
            } else {
                (
                    Square::from_file_rank(File::A, from.rank()),
                    Square::from_file_rank(File::D, from.rank()),
                )
            };
            let rook = Piece::new(us, PieceType::Rook);
            self.move_piece_internal(rook_to, rook_from, rook);
        } else if m.is_promotion() {
            // Remove promoted piece from `to`
            self.remove_piece_internal(to);

            // Put pawn back on `from`
            self.put_piece_internal(from, Piece::new(us, PieceType::Pawn));

            // Restore captured piece on `to` if any
            if let Some(cap_pt) = undo.captured_piece {
                self.put_piece_internal(to, Piece::new(!us, cap_pt));
            }
        } else {
            // Normal move: move piece back
            let piece = match self.mailbox[to.index()] {
                Some(p) => p,
                None => return,
            };
            self.move_piece_internal(to, from, piece);

            // Restore captured piece
            if let Some(cap_pt) = undo.captured_piece {
                self.put_piece_internal(to, Piece::new(!us, cap_pt));
            }
        }

        // Restore state from undo
        self.castling = undo.castling;
        self.en_passant = undo.en_passant;
        self.halfmove_clock = undo.halfmove_clock;
        self.zobrist = undo.zobrist;
    }

    // --- Internal helpers for piece manipulation ---

    /// Place a piece on a square, updating bitboards, mailbox, and Zobrist.
    fn put_piece_internal(&mut self, sq: Square, piece: Piece) {
        let bb = Bitboard::from_square(sq);
        self.piece_bb[piece.piece_type.index()] |= bb;
        self.color_bb[piece.color.index()] |= bb;
        self.mailbox[sq.index()] = Some(piece);
        self.zobrist ^= zobrist::piece_key(piece.color, piece.piece_type, sq);
    }

    /// Remove whatever piece is on a square, updating bitboards, mailbox, and Zobrist.
    fn remove_piece_internal(&mut self, sq: Square) {
        if let Some(piece) = self.mailbox[sq.index()] {
            let bb = Bitboard::from_square(sq);
            self.piece_bb[piece.piece_type.index()] &= !bb;
            self.color_bb[piece.color.index()] &= !bb;
            self.mailbox[sq.index()] = None;
            self.zobrist ^= zobrist::piece_key(piece.color, piece.piece_type, sq);
        }
    }

    /// Move a piece from one square to another, updating bitboards, mailbox, and Zobrist.
    fn move_piece_internal(&mut self, from: Square, to: Square, piece: Piece) {
        let from_to = Bitboard::from_square(from) | Bitboard::from_square(to);
        self.piece_bb[piece.piece_type.index()] ^= from_to;
        self.color_bb[piece.color.index()] ^= from_to;
        self.mailbox[from.index()] = None;
        self.mailbox[to.index()] = Some(piece);
        self.zobrist ^= zobrist::piece_key(piece.color, piece.piece_type, from);
        self.zobrist ^= zobrist::piece_key(piece.color, piece.piece_type, to);
    }
}

// --- FEN parsing and serialization ---

impl Board {
    /// Standard starting position FEN.
    pub const STARTPOS_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    /// Parse a FEN string into a Board.
    pub fn from_fen(fen: &str) -> crate::Result<Self> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(crate::Error::InvalidFen(
                "FEN must have at least 4 fields".to_string(),
            ));
        }

        let mut board = Board::empty();

        // 1. Piece placement
        let ranks: Vec<&str> = parts[0].split('/').collect();
        if ranks.len() != 8 {
            return Err(crate::Error::InvalidFen(format!(
                "expected 8 ranks, got {}",
                ranks.len()
            )));
        }
        for (rank_idx, rank_str) in ranks.iter().rev().enumerate() {
            let mut file = 0i32;
            for ch in rank_str.chars() {
                match ch {
                    '1'..='8' => {
                        file += (ch as i32) - ('0' as i32);
                    }
                    _ => {
                        if file >= 8 {
                            return Err(crate::Error::InvalidFen(format!(
                                "too many files on rank {}",
                                rank_idx + 1
                            )));
                        }
                        let piece = Piece::from_char(ch).ok_or_else(|| {
                            crate::Error::InvalidFen(format!("invalid piece: {ch}"))
                        })?;
                        let sq = Square::from_file_rank(
                            File::new(file as u8),
                            Rank::new(rank_idx as u8),
                        );
                        board.put_piece(piece, sq);
                        file += 1;
                    }
                }
            }
            if file != 8 {
                return Err(crate::Error::InvalidFen(format!(
                    "rank {} has {file} files, expected 8",
                    rank_idx + 1
                )));
            }
        }

        // 2. Side to move
        board.side_to_move = match parts[1] {
            "w" => Color::White,
            "b" => Color::Black,
            _ => {
                return Err(crate::Error::InvalidFen(format!(
                    "invalid side to move: {}",
                    parts[1]
                )));
            }
        };

        // 3. Castling rights
        board.castling = parts[2].parse()?;

        // 4. En passant
        if parts[3] == "-" {
            board.en_passant = None;
        } else {
            board.en_passant = Some(parts[3].parse()?);
        }

        // 5. Halfmove clock (optional)
        if parts.len() > 4 {
            board.halfmove_clock = parts[4]
                .parse::<u16>()
                .map_err(|e| crate::Error::InvalidFen(format!("invalid halfmove clock: {e}")))?;
        }

        // 6. Fullmove number (optional)
        if parts.len() > 5 {
            board.fullmove_number = parts[5]
                .parse::<u16>()
                .map_err(|e| crate::Error::InvalidFen(format!("invalid fullmove number: {e}")))?;
        }

        board.refresh_zobrist();
        Ok(board)
    }

    /// Serialize the board to a FEN string.
    pub fn to_fen(&self) -> String {
        let mut fen = String::new();

        // 1. Piece placement
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let sq = Square::from_file_rank(File::new(file), Rank::new(rank));
                match self.mailbox[sq.index()] {
                    Some(piece) => {
                        if empty > 0 {
                            fen.push(char::from(b'0' + empty));
                            empty = 0;
                        }
                        fen.push(piece.to_char());
                    }
                    None => {
                        empty += 1;
                    }
                }
            }
            if empty > 0 {
                fen.push(char::from(b'0' + empty));
            }
            if rank > 0 {
                fen.push('/');
            }
        }

        // 2. Side to move
        fen.push(' ');
        fen.push(self.side_to_move.to_char());

        // 3. Castling rights
        fen.push(' ');
        fen.push_str(&self.castling.to_string());

        // 4. En passant
        fen.push(' ');
        match self.en_passant {
            Some(sq) => fen.push_str(&sq.to_string()),
            None => fen.push('-'),
        }

        // 5. Halfmove clock
        fen.push(' ');
        fen.push_str(&self.halfmove_clock.to_string());

        // 6. Fullmove number
        fen.push(' ');
        fen.push_str(&self.fullmove_number.to_string());

        fen
    }
}

/// Display the board in a human-readable ASCII format.
impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "  a b c d e f g h")?;
        for rank in (0..8).rev() {
            write!(f, "{} ", rank + 1)?;
            for file in 0..8 {
                let sq = Square::from_file_rank(File::new(file), Rank::new(rank));
                match self.mailbox[sq.index()] {
                    Some(piece) => write!(f, "{} ", piece.to_char())?,
                    None => write!(f, ". ")?,
                }
            }
            writeln!(f, "{}", rank + 1)?;
        }
        writeln!(f, "  a b c d e f g h")?;
        writeln!(
            f,
            "\nSide: {} | Castling: {} | EP: {} | HM: {} | FM: {}",
            self.side_to_move,
            self.castling,
            self.en_passant.map_or("-".to_string(), |sq| sq.to_string()),
            self.halfmove_clock,
            self.fullmove_number,
        )?;
        writeln!(f, "Zobrist: 0x{:016x}", self.zobrist)?;
        Ok(())
    }
}

impl fmt::Debug for Board {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Board({})", self.to_fen())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn starting_position_fen_roundtrip() {
        let board = Board::from_fen(Board::STARTPOS_FEN).unwrap();
        assert_eq!(board.to_fen(), Board::STARTPOS_FEN);
    }

    #[test]
    fn starting_position_pieces() {
        let board = Board::starting_position();
        // White has 8 pawns
        assert_eq!(board.pieces(Color::White, PieceType::Pawn).popcount(), 8);
        // Black has 8 pawns
        assert_eq!(board.pieces(Color::Black, PieceType::Pawn).popcount(), 8);
        // 2 knights each
        assert_eq!(board.pieces(Color::White, PieceType::Knight).popcount(), 2);
        assert_eq!(board.pieces(Color::Black, PieceType::Knight).popcount(), 2);
        // Kings
        assert_eq!(board.king_square(Color::White), Square::E1);
        assert_eq!(board.king_square(Color::Black), Square::E8);
        // 32 total pieces
        assert_eq!(board.occupied().popcount(), 32);
        // White to move
        assert_eq!(board.side_to_move(), Color::White);
        // All castling rights
        assert_eq!(board.castling_rights(), CastlingRights::ALL);
    }

    #[test]
    fn fen_with_en_passant() {
        let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
        let board = Board::from_fen(fen).unwrap();
        assert_eq!(board.en_passant_square(), Some(Square::E3));
        assert_eq!(board.side_to_move(), Color::Black);
        assert_eq!(board.to_fen(), fen);
    }

    #[test]
    fn fen_no_castling() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1";
        let board = Board::from_fen(fen).unwrap();
        assert_eq!(board.castling_rights(), CastlingRights::NONE);
        assert_eq!(board.to_fen(), fen);
    }

    #[test]
    fn fen_complex_position() {
        // Kiwipete position
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let board = Board::from_fen(fen).unwrap();
        assert_eq!(board.to_fen(), fen);
    }

    #[test]
    fn zobrist_consistency() {
        let board = Board::starting_position();
        assert_eq!(board.zobrist_hash(), board.compute_zobrist());
    }

    #[test]
    fn zobrist_different_positions() {
        let board1 = Board::starting_position();
        let board2 =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1").unwrap();
        assert_ne!(board1.zobrist_hash(), board2.zobrist_hash());
    }

    #[test]
    fn in_check_detection() {
        // White king on e1 with black rook on e8, no blockers on e-file
        let board = Board::from_fen("4r3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert!(board.in_check());

        // Starting position - not in check
        let board = Board::starting_position();
        assert!(!board.in_check());
    }

    #[test]
    fn mailbox_bitboard_consistency() {
        let board = Board::starting_position();
        for sq in Square::ALL {
            let has_piece = board.piece_on(sq).is_some();
            let in_occupied = board.occupied().contains(sq);
            assert_eq!(has_piece, in_occupied, "inconsistency on {sq}");

            if let Some(piece) = board.piece_on(sq) {
                assert!(
                    board.pieces(piece.color, piece.piece_type).contains(sq),
                    "piece {piece} on {sq} not in correct bitboard"
                );
            }
        }
    }

    #[test]
    fn invalid_fen_rejected() {
        assert!(Board::from_fen("").is_err());
        assert!(Board::from_fen("not a fen").is_err());
        assert!(Board::from_fen("8/8/8/8/8/8/8/8 x KQkq - 0 1").is_err());
    }

    #[test]
    fn invalid_fen_wrong_rank_count() {
        // Only 7 ranks
        assert!(Board::from_fen("8/8/8/8/8/8/8 w - - 0 1").is_err());
        // 9 ranks
        assert!(Board::from_fen("8/8/8/8/8/8/8/8/8 w - - 0 1").is_err());
    }

    #[test]
    fn invalid_fen_wrong_file_count() {
        // Rank with only 7 files
        assert!(Board::from_fen("8/8/8/8/8/8/8/7 w - - 0 1").is_err());
        // Rank with 9 files
        assert!(Board::from_fen("8/8/8/8/8/8/8/PPPPPPPPP w - - 0 1").is_err());
    }

    #[test]
    fn invalid_fen_bad_piece_char() {
        assert!(Board::from_fen("8/8/8/8/8/8/8/X7 w - - 0 1").is_err());
    }

    #[test]
    fn invalid_fen_bad_castling() {
        assert!(Board::from_fen("8/8/8/8/8/8/8/8 w Z - 0 1").is_err());
    }

    #[test]
    fn empty_board_defaults() {
        let board = Board::empty();
        assert_eq!(board.occupied().popcount(), 0);
        assert_eq!(board.side_to_move(), Color::White);
        assert_eq!(board.castling_rights(), CastlingRights::NONE);
        assert_eq!(board.en_passant_square(), None);
        assert_eq!(board.halfmove_clock(), 0);
        assert_eq!(board.fullmove_number(), 1);
        for sq in Square::ALL {
            assert!(board.piece_on(sq).is_none());
        }
    }

    #[test]
    fn is_capture_detection() {
        let board = Board::from_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
            .unwrap();
        // e4xd5 is a capture
        let capture = crate::moves::Move::new(Square::E4, Square::D5);
        assert!(board.is_capture(capture));
        // Nc3 is not a capture
        let quiet = crate::moves::Move::new(Square::E4, Square::E5);
        assert!(!board.is_capture(quiet));
    }

    #[test]
    fn is_capture_en_passant() {
        let board =
            Board::from_fen("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
                .unwrap();
        let ep = crate::moves::Move::new_en_passant(Square::E5, Square::F6);
        assert!(board.is_capture(ep));
    }

    #[test]
    fn fifty_move_draw() {
        let board = Board::from_fen("8/8/8/4k3/8/8/8/4K3 w - - 100 200").unwrap();
        assert!(board.is_fifty_move_draw());

        let board = Board::from_fen("8/8/8/4k3/8/8/8/4K3 w - - 99 200").unwrap();
        assert!(!board.is_fifty_move_draw());
    }

    #[test]
    fn insufficient_material_k_vs_k() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert!(board.is_insufficient_material());
    }

    #[test]
    fn insufficient_material_kn_vs_k() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/3NK3 w - - 0 1").unwrap();
        assert!(board.is_insufficient_material());
        let board = Board::from_fen("3nk3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert!(board.is_insufficient_material());
    }

    #[test]
    fn insufficient_material_kb_vs_k() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/3BK3 w - - 0 1").unwrap();
        assert!(board.is_insufficient_material());
    }

    #[test]
    fn insufficient_material_kb_vs_kb_same_color() {
        // Both bishops on dark squares (c1 and f8)
        let board = Board::from_fen("5b2/8/8/8/8/8/8/2B1K2k w - - 0 1").unwrap();
        assert!(board.is_insufficient_material());
    }

    #[test]
    fn sufficient_material_kb_vs_kb_diff_color() {
        // White bishop on dark (c1), black bishop on light (c8)
        let board = Board::from_fen("2b1k3/8/8/8/8/8/8/2B1K3 w - - 0 1").unwrap();
        assert!(!board.is_insufficient_material());
    }

    #[test]
    fn sufficient_material_with_pawns() {
        let board = Board::from_fen("4k3/p7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert!(!board.is_insufficient_material());
    }

    #[test]
    fn sufficient_material_with_rook() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
        assert!(!board.is_insufficient_material());
    }

    #[test]
    fn repetition_detection() {
        let mut board = Board::starting_position();
        // Nf3 Nf6 Ng1 Ng8 → back to starting position
        let nf3 = crate::moves::Move::new(Square::G1, Square::F3);
        let nf6 = crate::moves::Move::new(Square::G8, Square::F6);
        let ng1 = crate::moves::Move::new(Square::F3, Square::G1);
        let ng8 = crate::moves::Move::new(Square::F6, Square::G8);

        board.make_move(nf3);
        board.make_move(nf6);
        board.make_move(ng1);
        board.make_move(ng8);

        assert!(board.is_repetition());
    }

    #[test]
    fn no_repetition_at_start() {
        let board = Board::starting_position();
        assert!(!board.is_repetition());
    }

    // Halfmove clock: resets on pawn move, resets on capture, increments otherwise.
    #[test]
    fn halfmove_clock_behavior() {
        let mut board = Board::starting_position();
        assert_eq!(board.halfmove_clock(), 0);

        // Knight move increments clock
        let nf3 = crate::moves::Move::new(Square::G1, Square::F3);
        board.make_move(nf3);
        assert_eq!(board.halfmove_clock(), 1);

        // Another knight move increments again
        let nc6 = crate::moves::Move::new(Square::B8, Square::C6);
        board.make_move(nc6);
        assert_eq!(board.halfmove_clock(), 2);

        // Pawn move resets clock
        let e4 = crate::moves::Move::new(Square::E2, Square::E4);
        board.make_move(e4);
        assert_eq!(board.halfmove_clock(), 0);
    }

    // Halfmove clock resets on captures too.
    #[test]
    fn halfmove_clock_resets_on_capture() {
        // Position with a capture available and some moves already made (clock > 0).
        let mut board =
            Board::from_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 4 3")
                .unwrap();
        assert_eq!(board.halfmove_clock(), 4);
        // exd5 is a capture
        let exd5 = crate::moves::Move::new(Square::E4, Square::D5);
        board.make_move(exd5);
        assert_eq!(board.halfmove_clock(), 0);
    }

    // No repetition when castling rights differ: moving the king and back
    // loses castling rights, so the position hash differs from the original.
    #[test]
    fn no_repetition_different_castling_rights() {
        let mut board =
            Board::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1").unwrap();
        let original_hash = board.zobrist_hash();

        // Ke2 (loses white castling rights), then random black move, then Ke1
        let ke2 = crate::moves::Move::new(Square::E1, Square::E2);
        let a6 = crate::moves::Move::new(Square::A7, Square::A6);
        let ke1 = crate::moves::Move::new(Square::E2, Square::E1);
        let a5 = crate::moves::Move::new(Square::A6, Square::A5);

        board.make_move(ke2);
        board.make_move(a6);
        board.make_move(ke1);
        board.make_move(a5);

        // Pieces are in different places (a5 pawn) AND castling rights differ.
        // Hash must differ from original.
        assert_ne!(board.zobrist_hash(), original_hash);
        assert!(!board.is_repetition());
    }

    // Strict en passant: double push only records EP if enemy pawn can capture
    #[test]
    fn strict_en_passant_recording() {
        // Starting position: 1.e4 should NOT record en passant (no black pawn on d4/f4)
        let mut board = Board::starting_position();
        let e2e4 = crate::moves::Move::new(Square::E2, Square::E4);
        board.make_move(e2e4);
        assert_eq!(
            board.en_passant_square(),
            None,
            "EP should not be set when no enemy pawn can capture"
        );

        // Position where EP should be recorded (black pawn on d4, white plays e2e4)
        let mut board =
            Board::from_fen("rnbqkbnr/ppp1pppp/8/8/3p4/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        board.make_move(e2e4);
        assert_eq!(
            board.en_passant_square(),
            Some(Square::E3),
            "EP should be set when enemy pawn can capture"
        );
    }

    #[test]
    fn null_move_flips_side() {
        let mut board = Board::starting_position();
        assert_eq!(board.side_to_move(), Color::White);
        board.make_null_move();
        assert_eq!(board.side_to_move(), Color::Black);
        board.unmake_null_move();
        assert_eq!(board.side_to_move(), Color::White);
    }

    #[test]
    fn null_move_zobrist_consistency() {
        let mut board = Board::starting_position();
        let original_hash = board.zobrist_hash();
        board.make_null_move();
        // Hash should differ after null move
        assert_ne!(board.zobrist_hash(), original_hash);
        // Incremental hash should match full recomputation
        assert_eq!(board.zobrist_hash(), board.compute_zobrist());
        board.unmake_null_move();
        // Hash should be restored
        assert_eq!(board.zobrist_hash(), original_hash);
    }

    #[test]
    fn null_move_clears_en_passant() {
        let mut board =
            Board::from_fen("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
                .unwrap();
        assert!(board.en_passant_square().is_some());
        board.make_null_move();
        assert_eq!(board.en_passant_square(), None);
        assert_eq!(board.zobrist_hash(), board.compute_zobrist());
        board.unmake_null_move();
        assert!(board.en_passant_square().is_some());
    }

    #[test]
    fn null_move_preserves_pieces() {
        let mut board = Board::starting_position();
        let occ_before = board.occupied().popcount();
        board.make_null_move();
        assert_eq!(board.occupied().popcount(), occ_before);
        board.unmake_null_move();
        assert_eq!(board.occupied().popcount(), occ_before);
    }

    #[test]
    fn has_non_pawn_material_starting() {
        let board = Board::starting_position();
        assert!(board.has_non_pawn_material(Color::White));
        assert!(board.has_non_pawn_material(Color::Black));
    }

    #[test]
    fn has_non_pawn_material_k_vs_k() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert!(!board.has_non_pawn_material(Color::White));
        assert!(!board.has_non_pawn_material(Color::Black));
    }

    #[test]
    fn has_non_pawn_material_kp_vs_kp() {
        let board = Board::from_fen("4k3/4p3/8/8/8/8/4P3/4K3 w - - 0 1").unwrap();
        assert!(!board.has_non_pawn_material(Color::White));
        assert!(!board.has_non_pawn_material(Color::Black));
    }
}
