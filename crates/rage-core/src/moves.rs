use std::fmt;

use crate::types::{PieceType, Square};

/// A compact 16-bit move encoding.
///
/// ```text
/// Bits 0-5:   From square (0-63)
/// Bits 6-11:  To square (0-63)
/// Bits 12-13: Promotion piece (0=Knight, 1=Bishop, 2=Rook, 3=Queen)
/// Bits 14-15: Flags (0=normal, 1=promotion, 2=en passant, 3=castling)
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Move(u16);

impl Move {
    // Flag constants
    const FLAG_NORMAL: u16 = 0;
    const FLAG_PROMOTION: u16 = 1 << 14;
    const FLAG_EN_PASSANT: u16 = 2 << 14;
    const FLAG_CASTLING: u16 = 3 << 14;
    const FLAG_MASK: u16 = 3 << 14;

    // Promotion piece encoding (bits 12-13)
    const PROMO_KNIGHT: u16 = 0 << 12;
    const PROMO_BISHOP: u16 = 1 << 12;
    const PROMO_ROOK: u16 = 2 << 12;
    const PROMO_QUEEN: u16 = 3 << 12;

    /// Lookup table for decoding promotion piece from bits 12-13.
    const PROMO_PIECES: [PieceType; 4] = [
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
    ];

    /// The null/empty move (all zeros). Used as a sentinel.
    pub const NULL: Self = Self(0);

    // --- Construction ---

    /// Normal move (non-capture, non-special). Also used for regular captures.
    pub const fn new(from: Square, to: Square) -> Self {
        Self((from.index() as u16) | ((to.index() as u16) << 6) | Self::FLAG_NORMAL)
    }

    /// Promotion move. `promo` must be Knight, Bishop, Rook, or Queen.
    pub const fn new_promotion(from: Square, to: Square, promo: PieceType) -> Self {
        let promo_bits = match promo {
            PieceType::Knight => Self::PROMO_KNIGHT,
            PieceType::Bishop => Self::PROMO_BISHOP,
            PieceType::Rook => Self::PROMO_ROOK,
            PieceType::Queen => Self::PROMO_QUEEN,
            // Pawn and King can't be promotion targets; encode as Knight (caller error)
            _ => Self::PROMO_KNIGHT,
        };
        Self((from.index() as u16) | ((to.index() as u16) << 6) | promo_bits | Self::FLAG_PROMOTION)
    }

    /// En passant capture.
    pub const fn new_en_passant(from: Square, to: Square) -> Self {
        Self((from.index() as u16) | ((to.index() as u16) << 6) | Self::FLAG_EN_PASSANT)
    }

    /// Castling move (king moves to its castling destination).
    pub const fn new_castling(from: Square, to: Square) -> Self {
        Self((from.index() as u16) | ((to.index() as u16) << 6) | Self::FLAG_CASTLING)
    }

    // --- Accessors ---

    pub const fn from_sq(self) -> Square {
        Square::new((self.0 & 0x3F) as u8)
    }

    pub const fn to_sq(self) -> Square {
        Square::new(((self.0 >> 6) & 0x3F) as u8)
    }

    pub const fn is_promotion(self) -> bool {
        self.0 & Self::FLAG_MASK == Self::FLAG_PROMOTION
    }

    pub const fn is_en_passant(self) -> bool {
        self.0 & Self::FLAG_MASK == Self::FLAG_EN_PASSANT
    }

    pub const fn is_castling(self) -> bool {
        self.0 & Self::FLAG_MASK == Self::FLAG_CASTLING
    }

    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    /// Returns the promotion piece type. Only valid when `is_promotion()` is true.
    pub const fn promotion_piece(self) -> PieceType {
        Self::PROMO_PIECES[((self.0 >> 12) & 3) as usize]
    }

    pub const fn raw(self) -> u16 {
        self.0
    }
}

/// Display as UCI long algebraic notation (e.g. "e2e4", "e7e8q").
impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_null() {
            return write!(f, "0000");
        }
        write!(f, "{}{}", self.from_sq(), self.to_sq())?;
        if self.is_promotion() {
            let ch = match self.promotion_piece() {
                PieceType::Knight => 'n',
                PieceType::Bishop => 'b',
                PieceType::Rook => 'r',
                PieceType::Queen => 'q',
                _ => '?',
            };
            write!(f, "{ch}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Move({self})")
    }
}

// ---------------------------------------------------------------------------
// MoveList - stack-allocated move buffer
// ---------------------------------------------------------------------------

/// A stack-allocated list of moves. Maximum capacity of 256, which is more than
/// enough for any legal chess position (the maximum is ~218).
pub struct MoveList {
    moves: [Move; Self::MAX_MOVES],
    len: usize,
}

impl MoveList {
    pub const MAX_MOVES: usize = 256;

    pub const fn new() -> Self {
        Self {
            moves: [Move::NULL; Self::MAX_MOVES],
            len: 0,
        }
    }

    pub fn push(&mut self, m: Move) {
        debug_assert!(self.len < Self::MAX_MOVES);
        self.moves[self.len] = m;
        self.len += 1;
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }

    pub fn iter(&self) -> impl Iterator<Item = &Move> {
        self.moves[..self.len].iter()
    }

    pub fn contains(&self, m: Move) -> bool {
        self.moves[..self.len].contains(&m)
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_move() {
        let m = Move::new(Square::E2, Square::E4);
        assert_eq!(m.from_sq(), Square::E2);
        assert_eq!(m.to_sq(), Square::E4);
        assert!(!m.is_promotion());
        assert!(!m.is_en_passant());
        assert!(!m.is_castling());
        assert!(!m.is_null());
        assert_eq!(m.to_string(), "e2e4");
    }

    #[test]
    fn promotion_move() {
        let m = Move::new_promotion(Square::E7, Square::E8, PieceType::Queen);
        assert_eq!(m.from_sq(), Square::E7);
        assert_eq!(m.to_sq(), Square::E8);
        assert!(m.is_promotion());
        assert_eq!(m.promotion_piece(), PieceType::Queen);
        assert_eq!(m.to_string(), "e7e8q");
    }

    #[test]
    fn en_passant_move() {
        let m = Move::new_en_passant(Square::E5, Square::D6);
        assert_eq!(m.from_sq(), Square::E5);
        assert_eq!(m.to_sq(), Square::D6);
        assert!(m.is_en_passant());
        assert!(!m.is_promotion());
        assert_eq!(m.to_string(), "e5d6");
    }

    #[test]
    fn castling_move() {
        let m = Move::new_castling(Square::E1, Square::G1);
        assert_eq!(m.from_sq(), Square::E1);
        assert_eq!(m.to_sq(), Square::G1);
        assert!(m.is_castling());
        assert_eq!(m.to_string(), "e1g1");
    }

    #[test]
    fn null_move() {
        assert!(Move::NULL.is_null());
        assert_eq!(Move::NULL.to_string(), "0000");
    }

    #[test]
    fn all_promotion_pieces() {
        for &pt in &PieceType::PROMOTION {
            let m = Move::new_promotion(Square::A7, Square::A8, pt);
            assert!(m.is_promotion());
            assert_eq!(m.promotion_piece(), pt);
        }
    }

    #[test]
    fn movelist_basic() {
        let mut list = MoveList::new();
        assert!(list.is_empty());

        list.push(Move::new(Square::E2, Square::E4));
        list.push(Move::new(Square::D2, Square::D4));
        assert_eq!(list.len(), 2);
        assert!(list.contains(Move::new(Square::E2, Square::E4)));
        assert!(!list.contains(Move::new(Square::A2, Square::A4)));
    }
}
