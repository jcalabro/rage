use std::fmt;
use std::str::FromStr;

use crate::error::{Error, Result};

// ---------------------------------------------------------------------------
// File
// ---------------------------------------------------------------------------

/// A file (column) on the chess board: A=0, B=1, ..., H=7.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct File(u8);

impl File {
    pub const NUM: usize = 8;

    pub const A: Self = Self(0);
    pub const B: Self = Self(1);
    pub const C: Self = Self(2);
    pub const D: Self = Self(3);
    pub const E: Self = Self(4);
    pub const F: Self = Self(5);
    pub const G: Self = Self(6);
    pub const H: Self = Self(7);

    pub const ALL: [Self; 8] = [
        Self::A,
        Self::B,
        Self::C,
        Self::D,
        Self::E,
        Self::F,
        Self::G,
        Self::H,
    ];

    pub const fn new(index: u8) -> Self {
        debug_assert!(index < 8);
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub const fn from_char(c: char) -> Option<Self> {
        match c {
            'a' => Some(Self::A),
            'b' => Some(Self::B),
            'c' => Some(Self::C),
            'd' => Some(Self::D),
            'e' => Some(Self::E),
            'f' => Some(Self::F),
            'g' => Some(Self::G),
            'h' => Some(Self::H),
            _ => None,
        }
    }

    pub const fn to_char(self) -> char {
        (b'a' + self.0) as char
    }
}

impl fmt::Display for File {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

// ---------------------------------------------------------------------------
// Rank
// ---------------------------------------------------------------------------

/// A rank (row) on the chess board: R1=0, R2=1, ..., R8=7.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rank(u8);

impl Rank {
    pub const NUM: usize = 8;

    pub const R1: Self = Self(0);
    pub const R2: Self = Self(1);
    pub const R3: Self = Self(2);
    pub const R4: Self = Self(3);
    pub const R5: Self = Self(4);
    pub const R6: Self = Self(5);
    pub const R7: Self = Self(6);
    pub const R8: Self = Self(7);

    pub const ALL: [Self; 8] = [
        Self::R1,
        Self::R2,
        Self::R3,
        Self::R4,
        Self::R5,
        Self::R6,
        Self::R7,
        Self::R8,
    ];

    pub const fn new(index: u8) -> Self {
        debug_assert!(index < 8);
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub const fn from_char(c: char) -> Option<Self> {
        match c {
            '1' => Some(Self::R1),
            '2' => Some(Self::R2),
            '3' => Some(Self::R3),
            '4' => Some(Self::R4),
            '5' => Some(Self::R5),
            '6' => Some(Self::R6),
            '7' => Some(Self::R7),
            '8' => Some(Self::R8),
            _ => None,
        }
    }

    pub const fn to_char(self) -> char {
        (b'1' + self.0) as char
    }
}

impl fmt::Display for Rank {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

impl fmt::Debug for Rank {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

// ---------------------------------------------------------------------------
// Square
// ---------------------------------------------------------------------------

/// A square on the chess board (0-63).
///
/// Uses Little-Endian Rank-File (LERF) mapping:
/// a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Square(u8);

impl Square {
    pub const NUM: usize = 64;

    pub const fn new(index: u8) -> Self {
        debug_assert!(index < 64);
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub const fn file(self) -> File {
        File::new(self.0 & 7)
    }

    pub const fn rank(self) -> Rank {
        Rank::new(self.0 >> 3)
    }

    pub const fn from_file_rank(file: File, rank: Rank) -> Self {
        Self(rank.0 * 8 + file.0)
    }

    /// Flip vertically (swap white/black perspective).
    pub const fn flip_rank(self) -> Self {
        Self(self.0 ^ 56)
    }

    /// Flip horizontally (mirror along vertical axis).
    pub const fn flip_file(self) -> Self {
        Self(self.0 ^ 7)
    }

    pub const ALL: [Self; 64] = {
        let mut arr = [Self(0); 64];
        let mut i = 0u8;
        while i < 64 {
            arr[i as usize] = Self(i);
            i += 1;
        }
        arr
    };

    // Named constants
    pub const A1: Self = Self(0);
    pub const B1: Self = Self(1);
    pub const C1: Self = Self(2);
    pub const D1: Self = Self(3);
    pub const E1: Self = Self(4);
    pub const F1: Self = Self(5);
    pub const G1: Self = Self(6);
    pub const H1: Self = Self(7);
    pub const A2: Self = Self(8);
    pub const B2: Self = Self(9);
    pub const C2: Self = Self(10);
    pub const D2: Self = Self(11);
    pub const E2: Self = Self(12);
    pub const F2: Self = Self(13);
    pub const G2: Self = Self(14);
    pub const H2: Self = Self(15);
    pub const A3: Self = Self(16);
    pub const B3: Self = Self(17);
    pub const C3: Self = Self(18);
    pub const D3: Self = Self(19);
    pub const E3: Self = Self(20);
    pub const F3: Self = Self(21);
    pub const G3: Self = Self(22);
    pub const H3: Self = Self(23);
    pub const A4: Self = Self(24);
    pub const B4: Self = Self(25);
    pub const C4: Self = Self(26);
    pub const D4: Self = Self(27);
    pub const E4: Self = Self(28);
    pub const F4: Self = Self(29);
    pub const G4: Self = Self(30);
    pub const H4: Self = Self(31);
    pub const A5: Self = Self(32);
    pub const B5: Self = Self(33);
    pub const C5: Self = Self(34);
    pub const D5: Self = Self(35);
    pub const E5: Self = Self(36);
    pub const F5: Self = Self(37);
    pub const G5: Self = Self(38);
    pub const H5: Self = Self(39);
    pub const A6: Self = Self(40);
    pub const B6: Self = Self(41);
    pub const C6: Self = Self(42);
    pub const D6: Self = Self(43);
    pub const E6: Self = Self(44);
    pub const F6: Self = Self(45);
    pub const G6: Self = Self(46);
    pub const H6: Self = Self(47);
    pub const A7: Self = Self(48);
    pub const B7: Self = Self(49);
    pub const C7: Self = Self(50);
    pub const D7: Self = Self(51);
    pub const E7: Self = Self(52);
    pub const F7: Self = Self(53);
    pub const G7: Self = Self(54);
    pub const H7: Self = Self(55);
    pub const A8: Self = Self(56);
    pub const B8: Self = Self(57);
    pub const C8: Self = Self(58);
    pub const D8: Self = Self(59);
    pub const E8: Self = Self(60);
    pub const F8: Self = Self(61);
    pub const G8: Self = Self(62);
    pub const H8: Self = Self(63);
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}", self.file(), self.rank())
    }
}

impl fmt::Debug for Square {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl FromStr for Square {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        let bytes = s.as_bytes();
        if bytes.len() != 2 {
            return Err(Error::InvalidSquare(s.to_string()));
        }
        let file =
            File::from_char(bytes[0] as char).ok_or_else(|| Error::InvalidSquare(s.to_string()))?;
        let rank =
            Rank::from_char(bytes[1] as char).ok_or_else(|| Error::InvalidSquare(s.to_string()))?;
        Ok(Self::from_file_rank(file, rank))
    }
}

// ---------------------------------------------------------------------------
// Color
// ---------------------------------------------------------------------------

/// A side: White or Black.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    pub const NUM: usize = 2;

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn flip(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    pub const fn from_char(c: char) -> Option<Self> {
        match c {
            'w' => Some(Color::White),
            'b' => Some(Color::Black),
            _ => None,
        }
    }

    pub const fn to_char(self) -> char {
        match self {
            Color::White => 'w',
            Color::Black => 'b',
        }
    }

    /// Pawn push direction: +8 for White, -8 for Black (as i8).
    pub const fn pawn_push(self) -> i8 {
        match self {
            Color::White => 8,
            Color::Black => -8,
        }
    }

    /// Rank where pawns start.
    pub const fn pawn_rank(self) -> Rank {
        match self {
            Color::White => Rank::R2,
            Color::Black => Rank::R7,
        }
    }

    /// Rank where pawns promote.
    pub const fn promotion_rank(self) -> Rank {
        match self {
            Color::White => Rank::R8,
            Color::Black => Rank::R1,
        }
    }

    /// Back rank (where king/rooks start).
    pub const fn back_rank(self) -> Rank {
        match self {
            Color::White => Rank::R1,
            Color::Black => Rank::R8,
        }
    }
}

impl std::ops::Not for Color {
    type Output = Self;
    fn not(self) -> Self {
        self.flip()
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

impl fmt::Debug for Color {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Color::White => write!(f, "White"),
            Color::Black => write!(f, "Black"),
        }
    }
}

// ---------------------------------------------------------------------------
// PieceType
// ---------------------------------------------------------------------------

/// A piece type (without color): Pawn, Knight, Bishop, Rook, Queen, King.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum PieceType {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl PieceType {
    pub const NUM: usize = 6;

    pub const ALL: [Self; 6] = [
        Self::Pawn,
        Self::Knight,
        Self::Bishop,
        Self::Rook,
        Self::Queen,
        Self::King,
    ];

    /// The four piece types a pawn can promote to.
    pub const PROMOTION: [Self; 4] = [Self::Knight, Self::Bishop, Self::Rook, Self::Queen];

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn to_char(self) -> char {
        match self {
            Self::Pawn => 'P',
            Self::Knight => 'N',
            Self::Bishop => 'B',
            Self::Rook => 'R',
            Self::Queen => 'Q',
            Self::King => 'K',
        }
    }

    pub const fn from_char(c: char) -> Option<Self> {
        match c {
            'P' | 'p' => Some(Self::Pawn),
            'N' | 'n' => Some(Self::Knight),
            'B' | 'b' => Some(Self::Bishop),
            'R' | 'r' => Some(Self::Rook),
            'Q' | 'q' => Some(Self::Queen),
            'K' | 'k' => Some(Self::King),
            _ => None,
        }
    }
}

impl fmt::Display for PieceType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

impl fmt::Debug for PieceType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

// ---------------------------------------------------------------------------
// Piece
// ---------------------------------------------------------------------------

/// A colored piece (e.g. White Knight, Black Queen).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Piece {
    pub color: Color,
    pub piece_type: PieceType,
}

impl Piece {
    pub const fn new(color: Color, piece_type: PieceType) -> Self {
        Self { color, piece_type }
    }

    /// Character representation: PNBRQK (white), pnbrqk (black).
    pub const fn to_char(self) -> char {
        let c = self.piece_type.to_char();
        match self.color {
            Color::White => c,
            // Upper to lower ASCII offset is 32
            Color::Black => (c as u8 + 32) as char,
        }
    }

    pub const fn from_char(c: char) -> Option<Self> {
        let color = if (c as u8) >= b'a' {
            Color::Black
        } else {
            Color::White
        };
        let piece_type = match c {
            'P' | 'p' => PieceType::Pawn,
            'N' | 'n' => PieceType::Knight,
            'B' | 'b' => PieceType::Bishop,
            'R' | 'r' => PieceType::Rook,
            'Q' | 'q' => PieceType::Queen,
            'K' | 'k' => PieceType::King,
            _ => return None,
        };
        Some(Self::new(color, piece_type))
    }
}

impl fmt::Display for Piece {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

impl fmt::Debug for Piece {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

// ---------------------------------------------------------------------------
// CastlingRights
// ---------------------------------------------------------------------------

/// Castling availability flags (4 bits: WK, WQ, BK, BQ).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CastlingRights(u8);

impl CastlingRights {
    pub const WHITE_KINGSIDE: u8 = 1;
    pub const WHITE_QUEENSIDE: u8 = 2;
    pub const BLACK_KINGSIDE: u8 = 4;
    pub const BLACK_QUEENSIDE: u8 = 8;

    pub const NONE: Self = Self(0);
    pub const ALL: Self = Self(0xF);
    pub const WHITE_BOTH: Self = Self(Self::WHITE_KINGSIDE | Self::WHITE_QUEENSIDE);
    pub const BLACK_BOTH: Self = Self(Self::BLACK_KINGSIDE | Self::BLACK_QUEENSIDE);

    pub const NUM_VARIANTS: usize = 16;

    pub const fn new(bits: u8) -> Self {
        Self(bits & 0xF)
    }

    pub const fn bits(self) -> u8 {
        self.0
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub const fn contains(self, flag: u8) -> bool {
        self.0 & flag != 0
    }

    pub const fn remove(self, flag: u8) -> Self {
        Self(self.0 & !flag)
    }

    pub const fn add(self, flag: u8) -> Self {
        Self((self.0 | flag) & 0xF)
    }

    pub const fn can_castle_kingside(self, color: Color) -> bool {
        match color {
            Color::White => self.contains(Self::WHITE_KINGSIDE),
            Color::Black => self.contains(Self::BLACK_KINGSIDE),
        }
    }

    pub const fn can_castle_queenside(self, color: Color) -> bool {
        match color {
            Color::White => self.contains(Self::WHITE_QUEENSIDE),
            Color::Black => self.contains(Self::BLACK_QUEENSIDE),
        }
    }

    /// Mask table: for each square, the castling rights that *survive* when a
    /// piece moves from or to that square.
    /// Usage: `rights = CastlingRights::new(rights.bits() & MASK[from] & MASK[to])`
    const SQUARE_MASK: [u8; Square::NUM] = {
        let mut masks = [0xFFu8; Square::NUM];
        masks[Square::A1.index()] = 0xFF ^ Self::WHITE_QUEENSIDE;
        masks[Square::E1.index()] = 0xFF ^ (Self::WHITE_KINGSIDE | Self::WHITE_QUEENSIDE);
        masks[Square::H1.index()] = 0xFF ^ Self::WHITE_KINGSIDE;
        masks[Square::A8.index()] = 0xFF ^ Self::BLACK_QUEENSIDE;
        masks[Square::E8.index()] = 0xFF ^ (Self::BLACK_KINGSIDE | Self::BLACK_QUEENSIDE);
        masks[Square::H8.index()] = 0xFF ^ Self::BLACK_KINGSIDE;
        masks
    };

    /// Update castling rights after a move from `from` to `to`.
    /// Handles king moves, rook moves, and rook captures in one branchless op.
    pub const fn update(self, from: Square, to: Square) -> Self {
        Self(self.0 & Self::SQUARE_MASK[from.index()] & Self::SQUARE_MASK[to.index()])
    }
}

impl fmt::Display for CastlingRights {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == 0 {
            return write!(f, "-");
        }
        if self.contains(Self::WHITE_KINGSIDE) {
            write!(f, "K")?;
        }
        if self.contains(Self::WHITE_QUEENSIDE) {
            write!(f, "Q")?;
        }
        if self.contains(Self::BLACK_KINGSIDE) {
            write!(f, "k")?;
        }
        if self.contains(Self::BLACK_QUEENSIDE) {
            write!(f, "q")?;
        }
        Ok(())
    }
}

impl fmt::Debug for CastlingRights {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "CastlingRights({self})")
    }
}

impl FromStr for CastlingRights {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        if s == "-" {
            return Ok(Self::NONE);
        }
        let mut rights = Self::NONE;
        for c in s.chars() {
            match c {
                'K' => rights = rights.add(Self::WHITE_KINGSIDE),
                'Q' => rights = rights.add(Self::WHITE_QUEENSIDE),
                'k' => rights = rights.add(Self::BLACK_KINGSIDE),
                'q' => rights = rights.add(Self::BLACK_QUEENSIDE),
                _ => {
                    return Err(Error::InvalidFen(format!(
                        "invalid castling character: {c}"
                    )));
                }
            }
        }
        Ok(rights)
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
    fn square_from_file_rank() {
        assert_eq!(Square::from_file_rank(File::A, Rank::R1), Square::A1);
        assert_eq!(Square::from_file_rank(File::E, Rank::R4), Square::E4);
        assert_eq!(Square::from_file_rank(File::H, Rank::R8), Square::H8);
    }

    #[test]
    fn square_file_rank_roundtrip() {
        for sq in Square::ALL {
            let file = sq.file();
            let rank = sq.rank();
            assert_eq!(Square::from_file_rank(file, rank), sq);
        }
    }

    #[test]
    fn square_display_parse_roundtrip() {
        for sq in Square::ALL {
            let s = sq.to_string();
            let parsed: Square = s.parse().unwrap();
            assert_eq!(parsed, sq);
        }
    }

    #[test]
    fn square_display() {
        assert_eq!(Square::A1.to_string(), "a1");
        assert_eq!(Square::E4.to_string(), "e4");
        assert_eq!(Square::H8.to_string(), "h8");
    }

    #[test]
    fn square_flip_rank() {
        assert_eq!(Square::A1.flip_rank(), Square::A8);
        assert_eq!(Square::E4.flip_rank(), Square::E5);
        assert_eq!(Square::H8.flip_rank(), Square::H1);
        // Double flip is identity
        for sq in Square::ALL {
            assert_eq!(sq.flip_rank().flip_rank(), sq);
        }
    }

    #[test]
    fn square_flip_file() {
        assert_eq!(Square::A1.flip_file(), Square::H1);
        assert_eq!(Square::E4.flip_file(), Square::D4);
        for sq in Square::ALL {
            assert_eq!(sq.flip_file().flip_file(), sq);
        }
    }

    #[test]
    fn square_parse_invalid() {
        assert!("".parse::<Square>().is_err());
        assert!("a".parse::<Square>().is_err());
        assert!("a9".parse::<Square>().is_err());
        assert!("i1".parse::<Square>().is_err());
        assert!("abc".parse::<Square>().is_err());
    }

    #[test]
    fn color_flip() {
        assert_eq!(!Color::White, Color::Black);
        assert_eq!(!Color::Black, Color::White);
    }

    #[test]
    fn piece_char_roundtrip() {
        let cases = [
            ('P', Color::White, PieceType::Pawn),
            ('N', Color::White, PieceType::Knight),
            ('B', Color::White, PieceType::Bishop),
            ('R', Color::White, PieceType::Rook),
            ('Q', Color::White, PieceType::Queen),
            ('K', Color::White, PieceType::King),
            ('p', Color::Black, PieceType::Pawn),
            ('n', Color::Black, PieceType::Knight),
            ('b', Color::Black, PieceType::Bishop),
            ('r', Color::Black, PieceType::Rook),
            ('q', Color::Black, PieceType::Queen),
            ('k', Color::Black, PieceType::King),
        ];
        for (ch, color, pt) in cases {
            let piece = Piece::from_char(ch).unwrap();
            assert_eq!(piece.color, color);
            assert_eq!(piece.piece_type, pt);
            assert_eq!(piece.to_char(), ch);
        }
    }

    #[test]
    fn castling_rights_display_parse() {
        let cases = [
            (CastlingRights::ALL, "KQkq"),
            (CastlingRights::NONE, "-"),
            (CastlingRights::WHITE_BOTH, "KQ"),
            (CastlingRights::BLACK_BOTH, "kq"),
            (CastlingRights::new(CastlingRights::WHITE_KINGSIDE), "K"),
        ];
        for (rights, expected) in cases {
            assert_eq!(rights.to_string(), expected);
            let parsed: CastlingRights = expected.parse().unwrap();
            assert_eq!(parsed, rights);
        }
    }

    #[test]
    fn castling_rights_update() {
        let all = CastlingRights::ALL;
        // White king moves from e1
        let after = all.update(Square::E1, Square::F1);
        assert!(!after.can_castle_kingside(Color::White));
        assert!(!after.can_castle_queenside(Color::White));
        assert!(after.can_castle_kingside(Color::Black));
        assert!(after.can_castle_queenside(Color::Black));

        // White rook moves from h1
        let after = all.update(Square::H1, Square::H3);
        assert!(!after.can_castle_kingside(Color::White));
        assert!(after.can_castle_queenside(Color::White));

        // Capture on a8 (to-square triggers mask)
        let after = all.update(Square::B6, Square::A8);
        assert!(after.can_castle_kingside(Color::Black));
        assert!(!after.can_castle_queenside(Color::Black));
    }

    #[test]
    fn file_rank_all() {
        for (i, file) in File::ALL.iter().enumerate() {
            assert_eq!(file.index(), i);
        }
        for (i, rank) in Rank::ALL.iter().enumerate() {
            assert_eq!(rank.index(), i);
        }
    }

    #[test]
    fn piecetype_all() {
        for (i, pt) in PieceType::ALL.iter().enumerate() {
            assert_eq!(pt.index(), i);
        }
    }
}
