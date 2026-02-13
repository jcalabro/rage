use std::fmt;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

use crate::types::Square;

/// A 64-bit bitboard representing a set of squares on the chess board.
///
/// Each bit corresponds to a square: bit 0 = a1, bit 1 = b1, ..., bit 63 = h8.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Bitboard(pub u64);

impl Bitboard {
    pub const EMPTY: Self = Self(0);
    pub const FULL: Self = Self(!0);

    // File constants
    pub const FILE_A: Self = Self(0x0101_0101_0101_0101);
    pub const FILE_B: Self = Self(0x0202_0202_0202_0202);
    pub const FILE_C: Self = Self(0x0404_0404_0404_0404);
    pub const FILE_D: Self = Self(0x0808_0808_0808_0808);
    pub const FILE_E: Self = Self(0x1010_1010_1010_1010);
    pub const FILE_F: Self = Self(0x2020_2020_2020_2020);
    pub const FILE_G: Self = Self(0x4040_4040_4040_4040);
    pub const FILE_H: Self = Self(0x8080_8080_8080_8080);

    pub const FILES: [Self; 8] = [
        Self::FILE_A,
        Self::FILE_B,
        Self::FILE_C,
        Self::FILE_D,
        Self::FILE_E,
        Self::FILE_F,
        Self::FILE_G,
        Self::FILE_H,
    ];

    // Rank constants
    pub const RANK_1: Self = Self(0x0000_0000_0000_00FF);
    pub const RANK_2: Self = Self(0x0000_0000_0000_FF00);
    pub const RANK_3: Self = Self(0x0000_0000_00FF_0000);
    pub const RANK_4: Self = Self(0x0000_0000_FF00_0000);
    pub const RANK_5: Self = Self(0x0000_00FF_0000_0000);
    pub const RANK_6: Self = Self(0x0000_FF00_0000_0000);
    pub const RANK_7: Self = Self(0x00FF_0000_0000_0000);
    pub const RANK_8: Self = Self(0xFF00_0000_0000_0000);

    pub const RANKS: [Self; 8] = [
        Self::RANK_1,
        Self::RANK_2,
        Self::RANK_3,
        Self::RANK_4,
        Self::RANK_5,
        Self::RANK_6,
        Self::RANK_7,
        Self::RANK_8,
    ];

    // --- Construction ---

    pub const fn new(bits: u64) -> Self {
        Self(bits)
    }

    pub const fn from_square(sq: Square) -> Self {
        Self(1u64 << sq.index())
    }

    // --- Queries ---

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn is_not_empty(self) -> bool {
        self.0 != 0
    }

    pub const fn contains(self, sq: Square) -> bool {
        self.0 & (1u64 << sq.index()) != 0
    }

    pub const fn popcount(self) -> u32 {
        self.0.count_ones()
    }

    /// Returns the index of the least significant set bit, or None if empty.
    pub const fn lsb(self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            Some(Square::new(self.0.trailing_zeros() as u8))
        }
    }

    /// Returns the index of the most significant set bit, or None if empty.
    pub const fn msb(self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            Some(Square::new(63 - self.0.leading_zeros() as u8))
        }
    }

    /// Pop and return the least significant set bit.
    pub fn pop_lsb(&mut self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            let sq = self.0.trailing_zeros() as u8;
            self.0 &= self.0 - 1; // clear LSB
            Some(Square::new(sq))
        }
    }

    // --- Mutation ---

    pub const fn set(self, sq: Square) -> Self {
        Self(self.0 | (1u64 << sq.index()))
    }

    pub const fn clear(self, sq: Square) -> Self {
        Self(self.0 & !(1u64 << sq.index()))
    }

    pub const fn toggle(self, sq: Square) -> Self {
        Self(self.0 ^ (1u64 << sq.index()))
    }

    // --- Directional shifts ---
    // These shifts mask off the wrapping file to prevent pieces from
    // teleporting across the board.

    pub const fn north(self) -> Self {
        Self(self.0 << 8)
    }

    pub const fn south(self) -> Self {
        Self(self.0 >> 8)
    }

    pub const fn east(self) -> Self {
        Self((self.0 & !Self::FILE_H.0) << 1)
    }

    pub const fn west(self) -> Self {
        Self((self.0 & !Self::FILE_A.0) >> 1)
    }

    pub const fn north_east(self) -> Self {
        Self((self.0 & !Self::FILE_H.0) << 9)
    }

    pub const fn north_west(self) -> Self {
        Self((self.0 & !Self::FILE_A.0) << 7)
    }

    pub const fn south_east(self) -> Self {
        Self((self.0 & !Self::FILE_H.0) >> 7)
    }

    pub const fn south_west(self) -> Self {
        Self((self.0 & !Self::FILE_A.0) >> 9)
    }
}

// --- Display ---

impl fmt::Display for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for rank in (0..8).rev() {
            for file in 0..8 {
                let sq = Square::new(rank * 8 + file);
                if self.contains(sq) {
                    write!(f, "1 ")?;
                } else {
                    write!(f, ". ")?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl fmt::Debug for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Bitboard(0x{:016x})", self.0)
    }
}

// --- Operator overloads ---

impl BitAnd for Bitboard {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl BitOr for Bitboard {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl BitXor for Bitboard {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}

impl Not for Bitboard {
    type Output = Self;
    fn not(self) -> Self {
        Self(!self.0)
    }
}

impl BitAndAssign for Bitboard {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitOrAssign for Bitboard {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitXorAssign for Bitboard {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

// --- Iterator ---

/// Iterator over the set squares in a bitboard.
pub struct BitboardIter(Bitboard);

impl Iterator for BitboardIter {
    type Item = Square;

    fn next(&mut self) -> Option<Square> {
        self.0.pop_lsb()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.0.popcount() as usize;
        (n, Some(n))
    }
}

impl ExactSizeIterator for BitboardIter {}

impl IntoIterator for Bitboard {
    type Item = Square;
    type IntoIter = BitboardIter;

    fn into_iter(self) -> BitboardIter {
        BitboardIter(self)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_square_contains() {
        for sq in Square::ALL {
            let bb = Bitboard::from_square(sq);
            assert!(bb.contains(sq));
            assert_eq!(bb.popcount(), 1);
        }
    }

    #[test]
    fn set_clear_toggle() {
        let mut bb = Bitboard::EMPTY;
        bb = bb.set(Square::E4);
        assert!(bb.contains(Square::E4));
        assert_eq!(bb.popcount(), 1);

        bb = bb.set(Square::D5);
        assert_eq!(bb.popcount(), 2);

        bb = bb.clear(Square::E4);
        assert!(!bb.contains(Square::E4));
        assert!(bb.contains(Square::D5));

        bb = bb.toggle(Square::D5);
        assert!(bb.is_empty());
    }

    #[test]
    fn lsb_msb() {
        let bb = Bitboard::from_square(Square::C3) | Bitboard::from_square(Square::G7);
        assert_eq!(bb.lsb(), Some(Square::C3));
        assert_eq!(bb.msb(), Some(Square::G7));
        assert_eq!(Bitboard::EMPTY.lsb(), None);
        assert_eq!(Bitboard::EMPTY.msb(), None);
    }

    #[test]
    fn pop_lsb_iteration() {
        let bb = Bitboard::from_square(Square::A1)
            | Bitboard::from_square(Square::E4)
            | Bitboard::from_square(Square::H8);
        let squares: Vec<Square> = bb.into_iter().collect();
        assert_eq!(squares, vec![Square::A1, Square::E4, Square::H8]);
    }

    #[test]
    fn iterator_exact_size() {
        let bb = Bitboard::from_square(Square::B2)
            | Bitboard::from_square(Square::C3)
            | Bitboard::from_square(Square::D4);
        let iter = bb.into_iter();
        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn directional_shifts() {
        let e4 = Bitboard::from_square(Square::E4);
        assert_eq!(e4.north(), Bitboard::from_square(Square::E5));
        assert_eq!(e4.south(), Bitboard::from_square(Square::E3));
        assert_eq!(e4.east(), Bitboard::from_square(Square::F4));
        assert_eq!(e4.west(), Bitboard::from_square(Square::D4));
        assert_eq!(e4.north_east(), Bitboard::from_square(Square::F5));
        assert_eq!(e4.north_west(), Bitboard::from_square(Square::D5));
        assert_eq!(e4.south_east(), Bitboard::from_square(Square::F3));
        assert_eq!(e4.south_west(), Bitboard::from_square(Square::D3));
    }

    #[test]
    fn shifts_dont_wrap() {
        // h-file pieces shouldn't wrap east
        let h4 = Bitboard::from_square(Square::H4);
        assert!(h4.east().is_empty());
        assert!(h4.north_east() == Bitboard::EMPTY);
        assert!(h4.south_east() == Bitboard::EMPTY);

        // a-file pieces shouldn't wrap west
        let a4 = Bitboard::from_square(Square::A4);
        assert!(a4.west().is_empty());
        assert!(a4.north_west() == Bitboard::EMPTY);
        assert!(a4.south_west() == Bitboard::EMPTY);
    }

    #[test]
    fn file_rank_constants() {
        // File A should have exactly 8 bits set, one per rank
        assert_eq!(Bitboard::FILE_A.popcount(), 8);
        for rank in Rank::ALL {
            let sq = Square::from_file_rank(crate::types::File::A, rank);
            assert!(Bitboard::FILE_A.contains(sq));
        }

        // Rank 1 should have exactly 8 bits set, one per file
        assert_eq!(Bitboard::RANK_1.popcount(), 8);
        for file in crate::types::File::ALL {
            let sq = Square::from_file_rank(file, Rank::R1);
            assert!(Bitboard::RANK_1.contains(sq));
        }
    }

    #[test]
    fn bitwise_ops() {
        let a = Bitboard::from_square(Square::A1) | Bitboard::from_square(Square::B2);
        let b = Bitboard::from_square(Square::B2) | Bitboard::from_square(Square::C3);

        assert_eq!((a & b), Bitboard::from_square(Square::B2));
        assert_eq!(
            (a | b),
            Bitboard::from_square(Square::A1)
                | Bitboard::from_square(Square::B2)
                | Bitboard::from_square(Square::C3)
        );
        assert_eq!(
            (a ^ b),
            Bitboard::from_square(Square::A1) | Bitboard::from_square(Square::C3)
        );
        assert_eq!((!Bitboard::EMPTY), Bitboard::FULL);
    }

    use crate::types::Rank;
}
