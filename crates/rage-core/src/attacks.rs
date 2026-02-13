//! Precomputed attack tables for all piece types.
//!
//! - Knight, king, pawn: simple lookup tables computed at compile time.
//! - Bishop, rook: magic bitboard tables initialized lazily at first use
//!   using hardcoded magic numbers (no brute-force search at startup).

use std::sync::LazyLock;

use crate::bitboard::Bitboard;
use crate::types::{Color, Square};

// ---------------------------------------------------------------------------
// Knight attacks (const computed)
// ---------------------------------------------------------------------------

const fn compute_knight_attacks(sq: u8) -> u64 {
    let bb = 1u64 << sq;
    let mut attacks = 0u64;

    attacks |= (bb << 17) & !0x0101_0101_0101_0101; // +2 ranks, +1 file
    attacks |= (bb << 15) & !0x8080_8080_8080_8080; // +2 ranks, -1 file
    attacks |= (bb >> 15) & !0x0101_0101_0101_0101; // -2 ranks, +1 file
    attacks |= (bb >> 17) & !0x8080_8080_8080_8080; // -2 ranks, -1 file
    attacks |= (bb << 10) & !0x0303_0303_0303_0303; // +1 rank, +2 files
    attacks |= (bb << 6) & !0xC0C0_C0C0_C0C0_C0C0; // +1 rank, -2 files
    attacks |= (bb >> 6) & !0x0303_0303_0303_0303; // -1 rank, +2 files
    attacks |= (bb >> 10) & !0xC0C0_C0C0_C0C0_C0C0; // -1 rank, -2 files

    attacks
}

const KNIGHT_ATTACKS: [Bitboard; 64] = {
    let mut table = [Bitboard::EMPTY; 64];
    let mut i = 0u8;
    while i < 64 {
        table[i as usize] = Bitboard::new(compute_knight_attacks(i));
        i += 1;
    }
    table
};

pub fn knight_attacks(sq: Square) -> Bitboard {
    KNIGHT_ATTACKS[sq.index()]
}

// ---------------------------------------------------------------------------
// King attacks (const computed)
// ---------------------------------------------------------------------------

const fn compute_king_attacks(sq: u8) -> u64 {
    let bb = 1u64 << sq;
    let mut attacks = 0u64;
    let not_a = !0x0101_0101_0101_0101u64;
    let not_h = !0x8080_8080_8080_8080u64;

    attacks |= bb << 8;
    attacks |= bb >> 8;
    attacks |= (bb << 1) & not_a;
    attacks |= (bb >> 1) & not_h;
    attacks |= (bb << 9) & not_a;
    attacks |= (bb << 7) & not_h;
    attacks |= (bb >> 7) & not_a;
    attacks |= (bb >> 9) & not_h;

    attacks
}

const KING_ATTACKS: [Bitboard; 64] = {
    let mut table = [Bitboard::EMPTY; 64];
    let mut i = 0u8;
    while i < 64 {
        table[i as usize] = Bitboard::new(compute_king_attacks(i));
        i += 1;
    }
    table
};

pub fn king_attacks(sq: Square) -> Bitboard {
    KING_ATTACKS[sq.index()]
}

// ---------------------------------------------------------------------------
// Pawn attacks (const computed)
// ---------------------------------------------------------------------------

const fn compute_pawn_attacks_white(sq: u8) -> u64 {
    let bb = 1u64 << sq;
    let not_a = !0x0101_0101_0101_0101u64;
    let not_h = !0x8080_8080_8080_8080u64;
    ((bb << 9) & not_a) | ((bb << 7) & not_h)
}

const fn compute_pawn_attacks_black(sq: u8) -> u64 {
    let bb = 1u64 << sq;
    let not_a = !0x0101_0101_0101_0101u64;
    let not_h = !0x8080_8080_8080_8080u64;
    ((bb >> 7) & not_a) | ((bb >> 9) & not_h)
}

const PAWN_ATTACKS: [[Bitboard; 64]; 2] = {
    let mut table = [[Bitboard::EMPTY; 64]; 2];
    let mut i = 0u8;
    while i < 64 {
        table[0][i as usize] = Bitboard::new(compute_pawn_attacks_white(i));
        table[1][i as usize] = Bitboard::new(compute_pawn_attacks_black(i));
        i += 1;
    }
    table
};

pub fn pawn_attacks(sq: Square, color: Color) -> Bitboard {
    PAWN_ATTACKS[color.index()][sq.index()]
}

// ---------------------------------------------------------------------------
// Magic bitboards for sliding pieces (bishop, rook)
// ---------------------------------------------------------------------------

// Hardcoded magic numbers. These were found via PRNG search and produce
// collision-free lookup tables for all 64 squares.

#[rustfmt::skip]
const BISHOP_MAGICS: [u64; 64] = [
    0x004002042c102041, 0x0002104200810109, 0x0141412201824004, 0x04082080208c0000,
    0x2004042040181000, 0x4003090840800138, 0x0027008210410511, 0x0602440404010461,
    0x0060040890240680, 0x0200115011084082, 0x0800c10401005104, 0x0042044502000010,
    0x0100040420402008, 0xc140808230414100, 0x2040007402201000, 0x8006008200900402,
    0x504000a004440088, 0x2820900801010200, 0x0004000804240110, 0x222800040218a211,
    0x0811020820080009, 0x4802000101008220, 0x0209000a88095040, 0x8082008316411400,
    0x0220202308020410, 0x801030110a160612, 0x2800300082048601, 0x0010802008020120,
    0x0044840150802002, 0x0000802206021000, 0x8002205000841000, 0x8020550002411800,
    0x040120088010100a, 0x0021100281300480, 0x1180208808100420, 0x0044400821020200,
    0x200a0084000a0220, 0x0001100080610042, 0x1004409200040100, 0x0102040023404200,
    0x88010188200040a0, 0x0100480804008800, 0x01200a0090080200, 0x000e004010480200,
    0x184222200a000100, 0x0208095801210200, 0x2210010101004420, 0x0022044910211a02,
    0x0104050111110000, 0x002200a208020080, 0x0004044200900000, 0x2061140042020320,
    0x0000948821070012, 0x0000080308420020, 0x4008101102240405, 0x0010220851002000,
    0x3100840452022040, 0x0000108044022004, 0x0412080042080404, 0x0023088010840401,
    0x0009108010020880, 0x0320000810018202, 0x0708600304480088, 0x2490045000802100,
];

#[rustfmt::skip]
const ROOK_MAGICS: [u64; 64] = [
    0x0100102100408000, 0x0840004010002000, 0x0200200a00401080, 0x8b00100100040820,
    0x4a00141e00982010, 0x0b00020900080c00, 0x2580010002000080, 0x018008d100022080,
    0x0120800028401080, 0x8000c00050002002, 0x1000802000801000, 0x8001000810010021,
    0x0000808008000400, 0x0001800201800400, 0x0004808011000200, 0x2812000100804402,
    0x8480004000402002, 0x0030134020004000, 0x8400808020001004, 0x2411010008201005,
    0x9404008080040800, 0x0112808004000200, 0x2001040008702241, 0x0252020000804104,
    0x6080004140022000, 0x0020200080804000, 0x0200802200104a00, 0x4100080080801000,
    0x0000080080040080, 0x8000040080020080, 0x020100210002005c, 0x00088c020014a441,
    0x0204400284800020, 0x0010201000404000, 0x0048200041001504, 0x00b2841000800800,
    0x0060110005004800, 0x0000104008010420, 0x1481000405000a00, 0x0040004882001504,
    0x1000209040008008, 0x0000402010014000, 0x0002001040820028, 0x4a01001000090021,
    0x0211001008010004, 0x0119000400090052, 0x0a00040200010100, 0x0015012080420004,
    0x0880004000200040, 0x1400810040002100, 0x0501004810200100, 0x4908800800100080,
    0x1020208100401002, 0x0001000208040100, 0x1101800100020080, 0x0120800041002080,
    0x00160100c060d086, 0x08c1002204428016, 0x0004c02012020882, 0x004a700020c80d01,
    0x0441001004020801, 0x401d000400020801, 0x8c5a011002080084, 0x102004050c204482,
];

struct MagicEntry {
    mask: u64,
    magic: u64,
    shift: u8,
    offset: usize,
}

struct MagicTables {
    bishop_entries: [MagicEntry; 64],
    rook_entries: [MagicEntry; 64],
    bishop_attacks: Box<[Bitboard]>,
    rook_attacks: Box<[Bitboard]>,
}

/// Compute the relevant blocker mask for a bishop on `sq`.
fn bishop_mask(sq: u8) -> u64 {
    let rank = (sq / 8) as i32;
    let file = (sq % 8) as i32;
    let mut mask = 0u64;

    for &(dr, df) in &[(1, 1), (1, -1), (-1, 1), (-1, -1)] {
        let mut r = rank + dr;
        let mut f = file + df;
        while r > 0 && r < 7 && f > 0 && f < 7 {
            mask |= 1u64 << (r * 8 + f);
            r += dr;
            f += df;
        }
    }
    mask
}

/// Compute the relevant blocker mask for a rook on `sq`.
fn rook_mask(sq: u8) -> u64 {
    let rank = (sq / 8) as i32;
    let file = (sq % 8) as i32;
    let mut mask = 0u64;

    for r in (rank + 1)..7 {
        mask |= 1u64 << (r * 8 + file);
    }
    for r in 1..rank {
        mask |= 1u64 << (r * 8 + file);
    }
    for f in (file + 1)..7 {
        mask |= 1u64 << (rank * 8 + f);
    }
    for f in 1..file {
        mask |= 1u64 << (rank * 8 + f);
    }
    mask
}

/// Compute bishop attacks from `sq` given `occupied` blockers.
fn bishop_attacks_slow(sq: u8, occupied: u64) -> u64 {
    let rank = (sq / 8) as i32;
    let file = (sq % 8) as i32;
    let mut attacks = 0u64;

    for &(dr, df) in &[(1, 1), (1, -1), (-1, 1), (-1, -1)] {
        let mut r = rank + dr;
        let mut f = file + df;
        while (0..8).contains(&r) && (0..8).contains(&f) {
            let bit = 1u64 << (r * 8 + f);
            attacks |= bit;
            if occupied & bit != 0 {
                break;
            }
            r += dr;
            f += df;
        }
    }
    attacks
}

/// Compute rook attacks from `sq` given `occupied` blockers.
fn rook_attacks_slow(sq: u8, occupied: u64) -> u64 {
    let rank = (sq / 8) as i32;
    let file = (sq % 8) as i32;
    let mut attacks = 0u64;

    for &(dr, df) in &[(1, 0), (-1, 0), (0, 1), (0, -1)] {
        let mut r = rank + dr;
        let mut f = file + df;
        while (0..8).contains(&r) && (0..8).contains(&f) {
            let bit = 1u64 << (r * 8 + f);
            attacks |= bit;
            if occupied & bit != 0 {
                break;
            }
            r += dr;
            f += df;
        }
    }
    attacks
}

/// Enumerate the next subset of a mask using the Carry-Rippler trick.
const fn next_subset(subset: u64, mask: u64) -> u64 {
    subset.wrapping_sub(mask) & mask
}

/// Build the attack lookup table for one square given a known magic number.
fn build_table(sq: u8, mask: u64, magic: u64, attacks_fn: fn(u8, u64) -> u64) -> Vec<Bitboard> {
    let bits = mask.count_ones();
    let size = 1usize << bits;
    let shift = 64 - bits as u8;
    let mut table = vec![Bitboard::EMPTY; size];

    let mut subset = 0u64;
    loop {
        let index = (subset.wrapping_mul(magic) >> shift) as usize;
        table[index] = Bitboard::new(attacks_fn(sq, subset));
        subset = next_subset(subset, mask);
        if subset == 0 {
            break;
        }
    }
    table
}

fn init_magic_tables() -> MagicTables {
    let mut bishop_entries = Vec::with_capacity(64);
    let mut rook_entries = Vec::with_capacity(64);
    let mut bishop_attacks_vec = Vec::new();
    let mut rook_attacks_vec = Vec::new();

    for sq in 0u8..64 {
        let mask = bishop_mask(sq);
        let magic = BISHOP_MAGICS[sq as usize];
        let bits = mask.count_ones();
        let offset = bishop_attacks_vec.len();
        let table = build_table(sq, mask, magic, bishop_attacks_slow);
        bishop_attacks_vec.extend_from_slice(&table);
        bishop_entries.push(MagicEntry {
            mask,
            magic,
            shift: 64 - bits as u8,
            offset,
        });
    }

    for sq in 0u8..64 {
        let mask = rook_mask(sq);
        let magic = ROOK_MAGICS[sq as usize];
        let bits = mask.count_ones();
        let offset = rook_attacks_vec.len();
        let table = build_table(sq, mask, magic, rook_attacks_slow);
        rook_attacks_vec.extend_from_slice(&table);
        rook_entries.push(MagicEntry {
            mask,
            magic,
            shift: 64 - bits as u8,
            offset,
        });
    }

    // SAFETY: both Vecs have exactly 64 elements, matching the target array size.
    let Ok(bishop_arr) = bishop_entries.try_into() else {
        unreachable!()
    };
    let Ok(rook_arr) = rook_entries.try_into() else {
        unreachable!()
    };

    MagicTables {
        bishop_entries: bishop_arr,
        rook_entries: rook_arr,
        bishop_attacks: bishop_attacks_vec.into_boxed_slice(),
        rook_attacks: rook_attacks_vec.into_boxed_slice(),
    }
}

static MAGIC: LazyLock<MagicTables> = LazyLock::new(init_magic_tables);

/// Force initialization of the magic bitboard tables.
/// Call this at program startup to pay the cost upfront rather than on first use.
pub fn init() {
    LazyLock::force(&MAGIC);
}

/// Get bishop attack bitboard for a given square and occupancy.
pub fn bishop_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    let entry = &MAGIC.bishop_entries[sq.index()];
    let index = ((occupied.0 & entry.mask).wrapping_mul(entry.magic) >> entry.shift) as usize;
    MAGIC.bishop_attacks[entry.offset + index]
}

/// Get rook attack bitboard for a given square and occupancy.
pub fn rook_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    let entry = &MAGIC.rook_entries[sq.index()];
    let index = ((occupied.0 & entry.mask).wrapping_mul(entry.magic) >> entry.shift) as usize;
    MAGIC.rook_attacks[entry.offset + index]
}

/// Get queen attack bitboard (union of bishop and rook attacks).
pub fn queen_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    bishop_attacks(sq, occupied) | rook_attacks(sq, occupied)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knight_attacks_center() {
        let attacks = knight_attacks(Square::E4);
        assert_eq!(attacks.popcount(), 8);
        assert!(attacks.contains(Square::D6));
        assert!(attacks.contains(Square::F6));
        assert!(attacks.contains(Square::C5));
        assert!(attacks.contains(Square::G5));
        assert!(attacks.contains(Square::C3));
        assert!(attacks.contains(Square::G3));
        assert!(attacks.contains(Square::D2));
        assert!(attacks.contains(Square::F2));
    }

    #[test]
    fn knight_attacks_corner() {
        let attacks = knight_attacks(Square::A1);
        assert_eq!(attacks.popcount(), 2);
        assert!(attacks.contains(Square::B3));
        assert!(attacks.contains(Square::C2));
    }

    #[test]
    fn king_attacks_center() {
        let attacks = king_attacks(Square::E4);
        assert_eq!(attacks.popcount(), 8);
    }

    #[test]
    fn king_attacks_corner() {
        let attacks = king_attacks(Square::A1);
        assert_eq!(attacks.popcount(), 3);
        assert!(attacks.contains(Square::A2));
        assert!(attacks.contains(Square::B1));
        assert!(attacks.contains(Square::B2));
    }

    #[test]
    fn pawn_attacks_white() {
        let attacks = pawn_attacks(Square::E4, Color::White);
        assert_eq!(attacks.popcount(), 2);
        assert!(attacks.contains(Square::D5));
        assert!(attacks.contains(Square::F5));
    }

    #[test]
    fn pawn_attacks_black() {
        let attacks = pawn_attacks(Square::E4, Color::Black);
        assert_eq!(attacks.popcount(), 2);
        assert!(attacks.contains(Square::D3));
        assert!(attacks.contains(Square::F3));
    }

    #[test]
    fn pawn_attacks_edge() {
        let attacks = pawn_attacks(Square::A2, Color::White);
        assert_eq!(attacks.popcount(), 1);
        assert!(attacks.contains(Square::B3));

        let attacks = pawn_attacks(Square::H7, Color::Black);
        assert_eq!(attacks.popcount(), 1);
        assert!(attacks.contains(Square::G6));
    }

    #[test]
    fn bishop_attacks_empty_board() {
        let attacks = bishop_attacks(Square::E4, Bitboard::EMPTY);
        assert_eq!(attacks.popcount(), 13);
    }

    #[test]
    fn bishop_attacks_blocked() {
        let occupied = Bitboard::from_square(Square::C6) | Bitboard::from_square(Square::F3);
        let attacks = bishop_attacks(Square::E4, occupied);
        assert!(attacks.contains(Square::D5));
        assert!(attacks.contains(Square::C6));
        assert!(!attacks.contains(Square::B7));
        assert!(attacks.contains(Square::F5));
        assert!(attacks.contains(Square::F3));
        assert!(!attacks.contains(Square::G2));
    }

    #[test]
    fn rook_attacks_empty_board() {
        let attacks = rook_attacks(Square::E4, Bitboard::EMPTY);
        assert_eq!(attacks.popcount(), 14);
    }

    #[test]
    fn rook_attacks_blocked() {
        let occupied = Bitboard::from_square(Square::E6) | Bitboard::from_square(Square::C4);
        let attacks = rook_attacks(Square::E4, occupied);
        assert!(attacks.contains(Square::E5));
        assert!(attacks.contains(Square::E6));
        assert!(!attacks.contains(Square::E7));
        assert!(attacks.contains(Square::D4));
        assert!(attacks.contains(Square::C4));
        assert!(!attacks.contains(Square::B4));
        assert!(attacks.contains(Square::E1));
        assert!(attacks.contains(Square::H4));
    }

    #[test]
    fn queen_attacks_is_bishop_union_rook() {
        let occupied = Bitboard::from_square(Square::C3) | Bitboard::from_square(Square::G7);
        let q = queen_attacks(Square::E4, occupied);
        let b = bishop_attacks(Square::E4, occupied);
        let r = rook_attacks(Square::E4, occupied);
        assert_eq!(q, b | r);
    }

    #[test]
    fn bishop_attacks_all_squares_nonempty() {
        for sq in Square::ALL {
            let attacks = bishop_attacks(sq, Bitboard::EMPTY);
            assert!(attacks.is_not_empty(), "bishop on {sq} has no attacks");
        }
    }

    #[test]
    fn rook_attacks_all_squares_14() {
        for sq in Square::ALL {
            let attacks = rook_attacks(sq, Bitboard::EMPTY);
            assert_eq!(
                attacks.popcount(),
                14,
                "rook on {sq} has {} attacks",
                attacks.popcount()
            );
        }
    }

    #[test]
    fn magic_tables_consistency() {
        let test_squares = [0u8, 7, 27, 36, 56, 63];
        let test_occupancies = [
            0u64,
            0x00FF_00FF_00FF_00FF,
            0xFF00_FF00_FF00_FF00,
            0x0000_0000_FFFF_FFFF,
        ];

        for &sq in &test_squares {
            for &occ in &test_occupancies {
                let b_magic = bishop_attacks(Square::new(sq), Bitboard::new(occ));
                let b_slow = Bitboard::new(bishop_attacks_slow(sq, occ));
                assert_eq!(
                    b_magic, b_slow,
                    "bishop mismatch: sq={sq}, occ=0x{occ:016x}"
                );

                let r_magic = rook_attacks(Square::new(sq), Bitboard::new(occ));
                let r_slow = Bitboard::new(rook_attacks_slow(sq, occ));
                assert_eq!(r_magic, r_slow, "rook mismatch: sq={sq}, occ=0x{occ:016x}");
            }
        }
    }
}
