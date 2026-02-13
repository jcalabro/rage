//! Precomputed attack tables for all piece types.
//!
//! - Knight, king, pawn: simple lookup tables computed at compile time.
//! - Bishop, rook: magic bitboard tables initialized lazily at first use.

use std::sync::LazyLock;

use crate::bitboard::Bitboard;
use crate::types::{Color, Square};

// ---------------------------------------------------------------------------
// Knight attacks (const computed)
// ---------------------------------------------------------------------------

const fn compute_knight_attacks(sq: u8) -> u64 {
    let bb = 1u64 << sq;
    let mut attacks = 0u64;

    // Two ranks up, one file left/right
    attacks |= (bb << 17) & !0x0101_0101_0101_0101; // +2 ranks, +1 file (not A-file)
    attacks |= (bb << 15) & !0x8080_8080_8080_8080; // +2 ranks, -1 file (not H-file)
    // Two ranks down, one file left/right
    attacks |= (bb >> 15) & !0x0101_0101_0101_0101; // -2 ranks, +1 file
    attacks |= (bb >> 17) & !0x8080_8080_8080_8080; // -2 ranks, -1 file
    // One rank up, two files left/right
    attacks |= (bb << 10) & !0x0303_0303_0303_0303; // +1 rank, +2 files (not AB-files)
    attacks |= (bb << 6) & !0xC0C0_C0C0_C0C0_C0C0; // +1 rank, -2 files (not GH-files)
    // One rank down, two files left/right
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

/// Get knight attack bitboard for a given square.
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

    attacks |= bb << 8; // north
    attacks |= bb >> 8; // south
    attacks |= (bb << 1) & not_a; // east
    attacks |= (bb >> 1) & not_h; // west
    attacks |= (bb << 9) & not_a; // north-east
    attacks |= (bb << 7) & not_h; // north-west
    attacks |= (bb >> 7) & not_a; // south-east
    attacks |= (bb >> 9) & not_h; // south-west

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

/// Get king attack bitboard for a given square.
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

/// Get pawn attack bitboard for a given square and color.
pub fn pawn_attacks(sq: Square, color: Color) -> Bitboard {
    PAWN_ATTACKS[color.index()][sq.index()]
}

// ---------------------------------------------------------------------------
// Magic bitboards for sliding pieces (bishop, rook)
// ---------------------------------------------------------------------------

struct MagicEntry {
    mask: u64,
    magic: u64,
    shift: u8,
    offset: usize,
}

struct MagicTables {
    bishop_entries: [MagicEntry; 64],
    rook_entries: [MagicEntry; 64],
    bishop_attacks: Vec<Bitboard>,
    rook_attacks: Vec<Bitboard>,
}

/// Compute the relevant blocker mask for a bishop on `sq`.
/// Excludes edge squares since blockers on edges don't affect the attack pattern
/// (the ray always stops at the edge).
fn bishop_mask(sq: u8) -> u64 {
    let rank = (sq / 8) as i32;
    let file = (sq % 8) as i32;
    let mut mask = 0u64;

    // Four diagonal directions, stopping one square before the edge
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

    // Four cardinal directions, stopping one square before the edge
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

/// Compute bishop attacks from `sq` given `occupied` blockers (including edges).
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

/// Compute rook attacks from `sq` given `occupied` blockers (including edges).
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

/// Enumerate the n-th subset of a bitmask using the Carry-Rippler trick.
/// Returns all subsets when iterated: start with 0, call repeatedly with the
/// previous result until it wraps back to 0.
const fn next_subset(subset: u64, mask: u64) -> u64 {
    subset.wrapping_sub(mask) & mask
}

/// Simple xorshift64 PRNG for magic number generation.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Generate a sparse random number suitable for use as a magic number.
fn sparse_random(rng: &mut u64) -> u64 {
    xorshift64(rng) & xorshift64(rng) & xorshift64(rng)
}

/// Find a magic number for a given square with the given mask and attack function.
fn find_magic(
    sq: u8,
    mask: u64,
    attacks_fn: fn(u8, u64) -> u64,
    rng: &mut u64,
) -> (u64, Vec<Bitboard>) {
    let bits = mask.count_ones();
    let size = 1usize << bits;
    let shift = 64 - bits as u8;

    // Enumerate all blocker configurations and their corresponding attacks
    let mut blockers = Vec::with_capacity(size);
    let mut attacks = Vec::with_capacity(size);
    let mut subset = 0u64;
    loop {
        blockers.push(subset);
        attacks.push(Bitboard::new(attacks_fn(sq, subset)));
        subset = next_subset(subset, mask);
        if subset == 0 {
            break;
        }
    }

    // Try random magic numbers until we find one with no collisions
    let mut table = vec![Bitboard::EMPTY; size];
    loop {
        let magic = sparse_random(rng);
        // Quick rejection: the magic must map the mask's high bits well
        if ((mask.wrapping_mul(magic)) >> 56).count_ones() < 6 {
            continue;
        }

        // Clear table
        table.fill(Bitboard::EMPTY);

        let mut ok = true;
        for i in 0..blockers.len() {
            let index = (blockers[i].wrapping_mul(magic) >> shift) as usize;
            if table[index].is_empty() {
                table[index] = attacks[i];
            } else if table[index] != attacks[i] {
                ok = false;
                break;
            }
            // Constructive collision (same attack set) is fine
        }

        if ok {
            return (magic, table);
        }
    }
}

fn init_magic_tables() -> MagicTables {
    // Fixed seed for deterministic startup
    let mut rng = 0x1234_5678_9ABC_DEF0u64;

    let mut bishop_entries: Vec<MagicEntry> = Vec::with_capacity(64);
    let mut rook_entries: Vec<MagicEntry> = Vec::with_capacity(64);
    let mut bishop_attacks = Vec::new();
    let mut rook_attacks = Vec::new();

    for sq in 0u8..64 {
        let mask = bishop_mask(sq);
        let bits = mask.count_ones();
        let offset = bishop_attacks.len();
        let (magic, table) = find_magic(sq, mask, bishop_attacks_slow, &mut rng);
        bishop_attacks.extend_from_slice(&table);
        bishop_entries.push(MagicEntry {
            mask,
            magic,
            shift: 64 - bits as u8,
            offset,
        });
    }

    for sq in 0u8..64 {
        let mask = rook_mask(sq);
        let bits = mask.count_ones();
        let offset = rook_attacks.len();
        let (magic, table) = find_magic(sq, mask, rook_attacks_slow, &mut rng);
        rook_attacks.extend_from_slice(&table);
        rook_entries.push(MagicEntry {
            mask,
            magic,
            shift: 64 - bits as u8,
            offset,
        });
    }

    MagicTables {
        bishop_entries: bishop_entries
            .try_into()
            .ok()
            .unwrap_or_else(|| unreachable!()),
        rook_entries: rook_entries
            .try_into()
            .ok()
            .unwrap_or_else(|| unreachable!()),
        bishop_attacks,
        rook_attacks,
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
        // a-file pawn only attacks one square
        let attacks = pawn_attacks(Square::A2, Color::White);
        assert_eq!(attacks.popcount(), 1);
        assert!(attacks.contains(Square::B3));

        // h-file pawn only attacks one square
        let attacks = pawn_attacks(Square::H7, Color::Black);
        assert_eq!(attacks.popcount(), 1);
        assert!(attacks.contains(Square::G6));
    }

    #[test]
    fn bishop_attacks_empty_board() {
        let attacks = bishop_attacks(Square::E4, Bitboard::EMPTY);
        // From e4, bishop can reach: d5,c6,b7,a8, f5,g6,h7, d3,c2,b1, f3,g2,h1 = 13
        assert_eq!(attacks.popcount(), 13);
    }

    #[test]
    fn bishop_attacks_blocked() {
        // Place blockers on c6 and f3
        let occupied = Bitboard::from_square(Square::C6) | Bitboard::from_square(Square::F3);
        let attacks = bishop_attacks(Square::E4, occupied);
        // NW diagonal: d5, c6 (blocked, includes c6)
        // NE diagonal: f5, g6, h7
        // SW diagonal: d3, c2, b1
        // SE diagonal: f3 (blocked, includes f3)
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
        // 7 squares in each direction on rank and file = 14
        assert_eq!(attacks.popcount(), 14);
    }

    #[test]
    fn rook_attacks_blocked() {
        let occupied = Bitboard::from_square(Square::E6) | Bitboard::from_square(Square::C4);
        let attacks = rook_attacks(Square::E4, occupied);
        // North: e5, e6 (blocked)
        assert!(attacks.contains(Square::E5));
        assert!(attacks.contains(Square::E6));
        assert!(!attacks.contains(Square::E7));
        // West: d4, c4 (blocked)
        assert!(attacks.contains(Square::D4));
        assert!(attacks.contains(Square::C4));
        assert!(!attacks.contains(Square::B4));
        // South and east: unblocked
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
        // Every square should have at least some bishop attacks on an empty board
        for sq in Square::ALL {
            let attacks = bishop_attacks(sq, Bitboard::EMPTY);
            assert!(attacks.is_not_empty(), "bishop on {sq} has no attacks");
        }
    }

    #[test]
    fn rook_attacks_all_squares_14() {
        // On an empty board, every rook has exactly 14 attack squares
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
        // Verify magic table results match the slow reference implementation
        // for a sample of positions
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
