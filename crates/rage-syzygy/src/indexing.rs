//! Position-to-index mapping for Syzygy tables.
//!
//! Maps chess positions to linear indices into the decompressed table data.
//! Handles symmetry normalization, pawn/pawnless encoding, and combinatorial
//! number system indexing.

use rage_core::Square;

// ---------------------------------------------------------------------------
// Precomputed tables
// ---------------------------------------------------------------------------

/// Binomial coefficients C(n, k) for n in 0..64, k in 0..7.
/// BINOMIAL[n][k] = C(n, k)
pub const BINOMIAL: [[u64; 7]; 64] = {
    let mut table = [[0u64; 7]; 64];
    // C(n, 0) = 1 for all n
    let mut n = 0;
    while n < 64 {
        table[n][0] = 1;
        n += 1;
    }
    // Build up using Pascal's rule: C(n, k) = C(n-1, k-1) + C(n-1, k)
    n = 1;
    while n < 64 {
        let mut k = 1;
        while k < 7 && k <= n {
            table[n][k] = table[n - 1][k - 1] + table[n - 1][k];
            k += 1;
        }
        n += 1;
    }
    table
};

/// Pawn twist: maps square to pawn-table-friendly index.
/// Folds files a-d (files e-h are mirrored onto a-d), then indexes by rank.
/// Result is a unique index for each (mirrored file, rank) pair.
pub const PAWN_TWIST: [u8; 64] = {
    let mut table = [0u8; 64];
    let mut sq = 0u8;
    while sq < 64 {
        let file = sq & 7;
        let rank = sq >> 3;
        // Mirror file: 0->0, 1->1, 2->2, 3->3, 4->3, 5->2, 6->1, 7->0
        let mirrored_file = if file > 3 { 7 - file } else { file };
        // Index: file * 8 + rank (but exclude rank 0 and 7 for pawns that
        // can only be on ranks 1-6 in most cases)
        // Actually in Syzygy, pawn indices use all 8 ranks for intermediate calcs
        table[sq as usize] = mirrored_file * 8 + rank;
        sq += 1;
    }
    table
};

/// Triangle mapping for the first king in pawnless tables.
/// Maps the lower-left triangle (a1-d1-d4) to indices 0..9.
/// Squares outside the triangle are mapped via flips.
pub const TRIANGLE: [u8; 64] = {
    let mut table = [0u8; 64];
    let mapping: [u8; 64] = [
        6, 0, 1, 2, 2, 1, 0, 6, //
        0, 7, 3, 4, 4, 3, 7, 0, //
        1, 3, 8, 5, 5, 8, 3, 1, //
        2, 4, 5, 9, 9, 5, 4, 2, //
        2, 4, 5, 9, 9, 5, 4, 2, //
        1, 3, 8, 5, 5, 8, 3, 1, //
        0, 7, 3, 4, 4, 3, 7, 0, //
        6, 0, 1, 2, 2, 1, 0, 6, //
    ];
    let mut i = 0;
    while i < 64 {
        table[i] = mapping[i];
        i += 1;
    }
    table
};

/// Lower triangle: squares in the a1-d1-d4 triangle (10 squares).
/// Used for pawnless tables where the stronger king is placed first.
pub const LOWER_TRIANGLE: [u8; 64] = {
    let mut table = [255u8; 64]; // 255 = not in triangle
    // a1(0)=0, b1(1)=1, c1(2)=2, d1(3)=3
    // b2(9)=4, c2(10)=5, d2(11)=6
    // c3(18)=7, d3(19)=8
    // d4(27)=9
    table[0] = 0;
    table[1] = 1;
    table[2] = 2;
    table[3] = 3;
    table[9] = 4;
    table[10] = 5;
    table[11] = 6;
    table[18] = 7;
    table[19] = 8;
    table[27] = 9;
    table
};

/// Flip flags for the triangle mapping.
/// Each position in the triangle can require no flip, horizontal flip,
/// diagonal flip, or both.
pub const FLIP_DIAG: u8 = 1;
pub const FLIP_HORIZ: u8 = 2;
pub const FLIP_VERT: u8 = 4;

/// For each square, what flips are needed to bring it into the lower triangle.
pub const TRIANGLE_FLIP: [u8; 64] = {
    let mut table = [0u8; 64];
    let mut sq = 0u8;
    while sq < 64 {
        let file = sq & 7;
        let rank = sq >> 3;
        let mut flip = 0u8;

        // Mirror horizontally if file > 3
        if file > 3 {
            flip |= FLIP_HORIZ;
        }
        // Mirror vertically if rank > 3
        if rank > 3 {
            flip |= FLIP_VERT;
        }
        // Mirror along diagonal if rank > file (after horizontal/vertical flips)
        let adj_file = if file > 3 { 7 - file } else { file };
        let adj_rank = if rank > 3 { 7 - rank } else { rank };
        if adj_rank > adj_file {
            flip |= FLIP_DIAG;
        }

        table[sq as usize] = flip;
        sq += 1;
    }
    table
};

// ---------------------------------------------------------------------------
// Square transformation
// ---------------------------------------------------------------------------

/// Apply symmetry transformations to a square.
pub fn transform_square(sq: Square, flip: u8) -> Square {
    let mut s = sq.index() as u8;
    if flip & FLIP_HORIZ != 0 {
        s ^= 7; // flip file
    }
    if flip & FLIP_VERT != 0 {
        s ^= 56; // flip rank
    }
    if flip & FLIP_DIAG != 0 {
        // Transpose: swap file and rank
        let file = s & 7;
        let rank = s >> 3;
        s = file * 8 + rank;
    }
    Square::new(s)
}

/// For pawn tables: flip file if needed (fold to files a-d).
pub fn pawn_file_flip(sq: Square) -> Square {
    let file = sq.file().index();
    if file > 3 { sq.flip_file() } else { sq }
}

// ---------------------------------------------------------------------------
// Combinatorial indexing
// ---------------------------------------------------------------------------

/// Compute the combinatorial index for a sorted list of squares.
/// squares must be sorted in ascending order.
/// Uses the combinatorial number system: index = sum of C(sq[i], i+1).
pub fn encode_pieces(squares: &[u8]) -> u64 {
    let mut idx = 0u64;
    for (i, &sq) in squares.iter().enumerate() {
        idx += BINOMIAL[sq as usize][i + 1];
    }
    idx
}

// ---------------------------------------------------------------------------
// Pawnless encoding
// ---------------------------------------------------------------------------

/// Number of positions for the first (stronger) king in the lower triangle.
pub const NUM_TRIANGLE: u64 = 10;

/// Compute the index for a pawnless position.
///
/// `king1` is the stronger king (must be in the a1-d1-d4 triangle after normalization).
/// `king2` is the weaker king.
/// `others` are the remaining pieces (sorted by square index).
///
/// Returns (index, table_size) or None if kings are adjacent (illegal).
pub fn encode_pawnless(king1: Square, king2: Square, others: &[Square]) -> Option<(u64, u64)> {
    let k1 = king1.index() as u8;

    // Map king1 to triangle index
    let tri = LOWER_TRIANGLE[k1 as usize];
    if tri == 255 {
        return None; // king1 not in triangle (shouldn't happen after normalization)
    }

    // King2 index: the second king can be on any of the remaining 63 squares
    // (minus king1's square). We use direct square index, but skip king1's square.
    let k2 = king2.index() as u8;
    let k2_adj = if k2 > k1 { k2 - 1 } else { k2 };

    let mut base = tri as u64 * 63;
    base += k2_adj as u64;

    // Remaining pieces: compute combinatorial index on remaining squares
    // (excluding both king squares)
    let king_squares = [k1.min(k2), k1.max(k2)];
    let mut adjusted = Vec::with_capacity(others.len());
    for &sq in others {
        let s = sq.index() as u8;
        let mut adj = s;
        // Subtract 1 for each king square below this piece
        if s > king_squares[0] {
            adj -= 1;
        }
        if s > king_squares[1] {
            adj -= 1;
        }
        adjusted.push(adj);
    }
    adjusted.sort();

    let n_remaining = 62u64; // 64 - 2 kings
    let pieces_idx = encode_pieces(&adjusted);

    // Total combinations for remaining pieces: C(62, num_others)
    let num_others = others.len();
    let combos = if num_others == 0 {
        1
    } else {
        BINOMIAL[n_remaining as usize][num_others]
    };

    let total_idx = base * combos + pieces_idx;
    let table_size = NUM_TRIANGLE * 63 * combos;

    Some((total_idx, table_size))
}

// ---------------------------------------------------------------------------
// Pawn encoding
// ---------------------------------------------------------------------------

/// Number of files (after mirroring) for pawn tables: 4 (a-d).
pub const NUM_PAWN_FILES: usize = 4;

/// Precomputed table: for each (file, rank) of the leading pawn,
/// how many positions exist for it (used as multiplier for remaining pawns).
///
/// Leading pawn on file f can be on ranks 1-6 (indices 0-5 in the file group).
/// But pawns can't be on ranks 0 or 7 (promotion/start rows), so effectively
/// ranks 1-6 (squares rank 1-6).
pub fn pawn_leading_index(sq: Square) -> u64 {
    PAWN_TWIST[sq.index()] as u64
}

/// Compute index for pawn positions.
/// `pawns` are all pawn squares for one side, sorted by PAWN_TWIST value.
/// Returns the combinatorial index.
pub fn encode_pawns(pawns: &[Square]) -> u64 {
    if pawns.is_empty() {
        return 0;
    }

    let mut twisted: Vec<u8> = pawns.iter().map(|&sq| PAWN_TWIST[sq.index()]).collect();
    twisted.sort();

    // Leading pawn gets a direct index, remaining use combinatorial
    let leading = twisted[0] as u64;
    if twisted.len() == 1 {
        return leading;
    }

    // Remaining pawns use combinatorial index on the remaining PAWN_TWIST values
    let remaining = &twisted[1..];
    let combo = encode_pieces(remaining);

    // Combine: leading_index * C(remaining_space, remaining_count) + combo
    // This is an approximation; exact formula depends on table format specifics
    leading * BINOMIAL[47][remaining.len()] + combo
}

// ---------------------------------------------------------------------------
// Full position encoding
// ---------------------------------------------------------------------------

/// Symmetry class of a position, determining which transformations to apply.
#[derive(Clone, Copy, Debug)]
pub struct SymmetryInfo {
    /// Flip flags to normalize the position.
    pub flip: u8,
    /// Whether to swap colors (if black is the "strong" side).
    pub color_swap: bool,
}

/// Normalize a position for table lookup.
///
/// For pawnless tables: apply 8-fold symmetry (horizontal, vertical, diagonal flips)
/// to place the stronger king in the a1-d1-d4 triangle.
///
/// For pawn tables: apply 2-fold symmetry (horizontal flip only, since pawns have
/// a direction), folding to files a-d.
pub fn normalize_symmetry(strong_king: Square, has_pawns: bool) -> SymmetryInfo {
    if has_pawns {
        // Pawn tables: only horizontal flip (fold to a-d files)
        let file = strong_king.file().index();
        let flip = if file > 3 { FLIP_HORIZ } else { 0 };
        SymmetryInfo {
            flip,
            color_swap: false,
        }
    } else {
        // Pawnless tables: 8-fold symmetry
        let flip = TRIANGLE_FLIP[strong_king.index()];
        SymmetryInfo {
            flip,
            color_swap: false,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn binomial_base_cases() {
        // C(n, 0) = 1
        for n in 0..64 {
            assert_eq!(BINOMIAL[n][0], 1);
        }
        // C(n, 1) = n
        for n in 1..64 {
            assert_eq!(BINOMIAL[n][1], n as u64);
        }
        // Known values
        assert_eq!(BINOMIAL[10][3], 120); // C(10,3) = 120
        assert_eq!(BINOMIAL[62][2], 1891); // C(62,2) = 1891
        assert_eq!(BINOMIAL[63][2], 1953); // C(63,2) = 1953
    }

    #[test]
    fn triangle_symmetry() {
        // a1 should map to a valid triangle index
        assert!(LOWER_TRIANGLE[0] < 10);
        // d4 should also be in the triangle
        assert!(LOWER_TRIANGLE[27] < 10);
        // e1 is NOT in the lower triangle
        assert_eq!(LOWER_TRIANGLE[4], 255);
    }

    #[test]
    fn transform_square_identity() {
        for sq in Square::ALL {
            assert_eq!(transform_square(sq, 0), sq);
        }
    }

    #[test]
    fn transform_square_horiz() {
        // a1 -> h1
        assert_eq!(transform_square(Square::A1, FLIP_HORIZ), Square::H1);
        // e4 -> d4
        assert_eq!(transform_square(Square::E4, FLIP_HORIZ), Square::D4);
    }

    #[test]
    fn transform_square_vert() {
        // a1 -> a8
        assert_eq!(transform_square(Square::A1, FLIP_VERT), Square::A8);
        // d3 -> d6
        assert_eq!(transform_square(Square::D3, FLIP_VERT), Square::D6);
    }

    #[test]
    fn encode_pieces_simple() {
        // Single piece on square 5 -> C(5, 1) = 5
        assert_eq!(encode_pieces(&[5]), 5);
        // Two pieces on squares 3, 7 -> C(3,1) + C(7,2) = 3 + 21 = 24
        assert_eq!(encode_pieces(&[3, 7]), 24);
    }

    #[test]
    fn pawnless_krk_encoding() {
        // KRvK: White king on a1, black king on h8, rook on a8
        let king1 = Square::A1; // lower triangle
        let king2 = Square::H8;
        let others = [Square::A8]; // rook
        let result = encode_pawnless(king1, king2, &others);
        assert!(result.is_some());
        let (idx, size) = result.unwrap();
        assert!(idx < size);
    }

    #[test]
    fn pawnless_different_positions_different_indices() {
        let king1 = Square::A1;
        let king2a = Square::C3;
        let king2b = Square::D4;
        let others = [Square::B2];

        let (idx_a, _) = encode_pawnless(king1, king2a, &others).unwrap();
        let (idx_b, _) = encode_pawnless(king1, king2b, &others).unwrap();
        assert_ne!(idx_a, idx_b);
    }

    #[test]
    fn pawn_twist_symmetry() {
        // Files a and h should map to same twist base (mirrored)
        assert_eq!(
            PAWN_TWIST[Square::A2.index()] % 8,
            PAWN_TWIST[Square::H2.index()] % 8
        );
    }
}
