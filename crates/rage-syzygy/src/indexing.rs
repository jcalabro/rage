//! Position-to-index mapping for Syzygy tables.
//!
//! Maps chess positions to linear indices into the decompressed table data.
//! Uses the Syzygy combinatorial encoding scheme with group-based factors.
//!
//! Reference: shakmaty-syzygy by niklasf.

use rage_core::{Color, PieceType};

use crate::material::MaterialKey;

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Compute binomial coefficient C(n, k).
pub const fn binomial(mut n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    let k2 = n - k;
    let k = if k > k2 { k2 } else { k };
    let mut r = 1u64;
    let mut d = 1u64;
    while d <= k {
        r = r * n / d;
        n -= 1;
        d += 1;
    }
    r
}

/// Check if a square is off the a1-h8 diagonal (file != rank).
pub fn offdiag(sq: usize) -> bool {
    (sq & 7) != (sq >> 3)
}

/// Flip a square vertically (rank flip).
fn flip_vert(sq: usize) -> usize {
    sq ^ 56
}

/// Flip a square horizontally (file flip).
fn flip_horiz(sq: usize) -> usize {
    sq ^ 7
}

/// Transpose a square along the a1-h8 diagonal (swap file and rank).
fn flip_diag(sq: usize) -> usize {
    let file = sq & 7;
    let rank = sq >> 3;
    file * 8 + rank
}

// ---------------------------------------------------------------------------
// Lookup tables
// ---------------------------------------------------------------------------

/// Maps squares into the a1-d1-d4 triangle (10 positions).
/// Values 0-5 are off-diagonal, 6-9 are on the diagonal.
#[rustfmt::skip]
pub const TRIANGLE: [u64; 64] = [
    6, 0, 1, 2, 2, 1, 0, 6,
    0, 7, 3, 4, 4, 3, 7, 0,
    1, 3, 8, 5, 5, 8, 3, 1,
    2, 4, 5, 9, 9, 5, 4, 2,
    2, 4, 5, 9, 9, 5, 4, 2,
    1, 3, 8, 5, 5, 8, 3, 1,
    0, 7, 3, 4, 4, 3, 7, 0,
    6, 0, 1, 2, 2, 1, 0, 6,
];

/// Inverse of TRIANGLE: maps triangle index back to a representative square.
const INV_TRIANGLE: [usize; 10] = [1, 2, 3, 10, 11, 19, 0, 9, 18, 27];

/// Maps the b1-h1-h7 triangle to 0..=27. Diagonal squares map to 28..=35.
#[rustfmt::skip]
const LOWER: [u64; 64] = [
    28,  0,  1,  2,  3,  4,  5,  6,
     0, 29,  7,  8,  9, 10, 11, 12,
     1,  7, 30, 13, 14, 15, 16, 17,
     2,  8, 13, 31, 18, 19, 20, 21,
     3,  9, 14, 18, 32, 22, 23, 24,
     4, 10, 15, 19, 22, 33, 25, 26,
     5, 11, 16, 20, 23, 25, 34, 27,
     6, 12, 17, 21, 24, 26, 27, 35,
];

/// Remapping for encoding multiple identical pieces.
#[rustfmt::skip]
const MULT_TWIST: [u64; 64] = [
    15, 63, 55, 47, 40, 48, 56, 12,
    62, 11, 39, 31, 24, 32,  8, 57,
    54, 38,  7, 23, 16,  4, 33, 49,
    46, 30, 22,  3,  0, 17, 25, 41,
    45, 29, 21,  2,  1, 18, 26, 42,
    53, 37,  6, 20, 19,  5, 34, 50,
    61, 10, 36, 28, 27, 35,  9, 58,
    14, 60, 52, 44, 43, 51, 59, 13,
];

/// Sentinel for invalid entries in KK_IDX and PP_IDX.
const Z0: u64 = u64::MAX;

/// Encoding of all 462 configurations of two non-connected kings.
/// Indexed by [TRIANGLE[king1_sq]][king2_sq].
#[rustfmt::skip]
const KK_IDX: [[u64; 64]; 10] = [[
     Z0,  Z0,  Z0,   0,   1,   2,   3,   4,
     Z0,  Z0,  Z0,   5,   6,   7,   8,   9,
     10,  11,  12,  13,  14,  15,  16,  17,
     18,  19,  20,  21,  22,  23,  24,  25,
     26,  27,  28,  29,  30,  31,  32,  33,
     34,  35,  36,  37,  38,  39,  40,  41,
     42,  43,  44,  45,  46,  47,  48,  49,
     50,  51,  52,  53,  54,  55,  56,  57,
], [
     58,  Z0,  Z0,  Z0,  59,  60,  61,  62,
     63,  Z0,  Z0,  Z0,  64,  65,  66,  67,
     68,  69,  70,  71,  72,  73,  74,  75,
     76,  77,  78,  79,  80,  81,  82,  83,
     84,  85,  86,  87,  88,  89,  90,  91,
     92,  93,  94,  95,  96,  97,  98,  99,
    100, 101, 102, 103, 104, 105, 106, 107,
    108, 109, 110, 111, 112, 113, 114, 115,
], [
    116, 117,  Z0,  Z0,  Z0, 118, 119, 120,
    121, 122,  Z0,  Z0,  Z0, 123, 124, 125,
    126, 127, 128, 129, 130, 131, 132, 133,
    134, 135, 136, 137, 138, 139, 140, 141,
    142, 143, 144, 145, 146, 147, 148, 149,
    150, 151, 152, 153, 154, 155, 156, 157,
    158, 159, 160, 161, 162, 163, 164, 165,
    166, 167, 168, 169, 170, 171, 172, 173,
], [
    174,  Z0,  Z0,  Z0, 175, 176, 177, 178,
    179,  Z0,  Z0,  Z0, 180, 181, 182, 183,
    184,  Z0,  Z0,  Z0, 185, 186, 187, 188,
    189, 190, 191, 192, 193, 194, 195, 196,
    197, 198, 199, 200, 201, 202, 203, 204,
    205, 206, 207, 208, 209, 210, 211, 212,
    213, 214, 215, 216, 217, 218, 219, 220,
    221, 222, 223, 224, 225, 226, 227, 228,
], [
    229, 230,  Z0,  Z0,  Z0, 231, 232, 233,
    234, 235,  Z0,  Z0,  Z0, 236, 237, 238,
    239, 240,  Z0,  Z0,  Z0, 241, 242, 243,
    244, 245, 246, 247, 248, 249, 250, 251,
    252, 253, 254, 255, 256, 257, 258, 259,
    260, 261, 262, 263, 264, 265, 266, 267,
    268, 269, 270, 271, 272, 273, 274, 275,
    276, 277, 278, 279, 280, 281, 282, 283,
], [
    284, 285, 286, 287, 288, 289, 290, 291,
    292, 293,  Z0,  Z0,  Z0, 294, 295, 296,
    297, 298,  Z0,  Z0,  Z0, 299, 300, 301,
    302, 303,  Z0,  Z0,  Z0, 304, 305, 306,
    307, 308, 309, 310, 311, 312, 313, 314,
    315, 316, 317, 318, 319, 320, 321, 322,
    323, 324, 325, 326, 327, 328, 329, 330,
    331, 332, 333, 334, 335, 336, 337, 338,
], [
     Z0,  Z0, 339, 340, 341, 342, 343, 344,
     Z0,  Z0, 345, 346, 347, 348, 349, 350,
     Z0,  Z0, 441, 351, 352, 353, 354, 355,
     Z0,  Z0,  Z0, 442, 356, 357, 358, 359,
     Z0,  Z0,  Z0,  Z0, 443, 360, 361, 362,
     Z0,  Z0,  Z0,  Z0,  Z0, 444, 363, 364,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 445, 365,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 446,
], [
     Z0,  Z0,  Z0, 366, 367, 368, 369, 370,
     Z0,  Z0,  Z0, 371, 372, 373, 374, 375,
     Z0,  Z0,  Z0, 376, 377, 378, 379, 380,
     Z0,  Z0,  Z0, 447, 381, 382, 383, 384,
     Z0,  Z0,  Z0,  Z0, 448, 385, 386, 387,
     Z0,  Z0,  Z0,  Z0,  Z0, 449, 388, 389,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 450, 390,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 451,
], [
    452, 391, 392, 393, 394, 395, 396, 397,
     Z0,  Z0,  Z0,  Z0, 398, 399, 400, 401,
     Z0,  Z0,  Z0,  Z0, 402, 403, 404, 405,
     Z0,  Z0,  Z0,  Z0, 406, 407, 408, 409,
     Z0,  Z0,  Z0,  Z0, 453, 410, 411, 412,
     Z0,  Z0,  Z0,  Z0,  Z0, 454, 413, 414,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 455, 415,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 456,
], [
    457, 416, 417, 418, 419, 420, 421, 422,
     Z0, 458, 423, 424, 425, 426, 427, 428,
     Z0,  Z0,  Z0,  Z0,  Z0, 429, 430, 431,
     Z0,  Z0,  Z0,  Z0,  Z0, 432, 433, 434,
     Z0,  Z0,  Z0,  Z0,  Z0, 435, 436, 437,
     Z0,  Z0,  Z0,  Z0,  Z0, 459, 438, 439,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 460, 440,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 461,
]];

/// Encoding of a pair of identical pieces.
/// Indexed by [TRIANGLE[piece1_sq]][piece2_sq].
#[rustfmt::skip]
const PP_IDX: [[u64; 64]; 10] = [[
      0,  Z0,   1,   2,   3,   4,   5,   6,
      7,   8,   9,  10,  11,  12,  13,  14,
     15,  16,  17,  18,  19,  20,  21,  22,
     23,  24,  25,  26,  27,  28,  29,  30,
     31,  32,  33,  34,  35,  36,  37,  38,
     39,  40,  41,  42,  43,  44,  45,  46,
     Z0,  47,  48,  49,  50,  51,  52,  53,
     54,  55,  56,  57,  58,  59,  60,  61,
], [
     62,  Z0,  Z0,  63,  64,  65,  Z0,  66,
     Z0,  67,  68,  69,  70,  71,  72,  Z0,
     73,  74,  75,  76,  77,  78,  79,  80,
     81,  82,  83,  84,  85,  86,  87,  88,
     89,  90,  91,  92,  93,  94,  95,  96,
     Z0,  97,  98,  99, 100, 101, 102, 103,
     Z0, 104, 105, 106, 107, 108, 109,  Z0,
    110,  Z0, 111, 112, 113, 114,  Z0, 115,
], [
    116,  Z0,  Z0,  Z0, 117,  Z0,  Z0, 118,
     Z0, 119, 120, 121, 122, 123, 124,  Z0,
     Z0, 125, 126, 127, 128, 129, 130,  Z0,
    131, 132, 133, 134, 135, 136, 137, 138,
     Z0, 139, 140, 141, 142, 143, 144, 145,
     Z0, 146, 147, 148, 149, 150, 151,  Z0,
     Z0, 152, 153, 154, 155, 156, 157,  Z0,
    158,  Z0,  Z0, 159, 160,  Z0,  Z0, 161,
], [
    162,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 163,
     Z0, 164,  Z0, 165, 166, 167, 168,  Z0,
     Z0, 169, 170, 171, 172, 173, 174,  Z0,
     Z0, 175, 176, 177, 178, 179, 180,  Z0,
     Z0, 181, 182, 183, 184, 185, 186,  Z0,
     Z0,  Z0, 187, 188, 189, 190, 191,  Z0,
     Z0, 192, 193, 194, 195, 196, 197,  Z0,
    198,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 199,
], [
    200,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 201,
     Z0, 202,  Z0,  Z0, 203,  Z0, 204,  Z0,
     Z0,  Z0, 205, 206, 207, 208,  Z0,  Z0,
     Z0, 209, 210, 211, 212, 213, 214,  Z0,
     Z0,  Z0, 215, 216, 217, 218, 219,  Z0,
     Z0,  Z0, 220, 221, 222, 223,  Z0,  Z0,
     Z0, 224,  Z0, 225, 226,  Z0, 227,  Z0,
    228,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 229,
], [
    230,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 231,
     Z0, 232,  Z0,  Z0,  Z0,  Z0, 233,  Z0,
     Z0,  Z0, 234,  Z0, 235, 236,  Z0,  Z0,
     Z0,  Z0, 237, 238, 239, 240,  Z0,  Z0,
     Z0,  Z0,  Z0, 241, 242, 243,  Z0,  Z0,
     Z0,  Z0, 244, 245, 246, 247,  Z0,  Z0,
     Z0, 248,  Z0,  Z0,  Z0,  Z0, 249,  Z0,
    250,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 251,
], [
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 259,
     Z0, 252,  Z0,  Z0,  Z0,  Z0, 260,  Z0,
     Z0,  Z0, 253,  Z0,  Z0, 261,  Z0,  Z0,
     Z0,  Z0,  Z0, 254, 262,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0, 255,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0, 256,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 257,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 258,
], [
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 268,  Z0,
     Z0,  Z0, 263,  Z0,  Z0, 269,  Z0,  Z0,
     Z0,  Z0,  Z0, 264, 270,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0, 265,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0, 266,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 267,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,
], [
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0, 274,  Z0,  Z0,
     Z0,  Z0,  Z0, 271, 275,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0, 272,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0, 273,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,
], [
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0, 277,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0, 276,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,
]];

/// The a7-a5-c5 triangle bitboard (used for PP_IDX symmetry normalization).
const TEST45: u64 = 0x1_0307_0000_0000;

// ---------------------------------------------------------------------------
// Precomputed constants
// ---------------------------------------------------------------------------

struct Consts {
    mult_idx: [[u64; 10]; 5],
    mult_factor: [u64; 5],
}

const CONSTS: Consts = Consts::new();

impl Consts {
    const fn new() -> Consts {
        let mut mult_idx = [[0u64; 10]; 5];
        let mut mult_factor = [0u64; 5];

        let mut i = 0;
        while i < 5 {
            let mut s = 0u64;
            let mut j = 0;
            while j < 10 {
                mult_idx[i][j] = s;
                s += if i == 0 {
                    1
                } else {
                    binomial(MULT_TWIST[INV_TRIANGLE[j]], i as u64)
                };
                j += 1;
            }
            mult_factor[i] = s;
            i += 1;
        }

        Consts {
            mult_idx,
            mult_factor,
        }
    }
}

// ---------------------------------------------------------------------------
// Piece representation for table encoding
// ---------------------------------------------------------------------------

/// A piece in the table's piece list (color + type).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TablePiece {
    pub color: Color,
    pub piece_type: PieceType,
}

/// Parse a piece from a Syzygy table header nibble.
/// Nibble encoding: bits 0-2 = piece type (1=P,2=N,3=B,4=R,5=Q,6=K),
/// bit 3 = color (0=white, 1=black).
pub fn piece_from_nibble(nibble: u8) -> Option<TablePiece> {
    let color = if nibble & 8 == 0 {
        Color::White
    } else {
        Color::Black
    };
    let pt = match nibble & 7 {
        1 => PieceType::Pawn,
        2 => PieceType::Knight,
        3 => PieceType::Bishop,
        4 => PieceType::Rook,
        5 => PieceType::Queen,
        6 => PieceType::King,
        _ => return None,
    };
    Some(TablePiece {
        color,
        piece_type: pt,
    })
}

// ---------------------------------------------------------------------------
// Group data and factor computation
// ---------------------------------------------------------------------------

/// Piece grouping and factor information for encoding a subtable.
#[derive(Clone, Debug)]
pub struct GroupData {
    /// Ordered piece list from the table header.
    pub pieces: Vec<TablePiece>,
    /// Length of each group (e.g. [3, 1] for KRvK with 3 leading + 1 remaining).
    pub lens: Vec<usize>,
    /// Factor for each group, plus total size at index lens.len().
    pub factors: Vec<u64>,
}

/// Determine the leading group length based on material configuration.
fn leading_group_len(material: &MaterialKey) -> usize {
    if material.has_pawns() {
        0
    } else {
        let unique = material.unique_pieces();
        let min_like = material.min_like_man();
        if unique >= 3 {
            3
        } else if unique == 2 {
            2
        } else {
            min_like
        }
    }
}

impl GroupData {
    /// Create GroupData from a piece list, computing group lengths and factors.
    ///
    /// `pieces`: ordered piece list from the table header.
    /// `order`: encoding order bytes from the table header.
    /// `material`: material configuration.
    pub fn new(pieces: Vec<TablePiece>, order: [u8; 2], material: &MaterialKey) -> Option<Self> {
        if pieces.len() < 2 {
            return None;
        }

        let has_pawns = material.has_pawns();
        let unique = material.unique_pieces();
        let min_like = material.min_like_man();

        // Compute group lengths.
        let first_len = if has_pawns {
            0
        } else if unique >= 3 {
            3
        } else if unique == 2 {
            2
        } else {
            min_like
        };

        let mut lens = Vec::new();
        if first_len > 0 {
            lens.push(first_len);
        }

        // Group remaining pieces by equality (adjacent identical pieces).
        let mut i = first_len;
        while i < pieces.len() {
            let start = i;
            while i < pieces.len()
                && pieces[i].color == pieces[start].color
                && pieces[i].piece_type == pieces[start].piece_type
            {
                i += 1;
            }
            lens.push(i - start);
        }

        // Both sides have pawns?
        let pp = material.white_pawn_count() > 0 && material.black_pawn_count() > 0;

        // Compute factors.
        let mut factors = vec![0u64; lens.len() + 1];
        let mut free_squares = 64 - lens[0] - if pp { lens[1] } else { 0 };
        let mut next = if pp { 2 } else { 1 };
        let mut idx = 1u64;
        let mut k = 0u8;

        while next < lens.len() || k == order[0] || k == order[1] {
            if k == order[0] {
                factors[0] = idx;

                if has_pawns {
                    // Pawn tables: not yet implemented
                    return None;
                } else if unique >= 3 {
                    idx *= 31_332;
                } else if unique == 2 {
                    idx *= 462;
                } else if min_like == 2 {
                    idx *= 278;
                } else {
                    idx *= CONSTS.mult_factor[min_like - 1];
                }
            } else if k == order[1] {
                factors[1] = idx;
                idx *= binomial(48 - lens[0] as u64, lens[1] as u64);
            } else {
                if next >= lens.len() {
                    break;
                }
                factors[next] = idx;
                idx *= binomial(free_squares as u64, lens[next] as u64);
                free_squares -= lens[next];
                next += 1;
            }
            k += 1;

            // Safety: prevent infinite loop
            if k > 20 {
                return None;
            }
        }

        factors[lens.len()] = idx;

        Some(GroupData {
            pieces,
            lens,
            factors,
        })
    }

    /// Total table size (number of positions in this subtable).
    pub fn tb_size(&self) -> u64 {
        self.factors[self.lens.len()]
    }
}

// ---------------------------------------------------------------------------
// Position encoding
// ---------------------------------------------------------------------------

use crate::table::Table;
use rage_core::Board;

/// Encode a board position for table lookup.
///
/// Returns `(subtable_index, position_index)` or `None` if the position
/// cannot be encoded (pawn tables not implemented, invalid position, etc.).
pub fn encode_position(board: &Board, table: &Table) -> Option<(usize, u64)> {
    if table.has_pawns {
        return None; // Pawn tables not yet implemented
    }

    if table.files.is_empty() || table.files[0].sides.is_empty() {
        return None;
    }

    let material = &table.material;

    // Determine if we need to flip colors.
    let board_mat = MaterialKey::from_board(board);
    let symmetric = material.is_symmetric();
    let symmetric_btm = symmetric && board.side_to_move() == Color::Black;
    let black_stronger = board_mat.table_name() != material.table_name()
        && board_mat.table_name_swapped() != material.table_name();

    // If neither the normal nor swapped key matches, but the swapped one does...
    let black_stronger = if !black_stronger {
        // Check if the board's material matches with colors swapped
        let bm = &board_mat;
        let tm = material;
        bm.counts[0] != tm.counts[0] || bm.counts[1] != tm.counts[1]
    } else {
        black_stronger
    };

    let flip = symmetric_btm || black_stronger;
    let bside = (board.side_to_move() == Color::Black) ^ flip;

    let file_data = &table.files[0]; // pawnless → file 0

    // Select the side (subtable).
    let side_idx = if bside { file_data.sides.len() - 1 } else { 0 };
    let side = &file_data.sides[side_idx];
    let groups = match &side.pairs.groups {
        Some(g) => g,
        None => return Some((side_idx, 0)), // single-value table: any index works
    };

    // Collect squares in the order specified by the piece list.
    let mut squares: Vec<usize> = Vec::with_capacity(groups.pieces.len());
    let mut used = 0u64; // bitmask of used squares

    for piece in &groups.pieces {
        let color = if flip { !piece.color } else { piece.color };
        let pt = piece.piece_type;

        // Find the first unused square with this piece type and color.
        let bb = board.pieces(color, pt);
        let mut found = None;
        for sq in bb {
            if used & (1u64 << sq.index()) == 0 {
                found = Some(sq);
                break;
            }
        }
        let sq = found?;
        used |= 1u64 << sq.index();

        // If flip, also flip the square vertically.
        let sq_idx = if flip {
            flip_vert(sq.index())
        } else {
            sq.index()
        };
        squares.push(sq_idx);
    }

    // Apply pawnless normalization.
    // Step 1: If first square's file >= E, flip all horizontally.
    if (squares[0] & 7) >= 4 {
        for sq in &mut squares {
            *sq = flip_horiz(*sq);
        }
    }

    // Step 2: If first square's rank >= 5, flip all vertically.
    if (squares[0] >> 3) >= 4 {
        for sq in &mut squares {
            *sq = flip_vert(*sq);
        }
    }

    // Step 3: Check diagonal. Find first piece in leading group that's off-diagonal.
    // If it's above the diagonal, flip all along the diagonal.
    for i in 0..groups.lens[0] {
        let file = squares[i] & 7;
        let rank = squares[i] >> 3;
        if file == rank {
            continue; // on diagonal, check next
        }
        if rank > file {
            // Above diagonal: flip all remaining squares diagonally.
            for sq in &mut squares[i..] {
                *sq = flip_diag(*sq);
            }
        }
        break;
    }

    // Compute leading group index.
    let num_unique = table.num_unique_pieces;
    let min_like = table.min_like_man;

    let leading_idx = if num_unique > 2 {
        // 3+ unique pieces.
        let adjust1 = u64::from(squares[1] > squares[0]);
        let adjust2 = u64::from(squares[2] > squares[0]) + u64::from(squares[2] > squares[1]);

        if offdiag(squares[0]) {
            TRIANGLE[squares[0]] * 63 * 62
                + (squares[1] as u64 - adjust1) * 62
                + (squares[2] as u64 - adjust2)
        } else if offdiag(squares[1]) {
            6 * 63 * 62
                + (squares[0] >> 3) as u64 * 28 * 62
                + LOWER[squares[1]] * 62
                + squares[2] as u64
                - adjust2
        } else if offdiag(squares[2]) {
            6 * 63 * 62
                + 4 * 28 * 62
                + (squares[0] >> 3) as u64 * 7 * 28
                + ((squares[1] >> 3) as u64 - adjust1) * 28
                + LOWER[squares[2]]
        } else {
            6 * 63 * 62
                + 4 * 28 * 62
                + 4 * 7 * 28
                + (squares[0] >> 3) as u64 * 7 * 6
                + ((squares[1] >> 3) as u64 - adjust1) * 6
                + ((squares[2] >> 3) as u64 - adjust2)
        }
    } else if num_unique == 2 {
        // Two kings only.
        let idx = KK_IDX[TRIANGLE[squares[0]] as usize][squares[1]];
        if idx == Z0 {
            return None; // Adjacent kings (invalid)
        }
        idx
    } else if min_like == 2 {
        // Pair of identical pieces.
        let mut sq = squares.clone();

        if TRIANGLE[sq[0]] > TRIANGLE[sq[1]] {
            sq.swap(0, 1);
        }

        if (sq[0] & 7) >= 4 {
            for s in &mut sq {
                *s = flip_horiz(*s);
            }
        }

        if (sq[0] >> 3) >= 4 {
            for s in &mut sq {
                *s = flip_vert(*s);
            }
        }

        let file0 = sq[0] & 7;
        let rank0 = sq[0] >> 3;
        let file1 = sq[1] & 7;
        let rank1 = sq[1] >> 3;
        if rank0 > file0 || (!offdiag(sq[0]) && rank1 > file1) {
            for s in &mut sq {
                *s = flip_diag(*s);
            }
        }

        // Check TEST45 triangle.
        if (TEST45 & (1u64 << sq[1])) != 0 && TRIANGLE[sq[0]] == TRIANGLE[sq[1]] {
            sq.swap(0, 1);
            for s in &mut sq {
                *s = flip_diag(flip_vert(*s));
            }
        }

        let idx = PP_IDX[TRIANGLE[sq[0]] as usize][sq[1]];
        if idx == Z0 {
            return None;
        }

        // Update squares for remaining groups.
        squares[0] = sq[0];
        squares[1] = sq[1];
        idx
    } else {
        // Multiple identical pieces (3+).
        let len0 = groups.lens[0];
        let mut sq = squares.clone();

        // Sort to find the smallest TRIANGLE value.
        for i in 1..len0 {
            if TRIANGLE[sq[0]] > TRIANGLE[sq[i]] {
                sq.swap(0, i);
            }
        }

        if (sq[0] & 7) >= 4 {
            for s in &mut sq {
                *s = flip_horiz(*s);
            }
        }

        if (sq[0] >> 3) >= 4 {
            for s in &mut sq {
                *s = flip_vert(*s);
            }
        }

        if (sq[0] >> 3) > (sq[0] & 7) {
            for s in &mut sq {
                *s = flip_diag(*s);
            }
        }

        // Sort remaining by MULT_TWIST.
        for i in 1..len0 {
            for j in (i + 1)..len0 {
                if MULT_TWIST[sq[i]] > MULT_TWIST[sq[j]] {
                    sq.swap(i, j);
                }
            }
        }

        let mut idx = CONSTS.mult_idx[len0 - 1][TRIANGLE[sq[0]] as usize];
        for i in 1..len0 {
            idx += binomial(MULT_TWIST[sq[i]], i as u64);
        }

        // Update squares.
        squares[..sq.len()].copy_from_slice(&sq[..]);

        idx
    };

    let mut idx = leading_idx * groups.factors[0];

    // Encode remaining groups.
    let mut group_sq = groups.lens[0];
    let mut next = 1;
    let remaining_pawns = material.white_pawn_count() > 0 && material.black_pawn_count() > 0;
    let mut rp = remaining_pawns;

    for grp_idx in 1..groups.lens.len() {
        let lens = groups.lens[grp_idx];
        let group_start = group_sq;
        let group_end = group_sq + lens;

        // Sort squares within this group.
        squares[group_start..group_end].sort();

        let mut n = 0u64;
        for (i, &group_square) in squares[group_start..group_end].iter().enumerate() {
            // Count how many previously placed squares are below this one.
            let adjust = squares[..group_start]
                .iter()
                .filter(|&&s| group_square > s)
                .count() as u64;

            n += binomial(
                group_square as u64 - adjust - if rp { 8 } else { 0 },
                i as u64 + 1,
            );
        }

        rp = false;
        idx += n * groups.factors[next];
        group_sq += lens;
        next += 1;
    }

    Some((side_idx, idx))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn binomial_base_cases() {
        assert_eq!(binomial(0, 0), 1);
        assert_eq!(binomial(5, 0), 1);
        assert_eq!(binomial(5, 1), 5);
        assert_eq!(binomial(10, 3), 120);
        assert_eq!(binomial(62, 2), 1891);
        assert_eq!(binomial(63, 2), 1953);
    }

    #[test]
    fn triangle_symmetry() {
        // a1 = index 0, TRIANGLE = 6 (diagonal)
        assert_eq!(TRIANGLE[0], 6);
        // b1 = index 1, TRIANGLE = 0 (off-diagonal)
        assert_eq!(TRIANGLE[1], 0);
        // d4 = index 27, TRIANGLE = 9 (diagonal)
        assert_eq!(TRIANGLE[27], 9);
    }

    #[test]
    fn offdiag_test() {
        // a1: file=0, rank=0, on diagonal
        assert!(!offdiag(0));
        // b1: file=1, rank=0, off diagonal
        assert!(offdiag(1));
        // b2: file=1, rank=1, on diagonal
        assert!(!offdiag(9));
    }

    #[test]
    fn kk_idx_valid_entries() {
        // Count non-Z0 entries: should be 462
        let count: usize = KK_IDX
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&v| v != Z0)
            .count();
        assert_eq!(count, 462);
    }

    #[test]
    fn pp_idx_valid_entries() {
        // Count non-Z0 entries: should be 278
        let count: usize = PP_IDX
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&v| v != Z0)
            .count();
        assert_eq!(count, 278);
    }

    #[test]
    fn mult_factor_values() {
        // mult_factor[0] = 10 (10 triangle positions)
        assert_eq!(CONSTS.mult_factor[0], 10);
        // mult_factor[1] should be > 10
        assert!(CONSTS.mult_factor[1] > 10);
    }
}
