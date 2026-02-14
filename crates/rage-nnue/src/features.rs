use rage_core::{Color, PieceType, Square};

use crate::arch::{FEATURES_PER_PERSPECTIVE, NUM_KING_BUCKETS};

/// King bucket lookup table. Maps the 32 "half-board" king squares (files a-d
/// after mirroring) to one of 10 buckets. Indexed by `half_sq` where
/// `half_sq = rank * 4 + file` for files 0..3.
///
/// Layout (from White's perspective, rank 0 = rank 1 at bottom):
/// ```text
///   Rank 8:  9  9  8  8  |  mirrored
///   Rank 7:  7  7  7  7  |
///   Rank 6:  6  6  6  6  |
///   Rank 5:  5  5  5  5  |
///   Rank 4:  4  4  4  4  |
///   Rank 3:  3  3  3  3  |
///   Rank 2:  2  2  2  2  |
///   Rank 1:  0  0  1  1  |
///           a  b  c  d    e  f  g  h
/// ```
const KING_BUCKET_TABLE: [usize; 32] = [
    // rank 1 (a1-d1)
    0, 0, 1, 1, // rank 2
    2, 2, 2, 2, // rank 3
    3, 3, 3, 3, // rank 4
    4, 4, 4, 4, // rank 5
    5, 5, 5, 5, // rank 6
    6, 6, 6, 6, // rank 7
    7, 7, 7, 7, // rank 8 (a8-d8)
    8, 8, 9, 9,
];

/// Returns whether the king is on the queenside (files a-d).
fn is_queenside(sq: Square) -> bool {
    sq.file().index() < 4
}

/// Compute the king bucket for a given king square.
/// The board is horizontally mirrored so that files e-h map to d-a,
/// then the resulting half-board square indexes into the bucket table.
pub fn king_bucket(king_sq: Square) -> usize {
    let sq = if is_queenside(king_sq) {
        king_sq
    } else {
        king_sq.flip_file()
    };
    let half_sq = sq.rank().index() * 4 + sq.file().index();
    KING_BUCKET_TABLE[half_sq]
}

/// Orient a square for feature indexing.
/// - For the White perspective: no rank flip needed.
/// - For the Black perspective: flip rank (so Black's back rank = rank 0).
/// - If the perspective's king is on files e-h, flip file (horizontal mirror).
fn orient_square(perspective: Color, king_sq: Square, sq: Square) -> Square {
    let mut s = match perspective {
        Color::White => sq,
        Color::Black => sq.flip_rank(),
    };
    if !is_queenside(king_sq) {
        s = s.flip_file();
    }
    s
}

/// Compute the feature index for a piece from a given perspective.
///
/// Index = bucket * 768 + color_offset * 384 + piece_type * 64 + oriented_square
///
/// `color_offset`: 0 if the piece color matches the perspective, 1 otherwise.
/// This means "own pieces" vs "opponent pieces" from each perspective.
pub fn feature_index(
    perspective: Color,
    king_sq: Square,
    piece_color: Color,
    piece_type: PieceType,
    piece_sq: Square,
) -> usize {
    let king_for_orient = match perspective {
        Color::White => king_sq,
        Color::Black => king_sq.flip_rank(),
    };
    let bucket = king_bucket(king_for_orient);
    let oriented_sq = orient_square(perspective, king_for_orient, piece_sq);
    let color_offset = if piece_color == perspective { 0 } else { 1 };

    bucket * FEATURES_PER_PERSPECTIVE
        + color_offset * 384
        + piece_type.index() * 64
        + oriented_sq.index()
}

/// Maximum valid feature index (exclusive upper bound).
pub const MAX_FEATURE_INDEX: usize = NUM_KING_BUCKETS * FEATURES_PER_PERSPECTIVE;

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn king_bucket_range() {
        for sq in Square::ALL {
            let bucket = king_bucket(sq);
            assert!(
                bucket < NUM_KING_BUCKETS,
                "bucket {bucket} out of range for {sq}"
            );
        }
    }

    #[test]
    fn king_bucket_symmetric() {
        // Files a-d and e-h should produce the same bucket (horizontal mirror).
        for rank in 0..8u8 {
            for file in 0..4u8 {
                let left =
                    Square::from_file_rank(rage_core::File::new(file), rage_core::Rank::new(rank));
                let right = left.flip_file();
                assert_eq!(
                    king_bucket(left),
                    king_bucket(right),
                    "bucket mismatch for {left} vs {right}"
                );
            }
        }
    }

    #[test]
    fn feature_index_range() {
        for perspective in [Color::White, Color::Black] {
            for king_sq in Square::ALL {
                for &pt in &PieceType::ALL {
                    for &pc in &[Color::White, Color::Black] {
                        for sq in Square::ALL {
                            let idx = feature_index(perspective, king_sq, pc, pt, sq);
                            assert!(
                                idx < MAX_FEATURE_INDEX,
                                "feature index {idx} out of range for perspective={perspective:?} king={king_sq} color={pc:?} pt={pt:?} sq={sq}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn feature_index_unique_for_distinct_inputs() {
        // For a fixed perspective and king square, all (color, pt, sq) triples
        // should produce unique feature indices.
        let perspective = Color::White;
        let king_sq = Square::E1;

        let mut indices = std::collections::HashSet::new();
        for &pt in &PieceType::ALL {
            for &pc in &[Color::White, Color::Black] {
                for sq in Square::ALL {
                    let idx = feature_index(perspective, king_sq, pc, pt, sq);
                    assert!(
                        indices.insert(idx),
                        "duplicate feature index {idx} for color={pc:?} pt={pt:?} sq={sq}"
                    );
                }
            }
        }

        // 6 piece types * 2 colors * 64 squares = 768 unique features
        assert_eq!(indices.len(), 768);
    }

    #[test]
    fn king_bucket_known_values() {
        // White king on e1 -> flip to d1 (file 3, rank 0) -> bucket 1
        assert_eq!(king_bucket(Square::E1), 1);
        // White king on a1 -> file 0, rank 0 -> bucket 0
        assert_eq!(king_bucket(Square::A1), 0);
        // White king on d4 -> file 3, rank 3 -> bucket 4
        assert_eq!(king_bucket(Square::D4), 4);
        // White king on h8 -> flip to a8 (file 0, rank 7) -> bucket 8
        assert_eq!(king_bucket(Square::H8), 8);
    }
}
