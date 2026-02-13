use rage_core::*;

/// Simple xorshift64 PRNG (same algorithm as zobrist.rs). Seed must be nonzero.
pub struct Xorshift(pub u64);

impl Xorshift {
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Verify all structural invariants of a Board.
///
/// Checks bitboard/mailbox agreement, disjointness, king count, pawn limits,
/// piece count limits, and Zobrist consistency. Panics with a descriptive
/// message if any invariant is violated.
pub fn assert_board_consistent(board: &Board) {
    // 1. Mailbox agrees with occupied bitboard on every square.
    for sq in Square::ALL {
        let piece = board.piece_on(sq);
        let in_occupied = board.occupied().contains(sq);
        assert_eq!(
            piece.is_some(),
            in_occupied,
            "mailbox/occupied mismatch on {sq}: piece_on={piece:?}, in_occupied={in_occupied}"
        );
        if let Some(p) = piece {
            assert!(
                board.pieces(p.color, p.piece_type).contains(sq),
                "{p} on {sq} missing from pieces({:?}, {:?})",
                p.color,
                p.piece_type
            );
            assert!(
                board.color_bb(p.color).contains(sq),
                "{p} on {sq} missing from color_bb({:?})",
                p.color
            );
        }
    }

    // 2. Color bitboards are disjoint.
    assert!(
        (board.color_bb(Color::White) & board.color_bb(Color::Black)).is_empty(),
        "white and black bitboards overlap"
    );

    // 3. Color bitboards union to occupied.
    assert_eq!(
        board.color_bb(Color::White) | board.color_bb(Color::Black),
        board.occupied(),
        "color BBs don't union to occupied"
    );

    // 4. Piece-type bitboards are pairwise disjoint.
    for (i, &pt_i) in PieceType::ALL.iter().enumerate() {
        for &pt_j in &PieceType::ALL[i + 1..] {
            assert!(
                (board.piece_bb(pt_i) & board.piece_bb(pt_j)).is_empty(),
                "piece BBs for {pt_i:?} and {pt_j:?} overlap"
            );
        }
    }

    // 5. Piece-type bitboards union to occupied.
    let mut all_pieces = Bitboard::EMPTY;
    for pt in PieceType::ALL {
        all_pieces = all_pieces | board.piece_bb(pt);
    }
    assert_eq!(
        all_pieces,
        board.occupied(),
        "piece-type BBs don't union to occupied"
    );

    // 6. Exactly one king per side.
    assert_eq!(
        board.pieces(Color::White, PieceType::King).popcount(),
        1,
        "white must have exactly 1 king"
    );
    assert_eq!(
        board.pieces(Color::Black, PieceType::King).popcount(),
        1,
        "black must have exactly 1 king"
    );

    // 7. Pawns: at most 8 per side, none on rank 1 or 8.
    assert!(
        board.pieces(Color::White, PieceType::Pawn).popcount() <= 8,
        "white has more than 8 pawns"
    );
    assert!(
        board.pieces(Color::Black, PieceType::Pawn).popcount() <= 8,
        "black has more than 8 pawns"
    );
    let pawns = board.piece_bb(PieceType::Pawn);
    assert!(
        (pawns & Bitboard::RANK_1).is_empty(),
        "pawns found on rank 1"
    );
    assert!(
        (pawns & Bitboard::RANK_8).is_empty(),
        "pawns found on rank 8"
    );

    // 8. At most 16 pieces per side.
    assert!(
        board.color_bb(Color::White).popcount() <= 16,
        "white has more than 16 pieces"
    );
    assert!(
        board.color_bb(Color::Black).popcount() <= 16,
        "black has more than 16 pieces"
    );

    // 9. Zobrist hash matches full recomputation.
    assert_eq!(
        board.zobrist_hash(),
        board.compute_zobrist(),
        "incremental Zobrist doesn't match full recompute"
    );
}
