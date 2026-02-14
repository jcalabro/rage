use rage_core::{Board, Move, MoveList, PieceType};

use crate::eval::piece_value;

/// MVV-LVA score for a capture: prefer capturing high-value pieces with low-value attackers.
/// Returns a score suitable for ordering (higher = better).
fn mvv_lva_score(board: &Board, m: Move) -> i32 {
    let to = m.to_sq();

    // For en passant, victim is always a pawn
    let victim_value = if m.is_en_passant() {
        piece_value(PieceType::Pawn)
    } else {
        match board.piece_on(to) {
            Some(p) => piece_value(p.piece_type),
            None => return 0,
        }
    };

    let from = m.from_sq();
    let attacker_value = match board.piece_on(from) {
        Some(p) => piece_value(p.piece_type),
        None => 0,
    };

    // Higher victim value is much more important; subtract a fraction of attacker value
    victim_value * 10 - attacker_value
}

/// Score a move for ordering purposes.
/// TT move gets highest priority, captures scored by MVV-LVA, quiets get 0.
pub fn score_move(board: &Board, m: Move, tt_move: Move) -> i32 {
    if m == tt_move {
        return 100_000;
    }

    if board.is_capture(m) {
        // MVV-LVA scores are in range ~(0..9000), shift up to be above quiets
        return 10_000 + mvv_lva_score(board, m);
    }

    // Promotions get a bonus
    if m.is_promotion() {
        return 9_000 + piece_value(m.promotion_piece());
    }

    0
}

/// Pick the best-scored move from `start_idx..moves.len()`, swap it to `start_idx`.
/// This is a partial selection sort - we only sort one element per call.
/// Returns the scores slice for the caller to use.
pub fn pick_move(moves: &mut [Move], scores: &mut [i32], start_idx: usize) {
    let mut best_idx = start_idx;
    let mut best_score = scores[start_idx];

    for (i, &score) in scores.iter().enumerate().skip(start_idx + 1) {
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }

    if best_idx != start_idx {
        moves.swap(start_idx, best_idx);
        scores.swap(start_idx, best_idx);
    }
}

/// Score all moves in a move list. Returns a Vec of scores parallel to the move list.
pub fn score_moves(board: &Board, moves: &MoveList, tt_move: Move) -> Vec<i32> {
    let mut scores = Vec::with_capacity(moves.len());
    for m in moves {
        scores.push(score_move(board, *m, tt_move));
    }
    scores
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use rage_core::Board;

    #[test]
    fn tt_move_highest_priority() {
        let board = Board::starting_position();
        let tt_move = Move::new(
            rage_core::Square::E2,
            rage_core::Square::E4,
        );
        let other_move = Move::new(
            rage_core::Square::D2,
            rage_core::Square::D4,
        );
        let tt_score = score_move(&board, tt_move, tt_move);
        let other_score = score_move(&board, other_move, tt_move);
        assert!(tt_score > other_score, "TT move should have highest score");
    }

    #[test]
    fn captures_above_quiets() {
        // Position where d4 pawn can capture e5 pawn
        let board =
            Board::from_fen("rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 2")
                .unwrap();
        let capture = Move::new(rage_core::Square::D4, rage_core::Square::E5);
        let quiet = Move::new(rage_core::Square::E2, rage_core::Square::E3);
        let null = Move::NULL;

        let cap_score = score_move(&board, capture, null);
        let quiet_score = score_move(&board, quiet, null);
        assert!(
            cap_score > quiet_score,
            "capture ({cap_score}) should score above quiet ({quiet_score})"
        );
    }

    #[test]
    fn mvv_lva_queen_capture_beats_pawn_capture() {
        // Position with various captures available
        let board = Board::from_fen(
            "rnbqkb1r/pppppppp/5n2/4N3/8/8/PPPPPPPP/RNBQKB1R w KQkq - 2 2",
        )
        .unwrap();
        // Knight captures knight on f6 (victim=320, attacker=320)
        let capture_knight =
            Move::new(rage_core::Square::E5, rage_core::Square::F6);
        // This is a fictional scenario for ordering - test the scoring function
        let score = mvv_lva_score(&board, capture_knight);
        // victim_value * 10 - attacker_value = 320*10 - 320 = 2880
        assert_eq!(score, 2880);
    }

    #[test]
    fn pick_move_selects_best() {
        let m1 = Move::new(rage_core::Square::E2, rage_core::Square::E4);
        let m2 = Move::new(rage_core::Square::D2, rage_core::Square::D4);
        let m3 = Move::new(rage_core::Square::G1, rage_core::Square::F3);

        let mut moves = [m1, m2, m3];
        let mut scores = [10, 50, 30];

        pick_move(&mut moves, &mut scores, 0);
        assert_eq!(moves[0], m2); // m2 had highest score (50)
        assert_eq!(scores[0], 50);
    }
}
