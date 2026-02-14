use rage_core::{Bitboard, Board, Color, Move, PieceType, Square, attacks};

use crate::eval::piece_value;

/// SEE piece values (same as eval piece values, but king gets a large value
/// so it can participate in the exchange sequence correctly).
const SEE_VALUES: [i32; PieceType::NUM] = [100, 320, 330, 500, 900, 20_000];

fn see_piece_value(pt: PieceType) -> i32 {
    SEE_VALUES[pt.index()]
}

/// Returns the least valuable attacker of `sq` from `side` given `occupied`,
/// and its piece type. Returns None if no attacker exists.
fn least_valuable_attacker(
    board: &Board,
    sq: Square,
    side: Color,
    occupied: Bitboard,
) -> Option<(Square, PieceType)> {
    // Check piece types from least to most valuable
    let side_bb = board.color_bb(side) & occupied;

    // Pawns
    let pawn_attackers =
        attacks::pawn_attacks(sq, !side) & board.piece_bb(PieceType::Pawn) & side_bb;
    if let Some(s) = pawn_attackers.lsb() {
        return Some((s, PieceType::Pawn));
    }

    // Knights
    let knight_attackers =
        attacks::knight_attacks(sq) & board.piece_bb(PieceType::Knight) & side_bb;
    if let Some(s) = knight_attackers.lsb() {
        return Some((s, PieceType::Knight));
    }

    // Bishops
    let bishop_attackers =
        attacks::bishop_attacks(sq, occupied) & board.piece_bb(PieceType::Bishop) & side_bb;
    if let Some(s) = bishop_attackers.lsb() {
        return Some((s, PieceType::Bishop));
    }

    // Rooks
    let rook_attackers =
        attacks::rook_attacks(sq, occupied) & board.piece_bb(PieceType::Rook) & side_bb;
    if let Some(s) = rook_attackers.lsb() {
        return Some((s, PieceType::Rook));
    }

    // Queens
    let queen_attackers = (attacks::bishop_attacks(sq, occupied)
        | attacks::rook_attacks(sq, occupied))
        & board.piece_bb(PieceType::Queen)
        & side_bb;
    if let Some(s) = queen_attackers.lsb() {
        return Some((s, PieceType::Queen));
    }

    // Kings
    let king_attackers = attacks::king_attacks(sq) & board.piece_bb(PieceType::King) & side_bb;
    if let Some(s) = king_attackers.lsb() {
        return Some((s, PieceType::King));
    }

    None
}

/// Static Exchange Evaluation: returns the net material gain of the capture
/// sequence starting with move `m`.
pub fn see(board: &Board, m: Move) -> i32 {
    let from = m.from_sq();
    let to = m.to_sq();

    // Determine initial gain
    let gain = if m.is_en_passant() {
        piece_value(PieceType::Pawn)
    } else if m.is_promotion() {
        // For promotions: gain the captured piece + (promo_value - pawn_value)
        let captured_val = board
            .piece_on(to)
            .map_or(0, |p| see_piece_value(p.piece_type));
        captured_val + see_piece_value(m.promotion_piece()) - see_piece_value(PieceType::Pawn)
    } else {
        board
            .piece_on(to)
            .map_or(0, |p| see_piece_value(p.piece_type))
    };

    // The attacker piece type
    let attacker_pt = if m.is_promotion() {
        m.promotion_piece()
    } else {
        match board.piece_on(from) {
            Some(p) => p.piece_type,
            None => return 0,
        }
    };

    let stm = board.side_to_move();
    let mut side = !stm; // opponent responds next
    let mut occupied = board.occupied();

    // Remove the initial attacker from occupied
    occupied &= !Bitboard::from_square(from);
    // For en passant, also remove the captured pawn
    if m.is_en_passant() {
        let ep_capture_sq = Square::from_file_rank(to.file(), from.rank());
        occupied &= !Bitboard::from_square(ep_capture_sq);
    }

    // Build the gain stack
    let mut gains = [0i32; 32];
    gains[0] = gain;
    let mut depth = 1;
    let mut current_attacker_value = see_piece_value(attacker_pt);

    loop {
        // Find the least valuable attacker for `side` on `to`
        let Some((attacker_sq, pt)) = least_valuable_attacker(board, to, side, occupied) else {
            break;
        };

        // Record the gain: capture current piece on the square
        gains[depth] = current_attacker_value - gains[depth - 1];
        current_attacker_value = see_piece_value(pt);

        // Pruning: if even capturing for free doesn't improve, we can stop
        if gains[depth].max(-gains[depth - 1]) < 0 {
            depth += 1;
            break;
        }

        // Remove this attacker from occupied (reveals x-ray attackers)
        occupied &= !Bitboard::from_square(attacker_sq);
        side = !side;
        depth += 1;

        if depth >= 32 {
            break;
        }
    }

    // Negamax the gain stack
    depth -= 1;
    while depth > 0 {
        gains[depth - 1] = -((-gains[depth - 1]).max(gains[depth]));
        depth -= 1;
    }

    gains[0]
}

/// Returns true if the SEE of move `m` is >= `threshold`.
/// Early-exit optimization for when we only need a yes/no answer.
pub fn see_ge(board: &Board, m: Move, threshold: i32) -> bool {
    see(board, m) >= threshold
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use rage_core::Board;

    #[test]
    fn pawn_captures_queen_defended_pawn() {
        // d5 pawn defended by Qd8 through open d7 square
        let board = Board::from_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
            .unwrap();
        let m = Move::new(Square::E4, Square::D5);
        // PxP (+100), QxP (-100) => net 0
        assert_eq!(see(&board, m), 0);
    }

    #[test]
    fn pawn_captures_truly_undefended_pawn() {
        // d5 pawn with no defenders
        let board = Board::from_fen("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
        let m = Move::new(Square::E4, Square::D5);
        assert_eq!(see(&board, m), 100);
    }

    #[test]
    fn pawn_captures_knight_defended_by_pawn() {
        // e4 pawn captures Nd5, d5 is defended by c6 pawn
        let board =
            Board::from_fen("r1bqkbnr/pp2pppp/2p5/3n4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3")
                .unwrap();
        let m = Move::new(Square::E4, Square::D5);
        // PxN (+320), pxP (-100) => net 220
        assert_eq!(see(&board, m), 220);
    }

    #[test]
    fn knight_captures_pawn_defended_by_queen() {
        // Nd5 captures e7 pawn, which is defended by Qd8
        let board =
            Board::from_fen("rnbqkb1r/pppppppp/8/3N4/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 2").unwrap();
        let m = Move::new(Square::D5, Square::E7);
        // NxP (+100), QxN (-320) => net -220
        let score = see(&board, m);
        assert!(score < 0, "NxP defended by queen should lose, got {score}");
    }

    #[test]
    fn queen_captures_undefended_pawn() {
        // Qd3 captures isolated pawn on d5 (no defenders visible)
        let board = Board::from_fen("4k3/8/8/3p4/8/3Q4/8/4K3 w - - 0 1").unwrap();
        let m = Move::new(Square::D3, Square::D5);
        assert_eq!(see(&board, m), 100);
    }

    #[test]
    fn queen_captures_pawn_defended_by_pawn() {
        // Qd3 captures d5 pawn, defended by e6 pawn
        let board = Board::from_fen("4k3/8/4p3/3p4/8/3Q4/8/4K3 w - - 0 1").unwrap();
        let m = Move::new(Square::D3, Square::D5);
        // QxP (+100), pxQ (-900) => net -800
        let score = see(&board, m);
        assert!(score < 0, "QxP defended by pawn should lose, got {score}");
    }

    #[test]
    fn see_ge_threshold() {
        // Truly undefended pawn capture
        let board = Board::from_fen("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
        let m = Move::new(Square::E4, Square::D5);
        assert!(see_ge(&board, m, 0));
        assert!(see_ge(&board, m, 100));
        assert!(!see_ge(&board, m, 200));
    }

    #[test]
    fn xray_rook_behind_rook() {
        // White: Re1 + Re2, Black: Re7 + Re8. White Rxe7.
        // RxR (+500), RxR (-500), RxR (+500) => net 500
        let board = Board::from_fen("4r3/4r3/8/8/8/8/4R3/4RK1k w - - 0 1").unwrap();
        let m = Move::new(Square::E1, Square::E7);
        assert_eq!(see(&board, m), 500);
    }

    #[test]
    fn en_passant_capture_defended() {
        // EP: Exf6, but e7 pawn can recapture on f6
        let board =
            Board::from_fen("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
                .unwrap();
        let m = Move::new_en_passant(Square::E5, Square::F6);
        // PxP (+100), pxP (-100) => net 0
        assert_eq!(see(&board, m), 0);
    }

    #[test]
    fn en_passant_capture_undefended() {
        // EP: Exf6, f6 is not defended
        let board = Board::from_fen("4k3/8/8/4Pp2/8/8/8/4K3 w - f6 0 1").unwrap();
        let m = Move::new_en_passant(Square::E5, Square::F6);
        assert_eq!(see(&board, m), 100);
    }

    #[test]
    fn equal_trade_knights() {
        // White Nf3 captures Nd4, defended by nothing else
        let board = Board::from_fen("4k3/8/8/8/3n4/5N2/8/4K3 w - - 0 1").unwrap();
        let m = Move::new(Square::F3, Square::D4);
        // NxN (+320), no recapture => net 320
        assert_eq!(see(&board, m), 320);
    }

    #[test]
    fn rook_captures_defended_pawn() {
        // Rd1 captures d5 pawn defended by e6 pawn
        let board = Board::from_fen("4k3/8/4p3/3p4/8/8/8/3RK3 w - - 0 1").unwrap();
        let m = Move::new(Square::D1, Square::D5);
        // RxP (+100), pxR (-500) => net -400
        let score = see(&board, m);
        assert!(score < 0, "RxP defended by pawn should lose, got {score}");
    }
}
