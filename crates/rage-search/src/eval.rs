use rage_core::{Board, Color, PieceType, Square};

/// Mate score baseline (checkmate). Actual mate scores are MATE_SCORE - ply.
pub const MATE_SCORE: i32 = 30_000;

/// Score representing infinity (used for initial alpha/beta bounds).
pub const INFINITY: i32 = 31_000;

/// Material values in centipawns.
const MATERIAL_VALUE: [i32; PieceType::NUM] = [
    100, // Pawn
    320, // Knight
    330, // Bishop
    500, // Rook
    900, // Queen
    0,   // King (not counted in material)
];

/// Returns the material value of the given piece type in centipawns.
pub fn piece_value(pt: PieceType) -> i32 {
    MATERIAL_VALUE[pt.index()]
}

/// Game phase weight per piece type (for phase interpolation).
/// Total starting phase = 4*1 + 4*1 + 4*2 + 2*4 = 24
const PHASE_WEIGHT: [i32; PieceType::NUM] = [
    0, // Pawn
    1, // Knight
    1, // Bishop
    2, // Rook
    4, // Queen
    0, // King
];

const TOTAL_PHASE: i32 = 24;

// ---------------------------------------------------------------------------
// Piece-Square Tables (from white's perspective, A1=index 0)
//
// Middlegame tables adapted from PeSTO / CPW.
// Each table is [64] indexed by square (LERF: a1=0, h8=63).
// Positive = bonus, negative = penalty.
// ---------------------------------------------------------------------------

#[rustfmt::skip]
const MG_PAWN_TABLE: [i32; Square::NUM] = [
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
];

#[rustfmt::skip]
const EG_PAWN_TABLE: [i32; Square::NUM] = [
      0,   0,   0,   0,   0,   0,   0,   0,
     80,  80,  80,  80,  80,  80,  80,  80,
     50,  50,  50,  50,  50,  50,  50,  50,
     30,  30,  30,  30,  30,  30,  30,  30,
     20,  20,  20,  20,  20,  20,  20,  20,
     10,  10,  10,  10,  10,  10,  10,  10,
     10,  10,  10,  10,  10,  10,  10,  10,
      0,   0,   0,   0,   0,   0,   0,   0,
];

#[rustfmt::skip]
const MG_KNIGHT_TABLE: [i32; Square::NUM] = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
];

#[rustfmt::skip]
const EG_KNIGHT_TABLE: [i32; Square::NUM] = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
];

#[rustfmt::skip]
const MG_BISHOP_TABLE: [i32; Square::NUM] = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
];

#[rustfmt::skip]
const EG_BISHOP_TABLE: [i32; Square::NUM] = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
];

#[rustfmt::skip]
const MG_ROOK_TABLE: [i32; Square::NUM] = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   5,   5,   0,   0,   0,
];

#[rustfmt::skip]
const EG_ROOK_TABLE: [i32; Square::NUM] = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   5,   5,   0,   0,   0,
];

#[rustfmt::skip]
const MG_QUEEN_TABLE: [i32; Square::NUM] = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
];

#[rustfmt::skip]
const EG_QUEEN_TABLE: [i32; Square::NUM] = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
];

#[rustfmt::skip]
const MG_KING_TABLE: [i32; Square::NUM] = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
];

#[rustfmt::skip]
const EG_KING_TABLE: [i32; Square::NUM] = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
];

/// MG PST indexed by [PieceType][Square].
const MG_PST: [[i32; Square::NUM]; PieceType::NUM] = [
    MG_PAWN_TABLE,
    MG_KNIGHT_TABLE,
    MG_BISHOP_TABLE,
    MG_ROOK_TABLE,
    MG_QUEEN_TABLE,
    MG_KING_TABLE,
];

/// EG PST indexed by [PieceType][Square].
const EG_PST: [[i32; Square::NUM]; PieceType::NUM] = [
    EG_PAWN_TABLE,
    EG_KNIGHT_TABLE,
    EG_BISHOP_TABLE,
    EG_ROOK_TABLE,
    EG_QUEEN_TABLE,
    EG_KING_TABLE,
];

/// Evaluate the position from the side-to-move's perspective.
/// Returns score in centipawns.
pub fn evaluate(board: &Board) -> i32 {
    let mut mg_score = [0i32; Color::NUM];
    let mut eg_score = [0i32; Color::NUM];
    let mut phase = 0i32;

    for &color in &[Color::White, Color::Black] {
        let ci = color.index();
        for &pt in &PieceType::ALL {
            let bb = board.pieces(color, pt);
            for sq in bb {
                // Material
                mg_score[ci] += MATERIAL_VALUE[pt.index()];
                eg_score[ci] += MATERIAL_VALUE[pt.index()];

                // PST: tables are from white's perspective, flip for black
                let pst_sq = match color {
                    Color::White => sq.index(),
                    Color::Black => sq.flip_rank().index(),
                };
                mg_score[ci] += MG_PST[pt.index()][pst_sq];
                eg_score[ci] += EG_PST[pt.index()][pst_sq];

                // Phase
                phase += PHASE_WEIGHT[pt.index()];
            }
        }
    }

    // Interpolate between middlegame and endgame scores based on phase
    // phase = 0 means endgame, phase = TOTAL_PHASE means middlegame
    let phase = phase.min(TOTAL_PHASE); // clamp

    let stm = board.side_to_move().index();
    let opponent = (stm ^ 1) & 1;

    let mg = mg_score[stm] - mg_score[opponent];
    let eg = eg_score[stm] - eg_score[opponent];

    (mg * phase + eg * (TOTAL_PHASE - phase)) / TOTAL_PHASE
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn starting_position_roughly_zero() {
        let board = Board::starting_position();
        let score = evaluate(&board);
        // Starting position should be roughly equal (within PST asymmetry + tempo)
        assert!(
            score.abs() < 50,
            "starting position eval {score} too far from 0"
        );
    }

    #[test]
    fn white_up_queen() {
        // White has an extra queen
        let board =
            Board::from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let score = evaluate(&board);
        assert!(score > 800, "should be winning for white, got {score}");
    }

    #[test]
    fn black_up_queen() {
        let board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1").unwrap();
        let score = evaluate(&board);
        assert!(
            score > 800,
            "should be winning for black (STM), got {score}"
        );
    }

    #[test]
    fn symmetry() {
        // Mirror position should give same score (since eval is from STM perspective)
        let white_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1";
        let black_fen = "rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let white_board = Board::from_fen(white_fen).unwrap();
        let black_board = Board::from_fen(black_fen).unwrap();
        let w_score = evaluate(&white_board);
        let b_score = evaluate(&black_board);
        assert_eq!(
            w_score, b_score,
            "mirrored positions should have equal eval"
        );
    }

    #[test]
    fn endgame_king_centralization() {
        // King in center should be better in endgame
        let center_king = Board::from_fen("8/8/8/3K4/8/8/8/4k3 w - - 0 1").unwrap();
        let corner_king = Board::from_fen("K7/8/8/8/8/8/8/4k3 w - - 0 1").unwrap();
        let center_score = evaluate(&center_king);
        let corner_score = evaluate(&corner_king);
        assert!(
            center_score > corner_score,
            "center king ({center_score}) should be better than corner ({corner_score})"
        );
    }

    #[test]
    fn material_values_reasonable() {
        assert_eq!(piece_value(PieceType::Pawn), 100);
        assert_eq!(piece_value(PieceType::Knight), 320);
        assert_eq!(piece_value(PieceType::Bishop), 330);
        assert_eq!(piece_value(PieceType::Rook), 500);
        assert_eq!(piece_value(PieceType::Queen), 900);
        assert_eq!(piece_value(PieceType::King), 0);
    }
}
