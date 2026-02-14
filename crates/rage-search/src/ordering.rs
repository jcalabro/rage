use rage_core::{Board, Color, Move, PieceType, Square};

use crate::eval::piece_value;
use crate::see;

/// Score tiers for move ordering (higher = searched first).
const TT_MOVE_SCORE: i32 = 100_000;
const GOOD_CAPTURE_BASE: i32 = 30_000;
const KILLER1_SCORE: i32 = 20_000;
const KILLER2_SCORE: i32 = 19_000;
const COUNTER_MOVE_SCORE: i32 = 18_000;
const BAD_CAPTURE_BASE: i32 = -10_000;

/// Maximum search ply for history/killer arrays.
pub const MAX_PLY: usize = 128;

/// MVV-LVA score for a capture (range ~0..9000).
fn mvv_lva_score(board: &Board, m: Move) -> i32 {
    let to = m.to_sq();

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

    victim_value * 10 - attacker_value
}

/// Killer move table: 2 killer moves per ply.
pub struct Killers {
    table: [[Move; 2]; MAX_PLY],
}

impl Default for Killers {
    fn default() -> Self {
        Self::new()
    }
}

impl Killers {
    pub fn new() -> Self {
        Self {
            table: [[Move::NULL; 2]; MAX_PLY],
        }
    }

    pub fn get(&self, ply: usize) -> [Move; 2] {
        self.table[ply]
    }

    /// Record a killer move. Shifts killer[0] to killer[1] and stores new move as killer[0].
    pub fn store(&mut self, ply: usize, m: Move) {
        if m == self.table[ply][0] {
            return; // Already the primary killer
        }
        self.table[ply][1] = self.table[ply][0];
        self.table[ply][0] = m;
    }

    pub fn clear(&mut self) {
        self.table = [[Move::NULL; 2]; MAX_PLY];
    }
}

/// History heuristic table: indexed by [color][from_sq][to_sq].
pub struct History {
    table: [[[i32; Square::NUM]; Square::NUM]; Color::NUM],
}

/// Maximum absolute value for history scores (prevents overflow).
const HISTORY_MAX: i32 = 16384;

impl Default for History {
    fn default() -> Self {
        Self::new()
    }
}

impl History {
    pub fn new() -> Self {
        Self {
            table: [[[0; Square::NUM]; Square::NUM]; Color::NUM],
        }
    }

    pub fn get(&self, color: Color, m: Move) -> i32 {
        self.table[color.index()][m.from_sq().index()][m.to_sq().index()]
    }

    /// Update history with gravity: bonus decays existing values toward zero.
    pub fn update(&mut self, color: Color, m: Move, bonus: i32) {
        let entry = &mut self.table[color.index()][m.from_sq().index()][m.to_sq().index()];
        // Gravity formula: prevents unbounded growth
        *entry += bonus - *entry * bonus.abs() / HISTORY_MAX;
        *entry = (*entry).clamp(-HISTORY_MAX, HISTORY_MAX);
    }

    pub fn clear(&mut self) {
        self.table = [[[0; Square::NUM]; Square::NUM]; Color::NUM];
    }
}

/// Continuation history entry: [curr_piece][curr_to_sq] → score.
/// Used as the inner slice of the full continuation history table.
pub type ContHistEntry = [[i32; Square::NUM]; PieceType::NUM];

/// Continuation history: indexed by [prev_piece][prev_to_sq] → ContHistEntry.
/// Tracks how good a move is given the previous move played.
/// Size: 6 × 64 × 6 × 64 = 147,456 entries × 4 bytes ≈ 576 KB (heap-allocated).
pub struct ContinuationHistory {
    table: Box<[ContHistEntry]>,
}

impl Default for ContinuationHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl ContinuationHistory {
    pub fn new() -> Self {
        let size = PieceType::NUM * Square::NUM;
        Self {
            table: vec![[[0i32; Square::NUM]; PieceType::NUM]; size].into_boxed_slice(),
        }
    }

    fn entry_index(prev_piece: PieceType, prev_to: Square) -> usize {
        prev_piece.index() * Square::NUM + prev_to.index()
    }

    pub fn get_entry(&self, prev_piece: PieceType, prev_to: Square) -> &ContHistEntry {
        &self.table[Self::entry_index(prev_piece, prev_to)]
    }

    pub fn update(
        &mut self,
        prev_piece: PieceType,
        prev_to: Square,
        curr_piece: PieceType,
        curr_to: Square,
        bonus: i32,
    ) {
        let entry = &mut self.table[Self::entry_index(prev_piece, prev_to)][curr_piece.index()]
            [curr_to.index()];
        // Gravity formula: same as butterfly history
        *entry += bonus - *entry * bonus.abs() / HISTORY_MAX;
        *entry = (*entry).clamp(-HISTORY_MAX, HISTORY_MAX);
    }

    pub fn clear(&mut self) {
        self.table.fill([[0; Square::NUM]; PieceType::NUM]);
    }
}

/// Counter move table: indexed by [piece_type][to_sq] of the previous move.
pub struct CounterMoves {
    table: [[Move; Square::NUM]; PieceType::NUM],
}

impl Default for CounterMoves {
    fn default() -> Self {
        Self::new()
    }
}

impl CounterMoves {
    pub fn new() -> Self {
        Self {
            table: [[Move::NULL; Square::NUM]; PieceType::NUM],
        }
    }

    pub fn get(&self, prev_piece: PieceType, prev_to: Square) -> Move {
        self.table[prev_piece.index()][prev_to.index()]
    }

    pub fn store(&mut self, prev_piece: PieceType, prev_to: Square, m: Move) {
        self.table[prev_piece.index()][prev_to.index()] = m;
    }

    pub fn clear(&mut self) {
        self.table = [[Move::NULL; Square::NUM]; PieceType::NUM];
    }
}

/// Context for scoring moves at a given ply.
pub struct MoveOrderContext<'a> {
    pub tt_move: Move,
    pub killers: [Move; 2],
    pub counter_move: Move,
    pub history_color: Color,
    pub cont_hist: Option<&'a ContHistEntry>,
}

/// Score a single move for ordering.
pub fn score_move(board: &Board, m: Move, ctx: &MoveOrderContext<'_>, history: &History) -> i32 {
    if m == ctx.tt_move {
        return TT_MOVE_SCORE;
    }

    if board.is_capture(m) || m.is_promotion() {
        if m.is_promotion() && !board.is_capture(m) {
            // Non-capture promotion
            return GOOD_CAPTURE_BASE + piece_value(m.promotion_piece());
        }
        // Use SEE to classify captures as good or bad
        let see_val = see::see(board, m);
        if see_val >= 0 {
            return GOOD_CAPTURE_BASE + mvv_lva_score(board, m);
        }
        return BAD_CAPTURE_BASE + mvv_lva_score(board, m);
    }

    // Quiet moves
    if m == ctx.killers[0] {
        return KILLER1_SCORE;
    }
    if m == ctx.killers[1] {
        return KILLER2_SCORE;
    }
    if m == ctx.counter_move {
        return COUNTER_MOVE_SCORE;
    }

    // History score for remaining quiets: butterfly + continuation
    let mut score = history.get(ctx.history_color, m);
    if let Some(ch) = ctx.cont_hist {
        let piece = board
            .piece_on(m.from_sq())
            .map_or(PieceType::Pawn, |p| p.piece_type);
        score += ch[piece.index()][m.to_sq().index()];
    }
    score
}

/// Pick the best-scored move from `start_idx..moves.len()`, swap it to `start_idx`.
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

/// Score all moves into a pre-allocated slice (avoids heap allocation).
pub fn score_moves_into(
    board: &Board,
    moves: &[Move],
    scores: &mut [i32],
    ctx: &MoveOrderContext<'_>,
    history: &History,
) {
    for (i, m) in moves.iter().enumerate() {
        scores[i] = score_move(board, *m, ctx, history);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use rage_core::Board;

    fn default_ctx(tt_move: Move) -> MoveOrderContext<'static> {
        MoveOrderContext {
            tt_move,
            killers: [Move::NULL; 2],
            counter_move: Move::NULL,
            history_color: Color::White,
            cont_hist: None,
        }
    }

    #[test]
    fn tt_move_highest_priority() {
        let board = Board::starting_position();
        let history = History::new();
        let tt_move = Move::new(rage_core::Square::E2, rage_core::Square::E4);
        let other_move = Move::new(rage_core::Square::D2, rage_core::Square::D4);
        let ctx = default_ctx(tt_move);
        let tt_score = score_move(&board, tt_move, &ctx, &history);
        let other_score = score_move(&board, other_move, &ctx, &history);
        assert!(tt_score > other_score, "TT move should have highest score");
    }

    #[test]
    fn captures_above_quiets() {
        let board =
            Board::from_fen("rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 2")
                .unwrap();
        let history = History::new();
        let capture = Move::new(rage_core::Square::D4, rage_core::Square::E5);
        let quiet = Move::new(rage_core::Square::E2, rage_core::Square::E3);
        let ctx = default_ctx(Move::NULL);
        let cap_score = score_move(&board, capture, &ctx, &history);
        let quiet_score = score_move(&board, quiet, &ctx, &history);
        assert!(
            cap_score > quiet_score,
            "capture ({cap_score}) should score above quiet ({quiet_score})"
        );
    }

    #[test]
    fn killer_above_normal_quiet() {
        let board = Board::starting_position();
        let history = History::new();
        let killer = Move::new(rage_core::Square::E2, rage_core::Square::E4);
        let quiet = Move::new(rage_core::Square::D2, rage_core::Square::D4);
        let ctx = MoveOrderContext {
            tt_move: Move::NULL,
            killers: [killer, Move::NULL],
            counter_move: Move::NULL,
            history_color: Color::White,
            cont_hist: None,
        };
        let killer_score = score_move(&board, killer, &ctx, &history);
        let quiet_score = score_move(&board, quiet, &ctx, &history);
        assert!(
            killer_score > quiet_score,
            "killer ({killer_score}) should score above quiet ({quiet_score})"
        );
    }

    #[test]
    fn history_affects_quiet_ordering() {
        let board = Board::starting_position();
        let mut history = History::new();
        let m1 = Move::new(rage_core::Square::E2, rage_core::Square::E4);
        let m2 = Move::new(rage_core::Square::D2, rage_core::Square::D4);
        history.update(Color::White, m1, 500);
        let ctx = default_ctx(Move::NULL);
        let s1 = score_move(&board, m1, &ctx, &history);
        let s2 = score_move(&board, m2, &ctx, &history);
        assert!(
            s1 > s2,
            "move with history ({s1}) should score above ({s2})"
        );
    }

    #[test]
    fn killers_store_and_shift() {
        let mut killers = Killers::new();
        let m1 = Move::new(rage_core::Square::E2, rage_core::Square::E4);
        let m2 = Move::new(rage_core::Square::D2, rage_core::Square::D4);
        killers.store(0, m1);
        assert_eq!(killers.get(0)[0], m1);
        killers.store(0, m2);
        assert_eq!(killers.get(0)[0], m2);
        assert_eq!(killers.get(0)[1], m1);
    }

    #[test]
    fn pick_move_selects_best() {
        let m1 = Move::new(rage_core::Square::E2, rage_core::Square::E4);
        let m2 = Move::new(rage_core::Square::D2, rage_core::Square::D4);
        let m3 = Move::new(rage_core::Square::G1, rage_core::Square::F3);

        let mut moves = [m1, m2, m3];
        let mut scores = [10, 50, 30];

        pick_move(&mut moves, &mut scores, 0);
        assert_eq!(moves[0], m2);
        assert_eq!(scores[0], 50);
    }

    #[test]
    fn cont_history_update_and_get() {
        let mut ch = ContinuationHistory::new();
        let prev_piece = PieceType::Knight;
        let prev_to = rage_core::Square::F3;
        let curr_piece = PieceType::Pawn;
        let curr_to = rage_core::Square::E4;

        // Initially zero
        let entry = ch.get_entry(prev_piece, prev_to);
        assert_eq!(entry[curr_piece.index()][curr_to.index()], 0);

        // After update, non-zero
        ch.update(prev_piece, prev_to, curr_piece, curr_to, 500);
        let entry = ch.get_entry(prev_piece, prev_to);
        assert!(
            entry[curr_piece.index()][curr_to.index()] > 0,
            "positive bonus should produce positive score"
        );

        // Negative bonus
        ch.update(prev_piece, prev_to, curr_piece, curr_to, -1000);
        let entry = ch.get_entry(prev_piece, prev_to);
        assert!(
            entry[curr_piece.index()][curr_to.index()] < 0,
            "large negative bonus should flip sign"
        );
    }

    #[test]
    fn cont_history_gravity_prevents_overflow() {
        let mut ch = ContinuationHistory::new();
        let prev_piece = PieceType::Rook;
        let prev_to = rage_core::Square::D1;
        let curr_piece = PieceType::Bishop;
        let curr_to = rage_core::Square::C4;

        // Repeatedly apply large bonuses
        for _ in 0..1000 {
            ch.update(prev_piece, prev_to, curr_piece, curr_to, 500);
        }

        let entry = ch.get_entry(prev_piece, prev_to);
        let val = entry[curr_piece.index()][curr_to.index()];
        assert!(
            val <= HISTORY_MAX && val >= -HISTORY_MAX,
            "gravity should keep value in [-{HISTORY_MAX}, {HISTORY_MAX}], got {val}"
        );
    }

    #[test]
    fn cont_history_clear_zeroes() {
        let mut ch = ContinuationHistory::new();

        // Populate several entries
        ch.update(PieceType::Pawn, rage_core::Square::E4, PieceType::Knight, rage_core::Square::F3, 300);
        ch.update(PieceType::Queen, rage_core::Square::D1, PieceType::Rook, rage_core::Square::A1, -200);

        ch.clear();

        // All entries should be zero
        let e1 = ch.get_entry(PieceType::Pawn, rage_core::Square::E4);
        let e2 = ch.get_entry(PieceType::Queen, rage_core::Square::D1);
        assert_eq!(e1[PieceType::Knight.index()][rage_core::Square::F3.index()], 0);
        assert_eq!(e2[PieceType::Rook.index()][rage_core::Square::A1.index()], 0);
    }

    #[test]
    fn cont_history_independent_entries() {
        let mut ch = ContinuationHistory::new();
        let curr_piece = PieceType::Pawn;
        let curr_to = rage_core::Square::E4;

        // Update for one previous move context
        ch.update(PieceType::Knight, rage_core::Square::F3, curr_piece, curr_to, 500);

        // Different previous move context should be unaffected
        let other = ch.get_entry(PieceType::Bishop, rage_core::Square::C4);
        assert_eq!(
            other[curr_piece.index()][curr_to.index()],
            0,
            "different prev move context should be independent"
        );
    }

    #[test]
    fn cont_hist_boosts_quiet_scoring() {
        // In the starting position, e2e4 is a quiet pawn move.
        // With positive cont_hist for (Pawn, e4), it should score higher.
        let board = Board::starting_position();
        let history = History::new();
        let m = Move::new(rage_core::Square::E2, rage_core::Square::E4);

        let score_without = score_move(&board, m, &default_ctx(Move::NULL), &history);

        let mut ch_entry = [[0i32; Square::NUM]; PieceType::NUM];
        ch_entry[PieceType::Pawn.index()][rage_core::Square::E4.index()] = 5000;

        let ctx_with = MoveOrderContext {
            tt_move: Move::NULL,
            killers: [Move::NULL; 2],
            counter_move: Move::NULL,
            history_color: Color::White,
            cont_hist: Some(&ch_entry),
        };
        let score_with = score_move(&board, m, &ctx_with, &history);

        assert!(
            score_with > score_without,
            "cont_hist bonus should increase quiet score: {score_with} vs {score_without}"
        );
        assert_eq!(
            score_with - score_without, 5000,
            "cont_hist should add the exact entry value"
        );
    }

    #[test]
    fn cont_hist_does_not_affect_captures() {
        // Captures should be scored by SEE/MVV-LVA regardless of cont_hist
        let board =
            Board::from_fen("rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 2")
                .unwrap();
        let history = History::new();
        let capture = Move::new(rage_core::Square::D4, rage_core::Square::E5);

        let score_without = score_move(&board, capture, &default_ctx(Move::NULL), &history);

        let mut ch_entry = [[0i32; Square::NUM]; PieceType::NUM];
        ch_entry[PieceType::Pawn.index()][rage_core::Square::E5.index()] = 9999;
        let ctx_with = MoveOrderContext {
            tt_move: Move::NULL,
            killers: [Move::NULL; 2],
            counter_move: Move::NULL,
            history_color: Color::White,
            cont_hist: Some(&ch_entry),
        };
        let score_with = score_move(&board, capture, &ctx_with, &history);

        assert_eq!(
            score_without, score_with,
            "cont_hist should not affect capture scoring"
        );
    }

    #[test]
    fn cont_hist_does_not_affect_killers() {
        // Killer moves should keep their fixed score regardless of cont_hist
        let board = Board::starting_position();
        let history = History::new();
        let killer = Move::new(rage_core::Square::E2, rage_core::Square::E4);

        let mut ch_entry = [[0i32; Square::NUM]; PieceType::NUM];
        ch_entry[PieceType::Pawn.index()][rage_core::Square::E4.index()] = 9999;
        let ctx = MoveOrderContext {
            tt_move: Move::NULL,
            killers: [killer, Move::NULL],
            counter_move: Move::NULL,
            history_color: Color::White,
            cont_hist: Some(&ch_entry),
        };
        let score = score_move(&board, killer, &ctx, &history);

        assert_eq!(score, KILLER1_SCORE, "killer should use fixed score, not cont_hist");
    }

    #[test]
    fn score_moves_into_uses_cont_hist() {
        let board = Board::starting_position();
        let history = History::new();
        let m1 = Move::new(rage_core::Square::E2, rage_core::Square::E4);
        let m2 = Move::new(rage_core::Square::D2, rage_core::Square::D4);
        let moves = [m1, m2];

        // Without cont_hist: both should score 0 (no butterfly history)
        let ctx_none = default_ctx(Move::NULL);
        let mut scores_none = [0i32; 2];
        score_moves_into(&board, &moves, &mut scores_none, &ctx_none, &history);
        assert_eq!(scores_none[0], 0);
        assert_eq!(scores_none[1], 0);

        // With cont_hist boosting m1 (Pawn to e4)
        let mut ch_entry = [[0i32; Square::NUM]; PieceType::NUM];
        ch_entry[PieceType::Pawn.index()][rage_core::Square::E4.index()] = 3000;
        let ctx_with = MoveOrderContext {
            tt_move: Move::NULL,
            killers: [Move::NULL; 2],
            counter_move: Move::NULL,
            history_color: Color::White,
            cont_hist: Some(&ch_entry),
        };
        let mut scores_with = [0i32; 2];
        score_moves_into(&board, &moves, &mut scores_with, &ctx_with, &history);
        assert_eq!(scores_with[0], 3000, "m1 should get cont_hist boost");
        assert_eq!(scores_with[1], 0, "m2 should be unaffected");
    }
}
