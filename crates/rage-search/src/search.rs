use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use rage_core::{Board, Move, PieceType};

use crate::eval::{INFINITY, MATE_SCORE, evaluate, piece_value};
use crate::ordering::{
    CounterMoves, History, Killers, MAX_PLY, MoveOrderContext, pick_move, score_moves,
};
use crate::see;
use crate::time::{SearchLimits, TimeManager};
use crate::tt::{Bound, TranspositionTable};

/// Delta pruning margin in quiescence search.
const DELTA_MARGIN: i32 = 200;

/// Initial aspiration window delta.
const ASPIRATION_DELTA: i32 = 25;

/// LMR reduction table: precomputed as LMR_TABLE[depth][move_count].
/// R = max(1, floor(0.75 + ln(d) * ln(m) / 2.25))
const LMR_TABLE_SIZE: usize = 64;
#[allow(clippy::needless_range_loop)]
static LMR_TABLE: std::sync::LazyLock<[[u32; LMR_TABLE_SIZE]; LMR_TABLE_SIZE]> =
    std::sync::LazyLock::new(|| {
        let mut table = [[0u32; LMR_TABLE_SIZE]; LMR_TABLE_SIZE];
        for depth in 1..LMR_TABLE_SIZE {
            for moves in 1..LMR_TABLE_SIZE {
                let d = depth as f64;
                let m = moves as f64;
                let r = (0.75 + d.ln() * m.ln() / 2.25).floor() as u32;
                table[depth][moves] = r.max(1);
            }
        }
        table
    });

/// LMP thresholds indexed by depth (0..=8).
const LMP_THRESHOLDS: [u32; 9] = [0, 5, 8, 13, 20, 30, 42, 56, 72];

/// Search result returned to the UCI layer.
pub struct SearchResult {
    pub best_move: Move,
    pub score: i32,
    pub depth: u32,
    pub nodes: u64,
    pub pv: Vec<Move>,
}

/// Info callback: called after each completed iteration.
pub type InfoCallback = Box<dyn Fn(&SearchInfo) + Send>;

/// Information about a completed search iteration.
pub struct SearchInfo {
    pub depth: u32,
    pub score: i32,
    pub nodes: u64,
    pub time_ms: u64,
    pub pv: Vec<Move>,
    pub hashfull: u32,
}

impl SearchInfo {
    /// Format score as UCI string (cp or mate).
    pub fn score_string(&self) -> String {
        if self.score.abs() > MATE_SCORE - 256 {
            let mate_distance = if self.score > 0 {
                (MATE_SCORE - self.score + 1) / 2
            } else {
                -(MATE_SCORE + self.score + 1) / 2
            };
            format!("mate {mate_distance}")
        } else {
            format!("cp {}", self.score)
        }
    }

    /// Nodes per second.
    pub fn nps(&self) -> u64 {
        if self.time_ms == 0 {
            0
        } else {
            self.nodes * 1000 / self.time_ms
        }
    }
}

/// Per-ply search stack entry.
struct StackEntry {
    static_eval: i32,
    move_played: Move,
    piece_moved: PieceType,
}

impl StackEntry {
    const EMPTY: Self = Self {
        static_eval: 0,
        move_played: Move::NULL,
        piece_moved: PieceType::Pawn,
    };
}

/// The main search engine.
pub struct Searcher {
    pub tt: TranspositionTable,
    killers: Killers,
    history: History,
    counter_moves: CounterMoves,
    stack: [StackEntry; MAX_PLY + 4], // +4 for safety with ply-2 lookback
    nodes: u64,
    stopped: bool,
}

impl Searcher {
    pub fn new(tt_size_mb: usize) -> Self {
        Self {
            tt: TranspositionTable::new(tt_size_mb),
            killers: Killers::new(),
            history: History::new(),
            counter_moves: CounterMoves::new(),
            stack: [const { StackEntry::EMPTY }; MAX_PLY + 4],
            nodes: 0,
            stopped: false,
        }
    }

    /// Run an iterative deepening search with aspiration windows.
    pub fn search(
        &mut self,
        board: &mut Board,
        limits: &SearchLimits,
        stop: Arc<AtomicBool>,
        info_callback: Option<InfoCallback>,
    ) -> SearchResult {
        self.tt.new_search();
        self.nodes = 0;
        self.stopped = false;
        self.killers.clear();

        let mut tm = TimeManager::new(limits, board.side_to_move(), stop);

        let mut best_move = Move::NULL;
        let mut best_score = -INFINITY;
        let mut best_pv = Vec::new();
        let mut prev_best_move = Move::NULL;
        let mut stability_count = 0u32;

        for depth in 1..=tm.max_depth().min(MAX_PLY as u32) {
            let score;

            if depth < 4 {
                // Small depths: full window
                score = self.negamax(board, &mut tm, -INFINITY, INFINITY, depth, 0, false);
            } else {
                // Aspiration windows
                let mut delta = ASPIRATION_DELTA;
                let mut alpha = best_score - delta;
                let mut beta = best_score + delta;
                let mut fail_count = 0;
                let mut aspiration_score;

                loop {
                    aspiration_score = self.negamax(board, &mut tm, alpha, beta, depth, 0, false);
                    if self.stopped {
                        break;
                    }

                    if aspiration_score <= alpha {
                        // Fail low: widen alpha
                        alpha = if fail_count >= 2 {
                            -INFINITY
                        } else {
                            (aspiration_score - delta).max(-INFINITY)
                        };
                        fail_count += 1;
                    } else if aspiration_score >= beta {
                        // Fail high: widen beta
                        beta = if fail_count >= 2 {
                            INFINITY
                        } else {
                            (aspiration_score + delta).min(INFINITY)
                        };
                        fail_count += 1;
                    } else {
                        // In window — done
                        break;
                    }

                    delta *= 2;

                    if self.stopped {
                        break;
                    }
                }

                if self.stopped {
                    break;
                }

                score = aspiration_score;
            }

            if self.stopped {
                break;
            }

            best_score = score;

            // Extract PV from TT
            let pv = self.extract_pv(board, depth);
            if let Some(&first) = pv.first() {
                best_move = first;
            }
            best_pv = pv;

            // Track best-move stability for time management
            if best_move == prev_best_move {
                stability_count += 1;
            } else {
                stability_count = 0;
                prev_best_move = best_move;
            }

            if let Some(ref cb) = info_callback {
                cb(&SearchInfo {
                    depth,
                    score: best_score,
                    nodes: self.nodes,
                    time_ms: tm.elapsed_ms(),
                    pv: best_pv.clone(),
                    hashfull: self.tt.hashfull(),
                });
            }

            // Don't start a new iteration if soft limit reached
            // Apply best-move stability: stable moves use less time
            if stability_count >= 5 {
                if tm.soft_limit_reached_fraction(50) {
                    break;
                }
            } else if tm.soft_limit_reached() {
                break;
            }

            // If we found a forced mate, no need to search deeper
            if best_score.abs() > MATE_SCORE - 256 {
                break;
            }
        }

        // If we got stopped before completing any iteration, fall back to
        // the first legal move so we never return a null move.
        if best_move.is_null() {
            let moves = board.generate_legal_moves();
            if !moves.is_empty() {
                best_move = moves[0];
                best_pv = vec![best_move];
            }
        }

        SearchResult {
            best_move,
            score: best_score,
            depth: best_pv.len() as u32,
            nodes: self.nodes,
            pv: best_pv,
        }
    }

    /// Principal Variation Search with alpha-beta pruning.
    #[allow(clippy::too_many_arguments)]
    fn negamax(
        &mut self,
        board: &mut Board,
        tm: &mut TimeManager,
        mut alpha: i32,
        beta: i32,
        mut depth: u32,
        ply: u32,
        null_allowed: bool,
    ) -> i32 {
        self.nodes += 1;

        // Check for time/stop periodically
        if tm.should_stop(self.nodes) {
            self.stopped = true;
            return 0;
        }

        let is_root = ply == 0;
        let pv_node = beta - alpha > 1;
        let in_check = board.in_check();

        // Draw detection
        if !is_root && is_draw(board) {
            return 0;
        }

        // Check extension: extend when in check to avoid horizon effect
        if in_check {
            depth += 1;
        }

        // Quiescence at leaf
        if depth == 0 {
            return self.quiescence(board, tm, alpha, beta, ply);
        }

        // Mate distance pruning
        if !is_root {
            let mating_score = MATE_SCORE - ply as i32;
            let mated_score = -(MATE_SCORE - (ply as i32 + 1));
            let new_alpha = alpha.max(mated_score);
            let new_beta = beta.min(mating_score);
            if new_alpha >= new_beta {
                return new_alpha;
            }
            alpha = new_alpha;
        }

        // TT probe
        let hash = board.zobrist_hash();
        let tt_move = if let Some(entry) = self.tt.probe(hash, ply) {
            // TT cutoff (only at non-PV nodes)
            if !pv_node && entry.depth >= depth as u8 {
                match entry.bound {
                    Bound::Exact => return entry.score,
                    Bound::Lower => {
                        if entry.score >= beta {
                            return entry.score;
                        }
                    }
                    Bound::Upper => {
                        if entry.score <= alpha {
                            return entry.score;
                        }
                    }
                    Bound::None => {}
                }
            }
            entry.best_move
        } else {
            Move::NULL
        };

        // Static evaluation (not useful when in check)
        let static_eval = if in_check { -INFINITY } else { evaluate(board) };
        self.stack[ply as usize + 2].static_eval = static_eval;

        // Improving: is our static eval better than 2 plies ago?
        let improving = !in_check && ply >= 2 && static_eval > self.stack[ply as usize].static_eval;

        // --- Pre-move-generation pruning (non-PV, non-check) ---

        if !pv_node && !in_check {
            // Reverse futility pruning (static null move pruning)
            if depth <= 6 && static_eval - rfp_margin(depth, improving) >= beta {
                return static_eval;
            }

            // Razoring: if eval is way below alpha, drop to qsearch
            if depth <= 2 && static_eval + 300 <= alpha {
                let q_score = self.quiescence(board, tm, alpha, beta, ply);
                if q_score <= alpha {
                    return q_score;
                }
            }

            // Null move pruning
            if null_allowed
                && depth >= 3
                && static_eval >= beta
                && board.has_non_pawn_material(board.side_to_move())
            {
                let r = 3 + depth / 6;
                board.make_null_move();
                let null_score = -self.negamax(
                    board,
                    tm,
                    -beta,
                    -beta + 1,
                    depth.saturating_sub(r),
                    ply + 1,
                    false, // no consecutive null moves
                );
                board.unmake_null_move();

                if self.stopped {
                    return 0;
                }

                if null_score >= beta {
                    // Don't return unproven mate scores
                    if null_score > MATE_SCORE - 256 {
                        return beta;
                    }
                    return null_score;
                }
            }
        }

        // Generate legal moves
        let moves = board.generate_legal_moves();
        if moves.is_empty() {
            return if in_check {
                -(MATE_SCORE - ply as i32) // Checkmate
            } else {
                0 // Stalemate
            };
        }

        // Get previous move info for counter moves
        let prev_move = if ply >= 1 {
            self.stack[ply as usize + 1].move_played
        } else {
            Move::NULL
        };
        let prev_piece = if ply >= 1 {
            self.stack[ply as usize + 1].piece_moved
        } else {
            PieceType::Pawn
        };
        let counter_move = if !prev_move.is_null() {
            self.counter_moves.get(prev_piece, prev_move.to_sq())
        } else {
            Move::NULL
        };

        // Score and sort moves
        let ctx = MoveOrderContext {
            tt_move,
            killers: self.killers.get(ply as usize),
            counter_move,
            history_color: board.side_to_move(),
        };
        let mut move_buf: Vec<Move> = moves.as_slice().to_vec();
        let mut scores = score_moves(board, &moves, &ctx, &self.history);

        let mut best_move = move_buf[0];
        let mut best_score = -INFINITY;
        let mut bound = Bound::Upper;
        let mut move_count = 0u32;
        let stm = board.side_to_move();

        for i in 0..move_buf.len() {
            pick_move(&mut move_buf, &mut scores, i);
            let m = move_buf[i];
            let is_capture = board.is_capture(m);
            let is_quiet = !is_capture && !m.is_promotion();
            move_count += 1;

            // --- Move-level pruning (non-root, non-PV, non-check) ---
            if !is_root && !pv_node && !in_check && best_score > -MATE_SCORE + 256 {
                // Late move pruning
                if is_quiet && depth <= 8 && move_count > lmp_threshold(depth, improving) {
                    continue;
                }

                // Futility pruning for quiet moves
                if is_quiet && depth <= 6 && static_eval + futility_margin(depth) <= alpha {
                    continue;
                }

                // SEE pruning for bad captures
                if is_capture && depth <= 4 && !see::see_ge(board, m, 0) {
                    continue;
                }

                // SEE pruning for quiet moves at low depth
                if is_quiet && depth <= 6 && !see::see_ge(board, m, -20 * depth as i32) {
                    continue;
                }
            }

            // Record move info in search stack so child nodes can read it
            let piece_moved = board
                .piece_on(m.from_sq())
                .map_or(PieceType::Pawn, |p| p.piece_type);
            self.stack[ply as usize + 2].move_played = m;
            self.stack[ply as usize + 2].piece_moved = piece_moved;

            board.make_move(m);

            // PVS: first move gets full window, rest get zero window
            let score;
            if move_count == 1 {
                // First move: full window
                score = -self.negamax(board, tm, -beta, -alpha, depth - 1, ply + 1, true);
            } else {
                // Late move reductions
                let mut reduction = 0u32;
                if depth >= 3 && move_count > 1 && is_quiet {
                    let d = (depth as usize).min(LMR_TABLE_SIZE - 1);
                    let mc = (move_count as usize).min(LMR_TABLE_SIZE - 1);
                    reduction = LMR_TABLE[d][mc];

                    // Adjust reductions
                    if pv_node {
                        reduction = reduction.saturating_sub(1);
                    }
                    if !improving {
                        reduction += 1;
                    }

                    // Reduce less for killers/counter moves
                    if m == ctx.killers[0] || m == ctx.killers[1] || m == counter_move {
                        reduction = reduction.saturating_sub(1);
                    }

                    // Reduce based on history
                    let hist = self.history.get(stm, m);
                    if hist > 0 {
                        reduction = reduction.saturating_sub(1);
                    } else if hist < -2000 {
                        reduction += 1;
                    }

                    // Don't reduce below 1
                    reduction = reduction.min(depth - 2);
                }

                // Zero-window search (possibly with reduction)
                let mut s = -self.negamax(
                    board,
                    tm,
                    -alpha - 1,
                    -alpha,
                    depth - 1 - reduction,
                    ply + 1,
                    true,
                );

                // Re-search at full depth if reduced search failed high
                if s > alpha && reduction > 0 {
                    s = -self.negamax(board, tm, -alpha - 1, -alpha, depth - 1, ply + 1, true);
                }

                // Re-search with full window if zero-window failed high
                if s > alpha && s < beta {
                    s = -self.negamax(board, tm, -beta, -alpha, depth - 1, ply + 1, true);
                }

                score = s;
            }

            board.unmake_move(m);

            if self.stopped {
                return 0;
            }

            if score > best_score {
                best_score = score;
                best_move = m;

                if score > alpha {
                    alpha = score;
                    bound = Bound::Exact;

                    if score >= beta {
                        bound = Bound::Lower;

                        // Update killers, history, and counter moves on quiet beta cutoff
                        if is_quiet {
                            self.killers.store(ply as usize, m);
                            let bonus = (depth * depth) as i32;
                            self.history.update(stm, m, bonus);

                            // Give negative history to all previously tried quiet moves
                            for &prev in &move_buf[..i] {
                                if !board.is_capture(prev) && !prev.is_promotion() {
                                    self.history.update(stm, prev, -bonus);
                                }
                            }

                            // Counter move
                            if !prev_move.is_null() {
                                self.counter_moves.store(prev_piece, prev_move.to_sq(), m);
                            }
                        }

                        break;
                    }
                }
            }
        }

        // Store in TT
        self.tt
            .store(hash, best_move, best_score, depth as u8, bound, ply);

        best_score
    }

    /// Quiescence search: only search captures to avoid horizon effects.
    #[allow(clippy::only_used_in_recursion)]
    fn quiescence(
        &mut self,
        board: &mut Board,
        tm: &mut TimeManager,
        mut alpha: i32,
        beta: i32,
        ply: u32,
    ) -> i32 {
        self.nodes += 1;

        if tm.should_stop(self.nodes) {
            self.stopped = true;
            return 0;
        }

        // Standing pat
        let stand_pat = evaluate(board);
        if stand_pat >= beta {
            return stand_pat;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        // Generate all legal moves, then filter to captures
        let moves = board.generate_legal_moves();

        // Score captures with MVV-LVA
        let mut captures: Vec<(Move, i32)> = Vec::new();
        for m in &moves {
            if board.is_capture(*m) || m.is_promotion() {
                // Delta pruning: skip if capture can't possibly raise alpha
                let max_gain = if m.is_en_passant() {
                    piece_value(PieceType::Pawn)
                } else if m.is_promotion() {
                    piece_value(m.promotion_piece())
                        + match board.piece_on(m.to_sq()) {
                            Some(p) => piece_value(p.piece_type),
                            None => 0,
                        }
                } else {
                    match board.piece_on(m.to_sq()) {
                        Some(p) => piece_value(p.piece_type),
                        None => continue,
                    }
                };

                if stand_pat + max_gain + DELTA_MARGIN < alpha {
                    continue; // Delta pruning
                }

                // SEE pruning in qsearch: skip SEE-negative captures
                if !see::see_ge(board, *m, 0) {
                    continue;
                }

                captures.push((*m, mvv_lva_qsearch(board, *m)));
            }
        }

        // Sort captures by score (descending)
        captures.sort_by(|a, b| b.1.cmp(&a.1));

        for (m, _score) in &captures {
            board.make_move(*m);
            let score = -self.quiescence(board, tm, -beta, -alpha, ply + 1);
            board.unmake_move(*m);

            if self.stopped {
                return 0;
            }

            if score >= beta {
                return score;
            }
            if score > alpha {
                alpha = score;
            }
        }

        alpha
    }

    /// Extract the principal variation from the TT.
    fn extract_pv(&self, board: &mut Board, max_depth: u32) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut seen_hashes = Vec::new();

        for _ in 0..max_depth {
            let hash = board.zobrist_hash();
            if seen_hashes.contains(&hash) {
                break;
            }
            seen_hashes.push(hash);

            if let Some(entry) = self.tt.probe(hash, 0) {
                let m = entry.best_move;
                if m.is_null() {
                    break;
                }
                let legal_moves = board.generate_legal_moves();
                if !legal_moves.contains(m) {
                    break;
                }
                pv.push(m);
                board.make_move(m);
            } else {
                break;
            }
        }

        for m in pv.iter().rev() {
            board.unmake_move(*m);
        }

        pv
    }
}

/// Simple MVV-LVA score for qsearch ordering (no SEE).
fn mvv_lva_qsearch(board: &Board, m: Move) -> i32 {
    let victim = if m.is_en_passant() {
        piece_value(PieceType::Pawn)
    } else {
        board
            .piece_on(m.to_sq())
            .map_or(0, |p| piece_value(p.piece_type))
    };

    let attacker = board
        .piece_on(m.from_sq())
        .map_or(0, |p| piece_value(p.piece_type));

    victim * 10 - attacker
}

/// Reverse futility pruning margin.
fn rfp_margin(depth: u32, improving: bool) -> i32 {
    let base = depth as i32 * 80;
    if improving { base } else { base + 80 }
}

/// Futility pruning margin.
fn futility_margin(depth: u32) -> i32 {
    depth as i32 * 100
}

/// LMP threshold: how many quiet moves to try at a given depth.
fn lmp_threshold(depth: u32, improving: bool) -> u32 {
    let idx = (depth as usize).min(LMP_THRESHOLDS.len() - 1);
    let base = LMP_THRESHOLDS[idx];
    if improving { base * 2 } else { base }
}

/// Check if the current position is a draw.
fn is_draw(board: &Board) -> bool {
    board.is_fifty_move_draw() || board.is_insufficient_material() || board.is_repetition()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::*;

    fn search_position(fen: &str, depth: u32) -> SearchResult {
        let mut board = Board::from_fen(fen).unwrap();
        let mut searcher = Searcher::new(16);
        let limits = SearchLimits {
            depth: Some(depth),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        searcher.search(&mut board, &limits, stop, None)
    }

    #[test]
    fn mate_in_one_white() {
        let result = search_position(
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            4,
        );
        assert_eq!(result.best_move.to_string(), "h5f7");
        assert!(result.score > MATE_SCORE - 10, "should find mate");
    }

    #[test]
    fn mate_in_one_back_rank() {
        let result = search_position("6k1/5ppp/8/8/8/8/8/3RK3 w - - 0 1", 4);
        assert_eq!(result.best_move.to_string(), "d1d8");
        assert!(result.score > MATE_SCORE - 10, "should find back rank mate");
    }

    #[test]
    fn white_up_material() {
        let result = search_position("rnbqkbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQq - 0 1", 3);
        assert!(
            result.score > 400,
            "white should be clearly winning (up a rook), got {}",
            result.score
        );
    }

    #[test]
    fn starting_position_reasonable() {
        let result = search_position(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            4,
        );
        assert!(
            result.score.abs() < 100,
            "starting position should be roughly equal, got {}",
            result.score
        );
        assert!(!result.best_move.is_null(), "must find a move");
    }

    #[test]
    fn stalemate_detected() {
        let result = search_position("k7/8/KQ6/8/8/8/8/8 w - - 0 1", 4);
        assert!(!result.best_move.is_null());
    }

    #[test]
    fn nodes_counted() {
        let result = search_position(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            3,
        );
        assert!(result.nodes > 0, "should have searched at least some nodes");
    }

    #[test]
    fn depth_1_finds_move() {
        let result = search_position(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            1,
        );
        assert!(!result.best_move.is_null());
    }

    #[test]
    fn kq_vs_k_winning() {
        let result = search_position("4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1", 4);
        assert!(
            result.score > 800,
            "K+Q vs K should be heavily winning, got {}",
            result.score
        );
    }

    #[test]
    fn checkmate_position_returns_null() {
        let result = search_position(
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
            3,
        );
        assert!(result.best_move.is_null(), "checkmate: no move to make");
    }

    #[test]
    fn stalemate_position_returns_null() {
        let result = search_position("k7/8/KQ6/8/8/8/8/8 b - - 0 1", 3);
        assert!(result.best_move.is_null(), "stalemate: no move to make");
    }

    #[test]
    fn early_stop_returns_valid_move() {
        let mut board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let mut searcher = Searcher::new(16);
        let limits = SearchLimits {
            depth: Some(10),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(true));
        let result = searcher.search(&mut board, &limits, stop, None);
        assert!(
            !result.best_move.is_null(),
            "early stop should still return a valid move"
        );
    }

    #[test]
    fn ruy_lopez_reasonable() {
        let result = search_position(
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
            5,
        );
        assert!(!result.best_move.is_null(), "must find a move");
        assert!(
            result.score.abs() < 150,
            "Ruy Lopez should be roughly equal, got {}",
            result.score
        );
    }

    #[test]
    fn mate_in_two() {
        let result = search_position("k7/rp6/p7/8/8/8/5Q2/3R1K2 w - - 0 1", 5);
        assert!(
            result.score > MATE_SCORE - 20,
            "should find forced mate, got {}",
            result.score
        );
    }

    #[test]
    fn captures_hanging_piece() {
        let result = search_position(
            "rnb1kbnr/pppppppp/8/3q4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
            3,
        );
        assert_eq!(
            result.best_move.to_string(),
            "e4d5",
            "should capture hanging queen"
        );
    }

    #[test]
    fn info_callback_called() {
        let mut board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let mut searcher = Searcher::new(16);
        let limits = SearchLimits {
            depth: Some(3),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let callback_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let counter = Arc::clone(&callback_count);
        let cb: InfoCallback = Box::new(move |_info| {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        searcher.search(&mut board, &limits, stop, Some(cb));
        assert!(
            callback_count.load(Ordering::Relaxed) >= 3,
            "info callback should fire for each completed depth"
        );
    }

    #[test]
    fn pv_extraction_works() {
        let result = search_position(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            3,
        );
        assert!(!result.pv.is_empty(), "PV should not be empty at depth 3");
        assert!(
            result.pv.len() >= 2,
            "PV should have at least 2 moves at depth 3, got {}",
            result.pv.len()
        );
    }

    #[test]
    fn search_respects_depth_limit() {
        let result1 = search_position(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            1,
        );
        let result5 = search_position(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            5,
        );
        assert!(
            result5.nodes > result1.nodes * 5,
            "depth 5 ({}) should search much more than depth 1 ({})",
            result5.nodes,
            result1.nodes
        );
    }

    #[test]
    fn check_extension_finds_deeper_mate() {
        // Position where check extension helps find the mate
        // Mate requires seeing through a check sequence
        let result = search_position(
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            2,
        );
        assert_eq!(result.best_move.to_string(), "h5f7");
    }

    #[test]
    fn pruning_reduces_nodes() {
        // With all the pruning techniques, depth 6 should be much faster
        let result = search_position(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            6,
        );
        assert!(!result.best_move.is_null());
        // Just verify it completes in a reasonable number of nodes
        assert!(
            result.nodes < 10_000_000,
            "depth 6 with pruning should use < 10M nodes, got {}",
            result.nodes
        );
    }
}
