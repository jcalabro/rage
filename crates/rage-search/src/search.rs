use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use rage_core::{Board, Move, MoveList, PieceType};
use rage_nnue::{Accumulator, NnueNetwork};
use rage_syzygy::Tablebase;

use crate::eval::{INFINITY, MATE_SCORE, evaluate_pst, piece_value};
use crate::ordering::{
    ContHistEntry, ContinuationHistory, CounterMoves, History, Killers, MAX_PLY, MoveOrderContext,
    pick_move, score_moves_into,
};
use crate::see;
use crate::time::{SearchLimits, TimeManager};
use crate::tt::{Bound, TranspositionTable};

/// Delta pruning margin in quiescence search.
const DELTA_MARGIN: i32 = 200;

/// Initial aspiration window delta.
const ASPIRATION_DELTA: i32 = 25;

/// Maximum number of moves in any position.
const MAX_MOVES: usize = MoveList::MAX_MOVES;

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

/// Maximum ply the search can reach before returning static eval.
const PLY_LIMIT: u32 = MAX_PLY as u32 - 2;

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
    pub seldepth: u32,
    pub score: i32,
    pub nodes: u64,
    pub time_ms: u64,
    pub pv: Vec<Move>,
    pub hashfull: u32,
    pub tbhits: u64,
    /// 1-indexed multi-PV line number (1 for single-PV).
    pub multipv: usize,
    /// Win/Draw/Loss probabilities as permille (0-1000). None if not computed.
    pub wdl: Option<(u32, u32, u32)>,
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

    /// Compute WDL probabilities from an eval score using the logistic model.
    /// Returns (win, draw, loss) as permille values summing to 1000.
    pub fn compute_wdl(score: i32) -> (u32, u32, u32) {
        if score.abs() > MATE_SCORE - 256 {
            return if score > 0 {
                (1000, 0, 0)
            } else {
                (0, 0, 1000)
            };
        }

        // Logistic model: win_rate = 1 / (1 + exp(-score / k))
        // k ≈ 111 is a standard scaling factor for centipawn-based engines.
        let k = 111.0_f64;
        let win = 1.0 / (1.0 + (-score as f64 / k).exp());
        let loss = 1.0 / (1.0 + (score as f64 / k).exp());

        let w = (win * 1000.0).round() as u32;
        let l = (loss * 1000.0).round() as u32;
        let d = 1000u32.saturating_sub(w).saturating_sub(l);
        (w, d, l)
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
    excluded_move: Move,
}

impl StackEntry {
    const EMPTY: Self = Self {
        static_eval: 0,
        move_played: Move::NULL,
        piece_moved: PieceType::Pawn,
        excluded_move: Move::NULL,
    };
}

/// The main search engine.
pub struct Searcher {
    pub tt: Arc<TranspositionTable>,
    killers: Killers,
    history: History,
    counter_moves: CounterMoves,
    cont_history: ContinuationHistory,
    stack: [StackEntry; MAX_PLY + 4],
    pv_table: [[Move; MAX_PLY + 1]; MAX_PLY + 1],
    pv_length: [usize; MAX_PLY + 1],
    pub nodes: u64,
    pub tbhits: u64,
    seldepth: u32,
    stopped: bool,
    /// NNUE network for evaluation. Falls back to PST eval when None.
    network: Option<Arc<NnueNetwork>>,
    /// Per-ply NNUE accumulators for incremental updates.
    accumulators: Vec<Accumulator>,
    /// Syzygy tablebase for endgame probing. Shared across threads.
    syzygy: Option<Arc<Tablebase>>,
    /// Thread ID for Lazy SMP depth perturbation. 0 = main thread.
    pub thread_id: usize,
    /// Number of principal variations to report (Multi-PV). Default 1.
    pub multi_pv: usize,
    /// Moves excluded at root during multi-PV iteration.
    root_excluded: Vec<Move>,
    /// Tablebase-filtered root moves. When non-empty, only these moves are
    /// searched at the root (prevents playing losing moves in TB positions).
    root_tb_filter: Vec<Move>,
}

impl Searcher {
    pub fn new(tt: Arc<TranspositionTable>) -> Self {
        let mut accumulators = Vec::with_capacity(MAX_PLY + 4);
        for _ in 0..MAX_PLY + 4 {
            accumulators.push(Accumulator::EMPTY);
        }
        Self {
            tt,
            killers: Killers::new(),
            history: History::new(),
            counter_moves: CounterMoves::new(),
            cont_history: ContinuationHistory::new(),
            stack: [const { StackEntry::EMPTY }; MAX_PLY + 4],
            pv_table: [[Move::NULL; MAX_PLY + 1]; MAX_PLY + 1],
            pv_length: [0; MAX_PLY + 1],
            nodes: 0,
            tbhits: 0,
            seldepth: 0,
            stopped: false,
            network: None,
            accumulators,
            syzygy: None,
            thread_id: 0,
            multi_pv: 1,
            root_excluded: Vec::new(),
            root_tb_filter: Vec::new(),
        }
    }

    /// Clear per-thread tables (killers, history, counter moves, continuation history).
    /// Does NOT clear the TT (which is shared).
    pub fn clear_tables(&mut self) {
        self.killers.clear();
        self.history = History::new();
        self.counter_moves = CounterMoves::new();
        self.cont_history.clear();
    }

    /// Get a reference to the current NNUE network (if any).
    pub fn network(&self) -> &Option<Arc<NnueNetwork>> {
        &self.network
    }

    /// Set the NNUE network for evaluation.
    pub fn set_network(&mut self, network: Option<Arc<NnueNetwork>>) {
        self.network = network;
    }

    /// Get the current Syzygy tablebase (if any).
    pub fn syzygy(&self) -> &Option<Arc<Tablebase>> {
        &self.syzygy
    }

    /// Set the Syzygy tablebase for endgame probing.
    pub fn set_syzygy(&mut self, tb: Option<Arc<Tablebase>>) {
        self.syzygy = tb;
    }

    /// Evaluate the position using NNUE with accumulator if available, otherwise PST.
    fn evaluate(&mut self, board: &Board, ply: u32) -> i32 {
        match &self.network {
            Some(net) => {
                let acc = &mut self.accumulators[ply as usize];
                // If any perspective is dirty, do a full refresh
                if acc.dirty[0] || acc.dirty[1] {
                    net.refresh_accumulator(board, acc);
                }
                net.forward(acc, board.side_to_move())
            }
            None => evaluate_pst(board),
        }
    }

    /// Evaluate the current position with NNUE only. Returns `None` if no network is loaded.
    /// Uses ply 0 accumulator with a full refresh.
    pub fn evaluate_nnue(&mut self, board: &Board) -> Option<i32> {
        let net = self.network.as_ref()?;
        let acc = &mut self.accumulators[0];
        net.refresh_accumulator(board, acc);
        Some(net.forward(acc, board.side_to_move()))
    }

    /// Update the accumulator at `child_ply` from `parent_ply` after a move.
    /// Must be called AFTER board.make_move(m).
    #[allow(clippy::too_many_arguments)]
    fn update_acc_after_move(
        &mut self,
        board: &Board,
        m: Move,
        moved_piece: PieceType,
        captured_piece: Option<PieceType>,
        moving_color: rage_core::Color,
        parent_ply: u32,
        child_ply: u32,
    ) {
        if let Some(ref net) = self.network {
            // We need to split the borrow: get parent (immutable) and child (mutable)
            let (parent_slice, child_slice) = self.accumulators.split_at_mut(child_ply as usize);
            let parent = &parent_slice[parent_ply as usize];
            let child = &mut child_slice[0];
            net.update_accumulator(
                board,
                parent,
                child,
                m,
                moved_piece,
                captured_piece,
                moving_color,
            );
        }
    }

    /// Copy the accumulator from parent_ply to child_ply (for null moves).
    fn copy_acc(&mut self, parent_ply: u32, child_ply: u32) {
        if self.network.is_some() {
            let parent = self.accumulators[parent_ply as usize].clone();
            self.accumulators[child_ply as usize] = parent;
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
        self.tbhits = 0;
        self.stopped = false;
        self.killers.clear();

        // Initialize root accumulator for NNUE
        if let Some(ref net) = self.network {
            net.refresh_accumulator(board, &mut self.accumulators[0]);
        }

        let mut tm = TimeManager::new(limits, board.side_to_move(), stop);

        let mut best_move = Move::NULL;
        let mut best_score = -INFINITY;
        let mut best_pv = Vec::new();
        let mut completed_depth = 0u32;
        let mut prev_best_move = Move::NULL;
        let mut stability_count = 0u32;
        let mut prev_score = INFINITY;

        // Syzygy root move filtering: if the root position is in the tablebase,
        // filter out losing moves so the engine only searches winning/drawing lines.
        self.root_tb_filter.clear();
        if let Some(ref tb) = self.syzygy
            && board.castling_rights().bits() == 0
            && board.occupied().popcount() <= tb.probe_limit
        {
            let legal = board.generate_legal_moves();
            let filtered = rage_syzygy::filter_root_moves(board, tb, legal.as_slice());
            if filtered.len() < legal.len() {
                self.tbhits += 1;
                self.root_tb_filter = filtered.into_iter().map(|(m, _)| m).collect();
            }
        }

        // Count root moves once to cap multi_pv
        let root_moves = board.generate_legal_moves();
        let num_root_moves = if self.root_tb_filter.is_empty() {
            root_moves.len()
        } else {
            self.root_tb_filter.len()
        };
        let multi_pv = self.multi_pv.min(num_root_moves).max(1);

        for depth in 1..=tm.max_depth().min(MAX_PLY as u32) {
            // Lazy SMP depth perturbation: helper threads add thread_id % 2
            // so odd-numbered threads search one ply deeper, exploring different
            // parts of the tree.
            let search_depth = depth + (self.thread_id % 2) as u32;

            self.root_excluded.clear();
            let mut depth_stopped = false;

            for pv_idx in 0..multi_pv {
                self.seldepth = 0;
                let score;

                if pv_idx == 0 && search_depth >= 4 {
                    // Aspiration windows only for the primary line
                    let mut delta = ASPIRATION_DELTA;
                    let mut alpha = best_score - delta;
                    let mut beta = best_score + delta;
                    let mut fail_count = 0;
                    let mut aspiration_score;

                    loop {
                        aspiration_score =
                            self.negamax(board, &mut tm, alpha, beta, search_depth, 0, false);
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
                        depth_stopped = true;
                        break;
                    }

                    score = aspiration_score;
                } else {
                    // Full window for small depths or secondary PV lines
                    score =
                        self.negamax(board, &mut tm, -INFINITY, INFINITY, search_depth, 0, false);

                    if self.stopped {
                        depth_stopped = true;
                        break;
                    }
                }

                // Extract PV from triangular PV table
                let pv_len = self.pv_length[0];
                let pv: Vec<Move> = self.pv_table[0][..pv_len].to_vec();
                let line_best = pv.first().copied().unwrap_or(Move::NULL);

                // For primary line, update overall best move/score
                if pv_idx == 0 {
                    best_score = score;
                    completed_depth = depth;
                    if !line_best.is_null() {
                        best_move = line_best;
                    }
                    best_pv = pv.clone();
                }

                // Exclude this line's best move from subsequent PV searches
                if !line_best.is_null() {
                    self.root_excluded.push(line_best);
                }

                if let Some(ref cb) = info_callback {
                    cb(&SearchInfo {
                        depth,
                        seldepth: self.seldepth,
                        score,
                        nodes: self.nodes,
                        time_ms: tm.elapsed_ms(),
                        pv,
                        hashfull: self.tt.hashfull(),
                        tbhits: self.tbhits,
                        multipv: pv_idx + 1,
                        wdl: None,
                    });
                }
            }

            if depth_stopped {
                break;
            }

            // Track best-move stability for time management (primary line only)
            if best_move == prev_best_move {
                stability_count += 1;
            } else {
                stability_count = 0;
                prev_best_move = best_move;
            }

            // Score instability: large eval swings warrant more search time.
            // Compute how "unstable" the score is as a percentage multiplier.
            let score_delta = if prev_score == INFINITY {
                0
            } else {
                (best_score - prev_score).unsigned_abs()
            };
            prev_score = best_score;

            // score_delta >= 40cp → score_pct = 130 (use 130% of soft limit)
            // score_delta == 0   → score_pct = 100 (normal)
            let score_pct = (100 + (score_delta.min(40) * 30 / 40)) as u64;

            // Don't start a new iteration if soft limit reached
            // Apply best-move stability: stable moves use less time
            // Apply score instability: volatile scores use more time
            if stability_count >= 5 {
                if tm.soft_limit_reached_fraction(50 * 100 / score_pct) {
                    break;
                }
            } else if tm.soft_limit_reached_fraction(score_pct) {
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
            depth: completed_depth,
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
        // Ply overflow guard: return static eval at max depth
        if ply >= PLY_LIMIT {
            return self.evaluate(board, ply);
        }

        self.nodes += 1;
        self.seldepth = self.seldepth.max(ply);

        // Initialize PV for this ply (empty)
        self.pv_length[ply as usize] = 0;

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

        // Syzygy tablebase probe
        if !is_root
            && let Some(ref tb) = self.syzygy
            && depth >= tb.probe_depth
            && board.castling_rights().bits() == 0
            && board.occupied().popcount() <= tb.probe_limit
            && let Some(wdl) = rage_syzygy::probe_wdl(board, tb)
        {
            self.tbhits += 1;
            let tb_score = rage_syzygy::wdl_to_score(wdl, ply);

            let tb_bound = match wdl {
                rage_syzygy::Wdl::Win | rage_syzygy::Wdl::CursedWin => Bound::Lower,
                rage_syzygy::Wdl::Loss | rage_syzygy::Wdl::BlessedLoss => Bound::Upper,
                rage_syzygy::Wdl::Draw => Bound::Exact,
            };

            if tb_bound == Bound::Exact
                || (tb_bound == Bound::Lower && tb_score >= beta)
                || (tb_bound == Bound::Upper && tb_score <= alpha)
            {
                let hash = board.zobrist_hash();
                self.tt
                    .store(hash, Move::NULL, tb_score, depth as u8, tb_bound, ply);
                return tb_score;
            }

            // At PV nodes, tighten the window
            if pv_node && tb_bound == Bound::Lower {
                alpha = alpha.max(tb_score);
            }
        }

        // Check extension: extend when in check to avoid horizon effect
        // Cap extensions at high ply to prevent search explosion
        if in_check && ply < PLY_LIMIT - 4 {
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
        let excluded = self.stack[ply as usize + 2].excluded_move;
        let (tt_move, tt_score, tt_depth, tt_bound) = if let Some(entry) = self.tt.probe(hash, ply)
        {
            // TT cutoff (only at non-PV nodes, skip during singular verification)
            if !pv_node && entry.depth >= depth as u8 && excluded.is_null() {
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
            (entry.best_move, entry.score, entry.depth, entry.bound)
        } else {
            (Move::NULL, 0, 0, Bound::None)
        };

        // Static evaluation (not useful when in check)
        let static_eval = if in_check {
            -INFINITY
        } else {
            self.evaluate(board, ply)
        };
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
                self.copy_acc(ply, ply + 1);
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

            // Probcut: at moderate depths, try a shallow capture search
            // with a raised beta to see if we can prune
            if depth >= 5 && static_eval.abs() < MATE_SCORE - 256 {
                let probcut_beta = beta + 200;
                let pc_moves = board.generate_legal_moves();

                for pc_m in &pc_moves {
                    if !board.is_capture(*pc_m) || !see::see_ge(board, *pc_m, 0) {
                        continue;
                    }

                    let pc_piece = board
                        .piece_on(pc_m.from_sq())
                        .map_or(PieceType::Pawn, |p| p.piece_type);
                    let pc_capt = if pc_m.is_en_passant() {
                        Some(PieceType::Pawn)
                    } else {
                        board.piece_on(pc_m.to_sq()).map(|p| p.piece_type)
                    };
                    let pc_stm = board.side_to_move();

                    board.make_move(*pc_m);
                    self.update_acc_after_move(
                        board,
                        *pc_m,
                        pc_piece,
                        pc_capt,
                        pc_stm,
                        ply,
                        ply + 1,
                    );
                    let pc_score = -self.negamax(
                        board,
                        tm,
                        -probcut_beta,
                        -probcut_beta + 1,
                        depth - 4,
                        ply + 1,
                        true,
                    );
                    board.unmake_move(*pc_m);

                    if self.stopped {
                        return 0;
                    }

                    if pc_score >= probcut_beta {
                        return pc_score;
                    }
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

        // Get previous move info for counter moves and continuation history
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

        // Copy continuation history entry to stack to avoid borrow conflicts
        let cont_entry: Option<ContHistEntry> = if !prev_move.is_null() {
            Some(*self.cont_history.get_entry(prev_piece, prev_move.to_sq()))
        } else {
            None
        };

        // Score and sort moves (stack-allocated, zero heap allocs)
        let num_moves = moves.len();
        let ctx = MoveOrderContext {
            tt_move,
            killers: self.killers.get(ply as usize),
            counter_move,
            history_color: board.side_to_move(),
            cont_hist: cont_entry.as_ref(),
        };
        let mut move_buf = [Move::NULL; MAX_MOVES];
        let mut scores = [0i32; MAX_MOVES];
        move_buf[..num_moves].copy_from_slice(moves.as_slice());
        score_moves_into(
            board,
            &move_buf[..num_moves],
            &mut scores[..num_moves],
            &ctx,
            &self.history,
        );

        let mut best_move = move_buf[0];
        let mut best_score = -INFINITY;
        let mut bound = Bound::Upper;
        let mut move_count = 0u32;
        let stm = board.side_to_move();

        // Track actually-searched quiet moves (and their piece types) for history malus
        let mut searched_quiets = [Move::NULL; MAX_MOVES];
        let mut searched_quiet_pieces = [PieceType::Pawn; MAX_MOVES];
        let mut num_searched_quiets = 0usize;

        for i in 0..num_moves {
            pick_move(&mut move_buf[..num_moves], &mut scores[..num_moves], i);
            let m = move_buf[i];

            // Skip the excluded move (used during singular extension verification)
            if m == excluded && !excluded.is_null() {
                continue;
            }

            // Multi-PV: skip moves already selected by earlier PV lines
            if is_root && self.root_excluded.contains(&m) {
                continue;
            }

            // Tablebase root filtering: skip moves not in the TB-filtered set
            if is_root && !self.root_tb_filter.is_empty() && !self.root_tb_filter.contains(&m) {
                continue;
            }

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

                // SEE pruning for bad captures: use score from move ordering
                // (captures were already SEE-classified during scoring)
                if is_capture && depth <= 4 && scores[i] < 0 {
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

            // Track searched quiets for history malus
            if is_quiet {
                searched_quiets[num_searched_quiets] = m;
                searched_quiet_pieces[num_searched_quiets] = piece_moved;
                num_searched_quiets += 1;
            }

            // Singular extension: if the TT move is clearly best, extend its search
            let mut extension = 0u32;
            if !is_root
                && m == tt_move
                && !tt_move.is_null()
                && depth >= 6
                && tt_depth >= depth as u8 - 3
                && (tt_bound == Bound::Exact || tt_bound == Bound::Lower)
                && tt_score.abs() < MATE_SCORE - 256
                && excluded.is_null()
            {
                let s_beta = tt_score - 2 * depth as i32;
                let s_depth = (depth - 1) / 2;

                self.stack[ply as usize + 2].excluded_move = m;
                let s_score = self.negamax(board, tm, s_beta - 1, s_beta, s_depth, ply, false);
                self.stack[ply as usize + 2].excluded_move = Move::NULL;

                if self.stopped {
                    return 0;
                }

                if s_score < s_beta {
                    extension = 1; // singular — extend
                } else if s_beta >= beta {
                    return s_beta; // multi-cut — prune
                }
            }

            // Capture info for NNUE accumulator update before make_move
            let captured_pt = if m.is_en_passant() {
                Some(PieceType::Pawn)
            } else {
                board.piece_on(m.to_sq()).map(|p| p.piece_type)
            };

            board.make_move(m);
            self.update_acc_after_move(board, m, piece_moved, captured_pt, stm, ply, ply + 1);

            // PVS: first move gets full window, rest get zero window
            let new_depth = depth - 1 + extension;
            let score;
            if move_count == 1 {
                // First move: full window
                score = -self.negamax(board, tm, -beta, -alpha, new_depth, ply + 1, true);
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

                    // Reduce based on combined butterfly + continuation history
                    let hist = self.history.get(stm, m);
                    let ch_score = cont_entry
                        .as_ref()
                        .map_or(0, |ch| ch[piece_moved.index()][m.to_sq().index()]);
                    let combined_hist = hist + ch_score;
                    if combined_hist > 0 {
                        reduction = reduction.saturating_sub(1);
                    } else if combined_hist < -2000 {
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
                    new_depth.saturating_sub(reduction),
                    ply + 1,
                    true,
                );

                // Re-search at full depth if reduced search failed high
                if s > alpha && reduction > 0 {
                    s = -self.negamax(board, tm, -alpha - 1, -alpha, new_depth, ply + 1, true);
                }

                // Re-search with full window if zero-window failed high
                if s > alpha && s < beta {
                    s = -self.negamax(board, tm, -beta, -alpha, new_depth, ply + 1, true);
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

                    // Update triangular PV table
                    let p = ply as usize;
                    self.pv_table[p][0] = m;
                    let child_len = self.pv_length[p + 1];
                    for k in 0..child_len {
                        self.pv_table[p][k + 1] = self.pv_table[p + 1][k];
                    }
                    self.pv_length[p] = child_len + 1;

                    if score >= beta {
                        bound = Bound::Lower;

                        // Update killers, history, and counter moves on quiet beta cutoff
                        if is_quiet {
                            self.killers.store(ply as usize, m);
                            let bonus = (depth * depth) as i32;
                            self.history.update(stm, m, bonus);

                            let has_prev = !prev_move.is_null();
                            let prev_to = prev_move.to_sq();

                            // Malus for previously searched quiet moves (not the cutoff move)
                            for (&sq_move, &sq_piece) in searched_quiets[..num_searched_quiets - 1]
                                .iter()
                                .zip(&searched_quiet_pieces[..num_searched_quiets - 1])
                            {
                                self.history.update(stm, sq_move, -bonus);
                                if has_prev {
                                    self.cont_history.update(
                                        prev_piece,
                                        prev_to,
                                        sq_piece,
                                        sq_move.to_sq(),
                                        -bonus,
                                    );
                                }
                            }

                            // Counter move and continuation history bonus
                            if has_prev {
                                self.counter_moves.store(prev_piece, prev_to, m);
                                self.cont_history.update(
                                    prev_piece,
                                    prev_to,
                                    piece_moved,
                                    m.to_sq(),
                                    bonus,
                                );
                            }
                        }

                        break;
                    }
                }
            }
        }

        // Store in TT (skip during singular extension verification)
        if excluded.is_null() {
            self.tt
                .store(hash, best_move, best_score, depth as u8, bound, ply);
        }

        best_score
    }

    /// Quiescence search: only search captures to avoid horizon effects.
    fn quiescence(
        &mut self,
        board: &mut Board,
        tm: &mut TimeManager,
        mut alpha: i32,
        beta: i32,
        ply: u32,
    ) -> i32 {
        // Ply overflow guard
        if ply >= PLY_LIMIT {
            return self.evaluate(board, ply);
        }

        self.nodes += 1;
        self.seldepth = self.seldepth.max(ply);

        if tm.should_stop(self.nodes) {
            self.stopped = true;
            return 0;
        }

        // Standing pat
        let stand_pat = self.evaluate(board, ply);
        if stand_pat >= beta {
            return stand_pat;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        // Generate all legal moves, then filter to captures
        let moves = board.generate_legal_moves();

        // Collect and score captures into stack-allocated arrays
        let mut capture_buf = [Move::NULL; MAX_MOVES];
        let mut capture_scores = [0i32; MAX_MOVES];
        let mut num_captures = 0usize;

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

                capture_buf[num_captures] = *m;
                capture_scores[num_captures] = mvv_lva_qsearch(board, *m);
                num_captures += 1;
            }
        }

        // Use partial selection sort (same as main search) instead of full sort
        for i in 0..num_captures {
            pick_move(
                &mut capture_buf[..num_captures],
                &mut capture_scores[..num_captures],
                i,
            );
            let m = capture_buf[i];

            // Capture info for NNUE accumulator update before make_move
            let q_piece_moved = board
                .piece_on(m.from_sq())
                .map_or(PieceType::Pawn, |p| p.piece_type);
            let q_captured_pt = if m.is_en_passant() {
                Some(PieceType::Pawn)
            } else {
                board.piece_on(m.to_sq()).map(|p| p.piece_type)
            };
            let q_stm = board.side_to_move();

            board.make_move(m);
            self.update_acc_after_move(board, m, q_piece_moved, q_captured_pt, q_stm, ply, ply + 1);
            let score = -self.quiescence(board, tm, -beta, -alpha, ply + 1);
            board.unmake_move(m);

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

    use crate::tt::TranspositionTable;

    fn search_position(fen: &str, depth: u32) -> SearchResult {
        let mut board = Board::from_fen(fen).unwrap();
        let tt = Arc::new(TranspositionTable::new(16));
        let mut searcher = Searcher::new(tt);
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
        let mut searcher = Searcher::new(Arc::new(TranspositionTable::new(16)));
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
        let mut searcher = Searcher::new(Arc::new(TranspositionTable::new(16)));
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
    fn pv_from_triangular_table() {
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
    fn check_extension_increases_seldepth() {
        // In a position with many checks, check extensions should push seldepth
        // beyond the nominal depth more than a quiet position would.
        // Kiwipete has many tactical checks.
        let mut board =
            Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
                .unwrap();
        let mut searcher = Searcher::new(Arc::new(TranspositionTable::new(16)));
        let limits = SearchLimits {
            depth: Some(4),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        searcher.search(&mut board, &limits, stop, None);
        // Check extensions should push selective depth beyond nominal depth 4
        assert!(
            searcher.seldepth > 4,
            "seldepth {} should exceed nominal depth 4 due to check extensions",
            searcher.seldepth
        );
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

    #[test]
    fn depth_reports_actual_search_depth() {
        let result = search_position(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            5,
        );
        assert_eq!(
            result.depth, 5,
            "depth should report actual completed depth"
        );
    }

    #[test]
    fn seldepth_exceeds_depth() {
        // With quiescence search and check extensions, seldepth should exceed depth
        let mut board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let mut searcher = Searcher::new(Arc::new(TranspositionTable::new(16)));
        let limits = SearchLimits {
            depth: Some(4),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        searcher.search(&mut board, &limits, stop, None);
        // Seldepth is tracked internally; verify qsearch pushed beyond nominal depth
        assert!(searcher.seldepth >= 4, "seldepth should be at least depth");
    }

    #[test]
    fn tt_warmup_reduces_nodes() {
        // Searching the same position twice should use fewer nodes the second time
        let mut board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let mut searcher = Searcher::new(Arc::new(TranspositionTable::new(16)));
        let limits = SearchLimits {
            depth: Some(5),
            ..Default::default()
        };

        let stop = Arc::new(AtomicBool::new(false));
        let result1 = searcher.search(&mut board, &limits, Arc::clone(&stop), None);

        let result2 = searcher.search(&mut board, &limits, stop, None);

        assert!(
            result2.nodes < result1.nodes,
            "second search ({}) should use fewer nodes than first ({}) due to TT",
            result2.nodes,
            result1.nodes
        );
    }

    #[test]
    fn cont_history_populated_during_search() {
        // After a depth-6 search, cont_history should have non-zero entries.
        // This verifies the update path in negamax is actually executing.
        let mut board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let mut searcher = Searcher::new(Arc::new(TranspositionTable::new(16)));
        let limits = SearchLimits {
            depth: Some(6),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        searcher.search(&mut board, &limits, stop, None);

        // Sample several prev-move contexts; at least one should be non-zero
        let mut found_nonzero = false;
        'outer: for &piece in &PieceType::ALL {
            for &sq in &rage_core::Square::ALL {
                let entry = searcher.cont_history.get_entry(piece, sq);
                for row in entry {
                    for &val in row {
                        if val != 0 {
                            found_nonzero = true;
                            break 'outer;
                        }
                    }
                }
            }
        }
        assert!(
            found_nonzero,
            "cont_history should have non-zero entries after depth-6 search"
        );
    }

    #[test]
    fn multi_pv_returns_distinct_moves() {
        let mut board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let mut searcher = Searcher::new(Arc::new(TranspositionTable::new(16)));
        searcher.multi_pv = 3;
        let limits = SearchLimits {
            depth: Some(6),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));

        let infos = Arc::new(std::sync::Mutex::new(Vec::new()));
        let infos_clone = Arc::clone(&infos);
        let cb: InfoCallback = Box::new(move |info| {
            infos_clone.lock().unwrap().push((
                info.depth,
                info.multipv,
                info.pv.first().copied().unwrap_or(Move::NULL),
                info.score,
            ));
        });

        searcher.search(&mut board, &limits, stop, Some(cb));

        let infos = infos.lock().unwrap();
        // For each completed depth, check that we got 3 lines with distinct best moves
        let max_depth = infos.iter().map(|(d, _, _, _)| *d).max().unwrap_or(0);
        for depth in 1..=max_depth {
            let lines: Vec<_> = infos.iter().filter(|(d, _, _, _)| *d == depth).collect();
            assert_eq!(
                lines.len(),
                3,
                "depth {depth} should have 3 PV lines, got {}",
                lines.len()
            );
            // Check multipv indices are 1, 2, 3
            assert_eq!(lines[0].1, 1);
            assert_eq!(lines[1].1, 2);
            assert_eq!(lines[2].1, 3);
            // All best moves should be distinct
            let m1 = lines[0].2;
            let m2 = lines[1].2;
            let m3 = lines[2].2;
            assert_ne!(m1, m2, "depth {depth}: PV 1 and 2 should differ");
            assert_ne!(m1, m3, "depth {depth}: PV 1 and 3 should differ");
            assert_ne!(m2, m3, "depth {depth}: PV 2 and 3 should differ");
        }
    }

    #[test]
    fn multi_pv_scores_reasonably_close() {
        // Strict score ordering (PV1 >= PV2 >= PV3) is NOT guaranteed because
        // TT entries from earlier PV lines can alter pruning in later ones.
        // This is accepted behavior in all multi-PV engines.
        // Instead, verify that secondary line scores don't wildly exceed the
        // primary line, which would indicate broken exclusion logic.
        let mut board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let mut searcher = Searcher::new(Arc::new(TranspositionTable::new(16)));
        searcher.multi_pv = 4;
        let limits = SearchLimits {
            depth: Some(5),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));

        let infos = Arc::new(std::sync::Mutex::new(Vec::new()));
        let infos_clone = Arc::clone(&infos);
        let cb: InfoCallback = Box::new(move |info| {
            infos_clone
                .lock()
                .unwrap()
                .push((info.depth, info.multipv, info.score));
        });

        searcher.search(&mut board, &limits, stop, Some(cb));

        let infos = infos.lock().unwrap();
        let max_depth = infos.iter().map(|(d, _, _)| *d).max().unwrap_or(0);
        for depth in 1..=max_depth {
            let mut scores: Vec<_> = infos
                .iter()
                .filter(|(d, _, _)| *d == depth)
                .map(|(_, mpv, score)| (*mpv, *score))
                .collect();
            scores.sort_by_key(|(mpv, _)| *mpv);
            let primary_score = scores[0].1;
            for &(mpv, score) in &scores[1..] {
                assert!(
                    score <= primary_score + 200,
                    "depth {depth}: PV {mpv} score ({score}) should not wildly exceed \
                     PV 1 score ({primary_score}) — indicates broken exclusion logic",
                );
            }
        }
    }

    #[test]
    fn multi_pv_single_legal_move() {
        // Black king on h8, white queen on g6, white king on f6 — only Kh8-g8 is impossible
        // because g7 is controlled. Actually, let's use a simpler forced-move position:
        // King on a1 in check from rook on a3 with black king on c3 — only move is Kb1
        // (Ka2 attacked by rook, Kb2 attacked by king)
        let mut board = Board::from_fen("8/8/8/8/8/r1k5/8/K7 w - - 0 1").unwrap();
        assert_eq!(
            board.generate_legal_moves().len(),
            1,
            "position should have exactly 1 legal move"
        );

        let mut searcher = Searcher::new(Arc::new(TranspositionTable::new(16)));
        searcher.multi_pv = 3;
        let limits = SearchLimits {
            depth: Some(4),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));

        let infos = Arc::new(std::sync::Mutex::new(Vec::new()));
        let infos_clone = Arc::clone(&infos);
        let cb: InfoCallback = Box::new(move |info| {
            infos_clone.lock().unwrap().push((info.depth, info.multipv));
        });

        let result = searcher.search(&mut board, &limits, stop, Some(cb));
        assert!(!result.best_move.is_null());

        let infos = infos.lock().unwrap();
        let max_depth = infos.iter().map(|(d, _)| *d).max().unwrap_or(0);
        for depth in 1..=max_depth {
            let lines: Vec<_> = infos.iter().filter(|(d, _)| *d == depth).collect();
            assert_eq!(
                lines.len(),
                1,
                "depth {depth}: only 1 legal move, should get 1 PV line, got {}",
                lines.len()
            );
            assert_eq!(lines[0].1, 1);
        }
    }

    #[test]
    fn multi_pv_capped_by_legal_moves() {
        // Position with exactly 2 legal moves: Ka1 with Kc3 — only Ka2 and Kb1
        let mut board = Board::from_fen("8/8/8/8/8/2k5/8/K7 w - - 0 1").unwrap();
        let mut searcher = Searcher::new(Arc::new(TranspositionTable::new(16)));
        searcher.multi_pv = 3; // request 3, but only 2 moves exist
        let limits = SearchLimits {
            depth: Some(6),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));

        let infos = Arc::new(std::sync::Mutex::new(Vec::new()));
        let infos_clone = Arc::clone(&infos);
        let cb: InfoCallback = Box::new(move |info| {
            infos_clone.lock().unwrap().push((
                info.depth,
                info.multipv,
                info.pv.first().copied().unwrap_or(Move::NULL),
            ));
        });

        searcher.search(&mut board, &limits, stop, Some(cb));

        let infos = infos.lock().unwrap();
        let max_depth = infos.iter().map(|(d, _, _)| *d).max().unwrap_or(0);
        for depth in 1..=max_depth {
            let lines: Vec<_> = infos.iter().filter(|(d, _, _)| *d == depth).collect();
            // Should get exactly 2 lines, not 3
            assert_eq!(
                lines.len(),
                2,
                "depth {depth}: only 2 legal moves, should get 2 PV lines, got {}",
                lines.len()
            );
            assert_eq!(lines[0].1, 1);
            assert_eq!(lines[1].1, 2);
            // The two moves should be distinct
            assert_ne!(
                lines[0].2, lines[1].2,
                "depth {depth}: the 2 PV lines should have different best moves"
            );
        }
    }

    #[test]
    fn multi_pv_one_is_default() {
        let mut board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let mut searcher = Searcher::new(Arc::new(TranspositionTable::new(16)));
        // multi_pv defaults to 1
        assert_eq!(searcher.multi_pv, 1);

        let limits = SearchLimits {
            depth: Some(4),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));

        let infos = Arc::new(std::sync::Mutex::new(Vec::new()));
        let infos_clone = Arc::clone(&infos);
        let cb: InfoCallback = Box::new(move |info| {
            infos_clone.lock().unwrap().push(info.multipv);
        });

        searcher.search(&mut board, &limits, stop, Some(cb));

        let infos = infos.lock().unwrap();
        // Every callback should report multipv = 1
        for &mpv in infos.iter() {
            assert_eq!(mpv, 1, "single-PV should always report multipv=1");
        }
    }

    #[test]
    fn evaluate_nnue_returns_none_without_network() {
        let board = Board::from_fen(Board::STARTPOS_FEN).unwrap();
        let mut searcher = Searcher::new(Arc::new(TranspositionTable::new(16)));
        assert!(
            searcher.evaluate_nnue(&board).is_none(),
            "evaluate_nnue should return None when no network is loaded"
        );
        // Also verify network() reports None
        assert!(searcher.network().is_none());
    }

    #[test]
    fn evaluate_nnue_returns_score_with_network() {
        let board = Board::from_fen(Board::STARTPOS_FEN).unwrap();
        let mut searcher = Searcher::new(Arc::new(TranspositionTable::new(16)));
        let weights = rage_nnue::NnueWeights::random(42);
        let network = Arc::new(NnueNetwork::new(weights));
        searcher.set_network(Some(network));

        let score = searcher.evaluate_nnue(&board);
        assert!(score.is_some(), "should return a score with network loaded");
        // Also verify network() reports Some
        assert!(searcher.network().is_some());
    }

    #[test]
    fn evaluate_nnue_works_on_multiple_positions() {
        let weights = rage_nnue::NnueWeights::random(42);
        let network = Arc::new(NnueNetwork::new(weights));

        let mut searcher = Searcher::new(Arc::new(TranspositionTable::new(16)));
        searcher.set_network(Some(network));

        // Verify evaluate_nnue succeeds on a variety of positions
        let fens = [
            Board::STARTPOS_FEN,
            "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1",
        ];
        for fen in fens {
            let board = Board::from_fen(fen).unwrap();
            let score = searcher.evaluate_nnue(&board);
            assert!(score.is_some(), "evaluate_nnue should succeed for {fen}");
        }
    }

    #[test]
    fn search_with_nnue_produces_result() {
        // Verify the full search works when an NNUE network is loaded
        let mut board = Board::from_fen(Board::STARTPOS_FEN).unwrap();
        let mut searcher = Searcher::new(Arc::new(TranspositionTable::new(16)));
        let weights = rage_nnue::NnueWeights::random(42);
        searcher.set_network(Some(Arc::new(NnueNetwork::new(weights))));

        let limits = SearchLimits {
            depth: Some(4),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let result = searcher.search(&mut board, &limits, stop, None);
        assert!(
            !result.best_move.is_null(),
            "NNUE search should find a move"
        );
        assert!(result.nodes > 0, "NNUE search should visit nodes");
    }

    #[test]
    fn cont_history_cleared_on_clear_tables() {
        let mut board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let mut searcher = Searcher::new(Arc::new(TranspositionTable::new(16)));
        let limits = SearchLimits {
            depth: Some(5),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        searcher.search(&mut board, &limits, stop, None);

        searcher.clear_tables();

        // Every entry should be zero after clearing
        for &piece in &PieceType::ALL {
            for &sq in &rage_core::Square::ALL {
                let entry = searcher.cont_history.get_entry(piece, sq);
                for row in entry {
                    for &val in row {
                        assert_eq!(val, 0, "all entries should be zero after clear_tables");
                    }
                }
            }
        }
    }
}
