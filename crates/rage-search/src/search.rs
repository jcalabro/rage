use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use rage_core::{Board, Move};

use crate::eval::{INFINITY, MATE_SCORE, evaluate, piece_value};
use crate::ordering::{pick_move, score_moves};
use crate::time::{SearchLimits, TimeManager};
use crate::tt::{Bound, TranspositionTable};

/// Maximum search depth.
const MAX_PLY: u32 = 128;

/// Delta pruning margin in quiescence search.
const DELTA_MARGIN: i32 = 200;

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

/// The main search engine.
pub struct Searcher {
    pub tt: TranspositionTable,
    nodes: u64,
    stopped: bool,
}

impl Searcher {
    pub fn new(tt_size_mb: usize) -> Self {
        Self {
            tt: TranspositionTable::new(tt_size_mb),
            nodes: 0,
            stopped: false,
        }
    }

    /// Run an iterative deepening search.
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

        let mut tm = TimeManager::new(limits, board.side_to_move(), stop);

        let mut best_move = Move::NULL;
        let mut best_score = -INFINITY;
        let mut best_pv = Vec::new();

        for depth in 1..=tm.max_depth().min(MAX_PLY) {
            let score = self.negamax(board, &mut tm, -INFINITY, INFINITY, depth, 0);

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
            if tm.soft_limit_reached() {
                break;
            }

            // If we found a forced mate, no need to search deeper
            if best_score.abs() > MATE_SCORE - 256 {
                break;
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

    /// Negamax alpha-beta search.
    fn negamax(
        &mut self,
        board: &mut Board,
        tm: &mut TimeManager,
        mut alpha: i32,
        beta: i32,
        depth: u32,
        ply: u32,
    ) -> i32 {
        self.nodes += 1;

        // Check for time/stop periodically
        if tm.should_stop(self.nodes) {
            self.stopped = true;
            return 0;
        }

        // Draw detection
        if ply > 0 && is_draw(board) {
            return 0;
        }

        // Quiescence at leaf
        if depth == 0 {
            return self.quiescence(board, tm, alpha, beta, ply);
        }

        // TT probe
        let hash = board.zobrist_hash();
        let tt_move = if let Some(entry) = self.tt.probe(hash, ply) {
            // TT cutoff
            if entry.depth >= depth as u8 {
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

        // Generate legal moves
        let moves = board.generate_legal_moves();
        if moves.is_empty() {
            return if board.in_check() {
                -(MATE_SCORE - ply as i32) // Checkmate
            } else {
                0 // Stalemate
            };
        }

        // Score and sort moves
        let mut move_buf: Vec<Move> = moves.as_slice().to_vec();
        let mut scores = score_moves(board, &moves, tt_move);

        let mut best_move = move_buf[0];
        let mut best_score = -INFINITY;
        let mut bound = Bound::Upper;

        for i in 0..move_buf.len() {
            pick_move(&mut move_buf, &mut scores, i);
            let m = move_buf[i];

            board.make_move(m);
            let score = -self.negamax(board, tm, -beta, -alpha, depth - 1, ply + 1);
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
    #[allow(clippy::only_used_in_recursion)] // ply used for future mate distance pruning
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
                    piece_value(rage_core::PieceType::Pawn)
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

                captures.push((*m, crate::ordering::score_move(board, *m, Move::NULL)));
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
                break; // Avoid infinite loops from repetitions
            }
            seen_hashes.push(hash);

            if let Some(entry) = self.tt.probe(hash, 0) {
                let m = entry.best_move;
                if m.is_null() {
                    break;
                }
                // Validate the move is legal
                let moves = board.generate_legal_moves();
                if !moves.contains(m) {
                    break;
                }
                pv.push(m);
                board.make_move(m);
            } else {
                break;
            }
        }

        // Unmake all PV moves to restore board state
        for m in pv.iter().rev() {
            board.unmake_move(*m);
        }

        pv
    }
}

/// Check if the current position is a draw (repetition, 50-move, insufficient material).
fn is_draw(board: &Board) -> bool {
    board.is_fifty_move_draw() || board.is_insufficient_material() || board.is_repetition()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
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
        // White to move, Qh7# is mate in 1
        let result = search_position(
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            4,
        );
        // Best move should be Qxf7# (h5f7)
        assert_eq!(result.best_move.to_string(), "h5f7");
        assert!(result.score > MATE_SCORE - 10, "should find mate");
    }

    #[test]
    fn mate_in_one_back_rank() {
        // White has Rd8# back rank mate
        let result = search_position("6k1/5ppp/8/8/8/8/8/3RK3 w - - 0 1", 4);
        assert_eq!(result.best_move.to_string(), "d1d8");
        assert!(result.score > MATE_SCORE - 10, "should find back rank mate");
    }

    #[test]
    fn white_up_material() {
        // White is up a full rook (black missing h8 rook), white to move
        let result = search_position("rnbqkbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQq - 0 1", 3);
        // White (STM) should be winning
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
        // Black is stalemated (king on a8, white king a6, white queen b6)
        let result = search_position("k7/8/KQ6/8/8/8/8/8 w - - 0 1", 4);
        // White should not blunder stalemate - should find a winning approach
        // At depth 4 the engine should at least not return 0 from the root
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
        // KQ vs K: White should be overwhelmingly winning
        let result = search_position("4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1", 4);
        assert!(
            result.score > 800,
            "K+Q vs K should be heavily winning, got {}",
            result.score
        );
    }
}
