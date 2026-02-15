//! Single game player.
//!
//! Plays a complete game between two UCI engines with clock management,
//! game-over detection via rage-core Board, and adjudication.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use rage_core::Board;

use crate::types::{
    GameOutcome, GameRecord, LiveGameState, MatchConfig, SearchOverrides, TerminationReason,
    TimeControl,
};
use crate::uci::{GoParams, UciEngine};

/// Timeout for engine sync commands.
const SYNC_TIMEOUT: Duration = Duration::from_secs(10);

/// Timeout for a single move (hard ceiling beyond time control).
const MOVE_TIMEOUT: Duration = Duration::from_secs(120);

/// Play a single game between two engines.
///
/// `white` and `black` are the engines. `fen` is the starting position.
/// `engine_a_is_white` indicates which engine is "engine A" for result tracking.
pub fn play_game(
    white: &mut UciEngine,
    black: &mut UciEngine,
    fen: &str,
    config: &MatchConfig,
    engine_a_is_white: bool,
    live_state: Option<&Mutex<LiveGameState>>,
) -> GameRecord {
    let start = Instant::now();

    let Ok(mut board) = Board::from_fen(fen) else {
        return GameRecord {
            opening_fen: fen.to_string(),
            outcome: GameOutcome::Draw,
            termination: TerminationReason::IllegalMove,
            ply_count: 0,
            moves: Vec::new(),
            duration: Duration::ZERO,
            engine_a_is_white,
        };
    };

    let mut moves: Vec<String> = Vec::new();
    let mut white_time_ms = config.tc.base_ms as i64;
    let mut black_time_ms = config.tc.base_ms as i64;

    // Adjudication state
    let mut consecutive_win_plies = 0i32;
    let mut win_side: Option<GameOutcome> = None;
    let mut consecutive_draw_plies = 0u32;

    // Tell engines about the new game
    let _ = white.sync(SYNC_TIMEOUT);
    let _ = black.sync(SYNC_TIMEOUT);

    for ply in 0..(config.max_moves * 2) {
        let is_white_turn = board.side_to_move() == rage_core::Color::White;
        let engine = if is_white_turn {
            &mut *white
        } else {
            &mut *black
        };

        // Set position
        if let Err(_e) = engine.set_position(fen, &moves) {
            return make_record(
                fen,
                if is_white_turn {
                    GameOutcome::BlackWin
                } else {
                    GameOutcome::WhiteWin
                },
                TerminationReason::EngineCrash,
                ply,
                moves,
                start.elapsed(),
                engine_a_is_white,
            );
        }

        // Build go params
        let go_params = build_go_params(
            white_time_ms,
            black_time_ms,
            &config.tc,
            &config.search_overrides,
        );

        let move_start = Instant::now();
        let result = engine.go(&go_params, MOVE_TIMEOUT);

        let go_result = match result {
            Ok(r) => r,
            Err(_e) => {
                return make_record(
                    fen,
                    if is_white_turn {
                        GameOutcome::BlackWin
                    } else {
                        GameOutcome::WhiteWin
                    },
                    TerminationReason::EngineCrash,
                    ply,
                    moves,
                    start.elapsed(),
                    engine_a_is_white,
                );
            }
        };

        // Update clock (only if using time control, not fixed depth/nodes)
        if config.search_overrides.depth.is_none() && config.search_overrides.nodes.is_none() {
            let elapsed_ms = move_start.elapsed().as_millis() as i64;
            let overhead = config.move_overhead_ms as i64;
            if is_white_turn {
                white_time_ms -= elapsed_ms + overhead;
                white_time_ms += config.tc.inc_ms as i64;
            } else {
                black_time_ms -= elapsed_ms + overhead;
                black_time_ms += config.tc.inc_ms as i64;
            }

            // Flag: engine ran out of time
            if (is_white_turn && white_time_ms < 0) || (!is_white_turn && black_time_ms < 0) {
                return make_record(
                    fen,
                    if is_white_turn {
                        GameOutcome::BlackWin
                    } else {
                        GameOutcome::WhiteWin
                    },
                    TerminationReason::EngineTimeout,
                    ply,
                    moves,
                    start.elapsed(),
                    engine_a_is_white,
                );
            }
        }

        // Try to apply the move
        let bestmove = &go_result.bestmove;
        let parsed_move = board.parse_uci_move(bestmove);
        let chess_move = match parsed_move {
            Ok(m) => m,
            Err(_) => {
                return make_record(
                    fen,
                    if is_white_turn {
                        GameOutcome::BlackWin
                    } else {
                        GameOutcome::WhiteWin
                    },
                    TerminationReason::IllegalMove,
                    ply,
                    moves,
                    start.elapsed(),
                    engine_a_is_white,
                );
            }
        };

        board.make_move(chess_move);
        moves.push(bestmove.clone());

        // Update live game state for TUI
        if let Some(ls) = live_state
            && let Ok(mut state) = ls.lock()
        {
            state.fen = board.to_fen();
            state.last_move = Some(bestmove.clone());
            state.move_number = (ply as u16 / 2) + 1;
            state.score_cp = go_result.score_cp;
            state.is_mate_score = go_result.is_mate_score;
            state.ply = ply + 1;
        }

        // Check game termination conditions
        if board.is_checkmate() {
            // The side that just moved delivered checkmate
            let outcome = if is_white_turn {
                GameOutcome::WhiteWin
            } else {
                GameOutcome::BlackWin
            };
            return make_record(
                fen,
                outcome,
                TerminationReason::Checkmate,
                ply + 1,
                moves,
                start.elapsed(),
                engine_a_is_white,
            );
        }

        if board.is_stalemate() {
            return make_record(
                fen,
                GameOutcome::Draw,
                TerminationReason::Stalemate,
                ply + 1,
                moves,
                start.elapsed(),
                engine_a_is_white,
            );
        }

        if board.is_fifty_move_draw() {
            return make_record(
                fen,
                GameOutcome::Draw,
                TerminationReason::FiftyMoveRule,
                ply + 1,
                moves,
                start.elapsed(),
                engine_a_is_white,
            );
        }

        if board.is_repetition() {
            return make_record(
                fen,
                GameOutcome::Draw,
                TerminationReason::ThreefoldRepetition,
                ply + 1,
                moves,
                start.elapsed(),
                engine_a_is_white,
            );
        }

        if board.is_insufficient_material() {
            return make_record(
                fen,
                GameOutcome::Draw,
                TerminationReason::InsufficientMaterial,
                ply + 1,
                moves,
                start.elapsed(),
                engine_a_is_white,
            );
        }

        // Score-based adjudication
        if let Some(score) = go_result.score_cp {
            if !go_result.is_mate_score {
                // Win adjudication
                let abs_score = score.unsigned_abs() as i32;
                if abs_score >= config.win_adj_score {
                    let current_winner =
                        if (is_white_turn && score > 0) || (!is_white_turn && score < 0) {
                            GameOutcome::WhiteWin
                        } else {
                            GameOutcome::BlackWin
                        };
                    if win_side == Some(current_winner) {
                        consecutive_win_plies += 1;
                    } else {
                        win_side = Some(current_winner);
                        consecutive_win_plies = 1;
                    }
                    if consecutive_win_plies >= config.win_adj_count as i32 {
                        return make_record(
                            fen,
                            current_winner,
                            TerminationReason::AdjudicationWin,
                            ply + 1,
                            moves,
                            start.elapsed(),
                            engine_a_is_white,
                        );
                    }
                } else {
                    consecutive_win_plies = 0;
                    win_side = None;
                }

                // Draw adjudication
                if abs_score <= config.draw_adj_score {
                    consecutive_draw_plies += 1;
                    if consecutive_draw_plies >= config.draw_adj_count {
                        return make_record(
                            fen,
                            GameOutcome::Draw,
                            TerminationReason::AdjudicationDraw,
                            ply + 1,
                            moves,
                            start.elapsed(),
                            engine_a_is_white,
                        );
                    }
                } else {
                    consecutive_draw_plies = 0;
                }
            } else {
                // Mate score: treat as win adjudication immediately
                let winner = if (is_white_turn && score > 0) || (!is_white_turn && score < 0) {
                    GameOutcome::WhiteWin
                } else {
                    GameOutcome::BlackWin
                };
                consecutive_win_plies += 1;
                win_side = Some(winner);
                if consecutive_win_plies >= 2 {
                    return make_record(
                        fen,
                        winner,
                        TerminationReason::AdjudicationWin,
                        ply + 1,
                        moves,
                        start.elapsed(),
                        engine_a_is_white,
                    );
                }
                consecutive_draw_plies = 0;
            }
        }
    }

    // Max moves reached
    make_record(
        fen,
        GameOutcome::Draw,
        TerminationReason::MaxMovesReached,
        config.max_moves * 2,
        moves,
        start.elapsed(),
        engine_a_is_white,
    )
}

fn build_go_params(
    white_time_ms: i64,
    black_time_ms: i64,
    tc: &TimeControl,
    overrides: &SearchOverrides,
) -> GoParams {
    if let Some(depth) = overrides.depth {
        return GoParams {
            depth: Some(depth),
            ..Default::default()
        };
    }
    if let Some(nodes) = overrides.nodes {
        return GoParams {
            nodes: Some(nodes),
            ..Default::default()
        };
    }

    let wtime = white_time_ms.max(1) as u64;
    let btime = black_time_ms.max(1) as u64;

    GoParams {
        wtime: Some(wtime),
        btime: Some(btime),
        winc: if tc.inc_ms > 0 { Some(tc.inc_ms) } else { None },
        binc: if tc.inc_ms > 0 { Some(tc.inc_ms) } else { None },
        movestogo: if tc.movestogo > 0 {
            Some(tc.movestogo)
        } else {
            None
        },
        ..Default::default()
    }
}

fn make_record(
    fen: &str,
    outcome: GameOutcome,
    termination: TerminationReason,
    ply_count: u32,
    moves: Vec<String>,
    duration: Duration,
    engine_a_is_white: bool,
) -> GameRecord {
    GameRecord {
        opening_fen: fen.to_string(),
        outcome,
        termination,
        ply_count,
        moves,
        duration,
        engine_a_is_white,
    }
}
