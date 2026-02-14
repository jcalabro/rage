use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use rage_core::{Board, Color};
use rage_search::search::Searcher;
use rage_search::time::SearchLimits;
use rand::Rng;

use crate::data::{TrainingEntry, WdlResult};

/// Configuration for self-play data generation.
pub struct DatagenConfig {
    /// Number of games to play.
    pub num_games: usize,
    /// Search depth for each position evaluation.
    pub depth: u32,
    /// Random plies from start before recording data.
    pub random_plies: u32,
    /// Minimum absolute score to skip (avoid recording drawn positions).
    pub adjudication_score: i32,
    /// Maximum game length in plies.
    pub max_plies: u32,
}

impl Default for DatagenConfig {
    fn default() -> Self {
        Self {
            num_games: 1000,
            depth: 8,
            random_plies: 8,
            adjudication_score: 3000,
            max_plies: 400,
        }
    }
}

/// Result of a finished game for WDL labeling.
enum GameResult {
    WhiteWin,
    BlackWin,
    Draw,
}

/// Generate training data via self-play and write to binary file.
pub fn generate_data(config: &DatagenConfig, output_path: &Path) -> std::io::Result<usize> {
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);
    let mut total_positions = 0usize;
    let mut rng = rand::thread_rng();

    for game_idx in 0..config.num_games {
        if game_idx % 100 == 0 {
            eprintln!(
                "Game {}/{} ({} positions so far)",
                game_idx, config.num_games, total_positions
            );
        }

        let positions = play_game(config, &mut rng);
        for entry in &positions {
            writer.write_all(&entry.to_bytes())?;
        }
        total_positions += positions.len();
    }

    writer.flush()?;
    eprintln!(
        "Generated {} positions from {} games",
        total_positions, config.num_games
    );
    Ok(total_positions)
}

/// Play a single self-play game and collect training entries.
fn play_game<R: Rng>(config: &DatagenConfig, rng: &mut R) -> Vec<TrainingEntry> {
    let mut board = Board::starting_position();
    let stop = Arc::new(AtomicBool::new(false));
    let mut searcher = Searcher::new(16); // 16 MB TT

    // Collected positions (without WDL yet)
    let mut positions: Vec<(TrainingEntry, Color)> = Vec::new();

    // Play random opening moves
    for _ in 0..config.random_plies {
        let moves = board.generate_legal_moves();
        if moves.is_empty() {
            break;
        }
        let idx = rng.gen_range(0..moves.len());
        board.make_move(moves[idx]);
    }

    // Play the game with search
    let mut ply_count = 0u32;
    loop {
        if ply_count >= config.max_plies {
            break;
        }

        // Check for game-ending conditions
        let moves = board.generate_legal_moves();
        if moves.is_empty() {
            break; // checkmate or stalemate
        }
        if board.is_fifty_move_draw() || board.is_repetition() || board.is_insufficient_material() {
            break;
        }

        // Search the position
        let limits = SearchLimits {
            depth: Some(config.depth),
            ..Default::default()
        };
        stop.store(false, Ordering::Relaxed);
        let result = searcher.search(&mut board, &limits, stop.clone(), None);

        let score = result.score;
        let best_move = result.best_move;

        // Adjudicate if score is too extreme
        if score.abs() >= config.adjudication_score {
            break;
        }

        // Record position (score from STM perspective)
        let stm = board.side_to_move();
        let entry =
            TrainingEntry::from_board(&board, score as i16, WdlResult::Draw, best_move.raw());
        positions.push((entry, stm));

        // Make the best move
        board.make_move(best_move);
        ply_count += 1;
    }

    // Determine game result
    let game_result = determine_result(&mut board);

    // Label all positions with the game result
    positions
        .into_iter()
        .map(|(mut entry, stm)| {
            entry.wdl = match (&game_result, stm) {
                (GameResult::WhiteWin, Color::White) | (GameResult::BlackWin, Color::Black) => {
                    WdlResult::Win
                }
                (GameResult::WhiteWin, Color::Black) | (GameResult::BlackWin, Color::White) => {
                    WdlResult::Loss
                }
                (GameResult::Draw, _) => WdlResult::Draw,
            };
            entry
        })
        .collect()
}

/// Determine the result of a finished game.
fn determine_result(board: &mut Board) -> GameResult {
    if board.is_checkmate() {
        // The side to move is in checkmate, so they lost
        match board.side_to_move() {
            Color::White => GameResult::BlackWin,
            Color::Black => GameResult::WhiteWin,
        }
    } else {
        GameResult::Draw
    }
}
