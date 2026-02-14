use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use rage_core::{Board, Color};
use rage_search::search::Searcher;
use rage_search::time::SearchLimits;
use rage_search::tt::TranspositionTable;
use rand::Rng;
use rayon::prelude::*;

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
    /// Number of threads for parallel game generation.
    pub threads: usize,
}

impl Default for DatagenConfig {
    fn default() -> Self {
        Self {
            num_games: 1000,
            depth: 8,
            random_plies: 8,
            adjudication_score: 3000,
            max_plies: 400,
            threads: 0, // 0 = use all available cores
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
    let threads = if config.threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        config.threads
    };

    eprintln!(
        "Generating {} games at depth {} with {} threads...",
        config.num_games, config.depth, threads
    );

    // Configure rayon thread pool
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(std::io::Error::other)?;

    let games_completed = AtomicUsize::new(0);

    let all_positions: Vec<Vec<TrainingEntry>> = pool.install(|| {
        (0..config.num_games)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                let positions = play_game(config, &mut rng);

                let done = games_completed.fetch_add(1, Ordering::Relaxed) + 1;
                if done.is_multiple_of(100) || done == config.num_games {
                    eprintln!("  Games: {}/{}", done, config.num_games);
                }

                positions
            })
            .collect()
    });

    // Write all entries sequentially
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);
    let mut total_positions = 0usize;

    for positions in &all_positions {
        for entry in positions {
            writer.write_all(&entry.to_bytes())?;
        }
        total_positions += positions.len();
    }
    writer.flush()?;

    eprintln!(
        "Generated {} positions from {} games -> {}",
        total_positions,
        config.num_games,
        output_path.display()
    );
    Ok(total_positions)
}

/// Play a single self-play game and collect training entries.
fn play_game<R: Rng>(config: &DatagenConfig, rng: &mut R) -> Vec<TrainingEntry> {
    let mut board = Board::starting_position();
    let stop = Arc::new(AtomicBool::new(false));
    let tt = Arc::new(TranspositionTable::new(16));
    let mut searcher = Searcher::new(tt); // 16 MB TT

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
