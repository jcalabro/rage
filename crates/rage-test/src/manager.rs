//! Concurrent match runner.
//!
//! Executes game pairs in parallel across worker threads. Each worker owns
//! two engine instances and plays game pairs from a shared opening book.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use crate::game::play_game;
use crate::opening::OpeningBook;
use crate::types::{MatchConfig, PairResult};
use crate::uci::UciEngine;

/// How many pairs to play.
#[derive(Debug, Clone, Copy)]
pub enum MatchMode {
    /// Play pairs until stopped externally.
    Unbounded,
    /// Play exactly N pairs.
    FixedPairs(u32),
}

/// Callback invoked after each completed game pair.
pub type PairCallback = Box<dyn Fn(&PairResult) + Send + Sync>;

/// Run a match: play game pairs across worker threads.
///
/// Returns when all pairs are complete (FixedPairs) or when `stop` is set (Unbounded).
/// Calls `on_pair` after each completed pair (from the collector thread).
pub fn run_match(
    config: &MatchConfig,
    mode: MatchMode,
    book: &OpeningBook,
    stop: Arc<AtomicBool>,
    on_pair: PairCallback,
) -> Vec<PairResult> {
    let concurrency = config.concurrency.max(1) as usize;
    let total_pairs = match mode {
        MatchMode::FixedPairs(n) => Some(n as usize),
        MatchMode::Unbounded => None,
    };

    // Atomic counter for claiming opening indices
    let pair_counter = Arc::new(AtomicUsize::new(0));

    // Channel for results
    let (result_tx, result_rx) = mpsc::channel::<PairResult>();

    // Spawn worker threads
    let mut handles = Vec::new();
    for _worker_id in 0..concurrency {
        let config = config.clone();
        let stop = Arc::clone(&stop);
        let pair_counter = Arc::clone(&pair_counter);
        let result_tx = result_tx.clone();
        let book_fens: Vec<String> = (0..book.len()).map(|i| book.get(i).to_string()).collect();
        let book_len = book.len();

        let handle = std::thread::spawn(move || {
            worker_loop(
                &config,
                &book_fens,
                book_len,
                total_pairs,
                &pair_counter,
                &stop,
                &result_tx,
            );
        });
        handles.push(handle);
    }

    // Drop our copy of the sender so result_rx closes when workers are done
    drop(result_tx);

    // Collect results
    let mut results = Vec::new();
    for pair_result in result_rx {
        on_pair(&pair_result);
        results.push(pair_result);
    }

    // Wait for all workers
    for handle in handles {
        let _ = handle.join();
    }

    results
}

fn worker_loop(
    config: &MatchConfig,
    book_fens: &[String],
    book_len: usize,
    total_pairs: Option<usize>,
    pair_counter: &AtomicUsize,
    stop: &AtomicBool,
    result_tx: &mpsc::Sender<PairResult>,
) {
    let engine_timeout = Duration::from_secs(30);

    // Spawn engines for this worker
    let mut engine_a = match UciEngine::new(&config.engine_a, engine_timeout) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to start engine A: {e}");
            return;
        }
    };
    let mut engine_b = match UciEngine::new(&config.engine_b, engine_timeout) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to start engine B: {e}");
            engine_a.quit();
            return;
        }
    };

    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }

        // Claim next pair index
        let pair_idx = pair_counter.fetch_add(1, Ordering::Relaxed);
        if let Some(total) = total_pairs
            && pair_idx >= total
        {
            break;
        }

        let opening_idx = pair_idx % book_len;
        let fen = &book_fens[opening_idx];

        // Restart engines if crashed
        if !engine_a.is_alive() {
            let _ = std::mem::replace(
                &mut engine_a,
                match UciEngine::new(&config.engine_a, engine_timeout) {
                    Ok(e) => e,
                    Err(_) => break,
                },
            );
        }
        if !engine_b.is_alive() {
            let _ = std::mem::replace(
                &mut engine_b,
                match UciEngine::new(&config.engine_b, engine_timeout) {
                    Ok(e) => e,
                    Err(_) => break,
                },
            );
        }

        // Tell engines about new game
        let _ = engine_a.new_game(engine_timeout);
        let _ = engine_b.new_game(engine_timeout);

        // Game 1: A=white, B=black
        let game_a_white = play_game(&mut engine_a, &mut engine_b, fen, config, true);

        if stop.load(Ordering::Relaxed) {
            break;
        }

        // Game 2: A=black, B=white (same opening)
        let _ = engine_a.new_game(engine_timeout);
        let _ = engine_b.new_game(engine_timeout);
        let game_a_black = play_game(&mut engine_b, &mut engine_a, fen, config, false);

        let pair = PairResult {
            opening_index: opening_idx,
            game_a_white,
            game_a_black,
        };

        if result_tx.send(pair).is_err() {
            break;
        }
    }

    engine_a.quit();
    engine_b.quit();
}
