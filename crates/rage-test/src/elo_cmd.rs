//! Fixed-game Elo test.
//!
//! Plays a fixed number of game pairs and reports the Elo difference
//! with confidence intervals.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use crate::manager::{self, MatchMode};
use crate::opening::OpeningBook;
use crate::stats;
use crate::types::{EloConfig, MatchStats, PairResult};

/// Live progress snapshot for TUI rendering.
#[derive(Debug, Clone)]
pub struct EloProgress {
    pub stats: MatchStats,
    pub pairs_target: u32,
    pub elo: Option<stats::EloEstimate>,
    pub elapsed: Duration,
    pub games_per_min: f64,
    pub latest_pair: Option<PairResult>,
}

/// Final Elo report.
#[derive(Debug, Clone)]
pub struct EloReport {
    pub stats: MatchStats,
    pub elo: Option<stats::EloEstimate>,
    pub elapsed: Duration,
    pub games_per_min: f64,
}

/// Run a fixed-game Elo test.
pub fn run_elo_test(
    config: &EloConfig,
    book: &mut OpeningBook,
    stop: Arc<AtomicBool>,
    progress_cb: impl Fn(&EloProgress) + Send + Sync + 'static,
) -> EloReport {
    let start = Instant::now();
    let stats = Arc::new(std::sync::Mutex::new(MatchStats::default()));
    let pairs_target = config.pairs;

    let stats_clone = Arc::clone(&stats);
    let callback = Box::new(move |pair: &PairResult| {
        let Ok(mut s) = stats_clone.lock() else {
            return;
        };
        s.add_pair(pair);

        let elo = stats::elo_estimate_pentanomial(&s.pentanomial);
        let elapsed = start.elapsed();
        let games_per_min = if elapsed.as_secs() > 0 {
            s.total_games() as f64 / elapsed.as_secs_f64() * 60.0
        } else {
            0.0
        };

        let progress = EloProgress {
            stats: s.clone(),
            pairs_target,
            elo,
            elapsed,
            games_per_min,
            latest_pair: Some(pair.clone()),
        };
        progress_cb(&progress);
    });

    book.shuffle(&mut rand::thread_rng());

    let _results = manager::run_match(
        &config.match_config,
        MatchMode::FixedPairs(config.pairs),
        book,
        stop,
        callback,
    );

    let Ok(s) = stats.lock() else {
        return EloReport {
            stats: MatchStats::default(),
            elo: None,
            elapsed: start.elapsed(),
            games_per_min: 0.0,
        };
    };
    let elo = stats::elo_estimate_pentanomial(&s.pentanomial);
    let elapsed = start.elapsed();
    let games_per_min = if elapsed.as_secs() > 0 {
        s.total_games() as f64 / elapsed.as_secs_f64() * 60.0
    } else {
        0.0
    };

    EloReport {
        stats: s.clone(),
        elo,
        elapsed,
        games_per_min,
    }
}
