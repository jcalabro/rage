//! SPRT (Sequential Probability Ratio Test) orchestration.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::manager::{self, MatchMode};
use crate::opening::OpeningBook;
use crate::stats;
use crate::types::{MatchStats, PairResult, SprtConfig, SprtDecision};

/// Live progress snapshot for TUI rendering.
#[derive(Debug, Clone)]
pub struct SprtProgress {
    pub stats: MatchStats,
    pub llr: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub decision: SprtDecision,
    pub elo: Option<crate::stats::EloEstimate>,
    pub elapsed: Duration,
    pub games_per_min: f64,
    pub latest_pair: Option<PairResult>,
}

/// Final SPRT report.
#[derive(Debug, Clone)]
pub struct SprtReport {
    pub stats: MatchStats,
    pub llr: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub decision: SprtDecision,
    pub elo: Option<crate::stats::EloEstimate>,
    pub elapsed: Duration,
    pub games_per_min: f64,
}

/// Run an SPRT test.
///
/// `progress_cb` is called after each pair with the current progress.
/// Returns when the SPRT crosses a bound or `stop` is set externally.
pub fn run_sprt(
    config: &SprtConfig,
    book: &mut OpeningBook,
    stop: Arc<AtomicBool>,
    progress_cb: impl Fn(&SprtProgress) + Send + Sync + 'static,
) -> SprtReport {
    let (lower_bound, upper_bound) = stats::sprt_bounds(config.alpha, config.beta);
    let start = Instant::now();

    let stats = Arc::new(std::sync::Mutex::new(MatchStats::default()));
    let elo0 = config.elo0;
    let elo1 = config.elo1;
    let stop_clone = Arc::clone(&stop);

    let stats_clone = Arc::clone(&stats);
    let callback = Box::new(move |pair: &PairResult| {
        let Ok(mut s) = stats_clone.lock() else {
            return;
        };
        s.add_pair(pair);

        let llr = stats::sprt_llr(&s.pentanomial, elo0, elo1);
        let decision = stats::sprt_decision(llr, lower_bound, upper_bound, s.total_pairs());
        let elo = stats::elo_estimate_pentanomial(&s.pentanomial);
        let elapsed = start.elapsed();
        let games_per_min = if elapsed.as_secs() > 0 {
            s.total_games() as f64 / elapsed.as_secs_f64() * 60.0
        } else {
            0.0
        };

        let progress = SprtProgress {
            stats: s.clone(),
            llr,
            lower_bound,
            upper_bound,
            decision,
            elo,
            elapsed,
            games_per_min,
            latest_pair: Some(pair.clone()),
        };
        progress_cb(&progress);

        // Stop if SPRT has decided
        if decision != SprtDecision::InProgress {
            stop_clone.store(true, Ordering::Relaxed);
        }
    });

    // Shuffle openings
    book.shuffle(&mut rand::thread_rng());

    let _results = manager::run_match(
        &config.match_config,
        MatchMode::Unbounded,
        book,
        stop,
        callback,
    );

    // Final report
    let Ok(s) = stats.lock() else {
        return SprtReport {
            stats: MatchStats::default(),
            llr: 0.0,
            lower_bound,
            upper_bound,
            decision: SprtDecision::InProgress,
            elo: None,
            elapsed: start.elapsed(),
            games_per_min: 0.0,
        };
    };
    let llr = stats::sprt_llr(&s.pentanomial, elo0, elo1);
    let decision = stats::sprt_decision(llr, lower_bound, upper_bound, s.total_pairs());
    let elo = stats::elo_estimate_pentanomial(&s.pentanomial);
    let elapsed = start.elapsed();
    let games_per_min = if elapsed.as_secs() > 0 {
        s.total_games() as f64 / elapsed.as_secs_f64() * 60.0
    } else {
        0.0
    };

    SprtReport {
        stats: s.clone(),
        llr,
        lower_bound,
        upper_bound,
        decision,
        elo,
        elapsed,
        games_per_min,
    }
}
