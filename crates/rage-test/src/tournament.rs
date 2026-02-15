//! Tournament mode: round-robin between multiple engines.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::manager::{self, MatchMode};
use crate::opening::OpeningBook;
use crate::stats;
use crate::types::{MatchConfig, MatchStats, PairResult, TournamentConfig};

/// Results for a single matchup in the tournament.
#[derive(Debug, Clone)]
pub struct MatchupResult {
    pub engine_a_idx: usize,
    pub engine_b_idx: usize,
    pub stats: MatchStats,
    pub elo: Option<stats::EloEstimate>,
}

/// Live progress snapshot.
#[derive(Debug, Clone)]
pub struct TournamentProgress {
    pub matchups: Vec<MatchupResult>,
    pub current_matchup: Option<(usize, usize)>,
    pub elapsed: Duration,
}

/// Final tournament report.
#[derive(Debug, Clone)]
pub struct TournamentReport {
    pub engines: Vec<String>,
    pub matchups: Vec<MatchupResult>,
    pub rankings: Vec<(usize, f64)>,
    pub elapsed: Duration,
}

/// Run a round-robin tournament.
pub fn run_tournament(
    config: &TournamentConfig,
    book: &mut OpeningBook,
    stop: Arc<AtomicBool>,
    progress_cb: impl Fn(&TournamentProgress) + Send + Sync + 'static,
) -> TournamentReport {
    let start = Instant::now();
    let n = config.engines.len();
    let mut matchups: Vec<MatchupResult> = Vec::new();

    // Generate all pairs
    let mut pairs_to_play: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            pairs_to_play.push((i, j));
        }
    }

    for &(i, j) in &pairs_to_play {
        if stop.load(Ordering::Relaxed) {
            break;
        }

        let match_config = MatchConfig {
            engine_a: config.engines[i].clone(),
            engine_b: config.engines[j].clone(),
            tc: config.tc,
            search_overrides: config.search_overrides,
            concurrency: config.concurrency,
            opening_file: config.opening_file.clone(),
            move_overhead_ms: config.move_overhead_ms,
            max_moves: config.max_moves,
            win_adj_score: config.win_adj_score,
            win_adj_count: config.win_adj_count,
            draw_adj_score: config.draw_adj_score,
            draw_adj_count: config.draw_adj_count,
        };

        let stats = Arc::new(std::sync::Mutex::new(MatchStats::default()));
        let stats_clone = Arc::clone(&stats);

        let callback = Box::new(move |pair: &PairResult| {
            let Ok(mut s) = stats_clone.lock() else {
                return;
            };
            s.add_pair(pair);
        });

        book.shuffle(&mut rand::thread_rng());

        let match_stop = Arc::new(AtomicBool::new(false));
        // Mirror the global stop
        let match_stop_clone = Arc::clone(&match_stop);
        let stop_clone = Arc::clone(&stop);
        std::thread::spawn(move || {
            while !stop_clone.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(100));
            }
            match_stop_clone.store(true, Ordering::Relaxed);
        });

        let _results = manager::run_match(
            &match_config,
            MatchMode::FixedPairs(config.rounds),
            book,
            match_stop,
            callback,
            None,
        );

        let Ok(s) = stats.lock() else {
            continue;
        };
        let elo = stats::elo_estimate_pentanomial(&s.pentanomial);

        let matchup = MatchupResult {
            engine_a_idx: i,
            engine_b_idx: j,
            stats: s.clone(),
            elo,
        };
        matchups.push(matchup);

        let progress = TournamentProgress {
            matchups: matchups.clone(),
            current_matchup: Some((i, j)),
            elapsed: start.elapsed(),
        };
        progress_cb(&progress);
    }

    // Compute rankings: average score for each engine
    let mut engine_scores = vec![(0.0f64, 0u64); n];
    for m in &matchups {
        let total = m.stats.total_games();
        if total > 0 {
            let score_a = m.stats.score();
            engine_scores[m.engine_a_idx].0 += score_a * total as f64;
            engine_scores[m.engine_a_idx].1 += total;
            engine_scores[m.engine_b_idx].0 += (1.0 - score_a) * total as f64;
            engine_scores[m.engine_b_idx].1 += total;
        }
    }

    let mut rankings: Vec<(usize, f64)> = engine_scores
        .iter()
        .enumerate()
        .map(|(idx, (score_sum, total))| {
            let avg = if *total > 0 {
                score_sum / *total as f64
            } else {
                0.5
            };
            (idx, avg)
        })
        .collect();
    rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    TournamentReport {
        engines: config.engines.iter().map(|e| e.display_name()).collect(),
        matchups,
        rankings,
        elapsed: start.elapsed(),
    }
}
