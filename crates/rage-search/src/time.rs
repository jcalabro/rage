use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use rage_core::Color;

/// Search limits from the UCI `go` command.
#[derive(Clone, Debug, Default)]
pub struct SearchLimits {
    pub depth: Option<u32>,
    pub movetime: Option<u64>,
    pub wtime: Option<u64>,
    pub btime: Option<u64>,
    pub winc: Option<u64>,
    pub binc: Option<u64>,
    pub movestogo: Option<u32>,
    pub infinite: bool,
    pub nodes: Option<u64>,
}

/// Time manager: determines how long the engine can search.
pub struct TimeManager {
    start: Instant,
    soft_limit_ms: u64,
    hard_limit_ms: u64,
    max_depth: u32,
    max_nodes: u64,
    stop: Arc<AtomicBool>,
    node_check_counter: u64,
}

/// Safety margin subtracted from allocated time to account for overhead.
const SAFETY_MARGIN_MS: u64 = 50;

/// Check the clock every this many nodes.
const NODE_CHECK_INTERVAL: u64 = 2048;

impl TimeManager {
    /// Create a time manager from search limits.
    pub fn new(limits: &SearchLimits, stm: Color, stop: Arc<AtomicBool>) -> Self {
        let (soft_limit_ms, hard_limit_ms) = Self::compute_limits(limits, stm);

        Self {
            start: Instant::now(),
            soft_limit_ms,
            hard_limit_ms,
            max_depth: limits.depth.unwrap_or(128),
            max_nodes: limits.nodes.unwrap_or(u64::MAX),
            stop,
            node_check_counter: 0,
        }
    }

    fn compute_limits(limits: &SearchLimits, stm: Color) -> (u64, u64) {
        if limits.infinite {
            return (u64::MAX, u64::MAX);
        }

        if let Some(movetime) = limits.movetime {
            let hard = movetime.saturating_sub(SAFETY_MARGIN_MS);
            return (hard, hard);
        }

        // Clock-based time allocation
        let (our_time, our_inc) = match stm {
            Color::White => (limits.wtime, limits.winc),
            Color::Black => (limits.btime, limits.binc),
        };

        if let Some(time_ms) = our_time {
            let inc = our_inc.unwrap_or(0);
            let moves_to_go = limits.movestogo.unwrap_or(25) as u64;
            let moves_to_go = moves_to_go.max(1);

            let allocated = time_ms / moves_to_go + inc;
            // Don't use more than 75% of remaining time
            let allocated = allocated.min(time_ms * 3 / 4);

            let soft = (allocated * 6 / 10).saturating_sub(SAFETY_MARGIN_MS);
            let hard = allocated.saturating_sub(SAFETY_MARGIN_MS);

            return (soft.max(1), hard.max(1));
        }

        // Depth-only or node-only limit
        (u64::MAX, u64::MAX)
    }

    /// Returns true if the engine should stop searching (time exceeded or stop requested).
    /// Only does the expensive clock check every NODE_CHECK_INTERVAL nodes.
    pub fn should_stop(&mut self, nodes: u64) -> bool {
        if self.stop.load(Ordering::Relaxed) {
            return true;
        }

        if nodes >= self.max_nodes {
            return true;
        }

        self.node_check_counter += 1;
        if self.node_check_counter >= NODE_CHECK_INTERVAL {
            self.node_check_counter = 0;
            if self.elapsed_ms() >= self.hard_limit_ms {
                return true;
            }
        }

        false
    }

    /// Returns true if the soft time limit has been exceeded (don't start new iteration).
    pub fn soft_limit_reached(&self) -> bool {
        if self.soft_limit_ms == u64::MAX {
            return false;
        }
        self.elapsed_ms() >= self.soft_limit_ms
    }

    /// Returns true if the given percentage of the soft limit has been reached.
    /// `pct` is 0-100. Used for best-move stability: stable moves use less time.
    pub fn soft_limit_reached_fraction(&self, pct: u64) -> bool {
        if self.soft_limit_ms == u64::MAX {
            return false;
        }
        self.elapsed_ms() >= self.soft_limit_ms * pct / 100
    }

    /// Maximum search depth.
    pub fn max_depth(&self) -> u32 {
        self.max_depth
    }

    /// Elapsed time since search started, in milliseconds.
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn infinite_limits() {
        let limits = SearchLimits {
            infinite: true,
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let tm = TimeManager::new(&limits, Color::White, stop);
        assert!(!tm.soft_limit_reached());
        assert_eq!(tm.max_depth(), 128);
    }

    #[test]
    fn depth_limit() {
        let limits = SearchLimits {
            depth: Some(6),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let tm = TimeManager::new(&limits, Color::White, stop);
        assert_eq!(tm.max_depth(), 6);
    }

    #[test]
    fn movetime_limit() {
        let limits = SearchLimits {
            movetime: Some(1000),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let tm = TimeManager::new(&limits, Color::White, stop);
        assert_eq!(tm.hard_limit_ms, 950);
        assert_eq!(tm.soft_limit_ms, 950);
    }

    #[test]
    fn clock_time_allocation() {
        let limits = SearchLimits {
            wtime: Some(60_000),
            winc: Some(1_000),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let tm = TimeManager::new(&limits, Color::White, stop);
        // 60000/25 + 1000 = 3400, soft = 3400*0.6 - 50 = 1990, hard = 3400 - 50 = 3350
        assert!(tm.soft_limit_ms > 0);
        assert!(tm.hard_limit_ms > tm.soft_limit_ms);
    }

    #[test]
    fn external_stop() {
        let limits = SearchLimits {
            infinite: true,
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let mut tm = TimeManager::new(&limits, Color::White, Arc::clone(&stop));
        assert!(!tm.should_stop(0));

        stop.store(true, Ordering::Relaxed);
        assert!(tm.should_stop(0));
    }
}
