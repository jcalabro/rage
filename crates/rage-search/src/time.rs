use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use web_time::Instant;

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
    /// When true, the engine is pondering (searching on opponent's time).
    /// Time limits are ignored until `ponderhit` flips this to false.
    pub ponder: Option<Arc<AtomicBool>>,
    /// Offset in milliseconds to subtract from elapsed time after `ponderhit`.
    /// Set by the UCI layer when `ponderhit` is received so that time management
    /// starts from the moment the opponent moved, not from when pondering began.
    pub ponder_offset: Option<Arc<AtomicU64>>,
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
    /// While true, time/node limits are ignored (only external stop is checked).
    ponder: Option<Arc<AtomicBool>>,
    /// Milliseconds to subtract from elapsed time after ponderhit.
    ponder_offset: Option<Arc<AtomicU64>>,
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
            ponder: limits.ponder.clone(),
            ponder_offset: limits.ponder_offset.clone(),
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
    /// While pondering, only the external stop flag is checked.
    pub fn should_stop(&mut self, nodes: u64) -> bool {
        if self.stop.load(Ordering::Relaxed) {
            return true;
        }

        // While pondering, ignore time and node limits
        if self.is_pondering() {
            return false;
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
    /// Always returns false while pondering.
    pub fn soft_limit_reached(&self) -> bool {
        if self.is_pondering() || self.soft_limit_ms == u64::MAX {
            return false;
        }
        self.elapsed_ms() >= self.soft_limit_ms
    }

    /// Returns true if the given percentage of the soft limit has been reached.
    /// `pct` is 0-100. Used for best-move stability: stable moves use less time.
    /// Always returns false while pondering.
    pub fn soft_limit_reached_fraction(&self, pct: u64) -> bool {
        if self.is_pondering() || self.soft_limit_ms == u64::MAX {
            return false;
        }
        self.elapsed_ms() >= self.soft_limit_ms * pct / 100
    }

    /// Maximum search depth.
    pub fn max_depth(&self) -> u32 {
        self.max_depth
    }

    /// Returns true if the engine is currently pondering (searching on opponent's time).
    fn is_pondering(&self) -> bool {
        self.ponder
            .as_ref()
            .is_some_and(|p| p.load(Ordering::Relaxed))
    }

    /// Elapsed time since search started, in milliseconds.
    /// After ponderhit, subtracts the ponder offset so that time management
    /// starts from the moment the opponent moved.
    pub fn elapsed_ms(&self) -> u64 {
        let raw = self.start.elapsed().as_millis() as u64;
        let offset = self
            .ponder_offset
            .as_ref()
            .map_or(0, |o| o.load(Ordering::Relaxed));
        raw.saturating_sub(offset)
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

    #[test]
    fn ponder_ignores_time_limits() {
        let ponder_flag = Arc::new(AtomicBool::new(true));
        let ponder_offset = Arc::new(AtomicU64::new(0));
        let limits = SearchLimits {
            wtime: Some(1), // 1ms — would immediately expire without pondering
            ponder: Some(Arc::clone(&ponder_flag)),
            ponder_offset: Some(Arc::clone(&ponder_offset)),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let mut tm = TimeManager::new(&limits, Color::White, stop);

        // Even though wtime is tiny, pondering suppresses time checks
        assert!(!tm.should_stop(0));
        assert!(!tm.soft_limit_reached());
        assert!(!tm.soft_limit_reached_fraction(50));
    }

    #[test]
    fn ponderhit_enables_time_management() {
        let ponder_flag = Arc::new(AtomicBool::new(true));
        let ponder_offset = Arc::new(AtomicU64::new(0));
        let limits = SearchLimits {
            movetime: Some(100_000), // 100 seconds — won't expire during test
            ponder: Some(Arc::clone(&ponder_flag)),
            ponder_offset: Some(Arc::clone(&ponder_offset)),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let tm = TimeManager::new(&limits, Color::White, stop);

        // While pondering: soft limit not reached
        assert!(!tm.soft_limit_reached());

        // Simulate ponderhit: flip flag, set offset to elapsed time
        let elapsed = tm.start.elapsed().as_millis() as u64;
        ponder_offset.store(elapsed, Ordering::Relaxed);
        ponder_flag.store(false, Ordering::Relaxed);

        // After ponderhit: time management is active but we have plenty of time
        assert!(!tm.soft_limit_reached());
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", ignore)]
    fn ponder_offset_adjusts_elapsed() {
        let ponder_offset = Arc::new(AtomicU64::new(0));
        let limits = SearchLimits {
            ponder_offset: Some(Arc::clone(&ponder_offset)),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let tm = TimeManager::new(&limits, Color::White, stop);

        // Sleep briefly so elapsed_ms > 0
        std::thread::sleep(std::time::Duration::from_millis(10));
        let raw_elapsed = tm.start.elapsed().as_millis() as u64;
        assert!(raw_elapsed >= 10);

        // Set offset to just under raw elapsed — elapsed_ms should be near zero
        ponder_offset.store(raw_elapsed, Ordering::Relaxed);
        let adjusted = tm.elapsed_ms();
        assert!(
            adjusted < 5,
            "adjusted elapsed should be near zero, got {adjusted}"
        );
    }

    #[test]
    fn ponder_stop_still_works() {
        let ponder_flag = Arc::new(AtomicBool::new(true));
        let limits = SearchLimits {
            ponder: Some(Arc::clone(&ponder_flag)),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let mut tm = TimeManager::new(&limits, Color::White, Arc::clone(&stop));

        // While pondering, external stop still works
        assert!(!tm.should_stop(0));
        stop.store(true, Ordering::Relaxed);
        assert!(tm.should_stop(0));
    }
}
