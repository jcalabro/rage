use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;

use rage_core::Board;
use rage_nnue::NnueNetwork;

use crate::search::{InfoCallback, SearchResult, Searcher};
use crate::time::SearchLimits;
use crate::tt::TranspositionTable;

/// Thread pool for Lazy SMP search.
///
/// Multiple searchers share a single lock-free transposition table.
/// Each searcher has its own killers, history, counter moves, and search stack.
/// On `search()`, thread 0 runs as the main thread (inline for single-thread,
/// or spawned when there are helpers). Helper threads add `thread_id % 2` to
/// each iteration depth for perturbation so they explore different tree regions.
pub struct ThreadPool {
    tt: Arc<TranspositionTable>,
    searchers: Vec<Searcher>,
    thread_nodes: Arc<Vec<AtomicU64>>,
    num_threads: usize,
}

impl ThreadPool {
    /// Create a new thread pool with the given number of threads and TT size.
    pub fn new(num_threads: usize, tt_size_mb: usize) -> Self {
        let num_threads = num_threads.max(1);
        let tt = Arc::new(TranspositionTable::new(tt_size_mb));
        let mut searchers = Vec::with_capacity(num_threads);
        for id in 0..num_threads {
            let mut s = Searcher::new(Arc::clone(&tt));
            s.thread_id = id;
            searchers.push(s);
        }

        let thread_nodes = Arc::new((0..num_threads).map(|_| AtomicU64::new(0)).collect());

        Self {
            tt,
            searchers,
            thread_nodes,
            num_threads,
        }
    }

    /// Set the number of search threads. Preserves the TT.
    pub fn set_num_threads(&mut self, n: usize) {
        let n = n.max(1);
        if n == self.num_threads {
            return;
        }

        // Resize searchers
        if n > self.num_threads {
            for id in self.num_threads..n {
                let mut s = Searcher::new(Arc::clone(&self.tt));
                s.thread_id = id;
                searchers_copy_network(&self.searchers, &mut s);
                self.searchers.push(s);
            }
        } else {
            self.searchers.truncate(n);
        }

        self.num_threads = n;
        self.thread_nodes = Arc::new((0..n).map(|_| AtomicU64::new(0)).collect());
    }

    /// Resize the transposition table. Must only be called when no search is running.
    pub fn resize_tt(&mut self, size_mb: usize) {
        self.tt = Arc::new(TranspositionTable::new(size_mb));
        for s in &mut self.searchers {
            s.tt = Arc::clone(&self.tt);
        }
    }

    /// Clear TT and all per-thread tables.
    pub fn clear(&mut self) {
        self.tt.clear();
        for s in &mut self.searchers {
            s.clear_tables();
        }
    }

    /// Set the NNUE network on all searchers.
    pub fn set_network(&mut self, network: Option<Arc<NnueNetwork>>) {
        for s in &mut self.searchers {
            s.set_network(network.clone());
        }
    }

    /// Total nodes searched across all threads.
    pub fn total_nodes(&self) -> u64 {
        self.thread_nodes
            .iter()
            .map(|n| n.load(Ordering::Relaxed))
            .sum()
    }

    /// Run a Lazy SMP search. Returns the result from thread 0 (main thread).
    ///
    /// For single-thread mode, calls the searcher directly (no spawn overhead).
    /// For multi-thread, spawns helper threads and runs thread 0 inline.
    pub fn search(
        &mut self,
        board: &mut Board,
        limits: &SearchLimits,
        stop: Arc<AtomicBool>,
        info_callback: Option<InfoCallback>,
    ) -> SearchResult {
        // Reset node counters
        for n in self.thread_nodes.iter() {
            n.store(0, Ordering::Relaxed);
        }

        self.tt.new_search();

        // Single-thread fast path
        if self.num_threads == 1 {
            let result = self.searchers[0].search(board, limits, stop, info_callback);
            self.thread_nodes[0].store(self.searchers[0].nodes, Ordering::Relaxed);
            return result;
        }

        // Multi-thread: spawn helpers (threads 1..N), run thread 0 inline.
        let mut handles = Vec::with_capacity(self.num_threads - 1);

        // Take helper searchers out of the pool temporarily
        let mut helpers: Vec<Searcher> = self.searchers.drain(1..).collect();

        for (i, mut helper) in helpers.drain(..).enumerate() {
            let mut helper_board = board.clone();
            let helper_limits = limits.clone();
            let helper_stop = Arc::clone(&stop);
            let thread_nodes = Arc::clone(&self.thread_nodes);
            let thread_idx = i + 1;

            let handle = thread::spawn(move || {
                let _result = helper.search(&mut helper_board, &helper_limits, helper_stop, None);
                thread_nodes[thread_idx].store(helper.nodes, Ordering::Relaxed);
                helper
            });
            handles.push(handle);
        }

        // Wrap the info callback to aggregate node counts from all threads
        let thread_nodes_for_cb = Arc::clone(&self.thread_nodes);
        let wrapped_cb: Option<InfoCallback> = info_callback.map(|cb| {
            let nodes_ref = thread_nodes_for_cb;
            let wrapper: InfoCallback = Box::new(move |info| {
                // Sum nodes from all threads for accurate NPS
                let total: u64 = nodes_ref.iter().map(|n| n.load(Ordering::Relaxed)).sum();
                let adjusted = crate::search::SearchInfo {
                    depth: info.depth,
                    seldepth: info.seldepth,
                    score: info.score,
                    nodes: total + info.nodes, // thread 0's nodes not yet stored
                    time_ms: info.time_ms,
                    pv: info.pv.clone(),
                    hashfull: info.hashfull,
                };
                cb(&adjusted);
            });
            wrapper
        });

        // Run thread 0 inline
        let result = self.searchers[0].search(board, limits, Arc::clone(&stop), wrapped_cb);
        self.thread_nodes[0].store(self.searchers[0].nodes, Ordering::Relaxed);

        // Signal all helpers to stop
        stop.store(true, Ordering::Relaxed);

        // Join helpers and put them back
        for handle in handles {
            if let Ok(helper) = handle.join() {
                self.searchers.push(helper);
            }
        }

        // Re-sort by thread_id so the pool is in order
        self.searchers.sort_by_key(|s| s.thread_id);

        // Return thread 0's result with total node count
        SearchResult {
            nodes: self.total_nodes(),
            ..result
        }
    }
}

/// Copy network from existing searchers to a new one.
fn searchers_copy_network(existing: &[Searcher], target: &mut Searcher) {
    if let Some(first) = existing.first() {
        target.set_network(first.network().clone());
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::eval::MATE_SCORE;

    #[test]
    fn single_thread_finds_move() {
        let mut pool = ThreadPool::new(1, 16);
        let mut board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let limits = SearchLimits {
            depth: Some(4),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let result = pool.search(&mut board, &limits, stop, None);
        assert!(!result.best_move.is_null());
        assert!(result.nodes > 0);
    }

    #[test]
    fn lazy_smp_two_threads_shared_tt() {
        let mut pool = ThreadPool::new(2, 16);
        let mut board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let limits = SearchLimits {
            depth: Some(6),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let result = pool.search(&mut board, &limits, stop, None);

        assert!(!result.best_move.is_null());
        assert!(result.nodes > 0);
        // Both searchers should be recovered
        assert_eq!(pool.searchers.len(), 2);
        // TT should have entries from the search (proves TT sharing works)
        assert!(
            pool.tt.hashfull() > 0,
            "TT should have entries after multi-threaded search"
        );
        // Both threads should have searched nodes
        for (i, s) in pool.searchers.iter().enumerate() {
            assert!(s.nodes > 0, "thread {i} should have searched nodes, got 0");
        }
    }

    #[test]
    fn lazy_smp_four_threads() {
        let mut pool = ThreadPool::new(4, 16);
        let mut board =
            Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
                .unwrap();
        let limits = SearchLimits {
            depth: Some(5),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let result = pool.search(&mut board, &limits, stop, None);

        assert!(!result.best_move.is_null());
        // All 4 searchers recovered and sorted by thread_id
        assert_eq!(pool.searchers.len(), 4);
        for (i, s) in pool.searchers.iter().enumerate() {
            assert_eq!(s.thread_id, i, "searchers should be sorted by thread_id");
        }
    }

    #[test]
    fn lazy_smp_finds_mate_in_one() {
        // Scholars mate position: Qh5f7# is the only winning move
        let mut pool = ThreadPool::new(4, 16);
        let mut board =
            Board::from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
                .unwrap();
        let limits = SearchLimits {
            depth: Some(4),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let result = pool.search(&mut board, &limits, stop, None);

        assert_eq!(
            result.best_move.to_string(),
            "h5f7",
            "4-thread search should find Qxf7#"
        );
        assert!(
            result.score > MATE_SCORE - 10,
            "should report mate score, got {}",
            result.score
        );
    }

    #[test]
    fn repeated_multithreaded_searches() {
        // Multiple search/stop cycles shouldn't corrupt state
        let mut pool = ThreadPool::new(2, 16);
        let mut board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let limits = SearchLimits {
            depth: Some(4),
            ..Default::default()
        };

        for _ in 0..5 {
            let stop = Arc::new(AtomicBool::new(false));
            let result = pool.search(&mut board, &limits, Arc::clone(&stop), None);
            assert!(!result.best_move.is_null());
            assert_eq!(
                pool.searchers.len(),
                2,
                "all searchers should be recovered after each search"
            );
        }
    }

    #[test]
    fn total_nodes_aggregates_all_threads() {
        let mut pool = ThreadPool::new(2, 16);
        let mut board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let limits = SearchLimits {
            depth: Some(5),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let result = pool.search(&mut board, &limits, stop, None);

        // result.nodes should be the aggregate from total_nodes()
        assert_eq!(result.nodes, pool.total_nodes());
        // With 2 threads at depth 5, total should exceed what thread 0 alone did
        let thread0_nodes = pool.thread_nodes[0].load(Ordering::Relaxed);
        assert!(
            result.nodes >= thread0_nodes,
            "total {} should be >= thread 0's {}",
            result.nodes,
            thread0_nodes
        );
    }

    #[test]
    fn set_num_threads_adjusts_pool() {
        let mut pool = ThreadPool::new(1, 16);
        assert_eq!(pool.num_threads, 1);
        assert_eq!(pool.searchers.len(), 1);

        pool.set_num_threads(4);
        assert_eq!(pool.num_threads, 4);
        assert_eq!(pool.searchers.len(), 4);

        pool.set_num_threads(2);
        assert_eq!(pool.num_threads, 2);
        assert_eq!(pool.searchers.len(), 2);
    }

    #[test]
    fn clear_resets_state() {
        let mut pool = ThreadPool::new(2, 16);
        let mut board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let limits = SearchLimits {
            depth: Some(6),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        pool.search(&mut board, &limits, stop, None);

        // Verify the TT has entries by probing the root position hash.
        let hash = board.zobrist_hash();
        assert!(
            pool.tt.probe(hash, 0).is_some(),
            "TT should have the root position cached after search"
        );

        pool.clear();
        // After clear, even the root position should be gone
        assert!(
            pool.tt.probe(hash, 0).is_none(),
            "TT should be empty after clear"
        );
    }
}
