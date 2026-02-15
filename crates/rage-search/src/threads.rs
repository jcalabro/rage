use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::thread;

use rage_core::Board;
use rage_nnue::NnueNetwork;
use rage_syzygy::Tablebase;

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
                searchers_copy_settings(&self.searchers, &mut s);
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

    /// Get the NNUE network from the first searcher (if any).
    pub fn network(&self) -> Option<&Arc<NnueNetwork>> {
        self.searchers.first().and_then(|s| s.network().as_ref())
    }

    /// Set the Syzygy tablebase on all searchers.
    pub fn set_syzygy(&mut self, tb: Option<Arc<Tablebase>>) {
        for s in &mut self.searchers {
            s.set_syzygy(tb.clone());
        }
    }

    /// Get the Syzygy tablebase from the first searcher (if any).
    pub fn syzygy(&self) -> Option<&Arc<Tablebase>> {
        self.searchers.first().and_then(|s| s.syzygy().as_ref())
    }

    /// Set the number of principal variations to report (Multi-PV).
    pub fn set_multi_pv(&mut self, n: usize) {
        for s in &mut self.searchers {
            s.multi_pv = n;
        }
    }

    /// Evaluate the current position with NNUE (if a network is loaded).
    /// Returns `None` if no network is loaded.
    pub fn evaluate_nnue(&mut self, board: &Board) -> Option<i32> {
        let searcher = self.searchers.first_mut()?;
        searcher.evaluate_nnue(board)
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

        // Single-thread fast path (always used on wasm)
        if self.num_threads == 1 {
            let result = self.searchers[0].search(board, limits, stop, info_callback);
            self.thread_nodes[0].store(self.searchers[0].nodes, Ordering::Relaxed);
            return result;
        }

        #[cfg(target_arch = "wasm32")]
        unreachable!();

        #[cfg(not(target_arch = "wasm32"))]
        {
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
                    let _result =
                        helper.search(&mut helper_board, &helper_limits, helper_stop, None);
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
                        tbhits: info.tbhits,
                        multipv: info.multipv,
                        wdl: info.wdl,
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
}

/// Copy network and syzygy settings from existing searchers to a new one.
fn searchers_copy_settings(existing: &[Searcher], target: &mut Searcher) {
    if let Some(first) = existing.first() {
        target.set_network(first.network().clone());
        target.set_syzygy(first.syzygy().clone());
        target.multi_pv = first.multi_pv;
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
    #[cfg_attr(target_arch = "wasm32", ignore)]
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
    #[cfg_attr(target_arch = "wasm32", ignore)]
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
    #[cfg_attr(target_arch = "wasm32", ignore)]
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
    #[cfg_attr(target_arch = "wasm32", ignore)]
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
    #[cfg_attr(target_arch = "wasm32", ignore)]
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
    #[cfg_attr(target_arch = "wasm32", ignore)]
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

    fn make_test_network() -> Arc<NnueNetwork> {
        let weights = rage_nnue::NnueWeights::random(42);
        Arc::new(NnueNetwork::new(weights))
    }

    #[test]
    fn network_returns_none_by_default() {
        let pool = ThreadPool::new(1, 16);
        assert!(pool.network().is_none(), "new pool should have no network");
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", ignore)]
    fn set_network_propagates_to_all_threads() {
        let mut pool = ThreadPool::new(4, 16);
        let network = make_test_network();
        pool.set_network(Some(network));

        assert!(pool.network().is_some(), "pool should report network");
        for (i, s) in pool.searchers.iter().enumerate() {
            assert!(s.network().is_some(), "thread {i} should have the network");
        }
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", ignore)]
    fn set_network_none_clears_all_threads() {
        let mut pool = ThreadPool::new(2, 16);
        pool.set_network(Some(make_test_network()));
        assert!(pool.network().is_some());

        pool.set_network(None);
        assert!(pool.network().is_none(), "should be cleared");
        for (i, s) in pool.searchers.iter().enumerate() {
            assert!(
                s.network().is_none(),
                "thread {i} should have no network after clear"
            );
        }
    }

    #[test]
    fn evaluate_nnue_without_network() {
        let mut pool = ThreadPool::new(1, 16);
        let board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        assert!(
            pool.evaluate_nnue(&board).is_none(),
            "should return None without network"
        );
    }

    #[test]
    fn evaluate_nnue_with_network() {
        let mut pool = ThreadPool::new(1, 16);
        pool.set_network(Some(make_test_network()));

        let board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        assert!(
            pool.evaluate_nnue(&board).is_some(),
            "should return a score with network loaded"
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", ignore)]
    fn network_survives_set_num_threads() {
        let mut pool = ThreadPool::new(1, 16);
        pool.set_network(Some(make_test_network()));

        // Scale up — new threads should inherit the network
        pool.set_num_threads(4);
        assert!(pool.network().is_some(), "network should survive resize up");
        for (i, s) in pool.searchers.iter().enumerate() {
            assert!(
                s.network().is_some(),
                "thread {i} should have network after resize up"
            );
        }

        // Scale down — remaining threads should keep the network
        pool.set_num_threads(2);
        assert!(
            pool.network().is_some(),
            "network should survive resize down"
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", ignore)]
    fn network_survives_clear() {
        let mut pool = ThreadPool::new(2, 16);
        pool.set_network(Some(make_test_network()));

        // Search to populate TT and history
        let mut board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let limits = SearchLimits {
            depth: Some(4),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        pool.search(&mut board, &limits, stop, None);

        pool.clear();
        // Network should still be set after clearing TT and history
        assert!(
            pool.network().is_some(),
            "clear() should not drop the network"
        );
        for (i, s) in pool.searchers.iter().enumerate() {
            assert!(
                s.network().is_some(),
                "thread {i} should still have network after clear"
            );
        }
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", ignore)]
    fn search_with_nnue_network() {
        let mut pool = ThreadPool::new(2, 16);
        pool.set_network(Some(make_test_network()));

        let mut board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let limits = SearchLimits {
            depth: Some(4),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let result = pool.search(&mut board, &limits, stop, None);
        assert!(
            !result.best_move.is_null(),
            "NNUE search should find a move"
        );
        assert!(result.nodes > 0);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", ignore)]
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

    #[test]
    fn multi_pv_through_thread_pool() {
        let mut pool = ThreadPool::new(1, 16);
        pool.set_multi_pv(3);

        let mut board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let limits = SearchLimits {
            depth: Some(5),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));

        let infos = Arc::new(std::sync::Mutex::new(Vec::new()));
        let infos_clone = Arc::clone(&infos);
        let cb: crate::search::InfoCallback = Box::new(move |info| {
            infos_clone.lock().unwrap().push((
                info.depth,
                info.multipv,
                info.pv.first().map(|m| m.to_string()).unwrap_or_default(),
            ));
        });

        let result = pool.search(&mut board, &limits, stop, Some(cb));
        assert!(!result.best_move.is_null());

        let infos = infos.lock().unwrap();
        let max_depth = infos.iter().map(|(d, _, _)| *d).max().unwrap_or(0);
        assert!(max_depth >= 5, "should complete at least depth 5");
        for depth in 1..=max_depth {
            let lines: Vec<_> = infos.iter().filter(|(d, _, _)| *d == depth).collect();
            assert_eq!(
                lines.len(),
                3,
                "depth {depth}: should get 3 PV lines via ThreadPool, got {}",
                lines.len()
            );
            assert_eq!(lines[0].1, 1);
            assert_eq!(lines[1].1, 2);
            assert_eq!(lines[2].1, 3);
        }
    }
}
