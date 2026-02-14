use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use rage_core::Move;

use crate::eval::MATE_SCORE;

/// Bound type stored in TT entries.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Bound {
    None = 0,
    Exact = 1,
    Lower = 2, // beta cutoff (fail-high)
    Upper = 3, // fail-low
}

impl Bound {
    const fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Exact,
            2 => Self::Lower,
            3 => Self::Upper,
            _ => Self::None,
        }
    }
}

/// Unpacked transposition table entry.
#[derive(Clone, Copy, Debug)]
pub struct TTEntry {
    pub key16: u16,
    pub best_move: Move,
    pub score: i32,
    pub depth: u8,
    pub bound: Bound,
    pub age: u8,
}

// Packed layout in a single u64:
// Bits  0-15:  verification key (upper 16 bits of Zobrist)
// Bits 16-31:  best move (raw u16)
// Bits 32-47:  score (i16, sign-extended to i32 on read)
// Bits 48-55:  depth (u8)
// Bits 56-57:  bound type (2 bits)
// Bits 58-63:  age (6 bits, 0-63)

impl TTEntry {
    fn pack(self) -> u64 {
        let key = self.key16 as u64;
        let mv = self.best_move.raw() as u64;
        let score = (self.score as i16 as u16) as u64;
        let depth = self.depth as u64;
        let bound = (self.bound as u8 as u64) & 0x3;
        let age = (self.age as u64) & 0x3F;

        key | (mv << 16) | (score << 32) | (depth << 48) | (bound << 56) | (age << 58)
    }

    fn unpack(data: u64) -> Self {
        let key16 = data as u16;
        let best_move = Move::from_raw((data >> 16) as u16);
        let score = ((data >> 32) as u16 as i16) as i32;
        let depth = (data >> 48) as u8;
        let bound = Bound::from_u8(((data >> 56) & 0x3) as u8);
        let age = ((data >> 58) & 0x3F) as u8;

        Self {
            key16,
            best_move,
            score,
            depth,
            bound,
            age,
        }
    }
}

/// The threshold above which a score is considered a mate score.
const MATE_THRESHOLD: i32 = MATE_SCORE - 256;

/// Adjust a mate score for TT storage: convert ply-relative to root-relative.
fn score_to_tt(score: i32, ply: u32) -> i32 {
    if score > MATE_THRESHOLD {
        score + ply as i32
    } else if score < -MATE_THRESHOLD {
        score - ply as i32
    } else {
        score
    }
}

/// Adjust a mate score from TT: convert root-relative back to ply-relative.
fn score_from_tt(score: i32, ply: u32) -> i32 {
    if score > MATE_THRESHOLD {
        score - ply as i32
    } else if score < -MATE_THRESHOLD {
        score + ply as i32
    } else {
        score
    }
}

/// Verification key: upper 16 bits of the Zobrist hash.
fn verification_key(hash: u64) -> u16 {
    (hash >> 48) as u16
}

/// Lock-free transposition table using atomic operations.
///
/// All entries are stored as `AtomicU64` with `Relaxed` ordering.
/// Torn reads/writes are acceptable — the verification key rejects garbage.
/// This is `Send + Sync` without any `unsafe`.
pub struct TranspositionTable {
    entries: Vec<AtomicU64>,
    mask: usize,
    generation: AtomicU8,
}

impl TranspositionTable {
    /// Create a new TT with the given size in megabytes.
    pub fn new(size_mb: usize) -> Self {
        let size_mb = size_mb.max(1);
        let num_entries = (size_mb * 1024 * 1024) / std::mem::size_of::<u64>();
        // Round down to power of 2
        let num_entries = num_entries.next_power_of_two() >> 1;
        let num_entries = num_entries.max(1024);

        let mut entries = Vec::with_capacity(num_entries);
        for _ in 0..num_entries {
            entries.push(AtomicU64::new(0));
        }

        Self {
            entries,
            mask: num_entries - 1,
            generation: AtomicU8::new(0),
        }
    }

    /// Probe the TT for the given position hash.
    /// Returns Some(entry) if found and verification key matches.
    pub fn probe(&self, hash: u64, ply: u32) -> Option<TTEntry> {
        let index = hash as usize & self.mask;
        let data = self.entries[index].load(Ordering::Relaxed);

        if data == 0 {
            return None;
        }

        let entry = TTEntry::unpack(data);
        if entry.key16 != verification_key(hash) {
            return None;
        }

        Some(TTEntry {
            score: score_from_tt(entry.score, ply),
            ..entry
        })
    }

    /// Store an entry in the TT. Lock-free via atomic store.
    pub fn store(&self, hash: u64, best_move: Move, score: i32, depth: u8, bound: Bound, ply: u32) {
        let index = hash as usize & self.mask;
        let entry = TTEntry {
            key16: verification_key(hash),
            best_move,
            score: score_to_tt(score, ply),
            depth,
            bound,
            age: self.generation.load(Ordering::Relaxed),
        };
        self.entries[index].store(entry.pack(), Ordering::Relaxed);
    }

    /// Clear all entries.
    pub fn clear(&self) {
        for entry in &self.entries {
            entry.store(0, Ordering::Relaxed);
        }
        self.generation.store(0, Ordering::Relaxed);
    }

    /// Resize the table to the given size in megabytes. Clears all entries.
    /// Must only be called when no search is running.
    pub fn resize(&mut self, size_mb: usize) {
        *self = Self::new(size_mb);
    }

    /// Increment the generation counter (called at the start of each new search).
    pub fn new_search(&self) {
        let current = self.generation.load(Ordering::Relaxed);
        self.generation
            .store(current.wrapping_add(1) & 0x3F, Ordering::Relaxed);
    }

    /// Returns approximate usage as permille (0-1000).
    pub fn hashfull(&self) -> u32 {
        let sample = self.entries.len().min(1000);
        let current_gen = self.generation.load(Ordering::Relaxed);
        let mut used = 0u32;
        for i in 0..sample {
            let data = self.entries[i].load(Ordering::Relaxed);
            if data != 0 {
                let entry = TTEntry::unpack(data);
                if entry.age == current_gen {
                    used += 1;
                }
            }
        }
        used * 1000 / sample as u32
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use rage_core::Square;

    #[test]
    fn store_and_probe() {
        let tt = TranspositionTable::new(1);
        let hash = 0x123456789ABCDEF0;
        let mv = Move::new(Square::E2, Square::E4);

        tt.store(hash, mv, 150, 5, Bound::Exact, 0);

        let entry = tt.probe(hash, 0).unwrap();
        assert_eq!(entry.best_move, mv);
        assert_eq!(entry.score, 150);
        assert_eq!(entry.depth, 5);
        assert_eq!(entry.bound, Bound::Exact);
    }

    #[test]
    fn probe_miss() {
        let tt = TranspositionTable::new(1);
        assert!(tt.probe(0x123456789ABCDEF0, 0).is_none());
    }

    #[test]
    fn key_mismatch() {
        let tt = TranspositionTable::new(1);
        let hash1 = 0xAAAA_0000_0000_0001;
        let hash2 = 0xBBBB_0000_0000_0001; // same index, different key

        tt.store(hash1, Move::NULL, 100, 3, Bound::Exact, 0);

        // Probe with different key at same index
        assert!(tt.probe(hash2, 0).is_none());
    }

    #[test]
    fn mate_score_adjustment() {
        let tt = TranspositionTable::new(1);
        let hash = 0x123456789ABCDEF0;

        // Store a "mate in 5" from ply 3 (score = MATE_SCORE - 5)
        let mate_score = MATE_SCORE - 5;
        tt.store(hash, Move::NULL, mate_score, 10, Bound::Exact, 3);

        // Probe at ply 7 - should adjust the mate distance
        let entry = tt.probe(hash, 7).unwrap();
        // Stored: MATE_SCORE - 5 + 3 = MATE_SCORE - 2 (root-relative)
        // Retrieved at ply 7: MATE_SCORE - 2 - 7 = MATE_SCORE - 9
        assert_eq!(entry.score, MATE_SCORE - 9);
    }

    #[test]
    fn negative_mate_score_adjustment() {
        let tt = TranspositionTable::new(1);
        let hash = 0x123456789ABCDEF0;

        // "Mated in 3" from ply 2
        let mated_score = -MATE_SCORE + 3;
        tt.store(hash, Move::NULL, mated_score, 10, Bound::Exact, 2);

        // Probe at ply 5
        let entry = tt.probe(hash, 5).unwrap();
        // Stored: -MATE_SCORE + 3 - 2 = -MATE_SCORE + 1 (root-relative)
        // Retrieved at ply 5: -MATE_SCORE + 1 + 5 = -MATE_SCORE + 6
        assert_eq!(entry.score, -MATE_SCORE + 6);
    }

    #[test]
    fn clear_empties_table() {
        let tt = TranspositionTable::new(1);
        let hash = 0x123456789ABCDEF0;
        tt.store(hash, Move::NULL, 100, 3, Bound::Exact, 0);
        assert!(tt.probe(hash, 0).is_some());

        tt.clear();
        assert!(tt.probe(hash, 0).is_none());
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let entry = TTEntry {
            key16: 0xABCD,
            best_move: Move::new(Square::E2, Square::E4),
            score: -150,
            depth: 12,
            bound: Bound::Upper,
            age: 7,
        };

        let packed = entry.pack();
        let unpacked = TTEntry::unpack(packed);

        assert_eq!(unpacked.key16, entry.key16);
        assert_eq!(unpacked.best_move, entry.best_move);
        assert_eq!(unpacked.score, entry.score);
        assert_eq!(unpacked.depth, entry.depth);
        assert_eq!(unpacked.bound, entry.bound);
        assert_eq!(unpacked.age, entry.age);
    }

    #[test]
    fn score_i16_range() {
        // Test that negative scores survive pack/unpack
        let entry = TTEntry {
            key16: 0,
            best_move: Move::NULL,
            score: -32000,
            depth: 0,
            bound: Bound::None,
            age: 0,
        };
        let packed = entry.pack();
        let unpacked = TTEntry::unpack(packed);
        assert_eq!(unpacked.score, -32000);
    }

    /// Stress test: multiple threads concurrently reading and writing the TT.
    /// Verifies no panics/crashes and that probed entries are internally consistent
    /// (the verification key mechanism rejects torn reads).
    #[test]
    fn concurrent_store_and_probe() {
        use std::sync::Arc;
        use std::thread;

        let tt = Arc::new(TranspositionTable::new(1));
        let num_threads = 4;
        let ops_per_thread = 10_000;

        let handles: Vec<_> = (0..num_threads)
            .map(|tid| {
                let tt = Arc::clone(&tt);
                thread::spawn(move || {
                    let mut valid_probes = 0u64;
                    let mut misses = 0u64;

                    for i in 0..ops_per_thread {
                        // Each thread writes entries with distinct hashes based on tid
                        let hash = (tid as u64 * 1_000_000 + i as u64) | ((tid as u64) << 48);
                        let score = (i % 1000) as i32 - 500;
                        let depth = (i % 30) as u8;
                        let mv = Move::new(Square::A1, Square::new((i % 64) as u8));
                        let bound = match i % 3 {
                            0 => Bound::Exact,
                            1 => Bound::Lower,
                            _ => Bound::Upper,
                        };

                        tt.store(hash, mv, score, depth, bound, 0);

                        // Immediately try to probe back - may miss due to
                        // overwrites from other threads, but any hit must be valid
                        if let Some(entry) = tt.probe(hash, 0) {
                            // If verification key matches, the data should be consistent:
                            // the entry we get back was a valid packed entry (not garbage)
                            assert!(
                                entry.depth <= 30,
                                "depth should be in valid range, got {}",
                                entry.depth
                            );
                            assert!(
                                entry.score >= -32768 && entry.score <= 32767,
                                "score should be in i16 range"
                            );
                            valid_probes += 1;
                        } else {
                            misses += 1;
                        }
                    }
                    (valid_probes, misses)
                })
            })
            .collect();

        let mut total_valid = 0u64;
        for handle in handles {
            let (valid, _misses) = handle.join().unwrap();
            total_valid += valid;
        }

        // At least some probes should succeed (not all will due to hash collisions)
        assert!(
            total_valid > 0,
            "at least some concurrent probes should succeed"
        );
    }
}
