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

/// Transposition table with always-replace policy.
pub struct TranspositionTable {
    entries: Vec<u64>,
    mask: usize,
    generation: u8,
}

impl TranspositionTable {
    /// Create a new TT with the given size in megabytes.
    pub fn new(size_mb: usize) -> Self {
        let size_mb = size_mb.max(1);
        let num_entries = (size_mb * 1024 * 1024) / std::mem::size_of::<u64>();
        // Round down to power of 2
        let num_entries = num_entries.next_power_of_two() >> 1;
        let num_entries = num_entries.max(1024);

        Self {
            entries: vec![0u64; num_entries],
            mask: num_entries - 1,
            generation: 0,
        }
    }

    /// Probe the TT for the given position hash.
    /// Returns Some(entry) if found and verification key matches.
    pub fn probe(&self, hash: u64, ply: u32) -> Option<TTEntry> {
        let index = hash as usize & self.mask;
        let data = self.entries[index];

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

    /// Store an entry in the TT.
    pub fn store(
        &mut self,
        hash: u64,
        best_move: Move,
        score: i32,
        depth: u8,
        bound: Bound,
        ply: u32,
    ) {
        let index = hash as usize & self.mask;
        let entry = TTEntry {
            key16: verification_key(hash),
            best_move,
            score: score_to_tt(score, ply),
            depth,
            bound,
            age: self.generation,
        };
        self.entries[index] = entry.pack();
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.fill(0);
        self.generation = 0;
    }

    /// Resize the table to the given size in megabytes. Clears all entries.
    pub fn resize(&mut self, size_mb: usize) {
        *self = Self::new(size_mb);
    }

    /// Increment the generation counter (called at the start of each new search).
    pub fn new_search(&mut self) {
        self.generation = self.generation.wrapping_add(1) & 0x3F;
    }

    /// Returns approximate usage as permille (0-1000).
    pub fn hashfull(&self) -> u32 {
        let sample = self.entries.len().min(1000);
        let mut used = 0u32;
        for i in 0..sample {
            if self.entries[i] != 0 {
                let entry = TTEntry::unpack(self.entries[i]);
                if entry.age == self.generation {
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
        let mut tt = TranspositionTable::new(1);
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
        let mut tt = TranspositionTable::new(1);
        let hash1 = 0xAAAA_0000_0000_0001;
        let hash2 = 0xBBBB_0000_0000_0001; // same index, different key

        tt.store(hash1, Move::NULL, 100, 3, Bound::Exact, 0);

        // Probe with different key at same index
        assert!(tt.probe(hash2, 0).is_none());
    }

    #[test]
    fn mate_score_adjustment() {
        let mut tt = TranspositionTable::new(1);
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
        let mut tt = TranspositionTable::new(1);
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
        let mut tt = TranspositionTable::new(1);
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
}
