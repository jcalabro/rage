//! Zobrist hashing for positions.
//!
//! Each board feature (piece on square, side to move, castling rights, en passant file)
//! is assigned a random 64-bit key. The position hash is the XOR of all active features.
//! This allows incremental updates: toggling a feature XORs its key in/out.

use crate::types::{CastlingRights, Color, File, PieceType, Square};

/// Precomputed Zobrist keys.
pub struct ZobristKeys {
    /// Keys for each (piece_type, color, square) combination.
    /// Indexed as [color][piece_type][square].
    pub pieces: [[[u64; Square::NUM]; PieceType::NUM]; Color::NUM],

    /// Key XORed when it's Black's turn to move.
    pub side_to_move: u64,

    /// Keys for each castling rights configuration (16 possible states).
    pub castling: [u64; CastlingRights::NUM_VARIANTS],

    /// Keys for en passant file (only the file matters, not the rank).
    /// Index 0 = no en passant, 1-8 = files A-H.
    pub en_passant: [u64; File::NUM + 1],
}

impl ZobristKeys {
    /// Generate keys using a simple xorshift64 PRNG with a fixed seed.
    const fn new() -> Self {
        let mut state = 0x9E37_79B9_7F4A_7C15u64; // golden ratio derived seed
        let mut keys = Self {
            pieces: [[[0u64; Square::NUM]; PieceType::NUM]; Color::NUM],
            side_to_move: 0,
            castling: [0u64; CastlingRights::NUM_VARIANTS],
            en_passant: [0u64; File::NUM + 1],
        };

        // Generate piece keys
        let mut color = 0usize;
        while color < Color::NUM {
            let mut piece = 0usize;
            while piece < PieceType::NUM {
                let mut sq = 0usize;
                while sq < Square::NUM {
                    state = xorshift64(state);
                    keys.pieces[color][piece][sq] = state;
                    sq += 1;
                }
                piece += 1;
            }
            color += 1;
        }

        // Side to move
        state = xorshift64(state);
        keys.side_to_move = state;

        // Castling rights
        let mut i = 0usize;
        while i < CastlingRights::NUM_VARIANTS {
            state = xorshift64(state);
            keys.castling[i] = state;
            i += 1;
        }

        // En passant
        let mut i = 0usize;
        while i <= File::NUM {
            state = xorshift64(state);
            keys.en_passant[i] = state;
            i += 1;
        }

        keys
    }
}

const fn xorshift64(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// Global Zobrist keys, computed at compile time.
pub static KEYS: ZobristKeys = ZobristKeys::new();

/// Compute the Zobrist key for a piece on a square.
pub const fn piece_key(color: Color, piece_type: PieceType, sq: Square) -> u64 {
    KEYS.pieces[color.index()][piece_type.index()][sq.index()]
}

/// Compute the Zobrist key for side-to-move (XOR this when it's Black's turn).
pub const fn side_key() -> u64 {
    KEYS.side_to_move
}

/// Compute the Zobrist key for castling rights.
pub const fn castling_key(rights: CastlingRights) -> u64 {
    KEYS.castling[rights.index()]
}

/// Compute the Zobrist key for en passant. Pass `None` for no en passant.
pub fn ep_key(ep_file: Option<File>) -> u64 {
    match ep_file {
        None => KEYS.en_passant[0],
        Some(f) => KEYS.en_passant[f.index() + 1],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_unique() {
        // Collect all keys and verify no duplicates (extremely unlikely with good PRNG)
        let mut all_keys = Vec::new();
        for color in 0..Color::NUM {
            for piece in 0..PieceType::NUM {
                for sq in 0..Square::NUM {
                    all_keys.push(KEYS.pieces[color][piece][sq]);
                }
            }
        }
        all_keys.push(KEYS.side_to_move);
        for k in &KEYS.castling {
            all_keys.push(*k);
        }
        for k in &KEYS.en_passant {
            all_keys.push(*k);
        }

        let len = all_keys.len();
        all_keys.sort();
        all_keys.dedup();
        assert_eq!(all_keys.len(), len, "duplicate Zobrist keys detected");
    }

    #[test]
    fn keys_are_nonzero() {
        for color in 0..Color::NUM {
            for piece in 0..PieceType::NUM {
                for sq in 0..Square::NUM {
                    assert_ne!(KEYS.pieces[color][piece][sq], 0);
                }
            }
        }
        assert_ne!(KEYS.side_to_move, 0);
    }

    #[test]
    fn xor_is_reversible() {
        let key = piece_key(Color::White, PieceType::Pawn, Square::E2);
        let hash = 0u64 ^ key;
        assert_eq!(hash ^ key, 0);
    }
}
