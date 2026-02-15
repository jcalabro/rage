//! Pure Rust Syzygy tablebase probing for the Rage chess engine.
//!
//! Provides WDL (Win/Draw/Loss) and DTZ (Distance To Zeroing) probing
//! for endgame positions with up to 7 pieces. Uses the Syzygy table
//! format (.rtbw for WDL, .rtbz for DTZ).
//!
//! # Wasm
//!
//! Syzygy probing is not available in wasm builds (tables are too large
//! for browser deployment). All public functions compile to no-ops on
//! `wasm32` targets, returning `None` for all probes.

// All modules are active on non-wasm targets.
// On wasm, we provide a stub Tablebase that returns None for everything.

#[cfg(not(target_arch = "wasm32"))]
mod cache;
#[cfg(not(target_arch = "wasm32"))]
mod compress;
#[cfg(not(target_arch = "wasm32"))]
mod embedded;
#[cfg(not(target_arch = "wasm32"))]
mod errors;
#[cfg(not(target_arch = "wasm32"))]
mod indexing;
#[cfg(not(target_arch = "wasm32"))]
mod material;
#[cfg(not(target_arch = "wasm32"))]
mod probe;
#[cfg(not(target_arch = "wasm32"))]
mod table;
#[cfg(not(target_arch = "wasm32"))]
mod tablebase;

// Re-exports for non-wasm
#[cfg(not(target_arch = "wasm32"))]
pub use embedded::extract_embedded_tables;
#[cfg(not(target_arch = "wasm32"))]
pub use probe::{Dtz, Wdl, filter_root_moves, probe_dtz, probe_wdl, wdl_to_score};
#[cfg(not(target_arch = "wasm32"))]
pub use tablebase::Tablebase;

// ---------------------------------------------------------------------------
// Wasm stub
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
pub use wasm_stub::*;

#[cfg(target_arch = "wasm32")]
mod wasm_stub {
    use rage_core::{Board, Move};
    use std::path::PathBuf;

    pub fn extract_embedded_tables() -> Option<PathBuf> {
        None
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub enum Wdl {
        Loss = 0,
        BlessedLoss = 1,
        Draw = 2,
        CursedWin = 3,
        Win = 4,
    }

    impl Wdl {
        pub fn negate(self) -> Self {
            self
        }
        pub fn to_score(self) -> i32 {
            0
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Dtz(pub i32);

    pub struct Tablebase {
        pub probe_depth: u32,
        pub probe_limit: u32,
    }

    impl Tablebase {
        pub fn new(_path: &str) -> Self {
            Self {
                probe_depth: 1,
                probe_limit: 0,
            }
        }
        pub fn is_available(&self) -> bool {
            false
        }
        pub fn set_probe_depth(&mut self, depth: u32) {
            self.probe_depth = depth;
        }
        pub fn set_probe_limit(&mut self, limit: u32) {
            self.probe_limit = limit;
        }
    }

    pub fn probe_wdl(_board: &mut Board, _tablebase: &Tablebase) -> Option<Wdl> {
        None
    }

    pub fn probe_dtz(_board: &mut Board, _tablebase: &Tablebase) -> Option<Dtz> {
        None
    }

    pub fn filter_root_moves(
        _board: &mut Board,
        _tablebase: &Tablebase,
        moves: &[Move],
    ) -> Vec<(Move, Wdl)> {
        moves.iter().map(|&m| (m, Wdl::Draw)).collect()
    }

    pub fn wdl_to_score(_wdl: Wdl, _ply: u32) -> i32 {
        0
    }
}
