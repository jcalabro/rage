//! Opening book for testing games.
//!
//! Provides a set of balanced opening positions (FEN strings) to reduce
//! variance in engine testing. Each opening is played twice (once per color)
//! to form a game pair.

use std::path::Path;

/// Default set of balanced opening positions drawn from common openings.
/// These positions are after a few moves to skip book-move variance while
/// remaining balanced enough for fair testing.
const DEFAULT_OPENINGS: &[&str] = &[
    // Starting position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    // Italian Game (4 moves)
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    // Ruy Lopez (3...a6)
    "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
    // Sicilian Najdorf (6 moves)
    "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    // French Defense (3...Nf6)
    "rnbqkb1r/ppp2ppp/4pn2/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 2 4",
    // Caro-Kann (4...Bf5)
    "rn1qkbnr/pp2pppp/2p5/3p1b2/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 1 4",
    // Queen's Gambit Declined
    "rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 2 3",
    // Slav Defense
    "rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 2 4",
    // King's Indian Defense
    "rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N5/PP3PPP/R1BQKBNR w KQkq - 0 5",
    // Nimzo-Indian Defense
    "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    // Queen's Indian Defense
    "rnbqk2r/p1pp1ppp/1p2pn2/8/2PP4/5NP1/PP2PP1P/RNBQKB1R b KQkq - 0 4",
    // English Opening (Symmetrical)
    "rnbqkbnr/pp1ppppp/8/2p5/2P5/5N2/PP1PPPPP/RNBQKB1R b KQkq - 1 2",
    // Scotch Game
    "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 3",
    // Petroff Defense
    "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    // Dutch Defense
    "rnbqkbnr/ppppp1pp/8/5p2/3P4/8/PPP1PPPP/RNBQKBNR w KQkq f6 0 2",
    // Pirc Defense
    "rnbqkb1r/ppp1pp1p/3p1np1/8/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 4",
    // Scandinavian Defense
    "rnbqkb1r/ppp1pppp/5n2/3P4/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3",
    // Grünfeld Defense
    "rnbqkb1r/ppp1pp1p/5np1/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
    // Benoni Defense
    "rnbqkb1r/pp1p1ppp/4pn2/2pP4/2P5/2N5/PP2PPPP/R1BQKBNR b KQkq - 1 4",
    // London System
    "rnbqkb1r/ppp1pppp/5n2/3p4/3P1B2/5N2/PPP1PPPP/RN1QKB1R b KQkq - 3 3",
    // Catalan Opening
    "rnbqkb1r/pppp1ppp/4pn2/8/2PP4/6P1/PP2PP1P/RNBQKBNR b KQkq - 0 3",
    // Philidor Defense
    "rnbqkbnr/ppp2ppp/3p4/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",
    // Vienna Game
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 1 2",
    // Four Knights Game
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 4",
    // Budapest Gambit
    "rnbqkb1r/pppp1ppp/5n2/4p3/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
    // Bogo-Indian Defense
    "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 2 4",
    // Semi-Slav Defense
    "rnbqkb1r/pp3ppp/2p1pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5",
    // Alekhine Defense
    "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
    // Modern Defense
    "rnbqk1nr/ppppppbp/6p1/8/3PP3/2N5/PPP2PPP/R1BQKBNR b KQkq - 2 3",
    // King's Gambit
    "rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq f3 0 2",
    // Tarrasch Defense
    "rnbqkb1r/pp3ppp/4pn2/2pp4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5",
    // Exchange QGD
    "rnbqkb1r/ppp2ppp/4pn2/3P4/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 4",
    // Symmetrical English
    "rnbqkbnr/pp1ppppp/8/2p5/2P5/2N5/PP1PPPPP/R1BQKBNR b KQkq - 1 2",
    // Trompowsky Attack
    "rnbqkb1r/pppppppp/5n2/6B1/3P4/8/PPP1PPPP/RN1QKBNR b KQkq - 2 2",
    // Owen Defense
    "rnbqkbnr/p1pppppp/1p6/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    // Réti Opening
    "rnbqkbnr/ppp1pppp/8/3p4/8/5NP1/PPPPPP1P/RNBQKB1R b KQkq - 0 2",
    // Bird Opening
    "rnbqkbnr/pppppppp/8/8/5P2/8/PPPPP1PP/RNBQKBNR b KQkq f3 0 1",
    // Closed Sicilian
    "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3",
    // Torre Attack
    "rnbqkb1r/ppp1pppp/5n2/3p2B1/3P4/5N2/PPP1PPPP/RN1QKB1R b KQkq - 3 3",
    // Colle System
    "rnbqkb1r/ppp1pppp/5n2/3p4/3P4/4PN2/PPP2PPP/RNBQKB1R b KQkq - 0 3",
    // Giuoco Piano
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    // Two Knights Defense
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    // Semi-Tarrasch
    "rnbqkb1r/pp3ppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 1 5",
    // Ragozin Defense
    "rnbqk2r/ppp1bppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 4 5",
    // Old Indian Defense
    "rnbqkb1r/ppp1pppp/3p1n2/8/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq - 2 3",
    // Symmetrical KID
    "rnbqk1nr/ppp1ppbp/3p2p1/8/2PP4/2N3P1/PP2PP1P/R1BQKBNR b KQkq - 0 4",
    // Evans Gambit
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/1PB1P3/5N2/P1PP1PPP/RNBQK2R b KQkq b3 0 4",
    // Queen's Gambit Accepted
    "rnbqkbnr/ppp1pppp/8/8/2pP4/5N2/PPP1PPPP/RNBQKB1R b KQkq - 1 3",
    // Kan Sicilian
    "rnbqkbnr/1p1p1ppp/p3p3/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 0 5",
    // Richter-Rauzer
    "r1bqkb1r/pp2pppp/2np1n2/6B1/3NP3/2N5/PPP2PPP/R2QKB1R b KQkq - 1 6",
];

/// An opening book: a collection of starting FEN positions.
pub struct OpeningBook {
    fens: Vec<String>,
}

impl OpeningBook {
    /// Create a book from a vector of FEN strings.
    pub fn from_fens(fens: Vec<String>) -> Self {
        OpeningBook { fens }
    }

    /// Load openings from a file (one FEN per line, blank lines and '#' comments ignored).
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read opening book: {e}"))?;

        let fens: Vec<String> = content
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| l.to_string())
            .collect();

        if fens.is_empty() {
            return Err("opening book contains no positions".to_string());
        }

        Ok(OpeningBook { fens })
    }

    /// Create a book from the built-in default openings.
    pub fn default_book() -> Self {
        OpeningBook {
            fens: DEFAULT_OPENINGS.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    /// Number of openings.
    pub fn len(&self) -> usize {
        self.fens.len()
    }

    /// Get opening FEN by index (wraps around if index >= len).
    pub fn get(&self, index: usize) -> &str {
        &self.fens[index % self.fens.len()]
    }

    /// Shuffle the opening order using the given RNG.
    pub fn shuffle(&mut self, rng: &mut impl rand::Rng) {
        use rand::seq::SliceRandom;
        self.fens.shuffle(rng);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_default_book_has_positions() {
        let book = OpeningBook::default_book();
        assert!(book.len() >= 40, "default book should have ~50 positions");
    }

    #[test]
    fn test_default_book_valid_fens() {
        use rage_core::Board;
        for (i, fen) in DEFAULT_OPENINGS.iter().enumerate() {
            assert!(
                Board::from_fen(fen).is_ok(),
                "default opening #{i} is invalid FEN: {fen}"
            );
        }
    }

    #[test]
    fn test_get_wraps_around() {
        let book = OpeningBook::default_book();
        let n = book.len();
        assert_eq!(book.get(0), book.get(n));
        assert_eq!(book.get(1), book.get(n + 1));
    }
}
