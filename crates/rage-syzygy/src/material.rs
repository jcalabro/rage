use rage_core::{Board, Color, PieceType};

/// Material key: counts of each piece type for both sides.
/// Used to identify which tablebase file to look up.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MaterialKey {
    /// Piece counts indexed by [color][piece_type] (excludes kings).
    /// piece_type ordering: Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4
    pub counts: [[u8; 5]; 2],
}

/// The piece types stored in material key (no King).
const PIECE_TYPES: [PieceType; 5] = [
    PieceType::Pawn,
    PieceType::Knight,
    PieceType::Bishop,
    PieceType::Rook,
    PieceType::Queen,
];

impl MaterialKey {
    /// Extract material key from a board position.
    pub fn from_board(board: &Board) -> Self {
        let mut counts = [[0u8; 5]; 2];
        for &color in &[Color::White, Color::Black] {
            for (i, &pt) in PIECE_TYPES.iter().enumerate() {
                counts[color.index()][i] = board.pieces(color, pt).popcount() as u8;
            }
        }
        Self { counts }
    }

    /// Total number of pieces (including kings).
    pub fn total_pieces(&self) -> u32 {
        let sum: u32 = self
            .counts
            .iter()
            .flat_map(|c| c.iter())
            .map(|&x| x as u32)
            .sum();
        sum + 2 // add 2 kings
    }

    /// Returns true if this position has any pawns.
    pub fn has_pawns(&self) -> bool {
        self.counts[0][0] > 0 || self.counts[1][0] > 0
    }

    /// Generate the Syzygy table filename (e.g. "KRvK", "KPvKP", "KQRvKR").
    ///
    /// Convention: stronger side first (more material). If equal, white first.
    /// Pieces ordered: K first, then Q, R, B, N, P.
    pub fn table_name(&self) -> String {
        let (strong, weak) = self.canonical_sides();
        let mut name = String::with_capacity(16);

        // Strong side
        name.push('K');
        Self::append_pieces(&mut name, &strong);

        name.push('v');

        // Weak side
        name.push('K');
        Self::append_pieces(&mut name, &weak);

        name
    }

    /// Returns (strong_side_counts, weak_side_counts) in canonical order.
    /// Strong side has more material value, or if equal, white is strong.
    fn canonical_sides(&self) -> ([u8; 5], [u8; 5]) {
        let w_val = Self::material_value(&self.counts[0]);
        let b_val = Self::material_value(&self.counts[1]);

        if b_val > w_val {
            (self.counts[1], self.counts[0])
        } else {
            (self.counts[0], self.counts[1])
        }
    }

    /// Compute a simple material value for ordering purposes.
    /// Q=9, R=5, B=3, N=3, P=1
    fn material_value(counts: &[u8; 5]) -> u32 {
        counts[0] as u32 // pawns
            + counts[1] as u32 * 3 // knights
            + counts[2] as u32 * 3 // bishops
            + counts[3] as u32 * 5 // rooks
            + counts[4] as u32 * 9 // queens
    }

    /// Append piece characters in Syzygy order: Q, R, B, N, P.
    fn append_pieces(name: &mut String, counts: &[u8; 5]) {
        // Order: Queen(4), Rook(3), Bishop(2), Knight(1), Pawn(0)
        for &(idx, ch) in &[(4, 'Q'), (3, 'R'), (2, 'B'), (1, 'N'), (0, 'P')] {
            for _ in 0..counts[idx] {
                name.push(ch);
            }
        }
    }

    /// Returns true if white is the "strong" side (more material, or equal).
    pub fn white_is_strong(&self) -> bool {
        Self::material_value(&self.counts[0]) >= Self::material_value(&self.counts[1])
    }

    /// Returns the side that corresponds to the "strong" side in the canonical ordering.
    pub fn strong_side(&self) -> Color {
        if self.white_is_strong() {
            Color::White
        } else {
            Color::Black
        }
    }

    /// Whether the material is symmetric (same pieces on both sides).
    pub fn is_symmetric(&self) -> bool {
        self.counts[0] == self.counts[1]
    }

    /// Number of unique piece roles: roles where count == 1 on a side, summed
    /// across both sides. Kings (always count=1 per side) add 2.
    pub fn unique_pieces(&self) -> usize {
        let mut count = 2; // both kings
        for side in &self.counts {
            for &c in side {
                if c == 1 {
                    count += 1;
                }
            }
        }
        count
    }

    /// Minimum count among roles with count >= 2 (the smallest group of
    /// identical pieces). Returns 0 if no such group exists.
    pub fn min_like_man(&self) -> usize {
        self.counts
            .iter()
            .flat_map(|s| s.iter())
            .copied()
            .filter(|&c| c >= 2)
            .min()
            .unwrap_or(0) as usize
    }

    /// White pawn count.
    pub fn white_pawn_count(&self) -> u8 {
        self.counts[0][0]
    }

    /// Black pawn count.
    pub fn black_pawn_count(&self) -> u8 {
        self.counts[1][0]
    }

    /// Generate table name with colors swapped (for matching asymmetric material).
    pub fn table_name_swapped(&self) -> String {
        let swapped = MaterialKey {
            counts: [self.counts[1], self.counts[0]],
        };
        swapped.table_name()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn material_key_startpos() {
        let board = Board::from_fen(Board::STARTPOS_FEN).unwrap();
        let key = MaterialKey::from_board(&board);
        // White: 8P 2N 2B 2R 1Q
        assert_eq!(key.counts[0], [8, 2, 2, 2, 1]);
        // Black: same
        assert_eq!(key.counts[1], [8, 2, 2, 2, 1]);
        assert_eq!(key.total_pieces(), 32);
        assert!(key.has_pawns());
    }

    #[test]
    fn material_key_kr_vs_k() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4K2R w - - 0 1").unwrap();
        let key = MaterialKey::from_board(&board);
        assert_eq!(key.counts[0], [0, 0, 0, 1, 0]); // White: 1 rook
        assert_eq!(key.counts[1], [0, 0, 0, 0, 0]); // Black: nothing
        assert_eq!(key.total_pieces(), 3);
        assert!(!key.has_pawns());
        assert_eq!(key.table_name(), "KRvK");
    }

    #[test]
    fn material_key_kp_vs_kp() {
        let board = Board::from_fen("4k3/4p3/8/8/8/8/4P3/4K3 w - - 0 1").unwrap();
        let key = MaterialKey::from_board(&board);
        assert_eq!(key.counts[0], [1, 0, 0, 0, 0]); // White: 1 pawn
        assert_eq!(key.counts[1], [1, 0, 0, 0, 0]); // Black: 1 pawn
        assert!(key.has_pawns());
        assert_eq!(key.table_name(), "KPvKP");
    }

    #[test]
    fn material_key_kqr_vs_kr() {
        let board = Board::from_fen("4k2r/8/8/8/8/8/8/3QK2R w - - 0 1").unwrap();
        let key = MaterialKey::from_board(&board);
        assert_eq!(key.table_name(), "KQRvKR");
    }

    #[test]
    fn material_key_black_stronger() {
        // Black has queen, white has nothing extra
        let board = Board::from_fen("4k3/3q4/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let key = MaterialKey::from_board(&board);
        // Strong side is black (has queen)
        assert_eq!(key.table_name(), "KQvK");
        assert!(!key.white_is_strong());
        assert_eq!(key.strong_side(), Color::Black);
    }

    #[test]
    fn material_key_kbn_vs_k() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/3BKN2 w - - 0 1").unwrap();
        let key = MaterialKey::from_board(&board);
        assert_eq!(key.table_name(), "KBNvK");
    }

    #[test]
    fn material_key_kpp_vs_kp() {
        let board = Board::from_fen("4k3/4p3/8/8/8/8/3PP3/4K3 w - - 0 1").unwrap();
        let key = MaterialKey::from_board(&board);
        assert_eq!(key.table_name(), "KPPvKP");
    }
}
