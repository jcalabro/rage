use rage_core::{Board, Color, Move, PieceType, Square};

use crate::accumulator::Accumulator;
use crate::arch::{HIDDEN_SIZE, OUTPUT_SCALE};
use crate::features;
use crate::weights::NnueWeights;

/// The NNUE network: feature transformer weights + output layer.
///
/// Architecture: HalfKAv2 with 768 features x 10 king buckets → HIDDEN_SIZE → 1.
/// The feature transformer maps sparse binary features to a dense accumulator.
/// The output layer applies CReLU and a single linear dot product.
pub struct NnueNetwork {
    weights: NnueWeights,
}

impl NnueNetwork {
    /// Create a network from loaded weights.
    pub fn new(weights: NnueWeights) -> Self {
        Self { weights }
    }

    /// Access the underlying weights (for incremental updates).
    pub fn weights(&self) -> &NnueWeights {
        &self.weights
    }

    /// Get the feature transformer weight column for a given feature index.
    pub fn ft_weight_column(&self, feature_idx: usize) -> &[i16; HIDDEN_SIZE] {
        &self.weights.ft_weights[feature_idx]
    }

    /// Get the feature transformer biases.
    pub fn ft_biases(&self) -> &[i16; HIDDEN_SIZE] {
        &self.weights.ft_biases
    }

    /// Fully refresh an accumulator from the board position.
    /// This iterates all pieces and adds their feature weight columns.
    pub fn refresh_accumulator(&self, board: &Board, acc: &mut Accumulator) {
        acc.init_from_biases(&self.weights.ft_biases);

        let feature_piece_types = PieceType::ALL;

        for perspective in [Color::White, Color::Black] {
            let king_sq = board.king_square(perspective);
            let pi = perspective.index();

            for &color in &[Color::White, Color::Black] {
                for &pt in &feature_piece_types {
                    let bb = board.pieces(color, pt);
                    for sq in bb {
                        let idx = features::feature_index(perspective, king_sq, color, pt, sq);
                        acc.add_feature(pi, &self.weights.ft_weights[idx]);
                    }
                }
            }
        }
    }

    /// Run the forward pass: CReLU on both perspectives, then linear output.
    ///
    /// `stm` is the side to move. The STM perspective comes first in the
    /// concatenated CReLU output.
    ///
    /// Returns the evaluation in centipawns.
    pub fn forward(&self, acc: &Accumulator, stm: Color) -> i32 {
        let stm_idx = stm.index();
        let nstm_idx = stm.flip().index();

        let stm_values = &acc.values[stm_idx];
        let nstm_values = &acc.values[nstm_idx];

        let raw =
            crate::simd::ops().crelu_linear(stm_values, nstm_values, &self.weights.output_weights);
        (raw + self.weights.output_bias) / OUTPUT_SCALE
    }

    /// Evaluate a board position from scratch (full refresh).
    /// Convenience method that creates a temporary accumulator.
    pub fn evaluate(&self, board: &Board) -> i32 {
        let mut acc = Accumulator::EMPTY;
        self.refresh_accumulator(board, &mut acc);
        self.forward(&acc, board.side_to_move())
    }

    /// Incrementally update a child accumulator from a parent accumulator
    #[allow(clippy::too_many_arguments)]
    /// after a move has been made on the board.
    ///
    /// `board` is the board state AFTER `make_move` has been called.
    /// `parent` is the accumulator from before the move.
    /// `child` receives the updated accumulator.
    /// `m` is the move that was made.
    /// `moved_piece` is the piece type that moved (before make_move).
    /// `captured_piece` is the captured piece type if any (before make_move).
    /// `moving_color` is the color that made the move (board.side_to_move().flip()).
    pub fn update_accumulator(
        &self,
        board: &Board,
        parent: &Accumulator,
        child: &mut Accumulator,
        m: Move,
        moved_piece: PieceType,
        captured_piece: Option<PieceType>,
        moving_color: Color,
    ) {
        let from = m.from_sq();
        let to = m.to_sq();
        let opponent = moving_color.flip();

        for perspective in [Color::White, Color::Black] {
            let pi = perspective.index();

            // If the king of this perspective moved, the bucket likely changed.
            // Full refresh is needed since all feature indices change.
            if moved_piece == PieceType::King && moving_color == perspective {
                child.dirty[pi] = true;
                continue;
            }

            // Copy parent values for this perspective
            child.values[pi] = parent.values[pi];
            child.dirty[pi] = false;

            let king_sq = board.king_square(perspective);

            // Determine the piece that arrived at 'to' (could be promo piece)
            let arriving_piece = if m.is_promotion() {
                m.promotion_piece()
            } else {
                moved_piece
            };

            if m.is_castling() {
                // Castling: king + rook both moved. The king is handled as a
                // feature (it's the OPPONENT's king from the other perspective,
                // or this perspective's king which already triggered dirty above).
                // Actually: the king IS a feature for the opponent's perspective.
                // But per our convention, we don't include the perspective's own king
                // as a feature (it's encoded by the bucket). The opponent's king IS
                // a feature though.
                //
                // If moving_color == perspective, we already marked dirty above.
                // If moving_color != perspective, the opponent's king moved — it IS a feature.

                // Remove old king feature
                let king_idx = features::feature_index(
                    perspective,
                    king_sq,
                    moving_color,
                    PieceType::King,
                    from,
                );
                child.sub_feature(pi, &self.weights.ft_weights[king_idx]);

                // Add new king feature
                let king_idx = features::feature_index(
                    perspective,
                    king_sq,
                    moving_color,
                    PieceType::King,
                    to,
                );
                child.add_feature(pi, &self.weights.ft_weights[king_idx]);

                // Determine rook from/to squares based on castling side
                let (rook_from, rook_to) = castling_rook_squares(to);

                // Remove old rook feature
                let rook_idx = features::feature_index(
                    perspective,
                    king_sq,
                    moving_color,
                    PieceType::Rook,
                    rook_from,
                );
                child.sub_feature(pi, &self.weights.ft_weights[rook_idx]);

                // Add new rook feature
                let rook_idx = features::feature_index(
                    perspective,
                    king_sq,
                    moving_color,
                    PieceType::Rook,
                    rook_to,
                );
                child.add_feature(pi, &self.weights.ft_weights[rook_idx]);
            } else {
                // Remove piece from old square
                let old_idx =
                    features::feature_index(perspective, king_sq, moving_color, moved_piece, from);
                child.sub_feature(pi, &self.weights.ft_weights[old_idx]);

                // Add piece to new square (possibly promoted)
                let new_idx =
                    features::feature_index(perspective, king_sq, moving_color, arriving_piece, to);
                child.add_feature(pi, &self.weights.ft_weights[new_idx]);

                // Remove captured piece
                if let Some(cap_pt) = captured_piece {
                    let cap_sq = if m.is_en_passant() {
                        // EP: captured pawn is on a different square than 'to'
                        ep_capture_square(to, moving_color)
                    } else {
                        to
                    };
                    let cap_idx =
                        features::feature_index(perspective, king_sq, opponent, cap_pt, cap_sq);
                    child.sub_feature(pi, &self.weights.ft_weights[cap_idx]);
                }
            }
        }
    }
}

/// Get the rook from/to squares for a castling move, given the king's destination.
fn castling_rook_squares(king_to: Square) -> (Square, Square) {
    match king_to {
        // White kingside: king g1, rook h1->f1
        sq if sq == Square::G1 => (Square::H1, Square::F1),
        // White queenside: king c1, rook a1->d1
        sq if sq == Square::C1 => (Square::A1, Square::D1),
        // Black kingside: king g8, rook h8->f8
        sq if sq == Square::G8 => (Square::H8, Square::F8),
        // Black queenside: king c8, rook a8->d8
        sq if sq == Square::C8 => (Square::A8, Square::D8),
        // Should never happen
        _ => (Square::A1, Square::A1),
    }
}

/// Get the square of the captured pawn in an en passant move.
fn ep_capture_square(ep_to: Square, moving_color: Color) -> Square {
    match moving_color {
        // White captures EP: pawn captured on rank below 'to'
        Color::White => Square::new(ep_to.index() as u8 - 8),
        // Black captures EP: pawn captured on rank above 'to'
        Color::Black => Square::new(ep_to.index() as u8 + 8),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn zero_weight_network_evaluates_to_zero() {
        let weights = NnueWeights::zeroed();
        let network = NnueNetwork::new(weights);
        let board = Board::starting_position();
        let score = network.evaluate(&board);
        assert_eq!(score, 0, "zero-weight network should evaluate to 0");
    }

    #[test]
    fn bias_only_network() {
        // With only output bias set, the eval should be bias / OUTPUT_SCALE
        let mut weights = NnueWeights::zeroed();
        weights.output_bias = OUTPUT_SCALE * 100; // +100 cp
        let network = NnueNetwork::new(weights);
        let board = Board::starting_position();
        let score = network.evaluate(&board);
        assert_eq!(score, 100, "bias-only network should return bias/scale");
    }

    #[test]
    fn random_weights_produces_nonzero_accumulator() {
        let weights = NnueWeights::random(42);
        let network = NnueNetwork::new(weights);
        let board = Board::starting_position();
        let mut acc = Accumulator::EMPTY;
        network.refresh_accumulator(&board, &mut acc);
        // Random weights should produce non-zero accumulator values.
        // The final score may round to 0 after dividing by OUTPUT_SCALE,
        // but the accumulator itself should be non-trivial.
        let any_nonzero = acc.values[0].iter().any(|&v| v != 0);
        assert!(
            any_nonzero,
            "random weights should produce non-zero accumulator"
        );
    }

    #[test]
    fn refresh_accumulator_idempotent() {
        let weights = NnueWeights::random(123);
        let network = NnueNetwork::new(weights);
        let board = Board::starting_position();

        let mut acc1 = Accumulator::EMPTY;
        network.refresh_accumulator(&board, &mut acc1);

        let mut acc2 = Accumulator::EMPTY;
        network.refresh_accumulator(&board, &mut acc2);

        assert_eq!(
            acc1.values, acc2.values,
            "two refreshes should be identical"
        );
    }

    #[test]
    fn different_positions_different_accumulators() {
        let weights = NnueWeights::random(99);
        let network = NnueNetwork::new(weights);

        let start = Board::starting_position();
        // Very different position (missing queens) to avoid collisions
        let other =
            Board::from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1").unwrap();

        let mut acc1 = Accumulator::EMPTY;
        let mut acc2 = Accumulator::EMPTY;
        network.refresh_accumulator(&start, &mut acc1);
        network.refresh_accumulator(&other, &mut acc2);
        // Different material should produce different accumulators with random weights.
        // (Final eval may collide after integer division by OUTPUT_SCALE, but
        // the raw accumulator values should differ.)
        assert_ne!(
            acc1.values, acc2.values,
            "different positions should have different accumulators"
        );
    }

    #[test]
    fn crelu_clamps_correctly() {
        use crate::arch::{CRELU_OUTPUT_SIZE, HIDDEN_SIZE};

        // Test with accumulator values at boundaries
        let mut stm = [0i16; HIDDEN_SIZE];
        let mut nstm = [0i16; HIDDEN_SIZE];

        // Set some values outside [0, QA]
        stm[0] = -100; // should clamp to 0
        stm[1] = 500; // should clamp to QA (255)
        stm[2] = 128; // should pass through
        nstm[0] = -1; // should clamp to 0
        nstm[1] = 255; // should pass through (exactly QA)

        let mut weights = [0i8; CRELU_OUTPUT_SIZE];
        weights[0] = 1;
        weights[1] = 1;
        weights[2] = 1;
        weights[HIDDEN_SIZE] = 1;
        weights[HIDDEN_SIZE + 1] = 1;

        let raw = crate::simd::ops().crelu_linear(&stm, &nstm, &weights);

        // Expected raw: 0*1 + 255*1 + 128*1 + 0*1 + 255*1 = 638
        assert_eq!(raw, 638);
        // After scale: 638 / OUTPUT_SCALE = 0 (integer division)
        assert_eq!(raw / OUTPUT_SCALE, 0);
    }

    /// Helper: verify incremental accumulator matches full refresh for every
    /// legal move from the given position.
    fn verify_incremental_for_position(network: &NnueNetwork, fen: &str) {
        let mut board = Board::from_fen(fen).unwrap();
        let moves = board.generate_legal_moves();

        let mut parent_acc = Accumulator::EMPTY;
        network.refresh_accumulator(&board, &mut parent_acc);

        for m in &moves {
            let m = *m;
            let piece_moved = board
                .piece_on(m.from_sq())
                .map_or(PieceType::Pawn, |p| p.piece_type);
            let captured_pt = if m.is_en_passant() {
                Some(PieceType::Pawn)
            } else {
                board.piece_on(m.to_sq()).map(|p| p.piece_type)
            };
            let stm = board.side_to_move();

            board.make_move(m);

            // Incremental update
            let mut inc_acc = Accumulator::EMPTY;
            network.update_accumulator(
                &board,
                &parent_acc,
                &mut inc_acc,
                m,
                piece_moved,
                captured_pt,
                stm,
            );

            // Full refresh for comparison
            let mut ref_acc = Accumulator::EMPTY;
            network.refresh_accumulator(&board, &mut ref_acc);

            // Compare non-dirty perspectives
            for pi in 0..2 {
                if !inc_acc.dirty[pi] {
                    assert_eq!(
                        inc_acc.values[pi], ref_acc.values[pi],
                        "accumulator mismatch at perspective {pi} after move {m} in position {fen}"
                    );
                } else {
                    // Dirty perspective (king moved): refresh and compare
                    let mut refreshed = inc_acc.clone();
                    network.refresh_accumulator(&board, &mut refreshed);
                    assert_eq!(
                        refreshed.values[pi], ref_acc.values[pi],
                        "refreshed dirty acc mismatch at perspective {pi} after move {m}"
                    );
                }
            }

            board.unmake_move(m);
        }
    }

    #[test]
    fn incremental_matches_refresh_startpos() {
        let network = NnueNetwork::new(NnueWeights::random(42));
        verify_incremental_for_position(&network, Board::STARTPOS_FEN);
    }

    #[test]
    fn incremental_matches_refresh_kiwipete() {
        let network = NnueNetwork::new(NnueWeights::random(42));
        verify_incremental_for_position(
            &network,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        );
    }

    #[test]
    fn incremental_matches_refresh_ep_position() {
        let network = NnueNetwork::new(NnueWeights::random(42));
        // Position with en passant available
        verify_incremental_for_position(
            &network,
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 1",
        );
    }

    #[test]
    fn incremental_matches_refresh_promotions() {
        let network = NnueNetwork::new(NnueWeights::random(42));
        // Position where promotions are available
        verify_incremental_for_position(&network, "8/P5k1/8/8/8/8/1K4p1/8 w - - 0 1");
    }

    #[test]
    fn incremental_matches_refresh_castling() {
        let network = NnueNetwork::new(NnueWeights::random(42));
        // Position where castling is available
        verify_incremental_for_position(
            &network,
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        );
    }

    #[test]
    fn incremental_multi_ply() {
        // Verify incremental updates work across multiple plies
        let network = NnueNetwork::new(NnueWeights::random(777));
        let mut board = Board::starting_position();

        let mut acc = Accumulator::EMPTY;
        network.refresh_accumulator(&board, &mut acc);

        // Play a series of moves, incrementally updating
        let moves_str = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"];
        for move_str in &moves_str {
            let m = board.parse_uci_move(move_str).unwrap();
            let piece_moved = board
                .piece_on(m.from_sq())
                .map_or(PieceType::Pawn, |p| p.piece_type);
            let captured_pt = if m.is_en_passant() {
                Some(PieceType::Pawn)
            } else {
                board.piece_on(m.to_sq()).map(|p| p.piece_type)
            };
            let stm = board.side_to_move();

            board.make_move(m);

            let mut new_acc = Accumulator::EMPTY;
            network.update_accumulator(
                &board,
                &acc,
                &mut new_acc,
                m,
                piece_moved,
                captured_pt,
                stm,
            );

            // Refresh any dirty perspectives
            if new_acc.dirty[0] || new_acc.dirty[1] {
                network.refresh_accumulator(&board, &mut new_acc);
            }

            // Verify against full refresh
            let mut ref_acc = Accumulator::EMPTY;
            network.refresh_accumulator(&board, &mut ref_acc);

            assert_eq!(
                new_acc.values, ref_acc.values,
                "multi-ply mismatch after {move_str}"
            );

            acc = new_acc;
        }
    }
}
