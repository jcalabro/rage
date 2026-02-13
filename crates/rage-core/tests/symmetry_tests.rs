#![allow(clippy::unwrap_used)]

mod common;

use rage_core::*;

/// Create a color-mirrored copy of a board.
///
/// Flips all pieces vertically, swaps colors, swaps castling rights,
/// flips en passant square, and flips side to move. The resulting
/// position is the "same game" from the opposite color's perspective.
fn mirror_board(board: &Board) -> Board {
    let mut mirrored = Board::empty();

    for sq in Square::ALL {
        if let Some(piece) = board.piece_on(sq) {
            let m_sq = sq.flip_rank();
            let m_piece = Piece::new(!piece.color, piece.piece_type);
            mirrored.put_piece(m_piece, m_sq);
        }
    }

    mirrored.set_side_to_move(!board.side_to_move());

    // Swap white <-> black castling rights.
    let old = board.castling_rights();
    let mut new_bits = 0u8;
    if old.contains(CastlingRights::WHITE_KINGSIDE) {
        new_bits |= CastlingRights::BLACK_KINGSIDE;
    }
    if old.contains(CastlingRights::WHITE_QUEENSIDE) {
        new_bits |= CastlingRights::BLACK_QUEENSIDE;
    }
    if old.contains(CastlingRights::BLACK_KINGSIDE) {
        new_bits |= CastlingRights::WHITE_KINGSIDE;
    }
    if old.contains(CastlingRights::BLACK_QUEENSIDE) {
        new_bits |= CastlingRights::WHITE_QUEENSIDE;
    }
    mirrored.set_castling(CastlingRights::new(new_bits));

    if let Some(ep) = board.en_passant_square() {
        mirrored.set_en_passant(Some(ep.flip_rank()));
    }

    mirrored.set_halfmove_clock(board.halfmove_clock());
    mirrored.set_fullmove_number(board.fullmove_number());
    mirrored.refresh_zobrist();
    mirrored
}

/// Standard perft positions used across the test suite.
const PERFT_POSITIONS: &[&str] = &[
    Board::STARTPOS_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
];

#[test]
fn mirror_preserves_legal_move_count() {
    for fen in PERFT_POSITIONS {
        let mut board = Board::from_fen(fen).unwrap();
        let mut mirrored = mirror_board(&board);

        let moves = board.generate_legal_moves();
        let mirror_moves = mirrored.generate_legal_moves();

        assert_eq!(
            moves.len(),
            mirror_moves.len(),
            "legal move count mismatch for {fen}"
        );
    }
}

#[test]
fn mirror_preserves_perft_depth2() {
    for fen in PERFT_POSITIONS {
        let mut board = Board::from_fen(fen).unwrap();
        let mut mirrored = mirror_board(&board);

        let nodes = perft(&mut board, 2);
        let mirror_nodes = perft(&mut mirrored, 2);

        assert_eq!(
            nodes, mirror_nodes,
            "perft(2) mismatch for {fen}: original={nodes}, mirrored={mirror_nodes}"
        );
    }
}

#[test]
#[ignore]
fn mirror_preserves_perft_depth4() {
    for fen in PERFT_POSITIONS {
        let mut board = Board::from_fen(fen).unwrap();
        let mut mirrored = mirror_board(&board);

        let nodes = perft(&mut board, 4);
        let mirror_nodes = perft(&mut mirrored, 4);

        assert_eq!(
            nodes, mirror_nodes,
            "perft(4) mismatch for {fen}: original={nodes}, mirrored={mirror_nodes}"
        );
    }
}

/// Mirror positions reached during random games — covers diverse board states.
#[test]
fn mirror_random_game_positions() {
    for game_seed in 1u64..=20 {
        let mut rng = common::Xorshift(game_seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut board = Board::starting_position();

        for _ in 0..40 {
            let mut mirrored = mirror_board(&board);
            let moves = board.generate_legal_moves();
            let mirror_moves = mirrored.generate_legal_moves();

            assert_eq!(
                moves.len(),
                mirror_moves.len(),
                "move count mismatch at {}: original={}, mirrored={}",
                board.to_fen(),
                moves.len(),
                mirror_moves.len()
            );

            if moves.is_empty() {
                break;
            }

            let idx = (rng.next_u64() as usize) % moves.len();
            board.make_move(moves.as_slice()[idx]);

            if board.halfmove_clock() >= 100 {
                break;
            }
        }
    }
}
