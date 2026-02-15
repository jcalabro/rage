//! WDL and DTZ probing for Syzygy tablebases.
//!
//! Given a chess position, determines the theoretical outcome (Win/Draw/Loss)
//! and distance to zeroing move (DTZ) by looking up the position in the
//! appropriate Syzygy table.

use rage_core::{Board, Move, PieceType};

use crate::material::MaterialKey;
use crate::tablebase::Tablebase;

/// Win/Draw/Loss result from a WDL probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Wdl {
    /// The side to move loses with best play.
    Loss = 0,
    /// The side to move loses but the opponent can't force a win under the
    /// 50-move rule ("blessed loss").
    BlessedLoss = 1,
    /// The position is a theoretical draw.
    Draw = 2,
    /// The side to move wins but can't force a win under the 50-move rule
    /// ("cursed win").
    CursedWin = 3,
    /// The side to move wins with best play.
    Win = 4,
}

impl Wdl {
    /// Convert from the raw table value (0-4) to Wdl.
    pub fn from_raw(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Loss),
            1 => Some(Self::BlessedLoss),
            2 => Some(Self::Draw),
            3 => Some(Self::CursedWin),
            4 => Some(Self::Win),
            _ => None,
        }
    }

    /// Convert to a signed integer for scoring: Win=2, CursedWin=1, Draw=0,
    /// BlessedLoss=-1, Loss=-2.
    pub fn to_score(self) -> i32 {
        match self {
            Self::Loss => -2,
            Self::BlessedLoss => -1,
            Self::Draw => 0,
            Self::CursedWin => 1,
            Self::Win => 2,
        }
    }

    /// Negate (flip perspective): Win becomes Loss, etc.
    pub fn negate(self) -> Self {
        match self {
            Self::Loss => Self::Win,
            Self::BlessedLoss => Self::CursedWin,
            Self::Draw => Self::Draw,
            Self::CursedWin => Self::BlessedLoss,
            Self::Win => Self::Loss,
        }
    }
}

/// Distance to zeroing move (pawn move or capture).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dtz(pub i32);

impl Dtz {
    /// The Wdl corresponding to this DTZ value.
    pub fn wdl(self) -> Wdl {
        if self.0 > 0 {
            Wdl::Win
        } else if self.0 < 0 {
            Wdl::Loss
        } else {
            Wdl::Draw
        }
    }
}

/// Probe the WDL value for the given position.
pub fn probe_wdl(board: &mut Board, tablebase: &Tablebase) -> Option<Wdl> {
    if board.castling_rights().bits() != 0 {
        return None;
    }

    let total_pieces = board.occupied().popcount();
    if total_pieces > tablebase.probe_limit {
        return None;
    }

    if total_pieces < 3 {
        return None;
    }

    let mat_key = MaterialKey::from_board(board);
    let table_name = mat_key.table_name();

    if tablebase.get_wdl(&table_name).is_err() {
        return None;
    }

    if board.en_passant_square().is_some() {
        return probe_wdl_with_ep(board, tablebase, &table_name);
    }

    probe_wdl_direct(board, tablebase, &table_name)
}

fn probe_wdl_direct(board: &mut Board, tablebase: &Tablebase, table_name: &str) -> Option<Wdl> {
    let moves = board.generate_legal_moves();
    let mut best_wdl = Wdl::Loss;
    let mut found_capture = false;

    for m in &moves {
        if !board.is_capture(*m) {
            continue;
        }
        found_capture = true;

        board.make_move(*m);
        let child_wdl = probe_wdl_after_move(board, tablebase);
        board.unmake_move(*m);

        if let Some(wdl) = child_wdl {
            let negated = wdl.negate();
            if negated as u8 > best_wdl as u8 {
                best_wdl = negated;
            }
            if best_wdl == Wdl::Win {
                return Some(Wdl::Win);
            }
        }
    }

    if !found_capture {
        return probe_table_value(board, tablebase, table_name);
    }

    if let Some(table_wdl) = probe_table_value(board, tablebase, table_name)
        && table_wdl as u8 > best_wdl as u8
    {
        best_wdl = table_wdl;
    }

    Some(best_wdl)
}

fn probe_wdl_with_ep(board: &mut Board, tablebase: &Tablebase, table_name: &str) -> Option<Wdl> {
    let mut best_wdl = Wdl::Loss;

    let moves = board.generate_legal_moves();

    for m in &moves {
        if !board.is_capture(*m) {
            continue;
        }

        board.make_move(*m);
        let child_wdl = probe_wdl_after_move(board, tablebase);
        board.unmake_move(*m);

        if let Some(wdl) = child_wdl {
            let negated = wdl.negate();
            if negated as u8 > best_wdl as u8 {
                best_wdl = negated;
            }
        }
    }

    if let Some(table_wdl) = probe_table_value(board, tablebase, table_name)
        && table_wdl as u8 > best_wdl as u8
    {
        best_wdl = table_wdl;
    }

    Some(best_wdl)
}

fn probe_wdl_after_move(board: &mut Board, tablebase: &Tablebase) -> Option<Wdl> {
    let total_pieces = board.occupied().popcount();
    if total_pieces > tablebase.probe_limit || total_pieces < 3 {
        return None;
    }

    let mat_key = MaterialKey::from_board(board);
    let table_name = mat_key.table_name();

    if tablebase.get_wdl(&table_name).is_err() {
        return None;
    }

    probe_wdl_direct(board, tablebase, &table_name)
}

/// Look up the raw WDL value for a position in the table.
fn probe_table_value(board: &Board, tablebase: &Tablebase, table_name: &str) -> Option<Wdl> {
    use crate::compress::decompress_value;
    use crate::indexing::encode_position;

    tablebase.read_wdl(table_name, |table| {
        let (side_idx, index) = encode_position(board, table)?;

        if side_idx >= table.files[0].sides.len() {
            return None;
        }

        let pairs = &table.files[0].sides[side_idx].pairs;

        let raw = match decompress_value(&table.data, pairs, index) {
            Ok(val) => val,
            Err(_) => return None,
        };

        Wdl::from_raw(raw)
    })?
}

/// Probe DTZ (Distance To Zeroing) for root move filtering.
pub fn probe_dtz(board: &mut Board, tablebase: &Tablebase) -> Option<Dtz> {
    if board.castling_rights().bits() != 0 {
        return None;
    }

    let total_pieces = board.occupied().popcount();
    if total_pieces > tablebase.probe_limit || total_pieces < 3 {
        return None;
    }

    let wdl = probe_wdl(board, tablebase)?;

    if wdl == Wdl::Draw || wdl == Wdl::BlessedLoss || wdl == Wdl::CursedWin {
        return Some(Dtz(0));
    }

    let mat_key = MaterialKey::from_board(board);
    let table_name = mat_key.table_name();

    if tablebase.get_dtz(&table_name).is_err() {
        return Some(Dtz(wdl.to_score()));
    }

    probe_dtz_via_moves(board, tablebase, wdl)
}

fn probe_dtz_via_moves(board: &mut Board, tablebase: &Tablebase, root_wdl: Wdl) -> Option<Dtz> {
    let moves = board.generate_legal_moves();

    if root_wdl == Wdl::Win {
        let mut best_dtz = i32::MAX;

        for m in &moves {
            let is_zeroing = board.is_capture(*m)
                || board
                    .piece_on(m.from_sq())
                    .is_some_and(|p| p.piece_type == PieceType::Pawn);

            board.make_move(*m);
            let child_wdl = probe_wdl(board, tablebase);
            board.unmake_move(*m);

            if let Some(wdl) = child_wdl {
                let negated = wdl.negate();
                if negated == Wdl::Win {
                    let dtz = if is_zeroing { 1 } else { 2 };
                    if dtz < best_dtz {
                        best_dtz = dtz;
                    }
                }
            }
        }

        if best_dtz < i32::MAX {
            return Some(Dtz(best_dtz));
        }
    } else if root_wdl == Wdl::Loss {
        let mut best_dtz = 0i32;

        for m in &moves {
            let is_zeroing = board.is_capture(*m)
                || board
                    .piece_on(m.from_sq())
                    .is_some_and(|p| p.piece_type == PieceType::Pawn);

            board.make_move(*m);
            let child_wdl = probe_wdl(board, tablebase);
            board.unmake_move(*m);

            if child_wdl.is_some() {
                let dtz = if is_zeroing { -1 } else { -2 };
                if dtz < best_dtz {
                    best_dtz = dtz;
                }
            }
        }

        return Some(Dtz(best_dtz));
    }

    Some(Dtz(root_wdl.to_score()))
}

/// Filter root moves based on DTZ values.
pub fn filter_root_moves(
    board: &mut Board,
    tablebase: &Tablebase,
    moves: &[Move],
) -> Vec<(Move, Wdl)> {
    let root_wdl = match probe_wdl(board, tablebase) {
        Some(wdl) => wdl,
        None => return moves.iter().map(|&m| (m, Wdl::Draw)).collect(),
    };

    let mut scored_moves = Vec::with_capacity(moves.len());

    for &m in moves {
        board.make_move(m);
        let child_wdl = probe_wdl(board, tablebase)
            .map(|w| w.negate())
            .unwrap_or(Wdl::Draw);
        board.unmake_move(m);

        scored_moves.push((m, child_wdl));
    }

    match root_wdl {
        Wdl::Win | Wdl::CursedWin => {
            let best = scored_moves
                .iter()
                .map(|(_, w)| *w as u8)
                .max()
                .unwrap_or(0);
            scored_moves
                .into_iter()
                .filter(|(_, w)| *w as u8 >= best)
                .collect()
        }
        Wdl::Draw => scored_moves
            .into_iter()
            .filter(|(_, w)| *w as u8 >= Wdl::Draw as u8)
            .collect(),
        Wdl::BlessedLoss | Wdl::Loss => scored_moves,
    }
}

/// Convert a WDL probe result to a search score.
pub fn wdl_to_score(wdl: Wdl, ply: u32) -> i32 {
    const TB_WIN: i32 = 30000 - 128 - 1;

    match wdl {
        Wdl::Win => TB_WIN - ply as i32,
        Wdl::CursedWin => 1,
        Wdl::Draw => 0,
        Wdl::BlessedLoss => -1,
        Wdl::Loss => -(TB_WIN - ply as i32),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn wdl_ordering() {
        assert!(Wdl::Win > Wdl::CursedWin);
        assert!(Wdl::CursedWin > Wdl::Draw);
        assert!(Wdl::Draw > Wdl::BlessedLoss);
        assert!(Wdl::BlessedLoss > Wdl::Loss);
    }

    #[test]
    fn wdl_negate() {
        assert_eq!(Wdl::Win.negate(), Wdl::Loss);
        assert_eq!(Wdl::Loss.negate(), Wdl::Win);
        assert_eq!(Wdl::Draw.negate(), Wdl::Draw);
        assert_eq!(Wdl::CursedWin.negate(), Wdl::BlessedLoss);
        assert_eq!(Wdl::BlessedLoss.negate(), Wdl::CursedWin);
    }

    #[test]
    fn wdl_from_raw() {
        assert_eq!(Wdl::from_raw(0), Some(Wdl::Loss));
        assert_eq!(Wdl::from_raw(2), Some(Wdl::Draw));
        assert_eq!(Wdl::from_raw(4), Some(Wdl::Win));
        assert_eq!(Wdl::from_raw(5), None);
    }

    #[test]
    fn wdl_to_score_values() {
        let win_score = wdl_to_score(Wdl::Win, 0);
        let loss_score = wdl_to_score(Wdl::Loss, 0);
        let draw_score = wdl_to_score(Wdl::Draw, 0);

        assert!(win_score > 0);
        assert!(loss_score < 0);
        assert_eq!(draw_score, 0);
        assert_eq!(win_score, -loss_score);
    }

    #[test]
    fn wdl_to_score_ply_adjustment() {
        let score_ply0 = wdl_to_score(Wdl::Win, 0);
        let score_ply5 = wdl_to_score(Wdl::Win, 5);
        assert!(score_ply0 > score_ply5);
    }

    #[test]
    fn dtz_wdl_consistency() {
        assert_eq!(Dtz(5).wdl(), Wdl::Win);
        assert_eq!(Dtz(-3).wdl(), Wdl::Loss);
        assert_eq!(Dtz(0).wdl(), Wdl::Draw);
    }

    #[test]
    fn probe_wdl_with_castling_returns_none() {
        let mut board = Board::from_fen(Board::STARTPOS_FEN).unwrap();
        let tb = Tablebase::new("");
        assert_eq!(probe_wdl(&mut board, &tb), None);
    }

    #[test]
    fn probe_wdl_too_many_pieces_returns_none() {
        let mut board = Board::from_fen("rnbqk3/pppp4/8/8/8/8/PPPP4/RNBQK3 w - - 0 1").unwrap();
        let tb = Tablebase::new("");
        assert_eq!(probe_wdl(&mut board, &tb), None);
    }

    #[test]
    fn probe_wdl_empty_tablebase_returns_none() {
        let mut board = Board::from_fen("4k3/8/8/8/8/8/8/4K2R w - - 0 1").unwrap();
        let tb = Tablebase::new("");
        assert_eq!(probe_wdl(&mut board, &tb), None);
    }

    #[test]
    fn cursed_win_blessed_loss_scores() {
        assert_eq!(wdl_to_score(Wdl::CursedWin, 0), 1);
        assert_eq!(wdl_to_score(Wdl::BlessedLoss, 0), -1);
    }

    #[test]
    fn probe_krk_simple() {
        use crate::embedded::extract_embedded_tables;

        let tb_path = extract_embedded_tables().expect("failed to extract tables");
        let tb = crate::Tablebase::new(tb_path.to_str().unwrap());

        let has_table = tb.has_wdl("KQvK");
        assert!(has_table);

        let result = tb.get_wdl("KQvK");
        assert!(result.is_ok());

        // KQvK: White king e2, queen d2, black king e8 — white wins
        let mut board = Board::from_fen("4k3/8/8/8/8/8/3QK3/8 w - - 0 1").unwrap();
        let result = probe_wdl(&mut board, &tb);
        assert!(result.is_some(), "probe_wdl returned None for KQvK");
    }

    #[test]
    fn probe_real_table_krvkr() {
        let tb = crate::Tablebase::new("/tmp/rage-syzygy");

        if !tb.has_wdl("KRvKR") {
            return;
        }

        let result = tb.get_wdl("KRvKR");
        assert!(result.is_ok(), "Failed to load KRvKR table");

        let mut board = Board::from_fen("3rk3/8/8/8/8/8/R3K3/8 w - - 0 1").unwrap();
        let result = probe_wdl(&mut board, &tb);
        assert!(result.is_some(), "probe_wdl returned None for KRvKR");
    }

    #[test]
    fn probe_real_table_kbvkb() {
        let tb = crate::Tablebase::new("/tmp/rage-syzygy");

        if !tb.has_wdl("KBvKB") {
            return;
        }

        let result = tb.get_wdl("KBvKB");
        assert!(result.is_ok(), "Failed to load KBvKB table");

        let mut board = Board::from_fen("B6k/b7/8/8/8/8/8/K7 w - - 0 1").unwrap();
        let result = probe_wdl(&mut board, &tb);
        assert!(result.is_some(), "probe_wdl returned None for KBvKB");
    }

    #[test]
    fn probe_real_table_knvkn() {
        let tb = crate::Tablebase::new("/tmp/rage-syzygy");

        if !tb.has_wdl("KNvKN") {
            return;
        }

        let result = tb.get_wdl("KNvKN");
        assert!(result.is_ok(), "Failed to load KNvKN table");

        let mut board = Board::from_fen("6k1/8/6n1/8/8/1N6/8/K7 w - - 0 1").unwrap();
        let result = probe_wdl(&mut board, &tb);
        assert!(result.is_some(), "probe_wdl returned None for KNvKN");
    }
}
