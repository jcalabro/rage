use rage_core::{Board, CastlingRights, Color, PieceType, Square};

/// Size of a single training entry in bytes.
pub const ENTRY_SIZE: usize = 40;

/// Maximum number of active features per perspective.
pub const MAX_ACTIVE_FEATURES: usize = 32;

/// Piece encoding for packed board (4 bits per square):
/// 0=empty, 1-6=White PNBRQK, 7-12=Black PNBRQK.
fn piece_to_nibble(color: Color, pt: PieceType) -> u8 {
    let base = match color {
        Color::White => 1,
        Color::Black => 7,
    };
    base + pt.index() as u8
}

/// Decode a nibble back to (color, piece_type), or None for empty.
fn nibble_to_piece(nibble: u8) -> Option<(Color, PieceType)> {
    match nibble {
        0 => None,
        1..=6 => Some((Color::White, PieceType::ALL[(nibble - 1) as usize])),
        7..=12 => Some((Color::Black, PieceType::ALL[(nibble - 7) as usize])),
        _ => None,
    }
}

/// WDL result from the side-to-move's perspective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WdlResult {
    Loss = 0,
    Draw = 1,
    Win = 2,
}

impl WdlResult {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Loss),
            1 => Some(Self::Draw),
            2 => Some(Self::Win),
            _ => None,
        }
    }
}

/// A single training entry: position + search score + game result.
///
/// Binary format (40 bytes):
/// ```text
/// [0..32]  : Packed board (4 bits/square, 2 squares/byte, LERF order)
/// [32]     : Side to move (0=White, 1=Black)
/// [33]     : Castling rights (KQkq bits)
/// [34]     : En passant file (0-7, or 0xFF=none)
/// [35..37] : Search score (i16, little-endian)
/// [37]     : WDL result (0=loss, 1=draw, 2=win from STM perspective)
/// [38..40] : Move played (u16 raw encoding, little-endian)
/// ```
#[derive(Debug, Clone)]
pub struct TrainingEntry {
    pub packed_board: [u8; 32],
    pub stm: Color,
    pub castling: CastlingRights,
    pub ep_file: Option<u8>,
    pub score: i16,
    pub wdl: WdlResult,
    pub move_played: u16,
}

impl TrainingEntry {
    /// Pack a board position into a training entry.
    pub fn from_board(board: &Board, score: i16, wdl: WdlResult, move_played: u16) -> Self {
        let mut packed_board = [0u8; 32];

        for sq_idx in 0..64usize {
            let sq = Square::new(sq_idx as u8);
            let nibble = board
                .piece_on(sq)
                .map_or(0, |p| piece_to_nibble(p.color, p.piece_type));

            let byte_idx = sq_idx / 2;
            if sq_idx % 2 == 0 {
                packed_board[byte_idx] |= nibble;
            } else {
                packed_board[byte_idx] |= nibble << 4;
            }
        }

        let ep_file = board.en_passant_square().map(|sq| sq.file().index() as u8);

        Self {
            packed_board,
            stm: board.side_to_move(),
            castling: board.castling_rights(),
            ep_file,
            score,
            wdl,
            move_played,
        }
    }

    /// Serialize to 40-byte binary format.
    pub fn to_bytes(&self) -> [u8; ENTRY_SIZE] {
        let mut buf = [0u8; ENTRY_SIZE];
        buf[0..32].copy_from_slice(&self.packed_board);
        buf[32] = self.stm.index() as u8;
        buf[33] = self.castling.bits();
        buf[34] = self.ep_file.unwrap_or(0xFF);
        buf[35..37].copy_from_slice(&self.score.to_le_bytes());
        buf[37] = self.wdl as u8;
        buf[38..40].copy_from_slice(&self.move_played.to_le_bytes());
        buf
    }

    /// Deserialize from 40-byte binary format.
    pub fn from_bytes(buf: &[u8; ENTRY_SIZE]) -> Option<Self> {
        let mut packed_board = [0u8; 32];
        packed_board.copy_from_slice(&buf[0..32]);

        let stm = match buf[32] {
            0 => Color::White,
            1 => Color::Black,
            _ => return None,
        };

        let castling = CastlingRights::new(buf[33]);
        let ep_file = if buf[34] == 0xFF {
            None
        } else if buf[34] < 8 {
            Some(buf[34])
        } else {
            return None;
        };

        let score = i16::from_le_bytes([buf[35], buf[36]]);
        let wdl = WdlResult::from_u8(buf[37])?;
        let move_played = u16::from_le_bytes([buf[38], buf[39]]);

        Some(Self {
            packed_board,
            stm,
            castling,
            ep_file,
            score,
            wdl,
            move_played,
        })
    }

    /// Unpack the board from the packed representation.
    /// Returns (square_index, color, piece_type) for each occupied square.
    pub fn pieces(&self) -> impl Iterator<Item = (usize, Color, PieceType)> + '_ {
        (0..64).filter_map(move |sq_idx| {
            let byte_idx = sq_idx / 2;
            let nibble = if sq_idx % 2 == 0 {
                self.packed_board[byte_idx] & 0x0F
            } else {
                (self.packed_board[byte_idx] >> 4) & 0x0F
            };
            nibble_to_piece(nibble).map(|(color, pt)| (sq_idx, color, pt))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_starting_position() {
        let board = Board::starting_position();
        let entry = TrainingEntry::from_board(&board, 50, WdlResult::Win, 0x0C44);
        let bytes = entry.to_bytes();
        let decoded = TrainingEntry::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.stm, Color::White);
        assert_eq!(decoded.score, 50);
        assert_eq!(decoded.wdl, WdlResult::Win);
        assert_eq!(decoded.move_played, 0x0C44);

        // Verify all 32 pieces are present
        let piece_count = decoded.pieces().count();
        assert_eq!(piece_count, 32, "starting position should have 32 pieces");
    }

    #[test]
    fn roundtrip_empty_board() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let entry = TrainingEntry::from_board(&board, -100, WdlResult::Loss, 0);
        let bytes = entry.to_bytes();
        let decoded = TrainingEntry::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.score, -100);
        assert_eq!(decoded.wdl, WdlResult::Loss);

        let piece_count = decoded.pieces().count();
        assert_eq!(piece_count, 2, "KvK should have 2 pieces");
    }

    #[test]
    fn entry_size() {
        let board = Board::starting_position();
        let entry = TrainingEntry::from_board(&board, 0, WdlResult::Draw, 0);
        let bytes = entry.to_bytes();
        assert_eq!(bytes.len(), 40);
    }
}
