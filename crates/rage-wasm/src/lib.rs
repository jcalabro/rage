#![allow(clippy::all)]

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use wasm_bindgen::prelude::*;

use rage_core::{Board, Move, Square};
use rage_nnue::{NnueNetwork, NnueWeights};
use rage_search::eval::MATE_SCORE;
use rage_search::search::{InfoCallback, Searcher};
use rage_search::time::SearchLimits;
use rage_search::tt::TranspositionTable;

/// Wrapper around `js_sys::Function` that implements `Send`.
///
// SAFETY: wasm32 is single-threaded — there are no other threads to send to.
struct SendFunction(js_sys::Function);

// SAFETY: wasm is single-threaded; this value never crosses a thread boundary.
unsafe impl Send for SendFunction {}

impl SendFunction {
    fn call1(&self, arg: &JsValue) -> Result<JsValue, JsValue> {
        self.0.call1(&JsValue::NULL, arg)
    }
}

// ---------------------------------------------------------------------------
// RageBoard — standalone chess library (no search)
// ---------------------------------------------------------------------------

/// A chess board for move generation, validation, and game state queries.
/// No search functionality — use `RageEngine` for that.
#[wasm_bindgen]
pub struct RageBoard {
    board: Board,
    /// Undo stack: stores moves played so `undo_move()` can work.
    move_stack: Vec<Move>,
}

#[wasm_bindgen]
impl RageBoard {
    /// Create a new board at the standard starting position.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            board: Board::starting_position(),
            move_stack: Vec::new(),
        }
    }

    /// Create a board from a FEN string.
    #[wasm_bindgen(js_name = fromFen)]
    pub fn from_fen(fen: &str) -> Result<RageBoard, JsError> {
        let board = Board::from_fen(fen).map_err(|e| JsError::new(&format!("{e}")))?;
        Ok(Self {
            board,
            move_stack: Vec::new(),
        })
    }

    /// Returns the FEN string of the current position.
    pub fn fen(&self) -> String {
        self.board.to_fen()
    }

    /// Returns an ASCII representation of the board.
    pub fn ascii(&self) -> String {
        format!("{}", self.board)
    }

    /// Play a move in UCI notation (e.g. "e2e4", "e7e8q").
    /// Returns an error if the move is illegal.
    #[wasm_bindgen(js_name = makeMove)]
    pub fn make_move(&mut self, uci: &str) -> Result<(), JsError> {
        let m = self
            .board
            .parse_uci_move(uci)
            .map_err(|e| JsError::new(&format!("{e}")))?;
        self.board.make_move(m);
        self.move_stack.push(m);
        Ok(())
    }

    /// Undo the last move. Returns false if there is no move to undo.
    #[wasm_bindgen(js_name = undoMove)]
    pub fn undo_move(&mut self) -> bool {
        if let Some(m) = self.move_stack.pop() {
            self.board.unmake_move(m);
            true
        } else {
            false
        }
    }

    /// Returns an array of legal moves in UCI notation.
    #[wasm_bindgen(js_name = legalMoves)]
    pub fn legal_moves(&mut self) -> Vec<String> {
        let moves = self.board.generate_legal_moves();
        let mut result = Vec::with_capacity(moves.len());
        for i in 0..moves.len() {
            result.push(moves[i].to_string());
        }
        result
    }

    /// Returns true if the current side is in checkmate.
    #[wasm_bindgen(js_name = isCheckmate)]
    pub fn is_checkmate(&mut self) -> bool {
        self.board.is_checkmate()
    }

    /// Returns true if the current side is in stalemate.
    #[wasm_bindgen(js_name = isStalemate)]
    pub fn is_stalemate(&mut self) -> bool {
        self.board.is_stalemate()
    }

    /// Returns true if the position is a draw (fifty-move, insufficient material, or repetition).
    #[wasm_bindgen(js_name = isDraw)]
    pub fn is_draw(&self) -> bool {
        self.board.is_fifty_move_draw()
            || self.board.is_insufficient_material()
            || self.board.is_repetition()
    }

    /// Returns true if the game is over (checkmate, stalemate, or draw).
    #[wasm_bindgen(js_name = isGameOver)]
    pub fn is_game_over(&mut self) -> bool {
        self.is_checkmate() || self.is_stalemate() || self.is_draw()
    }

    /// Returns true if the current side to move is in check.
    #[wasm_bindgen(js_name = inCheck)]
    pub fn in_check(&self) -> bool {
        self.board.in_check()
    }

    /// Returns "w" or "b" for the side to move.
    #[wasm_bindgen(js_name = sideToMove)]
    pub fn side_to_move(&self) -> String {
        match self.board.side_to_move() {
            rage_core::Color::White => "w".to_string(),
            rage_core::Color::Black => "b".to_string(),
        }
    }

    /// Returns the piece on the given square (e.g. "P", "n", "K"), or empty string if empty.
    /// Square is in algebraic notation (e.g. "e4").
    #[wasm_bindgen(js_name = pieceOn)]
    pub fn piece_on(&self, sq: &str) -> Result<String, JsError> {
        let square = parse_square(sq)?;
        match self.board.piece_on(square) {
            Some(p) => Ok(p.to_string()),
            None => Ok(String::new()),
        }
    }

    /// Returns the halfmove clock (for 50-move rule).
    #[wasm_bindgen(js_name = halfmoveClock)]
    pub fn halfmove_clock(&self) -> u16 {
        self.board.halfmove_clock()
    }

    /// Returns the fullmove number.
    #[wasm_bindgen(js_name = fullmoveNumber)]
    pub fn fullmove_number(&self) -> u16 {
        self.board.fullmove_number()
    }

    /// Returns true if the given UCI move is a capture.
    #[wasm_bindgen(js_name = isCapture)]
    pub fn is_capture(&mut self, uci: &str) -> Result<bool, JsError> {
        let m = self
            .board
            .parse_uci_move(uci)
            .map_err(|e| JsError::new(&format!("{e}")))?;
        Ok(self.board.is_capture(m))
    }

    /// Returns the Zobrist hash as a hex string.
    #[wasm_bindgen(js_name = zobristHash)]
    pub fn zobrist_hash(&self) -> String {
        format!("{:016x}", self.board.zobrist_hash())
    }

    /// Run perft (performance test) to the given depth. Returns the node count.
    pub fn perft(&mut self, depth: u32) -> f64 {
        rage_core::perft(&mut self.board, depth) as f64
    }

    /// Returns the castling rights as a string (e.g. "KQkq", "Kq", "-").
    #[wasm_bindgen(js_name = castlingRights)]
    pub fn castling_rights(&self) -> String {
        format!("{}", self.board.castling_rights())
    }

    /// Returns the number of half-moves (plies) played from the starting position.
    #[wasm_bindgen(js_name = plyCount)]
    pub fn ply_count(&self) -> usize {
        self.move_stack.len()
    }

    /// Deep clone this board, preserving undo history.
    #[wasm_bindgen(js_name = clone)]
    pub fn clone_board(&self) -> Self {
        Self {
            board: self.board.clone(),
            move_stack: self.move_stack.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// RageEngine — search-capable engine
// ---------------------------------------------------------------------------

/// A chess engine with NNUE evaluation and alpha-beta search.
#[wasm_bindgen]
pub struct RageEngine {
    board: Board,
    searcher: Searcher,
    network: Option<Arc<NnueNetwork>>,
    info_callback: Option<js_sys::Function>,
}

#[wasm_bindgen]
impl RageEngine {
    /// Create a new engine with the given hash table size in MB.
    #[wasm_bindgen(constructor)]
    pub fn new(hash_mb: usize) -> Self {
        let tt = Arc::new(TranspositionTable::new(hash_mb));
        let searcher = Searcher::new(tt);
        Self {
            board: Board::starting_position(),
            searcher,
            network: None,
            info_callback: None,
        }
    }

    /// Set the position from a FEN string.
    #[wasm_bindgen(js_name = setPosition)]
    pub fn set_position(&mut self, fen: &str) -> Result<(), JsError> {
        self.board = Board::from_fen(fen).map_err(|e| JsError::new(&format!("{e}")))?;
        Ok(())
    }

    /// Reset to the starting position.
    #[wasm_bindgen(js_name = setStartPosition)]
    pub fn set_start_position(&mut self) {
        self.board = Board::starting_position();
    }

    /// Play a move in UCI notation.
    #[wasm_bindgen(js_name = makeMove)]
    pub fn make_move(&mut self, uci: &str) -> Result<(), JsError> {
        let m = self
            .board
            .parse_uci_move(uci)
            .map_err(|e| JsError::new(&format!("{e}")))?;
        self.board.make_move(m);
        Ok(())
    }

    /// Load NNUE weights from raw bytes.
    #[wasm_bindgen(js_name = loadWeights)]
    pub fn load_weights(&mut self, data: &[u8]) -> Result<(), JsError> {
        let weights = NnueWeights::from_bytes(data).map_err(|e| JsError::new(&format!("{e}")))?;
        let network = Arc::new(NnueNetwork::new(weights));
        self.searcher.set_network(Some(Arc::clone(&network)));
        self.network = Some(network);
        Ok(())
    }

    /// Register a callback that is invoked after each search iteration.
    /// The callback receives a single object argument with fields:
    /// `{ depth, seldepth, score, nodes, timeMs, pv, hashfull, tbhits }`.
    /// Pass `null` or call `clearSearchInfo()` to remove the callback.
    #[wasm_bindgen(js_name = onSearchInfo)]
    pub fn on_search_info(&mut self, callback: js_sys::Function) {
        self.info_callback = Some(callback);
    }

    /// Remove the search info callback.
    #[wasm_bindgen(js_name = clearSearchInfo)]
    pub fn clear_search_info(&mut self) {
        self.info_callback = None;
    }

    /// Search to a fixed depth. Returns a `WasmSearchResult`.
    #[wasm_bindgen(js_name = searchDepth)]
    pub fn search_depth(&mut self, depth: u32) -> WasmSearchResult {
        let limits = SearchLimits {
            depth: Some(depth),
            ..Default::default()
        };
        self.run_search(&limits)
    }

    /// Search for a fixed time in milliseconds. Returns a `WasmSearchResult`.
    #[wasm_bindgen(js_name = searchTime)]
    pub fn search_time(&mut self, ms: u64) -> WasmSearchResult {
        let limits = SearchLimits {
            movetime: Some(ms),
            ..Default::default()
        };
        self.run_search(&limits)
    }

    /// Search with a node limit. Returns a `WasmSearchResult`.
    #[wasm_bindgen(js_name = searchNodes)]
    pub fn search_nodes(&mut self, nodes: f64) -> WasmSearchResult {
        let limits = SearchLimits {
            nodes: Some(nodes as u64),
            ..Default::default()
        };
        self.run_search(&limits)
    }

    /// Static NNUE evaluation of the current position.
    /// Returns `None` (as JS `undefined`) if no network is loaded.
    pub fn evaluate(&mut self) -> Option<i32> {
        self.searcher.evaluate_nnue(&self.board)
    }

    /// Returns an array of legal moves in UCI notation.
    #[wasm_bindgen(js_name = legalMoves)]
    pub fn legal_moves(&mut self) -> Vec<String> {
        let moves = self.board.generate_legal_moves();
        let mut result = Vec::with_capacity(moves.len());
        for i in 0..moves.len() {
            result.push(moves[i].to_string());
        }
        result
    }

    /// Returns true if the game is over.
    #[wasm_bindgen(js_name = isGameOver)]
    pub fn is_game_over(&mut self) -> bool {
        self.board.is_checkmate()
            || self.board.is_stalemate()
            || self.board.is_fifty_move_draw()
            || self.board.is_insufficient_material()
            || self.board.is_repetition()
    }

    /// Returns the FEN of the current position.
    pub fn fen(&self) -> String {
        self.board.to_fen()
    }

    /// Clear the transposition table and history for a new game.
    #[wasm_bindgen(js_name = newGame)]
    pub fn new_game(&mut self) {
        self.searcher.clear_tables();
        self.board = Board::starting_position();
    }
}

impl RageEngine {
    /// Shared search logic: builds info callback from stored JS function and runs search.
    fn run_search(&mut self, limits: &SearchLimits) -> WasmSearchResult {
        let stop = Arc::new(AtomicBool::new(false));
        let cb = self.make_info_callback();
        let result = self.searcher.search(&mut self.board, limits, stop, cb);
        WasmSearchResult::from_result(&result)
    }

    /// Convert the stored JS callback into a Rust `InfoCallback`, if set.
    fn make_info_callback(&self) -> Option<InfoCallback> {
        let js_fn = SendFunction(self.info_callback.clone()?);
        Some(Box::new(move |info| {
            let obj = js_sys::Object::new();
            let _ = js_sys::Reflect::set(&obj, &"depth".into(), &(info.depth).into());
            let _ = js_sys::Reflect::set(&obj, &"seldepth".into(), &(info.seldepth).into());
            let _ = js_sys::Reflect::set(&obj, &"score".into(), &(info.score).into());
            let _ = js_sys::Reflect::set(&obj, &"nodes".into(), &(info.nodes as f64).into());
            let _ = js_sys::Reflect::set(&obj, &"timeMs".into(), &(info.time_ms as f64).into());
            let pv: Vec<String> = info.pv.iter().map(|m| m.to_string()).collect();
            let _ = js_sys::Reflect::set(&obj, &"pv".into(), &pv.join(" ").into());
            let _ = js_sys::Reflect::set(&obj, &"hashfull".into(), &(info.hashfull).into());
            let _ = js_sys::Reflect::set(&obj, &"tbhits".into(), &(info.tbhits as f64).into());
            let _ = js_sys::Reflect::set(&obj, &"multipv".into(), &(info.multipv as u32).into());
            let _ = js_fn.call1(&obj);
        }))
    }
}

// ---------------------------------------------------------------------------
// WasmSearchResult
// ---------------------------------------------------------------------------

/// Search result returned to JavaScript.
#[wasm_bindgen]
pub struct WasmSearchResult {
    best_move: String,
    score: i32,
    depth: u32,
    nodes: f64,
    pv: String,
    is_mate: bool,
    mate_in: i32,
}

#[wasm_bindgen]
impl WasmSearchResult {
    /// Best move in UCI notation (e.g. "e2e4").
    #[wasm_bindgen(getter, js_name = bestMove)]
    pub fn best_move(&self) -> String {
        self.best_move.clone()
    }

    /// Score in centipawns (from the side to move's perspective).
    #[wasm_bindgen(getter)]
    pub fn score(&self) -> i32 {
        self.score
    }

    /// Search depth completed.
    #[wasm_bindgen(getter)]
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// Total nodes searched (as f64 for JS compatibility).
    #[wasm_bindgen(getter)]
    pub fn nodes(&self) -> f64 {
        self.nodes
    }

    /// Principal variation (space-separated UCI moves).
    #[wasm_bindgen(getter)]
    pub fn pv(&self) -> String {
        self.pv.clone()
    }

    /// True if the score is a forced mate.
    #[wasm_bindgen(getter, js_name = isMate)]
    pub fn is_mate(&self) -> bool {
        self.is_mate
    }

    /// Moves to mate (positive = we mate, negative = we get mated). 0 if not a mate score.
    #[wasm_bindgen(getter, js_name = mateIn)]
    pub fn mate_in(&self) -> i32 {
        self.mate_in
    }
}

impl WasmSearchResult {
    fn from_result(result: &rage_search::search::SearchResult) -> Self {
        let score = result.score;
        let is_mate = score.abs() >= MATE_SCORE - 128;
        let mate_in = if is_mate {
            let plies_to_mate = MATE_SCORE - score.abs();
            let moves = (plies_to_mate + 1) / 2;
            if score > 0 { moves } else { -moves }
        } else {
            0
        };

        let pv: Vec<String> = result.pv.iter().map(|m| m.to_string()).collect();

        Self {
            best_move: result.best_move.to_string(),
            score,
            depth: result.depth,
            nodes: result.nodes as f64,
            pv: pv.join(" "),
            is_mate,
            mate_in,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_square(s: &str) -> Result<Square, JsError> {
    if s.len() != 2 {
        return Err(JsError::new(&format!("invalid square: '{s}'")));
    }
    let bytes = s.as_bytes();
    let file = bytes[0].wrapping_sub(b'a');
    let rank = bytes[1].wrapping_sub(b'1');
    if file >= 8 || rank >= 8 {
        return Err(JsError::new(&format!("invalid square: '{s}'")));
    }
    Ok(Square::from_file_rank(
        rage_core::File::new(file),
        rage_core::Rank::new(rank),
    ))
}
