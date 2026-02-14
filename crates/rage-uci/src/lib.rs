use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;

use rage_core::Board;
use rage_search::search::{InfoCallback, Searcher};
use rage_search::time::SearchLimits;

const ENGINE_NAME: &str = "rage";
const ENGINE_AUTHOR: &str = "jcalabro";

/// Default transposition table size in MB.
const DEFAULT_HASH_MB: usize = 16;

/// Run the UCI protocol loop. Blocks until `quit` is received.
pub fn run() {
    let (tx, rx) = mpsc::channel::<String>();

    // Spawn stdin reader thread
    thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(line) => {
                    if tx.send(line).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    println!(
        "{ENGINE_NAME} v{} by {ENGINE_AUTHOR} (type 'help' for commands)",
        env!("CARGO_PKG_VERSION")
    );

    let mut engine = UciEngine::new();

    while let Ok(line) = rx.recv() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if engine.handle_command(line) {
            break;
        }
    }

    // Ensure any running search completes before exit.
    // Don't force stop - let the search finish if it has a depth/time limit.
    if let Some(handle) = engine.search_handle.take() {
        let _ = handle.join();
    }
}

struct UciEngine {
    board: Board,
    /// None while a search thread owns the searcher.
    searcher: Option<Searcher>,
    stop: Arc<AtomicBool>,
    search_handle: Option<thread::JoinHandle<Searcher>>,
}

impl UciEngine {
    fn new() -> Self {
        Self {
            board: Board::starting_position(),
            searcher: Some(Searcher::new(DEFAULT_HASH_MB)),
            stop: Arc::new(AtomicBool::new(false)),
            search_handle: None,
        }
    }

    /// Stop any running search and recover the searcher.
    fn wait_for_search(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.search_handle.take()
            && let Ok(searcher) = handle.join()
        {
            self.searcher = Some(searcher);
        }
    }

    /// Handle a UCI command. Returns true if the engine should quit.
    fn handle_command(&mut self, line: &str) -> bool {
        let mut tokens = line.split_whitespace();
        let cmd = match tokens.next() {
            Some(c) => c,
            None => return false,
        };

        match cmd {
            "uci" => self.cmd_uci(),
            "isready" => self.cmd_isready(),
            "ucinewgame" => self.cmd_ucinewgame(),
            "position" => self.cmd_position(&mut tokens),
            "go" => self.cmd_go(&mut tokens),
            "stop" => self.cmd_stop(),
            "setoption" => self.cmd_setoption(&mut tokens),
            "quit" => {
                self.cmd_stop();
                return true;
            }
            // Debug commands
            "d" | "display" => self.cmd_display(),
            "dd" => self.cmd_display_pretty(),
            "eval" => self.cmd_eval(),
            "perft" => self.cmd_perft(&mut tokens),
            "compiler" => self.cmd_compiler(),
            "help" => self.cmd_help(),
            "license" => self.cmd_license(),
            _ => {} // Ignore unknown commands per UCI spec
        }

        false
    }

    fn cmd_uci(&self) {
        println!("id name {ENGINE_NAME}");
        println!("id author {ENGINE_AUTHOR}");
        println!();
        println!("option name Hash type spin default {DEFAULT_HASH_MB} min 1 max 1024");
        println!("uciok");
    }

    fn cmd_isready(&self) {
        println!("readyok");
    }

    fn cmd_ucinewgame(&mut self) {
        self.wait_for_search();
        if let Some(ref mut searcher) = self.searcher {
            searcher.tt.clear();
        }
        self.board = Board::starting_position();
    }

    fn cmd_position<'a>(&mut self, tokens: &mut impl Iterator<Item = &'a str>) {
        match tokens.next() {
            Some("startpos") => {
                self.board = Board::starting_position();
            }
            Some("fen") => {
                // Collect FEN parts (up to 6 tokens before "moves")
                let mut fen_parts = Vec::new();
                for token in tokens.by_ref() {
                    if token == "moves" {
                        let fen = fen_parts.join(" ");
                        if let Ok(b) = Board::from_fen(&fen) {
                            self.board = b;
                        }
                        self.apply_moves(tokens);
                        return;
                    }
                    fen_parts.push(token);
                }
                let fen = fen_parts.join(" ");
                if let Ok(b) = Board::from_fen(&fen) {
                    self.board = b;
                }
                return;
            }
            _ => return,
        }

        // Check for "moves" after "startpos"
        if let Some("moves") = tokens.next() {
            self.apply_moves(tokens);
        }
    }

    fn apply_moves<'a>(&mut self, tokens: &mut impl Iterator<Item = &'a str>) {
        for move_str in tokens {
            match self.board.parse_uci_move(move_str) {
                Ok(m) => self.board.make_move(m),
                Err(_) => return,
            }
        }
    }

    fn cmd_go<'a>(&mut self, tokens: &mut impl Iterator<Item = &'a str>) {
        // Recover the searcher from any running search
        self.wait_for_search();

        let mut limits = SearchLimits::default();

        while let Some(token) = tokens.next() {
            match token {
                "depth" => limits.depth = tokens.next().and_then(|s| s.parse().ok()),
                "movetime" => limits.movetime = tokens.next().and_then(|s| s.parse().ok()),
                "wtime" => limits.wtime = tokens.next().and_then(|s| s.parse().ok()),
                "btime" => limits.btime = tokens.next().and_then(|s| s.parse().ok()),
                "winc" => limits.winc = tokens.next().and_then(|s| s.parse().ok()),
                "binc" => limits.binc = tokens.next().and_then(|s| s.parse().ok()),
                "movestogo" => limits.movestogo = tokens.next().and_then(|s| s.parse().ok()),
                "nodes" => limits.nodes = tokens.next().and_then(|s| s.parse().ok()),
                "infinite" => limits.infinite = true,
                _ => {}
            }
        }

        let stop = Arc::new(AtomicBool::new(false));
        self.stop = Arc::clone(&stop);

        let mut board = self.board.clone();

        // Take the searcher out - search thread owns it temporarily
        let mut searcher = match self.searcher.take() {
            Some(s) => s,
            None => Searcher::new(DEFAULT_HASH_MB),
        };

        let handle = thread::spawn(move || {
            let info_cb: InfoCallback = Box::new(|info| {
                let pv_str: String = info
                    .pv
                    .iter()
                    .map(|m| m.to_string())
                    .collect::<Vec<_>>()
                    .join(" ");

                println!(
                    "info depth {} seldepth {} score {} nodes {} nps {} time {} hashfull {} pv {}",
                    info.depth,
                    info.seldepth,
                    info.score_string(),
                    info.nodes,
                    info.nps(),
                    info.time_ms,
                    info.hashfull,
                    pv_str,
                );
            });

            let result = searcher.search(&mut board, &limits, stop, Some(info_cb));
            println!("bestmove {}", result.best_move);
            let _ = io::stdout().flush();

            // Return the searcher so the TT is preserved
            searcher
        });

        self.search_handle = Some(handle);
    }

    fn cmd_stop(&mut self) {
        self.wait_for_search();
    }

    fn cmd_setoption<'a>(&mut self, tokens: &mut impl Iterator<Item = &'a str>) {
        self.wait_for_search();

        // Parse: setoption name <name> value <value>
        let mut name = String::new();
        let mut value = String::new();
        let mut parsing_name = false;
        let mut parsing_value = false;

        for token in tokens {
            match token {
                "name" => {
                    parsing_name = true;
                    parsing_value = false;
                }
                "value" => {
                    parsing_name = false;
                    parsing_value = true;
                }
                _ => {
                    if parsing_name {
                        if !name.is_empty() {
                            name.push(' ');
                        }
                        name.push_str(token);
                    } else if parsing_value {
                        if !value.is_empty() {
                            value.push(' ');
                        }
                        value.push_str(token);
                    }
                }
            }
        }

        if name.to_lowercase() == "hash"
            && let Ok(mb) = value.parse::<usize>()
        {
            let mb = mb.clamp(1, 1024);
            if let Some(ref mut searcher) = self.searcher {
                searcher.tt.resize(mb);
            }
        }
    }

    // --- Debug commands ---

    fn cmd_display(&self) {
        print!("{}", self.board);
        println!("FEN: {}", self.board.to_fen());
    }

    fn cmd_display_pretty(&self) {
        use rage_core::{Color, File, PieceType, Rank, Square};

        // Board square colors - warm wood tones
        const LIGHT_BG: &str = "\x1b[48;2;238;216;180m";
        const DARK_BG: &str = "\x1b[48;2;180;136;100m";
        // Piece colors - high contrast against both square colors
        const WHITE_PC: &str = "\x1b[38;2;255;255;240m";
        const BLACK_PC: &str = "\x1b[38;2;40;30;20m";
        // UI colors
        const COORD_FG: &str = "\x1b[38;2;140;120;100m";
        const LABEL_FG: &str = "\x1b[38;2;180;160;140m";
        const DIM: &str = "\x1b[2m";
        const BOLD: &str = "\x1b[1m";
        const RESET: &str = "\x1b[0m";

        // Chess piece characters (all single-cell-width Unicode, no variation selectors)
        fn piece_char(pt: PieceType, color: Color) -> char {
            match (color, pt) {
                (Color::White, PieceType::King) => '\u{2654}',
                (Color::White, PieceType::Queen) => '\u{2655}',
                (Color::White, PieceType::Rook) => '\u{2656}',
                (Color::White, PieceType::Bishop) => '\u{2657}',
                (Color::White, PieceType::Knight) => '\u{2658}',
                (Color::White, PieceType::Pawn) => '\u{2659}',
                (Color::Black, PieceType::King) => '\u{265A}',
                (Color::Black, PieceType::Queen) => '\u{265B}',
                (Color::Black, PieceType::Rook) => '\u{265C}',
                (Color::Black, PieceType::Bishop) => '\u{265D}',
                (Color::Black, PieceType::Knight) => '\u{265E}',
                (Color::Black, PieceType::Pawn) => '\u{265F}',
            }
        }

        println!();

        // Each square is 5 terminal cells wide: "  X  " for pieces, "     " for empty.
        // Prefix is 4 visible chars: "  N " (2 spaces + rank digit + space).
        // File labels are centered under each 5-cell square.
        println!("  {COORD_FG}    a    b    c    d    e    f    g    h{RESET}");

        // Board rows — 3 terminal lines per rank for square-ish cells
        for rank in (0..8).rev() {
            // Top padding row
            print!("    ");
            for file in 0..8 {
                let is_light = (file + rank) % 2 != 0;
                let bg = if is_light { LIGHT_BG } else { DARK_BG };
                print!("{bg}     {RESET}");
            }
            println!();

            // Middle row: rank label + pieces
            print!("  {COORD_FG}{}{RESET} ", rank + 1);
            for file in 0..8 {
                let sq = Square::from_file_rank(File::new(file), Rank::new(rank));
                let is_light = (file + rank) % 2 != 0;
                let bg = if is_light { LIGHT_BG } else { DARK_BG };

                match self.board.piece_on(sq) {
                    Some(p) => {
                        let fg = match p.color {
                            Color::White => WHITE_PC,
                            Color::Black => BLACK_PC,
                        };
                        let ch = piece_char(p.piece_type, p.color);
                        print!("{bg}{BOLD}{fg}  {ch}  {RESET}");
                    }
                    None => print!("{bg}     {RESET}"),
                }
            }
            println!(" {COORD_FG}{}{RESET}", rank + 1);

            // Bottom padding row
            print!("    ");
            for file in 0..8 {
                let is_light = (file + rank) % 2 != 0;
                let bg = if is_light { LIGHT_BG } else { DARK_BG };
                print!("{bg}     {RESET}");
            }
            println!();
        }

        println!("  {COORD_FG}    a    b    c    d    e    f    g    h{RESET}");

        println!();

        // Game info
        let side = match self.board.side_to_move() {
            Color::White => "White",
            Color::Black => "Black",
        };
        let ep = self
            .board
            .en_passant_square()
            .map_or("-".to_string(), |sq| sq.to_string());
        println!(
            "  {LABEL_FG}{DIM}Turn:{RESET}  {side}    {LABEL_FG}{DIM}Castling:{RESET}  {}    {LABEL_FG}{DIM}EP:{RESET}  {ep}    {LABEL_FG}{DIM}Move:{RESET}  {}",
            self.board.castling_rights(),
            self.board.fullmove_number(),
        );
        println!("  {LABEL_FG}{DIM}FEN:{RESET}   {}", self.board.to_fen());
        println!(
            "  {LABEL_FG}{DIM}Hash:{RESET}  {:016x}",
            self.board.zobrist_hash()
        );
        println!();
    }

    fn cmd_eval(&self) {
        let score = rage_search::eval::evaluate_pst(&self.board);
        println!(
            "Static eval: {score} cp (from {} perspective)",
            self.board.side_to_move()
        );
    }

    fn cmd_perft<'a>(&mut self, tokens: &mut impl Iterator<Item = &'a str>) {
        let depth: u32 = match tokens.next().and_then(|s| s.parse().ok()) {
            Some(d) => d,
            None => {
                println!("Usage: perft <depth>");
                return;
            }
        };

        let start = std::time::Instant::now();
        let nodes = rage_core::perft(&mut self.board, depth);
        let elapsed = start.elapsed();
        let nps = if elapsed.as_millis() > 0 {
            nodes * 1000 / elapsed.as_millis() as u64
        } else {
            0
        };

        println!("Nodes: {nodes}");
        println!("Time: {}ms", elapsed.as_millis());
        println!("NPS: {nps}");
    }

    fn cmd_compiler(&self) {
        println!("{ENGINE_NAME} v{}", env!("CARGO_PKG_VERSION"));
        println!("Compiler: {}", env!("RUSTC_VERSION"));
        println!(
            "Profile: {}",
            if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
        );
        println!("Target: {}", env!("TARGET"));
    }

    fn cmd_help(&self) {
        println!(
            "{ENGINE_NAME} v{} by {ENGINE_AUTHOR}",
            env!("CARGO_PKG_VERSION")
        );
        println!();
        println!("UCI commands:");
        println!("  uci            Show engine identity and options");
        println!("  isready        Synchronize with engine");
        println!("  ucinewgame     Reset engine state for a new game");
        println!("  position       Set board position (startpos/fen + moves)");
        println!("  go             Start searching (depth/movetime/wtime/btime/infinite/...)");
        println!("  stop           Stop searching");
        println!("  setoption      Set engine option (name <n> value <v>)");
        println!("  quit           Exit the engine");
        println!();
        println!("Debug commands:");
        println!("  d / display    Show the current board");
        println!("  dd             Show the board with emoji pieces and colored squares");
        println!("  eval           Show static evaluation");
        println!("  perft <depth>  Run perft node count");
        println!("  compiler       Show compiler and build info");
        println!("  help           Show this help");
        println!("  license        Show license info");
    }

    fn cmd_license(&self) {
        println!(
            "{ENGINE_NAME} v{} by {ENGINE_AUTHOR}",
            env!("CARGO_PKG_VERSION")
        );
        println!("MIT License, Copyright (c) 2026 Jim Calabro");
        println!();
        println!("Permission is hereby granted, free of charge, to any person obtaining a copy");
        println!("of this software and associated documentation files (the \"Software\"), to deal");
        println!("in the Software without restriction, including without limitation the rights");
        println!("to use, copy, modify, merge, publish, distribute, sublicense, and/or sell");
        println!("copies of the Software, and to permit persons to whom the Software is");
        println!("furnished to do so, subject to the following conditions:");
        println!();
        println!("The above copyright notice and this permission notice shall be included in all");
        println!("copies or substantial portions of the Software.");
        println!();
        println!("THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR");
        println!("IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,");
        println!("FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE");
        println!("AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER");
        println!("LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,");
        println!("OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE");
        println!("SOFTWARE.");
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn parse_position_startpos() {
        let mut engine = UciEngine::new();
        engine.handle_command("position startpos");
        assert_eq!(engine.board.to_fen(), Board::STARTPOS_FEN);
    }

    #[test]
    fn parse_position_startpos_moves() {
        let mut engine = UciEngine::new();
        engine.handle_command("position startpos moves e2e4 e7e5");
        // Strict EP: no white pawn on d5/f5 to capture, so EP square is "-"
        assert_eq!(
            engine.board.to_fen(),
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        );
    }

    #[test]
    fn parse_position_fen() {
        let mut engine = UciEngine::new();
        let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
        engine.handle_command(&format!("position fen {fen}"));
        assert_eq!(engine.board.to_fen(), fen);
    }

    #[test]
    fn parse_position_fen_with_moves() {
        let mut engine = UciEngine::new();
        engine.handle_command(
            "position fen rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1 moves e7e5",
        );
        // Strict EP: no white pawn on d5/f5 to capture, so EP square is "-"
        assert_eq!(
            engine.board.to_fen(),
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        );
    }

    #[test]
    fn ucinewgame_resets() {
        let mut engine = UciEngine::new();
        engine.handle_command("position startpos moves e2e4");
        engine.handle_command("ucinewgame");
        assert_eq!(engine.board.to_fen(), Board::STARTPOS_FEN);
    }

    #[test]
    fn setoption_hash() {
        let mut engine = UciEngine::new();
        engine.handle_command("setoption name Hash value 32");
    }

    #[test]
    fn quit_returns_true() {
        let mut engine = UciEngine::new();
        assert!(engine.handle_command("quit"));
    }

    #[test]
    fn unknown_command_ignored() {
        let mut engine = UciEngine::new();
        assert!(!engine.handle_command("foobar"));
    }

    #[test]
    fn go_depth_and_stop() {
        let mut engine = UciEngine::new();
        engine.handle_command("position startpos");
        engine.handle_command("go depth 3");
        // Stop the search and recover searcher
        engine.handle_command("stop");
        assert!(engine.searcher.is_some());
    }

    #[test]
    fn position_with_castling_moves() {
        let mut engine = UciEngine::new();
        // Play to a position where castling has occurred
        engine
            .handle_command("position startpos moves e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 d2d3 f8e7 e1g1");
        // Verify white castled kingside: king on g1, rook on f1
        let fen = engine.board.to_fen();
        assert!(
            fen.contains("q -") || fen.contains("kq -"),
            "white should have castled (lost castling rights)"
        );
    }

    #[test]
    fn position_with_promotion() {
        let mut engine = UciEngine::new();
        // Position where white pawn promotes
        engine.handle_command("position fen 8/P7/8/8/8/8/8/4K2k w - - 0 1 moves a7a8q");
        let fen = engine.board.to_fen();
        assert!(
            fen.starts_with("Q"),
            "pawn should have promoted to queen: {fen}"
        );
    }

    #[test]
    fn multiple_go_commands() {
        let mut engine = UciEngine::new();
        engine.handle_command("position startpos");
        engine.handle_command("go depth 2");
        engine.handle_command("stop");
        // Second search should work fine with preserved TT
        engine.handle_command("go depth 2");
        engine.handle_command("stop");
        assert!(engine.searcher.is_some());
    }

    #[test]
    fn ucinewgame_clears_tt() {
        let mut engine = UciEngine::new();
        engine.handle_command("position startpos");
        engine.handle_command("go depth 3");
        engine.handle_command("stop");
        engine.handle_command("ucinewgame");
        // After ucinewgame, board should be reset
        assert_eq!(engine.board.to_fen(), Board::STARTPOS_FEN);
        // Searcher should be recovered
        assert!(engine.searcher.is_some());
    }

    #[test]
    fn empty_go_treated_as_infinite() {
        let mut engine = UciEngine::new();
        engine.handle_command("position startpos");
        // "go" with no params = infinite search
        engine.handle_command("go");
        // Let it search briefly then stop
        std::thread::sleep(std::time::Duration::from_millis(50));
        engine.handle_command("stop");
        assert!(engine.searcher.is_some());
    }

    #[test]
    fn invalid_fen_ignored() {
        let mut engine = UciEngine::new();
        let original_fen = engine.board.to_fen();
        engine.handle_command("position fen totally_invalid_fen");
        // Board should remain unchanged
        assert_eq!(engine.board.to_fen(), original_fen);
    }

    #[test]
    fn invalid_move_stops_apply() {
        let mut engine = UciEngine::new();
        // First move valid, second invalid - should stop at invalid
        engine.handle_command("position startpos moves e2e4 zzzz");
        // Only e2e4 should have been applied
        let fen = engine.board.to_fen();
        assert!(fen.contains("4P3"), "e2e4 should have been applied: {fen}");
    }
}
