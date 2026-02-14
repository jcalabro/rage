use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;

use rage_core::Board;
use rage_search::search::{InfoCallback, Searcher};
use rage_search::time::SearchLimits;

const ENGINE_NAME: &str = "Rage";
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
            "eval" => self.cmd_eval(),
            "perft" => self.cmd_perft(&mut tokens),
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
                    "info depth {} score {} nodes {} nps {} time {} hashfull {} pv {}",
                    info.depth,
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
        println!("{}", self.board);
        println!("FEN: {}", self.board.to_fen());
        println!("Hash: {:016x}", self.board.zobrist_hash());
    }

    fn cmd_eval(&self) {
        let score = rage_search::eval::evaluate(&self.board);
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
}
