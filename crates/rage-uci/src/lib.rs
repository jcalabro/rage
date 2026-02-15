use std::io::{self, BufRead, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Instant;

use rage_core::Board;
use rage_nnue::{NnueNetwork, NnueWeights};
use rage_search::search::{InfoCallback, SearchInfo};
use rage_search::threads::ThreadPool;
use rage_search::time::SearchLimits;
use rage_syzygy::Tablebase;

const ENGINE_NAME: &str = "rage";
const ENGINE_AUTHOR: &str = "jcalabro";

/// Default transposition table size in MB.
const DEFAULT_HASH_MB: usize = 16;

/// Default number of search threads.
const DEFAULT_THREADS: usize = 1;

/// Run the UCI protocol loop. Blocks until `quit` is received.
///
/// If `network` is provided, it is set on all search threads at startup.
/// If `syzygy_path` is provided, Syzygy tablebases are loaded from that path.
pub fn run(network: Option<Arc<NnueNetwork>>, syzygy_path: Option<String>) {
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

    let mut engine = UciEngine::new(network, syzygy_path);

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
    /// None while a search thread owns the pool.
    pool: Option<ThreadPool>,
    stop: Arc<AtomicBool>,
    search_handle: Option<thread::JoinHandle<ThreadPool>>,
    /// Shared flag: true while pondering, flipped to false on `ponderhit`.
    ponder_flag: Option<Arc<AtomicBool>>,
    /// Shared offset written on `ponderhit` so time management starts from that moment.
    ponder_offset: Option<Arc<AtomicU64>>,
    /// When pondering started, used to compute the offset on `ponderhit`.
    ponder_start: Option<Instant>,
    /// Current Syzygy path (for re-loading on setting changes).
    syzygy_path: Option<String>,
    /// Minimum search depth before probing tablebases.
    syzygy_probe_depth: u32,
    /// Maximum pieces for tablebase probing.
    syzygy_probe_limit: u32,
    /// Number of principal variations to report. Default 1.
    multi_pv: usize,
    /// Whether to show WDL (Win/Draw/Loss) probabilities in info strings.
    show_wdl: bool,
}

impl UciEngine {
    fn new(network: Option<Arc<NnueNetwork>>, syzygy_path: Option<String>) -> Self {
        let mut pool = ThreadPool::new(DEFAULT_THREADS, DEFAULT_HASH_MB);
        if let Some(net) = network {
            pool.set_network(Some(net));
        }

        // Use explicit path, or fall back to embedded 3+4 piece tables.
        let syzygy_path = syzygy_path.or_else(|| {
            rage_syzygy::extract_embedded_tables().map(|p| p.to_string_lossy().into_owned())
        });

        if let Some(ref path) = syzygy_path {
            let tb = Arc::new(Tablebase::new(path));
            if tb.is_available() {
                pool.set_syzygy(Some(tb));
            }
        }
        Self {
            board: Board::starting_position(),
            pool: Some(pool),
            stop: Arc::new(AtomicBool::new(false)),
            search_handle: None,
            ponder_flag: None,
            ponder_offset: None,
            ponder_start: None,
            syzygy_path,
            syzygy_probe_depth: 1,
            syzygy_probe_limit: 7,
            multi_pv: 1,
            show_wdl: false,
        }
    }

    /// Try to load an NNUE network from the given path and set it on the pool.
    /// Prints an error to stderr on failure (graceful fallback to PST eval).
    fn load_network(&mut self, path: &str) {
        if path.is_empty() || path == "<empty>" {
            // Clear the network, fall back to PST eval
            if let Some(ref mut pool) = self.pool {
                pool.set_network(None);
            }
            return;
        }

        match NnueWeights::from_file(Path::new(path)) {
            Ok(weights) => {
                let network = Arc::new(NnueNetwork::new(weights));
                if let Some(ref mut pool) = self.pool {
                    pool.set_network(Some(network));
                }
                eprintln!("info string loaded NNUE network from {path}");
            }
            Err(e) => {
                eprintln!("info string failed to load NNUE network: {e}");
            }
        }
    }

    /// Load Syzygy tablebases from the given path.
    /// The path can be colon or semicolon separated for multiple directories.
    fn load_syzygy(&mut self, path: &str) {
        if path.is_empty() || path == "<empty>" {
            if let Some(ref mut pool) = self.pool {
                pool.set_syzygy(None);
            }
            self.syzygy_path = None;
            return;
        }

        let mut tb = Tablebase::new(path);
        tb.set_probe_depth(self.syzygy_probe_depth);
        tb.set_probe_limit(self.syzygy_probe_limit);
        let tb = Arc::new(tb);

        if tb.is_available() {
            if let Some(ref mut pool) = self.pool {
                pool.set_syzygy(Some(Arc::clone(&tb)));
            }
            self.syzygy_path = Some(path.to_string());
        } else {
            eprintln!("info string no Syzygy tables found at {path}");
            self.syzygy_path = None;
        }
    }

    /// Stop any running search and recover the pool.
    fn wait_for_search(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.search_handle.take()
            && let Ok(pool) = handle.join()
        {
            self.pool = Some(pool);
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
            "ponderhit" => self.cmd_ponderhit(),
            "setoption" => self.cmd_setoption(&mut tokens),
            "register" => {
                println!("registration ok");
            }
            "debug" => { /* ignored — no debug output mode */ }
            "quit" => {
                self.cmd_stop();
                return true;
            }
            // Debug commands
            "bench" => self.cmd_bench(&mut tokens),
            "d" | "display" => self.cmd_display(),
            "dd" => self.cmd_display_pretty(),
            "eval" => self.cmd_eval(),
            "push" => self.cmd_push(&mut tokens),
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
        println!("option name Threads type spin default {DEFAULT_THREADS} min 1 max 256");
        println!("option name MultiPV type spin default 1 min 1 max 32");
        println!("option name Ponder type check default true");
        println!("option name EvalFile type string default <empty>");
        println!("option name SyzygyPath type string default <empty>");
        println!("option name SyzygyProbeDepth type spin default 1 min 1 max 100");
        println!("option name SyzygyProbeLimit type spin default 7 min 0 max 7");
        println!("option name UCI_ShowWDL type check default false");
        println!("uciok");
    }

    fn cmd_isready(&self) {
        println!("readyok");
    }

    fn cmd_ucinewgame(&mut self) {
        self.wait_for_search();
        if let Some(ref mut pool) = self.pool {
            pool.clear();
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
        // Recover the pool from any running search
        self.wait_for_search();

        let mut limits = SearchLimits::default();
        let mut is_ponder = false;

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
                "ponder" => is_ponder = true,
                _ => {}
            }
        }

        // Set up ponder atomics if pondering
        if is_ponder {
            let ponder_flag = Arc::new(AtomicBool::new(true));
            let ponder_offset = Arc::new(AtomicU64::new(0));
            limits.ponder = Some(Arc::clone(&ponder_flag));
            limits.ponder_offset = Some(Arc::clone(&ponder_offset));
            self.ponder_flag = Some(ponder_flag);
            self.ponder_offset = Some(ponder_offset);
            self.ponder_start = Some(Instant::now());
        } else {
            self.ponder_flag = None;
            self.ponder_offset = None;
            self.ponder_start = None;
        }

        let stop = Arc::new(AtomicBool::new(false));
        self.stop = Arc::clone(&stop);

        let mut board = self.board.clone();

        // Take the pool out - search thread owns it temporarily
        let mut pool = match self.pool.take() {
            Some(p) => p,
            None => ThreadPool::new(DEFAULT_THREADS, DEFAULT_HASH_MB),
        };

        let multi_pv = self.multi_pv;
        let show_wdl = self.show_wdl;
        let handle = thread::spawn(move || {
            let info_cb: InfoCallback = Box::new(move |info| {
                let pv_str: String = info
                    .pv
                    .iter()
                    .map(|m| m.to_string())
                    .collect::<Vec<_>>()
                    .join(" ");

                let tb_str = if info.tbhits > 0 {
                    format!(" tbhits {}", info.tbhits)
                } else {
                    String::new()
                };
                let mpv_str = if multi_pv > 1 {
                    format!(" multipv {}", info.multipv)
                } else {
                    String::new()
                };
                let wdl_str = if show_wdl {
                    let (w, d, l) = info
                        .wdl
                        .unwrap_or_else(|| SearchInfo::compute_wdl(info.score));
                    format!(" wdl {w} {d} {l}")
                } else {
                    String::new()
                };
                println!(
                    "info depth {} seldepth {} score {}{}{} nodes {} nps {} time {} hashfull {}{} pv {}",
                    info.depth,
                    info.seldepth,
                    info.score_string(),
                    wdl_str,
                    mpv_str,
                    info.nodes,
                    info.nps(),
                    info.time_ms,
                    info.hashfull,
                    tb_str,
                    pv_str,
                );
            });

            let result = pool.search(&mut board, &limits, stop, Some(info_cb));

            // Output bestmove with ponder move if PV has a second move
            if result.pv.len() >= 2 {
                println!("bestmove {} ponder {}", result.best_move, result.pv[1]);
            } else {
                println!("bestmove {}", result.best_move);
            }
            let _ = io::stdout().flush();

            // Return the pool so state is preserved
            pool
        });

        self.search_handle = Some(handle);
    }

    fn cmd_stop(&mut self) {
        self.ponder_flag = None;
        self.ponder_offset = None;
        self.ponder_start = None;
        self.wait_for_search();
    }

    fn cmd_ponderhit(&mut self) {
        // Compute how long we've been pondering and record it as the offset,
        // so time management starts from this moment.
        if let (Some(flag), Some(offset), Some(start)) =
            (&self.ponder_flag, &self.ponder_offset, self.ponder_start)
        {
            let elapsed_ms = start.elapsed().as_millis() as u64;
            offset.store(elapsed_ms, Ordering::Relaxed);
            flag.store(false, Ordering::Relaxed);
        }
        // Clear ponder state (atomics are already shared with the search)
        self.ponder_flag = None;
        self.ponder_offset = None;
        self.ponder_start = None;
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

        match name.to_lowercase().as_str() {
            "hash" => {
                if let Ok(mb) = value.parse::<usize>() {
                    let mb = mb.clamp(1, 1024);
                    if let Some(ref mut pool) = self.pool {
                        pool.resize_tt(mb);
                    }
                }
            }
            "threads" => {
                if let Ok(n) = value.parse::<usize>() {
                    let n = n.clamp(1, 256);
                    if let Some(ref mut pool) = self.pool {
                        pool.set_num_threads(n);
                    }
                }
            }
            "multipv" => {
                if let Ok(n) = value.parse::<usize>() {
                    self.multi_pv = n.clamp(1, 32);
                    if let Some(ref mut pool) = self.pool {
                        pool.set_multi_pv(self.multi_pv);
                    }
                }
            }
            "evalfile" => {
                self.load_network(&value);
            }
            "syzygypath" => {
                self.load_syzygy(&value);
            }
            "syzygyprobedepth" => {
                if let Ok(d) = value.parse::<u32>() {
                    self.syzygy_probe_depth = d.clamp(1, 100);
                    // Reload tablebase with new settings
                    if let Some(path) = self.syzygy_path.clone() {
                        self.load_syzygy(&path);
                    }
                }
            }
            "syzygyprobelimit" => {
                if let Ok(l) = value.parse::<u32>() {
                    self.syzygy_probe_limit = l.clamp(0, 7);
                    // Reload tablebase with new settings
                    if let Some(path) = self.syzygy_path.clone() {
                        self.load_syzygy(&path);
                    }
                }
            }
            "uci_showwdl" => {
                self.show_wdl = value == "true";
            }
            _ => {}
        }
    }

    // --- Debug commands ---

    fn cmd_push<'a>(&mut self, tokens: &mut impl Iterator<Item = &'a str>) {
        let move_str = match tokens.next() {
            Some(s) => s,
            None => {
                println!("Usage: push <move> (e.g. push e2e4)");
                return;
            }
        };
        match self.board.parse_uci_move(move_str) {
            Ok(m) => {
                self.board.make_move(m);
                self.cmd_display();
            }
            Err(e) => println!("Illegal move '{move_str}': {e}"),
        }
    }

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

    fn cmd_eval(&mut self) {
        let pst_score = rage_search::eval::evaluate_pst(&self.board);
        println!(
            "PST eval: {pst_score} cp (from {} perspective)",
            self.board.side_to_move()
        );

        // Show NNUE eval if a network is loaded
        if let Some(ref mut pool) = self.pool
            && let Some(nnue_score) = pool.evaluate_nnue(&self.board)
        {
            println!(
                "NNUE eval: {nnue_score} cp (from {} perspective)",
                self.board.side_to_move()
            );
        }
    }

    fn cmd_bench<'a>(&mut self, tokens: &mut impl Iterator<Item = &'a str>) {
        self.wait_for_search();
        let depth = tokens
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or(rage_search::bench::DEFAULT_BENCH_DEPTH);
        let network = self.pool.as_ref().and_then(|p| p.network()).map(Arc::clone);
        rage_search::bench::run_bench(depth, network);
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
        println!(
            "  go             Start searching (depth/movetime/wtime/btime/infinite/ponder/...)"
        );
        println!("  stop           Stop searching");
        println!(
            "  ponderhit      Opponent played the expected move (switch from ponder to normal)"
        );
        println!("  setoption      Set engine option (name <n> value <v>)");
        println!("  quit           Exit the engine");
        println!();
        println!("Options:");
        println!("  Hash <MB>      Transposition table size (1-1024, default {DEFAULT_HASH_MB})");
        println!("  Threads <N>    Number of search threads (1-256, default {DEFAULT_THREADS})");
        println!("  MultiPV <N>    Number of principal variations (1-32, default 1)");
        println!("  EvalFile <path> Load NNUE network from file (empty to clear)");
        println!("  SyzygyPath <path> Path to Syzygy tablebases (colon-separated)");
        println!("  SyzygyProbeDepth <N> Min depth for TB probing (1-100, default 1)");
        println!("  SyzygyProbeLimit <N> Max pieces for TB probing (0-7, default 7)");
        println!();
        println!("Debug commands:");
        println!("  push <move>    Play a move on the board (e.g. push e2e4)");
        println!("  bench [depth]  Run search benchmark (default depth 13)");
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
        let mut engine = UciEngine::new(None, None);
        engine.handle_command("position startpos");
        assert_eq!(engine.board.to_fen(), Board::STARTPOS_FEN);
    }

    #[test]
    fn parse_position_startpos_moves() {
        let mut engine = UciEngine::new(None, None);
        engine.handle_command("position startpos moves e2e4 e7e5");
        // Strict EP: no white pawn on d5/f5 to capture, so EP square is "-"
        assert_eq!(
            engine.board.to_fen(),
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        );
    }

    #[test]
    fn parse_position_fen() {
        let mut engine = UciEngine::new(None, None);
        let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
        engine.handle_command(&format!("position fen {fen}"));
        assert_eq!(engine.board.to_fen(), fen);
    }

    #[test]
    fn parse_position_fen_with_moves() {
        let mut engine = UciEngine::new(None, None);
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
        let mut engine = UciEngine::new(None, None);
        engine.handle_command("position startpos moves e2e4");
        engine.handle_command("ucinewgame");
        assert_eq!(engine.board.to_fen(), Board::STARTPOS_FEN);
    }

    #[test]
    fn setoption_hash() {
        let mut engine = UciEngine::new(None, None);
        engine.handle_command("setoption name Hash value 32");
    }

    #[test]
    fn setoption_multipv() {
        let mut engine = UciEngine::new(None, None);
        assert_eq!(engine.multi_pv, 1);
        engine.handle_command("setoption name MultiPV value 3");
        assert_eq!(engine.multi_pv, 3);
        // Clamped to valid range
        engine.handle_command("setoption name MultiPV value 0");
        assert_eq!(engine.multi_pv, 1);
        engine.handle_command("setoption name MultiPV value 999");
        assert_eq!(engine.multi_pv, 32);
    }

    #[test]
    fn setoption_threads() {
        let mut engine = UciEngine::new(None, None);
        engine.handle_command("setoption name Threads value 4");
    }

    #[test]
    fn quit_returns_true() {
        let mut engine = UciEngine::new(None, None);
        assert!(engine.handle_command("quit"));
    }

    #[test]
    fn unknown_command_ignored() {
        let mut engine = UciEngine::new(None, None);
        assert!(!engine.handle_command("foobar"));
    }

    #[test]
    fn go_depth_and_stop() {
        let mut engine = UciEngine::new(None, None);
        engine.handle_command("position startpos");
        engine.handle_command("go depth 3");
        // Stop the search and recover pool
        engine.handle_command("stop");
        assert!(engine.pool.is_some());
    }

    #[test]
    fn position_with_castling_moves() {
        let mut engine = UciEngine::new(None, None);
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
        let mut engine = UciEngine::new(None, None);
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
        let mut engine = UciEngine::new(None, None);
        engine.handle_command("position startpos");
        engine.handle_command("go depth 2");
        engine.handle_command("stop");
        // Second search should work fine with preserved TT
        engine.handle_command("go depth 2");
        engine.handle_command("stop");
        assert!(engine.pool.is_some());
    }

    #[test]
    fn ucinewgame_clears_tt() {
        let mut engine = UciEngine::new(None, None);
        engine.handle_command("position startpos");
        engine.handle_command("go depth 3");
        engine.handle_command("stop");
        engine.handle_command("ucinewgame");
        // After ucinewgame, board should be reset
        assert_eq!(engine.board.to_fen(), Board::STARTPOS_FEN);
        // Pool should be recovered
        assert!(engine.pool.is_some());
    }

    #[test]
    fn empty_go_treated_as_infinite() {
        let mut engine = UciEngine::new(None, None);
        engine.handle_command("position startpos");
        // "go" with no params = infinite search
        engine.handle_command("go");
        // Let it search briefly then stop
        std::thread::sleep(std::time::Duration::from_millis(50));
        engine.handle_command("stop");
        assert!(engine.pool.is_some());
    }

    #[test]
    fn invalid_fen_ignored() {
        let mut engine = UciEngine::new(None, None);
        let original_fen = engine.board.to_fen();
        engine.handle_command("position fen totally_invalid_fen");
        // Board should remain unchanged
        assert_eq!(engine.board.to_fen(), original_fen);
    }

    #[test]
    fn invalid_move_stops_apply() {
        let mut engine = UciEngine::new(None, None);
        // First move valid, second invalid - should stop at invalid
        engine.handle_command("position startpos moves e2e4 zzzz");
        // Only e2e4 should have been applied
        let fen = engine.board.to_fen();
        assert!(fen.contains("4P3"), "e2e4 should have been applied: {fen}");
    }

    #[test]
    fn go_ponder_sets_ponder_flag() {
        let mut engine = UciEngine::new(None, None);
        engine.handle_command("position startpos");
        engine.handle_command("go ponder wtime 60000 btime 60000");
        // Ponder flag should be set
        assert!(engine.ponder_flag.is_some());
        assert!(engine.ponder_offset.is_some());
        assert!(engine.ponder_start.is_some());
        // Let it search briefly then stop
        std::thread::sleep(std::time::Duration::from_millis(50));
        engine.handle_command("stop");
        assert!(engine.pool.is_some());
    }

    #[test]
    fn ponderhit_transitions_to_normal_search() {
        let mut engine = UciEngine::new(None, None);
        engine.handle_command("position startpos");
        engine.handle_command("go ponder wtime 60000 btime 60000");
        // Let it ponder briefly
        std::thread::sleep(std::time::Duration::from_millis(50));
        // Ponderhit: opponent played the expected move
        engine.handle_command("ponderhit");
        // Ponder state should be cleared
        assert!(engine.ponder_flag.is_none());
        assert!(engine.ponder_offset.is_none());
        assert!(engine.ponder_start.is_none());
        // Stop and verify pool recovered
        engine.handle_command("stop");
        assert!(engine.pool.is_some());
    }

    #[test]
    fn ponderhit_without_ponder_is_noop() {
        let mut engine = UciEngine::new(None, None);
        // ponderhit when not pondering should do nothing
        engine.handle_command("ponderhit");
        assert!(engine.ponder_flag.is_none());
    }

    #[test]
    fn go_without_ponder_clears_ponder_state() {
        let mut engine = UciEngine::new(None, None);
        engine.handle_command("position startpos");
        // First do a ponder search
        engine.handle_command("go ponder depth 3");
        std::thread::sleep(std::time::Duration::from_millis(50));
        engine.handle_command("stop");
        // Now do a normal search — ponder state should be cleared
        engine.handle_command("go depth 3");
        assert!(engine.ponder_flag.is_none());
        engine.handle_command("stop");
    }

    // --- NNUE network loading tests ---

    fn write_test_network(name: &str) -> std::path::PathBuf {
        let weights = rage_nnue::NnueWeights::random(42);
        let bytes = weights.to_bytes();
        let dir = std::env::temp_dir().join("rage_uci_tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}.nnue"));
        std::fs::write(&path, &bytes).unwrap();
        path
    }

    #[test]
    fn engine_constructed_with_network() {
        let weights = rage_nnue::NnueWeights::random(42);
        let network = Arc::new(NnueNetwork::new(weights));
        let engine = UciEngine::new(Some(network), None);
        let pool = engine.pool.as_ref().unwrap();
        assert!(
            pool.network().is_some(),
            "pool should have network when constructed with one"
        );
    }

    #[test]
    fn engine_constructed_without_network() {
        let engine = UciEngine::new(None, None);
        let pool = engine.pool.as_ref().unwrap();
        assert!(
            pool.network().is_none(),
            "pool should have no network by default"
        );
    }

    #[test]
    fn load_network_from_valid_file() {
        let path = write_test_network("load_valid");
        let mut engine = UciEngine::new(None, None);
        engine.load_network(path.to_str().unwrap());
        let pool = engine.pool.as_ref().unwrap();
        assert!(
            pool.network().is_some(),
            "network should be loaded from valid file"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_network_invalid_path_graceful() {
        let mut engine = UciEngine::new(None, None);
        engine.load_network("/tmp/does_not_exist_rage_test.nnue");
        // Should not crash, should fall back to no network
        let pool = engine.pool.as_ref().unwrap();
        assert!(
            pool.network().is_none(),
            "invalid path should not load a network"
        );
    }

    #[test]
    fn load_network_empty_clears_network() {
        let path = write_test_network("load_empty_clear");
        let mut engine = UciEngine::new(None, None);
        engine.load_network(path.to_str().unwrap());
        assert!(engine.pool.as_ref().unwrap().network().is_some());

        // Clear with empty string
        engine.load_network("");
        assert!(
            engine.pool.as_ref().unwrap().network().is_none(),
            "empty string should clear the network"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_network_empty_tag_clears_network() {
        let path = write_test_network("load_tag_clear");
        let mut engine = UciEngine::new(None, None);
        engine.load_network(path.to_str().unwrap());
        assert!(engine.pool.as_ref().unwrap().network().is_some());

        // Clear with <empty> tag (matches UCI advertised default)
        engine.load_network("<empty>");
        assert!(
            engine.pool.as_ref().unwrap().network().is_none(),
            "<empty> should clear the network"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn setoption_evalfile_loads_network() {
        let path = write_test_network("setoption_eval");
        let mut engine = UciEngine::new(None, None);
        engine.handle_command(&format!(
            "setoption name EvalFile value {}",
            path.to_str().unwrap()
        ));
        assert!(
            engine.pool.as_ref().unwrap().network().is_some(),
            "setoption EvalFile should load the network"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn setoption_evalfile_invalid_path_no_crash() {
        let mut engine = UciEngine::new(None, None);
        engine.handle_command("setoption name EvalFile value /tmp/nonexistent_rage.nnue");
        assert!(
            engine.pool.as_ref().unwrap().network().is_none(),
            "invalid EvalFile should not set a network"
        );
    }

    #[test]
    fn network_survives_ucinewgame() {
        let path = write_test_network("survives_newgame");
        let mut engine = UciEngine::new(None, None);
        engine.load_network(path.to_str().unwrap());
        assert!(engine.pool.as_ref().unwrap().network().is_some());

        engine.handle_command("ucinewgame");
        assert!(
            engine.pool.as_ref().unwrap().network().is_some(),
            "ucinewgame should not drop the network"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn network_survives_thread_change() {
        let path = write_test_network("survives_threads");
        let mut engine = UciEngine::new(None, None);
        engine.load_network(path.to_str().unwrap());

        engine.handle_command("setoption name Threads value 4");
        assert!(
            engine.pool.as_ref().unwrap().network().is_some(),
            "changing threads should preserve the network"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn search_with_loaded_network() {
        let path = write_test_network("search_loaded");
        let mut engine = UciEngine::new(None, None);
        engine.load_network(path.to_str().unwrap());
        engine.handle_command("position startpos");
        engine.handle_command("go depth 3");
        engine.handle_command("stop");
        assert!(
            engine.pool.is_some(),
            "pool should be recovered after search with NNUE"
        );
        let _ = std::fs::remove_file(&path);
    }
}
