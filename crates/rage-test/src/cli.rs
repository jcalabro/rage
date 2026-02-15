//! CLI argument parsing and subcommand dispatch.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use clap::{Parser, Subcommand};

use crate::bench_diff;
use crate::db::{self, ResultsDb};
use crate::elo_cmd;
use crate::opening::OpeningBook;
use crate::sprt;
use crate::tournament;
use crate::tui;
use crate::types::*;

#[derive(Parser)]
#[command(
    name = "rage-test",
    version,
    about = "Strength testing framework for UCI chess engines"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run SPRT test between two engines
    Sprt {
        /// Path to engine A (the one being tested)
        #[arg(long)]
        engine_a: PathBuf,
        /// Path to engine B (the baseline)
        #[arg(long)]
        engine_b: PathBuf,
        /// Time control (e.g., "8+0.08", "60+0.6", "40/60+0")
        #[arg(long, default_value = "8+0.08")]
        tc: String,
        /// Fixed search depth (overrides time control)
        #[arg(long)]
        depth: Option<u32>,
        /// Fixed node count (overrides time control)
        #[arg(long)]
        nodes: Option<u64>,
        /// Null hypothesis Elo bound (H0: Elo <= elo0)
        #[arg(long, default_value = "0")]
        elo0: f64,
        /// Alternative hypothesis Elo bound (H1: Elo >= elo1)
        #[arg(long, default_value = "5")]
        elo1: f64,
        /// Type I error rate
        #[arg(long, default_value = "0.05")]
        alpha: f64,
        /// Type II error rate
        #[arg(long, default_value = "0.05")]
        beta: f64,
        /// Number of concurrent game pairs
        #[arg(long, default_value = "1")]
        concurrency: u32,
        /// Path to opening book file (one FEN per line)
        #[arg(long)]
        book: Option<PathBuf>,
        /// UCI options for engine A (format: "Name=Value", can be repeated)
        #[arg(long = "option-a", value_name = "NAME=VALUE")]
        options_a: Vec<String>,
        /// UCI options for engine B (format: "Name=Value", can be repeated)
        #[arg(long = "option-b", value_name = "NAME=VALUE")]
        options_b: Vec<String>,
    },

    /// Run fixed-game Elo test
    Elo {
        #[arg(long)]
        engine_a: PathBuf,
        #[arg(long)]
        engine_b: PathBuf,
        #[arg(long, default_value = "8+0.08")]
        tc: String,
        #[arg(long)]
        depth: Option<u32>,
        #[arg(long)]
        nodes: Option<u64>,
        /// Number of game pairs to play
        #[arg(long, default_value = "100")]
        pairs: u32,
        #[arg(long, default_value = "1")]
        concurrency: u32,
        #[arg(long)]
        book: Option<PathBuf>,
        #[arg(long = "option-a", value_name = "NAME=VALUE")]
        options_a: Vec<String>,
        #[arg(long = "option-b", value_name = "NAME=VALUE")]
        options_b: Vec<String>,
    },

    /// Run round-robin tournament between multiple engines
    Tournament {
        /// Engine paths (at least 2)
        #[arg(long = "engine", required = true, num_args = 2..)]
        engines: Vec<PathBuf>,
        #[arg(long, default_value = "8+0.08")]
        tc: String,
        #[arg(long)]
        depth: Option<u32>,
        #[arg(long)]
        nodes: Option<u64>,
        /// Game pairs per matchup
        #[arg(long, default_value = "50")]
        rounds: u32,
        #[arg(long, default_value = "1")]
        concurrency: u32,
        #[arg(long)]
        book: Option<PathBuf>,
    },

    /// Show test result history
    History {
        /// Number of recent results to show
        #[arg(long, default_value = "20")]
        limit: u32,
    },

    /// Compare bench output between two engine binaries
    BenchDiff {
        #[arg(long)]
        engine_a: PathBuf,
        #[arg(long)]
        engine_b: PathBuf,
        /// Bench depth (default: engine's built-in default)
        #[arg(long)]
        depth: Option<u32>,
    },
}

pub fn run() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Sprt {
            engine_a,
            engine_b,
            tc,
            depth,
            nodes,
            elo0,
            elo1,
            alpha,
            beta,
            concurrency,
            book,
            options_a,
            options_b,
        } => {
            let tc_parsed = match TimeControl::parse(&tc) {
                Some(tc) => tc,
                None => {
                    eprintln!("Invalid time control: {tc}");
                    std::process::exit(1);
                }
            };

            let config = SprtConfig {
                match_config: MatchConfig {
                    engine_a: EngineConfig {
                        path: engine_a,
                        options: parse_options(&options_a),
                        name: None,
                    },
                    engine_b: EngineConfig {
                        path: engine_b,
                        options: parse_options(&options_b),
                        name: None,
                    },
                    tc: tc_parsed,
                    search_overrides: SearchOverrides { depth, nodes },
                    concurrency,
                    opening_file: book.clone(),
                    ..Default::default()
                },
                elo0,
                elo1,
                alpha,
                beta,
            };

            let mut opening_book = load_book(book.as_deref());
            let stop = Arc::new(AtomicBool::new(false));

            // Set up Ctrl+C handler
            let stop_clone = Arc::clone(&stop);
            ctrlc_handler(stop_clone);

            let config_lines = vec![
                format!(
                    "Engine A: {}  ({})",
                    config.match_config.engine_a.display_name(),
                    format_options(&config.match_config.engine_a.options),
                ),
                format!(
                    "Engine B: {}  ({})",
                    config.match_config.engine_b.display_name(),
                    format_options(&config.match_config.engine_b.options),
                ),
                format!(
                    "TC: {}  |  SPRT: H0 <= {elo0}, H1 >= {elo1} (a={alpha}, b={beta})",
                    tc,
                ),
            ];

            let is_tty = atty_stdout();

            if is_tty {
                run_sprt_tui(&config, &mut opening_book, stop, config_lines);
            } else {
                run_sprt_plain(&config, &mut opening_book, stop);
            }
        }

        Commands::Elo {
            engine_a,
            engine_b,
            tc,
            depth,
            nodes,
            pairs,
            concurrency,
            book,
            options_a,
            options_b,
        } => {
            let tc_parsed = match TimeControl::parse(&tc) {
                Some(tc) => tc,
                None => {
                    eprintln!("Invalid time control: {tc}");
                    std::process::exit(1);
                }
            };

            let elo_config = EloConfig {
                match_config: MatchConfig {
                    engine_a: EngineConfig {
                        path: engine_a,
                        options: parse_options(&options_a),
                        name: None,
                    },
                    engine_b: EngineConfig {
                        path: engine_b,
                        options: parse_options(&options_b),
                        name: None,
                    },
                    tc: tc_parsed,
                    search_overrides: SearchOverrides { depth, nodes },
                    concurrency,
                    opening_file: book.clone(),
                    ..Default::default()
                },
                pairs,
            };

            let mut opening_book = load_book(book.as_deref());
            let stop = Arc::new(AtomicBool::new(false));
            let stop_clone = Arc::clone(&stop);
            ctrlc_handler(stop_clone);

            let config_lines = vec![
                format!(
                    "Engine A: {}",
                    elo_config.match_config.engine_a.display_name(),
                ),
                format!(
                    "Engine B: {}",
                    elo_config.match_config.engine_b.display_name(),
                ),
                format!("TC: {}  |  Pairs: {pairs}", tc),
            ];

            let is_tty = atty_stdout();

            if is_tty {
                run_elo_tui(&elo_config, &mut opening_book, stop, config_lines);
            } else {
                run_elo_plain(&elo_config, &mut opening_book, stop);
            }
        }

        Commands::Tournament {
            engines,
            tc,
            depth,
            nodes,
            rounds,
            concurrency,
            book,
        } => {
            let tc_parsed = match TimeControl::parse(&tc) {
                Some(tc) => tc,
                None => {
                    eprintln!("Invalid time control: {tc}");
                    std::process::exit(1);
                }
            };

            let engine_configs: Vec<EngineConfig> = engines
                .into_iter()
                .map(|p| EngineConfig {
                    path: p,
                    options: Vec::new(),
                    name: None,
                })
                .collect();

            let tourn_config = TournamentConfig {
                engines: engine_configs,
                tc: tc_parsed,
                search_overrides: SearchOverrides { depth, nodes },
                concurrency,
                opening_file: book.clone(),
                move_overhead_ms: 20,
                max_moves: 300,
                win_adj_score: 10000,
                win_adj_count: 5,
                draw_adj_score: 5,
                draw_adj_count: 10,
                rounds,
            };

            let mut opening_book = load_book(book.as_deref());
            let stop = Arc::new(AtomicBool::new(false));
            let stop_clone = Arc::clone(&stop);
            ctrlc_handler(stop_clone);

            println!(
                "Starting tournament with {} engines...",
                tourn_config.engines.len()
            );

            let report =
                tournament::run_tournament(&tourn_config, &mut opening_book, stop, |_progress| {});

            tui::print_tournament_summary(&report);

            // Save to DB
            if let Ok(db_result) = ResultsDb::open_default() {
                for m in &report.matchups {
                    let _ = db_result.create_test(
                        "tournament",
                        &report.engines[m.engine_a_idx],
                        &report.engines[m.engine_b_idx],
                        &tc,
                        None,
                        None,
                    );
                }
            }
        }

        Commands::History { limit } => {
            let db = match ResultsDb::open_default() {
                Ok(db) => db,
                Err(e) => {
                    eprintln!("Failed to open database: {e}");
                    std::process::exit(1);
                }
            };

            let records = match db.list_tests(limit) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Failed to query database: {e}");
                    std::process::exit(1);
                }
            };

            if records.is_empty() {
                println!("No test results found.");
                return;
            }

            println!(
                "  ID      Type            Engine A            Engine B          TC     Pairs           Elo    Result  Date"
            );
            println!("{}", "-".repeat(110));

            for r in records {
                let elo_str = r
                    .elo
                    .map(|e| {
                        let err = r.elo_error.unwrap_or(0.0);
                        format!("{e:+.1}+/-{err:.1}")
                    })
                    .unwrap_or_else(|| "N/A".to_string());
                let decision = r.decision.as_deref().unwrap_or("-");
                println!(
                    "{:>4}  {:>8}  {:>20}  {:>20}  {:>10}  {:>8}  {:>12}  {:>8}  {}",
                    r.id,
                    r.test_type,
                    truncate(&r.engine_a, 20),
                    truncate(&r.engine_b, 20),
                    r.tc,
                    r.pairs,
                    elo_str,
                    decision,
                    &r.created_at[..r.created_at.len().min(16)],
                );
            }
        }

        Commands::BenchDiff {
            engine_a,
            engine_b,
            depth,
        } => {
            println!(
                "Running bench on {} and {}...",
                engine_a.display(),
                engine_b.display()
            );
            match bench_diff::run_bench_diff(&engine_a, &engine_b, depth) {
                Ok(diff) => {
                    let name_a = engine_a
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "engine-a".to_string());
                    let name_b = engine_b
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "engine-b".to_string());
                    println!();
                    print!("{}", bench_diff::format_bench_diff(&diff, &name_a, &name_b));
                }
                Err(e) => {
                    eprintln!("Bench diff failed: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
}

fn run_sprt_tui(
    config: &SprtConfig,
    book: &mut OpeningBook,
    stop: Arc<AtomicBool>,
    config_lines: Vec<String>,
) {
    let tui_result = tui::SprtTui::new(Arc::clone(&stop), config_lines);
    let mut sprt_tui = match tui_result {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to initialize TUI: {e}");
            // Fallback to plain mode
            run_sprt_plain(config, book, stop);
            return;
        }
    };

    let progress_handle = sprt_tui.progress_handle();

    let config_clone = config.clone();
    let book_clone_fens: Vec<String> = (0..book.len()).map(|i| book.get(i).to_string()).collect();
    let stop_clone = Arc::clone(&stop);

    // Run SPRT on a separate thread
    let search_handle = std::thread::spawn(move || {
        let mut inner_book = OpeningBook::from_fens(book_clone_fens);
        sprt::run_sprt(
            &config_clone,
            &mut inner_book,
            stop_clone,
            move |progress| {
                if let Ok(mut guard) = progress_handle.lock() {
                    *guard = Some(progress.clone());
                }
            },
        )
    });

    let _ = sprt_tui.run();
    let _final_progress = sprt_tui.cleanup();

    // Wait for search thread
    stop.store(true, Ordering::Relaxed);
    if let Ok(report) = search_handle.join() {
        tui::print_sprt_summary(&report);

        // Save to DB
        if let Ok(db_result) = ResultsDb::open_default() {
            let _ = db::save_sprt_result(
                &db_result,
                &config.match_config.engine_a.display_name(),
                &config.match_config.engine_b.display_name(),
                &config.match_config.tc.to_string(),
                &report,
                config.elo0,
                config.elo1,
            );
        }
    }
}

fn run_sprt_plain(config: &SprtConfig, book: &mut OpeningBook, stop: Arc<AtomicBool>) {
    let plain = std::sync::Mutex::new(tui::PlainProgress::new());
    let report = sprt::run_sprt(config, book, stop, move |progress| {
        if let Ok(mut p) = plain.lock() {
            p.print_sprt_progress(progress);
        }
    });
    tui::print_sprt_summary(&report);

    if let Ok(db_result) = ResultsDb::open_default() {
        let _ = db::save_sprt_result(
            &db_result,
            &config.match_config.engine_a.display_name(),
            &config.match_config.engine_b.display_name(),
            &config.match_config.tc.to_string(),
            &report,
            config.elo0,
            config.elo1,
        );
    }
}

fn run_elo_tui(
    config: &EloConfig,
    book: &mut OpeningBook,
    stop: Arc<AtomicBool>,
    config_lines: Vec<String>,
) {
    let tui_result = tui::EloTui::new(Arc::clone(&stop), config_lines);
    let mut elo_tui = match tui_result {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to initialize TUI: {e}");
            run_elo_plain(config, book, stop);
            return;
        }
    };

    let progress_handle = elo_tui.progress_handle();

    let config_clone = config.clone();
    let book_fens: Vec<String> = (0..book.len()).map(|i| book.get(i).to_string()).collect();
    let stop_clone = Arc::clone(&stop);

    let search_handle = std::thread::spawn(move || {
        let mut inner_book = OpeningBook::from_fens(book_fens);
        elo_cmd::run_elo_test(
            &config_clone,
            &mut inner_book,
            stop_clone,
            move |progress| {
                if let Ok(mut guard) = progress_handle.lock() {
                    *guard = Some(progress.clone());
                }
            },
        )
    });

    let _ = elo_tui.run();
    let _final_progress = elo_tui.cleanup();

    stop.store(true, Ordering::Relaxed);
    if let Ok(report) = search_handle.join() {
        tui::print_elo_summary(&report);

        if let Ok(db_result) = ResultsDb::open_default() {
            let _ = db::save_elo_result(
                &db_result,
                &config.match_config.engine_a.display_name(),
                &config.match_config.engine_b.display_name(),
                &config.match_config.tc.to_string(),
                &report,
            );
        }
    }
}

fn run_elo_plain(config: &EloConfig, book: &mut OpeningBook, stop: Arc<AtomicBool>) {
    let plain = std::sync::Mutex::new(tui::PlainProgress::new());
    let report = elo_cmd::run_elo_test(config, book, stop, move |progress| {
        if let Ok(mut p) = plain.lock() {
            p.print_elo_progress(progress);
        }
    });
    tui::print_elo_summary(&report);

    if let Ok(db_result) = ResultsDb::open_default() {
        let _ = db::save_elo_result(
            &db_result,
            &config.match_config.engine_a.display_name(),
            &config.match_config.engine_b.display_name(),
            &config.match_config.tc.to_string(),
            &report,
        );
    }
}

fn parse_options(opts: &[String]) -> Vec<(String, String)> {
    opts.iter()
        .filter_map(|s| {
            let (name, value) = s.split_once('=')?;
            Some((name.trim().to_string(), value.trim().to_string()))
        })
        .collect()
}

fn format_options(opts: &[(String, String)]) -> String {
    if opts.is_empty() {
        return "default".to_string();
    }
    opts.iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn load_book(path: Option<&std::path::Path>) -> OpeningBook {
    match path {
        Some(p) => match OpeningBook::from_file(p) {
            Ok(book) => book,
            Err(e) => {
                eprintln!("Warning: {e}, using default openings");
                OpeningBook::default_book()
            }
        },
        None => OpeningBook::default_book(),
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

fn atty_stdout() -> bool {
    use std::io::IsTerminal;
    std::io::stdout().is_terminal()
}

/// Set up a Ctrl+C handler that sets the stop flag.
fn ctrlc_handler(stop: Arc<AtomicBool>) {
    // Use a simple signal-based approach
    let _ = std::thread::spawn(move || {
        // This is a blocking approach using POSIX signals
        // For now, rely on the TUI's own Ctrl+C handling
        // and the crossterm event loop. The stop flag is set
        // by the TUI when it receives Ctrl+C.
        loop {
            std::thread::sleep(std::time::Duration::from_secs(3600));
            if stop.load(Ordering::Relaxed) {
                break;
            }
        }
    });
}
