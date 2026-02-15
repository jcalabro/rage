//! Ratatui terminal UI for live test progress.

use std::io::{self, Stdout};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crossterm::ExecutableCommand;
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, Paragraph};

use crate::elo_cmd::EloProgress;
use crate::sprt::SprtProgress;
use crate::types::SprtDecision;

/// TUI state for SPRT test.
pub struct SprtTui {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    progress: Arc<Mutex<Option<SprtProgress>>>,
    stop: Arc<AtomicBool>,
    config_lines: Vec<String>,
}

impl SprtTui {
    pub fn new(stop: Arc<AtomicBool>, config_lines: Vec<String>) -> io::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        stdout.execute(EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;

        Ok(SprtTui {
            terminal,
            progress: Arc::new(Mutex::new(None)),
            stop,
            config_lines,
        })
    }

    /// Get a clone of the progress Arc for the callback to write to.
    pub fn progress_handle(&self) -> Arc<Mutex<Option<SprtProgress>>> {
        Arc::clone(&self.progress)
    }

    /// Run the TUI event loop. Blocks until the test is done or Ctrl+C.
    pub fn run(&mut self) -> io::Result<()> {
        loop {
            self.draw()?;

            // Poll for events with a short timeout (render at ~4 Hz)
            if event::poll(Duration::from_millis(250))?
                && let Event::Key(key) = event::read()?
            {
                if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                    self.stop.store(true, Ordering::Relaxed);
                    break;
                }
                if key.code == KeyCode::Char('q') {
                    self.stop.store(true, Ordering::Relaxed);
                    break;
                }
            }

            // Check if test is done
            if self.stop.load(Ordering::Relaxed) {
                self.draw()?;
                break;
            }
        }
        Ok(())
    }

    fn draw(&mut self) -> io::Result<()> {
        let progress = self.progress.lock().ok().and_then(|g| g.clone());
        let config_lines = &self.config_lines;

        self.terminal.draw(|frame| {
            let area = frame.area();

            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(config_lines.len() as u16 + 2),
                    Constraint::Min(10),
                ])
                .split(area);

            // Config block
            let config_text: Vec<Line> = config_lines
                .iter()
                .map(|l| Line::from(l.as_str()))
                .collect();
            let config_block = Paragraph::new(config_text)
                .block(Block::default().borders(Borders::ALL).title(" rage-test "));
            frame.render_widget(config_block, chunks[0]);

            // Stats block
            if let Some(ref p) = progress {
                render_sprt_stats(frame, chunks[1], p);
            } else {
                let waiting = Paragraph::new("Waiting for results...")
                    .block(Block::default().borders(Borders::ALL).title(" Results "));
                frame.render_widget(waiting, chunks[1]);
            }
        })?;
        Ok(())
    }

    /// Clean up terminal and print final summary.
    pub fn cleanup(mut self) -> SprtProgress {
        let _ = disable_raw_mode();
        let _ = self.terminal.backend_mut().execute(LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();

        self.progress
            .lock()
            .ok()
            .and_then(|g| g.clone())
            .unwrap_or_else(|| SprtProgress {
                stats: Default::default(),
                llr: 0.0,
                lower_bound: 0.0,
                upper_bound: 0.0,
                decision: SprtDecision::InProgress,
                elo: None,
                elapsed: Duration::ZERO,
                games_per_min: 0.0,
                latest_pair: None,
            })
    }
}

fn render_sprt_stats(frame: &mut ratatui::Frame, area: Rect, progress: &SprtProgress) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6),
            Constraint::Length(3),
            Constraint::Min(3),
        ])
        .split(area);

    let s = &progress.stats;
    let elo_str = progress
        .elo
        .map(|e| format!("{:+.1} +/- {:.1}", e.elo, e.error_margin))
        .unwrap_or_else(|| "N/A".to_string());
    let los_str = progress
        .elo
        .map(|e| format!("{:.1}%", e.los * 100.0))
        .unwrap_or_else(|| "N/A".to_string());
    let elapsed = format_duration(progress.elapsed);

    let decision_style = match progress.decision {
        SprtDecision::Accept => Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD),
        SprtDecision::Reject => Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        SprtDecision::InProgress => Style::default().fg(Color::Yellow),
    };

    let stats_lines = vec![
        Line::from(format!(
            "Games: {}    W: {}   D: {}   L: {}",
            s.total_games(),
            s.wins,
            s.draws,
            s.losses
        )),
        Line::from(format!("Elo: {elo_str}   LOS: {los_str}")),
        Line::from(vec![
            Span::raw(format!(
                "LLR: {:.2} / {:.2}  ",
                progress.llr, progress.upper_bound
            )),
            Span::styled(format!("{}", progress.decision), decision_style),
        ]),
        Line::from(format!("Pentanomial: {}", s.pentanomial)),
        Line::from(format!(
            "Speed: ~{:.0} games/min  |  Elapsed: {elapsed}",
            progress.games_per_min
        )),
    ];

    let stats_widget = Paragraph::new(stats_lines)
        .block(Block::default().borders(Borders::ALL).title(" Results "));
    frame.render_widget(stats_widget, chunks[0]);

    // LLR progress bar
    let llr_range = progress.upper_bound - progress.lower_bound;
    let llr_normalized = if llr_range > 0.0 {
        ((progress.llr - progress.lower_bound) / llr_range).clamp(0.0, 1.0)
    } else {
        0.5
    };

    let gauge_color = match progress.decision {
        SprtDecision::Accept => Color::Green,
        SprtDecision::Reject => Color::Red,
        SprtDecision::InProgress => Color::Blue,
    };

    let llr_display = if progress.llr.is_infinite() {
        if progress.llr > 0.0 {
            "+inf".to_string()
        } else {
            "-inf".to_string()
        }
    } else {
        format!("{:.2}", progress.llr)
    };

    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" LLR "))
        .gauge_style(Style::default().fg(gauge_color))
        .ratio(llr_normalized)
        .label(format!(
            "{llr_display} [{:.2}, {:.2}]",
            progress.lower_bound, progress.upper_bound
        ));
    frame.render_widget(gauge, chunks[1]);

    // Latest pair info
    if let Some(ref pair) = progress.latest_pair {
        let g1 = outcome_letter(pair.game_a_white.outcome_for_a());
        let g2 = outcome_letter(pair.game_a_black.outcome_for_a());
        let fen_preview =
            &pair.game_a_white.opening_fen[..pair.game_a_white.opening_fen.len().min(30)];
        let latest = Paragraph::new(format!(
            "Latest: pair #{}  {}  {}{} for A  ({} ply)",
            s.total_pairs(),
            fen_preview,
            g1,
            g2,
            pair.game_a_white.ply_count
        ))
        .block(Block::default().borders(Borders::ALL));
        frame.render_widget(latest, chunks[2]);
    }
}

/// ELO test TUI state.
pub struct EloTui {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    progress: Arc<Mutex<Option<EloProgress>>>,
    stop: Arc<AtomicBool>,
    config_lines: Vec<String>,
}

impl EloTui {
    pub fn new(stop: Arc<AtomicBool>, config_lines: Vec<String>) -> io::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        stdout.execute(EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;

        Ok(EloTui {
            terminal,
            progress: Arc::new(Mutex::new(None)),
            stop,
            config_lines,
        })
    }

    pub fn progress_handle(&self) -> Arc<Mutex<Option<EloProgress>>> {
        Arc::clone(&self.progress)
    }

    pub fn run(&mut self) -> io::Result<()> {
        loop {
            self.draw()?;

            if event::poll(Duration::from_millis(250))?
                && let Event::Key(key) = event::read()?
            {
                if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                    self.stop.store(true, Ordering::Relaxed);
                    break;
                }
                if key.code == KeyCode::Char('q') {
                    self.stop.store(true, Ordering::Relaxed);
                    break;
                }
            }

            if self.stop.load(Ordering::Relaxed) {
                self.draw()?;
                break;
            }
        }
        Ok(())
    }

    fn draw(&mut self) -> io::Result<()> {
        let progress = self.progress.lock().ok().and_then(|g| g.clone());
        let config_lines = &self.config_lines;

        self.terminal.draw(|frame| {
            let area = frame.area();

            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(config_lines.len() as u16 + 2),
                    Constraint::Min(8),
                ])
                .split(area);

            let config_text: Vec<Line> = config_lines
                .iter()
                .map(|l| Line::from(l.as_str()))
                .collect();
            let config_block = Paragraph::new(config_text)
                .block(Block::default().borders(Borders::ALL).title(" rage-test "));
            frame.render_widget(config_block, chunks[0]);

            if let Some(ref p) = progress {
                render_elo_stats(frame, chunks[1], p);
            } else {
                let waiting = Paragraph::new("Waiting for results...")
                    .block(Block::default().borders(Borders::ALL).title(" Results "));
                frame.render_widget(waiting, chunks[1]);
            }
        })?;
        Ok(())
    }

    pub fn cleanup(mut self) -> EloProgress {
        let _ = disable_raw_mode();
        let _ = self.terminal.backend_mut().execute(LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();

        self.progress
            .lock()
            .ok()
            .and_then(|g| g.clone())
            .unwrap_or_else(|| EloProgress {
                stats: Default::default(),
                pairs_target: 0,
                elo: None,
                elapsed: Duration::ZERO,
                games_per_min: 0.0,
                latest_pair: None,
            })
    }
}

fn render_elo_stats(frame: &mut ratatui::Frame, area: Rect, progress: &EloProgress) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6),
            Constraint::Length(3),
            Constraint::Min(1),
        ])
        .split(area);

    let s = &progress.stats;
    let elo_str = progress
        .elo
        .map(|e| format!("{:+.1} +/- {:.1}", e.elo, e.error_margin))
        .unwrap_or_else(|| "N/A".to_string());
    let los_str = progress
        .elo
        .map(|e| format!("{:.1}%", e.los * 100.0))
        .unwrap_or_else(|| "N/A".to_string());
    let elapsed = format_duration(progress.elapsed);

    let stats_lines = vec![
        Line::from(format!(
            "Pairs: {} / {}    Games: {}",
            s.total_pairs(),
            progress.pairs_target,
            s.total_games()
        )),
        Line::from(format!("W: {}   D: {}   L: {}", s.wins, s.draws, s.losses)),
        Line::from(format!("Elo: {elo_str}   LOS: {los_str}")),
        Line::from(format!("Pentanomial: {}", s.pentanomial)),
        Line::from(format!(
            "Speed: ~{:.0} games/min  |  Elapsed: {elapsed}",
            progress.games_per_min
        )),
    ];

    let stats_widget = Paragraph::new(stats_lines)
        .block(Block::default().borders(Borders::ALL).title(" Results "));
    frame.render_widget(stats_widget, chunks[0]);

    // Progress bar
    let ratio = if progress.pairs_target > 0 {
        (s.total_pairs() as f64 / progress.pairs_target as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" Progress "))
        .gauge_style(Style::default().fg(Color::Blue))
        .ratio(ratio)
        .label(format!(
            "{} / {} pairs",
            s.total_pairs(),
            progress.pairs_target
        ));
    frame.render_widget(gauge, chunks[1]);
}

/// Non-interactive progress printer for piped output.
pub struct PlainProgress {
    last_print_pairs: u64,
}

impl PlainProgress {
    pub fn new() -> Self {
        PlainProgress {
            last_print_pairs: 0,
        }
    }

    pub fn print_sprt_progress(&mut self, p: &SprtProgress) {
        let pairs = p.stats.total_pairs();
        // Print every 10 pairs or on decision
        if pairs >= self.last_print_pairs + 10 || p.decision != SprtDecision::InProgress {
            let elo_str = p
                .elo
                .map(|e| format!("{:+.1}+/-{:.1}", e.elo, e.error_margin))
                .unwrap_or_else(|| "N/A".to_string());
            println!(
                "pairs={} W={} D={} L={} Elo={} LLR={:.2}/[{:.2},{:.2}] {}",
                pairs,
                p.stats.wins,
                p.stats.draws,
                p.stats.losses,
                elo_str,
                p.llr,
                p.lower_bound,
                p.upper_bound,
                p.decision,
            );
            self.last_print_pairs = pairs;
        }
    }

    pub fn print_elo_progress(&mut self, p: &EloProgress) {
        let pairs = p.stats.total_pairs();
        if pairs >= self.last_print_pairs + 10 || pairs == p.pairs_target as u64 {
            let elo_str = p
                .elo
                .map(|e| format!("{:+.1}+/-{:.1}", e.elo, e.error_margin))
                .unwrap_or_else(|| "N/A".to_string());
            println!(
                "pairs={}/{} W={} D={} L={} Elo={}",
                pairs, p.pairs_target, p.stats.wins, p.stats.draws, p.stats.losses, elo_str,
            );
            self.last_print_pairs = pairs;
        }
    }
}

/// Format a game outcome from engine A's perspective as a single letter.
fn outcome_letter(outcome: crate::types::GameOutcome) -> char {
    match outcome {
        crate::types::GameOutcome::WhiteWin => 'W',
        crate::types::GameOutcome::Draw => 'D',
        crate::types::GameOutcome::BlackWin => 'L',
    }
}

fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    let mins = secs / 60;
    let secs = secs % 60;
    if mins > 0 {
        format!("{mins}m {secs:02}s")
    } else {
        format!("{secs}s")
    }
}

/// Print a final SPRT summary to stdout (after TUI cleanup).
pub fn print_sprt_summary(report: &crate::sprt::SprtReport) {
    println!();
    println!("===============================================");
    println!("Result: {}", report.decision);
    println!(
        "Games: {} pairs  |  W: {}  D: {}  L: {}",
        report.stats.total_pairs(),
        report.stats.wins,
        report.stats.draws,
        report.stats.losses,
    );
    if let Some(elo) = report.elo {
        println!(
            "Elo: {:+.1} +/- {:.1} (95%)   LOS: {:.1}%",
            elo.elo,
            elo.error_margin,
            elo.los * 100.0
        );
    }
    println!(
        "LLR: {:.2}  [{:.2}, {:.2}]",
        report.llr, report.lower_bound, report.upper_bound
    );
    println!("Pentanomial: {}", report.stats.pentanomial);
    println!(
        "Duration: {}  |  ~{:.0} games/min",
        format_duration(report.elapsed),
        report.games_per_min
    );
    println!("===============================================");
}

/// Print a final Elo summary to stdout.
pub fn print_elo_summary(report: &crate::elo_cmd::EloReport) {
    println!();
    println!("===============================================");
    println!(
        "Games: {} pairs  |  W: {}  D: {}  L: {}",
        report.stats.total_pairs(),
        report.stats.wins,
        report.stats.draws,
        report.stats.losses,
    );
    if let Some(elo) = report.elo {
        println!(
            "Elo: {:+.1} +/- {:.1} (95%)   LOS: {:.1}%",
            elo.elo,
            elo.error_margin,
            elo.los * 100.0
        );
    }
    println!("Pentanomial: {}", report.stats.pentanomial);
    println!(
        "Duration: {}  |  ~{:.0} games/min",
        format_duration(report.elapsed),
        report.games_per_min
    );
    println!("===============================================");
}

/// Print a tournament summary to stdout.
pub fn print_tournament_summary(report: &crate::tournament::TournamentReport) {
    println!();
    println!("================= Tournament Results =================");
    println!("Duration: {}", format_duration(report.elapsed));
    println!();

    // Rankings
    println!("Rankings:");
    for (rank, &(idx, score)) in report.rankings.iter().enumerate() {
        let elo = crate::stats::elo_from_score(score);
        println!(
            "  {}. {} — score {:.1}% — Elo {:+.1}",
            rank + 1,
            report.engines[idx],
            score * 100.0,
            elo
        );
    }
    println!();

    // Cross-table
    println!("Matchups:");
    for m in &report.matchups {
        let elo_str = m
            .elo
            .map(|e| format!("{:+.1}+/-{:.1}", e.elo, e.error_margin))
            .unwrap_or_else(|| "N/A".to_string());
        println!(
            "  {} vs {}: W={} D={} L={} Elo={}",
            report.engines[m.engine_a_idx],
            report.engines[m.engine_b_idx],
            m.stats.wins,
            m.stats.draws,
            m.stats.losses,
            elo_str,
        );
    }
    println!("=====================================================");
}
