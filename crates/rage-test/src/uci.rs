//! UCI engine subprocess wrapper.
//!
//! Spawns a UCI engine as a child process and communicates via stdin/stdout.
//! Stdout is read on a dedicated thread and lines are sent via a channel,
//! enabling timeout support via `recv_timeout`.

use std::io::{BufRead, BufReader, Write as IoWrite};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;

use crate::types::EngineConfig;

/// Default timeout for engine responses.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Result of a "go" command.
#[derive(Debug, Clone)]
pub struct GoResult {
    /// Best move in UCI notation (e.g., "e2e4").
    pub bestmove: String,
    /// Ponder move if provided (e.g., "e7e5").
    pub ponder: Option<String>,
    /// Last reported score in centipawns (from engine's perspective).
    pub score_cp: Option<i32>,
    /// Whether the reported score was a mate score.
    pub is_mate_score: bool,
    /// Last reported depth.
    pub depth: u32,
    /// Last reported nodes.
    pub nodes: u64,
}

/// Parameters for a "go" command.
#[derive(Debug, Clone, Default)]
pub struct GoParams {
    pub wtime: Option<u64>,
    pub btime: Option<u64>,
    pub winc: Option<u64>,
    pub binc: Option<u64>,
    pub movestogo: Option<u32>,
    pub depth: Option<u32>,
    pub nodes: Option<u64>,
    pub movetime: Option<u64>,
}

impl GoParams {
    pub fn to_uci_string(&self) -> String {
        let mut parts = vec!["go".to_string()];
        if let Some(v) = self.wtime {
            parts.push(format!("wtime {v}"));
        }
        if let Some(v) = self.btime {
            parts.push(format!("btime {v}"));
        }
        if let Some(v) = self.winc {
            parts.push(format!("winc {v}"));
        }
        if let Some(v) = self.binc {
            parts.push(format!("binc {v}"));
        }
        if let Some(v) = self.movestogo {
            parts.push(format!("movestogo {v}"));
        }
        if let Some(v) = self.depth {
            parts.push(format!("depth {v}"));
        }
        if let Some(v) = self.nodes {
            parts.push(format!("nodes {v}"));
        }
        if let Some(v) = self.movetime {
            parts.push(format!("movetime {v}"));
        }
        parts.join(" ")
    }
}

/// Error type for UCI engine communication.
#[derive(Debug)]
pub enum UciError {
    /// Engine process failed to start.
    SpawnFailed(std::io::Error),
    /// Engine did not respond within the timeout.
    Timeout,
    /// Engine process exited unexpectedly.
    ProcessExited,
    /// Engine sent an unexpected response.
    UnexpectedResponse(String),
    /// I/O error communicating with engine.
    Io(std::io::Error),
    /// Engine failed to parse a UCI command.
    ProtocolError(String),
}

impl std::fmt::Display for UciError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UciError::SpawnFailed(e) => write!(f, "failed to spawn engine: {e}"),
            UciError::Timeout => write!(f, "engine timeout"),
            UciError::ProcessExited => write!(f, "engine process exited"),
            UciError::UnexpectedResponse(s) => write!(f, "unexpected response: {s}"),
            UciError::Io(e) => write!(f, "I/O error: {e}"),
            UciError::ProtocolError(s) => write!(f, "protocol error: {s}"),
        }
    }
}

/// A UCI engine subprocess.
pub struct UciEngine {
    child: Child,
    stdin: std::io::BufWriter<std::process::ChildStdin>,
    rx: Receiver<String>,
    /// Keep the sender alive so the reader thread doesn't panic.
    _tx: Sender<String>,
    /// Display name for this engine.
    pub name: String,
}

impl UciEngine {
    /// Spawn a new engine process and perform the UCI handshake.
    pub fn new(config: &EngineConfig, timeout: Duration) -> Result<Self, UciError> {
        let mut child = Command::new(&config.path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(UciError::SpawnFailed)?;

        let stdout = child.stdout.take().ok_or(UciError::ProcessExited)?;
        let child_stdin = child.stdin.take().ok_or(UciError::ProcessExited)?;

        let (tx, rx) = mpsc::channel();
        let tx_clone = tx.clone();

        // Reader thread: reads stdout line by line and sends to channel.
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                match line {
                    Ok(line) => {
                        if tx_clone.send(line).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        let stdin = std::io::BufWriter::new(child_stdin);
        let name = config.display_name();

        let mut engine = UciEngine {
            child,
            stdin,
            rx,
            _tx: tx,
            name,
        };

        // UCI handshake
        engine.send("uci")?;
        engine.wait_for("uciok", timeout)?;

        // Set UCI options
        for (name, value) in &config.options {
            engine.send(&format!("setoption name {name} value {value}"))?;
        }

        // Sync
        engine.sync(timeout)?;

        Ok(engine)
    }

    /// Send a command to the engine.
    pub fn send(&mut self, cmd: &str) -> Result<(), UciError> {
        writeln!(self.stdin, "{cmd}").map_err(UciError::Io)?;
        self.stdin.flush().map_err(UciError::Io)?;
        Ok(())
    }

    /// Send "isready" and wait for "readyok".
    pub fn sync(&mut self, timeout: Duration) -> Result<(), UciError> {
        self.send("isready")?;
        self.wait_for("readyok", timeout)
    }

    /// Send "ucinewgame" and sync.
    pub fn new_game(&mut self, timeout: Duration) -> Result<(), UciError> {
        self.send("ucinewgame")?;
        self.sync(timeout)
    }

    /// Set position with optional moves.
    pub fn set_position(&mut self, fen: &str, moves: &[String]) -> Result<(), UciError> {
        let cmd = if moves.is_empty() {
            format!("position fen {fen}")
        } else {
            format!("position fen {fen} moves {}", moves.join(" "))
        };
        self.send(&cmd)
    }

    /// Send "go" with the given parameters and wait for "bestmove".
    pub fn go(&mut self, params: &GoParams, timeout: Duration) -> Result<GoResult, UciError> {
        let cmd = params.to_uci_string();
        self.send(&cmd)?;

        let mut result = GoResult {
            bestmove: String::new(),
            ponder: None,
            score_cp: None,
            is_mate_score: false,
            depth: 0,
            nodes: 0,
        };

        loop {
            let line = self
                .rx
                .recv_timeout(timeout)
                .map_err(|_| UciError::Timeout)?;

            if line.starts_with("bestmove") {
                parse_bestmove(&line, &mut result);
                return Ok(result);
            } else if line.starts_with("info") {
                parse_info(&line, &mut result);
            }
            // Ignore other lines
        }
    }

    /// Send "quit" and wait for the process to exit.
    pub fn quit(mut self) {
        let _ = self.send("quit");
        let _ = self.child.wait();
    }

    /// Check if the engine process is still running.
    pub fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    /// Wait for a specific response line.
    fn wait_for(&self, target: &str, timeout: Duration) -> Result<(), UciError> {
        loop {
            let line = self
                .rx
                .recv_timeout(timeout)
                .map_err(|_| UciError::Timeout)?;
            if line.starts_with(target) {
                return Ok(());
            }
        }
    }
}

impl Drop for UciEngine {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}

/// Parse a "bestmove" line.
fn parse_bestmove(line: &str, result: &mut GoResult) {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() >= 2 {
        result.bestmove = parts[1].to_string();
    }
    if parts.len() >= 4 && parts[2] == "ponder" {
        result.ponder = Some(parts[3].to_string());
    }
}

/// Parse an "info" line, updating the result with the latest data.
fn parse_info(line: &str, result: &mut GoResult) {
    let parts: Vec<&str> = line.split_whitespace().collect();
    let mut i = 1; // skip "info"
    while i < parts.len() {
        match parts[i] {
            "depth" => {
                if let Some(v) = parts.get(i + 1).and_then(|s| s.parse().ok()) {
                    result.depth = v;
                }
                i += 2;
            }
            "nodes" => {
                if let Some(v) = parts.get(i + 1).and_then(|s| s.parse().ok()) {
                    result.nodes = v;
                }
                i += 2;
            }
            "score" => {
                if let Some(kind) = parts.get(i + 1) {
                    if let Some(v) = parts.get(i + 2).and_then(|s| s.parse().ok()) {
                        match *kind {
                            "cp" => {
                                result.score_cp = Some(v);
                                result.is_mate_score = false;
                            }
                            "mate" => {
                                result.score_cp = Some(v);
                                result.is_mate_score = true;
                            }
                            _ => {}
                        }
                    }
                    i += 3;
                } else {
                    i += 1;
                }
            }
            _ => {
                i += 1;
            }
        }
    }
}

/// Validate that an engine binary exists and is executable.
pub fn validate_engine_path(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Err(format!("engine binary not found: {}", path.display()));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = path.metadata()
            && meta.permissions().mode() & 0o111 == 0
        {
            return Err(format!(
                "engine binary is not executable: {}",
                path.display()
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_bestmove_simple() {
        let mut result = GoResult {
            bestmove: String::new(),
            ponder: None,
            score_cp: None,
            is_mate_score: false,
            depth: 0,
            nodes: 0,
        };
        parse_bestmove("bestmove e2e4", &mut result);
        assert_eq!(result.bestmove, "e2e4");
        assert!(result.ponder.is_none());
    }

    #[test]
    fn test_parse_bestmove_with_ponder() {
        let mut result = GoResult {
            bestmove: String::new(),
            ponder: None,
            score_cp: None,
            is_mate_score: false,
            depth: 0,
            nodes: 0,
        };
        parse_bestmove("bestmove e2e4 ponder e7e5", &mut result);
        assert_eq!(result.bestmove, "e2e4");
        assert_eq!(result.ponder.as_deref(), Some("e7e5"));
    }

    #[test]
    fn test_parse_info_score_cp() {
        let mut result = GoResult {
            bestmove: String::new(),
            ponder: None,
            score_cp: None,
            is_mate_score: false,
            depth: 0,
            nodes: 0,
        };
        parse_info("info depth 12 score cp 35 nodes 1234567", &mut result);
        assert_eq!(result.depth, 12);
        assert_eq!(result.score_cp, Some(35));
        assert!(!result.is_mate_score);
        assert_eq!(result.nodes, 1234567);
    }

    #[test]
    fn test_parse_info_score_mate() {
        let mut result = GoResult {
            bestmove: String::new(),
            ponder: None,
            score_cp: None,
            is_mate_score: false,
            depth: 0,
            nodes: 0,
        };
        parse_info("info depth 20 score mate 3", &mut result);
        assert_eq!(result.score_cp, Some(3));
        assert!(result.is_mate_score);
    }

    #[test]
    fn test_go_params_depth_only() {
        let params = GoParams {
            depth: Some(10),
            ..Default::default()
        };
        assert_eq!(params.to_uci_string(), "go depth 10");
    }

    #[test]
    fn test_go_params_full_tc() {
        let params = GoParams {
            wtime: Some(60000),
            btime: Some(60000),
            winc: Some(600),
            binc: Some(600),
            ..Default::default()
        };
        assert_eq!(
            params.to_uci_string(),
            "go wtime 60000 btime 60000 winc 600 binc 600"
        );
    }

    #[test]
    fn test_go_params_movestogo() {
        let params = GoParams {
            wtime: Some(60000),
            btime: Some(60000),
            movestogo: Some(40),
            ..Default::default()
        };
        assert_eq!(
            params.to_uci_string(),
            "go wtime 60000 btime 60000 movestogo 40"
        );
    }
}
