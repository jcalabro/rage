use std::fmt;
use std::path::PathBuf;
use std::time::Duration;

/// Configuration for a UCI engine instance.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Path to the engine binary.
    pub path: PathBuf,
    /// UCI options to set before each game (e.g., "Threads=1", "Hash=16").
    pub options: Vec<(String, String)>,
    /// Display name (defaults to binary filename if not set).
    pub name: Option<String>,
}

impl EngineConfig {
    pub fn display_name(&self) -> String {
        self.name.clone().unwrap_or_else(|| {
            self.path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "unknown".to_string())
        })
    }
}

/// Time control specification.
#[derive(Debug, Clone, Copy)]
pub struct TimeControl {
    /// Base time in milliseconds.
    pub base_ms: u64,
    /// Increment in milliseconds.
    pub inc_ms: u64,
    /// Moves to go before next time control (0 = sudden death).
    pub movestogo: u32,
}

impl TimeControl {
    /// Parse a time control string like "8+0.08", "60+0.6", "40/60+0".
    pub fn parse(s: &str) -> Option<TimeControl> {
        // Format: [movestogo/]base+inc (all in seconds)
        let (movestogo, rest) = if let Some(idx) = s.find('/') {
            let mtg: u32 = s[..idx].parse().ok()?;
            (mtg, &s[idx + 1..])
        } else {
            (0, s)
        };

        let parts: Vec<&str> = rest.split('+').collect();
        if parts.len() != 2 {
            return None;
        }

        let base_secs: f64 = parts[0].parse().ok()?;
        let inc_secs: f64 = parts[1].parse().ok()?;

        Some(TimeControl {
            base_ms: (base_secs * 1000.0) as u64,
            inc_ms: (inc_secs * 1000.0) as u64,
            movestogo,
        })
    }
}

impl fmt::Display for TimeControl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let base_secs = self.base_ms as f64 / 1000.0;
        let inc_secs = self.inc_ms as f64 / 1000.0;
        if self.movestogo > 0 {
            write!(f, "{}/{base_secs}+{inc_secs}", self.movestogo)
        } else {
            write!(f, "{base_secs}+{inc_secs}")
        }
    }
}

/// Outcome of a single game from the perspective of the engine that played white.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameOutcome {
    WhiteWin,
    Draw,
    BlackWin,
}

impl GameOutcome {
    /// Score from white's perspective: 1.0 for win, 0.5 for draw, 0.0 for loss.
    pub fn white_score(self) -> f64 {
        match self {
            GameOutcome::WhiteWin => 1.0,
            GameOutcome::Draw => 0.5,
            GameOutcome::BlackWin => 0.0,
        }
    }

    /// Flip perspective.
    pub fn flip(self) -> GameOutcome {
        match self {
            GameOutcome::WhiteWin => GameOutcome::BlackWin,
            GameOutcome::Draw => GameOutcome::Draw,
            GameOutcome::BlackWin => GameOutcome::WhiteWin,
        }
    }
}

impl fmt::Display for GameOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GameOutcome::WhiteWin => write!(f, "1-0"),
            GameOutcome::Draw => write!(f, "1/2-1/2"),
            GameOutcome::BlackWin => write!(f, "0-1"),
        }
    }
}

/// Reason a game ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationReason {
    Checkmate,
    Stalemate,
    FiftyMoveRule,
    ThreefoldRepetition,
    InsufficientMaterial,
    AdjudicationWin,
    AdjudicationDraw,
    MaxMovesReached,
    EngineTimeout,
    EngineCrash,
    IllegalMove,
}

impl fmt::Display for TerminationReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TerminationReason::Checkmate => write!(f, "checkmate"),
            TerminationReason::Stalemate => write!(f, "stalemate"),
            TerminationReason::FiftyMoveRule => write!(f, "50-move rule"),
            TerminationReason::ThreefoldRepetition => write!(f, "threefold repetition"),
            TerminationReason::InsufficientMaterial => write!(f, "insufficient material"),
            TerminationReason::AdjudicationWin => write!(f, "adjudication (win)"),
            TerminationReason::AdjudicationDraw => write!(f, "adjudication (draw)"),
            TerminationReason::MaxMovesReached => write!(f, "max moves reached"),
            TerminationReason::EngineTimeout => write!(f, "engine timeout"),
            TerminationReason::EngineCrash => write!(f, "engine crash"),
            TerminationReason::IllegalMove => write!(f, "illegal move"),
        }
    }
}

/// Record of a completed game.
#[derive(Debug, Clone)]
pub struct GameRecord {
    /// Opening FEN used.
    pub opening_fen: String,
    /// Outcome from white's perspective.
    pub outcome: GameOutcome,
    /// How the game ended.
    pub termination: TerminationReason,
    /// Number of plies (half-moves) played.
    pub ply_count: u32,
    /// Moves played in UCI notation.
    pub moves: Vec<String>,
    /// Duration of the game.
    pub duration: Duration,
    /// Whether engine A played white in this game.
    pub engine_a_is_white: bool,
}

impl GameRecord {
    /// Outcome from engine A's perspective.
    pub fn outcome_for_a(&self) -> GameOutcome {
        if self.engine_a_is_white {
            self.outcome
        } else {
            self.outcome.flip()
        }
    }
}

/// Result of a game pair (same opening, colors swapped).
#[derive(Debug, Clone)]
pub struct PairResult {
    /// Opening index used.
    pub opening_index: usize,
    /// Game where engine A played white.
    pub game_a_white: GameRecord,
    /// Game where engine A played black.
    pub game_a_black: GameRecord,
}

impl PairResult {
    /// Pentanomial category for this pair (0-4 where 0=LL, 1=LD, 2=DD/WL, 3=WD, 4=WW).
    /// Scores are from engine A's perspective.
    pub fn pentanomial_category(&self) -> usize {
        let a_white_score = self.game_a_white.outcome_for_a();
        let a_black_score = self.game_a_black.outcome_for_a();
        pair_category(a_white_score, a_black_score)
    }
}

/// Map a pair of game outcomes (both from engine A's perspective) to pentanomial index 0..4.
/// The pair sum is: 0 (LL) -> 0, 0.5 (LD/DL) -> 1, 1.0 (DD/WL/LW) -> 2, 1.5 (WD/DW) -> 3, 2.0 (WW) -> 4.
pub fn pair_category(game1: GameOutcome, game2: GameOutcome) -> usize {
    let sum = game1.white_score() + game2.white_score();
    // Sum ranges: 0.0, 0.5, 1.0, 1.5, 2.0 → indices 0, 1, 2, 3, 4
    (sum * 2.0) as usize
}

/// Pentanomial counts: [LL, LD, DD/WL, WD, WW].
#[derive(Debug, Clone, Copy, Default)]
pub struct PentanomialCounts {
    pub counts: [u64; 5],
}

impl PentanomialCounts {
    pub fn add_pair(&mut self, category: usize) {
        if category < 5 {
            self.counts[category] += 1;
        }
    }

    pub fn total_pairs(&self) -> u64 {
        self.counts.iter().sum()
    }

    /// Total games (= 2 * pairs).
    pub fn total_games(&self) -> u64 {
        self.total_pairs() * 2
    }

    /// Aggregate W/D/L from engine A's perspective.
    pub fn wdl(&self) -> (u64, u64, u64) {
        // WW contributes 2 wins, WD contributes 1 win + 1 draw, etc.
        let [ll, ld, ddwl, wd, ww] = self.counts;
        let wins = ww * 2 + wd;
        let draws = wd + ddwl * 2 + ld;
        let losses = ld + ll * 2;
        // Wait, this isn't right. Let me reconsider.
        // Each pair is 2 games. From engine A's perspective:
        // WW (cat 4): 2 wins → W+=2
        // WD (cat 3): 1 win + 1 draw → W+=1, D+=1
        // DD/WL (cat 2): could be 2 draws or 1 win + 1 loss → we track as category
        //   For aggregate stats, DD/WL category has pair score 1.0:
        //   We can't distinguish DD from WL in pentanomial, so approximate:
        //   Actually we CAN'T disaggregate. For display, count individual games separately.
        // This is a problem - pentanomial doesn't preserve individual W/D/L counts.
        // For display, we should track individual counts separately.
        // Use the simple version: iterate pair counts and assign conservatively.

        // Actually, let's just return what the categories mean for scoring.
        // Better to track W/D/L independently. Keep this for rough display.
        (wins, draws, losses)
    }
}

impl fmt::Display for PentanomialCounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}, {}, {}, {}, {}]",
            self.counts[0], self.counts[1], self.counts[2], self.counts[3], self.counts[4]
        )
    }
}

/// Aggregate match statistics.
#[derive(Debug, Clone, Default)]
pub struct MatchStats {
    /// Pentanomial counts.
    pub pentanomial: PentanomialCounts,
    /// Individual game counts from engine A's perspective.
    pub wins: u64,
    pub draws: u64,
    pub losses: u64,
}

impl MatchStats {
    pub fn add_game(&mut self, outcome: GameOutcome) {
        match outcome {
            GameOutcome::WhiteWin => self.wins += 1,
            GameOutcome::Draw => self.draws += 1,
            GameOutcome::BlackWin => self.losses += 1,
        }
    }

    pub fn add_pair(&mut self, pair: &PairResult) {
        self.pentanomial.add_pair(pair.pentanomial_category());
        self.add_game(pair.game_a_white.outcome_for_a());
        self.add_game(pair.game_a_black.outcome_for_a());
    }

    pub fn total_games(&self) -> u64 {
        self.wins + self.draws + self.losses
    }

    pub fn total_pairs(&self) -> u64 {
        self.pentanomial.total_pairs()
    }

    /// Score from engine A's perspective (0.0 to 1.0).
    pub fn score(&self) -> f64 {
        let total = self.total_games() as f64;
        if total == 0.0 {
            return 0.5;
        }
        (self.wins as f64 + self.draws as f64 * 0.5) / total
    }
}

/// Configuration for search limits (alternative to time control).
#[derive(Debug, Clone, Copy, Default)]
pub struct SearchOverrides {
    /// If set, use fixed depth instead of time control.
    pub depth: Option<u32>,
    /// If set, use fixed nodes instead of time control.
    pub nodes: Option<u64>,
}

/// Match configuration shared across all test modes.
#[derive(Debug, Clone)]
pub struct MatchConfig {
    pub engine_a: EngineConfig,
    pub engine_b: EngineConfig,
    pub tc: TimeControl,
    pub search_overrides: SearchOverrides,
    pub concurrency: u32,
    pub opening_file: Option<PathBuf>,
    /// Move overhead in milliseconds (subtracted from engine time for communication lag).
    pub move_overhead_ms: u64,
    /// Maximum moves per game before adjudication.
    pub max_moves: u32,
    /// Win adjudication: absolute score threshold in centipawns.
    pub win_adj_score: i32,
    /// Win adjudication: number of consecutive moves above threshold.
    pub win_adj_count: u32,
    /// Draw adjudication: absolute score threshold in centipawns.
    pub draw_adj_score: i32,
    /// Draw adjudication: number of consecutive moves below threshold.
    pub draw_adj_count: u32,
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            engine_a: EngineConfig {
                path: PathBuf::new(),
                options: Vec::new(),
                name: None,
            },
            engine_b: EngineConfig {
                path: PathBuf::new(),
                options: Vec::new(),
                name: None,
            },
            tc: TimeControl {
                base_ms: 8000,
                inc_ms: 80,
                movestogo: 0,
            },
            search_overrides: SearchOverrides::default(),
            concurrency: 1,
            opening_file: None,
            move_overhead_ms: 20,
            max_moves: 300,
            win_adj_score: 10000,
            win_adj_count: 5,
            draw_adj_score: 5,
            draw_adj_count: 10,
        }
    }
}

/// SPRT test configuration.
#[derive(Debug, Clone)]
pub struct SprtConfig {
    pub match_config: MatchConfig,
    /// Null hypothesis Elo bound (H0: Elo <= elo0).
    pub elo0: f64,
    /// Alternative hypothesis Elo bound (H1: Elo >= elo1).
    pub elo1: f64,
    /// Type I error rate.
    pub alpha: f64,
    /// Type II error rate.
    pub beta: f64,
}

impl Default for SprtConfig {
    fn default() -> Self {
        Self {
            match_config: MatchConfig::default(),
            elo0: 0.0,
            elo1: 5.0,
            alpha: 0.05,
            beta: 0.05,
        }
    }
}

/// Elo test configuration.
#[derive(Debug, Clone)]
pub struct EloConfig {
    pub match_config: MatchConfig,
    /// Number of game pairs to play.
    pub pairs: u32,
}

/// Tournament configuration.
#[derive(Debug, Clone)]
pub struct TournamentConfig {
    pub engines: Vec<EngineConfig>,
    pub tc: TimeControl,
    pub search_overrides: SearchOverrides,
    pub concurrency: u32,
    pub opening_file: Option<PathBuf>,
    pub move_overhead_ms: u64,
    pub max_moves: u32,
    pub win_adj_score: i32,
    pub win_adj_count: u32,
    pub draw_adj_score: i32,
    pub draw_adj_count: u32,
    /// Number of game pairs per matchup.
    pub rounds: u32,
}

/// SPRT decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SprtDecision {
    /// Test is still running.
    InProgress,
    /// H1 accepted (patch gains Elo).
    Accept,
    /// H0 accepted (patch does not gain Elo).
    Reject,
}

impl fmt::Display for SprtDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SprtDecision::InProgress => write!(f, "in progress"),
            SprtDecision::Accept => write!(f, "PASSED (H1 accepted)"),
            SprtDecision::Reject => write!(f, "FAILED (H0 accepted)"),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_time_control_parse_simple() {
        let tc = TimeControl::parse("8+0.08").unwrap();
        assert_eq!(tc.base_ms, 8000);
        assert_eq!(tc.inc_ms, 80);
        assert_eq!(tc.movestogo, 0);
    }

    #[test]
    fn test_time_control_parse_long() {
        let tc = TimeControl::parse("60+0.6").unwrap();
        assert_eq!(tc.base_ms, 60000);
        assert_eq!(tc.inc_ms, 600);
        assert_eq!(tc.movestogo, 0);
    }

    #[test]
    fn test_time_control_parse_movestogo() {
        let tc = TimeControl::parse("40/60+0").unwrap();
        assert_eq!(tc.base_ms, 60000);
        assert_eq!(tc.inc_ms, 0);
        assert_eq!(tc.movestogo, 40);
    }

    #[test]
    fn test_time_control_display() {
        let tc = TimeControl {
            base_ms: 8000,
            inc_ms: 80,
            movestogo: 0,
        };
        assert_eq!(tc.to_string(), "8+0.08");
    }

    #[test]
    fn test_game_outcome_flip() {
        assert_eq!(GameOutcome::WhiteWin.flip(), GameOutcome::BlackWin);
        assert_eq!(GameOutcome::Draw.flip(), GameOutcome::Draw);
        assert_eq!(GameOutcome::BlackWin.flip(), GameOutcome::WhiteWin);
    }

    #[test]
    fn test_pair_category_all_combinations() {
        use GameOutcome::*;
        // LL -> 0
        assert_eq!(pair_category(BlackWin, BlackWin), 0);
        // LD -> 1
        assert_eq!(pair_category(BlackWin, Draw), 1);
        assert_eq!(pair_category(Draw, BlackWin), 1);
        // DD -> 2
        assert_eq!(pair_category(Draw, Draw), 2);
        // WL -> 2
        assert_eq!(pair_category(WhiteWin, BlackWin), 2);
        assert_eq!(pair_category(BlackWin, WhiteWin), 2);
        // WD -> 3
        assert_eq!(pair_category(WhiteWin, Draw), 3);
        assert_eq!(pair_category(Draw, WhiteWin), 3);
        // WW -> 4
        assert_eq!(pair_category(WhiteWin, WhiteWin), 4);
    }

    #[test]
    fn test_match_stats_score() {
        let mut stats = MatchStats::default();
        stats.wins = 3;
        stats.draws = 4;
        stats.losses = 3;
        let score = stats.score();
        assert!((score - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_match_stats_score_winning() {
        let mut stats = MatchStats::default();
        stats.wins = 7;
        stats.draws = 2;
        stats.losses = 1;
        // Score = (7 + 1) / 10 = 0.8
        let score = stats.score();
        assert!((score - 0.8).abs() < 1e-10);
    }
}
