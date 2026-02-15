//! SQLite results database.
//!
//! Stores test results for historical tracking and analysis.

use std::path::{Path, PathBuf};

use rusqlite::{Connection, params};

use crate::types::SprtDecision;

/// Result data for finishing a test.
pub struct TestResult {
    pub pairs: i64,
    pub wins: i64,
    pub draws: i64,
    pub losses: i64,
    pub elo: Option<f64>,
    pub elo_error: Option<f64>,
    pub llr: Option<f64>,
    pub decision: Option<String>,
    pub elapsed_secs: f64,
}

/// Database for storing test results.
pub struct ResultsDb {
    conn: Connection,
}

/// A test record as stored in the database.
#[derive(Debug, Clone)]
pub struct TestRecord {
    pub id: i64,
    pub test_type: String,
    pub engine_a: String,
    pub engine_b: String,
    pub tc: String,
    pub elo0: Option<f64>,
    pub elo1: Option<f64>,
    pub pairs: i64,
    pub wins: i64,
    pub draws: i64,
    pub losses: i64,
    pub elo: Option<f64>,
    pub elo_error: Option<f64>,
    pub llr: Option<f64>,
    pub decision: Option<String>,
    pub elapsed_secs: f64,
    pub created_at: String,
}

impl ResultsDb {
    /// Open or create the results database at the default location.
    pub fn open_default() -> Result<Self, String> {
        let path = default_db_path()?;
        Self::open(&path)
    }

    /// Open or create the results database at a specific path.
    pub fn open(path: &Path) -> Result<Self, String> {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create database directory: {e}"))?;
        }

        let conn = Connection::open(path).map_err(|e| format!("failed to open database: {e}"))?;

        let db = ResultsDb { conn };
        db.create_tables()?;
        Ok(db)
    }

    fn create_tables(&self) -> Result<(), String> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_type TEXT NOT NULL,
                    engine_a TEXT NOT NULL,
                    engine_b TEXT NOT NULL,
                    tc TEXT NOT NULL,
                    elo0 REAL,
                    elo1 REAL,
                    pairs INTEGER NOT NULL DEFAULT 0,
                    wins INTEGER NOT NULL DEFAULT 0,
                    draws INTEGER NOT NULL DEFAULT 0,
                    losses INTEGER NOT NULL DEFAULT 0,
                    elo REAL,
                    elo_error REAL,
                    llr REAL,
                    decision TEXT,
                    elapsed_secs REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );",
            )
            .map_err(|e| format!("failed to create tables: {e}"))?;
        Ok(())
    }

    /// Create a new test record. Returns the test ID.
    pub fn create_test(
        &self,
        test_type: &str,
        engine_a: &str,
        engine_b: &str,
        tc: &str,
        elo0: Option<f64>,
        elo1: Option<f64>,
    ) -> Result<i64, String> {
        self.conn
            .execute(
                "INSERT INTO tests (test_type, engine_a, engine_b, tc, elo0, elo1)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![test_type, engine_a, engine_b, tc, elo0, elo1],
            )
            .map_err(|e| format!("failed to create test: {e}"))?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Update a test record with final results.
    pub fn finish_test(&self, test_id: i64, result: &TestResult) -> Result<(), String> {
        self.conn
            .execute(
                "UPDATE tests SET pairs=?2, wins=?3, draws=?4, losses=?5,
                 elo=?6, elo_error=?7, llr=?8, decision=?9, elapsed_secs=?10
                 WHERE id=?1",
                params![
                    test_id,
                    result.pairs,
                    result.wins,
                    result.draws,
                    result.losses,
                    result.elo,
                    result.elo_error,
                    result.llr,
                    result.decision,
                    result.elapsed_secs
                ],
            )
            .map_err(|e| format!("failed to update test: {e}"))?;
        Ok(())
    }

    /// List recent test records.
    pub fn list_tests(&self, limit: u32) -> Result<Vec<TestRecord>, String> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, test_type, engine_a, engine_b, tc, elo0, elo1,
                        pairs, wins, draws, losses, elo, elo_error, llr,
                        decision, elapsed_secs, created_at
                 FROM tests ORDER BY id DESC LIMIT ?1",
            )
            .map_err(|e| format!("failed to prepare query: {e}"))?;

        let records = stmt
            .query_map(params![limit], |row| {
                Ok(TestRecord {
                    id: row.get(0)?,
                    test_type: row.get(1)?,
                    engine_a: row.get(2)?,
                    engine_b: row.get(3)?,
                    tc: row.get(4)?,
                    elo0: row.get(5)?,
                    elo1: row.get(6)?,
                    pairs: row.get(7)?,
                    wins: row.get(8)?,
                    draws: row.get(9)?,
                    losses: row.get(10)?,
                    elo: row.get(11)?,
                    elo_error: row.get(12)?,
                    llr: row.get(13)?,
                    decision: row.get(14)?,
                    elapsed_secs: row.get(15)?,
                    created_at: row.get(16)?,
                })
            })
            .map_err(|e| format!("failed to query tests: {e}"))?;

        let mut result = Vec::new();
        for record in records {
            result.push(record.map_err(|e| format!("failed to read test record: {e}"))?);
        }
        Ok(result)
    }

    /// Get a single test record by ID.
    pub fn get_test(&self, test_id: i64) -> Result<Option<TestRecord>, String> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, test_type, engine_a, engine_b, tc, elo0, elo1,
                        pairs, wins, draws, losses, elo, elo_error, llr,
                        decision, elapsed_secs, created_at
                 FROM tests WHERE id=?1",
            )
            .map_err(|e| format!("failed to prepare query: {e}"))?;

        let mut rows = stmt
            .query_map(params![test_id], |row| {
                Ok(TestRecord {
                    id: row.get(0)?,
                    test_type: row.get(1)?,
                    engine_a: row.get(2)?,
                    engine_b: row.get(3)?,
                    tc: row.get(4)?,
                    elo0: row.get(5)?,
                    elo1: row.get(6)?,
                    pairs: row.get(7)?,
                    wins: row.get(8)?,
                    draws: row.get(9)?,
                    losses: row.get(10)?,
                    elo: row.get(11)?,
                    elo_error: row.get(12)?,
                    llr: row.get(13)?,
                    decision: row.get(14)?,
                    elapsed_secs: row.get(15)?,
                    created_at: row.get(16)?,
                })
            })
            .map_err(|e| format!("failed to query test: {e}"))?;

        match rows.next() {
            Some(Ok(record)) => Ok(Some(record)),
            Some(Err(e)) => Err(format!("failed to read test record: {e}")),
            None => Ok(None),
        }
    }
}

/// Default database path: ~/.rage-test/results.db
fn default_db_path() -> Result<PathBuf, String> {
    let home = std::env::var("HOME").map_err(|_| "HOME environment variable not set")?;
    Ok(PathBuf::from(home).join(".rage-test").join("results.db"))
}

/// Save SPRT results to the database.
pub fn save_sprt_result(
    db: &ResultsDb,
    engine_a: &str,
    engine_b: &str,
    tc: &str,
    report: &crate::sprt::SprtReport,
    elo0: f64,
    elo1: f64,
) -> Result<i64, String> {
    let test_id = db.create_test("sprt", engine_a, engine_b, tc, Some(elo0), Some(elo1))?;
    let decision_str = match report.decision {
        SprtDecision::Accept => "PASSED",
        SprtDecision::Reject => "FAILED",
        SprtDecision::InProgress => "STOPPED",
    };
    db.finish_test(
        test_id,
        &TestResult {
            pairs: report.stats.total_pairs() as i64,
            wins: report.stats.wins as i64,
            draws: report.stats.draws as i64,
            losses: report.stats.losses as i64,
            elo: report.elo.map(|e| e.elo),
            elo_error: report.elo.map(|e| e.error_margin),
            llr: Some(report.llr),
            decision: Some(decision_str.to_string()),
            elapsed_secs: report.elapsed.as_secs_f64(),
        },
    )?;
    Ok(test_id)
}

/// Save Elo test results to the database.
pub fn save_elo_result(
    db: &ResultsDb,
    engine_a: &str,
    engine_b: &str,
    tc: &str,
    report: &crate::elo_cmd::EloReport,
) -> Result<i64, String> {
    let test_id = db.create_test("elo", engine_a, engine_b, tc, None, None)?;
    db.finish_test(
        test_id,
        &TestResult {
            pairs: report.stats.total_pairs() as i64,
            wins: report.stats.wins as i64,
            draws: report.stats.draws as i64,
            losses: report.stats.losses as i64,
            elo: report.elo.map(|e| e.elo),
            elo_error: report.elo.map(|e| e.error_margin),
            llr: None,
            decision: Some("COMPLETE".to_string()),
            elapsed_secs: report.elapsed.as_secs_f64(),
        },
    )?;
    Ok(test_id)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn temp_db() -> ResultsDb {
        ResultsDb::open(Path::new(":memory:")).unwrap()
    }

    #[test]
    fn test_create_and_get_test() {
        let db = temp_db();
        let id = db
            .create_test(
                "sprt",
                "engine-a",
                "engine-b",
                "8+0.08",
                Some(0.0),
                Some(5.0),
            )
            .unwrap();
        assert_eq!(id, 1);

        let record = db.get_test(id).unwrap().unwrap();
        assert_eq!(record.test_type, "sprt");
        assert_eq!(record.engine_a, "engine-a");
        assert_eq!(record.engine_b, "engine-b");
        assert_eq!(record.tc, "8+0.08");
    }

    #[test]
    fn test_finish_test() {
        let db = temp_db();
        let id = db
            .create_test("elo", "a", "b", "60+0.6", None, None)
            .unwrap();
        db.finish_test(
            id,
            &TestResult {
                pairs: 100,
                wins: 30,
                draws: 40,
                losses: 30,
                elo: Some(3.5),
                elo_error: Some(12.0),
                llr: None,
                decision: Some("COMPLETE".to_string()),
                elapsed_secs: 120.5,
            },
        )
        .unwrap();

        let record = db.get_test(id).unwrap().unwrap();
        assert_eq!(record.pairs, 100);
        assert_eq!(record.wins, 30);
        assert_eq!(record.draws, 40);
        assert_eq!(record.losses, 30);
        assert!((record.elo.unwrap() - 3.5).abs() < 1e-6);
        assert_eq!(record.decision.as_deref(), Some("COMPLETE"));
    }

    #[test]
    fn test_list_tests() {
        let db = temp_db();
        for i in 0..5 {
            db.create_test("sprt", &format!("a{i}"), "b", "8+0.08", None, None)
                .unwrap();
        }
        let records = db.list_tests(3).unwrap();
        assert_eq!(records.len(), 3);
        // Should be in reverse order (most recent first)
        assert_eq!(records[0].id, 5);
        assert_eq!(records[1].id, 4);
        assert_eq!(records[2].id, 3);
    }

    #[test]
    fn test_get_nonexistent() {
        let db = temp_db();
        assert!(db.get_test(999).unwrap().is_none());
    }

    #[test]
    fn test_schema_idempotent() {
        let db = temp_db();
        // Creating tables again should succeed
        db.create_tables().unwrap();
        db.create_tables().unwrap();
    }
}
