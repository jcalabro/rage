//! Tablebase collection: directory scanning, lazy table loading, and
//! the main `Tablebase` struct that holds all loaded tables.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;

use crate::cache::BlockCache;
use crate::errors::{Result, SyzygyError};
use crate::table::{Table, TableType};

/// A collection of Syzygy tablebases loaded from one or more directories.
pub struct Tablebase {
    /// Paths to search for table files.
    paths: Vec<PathBuf>,
    /// WDL tables, keyed by material name (e.g. "KRvK").
    wdl_tables: RwLock<HashMap<String, Table>>,
    /// DTZ tables, keyed by material name.
    dtz_tables: RwLock<HashMap<String, Table>>,
    /// Block cache for decompressed data.
    pub cache: BlockCache,
    /// Maximum number of pieces supported.
    pub max_pieces: u32,
    /// Probe limit (max pieces for which to probe).
    pub probe_limit: u32,
    /// Probe depth (minimum search depth before probing).
    pub probe_depth: u32,
}

impl Tablebase {
    /// Create a new tablebase collection from a colon/semicolon-separated path string.
    pub fn new(path: &str) -> Self {
        let paths: Vec<PathBuf> = path
            .split([':',  ';'])
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .collect();

        // Scan for available tables
        let mut max_pieces = 0u32;
        let mut available = Vec::new();

        for dir in &paths {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(ext) = path.extension().and_then(|e| e.to_str())
                        && (ext == "rtbw" || ext == "rtbz")
                        && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                    {
                        let piece_count = count_pieces_from_name(stem);
                        if piece_count > max_pieces {
                            max_pieces = piece_count;
                        }
                        available.push(stem.to_string());
                    }
                }
            }
        }

        if !available.is_empty() {
            let unique: std::collections::HashSet<&str> =
                available.iter().map(|s| s.as_str()).collect();
            eprintln!(
                "info string found {} Syzygy tables (up to {max_pieces} pieces)",
                unique.len()
            );
        }

        Self {
            paths,
            wdl_tables: RwLock::new(HashMap::new()),
            dtz_tables: RwLock::new(HashMap::new()),
            cache: BlockCache::new(256),
            max_pieces,
            probe_limit: max_pieces.min(7),
            probe_depth: 1,
        }
    }

    /// Set the probe depth (minimum search depth before probing).
    pub fn set_probe_depth(&mut self, depth: u32) {
        self.probe_depth = depth;
    }

    /// Set the probe limit (maximum pieces for probing).
    pub fn set_probe_limit(&mut self, limit: u32) {
        self.probe_limit = limit.min(self.max_pieces);
    }

    /// Check if a WDL table is available for the given material name.
    pub fn has_wdl(&self, name: &str) -> bool {
        if let Ok(tables) = self.wdl_tables.read()
            && tables.contains_key(name)
        {
            return true;
        }
        self.find_table_file(name, "rtbw").is_some()
    }

    /// Check if a DTZ table is available for the given material name.
    pub fn has_dtz(&self, name: &str) -> bool {
        if let Ok(tables) = self.dtz_tables.read()
            && tables.contains_key(name)
        {
            return true;
        }
        self.find_table_file(name, "rtbz").is_some()
    }

    /// Get a reference to a loaded WDL table, loading it lazily if needed.
    pub fn get_wdl(&self, name: &str) -> Result<()> {
        if let Ok(tables) = self.wdl_tables.read()
            && tables.contains_key(name)
        {
            return Ok(());
        }

        let path = self
            .find_table_file(name, "rtbw")
            .ok_or_else(|| SyzygyError::MissingTable(name.to_string()))?;

        let table = Table::load(&path, TableType::Wdl)?;

        if let Ok(mut tables) = self.wdl_tables.write() {
            tables.insert(name.to_string(), table);
        }

        Ok(())
    }

    /// Get a reference to a loaded DTZ table, loading it lazily if needed.
    pub fn get_dtz(&self, name: &str) -> Result<()> {
        if let Ok(tables) = self.dtz_tables.read()
            && tables.contains_key(name)
        {
            return Ok(());
        }

        let path = self
            .find_table_file(name, "rtbz")
            .ok_or_else(|| SyzygyError::MissingTable(name.to_string()))?;

        let table = Table::load(&path, TableType::Dtz)?;

        if let Ok(mut tables) = self.dtz_tables.write() {
            tables.insert(name.to_string(), table);
        }

        Ok(())
    }

    /// Access a loaded WDL table for reading. Returns None if not loaded.
    pub fn read_wdl<F, T>(&self, name: &str, f: F) -> Option<T>
    where
        F: FnOnce(&Table) -> T,
    {
        self.wdl_tables
            .read()
            .ok()
            .and_then(|tables| tables.get(name).map(f))
    }

    /// Access a loaded DTZ table for reading. Returns None if not loaded.
    pub fn read_dtz<F, T>(&self, name: &str, f: F) -> Option<T>
    where
        F: FnOnce(&Table) -> T,
    {
        self.dtz_tables
            .read()
            .ok()
            .and_then(|tables| tables.get(name).map(f))
    }

    /// Find the path to a table file across all search directories.
    fn find_table_file(&self, name: &str, ext: &str) -> Option<PathBuf> {
        let filename = format!("{name}.{ext}");
        for dir in &self.paths {
            let path = dir.join(&filename);
            if path.exists() {
                return Some(path);
            }
        }
        None
    }

    /// Returns true if any tables are available.
    pub fn is_available(&self) -> bool {
        self.max_pieces > 0
    }
}

/// Count total pieces from a table name like "KRvK" -> 3, "KQRvKR" -> 5.
fn count_pieces_from_name(name: &str) -> u32 {
    name.chars()
        .filter(|c| matches!(c, 'K' | 'Q' | 'R' | 'B' | 'N' | 'P'))
        .count() as u32
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn count_pieces() {
        assert_eq!(count_pieces_from_name("KRvK"), 3);
        assert_eq!(count_pieces_from_name("KQRvKR"), 5);
        assert_eq!(count_pieces_from_name("KPvKP"), 4);
        assert_eq!(count_pieces_from_name("KBNvK"), 4);
        assert_eq!(count_pieces_from_name("KRBNPPvKQRP"), 10);
    }

    #[test]
    fn empty_tablebase() {
        let tb = Tablebase::new("");
        assert!(!tb.is_available());
        assert_eq!(tb.max_pieces, 0);
    }

    #[test]
    fn nonexistent_path() {
        let tb = Tablebase::new("/nonexistent/path/to/syzygy");
        assert!(!tb.is_available());
    }

    #[test]
    fn missing_table_returns_error() {
        let tb = Tablebase::new("");
        let result = tb.get_wdl("KRvK");
        assert!(result.is_err());
    }
}
