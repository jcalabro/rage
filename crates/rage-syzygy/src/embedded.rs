//! Embedded 3+4 piece Syzygy tablebase files.
//!
//! These tables are compiled into the binary via `include_bytes!` so the
//! engine ships with perfect endgame play for positions with up to 4 pieces
//! out of the box, without requiring any external files.
//!
//! Total embedded data: ~4.4 MB (70 files: 35 endings x 2 table types).

use std::path::PathBuf;

/// Each embedded table: (filename, file contents).
const EMBEDDED_TABLES: &[(&str, &[u8])] = &[
    // 3-piece tables
    (
        "KBvK.rtbw",
        include_bytes!("../../../tables/syzygy/KBvK.rtbw"),
    ),
    (
        "KBvK.rtbz",
        include_bytes!("../../../tables/syzygy/KBvK.rtbz"),
    ),
    (
        "KNvK.rtbw",
        include_bytes!("../../../tables/syzygy/KNvK.rtbw"),
    ),
    (
        "KNvK.rtbz",
        include_bytes!("../../../tables/syzygy/KNvK.rtbz"),
    ),
    (
        "KPvK.rtbw",
        include_bytes!("../../../tables/syzygy/KPvK.rtbw"),
    ),
    (
        "KPvK.rtbz",
        include_bytes!("../../../tables/syzygy/KPvK.rtbz"),
    ),
    (
        "KQvK.rtbw",
        include_bytes!("../../../tables/syzygy/KQvK.rtbw"),
    ),
    (
        "KQvK.rtbz",
        include_bytes!("../../../tables/syzygy/KQvK.rtbz"),
    ),
    (
        "KRvK.rtbw",
        include_bytes!("../../../tables/syzygy/KRvK.rtbw"),
    ),
    (
        "KRvK.rtbz",
        include_bytes!("../../../tables/syzygy/KRvK.rtbz"),
    ),
    // 4-piece tables (same side)
    (
        "KBBvK.rtbw",
        include_bytes!("../../../tables/syzygy/KBBvK.rtbw"),
    ),
    (
        "KBBvK.rtbz",
        include_bytes!("../../../tables/syzygy/KBBvK.rtbz"),
    ),
    (
        "KBNvK.rtbw",
        include_bytes!("../../../tables/syzygy/KBNvK.rtbw"),
    ),
    (
        "KBNvK.rtbz",
        include_bytes!("../../../tables/syzygy/KBNvK.rtbz"),
    ),
    (
        "KBPvK.rtbw",
        include_bytes!("../../../tables/syzygy/KBPvK.rtbw"),
    ),
    (
        "KBPvK.rtbz",
        include_bytes!("../../../tables/syzygy/KBPvK.rtbz"),
    ),
    (
        "KNNvK.rtbw",
        include_bytes!("../../../tables/syzygy/KNNvK.rtbw"),
    ),
    (
        "KNNvK.rtbz",
        include_bytes!("../../../tables/syzygy/KNNvK.rtbz"),
    ),
    (
        "KNPvK.rtbw",
        include_bytes!("../../../tables/syzygy/KNPvK.rtbw"),
    ),
    (
        "KNPvK.rtbz",
        include_bytes!("../../../tables/syzygy/KNPvK.rtbz"),
    ),
    (
        "KPPvK.rtbw",
        include_bytes!("../../../tables/syzygy/KPPvK.rtbw"),
    ),
    (
        "KPPvK.rtbz",
        include_bytes!("../../../tables/syzygy/KPPvK.rtbz"),
    ),
    (
        "KQBvK.rtbw",
        include_bytes!("../../../tables/syzygy/KQBvK.rtbw"),
    ),
    (
        "KQBvK.rtbz",
        include_bytes!("../../../tables/syzygy/KQBvK.rtbz"),
    ),
    (
        "KQNvK.rtbw",
        include_bytes!("../../../tables/syzygy/KQNvK.rtbw"),
    ),
    (
        "KQNvK.rtbz",
        include_bytes!("../../../tables/syzygy/KQNvK.rtbz"),
    ),
    (
        "KQPvK.rtbw",
        include_bytes!("../../../tables/syzygy/KQPvK.rtbw"),
    ),
    (
        "KQPvK.rtbz",
        include_bytes!("../../../tables/syzygy/KQPvK.rtbz"),
    ),
    (
        "KQQvK.rtbw",
        include_bytes!("../../../tables/syzygy/KQQvK.rtbw"),
    ),
    (
        "KQQvK.rtbz",
        include_bytes!("../../../tables/syzygy/KQQvK.rtbz"),
    ),
    (
        "KQRvK.rtbw",
        include_bytes!("../../../tables/syzygy/KQRvK.rtbw"),
    ),
    (
        "KQRvK.rtbz",
        include_bytes!("../../../tables/syzygy/KQRvK.rtbz"),
    ),
    (
        "KRBvK.rtbw",
        include_bytes!("../../../tables/syzygy/KRBvK.rtbw"),
    ),
    (
        "KRBvK.rtbz",
        include_bytes!("../../../tables/syzygy/KRBvK.rtbz"),
    ),
    (
        "KRNvK.rtbw",
        include_bytes!("../../../tables/syzygy/KRNvK.rtbw"),
    ),
    (
        "KRNvK.rtbz",
        include_bytes!("../../../tables/syzygy/KRNvK.rtbz"),
    ),
    (
        "KRPvK.rtbw",
        include_bytes!("../../../tables/syzygy/KRPvK.rtbw"),
    ),
    (
        "KRPvK.rtbz",
        include_bytes!("../../../tables/syzygy/KRPvK.rtbz"),
    ),
    (
        "KRRvK.rtbw",
        include_bytes!("../../../tables/syzygy/KRRvK.rtbw"),
    ),
    (
        "KRRvK.rtbz",
        include_bytes!("../../../tables/syzygy/KRRvK.rtbz"),
    ),
    // 4-piece tables (split sides)
    (
        "KBvKB.rtbw",
        include_bytes!("../../../tables/syzygy/KBvKB.rtbw"),
    ),
    (
        "KBvKB.rtbz",
        include_bytes!("../../../tables/syzygy/KBvKB.rtbz"),
    ),
    (
        "KBvKN.rtbw",
        include_bytes!("../../../tables/syzygy/KBvKN.rtbw"),
    ),
    (
        "KBvKN.rtbz",
        include_bytes!("../../../tables/syzygy/KBvKN.rtbz"),
    ),
    (
        "KBvKP.rtbw",
        include_bytes!("../../../tables/syzygy/KBvKP.rtbw"),
    ),
    (
        "KBvKP.rtbz",
        include_bytes!("../../../tables/syzygy/KBvKP.rtbz"),
    ),
    (
        "KNvKN.rtbw",
        include_bytes!("../../../tables/syzygy/KNvKN.rtbw"),
    ),
    (
        "KNvKN.rtbz",
        include_bytes!("../../../tables/syzygy/KNvKN.rtbz"),
    ),
    (
        "KNvKP.rtbw",
        include_bytes!("../../../tables/syzygy/KNvKP.rtbw"),
    ),
    (
        "KNvKP.rtbz",
        include_bytes!("../../../tables/syzygy/KNvKP.rtbz"),
    ),
    (
        "KPvKP.rtbw",
        include_bytes!("../../../tables/syzygy/KPvKP.rtbw"),
    ),
    (
        "KPvKP.rtbz",
        include_bytes!("../../../tables/syzygy/KPvKP.rtbz"),
    ),
    (
        "KQvKB.rtbw",
        include_bytes!("../../../tables/syzygy/KQvKB.rtbw"),
    ),
    (
        "KQvKB.rtbz",
        include_bytes!("../../../tables/syzygy/KQvKB.rtbz"),
    ),
    (
        "KQvKN.rtbw",
        include_bytes!("../../../tables/syzygy/KQvKN.rtbw"),
    ),
    (
        "KQvKN.rtbz",
        include_bytes!("../../../tables/syzygy/KQvKN.rtbz"),
    ),
    (
        "KQvKP.rtbw",
        include_bytes!("../../../tables/syzygy/KQvKP.rtbw"),
    ),
    (
        "KQvKP.rtbz",
        include_bytes!("../../../tables/syzygy/KQvKP.rtbz"),
    ),
    (
        "KQvKQ.rtbw",
        include_bytes!("../../../tables/syzygy/KQvKQ.rtbw"),
    ),
    (
        "KQvKQ.rtbz",
        include_bytes!("../../../tables/syzygy/KQvKQ.rtbz"),
    ),
    (
        "KQvKR.rtbw",
        include_bytes!("../../../tables/syzygy/KQvKR.rtbw"),
    ),
    (
        "KQvKR.rtbz",
        include_bytes!("../../../tables/syzygy/KQvKR.rtbz"),
    ),
    (
        "KRvKB.rtbw",
        include_bytes!("../../../tables/syzygy/KRvKB.rtbw"),
    ),
    (
        "KRvKB.rtbz",
        include_bytes!("../../../tables/syzygy/KRvKB.rtbz"),
    ),
    (
        "KRvKN.rtbw",
        include_bytes!("../../../tables/syzygy/KRvKN.rtbw"),
    ),
    (
        "KRvKN.rtbz",
        include_bytes!("../../../tables/syzygy/KRvKN.rtbz"),
    ),
    (
        "KRvKP.rtbw",
        include_bytes!("../../../tables/syzygy/KRvKP.rtbw"),
    ),
    (
        "KRvKP.rtbz",
        include_bytes!("../../../tables/syzygy/KRvKP.rtbz"),
    ),
    (
        "KRvKR.rtbw",
        include_bytes!("../../../tables/syzygy/KRvKR.rtbw"),
    ),
    (
        "KRvKR.rtbz",
        include_bytes!("../../../tables/syzygy/KRvKR.rtbz"),
    ),
];

/// Extract the embedded Syzygy tables to a temporary directory.
///
/// Returns the path to the directory containing the extracted tables,
/// or `None` if extraction fails. Files are only written if they don't
/// already exist with the correct size (idempotent across runs).
pub fn extract_embedded_tables() -> Option<PathBuf> {
    let dir = std::env::temp_dir().join("rage-syzygy");
    std::fs::create_dir_all(&dir).ok()?;

    for &(name, data) in EMBEDDED_TABLES {
        let path = dir.join(name);
        // Only write if missing or wrong size (avoids redundant I/O on restart).
        let needs_write = match std::fs::metadata(&path) {
            Ok(meta) => meta.len() != data.len() as u64,
            Err(_) => true,
        };
        if needs_write {
            std::fs::write(&path, data).ok()?;
        }
    }

    Some(dir)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn embedded_table_count() {
        assert_eq!(EMBEDDED_TABLES.len(), 70);
    }

    #[test]
    fn all_tables_non_empty() {
        for &(name, data) in EMBEDDED_TABLES {
            assert!(!data.is_empty(), "embedded table {name} is empty");
        }
    }

    #[test]
    fn extract_tables_to_temp() {
        let dir = extract_embedded_tables().unwrap();
        assert!(dir.exists());
        // Verify all files were extracted
        for &(name, data) in EMBEDDED_TABLES {
            let path = dir.join(name);
            assert!(path.exists(), "missing extracted file: {name}");
            let on_disk = std::fs::read(&path).unwrap();
            assert_eq!(on_disk.len(), data.len(), "size mismatch for {name}");
        }
    }
}
