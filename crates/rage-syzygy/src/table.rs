//! Syzygy table file parsing (.rtbw for WDL, .rtbz for DTZ).
//!
//! Handles reading the file header, setting up subtable compression parameters,
//! and providing access to the raw table data.

use std::fs;
use std::path::Path;

use crate::compress::{PairsData, parse_pairs_data};
use crate::errors::{Result, SyzygyError};
use crate::material::MaterialKey;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Magic number for WDL tables (.rtbw).
const WDL_MAGIC: [u8; 4] = [0x71, 0xE8, 0x23, 0x5D];

/// Magic number for DTZ tables (.rtbz).
const DTZ_MAGIC: [u8; 4] = [0xD7, 0x66, 0x0C, 0xA5];

// ---------------------------------------------------------------------------
// Table types
// ---------------------------------------------------------------------------

/// Type of table file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableType {
    Wdl,
    Dtz,
}

/// A parsed Syzygy table file.
pub struct Table {
    /// The type of table (WDL or DTZ).
    pub table_type: TableType,
    /// Raw file data.
    pub data: Vec<u8>,
    /// Material key for this table.
    pub material: MaterialKey,
    /// Number of pieces (including kings).
    pub num_pieces: u32,
    /// Whether the table has pawns.
    pub has_pawns: bool,
    /// Header flags byte.
    pub flags: u8,
    /// Whether the table is split (has separate subtables for each side).
    pub split: bool,
    /// Compression parameters for each subtable.
    /// WDL: up to 4 subtables (Win, CursedWin/BlessedLoss, Draw, Loss) x 2 sides
    /// DTZ: 1-4 subtables depending on encoding
    pub pairs: Vec<PairsData>,
    /// Mapping from (side_to_move, wdl_value) to subtable index.
    /// For WDL: pairs[map[side][wdl+2]] gives the right subtable.
    pub num_subtables: usize,
}

impl Table {
    /// Load and parse a table file from disk.
    pub fn load(path: &Path, table_type: TableType) -> Result<Self> {
        let data = fs::read(path).map_err(SyzygyError::Io)?;
        Self::parse(data, table_type, path)
    }

    /// Parse table data (for testing without file I/O).
    pub fn parse(data: Vec<u8>, table_type: TableType, path: &Path) -> Result<Self> {
        if data.len() < 16 {
            return Err(SyzygyError::Corrupted(format!(
                "file too small: {} bytes",
                data.len()
            )));
        }

        // Verify magic number
        let expected_magic = match table_type {
            TableType::Wdl => &WDL_MAGIC,
            TableType::Dtz => &DTZ_MAGIC,
        };
        if &data[0..4] != expected_magic {
            return Err(SyzygyError::Corrupted(format!(
                "invalid magic number in {}",
                path.display()
            )));
        }

        // Parse header
        // Byte 4: number of pieces (both sides combined, including kings)
        let num_pieces = data[4] as u32;

        // Byte 5: flags
        // Bit 0: has pawns
        // Bit 1: split (separate subtables per side)
        // Bits 2-3: encoding type
        let flags = data[5];
        let has_pawns = flags & 1 != 0;
        let split = flags & 2 != 0;

        // Determine number of subtables
        let num_subtables = if table_type == TableType::Wdl {
            if split { 8 } else { 4 }
        } else {
            // DTZ: fewer subtables
            if split { 4 } else { 2 }
        };

        // Extract material key from filename
        let material = material_key_from_path(path)?;

        // Parse subtable compression parameters
        // The pairs data starts after the header (byte 6 onwards)
        let mut pairs = Vec::with_capacity(num_subtables);
        let mut offset = 6;

        // Read subtable sizes first (before pairs data)
        // Each subtable has a 1-byte flags field
        let mut sub_flags = Vec::with_capacity(num_subtables);
        for _ in 0..num_subtables {
            if offset >= data.len() {
                return Err(SyzygyError::Corrupted("truncated subtable flags".into()));
            }
            sub_flags.push(data[offset]);
            offset += 1;
        }

        // Pad to even boundary
        if offset & 1 != 0 {
            offset += 1;
        }

        // Now parse pairs data for each subtable
        // Note: in real Syzygy files the structure is more complex (subtable sizes
        // are stored separately), but this gives us the right framework.
        for &sf in &sub_flags {
            if offset >= data.len() {
                // Some subtables might not have data (e.g., empty subtables)
                pairs.push(PairsData::single(0));
                continue;
            }

            // Estimate number of values for this subtable
            // In practice this comes from the index calculation, but for initial
            // parsing we use a placeholder
            let num_values = estimate_subtable_size(num_pieces, has_pawns);

            match parse_pairs_data(&data[offset..], sf, num_values) {
                Ok((pd, consumed)) => {
                    let mut pd = pd;
                    pd.data_offset += offset; // adjust to absolute file offset
                    pairs.push(pd);
                    offset += consumed;
                }
                Err(_) => {
                    // If parsing fails, use a single-value fallback
                    pairs.push(PairsData::single(0));
                }
            }
        }

        Ok(Self {
            table_type,
            data,
            material,
            num_pieces,
            has_pawns,
            flags,
            split,
            pairs,
            num_subtables,
        })
    }
}

/// Estimate the number of values in a subtable based on piece count.
/// This is a rough estimate used during initial parsing; the exact value
/// comes from the index calculation.
fn estimate_subtable_size(num_pieces: u32, has_pawns: bool) -> u64 {
    if has_pawns {
        // Rough estimate for pawn tables
        match num_pieces {
            3 => 18_816,
            4 => 1_140_480,
            5 => 68_428_800,
            _ => 1_000_000,
        }
    } else {
        // Pawnless tables
        match num_pieces {
            3 => 10 * 63, // triangle(10) * 63 remaining squares
            4 => 10 * 63 * 62,
            5 => 10 * 63 * BINOM_62_2,
            _ => 1_000_000,
        }
    }
}

const BINOM_62_2: u64 = 1891;

/// Extract material key from a table filename.
/// Example: "KRvK.rtbw" -> MaterialKey for KR vs K
fn material_key_from_path(path: &Path) -> Result<MaterialKey> {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| SyzygyError::Corrupted("invalid filename".into()))?;

    parse_material_key(stem)
}

/// Parse a material key string like "KRvK" or "KQRvKR".
pub fn parse_material_key(name: &str) -> Result<MaterialKey> {
    let parts: Vec<&str> = name.split('v').collect();
    if parts.len() != 2 {
        return Err(SyzygyError::Corrupted(format!(
            "invalid table name: {name}"
        )));
    }

    let white_counts = parse_side_pieces(parts[0])?;
    let black_counts = parse_side_pieces(parts[1])?;

    Ok(MaterialKey {
        counts: [white_counts, black_counts],
    })
}

/// Parse piece characters for one side (e.g. "KR" -> [0,0,0,1,0]).
/// The leading 'K' is mandatory and skipped (kings are implicit).
fn parse_side_pieces(s: &str) -> Result<[u8; 5]> {
    let mut counts = [0u8; 5];
    let mut chars = s.chars();

    // First character must be 'K'
    match chars.next() {
        Some('K') => {}
        _ => {
            return Err(SyzygyError::Corrupted(format!(
                "side must start with K: {s}"
            )));
        }
    }

    for ch in chars {
        match ch {
            'P' => counts[0] += 1,
            'N' => counts[1] += 1,
            'B' => counts[2] += 1,
            'R' => counts[3] += 1,
            'Q' => counts[4] += 1,
            _ => {
                return Err(SyzygyError::Corrupted(format!(
                    "invalid piece character: {ch}"
                )));
            }
        }
    }

    Ok(counts)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn parse_material_key_krk() {
        let key = parse_material_key("KRvK").unwrap();
        assert_eq!(key.counts[0], [0, 0, 0, 1, 0]); // white: rook
        assert_eq!(key.counts[1], [0, 0, 0, 0, 0]); // black: nothing
        assert_eq!(key.table_name(), "KRvK");
    }

    #[test]
    fn parse_material_key_kqr_vs_kr() {
        let key = parse_material_key("KQRvKR").unwrap();
        assert_eq!(key.counts[0], [0, 0, 0, 1, 1]); // white: rook + queen
        assert_eq!(key.counts[1], [0, 0, 0, 1, 0]); // black: rook
    }

    #[test]
    fn parse_material_key_kp_vs_kp() {
        let key = parse_material_key("KPvKP").unwrap();
        assert_eq!(key.counts[0], [1, 0, 0, 0, 0]); // white: pawn
        assert_eq!(key.counts[1], [1, 0, 0, 0, 0]); // black: pawn
        assert!(key.has_pawns());
    }

    #[test]
    fn parse_material_key_invalid() {
        assert!(parse_material_key("KRK").is_err()); // no 'v'
        assert!(parse_material_key("RvK").is_err()); // no leading K
        assert!(parse_material_key("KXvK").is_err()); // invalid piece
    }

    #[test]
    fn wdl_magic_correct() {
        // Verify magic bytes match Syzygy spec
        assert_eq!(WDL_MAGIC, [0x71, 0xE8, 0x23, 0x5D]);
        assert_eq!(DTZ_MAGIC, [0xD7, 0x66, 0x0C, 0xA5]);
    }

    #[test]
    fn parse_side_pieces_basic() {
        assert_eq!(parse_side_pieces("K").unwrap(), [0, 0, 0, 0, 0]);
        assert_eq!(parse_side_pieces("KR").unwrap(), [0, 0, 0, 1, 0]);
        assert_eq!(parse_side_pieces("KQR").unwrap(), [0, 0, 0, 1, 1]);
        assert_eq!(parse_side_pieces("KBNP").unwrap(), [1, 1, 1, 0, 0]);
    }
}
