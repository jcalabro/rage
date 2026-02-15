//! Syzygy table file parsing (.rtbw for WDL, .rtbz for DTZ).
//!
//! Handles reading the file header, setting up subtable compression parameters,
//! and providing access to the raw table data.

use std::fs;
use std::path::Path;

use crate::compress::PairsData;
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

        // Extract material key from filename (needed to know piece count)
        let material = material_key_from_path(path)?;

        // Parse header (Syzygy WDL format)
        // Byte 4: flags
        // Bit 0 (0x01): split flag (WDL never split, but DTZ can be)
        // Bit 1 (0x02): files flag (4 files if set, 1 if not) - only for pawn tables
        let flags_byte_4 = data[4];
        let split = flags_byte_4 & 0x01 != 0;
        let has_pawns = flags_byte_4 & 0x02 != 0;

        // For WDL, pawnless tables: always 1 subtable (no split)
        // For WDL, pawn tables: 4 subtables (one per file)
        let num_subtables = if has_pawns { 4 } else { 1 };

        // Count of pieces from material key
        let num_pieces = material.total_pieces();

        // Count of pieces excluding kings (for future use when parsing complex headers)
        let _num_non_king_pieces = (num_pieces as usize).saturating_sub(2).max(0);

        // Parse subtable compression parameters
        let mut pairs = Vec::with_capacity(num_subtables);

        if !has_pawns {
            // Pawnless tables: piece data at byte 5, pairs data after alignment
            // Following Fathom reference: data += be->num + 1; data += (uintptr_t)data & 1;
            // be->num = num_pieces - 2

            // TODO: Implement actual pairs data parsing
            // Following Fathom reference, pairs data is at:
            // offset = 5 + 1 + num_pieces - 2 = 4 + num_pieces
            // Then aligned to even boundary: offset += offset & 1
            //
            // The embedded test tables use a non-standard format that doesn't match
            // the official Syzygy spec. Real tables from fathom-tb.org follow this format.
            // For now, stub with Draw.

            pairs.push(PairsData::single(2));
        } else {
            // Pawn tables: 4 files, each with encoding + pairs data
            // TODO: implement pawn table parsing
            for _ in 0..num_subtables {
                pairs.push(PairsData::single(0));
            }
        }

        Ok(Self {
            table_type,
            data,
            material,
            num_pieces,
            has_pawns,
            flags: flags_byte_4,
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
