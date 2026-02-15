//! Syzygy table file parsing (.rtbw for WDL, .rtbz for DTZ).
//!
//! Handles reading the file header, setting up subtable compression parameters,
//! and providing access to the raw table data.
//!
//! Reference: shakmaty-syzygy by niklasf.

use std::fs;
use std::path::Path;

use crate::compress::{PairsData, parse_pairs_data};
use crate::errors::{Result, SyzygyError};
use crate::indexing::{GroupData, TablePiece, piece_from_nibble};
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

/// Encoding and compression for one side of a subtable.
#[derive(Clone, Debug)]
pub struct SideData {
    pub pairs: PairsData,
}

/// Encoding and compression for both sides of a file subtable.
#[derive(Clone, Debug)]
pub struct FileData {
    pub sides: Vec<SideData>,
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
    /// File subtables (1 for pawnless, 4 for pawn tables).
    pub files: Vec<FileData>,
    /// Number of unique piece roles (from material key).
    pub num_unique_pieces: usize,
    /// Minimum count among groups of 2+ identical pieces.
    pub min_like_man: usize,
}

impl Table {
    /// Load and parse a table file from disk.
    pub fn load(path: &Path, table_type: TableType) -> Result<Self> {
        let data = fs::read(path).map_err(SyzygyError::Io)?;
        Self::parse(data, table_type, path)
    }

    /// Parse table data following the Syzygy format specification.
    ///
    /// Layout:
    /// 1. Magic (4 bytes) + flags (1 byte)
    /// 2. Group data: order bytes + piece lists for each file/side
    /// 3. Align to even boundary
    /// 4. Pairs headers (sequentially for all subtables)
    /// 5. Sparse indices (grouped for all subtables)
    /// 6. Block lengths (grouped for all subtables)
    /// 7. Compressed data (64-byte aligned, grouped for all subtables)
    pub fn parse(data: Vec<u8>, table_type: TableType, path: &Path) -> Result<Self> {
        if data.len() < 16 {
            return Err(SyzygyError::Corrupted(format!(
                "file too small: {} bytes",
                data.len()
            )));
        }

        // 1. Verify magic number.
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

        // Extract material key from filename.
        let material = material_key_from_path(path)?;
        let num_pieces = material.total_pieces();
        let num_unique_pieces = material.unique_pieces();
        let min_like_man = material.min_like_man();

        // Read flags.
        let flags_byte = data[4];
        let split = flags_byte & 0x01 != 0;
        let has_pawns = flags_byte & 0x02 != 0;

        let pp = material.white_pawn_count() > 0 && material.black_pawn_count() > 0;
        let num_files = if has_pawns { 4usize } else { 1 };
        // For WDL: 2 sides if not symmetric, 1 if symmetric.
        // For DTZ: always 1 side.
        let num_sides = if table_type == TableType::Wdl && split {
            2usize
        } else {
            1
        };

        // 2. Parse group data: order bytes and piece lists.
        let mut ptr = 5usize;

        // For each file, read order bytes and piece list, build GroupData.
        let mut file_groups: Vec<Vec<GroupData>> = Vec::with_capacity(num_files);

        for file in 0..num_files {
            if ptr + 1 + usize::from(pp) > data.len() {
                return Err(SyzygyError::Corrupted("order bytes truncated".into()));
            }

            // Read order bytes.
            let order_0 = [data[ptr] & 0xf, if pp { data[ptr + 1] & 0xf } else { 0xf }];
            let order_1 = [data[ptr] >> 4, if pp { data[ptr + 1] >> 4 } else { 0xf }];

            ptr += 1 + usize::from(pp);

            // Read piece list for each side.
            let mut side_groups = Vec::with_capacity(num_sides);
            for side in 0..num_sides {
                if ptr + num_pieces as usize > data.len() {
                    return Err(SyzygyError::Corrupted("piece list truncated".into()));
                }

                // Parse pieces from the header bytes.
                let pieces = parse_piece_list(&data, ptr, num_pieces as usize, side)?;

                let order = if side == 0 { order_0 } else { order_1 };
                let groups = GroupData::new(pieces, order, &material).ok_or_else(|| {
                    SyzygyError::Corrupted(format!(
                        "failed to compute GroupData for file {file} side {side}"
                    ))
                })?;
                side_groups.push(groups);
            }

            // Only the first side's piece list advances the pointer.
            // (Both sides share the same piece bytes, just interpreted differently.)
            if file == 0 {
                ptr += num_pieces as usize;
            }

            file_groups.push(side_groups);
        }

        // 3. Align to even boundary.
        ptr += ptr & 1;

        // 4. Parse pairs headers sequentially for all subtables.
        let mut files: Vec<FileData> = Vec::with_capacity(num_files);
        for file_group in file_groups.iter().take(num_files) {
            let mut sides = Vec::with_capacity(num_sides);
            for side_group in file_group.iter().take(num_sides) {
                let groups = side_group.clone();
                let (pairs, next_ptr) = parse_pairs_data(&data, ptr, groups)?;
                sides.push(SideData { pairs });
                ptr = next_ptr;
            }
            files.push(FileData { sides });
        }

        // 5. Assign sparse index offsets (grouped for all subtables).
        for file in &mut files {
            for side in &mut file.sides {
                side.pairs.sparse_index = ptr;
                ptr += side.pairs.sparse_index_size as usize * 6;
            }
        }

        // 6. Assign block length offsets (grouped for all subtables).
        for file in &mut files {
            for side in &mut file.sides {
                side.pairs.block_lengths = ptr;
                ptr += side.pairs.block_length_size as usize * 2;
            }
        }

        // 7. Assign compressed data offsets (64-byte aligned).
        for file in &mut files {
            for side in &mut file.sides {
                ptr = (ptr + 0x3f) & !0x3f; // 64-byte alignment
                side.pairs.data = ptr;
                ptr += side.pairs.blocks_num as usize * side.pairs.block_size as usize;
            }
        }

        Ok(Self {
            table_type,
            data,
            material,
            num_pieces,
            has_pawns,
            flags: flags_byte,
            split,
            files,
            num_unique_pieces,
            min_like_man,
        })
    }
}

/// Parse a piece list from raw header bytes.
///
/// Each byte encodes two pieces: low nibble for side 0, high nibble for side 1.
/// This matches how shakmaty-syzygy reads pieces with `side.fold_wb(low, high)`.
fn parse_piece_list(
    data: &[u8],
    offset: usize,
    count: usize,
    side: usize,
) -> Result<Vec<TablePiece>> {
    let mut pieces = Vec::with_capacity(count);
    for i in 0..count {
        let byte = data[offset + i];
        let nibble = if side == 0 { byte & 0xf } else { byte >> 4 };
        let piece = piece_from_nibble(nibble)
            .ok_or_else(|| SyzygyError::Corrupted(format!("invalid piece nibble: {nibble:#x}")))?;
        pieces.push(piece);
    }
    Ok(pieces)
}

/// Extract material key from a table filename.
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
fn parse_side_pieces(s: &str) -> Result<[u8; 5]> {
    let mut counts = [0u8; 5];
    let mut chars = s.chars();

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
        assert_eq!(key.counts[0], [0, 0, 0, 1, 0]);
        assert_eq!(key.counts[1], [0, 0, 0, 0, 0]);
        assert_eq!(key.table_name(), "KRvK");
    }

    #[test]
    fn parse_material_key_kqr_vs_kr() {
        let key = parse_material_key("KQRvKR").unwrap();
        assert_eq!(key.counts[0], [0, 0, 0, 1, 1]);
        assert_eq!(key.counts[1], [0, 0, 0, 1, 0]);
    }

    #[test]
    fn parse_material_key_kp_vs_kp() {
        let key = parse_material_key("KPvKP").unwrap();
        assert_eq!(key.counts[0], [1, 0, 0, 0, 0]);
        assert_eq!(key.counts[1], [1, 0, 0, 0, 0]);
        assert!(key.has_pawns());
    }

    #[test]
    fn parse_material_key_invalid() {
        assert!(parse_material_key("KRK").is_err());
        assert!(parse_material_key("RvK").is_err());
        assert!(parse_material_key("KXvK").is_err());
    }

    #[test]
    fn wdl_magic_correct() {
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
