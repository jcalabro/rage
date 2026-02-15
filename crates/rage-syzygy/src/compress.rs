//! RE-PAIR + Huffman decompression for Syzygy tables.
//!
//! Syzygy tables use a two-stage compression:
//! 1. RE-PAIR: A grammar-based compression that replaces frequent symbol pairs
//!    with new symbols, building a binary tree of expansions.
//! 2. Huffman coding: Variable-length codes for the RE-PAIR symbols.
//!
//! Reference: shakmaty-syzygy by niklasf.

use crate::errors::{Result, SyzygyError};
use crate::indexing::GroupData;

// ---------------------------------------------------------------------------
// Symbol tree
// ---------------------------------------------------------------------------

/// A node in the RE-PAIR symbol tree. Each symbol is 3 bytes packed.
#[derive(Clone, Debug)]
pub struct Symbol {
    lr: [u8; 3],
    pub len: u8,
}

impl Symbol {
    fn new() -> Symbol {
        Symbol { lr: [0; 3], len: 0 }
    }

    /// Left child index.
    pub fn left(&self) -> u16 {
        (u16::from(self.lr[1] & 0xf) << 8) | u16::from(self.lr[0])
    }

    /// Right child index.
    pub fn right(&self) -> u16 {
        (u16::from(self.lr[2]) << 4) | (u16::from(self.lr[1]) >> 4)
    }
}

// ---------------------------------------------------------------------------
// PairsData
// ---------------------------------------------------------------------------

/// Parsed compression parameters for a subtable.
#[derive(Clone, Debug)]
pub struct PairsData {
    /// Encoding flags.
    pub flags: u8,
    /// Piece configuration encoding info.
    pub groups: Option<GroupData>,

    /// Block size in bytes (1 << header[1]).
    pub block_size: u32,
    /// Span for sparse index (1 << header[2]).
    pub span: u32,
    /// Number of blocks in the table.
    pub blocks_num: u32,

    /// Minimum Huffman symbol length in bits.
    pub min_symlen: u8,
    /// Offset in file of the lowest_sym array (u16 LE entries).
    pub lowest_sym: usize,
    /// 64-bit padded base values for Huffman decoding.
    pub base: Vec<u64>,
    /// The symbol table (RE-PAIR tree nodes).
    pub symbols: Vec<Symbol>,

    /// Offset of the sparse index in the file.
    pub sparse_index: usize,
    /// Number of sparse index entries.
    pub sparse_index_size: u32,

    /// Offset of the block length table in the file.
    pub block_lengths: usize,
    /// Number of block length entries (blocks_num + padding).
    pub block_length_size: u32,

    /// Start of compressed data in the file.
    pub data: usize,

    /// For single-value tables: the stored value.
    pub single_value: Option<u8>,
}

impl PairsData {
    /// Create a PairsData for a single-value subtable.
    pub fn single(value: u8) -> Self {
        Self {
            flags: 0x80,
            groups: None,
            block_size: 0,
            span: 0,
            blocks_num: 0,
            min_symlen: value,
            lowest_sym: 0,
            base: Vec::new(),
            symbols: Vec::new(),
            sparse_index: 0,
            sparse_index_size: 0,
            block_lengths: 0,
            block_length_size: 0,
            data: 0,
            single_value: Some(value),
        }
    }

    /// Total table size from groups.
    pub fn tb_size(&self) -> u64 {
        match &self.groups {
            Some(g) => g.tb_size(),
            None => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Build the symbol table by reading the binary tree from file data.
fn read_symbols(
    data: &[u8],
    btree_offset: usize,
    symbols: &mut [Symbol],
    visited: &mut [bool],
    sym: u16,
    depth: u8,
) -> Result<()> {
    let idx = sym as usize;
    if idx >= symbols.len() || visited[idx] {
        return Ok(());
    }
    if depth == 0 {
        return Err(SyzygyError::Corrupted("symbol tree too deep".into()));
    }

    let off = btree_offset + idx * 3;
    if off + 3 > data.len() {
        return Err(SyzygyError::Corrupted("symbol tree truncated".into()));
    }

    let mut symbol = Symbol::new();
    symbol.lr[0] = data[off];
    symbol.lr[1] = data[off + 1];
    symbol.lr[2] = data[off + 2];

    if symbol.right() == 0xfff {
        // Leaf node.
        symbol.len = 0;
    } else {
        read_symbols(
            data,
            btree_offset,
            symbols,
            visited,
            symbol.left(),
            depth - 1,
        )?;
        read_symbols(
            data,
            btree_offset,
            symbols,
            visited,
            symbol.right(),
            depth - 1,
        )?;

        let left_len = symbols[symbol.left() as usize].len;
        let right_len = symbols[symbol.right() as usize].len;
        symbol.len = left_len
            .checked_add(right_len)
            .and_then(|s| s.checked_add(1))
            .ok_or_else(|| SyzygyError::Corrupted("symbol length overflow".into()))?;
    }

    symbols[idx] = symbol;
    visited[idx] = true;
    Ok(())
}

/// Parse a pairs data header from file data at the given offset.
///
/// Returns `(PairsData, next_offset)`. The PairsData will have
/// `sparse_index`, `block_lengths`, and `data` set to 0 — these are
/// filled in later during the table-level offset assignment phase.
pub fn parse_pairs_data(data: &[u8], ptr: usize, groups: GroupData) -> Result<(PairsData, usize)> {
    if ptr >= data.len() {
        return Err(SyzygyError::Corrupted(
            "pairs data offset out of range".into(),
        ));
    }

    let flags = data[ptr];

    // Single-value table.
    if flags & 0x80 != 0 {
        if ptr + 1 >= data.len() {
            return Err(SyzygyError::Corrupted(
                "single-value table too short".into(),
            ));
        }
        let value = data[ptr + 1];
        return Ok((PairsData::single(value), ptr + 2));
    }

    // Full header: 10 bytes.
    if ptr + 10 > data.len() {
        return Err(SyzygyError::Corrupted(format!(
            "pairs header too short at offset {ptr}"
        )));
    }

    let tb_size = groups.tb_size();
    let block_size = 1u32
        .checked_shl(u32::from(data[ptr + 1]))
        .ok_or_else(|| SyzygyError::Corrupted("block_size overflow".into()))?;
    let span = 1u32
        .checked_shl(u32::from(data[ptr + 2]))
        .ok_or_else(|| SyzygyError::Corrupted("span overflow".into()))?;
    let sparse_index_size = tb_size.div_ceil(u64::from(span)) as u32;
    let padding = data[ptr + 3];
    let blocks_num =
        u32::from_le_bytes([data[ptr + 4], data[ptr + 5], data[ptr + 6], data[ptr + 7]]);
    let block_length_size = blocks_num
        .checked_add(u32::from(padding))
        .ok_or_else(|| SyzygyError::Corrupted("block_length_size overflow".into()))?;

    let max_symlen = data[ptr + 8];
    let min_symlen = data[ptr + 9];
    if max_symlen > 32 || min_symlen > 32 || max_symlen < min_symlen {
        return Err(SyzygyError::Corrupted(format!(
            "invalid symlen: min={min_symlen}, max={max_symlen}"
        )));
    }

    let h = (max_symlen - min_symlen + 1) as usize;
    let lowest_sym = ptr + 10;

    // Build base array from the lowest_sym entries.
    if lowest_sym + h * 2 > data.len() {
        return Err(SyzygyError::Corrupted("lowest_sym table truncated".into()));
    }

    let mut base = vec![0u64; h];
    // Compute base values (working backwards).
    for i in (0..h.saturating_sub(1)).rev() {
        let off = lowest_sym + i * 2;
        let this_val = u64::from(u16::from_le_bytes([data[off], data[off + 1]]));
        let next_val = u64::from(u16::from_le_bytes([data[off + 2], data[off + 3]]));
        base[i] = (base[i + 1] + this_val - next_val) / 2;
    }
    // Left-shift each base to fill 64 bits.
    for (i, b) in base.iter_mut().enumerate() {
        let shift = 64u32.saturating_sub(u32::from(min_symlen) + i as u32);
        *b = b.checked_shl(shift).unwrap_or(0);
    }

    // Read symbol count and build symbol table.
    let sym_ptr = lowest_sym + h * 2;
    if sym_ptr + 2 > data.len() {
        return Err(SyzygyError::Corrupted("symbol count truncated".into()));
    }
    let num_syms = u16::from_le_bytes([data[sym_ptr], data[sym_ptr + 1]]) as usize;
    let btree = sym_ptr + 2;

    if btree + num_syms * 3 > data.len() {
        return Err(SyzygyError::Corrupted("symbol tree truncated".into()));
    }

    let mut symbols = vec![Symbol::new(); num_syms];
    let mut visited = vec![false; num_syms];
    for s in 0..num_syms as u16 {
        read_symbols(data, btree, &mut symbols, &mut visited, s, 16)?;
    }

    let next = btree + num_syms * 3 + (num_syms & 1);

    Ok((
        PairsData {
            flags,
            groups: Some(groups),
            block_size,
            span,
            blocks_num,
            min_symlen,
            lowest_sym,
            base,
            symbols,
            sparse_index: 0,
            sparse_index_size,
            block_lengths: 0,
            block_length_size,
            data: 0,
            single_value: None,
        },
        next,
    ))
}

// ---------------------------------------------------------------------------
// Decompression
// ---------------------------------------------------------------------------

/// Decompress a single value from a Syzygy subtable.
pub fn decompress_value(data: &[u8], d: &PairsData, idx: u64) -> Result<u8> {
    // Single-value optimization.
    if d.single_value.is_some() {
        return Ok(d.min_symlen);
    }

    // Sparse index lookup: find the approximate block.
    let main_idx = (idx / u64::from(d.span)) as usize;
    let si_off = d.sparse_index + main_idx * 6;
    if si_off + 6 > data.len() {
        return Err(SyzygyError::Corrupted("sparse index out of range".into()));
    }
    let block = u32::from_le_bytes([
        data[si_off],
        data[si_off + 1],
        data[si_off + 2],
        data[si_off + 3],
    ]);
    let offset = i64::from(u16::from_le_bytes([data[si_off + 4], data[si_off + 5]]));

    let mut lit_idx = idx as i64 % i64::from(d.span) - i64::from(d.span) / 2 + offset;
    let mut block = block;

    // Walk block lengths to find the exact block.
    if lit_idx < 0 {
        loop {
            if block == 0 {
                return Err(SyzygyError::Corrupted("block underflow".into()));
            }
            block -= 1;
            let bl_off = d.block_lengths + block as usize * 2;
            if bl_off + 2 > data.len() {
                return Err(SyzygyError::Corrupted("block length out of range".into()));
            }
            lit_idx += i64::from(u16::from_le_bytes([data[bl_off], data[bl_off + 1]])) + 1;
            if lit_idx >= 0 {
                break;
            }
        }
    } else {
        loop {
            let bl_off = d.block_lengths + block as usize * 2;
            if bl_off + 2 > data.len() {
                return Err(SyzygyError::Corrupted("block length out of range".into()));
            }
            let block_length = i64::from(u16::from_le_bytes([data[bl_off], data[bl_off + 1]])) + 1;
            if lit_idx < block_length {
                break;
            }
            lit_idx -= block_length;
            block += 1;
        }
    }

    // Read the compressed block into a buffer (block_size + 4 bytes for refill).
    let block_data_start = d.data + block as usize * d.block_size as usize;
    let block_data_end = block_data_start + d.block_size as usize + 4;
    if block_data_end > data.len() {
        return Err(SyzygyError::Corrupted("block data out of range".into()));
    }
    let block_data = &data[block_data_start..block_data_end];

    // Big-endian Huffman decoding.
    let mut buf = u64::from_be_bytes([
        block_data[0],
        block_data[1],
        block_data[2],
        block_data[3],
        block_data[4],
        block_data[5],
        block_data[6],
        block_data[7],
    ]);
    let mut buf_size = 64usize;
    let mut cursor = 8usize; // next byte to read from block_data

    let mut sym;

    loop {
        // Find the Huffman symbol.
        let mut len = 0;
        while len < d.base.len() && buf < d.base[len] {
            len += 1;
        }
        if len >= d.base.len() {
            return Err(SyzygyError::Corrupted("Huffman decode failed".into()));
        }

        sym = ((buf - d.base[len]) >> (64 - len - usize::from(d.min_symlen))) as u16;
        // Add the lowest_sym offset for this code length.
        let ls_off = d.lowest_sym + len * 2;
        if ls_off + 2 > data.len() {
            return Err(SyzygyError::Corrupted(
                "lowest_sym read out of range".into(),
            ));
        }
        sym += u16::from_le_bytes([data[ls_off], data[ls_off + 1]]);

        if sym as usize >= d.symbols.len() {
            return Err(SyzygyError::Corrupted("symbol index out of range".into()));
        }

        if lit_idx < i64::from(d.symbols[sym as usize].len) + 1 {
            break;
        }

        lit_idx -= i64::from(d.symbols[sym as usize].len) + 1;
        let consumed = usize::from(d.min_symlen) + len;
        buf <<= consumed;
        buf_size -= consumed;

        // Refill buffer.
        if buf_size <= 32 && cursor + 4 <= block_data.len() {
            let refill = u64::from(u32::from_be_bytes([
                block_data[cursor],
                block_data[cursor + 1],
                block_data[cursor + 2],
                block_data[cursor + 3],
            ]));
            buf |= refill << (64 - buf_size - 32);
            buf_size += 32;
            cursor += 4;
        }
    }

    // Expand the RE-PAIR symbol to find the literal at lit_idx.
    let mut symbol = &d.symbols[sym as usize];
    loop {
        if symbol.len == 0 {
            return Ok(symbol.left() as u8);
        }

        let left = &d.symbols[symbol.left() as usize];
        if lit_idx < i64::from(left.len) + 1 {
            symbol = left;
        } else {
            lit_idx -= i64::from(left.len) + 1;
            symbol = &d.symbols[symbol.right() as usize];
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn single_value_table() {
        let pairs = PairsData::single(2);
        assert_eq!(pairs.single_value, Some(2));
        let result = decompress_value(&[], &pairs, 0).unwrap();
        assert_eq!(result, 2);
        let result = decompress_value(&[], &pairs, 42).unwrap();
        assert_eq!(result, 2);
    }

    #[test]
    fn symbol_left_right() {
        let s = Symbol {
            lr: [0xAB, 0xCD, 0xEF],
            len: 5,
        };
        // left = (0xD & 0xf) << 8 | 0xAB = 0xDAB
        assert_eq!(s.left(), 0xDAB);
        // right = 0xEF << 4 | 0xC >> 4 = 0xEFC
        assert_eq!(s.right(), 0xEFC);
    }
}
