//! RE-PAIR + Huffman decompression for Syzygy tables.
//!
//! Syzygy tables use a two-stage compression:
//! 1. RE-PAIR: A grammar-based compression that replaces frequent symbol pairs
//!    with new symbols, building a binary tree of expansions.
//! 2. Huffman coding: Variable-length codes for the RE-PAIR symbols.
//!
//! To decompress a value at a given index:
//! 1. Use the index table to find which block contains the index
//! 2. Use the size table to find where the compressed block starts
//! 3. Read Huffman-coded symbols from the bit stream
//! 4. Recursively expand RE-PAIR symbols to get the literal value

use crate::errors::{Result, SyzygyError};

/// Parsed compression parameters for a subtable.
#[derive(Clone, Debug)]
pub struct PairsData {
    /// Index table: maps logical block index to sparse index offset.
    pub index_table: Vec<u64>,
    /// Size table: compressed size (in bytes) of each block.
    pub size_table: Vec<u16>,
    /// Span length for each block (number of values per block).
    pub block_size: u32,
    /// Number of blocks.
    pub num_blocks: u32,
    /// Symbol lengths (number of literal values each symbol expands to).
    pub sym_len: Vec<u8>,
    /// Symbol children: sym_pat[i] gives the left child, sym_pat[i] + sym_len offset
    /// gives the right child. Leaf symbols represent literal values.
    pub sym_pat: Vec<u16>,
    /// Minimum symbol length in the Huffman code.
    pub min_sym_len: u8,
    /// Huffman decode table: base code for each length.
    pub base: Vec<u64>,
    /// Offset into symbol list for each code length.
    pub offset: Vec<u16>,
    /// Number of symbols.
    pub num_syms: u32,
    /// Offset in the file data where compressed blocks begin.
    pub data_offset: usize,
    /// Number of values in this subtable.
    pub num_values: u64,
    /// Single-value optimization: when true, the table stores a single value.
    pub single_value: Option<u8>,
}

impl PairsData {
    /// Create a PairsData for a subtable that stores a single constant value.
    pub fn single(value: u8) -> Self {
        Self {
            index_table: Vec::new(),
            size_table: Vec::new(),
            block_size: 0,
            num_blocks: 0,
            sym_len: Vec::new(),
            sym_pat: Vec::new(),
            min_sym_len: 0,
            base: Vec::new(),
            offset: Vec::new(),
            num_syms: 0,
            data_offset: 0,
            num_values: 0,
            single_value: Some(value),
        }
    }
}

/// Bit reader for reading variable-length Huffman codes from a byte stream.
struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    /// Bit buffer (holds up to 64 bits).
    buf: u64,
    /// Number of valid bits in the buffer.
    buf_bits: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8], byte_offset: usize) -> Self {
        Self {
            data,
            byte_pos: byte_offset,
            buf: 0,
            buf_bits: 0,
        }
    }

    /// Ensure at least `n` bits are available in the buffer.
    fn fill(&mut self, n: u32) {
        while self.buf_bits < n {
            if self.byte_pos < self.data.len() {
                self.buf |= (self.data[self.byte_pos] as u64) << self.buf_bits;
                self.byte_pos += 1;
                self.buf_bits += 8;
            } else {
                break;
            }
        }
    }

    /// Peek at the next `n` bits without consuming them.
    fn peek(&mut self, n: u32) -> u64 {
        self.fill(n);
        self.buf & ((1u64 << n) - 1)
    }

    /// Skip `n` bits.
    fn skip(&mut self, n: u32) {
        self.fill(n);
        self.buf >>= n;
        self.buf_bits -= n;
    }

    /// Read the next `n` bits and advance.
    fn read(&mut self, n: u32) -> u64 {
        let val = self.peek(n);
        self.skip(n);
        val
    }
}

/// Decompress a single value from a Syzygy subtable.
///
/// `data` is the entire table file data.
/// `pairs` is the parsed compression parameters.
/// `index` is the logical index of the value to decompress.
pub fn decompress_value(data: &[u8], pairs: &PairsData, index: u64) -> Result<u8> {
    // Single-value optimization
    if let Some(val) = pairs.single_value {
        return Ok(val);
    }

    if pairs.block_size == 0 {
        return Err(SyzygyError::Corrupted("block_size is 0".into()));
    }

    // Find which block contains this index
    let block_idx = (index / pairs.block_size as u64) as usize;
    let idx_within_block = (index % pairs.block_size as u64) as usize;

    if block_idx >= pairs.index_table.len() {
        return Err(SyzygyError::Corrupted(format!(
            "block index {block_idx} out of range (max {})",
            pairs.index_table.len()
        )));
    }

    // Get the byte offset and bit offset for this block
    let sparse_idx = pairs.index_table[block_idx];
    let block_byte_offset = sparse_idx as usize;

    // Each index_table entry points to the start of the compressed block data
    // relative to data_offset
    let abs_offset = pairs.data_offset + block_byte_offset;
    if abs_offset >= data.len() {
        return Err(SyzygyError::Corrupted("block offset out of file".into()));
    }

    let mut reader = BitReader::new(data, abs_offset);

    // Skip to the desired value within the block by decoding symbols
    let mut values_remaining = idx_within_block;
    loop {
        let sym = decode_huffman_symbol(&mut reader, pairs)?;
        let sym_length = sym_expand_length(sym, pairs);

        if values_remaining < sym_length as usize {
            // The desired value is within this symbol's expansion
            return expand_symbol_at(sym, values_remaining, pairs);
        }
        values_remaining -= sym_length as usize;
    }
}

/// Decode one Huffman symbol from the bit stream.
fn decode_huffman_symbol(reader: &mut BitReader<'_>, pairs: &PairsData) -> Result<u32> {
    // Read bits one at a time, compare against base codes
    let mut code = 0u64;
    let mut len = pairs.min_sym_len as usize;

    reader.fill(32); // prefill buffer

    loop {
        if len >= pairs.base.len() + pairs.min_sym_len as usize {
            return Err(SyzygyError::Corrupted("Huffman code too long".into()));
        }

        let bit = reader.read(1);
        code = (code << 1) | bit;

        let base_idx = len - pairs.min_sym_len as usize;
        if base_idx < pairs.base.len() && code <= pairs.base[base_idx] {
            let offset_idx = base_idx;
            if offset_idx < pairs.offset.len() {
                let sym = (code as u32)
                    .wrapping_sub(pairs.base[base_idx] as u32)
                    .wrapping_add(pairs.offset[offset_idx] as u32);
                if sym < pairs.num_syms {
                    return Ok(sym);
                }
            }
        }

        len += 1;
        if len > 64 {
            return Err(SyzygyError::Corrupted("Huffman decode failed".into()));
        }
    }
}

/// Get the number of literal values a symbol expands to.
fn sym_expand_length(sym: u32, pairs: &PairsData) -> u32 {
    if (sym as usize) < pairs.sym_len.len() {
        pairs.sym_len[sym as usize] as u32 + 1
    } else {
        1
    }
}

/// Expand a RE-PAIR symbol and return the value at `offset` within the expansion.
fn expand_symbol_at(sym: u32, offset: usize, pairs: &PairsData) -> Result<u8> {
    // If this is a leaf (sym_len == 0), return the literal value
    if (sym as usize) < pairs.sym_len.len() && pairs.sym_len[sym as usize] == 0 {
        if offset != 0 {
            return Err(SyzygyError::Corrupted("offset into leaf symbol".into()));
        }
        // The literal value is the symbol index itself (mod number of literals)
        return Ok(sym as u8);
    }

    // Non-leaf: recurse into children
    if (sym as usize) >= pairs.sym_pat.len() {
        return Err(SyzygyError::Corrupted("symbol out of range".into()));
    }

    let left = pairs.sym_pat[sym as usize] as u32;
    let left_len = sym_expand_length(left, pairs) as usize;

    if offset < left_len {
        expand_symbol_at(left, offset, pairs)
    } else {
        // Right child is stored as left + 1 in the symbol table
        let right = left + 1;
        expand_symbol_at(right, offset - left_len, pairs)
    }
}

/// Parse compression parameters from the table file header.
///
/// `data` is the file data starting from the pairs-data section.
/// `flags` indicates the encoding type (from the table header).
/// `num_values` is the total number of values in this subtable.
///
/// Returns the PairsData and the number of bytes consumed from `data`.
pub fn parse_pairs_data(data: &[u8], flags: u8, num_values: u64) -> Result<(PairsData, usize)> {
    // Flag bit 7: single-value table
    // Structure: byte 0 = flags (with 0x80 bit set), byte 1 = value
    if flags & 0x80 != 0 {
        if data.len() < 2 {
            return Err(SyzygyError::Corrupted("single-value table too short".into()));
        }
        // Value is at byte 1
        return Ok((PairsData::single(data[1]), 2));
    }

    if data.len() < 10 {
        eprintln!("debug: pairs data too short: {} bytes", data.len());
        return Err(SyzygyError::Corrupted(format!(
            "pairs data too short: {} < 10",
            data.len()
        )));
    }

    let mut pos = 0;

    // Byte 0: block_size_shift (log2 of block size)
    let block_size_shift = data[pos];
    pos += 1;

    // Byte 1: min_sym_len
    let min_sym_len = data[pos];
    pos += 1;

    // Bytes 2-3: number of Huffman code lengths
    let num_lengths = read_u16_le(data, pos) as usize;
    pos += 2;

    let block_size = 1u32 << block_size_shift;

    if block_size_shift > 30 {
        return Err(SyzygyError::Corrupted(format!(
            "block_size_shift too large: {}",
            block_size_shift
        )));
    }

    // Read Huffman code length distribution
    if pos + num_lengths > data.len() {
        return Err(SyzygyError::Corrupted(
            "code length distribution truncated".into(),
        ));
    }

    // Build base and offset tables for Huffman decoding
    let mut base = Vec::with_capacity(num_lengths);
    let mut offset = Vec::with_capacity(num_lengths);
    let mut num_syms = 0u32;

    for i in 0..num_lengths {
        if pos + i >= data.len() {
            return Err(SyzygyError::Corrupted("code length distribution truncated".into()));
        }
        let count = data[pos + i] as u32;
        offset.push(num_syms as u16);
        num_syms += count;
        // Base code for this length
        if i == 0 {
            base.push((count as u64).wrapping_sub(1));
        } else {
            if base.is_empty() {
                return Err(SyzygyError::Corrupted("huffman base table underflow".into()));
            }
            base.push(((base[i - 1] + 1) * 2 + count as u64).wrapping_sub(1));
        }
    }
    pos += num_lengths;

    // Pad to even byte boundary
    if pos & 1 != 0 {
        pos += 1;
    }

    // Read symbol patterns (RE-PAIR tree)
    let sym_pat_size = num_syms as usize * 3; // 3 bytes per symbol entry
    if pos + sym_pat_size > data.len() {
        return Err(SyzygyError::Corrupted("symbol patterns truncated".into()));
    }

    let mut sym_pat = Vec::with_capacity(num_syms as usize);
    let mut sym_len = Vec::with_capacity(num_syms as usize);
    for i in 0..num_syms as usize {
        let base_pos = pos + i * 3;
        let left = read_u16_le(data, base_pos);
        let len_byte = data[base_pos + 2];
        sym_pat.push(left);
        sym_len.push(len_byte);
    }
    pos += sym_pat_size;

    // Pad to 8-byte boundary
    pos = (pos + 7) & !7;

    // Index table
    // If num_values is an estimate and doesn't fit, try to infer from available space.
    // Available bytes from current position: data.len() - pos
    // For N blocks, we need: 6*(N+1) + 2*N = 8N + 6 bytes (index + size tables)
    let mut num_blocks = if block_size > 0 {
        num_values.div_ceil(block_size as u64) as u32
    } else {
        0
    };

    // If this doesn't fit, reduce num_blocks to fit available space
    let available_bytes = (data.len() - pos) as u32;
    let max_blocks_for_space = if available_bytes > 6 {
        // 6*(N+1) + 2*N <= available => 8N + 6 <= available => N <= (available-6)/8
        (available_bytes - 6) / 8
    } else {
        0
    };

    let actual_num_values = if num_blocks > max_blocks_for_space {
        eprintln!(
            "Reducing num_blocks from {} to {} to fit available {} bytes",
            num_blocks, max_blocks_for_space, available_bytes
        );
        num_blocks = max_blocks_for_space;
        // Recalculate num_values to match the blocks we can actually read
        if num_blocks > 0 && block_size > 0 {
            let adjusted = (num_blocks as u64) * (block_size as u64);
            eprintln!("Adjusted num_values from {} to {}", num_values, adjusted);
            adjusted
        } else {
            num_values
        }
    } else {
        num_values
    };

    let index_entries = (num_blocks as usize + 1).max(1);
    let index_size = index_entries * 6; // 6 bytes per entry (48-bit)
    if pos + index_size > data.len() {
        return Err(SyzygyError::Corrupted("index table truncated".into()));
    }

    let mut index_table = Vec::with_capacity(index_entries);
    for i in 0..index_entries {
        let off = pos + i * 6;
        let val = read_u48_le(data, off);
        index_table.push(val);
    }
    pos += index_size;

    // Size table
    let size_entries = num_blocks as usize;
    let size_table_size = size_entries * 2;
    if pos + size_table_size > data.len() {
        return Err(SyzygyError::Corrupted("size table truncated".into()));
    }

    let mut size_table = Vec::with_capacity(size_entries);
    for i in 0..size_entries {
        size_table.push(read_u16_le(data, pos + i * 2));
    }
    pos += size_table_size;

    // Data starts at current position (compressed blocks)
    let data_offset = pos;

    Ok((
        PairsData {
            index_table,
            size_table,
            block_size,
            num_blocks,
            sym_len,
            sym_pat,
            min_sym_len,
            base,
            offset,
            num_syms,
            data_offset,
            num_values: actual_num_values,
            single_value: None,
        },
        pos,
    ))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn read_u16_le(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

fn read_u48_le(data: &[u8], offset: usize) -> u64 {
    let mut bytes = [0u8; 8];
    bytes[..6].copy_from_slice(&data[offset..offset + 6]);
    u64::from_le_bytes(bytes)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn single_value_table() {
        let pairs = PairsData::single(2);
        assert_eq!(pairs.single_value, Some(2));
        // Any index returns the single value
        let result = decompress_value(&[], &pairs, 0).unwrap();
        assert_eq!(result, 2);
        let result = decompress_value(&[], &pairs, 42).unwrap();
        assert_eq!(result, 2);
    }

    #[test]
    fn bit_reader_basic() {
        let data = [0b10110100u8, 0b01101001u8];
        let mut reader = BitReader::new(&data, 0);

        // Read 4 bits: should get 0b0100 = 4 (LSB first)
        let v = reader.read(4);
        assert_eq!(v, 0b0100);

        // Read 4 more: should get 0b1011 = 11
        let v = reader.read(4);
        assert_eq!(v, 0b1011);
    }

    #[test]
    fn read_u16_le_basic() {
        let data = [0x34, 0x12];
        assert_eq!(read_u16_le(&data, 0), 0x1234);
    }

    #[test]
    fn read_u48_le_basic() {
        let data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let val = read_u48_le(&data, 0);
        assert_eq!(val, 0x0006_0504_0302_01);
    }
}
