use std::fs;
use std::path::Path;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use rage_core::{Color, PieceType, Square};
use rage_nnue::features::feature_index;
use rand::Rng;
use rand::seq::SliceRandom;

use crate::data::{ENTRY_SIZE, TrainingEntry};

/// Maximum active features per perspective (all non-king pieces + king = at most 16 per side).
const MAX_ACTIVE: usize = 32;

/// A batch of training data ready for the model.
pub struct Batch<B: Backend> {
    pub white_features: Tensor<B, 2, Int>,
    pub black_features: Tensor<B, 2, Int>,
    pub white_count: Tensor<B, 1, Int>,
    pub black_count: Tensor<B, 1, Int>,
    pub stm: Tensor<B, 1, Int>,
    pub target_scores: Tensor<B, 2>,
    pub target_wdl: Tensor<B, 2>,
}

/// Extracted features for a single position (CPU-side, no tensors).
struct PositionFeatures {
    white_indices: [i64; MAX_ACTIVE],
    black_indices: [i64; MAX_ACTIVE],
    white_count: i64,
    black_count: i64,
    stm: i64,
    score: f32,
    wdl: f32,
}

/// All training entries loaded into memory.
pub struct TrainingDataset {
    entries: Vec<TrainingEntry>,
}

impl TrainingDataset {
    /// Load all entries from a binary training data file.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let data = fs::read(path)?;
        let num_entries = data.len() / ENTRY_SIZE;
        let mut entries = Vec::with_capacity(num_entries);

        for i in 0..num_entries {
            let offset = i * ENTRY_SIZE;
            let chunk: &[u8; ENTRY_SIZE] = data[offset..offset + ENTRY_SIZE]
                .try_into()
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "short entry"))?;
            if let Some(entry) = TrainingEntry::from_bytes(chunk) {
                entries.push(entry);
            }
        }

        eprintln!(
            "Loaded {} training entries from {}",
            entries.len(),
            path.display()
        );
        Ok(Self { entries })
    }

    /// Shuffle the dataset in place.
    pub fn shuffle<R: Rng>(&mut self, rng: &mut R) {
        self.entries.shuffle(rng);
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Iterate over batches of the given size.
    pub fn batches<B: Backend>(&self, batch_size: usize, device: &B::Device) -> BatchIter<'_, B> {
        BatchIter {
            dataset: self,
            offset: 0,
            batch_size,
            device: device.clone(),
        }
    }
}

pub struct BatchIter<'a, B: Backend> {
    dataset: &'a TrainingDataset,
    offset: usize,
    batch_size: usize,
    device: B::Device,
}

impl<B: Backend> Iterator for BatchIter<'_, B> {
    type Item = Batch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.dataset.entries.len() {
            return None;
        }

        let end = (self.offset + self.batch_size).min(self.dataset.entries.len());
        let slice = &self.dataset.entries[self.offset..end];
        self.offset = end;

        // Skip incomplete final batches (they cause shape mismatches and bias gradients)
        if slice.len() < self.batch_size {
            return None;
        }

        Some(entries_to_batch::<B>(slice, &self.device))
    }
}

/// Extract NNUE features from a training entry.
fn extract_features(entry: &TrainingEntry) -> PositionFeatures {
    let mut white_king_sq = Square::A1;
    let mut black_king_sq = Square::A1;

    // First pass: find king squares
    for (sq_idx, color, pt) in entry.pieces() {
        if pt == PieceType::King {
            let sq = Square::new(sq_idx as u8);
            match color {
                Color::White => white_king_sq = sq,
                Color::Black => black_king_sq = sq,
            }
        }
    }

    let mut white_indices = [0i64; MAX_ACTIVE];
    let mut black_indices = [0i64; MAX_ACTIVE];
    let mut wc = 0usize;
    let mut bc = 0usize;

    // Second pass: compute feature indices for all pieces
    for (sq_idx, color, pt) in entry.pieces() {
        let sq = Square::new(sq_idx as u8);

        if wc < MAX_ACTIVE {
            white_indices[wc] = feature_index(Color::White, white_king_sq, color, pt, sq) as i64;
            wc += 1;
        }
        if bc < MAX_ACTIVE {
            black_indices[bc] = feature_index(Color::Black, black_king_sq, color, pt, sq) as i64;
            bc += 1;
        }
    }

    let stm = match entry.stm {
        Color::White => 0i64,
        Color::Black => 1i64,
    };

    let score = entry.score as f32;
    let wdl = match entry.wdl {
        crate::data::WdlResult::Loss => 0.0f32,
        crate::data::WdlResult::Draw => 0.5f32,
        crate::data::WdlResult::Win => 1.0f32,
    };

    PositionFeatures {
        white_indices,
        black_indices,
        white_count: wc as i64,
        black_count: bc as i64,
        stm,
        score,
        wdl,
    }
}

/// Convert a slice of training entries into a Batch of tensors.
fn entries_to_batch<B: Backend>(entries: &[TrainingEntry], device: &B::Device) -> Batch<B> {
    let bs = entries.len();

    let mut white_feat_data = vec![0i64; bs * MAX_ACTIVE];
    let mut black_feat_data = vec![0i64; bs * MAX_ACTIVE];
    let mut white_count_data = vec![0i64; bs];
    let mut black_count_data = vec![0i64; bs];
    let mut stm_data = vec![0i64; bs];
    let mut score_data = vec![0f32; bs];
    let mut wdl_data = vec![0f32; bs];

    for (i, entry) in entries.iter().enumerate() {
        let feats = extract_features(entry);
        white_feat_data[i * MAX_ACTIVE..(i + 1) * MAX_ACTIVE].copy_from_slice(&feats.white_indices);
        black_feat_data[i * MAX_ACTIVE..(i + 1) * MAX_ACTIVE].copy_from_slice(&feats.black_indices);
        white_count_data[i] = feats.white_count;
        black_count_data[i] = feats.black_count;
        stm_data[i] = feats.stm;
        score_data[i] = feats.score;
        wdl_data[i] = feats.wdl;
    }

    Batch {
        white_features: Tensor::from_data(
            TensorData::new(white_feat_data, vec![bs, MAX_ACTIVE]),
            device,
        ),
        black_features: Tensor::from_data(
            TensorData::new(black_feat_data, vec![bs, MAX_ACTIVE]),
            device,
        ),
        white_count: Tensor::from_data(TensorData::new(white_count_data, vec![bs]), device),
        black_count: Tensor::from_data(TensorData::new(black_count_data, vec![bs]), device),
        stm: Tensor::from_data(TensorData::new(stm_data, vec![bs]), device),
        target_scores: Tensor::from_data(TensorData::new(score_data, vec![bs, 1]), device),
        target_wdl: Tensor::from_data(TensorData::new(wdl_data, vec![bs, 1]), device),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::data::WdlResult;
    use rage_core::Board;

    #[test]
    fn extract_features_starting_position() {
        let board = Board::starting_position();
        let entry = TrainingEntry::from_board(&board, 15, WdlResult::Draw, 0);
        let feats = extract_features(&entry);

        // Starting position has 32 pieces
        assert_eq!(feats.white_count, 32);
        assert_eq!(feats.black_count, 32);
        assert_eq!(feats.stm, 0); // White to move
        assert!((feats.score - 15.0).abs() < f32::EPSILON);
        assert!((feats.wdl - 0.5).abs() < f32::EPSILON);

        // All feature indices should be in valid range
        for i in 0..feats.white_count as usize {
            assert!(
                (feats.white_indices[i] as usize) < rage_nnue::arch::TOTAL_INPUT_FEATURES,
                "white feature index {} out of range",
                feats.white_indices[i]
            );
        }
        for i in 0..feats.black_count as usize {
            assert!(
                (feats.black_indices[i] as usize) < rage_nnue::arch::TOTAL_INPUT_FEATURES,
                "black feature index {} out of range",
                feats.black_indices[i]
            );
        }
    }

    #[test]
    fn extract_features_kk_endgame() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let entry = TrainingEntry::from_board(&board, 0, WdlResult::Draw, 0);
        let feats = extract_features(&entry);

        // Only 2 pieces (2 kings)
        assert_eq!(feats.white_count, 2);
        assert_eq!(feats.black_count, 2);
    }

    #[test]
    fn batch_creation() {
        use burn::backend::NdArray;

        let board = Board::starting_position();
        let entries: Vec<TrainingEntry> = (0..4)
            .map(|i| TrainingEntry::from_board(&board, i * 10, WdlResult::Win, 0))
            .collect();

        let device = burn::tensor::Device::<NdArray>::default();
        let batch = entries_to_batch::<NdArray>(&entries, &device);

        assert_eq!(batch.white_features.dims(), [4, MAX_ACTIVE]);
        assert_eq!(batch.black_features.dims(), [4, MAX_ACTIVE]);
        assert_eq!(batch.white_count.dims(), [4]);
        assert_eq!(batch.target_scores.dims(), [4, 1]);
        assert_eq!(batch.target_wdl.dims(), [4, 1]);
    }
}
