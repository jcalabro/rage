use crate::arch::HIDDEN_SIZE;

/// NNUE accumulator holding the dense hidden-layer values for both perspectives.
///
/// Each perspective has `HIDDEN_SIZE` i16 values. These are initialized from the
/// feature transformer biases and then incrementally updated as pieces move.
#[derive(Clone)]
pub struct Accumulator {
    /// Accumulator values for each perspective: [White, Black].
    pub values: [[i16; HIDDEN_SIZE]; 2],
    /// Whether each perspective needs a full refresh (e.g., after king bucket change).
    pub dirty: [bool; 2],
}

impl Accumulator {
    /// An empty accumulator (all zeros, both perspectives dirty).
    pub const EMPTY: Self = Self {
        values: [[0i16; HIDDEN_SIZE]; 2],
        dirty: [true, true],
    };

    /// Initialize the accumulator with the feature transformer biases.
    pub fn init_from_biases(&mut self, biases: &[i16; HIDDEN_SIZE]) {
        self.values[0] = *biases;
        self.values[1] = *biases;
        self.dirty = [false, false];
    }

    /// Add a feature's weight column to a perspective's accumulator.
    pub fn add_feature(&mut self, perspective: usize, weights: &[i16; HIDDEN_SIZE]) {
        crate::simd::ops().accumulator_add(&mut self.values[perspective], weights);
    }

    /// Subtract a feature's weight column from a perspective's accumulator.
    pub fn sub_feature(&mut self, perspective: usize, weights: &[i16; HIDDEN_SIZE]) {
        crate::simd::ops().accumulator_sub(&mut self.values[perspective], weights);
    }
}
