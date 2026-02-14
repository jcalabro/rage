use super::SimdOps;
use crate::arch::{CRELU_OUTPUT_SIZE, HIDDEN_SIZE, QA};

/// Scalar (portable) SIMD implementation. Used as fallback on all platforms
/// and as the reference implementation for correctness testing.
pub struct ScalarOps;

impl SimdOps for ScalarOps {
    fn name(&self) -> &'static str {
        "scalar"
    }

    fn accumulator_add(&self, acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
        for (a, &w) in acc.iter_mut().zip(weights.iter()) {
            *a = a.wrapping_add(w);
        }
    }

    fn accumulator_sub(&self, acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
        for (a, &w) in acc.iter_mut().zip(weights.iter()) {
            *a = a.wrapping_sub(w);
        }
    }

    fn crelu_linear(
        &self,
        stm: &[i16; HIDDEN_SIZE],
        nstm: &[i16; HIDDEN_SIZE],
        weights: &[i8; CRELU_OUTPUT_SIZE],
    ) -> i32 {
        let mut output = 0i32;

        for (&val, &w) in stm.iter().zip(weights[..HIDDEN_SIZE].iter()) {
            let clamped = val.max(0).min(QA as i16) as i32;
            output += clamped * w as i32;
        }

        for (&val, &w) in nstm.iter().zip(weights[HIDDEN_SIZE..].iter()) {
            let clamped = val.max(0).min(QA as i16) as i32;
            output += clamped * w as i32;
        }

        output
    }
}
