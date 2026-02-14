use std::arch::wasm32::*;

use super::SimdOps;
use crate::arch::{CRELU_OUTPUT_SIZE, HIDDEN_SIZE, QA};

/// WebAssembly SIMD implementation. Processes 8 i16 values per iteration (128-bit v128).
/// Requires compilation with `-C target-feature=+simd128`.
pub struct WasmSimdOps;

impl SimdOps for WasmSimdOps {
    fn name(&self) -> &'static str {
        "wasm-simd128"
    }

    fn accumulator_add(&self, acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
        // SAFETY: wasm SIMD128 is available (guaranteed by cfg target_feature).
        // All pointer offsets stay within array bounds (HIDDEN_SIZE divisible by 8).
        unsafe {
            for i in 0..HIDDEN_SIZE / 8 {
                let offset = i * 8;
                let a = v128_load(acc.as_ptr().add(offset).cast());
                let w = v128_load(weights.as_ptr().add(offset).cast());
                let result = i16x8_add(a, w);
                v128_store(acc.as_mut_ptr().add(offset).cast(), result);
            }
        }
    }

    fn accumulator_sub(&self, acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
        // SAFETY: Same as accumulator_add.
        unsafe {
            for i in 0..HIDDEN_SIZE / 8 {
                let offset = i * 8;
                let a = v128_load(acc.as_ptr().add(offset).cast());
                let w = v128_load(weights.as_ptr().add(offset).cast());
                let result = i16x8_sub(a, w);
                v128_store(acc.as_mut_ptr().add(offset).cast(), result);
            }
        }
    }

    fn crelu_linear(
        &self,
        stm: &[i16; HIDDEN_SIZE],
        nstm: &[i16; HIDDEN_SIZE],
        weights: &[i8; CRELU_OUTPUT_SIZE],
    ) -> i32 {
        // SAFETY: wasm SIMD128 is available. All pointer offsets are within bounds.
        // Weight loads use v128_load64_zero (64-bit) for 8 i8 values.
        unsafe {
            let zero = i16x8_splat(0);
            let qa = i16x8_splat(QA as i16);
            let mut sum = i32x4_splat(0);

            // Process STM perspective
            for i in 0..HIDDEN_SIZE / 8 {
                let offset = i * 8;

                let vals = v128_load(stm.as_ptr().add(offset).cast());
                let clamped = i16x8_min(i16x8_max(vals, zero), qa);

                // Load 8 i8 weights (64 bits) and sign-extend to i16
                let w64 = v128_load64_zero(weights.as_ptr().add(offset).cast());
                let w16 = i16x8_extend_low_i8x16(w64);

                // Extended multiply low/high halves -> i32, then accumulate
                sum = i32x4_add(sum, i32x4_extmul_low_i16x8(clamped, w16));
                sum = i32x4_add(sum, i32x4_extmul_high_i16x8(clamped, w16));
            }

            // Process NSTM perspective
            for i in 0..HIDDEN_SIZE / 8 {
                let offset = i * 8;

                let vals = v128_load(nstm.as_ptr().add(offset).cast());
                let clamped = i16x8_min(i16x8_max(vals, zero), qa);

                let w64 = v128_load64_zero(weights.as_ptr().add(HIDDEN_SIZE + offset).cast());
                let w16 = i16x8_extend_low_i8x16(w64);

                sum = i32x4_add(sum, i32x4_extmul_low_i16x8(clamped, w16));
                sum = i32x4_add(sum, i32x4_extmul_high_i16x8(clamped, w16));
            }

            // Horizontal sum of 4 i32 values
            let shuffled = i32x4_shuffle::<2, 3, 0, 1>(sum, sum);
            let sum2 = i32x4_add(sum, shuffled);
            let shuffled2 = i32x4_shuffle::<1, 0, 3, 2>(sum2, sum2);
            let final_sum = i32x4_add(sum2, shuffled2);
            i32x4_extract_lane::<0>(final_sum)
        }
    }
}
