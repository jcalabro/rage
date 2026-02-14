use std::arch::aarch64::*;

use super::SimdOps;
use crate::arch::{CRELU_OUTPUT_SIZE, HIDDEN_SIZE, QA};

/// ARM NEON SIMD implementation. Processes 8 i16 values per iteration (128-bit registers).
/// NEON is always available on AArch64.
pub struct NeonOps;

impl SimdOps for NeonOps {
    fn name(&self) -> &'static str {
        "neon"
    }

    fn accumulator_add(&self, acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
        // SAFETY: NEON is always available on AArch64. All pointer offsets stay
        // within array bounds (HIDDEN_SIZE divisible by 8).
        unsafe {
            for i in 0..HIDDEN_SIZE / 8 {
                let offset = i * 8;
                let a = vld1q_s16(acc.as_ptr().add(offset));
                let w = vld1q_s16(weights.as_ptr().add(offset));
                let result = vaddq_s16(a, w);
                vst1q_s16(acc.as_mut_ptr().add(offset), result);
            }
        }
    }

    fn accumulator_sub(&self, acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
        // SAFETY: Same as accumulator_add.
        unsafe {
            for i in 0..HIDDEN_SIZE / 8 {
                let offset = i * 8;
                let a = vld1q_s16(acc.as_ptr().add(offset));
                let w = vld1q_s16(weights.as_ptr().add(offset));
                let result = vsubq_s16(a, w);
                vst1q_s16(acc.as_mut_ptr().add(offset), result);
            }
        }
    }

    fn crelu_linear(
        &self,
        stm: &[i16; HIDDEN_SIZE],
        nstm: &[i16; HIDDEN_SIZE],
        weights: &[i8; CRELU_OUTPUT_SIZE],
    ) -> i32 {
        // SAFETY: NEON is always available on AArch64. All pointer offsets
        // are within array bounds. Weight loads use vld1_s8 (64-bit) for
        // 8 i8 values; max offset = HIDDEN_SIZE + HIDDEN_SIZE - 8 = 504,
        // within CRELU_OUTPUT_SIZE (512).
        unsafe {
            let zero = vdupq_n_s16(0);
            let qa = vdupq_n_s16(QA as i16);
            let mut sum = vdupq_n_s32(0);

            // Process STM perspective
            for i in 0..HIDDEN_SIZE / 8 {
                let offset = i * 8;

                let vals = vld1q_s16(stm.as_ptr().add(offset));
                let clamped = vminq_s16(vmaxq_s16(vals, zero), qa);

                // Load 8 i8 weights and sign-extend to i16
                let w8 = vld1_s8(weights.as_ptr().add(offset));
                let w16 = vmovl_s8(w8);

                // Multiply-accumulate low and high halves separately (i16 -> i32)
                sum = vmlal_s16(sum, vget_low_s16(clamped), vget_low_s16(w16));
                sum = vmlal_s16(sum, vget_high_s16(clamped), vget_high_s16(w16));
            }

            // Process NSTM perspective
            for i in 0..HIDDEN_SIZE / 8 {
                let offset = i * 8;

                let vals = vld1q_s16(nstm.as_ptr().add(offset));
                let clamped = vminq_s16(vmaxq_s16(vals, zero), qa);

                let w8 = vld1_s8(weights.as_ptr().add(HIDDEN_SIZE + offset));
                let w16 = vmovl_s8(w8);

                sum = vmlal_s16(sum, vget_low_s16(clamped), vget_low_s16(w16));
                sum = vmlal_s16(sum, vget_high_s16(clamped), vget_high_s16(w16));
            }

            // Horizontal sum of 4 i32 values
            vaddvq_s32(sum)
        }
    }
}
