#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::SimdOps;
use crate::arch::{CRELU_OUTPUT_SIZE, HIDDEN_SIZE, QA};

/// SSE4.1 SIMD implementation. Processes 8 i16 values per iteration (128-bit registers).
pub struct Sse41Ops;

impl SimdOps for Sse41Ops {
    fn name(&self) -> &'static str {
        "sse41"
    }

    fn accumulator_add(&self, acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
        // SAFETY: SSE4.1 availability verified via is_x86_feature_detected! at init time.
        // Pointers are valid and derived from properly sized arrays. HIDDEN_SIZE (256)
        // is divisible by 8 (elements per 128-bit register).
        unsafe { sse41_accumulator_add(acc, weights) }
    }

    fn accumulator_sub(&self, acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
        // SAFETY: Same as accumulator_add.
        unsafe { sse41_accumulator_sub(acc, weights) }
    }

    fn crelu_linear(
        &self,
        stm: &[i16; HIDDEN_SIZE],
        nstm: &[i16; HIDDEN_SIZE],
        weights: &[i8; CRELU_OUTPUT_SIZE],
    ) -> i32 {
        // SAFETY: SSE4.1 verified at init time. All arrays are properly sized.
        unsafe { sse41_crelu_linear(stm, nstm, weights) }
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse41_accumulator_add(acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
    // SAFETY: SSE4.1 is available (guaranteed by target_feature + caller check).
    // All pointer offsets stay within array bounds: offset ranges [0, HIDDEN_SIZE)
    // with HIDDEN_SIZE divisible by 8.
    unsafe {
        let acc_ptr = acc.as_mut_ptr();
        let w_ptr = weights.as_ptr();

        for i in 0..HIDDEN_SIZE / 8 {
            let offset = i * 8;
            let a = _mm_loadu_si128(acc_ptr.add(offset).cast());
            let w = _mm_loadu_si128(w_ptr.add(offset).cast());
            let result = _mm_add_epi16(a, w);
            _mm_storeu_si128(acc_ptr.add(offset).cast(), result);
        }
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse41_accumulator_sub(acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
    // SAFETY: Same as sse41_accumulator_add.
    unsafe {
        let acc_ptr = acc.as_mut_ptr();
        let w_ptr = weights.as_ptr();

        for i in 0..HIDDEN_SIZE / 8 {
            let offset = i * 8;
            let a = _mm_loadu_si128(acc_ptr.add(offset).cast());
            let w = _mm_loadu_si128(w_ptr.add(offset).cast());
            let result = _mm_sub_epi16(a, w);
            _mm_storeu_si128(acc_ptr.add(offset).cast(), result);
        }
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse41_crelu_linear(
    stm: &[i16; HIDDEN_SIZE],
    nstm: &[i16; HIDDEN_SIZE],
    weights: &[i8; CRELU_OUTPUT_SIZE],
) -> i32 {
    // SAFETY: SSE4.1 is available. All pointer offsets are within array bounds.
    // HIDDEN_SIZE (256) is divisible by 8. Weight loads use _mm_loadl_epi64 (64-bit)
    // for 8 i8 values; max offset is HIDDEN_SIZE + HIDDEN_SIZE - 8 = 504,
    // well within CRELU_OUTPUT_SIZE (512).
    unsafe {
        let zero = _mm_setzero_si128();
        let qa_vec = _mm_set1_epi16(QA as i16);
        let ones = _mm_set1_epi16(1);
        let mut sum = _mm_setzero_si128();

        // Process STM perspective
        for i in 0..HIDDEN_SIZE / 8 {
            let offset = i * 8;

            let vals = _mm_loadu_si128(stm.as_ptr().add(offset).cast());
            let clamped = _mm_min_epi16(_mm_max_epi16(vals, zero), qa_vec);

            // Load 8 i8 weights (64 bits) and sign-extend to 8 i16
            let w64 = _mm_loadl_epi64(weights.as_ptr().add(offset).cast());
            let w128 = _mm_cvtepi8_epi16(w64);

            let products = _mm_mullo_epi16(clamped, w128);
            let pairs = _mm_madd_epi16(products, ones);
            sum = _mm_add_epi32(sum, pairs);
        }

        // Process NSTM perspective
        for i in 0..HIDDEN_SIZE / 8 {
            let offset = i * 8;

            let vals = _mm_loadu_si128(nstm.as_ptr().add(offset).cast());
            let clamped = _mm_min_epi16(_mm_max_epi16(vals, zero), qa_vec);

            let w64 = _mm_loadl_epi64(weights.as_ptr().add(HIDDEN_SIZE + offset).cast());
            let w128 = _mm_cvtepi8_epi16(w64);

            let products = _mm_mullo_epi16(clamped, w128);
            let pairs = _mm_madd_epi16(products, ones);
            sum = _mm_add_epi32(sum, pairs);
        }

        hsum_epi32_sse(sum)
    }
}

/// Horizontal sum of 4 i32 values in an __m128i register.
#[target_feature(enable = "sse4.1")]
unsafe fn hsum_epi32_sse(v: __m128i) -> i32 {
    // SAFETY: SSE4.1 is available (guaranteed by target_feature + caller check).
    // All intrinsics operate on register values, no memory access.
    let shuffled = _mm_shuffle_epi32(v, 0b_01_00_11_10);
    let sum2 = _mm_add_epi32(v, shuffled);
    let shuffled2 = _mm_shuffle_epi32(sum2, 0b_00_00_00_01);
    let final_sum = _mm_add_epi32(sum2, shuffled2);
    _mm_cvtsi128_si32(final_sum)
}
