#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::SimdOps;
use crate::arch::{CRELU_OUTPUT_SIZE, HIDDEN_SIZE, QA};

/// AVX2 SIMD implementation. Processes 16 i16 values per iteration (256-bit registers).
pub struct Avx2Ops;

impl SimdOps for Avx2Ops {
    fn name(&self) -> &'static str {
        "avx2"
    }

    fn accumulator_add(&self, acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
        // SAFETY: AVX2 availability verified via is_x86_feature_detected! at init time.
        // Pointers are valid and derived from properly sized arrays. HIDDEN_SIZE (256)
        // is divisible by 16 (elements per 256-bit register).
        unsafe { avx2_accumulator_add(acc, weights) }
    }

    fn accumulator_sub(&self, acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
        // SAFETY: Same as accumulator_add.
        unsafe { avx2_accumulator_sub(acc, weights) }
    }

    fn crelu_linear(
        &self,
        stm: &[i16; HIDDEN_SIZE],
        nstm: &[i16; HIDDEN_SIZE],
        weights: &[i8; CRELU_OUTPUT_SIZE],
    ) -> i32 {
        // SAFETY: AVX2 verified at init time. All arrays are properly sized.
        unsafe { avx2_crelu_linear(stm, nstm, weights) }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_accumulator_add(acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
    // SAFETY: AVX2 is available (guaranteed by target_feature + caller check).
    // All pointer offsets stay within array bounds: offset ranges [0, HIDDEN_SIZE)
    // with HIDDEN_SIZE divisible by 16.
    unsafe {
        let acc_ptr = acc.as_mut_ptr();
        let w_ptr = weights.as_ptr();

        for i in 0..HIDDEN_SIZE / 16 {
            let offset = i * 16;
            let a = _mm256_loadu_si256(acc_ptr.add(offset).cast());
            let w = _mm256_loadu_si256(w_ptr.add(offset).cast());
            let result = _mm256_add_epi16(a, w);
            _mm256_storeu_si256(acc_ptr.add(offset).cast(), result);
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_accumulator_sub(acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
    // SAFETY: Same as avx2_accumulator_add.
    unsafe {
        let acc_ptr = acc.as_mut_ptr();
        let w_ptr = weights.as_ptr();

        for i in 0..HIDDEN_SIZE / 16 {
            let offset = i * 16;
            let a = _mm256_loadu_si256(acc_ptr.add(offset).cast());
            let w = _mm256_loadu_si256(w_ptr.add(offset).cast());
            let result = _mm256_sub_epi16(a, w);
            _mm256_storeu_si256(acc_ptr.add(offset).cast(), result);
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_crelu_linear(
    stm: &[i16; HIDDEN_SIZE],
    nstm: &[i16; HIDDEN_SIZE],
    weights: &[i8; CRELU_OUTPUT_SIZE],
) -> i32 {
    // SAFETY: AVX2 is available. All pointer offsets are within array bounds.
    // HIDDEN_SIZE (256) is divisible by 16. Weight loads use _mm_loadu_si128 (128-bit)
    // for 16 i8 values; max offset is HIDDEN_SIZE + HIDDEN_SIZE - 16 = 496,
    // well within CRELU_OUTPUT_SIZE (512).
    unsafe {
        let zero = _mm256_setzero_si256();
        let qa_vec = _mm256_set1_epi16(QA as i16);
        let ones = _mm256_set1_epi16(1);
        let mut sum = _mm256_setzero_si256();

        // Process STM perspective (first HIDDEN_SIZE weights)
        for i in 0..HIDDEN_SIZE / 16 {
            let offset = i * 16;

            let vals = _mm256_loadu_si256(stm.as_ptr().add(offset).cast());
            let clamped = _mm256_min_epi16(_mm256_max_epi16(vals, zero), qa_vec);

            // Load 16 i8 weights and sign-extend to 16 i16
            let w128 = _mm_loadu_si128(weights.as_ptr().add(offset).cast());
            let w256 = _mm256_cvtepi8_epi16(w128);

            // i16 * i16 -> i16 (safe: max |product| = 255 * 128 = 32640, fits in i16)
            let products = _mm256_mullo_epi16(clamped, w256);

            // Pairwise sum adjacent i16 to i32
            let pairs = _mm256_madd_epi16(products, ones);
            sum = _mm256_add_epi32(sum, pairs);
        }

        // Process NSTM perspective (weights[HIDDEN_SIZE..])
        for i in 0..HIDDEN_SIZE / 16 {
            let offset = i * 16;

            let vals = _mm256_loadu_si256(nstm.as_ptr().add(offset).cast());
            let clamped = _mm256_min_epi16(_mm256_max_epi16(vals, zero), qa_vec);

            let w128 = _mm_loadu_si128(weights.as_ptr().add(HIDDEN_SIZE + offset).cast());
            let w256 = _mm256_cvtepi8_epi16(w128);

            let products = _mm256_mullo_epi16(clamped, w256);
            let pairs = _mm256_madd_epi16(products, ones);
            sum = _mm256_add_epi32(sum, pairs);
        }

        hsum_epi32_avx2(sum)
    }
}

/// Horizontal sum of 8 i32 values in an __m256i register.
#[target_feature(enable = "avx2")]
unsafe fn hsum_epi32_avx2(v: __m256i) -> i32 {
    // SAFETY: AVX2 is available (guaranteed by target_feature + caller check).
    // All intrinsics operate on register values, no memory access.

    // Add high 128 bits to low 128 bits -> 4 x i32
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(lo, hi);

    // Shuffle pairs and add -> 2 unique i32
    let shuffled = _mm_shuffle_epi32(sum128, 0b_01_00_11_10);
    let sum2 = _mm_add_epi32(sum128, shuffled);

    // Final reduction -> 1 i32
    let shuffled2 = _mm_shuffle_epi32(sum2, 0b_00_00_00_01);
    let final_sum = _mm_add_epi32(sum2, shuffled2);

    _mm_cvtsi128_si32(final_sum)
}
