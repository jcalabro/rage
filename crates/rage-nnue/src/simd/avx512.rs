#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::SimdOps;
use crate::arch::{CRELU_OUTPUT_SIZE, HIDDEN_SIZE, QA};

/// AVX-512 SIMD implementation. Processes 32 i16 values per iteration (512-bit registers).
pub struct Avx512Ops;

impl SimdOps for Avx512Ops {
    fn name(&self) -> &'static str {
        "avx512"
    }

    fn accumulator_add(&self, acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
        // SAFETY: AVX-512F+BW availability verified via is_x86_feature_detected! at init time.
        // Pointers are valid and derived from properly sized arrays. HIDDEN_SIZE (256)
        // is divisible by 32 (elements per 512-bit register).
        unsafe { avx512_accumulator_add(acc, weights) }
    }

    fn accumulator_sub(&self, acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
        // SAFETY: Same as accumulator_add.
        unsafe { avx512_accumulator_sub(acc, weights) }
    }

    fn crelu_linear(
        &self,
        stm: &[i16; HIDDEN_SIZE],
        nstm: &[i16; HIDDEN_SIZE],
        weights: &[i8; CRELU_OUTPUT_SIZE],
    ) -> i32 {
        // SAFETY: AVX-512F+BW verified at init time. All arrays are properly sized.
        unsafe { avx512_crelu_linear(stm, nstm, weights) }
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn avx512_accumulator_add(acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
    // SAFETY: AVX-512F+BW is available (guaranteed by target_feature + caller check).
    // All pointer offsets stay within array bounds: offset ranges [0, HIDDEN_SIZE)
    // with HIDDEN_SIZE divisible by 32.
    unsafe {
        let acc_ptr = acc.as_mut_ptr();
        let w_ptr = weights.as_ptr();

        for i in 0..HIDDEN_SIZE / 32 {
            let offset = i * 32;
            let a = _mm512_loadu_si512(acc_ptr.add(offset).cast());
            let w = _mm512_loadu_si512(w_ptr.add(offset).cast());
            let result = _mm512_add_epi16(a, w);
            _mm512_storeu_si512(acc_ptr.add(offset).cast(), result);
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn avx512_accumulator_sub(acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
    // SAFETY: Same as avx512_accumulator_add.
    unsafe {
        let acc_ptr = acc.as_mut_ptr();
        let w_ptr = weights.as_ptr();

        for i in 0..HIDDEN_SIZE / 32 {
            let offset = i * 32;
            let a = _mm512_loadu_si512(acc_ptr.add(offset).cast());
            let w = _mm512_loadu_si512(w_ptr.add(offset).cast());
            let result = _mm512_sub_epi16(a, w);
            _mm512_storeu_si512(acc_ptr.add(offset).cast(), result);
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn avx512_crelu_linear(
    stm: &[i16; HIDDEN_SIZE],
    nstm: &[i16; HIDDEN_SIZE],
    weights: &[i8; CRELU_OUTPUT_SIZE],
) -> i32 {
    // SAFETY: AVX-512F+BW is available. All pointer offsets are within array bounds.
    // HIDDEN_SIZE (256) is divisible by 32. Weight loads use _mm256_loadu_si256 (256-bit)
    // for 32 i8 values, then sign-extended to 512-bit; max offset is
    // HIDDEN_SIZE + HIDDEN_SIZE - 32 = 480, well within CRELU_OUTPUT_SIZE (512).
    unsafe {
        let zero = _mm512_setzero_si512();
        let qa_vec = _mm512_set1_epi16(QA as i16);
        let ones = _mm512_set1_epi16(1);
        let mut sum = _mm512_setzero_si512();

        // Process STM perspective (first HIDDEN_SIZE weights)
        for i in 0..HIDDEN_SIZE / 32 {
            let offset = i * 32;

            let vals = _mm512_loadu_si512(stm.as_ptr().add(offset).cast());
            let clamped = _mm512_min_epi16(_mm512_max_epi16(vals, zero), qa_vec);

            // Load 32 i8 weights and sign-extend to 32 i16
            let w256 = _mm256_loadu_si256(weights.as_ptr().add(offset).cast());
            let w512 = _mm512_cvtepi8_epi16(w256);

            // i16 * i16 -> i16 (safe: max |product| = 255 * 128 = 32640, fits in i16)
            let products = _mm512_mullo_epi16(clamped, w512);

            // Pairwise sum adjacent i16 to i32
            let pairs = _mm512_madd_epi16(products, ones);
            sum = _mm512_add_epi32(sum, pairs);
        }

        // Process NSTM perspective (weights[HIDDEN_SIZE..])
        for i in 0..HIDDEN_SIZE / 32 {
            let offset = i * 32;

            let vals = _mm512_loadu_si512(nstm.as_ptr().add(offset).cast());
            let clamped = _mm512_min_epi16(_mm512_max_epi16(vals, zero), qa_vec);

            let w256 = _mm256_loadu_si256(weights.as_ptr().add(HIDDEN_SIZE + offset).cast());
            let w512 = _mm512_cvtepi8_epi16(w256);

            let products = _mm512_mullo_epi16(clamped, w512);
            let pairs = _mm512_madd_epi16(products, ones);
            sum = _mm512_add_epi32(sum, pairs);
        }

        _mm512_reduce_add_epi32(sum)
    }
}
