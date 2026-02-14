use std::sync::OnceLock;

use crate::arch::{CRELU_OUTPUT_SIZE, HIDDEN_SIZE};

mod scalar;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse41;

#[cfg(target_arch = "aarch64")]
mod neon;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm;

/// SIMD-accelerated operations for NNUE inference.
///
/// Each implementation must produce bit-identical results to the scalar reference.
pub trait SimdOps: Sync {
    /// Human-readable name of this SIMD tier (e.g., "avx2", "scalar").
    fn name(&self) -> &'static str;

    /// Add weight column to accumulator: `acc[i] += weights[i]`.
    fn accumulator_add(&self, acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]);

    /// Subtract weight column from accumulator: `acc[i] -= weights[i]`.
    fn accumulator_sub(&self, acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]);

    /// Fused CReLU + linear output computation.
    ///
    /// Clamps both perspectives to `[0, QA]`, computes dot product with output weights.
    /// Returns the raw sum (caller adds bias and divides by OUTPUT_SCALE).
    fn crelu_linear(
        &self,
        stm: &[i16; HIDDEN_SIZE],
        nstm: &[i16; HIDDEN_SIZE],
        weights: &[i8; CRELU_OUTPUT_SIZE],
    ) -> i32;
}

static OPS: OnceLock<&'static dyn SimdOps> = OnceLock::new();

/// Get the SIMD ops for the current platform. Auto-initializes on first call.
pub fn ops() -> &'static dyn SimdOps {
    *OPS.get_or_init(init)
}

/// Detect CPU features and select the best available SIMD tier.
#[allow(unreachable_code)]
fn init() -> &'static dyn SimdOps {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return &avx2::Avx2Ops;
        }
        if is_x86_feature_detected!("sse4.1") {
            return &sse41::Sse41Ops;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return &neon::NeonOps;
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return &wasm::WasmSimdOps;
    }

    &scalar::ScalarOps
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::arch::QA;

    /// Simple xorshift64 PRNG for generating test data.
    struct Xorshift(u64);

    impl Xorshift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn next_i16(&mut self) -> i16 {
            self.next() as i16
        }

        fn next_i8(&mut self) -> i8 {
            self.next() as i8
        }
    }

    fn random_i16_array(seed: u64) -> [i16; HIDDEN_SIZE] {
        let mut rng = Xorshift(seed);
        let mut arr = [0i16; HIDDEN_SIZE];
        for v in &mut arr {
            *v = rng.next_i16();
        }
        arr
    }

    fn random_i8_array(seed: u64) -> [i8; CRELU_OUTPUT_SIZE] {
        let mut rng = Xorshift(seed);
        let mut arr = [0i8; CRELU_OUTPUT_SIZE];
        for v in &mut arr {
            *v = rng.next_i8();
        }
        arr
    }

    /// Reference scalar implementation for comparison.
    fn scalar_crelu_linear(
        stm: &[i16; HIDDEN_SIZE],
        nstm: &[i16; HIDDEN_SIZE],
        weights: &[i8; CRELU_OUTPUT_SIZE],
    ) -> i32 {
        scalar::ScalarOps.crelu_linear(stm, nstm, weights)
    }

    #[test]
    fn dispatch_selects_valid_tier() {
        let ops = super::ops();
        let name = ops.name();
        assert!(
            ["scalar", "sse41", "avx2", "neon", "wasm-simd128"].contains(&name),
            "unexpected SIMD tier: {name}"
        );
    }

    #[test]
    fn scalar_add_sub_roundtrip() {
        let mut acc = random_i16_array(1);
        let original = acc;
        let weights = random_i16_array(2);

        scalar::ScalarOps.accumulator_add(&mut acc, &weights);
        // After add, values should differ (with high probability)
        assert_ne!(acc, original);

        scalar::ScalarOps.accumulator_sub(&mut acc, &weights);
        // After sub, should be back to original
        assert_eq!(acc, original);
    }

    #[test]
    fn scalar_crelu_known_values() {
        let mut stm = [0i16; HIDDEN_SIZE];
        let mut nstm = [0i16; HIDDEN_SIZE];

        stm[0] = -100; // clamps to 0
        stm[1] = 500; // clamps to QA (255)
        stm[2] = 128; // passes through
        nstm[0] = -1; // clamps to 0
        nstm[1] = 255; // exactly QA, passes through

        let mut weights = [0i8; CRELU_OUTPUT_SIZE];
        weights[0] = 1;
        weights[1] = 1;
        weights[2] = 1;
        weights[HIDDEN_SIZE] = 1;
        weights[HIDDEN_SIZE + 1] = 1;

        let result = scalar::ScalarOps.crelu_linear(&stm, &nstm, &weights);
        // Expected: 0*1 + 255*1 + 128*1 + 0*1 + 255*1 = 638
        assert_eq!(result, 638);
    }

    #[test]
    fn scalar_crelu_boundary_values() {
        let mut stm = [0i16; HIDDEN_SIZE];
        let nstm = [0i16; HIDDEN_SIZE];

        // Test exact boundary: 0 and QA
        stm[0] = 0;
        stm[1] = QA as i16;
        stm[2] = -32768; // extreme negative
        stm[3] = 32767; // extreme positive

        let mut weights = [0i8; CRELU_OUTPUT_SIZE];
        weights[0] = 1;
        weights[1] = 1;
        weights[2] = 1;
        weights[3] = 1;

        let result = scalar::ScalarOps.crelu_linear(&stm, &nstm, &weights);
        // Expected: 0*1 + 255*1 + 0*1 + 255*1 = 510
        assert_eq!(result, 510);
    }

    // --- AVX2 tests (x86/x86_64 only) ---

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    mod avx2_tests {
        use super::*;

        fn has_avx2() -> bool {
            is_x86_feature_detected!("avx2")
        }

        #[test]
        fn avx2_accumulator_add_matches_scalar() {
            if !has_avx2() {
                return;
            }
            for seed in 1..=10u64 {
                let mut acc_scalar = random_i16_array(seed);
                let mut acc_avx2 = acc_scalar;
                let weights = random_i16_array(seed + 100);

                scalar::ScalarOps.accumulator_add(&mut acc_scalar, &weights);
                super::super::avx2::Avx2Ops.accumulator_add(&mut acc_avx2, &weights);
                assert_eq!(acc_scalar, acc_avx2, "add mismatch for seed {seed}");
            }
        }

        #[test]
        fn avx2_accumulator_sub_matches_scalar() {
            if !has_avx2() {
                return;
            }
            for seed in 1..=10u64 {
                let mut acc_scalar = random_i16_array(seed);
                let mut acc_avx2 = acc_scalar;
                let weights = random_i16_array(seed + 100);

                scalar::ScalarOps.accumulator_sub(&mut acc_scalar, &weights);
                super::super::avx2::Avx2Ops.accumulator_sub(&mut acc_avx2, &weights);
                assert_eq!(acc_scalar, acc_avx2, "sub mismatch for seed {seed}");
            }
        }

        #[test]
        fn avx2_crelu_linear_matches_scalar() {
            if !has_avx2() {
                return;
            }
            for seed in 1..=10u64 {
                let stm = random_i16_array(seed);
                let nstm = random_i16_array(seed + 50);
                let weights = random_i8_array(seed + 100);

                let scalar_result = scalar_crelu_linear(&stm, &nstm, &weights);
                let avx2_result = super::super::avx2::Avx2Ops.crelu_linear(&stm, &nstm, &weights);
                assert_eq!(
                    scalar_result, avx2_result,
                    "crelu_linear mismatch for seed {seed}"
                );
            }
        }
    }

    // --- SSE4.1 tests (x86/x86_64 only) ---

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    mod sse41_tests {
        use super::*;

        fn has_sse41() -> bool {
            is_x86_feature_detected!("sse4.1")
        }

        #[test]
        fn sse41_accumulator_add_matches_scalar() {
            if !has_sse41() {
                return;
            }
            for seed in 1..=10u64 {
                let mut acc_scalar = random_i16_array(seed);
                let mut acc_sse41 = acc_scalar;
                let weights = random_i16_array(seed + 100);

                scalar::ScalarOps.accumulator_add(&mut acc_scalar, &weights);
                super::super::sse41::Sse41Ops.accumulator_add(&mut acc_sse41, &weights);
                assert_eq!(acc_scalar, acc_sse41, "add mismatch for seed {seed}");
            }
        }

        #[test]
        fn sse41_accumulator_sub_matches_scalar() {
            if !has_sse41() {
                return;
            }
            for seed in 1..=10u64 {
                let mut acc_scalar = random_i16_array(seed);
                let mut acc_sse41 = acc_scalar;
                let weights = random_i16_array(seed + 100);

                scalar::ScalarOps.accumulator_sub(&mut acc_scalar, &weights);
                super::super::sse41::Sse41Ops.accumulator_sub(&mut acc_sse41, &weights);
                assert_eq!(acc_scalar, acc_sse41, "sub mismatch for seed {seed}");
            }
        }

        #[test]
        fn sse41_crelu_linear_matches_scalar() {
            if !has_sse41() {
                return;
            }
            for seed in 1..=10u64 {
                let stm = random_i16_array(seed);
                let nstm = random_i16_array(seed + 50);
                let weights = random_i8_array(seed + 100);

                let scalar_result = scalar_crelu_linear(&stm, &nstm, &weights);
                let sse41_result =
                    super::super::sse41::Sse41Ops.crelu_linear(&stm, &nstm, &weights);
                assert_eq!(
                    scalar_result, sse41_result,
                    "crelu_linear mismatch for seed {seed}"
                );
            }
        }
    }

    #[test]
    fn full_forward_consistency() {
        // Verify that the dispatched ops produce the same result as scalar
        // for a complete forward pass equivalent.
        let stm = random_i16_array(42);
        let nstm = random_i16_array(43);
        let weights = random_i8_array(44);

        let scalar_result = scalar_crelu_linear(&stm, &nstm, &weights);
        let dispatched_result = ops().crelu_linear(&stm, &nstm, &weights);

        assert_eq!(
            scalar_result,
            dispatched_result,
            "dispatched ops ({}) differs from scalar",
            ops().name()
        );
    }
}
