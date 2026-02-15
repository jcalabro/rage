/// Number of king buckets for the HalfKAv2 feature transformer.
/// 32 effective king half-squares (files a-d, mirrored) map to 10 buckets.
pub const NUM_KING_BUCKETS: usize = 10;

/// Features per perspective: 64 squares x 6 piece types x 2 colors.
pub const FEATURES_PER_PERSPECTIVE: usize = 768;

/// Total input features: king buckets x features per perspective.
pub const TOTAL_INPUT_FEATURES: usize = NUM_KING_BUCKETS * FEATURES_PER_PERSPECTIVE;

/// Hidden layer size (accumulator width).
pub const HIDDEN_SIZE: usize = 256;

/// CReLU output: two perspectives concatenated.
pub const CRELU_OUTPUT_SIZE: usize = HIDDEN_SIZE * 2;

/// CReLU clamp maximum for i16 accumulator values.
pub const QA: i32 = 255;

/// Output weight quantization factor.
pub const QB: i32 = 64;

/// Final output scale: divide raw output by this to get centipawns.
pub const OUTPUT_SCALE: i32 = QA * QB;

// Compile-time assertions for SIMD alignment.
const _: () = assert!(
    HIDDEN_SIZE.is_multiple_of(32),
    "HIDDEN_SIZE must be divisible by 32 for AVX-512 SIMD (also satisfies AVX2/SSE4.1)"
);
const _: () = assert!(
    CRELU_OUTPUT_SIZE.is_multiple_of(32),
    "CRELU_OUTPUT_SIZE must be divisible by 32 for AVX-512 SIMD (also satisfies AVX2/SSE4.1)"
);
