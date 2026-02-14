use crate::arch::{CRELU_OUTPUT_SIZE, HIDDEN_SIZE, NUM_KING_BUCKETS, TOTAL_INPUT_FEATURES};

/// Magic bytes identifying a rage NNUE weight file.
const MAGIC: [u8; 4] = *b"RAGE";

/// Current weight file format version.
const VERSION: u32 = 1;

/// The raw NNUE weight data.
pub struct NnueWeights {
    /// Feature transformer weights: [TOTAL_INPUT_FEATURES][HIDDEN_SIZE].
    /// Each row is the weight column for one input feature.
    pub ft_weights: Box<[[i16; HIDDEN_SIZE]; TOTAL_INPUT_FEATURES]>,
    /// Feature transformer biases: [HIDDEN_SIZE].
    pub ft_biases: Box<[i16; HIDDEN_SIZE]>,
    /// Output layer weights: [CRELU_OUTPUT_SIZE].
    /// First HIDDEN_SIZE for STM perspective, next HIDDEN_SIZE for NSTM.
    pub output_weights: Box<[i8; CRELU_OUTPUT_SIZE]>,
    /// Output layer bias (pre-scaled by QA * QB).
    pub output_bias: i32,
}

/// Allocate zeroed ft_weights on the heap without touching the stack.
fn zeroed_ft_weights() -> Box<[[i16; HIDDEN_SIZE]; TOTAL_INPUT_FEATURES]> {
    // SAFETY: An array of zeroed i16 values is a valid initialized state.
    unsafe { Box::new_zeroed().assume_init() }
}

impl NnueWeights {
    /// Generate deterministic pseudo-random weights from a seed.
    /// Useful for testing — the weights have no chess knowledge.
    pub fn random(seed: u64) -> Self {
        let mut rng = SimpleRng(seed);

        let mut ft_weights = zeroed_ft_weights();
        for row in ft_weights.iter_mut() {
            for val in row.iter_mut() {
                *val = (rng.next_u64() % 201) as i16 - 100; // [-100, 100]
            }
        }

        let mut ft_biases = Box::new([0i16; HIDDEN_SIZE]);
        for val in ft_biases.iter_mut() {
            *val = (rng.next_u64() % 101) as i16 - 50; // [-50, 50]
        }

        let mut output_weights = Box::new([0i8; CRELU_OUTPUT_SIZE]);
        for val in output_weights.iter_mut() {
            *val = ((rng.next_u64() % 129) as i32 - 64) as i8; // [-64, 64]
        }

        let output_bias = (rng.next_u64() % 2001) as i32 - 1000;

        Self {
            ft_weights,
            ft_biases,
            output_weights,
            output_bias,
        }
    }

    /// Create a zero-initialized weight set. Useful for testing.
    pub fn zeroed() -> Self {
        Self {
            ft_weights: zeroed_ft_weights(),
            ft_biases: Box::new([0i16; HIDDEN_SIZE]),
            output_weights: Box::new([0i8; CRELU_OUTPUT_SIZE]),
            output_bias: 0,
        }
    }

    /// Serialize weights to bytes in the rage NNUE binary format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let ft_size = TOTAL_INPUT_FEATURES * HIDDEN_SIZE * 2; // i16
        let bias_size = HIDDEN_SIZE * 2; // i16
        let output_size = CRELU_OUTPUT_SIZE; // i8
        let total = 32 + ft_size + bias_size + output_size + 4; // header + data + output_bias
        let mut buf = Vec::with_capacity(total);

        // Header (32 bytes)
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&(HIDDEN_SIZE as u32).to_le_bytes());
        buf.extend_from_slice(&(NUM_KING_BUCKETS as u32).to_le_bytes());
        buf.extend_from_slice(&(TOTAL_INPUT_FEATURES as u32).to_le_bytes());
        // Reserved (12 bytes to pad header to 32)
        buf.extend_from_slice(&[0u8; 12]);

        // Feature transformer weights
        for row in self.ft_weights.iter() {
            for &val in row.iter() {
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }

        // Feature transformer biases
        for &val in self.ft_biases.iter() {
            buf.extend_from_slice(&val.to_le_bytes());
        }

        // Output weights
        for &val in self.output_weights.iter() {
            buf.push(val as u8);
        }

        // Output bias
        buf.extend_from_slice(&self.output_bias.to_le_bytes());

        buf
    }

    /// Deserialize weights from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, WeightError> {
        if data.len() < 32 {
            return Err(WeightError::TruncatedFile);
        }

        // Check magic
        if data[0..4] != MAGIC {
            return Err(WeightError::BadMagic);
        }

        // Check version
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version != VERSION {
            return Err(WeightError::WrongVersion(version));
        }

        // Read architecture params
        let hidden = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let buckets = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
        let total_features = u32::from_le_bytes([data[16], data[17], data[18], data[19]]) as usize;

        if hidden != HIDDEN_SIZE
            || buckets != NUM_KING_BUCKETS
            || total_features != TOTAL_INPUT_FEATURES
        {
            return Err(WeightError::ArchMismatch);
        }

        let ft_size = TOTAL_INPUT_FEATURES * HIDDEN_SIZE * 2;
        let bias_size = HIDDEN_SIZE * 2;
        let output_size = CRELU_OUTPUT_SIZE;
        let expected_len = 32 + ft_size + bias_size + output_size + 4;

        if data.len() < expected_len {
            return Err(WeightError::TruncatedFile);
        }

        let mut offset = 32;

        // Read feature transformer weights
        let mut ft_weights = zeroed_ft_weights();
        for row in ft_weights.iter_mut() {
            for val in row.iter_mut() {
                *val = i16::from_le_bytes([data[offset], data[offset + 1]]);
                offset += 2;
            }
        }

        // Read feature transformer biases
        let mut ft_biases = Box::new([0i16; HIDDEN_SIZE]);
        for val in ft_biases.iter_mut() {
            *val = i16::from_le_bytes([data[offset], data[offset + 1]]);
            offset += 2;
        }

        // Read output weights
        let mut output_weights = Box::new([0i8; CRELU_OUTPUT_SIZE]);
        for val in output_weights.iter_mut() {
            *val = data[offset] as i8;
            offset += 1;
        }

        // Read output bias
        let output_bias = i32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);

        Ok(Self {
            ft_weights,
            ft_biases,
            output_weights,
            output_bias,
        })
    }
}

/// Errors that can occur when loading NNUE weights.
#[derive(Debug)]
pub enum WeightError {
    BadMagic,
    WrongVersion(u32),
    ArchMismatch,
    TruncatedFile,
}

impl std::fmt::Display for WeightError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadMagic => write!(f, "invalid magic bytes (not a rage NNUE file)"),
            Self::WrongVersion(v) => write!(f, "unsupported weight file version: {v}"),
            Self::ArchMismatch => write!(f, "architecture parameters do not match"),
            Self::TruncatedFile => write!(f, "weight file is truncated"),
        }
    }
}

impl std::error::Error for WeightError {}

/// Simple xorshift64 PRNG for deterministic random weights.
struct SimpleRng(u64);

impl SimpleRng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn random_is_deterministic() {
        let w1 = NnueWeights::random(42);
        let w2 = NnueWeights::random(42);

        assert_eq!(
            w1.ft_biases.as_ref(),
            w2.ft_biases.as_ref(),
            "same seed should produce identical biases"
        );
        assert_eq!(
            w1.output_bias, w2.output_bias,
            "same seed should produce identical output bias"
        );
    }

    #[test]
    fn different_seeds_different_weights() {
        let w1 = NnueWeights::random(1);
        let w2 = NnueWeights::random(2);

        assert_ne!(
            w1.ft_biases.as_ref(),
            w2.ft_biases.as_ref(),
            "different seeds should produce different biases"
        );
    }

    #[test]
    fn roundtrip_serialization() {
        let original = NnueWeights::random(777);
        let bytes = original.to_bytes();
        let loaded = NnueWeights::from_bytes(&bytes).unwrap();

        assert_eq!(original.ft_biases.as_ref(), loaded.ft_biases.as_ref());
        assert_eq!(
            original.output_weights.as_ref(),
            loaded.output_weights.as_ref()
        );
        assert_eq!(original.output_bias, loaded.output_bias);

        // Spot-check a few ft_weight rows
        for i in [0, 100, 1000, TOTAL_INPUT_FEATURES - 1] {
            assert_eq!(
                original.ft_weights[i], loaded.ft_weights[i],
                "ft_weights row {i} mismatch"
            );
        }
    }

    #[test]
    fn bad_magic_rejected() {
        let mut bytes = NnueWeights::random(1).to_bytes();
        bytes[0] = b'X';
        let result = NnueWeights::from_bytes(&bytes);
        assert!(matches!(result, Err(WeightError::BadMagic)));
    }

    #[test]
    fn wrong_version_rejected() {
        let mut bytes = NnueWeights::random(1).to_bytes();
        bytes[4] = 99; // bogus version
        bytes[5] = 0;
        bytes[6] = 0;
        bytes[7] = 0;
        let result = NnueWeights::from_bytes(&bytes);
        assert!(matches!(result, Err(WeightError::WrongVersion(99))));
    }

    #[test]
    fn truncated_file_rejected() {
        let bytes = NnueWeights::random(1).to_bytes();
        let result = NnueWeights::from_bytes(&bytes[..31]);
        assert!(matches!(result, Err(WeightError::TruncatedFile)));

        // Also test truncated body
        let result = NnueWeights::from_bytes(&bytes[..100]);
        assert!(matches!(result, Err(WeightError::TruncatedFile)));
    }

    #[test]
    fn file_size_correct() {
        let bytes = NnueWeights::random(1).to_bytes();
        let expected = 32 // header
            + TOTAL_INPUT_FEATURES * HIDDEN_SIZE * 2  // ft_weights (i16)
            + HIDDEN_SIZE * 2                          // ft_biases (i16)
            + CRELU_OUTPUT_SIZE                        // output_weights (i8)
            + 4; // output_bias (i32)
        assert_eq!(bytes.len(), expected, "file size mismatch");
    }
}
