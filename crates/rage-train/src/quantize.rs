use std::fs;
use std::path::Path;

use burn::module::Module;
use burn::record::CompactRecorder;
use burn::tensor::backend::Backend;
use rage_nnue::NnueWeights;
use rage_nnue::arch::{CRELU_OUTPUT_SIZE, HIDDEN_SIZE, QA, QB, TOTAL_INPUT_FEATURES};

use crate::model::{HalfKAv2Config, HalfKAv2Model};

/// Quantize f32 model weights to the rage-nnue integer format.
///
/// - Feature transformer weights: `round(f32 * QA)` → `i16`
/// - Feature transformer biases:  `round(f32 * QA)` → `i16`
/// - Output weights:              `round(f32 * QB)` → `i8`
/// - Output bias:                 `round(f32 * QA * QB)` → `i32`
///
/// `ft_weights` must have exactly `TOTAL_INPUT_FEATURES` rows of `HIDDEN_SIZE` each.
pub fn quantize_weights(
    ft_weights: &[[f32; HIDDEN_SIZE]],
    ft_biases: &[f32; HIDDEN_SIZE],
    output_weights: &[f32; CRELU_OUTPUT_SIZE],
    output_bias: f32,
) -> NnueWeights {
    assert_eq!(ft_weights.len(), TOTAL_INPUT_FEATURES);
    let mut weights = NnueWeights::zeroed();

    // Feature transformer weights: f32 → i16
    for (i, row) in ft_weights.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            let quantized = (val * QA as f32).round() as i32;
            weights.ft_weights[i][j] = quantized.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }
    }

    // Feature transformer biases: f32 → i16
    for (j, &val) in ft_biases.iter().enumerate() {
        let quantized = (val * QA as f32).round() as i32;
        weights.ft_biases[j] = quantized.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }

    // Output weights: f32 → i8
    for (j, &val) in output_weights.iter().enumerate() {
        let quantized = (val * QB as f32).round() as i32;
        weights.output_weights[j] = quantized.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
    }

    // Output bias: f32 → i32
    weights.output_bias = (output_bias * (QA * QB) as f32).round() as i32;

    weights
}

/// Save quantized weights to a .nnue file.
pub fn save_nnue(weights: &NnueWeights, path: &Path) -> std::io::Result<()> {
    let bytes = weights.to_bytes();
    fs::write(path, bytes)
}

/// Extract f32 weights from a trained Burn model and quantize to NnueWeights.
pub fn extract_and_quantize<B: Backend>(model: &HalfKAv2Model<B>) -> NnueWeights {
    // Feature transformer embedding weights: [TOTAL_INPUT_FEATURES, HIDDEN_SIZE]
    let ft_data = model.ft.weight.val().into_data();
    let ft_vec: Vec<f32> = ft_data.to_vec().unwrap_or_default();

    let mut ft_weights = vec![[0.0f32; HIDDEN_SIZE]; TOTAL_INPUT_FEATURES];
    for i in 0..TOTAL_INPUT_FEATURES {
        for j in 0..HIDDEN_SIZE {
            ft_weights[i][j] = ft_vec[i * HIDDEN_SIZE + j];
        }
    }

    // Feature transformer bias: extracted from the ft_bias linear layer.
    // The linear layer has weight [HIDDEN_SIZE, HIDDEN_SIZE] and bias [HIDDEN_SIZE].
    // We only need the bias vector.
    let mut ft_biases = [0.0f32; HIDDEN_SIZE];
    if let Some(ref bias_param) = model.ft_bias.bias {
        let bias_data = bias_param.val().into_data();
        let bias_vec: Vec<f32> = bias_data.to_vec().unwrap_or_default();
        for (j, &val) in bias_vec.iter().enumerate().take(HIDDEN_SIZE) {
            ft_biases[j] = val;
        }
    }

    // Output layer weights: [1, CRELU_OUTPUT_SIZE] → flatten to [CRELU_OUTPUT_SIZE]
    let out_w_data = model.output.weight.val().into_data();
    let out_w_vec: Vec<f32> = out_w_data.to_vec().unwrap_or_default();
    let mut output_weights = [0.0f32; CRELU_OUTPUT_SIZE];
    for (j, &val) in out_w_vec.iter().enumerate().take(CRELU_OUTPUT_SIZE) {
        output_weights[j] = val;
    }

    // Output bias: [1] → scalar
    let output_bias = if let Some(ref bias_param) = model.output.bias {
        let data = bias_param.val().into_data();
        let vec: Vec<f32> = data.to_vec().unwrap_or_default();
        vec.first().copied().unwrap_or(0.0)
    } else {
        0.0
    };

    quantize_weights(&ft_weights, &ft_biases, &output_weights, output_bias)
}

/// Load a Burn model checkpoint and quantize it to a .nnue file.
pub fn load_and_quantize<B: Backend>(
    model_path: &Path,
    output_path: &Path,
    device: &B::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    let model: HalfKAv2Model<B> = HalfKAv2Config::new()
        .init(device)
        .load_file(model_path, &CompactRecorder::new(), device)
        .map_err(|e| format!("Failed to load model checkpoint: {e}"))?;

    let weights = extract_and_quantize(&model);
    save_nnue(&weights, output_path)?;

    let file_size = fs::metadata(output_path)?.len();
    eprintln!(
        "Quantized model saved to {} ({:.2} MB)",
        output_path.display(),
        file_size as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn zero_ft_weights() -> Vec<[f32; HIDDEN_SIZE]> {
        vec![[0.0f32; HIDDEN_SIZE]; TOTAL_INPUT_FEATURES]
    }

    #[test]
    fn quantize_zero_weights() {
        let ft_weights = zero_ft_weights();
        let ft_biases = [0.0f32; HIDDEN_SIZE];
        let output_weights = [0.0f32; CRELU_OUTPUT_SIZE];
        let output_bias = 0.0;

        let weights = quantize_weights(&ft_weights, &ft_biases, &output_weights, output_bias);
        assert_eq!(weights.output_bias, 0);

        for &b in weights.ft_biases.iter() {
            assert_eq!(b, 0);
        }
    }

    #[test]
    fn quantize_known_values() {
        let ft_weights = zero_ft_weights();
        let ft_biases = [0.0f32; HIDDEN_SIZE];
        let output_weights = [0.0f32; CRELU_OUTPUT_SIZE];
        let output_bias = 1.0; // Should become QA * QB = 16320

        let weights = quantize_weights(&ft_weights, &ft_biases, &output_weights, output_bias);
        assert_eq!(weights.output_bias, (QA * QB) as i32);
    }

    #[test]
    fn quantize_clamps_extreme_values() {
        let ft_weights = zero_ft_weights();
        let mut ft_biases = [0.0f32; HIDDEN_SIZE];
        ft_biases[0] = 1000.0; // Very large, should clamp to i16::MAX after scaling

        let output_weights = [0.0f32; CRELU_OUTPUT_SIZE];
        let output_bias = 0.0;

        let weights = quantize_weights(&ft_weights, &ft_biases, &output_weights, output_bias);
        // 1000.0 * 255 = 255000, clamped to i16::MAX = 32767
        assert_eq!(weights.ft_biases[0], i16::MAX);
    }
}
