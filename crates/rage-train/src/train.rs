use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;

/// Training configuration.
pub struct TrainConfig {
    /// Number of training epochs.
    pub epochs: usize,
    /// Batch size.
    pub batch_size: usize,
    /// Initial learning rate.
    pub learning_rate: f64,
    /// Lambda for score vs WDL loss weighting.
    pub lambda: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 20,
            batch_size: 16384,
            learning_rate: 1e-3,
            lambda: 0.75,
        }
    }
}

/// Sigmoid scaling factor for converting centipawn scores to win probability.
const SIGMOID_SCALE: f64 = 1.0 / 400.0;

/// Compute the combined training loss.
///
/// `λ * MSE(sigmoid(pred), sigmoid(target)) + (1-λ) * CE(pred_wdl, actual_wdl)`
///
/// - `predictions`: [batch, 1] model output (in internal units)
/// - `target_scores`: [batch, 1] search scores (in centipawns)
/// - `target_wdl`: [batch, 1] game result (0=loss, 0.5=draw, 1=win)
/// - `lambda`: weighting between score MSE and WDL cross-entropy
pub fn compute_loss<B: AutodiffBackend>(
    predictions: Tensor<B, 2>,
    target_scores: Tensor<B, 2>,
    target_wdl: Tensor<B, 2>,
    lambda: f64,
) -> Tensor<B, 1> {
    let device = predictions.device();
    let scale = Tensor::from_floats([SIGMOID_SCALE as f32], &device);

    // Sigmoid of predictions and targets (score component)
    let pred_sigmoid = activation_sigmoid(predictions.clone() * scale.clone());
    let target_sigmoid = activation_sigmoid(target_scores * scale);

    // MSE loss on sigmoid-transformed scores
    let diff = pred_sigmoid - target_sigmoid;
    let mse_loss = (diff.clone() * diff).mean();

    // WDL cross-entropy: pred_wdl = sigmoid(prediction), actual = target_wdl
    let pred_wdl = activation_sigmoid(predictions);
    let epsilon = Tensor::from_floats([1e-7f32], &device);
    let one = Tensor::ones_like(&pred_wdl);

    let ce_loss = -(target_wdl.clone() * (pred_wdl.clone() + epsilon.clone()).log()
        + (one.clone() - target_wdl) * (one - pred_wdl + epsilon).log())
    .mean();

    // Combined loss
    let lambda_t = Tensor::from_floats([lambda as f32], &device);
    let one_minus_lambda = Tensor::from_floats([1.0 - lambda as f32], &device);

    (mse_loss * lambda_t + ce_loss * one_minus_lambda).reshape([1])
}

/// Sigmoid activation function.
fn activation_sigmoid<B: AutodiffBackend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    burn::tensor::activation::sigmoid(x)
}
