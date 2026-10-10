use std::path::Path;
use std::time::Instant;

use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{Device, Tensor};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::loader::TrainingDataset;
use crate::model::{HalfKAv2Config, HalfKAv2Model};

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
/// `lambda * MSE(sigmoid(pred), sigmoid(target)) + (1-lambda) * CE(pred_wdl, actual_wdl)`
///
/// - `predictions`: [batch, 1] model output (in internal units)
/// - `target_scores`: [batch, 1] search scores (in centipawns)
/// - `target_wdl`: [batch, 1] game result (0=loss, 0.5=draw, 1=win)
/// - `lambda`: weighting between score MSE and WDL cross-entropy
pub fn compute_loss(
    predictions: Tensor<2>,
    target_scores: Tensor<2>,
    target_wdl: Tensor<2>,
    lambda: f64,
) -> Tensor<1> {
    let device = predictions.device();
    let scale = Tensor::from_floats([[SIGMOID_SCALE as f32]], &device);

    // Sigmoid of predictions and targets (score component)
    let pred_sigmoid = activation_sigmoid(predictions.clone() * scale.clone());
    let target_sigmoid = activation_sigmoid(target_scores * scale);

    // MSE loss on sigmoid-transformed scores
    let diff = pred_sigmoid - target_sigmoid;
    let mse_loss = (diff.clone() * diff).mean();

    // WDL cross-entropy: pred_wdl = sigmoid(prediction), actual = target_wdl
    let pred_wdl = activation_sigmoid(predictions);
    let epsilon = Tensor::from_floats([[1e-7f32]], &device);
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
fn activation_sigmoid(x: Tensor<2>) -> Tensor<2> {
    burn::tensor::activation::sigmoid(x)
}

/// Cosine annealing learning rate schedule.
fn cosine_lr(step: usize, total_steps: usize, lr_max: f64) -> f64 {
    let lr_min = lr_max / 100.0;
    let progress = step as f64 / total_steps.max(1) as f64;
    lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f64::consts::PI * progress).cos())
}

/// Run the full training pipeline.
pub fn train(
    config: &TrainConfig,
    data_path: &Path,
    output_dir: &Path,
    device: &Device,
) -> Result<(), Box<dyn std::error::Error>> {
    // Parameters only track gradients when created on an autodiff device.
    let device = &device.clone().autodiff();

    eprintln!("Loading training data from {}...", data_path.display());
    let mut dataset = TrainingDataset::load(data_path)?;
    eprintln!(
        "Dataset: {} entries, batch size {}, ~{} batches/epoch",
        dataset.len(),
        config.batch_size,
        dataset.len() / config.batch_size
    );

    let model: HalfKAv2Model = HalfKAv2Config::new().init(device);
    let mut optim = AdamConfig::new().init();

    let batches_per_epoch = dataset.len() / config.batch_size;
    let total_steps = config.epochs * batches_per_epoch;

    eprintln!(
        "Training for {} epochs ({} total steps), lr={:.1e}",
        config.epochs, total_steps, config.learning_rate
    );

    std::fs::create_dir_all(output_dir)?;
    let mut rng = StdRng::seed_from_u64(42);
    let mut model = model;
    let mut global_step = 0usize;

    for epoch in 0..config.epochs {
        let epoch_start = Instant::now();
        dataset.shuffle(&mut rng);

        let mut epoch_loss = 0.0f64;
        let mut num_batches = 0usize;

        for batch in dataset.batches(config.batch_size, device) {
            let lr = cosine_lr(global_step, total_steps, config.learning_rate);

            let predictions = model.forward(
                batch.white_features,
                batch.black_features,
                batch.white_count,
                batch.black_count,
                batch.stm,
            );

            let loss = compute_loss(
                predictions,
                batch.target_scores,
                batch.target_wdl,
                config.lambda,
            );

            // Extract loss value before backprop
            let loss_val: f64 = loss
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .map_or(0.0, |v| v.first().copied().unwrap_or(0.0) as f64);

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);

            epoch_loss += loss_val;
            num_batches += 1;
            global_step += 1;

            if num_batches.is_multiple_of(100) {
                let avg = epoch_loss / num_batches as f64;
                eprintln!(
                    "  epoch {}/{} batch {}/{} loss={:.6} lr={:.2e}",
                    epoch + 1,
                    config.epochs,
                    num_batches,
                    batches_per_epoch,
                    avg,
                    lr,
                );
            }
        }

        let elapsed = epoch_start.elapsed();
        let avg_loss = if num_batches > 0 {
            epoch_loss / num_batches as f64
        } else {
            0.0
        };
        eprintln!(
            "Epoch {}/{}: avg_loss={:.6} batches={} time={:.1}s",
            epoch + 1,
            config.epochs,
            avg_loss,
            num_batches,
            elapsed.as_secs_f64(),
        );

        // Save checkpoint
        let checkpoint_path = output_dir.join(format!("epoch_{}", epoch + 1));
        model
            .clone()
            .save_file(&checkpoint_path)
            .map_err(|e| format!("Failed to save checkpoint: {e}"))?;
        eprintln!("  Saved checkpoint to {}", checkpoint_path.display());
    }

    eprintln!("Training complete.");
    Ok(())
}
