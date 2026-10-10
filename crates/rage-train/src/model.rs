use burn::module::Module;
use burn::nn;
use burn::tensor::activation;
use burn::tensor::{Device, Int, Tensor};

/// Architecture constants matching rage-nnue.
const TOTAL_INPUT_FEATURES: usize = 7680;
const HIDDEN_SIZE: usize = 256;
const CRELU_OUTPUT_SIZE: usize = HIDDEN_SIZE * 2;

/// Maximum number of active features per perspective in a position.
const MAX_ACTIVE: usize = 32;

/// HalfKAv2 NNUE network for training in f32.
///
/// Architecture: sparse features → embedding sum → CReLU → linear → scalar.
/// During training, the feature transformer uses `nn::Embedding` with manual
/// sum reduction (equivalent to EmbeddingBag with mode="sum").
#[derive(Module, Debug)]
pub struct HalfKAv2Model {
    /// Feature transformer embedding table: [TOTAL_INPUT_FEATURES, HIDDEN_SIZE].
    pub ft: nn::Embedding,
    /// Feature transformer bias: [HIDDEN_SIZE] via a linear layer.
    pub ft_bias: nn::Linear,
    /// Output layer: [CRELU_OUTPUT_SIZE] → [1].
    pub output: nn::Linear,
}

/// Configuration for the HalfKAv2 model.
#[derive(burn::config::Config, Debug)]
pub struct HalfKAv2Config {
    #[config(default = 7680)]
    pub total_features: usize,
    #[config(default = 256)]
    pub hidden_size: usize,
}

impl HalfKAv2Config {
    /// Initialize the model with the given device.
    pub fn init(&self, device: &Device) -> HalfKAv2Model {
        let ft = nn::EmbeddingConfig::new(self.total_features, self.hidden_size).init(device);
        // Use a linear layer with input_size = hidden_size for the bias only
        let ft_bias = nn::LinearConfig::new(self.hidden_size, self.hidden_size)
            .with_bias(true)
            .init(device);
        let output = nn::LinearConfig::new(self.hidden_size * 2, 1)
            .with_bias(true)
            .init(device);

        HalfKAv2Model {
            ft,
            ft_bias,
            output,
        }
    }
}

impl HalfKAv2Model {
    /// Forward pass for a batch of positions.
    ///
    /// Arguments:
    /// - `white_features`: [batch, MAX_ACTIVE] active feature indices for White perspective
    /// - `black_features`: [batch, MAX_ACTIVE] active feature indices for Black perspective
    /// - `white_count`: [batch] number of valid features per sample for White
    /// - `black_count`: [batch] number of valid features per sample for Black
    /// - `stm`: [batch] side to move (0=White, 1=Black)
    ///
    /// Returns: [batch, 1] predicted evaluation score.
    pub fn forward(
        &self,
        white_features: Tensor<2, Int>,
        black_features: Tensor<2, Int>,
        white_count: Tensor<1, Int>,
        black_count: Tensor<1, Int>,
        stm: Tensor<1, Int>,
    ) -> Tensor<2> {
        let device = white_features.device();
        let batch_size = white_features.dims()[0];

        // Compute accumulators for both perspectives
        let white_acc = self.accumulate(&white_features, &white_count, &device);
        let black_acc = self.accumulate(&black_features, &black_count, &device);

        // Determine STM/NSTM accumulators based on side to move
        // stm=0 means White is STM, stm=1 means Black is STM
        let stm_float = stm.float().reshape([batch_size, 1]); // [batch, 1]
        let one = Tensor::ones([batch_size, 1], &device);

        // STM acc = white * (1-stm) + black * stm
        let stm_acc = white_acc.clone() * (one.clone() - stm_float.clone())
            + black_acc.clone() * stm_float.clone();
        // NSTM acc = black * (1-stm) + white * stm
        let nstm_acc = black_acc * (one - stm_float.clone()) + white_acc * stm_float;

        // CReLU: clamp to [0, 1]
        let stm_crelu = activation::relu(stm_acc).clamp_max(1.0);
        let nstm_crelu = activation::relu(nstm_acc).clamp_max(1.0);

        // Concatenate [stm, nstm] → [batch, CRELU_OUTPUT_SIZE]
        let combined = Tensor::cat(vec![stm_crelu, nstm_crelu], 1);

        // Output layer
        self.output.forward(combined)
    }

    /// Compute accumulator for one perspective using embedding sum + bias.
    fn accumulate(
        &self,
        features: &Tensor<2, Int>,
        count: &Tensor<1, Int>,
        device: &Device,
    ) -> Tensor<2> {
        let batch_size = features.dims()[0];

        // Look up embeddings: [batch, MAX_ACTIVE, HIDDEN_SIZE]
        let embeddings = self.ft.forward(features.clone());

        // Create mask from counts: [batch, MAX_ACTIVE, 1]
        let max_feat = MAX_ACTIVE;
        let range = Tensor::<1, Int>::arange(0..max_feat as i64, device);
        let range = range.reshape([1, max_feat]); // [1, MAX_ACTIVE]
        let count_expanded = count.clone().reshape([batch_size, 1]); // [batch, 1]
        let mask = range
            .lower(count_expanded)
            .float()
            .reshape([batch_size, max_feat, 1]);

        // Masked sum of embeddings: [batch, MAX_ACTIVE, HIDDEN_SIZE] → [batch, 1, HIDDEN_SIZE] → [batch, HIDDEN_SIZE]
        let masked = embeddings * mask;
        let sum = masked.sum_dim(1).reshape([batch_size, HIDDEN_SIZE]);

        // Add bias (extract just the bias from the linear layer by passing zeros)
        let zeros = Tensor::zeros([batch_size, HIDDEN_SIZE], device);
        let bias = self.ft_bias.forward(zeros); // This gives us just the bias
        sum + bias
    }
}
