#![allow(clippy::unwrap_used, clippy::expect_used)]

mod data;
mod datagen;
mod loader;
mod model;
mod quantize;
mod train;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use clap::{Parser, Subcommand};
use rage_nnue::{NnueNetwork, NnueWeights};

/// NNUE training pipeline for the Rage chess engine.
#[derive(Parser)]
#[command(name = "rage-train", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate training data via self-play.
    Datagen {
        /// Output file path for training data.
        #[arg(short, long, default_value = "training_data.bin")]
        output: PathBuf,
        /// Number of self-play games.
        #[arg(short = 'n', long, default_value_t = 1000)]
        games: usize,
        /// Search depth per position.
        #[arg(short, long, default_value_t = 8)]
        depth: u32,
        /// Number of random opening moves.
        #[arg(long, default_value_t = 8)]
        random_plies: u32,
        /// Number of threads (0 = all available cores).
        #[arg(short, long, default_value_t = 0)]
        threads: usize,
        /// Path to NNUE weights file for evaluation (omit for PST-only eval).
        #[arg(long)]
        evalfile: Option<PathBuf>,
    },
    /// Train the NNUE network.
    Train {
        /// Input training data file.
        #[arg(short, long)]
        data: PathBuf,
        /// Output directory for model checkpoints.
        #[arg(short, long, default_value = "model")]
        output: PathBuf,
        /// Number of training epochs.
        #[arg(short, long, default_value_t = 20)]
        epochs: usize,
        /// Batch size.
        #[arg(short, long, default_value_t = 16384)]
        batch_size: usize,
        /// Learning rate.
        #[arg(long, default_value_t = 1e-3)]
        learning_rate: f64,
    },
    /// Quantize a trained f32 model to .nnue format.
    Quantize {
        /// Input model checkpoint path (without extension).
        #[arg(short, long)]
        model: PathBuf,
        /// Output .nnue file path.
        #[arg(short, long, default_value = "default.nnue")]
        output: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Datagen {
            output,
            games,
            depth,
            random_plies,
            threads,
            evalfile,
        } => {
            let network = evalfile.map(|path| {
                let weights = NnueWeights::from_file(Path::new(&path)).unwrap_or_else(|e| {
                    panic!("failed to load NNUE weights from {}: {e}", path.display())
                });
                eprintln!("info string loaded NNUE network from {}", path.display());
                Arc::new(NnueNetwork::new(weights))
            });
            let config = datagen::DatagenConfig {
                num_games: games,
                depth,
                random_plies,
                threads,
                network,
                ..Default::default()
            };
            match datagen::generate_data(&config, &output) {
                Ok(n) => println!("Generated {n} training positions"),
                Err(e) => eprintln!("Error generating data: {e}"),
            }
        }
        Command::Train {
            data,
            output,
            epochs,
            batch_size,
            learning_rate,
        } => {
            let config = train::TrainConfig {
                epochs,
                batch_size,
                learning_rate,
                ..Default::default()
            };

            #[cfg(feature = "cuda")]
            let device = burn::tensor::Device::cuda(0);
            #[cfg(not(feature = "cuda"))]
            let device = burn::tensor::Device::flex();

            if let Err(e) = train::train(&config, &data, &output, &device) {
                eprintln!("Training error: {e}");
            }
        }
        Command::Quantize { model, output } => {
            let device = burn::tensor::Device::flex();
            if let Err(e) = quantize::load_and_quantize(&model, &output, &device) {
                eprintln!("Quantization error: {e}");
            }
        }
    }
}
