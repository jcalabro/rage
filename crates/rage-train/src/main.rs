#![allow(clippy::unwrap_used, clippy::expect_used)]

mod data;
mod datagen;
mod model;
mod quantize;
mod train;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

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
        /// Input model directory.
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
        } => {
            let config = datagen::DatagenConfig {
                num_games: games,
                depth,
                random_plies,
                ..Default::default()
            };
            match datagen::generate_data(&config, &output) {
                Ok(n) => println!("Generated {n} training positions"),
                Err(e) => eprintln!("Error generating data: {e}"),
            }
        }
        Command::Train {
            data: _data,
            output: _output,
            epochs: _epochs,
            batch_size: _batch_size,
            learning_rate: _lr,
        } => {
            // TODO: Implement training loop with Burn
            eprintln!("Training is not yet implemented. Use datagen and quantize.");
        }
        Command::Quantize {
            model: _model,
            output: _output,
        } => {
            // TODO: Load Burn model and quantize
            eprintln!("Quantization from Burn model is not yet implemented.");
            eprintln!("Use the quantize module API directly for programmatic quantization.");
        }
    }
}
