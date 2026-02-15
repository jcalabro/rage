use std::sync::Arc;

use rage_nnue::{NnueNetwork, NnueWeights};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse --evalfile flag (can appear anywhere before subcommands)
    let network = parse_evalfile(&args);
    let syzygy_path = parse_syzygypath(&args);

    // Find the first positional argument (skipping flags and their values)
    let positional = positional_args(&args);

    if positional.first().map(|s| s.as_str()) == Some("bench") {
        let depth = positional
            .get(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(rage_search::bench::DEFAULT_BENCH_DEPTH);
        rage_search::bench::run_bench(depth, network);
        return;
    }

    rage_uci::run(network, syzygy_path);
}

/// Parse `--evalfile <path>` from the command line arguments.
/// Returns the loaded network or None.
fn parse_evalfile(args: &[String]) -> Option<Arc<NnueNetwork>> {
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--evalfile" {
            if let Some(path) = args.get(i + 1) {
                match NnueWeights::from_file(std::path::Path::new(path)) {
                    Ok(weights) => {
                        eprintln!("info string loaded NNUE network from {path}");
                        return Some(Arc::new(NnueNetwork::new(weights)));
                    }
                    Err(e) => {
                        eprintln!("info string failed to load NNUE network: {e}");
                        return None;
                    }
                }
            }
            eprintln!("info string --evalfile requires a path argument");
            return None;
        }
        i += 1;
    }
    None
}

/// Parse `--syzygypath <path>` from the command line arguments.
fn parse_syzygypath(args: &[String]) -> Option<String> {
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--syzygypath" {
            return args.get(i + 1).cloned();
        }
        i += 1;
    }
    None
}

/// Collect positional arguments, skipping known flags and their values.
fn positional_args(args: &[String]) -> Vec<String> {
    let mut result = Vec::new();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--evalfile" || args[i] == "--syzygypath" {
            i += 2; // skip flag and value
            continue;
        }
        result.push(args[i].clone());
        i += 1;
    }
    result
}
