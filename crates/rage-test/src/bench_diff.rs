//! Bench diff: compare `rage bench` output between two engine binaries.

use std::path::Path;
use std::process::Command;

/// Result of running bench on a single engine.
#[derive(Debug, Clone)]
pub struct BenchOutput {
    pub total_nodes: u64,
    pub elapsed_ms: u64,
    pub nps: u64,
}

/// Comparison of two bench runs.
#[derive(Debug, Clone)]
pub struct BenchDiff {
    pub engine_a: BenchOutput,
    pub engine_b: BenchOutput,
    pub nodes_match: bool,
    pub nps_ratio: f64,
}

/// Run `<engine> bench [depth]` and parse the output.
pub fn run_bench(engine_path: &Path, depth: Option<u32>) -> Result<BenchOutput, String> {
    let mut cmd = Command::new(engine_path);
    cmd.arg("bench");
    if let Some(d) = depth {
        cmd.arg(d.to_string());
    }

    let output = cmd
        .output()
        .map_err(|e| format!("failed to run {}: {e}", engine_path.display()))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("bench exited with {}: {stderr}", output.status));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_bench_output(&stdout)
}

/// Parse bench output, looking for the summary lines:
/// ```text
/// ===========================
/// Total time : 1234 ms
/// Total nodes: 567890
/// Nodes/sec  : 460
/// ```
fn parse_bench_output(output: &str) -> Result<BenchOutput, String> {
    let mut total_nodes: Option<u64> = None;
    let mut elapsed_ms: Option<u64> = None;
    let mut nps: Option<u64> = None;

    for line in output.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Total nodes:") {
            total_nodes = rest.trim().parse().ok();
        } else if let Some(rest) = line.strip_prefix("Total time :") {
            // "1234 ms"
            let rest = rest.trim();
            if let Some(ms_str) = rest.strip_suffix("ms") {
                elapsed_ms = ms_str.trim().parse().ok();
            }
        } else if let Some(rest) = line.strip_prefix("Nodes/sec  :") {
            nps = rest.trim().parse().ok();
        }
    }

    let total_nodes = total_nodes.ok_or("could not find 'Total nodes' in bench output")?;
    let elapsed_ms = elapsed_ms.unwrap_or(0);
    let nps = nps.unwrap_or(0);

    Ok(BenchOutput {
        total_nodes,
        elapsed_ms,
        nps,
    })
}

/// Run bench on both engines and compare.
pub fn run_bench_diff(
    engine_a: &Path,
    engine_b: &Path,
    depth: Option<u32>,
) -> Result<BenchDiff, String> {
    let a = run_bench(engine_a, depth)?;
    let b = run_bench(engine_b, depth)?;

    let nodes_match = a.total_nodes == b.total_nodes;
    let nps_ratio = if b.nps > 0 {
        a.nps as f64 / b.nps as f64
    } else {
        0.0
    };

    Ok(BenchDiff {
        engine_a: a,
        engine_b: b,
        nodes_match,
        nps_ratio,
    })
}

/// Format a bench diff report for display.
pub fn format_bench_diff(diff: &BenchDiff, name_a: &str, name_b: &str) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{name_a}: {} nodes, {} ms, {} nps\n",
        diff.engine_a.total_nodes, diff.engine_a.elapsed_ms, diff.engine_a.nps
    ));
    out.push_str(&format!(
        "{name_b}: {} nodes, {} ms, {} nps\n",
        diff.engine_b.total_nodes, diff.engine_b.elapsed_ms, diff.engine_b.nps
    ));
    out.push('\n');

    if diff.nodes_match {
        out.push_str("Nodes: MATCH (search behavior is identical)\n");
    } else {
        out.push_str(&format!(
            "Nodes: DIFFER ({} vs {}) — search behavior changed\n",
            diff.engine_a.total_nodes, diff.engine_b.total_nodes
        ));
    }

    out.push_str(&format!("NPS ratio: {:.2}x\n", diff.nps_ratio));

    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_bench_output() {
        let output = r#"Position  1/20: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 depth 13  nodes 12345
===========================
Total time : 1234 ms
Total nodes: 567890
Nodes/sec  : 460
"#;
        let result = parse_bench_output(output).unwrap();
        assert_eq!(result.total_nodes, 567890);
        assert_eq!(result.elapsed_ms, 1234);
        assert_eq!(result.nps, 460);
    }

    #[test]
    fn test_parse_bench_output_missing_nodes() {
        let output = "some random output\n";
        assert!(parse_bench_output(output).is_err());
    }
}
