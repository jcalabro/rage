fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 && args[1] == "bench" {
        let depth = args
            .get(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(rage_search::bench::DEFAULT_BENCH_DEPTH);
        rage_search::bench::run_bench(depth);
        return;
    }

    rage_uci::run();
}
