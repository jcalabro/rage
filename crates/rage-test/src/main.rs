mod bench_diff;
mod cli;
mod db;
mod elo_cmd;
mod game;
mod manager;
mod opening;
mod sprt;
mod stats;
mod tournament;
mod tui;
mod types;
mod uci;

fn main() {
    cli::run();
}
