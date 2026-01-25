use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use surveyhero::txt_writer::md_to_txt;

#[derive(Parser)]
pub struct Cli {
    pub source: PathBuf,

    pub dist: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let dist = cli
        .dist
        .as_ref()
        .map_or(cli.source.parent().unwrap(), |v| v);
    md_to_txt(&cli.source, dist)
}
