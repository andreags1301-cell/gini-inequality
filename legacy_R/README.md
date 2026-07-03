# Legacy R implementation

`inequality_analysis_send.R` is the **original** analysis, written in R and used
for the seminar paper. It is kept here as the reference implementation.

The Python pipeline in [`../src/`](../src) reproduces its results to three
decimals (see the main [README](../README.md#python--r-replication)). New work
should go in the Python code; this script is preserved for provenance only.

## Running it

Requires R with: `plm`, `sandwich`, `lmtest`, `tidyverse`, `ggrepel`,
`patchwork`, `scales` (the script installs missing packages via `pacman`).

The script reads the data through an interactive file picker
(`read.csv(file.choose())`) — select `../data/gini_db.csv` when prompted.
