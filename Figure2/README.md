# Figure 2: Microglial features and cross-species annotation transfer

This directory automatically reproduces **2b and 2f**.

## Environment and paths

For environment installation, see the [project documentation](../README.md#2-install-the-environment). Public data used for this figure are available from [Zenodo: 10.5281/zenodo.23765892](https://doi.org/10.5281/zenodo.23765892).

Before running from this directory, you can set:

```bash
export ANALYSIS_DATA_DIR=/absolute/path/to/analysis_data
export ANALYSIS_RESULTS_DIR=/absolute/path/to/analysis_results
export PYTHON_BIN="$(command -v python)"
```

`ANALYSIS_DATA_DIR` should directly contain Figure2–6. If these variables are not set, the entry point reads from `analysis_data` beside the code package and writes results to `results` at the code-package root.

## Automatic execution

Run from this directory:

```bash
bash run_all.sh --check
bash run_all.sh --dry-run
bash run_all.sh
```

`--check` checks the data directory and import dependencies; `--dry-run` only displays commands; running without arguments executes the steps in order. Execution stops if any step fails. The script can also be invoked by absolute path from another working directory. Reruns may overwrite files with the same names, so use separate results directories for different runs.

## Workflow

| Panel | Analysis | Entry point |
| --- | --- | --- |
| 2b | Joint expression dotplot of ranked genes in human and mouse microglia. | `code/microglia/plot_human_mouse_mgc_dotplot.py` |
| 2f | Within- and cross-species kNN annotation transfer and macro-F1 evaluation in the nine-species spermatogenesis dataset. | `code/testis/evaluate_spermatogenesis_transfer.py` |

The main entry point runs 2b followed by 2f. `plot_gene_rank_dotplots.py` is an expression-analysis helper module and can also independently read the flat ranking CSV files included in the package.

## Inputs and outputs

- Inputs: `analysis_data/Figure2/microglia/` and `Figure2/testis/testis_integrated.h5ad`.
- Outputs: `$ANALYSIS_RESULTS_DIR/Figure2/microglia/` and `Figure2/testis_transfer/`, including figures, individual evaluation tables and analysis configurations.
