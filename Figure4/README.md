# Figure 4: Spatial annotation transfer, OPC features and regional aging

This directory automatically reproduces **4d, 4e, 4i, 4j, 4k and 4l**. The unified entry point is `run_all.sh`.

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

## Execution order

| Order | Panels | Analysis and entry point |
| --- | --- | --- |
| 1 | 4d | `code/xenium/plot_leiden_celltype_sankey.py`: generate only the Sankey with at least 250 cells per flow, then run `evaluate_leiden_celltype_consistency.py`. |
| 2 | 4e | `code/xenium/reproduce_transfer.py`: mouse-to-human annotation transfer using cosine distance, distance weighting and k=101. |
| 3 | 4i–j | `code/attention_aging/extract_opc_violin_source_data.py` extracts expression and checks sample summaries; `plot_figure4_i_j.py` recomputes OPC rank heatmaps and Olig2/OLIG2 expression plots. |
| 4 | 4k–l | `code/mouse_regional_aging/run_analysis.sh`: rebuild caches, perform regional-aging analysis, run 1,000 baseline permutations and 9,999 full downstream-pipeline permutations, then plot and verify results. |

Individual mice are the statistical units in the regional-aging analysis. Lines across age groups connect group means and do not represent longitudinal trajectories of the same mice. The percentage of cells with high age scores is not a validated senescent-cell fraction.

## Inputs and outputs

Inputs are located under `analysis_data/Figure4/`, in `xenium/`, `aging/` and `attention_aging/`.

The default Figure 4 output directory is `$ANALYSIS_RESULTS_DIR/Figure4`; it can be overridden with `FIGURE4_OUTPUT_DIR`. Its structure is:

```text
results/
├── xenium/
│   ├── sankey_leiden_to_Celltype_min250.*
│   └── reproduced_transfer/
├── attention_aging/
└── mouse_regional_aging/
    ├── cache/
    ├── figures/
    └── tables/
```

Only the min250 Sankey is retained. The regional-aging workflow does not generate `figure_layout.json`, `input_provenance.json`, `sampling.json`, `verification.json` or a Chinese results report. Verification still runs and prints to the terminal.

Full permutation analysis may take substantial time; results with fewer permutations should not replace the main-figure analysis.
