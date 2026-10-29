# Mouse regional-aging analysis

This module generates individual-level data, regional changes, figures and permutation-test results for panels 4k/l. Running the full figure workflow through the top-level Figure4 `run_all.sh` is recommended.

## Standalone execution

Run from this module's directory:

```bash
export ANALYSIS_DATA_DIR=/absolute/path/to/analysis_data
export FIGURE4_OUTPUT_DIR=/absolute/path/to/analysis_results/Figure4
PYTHON_BIN=python bash run_analysis.sh
```

Outputs are written to `$FIGURE4_OUTPUT_DIR/results/mouse_regional_aging/`, containing `cache/`, `tables/` and `figures/`. When run through the main Figure4 entry point, the default Figure 4 directory is `$ANALYSIS_RESULTS_DIR/Figure4`.

## Analysis order

1. Read the integrated representation and mouse metadata, then rebuild caches in the results directory.
2. Calculate age directions and young-reference thresholds using a leave-one-mouse-out strategy, and generate individual and regional summaries.
3. Run 1,000 baseline permutations and 9,999 full downstream-pipeline permutations.
4. Generate combined figures, separate subpanels and source-data tables, then verify sample correspondence, statistics and figure outputs.

Inputs are the integrated H5AD and mouse H5AD under `analysis_data/Figure4/aging/`. The analysis uses the fixed 25-dimensional `X_HENOSIS` representation without retraining or reintegrating data. Parameters are stored in `config/analysis.json`.

Lines across age groups connect group means only and do not represent longitudinal follow-up of individuals. The percentage of cells with high age scores is not a validated senescent-cell fraction. See `Methods.md` and `Figure4_legend.md` for statistical definitions.

The five previously excluded JSON/Chinese results-report files are not generated. Verification still runs and prints to the terminal.
