# Figure 5: Cerebellar chromatin accessibility and regulatory features

This directory automatically reproduces **5b–e** and performs related motif, chromVAR and sequence-example analyses. The main figure uses the `top1000` configuration.

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

## Additional environment configuration

In addition to the Python dependencies, this workflow requires:

- R 4.3 and `Biostrings`, `Matrix`, `S4Vectors`, `SummarizedExperiment`, `TFBSTools`, `chromVAR` and `motifmatchr`.
- The `fimo` executable from MEME Suite.

Use `environment.yml` at the project root to configure the full environment. If the programs are installed in separate environments, specify:

```bash
export RSCRIPT=/path/to/r-environment/bin/Rscript
export FIMO_BIN=/path/to/meme-environment/bin/fimo
```

The entry point sets `SCATAC_PROJECT_ROOT` to `$ANALYSIS_DATA_DIR/Figure5` by default; this directory should contain `data/`. `--check` also checks the R packages and FIMO.

## Main panels and code

| Panel | Analysis | Code |
| --- | --- | --- |
| 5b | Global UMAP of six species. | `code/src/01_global_umap/plot_global_umap.py` |
| 5c | Astroglial developmental-stage UMAP. | `code/src/02_astroglia_umap/plot_astroglia_stage_umap.py` |
| 5d | Developmental motif-family signals in astroglia. | `s05_prep_family_trajectory.py` and `s05_plot_family_trajectory_standardized.py` in the regulatory workflow. |
| 5e | Concordance between motif enrichment in oligodendrocyte-ranked regions and chromVAR signals. | `s09_oligo_raw_validation.py` in the regulatory workflow. |

## Execution order

1. `code/run_all.sh 00 01 02`: input audit, global UMAP, developmental-stage UMAP and related cross-species alignment summaries.
2. Regulatory stage `prepare`: connect packaged inputs and prepare the ranking set; generate a code-integrity manifest on the first run.
3. `configs` and `audit`: generate configurations and check inputs and developmental windows.
4. `cache`: extract genomic sequences and scan motifs with FIMO.
5. `shared`: generate chromVAR inputs, run chromVAR and generate shared rank summaries and sequence examples.
6. `lock`: lock the shared results for subsequent analyses.
7. `top1000`: run the regulatory analyses and plots corresponding to the main figure.

These regulatory stages are scheduled by `code/src/04_peak_regulatory/run_all.sh`. The top-level entry point executes them sequentially without requiring manual stage-by-stage execution.

## Inputs and outputs

Inputs include the integrated cell H5AD, species- and stage-level accessibility matrices, ranked peaks, reference genome FASTA files and the JASPAR motif database. The bonobo analysis uses the actual packaged file `panpan1.1_sequence_compatible_genomic.fna.gz`.

UMAP and related outputs are written to `$ANALYSIS_RESULTS_DIR/Figure5/`. Regulatory results are accessible through the `04_peak_regulatory` link in that directory; the actual results and caches are stored in:

```text
code/src/04_peak_regulatory/vendor/result_v4/
```

This location satisfies the regulatory workflow's path checks. No existing cache is required for the first run, but sequence scans, chromVAR and random controls can be time-consuming. Changing the results base directory only changes the public link to regulatory results; use a separate Figure5 code copy for a fully independent regulatory run. Do not move caches or the actual results directory while the workflow is running.
