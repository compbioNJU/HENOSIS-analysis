# Figure 6: PBMC and human–rhesus macaque RNA–ATAC analyses

This directory automatically reproduces **6b–i**. Panels 6g/h use astrocytes (ASC), whereas 6i uses oligodendrocytes (ODC).

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
| 1 | 6b–c | `code/pbmc/analyze_b_cell_gene_peak_profiles.ipynb`: B-cell gene-score quartiles, RNA expression and ATAC metagene profiles. |
| 2 | 6d | `code/pbmc/plot_b_cell_ranked_gene_expression.ipynb`: expression dotplot of B-cell ranked genes. |
| 3 | 6e | `code/pbmc/analyze_b_cell_peak_validation.ipynb` and `plot_peak_quartiles.py`: ranked-peak accessibility, DAR enrichment and grouped displays. |
| 4 | 6f | `code/cross_species/plot_unsupervised_umap.py`: unsupervised integration UMAP of human RNA and rhesus macaque ATAC. |
| 5 | 6g–h | Run `map_human_macaque_orthologs.ipynb` followed by `analyze_macaque_gene_peak_accessibility.ipynb`: M1donor ASC RNA expression and M1mac3 ASC gene-region accessibility. |
| 6 | 6i | `code/cross_species/plot_odc_ranked_gene_expression.ipynb`: expression dotplot of ranked genes in human M1donor ODC. |

Cross-species files are located in `code/cross_species/`. The main entry point executes notebook cells sequentially through `code/run_notebook.py`; Jupyter does not need to be started, and notebook execution outputs are not saved. Panels 6f–i are scheduled by `code/cross_species/run_fghi.py`.

## Inputs and outputs

- PBMC inputs are located in `analysis_data/Figure6/pbmc/`.
- Human–rhesus macaque inputs are located in `analysis_data/Figure6/brain/`, including the unsupervised integrated H5AD, assay matrices, rankings, orthology tables and genomic annotations.
- Results are written to `$ANALYSIS_RESULTS_DIR/Figure6/pbmc/` and `Figure6/brain/`.

Cross-species feature analyses use `motc` rankings from the unsupervised model. Panels 6g/h analyze the complete ATAC peak set; orthology mapping is used for post-training comparisons. The ODC expression plots preserve the original detailed cell-type labels; OPC also remains separate in the merged display.
