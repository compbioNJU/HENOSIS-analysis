# Figure 6 — PBMC and human–macaque RNA–ATAC

| Panel | Entry notebook or script | Analysis |
|---|---|---|
| 6b–c | `code/pbmc/analyze_b_cell_gene_peak_profiles.ipynb` | B-cell gene ranking, metagene profile and RNA expression. |
| 6d | `code/pbmc/plot_b_cell_ranked_gene_expression.ipynb` | Ranked-gene expression dotplot. |
| 6e | `code/pbmc/analyze_b_cell_peak_validation.ipynb`; `code/pbmc/plot_peak_quartiles.py` | Peak accessibility, DAR enrichment and labeled quartiles. |
| 6g–h | `code/cross_species/map_human_macaque_orthologs.ipynb`; `code/cross_species/analyze_macaque_gene_peak_accessibility.ipynb` | Gene mapping and paired gene–peak analysis. |
| 6i | `code/cross_species/plot_asc_ranked_gene_expression.ipynb` | ASC ranked-gene expression dotplot. |

Each notebook appears once and has no saved outputs. The 6b/c notebook copy uses `N_GENE_QUANTILES = 4` for the four-quartile main panel. The 6g/h sample combination must likewise be checked against the figure being prepared. Notebook paths must be configured before execution.
