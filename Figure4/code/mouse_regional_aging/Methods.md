# Methods

## Inputs and independence of implementation

The implementation reads `analysis_data/Figure4/aging/human_mouse_aging_integrated.h5ad` and `mouse_concated_adata_with_meta.h5ad`. The joint representation in the packaged atlas is stored as `X_HENOSIS`. Derived mouse caches are rebuilt under the output directory (`results/mouse_regional_aging/cache`), leaving packaged inputs unchanged; results default to `review_output/Figure4/results/mouse_regional_aging` and can be redirected with `FIGURE4_OUTPUT_DIR`.

The primary representation is the existing 25-dimensional `obsm['X_HENOSIS']`. Only mouse cells enter downstream directions, thresholds and tests, but the representation itself is a fixed joint human–mouse representation. Model training and representation integration are not repeated. Precise linkage to a particular upstream checkpoint is not established by the filenames; provenance records the actual files and cell alignment used.

## Animals, directions and high-scoring cells

There are 20 mice: 4 young (<6 months), 7 middle-aged (6–<21 months) and 9 older (≥21 months). Oligodendrocytes and microglia are analyzed across CC/ACO, STR and CTX. Individual mice, not cells or regions, are the statistical replicates.

For each lineage and held-out mouse, compute each training animal's mean embedding, then take the normalized older-minus-young difference between equally weighted animal centroids. Project cells from the held-out mouse onto this direction. For each training-young mouse separately, calculate the projection Q90; average these quantiles equally across young reference animals, excluding the held-out mouse. The percentage of held-out cells above threshold is labeled “Cells with high age scores (%)”. This is not a validated senescent-cell fraction. Regional calculations use the same held-out direction but lineage- and region-specific young thresholds. A mouse-region requires at least 30 lineage cells.

## Effects, null distributions and multiplicity

Lineage effects are older-minus-young group-mean percentages. For each glial lineage, define the within-mouse contrast as CC/ACO minus CTX percentage. We report its stage Spearman correlation and its older-minus-young change, the latter in percentage points. Because each region has its own young reference, these contrasts compare relative-to-young percentages, not absolute raw projections.

All 9,999 permutations are newly computed in this directory. Each draw reassigns mouse age-group labels while preserving 4/7/9 group sizes, and the same draw is shared across all lineages. Every held-out direction, young-reference threshold, percentage and final statistic is refitted. To make computation feasible, each mouse-lineage has at most 500 fixed sampled cells and each mouse-region-glial lineage has at most 300 fixed sampled cells. Observed subsample statistics and all null replicates use identical cells. The exact sampled cell IDs are retained in `fixed_permutation_cell_samples.parquet`. RNG seed is 20261027. Parallel workers evaluate lineages independently using identical cached label permutations; no model or statistical parameter is selected using the resulting P values.


Full-cell effects are plotted. P and q values refer to the distinct fixed-subsample observations. For oligodendrocytes, the full-cell older-minus-young regional change is 24.89 pp and the subsample statistic is 23.47 pp. For microglia, these are 23.95 pp and 25.86 pp. The heatmap is descriptive, with no additional tests for CC/ACO–STR or STR–CTX.

## Graphics


Main k uses small translucent individual points and larger age-group mean points joined only within a region. Main l uses blue for oligodendrocytes and orange for microglia, with slight horizontal offsets and connected group means. No line connects different animals as if they were longitudinally observed. Each panel has its own explicit legend. PDF font type is 42 and SVG keeps text editable.

The analysis is exploratory and conditional on the frozen representation. Source mouse sex is unavailable, and unrestricted age-label exchangeability cannot rule out unrecorded confounding. No independent molecular program, cell–cell interaction, causal mechanism or cross-species conservation is demonstrated by these panels.

For the two regional older-minus-young CC/ACO–CTX contrasts, two-sided permutation P values are corrected with the Benjamini–Hochberg procedure across these two comparisons.

The default workflow writes analysis tables and figures without figure_layout.json, input_provenance.json, sampling.json, verification.json or a Chinese report. Validation remains enabled and reports to stdout.
