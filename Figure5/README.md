# Figure 5 — cerebellar scATAC analysis

The pipeline under `code/src/` retains separate stages for input checks, peak ranking, motif enrichment, chromVAR validation and plotting. `code/run_all.sh` is the workflow entry point. The mapped main panels use the `top1000` configuration.

| Panel | Entry point | Analysis |
|---|---|---|
| 5d | `code/src/04_peak_regulatory/vendor/steps/s05_prep_family_trajectory.py`; `s05_plot_family_trajectory_standardized.py` | Astroglia motif-family trajectory. |
| 5e | `code/src/04_peak_regulatory/vendor/steps/s09_oligo_raw_validation.py` | Oligodendrocyte motif and chromVAR concordance. |

The pipeline also has `top500` and `top2000` configurations. Genome FASTA, JASPAR matrices, peak rankings and scATAC matrices are external inputs. Configure paths and dependencies before execution.
