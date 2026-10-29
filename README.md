<p align="center">
  <img src="Figure1\logo.png" alt="Henosis" width="360">
</p>

# HENOSIS unifies single-cell landscapes across species and modalities through sequence-informed integration

## Main-figure analysis code

This repository organizes the downstream analyses for Figures 1–6 by biological question and execution step. Preprocessing, statistical analysis and plotting remain separate, and each figure has a `run_all.sh` entry point that executes the included steps in dependency order.

The workflows start from deposited representations, feature rankings and assay matrices. They do not repeat HENOSIS model training. All deposited HENOSIS representations were generated through unsupervised integration.

| Figure                       | Analysis                                                                                                              | Included main panels                                                   |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| [Figure 1](Figure1/README.md) | Model architecture and applications.                                                                                  | 1a–b are diagrams; no automatic drawing or numerical analysis script. |
| [Figure 2](Figure2/README.md) | Microglial ranked-gene expression and nine-species spermatogenesis annotation transfer.                               | 2b, 2f                                                                 |
| [Figure 3](Figure3/README.md) | Fifteen-species heart phylogeny, UMAP, cell-class conservation, matching advantage, orthology and expression support. | 3a–h                                                                  |
| [Figure 4](Figure4/README.md) | Xenium cell-type transfer, OPC gene rankings/expression and regional age-associated glial changes.                    | 4d–e, 4i–l                                                           |
| [Figure 5](Figure5/README.md) | Cerebellar scATAC UMAPs, developmental motif-family signals and motif/chromVAR concordance.                           | 5b–e                                                                  |
| [Figure 6](Figure6/README.md) | Within- and cross-species RNA–ATAC integration and gene/peak interpretation.                                         | 6b–i                                                                  |

Figure-specific READMEs are provided in English. The included workflows also produce some related extended analyses. Panels without packaged entry points are listed under [Scope](#scope).

## 1. Download and organize the data

Analysis data are deposited on Zenodo:

**[HENOSIS analysis data — DOI: 10.5281/zenodo.23765892](https://doi.org/10.5281/zenodo.23765892)**

Download the figure-specific archives and extract them into a common `analysis_data` directory. The supplied archives are named `Figure2.tar.gz` through `Figure6.tar.gz`; each already contains its top-level figure directory.

Recommended layout:

```text
project/
├── analysis_code/                 # this repository
│   ├── README.md
│   ├── requirements.txt
│   ├── environment.yml
│   ├── Figure1/
│   ├── Figure2/
│   ├── Figure3/
│   ├── Figure4/
│   ├── Figure5/
│   └── Figure6/
└── analysis_data/
    ├── Figure2/
    ├── Figure3/
    ├── Figure4/
    ├── Figure5/
    └── Figure6/
```

From the repository directory, for example:

```bash
mkdir -p ../analysis_data
for figure in Figure{2..6}; do
    tar -xzf "/path/to/downloads/${figure}.tar.gz" -C ../analysis_data
done
```

Do not add a second `Figure2/` wrapper around the directory already contained in the archive. If your repository or data directory has a different location, set the environment variables below explicitly.

## 2. Install the environment

The analysis scripts use Python 3.11. Figure 5 additionally requires R/Bioconductor and the FIMO executable from MEME Suite. The downstream workflows do not require a GPU or CUDA.

### Full environment, including Figure 5

Run from the repository directory:

```bash
conda env create -f environment.yml
conda activate henosis-analysis
```

[environment.yml](environment.yml) includes Python, R 4.3.3, the Bioconductor 3.18 package series used for chromVAR analysis, and MEME 5.5.9. Python packages are installed from [requirements.txt](requirements.txt).

The R dependencies include `Biostrings`, `Matrix`, `S4Vectors`, `SummarizedExperiment`, `TFBSTools`, `chromVAR` and `motifmatchr`.

### Python-only environment for Figures 2, 3, 4 and 6

```bash
conda create -n henosis-analysis-python python=3.11 pip
conda activate henosis-analysis-python
python -m pip install -r requirements.txt
```

Figure 6 notebooks are executed directly by the supplied Python runner; starting Jupyter or configuring a notebook kernel is unnecessary. Figure 4 Sankey export uses the pinned Plotly/Kaleido versions in `requirements.txt`.

The pinned Python versions match the locally tested environment. Figure-specific dependency checks passed using the installed Python, R and FIMO components; creation of a fresh combined Conda environment has not yet been independently validated.

## 3. Configure input and output paths

From the repository directory:

```bash
export ANALYSIS_DATA_DIR="$(cd ../analysis_data && pwd)"
export ANALYSIS_RESULTS_DIR="$(pwd)/../analysis_results"
export PYTHON_BIN="$(command -v python)"
```

| Variable                 | Meaning                                                         | Default                                  |
| ------------------------ | --------------------------------------------------------------- | ---------------------------------------- |
| `ANALYSIS_DATA_DIR`    | Directory containing the extracted Figure2–6 data directories. | `analysis_data` beside the repository. |
| `ANALYSIS_RESULTS_DIR` | Base directory for regenerated results.                         | `results/` in the repository.          |
| `PYTHON_BIN`           | Python interpreter.                                             | `python`                               |
| `FIGURE4_OUTPUT_DIR`   | Optional Figure 4 output directory override.                    | `$ANALYSIS_RESULTS_DIR/Figure4`        |
| `SCATAC_PROJECT_ROOT`  | Figure 5 data root, containing `data/`.                       | `$ANALYSIS_DATA_DIR/Figure5`           |
| `RSCRIPT`              | Rscript executable for Figure 5.                                | `Rscript`                              |
| `FIMO_BIN`             | FIMO executable for Figure 5.                                   | `fimo`                                 |

If Python, R and FIMO are installed in separate environments, specify their executables:

```bash
export PYTHON_BIN=/path/to/python-environment/bin/python
export RSCRIPT=/path/to/r-environment/bin/Rscript
export FIMO_BIN=/path/to/meme-environment/bin/fimo
```

Keep the deposited input files unchanged. Use a different results directory for each run to preserve earlier outputs. Figure 5 has an additional cache-location constraint described below.

## 4. Check the setup

Every figure entry point supports:

```bash
bash Figure4/run_all.sh --check
bash Figure4/run_all.sh --dry-run
```

- `--check` checks the figure data directory and Python imports; Figure 5 also checks its R packages and FIMO. It does not validate every input file or run the analyses.
- `--dry-run` prints the commands without executing them or checking installed dependencies.
- Running without arguments executes the included workflow and stops at the first failed step.

The scripts locate their own directories, so they can also be invoked by absolute path from another working directory.

## 5. Reproduce the figures

### Figure 1: architecture and application diagrams

```bash
bash Figure1/run_all.sh
```

The entry point prints an explanation and exits. Figure 1a–b are diagrams and have no automatic drawing or statistical workflow in this release.

### Figure 2: microglial features and annotation transfer

```bash
bash Figure2/run_all.sh --check
bash Figure2/run_all.sh
```

Execution order:

1. Read human and mouse microglial rankings and expression matrices, then draw the joint ranked-gene dotplot (2b).
2. Evaluate within- and cross-species kNN annotation transfer in the nine-species testis dataset and write macro-F1 results (2f).

Outputs: `$ANALYSIS_RESULTS_DIR/Figure2/microglia/` and `Figure2/testis_transfer/`.

### Figure 3: fifteen-species heart analysis

```bash
bash Figure3/run_all.sh --check
bash Figure3/run_all.sh
```

Execution order:

1. Draw the divergence-time reference tree and unsupervised heart-cell UMAP (3a–b).
2. Recompute ranked-gene expression support, the four-species cardiomyocyte display and related expression examples (3h and associated analyses).
3. Calculate cell-class centroids, cross-species cosine similarities and cardiomyocyte heatmaps (3c, 3f).
4. Summarize species-pair groups and cell-class matching advantage (3c–e).
5. Combine human-referenced orthology annotations with source-expression support (3g).

The expression-display steps depend on tables written earlier in this workflow. Only the unsupervised heart representation is used.

Outputs: `$ANALYSIS_RESULTS_DIR/Figure3/`.

### Figure 4: spatial transfer and regional aging

```bash
bash Figure4/run_all.sh --check
bash Figure4/run_all.sh
```

Execution order:

1. Generate the Leiden-to-cell-type Sankey using a minimum of 250 cells per flow and calculate the corresponding consistency statistics (4d).
2. Run mouse-to-human label transfer using distance-weighted cosine kNN with `k=101` (4e).
3. Extract OPC expression inputs, validate sample summaries and recompute attention-ranking heatmaps and Olig2/OLIG2 expression plots (4i–j).
4. Run mouse regional-aging analysis, the 1,000-permutation baseline, 9,999 full downstream-pipeline permutations, plotting and verification (4k–l).

Outputs: `$ANALYSIS_RESULTS_DIR/Figure4/results/xenium/`, `results/attention_aging/` and `results/mouse_regional_aging/`. Only the min250 Sankey is generated. Permutation analysis may take substantial time.

### Figure 5: cerebellar chromatin and regulatory features

```bash
bash Figure5/run_all.sh --check
bash Figure5/run_all.sh
```

Execution order:

1. Audit inputs, draw the global UMAP and astroglial developmental-stage UMAPs, and calculate related alignment summaries (5b–c).
2. Prepare the regulatory workflow, derive its configurations and audit inputs.
3. Extract genomic sequences and scan motifs with FIMO to build the cache.
4. Generate chromVAR inputs, run chromVAR, and generate shared rank/sequence summaries.
5. Lock shared results and execute the `top1000` analysis for developmental motif-family signals and oligodendrocyte concordance (5d–e), with related regulatory outputs.

The bonobo genome configuration uses the deposited `panpan1.1_sequence_compatible_genomic.fna.gz`. No previous-run FIMO cache is required. Cold-cache sequence scans and chromVAR can be time-consuming.

UMAP outputs are written under `$ANALYSIS_RESULTS_DIR/Figure5/`. Regulatory outputs are exposed at `Figure5/04_peak_regulatory`, which links to the actual result tree inside:

```text
Figure5/code/src/04_peak_regulatory/vendor/result_v4/
```

The internal location satisfies the regulatory workflow's path checks. Changing `ANALYSIS_RESULTS_DIR` changes the public link but does not create a new regulatory cache. Use a separate Figure5 code copy for an independent regulatory run. Do not move the internal result tree while the workflow is running.

### Figure 6: PBMC and human–macaque RNA–ATAC analysis

```bash
bash Figure6/run_all.sh --check
bash Figure6/run_all.sh
```

Execution order:

1. Compute PBMC B-cell gene-score quartiles, RNA expression and ATAC metagene profiles (6b–c).
2. Draw the B-cell ranked-gene expression dotplot (6d).
3. Validate ranked peaks and draw peak-quartile/accessibility summaries (6e).
4. Draw the unsupervised human–macaque integration UMAP (6f).
5. Map human genes to macaque orthologs, then analyze M1donor ASC RNA expression and M1mac3 ASC accessibility (6g–h).
6. Draw the M1donor ODC ranked-gene expression dotplot (6i).

Outputs: `$ANALYSIS_RESULTS_DIR/Figure6/pbmc/` and `Figure6/brain/`. Notebook files remain free of stored execution outputs. Batch progress and failures are printed to the terminal; batch-summary CSV files are not generated.

## 6. Run all included workflows

After configuring the environment and paths, run from the repository directory:

```bash
(
    set -e
    for figure in Figure{1..6}; do
        bash "$figure/run_all.sh"
    done
)
```

To save a terminal log for one figure:

```bash
bash Figure4/run_all.sh > figure4.log 2>&1
```

Existing files with the same names can be overwritten by a rerun.

## Scope

This release includes entry points for 28 main-figure panels and two Figure 1 diagrams. Main panels without packaged analysis/drawing entry points are **2a, 2c–e, 4a–c, 4f–h, 5a and 6a**. Availability of an integrated H5AD alone does not imply that its corresponding panel or benchmark is implemented here.

Successful execution establishes that the included analysis completed; exact numerical and visual agreement with every manuscript panel requires a separate comparison. Full model training and all alternative-method benchmarks are outside these downstream workflows.

## Data and citation
