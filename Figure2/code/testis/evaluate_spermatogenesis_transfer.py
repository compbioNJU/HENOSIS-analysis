#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-species KNN transfer-learning benchmark

Experimental design
-------------------
1. adata.obs["sp"]:
   species label.

2. adata.obs["Celltype"]:
   cell-type label.

3. adata.obsm[EMBEDDING_KEY]:
   cell embedding used for KNN classification.

4. Matrix layout:
   rows = source species (the training species)
   columns = target species (the test species)

5. Off-diagonal cells:
   train on the source species, test on the target species.

6. Diagonal cells:
   a stratified train/test split within one species, so that training and test
   cells never overlap.

7. For every source-target species pair, only cell types that both species
   share and that reach the minimum cell count are evaluated.

8. Outputs:
   - one-vs-rest F1 per cell type
   - one-vs-rest accuracy per cell type
   - precision, recall and support per cell type
   - overall multiclass macro-F1
   - overall multiclass accuracy
   - mean F1 across cell types
   - mean binary accuracy across cell types
   - PNG and text-editable PDF heatmaps
"""

import os
import re
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. User parameters: edit these
# ============================================================

DATA_ROOT = Path(os.environ.get("ANALYSIS_DATA_DIR", Path(__file__).resolve().parents[4] / "analysis_data"))
H5AD_PATH = str(DATA_ROOT / "Figure2/testis/testis_integrated.h5ad")

SAVE_DIR = str(Path(__file__).resolve().parents[2] / "results/testis_transfer")

# Must be given explicitly; no automatic search
# For example:
# EMBEDDING_KEY = "X_emb"
# EMBEDDING_KEY = "X_pca"
# EMBEDDING_KEY = "X_umap"
EMBEDDING_KEY = "X_HENOSIS"

SPECIES_KEY = "Species"
CELLTYPE_KEY = "Celltype"

# KNN parameters
N_NEIGHBORS = 150
KNN_WEIGHTS = "distance"
KNN_METRIC = "euclidean"
N_JOBS = -1

# Maximum number of cells sampled per species and cell type
MAX_TRAIN_PER_CELLTYPE = 3000
MAX_TEST_PER_CELLTYPE = 3000

# Minimum cells of a cell type in each species for it to be evaluated
MIN_CELLS_PER_CELLTYPE = 20

# Test-set fraction for the same-species diagonal
DIAGONAL_TEST_SIZE = 0.2

# Whether to L2-normalize each cell vector in the embedding
L2_NORMALIZE_EMBEDDING = True

RANDOM_SEED = 42

# ============================================================
# Species names and phylogenetic order
# ============================================================
# All internal computation uses the scientific names in adata.obs["sp"];
# CSV row/column names and heatmap labels are converted to common names.
SPECIES_COMMON_NAME = {
    "Homo_sapiens": "Human",
    "Pan_paniscus": "Bonobo",
    "Pan_troglodytes": "Chimpanzee",
    "Gorilla_gorilla": "Gorilla",
    "Nomascus_leucogenys": "Gibbon",
    "Macaca_mulatta": "Rhesus",
    "Callithrix_jacchus": "Marmoset",
    "Mus_musculus": "Mouse",
    "Monodelphis_domestica": "Opossum",
    "Ornithorhynchus_anatinus": "Platypus",
    "Gallus_gallus": "Chicken",
}

# Ordered by phylogenetic proximity to humans, from closest to farthest.
# Bonobo and chimpanzee are sister taxa, so their order is a display choice only
# and does not imply a definite difference in their true distance to humans.
SPECIES_ORDER = [
    "Homo_sapiens",
    "Pan_paniscus",
    "Pan_troglodytes",
    "Gorilla_gorilla",
    "Nomascus_leucogenys",
    "Macaca_mulatta",
    "Callithrix_jacchus",
    "Mus_musculus",
    "Monodelphis_domestica",
    "Ornithorhynchus_anatinus",
    "Gallus_gallus",
]

# Optional: restrict the analysis to specific cell types
# Set to None to analyse every valid cell type
CELLTYPE_ORDER = None

# Example:
# CELLTYPE_ORDER = [
#     "Spermatogonium",
#     "Male germ line stem cell",
#     "Spermatocyte",
#     "Early spermatid",
#     "Late spermatid",
#     "Sertoli cell",
#     "Somatic cell",
# ]

# Whether to drop labels such as unknown or nan
FILTER_INVALID_LABELS = True

# Plotting parameters
DPI = 300
FIGSIZE_PER_SPECIES = 0.85
ANNOTATION_FONT_SIZE = 7
TICK_FONT_SIZE = 9

# Keep text editable in PDFs
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 2. Shared helpers
# ============================================================

def sanitize_filename(text):
    """Convert a cell-type name into a filesystem-safe name."""
    text = str(text)
    text = re.sub(r"[^\w.\-]+", "_", text)
    text = text.strip("_")

    if not text:
        text = "unknown"

    return text


def species_to_common_name(species):
    """Map a scientific name to its common name; keep the original if undefined."""
    return SPECIES_COMMON_NAME.get(str(species), str(species))


def rename_species_matrix_to_common_names(matrix):
    """Convert the row/column scientific names of a source x target matrix to common names."""
    renamed = matrix.copy()
    renamed.index = [
        species_to_common_name(species)
        for species in renamed.index
    ]
    renamed.columns = [
        species_to_common_name(species)
        for species in renamed.columns
    ]
    return renamed


def validate_adata(adata):
    """Check that the AnnData object has the required fields."""

    missing_obs_columns = [
        column
        for column in [SPECIES_KEY, CELLTYPE_KEY]
        if column not in adata.obs.columns
    ]

    if missing_obs_columns:
        raise KeyError(
            "adata.obs is missing required columns: "
            f"{missing_obs_columns}\n"
            f"current obs columns: {adata.obs.columns.tolist()}"
        )

    if EMBEDDING_KEY not in adata.obsm:
        raise KeyError(
            f"{EMBEDDING_KEY!r} is not present in adata.obsm.\n"
            f"current adata.obsm keys: {list(adata.obsm.keys())}"
        )

    embedding = np.asarray(adata.obsm[EMBEDDING_KEY])

    if embedding.ndim != 2:
        raise ValueError(
            f"adata.obsm[{EMBEDDING_KEY!r}] must be a 2-D matrix, "
            f"but its shape is {embedding.shape}"
        )

    if embedding.shape[0] != adata.n_obs:
        raise ValueError(
            f"Embedding row count does not match the number of cells:\n"
            f"embedding.shape[0] = {embedding.shape[0]}\n"
            f"adata.n_obs = {adata.n_obs}"
        )

    if not np.isfinite(embedding).all():
        raise ValueError(
            f"adata.obsm[{EMBEDDING_KEY!r}] contains NaN or Inf."
        )


def clean_metadata_and_embedding(adata):
    """
    Clean the species and cell-type labels and return the cleaned obs and embedding.
    """

    obs = adata.obs[[SPECIES_KEY, CELLTYPE_KEY]].copy()

    obs[SPECIES_KEY] = (
        obs[SPECIES_KEY]
        .astype(str)
        .str.strip()
    )

    obs[CELLTYPE_KEY] = (
        obs[CELLTYPE_KEY]
        .astype(str)
        .str.strip()
    )

    valid_mask = np.ones(adata.n_obs, dtype=bool)

    if FILTER_INVALID_LABELS:
        invalid_labels = {
            "",
            "nan",
            "NaN",
            "none",
            "None",
            "unknown",
            "Unknown",
            "UNKNOWN",
        }

        valid_mask &= ~obs[SPECIES_KEY].isin(invalid_labels).to_numpy()
        valid_mask &= ~obs[CELLTYPE_KEY].isin(invalid_labels).to_numpy()

    valid_positions = np.flatnonzero(valid_mask)

    obs = obs.iloc[valid_positions].copy()
    obs.reset_index(drop=True, inplace=True)

    embedding = np.asarray(
        adata.obsm[EMBEDDING_KEY]
    )[valid_positions]

    embedding = embedding.astype(
        np.float32,
        copy=False,
    )

    if L2_NORMALIZE_EMBEDDING:
        embedding = normalize(
            embedding,
            norm="l2",
            axis=1,
            copy=True,
        ).astype(np.float32)

    return obs, embedding


def determine_species_order(obs):
    """Determine the species order."""

    available_species = set(
        obs[SPECIES_KEY].unique().tolist()
    )

    if SPECIES_ORDER is None:
        return sorted(available_species)

    missing_species = [
        species
        for species in SPECIES_ORDER
        if species not in available_species
    ]

    if missing_species:
        print(
            "Warning: these requested species are absent from the data and will be ignored:",
            missing_species,
        )

    species_list = [
        species
        for species in SPECIES_ORDER
        if species in available_species
    ]

    if len(species_list) < 2:
        raise ValueError(
            "Fewer than two species are available for analysis."
        )

    return species_list


def determine_celltype_order(obs):
    """Determine the cell types to analyse."""

    available_celltypes = set(
        obs[CELLTYPE_KEY].unique().tolist()
    )

    if CELLTYPE_ORDER is None:
        return sorted(available_celltypes)

    missing_celltypes = [
        celltype
        for celltype in CELLTYPE_ORDER
        if celltype not in available_celltypes
    ]

    if missing_celltypes:
        print(
            "Warning: these requested cell types are absent from the data and will be ignored:",
            missing_celltypes,
        )

    celltype_list = [
        celltype
        for celltype in CELLTYPE_ORDER
        if celltype in available_celltypes
    ]

    if len(celltype_list) < 2:
        raise ValueError(
            "Fewer than two cell types are available for analysis."
        )

    return celltype_list


def get_celltype_counts(obs, species):
    """Count the cells of each cell type in one species."""

    return (
        obs.loc[
            obs[SPECIES_KEY].eq(species),
            CELLTYPE_KEY,
        ]
        .value_counts()
    )


def get_common_valid_celltypes(
    obs,
    source_species,
    target_species,
    allowed_celltypes,
):
    """
    Cell types present in both source and target that reach the minimum cell count.
    """

    source_counts = get_celltype_counts(
        obs,
        source_species,
    )

    target_counts = get_celltype_counts(
        obs,
        target_species,
    )

    valid_celltypes = []

    for celltype in allowed_celltypes:
        source_n = int(
            source_counts.get(celltype, 0)
        )

        target_n = int(
            target_counts.get(celltype, 0)
        )

        if (
            source_n >= MIN_CELLS_PER_CELLTYPE
            and target_n >= MIN_CELLS_PER_CELLTYPE
        ):
            valid_celltypes.append(celltype)

    return valid_celltypes


def sample_indices_by_celltype(
    obs,
    species,
    celltypes,
    max_per_celltype,
    random_seed,
):
    """
    Randomly sample cells per cell type within one species.
    """

    rng = np.random.default_rng(random_seed)

    selected_indices = []

    species_array = obs[SPECIES_KEY].to_numpy()
    celltype_array = obs[CELLTYPE_KEY].to_numpy()

    for celltype in celltypes:
        current_indices = np.flatnonzero(
            (species_array == species)
            & (celltype_array == celltype)
        )

        if len(current_indices) == 0:
            continue

        if (
            max_per_celltype is not None
            and len(current_indices) > max_per_celltype
        ):
            current_indices = rng.choice(
                current_indices,
                size=max_per_celltype,
                replace=False,
            )

        selected_indices.extend(
            current_indices.tolist()
        )

    selected_indices = np.asarray(
        selected_indices,
        dtype=np.int64,
    )

    rng.shuffle(selected_indices)

    return selected_indices


def stratified_diagonal_split(
    obs,
    species,
    celltypes,
    random_seed,
):
    """
    Same-species diagonal:
    an independent train/test split per cell type.

    This is more stable than a single train_test_split over all cells and makes
    the training and test counts of each cell type exactly controllable.
    """

    rng = np.random.default_rng(random_seed)

    train_indices = []
    test_indices = []

    species_array = obs[SPECIES_KEY].to_numpy()
    celltype_array = obs[CELLTYPE_KEY].to_numpy()

    for celltype in celltypes:
        current_indices = np.flatnonzero(
            (species_array == species)
            & (celltype_array == celltype)
        )

        if len(current_indices) < 2:
            continue

        current_indices = current_indices.copy()
        rng.shuffle(current_indices)

        n_test = int(
            np.ceil(
                len(current_indices)
                * DIAGONAL_TEST_SIZE
            )
        )

        n_test = max(1, n_test)
        n_test = min(
            n_test,
            len(current_indices) - 1,
        )

        current_test = current_indices[:n_test]
        current_train = current_indices[n_test:]

        if (
            MAX_TRAIN_PER_CELLTYPE is not None
            and len(current_train) > MAX_TRAIN_PER_CELLTYPE
        ):
            current_train = rng.choice(
                current_train,
                size=MAX_TRAIN_PER_CELLTYPE,
                replace=False,
            )

        if (
            MAX_TEST_PER_CELLTYPE is not None
            and len(current_test) > MAX_TEST_PER_CELLTYPE
        ):
            current_test = rng.choice(
                current_test,
                size=MAX_TEST_PER_CELLTYPE,
                replace=False,
            )

        train_indices.extend(
            current_train.tolist()
        )

        test_indices.extend(
            current_test.tolist()
        )

    train_indices = np.asarray(
        train_indices,
        dtype=np.int64,
    )

    test_indices = np.asarray(
        test_indices,
        dtype=np.int64,
    )

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return train_indices, test_indices


def calculate_metrics(
    y_true,
    y_pred,
    evaluated_celltypes,
):
    """
    Compute per-cell-type and overall metrics.
    """

    per_celltype_metrics = {}

    for celltype in evaluated_celltypes:
        true_binary = (
            y_true == celltype
        ).astype(int)

        pred_binary = (
            y_pred == celltype
        ).astype(int)

        per_celltype_metrics[celltype] = {
            "f1": float(
                f1_score(
                    true_binary,
                    pred_binary,
                    zero_division=0,
                )
            ),
            "accuracy": float(
                accuracy_score(
                    true_binary,
                    pred_binary,
                )
            ),
            "precision": float(
                precision_score(
                    true_binary,
                    pred_binary,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    true_binary,
                    pred_binary,
                    zero_division=0,
                )
            ),
            "support": int(
                np.sum(true_binary)
            ),
        }

    overall_metrics = {
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=evaluated_celltypes,
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "n_test": int(len(y_true)),
        "n_celltypes": int(
            len(evaluated_celltypes)
        ),
    }

    return per_celltype_metrics, overall_metrics


def initialize_matrix(species_list):
    """Initialise an empty source x target matrix."""

    return pd.DataFrame(
        np.nan,
        index=species_list,
        columns=species_list,
        dtype=float,
    )


def plot_heatmap(
    matrix,
    title,
    colorbar_title,
    output_prefix,
    value_format=".2f",
    vmin=0.0,
    vmax=1.0,
):
    """
    Draw and save the heatmap.

    Outputs:
    - PNG
    - text-editable PDF
    """

    n_rows, n_columns = matrix.shape

    figure_width = max(
        5.5,
        n_columns * FIGSIZE_PER_SPECIES + 2.5,
    )

    figure_height = max(
        5.0,
        n_rows * FIGSIZE_PER_SPECIES + 2.0,
    )

    fig, ax = plt.subplots(
        figsize=(
            figure_width,
            figure_height,
        )
    )

    values = matrix.to_numpy(dtype=float)

    image = ax.imshow(
        values,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_xticks(
        np.arange(n_columns)
    )

    ax.set_yticks(
        np.arange(n_rows)
    )

    ax.set_xticklabels(
        matrix.columns,
        rotation=90,
        fontsize=TICK_FONT_SIZE,
    )

    ax.set_yticklabels(
        matrix.index,
        fontsize=TICK_FONT_SIZE,
    )

    ax.set_xlabel(
        "Target species",
        fontsize=11,
    )

    ax.set_ylabel(
        "Source species",
        fontsize=11,
    )

    ax.set_title(
        title,
        fontsize=12,
        pad=12,
    )

    for row_index in range(n_rows):
        for column_index in range(n_columns):
            value = values[
                row_index,
                column_index,
            ]

            if np.isnan(value):
                label = "NA"
                font_color = "black"
            else:
                label = format(
                    value,
                    value_format,
                )

                if value <= 0.20 or value >= 0.80:
                    font_color = "white"
                else:
                    font_color = "black"

            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=ANNOTATION_FONT_SIZE,
                color=font_color,
            )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        orientation="horizontal",
        fraction=0.055,
        pad=0.20,
    )

    colorbar.ax.set_title(
        colorbar_title,
        fontsize=9,
        pad=6,
    )

    ax.tick_params(
        axis="both",
        length=0,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()

    fig.savefig(
        f"{output_prefix}.png",
        dpi=DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        f"{output_prefix}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def calculate_nanmean_matrix(
    matrices,
    species_list,
):
    """
    Compute the NaN-aware mean of several source x target matrices, cell by cell.
    """

    array_stack = np.stack(
        [
            matrix.to_numpy(dtype=float)
            for matrix in matrices
        ],
        axis=0,
    )

    valid_count = np.sum(
        np.isfinite(array_stack),
        axis=0,
    )

    value_sum = np.nansum(
        array_stack,
        axis=0,
    )

    mean_array = np.full(
        value_sum.shape,
        np.nan,
        dtype=float,
    )

    valid_positions = valid_count > 0

    mean_array[valid_positions] = (
        value_sum[valid_positions]
        / valid_count[valid_positions]
    )

    return pd.DataFrame(
        mean_array,
        index=species_list,
        columns=species_list,
    )


# ============================================================
# 3. Main analysis
# ============================================================

def run_cross_species_knn(adata):
    """Run the full cross-species KNN transfer-learning analysis."""

    validate_adata(adata)

    output_root = Path(SAVE_DIR)

    matrix_dir = (
        output_root
        / "matrices"
    )

    per_celltype_matrix_dir = (
        matrix_dir
        / "per_celltype"
    )

    summary_matrix_dir = (
        matrix_dir
        / "summary"
    )

    figure_dir = (
        output_root
        / "figures"
    )

    per_celltype_figure_dir = (
        figure_dir
        / "per_celltype"
    )

    summary_figure_dir = (
        figure_dir
        / "summary"
    )

    for directory in [
        output_root,
        matrix_dir,
        per_celltype_matrix_dir,
        summary_matrix_dir,
        figure_dir,
        per_celltype_figure_dir,
        summary_figure_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print("=" * 80)
    print("Loading and cleaning data")
    print("=" * 80)

    obs, embedding = clean_metadata_and_embedding(
        adata
    )

    species_list = determine_species_order(
        obs
    )

    all_celltypes = determine_celltype_order(
        obs
    )

    # Keep only the selected species and cell types
    keep_mask = (
        obs[SPECIES_KEY].isin(species_list)
        & obs[CELLTYPE_KEY].isin(all_celltypes)
    ).to_numpy()

    obs = obs.loc[
        keep_mask
    ].copy()

    embedding = embedding[
        keep_mask
    ]

    obs.reset_index(
        drop=True,
        inplace=True,
    )

    print(f"H5AD path       : {H5AD_PATH}")
    print(f"Cells           : {len(obs):,}")
    print(f"Embedding key   : {EMBEDDING_KEY}")
    print(f"Embedding shape : {embedding.shape}")
    print(f"Species key     : {SPECIES_KEY}")
    print(f"Celltype key    : {CELLTYPE_KEY}")
    print(f"Species number  : {len(species_list)}")
    print(f"Celltype number : {len(all_celltypes)}")
    print(f"KNN K           : {N_NEIGHBORS}")
    print(f"KNN metric      : {KNN_METRIC}")
    print(f"KNN weights     : {KNN_WEIGHTS}")
    print(f"Random seed     : {RANDOM_SEED}")
    print(f"Output          : {output_root.resolve()}")

    print("\nSpecies and cell counts:")

    for species in species_list:
        n_cells = int(
            obs[SPECIES_KEY]
            .eq(species)
            .sum()
        )

        print(
            f"  {species}: {n_cells:,}"
        )

    # Save the cell-type count table
    celltype_count_table = pd.crosstab(
        obs[SPECIES_KEY],
        obs[CELLTYPE_KEY],
    )

    celltype_count_table = (
        celltype_count_table
        .reindex(
            index=species_list,
            columns=all_celltypes,
            fill_value=0,
        )
    )

    # Save the cell-type count table with common names
    celltype_count_table_common = celltype_count_table.copy()
    celltype_count_table_common.index = [
        species_to_common_name(species)
        for species in celltype_count_table_common.index
    ]
    celltype_count_table_common.index.name = "species_common_name"

    celltype_count_table_common.to_csv(
        output_root
        / "species_celltype_counts.csv"
    )

    # Also save scientific names, common names and display order for traceability
    species_name_table = pd.DataFrame(
        {
            "phylogenetic_order": np.arange(
                1,
                len(species_list) + 1,
            ),
            "scientific_name": species_list,
            "common_name": [
                species_to_common_name(species)
                for species in species_list
            ],
        }
    )
    species_name_table.to_csv(
        output_root / "species_name_mapping.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Initialise the result matrices
    # --------------------------------------------------------

    per_celltype_f1 = {
        celltype: initialize_matrix(
            species_list
        )
        for celltype in all_celltypes
    }

    per_celltype_accuracy = {
        celltype: initialize_matrix(
            species_list
        )
        for celltype in all_celltypes
    }

    per_celltype_precision = {
        celltype: initialize_matrix(
            species_list
        )
        for celltype in all_celltypes
    }

    per_celltype_recall = {
        celltype: initialize_matrix(
            species_list
        )
        for celltype in all_celltypes
    }

    per_celltype_support = {
        celltype: initialize_matrix(
            species_list
        )
        for celltype in all_celltypes
    }

    overall_macro_f1 = initialize_matrix(
        species_list
    )

    overall_accuracy = initialize_matrix(
        species_list
    )

    n_train_matrix = initialize_matrix(
        species_list
    )

    n_test_matrix = initialize_matrix(
        species_list
    )

    n_evaluated_celltypes_matrix = initialize_matrix(
        species_list
    )

    long_records = []

    total_pairs = (
        len(species_list)
        * len(species_list)
    )

    pair_number = 0

    # --------------------------------------------------------
    # Source x target loop
    # --------------------------------------------------------

    for source_index, source_species in enumerate(
        species_list
    ):
        for target_index, target_species in enumerate(
            species_list
        ):
            pair_number += 1

            pair_seed = (
                RANDOM_SEED
                + source_index * 10000
                + target_index * 100
            )

            print("\n" + "-" * 80)
            print(
                f"[{pair_number}/{total_pairs}] "
                f"{source_species} -> {target_species}"
            )

            common_celltypes = (
                get_common_valid_celltypes(
                    obs=obs,
                    source_species=source_species,
                    target_species=target_species,
                    allowed_celltypes=all_celltypes,
                )
            )

            print(
                "Number of shared valid cell types: "
                f"{len(common_celltypes)}"
            )

            if len(common_celltypes) < 2:
                print(
                    "Skipping: fewer than 2 shared valid cell types."
                )
                continue

            print(
                "Shared valid cell types:",
                ", ".join(common_celltypes),
            )

            # ------------------------------------------------
            # Build the training and test indices
            # ------------------------------------------------

            if source_species == target_species:
                train_indices, test_indices = (
                    stratified_diagonal_split(
                        obs=obs,
                        species=source_species,
                        celltypes=common_celltypes,
                        random_seed=pair_seed,
                    )
                )
            else:
                train_indices = (
                    sample_indices_by_celltype(
                        obs=obs,
                        species=source_species,
                        celltypes=common_celltypes,
                        max_per_celltype=MAX_TRAIN_PER_CELLTYPE,
                        random_seed=pair_seed,
                    )
                )

                test_indices = (
                    sample_indices_by_celltype(
                        obs=obs,
                        species=target_species,
                        celltypes=common_celltypes,
                        max_per_celltype=MAX_TEST_PER_CELLTYPE,
                        random_seed=pair_seed + 1,
                    )
                )

            if (
                len(train_indices) == 0
                or len(test_indices) == 0
            ):
                print(
                    "Skipping: the training or test set is empty."
                )
                continue

            X_train = embedding[
                train_indices
            ]

            X_test = embedding[
                test_indices
            ]

            y_train = (
                obs.iloc[train_indices][CELLTYPE_KEY]
                .astype(str)
                .to_numpy()
            )

            y_test = (
                obs.iloc[test_indices][CELLTYPE_KEY]
                .astype(str)
                .to_numpy()
            )

            train_celltypes = set(
                np.unique(y_train)
            )

            test_celltypes = set(
                np.unique(y_test)
            )

            evaluated_celltypes = [
                celltype
                for celltype in common_celltypes
                if (
                    celltype in train_celltypes
                    and celltype in test_celltypes
                )
            ]

            if len(evaluated_celltypes) < 2:
                print(
                    "Skipping: fewer than 2 cell types are actually evaluable."
                )
                continue

            # Guard against a potential inconsistency
            train_keep = np.isin(
                y_train,
                evaluated_celltypes,
            )

            test_keep = np.isin(
                y_test,
                evaluated_celltypes,
            )

            X_train = X_train[
                train_keep
            ]

            y_train = y_train[
                train_keep
            ]

            X_test = X_test[
                test_keep
            ]

            y_test = y_test[
                test_keep
            ]

            effective_k = min(
                N_NEIGHBORS,
                len(X_train),
            )

            if effective_k < 1:
                print(
                    "Skipping: no training cells are available."
                )
                continue

            print(
                f"Training cells: {len(X_train):,}"
            )

            print(
                f"Test cells: {len(X_test):,}"
            )

            print(
                f"Effective K: {effective_k}"
            )

            # ------------------------------------------------
            # Train KNN
            # ------------------------------------------------

            classifier = KNeighborsClassifier(
                n_neighbors=effective_k,
                weights=KNN_WEIGHTS,
                metric=KNN_METRIC,
                algorithm="auto",
                n_jobs=N_JOBS,
            )

            classifier.fit(
                X_train,
                y_train,
            )

            y_pred = classifier.predict(
                X_test
            )

            # ------------------------------------------------
            # Compute metrics
            # ------------------------------------------------

            (
                celltype_metrics,
                overall_metrics,
            ) = calculate_metrics(
                y_true=y_test,
                y_pred=y_pred,
                evaluated_celltypes=evaluated_celltypes,
            )

            overall_macro_f1.loc[
                source_species,
                target_species,
            ] = overall_metrics[
                "macro_f1"
            ]

            overall_accuracy.loc[
                source_species,
                target_species,
            ] = overall_metrics[
                "accuracy"
            ]

            n_train_matrix.loc[
                source_species,
                target_species,
            ] = len(y_train)

            n_test_matrix.loc[
                source_species,
                target_species,
            ] = len(y_test)

            n_evaluated_celltypes_matrix.loc[
                source_species,
                target_species,
            ] = len(evaluated_celltypes)

            print(
                "Macro-F1: "
                f"{overall_metrics['macro_f1']:.4f}"
            )

            print(
                "Accuracy: "
                f"{overall_metrics['accuracy']:.4f}"
            )

            for celltype, metrics in celltype_metrics.items():
                per_celltype_f1[
                    celltype
                ].loc[
                    source_species,
                    target_species,
                ] = metrics["f1"]

                per_celltype_accuracy[
                    celltype
                ].loc[
                    source_species,
                    target_species,
                ] = metrics["accuracy"]

                per_celltype_precision[
                    celltype
                ].loc[
                    source_species,
                    target_species,
                ] = metrics["precision"]

                per_celltype_recall[
                    celltype
                ].loc[
                    source_species,
                    target_species,
                ] = metrics["recall"]

                per_celltype_support[
                    celltype
                ].loc[
                    source_species,
                    target_species,
                ] = metrics["support"]

                long_records.append(
                    {
                        "source_species": source_species,
                        "source_common_name": species_to_common_name(
                            source_species
                        ),
                        "target_species": target_species,
                        "target_common_name": species_to_common_name(
                            target_species
                        ),
                        "celltype": celltype,
                        "f1": metrics["f1"],
                        "binary_accuracy": metrics[
                            "accuracy"
                        ],
                        "precision": metrics[
                            "precision"
                        ],
                        "recall": metrics[
                            "recall"
                        ],
                        "support": metrics[
                            "support"
                        ],
                        "overall_macro_f1": overall_metrics[
                            "macro_f1"
                        ],
                        "overall_accuracy": overall_metrics[
                            "accuracy"
                        ],
                        "n_train": int(
                            len(y_train)
                        ),
                        "n_test": int(
                            len(y_test)
                        ),
                        "n_evaluated_celltypes": int(
                            len(
                                evaluated_celltypes
                            )
                        ),
                    }
                )

    # ========================================================
    # Convert the scientific-name rows/columns of every result matrix to common names
    # ========================================================
    scientific_species_list = species_list.copy()
    common_species_list = [
        species_to_common_name(species)
        for species in scientific_species_list
    ]

    if len(set(common_species_list)) != len(common_species_list):
        raise ValueError(
            "Duplicate names appeared after the common-name mapping; check SPECIES_COMMON_NAME."
        )

    for celltype in all_celltypes:
        per_celltype_f1[celltype] = (
            rename_species_matrix_to_common_names(
                per_celltype_f1[celltype]
            )
        )
        per_celltype_accuracy[celltype] = (
            rename_species_matrix_to_common_names(
                per_celltype_accuracy[celltype]
            )
        )
        per_celltype_precision[celltype] = (
            rename_species_matrix_to_common_names(
                per_celltype_precision[celltype]
            )
        )
        per_celltype_recall[celltype] = (
            rename_species_matrix_to_common_names(
                per_celltype_recall[celltype]
            )
        )
        per_celltype_support[celltype] = (
            rename_species_matrix_to_common_names(
                per_celltype_support[celltype]
            )
        )

    overall_macro_f1 = rename_species_matrix_to_common_names(
        overall_macro_f1
    )
    overall_accuracy = rename_species_matrix_to_common_names(
        overall_accuracy
    )
    n_train_matrix = rename_species_matrix_to_common_names(
        n_train_matrix
    )
    n_test_matrix = rename_species_matrix_to_common_names(
        n_test_matrix
    )
    n_evaluated_celltypes_matrix = (
        rename_species_matrix_to_common_names(
            n_evaluated_celltypes_matrix
        )
    )

    # The averaged matrices, CSVs and plots all use the common-name order from here on
    species_list = common_species_list

    # ========================================================
    # 4. Save the per-cell-type matrices
    # ========================================================

    print("\n" + "=" * 80)
    print("Saving the per-cell-type result matrices")
    print("=" * 80)

    for celltype in all_celltypes:
        safe_name = sanitize_filename(
            celltype
        )

        per_celltype_f1[
            celltype
        ].to_csv(
            per_celltype_matrix_dir
            / f"{safe_name}.f1.csv"
        )

        per_celltype_accuracy[
            celltype
        ].to_csv(
            per_celltype_matrix_dir
            / f"{safe_name}.binary_accuracy.csv"
        )

        per_celltype_precision[
            celltype
        ].to_csv(
            per_celltype_matrix_dir
            / f"{safe_name}.precision.csv"
        )

        per_celltype_recall[
            celltype
        ].to_csv(
            per_celltype_matrix_dir
            / f"{safe_name}.recall.csv"
        )

        per_celltype_support[
            celltype
        ].to_csv(
            per_celltype_matrix_dir
            / f"{safe_name}.support.csv"
        )

    # ========================================================
    # 5. Summarise the averaged results
    # ========================================================

    mean_per_celltype_f1 = (
        calculate_nanmean_matrix(
            matrices=[
                per_celltype_f1[
                    celltype
                ]
                for celltype in all_celltypes
            ],
            species_list=species_list,
        )
    )

    mean_per_celltype_binary_accuracy = (
        calculate_nanmean_matrix(
            matrices=[
                per_celltype_accuracy[
                    celltype
                ]
                for celltype in all_celltypes
            ],
            species_list=species_list,
        )
    )

    overall_macro_f1.to_csv(
        summary_matrix_dir
        / "overall_multiclass_macro_f1.csv"
    )

    overall_accuracy.to_csv(
        summary_matrix_dir
        / "overall_multiclass_accuracy.csv"
    )

    mean_per_celltype_f1.to_csv(
        summary_matrix_dir
        / "mean_per_celltype_f1.csv"
    )

    mean_per_celltype_binary_accuracy.to_csv(
        summary_matrix_dir
        / "mean_per_celltype_binary_accuracy.csv"
    )

    n_train_matrix.to_csv(
        summary_matrix_dir
        / "number_of_train_cells.csv"
    )

    n_test_matrix.to_csv(
        summary_matrix_dir
        / "number_of_test_cells.csv"
    )

    n_evaluated_celltypes_matrix.to_csv(
        summary_matrix_dir
        / "number_of_evaluated_celltypes.csv"
    )

    long_metrics = pd.DataFrame(
        long_records
    )

    long_metrics.to_csv(
        output_root
        / "all_source_target_celltype_metrics.csv",
        index=False,
    )

    # ========================================================
    # 6. Plot the per-cell-type heatmaps
    # ========================================================

    print("\n" + "=" * 80)
    print("Plotting the per-cell-type heatmaps")
    print("=" * 80)

    for celltype in all_celltypes:
        safe_name = sanitize_filename(
            celltype
        )

        f1_matrix = per_celltype_f1[
            celltype
        ]

        accuracy_matrix = (
            per_celltype_accuracy[
                celltype
            ]
        )

        if np.isfinite(
            f1_matrix.to_numpy()
        ).any():
            plot_heatmap(
                matrix=f1_matrix,
                title=(
                    f"{celltype}\n"
                    "Cross-species KNN Transfer"
                ),
                colorbar_title="F1 Score",
                output_prefix=str(
                    per_celltype_figure_dir
                    / f"{safe_name}.f1"
                ),
            )

        if np.isfinite(
            accuracy_matrix.to_numpy()
        ).any():
            plot_heatmap(
                matrix=accuracy_matrix,
                title=(
                    f"{celltype}\n"
                    "Cross-species KNN Transfer"
                ),
                colorbar_title=(
                    "One-vs-rest Accuracy"
                ),
                output_prefix=str(
                    per_celltype_figure_dir
                    / f"{safe_name}.binary_accuracy"
                ),
            )

    # ========================================================
    # 7. Plot the summary heatmap
    # ========================================================

    print("\n" + "=" * 80)
    print("Plotting the summary heatmap")
    print("=" * 80)

    plot_heatmap(
        matrix=overall_macro_f1,
        title=(
            "Cross-species KNN Transfer"
        ),
        colorbar_title=(
            "Multiclass Macro F1 Score"
        ),
        output_prefix=str(
            summary_figure_dir
            / "overall_multiclass_macro_f1"
        ),
    )

    plot_heatmap(
        matrix=overall_accuracy,
        title=(
            "Cross-species KNN Transfer"
        ),
        colorbar_title=(
            "Multiclass Accuracy"
        ),
        output_prefix=str(
            summary_figure_dir
            / "overall_multiclass_accuracy"
        ),
    )

    plot_heatmap(
        matrix=mean_per_celltype_f1,
        title=(
            "Cross-species KNN Transfer"
        ),
        colorbar_title=(
            "Mean Per-cell-type F1 Score"
        ),
        output_prefix=str(
            summary_figure_dir
            / "mean_per_celltype_f1"
        ),
    )

    plot_heatmap(
        matrix=mean_per_celltype_binary_accuracy,
        title=(
            "Cross-species KNN Transfer"
        ),
        colorbar_title=(
            "Mean Per-cell-type Binary Accuracy"
        ),
        output_prefix=str(
            summary_figure_dir
            / "mean_per_celltype_binary_accuracy"
        ),
    )

    # ========================================================
    # 8. Save the parameters
    # ========================================================

    run_parameters = {
        "h5ad_path": H5AD_PATH,
        "save_dir": SAVE_DIR,
        "embedding_key": EMBEDDING_KEY,
        "species_key": SPECIES_KEY,
        "celltype_key": CELLTYPE_KEY,
        "n_neighbors": N_NEIGHBORS,
        "knn_weights": KNN_WEIGHTS,
        "knn_metric": KNN_METRIC,
        "n_jobs": N_JOBS,
        "max_train_per_celltype": (
            MAX_TRAIN_PER_CELLTYPE
        ),
        "max_test_per_celltype": (
            MAX_TEST_PER_CELLTYPE
        ),
        "min_cells_per_celltype": (
            MIN_CELLS_PER_CELLTYPE
        ),
        "diagonal_test_size": (
            DIAGONAL_TEST_SIZE
        ),
        "l2_normalize_embedding": (
            L2_NORMALIZE_EMBEDDING
        ),
        "random_seed": RANDOM_SEED,
        "species_order_scientific_name": scientific_species_list,
        "species_order_common_name": common_species_list,
        "species_common_name_mapping": {
            species: species_to_common_name(species)
            for species in scientific_species_list
        },
        "phylogenetic_order_reference": (
            "Ordered by phylogenetic proximity to human; "
            "bonobo and chimpanzee are sister taxa and their display order "
            "does not imply a resolved difference in distance to human."
        ),
        "celltype_order": all_celltypes,
    }

    with open(
        output_root
        / "run_parameters.json",
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            run_parameters,
            file_handle,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("Analysis complete")
    print("=" * 80)

    print(
        "Results directory:",
        output_root.resolve(),
    )

    print("\nMain summary results:")

    print(
        summary_matrix_dir
        / "overall_multiclass_macro_f1.csv"
    )

    print(
        summary_matrix_dir
        / "overall_multiclass_accuracy.csv"
    )

    print(
        summary_matrix_dir
        / "mean_per_celltype_f1.csv"
    )

    print(
        summary_matrix_dir
        / "mean_per_celltype_binary_accuracy.csv"
    )

    return {
        "overall_macro_f1": (
            overall_macro_f1
        ),
        "overall_accuracy": (
            overall_accuracy
        ),
        "mean_per_celltype_f1": (
            mean_per_celltype_f1
        ),
        "mean_per_celltype_binary_accuracy": (
            mean_per_celltype_binary_accuracy
        ),
        "per_celltype_f1": (
            per_celltype_f1
        ),
        "per_celltype_accuracy": (
            per_celltype_accuracy
        ),
        "long_metrics": (
            long_metrics
        ),
    }


# ============================================================
# 4. Load the AnnData object and run
# ============================================================

def main():
    print("=" * 80)
    print("Loading AnnData")
    print("=" * 80)

    if not os.path.isfile(H5AD_PATH):
        raise FileNotFoundError(
            f"H5AD file not found: {H5AD_PATH}"
        )

    adata = sc.read_h5ad(
        H5AD_PATH
    )

    print(adata)

    print(
        "adata.obs columns: ",
        adata.obs.columns.tolist(),
    )

    print(
        "adata.obsm keys: ",
        list(adata.obsm.keys()),
    )

    results = run_cross_species_knn(
        adata
    )

    return results


if __name__ == "__main__":
    main()