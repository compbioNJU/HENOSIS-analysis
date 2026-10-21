"""Reading the atlas, and writing results.

Loading policy
--------------
The cell-level input is a 345k x 64 HDF5 file that is ~400 MB on disk. Every
script here reads it **backed** (`ad.read_h5ad(..., backed="r")`) and pulls only
the `obs` columns and the one `obsm` key it needs. Reading the whole thing into
memory is not wrong, only wasteful, and under concurrency it is the difference
between six workers and two.

Deliberate omissions
--------------------
`uns["Celltype_colors"]` is NOT used anywhere, and neither is
`uns["celltype_mapping_figure1_to_figure2"]`. In the input for this release both
are stale:

  * `Celltype_colors` holds 22 colours while `Celltype` has 23 categories, so
    zipping them truncates and silently mis-colours the tail;
  * `celltype_mapping_figure1_to_figure2` has no row for the new `progenitor`
    label.

Colours and the mapping therefore come from `src/common/labels.py`, which is
explicit and complete. `audit_input()` re-derives both from the file and reports
any disagreement, so the staleness is visible rather than assumed away.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import anndata as ad
import numpy as np
import pandas as pd

from . import labels

# obs columns any script here may ask for. Asking for one that the atlas does not
# have is an error, not a silent skip: a missing `Celltype` would otherwise turn
# into an all-False mask and an empty figure.
KNOWN_OBS_COLUMNS = (
    "Org_Celltype", "Celltype", "CellID", "cell", "Species", "Common Name",
    "Latin Name", "Tissue", "Org_stage",
)

# Columns `labels.build_design` needs. Kept next to the reader so that the two
# cannot drift: adding a column to the design without adding it here fails at the
# `build_design` call with the column name, not with a KeyError somewhere later.
DESIGN_SOURCE_COLUMNS = ("Common Name", "Species", "Org_stage")


def open_atlas(path: str | Path) -> ad.AnnData:
    """Open the atlas backed. Caller is responsible for `.file.close()`."""
    return ad.read_h5ad(str(path), backed="r")


def read_obs(adata: ad.AnnData, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """`obs` as a plain DataFrame, with the design columns appended."""
    available = list(adata.obs.columns)
    wanted = list(columns) if columns else available
    missing = [c for c in wanted if c not in available]
    if missing:
        raise KeyError(
            f"requested obs column(s) {missing} are not in the file. Available: "
            f"{available}"
        )
    frame = adata.obs[wanted].copy()
    frame.index = adata.obs_names.copy()
    return frame


def read_embedding(adata: ad.AnnData, key: str) -> np.ndarray:
    """An `obsm` key as float32, validated.

    A NaN in the embedding would propagate into a centroid and then into a
    distance, and would show up as a plausible-looking number rather than as an
    error, so finiteness is checked at the boundary.
    """
    if key not in adata.obsm:
        raise KeyError(
            f"obsm['{key}'] is missing. Present keys: {list(adata.obsm.keys())}"
        )
    values = np.asarray(adata.obsm[key], dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"obsm['{key}'] must be 2-D, got shape {values.shape}")
    if not np.isfinite(values).all():
        bad = int((~np.isfinite(values)).any(axis=1).sum())
        raise ValueError(f"obsm['{key}'] contains non-finite values in {bad} cells")
    return values


def load_design_multi(
    path: str | Path,
    obs_columns: Iterable[str] = (),
    *,
    embeddings: Sequence[str] = (),
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    """As `load_design`, but returns a dict of embeddings.

    Analysis 03 needs both `X_HENOSIS` (the direction is estimated in it) and
    `X_umap` (the small multiples are drawn in it). Reading the file twice to get
    one key each time would work, but the two reads could disagree if the file is
    replaced between them, so they are taken from the same handle.
    """
    adata = open_atlas(path)
    try:
        wanted = list(dict.fromkeys(list(DESIGN_SOURCE_COLUMNS) + list(obs_columns)))
        obs = read_obs(adata, wanted)
        design = labels.build_design(obs).frame()
        obs = pd.concat([obs, design], axis=1)
        arrays = {key: read_embedding(adata, key) for key in embeddings}
        info = _info(adata, path)
    finally:
        adata.file.close()
    return obs, arrays, info


def load_design(
    path: str | Path,
    obs_columns: Iterable[str] = (),
    *,
    with_embedding: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None, dict[str, Any]]:
    """Open, read obs + design, optionally read one embedding, close.

    Returns `(obs_with_design, embedding_or_None, info)`. `obs_with_design` is
    the requested obs columns plus every column of `labels.Design`.
    """
    adata = open_atlas(path)
    try:
        # The design columns are derived from obs, so the columns they need are
        # added here rather than being something each caller has to remember.
        wanted = list(dict.fromkeys(list(DESIGN_SOURCE_COLUMNS) + list(obs_columns)))
        obs = read_obs(adata, wanted)
        design = labels.build_design(obs).frame()
        obs = pd.concat([obs, design], axis=1)
        embedding = read_embedding(adata, with_embedding) if with_embedding else None
        info = _info(adata, path)
    finally:
        adata.file.close()
    return obs, embedding, info


def _info(adata: ad.AnnData, path: str | Path) -> dict[str, Any]:
    return {
        "path": str(Path(path).resolve()),
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "obsm_keys": list(adata.obsm.keys()),
        "obs_columns": list(adata.obs.columns),
    }


# ---------------------------------------------------------------------------
# Input audit
# ---------------------------------------------------------------------------

def audit_input(path: str | Path) -> dict[str, Any]:
    """Report what the input actually contains, including its stale `uns` entries.

    Written out next to the results so that a reader can see the release the
    figures were drawn from without opening the h5ad.
    """
    adata = open_atlas(path)
    try:
        obs = adata.obs
        celltype_counts = obs["Celltype"].astype(str).value_counts()
        species_counts = obs["Species"].astype(str).value_counts()

        report: dict[str, Any] = {
            "path": str(Path(path).resolve()),
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "obsm": {k: list(np.shape(adata.obsm[k])) for k in adata.obsm},
            "obs_columns": list(obs.columns),
            "species_cells": species_counts.to_dict(),
            "celltype_cells": celltype_counts.to_dict(),
            "stage_cells": obs["Org_stage"].astype(str).value_counts().to_dict(),
        }

        # -- the relabelling that defines this release ------------------------
        human = obs["Species"].astype(str).eq("human")
        human_cs = human & obs["Org_stage"].astype(str).isin(
            ["CS18-19", "CS20", "CS22-23"]
        )
        report["release_edit"] = {
            "description": (
                "human Carnegie stage 18-23 cells carry 'progenitor' instead of "
                "'astroglia (progenitors+astrocytes)'"
            ),
            "human_cs18_23_cells": int(human_cs.sum()),
            "human_cs18_23_labelled_progenitor": int(
                (human_cs & obs["Celltype"].astype(str).eq(labels.PROGENITOR_LABEL)).sum()
            ),
            "progenitor_cells_total": int(
                obs["Celltype"].astype(str).eq(labels.PROGENITOR_LABEL).sum()
            ),
            "progenitor_species": sorted(
                obs.loc[obs["Celltype"].astype(str).eq(labels.PROGENITOR_LABEL), "Species"]
                .astype(str).unique().tolist()
            ),
            "astroglia_cells_total": int(
                obs["Celltype"].astype(str).eq(labels.ASTROGLIA_FAMILY_LABEL).sum()
            ),
            "astroglia_species": sorted(
                obs.loc[obs["Celltype"].astype(str).eq(labels.ASTROGLIA_FAMILY_LABEL), "Species"]
                .astype(str).unique().tolist()
            ),
        }

        # -- stale uns entries, derived rather than assumed -------------------
        stale: dict[str, Any] = {}
        celltype_categories = list(obs["Celltype"].cat.categories)
        if "Celltype_colors" in adata.uns:
            stored = list(np.asarray(adata.uns["Celltype_colors"]).ravel())
            stale["Celltype_colors"] = {
                "n_stored": len(stored),
                "n_categories": len(celltype_categories),
                "used_by_this_repository": False,
                "problem": (
                    None if len(stored) == len(celltype_categories)
                    else f"{len(celltype_categories)} Celltype categories but "
                         f"{len(stored)} colours stored; pairing them by position "
                         f"would mis-colour the tail. The palette used here is "
                         f"explicit and lives in src/common/labels.py."
                ),
            }
        if "celltype_mapping_figure1_to_figure2" in adata.uns:
            mapping = adata.uns["celltype_mapping_figure1_to_figure2"]
            mapped = set(pd.DataFrame(mapping)["Celltype"].astype(str))
            stale["celltype_mapping_figure1_to_figure2"] = {
                "n_rows": len(mapped),
                "missing_celltypes": sorted(set(celltype_categories) - mapped),
                "used_by_this_repository": False,
            }
        report["stale_uns"] = stale
        return report
    finally:
        adata.file.close()


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_jsonable) + "\n",
                    encoding="utf-8")
    return path


def _jsonable(value: Any) -> Any:
    """Make numpy scalars/arrays serialisable without importing numpy into json."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")


def write_table(path: str | Path, frame: pd.DataFrame, *, index: bool = False,
                sep: str = ",") -> Path:
    """Tabular output. CSV is the default because these are read by eye too."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index, sep=sep)
    return path


