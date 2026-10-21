"""Estimators shared by analyses 02 and 03, plus the DTW used in 02 and 03.

The unit of inference throughout is a **(sample, stage, cell type) centroid**,
never a cell. Cells from one library are not independent observations of a
developmental trajectory, so every statistic here takes a table of centroids and
their per-group metadata, and nothing takes a cell-by-cell embedding.

Two conventions that are easy to get subtly wrong are made explicit:

  * `slope_direction` returns a UNIT vector, so that two species' directions can
    be compared with a dot product without a degenerate group (many samples, all
    at one stage) dominating by magnitude alone.
  * `dtw_path` reports the cost of the matched pairs, not the accumulated cost.
    Dividing an accumulated cost by path length is a different statistic and the
    two are not comparable across matrices of different shape.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def sample_centroids(
    obs: pd.DataFrame,
    embedding: np.ndarray,
    keys: Sequence[str],
    min_cells: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Average the embedding within each key combination.

    Groups with fewer than `min_cells` cells are dropped, because a centroid
    built from four cells is closer to one cell's noise than to its group's
    mean. The dropped groups are returned in the metadata as `dropped=True`
    rows with no vector, so a caller can report them rather than lose them.

    Returns `(metadata, centroids)` aligned by row, centroids float32.
    """
    # `obs` and `embedding` must describe the SAME rows: group indices are
    # positions within `obs` and are used to slice `embedding` directly. Passing a
    # filtered frame alongside the full embedding therefore reads an arbitrary
    # other set of cells and still returns a plausible centroid -- it produced a
    # sign-flipped correlation twice while this repository was being written. The
    # check is cheap and turns a silent wrong answer into a named error.
    if len(obs) != len(embedding):
        raise ValueError(
            f"obs has {len(obs)} rows but the embedding has {len(embedding)}; they "
            f"must be row-aligned. Subset the embedding with the same mask "
            f"(e.g. `embedding[obs['pos'].to_numpy()]`) rather than passing a "
            f"filtered frame next to the full array."
        )
    frame = obs.loc[:, list(keys)].copy()
    # `groupby` on a categorical with unused categories would emit empty groups;
    # observed=True keeps the index to combinations that actually occur.
    grouped = frame.groupby(list(keys), observed=True)

    kept_meta: list[dict] = []
    kept_vectors: list[np.ndarray] = []
    dropped: list[dict] = []
    # positional indices, so the embedding can be sliced without a lookup
    positions = np.arange(len(obs))

    for values, idx in grouped.indices.items():
        if not isinstance(values, tuple):
            values = (values,)
        record = dict(zip(keys, values))
        record["n_cells"] = int(len(idx))
        if len(idx) < min_cells:
            record["dropped"] = True
            dropped.append(record)
            continue
        record["dropped"] = False
        kept_meta.append(record)
        kept_vectors.append(embedding[positions[idx]].mean(axis=0))

    meta = pd.DataFrame(kept_meta + dropped)
    if len(kept_vectors):
        vectors = np.vstack(kept_vectors).astype(np.float32)
    else:
        vectors = np.zeros((0, embedding.shape[1]), dtype=np.float32)
    # kept rows first, in the same order as `vectors`
    meta = meta.sort_values("dropped", kind="stable").reset_index(drop=True)
    return meta, vectors


def dropped_report(meta: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """The groups that did not reach the cell threshold, with their sizes."""
    if "dropped" not in meta.columns:
        return pd.DataFrame(columns=list(keys) + ["n_cells"])
    return (meta.loc[meta["dropped"], list(keys) + ["n_cells"]]
            .sort_values("n_cells", ascending=False).reset_index(drop=True))


# ---------------------------------------------------------------------------
# Vectors
# ---------------------------------------------------------------------------

def unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation, with an explicit guard for zero rows.

    A zero row can only come from a bug upstream (a centroid that is exactly the
    origin); raising is better than returning NaN that later reads as a large
    distance.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(
            f"{int((norms == 0).ravel().sum())} row(s) of the embedding are "
            f"exactly zero and cannot be normalised to a direction"
        )
    return matrix / norms


def slope_direction(x: np.ndarray, t: np.ndarray) -> np.ndarray | None:
    """Unit direction of the multivariate regression of `x` on `t`.

    This is the developmental *direction*: the derivative of the embedding with
    respect to developmental time, up to scale. Only the direction is retained,
    which is what makes two species comparable -- their sampling densities differ
    by an order of magnitude, and a slope magnitude would mostly measure that.

    Returns None when the design cannot identify a direction (one distinct time
    point, or zero total variance in `t`).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or t.ndim != 1 or len(x) != len(t):
        raise ValueError(f"shape mismatch: x={x.shape}, t={t.shape}")
    if len(np.unique(t)) < 2:
        return None
    centered = t - t.mean()
    denominator = float(centered @ centered)
    if denominator <= 0:
        return None
    slope = centered @ (x - x.mean(axis=0)) / denominator
    norm = float(np.linalg.norm(slope))
    if norm == 0:
        return None
    return (slope / norm).astype(np.float32)


def permutation_pvalue(observed: float, null: np.ndarray) -> float:
    """One-sided p with the observed value folded into the null.

    `(1 + #{null >= observed}) / (1 + n)` never returns 0, which matters because
    a reported p of exactly zero is not achievable from a finite permutation
    test and invites the reader to ask what the true resolution was.
    """
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0:
        return float("nan")
    return float((1 + np.sum(null >= observed)) / (1 + null.size))


# ---------------------------------------------------------------------------
# Cross-species cost and dynamic time warping
# ---------------------------------------------------------------------------

def stage_cost_matrix(
    vectors: dict[tuple[str, str, str], np.ndarray],
    source_species: str,
    source_stages: Sequence[str],
    target_species: str,
    target_stages: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cell-type-balanced cosine distance between two species' stage vectors.

    Each stage vector is already an average of unit sample vectors, renormalised
    to unit length, so `1 - dot` is a cosine distance. Every cell type present at
    BOTH stages contributes equally, whatever its cell count -- the alternative
    (weighting by cells) makes the matrix mostly a statement about which cell
    type is most abundant, which is exactly the confound this is trying to avoid.

    Returns `(cost, shared)`; `cost` is NaN where the two stages share no cell
    type and `shared` records how many were available, so a distance can never be
    read without knowing what it was computed from.
    """
    cost = pd.DataFrame(
        np.nan, index=list(source_stages), columns=list(target_stages), dtype=float
    )
    shared = pd.DataFrame(
        0, index=list(source_stages), columns=list(target_stages), dtype=int
    )
    for source_stage in source_stages:
        for target_stage in target_stages:
            common = sorted(
                ct for (sp, st, ct) in vectors
                if sp == target_species and st == target_stage
                and (source_species, source_stage, ct) in vectors
            )
            if not common:
                continue
            a = np.vstack([vectors[(source_species, source_stage, ct)] for ct in common])
            b = np.vstack([vectors[(target_species, target_stage, ct)] for ct in common])
            cost.loc[source_stage, target_stage] = float(np.mean(1.0 - np.sum(a * b, axis=1)))
            shared.loc[source_stage, target_stage] = len(common)
    return cost, shared


def dtw_path(cost: pd.DataFrame | np.ndarray, *, band: int | None = None):
    """Unconstrained DTW over a cost matrix.

    Missing entries (two stages sharing no cell type) get the finite maximum plus
    one, i.e. they are made as expensive as the worst observed pair rather than
    forbidden, because forbidding them can remove the only admissible path.

    `band` optionally restricts the path to `|i - j| <= band`; with a band the
    function reports whether the returned path is constrained, because a
    constrained path and an unconstrained one are different claims about a
    species' stage correspondence.

    Returns `(path, total_cost, accumulated)` where `path` is a list of
    `(row, column)` pairs and `total_cost` is the ACCUMULATED cost along it.
    """
    array = cost.to_numpy(dtype=float) if isinstance(cost, pd.DataFrame) else np.asarray(cost, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"cost must be 2-D, got {array.shape}")
    n, m = array.shape
    if n == 0 or m == 0:
        return [], float("nan"), np.zeros_like(array)
    if not np.isfinite(array).all():
        finite = array[np.isfinite(array)]
        penalty = (finite.max() + 1.0) if finite.size else 1.0
        array = np.where(np.isfinite(array), array, penalty)

    rows = np.arange(n)[:, None]
    cols = np.arange(m)[None, :]
    allowed = np.ones((n, m), dtype=bool)
    if band is not None:
        allowed = np.abs(rows - cols) <= band
        allowed[0, 0] = True
        allowed[n - 1, m - 1] = True

    accumulated = np.full((n, m), np.inf)
    back = np.full((n, m, 2), -1, dtype=int)
    if not allowed[0, 0]:
        raise ValueError("band excludes the origin; the path cannot start")
    accumulated[0, 0] = array[0, 0]
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            if not allowed[i, j]:
                continue
            best = np.inf
            best_prev = (-1, -1)
            for prev in ((i - 1, j), (i, j - 1), (i - 1, j - 1)):
                pi, pj = prev
                if pi < 0 or pj < 0:
                    continue
                if accumulated[pi, pj] < best:
                    best = accumulated[pi, pj]
                    best_prev = prev
            if np.isfinite(best):
                accumulated[i, j] = array[i, j] + best
                back[i, j] = best_prev

    if not np.isfinite(accumulated[n - 1, m - 1]):
        raise ValueError(
            "no admissible DTW path reaches the last cell; the cost matrix is "
            "mostly missing or the band is too narrow"
        )

    path: list[tuple[int, int]] = []
    i, j = n - 1, m - 1
    while i >= 0 and j >= 0:
        path.append((int(i), int(j)))
        if i == 0 and j == 0:
            break
        i, j = int(back[i, j, 0]), int(back[i, j, 1])
    path.reverse()
    return path, float(accumulated[n - 1, m - 1]), accumulated


def dtw_path_table(
    path: Sequence[tuple[int, int]],
    cost: pd.DataFrame,
    shared: pd.DataFrame,
    source_species: str,
    target_species: str,
) -> pd.DataFrame:
    """The path as a tidy table, one row per matched pair."""
    rows = []
    for order, (i, j) in enumerate(path):
        source_stage = cost.index[i]
        target_stage = cost.columns[j]
        rows.append({
            "path_order": order,
            "source_species": source_species,
            "source_stage": source_stage,
            "target_species": target_species,
            "target_stage": target_stage,
            "cost": float(cost.iloc[i, j]),
            "n_shared_celltypes": int(shared.iloc[i, j]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Leave-one-species-out
# ---------------------------------------------------------------------------

def loso_stage_prediction(
    meta: pd.DataFrame,
    vectors: np.ndarray,
    *,
    species_column: str = "species",
    stage_column: str = "stage_rank",
    center_within_species: bool = True,
    ridge_alphas: np.ndarray | None = None,
    min_samples: int = 3,
    min_stages: int = 2,
    eligible_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict each species' stage from the others, leaving that species out.

    Species-mean centring is applied BEFORE the split, using each species' own
    mean. Without it the first principal direction of the training set is
    "which species", the regression predicts species rather than age, and the
    held-out correlation is high for the wrong reason. Centring removes the
    species offset while leaving within-species developmental spread untouched.

    `eligible_only` restricts the TRAINING set to species that could themselves be
    evaluated -- at least `min_stages` distinct stages and `min_samples` samples.
    A single-stage species contributes samples that, after centring, sit at the
    origin with almost no variance: they carry no within-species developmental
    direction, so they cannot inform the axis, but they do enter the regression's
    standardisation and regularisation. Training on a population that includes
    species the test refuses to evaluate makes the training and evaluation sets
    different populations, and the asymmetry is not harmless -- on this dataset,
    including or excluding one single-stage species (macaque, 3 samples out of 53)
    moves the held-out human correlation between +0.183 and +0.383.

    Returns `(predictions, metrics)`; metrics carries Spearman rho per held-out
    species, computed on the species' own samples only.
    """
    from scipy.stats import spearmanr
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if ridge_alphas is None:
        ridge_alphas = np.logspace(-3, 3, 25)

    work = meta.reset_index(drop=True).copy()
    x = np.asarray(vectors, dtype=np.float64).copy()

    if eligible_only:
        eligible = [
            species for species, group in work.groupby(species_column, observed=True)
            if group[stage_column].nunique() >= min_stages and len(group) >= min_samples
        ]
        keep = work[species_column].isin(eligible).to_numpy()
        work, x = work.loc[keep].reset_index(drop=True), x[keep]

    if center_within_species:
        for _, idx in work.groupby(species_column, observed=True).indices.items():
            x[idx] = x[idx] - x[idx].mean(axis=0)

    predictions = []
    for held_out in sorted(work[species_column].unique()):
        test_mask = (work[species_column] == held_out).to_numpy()
        train_mask = ~test_mask
        if train_mask.sum() < 2 or test_mask.sum() == 0:
            continue
        y_train = work.loc[train_mask, stage_column].to_numpy(dtype=float)
        if len(np.unique(y_train)) < 2:
            continue
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=ridge_alphas))
        model.fit(x[train_mask], y_train)
        frame = work.loc[test_mask, [species_column, stage_column]].copy()
        frame["predicted_stage_rank"] = model.predict(x[test_mask])
        frame["held_out_species"] = held_out
        predictions.append(frame)

    pred = (pd.concat(predictions, ignore_index=True) if predictions
            else pd.DataFrame(columns=["species", "stage_rank",
                                       "predicted_stage_rank", "held_out_species"]))
    rows = []
    for held_out, group in pred.groupby("held_out_species", observed=True):
        if len(group) < 3 or group[stage_column].nunique() < 2:
            rho, pvalue = float("nan"), float("nan")
        else:
            rho, pvalue = spearmanr(group[stage_column], group["predicted_stage_rank"])
        rows.append({
            "held_out_species": held_out,
            "n_samples": int(len(group)),
            "n_stages": int(group[stage_column].nunique()),
            "spearman_rho": float(rho),
            "spearman_p": float(pvalue),
        })
    metrics = pd.DataFrame(rows).sort_values("held_out_species").reset_index(drop=True)
    return pred, metrics


# ---------------------------------------------------------------------------
# Bootstrap over samples
# ---------------------------------------------------------------------------

