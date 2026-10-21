"""Cross-species developmental alignment from the `X_HENOSIS` representation.

Pipeline, in order:

    cells  ->  sample x cell-type centroids          (analysis.min_cells_per_sample)
           ->  stage x cell-type vectors             (unit-normalised, then averaged)
           ->  stage x stage cost matrix             (cell-type-balanced cosine)
           ->  a monotone stage correspondence       (dynamic time warping)

Why each step is what it is
---------------------------
*The first average is over cells within a sample.* Cells from one library share a
dissection, a batch and a donor, so treating them as independent observations of
a developmental trajectory inflates every subsequent statistic.

*The second average is over samples within a stage.* Sample vectors are unit
normalised BEFORE averaging, so a deep library contributes one vote rather than
many. Without that, the stage vector is dominated by whichever sample happened to
have the most cells.

*The distance is cell-type-balanced.* Every cell type present at both stages
contributes equally. Weighting by cell count would make the matrix mostly a
statement about which cell type is most abundant -- granule cells are 49% of the
atlas -- rather than about regulatory state.

*Deep-nuclei and white-matter samples are excluded from the alignment by default.*
`adult_DN` and `P63_DN` are a dissection region, not a later age; including them
makes a stage difference out of an anatomy difference. The count excluded is
reported, never silently dropped.

This module is shared by analyses 02 and 03. In the previous release the same
logic existed twice, in two scripts, with two different `stage_rank` conventions;
the second copy also walked every stage for single-stage species because the DTW
had no band. `min_stages_for_dtw` below is that fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from . import labels, stats


@dataclass
class Alignment:
    """Everything one alignment run produces, before any of it is drawn."""

    centroids: pd.DataFrame
    centroid_vectors: np.ndarray
    stage_vectors: dict[tuple[str, str, str], np.ndarray]
    excluded: pd.DataFrame
    min_cells: int
    dropped: pd.DataFrame = field(default_factory=pd.DataFrame)

    def pairs(self, source: str, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Cost and shared-cell-type matrices for one species pair."""
        source_stages = stage_axis(self, source)
        target_stages = stage_axis(self, target)
        return stats.stage_cost_matrix(
            self.stage_vectors, source, source_stages, target, target_stages
        )


def stage_axis(alignment: Alignment, species: str) -> list[str]:
    """The stages of `species` that survive into the cost matrix.

    Chronological order, with the deep-nuclei stages removed -- those samples are
    excluded by `special_anatomy` upstream, so keeping their row would produce a
    row of NaN.
    """
    present = {
        stage for (sp, stage, _) in alignment.stage_vectors if sp == species
    }
    return [stage for stage in labels.STAGE_ORDER[species]
            if stage in present and stage not in labels.DEEP_NUCLEI_STAGES]


def build_alignment(
    obs: pd.DataFrame,
    embedding: np.ndarray,
    *,
    min_cells: int,
    exclude_special_anatomy: bool = True,
    celltypes: Sequence[str] | None = None,
) -> Alignment:
    """Run the two averaging levels and store the result.

    `obs` must come from `io.load_design`, so it carries `species`, `stage`,
    `stage_rank`, `special_anatomy` and a `pos` column -- which the caller adds,
    because the index is a cell id and cannot index `embedding`.
    """
    if "pos" not in obs.columns:
        raise ValueError(
            "obs must carry a 'pos' column of row positions into the embedding; "
            "the index is a cell id and cannot be used to slice an array"
        )
    work = obs
    if celltypes is not None:
        work = work.loc[work["Celltype"].astype(str).isin(list(celltypes))]
    if work.empty:
        raise ValueError(f"no cells left after filtering to cell types {list(celltypes or [])}")

    keys = ["species", "sample_id", "stage", "stage_rank", "Celltype", "special_anatomy"]
    meta, vectors = stats.sample_centroids(work, embedding, keys, min_cells)

    kept = meta["dropped"].eq(False)
    dropped = stats.dropped_report(meta, keys)
    centroids = meta.loc[kept].reset_index(drop=True)
    # `stats.sample_centroids` sorts kept rows first, so the vectors line up with
    # the kept metadata by position; assert rather than assume.
    if len(centroids) != len(vectors):
        raise AssertionError(
            f"centroid metadata and vectors disagree: {len(centroids)} vs {len(vectors)}"
        )

    if exclude_special_anatomy:
        keep = ~centroids["special_anatomy"].to_numpy()
    else:
        keep = np.ones(len(centroids), dtype=bool)
    excluded = centroids.loc[~keep].copy()
    centroids = centroids.loc[keep].reset_index(drop=True)
    unit = stats.unit_rows(vectors[keep].astype(np.float64))

    # second level: samples -> stage, per cell type
    frame = centroids.assign(_row=np.arange(len(centroids)))
    stage_vectors: dict[tuple[str, str, str], np.ndarray] = {}
    for (species, stage, celltype), group in frame.groupby(
        ["species", "stage", "Celltype"], observed=True
    ):
        mean = unit[group["_row"].to_numpy()].mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm == 0:
            continue
        stage_vectors[(species, stage, celltype)] = (mean / norm).astype(np.float32)

    return Alignment(
        centroids=centroids,
        centroid_vectors=vectors[keep].astype(np.float32),
        stage_vectors=stage_vectors,
        excluded=excluded,
        min_cells=min_cells,
        dropped=dropped,
    )


def align_pair(
    alignment: Alignment,
    source: str,
    target: str,
    *,
    min_stages_for_dtw: int = 3,
    band: int | None = None,
) -> dict:
    """Cost matrix plus DTW correspondence for one species pair.

    `min_stages_for_dtw` guards against a degenerate path: with two stages (or
    one) the unconstrained DTW reaches the last cell by walking every column of
    the target, which reads as a stage correspondence but is an artefact of the
    matrix being short. Such a pair gets its cost matrix -- which is meaningful
    on its own -- and no path.
    """
    cost, shared = alignment.pairs(source, target)
    usable = bool(
        cost.notna().any().any()
        and len(cost) >= min_stages_for_dtw
        and cost.shape[1] >= min_stages_for_dtw
    )
    result = {
        "source": source,
        "target": target,
        "cost": cost,
        "shared": shared,
        "dtw_attempted": usable,
        "path": [],
        "total_cost": float("nan"),
        "reason": "",
        "min_stages": min_stages_for_dtw,
    }
    if not usable:
        result["reason"] = (
            f"fewer than {min_stages_for_dtw} stages on one axis or an entirely "
            f"empty cost matrix; a path here would be an artefact of the matrix "
            f"being short, not a stage correspondence"
        )
        return result

    path, total, _ = stats.dtw_path(cost, band=band)
    result["path"] = path
    result["total_cost"] = total
    result["path_table"] = stats.dtw_path_table(path, cost, shared, source, target)
    return result


def plot_alignment_grid(
    results: Sequence[dict],
    *,
    path_color: str = "#F0E442",
    cmap: str = "mako_r",
    figsize: tuple[float, float] | None = None,
    suptitle: str | None = None,
):
    """A row of per-pair heatmaps with the DTW path overlaid.

    One shared colour scale across panels, taken from the largest finite cost
    observed. Per-panel scales would make two pairs look equally well matched
    when their costs differ severalfold.
    """
    import matplotlib.pyplot as plt

    from . import plotting

    cmap = plotting.get_cmap(cmap, fallback="GnBu_r")
    panels = [r for r in results if r["cost"].notna().any().any()]
    if not panels:
        raise ValueError("no pair produced a non-empty cost matrix")
    ncols = len(panels)
    if figsize is None:
        figsize = (max(3.2 * ncols, 8.0), 4.6)
    fig, axes = plt.subplots(1, ncols, figsize=figsize, squeeze=False)
    vmax = max(float(np.nanmax(p["cost"].to_numpy())) for p in panels)

    for ax, panel in zip(axes.ravel(), panels):
        cost = panel["cost"]
        image = ax.imshow(cost.to_numpy(), cmap=cmap, vmin=0, vmax=vmax,
                          aspect="auto", origin="upper")
        ax.set_xticks(range(cost.shape[1]))
        ax.set_xticklabels(cost.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(cost.shape[0]))
        ax.set_yticklabels(cost.index, fontsize=7)
        ax.set_xlabel(f"{panel['target']} stage", fontsize=8)
        ax.set_ylabel(f"{panel['source']} stage", fontsize=8)
        ax.set_title(f"{panel['source']} to {panel['target']}", fontsize=9)
        if panel["path"]:
            ax.plot([j + 0.5 for _, j in panel["path"]],
                    [i + 0.5 for i, _ in panel["path"]],
                    color=path_color, lw=1.6, marker="o", ms=2.5, zorder=3)
        else:
            # A short in-panel note, not a title. The full reason is long enough
            # to collide with the neighbouring panels, and it belongs in
            # alignment_summary.json and the printed log, where it can be read --
            # the figure only has to say that the panel has no correspondence and
            # stop the reader assuming one was drawn and just hard to see.
            ax.text(0.5, 0.5, "no correspondence\n(fewer than "
                    f"{panel.get('min_stages', 3)} stages)",
                    transform=ax.transAxes, ha="center", va="center", fontsize=8,
                    color="#333333",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec="#999999", lw=0.6, alpha=0.9))

    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.90, top=0.86 if suptitle else 0.88,
                        bottom=0.18, wspace=0.45)
    bar_ax = fig.add_axes((0.92, 0.18, 0.014, 0.70))
    fig.colorbar(image, cax=bar_ax).set_label(
        "Cell-type-balanced cosine distance", fontsize=8
    )
    bar_ax.tick_params(labelsize=7)
    return fig
