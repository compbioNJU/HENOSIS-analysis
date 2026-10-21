#!/usr/bin/env python3
"""Step 2a: one UMAP per species, astroglia only, coloured by developmental stage.

Each panel shows every cell of the atlas in grey to keep the six panels on one
coordinate frame, then the species' astroglia on top, coloured by a within-species
relative developmental rank, with the stage centroids joined in rank order.

What changes in this release
----------------------------
Human Carnegie stages 18-23 are no longer astroglia, so the human panel loses its
four earliest time points and now starts at 11wpc. Those cells are not gone from
the figure: they are drawn as their own panel (`progenitor`) because a reader
comparing this release with the previous one needs to see where they went.

The panel keeps the same coordinate frame as the astroglia panels, so the two can
be laid out side by side.

Outputs (`results/02_astroglia_umap/`)
    astroglia_<species>_stage_umap.{pdf,png}    6 panels, one per species
    progenitor_human_stage_umap.{pdf,png}       the relabelled cells, same frame
    legend_astroglia_developmental_stage.{pdf,png}
    raster_layers/<panel>_points.png            the point-only bitmap used in each PDF
    astroglia_stage_centroids.csv               per species x stage: n cells, centroid
    astroglia_counted_cells.csv                 which cell types each panel contains
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import io, labels, plotting  # noqa: E402
from common.config import ensure_dir, load_config  # noqa: E402

POINT_SIZE = 0.5
BACKGROUND_SIZE = 0.15
BACKGROUND_COLOR = "#D5D5D5"
BACKGROUND_ALPHA = 0.25
TRAJECTORY_COLOR = "#222222"
LEADER_COLOR = "#333333"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--point-size", type=float, default=POINT_SIZE)
    return parser.parse_args()


def global_limits(xy: np.ndarray, pad_fraction: float = 0.02):
    """Axis limits from EVERY cell, so all panels share one frame.

    Computing the limits per panel would silently rescale each species and make
    six clouds of different sizes look comparable when they are not.
    """
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    pad_x = pad_fraction * (x_max - x_min)
    pad_y = pad_fraction * (y_max - y_min)
    return (x_min - pad_x, x_max + pad_x), (y_min - pad_y, y_max + pad_y)


def stage_centroids(frame: pd.DataFrame, xy: np.ndarray) -> pd.DataFrame:
    """Median position per stage, in rank order.

    `frame` must carry a `pos` column of row positions into `xy`: the obs index is
    a cell id, so it cannot be used to index the embedding array.

    The median rather than the mean: early samples are small and a handful of
    outlying cells moves a mean centroid enough to bend the trajectory line.
    """
    rows = []
    for (stage, rank), group in frame.groupby(["stage", "stage_rank"], observed=True):
        positions = xy[group["pos"].to_numpy()]
        rows.append({
            "stage": stage,
            "stage_rank": float(rank),
            "n_cells": int(len(group)),
            "centroid_x": float(np.median(positions[:, 0])),
            "centroid_y": float(np.median(positions[:, 1])),
        })
    return (pd.DataFrame(rows).sort_values(["stage_rank", "stage"])
            .reset_index(drop=True))


def draw_panel(panel: str, focal: pd.DataFrame, all_xy: np.ndarray, focal_xy: np.ndarray,
               centroids: pd.DataFrame, limits, out_dir: Path, dpi: int,
               point_size: float, title: str) -> Path:
    """One species panel: raster point layer plus vector annotations."""
    xlim, ylim = limits

    focal_colors = np.array([
        plt.get_cmap(labels.STAGE_CMAP)(rank)
        for rank in focal["stage_rank"].to_numpy()
    ])
    layer = plotting.render_point_layer(
        focal_xy, focal_colors, size=point_size,
        xlim=xlim, ylim=ylim,
        background_xy=all_xy, background_color=BACKGROUND_COLOR,
        background_size=BACKGROUND_SIZE, background_alpha=BACKGROUND_ALPHA,
        inches=6.0, dpi=dpi,
    )
    # The clean point-only bitmap, kept as a layer for figure assembly.
    from matplotlib.figure import Figure
    clean = Figure(figsize=(6, 6), dpi=dpi)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(clean)
    clean_ax = clean.add_axes((0, 0, 1, 1))
    clean_ax.imshow(layer, extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                    origin="upper", aspect="auto")
    clean_ax.set_axis_off()
    (out_dir / "raster_layers").mkdir(parents=True, exist_ok=True)
    clean.savefig(out_dir / "raster_layers" / f"{panel}_points.png", dpi=dpi,
                  transparent=True)
    plt.close(clean)

    fig, ax = plt.subplots(figsize=(6, 6))
    plotting.paste_point_layer(ax, layer, xlim, ylim)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    plotting.clean_axis(ax)

    if len(centroids) > 1:
        ax.plot(centroids["centroid_x"], centroids["centroid_y"],
                color=TRAJECTORY_COLOR, lw=0.9, alpha=0.8, zorder=3)
    ax.scatter(centroids["centroid_x"], centroids["centroid_y"], s=24,
               c=centroids["stage_rank"], cmap=labels.STAGE_CMAP, vmin=0, vmax=1,
               edgecolor="black", linewidth=0.45, zorder=4)

    # Labels alternate sides so that consecutive stages on a curved trajectory do
    # not stack, then are pushed apart along y and clamped inside the panel.
    y_span = ylim[1] - ylim[0]
    gap = 0.030 * y_span
    low, high = ylim[0] + 0.025 * y_span, ylim[1] - 0.025 * y_span
    for side in (-1, 1):
        mask = np.arange(len(centroids)) % 2 == (0 if side < 0 else 1)
        if not mask.any():
            continue
        subset = centroids[mask]
        placed = plotting.spread_labels(subset["centroid_y"].to_numpy(), gap, low, high)
        offset = 0.030 * (xlim[1] - xlim[0]) * side
        for (_, row), y in zip(subset.iterrows(), placed):
            ax.annotate(
                row["stage"],
                xy=(row["centroid_x"], row["centroid_y"]),
                xytext=(row["centroid_x"] + offset, y),
                fontsize=8, ha="left" if side > 0 else "right", va="center",
                arrowprops=dict(arrowstyle="-", color=LEADER_COLOR, lw=0.6,
                                shrinkA=1.5, shrinkB=1.5),
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.82),
                zorder=6,
            )

    ax.set_title(title, fontsize=9, pad=6)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02)
    paths = plotting.save_figure(fig, out_dir / panel, dpi=dpi, transparent=True)
    print(f"[OK] {panel}: {len(focal):,} cells, {len(centroids)} stages")
    return paths[0]


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    plotting.apply_style(args.dpi)

    h5ad = cfg.require_input("h5ad")
    astroglia = str(cfg.param("astroglia_label"))
    progenitor = str(cfg.param("progenitor_label"))
    out = ensure_dir(cfg.result_dir("02_astroglia_umap"))

    obs, embedding, info = io.load_design(
        h5ad, obs_columns=["Celltype", "Common Name"], with_embedding="X_umap"
    )
    xy = embedding[:, :2].astype(np.float64)
    limits = global_limits(xy)
    print(f"[READ] {info['path']}  ({info['n_obs']:,} cells)")
    print(f"[FRAME] x={limits[0]}  y={limits[1]}  (shared by every panel)")

    celltype = obs["Celltype"].astype(str)
    # Row positions into `xy`. The obs index is a cell id, so every subsequent
    # subset carries its position with it rather than trying to index an array
    # with a label.
    obs = obs.assign(pos=np.arange(len(obs)))
    populations = [
        (astroglia, "astroglia", None, None),
        (progenitor, "progenitor", "Human", progenitor),
    ]

    counted = []
    all_centroids = []
    for label, panel_kind, only_species, display in populations:
        members = obs.loc[celltype.eq(label)]
        if only_species:
            members = members.loc[members["species"].eq(only_species)]
        if members.empty:
            print(f"[SKIP] {label}: no cells")
            continue
        species_list = ([only_species] if only_species
                        else [s for s in labels.SPECIES_ORDER
                              if s in set(members["species"])])
        counted.append({"panel_kind": panel_kind, "Celltype": label,
                        "n_cells": int(len(members)),
                        "species": ";".join(species_list)})
        for species in species_list:
            subset = members.loc[members["species"].eq(species)]
            if subset.empty:
                continue
            positions = subset["pos"].to_numpy()
            centroids = stage_centroids(subset, xy)
            centroids.insert(0, "species", species)
            centroids.insert(0, "panel_kind", panel_kind)
            all_centroids.append(centroids)
            if only_species:
                panel = f"progenitor_{species.lower()}_stage_umap"
                title = (f"{species} {display}  |  Carnegie stage 18-23  |  "
                         f"{len(subset):,} cells")
            else:
                panel = f"astroglia_{species.lower()}_stage_umap"
                title = (f"{species} astroglia  |  "
                         f"{subset['stage'].nunique()} stages  |  {len(subset):,} cells")
            draw_panel(panel, subset, xy, xy[positions], centroids, limits,
                       out, args.dpi, args.point_size, title)

    # -- legend ---------------------------------------------------------------
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    legend = Figure(figsize=(1.35, 3.6))
    FigureCanvasAgg(legend)
    cax = legend.add_axes((0.26, 0.16, 0.20, 0.68))
    colorbar = legend.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=labels.STAGE_CMAP),
                               cax=cax)
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["Early", "Late"])
    colorbar.set_label("Within-species relative stage", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    plotting.save_figure(legend, out / "legend_astroglia_developmental_stage",
                         dpi=args.dpi, transparent=True, pad=True)

    # -- tables ---------------------------------------------------------------
    io.write_table(out / "astroglia_stage_centroids.csv",
                   pd.concat(all_centroids, ignore_index=True))
    io.write_table(out / "astroglia_counted_cells.csv", pd.DataFrame(counted))

    # Per-species composition, so the reader can see what the relabel removed
    # rather than inferring it from a smaller cell count.
    composition = (obs.loc[celltype.isin([astroglia, progenitor])]
                   .groupby(["Celltype", "species", "stage"], observed=True).size()
                   .rename("n_cells").reset_index())
    io.write_table(out / "astroglia_cells_by_stage.csv", composition)

    io.write_json(out / "astroglia_umap_parameters.json", {
        "input": info["path"],
        "astroglia_label": astroglia,
        "progenitor_label": progenitor,
        "shared_axis_limits": {"x": list(limits[0]), "y": list(limits[1])},
        "limits_from": "all cells, so the six species panels are comparable",
        "stage_rank_calibration": "labels.STAGE_RANK (relative age in [0,1])",
        "point_size": args.point_size,
        "raster_layers": "results/02_astroglia_umap/raster_layers/*_points.png",
        "release_edit": (
            "human Carnegie stage 18-23 is absent from the astroglia panels and "
            "is drawn as its own 'progenitor' panel instead"
        ),
    })
    print(f"\n[DONE] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
