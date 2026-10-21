#!/usr/bin/env python3
"""Step 1: the atlas UMAP, coloured by species and by cell type.

The layout is READ, never computed: `obsm['X_umap']` is the RAPIDS/cuML layout
of the 50-dimensional `X_HENOSIS` representation (n_neighbors 15, euclidean,
random_state 0) that ships with the atlas. Recomputing it here would make the
figure depend on this repository's CPU layout instead of the published one, and
the two are known to disagree.

What changes in this release
----------------------------
`progenitor` is a new cell-type category (the 8,855 human Carnegie stage 18-23
cells that used to be counted as astroglia). It gets its own colour from the
frozen palette.

The palette is a literal table rather than a colormap index. The previous
version assigned colours by sorted-label position over tab20+tab20b+tab20c, which
means inserting a label shifts every colour after it -- adding `progenitor`
between `parabrachial` and `unipolar brush cell (UBC)` would have silently
recoloured an unrelated cell type and broken the palette that the per-species
highlight and stacked-bar figures read from the colour CSV. Freezing the table
keeps every existing colour byte-identical, so the only visible difference from
the previous release is the new category.

Outputs (`results/01_global_umap/`)
    umap_<stem>_species.{png,pdf}        345k points, one colour per species
    umap_<stem>_Celltype.{png,pdf}       345k points, one colour per cell type
    legend_<stem>_{species,Celltype}.pdf standalone legends, editable text
    colors_<stem>_{species,Celltype}.csv the palettes actually used
    umap_parameters.json                 which layout, which point size, from which input
    celltype_cells_by_species.csv        composition, so a reader can check a colour by count
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

# Point size for a 345k-cell cloud. Above ~4 points per 10k pixels the cloud
# turns into a solid blob and the density structure that the figure is for
# disappears; below ~0.15 the sparse early-embryonic samples vanish.
REFERENCE_CELLS = 120_000.0
MIN_POINT_SIZE, MAX_POINT_SIZE = 0.15, 8.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--umap-key", default="X_umap",
                        help="obsm key holding the precomputed layout")
    parser.add_argument("--point-size", type=float, default=None,
                        help="override the automatic size")
    parser.add_argument("--dpi", type=int, default=plotting.DPI_PNG)
    return parser.parse_args()


def automatic_point_size(n_obs: int, requested: float | None) -> float:
    if requested is not None:
        return float(requested)
    return float(np.clip(REFERENCE_CELLS / max(n_obs, 1), MIN_POINT_SIZE, MAX_POINT_SIZE))


def plot_one(xy: np.ndarray, colors: np.ndarray, size: float, dpi: int,
             path_stem: Path, *, title: str, subtitle: str,
             require_text: bool = True) -> None:
    """One panel: rasterised points, vector text.

    `rasterized=True` is applied to the scatter only. The title is drawn as a
    separate artist, so it stays selectable in the PDF even though the cloud
    beneath it is a bitmap -- vectorising 345k markers instead would produce a
    figure file of well over 100 MB with no visual gain.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    plotting.scatter_points(ax, xy, color=colors, size=size)
    plotting.clean_axis(ax)
    fig.suptitle(title, fontsize=10, y=0.98)
    ax.set_title(subtitle, fontsize=8, color="#444444")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01)
    plotting.save_figure(fig, path_stem, dpi=dpi, transparent=True,
                         require_text=require_text)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    plotting.apply_style(args.dpi)

    h5ad = cfg.require_input("h5ad")
    out = ensure_dir(cfg.result_dir("01_global_umap"))

    obs, embedding, info = io.load_design(
        h5ad, obs_columns=["Species", "Common Name", "Celltype"],
        with_embedding=args.umap_key,
    )
    xy = embedding[:, :2]
    if xy.shape[1] != 2:
        raise ValueError(f"obsm['{args.umap_key}'] must have at least 2 columns")

    size = automatic_point_size(len(obs), args.point_size)
    stem = Path(h5ad).stem
    print(f"[READ] {info['path']}  ({info['n_obs']:,} cells)")
    print(f"[UMAP] key={args.umap_key}  point size={size:.4f}")

    # -- species -------------------------------------------------------------
    species_present = [s for s in labels.SPECIES_ORDER if s in set(obs["species"])]
    extra_species = sorted(set(obs["species"]) - set(labels.SPECIES_ORDER))
    species_labels = species_present + extra_species
    species_colors = plotting.categorical_colors(
        species_labels, labels.SPECIES_COLORS, fallback_cmap="tab10", name="species"
    )
    # explicit display order: the palette's order, then anything unlisted
    species_colors = {label: species_colors[label] for label in species_labels}

    # -- cell type -----------------------------------------------------------
    celltype_labels = labels.celltype_order(obs["Celltype"].astype(str).unique())
    celltype_colors = plotting.categorical_colors(
        celltype_labels, labels.CELLTYPE_COLORS, name="Celltype"
    )
    celltype_colors = {label: celltype_colors[label] for label in celltype_labels}

    panels = [
        ("species", obs["species"].astype(str).to_numpy(), species_colors,
         "Species", f"{len(species_labels)} species"),
        ("Celltype", obs["Celltype"].astype(str).to_numpy(), celltype_colors,
         "Cell type", f"{len(celltype_labels)} cell types"),
    ]
    for key, values, color_map, title, subtitle in panels:
        colors = np.array([color_map[v] for v in values])
        plot_one(xy, colors, size, args.dpi, out / f"umap_{stem}_{key}",
                 title=title,
                 subtitle=f"{subtitle}  |  {info['n_obs']:,} cells  |  layout: obsm['{args.umap_key}']")
        # A standalone legend, so the panel above can stay label-free.
        plotting.write_legend(out / f"legend_{stem}_{key}", color_map, key)
        io.write_table(
            out / f"colors_{stem}_{key}.csv",
            pd.DataFrame({key: list(color_map), "color": list(color_map.values())}),
        )
        print(f"[OK] {key}: {len(color_map)} labels")

    # -- composition, so a colour can be checked against a number -------------
    composition = (obs.groupby(["Celltype", "species"], observed=True).size()
                   .rename("n_cells").reset_index()
                   .pivot(index="Celltype", columns="species", values="n_cells")
                   .fillna(0).astype(int)
                   .reindex(celltype_labels))
    composition["total"] = composition.sum(axis=1)
    composition.index.name = "Celltype"
    io.write_table(out / "celltype_cells_by_species.csv", composition.reset_index(),
                   index=False)

    io.write_json(out / "umap_parameters.json", {
        "input": info["path"],
        "n_obs": info["n_obs"],
        "n_vars": info["n_vars"],
        "umap_key": args.umap_key,
        "umap_computed_here": False,
        "umap_source": "shipped with the atlas (RAPIDS/cuML on obsm['X_HENOSIS'])",
        "point_size": size,
        "figure_inches": [6, 6],
        "dpi_png": args.dpi,
        "species_colors": species_colors,
        "celltype_colors": celltype_colors,
        "release_edit": (
            f"'{labels.PROGENITOR_LABEL}' is a new category in this release; "
            f"the other palettes are unchanged from the previous release"
        ),
    })
    print("\ncell type x species:")
    print(composition.to_string())
    print(f"\n[DONE] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
