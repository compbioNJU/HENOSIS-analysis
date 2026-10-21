#!/usr/bin/env python3
"""Plot one joint human/mouse MGC marker-expression dotplot."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_gene_rank_dotplots as core


SPECIES = ("homo_sapiens", "Mus_musculus")
SPECIES_LABELS = {"homo_sapiens": "Human", "Mus_musculus": "Mouse"}


def parse_args():
    here = Path(__file__).resolve().parent
    data_root = Path(os.environ.get("ANALYSIS_DATA_DIR", here.parents[3] / "analysis_data"))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank-dir", type=Path, default=data_root / "Figure2/microglia/ranked_genes")
    p.add_argument("--h5ad-dir", type=Path,
                   default=data_root / "Figure2/microglia")
    p.add_argument("--output-dir", type=Path, default=here.parents[1] / "results/microglia")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--color-clip-percentile", type=float, default=99.0)
    p.add_argument("--panel-width", type=float, default=620.0,
                   help="Bubble-matrix panel width in proportional units")
    p.add_argument("--panel-height", type=float, default=490.0,
                   help="Bubble-matrix panel height in proportional units")
    return p.parse_args()


def marker_union(rank_dir, top_n):
    tables = {sp: core.load_markers(rank_dir / f"{sp}_MGC.csv", top_n) for sp in SPECIES}
    appearances = {}
    display = {}
    rank_sum = {}
    for sp, table in tables.items():
        for row in table.itertuples(index=False):
            key = row.gene.upper()
            appearances.setdefault(key, set()).add(sp)
            display.setdefault(key, row.gene.upper())
            rank_sum[key] = rank_sum.get(key, 0.0) + float(row.rank)
    # Shared genes first, then average/best marker rank.
    keys = sorted(appearances, key=lambda k: (-len(appearances[k]), rank_sum[k] / len(appearances[k]), k))
    return [display[k] for k in keys], tables


def species_summary(species, genes, h5ad_dir):
    marker_frame = pd.DataFrame({"gene": genes, "rank": np.arange(1, len(genes) + 1)})
    adata = ad.read_h5ad(h5ad_dir / f"{species}.h5ad", backed="r")
    try:
        resolved, indices = core.resolve_genes(marker_frame, adata.var_names)
        summary, cell_order = core.summarize(adata, resolved, indices, "Celltype", "X", 10000.0)
    finally:
        adata.file.close()
    cell_order = ["MGC"] + [x for x in cell_order if x != "MGC"]
    summary["species"] = species
    return summary, resolved, cell_order


def plot_joint(summary, marker_tables, cell_orders, output_base, percentile, dpi,
               panel_width, panel_height):
    x_order = cell_orders[SPECIES[0]]
    if set(x_order) != set(cell_orders[SPECIES[1]]):
        raise ValueError("Human and mouse cell-type sets differ")
    x_lookup = {celltype: i for i, celltype in enumerate(x_order)}
    data = summary.copy()
    data["gene_key"] = data["gene"].str.upper()
    panel_genes = {sp: table.sort_values("rank").gene.astype(str).tolist()
                   for sp, table in marker_tables.items()}
    data = pd.concat([
        data[(data.species == sp) & data.gene_key.isin([g.upper() for g in panel_genes[sp]])].copy()
        for sp in SPECIES
    ], ignore_index=True)
    data["x"] = data.celltype.map(x_lookup)
    data["size"] = core.MIN_SIZE + data.fraction_expressed * (core.MAX_SIZE - core.MIN_SIZE)
    vmax = float(np.nanpercentile(data.mean_log1p_expression, percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    if panel_width <= 0 or panel_height <= 0:
        raise ValueError("Panel width and height must be positive")
    fig, ax = plt.subplots(figsize=(11.5, 8.6))
    # Keep the bubble-matrix region itself at exactly 620:490 by default;
    # titles, tick labels, colorbar, and legend are outside this ratio.
    ax.set_box_aspect(panel_height / panel_width)
    y_ticks, y_labels = [], []
    scatter = None
    panel_offsets = {SPECIES[0]: 0, SPECIES[1]: len(panel_genes[SPECIES[0]])}
    for sp in SPECIES:
        panel = data[data.species == sp].copy()
        genes = panel_genes[sp]
        offset = panel_offsets[sp]
        panel["y"] = panel.gene_key.map({gene.upper(): i + offset for i, gene in enumerate(genes)})
        data.loc[panel.index, "y"] = panel["y"]
        scatter = ax.scatter(panel.x, panel.y, s=panel["size"], c=panel.mean_log1p_expression,
                             cmap=core.CMAP, vmin=0, vmax=vmax, edgecolors="#4C4C4C", linewidths=.4)
        y_ticks.extend(range(offset, offset + len(genes)))
        y_labels.extend(genes)
    ax.set_yticks(y_ticks, y_labels)
    ax.invert_yaxis()
    ax.set_xticks(range(len(x_order)), x_order, rotation=90, ha="center", va="top")
    ax.set(xlabel="Cell type", ylabel="MGC marker gene",
           title="Human and mouse MGC marker expression | Top 10 per species")
    ax.text(-.095, .76, "Human", transform=ax.transAxes, rotation=90, ha="center", va="center", fontweight="bold")
    ax.text(-.095, .24, "Mouse", transform=ax.transAxes, rotation=90, ha="center", va="center", fontweight="bold")
    ax.grid(color="#E7E7E7", linewidth=.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.colorbar(scatter, ax=ax, fraction=.022, pad=.025).set_label("Mean log1p expression")
    fractions = [.25, .5, .75, 1]
    handles = [ax.scatter([], [], s=core.MIN_SIZE+f*(core.MAX_SIZE-core.MIN_SIZE),
                          facecolor="#D9C8A9", edgecolor="#4C4C4C", linewidth=.4) for f in fractions]
    ax.legend(handles, [f"{f:.0%}" for f in fractions], title="Cells expressing",
              frameon=False, bbox_to_anchor=(1.12, 1), loc="upper left", labelspacing=1.3)
    fig.subplots_adjust(left=.15, right=.80, bottom=.18, top=.92)
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return data


def main():
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be >= 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    genes, marker_tables = marker_union(args.rank_dir, args.top_n)
    summaries, resolved_tables, cell_orders = [], {}, {}
    for species in SPECIES:
        summary, resolved, order = species_summary(species, genes, args.h5ad_dir)
        summaries.append(summary)
        resolved_tables[species] = resolved
        cell_orders[species] = order
        print(f"[OK] {species}: {len(order)} cell types, "
              f"{(resolved.match_status != 'missing').sum()}/{len(genes)} genes matched", flush=True)
    output_base = args.output_dir / f"human_mouse_MGC_expression_dotplot_top{args.top_n}"
    plotting = plot_joint(pd.concat(summaries, ignore_index=True), marker_tables, cell_orders,
                          output_base, args.color_clip_percentile, args.dpi,
                          args.panel_width, args.panel_height)
    plotting.to_csv(output_base.with_name(output_base.name + "_expression_summary.csv"), index=False)
    pd.concat([table.assign(species=sp) for sp, table in marker_tables.items()]).to_csv(
        output_base.with_name(output_base.name + "_source_markers.csv"), index=False)
    pd.concat([table.assign(species=sp) for sp, table in resolved_tables.items()]).to_csv(
        output_base.with_name(output_base.name + "_gene_matches.csv"), index=False)
    print(f"Saved joint dotplot to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
