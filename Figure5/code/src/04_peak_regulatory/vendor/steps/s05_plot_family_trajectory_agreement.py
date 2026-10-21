#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SUPPLEMENTARY: how well the model and raw trajectories agree, in one view.

WHY: the overlay figure lets a reader see agreement family by family, but not
summarise it. This plate puts the two summary statistics side by side, one row per
family and one column per species:

    left   rho          the 3-point Spearman between the model and raw trajectory
    right  concordance  over the two consecutive transitions, the fraction whose
                        sign agrees (0, 0.5 or 1)

Both are read from the prep table -- nothing is computed here.

CAVEAT that the figure states: a 3-point Spearman has a degenerate p-value, so
the upstream table flags `p_is_reliable = False` for every per-species row. The
number is still a useful descriptive summary of shape agreement; it is not a
significance test. The pooled `species_x_stage` rows have n = 9 and are the only
ones whose p-value is marked reliable; they are drawn as a separate column and
labelled as pooled.

    python script/s05_plot_family_trajectory_agreement.py --config config/analysis_v2.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import family_trajectory_style as S  # noqa: E402

POOLED = "species_x_stage"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(C.CONFIG_DIR / "analysis_v2.yaml"))
    parser.add_argument("--in-dir", default=None)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--species", nargs="*", default=None)
    parser.add_argument("--out-prefix", default=None)
    args = parser.parse_args()

    config = C.load_config(args.config)
    in_dir = Path(args.in_dir) if args.in_dir else C.results_dir(config, "05_astroglia_validation")
    path = in_dir / "family_trajectory_agreement.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; run script/s05_prep_family_trajectory.py first"
        )
    table = pd.read_csv(path, sep="\t")

    families = args.families or [f for f in S.FAMILY_ORDER
                                 if f in set(table.family.dropna())]
    per_species = table[table.species != POOLED]
    species = args.species or [s for s in ("human", "marmoset", "mouse")
                               if s in set(per_species.species)]
    pooled = table[table.species == POOLED].set_index("family")

    rho = per_species.pivot_table(index="family", columns="species", values="rho")
    rho = rho.reindex(index=families, columns=species)
    conc = per_species.pivot_table(index="family", columns="species", values="concordance")
    conc = conc.reindex(index=families, columns=species)

    fig, axes = plt.subplots(1, 2, figsize=(4.6 + 1.5 * len(species), 0.62 * len(families) + 4.2),
                             gridspec_kw={"width_ratios": [1.0, 1.0]})

    # ---- left: trajectory correlation heatmap -------------------------------
    ax = axes[0]
    values = rho.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(species)))
    ax.set_xticklabels(species, rotation=20, ha="right")
    ax.set_yticks(range(len(families)))
    # compact row label: the long parenthetical pushed "Nuclear receptor" off the
    # left edge of the figure
    labels = [f"{f}  [{pooled.rho.get(f, np.nan):.2f}]" if f in pooled.index else f
              for f in families]
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(families)):
        for j in range(len(species)):
            value = values[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(value) > 0.6 else S.INK)
    ax.set_title("trajectory correlation\n(3-point Spearman, model vs raw)", fontsize=10)
    S.style_axes(ax, grid_axis="none")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).set_label("rho", fontsize=9)

    # ---- right: direction concordance dot plot ------------------------------
    ax = axes[1]
    for i, family in enumerate(families):
        for j, sp in enumerate(species):
            value = conc.loc[family, sp] if family in conc.index else np.nan
            if not np.isfinite(value):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color=S.MUTED)
                continue
            ax.scatter(j, i, s=90 + 380 * value, color=S.family_color(family),
                       alpha=0.85, edgecolor="white", linewidth=1.2, zorder=3)
            ax.text(j, i, f"{value:g}", ha="center", va="center", fontsize=8.5,
                    color="white", zorder=4)
    ax.set_xlim(-0.6, len(species) - 0.4)
    ax.set_ylim(len(families) - 0.5, -0.5)
    ax.set_xticks(range(len(species)))
    ax.set_xticklabels(species, rotation=20, ha="right")
    ax.set_yticks([])
    ax.set_title("direction concordance\n(agreement over the 2 transitions)", fontsize=10)
    S.style_axes(ax, grid_axis="none")
    markers = [plt.Line2D([], [], marker="o", linestyle="", markersize=np.sqrt(90 + 380 * v) / 2,
                          markerfacecolor=S.MUTED, markeredgecolor="white",
                          label=f"{int(round(2*v))}/2 agree")
               for v in (0.0, 0.5, 1.0)]
    ax.legend(handles=markers, loc="upper center", bbox_to_anchor=(0.5, -0.11),
              frameon=False, fontsize=8.5, ncol=3, columnspacing=1.0,
              handletextpad=0.4)

    fig.subplots_adjust(left=0.235, right=0.975, top=0.84, bottom=0.40, wspace=0.34)
    fig.suptitle("Astroglia: agreement between model-derived and raw chromVAR "
                 "family trajectories", fontsize=11.5, color=S.INK, y=0.95)
    fig.text(
        0.5, 0.255,
        "rho = Spearman correlation of the two 3-point trajectories\n"
        "(prenatal, postnatal, adult) for one species and family.\n"
        "The bracketed value in each row label is pooled over all 9\n"
        "species x stage points, the only scope whose p-value the\n"
        "upstream table marks reliable.\n"
        "A 3-point correlation has a degenerate p-value, so rho here is\n"
        "a descriptive summary of shape agreement, not a significance test.\n"
        "Concordance counts, over the two consecutive transitions, how\n"
        "many move the same way in both sources; a tie is excluded.",
        ha="center", va="top", fontsize=8.5, color=S.MUTED,
    )

    prefix = Path(args.out_prefix) if args.out_prefix else \
        in_dir / "family_trajectory_agreement_summary"
    # PDF only: the raster copy of this figure is not part of the paper's
    # deliverable set, so it is not written.
    for written in S.save(fig, prefix, formats=("pdf",),
                          dpi=int(config.get("plotting", {}).get("dpi", 200)),
                          tight=False):
        C.log("wrote", written.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
