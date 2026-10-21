#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SUPPLEMENTARY: family trajectories on their native scales, one source per row.

WHY this is supplementary and drawn this way
--------------------------------------------
Model family score and raw chromVAR family score are different statistics on
different peak universes; their magnitudes differ by roughly 40x. The previous
version drew both on one shared y-axis, which flattened every model curve onto
y = 0. Here the two sources never share an axis: each species gets a column, the
top row carries the raw chromVAR trajectories and the bottom row the model ones,
each with its own y scale. That keeps the native magnitudes readable while making
the "do not compare these in magnitude" point structurally rather than in a note.

The standardised companion figure is the one that supports the validation claim;
this one shows what each source looks like on its own terms.

    python script/s05_plot_family_trajectory_absolute.py --config config/analysis_v2.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import family_trajectory_style as S  # noqa: E402

# top row = raw, bottom row = model: the raw profile is the validation reference
ROW_ORDER = ("raw_chromVAR", "model")


def row_titles(cutoff: str) -> dict:
    """Row titles; the model row names this arm's own cut-off.

    The cut-off comes from the arm's config because this figure is drawn once per
    arm -- a hardcoded "Top500" would caption the Top1000 and Top2000 arms' kept
    figures with a cut-off their numbers did not come from.
    """
    return {"raw_chromVAR": "raw chromVAR (all QC peaks)",
            "model": f"model ({cutoff} motif enrichment)"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(C.CONFIG_DIR / "analysis_v2.yaml"))
    parser.add_argument("--in-dir", default=None)
    parser.add_argument("--families", nargs="*", default=list(S.FAMILY_ORDER))
    parser.add_argument("--species", nargs="*", default=None)
    parser.add_argument("--out-prefix", default=None)
    args = parser.parse_args()

    config = C.load_config(args.config)
    # the cut-off is the ARM's own, read from its config: this figure is drawn once
    # per arm, so a hardcoded "Top500" would caption the Top1000 and Top2000 arms
    # with a cut-off their numbers did not come from.
    titles = row_titles(C.topk_label(config))
    in_dir = Path(args.in_dir) if args.in_dir else C.results_dir(config, "05_astroglia_validation")
    source_path = in_dir / "family_trajectory_model_vs_raw_long.tsv"
    if not source_path.exists():
        raise FileNotFoundError(
            f"{source_path} missing; run script/s05_prep_family_trajectory.py first"
        )
    long = S.order_stages(pd.read_csv(source_path, sep="\t"))
    long = long[long.family.isin(args.families)]
    species = args.species or [s for s in ("human", "marmoset", "mouse")
                               if s in set(long.species)]
    long = long[long.species.isin(species)]

    fig, axes = plt.subplots(2, len(species), figsize=(4.3 * len(species), 7.4),
                             sharex=True, squeeze=False)

    for column, sp in enumerate(species):
        block = long[long.species == sp]
        for row, source in enumerate(ROW_ORDER):
            ax = axes[row][column]
            for family in args.families:
                line = block[(block.family == family) & (block.source == source)] \
                    .sort_values("stage")
                if line.empty:
                    continue
                linestyle, marker, _ = S.SOURCE_STYLE[source]
                ax.plot(range(len(S.STAGE_ORDER)), line.score.to_numpy(dtype=float),
                        linestyle=linestyle, marker=marker, color=S.family_color(family),
                        linewidth=1.8, markersize=5.5, markerfacecolor="white",
                        markeredgewidth=1.2)
            ax.axhline(0, color=S.GRID, linewidth=0.8)
            S.style_axes(ax)
            ax.set_title(f"{sp} -- {titles[source]}" if row == 0 else titles[source],
                         fontsize=9.5, color=S.INK)
            if column == 0:
                ax.set_ylabel("family score (centered, native scale)", fontsize=9, color=S.INK)
            # the two rows have different native magnitudes on purpose: say so on
            # the axis rather than letting the reader assume a shared scale
            ax.text(0.02, 0.94, f"y range {ax.get_ylim()[0]:.2f} .. {ax.get_ylim()[1]:.2f}",
                    transform=ax.transAxes, fontsize=7.5, color=S.MUTED, va="top")
            if row == len(ROW_ORDER) - 1:
                ax.set_xticks(range(len(S.STAGE_ORDER)))
                ax.set_xticklabels(S.STAGE_ORDER, rotation=20, ha="right")

    # The column titles sit just above the axes, so the family legend needs its
    # own band above them -- at top=0.88 it landed on top of the titles.
    fig.subplots_adjust(left=0.075, right=0.985, top=0.78, bottom=0.24,
                        hspace=0.42, wspace=0.18)

    fig.suptitle("Astroglia family trajectories on their native scales, "
                 "one source per row", fontsize=11.5, color=S.INK, y=0.955)
    family_handles = [
        plt.Line2D([], [], color=S.family_color(f), linewidth=2.5, label=f)
        for f in args.families
    ]
    fig.legend(handles=family_handles, loc="center", ncol=len(args.families),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.865))
    fig.text(
        0.5, 0.155,
        "Absolute values are shown on their native scales and are NOT directly\n"
        "comparable in magnitude between the two rows: model enrichment and\n"
        "chromVAR deviation are different statistics on different peak universes\n"
        "(they differ by roughly 40x here).\n"
        "Interpretation should focus on the within-source temporal ordering; the\n"
        "standardised companion figure is the one that compares the two sources.",
        ha="center", va="top", fontsize=8.5, color=S.MUTED,
    )

    prefix = Path(args.out_prefix) if args.out_prefix else \
        in_dir / "family_trajectory_model_vs_raw_absolute"
    for path in S.save(fig, prefix, dpi=int(config.get("plotting", {}).get("dpi", 200)),
                       tight=False):
        C.log("wrote", path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
