#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MAIN Astroglia validation figure: standardised model-vs-raw family trajectories.

WHY this is the main figure and the absolute-scale one is not
-------------------------------------------------------------
The model family score (centred log2 odds ratio over the model's Top-K peaks) and
the raw family score (chromVAR over all QC peaks) are different statistics on
different peak universes. Their magnitudes differ by roughly 40x, so drawing both
on one shared y-axis pushes every model curve onto y = 0 and hides the very
trajectory the figure exists to validate. Standardising each (species, family,
source) triple across the three windows puts both sources on a common scale of
SHAPE and DIRECTION, which is the actual claim:

    does the raw accessibility landscape move the same way as the model profile?

Reads only the prep tables -- no statistic is formed here.
See V2_PLAN section 15-3 rule 1.

    python script/s05_plot_family_trajectory_standardized.py --config config/analysis_v2.yaml
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

STAGE_ORDER = ("prenatal", "postnatal", "adult")
Y_LIMIT = 1.6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(C.CONFIG_DIR / "analysis_v2.yaml"))
    parser.add_argument("--in-dir", default=None)
    parser.add_argument("--families", nargs="*", default=list(S.FAMILY_ORDER))
    parser.add_argument("--species", nargs="*", default=None)
    parser.add_argument("--out-prefix", default=None)
    args = parser.parse_args()

    config = C.load_config(args.config)
    in_dir = Path(args.in_dir) if args.in_dir else C.results_dir(config, "05_astroglia_validation")
    source_path = in_dir / "family_trajectory_model_vs_raw_long.tsv"
    if not source_path.exists():
        raise FileNotFoundError(
            f"{source_path} missing; run script/s05_prep_family_trajectory.py first"
        )
    long = pd.read_csv(source_path, sep="\t")
    if "score_z" not in long.columns:
        raise ValueError(f"{source_path} lacks score_z; re-run the prep step")

    # the CSV round trip drops the categorical order; impose it explicitly or the
    # curves are drawn mirrored (see family_trajectory_style.order_stages)
    long = S.order_stages(long)
    long = long[long.family.isin(args.families)]
    species = args.species or [s for s in ("human", "marmoset", "mouse")
                               if s in set(long.species)]
    long = long[long.species.isin(species)]

    fig, axes = plt.subplots(1, len(species), figsize=(4.3 * len(species), 4.8),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, sp in zip(axes, species):
        block = long[long.species == sp]
        for family in args.families:
            fam = block[block.family == family]
            if fam.empty:
                continue
            color = S.family_color(family)
            for source in S.SOURCE_ORDER:
                line = fam[fam.source == source].sort_values("stage")
                if line.empty:
                    continue
                if len(line) != len(STAGE_ORDER):
                    raise ValueError(
                        f"{sp}/{family}/{source} carries {len(line)} stages, "
                        f"expected {len(STAGE_ORDER)}"
                    )
                linestyle, marker, _ = S.SOURCE_STYLE[source]
                ax.plot(range(len(STAGE_ORDER)), line.score_z.to_numpy(dtype=float),
                        linestyle=linestyle, marker=marker, color=color,
                        linewidth=1.8, markersize=6, markerfacecolor="white",
                        markeredgewidth=1.3, label=f"{family} ({source})")
        ax.set_title(sp, fontsize=11, color=S.INK)
        ax.set_xticks(range(len(STAGE_ORDER)))
        ax.set_xticklabels(STAGE_ORDER, rotation=20, ha="right")
        ax.set_ylim(-Y_LIMIT, Y_LIMIT)
        ax.axhline(0, color=S.GRID, linewidth=0.8)
        S.style_axes(ax)

    axes[0].set_ylabel("z-scored family trajectory", fontsize=10, color=S.INK)

    # Layout is done in figure coordinates on purpose. With two legends plus a
    # two-line caption, `bbox_inches="tight"` alone lets the legends land on top
    # of the title and of the x tick labels -- the first draft did exactly that.
    # Reserve the bands explicitly, then place each legend inside its own band.
    fig.subplots_adjust(left=0.06, right=0.99, top=0.80, bottom=0.36, wspace=0.08)

    fig.suptitle(
        "Astroglia: model-derived and raw chromVAR family trajectories, "
        "standardised within each species x family x source",
        fontsize=11.5, color=S.INK, y=0.97,
    )

    # band 1 (y ~ 0.86-0.93): family identity, by colour
    family_handles = [
        plt.Line2D([], [], color=S.family_color(f), linewidth=2.5, label=f)
        for f in args.families
    ]
    fig.legend(handles=family_handles, loc="center", ncol=len(args.families),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.90))

    # band 2 (y ~ 0.08-0.16): source identity, by line style + marker
    # the cut-off is the ARM's own, read from its config: this figure is drawn once
    # per arm, so a hardcoded "Top500" would caption the Top1000 and Top2000 arms'
    # kept figures with a cut-off their numbers did not come from.
    cutoff = C.topk_label(config)
    handles = [
        plt.Line2D([], [], color=S.MUTED, linestyle=S.SOURCE_STYLE[s][0],
                   marker=S.SOURCE_STYLE[s][1], markersize=6,
                   markerfacecolor="white", markeredgewidth=1.2,
                   label=S.source_label(s, cutoff))
        for s in S.SOURCE_ORDER
    ]
    fig.legend(handles=handles, loc="center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.235))

    fig.text(
        0.5, 0.185,
        "Each 3-point trajectory (prenatal, postnatal, adult) is z-scored within its own "
        "species x family x source, so the two sources are compared by SHAPE and DIRECTION, "
        "not magnitude.\nModel and raw chromVAR are different statistics on different peak "
        "universes; their absolute values are not comparable and are shown separately in "
        "the absolute-scale companion figure.",
        ha="center", va="top", fontsize=8, color=S.MUTED,
    )

    prefix = Path(args.out_prefix) if args.out_prefix else in_dir / "family_trajectory_model_vs_raw_standardized"
    written = S.save(fig, prefix, dpi=int(config.get("plotting", {}).get("dpi", 200)),
                     tight=False)
    for path in written:
        C.log("wrote", path.name)

    # the plotted values, so the figure is traceable to a table
    table = long[["species", "family", "analysis_level", "source", "stage", "score", "score_z"]]
    out_tsv = prefix.with_suffix(".tsv")
    table.to_csv(out_tsv, sep="\t", index=False)
    C.log("wrote", out_tsv.name, f"({len(table)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
