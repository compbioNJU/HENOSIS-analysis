#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Purkinje / Oligo model-vs-chromVAR validation plate, one design language.

WHY this replaces the two figures the steps used to emit
--------------------------------------------------------
Those figures had three defects, all of them presentational but none of them
cosmetic:

  1. the module panels put `raw chromVAR family score` (about 4-30) and
     `model family score` (about 0-2) on ONE y-axis, so every model marker was
     pinned to zero beside tall raw bars. A module with 6 of 6 species agreeing
     therefore read as "the raw signal is strong and the model sees nothing".
     The two are different statistics on different peak universes; they get
     separate colour scales here and agreement is reported as a sign count.
  2. the per-species scatter carried an OLS regression line while the reported
     statistic was Spearman. Those are different assumptions, and the line invites
     the reader to check linearity. It is gone; the rank statistic is annotated.
  3. all 879 motifs were drawn in one colour, so the panel could not show which
     of them belong to the pre-registered modules.

WHY there is no z-scoring here, unlike the Astroglia trajectory figure
---------------------------------------------------------------------
Astroglia has a developmental axis, so standardising three time points compares
trajectory shape. Purkinje and Oligo have no axis: the questions are the
cross-modal rank association across 879 motifs (Spearman, scale free already) and
the DIRECTION agreement of the modules. Z-scoring would answer neither.

Panels
    A  per-species scatter, unannotated motifs greyed, panel modules coloured,
       x = 0 and y = 0 references, Spearman rho annotated
    B  the per-species rho as a forest/dot, so the reader does not have to read
       six numbers out of six scatter plots
    C  module validation matrix: model F_abs | chromVAR C | direction agreement,
       three blocks with their own scales

Reads only the prep tables -- no statistic is formed here.

    python script/plot_celltype_validation.py --config config/analysis_v2.yaml --celltype Oligo
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
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import family_trajectory_style as S  # noqa: E402

STEP_DIR = {"Purkinje": "07_purkinje_validation", "Oligo": "09_oligo_validation"}
GREY = "#BFBFBF"
INK = S.INK
MUTED = S.MUTED

# the scatter's module colours reuse the project palette where a module name is
# shared with Astroglia (SOX, NFI, NF-Y, RFX) so the reader keeps one colour code
EXTRA_COLORS = {
    "Homeodomain_total": "#6A3D9A",
    "Homeodomain_nonPOU": "#9370DB",
    "POU": "#CAB2D6",
    "TFAP2": "#B15928",
    "bHLH_Ebox": "#1F78B4",
    "NKX": "#33A02C",
    "POU3": "#CC79A7",
    "unannotated": GREY,
}

# draw order in panel A: a derived aggregate is drawn under its subdivisions so
# the subdivisions stay visible where they overlap
DRAW_ORDER = ("Homeodomain_nonPOU", "POU", "Homeodomain_total")


def module_color(module: str) -> str:
    return EXTRA_COLORS.get(module, S.family_color(module))


def shade_concordant_quadrants(ax, warm: str, cool: str, alpha: float = 0.06) -> None:
    """Tint Q1 (both positive) and Q3 (both negative) of a signed scatter.

    The four quadrants are already implied by the two zero lines; tinting the two
    CONCORDANT ones makes the sign structure readable at a glance, and using the
    same warm/cool pair as the agreement panel below keeps one colour language for
    the whole figure (warm = concordant enrichment, cool = concordant depletion).

    The tint is auxiliary: it marks sign, not significance. A motif that lands in
    a shaded quadrant only satisfies sign(model) == sign(chromVAR); it is not
    evidence of enrichment, significance or TF activity, and most of the 879
    motifs sit close enough to the origin that their sign is barely defined. The
    primary statistic of this panel stays the Spearman rho printed in the corner.
    """
    from matplotlib.patches import Rectangle

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    for (left, bottom, width, height), colour in (
        ((0.0, 0.0, max(x1, 0.0), max(y1, 0.0)), warm),
        ((min(x0, 0.0), min(y0, 0.0), -min(x0, 0.0), -min(y0, 0.0)), cool),
    ):
        if width > 0 and height > 0:
            ax.add_patch(Rectangle((left, bottom), width, height, facecolor=colour,
                                   alpha=alpha, linewidth=0, zorder=0))
    ax.text(0.97, 0.95, "same sign", transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color=MUTED, style="italic", zorder=5)
    ax.text(0.03, 0.05, "same sign", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6.5, color=MUTED, style="italic", zorder=5)


def panel_a(fig, spec, long, species, rho_by_species, sc_by_species, celltype, cutoff):
    inner = spec.subgridspec(2, 3, hspace=0.46, wspace=0.32)
    axes = [fig.add_subplot(inner[i // 3, i % 3]) for i in range(len(species))]
    modules = [m for m in DRAW_ORDER if m in set(long.module)] + \
              sorted(m for m in set(long.module) if m not in DRAW_ORDER and m != "unannotated")

    for position, (ax, sp) in enumerate(zip(axes, species)):
        block = long[long.species == sp]
        grey = block[~block.in_panel]
        ax.scatter(grey.model_log2_or, grey.chromvar_C, s=5, c=GREY, alpha=0.45,
                   linewidths=0, zorder=2, rasterized=True)
        for module in modules:
            hits = block[block.in_panel & (block.module == module)]
            if hits.empty:
                continue
            ax.scatter(hits.model_log2_or, hits.chromvar_C, s=17,
                       c=module_color(module), alpha=0.95, linewidths=0.3,
                       edgecolors="white", zorder=4, label=module)
        ax.set_xlim(auto=None)
        ax.set_ylim(auto=None)
        ax.margins(0.06)
        shade_concordant_quadrants(ax, "#B2182B", "#2166AC")
        ax.axhline(0, color=S.INK, linewidth=0.7, zorder=1, alpha=0.55)
        ax.axvline(0, color=S.INK, linewidth=0.7, zorder=1, alpha=0.55)
        S.style_axes(ax, grid_axis="both")
        ax.set_title(sp, fontsize=10, color=INK)

        value = rho_by_species.get(sp, np.nan)
        if np.isfinite(value):
            ax.text(0.04, 0.95, f"Spearman rho = {value:.2f}", transform=ax.transAxes,
                    fontsize=8.5, color=INK, va="top", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
            ax.text(0.04, 0.865, f"n = {len(block)} motifs", transform=ax.transAxes,
                    fontsize=7.5, color=MUTED, va="top", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
        sc = sc_by_species.get(sp, {})
        if np.isfinite(sc.get("sc_model_fdr", np.nan)):
            ax.text(0.04, 0.79,
                    f"same sign (model FDR<0.05): "
                    f"{int(sc['n_same_sign_model_fdr'])}/{int(sc['n_motifs_model_fdr'])}",
                    transform=ax.transAxes, fontsize=7, color=MUTED, va="top", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
        ax.tick_params(labelsize=8)
        # the axis meaning is stated once per row/column, not on every panel
        if position % 3 == 0:
            ax.set_ylabel("chromVAR contrast\n(target - other cell types)",
                          fontsize=8, color=INK)
        if position >= len(species) - 3 or position == len(species) - 1:
            ax.set_xlabel(f"Model motif enrichment\n({cutoff} vs matched, log2 OR)",
                          fontsize=8, color=INK)
    return axes, modules


def panel_b(ax, similarity, species):
    order = similarity.set_index("species").reindex(species).dropna(subset=["rho"])
    y = np.arange(len(order))
    colors = ["#0072B2" if v > 0 else "#D55E00" for v in order.rho]
    ax.hlines(y, 0, order.rho, color=colors, linewidth=2.2, zorder=2)
    ax.scatter(order.rho, y, s=55, c=colors, zorder=3, edgecolors="white", linewidths=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(order.index, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color=S.GRID, linewidth=0.9)
    S.style_axes(ax, grid_axis="x")
    ax.set_xlabel("Spearman rho (879 motifs)", fontsize=9)
    n_pos = int((order.rho > 0).sum())
    ax.set_title(f"species-wise concordance: {n_pos} of {len(order)} species with rho > 0",
                 fontsize=10, color=INK)


def panel_c(fig, spec, support, summary, species, celltype, cutoff):
    inner = spec.subgridspec(1, 3, width_ratios=[1.0, 1.0, 0.72], wspace=0.34)
    ax_model = fig.add_subplot(inner[0, 0])
    ax_raw = fig.add_subplot(inner[0, 1])
    ax_agree = fig.add_subplot(inner[0, 2])

    order = list(summary.module)
    pivot_model = support.pivot_table(index="module", columns="species",
                                      values="model_F_abs").reindex(order, columns=species)
    pivot_raw = support.pivot_table(index="module", columns="species",
                                    values="chromvar_C").reindex(order, columns=species)
    agree = summary.set_index("module").reindex(order)
    levels = support.drop_duplicates("module").set_index("module").module_level.reindex(order)

    for block_index, (ax, pivot, title) in enumerate(
        ((ax_model, pivot_model, f"model F_abs\n({cutoff} motif enrichment)"),
         (ax_raw, pivot_raw, "chromVAR C\n(all QC peaks)"))
    ):
        values = pivot.to_numpy(dtype=float)
        # An empty matrix means the upstream recurrent-module step was not run, so
        # this cell type has no module layer to draw. nanmax has no identity on a
        # zero-size array, so the panel is drawn blank instead of crashing.
        limit = float(np.nanmax(np.abs(values))) if values.size else 1.0
        limit = limit or 1.0
        image = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(species)))
        ax.set_xticklabels(species, rotation=35, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(order)))
        # the two matrices share their rows, so only the left one carries the
        # labels; labelling both clipped the longer module names off the figure
        if block_index == 0:
            ax.set_yticklabels([f"{m} ({levels.get(m, '')})" for m in order], fontsize=8.5)
        else:
            ax.set_yticklabels([])
        ax.set_title(f"{title}\n(own colour scale)", fontsize=9.5, color=INK)
        S.style_axes(ax, grid_axis="none")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        for i in range(len(order)):
            for j in range(len(species)):
                value = values[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(value) > 0.55 * limit else INK)

    # A bare "sign agrees" count is not enough: `bHLH_Ebox` agrees in 6 of 6
    # species while BOTH statistics are negative there, i.e. the module is
    # consistently DEPLETED. Reading that as support for an oligo bHLH program
    # would be wrong, so the bar is split by the direction the two sources agree
    # ON. "Both negative" is a real, reproducible depletion result -- it is just
    # not evidence of enrichment.
    y = np.arange(len(order))
    n_species = agree.n_species.to_numpy(dtype=float)
    both_pos, both_neg = [], []
    for module in order:
        rows = support[support.module == module]
        both_pos.append(int(((rows.model_sign > 0) & (rows.chromvar_sign > 0)).sum()))
        both_neg.append(int(((rows.model_sign < 0) & (rows.chromvar_sign < 0)).sum()))
    both_pos = np.array(both_pos, dtype=float) / np.maximum(n_species, 1)
    both_neg = np.array(both_neg, dtype=float) / np.maximum(n_species, 1)
    discordant = np.clip(1.0 - both_pos - both_neg, 0.0, 1.0)

    ax_agree.barh(y, both_pos, color="#B2182B", alpha=0.92, zorder=2,
                  label="both positive (enriched)")
    ax_agree.barh(y, both_neg, left=both_pos, color="#2166AC", alpha=0.92, zorder=2,
                  label="both negative (depleted)")
    ax_agree.barh(y, discordant, left=both_pos + both_neg, color=GREY, alpha=0.75,
                  zorder=2, label="discordant")
    ax_agree.set_yticks(y)
    ax_agree.set_yticklabels([])
    ax_agree.invert_yaxis()
    ax_agree.set_xlim(0, 1.30)
    ax_agree.set_xticks([0, 0.5, 1.0])
    ax_agree.set_xlabel("fraction of species", fontsize=9)
    S.style_axes(ax_agree, grid_axis="x")
    ax_agree.set_title("sign agreement, split by direction\n(* secondary / comparator)",
                       fontsize=9.5, color=INK)
    ax_agree.legend(loc="upper center", bbox_to_anchor=(0.5, -0.155), frameon=False,
                    fontsize=7.5, ncol=2, handlelength=1.2, columnspacing=1.0)
    total = (both_pos + both_neg + discordant) * 0
    for i in range(len(order)):
        label = f"{int(round((both_pos[i] + both_neg[i]) * n_species[i]))}/{int(n_species[i])}"
        if not agree.gating.iloc[i]:
            label += "*"
        ax_agree.text(1.06, i, label, va="center", ha="left", fontsize=8.5, color=INK)
    return order


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(C.CONFIG_DIR / "analysis_v2.yaml"))
    parser.add_argument("--celltype", required=True, choices=sorted(STEP_DIR))
    parser.add_argument("--in-dir", default=None)
    parser.add_argument("--out-prefix", default=None)
    args = parser.parse_args()

    config = C.load_config(args.config)
    celltype = args.celltype
    in_dir = Path(args.in_dir) if args.in_dir else C.results_dir(config, STEP_DIR[celltype])
    stem = celltype.lower()

    long = pd.read_csv(in_dir / f"{stem}_model_vs_chromvar.tsv", sep="\t")
    support = pd.read_csv(in_dir / f"{stem}_module_support.tsv", sep="\t")
    summary = pd.read_csv(in_dir / f"{stem}_module_support_summary.tsv", sep="\t")
    similarity = pd.read_csv(in_dir / "model_vs_raw_similarity.tsv", sep="\t")
    sc_path = in_dir / f"{stem}_sign_concordance.tsv"
    sc_by_species = {}
    if sc_path.exists():
        sc_frame = pd.read_csv(sc_path, sep="\t")
        sc_by_species = {row.species: row._asdict() for row in sc_frame.itertuples(index=False)}

    rho_by_species = dict(zip(similarity.species, similarity.rho))
    # The species of this plate are the ELIGIBLE ones, i.e. those the validation
    # step scored. A species can carry a raw chromVAR profile while being excluded
    # from the model comparison -- bonobo Purkinje is dropped at 4 cells / 298
    # peaks -- and drawing it produces a blank panel whose model log2 OR is all
    # zero, which reads as a real null result rather than as an absent model.
    # the validation step writes a row for an INELIGIBLE species too, carrying a
    # NaN rho (bonobo Purkinje). A NaN is not a score, so such a species is not a
    # panel member: drawing it would put an empty axes with "rho = 0.00" beside
    # real ones and read as a null result.
    scored = {sp for sp, value in rho_by_species.items() if np.isfinite(value)}
    species = [s for s in C.SPECIES_ORDER if s in scored]
    unscored = sorted(set(rho_by_species) - scored)
    if unscored:
        C.log("not drawn (ineligible, no model-vs-raw score):", unscored)
    dropped = sorted((set(long.species) | set(rho_by_species)) - set(species))
    if dropped:
        C.log("not drawn (no eligible model comparison):", dropped)
        long = long[long.species.isin(species)]

    fig = plt.figure(figsize=(15.0, 14.6))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[2.25, 0.95, 1.5], hspace=0.46)
    # The cut-off is the ARM's own, read from its config: this plate is drawn once per
    # arm, so a hardcoded "Top500" would caption the Top1000 and Top2000 arms wrongly.
    cutoff = C.topk_label(config)
    axes_a, modules = panel_a(fig, gs[0], long, species, rho_by_species,
                              sc_by_species, celltype, cutoff)
    panel_b(fig.add_subplot(gs[1]), similarity, species)
    panel_c(fig, gs[2], support, summary, species, celltype, cutoff)

    fig.subplots_adjust(left=0.185, right=0.985, top=0.885, bottom=0.205)
    fig.suptitle(
        f"{celltype}: within-species concordance between model-derived motif "
        f"enrichment and full-peak chromVAR accessibility",
        fontsize=12.5, color=INK, y=0.968,
    )

    handles = [Line2D([], [], marker="o", linestyle="", markersize=6,
                      markerfacecolor=module_color(m), markeredgecolor="white", label=m)
               for m in modules]
    handles.append(Line2D([], [], marker="o", linestyle="", markersize=4,
                          markerfacecolor=GREY, markeredgecolor="none",
                          label="other catalog motifs"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925),
               ncol=min(len(handles), 5), frameon=False, fontsize=9)

    fig.text(
        0.5, 0.100,
        f"Axes. x = model motif enrichment: log2 odds ratio of the motif in the model's {cutoff} "
        f"peaks against its GC/length matched non-{cutoff} background (x > 0 = enriched in {cutoff}).\n"
        f"y = full-peak chromVAR contrast: {celltype} minus the equal-weighted mean of the other "
        "cell types, so y > 0 = more accessible in the target cell type.\n"
        "The two are different statistics on different peak universes; the scatter therefore "
        "shows their RANK ASSOCIATION and no line is fitted. The primary statistic is the "
        "Spearman rho in each panel.\n"
        "Shaded quadrants mark CONCORDANT SIGNS only; quadrant membership is descriptive and "
        "does not imply statistical significance or TF activity.\n"
        "C: model and chromVAR magnitudes are drawn as separate matrices with their own colour "
        "scales, never on one axis -- they differ by roughly an order of magnitude.\n"
        "Gate P3/O3 is decided on the 879-motif rho in panel B; the module panel interprets it "
        "and a secondary or comparator module may not gate.\n"
        "C right: agreement is split by the direction the sources agree ON -- 'both negative' is "
        "a reproducible depletion result, not evidence of enrichment.",
        ha="center", va="top", fontsize=8, color=MUTED,
    )

    prefix = Path(args.out_prefix) if args.out_prefix else \
        in_dir / f"{stem}_motif_module_validation"
    for path in S.save(fig, prefix, dpi=int(config.get("plotting", {}).get("dpi", 200)),
                       tight=False):
        C.log("wrote", path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
