"""Draw the existing B-cell peak-score quartile panel with explicit group sizes.

Reads the saved per-peak and statistics tables; does not recompute groups or tests.
All output names carry the ``_with_n`` suffix to preserve original figures.
"""

from pathlib import Path
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path(os.environ.get("ANALYSIS_RESULTS_DIR", Path(__file__).resolve().parents[3] / "results")) / "Figure6/pbmc/peak_validation"
ORDER = ["Q1", "Q2", "Q3", "Q4"]
COLORS = ["#D9D9D9", "#AFC7D8", "#6FA3B8", "#2F6F7E"]


def stars(p):
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def main():
    table = pd.read_csv(OUT / "top5000_peak_validation_table.csv")
    stats = pd.read_csv(OUT / "peak_score_quartile_statistics.csv")
    groups = [
        table.loc[table.score_quartile.eq(q), "delta_accessibility"].dropna().to_numpy()
        for q in ORDER
    ]
    counts = [len(x) for x in groups]
    assert sum(counts) == len(table), "Quartile labels or accessibility values are missing"

    summary = pd.DataFrame(
        {
            "score_quartile": ORDER,
            "n_peaks": counts,
            "min_score": [table.loc[table.score_quartile.eq(q), "score"].min() for q in ORDER],
            "max_score": [table.loc[table.score_quartile.eq(q), "score"].max() for q in ORDER],
        }
    )

    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    violin = ax.violinplot(groups, positions=range(1, 5), widths=0.78,
                           showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(violin["bodies"], COLORS):
        body.set_facecolor(color)
        body.set_edgecolor("#4A4A4A")
        body.set_linewidth(0.7)
        body.set_alpha(0.9)

    box = ax.boxplot(
        groups, positions=range(1, 5), widths=0.22, patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1F1F1F", "linewidth": 1.2},
        whiskerprops={"color": "#4A4A4A", "linewidth": 0.8},
        capprops={"color": "#4A4A4A", "linewidth": 0.8},
        boxprops={"color": "#4A4A4A", "linewidth": 0.8},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(0.8)

    for xpos, values in enumerate(groups, 1):
        sampled = rng.choice(values, min(500, len(values)), replace=False)
        ax.scatter(
            np.full(len(sampled), xpos) + rng.normal(0, 0.055, len(sampled)),
            sampled, s=5, color="#6D6D6D", alpha=0.24, linewidths=0,
            rasterized=True,
        )

    ax.axhline(0, color="#777777", lw=0.7, ls=(0, (3, 2)))
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels([f"{q}\nn={n:,}" for q, n in zip(ORDER, counts)])
    ax.set_xlabel("Peak-score quartile")
    ax.set_ylabel(r"B-cell specificity, $\Delta$ accessibility")
    ax.set_title("Peak score tracks B-cell-specific accessibility", loc="left", pad=12)

    ymax = max(np.percentile(x, 99) for x in groups)
    ymin = min(np.percentile(x, 1) for x in groups)
    step = max((ymax - ymin) * 0.10, 0.005)
    base = ymax + step
    for j, ref in enumerate(ORDER[:3]):
        row = stats.loc[stats.reference.eq(ref)].iloc[0]
        x1, x2, y = j + 1, 4, base + j * step
        ax.plot([x1, x1, x2, x2], [y, y + step * 0.22, y + step * 0.22, y],
                color="#4A4A4A", lw=0.7)
        ax.text((x1 + x2) / 2, y + step * 0.27, stars(float(row.padj)),
                ha="center", va="bottom", fontsize=8)

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E8E8E8", lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()

    base = OUT / "Figure_peak_rank_quartiles_with_n"
    for suffix, kwargs in [("pdf", {}), ("svg", {}), ("png", {"dpi": 600})]:
        path = base.with_suffix("." + suffix)
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, bbox_inches="tight", **kwargs)
    summary_path = OUT / "peak_score_quartile_counts_with_n.csv"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    summary.to_csv(summary_path, index=False)
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
