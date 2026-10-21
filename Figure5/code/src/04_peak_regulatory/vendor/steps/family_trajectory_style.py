"""Shared visual vocabulary for the Astroglia family-trajectory figures.

WHY a shared module: three figures (standardised overlay, absolute two-row, and
the agreement summary) show the same families. If each picked its own colours the
reader would have to relearn the legend between panels, so the mapping lives in
one place and the three scripts import it.

The encoding is deliberately two-dimensional and orthogonal:

    family -> COLOUR          source -> LINE STYLE + MARKER

so a reader can follow one family across both sources, or compare the two sources
within one family, without either channel being overloaded. Source is never
encoded by colour.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# Colour-blind safe; family identity is stable across every figure in the project.
FAMILY_COLORS = {
    "RFX": "#0072B2",              # blue
    "NF-Y": "#009E73",             # green
    "SOX": "#E69F00",              # orange
    "NFI": "#D55E00",              # red
    "Nuclear receptor": "#7A7A7A",  # grey: the secondary family
    "NuclearReceptor": "#7A7A7A",
}

# source -> (linestyle, marker, label). The label stored here is the
# arm-INDEPENDENT one: the model source is labelled per arm by source_label(),
# because v3 draws this figure family once per Top-K arm and a cut-off baked into
# a shared constant would caption the other arms' figures wrongly.
SOURCE_STYLE = {
    "model": ("-", "o", "model (top-ranked motif enrichment)"),
    "raw_chromVAR": ("--", "s", "raw chromVAR (all QC peaks)"),
}


def source_label(source: str, cutoff: str) -> str:
    """The legend label for one source; the model one names THIS arm's cut-off.

    The cut-off cannot live in SOURCE_STYLE: this module is shared by the three
    Top-K arms, so a hardcoded "Top500" would caption the Top1000 and Top2000
    arms' kept figures with a cut-off their numbers did not come from. The arm
    reads its own cut-off (C.topk_label) and passes it in, exactly as
    plot_celltype_validation.py passes its `cutoff` parameter.
    """
    if source == "model":
        return f"model ({cutoff} motif enrichment)"
    return SOURCE_STYLE[source][2]

STAGE_ORDER = ("prenatal", "postnatal", "adult")

FAMILY_ORDER = ("RFX", "NF-Y", "SOX", "NFI", "Nuclear receptor")
SOURCE_ORDER = ("model", "raw_chromVAR")

INK = "#1a1a1a"
MUTED = "#555555"
GRID = "#DDDDDD"


def family_color(family: str) -> str:
    return FAMILY_COLORS.get(str(family), "#333333")


def style_axes(ax, *, grid_axis: str = "y") -> None:
    """House axis style. `grid_axis="none"` draws no grid (matplotlib has no
    "none" for `ax.grid(axis=...)`, so it is handled here rather than at the call
    sites, which would otherwise each need an if)."""
    draw_grid = grid_axis in ("x", "y", "both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    if draw_grid:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)


def save(fig, prefix: Path, formats=("png", "pdf"), dpi: int = 200,
         tight: bool = True) -> list[Path]:
    """Write the figure in every requested format and return the paths."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = prefix.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, facecolor="white",
                    **({"bbox_inches": "tight"} if tight else {}))
        written.append(path)
    plt.close(fig)
    return written


def source_legend(fig, *, loc: str = "lower center", ncol: int = 2,
                  cutoff: str | None = None) -> None:
    """The one legend that every figure in this family carries.

    `cutoff` is the arm's own cut-off label (C.topk_label). It is optional only
    because a caller that has no config still gets the arm-independent label
    rather than a cut-off the figure's numbers did not come from.
    """
    handles = [
        plt.Line2D([], [], color=MUTED, linestyle=SOURCE_STYLE[s][0],
                   marker=SOURCE_STYLE[s][1], markersize=6,
                   markerfacecolor="white", markeredgewidth=1.2,
                   label=source_label(s, cutoff) if cutoff else SOURCE_STYLE[s][2])
        for s in SOURCE_ORDER
    ]
    fig.legend(handles=handles, loc=loc, ncol=ncol, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))


def order_stages(frame, column: str = "stage"):
    """Force the developmental order onto a stage column read back from a file.

    WHY this is not optional: the prep step writes `stage` as an ORDERED
    categorical, but a CSV round trip loses that -- pandas reads it back as a
    plain object column, and `sort_values("stage")` then sorts ALPHABETICALLY,
    i.e. adult, postnatal, prenatal. Every trajectory drawn from such a frame is
    mirrored, while the x-axis still reads prenatal, postnatal, adult. That is a
    silent wrong figure, so the ordering is imposed here rather than trusted.

    Raises if an unexpected stage appears, because a silently reordered or
    dropped window is the same class of failure.
    """
    import pandas as pd

    unexpected = sorted(set(frame[column]) - set(STAGE_ORDER))
    if unexpected:
        raise ValueError(f"unexpected {column} value(s): {unexpected}")
    frame = frame.copy()
    frame[column] = pd.Categorical(frame[column], STAGE_ORDER, ordered=True)
    return frame
