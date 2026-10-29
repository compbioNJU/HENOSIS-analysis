import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.collections import PolyCollection
random_state = 0
BOX_EDGE = "#2F2F2F"
POINT_COLOR = "#6A6A6A"
STAGE_PALETTE = {
    "young":  "#4DBBD5",
    "middle": "#00A087",
    "old":    "#E64B35",
}
def hex_to_rgba(hex_color, alpha=0.42):
    r, g, b = mcolors.to_rgb(hex_color)
    return (r, g, b, alpha)

def make_group_positions(order):
    return {str(g): i for i, g in enumerate(order)}

def draw_transparent_box_summary(
    ax,
    data,
    x_col,
    y_col,
    order,
    palette,
    box_width=0.22,
    whisker_width=0.10,
    box_alpha=0.38,
    edge_color=BOX_EDGE,
    linewidth=0.9,
):
    pos_map = make_group_positions(order)

    for group in order:
        vals = data.loc[data[x_col].astype(str) == str(group), y_col].dropna().values

        if len(vals) == 0:
            continue

        x = pos_map[str(group)]

        q1 = np.percentile(vals, 25)
        med = np.percentile(vals, 50)
        q3 = np.percentile(vals, 75)
        wlow = np.percentile(vals, 5)
        whigh = np.percentile(vals, 95)

        fill_color = palette.get(str(group), "#8DD3C7")

        ax.vlines(x, wlow, whigh, color=edge_color, linewidth=linewidth, zorder=7)
        ax.hlines(wlow, x - whisker_width / 2, x + whisker_width / 2, color=edge_color, linewidth=linewidth, zorder=7)
        ax.hlines(whigh, x - whisker_width / 2, x + whisker_width / 2, color=edge_color, linewidth=linewidth, zorder=7)

        rect = Rectangle(
            (x - box_width / 2, q1),
            box_width,
            q3 - q1,
            facecolor=hex_to_rgba(fill_color, box_alpha),
            edgecolor=edge_color,
            linewidth=linewidth,
            zorder=8,
        )
        ax.add_patch(rect)

        ax.hlines(med, x - box_width / 2, x + box_width / 2, color=edge_color, linewidth=1.2, zorder=9)

def add_jitter_points(
    ax,
    data,
    x_col,
    y_col,
    order,
    point_size=18,
    jitter_width=0.08,
    alpha=0.52,
    random_state=0,
):
    rng = np.random.default_rng(random_state)
    pos_map = make_group_positions(order)

    for group in order:
        df_g = data[data[x_col].astype(str) == str(group)].copy()

        if df_g.shape[0] == 0:
            continue

        x0 = pos_map[str(group)]
        x = x0 + rng.uniform(-jitter_width, jitter_width, size=df_g.shape[0])

        ax.scatter(
            x,
            df_g[y_col].values,
            s=point_size,
            color=POINT_COLOR,
            alpha=alpha,
            edgecolor="none",
            zorder=10,
        )

def violinplot_reference(
    ax,
    cell_df,
    sample_df,
    x_col,
    y_col_cell,
    y_col_sample,
    order,
    palette,
    title,
    ylabel,
    xlabel="",
):
    try:
        sns.violinplot(
            data=cell_df,
            x=x_col,
            y=y_col_cell,
            order=order,
            palette=palette,
            inner=None,
            cut=0,
            density_norm="width",
            linewidth=0.7,
            saturation=1,
            ax=ax,
        )
    except Exception:
        sns.violinplot(
            data=cell_df,
            x=x_col,
            y=y_col_cell,
            order=order,
            palette=palette,
            inner=None,
            cut=0,
            scale="width",
            linewidth=0.7,
            saturation=1,
            ax=ax,
        )

    for coll in ax.collections:
        if isinstance(coll, PolyCollection):
            coll.set_alpha(0.42)
            coll.set_edgecolor("#444444")
            coll.set_linewidth(0.65)

    draw_transparent_box_summary(
        ax=ax,
        data=sample_df,
        x_col=x_col,
        y_col=y_col_sample,
        order=order,
        palette=palette,
        box_width=0.20,
        whisker_width=0.11,
        box_alpha=0.42,
    )

    add_jitter_points(
        ax=ax,
        data=sample_df,
        x_col=x_col,
        y_col=y_col_sample,
        order=order,
        point_size=20,
        jitter_width=0.08,
        alpha=0.52,
        random_state=random_state,
    )

    ax.set_title(title, pad=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.35, linewidth=0.45)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def set_plot_style():
    sns.set_theme(style="whitegrid", context="paper")

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,

        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,

        "axes.linewidth": 0.8,
        "grid.linewidth": 0.45,
        "grid.alpha": 0.35,

        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,

        "axes.spines.top": False,
        "axes.spines.right": False,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
