import os
import re
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import matplotlib as mpl

from pathlib import Path


def _data_root() -> Path:
    override = os.environ.get("ANALYSIS_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[4] / "analysis_data"


DATA_ROOT = _data_root()
RESULTS_ROOT = Path(os.environ.get("ANALYSIS_RESULTS_DIR", DATA_ROOT.parent / "analysis_code" / "results")).expanduser().resolve()


# ============================================================
# Matplotlib PDF 字体设置：保证 PDF 文字可编辑
# ============================================================

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# ============================================================
# 0. 输入输出路径
# ============================================================

adata_path = str(DATA_ROOT / "Figure6/brain/human_macaque_rna_atac_integrated.h5ad")

celltype_color_csv = str(DATA_ROOT / "Figure6/brain/cell_type_color_map.nature.csv")

# 总输出目录：每个 h5ad 会在这个目录下生成一个独立子文件夹
base_outdir = str(RESULTS_ROOT / "Figure6/brain/unsupervised_umap")


# ============================================================
# 0.1 自动生成当前 h5ad 的输出子文件夹
# ============================================================

prefix = os.path.splitext(os.path.basename(adata_path))[0]

# 防止特殊字符影响文件名
prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix)

# 每个 h5ad 一个独立文件夹
outdir = os.path.join(base_outdir, prefix)
os.makedirs(outdir, exist_ok=True)

print("File prefix:")
print(prefix)
print("Output directory:")
print(outdir)


# ============================================================
# 1. 读取 AnnData
# ============================================================

adata = ad.read_h5ad(adata_path)

celltype_col = "celltype"
species_col = "species"


# ============================================================
# 2. 检查字段
# ============================================================

for col in [celltype_col, species_col]:
    if col not in adata.obs.columns:
        raise ValueError(f"adata.obs 中找不到列: {col}")

if "X_umap" not in adata.obsm.keys():
    raise ValueError(
        "adata.obsm 中找不到 X_umap。"
        "请先计算 UMAP，或者确认 UMAP 坐标是否存储在其他 obsm key 中。"
    )


# ============================================================
# 3. 转成 category
# ============================================================

for col in [celltype_col, species_col]:
    adata.obs[col] = adata.obs[col].astype(str).astype("category")


# ============================================================
# 4. 默认 Nature-style palette，无黑色
# ============================================================

default_palette = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#9C755F",
    "#BAB0AC",
    "#8CD17D",
    "#A0CBE8",
    "#FFBE7D",
    "#D37295",
    "#86BCB6",
    "#B6992D",
    "#499894",
    "#FABFD2",
    "#79706E",
    "#D4A6C8",
    "#9D7660",
]


# ============================================================
# 5. 读取 cell type 颜色 CSV
# ============================================================

celltype_color_df = pd.read_csv(celltype_color_csv)

required_cols = {"cell_type", "color"}
missing_cols = required_cols - set(celltype_color_df.columns)
if missing_cols:
    raise ValueError(f"颜色 CSV 缺少必要列: {missing_cols}")

celltype_color_map_raw = dict(
    zip(
        celltype_color_df["cell_type"].astype(str),
        celltype_color_df["color"].astype(str),
    )
)

adata_celltypes = list(adata.obs[celltype_col].cat.categories)

missing_celltypes = [
    ct for ct in adata_celltypes
    if ct not in celltype_color_map_raw
]

if missing_celltypes:
    raise ValueError(
        "以下 cell type 在 h5ad 中存在，但在颜色 CSV 中没有对应颜色:\n"
        + "\n".join(missing_celltypes)
    )

# 按当前 h5ad 中的 category 顺序重排
celltype_color_map = {
    ct: celltype_color_map_raw[ct]
    for ct in adata_celltypes
}

print("\nCell type color map:")
for k, v in celltype_color_map.items():
    print(k, v)


# ============================================================
# 6. sp 默认顺序 + 默认色板
# ============================================================

species_categories = list(adata.obs[species_col].cat.categories)

if len(species_categories) > len(default_palette):
    raise ValueError(
        f"sp 类别数 {len(species_categories)} 超过默认色板长度 {len(default_palette)}，请补充 default_palette。"
    )

species_color_map = {
    sp: color
    for sp, color in zip(species_categories, default_palette[:len(species_categories)])
}

print("\nSpecies color map, default order:")
for k, v in species_color_map.items():
    print(k, v)


# ============================================================
# 7. 写入 AnnData.uns
# ============================================================

adata.uns[f"{celltype_col}_colors"] = [
    celltype_color_map[str(x)]
    for x in adata.obs[celltype_col].cat.categories
]

adata.uns[f"{species_col}_colors"] = [
    species_color_map[str(x)]
    for x in adata.obs[species_col].cat.categories
]


# ============================================================
# 8. 绘制正方形 UMAP
# ============================================================

def make_square_axis(ax, x, y, pad_ratio=0.04):
    """
    强制 UMAP 坐标范围为正方形，避免图形被拉伸。
    """
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2

    xrange = xmax - xmin
    yrange = ymax - ymin

    max_range = max(xrange, yrange)
    max_range = max_range * (1 + pad_ratio)

    ax.set_xlim(xmid - max_range / 2, xmid + max_range / 2)
    ax.set_ylim(ymid - max_range / 2, ymid + max_range / 2)

    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)


def plot_umap_square(
    adata,
    color,
    out_png,
    title="",
    point_size=3,
    alpha=0.8,
    dpi=600,
):
    fig, ax = plt.subplots(figsize=(5, 5))

    palette = celltype_color_map if color == celltype_col else species_color_map
    umap = adata.obsm["X_umap"]
    labels = adata.obs[color].astype(str).to_numpy()
    for label in adata.obs[color].cat.categories:
        mask = labels == str(label)
        ax.scatter(
            umap[mask, 0], umap[mask, 1],
            s=point_size, c=palette[str(label)], alpha=alpha,
            linewidths=0, rasterized=True,
        )
    ax.set_title(title)

    umap = adata.obsm["X_umap"]
    make_square_axis(
        ax,
        x=umap[:, 0],
        y=umap[:, 1],
        pad_ratio=0.04,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # 不使用 bbox_inches="tight"，否则正方形画布可能被裁剪成非正方形
    fig.savefig(
        out_png,
        dpi=dpi,
        transparent=True,
        pad_inches=0,
    )

    plt.close(fig)


# ============================================================
# 9. 单独保存 legend
# ============================================================

def save_legend(
    color_map,
    out_pdf,
    out_png,
    title=None,
    ncol=1,
    marker_size=7,
    fontsize=8,
):
    labels = list(color_map.keys())
    colors = [color_map[x] for x in labels]

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="none",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=marker_size,
            label=label,
        )
        for label, color in zip(labels, colors)
    ]

    fig_height = max(1.8, 0.32 * len(labels) / ncol)
    fig_width = max(2.5, 2.8 * ncol)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    ax.legend(
        handles=handles,
        labels=labels,
        loc="center left",
        frameon=False,
        title=title,
        fontsize=fontsize,
        title_fontsize=fontsize + 1,
        ncol=ncol,
        handletextpad=0.5,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    fig.savefig(out_pdf, dpi=600, bbox_inches="tight", transparent=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)


# ============================================================
# 10. 输出 UMAP 图
# ============================================================

point_size = 5
alpha = 1.0

plot_umap_square(
    adata,
    color=celltype_col,
    out_png=os.path.join(outdir, f"{prefix}_umap_cell_type.nature.png"),
    title="",
    point_size=point_size,
    alpha=alpha,
)

plot_umap_square(
    adata,
    color=species_col,
    out_png=os.path.join(outdir, f"{prefix}_umap_sp.nature.png"),
    title="",
    point_size=point_size,
    alpha=alpha,
)


# ============================================================
# 11. 输出 legend
# ============================================================

save_legend(
    celltype_color_map,
    out_pdf=os.path.join(outdir, f"{prefix}_legend_cell_type.nature.pdf"),
    out_png=os.path.join(outdir, f"{prefix}_legend_cell_type.nature.png"),
    title="Cell type",
    ncol=1,
)

save_legend(
    species_color_map,
    out_pdf=os.path.join(outdir, f"{prefix}_legend_sp.nature.pdf"),
    out_png=os.path.join(outdir, f"{prefix}_legend_sp.nature.png"),
    title="Species",
    ncol=1,
)


# ============================================================
# 12. 保存颜色映射
# ============================================================

pd.DataFrame({
    "cell_type": list(celltype_color_map.keys()),
    "color": list(celltype_color_map.values()),
}).to_csv(
    os.path.join(outdir, f"{prefix}_cell_type_color_map.used.nature.csv"),
    index=False,
)

pd.DataFrame({
    "sp": list(species_color_map.keys()),
    "color": list(species_color_map.values()),
}).to_csv(
    os.path.join(outdir, f"{prefix}_sp_color_map.default_order.nature.csv"),
    index=False,
)


# ============================================================
# 13. 保存带颜色信息的 h5ad
# ============================================================

adata.write_h5ad(
    os.path.join(outdir, f"{prefix}.with_paper_colors.nature.h5ad")
)

print("\nDone.")
print("Output directory:")
print(outdir)
print("\nOutput prefix:")
print(prefix)
