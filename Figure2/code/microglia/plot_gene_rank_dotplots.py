#!/usr/bin/env python3
"""Plot expression of each marker CSV's genes across all cell types."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import sparse

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
CMAP = LinearSegmentedColormap.from_list(
    "nature_warm", ["#FFF7BC", "#FEC44F", "#F03B20", "#99000D"]
)
MIN_SIZE, MAX_SIZE = 8.0, 230.0
MERGED_L6_LABEL = "L6 IT"
CELLTYPE_ORDER = ["ASC", "ChC", "Endo", "L2/3 IT", "L4/5 IT", "L5 ET", "L5/6 NP",
                  "L6 CT", "L6 IT", "L6 IT CAR3", "L6b", "LAMP5", "MGC", "ODC",
                  "OPC", "PVALB", "SNCG", "SST", "VIP", "VLMC"]


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank-dir", type=Path, default=here.parent / "data/scRNAseq/gene_rank")
    p.add_argument("--h5ad-dir", type=Path,
                   default=Path("data/figure2/motor"))
    p.add_argument("--output-dir", type=Path, default=here / "gene_rank_dotplot_rank10")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--celltype-column", default="Celltype")
    p.add_argument("--expression-source", choices=("X", "counts-log1p"), default="X",
                   help="X uses the existing log1p matrix; counts-log1p normalizes counts to 1e4 then log1p")
    p.add_argument("--target-sum", type=float, default=10000.0)
    p.add_argument("--color-clip-percentile", type=float, default=99.0)
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()


def load_markers(path, top_n):
    frame = pd.read_csv(path)
    if not {"gene", "rank"} <= set(frame.columns):
        raise ValueError(f"{path}: gene/rank columns are required")
    frame["gene"] = frame["gene"].astype(str).str.strip()
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    return frame.dropna(subset=["gene", "rank"]).sort_values("rank").drop_duplicates("gene").head(top_n)


def resolve_genes(markers, var_names):
    exact = {str(g): i for i, g in enumerate(var_names)}
    lower = {}
    for i, gene in enumerate(map(str, var_names)):
        lower.setdefault(gene.lower(), []).append((gene, i))
    rows, indices = [], []
    for row in markers.itertuples(index=False):
        query = row.gene
        if query in exact:
            actual, idx, status = query, exact[query], "exact"
        elif len(lower.get(query.lower(), [])) == 1:
            actual, idx = lower[query.lower()][0]
            status = "case_insensitive"
        else:
            rows.append({**row._asdict(), "adata_gene": "", "match_status": "missing"})
            continue
        rows.append({**row._asdict(), "adata_gene": actual, "match_status": status})
        indices.append(idx)
    return pd.DataFrame(rows), indices


def materialize(matrix):
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    return matrix.tocsr() if sparse.issparse(matrix) else np.asarray(matrix)


def summarize(adata, resolved, indices, celltype_column, expression_source, target_sum):
    matched = resolved[resolved.match_status.isin(["exact", "case_insensitive"])]
    expr = materialize(adata.X[:, indices])
    counts = materialize(adata.layers["counts"][:, indices])
    if expression_source == "counts-log1p":
        all_counts = adata.layers["counts"]
        totals = np.empty(adata.n_obs, dtype=float)
        for start in range(0, adata.n_obs, 2048):
            totals[start:start+2048] = np.asarray(all_counts[start:start+2048].sum(axis=1)).ravel()
        scale = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
        if sparse.issparse(counts):
            expr = sparse.diags(scale).dot(counts).tocsr()
            expr.data = np.log1p(expr.data)
        else:
            expr = np.log1p(counts * scale[:, None])

    groups = adata.obs[celltype_column].astype(str).replace(
        {"L6 IT": MERGED_L6_LABEL, "L6 IT CAR3": MERGED_L6_LABEL}
    )
    present = set(groups.unique())
    base_order = [MERGED_L6_LABEL if x in {"L6 IT", "L6 IT CAR3"} else x for x in CELLTYPE_ORDER]
    base_order = list(dict.fromkeys(base_order))
    order = [x for x in base_order if x in present] + sorted(present - set(base_order))
    rows = []
    for group in order:
        mask = (groups == group).to_numpy()
        e, c = expr[mask], counts[mask]
        means = np.asarray(e.mean(axis=0)).ravel()
        fractions = np.asarray((c > 0).mean(axis=0)).ravel() if sparse.issparse(c) else (c > 0).mean(axis=0)
        for gene, actual, mean, fraction in zip(matched.gene, matched.adata_gene, means, fractions):
            rows.append({"celltype": group, "gene": gene, "adata_gene": actual,
                         "mean_log1p_expression": float(mean), "fraction_expressed": float(fraction)})
    return pd.DataFrame(rows), order


def plot(summary, resolved, cell_order, species, target_celltype, output_base, percentile, dpi):
    genes = resolved.loc[resolved.match_status.isin(["exact", "case_insensitive"])].sort_values("rank").gene.tolist()
    target_label = target_celltype.replace("_", "/")
    if target_label in {"L6 IT", "L6 IT CAR3"}:
        target_label = MERGED_L6_LABEL
    if target_label in cell_order:
        cell_order = [target_label] + [x for x in cell_order if x != target_label]
    data = summary.copy()
    data["x"] = data.celltype.map({x: i for i, x in enumerate(cell_order)})
    data["y"] = data.gene.map({x: i for i, x in enumerate(genes)})
    data["size"] = MIN_SIZE + data.fraction_expressed * (MAX_SIZE - MIN_SIZE)
    vmax = float(np.nanpercentile(data.mean_log1p_expression, percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    fig, ax = plt.subplots(figsize=(max(11, .58*len(cell_order)+3), max(6, .42*len(genes)+2)))
    scatter = ax.scatter(data.x, data.y, s=data["size"], c=data.mean_log1p_expression,
                         cmap=CMAP, vmin=0, vmax=vmax, edgecolors="#4C4C4C", linewidths=.4)
    ax.set_xticks(range(len(cell_order)), cell_order, rotation=45, ha="right")
    ax.set_yticks(range(len(genes)), genes)
    ax.invert_yaxis()
    ax.set(xlabel="Cell type", ylabel="Marker gene",
           title=f"{species.replace('_', ' ').capitalize()} | {target_celltype} markers")
    ax.grid(color="#E7E7E7", linewidth=.5); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.colorbar(scatter, ax=ax, fraction=.025, pad=.03).set_label("Mean log1p expression")
    fractions = [.25, .5, .75, 1.0]
    handles = [ax.scatter([], [], s=MIN_SIZE+f*(MAX_SIZE-MIN_SIZE), facecolor="#CCD5DA",
                          edgecolor="#4C4C4C", linewidth=.4) for f in fractions]
    ax.legend(handles, [f"{f:.0%}" for f in fractions], title="Cells expressing",
              frameon=False, bbox_to_anchor=(1.12, 1), loc="upper left", labelspacing=1.3)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if args.top_n < 1: raise ValueError("--top-n must be >= 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for species_dir in sorted(p for p in args.rank_dir.iterdir() if p.is_dir()):
        species, h5ad_path = species_dir.name, args.h5ad_dir / f"{species_dir.name}.h5ad"
        if not h5ad_path.exists():
            raise FileNotFoundError(h5ad_path)
        adata = ad.read_h5ad(h5ad_path, backed="r")
        try:
            if args.celltype_column not in adata.obs: raise ValueError(f"{h5ad_path}: missing obs[{args.celltype_column}]")
            if "counts" not in adata.layers: raise ValueError(f"{h5ad_path}: missing counts layer")
            outdir = args.output_dir / species
            outdir.mkdir(parents=True, exist_ok=True)
            for csv_path in sorted(species_dir.glob("*.csv")):
                target = csv_path.stem
                markers = load_markers(csv_path, args.top_n)
                resolved, indices = resolve_genes(markers, adata.var_names)
                matched_n = int(resolved.match_status.isin(["exact", "case_insensitive"]).sum())
                safe = re.sub(r"[^A-Za-z0-9._-]+", "_", target).strip("_")
                base = outdir / f"expr_dotplot_{safe}_top{args.top_n}"
                if matched_n == 0:
                    report.append({"species": species, "cell_type": target, "status": "no_genes_matched"})
                    continue
                summary, order = summarize(adata, resolved, indices, args.celltype_column,
                                           args.expression_source, args.target_sum)
                plot(summary, resolved, order, species, target, base, args.color_clip_percentile, args.dpi)
                resolved.to_csv(base.with_name(base.name + "_selected_genes.csv"), index=False)
                summary.to_csv(base.with_name(base.name + "_expression_summary.csv"), index=False)
                report.append({"species": species, "cell_type": target, "status": "ok",
                               "selected_n": len(markers), "matched_n": matched_n, "expression_source": args.expression_source})
                print(f"[OK] {species}/{target}: {matched_n}/{len(markers)} genes", flush=True)
        finally:
            adata.file.close()
    pd.DataFrame(report).to_csv(args.output_dir / "run_report.csv", index=False)
    print(f"Saved {sum(x['status']=='ok' for x in report)} expression dotplots to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
