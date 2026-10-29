"""Evaluate Leiden/Celltype agreement from Sankey flow-count tables."""

from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


HERE = Path(__file__).resolve()
OUT = Path(os.environ.get("FIGURE4_OUTPUT_DIR", HERE.parents[4] / "review_output/Figure4")) / "results/xenium"


def evaluate(frame, version, species):
    table = frame.pivot_table(index="leiden", columns="Celltype", values="count",
                              aggfunc="sum", fill_value=0)
    table = table.reindex(sorted(table.index, key=int))
    cluster_totals = table.sum(axis=1)
    dominant = table.idxmax(axis=1)
    dominant_n = table.max(axis=1)
    cluster_rows = pd.DataFrame({
        "version": version, "species": species, "leiden": table.index,
        "n_cells": cluster_totals.to_numpy(),
        "dominant_Celltype": dominant.to_numpy(),
        "dominant_count": dominant_n.to_numpy(),
        "dominant_fraction": (dominant_n / cluster_totals).to_numpy(),
    })
    leiden = np.repeat(frame["leiden"].astype(str).to_numpy(), frame["count"].to_numpy())
    celltype = np.repeat(frame["Celltype"].astype(str).to_numpy(), frame["count"].to_numpy())
    summary = {
        "version": version, "species": species, "n_cells": len(leiden),
        "n_clusters": table.shape[0], "n_celltypes": table.shape[1],
        "purity": dominant_n.sum() / cluster_totals.sum(),
        "NMI": normalized_mutual_info_score(leiden, celltype),
        "ARI": adjusted_rand_score(leiden, celltype),
    }
    return summary, cluster_rows, table


def main():
    summaries, clusters = [], []
    for version, filename in [
        ("min250", "sankey_leiden_to_Celltype_min250_counts.csv"),
    ]:
        frame = pd.read_csv(OUT / filename)
        for species in ["combined", *sorted(frame["sp"].unique())]:
            subset = frame if species == "combined" else frame[frame["sp"] == species]
            summary, cluster_rows, table = evaluate(subset, version, species)
            summaries.append(summary)
            clusters.append(cluster_rows)
            table.to_csv(OUT / f"leiden_celltype_matrix_{version}_{species}.csv")
    pd.DataFrame(summaries).to_csv(OUT / "leiden_celltype_consistency.csv", index=False)
    pd.concat(clusters, ignore_index=True).to_csv(
        OUT / "leiden_celltype_cluster_purity.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
