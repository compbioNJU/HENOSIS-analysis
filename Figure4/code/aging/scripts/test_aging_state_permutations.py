#!/usr/bin/env python3
"""Donor-label permutation rebuilding the LODO pipeline on balanced cell samples."""
from pathlib import Path
import os
import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RNG = np.random.default_rng(20260822)
NPERM = int(os.environ.get("NPERM", "1000"))
Q = 0.90
CTS = [
    "Excitatory neuron",
    "Inhibitory neuron",
    "Astrocyte",
    "Oligodendrocyte",
    "OPC",
    "Microglia",
    "Vascular-associated",
]
SMAP = {"young": 0, "middle": 1, "old": 2}
B = Path(os.environ.get("ANALYSIS_DATA_DIR", Path(__file__).resolve().parents[5] / "analysis_data")) / "Figure4/aging"
OUT = Path(os.environ.get("ANALYSIS_RESULTS_DIR", Path(__file__).resolve().parents[3] / "results/aging"))
OUT.mkdir(parents=True, exist_ok=True)
I = (
    B
    / "human_mouse_aging_integrated.h5ad"
)
H = B / "human_concated_adata_with_meta.h5ad"
M = B / "mouse_concated_adata_with_meta.h5ad"
a = ad.read_h5ad(I, backed="r")
h = ad.read_h5ad(H, backed="r")
m = ad.read_h5ad(M, backed="r")
X = np.asarray(a.obsm["X_HENOSIS"], dtype=np.float32)
bar = a.obs.barcode.astype(str).to_numpy()
dm = pd.concat(
    [
        pd.Series(h.obs.donor_id.astype(str).to_numpy(), index=h.obs_names.astype(str)),
        pd.Series(m.obs.mouse_id.astype(str).to_numpy(), index=m.obs_names.astype(str)),
    ]
)
rm = pd.Series(m.obs.region.astype(str).to_numpy(), index=m.obs_names.astype(str))
meta = a.obs[["sp", "age_stage", "Celltype"]].copy().reset_index(drop=True)
meta["species"] = np.where(meta.sp.astype(str).eq("homo_sapiens"), "Human", "Mouse")
meta["donor"] = dm.reindex(bar).to_numpy()
meta["region"] = rm.reindex(bar).fillna("Human cortex").to_numpy()
meta["age_stage"] = meta.age_stage.astype(str)
donors = meta[["species", "donor", "age_stage"]].drop_duplicates().copy()
donors["donor"] = donors.donor.astype(str)
keys = list(zip(donors.species, donors.donor))
keypos = {k: i for i, k in enumerate(keys)}
true_labels = donors.age_stage.to_numpy()


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else np.full_like(v, np.nan)


# Full-cell donor centroids define each permuted aging direction. Fixed balanced
# samples are used for thresholds and fractions in both observed and null runs.
cent = {}
samples = {}
region_samples = {}
for (sp, d, ct), ii in (
    meta[meta.Celltype.isin(CTS)]
    .groupby(["species", "donor", "Celltype"], observed=True)
    .groups.items()
):
    ix = np.asarray(list(ii), dtype=int)
    k = (sp, str(d), str(ct))
    cent[k] = X[ix].mean(0)
    take = RNG.choice(ix, size=min(500, len(ix)), replace=False)
    samples[k] = X[take]
    if sp == "Mouse" and ct in ["Oligodendrocyte", "Microglia"]:
        rr = meta.region.iloc[ix].to_numpy()
        for region in ["CC/ACO", "STR", "CTX"]:
            z = ix[rr == region]
            if len(z) >= 30:
                region_samples[(str(d), str(ct), region)] = X[
                    RNG.choice(z, size=min(300, len(z)), replace=False)
                ]


def one_run(labels):
    lab = {k: s for k, s in zip(keys, labels)}
    frac = []
    regional = []
    for ct in CTS:
        available = [k for k in keys if (k[0], k[1], ct) in cent]
        axes = {}
        for held in available:
            av = []
            for sp in ["Human", "Mouse"]:
                yy = [
                    cent[(s, d, ct)]
                    for s, d in available
                    if s == sp and (s, d) != held and lab[(s, d)] == "young"
                ]
                oo = [
                    cent[(s, d, ct)]
                    for s, d in available
                    if s == sp and (s, d) != held and lab[(s, d)] == "old"
                ]
                av.append(unit(np.mean(oo, 0) - np.mean(yy, 0)))
            axes[held] = unit(av[0] + av[1])
        # Each held donor is projected only on an axis that excluded it.
        for held in available:
            sp, d = held
            axis = axes[held]
            train_y = [
                k for k in available if k[0] == sp and k != held and lab[k] == "young"
            ]
            th = np.mean(
                [np.quantile(samples[(s, dd, ct)] @ axis, Q) for s, dd in train_y]
            )
            f = np.mean(samples[(sp, d, ct)] @ axis > th)
            frac.append([sp, d, ct, lab[held], f])
            if sp == "Mouse" and ct in ["Oligodendrocyte", "Microglia"]:
                for region in ["CC/ACO", "STR", "CTX"]:
                    hk = (d, ct, region)
                    if hk not in region_samples:
                        continue
                    refs = []
                    for s, dd in train_y:
                        rk = (dd, ct, region)
                        if rk in region_samples:
                            refs.append(np.quantile(region_samples[rk] @ axis, Q))
                    if refs:
                        regional.append(
                            [
                                d,
                                ct,
                                region,
                                lab[held],
                                np.mean(region_samples[hk] @ axis > np.mean(refs)),
                            ]
                        )
    f = pd.DataFrame(
        frac, columns=["species", "donor", "cell_type", "stage", "fraction"]
    )
    metrics = []
    for ct in ["Oligodendrocyte", "Microglia"]:
        z = f[f.cell_type.eq(ct)].groupby(["species", "stage"]).fraction.mean()
        dh = z.get(("Human", "old"), np.nan) - z.get(("Human", "young"), np.nan)
        dm = z.get(("Mouse", "old"), np.nan) - z.get(("Mouse", "young"), np.nan)
        metrics.append(min(dh, dm))
    r = pd.DataFrame(
        regional, columns=["donor", "cell_type", "region", "stage", "fraction"]
    )
    for ct in ["Oligodendrocyte", "Microglia"]:
        w = (
            r[r.cell_type.eq(ct)]
            .pivot_table(index=["donor", "stage"], columns="region", values="fraction")
            .dropna()
            .reset_index()
        )
        metrics.append(spearmanr(w.stage.map(SMAP), w["CC/ACO"] - w.CTX).statistic)
    return np.asarray(metrics, float)


observed = one_run(true_labels)
null = np.empty((NPERM, 4), float)
for i in range(NPERM):
    labels = true_labels.copy()
    for sp in ["Human", "Mouse"]:
        z = np.flatnonzero(donors.species.to_numpy() == sp)
        labels[z] = RNG.permutation(labels[z])
    null[i] = one_run(labels)
    if (i + 1) % 100 == 0:
        print(f"{i+1}/{NPERM}", flush=True)
names = [
    "Oligodendrocyte_conservative_joint_occupancy_effect",
    "Microglia_conservative_joint_occupancy_effect",
    "Oligodendrocyte_CC_minus_CTX_occupancy_rho",
    "Microglia_CC_minus_CTX_occupancy_rho",
]
summary = []
for j, n in enumerate(names):
    p = (1 + np.sum(null[:, j] >= observed[j])) / (NPERM + 1)
    summary.append(
        [
            n,
            observed[j],
            p,
            np.quantile(null[:, j], 0.025),
            np.quantile(null[:, j], 0.975),
            NPERM,
        ]
    )
pd.DataFrame(
    summary,
    columns=[
        "metric",
        "observed_balanced_sample",
        "full_pipeline_permutation_p",
        "null_2.5",
        "null_97.5",
        "n_permutations",
    ],
).to_csv(OUT / "full_pipeline_permutation_statistics.csv", index=False)
pd.DataFrame(null, columns=names).to_csv(
    OUT / "full_pipeline_permutation_null.csv.gz", index=False, compression="gzip"
)
print(
    pd.DataFrame(
        summary,
        columns=[
            "metric",
            "observed_balanced_sample",
            "full_pipeline_permutation_p",
            "null_2.5",
            "null_97.5",
            "n_permutations",
        ],
    ).to_string(index=False)
)
