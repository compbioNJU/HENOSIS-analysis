#!/usr/bin/env python3
"""LODO cross-fitted, donor-balanced analysis using X_HENOSIS only."""
from pathlib import Path
import os
import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

RNG = np.random.default_rng(20260822)
NBOOT = 2000
QS = (0.80, 0.90, 0.95)
CTS = [
    "Excitatory neuron",
    "Inhibitory neuron",
    "Astrocyte",
    "Oligodendrocyte",
    "OPC",
    "Microglia",
    "Vascular-associated",
]
STAGES = ["young", "middle", "old"]
SMAP = {x: i for i, x in enumerate(STAGES)}
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
donor_map = pd.concat(
    [
        pd.Series(h.obs.donor_id.astype(str).to_numpy(), index=h.obs_names.astype(str)),
        pd.Series(m.obs.mouse_id.astype(str).to_numpy(), index=m.obs_names.astype(str)),
    ]
)
region_map = pd.Series(
    m.obs.region.astype(str).to_numpy(), index=m.obs_names.astype(str)
)
meta = a.obs[["sp", "age_stage", "Celltype"]].copy().reset_index(drop=True)
meta["species"] = np.where(meta.sp.astype(str).eq("homo_sapiens"), "Human", "Mouse")
meta["donor"] = donor_map.reindex(bar).to_numpy()
meta["region"] = region_map.reindex(bar).fillna("Human cortex").to_numpy()
meta["age_stage"] = meta.age_stage.astype(str)
meta["stage"] = meta.age_stage.map(SMAP)

# Cell indices and equal-weight donor centroids.
indices = {}
centroids = []
for (sp, d, ct), ii in (
    meta[meta.Celltype.isin(CTS)]
    .groupby(["species", "donor", "Celltype"], observed=True)
    .groups.items()
):
    ix = np.asarray(list(ii), dtype=int)
    indices[(sp, str(d), str(ct))] = ix
    centroids.append(
        [sp, str(d), str(ct), meta.age_stage.iloc[ix[0]], len(ix), *X[ix].mean(0)]
    )
dcols = [f"d{i}" for i in range(X.shape[1])]
dc = pd.DataFrame(
    centroids, columns=["species", "donor", "cell_type", "age_stage", "n_cells"] + dcols
)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.full_like(v, np.nan)


def shared_axis(cell_type, held_species=None, held_donor=None, stage_override=None):
    g = dc[dc.cell_type.eq(cell_type)].copy()
    if held_donor is not None:
        g = g[~(g.species.eq(held_species) & g.donor.eq(str(held_donor)))]
    if stage_override is not None:
        g["age_stage"] = [stage_override[(sp, d)] for sp, d in zip(g.species, g.donor)]
    axes = []
    for sp in ["Human", "Mouse"]:
        z = g[g.species.eq(sp)]
        young = z[z.age_stage.eq("young")][dcols].to_numpy().mean(0)
        old = z[z.age_stage.eq("old")][dcols].to_numpy().mean(0)
        axes.append(unit(old - young))
    return unit(axes[0] + axes[1]), axes[0], axes[1]


def balanced_threshold(axis, species, cell_type, q, held_donor=None, region=None):
    vals = []
    donors = dc[
        (dc.species.eq(species))
        & (dc.cell_type.eq(cell_type))
        & (dc.age_stage.eq("young"))
    ].donor
    for d in donors:
        if held_donor is not None and str(d) == str(held_donor):
            continue
        ix = indices[(species, str(d), cell_type)]
        if region is not None:
            ix = ix[meta.region.iloc[ix].to_numpy() == region]
        if len(ix) >= 30:
            vals.append(np.quantile(X[ix] @ axis, q))
    return float(np.mean(vals)) if vals else np.nan


# LODO prediction: each donor's feature and thresholds exclude that donor.
rows = []
regional = []
for _, r in dc[["species", "donor", "age_stage"]].drop_duplicates().iterrows():
    sp, d, stage = r.species, str(r.donor), r.age_stage
    for ct in CTS:
        if (sp, d, ct) not in indices:
            continue
        axis, _, _ = shared_axis(ct, sp, d)
        ix = indices[(sp, d, ct)]
        held = X[ix] @ axis
        for q in QS:
            threshold = balanced_threshold(axis, sp, ct, q, d)
            rows.append(
                [
                    sp,
                    d,
                    stage,
                    SMAP[stage],
                    ct,
                    q,
                    len(ix),
                    float(np.mean(held > threshold)),
                    threshold,
                ]
            )
        if sp == "Mouse" and ct in ["Oligodendrocyte", "Microglia"]:
            regs = meta.region.iloc[ix].to_numpy()
            for region in ["CC/ACO", "STR", "CTX"]:
                z = held[regs == region]
                if len(z) < 30:
                    continue
                for q in QS:
                    threshold = balanced_threshold(axis, "Mouse", ct, q, d, region)
                    regional.append(
                        [
                            d,
                            stage,
                            SMAP[stage],
                            ct,
                            region,
                            q,
                            len(z),
                            float(np.mean(z > threshold)),
                            threshold,
                        ]
                    )
lodo = pd.DataFrame(
    rows,
    columns=[
        "species",
        "donor",
        "age_stage",
        "stage",
        "cell_type",
        "quantile",
        "n_cells",
        "aging_state_occupancy",
        "training_threshold",
    ],
)
lodo.to_csv(OUT / "lodo_donor_aging_state_occupancy.csv", index=False)
reg = pd.DataFrame(
    regional,
    columns=[
        "donor",
        "age_stage",
        "stage",
        "cell_type",
        "region",
        "quantile",
        "n_cells",
        "aging_state_occupancy",
        "training_threshold",
    ],
)
reg.to_csv(OUT / "lodo_mouse_regional_glial_occupancy.csv", index=False)

# Cross-species lineage effects and Q sensitivity.
effects = []
for (q, sp, ct), g in lodo.groupby(["quantile", "species", "cell_type"]):
    means = g.groupby("age_stage").aging_state_occupancy.mean()
    sr = spearmanr(g.stage, g.aging_state_occupancy)
    effects.append(
        [
            q,
            sp,
            ct,
            means.get("young", np.nan),
            means.get("middle", np.nan),
            means.get("old", np.nan),
            means.get("old", np.nan) - means.get("young", np.nan),
            sr.statistic,
            sr.pvalue,
        ]
    )
effects = pd.DataFrame(
    effects,
    columns=[
        "quantile",
        "species",
        "cell_type",
        "young_occupancy",
        "middle_occupancy",
        "old_occupancy",
        "old_minus_young_occupancy_change",
        "stage_rho",
        "stage_pvalue",
    ],
)
wide = effects.pivot(
    index=["quantile", "cell_type"],
    columns="species",
    values="old_minus_young_occupancy_change",
).reset_index()
wide["conservative_joint_occupancy_effect"] = wide[["Human", "Mouse"]].min(axis=1)
wide.to_csv(OUT / "lodo_cross_species_occupancy_effects.csv", index=False)

# Human-mouse direction alignment with stage-stratified donor bootstrap.
align = []
for ct in CTS:
    full = dc[dc.cell_type.eq(ct)]
    vh = []
    vm = []
    for sp, target in [("Human", vh), ("Mouse", vm)]:
        z = full[full.species.eq(sp)]
        target.extend(
            [z[z.age_stage.eq(st)][dcols].to_numpy() for st in ["young", "old"]]
        )
    obs = float(
        np.dot(unit(vh[1].mean(0) - vh[0].mean(0)), unit(vm[1].mean(0) - vm[0].mean(0)))
    )
    boots = []
    for _ in range(NBOOT):
        ah = unit(
            vh[1][RNG.integers(len(vh[1]), size=len(vh[1]))].mean(0)
            - vh[0][RNG.integers(len(vh[0]), size=len(vh[0]))].mean(0)
        )
        am = unit(
            vm[1][RNG.integers(len(vm[1]), size=len(vm[1]))].mean(0)
            - vm[0][RNG.integers(len(vm[0]), size=len(vm[0]))].mean(0)
        )
        boots.append(np.dot(ah, am))
    align.append(
        [
            ct,
            obs,
            np.quantile(boots, 0.025),
            np.quantile(boots, 0.975),
            np.mean(np.asarray(boots) <= 0),
        ]
    )
align = pd.DataFrame(
    align,
    columns=[
        "cell_type",
        "axis_cosine",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "bootstrap_p_nonpositive",
    ],
)
align["bootstrap_q_BH"] = multipletests(align.bootstrap_p_nonpositive, method="fdr_bh")[
    1
]
align.to_csv(OUT / "direction_alignment_bootstrap.csv", index=False)

# Regional statistics at all thresholds.
rstats = []
for (q, ct), g in reg.groupby(["quantile", "cell_type"]):
    w = (
        g.pivot_table(
            index=["donor", "age_stage", "stage"],
            columns="region",
            values="aging_state_occupancy",
        )
        .dropna()
        .reset_index()
    )
    w["CC_minus_CTX"] = w["CC/ACO"] - w["CTX"]
    sr = spearmanr(w.stage, w.CC_minus_CTX)
    means = g.groupby(["region", "age_stage"]).aging_state_occupancy.mean().unstack()
    delta = means["old"] - means["young"]
    rstats.append(
        [
            q,
            ct,
            len(w),
            sr.statistic,
            sr.pvalue,
            delta.get("CC/ACO"),
            delta.get("STR"),
            delta.get("CTX"),
        ]
    )
rstats = pd.DataFrame(
    rstats,
    columns=[
        "quantile",
        "cell_type",
        "n_donors",
        "CC_minus_CTX_stage_rho",
        "pvalue",
        "old_minus_young_occupancy_CC_ACO",
        "old_minus_young_occupancy_STR",
        "old_minus_young_occupancy_CTX",
    ],
)
rstats.to_csv(OUT / "lodo_regional_vulnerability_statistics.csv", index=False)

# Figure 1 (Q90).
sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
mpl.rcParams.update(
    {
        "text.color": "black",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "axes.titlecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    }
)
q90 = lodo[np.isclose(lodo["quantile"], 0.9)].copy()
q90["sp_label"] = q90.species.map({"Human": "homo_sapiens", "Mouse": "Mus_musculus"})
g = sns.catplot(
    data=q90,
    x="age_stage",
    y="aging_state_occupancy",
    hue="sp_label",
    hue_order=["Mus_musculus", "homo_sapiens"],
    col="cell_type",
    col_wrap=7,
    order=STAGES,
    kind="box",
    showfliers=False,
    height=3.35,
    aspect=0.86,
    sharey=True,
    width=0.64,
    gap=0.18,
    palette={"Mus_musculus": "#1f77b4", "homo_sapiens": "#ff7f0e"},
)
g.set_axis_labels("", "")
g.set_titles("{col_name}")
g.set(ylim=(-0.02, 1.0))
g.figure.supxlabel("Relative age stage", y=0.025)
g.figure.supylabel("Cross-fitted aging-state occupancy", x=0.008)
g.figure.suptitle(
    "Cross-fitted occupancy of aging-associated cell states", fontweight="bold", y=1.04
)
g._legend.set_title("sp")
g._legend.set_bbox_to_anchor((1.005, 0.5))
g._legend.set_loc("center left")
g.figure.tight_layout(rect=(0.018, 0.06, 0.91, 1))
g.savefig(
    OUT / "MainFigure1_crossfitted_aging_state_occupancy.pdf", bbox_inches="tight"
)
g.savefig(
    OUT / "MainFigure1_crossfitted_aging_state_occupancy.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(g.figure)

# Figure 2: magnitude plus geometry.
w90 = wide[np.isclose(wide["quantile"], 0.9)].copy()
fig, axs = plt.subplots(1, 2, figsize=(10.6, 5.101))
axs[0].scatter(
    w90.Human,
    w90.Mouse,
    s=80,
    color="#7A0177",
    edgecolor="black",
    linewidth=0.7,
    zorder=3,
)
axs[0].axhline(0, color="black", lw=0.8)
axs[0].axvline(0, color="black", lw=0.8)
offsets = {
    "Oligodendrocyte": (4, 2),
    "Microglia": (4, 2),
    "Astrocyte": (4, 2),
    "OPC": (4, -10),
    "Vascular-associated": (-112, 5),
    "Inhibitory neuron": (-96, -13),
    "Excitatory neuron": (-112, -2),
}
for _, r in w90.iterrows():
    axs[0].annotate(
        r.cell_type,
        (r.Human, r.Mouse),
        xytext=offsets[r.cell_type],
        textcoords="offset points",
        fontsize=8,
    )
axs[0].set(
    xlabel="Human Δ aging-state occupancy (old − young)",
    ylabel="Mouse Δ aging-state occupancy (old − young)",
    title="Cross-species aging-state accumulation",
)
aa = align.sort_values("axis_cosine")
y = np.arange(len(aa))
axs[1].errorbar(
    aa.axis_cosine,
    y,
    xerr=[aa.axis_cosine - aa.bootstrap_ci_low, aa.bootstrap_ci_high - aa.axis_cosine],
    fmt="o",
    color="#2166AC",
    ecolor="black",
    capsize=3,
)
axs[1].axvline(0, color="black", lw=0.8, ls="--")
axs[1].set(
    yticks=y,
    yticklabels=aa.cell_type,
    xlabel="Human–Mouse aging-direction cosine",
    ylabel="",
    title="Cross-species aging geometry",
)
fig.suptitle(
    "Cross-species concordance in aging-state occupancy and direction",
    fontweight="bold",
    y=0.98,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "MainFigure2_occupancy_direction_concordance.pdf")
fig.savefig(
    OUT / "MainFigure2_occupancy_direction_concordance.png",
    dpi=100,
)
plt.close(fig)

# Figure 3: Q90 regional effects and within-animal contrast.
rq = reg[np.isclose(reg["quantile"], 0.9)]
means = (
    rq.groupby(["region", "cell_type", "age_stage"])
    .aging_state_occupancy.mean()
    .unstack()
)
delta = (
    (means.old - means.young)
    .unstack("cell_type")
    .reindex(index=["CC/ACO", "STR", "CTX"], columns=["Oligodendrocyte", "Microglia"])
)
fig, axs = plt.subplots(
    1, 3, figsize=(9.6, 5.101), gridspec_kw={"width_ratios": [0.95, 1, 1]}
)
sns.heatmap(
    delta,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    vmin=0,
    vmax=max(0.42, float(delta.max().max())),
    linewidths=1,
    linecolor="black",
    cbar_kws={"label": "old − young occupancy change"},
    ax=axs[0],
)
axs[0].set(xlabel="", ylabel="", title="Regional occupancy change")
axs[0].tick_params(axis="x", rotation=25)
axs[0].tick_params(axis="y", rotation=0)
perm_p = {}
perm_file = OUT / "full_pipeline_permutation_statistics.csv"
if perm_file.exists():
    pp = pd.read_csv(perm_file).set_index("metric").full_pipeline_permutation_p
    perm_p = {
        "Oligodendrocyte": pp.get("Oligodendrocyte_CC_minus_CTX_occupancy_rho", np.nan),
        "Microglia": pp.get("Microglia_CC_minus_CTX_occupancy_rho", np.nan),
    }
for ax, ct in zip(axs[1:], ["Oligodendrocyte", "Microglia"]):
    z = (
        rq[rq.cell_type.eq(ct)]
        .pivot_table(
            index=["donor", "age_stage", "stage"],
            columns="region",
            values="aging_state_occupancy",
        )
        .dropna()
        .reset_index()
    )
    z["contrast"] = z["CC/ACO"] - z.CTX
    sns.stripplot(
        data=z,
        x="age_stage",
        y="contrast",
        order=STAGES,
        hue="age_stage",
        palette={"young": "#4C78A8", "middle": "#F2A541", "old": "#B2182B"},
        jitter=0.13,
        size=6,
        edgecolor="black",
        linewidth=0.6,
        legend=False,
        ax=ax,
    )
    med = z.groupby("age_stage").contrast.median().reindex(STAGES)
    ax.plot(range(3), med, color="black", marker="D", lw=1.5, ms=6)
    ax.axhline(0, color="black", ls="--", lw=1)
    sr = spearmanr(z.stage, z.contrast)
    ptext = (
        rf"$P_{{perm}}$ = {perm_p[ct]:.3f}"
        if ct in perm_p and np.isfinite(perm_p[ct])
        else rf"$P$ = {sr.pvalue:.1e}"
    )
    ax.text(
        0.04,
        0.96,
        rf"$\rho$ = {sr.statistic:.2f}, {ptext}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax.set(
        xlabel="Relative age stage",
        ylabel="CC/ACO − CTX aging-state occupancy",
        title=ct,
    )
fig.suptitle(
    "White-matter-enriched remodeling of aging-associated glial states",
    fontweight="bold",
    y=0.98,
)
fig.text(
    0.5,
    0.005,
    "LODO cross-fitting; region- and lineage-specific donor-balanced young references (Q90).",
    ha="center",
    fontsize=9,
    color=".3",
)
fig.tight_layout(rect=(0, 0.04, 1, 0.93))
fig.savefig(OUT / "MainFigure3_white_matter_enriched_remodeling.pdf")
fig.savefig(
    OUT / "MainFigure3_white_matter_enriched_remodeling.png",
    dpi=100,
)
plt.close(fig)

print(
    "\nQ90 EFFECTS\n",
    w90.sort_values("conservative_joint_occupancy_effect", ascending=False).to_string(
        index=False
    ),
)
print(
    "\nALIGNMENT\n",
    align.sort_values("axis_cosine", ascending=False).to_string(index=False),
)
print("\nREGIONAL SENSITIVITY\n", rstats.to_string(index=False))
