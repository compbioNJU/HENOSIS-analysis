#!/usr/bin/env python3
"""Step 3: does `X_HENOSIS` carry a developmental direction, and is it shared?

Four questions, each with its own figure and its own table:

    Fig1  where do the cells of a lineage sit across developmental time?
    Fig2  do two species' astroglia move in the SAME direction in X_HENOSIS?
    Fig3  can one species' stage be predicted from the others'?
    Fig4  which cell types show a conserved direction, and which do not?

The unit of inference is a sample centroid everywhere. Cells share a library, a
dissection and a donor, so a cell-level p-value would describe cell-to-cell
noise and read as far stronger evidence than the design supports.

What changes in this release
----------------------------
Human Carnegie stages 18-23 are no longer astroglia, so the human astroglia
trajectory now starts at 11wpc. The cells themselves are still in the atlas and
still have a direction, under the name `progenitor`; because that class exists in
human only, it can never enter a cross-species comparison, and it is reported as
single-species rather than dropped (`single_species_celltypes.csv`).

Outputs
-------
    results/03_developmental/
        Fig1_<lineage>_stage_umap_small_multiples.{pdf,png}   one per focus lineage
        Fig2_astroglia_direction_cosine_heatmap.{pdf,png}
        Fig3_astroglia_LOSO_stage_prediction.{pdf,png}
        Fig4_all_celltype_direction_conservation.{pdf,png}
        celltype_stage_design.csv
        sample_centroid_metadata.csv, sample_centroid_embedding_X_HENOSIS.npy
        developmental_direction_design.csv
        astroglia_pairwise_direction_cosine.csv
        astroglia_pairwise_permutation_pvalues.csv
        astroglia_permutation_null.npz
        astroglia_LOSO_stage_predictions.csv
        astroglia_LOSO_stage_prediction_metrics.csv
        loso_stage_rank_vs_stage_index.csv
        all_celltype_developmental_direction_conservation.csv
        single_species_celltypes.csv
        analysis_summary.txt, developmental_direction_summary.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import io, labels, plotting, stats  # noqa: E402
from common.config import ensure_dir, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--permutations", type=int, default=None,
                        help="override analysis.permutations")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Fig 1: lineage small multiples
# ---------------------------------------------------------------------------

def plot_lineage_small_multiples(obs: pd.DataFrame, xy: np.ndarray, lineage: str,
                                 out: Path, dpi: int) -> list[Path]:
    """One panel per species, that lineage's cells coloured by stage rank."""
    members = obs.loc[obs["Celltype"].astype(str).eq(lineage)]
    if members.empty:
        print(f"[SKIP] {lineage}: no cells")
        return []
    species_list = [s for s in labels.SPECIES_ORDER if s in set(members["species"])]
    ncols = len(species_list)
    fig, axes = plt.subplots(1, ncols, figsize=(2.6 * ncols + 0.8, 3.2), squeeze=False)

    for ax, species in zip(axes.ravel(), species_list):
        subset = members.loc[members["species"].eq(species)]
        plotting.scatter_points(ax, xy, color="#E8E8E8", size=0.05, alpha=0.5, zorder=1)
        positions = subset["pos"].to_numpy()
        colors = plt.get_cmap(labels.STAGE_CMAP)(subset["stage_rank"].to_numpy())
        plotting.scatter_points(ax, xy[positions], color=colors, size=0.6,
                                alpha=0.9, zorder=2)
        # trajectory through the per-stage medians, in rank order
        medians = (subset.groupby(["stage_rank", "stage"], observed=True)
                   .apply(lambda g: pd.Series(
                       {"x": np.median(xy[g["pos"].to_numpy(), 0]),
                        "y": np.median(xy[g["pos"].to_numpy(), 1])}),
                       include_groups=False)
                   .reset_index().sort_values("stage_rank"))
        if len(medians) > 1:
            ax.plot(medians["x"], medians["y"], color="#222222", lw=0.8, alpha=0.8,
                    zorder=3)
        ax.scatter(medians["x"], medians["y"], s=12, c=medians["stage_rank"],
                   cmap=labels.STAGE_CMAP, vmin=0, vmax=1, edgecolor="black",
                   linewidth=0.4, zorder=4)
        plotting.clean_axis(ax)
        ax.set_title(f"{species}\n{len(subset):,} cells, "
                     f"{subset['stage'].nunique()} stages", fontsize=8)

    fig.suptitle(f"{lineage} across developmental time", fontsize=10)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.78, bottom=0.04, wspace=0.12)
    slug = lineage.split(" (")[0].replace(" ", "_").replace("/", "_")
    return list(plotting.save_figure(
        fig, out / f"Fig1_{slug}_stage_umap_small_multiples", dpi=dpi,
        transparent=True,
    ))


# ---------------------------------------------------------------------------
# Fig 2: direction cosine between species
# ---------------------------------------------------------------------------

def plot_direction_heatmap(cosine: pd.DataFrame, pvalues: pd.DataFrame,
                           lineage: str, out: Path, dpi: int) -> list[Path]:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    cmap = plotting.get_cmap("vlag", fallback="coolwarm")
    image = ax.imshow(cosine.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cosine.columns)))
    ax.set_xticklabels(cosine.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cosine.index)))
    ax.set_yticklabels(cosine.index, fontsize=8)
    for i in range(len(cosine.index)):
        for j in range(len(cosine.columns)):
            value = cosine.iloc[i, j]
            if not np.isfinite(value):
                continue
            p = pvalues.iloc[i, j]
            text = f"{value:.2f}" + ("*" if np.isfinite(p) and p < 0.05 else "")
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if abs(value) > 0.55 else "black")
    ax.set_title(f"{lineage}: cosine between species'\ndevelopmental directions "
                 f"(* permutation p < 0.05)", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).set_label(
        "cosine of unit directions", fontsize=8)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.82, bottom=0.20)
    return list(plotting.save_figure(
        fig, out / f"Fig2_{lineage.split(' (')[0]}_direction_cosine_heatmap", dpi=dpi))


# ---------------------------------------------------------------------------
# Fig 3: leave-one-species-out
# ---------------------------------------------------------------------------

def plot_loso(predictions: pd.DataFrame, metrics: pd.DataFrame, lineage: str,
              out: Path, dpi: int) -> list[Path]:
    plot = predictions.dropna(subset=["stage_rank", "predicted_stage_rank"])
    species_list = [s for s in labels.SPECIES_ORDER
                    if s in set(plot["held_out_species"])]
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for species in species_list:
        subset = plot.loc[plot["held_out_species"].eq(species)]
        ax.scatter(subset["stage_rank"], subset["predicted_stage_rank"], s=26,
                   color=labels.SPECIES_COLORS.get(species, "#333333"),
                   edgecolor="white", linewidth=0.4, zorder=3,
                   label=f"{species} (rho={metrics.set_index('held_out_species').loc[species, 'spearman_rho']:.2f})"
                   if species in set(metrics["held_out_species"]) else species)
    ax.set_xlabel("observed relative stage", fontsize=8)
    ax.set_ylabel("predicted relative stage (held-out species)", fontsize=8)
    ax.set_title(f"{lineage}: leave-one-species-out\nstage prediction", fontsize=9)
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.88, bottom=0.12)
    return list(plotting.save_figure(
        fig, out / f"Fig3_{lineage.split(' (')[0]}_LOSO_stage_prediction", dpi=dpi))


# ---------------------------------------------------------------------------
# Fig 4: conservation across cell types
# ---------------------------------------------------------------------------

def plot_conservation(conservation: pd.DataFrame, out: Path, dpi: int) -> list[Path]:
    frame = conservation.sort_values("mean_direction_cosine")
    fig, ax = plt.subplots(figsize=(6.4, max(3.4, 0.34 * len(frame) + 1.2)))
    colors = plt.get_cmap("viridis")(
        (frame["n_species"] - frame["n_species"].min())
        / max(frame["n_species"].max() - frame["n_species"].min(), 1)
    )
    ax.barh(range(len(frame)), frame["mean_direction_cosine"], color=colors)
    ax.set_yticks(range(len(frame)))
    ax.set_yticklabels(frame["Celltype"], fontsize=7)
    ax.axvline(0, color="#555555", lw=0.7)
    ax.set_xlabel("mean pairwise cosine between species' developmental directions",
                  fontsize=8)
    ax.set_xlim(-1, 1)
    for i, (value, n) in enumerate(zip(frame["mean_direction_cosine"],
                                       frame["n_species"])):
        ax.text(value + (0.02 if value >= 0 else -0.02), i, f"n={int(n)}",
                va="center", ha="left" if value >= 0 else "right", fontsize=6,
                color="#444444")
    ax.set_title("Direction conservation by cell type", fontsize=9)
    fig.subplots_adjust(left=0.26, right=0.97, top=0.93, bottom=0.10)
    return list(plotting.save_figure(
        fig, out / "Fig4_all_celltype_direction_conservation", dpi=dpi))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    plotting.apply_style(args.dpi)

    h5ad = cfg.require_input("h5ad")
    min_cells = int(cfg.param("min_cells_per_sample"))
    min_samples = int(cfg.param("min_samples_for_direction"))
    min_stages = int(cfg.param("min_stages_for_direction"))
    n_perm = args.permutations or int(cfg.param("permutations"))
    seed = args.seed if args.seed is not None else int(cfg.param("seed"))
    astroglia = str(cfg.param("astroglia_label"))
    focus = [str(c) for c in cfg.param("focus_celltypes")]
    out = ensure_dir(cfg.result_dir("03_developmental"))

    obs, arrays, info = io.load_design_multi(
        h5ad, obs_columns=["Celltype", "Common Name"], embeddings=("X_HENOSIS", "X_umap")
    )
    obs = obs.assign(pos=np.arange(len(obs)))
    embedding = arrays["X_HENOSIS"]
    xy = arrays["X_umap"][:, :2].astype(np.float64)
    print(f"[READ] {info['path']}  ({info['n_obs']:,} cells, X_HENOSIS {embedding.shape[1]}-d)")

    # -- design table --------------------------------------------------------
    design = (obs.groupby(["Celltype", "species", "stage", "stage_rank"], observed=True)
              .size().rename("n_cells").reset_index())
    io.write_table(out / "celltype_stage_design.csv", design)

    # -- sample centroids ----------------------------------------------------
    # `stage_index` rides along so the sensitivity of the result to the rank
    # calibration can be reported from the same centroids, rather than by
    # rebuilding them under a second convention.
    keys = ["species", "sample_id", "stage", "stage_rank", "stage_index", "Celltype"]
    meta, vectors = stats.sample_centroids(obs, embedding, keys, min_cells)
    kept = meta["dropped"].eq(False)
    centroids = meta.loc[kept].reset_index(drop=True)
    centroid_vectors = vectors.astype(np.float32)
    io.write_table(out / "sample_centroid_metadata.csv", centroids)
    np.save(out / "sample_centroid_embedding_X_HENOSIS.npy", centroid_vectors)
    io.write_table(out / "sample_centroids_dropped_groups.csv",
                   stats.dropped_report(meta, keys))
    print(f"[CENTROIDS] {len(centroids):,} sample x cell-type centroids "
          f"(>= {min_cells} cells); "
          f"{int(meta['dropped'].sum()):,} groups below threshold")

    # -- directions ----------------------------------------------------------
    frame = centroids.assign(_row=np.arange(len(centroids)))
    directions: dict[tuple[str, str], np.ndarray] = {}
    direction_rows = []
    for (celltype, species), group in frame.groupby(["Celltype", "species"],
                                                    observed=True):
        rows = group["_row"].to_numpy()
        direction = stats.slope_direction(
            centroid_vectors[rows].astype(float), group["stage_rank"].to_numpy(float)
        )
        estimable = (direction is not None
                     and group["stage_rank"].nunique() >= min_stages
                     and len(group) >= min_samples)
        direction_rows.append({
            "Celltype": celltype, "species": species,
            "n_samples": int(len(group)),
            "n_stages": int(group["stage_rank"].nunique()),
            "n_cells": int(group["n_cells"].sum()),
            "estimable": bool(estimable),
            "reason": "" if estimable else _not_estimable(
                direction, group, min_stages, min_samples),
        })
        if estimable:
            directions[(celltype, species)] = direction
    direction_design = pd.DataFrame(direction_rows).sort_values(["Celltype", "species"])
    io.write_table(out / "developmental_direction_design.csv", direction_design)
    print(f"[DIRECTIONS] {len(directions)} estimable (cell type, species) directions")

    # which cell types are single-species by construction
    present = (obs.groupby("Celltype", observed=True)["species"].nunique()
               .rename("n_species").reset_index())
    present["is_single_species"] = present["n_species"] < 2
    single = present.loc[present["is_single_species"]]
    io.write_table(out / "single_species_celltypes.csv", single)
    for row in single.itertuples():
        print(f"  [SINGLE-SPECIES] {row.Celltype}: present in {int(row.n_species)} species, "
              f"so it cannot enter a cross-species direction comparison")

    # -- Fig 1 ---------------------------------------------------------------
    lineages = focus + [str(cfg.param("progenitor_label"))]
    for lineage in dict.fromkeys(lineages):
        plot_lineage_small_multiples(obs, xy, lineage, out, args.dpi)

    # -- Fig 2: astroglia direction cosine + permutation ---------------------
    #
    # `groupby(...).indices` yields positions INSIDE the grouped frame, not
    # positions in `centroid_vectors`. Using them directly would silently read
    # another cell type's rows and still produce a plausible cosine. So each
    # astroglia row carries its global position in `_row`, and that is what is
    # used to index the vectors.
    rng = np.random.default_rng(seed)
    astro_frame = centroids.loc[centroids["Celltype"].eq(astroglia)].reset_index(drop=True)
    astro_frame = astro_frame.assign(
        _global=centroids.index[centroids["Celltype"].eq(astroglia).to_numpy()]
    )
    species_present, null_draws = [], {}
    by_species = {
        s: astro_frame["_global"].to_numpy()[idx]
        for s, idx in astro_frame.groupby("species", observed=True).indices.items()
    }
    dir_astro = {s: directions[(astroglia, s)] for s in by_species
                 if (astroglia, s) in directions}
    for species, direction in sorted(dir_astro.items()):
        rows = by_species[species]
        t = astro_frame.set_index("_global").loc[rows, "stage_rank"].to_numpy(float)
        x = centroid_vectors[rows].astype(float)
        draws = np.array([
            d for d in (stats.slope_direction(x, rng.permutation(t))
                        for _ in range(n_perm)) if d is not None
        ])
        if draws.ndim != 2 or draws.shape[1] != centroid_vectors.shape[1]:
            raise AssertionError(
                f"{species}: permutation draws have shape {draws.shape}; expected "
                f"(n, {centroid_vectors.shape[1]}). An empty or ragged null would "
                f"make the permutation p-value meaningless."
            )
        null_draws[species] = draws
        species_present.append(species)

    cosine = pd.DataFrame(np.nan, index=species_present, columns=species_present)
    pvalues = pd.DataFrame(np.nan, index=species_present, columns=species_present)
    for i, a in enumerate(species_present):
        for b in species_present[i:]:
            observed = float(np.dot(dir_astro[a], dir_astro[b]))
            null = null_draws[a] @ null_draws[b].T
            p = stats.permutation_pvalue(observed, null)
            cosine.loc[a, b] = cosine.loc[b, a] = observed
            pvalues.loc[a, b] = pvalues.loc[b, a] = p
    io.write_table(out / "astroglia_pairwise_direction_cosine.csv", cosine, index=True)
    io.write_table(out / "astroglia_pairwise_permutation_pvalues.csv", pvalues, index=True)
    np.savez_compressed(out / "astroglia_permutation_null.npz",
                        **{k: v for k, v in null_draws.items()})
    plot_direction_heatmap(cosine, pvalues, astroglia, out, args.dpi)
    print("\n[FIG2] astroglia direction cosine (lower triangle, * = p<0.05):")
    print(_lower_triangle(cosine, pvalues).round(3).to_string())

    # -- Fig 3: LOSO ---------------------------------------------------------
    astro_rows = frame.loc[frame["Celltype"].eq(astroglia)].reset_index(drop=True)
    astro_vectors = centroid_vectors[frame.loc[frame["Celltype"].eq(astroglia),
                                              "_row"].to_numpy()]
    pred, metrics = stats.loso_stage_prediction(astro_rows, astro_vectors)
    io.write_table(out / "astroglia_LOSO_stage_predictions.csv", pred)
    io.write_table(out / "astroglia_LOSO_stage_prediction_metrics.csv", metrics)
    plot_loso(pred, metrics, astroglia, out, args.dpi)

    # the rank calibration is a choice; report how much the answer depends on it
    sens_rows = []
    for rank_column in ("stage_rank", "stage_index"):
        frame_alt = astro_rows.assign(stage_rank=astro_rows[rank_column])
        _, alt_metrics = stats.loso_stage_prediction(frame_alt, astro_vectors)
        for row in alt_metrics.itertuples():
            sens_rows.append({
                "rank_convention": rank_column,
                "held_out_species": row.held_out_species,
                "n_samples": row.n_samples,
                "spearman_rho": row.spearman_rho,
                "spearman_p": row.spearman_p,
            })
    sensitivity = pd.DataFrame(sens_rows)
    io.write_table(out / "loso_stage_rank_vs_stage_index.csv", sensitivity)
    print("\n[FIG3] leave-one-species-out stage prediction:")
    print(metrics.round(4).to_string(index=False))
    print("  same test under the even-spaced rank calibration (sensitivity):")
    print(sensitivity.pivot(index="held_out_species", columns="rank_convention",
                            values="spearman_rho").round(3).to_string())

    # -- Fig 4: conservation -------------------------------------------------
    conservation_rows = []
    for celltype in sorted({ct for ct, _ in directions}):
        available = [s for s in species_present if (celltype, s) in directions]
        available = [s for s in labels.SPECIES_ORDER if s in set(available)]
        if len(available) < 2:
            continue
        pairwise = [
            float(np.dot(directions[(celltype, a)], directions[(celltype, b)]))
            for i, a in enumerate(available) for b in available[i + 1:]
        ]
        conservation_rows.append({
            "Celltype": celltype,
            "n_species": len(available),
            "n_species_pairs": len(pairwise),
            "mean_direction_cosine": float(np.mean(pairwise)),
            "median_direction_cosine": float(np.median(pairwise)),
            "min_direction_cosine": float(np.min(pairwise)),
            "max_direction_cosine": float(np.max(pairwise)),
            "species": ";".join(available),
        })
    conservation = (pd.DataFrame(conservation_rows)
                    .sort_values("mean_direction_cosine", ascending=False)
                    .reset_index(drop=True))
    io.write_table(out / "all_celltype_developmental_direction_conservation.csv", conservation)
    if len(conservation):
        plot_conservation(conservation, out, args.dpi)
    print("\n[FIG4] direction conservation (mean cosine across species pairs):")
    print(conservation.round(3).to_string(index=False))

    # -- summary -------------------------------------------------------------
    astro_rho = metrics.set_index("held_out_species")["spearman_rho"] if len(metrics) else pd.Series(dtype=float)
    summary = [
        f"input: {info['path']}",
        f"cells: {info['n_obs']:,}   embedding: obsm['X_HENOSIS'] ({embedding.shape[1]}-d)",
        f"independent unit: sample x cell-type centroid (>= {min_cells} cells)",
        f"sample centroids retained: {len(centroids):,}",
        f"estimable (cell type, species) directions: {len(directions)}",
        "",
        f"astroglia = '{astroglia}'",
        "  human Carnegie stage 18-23 contributes 0 cells in this release; "
        "human astroglia now start at 11wpc",
        f"  species with an estimable astroglia direction: {', '.join(species_present)}",
        "  LOSO Spearman rho: "
        + ", ".join(f"{s}={astro_rho.get(s, float('nan')):.2f}" for s in species_present),
        "",
        "single-species cell types (cannot enter a cross-species comparison): "
        + ", ".join(single["Celltype"].tolist()),
        "",
        "stage rank convention: labels.STAGE_RANK (relative age in [0,1]); "
        "the even-spaced alternative is reported in loso_stage_rank_vs_stage_index.csv",
        f"permutations: {n_perm}, seed: {seed}",
    ]
    text = "\n".join(summary) + "\n"
    (out / "analysis_summary.txt").write_text(text, encoding="utf-8")
    io.write_json(out / "developmental_direction_summary.json", {
        "input": info["path"],
        "n_cells": info["n_obs"],
        "embedding_dimensions": int(embedding.shape[1]),
        "min_cells_per_sample": min_cells,
        "min_samples_for_direction": min_samples,
        "min_stages_for_direction": min_stages,
        "permutations": n_perm,
        "seed": seed,
        "n_centroids": int(len(centroids)),
        "n_directions": len(directions),
        "astroglia_label": astroglia,
        "astroglia_species_with_direction": species_present,
        "astroglia_loso_spearman": {
            row.held_out_species: row.spearman_rho for row in metrics.itertuples()
        },
        "single_species_celltypes": single["Celltype"].tolist(),
        "top_conserved": conservation.head(5).to_dict(orient="records"),
        "bottom_conserved": conservation.tail(5).to_dict(orient="records"),
        "release_edit": (
            "human Carnegie stages 18-23 are 'progenitor' and are human-only, so "
            "they are absent from the astroglia trajectory and from every "
            "cross-species direction comparison"
        ),
    })
    print("\n" + text)
    print(f"[DONE] {out}")
    return 0


def _not_estimable(direction, group, min_stages, min_samples) -> str:
    """Why a (cell type, species) group has no direction, in the words of the rule."""
    if direction is None:
        return "no direction identifiable (one distinct stage, or zero spread in stage)"
    if group["stage_rank"].nunique() < min_stages:
        return f"only {int(group['stage_rank'].nunique())} distinct stages (< {min_stages})"
    if len(group) < min_samples:
        return f"only {len(group)} samples (< {min_samples})"
    return "unknown"


def _lower_triangle(cosine: pd.DataFrame, pvalues: pd.DataFrame) -> pd.DataFrame:
    """The pairwise values only (strict lower triangle), with p<0.05 marked.

    The matrix is symmetric, so printing all of it doubles every number and makes
    the table harder to check against the figure. The diagonal is each species
    against itself (cosine 1 by construction) and carries no information.
    """
    mask = np.tril(np.ones(cosine.shape, dtype=bool), k=-1)
    values = cosine.where(mask)
    stars = pvalues.where(mask)
    return pd.DataFrame(
        np.where(stars.to_numpy() < 0.05, "  *", ""),
        index=values.index, columns=values.columns,
    ).radd(values.round(3).astype(str))


if __name__ == "__main__":
    raise SystemExit(main())
