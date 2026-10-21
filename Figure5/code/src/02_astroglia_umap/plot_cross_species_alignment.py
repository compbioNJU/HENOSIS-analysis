#!/usr/bin/env python3
"""Step 2b: three pairwise cross-species developmental alignments.

Marmoset -> Human, Mouse -> Human and Marmoset -> Mouse, computed from `X_HENOSIS`
by the two-level averaging and cell-type-balanced cosine distance described in
`common/alignment.py`.

Which cells enter
-----------------
ALL cell types of the atlas, not only astroglia. The cost at each stage pair is
the mean cosine distance over the cell types the two stages share, so a stage
pair is described by the whole regulatory state of the tissue at that point, and
the comparison is not restricted to one lineage's view of development. Deep
nuclei and white-matter samples are excluded (`special_anatomy`), and the number
excluded is written to `alignment_excluded_samples.csv`.

What changes in this release
----------------------------
`progenitor` exists as a cell type. It is human-only, so it can never appear in
the shared-cell-type set of a cross-species pair, and the cost matrices are
therefore unchanged in construction -- but the human Carnegie-stage columns now
share one cell type fewer with the other species, which is the intended effect of
removing the false astroglia equivalence.

Outputs (`results/02_astroglia_umap/`)
    cross_species_developmental_alignment_3pairs.{pdf,png}
    stage_cost_<source>_to_<target>.csv
    stage_shared_celltypes_<source>_to_<target>.csv
    stage_DTW_path_<source>_to_<target>.csv
    alignment_dropped_groups.csv, alignment_excluded_samples.csv, alignment_summary.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from common import alignment as align  # noqa: E402
from common import io, labels, plotting  # noqa: E402
from common.config import ensure_dir, load_config  # noqa: E402

# The three pairs whose two species both have a densely sampled series. Only
# these have enough stages on both axes for a DTW correspondence to mean
# anything, and only these are used for the alignment claim.
DENSE_PAIRS: tuple[tuple[str, str], ...] = (
    ("Marmoset", "Human"),
    ("Mouse", "Human"),
    ("Marmoset", "Mouse"),
)

# Every species against human. Bonobo (2 stages), Opossum (2) and Macaque (1)
# cannot support a path, so they contribute their cost matrix -- which is a
# well-defined comparison -- and are reported as having no correspondence
# instead of being given a degenerate one.
TO_HUMAN_PAIRS: tuple[tuple[str, str], ...] = (
    ("Bonobo", "Human"),
    ("Macaque", "Human"),
    ("Marmoset", "Human"),
    ("Mouse", "Human"),
    ("Opossum", "Human"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--include-special-anatomy", action="store_true",
                        help="keep deep-nuclei / white-matter samples (off by default)")
    parser.add_argument("--band", type=int, default=None,
                        help="Sakoe-Chiba band width for the DTW path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    plotting.apply_style(args.dpi)

    h5ad = cfg.require_input("h5ad")
    min_cells = int(cfg.param("min_cells_per_sample"))
    min_stages = int(cfg.param("min_stages_for_dtw"))
    out = ensure_dir(cfg.result_dir("02_astroglia_umap"))

    obs, embedding, info = io.load_design(
        h5ad, obs_columns=["Celltype", "Common Name"], with_embedding="X_HENOSIS"
    )
    obs = obs.assign(pos=np.arange(len(obs)))
    print(f"[READ] {info['path']}  ({info['n_obs']:,} cells, "
          f"X_HENOSIS {embedding.shape[1]}-d)")

    built = align.build_alignment(
        obs, embedding, min_cells=min_cells,
        exclude_special_anatomy=not args.include_special_anatomy,
    )
    print(f"[CENTROIDS] {len(built.centroids):,} sample x cell-type centroids "
          f"(>={min_cells} cells)")
    if len(built.excluded):
        print(f"[EXCLUDED] {len(built.excluded):,} deep-nuclei / white-matter centroids")
    print(f"[STAGE VECTORS] {len(built.stage_vectors):,} (species, stage, cell type) vectors")

    summaries = {}
    for pair_set, pairs, figure_stem, suptitle in (
        ("dense", DENSE_PAIRS, "cross_species_developmental_alignment_3pairs",
         "Cross-species developmental alignment from X_HENOSIS -- the three densely "
         "sampled pairs (cell-type-balanced cosine distance; deep nuclei excluded)"),
        ("to_human", TO_HUMAN_PAIRS, "cross_species_developmental_alignment_to_human",
         "Every species against human -- cost matrices for all five; a DTW "
         "correspondence only where both axes have >= "
         f"{min_stages} stages (deep nuclei excluded)"),
    ):
        print(f"\n--- pair set '{pair_set}' ---")
        results = []
        for source, target in pairs:
            result = align.align_pair(built, source, target,
                                      min_stages_for_dtw=min_stages, band=args.band)
            cost, shared = result["cost"], result["shared"]
            prefix = f"{source.lower()}_to_{target.lower()}"
            io.write_table(out / f"stage_cost_{prefix}.csv", cost, index=True)
            io.write_table(out / f"stage_shared_celltypes_{prefix}.csv", shared, index=True)
            if result["path"]:
                io.write_table(out / f"stage_DTW_path_{prefix}.csv", result["path_table"])
                mapping = ", ".join(
                    f"{s}->{t}" for s, t in zip(
                        result["path_table"]["source_stage"],
                        result["path_table"]["target_stage"],
                    )
                )
                print(f"[{source}->{target}] {len(result['path'])} steps, "
                      f"accumulated cost {result['total_cost']:.3f}")
                print(f"    {mapping}")
            else:
                print(f"[{source}->{target}] no path: {result['reason']}")
            results.append(result)

        if all(r["cost"].notna().any().any() for r in results):
            fig = align.plot_alignment_grid(results, suptitle=suptitle)
            plotting.save_figure(fig, out / figure_stem, dpi=args.dpi, pad=False)

        summaries[pair_set] = [
            {
                "source": r["source"], "target": r["target"],
                "n_steps": len(r["path"]) or None,
                "accumulated_cost": r["total_cost"] if r["path"] else None,
                "no_path_reason": r["reason"],
                "cost_min": (float(np.nanmin(r["cost"].to_numpy()))
                             if r["cost"].notna().any().any() else None),
                "cost_max": (float(np.nanmax(r["cost"].to_numpy()))
                             if r["cost"].notna().any().any() else None),
            }
            for r in results
        ]

    io.write_table(out / "alignment_dropped_groups.csv", built.dropped)
    io.write_table(out / "alignment_excluded_samples.csv", built.excluded)
    io.write_json(out / "alignment_summary.json", {
        "input": info["path"],
        "embedding": f"obsm['X_HENOSIS'] ({embedding.shape[1]}-d)",
        "min_cells_per_sample": min_cells,
        "min_stages_for_dtw": min_stages,
        "dtw_band": args.band,
        "special_anatomy_excluded": not args.include_special_anatomy,
        "n_centroids": int(len(built.centroids)),
        "n_excluded_centroids": int(len(built.excluded)),
        "n_dropped_groups": int(len(built.dropped)),
        "n_stage_vectors": int(len(built.stage_vectors)),
        "pair_sets": summaries,
        "progenitor_note": (
            f"'{labels.PROGENITOR_LABEL}' is human-only, so it is never part of a "
            f"cross-species shared-cell-type set; it changes the human Carnegie-stage "
            f"columns only by removing the astroglia match at those stages"
        ),
    })
    print(f"\n[DONE] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
