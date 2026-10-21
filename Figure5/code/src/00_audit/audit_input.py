#!/usr/bin/env python3
"""Step 0: report what the input file actually contains, before anything is drawn.

Writes `results/00_audit/` :
    input_report.json        files, shapes, obsm keys, per-species/stage/celltype counts
    input_report.txt         the same, as prose
    stage_vocabulary.csv     the species x stage x rank table the analyses use
    celltype_palette.csv     the cell-type -> colour map the figures use
    cell_counts_celltype_by_species.csv
    cell_counts_celltype_by_stage.csv
    release_edit_check.csv   the relabelling this release is defined by, verified

Why this is a step and not a comment: three separate claims are made about the
input elsewhere in this repository -- that human Carnegie stages 18-23 carry
`progenitor`, that the palette is complete, and that every (species, stage) pair
has a rank. Each is checked here and the outcome is written down.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from common import io, labels, plotting  # noqa: E402
from common.config import ensure_dir, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    plotting.apply_style()

    h5ad = cfg.require_input("h5ad")
    out = ensure_dir(cfg.result_dir("00_audit"))

    print(f"[READ] {h5ad}")
    report = io.audit_input(h5ad)

    obs, _, info = io.load_design(
        h5ad, obs_columns=["Org_Celltype", "Celltype", "Species", "Common Name", "Org_stage"]
    )
    report["design_columns"] = list(labels.Design.__dataclass_fields__)

    # -- 1. the relabelling this release is defined by ------------------------
    #
    # Only the ASTROGLIA part of human Carnegie stages 18-23 was relabelled. The
    # other ~23k cells at those stages (Purkinje, granule cells, VZ neuroblasts,
    # ...) already carried correct labels and are untouched. So the claim is not
    # "every CS18-23 cell is a progenitor" -- that would be false, 8,855 of 32,333
    # is the right number -- but the two things the relabel actually asserts: the
    # astroglia label is gone from those stages, and every `progenitor` cell is
    # one of those stages in human.
    human_cs = obs["species"].eq("Human") & obs["stage"].isin(["CS18-19", "CS20", "CS22-23"])
    is_progenitor = obs["Celltype"].eq(labels.PROGENITOR_LABEL)
    is_astroglia = obs["Celltype"].eq(labels.ASTROGLIA_FAMILY_LABEL)
    checks = [
        {
            "check": "no human Carnegie stage 18-23 cell is labelled astroglia",
            "expected": "0 cells",
            "observed": f"{int((human_cs & is_astroglia).sum()):,}",
            "passes": bool((human_cs & is_astroglia).sum() == 0),
        },
        {
            "check": "every 'progenitor' cell is human and at Carnegie stage 18-23",
            "expected": "all",
            "observed": f"{int((is_progenitor & human_cs).sum()):,} / {int(is_progenitor.sum()):,}",
            "passes": bool((is_progenitor & human_cs).sum() == is_progenitor.sum()),
        },
        {
            "check": "'progenitor' occurs only in human",
            "expected": "human only",
            "observed": ", ".join(sorted(obs.loc[is_progenitor, "species"].unique())),
            "passes": bool(set(obs.loc[is_progenitor, "species"].unique()) <= {"Human"}),
        },
        {
            "check": "no 'progenitor' cell is at a non-Carnegie human stage",
            "expected": "0 cells",
            "observed": f"{int((is_progenitor & ~human_cs).sum()):,}",
            "passes": bool((is_progenitor & ~human_cs).sum() == 0),
        },
        {
            "check": "every (species, stage) pair has a developmental rank",
            "expected": "no NaN",
            "observed": f"{int(obs['stage_rank'].isna().sum()):,} missing",
            "passes": bool(obs["stage_rank"].notna().all()),
        },
    ]
    edits = pd.DataFrame(checks)
    io.write_table(out / "release_edit_check.csv", edits)
    for row in edits.itertuples():
        print(f"  [{'PASS' if row.passes else 'FAIL'}] {row.check}: {row.observed}")

    # -- 2. vocabulary and palette -------------------------------------------
    io.write_table(out / "stage_vocabulary.csv", labels.stage_rank_table())

    present = sorted(obs["Celltype"].astype(str).unique())
    palette = {label: labels.CELLTYPE_COLORS.get(label, "") for label in labels.celltype_order(present)}
    missing = [label for label, color in palette.items() if not color]
    if missing:
        raise SystemExit(
            f"cell type(s) {missing} have no colour in labels.CELLTYPE_COLORS. "
            f"Add them there: an unlisted label would otherwise be coloured by "
            f"whichever colormap index it happened to fall on."
        )
    io.write_table(out / "celltype_palette.csv",
                   pd.DataFrame({"Celltype": list(palette), "color": list(palette.values())}))

    # -- 3. composition ------------------------------------------------------
    counts = (obs.groupby(["Celltype", "species"], observed=True).size()
              .rename("n_cells").reset_index())
    io.write_table(out / "cell_counts_celltype_by_species.csv", counts)
    stage_counts = (obs.groupby(["Celltype", "species", "stage"], observed=True).size()
                    .rename("n_cells").reset_index())
    io.write_table(out / "cell_counts_celltype_by_stage.csv", stage_counts)

    report["celltype_palette"] = palette
    report["release_edit_check"] = edits.to_dict(orient="records")
    io.write_json(out / "input_report.json", report)

    summary = [
        f"input: {info['path']}",
        f"cells: {info['n_obs']:,}   variables: {info['n_vars']}",
        f"obsm: {info['obsm_keys']}",
        f"species ({len(report['species_cells'])}): "
        + ", ".join(f"{k}={v:,}" for k, v in sorted(report["species_cells"].items())),
        f"cell types ({len(present)}): "
        + ", ".join(f"{label}={report['celltype_cells'].get(label, 0):,}"
                    for label in labels.celltype_order(present)),
        "",
        "release edit: the astroglia of human Carnegie stage 18-23 moved from "
        f"'{labels.ASTROGLIA_FAMILY_LABEL}' to '{labels.PROGENITOR_LABEL}'",
        f"  human CS18-23 cells (all cell types): {int(human_cs.sum()):,}",
        f"  of which now '{labels.PROGENITOR_LABEL}': {int((is_progenitor & human_cs).sum()):,}"
        f"   (the rest keep their previous labels)",
        f"  '{labels.ASTROGLIA_FAMILY_LABEL}' total: {int(is_astroglia.sum()):,}"
        f"   (human: {int((is_astroglia & obs['species'].eq('Human')).sum()):,},"
        f" no Carnegie stage)",
        f"  '{labels.PROGENITOR_LABEL}' total: {int(is_progenitor.sum()):,}",
        f"  all release-edit checks pass: {bool(edits['passes'].all())}",
        "",
        "stale in the input file, deliberately unused:",
        *[f"  uns['{key}']: {value}" for key, value in report.get("stale_uns", {}).items()],
    ]
    (out / "input_report.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    print(f"\n[DONE] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
