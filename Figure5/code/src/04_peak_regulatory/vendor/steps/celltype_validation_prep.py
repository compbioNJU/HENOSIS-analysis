#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Derive the plotting inputs for the Purkinje / Oligo validation figures.

WHY a prep step
---------------
The plotting layer must not form statistics (V2_PLAN section 15-3 rule 1), so
everything the validation plate needs is assembled here:

  * a long per-motif table joining the model's log2 odds ratio to the raw
    chromVAR C, with the cell-type's module membership attached, so the scatter
    can grey the unannotated motifs and colour the modules without ever reading
    the module map itself;
  * a per (module, species) support table in the vocabulary the figure uses --
    `model_F_abs`, `chromvar_C`, the two signs and whether they agree;
  * a per-module summary: how many species agree, and the agreement rate.

WHY the two sources must NOT share an axis
------------------------------------------
`model_F_abs` runs about 0-2 and `chromvar_C` about 4-30 on this data. The
figures that put them on one axis showed every model marker pinned to zero next
to tall raw bars, which reads as "the model sees nothing" when the two are simply
different statistics on different peak universes. The plate therefore draws them
as SEPARATE matrices with their own colour scales, and reports agreement as a
sign count rather than as a shared magnitude.

This mirrors the Astroglia family-trajectory rework, with one deliberate
difference recorded in V2_PLAN 15-4: Purkinje/Oligo have no developmental
axis, so their sources are NOT z-scored. There is no trajectory shape to compare;
the questions are the cross-modal rank association over 879 motifs and the sign
agreement of the pre-registered modules.

    python script/celltype_validation_prep.py --config config/analysis_v2.yaml --celltype Oligo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

STEP_DIR = {"Purkinje": "07_purkinje_validation", "Oligo": "09_oligo_validation"}
PROFILE = {"Purkinje": "purkinje_raw_profile.tsv", "Oligo": "oligo_raw_profile.tsv"}
SUPPORT = {"Purkinje": "module_raw_support.tsv", "Oligo": "module_raw_support.tsv"}
SIMILARITY = {"Purkinje": "model_vs_raw_similarity.tsv", "Oligo": "model_vs_raw_similarity.tsv"}

UNANNOTATED = "unannotated"


def sign_of(values: pd.Series) -> pd.Series:
    out = np.zeros(len(values), dtype=int)
    finite = np.isfinite(values.to_numpy(dtype=float))
    numbers = values.to_numpy(dtype=float)
    out[finite & (numbers > 0)] = 1
    out[finite & (numbers < 0)] = -1
    return pd.Series(out, index=values.index)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(C.CONFIG_DIR / "analysis_v2.yaml"))
    parser.add_argument("--celltype", required=True, choices=sorted(STEP_DIR))
    parser.add_argument("--in-dir", default=None)
    args = parser.parse_args()

    config = C.load_config(args.config)
    celltype = args.celltype
    in_dir = Path(args.in_dir) if args.in_dir else C.results_dir(config, STEP_DIR[celltype])
    stem = celltype.lower()

    profile_path = in_dir / PROFILE[celltype]
    support_path = in_dir / SUPPORT[celltype]
    for path in (profile_path, support_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run the {celltype} validation step first")

    # ---- per-motif long table with module membership ------------------------
    profile = pd.read_csv(profile_path, sep="\t")
    required = {"species", "motif_id", "motif_name", "log2_OR", "C"}
    absent = sorted(required - set(profile.columns))
    if absent:
        raise ValueError(f"{profile_path} lacks columns {absent}")

    module_map = C.celltype_module_table(config)
    membership = module_map[module_map.celltype == celltype][
        ["motif_id", "module", "level", "member_type"]
    ].rename(columns={"level": "module_level"})
    # a motif of this cell type belongs to at most one atomic module (the map
    # enforces that), so a plain left join cannot duplicate a motif
    if membership.motif_id.duplicated().any():
        duplicated = sorted(membership.motif_id[membership.motif_id.duplicated()].unique())[:5]
        raise ValueError(
            f"module map claims {duplicated} under more than one {celltype} module"
        )

    longest = profile.rename(columns={"log2_OR": "model_log2_or", "C": "chromvar_C"})
    longest = longest.merge(membership, on="motif_id", how="left")
    longest["module"] = longest.module.fillna(UNANNOTATED)
    longest["module_level"] = longest.module_level.fillna(UNANNOTATED)
    longest["member_type"] = longest.member_type.fillna(UNANNOTATED)
    longest.insert(0, "celltype", celltype)
    # the scatter greys everything that is not in the cell type's panel; a
    # heterodimer is not drawn as a panel member because a module median is built
    # from monomers only, and the figure must not imply otherwise
    longest["in_panel"] = (longest.module != UNANNOTATED) & (longest.member_type == "monomer")
    longest = longest.drop(columns=["member_type"])

    order = ["celltype", "species", "motif_id", "motif_name", "model_log2_or", "chromvar_C",
             "module", "module_level", "in_panel"]
    longest = longest[order].sort_values(["species", "motif_id"]).reset_index(drop=True)
    out_long = in_dir / f"{stem}_model_vs_chromvar.tsv"
    longest.to_csv(out_long, sep="\t", index=False)
    C.log("wrote", out_long.name, f"({len(longest)} rows, "
          f"{int(longest.in_panel.sum())} in-panel motif rows)")

    # ---- per (module, species) support, in the vocabulary the figure uses ----
    support = pd.read_csv(support_path, sep="\t")
    # The two steps spell the same two fields differently (s07: analysis_level /
    # may_gate, s09: level / gating), so both are accepted rather than one of the
    # two figures silently failing to build.
    aliases = {"module_level": ("module_level", "level", "analysis_level"),
               "gating": ("gating", "may_gate")}
    resolved = {}
    for target, candidates in aliases.items():
        found = next((c for c in candidates if c in support.columns), None)
        if found is None:
            raise ValueError(
                f"{support_path} has none of {candidates} for the {target} column"
            )
        resolved[target] = found
    required_support = {"module", "species", "model_module_score",
                        "raw_module_score", "sign_agree"}
    absent = sorted(required_support - set(support.columns))
    if absent:
        raise ValueError(f"{support_path} lacks columns {absent}")

    keep = {resolved["module_level"]: "module_level",
            resolved["gating"]: "gating",
            "module": "module", "species": "species",
            "model_module_score": "model_F_abs", "raw_module_score": "chromvar_C",
            "sign_agree": "sign_agree",
            "n_members_model": "n_members_model", "n_members_raw": "n_members_raw"}
    keep = {k: v for k, v in keep.items() if k in support.columns}
    tidy = support[list(keep)].rename(columns=keep).copy()
    # provenance is authoritative in the frozen module map, and one of the two
    # support tables does not carry it at all
    provenance = module_map[module_map.celltype == celltype][["module", "provenance"]]
    provenance = provenance.drop_duplicates("module")
    tidy = tidy.merge(provenance, on="module", how="left")
    tidy.insert(0, "celltype", celltype)
    tidy["model_sign"] = sign_of(tidy.model_F_abs)
    tidy["chromvar_sign"] = sign_of(tidy.chromvar_C)
    # recomputed rather than trusted: the upstream `sign_agree` is defined the
    # same way, and a disagreement between the two would be a defect worth seeing
    same_sign = (tidy.model_sign == tidy.chromvar_sign) & (tidy.model_sign != 0)
    if not same_sign.equals(tidy.sign_agree.astype(bool)):
        bad = tidy[~same_sign.eq(tidy.sign_agree.astype(bool))]
        C.log("WARNING: recomputed sign agreement disagrees with the upstream column on "
              f"{len(bad)} row(s): {bad[['module', 'species']].to_dict('records')[:3]}")
    tidy = tidy.sort_values(["module_level", "module", "species"]).reset_index(drop=True)
    out_support = in_dir / f"{stem}_module_support.tsv"
    tidy.to_csv(out_support, sep="\t", index=False)
    C.log("wrote", out_support.name, f"({len(tidy)} rows)")

    # ---- per-module summary -------------------------------------------------
    rows = []
    for (module, level, gating), group in tidy.groupby(
        ["module", "module_level", "gating"], sort=False
    ):
        scored = group[group.sign_agree.notna()]
        rows.append(dict(
            celltype=celltype,
            module=module,
            module_level=level,
            gating=bool(gating),
            n_species=int(len(scored)),
            n_agree=int(scored.sign_agree.sum()),
            agreement_rate=(float(scored.sign_agree.mean()) if len(scored) else np.nan),
            n_model_positive=int((scored.model_sign > 0).sum()),
            n_chromvar_positive=int((scored.chromvar_sign > 0).sum()),
        ))
    # A step whose module layer is blocked upstream writes a header-only support
    # table (s07 does this when 06_purkinje_model/ was not produced). The summary
    # is then an empty table that still carries its columns -- `pd.DataFrame([])`
    # would have no columns and `sort_values` would fail on the `gating` key.
    summary_columns = ["celltype", "module", "module_level", "gating", "n_species",
                       "n_agree", "agreement_rate", "n_model_positive",
                       "n_chromvar_positive"]
    summary = pd.DataFrame(rows, columns=summary_columns).sort_values(
        ["gating", "n_agree"], ascending=[False, False]
    ).reset_index(drop=True)
    out_summary = in_dir / f"{stem}_module_support_summary.tsv"
    summary.to_csv(out_summary, sep="\t", index=False)
    C.log("wrote", out_summary.name, f"({len(summary)} modules)")
    print(summary.to_string(index=False))

    # ---- per-species sign concordance (descriptive, NOT a gate) --------------
    # The Spearman rho stays the primary statistic. Sign concordance is reported
    # beside it because a reader will notice the quadrant structure anyway, and
    # an unreported version invites the wrong reading: most of the 879 motifs sit
    # near the origin, so `N(xy > 0) / 879` counts many motifs whose sign is
    # barely defined. The FDR-restricted column restricts the same count to the
    # motifs the model itself calls significant, which is the interpretable one.
    scores_path = C.results_dir(config, "01_model_motif") / "all_model_motif_scores.tsv"
    sc_rows = []
    if scores_path.exists():
        scores = pd.read_csv(scores_path, sep="\t",
                             usecols=["analysis", "species", "motif_id", "FDR"])
        scores = scores[scores.analysis == celltype]
        joined = longest.merge(scores[["species", "motif_id", "FDR"]],
                               on=["species", "motif_id"], how="left")
        for sp, block in joined.groupby("species", sort=False):
            xy = block.model_log2_or * block.chromvar_C
            usable = np.isfinite(xy) & (block.model_log2_or != 0) & (block.chromvar_C != 0)
            supported = usable & (block.FDR < float(config["model_analysis"]["fdr"]))
            sc_rows.append(dict(
                celltype=celltype, species=sp,
                n_motifs=int(usable.sum()),
                n_same_sign=int((xy[usable] > 0).sum()),
                sc=(float((xy[usable] > 0).mean()) if usable.any() else np.nan),
                n_motifs_model_fdr=int(supported.sum()),
                n_same_sign_model_fdr=int((xy[supported] > 0).sum()),
                sc_model_fdr=(float((xy[supported] > 0).mean())
                              if supported.any() else np.nan),
            ))
    else:
        C.log("WARNING: no s01 score table; sign concordance not computed")
    sc_frame = pd.DataFrame(sc_rows)
    if not sc_frame.empty:
        out_sc = in_dir / f"{stem}_sign_concordance.tsv"
        sc_frame.to_csv(out_sc, sep="\t", index=False)
        C.log("wrote", out_sc.name, f"({len(sc_frame)} species)")
        print(sc_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
