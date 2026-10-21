#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace every Astroglia family score back to its period, and flag H1 leakage.

WHY this exists
---------------
A summary sentence once reported "human NFI early = -0.35" as if it were the
cross-species prenatal value. It was not: -0.35 is human's early_embryonic window
(CS18-23), which this design keeps OUT of every cross-species statistic. Human's
cross-species prenatal is the 11wpc+15-17wpc model, and its NFI score is a
different number. Nothing was wrong with the pipeline -- the gate reads the
cross-window labels, so H1 could not enter it -- but the label was wrong in prose,
which is exactly the kind of error a reader cannot catch.

So the mapping is written out, one row per (species, period), with the two facts
that a summary must never conflate:

  * which stage a period maps to, and whether that stage is cross-species;
  * the family score that period actually carries, and under WHICH centering.

The centering column matters because a family score is only defined once the
periods that enter its baseline are fixed, and for human the two candidate scopes
give different numbers (and, for RFX and NF-Y, an opposite sign at prenatal). The
frozen pipeline's resolution is now explicit in the artefacts:

  * `family_trajectory.tsv` -- EVERY cross-species statistic -- centres over the
    species' cross-species windows only, so the single-species window cannot move
    a cross-species number. Its `centering_scope` column says so.
  * `family_trajectory_within_species.tsv` centres over all of the species'
    periods, because its axis contains all of them; each row carries
    `analysis_scope` (cross_species / species_only) and `included_cross_species`.

For marmoset and mouse the two scopes coincide -- all their periods are
cross-species windows -- so only human is affected. Both numbers are legitimate;
this table prints them side by side so that no summary ever presents one as the
other.

    python script/audit_astroglia_stage_trace.py --config config/analysis_v2.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

GATE_FAMILIES = ("RFX", "NFI", "NF-Y", "SOX")


def family_scores(frame: pd.DataFrame, periods: list[str], motif_ids: set[str]) -> pd.Series:
    """Method A: centre each motif across the given periods, then take the median."""
    sub = frame[frame.period.isin(periods) & frame.motif_id.isin(motif_ids)].copy()
    if sub.empty:
        return pd.Series(dtype=float)
    sub["centered"] = sub.groupby("motif_id").log2_OR.transform(lambda v: v - v.mean())
    return sub.groupby("period").centered.median()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(C.CONFIG_DIR / "analysis_v2.yaml"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = C.load_config(args.config)
    audit_dir = C.results_dir(config, "00_audit")

    scores = pd.read_csv(C.results_dir(config, "01_model_motif") / "all_model_motif_scores.tsv",
                         sep="\t")
    scores = scores[scores.analysis == "Astroglia"]
    stage = C.stage_table(config)
    cross = set(C.cross_windows(config))
    single = set(C.single_species_windows(config))
    family_table = C.tf_family_table()

    rows = []
    for species, periods in C.species_axis(config).items():
        block = scores[scores.species == species]
        if block.empty:
            continue
        all_scope = family_scores(block, periods, set())
        cross_periods = [p for p in periods
                         if stage.set_index("rank_period").cross_window.get(p) in cross]
        cross_scope = family_scores(block, cross_periods, set())
        for period in periods:
            window = stage.set_index("rank_period").cross_window.get(period, "")
            is_cross = window in cross
            for family in GATE_FAMILIES:
                members = set(family_table[(family_table.family == family)
                                           & (family_table.member_type == "monomer")].motif_id)
                every = family_scores(block, periods, members).get(period, np.nan)
                crosswise = family_scores(block, cross_periods, members).get(period, np.nan)
                rows.append(dict(
                    species=species, period=period, window=window,
                    analysis_scope="cross_species" if is_cross else "species_only",
                    included_cross_species=bool(is_cross),
                    is_single_species_window=bool(window in single),
                    family=family,
                    family_score_center_all_periods=float(every) if pd.notna(every) else np.nan,
                    family_score_center_cross_windows_only=(
                        float(crosswise) if pd.notna(crosswise) else np.nan),
                    scopes_agree=bool(
                        pd.isna(every) and pd.isna(crosswise)
                        or pd.notna(every) and pd.notna(crosswise)
                        and abs(every - crosswise) < 1e-9
                    ),
                ))
    trace = pd.DataFrame(rows)

    out = Path(args.out) if args.out else audit_dir / "astroglia_stage_gate_trace.tsv"
    trace.to_csv(out, sep="\t", index=False)
    C.log("wrote", C.display_path(out), f"({len(trace)} rows)")

    # ---- the two things a summary must never get wrong ---------------------
    leaks = trace[trace.is_single_species_window & trace.included_cross_species]
    if not leaks.empty:
        print("DEFECT: a single-species window is marked as cross-species:")
        print(leaks.to_string(index=False))
    else:
        print(f"OK  no single-species window is marked cross-species "
              f"({int(trace.is_single_species_window.sum() // len(GATE_FAMILIES))} "
              f"period(s) are species-only)")

    disagree = trace[~trace.scopes_agree & trace.included_cross_species]
    if not disagree.empty:
        print("\nNOTE: these cross-species cells depend on WHICH periods enter the centering")
        print("      baseline. This table shows both so the difference is visible; the frozen")
        print("      pipeline resolves it by using the cross-window-only scope for EVERY")
        print("      cross-species statistic (`family_trajectory.tsv`, the validation tables,")
        print("      Gate A4), and keeping the all-periods scope only in")
        print("      `family_trajectory_within_species.tsv`, where the species-only window is")
        print("      legitimate content. Affected:", sorted(set(disagree.species)))
        print(disagree[["species", "period", "window", "family",
                        "family_score_center_all_periods",
                        "family_score_center_cross_windows_only"]].round(4).to_string(index=False))
    else:
        print("OK  the two centering scopes agree on every cross-species cell")

    print("\n--- per-species NFI trace (the family the summary got wrong) ---")
    print(trace[trace.family == "NFI"][
        ["species", "period", "window", "included_cross_species",
         "family_score_center_all_periods", "family_score_center_cross_windows_only"]
    ].round(4).to_string(index=False))

    return 1 if not leaks.empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
