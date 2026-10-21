#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Derive the plotting inputs for the Astroglia model-vs-raw family trajectories.

WHY this step exists
--------------------
The family-trajectory figure used to put the model's centered log2 OR and the raw
chromVAR family score on ONE shared y-axis. That is the wrong comparison: the two
are different statistics on different peak universes and their magnitudes differ
by roughly 40x (human adult NF-Y: raw -7.20 against model -0.17), so every model
curve collapses onto y = 0 and the reader cannot see the trajectory that is being
validated at all.

The fix is to compare SHAPE: standardise each (species, family, source) triple
across the three windows before plotting. That standardisation, and the per-family
direction concordance, are DERIVED QUANTITIES -- so they are computed here, in a
prep step, and written to tables. The plotting scripts then only read them, which
keeps them free of any statistic (see V2_PLAN section 15-3 rule 1 and
script/tests/test_plot_no_recompute.py).

Inputs  (written by s05_astroglia_raw_validation.py):
    results_v2/05_astroglia_validation/family_trajectory_model_vs_raw.tsv
    results_v2/05_astroglia_validation/family_trajectory_correlation.tsv

Outputs (read by the three s05_plot_family_trajectory_*.py scripts):
    family_trajectory_model_vs_raw_long.tsv   tidy: species, stage, family,
                                              source, score, ...
    family_trajectory_agreement.tsv           species x family: rho + concordance
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

WINDOW_ORDER = ("prenatal", "postnatal", "adult")
SOURCES = (("model", "model_family_score", "n_members_model"),
           ("raw_chromVAR", "raw_family_score", "n_members_raw"))


def zscore_within(series: pd.Series) -> pd.Series:
    """z-score a 3-point trajectory; a flat trajectory becomes 0,0,0 and is counted."""
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return pd.Series(np.full(len(values), np.nan), index=series.index)
    sd = float(np.std(values, ddof=0))
    if sd == 0.0:
        return pd.Series(np.zeros(len(values)), index=series.index)
    return pd.Series((values - values.mean()) / sd, index=series.index)


def sign_of(value: float) -> int:
    if not np.isfinite(value):
        return 0
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(C.CONFIG_DIR / "analysis_v2.yaml"))
    parser.add_argument("--in-dir", default=None,
                        help="override the 05_astroglia_validation directory")
    args = parser.parse_args()

    config = C.load_config(args.config)
    out_dir = Path(args.in_dir) if args.in_dir else C.results_dir(config, "05_astroglia_validation")
    wide_path = out_dir / "family_trajectory_model_vs_raw.tsv"
    corr_path = out_dir / "family_trajectory_correlation.tsv"
    if not wide_path.exists():
        raise FileNotFoundError(
            f"{wide_path} missing; run s05_astroglia_raw_validation.py first"
        )

    wide = pd.read_csv(wide_path, sep="\t")
    required = {"species", "stage", "family", "analysis_level"}
    absent = sorted(required - set(wide.columns))
    if absent:
        raise ValueError(f"{wide_path} lacks columns {absent}")

    # ---- tidy long form, one row per (species, stage, family, source) ---------
    rows: list[dict] = []
    for source, score_column, member_column in SOURCES:
        if score_column not in wide.columns:
            raise ValueError(f"{wide_path} lacks {score_column} for source {source}")
        for row in wide.itertuples():
            rows.append(dict(
                species=row.species,
                stage=row.stage,
                family=row.family,
                analysis_level=row.analysis_level,
                source=source,
                score=float(getattr(row, score_column)),
                n_members=int(getattr(row, member_column)) if member_column in wide.columns else 0,
            ))
    long = pd.DataFrame(rows)
    long["stage"] = pd.Categorical(long.stage, WINDOW_ORDER, ordered=True)
    long = long.sort_values(["species", "family", "source", "stage"]).reset_index(drop=True)

    flat_groups: list[str] = []
    standardised = []
    for key, group in long.groupby(["species", "family", "source"], sort=False):
        if len(group) != len(WINDOW_ORDER):
            raise ValueError(
                f"{key} carries {len(group)} windows, expected {len(WINDOW_ORDER)}"
            )
        z = zscore_within(group.sort_values("stage").score)
        if np.allclose(z.to_numpy(), 0.0):
            flat_groups.append("|".join(map(str, key)))
        standardised.append(z)
    long["score_z"] = pd.concat(standardised).sort_index()
    long = long.sort_values(["species", "family", "source", "stage"]).reset_index(drop=True)

    if flat_groups:
        C.log("WARNING: flat trajectories (sd=0), z set to 0:", flat_groups)

    long_out = out_dir / "family_trajectory_model_vs_raw_long.tsv"
    long.to_csv(long_out, sep="\t", index=False)
    C.log("wrote", long_out.name, f"({len(long)} rows)")

    # ---- agreement: existing 3-point rho plus a per-family direction count ----
    agreement = None
    if corr_path.exists():
        corr = pd.read_csv(corr_path, sep="\t").rename(columns={"scope": "species"})
        keep = [c for c in ("species", "family", "analysis_level", "n_points",
                            "rho", "pvalue", "p_is_reliable") if c in corr.columns]
        agreement = corr[keep].copy()

    # direction concordance per (species, family): over the two consecutive
    # transitions, does the model move the same way as the raw profile?
    direction_rows = []
    for (species, family), group in long.groupby(["species", "family"], sort=False):
        per_source = {
            source: group[group.source == source].sort_values("stage").score.to_numpy(dtype=float)
            for source in ("model", "raw_chromVAR")
        }
        deltas = {source: np.diff(values) for source, values in per_source.items()}
        matched, usable, model_dir, raw_dir = 0, 0, [], []
        for index in range(len(WINDOW_ORDER) - 1):
            m, r = sign_of(deltas["model"][index]), sign_of(deltas["raw_chromVAR"][index])
            model_dir.append(m)
            raw_dir.append(r)
            if m == 0 or r == 0:
                continue  # a tie is not a disagreement
            usable += 1
            matched += int(m == r)
        direction_rows.append(dict(
            species=species,
            family=family,
            analysis_level=group.analysis_level.iloc[0],
            n_transitions=usable,
            n_matched=matched,
            concordance=(matched / usable) if usable else np.nan,
            model_directions="|".join(map(str, model_dir)),
            raw_directions="|".join(map(str, raw_dir)),
        ))
    direction = pd.DataFrame(direction_rows)

    if agreement is not None:
        agreement = agreement.merge(direction, on=["species", "family"], how="outer",
                                    suffixes=("", "_direction"))
        # the merge duplicates the level column; keep one, and restore the
        # integer dtypes the outer join promoted to float
        agreement = agreement.drop(columns=[c for c in agreement.columns
                                            if c == "analysis_level_direction"])
        for column in ("n_transitions", "n_matched"):
            if column in agreement.columns:
                agreement[column] = agreement[column].astype("Int64")
    else:
        agreement = direction
    agreement_out = out_dir / "family_trajectory_agreement.tsv"
    agreement.to_csv(agreement_out, sep="\t", index=False)
    C.log("wrote", agreement_out.name, f"({len(agreement)} rows)")

    print(long.groupby(["species", "family", "source"]).size().unstack(fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
