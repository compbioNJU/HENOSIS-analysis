#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score the frozen Oligo science-anchor panel, species by species.

WHY this exists
---------------
The primary Oligo result is a family/module statement. A reader of the source
paper's cerebellar GRN asks the narrower question -- what about OLIG2 itself, or
SOX10, or NKX2-2? -- and a module median cannot answer it, because a module
median is taken over its members and a member that matches no peak contributes a
zero. This step reads the frozen per-motif enrichment, the frozen chromVAR
contrast and the frozen matched-random draws for the matrices that
`script/s17b_audit_oligo_science_anchors.py` resolved from the panel declared in
`config["oligo_science_anchors"]`, and writes three tables plus a README.

    anchor_species_scores.tsv         one row per (species x anchor matrix)
    anchor_random500_tests.tsv        the two empirical tail probabilities
    anchor_cross_species_summary.tsv  one row per anchor matrix
    README.md                         panel, rules, unmeasurable matrices, bounds

WHAT IT READS, AND WHAT IT REFUSES TO DO
----------------------------------------
`model_log2_OR` and `model_FDR` are read from the arm's own results tree,
`result_v3/top<K>/01_model_motif/all_model_motif_scores.tsv`, for the model
`Oligo__<species>__all_development`; `chromvar_C` is read from
`result_v3/top<K>/09_oligo_validation/oligo_model_vs_chromvar.tsv`. Nothing upstream
is recomputed. The model log2 OR read from the enrichment table is cross-checked
against the `model_log2_or` column of the chromVAR table and any disagreement is
reported (stdout, the README and the run summary), never silently absorbed.

The ONLY new statistics formed here are the two empirical tail probabilities of
the matched-random test:

    p_upper     = (1 + #{null >= observed}) / (1 + n)
    p_lower     = (1 + #{null <= observed}) / (1 + n)
    p_two_sided = min(1, 2 * min(p_upper, p_lower))

and NOTHING ELSE. In particular no cross-species correlation between an anchor's
model log2 OR and its chromVAR C is computed, written or named in any table here:
that would be a third statistic, and with n = 6 species, no null and no p-value
nothing in the frozen contract would bound how it is read.

WHY THE LOWER TAIL IS NOT OPTIONAL
----------------------------------
The per-motif FDR in this project comes from a ONE-SIDED Fisher test for
ENRICHMENT. A log2 odds ratio below zero is therefore NOT evidence of depletion
and may never be reported as "significant depletion" on the strength of that
FDR. Hence the enrichment flag is named `model_fdr_positive_supported` -- so that
it cannot be read as a depletion flag -- and the only depletion statement this
analysis makes is `lower_tail_supported`, which needs the null draws.

HOW AN UNMEASURABLE MATRIX IS REPRESENTED
-----------------------------------------
A matrix that matches no peak at the frozen FIMO threshold has no measured
enrichment: the upstream table carries an exactly-zero log2 OR for it, which is
indistinguishable in print from a measured zero. Every such row is therefore
kept as a NAMED row whose scores are ABSENT (NaN), never zero, flagged
`measurable=FALSE`, and excluded from every per-anchor aggregate with the
exclusion visible as the row's own `n_species_available`/`n_species_excluded`;
in the random-draw table such a matrix is `n_null_usable=0`, which is this
table's own way of saying that no test was evaluable (its 200 draws are the
degenerate all-zero statistic of a matrix that matches nothing, not draws of an
enrichment). Nothing is dropped and nothing is substituted.

BOUNDARY
--------
This is a SUPPLEMENTARY, post-primary analysis for the Oligo arm only. It is not
a gate: nothing here may be wired into any verdict rule, and running it must
leave the two claim artefacts of this project byte-identical -- which it does,
because it neither reads nor writes anything under the claim output directory of
`results_v2`. The guard is
`script/tests/test_oligo_anchor_does_not_change_claims.py`.

    python script/s17c_oligo_science_anchor_analysis.py \\
        --config config/analysis_v2.yaml [--outdir DIR] [--null-repeats N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

STEP_DIR = "09_oligo_validation"
SUBDIR = "science_anchors"
MANIFEST_FILE = "anchor_manifest.tsv"
SCORES_FILE = "anchor_species_scores.tsv"
TESTS_FILE = "anchor_random500_tests.tsv"
SUMMARY_FILE = "anchor_cross_species_summary.tsv"
README_FILE = "README.md"

MODEL_STEM = "Oligo__{species}__all_development"

# Flags are written as the same TRUE / FALSE tokens the manifest uses, so that a
# reader of the four tables has one spelling for one concept. A sign that has no
# verdict (a tie) is written as an EMPTY cell, which is neither of the two.
TRUE, FALSE = "TRUE", "FALSE"

# Frozen column orders. Downstream reads these by name (the s17d figure resolves
# each role through an alias list), so the order is part of the contract.
SCORE_COLUMNS = [
    "species", "requested_name", "group", "motif_id", "motif_name", "measurable",
    "model_log2_OR", "model_FDR", "model_fdr_positive_supported",
    "chromvar_C", "model_sign", "chromvar_sign", "sign_concordant",
]
TEST_COLUMNS = [
    "species", "requested_name", "motif_id", "n_null_usable", "observed_log2_OR",
    "null_median", "null_q025", "null_q975", "p_upper", "p_lower", "p_two_sided",
    "lower_tail_supported", "upper_tail_supported",
]
SUMMARY_COLUMNS = [
    "requested_name", "group", "motif_id", "motif_name", "measurable",
    "n_species_available", "n_model_positive", "n_model_negative",
    "n_raw_positive", "n_raw_negative", "n_sign_concordant",
    "sign_concordance_fraction", "median_model_log2OR", "median_chromvar_C",
    "n_model_fdr_positive_supported", "n_lower_tail_supported",
    "n_upper_tail_supported",
    # trailing, explicit: the count of species excluded because the matrix has no
    # measured value there, so the exclusion is a number and not something a
    # reader has to infer from a denominator
    "n_species_excluded",
]
# No cross-species correlation column exists here, on purpose. A Spearman rho
# across the six species between an anchor's model log2 OR and its chromVAR C
# would be a THIRD quantity formed by this step, and the frozen contract permits
# exactly two new statistics (the empirical tails below). With n = 6, with no
# null and with no p-value such a rho is not bounded by anything the contract
# froze, and a reader could quote it as evidence that model and raw "agree across
# species" for a named regulator. It is therefore not formed, not written and not
# named in any table of this directory.

MANIFEST_COLUMNS = [
    "requested_name", "group", "motif_id", "catalog_name", "present",
    "n_catalog_ids", "measurable", "n_foreground_hits_max", "status",
]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def is_true(value: object) -> bool:
    """TRUE / TRUE-as-string / 1 / yes -- the spellings the tables may carry."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(value) and value != 0)
    return str(value).strip().upper() in {"TRUE", "T", "1", "1.0", "YES", "Y"}


def sign_of(value: float) -> int:
    """+1 / -1 / 0, where 0 means "no sign": non-finite, or exactly zero.

    An exactly-zero score is a tie, not a small negative: it carries no
    direction, so it may not be folded into either sign count.
    """
    if not np.isfinite(value):
        return 0
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def nanmedian(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else float("nan")


def resolve_outdir(config: dict, outdir: str | None) -> Path:
    """Where the tables go. `--outdir` redirects them without touching results."""
    if outdir:
        path = Path(outdir)
        if not path.is_absolute():
            path = Path.cwd() / path
    else:
        path = C.config_path(config, "results") / STEP_DIR / SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def manifest_path(config: dict, out_dir: Path, override: str | None) -> Path:
    if override:
        path = Path(override)
        return path if path.is_absolute() else Path.cwd() / path
    del config
    return out_dir / MANIFEST_FILE


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def load_manifest(path: Path) -> pd.DataFrame:
    """The frozen resolution, written by the s17b audit.

    The panel is NOT re-resolved here. `anchor_manifest.tsv` is the authority on
    which requested name became which catalog matrix and on which matrix is
    measurable; re-deriving it in a second place would let the two disagree, and
    a disagreement about which matrix belongs to a name is exactly how a matrix
    gets picked on the outcome.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. The frozen panel resolution is written by "
            "script/s17b_audit_oligo_science_anchors.py; run it first (its "
            "--outdir must be the directory this step reads)."
        )
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = [name for name in MANIFEST_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} lacks columns {missing}; the panel cannot be read")
    if frame.empty:
        raise ValueError(f"{path} carries no row, so no anchor can be scored")
    return frame


def read_model_scores(config: dict, models: list[str],
                      motif_ids: list[str]) -> pd.DataFrame:
    """The frozen per-motif enrichment for the six Oligo models."""
    path = C.config_path(config, "results") / "01_model_motif" / "all_model_motif_scores.tsv"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; the anchor scores are read from it")
    frame = pd.read_csv(
        path, sep="\t",
        usecols=["model", "species", "motif_id", "motif_name", "fg_hits", "bg_hits",
                 "log2_OR", "FDR"],
    )
    frame = frame[frame.model.isin(models) & frame.motif_id.isin(motif_ids)]
    if frame.duplicated(["model", "motif_id"]).any():
        raise ValueError(f"{path} carries more than one row for a (model, motif) pair")
    return frame.set_index(["model", "motif_id"], drop=False)


def read_chromvar(config: dict, motif_ids: list[str]) -> pd.DataFrame:
    """The frozen per-motif chromVAR contrast, for the same matrix universe."""
    path = C.config_path(config, "results") / STEP_DIR / "oligo_model_vs_chromvar.tsv"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; the anchor raw values are read from it")
    frame = pd.read_csv(
        path, sep="\t",
        usecols=["celltype", "species", "motif_id", "motif_name", "model_log2_or",
                 "chromvar_C"],
    )
    celltypes = sorted(set(frame.celltype.astype(str)))
    if celltypes != ["Oligo"]:
        raise ValueError(f"{path} carries celltypes {celltypes}, expected only Oligo")
    frame = frame[frame.motif_id.isin(motif_ids)]
    if frame.duplicated(["species", "motif_id"]).any():
        raise ValueError(f"{path} carries more than one row for a (species, motif) pair")
    return frame.set_index(["species", "motif_id"], drop=False)


def read_null_draws(path: Path, models: list[str], motif_ids: list[str],
                    null_repeats: int) -> dict[tuple[str, str], list[float]]:
    """{ (model, motif_id): [draw, ...] } for repeats with index < null_repeats.

    Streamed in chunks: the file holds every model of the project, and only the
    six Oligo models and the anchor matrices are of interest. A row whose
    `log2_OR` is not a number is kept as NaN rather than dropped, so the count of
    usable draws is a fact about the file and not about this reader.
    """
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; the matched-random test needs its draws")
    wanted_models = set(models)
    wanted_motifs = set(motif_ids)
    draws: dict[tuple[str, str], list[float]] = {}
    reader = pd.read_csv(
        path, sep="\t", usecols=["model", "repeat", "motif_id", "log2_OR"],
        chunksize=250_000, dtype={"model": str, "motif_id": str},
    )
    for chunk in reader:
        keep = chunk[
            chunk.model.isin(wanted_models)
            & chunk.motif_id.isin(wanted_motifs)
            & (chunk.repeat.astype("int64") < int(null_repeats))
        ]
        if keep.empty:
            continue
        values = pd.to_numeric(keep.log2_OR, errors="coerce").to_numpy(dtype=float)
        for key, value in zip(zip(keep.model, keep.motif_id), values):
            draws.setdefault((str(key[0]), str(key[1])), []).append(float(value))
    return draws


# --------------------------------------------------------------------------
# the statistics
# --------------------------------------------------------------------------


def tail_tests(observed: float, draws: list[float], fdr: float) -> dict[str, object]:
    """The two empirical tail probabilities, and nothing else.

    The count of usable draws is the number of repeats that carry a finite value.
    When that count is zero -- or when there is no observed value to compare
    against -- every probability is NaN and both support flags are FALSE: a
    probability formed from zero draws would be a number with no data behind it.
    """
    usable = [v for v in draws if np.isfinite(v)]
    n = len(usable)
    out: dict[str, object] = dict(n_null_usable=n, p_upper=np.nan, p_lower=np.nan,
                                 p_two_sided=np.nan, null_median=np.nan,
                                 null_q025=np.nan, null_q975=np.nan,
                                 lower_tail_supported=FALSE,
                                 upper_tail_supported=FALSE)
    if n == 0 or not np.isfinite(observed):
        return out
    array = np.asarray(usable, dtype=float)
    p_upper = (1.0 + float((array >= observed).sum())) / (1.0 + n)
    p_lower = (1.0 + float((array <= observed).sum())) / (1.0 + n)
    out.update(
        p_upper=p_upper,
        p_lower=p_lower,
        p_two_sided=min(1.0, 2.0 * min(p_upper, p_lower)),
        null_median=float(np.median(array)),
        null_q025=float(np.quantile(array, 0.025)),
        null_q975=float(np.quantile(array, 0.975)),
        lower_tail_supported=TRUE if p_lower < fdr else FALSE,
        upper_tail_supported=TRUE if p_upper < fdr else FALSE,
    )
    return out


# There is deliberately NO cross-species correlation function here. See the note
# under SUMMARY_COLUMNS: this step forms the two empirical tails and nothing else.


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def anchor_matrices(manifest: pd.DataFrame) -> list[dict[str, object]]:
    """Every present requested name x catalog matrix, in the manifest's own order."""
    out: list[dict[str, object]] = []
    for row in manifest.itertuples():
        if not is_true(row.present):
            continue
        motif_id = str(row.motif_id).strip()
        if not motif_id:
            raise ValueError(
                f"manifest row {row.requested_name!r} says present but carries no "
                "motif_id; a present anchor without a matrix cannot be scored"
            )
        out.append(
            dict(
                requested_name=str(row.requested_name),
                group=str(row.group),
                motif_id=motif_id,
                motif_name=str(row.catalog_name),
                measurable=is_true(row.measurable),
                n_catalog_ids=int(float(row.n_catalog_ids or 0)),
            )
        )
    if not out:
        raise ValueError("the manifest names no matrix with present=TRUE")
    seen = [str(entry["motif_id"]) for entry in out]
    if len(set(seen)) != len(seen):
        raise ValueError(
            "the manifest names the same motif_id under more than one requested "
            f"name ({sorted(seen)}); a per-matrix aggregate would then merge two "
            "different anchors, so this is refused rather than guessed"
        )
    return out


def build(config: dict, manifest: pd.DataFrame, fdr: float, null_repeats: int,
          report: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    species_axis = list(C.SPECIES_ORDER)
    models = [MODEL_STEM.format(species=species) for species in species_axis]
    anchors = anchor_matrices(manifest)
    motif_ids = sorted({str(a["motif_id"]) for a in anchors})

    scores_in = read_model_scores(config, models, motif_ids)
    chromvar_in = read_chromvar(config, motif_ids)
    draws = read_null_draws(
        C.config_path(config, "results") / "01_model_motif" / "random_null_scores.tsv.gz",
        models, motif_ids, null_repeats,
    )
    report["null_repeats_requested"] = int(null_repeats)
    report["draws_found"] = int(sum(len(v) for v in draws.values()))

    # ---- cross-check: the same log2 OR read from two frozen tables ----------
    disagreements: list[str] = []
    compared = 0
    largest_gap = 0.0
    for anchor in anchors:
        for species in species_axis:
            raw_key = (species, str(anchor["motif_id"]))
            model_key = (MODEL_STEM.format(species=species), str(anchor["motif_id"]))
            if raw_key not in chromvar_in.index or model_key not in scores_in.index:
                continue
            left = float(scores_in.loc[model_key].log2_OR)
            right = float(chromvar_in.loc[raw_key].model_log2_or)
            compared += 1
            if not (np.isfinite(left) and np.isfinite(right)) or left != right:
                gap = abs(left - right) if np.isfinite(left) and np.isfinite(right) else np.inf
                largest_gap = max(largest_gap, gap if np.isfinite(gap) else float("inf"))
                if len(disagreements) < 10:
                    disagreements.append(
                        f"{anchor['requested_name']}/{anchor['motif_id']}/{species}: "
                        f"enrichment={left!r} chromvar={right!r}"
                    )
    report["cross_check_rows"] = compared
    report["cross_check_disagreements"] = disagreements
    report["cross_check_max_abs_gap"] = largest_gap

    # ---- per (species x anchor matrix) -------------------------------------
    score_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    per_matrix: dict[str, dict[str, list[float]]] = {}
    missing_cells: list[str] = []
    species_level_unmeasurable: list[str] = []

    for anchor in anchors:
        motif_id = str(anchor["motif_id"])
        collected = per_matrix.setdefault(
            motif_id,
            dict(model=[], raw=[], model_sign=[], raw_sign=[], concordant=[],
                 model_fdr_supported=[], lower=[], upper=[]),
        )
        for species in species_axis:
            model_key = (MODEL_STEM.format(species=species), motif_id)
            raw_key = (species, motif_id)
            has_model = model_key in scores_in.index
            has_raw = raw_key in chromvar_in.index
            if not has_model and not has_raw:
                missing_cells.append(f"{anchor['requested_name']}/{motif_id}/{species}")

            model_value = float(scores_in.loc[model_key].log2_OR) if has_model else float("nan")
            model_fdr = float(scores_in.loc[model_key].FDR) if has_model else float("nan")
            raw_value = float(chromvar_in.loc[raw_key].chromvar_C) if has_raw else float("nan")
            hits = float(scores_in.loc[model_key].fg_hits) + float(
                scores_in.loc[model_key].bg_hits) if has_model else float("nan")

            # A matrix is measurable per the manifest. A species whose two peak
            # sets carry no hit at all cannot yield a value either, and its
            # upstream log2 OR is an exactly-zero placeholder: it is absent here
            # and it is named, not quietly zeroed.
            measurable = bool(anchor["measurable"])
            if measurable and np.isfinite(hits) and hits == 0:
                measurable = False
                species_level_unmeasurable.append(
                    f"{anchor['requested_name']}/{motif_id}/{species}"
                )
            if not measurable:
                model_value = float("nan")
                model_fdr = float("nan")
                raw_value = float("nan")

            model_sign = sign_of(model_value)
            raw_sign = sign_of(raw_value)
            if model_sign == 0 or raw_sign == 0:
                concordant = ""            # a tie has no verdict either way
            else:
                concordant = TRUE if model_sign == raw_sign else FALSE
            positive_supported = (
                TRUE if (np.isfinite(model_fdr) and np.isfinite(model_value)
                         and model_fdr < fdr and model_value > 0) else FALSE
            )

            score_rows.append(
                dict(
                    species=species,
                    requested_name=str(anchor["requested_name"]),
                    group=str(anchor["group"]),
                    motif_id=motif_id,
                    motif_name=str(anchor["motif_name"]),
                    measurable=TRUE if measurable else FALSE,
                    model_log2_OR=model_value,
                    model_FDR=model_fdr,
                    model_fdr_positive_supported=positive_supported,
                    chromvar_C=raw_value,
                    model_sign=model_sign,
                    chromvar_sign=raw_sign,
                    sign_concordant=concordant,
                )
            )

            # An unmeasurable matrix is excluded from the random-draw test with
            # the exclusion counted: n_null_usable=0 is this table's way of
            # saying "not evaluable", because the 200 upstream draws of a matrix
            # that matches nothing are the degenerate all-zero statistic, not
            # draws of an enrichment.
            usable_draws = [] if not measurable else draws.get(model_key, [])
            test = tail_tests(model_value, usable_draws, fdr)
            observed = model_value if measurable else float("nan")
            test_rows.append(
                dict(
                    species=species,
                    requested_name=str(anchor["requested_name"]),
                    motif_id=motif_id,
                    n_null_usable=int(test["n_null_usable"]),
                    observed_log2_OR=observed,
                    null_median=float(test["null_median"]),
                    null_q025=float(test["null_q025"]),
                    null_q975=float(test["null_q975"]),
                    p_upper=float(test["p_upper"]),
                    p_lower=float(test["p_lower"]),
                    p_two_sided=float(test["p_two_sided"]),
                    lower_tail_supported=str(test["lower_tail_supported"]),
                    upper_tail_supported=str(test["upper_tail_supported"]),
                )
            )

            collected["model"].append(model_value)
            collected["raw"].append(raw_value)
            collected["model_sign"].append(model_sign)
            collected["raw_sign"].append(raw_sign)
            collected["concordant"].append(concordant == TRUE)
            collected["model_fdr_supported"].append(positive_supported == TRUE)
            collected["lower"].append(test["lower_tail_supported"] == TRUE)
            collected["upper"].append(test["upper_tail_supported"] == TRUE)

    report["missing_cells"] = missing_cells
    report["species_level_unmeasurable"] = species_level_unmeasurable

    # ---- one row per anchor matrix -----------------------------------------
    summary_rows: list[dict[str, object]] = []
    for anchor in anchors:
        motif_id = str(anchor["motif_id"])
        block = per_matrix[motif_id]
        finite_model = [v for v in block["model"] if np.isfinite(v)]
        finite_raw = [v for v in block["raw"] if np.isfinite(v)]
        available = sum(
            1 for m, r in zip(block["model"], block["raw"])
            if np.isfinite(m) and np.isfinite(r)
        )
        excluded = sum(
            1 for m, r in zip(block["model"], block["raw"])
            if not (np.isfinite(m) and np.isfinite(r))
        )
        concordant = int(sum(block["concordant"]))
        summary_rows.append(
            dict(
                requested_name=str(anchor["requested_name"]),
                group=str(anchor["group"]),
                motif_id=motif_id,
                motif_name=str(anchor["motif_name"]),
                measurable=TRUE if anchor["measurable"] else FALSE,
                n_species_available=available,
                n_model_positive=int(sum(1 for s in block["model_sign"] if s > 0)),
                n_model_negative=int(sum(1 for s in block["model_sign"] if s < 0)),
                n_raw_positive=int(sum(1 for s in block["raw_sign"] if s > 0)),
                n_raw_negative=int(sum(1 for s in block["raw_sign"] if s < 0)),
                n_sign_concordant=concordant,
                # the denominator is explicit: every species that carries both a
                # model and a chromVAR value. A species whose value is exactly
                # zero has no sign, so it can never be concordant and it stays in
                # the denominator -- a tie is not agreement.
                sign_concordance_fraction=(concordant / available) if available else float("nan"),
                median_model_log2OR=nanmedian(finite_model),
                median_chromvar_C=nanmedian(finite_raw),
                n_model_fdr_positive_supported=int(sum(block["model_fdr_supported"])),
                n_lower_tail_supported=int(sum(block["lower"])),
                n_upper_tail_supported=int(sum(block["upper"])),
                n_species_excluded=excluded,
            )
        )

    scores = pd.DataFrame(score_rows, columns=SCORE_COLUMNS)
    tests = pd.DataFrame(test_rows, columns=TEST_COLUMNS)
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    return scores, tests, summary


# --------------------------------------------------------------------------
# README
# --------------------------------------------------------------------------


def readme_text(config: dict, manifest: pd.DataFrame, scores: pd.DataFrame,
                tests: pd.DataFrame, summary: pd.DataFrame, report: dict,
                out_dir: Path) -> str:
    fdr = float(config["oligo_science_anchors"].get("fdr", 0.05))
    n_rows = int(config["oligo_science_anchors"].get("null_repeats", 200))
    # The source tree and the cut-off are the ARM's own, read from its config: this
    # README is written once per arm, and a hardcoded `results_v2` tree or a
    # hardcoded "Top500" would name a tree and a cut-off its numbers did not come
    # from (the Top1000/Top2000 arms read result_v3/top1000 and result_v3/top2000).
    cutoff = C.topk_label(config)
    results_tree = str(config["paths"]["results"])

    panel = manifest[[
        "group", "requested_name", "motif_id", "catalog_name", "present",
        "n_catalog_ids", "measurable", "status",
    ]]
    unmeasurable = manifest[
        (panel.present.str.upper() == "TRUE") & (panel.measurable.str.upper() == "FALSE")
    ]
    absent = manifest[panel.present.str.upper() == "FALSE"]
    # a row that matched nothing carries an empty count, not a 0: nothing was counted
    n_catalog = pd.to_numeric(panel.n_catalog_ids, errors="coerce").fillna(0)
    duplicated = panel[n_catalog > 1]

    def table(frame: pd.DataFrame, header: str) -> list[str]:
        lines = [f"| {' | '.join(header)} |",
                 f"|{'|'.join('---' for _ in header)}|"]
        for row in frame[header].itertuples(index=False):
            lines.append(f"| {' | '.join(str(value) for value in row)} |")
        return lines

    lines: list[str] = []
    add = lines.append

    add("# Oligo science anchors: the named regulators, species by species")
    add("")
    add("Supplementary, post-primary analysis for the **Oligo arm only**. It reads the")
    add("frozen enrichment, the frozen chromVAR contrast and the frozen matched-random")
    add("draws, and writes the four files in this directory. It forms no gate, it was")
    add("frozen before any of its numbers were looked at, and it changes no claim.")
    add("")
    add("The two claim artefacts of this project -- `FINAL_CLAIM_TABLE.tsv` and")
    add("`CLAIM_EVIDENCE.json`, in the project's claim output directory -- are")
    add("neither read nor written here, and stay byte-identical across a run. Nothing in")
    add("this directory may be wired into any verdict rule.")
    add("")
    add("## The question")
    add("")
    add("The primary Oligo result is a module statement: the SOX module is recurrent, the")
    add("bHLH/E-box module is consistently negative in every species, NKX and POU3 are")
    add("secondary. A module is a median over its members, so it cannot answer the")
    add("reader's next question -- do the individual regulators the source paper's")
    add("cerebellar GRN names behave the way the module does? Is OLIG2 itself negative,")
    add("or is it diluted by the other E-box motifs? Are NKX2-2 and POU3F3 better")
    add("conserved than their families? That is the whole of this analysis.")
    add("")
    add("## The frozen panel, and how it was chosen")
    add("")
    add("`oligo_science_anchors` in `config/analysis_v2.yaml` names the regulators as the")
    add("source paper names them for the oligodendrocyte lineage -- nothing was added")
    add("because its result was interesting, and nothing was dropped because it was not:")
    add("")
    panel_names: list[tuple[str, str]] = []
    for group, value in config["oligo_science_anchors"].items():
        if isinstance(value, (list, tuple)):
            for name in value:
                panel_names.append((str(group), str(name)))
    add("| group | requested name |")
    add("|---|---|")
    for group, name in panel_names:
        add(f"| {group} | {name} |")
    add("")
    add(f"Two scalars sit in the same config block and are used here as they are frozen:")
    add(f"`fdr = {fdr}` and `null_repeats = {n_rows}`.")
    add("")
    add("## The four resolution rules (frozen before results, not negotiable)")
    add("")
    add("1. **Case-insensitive EXACT match** on the catalog motif name. Case-insensitive")
    add("   matters -- the catalog answers `OLIG2` twice, `MA0678.1 OLIG2` and")
    add("   `MA1997.2 Olig2`. Exact matters -- a prefix would attach `SOX1` to nine")
    add("   matrices and a substring would attach `Pou5f1::Sox2` to `SOX2`.")
    add("2. **Every match is kept.** A requested name matching more than one catalog")
    add("   matrix contributes every match, one row each, all flagged")
    add("   `n_catalog_ids > 1`. Keeping the measurable one and dropping the other would")
    add("   be selecting a matrix on the outcome.")
    add("3. **No substitution.** A requested name matching nothing is `present=FALSE` and")
    add("   is never replaced by a similar motif.")
    add("4. **No peak means no score.** A matrix that matches no peak at the frozen FIMO")
    add("   per-site threshold of 1e-4 is `measurable=FALSE`; its scores are ABSENT")
    add("   (NaN), never zero, and it is excluded from every per-anchor aggregate with")
    add("   the exclusion counted.")
    add("")
    add("The resolution itself is written by `script/s17b_audit_oligo_science_anchors.py`")
    add("into `anchor_manifest.tsv`, which is the authority this analysis reads:")
    add("re-deriving it in a second place would let two resolutions disagree.")
    add("")
    add("## The frozen panel as resolved")
    add("")
    lines.extend(table(panel, ["group", "requested_name", "motif_id", "catalog_name",
                               "present", "n_catalog_ids", "measurable", "status"]))
    add("")
    if len(absent):
        add("A requested name that matched nothing is a named row above and is NOT")
        add("substituted; it therefore contributes no matrix row to any table.")
        add("")
    else:
        add("Every requested name matched the catalog, so there is no absent row.")
        add("")
    requested = [name for _, name in panel_names]
    if len(duplicated):
        names = sorted(set(duplicated.requested_name))
        n_present = int((panel.present.str.upper() == "TRUE").sum())
        add(f"The ambiguous requested name(s) -- {', '.join(names)} -- matched the catalog")
        add("more than once, and every match is carried: that is why the panel's")
        add(f"{len(requested)} requested names become {n_present} matrices")
        add(f"({len(requested)} names, {len(panel.group.unique())} groups).")
    add("")
    add("## The matrices that cannot be evaluated")
    add("")
    if len(unmeasurable):
        lines.extend(table(unmeasurable, ["requested_name", "motif_id", "catalog_name",
                                          "n_foreground_hits_max", "status"]))
        add("")
        add("`n_foreground_hits_max` is the largest `fg_hits` that matrix reaches in ANY")
        add("model of the frozen enrichment table, so 0 means the matrix reaches no peak at")
        add("the frozen threshold anywhere in the project, not merely in one model. In each")
        add("case the upstream table carries an exactly-zero log2 odds ratio, which prints")
        add("like a measured zero; here each is a NaN score on a named row flagged")
        add("`measurable=FALSE`, and its random-draw row is `n_null_usable=0` because its")
        add("200 draws are the degenerate all-zero statistic of a matrix that matches")
        add("nothing, not draws of an enrichment. None of the three enters any aggregate.")
        add("")
        for row in unmeasurable.itertuples():
            detail = {
                "MA0442.3": "its PWM is only six base pairs wide, which is why it never "
                            "reaches the per-site threshold;",
                "MA1997.2": "it is the SECOND matrix the name `OLIG2` resolves to, so the "
                            "ambiguity and the absent score compound;",
                "MA0675.2": "it is the only NKX6 matrix the panel names, so the NKX family "
                            "has no measurable NKX6 member here;",
            }.get(str(row.motif_id), "it simply reaches no peak;")
            add(f"* **{row.requested_name} / {row.motif_id} ({row.catalog_name})** -- "
                f"{detail}")
            add(f"  its scores in `{SCORES_FILE}` are empty and in `{TESTS_FILE}` its")
            add("  `observed_log2_OR` is empty with `n_null_usable = 0`.")
        add("")
    else:
        add("No matrix in this panel is unmeasurable on the frozen data.")
        add("")
    add("### What that costs the two questions")
    add("")
    add("* **Is OLIG2 itself negative?** The catalog answers `OLIG2` with two matrices")
    add("  and only one of them is measurable, so the OLIG2 answer rests on that single")
    add("  matrix; no agreement between the two Olig2 PWMs can be reported, and the")
    add("  bHLH/E-box module's direction cannot be read as an Olig2 result at all. A")
    add("  module median is taken over its listed members, and one of those members")
    add("  matches no peak, so the module statement and the OLIG2 statement are of")
    add("  different things. This is the reason the named-motif reading exists.")
    add("* **Are NKX2-2 and POU3F3/POU3F4 better conserved than their families?** The")
    add("  POU3 half is answerable as asked. The NKX half is not: NKX6-2 reaches no")
    add("  peak, so the NKX answer is an NKX2-2 answer and must not be reported as an")
    add("  NKX-family result.")
    add("* **Is the SOX signal a SOX10 statement?** It is not, and cannot be: SOX10's")
    add("  PWM is six base pairs wide and never reaches the threshold, so SOX10 has no")
    add("  measured value in this framework. The SOX answer is a SOX6/SOX8 answer.")
    add("")
    add("## The statistics, and the one rule that must not be broken")
    add("")
    add("`model_log2_OR` and `model_FDR` are read from")
    add(f"`{results_tree}/01_model_motif/all_model_motif_scores.tsv` (model")
    add(f"`Oligo__<species>__all_development`, {cutoff} versus the GC/length-matched")
    add(f"non-{cutoff} background); `chromvar_C` is read from")
    add(f"`{results_tree}/{STEP_DIR}/oligo_model_vs_chromvar.tsv`. Nothing upstream is")
    add("recomputed. The model log2 OR read from the enrichment table is cross-checked")
    add("against the `model_log2_or` column of the chromVAR table: "
        f"{report['cross_check_rows']} row(s) compared, "
        f"{len(report['cross_check_disagreements'])} disagreement(s)"
        + (f", e.g. {report['cross_check_disagreements'][0]}"
           if report["cross_check_disagreements"] else "") + ".")
    add("")
    add(f"**The per-motif FDR is a one-sided Fisher test for ENRICHMENT.** A log2 odds")
    add("ratio below zero is therefore not evidence of depletion, and this analysis never")
    add("reports it as such. That is why the enrichment flag is named")
    add("`model_fdr_positive_supported` -- `FDR < "
        f"{fdr}` **and** `log2_OR > 0` -- so that it cannot be misread as a depletion")
    add("flag, and why the only depletion statement in these tables is")
    add("`lower_tail_supported`, which needs the matched-random draws:")
    add("")
    add("```")
    add("p_upper     = (1 + #{null >= observed}) / (1 + n)")
    add("p_lower     = (1 + #{null <= observed}) / (1 + n)")
    add("p_two_sided = min(1, 2 * min(p_upper, p_lower))")
    add(f"lower_tail_supported = p_lower < {fdr}      upper_tail_supported = p_upper < {fdr}")
    add("```")
    add("")
    add("The null draws are the repeats of `random_null_scores.tsv.gz` filtered to one")
    add(f"model and one matrix; `n_null_usable` is how many of the {n_rows} repeats carry")
    add("a finite value. An unmeasurable matrix is the one case where that count is 0.")
    add("")
    add("## The tables")
    add("")
    add(f"* `{SCORES_FILE}` -- one row per (species x anchor matrix): "
        f"{len(scores)} rows.")
    add("  `model_sign` and `chromvar_sign` are -1 / 0 / +1, where 0 means no sign")
    add("  (non-finite or exactly zero). `sign_concordant` is TRUE only when both signs")
    add("  are non-zero and equal; a tie is neither agreement nor disagreement, so it is")
    add("  an EMPTY cell, not FALSE.")
    add(f"* `{TESTS_FILE}` -- one row per (species x anchor matrix): {len(tests)} rows.")
    add("  `observed_log2_OR` is the frozen model log2 OR; the p-values are the two")
    add("  empirical tails above. `lower_tail_supported` and `upper_tail_supported` are")
    add("  the only enrichment/depletion statements made anywhere in this analysis.")
    add(f"* `{SUMMARY_FILE}` -- one row per anchor matrix: {len(summary)} rows. Every")
    add("  count is over species that actually carry a value and `n_species_available`")
    add("  states the denominator; `n_species_excluded` counts the species dropped for")
    add("  having no measured value, so the exclusion is a number. A species whose value")
    add("  is exactly zero has no sign, so it can never be concordant and it stays in the")
    add(f"  `sign_concordance_fraction` denominator -- n_sign_concordant / n_species_available.")
    add("")
    add("## What is deliberately NOT computed: no cross-species correlation")
    add("")
    add("The only new statistics this analysis forms are the two empirical tails above.")
    add("It does NOT form the Spearman correlation across the six species between an")
    add("anchor's model log2 odds ratio and its chromVAR C, and no column of any table in")
    add("this directory carries one. Such a rho would be a THIRD quantity formed by this")
    add("step, outside the two the contract freezes; with n = 6 it carries no null and no")
    add("p-value, so nothing bounds how a reader could quote it -- e.g. as evidence that")
    add("model and chromVAR 'agree across species' for OLIG2 or for a SOX -- and six")
    add("points can describe the shape of an agreement but cannot establish one. It is")
    add("not computed, not written, and it may not appear in any figure legend as")
    add("evidence.")
    add("")
    add("What the summary does carry beyond the two tails are descriptive summaries of")
    add("the values read from the frozen tables -- `median_model_log2OR`,")
    add("`median_chromvar_C`, `sign_concordance_fraction` and the `n_*` counts. They are")
    add("reported as descriptions of what was read and are never a test of anything.")
    add("")
    add("## Reproducing it")
    add("")
    add("```")
    add("python script/s17b_audit_oligo_science_anchors.py --config config/analysis_v2.yaml")
    add("python script/s17c_oligo_science_anchor_analysis.py --config config/analysis_v2.yaml")
    add("```")
    add("")
    add(f"The tables above were written to `{out_dir.name}/`; `--outdir` redirects them,")
    add(f"and `--null-repeats N` uses only the first N repeats of the matched-random set")
    add("(it does not change the frozen default).")
    add("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=str(C.CONFIG_DIR / "analysis_v2.yaml"))
    parser.add_argument(
        "--outdir", default=None,
        help=f"directory to write the anchor tables into; defaults to "
             f"<results>/{STEP_DIR}/{SUBDIR}",
    )
    parser.add_argument(
        "--manifest", default=None,
        help=f"the frozen panel resolution to read; default: <outdir>/{MANIFEST_FILE}",
    )
    parser.add_argument(
        "--null-repeats", type=int, default=None,
        help="use only repeats with index < N from the matched-random set; "
             "default: the frozen `oligo_science_anchors.null_repeats`",
    )
    args = parser.parse_args(argv)

    C.main_guard()
    config = C.load_config(args.config)
    block = config.get("oligo_science_anchors")
    if not isinstance(block, dict) or not block:
        raise ValueError("config declares no `oligo_science_anchors` block")

    fdr = float(block.get("fdr", 0.05))
    null_repeats = (int(args.null_repeats) if args.null_repeats is not None
                    else int(block.get("null_repeats", 200)))

    out_dir = resolve_outdir(config, args.outdir)
    manifest = load_manifest(manifest_path(config, out_dir, args.manifest))

    report: dict[str, object] = {}
    scores, tests, summary = build(config, manifest, fdr, null_repeats, report)

    C.save_table(scores, out_dir / SCORES_FILE, sep="\t")
    C.save_table(tests, out_dir / TESTS_FILE, sep="\t")
    C.save_table(summary, out_dir / SUMMARY_FILE, sep="\t")
    (out_dir / README_FILE).write_text(
        readme_text(config, manifest, scores, tests, summary, report, out_dir) + "\n"
    )
    C.log("wrote", out_dir / README_FILE)

    # ---- the run's own audit trail, printed rather than hidden --------------
    C.log("panel:", len(manifest), "manifest row(s);",
          len(scores), "scored (species x matrix) rows;",
          len(summary), "matrix row(s) in the summary")
    unmeasurable = summary[summary.measurable == FALSE]
    if len(unmeasurable):
        C.log("UNMEASURABLE (scores absent, never zero; excluded from every aggregate):",
              ", ".join(f"{r.requested_name}={r.motif_id}" for r in unmeasurable.itertuples()))
    ambiguous = summary[summary.requested_name.duplicated(keep=False)]
    if len(ambiguous):
        C.log("AMBIGUOUS NAME (every match carried):",
              ", ".join(f"{r.requested_name}={r.motif_id}" for r in ambiguous.itertuples()))
    if report["cross_check_disagreements"]:
        C.log("WARNING: the enrichment table and the chromVAR table disagree on the",
              "model log2 OR for", len(report["cross_check_disagreements"]), "row(s):",
              "; ".join(report["cross_check_disagreements"][:3]))
    else:
        C.log("cross-check:", report["cross_check_rows"],
              "row(s) agree on the model log2 OR between the enrichment table and the",
              "chromVAR table (max abs gap",
              f"{report['cross_check_max_abs_gap']:g})")
    if report["missing_cells"]:
        C.log("WARNING: no model and no chromVAR row for:",
              ", ".join(report["missing_cells"][:5]))
    if report["species_level_unmeasurable"]:
        C.log("WARNING: species whose two peak sets carry no hit for an otherwise",
              "measurable matrix (recorded absent):",
              ", ".join(report["species_level_unmeasurable"][:5]))
    C.log("tail support:",
          int(summary.n_lower_tail_supported.sum()), "lower-tail,",
          int(summary.n_upper_tail_supported.sum()), "upper-tail cell-level flags;",
          int(summary.n_model_fdr_positive_supported.sum()), "positive-enrichment flags")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
