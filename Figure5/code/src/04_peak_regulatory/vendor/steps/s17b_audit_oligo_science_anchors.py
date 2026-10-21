#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve the frozen Oligo science-anchor panel against the motif catalog.

WHY this exists
---------------
The primary Oligo result is a family/module statement: the SOX module is
recurrent, the bHLH/E-box module is consistently depleted, NKX and POU3 are
secondary. A reader of the source paper's cerebellar GRN will immediately ask
the narrower question -- what about OLIG2 itself, SOX10, NKX2-2, POU3F3? -- and
that question cannot be answered from a module median, because a module median
is taken over its members and a member that matches no peak contributes a
zero. This audit is the input to that supplementary analysis: it says, for each
name the paper names, WHICH catalog matrix realises it and WHETHER that matrix
is measurable at all in this framework.

WHAT it does
------------
It reads the panel from `config["oligo_science_anchors"]` -- never a hardcoded
name or group -- and resolves every requested name against the motif catalog
with a case-insensitive EXACT match on the catalog motif name. It writes one row
per (requested name, catalog matrix) pair, so a name that matches two matrices
contributes two rows, and it writes a row even for a name that matches nothing.

The four frozen resolution rules, mirrored from the config comment:

  * case-insensitive EXACT match; no fuzzy, no prefix, no alias;
  * a name matching more than one matrix contributes EVERY match, one row each,
    all of them carrying `n_catalog_ids`, and the status names the ambiguity.
    Reporting only the measurable one would be selecting a matrix on the
    outcome;
  * a name matching nothing is `present=FALSE`, status `not_in_catalog`, with an
    empty motif_id; it is NEVER substituted by a similar motif;
  * a matrix the enrichment table carries with no foreground hit anywhere across
    its models is `measurable=FALSE`, status `zero_hit_unmeasurable`. Its scores
    are ABSENT, not zero, and `n_foreground_hits_max` is 0: that zero IS the
    reason, and it is a measurement of hit count;
  * a catalog matrix the enrichment table NEVER SCORED -- its motif_id has no row
    in `all_model_motif_scores.tsv` at all -- is NOT the same fact as a measured
    zero, and is never written as one. It takes `measurable=FALSE` with an EMPTY
    `n_foreground_hits_max` and the distinct status `not_in_enrichment_table`,
    exactly as a name with no matrix carries an EMPTY hit count rather than a 0,
    because there was nothing to count. "This matrix matched no peak" and "this
    table never looked at this matrix" are different facts, and only the first is
    a measurement; a catalogue/scoring release skew reported as a zero hit would
    be a fabricated negative on the very anchors this audit exists to measure.

`n_foreground_hits_max` is the maximum of `fg_hits` over every row of
`all_model_motif_scores.tsv` carrying that motif_id (all 21 models, all three
analyses). A matrix is measurable if the table scores it at all AND that maximum
is at least 1, i.e. the matrix matches at least one peak somewhere in the project
at the frozen FIMO threshold of 1e-4. The maximum, not one model's count, because
the question is whether the matrix can be evaluated at all, not how often one
model's Top-500 set carries it. A motif_id the table does not carry at all has no
maximum to take, so it gets an EMPTY cell and a status of its own -- see below.

BOUNDARIES
----------
This is a SUPPLEMENTARY, post-primary audit for the Oligo arm only. It reads
values and recomputes nothing: no upstream statistic is re-derived here. It is
not a gate: nothing downstream of it may be wired into s11 or into any PASS/FAIL
rule, and running it must leave `results_v2/final/FINAL_CLAIM_TABLE.tsv` and
`results_v2/final/CLAIM_EVIDENCE.json` byte-identical.

This step writes exactly one file, `anchor_manifest.tsv`, into the
`09_oligo_validation/science_anchors/` directory (or into `--outdir`, which the
tests use to redirect it), and prints a summary to stdout. Nothing is dropped
silently: an absent, duplicated or unmeasurable anchor is always a named row.

    python script/s17b_audit_oligo_science_anchors.py --config config/analysis_v2.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

MANIFEST_NAME = "anchor_manifest.tsv"
STEP_DIR = "09_oligo_validation"
SUBDIR = "science_anchors"

# Frozen column order. Downstream steps (the anchor analysis, the anchor plot
# and the anchor tests) read this by name; the order is part of the contract.
COLUMNS = [
    "requested_name",
    "group",
    "motif_id",
    "catalog_name",
    "present",
    "n_catalog_ids",
    "measurable",
    "n_foreground_hits_max",
    "status",
]

NOT_IN_CATALOG = "not_in_catalog"
ZERO_HIT = "zero_hit_unmeasurable"
NOT_SCORED = "not_in_enrichment_table"
OK = "ok"
DUPLICATE_SUFFIX = "_duplicate_name"


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def catalog_table(config: dict) -> tuple[pd.DataFrame, str]:
    """(motif_id, motif_name) for the whole library, plus where it came from.

    `results_v2/01_model_motif/motif_catalog.tsv` is the s01 artefact when it is
    present; otherwise the same catalog is read from the MEME library through
    `common.motif_catalog`, which is the identical id+name list (the enrichment
    table's 879 rows are keyed to it). The source actually used is echoed to
    stdout, so a reader can always tell which one produced the manifest.
    """
    written = C.config_path(config, "results") / "01_model_motif" / "motif_catalog.tsv"
    if written.exists():
        frame = pd.read_csv(written, sep="\t")
        missing = {"motif_id", "motif_name"} - set(frame.columns)
        if missing:
            raise ValueError(f"{written} lacks columns {sorted(missing)}")
        return frame[["motif_id", "motif_name"]].copy(), str(written)

    frame = C.motif_catalog(config)
    return frame, str(C.motif_database(config)) + " (motif_catalog.tsv absent)"


def foreground_hits(config: dict) -> tuple[pd.Series, int]:
    """(max fg_hits per motif_id over every row, number of models in the table)."""
    path = C.config_path(config, "results") / "01_model_motif" / "all_model_motif_scores.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; the anchor audit asks which matrices match at least "
            "one peak, which is a property of the enrichment table"
        )
    frame = pd.read_csv(path, sep="\t", usecols=["model", "motif_id", "fg_hits"])
    return frame.groupby("motif_id").fg_hits.max(), int(frame.model.nunique())


def panel(config: dict) -> tuple[list[tuple[str, str]], dict[str, object]]:
    """(groups in the config dict's own order, non-list keys found in the block).

    The block holds both the panel and two scalars (`null_repeats`, `fdr`). The
    scalars are returned rather than ignored so the summary can print them: a
    key that silently disappears from a config block is how a frozen parameter
    goes missing.
    """
    block = config.get("oligo_science_anchors")
    if not isinstance(block, dict) or not block:
        raise ValueError("config declares no `oligo_science_anchors` block")
    groups: list[tuple[str, str]] = []
    scalars: dict[str, object] = {}
    for group, value in block.items():
        if isinstance(value, (list, tuple)):
            for name in value:
                groups.append((str(group), str(name)))
        else:
            scalars[str(group)] = value
    if not groups:
        raise ValueError("`oligo_science_anchors` declares no anchor name")
    return groups, scalars


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------


def status_for(present: bool, measurable: bool, n_catalog_ids: int, *, scored: bool = True) -> str:
    """The row's status, naming the duplicate ambiguity when there is one.

    A matrix can be both duplicated and unmeasurable (the OLIG2 name has exactly
    that pair), so the two facts are composed rather than one silently winning:
    the base status says whether the matrix can be measured, and the suffix says
    that the panel name did not identify a single matrix. Neither fact may hide
    the other -- the duplicate is why the row exists twice, the zero hit is why
    its score is absent.

    `scored` separates the two ways a matrix is unmeasurable, and they are never
    merged: `ZERO_HIT` is a measurement (the table scores the matrix and it
    reaches no peak), `NOT_SCORED` is a missing input (the table has no row for
    the matrix at all), and only the first may be read as a hit count of zero.
    """
    if not present:
        return NOT_IN_CATALOG
    if measurable:
        base = OK
    else:
        base = ZERO_HIT if scored else NOT_SCORED
    return base + DUPLICATE_SUFFIX if n_catalog_ids > 1 else base


def build_manifest(config: dict) -> pd.DataFrame:
    """The audit table, one row per (requested name, catalog matrix) pair."""
    return _resolve(config)[0]


def _resolve(config: dict) -> tuple[pd.DataFrame, dict[str, object]]:
    """(manifest, what it was resolved against) -- the frame alone is the answer."""
    catalog, catalog_source = catalog_table(config)
    hits, n_models = foreground_hits(config)
    wanted, scalars = panel(config)

    lookup: dict[str, list[tuple[str, str]]] = {}
    for name in {n for _, n in wanted}:
        matches = catalog[catalog.motif_name.str.lower() == name.lower()]
        lookup[name] = list(zip(matches.motif_id, matches.motif_name))

    rows: list[dict] = []
    for group, name in wanted:
        matches = lookup[name]
        present = bool(matches)
        n_ids = len(matches)
        if not present:
            rows.append(
                dict(
                    requested_name=name,
                    group=group,
                    motif_id="",
                    catalog_name="",
                    present="FALSE",
                    n_catalog_ids=0,
                    measurable="FALSE",
                    n_foreground_hits_max="",
                    status=status_for(present=False, measurable=False, n_catalog_ids=0),
                )
            )
            continue
        for motif_id, catalog_name in matches:
            # A missing key is NOT a measured zero: the table has no row for this
            # matrix, so there is no hit count to report and the cell stays empty.
            # Defaulting to 0 here would assert "no peak carries the matrix" about
            # a matrix the table never scored -- the exact shape of a real
            # zero-hit row, and indistinguishable from it downstream.
            value = hits.get(motif_id)
            scored = value is not None and not pd.isna(value)
            n_hits = int(value) if scored else None
            measurable = scored and n_hits >= 1
            rows.append(
                dict(
                    requested_name=name,
                    group=group,
                    motif_id=motif_id,
                    catalog_name=catalog_name,
                    present="TRUE",
                    n_catalog_ids=n_ids,
                    measurable="TRUE" if measurable else "FALSE",
                    n_foreground_hits_max="" if n_hits is None else n_hits,
                    status=status_for(present, measurable, n_ids, scored=scored),
                )
            )

    frame = pd.DataFrame(rows, columns=COLUMNS)
    info = dict(
        catalog_source=catalog_source,
        n_catalog=len(catalog),
        n_models=n_models,
        scalars=scalars,
    )
    return frame, info


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


def resolve_outdir(config: dict, outdir: str | None) -> Path:
    """Where the manifest goes. `--outdir` redirects it without touching results_v2."""
    if outdir:
        path = Path(outdir)
        if not path.is_absolute():
            path = Path.cwd() / path
    else:
        path = C.config_path(config, "results") / STEP_DIR / SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def summarise(frame: pd.DataFrame, info: dict, out_path: Path) -> None:
    C.log("catalog:", info["n_catalog"], "matrices from", info["catalog_source"])
    C.log(
        "enrichment table:",
        info["n_models"],
        "models; measurability is max fg_hits over all of them",
    )
    if info["scalars"]:
        C.log(
            "panel scalars (not anchors):",
            ", ".join(f"{k}={v}" for k, v in info["scalars"].items()),
        )
    C.log("requested names:", len(frame.requested_name.unique()), "-> rows:", len(frame))

    for group in dict.fromkeys(frame.group):
        C.log(f"  group {group}")
        for row in frame[frame.group == group].itertuples():
            if row.present == "FALSE":
                C.log(f"    {row.requested_name:<9} absent from the catalog ({row.status})")
                continue
            if row.measurable == "TRUE":
                note = ""
            elif str(row.status).startswith(NOT_SCORED):
                note = "  <-- NOT SCORED: the enrichment table has no row for this matrix"
            else:
                note = "  <-- NO PEAK MATCHES: score absent"
            dup = f" [1 of {row.n_catalog_ids} matrices for this name]" if row.n_catalog_ids > 1 else ""
            C.log(
                f"    {row.requested_name:<9} {row.motif_id:<10} {row.catalog_name:<12} "
                f"max_fg_hits={row.n_foreground_hits_max:<4} {row.status}{dup}{note}"
            )

    counts = frame.status.value_counts().to_dict()
    C.log("status counts:", counts)

    absent = frame[frame.present == "FALSE"].requested_name.tolist()
    if absent:
        C.log("NOT IN CATALOG (never substituted by a similar motif):", ", ".join(absent))
    unmeasurable = frame[frame.status.str.startswith(ZERO_HIT)]
    if len(unmeasurable):
        C.log(
            "UNMEASURABLE at the frozen FIMO threshold (no peak carries the matrix, "
            "so its scores are absent rather than zero):",
            ", ".join(f"{r.requested_name}={r.motif_id}" for r in unmeasurable.itertuples()),
        )
    unscored = frame[frame.status.str.startswith(NOT_SCORED)]
    if len(unscored):
        C.log(
            "NOT SCORED BY THE ENRICHMENT TABLE (no row for these matrix ids in "
            "all_model_motif_scores.tsv, so nothing was measured -- a catalogue/"
            "scoring skew, not a zero hit; their hit count is EMPTY, never 0):",
            ", ".join(f"{r.requested_name}={r.motif_id}" for r in unscored.itertuples()),
        )
    duplicated = frame[frame.n_catalog_ids > 1]
    if len(duplicated):
        C.log(
            "AMBIGUOUS NAMES (every match is kept; the name alone does not identify "
            "one matrix):",
            ", ".join(
                f"{name} -> {list(group.motif_id)}"
                for name, group in duplicated.groupby("requested_name", sort=False)
            ),
        )
    C.log("wrote", out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(C.CONFIG_DIR / "analysis_v2.yaml"))
    parser.add_argument(
        "--outdir",
        default=None,
        help=(
            "directory to write anchor_manifest.tsv into; defaults to "
            f"<results>/{STEP_DIR}/{SUBDIR}"
        ),
    )
    args = parser.parse_args(argv)

    C.main_guard()
    config = C.load_config(args.config)

    frame, info = _resolve(config)
    out_dir = resolve_outdir(config, args.outdir)
    out_path = out_dir / MANIFEST_NAME
    C.save_table(frame, out_path, sep="\t")
    summarise(frame, info, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
