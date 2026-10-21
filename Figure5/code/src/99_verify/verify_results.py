#!/usr/bin/env python3
"""Acceptance check over the whole results tree.

Checks, in order of how expensive the failure would be to discover late:

  1. EVERY PDF keeps editable text. This is the requirement most easily lost
     silently -- a figure saved with a rasterised axes, or through a different
     code path, still looks correct.
  2. NO figure or table is empty, and every PNG has a plausible size. A
     zero-byte figure from a crashed write is otherwise indistinguishable from a
     figure nobody looked at.
  3. THE DECLARED OUTPUTS EXIST. Each analysis declares what it produces; a
     missing one is reported by name.
  4. THE RELEASE PREMISE HOLDS in the written tables: no figure or table in this
     repository should show human astroglia at a Carnegie stage, because there
     are none.
  5. DETERMINISM, where cheap: re-running an analysis would produce the same
     bytes. Not checked here -- see the note in the output.

Writes `results/verify/verify_report.json` and a one-line-per-check `.txt`, and
exits non-zero if anything failed, so it can gate a release.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from common import plotting  # noqa: E402

# What each analysis promises to produce. Paths are relative to results/.
CONTRACT: dict[str, list[str]] = {
    "00_audit": [
        "input_report.json", "input_report.txt", "stage_vocabulary.csv",
        "celltype_palette.csv", "release_edit_check.csv",
    ],
    "01_global_umap": [
        "umap_*_species.pdf", "umap_*_species.png",
        "umap_*_Celltype.pdf", "umap_*_Celltype.png",
        "legend_*_species.pdf", "legend_*_Celltype.pdf",
        "colors_*_species.csv", "colors_*_Celltype.csv",
        "celltype_cells_by_species.csv", "umap_parameters.json",
    ],
    "02_astroglia_umap": [
        "astroglia_human_stage_umap.pdf", "astroglia_mouse_stage_umap.pdf",
        "astroglia_marmoset_stage_umap.pdf", "astroglia_bonobo_stage_umap.pdf",
        "astroglia_macaque_stage_umap.pdf", "astroglia_opossum_stage_umap.pdf",
        "progenitor_human_stage_umap.pdf",
        "legend_astroglia_developmental_stage.pdf",
        "astroglia_stage_centroids.csv", "astroglia_counted_cells.csv",
        "astroglia_cells_by_stage.csv",
        "cross_species_developmental_alignment_3pairs.pdf",
        "cross_species_developmental_alignment_to_human.pdf",
        "alignment_summary.json",
    ],
    "03_developmental": [
        "Fig1_astroglia_stage_umap_small_multiples.pdf",
        "Fig1_progenitor_stage_umap_small_multiples.pdf",
        "Fig2_astroglia_direction_cosine_heatmap.pdf",
        "Fig3_astroglia_LOSO_stage_prediction.pdf",
        "Fig4_all_celltype_direction_conservation.pdf",
        "astroglia_pairwise_direction_cosine.csv",
        "astroglia_pairwise_permutation_pvalues.csv",
        "astroglia_LOSO_stage_prediction_metrics.csv",
        "loso_stage_rank_vs_stage_index.csv",
        "all_celltype_developmental_direction_conservation.csv",
        "single_species_celltypes.csv",
        "analysis_summary.txt", "developmental_direction_summary.json",
    ],
    # The arm directories are the ones the vendored steps write; the four shared
    # steps appear inside each arm as symlinks into _shared/.
    "04_peak_regulatory": [
        "release_audit/audit_checks.tsv", "release_audit/astroglia_windows.tsv",
        "_shared.sha256", "_shared/relabel_audit.tsv",
        "_shared/03_chromvar_input", "_shared/04_chromvar",
        "_shared/13_rank_curve", "_shared/16_sequence_maps",
        "top500/00_audit", "top500/01_model_motif", "top500/02_astroglia_model",
        "top500/05_astroglia_validation", "top500/06_purkinje_model",
        "top500/07_purkinje_validation", "top500/08_oligo_model",
        "top500/09_oligo_validation", "top500/15_motif_forest",
    ],
}

MIN_PNG_BYTES = 2_000


def iter_files(root: Path, suffix: str | None = None) -> list[Path]:
    """Every file under `root`, FOLLOWING symlinked directories, deduplicated.

    `Path.rglob` does not descend into a symlinked directory, and analysis 04's
    results are exposed at `results/04_peak_regulatory` as a symlink -- so
    `results_root.rglob("*.pdf")` silently missed 92 of 113 PDFs. An acceptance
    check that under-counts is worse than none: it reports "0 not OK" about a
    subset while reading as a statement about everything.

    `os.walk(followlinks=True)` does follow, but then reaches the same file
    twice through the arm symlinks (`top500/03_chromvar_input ->
    ../_shared/03_chromvar_input`), so results are de-duplicated by real path.
    """
    seen: dict[str, Path] = {}
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            if suffix and not name.endswith(suffix):
                continue
            path = Path(dirpath) / name
            try:
                key = os.path.realpath(path)
            except OSError:  # pragma: no cover - broken link
                key = str(path)
            seen.setdefault(key, path)
    return sorted(seen.values(), key=lambda p: str(p))


def resolve(results_root: Path, analysis: str, pattern: str) -> list[Path]:
    base = results_root / analysis
    if any(ch in pattern for ch in "*?["):
        return sorted(base.glob(pattern))
    return [base / pattern] if (base / pattern).exists() else []


def check_contract(results_root: Path) -> list[dict]:
    rows = []
    for analysis, patterns in CONTRACT.items():
        for pattern in patterns:
            found = resolve(results_root, analysis, pattern)
            rows.append({
                "check": "contract", "analysis": analysis, "item": pattern,
                "status": "OK" if found else "MISSING",
                "detail": f"{len(found)} match(es)" if found else "not found",
            })
    return rows


def check_pdfs(results_root: Path) -> list[dict]:
    """Every PDF must carry text that an editor can actually modify.

    Two failure modes are distinguished, because they need different fixes:
    `NO_TEXT` means the labels were rasterised; `TYPE3_FONT` means the text is
    there and selectable but stored as drawing procedures, so it cannot be
    retyped. The second is matplotlib's default and is what `pdf.fonttype = 42`
    prevents, so seeing it means some code path bypassed the style.
    """
    rows = []
    for path in iter_files(results_root, ".pdf"):
        report = plotting.pdf_text_report([path])[0]
        if report["editable_text"]:
            status = "OK"
        elif not report["extractable"]:
            status = "NO_TEXT"
        else:
            status = "TYPE3_FONT"
        rows.append({
            "check": "pdf_text", "analysis": path.parent.name, "item": path.name,
            "status": status,
            "detail": f"{report['characters']} characters, fonts {report['fonts']}"
                      + (f" | {report.get('error', '')}" if status != "OK" else ""),
        })
    return rows


def check_nonempty(results_root: Path) -> list[dict]:
    rows = []
    wanted = {".png", ".csv", ".tsv", ".json", ".txt", ".npy", ".npz"}
    for path in iter_files(results_root):
        if path.suffix.lower() not in wanted:
            continue
        size = path.stat().st_size
        threshold = MIN_PNG_BYTES if path.suffix.lower() == ".png" else 1
        rows.append({
            "check": "nonempty", "analysis": path.parent.name, "item": path.name,
            "status": "OK" if size >= threshold else "EMPTY",
            "detail": f"{size} bytes",
        })
    return rows


def check_release_premise(results_root: Path) -> list[dict]:
    """No written table may place human astroglia at a Carnegie stage.

    This is the one claim the whole release rests on, so it is checked against
    the artefacts rather than only against the input.
    """
    rows = []
    astroglia_csv = results_root / "02_astroglia_umap" / "astroglia_cells_by_stage.csv"
    if astroglia_csv.exists():
        frame = pd.read_csv(astroglia_csv)
        bad = frame[(frame["Celltype"].astype(str).str.startswith("astroglia"))
                    & (frame["species"] == "Human")
                    & (frame["stage"].isin(["CS18-19", "CS20", "CS22-23"]))
                    & (frame["n_cells"] > 0)]
        rows.append({
            "check": "release_premise", "analysis": "02_astroglia_umap",
            "item": "astroglia_cells_by_stage.csv",
            "status": "OK" if bad.empty else "FAIL",
            "detail": (f"{int(bad['n_cells'].sum())} human astroglia at Carnegie stages"
                       if not bad.empty else "no human astroglia at Carnegie stages"),
        })
    windows = results_root / "04_peak_regulatory" / "release_audit" / "astroglia_windows.tsv"
    if windows.exists():
        frame = pd.read_csv(windows, sep="\t")
        early = frame.loc[frame["window"].eq("early_embryonic")]
        if len(early):
            rows.append({
                "check": "release_premise", "analysis": "04_peak_regulatory",
                "item": "astroglia_windows.tsv",
                "status": "OK" if not bool(early["in_this_release"].iloc[0]) else "FAIL",
                "detail": ("early_embryonic excluded from the astroglia arm"
                           if not bool(early["in_this_release"].iloc[0])
                           else "early_embryonic still in the arm"),
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=REPO / "results")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    results_root = args.results
    if not results_root.is_dir():
        raise SystemExit(f"no results tree at {results_root}")
    out = args.out or (results_root / "verify")
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    rows += check_contract(results_root)
    rows += check_pdfs(results_root)
    rows += check_nonempty(results_root)
    rows += check_release_premise(results_root)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "verify_report.tsv", sep="\t", index=False)

    failed_statuses = {"MISSING", "NO_TEXT", "TYPE3_FONT", "ERROR", "EMPTY", "FAIL"}
    failed = frame.loc[frame["status"].isin(failed_statuses)]

    print(f"[VERIFY] {results_root}")
    for check, group in frame.groupby("check"):
        bad = group.loc[group["status"].isin(failed_statuses)]
        print(f"  {check:18s} {len(group):5d} checked, {len(bad):3d} not OK")
    if len(failed):
        print("\nthe following did not pass:")
        for row in failed.itertuples():
            print(f"  {row.status:8s} {row.analysis}/{row.item}: {row.detail}")

    summary = {
        "results_root": str(results_root),
        "checks": int(len(frame)),
        "not_ok": int(len(failed)),
        "by_status": frame["status"].value_counts().to_dict(),
        "not_checked": {
            "determinism": (
                "re-running an analysis and comparing bytes is not done here. The "
                "permutation test and the DTW are seeded, so it is expected to "
                "hold; checking it costs a full re-run."
            ),
        },
    }
    (out / "verify_report.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"\n[DONE] {len(frame)} checks, {len(failed)} not OK -> {out}")
    return 1 if len(failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
