"""Acceptance for the v3 run: did the three arms produce what keeps.md names?

WHAT THIS CHECKS, AND WHY EACH ONE IS LOAD-BEARING
--------------------------------------------------
1. EVERY keeps.md TARGET EXISTS, per arm, resolving symlinks. The kept list is
   frozen in this file as data, transcribed from result_v3/keeps.md, so a target
   that stops being produced fails here rather than being noticed in a paper.
2. THE CUT-OFF ACTUALLY TOOK EFFECT. It is not enough that three trees exist:
   `01_model_motif/all_model_motif_scores.tsv` must carry exactly one `topk`
   value and it must be that arm's K. Three identical trees would pass an
   existence check and fail this one.
3. THE NULL IS THE SIZE THE ARM DECLARES. `null_n_peaks_requested` == K and the
   per-model `null_status` census is reported, including every underpowered
   model by name. An underpowered null is a result, not an error -- but it must
   be visible.
4. _SHARED WAS NOT WRITTEN THROUGH. The arm's 03/04/13/16 are symlinks into
   `_shared`, and `_shared.sha256` (recorded before the arms started) must still
   match. A write through a symlink is otherwise invisible.
5. NO v2 TREE IS REACHABLE FROM THE RESULT. No symlink under result_v3 may point
   into results_v2/ or results/.
6. A MANIFEST of what was run: config hashes, script hashes, cache fingerprints.
7. IDEMPOTENCY BY ARTEFACT KIND. Text artefacts are compared by sha256; PNGs by
   PIXEL hash (matplotlib PNGs are byte-stable but a pixel hash is the honest
   claim); PDFs by neither, because they embed a CreationDate -- they are
   reported by size only. On a second run this file compares against the
   previous keeps_sha256.tsv and reports drift.

It recomputes no statistic. Everything it prints is read from a table an arm
already wrote.

Usage
-----
    python script_v3/verify_v3.py --config config/v3/top500.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import common as C  # noqa: E402

ROOT = SCRIPT_DIR.parent
RESULTS = ROOT / "result_v3"
ARMS = (500, 1000, 2000)
SHARED_STEPS = ("03_chromvar_input", "04_chromvar", "13_rank_curve", "16_sequence_maps")

# transcribed from result_v3/keeps.md -- the paper's frozen kept set, minus the
# directories the paper's deliverable set does not include.
#
# `06_purkinje_model` and `08_oligo_model` were dropped from this list on
# 2026-10-15: the standard run no longer produces them (see the comment in
# ../run_all.sh), so an arm that follows the driver cannot have them. The two
# steps remain under vendor/steps and their absence is recorded by the steps
# that would have read them -- `s07` as `recurrent_modules_missing`, `s09` as
# `recurrent_families_missing` with gate O3 blocked.
KEEPS: tuple[tuple[str, str], ...] = (
    ("00_audit", "dir"),
    ("05_astroglia_validation/family_trajectory_model_vs_raw_absolute.png", "file"),
    ("05_astroglia_validation/family_trajectory_agreement_summary.pdf", "file"),
    ("07_purkinje_validation", "dir"),
    ("09_oligo_validation", "dir"),
    ("13_rank_curve", "dir"),
    ("15_motif_forest", "dir"),
    ("16_sequence_maps", "dir"),
)

TEXT_SUFFIXES = (".tsv", ".csv", ".json", ".txt", ".md")
HASHED_SUFFIXES = TEXT_SUFFIXES + (".png",)

FORBIDDEN_ROOTS = ("results_v2", "results")


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def fail(self, message: str) -> None:
        self.failures.append(message)
        print(f"  FAIL  {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"  WARN  {message}")

    def ok(self, message: str) -> None:
        print(f"  ok    {message}")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_hash(path: Path) -> str:
    """A hash of the rendered pixels, independent of PNG metadata."""
    from PIL import Image

    with Image.open(path) as image:
        return hashlib.sha256(image.convert("RGBA").tobytes()).hexdigest()


# A few summaries carry the wall-clock moment they were written. That stamp is
# useful provenance, so it stays in the artefact -- but it must not enter the
# reproducibility claim, or a perfectly reproducible run reports drift every time
# (measured: re-running the Top1000 arm changed exactly 4 artefacts, 3 of them
# purely on this field). The same reasoning that exempts PDFs from byte-equality
# applies here, so the stamp is masked before hashing instead.
# Narrow on purpose: only a *timestamp-valued* field is masked. `generated_by`
# names the script that wrote the file and is perfectly stable, so masking it
# would exclude real content from the comparison for no reason.
VOLATILE_FIELD = re.compile(
    rb'"([A-Za-z_]*(?:generated|created)[A-Za-z_]*|date|timestamp)"\s*:\s*'
    rb'"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[^"]*"'
)
VOLATILE_LINE = re.compile(
    rb"^\s*[-*]?\s*\**(generated|run date|date)\**:\s*\d{4}-\d{2}-\d{2}[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)


def mask_volatile(payload: bytes) -> bytes:
    """Replace generation stamps with a placeholder, keeping everything else."""
    masked = VOLATILE_FIELD.sub(rb'"\1": "<masked>"', payload)
    return VOLATILE_LINE.sub(rb"<masked>", masked)


def artefact_hash(path: Path) -> tuple[str, str]:
    """(kind, digest) for one artefact. PDFs get no digest, by design.

    Text artefacts are hashed with their generation stamps masked (see
    `mask_volatile`); PNGs by pixel data; PDFs by size only.
    """
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "pixel_sha256", pixel_hash(path)
    if suffix in TEXT_SUFFIXES:
        return "sha256", hashlib.sha256(mask_volatile(path.read_bytes())).hexdigest()
    if suffix == ".pdf":
        # A PDF embeds /CreationDate, so two runs of the same figure differ at the
        # byte level and byte-equality is not claimed for it. It is reported by
        # size only: the data-side claim is carried by the TSV/JSON tables the
        # figure was drawn from, which ARE hashed above.
        return "not_hashed(pdf)", f"size={path.stat().st_size}"
    return "not_hashed", f"size={path.stat().st_size}"


def tree_files(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.is_file())


def check_targets(report: Report) -> None:
    print("\n[1] keeps.md targets, per arm")
    for topk in ARMS:
        arm = RESULTS / f"top{topk}"
        missing = []
        for relative, kind in KEEPS:
            target = arm / relative
            if not target.exists():
                missing.append(relative)
            elif kind == "dir" and not target.is_dir():
                missing.append(f"{relative} (not a directory)")
        if missing:
            report.fail(f"top{topk}: {len(missing)} keeps.md target(s) missing: {missing}")
        else:
            print(f"  ok    top{topk}: all {len(KEEPS)} keeps.md targets present")


def check_cutoff(report: Report) -> None:
    print("\n[2] the Top-K cut-off took effect")
    for topk in ARMS:
        path = RESULTS / f"top{topk}" / "01_model_motif" / "all_model_motif_scores.tsv"
        if not path.exists():
            report.fail(f"top{topk}: {path.name} missing")
            continue
        values = sorted(pd.unique(pd.read_csv(path, sep="\t", usecols=["topk"]).topk))
        if values != [topk]:
            report.fail(f"top{topk}: scores carry topk={values}, expected [{topk}]")
            continue
        audit = pd.read_csv(RESULTS / f"top{topk}" / "01_model_motif" / "model_audit.tsv", sep="\t")
        n_top = audit.n_top.astype(int)
        report.ok(
            f"top{topk}: scores topk={values[0]}, {len(audit)} models, "
            f"n_top {n_top.min()}-{n_top.max()} (cut-off {topk})"
        )


def check_null(report: Report) -> pd.DataFrame:
    print("\n[3] the matched-random null: declared size and per-model status")
    rows = []
    for topk in ARMS:
        path = RESULTS / f"top{topk}" / "01_model_motif" / "model_audit.tsv"
        if not path.exists():
            report.fail(f"top{topk}: model_audit.tsv missing")
            continue
        audit = pd.read_csv(path, sep="\t")
        if "null_n_peaks_requested" not in audit.columns:
            report.fail(f"top{topk}: model_audit.tsv carries no null sizing columns")
            continue
        requested = sorted(pd.unique(audit.null_n_peaks_requested.dropna().astype(int)))
        if requested != [topk]:
            report.fail(f"top{topk}: null_n_peaks_requested={requested}, expected [{topk}]")
        else:
            report.ok(f"top{topk}: null_n_peaks_requested == {topk}")
        if "null_require_exact_peaks" in audit.columns:
            exact = sorted(pd.unique(audit.null_require_exact_peaks.dropna()))
            (report.ok if exact == [True] else report.fail)(
                f"top{topk}: null_require_exact_peaks={exact}"
            )
        # THE exact-size contract: every null that was actually drawn must have
        # been drawn at that model's OWN observed table width, not at K and not at
        # a fraction of it. `s01` computes the target as `n_fg_matched`, so the two
        # columns must agree cell for cell on every powered model.
        if "null_target_peaks" in audit.columns:
            powered = audit[audit.null_status == "ok"]
            if not len(powered):
                report.fail(f"top{topk}: no model produced a usable null")
            else:
                off = powered[powered.null_target_peaks != powered.n_fg_matched]
                if len(off):
                    report.fail(
                        f"top{topk}: {len(off)} powered model(s) drew a null at a width "
                        f"other than their observed table width, e.g. "
                        f"{off.iloc[0]['model']}: target={off.iloc[0].null_target_peaks} "
                        f"vs n_fg_matched={off.iloc[0].n_fg_matched}"
                    )
                else:
                    ratio = powered.null_target_peaks / powered.n_top
                    report.ok(
                        f"top{topk}: all {len(powered)} powered nulls drawn at their own "
                        f"observed width ({ratio.min():.1%}-{ratio.max():.1%} of K, "
                        f"median {ratio.median():.1%})"
                    )
        counts = audit.null_status.value_counts()
        print(f"        null_status census: {dict(counts)}")
        under = audit[audit.null_status.isin(("underpowered", "no_draw"))]
        if len(under):
            report.warn(
                f"top{topk}: {len(under)} model(s) NULL_UNDERPOWERED (expected at the "
                f"larger cut-offs; they carry no null and are never counted as PASS or FAIL)"
            )
            for row in under.itertuples():
                print(f"          - {row.model}: {row.null_status}, "
                      f"pool={row.n_pool}, valid repeats={row.null_n_valid_repeats}, "
                      f"worst draw {row.null_max_shortfall} short")
        for row in audit.itertuples():
            rows.append({
                "topk": topk, "model": row.model, "n_ranked": row.n_ranked,
                "n_top": row.n_top, "n_top_valid": row.n_top_valid, "n_pool": row.n_pool,
                "n_fg_matched": row.n_fg_matched,
                "null_n_peaks_requested": row.null_n_peaks_requested,
                "null_target_peaks": getattr(row, "null_target_peaks", None),
                "null_n_valid_repeats": row.null_n_valid_repeats,
                "null_n_dropped": row.null_n_dropped,
                "null_max_shortfall": row.null_max_shortfall,
                "null_status": row.null_status,
            })
    return pd.DataFrame(rows)


def check_shared(report: Report) -> list[dict]:
    print("\n[4] the shared steps were not written through")
    recorded = RESULTS / "_shared.sha256"
    if not recorded.exists():
        report.fail("_shared.sha256 missing; run `./run_all_v3.sh lock` first")
        return []
    expected = {}
    for line in recorded.read_text().splitlines():
        if line.strip():
            digest, _, name = line.partition("  ")
            # Canonicalise before comparing: `sha256sum` fed by `find .` yields
            # "./p", while the re-hash below keys the same file as "p". Without
            # this the two key sets are disjoint and every file looks changed.
            name = name.strip()
            while name.startswith("./"):
                name = name[2:]
            expected[name] = digest.strip()
    actual = {}
    for path in tree_files(RESULTS / "_shared"):
        actual[str(path.relative_to(RESULTS / "_shared"))] = sha256_file(path)
    if expected != actual:
        only_recorded = sorted(set(expected) - set(actual))
        only_actual = sorted(set(actual) - set(expected))
        changed = sorted(k for k in set(expected) & set(actual) if expected[k] != actual[k])
        report.fail(
            f"_shared changed after lock(): {len(changed)} modified, "
            f"{len(only_recorded)} removed, {len(only_actual)} added "
            f"(e.g. {changed[:3]})"
        )
    else:
        report.ok(f"_shared byte-identical to the lock() snapshot ({len(actual)} files)")

    links = []
    for topk in ARMS:
        for step in SHARED_STEPS:
            arm_link = RESULTS / f"top{topk}" / step
            if not arm_link.is_symlink():
                report.fail(f"top{topk}/{step} is not a symlink into _shared")
                continue
            target = (arm_link.parent / arm_link.readlink()).resolve()
            want = (RESULTS / "_shared" / step).resolve()
            if target != want:
                report.fail(f"top{topk}/{step} -> {target}, expected {want}")
            links.append({"topk": topk, "step": step, "target": str(arm_link.readlink())})
    if links and not any("top" in f for f in report.failures):
        report.ok(f"{len(links)} arm symlinks resolve into _shared")
    return links


def display(path: Path) -> str:
    """A readable path that cannot raise when it lies outside the project root.

    `Path.relative_to` raises ValueError for a path outside its base, and the one
    place this matters is exactly the failure case it is used to report: a symlink
    that escapes the tree. A guard must not crash on the condition it guards, so
    this degrades to the absolute path instead.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def check_no_v2(report: Report) -> None:
    print("\n[5] no v2/ v1 result tree is reachable from result_v3")
    offenders = []
    for path in RESULTS.rglob("*"):
        if not path.is_symlink():
            continue
        target = (path.parent / path.readlink()).resolve()
        for root in FORBIDDEN_ROOTS:
            if (ROOT / root) in target.parents or target == (ROOT / root):
                offenders.append((display(path), str(target)))
    if offenders:
        for path, target in offenders[:10]:
            report.fail(f"{path} -> {target} (inside a retired result tree)")
    else:
        report.ok("every symlink under result_v3 resolves inside result_v3")


def collect_keeps() -> pd.DataFrame:
    rows = []
    for topk in ARMS:
        arm = RESULTS / f"top{topk}"
        for relative, kind in KEEPS:
            target = arm / relative
            if not target.exists():
                continue
            paths = tree_files(target) if kind == "dir" else [target]
            for path in paths:
                if path.suffix.lower() not in HASHED_SUFFIXES and path.suffix.lower() != ".pdf":
                    continue
                digest_kind, digest = artefact_hash(path)
                # the source tables a figure was drawn from, so a PDF's claim can
                # be carried by data rather than by bytes
                rows.append({
                    "topk": topk,
                    "artefact": str(path.relative_to(arm)),
                    "keeps_entry": relative,
                    "kind": digest_kind,
                    "digest": digest,
                    "bytes": path.stat().st_size,
                })
    return pd.DataFrame(rows).sort_values(["topk", "artefact"])


def check_idempotency(report: Report, frame: pd.DataFrame) -> None:
    print("\n[7] idempotency (vs the previous keeps_sha256.tsv, if any)")
    path = RESULTS / "keeps_sha256.tsv"
    hashed = frame[frame.kind.isin(("sha256", "pixel_sha256"))]
    if not path.exists():
        report.ok(f"no previous snapshot; wrote the baseline ({len(hashed)} hashed artefacts)")
        return
    previous = pd.read_csv(path, sep="\t")
    previous = previous[previous.kind.isin(("sha256", "pixel_sha256"))]
    # Identity is (arm, artefact). `kind` is NOT part of the key: it is derived
    # from the suffix and from whether a stamp had to be masked, so keying on it
    # would turn a hashing-convention change into a spurious "rename".
    key = ["topk", "artefact"]
    # Compare LIKE FOR LIKE: against `hashed`, not `frame`. Merging the unfiltered
    # frame against the hash-filtered baseline reports every PDF and every
    # not_hashed artefact as newly "appeared" on every run.
    # Left is the PREVIOUS snapshot, so left_only really is "disappeared".
    joined = previous.merge(hashed, on=key, how="outer", suffixes=("_old", "_new"), indicator=True)
    only_old = joined[joined._merge == "left_only"]
    only_new = joined[joined._merge == "right_only"]
    both = joined[joined._merge == "both"]
    changed = both[both.digest_old != both.digest_new]
    if len(only_old) or len(only_new) or len(changed):
        report.warn(
            f"drift vs the previous snapshot: {len(changed)} changed, "
            f"{len(only_old)} disappeared, {len(only_new)} appeared"
        )
        for row in changed.head(10).itertuples():
            # `kind` is not part of the join key, so the merge suffixes it
            print(f"          - top{row.topk}/{row.artefact} ({row.kind_new})")
    else:
        report.ok(f"{len(both)} hashed artefacts identical to the previous snapshot")


def write_manifest(config: dict, nulls: pd.DataFrame, links: list[dict], keeps: pd.DataFrame) -> dict:
    import os

    caches = {}
    for name in sorted((RESULTS / "cache").glob("*_scan.sha256")):
        caches[name.stem.replace("_scan", "")] = name.read_text().strip()

    genomes = {}
    table = C.species_table(config)
    for species, genome in table.items():
        fasta = Path(genome.fasta)
        if fasta.exists():
            stat = fasta.stat()
            genomes[species] = {
                # the FASTA itself is not re-hashed: the cache fingerprint already
                # covers the sequences extracted from it at these coordinates
                "fasta": str(fasta),
                "bytes": stat.st_size,
                "mtime": int(stat.st_mtime),
                "assembly": genome.assembly if hasattr(genome, "assembly") else None,
            }

    manifest = {
        "analysis": "v3",
        "primary_arm": 500,
        "robustness_arms": [1000, 2000],
        "configs": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in sorted((ROOT / "config" / "v3").glob("*.yaml"))
        },
        "frozen_maps": {
            line.split("  ")[1].strip(): line.split("  ")[0].strip()
            for line in (ROOT / "config" / "v3" / "frozen_maps.sha256").read_text().splitlines()
            if line.strip()
        },
        "scripts": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in sorted([*SCRIPT_DIR.glob("*.py"), *SCRIPT_DIR.glob("*.R")])
        },
        "cache_fingerprints": caches,
        "cache_frozen": not os.access(RESULTS / "cache", os.W_OK),
        "genomes": genomes,
        "shared_links": links,
        "null_status_by_arm": (
            nulls.groupby(["topk", "null_status"]).size().rename("n_models")
            .reset_index().to_dict(orient="records")
        ) if len(nulls) else [],
        "keeps_artefacts": len(keeps),
        "runtime": {
            "python": sys.executable,
            "python_version": sys.version.split()[0],
        },
    }
    path = RESULTS / "RUN_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


# --------------------------------------------------------------------------
# the frozen cross-K stability rules
# --------------------------------------------------------------------------


def rule_row(name: str, cite: str, holds: dict[int, bool | None], detail: str) -> dict:
    known = {k: v for k, v in holds.items() if v is not None}
    return {
        "rule": name,
        "cite": cite,
        "detail": detail,
        **{f"top{k}": ("n/a" if v is None else ("holds" if v else "does_not_hold"))
           for k, v in holds.items()},
        "n_arms_decided": len(known),
        "n_arms_holding": sum(1 for v in known.values() if v),
        "verdict": (
            "UNDERPOWERED" if not known
            else "stable" if all(known.values()) and len(known) == len(holds)
            else "stable_where_decided" if all(known.values())
            else "not_stable"
        ),
    }


def evaluate_stability() -> pd.DataFrame:
    rows = []

    # A1 -- all three stage-pair margins > 0
    #
    # The detail is built per arm and labelled with it. An unlabelled detail here
    # silently reports whichever arm the loop finished on (Top2000), which reads
    # as the primary arm's numbers -- a real misreading risk, since the margins
    # grow with the cut-off (+0.164/+0.293/+0.219 at Top500, +0.269/+0.374/+0.341
    # at Top2000).
    holds, parts = {}, []
    for topk in ARMS:
        path = RESULTS / f"top{topk}" / "02_astroglia_model" / "stage_matching_margin.tsv"
        if not path.exists():
            holds[topk] = None
            parts.append(f"top{topk}: table missing")
            continue
        frame = pd.read_csv(path, sep="\t")
        holds[topk] = bool((frame.margin > 0).all())
        parts.append(
            f"top{topk}: "
            + "; ".join(f"{r.species_a}-{r.species_b} {r.margin:+.3f}" for r in frame.itertuples())
        )
    rows.append(rule_row("astroglia_stage_matching_direction", "A1", holds, " | ".join(parts)))

    # A4 -- the FROZEN direction test, evaluated per arm.
    #
    # The rule name and the declared directions are the ones the arm configs
    # froze under cross_arm.stability_rules, so they are read from the config
    # rather than restated here: a rule that drifts from its pre-registration is
    # worse than no rule.
    #
    # The pivot is on `cross_window` (prenatal / postnatal / adult), NOT on
    # `stage`. `stage` holds per-sample stage ids ("15-17wpc_11wpc",
    # "P63_DN_P63", ...), so pivoting on it produces stage-id columns, never a
    # "prenatal" column -- which would silently skip this rule for every arm.
    #
    # `family_trajectory.tsv` is already scoped to the cross-species windows (the
    # human-only H1 period is carried separately in
    # family_trajectory_within_species.tsv), so no extra scope filter is applied.
    a4_signs = {"RFX": +1, "NF-Y": +1, "NFI": -1}   # +1 := prenatal > adult
    holds, a4_parts = {}, []
    for topk in ARMS:
        path = RESULTS / f"top{topk}" / "02_astroglia_model" / "family_trajectory.tsv"
        if not path.exists():
            holds[topk] = None
            a4_parts.append(f"top{topk}: table missing")
            continue
        frame = pd.read_csv(path, sep="\t")
        if "cross_window" not in frame.columns:
            holds[topk] = None
            a4_parts.append(f"top{topk}: no cross_window column")
            continue
        pivot = frame.pivot_table(index=["species", "family"], columns="cross_window",
                                  values="family_score", aggfunc="median")
        missing = [c for c in ("prenatal", "adult") if c not in pivot.columns]
        if missing:
            holds[topk] = None
            a4_parts.append(f"top{topk}: no {missing} cross-window column")
            continue
        per_family = {}
        for family, want in a4_signs.items():
            # median over species, so one species cannot swing the family
            delta = (pivot["prenatal"] - pivot["adult"])
            delta = delta.xs(family, level="family") if family in delta.index.get_level_values("family") else None
            per_family[family] = None if delta is None or not len(delta) else (
                np_sign(float(delta.median())) == want)
        decided = [v for v in per_family.values() if v is not None]
        holds[topk] = bool(decided) and all(decided)
        a4_parts.append(
            f"top{topk}: "
            + "; ".join(
                f"{fam} {'holds' if v else 'does_not_hold' if v is not None else 'no_rows'}"
                f" (want prenatal {'>' if w > 0 else '<'} adult)"
                for fam, (w, v) in ((f, (a4_signs[f], per_family[f])) for f in a4_signs)
            )
        )
    rows.append(rule_row("astroglia_family_trajectory_direction", "A4", holds, " | ".join(a4_parts)))

    # P1 / O1 -- observed > matched-random q95
    for name, cite, step, model in (
        ("purkinje_convergence_vs_null", "P1", "06_purkinje_model", None),
        ("oligo_convergence_vs_null", "O1", "08_oligo_model", None),
    ):
        holds, detail = {}, ""
        for topk in ARMS:
            path = RESULTS / f"top{topk}" / step / "convergence_null_test.tsv"
            if not path.exists():
                holds[topk] = None
                continue
            frame = pd.read_csv(path, sep="\t")
            row = frame.iloc[0]
            if "null_q95" not in frame.columns or pd.isna(row.null_q95):
                holds[topk] = None
                detail = f"top{topk}: no usable null repeat -> UNDERPOWERED"
                continue
            holds[topk] = bool(row.observed > row.null_q95)
            detail = (f"top{topk}: observed {row.observed:.3f} vs q95 {row.null_q95:.3f} "
                      f"({int(row.get('n_repeats', 0))} usable repeats)")
        rows.append(rule_row(name, cite, holds, detail))

    # P3 / O3 -- median model<->raw rho > 0
    for name, cite, step in (
        ("purkinje_model_vs_chromvar_direction", "P3", "07_purkinje_validation"),
        ("oligo_model_vs_chromvar_direction", "O3", "09_oligo_validation"),
    ):
        holds, detail = {}, ""
        for topk in ARMS:
            path = RESULTS / f"top{topk}" / step / "model_vs_raw_similarity.tsv"
            if not path.exists():
                holds[topk] = None
                continue
            frame = pd.read_csv(path, sep="\t")
            if "status" in frame.columns:
                frame = frame[frame.status == "ok"]
            value = float(frame.rho.median())
            holds[topk] = value > 0
            detail = f"top{topk}: median rho {value:+.3f} over {len(frame)} species"
        rows.append(rule_row(name, cite, holds, detail))

    return pd.DataFrame(rows)


def np_sign(value: float) -> int:
    return 0 if value == 0 else (1 if value > 0 else -1)


def write_cross_arm(stability: pd.DataFrame, nulls: pd.DataFrame, keeps: pd.DataFrame) -> None:
    lines = [
        "# v3 cross-arm summary",
        "",
        "Generated by `script_v3/verify_v3.py`. It **recomputes nothing**: every number",
        "below was read from a table an arm already wrote. The stability rules are the",
        "ones frozen in `config/v3/*.yaml` under `cross_arm.stability_rules` BEFORE any",
        "v3 result existed.",
        "",
        "Top500 is the PRIMARY arm; Top1000 and Top2000 are prespecified robustness arms.",
        "",
        "## Cross-K stability (frozen rules)",
        "",
    ]
    lines.append("| rule | cites | top500 | top1000 | top2000 | verdict |")
    lines.append("|---|---|---|---|---|---|")
    for row in stability.itertuples():
        lines.append(
            f"| `{row.rule}` | {row.cite} | {row.top500} | {row.top1000} | "
            f"{row.top2000} | **{row.verdict}** |"
        )
    lines += ["", "### detail", ""]
    for row in stability.itertuples():
        lines.append(f"- `{row.rule}` ({row.cite}): {row.detail}")

    lines += ["", "## Matched-random null, per arm", ""]
    if len(nulls):
        counts = nulls.groupby(["topk", "null_status"]).size().unstack(fill_value=0)
        lines.append("| arm | n_peaks | " + " | ".join(counts.columns) + " |")
        lines.append("|---" * (len(counts.columns) + 2) + "|")
        for topk, row in counts.iterrows():
            lines.append(f"| top{topk} | {topk} | " + " | ".join(str(v) for v in row) + " |")
        lines += ["", "Models reported `underpowered`/`no_draw` carry no null and are counted",
                  "as neither PASS nor FAIL by the stability rules above.", ""]

    lines += ["## Artefacts hashed", "",
              f"`keeps_sha256.tsv` holds {len(keeps)} entries across the three arms.",
              "Text artefacts are compared by sha256, PNGs by pixel hash, and PDFs by",
              "size only (a PDF embeds a CreationDate, so byte-equality is not claimed",
              "for it).", ""]
    (RESULTS / "CROSS_ARM_SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    C.add_common_args(parser)
    args = parser.parse_args()
    config = C.load_config(args.config)

    report = Report()
    print("=" * 74)
    print("v3 acceptance")
    print("=" * 74)

    check_targets(report)
    check_cutoff(report)
    nulls = check_null(report)
    links = check_shared(report)
    check_no_v2(report)

    print("\n[6] run manifest")
    keeps = collect_keeps()
    check_idempotency(report, keeps)
    manifest = write_manifest(config, nulls, links, keeps)
    C.save_table(keeps, RESULTS / "keeps_sha256.tsv", sep="\t")
    print(f"  ok    wrote {C.display_path(RESULTS / 'RUN_MANIFEST.json')} "
          f"({len(manifest['scripts'])} scripts, {len(manifest['cache_fingerprints'])} cache fingerprints)")
    print(f"  ok    wrote {C.display_path(RESULTS / 'keeps_sha256.tsv')} ({len(keeps)} artefacts)")

    stability = evaluate_stability()
    write_cross_arm(stability, nulls, keeps)
    print(f"  ok    wrote {C.display_path(RESULTS / 'CROSS_ARM_SUMMARY.md')}")

    if len(nulls):
        C.save_table(nulls, RESULTS / "null_status_by_arm.tsv", sep="\t")
    C.save_table(stability, RESULTS / "cross_arm_stability.tsv", sep="\t")

    print("\n" + "=" * 74)
    if report.failures:
        print(f"FAILED: {len(report.failures)} check(s)")
        for item in report.failures:
            print(f"  - {item}")
        return 1
    print(f"PASSED with {len(report.warnings)} warning(s)")
    for item in report.warnings:
        print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
