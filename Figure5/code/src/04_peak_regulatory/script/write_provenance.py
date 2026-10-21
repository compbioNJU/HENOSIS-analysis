#!/usr/bin/env python3
"""Freeze the provenance of the vendored core, and write PROVENANCE.md.

`vendor/` holds code that was NOT rewritten. The value of that decision depends
entirely on being able to show that the code is what it claims to be, so its
hashes are recorded here and re-checked by `script/release_audit.py` before every
run. If a vendored file ever changes, the audit fails and the run stops -- which
is the point: a silent edit inside the frozen core would make every downstream
number unattributable.

Run this once after populating `vendor/`, and again only if you deliberately
replace a vendored file (in which case say so in PROVENANCE.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
VENDOR = PKG / "vendor"
REPO = PKG.parents[1]

UPSTREAM = (
    "cross_species_peak_regulatory_analysis/script_v3 "
    "(the previous release's executed pipeline, result_v3 in that tree)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    """Best-effort upstream revision, if the source tree happens to be a repo."""
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5ad", type=Path, required=True,
                        help="the cell-level input this release is defined by")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="the peak-regulatory project's data/ directory")
    args = parser.parse_args()

    files = {
        str(path.relative_to(VENDOR)): sha256(path)
        for path in sorted(VENDOR.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".R", ".tsv", ".yaml"}
    }
    manifest = {
        "what_is_vendored": (
            "the previous release's executed pipeline: the FIMO scan and its "
            "cache, the GC/length-matched background, the exact-size matched "
            "random null, BH FDR, the chromVAR R step, the frozen config maps, "
            "and the step scripts whose behaviour this release does not change"
        ),
        "what_is_new": (
            "script/ in this package: the arm config generator, the vendored-core "
            "wiring, the release audit, the relabelling chromVAR exporter, and "
            "the driver"
        ),
        "upstream": UPSTREAM,
        "upstream_git_revision": git_revision(HERE.parents[3]),
        "files": files,
        "release_inputs": {
            "h5ad": str(args.h5ad.resolve()),
            "h5ad_sha256": sha256(args.h5ad) if args.h5ad.exists() else None,
            "data_root": str(args.data_root.resolve()),
        },
    }
    (VENDOR / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(f"[WROTE] {VENDOR / 'MANIFEST.json'}  ({len(files)} files)")

    lines = [
        "# Provenance of the vendored core",
        "",
        "This package is the previous release's peak-regulatory pipeline, run",
        "unchanged, behind a new analysis layer. Everything under `vendor/` is a",
        "byte-for-byte copy of the executed pipeline; everything under `script/`",
        "is new.",
        "",
        "## Why any of it is vendored rather than rewritten",
        "",
        "The parts that stay frozen are the parts whose correctness is not in",
        "question and whose failure would be silent: the FIMO scan and its cache,",
        "the GC- and length-matched background, the exact-size matched random",
        "null, the BH FDR, and the chromVAR step. Re-implementing those would",
        "change numbers without changing any scientific decision, and the change",
        "would be invisible in the output. They are copied, hash-pinned, and",
        "re-verified on every run.",
        "",
        f"Upstream: {UPSTREAM}",
        "",
        "## Frozen file hashes",
        "",
        "`script/release_audit.py` recomputes these before the arms run and stops",
        "the run if any of them has drifted.",
        "",
        "| file | sha256 |",
        "|---|---|",
    ]
    for name, digest in files.items():
        lines.append(f"| `{name}` | `{digest[:16]}...` |")
    lines += [
        "",
        "Full hashes are in `MANIFEST.json`.",
        "",
        "## The release input",
        "",
        f"* cell-level file: `{manifest['release_inputs']['h5ad']}`",
        f"* sha256: `{manifest['release_inputs']['h5ad_sha256']}`",
        f"* raw-matrix data root: `{manifest['release_inputs']['data_root']}`",
        "",
        "## What is new",
        "",
        "| script | role |",
        "|---|---|",
        "| `script/make_configs.py` | derives the four arm configs from the vendored base and asserts the diff is exactly the declared key set |",
        "| `script/prepare.py` | wires the vendored core's roots and seeds its FIMO cache |",
        "| `script/release_audit.py` | verifies the vendor hashes, the window set and the relabelling premise before anything runs |",
        "| `script/s03_export_chromvar_inputs_relabelled.py` | the vendored chromVAR exporter with the cell-type relabelling applied |",
        "| `run_all.sh` | the driver |",
        "",
        "## What this release changes scientifically",
        "",
        "One thing: the human early_embryonic window leaves the Astroglia arm.",
        "Those cells are `progenitor` in the cell-level file, so a model defined on",
        "them is no longer a model of astroglia, and the corresponding ranking file",
        "is excluded from `artifacts/ranked_astroglia_v4/`. The raw matrices still",
        "label them `astroglia`; the exporter relabels them before building",
        "pseudobulks so that the two layers cannot disagree. The Purkinje and Oligo",
        "arms are unaffected, and are therefore a control on the rewrite.",
        "",
    ]
    text = "\n".join(lines)
    (PKG / "PROVENANCE.md").write_text(text, encoding="utf-8")
    print(f"[WROTE] {PKG / 'PROVENANCE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
