#!/usr/bin/env python3
"""Make the vendored core runnable: wire its data root, seed the FIMO cache.

The vendored `common.py` resolves every relative path against
`ROOT = <its own directory>/..`, i.e. `vendor/`, and reads `data/...` and
`config/...` from there. Rather than edit that file -- which would break the
byte-for-byte provenance that makes it trustworthy -- this script supplies the
one root it needs:

    vendor/data     -> the peak-regulatory project's data directory
                       (genomes, motifs, all_species_h5ad, frozen ranked dirs)
    vendor/config   -> already inside vendor/; nothing to do

`vendor/result_v4/` for the results, NOT `vendor/results/`. Two vendored guards
constrain that location from opposite sides -- `s06_purkinje_model.py` requires
its output inside the project root, `s17_motif_forest.py` and
`s18_sequence_maps.py` refuse anything under `results/` -- and `result_v4/` is
the intersection. Both guards aborted this pipeline in turn before the layout
matched them; see `script/make_configs.py` for the full account.

Idempotent, and refuses to replace a real directory: only a missing path or an
existing symlink is touched.

The FIMO cache is COPIED rather than linked, because `s18_sequence_maps.py`
writes into its cache directory and the previous release's cache is read-only on
purpose. A copy keeps this run from touching that tree at all. The cache is
derived data -- `vendor/steps/warm_scan_cache.py` regenerates it from the h5ads,
the FASTA files and the motif database -- so copying is a time saver, not a
hidden dependency.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
VENDOR = PKG / "vendor"
REPO = PKG.parents[1]

ENV_PROJECT_ROOT = "SCATAC_PROJECT_ROOT"


def project_root() -> Path:
    """The `scATAC_science_result/` directory that holds `data/`."""
    env = os.environ.get(ENV_PROJECT_ROOT)
    if env:
        return Path(env).expanduser().resolve()
    # HERE = .../analysis_use_new_data_20260917/src/04_peak_regulatory/script
    #   parents: 0=04_peak_regulatory 1=src 2=analysis_use_new_data_20260917
    #            3=scATAC_science_result
    return (HERE.parents[5] / "analysis_data" / "Figure5").resolve()


def link(target: Path, link_path: Path, *, what: str) -> None:
    if link_path.is_symlink():
        if link_path.resolve() == target.resolve():
            print(f"[KEEP] {what}: {link_path.name} -> {target}")
            return
        link_path.unlink()
    elif link_path.exists():
        raise SystemExit(
            f"{what}: {link_path} exists and is not a symlink. Move it aside; "
            f"this script will not replace a real directory."
        )
    link_path.symlink_to(target)
    print(f"[LINK] {what}: {link_path.name} -> {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="the peak-regulatory project's data/ directory")
    parser.add_argument("--cache-source", type=Path, default=None,
                        help="an existing warmed FIMO cache to copy")
    parser.add_argument("--force-cache", action="store_true",
                        help="re-copy the cache even if one is already present")
    args = parser.parse_args()

    # default data root: the vendored base config's own data directory
    data_root = args.data_root
    if data_root is None:
        root = project_root()
        data_root = (root / "data")
    if not data_root.is_dir():
        raise SystemExit(
            f"missing data root: {data_root}\n"
            f"Pass --data-root, or set {ENV_PROJECT_ROOT} to the "
            f"scATAC_science_result directory that contains "
            f"cross_species_peak_regulatory_analysis/."
        )

    # 1. vendor/data
    link(data_root, VENDOR / "data", what="data root")

    # 2. the results tree, inside vendor/ but beside `results/`.
    #
    # Both vendored guards have to hold at once: s06 requires the output inside
    # the project root, s17/s18 refuse anything under `results/`. A symlink cannot
    # substitute -- s06 resolves, so a link pointing outside resolves back out --
    # so the directory is genuinely created here. The repository exposes it at its
    # own top level as `results/04_peak_regulatory`, which is where a reader looks.
    results = VENDOR / "result_v4"
    results.mkdir(parents=True, exist_ok=True)
    reader_view = REPO / "results" / "04_peak_regulatory"
    reader_view.parent.mkdir(parents=True, exist_ok=True)
    if reader_view.is_symlink():
        reader_view.unlink()
    elif reader_view.exists():
        raise SystemExit(
            f"{reader_view} exists and is not a symlink. Move it aside; this script "
            f"will not replace a real directory."
        )
    reader_view.symlink_to(results)
    print(f"[OK] results: {results}")
    print(f"     reader view: {reader_view}")

    if os.environ.get("ANALYSIS_RESULTS_DIR"):
        public_view = Path(os.environ["ANALYSIS_RESULTS_DIR"]).expanduser().resolve() / "Figure5" / "04_peak_regulatory"
        public_view.parent.mkdir(parents=True, exist_ok=True)
        link(results, public_view, what="regulatory output view")

    # 3. the Astroglia ranking set for this release
    #
    # The set is declared in a checked-in manifest and the symlinks are DERIVED
    # from it, rather than the directory being the source of truth. A manifest
    # shows up in a diff as "this file is now excluded and here is why"; a
    # directory of symlinks does not, and an absolute symlink does not travel to
    # another machine at all.
    manifest = PKG / "artifacts" / "ranked_astroglia_v4.tsv"
    if not manifest.exists():
        raise SystemExit(f"missing ranking-set manifest: {manifest}")
    ranking_dir = PKG / "artifacts" / "ranked_astroglia_v4"
    ranking_dir.mkdir(parents=True, exist_ok=True)
    source_dir = data_root / "stage_split_h5ad" / "peak_ranked"
    kept, excluded, missing_sources = [], [], []
    with manifest.open("r", encoding="utf-8") as handle:
        next(handle)  # header
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            name, decision = parts[0], parts[1]
            (kept if decision == "kept" else excluded).append(name)
            if decision != "kept":
                continue
            src = source_dir / name
            if not src.exists():
                missing_sources.append(name)
                continue
            link(src, ranking_dir / name, what=f"ranking {name}")
    for stale in ranking_dir.glob("*.csv"):
        if stale.name not in kept:
            stale.unlink()
            print(f"[DROP] ranking {stale.name} (not in the manifest)")
    if missing_sources:
        raise SystemExit(
            f"ranking file(s) listed as kept are absent from {source_dir}: "
            f"{missing_sources}"
        )
    print(f"[OK] Astroglia ranking set: {len(kept)} kept, {len(excluded)} excluded "
          f"({', '.join(excluded)})")

    # 4. the cache
    cache = results / "cache"
    required = []
    for species in ("human", "mouse", "marmoset", "macaque", "bonobo", "opossum"):
        required += [f"{species}_sequences.csv.gz",
                     f"{species}_motif_peak_hits.csv.gz",
                     f"{species}_scan.sha256"]
    if cache.is_dir() and not args.force_cache and all((cache / n).exists() for n in required):
        print(f"[KEEP] cache: {cache} ({len(list(cache.iterdir()))} entries)")
    else:
        source = args.cache_source
        if source is None:
            candidate = (data_root.parent / "result_v3" / "cache")
            source = candidate if candidate.is_dir() else None
        if source is None:
            print("[WARN] no cache source found; the arm run will have to scan "
                  "sequences from scratch. Warm it with:\n"
                  "       vendor/steps/warm_scan_cache.py --config "
                  "config/arms/_shared.yaml --workers 6")
        else:
            if cache.exists():
                # only remove what this script put there
                for entry in cache.iterdir():
                    if entry.is_dir():
                        shutil.rmtree(entry, ignore_errors=True)
                    else:
                        entry.chmod(0o644)
                        entry.unlink()
            cache.mkdir(parents=True, exist_ok=True)
            print(f"[COPY] cache {source} -> {cache}")
            for entry in source.iterdir():
                target = cache / entry.name
                if entry.is_dir():
                    shutil.copytree(entry, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(entry, target)
                    target.chmod(0o644)
            missing = [n for n in required if not (cache / n).exists()]
            if missing:
                raise SystemExit(f"cache copy incomplete, missing: {missing}")
            print(f"[OK] cache: {len(list(cache.iterdir()))} entries")

    # ensure the cache is writable: s18 writes motif positions into it
    for path in cache.rglob("*"):
        try:
            path.chmod(path.stat().st_mode | 0o200)
        except OSError:
            pass

    # A source-only release has no generated vendor manifest yet. Freeze the
    # current packaged code once; never overwrite an existing integrity record.
    if not (VENDOR / "MANIFEST.json").exists():
        subprocess.run([
            sys.executable, str(HERE / "write_provenance.py"),
            "--h5ad", str(data_root / "integrated_cells.h5ad"),
            "--data-root", str(data_root),
        ], check=True)

    print(f"\n[DONE] vendored core is runnable:\n"
          f"  vendor/data    -> {data_root}\n"
          f"  vendor/results -> {results}\n"
          f"  cache          -> {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
