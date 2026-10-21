"""Step 3: export the chromVAR inputs as species-level pseudobulk count matrices.

Why this shape (readme 15 / 16):

  * chromVAR must see RAW integer peak counts over EVERY QC peak of a species,
    never the normalised 10k model matrix and never a subset of the ranked
    peaks. The feature universe is therefore the full `var` of
    data/all_species_h5ad/<species>_deepcerevo_2026.h5ad, and the model ranking
    plays no role here at all.
  * running chromVAR on 10^5 individual cells would turn every downstream
    statistic into a cell-level pseudoreplicate, so counts are summed inside
    sample x Org_stage x Org_Celltype pseudobulk units. Summing raw integers is
    exact, so the unit stays a faithful aggregate of the cells it contains while
    each unit contributes exactly one observation.
  * the per-period h5ad files only supply membership, and even that is derived
    from obs['Org_stage'] through the period table the run declares
    (`paths.stage_map`: config/stage_map_v2.tsv under the v2 config,
    config/stage_map.tsv under v1) rather than copied, so the two sources are
    compared (stage_split_consistency.tsv) instead of silently mixed. The
    directory that comparison reads is `paths.stage_split_dir` (v2:
    data/stage_split_h5ad/full, v1: data/stage_split_h5ad) and is therefore
    read from the config, never hardcoded; the comparison itself is per
    cross_window -- per v2 period -- not per file name. A directory that holds
    no h5ad file is reported as UNVERIFIED, never as an empty PASS.

The export is a pure function of the h5ad and the frozen config: nothing here is
read back from results/, so the matrix cannot be tuned after the fact.

Per species the script writes counts.mtx.gz, peaks.tsv.gz, pseudobulk_metadata.tsv,
period_membership.tsv.gz, stage_split_consistency.tsv and export_audit.tsv, plus
dropped_units.tsv so that every dropped pseudobulk unit (not just the totals) has
an audit row. After writing it re-reads the matrix and recomputes a couple of
units straight from the h5ad; a non-zero difference is fatal.
"""

from __future__ import annotations

import argparse
import gzip
import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread, mmwrite

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

STEP = "03_chromvar_input"
AUDIT_COLUMNS = [
    "species",
    "n_cells_in",
    "n_cells_used",
    "n_dropped_small_cells",
    "n_peaks",
    "n_pseudobulk",
    "n_dropped_units",
    "total_counts",
    "matrix_nnz",
    "seconds",
    "verify_units_checked",
    "verify_max_abs_diff",
]
CONSISTENCY_COLUMNS = [
    "stage_split_file",
    "species",
    "cross_window",
    "n_in_stage_split",
    "n_assigned_here",
    "n_in_both",
    "n_only_stage_split",
    "n_only_assigned",
    "status",
]

# Status values of stage_split_consistency.tsv that all mean "this period was
# NOT verified". None of them may be read as a PASS, which is why an empty or
# missing reference directory can never produce an empty (vacuously green) table.
STATUS_PASS = "PASS"
STATUS_MISMATCH = "MISMATCH"
STATUS_UNMAPPED_FILE = "unmapped_stage_split_file"
STATUS_MISSING_FILE = "missing_stage_split_file"
STATUS_NO_SPECIES_FILE = "no_stage_split_file_for_species"
STATUS_NO_WINDOW = "no_cross_window_declared_for_species"
STATUS_EMPTY_DIR = "configured_dir_has_no_h5ad_files"
STATUS_MISSING_DIR = "configured_stage_split_dir_missing"


# --------------------------------------------------------------------------
# deterministic gzip writers
# --------------------------------------------------------------------------


def gzip_text(path: Path) -> io.TextIOWrapper:
    """Text handle over gzip with mtime pinned to 0 so reruns are byte-identical."""
    handle = gzip.GzipFile(filename=str(path), mode="wb", compresslevel=6, mtime=0)
    return io.TextIOWrapper(handle, encoding="utf-8", newline="\n")


def gzip_binary(path: Path) -> gzip.GzipFile:
    return gzip.GzipFile(filename=str(path), mode="wb", compresslevel=6, mtime=0)


def write_peak_table(peaks: np.ndarray, path: Path) -> None:
    frame = pd.DataFrame({"peak": peaks})
    with gzip_text(path) as handle:
        frame.to_csv(handle, sep="\t", index=False)
    C.log("wrote", path, f"({len(frame)} peaks)")


def write_matrix_market(matrix: sparse.spmatrix, path: Path) -> None:
    if matrix.dtype.kind not in "iu":
        raise TypeError("counts.mtx.gz must hold integer values")
    with gzip_binary(path) as handle:
        mmwrite(handle, matrix.astype(np.int64), symmetry="general", field="integer")
    C.log("wrote", path, f"({matrix.shape[0]}x{matrix.shape[1]}, {matrix.nnz} nnz)")


# --------------------------------------------------------------------------
# pseudobulk units
# --------------------------------------------------------------------------


@dataclass
class PseudobulkUnits:
    """Kept units in column order, plus the cell -> column assignment."""

    frame: pd.DataFrame
    unit_of_cell: np.ndarray
    dropped: pd.DataFrame


def build_units(obs: pd.DataFrame, config: dict) -> PseudobulkUnits:
    """sample x Org_stage x Org_Celltype units with at least min_cells cells.

    Units below the threshold are dropped, never merged and never rescued; every
    one of their cells is accounted for in `dropped` so the loss is auditable.
    Order is the lexicographic order of (sample, stage, celltype), so the column
    order of counts.mtx.gz is reproducible.
    """
    min_cells = int(config["chromvar"]["min_cells_per_pseudobulk"])
    key = pd.DataFrame(
        {
            "sample": [C.sample_id_from_cell_id(cell_id) for cell_id in obs.index],
            "stage": obs["Org_stage"].to_numpy(),
            "celltype": obs["Org_Celltype"].to_numpy(),
        }
    )
    sizes = key.groupby(["sample", "stage", "celltype"], sort=True).size()
    kept = sizes[sizes >= min_cells]
    dropped = sizes[sizes < min_cells]

    order = list(kept.index)
    lookup = {tuple(group): index for index, group in enumerate(order)}
    unit_of_cell = np.fromiter(
        (lookup.get(tuple(group), -1) for group in zip(key["sample"], key["stage"], key["celltype"])),
        dtype=np.int64,
        count=len(key),
    )

    frame = pd.DataFrame(
        {
            "sample": [group[0] for group in order],
            "stage": [group[1] for group in order],
            "celltype": [group[2] for group in order],
            "n_cells": kept.to_numpy(),
        }
    )
    frame.insert(0, "pseudobulk", frame["sample"] + "|" + frame["stage"] + "|" + frame["celltype"])

    dropped_frame = pd.DataFrame(
        {
            "sample": [group[0] for group in dropped.index],
            "stage": [group[1] for group in dropped.index],
            "celltype": [group[2] for group in dropped.index],
            "n_cells": dropped.to_numpy(),
        }
    )
    dropped_frame.insert(
        0,
        "pseudobulk",
        dropped_frame["sample"] + "|" + dropped_frame["stage"] + "|" + dropped_frame["celltype"],
    )
    dropped_frame["reason"] = f"n_cells < chromvar.min_cells_per_pseudobulk ({min_cells})"
    return PseudobulkUnits(frame=frame, unit_of_cell=unit_of_cell, dropped=dropped_frame)


def stream_unit_sums(
    source: Path,
    unit_of_cell: np.ndarray,
    n_units: int,
    n_peaks: int,
    block_cells: int,
) -> np.ndarray:
    """Raw integer sums of X per pseudobulk, one block of cells at a time.

    The accumulator is peak x pseudobulk (Fortran order, so a column write is
    contiguous) and is the only dense object in the script; the cell x peak
    matrix itself is never densified. Every cell of the h5ad is visited exactly
    once, cells mapped to -1 fall into no column.
    """
    accumulator = np.zeros((n_peaks, n_units), dtype=np.int64, order="F")
    n_blocks = 0
    for start, block, n_var in C.iter_matrix_blocks(source, block_cells=block_cells):
        if n_var != n_peaks:
            raise RuntimeError(f"{source}: block reports {n_var} peaks, expected {n_peaks}")
        if block.data.size and int(block.data.min()) < 0:
            raise RuntimeError(f"{source}: negative values in X; not raw counts")
        units = unit_of_cell[start: start + block.shape[0]]
        for unit in np.unique(units):
            if unit < 0:
                continue
            rows = np.flatnonzero(units == unit)
            column = np.asarray(block[rows].sum(axis=0)).ravel()
            accumulator[:, int(unit)] += column.astype(np.int64, copy=False)
        n_blocks += 1
        if n_blocks % 20 == 0:
            C.log("  streamed", start + block.shape[0], "/", len(unit_of_cell), "cells")
    return accumulator


# --------------------------------------------------------------------------
# membership / consistency
# --------------------------------------------------------------------------


def configured_stage_split_dir(config: dict) -> Path:
    """The directory the consistency audit reads, as declared in the config.

    v2 sets `paths.stage_split_dir: data/stage_split_h5ad/full` (the directory
    was named `stage_split_h5ad_v2` before the rename); v1 leaves the key out
    and keeps auditing `data/stage_split_h5ad`. Reading the key here (and
    resolving it against the project root exactly like every other `paths:` entry)
    is what lets the same script audit either period definition, and it is why
    the directory is never spelled out in this file.
    """
    if (config.get("paths") or {}).get("stage_split_dir"):
        return C.config_path(config, "stage_split_dir")
    return C.data_file(config, "stage_split_h5ad")


def cross_window_map(config: dict) -> dict[tuple[str, str], str]:
    """(species, raw Org_stage token) -> cross-species window.

    The period table is the one the config declares (`paths.stage_map`), so a v2
    run maps the v2 tokens and a v1 run still maps the v1 tokens; a token that
    the table does not mention yields "" and is reported as unmapped instead of
    being folded into some neighbouring window.
    """
    table = C.stage_table(config)
    source = C.display_path(C.stage_map_path(config))
    out: dict[tuple[str, str], str] = {}
    for row in table.itertuples():
        for token in str(row.raw_stages).split("|"):
            key = (row.species, token.strip())
            if out.get(key, row.cross_window) != row.cross_window:
                raise ValueError(f"stage {key} maps to two windows in {source}")
            out[key] = row.cross_window
    return out


def period_membership(
    obs: pd.DataFrame,
    species: str,
    windows: dict[tuple[str, str], str],
) -> pd.DataFrame:
    """cell_id -> stage -> window for every cell of the species h5ad."""
    stage = obs["Org_stage"].to_numpy()
    return pd.DataFrame(
        {
            "cell_id": obs.index.to_numpy(),
            "species": species,
            "sample": [C.sample_id_from_cell_id(cell_id) for cell_id in obs.index],
            "Org_stage": stage,
            "cross_window": [windows.get((species, token), "") for token in stage],
            "Org_Celltype": obs["Org_Celltype"].to_numpy(),
        }
    )


def declared_split_files(config: dict, species: str) -> tuple[list[str], dict[str, list[str]]]:
    """window -> the h5ad files the period table declares for it, for one species."""
    order: list[str] = []
    by_window: dict[str, list[str]] = {}
    for row in C.stage_table(config).itertuples():
        if str(row.species) != species:
            continue
        window = str(row.cross_window)
        if window not in by_window:
            by_window[window] = []
            order.append(window)
        by_window[window].append(Path(str(row.raw_h5ad)).name)
    return order, by_window


def _consistency_row(
    stage_split_file: str,
    species: str,
    window: str,
    n_in_stage_split: int,
    n_assigned_here: int,
    n_in_both: int,
    n_only_stage_split: int,
    n_only_assigned: int,
    status: str,
) -> dict:
    return dict(
        stage_split_file=stage_split_file,
        species=species,
        cross_window=window,
        n_in_stage_split=n_in_stage_split,
        n_assigned_here=n_assigned_here,
        n_in_both=n_in_both,
        n_only_stage_split=n_only_stage_split,
        n_only_assigned=n_only_assigned,
        status=status,
    )


def stage_split_consistency(
    config: dict,
    species: str,
    assigned: dict[str, set[str]],
    split_dir: Path | None = None,
) -> pd.DataFrame:
    """Compare the period files' cell IDs with the cells assigned here, per period.

    The reference files are only ever an audit reference: a mismatch is recorded
    and never used to change the assignment, which stays a pure function of
    obs['Org_stage'] through the configured period table.

    The comparison is per cross_window, i.e. per v2 period, never per file name:
    every file the period table declares for a window is unioned into that
    window's cell set, so a window that is split over several v1-era files (or
    merged into one v2 file) is still compared as one period. The directory that
    is read comes from `paths.stage_split_dir` and defaults to the v1 location.

    A window whose file is absent, and a directory that holds no h5ad at all, are
    reported with an explicit status -- the table is never silently empty, so an
    unreadable reference can not be mistaken for a clean PASS.
    """
    root = Path(split_dir) if split_dir is not None else configured_stage_split_dir(config)
    order, by_window = declared_split_files(config, species)
    known = {name for names in by_window.values() for name in names}
    if root.is_dir():
        all_h5ad = sorted(path.name for path in root.glob("*.h5ad"))
    else:
        all_h5ad = []
    present = [name for name in all_h5ad if name.startswith(f"{species}_")]

    rows: list[dict] = []
    unverified = None
    if not root.is_dir():
        unverified = STATUS_MISSING_DIR
    elif not all_h5ad:
        unverified = STATUS_EMPTY_DIR
    elif not order:
        # the period table defines no window for this species (v2 defines the
        # three developmental species only), so nothing can be verified here
        unverified = STATUS_NO_WINDOW

    if unverified is not None:
        C.log(
            "WARNING",
            species,
            "stage-split consistency NOT verified:",
            unverified,
            "->",
            C.display_path(root),
            "(paths.stage_split_dir; under v1 the h5ad files now live in its bak/ subdirectory)",
        )
        # One row per period this species actually declares, each carrying the
        # reason it could not be checked. A species whose period table entry is
        # empty (bonobo, macaque and opossum under both period definitions)
        # declares no period at all, so there is nothing to enumerate and the
        # table stays header-only -- exactly what the frozen v1 tables hold. The
        # exclusion is still reported: the WARNING above names the species, the
        # reason and the directory that was read.
        for window in order:
            here = len(assigned.get(window, ()))
            rows.append(
                _consistency_row(
                    ";".join(sorted(by_window[window])),
                    species,
                    window,
                    0,
                    here,
                    0,
                    0,
                    here,
                    unverified,
                )
            )
    elif not present:
        # The directory holds h5ad files, just none for this species: the audit
        # can not say anything about these periods either.
        C.log(
            "WARNING",
            species,
            "stage-split consistency NOT verified:",
            STATUS_NO_SPECIES_FILE,
            "->",
            C.display_path(root),
        )
        rows.append(
            _consistency_row(C.display_path(root), species, "", 0, 0, 0, 0, 0, STATUS_NO_SPECIES_FILE)
        )
    else:
        for window in order:
            names = sorted(name for name in by_window[window] if name in present)
            expected = ";".join(sorted(by_window[window]))
            here = assigned.get(window, set())
            if not names:
                rows.append(
                    _consistency_row(expected, species, window, 0, len(here), 0, 0, len(here), STATUS_MISSING_FILE)
                )
                continue
            split_cells: set[str] = set()
            for name in names:
                split_cells |= set(C.obs_frame(root / name)[0].index)
            both = split_cells & here
            status = STATUS_PASS if split_cells == here else STATUS_MISMATCH
            if names != sorted(by_window[window]):
                # part of this period was compared, part of it is missing
                status = STATUS_MISSING_FILE
            rows.append(
                _consistency_row(
                    ";".join(names),
                    species,
                    window,
                    len(split_cells),
                    len(here),
                    len(both),
                    len(split_cells - here),
                    len(here - split_cells),
                    status,
                )
            )

    # A file the period table does not claim is never folded into a window; it is
    # reported on its own row so the exclusion is visible.
    for name in present:
        if name in known:
            continue
        split_cells = set(C.obs_frame(root / name)[0].index)
        rows.append(
            _consistency_row(name, species, "", len(split_cells), 0, 0, len(split_cells), 0, STATUS_UNMAPPED_FILE)
        )

    frame = pd.DataFrame(rows, columns=CONSISTENCY_COLUMNS)
    # File-name order, the order the v1 table was published in.
    return frame.sort_values("stage_split_file", kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------


def direct_unit_sum(source: Path, rows: np.ndarray, n_peaks: int) -> np.ndarray:
    """Recompute one pseudobulk straight from the h5ad rows of its cells."""
    import h5py

    with h5py.File(source, "r") as handle:
        group = handle["X"]
        indptr = group["indptr"]
        data = group["data"]
        indices = group["indices"]
        blocks = []
        for row in rows:
            low = int(indptr[row])
            high = int(indptr[row + 1])
            blocks.append((data[low:high], indices[low:high]))
    pointer = np.zeros(len(rows) + 1, dtype=np.int64)
    for index, (values, _) in enumerate(blocks):
        pointer[index + 1] = pointer[index] + len(values)
    matrix = sparse.csr_matrix(
        (
            np.concatenate([values for values, _ in blocks]),
            np.concatenate([index for _, index in blocks]),
            pointer,
        ),
        shape=(len(rows), n_peaks),
    )
    return np.asarray(matrix.sum(axis=0)).ravel().astype(np.int64)


def verify_export(
    source: Path,
    mtx_path: Path,
    peaks: np.ndarray,
    units: PseudobulkUnits,
    unit_of_cell: np.ndarray,
    library_total: np.ndarray,
    seed: int,
    max_cells: int = 500,
) -> tuple[int, int]:
    """Read the written matrix back and recompute a couple of units directly.

    Returns (n_units_checked, max_abs_difference). A non-zero difference means
    the streamed accumulation and the file on disk disagree, which is fatal.
    """
    with gzip.GzipFile(filename=str(mtx_path), mode="rb") as handle:
        matrix = mmread(handle)
    matrix = sparse.csc_matrix(matrix)
    if matrix.shape != (len(peaks), len(units.frame)):
        raise RuntimeError(f"{mtx_path}: shape {matrix.shape} != ({len(peaks)}, {len(units.frame)})")
    column_sums = np.asarray(matrix.sum(axis=0)).ravel().astype(np.int64)
    if not np.array_equal(column_sums, library_total):
        raise RuntimeError(f"{mtx_path}: mtx column totals disagree with library_total")

    sizes = units.frame["n_cells"].to_numpy()
    picks = [int(np.argmin(sizes))]
    eligible = np.flatnonzero(sizes <= max_cells)
    rng = np.random.default_rng([seed, 3])
    alternatives = eligible[eligible != picks[0]]
    if len(alternatives):
        picks.append(int(rng.choice(alternatives)))

    checked = 0
    worst = 0
    for unit in picks:
        rows = np.flatnonzero(unit_of_cell == unit)
        recomputed = direct_unit_sum(source, rows, len(peaks))
        stored = matrix[:, unit].toarray().ravel().astype(np.int64)
        worst = max(worst, int(np.abs(recomputed - stored).max()))
        checked += 1
        C.log(
            "  verified unit",
            units.frame.pseudobulk.iloc[unit],
            f"({len(rows)} cells, n_peaks with counts={int((recomputed > 0).sum())})",
        )
    return checked, worst


# --------------------------------------------------------------------------
# per-species export
# --------------------------------------------------------------------------


def export_species(
    config: dict,
    species: str,
    h5ad_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    split_dir: str | Path | None = None,
    block_cells: int = 2000,
    verify: bool = True,
) -> dict:
    started = time.time()
    source = (
        Path(h5ad_path)
        if h5ad_path is not None
        else C.data_file(config, f"all_species_h5ad/{species}_deepcerevo_2026.h5ad")
    )
    out = Path(out_dir) if out_dir is not None else C.results_dir(config, STEP) / species
    out.mkdir(parents=True, exist_ok=True)

    obs, _ = C.obs_frame(source)
    peaks = C.var_index(source)
    units = build_units(obs, config)
    C.log(
        species,
        f"cells={len(obs)} peaks={len(peaks)} pseudobulks={len(units.frame)}",
        f"dropped_units={len(units.dropped)} dropped_cells={int(units.dropped.n_cells.sum())}",
    )

    accumulator = stream_unit_sums(source, units.unit_of_cell, len(units.frame), len(peaks), block_cells)
    library_total = np.asarray(accumulator.sum(axis=0)).ravel().astype(np.int64)
    matrix = sparse.csc_matrix(accumulator)
    del accumulator

    write_matrix_market(matrix, out / "counts.mtx.gz")
    write_peak_table(peaks, out / "peaks.tsv.gz")

    metadata = units.frame.copy()
    metadata["library_total"] = library_total
    C.save_table(
        metadata[["pseudobulk", "sample", "stage", "celltype", "n_cells", "library_total"]],
        out / "pseudobulk_metadata.tsv",
        sep="\t",
    )

    windows = cross_window_map(config)
    membership = period_membership(obs, species, windows)
    with gzip_text(out / "period_membership.tsv.gz") as handle:
        membership.to_csv(handle, sep="\t", index=False)
    C.log("wrote", out / "period_membership.tsv.gz", f"({len(membership)} cells)")

    # Every cell stays in the table, but a stage token the period table does not
    # define (v2 defines the three developmental species only; macaque, bonobo
    # and opossum have no window there) is named out loud rather than left as an
    # empty cross_window nobody notices.
    unmapped = membership[membership.cross_window == ""]
    if len(unmapped):
        counts = unmapped.Org_stage.astype(str).value_counts().sort_index()
        C.log(
            "WARNING",
            species,
            f"{len(unmapped)}/{len(membership)} cells have no cross_window in",
            C.display_path(C.stage_map_path(config)),
            "->",
            ", ".join(f"{stage} x{n}" for stage, n in counts.items()),
        )

    C.save_table(units.dropped, out / "dropped_units.tsv", sep="\t")

    assigned: dict[str, set[str]] = {}
    for window, group in membership.groupby("cross_window"):
        assigned[str(window)] = set(group.cell_id)
    consistency = stage_split_consistency(config, species, assigned, split_dir)
    C.save_table(consistency, out / "stage_split_consistency.tsv", sep="\t")

    checked, worst = (0, 0)
    if verify:
        checked, worst = verify_export(
            source, out / "counts.mtx.gz", peaks, units, units.unit_of_cell, library_total, C.seed(config)
        )
        if worst != 0:
            raise RuntimeError(f"{species}: written matrix disagrees with a direct recomputation ({worst})")

    audit = {
        "species": species,
        "n_cells_in": int(len(obs)),
        "n_cells_used": int(len(obs) - units.dropped.n_cells.sum()),
        "n_dropped_small_cells": int(units.dropped.n_cells.sum()),
        "n_peaks": int(len(peaks)),
        "n_pseudobulk": int(len(units.frame)),
        "n_dropped_units": int(len(units.dropped)),
        "total_counts": int(library_total.sum()),
        "matrix_nnz": int(matrix.nnz),
        "seconds": round(time.time() - started, 1),
        "verify_units_checked": int(checked),
        "verify_max_abs_diff": int(worst),
    }
    C.save_table(pd.DataFrame([audit])[AUDIT_COLUMNS], out / "export_audit.tsv", sep="\t")
    return audit


def write_root_audit(root: Path) -> pd.DataFrame:
    """Concatenate the per-species audits so `one row per species` also holds
    for a single table; rebuilt from disk, so --species runs stay consistent."""
    frames = [pd.read_csv(path, sep="\t") for path in sorted(root.glob("*/export_audit.tsv"))]
    if not frames:
        raise RuntimeError(f"no per-species export_audit.tsv under {root}")
    combined = pd.concat(frames, ignore_index=True)
    combined["_order"] = combined.species.map({s: i for i, s in enumerate(C.SPECIES_ORDER)})
    combined = combined.sort_values(["_order", "species"]).drop(columns="_order").reset_index(drop=True)
    return C.save_table(combined[AUDIT_COLUMNS], root / "export_audit.tsv", sep="\t")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    C.add_common_args(parser)
    parser.add_argument("--species", nargs="*", default=None, help="limit the run to these species")
    parser.add_argument(
        "--block-cells", type=int, default=2000, help="cells streamed from X per iteration"
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="skip the read-back check; the verify_* audit columns then stay 0",
    )
    args = parser.parse_args()

    C.main_guard()
    config = C.load_config(args.config)
    root = C.results_dir(config, STEP)
    for species in args.species or C.SPECIES_ORDER:
        with C.StepTimer(f"export {species}"):
            export_species(
                config,
                species,
                block_cells=args.block_cells,
                verify=not args.skip_verify,
            )
    write_root_audit(root)


if __name__ == "__main__":
    main()
