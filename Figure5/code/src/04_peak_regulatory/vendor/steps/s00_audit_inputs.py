"""Step 0: the hard input gate of the frozen cross-species peak analysis.

WHY this file exists
--------------------
Every later step (motif enrichment, chromVAR, the final claim table) trusts
three empirical facts that `readme.md` section 3 declares as hard gates and
that no config file can prove: `adata.X` really is the raw integer count
matrix, every peak coordinate really resolves against the declared reference
FASTA, and the metadata columns the analyses select on really are present and
complete. Getting any of those wrong silently invalidates every downstream
number, so this step measures them from the files themselves (never from the
comments in the config), writes one audit row per species x gate, and exits
non-zero while any gate is FAILing.

No sample may be dropped silently: a missing file, an unparsable coordinate, a
peak that cannot be extracted or a cell without a developmental window all end
up as an explicit row in an audit table under the run's audit directory.

WHICH period definition this run audits
---------------------------------------
The period table is whatever the config declares -- `common.stage_map_path` /
`common.stage_table` resolve `paths.stage_map` (v1: `config/stage_map.tsv`,
v2: `config/stage_map_v2.tsv`), so the `Org_stage -> cross_window` coverage
check, the per-period split files and the audit text all follow the config
instead of a path frozen in this file. A config that also declares a
`species_filter` block (v2 does; v1 has none) is audited for it as well: one
row per (analysis, species) carrying the target-celltype count and the
keep/drop verdict, plus the `SPECIES_FILTER_OK` gate. That gate is emitted only
when such a block exists, so a run without one is audited exactly as before.
"""

from __future__ import annotations

import argparse
import re
import sys
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

TARGET_CELLTYPES = ("astroglia", "Purkinje", "oligo", "GC")
GATES = ("RAW_COUNTS_OK", "PEAK_COORDINATES_OK", "GENOME_MATCH_OK", "METADATA_OK")

# v2 declares `species_filter`: a species enters a cell-type analysis only above
# a floor of cells of that type. The gate below is emitted only when the config
# declares such a block, so a config without one (v1) keeps the four gates and
# every byte of its published output.
SPECIES_FILTER_GATE = "SPECIES_FILTER_OK"
SPECIES_FILTER_COLUMNS = (
    "analysis",
    "species",
    "celltype",
    "n_cells",
    "min_cells",
    "n_ranked",
    "min_ranked_peaks",
    "measurable",
    "passed",
    "action",
    "n_models",
)

SPECIES_AUDIT_COLUMNS = (
    "species",
    "n_cells",
    "n_peaks",
    "raw_count_location",
    "raw_integer_fraction",
    "celltype_column",
    "stage_column",
    "sample_column",
    "donor_column",
    "peak_coordinate_format",
    "genome_assembly",
    "fasta_path",
    "seqname_overlap",
    "n_peaks_extractable",
    "extractable_fraction",
    "n_peaks_valid_acgt",
    "raw_count_min",
    "raw_count_max",
    "raw_count_is_int",
)

# thresholds frozen by readme section 3 / IMPLEMENTATION_SPEC.md
MIN_INTEGER_FRACTION = 0.999
MIN_EXTRACTABLE_FRACTION = 0.99
EXTRACTION_SAMPLE = 2000
COORD_CHECK_SAMPLE = 3000
VAR_COLUMN_AGREEMENT_MIN = 0.99

SAMPLE_COLUMN_PREFERENCE = ("sample", "LibraryID", "LibraryID_long")
DONOR_COLUMN_PREFERENCE = ("orig.ident",)
CELL_ID_PREFIX = "cell_id_prefix"

# the peak-id prefix is the token before the first "_" in this atlas' naming
PREFIX_RE = re.compile(r"^([A-Za-z0-9.]+?_)")
BODY_RE = re.compile(r"^([^:]+):(\d+)-(\d+)$")


# --------------------------------------------------------------------------
# small file probes
# --------------------------------------------------------------------------


@dataclass
class XStats:
    """What `adata.X` (and, if X is unusable, the layers) actually contains."""

    shape: tuple[int, int]
    sparse: bool
    dtype: str
    integer_dtype: bool
    n_values: int
    minimum: float | None
    maximum: float | None
    fraction_integer: float
    layer_names: list[str] = field(default_factory=list)
    layer_candidate: str = ""


def _accumulate(stats: dict, values: np.ndarray) -> None:
    if values.size == 0:
        return
    stats["n_values"] += int(values.size)
    low = float(np.min(values))
    high = float(np.max(values))
    stats["minimum"] = low if stats["minimum"] is None else min(stats["minimum"], low)
    stats["maximum"] = high if stats["maximum"] is None else max(stats["maximum"], high)
    if values.dtype.kind not in "iu":
        stats["n_non_integer"] += int(np.count_nonzero(values != np.floor(values)))


def stream_x_stats(path: Path) -> XStats:
    """Min/max/integer-fraction of the raw X, streaming block by block.

    The whole matrix is never materialised: `common.iter_matrix_blocks` hands
    out one cell block at a time, and a dense X falls back to row blocks.
    """
    with h5py.File(path, "r") as handle:
        group = handle["X"]
        sparse = isinstance(group, h5py.Group)
        # a CSR group carries its shape as an attribute; a dense X carries the
        # standard dataset shape, so both layouts must be read before use
        shape = (
            tuple(int(v) for v in group.attrs["shape"])
            if sparse
            else tuple(int(v) for v in group.shape)
        )
        layer_names = list(handle["layers"].keys()) if "layers" in handle else []

    stats: dict = {"n_values": 0, "minimum": None, "maximum": None, "n_non_integer": 0}
    dtype = ""
    if sparse:
        for _, block, _ in C.iter_matrix_blocks(path):
            dtype = dtype or str(block.data.dtype)
            _accumulate(stats, block.data)
    else:
        with h5py.File(path, "r") as handle:
            dataset = handle["X"]
            dtype = str(dataset.dtype)
            rows = max(1, int(2e7 // max(shape[1], 1)))
            for start in range(0, shape[0], rows):
                _accumulate(stats, np.asarray(dataset[start : start + rows, :]).ravel())

    integer_dtype = dtype.startswith("int") or dtype.startswith("uint")
    n_values = stats["n_values"]
    candidate = ""
    if integer_dtype is False and layer_names:
        candidate = _non_negative_layer(path, layer_names)
    return XStats(
        shape=shape,
        sparse=sparse,
        dtype=dtype,
        integer_dtype=integer_dtype,
        n_values=n_values,
        minimum=stats["minimum"],
        maximum=stats["maximum"],
        fraction_integer=1.0 if integer_dtype else 1.0 - stats["n_non_integer"] / max(n_values, 1),
        layer_names=layer_names,
        layer_candidate=candidate,
    )


def _non_negative_layer(path: Path, names: list[str]) -> str:
    """First layer whose leading block is non-negative numeric (raw-count candidate)."""
    for name in names:
        with h5py.File(path, "r") as handle:
            item = handle["layers"][name]
            if isinstance(item, h5py.Group):
                values = item["data"][:200_000]
            else:
                values = np.asarray(item[: min(2000, item.shape[0]), :]).ravel()
        if values.size and values.dtype.kind in "iuf" and float(np.min(values)) >= 0:
            return name
    return ""


def measure_layer(path: Path, name: str) -> XStats:
    """Min/max/integer-fraction of one layer, streamed like `stream_x_stats`.

    Needed when `X` is not a usable count matrix: the numbers reported next to
    `raw_count_location` must be measured on the array that name points at, not
    borrowed from the array that was rejected.
    """
    stats: dict = {"n_values": 0, "minimum": None, "maximum": None, "n_non_integer": 0}
    dtype = ""
    shape = (0, 0)
    sparse = False
    with h5py.File(path, "r") as handle:
        item = handle["layers"][name]
        sparse = isinstance(item, h5py.Group)
        if sparse:
            shape = tuple(int(v) for v in item.attrs["shape"])
            data = item["data"]
            dtype = str(data.dtype)
            for start in range(0, int(data.shape[0]), 5_000_000):
                _accumulate(stats, np.asarray(data[start : start + 5_000_000]))
        else:
            shape = tuple(int(v) for v in item.shape)
            dtype = str(item.dtype)
            rows = max(1, int(2e7 // max(shape[1], 1)))
            for start in range(0, shape[0], rows):
                _accumulate(stats, np.asarray(item[start : start + rows, :]).ravel())
    integer_dtype = dtype.startswith("int") or dtype.startswith("uint")
    n_values = stats["n_values"]
    return XStats(
        shape=shape,
        sparse=sparse,
        dtype=dtype,
        integer_dtype=integer_dtype,
        n_values=n_values,
        minimum=stats["minimum"],
        maximum=stats["maximum"],
        fraction_integer=1.0 if integer_dtype else 1.0 - stats["n_non_integer"] / max(n_values, 1),
    )


def resolve_count_source(path: Path, x_stats: XStats) -> tuple[str, XStats]:
    """Which array holds the raw counts, and the stats measured on that array.

    `X` is accepted only when it is numeric, non-negative AND integral, so an
    h5ad whose `X` was normalised in place -- exactly the mistake this gate
    exists to catch -- falls back to the first raw-count-looking layer and the
    reported min/max/integer fraction are re-measured on that layer instead of
    being borrowed from `X`. If no array qualifies the location is empty
    (reported as `unknown`) and `X`'s own numbers are returned, so the FAIL row
    always shows what was actually measured.
    """
    x_numeric = bool(x_stats.dtype) and np.issubdtype(np.dtype(x_stats.dtype), np.number)
    x_usable = (
        x_numeric
        and x_stats.n_values > 0
        and x_stats.minimum is not None
        and x_stats.minimum >= 0
    )
    if x_usable and x_stats.fraction_integer > MIN_INTEGER_FRACTION:
        return "X", x_stats
    if x_stats.layer_candidate:
        return f"layers['{x_stats.layer_candidate}']", measure_layer(path, x_stats.layer_candidate)
    if x_usable:
        return "X", x_stats
    return "", x_stats


def var_frame(path: Path) -> pd.DataFrame:
    """The peak index plus any `chrom`/`start`/`end` columns of `var`."""
    with h5py.File(path, "r") as handle:
        var = handle["var"]
        index_name = var.attrs["_index"]
        index_name = index_name.decode() if isinstance(index_name, bytes) else str(index_name)
        columns = {"peak": C.read_obs_column(var, index_name)}
        for name in ("chrom", "start", "end"):
            if name in var:
                columns[name] = np.asarray(C.read_obs_column(var, name))
    return pd.DataFrame(columns)


def peak_body(peak: str, prefix: str) -> tuple[str, int, int] | None:
    """(chrom as written, start, end) without the chrom_style rewrite of parse_peak."""
    body = peak[len(prefix):] if peak.startswith(prefix) else peak
    match = BODY_RE.match(body)
    if not match:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))


def observed_prefix(peaks: list[str], limit: int = 5000) -> str:
    sample = peaks[:: max(1, len(peaks) // limit)][:limit]
    counts = Counter()
    for peak in sample:
        match = PREFIX_RE.match(peak)
        counts[match.group(1) if match else "<unparsable>"] += 1
    if len(counts) != 1:
        return ";".join(f"{name}x{n}" for name, n in counts.most_common(4))
    return counts.most_common(1)[0][0]


def display_path(path: Path) -> str:
    """Project-relative when the file lives inside the project, absolute otherwise."""
    try:
        return str(path.relative_to(C.ROOT))
    except ValueError:
        return str(path)


def annotate_known_limitations(gates: pd.DataFrame) -> pd.DataFrame:
    """Mark gate failures that were pre-registered in config/known_limitations.tsv.

    A pre-registered failure is still reported as FAIL -- the audit never hides a
    gate -- but it is marked non-blocking so that a limitation of the *input
    data* (for example a reference genome that is missing two contigs) does not
    stop an analysis that is otherwise complete. The reason and the mitigation
    travel with the row, so a reader can always see what was given up.
    """
    gates = gates.copy()
    gates["blocking"] = gates.status == "FAIL"
    gates["exception_reason"] = ""
    gates["exception_mitigation"] = ""

    path = C.CONFIG_DIR / "known_limitations.tsv"
    if not path.exists():
        return gates
    known = pd.read_csv(path, sep="\t")
    for row in known.itertuples():
        match = (gates.species == row.species) & (gates.gate == row.gate) & (gates.status == "FAIL")
        if not match.any():
            C.log("WARNING known limitation declared but not triggered:", row.species, row.gate)
            continue
        gates.loc[match, "blocking"] = False
        gates.loc[match, "exception_reason"] = row.reason
        gates.loc[match, "exception_mitigation"] = row.mitigation
    return gates


def save_audit_table(frame: pd.DataFrame, path: Path) -> None:
    """`common.save_table` for the real outputs, plain TSV for --out-dir runs.

    `common.save_table` logs a project-relative path, which only exists when the
    audit is written under `results/`; the test runs redirect it elsewhere.
    """
    if path.is_relative_to(C.ROOT):
        C.save_table(frame, path, sep="\t")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, sep="\t")
    C.log("wrote", str(path), f"({len(frame)} rows)")


def truncate(names: list[str], limit: int = 6) -> str:
    if len(names) <= limit:
        return "[" + ",".join(names) + "]"
    return "[" + ",".join(names[:limit]) + f",...(+{len(names) - limit})]"


# --------------------------------------------------------------------------
# stage window mapping
# --------------------------------------------------------------------------


def stage_windows(stage_map: pd.DataFrame) -> dict[str, dict[str, str]]:
    """{species: {raw Org_stage: cross_window}} from the config's period table."""
    windows: dict[str, dict[str, str]] = {}
    for row in stage_map.itertuples():
        mapping = windows.setdefault(str(row.species), {})
        for stage in str(row.raw_stages).split("|"):
            stage = stage.strip()
            if stage:
                mapping[stage] = str(row.cross_window)
    return windows


def coverage_rows(
    species: str,
    obs: pd.DataFrame,
    windows: dict[str, dict[str, str]],
) -> tuple[list[dict], int, list[str], bool]:
    """Per-Org_stage coverage of the cross_window mapping plus unmapped count."""
    declared = species in windows
    mapping = windows.get(species, {})
    stages = obs["Org_stage"].astype(str)
    rows: list[dict] = []
    for stage, count in stages.value_counts().items():
        rows.append(
            dict(
                species=species,
                org_stage=str(stage),
                cross_window=mapping.get(str(stage), ""),
                n_cells=int(count),
                mapped=bool(str(stage) in mapping),
                species_declared=declared,
            )
        )
    rows.sort(key=lambda row: (-row["n_cells"], row["org_stage"]))
    unmapped = int(sum(row["n_cells"] for row in rows if not row["mapped"]))
    unmapped_names = sorted(row["org_stage"] for row in rows if not row["mapped"])
    return rows, unmapped, unmapped_names, declared


def stage_split_dir(config: dict) -> Path:
    """Where the per-period split h5ads of this run live.

    v1 leaves `paths.stage_split_dir` out and keeps `data/stage_split_h5ad`; v2
    declares the developmental split (`data/stage_split_h5ad/full`, the
    full-peak space whose cells are the same slice as the model space). This
    release ships the v2 definition under that directory (renamed from
    `stage_split_h5ad_v2`), so the undeclared-key fallback resolves to the
    parent of `full/`. The files are an audit reference only -- which cells
    belong to a period -- so a config that omits the key keeps that directory
    instead of failing.
    """
    declared = (config.get("paths") or {}).get("stage_split_dir")
    if declared:
        path = Path(declared)
        return path if path.is_absolute() else C.ROOT / path
    return C.data_file(config, "stage_split_h5ad")


def declared_split_files(stage_map: pd.DataFrame, species: str) -> list[str]:
    """The split h5ad file names the period table declares for one species.

    The table's `raw_h5ad` column is the run's own statement of which split file
    each of its periods was written to, so it -- never a glob of the directory --
    is what says which files must be there. A species the table does not declare
    declares none, and a table without the column declares none either.
    """
    if "raw_h5ad" not in stage_map.columns:
        return []
    names: list[str] = []
    for entry in stage_map.itertuples():
        if str(entry.species) != species:
            continue
        raw = getattr(entry, "raw_h5ad", "")
        if not isinstance(raw, str) or not raw.strip():
            continue  # an unfilled cell declares no file, and is not one named "nan"
        name = Path(raw.strip()).name
        if name not in names:
            names.append(name)
    return names


# --------------------------------------------------------------------------
# species filter (only when the config declares one)
# --------------------------------------------------------------------------


def species_filter_active(config: dict) -> bool:
    """True when this config asks for a per-celltype species floor."""
    # v2 declares `eligibility`; v1 declared `species_filter`. Reading only one
    # of the two silently disables the filter -- and worse, leaves a stale table
    # from an earlier run in place, pretending to be current.
    return bool((config.get("eligibility") or config.get("species_filter")))


def species_filter_rows(
    config: dict,
    manifest: pd.DataFrame,
    species_scope: list[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """One audit row per (analysis, species) of the configured species filter.

    The candidate species of an analysis are the species that carry a ranked
    model for it -- exactly the rows `common.apply_species_filter` decides on --
    restricted to `species_scope` when the audit itself is scoped to a subset.
    The count comes from `common.species_cell_counts` and the keep/drop verdict
    from `common.species_passing_filter`, so this table cannot disagree with the
    filter the later steps apply.

    A pair whose target celltype is not a value of the configured celltype
    column is recorded as *unmeasured* with n_cells=0 rather than as a plain
    zero: "this vocabulary has no such celltype" is not the same fact as "this
    species has no cells of that type", and a silently zeroed count is exactly
    the failure this table exists to prevent.
    """
    column = ((config.get("eligibility") or config.get("species_filter")) or {}).get("celltype_column", "Org_Celltype")
    counts = C.species_cell_counts(config)
    lookup = counts.set_index(["species", "celltype"]).n_cells.to_dict()
    measured_species = set(counts.species.unique())
    vocabulary = set(counts.celltype.unique())
    floor = C.min_cells_per_species(config)
    peak_floor = C.min_ranked_peaks(config)
    scope = None if species_scope is None else set(species_scope)

    rows: list[dict] = []
    notes: list[str] = []
    for (analysis, celltype), group in manifest.groupby(
        ["analysis", "org_celltype"], sort=True
    ):
        analysis, celltype = str(analysis), str(celltype)
        present = set(group.species)
        pool = [species for species in C.SPECIES_ORDER if species in present]
        pool += sorted(species for species in present if species not in C.SPECIES_ORDER)
        if scope is not None:
            pool = [species for species in pool if species in scope]
        if not pool:
            continue
        kept, _dropped = C.species_passing_filter(config, celltype, pool)
        kept_set = set(kept)
        for species in pool:
            n_cells = int(lookup.get((species, celltype), 0))
            measured = species in measured_species and celltype in vocabulary
            if species not in measured_species:
                action = f"unmeasured: no integrated h5ad of {species} was readable"
                passed = False
            elif celltype not in vocabulary:
                action = f"unmeasured: '{celltype}' is not a value of {column} in any species"
                passed = False
            else:
                # An analysis admits a species on BOTH criteria: enough cells of
                # the target type, and a ranking long enough to define a Top-k
                # foreground. `common.apply_species_filter` enforces both, so this
                # table reports both -- a species passing the cell floor but
                # failing the peak floor must not read as "kept".
                n_ranked = int(group.loc[group.species == species, "n_ranked"].min())
                reasons = []
                if n_cells < floor:
                    reasons.append(f"{n_cells} target cells < {floor}")
                if n_ranked < peak_floor:
                    reasons.append(f"{n_ranked} ranked peaks < {peak_floor}")
                passed = (species in kept_set) and not reasons
                action = (
                    f"kept: {n_cells} >= {floor} and {n_ranked} >= {peak_floor}"
                    if passed
                    else "dropped: " + "; ".join(reasons)
                )
            rows.append(
                dict(
                    analysis=analysis,
                    species=species,
                    celltype=celltype,
                    n_cells=n_cells,
                    min_cells=floor,
                    n_ranked=(
                        int(group.loc[group.species == species, "n_ranked"].min())
                        if measured else -1
                    ),
                    min_ranked_peaks=peak_floor,
                    measurable=bool(measured),
                    passed=bool(passed),
                    action=action,
                    n_models=int((group.species == species).sum()),
                )
            )
            if measured and not passed:
                notes.append(
                    f"{species}: species filter drops {analysis} ({celltype}={n_cells} cells, "
                    f"ranked peaks={int(group.loc[group.species == species, 'n_ranked'].min())}; "
                    f"floors are cells>={floor}, peaks>={peak_floor})"
                )
    return rows, notes


def species_filter_gates(
    config: dict,
    manifest: pd.DataFrame,
    rows: list[dict],
    species_scope: list[str] | None = None,
) -> list[dict]:
    """The `SPECIES_FILTER_OK` gate: one row per species that has a candidate model.

    A drop is a data fact and never fails the gate. What fails is a filter whose
    decisions cannot be reconciled with what the later steps will actually do:
    a count that could not be measured (no readable integrated h5ad, or a
    target celltype that is not a value of the configured celltype column), or a
    set of dropped (analysis, species) pairs that differs from
    `common.apply_species_filter` on the same manifest. Both are defects in the
    filter definition, so they are reported as blocking.
    """
    column = ((config.get("eligibility") or config.get("species_filter")) or {}).get("celltype_column", "Org_Celltype")
    floor = C.min_cells_per_species(config)
    scope = None if species_scope is None else set(species_scope)

    _kept_manifest, dropped_models = C.apply_species_filter(config, manifest)
    applied: set[tuple[str, str]] = set()
    for row in dropped_models.itertuples():
        if scope is not None and str(row.species) not in scope:
            continue
        applied.add((str(row.analysis), str(row.species)))

    by_species: dict[str, list[dict]] = {}
    for row in rows:
        by_species.setdefault(row["species"], []).append(row)

    gates: list[dict] = []
    for species, species_rows in by_species.items():
        kept = [row for row in species_rows if row["passed"]]
        dropped = [row for row in species_rows if not row["passed"]]
        unmeasured = [row for row in species_rows if not row["measurable"]]
        if unmeasured:
            named = ", ".join(f"{row['analysis']} ({row['action']})" for row in unmeasured)
            detail = (
                f"min_cells_per_species={floor} ({column}); not measurable: {named}"
            )
            gates.append(_gate(species, SPECIES_FILTER_GATE, "FAIL", detail))
            continue
        recorded = {(row["analysis"], species) for row in dropped}
        expected = {pair for pair in applied if pair[1] == species}
        mismatch = sorted(recorded ^ expected)
        if mismatch:
            detail = (
                f"min_cells_per_species={floor} ({column}); recorded drops disagree with "
                f"common.apply_species_filter on "
                f"{truncate([f'{analysis}/{sp}' for analysis, sp in mismatch])}"
            )
            gates.append(_gate(species, SPECIES_FILTER_GATE, "FAIL", detail))
            continue
        kept_labels = [f"{row['analysis']}={row['n_cells']}" for row in kept]
        dropped_labels = [f"{row['analysis']}={row['n_cells']}" for row in dropped]
        detail = (
            f"min_cells_per_species={floor} ({column}); "
            f"n_models={sum(row['n_models'] for row in species_rows)}; "
            f"kept={truncate(kept_labels)} dropped={truncate(dropped_labels)}"
        )
        gates.append(_gate(species, SPECIES_FILTER_GATE, "PASS", detail))
    return gates


# --------------------------------------------------------------------------
# per-species audit
# --------------------------------------------------------------------------


@dataclass
class SpeciesResult:
    species: str
    row: dict
    gates: list[dict]
    coverage: list[dict]
    celltype_counts: list[dict]
    fasta_seqs: int
    notes: list[str]


def _gate(species: str, gate: str, status: str, detail: str) -> dict:
    return dict(species=species, gate=gate, status=status, detail=detail)


def _all_fail(species: str, detail: str) -> list[dict]:
    return [_gate(species, gate, "FAIL", detail) for gate in GATES]


def extraction_use_cache(config: dict, species: str, peaks: list[str]) -> tuple[bool, str]:
    """Use the shared sequence cache only when it already covers the sample.

    `common.read_sequences(use_cache=True)` rewrites the cache file when the
    request is not covered by it, and the background warm-up job
    (`script/warm_scan_cache.py`) writes that same file. Reading a covering
    cache is free and race-free; a partial write from here is what the ban on
    racing the warm-up job forbids, so the cache is only *used*, never grown.
    """
    cache = C.cache_dir(config) / f"{species}_sequences.csv.gz"
    if not cache.exists():
        return False, "cache absent"
    try:
        cached = pd.read_csv(cache, usecols=["peak"])
    except (OSError, ValueError, EOFError, zlib.error, pd.errors.ParserError):
        return False, "cache unreadable (concurrent write?)"
    missing = len(set(peaks) - set(cached.peak.astype(str)))
    if missing == 0:
        return True, f"cache covers the sample ({len(cached)} peaks cached)"
    return False, f"cache misses {missing}/{len(peaks)} sampled peaks"


def extract_sequences(
    config: dict, genome: C.SpeciesGenome, species: str, sampled: list[str]
) -> tuple[pd.DataFrame, str]:
    """Extract the sampled peaks, falling back to a cache-free read on error."""
    use_cache, note = extraction_use_cache(config, species, sampled)
    if use_cache:
        try:
            return C.read_sequences(config, genome, sampled, use_cache=True), note
        except (OSError, ValueError, EOFError, zlib.error, pd.errors.ParserError):
            note = "sequence cache unreadable (concurrent write?); re-extracted without it"
    return C.read_sequences(config, genome, sampled, use_cache=False), note


def audit_species(
    config: dict,
    genome: C.SpeciesGenome,
    species: str,
    h5ad_path: Path,
    stage_map: pd.DataFrame,
    ranked_peaks: list[str],
    extraction_sample: int = EXTRACTION_SAMPLE,
) -> SpeciesResult:
    row = {column: "" for column in SPECIES_AUDIT_COLUMNS}
    row["species"] = species
    row["genome_assembly"] = genome.assembly
    row["fasta_path"] = display_path(genome.fasta)
    row["peak_coordinate_format"] = genome.prefix
    notes: list[str] = []

    if not h5ad_path.exists():
        detail = f"h5ad missing: {h5ad_path}"
        return SpeciesResult(
            species, row, _all_fail(species, detail), [], [], 0, [detail]
        )
    if not genome.fasta.exists():
        detail = f"FASTA missing: {genome.fasta}"
        return SpeciesResult(
            species, row, _all_fail(species, detail), [], [], 0, [detail]
        )

    x_stats = stream_x_stats(h5ad_path)
    row["n_cells"] = x_stats.shape[0]

    var = var_frame(h5ad_path)
    peaks = [str(peak) for peak in var.peak]
    row["n_peaks"] = len(peaks)
    row["peak_coordinate_format"] = observed_prefix(peaks)

    obs, index_name = C.obs_frame(h5ad_path)
    cell_prefixes = {C.sample_id_from_cell_id(cell) for cell in map(str, obs.index)}

    # ---- RAW_COUNTS_OK ---------------------------------------------------
    location, counts = resolve_count_source(h5ad_path, x_stats)
    raw_ok = (
        bool(location)
        and counts.n_values > 0
        and counts.fraction_integer > MIN_INTEGER_FRACTION
    )
    row["raw_count_location"] = location or "unknown"
    row["raw_integer_fraction"] = f"{counts.fraction_integer:.6f}"
    row["raw_count_min"] = "" if counts.minimum is None else int(counts.minimum)
    row["raw_count_max"] = "" if counts.maximum is None else int(counts.maximum)
    row["raw_count_is_int"] = bool(counts.integer_dtype)
    layers = ",".join(x_stats.layer_names) if x_stats.layer_names else "none"
    raw_detail = (
        f"location={row['raw_count_location']}; layers={layers}; X={x_stats.dtype} "
        f"{'sparse' if x_stats.sparse else 'dense'}; n_values={counts.n_values}; "
        f"min={row['raw_count_min']}; max={row['raw_count_max']}; "
        f"integer_fraction={counts.fraction_integer:.6f} (min {MIN_INTEGER_FRACTION})"
    )
    if x_stats.layer_candidate:
        raw_detail += f"; count layer candidate=layers['{x_stats.layer_candidate}']"
    if counts is not x_stats:
        raw_detail += (
            f"; reported numbers measured on layers['{x_stats.layer_candidate}'] "
            f"({counts.dtype}); X rejected (min={x_stats.minimum}, "
            f"integer_fraction={x_stats.fraction_integer:.6f})"
        )

    # ---- PEAK_COORDINATES_OK --------------------------------------------
    parse_fail = 0
    chroms: set[str] = set()
    for peak in peaks:
        try:
            chrom, _, _ = C.parse_peak(peak, genome)
        except ValueError:
            parse_fail += 1
            continue
        chroms.add(chrom)

    body_fail = 0
    for peak in peaks:
        if genome.prefix and not peak.startswith(genome.prefix):
            body_fail += 1
    prefix_ok = 1.0 - body_fail / max(len(peaks), 1)

    rng = np.random.default_rng([C.seed(config), 0])
    coord_idx = rng.choice(len(peaks), size=min(COORD_CHECK_SAMPLE, len(peaks)), replace=False)
    agree = total = 0
    if {"chrom", "start", "end"} <= set(var.columns):
        for i in coord_idx:
            parsed = peak_body(peaks[i], genome.prefix)
            if parsed is None:
                continue
            total += 1
            # var.chrom carries the assembly prefix as well (`hg38_chr1`), so the
            # peak id and the var column are compared in the same namespace
            if (
                f"{genome.prefix}{parsed[0]}" == str(var.chrom.iloc[i])
                and parsed[1] == int(var.start.iloc[i])
                and parsed[2] == int(var.end.iloc[i])
            ):
                agree += 1
    var_agreement = agree / total if total else float("nan")

    ranked = list(dict.fromkeys(ranked_peaks))
    peak_set = set(peaks)
    ranked_missing = [peak for peak in ranked if peak not in peak_set]
    ranked_bad = 0
    for peak in ranked:
        try:
            C.parse_peak(peak, genome)
        except ValueError:
            ranked_bad += 1

    coords_ok = (
        len(peaks) > 0
        and parse_fail == 0
        and prefix_ok > MIN_EXTRACTABLE_FRACTION
        and ranked_bad == 0
        and len(ranked_missing) / max(len(ranked), 1) <= 0.01
        and (np.isnan(var_agreement) or var_agreement >= VAR_COLUMN_AGREEMENT_MIN)
    )
    coord_detail = (
        f"n_peaks={len(peaks)}; prefix_observed={row['peak_coordinate_format']} "
        f"(config {genome.prefix}); parse_fail={parse_fail}; prefix_match={prefix_ok:.4f}; "
        f"var_chrom/start/end agreement={agree}/{total}; "
        f"ranked_peaks={len(ranked)} not_in_var={len(ranked_missing)} parse_fail={ranked_bad}"
    )
    if ranked_missing:
        coord_detail += f"; first_missing={truncate(sorted(ranked_missing), 3)}"

    # ---- GENOME_MATCH_OK -------------------------------------------------
    fasta_names = C.fasta_sequence_names(genome.fasta)
    missing_seq = sorted(chroms - fasta_names)
    seqname_overlap = f"{len(chroms)}/{len(fasta_names)}; missing={truncate(missing_seq)}"

    usable = [peak for peak in peaks if peak_body(peak, genome.prefix) is not None]
    draw = rng.choice(len(usable), size=min(extraction_sample, len(usable)), replace=False)
    sampled = sorted(usable[i] for i in draw)
    if sampled:
        sequences, cache_note = extract_sequences(config, genome, species, sampled)
        extracted = int((sequences.sequence.astype(str).str.len() > 0).sum())
        valid = int(sequences.valid.sum())
    else:
        # no peak id is extractable at all: `common.read_sequences` answers an
        # empty request with a column-less frame, so it must not be called here
        extracted = valid = 0
        cache_note = "no extractable peak id (empty peak set or unparsable coordinates)"
    extractable_fraction = extracted / max(len(sampled), 1)

    # The gate is the spec's frozen quantity: the fraction of sampled peaks whose
    # reference sequence could be extracted at all. `valid_acgt` is reported raw
    # next to it -- the two diverge on a peak recovered inside an N/IUPAC run --
    # and the threshold stays anchored on extractable_fraction, which is how the
    # published v1 audit_gates.tsv words it.
    genome_ok = not missing_seq and extractable_fraction > MIN_EXTRACTABLE_FRACTION
    genome_detail = (
        f"seqname_overlap={seqname_overlap}; sampled={len(sampled)} (seed {C.seed(config)}); "
        f"extractable={extracted}/{len(sampled)} ({extractable_fraction:.4f}, min "
        f"{MIN_EXTRACTABLE_FRACTION}); valid_acgt={valid}; "
        f"fasta_sequences={len(fasta_names)}; {cache_note}"
    )

    row["seqname_overlap"] = seqname_overlap
    row["n_peaks_extractable"] = extracted
    row["extractable_fraction"] = f"{extractable_fraction:.4f}"
    row["n_peaks_valid_acgt"] = valid

    # ---- METADATA_OK -----------------------------------------------------
    celltype_column = "Org_Celltype" if "Org_Celltype" in obs.columns else ""
    stage_column = "Org_stage" if "Org_stage" in obs.columns else ""
    row["celltype_column"] = celltype_column or "absent"
    row["stage_column"] = stage_column or "absent"
    sample_column = next((c for c in SAMPLE_COLUMN_PREFERENCE if c in obs.columns), CELL_ID_PREFIX)
    donor_column = next((c for c in DONOR_COLUMN_PREFERENCE if c in obs.columns), CELL_ID_PREFIX)
    row["sample_column"] = sample_column
    row["donor_column"] = donor_column

    celltype_counts: list[dict] = []
    if celltype_column:
        counts = obs[celltype_column].astype(str).value_counts()
        for name, count in counts.items():
            celltype_counts.append(
                dict(
                    species=species,
                    scope="species_h5ad",
                    source=h5ad_path.name,
                    org_celltype=str(name),
                    is_target=bool(str(name) in TARGET_CELLTYPES),
                    n_cells=int(count),
                )
            )

    missing_meta = 0
    for column in (celltype_column, stage_column):
        if not column:
            missing_meta += len(obs)
            continue
        missing_meta += int((obs[column].astype(str).str.strip() == "").sum())

    coverage, unmapped, unmapped_names, declared = (
        coverage_rows(species, obs, stage_windows(stage_map))
        if stage_column
        else ([], len(obs), [], species in stage_windows(stage_map))
    )
    if unmapped:
        notes.append(
            f"{species}: {unmapped}/{len(obs)} cells have no Org_stage -> cross_window mapping "
            f"(stages={truncate(unmapped_names)}; species in stage_map={declared})"
        )

    sample_units = int(obs[sample_column].astype(str).nunique()) if sample_column != CELL_ID_PREFIX else len(cell_prefixes)
    metadata_ok = (
        bool(celltype_column)
        and bool(stage_column)
        and missing_meta == 0
        and (not declared or unmapped == 0)
    )
    metadata_detail = (
        f"celltype_column={row['celltype_column']} ({len(celltype_counts)} categories, "
        f"missing={int((obs[celltype_column].astype(str).str.strip() == '').sum()) if celltype_column else len(obs)}); "
        f"stage_column={row['stage_column']} "
        f"({obs[stage_column].nunique() if stage_column else 0} categories); "
        f"sample_column={sample_column} ({sample_units} units); donor_column={donor_column}; "
        f"cross_window: {len(obs) - unmapped}/{len(obs)} cells mapped"
    )
    if declared:
        metadata_detail += " (unmapped=0 required)"
    else:
        metadata_detail += (
            f" (species absent from stage_map: cross_window not defined for "
            f"{truncate(unmapped_names)}, used by Purkinje/Oligo only)"
        )
    if sample_column == CELL_ID_PREFIX:
        metadata_detail += "; no sample/donor column present, cell_id prefix used"

    gates = [
        _gate(species, "RAW_COUNTS_OK", "PASS" if raw_ok else "FAIL", raw_detail),
        _gate(species, "PEAK_COORDINATES_OK", "PASS" if coords_ok else "FAIL", coord_detail),
        _gate(species, "GENOME_MATCH_OK", "PASS" if genome_ok else "FAIL", genome_detail),
        _gate(species, "METADATA_OK", "PASS" if metadata_ok else "FAIL", metadata_detail),
    ]
    notes.append(f"{species}: index={index_name}; x={x_stats.dtype}; {cache_note}")
    return SpeciesResult(
        species=species,
        row=row,
        gates=gates,
        coverage=coverage,
        celltype_counts=celltype_counts,
        fasta_seqs=len(fasta_names),
        notes=notes,
    )


def audit_stage_splits(
    stage_map: pd.DataFrame, species: str, split_dir: Path | None = None
) -> tuple[list[dict], list[str]]:
    """Count target celltypes per stage_split file and verify its declared stages.

    `split_dir` is the period-split directory of the run (`stage_split_dir`); it
    holds both the per-period cells and the `raw_h5ad` file names the period
    table names, so the declared stages are checked against the split files of
    the *same* period definition. When it is not given the v1 directory is used.

    A declared `raw_h5ad` file that is not under that directory -- or a directory
    that holds no `*.h5ad` at all -- is named in a note here and recorded as a
    missing input by `inventory_rows`, so a mis-pointed or emptied split root can
    never read as a species with nothing to audit.
    """
    rows: list[dict] = []
    notes: list[str] = []
    split_dir = Path(split_dir) if split_dir is not None else C.ROOT / "data" / "stage_split_h5ad"
    declared: set[str] = set()
    for entry in stage_map.itertuples():
        if str(entry.species) != species:
            continue
        declared.update(stage.strip() for stage in str(entry.raw_stages).split("|") if stage.strip())
    # A glob can only report what IS in the directory: a split root that holds
    # none of the files the period table declares yields zero rows and zero
    # notes, which reads exactly like a species with nothing to audit -- the
    # silent loss this audit exists to prevent. So the directory is compared
    # with the table and every declared file that is absent is named. "On disk"
    # is measured exactly as `inventory_rows` measures it, so the note and the
    # inventory row for one file can never disagree.
    declared_files = declared_split_files(stage_map, species)
    present_names = {path.name for path in split_dir.glob("*.h5ad")}
    absent_files = [name for name in declared_files if name not in present_names]
    if declared_files and not any(split_dir.glob("*.h5ad")):
        notes.append(
            f"{species}: {display_path(split_dir)} holds no *.h5ad split file at all, so the "
            f"period membership of {species} is not audited (period table declares "
            f"{truncate(declared_files)}); each declared file is recorded as a missing input"
        )
    elif absent_files:
        notes.append(
            f"{species}: {len(absent_files)}/{len(declared_files)} split files declared by the "
            f"period table are absent under {display_path(split_dir)}: {truncate(absent_files)}; "
            f"their cells are not audited"
        )
    for path in sorted(split_dir.glob(f"{species}_*.h5ad")):
        frame, _ = C.obs_frame(path)
        if "Org_Celltype" not in frame.columns:
            notes.append(f"{path.name}: no Org_Celltype column; {len(frame)} cells not counted")
            continue
        counts = frame["Org_Celltype"].astype(str).value_counts()
        for name, count in counts.items():
            rows.append(
                dict(
                    species=species,
                    scope="stage_split",
                    source=path.name,
                    org_celltype=str(name),
                    is_target=bool(str(name) in TARGET_CELLTYPES),
                    n_cells=int(count),
                )
            )
        if "Org_stage" in frame.columns and declared:
            extra = sorted(set(frame["Org_stage"].astype(str)) - declared)
            if extra:
                notes.append(
                    f"{path.name}: Org_stage outside the stage_map row of {species}: {truncate(extra)}"
                )
    return rows, notes


# --------------------------------------------------------------------------
# inputs inventory
# --------------------------------------------------------------------------


def inventory_rows(
    config: dict,
    manifest: pd.DataFrame,
    genomes: dict[str, C.SpeciesGenome],
    fasta_seqs: dict[str, int],
    h5ad_shapes: dict[str, tuple[int, int]],
    hash_files: bool,
    stage_map: pd.DataFrame | None = None,
    species_scope: list[str] | None = None,
) -> list[dict]:
    rows: list[dict] = []

    def add(kind: str, name: str, path: Path, shape: str, detail: str) -> None:
        exists = path.exists()
        rows.append(
            dict(
                kind=kind,
                name=name,
                path=display_path(path),
                exists=exists,
                size_bytes=int(path.stat().st_size) if exists else 0,
                sha256=C.sha256_file(path) if (exists and hash_files) else "",
                n_rows_or_shape=shape if exists else "",
                detail=detail,
            )
        )

    for species, genome in genomes.items():
        add(
            "species_h5ad",
            f"{species}_deepcerevo_2026.h5ad",
            C.ROOT / "data" / "all_species_h5ad" / f"{species}_deepcerevo_2026.h5ad",
            "x".join(str(v) for v in h5ad_shapes.get(species, ("", ""))),
            "all QC peaks; cells x peaks",
        )
        add(
            "fasta",
            genome.fasta.name,
            genome.fasta,
            str(fasta_seqs.get(species, "")),
            f"assembly={genome.assembly}; prefix={genome.prefix}; chrom_style={genome.chrom_style}",
        )

    split_root = stage_split_dir(config)
    present_splits: set[str] = set()
    for path in sorted(split_root.glob("*.h5ad")):
        present_splits.add(path.name)
        try:
            with h5py.File(path, "r") as handle:
                shape = tuple(int(v) for v in handle["X"].attrs["shape"])
            frame_shape = "x".join(str(v) for v in shape)
        except (OSError, KeyError, ValueError):
            frame_shape = "unreadable"
        add("stage_split_h5ad", path.name, path, frame_shape, "period membership only")

    # What the period table declares, not what the directory happens to hold, is
    # the ground truth for the split files: a declared file that is not on disk
    # means the cells of that period are not audited at all, so it is recorded as
    # a missing input rather than dropping out of the inventory. Rows are sorted
    # by file name, exactly as the on-disk glob is, so an absent file occupies the
    # same position it would occupy if it were present.
    if stage_map is not None and species_scope is not None:
        absent_declared: dict[str, str] = {}
        for species in species_scope:
            for name in declared_split_files(stage_map, species):
                if name not in present_splits:
                    absent_declared.setdefault(name, species)
        for name, species in sorted(absent_declared.items()):
            add(
                "stage_split_h5ad",
                name,
                split_root / name,
                "",
                f"period split file declared by the period table for {species}; not found "
                f"under {display_path(split_root)}",
            )

    for row in manifest.itertuples():
        add(
            "ranked_csv",
            row.path.name,
            Path(row.path),
            str(row.n_ranked),
            f"model={row.model}; {'EMPTY ranking (skipped downstream)' if row.is_empty else 'ok'}",
        )

    motif_db = C.motif_database(config)
    n_motifs = sum(1 for line in motif_db.read_text().splitlines() if line.startswith("MOTIF "))
    expected = int(config["motif_database"]["n_motifs"])
    add(
        "motif_db",
        motif_db.name,
        motif_db,
        str(n_motifs),
        f"expected n_motifs={expected}; match={n_motifs == expected}",
    )
    return rows


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def write_summary(
    path: Path,
    species_results: list[SpeciesResult],
    gates: pd.DataFrame,
    audit_dir: Path,
    inventory: pd.DataFrame,
    notes: list[str],
    stage_map_display: str | None = None,
    species_filter: pd.DataFrame | None = None,
) -> bool:
    """Render the human-readable audit for one run.

    `stage_map_display` is the period table the run was audited against and
    `species_filter` the (analysis, species) table of the configured species
    floor. Both are optional: a run without a species filter (v1) renders the
    same sections, the same four gate columns and the same period-table label it
    always has, so its published summary is reproduced byte for byte.
    """
    overall = bool((gates.status == "PASS").all())
    known = gates[(gates.status == "FAIL") & (~gates.blocking)] if "blocking" in gates else gates.iloc[:0]
    if overall:
        headline = "PASS"
    elif len(known) == int((gates.status == "FAIL").sum()):
        headline = f"PASS with {len(known)} pre-registered non-blocking gate failure(s)"
    else:
        headline = "FAIL"
    # the species-filter gate only exists when the config declares a filter, so
    # the gate table gains a column exactly for the runs that have the gate
    gate_columns = list(GATES)
    if SPECIES_FILTER_GATE in set(gates.gate):
        gate_columns.append(SPECIES_FILTER_GATE)
    stage_map_label = stage_map_display or display_path(C.stage_map_path())
    lines = [
        "# Step 0 input audit",
        "",
        f"Overall: **{headline}**",
        "",
        "Hard gates per species (readme section 3):",
        "",
        "| species | " + " | ".join(gate_columns) + " |",
        "| --- |" + " --- |" * len(gate_columns),
    ]
    for result in species_results:
        status = {gate["gate"]: gate["status"] for gate in result.gates}
        lines.append(
            "| " + result.species + " | " + " | ".join(status.get(g, "MISSING") for g in gate_columns) + " |"
        )
    lines += ["", "## Input inventory", ""]
    lines.append(f"`input_inventory.tsv`: {len(inventory)} entries "
                 f"({inventory.kind.value_counts().to_dict()}); "
                 f"{int((~inventory.exists).sum())} missing on disk.")
    lines += ["", "## Target celltype counts (integrated `Org_Celltype`)", ""]
    lines.append("| species | " + " | ".join(TARGET_CELLTYPES) + " |")
    lines.append("| --- |" + " --- |" * len(TARGET_CELLTYPES))
    for result in species_results:
        counts = {
            row["org_celltype"]: row["n_cells"]
            for row in result.celltype_counts
            if row["scope"] == "species_h5ad"
        }
        lines.append(
            "| "
            + result.species
            + " | "
            + " | ".join(str(counts.get(name, 0)) for name in TARGET_CELLTYPES)
            + " |"
        )
    if species_filter is not None and len(species_filter):
        dropped = species_filter[~species_filter.passed]
        lines += ["", "## Species filter", ""]
        lines.append(
            f"`species_filter.tsv`: {len(species_filter)} (analysis, species) rows; "
            f"min_cells_per_species={int(species_filter.min_cells.max())}; "
            f"{len(dropped)} dropped and recorded."
        )
        lines += [
            "",
            "| analysis | species | celltype | n_cells | min_cells | verdict |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in species_filter.itertuples():
            verdict = "kept" if row.passed else "dropped"
            lines.append(
                f"| {row.analysis} | {row.species} | {row.celltype} | {row.n_cells} "
                f"| {row.min_cells} | {verdict} |"
            )
    # The files THIS step writes -- a literal set, not a directory scan.
    #
    # Scanning `audit_dir` made the summary depend on run HISTORY rather than on
    # the inputs: on a re-run it also listed the previous run's `AUDIT_SUMMARY.md`
    # and `astroglia_stage_gate_trace.tsv` (written by a LATER step), which changed
    # the file by 50 bytes for no reason. Measured: re-running the Top1000 arm
    # turned this document from 5369 into 5419 bytes while every number in it
    # stayed identical. `AUDIT_SUMMARY.md` is deliberately absent from the list --
    # it is the document doing the listing, and leaving it out is what makes this
    # line match the published v2 summary exactly.
    written = [
        "audit_gates.tsv",
        "celltype_counts.tsv",
        "input_inventory.tsv",
        "species_audit.tsv",
        "stage_map_coverage.tsv",
    ]
    if species_filter is not None and len(species_filter):
        written.append("species_filter.tsv")
    lines += ["", "Files: " + ", ".join(sorted(written)), ""]

    failing = gates[gates.status == "FAIL"]
    lines += ["## FAILing gates", ""]
    if failing.empty:
        lines.append("None.")
    else:
        for row in failing.itertuples():
            blocking = getattr(row, "blocking", True)
            label = "BLOCKING" if blocking else "pre-registered, non-blocking"
            lines.append(f"- `{row.species}` / `{row.gate}` ({label}): {row.detail}")
            if not blocking:
                lines.append(f"  - reason: {row.exception_reason}")
                lines.append(f"  - mitigation: {row.exception_mitigation}")
    lines += ["", "## Notes", ""]
    lines += [f"- {note}" for note in notes]
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- `raw_integer_fraction` is measured over every stored value of `X` (sparse entries only);",
        "  with an integer dtype it is 1.0 by construction.",
        "- `n_peaks_extractable` / `extractable_fraction` come from a fixed-seed sample of the",
        "  species' full QC peak set, extracted through `common.read_sequences`.",
        "- The `Org_stage -> cross_window` completeness requirement is enforced for the species",
        f"  that `{stage_map_label}` declares; species absent from that table (Purkinje/Oligo",
        "  only, no developmental window comparison) are reported as unmapped in",
        "  `stage_map_coverage.tsv` instead of failing the gate.",
        "- A FAIL here blocks the pipeline by design: fix the input or the config, never the gate.",
    ]
    if species_filter is not None and len(species_filter):
        lines += [
            "",
            "- `SPECIES_FILTER_OK` checks that every (analysis, species) decision of the",
            "  configured species floor is recorded with the cell count behind it. A drop is a",
            "  data fact and passes; a count that cannot be measured, or a recorded drop that",
            "  disagrees with `common.apply_species_filter`, fails.",
        ]
    lines += [""]
    path.write_text("\n".join(lines))
    C.log("wrote", display_path(path))
    return overall


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def build_gates_frame(results: list[SpeciesResult], species_list: list[str]) -> pd.DataFrame:
    rows = []
    for species in species_list:
        result = next((item for item in results if item.species == species), None)
        if result is None:
            rows += _all_fail(species, "species not audited")
            continue
        rows += result.gates
    frame = pd.DataFrame(rows)[["species", "gate", "status", "detail"]]
    return frame


def attach_filter_gate(result: SpeciesResult, gates: dict[str, dict]) -> SpeciesResult:
    """Append the species-filter gate to a species' gate list, when it has one.

    A species the filter never decides on (no ranked model) gets no such gate:
    an invented PASS would claim a check that never ran.
    """
    gate = gates.get(result.species)
    if gate is not None:
        result.gates.append(gate)
    return result


def main() -> int:
    C.main_guard()
    parser = argparse.ArgumentParser(description="Step 0: audit every input of the frozen pipeline")
    C.add_common_args(parser)
    parser.add_argument("--species", nargs="*", default=None, help="audit only these species")
    parser.add_argument("--out-dir", default=None, help="write the audit here (tests); default results/00_audit")
    parser.add_argument("--no-hash", action="store_true", help="skip sha256 in the inventory (fast reruns)")
    args = parser.parse_args()

    if args.species and not args.out_dir:
        # a subset run shares the canonical output directory, so allowing it
        # would replace the full input gate with a partial one -- and a partial
        # gate that happens to exclude the failing species reports PASS with
        # exit 0. Partial audits are only allowed where they cannot masquerade
        # as the canonical tables.
        parser.error(
            "--species audits only a subset of the species and would overwrite the "
            "canonical results/00_audit/ tables of the full input gate with a partial "
            "audit; pass --out-dir to redirect a subset run"
        )

    config = C.load_config(args.config)
    species_list = args.species or list(C.SPECIES_ORDER)
    audit_dir = Path(args.out_dir) if args.out_dir else C.results_dir(config, "00_audit")
    audit_dir.mkdir(parents=True, exist_ok=True)

    genomes = C.species_table(config)
    # the period table, the split directory and the species floor all come from
    # the config: v2 audits its developmental periods against its own table, v1
    # keeps reading the frozen one, and neither path is written into this file
    stage_map = C.stage_table(config)
    split_dir = stage_split_dir(config)
    manifest = C.ranking_files(config)
    ranked_by_species: dict[str, list[str]] = {}
    for row in manifest.itertuples():
        if row.is_empty:
            continue
        frame = pd.read_csv(row.path, usecols=["gene"])
        ranked_by_species.setdefault(row.species, []).extend(map(str, frame.gene))

    results: list[SpeciesResult] = []
    notes: list[str] = []
    filter_rows: list[dict] = []
    filter_gates: dict[str, dict] = {}
    if species_filter_active(config):
        filter_rows, filter_notes = species_filter_rows(config, manifest, species_list)
        notes += filter_notes
        for row in species_filter_gates(config, manifest, filter_rows, species_list):
            filter_gates[row["species"]] = row
            C.log("SPECIES FILTER", f"{row['species']:<9}", row["status"], row["detail"])
    if args.species:
        notes.append(
            f"PARTIAL AUDIT: only {truncate(species_list, limit=len(species_list))} were audited "
            f"(--species); the canonical gate covers all of "
            f"{truncate(list(C.SPECIES_ORDER), limit=len(C.SPECIES_ORDER))}"
        )
    with C.StepTimer("s00 audit"):
        for species in species_list:
            genome = genomes.get(species)
            if genome is None:
                detail = "species absent from config/species_genome.tsv"
                notes.append(f"{species}: {detail}")
                row = {column: "" for column in SPECIES_AUDIT_COLUMNS}
                row["species"] = species
                results.append(
                    attach_filter_gate(
                        SpeciesResult(species, row, _all_fail(species, detail), [], [], 0, [detail]),
                        filter_gates,
                    )
                )
                continue
            with C.StepTimer(f"audit {species}"):
                result = audit_species(
                    config,
                    genome,
                    species,
                    C.ROOT / "data" / "all_species_h5ad" / f"{species}_deepcerevo_2026.h5ad",
                    stage_map,
                    ranked_by_species.get(species, []),
                )
                split_rows, split_notes = audit_stage_splits(stage_map, species, split_dir)
                result.celltype_counts += split_rows
                notes += split_notes
                results.append(attach_filter_gate(result, filter_gates))
            status = " ".join(f"{gate['gate']}={gate['status']}" for gate in result.gates)
            C.log("AUDIT", f"{species:<9}", status)
            counts = {
                row["org_celltype"]: row["n_cells"]
                for row in result.celltype_counts
                if row["scope"] == "species_h5ad"
            }
            target_line = " ".join(f"{name}={counts.get(name, 0)}" for name in TARGET_CELLTYPES)
            print(f"[s00] {species:<9} target celltypes (integrated obs): {target_line}", flush=True)
            minimum = int(config["chromvar"].get("min_cells_per_celltype", 30))
            for name in TARGET_CELLTYPES:
                if counts.get(name, 0) < minimum:
                    notes.append(
                        f"{species}: target celltype {name} has {counts.get(name, 0)} cells "
                        f"(< min_cells_per_celltype={minimum}); check it before the chromVAR step"
                    )
            notes += result.notes

    gates = build_gates_frame(results, species_list)
    audit = pd.DataFrame([result.row for result in results])[list(SPECIES_AUDIT_COLUMNS)]
    coverage = pd.DataFrame(
        [row for result in results for row in result.coverage],
        columns=["species", "org_stage", "cross_window", "n_cells", "mapped", "species_declared"],
    )
    counts_frame = pd.DataFrame(
        [row for result in results for row in result.celltype_counts],
        columns=["species", "scope", "source", "org_celltype", "is_target", "n_cells"],
    )

    h5ad_shapes = {
        result.species: (int(result.row.get("n_cells") or 0), int(result.row.get("n_peaks") or 0))
        for result in results
        if result.row.get("n_cells") != ""
    }
    inventory = pd.DataFrame(
        inventory_rows(
            config,
            manifest,
            {name: genomes[name] for name in species_list if name in genomes},
            {result.species: result.fasta_seqs for result in results},
            h5ad_shapes,
            hash_files=not args.no_hash,
            stage_map=stage_map,
            species_scope=species_list,
        )
    )

    gates = annotate_known_limitations(gates)
    blocking = gates[(gates.status == "FAIL") & gates.blocking]

    # the species-filter table only exists for a config that declares a filter,
    # so a run without one writes exactly the five tables it always wrote
    species_filter_frame = pd.DataFrame(filter_rows, columns=list(SPECIES_FILTER_COLUMNS))
    save_audit_table(inventory, audit_dir / "input_inventory.tsv")
    save_audit_table(audit, audit_dir / "species_audit.tsv")
    save_audit_table(gates, audit_dir / "audit_gates.tsv")
    save_audit_table(counts_frame, audit_dir / "celltype_counts.tsv")
    save_audit_table(coverage, audit_dir / "stage_map_coverage.tsv")
    if not species_filter_frame.empty:
        save_audit_table(species_filter_frame, audit_dir / "species_filter.tsv")
    overall = write_summary(
        audit_dir / "AUDIT_SUMMARY.md",
        results,
        gates,
        audit_dir,
        inventory,
        notes,
        stage_map_display=display_path(C.stage_map_path(config)),
        species_filter=species_filter_frame,
    )

    C.log("AUDIT OVERALL", "PASS" if overall else "FAIL")
    for row in gates[gates.status == "FAIL"].itertuples():
        tag = "KNOWN " if not row.blocking else "FAIL"
        C.log(tag, row.species, row.gate, row.detail)
    if not blocking.empty:
        C.log("BLOCKING GATE FAILURES:", int(len(blocking)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
