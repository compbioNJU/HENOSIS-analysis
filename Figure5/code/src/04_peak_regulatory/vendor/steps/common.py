"""Shared core for the frozen cross-species peak regulatory analysis.

This module is the contract every step script (s00 .. s12) is written against.
It owns:

  * configuration loading (config/analysis.yaml, species_genome.tsv,
    stage_map.tsv, tf_family_map.tsv)
  * the ranked-CSV manifest, i.e. what a "model" is in this analysis
  * reference-sequence extraction with a checksum-keyed cache
  * GC/length one-to-one matching without replacement
  * FIMO motif scanning of a whole species universe, parallel and cached
  * Fisher enrichment over all 879 motifs with BH correction
  * the matched random-500 control used by every "model vs null" claim
  * small statistics helpers (Spearman, exact permutation, family medians)

Nothing in here looks at results: all choices are configuration.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr
from statsmodels.stats.multitest import multipletests

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CONFIG_DIR = ROOT / "config"

FIMO_BIN = os.environ.get(
    "FIMO_BIN", "fimo"
)

ANALYSES = ("Astroglia", "Purkinje", "Oligo", "GC")
# ranked-file stem -> analysis label and target celltype
ANALYSIS_CELLTYPE = {
    "Astroglia": "astroglia",
    "Purkinje": "Purkinje",
    "Oligo": "oligo",
    "GC": "granule cell (GC)",
}
# Org_Celltype values that define the target population of each analysis
ANALYSIS_ORG_CELLTYPE = {
    "Astroglia": "astroglia",
    "Purkinje": "Purkinje",
    "Oligo": "oligo",
    "GC": "GC",
}
SPECIES_ORDER = ["human", "bonobo", "macaque", "marmoset", "mouse", "opossum"]

# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------


def log(*args: object) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *args, flush=True)


class StepTimer:
    def __init__(self, label: str) -> None:
        self.label = label

    def __enter__(self) -> "StepTimer":
        self.t0 = time.time()
        log("START", self.label)
        return self

    def __exit__(self, *exc: object) -> None:
        log("DONE ", self.label, f"{time.time() - self.t0:.1f}s")


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def load_config(path: str | Path | None = None) -> dict:
    import yaml

    if path is None:
        path = CONFIG_DIR / "analysis.yaml"
    with open(path) as handle:
        config = yaml.safe_load(handle)
    config["_path"] = str(path)
    config["_root"] = str(ROOT)
    return config


def config_path(config: dict, key: str) -> Path:
    """Resolve a path declared in the `paths:` block against the project root."""
    value = Path(config["paths"][key])
    return value if value.is_absolute() else ROOT / value


def results_dir(config: dict, step: str) -> Path:
    path = config_path(config, "results") / step
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir(config: dict) -> Path:
    path = config_path(config, "cache")
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir(config: dict) -> Path:
    path = config_path(config, "logs")
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed(config: dict) -> int:
    return int(config["seed"])


def topk_label(config: dict) -> str:
    """The cut-off this run is at, spelled the way a figure or table should spell it.

    v3 runs the same figures three times, once per arm, so any label that names
    the cut-off MUST come from the arm's config. A hardcoded "Top500" is not a
    cosmetic wart here: it would caption the Top1000 and Top2000 arms' kept
    figures with a cut-off those numbers did not come from.
    """
    return f"Top{int(config['model_analysis']['primary_topk'])}"


def null_label(config: dict) -> str:
    """The matched-random null's name, which v3 changed.

    v2's null was a random-500 draw that routinely landed short of 500. v3 draws
    every repeat at exactly the OBSERVED table's own width, which is regularly
    below K, so "random-500" (or random-K) misnames it in both size and kind.
    """
    return "matched-random (exact at the observed table width)"


def data_file(config: dict, relative: str) -> Path:
    return ROOT / "data" / relative


def motif_database(config: dict) -> Path:
    return ROOT / config["motif_database"]["file"]


# --------------------------------------------------------------------------
# species / genome table
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SpeciesGenome:
    species: str
    assembly: str
    fasta: Path
    prefix: str
    chrom_style: str  # "as_is" or "add_chr"


def species_table(config: dict | None = None) -> dict[str, SpeciesGenome]:
    frame = pd.read_csv(CONFIG_DIR / "species_genome.tsv", sep="\t")
    out: dict[str, SpeciesGenome] = {}
    for row in frame.itertuples():
        fasta = Path(row.fasta)
        if not fasta.is_absolute():
            fasta = ROOT / fasta
        out[row.species] = SpeciesGenome(
            species=row.species,
            assembly=row.assembly,
            fasta=fasta,
            prefix=row.prefix,
            chrom_style=row.chrom_style,
        )
    return out


def stage_map_path(config: dict | None = None) -> Path:
    """Which period table this run uses.

    v1 leaves `paths.stage_map` out and gets `config/stage_map.tsv`; v2 sets it
    to `config/stage_map_v2.tsv`. Keeping it in the config is what lets the same
    step scripts run against either period definition without editing them.
    """
    if config is not None:
        declared = (config.get("paths") or {}).get("stage_map")
        if declared:
            path = Path(declared)
            return path if path.is_absolute() else ROOT / path
    return CONFIG_DIR / "stage_map.tsv"


def stage_table(config: dict | None = None) -> pd.DataFrame:
    return pd.read_csv(stage_map_path(config), sep="\t")


def cross_windows(config: dict | None = None) -> tuple[str, ...]:
    """The windows that enter a cross-species comparison, in table order.

    A window declared for only one species is that species' own axis (human
    Carnegie 18-23 in v2, `human_only` in v1) and never enters a cross-species
    statistic; everything declared for two or more species does.
    """
    table = stage_table(config)
    counts = table.groupby("cross_window")["species"].nunique()
    return tuple(w for w in table.cross_window.drop_duplicates() if counts.get(w, 0) >= 2)


def single_species_windows(config: dict | None = None) -> tuple[str, ...]:
    """Windows belonging to exactly one species; excluded from cross-species work."""
    table = stage_table(config)
    counts = table.groupby("cross_window")["species"].nunique()
    return tuple(w for w in table.cross_window.drop_duplicates() if counts.get(w, 0) < 2)


def species_axis(config: dict | None = None) -> dict[str, list[str]]:
    """species -> its periods in table order, including the single-species ones."""
    table = stage_table(config)
    return {
        species: list(group.rank_period)
        for species, group in table.groupby("species", sort=False)
    }


def min_cells_per_species(config: dict) -> int:
    """Cell floor for a species to enter a cell-type analysis."""
    block = config.get("eligibility") or config.get("species_filter") or {}
    return int(block.get("min_target_cells", block.get("min_cells_per_species", 0)))


def min_ranked_peaks(config: dict) -> int:
    """Peak floor: a ranking shorter than this cannot define a Top-k foreground."""
    block = config.get("eligibility") or {}
    return int(block.get("min_ranked_peaks", 0))


def species_cell_counts(config: dict, celltype_column: str | None = None) -> pd.DataFrame:
    """Cells per (species, celltype) in the integrated h5ad, for the species filter.

    Returns a long frame `species, celltype, n_cells`. This is what decides which
    species may enter a cell-type analysis: a species below the configured floor
    is dropped from that analysis and named in the audit, never silently kept.
    """
    block = config.get("eligibility") or config.get("species_filter") or {}
    column = celltype_column or block.get("celltype_column", "Org_Celltype")
    rows = []
    for species in SPECIES_ORDER:
        path = data_file(config, f"all_species_h5ad/{species}_deepcerevo_2026.h5ad")
        if not path.exists():
            continue
        obs, _ = obs_frame(path)
        counts = obs[column].astype(str).value_counts()
        for celltype, n in counts.items():
            rows.append(dict(species=species, celltype=celltype, n_cells=int(n)))
    return pd.DataFrame(rows)


def species_passing_filter(
    config: dict, celltype: str, candidates: list[str] | None = None
) -> tuple[list[str], list[tuple[str, int]]]:
    """Split candidate species into (kept, dropped-with-count) for one celltype."""
    counts = species_cell_counts(config)
    sub = counts[counts.celltype == celltype].set_index("species").n_cells.to_dict()
    pool = candidates if candidates is not None else list(SPECIES_ORDER)
    floor = min_cells_per_species(config)
    kept, dropped = [], []
    for species in pool:
        n = int(sub.get(species, 0))
        if n >= floor:
            kept.append(species)
        else:
            dropped.append((species, n))
    return kept, dropped


def tf_family_table() -> pd.DataFrame:
    return pd.read_csv(CONFIG_DIR / "tf_family_map.tsv", sep="\t")


# --------------------------------------------------------------------------
# cell-type-specific motif modules (the v2 interpretation layer)
# --------------------------------------------------------------------------


def module_map_path(config: dict | None = None) -> Path:
    declared = ((config or {}).get("modules") or {}).get("file")
    if declared:
        path = Path(declared)
        return path if path.is_absolute() else ROOT / path
    return CONFIG_DIR / "celltype_motif_module_map.tsv"


def celltype_module_table(config: dict | None = None) -> pd.DataFrame:
    """The frozen module map: one row per (celltype, module, motif, member_type).

    Modules are atomic and disjoint within a cell type. This is an INTERPRETATION
    layer over the unbiased 879-motif result -- it never changes a primary
    statistic, and a module at a non-gating level can never decide a claim.
    """
    path = module_map_path(config)
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run script/make_module_map.py first"
        )
    frame = pd.read_csv(path, sep="\t")
    required = {"celltype", "module", "level", "member_type", "motif_name", "motif_id"}
    absent = sorted(required - set(frame.columns))
    if absent:
        raise ValueError(f"{path} lacks columns {absent}")
    return frame


def modules_for(config: dict, celltype: str, level: str | None = None) -> list[str]:
    table = celltype_module_table(config)
    sub = table[table.celltype == celltype]
    if level is not None:
        sub = sub[sub.level == level]
    return list(dict.fromkeys(sub.module))


def module_members(
    config: dict, celltype: str, module: str, monomers_only: bool = True
) -> list[str]:
    """Motif IDs of one atomic module.

    Heterodimers are excluded by default: a module median must be built from
    binding models of one factor, not from a fusion of two.
    """
    table = celltype_module_table(config)
    sub = table[(table.celltype == celltype) & (table.module == module)]
    if monomers_only:
        sub = sub[sub.member_type == "monomer"]
    return sorted(sub.motif_id.unique())


def module_aggregates(config: dict, celltype: str) -> dict[str, list[str]]:
    """Derived module unions declared in the config.

    `Homeodomain_total` is not a third set of rows: it is the union of two atomic
    modules, so no motif is ever counted twice and the subdivisions stay readable.
    """
    declared = ((config or {}).get("modules") or {}).get("aggregates") or {}
    return dict(declared.get(celltype, {}))


def module_is_gating(config: dict, level: str) -> bool:
    """Whether a module at this level may take part in a PASS/FAIL rule."""
    blocked = ((config or {}).get("modules") or {}).get("non_gating_levels") or []
    return level not in set(blocked)


def module_family_table(config: dict, celltype: str):
    """The cell-type module map in `tf_family_table` shape.

    `family_trajectory` and the family figures were written against a frame with
    `motif_id / family / analysis_level / member_type`. Handing them the cell-type
    panel in that shape is what makes all three analyses read ONE authoritative
    map instead of Astroglia reading the old project-wide one and the other two
    reading the module map: the module is the family, `level` is `analysis_level`.

    Derived aggregates (`Homeodomain_total`) are NOT included: they are unions of
    two atomic modules, and `family_trajectory` groups by a single column, so
    including them would double count every POU and homeodomain motif.
    """
    table = celltype_module_table(config)
    sub = table[table.celltype == celltype]
    if sub.empty:
        raise ValueError(f"{module_map_path(config)} declares no module for {celltype}")
    return (
        sub.rename(columns={"module": "family", "level": "analysis_level"})
        [["motif_id", "motif_name", "family", "analysis_level", "member_type"]]
        .drop_duplicates(["motif_id", "family"])
        .reset_index(drop=True)
    )


def primary_families() -> list[str]:
    table = tf_family_table()
    values = table.loc[table.analysis_level == "primary", "family"].dropna().unique()
    return sorted(v for v in values if v)


def family_members(family: str, monomers_only: bool = True) -> list[str]:
    table = tf_family_table()
    sub = table[table.family == family]
    if monomers_only:
        sub = sub[sub.member_type == "monomer"]
    return sorted(sub.motif_id.unique())


# --------------------------------------------------------------------------
# ranked model manifest
# --------------------------------------------------------------------------

# v1 named a ranked file "<species>_<period>_<celltype>_deepcerevo_2026.csv".
_RANK_RE_ASTRO = re.compile(
    r"^(human|mouse|marmoset)_(.+)_astroglia_deepcerevo_2026\.csv$"
)
_RANK_RE_GC = re.compile(r"^(human|mouse|marmoset)_(.+)_GC_deepcerevo_2026\.csv$")
# v2 names it "<species>-<period>.csv" with no celltype token: the regenerated
# astroglia rankings are the only per-period ranking produced for that definition.
_RANK_RE_V2 = re.compile(r"^(human|mouse|marmoset)-(.+)\.csv$")
_RANK_RE_BTACSUB = re.compile(
    r"^btacsub_(Purkinje|oligo)_(human|mouse|marmoset|macaque|opossum|bonobo)"
    r"_deepcerevo_2026\.csv$"
)


def ranked_dirs(config: dict) -> dict[str, Path]:
    """Where each analysis' ranked CSVs live, as declared in the config.

    v1 implicitly used `data/peak_ranked`; v2 declares two directories, so the
    Astroglia rankings can move to the developmental periods while Purkinje and
    Oligo keep the frozen, unchanged files.
    """
    declared = (config.get("paths") or {}).get("ranked_dirs")
    if not declared:
        root = data_file(config, "peak_ranked")
        return {"astroglia": root, "purkinje_oligo": root / "purkinje_oligo"}
    out: dict[str, Path] = {}
    for key, value in declared.items():
        path = Path(value)
        out[key] = path if path.is_absolute() else ROOT / path
    return out


def ranking_files(config: dict) -> pd.DataFrame:
    """Enumerate every ranked CSV and describe it as one model."""
    dirs = ranked_dirs(config)
    rows: list[dict] = []

    # A declared directory that does not exist means the run is pointed at the
    # wrong tree; skipping it would silently drop an entire analysis arm, so it
    # is an error, not a warning.
    for key, directory in dirs.items():
        if not directory.is_dir():
            raise FileNotFoundError(
                f"ranked_dirs.{key} does not exist: {directory}. The analysis root's "
                "data/ directory holds symlinks; a path declared in the config must "
                "resolve inside it."
            )

    astro_dir = dirs.get("astroglia")
    if astro_dir is not None:
        for path in sorted(astro_dir.glob("*.csv")):
            for analysis, pattern in (
                ("Astroglia", _RANK_RE_ASTRO),
                ("GC", _RANK_RE_GC),
                ("Astroglia", _RANK_RE_V2),
            ):
                match = pattern.match(path.name)
                if match:
                    rows.append(
                        dict(analysis=analysis, species=match.group(1),
                             period=match.group(2), path=path)
                    )
                    break
            else:
                unclaimed = path.name
                log("WARNING ranked CSV matched no naming rule and was skipped:", unclaimed)

    btac_dir = dirs.get("purkinje_oligo")
    if btac_dir is not None:
        for path in sorted(btac_dir.glob("*.csv")):
            match = _RANK_RE_BTACSUB.match(path.name)
            if match:
                analysis = "Purkinje" if match.group(1) == "Purkinje" else "Oligo"
                rows.append(
                    dict(analysis=analysis, species=match.group(2),
                         period="all_development", path=path)
                )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"no ranked CSV found under {list(map(str, dirs.values()))}")

    frame["model"] = (
        frame.analysis + "__" + frame.species + "__" + frame.period
    )
    frame["celltype"] = frame.analysis.map(ANALYSIS_CELLTYPE)
    frame["org_celltype"] = frame.analysis.map(ANALYSIS_ORG_CELLTYPE)
    frame["n_ranked"] = [len(pd.read_csv(p)) for p in frame.path]
    frame["sha256"] = [sha256_file(p) for p in frame.path]
    frame["is_empty"] = frame.n_ranked == 0
    return frame.sort_values(["analysis", "species", "period"]).reset_index(drop=True)


def apply_species_filter(config: dict, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop models whose species has too few cells of the model's target type.

    Returns (kept manifest, audit rows). The audit frame names every dropped
    model with the cell count that caused the drop, so a reader can always see
    what was given up and why. Only applies when `species_filter` is configured;
    v1 has no such block and is left untouched.
    """
    filter_config = config.get("eligibility") or config.get("species_filter")
    if not filter_config:
        return manifest, pd.DataFrame(
            columns=["analysis", "species", "celltype", "n_cells", "min_cells", "action"]
        )

    counts = species_cell_counts(config, filter_config.get("celltype_column", "Org_Celltype"))
    lookup = counts.set_index(["species", "celltype"]).n_cells.to_dict()
    floor = min_cells_per_species(config)
    peak_floor = min_ranked_peaks(config)
    org = manifest.org_celltype.astype(str)

    audit_rows, keep = [], []
    for row, celltype in zip(manifest.itertuples(), org):
        n = int(lookup.get((row.species, celltype), 0))
        n_peaks = int(row.n_ranked)
        reasons = []
        if n < floor:
            reasons.append(f"{n} target cells < {floor}")
        if n_peaks < peak_floor:
            reasons.append(f"{n_peaks} ranked peaks < {peak_floor}")
        if reasons:
            audit_rows.append(dict(
                analysis=row.analysis, species=row.species, celltype=celltype,
                model=row.model, n_cells=n, min_cells=floor,
                n_ranked=n_peaks, min_ranked_peaks=peak_floor,
                action="dropped: " + "; ".join(reasons),
            ))
        else:
            keep.append(row.Index)
    kept = manifest.loc[keep].reset_index(drop=True)
    return kept, pd.DataFrame(audit_rows)


def load_rankings(config: dict) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return (manifest, {model: DataFrame(peak, score, rank) sorted by score desc}).

    Ties are broken by peak name so the ranking is deterministic and reproducible
    from the CSV alone.
    """
    manifest = ranking_files(config)
    rankings: dict[str, pd.DataFrame] = {}
    for row in manifest.itertuples():
        frame = pd.read_csv(row.path).rename(columns={"gene": "peak"})
        if frame.empty:
            rankings[row.model] = frame.assign(rank=pd.Series(dtype=int))
            continue
        if not frame.peak.is_unique:
            raise ValueError(f"duplicate peaks in {row.path}")
        if not np.isfinite(frame.score).all():
            raise ValueError(f"non-finite score in {row.path}")
        frame = frame.sort_values(["score", "peak"], ascending=[False, True]).reset_index(drop=True)
        frame["rank"] = np.arange(1, len(frame) + 1)
        rankings[row.model] = frame
    return manifest, rankings


def model_species_peaks(manifest: pd.DataFrame, rankings: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    """Per species, the union of the ranked universes across all its models."""
    out: dict[str, set[str]] = {}
    for row in manifest.itertuples():
        out.setdefault(row.species, set()).update(rankings[row.model].peak)
    return {species: sorted(peaks) for species, peaks in out.items()}


# --------------------------------------------------------------------------
# tiny io helpers
# --------------------------------------------------------------------------


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def display_path(path: Path) -> str:
    """Project-relative when inside the project, absolute otherwise.

    The artefact is already on disk by the time this is called, so a results
    directory configured outside the project tree must not turn a successful
    write into an exception.
    """
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def save_table(frame: pd.DataFrame, path: Path, **kwargs: object) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, **kwargs)  # type: ignore[arg-type]
    log("wrote", display_path(path), f"({len(frame)} rows)")
    return path


def _json_safe(value: object) -> object:
    """Make a payload strictly valid JSON.

    `json.dumps(default=str)` would turn a NaN or infinity into the bare tokens
    `NaN` / `Infinity`, which are not JSON and are rejected by a strict parser.
    A statistic that could not be computed is written as null instead, so that a
    downstream reader sees "missing" rather than a number that is not a number.
    """
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def save_json(payload: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_safe(payload), indent=2, allow_nan=False)
    path.write_text(text)
    log("wrote", display_path(path))
    return path


# --------------------------------------------------------------------------
# reference sequence extraction
# --------------------------------------------------------------------------


def parse_peak(peak: str, genome: SpeciesGenome) -> tuple[str, int, int]:
    """peak id -> (chromosome name as it appears in the FASTA, start, end)."""
    body = peak[len(genome.prefix):] if peak.startswith(genome.prefix) else peak
    match = re.fullmatch(r"([^:]+):(\d+)-(\d+)", body)
    if not match:
        raise ValueError(f"cannot parse peak coordinate: {peak}")
    chrom = match.group(1)
    if genome.chrom_style == "add_chr" and not chrom.startswith("chr"):
        chrom = "chr" + chrom
    return chrom, int(match.group(2)), int(match.group(3))


def fasta_sequence_names(fasta: Path) -> set[str]:
    names: set[str] = set()
    with gzip.open(fasta, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                names.add(line[1:].split()[0])
    return names


def read_sequences(
    config: dict,
    genome: SpeciesGenome,
    peaks: list[str],
    use_cache: bool = True,
) -> pd.DataFrame:
    """Extract peak sequences from the reference FASTA.

    Returns columns peak, chrom, start, end, sequence, valid, length, gc. Peaks
    whose sequence cannot be recovered keep an empty sequence and valid=False;
    they are never silently dropped, callers must decide.
    """
    cache = cache_dir(config) / f"{genome.species}_sequences.csv.gz"
    wanted = list(dict.fromkeys(peaks))
    if use_cache and cache.exists():
        cached = pd.read_csv(cache).fillna({"sequence": ""})
        if set(wanted) <= set(cached.peak):
            return cached[cached.peak.isin(set(wanted))].set_index("peak").loc[wanted].reset_index()

    by_chrom: dict[str, list[tuple[str, int, int]]] = {}
    coords: dict[str, tuple[str, int, int] | None] = {}
    for peak in wanted:
        try:
            chrom, start, end = parse_peak(peak, genome)
        except ValueError:
            coords[peak] = None
            continue
        coords[peak] = (chrom, start, end)
        by_chrom.setdefault(chrom, []).append((peak, start, end))

    sequences: dict[str, str] = {}
    current: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        if current is None or current not in by_chrom:
            return
        seq = "".join(chunks).upper()
        for peak, start, end in by_chrom[current]:
            if 0 <= start < end <= len(seq):
                sequences[peak] = seq[start:end]

    with gzip.open(genome.fasta, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                flush()
                current = line[1:].split()[0]
                chunks = []
            elif current is not None:
                chunks.append(line.strip())
        flush()

    rows = []
    for peak in wanted:
        seq = sequences.get(peak, "")
        coord = coords[peak]
        rows.append(
            dict(
                peak=peak,
                chrom=coord[0] if coord else "",
                start=coord[1] if coord else -1,
                end=coord[2] if coord else -1,
                sequence=seq,
                valid=bool(seq) and bool(re.fullmatch(r"[ACGT]+", seq)),
                length=len(seq),
                gc=(seq.count("G") + seq.count("C")) / len(seq) if seq else np.nan,
            )
        )
    frame = pd.DataFrame(rows)

    if use_cache:
        if cache.exists():
            old = pd.read_csv(cache).fillna({"sequence": ""})
            frame = pd.concat([old, frame], ignore_index=True).drop_duplicates("peak", keep="last")
        cache.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache, index=False)
    return frame


# --------------------------------------------------------------------------
# GC / length matched pairing
# --------------------------------------------------------------------------


def match_pairs(
    foreground: pd.DataFrame,
    pool: pd.DataFrame,
    rng: np.random.Generator,
    caliper: float = 0.03,
) -> pd.DataFrame:
    """One-to-one, without-replacement matching of foreground peaks to a pool.

    A foreground peak may only be paired with a pool peak of identical length
    and GC fraction within `caliper`. Ties on the closest GC are broken at
    random. Unmatched foreground peaks are reported in the returned frame with
    bg_peak = "" so that the loss is always visible.
    """
    pool = pool.reset_index(drop=True)
    available = np.ones(len(pool), dtype=bool)
    pool_length = pool.length.to_numpy()
    pool_gc = pool.gc.to_numpy(dtype=float)
    pool_name = pool.peak.to_numpy()

    order = foreground.sort_values("peak")
    rows = []
    for row in order.itertuples():
        candidates = np.flatnonzero(
            available & (pool_length == row.length) & (np.abs(pool_gc - row.gc) <= caliper)
        )
        if not len(candidates):
            rows.append(dict(fg_peak=row.peak, bg_peak="", gc_difference=np.nan, length=row.length))
            continue
        distance = np.abs(pool_gc[candidates] - row.gc)
        best = candidates[distance <= distance.min() + 1e-12]
        pick = int(best[rng.integers(len(best))])
        available[pick] = False
        rows.append(
            dict(
                fg_peak=row.peak,
                bg_peak=pool_name[pick],
                gc_difference=float(abs(row.gc - pool_gc[pick])),
                length=int(row.length),
            )
        )
    return pd.DataFrame(rows)


def matched_background(
    foreground: pd.DataFrame,
    pool: pd.DataFrame,
    rng: np.random.Generator,
    caliper: float = 0.03,
) -> tuple[list[str], pd.DataFrame]:
    """Matched background for a foreground set; returns (background peaks, pairs).

    Kept for callers that only need the background. For any Fisher test the
    foreground MUST be reduced to the matched subset as well, otherwise
    |foreground| != |background| and every log2_OR picks up a constant shift.
    Use `matched_foreground_background` instead.
    """
    pairs = match_pairs(foreground, pool, rng, caliper)
    background = [p for p in pairs.bg_peak if p]
    return background, pairs


def matched_foreground_background(
    foreground: pd.DataFrame,
    pool: pd.DataFrame,
    rng: np.random.Generator,
    caliper: float = 0.03,
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Equal-sized GC/length matched foreground and background.

    Returns (foreground peaks, background peaks, pairs). Only foreground peaks
    that found a partner survive, so the two lists always have the same length
    and the Fisher table is square. The number of dropped foreground peaks is
    len(foreground) - len(returned foreground) and must be audited by callers.
    """
    pairs = match_pairs(foreground, pool, rng, caliper)
    keep = pairs[pairs.bg_peak != ""]
    return keep.fg_peak.tolist(), keep.bg_peak.tolist(), pairs


def draw_disjoint_matched_sets(
    template: pd.DataFrame,
    pool: pd.DataFrame,
    rng: np.random.Generator,
    caliper: float = 0.03,
) -> tuple[list[str], list[str]]:
    """Two disjoint, equally sized, GC/length matched samples from `pool`.

    Each element of `template` contributes at most one peak to the foreground
    and at most one to the background, both drawn with the same length and GC
    caliper rules as the observed matching. Template elements that cannot
    contribute to both sides are dropped, so the two returned lists always have
    identical length. This is what makes a random-500 null comparable to an
    observed Top-500 enrichment: both sides of the Fisher table are matched and
    balanced.
    """
    first = match_pairs(template, pool, rng, caliper)
    first_ok = first[first.bg_peak != ""]
    if first_ok.empty:
        return [], []
    remaining = pool[~pool.peak.isin(set(first_ok.bg_peak))]
    second = match_pairs(template[template.peak.isin(set(first_ok.fg_peak))], remaining, rng, caliper)
    second_ok = second[second.bg_peak != ""]
    if second_ok.empty:
        return [], []
    shared = set(first_ok.fg_peak) & set(second_ok.fg_peak)
    fg = first_ok[first_ok.fg_peak.isin(shared)].sort_values("fg_peak").bg_peak.tolist()
    bg = second_ok[second_ok.fg_peak.isin(shared)].sort_values("fg_peak").bg_peak.tolist()
    return fg, bg


# --------------------------------------------------------------------------
# JASPAR motif catalog + FIMO scanning
# --------------------------------------------------------------------------


def motif_catalog(config: dict) -> pd.DataFrame:
    rows = []
    for line in motif_database(config).read_text().splitlines():
        if line.startswith("MOTIF "):
            parts = line.split()
            rows.append((parts[1], parts[2] if len(parts) > 2 else parts[1]))
    frame = pd.DataFrame(rows, columns=["motif_id", "motif_name"])
    expected = int(config["motif_database"]["n_motifs"])
    if len(frame) != expected:
        raise RuntimeError(f"expected {expected} motifs, found {len(frame)}")
    return frame


def _fimo_chunk(payload: tuple[str, str, str, str]) -> str:
    """Run FIMO on one FASTA chunk. Returns the output path."""
    fimo_bin, meme, fasta_path, out_path = payload
    with open(out_path, "w") as handle:
        subprocess.run(
            [fimo_bin, "--text", "--thresh", "1e-4", meme, fasta_path],
            stdout=handle,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    return out_path


def scan_motifs(
    config: dict,
    genome: SpeciesGenome,
    sequences: pd.DataFrame,
    workers: int = 16,
    force: bool = False,
) -> dict[str, set[str]]:
    """FIMO-scan every valid sequence of `sequences`, cached by content hash.

    Returns {motif_id: set(peak)}. The scan is performed only on peaks with a
    valid ACGT sequence; peaks without sequence are absent from every set, which
    is why the enrichment functions always take the evaluable foreground and
    background explicitly.
    """
    catalog = motif_catalog(config)
    threshold = float(config["motif_database"]["fimo_threshold"])
    if abs(threshold - 1e-4) > 1e-12:
        raise RuntimeError("FIMO cache is keyed to thresh=1e-4; update the cache key if changed")

    usable = sequences[sequences.valid].sort_values("peak").reset_index(drop=True)
    if usable.empty:
        return {}
    # FIMO truncates a FASTA header at the first colon, so peak ids such as
    # "hg38_chr1:817098-817598" would come back as "hg38_chr1" and every hit on
    # a chromosome would be attributed to a single pseudo-peak. Sequences are
    # therefore submitted under colon-free surrogate names and mapped back.
    surrogate = [f"seq{index:07d}" for index in range(len(usable))]
    usable = usable.assign(surrogate=surrogate)
    memo = "\n".join(f">{n}\n{s}" for n, s in zip(usable.surrogate, usable.sequence))
    memo += motif_database(config).read_text() + f"|thresh={threshold}|fimo={FIMO_BIN}|naming=seqNNNNNNN"
    fingerprint = sha256_text(memo)

    cache = cache_dir(config)
    hits_path = cache / f"{genome.species}_motif_peak_hits.csv.gz"
    stamp_path = cache / f"{genome.species}_scan.sha256"

    if not force and hits_path.exists() and stamp_path.exists() and stamp_path.read_text() == fingerprint:
        hits_frame = pd.read_csv(hits_path)
        log("motif scan cache hit", genome.species, f"{len(hits_frame)} motif-peak pairs")
        return {k: set(v) for k, v in hits_frame.groupby("motif_id").peak.apply(list).items()}
    if hits_path.exists() and stamp_path.exists() and stamp_path.read_text() != fingerprint:
        log("motif scan cache STALE, rescanning", genome.species)

    chunk_dir = cache / f"fimo_{genome.species}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = max(1, min(workers * 2, int(np.ceil(len(usable) / 250))))
    bounds = np.linspace(0, len(usable), n_chunks + 1).astype(int)

    jobs = []
    for index in range(n_chunks):
        part = usable.iloc[bounds[index]: bounds[index + 1]]
        if part.empty:
            continue
        fasta_path = chunk_dir / f"chunk_{index:03d}.fa"
        fasta_path.write_text("".join(f">{n}\n{s}\n" for n, s in zip(part.surrogate, part.sequence)))
        jobs.append(
            (FIMO_BIN, str(motif_database(config)), str(fasta_path), str(chunk_dir / f"chunk_{index:03d}.tsv"))
        )
    log("motif scan", genome.species, f"{len(usable)} peaks in {len(jobs)} chunks")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_fimo_chunk, jobs))

    lookup = dict(zip(usable.surrogate, usable.peak))
    hits: dict[str, set[str]] = {}
    seen_unknown: set[str] = set()
    for _, _, _, out_path in jobs:
        out = Path(out_path)
        for chunk in pd.read_csv(out, sep="\t", comment="#", chunksize=500_000):
            for motif_id, group in chunk.groupby("motif_id"):
                mapped = set()
                for name in group.sequence_name:
                    peak = lookup.get(name)
                    if peak is None:
                        seen_unknown.add(name)
                    else:
                        mapped.add(peak)
                hits.setdefault(motif_id, set()).update(mapped)
    if seen_unknown:
        raise RuntimeError(
            f"FIMO returned {len(seen_unknown)} sequence names that are not in the submitted set, "
            f"e.g. {sorted(seen_unknown)[:5]}; the surrogate-name mapping is broken"
        )
    for job in jobs:
        Path(job[2]).unlink(missing_ok=True)
        Path(job[3]).unlink(missing_ok=True)

    rows = [(motif_id, peak) for motif_id, peaks in hits.items() for peak in sorted(peaks)]
    frame = pd.DataFrame(rows, columns=["motif_id", "peak"])
    frame.to_csv(hits_path, index=False)
    stamp_path.write_text(fingerprint)
    log("motif scan done", genome.species, f"{len(frame)} motif-peak pairs")
    return hits


# --------------------------------------------------------------------------
# enrichment
# --------------------------------------------------------------------------


def enrichment(
    catalog: pd.DataFrame,
    hits: dict[str, set[str]],
    foreground: list[str] | set[str],
    background: list[str] | set[str],
) -> pd.DataFrame:
    """One-sided Fisher enrichment of a foreground against a matched background.

    Every motif of the catalog is reported, significant or not. log2_OR uses a
    Haldane-Anscombe style 0.5 correction so that zero cells stay finite.
    """
    fg = set(foreground)
    bg = set(background)
    n_fg, n_bg = len(fg), len(bg)
    if n_fg == 0 or n_bg == 0:
        raise ValueError("empty foreground or background")
    rows = []
    for motif_id, motif_name in zip(catalog.motif_id, catalog.motif_name):
        peak_set = hits.get(motif_id, set())
        a = len(peak_set & fg)
        b = len(peak_set & bg)
        p_value = fisher_exact([[a, n_fg - a], [b, n_bg - b]], alternative="greater")[1]
        odds = np.log2(
            ((a + 0.5) * (n_bg - b + 0.5)) / ((n_fg - a + 0.5) * (b + 0.5))
        )
        rows.append(
            dict(
                motif_id=motif_id,
                motif_name=motif_name,
                fg_hits=a,
                bg_hits=b,
                n_fg=n_fg,
                n_bg=n_bg,
                odds_ratio=float(2.0 ** odds),
                log2_OR=float(odds),
                pvalue=float(p_value),
            )
        )
    frame = pd.DataFrame(rows)
    frame["FDR"] = multipletests(frame.pvalue, method="fdr_bh")[1]
    return frame


def add_significance(frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    frame = frame.copy()
    frame["significant"] = (frame.FDR < alpha) & (frame.log2_OR > 0)
    return frame


# --------------------------------------------------------------------------
# matched random control
# --------------------------------------------------------------------------


def random_matched_foregrounds(
    top: pd.DataFrame,
    pool: pd.DataFrame,
    n_repeats: int,
    n_peaks: int,
    rng: np.random.Generator,
    caliper: float = 0.03,
) -> list[pd.DataFrame]:
    """Random foreground sets whose GC/length distribution matches `top`.

    Each repeat draws `n_peaks` peaks from `pool` through the same one-to-one
    GC/length matching used for the observed background, so the only thing that
    differs from the real Top-k set is *which* peaks the model ranked highest.
    """
    if len(pool) < n_peaks:
        raise ValueError("pool smaller than the requested random peak count")
    template = top[["peak", "length", "gc"]].copy()
    if len(template) > n_peaks:
        template = template.sample(n=n_peaks, random_state=int(rng.integers(2**31)))
    template = template.sort_values("peak").reset_index(drop=True)
    draws = []
    for _ in range(n_repeats):
        pairs = match_pairs(template, pool, rng, caliper)
        draws.append(pd.DataFrame({"peak": [p for p in pairs.bg_peak if p]}))
    return draws


def _serve_one(
    candidates: np.ndarray, distances: np.ndarray, available: np.ndarray, rng
) -> int | None:
    """Place one free pool peak for one template peak, by the project's rule.

    The candidate set is the free pool peaks of identical length within the GC
    caliper; among those the ones at the smallest GC distance compete, and one is
    taken uniformly at random. This is `match_pairs`'s rule exactly, which is why
    the observed matching and this draw stay comparable.
    """
    free = available[candidates]
    if not free.any():
        return None
    usable = candidates[free]
    distance = distances[free]
    nearest = usable[distance <= distance.min() + 1e-12]
    pick = int(nearest[rng.integers(len(nearest))])
    available[pick] = False
    return pick


def draw_resampled_matched_sets(
    template: pd.DataFrame,
    pool: pd.DataFrame,
    rng: np.random.Generator,
    caliper: float,
    target: int,
    max_rounds: int = 40,
) -> tuple[list[str], list[str]]:
    """Two disjoint, equally sized, GC/length matched draws of exactly `target` pairs.

    Each template peak is asked for TWO distinct pool peaks at once, and template
    peaks are served rarest-candidate-first so the peaks with the fewest possible
    partners are not starved by the abundant ones. When the template is exhausted
    short of `target`, further template peaks are drawn WITH REPLACEMENT from the
    template's own (length, GC) profile and served too, so the drawn profile
    still tracks the Top-K profile while the requested size stays reachable.
    Returns short only when the pool genuinely cannot supply the draw.

    WHY NOT the two simpler drawers
    -------------------------------
    `draw_disjoint_matched_sets` runs two name-ordered greedy passes and, in its
    own words, "routinely finishes a few peaks short" -- it cannot hit a size at
    all. `s15_rank_curve`'s allocator (one rarest-first pass over the template,
    no resampling) can, but MEASURED on this data it reaches the exact target for
    only 3 of 21 models at Top500 and 0 of 21 at Top1000 / Top2000: the template's
    GC tails have no second partner and the pass simply ends. That is also why
    s15 itself carries a `MIN_DRAW_FRACTION = 0.9` floor, with the comment that
    "demanding equality discarded all 200 repeats for 15 model x cut-off cells
    and left them with no band at all". Resampling the template removes that
    dependence on the tails while leaving the pairing rule untouched.

    The PAIRING RULE is the observed one, unchanged: one-to-one, without
    replacement, identical length, GC within `caliper`, ties on the closest GC
    broken uniformly at random, with the foreground matched to the template peak
    and the background matched to the FOREGROUND -- so `|fg_gc - bg_gc| <= caliper`
    exactly as in `matched_foreground_background` and in
    `draw_disjoint_matched_sets`. Only the pool-allocation ORDER is local, and
    order is not part of the statistical rule. (Matching both partners against the
    template is the tempting shorthand and is WRONG: it lets the gap reach
    `2 * caliper`. See the measured comparison in `serve` below.)

    There is a hard arithmetic floor: `target` pairs need `2 * target` distinct
    pool peaks, and the pool is the model's own non-top-k peaks. A model whose
    pool is smaller than that can NEVER produce a valid repeat -- reported through
    `n_dropped`/`max_shortfall`, never papered over by accepting a smaller draw.
    """
    pool_length = pool.length.to_numpy()
    pool_gc = pool.gc.to_numpy(dtype=float)
    pool_name = pool.peak.to_numpy()
    tpl_length = template.length.to_numpy()
    tpl_gc = template.gc.to_numpy(dtype=float)

    # A partner window is resolved by bisection on a per-length, GC-sorted index,
    # so the second pick can be centred on the FIRST pick rather than on the
    # template. Scanning the pool linearly for that (the obvious implementation)
    # makes the resample loop quadratic on a 2,000-peak template.
    by_length: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for length in np.unique(pool_length):
        index = np.flatnonzero(pool_length == length)
        ordered = index[np.argsort(pool_gc[index], kind="stable")]
        by_length[float(length)] = (pool_gc[ordered], ordered)

    def window(length: float, gc: float) -> tuple[np.ndarray, np.ndarray]:
        entry = by_length.get(float(length))
        if entry is None:
            return np.empty(0, dtype=int), np.empty(0, dtype=float)
        gcs, ordered = entry
        lo = int(np.searchsorted(gcs, gc - caliper, side="left"))
        hi = int(np.searchsorted(gcs, gc + caliper, side="right"))
        candidates = ordered[lo:hi]
        return candidates, np.abs(pool_gc[candidates] - gc)

    candidates: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    for length, gc in zip(tpl_length, tpl_gc):
        index, distance = window(length, gc)
        candidates.append(index)
        distances.append(distance)

    available = np.ones(len(pool), dtype=bool)
    foreground: list[str] = []
    background: list[str] = []

    def serve(row: int) -> bool:
        # TWO PASSES, mirroring the observed rule rather than approximating it:
        # the foreground is a pool peak within `caliper` of the TEMPLATE peak (the
        # observed foreground IS the Top-K peak, so this is its analogue), and the
        # background is then a pool peak within `caliper` of the FOREGROUND.
        #
        # Matching both against the template instead -- the obvious shorthand --
        # lets |fg_gc - bg_gc| reach 2 * caliper and breaks the one property this
        # null exists to have. Measured on marmoset Astroglia at Top2000:
        #   observed matched_foreground_background : max |fg-bg| = 0.0300 (= caliper)
        #   v2's draw_disjoint_matched_sets        : max |fg-bg| = 0.0300
        #   matching both against the template     : max |fg-bg| = 0.0520  <-- breaks
        pick_fg = _serve_one(candidates[row], distances[row], available, rng)
        if pick_fg is None:
            return False
        second, distance = window(pool_length[pick_fg], pool_gc[pick_fg])
        pick_bg = _serve_one(second, distance, available, rng)
        if pick_bg is None:
            available[pick_fg] = True
            return False
        foreground.append(str(pool_name[pick_fg]))
        background.append(str(pool_name[pick_bg]))
        return True

    order = sorted(range(len(candidates)), key=lambda i: (len(candidates[i]), i))
    for row in order:
        if len(foreground) >= target:
            break
        serve(row)

    rounds = 0
    while len(foreground) < target and rounds < max_rounds:
        rounds += 1
        for row in rng.permutation(len(candidates)):
            if len(foreground) >= target:
                break
            serve(int(row))
    return foreground, background


@dataclass(frozen=True)
class NullDraw:
    """What the matched-random null produced for one model, including its misses.

    `n_valid` is the number of repeats that entered the null. Under the v3 exact
    contract a repeat is valid only when its draw placed exactly `target` peaks on
    both sides; `n_dropped` and `max_shortfall` are carried so that a model whose
    pool cannot support the draw is visible as an underpowered null rather than as
    a silently thinner one.

    `target` and `n_peaks` are DIFFERENT numbers and the distinction is the whole
    point: `n_peaks` is the nominal Top-K control size (the arm's cut-off), while
    `target` is the width of the OBSERVED Fisher table -- `n_fg_matched`, how many
    of the model's Top-K peaks actually found a matched partner, which is
    regularly below `n_peaks`. The exactness is enforced against `target`.
    """

    n_requested: int
    n_valid: int
    n_dropped: int
    n_peaks: int
    target: int
    min_valid_repeats: int
    shortfalls: tuple[int, ...]
    exact: bool

    @property
    def max_shortfall(self) -> int:
        return max(self.shortfalls) if self.shortfalls else 0

    def status(self) -> str:
        if self.n_valid == 0:
            return "no_draw"
        if self.min_valid_repeats and self.n_valid < self.min_valid_repeats:
            return "underpowered"
        return "ok"


def random_null_enrichment(
    config: dict,
    catalog: pd.DataFrame,
    hits: dict[str, set[str]],
    top: pd.DataFrame,
    pool: pd.DataFrame,
    rng: np.random.Generator,
    n_repeats: int | None = None,
    n_peaks: int | None = None,
    caliper: float = 0.03,
    target: int | None = None,
) -> tuple[pd.DataFrame, list[list[str]], NullDraw]:
    """Matched random null of the model motif profile, at a fixed size.

    For every repeat: a matched random foreground is drawn from the model's own
    non-Top-k pool, and its matched background is drawn from what is left. The
    resulting log2_OR profile is directly comparable to the observed Top-k
    profile because it was produced by exactly the same enrichment call.

    THE SIZE CONTRACT (`random_control.require_exact_n_peaks`, v3)
    -------------------------------------------------------------
    The observed Fisher table is NOT K peaks wide. `K` is the ranking cut-off;
    the table's width is `n_fg_matched`, how many of the model's Top-K peaks
    actually found a GC/length-matched partner -- median 97.8% / 98.6% / 94.9% of
    K at Top500/1000/2000, worst case 82% / 66% / 73%. A null drawn at K would
    therefore differ from the observation by sampling depth as well as by chance,
    which is the thing being controlled. So the caller passes the observed table's
    own width as `target`, and every emitted repeat has EXACTLY that width on both
    sides. A repeat that comes up short is dropped and counted, never accepted at
    a smaller size.

    `n_peaks` stays the nominal Top-K control size and is what the emitted table
    records as `n_peaks_requested`, so a consumer comparing against
    `random_control.n_peaks` still sees the cut-off an arm ran at.

    Without the flag the v1/v2 rule applies (`len(fg) >= min_peaks_per_repeat`),
    which is what leaves a v1-style run byte-identical.

    An empty result is NOT an error. A model whose pool cannot place the draw is
    reported through the returned `NullDraw` so the caller can mark it
    NULL_UNDERPOWERED; aborting here would take the whole analysis down for one
    model's arithmetic.
    """
    control = config["random_control"]
    n_repeats = int(control["n_repeats"] if n_repeats is None else n_repeats)
    n_peaks = int(control["n_peaks"] if n_peaks is None else n_peaks)
    minimum = int(control.get("min_peaks_per_repeat", max(50, n_peaks // 2)))
    exact = bool(control.get("require_exact_n_peaks", False))
    min_valid = int(control.get("min_valid_repeats", 0))

    template = top[["peak", "length", "gc"]].copy()
    if len(template) > n_peaks:
        template = template.sample(n=n_peaks, random_state=int(rng.integers(2**31)))
    template = template.sort_values("peak").reset_index(drop=True)

    draw_size: int | None = None
    if exact:
        nominal = n_peaks if target is None else int(target)
        draw_size = int(min(nominal, n_peaks, len(template)))

    rows = []
    peak_sets: list[list[str]] = []
    shortfalls: list[int] = []
    skipped = 0
    for repeat in range(n_repeats):
        if exact:
            fg, background = draw_resampled_matched_sets(
                template, pool, rng, caliper, draw_size
            )
        else:
            fg, background = draw_disjoint_matched_sets(template, pool, rng, caliper)
        if len(background) != len(fg):
            skipped += 1
            shortfalls.append(min(len(fg), len(background)))
            continue
        if exact:
            if len(fg) != draw_size:
                skipped += 1
                shortfalls.append(len(fg))
                continue
        elif len(fg) < minimum:
            skipped += 1
            shortfalls.append(len(fg))
            continue
        frame = enrichment(catalog, hits, fg, background)
        frame["repeat"] = repeat
        frame["n_fg"] = len(fg)
        frame["n_bg"] = len(background)
        rows.append(frame)
        peak_sets.append(fg)

    draw = NullDraw(
        n_requested=n_repeats,
        n_valid=len(rows),
        n_dropped=skipped,
        n_peaks=n_peaks,
        target=int(draw_size if draw_size is not None else len(template)),
        min_valid_repeats=min_valid,
        shortfalls=tuple(shortfalls),
        exact=exact,
    )
    if skipped:
        rule = f"exact at {draw_size}" if exact else f"min {minimum}"
        log(
            f"random null ({rule}): {skipped}/{n_repeats} repeats dropped, "
            f"{len(rows)} valid"
            + (f", worst draw {draw.max_shortfall}" if exact and shortfalls else "")
        )
    frame = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["motif_id", "motif_name", "fg_hits", "bg_hits",
                                   "odds_ratio", "log2_OR", "pvalue", "FDR",
                                   "significant", "repeat", "n_fg", "n_bg"])
    )
    return frame, peak_sets, draw


# --------------------------------------------------------------------------
# family aggregation
# --------------------------------------------------------------------------


def family_trajectory(
    profile: pd.DataFrame,
    family_table: pd.DataFrame,
    value: str = "log2_OR",
    motif_col: str = "motif_id",
) -> pd.DataFrame:
    """Per-motif species baseline removal, then median per family.

    E*_{s,t,m} = E_{s,t,m} - mean_t(E_{s,t,m}) within a species, then
    F_{s,t,f} = median over the monomeric members of family f.
    """
    annotations = family_table[["motif_id", "family", "analysis_level", "member_type"]]
    frame = profile.merge(annotations, on="motif_id", how="left")
    frame = frame[frame.family.notna() & (frame.family != "")].copy()
    if frame.empty:
        raise ValueError("no motif of the profile is annotated with a family")

    baseline = frame.groupby(["species", motif_col])[value].transform("mean")
    frame["centered"] = frame[value] - baseline

    monomers = frame[frame.member_type == "monomer"]
    grouped = monomers.groupby(["species", "stage", "family"], as_index=False)
    out = grouped.agg(
        family_score=("centered", "median"),
        n_members=("centered", "size"),
    )
    levels = monomers.drop_duplicates("family").set_index("family").analysis_level
    out["analysis_level"] = out.family.map(levels)
    if "significant" in monomers.columns:
        sig = grouped["significant"].sum().rename(columns={"significant": "n_significant"})
        out = out.merge(sig, on=["species", "stage", "family"], how="left")
    return out


# --------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------


def spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    result = spearmanr(a, b)
    return float(result.statistic), float(result.pvalue)


def similarity_matrix(profiles: dict[str, np.ndarray]) -> pd.DataFrame:
    keys = list(profiles)
    matrix = np.eye(len(keys))
    pvalues = np.ones((len(keys), len(keys)))
    for i, key_i in enumerate(keys):
        for j, key_j in enumerate(keys):
            if j <= i:
                continue
            rho, p_value = spearman(profiles[key_i], profiles[key_j])
            matrix[i, j] = matrix[j, i] = rho
            pvalues[i, j] = pvalues[j, i] = p_value
    return pd.DataFrame(matrix, index=keys, columns=keys)


def matching_margin(matrix: np.ndarray, labels: list[str]) -> tuple[float, float, float]:
    """Mean of the matching diagonal minus mean of the off-diagonal cells."""
    n = len(labels)
    diagonal = np.diag(matrix)
    mask = ~np.eye(n, dtype=bool)
    return float(diagonal.mean()), float(matrix[mask].mean()), float(diagonal.mean() - matrix[mask].mean())


def permutation_margin_pvalue(matrix: np.ndarray, labels: list[str]) -> tuple[float, float, float, float]:
    """Exact permutation p for the matching margin, over ALL n! relabellings.

    Row labels are fixed and the column labels are permuted through every one of
    the n! orderings, the margin being recomputed each time.

    The p-value is the exhaustive-enumeration form p = #{margin_perm >= observed}
    / n!. The identity ordering IS one of the enumerated permutations and is
    counted in the numerator -- that is what makes the test exact -- but it is
    NOT double counted by an added one, because no permutation is sampled: every
    ordering is evaluated. The smallest attainable value is therefore 1/n! =
    1/6 = 0.167 for three stages, and it is attained only when the observed
    margin is the maximum over all orderings.

    (Do not confuse this with the (1 + #{...}) / (1 + n) convention, which is
    the Monte-Carlo correction for a randomly sampled subset of permutations and
    only inflates an already exhaustive p-value.)

    Returns (observed, p_exact, min_over_permutations, n_permutations).
    """
    from itertools import permutations

    n = len(labels)
    all_perm = list(permutations(range(n)))
    observed = matching_margin(matrix, labels)[2]
    null = np.array([matching_margin(matrix[:, list(p)], labels)[2] for p in all_perm])
    p_value = float(np.sum(null >= observed) / len(null))
    return observed, p_value, float(null.min()), len(null)


def bootstrap_or_permutation_note() -> str:
    return "exact permutation over label orderings; no bootstrap of edges"


# --------------------------------------------------------------------------
# h5ad streaming
# --------------------------------------------------------------------------


def decode(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in "OS":
        return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values], object)
    return values


def read_obs_column(obs: "object", name: str) -> np.ndarray:
    import h5py

    item = obs[name]
    if isinstance(item, h5py.Group):
        codes = item["codes"][:]
        categories = read_obs_column(item, "categories")
        out = np.full(len(codes), "", object)
        valid = codes >= 0
        out[valid] = categories[codes[valid]]
        return out
    return decode(item[:])


def obs_frame(path: Path) -> tuple[pd.DataFrame, str]:
    """Read the full obs table of an h5ad as a DataFrame, plus the index name."""
    import h5py

    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        index_name = obs.attrs["_index"]
        index_name = index_name.decode() if isinstance(index_name, bytes) else str(index_name)
        columns = {}
        for name in obs.keys():
            if name == index_name:
                continue
            values = read_obs_column(obs, name)
            if values.ndim == 1:
                columns[name] = values
        frame = pd.DataFrame(columns)
        frame.index = pd.Index(read_obs_column(obs, index_name), name=index_name)
        return frame, index_name


def sample_id_from_cell_id(cell_id: str) -> str:
    """Library/donor unit encoded in the cell id: everything before the barcode."""
    return cell_id.split("#", 1)[0]


def iter_matrix_blocks(path: Path, block_cells: int = 2000):
    """Yield (row_offset, csr_matrix, n_var) blocks of the raw X of an h5ad.

    X is assumed to be a CSR-encoded int count matrix in the canonical 10x
    layout. The caller streams, so memory stays bounded by one block.
    """
    import h5py
    from scipy import sparse

    with h5py.File(path, "r") as handle:
        group = handle["X"]
        if not isinstance(group, h5py.Group):
            raise RuntimeError(f"{path}: X is dense, streaming reader expects sparse")
        shape = tuple(int(x) for x in group.attrs["shape"])
        n_obs, n_var = shape
        indptr = group["indptr"]
        data = group["data"]
        indices = group["indices"]
        for start in range(0, n_obs, block_cells):
            stop = min(start + block_cells, n_obs)
            lo = int(indptr[start])
            hi = int(indptr[stop])
            block = sparse.csr_matrix(
                (data[lo:hi], indices[lo:hi], indptr[start: stop + 1] - lo),
                shape=(stop - start, n_var),
            )
            yield start, block, n_var


def var_index(path: Path) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as handle:
        var = handle["var"]
        name = var.attrs["_index"]
        name = name.decode() if isinstance(name, bytes) else str(name)
        return read_obs_column(var, name)


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------


def add_common_args(parser: "object") -> None:
    parser.add_argument("--config", default=str(CONFIG_DIR / "analysis.yaml"))
    parser.add_argument("--repeats", type=int, default=None, help="override random_control.n_repeats")
    parser.add_argument("--topk", type=int, default=None, help="override model_analysis.primary_topk")
    parser.add_argument("--force", action="store_true", help="ignore caches")


def main_guard() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError("python 3.10+ required")
