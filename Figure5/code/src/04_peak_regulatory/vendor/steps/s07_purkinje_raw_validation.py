"""Purkinje raw chromVAR validation (readme section 26, gate P3).

WHY this step exists. `s06` can only show that the model *says* the Purkinje
top-ranked peaks carry a recurrent regulatory program. This step asks an
independent measurement -- chromVAR deviations over every QC peak of the species,
per pseudobulk -- the same question: does raw accessibility place the same motifs
in the same direction inside Purkinje cells relative to the other cerebellar cell
types?

WHAT gate P3 is, and what it is not. Gate P3 is the primary validation gate of the
Purkinje claim and is decided on PRIMARY evidence only: within each species, the
Spearman rho between the model's per-motif log2_OR and the raw per-motif C over the
full model/raw motif intersection (`model_vs_raw_similarity.tsv`), required to be
positive in at least GATE_MIN_SPECIES_AGREE of the species the gate is read over.
That is exactly the `G2` gate V2_PLAN section 15-2 freezes for C4 -- "the
Spearman over the 879 motifs, never over any module aggregate" -- and V2_PLAN's
test_module_layer_invariance.py is why it has to be read this way: a gate whose
value can be moved by editing the module map is a gate the interpretation layer
decides, and the frozen design forbids that. The module layer's own reading of the
same evidence (is there a recurrent primary module whose raw module score carries
the model's sign in enough species?) is reported in full as an INTERPRETATION
payload (`summary["gate_p3"]`, `module_raw_support.tsv`, the figure), labelled
`layer: interpretation, primary_evidence: false`, and it never sets `gates.P3`,
this step's status, or anything else a claim is decided on. Pointing `modules.file`
at a map with no rows therefore changes only which modules are named and whether
that reading is available -- the profile, the similarity table, the gate string and
the status all stay byte-identical.

WHY every comparison is within species. chromVAR Z is produced per species on
that species' own peak universe, its own GC background and its own pseudobulk
set. A Z of +2 in mouse and a Z of +2 in human are therefore not on a common
scale, and readme section 26 forbids comparing raw strength across species.
Only the rank correlation between the model profile and the raw profile,
computed separately inside each species, is interpretable.

WHY the "other celltype" average is equal-weighted by celltype. Purkinje is one
population among several with very different cell numbers; weighting by cells
would let whichever other population was sequenced most dominate the contrast
and turn the statistic into a comparison against that one population instead of
against "the rest of the cerebellum".

WHY both sides of the MODULE READING are read uncentered (the earlier fix, kept
verbatim here for the interpretation payload). The reading asks whether raw
accessibility places a model-defined recurrent module in the same direction as the
model does, so the raw module score -- the median C over the module's monomeric
members -- has to be the direct analogue of the model module score, which is
`s06`'s per-species median member log2_OR and carries no cross-species centering.
Subtracting a per-motif baseline from one side only makes the two signs
incomparable: a program that is uniformly Purkinje-elevated in every species
centers to exactly zero on the raw side and would be reported as disagreeing with
a model that calls it positive everywhere.

The cross-species centered module scores are still computed for both sides --
that is readme section 11 applied to the species axis, and it is what
`common.family_trajectory` does when its `species` and `stage` roles are swapped
for a single-stage (single-period) analysis -- and reported as the supplementary
`sign_agree_centered` reading, which asks whether a module's cross-species
*variation* tracks the model's. It is never the module reading's verdict, for the
reason above. Both bases are stated in `module_score_basis` and in
`gate_p3.comparison_basis` of the summary, so the reading is self-describing.

WHY the module panel is the PURKINJE module panel and not a family map. Module
support here is the cell-type-specific Purkinje panel declared in the config's
`modules.file` (`config/celltype_motif_module_map.tsv`), not the project-wide
`config/tf_family_map.tsv` that v1 forced on all three analyses. The panel is the
same one `s06` reports: the atomic Purkinje modules (Homeodomain_nonPOU and POU,
primary, plus TFAP2, primary, plus the empirical secondary NF-Y and RFX) and the
config-declared derived aggregate `Homeodomain_total`. Members are ALWAYS read
through `common.module_members`, never re-derived from the map, and the aggregate
is the UNION of its two atomic parts, so no motif is ever counted twice. A
module's level decides whether it may be read as agreement: at `secondary` (or
`comparator`) it is reported and can never set `p3_support`. This is an
INTERPRETATION layer over the unbiased 879-motif result; it never changes a
primary number -- the raw profile and the model-vs-raw Spearman table are computed
over the full motif intersection and never consult the module map.

WHY the species set is the filtered/eligible manifest. The models this step
validates are the ranked Purkinje models, and *which* of them may be validated is
decided by the config's eligibility block. The v2 config declares
`eligibility: {min_target_cells: 100, min_ranked_peaks: 500}`; the v1 config
declared `species_filter` instead. Both are read through
`common.apply_species_filter`, which enforces BOTH floors. Under v2 that removes
bonobo Purkinje (4 cells AND 298 ranked peaks), so the gate is read over five
species. The set is always derived from the ranked manifest -- never from a
literal list -- and a species that has a model but no usable raw profile, or a raw
profile but no model, is named with its reason in purkinje_raw_audit.tsv, in the
summary, and as an excluded row of model_vs_raw_similarity.tsv rather than
quietly missing from the Spearman table.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

STEP = "07_purkinje_validation"
ANALYSIS = "Purkinje"
TARGET_CELLTYPE = C.ANALYSIS_ORG_CELLTYPE[ANALYSIS]
# The pre-registered gate P3 threshold: how many species must agree on the sign of
# the model<->raw reading. It is the frozen readme value and NOT a count of species:
# the number of species the gate is evaluated over is derived at runtime from the
# filtered manifest, so the same constant reads "4 of 6 species" under v1 and "4 of 5
# species" under v2 (bonobo's 4 Purkinje cells / 298 ranked peaks fail eligibility).
GATE_MIN_SPECIES_AGREE = 4
# The Purkinje panel is small (five atomic modules plus one declared aggregate);
# the figure shows every module, gating ones first.
MAX_MODULE_PANELS = 6
# Where the upstream step writes the model's own module recurrence.
RECURRENCE_STEP_DIR = "06_purkinje_model"
RECURRENCE_PATTERNS = ("recurrent_*.tsv", "*recurrence*.tsv")
# How a row of module_raw_support.tsv says where its model module score came from.
RECOMPUTED_SCORE_SOURCE = "recomputed_from_01_model_motif_readme_section_11"

PROFILE_COLUMNS = [
    "species", "motif_id", "motif_name", "log2_OR", "C", "c_simple_all",
    "purkinje_mean_z", "other_equal_weighted_mean_z",
    "purkinje_mean_z_simple", "other_equal_weighted_mean_z_simple",
    "n_purkinje_pseudobulk", "n_purkinje_cells",
    "n_other_celltypes", "n_other_pseudobulk", "n_other_cells",
]
SIMILARITY_COLUMNS = [
    "species", "rho", "p", "n_motifs", "n_pseudobulk_target", "n_pseudobulk_other",
    "target_cells", "other_cells", "n_celltypes_other", "rho_simple", "p_simple",
    "p_fdr_bh", "status",
]
# One row per (module, species). `may_gate` is the module layer's only authority:
# a module below `primary` is reported and can never set `p3_support`.
MODULE_COLUMNS = [
    "module", "analysis_level", "may_gate", "is_recurrent", "species",
    "raw_module_score", "raw_module_score_centered",
    "model_module_score", "model_module_score_centered",
    "sign_agree", "sign_agree_centered",
    "n_members_raw", "n_members_model",
    "n_species_agree", "n_species_agree_centered", "n_species_scored",
    "p3_support", "model_score_source",
]
AUDIT_COLUMNS = ["species", "kind", "item", "n_cells", "n_pseudobulk", "detail", "status"]

# Reference data-viz palette: the diverging pair encodes the SIGN of a module
# score (never module identity, which is carried by position and by text).
PALETTE = {
    "positive": "#2a78d6",
    "negative": "#e34948",
    "ink": "#0b0b0b",
    "ink_muted": "#52514e",
    "grid": "#d8d7d2",
    "surface": "#fcfcfb",
}


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------


class Audit:
    """Every skip, drop or substitution gets a row; nothing is dropped in silence."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(
        self,
        species: str,
        kind: str,
        item: str,
        detail: str = "",
        n_cells: object = "",
        n_pseudobulk: object = "",
        status: str = "recorded",
    ) -> None:
        self.rows.append(
            dict(
                species=species, kind=kind, item=item, n_cells=n_cells,
                n_pseudobulk=n_pseudobulk, detail=detail, status=status,
            )
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=AUDIT_COLUMNS)


# --------------------------------------------------------------------------
# the cell-type-specific module panel
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModuleSpec:
    """One row of the Purkinje panel: an atomic module or a declared aggregate."""

    module: str
    level: str          # the map's level; "primary" / "secondary" / "comparator"
    may_gate: bool      # level is gating AND, for an aggregate, every part is
    members: tuple[str, ...]   # monomeric motif ids, deduplicated
    parts: tuple[str, ...]     # atomic modules this is built from ("" for atomic)
    derived: bool


def load_module_panel(config: dict, audit: Audit) -> dict[str, ModuleSpec]:
    """The Purkinje module panel, read through the common.py helpers only.

    Atomic modules come from `common.celltype_module_table`; their members come
    from `common.module_members` (monomers only -- a module median must be built
    from binding models of one factor, not from a heterodimer fusion). The
    aggregates are the config's own derived unions (`modules.aggregates`): their
    members are the UNION of their parts' members, so `Homeodomain_total` is the
    sum of `Homeodomain_nonPOU` and `POU` without any motif being counted twice.

    A panel that cannot be read is reported with a reason and returns empty; the
    primary tables of this step (the raw profile and the model-vs-raw similarity,
    both computed over the unbiased motif intersection) are written either way.
    """
    try:
        table = C.celltype_module_table(config)
    except Exception as exception:  # a missing/empty map must not take the numbers down
        audit.add(
            "-", "module_panel", "celltype_motif_module_map",
            detail=f"the module map could not be read: {exception!r}", status="blocked",
        )
        return {}
    sub = table[table.celltype.astype(str) == ANALYSIS]
    if sub.empty:
        audit.add(
            "-", "module_panel", ANALYSIS,
            detail="the module map carries no row for this cell type", status="blocked",
        )
        return {}

    panel: dict[str, ModuleSpec] = {}
    for module in C.modules_for(config, ANALYSIS):
        members = tuple(C.module_members(config, ANALYSIS, module, monomers_only=True))
        levels = sorted(set(sub.loc[sub.module == module, "level"].astype(str)))
        level = levels[0] if len(levels) == 1 else "derived_mixed"
        if len(levels) > 1:
            audit.add(
                "-", "module_panel", module,
                detail=f"the map gives this module more than one level: {levels}",
                status="substituted",
            )
        if not members:
            audit.add(
                "-", "module_panel", module,
                detail="no monomeric member survives in the map", status="excluded",
            )
        panel[module] = ModuleSpec(
            module=module, level=level, may_gate=C.module_is_gating(config, level),
            members=members, parts=(), derived=False,
        )

    for aggregate, parts in C.module_aggregates(config, ANALYSIS).items():
        parts = [str(part) for part in parts]
        missing = [part for part in parts if part not in panel]
        if missing:
            audit.add(
                "-", "module_panel", aggregate,
                detail=(f"declared aggregate of {parts}, but {missing} are not atomic modules "
                        f"of the {ANALYSIS} panel; the aggregate is not reported"),
                status="excluded",
            )
            continue
        seen: dict[str, None] = {}
        for part in parts:
            for member in panel[part].members:
                seen.setdefault(member, None)
        levels = sorted({panel[part].level for part in parts})
        level = levels[0] if len(levels) == 1 else "derived_mixed"
        panel[aggregate] = ModuleSpec(
            module=aggregate, level=level,
            may_gate=(
                C.module_is_gating(config, level)
                and all(panel[part].may_gate for part in parts)
            ),
            members=tuple(sorted(seen)), parts=tuple(parts), derived=True,
        )
        audit.add(
            "-", "module_panel", aggregate,
            detail=(f"derived union of {parts}: {len(panel[aggregate].members)} monomeric members, "
                    f"level={level}, may_gate={panel[aggregate].may_gate}"),
            status="ok",
        )
    return panel


def panel_annotation(panel: dict[str, ModuleSpec]) -> pd.DataFrame:
    """The panel as the annotation frame `common.family_trajectory` consumes.

    That helper is generic median-over-members aggregation; it happens to be
    written against a column called `family`, so the module name is carried in
    that column. The VALUES are module names of the Purkinje panel.
    """
    rows = [
        dict(motif_id=member, family=spec.module, analysis_level=spec.level,
             member_type="monomer")
        for spec in panel.values()
        for member in spec.members
    ]
    return pd.DataFrame(rows, columns=["motif_id", "family", "analysis_level", "member_type"])


# --------------------------------------------------------------------------
# species set
# --------------------------------------------------------------------------


def _display_order(names: Iterable[str]) -> list[str]:
    """Order species for display: the shared canonical order first, then the rest.

    Only the ORDER comes from the shared constant. The SET of species this step
    analyses is always derived from the ranked manifest through
    `common.apply_species_filter` (see `load_species_set`), never from
    `common.SPECIES_ORDER`, which is the fixed roster of the wider project and
    says nothing about which species may enter *this* analysis.
    """
    order = {name: index for index, name in enumerate(C.SPECIES_ORDER)}
    unique = list(dict.fromkeys(str(name) for name in names))
    return sorted(unique, key=lambda name: (order.get(name, len(order)), name))


@dataclass
class SpeciesSet:
    """Who this step analyses, and who it will not analyse, and why."""

    expected: list[str]           # species the eligibility filter kept: the analysis set
    declared: list[str]           # species that have a ranked Purkinje model
    dropped: dict[str, str]       # species -> the reason the filter dropped it
    source: str                   # where the ranked model manifest was read from
    min_cells: int                # eligibility.min_target_cells
    min_ranked_peaks: int         # eligibility.min_ranked_peaks
    celltype_column: str          # eligibility.celltype_column
    block: str                    # the config block that supplied the rule
    configured: bool              # does the config declare an eligibility block at all
    error: str = ""               # non-empty when the manifest could not be read

    @property
    def n_expected(self) -> int:
        return len(self.expected)


def _eligibility_block(config: dict) -> dict:
    """The species-inclusion rule, under whichever key the config declares it.

    v2 renamed the v1 `species_filter` key to `eligibility`; `common` accepts and
    enforces either, so this step reads the same block through the same helpers
    and never keeps a second copy of the rule.
    """
    return config.get("eligibility") or config.get("species_filter") or {}


def load_species_set(config: dict, audit: Audit) -> SpeciesSet:
    """Derive this step's species set from the eligibility-filtered ranked manifest.

    The candidate set is the ranked model manifest -- the models that were
    actually built -- restricted to `ANALYSIS`, then passed through
    `common.apply_species_filter`: a species keeps its model only if it has at
    least `min_target_cells` cells of the target celltype in the integrated h5ad
    AND at least `min_ranked_peaks` ranked peaks. Under the v2 config those floors
    are 100 cells and 500 peaks and the filter removes bonobo Purkinje (4 cells,
    298 peaks), so the gate is read over five species.

    Every model the filter drops gets an audit row naming the species, the cell
    count, the peak count and the floors, so the exclusion is visible rather than
    implied by the absence of a row in the Spearman table.
    """
    block = _eligibility_block(config)
    declared: list[str] = []
    dropped: dict[str, str] = {}
    source = ""
    expected: list[str] = []
    error = ""
    try:
        manifest = C.ranking_files(config)
        manifest = manifest[manifest.analysis.astype(str).str.lower() == ANALYSIS.lower()].copy()
        declared = _display_order(manifest.species.astype(str))
        kept, filter_audit = C.apply_species_filter(config, manifest)
        expected = _display_order(kept.species.astype(str))
        for row in filter_audit.itertuples():
            species = str(row.species)
            reason = f"{row.action} (target celltype {row.celltype}; model {row.model})"
            dropped[species] = reason
            audit.add(
                species, "species_excluded", f"model {row.model}", n_cells=int(row.n_cells),
                detail=(f"eligibility: {reason}; the species is dropped from this analysis "
                        f"because it has fewer than {row.min_cells} {row.celltype} cells "
                        f"or fewer than {row.min_ranked_peaks} ranked peaks"),
                status="excluded",
            )
        dirs = C.ranked_dirs(config)
        base = dirs.get("purkinje_oligo") or dirs.get("astroglia") or C.ROOT
        source = C.display_path(base)
    except Exception as exception:  # a manifest that cannot be read is reported, not raised
        expected, error = [], repr(exception)
        audit.add("-", "input_missing", "ranked model manifest", detail=error, status="blocked")
    return SpeciesSet(
        expected=expected,
        declared=declared,
        dropped=dropped,
        source=source,
        min_cells=C.min_cells_per_species(config),
        min_ranked_peaks=C.min_ranked_peaks(config),
        celltype_column=str(block.get("celltype_column", "")),
        block="eligibility" if config.get("eligibility") else (
            ("eligibility" if config.get("eligibility") else "species_filter") if (config.get("eligibility") or config.get("species_filter")) else ""
        ),
        configured=bool(block),
        error=error,
    )


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


@dataclass
class PurkinjeInputs:
    model_scores: Path
    recurrence: Path | None = None
    chromvar: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, Path] = field(default_factory=dict)


def find_recurrence_table(config: dict) -> Path | None:
    """The upstream step's module-recurrence table, discovered rather than assumed.

    s06 owns the definition of a recurrent module; this step only reads it. The
    file name is discovered from the step's own output directory (a `recurrent_*`
    or `*recurrence*` table) so that renaming the artefact upstream does not
    silently keep this step on an older one, and the name actually used is
    audited and recorded in the summary.
    """
    root = C.config_path(config, "results") / RECURRENCE_STEP_DIR
    if not root.is_dir():
        return None
    found: list[Path] = []
    for pattern in RECURRENCE_PATTERNS:
        for path in sorted(root.glob(pattern)):
            if path not in found:
                found.append(path)
    return found[0] if found else None


def collect_inputs(config: dict) -> PurkinjeInputs:
    """The per-species inputs this step reads, discovered on disk.

    The species keys come from the directories that are actually present, not from
    a roster: whatever wrote a chromVAR Z matrix or a pseudobulk metadata file is
    read, and whether it may then enter the analysis is decided separately against
    the eligible manifest (see `load_species_set`).
    """
    results = C.config_path(config, "results")
    inputs = PurkinjeInputs(
        model_scores=results / "01_model_motif" / "all_model_motif_scores.tsv",
        recurrence=find_recurrence_table(config),
    )
    chromvar_root = results / "04_chromvar"
    metadata_root = results / "03_chromvar_input"
    if chromvar_root.is_dir():
        for path in sorted(chromvar_root.glob("*/chromvar_z.tsv.gz")):
            inputs.chromvar[path.parent.name] = path
    if metadata_root.is_dir():
        for path in sorted(metadata_root.glob("*/pseudobulk_metadata.tsv")):
            inputs.metadata[path.parent.name] = path
    return inputs


def _as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def load_model_profiles(
    config: dict, path: Path, topk: int, audit: Audit
) -> dict[str, pd.DataFrame]:
    """One log2_OR profile per species for the Purkinje model."""
    frame = pd.read_csv(path, sep="\t")
    required = {"analysis", "species", "motif_id", "log2_OR"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {missing}")
    frame = frame[frame.analysis.astype(str).str.lower() == ANALYSIS.lower()].copy()
    if frame.empty:
        return {}

    if "topk" in frame.columns:
        available = sorted(int(v) for v in pd.unique(frame.topk.dropna()))
        if topk not in available:
            chosen = min(available, key=lambda value: (abs(value - topk), value))
            audit.add(
                "-", "model_selection", f"topk={topk}",
                detail=f"available topk={available}; using {chosen}", status="substituted",
            )
            topk = chosen
        frame = frame[frame.topk.astype(int) == topk]

    if "significant" in frame.columns:
        frame["significant"] = _as_bool(frame.significant)
    elif "FDR" in frame.columns:
        frame = C.add_significance(frame, float(config["model_analysis"]["fdr"]))

    profiles: dict[str, pd.DataFrame] = {}
    for species, group in frame.groupby("species", sort=False):
        models = sorted(group.model.astype(str).unique()) if "model" in group.columns else []
        if len(models) > 1:
            audit.add(
                species, "model_selection", "multiple models for one species",
                detail=f"{models}; using {models[0]}", status="substituted",
            )
            group = group[group.model.astype(str) == models[0]]
        columns = ["motif_id", "motif_name", "log2_OR"]
        if "significant" in group.columns:
            columns.append("significant")
        deduped = group[columns].drop_duplicates("motif_id")
        if len(deduped) < len(group):
            audit.add(
                str(species), "model_selection", "duplicate motif rows",
                detail=(f"{len(group) - len(deduped)} duplicate motif_id rows dropped (first kept) "
                        f"of {len(group)} rows"),
                status="excluded",
            )
        profiles[str(species)] = deduped.sort_values("motif_id").reset_index(drop=True)
    return profiles


def load_pseudobulk_metadata(path: Path, species: str, audit: Audit) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    if "pseudobulk" not in frame.columns:
        first = str(frame.columns[0])
        if not first.startswith("Unnamed"):
            raise RuntimeError(
                f"{path} has no `pseudobulk` column and its first column is `{first}`, which is not "
                f"a written-out index; refusing to guess the pseudobulk unit"
            )
        frame = frame.rename(columns={first: "pseudobulk"})
        audit.add(species, "metadata", "pseudobulk column", detail="taken from the unnamed first column")
    missing = sorted({"celltype", "n_cells"} - set(frame.columns))
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {missing}")
    frame = frame.copy()
    frame["pseudobulk"] = frame.pseudobulk.astype(str)
    frame["celltype"] = frame.celltype.astype(str)
    frame["n_cells"] = pd.to_numeric(frame.n_cells, errors="coerce").fillna(0).astype(int)
    return frame


def load_chromvar_z(path: Path, pseudobulk_ids: set[str], species: str, audit: Audit) -> pd.DataFrame:
    """Read a motif x pseudobulk Z matrix, transposing it if s04 wrote it the other way."""
    frame = pd.read_csv(path, sep="\t", index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = [str(column) for column in frame.columns]
    by_row = len(pseudobulk_ids & set(frame.index))
    by_column = len(pseudobulk_ids & set(frame.columns))
    if max(by_row, by_column) == 0:
        raise RuntimeError(
            f"{path}: none of the {len(pseudobulk_ids)} pseudobulk ids of the metadata appear "
            f"in the Z matrix (index n={frame.shape[0]}, columns n={frame.shape[1]})"
        )
    if by_column >= by_row:
        orientation = "motif_x_pseudobulk(as_written)"
    else:
        frame = frame.T
        orientation = "transposed_from_pseudobulk_x_motif"
    audit.add(
        species, "chromvar_orientation", path.name,
        detail=f"{orientation}; {frame.shape[0]} motifs x {frame.shape[1]} pseudobulks; "
               f"{max(by_row, by_column)}/{len(pseudobulk_ids)} pseudobulk ids matched",
        status="ok",
    )
    return frame.astype(float)


# --------------------------------------------------------------------------
# raw Purkinje profile
# --------------------------------------------------------------------------


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(values.shape[0], np.nan)
    return (values * weights).sum(axis=1) / total


def raw_purkinje_profile(
    config: dict, species: str, z: pd.DataFrame, meta: pd.DataFrame, audit: Audit
) -> tuple[pd.DataFrame | None, dict, str]:
    """C_{s,m} = weighted mean Z over Purkinje pseudobulks minus the equal-weighted
    mean over the other celltypes of (weighted mean Z over that celltype's pseudobulks).

    `c_simple_all` is the fully unweighted variant (plain mean of the Purkinje
    pseudobulks minus the equal-weighted mean of the other celltypes' plain
    means), kept so the n_cells weighting choice is visible in the output rather
    than hidden inside it.

    Returns (profile, per-species stats, reason). `reason` is "" when the profile
    was computed and otherwise names why this species has no usable raw profile,
    so the caller can report the species as excluded with that reason instead of
    letting it disappear from the Spearman table.
    """
    min_cells_ct = int(config["chromvar"]["min_cells_per_celltype"])
    min_cells_pb = int(config["chromvar"]["min_cells_per_pseudobulk"])

    declared = set(meta.pseudobulk)
    small = meta[meta.n_cells < min_cells_pb]
    for row in small.itertuples():
        audit.add(
            species, "pseudobulk_dropped", str(row.pseudobulk), n_cells=int(row.n_cells),
            n_pseudobulk=1, detail=f"celltype={row.celltype}; n_cells < {min_cells_pb}",
            status="excluded",
        )
    meta = meta[meta.n_cells >= min_cells_pb].copy()

    columns = set(z.columns)
    missing_pb = meta[~meta.pseudobulk.isin(columns)]
    for row in missing_pb.itertuples():
        audit.add(
            species, "pseudobulk_dropped", str(row.pseudobulk), n_cells=int(row.n_cells),
            n_pseudobulk=1, detail=f"celltype={row.celltype}; absent from the chromVAR Z matrix",
            status="excluded",
        )
    meta = meta[meta.pseudobulk.isin(columns)].copy()
    orphans = sorted(columns - declared)
    if orphans:
        audit.add(
            species, "pseudobulk_without_metadata", f"{len(orphans)} pseudobulks", n_pseudobulk=len(orphans),
            detail=f"present in the Z matrix, absent from the metadata: {orphans[:5]}",
            status="excluded",
        )

    # the target mask is case-insensitive unconditionally, not only as a fallback: a species
    # whose metadata spells the target two ways (e.g. `Purkinje` and `purkinje`) must not have
    # the twin left standing as an independent other celltype, which would subtract target
    # pseudobulks from the very baseline C is measured against
    exact = meta.celltype == TARGET_CELLTYPE
    loose = meta.celltype.str.lower() == TARGET_CELLTYPE.lower()
    target = meta[loose]
    if target.empty:
        audit.add(
            species, "species_skipped", "no target pseudobulk", detail=f"no `{TARGET_CELLTYPE}` pseudobulk",
            status="excluded",
        )
        return None, {}, f"no usable `{TARGET_CELLTYPE}` pseudobulk (>= {min_cells_pb} cells)"
    if not exact.any():
        audit.add(
            species, "celltype_target", TARGET_CELLTYPE, n_cells=int(target.n_cells.sum()),
            n_pseudobulk=len(target), detail="matched case-insensitively", status="substituted",
        )
    else:
        twins = sorted(set(meta.loc[loose & ~exact, "celltype"].unique()))
        if twins:
            audit.add(
                species, "celltype_target", TARGET_CELLTYPE, n_cells=int(target.n_cells.sum()),
                n_pseudobulk=len(target),
                detail=(f"differently cased labels of the target folded into the target population "
                        f"and kept out of the other-celltype average: {twins}"),
                status="substituted",
            )

    # every label that carries target cells is excluded from the other-celltype average
    target_labels = set(target.celltype.unique())
    totals = meta.groupby("celltype", as_index=False).agg(
        n_cells=("n_cells", "sum"), n_pseudobulk=("pseudobulk", "size")
    )
    other_celltypes: list[str] = []
    for row in totals.itertuples():
        if row.celltype in target_labels:
            continue
        if row.n_cells < min_cells_ct:
            audit.add(
                species, "celltype_excluded", str(row.celltype), n_cells=int(row.n_cells),
                n_pseudobulk=int(row.n_pseudobulk),
                detail=f"total cells < min_cells_per_celltype={min_cells_ct}", status="excluded",
            )
            continue
        other_celltypes.append(str(row.celltype))
        audit.add(
            species, "celltype_used_other", str(row.celltype), n_cells=int(row.n_cells),
            n_pseudobulk=int(row.n_pseudobulk), detail="equal-weighted celltype in the contrast",
            status="ok",
        )
    if not other_celltypes:
        audit.add(
            species, "species_skipped", "no other celltype",
            detail=f"every other celltype is below {min_cells_ct} cells", status="excluded",
        )
        return None, {}, f"no other celltype above {min_cells_ct} cells to contrast against"

    motifs = list(dict.fromkeys(z.index))
    value = z.loc[motifs]

    target_ids = target.pseudobulk.tolist()
    target_weights = target.set_index("pseudobulk").n_cells.loc[target_ids].to_numpy(dtype=float)
    target_matrix = value[target_ids].to_numpy(dtype=float)
    purkinje_mean = _weighted_mean(target_matrix, target_weights)
    purkinje_simple = target_matrix.mean(axis=1)

    others_weighted: list[np.ndarray] = []
    others_simple: list[np.ndarray] = []
    n_other_pb = 0
    n_other_cells = 0
    for celltype in other_celltypes:
        subset = meta[meta.celltype == celltype]
        ids = subset.pseudobulk.tolist()
        weights = subset.set_index("pseudobulk").n_cells.loc[ids].to_numpy(dtype=float)
        matrix = value[ids].to_numpy(dtype=float)
        others_weighted.append(_weighted_mean(matrix, weights))
        others_simple.append(matrix.mean(axis=1))
        n_other_pb += len(ids)
        n_other_cells += int(subset.n_cells.sum())
    other_mean = np.vstack(others_weighted).mean(axis=0).ravel()
    other_mean_simple = np.vstack(others_simple).mean(axis=0).ravel()

    audit.add(
        species, "celltype_target", TARGET_CELLTYPE, n_cells=int(target.n_cells.sum()),
        n_pseudobulk=len(target_ids), detail="n_cells-weighted mean over its pseudobulks", status="ok",
    )
    stats = dict(
        n_purkinje_pseudobulk=len(target_ids),
        n_purkinje_cells=int(target.n_cells.sum()),
        n_other_celltypes=len(other_celltypes),
        n_other_pseudobulk=n_other_pb,
        n_other_cells=n_other_cells,
    )
    frame = pd.DataFrame(
        {
            "species": species,
            "motif_id": motifs,
            "C": purkinje_mean - other_mean,
            "c_simple_all": purkinje_simple - other_mean_simple,
            "purkinje_mean_z": purkinje_mean,
            "other_equal_weighted_mean_z": other_mean,
            "purkinje_mean_z_simple": purkinje_simple,
            "other_equal_weighted_mean_z_simple": other_mean_simple,
            "n_purkinje_pseudobulk": stats["n_purkinje_pseudobulk"],
            "n_purkinje_cells": stats["n_purkinje_cells"],
            "n_other_celltypes": stats["n_other_celltypes"],
            "n_other_pseudobulk": stats["n_other_pseudobulk"],
            "n_other_cells": stats["n_other_cells"],
        }
    )
    return frame, stats, ""


# --------------------------------------------------------------------------
# module aggregation
# --------------------------------------------------------------------------


def centered_module_scores(
    profile: pd.DataFrame,
    value: str,
    annotation: pd.DataFrame,
    significant_col: str | None = None,
) -> pd.DataFrame:
    """Median member score after each motif's cross-species mean is removed.

    `common.family_trajectory` removes a per-motif baseline *within* a species
    across `stage`. A Purkinje model has exactly one stage, so the two columns are
    swapped: the constant `species` column is the analysis label and `stage`
    carries the real species. The subtracted baseline is then that motif's
    cross-species mean, i.e. readme section 11 applied to the species axis. The
    annotation frame carries module names in `family_trajectory`'s `family`
    column.
    """
    work = pd.DataFrame(
        {
            "species": ANALYSIS,
            "stage": profile["species"].astype(str).to_numpy(),
            "motif_id": profile["motif_id"].to_numpy(),
            value: profile[value].to_numpy(),
        }
    )
    if significant_col is not None and significant_col in profile.columns:
        work["significant"] = _as_bool(profile[significant_col]).to_numpy()
    out = C.family_trajectory(work, annotation, value=value, motif_col="motif_id")
    # the constant column carried the analysis label; the real species sits in `stage`
    out = out.drop(columns=["species"]).rename(
        columns={"stage": "species", "family_score": f"{value}_module_score"}
    )
    if "n_significant" in out.columns:
        out = out.rename(columns={"n_significant": "n_members_significant"})
    return out


def _sign_agreement(raw: pd.Series, model: pd.Series) -> pd.Series:
    return (np.sign(raw.fillna(0.0)) * np.sign(model.fillna(0.0))) > 0


def module_raw_support(
    recurrent_modules: set[str],
    recurrent_declared: list[str],
    panel: dict[str, ModuleSpec],
    raw_profile: pd.DataFrame,
    model_long: pd.DataFrame,
    annotation: pd.DataFrame,
    model_source: str,
    audit: Audit,
) -> pd.DataFrame:
    """Per module and species: the uncentered raw score against the model score.

    Every module of the Purkinje panel gets a row, whether or not the model calls
    it recurrent -- that is what makes the secondary modules (NF-Y, RFX) and the
    non-recurrent primaries visible instead of absent. `may_gate` carries the
    level rule, `is_recurrent` carries the model's own recurrence call, and only
    a module that is BOTH may set `p3_support`.

    The gate compares like with like. `raw_module_score` is the uncentered median
    C over the module's monomeric members, the direct analogue of
    `model_module_score`, which is s06's own per-species module score whenever its
    recurrence table carries it and otherwise the same median recomputed from the
    model profile. `raw_module_score_centered` and `model_module_score_centered`
    are the same two quantities with each motif's cross-species mean removed;
    their agreement is the supplementary `sign_agree_centered` reading, never the
    gate.
    """
    raw_centered = centered_module_scores(raw_profile, "C", annotation)
    model_centered = centered_module_scores(model_long, "log2_OR", annotation, "significant")

    for name in recurrent_declared:
        if name not in panel:
            audit.add(
                "-", "module_recurrence", name,
                detail=("declared recurrent by the model but not a module of the "
                        f"{ANALYSIS} panel; it is reported here and never scored"),
                status="excluded",
            )

    records: list[dict] = []
    for module, spec in panel.items():
        members = set(spec.members)
        raw_rows = raw_centered[raw_centered.family == module]
        model_rows = model_centered[model_centered.family == module]
        if raw_rows.empty or not members:
            audit.add(
                "-", "module_support", module,
                detail="no monomeric member of this module appears in the raw profile",
                status="excluded",
            )
            continue
        raw_uncentered = (
            raw_profile[raw_profile.motif_id.isin(members)]
            .groupby("species", as_index=False).C.median()
            .rename(columns={"C": "raw_module_score"})
        )
        model_uncentered = (
            model_long[model_long.motif_id.isin(members)]
            .groupby("species", as_index=False).log2_OR.median()
            .rename(columns={"log2_OR": "model_module_score"})
        )
        merged = raw_rows[["species", "C_module_score", "n_members"]].rename(
            columns={"C_module_score": "raw_module_score_centered", "n_members": "n_members_raw"}
        ).merge(raw_uncentered, on="species", how="left").merge(
            model_uncentered, on="species", how="left"
        ).merge(
            model_rows[["species", "log2_OR_module_score", "n_members"]].rename(
                columns={"log2_OR_module_score": "model_module_score_centered",
                         "n_members": "n_members_model"}
            ),
            on="species",
            how="left",
        )
        merged["sign_agree"] = _sign_agreement(merged.raw_module_score, merged.model_module_score)
        merged["sign_agree_centered"] = _sign_agreement(
            merged.raw_module_score_centered, merged.model_module_score_centered
        )
        n_scored = int((merged.model_module_score.notna() & merged.raw_module_score.notna()).sum())
        n_agree = int(merged.sign_agree.sum())
        is_recurrent = module in recurrent_modules
        may_gate = bool(spec.may_gate and is_recurrent)
        records.extend(
            dict(
                module=module,
                analysis_level=spec.level,
                may_gate=bool(spec.may_gate),
                is_recurrent=bool(is_recurrent),
                species=str(row.species),
                raw_module_score=(
                    float(row.raw_module_score) if pd.notna(row.raw_module_score) else np.nan
                ),
                raw_module_score_centered=(
                    float(row.raw_module_score_centered)
                    if pd.notna(row.raw_module_score_centered) else np.nan
                ),
                model_module_score=(
                    float(row.model_module_score) if pd.notna(row.model_module_score) else np.nan
                ),
                model_module_score_centered=(
                    float(row.model_module_score_centered)
                    if pd.notna(row.model_module_score_centered) else np.nan
                ),
                sign_agree=bool(row.sign_agree),
                sign_agree_centered=bool(row.sign_agree_centered),
                n_members_raw=int(row.n_members_raw),
                n_members_model=int(row.n_members_model) if pd.notna(row.n_members_model) else 0,
                n_species_agree=n_agree,
                n_species_agree_centered=int(merged.sign_agree_centered.sum()),
                n_species_scored=n_scored,
                p3_support=bool(may_gate and n_agree >= GATE_MIN_SPECIES_AGREE),
                model_score_source=model_source,
            )
            for row in merged.itertuples()
        )
    return _sort_module_frame(pd.DataFrame(records, columns=MODULE_COLUMNS), list(panel))


def _sort_module_frame(frame: pd.DataFrame, panel_order: list[str]) -> pd.DataFrame:
    """Passing gating modules first, then the rest of the panel in panel order."""
    if frame.empty:
        return frame
    order = {module: index for index, module in enumerate(panel_order)}
    species_order = {name: index for index, name in enumerate(C.SPECIES_ORDER)}
    ranked = frame.assign(
        _pass=frame.p3_support.astype(int),
        _may=frame.may_gate.astype(int),
        _order=frame.module.map(order).fillna(len(order)),
        _species_order=frame.species.map(species_order).fillna(len(species_order)),
    )
    return (
        ranked.sort_values(
            ["_pass", "_may", "n_species_agree", "_order", "_species_order"],
            ascending=[False, False, False, True, True],
        )
        .drop(columns=["_pass", "_may", "_order", "_species_order"])
        .reset_index(drop=True)
    )


@dataclass
class RecurrenceTable:
    """The upstream model step's own recurrence call, as this step reads it."""

    modules: set[str]                 # panel modules the model calls recurrent
    declared: list[str]               # every module name the table carries
    column: str                       # the column the module identity was read from
    frame: pd.DataFrame               # the recurrent rows only
    error: str = ""                   # non-empty when the table could not be read


def load_recurrent_modules(
    path: Path, panel: dict[str, ModuleSpec], audit: Audit
) -> RecurrenceTable:
    """The modules the upstream model step called recurrent.

    The module-identity column is found by content -- `module` when the producer
    names it that way, otherwise the column whose values name panel modules -- so a
    renamed producer column cannot silently empty the set, and the column actually
    used is audited. Names outside the panel are returned in `declared` and audited
    by the caller, never dropped in silence.
    """
    frame = pd.read_csv(path, sep="\t")
    if frame.empty:
        audit.add(
            "-", "module_recurrence", path.name,
            detail="the recurrence table is empty: the model declares no recurrent module",
            status="substituted",
        )
        return RecurrenceTable(set(), [], "", frame)
    if "is_recurrent" in frame.columns:
        keep = _as_bool(frame.is_recurrent)
        dropped = sorted(frame.loc[~keep, frame.columns[0]].astype(str).unique())
        if dropped:
            audit.add(
                "-", "module_recurrence", "is_recurrent==False",
                detail=f"excluded {len(dropped)} rows: {dropped}", status="excluded",
            )
        frame = frame[keep]

    panel_names = set(panel)
    column = "module" if "module" in frame.columns else None
    if column is None:
        best, best_hits = None, 0
        for name in frame.columns:
            if frame[name].dtype.kind not in "O":
                continue
            hits = len(set(frame[name].astype(str)) & panel_names)
            if hits > best_hits:
                best, best_hits = str(name), hits
        column = best
    if column is None:
        # an interface mismatch, not a missing file: reported so the gate stays PENDING
        # rather than being decided on a vocabulary this step cannot read
        error = (
            f"{path} has no column whose values name {ANALYSIS} panel modules "
            f"(columns: {list(frame.columns)}); the recurrence vocabulary cannot be read"
        )
        audit.add("-", "module_recurrence", path.name, detail=error, status="blocked")
        return RecurrenceTable(set(), [], "", frame, error=error)
    audit.add(
        "-", "module_recurrence", path.name,
        detail=f"module identity read from column `{column}`", status="ok",
    )
    names = [str(value) for value in frame[column].dropna().unique()]
    if len(names) != len(frame[column]):
        audit.add(
            "-", "module_recurrence", "duplicate module rows",
            detail=f"{len(frame[column]) - len(names)} duplicate rows dropped (first kept)",
            status="substituted",
        )
    return RecurrenceTable(
        modules={name for name in names if name in panel_names},
        declared=names, column=column, frame=frame,
    )


def model_scores_from_recurrent(frame: pd.DataFrame, column: str) -> pd.DataFrame | None:
    """The per-species model module score carried by the recurrence table, if any.

    s06 writes one `score_<species>` column per species (the uncentered median
    member log2_OR). A long layout with a `species` column and a score column is
    accepted too. The species columns are read off the frame itself, so a species
    the model ran but this analysis excludes -- bonobo under v2 -- is carried here
    and simply never joined onto a scored row.
    """
    if column not in frame.columns:
        return None
    prefixed = [str(name) for name in frame.columns if str(name).startswith("score_")]
    if prefixed:
        long = frame.melt(id_vars=[column], value_vars=prefixed,
                          var_name="column", value_name="model_module_score")
        return long.assign(
            module=long[column].astype(str), species=long.column.str[len("score_"):]
        ).loc[:, ["module", "species", "model_module_score"]]
    score_columns = [name for name in ("model_module_score", "module_score", "score")
                     if name in frame.columns]
    if "species" in frame.columns and score_columns:
        return frame[[column, "species", score_columns[0]]].rename(
            columns={column: "module", score_columns[0]: "model_module_score"}
        )
    return None


def apply_provided_model_scores(support: pd.DataFrame, provided: pd.DataFrame) -> pd.DataFrame:
    """Fill the model sign from the one the model itself reported, never blank it.

    FILL, do not replace. The upstream table is the model's RECURRENT-ONLY subset
    (s06 writes `module_frame[module_frame.is_recurrent]`), so every panel module
    the model did not call recurrent is simply absent from it. A replace-join turns
    those rows into `model_module_score = NaN`, which `_sign_agreement` maps to sign
    0 -- and the module is then reported, in the audit table, in the summary and in
    the figure, as a raw/model disagreement that never happened, while
    `n_species_scored` still counts the species. The recomputed median member
    log2_OR `module_raw_support` already produced for exactly those rows is kept;
    the upstream column only overwrites the rows it actually carries. `model_score_source`
    records per row which of the two was used, so the provenance is visible rather
    than implied.
    """
    if support.empty:
        return support
    supplied = provided.dropna(subset=["model_module_score"])
    merged = support.merge(
        supplied[["module", "species", "model_module_score"]].rename(
            columns={"model_module_score": "model_module_score_upstream"}
        ),
        on=["module", "species"],
        how="left",
    )
    from_upstream = merged.model_module_score_upstream.notna()
    merged["model_module_score"] = merged.model_module_score_upstream.where(
        from_upstream, merged.model_module_score
    )
    merged = merged.drop(columns=["model_module_score_upstream"])
    # `model_score_source` is per row: the upstream table's own description (which names
    # its file and its score_<species> columns) for the rows it supplies, and the
    # recomputed marker for the rows it does not. A blanked row would have said the
    # latter without saying so.
    merged["model_score_source"] = np.where(
        from_upstream, merged.model_score_source.astype(str), RECOMPUTED_SCORE_SOURCE
    )
    merged["sign_agree"] = _sign_agreement(merged.raw_module_score, merged.model_module_score)
    for column, agree in (("n_species_agree", "sign_agree"),
                          ("n_species_agree_centered", "sign_agree_centered")):
        merged[column] = merged.groupby("module", sort=False)[agree].transform("sum").astype(int)
    # `is_recurrent` belongs in this conjunction exactly as it does in
    # module_raw_support: `may_gate` alone is the module's LEVEL rule, and a primary
    # module the model did not call recurrent is not what this reading is about. On the
    # blanking path the two were indistinguishable (a blanked row scored 0 agreement);
    # now that the scores are filled, dropping the conjunct would report a
    # non-recurrent module as passing.
    merged["p3_support"] = (
        merged.may_gate & merged.is_recurrent
        & (merged.n_species_agree >= GATE_MIN_SPECIES_AGREE)
    )
    return _sort_module_frame(merged[MODULE_COLUMNS], list(dict.fromkeys(merged.module)))


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------


def write_figure(
    out_dir: Path,
    similarity: pd.DataFrame,
    profiles: dict[str, pd.DataFrame],
    module_support: pd.DataFrame,
    formats: list[str],
    dpi: int,
    n_expected: int,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # display order only; the panel set is exactly the species that were scored
    species = _display_order(profiles)
    modules = list(dict.fromkeys(module_support.module.tolist()))[:MAX_MODULE_PANELS]
    ncols = 3
    nrows = max(1, int(np.ceil(len(species) / ncols)))
    height_ratios = [1.0] * nrows + [0.62] * len(modules)
    fig = plt.figure(
        figsize=(4.3 * ncols, 3.3 * nrows + 1.75 * len(modules) + 1.0),
        facecolor=PALETTE["surface"],
    )
    grid = fig.add_gridspec(len(height_ratios), ncols, height_ratios=height_ratios,
                            hspace=0.55, wspace=0.3)
    rho_by_species = similarity.set_index("species").rho.to_dict() if not similarity.empty else {}

    for index, name in enumerate(species):
        axes = fig.add_subplot(grid[index // ncols, index % ncols])
        profile = profiles[name]
        axes.scatter(profile.log2_OR, profile.C, s=16, alpha=0.65, linewidths=0,
                     color=PALETTE["positive"])
        if len(profile) > 2:
            slope, intercept = np.polyfit(profile.log2_OR.to_numpy(dtype=float),
                                          profile.C.to_numpy(dtype=float), 1)
            xs = np.linspace(profile.log2_OR.min(), profile.log2_OR.max(), 32)
            axes.plot(xs, slope * xs + intercept, lw=1.4, color=PALETTE["ink_muted"])
        axes.axhline(0.0, lw=1.0, color=PALETTE["grid"])
        axes.axvline(0.0, lw=1.0, color=PALETTE["grid"])
        axes.set_title(name, fontsize=11, color=PALETTE["ink"])
        axes.set_xlabel("model log2 odds-ratio", fontsize=9, color=PALETTE["ink_muted"])
        axes.set_ylabel("raw chromVAR C\n(Purkinje - other celltypes)", fontsize=9,
                        color=PALETTE["ink_muted"])
        rho = rho_by_species.get(name, float("nan"))
        axes.text(0.04, 0.95, f"rho = {rho:.2f}" if np.isfinite(rho) else "rho = n/a",
                  transform=axes.transAxes, va="top", fontsize=10, color=PALETTE["ink"])
        axes.tick_params(labelsize=8, colors=PALETTE["ink_muted"])
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axes.spines[side].set_color(PALETTE["grid"])
        axes.grid(True, lw=0.6, color=PALETTE["grid"], alpha=0.6)
        axes.set_axisbelow(True)

    # one row per module: a grouped bar chart would collide at this width, and
    # separate rows let each module carry its own agreement count in its title
    for module_index, module in enumerate(modules):
        axes = fig.add_subplot(grid[nrows + module_index, :])
        subset = module_support[module_support.module == module].set_index("species")
        present = _display_order(set(module_support.species))
        positions = np.arange(len(present))
        xs, ys, colors, model_ys = [], [], [], []
        for species_index, name in enumerate(present):
            if name not in subset.index:
                continue
            row = subset.loc[name]
            xs.append(float(positions[species_index]))
            ys.append(float(row.raw_module_score))
            colors.append(PALETTE["positive"] if row.raw_module_score > 0 else PALETTE["negative"])
            model_ys.append(float(row.model_module_score))
        if xs:
            axes.vlines(xs, np.minimum(ys, model_ys), np.maximum(ys, model_ys),
                        lw=1.0, color=PALETTE["grid"], zorder=1)
            axes.bar(xs, ys, width=0.62, color=colors, linewidth=0, zorder=2)
            axes.scatter(xs, model_ys, marker="D", s=24, color=PALETTE["ink"],
                         edgecolors=PALETTE["surface"], linewidths=0.8, zorder=3)
        axes.axhline(0.0, lw=1.2, color=PALETTE["ink_muted"])
        axes.set_xticks(positions)
        axes.set_xticklabels(present, fontsize=8, color=PALETTE["ink_muted"])
        axes.set_ylabel("module score\n(uncentered)", fontsize=8, color=PALETTE["ink_muted"])
        agree = int(subset.sign_agree.sum())
        level = str(subset.analysis_level.iloc[0]) if len(subset) else ""
        recurrent = bool(subset.is_recurrent.iloc[0]) if len(subset) else False
        # the denominator of the verdict is the number of species the analysis was
        # defined over (the eligible manifest), not the number that happens to have
        # a row in this module; the threshold itself is the frozen gate constant
        if not subset.may_gate.iloc[0]:
            verdict = "reported, never read as agreement"
        elif not recurrent:
            verdict = "not recurrent in the model, not read as agreement"
        else:
            verdict = ("reaches " if agree >= GATE_MIN_SPECIES_AGREE else "does not reach ") + (
                f"{GATE_MIN_SPECIES_AGREE}/{n_expected} agreement"
            )
        # the module layer's reading, labelled as such: gate P3 itself is the rho in the
        # species panels above and is never decided by these rows
        axes.set_title(
            f"{module} ({level}) [interpretation] - raw sign agrees with the model in "
            f"{agree}/{len(present)} species ({verdict})",
            fontsize=10, color=PALETTE["ink"], loc="left",
        )
        axes.tick_params(labelsize=8, colors=PALETTE["ink_muted"])
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axes.spines[side].set_color(PALETTE["grid"])
        axes.grid(True, axis="y", lw=0.6, color=PALETTE["grid"], alpha=0.6)
        axes.set_axisbelow(True)
        if module_index == len(modules) - 1:
            axes.legend(
                handles=[
                    Patch(facecolor=PALETTE["positive"], label="raw module score > 0"),
                    Patch(facecolor=PALETTE["negative"], label="raw module score < 0"),
                    Line2D([], [], marker="D", color=PALETTE["ink"], linestyle="none",
                           markeredgecolor=PALETTE["surface"], label="model module score"),
                ],
                fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.45), ncols=3,
            )

    fig.suptitle(
        "Purkinje model motif profile vs raw chromVAR deviation (within species); "
        "gate P3 = rho > 0 in enough species",
        fontsize=13, color=PALETTE["ink"],
    )
    paths: list[Path] = []
    for fmt in formats:
        path = out_dir / f"purkinje_model_vs_raw.{fmt}"
        fig.savefig(path, dpi=dpi, facecolor=PALETTE["surface"], bbox_inches="tight")
        C.log("wrote", C.display_path(path))
        paths.append(path)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------
# summary helpers
# --------------------------------------------------------------------------


def _num(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _base_summary(config: dict, topk: int, out_dir: Path, species_set: SpeciesSet) -> dict:
    min_ct = int(config["chromvar"]["min_cells_per_celltype"])
    n_expected = species_set.n_expected
    summary = {
        "step": "s07_purkinje_raw_validation",
        "analysis": ANALYSIS,
        "readme_section": "26 (B5), gate P3",
        "status": "PENDING",
        "pending_reasons": [],
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"path": config.get("_path"), "seed": C.seed(config), "primary_topk": topk},
        "parameters": {
            "target_celltype": TARGET_CELLTYPE,
            "min_cells_per_celltype": min_ct,
            "min_cells_per_pseudobulk": int(config["chromvar"]["min_cells_per_pseudobulk"]),
            "n_motifs_expected": int(config["motif_database"]["n_motifs"]),
            "gate_min_species_agree": GATE_MIN_SPECIES_AGREE,
            # every species the eligibility filter kept, i.e. the species the gate is
            # read over; not a roster length
            "n_species_expected": n_expected,
        },
        "raw_profile": {
            "definition": (
                "C_{s,m} = mean over Purkinje pseudobulks of Z[m, .] - equal-weighted mean over the "
                "OTHER celltypes of (mean over that celltype's pseudobulks of Z[m, .])"
            ),
            "pseudobulk_averaging": (
                "averages inside a celltype are weighted by that pseudobulk's n_cells; the average "
                "across other celltypes is equal-weighted per celltype, never per cell"
            ),
            "c_simple_all": (
                "secondary fully-unweighted variant (plain pseudobulk means on both sides), reported "
                "so the weighting choice is inspectable rather than hidden"
            ),
            "min_cells_per_celltype_rule": (
                f"celltypes with fewer than {min_ct} total cells are excluded from the other-celltype "
                "average and recorded with their cell counts in purkinje_raw_audit.tsv"
            ),
            "cross_species_comparison": (
                "forbidden and not performed: chromVAR Z is computed per species on that species' own "
                "peak universe and background"
            ),
        },
        "model_vs_raw": {},
        # Gate P3 is the primary chromVAR validation gate (V2_PLAN section 15-2, G2) and is
        # read on the primary profile alone; the module layer's own reading of the same
        # evidence sits in `gate_p3` and never decides it.
        "primary_validation": {
            "criterion": (
                f"the model's Purkinje program is supported by raw accessibility in the same "
                f"direction: within a species, the Spearman rho between the model's per-motif "
                f"log2_OR and the raw per-motif C over the full motif intersection must be > 0 in "
                f"at least {GATE_MIN_SPECIES_AGREE} of the species the gate is read over"
            ),
            "layer": "primary",
            "primary_evidence": True,
            "evaluated": False,
            "pass": False,
            "reads_module_map": False,
            "verdict_reason": "",
        },
        # the basis the module layer's reading is taken on, named here so it is
        # self-describing; "uncentered" is the operation BOTH of its sides get
        "module_score_basis": {
            "module_reading": "uncentered",
            "supplementary_reading": "cross_species_centered",
            "detail": (
                "The module layer's reading compares like with like. The raw module score (median C "
                "over the module's monomeric members) and the model module score (the same median "
                "member log2_OR, from s06's score_<species> when the row is present and otherwise "
                "recomputed) are BOTH read uncentered -- neither side has a cross-species baseline "
                "removed. The cross-species centered pair (raw_module_score_centered vs "
                "model_module_score_centered, sign_agree_centered / n_species_agree_centered) is a "
                "supplementary reading only. None of this decides gate P3, which is read on the "
                "primary profile (see primary_validation)."
            ),
        },
        # the module layer's reading of the same evidence: interpretation only. Its
        # payload is filled in after the primary numbers exist.
        "gate_p3": {
            "layer": "interpretation",
            "primary_evidence": False,
            "criterion": (
                f"at least one model-defined recurrent primary module (a module whose level is not "
                f"in modules.non_gating_levels) whose uncentered raw chromVAR module score (the "
                f"median C over the module's monomeric members) has the same sign as the uncentered "
                f"model module score in >= {GATE_MIN_SPECIES_AGREE} of "
                f"{n_expected} species"
            ),
            "module_reading_evaluated": False,
            "module_reading_passes": False,
            "unavailable_reasons": [],
            "comparison_basis": "uncentered",
            "verdict_reason": "",
        },
        # the module layer's own state; filled in once the primary numbers exist
        "interpretation": {
            "layer": "interpretation",
            "primary_evidence": False,
            "available": False,
            "passed": None,
            "reasons": [],
            "payload": "gate_p3",
            "table": "module_raw_support.tsv",
            "role": (
                "the module layer's reading of the evidence gate P3 is decided on. It is reported "
                "here and in `gate_p3`, and it decides nothing: gate P3, this step's status, the "
                "raw profile and the similarity table are all computed without it."
            ),
        },
        "gates": {"P3": "PENDING"},
        "audit": {"table": "purkinje_raw_audit.tsv"},
        "interpretation_boundary": [
            "chromVAR Z is the per-motif deviation of a pseudobulk from the GC/background expectation "
            "of its own species' peak universe. It is not TF activity, not TF abundance, not TF "
            "binding, and not a measurement of any individual transcription factor.",
            "C is a contrast between Purkinje pseudobulks and the equal-weighted average of the other "
            "cerebellar celltypes. It measures relative accessibility deviation, not absolute "
            "Purkinje-specific accessibility.",
            "Every statistic here is computed within a single species. Raw C and raw Z are not "
            "comparable across species, because each species has its own peak universe, GC "
            "background, motif-peak matches and pseudobulk composition (readme section 26).",
            "The Spearman correlation between the model log2_OR profile and the raw C profile is "
            "descriptive: the motifs are not independent observations (they cluster into modules and "
            "share peaks), so its p-values are not calibrated family-wise.",
            "Gate P3 is a direction-of-effect concordance test between a model-derived enrichment "
            "statistic and a raw accessibility-deviation statistic. Agreement is consistent with the "
            "model encoding a real Purkinje regulatory program; it is not proof of TF-driven "
            "regulation and carries no causal claim.",
            "Gate P3 is decided on the PRIMARY evidence only: within a species, the Spearman rho "
            "between the model's per-motif log2_OR and the raw per-motif C over the full model/raw "
            "motif intersection, required to be > 0 in at least the frozen threshold number of "
            "species. It reads no module, no level and no panel (V2_PLAN section 15-2, G2: the "
            "879-motif model<->chromVAR Spearman), so emptying the module map cannot move it.",
            "The module panel is the Purkinje-specific interpretation layer of the config's module "
            "map (Homeodomain_nonPOU / POU / TFAP2 primary, NF-Y / RFX empirical secondary, and the "
            "derived Homeodomain_total union). Members always come through common.module_members and "
            "Homeodomain_total is the union of its two atomic parts, so no motif is counted twice. "
            "Its own reading is descriptive: only a primary (gating-level) module may be read as "
            "agreement -- a secondary or comparator module is reported and can never set p3_support "
            "-- and no module reading, at any level, can move gate P3.",
            "The module layer is interpretation only, and it is not what makes this run PENDING. The "
            "raw profile and the model-vs-raw Spearman table are computed over the full motif "
            "intersection and never consult the module map, so emptying the map cannot change a "
            "primary number, a gate status or this step's status; it can only change which named "
            "modules are reported and whether the module-layer reading is available at all.",
            "The cross-species centered module scores (raw_module_score_centered, "
            "model_module_score_centered) and their agreement (sign_agree_centered / "
            "n_species_agree_centered) are a supplementary reading of the module layer: each motif's "
            "cross-species mean is removed from both sides, so it asks whether a module's "
            "cross-species variation tracks the model's. It is deliberately not the reading that is "
            "reported as the module verdict, because a program that is equally Purkinje-elevated in "
            "every species centers to exactly zero and would be scored as disagreeing.",
        ],
        "outputs": {"directory": str(out_dir)},
    }
    if species_set.configured:
        # the block exists only for a config that declares one; its reason text names
        # both floors, because v2 enforces cells AND ranked peaks
        summary["species_filter"] = {
            "config_block": species_set.block,
            "rule": (
                f"a species enters this analysis only if it has at least "
                f"{species_set.min_cells} cells of the target celltype "
                f"({species_set.celltype_column}) in the integrated h5ad AND at least "
                f"{species_set.min_ranked_peaks} ranked peaks"
            ),
            "min_cells_per_species": species_set.min_cells,
            "min_ranked_peaks": species_set.min_ranked_peaks,
            "celltype_column": species_set.celltype_column,
            "target_celltype": TARGET_CELLTYPE,
            "source": species_set.source,
            "species_declared": species_set.declared,
            "n_species_declared": len(species_set.declared),
            "species_kept": species_set.expected,
            "n_species_kept": n_expected,
            "species_dropped": [
                {"species": name, "reason": species_set.dropped[name]}
                for name in _display_order(species_set.dropped)
            ],
            "note": (
                "n_species_expected counts the species the eligibility rule kept. A species it "
                "dropped is never scored: it is named here, in purkinje_raw_audit.tsv and, when "
                "it still has model or raw input on disk, as an excluded row of "
                "model_vs_raw_similarity.tsv carrying the same reason."
            ),
        }
    return summary


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def _write_pending(out_dir: Path, audit: Audit, summary: dict, detail: list[str]) -> int:
    summary["pending_reasons"].append({"reason": "missing_input", "detail": detail})
    for item in detail:
        audit.add("-", "input_missing", "input", detail=item, status="blocked")
    # a PENDING primary gate is never left without a reason attached to it. The module
    # reading is absent here too (there is no primary profile to read), and it is told so
    # separately: what makes the run PENDING is the missing primary input, never the layer.
    reason = "not evaluable: " + "; ".join(str(item) for item in detail)
    summary["primary_validation"]["verdict_reason"] = reason
    summary["gate_p3"]["unavailable_reasons"] = [
        {
            "reason": "missing_input",
            "detail": (
                f"{reason}; the module-layer reading is reported as unavailable for the same "
                "reason, and never as the cause"
            ),
        }
    ]
    summary["gate_p3"]["verdict_reason"] = reason
    summary["interpretation"]["reasons"] = list(summary["gate_p3"]["unavailable_reasons"])

    # a PENDING run must not leave a previous COMPLETE run's tables standing next to a summary
    # that says PENDING, so each is overwritten with a header-only table of the same columns
    # (overwriting is permitted; the reset is recorded rather than silent)
    reset: list[str] = []
    for name, columns in (
        ("purkinje_raw_profile.tsv", PROFILE_COLUMNS),
        ("model_vs_raw_similarity.tsv", SIMILARITY_COLUMNS),
        ("module_raw_support.tsv", MODULE_COLUMNS),
    ):
        path = out_dir / name
        existed = path.exists()
        C.save_table(pd.DataFrame(columns=columns), path, sep="\t")
        reset.append(f"{name} ({'overwrote a previous run' if existed else 'created empty'})")
    audit.add("-", "output_reset", "stale tables cleared", detail="; ".join(reset), status="substituted")

    stale_figures = sorted(path.name for path in out_dir.glob("purkinje_model_vs_raw.*"))
    summary["outputs"]["figures"] = []
    summary["outputs"]["tables"] = [
        "purkinje_raw_profile.tsv", "model_vs_raw_similarity.tsv",
        "module_raw_support.tsv", "purkinje_raw_audit.tsv",
    ]
    if stale_figures:
        summary["outputs"]["stale_figures"] = stale_figures
        summary["outputs"]["stale_figures_note"] = (
            "this PENDING run wrote no figure; the listed figure files are left over from an "
            "earlier run of this step and must not be read as this run's output"
        )
    C.save_table(audit.frame(), out_dir / "purkinje_raw_audit.tsv", sep="\t")
    C.save_json(summary, out_dir / "purkinje_validation_summary.json")
    C.log("STATUS: PENDING", detail)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Purkinje raw chromVAR validation (readme 26)")
    C.add_common_args(parser)
    args = parser.parse_args()
    C.main_guard()

    config = C.load_config(args.config)
    topk = int(args.topk or config["model_analysis"]["primary_topk"])
    out_dir = C.results_dir(config, STEP)
    audit = Audit()
    # who this step is allowed to analyse, and who it is not, decided by the ranked
    # manifest and the config's eligibility block before anything is read
    species_set = load_species_set(config, audit)
    summary = _base_summary(config, topk, out_dir, species_set)
    # the cell-type-specific module panel this step interprets through: read through
    # common.module_members only, never a second copy of the map
    panel = load_module_panel(config, audit)
    summary["module_panel"] = {
        "celltype": ANALYSIS,
        "source": C.display_path(C.module_map_path(config)),
        "n_modules": len(panel),
        "n_gating_modules": int(sum(1 for spec in panel.values() if spec.may_gate)),
        "aggregates": C.module_aggregates(config, ANALYSIS),
        "non_gating_levels": list(
            ((config.get("modules") or {}).get("non_gating_levels") or [])
        ),
        "modules": [
            {
                "module": spec.module, "analysis_level": spec.level,
                "may_gate": bool(spec.may_gate), "derived": bool(spec.derived),
                "derived_from": list(spec.parts), "n_monomer_members": len(spec.members),
            }
            for spec in panel.values()
        ],
        "rule": (
            "members come from common.module_members(config, 'Purkinje', module); a derived "
            "aggregate is the union of its parts' members, so no motif is counted twice; only a "
            "module whose level is not in modules.non_gating_levels may be read as agreement by the "
            "module layer, and no module at any level may decide gate P3 (see primary_validation)"
        ),
    }

    with C.StepTimer("s07 purkinje raw validation"):
        inputs = collect_inputs(config)
        summary["inputs"] = {
            "all_model_motif_scores": str(inputs.model_scores),
            "recurrent_modules": str(inputs.recurrence) if inputs.recurrence else None,
            "chromvar_z": {name: str(path) for name, path in sorted(inputs.chromvar.items())},
            "pseudobulk_metadata": {name: str(path) for name, path in sorted(inputs.metadata.items())},
            "sha256": {
                str(path): C.sha256_file(path)
                for path in [inputs.model_scores, *([inputs.recurrence] if inputs.recurrence else []),
                             *sorted(inputs.chromvar.values()), *sorted(inputs.metadata.values())]
                if path.exists()
            },
        }

        results_root = C.config_path(config, "results")
        blocking: list[str] = []
        if species_set.error:
            blocking.append(f"cannot read the ranked model manifest: {species_set.error}")
        elif not species_set.expected:
            blocking.append(
                "the eligibility rule kept no species with a Purkinje model; check "
                "eligibility and the per-species cell counts and ranked peaks"
            )
        if not inputs.model_scores.exists():
            blocking.append(f"missing {inputs.model_scores}")
        if not inputs.chromvar:
            blocking.append(
                f"no {C.display_path(results_root / '04_chromvar')}/<species>/chromvar_z.tsv.gz found"
            )
        if not inputs.metadata:
            blocking.append(
                f"no {C.display_path(results_root / '03_chromvar_input')}"
                "/<species>/pseudobulk_metadata.tsv found"
            )
        if blocking:
            return _write_pending(out_dir, audit, summary, blocking)

        model_profiles = load_model_profiles(config, inputs.model_scores, topk, audit)
        if not model_profiles:
            return _write_pending(
                out_dir, audit, summary,
                [f"no Purkinje rows in {inputs.model_scores}"],
            )
        summary["model_definition"] = {
            "source": str(inputs.model_scores),
            "primary_topk": topk,
            "species_with_a_model_profile": _display_order(model_profiles),
        }

        # every species that has anything to contribute is accounted for before the
        # analysis runs: one with a model but no usable raw profile, or a raw profile
        # but no model, is named here with its reason rather than simply missing from
        # the Spearman table
        inventory = _display_order([*model_profiles, *inputs.chromvar, *inputs.metadata,
                                    *species_set.declared])
        excluded: list[dict] = []
        excluded_seen: set[str] = set()

        def record_exclusion(species: str, reason: str, kind: str) -> None:
            if species in excluded_seen:
                return
            excluded_seen.add(species)
            excluded.append(dict(species=species, reason=reason, kind=kind))

        for species in _display_order([*species_set.expected, *inventory]):
            if species not in species_set.expected:
                reason = species_set.dropped.get(species)
                if reason is None:
                    # no eligibility row explained it, so this one has to
                    reason = ("not in the eligibility-filtered ranked manifest "
                              "(no Purkinje model to validate)")
                    audit.add(
                        species, "species_excluded", "not in the filtered species set",
                        detail=reason, status="excluded",
                    )
                record_exclusion(species, reason, "species_filter")
                continue
            if species not in model_profiles:
                audit.add(species, "species_skipped", "no Purkinje model profile", status="excluded")
                record_exclusion(species, "no Purkinje model profile", "no_model")
            if species not in inputs.chromvar:
                audit.add(species, "species_skipped", "no chromvar_z.tsv.gz", status="excluded")
                record_exclusion(species, "no chromvar_z.tsv.gz", "no_raw_profile")
            elif species not in inputs.metadata:
                audit.add(species, "species_skipped", "no pseudobulk_metadata.tsv", status="excluded")
                record_exclusion(species, "no pseudobulk_metadata.tsv", "no_raw_profile")

        profile_frames: list[pd.DataFrame] = []
        raw_stats: dict[str, dict] = {}
        similarity_rows: list[dict] = []
        model_long_frames: list[pd.DataFrame] = []

        for species in species_set.expected:
            if species not in model_profiles or species not in inputs.chromvar or species not in inputs.metadata:
                continue
            meta = load_pseudobulk_metadata(inputs.metadata[species], species, audit)
            z = load_chromvar_z(inputs.chromvar[species], set(meta.pseudobulk), species, audit)
            raw, stats, reason = raw_purkinje_profile(config, species, z, meta, audit)
            if raw is None:
                record_exclusion(species, reason, "no_raw_profile")
                continue

            model = model_profiles[species]
            joined = raw.merge(model, on="motif_id", how="inner", suffixes=("_raw", ""))
            n_raw, n_model, n_common = len(raw), len(model), len(joined)
            # a strict subset leaves n_common == min(n_raw, n_model), so the loss is audited on
            # the two sides separately; otherwise a model or raw profile that dropped motifs
            # would silently shrink every Spearman and every module score
            if n_raw != n_common or n_model != n_common:
                absent_from_model = sorted(set(raw.motif_id) - set(joined.motif_id))
                absent_from_raw = sorted(set(model.motif_id) - set(joined.motif_id))
                audit.add(
                    species, "motif_coverage", "model/raw motif intersection",
                    n_pseudobulk=stats["n_purkinje_pseudobulk"],
                    detail=(f"model={n_model} raw={n_raw} common={n_common}; "
                            f"absent_from_model={absent_from_model[:5]} "
                            f"absent_from_raw={absent_from_raw[:5]}"),
                    status="partial" if n_common else "excluded",
                )
            if n_common < 3:
                audit.add(
                    species, "species_skipped", "insufficient motif overlap",
                    detail=f"common motifs = {n_common}", status="excluded",
                )
                record_exclusion(species, f"insufficient motif overlap (n={n_common})", "no_raw_profile")
                continue
            joined = joined.sort_values("motif_id").reset_index(drop=True)
            joined["species"] = species
            # motif names live in the model table; the chromVAR matrix is indexed by id only
            joined["motif_name"] = (
                joined.motif_name.fillna("").astype(str) if "motif_name" in joined.columns
                else joined.motif_id.astype(str)
            )
            profile_frames.append(joined[PROFILE_COLUMNS].copy())
            raw_stats[species] = stats
            model_columns = ["motif_id", "log2_OR"]
            if "significant" in joined.columns:
                model_columns.append("significant")
            model_long_frames.append(joined[["species"] + model_columns].copy())

            rho, p_value = C.spearman(joined.log2_OR.to_numpy(dtype=float), joined.C.to_numpy(dtype=float))
            rho_simple, p_simple = C.spearman(
                joined.log2_OR.to_numpy(dtype=float), joined.c_simple_all.to_numpy(dtype=float)
            )
            similarity_rows.append(
                dict(
                    species=species, rho=rho, p=p_value, n_motifs=n_common,
                    n_pseudobulk_target=stats["n_purkinje_pseudobulk"],
                    n_pseudobulk_other=stats["n_other_pseudobulk"],
                    target_cells=stats["n_purkinje_cells"], other_cells=stats["n_other_cells"],
                    n_celltypes_other=stats["n_other_celltypes"],
                    rho_simple=rho_simple, p_simple=p_simple, p_fdr_bh=np.nan, status="ok",
                )
            )

        if not similarity_rows:
            return _write_pending(
                out_dir, audit, summary,
                ["no species produced both a raw Purkinje profile and a model profile"],
            )

        # Every species the analysis expected but could not score is written into the
        # Spearman table as an excluded row carrying its reason, and every species the
        # eligibility rule dropped is written there too: an empty slot in the table is
        # never the only trace of a species. This is unconditional -- the table always
        # says which species are missing and why, whatever the config declares.
        excluded_records: list[dict] = [
            dict(
                species=item["species"], kind=item["kind"], reason=item["reason"],
                status=f"excluded_{item['kind']}: {item['reason']}",
                rho=np.nan, p=np.nan, n_motifs=np.nan,
                n_pseudobulk_target=np.nan, n_pseudobulk_other=np.nan,
                target_cells=np.nan, other_cells=np.nan, n_celltypes_other=np.nan,
                rho_simple=np.nan, p_simple=np.nan, p_fdr_bh=np.nan,
            )
            for item in excluded
        ]
        excluded_rows = [
            {name: record[name] for name in SIMILARITY_COLUMNS} for record in excluded_records
        ]

        species_order = {name: index for index, name in enumerate(C.SPECIES_ORDER)}
        similarity = pd.DataFrame([*similarity_rows, *excluded_rows], columns=SIMILARITY_COLUMNS)
        valid = similarity.p.notna()
        if valid.any():
            similarity.loc[valid, "p_fdr_bh"] = multipletests(
                similarity.loc[valid, "p"], method=str(config["statistics"]["fdr_method"])
            )[1]
        # scored species first in the project's display order, excluded ones after them
        similarity = (
            similarity.assign(
                _excluded=similarity.status.ne("ok"),
                _order=similarity.species.map(species_order).fillna(len(species_order)),
            )
            .sort_values(["_excluded", "_order", "species"])
            .drop(columns=["_excluded", "_order"])
            .reset_index(drop=True)
        )
        scored = similarity[similarity.status == "ok"]

        profiles = pd.concat(profile_frames, ignore_index=True).sort_values(
            ["species", "motif_id"],
            key=lambda col: col.map(species_order) if col.name == "species" else col,
        ).reset_index(drop=True)

        summary["raw_profile"].update(
            {
                "n_motifs": int(profiles.motif_id.nunique()),
                "species_computed": sorted(raw_stats, key=species_order.get),
                "species_skipped": [name for name in species_set.expected if name not in raw_stats],
                "per_species": {name: raw_stats[name] for name in sorted(raw_stats, key=species_order.get)},
            }
        )
        summary["model_vs_raw"] = {
            "per_species": [
                {
                    "species": row.species, "rho": _num(row.rho), "p": _num(row.p),
                    "p_fdr_bh": _num(row.p_fdr_bh), "n_motifs": int(row.n_motifs),
                    "rho_simple_unweighted": _num(row.rho_simple),
                }
                for row in scored.itertuples()
            ],
            "n_species_computed": int(len(scored)),
            "n_species_rho_positive": int((similarity.rho > 0).sum()),
            "n_species_rho_positive_fdr05": int(((similarity.rho > 0) & (similarity.p_fdr_bh < 0.05)).sum()),
            "median_rho": _num(similarity.rho.median()),
            "min_rho": _num(similarity.rho.min()),
            "max_rho": _num(similarity.rho.max()),
            "role": (
                "the per-species profile gate P3 is read on (this table's rho is the gate's "
                "statistic; see summary.primary_validation). The module layer's own reading of the "
                "same evidence is reported separately and never decides the gate."
            ),
        }
        if excluded_records:
            summary["model_vs_raw"]["species_excluded"] = [
                {
                    "species": record["species"],
                    "kind": record["kind"],
                    "reason": record["reason"],
                    "row_status": record["status"],
                }
                for record in excluded_records
            ]
            summary["model_vs_raw"]["n_species_expected"] = species_set.n_expected
            summary["model_vs_raw"]["n_species_rows"] = int(len(similarity))

        model_long = pd.concat(model_long_frames, ignore_index=True)
        annotation = panel_annotation(panel)

        # ---- the module layer's OWN reading of the same evidence -----------------
        # Everything below reads the primary profile and the module map and writes
        # neither. Its verdicts are published under `summary["gate_p3"]` as the
        # interpretation layer's reading of the Purkinje program; gate P3 itself is
        # decided further down, on the primary profile alone. `module_reading_reasons`
        # therefore records why the *reading* is unavailable -- never why the run is
        # PENDING, because the primary gate does not depend on this layer at all.
        module_support = pd.DataFrame(columns=MODULE_COLUMNS)
        model_source = ""
        recurrence_error = ""
        recurrent_modules: set[str] = set()
        recurrent_declared: list[str] = []
        module_reading_reasons: list[dict] = []
        if not panel:
            module_reading_reasons.append(
                {
                    "reason": "module_panel_missing",
                    "detail": (
                        "the Purkinje module panel could not be read from the config's module map, so "
                        "the module-layer reading of this evidence is not available. Gate P3 is "
                        "unaffected: it is read on the full motif-intersection model<->raw profile and "
                        "never consults the module map."
                    ),
                }
            )
            audit.add("-", "module_reading", "module panel", detail="empty or unreadable",
                      status="blocked")
        elif inputs.recurrence is None:
            module_reading_reasons.append(
                {
                    "reason": "recurrent_modules_missing",
                    "detail": (
                        f"no recurrent_*.tsv under "
                        f"{C.display_path(C.config_path(config, 'results') / RECURRENCE_STEP_DIR)}; "
                        "the module-layer reading is not available because the model's recurrent "
                        "modules are defined upstream by s06. Gate P3 is unaffected: it never "
                        "consults the upstream table."
                    ),
                }
            )
            audit.add("-", "module_reading", "recurrent module table", detail="absent",
                      status="blocked")
        else:
            recurrence = load_recurrent_modules(inputs.recurrence, panel, audit)
            recurrent_modules, recurrent_declared = recurrence.modules, recurrence.declared
            recurrence_error = recurrence.error
            if recurrence.error:
                module_reading_reasons.append(
                    {
                        "reason": "recurrent_modules_unreadable",
                        "detail": (
                            f"{recurrence.error}; the module-layer reading is not available. Gate "
                            "P3 is unaffected: it never consults the upstream table."
                        ),
                    }
                )
            if not recurrence_error:
                provided = model_scores_from_recurrent(recurrence.frame, recurrence.column)
                if provided is None or not provided.model_module_score.notna().any():
                    # the producer carries no per-species module score, so the model sign is
                    # recomputed from the model profile by the same readme section 11 median
                    provided = None
                    model_source = "recomputed_from_01_model_motif_readme_section_11"
                else:
                    model_source = f"{inputs.recurrence.name}:score_<species>"
                module_support = module_raw_support(
                    recurrent_modules, recurrent_declared, panel, profiles, model_long,
                    annotation, model_source, audit,
                )
                if provided is not None:
                    module_support = apply_provided_model_scores(module_support, provided)
                if not recurrent_declared:
                    audit.add(
                        "-", "module_reading", "no recurrent module declared",
                        detail=f"{inputs.recurrence.name} declares no module", status="fail",
                    )
                elif not recurrent_modules:
                    # the table exists but speaks a vocabulary that names no panel module:
                    # that is an interface mismatch, so the module reading stays unavailable
                    # rather than being reported as a disagreement
                    module_reading_reasons.append(
                        {
                            "reason": "recurrent_module_vocabulary_mismatch",
                            "detail": (
                                f"{inputs.recurrence.name} declares {len(recurrent_declared)} "
                                f"recurrent modules ({recurrent_declared[:5]}), none of which is a "
                                f"module of the {ANALYSIS} panel {list(panel)}; the module-layer "
                                f"reading of this evidence cannot be read"
                            ),
                        }
                    )
                    audit.add(
                        "-", "module_reading", "recurrent module vocabulary",
                        detail=f"declared {recurrent_declared[:8]}; none in the panel", status="blocked",
                    )
                if module_support.empty:
                    audit.add(
                        "-", "module_reading", "no evaluable module",
                        detail=f"{len(recurrent_modules)} recurrent modules, none with raw members",
                        status="blocked",
                    )
                unscored = module_support[module_support.model_module_score.isna()]
                if not unscored.empty:
                    audit.add(
                        "-", "module_reading", "model score missing for some module/species rows",
                        detail=(f"{len(unscored)} rows, e.g. "
                                f"{unscored[['module', 'species']].head(3).to_dict('records')}"),
                        status="partial",
                    )
                for module, group in module_support.groupby("module", sort=False):
                    first = group.iloc[0]
                    audit.add(
                        "-", "module_reading", str(module),
                        detail=(f"level={first.analysis_level} may_gate={bool(first.may_gate)} "
                                f"is_recurrent={bool(first.is_recurrent)}; uncentered raw module "
                                f"score agrees with the model in {int(first.n_species_agree)}/"
                                f"{int(first.n_species_scored)} species "
                                f"(supplementary centered reading: "
                                f"{int(first.n_species_agree_centered)})"),
                        n_pseudobulk=int(first.n_members_raw),
                        status="pass" if first.p3_support else "fail",
                    )

        # ---- gate P3: the PRIMARY reading ----------------------------------------
        # Gate P3 is decided on the primary evidence alone -- the within-species
        # Spearman between the model's per-motif log2_OR and the raw per-motif C over
        # the full model/raw motif intersection -- exactly as V2_PLAN section 15-2
        # freezes it ("G2 must be built on the Spearman over the 879 motifs, never on
        # any module aggregate"). It reads no module, no level and no panel, so
        # emptying the module map cannot move it; that is what V2_PLAN's
        # test_module_layer_invariance.py mandates. The module layer's own reading of
        # the same evidence is reported above/below as an interpretation payload.
        rho_positive = int((scored.rho > 0).sum())
        n_scored = int(len(scored))
        p3_evaluated = bool(n_scored > 0)
        p3_pass = bool(p3_evaluated and rho_positive >= GATE_MIN_SPECIES_AGREE)
        if not p3_evaluated:
            p3_reason = (
                "not evaluable: no species produced both a raw Purkinje profile and a model profile"
            )
        elif p3_pass:
            p3_reason = (
                f"passed: the model profile and the raw profile agree in direction "
                f"(Spearman rho > 0) in {rho_positive}/{n_scored} species"
            )
        elif n_scored < GATE_MIN_SPECIES_AGREE:
            p3_reason = (
                f"failed without a disagreement: only {n_scored} species could be scored, fewer "
                f"than the {GATE_MIN_SPECIES_AGREE} the frozen threshold requires"
            )
        else:
            p3_reason = (
                f"failed on the evidence: the model profile and the raw profile agree in direction "
                f"in {rho_positive}/{n_scored} species, below the required {GATE_MIN_SPECIES_AGREE}"
            )
        summary["primary_validation"] = {
            "criterion": (
                f"the model's Purkinje program is supported by raw accessibility in the same "
                f"direction: within a species, the Spearman rho between the model's per-motif "
                f"log2_OR and the raw per-motif C (Purkinje minus the equal-weighted mean of the "
                f"other celltypes) over the full motif intersection must be > 0 in at least "
                f"{GATE_MIN_SPECIES_AGREE} of the species the gate is read over"
            ),
            "layer": "primary",
            "primary_evidence": True,
            "evaluated": p3_evaluated,
            "pass": p3_pass,
            "statistic": "within-species Spearman rho(model log2_OR, raw C) over the motif intersection",
            "threshold_species_agreeing": GATE_MIN_SPECIES_AGREE,
            "n_species_scored": n_scored,
            "n_species_rho_positive": rho_positive,
            "median_rho": _num(scored.rho.median()) if n_scored else None,
            "n_motifs": int(scored.n_motifs.median()) if n_scored else 0,
            "comparison_basis": (
                "within species, over the full model/raw motif intersection; the raw side is the "
                "uncentered C and the model side is the uncentered log2_OR, so no cross-species "
                "centering enters and no module aggregation is read"
            ),
            "per_species": [
                {"species": row.species, "rho": _num(row.rho), "n_motifs": int(row.n_motifs)}
                for row in scored.itertuples()
            ],
            "reads_module_map": False,
            "module_layer_independence": (
                "this payload is computed from all_model_motif_scores.tsv and the per-species "
                "chromVAR z matrices only; pointing modules.file at an empty map cannot change "
                "`evaluated`, `pass`, `n_species_rho_positive` or the gate P3 string in `gates`"
            ),
            "verdict_reason": p3_reason,
        }
        summary["gates"]["P3"] = "PASS" if p3_pass else ("FAIL" if p3_evaluated else "PENDING")
        # this step's status answers for the PRIMARY gate only: the module layer is
        # optional by construction, so its absence never makes a completed run PENDING
        summary["status"] = "COMPLETE" if p3_evaluated else "PENDING"
        # how the module layer's own reading came out: never a gate, and `p3_status` is
        # the PRIMARY verdict, so a document walker that resolves gate P3 from this
        # payload can never read the interpretation layer as a second definition of it
        module_reading_evaluated = bool(
            panel and inputs.recurrence is not None
            and (not recurrent_declared or bool(recurrent_modules))
            and not module_support.empty
            and not recurrence_error
        )
        module_reading_passes = bool(
            module_reading_evaluated and module_support.p3_support.any()
        )
        best = None
        if not module_support.empty:
            top = module_support.sort_values(
                ["p3_support", "n_species_agree", "n_members_raw"], ascending=False
            ).iloc[0]
            best = {
                "module": str(top.module),
                "analysis_level": str(top.analysis_level),
                "may_gate": bool(top.may_gate),
                "is_recurrent": bool(top.is_recurrent),
                "n_species_agree": int(top.n_species_agree),
                "n_species_scored": int(top.n_species_scored),
                "passes": bool(top.p3_support),
            }
        summary["gate_p3"] = {
            "layer": "interpretation",
            "primary_evidence": False,
            "role": (
                "the module layer's reading of the same evidence gate P3 is decided on. It is "
                "reported in full and never decides gate P3 or this step's status: P3 is read on "
                "the full motif-intersection model<->raw profile (see summary.primary_validation) "
                "and is invariant to this payload, so emptying the module map can move everything "
                "here and the gate not at all."
            ),
            # the verdict of gate P3 as a document walker resolves it from this payload:
            # the primary verdict, never the module reading's below
            "p3_status": summary["gates"]["P3"],
            "criterion": (
                f"at least one model-defined recurrent primary module (a module whose level is not "
                f"in modules.non_gating_levels) whose uncentered raw chromVAR module score (the "
                f"median C over the module's monomeric members) has the same sign as the uncentered "
                f"model module score in >= {GATE_MIN_SPECIES_AGREE} of "
                f"{species_set.n_expected} species"
            ),
            "module_reading_evaluated": module_reading_evaluated,
            "module_reading_passes": module_reading_passes,
            "unavailable_reasons": module_reading_reasons,
            "comparison_basis": "uncentered",
            "comparison_basis_detail": (
                "both sides of the module reading are read uncentered: raw_module_score (median C "
                "over the module's monomeric members) against model_module_score (the same median "
                "member log2_OR). Centering one side only would compare a cross-species deviation "
                "against an absolute level, and a program that is uniformly Purkinje-elevated in "
                "every species would center to exactly zero and be reported as disagreeing."
            ),
            "module_panel_source": C.display_path(C.module_map_path(config)),
            "recurrent_modules_input": str(inputs.recurrence) if inputs.recurrence else None,
            "recurrent_modules_declared": recurrent_declared,
            "recurrent_modules_used": sorted(recurrent_modules),
            "gating_modules": [spec.module for spec in panel.values() if spec.may_gate],
            "nongating_modules": [spec.module for spec in panel.values() if not spec.may_gate],
            "model_module_score_source": model_source,
            "model_module_score_source_rule": (
                "per row, in module_raw_support.tsv `model_score_source`: `<recurrence file>:"
                "score_<species>` when the upstream table carries that row, otherwise "
                f"`{RECOMPUTED_SCORE_SOURCE}`. The upstream table is the model's recurrent-only "
                "subset, so a panel module the model did not call recurrent has no upstream row and "
                "keeps the recomputed score instead of being blanked."
            ),
            "raw_module_score_definition": (
                "median over the monomeric member motifs of C, with no cross-species centering; the "
                "direct analogue of the model's per-species median member log2_OR"
            ),
            "model_module_score_definition": (
                "FILLED, never replaced: the model's own per-species module score where the "
                "recurrence table carries it (the `score_<species>` columns, s06: uncentered median "
                "member log2_OR), and otherwise the same uncentered median member log2_OR recomputed "
                "from the model profile. A module absent from the upstream table therefore keeps its "
                "recomputed score, so `n_species_scored` and `n_species_agree` describe the same rows."
            ),
            "gating_rule": (
                "a module may be read as agreement (p3_support) only when its level is not in "
                "modules.non_gating_levels AND the model declares it recurrent; a secondary or "
                "comparator module is reported with may_gate=False and can never pass, and no module "
                "at any level can set gate P3"
            ),
            "supplementary_centered_model_reading": (
                "sign_agree_centered / n_species_agree_centered repeat the concordance after each "
                "motif's cross-species mean is removed from BOTH sides "
                "(raw_module_score_centered vs model_module_score_centered); that reading asks "
                "whether the module's cross-species variation tracks the model's, and it is reported "
                "next to the module verdict rather than deciding it, because a program that is "
                "uniformly Purkinje-elevated in every species centers to exactly zero"
            ),
            "verdict_reason": "",
            "best_module": best,
            "modules": [
                {
                    "module": str(row.module), "analysis_level": str(row.analysis_level),
                    "may_gate": bool(row.may_gate), "is_recurrent": bool(row.is_recurrent),
                    "n_species_agree": int(row.n_species_agree),
                    "n_species_agree_centered": int(row.n_species_agree_centered),
                    "n_species_scored": int(row.n_species_scored),
                    "n_members_raw": int(row.n_members_raw), "n_members_model": int(row.n_members_model),
                    "passes": bool(row.p3_support),
                }
                for row in module_support.drop_duplicates("module").itertuples()
            ],
        }
        summary["gate_p3"]["module_signs"] = [
            {
                "module": str(row.module), "species": str(row.species),
                "raw_module_score": _num(row.raw_module_score),
                "raw_module_score_centered": _num(row.raw_module_score_centered),
                "model_module_score": _num(row.model_module_score),
                "model_module_score_centered": _num(row.model_module_score_centered),
                "sign_agree": bool(row.sign_agree),
                "sign_agree_centered": bool(row.sign_agree_centered),
            }
            for row in module_support.itertuples()
        ]
        # A top-level home for the layer's own state, so a reader (or a checker) looking at
        # the summary for "why is the module reading not there" finds it without having to
        # know that the payload lives under `gate_p3`, and without mistaking it for a
        # reason this run is PENDING: the run is COMPLETE whenever gate P3 is evaluable.
        summary["interpretation"] = {
            "layer": "interpretation",
            "primary_evidence": False,
            "available": bool(module_reading_evaluated),
            "passed": (
                None if not module_reading_evaluated else bool(module_reading_passes)
            ),
            "reasons": module_reading_reasons,
            "payload": "gate_p3",
            "table": "module_raw_support.tsv",
            "role": (
                "the module layer's reading of the evidence gate P3 is decided on. It is reported "
                "here and in `gate_p3`, and it decides nothing: gate P3, this step's status, the "
                "raw profile and the similarity table are all computed without it."
            ),
        }
        # Self-describing module reading: its FAIL is never allowed to be read as "the raw
        # data disagrees" when the real reason is that no recurrent module was at a gating
        # level, or that the module layer is not available at all.
        gating_recurrent = [
            module for module in recurrent_modules
            if panel.get(module) is not None and panel[module].may_gate
        ]
        if not module_reading_evaluated:
            reasons = [
                f"{reason.get('reason')}: {reason.get('detail')}"
                for reason in module_reading_reasons
            ]
            summary["gate_p3"]["verdict_reason"] = (
                "not evaluable: " + "; ".join(reasons) if reasons
                else "not evaluable: no recurrent module of the panel has a scorable raw profile"
            )
        elif module_reading_passes:
            summary["gate_p3"]["verdict_reason"] = (
                f"passed: {best['module']} (level={best['analysis_level']}, recurrent) agrees in "
                f"{best['n_species_agree']}/{species_set.n_expected} species"
            )
        elif not gating_recurrent:
            summary["gate_p3"]["verdict_reason"] = (
                f"failed without a disagreement: the model's recurrent modules "
                f"{sorted(recurrent_modules)} resolve to no module at a gating level of the "
                f"{ANALYSIS} panel, and only a gating-level module may be read as agreement"
            )
        else:
            gating_rows = module_support[module_support.may_gate]
            if gating_rows.empty:
                summary["gate_p3"]["verdict_reason"] = (
                    f"failed without a disagreement: the model's recurrent gating-level modules "
                    f"{sorted(gating_recurrent)} have no monomeric member in the raw profile"
                )
            else:
                best_gating = gating_rows.sort_values(
                    "n_species_agree", ascending=False
                ).iloc[0]
                summary["gate_p3"]["verdict_reason"] = (
                    f"failed on the evidence: the best gating-level recurrent module "
                    f"{best_gating.module} agrees in {int(best_gating.n_species_agree)}/"
                    f"{species_set.n_expected} species, below the required "
                    f"{GATE_MIN_SPECIES_AGREE}"
                )

        C.save_table(profiles, out_dir / "purkinje_raw_profile.tsv", sep="\t")
        C.save_table(similarity, out_dir / "model_vs_raw_similarity.tsv", sep="\t")
        C.save_table(module_support, out_dir / "module_raw_support.tsv", sep="\t")

        # The figure is deliberately NOT written by this step. Its module panel put
        # the raw chromVAR score (about 4-30) and the model score (about 0-2) on
        # one y-axis, so every model marker was pinned to zero beside tall raw
        # bars -- a module with 6 of 6 species agreeing read as "the raw signal is
        # strong and the model sees nothing". The per-species scatter also carried
        # an OLS line while the reported statistic was Spearman. The visualisation
        # now belongs to:
        #   celltype_validation_prep.py      builds the long + support tables
        #   plot_celltype_validation.py      the A/B/C plate, shared by both lineages
        summary["outputs"]["figures"] = []
        summary["outputs"]["figures_note"] = (
            "produced by script/plot_celltype_validation.py from the tables below; "
            "this step no longer writes a figure (see its docstring for why)"
        )

        summary["outputs"]["tables"] = [
            "purkinje_raw_profile.tsv", "model_vs_raw_similarity.tsv",
            "module_raw_support.tsv", "purkinje_raw_audit.tsv",
        ]
        summary["audit"].update(
            {
                "n_rows": int(len(audit.rows)),
                "celltypes_excluded": [
                    {"species": row["species"], "celltype": row["item"], "n_cells": row["n_cells"]}
                    for row in audit.rows if row["kind"] == "celltype_excluded"
                ],
                "pseudobulks_dropped": [
                    {"species": row["species"], "pseudobulk": row["item"], "n_cells": row["n_cells"]}
                    for row in audit.rows if row["kind"] == "pseudobulk_dropped"
                ],
                "species_skipped": [
                    {"species": row["species"], "reason": row["item"]}
                    for row in audit.rows if row["kind"] == "species_skipped"
                ],
                "modules_excluded": [
                    {"module": row["item"], "reason": row["detail"]}
                    for row in audit.rows
                    if row["kind"] in {"module_panel", "module_recurrence", "module_support",
                                       "module_reading"}
                    and row["status"] in {"excluded", "blocked"}
                ],
            }
        )
        dropped_species = [
            {"species": row["species"], "item": row["item"], "n_cells": row["n_cells"],
             "reason": row["detail"]}
            for row in audit.rows if row["kind"] == "species_excluded"
        ]
        if dropped_species:  # only a config with an eligibility block can produce these
            summary["audit"]["species_excluded"] = dropped_species
        C.save_table(audit.frame(), out_dir / "purkinje_raw_audit.tsv", sep="\t")
        C.save_json(summary, out_dir / "purkinje_validation_summary.json")

        C.log(
            "STATUS:", summary["status"], "| gate P3 (primary):", summary["gates"]["P3"],
            f"| species scored: {len(scored)}/{species_set.n_expected}",
            "| rho>0 in", summary["primary_validation"]["n_species_rho_positive"], "species",
            "| median rho:", summary["model_vs_raw"]["median_rho"],
            "| module reading:",
            ("unavailable" if not module_reading_evaluated
             else ("PASS" if module_reading_passes else "FAIL")),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
