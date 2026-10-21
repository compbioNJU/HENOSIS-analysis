"""Oligodendrocyte raw chromVAR validation (readme section 28, gate O3).

WHY this step exists. `s08` can only show that the model *says* the oligodendrocyte
top-ranked peaks carry a recurrent regulatory program. This step asks an
independent measurement -- chromVAR deviations over every QC peak of the species,
per pseudobulk -- the same question: does raw accessibility place the same motifs
in the same direction inside oligodendrocytes relative to the other cerebellar
cell types? The raw side never saw the model ranking, the Top500, the matched
background or the enrichment, so it can contradict the model but cannot
manufacture it. This is a deliberate repeat of `s07` for the second glial
analysis (readme section 28: "a fully independent repeat of the Purkinje pipeline").

WHY every comparison is within species. chromVAR Z is produced per species on
that species' own peak universe, its own GC background and its own pseudobulk
set. A Z of +2 in mouse and a Z of +2 in human are therefore not on a common
scale, and readme section 26 forbids comparing raw strength across species.
Only the rank correlation between the model profile and the raw profile,
computed separately inside each species, is interpretable.

WHY the "other celltype" average is equal-weighted by celltype. `oligo` is one
population among several with very different cell numbers; weighting by cells
would let whichever other population was sequenced most dominate the contrast
and turn the statistic into a comparison against that one population instead of
against "the rest of the cerebellum".

WHY both sides of gate O3 are read uncentered. The gate asks whether raw
accessibility places a model-defined recurrent family in the same direction as
the model does, so the raw family score -- the median C over the family's
monomeric members -- has to be the direct analogue of the model family score,
which is `s08`'s per-species median member log2_OR (recomputed here from the s01
model profile) and carries no cross-species centering. Subtracting a per-motif
baseline from one side only makes the two signs incomparable: a program that is
uniformly oligodendrocyte-elevated in every species centers to exactly zero on
the raw side and would be reported as disagreeing with a model that calls it
positive everywhere. That uncentered median is the statistic readme section
25/B4 defines recurrence on. `s07` states its gate on exactly the same basis,
so the two validation steps cannot disagree about what a family score means.

The cross-species centered family scores are still computed for both sides --
that is readme section 11 applied to the species axis, and it is what
`common.family_trajectory` does when its `species` and `stage` roles are swapped
for a single-stage (single-period) analysis -- and reported as the supplementary
`sign_agree_centered` reading, which asks whether a family's cross-species
*variation* tracks the model's.

WHY a non-finite chromVAR Z is dropped from a correlation but never from a table.
chromVAR emits NaN for a motif whose peak matches are too few or too biased, and a
single NaN would otherwise propagate into a NaN rho that is still reported as an
`ok` species. Non-finite motif pairs are therefore removed before the Spearman,
counted in the similarity table and in the audit, and never counted as family
members. The same rule governs the module panel below: a member whose raw C or
model log2_OR is non-finite is excluded from that module's median for that
species, counted in `n_members_nonfinite_raw` / `n_members_nonfinite_model` and
in `oligo_raw_audit.tsv`, and never read as a zero and never counted as a member.

WHY the module panel is an interpretation and never a second test. The primary
evidence of this step is the unbiased 879-motif contrast and the recurrent
families `s08` defines over the frozen TF-family map. v2 additionally NAMES the
cell-type-specific motif modules of `config/celltype_motif_module_map.tsv` for the
oligodendrocyte panel (SOX and bHLH_Ebox at `primary`, NKX and POU3 at
`secondary`, NFI as a `comparator`), and this step reads them through `common`'s
accessors on exactly the sign-agreement basis gate O3 is read on: the uncentered
median C over a module's monomeric members against the uncentered median member
log2_OR, agreed per species and counted over the species this run analyses. The
two read-outs therefore cannot disagree about what a module score means. The
panel is INTERPRETATION -- it creates no gate, replaces no gate and overrides no
gate, so emptying the module map (or pointing `modules.file` at a map with no
rows) leaves every gate status, every similarity, every margin and every null
unchanged. Only a module whose level `common.module_is_gating` accepts may carry a
support verdict; a `secondary` or `comparator` module is reported with
`gating=False` and can never satisfy a PASS/FAIL rule. A module pools binding
models rather than measuring a factor, so its score is stated as a program
(`bHLH_Ebox` -> "bHLH/E-box-associated program", `SOX` -> the SOX-associated
program) and never as the activity of one member -- never `OLIG2 activity`, never
`SOX10 activity` (readme section 9).

WHY the expected oligodendrocyte factors are not the endpoint. readme section
28 forbids making SOX / OLIG / MYRF the target: the primary statistic is the
whole 879-motif profile, and those factors are only a sanity check performed
upstream in `s08`.

WHY the species set is read from the config and not written out here. v2 declares
its species-inclusion rule in the config's `eligibility` block (v1 declared the same
rule under `species_filter`), and a species enters a cell-type analysis only if it
clears BOTH floors: at least `min_target_cells` cells of the target type in the
integrated h5ad AND at least `min_ranked_peaks` ranked peaks in its ranked model CSV
(a ranking shorter than the peak floor cannot define a Top500 foreground at all).
This step therefore *derives* the species it runs over from the ranked model
manifest through `common.apply_species_filter`, which enforces both floors, and never
from a fixed species list: under v2 the oligodendrocyte population clears both floors
in every species, but the code would drop and audit any species that did not --
exactly as the Purkinje step does for bonobo, whose 4 cells and 298 ranked peaks fail
eligibility. A dropped species is named in `oligo_raw_audit.tsv` with the cell count
AND the peak count that removed it, and the gate denominator is the number of species
that survived. A config that declares no such block (v1) declares no exclusion at
all: the step then runs over the whole species axis as before, writes no extra
summary key and no extra audit row, which is what leaves a v1 run byte-identical.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

STEP = "09_oligo_validation"
ANALYSIS = "Oligo"
STEM = "oligo"
TARGET_CELLTYPE = C.ANALYSIS_ORG_CELLTYPE[ANALYSIS]
# Gate O3's agreement floor, readme section 25/B4: "the same sign in >= 4 of 6
# species". It is a fixed criterion -- a count of agreeing species -- and NOT a
# property of the data, so it is deliberately not derived from the species count.
# The denominator it is read against IS derived (see `species_expected` in
# `_base_summary`), because the config's eligibility rule decides how many species
# the analysis actually runs over; the floor stays the readme's absolute 4.
#
# A fixed floor read against a derived denominator can be unsatisfiable: a config
# whose filter trims the denominator below 4 leaves `n_species_agree` unable to
# ever reach the floor, so the criterion has no satisfying assignment. That is a
# statement about the config, not about the model, and it must never be published
# as a FAIL. `gate_o3_untestable` detects it; the gate is then reported PENDING,
# with the floor and the filtered count named in `pending_reasons`, in the gate's
# criterion and in an audit row.
GATE_MIN_SPECIES_AGREE = 4

PROFILE_COLUMNS = [
    "species", "motif_id", "motif_name", "log2_OR", "C", "c_simple_all",
    "oligo_mean_z", "other_equal_weighted_mean_z",
    "oligo_mean_z_simple", "other_equal_weighted_mean_z_simple",
    "n_oligo_pseudobulk", "n_oligo_cells",
    "n_other_celltypes", "n_other_pseudobulk", "n_other_cells",
]
SIMILARITY_COLUMNS = [
    "species", "rho", "p", "n_motifs", "n_motifs_dropped_nonfinite",
    "n_pseudobulk_target", "n_pseudobulk_other",
    "target_cells", "other_cells", "n_celltypes_other", "rho_simple", "p_simple",
    "p_fdr_bh", "status",
]
# raw_family_score / model_family_score are the uncentered medians the gate is read
# off; the *_centered block is the descriptive cross-species-centered control.
FAMILY_COLUMNS = [
    "family", "analysis_level", "species", "raw_family_score", "model_family_score",
    "sign_agree", "raw_family_score_centered", "model_family_score_centered",
    "sign_agree_centered", "n_members_raw", "n_members_model", "n_species_agree",
    "n_species_agree_centered", "n_species_scored", "o3_support", "model_score_source",
]
# The module panel repeats the family panel's columns on purpose: `module` is the
# module's name where the family panel carried a TF family, `level` its level in
# the frozen map, `gating` whether `common.module_is_gating` lets that level take
# part in a PASS/FAIL rule, and `module_support` the agreement verdict -- which is
# False for every non-gating module, whatever its species agree on.
MODULE_COLUMNS = [
    "module", "level", "gating", "provenance", "member_basis", "species",
    "raw_module_score", "model_module_score", "sign_agree",
    "raw_module_score_centered", "model_module_score_centered", "sign_agree_centered",
    "n_members_raw", "n_members_model",
    "n_members_finite_raw", "n_members_finite_model",
    "n_members_nonfinite_raw", "n_members_nonfinite_model",
    "n_species_agree", "n_species_agree_centered", "n_species_scored",
    "module_support", "program_wording", "model_score_source",
]
MODULE_TABLE = "module_raw_support.tsv"
AUDIT_COLUMNS = ["species", "kind", "item", "n_cells", "n_pseudobulk", "detail", "status"]

# Reference data-viz palette: the diverging pair encodes the SIGN of a family
# score (never family identity, which is carried by position and by text).
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
# inputs
# --------------------------------------------------------------------------


@dataclass
class OligoInputs:
    model_scores: Path
    recurrent: Path
    chromvar: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, Path] = field(default_factory=dict)


def collect_inputs(config: dict, species_list: list[str]) -> OligoInputs:
    results = C.config_path(config, "results")
    inputs = OligoInputs(
        model_scores=results / "01_model_motif" / "all_model_motif_scores.tsv",
        recurrent=results / "08_oligo_model" / "recurrent_families.tsv",
    )
    for species in species_list:
        z_path = results / "04_chromvar" / species / "chromvar_z.tsv.gz"
        meta_path = results / "03_chromvar_input" / species / "pseudobulk_metadata.tsv"
        if z_path.exists():
            inputs.chromvar[species] = z_path
        if meta_path.exists():
            inputs.metadata[species] = meta_path
    return inputs


def _axis_order(species: "list[str]") -> list[str]:
    """Deduplicate species and put them back on the canonical axis order.

    The filter is read off the ranked model manifest, whose rows are sorted
    alphabetically; the analysis is read off the canonical species axis, so a
    derived species set is always re-expressed in that order to keep the tables
    and the RNG-free iteration deterministic.
    """
    order = {name: index for index, name in enumerate(C.SPECIES_ORDER)}
    return sorted(dict.fromkeys(str(name) for name in species),
                  key=lambda name: (order.get(name, len(order)), name))


def _eligibility_block(config: dict) -> dict:
    """The species-inclusion rule under whichever key the config declares it.

    v2 renamed the v1 `species_filter` key to `eligibility`; `common` accepts and
    enforces either, so this step reads the same block through the same helpers and
    never keeps a second copy of the rule.
    """
    return config.get("eligibility") or config.get("species_filter") or {}


def analysis_species(config: dict, audit: Audit) -> tuple[list[str], dict | None]:
    """The species this analysis may run over, and the filter record (None under v1).

    A species enters a cell-type analysis only if it clears BOTH floors the config
    declares in its eligibility rule (v2: `eligibility`, v1: `species_filter`): at
    least `min_target_cells` cells of the target type in the integrated h5ad AND at
    least `min_ranked_peaks` ranked peaks in its ranked model CSV. The peak floor
    matters on its own because a ranking shorter than it cannot define a Top500
    foreground at all, which is exactly what removes
    `Purkinje__bonobo__all_development` (4 cells AND 298 ranked peaks) under v2.

    The species axis is therefore resolved here, from the ranked model manifest
    through `common.apply_species_filter`, instead of being written out: under v2
    every species clears both floors for `oligo`, but a species that did not would be
    removed and named in the audit with the criterion (cells or peaks) that removed
    it, and the gate denominator would shrink with it.

    A config that declares no block -- v1 -- declares no exclusion, so the whole
    species axis is returned, the record is None and the caller writes nothing extra.
    That is what leaves a v1 run byte-identical.

    The record is returned instead of being written to the summary directly because
    only the caller knows the config declares a block at all.
    """
    block = _eligibility_block(config)
    if not block:
        return list(C.SPECIES_ORDER), None

    floor = C.min_cells_per_species(config)
    peak_floor = C.min_ranked_peaks(config)
    column = str(block.get("celltype_column", "Org_Celltype"))
    block_name = "eligibility" if config.get("eligibility") else "species_filter"

    declared: list[str] = []
    kept: list[str] = []
    # one entry per dropped species, whichever model and whichever floor removed it
    dropped: dict[str, dict] = {}
    manifest_note = ""
    try:
        manifest = C.ranking_files(config)
        manifest = manifest[manifest.analysis.astype(str).str.lower() == ANALYSIS.lower()].copy()
        declared = _axis_order(manifest.species.astype(str).tolist())
        kept_manifest, filter_audit = C.apply_species_filter(config, manifest)
        kept = _axis_order(kept_manifest.species.astype(str).tolist())
        for row in filter_audit.itertuples():
            dropped.setdefault(str(row.species), dict(
                n_cells=int(row.n_cells), n_ranked=int(row.n_ranked),
                reason=str(row.action), model=str(row.model),
            ))
    except Exception as exc:  # noqa: BLE001 - an unreadable manifest is reported, not raised
        # the peak floor cannot be read off a manifest that will not load, but the cell
        # floor is a property of the h5ads and still applies; the shortfall is named
        # rather than silently turning the rule into a cell-only one
        manifest_note = (
            f"the ranked model manifest could not be read ({type(exc).__name__}: {exc}), so the "
            f"{peak_floor}-peak floor could not be applied; the {floor}-cell floor still is"
        )
        audit.add(
            "-", "species_filter", "ranked model manifest",
            detail=manifest_note + "; no primary verdict changes", status="blocked",
        )
        kept, dropped_cells = C.species_passing_filter(config, TARGET_CELLTYPE)
        for species, n_cells in dropped_cells:
            dropped[str(species)] = dict(
                n_cells=int(n_cells), n_ranked=None,
                reason=f"{int(n_cells)} target cells < {floor}", model="",
            )
    for species in _axis_order(list(dropped)):
        info = dropped[species]
        audit.add(
            species, "species_filter", TARGET_CELLTYPE,
            n_cells=info["n_cells"],
            detail=(
                f"{column} `{TARGET_CELLTYPE}`: {info['reason']}; the species is dropped "
                f"from the {ANALYSIS} analysis (model {info['model']}) because it has "
                f"fewer than {floor} `{TARGET_CELLTYPE}` cells or fewer than {peak_floor} "
                f"ranked peaks"
            ),
            status="excluded",
        )
    return kept, {
        "celltype": TARGET_CELLTYPE,
        "celltype_column": column,
        "block": block_name,
        "min_cells_per_species": floor,
        "min_target_cells": floor,
        "min_ranked_peaks": peak_floor,
        "rule": (
            f"a species enters the {ANALYSIS} analysis only if it has at least "
            f"{floor} cells with {column} == `{TARGET_CELLTYPE}` in its integrated h5ad "
            f"AND at least {peak_floor} ranked peaks in its ranked model CSV"
        ),
        "species_kept": kept,
        "species_excluded": [
            {
                "species": species,
                "n_cells": dropped[species]["n_cells"],
                "n_ranked": dropped[species]["n_ranked"],
                "reason": dropped[species]["reason"],
            }
            for species in _axis_order(list(dropped))
        ],
        "species_without_a_model": [
            name for name in C.SPECIES_ORDER if name not in declared
        ],
        "n_species_on_axis": int(len(C.SPECIES_ORDER)),
        "n_species_analysed": int(len(kept)),
        "n_species_excluded": int(len(dropped)),
    } | ({"manifest_error": manifest_note} if manifest_note else {})


def _as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def load_model_profiles(
    config: dict, path: Path, topk: int, audit: Audit
) -> dict[str, pd.DataFrame]:
    """One log2_OR profile per species for the Oligo model."""
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
        profiles[str(species)] = (
            group[columns].drop_duplicates("motif_id").sort_values("motif_id").reset_index(drop=True)
        )
    return profiles


def load_pseudobulk_metadata(path: Path, species: str, audit: Audit) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    if "pseudobulk" not in frame.columns:
        index_name = frame.columns[0] if not frame.index.name else frame.index.name
        if index_name and frame.index.is_unique:
            frame = frame.reset_index()
            frame = frame.rename(columns={frame.columns[0]: "pseudobulk"})
            audit.add(species, "metadata", "pseudobulk column", detail="taken from the file index")
        else:
            raise RuntimeError(f"{path} has no `pseudobulk` column and no usable index")
    missing = sorted({"celltype", "n_cells"} - set(frame.columns))
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {missing}")
    frame = frame.copy()
    frame["pseudobulk"] = frame.pseudobulk.astype(str)
    frame["celltype"] = frame.celltype.astype(str)
    frame["n_cells"] = pd.to_numeric(frame.n_cells, errors="coerce").fillna(0).astype(int)
    if frame.pseudobulk.duplicated().any():
        audit.add(
            species, "metadata", "duplicate pseudobulk rows",
            detail=f"{int(frame.pseudobulk.duplicated().sum())} rows dropped, first kept",
            status="substituted",
        )
        frame = frame.drop_duplicates("pseudobulk", keep="first")
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
# raw oligodendrocyte profile
# --------------------------------------------------------------------------


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(values.shape[0], np.nan)
    return (values * weights).sum(axis=1) / total


def raw_oligo_profile(
    config: dict, species: str, z: pd.DataFrame, meta: pd.DataFrame, audit: Audit
) -> tuple[pd.DataFrame | None, dict]:
    """C_{s,m} = weighted mean Z over oligo pseudobulks minus the equal-weighted
    mean over the other celltypes of (weighted mean Z over that celltype's pseudobulks).

    `c_simple_all` is the fully unweighted variant (plain mean of the oligo
    pseudobulks minus the equal-weighted mean of the other celltypes' plain
    means), kept so the n_cells weighting choice is visible in the output rather
    than hidden inside it.
    """
    min_cells_ct = int(config["chromvar"]["min_cells_per_celltype"])
    min_cells_pb = int(config["chromvar"]["min_cells_per_pseudobulk"])
    # every pseudobulk the metadata does declare, before any filter -- the orphan
    # test below must compare the Z matrix against what the metadata SAYS, not
    # against the subset that survived the size filter
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
    # whose metadata spells the target two ways (e.g. `oligo` and `Oligo`) must not have
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
        return None, {}
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
        return None, {}

    motifs = list(dict.fromkeys(z.index))
    value = z.loc[motifs]

    target_ids = target.pseudobulk.tolist()
    target_weights = target.set_index("pseudobulk").n_cells.loc[target_ids].to_numpy(dtype=float)
    target_matrix = value[target_ids].to_numpy(dtype=float)
    oligo_mean = _weighted_mean(target_matrix, target_weights)
    oligo_simple = target_matrix.mean(axis=1)

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
        n_oligo_pseudobulk=len(target_ids),
        n_oligo_cells=int(target.n_cells.sum()),
        n_other_celltypes=len(other_celltypes),
        n_other_pseudobulk=n_other_pb,
        n_other_cells=n_other_cells,
    )
    frame = pd.DataFrame(
        {
            "species": species,
            "motif_id": motifs,
            "C": oligo_mean - other_mean,
            "c_simple_all": oligo_simple - other_mean_simple,
            "oligo_mean_z": oligo_mean,
            "other_equal_weighted_mean_z": other_mean,
            "oligo_mean_z_simple": oligo_simple,
            "other_equal_weighted_mean_z_simple": other_mean_simple,
            "n_oligo_pseudobulk": stats["n_oligo_pseudobulk"],
            "n_oligo_cells": stats["n_oligo_cells"],
            "n_other_celltypes": stats["n_other_celltypes"],
            "n_other_pseudobulk": stats["n_other_pseudobulk"],
            "n_other_cells": stats["n_other_cells"],
        }
    )
    return frame, stats


# --------------------------------------------------------------------------
# family aggregation
# --------------------------------------------------------------------------


def _sign_agreement(raw: pd.Series, model: pd.Series) -> pd.Series:
    """Agreement of two family-score signs, with an explicit tie rule.

    A zero or missing score carries no direction, so it is never counted as
    agreement: sign() of it is 0 and the product is not > 0. This matters because
    a motif whose cross-species centering lands exactly on 0 (or a family member
    whose chromVAR Z is NaN) must not be scored as if it disagreed.
    """
    return (np.sign(raw.fillna(0.0)) * np.sign(model.fillna(0.0))) > 0


def _empty_centered_scores(value: str) -> pd.DataFrame:
    """The typed empty frame `centered_family_scores` returns when it can score nothing.

    Same columns and dtypes as the real result, so a caller that merges it sees
    "no row" rather than a schema error: an interpretation input that annotates
    nothing must degrade an interpretation, never raise.
    """
    return pd.DataFrame(
        {
            "species": pd.Series(dtype=object), "family": pd.Series(dtype=object),
            f"{value}_family_score": pd.Series(dtype=float),
            "n_members": pd.Series(dtype=int), "analysis_level": pd.Series(dtype=object),
        }
    )


def centered_family_scores(
    profile: pd.DataFrame,
    value: str,
    family_table: pd.DataFrame,
    significant_col: str | None = None,
) -> pd.DataFrame:
    """Median member score after each motif's cross-species mean is removed.

    `common.family_trajectory` removes a per-motif baseline *within* a species
    across `stage`. An Oligo model has exactly one stage, so the two columns are
    swapped: the constant `species` column is the analysis label and `stage`
    carries the real species. The subtracted baseline is then that motif's
    cross-species mean, i.e. readme section 11 applied to the species axis.

    Non-finite values are removed before the aggregation, so `n_members` counts
    the finite members the median was actually taken over.

    An annotation frame that shares no motif with the profile is handled here
    rather than inside `family_trajectory`, which raises on it: that case is an
    interpretation input that matched nothing (the module map's motif_ids are all
    unknown to the profile, say), so it must return the empty typed frame exactly as
    `uncentered_family_scores` does, and must never take a primary artefact down.
    """
    values = pd.to_numeric(profile[value], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(values)
    work = pd.DataFrame(
        {
            "species": ANALYSIS,
            "stage": profile["species"].astype(str).to_numpy()[keep],
            "motif_id": profile["motif_id"].to_numpy()[keep],
            value: values[keep],
        }
    )
    if significant_col is not None and significant_col in profile.columns:
        work["significant"] = _as_bool(profile[significant_col]).to_numpy()[keep]
    if work.empty:
        return _empty_centered_scores(value)
    # the guard `family_trajectory` lacks: it filters to the annotated motifs and
    # raises when that leaves nothing, so the intersection is tested here, on the
    # very same annotation frame it would be given
    annotations = family_table[["motif_id", "family"]].copy()
    annotations["family"] = annotations.family.fillna("").astype(str)
    annotations = annotations[annotations.family != ""]
    if not work["motif_id"].astype(str).isin(set(annotations.motif_id.astype(str))).any():
        return _empty_centered_scores(value)
    out = C.family_trajectory(work, family_table, value=value, motif_col="motif_id")
    # the constant column carried the analysis label; the real species sits in `stage`
    out = out.drop(columns=["species"]).rename(
        columns={"stage": "species", "family_score": f"{value}_family_score"}
    )
    if "n_significant" in out.columns:
        out = out.rename(columns={"n_significant": "n_members_significant"})
    return out


def uncentered_family_scores(
    profile: pd.DataFrame,
    value: str,
    family_table: pd.DataFrame,
) -> pd.DataFrame:
    """Per (family, species) median of `value` over the family's monomeric members.

    No centering: this is the statistic readme section 25/B4 defines family
    recurrence on (family score > 0 in >= 4 of 6 species), which is why gate O3 is
    read off it. Non-finite member values are dropped before the median and are not
    counted in `n_members`.
    """
    annotations = family_table[["motif_id", "family", "analysis_level", "member_type"]].copy()
    annotations["family"] = annotations.family.fillna("")
    values = pd.to_numeric(profile[value], errors="coerce").to_numpy(dtype=float)
    work = pd.DataFrame(
        {
            "species": profile["species"].astype(str).to_numpy(),
            "motif_id": profile["motif_id"].to_numpy(),
            value: values,
        }
    )
    work = work[np.isfinite(values)]
    merged = work.merge(annotations, on="motif_id", how="left")
    merged = merged[(merged.family != "") & (merged.member_type == "monomer")]
    if merged.empty:
        return pd.DataFrame(
            columns=["species", "family", "analysis_level", "family_score", "n_members"]
        )
    out = merged.groupby(["species", "family"], as_index=False).agg(
        family_score=(value, "median"), n_members=(value, "size")
    )
    levels = merged.drop_duplicates("family").set_index("family").analysis_level
    out["analysis_level"] = out.family.map(levels)
    return out


def recurrent_family_support(
    recurrent: pd.DataFrame,
    raw_profile: pd.DataFrame,
    model_long: pd.DataFrame,
    family_table: pd.DataFrame,
    model_source: str,
) -> pd.DataFrame:
    """Per recurrent family and species: the uncentered raw score against the model score.

    The gate compares like with like. `raw_family_score` is the uncentered median C
    over the family's monomeric members (readme section 26), the direct analogue
    of `model_family_score`, which is s08's own per-species family score (the
    uncentered median member log2_OR, readme section 25) recomputed from the
    model profile. `raw_family_score_centered` and `model_family_score_centered` are
    the same two quantities with each motif's cross-species mean removed; their
    agreement is the supplementary `sign_agree_centered` reading, never the gate: a
    family elevated in every species centers to exactly 0, while its uncentered
    score and the model score agree in every species.
    """
    raw_uncentered = uncentered_family_scores(raw_profile, "C", family_table)
    model_uncentered = uncentered_family_scores(model_long, "log2_OR", family_table)
    raw_centered = centered_family_scores(raw_profile, "C", family_table)
    model_centered = centered_family_scores(model_long, "log2_OR", family_table, "significant")

    records: list[dict] = []
    for family in dict.fromkeys(recurrent.family.tolist()):
        raw_rows = raw_uncentered[raw_uncentered.family == family]
        if raw_rows.empty:
            continue
        model_rows = model_uncentered[model_uncentered.family == family]
        raw_centered_rows = raw_centered[raw_centered.family == family]
        model_centered_rows = model_centered[model_centered.family == family]
        merged = raw_rows[["species", "family_score", "n_members"]].rename(
            columns={"family_score": "raw_family_score", "n_members": "n_members_raw"}
        ).merge(
            model_rows[["species", "family_score", "n_members"]].rename(
                columns={"family_score": "model_family_score", "n_members": "n_members_model"}
            ),
            on="species",
            how="left",
        ).merge(
            raw_centered_rows[["species", "C_family_score"]].rename(
                columns={"C_family_score": "raw_family_score_centered"}
            ),
            on="species",
            how="left",
        ).merge(
            model_centered_rows[["species", "log2_OR_family_score"]].rename(
                columns={"log2_OR_family_score": "model_family_score_centered"}
            ),
            on="species",
            how="left",
        )
        merged["sign_agree"] = _sign_agreement(merged.raw_family_score, merged.model_family_score)
        merged["sign_agree_centered"] = _sign_agreement(
            merged.raw_family_score_centered, merged.model_family_score_centered
        )
        n_agree = int(merged.sign_agree.sum())
        n_agree_centered = int(merged.sign_agree_centered.sum())
        n_scored = int((merged.raw_family_score.notna() & merged.model_family_score.notna()).sum())
        levels = family_table.loc[family_table.family == family, "analysis_level"]
        records.extend(
            dict(
                family=family,
                analysis_level=str(levels.iloc[0]) if len(levels) else "unassigned",
                species=str(row.species),
                raw_family_score=float(row.raw_family_score),
                model_family_score=float(row.model_family_score) if pd.notna(row.model_family_score) else np.nan,
                sign_agree=bool(row.sign_agree),
                raw_family_score_centered=(
                    float(row.raw_family_score_centered)
                    if pd.notna(row.raw_family_score_centered) else np.nan
                ),
                model_family_score_centered=(
                    float(row.model_family_score_centered)
                    if pd.notna(row.model_family_score_centered) else np.nan
                ),
                sign_agree_centered=bool(row.sign_agree_centered),
                n_members_raw=int(row.n_members_raw),
                n_members_model=int(row.n_members_model) if pd.notna(row.n_members_model) else 0,
                n_species_agree=n_agree,
                n_species_agree_centered=n_agree_centered,
                n_species_scored=n_scored,
                o3_support=bool(n_agree >= GATE_MIN_SPECIES_AGREE),
                model_score_source=model_source,
            )
            for row in merged.itertuples()
        )
    frame = pd.DataFrame(records, columns=FAMILY_COLUMNS)
    return _sort_family_frame(frame)


def _sort_family_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    order = {species: index for index, species in enumerate(C.SPECIES_ORDER)}
    return (
        frame.assign(_order=frame.species.map(order).fillna(len(order)))
        .sort_values(["o3_support", "n_species_agree", "family", "_order"],
                     ascending=[False, False, True, True])
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def load_recurrent_families(path: Path, audit: Audit) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    if "family" not in frame.columns:
        raise RuntimeError(f"{path} has no `family` column (columns: {list(frame.columns)})")
    frame = frame.copy()
    frame["family"] = frame.family.astype(str)
    # `s08` writes the already-filtered table under a `recurrent` flag; `is_recurrent`
    # is accepted as well so an unfiltered variant cannot silently pass as recurrent.
    flag = next((name for name in ("is_recurrent", "recurrent") if name in frame.columns), None)
    if flag is not None:
        keep = _as_bool(frame[flag])
        dropped = sorted(frame.loc[~keep, "family"].astype(str).unique())
        if dropped:
            audit.add(
                "-", "recurrent_filter", f"{flag}==False",
                detail=f"excluded {len(dropped)} families: {dropped}", status="excluded",
            )
        frame = frame[keep]
    else:
        audit.add(
            "-", "recurrent_filter", "recurrence flag absent",
            detail="every row of recurrent_families.tsv is treated as recurrent", status="substituted",
        )
    frame = frame[frame.family != ""]
    # a long (family, species, model_family_score) table is legitimate input, and so
    # is a wide one keyed by family alone; de-duplicating on `family` would keep one
    # row of a long table and silently discard every other species' model score, so
    # the key is the full row identity
    key = ["family", "species"] if "species" in frame.columns else ["family"]
    duplicates = int(frame.duplicated(key).sum())
    if duplicates:
        audit.add(
            "-", "recurrent_filter", "duplicate family rows",
            detail=f"{duplicates} duplicate rows dropped (first kept); key={key}",
            status="substituted",
        )
    return frame.drop_duplicates(key).reset_index(drop=True)


def model_scores_from_recurrent(
    frame: pd.DataFrame, species_list: list[str]
) -> pd.DataFrame | None:
    """A per-species model family score carried by recurrent_families.tsv, if it has one.

    The wide layout is recognised by carrying one bare column per species *this run*
    analyses. That set comes from the config's species filter, not from a fixed
    species count, so a filtered analysis recognises its own wide table and a table
    that is wide for the whole axis is still handled when nothing was filtered out.
    """
    if "family" not in frame.columns:
        return None
    species_columns = [species for species in species_list if species in frame.columns]
    if species_columns and len(species_columns) == len(species_list):
        return frame.melt(
            id_vars=["family"], value_vars=species_columns,
            var_name="species", value_name="model_family_score",
        )
    score_columns = [name for name in ("model_family_score", "family_score", "score") if name in frame.columns]
    if "species" in frame.columns and score_columns:
        return frame[["family", "species", score_columns[0]]].rename(
            columns={score_columns[0]: "model_family_score"}
        )
    return None


def apply_provided_model_scores(
    support: pd.DataFrame, provided: pd.DataFrame
) -> pd.DataFrame:
    """Swap the recomputed model family score for the one the model reported.

    `provided` carries the model's own uncentered per-species family score, so the
    sign comparison stays on the uncentered raw score. The agreement counts and the
    scored denominator are recomputed on the merged frame, so a family/species row
    without a provided score is counted as unscored rather than as a stale total.
    """
    if support.empty:
        return support
    merged = support.drop(columns=["model_family_score", "sign_agree"]).merge(
        provided[["family", "species", "model_family_score"]], on=["family", "species"], how="left"
    )
    merged["sign_agree"] = _sign_agreement(merged.raw_family_score, merged.model_family_score)
    merged["_scored"] = merged.raw_family_score.notna() & merged.model_family_score.notna()
    grouped = merged.groupby("family", sort=False)
    merged["n_species_agree"] = grouped.sign_agree.transform("sum").astype(int)
    merged["n_species_agree_centered"] = grouped.sign_agree_centered.transform("sum").astype(int)
    merged["n_species_scored"] = grouped._scored.transform("sum").astype(int)
    merged["o3_support"] = merged.n_species_agree >= GATE_MIN_SPECIES_AGREE
    return _sort_family_frame(merged.drop(columns="_scored")[FAMILY_COLUMNS])


# --------------------------------------------------------------------------
# module panel -- the v2 interpretation layer, read off the SAME statistic
# --------------------------------------------------------------------------


# A module pools many motifs that share a binding model, so its score is a
# program-level statement and never a factor-level one. The table below carries
# the phrasings the frozen panel requires (readme section 9: the bHLH/E-box
# module "can only be interpreted as a bHLH/E-box-associated program, not as OLIG2 activity"). A
# module with no declared alias falls back to "<module>-associated program" and is
# named under `wording_rules.unaliased` in the summary, so an unmapped module is
# never silently handed a factor-level claim.
PROGRAM_WORDING_ALIASES = {
    "bhlh_ebox": "bHLH/E-box-associated program",
    "bhlh": "bHLH/E-box-associated program",
    "e_box": "bHLH/E-box-associated program",
}
MODULE_ROLE = "interpretation"
MODULE_INVARIANCE = (
    "emptying the module map (or pointing modules.file at a map with no rows) leaves every gate "
    "status, every similarity, every margin and every null unchanged: the module layer is read-only "
    "interpretation and never decides a primary PASS/FAIL"
)


@dataclass
class ModuleEntry:
    """One module of the panel, declared before any result row is read."""

    module: str
    level: str
    gating: bool
    provenance: str
    members: list[str]
    member_basis: str
    n_heterodimers_excluded: int = 0
    # components a config-declared aggregate names that the map resolves to no
    # module at all: they contribute no member, which is never silent
    unresolved_components: list[str] = field(default_factory=list)


def module_program_wording(module: str) -> tuple[str, bool]:
    """How a module's score must be worded, and whether that wording was declared.

    Returns (wording, declared). `declared` is False when the phrasings table has
    nothing for this module, which is reported rather than hidden so an unmapped
    module's generic wording can be reviewed instead of assumed.
    """
    key = str(module).strip().lower().replace("-", "_")
    for candidate in (key, key.split("_", 1)[0]):
        wording = PROGRAM_WORDING_ALIASES.get(candidate)
        if wording:
            return wording, True
    return f"{module}-associated program", False


def module_celltype_label(config: dict, celltype: str) -> tuple[str | None, str]:
    """Which spelling of the target cell type the frozen map uses.

    The map's `celltype` column is the authority, so a map that spells the target
    differently -- never mind a difference of case -- must still be read: silently
    finding no module for a cell type the map *does* describe is exactly the kind
    of silent drop this project forbids. The match is therefore returned to the
    caller, which audits a case-insensitive one as a substitution.
    """
    table = C.celltype_module_table(config)
    labels = list(dict.fromkeys(table.celltype.astype(str)))
    if celltype in labels:
        return celltype, "exact"
    for label in labels:
        if label.strip().lower() == str(celltype).strip().lower():
            return label, "case_insensitive"
    return None, "absent"


def module_panel_entries(config: dict, label: str) -> list[ModuleEntry]:
    """Every atomic module of one cell type, plus the config's derived unions.

    Membership comes from `common.module_members` with `monomers_only=True` (a
    module median must be built from binding models of one factor, so a
    heterodimer cannot contribute) and the derived unions from
    `common.module_aggregates`, whose members are the union of their component
    modules and therefore count no motif twice.
    """
    table = C.celltype_module_table(config)
    sub = table[table.celltype.astype(str) == label]
    head = sub.drop_duplicates("module").set_index("module")
    mapped = {str(value) for value in sub.module.unique()}
    entries: list[ModuleEntry] = []
    for module in C.modules_for(config, label):
        module = str(module)
        level = str(head["level"].get(module, "unassigned"))
        entries.append(
            ModuleEntry(
                module=module,
                level=level,
                gating=bool(C.module_is_gating(config, level)),
                provenance=str(head["provenance"].get(module, "")),
                members=[str(m) for m in C.module_members(config, label, module, monomers_only=True)],
                member_basis="monomer",
                n_heterodimers_excluded=int(len(sub[(sub.module == module) & (sub.member_type != "monomer")])),
            )
        )
    for name, components in C.module_aggregates(config, label).items():
        components = [str(component) for component in components]
        members: list[str] = []
        for component in components:
            members.extend(C.module_members(config, label, component, monomers_only=True))
        # a component the map has no rows for contributes no member: that is a
        # declaration error in the config, so it is carried out of here and audited
        unresolved = [component for component in components if component not in mapped]
        component_levels = [entry.level for entry in entries if entry.module in components]
        level = (
            component_levels[0]
            if component_levels and len(set(component_levels)) == 1
            else "aggregate"
        )
        entries.append(
            ModuleEntry(
                module=str(name),
                level=level,
                gating=bool(component_levels)
                and all(C.module_is_gating(config, value) for value in component_levels),
                provenance=(
                    "derived union declared in the config under modules.aggregates: "
                    + " + ".join(components)
                ),
                members=sorted({str(m) for m in members}),
                member_basis="derived_union",
                unresolved_components=unresolved,
            )
        )
    return entries


def module_named_members(config: dict, label: str, entry: ModuleEntry) -> list[str]:
    """The module's own members whose gene name its `provenance` highlights.

    A provenance that names a factor ("OLIG2 regulates oligodendrocyte lineage
    formation") invites reading the pooled module score as *that* factor's
    activity. Those names are exactly what the wording rule forbids, and they are
    derived from the frozen map rather than listed in this script.
    """
    table = C.celltype_module_table(config)
    sub = table[(table.celltype.astype(str) == label) & (table.module.astype(str) == entry.module)]
    names = sorted({str(value) for value in sub.motif_name.astype(str)})
    tokens = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", entry.provenance))
    return [name for name in names if name in tokens]


def module_family_shape(entries: list[ModuleEntry]) -> pd.DataFrame:
    """The panel as a TF-family-shaped annotation frame.

    `uncentered_family_scores` and `centered_family_scores` are the functions gate
    O3 is read through, so the module panel is pushed through the very same code by
    renaming `module` to `family`: the median, the cross-species centering and the
    finite-member handling are then literally the same operations gate O3 uses,
    not a parallel implementation that could drift away from "the same
    sign-agreement basis".
    """
    rows = [
        dict(motif_id=motif, family=entry.module, analysis_level=entry.level, member_type="monomer")
        for entry in entries
        for motif in entry.members
    ]
    if not rows:
        return pd.DataFrame(columns=["motif_id", "family", "analysis_level", "member_type"])
    return (
        pd.DataFrame(rows)
        .drop_duplicates(["motif_id", "family"])
        .reset_index(drop=True)
    )


def finite_member_counts(
    profile: pd.DataFrame, value: str, entries: list[ModuleEntry]
) -> pd.DataFrame:
    """Per (module, species): members the profile holds, and how many are finite.

    This is the module panel's audit of the NaN rule. A member the profile holds
    but whose value is non-finite is excluded from the module's median for that
    species, never read as a zero, and its count is carried out of here so the
    caller can name it in the audit table instead of letting the median silently
    shrink.
    """
    union = {member for entry in entries for member in entry.members}
    columns = ["family", "species", "n_members_present", "n_members_finite", "n_members_nonfinite"]
    if profile.empty or not union:
        return pd.DataFrame(columns=columns)
    work = profile[profile.motif_id.isin(union)]
    rows = []
    for entry in entries:
        members = set(entry.members)
        sub = work[work.motif_id.isin(members)]
        for species, group in sub.groupby("species", sort=False):
            values = pd.to_numeric(group[value], errors="coerce").to_numpy(dtype=float)
            finite = int(np.isfinite(values).sum())
            rows.append(
                dict(
                    family=entry.module, species=str(species), n_members_present=int(len(group)),
                    n_members_finite=finite, n_members_nonfinite=int(len(group)) - finite,
                )
            )
    return pd.DataFrame(rows, columns=columns)


def module_raw_support(
    entries: list[ModuleEntry],
    raw_profile: pd.DataFrame,
    model_long: pd.DataFrame,
    audit: Audit,
) -> pd.DataFrame:
    """Per module and species: the uncentered raw score against the model score.

    The panel is read on gate O3's own basis. `raw_module_score` is the uncentered
    median C over the module's monomeric members and `model_module_score` the
    uncentered median member log2_OR recomputed from the model profile, so a
    module that is uniformly oligodendrocyte-elevated in every species scores
    positive on both sides. `sign_agree` is their per-species sign agreement and
    `module_support` is True only for a module whose level `gating` accepts AND
    whose agreement reaches `GATE_MIN_SPECIES_AGREE`: a `secondary` or
    `comparator` module is reported with `gating=False` and `module_support=False`
    whatever its species agree on, because such a level may never take part in a
    PASS/FAIL rule. The cross-species-centered pair is the supplementary reading,
    exactly as it is for the family panel.
    """
    if not entries or raw_profile.empty or model_long.empty:
        return pd.DataFrame(columns=MODULE_COLUMNS)
    shape = module_family_shape(entries)
    if shape.empty:
        return pd.DataFrame(columns=MODULE_COLUMNS)

    raw_uncentered = uncentered_family_scores(raw_profile, "C", shape)
    model_uncentered = uncentered_family_scores(model_long, "log2_OR", shape)
    raw_centered = centered_family_scores(raw_profile, "C", shape)
    model_centered = centered_family_scores(model_long, "log2_OR", shape, "significant")
    raw_counts = finite_member_counts(raw_profile, "C", entries)
    model_counts = finite_member_counts(model_long, "log2_OR", entries)

    records: list[dict] = []
    for entry in entries:
        raw_rows = raw_uncentered[raw_uncentered.family == entry.module]
        if raw_rows.empty:
            audit.add(
                "-", "module_panel", entry.module,
                detail=(
                    f"level={entry.level}, gating={entry.gating}: none of its "
                    f"{len(entry.members)} monomeric members carries a finite raw C in any species, "
                    "so the module has no raw score; it is reported here and in the summary with no "
                    "rows rather than dropped"
                ),
                status="blocked",
            )
            continue
        merged = (
            raw_rows[["species", "family_score", "n_members"]]
            .rename(columns={"family_score": "raw_module_score", "n_members": "n_members_raw"})
            .merge(
                model_uncentered.loc[
                    model_uncentered.family == entry.module, ["species", "family_score", "n_members"]
                ].rename(columns={"family_score": "model_module_score", "n_members": "n_members_model"}),
                on="species", how="left",
            )
            .merge(
                raw_centered.loc[
                    raw_centered.family == entry.module, ["species", "C_family_score"]
                ].rename(columns={"C_family_score": "raw_module_score_centered"}),
                on="species", how="left",
            )
            .merge(
                model_centered.loc[
                    model_centered.family == entry.module, ["species", "log2_OR_family_score"]
                ].rename(columns={"log2_OR_family_score": "model_module_score_centered"}),
                on="species", how="left",
            )
        )
        merged = merged.merge(
            raw_counts.loc[raw_counts.family == entry.module].rename(
                columns={
                    "n_members_finite": "n_members_finite_raw",
                    "n_members_nonfinite": "n_members_nonfinite_raw",
                }
            ),
            on="species", how="left",
        ).merge(
            model_counts.loc[model_counts.family == entry.module].drop(
                columns=["n_members_present"]
            ).rename(
                columns={
                    "n_members_finite": "n_members_finite_model",
                    "n_members_nonfinite": "n_members_nonfinite_model",
                }
            ),
            on="species", how="left",
        )
        merged["sign_agree"] = _sign_agreement(merged.raw_module_score, merged.model_module_score)
        merged["sign_agree_centered"] = _sign_agreement(
            merged.raw_module_score_centered, merged.model_module_score_centered
        )
        n_agree = int(merged.sign_agree.sum())
        n_agree_centered = int(merged.sign_agree_centered.sum())
        n_scored = int((merged.raw_module_score.notna() & merged.model_module_score.notna()).sum())
        support = bool(entry.gating and n_agree >= GATE_MIN_SPECIES_AGREE)
        wording, _ = module_program_wording(entry.module)

        # the two counts are produced by different code paths (the median helper and
        # the accounting above), so a disagreement is reported rather than trusted
        if not merged.empty:
            differs = (
                merged.n_members_finite_raw.astype("Int64")
                != merged.n_members_raw.astype("Int64")
            )
            mismatched = merged[
                (merged.n_members_finite_raw.notna() & differs).fillna(False).astype(bool)
            ]
            if len(mismatched):
                audit.add(
                    "-", "module_panel", f"{entry.module}: member accounting",
                    detail=(
                        f"{len(mismatched)} species row(s) where the finite-member count and the median "
                        "helper disagree; the median is reported but the disagreement is not explained"
                    ),
                    status="blocked",
                )
        if merged.n_members_nonfinite_raw.fillna(0).sum() > 0:
            audit.add(
                "-", "module_members_nonfinite", entry.module,
                n_pseudobulk=int(merged.n_members_nonfinite_raw.fillna(0).sum()),
                detail=(
                    f"level={entry.level}: {int(merged.n_members_nonfinite_raw.fillna(0).sum())} "
                    "member-motif/species pair(s) carry a non-finite raw C and "
                    f"{int(merged.n_members_nonfinite_model.fillna(0).sum())} a non-finite model "
                    "log2_OR; they are excluded from the module median and counted in "
                    "n_members_nonfinite_raw / n_members_nonfinite_model, never read as a zero"
                ),
                status="excluded",
            )
        audit.add(
            "-", "module_panel", entry.module,
            n_pseudobulk=int(merged.n_members_raw.max()),
            detail=(
                f"level={entry.level}, gating={entry.gating}, members={entry.member_basis}: "
                f"uncentered raw module score agrees with the model in {n_agree}/{n_scored} species "
                f"({'support' if support else 'no support'}"
                + (
                    ""
                    if entry.gating
                    else f"; level `{entry.level}` is not a gating level "
                         "(common.module_is_gating is False), so this module can never satisfy a "
                         "PASS/FAIL rule and its agreement is reported only"
                )
                + f"); wording: \"{wording}\""
            ),
            status=("pass" if support else ("fail" if entry.gating else "recorded")),
        )
        records.extend(
            dict(
                module=entry.module,
                level=entry.level,
                gating=bool(entry.gating),
                provenance=entry.provenance,
                member_basis=entry.member_basis,
                species=str(row.species),
                raw_module_score=float(row.raw_module_score),
                model_module_score=(
                    float(row.model_module_score) if pd.notna(row.model_module_score) else np.nan
                ),
                sign_agree=bool(row.sign_agree),
                raw_module_score_centered=(
                    float(row.raw_module_score_centered)
                    if pd.notna(row.raw_module_score_centered) else np.nan
                ),
                model_module_score_centered=(
                    float(row.model_module_score_centered)
                    if pd.notna(row.model_module_score_centered) else np.nan
                ),
                sign_agree_centered=bool(row.sign_agree_centered),
                n_members_raw=int(row.n_members_raw),
                n_members_model=(
                    int(row.n_members_model) if pd.notna(row.n_members_model) else 0
                ),
                n_members_finite_raw=(
                    int(row.n_members_finite_raw) if pd.notna(row.n_members_finite_raw) else 0
                ),
                n_members_finite_model=(
                    int(row.n_members_finite_model) if pd.notna(row.n_members_finite_model) else 0
                ),
                n_members_nonfinite_raw=(
                    int(row.n_members_nonfinite_raw) if pd.notna(row.n_members_nonfinite_raw) else 0
                ),
                n_members_nonfinite_model=(
                    int(row.n_members_nonfinite_model)
                    if pd.notna(row.n_members_nonfinite_model) else 0
                ),
                n_species_agree=n_agree,
                n_species_agree_centered=n_agree_centered,
                n_species_scored=n_scored,
                module_support=support,
                program_wording=wording,
                model_score_source="recomputed_from_the_model_profile_readme_section_11",
            )
            for row in merged.itertuples()
        )
    return _sort_module_frame(pd.DataFrame(records, columns=MODULE_COLUMNS))


def _sort_module_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    order = {species: index for index, species in enumerate(C.SPECIES_ORDER)}
    return (
        frame.assign(
            _order=frame.species.map(order).fillna(len(order)),
        )
        .sort_values(
            ["module_support", "gating", "n_species_agree", "module", "_order"],
            ascending=[False, False, False, True, True],
        )
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def module_names(frame: pd.DataFrame, mask: "pd.Series | None" = None) -> list[str]:
    """The module names of a panel table, optionally restricted by a boolean mask.

    An empty panel has no rows to mask, so the mask is skipped rather than applied:
    a frame with no rows must never be conflated with a frame whose modules were
    filtered out, and an empty object-dtype mask would otherwise be read as a list
    of column labels instead of a boolean vector.
    """
    if frame.empty:
        return []
    if mask is not None:
        frame = frame.loc[mask]
    return sorted(set(frame["module"].astype(str)))


def module_panel_records(
    config: dict, label: str, entries: list[ModuleEntry], frame: pd.DataFrame
) -> list[dict]:
    """One summary record per module of the panel, evaluable or not.

    A module that produced no row at all is still named here, with the member count
    and the wording that applies, so "no module panel" and "this module had no
    finite members" can never be confused.
    """
    records = []
    for entry in entries:
        rows = frame[frame["module"] == entry.module]
        wording, declared = module_program_wording(entry.module)
        named = module_named_members(config, label, entry)
        records.append(
            dict(
                module=entry.module,
                level=entry.level,
                gating=bool(entry.gating),
                member_basis=entry.member_basis,
                n_members=len(entry.members),
                n_heterodimers_excluded=int(entry.n_heterodimers_excluded),
                program_wording=wording,
                program_wording_declared=bool(declared),
                named_members=[str(name) for name in named],
                forbidden_phrasings=[f"{name} activity" for name in named],
                raw_module_score_definition=(
                    "median C over the module's monomeric members, uncentered, finite members only"
                ),
                model_module_score_definition=(
                    "the median of the model's per-species member log2_OR, uncentered, finite "
                    "members only"
                ),
                n_species_agree=int(rows.n_species_agree.iloc[0]) if len(rows) else 0,
                n_species_agree_centered=int(rows.n_species_agree_centered.iloc[0]) if len(rows) else 0,
                n_species_scored=int(rows.n_species_scored.iloc[0]) if len(rows) else 0,
                n_members_nonfinite_raw=int(rows.n_members_nonfinite_raw.sum()) if len(rows) else 0,
                module_support=bool(len(rows) and rows.module_support.iloc[0]),
                species_scored=sorted(rows.species.astype(str)) if len(rows) else [],
                provenance=entry.provenance,
            )
        )
    return records


def module_panel_declaration(
    config: dict,
    label: str,
    match: str,
    entries: list[ModuleEntry],
    note: str,
    denominator: int,
) -> dict:
    """The panel's declaration and boundary, written before any result row is read.

    This is the payload's identity: what the map declares for the cell type, which
    level may gate, how a module score must be worded, and the fact that none of it
    can move a primary verdict.
    """
    path = C.module_map_path(config)
    gating_levels = sorted({entry.level for entry in entries if entry.gating})
    non_gating_levels = sorted(
        set(str(value) for value in ((config.get("modules") or {}).get("non_gating_levels") or []))
        | {entry.level for entry in entries if not entry.gating}
    )
    wording: dict[str, dict] = {}
    unaliased: list[str] = []
    for entry in entries:
        text, declared = module_program_wording(entry.module)
        named = module_named_members(config, label, entry) if label else []
        wording[entry.module] = dict(
            program_wording=text,
            declared=bool(declared),
            named_members=[str(name) for name in named],
            rule=(
                f'report `{entry.module}` as "{text}", never as the activity of a single member '
                "factor"
                + (f" (never {' / '.join(str(name) + ' activity' for name in named)})" if named else "")
            ),
        )
        if not declared:
            unaliased.append(entry.module)
    return {
        "role": MODULE_ROLE,
        "governs_primary_claim": False,
        "gate_role": (
            "interpretation only: the panel names what the unbiased 879-motif result found and never "
            "decides gate O3, the model-vs-raw similarity or the recurrent-family support; the primary "
            "evidence stays the 879-motif profile, the matched-random null and the chromVAR profile"
        ),
        "celltype": TARGET_CELLTYPE,
        "source": {
            "file": C.display_path(path),
            "sha256": C.sha256_file(path) if path.exists() else None,
            "celltype_label": label,
            "celltype_match": match,
        },
        "n_modules": len(entries),
        "n_gating_modules": sum(1 for entry in entries if entry.gating),
        "modules_with_support": [],
        "gating_levels": gating_levels,
        "non_gating_levels": non_gating_levels,
        "gate_min_species_agree": GATE_MIN_SPECIES_AGREE,
        "species_denominator": int(denominator),
        "member_basis": (
            "monomeric members only (a module median is built from binding models of one factor; "
            "heterodimers are excluded and counted per module as n_heterodimers_excluded)"
        ),
        "n_heterodimers_excluded": int(sum(entry.n_heterodimers_excluded for entry in entries)),
        "criterion": (
            f"a module is reported as supported only if its level is a gating level AND its uncentered "
            f"raw module score has the same sign as the uncentered model module score in >= "
            f"{GATE_MIN_SPECIES_AGREE} of {int(denominator)} species; modules at level "
            f"{non_gating_levels} are reported with gating=False and can never satisfy a PASS/FAIL rule"
        ),
        "sign_agreement_rule": (
            "sign(raw) * sign(model) > 0 per species on the UNCENTERED module medians; a score of "
            "exactly 0 or a missing score carries no direction and is never counted as agreement"
        ),
        "supplementary_centered_control": (
            "sign_agree_centered / n_species_agree_centered repeat the concordance after each motif's "
            "cross-species mean is removed from BOTH sides; that reading asks whether a module's "
            "cross-species variation tracks the model's and is reported next to the panel rather than "
            "deciding it, because a program that is uniformly oligodendrocyte-elevated in every species "
            "centers to exactly zero"
        ),
        "nonfinite_rule": (
            "a member whose raw C or model log2_OR is non-finite is excluded from that module's median "
            "for that species, counted in n_members_nonfinite_raw / n_members_nonfinite_model and named "
            "in oligo_raw_audit.tsv, and is never read as a zero"
        ),
        "invariance": MODULE_INVARIANCE,
        "aggregates": {
            str(name): [str(component) for component in components]
            for name, components in (
                C.module_aggregates(config, label).items() if label else []
            )
        },
        "wording_rules": {
            "claim_boundary": (
                "a module's score is a property of the pooled binding models of its members, not of any "
                "single factor: it is never evidence of the activity, abundance or binding of one member "
                "TF, and it is always reported by its program wording"
            ),
            "modules": wording,
            "unaliased": unaliased,
        },
        "modules": [],
        "evaluated": False,
        "note": note,
    }


def module_boundary_entries(
    config: dict, entries: list[ModuleEntry], note: str
) -> list[str]:
    """The boundary statements the module panel adds to `interpretation_boundary`.

    Derived from the frozen map, so a different panel produces a different, still
    truthful, paragraph -- never a sentence about a module this run does not have.
    """
    path = C.display_path(C.module_map_path(config))
    if not entries:
        return [
            "The cell-type-specific motif-module panel of "
            f"{path} is an INTERPRETATION layer: it names what the unbiased 879-motif analysis found "
            "and never decides gate O3. No panel is claimed for this run ("
            f"{note or 'the map declares no module for this cell type'}), and no gate status, "
            "similarity, margin or null changes because of that.",
        ]
    gating = [entry.module for entry in entries if entry.gating]
    non_gating = [entry.module for entry in entries if not entry.gating]
    gating_levels = sorted({entry.level for entry in entries if entry.gating})
    non_gating_levels = sorted({entry.level for entry in entries if not entry.gating})
    examples = []
    for wanted in (True, False):
        for entry in entries:
            text, declared = module_program_wording(entry.module)
            if declared is wanted:
                examples.append(f'`{entry.module}` -> "{text}"')
                break
    return [
        (
            f"The cell-type-specific motif-module panel ({', '.join(entry.module for entry in entries)}) "
            f"of {path} is the INTERPRETATION of the primary evidence above: it never decides gate O3, "
            "the model-vs-raw similarity or the recurrence defined upstream in s08, and " + MODULE_INVARIANCE + "."
        ),
        (
            "Only a gating-level module may carry a support verdict: "
            + (f"{', '.join(gating)} (level(s) {', '.join(gating_levels)}, "
               "common.module_is_gating True) may" if gating else "no module of this panel may")
            + ", while "
            + (f"{', '.join(non_gating)} (level(s) {', '.join(non_gating_levels)}) are reported and can "
               "never satisfy a PASS/FAIL rule" if non_gating else "this panel declares no non-gating module")
            + "."
        ),
        (
            "A module score is a property of the pooled binding models of its members, not of one "
            "factor, so it is reported as the module's associated program ("
            + "; ".join(examples)
            + ") and never as the activity of a member TF: the panel carries no "
            "`OLIG2 activity` and no `SOX10 activity` claim, and a module's provenance naming one factor "
            "is not evidence that factor drives the module."
        ),
    ]


def load_module_panel(
    config: dict, audit: Audit, denominator: int
) -> tuple[dict, list[ModuleEntry], str, str, str]:
    """Read the frozen module map, returning (declaration, entries, label, match, error).

    The map is an interpretation input, so a map that is missing, unreadable or
    empty must not take the step -- or a primary verdict -- down. It is read
    through `common`'s accessors, the failure is named in the audit and in the
    panel's `note`, and the caller then writes an empty panel table: a reader can
    always tell "no module panel" from "the file is absent".
    """
    path = C.module_map_path(config)
    label, match, error = "", "absent", ""
    entries: list[ModuleEntry] = []
    try:
        label, match = module_celltype_label(config, TARGET_CELLTYPE)
        if label is not None:
            entries = module_panel_entries(config, label)
        else:
            table = C.celltype_module_table(config)
            error = (
                f"the module map declares no rows for '{TARGET_CELLTYPE}' "
                f"(it declares {sorted({str(value) for value in table.celltype})})"
            )
    except Exception as exc:  # noqa: BLE001 - an absent map is not a primary failure
        error = f"the module map could not be read ({type(exc).__name__}: {exc})"
    if match == "case_insensitive":
        audit.add(
            "-", "module_map", TARGET_CELLTYPE,
            detail=f"the map spells the target cell type as `{label}`; matched case-insensitively",
            status="substituted",
        )
    if error:
        audit.add(
            "-", "module_map", C.display_path(path), detail=error + "; no primary verdict changes",
            status="blocked",
        )
    elif entries:
        audit.add(
            "-", "module_map", C.display_path(path),
            n_pseudobulk=len(entries),
            detail=(
                f"{len(entries)} module(s) read for `{label}`; gating level(s) "
                f"{sorted({entry.level for entry in entries if entry.gating})}, reported-only level(s) "
                f"{sorted({entry.level for entry in entries if not entry.gating})}; interpretation only"
            ),
            status="ok",
        )
    # a derived union whose declared components resolve to no module of the map is a
    # declaration error in the config: it contributes no member, so it is named here
    # rather than silently shrinking the union it was supposed to extend
    for entry in entries:
        for component in entry.unresolved_components:
            audit.add(
                "-", "module_aggregate", f"{entry.module}: {component}",
                detail=(
                    f"the config's modules.aggregates.{label}.{entry.module} names `{component}`, "
                    f"which is not a module the map declares for `{label}`; the derived union "
                    "counts no member from it and stays a union of the modules that do resolve"
                ),
                status="blocked",
            )
    note = error or (
        f"{len(entries)} module(s) read for `{label}` from the frozen cell-type-specific map; "
        "interpretation only"
    )
    declaration = module_panel_declaration(config, label, match, entries, note, denominator)
    return declaration, entries, label, match, error



def write_figure(
    out_dir: Path,
    similarity: pd.DataFrame,
    profiles: dict[str, pd.DataFrame],
    family_support: pd.DataFrame,
    formats: list[str],
    dpi: int,
    module_support: pd.DataFrame | None = None,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    species = [name for name in C.SPECIES_ORDER if name in profiles]
    ncols = 3
    nrows = max(1, int(np.ceil(len(species) / ncols)))
    has_family = not family_support.empty
    modules = (
        list(dict.fromkeys(module_support["module"].tolist()))
        if module_support is not None and not module_support.empty
        else []
    )
    height_ratios = [1.0] * nrows + ([1.3] if has_family else []) + ([1.3] if modules else [])
    fig = plt.figure(
        figsize=(4.3 * ncols, 3.3 * len(height_ratios) + 1.0), facecolor=PALETTE["surface"],
    )
    grid = fig.add_gridspec(len(height_ratios), ncols, height_ratios=height_ratios,
                            hspace=0.5, wspace=0.3)
    rho_by_species = similarity.set_index("species").rho.to_dict() if not similarity.empty else {}

    for index, name in enumerate(species):
        axes = fig.add_subplot(grid[index // ncols, index % ncols])
        profile = profiles[name]
        # a motif whose chromVAR Z was NaN has a NaN C: plot it nowhere rather than
        # let np.polyfit fail on it and take the whole figure down
        profile = profile[
            np.isfinite(profile.log2_OR.to_numpy(dtype=float))
            & np.isfinite(profile.C.to_numpy(dtype=float))
        ]
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
        axes.set_ylabel("raw chromVAR C\n(oligo - other celltypes)", fontsize=9,
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

    if has_family:
        axes = fig.add_subplot(grid[nrows, :])
        families = list(dict.fromkeys(family_support.family.tolist()))
        present = [name for name in C.SPECIES_ORDER if name in set(family_support.species)]
        positions = np.arange(len(present))
        width = 0.8 / max(len(families), 1)
        drawn: list[tuple[str, int]] = []
        for family_index, family in enumerate(families):
            subset = family_support[family_support.family == family].set_index("species")
            xs, ys, colors, model_ys = [], [], [], []
            for species_index, name in enumerate(present):
                if name not in subset.index:
                    continue
                row = subset.loc[name]
                xs.append(positions[species_index] + (family_index - (len(families) - 1) / 2) * width)
                ys.append(float(row.raw_family_score))
                colors.append(PALETTE["positive"] if row.raw_family_score > 0 else PALETTE["negative"])
                model_ys.append(float(row.model_family_score))
            if not xs:
                continue
            axes.bar(xs, ys, width=width * 0.88, color=colors, linewidth=0)
            axes.scatter(xs, model_ys, marker="D", s=22, color=PALETTE["ink"],
                         edgecolors=PALETTE["surface"], linewidths=0.8, zorder=3)
            drawn.append((family, int(subset.sign_agree.sum())))
        axes.axhline(0.0, lw=1.2, color=PALETTE["ink_muted"])
        axes.set_xticks(positions)
        axes.set_xticklabels(present, fontsize=9, color=PALETTE["ink_muted"])
        axes.set_ylabel("raw family score\n(median C, uncentered)", fontsize=9,
                        color=PALETTE["ink_muted"])
        axes.set_title("recurrent family raw support (gate O3)", fontsize=11, color=PALETTE["ink"])
        axes.tick_params(labelsize=9, colors=PALETTE["ink_muted"])
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axes.spines[side].set_color(PALETTE["grid"])
        axes.grid(True, axis="y", lw=0.6, color=PALETTE["grid"], alpha=0.6)
        axes.set_axisbelow(True)
        low, high = axes.get_ylim()
        span = (high - low) or 1.0
        # One label per family, stacked in the head-room band. `s07` anchors every
        # label at the same spot, so with several recurrent families the names
        # overprint each other; a per-family band costs nothing and stays legible.
        n_families = max(len(drawn), 1)
        top_pad = (0.11 * n_families + 0.06) * span
        axes.set_ylim(low - 0.05 * span, high + top_pad * 1.08)
        for family_index, (family, agree) in enumerate(drawn):
            axes.text(
                float(np.mean(positions)),
                high + top_pad * (1.0 - (family_index + 0.5) / n_families),
                f"{family} (agree {agree}/{len(present)})",
                ha="center", va="center", fontsize=9, color=PALETTE["ink"],
            )
        axes.legend(
            handles=[
                Patch(facecolor=PALETTE["positive"], label="raw C > 0"),
                Patch(facecolor=PALETTE["negative"], label="raw C < 0"),
                Line2D([], [], marker="D", color=PALETTE["ink"], linestyle="none",
                       markeredgecolor=PALETTE["surface"], label="model family score"),
            ],
            fontsize=8, frameon=False, loc="lower right", ncols=3,
        )

    # the module panel, drawn from the same statistic as the family row above it.
    # A non-gating module (level `secondary` / `comparator`) is drawn faded and is
    # named as such in the title, because it is reported and can never gate.
    if modules:
        axes = fig.add_subplot(grid[nrows + (1 if has_family else 0), :])
        present = [name for name in C.SPECIES_ORDER if name in set(module_support.species)]
        positions = np.arange(len(present))
        width = 0.8 / max(len(modules), 1)
        drawn = []
        for module_index, module in enumerate(modules):
            subset = module_support[module_support["module"] == module].set_index("species")
            gating = bool(subset["gating"].any())
            xs, ys, colors, model_ys = [], [], [], []
            for species_index, name in enumerate(present):
                if name not in subset.index:
                    continue
                row = subset.loc[name]
                xs.append(positions[species_index] + (module_index - (len(modules) - 1) / 2) * width)
                ys.append(float(row.raw_module_score))
                colors.append(PALETTE["positive"] if row.raw_module_score > 0 else PALETTE["negative"])
                model_ys.append(float(row.model_module_score))
            if not xs:
                continue
            axes.bar(xs, ys, width=width * 0.88, color=colors, linewidth=0,
                     alpha=1.0 if gating else 0.45)
            axes.scatter(xs, model_ys, marker="D", s=22, color=PALETTE["ink"],
                         edgecolors=PALETTE["surface"], linewidths=0.8, zorder=3)
            drawn.append((module, str(subset.level.iloc[0]), gating, int(subset.sign_agree.sum())))
        axes.axhline(0.0, lw=1.2, color=PALETTE["ink_muted"])
        axes.set_xticks(positions)
        axes.set_xticklabels(present, fontsize=9, color=PALETTE["ink_muted"])
        axes.set_ylabel("raw module score\n(median C, uncentered)", fontsize=9,
                        color=PALETTE["ink_muted"])
        axes.set_title(
            "motif-module raw support (interpretation; only a primary-level module may gate)",
            fontsize=11, color=PALETTE["ink"],
        )
        axes.tick_params(labelsize=9, colors=PALETTE["ink_muted"])
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axes.spines[side].set_color(PALETTE["grid"])
        axes.grid(True, axis="y", lw=0.6, color=PALETTE["grid"], alpha=0.6)
        axes.set_axisbelow(True)
        low, high = axes.get_ylim()
        span = (high - low) or 1.0
        n_modules = max(len(drawn), 1)
        top_pad = (0.11 * n_modules + 0.06) * span
        axes.set_ylim(low - 0.05 * span, high + top_pad * 1.08)
        for module_index, (module, level, gating, agree) in enumerate(drawn):
            axes.text(
                float(np.mean(positions)),
                high + top_pad * (1.0 - (module_index + 0.5) / n_modules),
                f"{module} [{level}"
                + ("" if gating else ", not a gating level")
                + f"] (agree {agree}/{len(present)})",
                ha="center", va="center", fontsize=9, color=PALETTE["ink"],
            )
        # the faded swatch is a faded bar, not a third hue: the panel encodes the
        # level by opacity AND by the per-module label above, so a reader never has
        # to tell a faded blue from a grey to see which module may gate
        axes.legend(
            handles=[
                Patch(facecolor=PALETTE["positive"], label="raw module score > 0"),
                Patch(facecolor=PALETTE["negative"], label="raw module score < 0"),
                Patch(facecolor=PALETTE["positive"], alpha=0.45,
                      label="non-gating level (faded)"),
                Line2D([], [], marker="D", color=PALETTE["ink"], linestyle="none",
                       markeredgecolor=PALETTE["surface"], label="model module score"),
            ],
            fontsize=8, frameon=False, loc="lower right", ncols=4,
        )

    fig.suptitle("Oligodendrocyte model motif profile vs raw chromVAR deviation (within species)",
                 fontsize=13, color=PALETTE["ink"])
    paths: list[Path] = []
    for fmt in formats:
        path = out_dir / f"{STEM}_model_vs_raw.{fmt}"
        fig.savefig(path, dpi=dpi, facecolor=PALETTE["surface"], bbox_inches="tight")
        C.log("wrote", path.relative_to(C.ROOT))
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


def gate_o3_untestable(species_expected: list[str]) -> bool:
    """Whether the readme's fixed agreement floor can be satisfied at this denominator.

    `GATE_MIN_SPECIES_AGREE` is the readme's absolute count of agreeing species and
    `species_expected` is the species set the config's `species_filter` leaves for
    this analysis. When the filter trims that set below the floor the criterion
    `n_species_agree >= GATE_MIN_SPECIES_AGREE` has no satisfying assignment: every
    family's `o3_support` is False no matter what the data says, so a run whose
    species agree 3/3 would still be published as a model FAIL. That shortfall is
    the config's, not the model's, so it is reported as PENDING (untestable) rather
    than FAIL, and the floor and the filtered count are named in `pending_reasons`
    and in an audit row.
    """
    return len(species_expected) < GATE_MIN_SPECIES_AGREE


def _base_summary(config: dict, topk: int, out_dir: Path, species_expected: list[str]) -> dict:
    """The summary skeleton, before any input is read.

    `species_expected` is the species set the config's species filter lets into the
    analysis (every species on the axis when there is no filter, i.e. under v1). It
    is the denominator gate O3 is read against, so it is named here rather than
    assumed: a filtered analysis that dropped two species would state the gate as
    ">= 4 of 4" and not as ">= 4 of 6".

    The floor is fixed and the denominator is derived, so the two can disagree in
    the other direction as well: a filter that leaves fewer than the floor's
    species makes the criterion unsatisfiable, which `gate_o3_untestable` detects.
    The criterion then says so explicitly instead of stating a count of agreeing
    species that cannot be reached.
    """
    min_ct = int(config["chromvar"]["min_cells_per_celltype"])
    n_species_expected = len(species_expected)
    o3_criterion = (
        f"at least one model-defined recurrent family whose uncentered raw family score has "
        f"the same sign as the uncentered model family score in >= {GATE_MIN_SPECIES_AGREE} "
        f"of {n_species_expected} species"
    )
    if gate_o3_untestable(species_expected):
        o3_criterion += (
            f" -- UNTESTABLE at this denominator: the readme criterion needs "
            f"{GATE_MIN_SPECIES_AGREE} agreeing species but the configured species set leaves "
            f"only {n_species_expected}, so gate O3 is reported PENDING, never FAIL"
        )
    return {
        "step": "s09_oligo_raw_validation",
        "analysis": ANALYSIS,
        "readme_section": "28 (Analysis C), gate O3",
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
            "n_species_expected": n_species_expected,
        },
        "raw_profile": {
            "definition": (
                "C_{s,m} = mean over oligo pseudobulks of Z[m, .] - equal-weighted mean over the "
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
                "average and recorded with their cell counts in oligo_raw_audit.tsv"
            ),
            "cross_species_comparison": (
                "forbidden and not performed: chromVAR Z is computed per species on that species' own "
                "peak universe and background"
            ),
        },
        "model_vs_raw": {},
        # the basis the gate is read on, named here so the verdict is self-describing: the
        # gate compares like with like, and "uncentered" is the operation BOTH sides get.
        # `s07` states its gate the same way, with the same three keys.
        "family_score_basis": {
            "gate_o3": "uncentered",
            "supplementary_reading": "cross_species_centered",
            "detail": (
                "Gate O3 compares like with like. The raw family score (median C over the family's "
                "monomeric members) and the model family score (the same median member log2_OR, "
                "recomputed from the model profile when recurrent_families.tsv carries no per-species "
                "score layout this step recognises) are BOTH read uncentered -- neither side has a "
                "cross-species baseline removed. The cross-species centered pair "
                "(raw_family_score_centered vs model_family_score_centered, sign_agree_centered / "
                "n_species_agree_centered) is a supplementary reading only and never decides the gate."
            ),
        },
        "gate_o3": {
            "criterion": o3_criterion,
            "evaluated": False,
            "pass": False,
            "families": [],
        },
        "gates": {"O3": "PENDING"},
        "audit": {"table": "oligo_raw_audit.tsv"},
        "interpretation_boundary": [
            "chromVAR Z is the per-motif deviation of a pseudobulk from the GC/background expectation "
            "of its own species' peak universe. It is not TF activity, not TF abundance, not TF "
            "binding, and not a measurement of any individual transcription factor.",
            "C is a contrast between oligodendrocyte pseudobulks and the equal-weighted average of the "
            "other cerebellar celltypes. It measures relative accessibility deviation, not absolute "
            "oligodendrocyte-specific accessibility.",
            "Every statistic here is computed within a single species. Raw C and raw Z are not "
            "comparable across species, because each species has its own peak universe, GC "
            "background, motif-peak matches and pseudobulk composition (readme section 26).",
            "The Spearman correlation between the model log2_OR profile and the raw C profile is "
            "descriptive: the motifs are not independent observations (they cluster into families and "
            "share peaks), so its p-values are not calibrated family-wise.",
            "Gate O3 is a direction-of-effect concordance test between a model-derived enrichment "
            "statistic and a raw accessibility-deviation statistic. Agreement is consistent with the "
            "model encoding a real oligodendrocyte regulatory program; it is not proof of TF-driven "
            "regulation and carries no causal claim.",
            "Gate O3 is decided on uncentered family scores on both sides: the raw side is the median "
            "C over the family's monomeric members and the model side is s08's per-species median "
            "member log2_OR (or the same median recomputed from the model profile). Neither side has "
            "a cross-species baseline removed, so a family that is uniformly oligodendrocyte-elevated "
            "in every species counts as agreeing with the model. That is the statistic readme section "
            "25/B4 defines recurrence on, and n_members counts the finite members the median was "
            "actually taken over.",
            "The cross-species centered family scores (raw_family_score_centered, "
            "model_family_score_centered) and their agreement (sign_agree_centered / "
            "n_species_agree_centered) are a supplementary reading, not the gate: each motif's "
            "cross-species mean is removed from both sides, so it asks whether a family's "
            "cross-species variation tracks the model's. It is deliberately not the gate, because a "
            "program that is equally oligodendrocyte-elevated in every species centers to exactly "
            "zero and would be scored as disagreeing.",
            "Org_Celltype == 'oligo' is one label that pools oligodendrocyte progenitors and mature "
            "oligodendrocytes, so any developmental or maturation signal inside that population is "
            "averaged away by this contrast (readme section 28).",
            "SOX / OLIG / MYRF / NKX motifs are not the endpoint of this analysis; the primary "
            "statistic is the full 879-motif profile and the recurrence definition is upstream in s08.",
        ],
        "outputs": {"directory": str(out_dir)},
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def _write_pending(out_dir: Path, audit: Audit, summary: dict, detail: list[str]) -> int:
    summary["pending_reasons"].append({"reason": "missing_input", "detail": detail})
    for item in detail:
        audit.add("-", "input_missing", "input", detail=item, status="blocked")
    C.save_table(audit.frame(), out_dir / f"{STEM}_raw_audit.tsv", sep="\t")
    # the interpretation panel is written even when the step is PENDING, so a reader
    # never has to tell "no module panel" apart from "the file is missing"
    C.save_table(pd.DataFrame(columns=MODULE_COLUMNS), out_dir / MODULE_TABLE, sep="\t")
    C.save_json(summary, out_dir / f"{STEM}_validation_summary.json")
    C.log("STATUS: PENDING", detail)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Oligo raw chromVAR validation (readme 28)")
    C.add_common_args(parser)
    args = parser.parse_args()
    C.main_guard()

    config = C.load_config(args.config)
    topk = int(args.topk or config["model_analysis"]["primary_topk"])
    out_dir = C.results_dir(config, STEP)
    audit = Audit()
    # The species axis this run may analyse is a property of the config's
    # `species_filter` block, so it is resolved before a single input is looked at:
    # a species the filter removes must not appear even as a missing-input entry,
    # and the gate denominator has to be the filtered count.
    species_expected, filter_record = analysis_species(config, audit)
    summary = _base_summary(config, topk, out_dir, species_expected)
    # The interpretation panel is read here, before any result row: it is not a
    # primary input, so a map that is missing, unreadable or empty is recorded and
    # the step goes on -- and because the panel never reaches a rule, its absence
    # cannot move gate O3, the similarity or the recurrence upstream in s08.
    module_panel_decl, module_entries, module_label, module_match, module_error = load_module_panel(
        config, audit, len(species_expected)
    )
    summary["module_panel"] = module_panel_decl
    summary["interpretation_boundary"].extend(
        module_boundary_entries(config, module_entries, module_error)
    )
    # A filter that leaves fewer species than the readme's absolute agreement floor
    # leaves gate O3 unsatisfiable: `n_species_agree >= 4` can never hold, so every
    # family would be written as `fail` and the gate as FAIL purely because of the
    # config. The run is still completed and its numbers are still written; only the
    # verdict is withheld -- the gate is reported PENDING (untestable) instead.
    o3_untestable = gate_o3_untestable(species_expected)
    if filter_record is not None:
        # a config without the block declares no exclusion, so it gets no key: that
        # is what keeps a v1 summary byte-identical to the published one
        summary["species_filter"] = filter_record

    with C.StepTimer("s09 oligo raw validation"):
        inputs = collect_inputs(config, species_expected)
        summary["inputs"] = {
            "all_model_motif_scores": str(inputs.model_scores),
            "recurrent_families": str(inputs.recurrent),
            "chromvar_z": {name: str(path) for name, path in sorted(inputs.chromvar.items())},
            "pseudobulk_metadata": {name: str(path) for name, path in sorted(inputs.metadata.items())},
            "sha256": {
                str(path): C.sha256_file(path)
                for path in [inputs.model_scores, inputs.recurrent,
                             *sorted(inputs.chromvar.values()), *sorted(inputs.metadata.values())]
                if path.exists()
            },
        }

        # a filter that removes every species is a configuration/input problem, not a
        # run with no ChromVAR input: it must not be reported as "no chromvar_z found"
        if not species_expected:
            return _write_pending(
                out_dir, audit, summary,
                [
                    f"the config's {filter_record['block']} rule excludes every species from the "
                    f"{ANALYSIS} analysis: no species has >= "
                    f"{filter_record['min_cells_per_species']} `{TARGET_CELLTYPE}` cells in "
                    f"{filter_record['celltype_column']} and at least "
                    f"{filter_record['min_ranked_peaks']} ranked peaks"
                ],
            )

        blocking: list[str] = []
        if not inputs.model_scores.exists():
            blocking.append(f"missing {inputs.model_scores}")
        if not inputs.chromvar:
            blocking.append("no results/04_chromvar/<species>/chromvar_z.tsv.gz found")
        if not inputs.metadata:
            blocking.append("no results/03_chromvar_input/<species>/pseudobulk_metadata.tsv found")
        if blocking:
            return _write_pending(out_dir, audit, summary, blocking)

        model_profiles = load_model_profiles(config, inputs.model_scores, topk, audit)
        # a model profile for a species the filter removed is not part of this
        # analysis; the `species_filter` audit row already named the species and the
        # cell count that removed it, so this is a narrowing, not a silent drop. Rows
        # for a species off the species axis are dropped for the same reason -- they
        # are not a species this analysis runs over.
        allowed = set(species_expected)
        model_profiles = {
            name: frame for name, frame in model_profiles.items() if name in allowed
        }
        if not model_profiles:
            return _write_pending(
                out_dir, audit, summary,
                [f"no {ANALYSIS} rows in {inputs.model_scores} for the species this run may analyse"],
            )
        summary["model_definition"] = {
            "source": str(inputs.model_scores),
            "primary_topk": topk,
            "species_with_a_model_profile": sorted(model_profiles, key=C.SPECIES_ORDER.index),
        }

        for species in species_expected:
            if species not in model_profiles:
                audit.add(species, "species_skipped", "no Oligo model profile", status="excluded")
            if species not in inputs.chromvar:
                audit.add(species, "species_skipped", "no chromvar_z.tsv.gz", status="excluded")
            elif species not in inputs.metadata:
                audit.add(species, "species_skipped", "no pseudobulk_metadata.tsv", status="excluded")

        profile_frames: list[pd.DataFrame] = []
        raw_stats: dict[str, dict] = {}
        similarity_rows: list[dict] = []
        model_long_frames: list[pd.DataFrame] = []

        for species in species_expected:
            if species not in model_profiles or species not in inputs.chromvar or species not in inputs.metadata:
                continue
            meta = load_pseudobulk_metadata(inputs.metadata[species], species, audit)
            z = load_chromvar_z(inputs.chromvar[species], set(meta.pseudobulk), species, audit)
            raw, stats = raw_oligo_profile(config, species, z, meta, audit)
            if raw is None:
                continue

            model = model_profiles[species]
            joined = raw.merge(model, on="motif_id", how="inner", suffixes=("_raw", ""))
            n_raw, n_model, n_common = len(raw), len(model), len(joined)
            if n_common < min(n_raw, n_model):
                audit.add(
                    species, "motif_coverage", "model/raw motif intersection",
                    n_pseudobulk=stats["n_oligo_pseudobulk"],
                    detail=f"model={n_model} raw={n_raw} common={n_common}",
                    status="ok" if n_common else "excluded",
                )
            if n_common < 3:
                audit.add(
                    species, "species_skipped", "insufficient motif overlap",
                    detail=f"common motifs = {n_common}", status="excluded",
                )
                continue
            joined = joined.sort_values("motif_id").reset_index(drop=True)
            joined["species"] = species
            # motif names live in the model table; the chromVAR matrix is indexed by id only
            joined["motif_name"] = (
                joined.motif_name.fillna("").astype(str) if "motif_name" in joined.columns
                else joined.motif_id.astype(str)
            )

            # chromVAR legitimately emits NaN for a motif with too few peak matches; a
            # single NaN would turn the whole rank correlation into NaN, so the
            # non-finite pairs are dropped, counted, and never presented as computed
            model_values = joined.log2_OR.to_numpy(dtype=float)
            raw_values = joined.C.to_numpy(dtype=float)
            simple_values = joined.c_simple_all.to_numpy(dtype=float)
            finite = np.isfinite(model_values) & np.isfinite(raw_values)
            n_finite = int(finite.sum())
            n_dropped = n_common - n_finite
            if n_dropped:
                audit.add(
                    species, "motif_coverage", "non-finite model/raw motif pairs",
                    n_pseudobulk=stats["n_oligo_pseudobulk"],
                    detail=(f"dropped {n_dropped} of {n_common} motif pairs with a non-finite "
                            f"model log2_OR or raw C"),
                    status="excluded",
                )
            if n_finite < 3:
                audit.add(
                    species, "species_skipped", "insufficient finite motif overlap",
                    detail=f"finite motif pairs = {n_finite} of {n_common}", status="excluded",
                )
                continue

            profile_frames.append(joined[PROFILE_COLUMNS].copy())
            raw_stats[species] = stats
            model_columns = ["motif_id", "log2_OR"]
            if "significant" in joined.columns:
                model_columns.append("significant")
            model_long_frames.append(
                joined.assign(species=species)[["species"] + model_columns].copy()
            )

            rho, p_value = C.spearman(model_values[finite], raw_values[finite])
            rho_simple, p_simple = float("nan"), float("nan")
            simple_finite = finite & np.isfinite(simple_values)
            if int(simple_finite.sum()) >= 3:
                rho_simple, p_simple = C.spearman(model_values[simple_finite], simple_values[simple_finite])
            similarity_rows.append(
                dict(
                    species=species, rho=rho, p=p_value, n_motifs=n_finite,
                    n_motifs_dropped_nonfinite=n_dropped,
                    n_pseudobulk_target=stats["n_oligo_pseudobulk"],
                    n_pseudobulk_other=stats["n_other_pseudobulk"],
                    target_cells=stats["n_oligo_cells"], other_cells=stats["n_other_cells"],
                    n_celltypes_other=stats["n_other_celltypes"],
                    rho_simple=rho_simple, p_simple=p_simple, p_fdr_bh=np.nan,
                    status="ok" if np.isfinite(rho) else "nan",
                )
            )

        if not similarity_rows:
            return _write_pending(
                out_dir, audit, summary,
                ["no species produced both a raw oligodendrocyte profile and a model profile"],
            )

        # the run's own species order: the species the config's filter let in, in the
        # canonical axis order. A species excluded by the filter is not a "skipped"
        # species, it is an excluded one, and it is already named by the filter record
        species_order = {name: index for index, name in enumerate(species_expected)}
        similarity = pd.DataFrame(similarity_rows, columns=SIMILARITY_COLUMNS)
        valid = similarity.p.notna()
        if valid.any():
            similarity.loc[valid, "p_fdr_bh"] = multipletests(
                similarity.loc[valid, "p"], method=str(config["statistics"]["fdr_method"])
            )[1]
        similarity = similarity.sort_values("species", key=lambda col: col.map(species_order)).reset_index(drop=True)

        profiles = pd.concat(profile_frames, ignore_index=True).sort_values(
            ["species", "motif_id"],
            key=lambda col: col.map(species_order) if col.name == "species" else col,
        ).reset_index(drop=True)

        summary["raw_profile"].update(
            {
                "n_motifs": int(profiles.motif_id.nunique()),
                "species_computed": sorted(raw_stats, key=species_order.get),
                "species_skipped": [name for name in species_expected if name not in raw_stats],
                "per_species": {name: raw_stats[name] for name in sorted(raw_stats, key=species_order.get)},
            }
        )
        summary["model_vs_raw"] = {
            "per_species": [
                {
                    "species": row.species, "rho": _num(row.rho), "p": _num(row.p),
                    "p_fdr_bh": _num(row.p_fdr_bh), "n_motifs": int(row.n_motifs),
                    "n_motifs_dropped_nonfinite": int(row.n_motifs_dropped_nonfinite),
                    "status": str(row.status),
                    "rho_simple_unweighted": _num(row.rho_simple),
                }
                for row in similarity.itertuples()
            ],
            "n_species_computed": int(similarity.rho.notna().sum()),
            "n_species_with_nan_rho": int(similarity.rho.isna().sum()),
            "n_motif_pairs_dropped_nonfinite": int(similarity.n_motifs_dropped_nonfinite.sum()),
            "nonfinite_rule": (
                "motif pairs whose model log2_OR or raw C is non-finite are dropped before the "
                "Spearman and counted in n_motifs_dropped_nonfinite; a species row stays `ok` only "
                "if its rho was actually computed"
            ),
            "n_species_rho_positive": int((similarity.rho > 0).sum()),
            "n_species_rho_positive_fdr05": int(((similarity.rho > 0) & (similarity.p_fdr_bh < 0.05)).sum()),
            "median_rho": _num(similarity.rho.median()),
            "min_rho": _num(similarity.rho.min()),
            "max_rho": _num(similarity.rho.max()),
            "heterogeneous_across_species": bool(
                len(similarity) and np.sign(similarity.rho.min()) != np.sign(similarity.rho.max())
            ),
            "role": (
                "descriptive supporting evidence; the formal oligodendrocyte raw gate (O3) is the "
                "family-sign concordance test below"
            ),
        }

        family_table = C.tf_family_table()
        model_long = pd.concat(model_long_frames, ignore_index=True)

        family_support = pd.DataFrame(columns=FAMILY_COLUMNS)
        model_source = ""
        if not inputs.recurrent.exists():
            summary["pending_reasons"].append(
                {
                    "reason": "recurrent_families_missing",
                    "detail": (
                        f"{inputs.recurrent} is absent; gate O3 is not evaluable because the recurrent "
                        "families are defined upstream by s08 (readme 28)"
                    ),
                }
            )
            audit.add("-", "gate_o3", "recurrent_families.tsv", detail="absent", status="blocked")
        else:
            recurrent = load_recurrent_families(inputs.recurrent, audit)
            provided = model_scores_from_recurrent(recurrent, species_expected)
            if provided is not None:
                model_source = "recurrent_families.tsv"
                family_support = apply_provided_model_scores(
                    recurrent_family_support(recurrent, profiles, model_long, family_table, model_source),
                    provided,
                )
            else:
                model_source = "recomputed_from_01_model_motif_readme_section_11"
                family_support = recurrent_family_support(
                    recurrent, profiles, model_long, family_table, model_source
                )
            if family_support.empty:
                audit.add(
                    "-", "gate_o3", "no evaluable recurrent family",
                    detail=f"{len(recurrent)} recurrent rows, none with raw family members",
                    status="blocked",
                )
            for family, group in family_support.groupby("family", sort=False):
                first = group.iloc[0]
                audit.add(
                    "-", "gate_o3", str(family),
                    detail=(f"uncentered raw family score agrees with the model in "
                            f"{int(first.n_species_agree)}/{int(first.n_species_scored)} species "
                            f"(centered control reading: {int(first.n_species_agree_centered)})"
                            + (
                                f"; NOT a verdict: the configured species set leaves "
                                f"{len(species_expected)} species, fewer than the "
                                f"{GATE_MIN_SPECIES_AGREE} the criterion needs"
                                if o3_untestable else ""
                            )),
                    n_pseudobulk=int(first.n_members_raw),
                    status=("untestable" if o3_untestable
                            else ("pass" if first.o3_support else "fail")),
                )

        # the agreement floor is absolute and the denominator is filtered, so the
        # criterion can be unsatisfiable; that is a config shortfall, not a model
        # FAIL, and it must not be reported as one (see `gate_o3_untestable`)
        if o3_untestable:
            floor = filter_record["min_cells_per_species"] if filter_record else None
            shortfall = (
                f"the configured species set leaves {len(species_expected)} species "
                f"(gate_min_species_agree={GATE_MIN_SPECIES_AGREE})"
                if floor is None else
                f"the config's species_filter (min_cells_per_species={floor}) leaves "
                f"{len(species_expected)} species, fewer than gate_min_species_agree="
                f"{GATE_MIN_SPECIES_AGREE}"
            )
            summary["pending_reasons"].append(
                {
                    "reason": "gate_o3_untestable",
                    "detail": (
                        f"gate O3 is not testable at this denominator: {shortfall}, so "
                        f"`n_species_agree >= {GATE_MIN_SPECIES_AGREE}` has no satisfying "
                        f"assignment and the gate is reported PENDING, never FAIL; the numbers "
                        f"are reported, the verdict is withheld"
                    ),
                }
            )
            audit.add(
                "-", "gate_o3", "untestable_denominator",
                detail=(
                    f"{shortfall}: `n_species_agree >= {GATE_MIN_SPECIES_AGREE}` cannot be "
                    f"satisfied, so gate O3 is PENDING, not FAIL"
                ),
                status="blocked",
            )

        o3_evaluated = bool(
            inputs.recurrent.exists() and not family_support.empty and not o3_untestable
        )
        o3_pass = bool(o3_evaluated and family_support.o3_support.any())
        best = None
        if not family_support.empty:
            top = family_support.sort_values(
                ["n_species_agree", "n_members_raw"], ascending=False
            ).iloc[0]
            best = {
                "family": str(top.family),
                "n_species_agree": int(top.n_species_agree),
                "n_species_scored": int(top.n_species_scored),
                "passes": bool(top.o3_support),
            }
        o3_criterion = (
            f"at least one model-defined recurrent family (recurrent==True in "
            f"results/08_oligo_model/recurrent_families.tsv) whose uncentered raw chromVAR "
            f"family score (the median C over the family's monomeric members) has the same sign "
            f"as the uncentered model family score in >= {GATE_MIN_SPECIES_AGREE} of "
            f"{len(species_expected)} species"
        )
        if o3_untestable:
            o3_criterion += (
                f" -- UNTESTABLE at this denominator: the readme criterion needs "
                f"{GATE_MIN_SPECIES_AGREE} agreeing species but the configured species set leaves "
                f"only {len(species_expected)}, so gate O3 is reported PENDING, never FAIL"
            )
        summary["gate_o3"] = {
            "criterion": o3_criterion,
            "evaluated": o3_evaluated,
            "pass": o3_pass,
            "recurrent_families_input": str(inputs.recurrent),
            "recurrent_families_read": int(0 if family_support.empty else family_support.family.nunique()),
            "model_family_score_source": model_source,
            "raw_family_score_definition": (
                "median over the monomeric member motifs of C, with no cross-species centering; the "
                "direct analogue of the model's per-species median member log2_OR; finite members only"
            ),
            "model_family_score_definition": (
                "the model's own per-species family score, the uncentered median member log2_OR "
                "recomputed from the model profile, because recurrent_families.tsv carries no "
                "per-species score layout this step recognises"
            ),
            "sign_agreement_rule": (
                "sign(raw) * sign(model) > 0 per species; a score of exactly 0 or a missing score "
                "carries no direction and is never counted as agreement"
            ),
            "supplementary_centered_control": (
                "sign_agree_centered / n_species_agree_centered repeat the concordance after each "
                "motif's cross-species mean is removed from BOTH sides (raw_family_score_centered vs "
                "model_family_score_centered); that reading asks whether the family's cross-species "
                "variation tracks the model's, and it is reported next to the gate rather than "
                "deciding it, because a program that is uniformly oligodendrocyte-elevated in every "
                "species centers to exactly zero"
            ),
            "best_family": best,
            "families": [
                {
                    "family": str(row.family), "analysis_level": str(row.analysis_level),
                    "n_species_agree": int(row.n_species_agree),
                    "n_species_agree_centered": int(row.n_species_agree_centered),
                    "n_species_scored": int(row.n_species_scored),
                    "n_members_raw": int(row.n_members_raw), "n_members_model": int(row.n_members_model),
                    "passes": bool(row.o3_support),
                }
                for row in family_support.drop_duplicates("family").itertuples()
            ],
        }
        if o3_untestable:
            # only written when the criterion is unsatisfiable, so a v1 (unfiltered)
            # run gains no key and stays byte-identical to the published summary
            summary["gate_o3"]["untestable"] = True
            summary["gate_o3"]["untestable_definition"] = (
                f"`n_species_agree >= {GATE_MIN_SPECIES_AGREE}` has no satisfying assignment at "
                f"the configured denominator of {len(species_expected)} species, so no family's "
                f"sign agreement can support the gate; the gate verdict is withheld (PENDING) "
                f"rather than published as a FAIL of the model"
            )
        summary["gates"]["O3"] = "PASS" if o3_pass else ("FAIL" if o3_evaluated else "PENDING")
        summary["status"] = "COMPLETE" if o3_evaluated else "PENDING"

        # the interpretation panel is computed AFTER gate O3 is settled and is written
        # into its own summary block: it reads the same two profiles the family panel
        # reads and feeds nothing back, so no verdict above can depend on it -- and an
        # error in it (a map whose motif_ids annotate none of the profile, say) must
        # not take the primary tables down with it, exactly like the figure below
        try:
            module_support = module_raw_support(module_entries, profiles, model_long, audit)
        except Exception as error:  # noqa: BLE001 - interpretation never gates a primary artefact
            audit.add(
                "-", "module_panel", "module_raw_support",
                detail=(
                    f"the module panel could not be computed ({type(error).__name__}: {error}); "
                    "the raw profile, the similarity and the recurrent-family support are unaffected, "
                    "and the panel table is written empty rather than omitted"
                ),
                status="failed",
            )
            summary["module_panel"]["error"] = f"{type(error).__name__}: {error}"
            module_support = pd.DataFrame(columns=MODULE_COLUMNS)
        has_rows = not module_support.empty
        support_mask = module_support["module_support"].astype(bool) if has_rows else None
        gating_mask = module_support["gating"].astype(bool) if has_rows else None
        modules_with_support = module_names(module_support, support_mask)
        summary["module_panel"]["modules"] = module_panel_records(
            config, module_label, module_entries, module_support
        )
        summary["module_panel"]["evaluated"] = bool(has_rows)
        summary["module_panel"]["n_modules_with_rows"] = (
            int(module_support["module"].nunique()) if has_rows else 0
        )
        summary["module_panel"]["modules_with_support"] = modules_with_support
        summary["module_panel"]["n_modules_with_support"] = len(modules_with_support)
        summary["module_panel"]["gating_modules_without_support"] = module_names(
            module_support, (gating_mask & ~support_mask) if has_rows else None
        )
        summary["module_panel"]["species_scored"] = sorted(
            set(module_support["species"].astype(str)) if has_rows else set(),
            key=lambda name: C.SPECIES_ORDER.index(name)
            if name in C.SPECIES_ORDER else len(C.SPECIES_ORDER),
        )
        summary["module_panel"]["models_scored"] = sorted(raw_stats)

        C.save_table(profiles, out_dir / f"{STEM}_raw_profile.tsv", sep="\t")
        C.save_table(similarity, out_dir / "model_vs_raw_similarity.tsv", sep="\t")
        C.save_table(family_support, out_dir / "recurrent_family_raw_support.tsv", sep="\t")
        C.save_table(module_support, out_dir / MODULE_TABLE, sep="\t")

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
            f"{STEM}_raw_profile.tsv", "model_vs_raw_similarity.tsv",
            "recurrent_family_raw_support.tsv", MODULE_TABLE, f"{STEM}_raw_audit.tsv",
        ]
        audit_summary = {
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
        }
        if filter_record is not None:
            # a species the filter removed never reached the analysis, so it cannot
            # appear as a "skipped" species: it is reported here with the cell count
            audit_summary["species_excluded_by_filter"] = [
                {"species": row["species"], "celltype": row["item"], "n_cells": row["n_cells"]}
                for row in audit.rows if row["kind"] == "species_filter"
            ]
        summary["audit"].update(audit_summary)
        C.save_table(audit.frame(), out_dir / f"{STEM}_raw_audit.tsv", sep="\t")
        C.save_json(summary, out_dir / f"{STEM}_validation_summary.json")

        C.log(
            "STATUS:", summary["status"], "| gate O3:", summary["gates"]["O3"],
            "| species:", len(similarity), "| median rho:", summary["model_vs_raw"]["median_rho"],
            "| module panel:", summary["module_panel"]["n_modules_with_support"], "of",
            summary["module_panel"]["n_gating_modules"], "gating module(s) with support",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
