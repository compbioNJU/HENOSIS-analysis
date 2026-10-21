"""Step 5 - Astroglia raw chromVAR validation (readme sections 19 and 20).

WHY this step exists
--------------------
s02 shows that the model's top-ranked Astroglia peaks carry a developmental motif
program, and that the program recovers stage correspondence across species.
That is a statement about the model, not about the tissue. This step asks the
independent question: does the *same* direction of change appear in the raw
chromVAR deviations of the same species' cis-regulatory landscape?

Only two objects cross the model/raw boundary:

    delta E_m = log2_OR(post, m) - log2_OR(pre, m)      (s01, Astroglia only)
    delta C_m = mean_Z(post, m)  - mean_Z(pre, m)       (s04, window means)

Everything else - the 879-motif catalog, the TF-family map, the centering rule,
the stage windows and the transitions - is imported unchanged, so the only new
information in the comparison is the raw signal itself. The stage test over the
cross-species windows is carried over as well, and its raw p-value is an exact
label permutation over every column ordering of the raw matrix. It is
deliberately NOT a matched-random-peak null: no random-peak background exists on
the chromVAR side, and the model-side matched-random null is reused here read-only,
never substituted for the raw test.

Period definition
-----------------
The windows, the transitions between them and the species roster all come from
the run's own config (common.cross_windows over the period table the config
declares, plus the species-filtered model manifest), so this one script serves
either period definition: v1 gives the cross-species windows (prenatal,
postnatal, adult_DN), v2 gives (prenatal, postnatal, adult) with human Carnegie
18-23 as its own single-species window. Nothing below assumes a particular
number of windows, transitions or species - the counts and the names in every
rule string, payload key and centering note are the ones the config actually
produced. What is frozen is the quorum of species a gate needs, the four gate
families and alpha (readme 10 and 31).

Raw aggregation
---------------
The raw profile that drives the gates is the astroglia-only realisation of the
cross-window mean. A pseudobulk is already a sample-level unit (sample x stage x
celltype), so each astroglia pseudobulk enters the mean weighted by its own cell
count; restricting the population to the target celltype is what readme 19
implies and what the parallel Purkinje step (s07) does to its own target side.
The every-pseudobulk realisation - GC, UBC, oligo, immune, Purkinje and astroglia
alike - is reported as the documented sensitivity and never drives a gate: in the
real pseudobulk metadata the GC population alone carries about 70% of the cell
weight, which would let granule-cell accessibility decide an Astroglia verdict.

A missing chromVAR Z entry is never read as an observed zero either: chromVAR
emits NaN in exactly the degenerate cases (zero-count pseudobulk, zero-variance
background for a low-match motif), so the cell weight of a NaN entry leaves the
denominator together with the term it would have contributed, and a motif with no
finite Z anywhere in a window stays NaN so that every consumer drops it. Each
species x window reports the affected entry count, the number of pseudobulks
carrying them and the number of pseudobulks that still contribute cell weight in
audit.tsv and in raw_window_summary.tsv. A pseudobulk whose n_cells is 0,
negative or non-finite is audited on its own row for the same reason: it leaves
the mean while remaining in the pseudobulk count.

Provenance travels with the verdict. The weighting scheme a window mean actually
used - real cell counts, or the unit-weight fallback when s03's
pseudobulk_metadata.tsv is absent or has no usable n_cells - is recorded per
species and scope, carried into gates A3/A4 as input_weighting / input_degraded,
and a gate that would otherwise PASS on a degraded aggregation is reported
PENDING instead. Every table that feeds or reports the gates carries a gate_basis
column naming the realisation behind it, and the summary json states the same
thing in its top-level gate_basis field.

Nothing here is tuned on results: the transitions are consecutive pairs of the
config's cross-species windows, the species roster is the manifest that clears
the config's species filter, and the family gate and alpha mirror readme 10 and
31.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from itertools import combinations
from math import factorial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

STEP = "05_astroglia_validation"
MODEL_STEP = "01_model_motif"
CHROMVAR_STEP = "04_chromvar"
PSEUDOBULK_STEP = "03_chromvar_input"

# The analysis this step validates. Not a period or a species: it is the model
# arm whose raw counterpart this file compares, and it is what the manifest is
# filtered on to obtain the species roster.
ASTROGLIA_ANALYSIS = "Astroglia"
ASTROGLIA_CELLTYPE = C.ANALYSIS_CELLTYPE[ASTROGLIA_ANALYSIS]
# readme 10 freezes exactly these four; asserted against the family map at run time.
GATE_FAMILIES = ("RFX", "NFI", "NF-Y", "SOX")
# readme 31 freezes the QUORUM a gate needs, not the roster it is counted
# over: gate A3/A4 ask for at least this many agreeing species and this many
# agreeing families. How many species and families there are to agree is read
# off the run's own manifest and family map at run time, never written down here.
GATE_SPECIES_QUORUM = 2
GATE_FAMILIES_QUORUM = 2
# The Astroglia model is being validated, so the gate-driving raw profile
# is the astroglia-only one; the every-pseudobulk profile is a sensitivity only.
# Both realisations keep the column names they have always had (the plain names
# are the every-pseudobulk numbers, the *_astroglia* names the target-population
# ones), so which of them gates is stated explicitly here, in the gate payloads
# and in the summary definitions rather than implied by a column name.
PRIMARY_CELLTYPE_SCOPE = ASTROGLIA_CELLTYPE
ALL_PSEUDOBULK_SCOPE = "all_pseudobulks"
# window_means() labels the every-pseudobulk realisation "all" (the celltype=None
# branch), while the gate payload calls the same realisation ALL_PSEUDOBULK_SCOPE.
ALL_WINDOW_SCOPE = "all"
PRIMARY_RHO_COLUMN = "rho_astroglia_only"
# Written into the top-level `gate_basis` field of the summary json and into a
# `gate_basis` column of every table that feeds or reports gates A3/A4, so the
# raw realisation a verdict rests on is stated next to the numbers themselves
# instead of having to be inferred from a column name.
GATE_BASIS = PRIMARY_CELLTYPE_SCOPE
# The weighting scheme a window mean actually used: real cell counts, or the
# degraded unit-weight fallback (no pseudobulk_metadata.tsv, or no usable
# n_cells column for some pseudobulk).
WEIGHTING_N_CELLS = "n_cells"
WEIGHTING_UNIT_FILLED = "unit_filled"

# Reference categorical palette (validated: adjacent-pair CVD dE 9.1 light,
# normal-vision dE 22.9; aqua/yellow sit below 3:1 on the light surface, so every
# family figure also ships direct labels and the underlying TSV as the table view).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
FAMILY_COLORS = {
    "RFX": SERIES[0],
    "NFI": SERIES[1],
    "NF-Y": SERIES[2],
    "SOX": SERIES[3],
}

# --------------------------------------------------------------------------
# the run's period axes, read from the config
# --------------------------------------------------------------------------


def consecutive_pairs(windows: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """The transitions of a window axis: every consecutive pair, in table order."""
    return tuple(zip(windows, windows[1:]))


def period_axes(config: dict) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """(windows, transitions) as this config's own period table defines them.

    The windows are the cross-species ones in table order (v1: prenatal,
    postnatal, adult_DN; v2: prenatal, postnatal, adult) and the transitions are
    their consecutive pairs. Nothing here implies a particular number of either:
    a table declaring four cross-species windows yields four windows and three
    transitions, and the gates, tables and figures below all follow.

    `single_species_windows` (the window a single species owns - human_only in
    v1, early_embryonic in v2) is deliberately NOT a window here: that period is
    that species' own axis and never enters a cross-species statistic.
    """
    windows = tuple(C.cross_windows(config))
    return windows, consecutive_pairs(windows)


def single_species_window_note(config: dict) -> str:
    """Name the periods this run keeps out of the cross-species centering mean.

    Those are the windows the period table declares for ONE species only (v1's
    human_only, v2's early_embryonic). Read off the table rather than written
    down, so the note always names the run's own single-species period and the
    stages inside it. A table that declares none says so in as many words.
    """
    table = C.stage_table(config)
    parts = []
    for window in C.single_species_windows(config):
        for row in table[table.cross_window == window].itertuples():
            stages = "|".join(
                stage.strip() for stage in str(row.raw_stages).split("|") if stage.strip()
            )
            parts.append(f"{row.species}:{window}" + (f" = {stages}" if stages else ""))
    return ", ".join(parts) if parts else "no single-species window"


def raw_margin_null_note(n_windows: int) -> str:
    """What the raw margin test is, with the actual column-count spelled out."""
    return (
        f"exact permutation over the {factorial(n_windows)} column-label orderings of the RAW "
        f"chromVAR {n_windows}x{n_windows} stage-similarity matrix "
        "(common.permutation_margin_pvalue); this is a label permutation test of the raw "
        "matrix, NOT a matched-random-peak null"
    )


def raw_margin_permutation_floor_note(n_windows: int) -> str:
    """The smallest attainable exact p, which a raw margin test can never beat."""
    n_orderings = factorial(n_windows)
    return (
        f"with {n_windows} windows there are {n_orderings} column-label orderings, so the "
        f"smallest exact p_perm_exact is 2/{n_orderings + 1} = "
        f"{2.0 / (n_orderings + 1):.4f}; a raw margin test can never reach 0.05"
    )


def species_palette(species_order: tuple[str, ...] | list[str]) -> dict[str, str]:
    """Colour a species by its position in this run's manifest order.

    No species name is baked in here; for the manifest species the first entries
    reproduce the reference palette the published figures use.
    """
    return {
        str(species): SERIES[index % len(SERIES)]
        for index, species in enumerate(species_order)
    }


# --------------------------------------------------------------------------
# bookkeeping
# --------------------------------------------------------------------------


@dataclass
class Audit:
    """Every skip, drop or substitution gets exactly one row here."""

    rows: list[dict] = field(default_factory=list)
    # The models the config's species filter removed, kept as structured records
    # (model, species, celltype, n_cells) beside the prose row each one already
    # gets. A PENDING exit writes no gate payload, so without this the run would
    # name the models it gave up nowhere on disk.
    dropped_models: list[dict] = field(default_factory=list)

    def add(self, scope: str, item: str, action: str, detail: str = "", species: str = "") -> None:
        self.rows.append(
            dict(scope=scope, species=species, item=item, action=action, detail=detail)
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.rows, columns=["scope", "species", "item", "action", "detail"]
        )


def ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Keep an empty result a real table with a header instead of a blank file."""
    if frame.empty and not set(columns) <= set(frame.columns):
        return pd.DataFrame(columns=columns)
    return frame


def save_table_with_comment(frame: pd.DataFrame, path: Path, comments: list[str]) -> Path:
    """save_table plus leading '#' lines, so a null test can state what it is."""
    C.save_table(frame, path, sep="\t")
    body = path.read_text()
    path.write_text("".join(f"# {line}\n" for line in comments) + body)
    return path


def tag_gate_basis(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """State inside a table which raw realisation gates A3/A4 rest on.

    `columns` (when given) is applied first so an empty result still gets the
    full header; the `gate_basis` column is then added to every row, including a
    table that carries both realisations - there the column names split them and
    `gate_basis` names the one the gates read.
    """
    if columns is not None:
        frame = ensure_columns(frame, columns)
    frame = frame.copy()
    frame["gate_basis"] = GATE_BASIS
    return frame


def gate_provenance(
    provenance: dict[tuple[str, str], dict], species_list: list[str], scope: str
) -> dict[str, dict]:
    """Aggregation provenance of one realisation, per species.

    A species with no recorded provenance is reported as degraded: an unknown
    weighting scheme must not be read as a cell-count-weighted one.
    """
    info = {}
    for species in species_list:
        record = provenance.get((species, scope), {})
        info[species] = dict(
            weighting=record.get("weighting", "unknown"),
            metadata_source=record.get("metadata_source", ""),
            degraded=bool(record.get("degraded", True)),
        )
    return info


def sign_of(value: float) -> int:
    if not np.isfinite(value) or value == 0:
        return 0
    return 1 if value > 0 else -1


def transition_label(pre: str, post: str) -> str:
    return f"{pre}->{post}"


def _style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.grid(color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_color(INK_2)


# Figures of this step that ship as PDF only: their raster copy is not part of
# the paper's deliverable set, so it is not written. Declared here, beside the
# call it affects, so the exclusion is visible in the code rather than implicit.
PDF_ONLY_STEMS = frozenset({
    "delta_scatter_model_vs_raw",
})


def _save_figure(fig, out_dir: Path, stem: str, config: dict) -> None:
    dpi = int(config.get("plotting", {}).get("dpi", 200))
    formats = list(config.get("plotting", {}).get("formats", ["png"]))
    if stem in PDF_ONLY_STEMS:
        formats = [fmt for fmt in formats if str(fmt).lower() != "png"]
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
        # display_path, not relative_to: a results tree configured outside the
        # project must not turn a written figure into an exception (v2 writes to
        # its own tree, and a smoke test may point results elsewhere entirely)
        C.log("wrote", C.display_path(path))
    plt.close(fig)


# --------------------------------------------------------------------------
# configuration-derived lookups
# --------------------------------------------------------------------------


def load_stage_lookups(config: dict, audit: Audit) -> tuple[dict, dict]:
    """Return (stage -> window, rank_period -> window), both keyed by species.

    The table is whichever period definition THIS config declares
    (common.stage_table(config)): v1's config/stage_map.tsv or v2's
    config/stage_map_v2.tsv. The window names on the right-hand side are the
    table's own, so every lookup below is in the run's vocabulary.
    """
    table = C.stage_table(config)
    stage_lookup: dict[tuple[str, str], str] = {}
    period_lookup: dict[tuple[str, str], str] = {}
    for row in table.itertuples():
        species = str(row.species)
        period_lookup[(species, str(row.rank_period))] = str(row.cross_window)
        for stage in str(row.raw_stages).split("|"):
            stage = stage.strip()
            if not stage:
                continue
            key = (species, stage)
            if key in stage_lookup and stage_lookup[key] != str(row.cross_window):
                audit.add(
                    "stage_map",
                    f"{species}:{stage}",
                    "conflicting_mapping",
                    f"{stage_lookup[key]} vs {row.cross_window}",
                    species,
                )
                continue
            stage_lookup[key] = str(row.cross_window)
    return stage_lookup, period_lookup


def astroglia_species(config: dict, audit: Audit) -> tuple[str, ...]:
    """The species the Astroglia gates may be evaluated on, from the manifest.

    A species is in the roster only if the model manifest actually carries an
    Astroglia model for it AND that model clears the config's species filter.
    v1 declares no `species_filter` block, so common.apply_species_filter is a
    no-op there and every manifest species enters; v2 drops a species whose
    target-celltype count is below `min_cells_per_species`. A dropped model is
    an audit row naming the model and the count that dropped it (and, on the same
    Audit, a structured entry a PENDING summary can still report), never a silent
    absence. The order is the project's canonical species order, so the roster -
    and every payload keyed by it - is deterministic.
    """
    manifest = C.ranking_files(config)
    astroglia = manifest[manifest.analysis.astype(str) == ASTROGLIA_ANALYSIS].copy()
    if astroglia.empty:
        audit.add(
            "model_manifest",
            ASTROGLIA_ANALYSIS,
            "no_astroglia_model",
            "the manifest carries no Astroglia ranking, so the Astroglia gates "
            "have no species roster",
        )
        return ()
    kept, dropped = C.apply_species_filter(config, astroglia)
    for row in dropped.itertuples():
        # structured as well as prose: a PENDING run (every Astroglia model
        # dropped) writes no gate payload, so the summary can only name what it
        # gave up if the record survives outside the prose
        audit.dropped_models.append(
            dict(
                model=str(row.model),
                species=str(row.species),
                celltype=str(row.celltype),
                n_cells=int(row.n_cells),
            )
        )
        audit.add(
            "model_manifest",
            f"{row.species}:{row.model}",
            "species_below_cell_floor",
            f"{row.n_cells} cells of {row.celltype} < min_cells_per_species="
            f"{row.min_cells}; model dropped, {row.species} does not enter the "
            "Astroglia comparison",
            str(row.species),
        )
    present = set(kept.species.astype(str))
    return tuple(species for species in C.SPECIES_ORDER if species in present)


def assert_gate_families(config: dict, audit: Audit) -> list[str]:
    configured = C.primary_families()
    if sorted(configured) != sorted(GATE_FAMILIES):
        audit.add(
            "tf_family_map",
            "primary",
            "gate_family_mismatch",
            f"config has {configured}, gate is fixed at {list(GATE_FAMILIES)}",
        )
    return list(GATE_FAMILIES)


FAMILY_LAYER_FILE = C.CONFIG_DIR / "tf_family_map.tsv"


def family_layer_contract() -> dict:
    """Which file governs the Astroglia TF-family layer this gate rests on.

    Gate A4 -- the direction-concordance gate that settles the claim
    `Conserved TF-family trajectory` -- is a family-level statistic, i.e. an
    INTERPRETATION layer over the unbiased 879-motif profile, exactly like the
    cell-type-specific motif modules. It is governed, however, by
    `config/tf_family_map.tsv` and NOT by `modules.file`
    (`config/celltype_motif_module_map.tsv`): no function in this step (or in s02)
    reads a module-map accessor. Emptying `modules.file` therefore cannot move A4,
    every primary number or the claim status; emptying *this* file would. The
    governing path, the families and the level split are recorded here and in
    `astroglia_validation_summary.json`, so that "emptying the module map leaves
    every claim unchanged" is never read as covering this separate input.
    """
    table = C.tf_family_table()
    families = sorted(str(v) for v in table.family.dropna().unique() if str(v))
    levels = sorted(str(v) for v in table.analysis_level.dropna().unique() if str(v))
    primary = sorted(
        str(v)
        for v in table.loc[table.analysis_level == "primary", "family"].dropna().unique()
        if str(v)
    )
    return dict(
        source=C.display_path(FAMILY_LAYER_FILE),
        role="interpretation",
        governs="Gate A4 (claim 'Conserved TF-family trajectory')",
        families=families,
        levels=levels,
        primary_families=primary,
        non_primary_families=[f for f in families if f not in set(primary)],
        gate_families=list(GATE_FAMILIES),
        gating_rule=(
            f"A4 iterates the {len(GATE_FAMILIES)} frozen primary families "
            f"({', '.join(GATE_FAMILIES)}) only; a family at any other level can never "
            "contribute to the gate, whatever its trajectory does"
        ),
        member_scope="monomer; a heterodimer never enters a family median",
        module_map_read=False,
        note=(
            f"the Astroglia family layer is read from {C.display_path(FAMILY_LAYER_FILE)} and "
            "not from modules.file; like the module layer it is interpretation, but it is a "
            "separate frozen input, so emptying modules.file leaves it (and Gate A4) untouched"
        ),
    )


def _family_layer_or_absent() -> dict:
    """`family_layer_contract()`, or a recorded absence when the map is unreadable.

    A PENDING exit must not become a crash: the family map is an interpretation
    input, and a summary that cannot name it says so instead of failing.
    """
    try:
        return family_layer_contract()
    except Exception as exc:  # noqa: BLE001 - an unreadable family map is recorded
        return dict(
            source=C.display_path(FAMILY_LAYER_FILE),
            role="interpretation",
            present=False,
            module_map_read=False,
            note=f"the family map could not be read ({type(exc).__name__}: {exc})",
        )


# --------------------------------------------------------------------------
# model side
# --------------------------------------------------------------------------


def load_model_windows(
    config: dict, period_lookup: dict, audit: Audit, windows: tuple[str, ...]
) -> pd.DataFrame:
    """Astroglia log2_OR restricted to the run's cross-species windows.

    Returns columns species, stage(=cross_window), model, motif_id, motif_name,
    log2_OR, log2_OR_centered. The centering rule is s02's (subtract the
    per-motif mean over the periods that enter the comparison) - see the
    centering scope in the summary for why the scope is the shared cross-species
    windows rather than the species' full axis (a single-species window such as
    v1's human_only or v2's early_embryonic is excluded from that mean).
    """
    path = C.config_path(config, "results") / MODEL_STEP / "all_model_motif_scores.tsv"
    frame = pd.read_csv(path, sep="\t")
    required = ("analysis", "species", "period", "model", "motif_id", "motif_name", "log2_OR")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise RuntimeError(f"{path} is missing columns {missing}")

    frame = frame[frame.analysis.astype(str) == ASTROGLIA_ANALYSIS].copy()
    frame["cross_window"] = [
        period_lookup.get((str(s), str(p)), "") for s, p in zip(frame.species, frame.period)
    ]
    unmapped = frame[frame.cross_window == ""]
    for (species, period), group in unmapped.groupby(["species", "period"]):
        audit.add(
            "model_profile",
            f"{species}:{period}",
            "period_not_in_stage_map",
            f"{len(group)} rows excluded",
            str(species),
        )
    frame = frame[frame.cross_window != ""]
    dropped = frame[~frame.cross_window.isin(windows)]
    for (species, window), group in dropped.groupby(["species", "cross_window"]):
        audit.add(
            "model_profile",
            f"{species}:{window}",
            "outside_cross_species_windows",
            f"{len(group)} rows excluded",
            str(species),
        )
    frame = frame[frame.cross_window.isin(windows)].copy()
    frame = frame.rename(columns={"cross_window": "stage"})
    duplicated = int(frame.duplicated(["species", "stage", "motif_id"]).sum())
    if duplicated:
        audit.add(
            "model_profile",
            "duplicate_species_stage_motif",
            "duplicate_rows_dropped",
            f"{duplicated} rows (first kept)",
        )
        frame = frame.drop_duplicates(["species", "stage", "motif_id"])

    baseline = frame.groupby(["species", "motif_id"])["log2_OR"].transform("mean")
    frame["log2_OR_centered"] = frame["log2_OR"] - baseline
    return frame[["species", "stage", "model", "motif_id", "motif_name", "log2_OR", "log2_OR_centered"]]


def audit_model_window_coverage(
    model_windows: pd.DataFrame,
    gate_species: tuple[str, ...],
    windows: tuple[str, ...],
    audit: Audit,
) -> None:
    """Name every cross-species window the MODEL input lacks for a roster species.

    load_model_windows only audits the (species, period) pairs the input file
    actually carries, so a window that is absent for one species yields no rows
    and therefore no audit row at all: the species still enters the comparison
    from the raw side, its model pivot then loses that window, and
    build_delta_table's `if pre not in pivot.columns` guard drops the whole
    transition with nothing written down. Comparing the windows each roster
    species actually has against the run's own window list makes that loss
    visible, and the matching per-species counts in gates.A3.per_species
    (n_transitions_evaluated, windows_evaluated) keep `evaluated: true` from
    hiding a truncated axis.
    """
    present: dict[str, set[str]] = {}
    if not model_windows.empty:
        for species, group in model_windows.groupby("species"):
            present[str(species)] = set(group.stage.astype(str))
    for species in gate_species:
        have = present.get(str(species), set()) & set(windows)
        for index, window in enumerate(windows):
            if window in have:
                continue
            lost = []
            if index > 0:
                lost.append(transition_label(windows[index - 1], window))
            if index + 1 < len(windows):
                lost.append(transition_label(window, windows[index + 1]))
            audit.add(
                "model_profile",
                f"{species}:{window}",
                "missing_cross_window",
                f"the model input carries no {window} rows for {species}; the "
                f"transition(s) {', '.join(lost) if lost else '(none)'} are dropped "
                "from the model side (build_delta_table skips a transition whose "
                f"window is absent from the model pivot), so this species is evaluated "
                f"over {len(have)} of {len(windows)} cross-species window(s)",
                str(species),
            )


# --------------------------------------------------------------------------
# raw chromVAR side
# --------------------------------------------------------------------------


def load_chromvar_matrix(path: Path, catalog: pd.DataFrame, audit: Audit, species: str) -> pd.DataFrame:
    """motif x pseudobulk chromVAR Z. Layout is whatever s04 wrote; be tolerant."""
    catalog_ids = set(catalog.motif_id)
    frame = pd.read_csv(path, sep="\t")
    if frame.empty:
        raise RuntimeError(f"{path} is empty")

    motif_col = None
    for candidate in (frame.columns[0], "motif_id", "motif", "motif_name"):
        if candidate in frame.columns:
            values = frame[candidate].astype(str)
            if len(set(values) & catalog_ids) >= 0.5 * len(set(values)):
                motif_col = candidate
                break
    if motif_col is None:
        audit.add("chromvar", species, "unrecognized_motif_column", str(path.name), species)
        raise RuntimeError(f"{path}: cannot locate the motif id column")

    data = frame.set_index(motif_col)
    keep = [col for col in data.columns if pd.to_numeric(data[col], errors="coerce").notna().any()]
    dropped = [col for col in data.columns if col not in keep]
    if dropped:
        audit.add(
            "chromvar",
            species,
            "non_numeric_columns_dropped",
            ",".join(map(str, dropped)),
            species,
        )
    data = data[keep].apply(pd.to_numeric, errors="coerce")
    data.index = data.index.astype(str)

    unknown = sorted(set(data.index) - catalog_ids)
    if unknown:
        audit.add("chromvar", species, "motifs_not_in_catalog", f"{len(unknown)} dropped", species)
    data = data.loc[[m for m in data.index if m in catalog_ids]]
    if data.index.duplicated().any():
        duplicated = int(data.index.duplicated().sum())
        audit.add("chromvar", species, "duplicate_motif_rows", f"{duplicated} dropped", species)
        data = data[~data.index.duplicated(keep="first")]
    missing = sorted(catalog_ids - set(data.index))
    if missing:
        audit.add(
            "chromvar",
            species,
            "motifs_absent_from_matrix",
            f"{len(missing)} of {len(catalog_ids)}",
            species,
        )
    if not data.columns.is_unique:
        duplicated = int(pd.Index(data.columns).duplicated().sum())
        audit.add("chromvar", species, "duplicate_pseudobulk_columns", f"{duplicated} dropped", species)
        data = data.loc[:, ~pd.Index(data.columns).duplicated(keep="first")]
    return data


def load_pseudobulk_metadata(
    config: dict, species: str, columns: list[str], stage_lookup: dict, audit: Audit
) -> tuple[pd.DataFrame, str]:
    """pseudobulk -> (stage, celltype, n_cells), reindexed onto the Z columns."""
    path = (
        C.config_path(config, "results")
        / PSEUDOBULK_STEP
        / species
        / "pseudobulk_metadata.tsv"
    )
    if path.exists():
        frame = pd.read_csv(path, sep="\t")
        if "pseudobulk" not in frame.columns:
            frame = frame.rename(columns={frame.columns[0]: "pseudobulk"})
        for name in ("stage", "celltype", "n_cells"):
            if name not in frame.columns:
                audit.add("pseudobulk_metadata", species, f"missing_column_{name}", str(path.name), species)
                frame[name] = "" if name != "n_cells" else np.nan
        frame = frame.drop_duplicates("pseudobulk").set_index("pseudobulk")
        missing = [col for col in columns if col not in frame.index]
        if missing:
            audit.add(
                "pseudobulk_metadata",
                species,
                "pseudobulk_without_metadata",
                f"{len(missing)} of {len(columns)} dropped from the raw aggregate",
                species,
            )
        frame = frame.reindex([col for col in columns if col in frame.index])
        frame["pseudobulk"] = frame.index
        return frame.reset_index(drop=True), "pseudobulk_metadata.tsv"

    audit.add(
        "pseudobulk_metadata",
        species,
        "metadata_file_missing",
        "stages guessed from the pseudobulk name; unit weights",
        species,
    )
    stages = sorted(
        {stage for (sp, stage) in stage_lookup if sp == species}, key=len, reverse=True
    )
    rows = []
    for name in columns:
        tokens = [tok for tok in re.split(r"\|+|__|::|\.", str(name)) if tok]
        stage = next((tok for tok in tokens if tok in stages), "")
        if not stage:
            audit.add("pseudobulk_metadata", species, "stage_not_parsable", str(name), species)
            continue
        rows.append(dict(pseudobulk=name, stage=stage, celltype="", n_cells=np.nan))
    return pd.DataFrame(rows, columns=["pseudobulk", "stage", "celltype", "n_cells"]), "pseudobulk_name_guess"


def window_means(
    config: dict,
    species: str,
    z: pd.DataFrame,
    stage_lookup: dict,
    audit: Audit,
    windows: tuple[str, ...],
    celltype: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """n_cells-weighted mean of Z per cross_window (optionally one celltype only)."""
    columns = [str(col) for col in z.columns]
    meta, source = load_pseudobulk_metadata(config, species, columns, stage_lookup, audit)
    meta["pseudobulk"] = meta.pseudobulk.astype(str)
    meta = meta[meta.pseudobulk.isin(columns)].copy()

    if celltype is not None:
        meta = meta[meta.celltype.astype(str).str.lower() == celltype]
        if meta.empty:
            audit.add(
                "raw_windows",
                species,
                f"no_{celltype}_pseudobulk",
                f"{celltype}-only sensitivity not computed",
                species,
            )
            return pd.DataFrame(), pd.DataFrame()

    meta["n_cells"] = pd.to_numeric(meta.n_cells, errors="coerce")
    n_unit_weight = int(meta.n_cells.isna().sum())
    if n_unit_weight:
        audit.add(
            "raw_windows",
            species,
            "n_cells_missing",
            f"{n_unit_weight} pseudobulk given unit weight",
            species,
        )
        meta["n_cells"] = meta.n_cells.fillna(1.0)
    # The weighting scheme this species actually used. A gate payload carries it,
    # so a verdict can never be read as cell-count-weighted when the metadata
    # file was absent or its n_cells column was unusable and every pseudobulk
    # silently entered at weight 1.
    weighting = (
        WEIGHTING_N_CELLS
        if source == "pseudobulk_metadata.tsv" and not n_unit_weight
        else WEIGHTING_UNIT_FILLED
    )

    meta["cross_window"] = [stage_lookup.get((species, str(s)), "") for s in meta.stage]
    unmapped = meta[meta.cross_window == ""]
    for stage, group in unmapped.groupby("stage"):
        audit.add(
            "raw_windows",
            species,
            "stage_not_in_stage_map",
            f"{len(group)} pseudobulk ({stage}) dropped",
            species,
        )
    meta = meta[meta.cross_window != ""]
    outside = meta[~meta.cross_window.isin(windows)]
    for window, group in outside.groupby("cross_window"):
        audit.add(
            "raw_windows",
            species,
            "outside_cross_species_windows",
            f"{len(group)} pseudobulk ({window}) dropped",
            species,
        )
    meta = meta[meta.cross_window.isin(windows)].copy()

    profiles: dict[str, pd.Series] = {}
    summary_rows = []
    for window in windows:
        block = meta[meta.cross_window == window]
        if block.empty:
            audit.add("raw_windows", species, f"empty_window_{window}", "window not estimable", species)
            continue
        scope = celltype or "all"
        weight_value = block.n_cells.to_numpy(dtype=float)
        matrix = z[[col for col in block.pseudobulk]].to_numpy(dtype=float)
        usable = np.isfinite(weight_value) & (weight_value > 0)
        weights = np.where(usable, weight_value, 0.0)
        # A pseudobulk whose n_cells is 0, negative or non-finite carries no cell
        # weight and leaves the aggregate. It is still counted in n_pseudobulk, so
        # each one is audited on its own and the window reports how many
        # pseudobulks actually contributed next to how many were present.
        for name, value in zip(block.pseudobulk, weight_value):
            if not (np.isfinite(value) and value > 0):
                audit.add(
                    "raw_windows",
                    species,
                    f"{window}:{scope}:zero_weight_pseudobulk",
                    f"pseudobulk {name} has n_cells={float(value)}; weight forced to 0, so it "
                    f"is counted in n_pseudobulk but not in contributing_pseudobulk",
                    species,
                )
        if weights.sum() <= 0:
            audit.add("raw_windows", species, f"zero_weight_{window}", "window not estimable", species)
            continue
        # A missing chromVAR Z is NOT an observed zero. chromVAR emits NaN exactly
        # in the degenerate cases (a zero-count pseudobulk, a zero-variance
        # background for a low-match motif), so the cell weight it would have
        # carried is removed from the denominator together with the term it would
        # have contributed: every window mean averages over the finite entries
        # only. A motif with no finite Z anywhere in the window has no mean and
        # stays NaN, so every consumer drops it instead of seeing a tied zero.
        finite = np.isfinite(matrix)
        denominator = np.where(finite, weights, 0.0).sum(axis=1)
        numerator = (np.where(finite, matrix, 0.0) * weights).sum(axis=1)
        values = np.full(len(z.index), np.nan)
        np.divide(numerator, denominator, out=values, where=denominator > 0)
        profiles[window] = pd.Series(values, index=z.index, name=window)

        scope = celltype or "all"
        n_nonfinite = int((~finite).sum())
        n_unestimable = int((~np.isfinite(values)).sum())
        # how many pseudobulks carry at least one dropped Z entry, and how many
        # still contribute a cell weight to this window's mean
        n_pseudobulk_nonfinite = int((~finite).any(axis=0).sum())
        n_contributing = int((weights > 0).sum())
        # one row per species x window x scope: a missing Z is never invisible
        audit.add(
            "raw_windows",
            species,
            f"{window}:{scope}",
            f"nonfinite_z={n_nonfinite}/{matrix.size} entries in "
            f"{n_pseudobulk_nonfinite} pseudobulk(s); weights renormalised over the "
            f"finite entries only; {n_unestimable} motif(s) have no finite Z in this "
            f"window and stay NaN; {n_contributing} of {len(block)} pseudobulk(s) "
            f"contribute cell weight (weighting={weighting}); celltype_scope={scope}",
            species,
        )
        summary_rows.append(
            dict(
                species=species,
                window=window,
                celltype_scope=scope,
                n_pseudobulk=len(block),
                contributing_pseudobulk=n_contributing,
                n_cells=float(weights.sum()),
                n_motifs=int(np.isfinite(values).sum()),
                n_nonfinite_z=n_nonfinite,
                n_pseudobulk_with_nonfinite_z=n_pseudobulk_nonfinite,
                n_motifs_unestimable=n_unestimable,
                metadata_source=source,
                weighting=weighting,
            )
        )
    profile = pd.DataFrame(profiles)
    return profile, pd.DataFrame(summary_rows)


def has_all_windows(profile: pd.DataFrame, windows: tuple[str, ...]) -> bool:
    """True when this profile carries every window the run compares over."""
    return all(window in profile.columns for window in windows)


# --------------------------------------------------------------------------
# deltas and correlations
# --------------------------------------------------------------------------


def build_delta_table(
    model_windows: pd.DataFrame,
    raw_by_species: dict,
    species_list: list[str],
    windows: tuple[str, ...],
) -> pd.DataFrame:
    """One row per species x transition x motif, model delta next to raw delta.

    The transitions are the consecutive pairs of `windows`, so a run with two or
    five windows produces the matching number of rows per species.
    """
    rows: list[pd.DataFrame] = []
    for species in species_list:
        model = model_windows[model_windows.species == species]
        raw = raw_by_species.get(species)
        if raw is None or not has_all_windows(raw, windows):
            continue
        pivot = model.pivot_table(
            index="motif_id", columns="stage", values="log2_OR_centered", aggfunc="mean"
        )
        names = model.drop_duplicates("motif_id").set_index("motif_id").motif_name
        for pre, post in consecutive_pairs(windows):
            if pre not in pivot.columns or post not in pivot.columns:
                continue
            if pre not in raw.columns or post not in raw.columns:
                continue
            joined = pd.DataFrame(
                {
                    "motif_id": pivot.index,
                    "model_pre": pivot[pre].to_numpy(),
                    "model_post": pivot[post].to_numpy(),
                    "raw_pre": raw[pre].reindex(pivot.index).to_numpy(),
                    "raw_post": raw[post].reindex(pivot.index).to_numpy(),
                }
            )
            joined["motif_name"] = joined.motif_id.map(names).fillna("")
            joined["delta_model"] = joined.model_post - joined.model_pre
            joined["delta_chromvar"] = joined.raw_post - joined.raw_pre
            joined.insert(0, "species", species)
            joined.insert(1, "transition", transition_label(pre, post))
            joined.insert(2, "pre_window", pre)
            joined.insert(3, "post_window", post)
            rows.append(joined)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def correlate(table: pd.DataFrame, value_model: str, value_raw: str) -> dict:
    valid = table[[value_model, value_raw]].dropna()
    n = len(valid)
    if n < 3:
        return dict(n_motifs=n, rho=np.nan, pvalue=np.nan, n_concordant=0, n_discordant=0,
                    n_tied=0, concordance=np.nan, binom_pvalue=np.nan)
    rho, pvalue = C.spearman(valid[value_model].to_numpy(), valid[value_raw].to_numpy())
    signs_model = np.sign(valid[value_model].to_numpy())
    signs_raw = np.sign(valid[value_raw].to_numpy())
    concordant = int(np.sum((signs_model == signs_raw) & (signs_model != 0)))
    discordant = int(np.sum((signs_model == -signs_raw) & (signs_model != 0)))
    tied = int(n - concordant - discordant)
    if concordant + discordant > 0:
        binom_p = float(
            binomtest(concordant, concordant + discordant, 0.5, alternative="greater").pvalue
        )
    else:
        binom_p = np.nan
    return dict(
        n_motifs=n,
        rho=rho,
        pvalue=pvalue,
        n_concordant=concordant,
        n_discordant=discordant,
        n_tied=tied,
        concordance=(concordant / (concordant + discordant)) if (concordant + discordant) else np.nan,
        binom_pvalue=binom_p,
    )


def attach_families(table: pd.DataFrame, family_table: pd.DataFrame) -> pd.DataFrame:
    annotations = family_table[["motif_id", "family", "analysis_level"]].copy()
    annotations["family"] = annotations.family.fillna("")
    annotations["analysis_level"] = annotations.analysis_level.fillna("")
    out = table.merge(annotations.drop_duplicates("motif_id"), on="motif_id", how="left")
    out["family"] = out.family.fillna("")
    out["analysis_level"] = out.analysis_level.fillna("")
    return out


# --------------------------------------------------------------------------
# family trajectory, model vs raw
# --------------------------------------------------------------------------


def family_table_model_vs_raw(
    model_windows: pd.DataFrame,
    raw_profiles: dict[str, pd.DataFrame],
    family_table: pd.DataFrame,
    species_list: list[str],
    windows: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same family map, same centering rule, applied to both sides."""
    model_rows = []
    raw_rows = []
    for species in species_list:
        model = model_windows[model_windows.species == species][
            ["species", "stage", "motif_id", "log2_OR"]
        ]
        raw = raw_profiles.get(species)
        if raw is None or not has_all_windows(raw, windows):
            continue
        model_rows.append(model)
        long = raw.reindex(columns=list(windows)).reset_index()
        long.columns = ["motif_id", *list(windows)]
        long = long.melt(
            id_vars="motif_id", value_vars=list(windows), var_name="stage", value_name="raw_z"
        ).dropna()
        long.insert(0, "species", species)
        raw_rows.append(long)

    if not model_rows or not raw_rows:
        return pd.DataFrame(), pd.DataFrame()

    model_family = C.family_trajectory(
        pd.concat(model_rows, ignore_index=True), family_table, value="log2_OR"
    )
    raw_family = C.family_trajectory(
        pd.concat(raw_rows, ignore_index=True), family_table, value="raw_z"
    )
    model_family = model_family.rename(
        columns={"family_score": "model_family_score", "n_members": "n_members_model"}
    )
    raw_family = raw_family.rename(
        columns={"family_score": "raw_family_score", "n_members": "n_members_raw"}
    )
    merged = model_family.merge(
        raw_family[["species", "stage", "family", "raw_family_score", "n_members_raw"]],
        on=["species", "stage", "family"],
        how="outer",
    )
    merged = merged.sort_values(["family", "species", "stage"]).reset_index(drop=True)
    merged["delta_family"] = merged.model_family_score - merged.raw_family_score
    return merged, raw_family


def family_correlations(merged: pd.DataFrame, family_table: pd.DataFrame) -> pd.DataFrame:
    """Spearman of model vs raw family score, pooled over species x stage, and per species."""
    levels = (
        family_table.drop_duplicates("family").set_index("family").analysis_level
    )
    rows = []
    for family, block in merged.groupby("family"):
        for scope, sub in [("species_x_stage", block)] + [
            (str(species), group) for species, group in block.groupby("species")
        ]:
            valid = sub[["model_family_score", "raw_family_score"]].dropna()
            n = len(valid)
            if n < 3:
                rho, pvalue = np.nan, np.nan
            else:
                rho, pvalue = C.spearman(
                    valid.model_family_score.to_numpy(), valid.raw_family_score.to_numpy()
                )
            rows.append(
                dict(
                    family=family,
                    analysis_level=levels.get(family, ""),
                    scope=scope,
                    n_points=n,
                    rho=rho,
                    pvalue=pvalue,
                    p_is_reliable=bool(n >= 5),
                )
            )
    return pd.DataFrame(rows).sort_values(["family", "scope"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# raw stage similarity over the cross-species windows and its label-permutation null
# --------------------------------------------------------------------------


def cross_similarity(
    rows_profile: pd.DataFrame, cols_profile: pd.DataFrame, windows: tuple[str, ...]
) -> tuple[np.ndarray, list[dict]]:
    """n x n Spearman of rows_profile[window] against cols_profile[window]."""
    motifs = [m for m in rows_profile.index if m in set(cols_profile.index)]
    matrix = np.full((len(windows), len(windows)), np.nan)
    long_rows = []
    for i, window_a in enumerate(windows):
        for j, window_b in enumerate(windows):
            a = rows_profile.loc[motifs, window_a].to_numpy(dtype=float)
            b = cols_profile.loc[motifs, window_b].to_numpy(dtype=float)
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() < 3:
                continue
            rho, pvalue = C.spearman(a[valid], b[valid])
            matrix[i, j] = rho
            long_rows.append(
                dict(window_a=window_a, window_b=window_b, rho=rho, pvalue=pvalue, n_motifs=int(valid.sum()))
            )
    return matrix, long_rows


def top1_accuracy(matrix: np.ndarray) -> float:
    hits = 0
    total = 0
    for i in range(matrix.shape[0]):
        row = matrix[i]
        if not np.isfinite(row).any():
            continue
        total += 1
        if int(np.nanargmax(row)) == i:
            hits += 1
    return hits / total if total else np.nan


def stage_similarity_raw(
    profiles: dict[str, pd.DataFrame], species_list: list[str], windows: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows = []
    margin_rows = []
    for species_a, species_b in combinations(species_list, 2):
        rows_profile = profiles[species_a]
        cols_profile = profiles[species_b]
        matrix, cells = cross_similarity(rows_profile, cols_profile, windows)
        for cell in cells:
            cell.update(species_a=species_a, species_b=species_b)
            long_rows.append(cell)
        if not np.isfinite(matrix).all():
            continue
        mean_match, mean_mismatch, margin = C.matching_margin(matrix, list(windows))
        observed, p_perm, p_min, n_perm = C.permutation_margin_pvalue(matrix, list(windows))
        margin_rows.append(
            dict(
                species_a=species_a,
                species_b=species_b,
                mean_matching=mean_match,
                mean_mismatching=mean_mismatch,
                margin=margin,
                p_perm=p_perm,
                p_perm_min=p_min,
                n_permutations=n_perm,
                top1_accuracy=top1_accuracy(matrix),
                top1_direction=f"{species_a}_rows_over_{species_b}_cols",
                **{
                    f"matrix_{len(windows)}x{len(windows)}": ";".join(
                        ",".join(f"{value:.6f}" for value in matrix[i])
                        for i in range(len(windows))
                    )
                },
            )
        )
    return pd.DataFrame(long_rows), pd.DataFrame(margin_rows)


def model_null_margin_reference(
    config: dict,
    model_windows: pd.DataFrame,
    null_profiles: dict,
    audit: Audit,
    windows: tuple[str, ...],
) -> pd.DataFrame:
    """Model-side matched-random margins, read from s01's null (reference only)."""
    rows = []
    for species_a, species_b in combinations(sorted(model_windows.species.unique()), 2):
        if species_a not in null_profiles or species_b not in null_profiles:
            continue
        repeats = sorted(
            set(null_profiles[species_a]) & set(null_profiles[species_b])
        )
        margins = []
        for repeat in repeats:
            rows_profile = null_profiles[species_a][repeat]
            cols_profile = null_profiles[species_b][repeat]
            matrix, _ = cross_similarity(rows_profile, cols_profile, windows)
            if not np.isfinite(matrix).all():
                continue
            margins.append(C.matching_margin(matrix, list(windows))[2])
        if not margins:
            continue
        values = np.array(margins, dtype=float)
        rows.append(
            dict(
                species_a=species_a,
                species_b=species_b,
                n_model_null_repeats=len(values),
                model_null_mean=float(values.mean()),
                model_null_sd=float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                model_null_q95=float(np.quantile(values, 0.95)),
            )
        )
    return pd.DataFrame(rows)


def model_null_path(config: dict) -> Path:
    """Where this run's model-side matched-random null lives (s01's own output tree)."""
    return C.config_path(config, "results") / MODEL_STEP / "random_null_scores.tsv.gz"


def load_model_null_profiles(
    config: dict, model_windows: pd.DataFrame, audit: Audit, windows: tuple[str, ...]
) -> dict:
    """{species: {repeat: DataFrame(motif_id x window)}} from s01's matched-random null."""
    path = model_null_path(config)
    if not path.exists():
        audit.add("model_null", "random_null_scores.tsv.gz", "missing_file", "reference columns left empty")
        return {}
    header = pd.read_csv(path, sep="\t", nrows=0)
    columns = set(header.columns)
    motif_col = "motif_id" if "motif_id" in columns else ("motif" if "motif" in columns else None)
    if motif_col is None or not {"model", "repeat", "log2_OR"} <= columns:
        audit.add("model_null", "random_null_scores.tsv.gz", "unexpected_columns", ",".join(sorted(columns)))
        return {}
    parts = []
    reader = pd.read_csv(
        path, sep="\t", usecols=["model", "repeat", motif_col, "log2_OR"], chunksize=500_000
    )
    wanted = set(model_windows.model)
    for chunk in reader:
        chunk = chunk[chunk.model.isin(wanted)]
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        audit.add("model_null", "random_null_scores.tsv.gz", "no_astroglia_rows", "reference columns left empty")
        return {}
    null = pd.concat(parts, ignore_index=True).rename(columns={motif_col: "motif_id"})

    model_index = (
        model_windows.drop_duplicates("model").set_index("model")[["species", "stage"]]
    )
    null["species"] = null.model.map(model_index.species)
    null["stage"] = null.model.map(model_index.stage)
    null = null.dropna(subset=["species", "stage"])

    profiles: dict = {}
    for (species, repeat), block in null.groupby(["species", "repeat"]):
        matrix = block.pivot_table(
            index="motif_id", columns="stage", values="log2_OR", aggfunc="mean"
        )
        if not has_all_windows(matrix, windows):
            continue
        profiles.setdefault(str(species), {})[int(repeat)] = matrix
    return profiles


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------


def gate_a3(
    correlation_rows: list[dict],
    species_available: list[str],
    gate_species: tuple[str, ...],
    celltype_scope: str = PRIMARY_CELLTYPE_SCOPE,
    rho_key: str = PRIMARY_RHO_COLUMN,
    alternate_rows: list[dict] | None = None,
    input_provenance: dict | None = None,
    alternate_provenance: dict | None = None,
) -> dict:
    """readme 31 Gate A3, evaluated on the astroglia-only rho (rho_key).

    `gate_species` is the run's Astroglia roster (config-derived, see
    astroglia_species). The rule string and the two counts it quotes are built
    from that roster and from GATE_SPECIES_QUORUM, so a run over a different
    species set states the rule it actually applied.
    """
    per_species = {}
    for species in gate_species:
        values = [row for row in correlation_rows if row["species"] == species]
        positive = [row for row in values if np.isfinite(row[rho_key]) and row[rho_key] > 0]
        # The windows this species' statistic actually reached, in the run's own
        # table order. A species whose model side is missing a cross-species
        # window is evaluated over a truncated axis, and `evaluated: true` on its
        # own would not show it; these two fields do (the gap itself is also an
        # audit.tsv row, action=missing_cross_window).
        windows_evaluated: list[str] = []
        for row in values:
            for window in (row.get("pre_window"), row.get("post_window")):
                if window and str(window) not in windows_evaluated:
                    windows_evaluated.append(str(window))
        per_species[species] = dict(
            evaluated=species in species_available,
            rho_by_transition={row["transition"]: row[rho_key] for row in values},
            n_transitions_evaluated=len(values),
            windows_evaluated=windows_evaluated,
            n_transitions_with_positive_rho=len(positive),
            max_rho=max([row[rho_key] for row in values], default=np.nan),
        )
    # only a species the roster admits can count towards the quorum; the
    # denominator of the rule is the roster's own size
    n_positive = sum(
        1
        for species in species_available
        if species in per_species and per_species[species]["n_transitions_with_positive_rho"] > 0
    )
    n_evaluated = len([s for s in gate_species if s in species_available])
    provenance = input_provenance or {}
    degraded_species = sorted(
        species for species in species_available if provenance.get(species, {}).get("degraded", True)
    )
    if n_evaluated < GATE_SPECIES_QUORUM:
        status = "PENDING"
    else:
        status = "PASS" if n_positive >= GATE_SPECIES_QUORUM else "FAIL"
    if status == "PASS" and degraded_species:
        # A PASS produced from unit-weight / name-parsed aggregation must not
        # reach s11's claim table as a plain PASS.
        status = "PENDING"
    payload = dict(
        status=status,
        rule=(
            f"at least {GATE_SPECIES_QUORUM} of {len(gate_species)} species "
            f"({', '.join(gate_species)}) have rho > 0 for at least one transition"
        ),
        celltype_scope=celltype_scope,
        rho_column=rho_key,
        species_roster=list(gate_species),
        n_species_with_positive_rho=n_positive,
        n_species_evaluated=n_evaluated,
        n_species_required=GATE_SPECIES_QUORUM,
        input_weighting={
            species: provenance.get(species, {}).get("weighting", "unknown")
            for species in species_available
        },
        input_metadata_source={
            species: provenance.get(species, {}).get("metadata_source", "")
            for species in species_available
        },
        input_degraded=bool(degraded_species),
        input_degraded_species=degraded_species,
        input_degraded_reason=(
            "window means for "
            + ", ".join(degraded_species)
            + " were aggregated with the unit-weight fallback (no pseudobulk_metadata.tsv, "
            "or an unusable n_cells column), so the statistic is unweighted; see "
            "audit.tsv action=n_cells_missing / metadata_file_missing"
            if degraded_species
            else ""
        ),
        per_species=per_species,
    )
    if alternate_rows is not None:
        # Same rule on the every-pseudobulk realisation, reported so the reader
        # of a PASS knows which population produced it (and what the other one
        # would have said).
        payload["sensitivity_all_pseudobulks"] = gate_a3(
            alternate_rows,
            sorted({row["species"] for row in alternate_rows}),
            gate_species,
            celltype_scope=ALL_PSEUDOBULK_SCOPE,
            rho_key="rho",
            input_provenance=alternate_provenance,
        )
    return payload


def gate_a4(
    merged: pd.DataFrame,
    gate_species: tuple[str, ...],
    windows: tuple[str, ...],
    celltype_scope: str = PRIMARY_CELLTYPE_SCOPE,
    alternate: pd.DataFrame | None = None,
    input_provenance: dict | None = None,
    alternate_provenance: dict | None = None,
) -> dict:
    """Model direction consistency across species, confirmed by the raw direction.

    The direction compared is first->last of the run's cross-species windows (v1:
    prenatal minus adult_DN, v2: prenatal minus adult). Those two window names go
    into the payload and the rule string, so the fields a reader sees name the
    windows the verdict actually used instead of a frozen pair.

    The raw side of `merged` is the astroglia-only realisation; `alternate` is the
    every-pseudobulk one, reported as a sensitivity.
    """
    early_window, late_window = windows[0], windows[-1]
    direction = f"{early_window}_minus_{late_window}"
    model_key = f"model_direction_{direction}"
    raw_key = f"raw_direction_{direction}"
    evidence = {}
    n_pass = 0
    for family in GATE_FAMILIES:
        block = merged[merged.family == family]
        model_direction = {}
        raw_direction = {}
        for species in gate_species:
            sub = block[block.species == species].set_index("stage")
            if not {early_window, late_window} <= set(sub.index):
                continue
            model_direction[species] = sign_of(
                float(sub.loc[early_window, "model_family_score"])
                - float(sub.loc[late_window, "model_family_score"])
            )
            raw_direction[species] = sign_of(
                float(sub.loc[early_window, "raw_family_score"])
                - float(sub.loc[late_window, "raw_family_score"])
            )
        counts = {1: 0, -1: 0}
        for value in model_direction.values():
            if value != 0:
                counts[value] += 1
        consistent_sign = (
            1
            if counts[1] >= GATE_SPECIES_QUORUM
            else (-1 if counts[-1] >= GATE_SPECIES_QUORUM else 0)
        )
        n_consistent = max(counts.values()) if consistent_sign else 0
        n_raw_support = sum(
            1
            for species, value in model_direction.items()
            if value != 0 and raw_direction.get(species, 0) == value
        )
        passed = bool(
            consistent_sign != 0
            and n_consistent >= GATE_SPECIES_QUORUM
            and n_raw_support >= GATE_SPECIES_QUORUM
        )
        n_pass += int(passed)
        evidence[family] = dict(
            **{
                model_key: model_direction,
                raw_key: raw_direction,
            },
            consistent_direction=consistent_sign,
            n_species_consistent=n_consistent,
            n_species_raw_supporting=n_raw_support,
            passes=passed,
        )
    n_evaluated_families = sum(1 for value in evidence.values() if value[model_key])
    if n_evaluated_families == 0:
        status = "PENDING"
    else:
        status = "PASS" if n_pass >= GATE_FAMILIES_QUORUM else "FAIL"
    provenance = input_provenance or {}
    available = sorted({str(species) for species in merged.species}) if not merged.empty else []
    degraded_species = sorted(
        species for species in available if provenance.get(species, {}).get("degraded", True)
    )
    if status == "PASS" and degraded_species:
        status = "PENDING"
    payload = dict(
        status=status,
        rule=(
            f"at least {GATE_FAMILIES_QUORUM} of the {len(GATE_FAMILIES)} primary families "
            f"({', '.join(GATE_FAMILIES)}) change in the same "
            f"{transition_label(early_window, late_window)} direction in "
            f">= {GATE_SPECIES_QUORUM} of {len(gate_species)} species AND the raw chromVAR "
            f"direction agrees in >= {GATE_SPECIES_QUORUM} of {len(gate_species)} species"
        ),
        celltype_scope=celltype_scope,
        direction_windows=dict(early=early_window, late=late_window),
        species_roster=list(gate_species),
        n_families_passing=n_pass,
        n_families_required=GATE_FAMILIES_QUORUM,
        n_families_evaluated=n_evaluated_families,
        input_weighting={
            species: provenance.get(species, {}).get("weighting", "unknown")
            for species in available
        },
        input_degraded=bool(degraded_species),
        input_degraded_species=degraded_species,
        input_degraded_reason=(
            "the raw family score of "
            + ", ".join(degraded_species)
            + " came from unit-weight aggregation (see gates.A3.input_degraded_reason)"
            if degraded_species
            else ""
        ),
        per_family=evidence,
    )
    if alternate is not None:
        payload["sensitivity_all_pseudobulks"] = (
            gate_a4(
                alternate,
                gate_species,
                windows,
                celltype_scope=ALL_PSEUDOBULK_SCOPE,
                input_provenance=alternate_provenance,
            )
            if not alternate.empty
            else dict(
                status="PENDING",
                celltype_scope=ALL_PSEUDOBULK_SCOPE,
                reason="no species has the every-pseudobulk realisation for every "
                "cross-species window",
            )
        )
    return payload


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def figure_delta_scatter(
    delta: pd.DataFrame,
    correlations: list[dict],
    out_dir: Path,
    config: dict,
    gate_species: tuple[str, ...],
    transitions: tuple[tuple[str, str], ...],
) -> None:
    species_list = [s for s in gate_species if s in set(delta.species)]
    if not species_list:
        return
    colors = species_palette(species_list)
    labels = [transition_label(*pair) for pair in transitions]
    fig, axes = plt.subplots(
        len(labels), len(species_list), figsize=(3.5 * len(species_list), 3.6 * len(labels)), squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    for row, transition in enumerate(labels):
        for col, species in enumerate(species_list):
            ax = axes[row][col]
            _style_axis(ax)
            block = delta[(delta.species == species) & (delta.transition == transition)]
            ax.axhline(0, color=AXIS, linewidth=0.8)
            ax.axvline(0, color=AXIS, linewidth=0.8)
            # the primary (astroglia-only) realisation, the one the rho and the
            # gates are computed from
            ax.scatter(
                block.delta_model,
                block.delta_chromvar_astroglia,
                s=7,
                color=colors.get(species, SERIES[0]),
                alpha=0.6,
                linewidths=0,
            )
            record = next(
                (r for r in correlations if r["species"] == species and r["transition"] == transition),
                None,
            )
            subtitle = (
                f"rho = {record['rho']:.3f}, n = {record['n_motifs']}"
                if record and np.isfinite(record.get("rho", np.nan))
                else "rho = NA"
            )
            ax.set_title(f"{species} · {transition}\n{subtitle}", color=INK, fontsize=9, loc="left")
            ax.set_xlabel("delta model log2 OR", color=INK_2, fontsize=8)
            ax.set_ylabel("delta raw chromVAR Z (astroglia)", color=INK_2, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, out_dir, "delta_scatter_model_vs_raw", config)


def figure_family_trajectory(
    merged: pd.DataFrame,
    out_dir: Path,
    config: dict,
    gate_species: tuple[str, ...],
    windows: tuple[str, ...],
) -> None:
    species_list = [s for s in gate_species if s in set(merged.species)]
    if not species_list:
        return
    x = np.arange(len(windows))
    fig, axes = plt.subplots(1, len(species_list), figsize=(4.1 * len(species_list), 3.8), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for col, species in enumerate(species_list):
        ax = axes[0][col]
        _style_axis(ax)
        block = merged[merged.species == species]
        ends = []
        for family in GATE_FAMILIES:
            sub = block[block.family == family].set_index("stage")
            if not set(windows) <= set(sub.index):
                continue
            model = [float(sub.loc[window, "model_family_score"]) for window in windows]
            raw = [float(sub.loc[window, "raw_family_score"]) for window in windows]
            color = FAMILY_COLORS.get(family, SERIES[0])
            ax.plot(x, model, color=color, linewidth=2, marker="o", markersize=5, label=family)
            ax.plot(x, raw, color=color, linewidth=2, linestyle="--", marker="s",
                    markersize=5, markerfacecolor=SURFACE, markeredgewidth=1.4)
            ends.append((model[-1], family, color))
        ax.set_xticks(x)
        ax.set_xticklabels(list(windows))
        ax.set_title(species, color=INK, fontsize=9, loc="left")
        ax.set_ylabel("family score (centered)", color=INK_2, fontsize=8)
        ax.margins(x=0.28)
        # direct labels, pushed apart so four families never overprint each other
        low, high = ax.get_ylim()
        gap = 0.06 * (high - low)
        placed: list[float] = []
        for value, family, color in sorted(ends):
            y = value if not placed else max(value, placed[-1] + gap)
            placed.append(y)
            ax.annotate(
                family,
                xy=(x[-1], y),
                xytext=(6, 0),
                textcoords="offset points",
                color=color,
                fontsize=8,
                va="center",
                annotation_clip=False,
            )
        if placed:
            ax.set_ylim(low, max(high, placed[-1] + gap))
    style_handles = [
        plt.Line2D([], [], color=INK_2, linewidth=2, marker="o", markersize=5, label="model"),
        plt.Line2D([], [], color=INK_2, linewidth=2, linestyle="--", marker="s", markersize=5,
                   markerfacecolor=SURFACE, markeredgewidth=1.4, label="raw chromVAR"),
    ]
    fig.legend(handles=style_handles, loc="lower center", ncol=2, frameon=False, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    # The figure is deliberately NOT written here. Drawing the model's centered
    # log2 OR and the raw chromVAR family score on one shared y-axis compresses
    # every model curve onto y = 0 -- the two are different statistics on
    # different peak universes and differ by roughly 40x -- so the file that used
    # to be written here was a misleading visual, not a fixable one. The
    # visualisation now belongs to three dedicated scripts that each state what
    # is being compared:
    #   s05_plot_family_trajectory_standardized.py   MAIN: z-scored overlay
    #   s05_plot_family_trajectory_absolute.py       SUPP: one source per row
    #   s05_plot_family_trajectory_agreement.py      SUPP: rho + concordance
    # They read family_trajectory_model_vs_raw_long.tsv and
    # family_trajectory_agreement.tsv, both built by
    # s05_prep_family_trajectory.py from this step's tables.
    plt.close(fig)


# --------------------------------------------------------------------------
# pending guard
# --------------------------------------------------------------------------


def pending_summary(
    config: dict,
    out_dir: Path,
    reason: str,
    windows: tuple[str, ...],
    transitions: tuple[tuple[str, str], ...],
    audit: Audit | None = None,
) -> dict:
    """The PENDING report - and the audit that goes with it.

    An early exit produces no gate payload, which is exactly when the audit
    trail matters most: the models the config's species filter removed (and
    every other exclusion reached so far) would otherwise be described only
    inside gate payloads that were never written. audit.tsv is therefore written
    on every PENDING exit, and the same drop records are named in
    definitions.species_filter.
    """
    filter_config = (config.get("eligibility") or config.get("species_filter")) or {}
    dropped_models = list(audit.dropped_models) if audit is not None else []
    if audit is not None:
        C.save_table(audit.frame(), out_dir / "audit.tsv", sep="\t")
    summary = dict(
        step=STEP,
        status="PENDING",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        reason=reason,
        gate_basis=GATE_BASIS,
        definitions=dict(
            windows=list(windows),
            transitions=[transition_label(*pair) for pair in transitions],
            species_filter=dict(
                min_cells_per_species=C.min_cells_per_species(config),
                celltype_column=str(filter_config.get("celltype_column", "")),
                dropped_models=dropped_models,
                note=(
                    "a species enters the Astroglia comparison only above "
                    "species_filter.min_cells_per_species cells of the model's target "
                    "celltype; dropped_models names every Astroglia model the filter "
                    "removed, with the count that removed it (audit.tsv "
                    "action=species_below_cell_floor)"
                ),
            ),
        ),
        gates={
            "A3": dict(status="PENDING", reason=reason, celltype_scope=PRIMARY_CELLTYPE_SCOPE),
            "A4": dict(status="PENDING", reason=reason, celltype_scope=PRIMARY_CELLTYPE_SCOPE),
        },
        # A PENDING exit still says which file the family layer A4 rests on would
        # have been read from, so the boundary is stated in every summary this step
        # writes, not only in the ones that reach the gate.
        family_layer=_family_layer_or_absent(),
        raw_margin_null_note=raw_margin_null_note(len(windows)),
        audit_rows=int(len(audit.rows)) if audit is not None else 0,
        outputs=["astroglia_validation_summary.json", "audit.tsv"],
    )
    C.save_json(summary, out_dir / "astroglia_validation_summary.json")
    return summary


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def run(config: dict) -> dict:
    audit = Audit()
    out_dir = C.results_dir(config, STEP)
    catalog = C.motif_catalog(config)

    # The run's own period axes: windows in the config's table order, and the
    # transitions as their consecutive pairs. A cross-species comparison needs at
    # least one window pair; a table that declares fewer is reported, not crashed.
    windows, transitions = period_axes(config)
    if len(windows) < 2:
        reason = (
            f"config {C.display_path(C.stage_map_path(config))} declares "
            f"{len(windows)} cross-species window(s) ({list(windows)}); a cross-species "
            "comparison needs at least two"
        )
        C.log("PENDING:", reason)
        return pending_summary(config, out_dir, reason, windows, transitions, audit=audit)

    model_path = C.config_path(config, "results") / MODEL_STEP / "all_model_motif_scores.tsv"
    chromvar_root = C.config_path(config, "results") / CHROMVAR_STEP
    species_dirs = (
        sorted(p.name for p in chromvar_root.iterdir() if p.is_dir()) if chromvar_root.exists() else []
    )
    with_z = [
        species
        for species in species_dirs
        if (chromvar_root / species / "chromvar_z.tsv.gz").exists()
    ]

    if not model_path.exists():
        C.log("PENDING: missing", model_path)
        return pending_summary(
            config, out_dir, f"missing input {model_path}", windows, transitions, audit=audit
        )
    if not with_z:
        C.log("PENDING: no chromvar_z.tsv.gz under", chromvar_root)
        return pending_summary(
            config,
            out_dir,
            f"missing chromVAR output under {chromvar_root}",
            windows,
            transitions,
            audit=audit,
        )

    stage_lookup, period_lookup = load_stage_lookups(config, audit)
    assert_gate_families(config, audit)
    # Named in the step's own audit table, never left for a reader to infer from
    # the code: which file governs the family layer Gate A4 is decided on.
    audit.add(
        "tf_family_map",
        "source",
        "family_layer_source",
        (
            f"Gate A4 and the family trajectory read `{C.display_path(FAMILY_LAYER_FILE)}` "
            f"(primary families {C.primary_families()}); this is a separate frozen "
            "interpretation input and NOT modules.file, so it is untouched by the "
            "module-layer invariance of the primary numbers"
        ),
    )
    family_table = C.tf_family_table()
    family_table = family_table.assign(family=family_table.family.fillna(""))

    # Who may enter the comparison at all: the manifest species that carry an
    # Astroglia model above the config's cell-count floor.
    gate_species = astroglia_species(config, audit)
    if not gate_species:
        reason = (
            "no Astroglia model in the ranked manifest clears the configured species "
            f"filter ({len(audit.dropped_models)} model(s) removed), so gates A3/A4 have "
            "no species to be evaluated on; the removed models are named in "
            "definitions.species_filter.dropped_models and in audit.tsv"
        )
        C.log("PENDING:", reason)
        return pending_summary(config, out_dir, reason, windows, transitions, audit=audit)

    model_windows = load_model_windows(config, period_lookup, audit, windows)
    if model_windows.empty:
        C.log("PENDING: no Astroglia model rows for the cross-species windows")
        return pending_summary(
            config,
            out_dir,
            "no Astroglia model rows mapped to the cross windows",
            windows,
            transitions,
            audit=audit,
        )
    # A cross-species window the model input lacks for a roster species would
    # otherwise cost a whole transition with no record: audit it before the
    # evaluated species list is fixed below.
    audit_model_window_coverage(model_windows, gate_species, windows, audit)

    # raw_primary_profiles is the PRIMARY, gate-driving realisation (astroglia
    # only, carried by the *_astroglia* columns); raw_all_profiles is the
    # every-pseudobulk sensitivity, carried by the plain columns and never gated.
    raw_primary_profiles: dict[str, pd.DataFrame] = {}
    raw_all_profiles: dict[str, pd.DataFrame] = {}
    window_rows = []
    modelled = set(model_windows.species)
    for species in with_z:
        if species not in modelled:
            audit.add(
                "chromvar",
                species,
                "no_astroglia_model",
                "chromVAR present but the species has no Astroglia model; not evaluated",
                species,
            )
            continue
        raw = load_chromvar_matrix(
            chromvar_root / species / "chromvar_z.tsv.gz", catalog, audit, species
        )
        primary, primary_rows = window_means(
            config, species, raw, stage_lookup, audit, windows, celltype=ASTROGLIA_CELLTYPE
        )
        window_rows.append(primary_rows)
        if has_all_windows(primary, windows):
            raw_primary_profiles[species] = primary
        every, every_rows = window_means(config, species, raw, stage_lookup, audit, windows)
        window_rows.append(every_rows)
        if has_all_windows(every, windows):
            raw_all_profiles[species] = every

    window_summary = (
        pd.concat(window_rows, ignore_index=True) if window_rows else pd.DataFrame()
    )

    # (species, celltype_scope) -> how that window mean was actually aggregated,
    # so every gate payload can state whether its statistic is cell-count
    # weighted or came from the unit-weight fallback.
    provenance: dict[tuple[str, str], dict] = {}
    if not window_summary.empty:
        for row in window_summary.itertuples():
            provenance[(str(row.species), str(row.celltype_scope))] = dict(
                weighting=str(row.weighting),
                metadata_source=str(row.metadata_source),
                degraded=str(row.weighting) != WEIGHTING_N_CELLS,
            )

    species_list = [
        s
        for s in gate_species
        if s in raw_primary_profiles and has_all_windows(raw_primary_profiles[s], windows)
    ]
    # Every species that was read or could have been read is accounted for here,
    # in the gate roster or not: a species with full model + chromVAR coverage
    # that falls outside the manifest-derived roster must not disappear without
    # a trace.
    for species in sorted(set(with_z) | set(modelled)):
        if species not in with_z:
            audit.add("chromvar", species, "no_chromvar_output", "species not evaluated", species)
        elif species not in raw_primary_profiles:
            audit.add(
                "raw_windows",
                species,
                "astroglia_scope_incomplete",
                "the primary (astroglia) realisation has no complete cross-window profile; "
                "species not evaluated for gates A3/A4",
                species,
            )
        if species in with_z and species not in gate_species:
            audit.add(
                "chromvar",
                species,
                "species_outside_gate_scope",
                f"chromVAR present but {species} is not in the Astroglia gate roster "
                f"{list(gate_species)} (manifest species above the configured cell floor); "
                "window means were computed, gates A3/A4 not evaluated",
                species,
            )
    primary_provenance = gate_provenance(provenance, species_list, PRIMARY_CELLTYPE_SCOPE)
    all_provenance = gate_provenance(provenance, species_list, ALL_WINDOW_SCOPE)

    # the plain columns are the every-pseudobulk sensitivity; the astroglia-only
    # realisation is joined onto the same rows and is what the gates read
    delta = build_delta_table(model_windows, raw_all_profiles, species_list, windows)
    if delta.empty:
        C.log("PENDING: no species has every cross-species window with model and raw data")
        return pending_summary(
            config,
            out_dir,
            "no species with complete model+raw window coverage",
            windows,
            transitions,
            audit=audit,
        )

    delta = attach_families(delta, family_table)

    if raw_primary_profiles:
        primary_delta = build_delta_table(
            model_windows, raw_primary_profiles, species_list, windows
        )
        if not primary_delta.empty:
            primary_delta = primary_delta[
                ["species", "transition", "motif_id", "raw_pre", "raw_post", "delta_chromvar"]
            ].rename(
                columns={
                    "raw_pre": "raw_astroglia_pre",
                    "raw_post": "raw_astroglia_post",
                    "delta_chromvar": "delta_chromvar_astroglia",
                }
            )
            delta = delta.merge(
                primary_delta, on=["species", "transition", "motif_id"], how="left"
            )
    for column in ("raw_astroglia_pre", "raw_astroglia_post", "delta_chromvar_astroglia"):
        if column not in delta.columns:
            delta[column] = np.nan
    delta["same_sign"] = (
        np.sign(delta.delta_model) == np.sign(delta.delta_chromvar)
    ).where(delta.delta_chromvar.notna())
    delta["same_sign_astroglia_only"] = (
        np.sign(delta.delta_model) == np.sign(delta.delta_chromvar_astroglia)
    ).where(delta.delta_chromvar_astroglia.notna())

    correlation_rows = []
    concordance_rows = []
    alternate_correlation_rows = []
    for species in species_list:
        for pre, post in transitions:
            label = transition_label(pre, post)
            block = delta[(delta.species == species) & (delta.transition == label)]
            if block.empty:
                continue
            # plain columns: the every-pseudobulk sensitivity; *_astroglia_only:
            # the primary realisation gate A3 is evaluated on
            primary = correlate(block, "delta_model", "delta_chromvar_astroglia")
            every = correlate(block, "delta_model", "delta_chromvar")
            correlation_rows.append(
                dict(
                    species=species,
                    transition=label,
                    pre_window=pre,
                    post_window=post,
                    **every,
                    rho_astroglia_only=primary["rho"],
                    pvalue_astroglia_only=primary["pvalue"],
                    n_motifs_astroglia_only=primary["n_motifs"],
                    concordance_astroglia_only=primary["concordance"],
                    primary_celltype_scope=PRIMARY_CELLTYPE_SCOPE,
                )
            )
            alternate_correlation_rows.append(
                dict(
                    species=species,
                    transition=label,
                    pre_window=pre,
                    post_window=post,
                    **every,
                )
            )
            concordance_rows.append(
                dict(
                    species=species,
                    transition=label,
                    n_motifs=every["n_motifs"],
                    n_concordant=every["n_concordant"],
                    n_discordant=every["n_discordant"],
                    n_tied=every["n_tied"],
                    concordance=every["concordance"],
                    binom_pvalue=every["binom_pvalue"],
                    binom_test="one-sided greater vs 0.5 (ties excluded)",
                    n_motifs_astroglia_only=primary["n_motifs"],
                    concordance_astroglia_only=primary["concordance"],
                    primary_celltype_scope=PRIMARY_CELLTYPE_SCOPE,
                )
            )
    correlations = pd.DataFrame(correlation_rows)
    concordance = pd.DataFrame(concordance_rows)

    # the raw side of the family trajectory (gate A4 support) is the primary
    # astroglia-only realisation, not the every-pseudobulk one
    merged_family, _ = family_table_model_vs_raw(
        model_windows, raw_primary_profiles, family_table, species_list, windows
    )
    family_corr = (
        family_correlations(merged_family, family_table) if not merged_family.empty else pd.DataFrame()
    )
    merged_family_all = (
        family_table_model_vs_raw(
            model_windows, raw_all_profiles, family_table, species_list, windows
        )[0]
        if raw_all_profiles
        else pd.DataFrame()
    )

    long_similarity, margins = stage_similarity_raw(raw_primary_profiles, species_list, windows)
    _, model_margins = stage_similarity_raw(
        {
            species: model_windows[model_windows.species == species]
            .pivot_table(
                index="motif_id",
                columns="stage",
                values="log2_OR_centered",
                aggfunc="mean",
            )
            .reindex(columns=list(windows))
            for species in species_list
        },
        species_list,
        windows,
    )

    null_source_path = model_null_path(config)
    null_profiles = load_model_null_profiles(config, model_windows, audit, windows)
    model_null = (
        model_null_margin_reference(config, model_windows, null_profiles, audit, windows)
        if null_profiles
        else pd.DataFrame()
    )

    margin_table = margins.copy()
    if not margin_table.empty:
        margin_table = margin_table.merge(
            model_margins[["species_a", "species_b", "margin"]].rename(
                columns={"margin": "margin_model_only"}
            ),
            on=["species_a", "species_b"],
            how="left",
        )

    null_table = margins.copy()
    if not null_table.empty:
        null_table = null_table[
            ["species_a", "species_b", "margin", "p_perm", "p_perm_min", "n_permutations"]
        ].rename(columns={"margin": "margin_observed", "p_perm": "p_perm_exact"})
        null_table["test"] = (
            f"raw_label_permutation_{factorial(len(windows))}_column_orderings"
        )
        null_table["note"] = raw_margin_null_note(len(windows))
        # Over every column-label ordering the smallest attainable exact p is
        # 2/(n_perm + 1); the strongest statement available is that the observed
        # ordering is the best one of them all.
        null_table["p_perm_floor"] = [
            2.0 / (n + 1) for n in null_table.n_permutations
        ]
        null_table["observed_is_best_ordering"] = null_table.p_perm_exact <= null_table.p_perm_floor
        if not model_null.empty:
            null_table = null_table.merge(model_null, on=["species_a", "species_b"], how="left")
        else:
            for column in (
                "n_model_null_repeats",
                "model_null_mean",
                "model_null_sd",
                "model_null_q95",
            ):
                null_table[column] = np.nan
        # the model-side null lives wherever THIS run's config points, so the
        # column names the file the reference was actually read from
        null_table["model_null_source"] = (
            f"{C.display_path(null_source_path)} (model side only; not a raw null)"
        )

    gates = {
        "A3": gate_a3(
            correlation_rows,
            species_list,
            gate_species,
            alternate_rows=alternate_correlation_rows or None,
            input_provenance=primary_provenance,
            alternate_provenance=all_provenance,
        ),
        "A4": (
            gate_a4(
                merged_family,
                gate_species,
                windows,
                alternate=merged_family_all,
                input_provenance=primary_provenance,
                alternate_provenance=all_provenance,
            )
            if not merged_family.empty
            else dict(
                status="PENDING",
                celltype_scope=PRIMARY_CELLTYPE_SCOPE,
                input_provenance=primary_provenance,
            )
        ),
    }

    C.save_table(tag_gate_basis(delta), out_dir / "delta_model_vs_delta_chromvar.tsv", sep="\t")
    C.save_table(tag_gate_basis(correlations), out_dir / "delta_correlation.tsv", sep="\t")
    C.save_table(tag_gate_basis(concordance), out_dir / "direction_concordance.tsv", sep="\t")
    C.save_table(
        tag_gate_basis(
            merged_family,
            ["species", "stage", "family", "analysis_level", "family_score", "n_members"],
        ),
        out_dir / "family_trajectory_model_vs_raw.tsv",
        sep="\t",
    )
    C.save_table(
        tag_gate_basis(
            family_corr, ["family", "analysis_level", "scope", "n_points", "rho", "pvalue"]
        ),
        out_dir / "family_trajectory_correlation.tsv",
        sep="\t",
    )
    C.save_table(
        tag_gate_basis(
            long_similarity, ["species_a", "species_b", "window_a", "window_b", "rho"]
        ),
        out_dir / "stage_similarity_raw.tsv",
        sep="\t",
    )
    C.save_table(
        tag_gate_basis(margin_table, ["species_a", "species_b", "margin"]),
        out_dir / "stage_matching_margin_raw.tsv",
        sep="\t",
    )
    save_table_with_comment(
        tag_gate_basis(
            null_table, ["species_a", "species_b", "margin_observed", "p_perm_exact"]
        ),
        out_dir / "raw_margin_null_test.tsv",
        [
            "raw_margin_null_test.tsv",
            f"The raw {len(windows)}x{len(windows)} matrix is built from the PRIMARY, "
            "astroglia-only window means",
            f"(celltype_scope={PRIMARY_CELLTYPE_SCOPE}); the every-pseudobulk realisation",
            "is a sensitivity only and is not used here.",
            f"p_perm_exact = exact permutation over the {factorial(len(windows))} "
            "column-label orderings of the RAW",
            f"chromVAR {len(windows)}x{len(windows)} stage-similarity matrix "
            "(common.permutation_margin_pvalue).",
            "This is a label-permutation test of the raw matrix, NOT a matched-random-peak null:",
            "no random-peak background exists on the chromVAR side.",
            "model_null_* columns are the read-only model-side matched-random reference from",
            f"{C.display_path(null_source_path)}, reported for context only.",
        ],
    )
    if not window_summary.empty:
        # one row per realisation, so the gate basis is stated per row rather than
        # as a constant: only the PRIMARY_CELLTYPE_SCOPE rows drive gates A3/A4
        window_summary = window_summary.assign(
            is_gate_basis=window_summary.celltype_scope == GATE_BASIS
        )
        C.save_table(window_summary, out_dir / "raw_window_summary.tsv", sep="\t")
    C.save_table(audit.frame(), out_dir / "audit.tsv", sep="\t")

    figure_delta_scatter(delta, correlation_rows, out_dir, config, gate_species, transitions)
    figure_family_trajectory(merged_family, out_dir, config, gate_species, windows)

    summary = dict(
        step=STEP,
        status="COMPLETE",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        species_evaluated=species_list,
        species_roster=list(gate_species),
        species_with_chromvar=with_z,
        n_motifs_catalog=int(len(catalog)),
        # top-level statement of which raw realisation gates A3/A4; mirrored by a
        # `gate_basis` column in every gate-feeding table
        gate_basis=GATE_BASIS,
        gate_basis_note=(
            "raw realisation the pre-registered gates A3 and A4 are decided on; the "
            f"every-pseudobulk realisation ({ALL_PSEUDOBULK_SCOPE}) is a sensitivity "
            "reported as gates.A3/A4.sensitivity_all_pseudobulks and never drives a gate"
        ),
        definitions=dict(
            windows=list(windows),
            transitions=[transition_label(*pair) for pair in transitions],
            model_centering_scope=(
                f"per-motif mean over the cross-species windows of the species "
                f"({', '.join(windows)}); {single_species_window_note(config)} is excluded "
                "so the model and the raw side share one origin"
            ),
            raw_centering_scope=(
                f"per-motif mean over the same cross-species windows ({', '.join(windows)})"
            ),
            raw_window_mean=(
                "n_cells-weighted mean of pseudobulk chromVAR Z within the window, taken "
                "over the finite entries only: a missing Z removes its own cell weight from "
                "the denominator instead of entering as an observed zero, and a motif with "
                "no finite Z in the window stays NaN so every consumer drops it "
                "(per species x window counts in audit.tsv, action=nonfinite_z)"
            ),
            primary_celltype_scope=PRIMARY_CELLTYPE_SCOPE,
            raw_weighting_scheme=(
                f"{WEIGHTING_N_CELLS} when s03's pseudobulk_metadata.tsv supplied a usable "
                f"n_cells for every pseudobulk, {WEIGHTING_UNIT_FILLED} when it did not (file "
                "absent, or n_cells missing/non-numeric) and every pseudobulk entered at weight "
                "1. The scheme this run actually used is per species x window in "
                "raw_window_summary.tsv (weighting) and per species in "
                "gates.A3/A4.input_weighting; a gate whose statistic came from unit weights is "
                "reported PENDING rather than PASS"
            ),
            celltype_scope=(
                f"PRIMARY = {PRIMARY_CELLTYPE_SCOPE}: the astroglia-only window means are what "
                "gate A3 and gate A4 are evaluated on - they are the *_astroglia* columns of "
                "delta_model_vs_delta_chromvar.tsv (delta_chromvar_astroglia), the "
                f"delta_correlation.tsv / direction_concordance.tsv column {PRIMARY_RHO_COLUMN}, "
                "the raw side of family_trajectory_model_vs_raw.tsv (gate A4 raw support) and "
                f"the raw {len(windows)}x{len(windows)} stage similarity / matching margin "
                "(stage_similarity_raw.tsv, "
                "stage_matching_margin_raw.tsv, raw_margin_null_test.tsv) behind gate A3. The "
                "every-pseudobulk realisation (every celltype of the species, n_cells-weighted: "
                f"{ALL_PSEUDOBULK_SCOPE}, the plain columns of those tables) is a documented "
                "sensitivity, reported for gate A3/A4 as gates.A3/A4.sensitivity_all_pseudobulks, "
                "and never drives a gate. Restricting the gate to the population the Astroglia "
                f"{C.topk_label(config)} model is about matters because in the real pseudobulk metadata the GC "
                "population alone carries about 70% of the cell weight of a window."
            ),
            species_roster=(
                "the species gates A3/A4 may be evaluated on, derived from the ranked "
                "manifest restricted to the Astroglia analysis and filtered by the config's "
                "species_filter (a species enters only above min_cells_per_species cells of "
                "the model's target celltype; any model the filter drops is an audit row, "
                "action=species_below_cell_floor)"
            ),
        ),
        delta_correlation=correlation_rows,
        gates=gates,
        # The family layer's governing file, recorded next to the gate it decides:
        # it is interpretation, but it is NOT `modules.file`, so the module-layer
        # invariance statement does not cover it. `family_layer_contract()` is the
        # single source of this statement and the invariance test asserts it.
        family_layer=family_layer_contract(),
        raw_margin_null_note=raw_margin_null_note(len(windows)),
        raw_margin_permutation_floor=raw_margin_permutation_floor_note(len(windows)),
        audit_rows=int(len(audit.rows)),
        outputs=[
            "delta_model_vs_delta_chromvar.tsv",
            "delta_correlation.tsv",
            "direction_concordance.tsv",
            "family_trajectory_model_vs_raw.tsv",
            "family_trajectory_correlation.tsv",
            "stage_similarity_raw.tsv",
            "stage_matching_margin_raw.tsv",
            "raw_margin_null_test.tsv",
            "raw_window_summary.tsv",
            "audit.tsv",
            "delta_scatter_model_vs_raw.pdf (PDF only: the raster copy is not "
            "part of the paper's deliverable set)",
            "(family trajectory figures are produced by the dedicated "
            "s05_plot_family_trajectory_*.py scripts, not by this step)",
        ],
    )
    C.save_json(summary, out_dir / "astroglia_validation_summary.json")
    return summary


def main() -> None:
    C.main_guard()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    C.add_common_args(parser)
    args = parser.parse_args()
    config = C.load_config(args.config)
    with C.StepTimer(STEP):
        summary = run(config)
    if summary["status"] == "PENDING":
        C.log("STATUS PENDING:", summary.get("reason", ""))
        sys.exit(0)
    for gate, payload in summary["gates"].items():
        C.log(gate, payload["status"], payload.get("rule", ""))


if __name__ == "__main__":
    main()
