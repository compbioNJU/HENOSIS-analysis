"""Motif forest: one dot per individual motif, per species and period.

Why this file exists
--------------------
Every other step of this analysis speaks in modules: it reports a median over the
monomeric members of a module and moves on. A median is a summary -- it can be
carried by one member and hide the rest -- and the motifs a reader actually wants
to look up (LHX/DLX next to POU, SOX6/8/10 next to OLIG1/2/3, NKX2-2 next to
POU3F3/4, RFX next to NFI next to NF-Y next to SOX) are exactly the ones a module
median flattens. This step expands the modules back into their individual motif
models and draws their effect size as a dot plot:

    x      = log2_OR                (effect direction and size)
    area   = -log10(FDR)            (evidence)
    colour = the period / window the model was ranked on
    facet  = species                (the analysis names the figure itself)
    y      = one row per motif, grouped by the module it belongs to

Nothing is recomputed. The log2 odds ratio, its FDR and the significance flag are
read from the s01 table exactly as s01 wrote them: a figure that re-derives a test
can disagree with the table it cites, and this one must not. The one derived
quantity is a 95% Wald interval for log2_OR, built from the s01 table's own stored
2x2 cell counts (fg_hits / bg_hits / n_fg / n_bg) so that a dot of given size shows
how much of its effect survives its own sampling error; it is a display annotation
and decides nothing. If the table does not carry those counts, the interval columns
are empty and the run says so.

The motif lists are the module map, not a config list
----------------------------------------------------
What to draw is read from the FROZEN, cell-type-specific
``config/celltype_motif_module_map.tsv`` through the ``common`` helpers:

  * ``common.celltype_module_table``  -- the modules of one cell type, their level
                                         and their provenance
  * ``common.modules_for``            -- which modules that cell type declares
  * ``common.module_members``         -- the motifs of one module (monomers only:
                                         a heterodimer is not one factor's effect)

and every motif of every module the map declares for that cell type is drawn. The
one thing deliberately NOT drawn is the derived aggregate (``Homeodomain_total``):
it is the union of two atomic modules this panel already expands, and adding it
would give its motifs a second row and double-count them. A
module the map deliberately records -- a secondary or a comparator module -- is not
dropped in silence: its motifs travel with their ``module`` and ``module_level``,
the row block is labelled with the module and its level, and the summary names the
non-gating modules it drew. The layer still gates nothing: this step reads the s01
table and writes figures and tables, it never writes a primary artefact, and no dot
here can change a PASS/FAIL.

The config may narrow the panel with two optional keys,
``motif_forest.modules`` (per cell type) and ``motif_forest.levels``; anything they
remove is named in the exclusion table. A flat ``motif_forest.motifs`` list (the
shape v1 and the first v2 draft used) cannot say which cell type a motif belongs to
and no longer drives this panel -- it is reported in the summary rather than obeyed
or ignored quietly.

Everything that could be a constant is derived at run time
----------------------------------------------------------
No window name, species list, species count, motif name or Top-k value is written
into this file. They come from the config (``--config``, default
``config/analysis_v2.yaml``), the period table (``common.stage_table``), the shared
``common.SPECIES_ORDER`` / ``common.ANALYSES``, the module map, and the s01 table
itself, which is what says which models exist at all.

Where the output goes
---------------------
``--out-dir`` defaults to ``results_v2/15_motif_forest`` -- the v2 tree -- and the
abandoned ``results/`` tree is refused outright, because a default that silently
depends on which config was loaded is how a dead tree gets a fresh set of figures.

Nothing is dropped quietly: an analysis with no modules in the map, a module the
config narrowed away, a heterodimer the map records, a motif the s01 table has no
row for, a non-finite effect, a period the stage map does not declare and a model
below the eligibility floor each get a row in an exclusion table, and the same
counts appear in the summary json.

Outputs
-------
    <out_dir>/motif_effect.tsv
    <out_dir>/motif_forest_<analysis>.png / .pdf
    <out_dir>/motif_forest_manifest.tsv
    <out_dir>/motif_forest_audit.tsv
    <out_dir>/motif_forest_exclusions.tsv
    <out_dir>/motif_forest_summary.json

  * ``motif_effect.tsv`` -- the dots, one row per (analysis, species, window,
    motif) with the module and level it comes from.
  * ``motif_forest_audit.tsv`` -- the panel's membership ledger: one row per motif
    the module map records for a cell type (monomer and heterodimer alike), saying
    whether it was drawn and, if not, why.
  * ``motif_forest_exclusions.tsv`` -- one uniform long table
    (``kind, analysis, species, period, entity, n, reason``) for everything the
    figure does not contain, whatever the reason.
  * ``motif_forest_manifest.tsv`` -- provenance per plotted
    (analysis, species, window) cell: the figure, the effect table, the s01 file
    and its checksum, the config, the module map, the stage map, the size scale,
    and the note that says why this panel exists at all.

Run
---
    PY=python
    $PY script/s17_motif_forest.py --config config/analysis_v2.yaml

(This step was formerly ``script/s15_motif_effect_forest.py``.)
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

STEP = "15_motif_forest"
SCORES_NAME = "all_model_motif_scores.tsv"
UPSTREAM_STEP = "01_model_motif"
DEFAULT_CONFIG = "config/analysis_v2.yaml"

# The two result trees of the project. `results/` is the abandoned v1 tree and no
# v2 step may write into it any more; `results_v2/` is the only place this step
# writes, so `--out-dir` defaults there and the other tree is refused.
V2_RESULTS_ROOT = C.ROOT / "results_v2"
DEAD_RESULTS_ROOT = C.ROOT / "results"

Z_95 = 1.959963984540054

# The sentence every manifest row carries: the panel's whole reason for existing.
PANEL_NOTE = (
    "a module median can hide the individual motif, which is the whole reason this "
    "panel exists: each module is expanded back into its own motif models here"
)

MOTIF_AUDIT_COLUMNS = [
    "analysis",
    "module",
    "module_level",
    "provenance",
    "motif_id",
    "motif_name",
    "member_type",
    "drawn",
    "n_models",
    "note",
]
EXCLUSION_COLUMNS = ["kind", "analysis", "species", "period", "entity", "n", "reason"]
# The first fourteen columns are the panel's contract; the rest are provenance, so
# a dot can be traced back to the one s01 row it came from.
EFFECT_COLUMNS = [
    "analysis",
    "celltype",
    "species",
    "period",
    "motif_id",
    "motif_name",
    "module",
    "module_level",
    "log2_OR",
    "ci_low",
    "ci_high",
    "FDR",
    "neg_log10_FDR",
    "significant",
    "rank_period",
    "window_kind",
    "model",
    "topk",
]
MANIFEST_COLUMNS = [
    "analysis",
    "species",
    "period",
    "rank_period",
    "window_kind",
    "model",
    "modules",
    "module_levels",
    "n_motifs",
    "n_significant",
    "figure",
    "table",
    "source_file",
    "source_sha256",
    "config_file",
    "config_sha256",
    "module_map",
    "module_map_sha256",
    "stage_map",
    "topk",
    "fdr",
    "size_scale_max",
    "panel_note",
]

# --------------------------------------------------------------------------
# frozen look
#
# The same 180 mm width and categorical palette as the main figure step, so a
# reader cannot tell the two were drawn by different scripts. The eight slots
# are handed to windows in period-table order and are never cycled: a run with
# more distinct windows than slots is an error, not a generated hue.
# --------------------------------------------------------------------------

MM = 1.0 / 25.4
FIG_WIDTH_IN = 180.0 * MM
# A panel is one row per motif. The v2 modules expand to ~50-100 rows per cell
# type, so the height follows the row count at a legible pitch and only extreme
# cases are clipped: crushing 100 rows into a 4-inch box is what makes the labels
# unreadable, which is worse than a tall supplementary figure.
ROW_PITCH_IN = 0.155
MIN_PANEL_HEIGHT_IN = 1.6
MAX_PANEL_HEIGHT_IN = 20.0
MIN_PANEL_WIDTH_IN = 1.35

CATEGORICAL = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#d9d8d3"
ZERO_RULE = "#8a8880"

# marker area in points^2; the floor keeps a non-significant dot visible as a
# small ring rather than making "no evidence" and "absent" look alike.
AREA_MIN = 7.0
AREA_MAX = 150.0
DOT_EDGE_WIDTH = 0.7
# vertical offset between two JASPAR models of one transcription factor
DOT_DODGE = 0.30
MODULE_GAP = 0.55


def apply_style(dpi: int) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "axes.edgecolor": INK_SOFT,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
        }
    )


# --------------------------------------------------------------------------
# configuration-derived vocabulary
# --------------------------------------------------------------------------


def forest_block(config: dict) -> dict:
    block = config.get("motif_forest")
    if block is not None and not isinstance(block, dict):
        raise RuntimeError("config.motif_forest must be a mapping when present")
    return block or {}


def fdr_threshold(config: dict) -> float:
    return float(config["model_analysis"]["fdr"])


def primary_topk(config: dict) -> int:
    return int(config["model_analysis"]["primary_topk"])


def stage_table_checked(config: dict) -> pd.DataFrame:
    table = C.stage_table(config)
    missing = [c for c in ("species", "rank_period", "cross_window") if c not in table.columns]
    if missing:
        raise RuntimeError(
            f"{C.display_path(C.stage_map_path(config))} is missing column(s) {missing}; a "
            "ranking period cannot be mapped onto a cross-species window"
        )
    if table[["species", "rank_period", "cross_window"]].isna().any().any():
        raise RuntimeError(
            f"{C.display_path(C.stage_map_path(config))} has an incomplete species/period row"
        )
    return table


def window_lookup(
    config: dict,
) -> tuple[dict[tuple[str, str], str], dict[str, str], list[str], dict[str, list[str]]]:
    """Period -> window maps and the declared vocabulary, all from the period table.

    v2 declares early_embryonic/prenatal/postnatal/adult, and a cell type whose
    ranking period is not a stage-map period (Purkinje and Oligo are ranked over
    their whole development) lands here under its own name: the window a model
    belongs to is read off the table, never written in code.
    """
    table = stage_table_checked(config)
    per_species: dict[tuple[str, str], str] = {}
    by_period: dict[str, str] = {}
    species_by_window: dict[str, list[str]] = {}
    for row in table.itertuples():
        window = str(row.cross_window)
        per_species[(str(row.species), str(row.rank_period))] = window
        by_period.setdefault(str(row.rank_period), window)
        species_by_window.setdefault(window, [])
        if str(row.species) not in species_by_window[window]:
            species_by_window[window].append(str(row.species))
    declared = list(dict.fromkeys(str(w) for w in table.cross_window))
    return per_species, by_period, declared, species_by_window


def window_colour_map(
    declared: list[str], present: list[str]
) -> tuple[dict[str, str], list[str]]:
    """Fixed colour slot per window, in period-table order.

    A window the stage map does not declare (a whole-development ranking period,
    for instance) takes a slot after every declared one, in sorted order, so the
    assignment is a function of the vocabulary rather than of which windows one
    figure happens to contain.
    """
    extra = sorted(w for w in present if w not in declared)
    order = list(declared) + extra
    if len(order) > len(CATEGORICAL):
        raise RuntimeError(
            f"{len(order)} distinct windows ({order}) exceed the {len(CATEGORICAL)} frozen "
            "colour slots; facet or fold rather than cycling hues"
        )
    return {window: CATEGORICAL[i] for i, window in enumerate(order)}, order


# --------------------------------------------------------------------------
# the module map: what to draw
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModuleSpec:
    """One module of one cell type, expanded into its monomeric motif models."""

    analysis: str
    module: str
    level: str
    provenance: str
    label: str
    motif_ids: tuple[str, ...]


def module_levels_declared(config: dict) -> list[str] | None:
    declared = forest_block(config).get("levels")
    if declared is None:
        return None
    return [str(value) for value in declared]


def module_names_declared(config: dict, celltype: str) -> list[str] | None:
    declared = forest_block(config).get("modules")
    if declared is None:
        return None
    if not isinstance(declared, dict):
        raise RuntimeError("config.motif_forest.modules must map a cell type to a module list")
    if celltype not in declared:
        return None
    value = declared[celltype]
    return None if value is None else [str(item) for item in value]


def module_label(config: dict, module: str, level: str) -> str:
    """The rule label above a module block; a non-gating module says so."""
    return module if C.module_is_gating(config, level) else f"{module} ({level})"


def panel_plan(config: dict, celltype: str, exclusions: list[dict]) -> list[ModuleSpec]:
    """Every module the map declares for one cell type, expanded to its motifs.

    The map is the cell-type-specific authority, so nothing it records is dropped
    in silence: the only ways a module can be left out are the two optional config
    narrowings, and each of those writes an exclusion row naming the module.
    """
    table = C.celltype_module_table(config)
    sub = table[table.celltype.astype(str) == celltype]
    if sub.empty:
        return []

    declared_modules = module_names_declared(config, celltype)
    declared_levels = module_levels_declared(config)
    plan: list[ModuleSpec] = []
    for module in dict.fromkeys(sub.module.astype(str)):
        rows = sub[sub.module.astype(str) == module]
        level = str(rows.level.iloc[0])
        provenance = str(rows.provenance.iloc[0])
        if declared_modules is not None and module not in declared_modules:
            exclusions.append(
                dict(kind="module", analysis=celltype, species="", period="", entity=module,
                     n=int(len(rows)),
                     reason="module not named by config.motif_forest.modules for this cell "
                            "type; its motifs are not drawn")
            )
            continue
        if declared_levels is not None and level not in declared_levels:
            exclusions.append(
                dict(kind="module", analysis=celltype, species="", period="", entity=module,
                     n=int(len(rows)),
                     reason=f"module level {level!r} not in config.motif_forest.levels "
                            f"{declared_levels}; its motifs are not drawn")
            )
            continue
        monomers = C.module_members(config, celltype, module, monomers_only=True)
        # module-map row order is the order the map's author grouped the motifs in
        # (LHX together, DLX together, the HOX clusters in order), which reads far
        # better on an axis than sorting by JASPAR accession.
        ordered: list[str] = []
        for motif_id in rows.loc[rows.member_type == "monomer", "motif_id"].astype(str):
            if motif_id in monomers and motif_id not in ordered:
                ordered.append(motif_id)
        if not ordered:
            exclusions.append(
                dict(kind="module", analysis=celltype, species="", period="", entity=module,
                     n=int(len(rows)),
                     reason="module has no monomeric member in the module map; there is "
                            "nothing to expand into an individual motif")
            )
            continue
        plan.append(
            ModuleSpec(
                analysis=celltype,
                module=module,
                level=level,
                provenance=provenance,
                label=module_label(config, module, level),
                motif_ids=tuple(ordered),
            )
        )
    return plan


def module_ledger(
    config: dict, celltype: str, plan: list[ModuleSpec]
) -> list[dict]:
    """One row per motif the map records for this cell type, drawn or not.

    The map stores heterodimers alongside monomers so that a reader can see they
    were considered. The panel draws monomer models only -- a heterodimer is not
    one factor's effect -- so every heterodimer of a drawn module gets a ledger row
    saying exactly that, rather than vanishing between the map and the figure.
    """
    table = C.celltype_module_table(config)
    sub = table[table.celltype.astype(str) == celltype]
    rows: list[dict] = []
    for spec in plan:
        block = sub[sub.module.astype(str) == spec.module]
        for row in block.itertuples():
            member = str(row.member_type)
            if member == "monomer":
                note = "not drawn: no row for this motif in the upstream table"
            else:
                note = ("not drawn: the panel expands monomeric binding models, and this map "
                        f"row is a {member} of two factors")
            rows.append(
                dict(
                    analysis=celltype, module=spec.module, module_level=spec.level,
                    provenance=spec.provenance, motif_id=str(row.motif_id),
                    motif_name=str(row.motif_name), member_type=member,
                    drawn=False, n_models=0, note=note,
                )
            )
    return rows


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------


@dataclass
class Scores:
    frame: pd.DataFrame
    path: Path
    sha256: str


def load_scores(config: dict, override: str | None) -> Scores:
    if override:
        path = Path(override)
        if not path.is_absolute():
            path = Path.cwd() / path
    else:
        path = C.config_path(config, "results") / UPSTREAM_STEP / SCORES_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{C.display_path(path)} does not exist. This step reads the upstream motif table "
            "and never recomputes enrichment: run the motif-enrichment step against this "
            "config first."
        )
    frame = pd.read_csv(path, sep="\t")
    required = {
        "analysis", "species", "period", "model", "topk", "motif_id", "motif_name",
        "log2_OR", "FDR",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{C.display_path(path)} is missing column(s) {missing}")
    # The significance flag is read, never re-derived. A table without it cannot be
    # trusted to agree with the figure, so the run stops instead of inventing one.
    if "significant" not in frame.columns:
        raise RuntimeError(
            f"{C.display_path(path)} has no `significant` column. This step reads that flag "
            "rather than re-deriving the test, so it refuses to draw a figure that could "
            "disagree with the table it cites."
        )
    return Scores(frame=frame, path=path, sha256=C.sha256_file(path))


# --------------------------------------------------------------------------
# effect table
# --------------------------------------------------------------------------


def analysis_order(present: list[str]) -> list[str]:
    known = [a for a in C.ANALYSES if a in present]
    return known + sorted(a for a in present if a not in C.ANALYSES)


def species_order(present: list[str]) -> list[str]:
    known = [s for s in C.SPECIES_ORDER if s in present]
    return known + sorted(s for s in present if s not in C.SPECIES_ORDER)


def ci_columns(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, bool]:
    """95% Wald interval for log2_OR from the table's own stored 2x2 counts.

    This is the display interval the plan asks for next to the effect, and it is
    built from numbers the s01 table already wrote (fg_hits, bg_hits, n_fg, n_bg)
    with the same Haldane-Anscombe 0.5 correction the enrichment itself used. It
    re-tests nothing, decides nothing and, when the counts are absent, is reported
    as unavailable rather than guessed.
    """
    counts = ("fg_hits", "bg_hits", "n_fg", "n_bg")
    if not set(counts) <= set(frame.columns):
        empty = np.full(len(frame), np.nan)
        return empty, empty, False
    a = frame.fg_hits.to_numpy(dtype=float)
    b = frame.bg_hits.to_numpy(dtype=float)
    n_fg = frame.n_fg.to_numpy(dtype=float)
    n_bg = frame.n_bg.to_numpy(dtype=float)
    variance = (
        1.0 / (a + 0.5)
        + 1.0 / (b + 0.5)
        + 1.0 / (n_fg - a + 0.5)
        + 1.0 / (n_bg - b + 0.5)
    )
    half = Z_95 * np.sqrt(variance)
    centre = frame.log2_OR.to_numpy(dtype=float)
    return centre - half, centre + half, True


def build_effect_table(
    config: dict, scores: pd.DataFrame, plans: dict[str, list[ModuleSpec]],
    exclusions: list[dict]
) -> pd.DataFrame:
    """The dots: one row per (analysis, species, window, motif) actually drawn."""
    per_species, by_period, _, _ = window_lookup(config)
    cross = set(C.cross_windows(config))
    single = set(C.single_species_windows(config))

    # (analysis, motif_id) -> the module that claims it. A motif id may belong to
    # the same module of two cell types (SOX2 is in the Astroglia and the Oligo SOX
    # module) -- the key keeps each cell type's panel to its own modules.
    membership: dict[tuple[str, str], ModuleSpec] = {}
    name_of: dict[tuple[str, str], str] = {}
    for analysis, plan in plans.items():
        for spec in plan:
            for motif_id in spec.motif_ids:
                membership[(analysis, motif_id)] = spec

    keep = scores[scores.topk.astype("Int64") == primary_topk(config)].copy()
    other = scores[scores.topk.astype("Int64") != primary_topk(config)]
    if len(other):
        exclusions.append(
            dict(kind="model", analysis="|".join(sorted(other.analysis.astype(str).unique())),
                 species="", period="",
                 entity="|".join(sorted(other.model.astype(str).unique())), n=int(len(other)),
                 reason=f"topk != model_analysis.primary_topk ({primary_topk(config)}); "
                        "not plotted (a sensitivity table must not leak into the "
                        "canonical figure)")
        )

    keys = list(zip(keep.analysis.astype(str), keep.motif_id.astype(str)))
    keep = keep[[key in membership for key in keys]].copy()
    if keep.empty:
        return pd.DataFrame(columns=EFFECT_COLUMNS)
    keep["module"] = [membership[key].module for key in zip(keep.analysis.astype(str), keep.motif_id.astype(str))]
    keep["module_level"] = [
        membership[key].level for key in zip(keep.analysis.astype(str), keep.motif_id.astype(str))
    ]

    windows, kinds = [], []
    for row in keep.itertuples():
        period = str(row.period)
        window = per_species.get((str(row.species), period)) or by_period.get(period) or period
        windows.append(window)
        if window in cross:
            kinds.append("cross_species")
        elif window in single:
            kinds.append("single_species")
        else:
            kinds.append("undeclared_by_stage_map")
    keep["rank_period"] = keep["period"].astype(str)
    keep["period"] = windows
    keep["window_kind"] = kinds

    for window in sorted(set(keep.loc[keep.window_kind == "undeclared_by_stage_map", "period"])):
        exclusions.append(
            dict(kind="period", analysis="|".join(sorted(
                    keep.loc[keep.period == window, "analysis"].astype(str).unique())),
                 species="", period=window, entity=window, n=int((keep.period == window).sum()),
                 reason=f"period not declared in {C.display_path(C.stage_map_path(config))}; "
                        "plotted under its own name and not grouped into a cross-species window")
        )

    finite = np.isfinite(keep.log2_OR.to_numpy(dtype=float)) & np.isfinite(
        keep.FDR.to_numpy(dtype=float)
    )
    if not finite.all():
        bad = keep.loc[~finite]
        exclusions.append(
            dict(kind="point", analysis="|".join(sorted(bad.analysis.astype(str).unique())),
                 species="", period="",
                 entity="|".join(sorted(bad.model.astype(str).unique())), n=int(len(bad)),
                 reason="non-finite log2_OR or FDR in the upstream table; not plotted")
        )
        keep = keep.loc[finite]
    if keep.empty:
        return pd.DataFrame(columns=EFFECT_COLUMNS)

    alpha = fdr_threshold(config)
    # An FDR that underflowed to exactly zero has no finite -log10; it is shown
    # at the smallest positive FDR the table reports, which keeps the size scale
    # readable instead of collapsing every other dot to the floor.
    positive = keep.FDR[keep.FDR > 0]
    fdr_floor = float(positive.min()) if len(positive) else 1.0
    keep["neg_log10_FDR"] = -np.log10(keep.FDR.clip(lower=fdr_floor))
    # s01 flags a motif only when the effect is positive; read that flag rather
    # than re-deriving it, so the figure and the table can never disagree.
    keep["significant"] = keep.significant.astype(bool)
    ci_low, ci_high, _ = ci_columns(keep)
    keep["ci_low"] = ci_low
    keep["ci_high"] = ci_high
    keep["celltype"] = keep.celltype.astype(str) if "celltype" in keep.columns else ""
    keep = keep[EFFECT_COLUMNS].sort_values(
        ["analysis", "species", "period", "module", "motif_name", "motif_id"], kind="stable"
    )
    return keep.reset_index(drop=True)


def apply_eligibility(
    config: dict, effect: pd.DataFrame, exclusions: list[dict]
) -> tuple[pd.DataFrame, dict]:
    """Re-apply the v2 eligibility floors to the models this panel holds.

    The same ``common.apply_species_filter`` the upstream step uses, applied to the
    models the s01 table actually carries: idempotent when s01 already honoured the
    floors, and it catches a stale table instead of trusting one. Both floors are
    enforced and every drop is named here with the numbers that caused it (a v2
    example: the one Purkinje model that has too few cells AND too short a
    ranking). A model with no ranked file under this config cannot have its
    ranked-peak floor checked; it is kept, and said so, rather than dropped by a
    number nobody could compute.
    """
    block = config.get("eligibility") or config.get("species_filter")
    if not block or effect.empty:
        return effect, {
            "applied": False, "min_target_cells": None, "min_ranked_peaks": None,
            "n_models_dropped": 0, "dropped_models": [], "models_not_checked": [],
        }

    inventory = effect[["analysis", "species", "period", "model"]].drop_duplicates().reset_index(drop=True)
    inventory["celltype"] = inventory.analysis.map(C.ANALYSIS_CELLTYPE).fillna("")
    inventory["org_celltype"] = inventory.analysis.map(C.ANALYSIS_ORG_CELLTYPE)

    unknown = sorted(inventory.loc[inventory.org_celltype.isna(), "analysis"].unique())
    if unknown:
        # An analysis common.py has no target celltype for cannot be judged, so it
        # is kept and said out loud rather than dropped by a NaN lookup.
        exclusions.append(
            dict(kind="analysis", analysis="|".join(unknown), species="", period="",
                 entity="|".join(unknown), n="",
                 reason="no Org_Celltype mapping in common.ANALYSIS_ORG_CELLTYPE; left in "
                        "place, the eligibility floors are not applied to this analysis")
        )

    ranking = C.ranking_files(config)
    known = inventory.merge(
        ranking[["analysis", "species", "model", "n_ranked", "celltype", "org_celltype"]],
        on=["analysis", "species", "model"], how="inner", suffixes=("", "_ranked"),
    )
    known = known.drop(columns=["celltype_ranked", "org_celltype_ranked"], errors="ignore")
    known = known[known.org_celltype.notna()].reset_index(drop=True)

    unchecked = sorted(set(inventory.model) - set(known.model))
    for model in unchecked:
        row = inventory[inventory.model == model].iloc[0]
        exclusions.append(
            dict(kind="model", analysis=str(row.analysis), species=str(row.species),
                 period="|".join(sorted(inventory.loc[inventory.model == model, "period"]
                                        .astype(str).unique())),
                 entity=str(model), n="",
                 reason="no ranked file under this config's ranked_dirs, so the ranked-peak "
                        "floor cannot be checked; the model is kept and named here instead "
                        "of being dropped by a number that cannot be computed")
        )

    info = {
        "applied": True,
        "min_target_cells": C.min_cells_per_species(config),
        "min_ranked_peaks": C.min_ranked_peaks(config),
        "celltype_column": str(block.get("celltype_column", "Org_Celltype")),
        "n_models_dropped": 0,
        "dropped_models": [],
        "models_not_checked": unchecked,
    }
    if known.empty:
        return effect, info

    kept, audit = C.apply_species_filter(config, known)
    kept_models = set(kept.model)
    dropped = known[~known.model.isin(kept_models)]
    for row in audit.itertuples():
        n_models = int(((dropped.analysis == row.analysis) & (dropped.species == row.species)).sum())
        action = str(row.action)
        detail = action.split("dropped:", 1)[-1].strip() if "dropped:" in action else action
        exclusions.append(
            dict(kind="species", analysis=str(row.analysis), species=str(row.species),
                 period="", entity=str(row.model), n=int(row.n_cells),
                 reason=f"eligibility: {detail}; {n_models} model(s) of this species excluded "
                        "from this panel")
        )
    if len(kept) != len(known):
        effect = effect[~effect.model.isin(set(dropped.model))].reset_index(drop=True)
    info["n_models_dropped"] = int(len(dropped))
    info["dropped_models"] = sorted(dropped.model.astype(str))
    return effect, info


def coverage_exclusions(
    config: dict, effect: pd.DataFrame, exclusions: list[dict],
    plans: dict[str, list[ModuleSpec]], ledger: list[dict],
    analyses_available: list[str],
) -> list[str]:
    """Name every analysis, species and model the figure does not cover, and why.

    The analysis inventory is the UPSTREAM table's, not this panel's: an analysis
    the s01 table carries but the module map does not cover (or covers only with
    modules the config narrowed away) is the one case that would otherwise vanish
    between the input and the figure -- it has no plan, so nothing else here can
    see it. Passing `analyses_available` is what makes that case reachable, and
    the analyses it names are returned as well as written, so the summary can
    count them.

    The coverage of a module motif is checked per MODEL, not per cell type: a motif
    that got a dot in one species and a hole in another must not look complete
    because the row is missing from one model only.
    """
    # A species the floor already removed is named there, with the numbers that
    # removed it; repeating it here would be a second, emptier story.
    already = {(row["analysis"], row["species"]) for row in exclusions if row["kind"] == "species"}
    # Which cell types the module map itself knows, to tell "the map never covered
    # this analysis" from "the map covers it and the panel did not draw it".
    mapped = set(C.celltype_module_table(config).celltype.astype(str))
    analyses = analysis_order(
        sorted(set(analyses_available) | set(effect.analysis.astype(str)) | set(plans))
    )
    uncovered: list[str] = []
    for analysis in analyses:
        if analysis not in plans:
            uncovered.append(analysis)
            if analysis in mapped:
                reason = ("the cell-type module map declares modules for this analysis, but not "
                          "one of them reached the panel (a config narrowing or a module with no "
                          "monomeric member); see the module rows of this table")
            else:
                reason = ("no rows in the cell-type module map, so there is no module to "
                          "expand and no panel for this analysis")
            exclusions.append(
                dict(kind="analysis", analysis=analysis, species="", period="",
                     entity=analysis, n="", reason=reason)
            )
            continue
        sub = effect[effect.analysis == analysis]
        if sub.empty:
            exclusions.append(
                dict(kind="analysis", analysis=analysis, species="", period="",
                     entity=analysis, n="",
                     reason="none of this analysis' module motifs has a row in the upstream "
                            "table; there is nothing to draw")
            )
            continue
        present = set(sub.species.astype(str))
        for species in C.SPECIES_ORDER:
            if species in present or (analysis, species) in already:
                continue
            # No cell count here: the reason is that no model exists to judge, not
            # that a floor removed one, and printing a count would imply the latter.
            exclusions.append(
                dict(kind="species", analysis=analysis, species=species, period="",
                     entity=f"{analysis}__{species}", n="",
                     reason="no model for this species in the upstream table (no ranked file, "
                            "excluded upstream, or that arm still to run)")
            )
        expected = [motif_id for spec in plans[analysis] for motif_id in spec.motif_ids]
        for model, group in sub.groupby("model", sort=True):
            missing = [motif_id for motif_id in expected
                       if motif_id not in set(group.motif_id.astype(str))]
            if not missing:
                continue
            exclusions.append(
                dict(kind="motif", analysis=analysis, species=str(group.species.iloc[0]),
                     period="|".join(sorted(group.period.astype(str).unique())),
                     entity=str(model), n=len(missing),
                     reason="motif model(s) " + ",".join(sorted(missing))
                            + " have no row for this model in the upstream table, so they have "
                              "no dot")
            )
        # A monomeric motif the upstream table never scored for this cell type at all
        # leaves the ledger's `drawn` flag alone; its row says why instead of looking
        # skipped. Heterodimers have their own, and already correct, reason.
        seen_anywhere = set(sub.motif_id.astype(str))
        for entry in ledger:
            if (entry["analysis"] == analysis and entry["member_type"] == "monomer"
                    and entry["motif_id"] not in seen_anywhere):
                entry["note"] = ("not drawn: no row for this motif anywhere in the upstream "
                                 "table for this cell type; see the exclusion table")
    return uncovered


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------


def area_for(value: np.ndarray, vmax: float) -> np.ndarray:
    """Marker area in points^2: linear in -log10(FDR), floored at a visible dot."""
    scaled = np.clip(np.asarray(value, dtype=float) / max(vmax, 1e-9), 0.0, 1.0)
    return AREA_MIN + (AREA_MAX - AREA_MIN) * scaled


def size_ticks(vmax: float, fdr: float) -> list[float]:
    """Evidence values for the area legend, always including the s01 threshold.

    The threshold is kept even when it sits above the largest plotted value, so
    a run in which nothing reaches significance still says where the line is.
    """
    threshold = -math.log10(fdr)
    values = {0.0, float(vmax), float(vmax) / 2.0, threshold}
    return sorted({round(float(v), 2) for v in values})


def motif_layout(
    effect: pd.DataFrame, plan: list[ModuleSpec]
) -> tuple[list[str], dict[str, str], dict[str, float], list[tuple[str, str, float]]]:
    """Motif order, module of each name, y positions and the module rules.

    The panel's order is the module map's own order (each cell type's modules in
    the order the map declares them, each module's motifs in the order the map
    lists them), so the axis follows a cell-type-specific vocabulary rather than
    whichever motif happens to be significant in one species. A motif with two
    JASPAR models shares one row.
    """
    order: list[str] = []
    module_of: dict[str, str] = {}
    positions: dict[str, float] = {}
    blocks: list[tuple[str, str, float]] = []
    cursor = 0.0
    for spec in plan:
        names: list[str] = []
        for motif_id in spec.motif_ids:
            cell = effect[(effect.module == spec.module) & (effect.motif_id == motif_id)]
            if cell.empty:
                continue
            name = str(cell.motif_name.iloc[0])
            if name not in names:
                names.append(name)
        if not names:
            continue
        cursor += MODULE_GAP
        blocks.append((spec.label, spec.level, cursor - MODULE_GAP / 2.0))
        for name in names:
            order.append(name)
            module_of[name] = spec.module
            positions[name] = cursor
            cursor += 1.0
    return order, module_of, positions, blocks


def plot_analysis(
    analysis: str,
    effect: pd.DataFrame,
    plan: list[ModuleSpec],
    colour: dict[str, str],
    fdr: float,
    vmax: float,
    topk: int,
    dpi: int,
    formats: list[str],
    out_dir: Path,
    single_window_note: dict[str, str],
) -> list[Path]:
    sub = effect[effect.analysis == analysis]
    species_list = species_order(sorted(sub.species.astype(str).unique()))
    order, _, positions, blocks = motif_layout(sub, plan)

    ncols = min(3, len(species_list))
    nrows = int(math.ceil(len(species_list) / ncols))
    span = max(positions.values()) + 1.0
    panel_h = float(
        np.clip(ROW_PITCH_IN * span + 0.55, MIN_PANEL_HEIGHT_IN, MAX_PANEL_HEIGHT_IN)
    )
    panel_w = max(FIG_WIDTH_IN / ncols, MIN_PANEL_WIDTH_IN)

    # The grid, the shared x label and the legend strip are laid out in inches so
    # that a tall panel cannot push the legend over the axis labels.
    label_h = 0.58
    legend_h = 0.92
    title_h = 0.36
    fig_h = panel_h * nrows + label_h + legend_h + title_h
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel_w * ncols, fig_h), squeeze=False,
        sharex=True, sharey=True,
    )
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(
        left=0.082, right=0.988, top=1.0 - title_h / fig_h,
        bottom=(label_h + legend_h) / fig_h, wspace=0.20, hspace=0.28,
    )

    xmax = float(np.ceil(np.abs(sub.log2_OR).max() * 1.08 * 10) / 10) or 1.0
    ylim_top = max(positions.values()) + 0.55
    has_ci = bool(np.isfinite(sub.ci_low.to_numpy(dtype=float)).any())

    for index, species in enumerate(species_list):
        ax = axes[index // ncols][index % ncols]
        panel = sub[sub.species == species]
        ax.set_facecolor(SURFACE)

        # No per-row rule: the dots sit next to their labels. One rule per module
        # block, with the module (and its level when it may not gate) in the gap
        # above it, is what makes a 100-row axis read as a handful of named groups
        # -- and it is where a comparator module is visibly marked as one.
        for label, _level, y in blocks:
            ax.axhline(y, color=GRID, lw=0.7, zorder=1)
            # Right-aligned inside the panel: a long module label ("NuclearReceptor
            # (secondary)") would otherwise run past the axes and be painted over by
            # the neighbouring panel.
            ax.text(0.988, y, label, transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=5.4, color=INK_MUTED, zorder=6)
        ax.axvline(0.0, color=ZERO_RULE, lw=0.8, zorder=2)

        for (period, motif_name), cell in panel.groupby(["period", "motif_name"], sort=False):
            ids = sorted(cell.motif_id.astype(str).unique())
            for row in cell.itertuples():
                shift = (ids.index(str(row.motif_id)) - (len(ids) - 1) / 2.0) * DOT_DODGE
                y = positions[str(motif_name)] + shift
                area = float(area_for(np.array([row.neg_log10_FDR]), vmax)[0])
                edge = colour[str(period)]
                if has_ci and np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
                    lo, hi = sorted((float(row.ci_low), float(row.ci_high)))
                    ax.plot([max(lo, -xmax), min(hi, xmax)], [y, y], color=edge,
                            lw=DOT_EDGE_WIDTH, alpha=0.55, solid_capstyle="butt", zorder=3)
                ax.scatter(
                    [row.log2_OR], [y], s=area,
                    facecolors=edge if bool(row.significant) else SURFACE,
                    edgecolors=edge, linewidths=DOT_EDGE_WIDTH, zorder=4,
                )

        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(ylim_top, -0.65)
        ax.set_yticks([positions[n] for n in order])
        ticks = np.linspace(-xmax, xmax, 5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        if index % ncols == 0:
            ax.set_yticklabels(order)
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(species, color=INK, pad=3, loc="left")
        n_periods = int(panel.period.nunique())
        ax.text(1.0, 1.02, f"{n_periods} period" + ("s" if n_periods != 1 else ""),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, color=INK_MUTED)

    for index in range(len(species_list), nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")

    fig.suptitle(f"{analysis} — motif-level effect sizes by module", fontsize=8.5, color=INK,
                 y=1.0 - 0.06 / fig_h, va="top")
    x_label = f"log2 odds ratio (Top-{topk} vs GC/length-matched background)"
    if has_ci:
        x_label += "; bar = 95% CI from the table's own counts"
    fig.text(0.5, (label_h + legend_h - 0.26) / fig_h, x_label,
             ha="center", va="center", fontsize=7, color=INK_SOFT)

    # A legend strip under the grid, in three groups: identity (colour), evidence
    # (area) and the s01 significance rule. A hollow dot above the threshold
    # circle is an FDR-significant depletion (log2 OR < 0), which is why both
    # the ring and the size scale are shown.
    legend_frac = legend_h / fig_h
    present_windows = [w for w in colour if w in set(sub.period)]
    handles_w = [
        Line2D([], [], linestyle="", marker="o", markersize=3.6, markerfacecolor=colour[w],
               markeredgecolor=colour[w], label=w + single_window_note.get(w, ""))
        for w in present_windows
    ]
    handles_s = [
        Line2D([], [], linestyle="", marker="o",
               markersize=math.sqrt(float(area_for(np.array([v]), vmax)[0])),
               markerfacecolor=INK_MUTED, markeredgecolor=INK_MUTED,
               label=f"{v:g}" + ("   s01 FDR" if abs(v + math.log10(fdr)) < 1e-9 else ""))
        for v in size_ticks(vmax, fdr)
    ]
    handles_sig = [
        Line2D([], [], linestyle="", marker="o", markersize=4.2, markerfacecolor=INK_SOFT,
               markeredgecolor=INK_SOFT,
               label=f"significant (s01: FDR < {fdr:g} and log2 OR > 0)"),
        Line2D([], [], linestyle="", marker="o", markersize=4.2, markerfacecolor=SURFACE,
               markeredgecolor=INK_SOFT, markeredgewidth=DOT_EDGE_WIDTH,
               label="not significant"),
        # The module blocks are labelled inside the panel; a parenthesised level
        # there is a module the config says may never gate, which has to be said.
        Line2D([], [], linestyle="none", label="(level) = non-gating module: never gates"),
    ]
    groups = (
        (0.004, 0.300, handles_w, "period / window", 2, 0.6),
        (0.312, 0.280, handles_s, "-log10 FDR (marker area)", 2, 1.6),
        (0.600, 0.396, handles_sig, "significance", 1, 1.4),
    )
    for x0, width, handles, title, cols, spacing in groups:
        strip = fig.add_axes([x0, 0.012, width, legend_frac * 0.95])
        strip.axis("off")
        legend = strip.legend(
            handles=handles, loc="upper left", ncols=cols, title=title, title_fontsize=6.5,
            labelspacing=spacing, handletextpad=0.6, columnspacing=1.2, borderpad=0.0,
        )
        legend._legend_box.align = "left"

    return save_figure(fig, out_dir, f"motif_forest_{safe_name(analysis)}", formats, dpi)


def save_figure(fig, out_dir: Path, stem: str, formats: list[str], dpi: int) -> list[Path]:
    written = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        # A vector backend stamps a fresh /CreationDate into every file, which
        # would make two runs of this deterministic script differ byte for byte.
        metadata = (
            {"CreationDate": None, "ModDate": None} if fmt.lower() in {"pdf", "svg"} else None
        )
        fig.savefig(path, dpi=dpi, facecolor=SURFACE, metadata=metadata)
        written.append(path)
    plt.close(fig)
    return written


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def single_window_notes(config: dict) -> dict[str, str]:
    """A legend note for windows that only one species contributes to."""
    _, _, _, species_by_window = window_lookup(config)
    single = set(C.single_species_windows(config))
    notes = {}
    for window in single:
        members = species_by_window.get(window, [])
        notes[window] = (
            f"  ({'/'.join(members)} only — not cross-species comparable)"
            if members
            else "  (single species — not cross-species comparable)"
        )
    return notes


def guard_output_dir(path: Path, allow_dead_tree: bool) -> Path:
    """Refuse to write anywhere inside the abandoned v1 `results/` tree."""
    dead = DEAD_RESULTS_ROOT.resolve()
    resolved = path.resolve()
    if resolved == dead or dead in resolved.parents:
        if not allow_dead_tree:
            raise SystemExit(
                f"refusing to write to {C.display_path(path)}: "
                f"{C.display_path(DEAD_RESULTS_ROOT)} is the abandoned v1 result tree and v2 "
                "artefacts live under results_v2. Drop --out-dir to use the v2 tree, or pass "
                "--allow-v1-results if overwriting the dead tree is really intended."
            )
    return path


def resolve_out_dir(config: dict, requested: str | None, allow_dead_tree: bool) -> Path:
    """Where this run writes: the v2 step directory unless asked otherwise.

    With no --out-dir the config's result root is used, which for the default v2
    config is results_v2/15_motif_forest. The check runs *before* the directory is
    created, so a mis-pointed config cannot even leave an empty v1 step directory
    behind.
    """
    if requested:
        path = Path(requested)
        if not path.is_absolute():
            path = Path.cwd() / path
        guard_output_dir(path, allow_dead_tree)
        path.mkdir(parents=True, exist_ok=True)
        return path
    configured = C.config_path(config, "results") / STEP
    guard_output_dir(configured, allow_dead_tree)
    return C.results_dir(config, STEP)


def config_motif_block_report(config: dict) -> dict:
    """What a flat `motif_forest.motifs` list declared, and why it is not obeyed.

    The panel is per cell type; a flat list of names has no cell type and cannot
    express one. Rather than obey it silently (wrong for two of the three panels)
    or ignore it silently, the summary records what it said.
    """
    declared = forest_block(config).get("motifs")
    if not declared:
        return {"present": False}
    names = [str(value).strip() for value in declared if str(value).strip()]
    return {
        "present": True,
        "declared_motifs": names,
        "used": False,
        "reason": "the panel is cell-type specific and comes from the module map; a flat motif "
                  "list without a cell type cannot express that, so it is reported and not "
                  "obeyed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--out-dir", default=None,
        help="output directory; defaults to the result root of the loaded config "
             f"({C.display_path(V2_RESULTS_ROOT)}/15_motif_forest for the default v2 config). "
             f"The abandoned v1 tree ({C.display_path(DEAD_RESULTS_ROOT)}) is refused.",
    )
    parser.add_argument(
        "--scores", default=None,
        help="override the upstream motif table (testing/rebuild); default is "
             "<results>/01_model_motif/all_model_motif_scores.tsv",
    )
    parser.add_argument("--dpi", type=int, default=None, help="override plotting.dpi")
    parser.add_argument("--formats", nargs="*", default=None, help="override plotting.formats")
    parser.add_argument(
        "--allow-v1-results", action="store_true",
        help="permit writing inside the abandoned v1 results/ tree (off by default)",
    )
    args = parser.parse_args()
    C.main_guard()

    config = C.load_config(args.config)
    plotting = config.get("plotting") or {}
    dpi = int(args.dpi if args.dpi is not None else plotting.get("dpi", 200))
    formats = list(
        args.formats if args.formats is not None else plotting.get("formats", ["png", "pdf"])
    )
    apply_style(dpi)

    fdr = fdr_threshold(config)
    topk = primary_topk(config)

    exclusions: list[dict] = []
    with C.StepTimer(f"s17 motif forest ({C.display_path(Path(config['_path']))})"):
        # The input is checked before the output directory is created: a run that
        # has nothing to read must not leave an empty step directory behind.
        scores = load_scores(config, args.scores)
        C.log("read", C.display_path(scores.path), f"{len(scores.frame)} rows")
        analyses_available = analysis_order(
            sorted(scores.frame.analysis.astype(str).unique())
        )

        # The module map decides what a panel contains, before anything else.
        plans: dict[str, list[ModuleSpec]] = {}
        ledger: list[dict] = []
        for analysis in analyses_available:
            plan = panel_plan(config, analysis, exclusions)
            if plan:
                plans[analysis] = plan
                ledger.extend(module_ledger(config, analysis, plan))
        if not plans:
            raise RuntimeError(
                "no analysis in the upstream table has modules in "
                f"{C.display_path(C.module_map_path(config))}; there is nothing to expand and "
                "refusing to write an empty figure."
            )

        effect = build_effect_table(config, scores.frame, plans, exclusions)
        effect, eligibility = apply_eligibility(config, effect, exclusions)
        # The upstream analysis inventory goes in, not just the plans: a cell type
        # the s01 table carries but the module map does not cover has no plan, so
        # this is the only place that can name it instead of dropping it quietly.
        analyses_uncovered = coverage_exclusions(
            config, effect, exclusions, plans, ledger, analyses_available
        )
        panels = [a for a in analyses_available if a in plans and not effect[effect.analysis == a].empty]
        if not panels:
            raise RuntimeError(
                "no requested module motif survived into the effect table; refusing to write "
                "an empty figure. Check the module map against the upstream table."
            )
        out_dir = resolve_out_dir(config, args.out_dir, args.allow_v1_results)

        colour, window_order = window_colour_map(
            window_lookup(config)[2], sorted(effect.period.astype(str).unique())
        )
        vmax = float(effect.neg_log10_FDR.max())
        notes = single_window_notes(config)
        has_ci = bool(np.isfinite(effect.ci_low.to_numpy(dtype=float)).any())

        C.save_table(effect, out_dir / "motif_effect.tsv", sep="\t")

        # The ledger: every motif the map records for a plotted cell type, drawn or
        # not, with the module and level it belongs to.
        ledger_frame = pd.DataFrame(ledger, columns=MOTIF_AUDIT_COLUMNS)
        if not ledger_frame.empty:
            drawn = set(zip(effect.analysis.astype(str), effect.motif_id.astype(str)))
            n_models = (
                effect.groupby(["analysis", "motif_id"]).model.nunique().to_dict()
            )
            for index, row in ledger_frame.iterrows():
                key = (str(row.analysis), str(row.motif_id))
                if key in drawn:
                    ledger_frame.at[index, "drawn"] = True
                    ledger_frame.at[index, "n_models"] = int(n_models.get(key, 0))
                    ledger_frame.at[index, "note"] = (
                        "drawn: monomeric member of a module in this cell type's panel"
                    )
            ledger_frame = ledger_frame.sort_values(
                ["analysis", "module", "member_type", "motif_name"], kind="stable"
            ).reset_index(drop=True)
        C.save_table(ledger_frame, out_dir / "motif_forest_audit.tsv", sep="\t")

        exclusion_frame = pd.DataFrame(exclusions, columns=EXCLUSION_COLUMNS)
        if not exclusion_frame.empty:
            exclusion_frame = exclusion_frame.sort_values(
                ["kind", "analysis", "species", "period", "entity"], kind="stable"
            ).reset_index(drop=True)
        C.save_table(exclusion_frame, out_dir / "motif_forest_exclusions.tsv", sep="\t")

        config_sha = C.sha256_file(Path(config["_path"]))
        module_map = C.module_map_path(config)
        module_sha = C.sha256_file(module_map)
        manifest_rows: list[dict] = []
        figures: dict[str, list[str]] = {}
        for analysis in panels:
            written = plot_analysis(
                analysis, effect, plans[analysis], colour, fdr, vmax, topk, dpi, formats,
                out_dir, notes,
            )
            figures[analysis] = [C.display_path(p) for p in written]
            sub = effect[effect.analysis == analysis]
            for species in species_order(sorted(sub.species.astype(str).unique())):
                for period, group in sub[sub.species == species].groupby("period", sort=False):
                    manifest_rows.append(
                        dict(
                            analysis=analysis,
                            species=species,
                            period=str(period),
                            rank_period="|".join(sorted(group.rank_period.astype(str).unique())),
                            window_kind="|".join(sorted(group.window_kind.astype(str).unique())),
                            model="|".join(sorted(group.model.astype(str).unique())),
                            modules="|".join(spec.module for spec in plans[analysis]),
                            module_levels="|".join(
                                dict.fromkeys(spec.level for spec in plans[analysis])
                            ),
                            n_motifs=int(group.motif_name.nunique()),
                            n_significant=int(group.significant.sum()),
                            figure=";".join(C.display_path(p) for p in written),
                            table=C.display_path(out_dir / "motif_effect.tsv"),
                            source_file=C.display_path(scores.path),
                            source_sha256=scores.sha256,
                            config_file=C.display_path(Path(config["_path"])),
                            config_sha256=config_sha,
                            module_map=C.display_path(module_map),
                            module_map_sha256=module_sha,
                            stage_map=C.display_path(C.stage_map_path(config)),
                            topk=topk,
                            fdr=fdr,
                            size_scale_max=vmax,
                            panel_note=PANEL_NOTE,
                        )
                    )
        manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
        C.save_table(manifest, out_dir / "motif_forest_manifest.tsv", sep="\t")

        summary = {
            "step": STEP,
            "generated_by": C.display_path(Path(__file__).resolve()),
            "config_file": C.display_path(Path(config["_path"])),
            "config_sha256": config_sha,
            "module_map": C.display_path(module_map),
            "module_map_sha256": module_sha,
            "stage_map": C.display_path(C.stage_map_path(config)),
            "source_file": C.display_path(scores.path),
            "source_sha256": scores.sha256,
            "topk": topk,
            "fdr_threshold": fdr,
            "panel_rationale": PANEL_NOTE,
            "motif_source": "config/celltype_motif_module_map.tsv expanded into its monomeric "
                            "motifs: every module the map declares for a cell type, whether it "
                            "gates or not",
            "config_motif_block": config_motif_block_report(config),
            "n_modules_drawn": int(sum(len(plan) for plan in plans.values())),
            "module_plan": {
                analysis: [
                    dict(module=spec.module, level=spec.level, n_motifs=len(spec.motif_ids),
                         gating=C.module_is_gating(config, spec.level),
                         provenance=spec.provenance)
                    for spec in plans[analysis]
                ]
                for analysis in plans
            },
            "non_gating_modules_drawn": sorted(
                {
                    spec.module
                    for plan in plans.values()
                    for spec in plan
                    if not C.module_is_gating(config, spec.level)
                }
            ),
            "n_analyses_uncovered": len(analyses_uncovered),
            "analyses_uncovered": analyses_uncovered,
            "n_motif_models_drawn": int(len(ledger_frame[ledger_frame.drawn])),
            "n_heterodimers_not_drawn": int(
                len(ledger_frame[~ledger_frame.drawn]) if len(ledger_frame) else 0
            ),
            "ci": {
                "available": has_ci,
                "method": "95% Wald interval for log2_OR from the upstream table's stored "
                          "fg_hits/bg_hits/n_fg/n_bg (Haldane-Anscombe 0.5); display only",
            }
            if has_ci
            else {
                "available": False,
                "reason": "the upstream table does not carry the 2x2 counts the interval is "
                          "built from; no interval is drawn and none is guessed",
            },
            "window_order": window_order,
            "cross_windows": list(C.cross_windows(config)),
            "single_species_windows": list(C.single_species_windows(config)),
            "n_effect_rows": int(len(effect)),
            "n_fdr_underflow": int((effect.FDR <= 0).sum()),
            "analyses": {
                analysis: {
                    "n_species": int(effect[effect.analysis == analysis].species.nunique()),
                    "species": species_order(
                        sorted(effect[effect.analysis == analysis].species.astype(str).unique())
                    ),
                    "n_points": int((effect.analysis == analysis).sum()),
                    "n_motif_names": int(effect.loc[effect.analysis == analysis, "motif_name"].nunique()),
                    "n_significant": int(effect.loc[effect.analysis == analysis, "significant"].sum()),
                    "modules": [spec.module for spec in plans[analysis]],
                    "periods": sorted(effect.loc[effect.analysis == analysis, "period"].astype(str).unique()),
                    "figure": figures.get(analysis, []),
                }
                for analysis in panels
            },
            "eligibility": eligibility,
            "n_exclusions": int(len(exclusion_frame)),
            "exclusion_kinds": (
                sorted(exclusion_frame.kind.astype(str).unique().tolist())
                if len(exclusion_frame)
                else []
            ),
            "size_scale": {
                "value_max": vmax,
                "area_min": AREA_MIN,
                "area_max": AREA_MAX,
                "note": "marker area is linear in -log10(FDR) and shared by every analysis of "
                        "this run, so the panels are comparable across figures",
            },
            "outputs": {
                "effect_table": C.display_path(out_dir / "motif_effect.tsv"),
                "manifest": C.display_path(out_dir / "motif_forest_manifest.tsv"),
                "audit": C.display_path(out_dir / "motif_forest_audit.tsv"),
                "exclusions": C.display_path(out_dir / "motif_forest_exclusions.tsv"),
                "summary": C.display_path(out_dir / "motif_forest_summary.json"),
            },
        }
        C.save_json(summary, out_dir / "motif_forest_summary.json")
        C.log(
            "wrote",
            len(effect),
            "effect rows,",
            len(manifest),
            "manifest rows,",
            len(exclusion_frame),
            "exclusion rows ->",
            C.display_path(out_dir),
        )


if __name__ == "__main__":
    main()
