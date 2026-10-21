"""Step 18 -- illustrative per-peak sequence maps (SUPPLEMENTARY, NOT EVIDENCE).

WHY this file exists
--------------------
s01 turns a model ranking into an enrichment table and every later step reasons
about those numbers. Nothing in that chain shows a reader what the operation
"for each primary module, take the highest-ranked peak that actually carries a
motif of that module and read the motifs off its sequence" looks like on one
concrete piece of DNA. This step draws that, and only that: for each model of
the run, one 500 bp reference sequence per primary module of the model's cell
type, and the cached FIMO hits laid out along it, coloured by TF family, capped
at `sequence_maps.max_hits_per_peak` hits per peak. The cap keeps the strongest
hits but always retains a motif of the module that chose the peak, so a track
cannot be labelled with a module whose motif it does not show.

The peak-selection rule is fixed by the v2 plan and is never widened to find a
more photogenic example:

    for each primary module of the model's cell type
        take the highest-ranked peak within Top50 that carries at least one
        motif of that module, as recorded by the frozen whole-universe FIMO
        scan;
        if no such peak exists, record NO_QUALIFYING_PEAK_TOP50 for that
        (model, module) and do NOT search past Top50.

Searching below Top50 would quietly turn a fixed illustration into a selected
example, so the negative case is reported as a negative case instead.

THE MODULE LAYER IS INTERPRETATION ONLY
---------------------------------------
The modules come from `config/celltype_motif_module_map.tsv` (cell-type
specific) plus the derived unions declared in `modules.aggregates`. Reading that
map decides which peaks this supplementary figure shows and nothing else: this
step reads no `all_model_motif_scores.tsv`, computes no gate, no similarity, no
margin and no null, and writes nothing outside its own step directory. A module
at a non-gating level (`secondary`, `comparator`) is never selected, and a
declared aggregate is only drawn when every module it unions is gating. Pointing
`modules.file` at an empty map therefore changes this figure and nothing else.

What this file is NOT
---------------------
It is an illustration of individual peaks. It is not a statistic, it carries no
null model, and its peaks are selected by rank inside a capped window, so the
selection cannot be steered by the enrichment outcome. The figure is
supplementary and must not be cited as support for any claim. The sentences in
`ILLUSTRATION_BOUNDARY` and `SUPPLEMENTARY_NOTE` are written into every figure,
into the manifest (as the caption) and into the summary, so the artefact cannot
be read out of context.

Design consequences of that purpose
-----------------------------------
  * every window, species and species count comes from the config, the stage map
    (`C.stage_table`, which follows `paths.stage_map`) or the filtered manifest.
    Nothing about the run is spelled out as a literal here;
  * the run is scoped by `C.ranking_files` + `C.apply_species_filter`, exactly
    like s01/s02/s06/s08, so a species below the eligibility floor keeps the
    same recorded reason it has everywhere else;
  * a *hit* is taken from the frozen, whole-universe FIMO scan
    (`cache/<species>_motif_peak_hits.csv.gz`, written by `common.scan_motifs`).
    That cache records which motifs hit a peak but not where; the positions are
    recovered by re-running FIMO -- with the identical frozen threshold, and
    only on the handful of representative peaks of this step -- into a separate
    position cache. The two are cross-checked peak by peak and any disagreement
    lands in the audit, so the sequence map can never silently contradict the
    scan the rest of the project is built on;
  * the same cache is what decides "this peak carries a motif of this module";
    when a species has no membership cache, no module of that species' models
    can be resolved at all and the model is skipped with that reason rather than
    guessed from a motif-free sequence;
  * nothing is dropped silently. Models removed by the species filter, models
    whose cell type is absent from the module map, models without a membership
    cache, empty rankings, (model, module) pairs with no qualifying peak in
    Top50, peaks without an extractable sequence, peaks without a hit, hits cut
    by the per-peak cap, stronger hits displaced so that the module's own motif
    survives that cap, peaks whose module motif the position scan could not
    reproduce, and motifs with no family assignment all get a row in
    sequence_maps_audit.tsv.

Outputs (under `C.results_dir(config, "16_sequence_maps")`, i.e. results_v2/)

    representative_peaks.tsv    one row per drawn peak: model, module, rank,
                                score, coordinates, GC, hit counts, families
    sequence_map_<model>.png / .pdf
                                one figure per model, one track per module
    sequence_maps_manifest.tsv  one row per model: what was drawn, which files,
                                the caption and the two boundary sentences
    sequence_maps_audit.tsv     every exclusion / caveat, with a reason
    sequence_maps_summary.json  run scope, counts, colour map, boundary text

A debug slice (`--limit-models`, `--model`, `--species`) writes into a
`scope_<token>/` subdirectory so a partial run can never overwrite the frozen
full-run tables. `--self-test` checks the components on hand-built frames and
then runs the real code path end to end against a throwaway config; it writes
nowhere near results/ or results_v2/ and takes about ten seconds.

The published tree (`results/`) is frozen, so this step refuses to write it:
`--results-root <dir>` redirects the output and `--allow-v1-results` is the
explicit opt-in for the case where that tree really is the target.

    python \
        script/s18_sequence_maps.py --config config/analysis_v2.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

STEP = "16_sequence_maps"

# The peak-selection rule fixed by the v2 plan: a primary module contributes at
# most the highest-ranked peak that carries one of its motifs, and only inside
# this rank window. The window is a presentation knob, never a species, window
# or cell-type name; it is read from `sequence_maps.top_rank_window` and only
# falls back to the plan's fixed value when the config does not declare it.
DEFAULT_TOP_RANK_WINDOW = 50
DEFAULT_MAX_HITS_PER_PEAK = 6

# The level name of the module map that the selection rule uses. A module at a
# level the config declares non-gating is never selected.
PRIMARY_LEVEL = "primary"

# The two sentences every figure, its caption and the manifest must carry. They
# are separate so a reader cannot get one without the other.
ILLUSTRATION_BOUNDARY = (
    "ILLUSTRATIVE ONLY -- an illustrative example of individual peaks, not "
    "evidence about the population. Motif presence is not TF binding or "
    "activity, and this figure is not statistical evidence for any claim."
)
SUPPLEMENTARY_NOTE = (
    "SUPPLEMENTARY -- this figure and its tables must not be cited as support "
    "for any claim."
)


def figure_caption(model: str) -> str:
    """The caption written next to every figure and into the manifest."""
    return (
        f"Sequence map for {model} (supplementary). One track per primary "
        f"module of the model's cell type: the highest-ranked peak within the "
        f"Top-ranking window that carries a motif of that module, its reference "
        f"sequence and the cached FIMO hits along it. "
        f"{ILLUSTRATION_BOUNDARY} {SUPPLEMENTARY_NOTE}"
    )

REPRESENTATIVE_COLUMNS = [
    "model",
    "analysis",
    "species",
    "period",
    "window",
    "window_scope",
    "module",
    "module_level",
    "module_n_members",
    "module_motifs_on_peak",
    "peak",
    "rank",
    "score",
    "chrom",
    "start",
    "end",
    "length",
    "gc",
    "valid",
    "n_hits",
    "n_hit_occurrences",
    "n_hits_drawn",
    "n_motifs_drawn",
    "families_hit",
    "families_hit_all",
    "hits_drawn",
]

MANIFEST_COLUMNS = [
    "model",
    "analysis",
    "species",
    "period",
    "window",
    "window_scope",
    "module_celltype",
    "n_modules",
    "n_modules_drawn",
    "n_modules_without_qualifying_peak",
    "modules_without_qualifying_peak",
    "n_peaks",
    "n_peaks_with_hits",
    "n_peaks_without_hits",
    "n_peaks_without_sequence",
    "n_hits_distinct",
    "n_hit_occurrences",
    "n_hits_drawn",
    "status",
    "figure_png",
    "figure_pdf",
    "figure_caption",
    "illustration_boundary",
    "supplementary_note",
]

AUDIT_COLUMNS = [
    "entity_type",
    "entity",
    "model",
    "species",
    "period",
    "reason",
    "detail",
    "n",
]

# --------------------------------------------------------------------------
# frozen look, shared with the other figure steps
# --------------------------------------------------------------------------

MM = 1.0 / 25.4
FIG_WIDTH_IN = 180.0 * MM

# Colour-blind safe categorical slots in a fixed order.
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
NEUTRAL_FILL = "#f0efec"
# The legend entry for a motif the project's TF-family map does not name. The
# v2 plan fixes this wording: it is a named bucket of annotated motifs, not a
# claim that the motif is un-annotated.
OTHER_MOTIF_COLOUR = "#b8b6ae"
OTHER_MOTIF_LABEL = "Other annotated motifs"
MAX_LEGEND_COLUMNS = 6

# track geometry, shared by draw_model_map and its label placement so a label
# can never be dropped on top of its neighbour's
PLOT_LEFT_FRACTION = 0.30
PLOT_RIGHT_FRACTION = 0.985
DATA_LEFT_FRACTION = -0.33
DATA_RIGHT_FRACTION = 1.30
HIT_LABEL_FONTSIZE = 5.0
# mean glyph advance as a fraction of the font size, for DejaVu Sans at 5 pt
GLYPH_ASPECT = 0.62
# vertical offsets, in lane heights, tried in order when two labels collide
LABEL_LEVELS = ((0, 0.0), (1, 0.30), (-1, -0.30))


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
            "axes.titlesize": 8,
            "axes.labelsize": 7.5,
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
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "text.color": INK,
        }
    )


def save_figure(fig: "plt.Figure", out_dir: Path, stem: str, formats: list[str], dpi: int) -> list[Path]:
    written = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        # Vector backends stamp a fresh /CreationDate, so two runs of this
        # otherwise deterministic script would differ byte for byte. None
        # removes the stamp; raster formats ignore it.
        metadata = {"CreationDate": None, "ModDate": None} if fmt.lower() in {"pdf", "svg"} else None
        fig.savefig(path, dpi=dpi, facecolor=SURFACE, metadata=metadata)
        written.append(path)
    plt.close(fig)
    return written


# --------------------------------------------------------------------------
# configuration-derived scope
# --------------------------------------------------------------------------


def selection_settings(config: dict) -> tuple[int, int]:
    """(rank window the module rule searches, hits drawn per peak).

    The rank window is the Top-N of the plan's rule; the hit cap is a drawing
    limit. Neither is ever derived from a species, a window or a cell type.
    """
    block = config.get("sequence_maps") or {}
    window = int(block.get("top_rank_window", DEFAULT_TOP_RANK_WINDOW))
    max_hits = int(block.get("max_hits_per_peak", DEFAULT_MAX_HITS_PER_PEAK))
    if window < 1:
        raise ValueError("sequence_maps.top_rank_window must be >= 1")
    if max_hits < 1:
        raise ValueError("sequence_maps.max_hits_per_peak must be >= 1")
    return window, max_hits


def retired_n_peaks_per_window(config: dict) -> int | None:
    """The pre-v2 `n_peaks_per_window` knob: recorded, no longer used.

    The v2 rule fixes how many peaks a model contributes (at most one per
    primary module), so this value selects nothing any more. It is reported in
    the summary rather than ignored, so a stale config cannot look as if it were
    still driving the figure.
    """
    block = config.get("sequence_maps") or {}
    value = block.get("n_peaks_per_window")
    return None if value is None else int(value)


def window_lookup(config: dict) -> dict[tuple[str, str], str]:
    """(species, rank_period) -> cross_window, straight from the period table.

    The table is whatever `paths.stage_map` points at, so v1 and v2 resolve
    their own windows without this file naming one. A model whose period is not
    a rank period of the table (the Purkinje/Oligo files carry a single
    all-development period) simply has no entry; that is audited, not guessed.
    """
    table = C.stage_table(config)
    return {
        (str(s), str(p)): str(w)
        for s, p, w in zip(table.species, table.rank_period, table.cross_window)
    }


def window_scope(window: str, config: dict) -> str:
    """cross_species / single_species / unresolved, decided by the table itself."""
    if not window:
        return "unresolved"
    if window in set(C.cross_windows(config)):
        return "cross_species"
    if window in set(C.single_species_windows(config)):
        return "single_species"
    return "unresolved"


def select_models(
    manifest: pd.DataFrame,
    limit_models: int | None,
    model_filter: str | None,
    species_filter: str | None,
) -> tuple[pd.DataFrame, str]:
    """Apply the debug slice to the (already species-filtered) manifest."""
    selected = manifest
    parts: list[str] = []
    if species_filter:
        wanted = [s.strip() for s in species_filter.split(",") if s.strip()]
        selected = selected[selected.species.isin(wanted)]
        parts.append("species=" + ",".join(wanted))
    if model_filter:
        selected = selected[selected.model.str.contains(model_filter, regex=False)]
        parts.append(f"model~{model_filter}")
    if limit_models is not None and len(selected) > limit_models:
        selected = selected.sort_values(["analysis", "species", "period"]).head(limit_models)
        parts.append(f"first={limit_models}")
    return selected.reset_index(drop=True), ";".join(parts)


def scope_directory(step_dir: Path, scope: str) -> Path:
    """Full run writes the frozen names; a debug slice writes to its own subdir."""
    if not scope:
        return step_dir
    token = hashlib.sha1(scope.encode()).hexdigest()[:8]
    path = step_dir / f"scope_{token}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "run_scope.txt").write_text(scope + "\n")
    return path


# --------------------------------------------------------------------------
# the module layer (INTERPRETATION ONLY: it chooses what is illustrated)
# --------------------------------------------------------------------------


def module_celltype_lookup(config: dict) -> dict[str, str]:
    """lower-cased cell-type token -> the module map's own spelling.

    A model carries an analysis label, a target cell type and an Org_Celltype.
    Whichever of the three matches a key of the frozen map (case insensitively)
    identifies the model's cell type, so no cell-type name is spelled out here.
    """
    table = C.celltype_module_table(config)
    return {str(name).lower(): str(name) for name in table.celltype.dropna().unique()}


def model_module_celltype(row: object, lookup: dict[str, str]) -> str:
    for value in (getattr(row, "analysis", ""), getattr(row, "celltype", ""), getattr(row, "org_celltype", "")):
        hit = lookup.get(str(value).lower())
        if hit:
            return hit
    return ""


def module_specs(config: dict, celltype: str) -> tuple[list[dict], list[dict]]:
    """The ordered modules a cell type contributes, plus the audit of the rest.

    The rule uses the map's `primary` modules and the derived unions declared
    under `modules.aggregates`. A module at a level the config calls non-gating
    is never selected, and an aggregate is only kept when every module it unions
    is gating, so this layer can never smuggle a non-gating module into the
    figure. Members are monomeric, matching every other module statistic in the
    project: a module is built from binding models of one factor.
    """
    table = C.celltype_module_table(config)
    sub = table[table.celltype == celltype]
    level_by_module: dict[str, str] = {}
    members_by_module: dict[str, list[str]] = {}
    for module, group in sub.groupby("module", sort=False):
        name = str(module)
        level_by_module[name] = str(group.level.iloc[0])
        monomer = group[group.member_type == "monomer"]
        members_by_module[name] = sorted(monomer.motif_id.dropna().astype(str).unique())

    specs: list[dict] = []
    audit: list[dict] = []
    if not C.module_is_gating(config, PRIMARY_LEVEL):
        # The rule draws primary modules. If a config ever declared that level
        # non-gating, the layer would have to withdraw rather than select it.
        audit.append(
            dict(
                entity_type="celltype",
                entity=celltype,
                reason="primary_level_declared_non_gating",
                detail=f"the config lists the level '{PRIMARY_LEVEL}' as non-gating, so no module of "
                f"{celltype} may be selected; nothing is drawn for this cell type",
                n=0,
            )
        )
        return [], audit
    for module in C.modules_for(config, celltype, PRIMARY_LEVEL):
        members = members_by_module.get(module, [])
        if not members:
            audit.append(
                dict(
                    entity_type="module",
                    entity=f"{celltype}/{module}",
                    reason="module_without_monomer_members",
                    detail=f"module {module} of {celltype} has no monomeric motif in the map; "
                    "it cannot be resolved to peaks and is not drawn",
                    n=0,
                )
            )
            continue
        specs.append(dict(module=module, level=PRIMARY_LEVEL, members=members, kind="map"))

    for name, constituents in C.module_aggregates(config, celltype).items():
        name = str(name)
        constituents = [str(c) for c in constituents]
        missing = [c for c in constituents if c not in level_by_module]
        if missing:
            audit.append(
                dict(
                    entity_type="aggregate",
                    entity=f"{celltype}/{name}",
                    reason="aggregate_names_unknown_module",
                    detail=f"{name} unions {sorted(missing)}, which the module map does not declare "
                    f"for {celltype}; the aggregate is not drawn",
                    n=len(missing),
                )
            )
            continue
        levels = [level_by_module[c] for c in constituents]
        blocked = [c for c, lv in zip(constituents, levels) if not C.module_is_gating(config, lv)]
        if blocked:
            audit.append(
                dict(
                    entity_type="aggregate",
                    entity=f"{celltype}/{name}",
                    reason="aggregate_over_non_gating_module",
                    detail=f"{name} unions {sorted(blocked)}, whose level is declared non-gating; "
                    "the aggregate is not drawn",
                    n=len(blocked),
                )
            )
            continue
        members = sorted({m for c in constituents for m in members_by_module.get(c, [])})
        if not members:
            audit.append(
                dict(
                    entity_type="aggregate",
                    entity=f"{celltype}/{name}",
                    reason="aggregate_without_monomer_members",
                    detail=f"{name} resolves to no monomeric motif for {celltype}",
                    n=0,
                )
            )
            continue
        specs.append(
            dict(module=name, level="union of " + "+".join(sorted(set(levels))), members=members, kind="aggregate")
        )
    return specs, audit


def top_ranked_peaks(
    manifest: pd.DataFrame, rankings: dict[str, pd.DataFrame], window: int
) -> pd.DataFrame:
    """The Top-`window` peaks of every model, in the model's own rank order.

    Rank alone identifies these. No motif, GC or sequence information enters
    here, so the window the module rule searches is fixed before any hit is
    looked at and cannot be steered by the outcome of the enrichment analysis.
    """
    rows = []
    for row in manifest.itertuples():
        frame = rankings.get(row.model)
        if frame is None or frame.empty:
            continue
        for hit in frame.head(window).itertuples():
            rows.append(
                dict(
                    model=row.model,
                    analysis=row.analysis,
                    species=row.species,
                    period=row.period,
                    peak=hit.peak,
                    rank=int(hit.rank),
                    score=float(hit.score),
                )
            )
    return pd.DataFrame(rows, columns=["model", "analysis", "species", "period", "peak", "rank", "score"])


def representative_peaks(
    manifest: pd.DataFrame,
    top: pd.DataFrame,
    motifs_by_peak: dict[str, dict[str, set[str]] | None],
    plans: dict[str, list[dict]],
    lookup: dict[str, str],
    window: int,
    name_by_motif: dict[str, str],
) -> tuple[pd.DataFrame, list[dict]]:
    """The highest-ranked Top-N peak of every primary module, or the negative row.

    Per (model, module) the rule is: the smallest rank inside the window whose
    peak carries at least one motif of the module in the frozen whole-universe
    scan. If no peak inside the window carries one, the pair is recorded as
    NO_QUALIFYING_PEAK_TOP50 and the search stops there. It is never widened to
    a lower rank, because a widened search is a selected example wearing a fixed
    rule's clothes.
    """
    audit: list[dict] = []
    rows: list[dict] = []
    # The window is enforced here, not only by the caller: nothing outside
    # `rank <= window` can be reached, so the rule cannot be widened by accident.
    if not top.empty:
        top = top[top["rank"] <= window]
    top_by_model = {
        str(model): group.sort_values("rank", kind="mergesort")
        for model, group in top.groupby("model", sort=False)
    } if not top.empty else {}

    for row in manifest.itertuples():
        celltype = model_module_celltype(row, lookup)
        if not celltype:
            audit.append(
                dict(
                    entity_type="model",
                    entity=row.model,
                    model=row.model,
                    species=row.species,
                    period=row.period,
                    reason="celltype_absent_from_module_map",
                    detail="no module map entry matches this model's analysis/target celltype; "
                    "with no primary module there is no peak to select",
                    n=0,
                )
            )
            continue
        specs = plans.get(celltype) or []
        if not specs:
            audit.append(
                dict(
                    entity_type="model",
                    entity=row.model,
                    model=row.model,
                    species=row.species,
                    period=row.period,
                    reason="celltype_without_primary_module",
                    detail=f"{celltype} declares no drawable primary module; no peak is selected",
                    n=0,
                )
            )
            continue
        frame = top_by_model.get(str(row.model))
        if frame is None or frame.empty:
            audit.append(
                dict(
                    entity_type="model",
                    entity=row.model,
                    model=row.model,
                    species=row.species,
                    period=row.period,
                    reason="empty_ranking",
                    detail="ranked CSV has no rows; no peak can be drawn",
                    n=0,
                )
            )
            continue
        by_peak = motifs_by_peak.get(row.species)
        if by_peak is None:
            audit.append(
                dict(
                    entity_type="model",
                    entity=row.model,
                    model=row.model,
                    species=row.species,
                    period=row.period,
                    reason="no_frozen_membership_for_species",
                    detail="this species has no frozen whole-universe membership cache (or no genome "
                    "row), so no peak can be shown to carry a module motif; the model is skipped "
                    "rather than guessed",
                    n=0,
                )
            )
            continue

        for spec in specs:
            members = set(spec["members"])
            hit = None
            carried: set[str] = set()
            for candidate in frame.itertuples():
                overlap = by_peak.get(str(candidate.peak), set()) & members
                if overlap:
                    hit, carried = candidate, overlap
                    break
            if hit is None:
                audit.append(
                    dict(
                        entity_type="module",
                        entity=spec["module"],
                        model=row.model,
                        species=row.species,
                        period=row.period,
                        reason="NO_QUALIFYING_PEAK_TOP50",
                        detail=f"no peak in the Top{window} of this model carries a motif of module "
                        f"{spec['module']} in the frozen scan; the search is not widened past the window",
                        n=0,
                    )
                )
                continue
            rows.append(
                dict(
                    model=row.model,
                    analysis=row.analysis,
                    species=row.species,
                    period=row.period,
                    module=spec["module"],
                    module_level=spec["level"],
                    module_n_members=len(members),
                    module_motifs_on_peak=";".join(
                        sorted({name_by_motif.get(m, m) for m in carried})
                    ),
                    module_motifs_on_peak_ids=";".join(sorted(carried)),
                    peak=hit.peak,
                    rank=int(hit.rank),
                    score=float(hit.score),
                )
            )
    columns = [
        "model",
        "analysis",
        "species",
        "period",
        "module",
        "module_level",
        "module_n_members",
        "module_motifs_on_peak",
        "module_motifs_on_peak_ids",
        "peak",
        "rank",
        "score",
    ]
    return pd.DataFrame(rows, columns=columns), audit


def attach_sequences(
    config: dict, peaks: pd.DataFrame, genomes: dict[str, C.SpeciesGenome]
) -> tuple[pd.DataFrame, list[dict]]:
    """Extract the reference sequence of every selected peak, per species."""
    audit: list[dict] = []
    frames = []
    for species, group in peaks.groupby("species", sort=False):
        genome = genomes.get(species)
        if genome is None:
            for row in group.itertuples():
                audit.append(
                    dict(
                        entity_type="model",
                        entity=row.model,
                        model=row.model,
                        species=species,
                        period=row.period,
                        reason="no_genome_in_species_table",
                        detail=f"{species} has no row in config/species_genome.tsv; model not drawn",
                        n=0,
                    )
                )
            continue
        sequences = C.read_sequences(config, genome, group.peak.tolist())
        frames.append(sequences.assign(species=species))
    if not frames:
        return pd.DataFrame(), audit
    # one row per peak: a peak id belongs to exactly one species' coordinate
    # space, so a duplicate would silently multiply every row that joins on it
    sequences = pd.concat(frames, ignore_index=True).drop_duplicates("peak", keep="first")
    return sequences.reset_index(drop=True), audit


# --------------------------------------------------------------------------
# motif positions: the cached scan for membership, FIMO for coordinates
# --------------------------------------------------------------------------

_POSITION_COLUMNS = ["peak", "seq_sha256", "motif_id", "start", "stop", "strand", "score", "pvalue"]


def _fimo_frame(text: str) -> pd.DataFrame:
    """Parse FIMO `--text` output into a typed frame of hits with coordinates."""
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    if not lines:
        return pd.DataFrame(columns=_POSITION_COLUMNS)
    header = lines[0].split("\t")
    rows = [line.split("\t") for line in lines[1:]]
    frame = pd.DataFrame(rows, columns=header)
    for column in ("motif_id", "motif_alt_id", "sequence_name", "strand"):
        if column in frame.columns:
            frame[column] = frame[column].astype(str)
    for column in ("start", "stop"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in ("score", "p-value", "q-value"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.rename(columns={"p-value": "pvalue", "q-value": "qvalue"})


def _run_fimo(config: dict, fasta_path: Path, out_path: Path) -> None:
    threshold = float(config["motif_database"]["fimo_threshold"])
    with open(out_path, "w") as handle:
        subprocess.run(
            [
                C.FIMO_BIN,
                "--text",
                "--thresh",
                f"{threshold:g}",
                str(C.motif_database(config)),
                str(fasta_path),
            ],
            stdout=handle,
            stderr=subprocess.PIPE,
            check=True,
        )


def scan_positions(
    config: dict, genome: C.SpeciesGenome, sequences: pd.DataFrame, force: bool = False
) -> pd.DataFrame:
    """FIMO coordinates for the representative peaks, in a content-keyed cache.

    The frozen whole-universe scan (`common.scan_motifs`) stores motif/peak
    membership only, so the positions of the hits along the sequence have to be
    recovered here. This scans a few dozen sequences with the identical frozen
    threshold into its own cache; it never touches the shared membership cache
    and never re-runs the universe scan.
    """
    threshold = float(config["motif_database"]["fimo_threshold"])
    if abs(threshold - 1e-4) > 1e-12:
        # The membership cache this step cross-checks against is keyed to
        # thresh=1e-4; a different threshold would make the comparison
        # meaningless rather than merely different.
        raise RuntimeError("FIMO cache is keyed to thresh=1e-4; update the cache key if changed")

    usable = sequences[sequences.valid].sort_values("peak").reset_index(drop=True)
    cache = C.cache_dir(config)
    table_path = cache / f"{genome.species}_motif_positions.csv.gz"
    stamp_path = cache / f"{genome.species}_motif_positions.sha256"
    environment = hashlib.sha256(
        "|".join(
            [
                C.sha256_file(C.motif_database(config)),
                f"thresh={threshold!r}",
                f"fimo={C.FIMO_BIN}",
                "naming=seqNNNNNNN",
                "columns=" + ",".join(_POSITION_COLUMNS),
            ]
        ).encode()
    ).hexdigest()

    digest_by_peak = {
        str(row.peak): hashlib.sha256(str(row.sequence).encode()).hexdigest()
        for row in usable.itertuples()
    }

    cached = pd.DataFrame(columns=_POSITION_COLUMNS)
    if table_path.exists() and stamp_path.exists() and stamp_path.read_text() == environment:
        cached = pd.read_csv(table_path, dtype={"motif_id": str})
        for column in _POSITION_COLUMNS:
            if column not in cached.columns:
                cached[column] = pd.Series(dtype=object)
        if force:
            cached = cached.iloc[0:0]

    # A peak counts as covered only when the cache holds a row for it *at this
    # exact sequence*; a peak with no hit keeps a sentinel row (motif_id "") so
    # it is not rescanned on every rerun.
    covered = set()
    if not cached.empty:
        fresh = cached[
            cached.apply(
                lambda row: digest_by_peak.get(str(row.peak)) == str(row.seq_sha256), axis=1
            )
        ]
        covered = set(fresh.peak.astype(str))
        cached = cached[cached.peak.astype(str).isin(covered)]

    todo = usable[~usable.peak.astype(str).isin(covered)].reset_index(drop=True)
    if todo.empty:
        C.log("motif positions cache hit", genome.species, f"{len(cached)} rows")
        return cached.reset_index(drop=True)

    C.log("motif position scan", genome.species, f"{len(todo)} representative peaks")
    fresh_rows: list[pd.DataFrame] = []
    chunk = 200
    workdir = Path(tempfile.mkdtemp(prefix=f"s16_fimo_{genome.species}_"))
    try:
        for start in range(0, len(todo), chunk):
            part = todo.iloc[start: start + chunk].reset_index(drop=True)
            surrogate = [f"seq{index:07d}" for index in range(len(part))]
            fasta_path = workdir / f"chunk_{start:05d}.fa"
            fasta_path.write_text(
                "".join(f">{name}\n{seq}\n" for name, seq in zip(surrogate, part.sequence))
            )
            out_path = workdir / f"chunk_{start:05d}.tsv"
            _run_fimo(config, fasta_path, out_path)
            hits = _fimo_frame(out_path.read_text())
            lookup = dict(zip(surrogate, part.peak.astype(str)))
            if hits.empty:
                hits = pd.DataFrame(columns=_POSITION_COLUMNS)
            else:
                hits = hits.assign(peak=hits.sequence_name.map(lookup))
                if hits.peak.isna().any():
                    unknown = sorted(set(hits.loc[hits.peak.isna(), "sequence_name"]))[:5]
                    raise RuntimeError(
                        f"FIMO returned sequence names outside the submitted set ({unknown}); "
                        "the surrogate-name mapping is broken"
                    )
                hits = hits[["peak", "motif_id", "start", "stop", "strand", "score", "pvalue"]]
            present = set(hits.peak) if not hits.empty else set()
            missing = [p for p in part.peak.astype(str) if p not in present]
            if missing:
                # sentinel row: scanned, no hit above threshold
                hits = pd.concat(
                    [
                        hits,
                        pd.DataFrame(
                            {
                                "peak": missing,
                                "motif_id": "",
                                "start": -1,
                                "stop": -1,
                                "strand": "",
                                "score": np.nan,
                                "pvalue": np.nan,
                            }
                        ),
                    ],
                    ignore_index=True,
                )
            hits["seq_sha256"] = hits.peak.map(digest_by_peak)
            fresh_rows.append(hits[["peak", "seq_sha256", "motif_id", "start", "stop", "strand", "score", "pvalue"]])
            fasta_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    fresh = pd.concat(fresh_rows, ignore_index=True)
    pooled = [fresh] if cached.empty else [cached, fresh]
    combined = (
        pd.concat(pooled, ignore_index=True)
        .drop_duplicates(["peak", "motif_id", "start", "stop", "strand"], keep="last")
        .sort_values(["peak", "motif_id", "start"])
        .reset_index(drop=True)
    )
    table_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(table_path, index=False)
    stamp_path.write_text(environment)
    C.log("wrote", C.display_path(table_path), f"({len(combined)} rows)")
    return combined


def membership_path(config: dict, genome: C.SpeciesGenome) -> Path:
    return C.cache_dir(config) / f"{genome.species}_motif_peak_hits.csv.gz"


def motif_sets_by_peak(
    config: dict, genome: C.SpeciesGenome, peaks: set[str]
) -> dict[str, set[str]] | None:
    """peak -> the motif ids the frozen scan found on it, restricted to `peaks`.

    Read-only: this is the authoritative hit set of the project, and the
    position scan above is checked against it. `None` means the species has no
    membership cache at all, so "this peak carries a motif of this module"
    cannot be decided and the caller must skip the model rather than guess.
    """
    path = membership_path(config, genome)
    if not path.exists():
        return None
    frame = pd.read_csv(path, dtype={"motif_id": str})
    frame = frame[frame.peak.isin(peaks)]
    return {str(peak): set(group.motif_id.astype(str)) for peak, group in frame.groupby("peak", sort=False)}


# --------------------------------------------------------------------------
# hit selection and layout
# --------------------------------------------------------------------------


def cap_hits(
    hits: pd.DataFrame, max_hits: int, required_groups: list[set[str]] | None = None
) -> tuple[pd.DataFrame, int, int]:
    """The `max_hits` strongest hits of a peak, deterministically.

    "Strongest" is the smallest FIMO p-value, ties broken by motif id then by
    position, so the drawn subset is a reproducible function of the scan and not
    of the order FIMO happened to print. The number cut is returned so the figure
    is never mistaken for the complete hit list.

    `required_groups` holds, for each module that chose this peak, the motifs of
    that module the peak carries. One occurrence per group is kept in turn before
    the remaining slots go to the strongest leftover hits, so a track labelled
    with a module shows a motif of that module even when several modules share
    the peak -- otherwise the illustration would hide the very reason the peak
    was selected. The cap still holds, and the number of stronger occurrences
    displaced by the guarantee is returned so it can be audited, not assumed.
    """
    if hits.empty:
        return hits, 0, 0
    ordered = hits.sort_values(["pvalue", "motif_id", "start"], kind="mergesort").reset_index(drop=True)
    groups = [set(group) for group in (required_groups or []) if group]
    if not groups:
        return ordered.head(max_hits), max(0, len(ordered) - max_hits), 0

    motif_of = {int(index): str(value) for index, value in ordered.motif_id.items()}
    kept: set[int] = set()
    while len(kept) < max_hits:
        progressed = False
        for group in groups:
            if len(kept) >= max_hits:
                break
            for index in ordered.index:
                if int(index) not in kept and motif_of[int(index)] in group:
                    kept.add(int(index))
                    progressed = True
                    break
        if not progressed:
            break
    for index in ordered.index:
        if len(kept) >= max_hits:
            break
        kept.add(int(index))
    drawn = ordered.loc[sorted(kept)]
    displaced = len(set(range(min(max_hits, len(ordered)))) - kept)
    return drawn.reset_index(drop=True), max(0, len(ordered) - len(drawn)), displaced


def family_colours(families: list[str]) -> dict[str, str]:
    """A stable family -> colour map.

    Primary families (the ones the rest of the project foregrounds) keep the
    categorical slots in their fixed order; the remaining families follow in
    sorted order; motifs the family map does not name are grouped under
    "Other annotated motifs" and drawn neutral. A family keeps its slot whether
    or not it is drawn in a given model.
    """
    ordered = [f for f in C.primary_families() if f in families]
    ordered += sorted(f for f in families if f not in ordered and f != OTHER_MOTIF_LABEL)
    colours = {}
    for index, family in enumerate(ordered):
        colours[family] = CATEGORICAL[index % len(CATEGORICAL)]
    if OTHER_MOTIF_LABEL in families:
        colours[OTHER_MOTIF_LABEL] = OTHER_MOTIF_COLOUR
    return colours


def pack_lanes(hits: pd.DataFrame) -> tuple[dict[int, int], int]:
    """Greedy first-fit lane assignment so overlapping boxes never hide."""
    lanes: list[int] = []
    assignment: dict[int, int] = {}
    for index, row in hits.sort_values(["start", "stop"], kind="mergesort").iterrows():
        placed = None
        for lane, occupied_until in enumerate(lanes):
            if row.start >= occupied_until:
                placed = lane
                lanes[lane] = int(row.stop)
                break
        if placed is None:
            placed = len(lanes)
            lanes.append(int(row.stop))
        assignment[index] = placed
    return assignment, max(1, len(lanes))


# --------------------------------------------------------------------------
# the figure
# --------------------------------------------------------------------------


def draw_model_map(
    model_row: dict,
    peaks: pd.DataFrame,
    drawn: pd.DataFrame,
    n_lanes: int,
    colours: dict[str, str],
) -> "plt.Figure":
    """One model, `n_peaks` sequence tracks, hits as family-coloured boxes."""
    n_peaks = len(peaks)
    row_height = 1.0
    lane_height = row_height / n_lanes
    height_in = 1.55 + n_peaks * n_lanes * 0.30
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, height_in))

    lengths = peaks.length.to_numpy(dtype=float)
    max_length = float(np.nanmax(lengths)) if len(lengths) else 500.0
    if not np.isfinite(max_length) or max_length <= 0:
        max_length = 500.0

    # bp per point on the printed figure, used only to keep hit labels apart
    data_span = (DATA_RIGHT_FRACTION - DATA_LEFT_FRACTION) * max_length
    axes_width_in = FIG_WIDTH_IN * (PLOT_RIGHT_FRACTION - PLOT_LEFT_FRACTION)
    bp_per_point = data_span / (axes_width_in * 72.0)

    by_peak = {peak: group for peak, group in drawn.groupby("peak", sort=False)} if not drawn.empty else {}

    for index, row in enumerate(peaks.itertuples()):
        base = index * row_height
        length = float(row.length) if isinstance(row.length, (int, float)) else np.nan
        track = length if (np.isfinite(length) and length > 0) else max_length
        ax.add_patch(
            Rectangle(
                (0, base + 0.04),
                track,
                row_height - 0.08,
                facecolor=NEUTRAL_FILL,
                edgecolor=GRID,
                linewidth=0.4,
                zorder=0,
            )
        )
        label = "\n".join(
            [
                f"module {row.module}   rank {row.rank}   score {row.score:.3f}",
                f"{row.peak}",
                f"stage {row.period}   window {row.window or 'n/a'}",
            ]
        )
        ax.text(
            (DATA_LEFT_FRACTION + 0.015) * max_length,
            base + row_height / 2,
            label,
            fontsize=5.4,
            color=INK_SOFT,
            ha="left",
            va="center",
            linespacing=1.35,
        )
        hits = by_peak.get(row.peak)
        if hits is None or hits.empty:
            # A track with no drawn box is either a peak whose sequence could not
            # be extracted, or a peak the frozen scan records a module motif on
            # but whose position scan came back empty (a mismatch, and therefore
            # an audit row). Either way the peak is drawn as it is: it was
            # selected by the fixed rule, and its emptiness is reported, not
            # hidden by swapping in a peak that has hits.
            note = (
                "no sequence extracted from the reference FASTA"
                if not row.valid
                else "position scan found no hit above the frozen FIMO threshold"
            )
            ax.text(
                0.5 * max_length,
                base + row_height / 2,
                note,
                fontsize=5.2,
                color=INK_MUTED,
                ha="center",
                va="center",
                style="italic",
            )
            continue
        assignment, _ = pack_lanes(hits)
        # Two hits can sit side by side in the same lane with only a few bp
        # between them; their labels would then be printed on top of each other.
        # Each (lane, level) keeps a cursor in bp, and a label that would start
        # before the previous one ends moves up (then down) by half a lane.
        label_cursors: dict[tuple[int, int], float] = {}
        for hit_index, hit in hits.iterrows():
            lane = assignment[hit_index]
            y = base + n_lanes * lane_height - (lane + 0.5) * lane_height
            start0 = float(hit.start) - 1.0
            width = max(float(hit.stop) - float(hit.start) + 1.0, 1.0)
            ax.add_patch(
                Rectangle(
                    (start0, y - 0.28 * lane_height),
                    width,
                    0.56 * lane_height,
                    facecolor=colours.get(hit.family, OTHER_MOTIF_COLOUR),
                    edgecolor=INK,
                    linewidth=0.35,
                    zorder=2,
                )
            )
            name = str(hit.motif_name)
            label_x = start0 + width + 0.006 * max_length
            span = HIT_LABEL_FONTSIZE * GLYPH_ASPECT * bp_per_point * max(len(name), 1)
            level, offset = LABEL_LEVELS[0]
            for candidate, candidate_offset in LABEL_LEVELS:
                if label_x >= label_cursors.get((lane, candidate), -np.inf):
                    level, offset = candidate, candidate_offset
                    break
            label_cursors[(lane, level)] = label_x + span
            ax.text(
                label_x,
                y + offset * lane_height,
                name,
                fontsize=HIT_LABEL_FONTSIZE,
                color=INK,
                ha="left",
                va="center",
                zorder=3,
            )

    ax.set_xlim(DATA_LEFT_FRACTION * max_length, DATA_RIGHT_FRACTION * max_length)
    ax.set_ylim(0, n_peaks * row_height)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, max_length + 1, 100))
    ax.set_xlabel(f"position in the {int(max_length)} bp peak (bp, 0-based)")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_bounds(0, max_length)

    drawn_families = [f for f in colours if f in set(drawn.family)] if not drawn.empty else []
    if drawn_families:
        handles = [
            Rectangle((0, 0), 1, 1, facecolor=colours[family], edgecolor=INK, linewidth=0.35)
            for family in drawn_families
        ]
        # Below the sequence tracks: above them the title is, and a legend over
        # a track would hide hits.
        fig.legend(
            handles,
            drawn_families,
            loc="lower left",
            bbox_to_anchor=(0.012, 0.075),
            ncol=min(len(drawn_families), MAX_LEGEND_COLUMNS),
            handlelength=1.0,
            handleheight=0.7,
            columnspacing=1.2,
        )

    window_text = (
        f"   window {model_row['window']} ({model_row['window_scope']})"
        if model_row["window"]
        # the reason a model has no window is the stage table, and it is recorded
        # there and in the audit; on the figure it would push the title off the
        # page, so the figure says only that there is none
        else "   window n/a"
    )
    # The model id already carries the period, so the title names the species,
    # the window and the cell type instead of repeating it -- which also keeps
    # the second line inside the figure for the longest period names.
    title = (
        f"Sequence map (illustrative, supplementary) -- {model_row['model']}\n"
        f"species {model_row['species']}{window_text}"
        f"   cell type {model_row.get('module_celltype', '') or 'n/a'}"
    )
    ax.set_title(title, loc="left", fontsize=7.6, pad=8)
    fig.subplots_adjust(
        left=PLOT_LEFT_FRACTION,
        right=PLOT_RIGHT_FRACTION,
        top=0.87,
        bottom=0.24,
    )
    # The two sentences live on the figure itself, not only in the manifest, so
    # a cropped panel cannot be read as evidence.
    fig.text(
        0.012,
        0.015,
        "\n".join(
            textwrap.wrap(ILLUSTRATION_BOUNDARY, width=155)
            + textwrap.wrap(SUPPLEMENTARY_NOTE, width=155)
        ),
        fontsize=5.2,
        color=INK_SOFT,
    )
    return fig


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def build_hit_table(
    config: dict,
    peaks: pd.DataFrame,
    sequences: pd.DataFrame,
    genomes: dict[str, C.SpeciesGenome],
    family_table: pd.DataFrame,
    catalog: pd.DataFrame,
    motifs_by_peak: dict[str, dict[str, set[str]] | None],
    max_hits: int,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """One row per drawn hit, with its family, position and label."""
    name_by_motif = dict(zip(catalog.motif_id, catalog.motif_name))
    family_by_motif = {
        str(row.motif_id): (str(row.family) if isinstance(row.family, str) and row.family else OTHER_MOTIF_LABEL)
        for row in family_table.itertuples()
    }

    merged = peaks.merge(
        sequences[["peak", "chrom", "start", "end", "length", "gc", "valid"]],
        on="peak",
        how="left",
    )

    hit_rows: list[dict] = []
    audit: list[dict] = []
    # FIMO reports every occurrence of a motif on the sequence, so a peak has
    # more hit occurrences than it has distinct motifs hit. The cap applies to
    # occurrences -- that is how many boxes a track can carry -- and both counts
    # are reported so the two units cannot be confused.
    occurrences: dict[str, int] = {}
    for species, group in merged.groupby("species", sort=False):
        genome = genomes.get(species)
        if genome is None:
            continue
        positions = scan_positions(config, genome, sequences[sequences.peak.isin(set(group.peak))], force=force)
        by_peak = motifs_by_peak.get(species)
        hits = positions[positions.motif_id.astype(str) != ""].copy()
        if not hits.empty:
            hits["motif_name"] = hits.motif_id.map(name_by_motif).fillna(hits.motif_id)
            hits["family"] = hits.motif_id.map(family_by_motif).fillna(OTHER_MOTIF_LABEL)

        # One pass per peak. The membership cross-check, the per-peak totals, the
        # cap and the drawn hit rows are properties of the peak, not of the module
        # that chose it, so a peak chosen for two modules is processed once and
        # the hit table stays 1:1 on (model, peak) instead of repeating every hit
        # once per module.
        carrier_groups: dict[tuple[str, str], list[set[str]]] = {}
        for row in group.itertuples():
            carriers = {
                motif
                for motif in str(getattr(row, "module_motifs_on_peak_ids", "")).split(";")
                if motif
            }
            if carriers:
                carrier_groups.setdefault((str(row.model), str(row.peak)), []).append(carriers)
        drawn_by_key: dict[tuple[str, str], pd.DataFrame] = {}
        for row in group.drop_duplicates(["model", "peak"]).itertuples():
            peak_hits = hits[hits.peak.astype(str) == row.peak]
            observed = set(peak_hits.motif_id.astype(str))
            if by_peak is None:
                expected: set[str] = set()
                audit.append(
                    dict(
                        entity_type="peak",
                        entity=row.peak,
                        model=row.model,
                        species=row.species,
                        period=row.period,
                        reason="no_frozen_membership_for_species",
                        detail="this species has no frozen whole-universe membership cache; "
                        "positions are taken from this step's own FIMO scan only",
                        n=len(observed),
                    )
                )
            else:
                expected = by_peak.get(str(row.peak), set())
            if by_peak is not None and observed != expected:
                audit.append(
                    dict(
                        entity_type="peak",
                        entity=row.peak,
                        model=row.model,
                        species=row.species,
                        period=row.period,
                        reason="membership_mismatch",
                        detail="position scan disagrees with the frozen whole-universe scan; "
                        f"position_only={len(observed - expected)} "
                        f"membership_only={len(expected - observed)}",
                        n=len(observed ^ expected),
                    )
                )
            if not row.valid:
                audit.append(
                    dict(
                        entity_type="peak",
                        entity=row.peak,
                        model=row.model,
                        species=row.species,
                        period=row.period,
                        reason="no_sequence",
                        detail="sequence could not be extracted from the reference FASTA; "
                        "kept and drawn as an empty track",
                        n=0,
                    )
                )
            occurrences[str(row.peak)] = len(peak_hits)
            required = carrier_groups.get((str(row.model), str(row.peak)), [])
            drawn, cut, displaced = cap_hits(peak_hits, max_hits, required)
            drawn_by_key[(str(row.model), str(row.peak))] = drawn
            if displaced:
                audit.append(
                    dict(
                        entity_type="peak",
                        entity=row.peak,
                        model=row.model,
                        species=row.species,
                        period=row.period,
                        reason="cap_kept_module_motif",
                        detail=f"{displaced} stronger hit occurrence(s) were displaced so the track "
                        f"can show a motif of the module(s) that chose this peak; the drawn subset is "
                        f"still capped at sequence_maps.max_hits_per_peak={max_hits}",
                        n=displaced,
                    )
                )
            if cut:
                audit.append(
                    dict(
                        entity_type="peak",
                        entity=row.peak,
                        model=row.model,
                        species=row.species,
                        period=row.period,
                        reason="hits_beyond_cap",
                        detail=f"{cut} further hit occurrence(s) above the threshold not drawn "
                        f"(cap sequence_maps.max_hits_per_peak={max_hits}); "
                        f"{len(observed)} distinct motif(s) hit this peak",
                        n=cut,
                    )
                )
            if len(drawn) == 0:
                audit.append(
                    dict(
                        entity_type="peak",
                        entity=row.peak,
                        model=row.model,
                        species=row.species,
                        period=row.period,
                        reason="no_motif_hit",
                        detail="the frozen scan records a module motif on this peak, but this step's "
                        "position scan found none above the threshold; the track is drawn empty "
                        "rather than swapped for a peak that has hits",
                        n=0,
                    )
                )
            for hit in drawn.itertuples():
                hit_rows.append(
                    dict(
                        model=row.model,
                        analysis=row.analysis,
                        species=row.species,
                        period=row.period,
                        window=row.window,
                        peak=row.peak,
                        rank=row.rank,
                        motif_id=str(hit.motif_id),
                        motif_name=str(hit.motif_name),
                        family=str(hit.family),
                        start=int(hit.start),
                        stop=int(hit.stop),
                        strand=str(hit.strand),
                        fimo_score=float(hit.score) if np.isfinite(hit.score) else np.nan,
                        fimo_pvalue=float(hit.pvalue) if np.isfinite(hit.pvalue) else np.nan,
                    )
                )

        # A second pass per module row: the peak was chosen because it carries a
        # motif of the module, so it is a caveat worth naming when the per-peak
        # cap leaves none of those motifs among the drawn boxes.
        for row in group.itertuples():
            drawn = drawn_by_key.get((str(row.model), str(row.peak)), pd.DataFrame())
            carriers = {
                motif
                for motif in str(getattr(row, "module_motifs_on_peak_ids", "")).split(";")
                if motif
            }
            drawn_motifs = set(drawn.motif_id.astype(str)) if not drawn.empty else set()
            if carriers and not (carriers & drawn_motifs):
                audit.append(
                    dict(
                        entity_type="module",
                        entity=row.module,
                        model=row.model,
                        species=row.species,
                        period=row.period,
                        reason="module_carrier_not_drawn",
                        detail=f"the peak was selected because it carries {len(carriers)} motif(s) of "
                        f"module {row.module}, but the per-peak cap "
                        f"(sequence_maps.max_hits_per_peak={max_hits}) left none of them among the "
                        f"{len(drawn)} drawn hit(s); the track does not show the module's own motif",
                        n=len(carriers),
                    )
                )

    columns = [
        "model",
        "analysis",
        "species",
        "period",
        "window",
        "peak",
        "rank",
        "motif_id",
        "motif_name",
        "family",
        "start",
        "stop",
        "strand",
        "fimo_score",
        "fimo_pvalue",
    ]
    hit_frame = pd.DataFrame(hit_rows, columns=columns)
    if not hit_frame.empty:
        hit_frame = hit_frame.sort_values(
            ["model", "rank", "start", "motif_id"], kind="mergesort"
        ).reset_index(drop=True)
    merged = merged.assign(
        n_hit_occurrences=[occurrences.get(p, 0) for p in merged.peak]
    )
    return hit_frame, merged, audit


def summarise(peaks: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    """One representative_peaks row per drawn peak, with the hits folded in.

    A peak can be the top peak of more than one model (the same locus ranked
    highly in two developmental periods), so the drawn hits are selected by
    (model, peak) and never by peak alone.
    """
    rows = []
    for row in peaks.itertuples():
        peak_hits = (
            hits[(hits.model == row.model) & (hits.peak.astype(str) == row.peak)]
            if not hits.empty
            else hits
        )
        families_drawn = sorted(set(peak_hits.family.astype(str))) if not peak_hits.empty else []
        rows.append(
            dict(
                model=row.model,
                analysis=row.analysis,
                species=row.species,
                period=row.period,
                window=row.window,
                window_scope=row.window_scope,
                module=row.module,
                module_level=row.module_level,
                module_n_members=row.module_n_members,
                module_motifs_on_peak=row.module_motifs_on_peak,
                peak=row.peak,
                rank=row.rank,
                score=row.score,
                chrom=row.chrom,
                start=row.start,
                end=row.end,
                length=row.length,
                gc=row.gc,
                valid=row.valid,
                n_hits=row.n_hits,
                n_hit_occurrences=row.n_hit_occurrences,
                n_hits_drawn=len(peak_hits),
                n_motifs_drawn=peak_hits.motif_id.nunique() if not peak_hits.empty else 0,
                families_hit=";".join(families_drawn),
                families_hit_all=";".join(sorted(set(str(row.families_hit_all).split(";")) - {""})),
                hits_drawn=";".join(
                    f"{h.motif_name}:{h.family}:{h.start}-{h.stop}({h.strand})"
                    for h in peak_hits.itertuples()
                ),
            )
        )
    frame = pd.DataFrame(rows, columns=REPRESENTATIVE_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(["model", "rank", "module"], kind="mergesort").reset_index(drop=True)
    return frame


def frozen_results_root() -> Path:
    """The published results root. No run of this step may write it."""
    return (C.ROOT / "results").resolve()


def resolved_step_dir(config: dict) -> Path:
    """Where this step would write, resolved WITHOUT creating anything.

    `C.results_dir` mkdirs as it resolves, so the guard has to look at the same
    path without that side effect; only after it has passed does the real call
    run.
    """
    return (C.config_path(config, "results") / STEP).resolve()


def refuse_v1_results_tree(config: dict, allow_v1_results: bool = False) -> None:
    """Refuse to write the published results tree unless explicitly allowed.

    That tree has to stay byte-for-byte frozen, and the guard cannot live in the
    config because the config is itself part of the frozen tree, so it lives
    here. `--results-root` redirects the write to a scratch root;
    `--allow-v1-results` is the deliberate opt-in for the case where the frozen
    tree really is the target.
    """
    target = resolved_step_dir(config)
    frozen_root = frozen_results_root()
    if (target == frozen_root or frozen_root in target.parents) and not allow_v1_results:
        raise RuntimeError(
            f"refusing to write the frozen results tree: {target}\n"
            "  rerun with --results-root <scratch-dir> to redirect the output,\n"
            "  or use --config config/analysis_v2.yaml,\n"
            "  or pass --allow-v1-results if the frozen tree is really the target"
        )


SKIP_STATUS = {
    "celltype_absent_from_module_map": "skipped_no_module_map",
    "celltype_without_primary_module": "skipped_no_primary_module",
    "empty_ranking": "skipped_empty_ranking",
    "no_frozen_membership_for_species": "skipped_no_membership",
}

EMPTY_HIT_COLUMNS = [
    "model",
    "analysis",
    "species",
    "period",
    "window",
    "peak",
    "rank",
    "motif_id",
    "motif_name",
    "family",
    "start",
    "stop",
    "strand",
    "fimo_score",
    "fimo_pvalue",
]


def run(
    config: dict,
    limit_models: int | None = None,
    model_filter: str | None = None,
    species_filter: str | None = None,
    force: bool = False,
    allow_v1_results: bool = False,
) -> dict:
    with C.StepTimer("s18 sequence maps"):
        refuse_v1_results_tree(config, allow_v1_results)
        rank_window, max_hits = selection_settings(config)
        retired = retired_n_peaks_per_window(config)
        step_dir = C.results_dir(config, STEP)

        manifest_all = C.ranking_files(config)
        kept, dropped = C.apply_species_filter(config, manifest_all)
        selected, scope = select_models(kept, limit_models, model_filter, species_filter)
        out_dir = scope_directory(step_dir, scope)

        _, rankings = C.load_rankings(config)
        catalog = C.motif_catalog(config)
        name_by_motif = dict(zip(catalog.motif_id, catalog.motif_name))
        family_table = C.tf_family_table()
        genomes = C.species_table(config)
        windows = window_lookup(config)

        audit_rows: list[dict] = []
        for row in dropped.itertuples():
            record = row._asdict() if hasattr(row, "_asdict") else {}
            audit_rows.append(
                dict(
                    entity_type="model",
                    entity=str(record.get("model", "")),
                    model=str(record.get("model", "")),
                    species=str(record.get("species", "")),
                    period="",
                    reason="eligibility_filter",
                    detail=str(record.get("action", "dropped by the eligibility filter")),
                    n=int(record.get("n_cells", 0)),
                )
            )
        for row in selected.itertuples():
            if not windows.get((row.species, row.period)):
                audit_rows.append(
                    dict(
                        entity_type="model",
                        entity=row.model,
                        model=row.model,
                        species=row.species,
                        period=row.period,
                        reason="period_not_in_stage_table",
                        detail="the model period is not a rank_period of this run's stage map, "
                        "so it has no cross_window; the model is still drawn",
                        n=0,
                    )
                )
            n_ranked = int(getattr(row, "n_ranked", 0))
            if 0 < n_ranked < rank_window:
                audit_rows.append(
                    dict(
                        entity_type="model",
                        entity=row.model,
                        model=row.model,
                        species=row.species,
                        period=row.period,
                        reason="ranking_shorter_than_window",
                        detail=f"the ranking holds {n_ranked} peak(s), fewer than the Top{rank_window} "
                        "window; every ranked peak was searched and none was added",
                        n=n_ranked,
                    )
                )

        # the modules each cell type may draw, resolved once from the frozen map
        lookup = module_celltype_lookup(config)
        plans: dict[str, list[dict]] = {}
        for celltype in sorted(set(lookup.values())):
            specs, notes = module_specs(config, celltype)
            plans[celltype] = specs
            for note in notes:
                audit_rows.append(dict(model="", species="", period="", **note))

        # stage 1: the rank window. No motif information enters here.
        top = top_ranked_peaks(selected, rankings, rank_window)

        # stage 2: the frozen whole-universe scan, restricted to the peaks that
        # can actually be chosen. It is read once per species and is what decides
        # "this peak carries a motif of this module".
        wanted_by_species: dict[str, set[str]] = {}
        for row in top.itertuples():
            wanted_by_species.setdefault(str(row.species), set()).add(str(row.peak))
        motifs_by_peak = {
            species: (motif_sets_by_peak(config, genomes[species], peaks) if species in genomes else None)
            for species, peaks in wanted_by_species.items()
        }
        # stage 3: the module rule, capped at the window.
        chosen, selection_audit = representative_peaks(
            selected, top, motifs_by_peak, plans, lookup, rank_window, name_by_motif
        )
        audit_rows.extend(selection_audit)
        model_skip: dict[str, str] = {}
        for entry in selection_audit:
            mapped = SKIP_STATUS.get(str(entry.get("reason", "")))
            if mapped and entry.get("model"):
                model_skip.setdefault(str(entry["model"]), mapped)

        if chosen.empty:
            merged = chosen.copy()
            hit_frame = pd.DataFrame(columns=EMPTY_HIT_COLUMNS)
        else:
            chosen["window"] = [
                windows.get((s, p), "") for s, p in zip(chosen.species, chosen.period)
            ]
            chosen["window_scope"] = [window_scope(w, config) for w in chosen.window]

            sequences, sequence_audit = attach_sequences(config, chosen, genomes)
            audit_rows.extend(sequence_audit)
            if sequences.empty:
                raise RuntimeError("no reference sequence could be extracted for any selected peak")

            hit_frame, merged, hit_audit = build_hit_table(
                config,
                chosen,
                sequences,
                genomes,
                family_table,
                catalog,
                motifs_by_peak,
                max_hits,
                force,
            )
            audit_rows.extend(hit_audit)

            # total hits and the families of those hits, straight from the frozen
            # membership cache, so the cap is reported against the real hit list
            family_by_motif_all = {
                str(row.motif_id): (
                    str(row.family) if isinstance(row.family, str) and row.family else OTHER_MOTIF_LABEL
                )
                for row in family_table.itertuples()
            }
            totals: dict[str, int] = {}
            family_all: dict[str, str] = {}
            for species, group in merged.groupby("species", sort=False):
                by_peak = motifs_by_peak.get(str(species))
                if by_peak is None:
                    continue
                for peak in group.peak:
                    motifs = by_peak.get(str(peak), set())
                    totals[str(peak)] = len(motifs)
                    family_all[str(peak)] = ";".join(
                        sorted({family_by_motif_all.get(m, OTHER_MOTIF_LABEL) for m in motifs})
                    )
            merged = merged.assign(
                n_hits=[totals.get(str(p), 0) for p in merged.peak],
                families_hit_all=[family_all.get(str(p), "") for p in merged.peak],
            )

        representative = summarise(merged, hit_frame) if not merged.empty else pd.DataFrame(
            columns=REPRESENTATIVE_COLUMNS
        )

        families = sorted(
            set(family_table.family.fillna("").astype(str)) - {""}
        ) + [OTHER_MOTIF_LABEL]
        colours = family_colours(families)

        plotting = config.get("plotting", {})
        formats = [str(fmt) for fmt in plotting.get("formats", ["png", "pdf"])]
        dpi = int(plotting.get("dpi", 200))
        apply_style(dpi)

        manifest_rows: list[dict] = []
        for row in selected.itertuples():
            peaks_here = representative[representative.model == row.model]
            hits_here = hit_frame[hit_frame.model == row.model] if not hit_frame.empty else hit_frame
            window = windows.get((row.species, row.period), "")
            celltype = model_module_celltype(row, lookup)
            spec_names = [str(spec["module"]) for spec in plans.get(celltype, [])]
            drawn_modules = list(dict.fromkeys(peaks_here.module.astype(str))) if not peaks_here.empty else []
            missing_modules = [m for m in spec_names if m not in set(drawn_modules)]
            status = model_skip.get(str(row.model), "")
            figure_png = figure_pdf = ""
            if not status:
                if peaks_here.empty:
                    status = "no_qualifying_peak_in_window"
                else:
                    status = "rendered"
            if status == "rendered":
                row_dict = dict(
                    model=row.model,
                    species=row.species,
                    period=row.period,
                    window=window,
                    window_scope=window_scope(window, config),
                    module_celltype=celltype,
                )
                merged_here = merged[merged.model == row.model].reset_index(drop=True)
                drawn = hits_here
                lanes = 1
                if not drawn.empty:
                    for _, group in drawn.groupby("peak"):
                        _, n_lanes = pack_lanes(group)
                        lanes = max(lanes, n_lanes)
                fig = draw_model_map(row_dict, merged_here, drawn, lanes, colours)
                stem = "sequence_map_" + row.model.replace(os.sep, "_")
                written = save_figure(fig, out_dir, stem, formats, dpi)
                by_ext = {path.suffix.lstrip("."): C.display_path(path) for path in written}
                figure_png = by_ext.get("png", "")
                figure_pdf = by_ext.get("pdf", "")
                if hits_here.empty:
                    status = "rendered_no_hits"
                elif (peaks_here.n_hits_drawn == 0).any():
                    status = "rendered_partial_hits"
            manifest_rows.append(
                dict(
                    model=row.model,
                    analysis=row.analysis,
                    species=row.species,
                    period=row.period,
                    window=window,
                    window_scope=window_scope(window, config),
                    module_celltype=celltype,
                    n_modules=len(spec_names),
                    n_modules_drawn=len(drawn_modules),
                    n_modules_without_qualifying_peak=len(missing_modules),
                    modules_without_qualifying_peak=";".join(missing_modules),
                    n_peaks=len(peaks_here),
                    n_peaks_with_hits=int((peaks_here.n_hits_drawn > 0).sum()) if not peaks_here.empty else 0,
                    n_peaks_without_hits=int((peaks_here.n_hits_drawn == 0).sum()) if not peaks_here.empty else 0,
                    n_peaks_without_sequence=int((~peaks_here.valid.astype(bool)).sum())
                    if not peaks_here.empty
                    else 0,
                    n_hits_distinct=int(peaks_here.n_hits.sum()) if not peaks_here.empty else 0,
                    n_hit_occurrences=int(peaks_here.n_hit_occurrences.sum())
                    if not peaks_here.empty
                    else 0,
                    n_hits_drawn=int(peaks_here.n_hits_drawn.sum()) if not peaks_here.empty else 0,
                    status=status,
                    figure_png=figure_png,
                    figure_pdf=figure_pdf,
                    figure_caption=figure_caption(str(row.model)),
                    illustration_boundary=ILLUSTRATION_BOUNDARY,
                    supplementary_note=SUPPLEMENTARY_NOTE,
                )
            )

        drawn_families = sorted(set(hit_frame.family)) if not hit_frame.empty else []
        if OTHER_MOTIF_LABEL in drawn_families:
            audit_rows.append(
                dict(
                    entity_type="motif",
                    entity=OTHER_MOTIF_LABEL,
                    model="",
                    species="",
                    period="",
                    reason="family_other_annotated_motifs",
                    detail="drawn hit(s) whose motif is not named by config/tf_family_map.tsv; "
                    "they are grouped as 'Other annotated motifs' and drawn in the neutral colour",
                    n=int((hit_frame.family == OTHER_MOTIF_LABEL).sum()),
                )
            )

        manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
        audit = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
        if not audit.empty:
            # a peak drawn for two modules is one peak with one caveat, not two
            audit = (
                audit.drop_duplicates()
                .sort_values(["entity_type", "model", "entity", "reason"], kind="mergesort")
                .reset_index(drop=True)
            )

        C.save_table(representative, out_dir / "representative_peaks.tsv", sep="\t")
        C.save_table(manifest, out_dir / "sequence_maps_manifest.tsv", sep="\t")
        C.save_table(audit, out_dir / "sequence_maps_audit.tsv", sep="\t")

        no_qualifying = audit[audit.reason == "NO_QUALIFYING_PEAK_TOP50"] if not audit.empty else audit
        summary = dict(
            step=STEP,
            config=str(config.get("_path", "")),
            stage_map=C.display_path(C.stage_map_path(config)),
            scope=scope,
            selection_rule=(
                f"per primary module, the highest-ranked peak within Top{rank_window} that carries at "
                "least one motif of the module in the frozen scan; NO_QUALIFYING_PEAK_TOP50 and no "
                "search past the window when none does"
            ),
            top_rank_window=int(rank_window),
            top_rank_window_source=(
                "config.sequence_maps.top_rank_window"
                if (config.get("sequence_maps") or {}).get("top_rank_window") is not None
                else f"default ({DEFAULT_TOP_RANK_WINDOW})"
            ),
            n_peaks_per_window_config=retired,
            n_peaks_per_window_effect=(
                "retired: the module rule fixes the number of peaks per model (at most one per "
                "primary module), so this value no longer selects anything"
                if retired is not None
                else "not declared"
            ),
            illustration_boundary=ILLUSTRATION_BOUNDARY,
            supplementary_note=SUPPLEMENTARY_NOTE,
            cross_windows=list(C.cross_windows(config)),
            single_species_windows=list(C.single_species_windows(config)),
            n_models_in_manifest=int(len(manifest_all)),
            n_models_after_species_filter=int(len(kept)),
            n_models_selected=int(len(selected)),
            dropped_models=sorted(str(m) for m in dropped.model) if len(dropped) else [],
            module_celltypes=sorted(set(lookup.values())),
            modules_by_celltype={ct: [spec["module"] for spec in specs] for ct, specs in plans.items()},
            n_models_rendered=int((manifest.status.str.startswith("rendered")).sum()) if not manifest.empty else 0,
            n_models_skipped=int((~manifest.status.str.startswith("rendered")).sum()) if not manifest.empty else 0,
            model_status={str(r.model): str(r.status) for r in manifest.itertuples()},
            n_modules_resolved=int(len(representative)),
            n_distinct_modules_drawn=int(representative.module.nunique()) if not representative.empty else 0,
            n_modules_without_qualifying_peak=int(len(no_qualifying)),
            modules_without_qualifying_peak=[
                f"{r.model}/{r.entity}" for r in no_qualifying.itertuples()
            ],
            n_representative_peaks=int(len(representative)),
            n_peaks_with_hits=int((representative.n_hits_drawn > 0).sum()) if not representative.empty else 0,
            n_peaks_without_hits=int((representative.n_hits_drawn == 0).sum()) if not representative.empty else 0,
            n_hits_drawn=int(len(hit_frame)),
            max_hits_per_peak=int(max_hits),
            family_colors=colours,
            audit_rows=int(len(audit)),
            outputs=C.display_path(out_dir),
        )
        C.save_json(summary, out_dir / "sequence_maps_summary.json")
        C.log(
            "summary",
            {
                k: summary[k]
                for k in ("n_models_selected", "n_models_rendered", "n_modules_without_qualifying_peak", "n_hits_drawn")
            },
        )
        return summary


# --------------------------------------------------------------------------
# self test
# --------------------------------------------------------------------------


def _unit_checks(scratch: Path) -> list[str]:
    """Component checks for the paths the real data does not exercise.

    The representative peaks of this dataset all have a sequence and all carry
    hits, so "no sequence" and "no hit" would otherwise never be drawn and could
    rot unnoticed. These build the frames by hand and call the real functions.
    """
    checks: list[str] = []
    v2 = C.load_config(C.CONFIG_DIR / "analysis_v2.yaml")
    declared = v2.get("sequence_maps") or {}
    assert selection_settings(v2) == (
        int(declared.get("top_rank_window", DEFAULT_TOP_RANK_WINDOW)),
        int(declared.get("max_hits_per_peak", DEFAULT_MAX_HITS_PER_PEAK)),
    )
    assert selection_settings({}) == (DEFAULT_TOP_RANK_WINDOW, DEFAULT_MAX_HITS_PER_PEAK)
    assert retired_n_peaks_per_window(v2) == int(declared["n_peaks_per_window"])
    assert retired_n_peaks_per_window({}) is None
    checks.append(
        "selection_settings: the rank window and the hit cap come from the config; the retired "
        "n_peaks_per_window is still recorded"
    )

    hits = pd.DataFrame(
        {
            "motif_id": ["MA1", "MA2", "MA3"],
            "motif_name": ["A", "B", "C"],
            "family": ["RFX", "RFX", "SOX"],
            "start": [10, 12, 40],
            "stop": [20, 22, 50],
            "pvalue": [1e-5, 1e-6, 1e-7],
            "strand": ["+", "+", "-"],
        }
    )
    capped, cut, displaced = cap_hits(hits, 2)
    assert cut == 1 and list(capped.motif_id) == ["MA3", "MA2"], "cap must keep the strongest hits"
    assert displaced == 0
    capped2, cut2, displaced2 = cap_hits(hits, 2, [{"MA1"}])
    assert set(capped2.motif_id) == {"MA3", "MA1"}, "a module motif must survive the cap"
    assert (cut2, displaced2) == (1, 1), "the displacement must be reported, not hidden"
    # two modules sharing a peak: each keeps a slot while the cap holds
    capped3, cut3, displaced3 = cap_hits(hits, 2, [{"MA1"}, {"MA2"}])
    assert set(capped3.motif_id) == {"MA1", "MA2"}, "a shared peak must show both modules"
    assert (cut3, displaced3) == (1, 1)
    _, lanes = pack_lanes(hits)
    assert lanes == 2, "overlapping hits must be laid out in separate lanes"
    checks.append(
        "cap_hits/pack_lanes: strongest-first cap that always keeps a module motif, and reports it; "
        "overlapping boxes never hidden"
    )

    colours = family_colours([OTHER_MOTIF_LABEL, "SOX", "RFX"])
    assert len({colours["RFX"], colours["SOX"], colours[OTHER_MOTIF_LABEL]}) == 3
    assert OTHER_MOTIF_LABEL == "Other annotated motifs"
    checks.append("family_colours: distinct colours, the unnamed-motif bucket kept neutral")

    # the module layer: no non-gating module may enter the selection
    map_table = C.celltype_module_table(v2)
    celltype = str(map_table.celltype.iloc[0])
    sub = map_table[map_table.celltype == celltype]
    blocked_modules = {
        str(module)
        for module, level in zip(sub.module, sub.level)
        if not C.module_is_gating(v2, str(level))
    }
    specs, _ = module_specs(v2, celltype)
    assert specs, "the frozen map must declare at least one drawable primary module"
    assert not ({spec["module"] for spec in specs if spec["kind"] == "map"} & blocked_modules)
    declared_modules = set(C.modules_for(v2, celltype, PRIMARY_LEVEL)) | set(
        C.module_aggregates(v2, celltype)
    )
    assert {spec["module"] for spec in specs} <= declared_modules
    # a config that ever declared the selection level non-gating must withdraw
    blocked_specs, blocked_audit = module_specs({"modules": {"non_gating_levels": [PRIMARY_LEVEL]}}, celltype)
    assert blocked_specs == [], "a non-gating level must never be selected"
    assert [entry["reason"] for entry in blocked_audit] == ["primary_level_declared_non_gating"]
    checks.append(
        "module_specs: only gating primary modules and declared aggregates are drawable"
    )

    # the selection rule itself, on hand-built frames inside and outside the window
    members = sorted(specs[0]["members"])
    lookup = {celltype.lower(): celltype}
    manifest = pd.DataFrame(
        [
            dict(
                model="M",
                analysis="A",
                species="s",
                period="p",
                celltype=celltype,
                org_celltype=celltype,
            )
        ]
    )
    top = pd.DataFrame(
        [
            dict(model="M", analysis="A", species="s", period="p", peak=f"pk{rank}", rank=rank, score=1.0)
            for rank in (1, 2, 3)
        ]
    )
    # pk1 (rank 1) and pk2 (rank 2) carry nothing; pk3 (rank 3) carries a motif of
    # the module, so the rule must take rank 3 and never fall back to rank 1
    motifs = {"s": {"pk3": {members[0]}}}
    chosen, audit_rows = representative_peaks(
        manifest, top, motifs, {celltype: [specs[0]]}, lookup, 3, {}
    )
    assert list(chosen.peak) == ["pk3"], "the rule must take the highest-ranked qualifying peak"
    assert not audit_rows
    # with the window at Top2 the same model has no qualifying peak: the negative
    # case is reported and the search is NOT widened to rank 3
    chosen2, audit2 = representative_peaks(
        manifest, top, motifs, {celltype: [specs[0]]}, lookup, 2, {}
    )
    assert chosen2.empty, "the rule must not search past the window"
    assert [entry["reason"] for entry in audit2] == ["NO_QUALIFYING_PEAK_TOP50"]
    checks.append(
        "representative_peaks: highest-ranked qualifying peak inside the window, "
        "NO_QUALIFYING_PEAK_TOP50 outside it"
    )

    # one peak that is the chosen peak of two models must report each model's own
    # drawn hits, not the union of both
    def demo_peak(model: str, period: str, peak: str, **extra: object) -> dict:
        row = dict(
            model=model,
            analysis="A",
            species="s",
            period=period,
            window="w",
            window_scope="cross_species",
            module="MOD",
            module_level=PRIMARY_LEVEL,
            module_n_members=1,
            module_motifs_on_peak="A",
            peak=peak,
            rank=1,
            score=0.5,
            chrom="chr1",
            start=1,
            end=500,
            length=500.0,
            gc=0.5,
            valid=True,
            n_hits=2,
            n_hit_occurrences=2,
            families_hit_all="RFX",
        )
        row.update(extra)
        return row

    peaks = pd.DataFrame(
        [
            demo_peak("A__s__p1", "p1", "pk:1-500"),
            demo_peak("A__s__p2", "p2", "pk:1-500", score=0.9),
        ]
    )
    drawn = pd.DataFrame(
        [
            dict(model="A__s__p1", peak="pk:1-500", rank=1, motif_id="MA1", motif_name="A",
                 family="RFX", start=1, stop=10, strand="+", fimo_score=1.0, fimo_pvalue=1e-5),
            dict(model="A__s__p2", peak="pk:1-500", rank=1, motif_id="MA1", motif_name="A",
                 family="RFX", start=1, stop=10, strand="+", fimo_score=1.0, fimo_pvalue=1e-5),
            dict(model="A__s__p2", peak="pk:1-500", rank=1, motif_id="MA2", motif_name="B",
                 family="SOX", start=100, stop=110, strand="-", fimo_score=1.0, fimo_pvalue=1e-6),
        ]
    )
    table = summarise(peaks, drawn)
    assert list(table.n_hits_drawn) == [1, 2], "a peak draws only its own model's hits"
    assert list(table.module) == ["MOD", "MOD"], "the representative table names the module"
    checks.append("summarise: a shared peak reports each model's own hits and its module")

    # the two empty-track paths, drawn from hand-built frames
    demo = pd.DataFrame(
        [
            demo_peak("M", "p", "pk1:1-500", rank=1, n_hits=1, n_hit_occurrences=1),
            demo_peak("M", "p", "pk2:1-500", rank=2, n_hits=0, n_hit_occurrences=0),
            demo_peak("M", "p", "pk3:1-500", rank=3, length=np.nan, valid=False,
                      n_hits=0, n_hit_occurrences=0),
        ]
    )
    demo_hits = pd.DataFrame(
        [
            dict(model="M", peak="pk1:1-500", rank=1, motif_id="MA1", motif_name="A",
                 family="RFX", start=10, stop=20, strand="+", fimo_score=1.0, fimo_pvalue=1e-5),
            dict(model="M", peak="pk1:1-500", rank=1, motif_id="MA2", motif_name="B",
                 family="SOX", start=18, stop=28, strand="-", fimo_score=1.0, fimo_pvalue=1e-6),
        ]
    )
    row = dict(
        model="M", species="s", period="p", window="w",
        window_scope="cross_species", module_celltype=celltype,
    )
    fig = draw_model_map(
        row, demo, demo_hits, 2, family_colours([OTHER_MOTIF_LABEL, "RFX", "SOX"])
    )
    written = save_figure(fig, scratch, "selftest_empty_tracks", ["png"], 60)
    assert written and written[0].exists()
    # the boundary sentences are wrapped, so compare on flattened whitespace
    texts = " ".join(" ".join(t.get_text().split()) for t in fig.findobj(plt.Text))
    assert "ILLUSTRATIVE ONLY" in texts, "the figure must carry the illustration boundary"
    assert "SUPPLEMENTARY" in texts, "the figure must carry the supplementary note"
    assert "not evidence about the population" in texts
    assert "not TF binding or activity" in texts
    assert "no sequence extracted from the reference FASTA" in texts
    assert "position scan found no hit above the frozen FIMO threshold" in texts
    checks.append("draw_model_map: hit-less and sequence-less tracks render and carry both sentences")
    return checks


def _scratch_config(scratch: Path, name: str, stem: str) -> dict:
    """A config whose results root points at a throwaway tree."""
    import yaml

    config = C.load_config(C.CONFIG_DIR / name)
    config["paths"] = dict(config["paths"])
    config["paths"]["results"] = str(scratch / stem)
    config.pop("_path", None)
    config.pop("_root", None)
    path = scratch / f"cfg_{stem}.yaml"
    path.write_text(yaml.safe_dump(config))
    return C.load_config(path)


def self_test(keep: bool = False) -> dict:
    """Run the real path on a tiny slice, then empty the module layer.

    Writes through `run()` into a throwaway `paths.results` and asserts the
    contract. It never reads or writes results/ or results_v2/.
    """
    checks: list[str] = []
    scratch = Path(tempfile.mkdtemp(prefix="s18_selftest_"))
    try:
        # the module layer is interpretation only: this step must not so much as
        # name the primary model/motif table, so it cannot change a primary
        # number, a similarity, a margin, a null or a gate status. The docstring
        # is stripped first, because it is allowed to say what is not read.
        body = Path(__file__).read_text().split('"""', 2)[-1]
        needle = "all_model_motif" + "_scores"
        assert needle not in body, "this step must never read the primary model/motif table"
        checks.append("static: this step does not reference the primary model/motif table")

        checks.extend(_unit_checks(scratch))

        # 1. the real module map, one model
        loaded = _scratch_config(scratch, "analysis_v2.yaml", "results")
        before = {p for p in (C.ROOT / "results", C.ROOT / "results_v2") if p.exists()}
        summary = run(loaded, limit_models=1)
        after = {p for p in (C.ROOT / "results", C.ROOT / "results_v2") if p.exists()}
        assert before == after, "self test touched a real results tree"
        out = Path(summary["outputs"])
        if not out.is_absolute():
            out = C.ROOT / out
        for required in (
            "representative_peaks.tsv",
            "sequence_maps_manifest.tsv",
            "sequence_maps_audit.tsv",
            "sequence_maps_summary.json",
        ):
            assert (out / required).exists(), f"missing {required}"
        manifest = pd.read_csv(out / "sequence_maps_manifest.tsv", sep="\t")
        assert len(manifest) == summary["n_models_selected"]
        for column in ("illustration_boundary", "supplementary_note", "figure_caption"):
            assert manifest[column].notna().all(), f"{column} must be on every manifest row"
        assert summary["n_hits_drawn"] >= 0
        assert all(
            row.n_modules_drawn + row.n_modules_without_qualifying_peak == row.n_modules
            for row in manifest.itertuples()
        ), "every module must be either drawn or named as unqualified"
        windows = set(manifest.window.dropna().astype(str)) - {""}
        assert windows <= set(C.cross_windows(loaded)) | set(
            C.single_species_windows(loaded)
        ), "a window outside the configured table appeared"
        checks.append(
            f"v2 config: {summary['n_models_rendered']} figure(s), {summary['n_hits_drawn']} hit(s), "
            f"{summary['n_modules_without_qualifying_peak']} module(s) with no qualifying peak"
        )
        C.log("self test ok", summary["outputs"])

        # 2. THE RULE: the module layer is interpretation only. An empty module
        # map must not raise and must not write anything except this step's own
        # tables; it must report the negative case per model instead of reaching
        # for a peak the fixed rule forbids.
        empty_map = scratch / "empty_module_map.tsv"
        empty_map.write_text("celltype\tmodule\tlevel\tmember_type\tmotif_name\tmotif_id\tprovenance\n")
        import yaml

        config = C.load_config(C.CONFIG_DIR / "analysis_v2.yaml")
        config["paths"] = dict(config["paths"])
        config["paths"]["results"] = str(scratch / "emptyresults")
        config["modules"] = dict(config.get("modules") or {})
        config["modules"]["file"] = str(empty_map)
        config.pop("_path", None)
        config.pop("_root", None)
        empty_config_path = scratch / "cfg_empty_modules.yaml"
        empty_config_path.write_text(yaml.safe_dump(config))
        empty = run(C.load_config(empty_config_path), limit_models=1)
        empty_out = Path(empty["outputs"])
        if not empty_out.is_absolute():
            empty_out = C.ROOT / empty_out
        empty_manifest = pd.read_csv(empty_out / "sequence_maps_manifest.tsv", sep="\t")
        assert empty["n_modules_resolved"] == 0
        assert empty["n_hits_drawn"] == 0
        assert (empty_manifest.n_modules == 0).all()
        assert empty_manifest.status.str.startswith("skipped").all(), empty_manifest.status.tolist()
        assert empty_manifest.figure_png.fillna("").eq("").all()
        checks.append(
            "empty module map: no crash, no figure, every model named as skipped and no peak invented"
        )
    finally:
        if not keep:
            shutil.rmtree(scratch, ignore_errors=True)
    return {"checks": checks}


def main() -> None:
    C.main_guard()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    C.add_common_args(parser)
    parser.add_argument("--limit-models", type=int, default=None, help="debug: draw only the first N models")
    parser.add_argument("--model", default=None, help="debug: keep models whose id contains this substring")
    parser.add_argument("--species", default=None, help="debug: comma-separated species to keep")
    parser.add_argument("--self-test", action="store_true", help="run the smoke test and exit")
    parser.add_argument("--keep-self-test", action="store_true", help="keep the self-test scratch directory")
    parser.add_argument(
        "--results-root",
        default=None,
        help="redirect this step's output root (e.g. a scratch dir) instead of "
        "paths.results, so the frozen results/ tree is never written",
    )
    parser.add_argument(
        "--allow-v1-results",
        action="store_true",
        help="permit writing the frozen results/ tree (off by default)",
    )
    args = parser.parse_args()

    if args.self_test:
        result = self_test(keep=args.keep_self_test)
        print(json.dumps(result, indent=2))
        return

    config = C.load_config(args.config)
    if args.results_root:
        config["paths"] = dict(config["paths"])
        config["paths"]["results"] = str(Path(args.results_root).expanduser().resolve())
        C.log("results root redirected to", config["paths"]["results"])
    refuse_v1_results_tree(config, args.allow_v1_results)
    run(
        config,
        limit_models=args.limit_models,
        model_filter=args.model,
        species_filter=args.species,
        force=args.force,
        allow_v1_results=args.allow_v1_results,
    )


if __name__ == "__main__":
    main()
