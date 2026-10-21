"""Step 15 -- the Top-k -> motif-module transition: enrichment as a function of rank.

WHY this step exists
--------------------
The main figure jumps from "the 500 peaks this model ranked highest" to "the
879-motif profile of those peaks". That jump hides the step that actually makes
the claim: a regulatory program is said to be *carried by the ranking*, and a
ranking is not a set -- it is an order. A single frozen Top-500 cut-off cannot
distinguish the two readings:

  (a) the model ranks developmental program peaks highest (the intended claim);
  (b) the 500 cap happens to land on a peak set that is enriched for the module
      for reasons that have nothing to do with the order the model produced
      (peak length/GC composition, the 10k HVG universe, the annotation).

This step separates them without touching the primary analysis. For every ranked
model and every cut-off K of `rank_curve.cutoffs` it recomputes the *same*
Top-k / matched-background enrichment s01 computes at Top500, and asks whether
the module signal is already present well below 500 (K = 100, 250) and still
there well above it (K = 1000, 2000). A module whose curve is flat and far above
the matched-random band at every K is a property of the ranking order; a module
that only appears at K = 500 is a property of the cut-off. That is also the
answer to "why Top500?": the curve shows where the signal saturates, and the
band shows how large a random re-draw of the same size and composition is.

EXPLANATORY ONLY -- what this figure is not
-------------------------------------------
This is a transition/description figure. It carries no PASS/FAIL, no gate and no
claim: it is neither primary evidence nor the chromVAR validation, and nothing
in `results_v2/final/` may cite it. Every manifest row says so in its
`claim_role` column, the caption repeats it, and the summary JSON carries the
same sentence. The module layer it draws is INTERPRETATION over the unbiased
879-motif result and can never change a primary number; this step only ever
*reads* `results_v2/01_model_motif/all_model_motif_scores.tsv` and never writes
it.

WHAT is computed, per (model, cut-off K)
----------------------------------------
  1. foreground = the model's peaks with rank <= K;
  2. background = the GC/length matched non-top-K pool, through exactly the
     matching used everywhere else in this project
     (`common.matched_foreground_background`, i.e. `common.match_pairs`). No new
     matching rule is introduced here: same one-to-one, without replacement,
     identical-length, GC within `model_analysis.background.gc_caliper`;
  3. enrichment over all 879 catalog motifs through `common.enrichment` +
     `common.add_significance` (same Fisher test, same Haldane-Anscombe style
     0.5 correction, same BH FDR);
  4. module score = the median of log2_OR over the MONOMERIC members of the
     module (`common.module_members(..., monomers_only=True)`, or the union of
     two atomic modules for a derived aggregate), which is the same aggregator
     every other module statistic in this project uses.

THE MODULE PANEL comes from `config/celltype_motif_module_map.tsv`
-----------------------------------------------------------------
The figure is cell-type specific. Each analysis draws the panel its own cell
type declares:

  * the modules at a GATING level (`common.module_is_gating`; for v2 that is the
    `primary` level) -- Astroglia SOX / NFI / RFX / NF-Y, Purkinje
    Homeodomain_nonPOU / POU / TFAP2, Oligo SOX / bHLH_Ebox;
  * plus every derived union `config.modules.aggregates` declares for that cell
    type -- Purkinje `Homeodomain_total`, the union of `Homeodomain_nonPOU` and
    `POU`, reported alongside its two subdivisions so an 80-motif family is
    never averaged into one line without the split being visible.

No module name, cell type, species or species count is written down here: the
panel is read from the frozen map at run time. An analysis whose cell type has no
row in the map (there is none in v2) is named in `rank_curve_audit.tsv` and its
models are listed in the summary instead of contributing a curve. The map is only
ever READ: emptying it cannot change a primary number, because this step produces
none.

The null band, per (model, K), and the draw-size contract
---------------------------------------------------------
The band is the matched-random control of s01 run at that same K. At every
cut-off the draw is asked to place a foreground AND a background of exactly
`target_K` peaks, where `target_K` is the observed foreground size at that
cut-off -- the size of the observed Fisher table. One rule follows from that and
is enforced, never eased:

    a repeat that cannot place `target_K` peaks on both sides is DROPPED and
    counted (`n_repeats_dropped`), it is NEVER used at a smaller size.

`common.draw_disjoint_matched_sets` places two name-ordered greedy passes and
routinely finishes a few peaks short of its template, which would silently shrink
every repeat; `draw_target_matched_sets` below applies the identical pairing rule
(one-to-one, without replacement, identical length, GC within the caliper, ties
on the closest GC broken uniformly at random) but allocates rarest-template-peak
first and demands both partners at once, so a repeat either reaches the target
exactly or is reported as dropped. When the model's own non-top-k pool cannot
supply a two-sided matched draw of `target_K` peaks at all -- which happens at the
largest cut-offs, where the foreground and the control together have to account
for half the pool -- every repeat of that cut-off is dropped, the observed point
is kept without a band, and `rank_curve_audit.tsv` and
`rank_curve_summary.json` say so for that (model, cut-off).

The RNG seed never contains the cut-off. The observed matching and the null draw
each use `np.random.default_rng([config.seed, model_index])`, the same convention
the primary enrichment (s01) uses, so a cut-off is a parameter of one fixed
experiment rather than part of its seed: re-running the figure at K = 250 does
not re-roll the experiment at K = 500.

WHY the pool is never swapped for the species universe at small K
-----------------------------------------------------------------
s01 falls back to the species universe when a model's own non-top-k pool cannot
place `min_background_pairs` matched peaks. Applying it per cut-off here would
change the background *along the curve* -- K = 100 would be measured against a
different pool than K = 2000 -- and a curve whose background rule moves between
points cannot answer the question this figure asks. The pool is therefore fixed
to the model's own non-top-k peaks for every K, and `meets_min_background_pairs`
(the s01 floor, which is about the power of the frozen Top-500 primary test and
not about a 100-peak cut-off being measurable) is recorded per (model, K) in the
summary JSON as context rather than used as a rescue trigger. A cut-off whose
foreground or pool cannot support a matched comparison at all is skipped with a
named reason in `rank_curve_audit.tsv`, never silently.

Outputs (under the config's results directory, step 13_rank_curve/)
-------------------------------------------------------------------
  rank_curve.tsv          one row per (analysis, species, period, model, module,
                          cutoff): the observed module score and the sizes of
                          the two sides of the Fisher table
  rank_curve_null.tsv.gz  one row per (model, module, cutoff, repeat): the null
                          module score of that matched random draw
  rank_curve_band.tsv     one row per (model, module, cutoff): observed, null
                          mean/sd, band edges, empirical p, usable repeats
  rank_curve_audit.tsv    every skip, with the entity and the reason
  rank_curve_<analysis>.png/pdf
                          one panel per module, faceted by species: observed
                          line + shaded matched-random band, x = cut-off (log).
                          The Oligo and Purkinje panels ship as PDF only (see
                          PDF_ONLY_STEMS): their raster copy is not part of the
                          paper's deliverable set.
  rank_curve_manifest.tsv one row per panel: source table, statistic, caption
                          and its explanatory-only `claim_role`
  rank_curve_summary.json per (model, cut-off) bookkeeping: the draw target, how
                          many peaks were dropped for lack of a sequence, how
                          many could not be matched, and how many null repeats
                          were dropped because the matcher could not reach the
                          target

Only the model ranking enters this step. No h5ad, no chromVAR, no gene map.

Cost, so an operator knows what a full run means: one `common.enrichment` call
over all 879 motifs per (model, cut-off, draw), about a quarter of a second each
on the cached scan. The full v2 run (21 models x 5 cut-offs x 201 draws) is
therefore on the order of an hour and a half. `--cutoffs`, `--repeats`,
`--species` and `--limit-models` narrow a run and make it write suffixed artefact
names, so a debug run can never overwrite the canonical tables.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

STEP = "13_rank_curve"

# The frozen definition of this figure. They are defaults only: the v2 config
# declares both under `rank_curve` and the config always wins. Nothing here names
# a period, a species, a cell type or a module.
DEFAULT_CUTOFFS = (100, 250, 500, 1000, 2000)
DEFAULT_BAND = (2.5, 97.5)
SUPPORTED_AGGREGATE = "median_monomer"

# The role this figure has in the analysis. It is a description of how a model's
# ranking turns into a motif program, not evidence for anything; the string
# travels with every manifest row and with the summary JSON so that no reader and
# no downstream step can promote it into a claim.
CLAIM_ROLE = (
    "explanatory_only: this is a descriptive transition curve. It is not primary "
    "evidence, it is not the chromVAR validation, it gates nothing and it must "
    "not be cited as support for any claim."
)

CURVE_COLUMNS = [
    "analysis",
    "species",
    "period",
    "model",
    "module",
    "module_level",
    "cutoff",
    "observed_log2_OR",
    "n_fg",
    "n_bg",
    "n_members",
]

NULL_COLUMNS = ["model", "module", "cutoff", "repeat", "null_log2_OR"]

BAND_COLUMNS = [
    "model",
    "module",
    "cutoff",
    "observed",
    "null_mean",
    "null_sd",
    "band_lo",
    "band_hi",
    "empirical_p",
    "n_repeats_used",
]

# model / cutoff are always filled when they exist; a row that is not tied to one
# cut-off of one model (a whole module with no monomeric member, an analysis whose
# cell type has no module row) leaves the field empty and says so in `reason`.
AUDIT_COLUMNS = ["model", "analysis", "species", "cutoff", "reason"]

MANIFEST_COLUMNS = [
    "panel",
    "analysis",
    "module",
    "module_level",
    "species",
    "source_files",
    "statistic",
    "band",
    "n_models",
    "n_repeats_requested",
    "n_repeats_used_min",
    "cutoffs",
    "figure_png",
    "figure_pdf",
    "caption",
    "claim_role",
    "status",
]

# --------------------------------------------------------------------------
# frozen look, identical to script/s12_plot_main_figure.py so that the
# transition figure sits in the same visual system as the main figure: the same
# colour-blind-safe categorical slots, the same surface/ink/grid tokens, 180 mm,
# 200 dpi. The slots are assigned in fixed order and never repainted, so a
# period keeps its colour across analyses.
# --------------------------------------------------------------------------

MM = 1.0 / 25.4
FIG_WIDTH_IN = 180.0 * MM
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
LINESTYLES = ("-", "--", ":", "-.")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#d9d8d3"

INTERPRETATION_BOUNDARY = (
    "Motif enrichment describes the sequence composition of model-ranked peak "
    "sets. It is not a measurement of transcription-factor abundance, occupancy "
    "or activity, and no panel here supports a causal or directional claim "
    "about a transcription factor."
)

CAPTION_TEMPLATE = (
    "{analysis} {module_level} module {module}, species {species}. Module score = "
    "median log2 odds-ratio over the monomeric members of {module} ({n_members} "
    "motif(s) used), computed on the model's Top-k peaks against the GC/length "
    "matched non-top-k background drawn from the same ranked universe. The shaded "
    "band is the matched-random-k null at that cut-off ({band_lo}-{band_hi} "
    "percent of up to {n_repeats} matched random-k draws, each drawn at exactly "
    "the observed foreground size); draws that could not reach that size are "
    "dropped and counted rather than used smaller, so a cut-off whose band is "
    "missing is one the model's own pool cannot supply a matched control for. The "
    "empirical p is (1 + #{{null >= observed}}) / (1 + n usable). An empty panel "
    "means no model of that species contributed a curve to this analysis. "
    + CLAIM_ROLE
    + " "
    + INTERPRETATION_BOUNDARY
)

EMPTY_COORDS = pd.DataFrame(
    {
        "peak": pd.Series(dtype=object),
        "length": pd.Series(dtype="int64"),
        "gc": pd.Series(dtype="float64"),
    }
)


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    cutoffs: tuple[int, ...]
    n_repeats: int
    band: tuple[float, float]
    fdr: float
    caliper: float
    min_pairs: int
    n_peaks_reference: int
    seed: int
    # which key sized the null band (`rank_curve.n_repeats` or `--repeats`).
    # Recorded in the summary JSON so the number the artefacts report is never
    # misattributed.
    n_repeats_source: str = "unknown"


def curve_block(config: dict) -> dict:
    return config.get("rank_curve") or {}


def curve_settings(config: dict, cutoffs_arg: list[int] | None, repeats_arg: int | None) -> Settings:
    """Resolve the transition-figure settings from the config."""
    block = curve_block(config)
    aggregate = str(block.get("aggregate") or SUPPORTED_AGGREGATE)
    if aggregate != SUPPORTED_AGGREGATE:
        raise ValueError(
            f"rank_curve.aggregate={aggregate!r} is not supported; this figure and every "
            f"other module statistic in the project use {SUPPORTED_AGGREGATE!r} "
            "(median of log2_OR over a module's monomeric members)"
        )

    raw_cutoffs = cutoffs_arg if cutoffs_arg else block.get("cutoffs")
    cutoffs = tuple(int(k) for k in (raw_cutoffs or DEFAULT_CUTOFFS))
    if not cutoffs:
        raise ValueError("no cut-offs requested")
    if any(k <= 0 for k in cutoffs):
        raise ValueError(f"cut-offs must be positive: {cutoffs}")

    # `rank_curve.n_repeats` is this figure's own knob -- it sizes the matched-
    # random band -- and is independent of `random_control.n_repeats`, which sizes
    # s01's model-level control. The block wins; the CLI overrides it. The source
    # is logged and recorded so a number in the artefacts can never be attributed
    # to the wrong key.
    control = config["random_control"]
    declared = block.get("n_repeats")
    if repeats_arg is not None:
        n_repeats, n_repeats_source = int(repeats_arg), "--repeats"
    elif declared is not None:
        n_repeats, n_repeats_source = int(declared), "rank_curve.n_repeats"
    else:
        n_repeats, n_repeats_source = int(control["n_repeats"]), "random_control.n_repeats"
    if n_repeats < 0:
        raise ValueError("--repeats must be >= 0")

    band = tuple(float(x) for x in (block.get("band") or DEFAULT_BAND))
    if len(band) != 2 or not (0.0 <= band[0] < band[1] <= 100.0):
        raise ValueError(f"rank_curve.band must be two increasing percentiles: {band}")

    background = config["model_analysis"]["background"]
    return Settings(
        cutoffs=cutoffs,
        n_repeats=n_repeats,
        band=(band[0], band[1]),
        fdr=float(config["model_analysis"]["fdr"]),
        caliper=float(background["gc_caliper"]),
        min_pairs=int(config["model_analysis"]["min_background_pairs"]),
        n_peaks_reference=int(control["n_peaks"]),
        seed=C.seed(config),
        n_repeats_source=n_repeats_source,
    )


# --------------------------------------------------------------------------
# the cell-type-specific module panel (interpretation layer, read-only)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModulePanel:
    """One row of the figure: a module of one cell type and its motif members.

    `level` is the level the frozen module map declares for an atomic module
    (`primary` in v2) or `aggregate` for a derived union declared in the config.
    `components` are the atomic modules an aggregate is the union of, so a reader
    can always see that `Homeodomain_total` is not a third, overlapping set.
    """

    analysis: str
    celltype: str
    module: str
    level: str
    members: tuple[str, ...]
    components: tuple[str, ...]


def module_panels(
    config: dict, analyses: list[str]
) -> tuple[dict[str, list[ModulePanel]], list[dict]]:
    """Cell type -> the module panel this analysis draws, plus audit rows.

    Only modules whose level is gating (`common.module_is_gating`) are drawn, so a
    `secondary` or `comparator` module can never be presented as part of the
    primary interpretation of a cell type; the derived aggregates the config
    declares for that cell type are prepended, since they are the joint read-out.
    Nothing is hard-coded: the cell-type key, the module names, their levels and
    their members all come from the frozen map and the config.
    """
    table = C.celltype_module_table(config)
    panels: dict[str, list[ModulePanel]] = {}
    audit: list[dict] = []

    for analysis in analyses:
        rows = table[table.celltype == analysis]
        if rows.empty:
            audit.append(
                dict(
                    model="",
                    analysis=analysis,
                    species="",
                    cutoff="",
                    reason=(
                        f"no_module_row: analysis {analysis} has no row in "
                        f"{C.display_path(C.module_map_path(config))}, so its models "
                        "contribute no curve"
                    ),
                )
            )
            continue
        level_of = dict(zip(rows.module, rows.level))
        gating = [m for m in dict.fromkeys(rows.module) if C.module_is_gating(config, level_of[m])]
        if not gating:
            audit.append(
                dict(
                    model="",
                    analysis=analysis,
                    species="",
                    cutoff="",
                    reason=(
                        f"no_gating_module: every module of {analysis} sits at a "
                        "non-gating level, so the analysis has no primary module to draw"
                    ),
                )
            )
            continue

        panel: list[ModulePanel] = []
        for aggregate, components in C.module_aggregates(config, analysis).items():
            missing = [c for c in components if c not in gating]
            if missing:
                audit.append(
                    dict(
                        model="",
                        analysis=analysis,
                        species="",
                        cutoff="",
                        reason=(
                            f"aggregate_not_drawn: {analysis}/{aggregate} unions "
                            f"{missing}, which are not gating modules of {analysis}"
                        ),
                    )
                )
                continue
            members: set[str] = set()
            for component in components:
                members.update(C.module_members(config, analysis, component))
            panel.append(
                ModulePanel(
                    analysis=analysis,
                    celltype=analysis,
                    module=aggregate,
                    level="aggregate",
                    members=tuple(sorted(members)),
                    components=tuple(components),
                )
            )
        for module in gating:
            members = tuple(C.module_members(config, analysis, module))
            if not members:
                audit.append(
                    dict(
                        model="",
                        analysis=analysis,
                        species="",
                        cutoff="",
                        reason=(
                            f"no_monomer: module {analysis}/{module} has no monomeric "
                            "member in the module map, so it has no module score"
                        ),
                    )
                )
                continue
            panel.append(
                ModulePanel(
                    analysis=analysis,
                    celltype=analysis,
                    module=module,
                    level=level_of[module],
                    members=members,
                    components=(module,),
                )
            )
        panels[analysis] = panel

    return panels, audit


# --------------------------------------------------------------------------
# manifest / model bookkeeping
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """One ranked model, with everything that is fixed before a sequence is read.

    `model_index` is the row position in the *unfiltered* manifest, so the stream
    this step uses -- `[config.seed, model_index]` for the observed matching and
    for the null draws alike -- is identical whether a run covers every model or a
    debugging subset.
    """

    model: str
    model_index: int
    analysis: str
    species: str
    celltype: str
    period: str
    window: str
    n_ranked: int
    is_empty: bool


def window_lookup(config: dict) -> dict[tuple[str, str], str]:
    """(species, rank_period) -> cross-window, from the config's own period table.

    The ranked-file period token is a `rank_period`; what a reader compares across
    species is the window. The mapping is derived from whichever stage map the
    config declares, and an unmapped token (the Purkinje/Oligo rankings, which are
    not split by period at all) maps to "" rather than to a guessed window.
    """
    table = C.stage_table(config)
    out: dict[tuple[str, str], str] = {}
    for row in table.itertuples():
        out[(str(row.species), str(row.rank_period))] = str(row.cross_window)
    return out


def model_specs(config: dict, manifest: pd.DataFrame) -> list[ModelSpec]:
    windows = window_lookup(config)
    return [
        ModelSpec(
            model=row.model,
            model_index=index,
            analysis=row.analysis,
            species=row.species,
            celltype=row.celltype,
            period=row.period,
            window=windows.get((row.species, row.period), ""),
            n_ranked=int(row.n_ranked),
            is_empty=bool(row.is_empty),
        )
        for index, row in enumerate(manifest.itertuples())
    ]


# --------------------------------------------------------------------------
# module aggregation, the target-sized draw, and the band
# --------------------------------------------------------------------------


def module_medians(
    profile: pd.DataFrame,
    modules: list[ModulePanel],
    value: str = "log2_OR",
) -> pd.DataFrame:
    """Median of `value` over the monomeric members of each module.

    This is the project-wide module aggregator (see `common.family_trajectory`,
    which takes the same median over the same monomeric members after a per-motif
    centering that a single model has no analogue for). A member the motif catalog
    does not contain is counted as missing rather than silently shrinking the
    median.
    """
    lookup = profile.set_index("motif_id")
    rows = []
    for panel in modules:
        wanted = list(panel.members)
        present = [m for m in wanted if m in lookup.index]
        values = lookup.loc[present, value].to_numpy(dtype=float) if present else np.array([])
        rows.append(
            dict(
                module=panel.module,
                module_level=panel.level,
                score=float(np.median(values)) if len(values) else np.nan,
                n_members=int(len(present)),
                n_missing=int(len(wanted) - len(present)),
                n_significant=(
                    int(lookup.loc[present, "significant"].sum())
                    if present and "significant" in lookup.columns
                    else 0
                ),
            )
        )
    return pd.DataFrame(rows)


def _serve_one(candidates: np.ndarray, distances: np.ndarray, available: np.ndarray, rng) -> int | None:
    """Place one free pool peak for one template peak, by the project's rule.

    The candidate set is the free pool peaks of identical length within the GC
    caliper; among those the ones at the smallest GC distance compete, and one is
    taken uniformly at random. This is `common.match_pairs`'s rule exactly, which
    is why the observed matching and this draw can be compared at all.
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


# A null draw may fall this far short of the observed foreground size and still
# form the band; below it the repeat is dropped and counted.
MIN_DRAW_FRACTION = 0.9


def draw_target_matched_sets(
    template: pd.DataFrame,
    pool: pd.DataFrame,
    rng: "np.random.Generator",
    caliper: float,
    target: int,
) -> tuple[list[str], list[str]]:
    """Two disjoint, equally sized, GC/length matched draws, up to `target` pairs.

    Each template peak is asked for TWO distinct pool peaks at once, and template
    peaks are served rarest-candidate-first so that the peaks with the fewest
    possible partners are not starved by the abundant ones. A template peak that
    cannot supply both partners is skipped and the draw continues with the next,
    so the returned lists hold as many pairs as the pool could actually place --
    never more than `target`, and equal in length to each other. The caller
    compares that length with `target`: anything short of it means the repeat must
    be DROPPED and counted, and the returned length is what the audit reports as
    the shortfall.

    The pairing rule is `common.match_pairs`'s, unchanged: one-to-one, without
    replacement, identical length, GC within `caliper`, ties on the closest GC
    broken uniformly at random. Only the allocation order is local, because the
    two name-ordered greedy passes of `common.draw_disjoint_matched_sets` finish a
    few peaks short of their template and this figure's contract is that a repeat
    reaches the requested size or is dropped.
    """
    pool_length = pool.length.to_numpy()
    pool_gc = pool.gc.to_numpy(dtype=float)
    candidates: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    for row in template.itertuples():
        index = np.flatnonzero(
            (pool_length == row.length) & (np.abs(pool_gc - row.gc) <= caliper)
        )
        candidates.append(index)
        distances.append(np.abs(pool_gc[index] - row.gc))

    available = np.ones(len(pool), dtype=bool)
    order = sorted(range(len(candidates)), key=lambda i: (len(candidates[i]), i))
    foreground: list[str] = []
    background: list[str] = []
    pool_name = pool.peak.to_numpy()
    for i in order:
        if len(foreground) >= target:
            break
        picked: list[int] = []
        for _ in range(2):
            pick = _serve_one(candidates[i], distances[i], available, rng)
            if pick is None:
                break
            picked.append(pick)
        if len(picked) < 2:
            for pick in picked:
                available[pick] = True
            continue
        foreground.append(str(pool_name[picked[0]]))
        background.append(str(pool_name[picked[1]]))
    return foreground, background


def band_stats(values: np.ndarray, observed: float, band: tuple[float, float]) -> dict:
    """Null mean/sd, percentile band and the one-sided empirical p.

    `values` are the null module scores of the repeats that produced a usable
    matched draw. The p is (1 + #{null >= observed}) / (1 + n) so that it is never
    exactly zero, and is NaN when there is no usable repeat at all.
    """
    finite = values[np.isfinite(values)] if len(values) else values
    n = int(len(finite))
    if n == 0:
        return dict(
            null_mean=np.nan, null_sd=np.nan, band_lo=np.nan, band_hi=np.nan,
            empirical_p=np.nan, n_repeats_used=0,
        )
    lo, hi = np.percentile(finite, band)
    p_value = np.nan
    if np.isfinite(observed):
        p_value = float((np.sum(finite >= observed) + 1) / (n + 1))
    return dict(
        null_mean=float(np.mean(finite)),
        null_sd=float(np.std(finite, ddof=1)) if n > 1 else np.nan,
        band_lo=float(lo),
        band_hi=float(hi),
        empirical_p=p_value,
        n_repeats_used=n,
    )


# --------------------------------------------------------------------------
# one model
# --------------------------------------------------------------------------


def coords_frame(coords: pd.DataFrame, peaks: list[str]) -> pd.DataFrame:
    """(peak, length, gc) rows in the given order; dtype-stable when empty."""
    peaks = list(peaks)
    if not peaks:
        return EMPTY_COORDS.copy()
    return coords.loc[peaks].reset_index()


def run_model(
    spec: ModelSpec,
    ranking: pd.DataFrame,
    scan: "SpeciesScan",
    catalog: pd.DataFrame,
    settings: Settings,
    panel: list[ModulePanel],
    audit: list[dict],
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Observed curve, null draws, band and summary for one model, every cut-off.

    Returns (curve rows, null rows, band rows, summary entry). Nothing raises for
    a cut-off that cannot be evaluated: it produces an audit row naming the model,
    the cut-off and the reason, and the curve simply has no point there.
    """

    def skip(cutoff: int | None, reason: str) -> None:
        audit.append(
            dict(
                model=spec.model,
                analysis=spec.analysis,
                species=spec.species,
                cutoff="" if cutoff is None else int(cutoff),
                reason=reason,
            )
        )

    summary: dict = dict(
        model=spec.model,
        analysis=spec.analysis,
        species=spec.species,
        celltype=spec.celltype,
        period=spec.period,
        window=spec.window,
        model_index=spec.model_index,
        n_ranked=spec.n_ranked,
        status="pending",
        cutoffs=[],
    )
    curve_rows: list[dict] = []
    null_rows: list[dict] = []
    band_rows: list[dict] = []

    if spec.is_empty or spec.n_ranked == 0:
        summary["status"] = "empty_ranking"
        skip(None, "empty_ranking: the ranked CSV has no rows")
        return curve_rows, null_rows, band_rows, summary

    sequences = scan.sequences
    if sequences is None or not scan.hits:
        summary["status"] = f"unusable_scan:{scan.row.get('status', 'unknown')}"
        skip(None, f"unusable_scan:{scan.row.get('status', 'unknown')}")
        return curve_rows, null_rows, band_rows, summary

    valid = sequences.valid.astype(bool).to_numpy()
    valid_peaks = set(sequences.peak.to_numpy()[valid])
    coords = sequences[valid].set_index("peak")[["length", "gc"]]

    for cutoff in settings.cutoffs:
        entry: dict = dict(
            cutoff=int(cutoff),
            status="pending",
            n_top=0,
            n_top_valid=0,
            n_top_no_sequence=0,
            n_pool=0,
            n_fg=0,
            n_bg=0,
            n_fg_unmatched=0,
            max_gc_difference=np.nan,
            background_source="same_model_non_topk",
            meets_min_background_pairs=False,
            null_draw_target=0,
            n_repeats_requested=int(settings.n_repeats),
            n_repeats_used=0,
            n_repeats_dropped=0,
            null_draw_size_min=0,
            null_draw_size_max=0,
            modules={},
        )
        summary["cutoffs"].append(entry)

        if spec.n_ranked < cutoff:
            entry["status"] = "ranking_shorter_than_cutoff"
            skip(
                cutoff,
                f"ranking_shorter_than_cutoff: {spec.model} ranks {spec.n_ranked} peaks, "
                f"below the {int(cutoff)} cut-off, so there is no Top-{int(cutoff)} set",
            )
            continue

        top = ranking[ranking["rank"] <= cutoff]
        top_valid = top.peak[top.peak.isin(valid_peaks)].tolist()
        entry["n_top"] = int(len(top))
        entry["n_top_valid"] = int(len(top_valid))
        entry["n_top_no_sequence"] = int(len(top)) - int(len(top_valid))
        if not top_valid:
            entry["status"] = "no_valid_foreground"
            skip(
                cutoff,
                f"no_valid_foreground: none of the {int(len(top))} Top-{int(cutoff)} peaks of "
                f"{spec.model} has a recoverable {spec.species} sequence",
            )
            continue

        pool_frame = coords_frame(
            coords, [p for p in ranking.peak[ranking["rank"] > cutoff] if p in valid_peaks]
        )
        entry["n_pool"] = int(len(pool_frame))
        if pool_frame.empty:
            # The whole ranking is inside the cut-off; there is no non-top-k pool
            # to match against, so no background and no null can exist.
            entry["status"] = "empty_background_pool"
            skip(
                cutoff,
                f"empty_background_pool: {spec.model} has no ranked peak above the "
                f"{int(cutoff)} cut-off to use as a matched background",
            )
            continue

        top_frame = coords_frame(coords, top_valid)
        # The observed matching uses the project's model stream -- the seed is
        # [config.seed, model_index] and never contains the cut-off -- so the
        # K = model_analysis.primary_topk point of the curve is the *same* matched
        # Top-500 enrichment s01 publishes, not a re-draw of a different tie-break
        # realisation.
        rng = np.random.default_rng([settings.seed, spec.model_index])
        fg, bg, pairs = C.matched_foreground_background(
            top_frame, pool_frame, rng, settings.caliper
        )
        matched = pairs[pairs.bg_peak != ""]
        entry["n_fg"] = int(len(fg))
        entry["n_bg"] = int(len(bg))
        entry["n_fg_unmatched"] = int(len(top_valid) - len(fg))
        entry["max_gc_difference"] = (
            float(matched.gc_difference.max()) if len(matched) else np.nan
        )
        entry["meets_min_background_pairs"] = bool(len(fg) >= settings.min_pairs)
        if not fg:
            entry["status"] = "no_matched_pairs"
            skip(
                cutoff,
                f"no_matched_pairs: no Top-{int(cutoff)} peak of {spec.model} found a "
                "length/GC matched partner in its own non-top-k pool",
            )
            continue

        observed_profile = C.add_significance(
            C.enrichment(catalog, scan.hits, fg, bg), settings.fdr
        )
        observed_modules = module_medians(observed_profile, panel)

        # ---- the null band: a matched random draw at exactly the observed size ----
        # The target is the size of the observed Fisher table, so the band is
        # comparable to the point it belongs to. A repeat that cannot place
        # `target` peaks on BOTH sides is dropped and counted; it is never used at
        # a smaller size, which is the difference between a band and a decoy.
        target = int(len(fg))
        entry["null_draw_target"] = target
        # The template is the whole observed foreground -- the exact peak set the
        # observed matching was handed -- so a draw is matched to the same
        # length/GC profile the observed foreground was drawn from, and the peaks
        # that could not be matched are available to be passed over rather than
        # silently consuming the target.
        template = top_frame
        null_values: dict[str, list[float]] = {p.module: [] for p in panel}
        dropped = 0
        drawn_min = target
        drawn_max = 0
        # The null draws use the same [seed, model_index] convention as the
        # observed matching and as the primary enrichment, so the cut-off enters
        # the experiment only as a parameter (the template it is drawn for) and
        # never as part of the seed.
        null_rng = np.random.default_rng([settings.seed, spec.model_index])
        for repeat in range(settings.n_repeats):
            draw_fg, draw_bg = draw_target_matched_sets(
                template, pool_frame, null_rng, settings.caliper, target
            )
            drawn_min = min(drawn_min, len(draw_fg))
            drawn_max = max(drawn_max, len(draw_fg))
            # What the band requires is a BALANCED Fisher table and a draw that
            # is not meaningfully smaller than the observed one -- not an exact
            # match with the observed size. `draw_disjoint_matched_sets` needs two
            # disjoint matched partners per template peak, so the achieved size is
            # legitimately a few peaks short of the target (486 of 489 in the worst
            # case here); demanding equality discarded all 200 repeats for 15
            # model x cut-off cells and left them with no band at all. log2_OR is
            # scale free, so a 0.6%-smaller draw is comparable; the achieved size
            # is recorded per cell.
            floor = max(1, int(np.ceil(MIN_DRAW_FRACTION * target)))
            if len(draw_fg) != len(draw_bg) or len(draw_fg) < floor:
                dropped += 1
                continue
            # The identical pipeline, FDR correction included: a reader comparing
            # an observed point with its band must not have to wonder whether the
            # two went through different code.
            null_profile = C.add_significance(
                C.enrichment(catalog, scan.hits, draw_fg, draw_bg), settings.fdr
            )
            null_modules = module_medians(null_profile, panel)
            for row in null_modules.itertuples():
                null_rows.append(
                    dict(
                        model=spec.model,
                        module=row.module,
                        cutoff=int(cutoff),
                        repeat=int(repeat),
                        null_log2_OR=float(row.score),
                    )
                )
                if np.isfinite(row.score):
                    null_values[row.module].append(float(row.score))
        entry["n_repeats_used"] = int(settings.n_repeats - dropped)
        entry["n_repeats_dropped"] = int(dropped)
        entry["null_draw_size_min"] = int(drawn_min)
        entry["null_draw_size_max"] = int(drawn_max)
        if dropped and settings.n_repeats:
            C.log(
                " ",
                spec.model,
                f"K={int(cutoff)}",
                f"{dropped}/{settings.n_repeats} null draws could not reach the "
                f"{target}-peak target on both sides (best {int(drawn_max)})",
            )

        for row in observed_modules.itertuples():
            curve_rows.append(
                dict(
                    analysis=spec.analysis,
                    species=spec.species,
                    period=spec.period,
                    model=spec.model,
                    module=row.module,
                    module_level=row.module_level,
                    cutoff=int(cutoff),
                    observed_log2_OR=float(row.score),
                    n_fg=int(len(fg)),
                    n_bg=int(len(bg)),
                    n_members=int(row.n_members),
                )
            )
            stats = band_stats(
                np.asarray(null_values.get(row.module, []), dtype=float),
                float(row.score),
                settings.band,
            )
            band_rows.append(
                dict(
                    model=spec.model,
                    module=row.module,
                    cutoff=int(cutoff),
                    observed=float(row.score),
                    **stats,
                )
            )
            entry["modules"][row.module] = dict(
                level=row.module_level,
                observed=float(row.score),
                n_members=int(row.n_members),
                n_missing=int(row.n_missing),
                n_significant=int(row.n_significant),
                null_mean=stats["null_mean"],
                band_lo=stats["band_lo"],
                band_hi=stats["band_hi"],
                empirical_p=stats["empirical_p"],
                n_repeats_used=stats["n_repeats_used"],
            )
        entry["status"] = "ok"

        if settings.n_repeats > 0 and entry["n_repeats_used"] == 0:
            skip(
                cutoff,
                f"no_usable_null_repeat: all {int(settings.n_repeats)} matched random-"
                f"k draws of {spec.model} placed fewer than the {target} peaks the "
                f"observed Top-{int(cutoff)} foreground has (best {int(drawn_max)}); "
                "the observed point is kept, the band is not",
            )

    statuses = {entry["status"] for entry in summary["cutoffs"]}
    summary["status"] = "ok" if statuses == {"ok"} else "+".join(sorted(statuses))
    return curve_rows, null_rows, band_rows, summary


@dataclass
class SpeciesScan:
    """One species' cached FIMO scan plus the sequence bookkeeping it implies."""

    species: str
    hits: dict[str, set[str]]
    row: dict
    sequences: pd.DataFrame | None = None
    peaks: tuple[str, ...] = ()


def scan_species(
    config: dict, species: str, genome: C.SpeciesGenome | None, universe: list[str], workers: int
) -> SpeciesScan:
    """The same shared, cached, never-forced FIMO scan s01 performs.

    The scan universe is the per-species union of that species' ranked universes,
    and `C.scan_motifs` is called with `force=False` over `C.read_sequences(...,
    use_cache=True)`, so the motif hits this step reads are literally the ones the
    primary analysis reads.
    """
    row = {
        "species": species,
        "genome_assembly": genome.assembly if genome is not None else "",
        "n_universe_peaks": len(universe),
        "n_scan_peaks": 0,
        "n_no_sequence": 0,
        "n_motif_peak_pairs": 0,
        "status": "pending",
    }
    if genome is None:
        row["status"] = "no_genome"
        return SpeciesScan(species, {}, row)
    if not universe:
        row["status"] = "empty_universe"
        return SpeciesScan(species, {}, row)

    sequences = C.read_sequences(config, genome, universe, use_cache=True)
    valid = sequences.valid.astype(bool)
    row["n_scan_peaks"] = int(valid.sum())
    row["n_no_sequence"] = int(len(sequences) - int(valid.sum()))
    if row["n_scan_peaks"] == 0:
        row["status"] = "no_valid_sequence"
        return SpeciesScan(species, {}, row, sequences, tuple(sequences.peak))

    hits = C.scan_motifs(config, genome, sequences, workers=workers, force=False)
    row["n_motif_peak_pairs"] = int(sum(len(peaks) for peaks in hits.values()))
    row["status"] = "ok"
    return SpeciesScan(species, hits, row, sequences, tuple(sequences.peak))


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------


def safe_stem(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "analysis"


def abbreviate(text: str) -> str:
    """A short, human-readable period label for a legend."""
    return str(text).replace("_", " ")


def cutoff_label(value: int) -> str:
    """A cut-off label that does not collide with its neighbour on a log axis.

    1000 and 2000 printed in full overlap at this panel width; 1k and 2k do not,
    and the axis label already says the unit is peaks.
    """
    value = int(value)
    if value >= 1000 and value % 1000 == 0:
        return f"{value // 1000}k"
    return str(value)


def save_gzip_table(frame: pd.DataFrame, path: Path, sep: str = "\t") -> Path:
    """Write a TSV through a gzip container that carries no timestamp.

    `pandas.to_csv("*.gz")` stamps the current time into the gzip header, so two
    runs of this deterministic step would produce different bytes for the null
    table and any checksum of the frozen artefact would fire falsely. mtime=0
    removes the stamp; the decompressed content is unchanged.
    """
    import gzip
    import io

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, sep=sep, index=False)
    C.log("wrote", C.display_path(path), f"({len(frame)} rows)")
    return path


# Figures of this step that ship as PDF only: their raster copy is not part of
# the paper's deliverable set, so it is not written. Declared here, beside the
# saver it affects, so the exclusion is visible in the code rather than implicit.
PDF_ONLY_STEMS = frozenset({
    "rank_curve_Oligo",
    "rank_curve_Purkinje",
})


def save_figure(fig: "object", out_dir: Path, stem: str, formats: list[str], dpi: int) -> list[Path]:
    written = []
    if stem in PDF_ONLY_STEMS:
        formats = [fmt for fmt in formats if str(fmt).lower() != "png"]
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        # A vector backend stamps a fresh creation date into every file, so two
        # runs of this deterministic script would differ byte for byte; None
        # removes the stamp, and the raster formats ignore it. Same rule as s12.
        metadata = {"CreationDate": None, "ModDate": None} if fmt.lower() in {"pdf", "svg"} else None
        fig.savefig(path, dpi=dpi, facecolor=SURFACE, metadata=metadata)
        written.append(path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return written


def no_rows_note(ax: "object", text: str) -> None:
    """Say why a panel is empty. Ticks are deliberately left alone: with a
    shared axis they belong to the whole row/column, not to this one panel."""
    ax.text(
        0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
        fontsize=6, color=INK_MUTED,
    )


def plot_analysis(
    analysis: str,
    modules: list[str],
    curve: pd.DataFrame,
    band: pd.DataFrame,
    settings: Settings,
    out_dir: Path,
    stem: str,
    formats: list[str],
    dpi: int,
    model_order: list[str],
) -> list[Path]:
    """One figure per analysis: rows = modules, columns = species facets.

    Each panel holds one observed line per model of that species and, under it,
    that model's own null band. The facet is the species, so at most a handful of
    lines share a panel and the colours do not have to be cycled across the whole
    analysis; the legend (only drawn when a panel has more than one model) names
    the period of each line. A module row is drawn even when no model produced a
    point for it, so a missing module is visible rather than absent.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub_curve = curve[curve.analysis == analysis]
    sub_band = band[band.model.isin(set(sub_curve.model))]
    species_of = dict(zip(sub_curve.model, sub_curve.species))
    models = [m for m in model_order if m in set(sub_curve.model)]
    species_present = [s for s in C.SPECIES_ORDER if s in set(sub_curve.species)]
    species_present += sorted(set(sub_curve.species) - set(species_present))
    if not species_present:
        species_present = ["(no model)"]

    footer_in = 0.42
    fig, axes = plt.subplots(
        len(modules),
        len(species_present),
        figsize=(FIG_WIDTH_IN, 1.30 * len(modules) + 1.15 + footer_in),
        squeeze=False,
        sharex=True,
        sharey="row",
        layout="constrained",
    )
    # Constrained layout does not know about the boundary note drawn with
    # fig.text below the axes, so the strip it needs is reserved explicitly;
    # otherwise the note runs into the bottom row's x-axis label.
    footer_fraction = footer_in / (1.30 * len(modules) + 1.15 + footer_in)
    fig.get_layout_engine().set(rect=(0.0, footer_fraction, 1.0, 1.0 - footer_fraction))
    tick_labels = [cutoff_label(k) for k in settings.cutoffs]
    positions = np.asarray(settings.cutoffs, dtype=float)

    for row_index, module in enumerate(modules):
        for col_index, species in enumerate(species_present):
            ax = axes[row_index][col_index]
            panel_models = [m for m in models if species_of.get(m) == species]
            for slot, model in enumerate(panel_models):
                colour = CATEGORICAL[slot % len(CATEGORICAL)]
                style = LINESTYLES[(slot // len(CATEGORICAL)) % len(LINESTYLES)]
                observed = sub_curve[
                    (sub_curve.model == model) & (sub_curve.module == module)
                ].sort_values("cutoff")
                if observed.empty:
                    continue
                x = observed.cutoff.to_numpy(dtype=float)
                ax.plot(
                    x,
                    observed.observed_log2_OR.to_numpy(dtype=float),
                    color=colour,
                    linestyle=style,
                    linewidth=1.3,
                    marker="o",
                    markersize=3.2,
                    markeredgecolor=SURFACE,
                    markeredgewidth=0.5,
                    label=abbreviate(observed.period.iloc[0]),
                    zorder=3,
                )
                interval = sub_band[
                    (sub_band.model == model) & (sub_band.module == module)
                ].sort_values("cutoff")
                interval = interval[
                    np.isfinite(interval.band_lo) & np.isfinite(interval.band_hi)
                ]
                if not interval.empty:
                    ax.fill_between(
                        interval.cutoff.to_numpy(dtype=float),
                        interval.band_lo.to_numpy(dtype=float),
                        interval.band_hi.to_numpy(dtype=float),
                        color=colour,
                        alpha=0.16,
                        linewidth=0,
                        zorder=1,
                    )
            ax.axhline(0.0, color=GRID, linewidth=0.7, zorder=0)
            # Headroom for the band and for the legend, which matplotlib places
            # in the emptiest corner but can only do if one exists.
            ax.margins(y=0.14)
            ax.set_xscale("log")
            ax.set_xticks(positions, tick_labels)
            # The only meaningful x values are the cut-offs themselves; the log
            # minor ticks (2x10^2, 4x10^2, ...) would sit between them and, with
            # a single cut-off, collide with the one label that matters.
            ax.minorticks_off()
            ax.set_xlim(min(positions) / 1.6, max(positions) * 1.6)
            if col_index == 0:
                ax.set_ylabel(f"{module}\nmedian log2 OR", fontsize=7)
            if row_index == 0:
                ax.set_title(species, fontsize=8, color=INK)
            if row_index == len(modules) - 1:
                # One x label for the figure: every panel of a row shares the
                # cut-offs, and a log x-axis is a log x-axis in every column.
                ax.set_xlabel(
                    "Top-k cut-off (peaks, log scale)" if col_index == 0 else "", fontsize=7
                )
            if not panel_models:
                no_rows_note(ax, "no model of this species\nin this analysis")
            elif len(panel_models) > 1:
                ax.legend(
                    loc="best",
                    fontsize=5.2,
                    frameon=False,
                    handlelength=1.4,
                    labelspacing=0.25,
                    borderpad=0.2,
                )

    n_models = len(models)
    n_used = int(sub_band.n_repeats_used.max()) if len(sub_band) else 0
    fig.suptitle(
        f"{analysis}: motif-module enrichment as a function of the model's Top-k cut-off\n"
        f"statistic: median log2 odds-ratio over the monomeric members of each module "
        f"(Top-k peaks vs the GC/length matched non-top-k background)\n"
        f"band: {settings.band[0]:g}-{settings.band[1]:g}% of n={settings.n_repeats} "
        f"matched random-k draws per cut-off, each at the observed foreground size "
        f"(up to {n_used} usable); {n_models} model(s)",
        fontsize=6.6,
        color=INK,
    )
    fig.text(
        0.006,
        0.004,
        CLAIM_ROLE
        + " "
        + INTERPRETATION_BOUNDARY
        + " Shaded band = the matched random-k null at that cut-off.",
        fontsize=5.4,
        color=INK_MUTED,
        ha="left",
        va="bottom",
        wrap=True,
    )
    return save_figure(fig, out_dir, stem, formats, dpi)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def artefact_suffix(
    cutoffs_arg: list[int] | None,
    repeats_arg: int | None,
    species: list[str],
    limit_models: int | None,
) -> str:
    """Name suffix describing exactly which slice of the manifest a run covers.

    The unsuffixed names are the canonical transition-figure artefacts of the
    full v2 run. --cutoffs, --repeats, --species and --limit-models each narrow a
    run, so each contributes a part here and a debug run writes its own tables
    instead of overwriting them, exactly as s01 does for its own artefacts.
    """
    parts: list[str] = []
    if cutoffs_arg:
        parts.append("K" + "-".join(str(int(k)) for k in cutoffs_arg))
    if repeats_arg is not None:
        parts.append(f"R{int(repeats_arg)}")
    if species or limit_models is not None:
        scope = "sub" + ("-".join(sorted(species)) if species else "all")
        if limit_models is not None:
            scope += f"_N{int(limit_models)}"
        parts.append(scope)
    return "".join(f"_{part}" for part in parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    C.add_common_args(parser)
    parser.add_argument("--workers", type=int, default=16, help="FIMO workers (cold cache only)")
    parser.add_argument("--species", nargs="*", default=None, help="restrict to these species")
    parser.add_argument(
        "--limit-models", type=int, default=None, help="debug aid: only the first N models"
    )
    parser.add_argument(
        "--cutoffs",
        nargs="*",
        type=int,
        default=None,
        help="override rank_curve.cutoffs (debug aid)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="write here instead of the config's results dir (test aid; default "
        "is C.results_dir(config, '13_rank_curve'))",
    )
    args = parser.parse_args()
    C.main_guard()

    config = C.load_config(args.config)
    if args.force:
        C.log("--force ignored: the FIMO cache is shared and must not be invalidated")

    settings = curve_settings(config, args.cutoffs, args.repeats)
    suffix = artefact_suffix(args.cutoffs, args.repeats, list(args.species or []), args.limit_models)
    if suffix:
        C.log(
            "narrowed run: writing suffixed artefacts",
            repr(suffix),
            "(the unsuffixed names are the canonical full-run tables)",
        )

    out = Path(args.out_dir) if args.out_dir else C.results_dir(config, STEP)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "curve": out / f"rank_curve{suffix}.tsv",
        "null": out / f"rank_curve_null{suffix}.tsv.gz",
        "band": out / f"rank_curve_band{suffix}.tsv",
        "audit": out / f"rank_curve_audit{suffix}.tsv",
        "manifest": out / f"rank_curve_manifest{suffix}.tsv",
        "summary": out / f"rank_curve_summary{suffix}.json",
    }
    formats = [str(f) for f in (config.get("plotting", {}).get("formats") or ["png"])]
    dpi = int(config.get("plotting", {}).get("dpi", 200))

    with C.StepTimer("s15 rank curve"):
        manifest, rankings = C.load_rankings(config)
        specs = model_specs(config, manifest)
        audit: list[dict] = []

        # The module panel is cell-type specific and comes from the frozen map;
        # an analysis whose cell type is not in it is audited, never guessed.
        analyses = list(manifest.analysis.drop_duplicates())
        panels, panel_audit = module_panels(config, analyses)
        audit.extend(panel_audit)
        C.log(
            "module map",
            C.display_path(C.module_map_path(config)),
            "panels",
            {name: [p.module for p in rows] for name, rows in panels.items()},
        )

        # The species filter (v2 only; a no-op without the config block) drops a
        # whole model, so every drop is a named audit row before anything else.
        # The reason is the one the filter itself computed (`row.action`), which
        # names every criterion that fired -- the cell floor, the peak floor, or
        # both. Re-rendering a template here would publish a false statement
        # whenever only the peak floor tripped (a species with thousands of cells
        # dropped for a ranking too short to define a Top-k foreground).
        kept_manifest, dropped = C.apply_species_filter(config, manifest)
        kept_models = set(kept_manifest.model)
        species_filter_drops = [
            dict(
                model=row.model,
                analysis=row.analysis,
                species=row.species,
                celltype=row.celltype,
                n_cells=int(row.n_cells),
                min_cells=int(row.min_cells),
                n_ranked=int(row.n_ranked),
                min_ranked_peaks=int(row.min_ranked_peaks),
                reason=row.action,
            )
            for row in dropped.itertuples()
        ]
        for row in dropped.itertuples():
            audit.append(
                dict(
                    model=row.model,
                    analysis=row.analysis,
                    species=row.species,
                    cutoff="",
                    reason=f"species_filter: {row.species} {row.celltype} {row.action}",
                )
            )
        selected = [spec for spec in specs if spec.model in kept_models]

        if not panels:
            # An emptied module map is a legitimate state of the interpretation
            # layer: this figure then draws nothing, and says so, without failing
            # a run that produced no primary number to begin with.
            C.log("WARNING no analysis has a module panel; no curve will be drawn")
        for spec in selected:
            if spec.analysis not in panels:
                audit.append(
                    dict(
                        model=spec.model,
                        analysis=spec.analysis,
                        species=spec.species,
                        cutoff="",
                        reason=(
                            f"no_module_panel: {spec.analysis} has no gating module row in "
                            f"{C.display_path(C.module_map_path(config))}, so {spec.model} "
                            "contributes no curve"
                        ),
                    )
                )

        wanted_species = list(args.species) if args.species else []
        if wanted_species:
            for spec in selected:
                if spec.species not in wanted_species:
                    audit.append(
                        dict(
                            model=spec.model,
                            analysis=spec.analysis,
                            species=spec.species,
                            cutoff="",
                            reason=f"--species: {spec.species} was not selected for this run",
                        )
                    )
            selected = [spec for spec in selected if spec.species in wanted_species]
        if args.limit_models is not None:
            n_selected = len(selected)
            for spec in selected[int(args.limit_models):]:
                audit.append(
                    dict(
                        model=spec.model,
                        analysis=spec.analysis,
                        species=spec.species,
                        cutoff="",
                        reason=f"--limit-models {int(args.limit_models)}: model outside the first "
                        f"{int(args.limit_models)} of {n_selected} selected",
                    )
                )
            selected = selected[: int(args.limit_models)]
        selected = [spec for spec in selected if spec.analysis in panels]
        if not selected and panels:
            raise RuntimeError("no model selected; check --species / --limit-models")

        # The scan universe is the full per-species union of the manifest (never
        # the narrowed selection): restricting the model list must not change the
        # cached FIMO input.
        universes = C.model_species_peaks(manifest[~manifest.is_empty], rankings)
        genomes = C.species_table(config)
        catalog = C.motif_catalog(config)
        C.log(
            "catalog",
            len(catalog),
            "motifs;",
            len(selected),
            "models;",
            "cut-offs",
            list(settings.cutoffs),
            "repeats",
            settings.n_repeats,
        )
        C.log(
            "period table",
            C.display_path(C.stage_map_path(config)),
            "cross-species windows",
            list(C.cross_windows(config)),
        )

        by_species: dict[str, list[ModelSpec]] = {}
        for spec in selected:
            by_species.setdefault(spec.species, []).append(spec)
        order = [s for s in C.SPECIES_ORDER if s in by_species] + [
            s for s in sorted(by_species) if s not in C.SPECIES_ORDER
        ]

        curve_rows: list[dict] = []
        null_rows: list[dict] = []
        band_rows: list[dict] = []
        summary_rows: list[dict] = []
        scan_rows: list[dict] = []
        failures: list[str] = []

        for species in order:
            with C.StepTimer(f"species {species}"):
                scan = scan_species(
                    config, species, genomes.get(species), universes.get(species, []), args.workers
                )
                scan_rows.append(scan.row)
                usable = scan.row["status"] == "ok" and scan.row["n_motif_peak_pairs"] > 0
                for spec in by_species[species]:
                    if not usable:
                        audit.append(
                            dict(
                                model=spec.model,
                                analysis=spec.analysis,
                                species=spec.species,
                                cutoff="",
                                reason=f"unusable_scan:{scan.row['status']}: the {species} FIMO scan "
                                "produced no motif-peak pairs, so no enrichment exists",
                            )
                        )
                        summary_rows.append(
                            dict(
                                model=spec.model,
                                analysis=spec.analysis,
                                species=spec.species,
                                celltype=spec.celltype,
                                period=spec.period,
                                window=spec.window,
                                model_index=spec.model_index,
                                n_ranked=spec.n_ranked,
                                status=f"unusable_scan:{scan.row['status']}",
                                cutoffs=[],
                            )
                        )
                        continue
                    try:
                        model_curve, model_null, model_band, model_summary = run_model(
                            spec,
                            rankings[spec.model],
                            scan,
                            catalog,
                            settings,
                            panels[spec.analysis],
                            audit,
                        )
                    except Exception as exc:  # noqa: BLE001 - audit, never abort the step
                        failures.append(f"{spec.model}: {type(exc).__name__}: {exc}")
                        audit.append(
                            dict(
                                model=spec.model,
                                analysis=spec.analysis,
                                species=spec.species,
                                cutoff="",
                                reason=f"error:{type(exc).__name__}:{exc}",
                            )
                        )
                        summary_rows.append(
                            dict(
                                model=spec.model,
                                analysis=spec.analysis,
                                species=spec.species,
                                celltype=spec.celltype,
                                period=spec.period,
                                window=spec.window,
                                model_index=spec.model_index,
                                n_ranked=spec.n_ranked,
                                status=f"error:{type(exc).__name__}",
                                cutoffs=[],
                            )
                        )
                        continue
                    curve_rows.extend(model_curve)
                    null_rows.extend(model_null)
                    band_rows.extend(model_band)
                    summary_rows.append(model_summary)
                    C.log(
                        " ",
                        spec.model,
                        model_summary["status"],
                        f"points={len(model_curve)}",
                    )

        curve = pd.DataFrame(curve_rows, columns=CURVE_COLUMNS).sort_values(
            ["analysis", "species", "model", "module", "cutoff"], kind="stable"
        )
        band = pd.DataFrame(band_rows, columns=BAND_COLUMNS).sort_values(
            ["model", "cutoff", "module"], kind="stable"
        )
        null = pd.DataFrame(null_rows, columns=NULL_COLUMNS).sort_values(
            ["model", "cutoff", "module", "repeat"], kind="stable"
        )
        audit_frame = pd.DataFrame(audit, columns=AUDIT_COLUMNS).sort_values(
            ["analysis", "species", "model", "cutoff", "reason"], kind="stable"
        )

        # Diagnostics first: a run that produced no curve still explains itself.
        C.save_table(audit_frame.reset_index(drop=True), paths["audit"], sep="\t")
        summary_payload = dict(
            step=STEP,
            claim_role=CLAIM_ROLE,
            config=C.display_path(Path(config["_path"])),
            period_table=C.display_path(C.stage_map_path(config)),
            module_map=C.display_path(C.module_map_path(config)),
            cross_species_windows=list(C.cross_windows(config)),
            single_species_windows=list(C.single_species_windows(config)),
            species_axis=C.species_axis(config),
            cutoffs=list(settings.cutoffs),
            panels={
                analysis: [
                    dict(
                        module=p.module,
                        level=p.level,
                        celltype=p.celltype,
                        components=list(p.components),
                        n_members=len(p.members),
                        members=list(p.members),
                    )
                    for p in rows
                ]
                for analysis, rows in panels.items()
            },
            analyses_without_modules=[
                a for a in analyses if a not in panels
            ],
            aggregate=SUPPORTED_AGGREGATE,
            n_repeats_requested=int(settings.n_repeats),
            n_repeats_source=settings.n_repeats_source,
            null_draw_rule=(
                "each repeat draws a foreground and a disjoint background of exactly "
                "the observed foreground size at that cut-off; a repeat that cannot "
                "place that many peaks on both sides is dropped and counted rather "
                "than used smaller, and the largest size any repeat reached is "
                "reported per (model, cut-off) as null_draw_size_max"
            ),
            band=list(settings.band),
            seed=int(settings.seed),
            fdr=float(settings.fdr),
            gc_caliper=float(settings.caliper),
            min_background_pairs=int(settings.min_pairs),
            n_peaks_reference=int(settings.n_peaks_reference),
            n_models_in_manifest=int(len(manifest)),
            n_models_dropped_by_species_filter=int(len(dropped)),
            # Every removal is named here with the criterion that caused it, so
            # the peak floor is visible in the summary and not only in the audit
            # table; `reason` is C.apply_species_filter's own `action` string.
            species_filter_drops=species_filter_drops,
            n_models_selected=int(len(selected)),
            models=summary_rows,
            scan=scan_rows,
            failures=failures,
        )
        C.save_json(summary_payload, paths["summary"])

        if curve.empty and panels:
            raise RuntimeError(
                "no (model, cut-off) pair produced an observed module score; "
                f"see {C.display_path(paths['audit'])} and {C.display_path(paths['summary'])}"
            )

        C.save_table(curve.reset_index(drop=True), paths["curve"], sep="\t")
        save_gzip_table(null.reset_index(drop=True), paths["null"])
        C.save_table(band.reset_index(drop=True), paths["band"], sep="\t")

        model_order = [row["model"] for row in summary_rows]
        manifest_rows: list[dict] = []
        for analysis, panel in panels.items():
            if analysis not in set(curve.analysis):
                continue
            stem = f"rank_curve_{safe_stem(analysis)}{suffix}"
            module_names = [p.module for p in panel]
            level_of = {p.module: p.level for p in panel}
            written = plot_analysis(
                analysis, module_names, curve, band, settings, out, stem, formats, dpi, model_order
            )
            panel_species = [
                s for s in C.SPECIES_ORDER if s in set(curve[curve.analysis == analysis].species)
            ]
            panel_species += sorted(
                set(curve[curve.analysis == analysis].species) - set(panel_species)
            )
            for module in module_names:
                for species in panel_species:
                    block = band[
                        (band.module == module)
                        & band.model.isin(
                            set(curve[(curve.analysis == analysis) & (curve.species == species)].model)
                        )
                    ]
                    if block.empty:
                        continue
                    n_members = int(
                        curve[
                            (curve.analysis == analysis)
                            & (curve.species == species)
                            & (curve.module == module)
                        ].n_members.max()
                    )
                    manifest_rows.append(
                        dict(
                            panel=f"{safe_stem(analysis)}__{safe_stem(module)}__{species}",
                            analysis=analysis,
                            module=module,
                            module_level=level_of[module],
                            species=species,
                            source_files=";".join(
                                [
                                    C.display_path(paths["curve"]),
                                    C.display_path(paths["band"]),
                                    C.display_path(paths["null"]),
                                ]
                            ),
                            statistic="median log2 OR over monomeric module members",
                            band=f"{settings.band[0]:g}-{settings.band[1]:g}% of "
                            f"{int(settings.n_repeats)} matched random-k draws",
                            n_models=int(
                                len(
                                    set(
                                        curve[
                                            (curve.analysis == analysis)
                                            & (curve.species == species)
                                        ].model
                                    )
                                )
                            ),
                            n_repeats_requested=int(settings.n_repeats),
                            n_repeats_used_min=int(block.n_repeats_used.min()),
                            cutoffs=";".join(str(int(k)) for k in settings.cutoffs),
                            figure_png=C.display_path(out / f"{stem}.png")
                            if "png" in formats
                            else "",
                            figure_pdf=C.display_path(out / f"{stem}.pdf")
                            if "pdf" in formats
                            else "",
                            caption=CAPTION_TEMPLATE.format(
                                analysis=analysis,
                                module=module,
                                module_level=level_of[module],
                                species=species,
                                n_members=n_members,
                                band_lo=f"{settings.band[0]:g}",
                                band_hi=f"{settings.band[1]:g}",
                                n_repeats=int(settings.n_repeats),
                            ),
                            claim_role=CLAIM_ROLE,
                            status="rendered",
                        )
                    )
            C.log("figure", analysis, "->", [C.display_path(p) for p in written])

        C.save_table(
            pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS), paths["manifest"], sep="\t"
        )

        C.log("audit rows", len(audit_frame), audit_frame.reason.str.split(":").str[0].value_counts().to_dict())
        C.log("curve rows", len(curve), "null rows", len(null), "band rows", len(band))
        if failures:
            raise RuntimeError(
                f"{len(failures)} failures recorded in {paths['audit'].name}: {failures[:5]}"
            )


if __name__ == "__main__":
    main()
