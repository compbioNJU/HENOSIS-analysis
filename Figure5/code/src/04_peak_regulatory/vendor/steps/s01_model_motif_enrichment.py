"""Step 2 -- model-derived motif profiles for every ranked model (readme 6/7).

WHY this step exists
--------------------
Every claim made later in this project is a statement about the *model*: the
peaks the model ranks highest carry a regulatory program, and that program is
what recovers cross-species developmental correspondence. Such a claim is only
falsifiable if the Top-k peaks are converted into one fixed, cell-type- and
stage-free readout that later steps (s02, s06, s08, s10) can compare without
querying the model again. This step produces that readout: 879 motif log2
odds-ratios per model against a GC/length matched background drawn from the
model's own ranked universe, plus the matched random-k null that makes "the
ranking did this, not the 10k HVG set" testable.

Design consequences of that purpose:

  * the FIMO scan is per species over the union of that species' ranked
    universes, so it happens once and shares its cache with
    script/warm_scan_cache.py;
  * all 879 motifs are reported per model, significant or not, because
    downstream similarity is computed on the whole profile;
  * the random control is streamed to disk repeat by repeat, so memory stays
    flat over a 200-repeat x 879-motif x 35-model control table. A repeat is a
    matched draw, not a sample of a fixed size: `common.draw_disjoint_matched_sets`
    needs two disjoint partners per template peak and drops the peaks that find
    none, so the draw legitimately ends up shorter than the size that was
    requested. Each row therefore carries the requested size, the size the repeat
    was constructed at and the size the matcher actually achieved, and
    model_audit.tsv carries the per-model min/max/mean of the achieved size so
    the shortfall is readable without recomputing it;
  * the unsuffixed artefact names belong to the run over the whole manifest --
    the run every downstream step reads under those names. The debug filters
    (--species, --limit-models) and --topk all contribute to the file-name
    suffix, so a narrow run writes its own tables instead of overwriting those,
    and every run also writes run_scope<suffix>.tsv recording which slice of the
    manifest its tables cover;
  * nothing is dropped silently: empty models, sequence-less peaks, unmatched
    foreground peaks and skipped control repeats all land in model_audit.tsv;
  * the model *population* is part of the analysis definition, not of this
    script: when the config declares a species-admission block -- v2's
    `eligibility` or v1's `species_filter`, read by the same
    `common.min_cells_per_species` / `min_ranked_peaks` / `apply_species_filter`
    the rest of the pipeline uses -- a model whose species has fewer cells of its
    target type than the cell floor, or whose ranking is shorter than the peak
    floor, does not enter that analysis. That is a different kind of exclusion
    from the ones above -- the dropped models have no run at all, so they cannot
    appear in model_audit.tsv, which carries one row per *run* model. They are
    therefore named with the cell count and peak count that removed them in a
    sibling `species_filter_audit<suffix>.tsv`, and counted in the run scope
    record. A config without either block declares no such exclusion and writes
    no such file, which is what leaves a v1-style run byte-identical;
  * the FIMO scan universe is taken from the manifest *before* that filter. The
    filter decides which models are reported, never which sequence is scanned, so
    a species keeps the union of all its models' ranked peaks -- dropping bonobo
    Purkinje must not shrink bonobo's universe and cold-start the shared cache --
    and v1 and v2 scan the same per-species universes.

Only the model *ranking* enters this step. No h5ad, no chromVAR, no gene map.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

STEP = "01_model_motif"

SCORE_COLUMNS = [
    "analysis",
    "species",
    "celltype",
    "period",
    "model",
    "topk",
    "motif_id",
    "motif_name",
    "fg_hits",
    "bg_hits",
    "n_fg",
    "n_bg",
    "odds_ratio",
    "log2_OR",
    "pvalue",
    "FDR",
    "significant",
]

AUDIT_COLUMNS = [
    "analysis",
    "species",
    "celltype",
    "period",
    "model",
    "model_index",
    "topk",
    "n_ranked",
    "n_top",
    "n_top_valid",
    "n_top_no_sequence",
    "n_pool_raw",
    "n_pool",
    "n_pool_no_sequence",
    "n_background",
    "background_source",
    "max_gc_difference",
    "n_fg_matched",
    "n_fg_unmatched",
    "meets_min_background_pairs",
    "n_null_repeats_requested",
    "n_null_repeats_ok",
    "null_n_peaks_requested",
    "null_n_peaks",
    "null_n_fg_min",
    "null_n_fg_max",
    "null_n_fg_mean",
    "null_n_bg_mean",
    "null_vs_observed_fg_delta",
    "null_status",
    "null_require_exact_peaks",
    "null_target_peaks",
    "null_n_valid_repeats",
    "null_n_dropped",
    "null_max_shortfall",
    "null_min_valid_repeats",
    "filtered_run",
    "n_models_selected",
    "status",
]

SCOPE_COLUMNS = ["item", "value"]

# The schema of species_filter_audit<suffix>.tsv, pinned here so the file has one
# shape whether or not anything was dropped: `common.apply_species_filter` returns
# a frame without the `model` column when the config declares no filter at all.
# Both floors of the v2 `eligibility` block are carried, so a model dropped for a
# short ranking names the ranking length and the floor, not only its cell count.
SPECIES_FILTER_COLUMNS = [
    "analysis",
    "species",
    "celltype",
    "model",
    "n_cells",
    "min_cells",
    "n_ranked",
    "min_ranked_peaks",
    "action",
]

SCAN_COLUMNS = [
    "species",
    "genome_assembly",
    "fasta",
    "n_universe_peaks",
    "n_scan_peaks",
    "n_no_sequence",
    "n_motif_peak_pairs",
    "n_motifs_with_hits",
    "fimo_threshold",
    "fimo_bin",
    "workers",
    "status",
]

SUMMARY_COLUMNS = [
    "model",
    "analysis",
    "species",
    "celltype",
    "period",
    "topk",
    "background_source",
    "null_status",
    "null_n_peaks",
    "n_repeats_requested",
    "n_repeats_used",
    "null_n_valid_repeats",
    "null_n_dropped",
    "min_valid_repeats",
    "n_sig_observed",
    "n_sig_null_mean",
    "n_sig_null_sd",
    "spearman_observed_vs_null_mean",
    "spearman_observed_vs_null_sd",
    "spearman_observed_vs_null_p",
    "spearman_observed_vs_null_side",
    "frac_null_nsig_ge_observed",
]

EMPTY_COORDS = pd.DataFrame(
    {
        "peak": pd.Series(dtype=object),
        "length": pd.Series(dtype="int64"),
        "gc": pd.Series(dtype="float64"),
    }
)


# --------------------------------------------------------------------------
# per-model bookkeeping
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Everything about a model that is fixed before any sequence is read."""

    model: str
    model_index: int
    analysis: str
    species: str
    celltype: str
    period: str
    n_ranked: int
    is_empty: bool


@dataclass(frozen=True)
class Settings:
    topk: int
    fdr: float
    min_pairs: int
    caliper: float
    n_repeats: int
    seed: int
    null_n_peaks: int
    # v3: the matched-random null is exact-K -- a repeat whose draw does not place
    # exactly `null_n_peaks` peaks on BOTH sides does not enter the null -- and a
    # model whose valid repeats fall below `min_valid_repeats` is recorded as
    # underpowered rather than rescued by a smaller control.
    require_exact_n_peaks: bool = False
    min_valid_repeats: int = 0


@dataclass(frozen=True)
class RunScope:
    """Which slice of the manifest a run covers, and therefore what it writes.

    Carried into every audit row so that a table produced by a filtered debug
    run is self-identifying even when it is read without its file name.
    """

    filtered: bool
    n_models_selected: int


@dataclass
class SpeciesScan:
    """One species' cached FIMO scan plus the sequence bookkeeping it implies."""

    species: str
    hits: dict[str, set[str]]
    row: dict
    sequences: pd.DataFrame | None = None
    peaks: tuple[str, ...] = ()


def with_model_index(manifest: pd.DataFrame) -> pd.DataFrame:
    """Stamp the row position of each model onto the manifest as loaded.

    `model_index` must be the row position in the *unfiltered* manifest, because
    it is the second component of that model's RNG seed `[config.seed,
    model_index]`: within one analysis definition the seed has to belong to the
    model, not to its position in whatever subset a particular run happens to
    cover. Fixing it here, before the config's species filter and before --species
    / --limit-models, means dropping bonobo Purkinje does not renumber -- and so
    does not re-seed -- every model after it.

    The index is nonetheless the position in *this config's* manifest, so its
    basis is the analysis definition and not the model: two definitions that rank
    different sets of models (v1's 32, v2's 22) place the same model at a
    different index and therefore seed it differently, and the matched background
    a model draws is not comparable across configs. That basis is recorded in the
    run's own record as `model_index_basis` (see `scope_frame`); a config-independent
    seed would instead re-seed v1 and break its published tables, so the derivation
    itself must not change here.
    """
    return manifest.assign(model_index=np.arange(len(manifest)))


def apply_config_species_filter(
    config: dict, manifest: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    """Drop the models whose species is too sparse for that model's target type.

    Returns (kept manifest, audit frame or None, info dict). The filter belongs to
    the analysis *definition* -- v2's `eligibility` block or v1's `species_filter`
    block, the same `config.get("eligibility") or config.get("species_filter")`
    expression `common.min_cells_per_species`, `common.min_ranked_peaks` and
    `common.apply_species_filter` already read -- not to a debugging flag, so it
    applies to the canonical run and to every narrow run alike. v2 declares ONLY
    `eligibility`, so reading `species_filter` alone would silently skip the v2
    rule; both keys are therefore read here, in the same precedence order common
    uses. A config that declares neither block gets its manifest back untouched
    with no audit frame and no info, and the caller writes no extra artefact, so
    it stays byte-identical.

    When the block exists, every dropped model is named in the returned frame
    with BOTH floors that removed it -- the target cell count against
    `min_target_cells` and the ranking length against `min_ranked_peaks`, each
    reported with its own bound. The frame is written beside the run's tables
    rather than into model_audit.tsv, which holds one row per model the run
    actually scored: a dropped model has no status, no background and no control
    to report, and mixing the two would make "one row per scored model" false for
    every downstream reader.
    """
    block = config.get("eligibility") or config.get("species_filter")
    if not block:
        return manifest, None, {}

    kept, audit = C.apply_species_filter(config, manifest)
    audit = audit.reindex(columns=SPECIES_FILTER_COLUMNS)
    info = {
        "block": "eligibility" if config.get("eligibility") else "species_filter",
        # Read through common, so a key the config happens to spell differently
        # (`min_target_cells` in v2) still reports the floor actually applied.
        "min_cells_per_species": int(C.min_cells_per_species(config)),
        "min_ranked_peaks": int(C.min_ranked_peaks(config)),
        "celltype_column": str(block.get("celltype_column", "Org_Celltype")),
        "n_models_before": int(len(manifest)),
        "n_models_after": int(len(kept)),
        "n_models_dropped": int(len(audit)),
    }
    return kept, audit, info


def model_specs(manifest: pd.DataFrame) -> list[ModelSpec]:
    """One spec per manifest row, in the manifest order.

    `model_index` is read off the manifest (see `with_model_index`) instead of
    being re-derived from the position in this frame, so that a manifest some
    upstream step has narrowed still yields each model its own seed.
    """
    return [
        ModelSpec(
            model=row.model,
            model_index=int(row.model_index),
            analysis=row.analysis,
            species=row.species,
            celltype=row.celltype,
            period=row.period,
            n_ranked=int(row.n_ranked),
            is_empty=bool(row.is_empty),
        )
        for row in manifest.itertuples()
    ]


def base_audit(spec: ModelSpec, settings: Settings, scope: RunScope) -> dict:
    return {
        "analysis": spec.analysis,
        "species": spec.species,
        "celltype": spec.celltype,
        "period": spec.period,
        "model": spec.model,
        "model_index": spec.model_index,
        "topk": settings.topk,
        "n_ranked": spec.n_ranked,
        "n_top": 0,
        "n_top_valid": 0,
        "n_top_no_sequence": 0,
        "n_pool_raw": 0,
        "n_pool": 0,
        "n_pool_no_sequence": 0,
        "n_background": 0,
        "background_source": "",
        "max_gc_difference": np.nan,
        "n_fg_matched": 0,
        "n_fg_unmatched": 0,
        "meets_min_background_pairs": False,
        "n_null_repeats_requested": 0,
        "n_null_repeats_ok": 0,
        "null_n_peaks_requested": int(settings.null_n_peaks),
        "null_n_peaks": 0,
        "null_n_fg_min": np.nan,
        "null_n_fg_max": np.nan,
        "null_n_fg_mean": np.nan,
        "null_n_bg_mean": np.nan,
        "null_vs_observed_fg_delta": np.nan,
        "null_status": "not_requested",
        "null_require_exact_peaks": bool(settings.require_exact_n_peaks),
        "null_target_peaks": 0,
        "null_n_valid_repeats": 0,
        "null_n_dropped": 0,
        "null_max_shortfall": 0,
        "null_min_valid_repeats": int(settings.min_valid_repeats),
        "filtered_run": scope.filtered,
        "n_models_selected": scope.n_models_selected,
        "status": "pending",
    }


def null_unavailable(audit: dict, settings: Settings, null: NullWriter | None) -> None:
    """Record a control that was requested but never reached a single draw.

    `not_requested` must keep meaning "the operator asked for no control"
    (`--repeats 0`), so a model that is skipped before the control can even be
    attempted -- empty ranking, no valid foreground, no matched pairs, unusable
    scan -- is marked `unavailable:<model status>` instead. Otherwise a run that
    asked for 200 repeats and a run that asked for none are indistinguishable in
    model_audit.tsv for exactly the models a reader most wants to check.
    """
    if null is not None:
        audit["n_null_repeats_requested"] = int(settings.n_repeats)
        audit["null_status"] = f"unavailable:{audit['status']}"


def coords_frame(coords: pd.DataFrame, peaks: list[str]) -> pd.DataFrame:
    """(peak, length, gc) rows in the given order; dtype-stable when empty."""
    peaks = list(peaks)
    if not peaks:
        return EMPTY_COORDS.copy()
    return coords.loc[peaks].reset_index()


def tag_scores(frame: pd.DataFrame, spec: ModelSpec, topk: int) -> pd.DataFrame:
    out = frame.copy()
    for name, value in reversed(
        [
            ("analysis", spec.analysis),
            ("species", spec.species),
            ("celltype", spec.celltype),
            ("period", spec.period),
            ("model", spec.model),
            ("topk", topk),
        ]
    ):
        out.insert(0, name, value)
    return out[SCORE_COLUMNS]


# --------------------------------------------------------------------------
# null table streaming
# --------------------------------------------------------------------------


class NullWriter:
    """Appends the per-repeat null log2_OR profile to one gzipped TSV.

    Written repeat by repeat, so the 200 x 879 rows of one model never have to
    coexist in memory with those of another model.

    Column semantics of the draw sizes, from what was asked for down to what the
    matcher actually placed:

      * `null_n_peaks_requested` -- `random_control.n_peaks` as frozen in the
        config: the nominal control size the run asked for, which for v3 is the
        arm's cut-off K (500 / 1000 / 2000). Constant per run, but carried on
        every row so the requested size travels with the table. This is the size a
        consumer comparing against `random_control.n_peaks` sees.
      * `n_fg`/`n_bg` -- under v3, the size the repeat was *constructed* at, which
        is `min(n_fg_matched, n_peaks, len(template))`: the width of the OBSERVED
        Fisher table, not K. Every emitted repeat has exactly this width on both
        sides, because a short draw is dropped rather than accepted. (In v1/v2
        this column held the requested size capped at the observed Top-k size, and
        the achieved draw was routinely shorter than it.)
      * `n_fg_drawn`/`n_bg_drawn` -- the peaks actually placed. Under v3 these
        EQUAL `n_fg`/`n_bg` on every emitted row by construction, since the
        producer drops any repeat that comes up short; they are retained so the
        equality is checkable from the table alone.
      * `null_n_peaks` -- repeats the constructed size, so the target is
        readable without knowing the column convention.

    The shortfall between requested, target and achieved draw is also summarised
    per model in `model_audit.tsv` (`null_n_peaks_requested`, `null_n_peaks`,
    `null_n_fg_min/max/mean`, `n_fg_matched` and `null_vs_observed_fg_delta`), so
    a reader does not have to decompress this table to see it.
    """

    COLUMNS = [
        "model",
        "repeat",
        "motif_id",
        "log2_OR",
        "pvalue",
        "FDR",
        "significant",
        "n_fg",
        "n_bg",
        "n_fg_drawn",
        "n_bg_drawn",
        "null_n_peaks",
        "null_n_peaks_requested",
    ]

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = gzip.open(path, "wt", newline="")
        self.writer = csv.writer(self.handle, delimiter="\t", lineterminator="\n")
        self.writer.writerow(self.COLUMNS)

    def write_model(
        self,
        model: str,
        frame: pd.DataFrame,
        alpha: float,
        draw_size: int,
        requested_size: int | None = None,
    ) -> pd.Series:
        """Write one model's repeats; return n_significant per repeat.

        `draw_size` is the size the repeats were constructed at; `requested_size`
        is the frozen control size they were capped from (`None` when the caller
        has no larger target, i.e. the draw was already at the requested size).
        """
        counts: dict[int, int] = {}
        requested = int(draw_size if requested_size is None else requested_size)
        for repeat, group in frame.groupby("repeat", sort=True):
            annotated = C.add_significance(group, alpha)
            block = pd.DataFrame(
                {
                    "model": model,
                    "repeat": int(repeat),
                    "motif_id": annotated.motif_id.to_numpy(),
                    "log2_OR": annotated.log2_OR.to_numpy(),
                    "pvalue": annotated.pvalue.to_numpy(),
                    "FDR": annotated.FDR.to_numpy(),
                    "significant": annotated.significant.to_numpy(),
                    "n_fg": int(draw_size),
                    "n_bg": int(draw_size),
                    "n_fg_drawn": annotated.n_fg.to_numpy(),
                    "n_bg_drawn": annotated.n_bg.to_numpy(),
                    "null_n_peaks": int(draw_size),
                    "null_n_peaks_requested": requested,
                }
            )
            block[self.COLUMNS].to_csv(self.handle, sep="\t", header=False, index=False)
            counts[int(repeat)] = int(block.significant.sum())
        return pd.Series(counts, dtype=int).sort_index()

    def close(self) -> None:
        self.handle.close()


# --------------------------------------------------------------------------
# species-level FIMO scan (shared cache with warm_scan_cache.py)
# --------------------------------------------------------------------------


def scan_species(
    config: dict,
    species: str,
    genome: C.SpeciesGenome | None,
    universe: list[str],
    workers: int,
) -> SpeciesScan:
    row = {
        "species": species,
        "genome_assembly": genome.assembly if genome is not None else "",
        "fasta": str(genome.fasta) if genome is not None else "",
        "n_universe_peaks": len(universe),
        "n_scan_peaks": 0,
        "n_no_sequence": 0,
        "n_motif_peak_pairs": 0,
        "n_motifs_with_hits": 0,
        "fimo_threshold": float(config["motif_database"]["fimo_threshold"]),
        "fimo_bin": C.FIMO_BIN,
        "workers": workers,
        "status": "pending",
    }
    if genome is None:
        row["status"] = "no_genome"
        return SpeciesScan(species, {}, row)
    if not universe:
        row["status"] = "empty_universe"
        return SpeciesScan(species, {}, row)

    sequences = C.read_sequences(config, genome, universe, use_cache=True)
    n_valid = int(sequences.valid.astype(bool).sum()) if len(sequences) else 0
    row["n_scan_peaks"] = n_valid
    row["n_no_sequence"] = int(len(sequences)) - n_valid
    if n_valid == 0:
        row["status"] = "no_valid_sequence"
        return SpeciesScan(species, {}, row, sequences, tuple(sequences.peak))

    hits = C.scan_motifs(config, genome, sequences, workers=workers, force=False)
    row["n_motif_peak_pairs"] = int(sum(len(peaks) for peaks in hits.values()))
    row["n_motifs_with_hits"] = len(hits)
    row["status"] = "ok"
    return SpeciesScan(species, hits, row, sequences, tuple(sequences.peak))


# --------------------------------------------------------------------------
# one model
# --------------------------------------------------------------------------


def run_model(
    spec: ModelSpec,
    ranking: pd.DataFrame,
    scan: SpeciesScan,
    catalog: pd.DataFrame,
    config: dict,
    settings: Settings,
    scope: RunScope,
    rng: np.random.Generator,
    null: NullWriter | None,
) -> tuple[pd.DataFrame | None, dict, dict | None]:
    audit = base_audit(spec, settings, scope)
    if spec.is_empty:
        audit["status"] = "empty_ranking"
        null_unavailable(audit, settings, null)
        return None, audit, None

    sequences = scan.sequences
    assert sequences is not None
    valid = sequences.valid.astype(bool).to_numpy()
    valid_peaks = set(sequences.peak.to_numpy()[valid])
    coords = sequences[valid].set_index("peak")[["length", "gc"]]

    top = ranking[ranking["rank"] <= settings.topk]
    audit["n_top"] = int(len(top))
    top_valid = top.peak[top.peak.isin(valid_peaks)].tolist()
    audit["n_top_valid"] = len(top_valid)
    audit["n_top_no_sequence"] = int(len(top)) - len(top_valid)
    if not top_valid:
        audit["status"] = "no_valid_foreground"
        null_unavailable(audit, settings, null)
        return None, audit, None

    top_frame = coords_frame(coords, top_valid)
    top_peak_set = set(top.peak)

    own = ranking[ranking["rank"] > settings.topk]
    audit["n_pool_raw"] = int(len(own))
    pool_frame = coords_frame(coords, own.peak[own.peak.isin(valid_peaks)].tolist())
    fg_peaks, bg_peaks, pairs = C.matched_foreground_background(
        top_frame, pool_frame, rng, settings.caliper
    )
    background_source = "same_model_non_topk"

    if len(fg_peaks) < settings.min_pairs:
        fallback = [p for p in scan.peaks if p in valid_peaks and p not in top_peak_set]
        audit["n_pool_raw"] = len(scan.peaks) - len(top)
        pool_frame = coords_frame(coords, fallback)
        background_source = "species_universe_fallback"
        fg_peaks, bg_peaks, pairs = C.matched_foreground_background(
            top_frame, pool_frame, rng, settings.caliper
        )

    matched = pairs[pairs.bg_peak != ""]
    audit["background_source"] = background_source
    audit["n_pool"] = int(len(pool_frame))
    audit["n_pool_no_sequence"] = max(0, int(audit["n_pool_raw"]) - int(len(pool_frame)))
    audit["n_fg_matched"] = len(fg_peaks)
    audit["n_background"] = len(bg_peaks)
    audit["n_fg_unmatched"] = len(top_valid) - len(fg_peaks)
    audit["max_gc_difference"] = float(matched.gc_difference.max()) if len(matched) else np.nan
    audit["meets_min_background_pairs"] = bool(len(fg_peaks) >= settings.min_pairs)
    if not fg_peaks:
        audit["status"] = "no_matched_pairs"
        null_unavailable(audit, settings, null)
        return None, audit, None

    scores = tag_scores(
        C.add_significance(C.enrichment(catalog, scan.hits, fg_peaks, bg_peaks), settings.fdr),
        spec,
        settings.topk,
    )
    audit["status"] = "ok"

    if null is None:
        audit["null_status"] = "not_requested"
        return scores, audit, None

    audit["n_null_repeats_requested"] = int(settings.n_repeats)
    # A model may rank fewer peaks than the frozen control size (Purkinje__bonobo
    # ranks 298): drawing 500 random peaks would then be incomparable with the
    # observed profile, so the draw is capped at the observed Top-k size. The cap
    # is deliberately NOT tightened to the matched foreground size: n_fg/n_bg in
    # the null table is what s06_purkinje_model.load_null compares against
    # random_control.n_peaks, so a cap below the frozen size would make every
    # repeat of every well-ranked model unusable to that consumer. The three
    # sizes therefore differ -- requested (config), target (this cap), achieved
    # (what draw_disjoint_matched_sets could place, always <= target) -- and all
    # of them, plus the observed matched size they are compared against, are
    # recorded per model below.
    n_peaks_eff = int(min(settings.null_n_peaks, len(top_frame)))
    audit["null_n_peaks"] = n_peaks_eff
    # The size the null must reproduce is the OBSERVED table's width, not K. K is
    # the ranking cut-off; this model's own matched foreground is what a Fisher
    # comparison is actually built from, and it is regularly below K (up to a
    # third of the Top-K peaks can find no matched partner). Drawing the null at K
    # would compare the observation against a deeper sample.
    null_target = int(audit["n_fg_matched"])
    audit["null_target_peaks"] = null_target
    try:
        null_frame, _, draw = C.random_null_enrichment(
            config,
            catalog,
            scan.hits,
            top_frame,
            pool_frame,
            rng,
            n_repeats=settings.n_repeats,
            n_peaks=n_peaks_eff,
            caliper=settings.caliper,
            target=null_target,
        )
    except Exception as exc:  # noqa: BLE001 - the observed profile stays valid
        audit["null_status"] = f"error:{type(exc).__name__}:{exc}"
        C.log("WARNING null control failed", spec.model, audit["null_status"])
        return scores, audit, None

    # Every miss is recorded before anything is decided from the draw: an exact-K
    # null that could not be drawn is a property of the model's pool size, and it
    # has to be legible in model_audit.tsv without recomputing the draw.
    audit["null_n_valid_repeats"] = int(draw.n_valid)
    audit["null_n_dropped"] = int(draw.n_dropped)
    audit["null_max_shortfall"] = int(draw.max_shortfall)
    audit["n_null_repeats_ok"] = int(draw.n_valid)
    audit["null_status"] = draw.status()

    if draw.n_valid == 0:
        # NOT an error, and deliberately not fatal: one model whose non-top-k pool
        # cannot supply a two-sided exact-K draw (it needs 2K distinct pool peaks)
        # must not take the whole analysis down with it. The model is reported as
        # `no_draw` and carries no null; downstream gates read that status and
        # report UNDERPOWERED rather than PASS or FAIL.
        C.log(
            "WARNING null empty",
            spec.model,
            f"status={audit['null_status']}",
            f"requested={draw.n_requested} dropped={draw.n_dropped}",
            f"target={draw.target} pool={audit['n_pool']}",
        )
        summary = blank_summary(spec, settings, audit)
        summary["null_status"] = audit["null_status"]
        summary["n_repeats_requested"] = int(draw.n_requested)
        summary["n_repeats_used"] = 0
        return scores, audit, summary

    if audit["null_status"] == "underpowered":
        C.log(
            "WARNING null underpowered",
            spec.model,
            f"{draw.n_valid}/{draw.n_requested} valid < min_valid_repeats={draw.min_valid_repeats}",
            f"worst draw {draw.max_shortfall} of {draw.target}",
        )

    sig_counts = null.write_model(
        spec.model, null_frame, settings.fdr, draw.target, settings.null_n_peaks
    )
    audit["null_n_fg_min"] = float(null_frame.n_fg.min())
    audit["null_n_fg_max"] = float(null_frame.n_fg.max())
    audit["null_n_fg_mean"] = float(null_frame.n_fg.mean())
    audit["null_n_bg_mean"] = float(null_frame.n_bg.mean())
    # Signed gap between the drawn control foreground and the observed foreground
    # it is compared against. Under the exact-K contract this is 0 whenever the
    # null is powered: the draw is the same size as the observed Top-k
    # foreground, which is the whole point of the contract. A non-zero value
    # therefore marks a model whose observed side was itself short (a
    # sequence-less peak), not the usual case it was under v2.
    audit["null_vs_observed_fg_delta"] = float(null_frame.n_fg.mean()) - float(audit["n_fg_matched"])
    summary = summarize_null(
        spec, settings, background_source, n_peaks_eff, null_frame, sig_counts, scores, catalog
    )
    summary["null_status"] = audit["null_status"]
    summary["null_n_valid_repeats"] = int(draw.n_valid)
    summary["null_n_dropped"] = int(draw.n_dropped)
    summary["min_valid_repeats"] = int(draw.min_valid_repeats)
    return scores, audit, summary


def blank_summary(spec: ModelSpec, settings: Settings, audit: dict) -> dict:
    """A summary row for a model whose observed/null comparison does not exist.

    Emitted so that `random_null_summary.tsv` carries exactly one row per model
    and the absence of a control is visible in the file, not implied by a
    missing line.
    """
    row = {name: np.nan for name in SUMMARY_COLUMNS}
    row.update(
        {
            "model": spec.model,
            "analysis": spec.analysis,
            "species": spec.species,
            "celltype": spec.celltype,
            "period": spec.period,
            "topk": settings.topk,
            "background_source": audit["background_source"],
            "null_status": audit["null_status"],
            "null_n_peaks": audit["null_n_peaks"],
            "n_repeats_requested": audit["n_null_repeats_requested"],
            "n_repeats_used": audit["n_null_repeats_ok"],
        }
    )
    return row


def summarize_null(
    spec: ModelSpec,
    settings: Settings,
    background_source: str,
    n_peaks_eff: int,
    null_frame: pd.DataFrame,
    sig_counts: pd.Series,
    scores: pd.DataFrame,
    catalog: pd.DataFrame,
) -> dict:
    """Observed profile vs the matched random-k null, for one model.

    `spearman_observed_vs_null_*` measures how similar the Top-k profile is to
    each null profile. Its empirical p uses the same statistic evaluated among
    the nulls themselves (each null's mean rho against the other nulls): it asks
    whether the model profile relates to random draws the way random draws relate
    to each other.

    The test is two-sided, and must be: a Top-k profile that really is different
    from a random draw sits in the LOWER tail of that reference (its rho against
    the nulls is below the nulls' rho among themselves), so a greater-tail p
    would report 1.0 for exactly the models with the strongest model signal --
    the opposite of the intended reading. `spearman_observed_vs_null_side`
    records which tail the observed statistic fell in so the p can never be read
    backwards.
    """
    matrix = (
        null_frame.pivot(index="motif_id", columns="repeat", values="log2_OR")
        .reindex(catalog.motif_id.to_numpy())
        .to_numpy(dtype=float)
    )
    n_used = matrix.shape[1]
    observed = scores.log2_OR.to_numpy(dtype=float)
    n_sig_observed = int(scores.significant.sum())

    rhos = np.array([C.spearman(observed, matrix[:, j])[0] for j in range(n_used)], dtype=float)
    mean_rho = float(np.mean(rhos)) if n_used else np.nan
    sd_rho = float(np.std(rhos, ddof=1)) if n_used > 1 else np.nan

    p_value = np.nan
    side = "not_evaluated"
    if n_used >= 2:
        pair = np.full((n_used, n_used), np.nan)
        for i in range(n_used):
            for j in range(i + 1, n_used):
                rho = C.spearman(matrix[:, i], matrix[:, j])[0]
                pair[i, j] = pair[j, i] = rho
        with np.errstate(invalid="ignore"):
            row_mean = np.nanmean(pair, axis=1)
        finite = np.isfinite(row_mean)
        if int(finite.sum()) >= 2:
            # Two-sided against the null-vs-null reference: the observed statistic
            # is compared by absolute deviation from the reference median, so a
            # model profile that is unusually *unlike* random draws (the expected
            # outcome) scores small instead of saturating at 1.0.
            centre = float(np.median(row_mean[finite]))
            deviation = np.abs(row_mean[finite] - centre)
            observed_deviation = abs(mean_rho - centre)
            p_value = float((np.sum(deviation >= observed_deviation) + 1) / (finite.sum() + 1))
            side = "lower" if mean_rho < centre else "upper"
            if mean_rho == centre:
                side = "equal"

    n_sig_null = sig_counts.to_numpy(dtype=float)
    return {
        "model": spec.model,
        "analysis": spec.analysis,
        "species": spec.species,
        "celltype": spec.celltype,
        "period": spec.period,
        "topk": settings.topk,
        "background_source": background_source,
        "null_n_peaks": n_peaks_eff,
        "n_repeats_requested": int(settings.n_repeats),
        "n_repeats_used": int(n_used),
        "n_sig_observed": n_sig_observed,
        "n_sig_null_mean": float(n_sig_null.mean()) if len(n_sig_null) else np.nan,
        "n_sig_null_sd": float(n_sig_null.std(ddof=1)) if len(n_sig_null) > 1 else np.nan,
        "spearman_observed_vs_null_mean": mean_rho,
        "spearman_observed_vs_null_sd": sd_rho,
        "spearman_observed_vs_null_p": p_value,
        "spearman_observed_vs_null_side": side,
        "frac_null_nsig_ge_observed": (
            float(np.mean(n_sig_null >= n_sig_observed)) if len(n_sig_null) else np.nan
        ),
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def artefact_suffix(topk: int | None, species: list[str], limit_models: int | None) -> str:
    """Name suffix describing exactly which slice of the manifest a run covers.

    The unsuffixed names are the frozen artefacts of the full 32-model run that
    every downstream step reads. --topk, --species and --limit-models each
    narrow a run, so each of them contributes a part here and a debug run writes
    `all_model_motif_scores_subhuman_N2.tsv` instead of overwriting the frozen
    table in place. Every artefact a run writes -- including the run_scope record
    that states the filter -- carries this suffix, so no two runs share a file
    unless they cover the same slice of the manifest.
    """
    parts: list[str] = []
    if topk is not None:
        parts.append(f"top{int(topk)}")
    if species or limit_models is not None:
        scope = "sub" + ("-".join(sorted(species)) if species else "all")
        if limit_models is not None:
            scope += f"_N{int(limit_models)}"
        parts.append(scope)
    return "".join(f"_{part}" for part in parts)


def scope_frame(
    species_filter: list[str],
    limit_models: int | None,
    topk_arg: int | None,
    settings: Settings,
    scope: RunScope,
    selected: list[ModelSpec],
    n_models_available: int,
    suffix: str,
    run_null: bool,
    species_info: dict | None = None,
) -> pd.DataFrame:
    """The run-scope record written beside the artefacts of one run.

    WHY a file and not just the suffix: a table gets copied, quoted or unzipped
    away from its name, and model_audit.tsv can only say *that* the run was
    filtered, not what the filter was. This record states, inside the run's own
    results directory, exactly which slice of the manifest its tables cover (the
    flags as typed, the selected model list, the control settings), so a partial
    table can never be mistaken for the frozen full-manifest run. It is written
    under the same suffix as the tables it describes, so a debug run records its
    scope without touching the canonical scope file.

    `species_info` is the analysis definition's species-admission block -- v2's
    `eligibility` or v1's `species_filter` -- when the config declares one. Its
    rows are appended only in that case, so a scope record of a config with
    neither block is byte-for-byte what it always was; a block-bearing record
    states the block's name, BOTH floors (the target-cell floor and the ranked-peak
    floor), the column the cell count was read from and how many models it
    removed, which is the run-level counterpart of
    species_filter_audit<suffix>.tsv naming them.

    The same gate carries `model_index_basis`, because the same condition makes it
    worth stating: a config that filters the model population is a config whose
    manifest -- and so whose per-model RNG seed, `[config.seed, model_index]` --
    is not v1's. The row does not change any seed; it says out loud that
    `model_index` is the model's position in *this config's* manifest, so a
    consumer reading a v2 table next to a v1 one is told the seeds are
    config-relative and the two are not RNG-matched.
    """
    rows: list[tuple[str, object]] = [
        ("filtered_run", bool(scope.filtered)),
        ("filter_species", ",".join(sorted(species_filter)) if species_filter else "(all)"),
        ("filter_limit_models", "" if limit_models is None else int(limit_models)),
        ("filter_topk", "" if topk_arg is None else int(topk_arg)),
        ("topk_used", int(settings.topk)),
        ("n_models_in_manifest", int(n_models_available)),
        ("n_models_selected", int(scope.n_models_selected)),
        ("models_selected", ";".join(spec.model for spec in selected)),
        ("output_suffix", suffix or "(none)"),
        ("null_control", "on" if run_null else "off"),
        ("n_repeats_requested", int(settings.n_repeats)),
        ("null_n_peaks_requested", int(settings.null_n_peaks)),
        ("fdr", float(settings.fdr)),
        ("gc_caliper", float(settings.caliper)),
    ]
    if species_info:
        rows.extend(
            [
                # Where every model_index in this run came from. Written with the
                # species_filter rows and for the same reason: this config's model
                # population is not v1's, so `[seed, model_index]` -- and with it
                # the matched background each model draws -- is a property of the
                # definition, not of the model, and the run has to say so. v1 ran
                # the frozen definition and writes no such row, which is what
                # leaves its scope record byte-identical.
                ("model_index_basis", "manifest_position"),
                ("species_filter_block", str(species_info["block"])),
                ("species_filter_min_cells", int(species_info["min_cells_per_species"])),
                ("species_filter_min_ranked_peaks", int(species_info["min_ranked_peaks"])),
                ("species_filter_celltype_column", str(species_info["celltype_column"])),
                ("n_models_before_species_filter", int(species_info["n_models_before"])),
                ("n_models_after_species_filter", int(species_info["n_models_after"])),
                ("n_models_dropped_by_species_filter", int(species_info["n_models_dropped"])),
            ]
        )
    return pd.DataFrame(rows, columns=SCOPE_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    C.add_common_args(parser)
    parser.add_argument("--workers", type=int, default=16, help="FIMO workers (cold cache only)")
    parser.add_argument("--species", nargs="*", default=None, help="restrict to these species")
    parser.add_argument(
        "--limit-models", type=int, default=None, help="debug aid: only the first N models"
    )
    args = parser.parse_args()
    C.main_guard()

    config = C.load_config(args.config)
    if args.force:
        C.log("--force ignored: the FIMO cache is shared and must not be invalidated")

    primary_topk = int(config["model_analysis"]["primary_topk"])
    topk = int(args.topk) if args.topk is not None else primary_topk
    species_filter = list(args.species) if args.species else []
    suffix = artefact_suffix(args.topk, species_filter, args.limit_models)
    if suffix:
        C.log(
            "narrowed run: writing suffixed artefacts",
            repr(suffix),
            "(the unsuffixed names are the frozen full-manifest tables)",
        )
    # --topk alone means "sensitivity table only": the control is skipped unless
    # --repeats is also given (s10 passes --repeats 0 for exactly that reason).
    run_null = args.repeats is None or int(args.repeats) > 0
    if args.topk is not None and args.repeats is None:
        run_null = False
    n_repeats = (
        int(args.repeats) if args.repeats is not None else int(config["random_control"]["n_repeats"])
    )
    control = config["random_control"]
    settings = Settings(
        topk=topk,
        fdr=float(config["model_analysis"]["fdr"]),
        min_pairs=int(config["model_analysis"]["min_background_pairs"]),
        caliper=float(config["model_analysis"]["background"]["gc_caliper"]),
        n_repeats=n_repeats,
        seed=C.seed(config),
        null_n_peaks=int(control["n_peaks"]),
        require_exact_n_peaks=bool(control.get("require_exact_n_peaks", False)),
        min_valid_repeats=int(control.get("min_valid_repeats", 0)),
    )

    out = C.results_dir(config, STEP)
    paths = {
        "scores": out / f"all_model_motif_scores{suffix}.tsv",
        "audit": out / f"model_audit{suffix}.tsv",
        "null": out / f"random_null_scores{suffix}.tsv.gz",
        "summary": out / f"random_null_summary{suffix}.tsv",
        "scan": out / f"motif_scan_audit{suffix}.tsv",
        "scope": out / f"run_scope{suffix}.tsv",
        # Only written when the config declares a species-admission block
        # (`eligibility` in v2, `species_filter` in v1): it names the models that
        # block removed, with the cell count and ranking length that removed them.
        "species_filter": out / f"species_filter_audit{suffix}.tsv",
    }
    C.log(
        "topk", topk, "file suffix", repr(suffix), "null control", run_null, "repeats", n_repeats
    )

    with C.StepTimer("s01 model motif enrichment"):
        loaded_manifest, rankings = C.load_rankings(config)
        # Every model's seed comes from its position in the manifest as loaded,
        # so no later narrowing of the model list can re-seed it.
        loaded_manifest = with_model_index(loaded_manifest)

        # The analysis definition's own species filter. v1 declares none, and is
        # handed its manifest back untouched with no audit frame to write.
        manifest, species_audit, species_info = apply_config_species_filter(
            config, loaded_manifest
        )
        if species_audit is not None:
            C.save_table(species_audit, paths["species_filter"], sep="\t")
            for row in species_audit.itertuples():
                C.log(
                    "species filter dropped",
                    row.model,
                    f"({row.species} {row.celltype}: {row.n_cells} cells of "
                    f"{row.min_cells}, {row.n_ranked} ranked peaks of "
                    f"{row.min_ranked_peaks})",
                )
            C.log(
                "species filter kept",
                species_info["n_models_after"],
                "of",
                species_info["n_models_before"],
                "models; floors",
                f"{species_info['min_cells_per_species']} cells of",
                species_info["celltype_column"],
                f"and {species_info['min_ranked_peaks']} ranked peaks",
                f"(block `{species_info['block']}`)",
            )

        specs = model_specs(manifest)
        wanted_species = species_filter or C.SPECIES_ORDER
        selected = [spec for spec in specs if spec.species in wanted_species]
        if args.limit_models is not None:
            selected = selected[: int(args.limit_models)]
        if not selected:
            raise RuntimeError(
                "no model selected; check --species / --limit-models"
                + (
                    " and the config's species-admission block (see "
                    f"{paths['species_filter'].name})"
                    if species_audit is not None
                    else ""
                )
            )
        scope = RunScope(
            filtered=bool(species_filter) or args.limit_models is not None,
            n_models_selected=len(selected),
        )

        # The scan universe is always the full per-species union of the manifest
        # as loaded, before the species filter: the filter decides which models
        # are reported, never which sequence is scanned, so dropping a model can
        # neither shrink a species' universe nor cold-start the shared FIMO
        # cache. Restricting the model list must not change the cached input
        # either, which is the same rule.
        universes = C.model_species_peaks(loaded_manifest[~loaded_manifest.is_empty], rankings)
        genomes = C.species_table(config)
        catalog = C.motif_catalog(config)
        C.log("catalog", len(catalog), "motifs;", len(selected), "models selected")

        by_species: dict[str, list[ModelSpec]] = {}
        for spec in selected:
            by_species.setdefault(spec.species, []).append(spec)
        order = [s for s in C.SPECIES_ORDER if s in by_species] + [
            s for s in sorted(by_species) if s not in C.SPECIES_ORDER
        ]

        score_frames: list[pd.DataFrame] = []
        audit_rows: list[dict] = []
        summary_rows: list[dict] = []
        scan_rows: list[dict] = []
        failures: list[str] = []

        null = NullWriter(paths["null"]) if run_null else None
        try:
            for species in order:
                with C.StepTimer(f"species {species}"):
                    scan = scan_species(
                        config, species, genomes.get(species), universes.get(species, []), args.workers
                    )
                    scan_rows.append(scan.row)
                    C.log(
                        species,
                        f"universe={scan.row['n_universe_peaks']}",
                        f"scanned={scan.row['n_scan_peaks']}",
                        f"pairs={scan.row['n_motif_peak_pairs']}",
                        scan.row["status"],
                    )

                    usable = scan.row["status"] == "ok" and scan.row["n_motif_peak_pairs"] > 0
                    for spec in by_species[species]:
                        if not usable:
                            audit = base_audit(spec, settings, scope)
                            audit["status"] = (
                                "empty_ranking"
                                if spec.is_empty
                                else f"skipped_scan_{scan.row['status']}"
                            )
                            null_unavailable(audit, settings, null)
                            audit_rows.append(audit)
                            C.log(" ", spec.model, audit["status"])
                            continue
                        rng = np.random.default_rng([settings.seed, spec.model_index])
                        try:
                            scores, audit, summary = run_model(
                                spec,
                                rankings[spec.model],
                                scan,
                                catalog,
                                config,
                                settings,
                                scope,
                                rng,
                                null,
                            )
                        except Exception as exc:  # noqa: BLE001 - audit, never abort the step
                            audit = base_audit(spec, settings, scope)
                            audit["status"] = f"error:{type(exc).__name__}:{exc}"
                            null_unavailable(audit, settings, null)
                            scores, summary = None, None
                            failures.append(f"{spec.model}: {type(exc).__name__}: {exc}")
                        audit_rows.append(audit)
                        if scores is not None:
                            score_frames.append(scores)
                        if summary is not None:
                            summary_rows.append(summary)
                        C.log(
                            " ",
                            spec.model,
                            audit["status"],
                            f"fg={audit['n_fg_matched']} bg={audit['n_background']}",
                            audit["background_source"],
                            f"null={audit['null_status']}",
                        )
        finally:
            if null is not None:
                null.close()

        if not score_frames:
            raise RuntimeError("no model produced a motif profile; nothing written")

        # The scope record is written first, so every run states which slice of
        # the manifest the tables next to it cover -- including the run that
        # writes the unsuffixed, canonical names.
        C.save_table(
            scope_frame(
                species_filter,
                args.limit_models,
                args.topk,
                settings,
                scope,
                selected,
                len(specs),
                suffix,
                run_null,
                species_info,
            ),
            paths["scope"],
            sep="\t",
        )

        scores = pd.concat(score_frames, ignore_index=True).sort_values(
            ["model", "motif_id"], kind="stable"
        )
        # save_table defaults to a comma separator; every artefact here is TSV.
        C.save_table(scores.reset_index(drop=True), paths["scores"], sep="\t")

        audit_frame = pd.DataFrame(audit_rows)[AUDIT_COLUMNS].sort_values(
            ["model"], kind="stable"
        )
        C.save_table(audit_frame.reset_index(drop=True), paths["audit"], sep="\t")
        C.save_table(pd.DataFrame(scan_rows)[SCAN_COLUMNS], paths["scan"], sep="\t")

        if run_null:
            audits = {row["model"]: row for row in audit_rows}
            computed = {row["model"]: row for row in summary_rows}
            summary_frame = pd.DataFrame(
                [
                    computed.get(spec.model, blank_summary(spec, settings, audits[spec.model]))
                    for spec in sorted(selected, key=lambda s: s.model)
                ]
            )[SUMMARY_COLUMNS]
            C.save_table(summary_frame.reset_index(drop=True), paths["summary"], sep="\t")
            C.log("null table", paths["null"].name, f"{paths['null'].stat().st_size / 2**20:.1f} MiB")

        C.log("audit status", audit_frame.status.value_counts().to_dict())
        C.log("models with a profile", len(score_frames), "/", len(selected))
        if failures:
            raise RuntimeError(
                f"{len(failures)} failures recorded in model_audit.tsv: {failures[:5]}"
            )


if __name__ == "__main__":
    main()
