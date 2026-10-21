"""Analysis A, model-only part (readme sections 8 to 14).

WHY this step exists
--------------------
s01 turned the Top-k ranked peaks of every Astroglia model into an 879-motif
log2 odds-ratio profile. This step reads that table alone and answers the
model-only questions of the frozen design:

  A2  does the model-derived motif program remodel along the Astroglia
      developmental axis of a species (within-species, its single-species
      window included)?
  A4  do the four pre-registered TF families follow a trajectory once the
      species-specific motif baseline is removed?
  A5/A6/A7  does the stage profile of one species recover the matching stage of
      another species better than a mismatching one (stage-by-stage Spearman,
      matching margin), exactly and beyond the matched-random null?

Nothing here touches the raw scATAC matrix: the chromVAR half of Gate A3/A4
belongs to s05 and is reported as PENDING. Every degree of freedom -- windows,
centering scope, family map, permutation scheme -- is configuration, and no
result is allowed to re-pick a motif, a family, a window or a Top-k.

Windows are a property of the period table this run is pointed at, never of this
module. `common.cross_windows(config)` gives the windows that enter a
cross-species comparison (v1: prenatal, postnatal, adult_DN; v2: prenatal,
postnatal, adult) and `common.single_species_windows(config)` the ones that
belong to a single species and never do (v1: human_only; v2: early_embryonic).
Every window constant below -- the stage matrix, the matching margin, the exact
permutation, the null and the file names -- is derived from those two tuples at
runtime, so the same script runs unchanged against either definition.

The output directory is results/02_astroglia_model/ (results_v2/... under v2).
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import common  # noqa: E402


def resolved_cross_windows(cross_windows: Sequence[str] | None) -> tuple[str, ...]:
    """The cross-species windows of this run, in period-table order.

    There is no module-level window constant: the set is read from the period
    table the config points at (v1 `config/stage_map.tsv`, v2
    `config/stage_map_v2.tsv`). The `None` fallback exists only for the legacy
    call sites in script/tests that import this module and call its helpers
    without a config; it resolves the process' default period table at runtime,
    it does not name a window.
    """
    return tuple(cross_windows) if cross_windows is not None else common.cross_windows()


# Representative motifs drawn on the heatmap next to the significant union. The
# preference list is frozen text, not a result: a family shows its namesake
# members first and falls back to alphabetical order if a namesake is absent.
N_REPRESENTATIVES = 3
REPRESENTATIVE_PREFERENCE = {
    "RFX": ["RFX1", "RFX2", "RFX3", "RFX4", "RFX5", "RFX7", "Rfx6"],
    "NFI": ["NFIA", "NFIB", "NFIC", "NFIX"],
    "NF-Y": ["NFYA", "NFYB", "NFYC"],
    "SOX": ["SOX2", "SOX9", "SOX10", "SOX4", "SOX8", "Sox11", "Sox3", "Sox5", "Sox6"],
}

SCORE_COLUMNS = (
    "analysis",
    "species",
    "period",
    "motif_id",
    "motif_name",
    "log2_OR",
    "FDR",
    "significant",
)
NULL_COLUMNS = ("model", "repeat", "motif_id", "log2_OR")

# Gate A2 is a claim about the distribution of matched random-500 margins, so it
# is only evaluated once every species pair has at least this many jointly usable
# null repeats. A null table that is ABSENT leaves the gate PENDING (absence of
# evidence is not a FAIL, which would be a positive claim that the random control
# beats the observed Top-500 result). A null table that is PRESENT but yields no
# pair at this threshold -- the empty intersection a differently named model set
# produces, for instance -- is a control that cannot be used: the gate is FAIL and
# the note says so, rather than silently reverting to PENDING.
MIN_NULL_REPEATS_FOR_GATE = 2

# Colors follow the project data-viz palette: a two-hue diverging ramp with a
# neutral gray midpoint for the heatmap, fixed categorical slots for families in
# their frozen map order, recessive chrome everywhere else.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
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
DIVERGING_STOPS = (
    "#0d366b",
    "#1c5cab",
    "#3987e5",
    "#86b6ef",
    "#cde2fb",
    "#f0efec",
    "#fbd5d3",
    "#f0a19e",
    "#e05f5c",
    "#b8342f",
    "#7d1f1b",
)


@dataclass(frozen=True)
class PairMargin:
    """Observed stage-matching result of one ordered species pair.

    `matrix` is len(cross_windows) x len(cross_windows): one row and one column
    per cross-species window of the run's period table.
    """

    species_a: str
    species_b: str
    matrix: np.ndarray
    mean_matching: float
    mean_mismatching: float
    margin: float
    p_perm: float
    p_perm_min: float
    n_permutations: int
    top1_accuracy: float


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Astroglia model-only analysis (s02)")
    common.add_common_args(parser)
    parser.add_argument("--scores", default=None, help="override the s01 motif-score table")
    parser.add_argument("--null-scores", default=None, help="override the s01 random-null table")
    parser.add_argument("--out-root", default=None, help="override paths.results (smoke tests)")
    return parser.parse_args(argv)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], path: Path) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise RuntimeError(f"{path}: missing column(s) {missing}")


def _boolish(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def read_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; run script/s01_model_motif_enrichment.py first "
            "(or pass --scores for a smoke test)"
        )
    frame = pd.read_csv(path, sep="\t")
    _require_columns(frame, SCORE_COLUMNS, path)
    frame["significant"] = _boolish(frame["significant"])
    return frame


def read_null_scores(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        common.log("WARN random-null table missing:", path, "-> Gate A2 stays PENDING")
        return None
    frame = pd.read_csv(path, sep="\t")
    _require_columns(frame, NULL_COLUMNS, path)
    return frame


def astroglia_stages(config: dict | None = None) -> pd.DataFrame:
    """The species -> rank_period -> cross_window table of this config, in order.

    The table is whatever the config's period declaration points at
    (`common.stage_map_path(config)`), so v1 and v2 read different files without
    this script naming either. A declared window must be one of the two kinds
    the period table itself defines -- a cross-species window (declared for two
    or more species) or a single-species window -- and the sets are derived from
    the table, never hardcoded.

    (species, cross_window) must be unique: the cross-species analysis keys every
    profile by that pair, so a stage map that folds two declared periods into one
    window would keep the first period on the observed path (pivot_table
    aggfunc='first') and the last on the null path (dict overwrite), silently
    deleting a declared period from the statistics instead of auditing it.
    """
    path = common.stage_map_path(config)
    table = common.stage_table(config)[["species", "rank_period", "cross_window"]].copy()
    if table[["species", "rank_period", "cross_window"]].isna().any().any():
        raise RuntimeError(f"{path} has an incomplete species/rank_period/cross_window row")
    cross = set(common.cross_windows(config))
    single = set(common.single_species_windows(config))
    unknown = sorted(set(table.cross_window) - cross - single)
    if unknown:
        raise RuntimeError(
            f"{path} declares cross_window(s) that are neither a cross-species window "
            f"({sorted(cross)}) nor a single-species window ({sorted(single)}): {unknown}"
        )
    folded = table[table.duplicated(["species", "cross_window"], keep=False)]
    if len(folded):
        clashes = sorted(
            f"{species}/{window} <- " + "|".join(sorted(group.rank_period))
            for (species, window), group in folded.groupby(["species", "cross_window"])
        )
        raise RuntimeError(
            f"{path} maps more than one declared period of one species to the same "
            f"cross_window ({'; '.join(clashes)}); the stage-matching analysis keys on "
            "(species, cross_window) and would silently drop all but one of those periods"
        )
    return table


def species_scope(
    config: dict,
    stage: pd.DataFrame,
    scores: pd.DataFrame,
    audit: list[dict],
) -> tuple[list[str], dict]:
    """The species this Astroglia run may analyse, and the audit of the rest.

    The species set is not a literal: it is the species that carry an Astroglia
    model in the manifest (`common.ranking_files`) AFTER
    `common.apply_species_filter` -- the configured rule that a species enters a
    cell-type analysis only above `min_cells_per_species` cells of that type.
    The order stays the period table's, so the pair direction and every
    `stage_matching_matrix_<a>_<b>.tsv` file name are config-ordered.

    Every species and model that leaves is written as an audit row naming the
    entity and the reason; nothing is dropped silently.
    """
    manifest = common.ranking_files(config)
    kept_manifest, filter_audit = common.apply_species_filter(config, manifest)
    astro_manifest = kept_manifest[kept_manifest.analysis == "Astroglia"]
    if astro_manifest.empty:
        raise RuntimeError(
            "the filtered manifest holds no Astroglia model; there is nothing to analyse"
        )

    order = list(stage_order(stage))
    rank = {species: index for index, species in enumerate(order)}
    declared = set(order)
    manifest_species = list(dict.fromkeys(astro_manifest.species))
    allowed = sorted((s for s in manifest_species if s in declared), key=rank.__getitem__)

    # 1. the species filter itself: one row per Astroglia model it removed.
    pre_filter_species = set(manifest.loc[manifest.analysis == "Astroglia", "species"])
    dropped_models: list[dict] = []
    if len(filter_audit):
        for row in filter_audit[filter_audit.analysis == "Astroglia"].itertuples():
            dropped_models.append(
                dict(
                    species=row.species,
                    model=row.model,
                    n_cells=int(row.n_cells),
                    min_cells=int(row.min_cells),
                )
            )
            audit.append(
                dict(
                    check="species_filter",
                    species=row.species,
                    stage="",
                    model=row.model,
                    status="dropped_species_filter",
                    detail=f"{int(row.n_cells)} cells of celltype '{row.celltype}' < "
                    f"species_filter.min_cells_per_species {int(row.min_cells)}; the model is "
                    "excluded from this analysis",
                )
            )

    # 2. species whose whole Astroglia arm the filter removed.
    for species in sorted(pre_filter_species - set(allowed)):
        if species in declared:
            audit.append(
                dict(
                    check="species_scope",
                    species=species,
                    stage="",
                    model="",
                    status="skipped_species",
                    detail="every Astroglia model of this species was removed by species_filter; "
                    "no period of it enters the analysis",
                )
            )

    # 3. species with a model in the filtered manifest but no row in the period table.
    for species in sorted(set(manifest_species) - declared):
        audit.append(
            dict(
                check="species_scope",
                species=species,
                stage="",
                model="",
                status="skipped_species",
                detail=f"carries an Astroglia model in the filtered manifest but no period in "
                f"{common.stage_map_path(config)}; its rank_period(s) cannot be placed",
            )
        )

    # 4. species the period table declares that the filtered manifest does not
    #    carry, but for which the score table nevertheless has rows, are dropped
    #    by build_profile -- it sees the actual rows -- and audited there.

    # 5. manifest models with no motif-score row: a declared period that cannot
    #    be analysed is named, never silently absent.
    if "model" in scores.columns:
        present_models = set(scores.model.astype(str))
        for row in astro_manifest.itertuples():
            if row.model not in present_models:
                audit.append(
                    dict(
                        check="manifest_model",
                        species=row.species,
                        stage=row.period,
                        model=row.model,
                        status="missing_from_scores",
                        detail="declared in the filtered Astroglia manifest but absent from the "
                        "motif-score table; its period contributes nothing",
                    )
                )

    # The config's species-admission block, under either spelling: v2 declares
    # `eligibility`, v1 `species_filter`. `common.apply_species_filter` above
    # already honours both, so the scope record has to report the same rule.
    admissibility_block = config.get("eligibility") or config.get("species_filter")
    block_name = "eligibility" if config.get("eligibility") else "species_filter"

    audit.append(
        dict(
            check="species_scope",
            species="|".join(allowed),
            stage="",
            model="",
            status="ok" if len(allowed) == len(manifest_species) else "reduced",
            detail=f"{len(allowed)}/{len(manifest_species)} Astroglia manifest species enter the "
            f"analysis; period table {common.stage_map_path(config)}; "
            + (
                f"species-admission block `{block_name}` configured"
                if admissibility_block
                else "no species-admission block configured"
            ),
        )
    )
    scope = {
        # v2 spells the block `eligibility`, v1 spells it `species_filter`; the
        # same precedence `common.min_cells_per_species` / `apply_species_filter`
        # use. Reading only the v1 key would report "no filter configured" for a
        # v2 run whose filter is in force.
        "configured": bool(admissibility_block),
        "block": block_name,
        "min_cells_per_species": common.min_cells_per_species(config),
        "celltype_column": (admissibility_block or {}).get("celltype_column", "Org_Celltype"),
        "manifest_species": manifest_species,
        "species": allowed,
        "dropped_models": dropped_models,
        "models": sorted(astro_manifest.model.astype(str)),
    }
    return allowed, scope


# --------------------------------------------------------------------------
# profile construction (A2)
# --------------------------------------------------------------------------


def build_profile(
    scores: pd.DataFrame,
    stage: pd.DataFrame,
    topk: int,
    allowed_species: Sequence[str] | None = None,
    cross_windows: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[dict], dict[str, list[str]]]:
    """Long Astroglia profile with the species-baseline-removed score attached.

    Centering scope is every period of the species present in the Astroglia
    profile, the single-species window included (centering_scope=all_species_periods).

    `allowed_species` is the species set of the filtered Astroglia manifest. A
    species that carries rows in the motif-score table but is not in it is
    removed and audited, never silently dropped; `None` keeps every species the
    score table has (the legacy call shape used by script/tests).
    """
    audit: list[dict] = []
    analysis_labels = scores.analysis.astype(str).str.strip()
    astro = scores[analysis_labels.str.lower() == "astroglia"].copy()
    if astro.empty:
        raise RuntimeError("the motif-score table contains no Astroglia model")
    variants = sorted(set(astro.analysis.astype(str)) - {"Astroglia"})
    if variants:
        audit.append(
            dict(
                check="analysis_label",
                species="",
                stage="",
                model="",
                status="accepted_variants",
                detail=f"Astroglia selected by case-insensitive match of {variants}",
            )
        )

    if "topk" in astro.columns:
        other = astro[astro.topk != topk]
        if len(other):
            audit.append(
                dict(
                    check="topk_filter",
                    species="",
                    stage="",
                    model="",
                    status="filtered_out",
                    detail=f"{len(other)} rows at topk != {topk} ignored",
                )
            )
        astro = astro[astro.topk == topk]
    if astro.empty:
        raise RuntimeError(f"no Astroglia model has topk == {topk}")

    if allowed_species is not None:
        allowed = set(allowed_species)
        outside = sorted(set(astro.species) - allowed)
        for species in outside:
            models = sorted(astro.loc[astro.species == species, "model"].unique()) if "model" in astro.columns else []
            audit.append(
                dict(
                    check="species_scope",
                    species=species,
                    stage="",
                    model="|".join(models),
                    status="skipped_species",
                    detail="not among the species of the filtered Astroglia manifest; every model "
                    "of this species is excluded from the analysis",
                )
            )
        if outside:
            astro = astro[astro.species.isin(allowed)].copy()
        if astro.empty:
            raise RuntimeError(
                "no Astroglia model belongs to a species of the filtered Astroglia manifest"
            )

    known = set(zip(stage.species, stage.rank_period))
    key = list(zip(astro.species, astro.period))
    unknown = sorted({f"{s}__{p}" for s, p in key if (s, p) not in known})
    for model in unknown:
        audit.append(
            dict(
                check="model_selection",
                species=model.split("__")[0],
                stage="",
                model=model,
                status="skipped_model",
                detail="(species, period) not declared in this run's period table",
            )
        )
    astro = astro[[(s, p) in known for s, p in key]].copy()

    duplicates = int(astro.duplicated(["species", "period", "motif_id"]).sum())
    if duplicates:
        audit.append(
            dict(
                check="duplicate_rows",
                species="",
                stage="",
                model="",
                status="dropped_duplicates",
                detail=f"{duplicates} duplicated (species, period, motif_id) rows kept once",
            )
        )
        astro = astro.drop_duplicates(["species", "period", "motif_id"], keep="first")

    labels = stage.rename(columns={"rank_period": "period"})
    profile = astro.merge(labels, on=["species", "period"], how="left", validate="many_to_one")
    profile = profile.rename(columns={"period": "stage"})

    counts = profile.groupby(["species", "stage"]).motif_id.nunique()
    # Species order is this run's period-table order (never the order in which
    # the s01 rows happen to arrive, which a shuffled input table would otherwise
    # re-label); it drives the audit order, the species_a/species_b direction of
    # every pair, the file name of every stage_matching_matrix_<a>_<b>.tsv and
    # the summary's species list.
    present_species = set(profile.species)
    ordered_species = [name for name in stage_order(stage) if name in present_species]
    for species in ordered_species:
        group = profile[profile.species == species]
        expected = sorted(set(stage.loc[stage.species == species, "rank_period"]))
        found = sorted(group.stage.unique())
        missing = [period for period in expected if period not in found]
        audit.append(
            dict(
                check="stage_coverage",
                species=species,
                stage="|".join(expected),
                model="",
                status="ok" if not missing else "missing_stages",
                detail=f"found {len(found)}/{len(expected)} declared periods"
                + (f"; missing {'|'.join(missing)}" if missing else ""),
            )
        )
        for period in expected:
            if period in found:
                audit.append(
                    dict(
                        check="stage_motif_count",
                        species=species,
                        stage=period,
                        model="",
                        status="ok",
                        detail=f"n_motifs={int(counts.get((species, period), 0))}",
                    )
                )

    species_motifs = {
        species: sorted(profile.loc[profile.species == species, "motif_id"].unique())
        for species in ordered_species
    }

    # A content-defined row order before the baseline removal makes the centered
    # value a function of the table content alone, not of its row order (the
    # groupby-mean summation order is otherwise input-order dependent).
    profile = profile.sort_values(["species", "stage", "motif_id"], kind="stable").reset_index(
        drop=True
    )
    profile["log2_OR_centered"] = profile["log2_OR"] - profile.groupby(["species", "motif_id"])[
        "log2_OR"
    ].transform("mean")

    # A SECOND centering, over the cross-species windows ONLY.
    #
    # The baseline above averages over every period the species has, which for
    # human includes the single-species early_embryonic window. That is the right
    # baseline for the WITHIN-SPECIES axis -- the heatmap shows human CS18-23
    # alongside H2/H3/H4 and all four must share one origin -- but it lets H1 move
    # the cross-species numbers through the per-motif mean. H1 is not a
    # cross-species window and must not enter a cross-species comparison in any
    # role, so every cross-species consumer reads this column instead.
    #
    # The two differ only for a species that has a single-species window; for
    # marmoset and mouse they are identical by construction.
    # NOT `resolved_cross_windows(None)`: that falls back to the process' default
    # period table, which is v1, and would mask out the run's real windows. The
    # set must come from the caller, which read it from this run's config.
    cross_windows_now = resolved_cross_windows(cross_windows)
    if cross_windows is None:
        raise RuntimeError(
            "build_profile needs the run's cross-species windows; the None fallback "
            "resolves the v1 period table and would silently mis-mask the profile"
        )
    cross_mask = profile.cross_window.isin(cross_windows_now)
    cross_mean = profile.loc[cross_mask].groupby(["species", "motif_id"])["log2_OR"].transform("mean")
    profile["log2_OR_centered_cross"] = np.nan
    profile.loc[cross_mask, "log2_OR_centered_cross"] = (
        profile.loc[cross_mask, "log2_OR"].to_numpy() - cross_mean.to_numpy()
    )
    audit.append(
        dict(
            check="centering_scope",
            species="",
            stage="",
            model="",
            status="ok",
            detail=(
                "`log2_OR_centered_cross` is centred over the species' cross-species windows "
                f"only ({'|'.join(cross_windows_now)}); a single-species window is NaN there and "
                "never enters a cross-species statistic. `log2_OR_centered` remains the "
                "all-periods baseline used by the within-species display"
            ),
        )
    )

    stage_rank = {
        species: {period: index for index, period in enumerate(periods)}
        for species, periods in stage_order(stage).items()
    }
    profile["_stage_rank"] = [
        stage_rank[species][period] for species, period in zip(profile.species, profile.stage)
    ]
    species_rank = {species: index for index, species in enumerate(stage_order(stage))}
    profile["_species_rank"] = profile.species.map(species_rank)
    profile = profile.sort_values(["_species_rank", "_stage_rank", "motif_id"]).reset_index(drop=True)
    profile = profile[
        [
            "species",
            "stage",
            "cross_window",
            "motif_id",
            "motif_name",
            "log2_OR",
            "FDR",
            "significant",
            "log2_OR_centered",
            "log2_OR_centered_cross",
        ]
    ]
    return profile, audit, species_motifs


def stage_order(stage: pd.DataFrame) -> dict[str, list[str]]:
    return {
        species: group.rank_period.tolist()
        for species, group in stage.groupby("species", sort=False)
    }


# --------------------------------------------------------------------------
# cross-species stage similarity, margin and exact permutation (A5/A6)
# --------------------------------------------------------------------------


def cross_window_wide(
    profile: pd.DataFrame,
    species: Sequence[str],
    cross_windows: Sequence[str] | None = None,
) -> pd.DataFrame:
    """motif x (species, window) matrix of the centered profile, cross windows only."""
    windows = resolved_cross_windows(cross_windows)
    cross = profile[profile.cross_window.isin(windows)]
    wide = cross.pivot_table(
        index="motif_id",
        columns=["species", "cross_window"],
        values="log2_OR_centered_cross",
        aggfunc="first",
    )
    columns = [(sp, window) for sp in species for window in windows]
    wide = wide.reindex(columns=columns)
    return wide


def species_window_similarity(
    wide: pd.DataFrame,
    species: Sequence[str],
    cross_windows: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Spearman similarity of every (species, cross-window) profile of `species`."""
    windows = resolved_cross_windows(cross_windows)
    profiles = {
        f"{sp}|{window}": wide[(sp, window)].to_numpy(dtype=float)
        for sp in species
        for window in windows
    }
    return common.similarity_matrix(profiles)


# Legacy public name: script/tests/adv_s10_verify.py imports this module and
# calls the helper by this name. The matrix is len(species) x len(species) over
# the (species, window) profiles for any period definition, so the name is kept
# only for that call site; nothing in this script uses it.
similarity_6x6 = species_window_similarity


def pair_margin(
    sim: pd.DataFrame,
    species_a: str,
    species_b: str,
    cross_windows: Sequence[str] | None = None,
) -> PairMargin:
    windows = resolved_cross_windows(cross_windows)
    rows = [f"{species_a}|{window}" for window in windows]
    cols = [f"{species_b}|{window}" for window in windows]
    matrix = sim.loc[rows, cols].to_numpy(dtype=float)
    labels = list(windows)
    mean_matching, mean_mismatching, margin = common.matching_margin(matrix, labels)
    observed, p_perm, p_min, n_perms = common.permutation_margin_pvalue(matrix, labels)
    if not np.isclose(observed, margin, atol=1e-12):
        raise RuntimeError("permutation helper disagrees with matching_margin")
    top1 = float(np.mean(np.argmax(matrix, axis=1) == np.arange(len(windows))))
    return PairMargin(
        species_a=species_a,
        species_b=species_b,
        matrix=matrix,
        mean_matching=mean_matching,
        mean_mismatching=mean_mismatching,
        margin=margin,
        p_perm=p_perm,
        p_perm_min=p_min,
        n_permutations=n_perms,
        top1_accuracy=top1,
    )


# --------------------------------------------------------------------------
# matched random-500 null of the same margin (A7)
# --------------------------------------------------------------------------


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, bool]:
    """(rho, is_finite) for two centered motif vectors.

    A constant vector (every motif carrying the same centered value) makes the
    rank correlation undefined and common.spearman returns NaN. The caller must
    EXCLUDE such a draw from the null distribution and count it; substituting
    0.0 would plant a phantom draw exactly at the centre of the null and shrink
    its spread, which is the opposite of a conservative control.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, _ = common.spearman(a, b)
    return float(rho), bool(np.isfinite(rho))


def null_stage_frames(
    null: pd.DataFrame,
    stage: pd.DataFrame,
    species: list[str],
    audit: list[dict],
) -> dict[tuple[str, str], pd.DataFrame]:
    """{(species, window): float frame repeat x motif} from the s01 random null."""
    lookup = {
        f"Astroglia__{row.species}__{row.rank_period}": (row.species, row.cross_window)
        for row in stage.itertuples()
    }
    astro_null = null[null.model.isin(set(lookup))].copy()
    n_models = int(null.model.nunique())
    audit.append(
        dict(
            check="null_model_coverage",
            species="",
            stage="",
            model="",
            status="ok" if astro_null.model.nunique() == len(lookup) else "reduced",
            detail=f"used {astro_null.model.nunique()}/{len(lookup)} Astroglia null models "
            f"(table holds {n_models} models in total)",
        )
    )
    nested = int(astro_null.duplicated(["model", "repeat", "motif_id"]).sum())
    if nested:
        audit.append(
            dict(
                check="null_duplicate_rows",
                species="",
                stage="",
                model="",
                status="dropped_duplicates",
                detail=f"{nested} duplicated (model, repeat, motif_id) rows kept once",
            )
        )
        astro_null = astro_null.drop_duplicates(["model", "repeat", "motif_id"], keep="first")

    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for model, group in astro_null.groupby("model", sort=False):
        key = lookup[model]
        frames[key] = group.pivot_table(
            index="repeat", columns="motif_id", values="log2_OR", aggfunc="first"
        ).sort_index()
    for species_name in species:
        for model, key in lookup.items():
            if key[0] != species_name:
                continue
            frame = frames.get(key)
            n_repeats = 0 if frame is None else len(frame)
            audit.append(
                dict(
                    check="null_repeat_coverage",
                    species=species_name,
                    stage=key[1],
                    model=model,
                    status="ok" if n_repeats else "missing",
                    detail=f"repeats={n_repeats}",
                )
            )
    return frames


def null_margins(
    frames: dict[tuple[str, str], pd.DataFrame],
    species: list[str],
    motif_ids: list[str],
    repeat_cap: int | None,
    audit: list[dict],
    cross_windows: Sequence[str] | None = None,
) -> pd.DataFrame:
    """One stage-matching margin per species pair per random repeat.

    Each repeat is centered exactly like the observed profile -- per motif, over
    the cross-species windows of that species within that repeat. A species whose
    period table carries an extra single-species window that is not a cross
    window centers its OBSERVED profile over one more period than the null does;
    the asymmetry is recorded under null_centering_scope in the summary.

    Repeats are intersected across the 2 x len(cross_windows) stage profiles of a
    pair on their raw indices, before any reindex, because s01 drops a failed
    repeat per model: a union would fill the repeats a model never ran with NaN
    and every vector built from them would come back degenerate. A repeat whose
    centered vectors are constant or non-finite is dropped and audited, never
    scored as a zero margin.
    """
    windows = resolved_cross_windows(cross_windows)
    n_windows = len(windows)
    if not frames:
        return pd.DataFrame(columns=["species_a", "species_b", "repeat", "margin"])

    null_motifs = set().union(*[set(frame.columns) for frame in frames.values()])
    usable_motifs = [motif for motif in motif_ids if motif in null_motifs]
    if len(usable_motifs) != len(motif_ids):
        audit.append(
            dict(
                check="null_motif_coverage",
                species="",
                stage="",
                model="",
                status="reduced",
                detail=f"{len(usable_motifs)}/{len(motif_ids)} observed motifs are present in "
                "the random-null table",
            )
        )
    if not usable_motifs:
        return pd.DataFrame(columns=["species_a", "species_b", "repeat", "margin"])

    # Per species, the repeats its cross-window profiles all produced. Taken from
    # the raw pivot indices, before any reindex, so a repeat one model never ran
    # cannot come back as an all-NaN column that then vanishes.
    species_repeats: dict[str, list[int]] = {}
    for species_name in species:
        keys = [(species_name, window) for window in windows]
        if any(key not in frames for key in keys):
            continue
        shared = set(frames[keys[0]].index)
        for key in keys[1:]:
            shared &= set(frames[key].index)
        species_repeats[species_name] = sorted(shared)

    # Center each repeat over the cross windows of its species: the motif mean a
    # repeat removes is built from that repeat alone, so no repeat can borrow
    # another repeat's baseline.
    centered: dict[tuple[str, str], np.ndarray] = {}
    for species_name, repeats in species_repeats.items():
        stack = np.stack(
            [
                frames[(species_name, window)]
                .reindex(index=repeats, columns=usable_motifs)
                .to_numpy(dtype=float)
                for window in windows
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            stack = stack - np.nanmean(stack, axis=0, keepdims=True)
        for index, window in enumerate(windows):
            centered[(species_name, window)] = stack[index]

    if repeat_cap is not None:
        audit.append(
            dict(
                check="null_repeat_cap",
                species="",
                stage="",
                model="",
                status="capped",
                detail=f"only the first {repeat_cap} repeats used (--repeats)",
            )
        )

    rows: list[dict] = []
    for species_a, species_b in combinations(species, 2):
        pair = f"{species_a}|{species_b}"
        if species_a not in species_repeats or species_b not in species_repeats:
            audit.append(
                dict(
                    check="null_pair_coverage",
                    species=pair,
                    stage="",
                    model="",
                    status="skipped_pair",
                    detail="at least one cross window has no random-null model",
                )
            )
            continue
        union = sorted(set(species_repeats[species_a]) | set(species_repeats[species_b]))
        usable = sorted(set(species_repeats[species_a]) & set(species_repeats[species_b]))
        audit.append(
            dict(
                check="null_repeat_intersection",
                species=pair,
                stage="|".join(windows),
                model="",
                status="ok" if union and len(usable) == len(union) else "reduced",
                detail=f"{len(usable)}/{len(union)} repeats present in all "
                f"{2 * n_windows} stage profiles of the pair",
            )
        )
        if repeat_cap is not None:
            usable = usable[:repeat_cap]
        position_a = {repeat: index for index, repeat in enumerate(species_repeats[species_a])}
        position_b = {repeat: index for index, repeat in enumerate(species_repeats[species_b])}
        n_degenerate = 0
        for repeat in usable:
            block_a = np.vstack(
                [centered[(species_a, window)][position_a[repeat]] for window in windows]
            )
            block_b = np.vstack(
                [centered[(species_b, window)][position_b[repeat]] for window in windows]
            )
            matrix = np.zeros((n_windows, n_windows))
            degenerate = False
            for i in range(n_windows):
                for j in range(n_windows):
                    rho, finite = _safe_spearman(block_a[i], block_b[j])
                    if not finite:
                        degenerate = True
                        break
                    matrix[i, j] = rho
                if degenerate:
                    break
            if degenerate:
                # The repeat is excluded from the null and counted; it is never
                # entered as a rho = 0 margin.
                n_degenerate += 1
                continue
            rows.append(
                dict(
                    species_a=species_a,
                    species_b=species_b,
                    repeat=repeat,
                    margin=common.matching_margin(matrix, list(windows))[2],
                )
            )
        if n_degenerate:
            audit.append(
                dict(
                    check="null_degenerate_repeat",
                    species=pair,
                    stage="",
                    model="",
                    status="dropped_repeats",
                    detail=f"{n_degenerate}/{len(usable)} repeat(s) whose {2 * n_windows} "
                    "centered vectors give an undefined (NaN) Spearman were excluded from the "
                    f"null and counted; the remaining {len(usable) - n_degenerate} enter "
                    "random_null_margin.tsv. No undefined correlation is ever scored as rho=0",
                )
            )
    return pd.DataFrame(rows, columns=["species_a", "species_b", "repeat", "margin"])


def margin_null_test(observed: pd.DataFrame, null_margin: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for row in observed.itertuples():
        values = null_margin[
            (null_margin.species_a == row.species_a) & (null_margin.species_b == row.species_b)
        ].margin.to_numpy(dtype=float)
        if len(values):
            null_mean = float(np.mean(values))
            null_sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            null_q95 = float(np.quantile(values, 0.95))
            empirical_p = float((1 + int(np.sum(values >= row.margin))) / (1 + len(values)))
        else:
            null_mean = null_sd = null_q95 = float("nan")
            empirical_p = float("nan")
        rows.append(
            dict(
                species_a=row.species_a,
                species_b=row.species_b,
                margin_observed=row.margin,
                null_mean=null_mean,
                null_sd=null_sd,
                null_q95=null_q95,
                empirical_p=empirical_p,
                n_repeats=len(values),
                passes_null=bool(len(values) and row.margin > null_q95),
            )
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# TF-family trajectory (A4)
# --------------------------------------------------------------------------


def family_trajectory_within_species(
    profile: pd.DataFrame,
    stage: pd.DataFrame,
    species: list[str],
    cross_windows: Sequence[str] | None = None,
) -> pd.DataFrame:
    """The same families along each species' OWN full axis, single-species window included.

    This is the only place human's early_embryonic period appears as a family
    score, and it is explicitly labelled: it answers a within-human question
    (does the family move along H1 -> H4) and is never fed into a cross-species
    statistic. Its baseline averages over the species' every period, which is the
    correct origin for an axis that contains all of them.
    """
    table = common.tf_family_table()
    trajectory = common.family_trajectory(profile, table, value="log2_OR", motif_col="motif_id")
    trajectory = trajectory.merge(
        stage.rename(columns={"rank_period": "stage"}),
        on=["species", "stage"],
        how="left",
        validate="many_to_one",
    )
    trajectory["centering_scope"] = "all_species_periods"
    windows = resolved_cross_windows(cross_windows)
    trajectory["analysis_scope"] = [
        "cross_species" if window in set(windows) else "species_only"
        for window in trajectory.cross_window
    ]
    trajectory["included_cross_species"] = trajectory.analysis_scope.eq("cross_species")
    return trajectory[
        ["species", "stage", "cross_window", "analysis_scope", "included_cross_species",
         "family", "analysis_level", "family_score", "n_members", "n_significant",
         "centering_scope"]
    ].reset_index(drop=True)


def family_trajectory_tables(
    profile: pd.DataFrame,
    stage: pd.DataFrame,
    species: list[str],
    cross_windows: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long family trajectory (common.family_trajectory) plus its wide matrix.

    The frame is restricted to the CROSS-SPECIES windows before the family median
    is taken, so the per-motif baseline this aggregates over contains no
    single-species window. Human's early_embryonic period is therefore absent from
    the table entirely rather than merely unplotted: it cannot move a cross-species
    score through the baseline, which is the role it used to play.

    The within-species axis is emitted separately by
    `family_trajectory_within_species` so that human's CS18-23 remains available
    where it is legitimate content.
    """
    windows = resolved_cross_windows(cross_windows)
    profile = profile[profile.cross_window.isin(windows)]
    table = common.tf_family_table()
    trajectory = common.family_trajectory(profile, table, value="log2_OR", motif_col="motif_id")
    trajectory = trajectory.merge(
        stage.rename(columns={"rank_period": "stage"}),
        on=["species", "stage"],
        how="left",
        validate="many_to_one",
    )
    trajectory["centering_scope"] = f"cross_species_windows_only ({'|'.join(windows)})"
    trajectory = trajectory[
        [
            "species",
            "stage",
            "cross_window",
            "family",
            "analysis_level",
            "family_score",
            "n_members",
            "n_significant",
            "centering_scope",
        ]
    ]
    order = {species_name: index for index, species_name in enumerate(species)}
    stage_rank = stage_order(stage)
    trajectory["_s"] = trajectory.species.map(order)
    trajectory["_t"] = [
        stage_rank[species_name].index(period)
        for species_name, period in zip(trajectory.species, trajectory.stage)
    ]
    trajectory["_f"] = trajectory.family.map(
        {family: index for index, family in enumerate(family_order(table))}
    )
    trajectory = trajectory.sort_values(["_s", "_t", "_f"]).drop(columns=["_s", "_t", "_f"])
    trajectory = trajectory.reset_index(drop=True)

    wide = trajectory.pivot_table(
        index=["species", "stage", "cross_window"], columns="family", values="family_score"
    ).reset_index()
    wide.columns.name = None
    wide["_s"] = wide.species.map(order)
    wide["_t"] = [
        stage_rank[species_name].index(period)
        for species_name, period in zip(wide.species, wide.stage)
    ]
    wide = wide.sort_values(["_s", "_t"]).drop(columns=["_s", "_t"]).reset_index(drop=True)
    return trajectory, wide


def family_order(table: pd.DataFrame) -> list[str]:
    """Frozen family order: primary families first, then secondary, both sorted."""
    primary = table.loc[table.analysis_level == "primary", "family"].dropna().unique()
    secondary = table.loc[table.analysis_level != "primary", "family"].dropna().unique()
    return sorted(v for v in primary if v) + sorted(v for v in secondary if v)


FAMILY_LAYER_FILE = common.CONFIG_DIR / "tf_family_map.tsv"


def family_layer_contract() -> dict:
    """Which file governs the Astroglia TF-family layer, stated out loud.

    The family trajectory drawn here and the direction-concordance gate A4 are an
    INTERPRETATION layer over the unbiased 879-motif profile, exactly like the
    cell-type-specific motif modules -- but they are governed by
    `config/tf_family_map.tsv`, NOT by `modules.file`
    (`config/celltype_motif_module_map.tsv`). This step reads no module-map
    accessor at all, so emptying `modules.file` cannot move A4, the trajectory or
    any primary number; emptying *this* file would. The path, the families, the
    levels and the per-level role are therefore recorded next to every family
    number this step writes, so that "emptying the module map leaves every claim
    unchanged" is never read as covering this separate input.

    The two maps agree on today's contents -- the four primary families here are
    exactly the module map's Astroglia primary modules, and the secondary family
    cannot gate either way -- and
    `script/tests/test_module_layer_invariance.py` asserts that agreement (and this
    layer's invariance to `modules.file`) against this function.
    """
    table = common.tf_family_table()
    families = sorted(str(v) for v in table.family.dropna().unique() if str(v))
    levels = sorted(str(v) for v in table.analysis_level.dropna().unique() if str(v))
    primary = sorted(
        str(v)
        for v in table.loc[table.analysis_level == "primary", "family"].dropna().unique()
        if str(v)
    )
    non_primary = [family for family in families if family not in set(primary)]
    return dict(
        source=common.display_path(FAMILY_LAYER_FILE),
        role="interpretation",
        governs=[
            "family_trajectory.tsv / family_trajectory_matrix.tsv",
            "Gate A4 (claim 'Conserved TF-family trajectory')",
        ],
        families=families,
        levels=levels,
        primary_families=primary,
        non_primary_families=non_primary,
        family_aggregation="median over the family's monomeric members of the per-species centered log2_OR",
        member_scope="monomer; a heterodimer never enters a family median",
        module_map_read=False,
        note=(
            f"the Astroglia family layer is read from {common.display_path(FAMILY_LAYER_FILE)} and "
            "not from modules.file; like the module layer it is interpretation, but it is a "
            "separate frozen input, so emptying modules.file leaves it (and A4) untouched"
        ),
    )


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


# Figures of this step that ship as PDF only: their raster copy is not part of
# the paper's deliverable set, so it is not written. Declared here, beside the
# calls it affects, so the exclusion is visible in the code rather than implicit.
PDF_ONLY_STEMS = frozenset({
    "stage_matching_margins",
    "heatmap_mouse_motif",
})


def _formats_for(stem: str, formats: list[str]) -> list[str]:
    """`formats` minus the raster copy for the figures listed in PDF_ONLY_STEMS."""
    if stem not in PDF_ONLY_STEMS:
        return formats
    return [fmt for fmt in formats if str(fmt).lower() != "png"]


def _save_figure(fig: "plt.Figure", out_dir: Path, stem: str, formats: list[str], dpi: int) -> list[str]:
    paths = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
        paths.append(str(path))
        common.log("wrote", path.name)
    plt.close(fig)
    return paths


def _style_axes(ax: "plt.Axes") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK_SECONDARY)


def select_heatmap_motifs(
    species_profile: pd.DataFrame,
    table: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    """(motif_ids to draw, representative ids, significant-union ids) for one species."""
    significant = sorted(species_profile.loc[species_profile.significant, "motif_id"].unique())
    representatives: list[str] = []
    for family in [f for f in family_order(table) if f in REPRESENTATIVE_PREFERENCE]:
        members = table[(table.family == family) & (table.member_type == "monomer")]
        by_name: dict[str, list[str]] = {}
        for row in members.itertuples():
            by_name.setdefault(row.motif_name, []).append(row.motif_id)
        chosen = [sorted(by_name[name])[0] for name in REPRESENTATIVE_PREFERENCE[family] if name in by_name]
        if len(chosen) < N_REPRESENTATIVES:
            extra = [m for m in sorted(members.motif_id.unique()) if m not in chosen]
            chosen += extra[: N_REPRESENTATIVES - len(chosen)]
        representatives += chosen[:N_REPRESENTATIVES]
    present = set(species_profile.motif_id.unique())
    representatives = [m for m in representatives if m in present]
    significant = [m for m in significant if m not in set(representatives)]
    return representatives + significant, representatives, significant


def plot_heatmap(
    species_profile: pd.DataFrame,
    table: pd.DataFrame,
    species: str,
    stage: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
    audit: list[dict],
) -> list[str]:
    motif_ids, representatives, significant = select_heatmap_motifs(species_profile, table)
    periods = [p for p in stage_order(stage)[species] if p in set(species_profile.stage)]
    if not motif_ids:
        audit.append(
            dict(
                check="heatmap_selection",
                species=species,
                stage="|".join(periods),
                model="",
                status="empty",
                detail="no significant motif and no representative motif available",
            )
        )
        return []
    audit.append(
        dict(
            check="heatmap_selection",
            species=species,
            stage="|".join(periods),
            model="",
            status="ok",
            detail=f"drawn={len(motif_ids)} (representatives={len(representatives)}, "
            f"significant_union={len(significant)}); full profile stays in the tsv",
        )
    )

    wide = species_profile.pivot_table(
        index="motif_id", columns="stage", values="log2_OR_centered", aggfunc="first"
    ).reindex(index=motif_ids, columns=periods)
    names = (
        species_profile.drop_duplicates("motif_id").set_index("motif_id").motif_name.reindex(motif_ids)
    )
    values = wide.to_numpy(dtype=float)
    limit = float(np.nanpercentile(np.abs(values), 99)) if np.isfinite(values).any() else 1.0
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0

    height = float(np.clip(0.16 * len(motif_ids) + 1.8, 4.0, 40.0))
    width = float(np.clip(1.7 * len(periods) + 3.4, 5.0, 12.0))
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    image = ax.imshow(
        values,
        cmap=LinearSegmentedColormap.from_list("astro_diverging", DIVERGING_STOPS),
        vmin=-limit,
        vmax=limit,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels([f"{species}\n{period}" for period in periods], fontsize=8)
    ax.set_yticks(range(len(motif_ids)))
    label_size = float(np.clip(170.0 / max(len(motif_ids), 1), 1.2, 8.0))
    ax.set_yticklabels(names.tolist(), fontsize=label_size)
    ax.tick_params(length=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK_SECONDARY)
    ax.set_xticks(np.arange(-0.5, len(periods), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(motif_ids), 1), minor=True)
    ax.grid(which="minor", color=GRIDLINE, linewidth=0.4 if len(motif_ids) < 120 else 0.2)
    ax.tick_params(which="minor", length=0)

    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("centered log2 OR", color=INK_SECONDARY, fontsize=8)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=7)
    bar.outline.set_visible(False)
    ax.set_title(
        f"Astroglia model motif program -- {species}\n"
        f"{len(motif_ids)} motifs drawn = union of significant motifs + "
        f"{len(representatives)} representative motifs of the four primary families",
        color=INK_PRIMARY,
        fontsize=10,
        loc="left",
    )
    stem = f"heatmap_{species}_motif"
    return _save_figure(fig, out_dir, stem, _formats_for(stem, formats), dpi)


def plot_family_trajectory(
    trajectory: pd.DataFrame,
    species: str,
    stage: pd.DataFrame,
    table: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    subset = trajectory[trajectory.species == species]
    if subset.empty:
        return []
    periods = [p for p in stage_order(stage)[species] if p in set(subset.stage)]
    windows = (
        stage[stage.species == species].set_index("rank_period").cross_window.to_dict()
    )
    families = [f for f in family_order(table) if f in set(subset.family)]
    levels = table.drop_duplicates("family").set_index("family").analysis_level.to_dict()

    fig, ax = plt.subplots(figsize=(1.5 * len(periods) + 4.0, 4.2), facecolor=SURFACE)
    _style_axes(ax)
    ax.axhline(0.0, color=BASELINE, linewidth=1.0, zorder=1)
    for index, family in enumerate(families):
        block = subset[subset.family == family].set_index("stage").reindex(periods)
        color = CATEGORICAL[index % len(CATEGORICAL)]
        primary = levels.get(family) == "primary"
        ax.plot(
            range(len(periods)),
            block.family_score.to_numpy(dtype=float),
            color=color,
            linewidth=2.0 if primary else 1.6,
            linestyle="-" if primary else (0, (4, 2)),
            marker="o" if primary else "s",
            markersize=5.0,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            label=family if primary else f"{family} (secondary)",
            zorder=3,
        )
        tail = block.family_score.to_numpy(dtype=float)[-1]
        if np.isfinite(tail):
            ax.annotate(
                family,
                xy=(len(periods) - 1, tail),
                xytext=(4, 0),
                textcoords="offset points",
                color=color,
                fontsize=8,
                va="center",
            )
    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(
        [f"{period}\n{windows.get(period, '')}" for period in periods], fontsize=8
    )
    ax.set_xlim(-0.4, len(periods) - 0.3)
    ax.set_ylabel("family score (median centered log2 OR)", color=INK_SECONDARY, fontsize=9)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="best")
    ax.set_title(
        f"Astroglia model TF-family trajectory -- {species}",
        color=INK_PRIMARY,
        fontsize=10,
        loc="left",
    )
    return _save_figure(fig, out_dir, f"family_trajectory_{species}", formats, dpi)


def plot_margins(
    observed: pd.DataFrame,
    null_test: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
    cutoff: str,
) -> list[str]:
    if observed.empty:
        return []
    labels = [f"{row.species_a}\nvs {row.species_b}" for row in observed.itertuples()]
    fig, ax = plt.subplots(figsize=(1.9 * len(observed) + 3.2, 4.2), facecolor=SURFACE)
    _style_axes(ax)
    x = np.arange(len(observed))
    ax.bar(
        x,
        observed.margin.to_numpy(dtype=float),
        width=0.55,
        color=CATEGORICAL[0],
        label=f"observed {cutoff} margin",
        zorder=2,
    )
    if not null_test.empty:
        ax.errorbar(
            x,
            null_test.null_mean.to_numpy(dtype=float),
            yerr=null_test.null_sd.to_numpy(dtype=float),
            fmt="o",
            color=INK_PRIMARY,
            markerfacecolor=SURFACE,
            markersize=6,
            linewidth=1.2,
            capsize=3,
            label="matched-random null (mean s.d.)",
            zorder=4,
        )
        for index, row in enumerate(null_test.itertuples()):
            if np.isfinite(row.null_q95):
                ax.plot(
                    [index - 0.30, index + 0.30],
                    [row.null_q95, row.null_q95],
                    color=INK_PRIMARY,
                    linewidth=2.0,
                    zorder=5,
                )
        ax.plot([], [], color=INK_PRIMARY, linewidth=2.0, label="null 95th percentile")
    ax.axhline(0.0, color=BASELINE, linewidth=1.0, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    upper = float(observed.margin.max())
    lower = float(min(0.0, observed.margin.min()))
    if not null_test.empty:
        upper = max(upper, float(null_test.null_q95.max()))
    span = max(upper - lower, 1e-9)
    ax.set_ylim(lower - 0.08 * span, upper + 0.26 * span)
    for index, row in enumerate(observed.itertuples()):
        ax.annotate(
            f"p_perm={row.p_perm:.3f}\ntop1={row.top1_accuracy:.2f}",
            xy=(index, row.margin),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            color=INK_SECONDARY,
            fontsize=8,
        )
    ax.set_ylabel("matching margin (Spearman)", color=INK_SECONDARY, fontsize=9)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        fontsize=8,
        labelcolor=INK_SECONDARY,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
    )
    ax.set_title(
        "Astroglia stage-matching margin, observed vs matched-random null",
        color=INK_PRIMARY,
        fontsize=10,
        loc="left",
    )
    fig.subplots_adjust(bottom=0.30)
    return _save_figure(fig, out_dir, "stage_matching_margins",
                        _formats_for("stage_matching_margins", formats), dpi)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    common.main_guard()
    args = parse_args(argv)
    config = common.load_config(args.config)
    if args.out_root:
        config["paths"]["results"] = str(args.out_root)

    input_dir = common.results_dir(config, "01_model_motif")
    out_dir = common.results_dir(config, "02_astroglia_model")
    scores_path = Path(args.scores) if args.scores else input_dir / "all_model_motif_scores.tsv"
    null_path = Path(args.null_scores) if args.null_scores else input_dir / "random_null_scores.tsv.gz"
    topk = int(args.topk) if args.topk else int(config["model_analysis"]["primary_topk"])
    # The cut-off this arm ran at, spelled the way a figure or gate definition
    # should spell it: v3 draws the same margin figure three times, so the label
    # must come from the arm's config, never from a hardcoded cut-off name.
    cutoff = common.topk_label(config)
    formats = [str(f) for f in config["plotting"]["formats"]]
    dpi = int(config["plotting"]["dpi"])
    alpha = float(config["model_analysis"]["fdr"])

    # Windows and species are properties of the config/period table/manifest, not
    # of this module, so they are resolved here and threaded through every
    # computation below (the stage matrix, the margin, the exact permutation, the
    # null and every output file name).
    cross_windows = common.cross_windows(config)
    single_windows = common.single_species_windows(config)
    n_windows = len(cross_windows)
    n_permutations = factorial(n_windows)
    stage_map_file = common.stage_map_path(config)

    audit: list[dict] = []
    with common.StepTimer("s02_astroglia_model"):
        scores = read_scores(scores_path)
        stage = astroglia_stages(config)
        # species_scope appends its own audit rows; it runs before the profile is
        # built so the manifest, not the score table, decides who is in.
        allowed_species, species_scope_info = species_scope(config, stage, scores, audit)
        profile, profile_audit, species_motifs = build_profile(
            scores, stage, topk, allowed_species, cross_windows
        )
        audit += profile_audit
        common.save_table(profile, out_dir / "astroglia_motif_profiles.tsv", sep="\t")

        family_table = common.tf_family_table()
        family_contract = family_layer_contract()
        # Named in the step's own audit table, never left for a reader to infer
        # from the code: which file governs the family trajectory and Gate A4.
        audit.append(
            dict(
                check="family_layer_source",
                species="",
                stage="",
                model="",
                status="ok",
                detail=(
                    f"the TF-family trajectory and Gate A4 read `{family_contract['source']}` "
                    f"(families {family_contract['families']}, primary "
                    f"{family_contract['primary_families']}); this is a separate frozen "
                    "interpretation input and NOT modules.file, so it is untouched by the "
                    "module-layer invariance of the primary numbers"
                ),
            )
        )
        # the run's windows are passed explicitly: the None fallback inside
        # `resolved_cross_windows` resolves the v1 period table and would silently
        # mask out this run's real windows
        trajectory, trajectory_wide = family_trajectory_tables(
            profile, stage, list(species_motifs), cross_windows)
        common.save_table(trajectory, out_dir / "family_trajectory.tsv", sep="\t")
        common.save_table(trajectory_wide, out_dir / "family_trajectory_matrix.tsv", sep="\t")
        # the within-species axis, single-species window included, kept SEPARATE so
        # the cross-species table above can be free of it
        common.save_table(
            family_trajectory_within_species(profile, stage, list(species_motifs), cross_windows),
            out_dir / "family_trajectory_within_species.tsv", sep="\t")

        present_windows = set(zip(profile.species, profile.cross_window))
        usable = [
            species
            for species in species_motifs
            if all((species, window) in present_windows for window in cross_windows)
        ]
        dropped = [species for species in species_motifs if species not in usable]
        for species in dropped:
            missing = [w for w in cross_windows if (species, w) not in present_windows]
            audit.append(
                dict(
                    check="cross_species_species",
                    species=species,
                    stage="|".join(missing),
                    model="",
                    status="skipped_species",
                    detail=f"missing cross-species window(s) {'|'.join(missing)} of the "
                    f"{n_windows} the period table declares; excluded from the cross-species "
                    "analysis",
                )
            )

        motif_ids: list[str] = []
        similarity = pd.DataFrame()
        wide = pd.DataFrame()
        margins: list[PairMargin] = []
        if len(usable) >= 2:
            wide = cross_window_wide(profile, usable, cross_windows)
            shared = set(wide.index)
            for species in usable:
                for window in cross_windows:
                    shared &= set(wide[(species, window)].dropna().index)
            motif_ids = sorted(shared)
            for species in usable:
                for window in cross_windows:
                    present = int(wide[(species, window)].notna().sum())
                    if present != len(motif_ids):
                        audit.append(
                            dict(
                                check="cross_window_motif_coverage",
                                species=species,
                                stage=window,
                                model="",
                                status="reduced",
                                detail=f"{present} motifs vs {len(motif_ids)} in the shared set",
                            )
                        )
            wide = wide.reindex(index=motif_ids)
            similarity = species_window_similarity(wide, usable, cross_windows)
            if not np.isfinite(similarity.to_numpy(dtype=float)).all():
                audit.append(
                    dict(
                        check="similarity_non_finite",
                        species="",
                        stage="",
                        model="",
                        status="non_finite_rho",
                        detail="a species-stage profile is constant; Spearman is undefined",
                    )
                )
            for species_a, species_b in combinations(usable, 2):
                margins.append(pair_margin(similarity, species_a, species_b, cross_windows))
        else:
            audit.append(
                dict(
                    check="cross_species_species",
                    species="|".join(usable),
                    stage="",
                    model="",
                    status="insufficient_species",
                    detail=f"fewer than two species carry all {n_windows} cross-species windows "
                    f"({', '.join(cross_windows)}) of the period table",
                )
            )

        long_rows: list[dict] = []
        matrix_rows: list[dict] = []
        for result in margins:
            rows = [f"{result.species_a}|{window}" for window in cross_windows]
            cols = [f"{result.species_b}|{window}" for window in cross_windows]
            block = similarity.loc[rows, cols]
            for window_a in cross_windows:
                for window_b in cross_windows:
                    rho, pvalue = common.spearman(
                        wide[(result.species_a, window_a)].to_numpy(dtype=float),
                        wide[(result.species_b, window_b)].to_numpy(dtype=float),
                    )
                    recorded = float(block.loc[f"{result.species_a}|{window_a}", f"{result.species_b}|{window_b}"])
                    if not np.isclose(rho, recorded, atol=1e-9, equal_nan=True):
                        raise RuntimeError(
                            "stage similarity disagrees with the "
                            f"{len(similarity)}x{len(similarity)} similarity matrix"
                        )
                    long_rows.append(
                        dict(
                            species_a=result.species_a,
                            window_a=window_a,
                            species_b=result.species_b,
                            window_b=window_b,
                            rho=rho,
                            pvalue=pvalue,
                        )
                    )
            matrix_rows.append(
                dict(species_a=result.species_a, species_b=result.species_b, matrix=result.matrix)
            )
        common.save_table(
            pd.DataFrame(
                long_rows,
                columns=["species_a", "window_a", "species_b", "window_b", "rho", "pvalue"],
            ),
            out_dir / f"stage_similarity_{n_windows}x{n_windows}.tsv",
            sep="\t",
        )

        observed = pd.DataFrame(
            [
                dict(
                    species_a=result.species_a,
                    species_b=result.species_b,
                    mean_matching=result.mean_matching,
                    mean_mismatching=result.mean_mismatching,
                    margin=result.margin,
                    p_perm=result.p_perm,
                    p_perm_min=result.p_perm_min,
                    n_permutations=result.n_permutations,
                    top1_accuracy=result.top1_accuracy,
                )
                for result in margins
            ],
            columns=[
                "species_a",
                "species_b",
                "mean_matching",
                "mean_mismatching",
                "margin",
                "p_perm",
                "p_perm_min",
                "n_permutations",
                "top1_accuracy",
            ],
        )
        common.save_table(observed, out_dir / "stage_matching_margin.tsv", sep="\t")
        for entry in matrix_rows:
            # save_table writes with index=False, so the row axis is carried as a
            # named column rather than as the frame index.
            matrix = pd.DataFrame(
                entry["matrix"], index=list(cross_windows), columns=list(cross_windows)
            )
            matrix = matrix.rename_axis(f"window_{entry['species_a']}").reset_index()
            common.save_table(
                matrix,
                out_dir / f"stage_matching_matrix_{entry['species_a']}_{entry['species_b']}.tsv",
                sep="\t",
            )

        null = read_null_scores(null_path)
        null_margin = pd.DataFrame(columns=["species_a", "species_b", "repeat", "margin"])
        null_test = pd.DataFrame(
            columns=[
                "species_a",
                "species_b",
                "margin_observed",
                "null_mean",
                "null_sd",
                "null_q95",
                "empirical_p",
                "n_repeats",
                "passes_null",
            ]
        )
        if null is not None and not observed.empty:
            frames = null_stage_frames(null, stage, usable, audit)
            null_margin = null_margins(
                frames, usable, motif_ids, args.repeats, audit, cross_windows
            )
            if null_margin.empty:
                audit.append(
                    dict(
                        check="null_margin",
                        species="",
                        stage="",
                        model="",
                        status="empty",
                        detail="no random repeat was usable for any species pair",
                    )
                )
            null_test = margin_null_test(observed, null_margin)
        common.save_table(null_margin, out_dir / "random_null_margin.tsv", sep="\t")
        common.save_table(null_test, out_dir / "margin_null_test.tsv", sep="\t")

        figures: list[str] = []
        for species in usable:
            figures += plot_heatmap(
                profile[profile.species == species],
                family_table,
                species,
                stage,
                out_dir,
                formats,
                dpi,
                audit,
            )
            figures += plot_family_trajectory(
                trajectory, species, stage, family_table, out_dir, formats, dpi
            )
        figures += plot_margins(observed, null_test, out_dir, formats, dpi, cutoff)

        representatives = {
            species: select_heatmap_motifs(profile[profile.species == species], family_table)[1]
            for species in usable
        }
        gate_a1 = bool(len(observed) and (observed.margin > 0).all())
        gate_a2 = bool(
            len(null_test)
            and null_test.n_repeats.ge(MIN_NULL_REPEATS_FOR_GATE).all()
            and null_test.passes_null.all()
        )
        # Three states, not two. A null table that is ABSENT is missing evidence:
        # PENDING, because a FAIL would be a positive claim that the random
        # control beats the observed Top-500 result. A null table that is present
        # but unusable (it holds no Astroglia draw, or no pair reaches
        # MIN_NULL_REPEATS_FOR_GATE jointly usable repeats) is a used-up control:
        # FAIL, and the note below says exactly that instead of claiming the null
        # was computed.
        has_usable_null = bool(
            len(null_test) and int(null_test.n_repeats.max()) >= MIN_NULL_REPEATS_FOR_GATE
        )
        if null is None:
            gate_a2_status = "PENDING"
        elif not has_usable_null:
            gate_a2_status = "FAIL"
        else:
            gate_a2_status = "PASS" if gate_a2 else "FAIL"
        gates = {
            "A1": dict(
                status="PASS" if gate_a1 else "FAIL",
                definition="every species pair has a model stage-matching margin > 0",
                evidence={
                    "pairs": len(observed),
                    "margins": {
                        f"{row.species_a}|{row.species_b}": round(row.margin, 6)
                        for row in observed.itertuples()
                    },
                },
            ),
            "A2": dict(
                status=gate_a2_status,
                definition=(
                    f"every species pair's observed {cutoff} margin beats the matched-"
                    "random null (observed > null 95th percentile), over at least "
                    f"{MIN_NULL_REPEATS_FOR_GATE} jointly usable null repeats per pair"
                ),
                evidence={
                    "passes_null": {
                        f"{row.species_a}|{row.species_b}": bool(row.passes_null)
                        for row in null_test.itertuples()
                    },
                    "empirical_p": {
                        f"{row.species_a}|{row.species_b}": row.empirical_p
                        for row in null_test.itertuples()
                    },
                    "repeats_used": {
                        f"{row.species_a}|{row.species_b}": int(row.n_repeats)
                        for row in null_test.itertuples()
                    },
                },
                note=(
                    "random null absent from results/01_model_motif (no file at "
                    f"{null_path}); Gate A2 is PENDING, not FAIL, until "
                    "s01_model_motif_enrichment.py has written it"
                    if null is None
                    else (
                        f"the random-null table at {null_path} is present but contains no usable "
                        f"Astroglia repeat for any of the {len(null_test)} species pairs (at least "
                        f"{MIN_NULL_REPEATS_FOR_GATE} jointly usable repeats per pair are "
                        "required); Gate A2 FAILS on that unusable control -- every pair is "
                        "listed in margin_null_test.tsv with n_repeats=0 and NaN null statistics"
                        if not has_usable_null
                        else f"computed from {null_path} over at least "
                        f"{MIN_NULL_REPEATS_FOR_GATE} jointly usable repeats per species pair"
                    )
                ),
            ),
            "A3": dict(
                status="PENDING",
                definition=(
                    "at least 2/3 of the species that entered the cross-species analysis "
                    f"({len(usable)}) have Spearman(delta model, delta chromVAR) > 0"
                ),
                note="requires chromVAR; filled by script/s05_astroglia_raw_validation.py",
            ),
            "A4": dict(
                status="PENDING",
                definition=(
                    "pre-specified TF families agree in developmental direction in >= 2/3 of the "
                    f"species that entered the cross-species analysis ({len(usable)}) and are "
                    "supported by raw chromVAR"
                ),
                note="requires chromVAR; filled by script/s05_astroglia_raw_validation.py",
            ),
        }

        # Everything below that a reader would otherwise have to infer from the
        # period table: which windows are cross-species, which belong to a single
        # species, and which periods of which species the profile actually carries.
        axis = stage_order(stage)
        window_of = {
            (row.species, str(row.rank_period)): str(row.cross_window) for row in stage.itertuples()
        }
        profile_periods = {
            species: sorted(set(profile.loc[profile.species == species, "stage"].astype(str)))
            for species in species_motifs
        }
        window_scope = {
            "stage_map": str(stage_map_file),
            "cross_species_windows": list(cross_windows),
            "single_species_windows": list(single_windows),
            "n_cross_species_windows": n_windows,
            "n_stage_permutations": n_permutations,
            "species": [
                {
                    "species": species,
                    "periods": [str(period) for period in axis[species]],
                    "cross_species_periods": [
                        str(period)
                        for period in axis[species]
                        if window_of.get((species, str(period))) in set(cross_windows)
                    ],
                    "single_species_periods": [
                        str(period)
                        for period in axis[species]
                        if window_of.get((species, str(period))) in set(single_windows)
                    ],
                    "periods_in_profile": profile_periods.get(species, []),
                    "in_cross_species_analysis": species in usable,
                }
                for species in axis
            ],
        }
        periods_text = "; ".join(
            f"{species} -> {'|'.join(periods)}" for species, periods in profile_periods.items()
        )
        wider = sorted(
            species
            for species in usable
            if len(profile_periods.get(species, [])) > n_windows
        )
        if wider:
            null_asymmetry = (
                "This is asymmetric with the observed profile"
                + ("" if len(wider) == 1 else "s")
                + " of "
                + ", ".join(wider)
                + ", whose period table adds a single-species window ("
                + ", ".join(single_windows)
                + ") that is part of the within-species axis but is never part of a cross-species "
                "comparison, so the observed profile removes a baseline over more periods than "
                "the null does. A species whose profile carries exactly the cross windows centers "
                "over the same number of periods in the observed and the null path."
            )
        else:
            null_asymmetry = (
                "No species carries a period beyond the cross windows, so the observed and the "
                "null paths center over the same periods."
            )

        summary = {
            "step": "s02_astroglia_model",
            "analysis": "Astroglia",
            "model_only_part": "readme sections 8 to 14 (A2, A4 model half, A5, A6, A7)",
            "input": {"scores": str(scores_path), "null_scores": str(null_path), "topk": topk},
            "fdr": alpha,
            "species": usable,
            "species_excluded": dropped,
            "stage_map": {
                species: [
                    dict(stage=row.rank_period, cross_window=row.cross_window)
                    for row in stage[stage.species == species].itertuples()
                ]
                for species in stage.species.unique()
            },
            "window_scope": window_scope,
            "species_filter": species_scope_info,
            # The family layer's own governing file, recorded next to every family
            # number this step writes: it is interpretation, but it is NOT
            # `modules.file`, so the module-layer invariance statement above does
            # not cover it. `family_layer_contract()` is the single source of this
            # statement and the invariance test asserts it.
            "family_layer": family_layer_contract(),
            "cross_windows": list(cross_windows),
            "single_species_windows": list(single_windows),
            "single_species_excluded_from_cross_species": bool(single_windows),
            "n_motifs_cross_species": len(motif_ids),
            "n_motifs_per_species": {species: len(ids) for species, ids in species_motifs.items()},
            "centering_scope": "all_species_periods",
            # The window names the baseline removal actually used, so the scope is
            # readable without inferring it from the period table: every period of
            # that species present in the profile -- the cross-species windows AND
            # the single-species window(s).
            "centering_windows": {
                "cross_species_windows": list(cross_windows),
                "single_species_windows": list(single_windows),
                "per_species_periods": profile_periods,
            },
            "centering_definition": (
                "E*_{s,t,m} = E_{s,t,m} - mean_t(E_{s,t,m}) with t running over every period of "
                "species s present in the Astroglia profile, the single-species window(s) "
                f"{list(single_windows)} included; the periods each species carries are "
                f"[{periods_text}]. This column is the within-species baseline: it is what the "
                "per-species heatmap and the within-species family trajectory are drawn from. The "
                "cross-species family trajectory and the "
                f"{n_windows}x{n_windows} stage similarity are NOT built from it -- they read "
                "log2_OR_centered_cross, the per-motif baseline removed over the cross-species "
                f"windows only ({', '.join(cross_windows)}), so that a single-species window "
                "cannot move a cross-species number (see family_centering_scope)."
            ),
            "null_centering_scope": (
                f"the {n_windows} cross-species windows only ({', '.join(cross_windows)}) within "
                "each repeat. The per-motif mean is removed within each repeat over those windows "
                "alone, exactly as for the observed profile, so the two margins stay comparable. "
                + null_asymmetry
            ),
            "n_permutations": n_permutations,
            "permutation_equivalence_note": (
                f"The readme asks for all {n_windows}! x {n_windows}! = {n_permutations ** 2} "
                "paired stage permutations. Fixing the row labels and enumerating the "
                f"{n_permutations} column orderings covers every distinct value of the margin: "
                "relabelling the fixed side applies the same permutation to both row and column "
                "indices, which leaves the set of matching (diagonal) pairs and hence the margin "
                f"unchanged. common.permutation_margin_pvalue therefore enumerates "
                f"{n_permutations} orderings and n_permutations is recorded as {n_permutations}."
            ),
            "matching_margin_definition": "mean(diagonal rho) - mean(offdiagonal rho)",
            "top1_accuracy_definition": "per pair, fraction of row stages whose row argmax is the matching column",
            "family_centering_scope": f"cross_species_windows_only ({'|'.join(cross_windows)})",
            "family_order": family_order(family_table),
            "family_aggregation": "common.family_trajectory: per-species motif baseline removal, then median over monomer members",
            "representative_motifs": representatives,
            "representative_rule": (
                "frozen name preference per primary family (RFX1/RFX2/RFX3, NFIA/NFIB/NFIC, "
                "NFYA/NFYB/NFYC, SOX2/SOX9/SOX10) taking up to 3 monomer members, filled "
                "alphabetically if a preferred name is absent; a display choice only, never used "
                "in a statistic"
            ),
            "heatmap_rule": (
                "rows = union of significant motifs (FDR < alpha and log2_OR > 0) of that species "
                "plus the representative motifs of the four primary families; the full 879-motif "
                "profile is written to astroglia_motif_profiles.tsv and never drawn in full"
            ),
            "random_null": {
                "available": null is not None,
                "repeats_used": int(null_test.n_repeats.max()) if len(null_test) else 0,
                "min_repeats_for_gate": MIN_NULL_REPEATS_FOR_GATE,
                "repeat_cap": args.repeats,
                "empirical_p_definition": "(1 + #{null margin >= observed margin}) / (1 + n)",
                "passes_null_definition": "observed margin > null 95th percentile",
            },
            "gates": gates,
            "interpretation_boundary": (
                f"log2_OR is the enrichment of the model's {cutoff} peaks for a motif against a "
                "GC/length matched background; its FDR is not evidence of TF activity, and a "
                "positive margin is a statement about the model's ranking of regulatory sequence, "
                "not about chromatin accessibility."
            ),
            "figures": figures,
        }
        common.save_table(
            pd.DataFrame(
                audit,
                columns=["check", "species", "stage", "model", "status", "detail"],
            ),
            out_dir / "astroglia_model_audit.tsv",
            sep="\t",
        )
        summary["files"] = sorted(
            {path.name for path in out_dir.iterdir() if path.is_file()} | {"astroglia_summary.json"}
        )
        common.save_json(summary, out_dir / "astroglia_summary.json")
        common.log(
            "gates",
            f"A1={gates['A1']['status']}",
            f"A2={gates['A2']['status']}",
            f"A3={gates['A3']['status']}",
            f"A4={gates['A4']['status']}",
        )


if __name__ == "__main__":
    main()
