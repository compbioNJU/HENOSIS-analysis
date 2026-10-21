"""Analysis C, model side: oligodendrocyte regulatory convergence across species.

Why this file exists
--------------------
readme.md section 28 freezes Analysis C as an *independent repeat* of the
Purkinje model pipeline: an n_species x 879 log2-OR profile matrix -> the
n_species x n_species cross-species Spearman similarity -> matched-random
null -> unbiased module recurrence. Two properties of that design are easy to
lose and are therefore explicit here.

  * No module is pre-selected. Recurrence is evaluated over every module the
    frozen, cell-type-specific map declares for Oligo and the results table
    reports ALL of them, recurrent or not. The SOX / OLIG / MYRF / NKX panel is
    emitted *afterwards* as a descriptive sanity table that fed no choice.
  * The null is not recomputed. s01 already produced, per model, the
    matched-random enrichment profiles through exactly the same foreground /
    background machinery. This step only re-reads those profiles and reduces
    each repeat to one mean-pairwise similarity, which is the single quantity
    the convergence gate O1 needs. Recomputing would mean re-running FIMO over a
    full species universe; that is both forbidden and unnecessary.

The similarity statistic is Spearman over the 879 log2-OR values, which is rank
based and therefore invariant to the per-species centering that the Astroglia
analysis uses; no centering is applied here (readme section 23 specifies
Spearman on E_{i,*} directly).

The species set is never a literal. `common.apply_species_filter` enforces the
configured ELIGIBILITY floors (>= `eligibility.min_target_cells` cells of the
analysis celltype AND >= `eligibility.min_ranked_peaks` ranked peaks) and returns
an audit frame naming every dropped model with the counts that caused the drop.
The step then builds its species list from the KEPT models only and names every
removal in `oligo_input_audit.tsv` and in `oligo_summary.json`
(`species_excluded`). Everything the run produces -- the matrix, the similarity
table, the null, the module recurrence columns -- is sized from
`len(species)` of that kept set, never from a module-level count.

THE MODULE LAYER IS INTERPRETATION ONLY (V2_PLAN section 1). The primary
evidence is the unbiased 879-motif profile, the matched-random null and the
chromVAR validation; the module panel only NAMES what the primary analysis
already found. Concretely:

  * every primary artefact of this step -- the 879-motif matrix, the species x
    species Spearman similarity and its p-values, the observed convergence, the
    matched-null convergence and the convergence null test, i.e. gate O1 -- is
    computed from `profile_long` before a single module is looked at, and is
    byte-identical whether or not the module map has rows;
  * gate O2 is the ONE gate that reads the module map: that is what makes it an
    interpretation-layer verdict, it is decided on `primary`-level modules only
    (`common.module_is_gating` is False for `secondary`/`comparator`), and it can
    never be flipped on by an emptied map -- an empty panel leaves O2 PENDING,
    never PASS;
  * the words "family"/"TF-family" do not appear in this step's numbers: the v1
    project-wide `config/tf_family_map.tsv` is not read here at all.

Gates: O1 observed mean pairwise similarity > 95th percentile of the
matched-random null; O2 at least one recurrent PRIMARY module; O3 left PENDING for
s09. O1 is only evaluated when the null file offers enough usable repeats for the
percentile rule to be defensible (see MIN_NULL_REPEAT_FRACTION); a degenerate
null is reported PENDING with an audit row rather than thresholded.

WORDING RULE (V2_PLAN section 9, frozen before any result was seen). The
`bHLH_Ebox` module is reported as a "bHLH/E-box associated program", NEVER as
"OLIG2 activity": its JASPAR members share one E-box and are not distinguishable
by PWM, so the module median cannot attribute the signal to any single factor.
Because the module median also cannot speak to an individual factor, OLIG1 /
OLIG2 / OLIG3 are additionally reported PER MOTIF in
`module_member_motifs_per_species.tsv`. The `NFI` module is emitted but labelled
`comparator` and is never counted as evidence of conservation.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import common  # noqa: E402

STEP = "08_oligo_model"
ANALYSIS = "Oligo"
STEM = "oligo"
# The species that may enter this analysis are NOT a literal here: they are the
# species of the filtered manifest (common.apply_species_filter), ordered by the
# frozen roster in common.SPECIES_ORDER. The config's `eligibility` block decides
# membership; this file only keeps the display order deterministic.
#
# V2_PLAN section 13 froze the module-recurrence rule as
#     K = max(4, ceil(2N/3))
# with N = the species actually kept by eligibility. `script/s06_purkinje_model.py`
# carries the identical pair, so the two twins cannot drift apart. The
# denominator is `len(species)` of THIS run -- never a literal count; the floor
# guarantees that losing a species cannot lower the bar below 4.
RECURRENT_SPECIES_FRACTION = 2 / 3
MIN_RECURRENT_SPECIES = 4

# Names that spell a derived species count out in words, so a caption or a label
# can name the scope it belongs to instead of asserting a fixed number. A
# six-species run still produces the published v1 wording.
_NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
}

# The frozen wording rule of V2_PLAN section 9. It is a CONSTANT so that the
# summary JSON, the module tables and the figure captions cannot drift apart,
# and it is deliberately not a per-run choice: the JASPAR bHLH members of the
# Oligo panel share one E-box, so the module median is a bHLH/E-box associated
# program and never a statement about one factor.
BHLH_EBOX_CLAIM = (
    "bHLH/E-box associated program: the JASPAR bHLH motifs of this module share "
    "a single E-box and are not distinguishable by PWM, so the module score is "
    "reported as a bHLH/E-box associated program and NEVER as OLIG2 activity"
)
# OLIG1/2/3 are members of the E-box module but the module median cannot speak
# to any one of them, so they are additionally reported per motif. The prefix is
# matched case-insensitively against the catalog name.
OLIG_FACTOR_PREFIX = "olig"
# The one module that carries the wording rule above. The name is the frozen map's
# own module name (V2_PLAN section 9), and nothing is selected or dropped by it:
# it only decides which sentence a reported module is captioned with.
EBOX_MODULE = "bHLH_Ebox"
# Frozen floor on the null coverage gate O1 needs before it may be evaluated:
# at least half of random_control.n_repeats, and never fewer repeats than
# 1/fdr, because below that the smallest attainable empirical p (1/(n+1)) is
# already above the frozen fdr and a PASS could not be defended by its own
# p-value.
MIN_NULL_REPEAT_FRACTION = 0.5

# Descriptive only (readme section 28): these names never filter anything.
SANITY_FACTORS = (
    "SOX2", "SOX4", "SOX8", "SOX9", "SOX10", "SOX11", "SOX17", "SOX21",
    "Olig1", "Olig2", "Olig3", "Myrf", "Nkx2-2", "Nkx6-1",
)

# Figure palette: sequential slot (blue) for the null mass, diverging
# blue<->red around a neutral gray midpoint for the signed correlation matrix.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#d9d8d3"
ACCENT = "#2a78d6"
OBSERVED_INK = "#e34948"
NEUTRAL_MID = "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list(
    "oligo_convergence",
    ["#0d366b", "#2a78d6", "#9ec5f4", NEUTRAL_MID, "#f2b0af", "#e34948", "#8f1f1e"],
)

REQUIRED_SCORE_COLUMNS = {
    "analysis", "species", "motif_id", "log2_OR", "FDR", "significant",
}


# --------------------------------------------------------------------------
# audit bookkeeping
# --------------------------------------------------------------------------


@dataclass
class Audit:
    """Every skip, drop or mismatch observed by this step, never silent."""

    rows: list[dict] = field(default_factory=list)

    def note(self, check: str, subject: str, value: object, status: str, detail: str = "") -> None:
        self.rows.append(
            dict(check=check, subject=subject, value=value, status=status, detail=detail)
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.rows, columns=["check", "subject", "value", "status", "detail"]
        )

    def summary(self) -> list[dict]:
        return list(self.rows)


# --------------------------------------------------------------------------
# species scope (config-driven, never a literal)
# --------------------------------------------------------------------------


def recurrence_threshold(n_species: int) -> int:
    """`K = max(4, ceil(2N/3))` -- V2_PLAN section 13, against the species in scope.

    `max(MIN_RECURRENT_SPECIES, ceil(RECURRENT_SPECIES_FRACTION * n_species))`, the
    identical expression `s06_purkinje_model.recurrence_threshold` uses. Six species
    give K = max(4, 4) = 4 and five give max(4, 4) = 4, so a dropped species cannot
    relax the rule -- it can only make it unreachable, and that is reported rather
    than absorbed. The threshold is a function of the kept species count alone: no
    module, level or map content can move it.
    """
    return max(MIN_RECURRENT_SPECIES, int(np.ceil(RECURRENT_SPECIES_FRACTION * n_species)))


def number_word(count: int) -> str:
    """Spell a small count out, so a label can name its scope instead of asserting it."""
    return _NUMBER_WORDS.get(int(count), str(int(count)))


def scoped_species(manifest: pd.DataFrame) -> list[str]:
    """Species that still own an Oligo model after the species filter.

    The set comes from the (filtered) manifest -- what the configured filter left
    -- and the order from the project's frozen species axis, so the same code
    yields the manifest's species for any config. A species outside that axis is
    appended rather than dropped, because nothing may vanish silently.
    """
    present = set(manifest.loc[manifest.analysis == ANALYSIS, "species"].astype(str))
    ordered = [name for name in common.SPECIES_ORDER if name in present]
    return ordered + sorted(present - set(ordered))


def filter_exclusions(filter_audit: pd.DataFrame) -> pd.DataFrame:
    """The species_filter drops that belong to this analysis, as an audit frame."""
    if filter_audit.empty:
        return filter_audit
    return filter_audit[filter_audit.analysis == ANALYSIS].reset_index(drop=True)


def species_census(
    config: dict, org_celltype: str, candidates: list[str], kept: list[str],
    manifest: pd.DataFrame,
) -> list[dict]:
    """Per-species counts behind the ELIGIBILITY decision, for both criteria.

    Descriptive bookkeeping for the summary: it is what lets a reader of
    `oligo_summary.json` see that the eligibility floors were evaluated rather
    than bypassed (e.g. macaque at 432 oligo cells against a floor of 100, and the
    number of ranked peaks the same model carries against the peak floor). The
    decision itself is `common.apply_species_filter`'s; this only reports the
    counts behind it, for BOTH criteria, so a drop can never be attributed to the
    wrong floor.
    """
    block = config.get("eligibility") or config.get("species_filter")
    if not block:
        return []
    column = block.get("celltype_column", "Org_Celltype")
    cell_floor = common.min_cells_per_species(config)
    peak_floor = common.min_ranked_peaks(config)
    counts = common.species_cell_counts(config, column)
    per_species = {
        str(row.species): int(row.n_cells)
        for row in counts.itertuples()
        if str(row.celltype) == org_celltype
    }
    peaks: dict[str, list[int]] = {}
    for row in manifest.itertuples():
        peaks.setdefault(str(row.species), []).append(int(row.n_ranked))
    in_scope = set(kept)
    return [
        dict(
            species=name,
            n_cells=per_species.get(name, 0),
            min_target_cells=cell_floor,
            celltype_column=column,
            ranked_peaks=sorted(peaks.get(name, [])),
            min_ranked_peaks=peak_floor,
            kept=bool(name in in_scope),
        )
        for name in candidates
    ]


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def load_oligo_scores(
    path: Path, audit: Audit, requested_topk: int, species: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Return (long profile frame, species x motif log2-OR matrix, Top-k used).

    The wide matrix carries the log2-OR of every motif of every species and is
    the object every downstream statistic is computed on. The third return value
    is the Top-k the input rows actually carry: the frozen primary Top-k is only
    a request, and s01 also writes the same schema with `_top250`/`_top1000`
    suffixes, so every "Top-k" label of this run must name the data, not the
    request.

    `species` is the scope the species filter left for this analysis. Every
    species in it must be present (the cross-species statistic is undefined
    otherwise), while a species outside it is dropped from the profile and
    audited -- the filter's decision is respected, never second-guessed here.
    """
    frame = pd.read_csv(path, sep="\t")
    missing = REQUIRED_SCORE_COLUMNS - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path}: missing columns {sorted(missing)}")

    audit.note("input", str(path), int(len(frame)), "info", "raw rows read")
    frame = frame[frame.analysis == ANALYSIS].copy()
    audit.note(
        "input_filter", "analysis==Oligo", int(len(frame)), "info",
        "rows of other analyses are not this step's input",
    )
    if frame.empty:
        raise RuntimeError(f"{path}: no rows with analysis == {ANALYSIS}")

    # The frozen Top-k is a request; the rows in the file are the data.
    effective_topk = requested_topk
    if "topk" not in frame.columns:
        audit.note(
            "topk", "topk column", "absent", "warn",
            f"cannot verify that the rows are Top{requested_topk}; the Top-k label "
            "is taken from the request",
        )
    else:
        unlabelled = int(frame.topk.isna().sum())
        if unlabelled:
            audit.note(
                "topk", "rows without a topk value", unlabelled, "dropped",
                "a row whose Top-k is unknown cannot be labelled or selected",
            )
            frame = frame[frame.topk.notna()].copy()
        topk_values = sorted(int(v) for v in frame.topk.unique())
        audit.note("topk", "topk_values_in_input", topk_values, "info", "")
        if requested_topk in topk_values:
            frame = frame[frame.topk.eq(requested_topk)].copy()
            audit.note(
                "topk", f"rows_kept_topk_{requested_topk}", int(len(frame)), "info",
                "rows of any other Top-k in the same file are excluded deterministically",
            )
        elif len(topk_values) == 1:
            effective_topk = topk_values[0]
            audit.note(
                "topk", "Top-k substituted", effective_topk, "substituted",
                f"input holds only Top{effective_topk} rows while the frozen primary is "
                f"Top{requested_topk}; every Top-k label of this run names Top{effective_topk}",
            )
        elif len(topk_values) > 1:
            raise RuntimeError(
                f"{path}: topk values {topk_values} but none is the requested "
                f"Top{requested_topk}; refusing to pick one by input row order"
            )
    if frame.empty:
        raise RuntimeError(f"{path}: no analysable row left after the Top-k selection")

    present = [s for s in frame.species.unique()]
    unknown = sorted(set(present) - set(common.SPECIES_ORDER))
    if unknown:
        audit.note(
            "species", "not_in_frozen_order", len(unknown), "dropped",
            f"species absent from SPECIES_ORDER: {unknown}",
        )
    # Present in the scores but not in the analysis scope: either the configured
    # species filter dropped it or it was never in the frozen roster above.
    out_of_scope = [s for s in present if s not in set(species)]
    if out_of_scope:
        audit.note(
            "eligibility_scope", "species outside the analysis scope", len(out_of_scope), "dropped",
            f"{out_of_scope} have Oligo rows in {path.name} but are not in the "
            f"{len(species)}-species scope {'|'.join(species)}; their rows are excluded",
        )
    missing_species = [s for s in species if s not in set(present)]
    if missing_species:
        raise RuntimeError(
            f"{path}: Oligo models missing for {missing_species}; the cross-species "
            f"convergence statistic is undefined without every species in scope "
            f"({len(species)}: {'|'.join(species)})"
        )
    frame = frame[frame.species.isin(species)].copy()

    duplicated = frame.duplicated(["species", "motif_id"]).sum()
    if duplicated:
        audit.note(
            "duplicate", "species x motif_id", int(duplicated), "dropped",
            "kept the first occurrence; a duplicated motif key would otherwise "
            "silently unbalance the Spearman vector",
        )
        frame = frame.drop_duplicates(["species", "motif_id"], keep="first")

    # Motif sets must agree across species, otherwise the profiles are not a
    # comparable 879-vector population. Any mismatch is recorded either way.
    per_species = {s: set(g.motif_id) for s, g in frame.groupby("species")}
    shared = set.intersection(*per_species.values())
    union = set.union(*per_species.values())
    if shared != union:
        for species, motifs in sorted(per_species.items()):
            only_here = sorted(motifs - shared)
            audit.note(
                "motif_intersection", species, len(only_here), "dropped",
                f"motifs absent from at least one other species: "
                f"{only_here[:10]}{'...' if len(only_here) > 10 else ''}",
            )
        audit.note(
            "motif_intersection", "shared_motifs", len(shared), "info",
            f"union had {len(union)} motifs; similarity uses the shared subset",
        )
    if not shared:
        raise RuntimeError(
            f"the {len(species)} species in scope share no motif; cannot build a "
            "profile matrix"
        )

    frame = frame[frame.motif_id.isin(shared)]
    motifs = sorted(shared)
    frame = frame.sort_values(["species", "motif_id"]).reset_index(drop=True)

    names = frame.groupby("motif_id").motif_name.nunique()
    inconsistent = int((names > 1).sum())
    if inconsistent:
        audit.note(
            "motif_name", "motif_id with >1 name", inconsistent, "warn",
            "the first name encountered is used",
        )

    wide = (
        frame.pivot(index="species", columns="motif_id", values="log2_OR")
        .reindex(index=species, columns=motifs)
    )
    if wide.isna().any().any():
        raise RuntimeError("profile matrix has holes after the intersection filter")

    audit.note("input", "n_motifs", len(motifs), "info", f"{len(species)} x {len(motifs)} matrix")
    return frame, wide, effective_topk


def load_null(
    path: Path | None, motifs: list[str], audit: Audit, species: list[str]
) -> pd.DataFrame:
    """Per-repeat mean pairwise similarity of the matched-random null's profiles.

    s01's `random_null_scores.tsv.gz` holds one log2-OR profile per (model,
    repeat). A repeat only enters the null when every species of the analysis
    scope contributes a complete profile, because the statistic under test is a
    cross-species one and the scope is whatever the species filter left.

    Audit rows from here carry the check label `random_null`, never the bare
    string `null`: `null` is one of pandas' default na_values, so writing it
    would make the label read back as NaN and `audit.check == "null"` would
    silently match nothing.
    """
    columns = [
        "repeat", "n_species", "n_pairs", "mean_pairwise_similarity",
        "median_pairwise_similarity",
    ]
    if path is None or not Path(path).exists():
        audit.note(
            "random_null", str(path), "missing", "unavailable",
            "s01 random_null_scores.tsv.gz not found; O1 cannot be evaluated",
        )
        return pd.DataFrame(columns=columns)

    null = pd.read_csv(path, sep="\t")
    audit.note("random_null", str(path), int(len(null)), "info", "raw rows read")
    if "analysis" in null.columns:
        null = null[null.analysis == ANALYSIS]
    elif "model" in null.columns:
        null = null[null.model.str.startswith(f"{ANALYSIS}__")]
    else:
        raise RuntimeError(f"{path}: needs an `analysis` or `model` column")

    if null.empty:
        audit.note(
            "random_null", f"{ANALYSIS} rows in the null table", 0, "warn",
            "no Oligo row survived the filter; O1 has nothing to be tested against",
        )
        return pd.DataFrame(columns=columns)

    for column in ("repeat", "motif_id", "log2_OR"):
        if column not in null.columns:
            raise RuntimeError(f"{path}: missing column {column}")

    # s01 keys the null by model ("Analysis__species__period"); a species column
    # is used when present and otherwise recovered from the model name.
    if "species" not in null.columns:
        parts = null.model.str.split("__", expand=True)
        if parts.shape[1] < 2:
            raise RuntimeError(f"{path}: cannot recover species from the model column")
        null = null.assign(species=parts[1])
        audit.note("random_null", "species_column", "derived from model", "info", "")

    n_models = null[["model"]].drop_duplicates().shape[0] if "model" in null.columns else np.nan
    audit.note("random_null", "oligo_models_in_null", n_models, "info", "")

    # A model of a species the filter dropped is read past, not filtered out in
    # silence: it is the same exclusion the scores table records, seen from the
    # null file. For a run with no drop this row is never emitted.
    if "model" in null.columns:
        outside = sorted(
            set(null.loc[~null.species.isin(species), "model"].astype(str))
        )
        if outside:
            audit.note(
                "eligibility_scope", "null models outside the analysis scope",
                len(outside), "dropped",
                f"{outside[:6]} belong to species that are not in the "
                f"{len(species)}-species scope {'|'.join(species)}; their repeats are not read",
            )

    null = null[null.species.isin(species) & null.motif_id.isin(set(motifs))]
    n_dropped = int(
        null.groupby(["model", "repeat", "species"]).motif_id.nunique().lt(len(motifs)).sum()
    ) if "model" in null.columns else 0
    audit.note(
        "random_null", "species x repeat x model units with incomplete motif vectors",
        n_dropped, "dropped" if n_dropped else "info",
        "a unit is dropped only if it cannot fill the shared motif vector",
    )

    repeats = sorted(null.repeat.unique())
    if not repeats:
        audit.note(
            "random_null", "oligo repeats in the null table", 0, "warn",
            "no Oligo row survived the filter; O1 falls back to PENDING",
        )
        return pd.DataFrame(columns=columns)

    matrix = null.pivot_table(
        index=["repeat", "species"], columns="motif_id", values="log2_OR", aggfunc="first"
    ).reindex(columns=motifs)

    rows: list[dict] = []
    for repeat in repeats:
        try:
            block = matrix.xs(repeat, level="repeat").reindex(index=species)
        except KeyError:
            continue
        if block.isna().any().any():
            continue
        profiles = {s: block.loc[s].to_numpy(dtype=float) for s in species}
        rhos = pairwise_rhos(profiles)
        rows.append(
            dict(
                repeat=int(repeat),
                n_species=len(species),
                n_pairs=len(rhos),
                mean_pairwise_similarity=float(rhos.mean()) if len(rhos) else np.nan,
                median_pairwise_similarity=float(np.median(rhos)) if len(rhos) else np.nan,
            )
        )

    dropped = len(repeats) - len(rows)
    if dropped:
        audit.note(
            "random_null", "repeats_dropped", int(dropped), "dropped",
            "repeats where at least one species had no complete profile",
        )
    audit.note("random_null", "repeats_used", len(rows), "info", f"of {len(repeats)} present")
    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def pairwise_rhos(profiles: dict[str, np.ndarray]) -> np.ndarray:
    keys = [s for s in profiles]
    values = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            values.append(common.spearman(profiles[keys[i]], profiles[keys[j]])[0])
    return np.asarray(values, dtype=float)


def similarity_and_pvalues(
    profiles: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = list(profiles)
    rho = common.similarity_matrix(profiles)
    pvalue = pd.DataFrame(np.ones((len(keys), len(keys))), index=keys, columns=keys)
    for i, key_i in enumerate(keys):
        for j, key_j in enumerate(keys):
            if j <= i:
                continue
            _, p = common.spearman(profiles[key_i], profiles[key_j])
            pvalue.loc[key_i, key_j] = pvalue.loc[key_j, key_i] = p
    return rho, pvalue


def observed_convergence(rho: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    keys = list(rho.columns)
    pairs = [
        dict(species_a=keys[i], species_b=keys[j], rho=float(rho.iloc[i, j]))
        for i in range(len(keys))
        for j in range(i + 1, len(keys))
    ]
    values = np.asarray([p["rho"] for p in pairs], dtype=float)
    off = rho.to_numpy()[~np.eye(len(keys), dtype=bool)]
    best, worst = int(np.argmax(values)), int(np.argmin(values))
    clean = off[np.isfinite(off)]
    summary = {
        "n_species": len(keys),
        "n_pairs": len(pairs),
        "mean_pairwise_similarity": float(clean.mean()),
        "median_pairwise_similarity": float(np.median(clean)),
        "sd_pairwise_similarity": float(clean.std(ddof=1)) if clean.size > 1 else np.nan,
        "min_pairwise_similarity": float(clean.min()),
        "max_pairwise_similarity": float(clean.max()),
        "min_pair": f"{pairs[worst]['species_a']}|{pairs[worst]['species_b']}",
        "max_pair": f"{pairs[best]['species_a']}|{pairs[best]['species_b']}",
    }
    table = pd.DataFrame(
        [
            dict(statistic="n_species", value=int(summary["n_species"]), detail=""),
            dict(
                statistic="n_pairs", value=int(summary["n_pairs"]),
                detail="unordered species pairs",
            ),
            dict(statistic="mean_pairwise_similarity", value=summary["mean_pairwise_similarity"], detail="O1 statistic"),
            dict(statistic="median_pairwise_similarity", value=summary["median_pairwise_similarity"], detail=""),
            dict(statistic="sd_pairwise_similarity", value=summary["sd_pairwise_similarity"], detail=""),
            dict(statistic="min_pairwise_similarity", value=summary["min_pairwise_similarity"], detail=summary["min_pair"]),
            dict(statistic="max_pairwise_similarity", value=summary["max_pairwise_similarity"], detail=summary["max_pair"]),
        ]
    )
    return summary, table


# --------------------------------------------------------------------------
# module layer (cell-type-specific, INTERPRETATION ONLY)
# --------------------------------------------------------------------------


def flagged(frame: pd.DataFrame, flag: str) -> pd.DataFrame:
    """Boolean-select a possibly-empty frame WITHOUT losing its columns.

    pandas drops the column index when an empty frame is indexed by an empty
    boolean Series, so `recurrent["module"]` would raise on the emptied-module-map
    run. Returning a copy of the empty frame instead keeps that run emitting the
    same column names as the populated one.
    """
    if frame.empty:
        return frame.copy()
    return frame[frame[flag].astype(bool)].reset_index(drop=True)


def column_list(frame: pd.DataFrame, name: str) -> list[str]:
    """The named column as a list of strings, or [] when the frame has no such column."""
    return frame[name].astype(str).tolist() if name in frame.columns else []


def o3_candidate_modules(config: dict) -> list[str] | None:
    """The frozen O3 candidate set for this arm, read straight from the config.

    `oligo_o3_candidate_modules` is an explicit list of MODULE names. It is what
    replaced the gating filter this step used to apply to build
    `recurrent_families.tsv` (`common.module_is_gating(config, level)`), so no
    `level` of the module map can add or remove a candidate any more -- which is
    what kept a re-levelling of a module from moving gate O2 and the O3 candidate
    set s09 rebuilds from that file.

    Returns `None` ONLY when the key is absent altogether (a v1 config that never
    declared it); the caller then keeps the legacy level-based filter so v1 does
    not move. An EMPTY list is a real value -- no module is eligible -- and is
    deliberately NOT treated as "absent".
    """
    if "oligo_o3_candidate_modules" not in config:
        return None
    return [str(name) for name in (config.get("oligo_o3_candidate_modules") or [])]


def selected_modules(frame: pd.DataFrame, modules: list[str]) -> pd.DataFrame:
    """Rows whose `module` is one of `modules`, WITHOUT losing columns when empty.

    Same empty-frame guarantee as `flagged`: an empty candidate set must yield the
    published column names, not an IndexError.
    """
    if frame.empty:
        return frame.copy()
    keep = set(str(name) for name in modules)
    return frame[frame.module.astype(str).isin(keep)].reset_index(drop=True)


def module_panel(config: dict, audit: Audit) -> pd.DataFrame:
    """The frozen Oligo module panel, straight from the cell-type module map.

    One row per (module, level, monomeric member) with the level's gating flag.
    `gating` is `common.module_is_gating(config, level)`, which is False for the
    `secondary` and `comparator` levels, so a module can never gate by accident
    and the rule that decides it lives in one place.

    A map that exists but declares no Oligo row -- the shape a zero-row map takes
    -- is NOT an error: the panel is empty, every module table is empty, and gate
    O2 is reported PENDING. That is what lets this step be re-run with an emptied
    module map to show the primary artefacts do not move.
    """
    columns = ["module", "level", "gating", "motif_name", "motif_id", "member_type"]
    try:
        table = common.celltype_module_table(config)
    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as error:
        audit.note(
            "module_map", f"modules for {ANALYSIS}", "unreadable", "unavailable",
            f"{error}; the module layer is not evaluated and gate O2 is PENDING",
        )
        return pd.DataFrame(columns=columns)

    table = table[table.celltype == ANALYSIS]
    n_modules = table.module.nunique() if len(table) else 0
    if table.empty:
        audit.note(
            "module_map", f"modules for {ANALYSIS}", 0, "unavailable",
            "the frozen module map declares no Oligo module; the module layer is "
            "not evaluated and gate O2 is PENDING",
        )
        return pd.DataFrame(columns=columns)

    panel = table[table.member_type == "monomer"].copy()
    heterodimers = int((table.member_type != "monomer").sum())
    duplicate = int(panel.duplicated(["module", "motif_id"]).sum())
    if duplicate:
        audit.note(
            "module_map", "duplicate (module, motif_id) rows", duplicate, "dropped",
            "first occurrence kept; a duplicated member would weight the median",
        )
        panel = panel.drop_duplicates(["module", "motif_id"], keep="first")
    panel["gating"] = [
        common.module_is_gating(config, str(level)) for level in panel.level
    ]
    audit.note(
        "module_map", "oligo modules in the frozen panel", int(n_modules), "info",
        "levels: "
        + ", ".join(
            f"{level}={n}"
            for level, n in table.drop_duplicates("module").level.value_counts().items()
        ),
    )
    audit.note(
        "module_map", "heterodimer member rows excluded from module scores",
        heterodimers, "excluded",
        "a module median must be built from binding models of one factor, not from "
        "a fusion of two (common.module_members monomers_only=True)",
    )
    empty_modules = sorted(set(table.module) - set(panel.module))
    if empty_modules:
        audit.note(
            "module_map", "modules with no usable monomer", len(empty_modules), "warn",
            f"{empty_modules} are reported with a missing score rather than dropped",
        )
    return panel[columns].reset_index(drop=True)


def module_claim_wording(module: str, level: str, gating: bool) -> str:
    """The frozen wording a module may be reported under (V2_PLAN sections 8/9).

    A non-gating module is labelled with its own level and the fact that it is not
    evidence of conservation; the E-box module gets the bHLH/E-box sentence; every
    other primary module is named as a "<module>-associated motif program", the
    same discipline the panel applies to SOX, because a motif family is not an
    activity measurement.
    """
    if not gating:
        return (
            f"{level} module: reported for completeness only; it never gates and it "
            "is NOT counted as evidence of cross-species conservation"
        )
    if str(module) == EBOX_MODULE:
        return BHLH_EBOX_CLAIM
    return f"{module}-associated motif program"


def module_scores(
    profile: pd.DataFrame, panel: pd.DataFrame, audit: Audit
) -> pd.DataFrame:
    """MEMBER-level table (`module_member_scores.tsv`): one row per member x species.

    The module aggregate `F_abs = median_m log2(OR)` is taken from here by
    `module_species_scores`, so the member rows a module score was computed over
    stay inspectable instead of being collapsed away.

    No centering is applied here. `common.family_trajectory` is deliberately not
    used either: it removes a within-species stage mean, and every species of this
    analysis has exactly one period, so every centered value would be 0 and the
    "module score > 0" test would degenerate into floating-point noise.
    """
    columns = [
        "module", "level", "gating", "motif_id", "motif_name",
        "species", "log2_OR", "FDR", "significant",
    ]
    if panel.empty:
        return pd.DataFrame(columns=columns)

    annotated = panel[["module", "level", "gating", "motif_id"]]
    merged = profile.merge(annotated, on="motif_id", how="inner")
    covered = sorted(set(merged.motif_id))
    absent = sorted(set(panel.motif_id) - set(covered))
    if absent:
        audit.note(
            "module_map", "panel members absent from the profile", len(absent), "warn",
            f"they are not in any module score: {absent[:10]}{'...' if len(absent) > 10 else ''}",
        )
    unassigned = sorted(set(profile.motif_id) - set(panel.motif_id))
    audit.note(
        "module_map", "profile motifs in no Oligo module", len(unassigned), "info",
        "they stay in the primary 879-motif profile; the module layer simply does "
        "not name them",
    )
    return merged[columns].sort_values(["module", "species", "motif_id"]).reset_index(drop=True)


def module_recurrence(
    scores: pd.DataFrame, panel: pd.DataFrame, species: list[str], min_species: int
) -> pd.DataFrame:
    """One row per module: the frozen recurrence rule, per species and overall.

    V2_PLAN section 13: a module is recurrent iff

        #{s : F_abs(s, module) > 0} >= K
          AND
        #{s : some monomeric member has FDR < alpha AND log2-OR > 0} >= K

    with K = max(4, ceil(2N/3)) over the kept species. The score used is the
    ABSOLUTE `F_abs`; the centered score (`F_centered = F_abs - mean_s F_abs`) is
    written to its own table and is for the heatmap only -- centering can flip the
    sign of a module that is uniformly elevated and must never decide recurrence.

    `gating` is carried from the panel, but it is descriptive only: which modules
    gate O2 is decided by the frozen `oligo_o3_candidate_modules` list the caller
    reads from the config, never by this column. Every module of the panel is
    present in the output, recurrent or not: recurrence is a descriptive rule, not
    a filter.
    """
    columns = [
        "module", "level", "gating", "n_members", "n_species_scored",
        "n_species_score_positive", "n_species_with_significant_member",
        "mean_module_score", "recurrent", "claim_wording",
    ]
    if panel.empty:
        return pd.DataFrame(columns=columns)

    levels = panel.drop_duplicates("module").set_index("module")[["level", "gating"]]
    n_members = panel.groupby("module").motif_id.nunique()
    ordered = list(levels.index)

    if scores.empty:
        score_wide = pd.DataFrame(index=ordered, columns=species, dtype=float)
        sig_wide = pd.DataFrame(index=ordered, columns=species, dtype=float)
    else:
        score_wide = scores.pivot(index="module", columns="species", values="module_score")
        sig_wide = scores.pivot(index="module", columns="species", values="n_sig_members")
        score_wide = score_wide.reindex(index=ordered, columns=species)
        sig_wide = sig_wide.reindex(index=ordered, columns=species)

    positive = (score_wide > 0).sum(axis=1)
    # V2_PLAN's second arm is an EXISTENCE test ("some motif ... "), so one
    # significant member is enough and the count of them is reported alongside.
    significant = (sig_wide.fillna(0) >= 1).sum(axis=1)

    out = pd.DataFrame(
        {
            "module": ordered,
            "level": [str(levels.level.get(m, "")) for m in ordered],
            "gating": [bool(levels.gating.get(m, False)) for m in ordered],
            "n_members": [int(n_members.get(m, 0)) for m in ordered],
            "n_species_scored": score_wide.notna().sum(axis=1).astype(int).to_numpy(),
            "n_species_score_positive": positive.astype(int).to_numpy(),
            "n_species_with_significant_member": significant.astype(int).to_numpy(),
            "mean_module_score": score_wide.mean(axis=1).to_numpy(),
        }
    )
    out["recurrent"] = (
        (out.n_species_score_positive >= min_species)
        & (out.n_species_with_significant_member >= min_species)
    )
    out["claim_wording"] = [
        module_claim_wording(m, lvl, bool(gate))
        for m, lvl, gate in zip(out.module, out.level, out.gating)
    ]
    for name in species:
        out[f"score_{name}"] = score_wide[name].to_numpy()
        out[f"n_sig_{name}"] = sig_wide[name].fillna(0).astype(int).to_numpy()
    return out.sort_values(
        ["recurrent", "n_species_score_positive", "mean_module_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def module_species_scores(member_scores: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse the member-level table to one ABSOLUTE score per (species, module).

    `F_abs = median_m log2-OR` over the module's monomeric members present in the
    profile, together with the number of members the median was taken over and the
    number of members that are significant in the frozen direction. This is the
    table the recurrence rule and the centered heatmap both read, so the two can
    never disagree about what a module's score is.
    """
    columns = [
        "species", "module", "level", "gating", "module_score", "n_members", "n_sig_members",
    ]
    if member_scores.empty:
        return pd.DataFrame(columns=columns)
    grouped = member_scores.groupby(["species", "module"], as_index=False).agg(
        module_score=("log2_OR", "median"),
        n_members=("log2_OR", "size"),
        n_sig_members=("significant", "sum"),
    )
    annotations = panel.drop_duplicates("module").set_index("module")[["level", "gating"]]
    grouped["level"] = grouped.module.map(annotations.level).astype(str)
    grouped["gating"] = grouped.module.map(annotations.gating).fillna(False).astype(bool)
    return grouped[columns].sort_values(["module", "species"]).reset_index(drop=True)


def centered_module_scores(
    aggregated: pd.DataFrame, species: list[str]
) -> pd.DataFrame:
    """`F_centered = F_abs - mean_s F_abs`, per module. Heatmap only.

    The cross-species mean is removed per MODULE (the axis a heatmap colours
    along), and the result is written to `module_scores_centered.tsv`. It is never
    fed to the recurrence rule: a module that is uniformly elevated in every
    species centers to exactly zero, so centering would mislabel exactly the
    conserved program this analysis is looking for.
    """
    if aggregated.empty:
        return pd.DataFrame(columns=["module", "level", "gating", *species])
    wide = aggregated.pivot(index="module", columns="species", values="module_score")
    wide = wide.reindex(columns=species)
    centered = wide.sub(wide.mean(axis=1), axis=0).reset_index()
    return centered[["module", *species]]


def module_member_motif_table(scores: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Every module member, per species, per motif -- the per-motif read-out.

    The module median cannot speak to an individual factor, so OLIG1 / OLIG2 /
    OLIG3 (members of the E-box module) are additionally reported per motif here,
    in every species, with their own log2-OR and FDR. The `is_olig_factor` flag is
    derived from the catalog name, not from a hand-written motif list.
    """
    columns = [
        "module", "level", "gating", "motif_name", "motif_id", "species",
        "log2_OR", "FDR", "significant", "is_olig_factor",
    ]
    if scores.empty:
        return pd.DataFrame(columns=columns)
    frame = scores.copy()
    frame["is_olig_factor"] = (
        frame.motif_name.astype(str).str.lower().str.startswith(OLIG_FACTOR_PREFIX)
    )
    return frame.sort_values(
        ["is_olig_factor", "module", "motif_name", "species"], ascending=[False, True, True, True]
    ).reset_index(drop=True)[columns]


# --------------------------------------------------------------------------
# sanity check (descriptive only)
# --------------------------------------------------------------------------


def sanity_check_table(
    profile: pd.DataFrame, catalog: pd.DataFrame, audit: Audit, species: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Per-species log2-OR / FDR of the expected oligodendrocyte factors.

    readme section 28: SOX / OLIG / MYRF / NKX may only be looked at *after*
    the unbiased result exists. Nothing in this script reads this table.
    """
    lookup = {name.lower(): (mid, name) for mid, name in zip(catalog.motif_id, catalog.motif_name)}
    frame = profile.copy()
    frame["_key"] = frame.motif_name.str.lower()

    rows = []
    missing = []
    in_catalog_but_not_profiled: list[str] = []
    for requested in SANITY_FACTORS:
        hit = lookup.get(requested.lower())
        if hit is None:
            missing.append(requested)
            continue
        motif_id, motif_name = hit
        sub = frame[frame.motif_id == motif_id]
        if sub.empty:
            in_catalog_but_not_profiled.append(requested)
        n_sig = int(sub.significant.sum())
        for name in species:
            cell = sub[sub.species == name]
            if cell.empty:
                rows.append(
                    dict(
                        requested_name=requested, motif_name=motif_name, motif_id=motif_id,
                        species=name, log2_OR=np.nan, FDR=np.nan, significant=False,
                        n_species_significant=n_sig,
                        evidence_role="descriptive_only_not_used_for_any_choice",
                    )
                )
                continue
            cell = cell.iloc[0]
            rows.append(
                dict(
                    requested_name=requested, motif_name=motif_name, motif_id=motif_id,
                    species=name, log2_OR=float(cell.log2_OR), FDR=float(cell.FDR),
                    significant=bool(cell.significant), n_species_significant=n_sig,
                    evidence_role="descriptive_only_not_used_for_any_choice",
                )
            )
    audit.note(
        "sanity_check", "requested factors absent from the catalog", len(missing),
        "unavailable" if missing else "info", ", ".join(missing) if missing else "",
    )
    if in_catalog_but_not_profiled:
        audit.note(
            "sanity_check", "requested factors in the catalog but not in the profile",
            len(in_catalog_but_not_profiled), "warn",
            ", ".join(in_catalog_but_not_profiled),
        )
    return pd.DataFrame(rows), missing


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def save_figure(fig: plt.Figure, stem: Path, formats: list[str], dpi: int) -> None:
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, facecolor=SURFACE)
        common.log("wrote", path.relative_to(common.ROOT) if str(path).startswith(str(common.ROOT)) else path)


def plot_similarity_heatmap(
    rho: pd.DataFrame, stem: Path, formats: list[str], dpi: int, topk: int
) -> None:
    values = rho.to_numpy(dtype=float)
    limit = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 0.0
    limit = min(1.0, max(0.1, limit))
    keys = list(rho.columns)
    fig, ax = plt.subplots(figsize=(6.6, 5.8), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    image = ax.imshow(values, cmap=DIVERGING, norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit))
    for i in range(len(keys)):
        for j in range(len(keys)):
            normed = (values[i, j] + limit) / (2 * limit) if limit else 0.5
            ink = "#ffffff" if abs(normed - 0.5) > 0.28 else INK
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color=ink)
    ax.set_xticks(range(len(keys)), labels=keys, rotation=45, ha="right", color=INK_SECONDARY)
    ax.set_yticks(range(len(keys)), labels=keys, color=INK_SECONDARY)
    ax.set_xticks(np.arange(-0.5, len(keys), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(keys), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        f"Oligo model Top-{topk}: cross-species motif-profile similarity",
        color=INK, fontsize=10,
    )
    bar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.03)
    bar.set_label("Spearman rho over the motif log2-OR profile", color=INK_SECONDARY, fontsize=8.5)
    bar.outline.set_visible(False)
    save_figure(fig, stem, formats, dpi)
    plt.close(fig)


def plot_convergence_null(
    null_values: np.ndarray, observed: float, q95: float, topk: int, null_name: str,
    stem: Path, formats: list[str], dpi: int, species: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    if null_values.size:
        ax.hist(null_values, bins=min(40, max(10, null_values.size // 5)),
                color=ACCENT, edgecolor=SURFACE, linewidth=0.6,
                label=f"{null_name}, {null_values.size} repeats")
    if np.isfinite(observed):
        ax.axvline(observed, color=OBSERVED_INK, linewidth=2.0,
                   label=f"observed Top{topk} ({observed:.3f})")
    if np.isfinite(q95):
        ax.axvline(q95, color=INK_SECONDARY, linewidth=1.6, linestyle="--",
                   label=f"null 95th percentile ({q95:.3f})")
    # The caption names the scope the run actually had, rendered from the count
    # rather than asserting one: a six-species run still reads "the six species",
    # exactly as the published v1 figure does.
    ax.set_xlabel(
        f"mean pairwise Spearman rho across the {number_word(len(species))} species",
        color=INK_SECONDARY, fontsize=9,
    )
    ax.set_ylabel("repeats", color=INK_SECONDARY, fontsize=9)
    ax.set_title(
        f"Oligo convergence against the {null_name}",
        color=INK, fontsize=10,
    )
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5)
    legend = ax.legend(frameon=False, fontsize=8.5)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    save_figure(fig, stem, formats, dpi)
    plt.close(fig)


def plot_module_heatmap(
    centered: pd.DataFrame, recurrence: pd.DataFrame, species: list[str],
    stem: Path, formats: list[str], dpi: int, topk: int,
) -> None:
    """Centered module scores -- the ONLY consumer of the centered score.

    The colour is `F_abs - mean_s F_abs`, which is what makes a heatmap readable
    across modules of different overall magnitude. It is labelled as centered on
    the figure itself, and the row annotation carries each module's level and
    recurrence so that a centered value is never mistaken for the statistic
    recurrence was decided on.
    """
    caption = (
        f"Module captions (every Oligo module, recurrent or not). {BHLH_EBOX_CLAIM}. "
        + " | ".join(f"{m}: {w}" for m, w in zip(recurrence.get("module", []), recurrence.get("claim_wording", [])))
    )
    if centered.empty:
        return caption
    by_module = recurrence.set_index("module")
    recurrent_flag = by_module.recurrent.to_dict()
    level_of = by_module.level.astype(str).to_dict()
    values = centered[species].to_numpy(dtype=float)
    finite = np.isfinite(values)
    limit = float(np.nanmax(np.abs(values))) if finite.any() else 0.0
    limit = min(2.0, max(0.05, limit))
    rows = [
        f"{m}  [{level_of.get(m, '?')}{'*, recurrent' if recurrent_flag.get(m) else ''}]"
        for m in centered.module
    ]
    fig, ax = plt.subplots(figsize=(1.4 + 1.1 * len(species), 0.55 * len(rows) + 2.2),
                           constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    image = ax.imshow(values, cmap=DIVERGING, norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
                      aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not finite[i, j]:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7.5, color=INK_SECONDARY)
                continue
            normed = (values[i, j] + limit) / (2 * limit)
            ink = "#ffffff" if abs(normed - 0.5) > 0.30 else INK
            ax.text(j, i, f"{values[i, j]:+.2f}", ha="center", va="center", fontsize=8, color=ink)
    ax.set_xticks(range(len(species)), labels=species, rotation=45, ha="right", color=INK_SECONDARY)
    ax.set_yticks(range(len(rows)), labels=rows, color=INK_SECONDARY, fontsize=8.5)
    ax.set_xticks(np.arange(-0.5, len(species), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        f"Oligo model Top-{topk}: CENTERED module scores (heatmap only)\n"
        f"colour = F_abs - mean over the {number_word(len(species))} species; "
        "recurrence is decided on the ABSOLUTE F_abs, not on this value",
        color=INK, fontsize=9.5,
    )
    ax.set_xlabel("* = recurrent module; secondary/comparator modules never gate", color=INK_SECONDARY, fontsize=8)
    bar = fig.colorbar(image, ax=ax, shrink=0.85, pad=0.02)
    bar.set_label("centered median log2-OR over monomeric members", color=INK_SECONDARY, fontsize=8)
    bar.outline.set_visible(False)
    save_figure(fig, stem, formats, dpi)
    plt.close(fig)
    return caption


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # The shared flags (--config/--topk/--repeats/--force) must parse here like
    # everywhere else: run_all.sh forwards its own arguments to every stage, so a
    # hand-rolled parser would abort the whole pipeline at this step.
    common.add_common_args(parser)
    parser.add_argument("--scores", default=None, help="override the s01 model motif scores table")
    parser.add_argument("--null-scores", default=None, help="override the s01 random null table")
    parser.add_argument(
        "--module-map", default=None,
        help="override modules.file; pointing it at a map with no Oligo row runs the "
             "step with the module layer disabled, which must not move a primary artefact",
    )
    parser.add_argument("--out-dir", default=None, help="override the output directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = common.load_config(args.config)
    audit = Audit()
    # Whether THIS config declares an eligibility block (v2) or the older species
    # filter (v1). A config with neither applies no filter, so the bookkeeping
    # below is not part of its summary at all. The block is additive only where
    # the config asked for it.
    has_species_filter = bool(config.get("eligibility") or config.get("species_filter"))
    if args.module_map:
        config.setdefault("modules", {})["file"] = str(args.module_map)

    if args.force:
        common.log("--force ignored: s08 recomputes nothing and caches nothing")
    if args.repeats is not None:
        common.log(
            "--repeats ignored:", int(args.repeats),
            "the null is read from s01's random_null_scores.tsv.gz, never recomputed",
        )

    results_root = common.config_path(config, "results")
    scores_path = Path(args.scores).resolve() if args.scores else results_root / "01_model_motif" / "all_model_motif_scores.tsv"
    null_path = Path(args.null_scores).resolve() if args.null_scores else results_root / "01_model_motif" / "random_null_scores.tsv.gz"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else common.results_dir(config, STEP)
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha = float(config["model_analysis"]["fdr"])
    requested_topk = (
        int(args.topk) if args.topk is not None else int(config["model_analysis"]["primary_topk"])
    )
    n_repeats_expected = int(config["random_control"]["n_repeats"])
    n_peaks_expected = int(config["random_control"]["n_peaks"])
    # The null's name is the ARM's own, from its config: v3 draws every repeat at
    # the model's OBSERVED table width, so "random-500" (or "random-K", which is
    # the nominal `n_peaks_expected` above) would misname it in both size and kind.
    null_name = common.null_label(config)
    dpi = int(config["plotting"]["dpi"])
    formats = list(config["plotting"]["formats"])

    with common.StepTimer(f"s08 {ANALYSIS} model"):
        if not scores_path.exists():
            raise RuntimeError(f"{scores_path} not found; run s01 first")
        catalog = common.motif_catalog(config)

        # ---- species scope (from the filtered manifest, never a literal) -----
        # `apply_species_filter` is a no-op for a config without `species_filter`
        # (v1 keeps its six Oligo models) and drops any species below the floor
        # under v2. Everything below reads the KEPT manifest, so no downstream
        # statistic can reintroduce a dropped species through the score table.
        org_celltype = common.ANALYSIS_ORG_CELLTYPE[ANALYSIS]
        unfiltered_manifest, _ = common.load_rankings(config)
        manifest, filter_audit = common.apply_species_filter(config, unfiltered_manifest)
        species = scoped_species(manifest)
        n_species_all = len(scoped_species(unfiltered_manifest))
        if not species:
            raise ValueError(
                f"the species filter left no {ANALYSIS} model to analyse; check "
                "species_filter.min_cells_per_species against the model manifest"
            )
        if len(species) < 2:
            # One species has no pair, so the convergence statistic of gate O1 and
            # the recurrence rule of gate O2 are both undefined: fail loudly rather
            # than publish an empty 1x1 matrix and a NaN gate.
            raise ValueError(
                f"the species filter left {len(species)} species ({'|'.join(species)}) for "
                f"the {ANALYSIS} analysis; a cross-species statistic needs at least two. "
                "Raise species_filter.min_cells_per_species or check the per-species counts."
            )
        exclusions = filter_exclusions(filter_audit)
        for row in exclusions.itertuples():
            audit.note(
                "species_eligibility",
                str(row.model),
                "eligibility",
                "WARN",
                f"{row.action}; model {row.model} is excluded from the {ANALYSIS} "
                f"analysis (both floors are evaluated independently: "
                f"min_target_cells={int(row.min_cells)} AND "
                f"min_ranked_peaks={int(getattr(row, 'min_ranked_peaks', 0))})",
            )
        species_excluded = [
            dict(
                species=str(row.species),
                celltype=str(row.celltype),
                model=str(row.model),
                n_cells=int(row.n_cells),
                min_cells=int(row.min_cells),
                n_ranked_peaks=int(getattr(row, "n_ranked", 0)),
                min_ranked_peaks=int(getattr(row, "min_ranked_peaks", 0)),
                reason=str(row.action),
            )
            for row in exclusions.itertuples()
        ]
        species_cells = species_census(
            config, org_celltype, scoped_species(unfiltered_manifest), species,
            unfiltered_manifest[unfiltered_manifest.analysis == ANALYSIS],
        )
        common.log(
            "species scope", f"{len(species)}: {'|'.join(species)}",
            "excluded by eligibility:",
            [f"{row['species']} ({row['reason']})" for row in species_excluded] or "none",
        )

        profile_long, wide, topk = load_oligo_scores(scores_path, audit, requested_topk, species)

        # significance is taken from s01 (the frozen upstream product) but
        # cross-checked against the frozen rule so a divergence is visible.
        recomputed = common.add_significance(
            profile_long[["FDR", "log2_OR"]].copy(), alpha
        ).significant
        mismatches = int((recomputed != profile_long.significant).sum())
        audit.note(
            "significant", "rows where s01.significant != (FDR<alpha & log2_OR>0)",
            mismatches, "warn" if mismatches else "info", f"alpha={alpha}",
        )
        if mismatches:
            profile_long = profile_long.assign(significant=recomputed)

        n_motifs = wide.shape[1]
        if n_motifs != int(config["motif_database"]["n_motifs"]):
            audit.note(
                "motif_set", "n_motifs in profile", n_motifs, "warn",
                f"frozen motif database holds {config['motif_database']['n_motifs']}",
            )

        profiles = {name: wide.loc[name].to_numpy(dtype=float) for name in species}
        rho, pvalue = similarity_and_pvalues(profiles)
        observed, observed_table = observed_convergence(rho)

        null_frame = load_null(null_path, list(wide.columns), audit, species)
        null_values = (
            null_frame.mean_pairwise_similarity.to_numpy(dtype=float)
            if len(null_frame) else np.asarray([], dtype=float)
        )

        min_species = recurrence_threshold(len(species))
        if len(species) < min_species:
            # The frozen rule needs more species than this scope has; gate O2 can
            # then never pass, which must be stated rather than discovered.
            audit.note(
                "module_map", "recurrence rule vs scope", len(species), "warn",
                f"K = max({MIN_RECURRENT_SPECIES}, ceil(2N/3)) = {min_species} while the "
                f"eligibility filter left {len(species)} species in scope; no module can "
                "be recurrent",
            )

        # ---- module layer (INTERPRETATION ONLY) --------------------------
        # Everything below reads `profile_long` and the frozen module map. It never
        # touches `wide`, `rho`, `pvalue`, `observed`, `null_values` or gate O1 --
        # those are computed above and are byte-identical with an emptied map.
        panel = module_panel(config, audit)
        module_long = module_scores(profile_long, panel, audit)
        aggregated = module_species_scores(module_long, panel)
        recurrence = module_recurrence(aggregated, panel, species, min_species)
        centered = centered_module_scores(aggregated, species)
        member_motifs = module_member_motif_table(module_long, panel)
        # Gate O2's candidate set comes from the CONFIG, not from the module map's
        # `level`. `oligo_o3_candidate_modules` is the frozen list this step reads;
        # it names explicitly the modules the old gating filter
        # (`common.module_is_gating`) admitted, so re-levelling a module can no
        # longer add or remove a candidate -- and can no longer move gate O2 or the
        # O3 candidate set s09 rebuilds from recurrent_families.tsv.
        frozen_candidates = o3_candidate_modules(config)
        if frozen_candidates is None:
            # Key absent (a v1 config that never declared it): legacy level-based
            # filter, so v1 does not move. The v2 config declares the list, so this
            # branch is never taken by the published analysis.
            eligible_recurrence = flagged(recurrence, "gating")
        else:
            eligible_recurrence = selected_modules(recurrence, frozen_candidates)
        # The recurrence rule itself is unchanged and is still what decides which
        # eligible module is counted.
        recurrent = flagged(eligible_recurrence, "recurrent")
        all_recurrent = flagged(recurrence, "recurrent")
        non_gating_recurrent = sorted(
            set(column_list(all_recurrent, "module"))
            - set(column_list(recurrent, "module"))
        )
        if non_gating_recurrent:
            audit.note(
                "module_map", "recurrent non-gating modules", len(non_gating_recurrent),
                "excluded",
                f"{non_gating_recurrent} clear the recurrence rule but their level does "
                "not gate (common.module_is_gating); they are reported, never counted "
                "towards gate O2",
            )
        olig_rows = (
            member_motifs[member_motifs.is_olig_factor] if not member_motifs.empty
            else member_motifs
        )
        olig_factors = sorted(set(olig_rows.motif_name.astype(str))) if not olig_rows.empty else []
        olig_factor_modules = sorted(set(olig_rows.module.astype(str))) if not olig_rows.empty else []
        audit.note(
            "module_map", "OLIG factors reported per motif", len(olig_factors),
            "info" if olig_factors else "unavailable",
            (
                f"{olig_factors} sit in module(s) {olig_factor_modules} and are "
                "additionally reported per motif in "
                "module_member_motifs_per_species.tsv; the module median cannot speak "
                "to any one of them"
                if olig_factors else
                f"no OLIG factor is a member of any module of this panel "
                f"(the E-box module is '{EBOX_MODULE}'), so there is nothing to report "
                "per motif"
            ),
        )

        # ---- gates -------------------------------------------------------
        min_repeats = max(
            int(np.ceil(1.0 / alpha)),
            int(np.ceil(MIN_NULL_REPEAT_FRACTION * n_repeats_expected)),
        )
        powered = bool(null_values.size >= min_repeats)
        if null_values.size:
            q95_descriptive = float(np.percentile(null_values, 95))
            null_mean = float(null_values.mean())
            null_sd = float(null_values.std(ddof=1)) if null_values.size > 1 else float("nan")
            empirical_p = float(
                (int(np.sum(null_values >= observed["mean_pairwise_similarity"])) + 1)
                / (null_values.size + 1)
            )
        else:
            q95_descriptive = null_mean = null_sd = empirical_p = float("nan")

        if powered:
            q95 = q95_descriptive
            passes = bool(observed["mean_pairwise_similarity"] > q95)
            o1_status = "PASS" if passes else "FAIL"
            o1_note = ""
        else:
            # Too few usable repeats: the 95th percentile of a handful of draws is
            # the near-maximum of noise, and 1/(n+1) > fdr means a PASS could not
            # be defended by its own p-value. Report PENDING, never threshold it.
            q95 = float("nan")
            passes = None
            o1_status = "PENDING"
            o1_note = (
                f"{int(null_values.size)} usable null repeats is below the frozen minimum of "
                f"{min_repeats} (>= 1/fdr={alpha} and >= {MIN_NULL_REPEAT_FRACTION:.0%} of "
                f"random_control.n_repeats={n_repeats_expected}); O1 is not evaluated"
            )
            audit.note(
                "gate", "O1 null coverage (insufficient)", int(null_values.size), "warn", o1_note,
            )

        if panel.empty:
            # No module is declared for Oligo: the panel makes no claim, so O2 is
            # PENDING rather than FAIL, and an emptied map can therefore never turn
            # a gate ON. This is also the state the module-isolation rerun lands in.
            o2_status, o2_passes = "PENDING", None
            o2_note = (
                "the frozen module map declares no gating Oligo module; the module "
                "layer is unavailable and gate O2 is not evaluated"
            )
        elif len(recurrent) >= 1:
            o2_status, o2_passes, o2_note = "PASS", True, ""
        else:
            o2_status, o2_passes = "FAIL", False
            o2_note = (
                "no primary module clears the recurrence rule"
                if not non_gating_recurrent else
                "no PRIMARY module clears the recurrence rule; "
                f"{non_gating_recurrent} clear it but do not gate"
            )

        null_test = pd.DataFrame(
            [
                dict(
                    comparison=f"mean_pairwise_similarity_Top{topk}_vs_matched_random",
                    observed=observed["mean_pairwise_similarity"],
                    null_mean=null_mean if powered else np.nan,
                    null_sd=null_sd if powered else np.nan,
                    null_q95=q95,
                    empirical_p=empirical_p if powered else np.nan,
                    n_repeats=int(null_values.size),
                    passes_null=("" if passes is None else bool(passes)),
                    gate="O1",
                    status=o1_status,
                )
            ]
        )

        # ---- write the contract outputs ----------------------------------
        matrix_table = wide.reset_index().rename(columns={"index": "species"})
        audit.note(
            "output", f"{STEM}_motif_matrix.tsv",
            f"{matrix_table.shape[0]}x{matrix_table.shape[1] - 1}", "info",
            f"species rows x motif columns (readme section 28 {len(species)} x "
            f"{n_motifs} orientation); s06_purkinje_model.py ships the Purkinje twin "
            "transposed (motif rows x species columns), so the two files are "
            "transposes of one another",
        )
        common.save_table(matrix_table, out_dir / f"{STEM}_motif_matrix.tsv", sep="\t")
        rho_out = rho.copy()
        rho_out.index.name = "species"
        # The stem states the matrix size from `len(species)`, never a literal: a
        # run whose species filter left m species ships m x m, so a reader (or a
        # glob) can never be handed a table labelled with a count it does not
        # have. Six in scope still yields the published v1 name byte for byte.
        similarity_stem = f"{STEM}_similarity_{len(species)}x{len(species)}.tsv"
        common.save_table(rho_out.reset_index(), out_dir / similarity_stem, sep="\t")
        p_out = pvalue.copy()
        p_out.index.name = "species"
        common.save_table(p_out.reset_index(), out_dir / f"{STEM}_similarity_pvalues.tsv", sep="\t")
        common.save_table(observed_table, out_dir / "convergence_observed.tsv", sep="\t")
        common.save_table(null_frame, out_dir / "random_null_convergence.tsv", sep="\t")
        common.save_table(null_test, out_dir / "convergence_null_test.tsv", sep="\t")

        # ---- module-layer outputs (interpretation; no primary number in here) --
        common.save_table(module_long, out_dir / "module_member_scores.tsv", sep="\t")
        common.save_table(recurrence, out_dir / "module_recurrence_all.tsv", sep="\t")
        common.save_table(recurrent, out_dir / "recurrent_modules.tsv", sep="\t")
        common.save_table(centered, out_dir / "module_scores_centered.tsv", sep="\t")
        common.save_table(aggregated, out_dir / "module_scores.tsv", sep="\t")
        common.save_table(member_motifs, out_dir / "module_member_motifs_per_species.tsv", sep="\t")
        # Two aliases of the module tables, kept ONLY because the sibling raw
        # validation step (s09) still reads that path and this file may not edit it.
        # They carry MODULE rows under the old key name `family`, so a reader that
        # still says "family" is reading the new module panel, not the retired
        # project-wide tf_family_map. Report the dependency in the summary.
        for frame, name, canonical in (
            (recurrent, "recurrent_families.tsv", "recurrent_modules.tsv"),
            (recurrence, "family_recurrence_all.tsv", "module_recurrence_all.tsv"),
        ):
            alias = frame.copy()
            if not alias.empty:
                alias["family"] = alias["module"]
                alias["table_semantics"] = "module_layer_v2_not_tf_family_map"
            common.save_table(alias, out_dir / name, sep="\t")
        audit.note(
            "output", "legacy family-named module tables", 2, "substituted",
            "recurrent_families.tsv / family_recurrence_all.tsv are ALIASES of "
            "recurrent_modules.tsv / module_recurrence_all.tsv, written only because "
            "script/s09_oligo_raw_validation.py still reads that path and this step may "
            "not edit it. They carry MODULE rows (module and family columns both name a "
            "module) and the retired project-wide tf_family_map is not their source.",
        )

        sanity, missing_factors = sanity_check_table(profile_long, catalog, audit, species)
        common.save_table(sanity, out_dir / "sanity_check_expected_factors.tsv", sep="\t")
        common.save_table(audit.frame(), out_dir / f"{STEM}_input_audit.tsv", sep="\t")

        plot_similarity_heatmap(rho, out_dir / "similarity_heatmap", formats, dpi, topk)
        plot_convergence_null(
            null_values, observed["mean_pairwise_similarity"], q95, topk, null_name,
            out_dir / "convergence_null", formats, dpi, species,
        )
        module_caption = plot_module_heatmap(
            centered, recurrence, species, out_dir / "module_score_heatmap", formats, dpi, topk,
        )

        n_pairs_kept = len(species) * (len(species) - 1) // 2
        n_pairs_all = n_species_all * (n_species_all - 1) // 2
        floor = common.min_cells_per_species(config)
        # Dropping a species cannot be made free: the observed mean and the null both
        # lose that species' pairs, so state the loss instead of adjusting the gate.
        # Proof material for the isolation claim: the sha256 of every PRIMARY
        # artefact this step writes. Two runs -- one with the real module map, one
        # with a map that has no Oligo row -- must agree on all of them, and the
        # hashes are published here so that comparison needs no second run to be
        # trusted. Nothing module-derived is in this list.
        primary_artefacts = {
            f"{STEM}_motif_matrix.tsv": out_dir / f"{STEM}_motif_matrix.tsv",
            similarity_stem: out_dir / similarity_stem,
            f"{STEM}_similarity_pvalues.tsv": out_dir / f"{STEM}_similarity_pvalues.tsv",
            "convergence_observed.tsv": out_dir / "convergence_observed.tsv",
            "random_null_convergence.tsv": out_dir / "random_null_convergence.tsv",
            "convergence_null_test.tsv": out_dir / "convergence_null_test.tsv",
        }
        primary_sha256 = {
            name: common.sha256_file(path) for name, path in primary_artefacts.items()
        }

        scope_note = (
            ""
            if not species_excluded
            else (
                f"{len(species_excluded)} species excluded by eligibility "
                f"(min_target_cells={floor}, "
                f"min_ranked_peaks={common.min_ranked_peaks(config)}): "
                + "; ".join(
                    f"{row['species']} ({row['n_cells']} {row['celltype']} cells, "
                    f"{row['n_ranked_peaks']} ranked peaks; {row['reason']})"
                    for row in species_excluded
                )
                + f". The convergence statistic is a mean over {n_pairs_kept} species pairs "
                f"instead of {n_pairs_all}, and the matched null is rebuilt on the same "
                "reduced species set; observed and null stay directly comparable, but the "
                "test carries less information than the unfiltered one. The thresholds were "
                "NOT relaxed to compensate: O1 still requires the observed mean to exceed "
                "the 95th percentile, and O2 still requires K = max(4, ceil(2N/3)) species "
                "over the reduced scope."
            )
        )

        summary = {
            "step": STEP,
            "analysis": ANALYSIS,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "inputs": {
                "model_motif_scores": str(scores_path),
                "model_motif_scores_sha256": common.sha256_file(scores_path),
                "random_null_scores": str(null_path),
                "random_null_scores_sha256": (
                    common.sha256_file(null_path) if Path(null_path).exists() else None
                ),
            },
            # The species scope of THIS run: the filtered manifest decides
            # membership, so these keys are the record of what gates O1 and O2
            # were run over and of what the species filter removed. They exist
            # only when the config declares a filter -- otherwise the species
            # filter removed nothing and a v1 summary must keep its published
            # key set (and key order) exactly.
            **(
                {
                    "species_order": species,
                    "n_species": len(species),
                    "n_species_before_filter": n_species_all,
                    "species_excluded": species_excluded,
                    "eligibility": dict(
                        applied=has_species_filter,
                        min_target_cells=int(floor),
                        min_ranked_peaks=common.min_ranked_peaks(config),
                        celltype_column=(config.get("eligibility") or {}).get("celltype_column"),
                        rule=(
                            "a species enters the Oligo analysis only if its model has "
                            ">= min_target_cells cells of the analysis celltype AND "
                            ">= min_ranked_peaks ranked peaks; the two floors are "
                            "evaluated independently by common.apply_species_filter"
                        ),
                        # Descriptive: the counts the filter compared against the
                        # floors. It is what shows a reader that Oligo's smallest
                        # species (macaque, 432 cells) cleared the cell floor rather
                        # than escaping it, and that the peak floor was checked too.
                        cell_counts=species_cells,
                    ),
                    "species_scope_note": scope_note,
                }
                if has_species_filter
                else {}
            ),
            "parameters": {
                "seed": common.seed(config),
                "topk": topk,
                "topk_requested": requested_topk,
                "fdr": alpha,
                "n_motifs": n_motifs,
                "species_order": species,
                "recurrence_min_species": min_species,
                # The rule's fraction and floor are named only where the config
                # declares an eligibility/filter block: a v1 summary has neither
                # key, and inserting them would re-order the published object.
                **(
                    {
                        "recurrence_fraction": float(RECURRENT_SPECIES_FRACTION),
                        "recurrence_floor": int(MIN_RECURRENT_SPECIES),
                    }
                    if has_species_filter
                    else {}
                ),
                "recurrence_rule": (
                    "K = max(4, ceil(2N/3)) with N = species kept by eligibility; a "
                    "module is recurrent iff F_abs > 0 in >= K species AND >= K species "
                    "carry a monomeric member at FDR < alpha with log2-OR > 0. F_abs is "
                    "the ABSOLUTE median log2-OR over monomeric members; centered scores "
                    "are for the heatmap only. Only modules whose level gates "
                    "(common.module_is_gating) can be counted."
                ),
                "null_repeats_expected": n_repeats_expected,
                "null_min_repeats_required": min_repeats,
                "null_n_peaks": n_peaks_expected,
                "similarity": "spearman on the raw per-species log2_OR vector",
                "module_aggregate": "median log2_OR over monomeric members (no centering)",
                "module_aggregate_centered": (
                    "F_centered = F_abs - mean_s F_abs, heatmap only (module_scores_centered.tsv)"
                ),
            },
            "matrix_orientation": {
                "file": f"{STEM}_motif_matrix.tsv",
                "layout": (
                    "species rows x motif columns (one leading `species` column), i.e. the "
                    f"readme section 28 {len(species)} x n_motif orientation"
                ),
                "twin": (
                    "s06_purkinje_model.py ships purkinje_motif_matrix.tsv as the transpose "
                    "(motif rows x species columns); the two files are transposes of one "
                    "another by design, not different data"
                ),
            },
            "observed": observed,
            "null": {
                # Descriptive only: the gate's own numbers live under gates.O1,
                # and they are NaN whenever `powered` is false.
                "n_repeats_used": int(null_values.size),
                "min_repeats_required": min_repeats,
                "powered": powered,
                "mean": null_mean,
                "sd": null_sd,
                "q95": q95_descriptive,
                "empirical_p": empirical_p,
                "source": str(null_path) if Path(null_path).exists() else "unavailable",
                "note": o1_note,
            },
            "recurrent_modules": column_list(recurrent, "module"),
            "n_recurrent_modules": int(len(recurrent)),
            # ---- the module layer, reported as INTERPRETATION --------------
            "module_layer": {
                "role": "interpretation_only",
                "primary_evidence": False,
                "rule": (
                    "V2_PLAN section 1: the module layer NAMES what the unbiased "
                    "879-motif analysis and the matched-random null already found. "
                    "It never decides a primary PASS/FAIL: gate O1 and every primary "
                    "table are computed before the module map is read, and are "
                    "byte-identical when the map is emptied."
                ),
                "source": str(common.module_map_path(config)),
                "n_modules_in_panel": int(0 if recurrence.empty else len(recurrence)),
                "n_gating_modules": int(
                    0 if recurrence.empty else int(recurrence.gating.sum())
                ),
                "modules": column_list(recurrence, "module"),
                "levels": (
                    sorted(set(recurrence.level.astype(str)))
                    if not recurrence.empty else []
                ),
                "non_gating_levels": sorted(
                    set(recurrence.loc[~recurrence.gating, "level"].astype(str))
                ) if not recurrence.empty else [],
                "recurrent_primary_modules": column_list(recurrent, "module"),
                "non_gating_modules_that_met_the_rule": non_gating_recurrent,
                "non_gating_rule": (
                    "a module at level secondary or comparator (common.module_is_gating "
                    "== False) is reported but can never gate and is NOT counted as "
                    "evidence of conservation"
                ),
                "comparator_modules": (
                    sorted(set(recurrence.loc[recurrence.level.astype(str) == "comparator", "module"]))
                    if not recurrence.empty else []
                ),
                "comparator_note": (
                    "the comparator module(s) above are present in the outputs and are "
                    "explicitly labelled `comparator` in module_recurrence_all.tsv "
                    "(column claim_wording); they are NOT evidence of conservation and "
                    "are excluded from gate O2"
                ),
                "secondary_modules": (
                    sorted(set(recurrence.loc[recurrence.level.astype(str) == "secondary", "module"]))
                    if not recurrence.empty else []
                ),
                "recurrence_uses": "absolute F_abs (module_scores.tsv / module_recurrence_all.tsv)",
                "heatmap_uses": (
                    "centered F_abs - mean_s F_abs only (module_scores_centered.tsv, "
                    "module_score_heatmap.png/.pdf)"
                ),
                "bhlh_wording_rule": BHLH_EBOX_CLAIM,
                "bhlh_per_motif_note": (
                    "the module median cannot speak to an individual factor, so every "
                    "member of the E-box module -- OLIG1 / OLIG2 / OLIG3 included -- is "
                    "additionally reported per motif in module_member_motifs_per_species.tsv"
                ),
                "olig_factors_reported_per_motif": olig_factors,
                "olig_factor_modules": olig_factor_modules,
                "olig_factors_table": "module_member_motifs_per_species.tsv",
                "tables": {
                    "scores": "module_scores.tsv",
                    "member_scores": "module_member_scores.tsv",
                    "recurrence_all": "module_recurrence_all.tsv",
                    "recurrent": "recurrent_modules.tsv",
                    "centered": "module_scores_centered.tsv",
                    "member_motifs": "module_member_motifs_per_species.tsv",
                    "figure_caption": "module_score_heatmap caption (see module_caption)",
                },
                "module_caption": module_caption,
                "empty_map_behaviour": (
                    "with a map that declares no Oligo module every module table is "
                    "empty and gate O2 is PENDING (never PASS); no primary number, no "
                    "similarity, no margin and no null repeat changes"
                ),
                "empty_map_can_only_weaken": True,
                # The evidence, not the claim: re-run this step with
                # `--module-map <a map with no Oligo row>` and these six hashes are
                # unchanged, while only the module tables and O2's own verdict move.
                "primary_artefact_sha256": primary_sha256,
                "primary_artefacts_listed": sorted(primary_artefacts),
            },
            "gates": {
                "O1": {
                    "status": o1_status,
                    "passes": passes,
                    "criterion": (
                        f"observed mean pairwise similarity > 95th percentile of the "
                        f"{null_name}"
                    ),
                    "note": o1_note,
                    # Scope bookkeeping, emitted only for a filtered config so
                    # the published v1 O1 object keeps its exact key set.
                    **(
                        {
                            "n_species_in_scope": len(species),
                            "species_scope_note": scope_note,
                        }
                        if has_species_filter
                        else {}
                    ),
                    "evidence": {
                        "observed": observed["mean_pairwise_similarity"],
                        "null_q95": q95,
                        "null_mean": null_mean if powered else np.nan,
                        "empirical_p": empirical_p if powered else np.nan,
                        "n_repeats": int(null_values.size),
                        "min_repeats_required": min_repeats,
                    },
                },
                "O2": {
                    "status": o2_status,
                    "passes": o2_passes,
                    "primary_evidence": False,
                    "layer": "module_interpretation",
                    "criterion": (
                        f"at least one recurrent PRIMARY module (>= {min_species} of "
                        f"{len(species)} species); a module at level secondary or "
                        "comparator can never satisfy this gate"
                    ),
                    "note": o2_note,
                    # The rule in force: max(floor, ceil(fraction x len(species))). For
                    # a five- or six-species run this is 4, so a dropped species cannot
                    # relax it. These keys describe the rule as applied to a FILTERED
                    # scope, so they are emitted only where the config declares a
                    # filter; a v1 O2 object keeps its published key set.
                    **(
                        {
                            "min_species_required": int(min_species),
                            "n_species_in_scope": len(species),
                            "n_species_before_filter": int(n_species_all),
                            "recurrence_fraction": float(RECURRENT_SPECIES_FRACTION),
                            "recurrence_floor": int(MIN_RECURRENT_SPECIES),
                            "recurrence_rule": (
                                f"F_abs > 0 in >= {min_species} of {len(species)} species AND a "
                                f"monomeric member at FDR < {alpha} with log2-OR > 0 in >= "
                                f"{min_species} of {len(species)} species; only modules whose "
                                "level gates may be counted"
                            ),
                            "species_scope_note": scope_note,
                        }
                        if has_species_filter
                        else {}
                    ),
                    "evidence": {
                        "n_recurrent_modules": int(len(recurrent)),
                        "modules": column_list(recurrent, "module"),
                        "n_modules_in_table": int(0 if recurrence.empty else len(recurrence)),
                        "n_gating_modules": int(0 if recurrence.empty else int(recurrence.gating.sum())),
                        "non_gating_modules_that_met_the_rule": non_gating_recurrent,
                        "module_map": str(common.module_map_path(config)),
                        "empty_module_map_verdict": (
                            "PENDING -- an emptied module map can never make O2 PASS"
                        ),
                    },
                },
                "O3": {
                    "status": "PENDING",
                    "passes": None,
                    "criterion": "model recurrent program supported in the same direction by raw oligo chromVAR",
                    "evidence": {"owner": "s09_oligo_raw_validation.py"},
                },
            },
            "sanity_check": {
                "evidence_role": "descriptive_only_not_used_for_any_choice",
                "requested_factors": list(SANITY_FACTORS),
                "factors_found": len(SANITY_FACTORS) - len(missing_factors),
                "factors_absent_from_catalog": missing_factors,
                "table": "sanity_check_expected_factors.tsv",
            },
            "audit": audit.summary(),
        }
        common.save_json(summary, out_dir / f"{STEM}_summary.json")

        common.log(
            "gates", f"O1={o1_status}",
            f"O2={o2_status} ({len(recurrent)} recurrent primary module(s) of "
            f"{0 if recurrence.empty else len(recurrence)} in the panel)",
            "O3=PENDING",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
