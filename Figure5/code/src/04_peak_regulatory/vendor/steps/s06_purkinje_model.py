"""Analysis B (Purkinje) model-only step: cross-species convergence and module recurrence.

Why this script exists: readme sections 21 to 25 ask one question about the
*model*: do the Top-k Purkinje peaks of the species in scope describe one shared
regulatory program, and is that agreement larger than what GC/length matched random HVG
peaks produce? The step consumes only the frozen outputs of s01
(`all_model_motif_scores.tsv`, `random_null_scores.tsv.gz`) plus the frozen,
cell-type-specific module map `config/celltype_motif_module_map.tsv`; it never re-ranks
peaks, never re-scans motifs and never touches chromVAR, so it is cheap and idempotent.

The species set is never a literal. `common.apply_species_filter` removes every model
whose species carries fewer than `eligibility.min_target_cells` cells of the analysis
celltype or fewer than `eligibility.min_ranked_peaks` ranked peaks; the step then builds
its species list from the KEPT models only and names every removal in
`purkinje_audit.tsv` and in `purkinje_summary.json` (`species_excluded`). A config
without that block (v1) is untouched: v1 keeps all six Purkinje models, v2 drops bonobo
Purkinje (4 cells AND 298 ranked peaks) and runs over five. Everything downstream -- the
observed matrix, the convergence statistic, the matched-random null, the recurrence
rule and the summary text -- is computed over `len(species)` of that kept set.

Gate P1 = observed mean pairwise Spearman similarity above the 95th percentile of the
matched-random null, evaluated over the kept species. It is PRIMARY evidence: it is
built from the unbiased 879-motif profile and the matched null alone and is bit-for-bit
unaffected by the module layer.

Gate P2 is the module layer's own read-out: at least one recurrent module of the
Purkinje panel, on the rule of V2_PLAN section 13 -- module score > 0 AND a monomeric
member motif with FDR < `model_analysis.fdr` and log2_OR > 0, each in at least
`K = max(4, ceil(2N/3))` species, N being the number of species this run kept (K = 4 for
the five-species v2 scope). Only modules whose level may gate count towards it:
`modules.non_gating_levels` (secondary, comparator) modules are reported in full but can
never satisfy it. Because it is decided by the frozen module map and by nothing else, it
is published under `summary["interpretation"]`, NOT as a member of the primary
`summary["gates"]` panel. V2_PLAN section 1 forbids the module layer from deciding a
primary PASS/FAIL; a gate whose status moved when a hand-edited interpretation file
changed would break that boundary at the step level even though every statistic stayed
put, so `gates` holds only the gates whose `layer` is primary (P1 here, P3 upstream).
The payload keeps the id `P2` and its `layer` field, so a reader that walks the document
(s11's `interpretation_gates`) still resolves it as this claim's interpretation gate.

Gate P3 needs raw chromVAR and belongs to
s07. Readme section 27 (phylogenetic distance) is secondary: it runs only with
`--phylogeny` and never feeds any gate.

The `--scores`, `--null-scores`, `--out-dir` and `--repeats` switches exist so that
`script/tests/` can exercise this file on a synthetic fixture without touching
`results/`; their defaults are the frozen production paths.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import linregress  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

ANALYSIS = "Purkinje"
# V2_PLAN section 13 froze the recurrence rule as K = max(4, ceil(2N/3)) with N the
# number of species this run kept. `2 / 3` and the old `4 / 6` are the same number, so
# `script/s10_sensitivity.py`, which carries the identical pair
# (RECURRENT_SPECIES_FRACTION / RECURRENCE_MIN_SPECIES), cannot drift from this step.
# The denominator is `len(species)` of THIS run -- the species that survived the
# eligibility filter -- never a literal count; the floor guarantees that losing a
# species can never lower the bar below the frozen 4.
RECURRENT_SPECIES_FRACTION = 2 / 3
MIN_RECURRENT_SPECIES = 4

# Names that spell a derived species count out in words. `frozen_six` is part of the
# published v1 artefact contract (`convergence_null_test.tsv`, `purkinje_summary.json`),
# so the label is rendered from the count instead of the count being written into the
# label: a six-species run still says `frozen_six`, a five-species run says
# `frozen_five`.
_NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
}

# Chart roles: a single-hue sequential ramp for magnitude, one accent for the null
# threshold, ink for text. Nothing else is colour-coded in this step.
SERIES_BLUE = "#2a78d6"
ACCENT_ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dcdcd8"


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------


@dataclass
class Audit:
    """Every skip, drop or substitution of this step lands here as one row."""

    rows: list[dict] = field(default_factory=list)

    def add(self, section: str, subject: str, check: str, status: str, detail: object = "") -> None:
        self.rows.append(
            dict(section=section, subject=subject, check=check, status=status, detail=detail)
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["section", "subject", "check", "status", "detail"])


# --------------------------------------------------------------------------
# species scope: derived from the config, never written down
# --------------------------------------------------------------------------


def number_word(value: int) -> str:
    """Render a species count as a word for artefact labels (`frozen_six`)."""
    value = int(value)
    return _NUMBER_WORDS.get(value, str(value))


def scope_label(n_species: int) -> str:
    """Name of the scope that holds every species of the run."""
    return f"frozen_{number_word(n_species)}"


def recurrence_threshold(n_species: int) -> int:
    """`K = max(4, ceil(2N/3))` (V2_PLAN section 13) against the species in scope.

    The ceiling is taken in integer arithmetic -- `(2N + 2) // 3` is exactly
    ceil(2N/3) -- so the bar cannot move on a floating-point tie; the fraction form
    `RECURRENT_SPECIES_FRACTION` is kept only so the value `s10_sensitivity` carries
    can be compared with it. Six species give `max(4, 4)`; five species give
    `max(4, ceil(10/3)) = 4`, so a dropped species cannot relax the rule -- it can
    only make it unreachable, and that is reported rather than absorbed.
    """
    ceiling = -(-2 * int(n_species) // 3)
    return max(MIN_RECURRENT_SPECIES, ceiling)


def scoped_species(manifest: pd.DataFrame) -> list[str]:
    """Species that still own a Purkinje model after the species filter.

    The set comes from the (filtered) manifest, the order from the project's frozen
    species axis, so the same code yields the manifest's species for any config.
    """
    present = set(manifest.loc[manifest.analysis == ANALYSIS, "species"].astype(str))
    return [name for name in C.SPECIES_ORDER if name in present]


def species_models(manifest: pd.DataFrame) -> dict[str, str]:
    sub = manifest[manifest.analysis == ANALYSIS]
    return {str(row.species): str(row.model) for row in sub.itertuples()}


def filter_exclusions(filter_audit: pd.DataFrame) -> pd.DataFrame:
    """The species_filter drops that belong to this analysis, as an audit frame."""
    if filter_audit.empty:
        return filter_audit
    return filter_audit[filter_audit.analysis == ANALYSIS].reset_index(drop=True)


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def read_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run s01_model_motif_enrichment.py first")
    frame = pd.read_csv(path, sep="\t")
    required = {"analysis", "species", "motif_id", "log2_OR", "FDR"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} lacks columns {missing}")
    sub = frame[frame.analysis == ANALYSIS].copy()
    if sub.empty:
        raise ValueError(f"{path} has no rows with analysis=={ANALYSIS}")
    sub["species"] = sub.species.astype(str)
    sub["motif_id"] = sub.motif_id.astype(str)
    return sub


@dataclass(frozen=True)
class SpeciesInventory:
    """What one species contributed to the model matrix, before any filtering."""

    species: str
    model: str
    n_ranked: int
    n_top: int
    n_fg: int
    n_motifs_present: int
    n_motifs_missing: int
    incomplete_topk: bool


def species_inventory(
    scores: pd.DataFrame,
    manifest: pd.DataFrame,
    species_list: list[str],
    catalog: set[str],
    topk: int,
) -> list[SpeciesInventory]:
    """One row per species that kept a model AND carries at least one scored motif.

    `species_list` is the filtered scope; a species that kept a model but has no
    rows in the frozen scores is not silently skipped here, it is named by the
    caller in the audit.
    """
    ranked = manifest[manifest.analysis == ANALYSIS].set_index("species")
    out: list[SpeciesInventory] = []
    for species in species_list:
        sub = scores[scores.species == species]
        if sub.empty:
            continue
        n_ranked = int(ranked.n_ranked.get(species, 0))
        n_top = int(min(topk, n_ranked)) if n_ranked else int(len(sub))
        out.append(
            SpeciesInventory(
                species=species,
                model=str(ranked.model.get(species, "")),
                n_ranked=n_ranked,
                n_top=n_top,
                n_fg=int(sub.n_fg.max()) if "n_fg" in sub.columns else n_top,
                n_motifs_present=len(set(sub.motif_id)),
                n_motifs_missing=len(catalog - set(sub.motif_id)),
                incomplete_topk=bool(n_ranked and n_ranked < topk),
            )
        )
    return out


def observed_matrix(scores: pd.DataFrame, species: list[str], audit: Audit) -> pd.DataFrame:
    """motif x species matrix of log2_OR in the frozen species order."""
    sub = scores[scores.species.isin(species)]
    n_dup = int(sub.duplicated(["species", "motif_id"]).sum())
    if n_dup:
        audit.add("matrix", "scores", "duplicate_species_motif_rows", "WARN", f"{n_dup} averaged")
    wide = sub.pivot_table(index="motif_id", columns="species", values="log2_OR", aggfunc="mean")
    wide = wide.reindex(columns=species)
    if wide.empty:
        raise ValueError("no motif could be placed in the observed matrix")
    return wide.sort_index()


def usable_motifs(wide: pd.DataFrame, species: list[str]) -> list[str]:
    """Motifs scored for every species of the scope: the only comparable basis."""
    return wide.index[wide[species].notna().all(axis=1)].tolist()


# --------------------------------------------------------------------------
# similarity statistics
# --------------------------------------------------------------------------


def profiles(wide: pd.DataFrame, species: list[str], motifs: list[str]) -> dict[str, np.ndarray]:
    block = wide.loc[motifs, species]
    return {name: block[name].to_numpy(dtype=float) for name in species}


def mean_pairwise(profiles_by_species: dict[str, np.ndarray]) -> tuple[float, np.ndarray]:
    """Mean of the off-diagonal pairwise Spearman similarities (readme section 23)."""
    keys = list(profiles_by_species)
    values = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            rho, _ = C.spearman(profiles_by_species[a], profiles_by_species[b])
            values.append(rho)
    array = np.asarray(values, dtype=float)
    return float(array.mean()), array


def similarity_frames(
    profiles_by_species: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """n x n Spearman matrix (via common.similarity_matrix) and its p-values.

    `n` is `len(profiles_by_species)`: the species kept by the species filter, not a
    frozen six.
    """
    matrix = C.similarity_matrix(profiles_by_species)
    keys = list(profiles_by_species)
    pvalues = pd.DataFrame(np.ones((len(keys), len(keys))), index=keys, columns=keys)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            _, p_value = C.spearman(profiles_by_species[a], profiles_by_species[b])
            pvalues.loc[a, b] = pvalues.loc[b, a] = p_value
    return matrix, pvalues


def similarity_long(matrix: pd.DataFrame, pvalues: pd.DataFrame, n_motifs: int) -> pd.DataFrame:
    rows = []
    keys = list(matrix.index)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            rows.append(
                dict(
                    species_a=a,
                    species_b=b,
                    rho=float(matrix.loc[a, b]),
                    pvalue=float(pvalues.loc[a, b]),
                    n_motifs=n_motifs,
                )
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# matched random-500 null
# --------------------------------------------------------------------------


@dataclass
class NullCoverage:
    """Which (repeat, species) null profiles exist, and what had to be dropped."""

    series: dict[tuple[int, str], pd.Series]
    repeats: list[int]
    available: dict[int, list[str]]
    n_rows_total: int
    has_n_fg: bool
    draw_sizes: dict[str, tuple[int, int]] = field(default_factory=dict)
    n_dropped_unbalanced: int = 0

    @property
    def n_profiles(self) -> int:
        return len(self.series)


def load_null(
    path: Path, models: dict[str, str], n_motifs: int, n_peaks: int, min_draw: int = 1
) -> NullCoverage:
    """Per (repeat, species) log2_OR vectors from the s01 matched-random null.

    A (repeat, species) profile is usable when s01 wrote a complete motif set for
    it and the matched draw behind it is a valid, balanced Fisher table.

    `n_peaks` is the *target* draw size and is deliberately NOT a filter. The
    one-to-one GC/length matcher behind the null needs two disjoint matched picks
    per template peak, so the achieved draw is legitimately smaller than the
    target for every model; s01 records the target in `n_fg` and the achieved
    counts in `n_fg_drawn`/`n_bg_drawn`. Filtering on the target would silently
    delete every species whose ranked universe is shorter than the target --
    bonobo Purkinje has 298 ranked peaks and opossum 477 -- from a readme test
    that is specified over every species of the analysis scope, whatever that
    scope turns out to be under the run's species filter. What actually has to
    hold is that the
    foreground and background of the same Fisher table have equal size (otherwise
    every log2_OR carries a constant shift) and that the draw is not trivially
    small; `min_draw` is the configured `random_control.min_peaks_per_repeat`.
    """
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run s01_model_motif_enrichment.py first")
    header = pd.read_csv(path, sep="\t", compression="infer", nrows=0)
    required = {"model", "repeat", "motif_id", "log2_OR"}
    missing = sorted(required - set(header.columns))
    if missing:
        raise ValueError(f"{path} lacks columns {missing}")
    # s01 writes a null for every non-empty model (~30 of them), so the file is many times
    # larger than the Purkinje models this step reads; read only the columns taken below
    # instead of materialising pvalue/FDR/significant for rows that are then discarded.
    keep = [
        c
        for c in ("model", "repeat", "motif_id", "log2_OR", "n_fg", "n_bg", "n_fg_drawn", "n_bg_drawn")
        if c in header.columns
    ]
    frame = pd.read_csv(path, sep="\t", compression="infer", usecols=keep)

    model_to_species = {model: species for species, model in models.items()}
    sub = frame[frame.model.isin(model_to_species)].copy()
    sub["species"] = sub.model.map(model_to_species)

    # the achieved draw is the quantity the Fisher table was built from
    fg_column = "n_fg_drawn" if "n_fg_drawn" in sub.columns else "n_fg"
    bg_column = "n_bg_drawn" if "n_bg_drawn" in sub.columns else "n_bg"
    has_n_fg = fg_column in sub.columns
    draw_sizes: dict[str, tuple[int, int]] = {}

    series: dict[tuple[int, str], pd.Series] = {}
    dropped_unbalanced = 0
    for (repeat, species), group in sub.groupby(["repeat", "species"], sort=True):
        if len(group) != n_motifs:
            continue
        if has_n_fg:
            achieved_fg = int(group[fg_column].min())
            achieved_bg = int(group[bg_column].min()) if bg_column in group.columns else achieved_fg
            if achieved_fg != achieved_bg:
                dropped_unbalanced += 1
                continue
            if achieved_fg < min_draw:
                continue
            draw_sizes.setdefault(str(species), (achieved_fg, int(group.n_fg.max()) if "n_fg" in group else achieved_fg))
        group = group.drop_duplicates("motif_id").sort_values("motif_id")
        series[(int(repeat), str(species))] = group.set_index("motif_id").log2_OR.astype(float)

    available: dict[int, list[str]] = {}
    for repeat, species in series:
        available.setdefault(repeat, []).append(species)
    for repeat in available:
        available[repeat] = [s for s in C.SPECIES_ORDER if s in set(available[repeat])]
    return NullCoverage(
        series=series,
        repeats=sorted(available),
        available=available,
        n_rows_total=len(sub),
        has_n_fg=has_n_fg,
        draw_sizes=draw_sizes,
        n_dropped_unbalanced=dropped_unbalanced,
    )


@dataclass(frozen=True)
class Scope:
    """A species set on which the observed and null statistics are directly comparable."""

    name: str
    species: tuple[str, ...]
    note: str = ""


def strict_scope(observed_species: list[str]) -> Scope:
    """The whole kept species set: the scope the readme test is specified over.

    The label spells out the derived count (`frozen_six` for a six-species run,
    `frozen_five` for the v2 five-species run) rather than encoding a count in the
    scope name.
    """
    return Scope(scope_label(len(observed_species)), tuple(observed_species))


def fallback_scope(per_repeat_species: dict[int, list[str]], observed_species: list[str]) -> Scope | None:
    """Largest observed subset that actually appears in the null, when the strict scope cannot.

    The pre-registered rule drops a species whose null draw is short from that repeat
    and skips the repeat once fewer species remain than the scope holds. A ranking too
    short to have a usable matched-random pool (bonobo Purkinje, 298 peaks) never appears in the
    null at all, so the same rule is applied to the best-covered subset and the
    substitution is reported in `convergence_null_test.tsv` and in
    `purkinje_summary.json`.
    """
    counts = {
        s: sum(1 for r in per_repeat_species if s in per_repeat_species[r]) for s in observed_species
    }
    if not counts:
        return None
    best = max(counts.values())
    if best < 2:
        return None
    subset = tuple(s for s in C.SPECIES_ORDER if s in observed_species and counts[s] >= best)
    if len(subset) < 2:
        return None
    dropped = [s for s in observed_species if s not in subset]
    return Scope(
        "null_covered_subset",
        subset,
        f"strict {number_word(len(observed_species))}-species rule unusable; "
        f"absent from the null in every repeat: {dropped}",
    )


def scope_repeats(scope: Scope, coverage: NullCoverage) -> list[int]:
    """Repeats in which every species of the scope has a usable null profile."""
    return [r for r in coverage.repeats if set(scope.species) <= set(coverage.available.get(r, []))]


def shared_motifs(
    wide: pd.DataFrame, coverage: NullCoverage, scope: Scope, repeats: list[int]
) -> list[str]:
    motifs = set(usable_motifs(wide, list(scope.species)))
    for repeat in repeats:
        for species in scope.species:
            motifs &= set(coverage.series[(repeat, species)].index)
    return [m for m in wide.index if m in motifs]


def null_convergence(
    coverage: NullCoverage,
    scope: Scope,
    motifs: list[str],
    repeat_limit: int | None = None,
) -> pd.DataFrame:
    """Per repeat mean pairwise similarity of the null, with the skip reason kept."""
    repeats = coverage.repeats if repeat_limit is None else coverage.repeats[:repeat_limit]
    rows = []
    for repeat in repeats:
        missing = [s for s in scope.species if s not in coverage.available.get(repeat, [])]
        if missing:
            rows.append(
                dict(
                    scope=scope.name,
                    repeat=repeat,
                    n_species_used=0,
                    mean_pairwise_similarity=np.nan,
                    used=False,
                    skip_reason=f"null_peak_count_or_coverage:{','.join(missing)}",
                )
            )
            continue
        block = {s: coverage.series[(repeat, s)].reindex(motifs).to_numpy(float) for s in scope.species}
        value, _ = mean_pairwise(block)
        rows.append(
            dict(
                scope=scope.name,
                repeat=repeat,
                n_species_used=len(scope.species),
                mean_pairwise_similarity=value,
                used=True,
                skip_reason="",
            )
        )
    columns = ["scope", "repeat", "n_species_used", "mean_pairwise_similarity", "used", "skip_reason"]
    return pd.DataFrame(rows, columns=columns)


def null_test_row(
    scope: Scope,
    observed: float,
    table: pd.DataFrame,
    n_motifs: int,
    is_primary: bool,
) -> dict:
    used = table[table.used]
    n_repeats = int(len(used))
    if n_repeats:
        values = used.mean_pairwise_similarity.to_numpy(float)
        null_mean = float(values.mean())
        null_sd = float(values.std(ddof=1)) if n_repeats > 1 else float("nan")
        q95 = float(np.percentile(values, 95))
        empirical_p = float((np.sum(values >= observed) + 1) / (n_repeats + 1))
        passes = bool(observed > q95)
    else:
        null_mean = null_sd = q95 = empirical_p = float("nan")
        passes = False
    return dict(
        scope=scope.name,
        is_primary_gate=is_primary,
        species_used="|".join(scope.species),
        n_species_used=len(scope.species),
        n_motifs_used=n_motifs,
        observed=observed,
        null_mean=null_mean,
        null_sd=null_sd,
        null_q95=q95,
        empirical_p=empirical_p,
        n_repeats=n_repeats,
        n_repeats_total=int(len(table)),
        n_repeats_skipped=int(len(table) - n_repeats),
        passes_null=passes,
        note=scope.note,
    )


# --------------------------------------------------------------------------
# module recurrence (V2_PLAN sections 12 and 13)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelModule:
    """One entry of a cell type's module panel: an atomic module or a derived union."""

    module: str
    level: str
    members: tuple[str, ...]
    is_gating: bool
    gating_basis: str
    derived_from: tuple[str, ...] = ()


def module_panel(config: dict, celltype: str) -> list[PanelModule]:
    """Every module of one cell type's panel: the atomic map modules + the aggregates.

    Both halves come from `common` (`modules_for` / `module_members` /
    `module_aggregates`), never from a literal, so pointing the config at another
    cell type -- or at a map with no aggregate -- changes the panel without a code
    change.

    Membership is always the MONOMERIC motif ids of the module: a module median has
    to be built from binding models of one factor, not from a fusion of two. A
    derived union such as `Homeodomain_total` contributes no rows of its own; it
    reuses the members of its constituents, so it can be reported next to its
    subdivisions without any motif being counted twice.

    A map with no rows for this cell type is NOT an error: the layer is optional and
    disabling it must leave the primary evidence untouched, so the panel simply comes
    back empty (and the caller reports that). A declared union whose constituents
    resolve to no members at all is dropped for the same reason -- it would otherwise
    be an aggregate of modules the map no longer carries.
    """
    table = C.celltype_module_table(config)
    sub = table[table.celltype == celltype]
    level_of = dict(zip(sub.module.astype(str), sub.level.astype(str)))

    panel: list[PanelModule] = []
    for module in C.modules_for(config, celltype):
        module = str(module)
        level = level_of.get(module, "unassigned")
        may_gate = bool(C.module_is_gating(config, level))
        panel.append(
            PanelModule(
                module=module,
                level=level,
                members=tuple(C.module_members(config, celltype, module)),
                is_gating=may_gate,
                gating_basis=f"map level={level}",
            )
        )
    for aggregate, constituents in C.module_aggregates(config, celltype).items():
        members: list[str] = []
        levels: list[str] = []
        for constituent in constituents:
            members.extend(C.module_members(config, celltype, str(constituent)))
            levels.append(level_of.get(str(constituent), "unassigned"))
        if not members:
            continue
        panel.append(
            PanelModule(
                module=str(aggregate),
                level="derived",
                members=tuple(sorted(set(members))),
                # A derived total may gate only when EVERY constituent may, so a
                # secondary module can never be smuggled past the gate inside a union.
                is_gating=bool(all(C.module_is_gating(config, level) for level in levels)),
                gating_basis="union of "
                + ", ".join(
                    f"{name}(level={level},"
                    f"{'gating' if C.module_is_gating(config, level) else 'non-gating'})"
                    for name, level in zip(constituents, levels)
                ),
                derived_from=tuple(str(constituent) for constituent in constituents),
            )
        )
    return panel


def module_recurrence(
    scores: pd.DataFrame,
    config: dict,
    celltype: str,
    species: list[str],
    alpha: float,
) -> tuple[pd.DataFrame, list[PanelModule]]:
    """One row per panel module, with the recurrence rule of V2_PLAN section 13.

    `F_abs` is the median of `log2_OR` over the module's monomeric members, computed
    per species with NO cross-species centering: the question is absolute convergence,
    not a developmental trajectory. A module is recurrent iff

        #{s : F_abs > 0} >= K  AND  #{s : a monomeric member has FDR < alpha and
        log2_OR > 0} >= K,   K = max(4, ceil(2N/3)), N = len(species),

    with `species` the set this run kept after the eligibility filter, so K is derived
    from the scope and is reported back to the summary. Recurrence is a DESCRIPTIVE
    rule, not a filter: every module of the panel gets a row, recurrent or not, and a
    non-gating module may well come out recurrent -- `is_gating` records whether that
    module may then decide Gate P2, which is a separate question.

    Heterodimers never enter the score or the significance criterion; only `monomer`
    rows of the module map do.

    An empty panel returns an empty table rather than raising: the module layer may be
    switched off, and that must never take the primary evidence down with it.
    """
    panel = module_panel(config, celltype)
    columns = [
        "module", "level", "is_derived", "derived_from", "is_gating", "gating_basis",
        "n_members", "n_species_scored", "n_species_score_pos", "n_species_member_sig",
        "module_score_median_across_species", "member_motifs_significant", "is_recurrent",
    ] + [f"score_{name}" for name in species]
    if not panel:
        # Typed empty frame: an all-object one would make `frame[frame.is_recurrent]`
        # select no column at all rather than no row, and the caller's downstream
        # boolean filters would then blow up.
        empty = pd.DataFrame(columns=columns)
        for name in ("is_derived", "is_gating", "is_recurrent"):
            empty[name] = empty[name].astype(bool)
        for name in ("n_members", "n_species_scored", "n_species_score_pos", "n_species_member_sig"):
            empty[name] = empty[name].astype("int64")
        empty["module_score_median_across_species"] = empty["module_score_median_across_species"].astype(float)
        for name in species:
            empty[f"score_{name}"] = empty[f"score_{name}"].astype(float)
        return empty, panel

    sub = scores[scores.species.isin(species)].copy()
    if "significant" not in sub.columns:
        sub = C.add_significance(sub, alpha)

    membership = pd.DataFrame(
        [(item.module, motif) for item in panel for motif in item.members],
        columns=["module", "motif_id"],
    )
    frame = sub.merge(membership, on="motif_id", how="inner")
    if frame.empty:
        raise ValueError(
            "the module panel is non-empty but no scored motif belongs to any of its "
            "modules; the module map and the motif catalog disagree"
        )

    grouped = frame.groupby(["module", "species"], as_index=False)
    scored = grouped.agg(module_score=("log2_OR", "median"), n_members=("log2_OR", "size"))
    scored = scored.merge(
        grouped["significant"].sum().rename(columns={"significant": "n_sig_members"}),
        on=["module", "species"],
        how="left",
    )
    significant = frame[frame.significant]
    sig_map = (
        (significant.motif_id + "|" + significant.species)
        .groupby(significant.module)
        .apply(lambda values: ";".join(sorted(set(values))))
    )
    threshold = recurrence_threshold(len(species))

    rows = []
    for item in panel:
        block = scored[scored.module == item.module]
        by_species = block.set_index("species").module_score
        n_pos = int((block.module_score > 0).sum())
        n_sig = int((block.n_sig_members > 0).sum())
        row = dict(
            module=item.module,
            level=item.level,
            is_derived=bool(item.derived_from),
            derived_from="|".join(item.derived_from),
            is_gating=bool(item.is_gating),
            gating_basis=item.gating_basis,
            n_members=int(len(item.members)),
            n_species_scored=int(len(block)),
            n_species_score_pos=n_pos,
            n_species_member_sig=n_sig,
            module_score_median_across_species=(
                float(block.module_score.median()) if len(block) else float("nan")
            ),
            member_motifs_significant=str(sig_map.get(item.module, "")),
            is_recurrent=bool(n_pos >= threshold and n_sig >= threshold),
        )
        for name in species:
            row[f"score_{name}"] = float(by_species.get(name, np.nan))
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["is_recurrent", "is_gating", "n_species_member_sig", "module"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True), panel


def centered_module_scores(frame: pd.DataFrame, species: list[str]) -> pd.DataFrame:
    """`F_centered = F_abs - mean_s F_abs`, for the HEATMAP ONLY.

    Subtracting the panel mean removes the sign every species shares and is a display
    transform, nothing more: recurrence, the recurrence counts and Gate P2 are all
    decided on the ABSOLUTE `F_abs` carried in the `score_<species>` columns of the
    recurrence table, and this frame is never fed back into any statistic.
    """
    block = frame.set_index("module")[[f"score_{name}" for name in species]].copy()
    block.columns = list(species)
    return block.sub(block.mean(axis=1), axis=0)


# --------------------------------------------------------------------------
# secondary: phylogenetic distance (readme section 27)
# --------------------------------------------------------------------------


def read_phylo_distances(path: Path, species: list[str]) -> pd.DataFrame:
    """Accept either a long pair table or a square species matrix.

    Returns one row per unordered species pair with the two species named in sorted
    order, so a symmetric matrix cannot contribute the same pair twice.
    """
    raw = pd.read_csv(path, sep="\t")
    columns = [str(c) for c in raw.columns]
    if set(species) <= set(columns) and set(species) <= set(raw.iloc[:, 0].astype(str)):
        square = raw.set_index(raw.columns[0])
        long = square.loc[list(species), list(species)].stack().rename("distance").reset_index()
        long.columns = ["species_a", "species_b", "distance"]
    else:
        long = raw.iloc[:, :3].copy()
        long.columns = ["species_a", "species_b", "distance"]
    long["species_a"] = long.species_a.astype(str)
    long["species_b"] = long.species_b.astype(str)
    long["distance"] = pd.to_numeric(long.distance, errors="coerce")
    long = long[long.species_a.isin(species) & long.species_b.isin(species)]
    long = long.dropna(subset=["distance"])
    long = long[long.species_a != long.species_b]
    long["key_a"] = np.minimum(long.species_a, long.species_b)
    long["key_b"] = np.maximum(long.species_a, long.species_b)
    long = long.groupby(["key_a", "key_b"], as_index=False).distance.mean()
    return long.rename(columns={"key_a": "species_a", "key_b": "species_b"})


def phylogeny_secondary(
    matrix: pd.DataFrame, species: list[str], distances: pd.DataFrame, n_motifs: int
) -> pd.DataFrame:
    """Regression of D_reg = 1 - S on the supplied distances. Never a gate."""
    rows = []
    for i, a in enumerate(species):
        for b in species[i + 1:]:
            rows.append(
                dict(
                    species_a=a,
                    species_b=b,
                    key_a=min(a, b),
                    key_b=max(a, b),
                    rho=float(matrix.loc[a, b]),
                )
            )
    frame = pd.DataFrame(rows)
    lookup = distances.assign(
        key_a=np.minimum(distances.species_a, distances.species_b),
        key_b=np.maximum(distances.species_a, distances.species_b),
    )
    frame = frame.merge(lookup[["key_a", "key_b", "distance"]], on=["key_a", "key_b"], how="inner")
    if len(frame) < 3:
        raise ValueError("fewer than three species pairs carry a phylogenetic distance")
    frame["d_reg"] = 1.0 - frame.rho
    fit = linregress(frame.distance.to_numpy(float), frame.d_reg.to_numpy(float))
    frame["fitted"] = fit.intercept + fit.slope * frame.distance
    frame["residual"] = frame.d_reg - frame.fitted
    frame["n_motifs"] = n_motifs
    frame["slope"] = fit.slope
    frame["intercept"] = fit.intercept
    frame["rvalue"] = fit.rvalue
    frame["pvalue"] = fit.pvalue
    frame["stderr"] = fit.stderr
    frame["n_pairs"] = len(frame)
    return frame.drop(columns=["key_a", "key_b"])


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str], dpi: int) -> None:
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        # matplotlib stamps a wall-clock /CreationDate into PDFs unless it is pinned, which
        # makes two runs over identical inputs differ byte for byte; the tables do not.
        metadata = {"CreationDate": None} if fmt == "pdf" else None
        fig.savefig(path, dpi=dpi, bbox_inches="tight", metadata=metadata)
        C.log("wrote", path)


def plot_similarity_heatmap(
    matrix: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
    n_motifs: int,
    mean_pair: float,
    cutoff: str,
) -> None:
    keys = list(matrix.index)
    values = matrix.to_numpy(float).copy()
    np.fill_diagonal(values, np.nan)
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    fig, axis = plt.subplots(figsize=(6.0, 5.2))
    image = axis.imshow(values, cmap="Blues", vmin=low, vmax=high)
    axis.set_xticks(range(len(keys)), keys, rotation=45, ha="right")
    axis.set_yticks(range(len(keys)), keys)
    for i in range(len(keys)):
        for j in range(len(keys)):
            if i == j:
                axis.text(j, i, "·", ha="center", va="center", color=MUTED)
            else:
                axis.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color=INK, fontsize=8)
    axis.set_title(
        f"Purkinje model {cutoff} motif similarity\n"
        f"{n_motifs} motifs, mean off-diagonal rho = {mean_pair:.3f}\n"
        f"descriptive mean over {len(keys)} species: {'|'.join(keys)}",
        fontsize=10,
    )
    bar = fig.colorbar(image, ax=axis, shrink=0.85)
    bar.set_label("Spearman rho of log2_OR")
    fig.tight_layout()
    save_figure(fig, out_dir, "similarity_heatmap", formats, dpi)
    plt.close(fig)


def plot_convergence_null(
    table: pd.DataFrame, test_row: dict, out_dir: Path, formats: list[str], dpi: int, cutoff: str
) -> None:
    used = table[table.used].mean_pairwise_similarity.to_numpy(float)
    observed = float(test_row["observed"])
    q95 = float(test_row["null_q95"])
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    if len(used):
        axis.hist(used, bins=25, color=SERIES_BLUE, edgecolor="white", linewidth=0.5)
    axis.axvline(observed, color=INK, linestyle="--", linewidth=1.6, label=f"observed {cutoff}")
    if np.isfinite(q95):
        axis.axvline(q95, color=ACCENT_ORANGE, linestyle=":", linewidth=1.6, label="null 95th pct")
    p_value = test_row["empirical_p"]
    caption = (
        "empirical p = n/a (no usable repeat)"
        if not np.isfinite(p_value)
        else f"empirical p = {p_value:.3g}"
    )
    axis.set_xlabel("mean pairwise Spearman rho (null: matched-random)")
    axis.set_ylabel(f"repeats (n = {len(used)})")
    axis.set_title(f"Purkinje convergence vs matched-random null\n{caption}", fontsize=10)
    axis.legend(frameon=False, fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, out_dir, "convergence_null", formats, dpi)
    plt.close(fig)


def plot_module_heatmap(
    centered: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
    n_species: int,
    threshold: int,
    recurrent_modules: list[str],
) -> None:
    """Module x species heatmap of the CENTERED module scores.

    The heatmap is the only consumer of `F_centered`; the annotation line says so,
    because a centered cell can be negative for a module whose absolute `F_abs` is
    positive in every species, and a reader must not take the colour as the
    recurrence verdict. Rows are ordered by the recurrence table, not by value.
    """
    modules = list(centered.index)
    species = list(centered.columns)
    values = centered.to_numpy(float)
    limit = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    limit = limit if limit > 0 else 1.0

    height = max(2.6, 0.42 * len(modules) + 1.9)
    fig, axis = plt.subplots(figsize=(max(6.0, 1.05 * len(species) + 3.4), height))
    image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(species)), species, rotation=45, ha="right")
    axis.set_yticks(range(len(modules)), modules)
    for i in range(len(modules)):
        for j in range(len(species)):
            if np.isfinite(values[i, j]):
                axis.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8, color=INK)
    axis.set_title(
        "Purkinje module scores, centered for display only\n"
        f"F_centered = F_abs - mean over the {n_species} species in scope; "
        f"recurrence uses the absolute F_abs (K = {threshold} of {n_species})\n"
        f"recurrent: {'|'.join(recurrent_modules) if recurrent_modules else 'none'}",
        fontsize=10,
    )
    bar = fig.colorbar(image, ax=axis, shrink=0.85)
    bar.set_label("F_abs centered on the per-species mean")
    fig.tight_layout()
    save_figure(fig, out_dir, "module_heatmap", formats, dpi)
    plt.close(fig)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def _maybe_float(value: object) -> float | None:
    """JSON has no NaN, so undefined statistics become null instead of a fake number."""
    number = float(value)  # type: ignore[arg-type]
    return None if not np.isfinite(number) else number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analysis B Purkinje model-only step")
    C.add_common_args(parser)
    parser.add_argument("--scores", default=None, help="override all_model_motif_scores.tsv")
    parser.add_argument(
        "--null-scores", dest="null_scores", default=None, help="override random_null_scores.tsv.gz"
    )
    parser.add_argument("--out-dir", dest="out_dir", default=None, help="override results/06_purkinje_model")
    parser.add_argument("--phylogeny", action="store_true", help="also fit the secondary phylo-distance regression")
    parser.add_argument("--phylo-distances", dest="phylo_distances", default=None)
    args = parser.parse_args(argv)

    C.main_guard()
    config = C.load_config(args.config)
    topk = int(args.topk if args.topk else config["model_analysis"]["primary_topk"])
    # The arm's own cut-off, spelled for the figure titles below: this step runs once
    # per arm, so a hardcoded "Top500" would caption the Top1000 and Top2000 arms wrong.
    cutoff = C.topk_label(config)
    alpha = float(config["model_analysis"]["fdr"])
    n_peaks = int(config["random_control"]["n_peaks"])
    n_motifs = int(config["motif_database"]["n_motifs"])
    formats = list(config["plotting"]["formats"])
    dpi = int(config["plotting"]["dpi"])

    out_dir = (
        Path(args.out_dir) if args.out_dir else C.results_dir(config, "06_purkinje_model")
    ).resolve()
    # Every table goes through common.save_table, which resolves its path against the project
    # root; an out-of-tree or unresolved --out-dir would otherwise fail only at the first
    # write, i.e. after the matrix, the null test and the gate had already run and after a
    # half-populated output directory had been created.
    if out_dir != C.ROOT and C.ROOT not in out_dir.parents:
        parser.error(f"--out-dir must stay inside the project root {C.ROOT}; got {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    motif_dir = C.results_dir(config, "01_model_motif")
    scores_path = Path(args.scores) if args.scores else motif_dir / "all_model_motif_scores.tsv"
    null_path = Path(args.null_scores) if args.null_scores else motif_dir / "random_null_scores.tsv.gz"

    audit = Audit()
    with C.StepTimer("s06 purkinje model"):
        catalog = set(C.motif_catalog(config).motif_id)
        unfiltered_manifest, _ = C.load_rankings(config)
        # The species set is decided here and nowhere else: `apply_species_filter` is a
        # no-op for a config without `species_filter` (v1 keeps its six models), and it
        # drops bonobo Purkinje (4 cells) under v2. Everything below reads the KEPT
        # manifest, so no downstream statistic can accidentally reintroduce a dropped
        # species through the frozen s01 score table.
        manifest, filter_audit = C.apply_species_filter(config, unfiltered_manifest)
        scores = read_scores(scores_path)
        # --topk only relabels the report: the profiles always come from the frozen
        # all_model_motif_scores.tsv, which carries the Top-k s01 actually scanned. Refuse a
        # mismatch rather than publishing a Top-k the data does not have.
        if args.topk and "topk" in scores.columns:
            frozen_topk = pd.to_numeric(scores.topk, errors="coerce").max()
            if pd.notna(frozen_topk) and int(args.topk) != int(frozen_topk):
                parser.error(
                    f"--topk {int(args.topk)} does not match the frozen {scores_path.name} "
                    f"(topk={int(frozen_topk)}); s06 never re-ranks peaks, so overriding it "
                    "would only mislabel the summary"
                )

        # --- species scope: from the filtered manifest, never a literal ------
        # `apply_species_filter` reads `eligibility` when the config has it (v2) and
        # falls back to `species_filter` (v1); the block actually in force is named in
        # the error, the audit and the summary so the two can never be confused.
        filter_block = "eligibility" if config.get("eligibility") else "species_filter"
        kept_species = scoped_species(manifest)
        n_species_all = len(scoped_species(unfiltered_manifest))
        if not kept_species:
            raise ValueError(
                f"the {filter_block} filter left no {ANALYSIS} model to analyse; check "
                f"{filter_block}.min_target_cells / min_ranked_peaks against the model manifest"
            )
        exclusions = filter_exclusions(filter_audit)
        for row in exclusions.itertuples():
            audit.add(
                filter_block,
                row.species,
                "cells_and_ranked_peaks",
                "WARN",
                f"{row.action}; {row.n_ranked} ranked peaks (floor {row.min_ranked_peaks}); model "
                f"{row.model} is excluded from the {ANALYSIS} analysis",
            )
        species_excluded = [
            dict(
                species=str(row.species),
                celltype=str(row.celltype),
                model=str(row.model),
                n_cells=int(row.n_cells),
                min_cells=int(row.min_cells),
                n_ranked=int(row.n_ranked),
                min_ranked_peaks=int(row.min_ranked_peaks),
                reason=str(row.action),
            )
            for row in exclusions.itertuples()
        ]

        models = species_models(manifest)
        inventory = species_inventory(scores, manifest, kept_species, catalog, topk)
        if not inventory:
            raise ValueError(
                f"no {ANALYSIS} species found in the model scores (kept species: {kept_species})"
            )
        # A kept model with no scored motif is a real loss, not a silent skip.
        for species in kept_species:
            if species not in {item.species for item in inventory}:
                audit.add(
                    "inventory",
                    species,
                    "scores_present",
                    "FAIL",
                    f"{ANALYSIS} model {models.get(species, '')} kept by the species filter but "
                    f"absent from {scores_path.name}; the species cannot enter any statistic",
                )
        # A scored species outside the scope must be named too, with the reason that
        # removed it (the species filter, or no ranked model in this run at all).
        filter_dropped = {str(row.species) for row in exclusions.itertuples()}
        for species in sorted(set(scores.species) - set(kept_species) - filter_dropped):
            audit.add(
                "inventory",
                species,
                "model_present",
                "WARN",
                f"{ANALYSIS} rows exist in {scores_path.name} but the filtered model manifest "
                f"holds no {ANALYSIS} model for this species; it is outside the scope of this run",
            )

        for item in inventory:
            audit.add(
                "inventory",
                item.species,
                "n_ranked/n_top/motifs",
                "WARN" if item.incomplete_topk else "PASS",
                f"n_ranked={item.n_ranked} n_top={item.n_top} n_fg={item.n_fg} "
                f"motifs={item.n_motifs_present} missing={item.n_motifs_missing}",
            )
        incomplete = [item.species for item in inventory if item.incomplete_topk]
        observed_species = [item.species for item in inventory]
        if len(observed_species) < 2:
            raise ValueError(
                f"{len(observed_species)} species in scope; a pairwise convergence statistic "
                f"needs at least two (kept: {kept_species})"
            )
        C.log("species", observed_species, "incomplete_top500", incomplete)

        # --- observed matrix and n x n similarity -------------------------
        wide = observed_matrix(scores, observed_species, audit)
        without_motifs = [s for s in observed_species if not wide[s].notna().any()]
        observed_species = [s for s in observed_species if wide[s].notna().any()]
        if without_motifs:
            audit.add("matrix", "observed", "species_without_any_motif", "FAIL", str(without_motifs))
        if len(observed_species) < 2:
            raise ValueError(
                f"only {len(observed_species)} species carry a scored motif; no pairwise "
                "convergence statistic exists"
            )
        obs_motifs = usable_motifs(wide, observed_species)
        if len(obs_motifs) < n_motifs:
            audit.add(
                "similarity",
                "observed",
                "motif_coverage",
                "WARN",
                f"{len(obs_motifs)}/{n_motifs} motifs scored in all of {observed_species}",
            )
        obs_profiles = profiles(wide, observed_species, obs_motifs)
        matrix, pvalues = similarity_frames(obs_profiles)
        observed_value, _ = mean_pairwise(obs_profiles)
        # Descriptive mean over every observed species; the gated value (Gate P1) may use the
        # narrower `null_covered_subset`, so name the species set everywhere this number is
        # reported and never let the two pass as one another.
        C.log(
            "observed mean pairwise similarity",
            round(observed_value, 4),
            f"over {len(observed_species)} species:",
            "|".join(observed_species),
        )

        # --- matched random-500 null --------------------------------------
        min_draw = int(config.get("random_control", {}).get("min_peaks_per_repeat", 1))
        coverage = load_null(null_path, models, n_motifs, n_peaks, min_draw=min_draw)
        for species, (achieved, target) in sorted(coverage.draw_sizes.items()):
            audit.add(
                "null_draw_size",
                species,
                achieved,
                # The audit has one status vocabulary: PASS/WARN/FAIL/INFO. Lowercase
                # verdicts made `set(audit.status)` two-valued case-wise and broke the
                # step's own smoke test; the label is the only thing normalised here.
                "INFO" if achieved == target else "WARN",
                f"achieved matched draw {achieved} peaks per arm against a target of {target}; "
                f"v3 draws every repeat at the observed table's own width, so the exact "
                f"producer reaches that target on every emitted repeat (min_draw={min_draw})",
            )
        if coverage.n_dropped_unbalanced:
            audit.add(
                "null_draw_size",
                "unbalanced Fisher tables",
                coverage.n_dropped_unbalanced,
                "WARN",
                "profiles whose matched foreground and background differ in size were dropped",
            )
        if not coverage.has_n_fg:
            audit.add(
                "null_coverage",
                "random_null_scores.tsv.gz",
                "n_fg_column_absent",
                "WARN",
                "the configured peak count cannot be checked per row; a (model, repeat) that "
                f"is present is assumed to be complete because s01 drops short draws (v3 draws "
                f"each repeat at exactly its own observed matched-foreground size, nominally "
                f"{n_peaks} peaks)",
            )
        for item in inventory:
            n_usable = sum(
                1 for r in coverage.repeats if item.species in coverage.available.get(r, [])
            )
            if coverage.has_n_fg:
                status = "PASS" if n_usable else "FAIL"
                detail = (
                    f"{n_usable}/{len(coverage.repeats)} repeats carry a complete exact-sized "
                    f"null, each drawn at its own observed matched-foreground width"
                )
            else:
                # s01 records no per-draw peak count, so the configured peak count cannot be
                # asserted here; the row must not claim a property the step never checked.
                status = "WARN" if n_usable else "FAIL"
                detail = (
                    f"{n_usable}/{len(coverage.repeats)} repeats carry a complete motif set; "
                    f"the per-draw peak count is not recorded by s01 (n_fg absent), so the "
                    f"configured {n_peaks}-peak requirement is unchecked"
                )
            audit.add("null_coverage", item.species, "usable_null_repeats", status, detail)

        strict = strict_scope(observed_species)
        gate_scope = strict
        if not scope_repeats(strict, coverage):
            fallback = fallback_scope(coverage.available, observed_species)
            if fallback is None:
                raise RuntimeError("no species set has a usable matched-random null")
            audit.add(
                "null_coverage",
                "scope",
                f"strict_{number_word(len(observed_species))}_species_rule",
                "WARN",
                fallback.note,
            )
            gate_scope = fallback
        scopes = [strict] if strict.name == gate_scope.name else [strict, gate_scope]

        null_tables: dict[str, pd.DataFrame] = {}
        test_rows: list[dict] = []
        for scope in scopes:
            repeats = scope_repeats(scope, coverage)
            motifs = shared_motifs(wide, coverage, scope, repeats)
            table = null_convergence(coverage, scope, motifs, repeat_limit=args.repeats)
            null_tables[scope.name] = table
            value, _ = mean_pairwise(profiles(wide, list(scope.species), motifs))
            row = null_test_row(scope, value, table, len(motifs), scope.name == gate_scope.name)
            test_rows.append(row)
            audit.add(
                "null_coverage",
                scope.name,
                "repeats_used",
                "PASS" if row["n_repeats"] else "FAIL",
                f"{row['n_repeats']}/{row['n_repeats_total']} repeats usable "
                f"for species {'|'.join(scope.species)}",
            )

        gate_row = next(row for row in test_rows if row["is_primary_gate"])
        C.log("gate scope", gate_row["scope"], "n_repeats", gate_row["n_repeats"], "passes", gate_row["passes_null"])

        # --- module recurrence (V2_PLAN section 13) -----------------------
        # The panel, the membership and the derived unions all come from the frozen
        # cell-type map through `common`, so this block never names a module itself.
        module_frame, panel = module_recurrence(scores, config, ANALYSIS, observed_species, alpha)
        recurrent = module_frame[module_frame.is_recurrent].copy()
        # Gate P2 is decided on the modules that may gate; a recurrent secondary or
        # comparator module is still reported, it just cannot carry the gate.
        gate_recurrent = recurrent[recurrent.is_gating]
        gating_levels = sorted({item.level for item in panel if item.is_gating})
        non_gating_levels = sorted({item.level for item in panel if not item.is_gating})
        if not panel:
            # The layer is optional by construction: switching it off leaves every
            # primary artefact identical and only makes the interpretation gate
            # unevaluable. Say so instead of letting an empty panel read as a failure.
            audit.add(
                "modules",
                str(C.module_map_path(config)),
                "panel_empty",
                "WARN",
                f"the module map declares no module for {ANALYSIS}; the interpretation "
                "layer is inactive, so Gate P2 is not evaluable and every primary "
                "artefact is computed exactly as with the layer present",
            )
        for item in panel:
            audit.add(
                "modules",
                item.module,
                "panel_member",
                "PASS" if item.is_gating else "WARN",
                f"level={item.level} monomers={len(item.members)} "
                f"gating={item.is_gating} ({item.gating_basis})",
            )
        panel_members = {motif for item in panel for motif in item.members}
        catalog_motifs = set(C.motif_catalog(config).motif_id)
        unassigned = sorted(catalog_motifs - panel_members)
        audit.add(
            "modules",
            "panel_coverage",
            "catalog_motifs_outside_panel",
            "INFO",
            f"{len(unassigned)}/{len(catalog_motifs)} catalog motifs belong to no {ANALYSIS} "
            "panel module; they stay in the unbiased 879-motif primary test and are only "
            "unnamed by the interpretation layer",
        )
        if non_gating_levels and not gating_levels:
            audit.add(
                "modules",
                "gating_levels",
                "no_gating_module_level",
                "FAIL",
                f"every module level ({non_gating_levels}) is declared non-gating, so Gate P2 "
                "cannot be satisfied by any module of this panel",
            )
        elif panel and not non_gating_levels:
            audit.add(
                "modules",
                "non_gating_levels",
                "config_declares_no_non_gating_level",
                "WARN",
                "modules.non_gating_levels is empty, so every level may gate; the intended "
                "reading is that only primary modules may decide Gate P2",
            )
        for item in panel:
            row = module_frame[module_frame.module == item.module].iloc[0]
            if int(row.n_species_scored) < len(observed_species):
                audit.add(
                    "modules",
                    item.module,
                    "species_without_scored_member",
                    "WARN",
                    f"{int(row.n_species_scored)}/{len(observed_species)} species carry a scored "
                    "monomer of this module; the remainder cannot contribute a positive score",
                )
            absent_members = len(item.members) and not bool(row.member_motifs_significant)
            if item.members and absent_members:
                audit.add(
                    "modules",
                    item.module,
                    "no_significant_member",
                    "INFO",
                    f"no monomer of {item.module} reaches FDR < {alpha} with log2_OR > 0 in any "
                    f"of the {len(observed_species)} species in scope",
                )
        if "significant" not in scores.columns:
            audit.add(
                "modules",
                "significance",
                "significant_column_absent",
                "WARN",
                "recomputed with common.add_significance (FDR < alpha and log2_OR > 0)",
            )
        for aggregate, constituents in C.module_aggregates(config, ANALYSIS).items():
            audit.add(
                "modules",
                aggregate,
                "derived_union",
                "INFO",
                f"{aggregate} is the union of {'|'.join(str(c) for c in constituents)} declared "
                "in modules.aggregates; it contributes no rows of its own, so no motif is "
                "counted twice and it is never summed with its subdivisions",
            )
        centered = centered_module_scores(module_frame, observed_species)

        # --- write tables ---------------------------------------------------
        core = [c for c in ("species", "motif_id", "motif_name", "log2_OR", "FDR", "significant",
                            "fg_hits", "bg_hits", "n_fg", "n_bg") if c in scores.columns]
        long_matrix = (
            scores[scores.species.isin(observed_species)][core]
            .sort_values(["species", "motif_id"])
            .reset_index(drop=True)
        )
        C.save_table(long_matrix, out_dir / "purkinje_motif_matrix_long.tsv", sep="\t")
        wide_matrix = wide[observed_species].T.reset_index().rename(columns={"motif_id": "species"})
        C.save_table(wide_matrix, out_dir / "purkinje_motif_matrix.tsv", sep="\t")
        # The matrix is len(observed_species) square; the stem says which, so a
        # five-species v2 run cannot ship a file labelled 6x6.
        similarity_stem = f"purkinje_similarity_{len(observed_species)}x{len(observed_species)}.tsv"
        C.save_table(matrix.rename_axis("species").reset_index(), out_dir / similarity_stem, sep="\t")
        C.save_table(
            pvalues.rename_axis("species").reset_index(), out_dir / "purkinje_similarity_pvalues.tsv", sep="\t"
        )
        C.save_table(
            similarity_long(matrix, pvalues, len(obs_motifs)), out_dir / "purkinje_similarity_long.tsv", sep="\t"
        )

        convergence_rows = []
        for scope in scopes:
            motifs = usable_motifs(wide, list(scope.species))
            value, pair = mean_pairwise(profiles(wide, list(scope.species), motifs))
            convergence_rows.append(
                dict(
                    scope=scope.name,
                    is_primary_gate=scope.name == gate_scope.name,
                    species_used="|".join(scope.species),
                    n_species_used=len(scope.species),
                    n_motifs_used=len(motifs),
                    n_pairs=len(pair),
                    mean_pairwise_similarity=value,
                    sd_pairwise=float(pair.std(ddof=1)) if len(pair) > 1 else float("nan"),
                    min_pairwise=float(pair.min()),
                    max_pairwise=float(pair.max()),
                )
            )
        C.save_table(pd.DataFrame(convergence_rows), out_dir / "convergence_observed.tsv", sep="\t")
        C.save_table(
            pd.concat(list(null_tables.values()), ignore_index=True),
            out_dir / "random_null_convergence.tsv",
            sep="\t",
        )
        test_table = pd.DataFrame(test_rows)
        C.save_table(test_table, out_dir / "convergence_null_test.tsv", sep="\t")
        # The recurrence table is the census of the whole panel, recurrent or not;
        # `recurrent_modules.tsv` is the subset and carries the same columns.
        C.save_table(module_frame, out_dir / "module_recurrence_all.tsv", sep="\t")
        C.save_table(recurrent, out_dir / "recurrent_modules.tsv", sep="\t")
        C.save_table(
            centered.rename_axis("module").reset_index(),
            out_dir / "module_scores_centered.tsv",
            sep="\t",
        )
        C.save_table(
            pd.DataFrame(
                [
                    dict(
                        module=item.module,
                        level=item.level,
                        is_derived=bool(item.derived_from),
                        derived_from="|".join(item.derived_from),
                        is_gating=bool(item.is_gating),
                        gating_basis=item.gating_basis,
                        n_monomers=len(item.members),
                    )
                    for item in panel
                ]
            ),
            out_dir / "purkinje_module_panel.tsv",
            sep="\t",
        )
        C.save_table(audit.frame(), out_dir / "purkinje_audit.tsv", sep="\t")
        C.save_table(
            pd.DataFrame(
                [
                    dict(
                        species=item.species,
                        model=item.model,
                        n_ranked=item.n_ranked,
                        n_top=item.n_top,
                        n_fg=item.n_fg,
                        n_motifs_present=item.n_motifs_present,
                        n_motifs_missing=item.n_motifs_missing,
                        incomplete_top500=item.incomplete_topk,
                    )
                    for item in inventory
                ]
            ),
            out_dir / "purkinje_species_inventory.tsv",
            sep="\t",
        )

        # --- figures --------------------------------------------------------
        plot_similarity_heatmap(matrix, out_dir, formats, dpi, len(obs_motifs), observed_value, cutoff)
        plot_convergence_null(null_tables[gate_scope.name], gate_row, out_dir, formats, dpi, cutoff)
        if panel:
            plot_module_heatmap(
                centered, out_dir, formats, dpi, len(observed_species),
                recurrence_threshold(len(observed_species)), recurrent.module.tolist(),
            )
        else:
            C.log("module layer inactive: no panel module to draw")

        # --- secondary phylogeny (never a gate) -----------------------------
        phylo_payload = None
        if args.phylogeny:
            if not args.phylo_distances:
                parser.error("--phylogeny requires --phylo-distances")
            distances = read_phylo_distances(Path(args.phylo_distances), list(matrix.index))
            phylo = phylogeny_secondary(matrix, list(matrix.index), distances, len(obs_motifs))
            C.save_table(phylo, out_dir / "purkinje_phylogeny_secondary.tsv", sep="\t")
            phylo_payload = dict(
                n_pairs=int(len(phylo)),
                slope=float(phylo.slope.iloc[0]),
                intercept=float(phylo.intercept.iloc[0]),
                rvalue=float(phylo.rvalue.iloc[0]),
                pvalue=float(phylo.pvalue.iloc[0]),
                distances_file=str(args.phylo_distances),
                used_in_gates=False,
            )

        # --- summary --------------------------------------------------------
        # Every count below is derived from the species this run actually kept.
        n_species = len(observed_species)
        recurrence_min = recurrence_threshold(n_species)
        n_pairs_kept = n_species * (n_species - 1) // 2
        n_pairs_all = n_species_all * (n_species_all - 1) // 2
        strict_row = test_table[test_table.scope == strict.name]
        # Dropping a species cannot be made free: the observed mean and the null both
        # lose that species' pairs, so state the loss instead of adjusting the gate.
        scope_note = (
            ""
            if not species_excluded
            else (
                f"{len(species_excluded)} species excluded by {filter_block}: "
                + "; ".join(
                    f"{row['species']} ({row['reason']})" for row in species_excluded
                )
                + f". The convergence statistic is a mean over {n_pairs_kept} species pairs "
                f"instead of {n_pairs_all}, and the matched null is rebuilt on the same reduced "
                "species set; observed and null stay directly comparable, but the test carries "
                "less information than the unfiltered one. The threshold was NOT relaxed to "
                "compensate: P1 still requires the observed mean to exceed the 95th percentile."
            )
        )
        # The derived unions are declared in the config, so the summary names them from
        # there rather than from a literal. This note is what keeps a reader from ever
        # adding the total to its own subdivisions.
        unions = {
            str(name): [str(constituent) for constituent in constituents]
            for name, constituents in C.module_aggregates(config, ANALYSIS).items()
        }
        unions_note = "; ".join(
            f"{name} is the derived union of {'|'.join(members)}" for name, members in unions.items()
        )
        unions_note = (
            unions_note
            + ". It contributes no rows of its own, so no motif is ever counted twice and it is "
            "never added to its subdivisions in any total."
            if unions_note
            else "the config declares no derived module union for this cell type"
        )

        summary = dict(
            step="s06_purkinje_model",
            analysis=ANALYSIS,
            topk_used=topk,
            fdr=alpha,
            species_order=observed_species,
            n_species=len(observed_species),
            n_species_before_filter=n_species_all,
            species_excluded=species_excluded,
            species_filter=dict(
                block=filter_block,
                applied=bool(config.get("eligibility") or config.get("species_filter")),
                min_target_cells=int(C.min_cells_per_species(config)),
                min_ranked_peaks=int(C.min_ranked_peaks(config)),
                celltype_column=(config.get(filter_block) or {}).get("celltype_column"),
            ),
            species_scope_note=scope_note,
            n_motifs_used=len(obs_motifs),
            n_motifs_catalog=n_motifs,
            incomplete_top500=incomplete,
            species_inventory=[
                dict(
                    species=item.species,
                    n_ranked=item.n_ranked,
                    n_top=item.n_top,
                    n_fg=item.n_fg,
                    n_motifs_present=item.n_motifs_present,
                )
                for item in inventory
            ],
            gates=dict(
                P1=dict(
                    statement=f"observed mean pairwise similarity > 95th percentile of the {C.null_label(config)}",
                    layer="primary",
                    # NO VERDICT when the null could not be drawn.
                    #
                    # v3's matched-random null is exact-sized, so a model whose
                    # non-top-k pool cannot supply the draw contributes ZERO usable
                    # repeats -- e.g. Purkinje__macaque at Top2000, whose pool holds
                    # 3502 peaks while a 1980-pair draw needs 3960. `passes_null` is
                    # False in that case because it is initialised False for an empty
                    # table, NOT because the observation was compared and lost, so
                    # reading it as a FAIL reports a power failure as a scientific
                    # negative. `passes` is null (no verdict) rather than false (a
                    # failed verdict) when PENDING, the same convention P3 uses below
                    # and the one `cross_arm.underpowered_policy` freezes.
                    status=("PENDING" if not int(gate_row["n_repeats"]) else
                            "PASS" if bool(gate_row["passes_null"]) else "FAIL"),
                    passes=(None if not int(gate_row["n_repeats"])
                            else bool(gate_row["passes_null"])),
                    basis_scope=gate_row["scope"],
                    species_used=gate_row["species_used"],
                    n_species_used=int(gate_row["n_species_used"]),
                    n_species_in_scope=len(observed_species),
                    n_repeats_used=int(gate_row["n_repeats"]),
                    n_repeats_skipped=int(gate_row["n_repeats_skipped"]),
                    observed=float(gate_row["observed"]),
                    null_mean=_maybe_float(gate_row["null_mean"]),
                    null_q95=_maybe_float(gate_row["null_q95"]),
                    empirical_p=_maybe_float(gate_row["empirical_p"]),
                    note=gate_row["note"],
                    species_scope_note=scope_note,
                ),
                P3=dict(
                    statement="model recurrent program supported by raw Purkinje chromVAR",
                    status="PENDING",
                    # Every gate object carries both keys: `passes` is null (no verdict)
                    # rather than false (a failed verdict) when the gate is not evaluated.
                    passes=None,
                    detail="raw chromVAR is not an input of s06; see s07_purkinje_raw_validation.py",
                ),
            ),
            # The module layer's read-out is deliberately NOT a member of `gates`.
            # `gates` is the PRIMARY gate panel: every member of it is decided by the
            # unbiased 879-motif profile, the matched random-500 null or (P3, upstream)
            # the raw chromVAR profile, so every member is bit-invariant to the module
            # map. P2 is the one gate that no primary evidence decides -- it is the
            # interpretation layer's verdict on that evidence -- and publishing it
            # inside `gates` would let a membership edit of the frozen module map move
            # a member of the step's primary gate panel, which V2_PLAN section 1
            # forbids. The payload keeps its id `P2`, its `layer` and every field it
            # had, so a reader that walks the document (s11's `interpretation_gates`)
            # still resolves it as this claim's interpretation gate; it is only kept
            # out of the panel that a primary PASS/FAIL is read from.
            interpretation=dict(
                role="interpretation_only_never_a_primary_gate",
                note=(
                    "P2 is the module layer's own verdict on the primary evidence. It is "
                    "reported here rather than in `gates` so that emptying the module map "
                    "(or pointing modules.file at a map with no rows) cannot move any member "
                    "of the primary gate panel: gates holds only the gates whose layer is "
                    "primary, and the module layer never decides a primary PASS/FAIL."
                ),
                P2=dict(
                    statement="at least one recurrent module of the Purkinje panel (primary modules only)",
                    layer="interpretation",
                    primary_evidence=False,
                    status=(
                        "PENDING"
                        if not panel
                        else ("PASS" if bool(len(gate_recurrent) >= 1) else "FAIL")
                    ),
                    # `passes` stays None when the layer is off: the gate was not
                    # evaluated, which is not the same as a failed verdict.
                    passes=None if not panel else bool(len(gate_recurrent) >= 1),
                    detail=(
                        ""
                        if panel
                        else "the module map declares no module for this cell type, so the "
                        "interpretation gate has nothing to evaluate"
                    ),
                    recurrent_modules=gate_recurrent.module.tolist(),
                    # Reported for the reader, but a recurrent module at a non-gating
                    # level is NOT a verdict: it is listed here and nowhere else.
                    recurrent_modules_all_levels=recurrent.module.tolist(),
                    n_modules_in_panel=int(len(panel)),
                    gating_levels=gating_levels,
                    non_gating_levels=non_gating_levels,
                    # The rule in force: K = max(floor, ceil(2N/3)) over the species this
                    # run kept. Five v2 species give max(4, 4) = 4, exactly as six did,
                    # so the dropped species cannot relax the bar.
                    min_species_required=int(recurrence_min),
                    n_species_in_scope=int(n_species),
                    recurrence_fraction=float(RECURRENT_SPECIES_FRACTION),
                    recurrence_floor=int(MIN_RECURRENT_SPECIES),
                    recurrence_rule=(
                        f"module F_abs (median monomeric log2_OR, ABSOLUTE -- not the centered "
                        f"heatmap value) > 0 in >= {recurrence_min} of {n_species} species AND a "
                        f"monomeric member at FDR < {alpha} with log2_OR > 0 in >= {recurrence_min} "
                        f"of {n_species} species, with {recurrence_min} = max({int(MIN_RECURRENT_SPECIES)}, "
                        f"ceil(2 x {n_species}/3)); only modules whose level may gate may satisfy "
                        f"this gate -- levels {non_gating_levels} are declared non-gating in "
                        "modules.non_gating_levels and can never decide P2, whatever their "
                        "recurrence column says"
                    ),
                ),
            ),
            **{
                f"strict_{number_word(len(strict.species))}_species_scope": dict(
                    usable=bool(strict.name == gate_scope.name),
                    species=list(strict.species),
                    n_species=int(len(strict.species)),
                    n_repeats_used=int(strict_row.n_repeats.iloc[0]) if len(strict_row) else 0,
                    note="" if strict.name == gate_scope.name else gate_scope.note,
                )
            },
            null_coverage=dict(
                profiles=int(coverage.n_profiles),
                rows_in_file=int(coverage.n_rows_total),
                repeats_in_file=len(coverage.repeats),
                repeats_requested=int(args.repeats) if args.repeats else None,
            ),
            module_layer=dict(
                map_file=str(C.module_map_path(config)),
                celltype=ANALYSIS,
                n_modules=int(len(panel)),
                modules=[item.module for item in panel],
                levels=sorted({item.level for item in panel}),
                gating_levels=gating_levels,
                non_gating_levels=non_gating_levels,
                gating_rule=(
                    "a module may decide Gate P2 only when common.module_is_gating(config, level) "
                    "is true; a derived union may gate only when every one of its constituents may"
                ),
                derived_unions=unions,
                derived_union_note=unions_note,
                recurrence_basis=(
                    "F_abs = median over the module's monomeric members of the ABSOLUTE log2_OR, "
                    "per species; no cross-species centering enters recurrence"
                ),
                centered_scores_note=(
                    "F_centered = F_abs - mean over the species in scope is written to "
                    "module_scores_centered.tsv and drawn in module_heatmap only; it is never "
                    "used for a recurrence count or for Gate P2"
                ),
                member_scope=(
                    "monomer; heterodimers never enter a module score or a significance count"
                ),
                recurrence_table="module_recurrence_all.tsv (the whole panel, recurrent or not)",
                recurrent_table="recurrent_modules.tsv",
                primary_evidence_untouched=(
                    "the 879-motif observed matrix, the mean pairwise similarity, the matched-random "
                    "null and Gate P1 are computed from all_model_motif_scores.tsv and "
                    "random_null_scores.tsv.gz alone and never read the module layer"
                ),
            ),
            module_significance_source="significant_column" if "significant" in scores.columns else "recomputed",
            module_member_scope="monomer",
            phylogeny_secondary=phylo_payload,
            outputs=sorted({p.name for p in out_dir.iterdir()} | {"purkinje_summary.json"}),
        )
        C.save_json(summary, out_dir / "purkinje_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
