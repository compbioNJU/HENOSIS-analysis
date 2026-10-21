"""Species, developmental-stage, cell-type and colour vocabulary.

This module is the single source of truth for every label the analyses use. A
duplicated stage table is how the previous release ended up with two different
`stage_rank` conventions in two scripts that were supposed to agree, so any
script that needs a rank imports it from here.

Three ideas are kept deliberately separate, because fusing them is what made the
old tables ambiguous:

    chronological order   what came before what          -> STAGE_ORDER
    stage_rank            a relative age in [0, 1]       -> stage_rank()
    observation unit      a (sample, stage, celltype)    -> sample_id()

`stage_rank` is a *calibration of developmental time*, not the ordinal position
of the stage in a list. Human Carnegie stages 18-23 span roughly nine days while
`adult` is years later, so evenly spacing the labels would misstate the axis.
The table below is the calibration used for every figure in this repository.
`stage_index` (even spacing) is provided alongside it so the sensitivity of a
result to the calibration can be reported rather than assumed away.

The `_DN` (deep nuclei) and `CerebellumWM` samples are a *dissection region*, not
an age. They receive their age-matched rank and are flagged by
`is_special_anatomy()`; whether they are dropped is each analysis's decision, and
each analysis states it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Species
# ---------------------------------------------------------------------------

# `Common Name` in the atlas -> the short name used in every figure.
COMMON_TO_SHORT: dict[str, str] = {
    "Human": "Human",
    "Bonobo": "Bonobo",
    "Rhesus macaque": "Macaque",
    "Common marmoset": "Marmoset",
    "House mouse": "Mouse",
    "Gray short-tailed opossum": "Opossum",
}

# `Species` in the atlas (lower case) -> short name.
SPECIES_TO_SHORT: dict[str, str] = {
    "human": "Human",
    "bonobo": "Bonobo",
    "macaque": "Macaque",
    "marmoset": "Marmoset",
    "mouse": "Mouse",
    "opossum": "Opossum",
}

# Fixed display order. This is a drawing order, not a claim about relatedness.
SPECIES_ORDER: tuple[str, ...] = (
    "Human", "Bonobo", "Macaque", "Marmoset", "Mouse", "Opossum",
)

# The three species with a densely sampled developmental series. The other
# three have two stages or fewer and can support a direction estimate only in
# the species that has >= 2 stages; they are never dropped silently -- whatever
# is excluded is reported.
DENSE_SERIES_SPECIES: tuple[str, ...] = ("Human", "Marmoset", "Mouse")
SPARSE_SERIES_SPECIES: tuple[str, ...] = ("Bonobo", "Opossum", "Macaque")

# ---------------------------------------------------------------------------
# Developmental stages
# ---------------------------------------------------------------------------

# Chronological order within each species, with `_DN` samples last.
STAGE_ORDER: dict[str, tuple[str, ...]] = {
    "Human": ("CS18-19", "CS20", "CS22-23", "11wpc", "15-17wpc",
              "newborn", "infant", "toddler", "adult", "adult_DN"),
    "Bonobo": ("6mo", "3yo"),
    "Macaque": ("P0",),
    "Marmoset": ("GD70", "GD72", "GD73", "GD76", "GD80", "GD82",
                 "P1", "P2", "P60", "adult"),
    "Mouse": ("E10", "E11", "E12", "E13", "E15", "E17",
              "P0", "P4", "P7", "P14", "P63", "P63_DN"),
    "Opossum": ("P21", "adult"),
}

# Relative developmental age in [0, 1], per species.
#
# The calibration is read off the sampling design rather than the label list:
# within a species the stages span very different real durations (eight mouse
# embryonic days vs. the two months from P14 to P63), and the values below place
# them accordingly. `adult_DN` and `P63_DN` carry their age-matched cortical
# rank because they are a dissection region rather than a later age.
STAGE_RANK: dict[str, dict[str, float]] = {
    "Human": {"CS18-19": 0.00, "CS20": 0.08, "CS22-23": 0.16, "11wpc": 0.25,
              "15-17wpc": 0.36, "newborn": 0.52, "infant": 0.68,
              "toddler": 0.82, "adult": 1.00, "adult_DN": 1.00},
    "Bonobo": {"6mo": 0.00, "3yo": 1.00},
    "Macaque": {"P0": 0.00},
    "Marmoset": {"GD70": 0.00, "GD72": 0.06, "GD73": 0.12, "GD76": 0.24,
                 "GD80": 0.34, "GD82": 0.42, "P1": 0.52, "P2": 0.56,
                 "P60": 0.78, "adult": 1.00},
    "Mouse": {"E10": 0.00, "E11": 0.08, "E12": 0.16, "E13": 0.24,
              "E15": 0.36, "E17": 0.48, "P0": 0.56, "P4": 0.64,
              "P7": 0.71, "P14": 0.80, "P63": 1.00, "P63_DN": 1.00},
    "Opossum": {"P21": 0.00, "adult": 1.00},
}

# Even spacing over the chronological list, for sensitivity only. `_DN` stages
# are placed by their position in STAGE_ORDER, i.e. last, which is what the
# even-spaced convention of the previous release did.
STAGE_INDEX: dict[str, dict[str, float]] = {
    species: {
        stage: (i / (len(stages) - 1) if len(stages) > 1 else 0.0)
        for i, stage in enumerate(stages)
    }
    for species, stages in STAGE_ORDER.items()
}

# Deep-nuclei / white-matter dissections: a region, not an age.
DEEP_NUCLEI_STAGES: frozenset[str] = frozenset({"adult_DN", "P63_DN"})
SPECIAL_ANATOMY_SAMPLE_PATTERN: str = "CerebellumDN|CerebellumWM"

# Broad developmental periods, used for grouping (not for ranking).
PERIOD_OF_STAGE: dict[str, str] = {
    "CS18-19": "early_embryonic", "CS20": "early_embryonic",
    "CS22-23": "early_embryonic",
    "11wpc": "prenatal", "15-17wpc": "prenatal",
    "GD70": "prenatal", "GD72": "prenatal", "GD73": "prenatal",
    "GD76": "prenatal", "GD80": "prenatal", "GD82": "prenatal",
    "E10": "prenatal", "E11": "prenatal", "E12": "prenatal",
    "E13": "prenatal", "E15": "prenatal", "E17": "prenatal",
    "newborn": "postnatal", "infant": "postnatal", "toddler": "postnatal",
    "P1": "postnatal", "P2": "postnatal", "P60": "postnatal",
    "P0": "postnatal", "P4": "postnatal", "P7": "postnatal", "P14": "postnatal",
    "adult": "adult", "adult_DN": "adult", "P63": "adult", "P63_DN": "adult",
    "6mo": "juvenile", "3yo": "adult", "P21": "juvenile",
}

# ---------------------------------------------------------------------------
# Cell types
# ---------------------------------------------------------------------------

# `Org_Celltype` (figure 1 vocabulary) -> `Celltype` (figure 2 vocabulary).
# Carried forward from the previous release with one addition: the cells that
# moved out of `astroglia` land in `progenitor`, which is what the atlas's own
# maturation field calls 100% of them.
FIGURE1_TO_FIGURE2: dict[str, str] = {
    "GABA_DN": "GABAergic DN",
    "GABA_MB": "GABAergic midbrain",
    "GC": "granule cell (GC)",
    "GC/UBC": "GC_UB",
    "MBO": "midbrain-originating cell",
    "NTZ_mixed": "mixed NTZ cells",
    "NTZ_neuroblast": "NTZ neuroblast",
    "Purkinje": "Purkinje",
    "UBC": "unipolar brush cell (UBC)",
    "VZ_neuroblast": "VZ neuroblast",
    "astroglia": "astroglia (progenitors+astrocytes)",
    "ependymal": "ependymal",
    "erythroid": "erythroid",
    "glut_DN": "glutamatergic DN",
    "immune": "immune",
    "interneuron": "interneuron",
    "isth_N": "isthmic nuclei neuron",
    "isthmic_neuroblast": "isthmic neuroblast",
    "meningeal": "meningeal",
    "motorneuron": "motor neuron",
    "mural/endoth": "mural/endothelial",
    "neural_crest_progenitor": "neural crest progenitor",
    "noradrenergic": "noradrenergic",
    "oligo": "oligodendrocyte",
    "parabrachial": "parabrachial",
    "progenitor": "progenitor",
}

# The family label whose definition changed in this release, and the class the
# human Carnegie-stage cells moved into.
ASTROGLIA_FAMILY_LABEL: str = "astroglia (progenitors+astrocytes)"
PROGENITOR_LABEL: str = "progenitor"

# Cell types whose developmental direction is never estimable because they are
# present in a single species' series. Reported, not silently dropped.
SINGLE_SPECIES_CELLTYPES: tuple[str, ...] = (PROGENITOR_LABEL,)

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

# Frozen. These are the colours of the previous release's Figure 1, kept byte for
# byte so that the only visible change in a re-run is the change in the data.
# They are listed explicitly rather than generated from a colormap index: an
# index-based palette shifts every colour after the first insertion, which would
# have recoloured `unipolar brush cell (UBC)` the moment `progenitor` appeared.
CELLTYPE_COLORS: dict[str, str] = {
    "GABAergic DN": "#1f77b4",
    "GC_UB": "#aec7e8",
    "NTZ neuroblast": "#ff7f0e",
    "Purkinje": "#ffbb78",
    "VZ neuroblast": "#2ca02c",
    "astroglia (progenitors+astrocytes)": "#98df8a",
    "ependymal": "#d62728",
    "erythroid": "#ff9896",
    "glutamatergic DN": "#9467bd",
    "granule cell (GC)": "#c5b0d5",
    "immune": "#8c564b",
    "interneuron": "#c49c94",
    "isthmic neuroblast": "#e377c2",
    "isthmic nuclei neuron": "#f7b6d2",
    "meningeal": "#7f7f7f",
    "midbrain-originating cell": "#c7c7c7",
    "mixed NTZ cells": "#bcbd22",
    "mural/endothelial": "#dbdb8d",
    "noradrenergic": "#17becf",
    "oligodendrocyte": "#9edae5",
    "parabrachial": "#393b79",
    "unipolar brush cell (UBC)": "#5254a3",
    # New in this release. Chosen from the same tab20b family as its neighbours
    # so that it reads as related to `astroglia` while staying separable at
    # point size.
    "progenitor": "#637939",
}

SPECIES_COLORS: dict[str, str] = {
    "Human": "#1F77B4",
    "Bonobo": "#8C564B",
    "Macaque": "#D62728",
    "Marmoset": "#2CA02C",
    "Mouse": "#FF7F0E",
    "Opossum": "#7F7F7F",
}

# A colour-blind-safe sequential ramp for "relative developmental stage", and
# its categorical companion for species panels.
STAGE_CMAP: str = "viridis"
DIVERGING_CMAP: str = "vlag"

# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------

ALL_STAGES: tuple[str, ...] = tuple(
    stage for species in SPECIES_ORDER for stage in STAGE_ORDER[species]
)

# (short species, stage) -> rank. Keyed on the PAIR, never on the stage alone:
# `P0` is both mouse and macaque, and `adult` is both marmoset and opossum.
_STAGE_RANK_LOOKUP: dict[tuple[str, str], float] = {
    (species, stage): rank
    for species, table in STAGE_RANK.items()
    for stage, rank in table.items()
}

_STAGE_INDEX_LOOKUP: dict[tuple[str, str], float] = {
    (species, stage): rank
    for species, table in STAGE_INDEX.items()
    for stage, rank in table.items()
}


@dataclass(frozen=True)
class Design:
    """Per-cell design columns derived from an atlas `obs` frame.

    Keeping this as one object rather than four loose Series means a script
    cannot accidentally use a rank computed under one convention alongside an
    order computed under another.
    """

    species: pd.Series          # short name
    species_key: pd.Series      # atlas `Species` value, lower case
    sample_id: pd.Series
    stage: pd.Series
    stage_rank: pd.Series
    stage_index: pd.Series
    special_anatomy: pd.Series  # bool

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "species": self.species,
            "species_key": self.species_key,
            "sample_id": self.sample_id,
            "stage": self.stage,
            "stage_rank": self.stage_rank,
            "stage_index": self.stage_index,
            "special_anatomy": self.special_anatomy,
        })


def short_species(common_name: pd.Series) -> pd.Series:
    """`Common Name` (or `Species`) -> short name. Fails loudly on anything new.

    A silent `.map()` here would turn an unrecognised species into NaN and then
    into a dropped group several steps later, where the cause is no longer
    visible.
    """
    values = common_name.astype(str)
    mapped = values.map(COMMON_TO_SHORT)
    if mapped.isna().any():
        mapped = values.map(SPECIES_TO_SHORT)
    unknown = sorted(set(values[mapped.isna()]))
    if unknown:
        raise ValueError(
            f"unrecognised species label(s): {unknown}. Known values are the "
            f"keys of COMMON_TO_SHORT / SPECIES_TO_SHORT in src/common/labels.py. "
            f"Add the species there rather than dropping it here."
        )
    return mapped


def sample_id(index: pd.Index) -> pd.Series:
    """`<sample>_<species>_<tissue>_<stage>[#barcode]` -> `<sample>`.

    The atlas index carries the library id before the first `#`. If a future
    atlas drops that convention, the barcode would be read as part of the sample
    and every sample-level statistic would silently change unit, so the shape is
    asserted rather than assumed.
    """
    texts = pd.Series(index.astype(str), index=index)
    if not texts.str.contains("#").all():
        bad = int((~texts.str.contains("#")).sum())
        raise ValueError(
            f"{bad} cell id(s) do not contain '#'; the sample id cannot be "
            f"recovered and the sample-level unit would change. Example: "
            f"{texts.iloc[0]!r}"
        )
    return texts.str.split("#").str[0]


def is_special_anatomy(sample: pd.Series, stage: pd.Series) -> pd.Series:
    """Deep nuclei / white matter: a dissection region, not a developmental age."""
    by_stage = stage.astype(str).isin(DEEP_NUCLEI_STAGES)
    by_sample = sample.astype(str).str.contains(
        SPECIAL_ANATOMY_SAMPLE_PATTERN, regex=True
    )
    return (by_stage | by_sample).astype(bool)


def build_design(obs: pd.DataFrame) -> Design:
    """Derive every design column from an `obs` frame in one place.

    `obs` must carry `Common Name` (or `Species`), `Org_stage`, and an index
    whose first `#`-separated field is the sample id.
    """
    if "Common Name" in obs.columns:
        species = short_species(obs["Common Name"])
    elif "Species" in obs.columns:
        species = short_species(obs["Species"])
    else:
        raise ValueError("obs has neither 'Common Name' nor 'Species'")
    if "Org_stage" not in obs.columns:
        raise ValueError("obs has no 'Org_stage' column")

    species_key = obs["Species"].astype(str) if "Species" in obs.columns else species.str.lower()
    sample = sample_id(obs.index)
    stage = obs["Org_stage"].astype(str)

    rank = pd.Series(
        [_STAGE_RANK_LOOKUP.get((s, t), np.nan) for s, t in zip(species, stage)],
        index=obs.index, dtype=float,
    )
    index_rank = pd.Series(
        [_STAGE_INDEX_LOOKUP.get((s, t), np.nan) for s, t in zip(species, stage)],
        index=obs.index, dtype=float,
    )
    missing = rank.isna()
    if missing.any():
        bad = pd.DataFrame({"species": species[missing], "stage": stage[missing]})
        bad = bad.drop_duplicates().sort_values(["species", "stage"])
        raise ValueError(
            "unrecognised (species, stage) pair(s) -- no developmental rank "
            "exists for them:\n" + bad.to_string(index=False) +
            "\nAdd them to STAGE_ORDER and STAGE_RANK in src/common/labels.py."
        )

    return Design(
        species=species,
        species_key=species_key,
        sample_id=sample,
        stage=stage,
        stage_rank=rank,
        stage_index=index_rank,
        special_anatomy=is_special_anatomy(sample, stage),
    )


def stage_rank_table() -> pd.DataFrame:
    """The stage vocabulary as a tidy table, for writing next to the results."""
    rows = []
    for species in SPECIES_ORDER:
        for position, stage in enumerate(STAGE_ORDER[species]):
            rows.append({
                "species": species,
                "position": position,
                "stage": stage,
                "stage_rank": STAGE_RANK[species][stage],
                "stage_index": STAGE_INDEX[species][stage],
                "period": PERIOD_OF_STAGE.get(stage, "unassigned"),
                "deep_nuclei": stage in DEEP_NUCLEI_STAGES,
            })
    return pd.DataFrame(rows)


def celltype_order(labels) -> list[str]:
    """Stable display order: the frozen palette order, then anything unknown."""
    known = list(CELLTYPE_COLORS)
    present = set(labels)
    ordered = [label for label in known if label in present]
    ordered += sorted(present - set(ordered))
    return ordered
