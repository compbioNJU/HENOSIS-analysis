#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Science-anchor plate for the Oligo arm: four bands, one row per anchor matrix.

WHY this figure exists
----------------------
The primary Oligo analysis works on whole motif modules (SOX, bHLH_Ebox, NKX,
POU3). A module median says what the family does on average, and an average can
hide the named regulator the source paper actually put in the cerebellar GRN.
This plate asks the narrower question on the anchors themselves:

    bHLH_Ebox is 6/6 concordant-NEGATIVE -- is OLIG2 itself negative, or is it
    diluted by the other E-box motifs?
    NKX and POU3 are 3/6 -- are NKX2-2 / POU3F3 / POU3F4 better conserved than
    their families?

It is a SUPPLEMENTARY, post-primary interpretation. It reads three frozen tables,
forms no statistic, gates nothing, and changes no claim.

WHY unmeasurable anchors are drawn rather than dropped
------------------------------------------------------
Three of the eleven frozen anchor matrices cannot be evaluated in this framework:
SOX10 (MA0442.3), NKX6-2 (MA0675.2) and one of the two matrices named OLIG2
(MA1997.2) match no peak at the frozen FIMO per-site threshold, and their PWMs
are too short/unspecific to change that. A blank cell would read as "no
enrichment" and a zero would read as "measured, and it is zero". Both are wrong.
Such rows are therefore DRAWN, hatched, and labelled
`unmeasurable (no peak matches)`; their scores are absent (NaN) and they are
excluded from every per-anchor aggregate.

WHY the two symbols in panel A mean opposite things
---------------------------------------------------
The project's per-motif FDR comes from a ONE-SIDED Fisher test for enrichment.
A negative log2 OR is therefore NOT evidence of depletion on that FDR, and the
depletion evidence is a separate statistic -- the lower tail of the
matched-random draws. Panel A marks them with two different symbols ('+' for
the FDR-supported enrichment, 'v' for the matched-random lower tail) precisely
so that one star can never be read as the other.

Layout
    A  model Top<K> log2 OR heatmap           (+ / v support symbols)
    B  full-peak chromVAR C heatmap           (OWN colour scale, never shared
                                               with A: the two differ ~10x)
    C  model/raw sign concordance             red both positive, blue both
                                               negative, grey discordant,
                                               white unavailable
    D  per-anchor cross-species concordance   n/N with N stated

Reads (nothing else, and nothing is recomputed here):

    <the arm config's results dir>/09_oligo_validation/science_anchors/
        anchor_manifest.tsv
        anchor_species_scores.tsv
        anchor_random500_tests.tsv
        anchor_cross_species_summary.tsv

Usage
    python script/s17d_plot_oligo_science_anchors.py \
        --config config/analysis_v2.yaml
    python script/s17d_plot_oligo_science_anchors.py --outdir <dir>   # same flag as s17b/s17c
    # tests: --input <dir> --summary <file> --output-prefix <stem>
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import family_trajectory_style as S  # noqa: E402

STEP = "09_oligo_validation"
SUBDIR = "science_anchors"
FIGURE_STEM = "oligo_science_anchor_heatmap"

MANIFEST_FILE = "anchor_manifest.tsv"
SCORES_FILE = "anchor_species_scores.tsv"
TESTS_FILE = "anchor_random500_tests.tsv"
SUMMARY_FILE = "anchor_cross_species_summary.tsv"

INK = S.INK
MUTED = S.MUTED
GRID = S.GRID

# One colour language with the rest of the project: warm = the two sources agree
# on ENRICHMENT, cool = they agree on DEPLETION, grey = they disagree.
POSITIVE = "#B2182B"
NEGATIVE = "#2166AC"
DISCORDANT = "#BFBFBF"
UNAVAILABLE = "#FFFFFF"
BAD_CELL = "#F2F2F2"          # a cell with no value, at all

# Categorical codes for panel C. The TIE code exists because a score of exactly
# zero has NO SIGN -- upstream leaves `sign_concordant` empty for such a cell
# rather than FALSE -- so it must not be drawn as "the two sources disagree". It
# keeps the discordant colour (it is not concordant either) but carries its own
# glyph, so a reader can tell a sign conflict from a tie.
CODE_DISCORDANT, CODE_NEGATIVE, CODE_POSITIVE, CODE_TIE = 0, 1, 2, 3
CONCORD_CMAP = ListedColormap([DISCORDANT, NEGATIVE, POSITIVE, DISCORDANT])
CONCORD_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], CONCORD_CMAP.N)

# --------------------------------------------------------------------------
# column resolution
#
# The three tables are written by the analysis step. Their column names are part
# of the contract, but a renamed column must fail LOUDLY here rather than
# silently drop a panel or a symbol, so every read goes through `_column` and
# reports the columns it actually found when nothing matches.
# --------------------------------------------------------------------------

CANDIDATES = {
    "group": ("group", "anchor_group", "family", "module"),
    "requested_name": ("requested_name", "anchor", "anchor_name", "requested",
                       "tf_name", "name"),
    "motif_id": ("motif_id", "matrix_id", "catalog_id", "jaspar_id", "id"),
    "motif_name": ("motif_name", "catalog_name", "matrix_name"),
    "species": ("species", "organism"),
    "n_catalog_ids": ("n_catalog_ids", "n_catalog_matches", "n_matches", "n_matrices"),
    "present": ("present", "is_present", "matched"),
    "measurable": ("measurable", "is_measurable", "has_hits"),
    "n_fg_hits": ("n_fg_hits", "n_foreground_hits_max", "n_foreground_hits",
                  "foreground_hits", "n_hits", "n_peak_hits"),
    "model_log2_or": ("model_log2_OR", "model_log2_or", "log2_or", "model_log2or",
                      "log2OR", "model_log2_odds_ratio"),
    "model_fdr": ("model_FDR", "model_fdr", "fdr", "model_fdr_bh", "q_value"),
    "chromvar_C": ("chromvar_C", "chromvar_c", "chromvar", "chromvar_contrast",
                   "chromvar_score"),
    "model_sign": ("model_sign",),
    "chromvar_sign": ("chromvar_sign",),
    "sign_concordant": ("sign_concordant", "sign_agree"),
    "observed_log2_or": ("observed_log2_OR", "observed_log2_or", "observed",
                         "observed_value"),
    "n_null_usable": ("n_null_usable", "n_null", "n_usable"),
    "model_fdr_positive_supported": ("model_fdr_positive_supported",
                                     "positive_supported", "enrichment_supported",
                                     "model_positive_supported"),
    "lower_tail_supported": ("lower_tail_supported", "depletion_supported",
                             "lower_tail_significant"),
    "p_upper": ("p_upper", "p_greater", "p_upper_tail"),
    "p_lower": ("p_lower", "p_less", "p_lower_tail"),
    "p_two_sided": ("p_two_sided", "p_two_tail"),
    "n_evaluable": ("n_species_available", "n_evaluable", "n_species_evaluable",
                    "n_species_scored", "n_scored", "n_species"),
    "n_both_positive": ("n_both_positive", "n_concordant_positive",
                        "n_both_pos", "n_both_enriched"),
    "n_both_negative": ("n_both_negative", "n_concordant_negative",
                        "n_both_neg", "n_both_depleted"),
    "n_discordant": ("n_discordant", "n_discordant_sign", "n_disagree"),
    "n_concordant": ("n_sign_concordant", "n_concordant", "n_same_sign",
                     "n_sign_agree", "n_concordant_species"),
    "concordance_fraction": ("sign_concordance_fraction", "concordance_fraction",
                             "concordant_fraction", "sc"),
    "n_model_fdr_positive_supported": ("n_model_fdr_positive_supported",
                                       "n_positive_supported",
                                       "n_species_model_positive_supported"),
    "n_lower_tail_supported": ("n_lower_tail_supported",
                               "n_depletion_supported",
                               "n_species_lower_tail_supported"),
    "n_species_drawn": ("n_species_drawn", "n_columns", "n_models"),
}


def _column(frame: pd.DataFrame, role: str, *, required: bool = True,
            what: str = "table") -> str | None:
    for name in CANDIDATES[role]:
        if name in frame.columns:
            return name
    if required:
        raise KeyError(
            f"{what}: no column for '{role}'. Looked for {list(CANDIDATES[role])}; "
            f"the file provides {list(frame.columns)}. Refusing to guess -- a "
            f"missing column here would silently drop a panel or a symbol."
        )
    return None


def _optional_column(frame: pd.DataFrame, role: str) -> str | None:
    return _column(frame, role, required=False)


def _first_present(record: dict, role: str) -> str | None:
    for name in CANDIDATES[role]:
        if name in record:
            return name
    return None


def _flag_series(series: pd.Series) -> np.ndarray:
    """Booleans that survive a CSV round trip (True/False, 1/0, yes/no)."""
    def one(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(np.isfinite(value) and value != 0)
        text = str(value).strip().lower()
        return text in {"true", "t", "1", "1.0", "yes", "y"}

    return np.array([one(v) for v in series], dtype=bool)


def _text_series(series: pd.Series) -> np.ndarray:
    """Strings with NaN / NA normalised to "" so a key can be built from them."""
    out = []
    for value in series:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            out.append("")
        else:
            text = str(value).strip()
            out.append("" if text.lower() in {"nan", "na", "none"} else text)
    return np.array(out, dtype=object)


def _finite_series(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


# --------------------------------------------------------------------------
# the row spine
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchorRow:
    group: str
    requested_name: str
    motif_id: str
    motif_name: str
    n_catalog_ids: int
    present: bool
    measurable: bool
    n_fg_hits: float
    duplicate_index: int          # 1-based position among same-named matrices

    @property
    def drawable(self) -> bool:
        return self.present and self.measurable

    @property
    def symbol(self) -> str:
        """Compact self-describing suffix used in the row label."""
        if not self.present:
            return "absent (no catalog match)"
        parts = []
        if self.n_catalog_ids > 1:
            parts.append(f"[{self.duplicate_index}/{self.n_catalog_ids}]")
        if not self.measurable:
            parts.append("unmeasurable (no peak matches)")
        return "  ".join(parts)

    @property
    def label(self) -> str:
        if not self.present:
            return f"{self.requested_name}\n{self.symbol}"
        head = f"{self.requested_name}  {self.motif_id}"
        if self.motif_name and self.motif_name.upper() != self.motif_id.upper():
            head += f"  ({self.motif_name})"
        suffix = self.symbol
        return f"{head}\n{suffix}" if suffix else head


def declared_groups(config: dict) -> list[tuple[str, list[str]]]:
    anchors = config.get("oligo_science_anchors")
    if not isinstance(anchors, dict):
        raise KeyError("config has no `oligo_science_anchors` block")
    groups = [(key, list(value)) for key, value in anchors.items()
              if isinstance(value, list) and value]
    if not groups:
        raise KeyError("`oligo_science_anchors` declares no anchor group")
    return groups


def build_rows(config: dict, manifest: pd.DataFrame,
               manifest_path: Path) -> list[AnchorRow]:
    """One AnchorRow per catalog matrix, ordered exactly as the config declares.

    The manifest is the authority on which matrices exist; the config is the
    authority on their order and on which names were requested. A disagreement in
    either direction is fatal, because either one would silently drop an anchor.
    """
    col_group = _column(manifest, "group", what=manifest_path.name)
    col_name = _column(manifest, "requested_name", what=manifest_path.name)
    col_mid = _column(manifest, "motif_id", required=False, what=manifest_path.name)
    col_mname = _column(manifest, "motif_name", required=False, what=manifest_path.name)
    col_present = _column(manifest, "present", required=False, what=manifest_path.name)
    col_meas = _column(manifest, "measurable", required=False, what=manifest_path.name)
    col_ncat = _column(manifest, "n_catalog_ids", required=False, what=manifest_path.name)
    col_hits = _column(manifest, "n_fg_hits", required=False, what=manifest_path.name)

    frame = manifest.copy()
    frame["_group"] = _text_series(frame[col_group])
    frame["_name"] = _text_series(frame[col_name])
    frame["_mid"] = _text_series(frame[col_mid]) if col_mid else ""
    frame["_mname"] = _text_series(frame[col_mname]) if col_mname else ""
    frame["_present"] = _flag_series(frame[col_present]) if col_present else True
    frame["_meas"] = _flag_series(frame[col_meas]) if col_meas else False
    frame["_ncat"] = (pd.to_numeric(frame[col_ncat], errors="coerce").fillna(1)
                      .astype(int).to_numpy() if col_ncat else 1)
    frame["_hits"] = _finite_series(frame[col_hits]) if col_hits else np.nan

    groups = declared_groups(config)
    declared = {(group, name) for group, names in groups for name in names}
    present_pairs = set(zip(frame["_group"], frame["_name"]))

    missing = sorted(f"{g}:{n}" for g, n in declared if (g, n) not in present_pairs)
    if missing:
        raise KeyError(
            "the frozen anchor list declares names that the manifest does not "
            f"resolve: {missing}. A requested name may be absent, but it must "
            "still be a row -- never dropped. Refusing to draw an incomplete plate."
        )
    extra = sorted(f"{g}:{n}" for g, n in present_pairs if (g, n) not in declared)
    if extra:
        raise KeyError(
            f"the manifest carries anchors the frozen config does not declare: {extra}. "
            "Adding an anchor after the fact is exactly what the frozen list prevents."
        )

    rows: list[AnchorRow] = []
    for group, names in groups:
        for name in names:
            block = frame[(frame["_group"] == group) & (frame["_name"] == name)]
            # deterministic within a name: catalog matrix id order
            block = block.sort_values("_mid", kind="stable")
            ncat = int(block["_ncat"].max()) if len(block) else 1
            for position, (_, record) in enumerate(block.iterrows(), start=1):
                rows.append(AnchorRow(
                    group=group,
                    requested_name=name,
                    motif_id=str(record["_mid"]),
                    motif_name=str(record["_mname"]),
                    n_catalog_ids=int(record["_ncat"]) or ncat,
                    present=bool(record["_present"]),
                    measurable=bool(record["_meas"]),
                    n_fg_hits=float(record["_hits"]),
                    duplicate_index=position,
                ))
    return rows


def group_blocks(rows: list[AnchorRow]) -> list[tuple[str, int, int]]:
    """(group, first_row_index, last_row_index_inclusive) in draw order."""
    blocks: list[tuple[str, int, int]] = []
    for index, row in enumerate(rows):
        if blocks and blocks[-1][0] == row.group:
            blocks[-1] = (blocks[-1][0], blocks[-1][1], index)
        else:
            blocks.append((row.group, index, index))
    return blocks


# --------------------------------------------------------------------------
# lookups into the two per-(anchor, species) tables
# --------------------------------------------------------------------------


def _cell_index(frame: pd.DataFrame, *, what: str) -> dict[tuple[str, str], dict]:
    """Index a per-(matrix, species) table, refusing ambiguous duplicates."""
    col_mid = _optional_column(frame, "motif_id")
    if col_mid is None:
        raise KeyError(f"{what}: no motif_id column, so cells cannot be addressed")
    col_species = _column(frame, "species", what=what)
    names = _text_series(frame[col_mid])
    species = _text_series(frame[col_species])
    index: dict[tuple[str, str], dict] = {}
    for position, key in enumerate(zip(names, species)):
        if key in index:
            raise KeyError(
                f"{what}: two rows for motif_id={key[0]!r} species={key[1]!r}. "
                "A per-cell table must have exactly one value per cell."
            )
        index[key] = frame.iloc[position].to_dict()
    return index


def species_axis(scores: pd.DataFrame, species_column: str) -> list[str]:
    found = [s for s in C.SPECIES_ORDER if s in set(_text_series(scores[species_column]))]
    unknown = sorted(set(_text_series(scores[species_column])) - set(C.SPECIES_ORDER))
    if unknown:
        raise KeyError(f"anchor_species_scores.tsv carries unknown species: {unknown}")
    return found


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------


def annotate_groups(fig, ax, rows: list[AnchorRow], *, names: bool) -> None:
    """Draw the row-group brackets in the left margin, beside the row labels.

    WHY this is measured instead of hard-coded: the group bracket has to sit
    immediately left of the longest row label, and that label is as long as the
    anchor demands ('unmeasurable (no peak matches)' is the widest). A fixed x
    would either collide with the labels or waste half the margin, so the tick
    label extents are measured after the layout is final.

    Drawn for every band, because the four bands are read independently; the
    group NAMES are drawn once, on band A where the row labels are first read.
    """
    renderer = fig.canvas.get_renderer()
    left_px = min((label.get_window_extent(renderer).x0
                   for label in ax.get_yticklabels() if label.get_text()),
                  default=0.0)
    x = left_px / fig.bbox.width - 0.010
    position = ax.get_position()
    n_rows = len(rows)
    for group, first, last in group_blocks(rows):
        y_top = position.y0 + position.height * (1.0 - first / n_rows)
        y_bottom = position.y0 + position.height * (1.0 - (last + 1) / n_rows)
        fig.add_artist(Line2D([x, x], [y_bottom, y_top],
                              transform=fig.transFigure, color=INK,
                              linewidth=1.2, solid_capstyle="butt"))
        if names:
            fig.text(x - 0.008, (y_top + y_bottom) / 2.0,
                     f"{group}\n{last - first + 1}", transform=fig.transFigure,
                     ha="right", va="center", fontsize=8.5, color=INK,
                     linespacing=1.1)


def style_matrix_axes(ax, rows: list[AnchorRow], species: list[str]) -> None:
    ax.set_xticks(range(len(species)))
    ax.set_xticklabels(species, rotation=30, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row.label for row in rows], fontsize=7.5, linespacing=1.15)
    ax.tick_params(length=0, colors=INK)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(pad=7)
    for _, first, _ in group_blocks(rows)[1:]:
        # white rule between groups: the row separator the plate is read with
        ax.axhline(first - 0.5, color="white", lw=2.8, zorder=5)


def hatch_no_value_rows(ax, rows: list[AnchorRow], n_species: int) -> None:
    """Hatch the rows whose scores cannot exist, so a white cell can never be
    read as a measured zero."""
    for index, row in enumerate(rows):
        if row.drawable:
            continue
        ax.add_patch(Rectangle((-0.5, index - 0.5), n_species, 1.0,
                               facecolor="none", edgecolor="#B4B4B4",
                               hatch="////", linewidth=0.0, zorder=6))


def panel_a(fig, spec, rows, species, values, symbols, limit, cutoff):
    ax = fig.add_subplot(spec)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(BAD_CELL)
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-limit,
                      vmax=limit, aspect="auto", interpolation="nearest")
    style_matrix_axes(ax, rows, species)
    hatch_no_value_rows(ax, rows, len(species))

    for i in range(len(rows)):
        for j in range(len(species)):
            value = values[i, j]
            if not np.isfinite(value) or not rows[i].drawable:
                continue
            # on a saturated cell the mark is drawn white, on a pale cell dark:
            # a single hard-coded colour would vanish on one half of the ramp
            ink = "white" if abs(value) > 0.55 * limit else INK
            if symbols["model_fdr_positive_supported"][i, j]:
                ax.plot(j, i, marker="+", markersize=9.5, markeredgewidth=1.8,
                        color=ink, linestyle="none", zorder=7)
            if symbols["lower_tail_supported"][i, j]:
                ax.plot(j, i, marker="v", markersize=7.0, linestyle="none",
                        markerfacecolor=ink, markeredgecolor="white",
                        markeredgewidth=0.7, zorder=7)

    ax.set_title(f"A   model {cutoff} log2 OR   ({cutoff} vs GC/length-matched "
                 f"non-{cutoff})", fontsize=10.5, color=INK, pad=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.015)
    cbar.set_label("log2 odds ratio", fontsize=8.5, color=INK)
    cbar.ax.tick_params(labelsize=7.5, colors=INK)
    cbar.outline.set_visible(False)
    return ax


def panel_b(fig, spec, rows, species, values, limit):
    ax = fig.add_subplot(spec)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(BAD_CELL)
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-limit,
                      vmax=limit, aspect="auto", interpolation="nearest")
    style_matrix_axes(ax, rows, species)
    hatch_no_value_rows(ax, rows, len(species))

    ax.set_title("B   full-peak chromVAR C   (all QC peaks; own colour scale, "
                 "never shared with A)", fontsize=10.5, color=INK, pad=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.015)
    cbar.set_label("chromVAR contrast C", fontsize=8.5, color=INK)
    cbar.ax.tick_params(labelsize=7.5, colors=INK)
    cbar.outline.set_visible(False)
    return ax


def panel_c(fig, spec, rows, species, concordance):
    ax = fig.add_subplot(spec)

    values = np.ma.masked_where(concordance < 0, concordance.astype(float))
    cmap = CONCORD_CMAP.copy()
    cmap.set_bad(UNAVAILABLE)
    ax.imshow(values, cmap=cmap, norm=CONCORD_NORM, aspect="auto",
              interpolation="nearest")
    style_matrix_axes(ax, rows, species)
    hatch_no_value_rows(ax, rows, len(species))

    glyph = {CODE_DISCORDANT: "+-", CODE_NEGATIVE: "--", CODE_POSITIVE: "++",
             CODE_TIE: "0"}
    for i in range(len(rows)):
        for j in range(len(species)):
            code = int(concordance[i, j])
            if code < 0:
                continue
            ax.text(j, i, glyph[code], ha="center", va="center", fontsize=7,
                    color="white" if code != CODE_DISCORDANT else INK, zorder=7)

    ax.set_title("C   model / chromVAR sign concordance   (per anchor × species)",
                 fontsize=10.5, color=INK, pad=8)
    handles = [
        Patch(facecolor=POSITIVE, edgecolor="white", label="both positive (++)"),
        Patch(facecolor=NEGATIVE, edgecolor="white", label="both negative (--)"),
        Patch(facecolor=DISCORDANT, edgecolor="white",
              label="discordant (+-) or no sign, 0"),
        Patch(facecolor=UNAVAILABLE, edgecolor=GRID, label="unavailable"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.26),
              frameon=False, fontsize=8, ncol=4, handlelength=1.1,
              columnspacing=1.2, borderaxespad=0.0)
    return ax


def panel_d(fig, spec, rows, species, counts, summary_source):
    ax = fig.add_subplot(spec)

    n_rows = len(rows)
    ax.set_xlim(0.0, 1.60)
    ax.set_ylim(n_rows - 0.5, -0.5)
    y = np.arange(n_rows)

    for index, row in enumerate(rows):
        record = counts[row]
        n_evaluable = record["n_evaluable"]
        if not row.drawable or n_evaluable <= 0:
            ax.text(0.015, index, row.symbol or "no evaluable species",
                    va="center", ha="left", fontsize=8, color=MUTED, style="italic")
            continue
        both_pos = record["n_both_positive"] / n_evaluable
        both_neg = record["n_both_negative"] / n_evaluable
        discordant = record["n_discordant"] / n_evaluable
        ax.barh(index, both_neg, color=NEGATIVE, height=0.62, zorder=3)
        ax.barh(index, discordant, left=both_neg, color=DISCORDANT, height=0.62,
                zorder=3)
        ax.barh(index, both_pos, left=both_neg + discordant, color=POSITIVE,
                height=0.62, zorder=3)
        concordant = record["n_both_positive"] + record["n_both_negative"]
        label = f"{concordant}/{n_evaluable}"
        if n_evaluable < len(species):
            label += f"   (N = {n_evaluable} of {len(species)} species scored)"
        ax.text(1.03, index, label, va="center", ha="left", fontsize=8.5, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels([row.label for row in rows], fontsize=7.5, linespacing=1.15)
    ax.tick_params(length=0, colors=INK)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"], fontsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    for _, first, _ in group_blocks(rows)[1:]:
        ax.axhline(first - 0.5, color="white", lw=2.8, zorder=6)
    ax.set_title("D   per-anchor cross-species concordance   (summary of C)",
                 fontsize=10.5, color=INK, pad=8)
    ax.set_xlabel("concordant fraction of EVALUABLE species, n/N "
                  "(N = species with both a model and a chromVAR value)\n"
                  f"counts: {summary_source}", fontsize=8.5, color=INK)
    return ax


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def assemble(config, rows, species, cells, tests, summary_frame) -> dict:
    """Collect the per-cell reads into the matrices the four panels draw.

    Nothing is estimated here: a value is what the table says, or it is absent.
    """
    n_rows, n_species = len(rows), len(species)
    model = np.full((n_rows, n_species), np.nan)
    raw = np.full((n_rows, n_species), np.nan)
    model_fdr = np.full((n_rows, n_species), np.nan)
    p_lower = np.full((n_rows, n_species), np.nan)
    flag_positive = np.zeros((n_rows, n_species), dtype=bool)
    flag_lower = np.zeros((n_rows, n_species), dtype=bool)
    concordance = np.full((n_rows, n_species), -1, dtype=int)

    fdr_threshold = float(config.get("oligo_science_anchors", {}).get("fdr", 0.05))
    masked_zeros: list[str] = []
    n_observed_mismatch = 0
    n_sign_mismatch = 0
    n_tie = 0
    # The two panel-A support flags are READ from the frozen tables, never
    # recomputed -- but the explicit column is trusted verbatim, so it is the one
    # path where the script cannot see a wrong mark. These two lists hold cells
    # whose explicit flag is TRUE while the frozen rule it names is false, so the
    # contradiction is reported instead of drawn silently.
    n_explicit_positive = 0
    n_explicit_lower = 0
    flag_positive_contradictions: list[str] = []
    flag_lower_contradictions: list[str] = []
    # which of the two symbol classes came from an explicit column in the frozen
    # tables and which had to be re-formed from the raw columns
    origin = {"model_fdr_positive_supported": {"column": None, "derived": 0},
              "lower_tail_supported": {"column": None, "derived": 0}}

    for i, row in enumerate(rows):
        if not row.drawable:
            continue
        for j, sp in enumerate(species):
            record = cells.get((row.motif_id, sp))
            if record is None:
                continue
            key_model = CANDIDATES["model_log2_or"]
            key_raw = CANDIDATES["chromvar_C"]
            value_model = np.nan
            value_raw = np.nan
            for candidate in key_model:
                if candidate in record:
                    value_model = record[candidate]
                    break
            for candidate in key_raw:
                if candidate in record:
                    value_raw = record[candidate]
                    break
            value_model = float(value_model) if value_model is not None else np.nan
            value_raw = float(value_raw) if value_raw is not None else np.nan
            if np.isfinite(value_model) and not row.measurable:
                masked_zeros.append(f"{row.requested_name}/{row.motif_id}/{sp}")
                value_model = np.nan
            if np.isfinite(value_raw) and not row.measurable:
                value_raw = np.nan
            model[i, j] = value_model
            raw[i, j] = value_raw
            for candidate in CANDIDATES["model_fdr"]:
                if candidate in record:
                    model_fdr[i, j] = float(record[candidate])
                    break
            test = tests.get((row.motif_id, sp))
            if test is not None:
                for candidate in CANDIDATES["p_lower"]:
                    if candidate in test:
                        p_lower[i, j] = float(test[candidate])
                        break
            # enrichment support: the FDR in this project is one-sided for
            # enrichment, so positive support also needs a positive log2 OR
            explicit = _first_present(record, "model_fdr_positive_supported")
            if explicit is not None:
                supplied = _flag_series(pd.Series([record[explicit]]))[0]
                flag_positive[i, j] = supplied
                origin["model_fdr_positive_supported"]["column"] = explicit
                n_explicit_positive += 1
                if supplied and not (
                        np.isfinite(model_fdr[i, j]) and np.isfinite(value_model)
                        and model_fdr[i, j] < fdr_threshold and value_model > 0):
                    flag_positive_contradictions.append(
                        f"{row.requested_name}/{row.motif_id}/{sp} "
                        f"(log2 OR={value_model:.4g}, FDR={model_fdr[i, j]:.4g})")
            else:
                flag_positive[i, j] = bool(
                    np.isfinite(model_fdr[i, j]) and np.isfinite(value_model)
                    and model_fdr[i, j] < fdr_threshold and value_model > 0)
                origin["model_fdr_positive_supported"]["derived"] += 1
            # depletion support: never the enrichment FDR, only the lower tail of
            # the matched random draws
            explicit = _first_present(record, "lower_tail_supported")
            source = record if explicit is not None else None
            if explicit is None and test is not None:
                explicit = _first_present(test, "lower_tail_supported")
                source = test if explicit is not None else None
            if explicit is not None and source is not None:
                supplied = _flag_series(pd.Series([source[explicit]]))[0]
                flag_lower[i, j] = supplied
                origin["lower_tail_supported"]["column"] = explicit
                n_explicit_lower += 1
                if supplied and not (np.isfinite(p_lower[i, j])
                                     and p_lower[i, j] <= fdr_threshold):
                    flag_lower_contradictions.append(
                        f"{row.requested_name}/{row.motif_id}/{sp} "
                        f"(p_lower={p_lower[i, j]:.4g})")
            else:
                flag_lower[i, j] = bool(np.isfinite(p_lower[i, j])
                                        and p_lower[i, j] <= fdr_threshold)
                origin["lower_tail_supported"]["derived"] += 1

            if np.isfinite(value_model) and np.isfinite(value_raw):
                if value_model == 0 or value_raw == 0:
                    # no sign on one side: not concordant, but not a conflict
                    # either, so it gets its own glyph under the grey colour
                    concordance[i, j] = CODE_TIE
                    n_tie += 1
                elif value_model > 0 and value_raw > 0:
                    concordance[i, j] = CODE_POSITIVE
                elif value_model < 0 and value_raw < 0:
                    concordance[i, j] = CODE_NEGATIVE
                else:
                    concordance[i, j] = CODE_DISCORDANT

            # read-only integrity checks: a disagreement between two frozen
            # tables is surfaced here, never repaired silently
            if test is not None:
                observed_column = _first_present(test, "observed_log2_or")
                if observed_column is not None and np.isfinite(value_model):
                    observed = pd.to_numeric(test[observed_column], errors="coerce")
                    if np.isfinite(observed) and abs(observed - value_model) > 1e-9:
                        n_observed_mismatch += 1
            flag_column = _first_present(record, "sign_concordant")
            if (flag_column is not None and np.isfinite(value_model)
                    and np.isfinite(value_raw) and concordance[i, j] >= 0):
                theirs = _flag_series(pd.Series([record[flag_column]]))[0]
                mine = concordance[i, j] in (CODE_POSITIVE, CODE_NEGATIVE)
                if bool(theirs) != bool(mine):
                    n_sign_mismatch += 1

    if masked_zeros:
        C.log(f"WARNING: finite anchor scores were written for unmeasurable "
              f"matrices and are masked here ({len(masked_zeros)} cells, e.g. "
              f"{masked_zeros[:3]}); their scores are ABSENT, never zero.")
    if n_observed_mismatch:
        C.log(f"WARNING: {n_observed_mismatch} cell(s) where "
              f"{TESTS_FILE}'s observed value differs from {SCORES_FILE}'s model "
              f"log2 OR; both are drawn from the scores table, and the "
              f"disagreement is reported rather than reconciled.")
    if n_sign_mismatch:
        C.log(f"WARNING: {n_sign_mismatch} cell(s) where {SCORES_FILE}'s "
              f"sign_concordant disagrees with the sign pair drawn in panel C.")
    if flag_positive_contradictions:
        C.log(f"WARNING: {len(flag_positive_contradictions)}/{n_explicit_positive} "
              f"cell(s) carry model_fdr_positive_supported=TRUE while the frozen "
              f"rule printed in the legend (model FDR < {fdr_threshold:g} AND "
              f"log2 OR > 0) is false for the same cell. The explicit flag is "
              f"drawn as supplied -- it is never recomputed -- and the "
              f"contradiction is reported rather than reconciled. Cells: "
              f"{flag_positive_contradictions[:3]}")
    if flag_lower_contradictions:
        C.log(f"WARNING: {len(flag_lower_contradictions)}/{n_explicit_lower} "
              f"cell(s) carry lower_tail_supported=TRUE while the frozen rule "
              f"printed in the legend (matched-random lower tail p < "
              f"{fdr_threshold:g}) is false for the same cell. The explicit flag "
              f"is drawn as supplied -- it is never recomputed -- and the "
              f"contradiction is reported rather than reconciled. Cells: "
              f"{flag_lower_contradictions[:3]}")
    if n_tie:
        C.log(f"{n_tie} cell(s) carry a score of exactly zero, so no sign: drawn "
              f"under the discordant colour with the glyph '0' and counted as "
              f"non-concordant, matching the upstream empty sign_concordant.")

    for role, note in origin.items():
        if note["column"] is not None:
            C.log(f"panel A symbol '{role}': read from column {note['column']!r}"
                  + (f" for {note['derived']} cell(s) that lacked it, where the "
                     f"frozen rule was re-formed from the raw columns"
                     if note["derived"] else ""))
        else:
            C.log(f"panel A symbol '{role}': the frozen tables carry no explicit "
                  f"column; re-formed from the raw columns on the frozen rule "
                  f"({note['derived']} cell(s))")
    C.log(f"panel A symbols: {int(flag_positive.sum())} enrichment-supported, "
          f"{int(flag_lower.sum())} depletion-supported cells")

    counts, source_text = per_anchor_counts(rows, concordance, summary_frame, species)
    return {
        "model": model, "raw": raw, "model_fdr": model_fdr, "p_lower": p_lower,
        "concordance": concordance,
        "symbols": {"model_fdr_positive_supported": flag_positive,
                    "lower_tail_supported": flag_lower},
        "counts": counts, "counts_source": source_text, "origin": origin,
    }


def per_anchor_counts(rows, concordance, summary_frame, species):
    """Panel D's per-anchor numbers.

    The cross-species summary table is the intended source and is read column by
    column, so a summary that carries the concordance totals but not the
    direction split still contributes what it has. Whatever it does not carry is
    counted off the concordance matrix built from anchor_species_scores.tsv --
    that is counting category membership, i.e. exactly what the summary does, and
    never a new statistic.

    The bar and its n/N label are always taken from ONE source for a given row,
    so a row can never show a split that does not sum to its own N. Where the two
    sources disagree, the disagreement is logged rather than averaged away.
    """
    lookup: dict[str, dict] = {}
    cols: dict[str, str | None] = {}
    if summary_frame is not None and len(summary_frame):
        for role in ("n_evaluable", "n_concordant", "n_both_positive",
                     "n_both_negative", "n_discordant"):
            cols[role] = _optional_column(summary_frame, role)
        col_mid = _optional_column(summary_frame, "motif_id")
        col_name = _optional_column(summary_frame, "requested_name")
        for position in range(len(summary_frame)):
            record = summary_frame.iloc[position]
            values = {}
            for role, column in cols.items():
                if column is None:
                    continue
                value = pd.to_numeric(record[column], errors="coerce")
                # a NaN count is not a count of zero
                values[role] = int(value) if np.isfinite(value) else None
            if not any(value is not None for value in values.values()):
                continue
            mid = "" if col_mid is None else str(
                _text_series(summary_frame[col_mid])[position])
            name = "" if col_name is None else str(
                _text_series(summary_frame[col_name])[position])
            if mid:
                lookup[f"id::{mid}"] = values
            if name:
                lookup.setdefault(f"name::{name.upper()}", values)

    counts: dict[AnchorRow, dict] = {}
    n_from_summary = 0
    n_derived = 0
    n_mismatch = 0
    for i, row in enumerate(rows):
        codes = concordance[i]
        available = codes >= 0
        matrix = {
            "n_evaluable": int(available.sum()),
            "n_both_positive": int((codes == CODE_POSITIVE).sum()),
            "n_both_negative": int((codes == CODE_NEGATIVE).sum()),
            # a tie is non-concordant, exactly as upstream's empty
            # sign_concordant makes it
            "n_discordant": int(((codes == CODE_DISCORDANT)
                                 | (codes == CODE_TIE)).sum()),
        }
        matrix["n_concordant"] = matrix["n_both_positive"] + matrix["n_both_negative"]
        record = matrix
        summary = None
        if lookup:
            summary = lookup.get(f"id::{row.motif_id}")
            if summary is None and row.n_catalog_ids <= 1:
                # a name that resolves to two catalog matrices has no single
                # summary row, so only an unambiguous name may fall back to one
                summary = lookup.get(f"name::{row.requested_name.upper()}")
        if summary is not None:
            candidate = dict(summary)
            for role in ("n_both_positive", "n_both_negative", "n_discordant",
                         "n_concordant", "n_evaluable"):
                if candidate.get(role) is None:
                    candidate[role] = matrix[role]
            split = (candidate["n_both_positive"] + candidate["n_both_negative"]
                     + candidate["n_discordant"])
            if (split == candidate["n_evaluable"]
                    and candidate["n_concordant"]
                    == candidate["n_both_positive"] + candidate["n_both_negative"]):
                record = candidate
                n_from_summary += 1
                if (candidate["n_evaluable"] != matrix["n_evaluable"]
                        or candidate["n_concordant"] != matrix["n_concordant"]):
                    n_mismatch += 1
            else:
                n_mismatch += 1
        if record is matrix:
            n_derived += 1
        counts[row] = record

    if n_from_summary and not n_derived:
        source_text = f"read from {SUMMARY_FILE}"
    elif n_from_summary:
        source_text = (f"read from {SUMMARY_FILE} where it is internally "
                       f"consistent, counted from the concordance matrix for "
                       f"{n_derived} row(s)")
    else:
        source_text = (f"counted from the concordance matrix ({SCORES_FILE}); "
                       f"{SUMMARY_FILE} does not carry a usable count set")
    if n_mismatch:
        C.log(f"WARNING: {n_mismatch} anchor row(s) where {SUMMARY_FILE} and the "
              f"concordance matrix disagree on n/N or on the direction split; the "
              f"matrix counts are drawn for those rows. This is surfaced, never "
              f"averaged away.")
    C.log("panel D count source:", source_text)
    return counts, source_text


def wrapped(text: str, width: int = 215) -> str:
    """Hard-wrap the boundary paragraph to the figure width.

    WHY: `fig.text` does not wrap, so a long sentence is silently clipped at the
    page edge -- the exact failure mode that makes a boundary note unreadable
    while still looking present. Each \n-separated paragraph is wrapped on its
    own so the paragraph structure survives.
    """
    import textwrap

    return "\n".join(textwrap.fill(paragraph, width=width)
                     if paragraph.strip() else paragraph
                     for paragraph in text.split("\n"))


def footer_text(config, rows, species, cells, table_dir: Path) -> str:
    n_unmeasurable = sum(1 for r in rows if r.present and not r.measurable)
    n_absent = sum(1 for r in rows if not r.present)
    n_duplicate = sum(1 for r in rows if r.n_catalog_ids > 1)
    n_drawable = sum(1 for r in rows if r.drawable)
    repeats = int(config.get("oligo_science_anchors", {}).get("null_repeats", 200))
    groups = ", ".join(group for group, _ in declared_groups(config))
    cutoff = C.topk_label(config)
    return (
        f"Source (read only, nothing recomputed here): "
        f"{C.display_path(table_dir)}/"
        f"{{{MANIFEST_FILE.replace('.tsv', '')}, {SCORES_FILE.replace('.tsv', '')}, "
        f"{TESTS_FILE.replace('.tsv', '')}, {SUMMARY_FILE.replace('.tsv', '')}}}.tsv\n"
        f"Rows: one row per catalog matrix, grouped and ordered {groups} as "
        f"{C.display_path(config['_path'])} declares -- {len(rows)} rows, "
        f"{n_drawable} measurable, "
        f"{n_unmeasurable} unmeasurable, {n_absent} absent, {n_duplicate} carrying a "
        f"requested name that resolves to more than one catalog matrix. Every match is a "
        f"row, marked [i/k]; a name matching nothing is recorded absent and never "
        f"substituted by a similar motif.\n"
        f"Unmeasurable rows are hatched and labelled 'unmeasurable (no peak matches)': the "
        f"matrix matches no peak at the frozen FIMO per-site threshold, so its scores are "
        f"ABSENT (NaN) -- never zero -- and it enters no per-anchor aggregate.\n"
        f"A = model {cutoff} log2 OR; B = full-peak chromVAR C on its OWN colour scale. They "
        f"are different statistics on different peak universes, about an order of magnitude "
        f"apart, and are never compared by magnitude.\n"
        f"A symbols mean OPPOSITE things: '+' = model FDR < 0.05 with log2 OR > 0 (enrichment "
        f"supported); 'v' = matched-random lower tail, p < 0.05 (depletion supported). The "
        f"per-motif FDR is one-sided for ENRICHMENT, so a negative log2 OR is never "
        f"significant depletion on it -- the {repeats}-draw lower tail is the depletion "
        f"evidence.\n"
        f"C = sign concordance per anchor x species; D = its per-anchor summary (n/N, N "
        f"stated). A score of exactly zero has no sign: such a cell keeps the discordant "
        f"colour, carries the glyph 0, and counts as non-concordant, matching the upstream "
        f"empty sign_concordant. This plate is post-primary interpretation for the Oligo "
        f"arm: it gates nothing, enters no PASS/FAIL rule, and changes no claim."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(C.CONFIG_DIR / "analysis_v2.yaml"))
    parser.add_argument("--input", default=None,
                        help="directory holding the four anchor tables "
                             f"(default: this config's results dir for "
                             f"{STEP}/{SUBDIR})")
    # the sibling anchor steps take the directory as --outdir, so this step
    # accepts the same flag with the same meaning: read the tables from there and
    # default the figure prefix into it
    parser.add_argument("--outdir", default=None,
                        help="alias of --input, matching s17b/s17c")
    parser.add_argument("--manifest", default=None, help=MANIFEST_FILE)
    parser.add_argument("--scores", default=None, help=SCORES_FILE)
    parser.add_argument("--tests", default=None, help=TESTS_FILE)
    parser.add_argument("--summary", default=None, help=SUMMARY_FILE)
    parser.add_argument("--output-prefix", default=None,
                        help=f"default: <input>/{FIGURE_STEM}")
    parser.add_argument("--dpi", type=int, default=None)
    args = parser.parse_args()

    if args.outdir and args.input:
        raise SystemExit("--outdir and --input are the same directory; pass one")
    config = C.load_config(args.config)
    given = args.outdir or args.input
    in_dir = Path(given) if given else (C.results_dir(config, STEP) / SUBDIR)
    paths = {
        "manifest": Path(args.manifest) if args.manifest else in_dir / MANIFEST_FILE,
        "scores": Path(args.scores) if args.scores else in_dir / SCORES_FILE,
        "tests": Path(args.tests) if args.tests else in_dir / TESTS_FILE,
        "summary": Path(args.summary) if args.summary else in_dir / SUMMARY_FILE,
    }
    for role, path in paths.items():
        if not path.exists():
            # a missing input is named, never skipped
            raise FileNotFoundError(f"anchor {role} table not found: {path}")

    manifest = pd.read_csv(paths["manifest"], sep="\t")
    scores = pd.read_csv(paths["scores"], sep="\t")
    tests = pd.read_csv(paths["tests"], sep="\t")
    summary_frame = pd.read_csv(paths["summary"], sep="\t") \
        if paths["summary"].exists() else None

    rows = build_rows(config, manifest, paths["manifest"])
    species = species_axis(scores, _column(scores, "species", what=SCORES_FILE))
    missing_species = [s for s in C.SPECIES_ORDER if s not in species]
    if missing_species:
        # name the resolved file, not the module-level stem: SCORES_FILE is a
        # string (the tables' shared contract name), so `.name` does not exist on
        # it and would turn the one path meant to make a missing species
        # non-silent into an opaque AttributeError before anything is drawn.
        C.log("WARNING: no column drawn for species absent from "
              f"{paths['scores'].name}: {missing_species}")

    cells = _cell_index(scores, what=SCORES_FILE)
    test_cells = _cell_index(tests, what=TESTS_FILE)
    C.log(f"{len(rows)} anchor rows x {len(species)} species "
          f"({', '.join(species)}); "
          f"{sum(1 for r in rows if not r.drawable)} row(s) not evaluable")

    bundle = assemble(config, rows, species, cells, test_cells, summary_frame)
    limit_a = float(np.nanmax(np.abs(bundle["model"]))) or 1.0
    limit_b = float(np.nanmax(np.abs(bundle["raw"]))) or 1.0

    fig = plt.figure(figsize=(13.5, 16.3))
    gs = GridSpec(4, 1, figure=fig, height_ratios=[1.0, 1.0, 1.0, 0.92],
                  hspace=0.60)
    # The cut-off is the ARM's own, read from its config: this plate is drawn
    # once per arm, so a hardcoded "Top500" would caption the Top1000 and
    # Top2000 arms' figures with a cut-off those numbers did not come from.
    cutoff = C.topk_label(config)
    axes_a = panel_a(fig, gs[0], rows, species, bundle["model"],
                     bundle["symbols"], limit_a, cutoff)
    axes_b = panel_b(fig, gs[1], rows, species, bundle["raw"], limit_b)
    axes_c = panel_c(fig, gs[2], rows, species, bundle["concordance"])
    axes_d = panel_d(fig, gs[3], rows, species, bundle["counts"],
                     bundle["counts_source"])

    fig.subplots_adjust(left=0.28, right=0.955, top=0.900, bottom=0.195)
    fig.suptitle(
        "Oligo science anchors: the named cerebellar-GRN regulators, species by species",
        fontsize=13.5, color=INK, y=0.985)

    handles = [
        Line2D([], [], marker="+", linestyle="none", markersize=9,
               markeredgewidth=1.8, color=INK,
               label="model FDR < 0.05 and log2 OR > 0 (enrichment supported)"),
        Line2D([], [], marker="v", linestyle="none", markersize=7,
               markerfacecolor=INK, markeredgecolor="white",
               label="matched-random lower tail p < 0.05 (depletion supported)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.972),
               frameon=False, fontsize=9, ncol=2, handletextpad=0.5,
               columnspacing=2.6)
    fig.text(0.55, 0.952,
             "the two symbols mean opposite things and are never interchangeable",
             ha="center", va="top", fontsize=8.5, color=MUTED, style="italic")

    # the layout must be final before the group brackets are measured off the
    # rendered tick labels
    fig.canvas.draw()
    for band, band_axes in enumerate((axes_a, axes_b, axes_c, axes_d)):
        annotate_groups(fig, band_axes, rows, names=(band == 0))

    # D's key and the boundary text sit BELOW D's own labels, so their y is taken
    # from D's measured axes position rather than from a guessed constant
    d_bottom = axes_d.get_position().y0
    fig.legend(handles=[
        Patch(facecolor=POSITIVE, edgecolor="none", label="both positive"),
        Patch(facecolor=DISCORDANT, edgecolor="none", label="discordant"),
        Patch(facecolor=NEGATIVE, edgecolor="none", label="both negative"),
    ], loc="upper center", bbox_to_anchor=(0.5, d_bottom - 0.048),
        frameon=False, fontsize=8.5, ncol=3, handlelength=1.1,
        columnspacing=1.6)
    fig.text(0.5, d_bottom - 0.088,
             wrapped(footer_text(config, rows, species, cells, in_dir)),
             ha="center", va="top", fontsize=7.3, color=MUTED, linespacing=1.5)

    prefix = Path(args.output_prefix) if args.output_prefix else \
        in_dir / FIGURE_STEM
    dpi = int(args.dpi if args.dpi is not None
              else config.get("plotting", {}).get("dpi", 200))
    formats = tuple(config.get("plotting", {}).get("formats", ["png", "pdf"]))
    for path in S.save(fig, prefix, formats=formats, dpi=dpi, tight=False):
        C.log("wrote", C.display_path(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
