#!/usr/bin/env python3
"""Reproduce orthology-aware cardiomyocyte/endothelial identity validation.

Ranked top-100 genes and the raw scRNA count matrices are read independently.
Missing annotated orthology is unresolved; it never proves genome absence.
"""

from pathlib import Path
import re
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from common.io import load_config, write_plot_csv, results_root
from common.names import SpeciesNames

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
CELLTYPES = ("Cardiomyocyte", "Endothelial cell")
RANK_DIRS = {"Cardiomyocyte": "Cardiomyocyte",
             "Endothelial cell": "Endothelial_cell"}
ORTHO_COLORS = {"one_to_one": "#167a6e",
                "other_ortholog": "#d48632", "unresolved": "#9aa2aa"}


def top_n(ranked, n):
    if len(ranked) <= n:
        return ranked.gene.tolist()
    scores = ranked.score.to_numpy()
    if scores[n - 1] != scores[n]:
        return ranked.gene.iloc[:n].tolist()
    cutoff = scores[n - 1]
    above = int((scores > cutoff).sum())
    tied = np.flatnonzero(scores == cutoff)
    chosen = np.random.default_rng(20260928).choice(
        tied, size=n - above, replace=False)
    return ranked.gene.iloc[:above].tolist() + ranked.gene.iloc[np.sort(chosen)].tolist()


def select_recurrent(mapped, celltype, n=15):
    part = mapped[(mapped.celltype == celltype) & mapped.symbol.notna()]
    unique = part.groupby(["species", "symbol"], as_index=False).agg(
        rank=("rank", "min"))
    summary = unique.groupby("symbol", as_index=False).agg(
        n_species_top100=("species", "nunique"),
        mean_best_rank=("rank", "mean"))
    summary = summary.sort_values(
        ["n_species_top100", "mean_best_rank", "symbol"],
        ascending=[False, True, True]).head(n).copy()
    summary.insert(0, "celltype", celltype)
    summary.insert(1, "selection_rank", np.arange(1, len(summary) + 1))
    return summary.reset_index(drop=True)


def availability_state(relationship_types):
    calls = set(relationship_types)
    if "ortholog_one2one" in calls:
        return "one_to_one"
    if any(str(t).startswith("ortholog_") for t in calls):
        return "other_ortholog"
    return "unresolved"


def expression_state(raw_status, n_cells, fraction):
    if raw_status == "no_major_cells" or n_cells == 0:
        return "no_major_cells"
    if raw_status != "measured" or pd.isna(fraction):
        return "not_assessable"
    return "detected" if fraction > 0 else "not_detected"


def bubble_area(fraction):
    """Matplotlib marker area in points squared for raw-count detection rate."""
    return 20 + 150 * fraction


GENE_PREFIX = re.compile(r"^gene-", re.I)
JUNK_ID = re.compile(
    r"^(LOC\d|AC\d|AL\d|AP\d|ZGC|ENS[A-Z]*G\d|XLOC|MSTRG|TCONS"
    r"|CG\d|CR\d|FBgn|WBGene|AMEX|bj-|Afa|Afu|Pye|Cio|LOC\d)|\d{4,}", re.I)
TIER_ORDER = {"T2_one2one": 0, "T3_other": 1, "T1_name": 2}


def native_keys(name):
    original = str(name).strip()
    canonical = GENE_PREFIX.sub("", original)
    if ":" in canonical:
        canonical = canonical.split(":", 1)[1]
    return list(dict.fromkeys(
        [original, canonical, original.lower(), canonical.lower()]))


def build_bridge(relationships, species, genes):
    """Index explicit orthology calls, plus a separately marked name tier."""
    human = [c for c in relationships if "hsapiens" in c
             or c == "Homo_sapiens_gene_symbol"]
    symbol_col = next((c for c in human
                       if "symbol" in c or "external_gene_name" in c), None)
    type_col = next((c for c in relationships if "orthology_type" in c), None)
    if type_col is None:
        type_col = next((c for c in relationships if "homology_type" in c), None)
    native_cols = [c for c in relationships if c not in human
                   and c != "Orthogroup"
                   and not c.endswith(("gene_count", "orthology_type",
                                       "homology_type"))
                   and "_gene" in c]
    if symbol_col is None or type_col is None or not native_cols:
        raise ValueError(f"Cannot identify orthology columns for {species}")
    bridge = {}

    def add(native, symbol, tier):
        entry = (str(symbol).strip().upper(), tier)
        for key in native_keys(native):
            values = bridge.setdefault(key, [])
            if entry not in values:
                values.append(entry)

    for col in native_cols:
        for native, symbol, relation in zip(
                relationships[col], relationships[symbol_col],
                relationships[type_col]):
            if not isinstance(native, str) or not isinstance(symbol, str):
                continue
            if not native.strip() or not symbol.strip():
                continue
            tier = "T2_one2one" if relation == "ortholog_one2one" else "T3_other"
            add(native, symbol, tier)
    for gene in genes:
        bare = GENE_PREFIX.sub("", str(gene).strip())
        if bare and bare[0].isalpha() and not JUNK_ID.search(bare):
            add(gene, bare, "T1_name")
    return bridge


def lookup(bridge, gene):
    result, seen = [], set()
    for key in native_keys(gene):
        for item in bridge.get(key, []):
            if item not in seen:
                result.append(item)
                seen.add(item)
    return result


def best_accepted(calls, species):
    accepted = [c for c in calls
                if c[1] in {"T2_one2one", "T3_other"}
                or (species == "Homo_sapiens" and c[1] == "T1_name")]
    return min(accepted, key=lambda c: TIER_ORDER[c[1]]) if accepted else None


def expression_metrics(counts, library_sizes, mask):
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        return np.nan, np.nan
    values = np.asarray(counts, dtype=float)[selected]
    library = np.asarray(library_sizes, dtype=float)[selected]
    fraction = float(np.mean(values > 0))
    mean = float(np.mean(np.log1p(values * 1e4 / np.maximum(library, 1))))
    return fraction, mean


def relationship_index(frame, species, symbols):
    if species == "Homo_sapiens":
        sym_col, human_col, native_col = (
            "Homo_sapiens_gene_symbol", "Homo_sapiens_gene_id",
            "Homo_sapiens_source_gene_id")
    elif "hsapiens_external_gene_name" in frame:
        prefix = {"Mus_musculus": "mmusculus",
                  "Drosophila_melanogaster": "dmelanogaster",
                  "Taeniopygia_guttata": "tguttata"}[species]
        sym_col, human_col, native_col = (
            "hsapiens_external_gene_name", "hsapiens_ensembl_gene_id",
            f"{prefix}_homolog_ensembl_gene")
    else:
        sym_col, human_col, native_col = (
            "Homo_sapiens_gene_symbol", "Homo_sapiens_gene_id",
            f"{species}_gene_id")
    if not {sym_col, human_col, native_col}.issubset(frame):
        raise ValueError(f"Missing orthology columns for {species}")
    type_col = next(c for c in frame if c.endswith("orthology_type"))
    name_col = (native_col.replace("_ensembl_gene", "_associated_gene_name")
                if sym_col == "hsapiens_external_gene_name"
                else sym_col if species == "Homo_sapiens"
                else f"{species}_gene_name")

    def joined(group, col):
        return (";".join(sorted(set(str(x) for x in group[col] if str(x))))
                if col in group else "")

    return {
        symbol: {
            "relationship_types": joined(group, type_col),
            "human_gene_ids": joined(group, human_col),
            "annotated_native_gene_ids": joined(group, native_col),
            "annotated_native_gene_names": joined(group, name_col),
            "orthogroups": joined(group, "Orthogroup"),
        }
        for symbol, group in frame[frame[sym_col].isin(symbols)].groupby(sym_col)
    }


def audit_frame(raw, relationships_by_species):
    rows = []
    symbols = set(raw.symbol)
    for species, part in raw.groupby("species", sort=False):
        mapping = relationship_index(
            relationships_by_species[species], species, symbols)
        for original in part.to_dict("records"):
            row = original.copy()
            row.update(mapping.get(row["symbol"], {}))
            calls = row.get("relationship_types", "")
            row["availability_state"] = availability_state(
                calls.split(";") if calls else [])
            row["expression_state"] = expression_state(
                row["status"], row["n_major_cells"],
                row["fraction_expressing"])
            row["genome_absence_confirmed"] = False
            row["audit_note"] = (
                "No annotated human orthology call; genome absence not established"
                if row["availability_state"] == "unresolved" else
                "Annotated ortholog exists; raw matrix lacks matching measurable gene"
                if row["expression_state"] == "not_assessable" else
                "No comparable Celltype cells"
                if row["expression_state"] == "no_major_cells" else
                "Nonzero raw counts in sampled Celltype cells"
                if row["expression_state"] == "detected" else
                "Zero raw counts in sampled Celltype cells")
            rows.append(row)
    return pd.DataFrame(rows)


def raw_library_sizes(handle, n_cells, chunk_size=5_000_000):
    matrix = handle["X"]
    if matrix.attrs.get("encoding-type") != "csc_matrix":
        raise ValueError("Raw X must be CSC counts")
    total = np.zeros(n_cells, dtype=np.float64)
    indices, data = matrix["indices"], matrix["data"]
    for start in range(0, len(data), chunk_size):
        stop = min(start + chunk_size, len(data))
        total += np.bincount(indices[start:stop], weights=data[start:stop],
                             minlength=n_cells)
    return total


def csc_gene_sum(handle, genes, n_cells):
    result = np.zeros(n_cells, dtype=np.float64)
    pointers = handle["X/indptr"]
    cell_indices, values = handle["X/indices"], handle["X/data"]
    for gene in genes:
        start, stop = int(pointers[gene]), int(pointers[gene + 1])
        if stop > start:
            np.add.at(result, cell_indices[start:stop], values[start:stop])
    return result


def read_category(handle, key):
    node = handle[key]
    if isinstance(node, h5py.Group) and "categories" in node:
        categories = node["categories"].asstr()[:]
        codes = node["codes"][:]
        if (codes < 0).any():
            raise ValueError(f"Missing categorical value in {key}")
        return categories[codes]
    values = node[:]
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v)
                       for v in values], dtype=object)


def model_celltype_map(model_path):
    with h5py.File(model_path) as handle:
        labels = read_category(handle, "obs/cell_type")
        major = read_category(handle, "obs/Celltype")
    table = pd.DataFrame({"celltype": labels, "major": major}).drop_duplicates()
    if table.celltype.duplicated().any():
        raise ValueError("One cell_type maps to multiple Celltypes")
    return dict(zip(table.celltype, table.major))


def relationship_path(orthology_root, species):
    return orthology_root / f"{species}.Homo_sapiens.all_relationships.csv"


def load_relationships(orthology_root, species):
    return pd.read_csv(relationship_path(orthology_root, species),
                       dtype=str, keep_default_na=False)


def ranked_top100(species_order, ranked_root, orthology_root):
    rows = []
    for species in species_order:
        selected = {}
        for celltype in CELLTYPES:
            folder = RANK_DIRS[celltype]
            path = ranked_root / folder / f"heart_{folder}_{species}.csv"
            ranked = pd.read_csv(path)
            ranked["gene"] = ranked.gene.astype(str).str.strip()
            ranked = ranked.drop_duplicates("gene", keep="first").reset_index(drop=True)
            genes = top_n(ranked, 100)
            if len(genes) != 100:
                raise ValueError(f"Expected 100 ranked genes: {species}, {celltype}")
            selected[celltype] = genes
        rel = load_relationships(orthology_root, species)
        bridge = build_bridge(
            rel, species, [gene for genes in selected.values() for gene in genes])
        for celltype, genes in selected.items():
            for rank, gene in enumerate(genes, 1):
                call = best_accepted(lookup(bridge, gene), species)
                rows.append({
                    "species": species, "celltype": celltype, "rank": rank,
                    "gene": gene, "symbol": call[0] if call else None,
                    "mapping_tier": call[1] if call else "unmapped_or_name_only"})
    result = pd.DataFrame(rows)
    if result.duplicated(["species", "celltype", "rank"]).any():
        raise ValueError("Duplicate ranked position")
    return result


def native_gene_indices(bridge, genes, symbols, species):
    targets = set(symbols)
    found = {symbol: [] for symbol in symbols}
    for index, gene in enumerate(genes):
        for symbol, tier in lookup(bridge, gene):
            if symbol in targets and (tier in {"T2_one2one", "T3_other"}
                                      or (species == "Homo_sapiens"
                                          and tier == "T1_name")):
                found[symbol].append((index, tier))
    result = {}
    for symbol, options in found.items():
        if not options:
            result[symbol] = ([], "no_ortholog")
            continue
        best = min(TIER_ORDER[tier] for _, tier in options)
        kept = sorted({i for i, tier in options if TIER_ORDER[tier] == best})
        tier = next(tier for _, tier in options if TIER_ORDER[tier] == best)
        result[symbol] = (kept, tier)
    return result


def raw_path(raw_root, species, fallback_root=None):
    matches = sorted((raw_root / species / "preprocessed_data").glob("*.h5ad"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one raw h5ad link for {species}: {matches}")
    if matches[0].exists():
        return matches[0]
    if fallback_root is not None:
        alternate = fallback_root / matches[0].name
        if alternate.exists():
            return alternate
    raise FileNotFoundError(f"Unreadable raw h5ad link for {species}: {matches[0]}")


def species_raw_expression(species, selected, raw_root, celltype_map, relationships,
                           fallback_root=None):
    path = raw_path(raw_root, species, fallback_root)
    symbols = selected.symbol.tolist()
    with h5py.File(path) as handle:
        genes = read_category(handle, "var/_index")
        raw_labels = read_category(handle, "obs/celltype")
        major = np.asarray([celltype_map.get(v) for v in raw_labels], dtype=object)
        n_cells = len(major)
        if list(handle["X"].attrs["shape"]) != [n_cells, len(genes)]:
            raise ValueError(f"Raw X shape mismatch: {species}")
        library = raw_library_sizes(handle, n_cells)
        bridge = build_bridge(relationships, species, genes)
        native = native_gene_indices(bridge, genes, symbols, species)
        rows = []
        for selection in selected.itertuples():
            indices, tier = native[selection.symbol]
            mask = major == selection.celltype
            if mask.any() and indices:
                counts = csc_gene_sum(handle, indices, n_cells)
                fraction, mean = expression_metrics(counts, library, mask)
                status = "measured"
            elif not mask.any():
                status, fraction, mean = "no_major_cells", np.nan, np.nan
            else:
                status, fraction, mean = "no_ortholog", np.nan, np.nan
            rows.append({
                "species": species, "celltype": selection.celltype,
                "symbol": selection.symbol, "status": status,
                "n_major_cells": int(mask.sum()),
                "n_unmapped_raw_cells": int(sum(v is None for v in major)),
                "n_native_genes": len(indices),
                "native_genes": ";".join(genes[indices]),
                "mapping_tier": tier,
                "fraction_expressing": fraction,
                "mean_log1p_cp10k": mean,
                "raw_h5ad": str(path),
                "selection_rank": int(selection.selection_rank),
                "n_species_top100": int(selection.n_species_top100)})
    return pd.DataFrame(rows)


def plotting_table(audit, names):
    ordered_species = names.order
    rows = []
    for celltype in CELLTYPES:
        part = audit[audit.celltype.eq(celltype)]
        genes = (part[["symbol", "selection_rank"]].drop_duplicates()
                 .sort_values("selection_rank").symbol.tolist())
        if len(genes) != 15:
            raise ValueError(f"Expected 15 symbols for {celltype}")
        for record in part.to_dict("records"):
            state = record["expression_state"]
            record["x"] = ordered_species.index(record["species"])
            record["y"] = genes.index(record["symbol"])
            record["color"] = ORTHO_COLORS[record["availability_state"]]
            record["marker"] = ("o" if state in {"detected", "not_detected"}
                                else "s" if state == "no_major_cells" else "D")
            record["size"] = (bubble_area(record["fraction_expressing"])
                              if state == "detected" else
                              37 if state == "not_detected" else 38)
            rows.append(record)
    frame = pd.DataFrame(rows)
    expected = len(ordered_species) * len(CELLTYPES) * 15
    if len(frame) != expected or frame.duplicated(
            ["species", "celltype", "symbol"]).any():
        raise ValueError(f"Expected {expected} unique plotted cells")
    return frame


def draw(frame, names, stem):
    fig, axes = plt.subplots(1, 2, figsize=(16, 9.2))
    for ax, celltype in zip(axes, CELLTYPES):
        part = frame[frame.celltype.eq(celltype)]
        for row in part.itertuples():
            state = row.expression_state
            if state == "detected":
                ax.scatter(row.x, row.y, s=row.size, facecolor=row.color,
                           edgecolor="#303030", linewidth=.3)
            elif state == "not_detected":
                ax.scatter(row.x, row.y, s=row.size, facecolor="white",
                           edgecolor=row.color, linewidth=1)
            else:
                ax.scatter(row.x, row.y, s=row.size, marker=row.marker,
                           facecolor="none", edgecolor=row.color, linewidth=1)
        genes = (part[["symbol", "selection_rank"]].drop_duplicates()
                 .sort_values("selection_rank").symbol.tolist())
        ax.set_xlim(-.6, 14.6)
        ax.set_ylim(14.6, -.6)
        ax.set_xticks(range(15),
                      [names.display(sp, compact=True) for sp in names.order],
                      rotation=90, ha="center", fontsize=7)
        for tick in ax.get_xticklabels():
            tick.set_fontstyle("italic")
        ax.set_yticks(range(15), genes, fontsize=8)
        ax.set_title(celltype, fontsize=11)
        ax.grid(color="#ededed", lw=.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    orthology_handles = [
        Patch(facecolor=ORTHO_COLORS[state], edgecolor="none", label=label)
        for state, label in (
            ("one_to_one", "1:1 ortholog"),
            ("other_ortholog", "Non-1:1 ortholog"),
            ("unresolved", "No annotated ortholog (not evidence of genomic absence)"),
        )
    ]
    fig.legend(handles=orthology_handles, title="Human orthology annotation",
               loc="upper left", bbox_to_anchor=(.055, .955), ncol=3,
               frameon=False, fontsize=8, title_fontsize=8,
               handletextpad=.5, columnspacing=1.8)
    assessment_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor="#606060", markeredgecolor="#303030",
               markersize=8, label="Detected in target-cell-type cells"),
        Line2D([0], [0], marker="D", linestyle="none",
               markerfacecolor="none", markeredgecolor="#606060",
               markersize=8, label="Not quantifiable from raw counts"),
    ]
    fig.legend(handles=assessment_handles, title="Raw-count evidence",
               loc="upper left", bbox_to_anchor=(.055, .875), ncol=2,
               frameon=False, fontsize=8, title_fontsize=8,
               handletextpad=.5, columnspacing=1.8)
    size_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor="#606060", markeredgecolor="#303030",
               markersize=np.sqrt(bubble_area(fraction)), label=f"{int(fraction * 100)}%")
        for fraction in (.25, .5, .75, 1.)
    ]
    fig.legend(handles=size_handles,
               title="Cells with nonzero counts within target cell type",
               loc="upper left", bbox_to_anchor=(.055, .795), ncol=4,
               frameon=False, fontsize=8, title_fontsize=8,
               handletextpad=.5, columnspacing=1.8)
    fig.suptitle("Orthology-aware validation of cell-type identity",
                 y=.995, fontsize=14)
    fig.subplots_adjust(left=.055, right=.99, bottom=.22, top=.705, wspace=.33)
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix("." + ext), dpi=350,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    config = load_config(ROOT / "config/inputs.yaml")
    names = SpeciesNames.from_csv(ROOT / "config/species_names.csv")
    out = results_root(ROOT) / "orthology_aware_identity_validation"
    out.mkdir(parents=True, exist_ok=True)
    ranked = ranked_top100(
        names.order, config["ranked_root"], config["orthology_root"])
    ranked.to_csv(out / "top100_human_ortholog_mapping.csv", index=False)
    selected = pd.concat(
        [select_recurrent(ranked, celltype) for celltype in CELLTYPES],
        ignore_index=True)
    if selected.groupby("celltype").size().to_dict() != dict.fromkeys(CELLTYPES, 15):
        raise ValueError("Fewer than 15 recurrent symbols")
    selected.to_csv(out / "top15_recurrent_symbols.csv", index=False)
    celltype_map = model_celltype_map(config["xmot"])
    expression_parts, audit_parts = [], []
    for species in names.order:
        relationships = load_relationships(config["orthology_root"], species)
        expression = species_raw_expression(
            species, selected, config["raw_root"], celltype_map, relationships,
            config["raw_fallback_root"])
        expression_parts.append(expression)
        audited = audit_frame(expression, {species: relationships})
        audited["orthology_source"] = str(
            relationship_path(config["orthology_root"], species))
        audit_parts.append(audited)
        print(f"Processed raw expression and orthology: {species}", flush=True)
    raw_expression = pd.concat(expression_parts, ignore_index=True)
    raw_expression.to_csv(out / "top15_raw_expression.csv", index=False)
    audit = pd.concat(audit_parts, ignore_index=True)
    audit.to_csv(out / "gene_availability_audit.csv", index=False)
    plot = plotting_table(audit, names)
    stem = out / "orthology_aware_identity_validation"
    write_plot_csv(stem.with_suffix(".csv"), plot,
                   ["species", "celltype", "symbol", "availability_state",
                    "expression_state", "fraction_expressing", "x", "y",
                    "color", "marker", "size"])
    draw(plot, names, stem)
    print(f"Saved {stem}.{{png,pdf,csv}}; {len(plot)} plotted cells")


if __name__ == "__main__":
    main()
