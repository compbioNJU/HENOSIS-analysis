#!/usr/bin/env python3
"""TimeTree-derived heart reference tree, not an X_HENOSIS-inferred phylogeny."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from common.io import load_config, write_plot_csv
from common.names import SpeciesNames

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                            "svg.fonttype": "none", "pdf.use14corefonts": False})
GRAY = "#929292"


def leaf_names(node: dict) -> list[str]:
    if node["species"] is not None:
        return [node["species"]]
    return [sp for child in node["children"] for sp in leaf_names(child)]


def build_tree(ages: pd.DataFrame, species: list[str]) -> dict:
    """Validate complete pairwise ages and collapse equal-age nodes."""
    if len(species) < 2 or len(set(species)) != len(species):
        raise ValueError("Expected at least two unique species")
    required = {"species_a", "species_b", "age_mya"}
    if not required.issubset(ages):
        raise ValueError(f"Age table missing columns: {required - set(ages)}")
    rank = {sp: i for i, sp in enumerate(species)}
    pair_age = {}
    for row in ages.itertuples(index=False):
        a, b, age = str(row.species_a), str(row.species_b), float(row.age_mya)
        if a not in rank or b not in rank or a == b:
            raise ValueError(f"Unexpected species pair: {a}, {b}")
        if not np.isfinite(age) or age <= 0:
            raise ValueError(f"Invalid age: {a}, {b}, {age}")
        pair = frozenset((a, b))
        if pair in pair_age:
            raise ValueError(f"Duplicate pair: {a}, {b}")
        pair_age[pair] = age
    expected = {frozenset((a, b)) for a, b in combinations(species, 2)}
    if set(pair_age) != expected:
        missing = [tuple(sorted(p)) for p in expected - set(pair_age)]
        raise ValueError(f"Age pairs missing: {missing}")
    matrix = np.zeros((len(species), len(species)), dtype=float)
    for i, a in enumerate(species):
        for j in range(i + 1, len(species)):
            matrix[i, j] = matrix[j, i] = pair_age[frozenset((a, species[j]))]

    def convert(node):
        if node.is_leaf():
            return {"species": species[node.id], "age_mya": 0.0, "children": []}
        age = float(node.dist)
        children = [convert(node.left), convert(node.right)]
        flat = []
        for child in children:
            if child["species"] is None and np.isclose(
                    child["age_mya"], age, rtol=0, atol=1e-5):
                flat.extend(child["children"])
            else:
                flat.append(child)
        flat.sort(key=lambda ch: max(rank[sp] for sp in leaf_names(ch)))
        return {"species": None, "age_mya": age, "children": flat}

    return convert(to_tree(linkage(squareform(matrix), method="average")))


def assign_layout(root: dict, species: list[str]) -> None:
    y_by_species = {sp: float(i) for i, sp in enumerate(species)}
    next_id = 0

    def visit(node):
        nonlocal next_id
        node["id"] = f"n{next_id}"
        next_id += 1
        node["x"] = -node["age_mya"]
        if node["species"] is not None:
            node["y"] = y_by_species[node["species"]]
        else:
            for child in node["children"]:
                visit(child)
            node["y"] = (min(ch["y"] for ch in node["children"]) +
                         max(ch["y"] for ch in node["children"])) / 2

    visit(root)


def edge_rows(root: dict, palette: dict[str, str]) -> list[dict]:
    tips = leaf_names(root)
    if set(tips) != set(palette):
        raise ValueError(f"Palette mismatch: {set(tips) ^ set(palette)}")
    if "y" not in root:
        assign_layout(root, tips)
    rows = []

    def visit(parent):
        for child in parent["children"]:
            rows.append({
                "parent_id": parent["id"], "child_id": child["id"],
                "parent_age_mya": parent["age_mya"],
                "child_age_mya": child["age_mya"],
                "x_start": parent["x"], "x_end": child["x"],
                "y": child["y"], "child_species": child["species"] or "",
                "color": palette[child["species"]] if child["species"] else GRAY})
            visit(child)

    visit(root)
    return rows


def node_rows(root: dict) -> list[dict]:
    rows = []

    def visit(node):
        descendants = leaf_names(node)
        rows.append({"node_id": node["id"], "age_mya": node["age_mya"],
                     "x": node["x"], "y": node["y"],
                     "species": node["species"] or "",
                     "descendants": ";".join(descendants),
                     "n_descendants": len(descendants)})
        for child in node["children"]:
            visit(child)

    visit(root)
    return rows


def cophenetic_age(root: dict, a: str, b: str) -> float:
    for child in root["children"]:
        if {a, b}.issubset(set(leaf_names(child))):
            return cophenetic_age(child, a, b)
    return float(root["age_mya"])


def draw(root: dict, names: SpeciesNames, palette: dict[str, str],
         stem: Path, dpi: int = 300) -> None:
    species = names.order
    assign_layout(root, species)
    fig, ax = plt.subplots(figsize=(13.6, 9.0))
    fig.patch.set_facecolor("white")

    def visit(parent):
        if not parent["children"]:
            return
        low = min(ch["y"] for ch in parent["children"])
        high = max(ch["y"] for ch in parent["children"])
        ax.plot([parent["x"], parent["x"]], [low, high],
                color=GRAY, lw=1.75, solid_capstyle="round")
        for child in parent["children"]:
            color = palette[child["species"]] if child["species"] else GRAY
            ax.plot([parent["x"], child["x"]], [child["y"], child["y"]],
                    color=color, lw=2.35 if child["species"] else 1.75,
                    solid_capstyle="round")
            visit(child)

    visit(root)
    for i, sp in enumerate(species):
        ax.scatter([0], [i], s=55, color=palette[sp], zorder=5)
        ax.annotate(names.display(sp), (0, i), xytext=(11, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontstyle="italic", fontsize=9.4, color="#222222")
    max_age = float(root["age_mya"])
    step = 100 if max_age > 450 else 50
    ticks = np.arange(0, np.ceil(max_age / step) * step + 1, step)
    ax.set_xlim(-max_age * 1.04, max_age * .34)
    ax.set_ylim(-.7, len(species) - .3)
    ax.set_yticks([])
    ax.set_xticks(-ticks)
    ax.set_xticklabels([str(int(t)) for t in ticks])
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Million years ago (MYA)", fontsize=11, labelpad=10)
    ax.tick_params(axis="x", labelsize=8.5, length=3, color="#555555")
    for side in ("left", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["top"].set_color("#777777")
    ax.spines["top"].set_linewidth(.8)
    ax.text(.02, -.035, "TimeTree pairwise ages; average-linkage summary",
            transform=ax.transAxes, va="top", fontsize=8, color="#666666")
    fig.tight_layout(pad=1.5)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix("." + ext), dpi=dpi,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/phylogenetic_tree")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    config = load_config(ROOT / "config/inputs.yaml")
    names = SpeciesNames.from_csv(ROOT / "config/species_names.csv")
    palette_csv = pd.read_csv(ROOT / "config/species_colors.csv")
    palette = dict(zip(palette_csv.label, palette_csv.color))
    ages = pd.read_csv(config["pairwise_ages"])
    root = build_tree(ages, names.order)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "heart_15_species_time_tree"
    draw(root, names, palette, stem, args.dpi)
    write_plot_csv(out / "heart_15_species_time_tree_nodes.csv",
                   pd.DataFrame(node_rows(root)), ["node_id", "age_mya", "species"])
    write_plot_csv(stem.with_suffix(".csv"),
                   pd.DataFrame(edge_rows(root, palette)),
                   ["parent_id", "child_id", "child_species", "color"])
    tips = pd.DataFrame([
        {"species_id": sp, "full_name": names.display(sp),
         "color": palette[sp], "y": i}
        for i, sp in enumerate(names.order)])
    write_plot_csv(out / "heart_15_species_time_tree_tips.csv", tips,
                   ["species_id", "full_name", "color", "y"])
    age_fit = ages[["species_a", "species_b", "age_mya"]].copy()
    age_fit["tree_age_mya"] = [
        cophenetic_age(root, a, b)
        for a, b in zip(age_fit.species_a, age_fit.species_b)]
    age_fit["residual_mya"] = age_fit.tree_age_mya - age_fit.age_mya
    write_plot_csv(out / "heart_15_species_time_tree_age_fit.csv", age_fit,
                   ["species_a", "species_b", "age_mya", "tree_age_mya",
                    "residual_mya"])
    print(f"Saved {stem}.{{pdf,svg,png}}")
    print(f"Maximum absolute pair-age residual: "
          f"{age_fit.residual_mya.abs().max():.3f} MYA")


if __name__ == "__main__":
    main()
