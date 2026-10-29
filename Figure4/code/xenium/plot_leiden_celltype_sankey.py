"""Plot cell counts flowing from Leiden clusters to original cell types.

Run: python plot_leiden_celltype_sankey.py
"""

import argparse
from pathlib import Path
import os

import h5py
import pandas as pd
import plotly.graph_objects as go


HERE = Path(__file__).resolve()
DATA = Path(os.environ.get("ANALYSIS_DATA_DIR", HERE.parents[4] / "analysis_data")) / "Figure4/xenium/brain_integrated.h5ad"
OUT = Path(os.environ.get("FIGURE4_OUTPUT_DIR", HERE.parents[4] / "review_output/Figure4")) / "results/xenium"


# Frozen Nature palettes: embedded to avoid runtime color-file dependencies.
LEIDEN_COLORS = dict(zip(map(str, range(9)), [
    "#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1",
    "#76B7B2", "#EDC948", "#9C755F", "#BAB0AC"]))
CELLTYPE_COLORS = {
    "Astrocyte": "#59A14F", "OPC": "#A0CBE8", "Microglia": "#E15759",
    "Endothelial": "#76B7B2", "ExcNeu": "#4E79A7", "InhNeu": "#F28E2B",
    "Oligo": "#B07AA1", "Vascular/Stromal": "#EDC948", "nan": "#9C755F",
}
SPECIES_COLORS = {"Mus_musculus": "#1f77b4", "homo_sapiens": "#ff7d0b"}


def translucent(hex_color, alpha=0.48):
    r, g, b = (int(hex_color[i:i+2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def read_categorical(obs, key):
    group = obs[key]
    categories = group["categories"].asstr()[:]
    codes = group["codes"][:]
    return categories, codes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-flow-cells", type=int, choices=(250,), default=250,
                        help="Minimum cells per species-specific flow (fixed: 250)")
    args = parser.parse_args()
    min_flow_cells = args.min_flow_cells
    if min_flow_cells < 1:
        parser.error("--min-flow-cells must be at least 1")
    with h5py.File(DATA) as file:
        obs = file["obs"]
        leiden_names, leiden_codes = read_categorical(obs, "leiden")
        celltype_names, celltype_codes = read_categorical(obs, "Celltype")
        species_names, species_codes = read_categorical(obs, "sp")

    valid = (leiden_codes >= 0) & (celltype_codes >= 0)
    counts = pd.DataFrame({
        "leiden": leiden_codes[valid],
        "Celltype": celltype_codes[valid],
        "sp": species_codes[valid],
    }).value_counts(sort=False).reset_index(name="count")
    counts = counts.sort_values(["leiden", "Celltype", "sp"])
    visible = counts[counts["count"] >= min_flow_cells].copy()

    left = sorted(range(len(leiden_names)), key=lambda i: int(leiden_names[i]))
    right = list(range(len(celltype_names)))
    left_index = {code: i for i, code in enumerate(left)}
    right_index = {code: len(left) + i for i, code in enumerate(right)}
    labels = [f"Leiden {leiden_names[i]}" for i in left] + [str(celltype_names[i]) for i in right]

    # Reference notebook: colored nodes and cell counts as flow widths.
    colors = [LEIDEN_COLORS[str(leiden_names[i])] for i in left] + [CELLTYPE_COLORS[str(celltype_names[i])] for i in right]
    species_colors = {name: translucent(color) for name, color in SPECIES_COLORS.items()}
    link_colors = [species_colors[str(species_names[i])] for i in visible["sp"]]
    figure = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(pad=12, thickness=16, label=labels, color=colors,
                  line=dict(color="rgba(0,0,0,0.25)", width=0.3)),
        link=dict(source=visible["leiden"].map(left_index).tolist(),
                  target=visible["Celltype"].map(right_index).tolist(),
                  value=visible["count"].tolist(),
                  color=link_colors,
                  customdata=[str(species_names[i]) for i in visible["sp"]],
                  hovertemplate="%{source.label} → %{target.label}<br>Species: %{customdata}<br>Cells: %{value}<extra></extra>"),
    ))
    figure.update_layout(title=f"Leiden → Celltype (≥{min_flow_cells} cells per flow)", width=1100, height=850,
                         font=dict(size=12), paper_bgcolor="white",
                         margin=dict(l=20, r=20, t=60, b=20))
    for x, species in [(0.36, "Mus_musculus"), (0.61, "homo_sapiens")]:
        figure.add_annotation(x=x, y=1.06, xref="paper", yref="paper",
                              text=f"● {species}", showarrow=False,
                              font=dict(size=13, color=SPECIES_COLORS[species]))

    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"sankey_leiden_to_Celltype_min{min_flow_cells}"
    figure.write_html(OUT / f"{stem}.html", include_plotlyjs=True)
    figure.write_image(OUT / f"{stem}.svg")
    figure.write_image(OUT / f"{stem}.pdf")
    visible.assign(leiden=visible["leiden"].map(lambda i: leiden_names[i]),
                   Celltype=visible["Celltype"].map(lambda i: celltype_names[i]),
                   sp=visible["sp"].map(lambda i: species_names[i])) \
        .to_csv(OUT / f"{stem}_counts.csv", index=False)
    print(f"Saved {stem} SVG, HTML and counts in {OUT}")
    print(f"Visible: {int(visible['count'].sum())} cells, {len(visible)} flows; "
          f"filtered: {int(counts['count'].sum() - visible['count'].sum())} cells, "
          f"{len(counts) - len(visible)} flows")


if __name__ == "__main__":
    main()
