#!/usr/bin/env python3
"""Export the chromVAR inputs, applying this release's cell-type relabelling.

WHAT IS NEW HERE, AND WHAT IS NOT
---------------------------------
Exactly one thing is new: the `Org_Celltype` label of the human Carnegie stage
18-23 `astroglia` cells is changed to `progenitor` before the pseudobulk units
are built. Everything else -- reading the raw count matrix, streaming the
accumulation, writing Matrix Market, the read-back verification -- is the
vendored `s03_export_chromvar_inputs.py`, unmodified and hash-pinned.

WHY IT HAS TO BE A RELABEL AND NOT A DELETION
---------------------------------------------
The source matrices (`data/all_species_h5ad/<species>_deepcerevo_2026.h5ad`)
still carry the previous annotation: their `Org_Celltype` says `astroglia` for
those cells. This release's cell-level file says `progenitor`. If the two were
left to disagree, the chromVAR arm would build a "human early_embryonic
astroglia" pseudobulk out of cells that the rest of this repository calls
progenitors, and the validation step would be comparing a model defined on one
cell type against raw accessibility measured on another.

Relabelling rather than dropping keeps every cell in the table. The consequence
is that the human early_embryonic window simply has no astroglia unit -- the same
outcome the Astroglia arm gets by excluding that window's ranking file -- and a
`progenitor` unit appears that this release does not analyse. Both are visible in
`dropped_units.tsv` and `pseudobulk_metadata.tsv` rather than being invisible.

HOW THE INTERCEPTION WORKS
--------------------------
The vendored exporter builds units in `build_units(obs, config)`. This script
imports that module and replaces that one function with a version that relabels
`obs` first. The read-back check inside the vendored code re-derives pseudobulk
totals from the same cell -> unit mapping, so it still verifies the arithmetic
exactly; it does not re-derive the labels, and therefore cannot disagree with the
relabel.

Usage:
    s03_export_chromvar_inputs_relabelled.py --config config/arms/_shared.yaml
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
VENDOR_STEPS = PKG / "vendor" / "steps"

# The rule this release applies, spelled once. Keying on stage as well as species
# matters: `astroglia` also exists at human 11wpc and later, and those cells are
# NOT relabelled.
RELABEL_SPECIES = "human"
RELABEL_STAGES = ("CS18-19", "CS20", "CS22-23")
RELABEL_FROM = "astroglia"
RELABEL_TO = "progenitor"

sys.path.insert(0, str(VENDOR_STEPS))


def relabel(obs, *, record: list[dict] | None = None):
    """Return `obs` with the one label change applied.

    A copy is made only when something actually changes, so a species that is not
    affected is passed through untouched -- which keeps the vendored code on its
    normal path for five of the six species.
    """
    import numpy as np

    species = np.array([str(value).split("_")[0] for value in obs["Species"].to_numpy()]) \
        if "Species" in obs.columns else None
    if species is None:
        # obs carries the species in its cell ids when the column is absent;
        # rather than guess, skip the relabel and say so loudly.
        if record is not None:
            record.append({"applied": False,
                           "reason": "obs has no 'Species' column"})
        return obs

    mask = (
        (species == RELABEL_SPECIES)
        & np.isin(obs["Org_stage"].to_numpy(), list(RELABEL_STAGES))
        & (obs["Org_Celltype"].to_numpy() == RELABEL_FROM)
    )
    n = int(mask.sum())
    if record is not None:
        record.append({
            "applied": True,
            "species": RELABEL_SPECIES,
            "stages": "|".join(RELABEL_STAGES),
            "from": RELABEL_FROM,
            "to": RELABEL_TO,
            "n_cells_relabelled": n,
            "n_cells_total": int(len(obs)),
        })
    if n == 0:
        return obs
    out = obs.copy()
    column = out["Org_Celltype"].astype(object)
    column[mask] = RELABEL_TO
    out["Org_Celltype"] = column
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # The vendored module owns --config / --species / --block-cells / --skip-verify;
    # this wrapper adds nothing to the CLI so that it stays a drop-in replacement.
    parser.add_argument("--config", default=None)
    parser.add_argument("--relabel-report", type=Path, default=None)
    known, _ = parser.parse_known_args()

    export = importlib.import_module("s03_export_chromvar_inputs")
    C = importlib.import_module("common")

    original_build_units = export.build_units
    records: list[dict] = []

    def build_units_relabelled(obs, config):
        return original_build_units(relabel(obs, record=records), config)

    export.build_units = build_units_relabelled
    C.log(
        "relabel active:",
        f"{RELABEL_SPECIES} {'|'.join(RELABEL_STAGES)} '{RELABEL_FROM}' -> '{RELABEL_TO}'",
    )

    argv = [sys.argv[0]] + sys.argv[1:]
    sys.argv = argv
    export.main()

    # Write the relabel audit into the same _shared tree the export wrote to.
    import pandas as pd

    config = C.load_config(known.config)
    root = C.results_dir(config, export.STEP)
    frame = pd.DataFrame(records)
    if frame.empty:
        frame = pd.DataFrame([{"applied": False, "reason": "no species exported"}])
    target = known.relabel_report or (root.parent / "relabel_audit.tsv")
    frame.to_csv(target, sep="\t", index=False)
    C.log("wrote", target, f"({len(frame)} row(s))")
    # plain column access, not itertuples: `from` is a Python keyword and cannot
    # be read as an attribute
    for record in frame.to_dict(orient="records"):
        if record.get("applied"):
            print(f"  [RELABEL] {record['species']} {record['stages']}: "
                  f"{record['n_cells_relabelled']:,} / {record['n_cells_total']:,} cells "
                  f"'{record['from']}' -> '{record['to']}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
