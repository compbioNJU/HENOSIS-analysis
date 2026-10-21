#!/usr/bin/env python3
"""Draw the sequence maps with the tracks ordered by true rank, top to bottom.

WHAT IS NEW HERE, AND WHAT IS NOT
---------------------------------
Exactly one thing is new: the vertical order of the tracks in the figure. The
selection rule, the ranks themselves, the motif hits, the colours, the lane
packing, the labels and the disclaimers are all the vendored
`s18_sequence_maps.py`, unmodified and hash-pinned.

WHY THE ORDER NEEDED CHANGING
-----------------------------
The vendored figure orders its tracks by the module map's order of first
appearance in `config/celltype_motif_module_map.tsv` -- for Astroglia that is
SOX, NFI, RFX, NF-Y -- and `draw_model_map` places row `index` at
`y = index * row_height` on an axis with `ylim = (0, n_peaks * row_height)` and
no inversion. So the first row of the frame lands at the BOTTOM.

The result, for `Astroglia__human__15-17wpc_11wpc`, was top-to-bottom:

    NF-Y (rank 1), RFX (rank 4), NFI (rank 2), SOX (rank 7)

which reads as if the ranks were arbitrary. They are not: each is the smallest
rank inside the Top-50 window whose peak carries a motif of that module, and the
gaps are real -- rank 3 carries NF-Y and NFI, both already claimed by earlier
peaks, so it is used by no module.

WHAT IS *NOT* CHANGED, AND MUST NOT BE
--------------------------------------
**The rank values are not renumbered.** Rank 1, 2, 4 and 7 are the model's real
global peak ranks, and the label on each track still prints the rank it actually
has. Renumbering them to 1-2-3-4 would tell the reader that these four peaks are
the model's top four, which is false -- ranks 3, 5 and 6 exist, carry other
motifs, and were not selected. Only the drawing order changes, so that the eye
can read the ranks in ascending order instead of having to hunt for them.

THE MECHANISM
-------------
Sorting the frame by rank DESCENDING is what produces an ascending order on
screen, because index 0 is drawn at the bottom. Ascending sort would keep the
confusing order. That inversion is the whole trick and is why this wrapper cannot
be written as an obvious `sort_values("rank")`.

The vendored `draw_model_map` reads `peaks` in only two places -- its length, and
the enumerate that draws the tracks -- and looks hits up by peak id, so
reordering the frame is safe and cannot change which hits are drawn.

The patched function is a module attribute, so the self-test in the vendored
`main()` is covered by the same patch rather than silently exercising the old
order.

Usage:
    s18_sequence_maps_rank_ordered.py --config config/arms/_shared.yaml
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
VENDOR_STEPS = PKG / "vendor" / "steps"

sys.path.insert(0, str(VENDOR_STEPS))

RANK_COLUMN = "rank"


def reorder(peaks):
    """The same rows, sorted so that they PRINT in ascending rank order.

    Descending here, because `draw_model_map` draws index 0 at the bottom of a
    non-inverted axis. A missing or non-numeric rank would reorder on a value the
    figure then misreports, so it is an error rather than a silent fallback.
    """
    if peaks is None or len(peaks) == 0:
        return peaks
    if RANK_COLUMN not in peaks.columns:
        raise KeyError(
            f"the frame handed to draw_model_map has no '{RANK_COLUMN}' column "
            f"(columns: {list(peaks.columns)}); the order cannot be set safely"
        )
    ordered = peaks.sort_values(RANK_COLUMN, ascending=False, kind="mergesort")
    return ordered.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # the vendored module owns the whole CLI; this wrapper adds nothing, so it
    # stays a drop-in replacement
    parser.add_argument("--config", default=None)
    known, _ = parser.parse_known_args()

    sequence_maps = importlib.import_module("s18_sequence_maps")
    C = importlib.import_module("common")

    original = sequence_maps.draw_model_map
    records: list[dict] = []

    def draw_model_map_rank_ordered(model_row, peaks, drawn, n_lanes, colours):
        ordered = reorder(peaks)
        if ordered is not None and len(ordered):
            # The audit must describe what a reader SEES, so it is taken from the
            # drawn frame reversed, not from an independent ascending sort. The
            # two differ whenever a model has two tracks at the same rank: the
            # drawn order is `sort_values(rank, descending)` and a stable sort
            # keeps the incoming frame order among equal ranks, so on screen
            # those ties appear reversed. Several models do have ties -- one peak
            # can be the best-scoring peak of more than one module, and then it is
            # drawn once per module.
            visual = ordered.iloc[::-1]
            records.append({
                "model": model_row.get("model", ""),
                "window": model_row.get("window", ""),
                "n_tracks": int(len(ordered)),
                "n_distinct_peaks": int(ordered["peak"].nunique()),
                "row_order_top_to_bottom": "|".join(
                    f"{m}:rank{int(r)}" for m, r in zip(visual["module"], visual[RANK_COLUMN])
                ),
            })
        return original(model_row, ordered, drawn, n_lanes, colours)

    sequence_maps.draw_model_map = draw_model_map_rank_ordered
    C.log("row order: tracks drawn by rank ASCENDING top-to-bottom "
          "(frame sorted descending, because index 0 is drawn at the bottom)")

    sys.argv = [sys.argv[0]] + sys.argv[1:]
    sequence_maps.main()

    import pandas as pd

    if "--self-test" in sys.argv[1:]:
        # A smoke test draws a hand-built demo model. It must not write the
        # row-order audit at all: the audit describes the real figures, and a
        # self-test run would otherwise replace a 20-row audit with an empty one.
        C.log("row-order audit: not written (--self-test run draws no real figures)")
        return 0

    config = C.load_config(known.config)
    out = C.results_dir(config, sequence_maps.STEP)
    frame = pd.DataFrame(records)

    # Keep only the models this run actually rendered, as listed by the manifest
    # the vendored step just wrote. Without this the `--self-test` path -- which
    # draws a hand-built demo model named "M" -- leaves its row in the audit, and
    # a self-test run then replaces the real 20-row audit with that single row.
    # The manifest is the authority on which figures exist, so filter by it rather
    # than by guessing at model-name shapes.
    manifest_path = out / "sequence_maps_manifest.tsv"
    if frame.empty:
        frame = pd.DataFrame([{"model": "", "window": "", "row_order_top_to_bottom": ""}])
    elif manifest_path.exists():
        rendered = set(pd.read_csv(manifest_path, sep="\t")["model"].astype(str))
        dropped = sorted(set(frame["model"]) - rendered)
        frame = frame[frame["model"].isin(rendered)].reset_index(drop=True)
        if dropped:
            C.log(f"row-order audit: excluded {len(dropped)} non-rendered model(s) "
                  f"drawn by the self-test: {dropped}")

    target = out / "row_order_audit.tsv"
    frame.to_csv(target, sep="\t", index=False)
    C.log("wrote", target, f"({len(frame)} figure(s))")
    for record in frame.head(4).to_dict(orient="records"):
        print(f"  [ORDER] {record['model']}  top->bottom: "
              f"{record['row_order_top_to_bottom']}")
    print("  [NOTE] ranks are printed as they really are -- these are model peak "
          "ranks inside the Top-K window, with gaps; they are not renumbered to 1..n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
