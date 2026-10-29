#!/usr/bin/env python3
"""Reproduce Figure 6f–i from the adjacent analysis_data package.

Only the unsupervised UMAP, M1donor/M1mac3 ASC gene–peak analysis, and ODC
ranked-gene dotplot are run. Results go to ANALYSIS_RESULTS_DIR or the package's
results directory. No rank-depth or top-peak subset analysis is performed.
"""

import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("ANALYSIS_DATA_DIR", HERE.parents[3] / "analysis_data")).expanduser().resolve()




def main() -> None:
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(DATA_ROOT)
    env = os.environ.copy()
    env["ANALYSIS_DATA_DIR"] = str(DATA_ROOT)
    steps = [
        ("6f", HERE / "plot_unsupervised_umap.py"),
        ("6h mapping", HERE / "map_human_macaque_orthologs.ipynb"),
        ("6g–h", HERE / "analyze_macaque_gene_peak_accessibility.ipynb"),
        ("6i", HERE / "plot_odc_ranked_gene_expression.ipynb"),
    ]
    for panel, path in steps:
        print(f"Running {panel}: {path.name}", flush=True)
        command = [sys.executable, str(path)] if path.suffix == ".py" else [
            sys.executable, str(HERE.parent / "run_notebook.py"), str(path)
        ]
        subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
