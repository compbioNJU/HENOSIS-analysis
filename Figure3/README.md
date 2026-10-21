# Figure 3 — 15-species heart analysis

Panels 3a and 3b use `phylogenetic_tree/run.py` and `umap/render.py`. Panels 3c–h use separate conservation, evolutionary representation, orthology and expression modules. `common/embedding.py` provides the shared loader used by the conservation and representation modules.

`code/scripts/run_all.sh` runs the included figure analyses in dependency order. Configure inputs in `code/config/inputs.yaml` and a writable output directory before execution. Tests and the separate heart annotation-transfer analysis are outside this main-figure package.
