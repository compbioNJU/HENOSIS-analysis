#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/phylogenetic_tree/run.py"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/umap/render.py" --condition 15S
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/expression/run.py"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/expression/plot_four_species_cardiomyocyte.py"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/expression/plot_all_species_by_celltype.py"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/conservation/run.py"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/evolutionary_representation/run.py"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/orthology_identity/run.py"
