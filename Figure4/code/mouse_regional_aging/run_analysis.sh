#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python="${PYTHON_BIN:-python}"
"$python" "$root/scripts/prepare.py"
"$python" "$root/scripts/run_mouse_analysis.py" --permutations 1000
"$python" "$root/scripts/full_pipeline_permutations.py"
"$python" "$root/scripts/plot_final.py"
"$python" "$root/scripts/export_subpanels.py"
"$python" "$root/scripts/verify.py"
