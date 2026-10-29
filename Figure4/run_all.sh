#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="${PYTHON_BIN:-python}"
dry_run=0
check_only=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=1 ;;
    --check) check_only=1 ;;
    -h|--help) printf '%s\n' 'Usage: bash run_all.sh [--check | --dry-run]' 'PYTHON_BIN selects Python; ANALYSIS_DATA_DIR selects input data; ANALYSIS_RESULTS_DIR selects the output base.'; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$arg" >&2; exit 2 ;;
  esac
done
export ANALYSIS_DATA_DIR="${ANALYSIS_DATA_DIR:-$root/../../analysis_data}"
export ANALYSIS_RESULTS_DIR="${ANALYSIS_RESULTS_DIR:-$root/../results}"
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
run() {
  printf '+'; printf ' %q' "$@"; printf '\n'
  if [ "$dry_run" -eq 0 ]; then "$@"; fi
}
cd "$root"
if [ "$dry_run" -eq 0 ]; then
  if [ ! -d "$ANALYSIS_DATA_DIR/Figure4" ]; then
    printf 'Missing data: %s/Figure4; set ANALYSIS_DATA_DIR.\n' "$ANALYSIS_DATA_DIR" >&2; exit 1
  fi
  python="$(command -v "$python")"
  export PYTHON_BIN="$python"
  export PATH="$(dirname "$python"):$PATH"
  "$python" -c 'import importlib; [importlib.import_module(x) for x in "numpy,pandas,scipy,h5py,sklearn,anndata,scanpy,statsmodels,matplotlib,seaborn,plotly,kaleido,pyarrow,pypdf,PIL,openpyxl".split(",") if x]; print("Python dependencies: OK")'
fi
export FIGURE4_OUTPUT_DIR="${FIGURE4_OUTPUT_DIR:-$ANALYSIS_RESULTS_DIR/Figure4}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$FIGURE4_OUTPUT_DIR/matplotlib_cache}"
if [ "$check_only" -eq 1 ]; then exit 0; fi
run "$python" -B code/xenium/plot_leiden_celltype_sankey.py --min-flow-cells 250
run "$python" -B code/xenium/evaluate_leiden_celltype_consistency.py
run "$python" -B code/xenium/reproduce_transfer.py
run "$python" -B code/attention_aging/extract_opc_violin_source_data.py
run "$python" -B code/attention_aging/plot_figure4_i_j.py
run bash code/mouse_regional_aging/run_analysis.sh
printf '%s\n' 'Figure 4: all included analysis steps finished.'
