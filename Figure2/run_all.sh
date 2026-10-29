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
  if [ ! -d "$ANALYSIS_DATA_DIR/Figure2" ]; then
    printf 'Missing data: %s/Figure2; set ANALYSIS_DATA_DIR.\n' "$ANALYSIS_DATA_DIR" >&2; exit 1
  fi
  python="$(command -v "$python")"
  export PYTHON_BIN="$python"
  export PATH="$(dirname "$python"):$PATH"
  "$python" -c 'import importlib; [importlib.import_module(x) for x in "numpy,pandas,scipy,anndata,scanpy,matplotlib,sklearn".split(",") if x]; print("Python dependencies: OK")'
fi
if [ "$check_only" -eq 1 ]; then exit 0; fi
run "$python" -B code/microglia/plot_human_mouse_mgc_dotplot.py --output-dir "$ANALYSIS_RESULTS_DIR/Figure2/microglia"
run "$python" -B code/testis/evaluate_spermatogenesis_transfer.py
printf '%s\n' 'Figure 2: all included analysis steps finished.'
