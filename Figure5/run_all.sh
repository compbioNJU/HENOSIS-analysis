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
  if [ ! -d "$ANALYSIS_DATA_DIR/Figure5" ]; then
    printf 'Missing data: %s/Figure5; set ANALYSIS_DATA_DIR.\n' "$ANALYSIS_DATA_DIR" >&2; exit 1
  fi
  python="$(command -v "$python")"
  export PYTHON_BIN="$python"
  export PATH="$(dirname "$python"):$PATH"
  "$python" -c 'import importlib; [importlib.import_module(x) for x in "numpy,pandas,scipy,h5py,yaml,anndata,scanpy,statsmodels,matplotlib,seaborn,fitz".split(",") if x]; print("Python dependencies: OK")'
fi
export PY="$python"
export SCATAC_PROJECT_ROOT="${SCATAC_PROJECT_ROOT:-$ANALYSIS_DATA_DIR/Figure5}"
export RSCRIPT="${RSCRIPT:-Rscript}" FIMO_BIN="${FIMO_BIN:-fimo}"
if [ "$dry_run" -eq 0 ]; then
  command -v "$RSCRIPT" >/dev/null
  command -v "$FIMO_BIN" >/dev/null
fi
run "$RSCRIPT" -e 'packages <- c("Biostrings","Matrix","S4Vectors","SummarizedExperiment","TFBSTools","chromVAR","motifmatchr"); missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly=TRUE)]; if (length(missing)) stop(paste("Missing R packages:",paste(missing,collapse=", "))); cat("R dependencies: OK\n")'
run "$FIMO_BIN" --version
if [ "$check_only" -eq 1 ]; then exit 0; fi
run bash code/run_all.sh 00 01 02
for stage in prepare configs audit cache shared lock top1000; do
  run bash code/src/04_peak_regulatory/run_all.sh "$stage"
done
printf '%s\n' 'Figure 5: all included analysis steps finished.'
