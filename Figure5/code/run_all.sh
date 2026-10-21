#!/usr/bin/env bash
# Run every analysis in this repository, in dependency order.
#
#   ./run_all.sh              all four analyses, then the acceptance check
#   ./run_all.sh 00 01 02 03  a subset, in the order given
#   ./run_all.sh 04           only the peak-regulatory analysis
#   ./run_all.sh verify       only the acceptance check
#
# Each analysis also runs on its own; see the README in its src/ directory.
#
# A FAILING STEP STOPS THE RUN. Every analysis here either produces a numbered
# table or fails; there is no partial success worth continuing past, and the
# acceptance check at the end would report a missing figure anyway -- later, and
# with less information about why.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"
LOGS="$REPO/logs"
mkdir -p "$LOGS"

# Threads inside one worker. Without this the BLAS pool of a 48-core box is
# spawned per process and analysis 04's concurrent arms get slower, not faster.
THREADS="${THREADS:-8}"
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"

step() {  # step <label> <log-name> <command...>
  local label="$1" log="$2"; shift 2
  echo
  echo "==================== $label ===================="
  local started=$SECONDS
  if "$@" 2>&1 | tee "$LOGS/$log"; then
    echo "[OK] $label  ($((SECONDS - started))s)  -> logs/$log"
  else
    echo "[FAIL] $label  ($((SECONDS - started))s)  -> logs/$log" >&2
    exit 1
  fi
}

run_00() { step "00 input audit"      "00_audit.log"      "$PY" "$REPO/src/00_audit/audit_input.py"; }
run_01() { step "01 global UMAP"      "01_global_umap.log" "$PY" "$REPO/src/01_global_umap/plot_global_umap.py"; }
run_02() {
  step "02a astroglia stage UMAPs" "02a_astroglia_umap.log" \
    "$PY" "$REPO/src/02_astroglia_umap/plot_astroglia_stage_umap.py"
  step "02b cross-species alignment" "02b_astroglia_alignment.log" \
    "$PY" "$REPO/src/02_astroglia_umap/plot_cross_species_alignment.py"
}
run_03() { step "03 developmental direction" "03_developmental.log" \
             "$PY" "$REPO/src/03_developmental/run_developmental_direction.py"; }
run_04() { step "04 peak-regulatory" "04_peak_regulatory.out" \
             bash "$REPO/src/04_peak_regulatory/run_all.sh" all; }
run_verify() { step "verify results" "99_verify.log" \
                 "$PY" "$REPO/src/99_verify/verify_results.py"; }

targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
  targets=(00 01 02 03 04 verify)
fi

for target in "${targets[@]}"; do
  case "$target" in
    00|00_audit)      run_00 ;;
    01|01_global_umap) run_01 ;;
    02|02_astroglia_umap) run_02 ;;
    03|03_developmental) run_03 ;;
    04|04_peak_regulatory) run_04 ;;
    verify|99)        run_verify ;;
    *)
      echo "unknown step '$target'; expected one of: 00 01 02 03 04 verify" >&2
      exit 2
      ;;
  esac
done

echo
echo "all requested steps completed"
