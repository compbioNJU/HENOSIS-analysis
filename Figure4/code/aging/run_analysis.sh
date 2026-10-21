#!/usr/bin/env bash
set -euo pipefail

ANALYSIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

cd "${ANALYSIS_DIR}"

# Override at launch if needed, for example: NPERM=100 bash run_analysis.sh
# The manuscript analysis uses the default of 1,000 full-pipeline permutations.
export NPERM="${NPERM:-1000}"

echo "[1/2] Running full-pipeline donor-label permutation analysis (NPERM=${NPERM})..."
"${PYTHON_BIN}" scripts/test_aging_state_permutations.py

echo "[2/2] Running cross-fitted analysis and regenerating Figures 1–3..."
"${PYTHON_BIN}" scripts/analyze_crossfitted_aging_states.py

echo "Analysis complete. Results: ${ANALYSIS_DIR}/results"
