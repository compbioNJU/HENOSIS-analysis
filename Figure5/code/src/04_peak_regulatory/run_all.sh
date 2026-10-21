#!/usr/bin/env bash
# Driver for the cross-species peak-regulatory analysis on the new cell types.
#
#   ./run_all.sh prepare    wire the vendored core's roots; seed its FIMO cache
#   ./run_all.sh configs    derive the four arm configs; assert the declared diff
#   ./run_all.sh audit      verify the vendor hashes, the window set, the relabel
#   ./run_all.sh cache      warm the FIMO/sequence cache if it is cold
#   ./run_all.sh shared     03_chromvar_input (relabelled) + 04_chromvar
#                           + 13_rank_curve + 16_sequence_maps, into _shared
#   ./run_all.sh lock       freeze _shared and publish it into each arm
#   ./run_all.sh top500     one arm, end to end        (PRIMARY)
#   ./run_all.sh top1000    one arm, end to end        (robustness)
#   ./run_all.sh top2000    one arm, end to end        (robustness)
#   ./run_all.sh arms       the three arms only (after shared + lock)
#   ./run_all.sh compare    reproducibility control against the previous
#                           release (PREVIOUS_RELEASE=/path/to/result_v3)
#   ./run_all.sh all        prepare -> configs -> audit -> cache -> shared
#                           -> lock -> three arms
#
# ORDERING IS NOT ARBITRARY
#   `lock` must follow `shared`, because s18 writes `<species>_motif_positions.csv.gz`
#   INTO the cache and would fail against a tree made read-only first. `lock` must
#   precede the arms, because a cold scan writes scratch chunks into
#   `cache/fimo_<species>/` and two arms scanning the same species at once would
#   clobber each other. After `lock` the arms only read the cache.
#
# THE `--config` ARGUMENT IS LOAD-BEARING. The vendored `common.add_common_args`
# defaults `--config` to `config/analysis.yaml`, which is the PREVIOUS release's
# v1 config pointing at the abandoned v1 results tree. Every step below passes
# `--config` explicitly. Do not hand-run a vendored step without it.
#
# WHAT IS VENDORED AND WHY: see PROVENANCE.md.
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"
RSCRIPT="${RSCRIPT:-Rscript}"
ARMS_CFG="$BASE/config/arms"
VENDOR_STEPS="$BASE/vendor/steps"
# The results tree: `vendor/result_v4/`, i.e. INSIDE the vendored project root but
# BESIDE its `results/` directory. That is the only location both vendored guards
# accept -- `s06_purkinje_model.py` requires its output inside the project root,
# `s17_motif_forest.py` and `s18_sequence_maps.py` refuse anything under
# `results/`. See script/make_configs.py for the two layouts that failed first.
# The repository also exposes this at `results/04_peak_regulatory` (a symlink) for
# readers, created by script/prepare.py.
RESULTS_DIR="$BASE/vendor/result_v4"
LOGS="$RESULTS_DIR/logs"
JOBS="${JOBS:-3}"

ARMS=(500 1000 2000)
SHARED_STEPS=(03_chromvar_input 04_chromvar 13_rank_curve 16_sequence_maps)

# EVERY PDF MUST KEEP EDITABLE TEXT. matplotlib's default pdf.fonttype is 3
# (Type-3 glyph procedures), which is not selectable or editable in Illustrator,
# Inkscape or Acrobat. The vendored steps are byte-frozen and cannot be edited to
# fix it, so the setting is injected through the environment instead -- this is
# the one knob that changes their figures without touching a byte of their code.
# See config/matplotlibrc.
export MATPLOTLIBRC="$BASE/config/matplotlibrc"
if [ ! -f "$MATPLOTLIBRC" ]; then
  echo "refusing to run: $MATPLOTLIBRC is missing, so every PDF would lose its" >&2
  echo "editable text. Restore it, or set MATPLOTLIBRC yourself." >&2
  exit 1
fi
echo "MATPLOTLIBRC=$MATPLOTLIBRC (pdf.fonttype=42, ps.fonttype=42)"

mkdir -p "$RESULTS_DIR" "$LOGS"
cd "$BASE"

py() {  # py <step> <arm-config> <tag> [extra args...]
  local step="$1" cfg="$2" tag="$3"; shift 3
  echo "=== $step [$tag] ==="
  "$PY" "$VENDOR_STEPS/$step" --config "$cfg" "$@" 2>&1 | tee "$LOGS/${step%.*}_${tag}.log"
}

pyc() {  # pyc <step> <celltype> <arm-config> <tag>
  local step="$1" celltype="$2" cfg="$3" tag="$4"; shift 4
  echo "=== $step --celltype $celltype [$tag] ==="
  "$PY" "$VENDOR_STEPS/$step" --config "$cfg" --celltype "$celltype" "$@" \
    2>&1 | tee "$LOGS/${step%.*}_${celltype}_${tag}.log"
}

stage_prepare() { "$PY" script/prepare.py; }

stage_configs() { "$PY" script/make_configs.py; }

stage_audit() { "$PY" script/release_audit.py --config "$ARMS_CFG/top500.yaml"; }

stage_cache() {
  py warm_scan_cache.py "$ARMS_CFG/_shared.yaml" shared --workers 6
}

stage_shared() {
  local cfg="$ARMS_CFG/_shared.yaml"
  # The ONE step whose behaviour this release changes: the cell-type relabelling
  # is applied before pseudobulk units are built. It replaces
  # `s03_export_chromvar_inputs.py` and is otherwise identical to it.
  "$PY" script/s03_export_chromvar_inputs_relabelled.py --config "$cfg" \
    2>&1 | tee "$LOGS/s03_export_chromvar_inputs_relabelled_shared.log"
  r_or_py s04_run_chromvar.R "$cfg" shared
  py s15_rank_curve.py "$cfg" shared
  # The sequence maps are drawn by a wrapper, not by the vendored step directly:
  # it changes ONLY the vertical order of the tracks (by true rank, ascending
  # top-to-bottom), leaving the ranks themselves untouched. See that script's
  # docstring for why the frame is sorted descending to achieve that.
  "$PY" script/s18_sequence_maps_rank_ordered.py --config "$cfg" \
    2>&1 | tee "$LOGS/s18_sequence_maps_rank_ordered_shared.log"
}

r_or_py() {
  local step="$1" cfg="$2" tag="$3"
  echo "=== $step [$tag] ==="
  if [[ "$step" == *.R ]]; then
    "$RSCRIPT" "$VENDOR_STEPS/$step" --config "$cfg" 2>&1 | tee "$LOGS/${step%.*}_${tag}.log"
  else
    "$PY" "$VENDOR_STEPS/$step" --config "$cfg" 2>&1 | tee "$LOGS/${step%.*}_${tag}.log"
  fi
}

stage_lock() {
  echo "=== publish shared into each arm ==="
  for topk in "${ARMS[@]}"; do
    mkdir -p "$RESULTS_DIR/top$topk"
    for step in "${SHARED_STEPS[@]}"; do
      target="$RESULTS_DIR/_shared/$step"
      if [ ! -d "$target" ]; then
        echo "refusing to link missing shared step $target" >&2
        exit 1
      fi
      ln -sfn "../_shared/$step" "$RESULTS_DIR/top$topk/$step"
    done
    echo "top$topk: linked ${SHARED_STEPS[*]}"
  done
  # a checksum of the frozen shared tree, so a write THROUGH a symlink after this
  # point cannot pass unnoticed
  ( cd "$RESULTS_DIR/_shared" && find . -type f | sed 's|^\./||' | LC_ALL=C sort \
      | xargs -r sha256sum ) > "$RESULTS_DIR/_shared.sha256"
  echo "recorded _shared.sha256 ($(wc -l < "$RESULTS_DIR/_shared.sha256") files)"
}

stage_arm() {
  local topk="$1"
  local cfg="$ARMS_CFG/top$topk.yaml"
  local tag="top$topk"

  echo "########## ARM top$topk ##########"
  py s00_audit_inputs.py            "$cfg" "$tag"
  py s01_model_motif_enrichment.py  "$cfg" "$tag"
  py audit_astroglia_stage_trace.py "$cfg" "$tag"
  py s02_astroglia_model.py         "$cfg" "$tag"
  py s05_astroglia_raw_validation.py "$cfg" "$tag"
  py s05_prep_family_trajectory.py  "$cfg" "$tag"
  py s05_plot_family_trajectory_standardized.py "$cfg" "$tag"
  py s05_plot_family_trajectory_absolute.py     "$cfg" "$tag"
  py s05_plot_family_trajectory_agreement.py    "$cfg" "$tag"
  # s06_purkinje_model and s08_oligo_model are NOT run: their output directories
  # (06_purkinje_model/, 08_oligo_model/) are not part of the paper's deliverable
  # set, so the standard run does not produce them. Both scripts are kept under
  # vendor/steps for readers who want to regenerate them.
  #
  # Both are OPTIONAL upstream tables for the steps that follow, and every
  # consumer already declares what its absence costs:
  #   * s07_purkinje_raw_validation (reads 06_purkinje_model/): records
  #     `recurrent_modules_missing` and blocks the module-layer reading only;
  #     gate P3 never consults the upstream table.
  #   * s09_oligo_raw_validation (reads 08_oligo_model/recurrent_families.tsv):
  #     records `recurrent_families_missing` and marks gate O3 `blocked`. The
  #     oligodendrocyte contrast that Figure 5e reports (model enrichment vs
  #     chromVAR, and the module support) does not read that table.
  # Run `py s06_purkinje_model.py "$cfg" "$tag"` / `py s08_oligo_model.py "$cfg" "$tag"`
  # before the consuming step to restore them.
  py s07_purkinje_raw_validation.py "$cfg" "$tag"
  pyc celltype_validation_prep.py Purkinje "$cfg" "$tag"
  pyc plot_celltype_validation.py Purkinje "$cfg" "$tag"
  py s09_oligo_raw_validation.py    "$cfg" "$tag"
  pyc celltype_validation_prep.py Oligo "$cfg" "$tag"
  pyc plot_celltype_validation.py Oligo "$cfg" "$tag"
  py s17_motif_forest.py            "$cfg" "$tag"
  py s17b_audit_oligo_science_anchors.py   "$cfg" "$tag"
  py s17c_oligo_science_anchor_analysis.py "$cfg" "$tag"
  py s17d_plot_oligo_science_anchors.py    "$cfg" "$tag"
  echo "########## ARM top$topk done ##########"
}

stage_arms() {
  local pids=() topk
  if [ "$JOBS" -le 1 ]; then
    for topk in "${ARMS[@]}"; do stage_arm "$topk"; done
    return
  fi
  for topk in "${ARMS[@]}"; do
    stage_arm "$topk" > "$LOGS/arm_top${topk}.log" 2>&1 &
    pids+=("$!")
    echo "launched arm top$topk (pid ${pids[-1]}) -> $LOGS/arm_top${topk}.log"
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if [ "$failed" -ne 0 ]; then
    echo "at least one arm failed; see $LOGS/arm_top*.log" >&2
    exit 1
  fi
  echo "all three arms completed"
}

case "${1:-all}" in
  prepare) stage_prepare ;;
  configs) stage_configs ;;
  audit)   stage_audit ;;
  cache)   stage_cache ;;
  shared)  stage_shared ;;
  lock)    stage_lock ;;
  top500)  stage_arm 500 ;;
  top1000) stage_arm 1000 ;;
  top2000) stage_arm 2000 ;;
  arms)    stage_arms ;;
  all)
    stage_prepare
    stage_configs
    stage_audit
    stage_cache
    stage_shared
    stage_lock
    stage_arms
    ;;
  *)
    echo "usage: $0 {prepare|configs|audit|cache|shared|lock|arms|top500|top1000|top2000|all}" >&2
    exit 2
    ;;
esac
