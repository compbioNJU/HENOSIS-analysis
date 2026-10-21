#!/usr/bin/env python3
"""Generate the arm configs for this release from the frozen vendored base.

Why generate rather than write four files by hand
-------------------------------------------------
The three arms must differ from the base in exactly the declared set of keys and
in nothing else. Four hand-maintained YAML files drift: a key copied into the
wrong arm stops being a robustness arm and becomes a second primary analysis,
and nothing in the output would show it. So each arm is derived, and `--check`
re-derives all four and asserts the diff against the base is exactly `DECLARED_DIFF`.

Edits are applied as TEXT, not as a YAML round-trip, so every explanatory comment
in the base survives into the arm configs. The price is that a substitution which
fails to match must be fatal rather than silent -- hence `substitute()`, which
raises if the pattern is absent or matches more than once.

What this release changes relative to the vendored base
-------------------------------------------------------
1. `paths.ranked_dirs.astroglia` points at `artifacts/ranked_astroglia_v4/`,
   which holds symlinks to nine of the ten Astroglia rankings. The tenth,
   `human-CS18-19_CS20_CS22-23.csv`, is excluded because the cells it ranks are
   no longer astroglia in this release (see README.md). Every other ranking file
   is the same file as before.
2. `paths.results`, `paths.cache` and `paths.logs` move into this repository's own
   results tree AND out from under the vendored core's own results root, because
   `s17` and `s18` refuse to write inside that root. See PATHS_NOTE.
3. The Top-K keys and the appended `cross_arm` block, as in the vendored base.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
VENDOR = PKG / "vendor"
BASE = VENDOR / "config" / "v3" / "base.yaml"
OUT_DIR = PKG / "config" / "arms"

ARMS = (500, 1000, 2000)
ROBUSTNESS = {500: [1000, 2000], 1000: [500, 2000], 2000: [500, 1000]}

# Where this release's outputs go, INSIDE the vendored core's project root but
# NOT under its `results/` directory.
#
# WHY THIS EXACT LOCATION
# -----------------------
# Two vendored guards constrain the output path, in opposite directions:
#
#   * `s06_purkinje_model.py` requires `--out-dir` to resolve INSIDE `C.ROOT`
#     (the vendored project root): `if out_dir != C.ROOT and C.ROOT not in
#     out_dir.parents: parser.error(...)`;
#   * `s17_motif_forest.py` and `s18_sequence_maps.py` refuse to write anywhere
#     under `C.ROOT / "results"`.
#
# The intersection is "inside the project root, beside `results/`" -- which is
# exactly the layout the previous release used, where the new tree was
# `result_v3/` next to the guarded `results/`. `result_v4/` follows that.
#
# The first run of this pipeline put the outputs outside the project root
# entirely, which satisfied the second guard and tripped the first, aborting the
# run at step 6 of 21 in all three arms.
#
# A symlink does NOT help: `s06` calls `.resolve()`, so pointing `result_v4` at a
# directory outside the project root resolves back out and trips the same guard.
# The directory has to be genuinely inside.
#
# The repository still exposes these results at its own top level, as the symlink
# `results/04_peak_regulatory -> vendor/result_v4`, so a reader does not have to
# know about the vendored core's layout to find them.
RESULTS_REL = "result_v4"
RESULTS_PREFIX = RESULTS_REL
LOGS_PREFIX = f"{RESULTS_REL}/logs"

PATHS_NOTE = """# ----------------------------------------------------------------------------
# `results`, `cache` and `logs` live inside this vendored tree, in `result_v4/`
# rather than in `results/`. That is not a preference: `s06_purkinje_model.py`
# requires its output inside the project root, while `s17_motif_forest.py` and
# `s18_sequence_maps.py` refuse to write under `results/`. `result_v4/` is the
# intersection, and it is the same shape the previous release used. See
# script/make_configs.py for the full reasoning.
# ----------------------------------------------------------------------------
paths:"""

# The only keys an arm config may differ from the base in.
DECLARED_DIFF = (
    "paths.results",
    "paths.cache",
    "paths.logs",
    "paths.ranked_dirs.astroglia",
    "model_analysis.primary_topk",
    "model_analysis.sensitivity_topk",
    "random_control.n_peaks",
    "random_control.min_peaks_per_repeat",
    "random_control.require_exact_n_peaks",
    "random_control.min_valid_repeats",
    "random_control.null_target",
    "sensitivity.topk",
    "cross_arm",
)

CROSS_ARM = """
# ============================================================================
# APPENDED BY THIS REPOSITORY'S script/make_configs.py. NOT PART OF THE BASE.
# ============================================================================
# Which arm is the claim and which are robustness, and the rules that keep them
# apart. Identical in all four configs, so it cannot be used to tell them apart.
cross_arm:
  primary_topk: 500
  robustness_topks: [1000, 2000]
  claim_rule:
    - "every primary claim is evaluated in the Top-500 arm only"
    - "a robustness arm may never re-define a primary result"
    - "the primary arm is never chosen by comparing the three"
  underpowered_policy: >
    A model whose valid matched-random repeats fall below
    random_control.min_valid_repeats is reported NULL_UNDERPOWERED and counted
    as neither PASS nor FAIL. It is never rescued by a smaller control, because
    shrinking the control until it fits would make the null a property of the
    sample size rather than of the model.
  shared_steps: [03_chromvar_input, 04_chromvar, 13_rank_curve, 16_sequence_maps]
  shared_step_reason: >
    None of these four reads model_analysis.primary_topk, so running them once
    per arm would produce three byte-identical copies and three chances to
    disagree. They are computed once under _shared/ and symlinked into each arm.
  release_note: >
    In this release the Astroglia arm covers nine windows, not ten: the human
    early_embryonic window is absent because those cells are no longer labelled
    astroglia. The Purkinje and Oligo arms are unchanged from the previous
    release, which makes them a reproducibility control on the rewrite.
"""


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def flatten(node: dict, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in (node or {}).items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, dotted + "."))
        else:
            flat[dotted] = value
    return flat


def substitute(text: str, pattern: str, replacement: str, *, arm: str) -> str:
    """A single, mandatory, unambiguous line-level substitution.

    Raises if the pattern is absent (the base moved and this edit silently
    stopped applying) or matches twice (the edit is ambiguous).
    """
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if len(matches) != 1:
        raise SystemExit(
            f"arm {arm}: pattern {pattern!r} matched {len(matches)} times in the "
            f"base config; expected exactly 1. The vendored base has changed and "
            f"this generator must be updated to match it."
        )
    return text[: matches[0].start()] + replacement + text[matches[0].end():]


def build_arm(arm: int, base_text: str, *, with_cross_arm: bool = True) -> str:
    others = ROBUSTNESS[arm]
    text = base_text
    text = substitute(text, r"^  results: \S+$",
                      f"  results: {RESULTS_PREFIX}/top{arm}", arm=f"top{arm}")
    text = substitute(text, r"^  cache: \S+$", f"  cache: {RESULTS_PREFIX}/cache",
                      arm=f"top{arm}")
    text = substitute(text, r"^  logs: \S+$", f"  logs: {LOGS_PREFIX}/top{arm}",
                      arm=f"top{arm}")
    text = substitute(text, r"^paths:$", PATHS_NOTE, arm=f"top{arm}")
    text = substitute(
        text,
        r"^    astroglia: \S+$",
        "    astroglia: ../artifacts/ranked_astroglia_v4",
        arm=f"top{arm}",
    )
    text = substitute(text, r"^  primary_topk: \d+$", f"  primary_topk: {arm}",
                      arm=f"top{arm}")
    text = substitute(text, r"^  sensitivity_topk: \[[^\]]*\]$",
                      f"  sensitivity_topk: [{others[0]}, {others[1]}]",
                      arm=f"top{arm}")
    text = substitute(text, r"^  n_peaks: \d+$", f"  n_peaks: {arm}",
                      arm=f"top{arm}")
    text = substitute(text, r"^  min_peaks_per_repeat: \d+$",
                      f"  min_peaks_per_repeat: {arm // 2}", arm=f"top{arm}")
    text = substitute(text, r"^  topk: \[[^\]]*\]$",
                      f"  topk: [{others[0]}, {others[1]}]", arm=f"top{arm}")
    # the three keys the base does not carry at all: appended to random_control
    text = substitute(
        text,
        r"^  min_peaks_per_repeat: \d+$",
        f"  min_peaks_per_repeat: {arm // 2}\n"
        f"  require_exact_n_peaks: true\n"
        f"  min_valid_repeats: 100\n"
        f"  null_target: observed_matched_foreground",
        arm=f"top{arm}",
    )
    if with_cross_arm:
        text = text.rstrip("\n") + "\n" + CROSS_ARM
    return text


def build_shared(base_text: str) -> str:
    text = base_text
    for pattern, replacement in (
        (r"^  results: \S+$", f"  results: {RESULTS_PREFIX}/_shared"),
        (r"^  cache: \S+$", f"  cache: {RESULTS_PREFIX}/cache"),
        (r"^  logs: \S+$", f"  logs: {LOGS_PREFIX}/_shared"),
        (r"^    astroglia: \S+$", "    astroglia: ../artifacts/ranked_astroglia_v4"),
        (r"^paths:$", PATHS_NOTE),
    ):
        text = substitute(text, pattern, replacement, arm="_shared")
    return text.rstrip("\n") + "\n" + CROSS_ARM


def changed_keys(base: dict, arm: dict) -> set[str]:
    """Dotted keys whose value differs, with `cross_arm` kept whole.

    `cross_arm` is compared at the top level rather than flattened: it is one
    appended block whose internal shape is this repository's business, and
    listing its seven sub-keys as seven separate declared differences would make
    the declaration unreadable without making it any stricter.
    """
    flat_base, flat_arm = flatten(base), flatten(arm)
    top = {k for k in ("cross_arm",) if (k in base) != (k in arm)}
    keys = (set(flat_base) | set(flat_arm)) - {
        k for k in (set(flat_base) | set(flat_arm)) if k.startswith("cross_arm.")
    }
    changed = {key for key in keys if flat_base.get(key) != flat_arm.get(key)}
    return changed | top


def intended(arm: int) -> dict[str, object]:
    """The exact value every declared key must hold in this arm."""
    others = ROBUSTNESS[arm]
    return {
        "paths.results": f"{RESULTS_PREFIX}/top{arm}",
        "paths.cache": f"{RESULTS_PREFIX}/cache",
        "paths.logs": f"{LOGS_PREFIX}/top{arm}",
        "paths.ranked_dirs.astroglia": "../artifacts/ranked_astroglia_v4",
        "model_analysis.primary_topk": arm,
        "model_analysis.sensitivity_topk": others,
        "random_control.n_peaks": arm,
        "random_control.min_peaks_per_repeat": arm // 2,
        "random_control.require_exact_n_peaks": True,
        "random_control.min_valid_repeats": 100,
        "random_control.null_target": "observed_matched_foreground",
        "sensitivity.topk": others,
    }


def intended_shared() -> dict[str, object]:
    return {
        "paths.results": f"{RESULTS_PREFIX}/_shared",
        "paths.cache": f"{RESULTS_PREFIX}/cache",
        "paths.logs": f"{LOGS_PREFIX}/_shared",
        "paths.ranked_dirs.astroglia": "../artifacts/ranked_astroglia_v4",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="re-derive and assert the diff, writing nothing")
    args = parser.parse_args()

    if not BASE.exists():
        raise SystemExit(f"vendored base config missing: {BASE}")
    base_text = BASE.read_text(encoding="utf-8")
    base = load(BASE)
    print(f"[BASE] {BASE}")
    print(f"       sha256 {hashlib.sha256(base_text.encode()).hexdigest()}")

    generated = {"_shared": build_shared(base_text)}
    for arm in ARMS:
        generated[f"top{arm}"] = build_arm(arm, base_text)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    for name, text in generated.items():
        parsed = yaml.safe_load(text)
        changed = changed_keys(base, parsed)
        expected = (set(DECLARED_DIFF) if name != "_shared"
                    else {"paths.results", "paths.cache", "paths.logs",
                          "paths.ranked_dirs.astroglia", "cross_arm"})

        # 1. nothing may change that was not declared
        undeclared = changed - expected
        # 2. every declared key must hold its intended value. Some of them
        #    already held it in the base (Top500 sets primary_topk to 500, which
        #    is the base's own value), so a missing entry in `changed` is fine as
        #    long as the value is right.
        wanted = intended_shared() if name == "_shared" else intended(int(name[3:]))
        flat = flatten(parsed)
        wrong = {
            key: (value, flat.get(key))
            for key, value in wanted.items() if flat.get(key) != value
        }

        ok = not undeclared and not wrong
        print(f"[{'OK' if ok else 'PROBLEM'}] {name}: {len(changed)} keys differ from base")
        if undeclared:
            problems.append(name)
            print(f"         changed but NOT declared: {sorted(undeclared)}")
        if wrong:
            problems.append(name)
            for key, (want, got) in sorted(wrong.items()):
                print(f"         {key}: wanted {want!r}, found {got!r}")
        if not args.check:
            (OUT_DIR / f"{name}.yaml").write_text(text, encoding="utf-8")

    if problems:
        raise SystemExit(
            f"config generation failed for {sorted(set(problems))}: a change "
            f"outside the declared key set, or a declared key with the wrong value"
        )
    print(f"\n[DONE] {'checked' if args.check else 'wrote'} {len(generated)} configs "
          f"-> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
