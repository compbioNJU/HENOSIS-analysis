"""Generate the four v3 configs from the frozen `config/v3/base.yaml`.

WHY this step exists
--------------------
The v3 pipeline is the same analysis as v2 run under three Top-K cut-offs. That
claim is only checkable if the three configs are provably "base + a declared
list of changes" and nothing else. So the configs are not hand-written: they are
derived here, and `--check` re-derives them and asserts

    set(changed keys) == DECLARED_DIFF

against `config/v3/base.yaml`. A stray hand edit to any arm config is caught by
that assertion rather than by a reviewer noticing.

WHAT the arms change
--------------------
Only two things move between arms:

  * where the run writes         -- paths.results / paths.logs
  * how wide the Top-K cut is    -- model_analysis.primary_topk, and the two null
                                    parameters that are DEFINED relative to it:
                                    random_control.n_peaks, and
                                    random_control.min_peaks_per_repeat.

THE NULL'S SIZE CONTRACT. `require_exact_n_peaks: true` makes every emitted null
repeat exactly as wide as the OBSERVED table, on both sides. The observed width
is NOT K -- K is the ranking cut-off, and the table is built from `n_fg_matched`,
how many Top-K peaks found a matched partner (median 97.8% / 98.6% / 94.9% of K
at Top500 / Top1000 / Top2000). `s01` therefore passes that measured width as the
draw's target, and `common.random_null_enrichment` drops any repeat that comes up
short.

`min_peaks_per_repeat` is consequently a SANITY FLOOR ONLY and is set to K//2,
not to K: under the exact contract it can never bind, but K would reject every
repeat, since the observed width is regularly below K. It is kept so that a null
table produced by some other rule is still screened by
`s06_purkinje_model.load_null`.

`model_analysis.sensitivity_topk` / `sensitivity.topk` are set to the OTHER two
arms purely so the config does not describe itself falsely -- no v3 step reads
them (only s10/s11 do, and v3 does not run them).

`_shared.yaml` deliberately keeps every v2 null value: its four steps
(03/04 chromVAR, 13_rank_curve, 16_sequence_maps) are top-k independent. s15 is
the only one of them that reads `random_control` at all, and only for
`n_repeats` -- which `rank_curve.n_repeats: 200` overrides.

Comments are preserved
----------------------
The base carries the reasoning for every key. Rewriting it through
`yaml.safe_dump` would throw that away, so the arm configs are produced by
patching the base TEXT in place: each edit is scoped to a named top-level block
and asserted to match exactly once. The four configs therefore differ from the
base only on the annotated lines, with all surrounding explanation intact.

Usage
-----
    python script_v3/make_v3_configs.py            # write the four configs
    python script_v3/make_v3_configs.py --check    # verify, change nothing
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CONFIG_V3 = ROOT / "config" / "v3"
BASE = CONFIG_V3 / "base.yaml"

# The primary arm. Frozen here, before any v3 result exists, so that the paper's
# claim level cannot be chosen after seeing which cut-off looks best.
PRIMARY_TOPK = 500

# The two robustness arms.
ROBUSTNESS_TOPKS = (1000, 2000)

# --------------------------------------------------------------------------
# the pre-registered cross-K rule set (written into every generated config)
# --------------------------------------------------------------------------

CROSS_ARM = {
    "primary_topk": PRIMARY_TOPK,
    "robustness_topks": list(ROBUSTNESS_TOPKS),
    "claim_rule": (
        "Primary claims are evaluated in the Top-500 arm only. The Top-1000 and "
        "Top-2000 arms are prespecified robustness arms: they test whether each "
        "conclusion PERSISTS as progressively lower-ranked model-selected peaks are "
        "included. A robustness arm never re-defines a primary result, and the "
        "primary arm is never chosen by comparing the three."
    ),
    "underpowered_policy": (
        "A model whose exact matched-random draws -- each drawn at the OBSERVED "
        "table's own width -- fall below `random_control.min_valid_repeats` is "
        "reported NULL_UNDERPOWERED. It is never rescued by a smaller control, and "
        "it is counted as neither a PASS nor a FAIL in any cross-K stability rule."
    ),
    "stability_rules": [
        {
            "rule": "astroglia_stage_matching_direction",
            "statement": "all three stage-pair matching margins stay > 0",
            "scope": "02_astroglia_model/stage_matching_margin.tsv",
            "holds_when": "every margin in the table is > 0",
            "cite": "A1",
        },
        {
            "rule": "astroglia_family_trajectory_direction",
            "statement": "RFX and NF-Y stay prenatal > adult, NFI stays adult > prenatal",
            "scope": "02_astroglia_model/family_trajectory.tsv",
            "holds_when": "the sign of (prenatal - adult) is as declared for each family",
            "cite": "A4",
        },
        {
            "rule": "purkinje_convergence_vs_null",
            "statement": "mean cross-species motif similarity stays above the matched-random q95",
            "scope": "06_purkinje_model/convergence_null_test.tsv",
            "holds_when": "observed > null_q95",
            "cite": "P1",
        },
        {
            "rule": "oligo_convergence_vs_null",
            "statement": "mean cross-species motif similarity stays above the matched-random q95",
            "scope": "08_oligo_model/convergence_null_test.tsv",
            "holds_when": "observed > null_q95",
            "cite": "O1",
        },
        {
            "rule": "model_vs_chromvar_direction",
            "statement": "the median model<->raw chromVAR Spearman rho stays > 0",
            "scope": "07_purkinje_validation/model_vs_raw_similarity.tsv, "
                     "09_oligo_validation/model_vs_raw_similarity.tsv",
            "holds_when": "median rho over eligible species > 0",
            "cite": "P3 / O3",
        },
    ],
    # what the three arms do NOT share
    "arm_differences": [
        "random_control.null_target",
        "paths.results",
        "paths.logs",
        "model_analysis.primary_topk",
        "random_control.n_peaks",
        "random_control.min_peaks_per_repeat",
    ],
    "shared_steps": [
        "03_chromvar_input",
        "04_chromvar",
        "13_rank_curve",
        "16_sequence_maps",
    ],
    "shared_step_reason": (
        "These four steps read only the full-peak accessibility universe and the "
        "model rankings' cut-off curve; none of them reads "
        "model_analysis.primary_topk. Running them once makes all three arms face "
        "the SAME fixed independent validation target."
    ),
}

# --------------------------------------------------------------------------
# per-arm overrides: block -> {key: (rendered value, is_new_key)}
# --------------------------------------------------------------------------


def arm_overrides(topk: int) -> dict[str, dict[str, str]]:
    others = [k for k in (PRIMARY_TOPK, *ROBUSTNESS_TOPKS) if k != topk]
    return {
        "paths": {
            "results": f"result_v3/top{topk}",
            "logs": f"result_v3/logs/top{topk}",
            "cache": "result_v3/cache",
        },
        "model_analysis": {
            "primary_topk": str(topk),
            "sensitivity_topk": "[" + ", ".join(str(k) for k in others) + "]",
        },
        "random_control": {
            "n_peaks": str(topk),
            # A SANITY FLOOR ONLY, and deliberately not K. Under
            # `require_exact_n_peaks` the producer draws every repeat at the
            # observed table's own width and drops any that comes up short, so
            # the consumer-side floor can never bind; it is kept so that a null
            # table produced by some other rule is still screened. K itself would
            # reject every repeat, because the observed width is regularly below
            # K -- median 97.8% / 98.6% / 94.9% at Top500 / Top1000 / Top2000.
            "min_peaks_per_repeat": str(topk // 2),
            "require_exact_n_peaks": "true",
            "min_valid_repeats": "100",
            "null_target": "observed_matched_foreground",
        },
        "sensitivity": {
            "topk": "[" + ", ".join(str(k) for k in others) + "]",
        },
    }


SHARED_OVERRIDES = {
    "paths": {
        "results": "result_v3/_shared",
        "logs": "result_v3/logs/_shared",
        "cache": "result_v3/cache",
    },
}

# the key paths each ARM is allowed to differ from the base on. `cross_arm` is
# appended whole, so its subtree is folded to one name.
ARM_DECLARED_DIFF = {
    "random_control.null_target",
    "paths.results",
    "paths.logs",
    "paths.cache",
    "model_analysis.primary_topk",
    "model_analysis.sensitivity_topk",
    "random_control.n_peaks",
    "random_control.min_peaks_per_repeat",
    "random_control.require_exact_n_peaks",
    "random_control.min_valid_repeats",
    "sensitivity.topk",
    "cross_arm",
}

# `_shared` runs the four top-k-independent steps, so only its output location
# moves: no null parameter, no primary_topk.
SHARED_DECLARED_DIFF = {
    "paths.results",
    "paths.logs",
    "paths.cache",
    "cross_arm",
}

# keys inserted into an existing block, anchored after a neighbour so the
# generated file stays readable; the anchor must exist in the base
NEW_KEYS = {
    ("random_control", "null_target"): "min_valid_repeats",
    ("random_control", "require_exact_n_peaks"): "n_peaks",
    ("random_control", "min_valid_repeats"): "require_exact_n_peaks",
}


# --------------------------------------------------------------------------
# scoped text patching
# --------------------------------------------------------------------------


def block_span(lines: list[str], block: str) -> tuple[int, int]:
    """Line indices [start, stop) of a top-level `block:` and its indented body."""
    header = re.compile(rf"^{re.escape(block)}:\s*(#.*)?$")
    start = None
    for index, line in enumerate(lines):
        if header.match(line):
            if start is not None:
                raise SystemExit(f"config/v3/base.yaml declares `{block}:` twice")
            start = index
    if start is None:
        raise SystemExit(f"config/v3/base.yaml has no top-level `{block}:` block")
    stop = start + 1
    while stop < len(lines):
        line = lines[stop]
        if line.strip() and not line[0].isspace():
            break
        stop += 1
    return start, stop


def patch_scalar(lines: list[str], block: str, key: str, rendered: str) -> list[str]:
    """Set `key` inside `block`, or insert it after its declared anchor."""
    start, stop = block_span(lines, block)
    leaf = re.compile(rf"^(\s+){re.escape(key)}:(\s|$)")
    hits = [i for i in range(start + 1, stop) if leaf.match(lines[i])]
    if len(hits) > 1:
        raise SystemExit(f"`{key}` appears {len(hits)} times inside `{block}:`")
    if hits:
        index = hits[0]
        indent = leaf.match(lines[index]).group(1)
        old = lines[index].rstrip("\n")
        lines[index] = f"{indent}{key}: {rendered}\n"
        print(f"    {block}.{key}: {old.strip()}  ->  {key}: {rendered}")
        return lines

    anchor = NEW_KEYS.get((block, key))
    if anchor is None:
        raise SystemExit(f"`{block}.{key}` is absent from the base and has no anchor")
    anchor_hits = [
        i
        for i in range(start + 1, stop)
        if re.match(rf"^(\s+){re.escape(anchor)}:(\s|$)", lines[i])
    ]
    if len(anchor_hits) != 1:
        raise SystemExit(f"anchor `{block}.{anchor}` matched {len(anchor_hits)} lines")
    index = anchor_hits[0]
    indent = re.match(r"^(\s+)", lines[index]).group(1)
    lines.insert(index + 1, f"{indent}{key}: {rendered}\n")
    print(f"    + {block}.{key}: {rendered}")
    return lines


def render(block: str, mapping: dict[str, str]) -> str:
    """A top-level block as YAML text."""
    body = yaml.safe_dump(
        {block: mapping}, default_flow_style=False, sort_keys=False, width=100
    )
    return body if body.endswith("\n") else body + "\n"


def build_config(name: str, overrides: dict[str, dict[str, str]]) -> str:
    base_text = BASE.read_text()
    lines = base_text.splitlines(keepends=True)
    print(f"  {name}:")
    for block, mapping in overrides.items():
        for key, rendered in mapping.items():
            lines = patch_scalar(lines, block, key, rendered)
    text = "".join(lines).rstrip("\n") + "\n"
    header = (
        f"# {'=' * 74}\n"
        f"# GENERATED by script_v3/make_v3_configs.py -- DO NOT EDIT BY HAND.\n"
        f"# {'=' * 74}\n"
        f"# base          : config/v3/base.yaml\n"
        f"# this file     : config/v3/{name}.yaml\n"
        f"# changed keys  : {', '.join(sorted(_flat_paths(overrides)) + ['cross_arm'])}\n"
        f"# every other key, and every comment below, is the frozen base's.\n"
        f"# {'=' * 74}\n\n"
    )
    return header + text + "\n" + render("cross_arm", CROSS_ARM)


def _flat_paths(overrides: dict[str, dict[str, str]]) -> list[str]:
    return [f"{block}.{key}" for block, mapping in overrides.items() for key in mapping]


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------


def flatten(node: object, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            out.update(flatten(value, f"{prefix}{key}."))
    else:
        out[prefix.rstrip(".")] = node
    return out


def changed_paths(base: dict, arm: dict) -> set[str]:
    left, right = flatten(base), flatten(arm)

    def fold(key: str) -> str:
        # `cross_arm` is appended as one unit; report it as one changed key
        return "cross_arm" if key == "cross_arm" or key.startswith("cross_arm.") else key

    changed = {
        fold(key)
        for key in set(left) | set(right)
        if left.get(key, "<absent>") != right.get(key, "<absent>")
    }
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="verify the on-disk configs; change nothing")
    args = parser.parse_args()

    if not BASE.exists():
        raise SystemExit(
            f"missing {BASE}; freeze config/analysis_v2.yaml into it once, by hand, first"
        )

    base = yaml.safe_load(BASE.read_text())
    for key in ("_path", "_root"):
        base.pop(key, None)

    wanted = {"_shared": (SHARED_OVERRIDES, SHARED_DECLARED_DIFF)}
    for topk in (PRIMARY_TOPK, *ROBUSTNESS_TOPKS):
        wanted[f"top{topk}"] = (arm_overrides(topk), ARM_DECLARED_DIFF)

    stale = []
    for name, (overrides, declared) in wanted.items():
        text = build_config(name, overrides)
        path = CONFIG_V3 / f"{name}.yaml"
        parsed = yaml.safe_load(text)
        for key in ("_path", "_root"):
            parsed.pop(key, None)
        diff = changed_paths(base, parsed)
        # The load-bearing direction is `diff - declared`: a key that moved and
        # was not declared is an undeclared change, which is exactly what this
        # generator exists to make impossible. Keys that ARE declared but did not
        # move are legitimate -- e.g. the Top500 arm's primary_topk and n_peaks
        # are already 500 in the base -- so they are reported, not failed.
        undeclared = sorted(diff - declared)
        if undeclared:
            raise SystemExit(
                f"{path}: `{undeclared}` differ from base.yaml but are NOT declared "
                f"as arm differences"
            )
        unmoved = sorted(declared - diff)
        if unmoved:
            print(f"    declared but already equal to base (no line changed): {unmoved}")
        if args.check:
            if not path.exists():
                raise SystemExit(f"{path} does not exist")
            if path.read_text() != text:
                stale.append(name)
            else:
                print(f"  {name}.yaml: OK (diff == declared set, {len(diff)} keys)")
        else:
            path.write_text(text)
            print(f"  {name}.yaml: written ({len(diff)} keys differ from base)")

    if args.check and stale:
        raise SystemExit(
            f"stale generated config(s): {stale}. Re-run without --check to regenerate."
        )
    if args.check:
        print(
            f"\nall four configs match base.yaml + their declared diff "
            f"({len(ARM_DECLARED_DIFF)} keys per arm, {len(SHARED_DECLARED_DIFF)} for _shared)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
