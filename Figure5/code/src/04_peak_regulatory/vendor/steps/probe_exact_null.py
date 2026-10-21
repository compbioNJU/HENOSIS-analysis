"""Can an exact matched draw be drawn at all -- per (model, arm), per drawer?

WHY this runs BEFORE the arms
-----------------------------
v3's matched-random null is supposed to be size-exact: the null's Fisher table
must have the same size as the observed one, because a null drawn at a different
depth differs from the observation by sampling depth as well as by chance.

That requirement turns out to be harder than it looks, and the answer has to be
known BEFORE a multi-hour arm run rather than after. This step measures it from
the real rankings, using only the sequence cache (peak length + GC) -- the FIMO
motif hits are irrelevant to feasibility, which is a property of the matching.

THE TWO NUMBERS THAT MATTER
---------------------------
`K` is the RANKING cut-off, not the size of a Fisher table. The observed table
size is `n_fg_matched`: how many of the model's Top-K peaks actually found a
GC/length-matched partner in the pool. On this data that is 488-499 of 500, not
500. So "exact-K" is ill-posed as stated, and there are two candidate targets:

    target_K    = K                    -- the ranking cut-off
    target_obs  = observed n_fg_matched -- the size the observation really has

THE TWO DRAWERS
---------------
`paired`     each template peak must supply TWO distinct matched pool peaks, and
             a peak that cannot is skipped. This is `s15_rank_curve`'s allocator.
`resampled`  the same, but when the template is exhausted short of the target,
             further template peaks are drawn WITH REPLACEMENT from the template's
             own (length, GC) profile and served too. The drawn profile therefore
             still tracks the Top-K profile, and the target becomes reachable
             whenever the pool can supply it at all.

The probe reports both, at both targets, so the null rule can be chosen on
measurements instead of on the assumption that exactness is achievable.

It decides nothing and writes only its own table.

Usage
-----
    python script_v3/probe_exact_null.py                  # 20 probe draws per cell
    python script_v3/probe_exact_null.py --repeats 50 --topk 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import common as C  # noqa: E402

ROOT = SCRIPT_DIR.parent
CONFIG_V3 = ROOT / "config" / "v3"
ARMS = (500, 1000, 2000)
DRAWERS = ("paired", "resampled")

COLUMNS = [
    "arm",
    "topk",
    "model",
    "analysis",
    "species",
    "n_ranked",
    "n_top",
    "n_top_valid",
    "n_pool",
    "target_K",
    "obs_fg_matched",
    "obs_shortfall",
    "drawer",
    "n_probe",
    "exact_at_K",
    "rate_at_K",
    "best_at_K",
    "exact_at_obs",
    "rate_at_obs",
    "best_at_obs",
    "pool_headroom",
    "obs_fg_gc_median",
    "null_fg_gc_median",
    "gc_ks_p",
    "max_fg_bg_gc_gap",
    "caliper",
    "balance",
    "verdict",
]


def draw_paired_matched_sets(
    template: pd.DataFrame,
    pool: pd.DataFrame,
    rng: np.random.Generator,
    caliper: float,
    target: int,
) -> tuple[list[str], list[str]]:
    """One pass over the template, two partners per peak, no resampling.

    This is `s15_rank_curve`'s allocator, kept HERE rather than in `common.py`
    because the pipeline does not use it: it is the control arm of this
    measurement. It exists in this file so the two drawers can be compared on
    identical inputs, which is the whole reason the probe runs both.
    """
    pool_length = pool.length.to_numpy()
    pool_gc = pool.gc.to_numpy(dtype=float)
    tpl_length = template.length.to_numpy()
    tpl_gc = template.gc.to_numpy(dtype=float)

    candidates: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    for length, gc in zip(tpl_length, tpl_gc):
        index = np.flatnonzero((pool_length == length) & (np.abs(pool_gc - gc) <= caliper))
        candidates.append(index)
        distances.append(np.abs(pool_gc[index] - gc))

    available = np.ones(len(pool), dtype=bool)
    pool_name = pool.peak.to_numpy()
    foreground: list[str] = []
    background: list[str] = []
    for row in sorted(range(len(candidates)), key=lambda i: (len(candidates[i]), i)):
        if len(foreground) >= target:
            break
        picked: list[int] = []
        for _ in range(2):
            pick = C._serve_one(candidates[row], distances[row], available, rng)
            if pick is None:
                break
            picked.append(pick)
        if len(picked) < 2:
            for pick in picked:
                available[pick] = True
            continue
        foreground.append(str(pool_name[picked[0]]))
        background.append(str(pool_name[picked[1]]))
    return foreground, background


def draw_resampled_matched_sets(
    template: pd.DataFrame,
    pool: pd.DataFrame,
    rng: np.random.Generator,
    caliper: float,
    target: int,
    max_rounds: int = 40,
) -> tuple[list[str], list[str]]:
    """`paired`, continued past the template by resampling it with replacement.

    Serving the template once can finish short of `target` even when the pool has
    thousands of spare peaks, because each template peak needs two partners
    inside a local GC band. Here the template's own (length, GC) profile is
    resampled with replacement and served until `target` pairs are placed, so the
    drawn profile still tracks the Top-K profile while the size becomes
    attainable. Returns short only when the pool genuinely cannot supply it.

    Candidate sets are precomputed once per template peak: availability changes
    between rounds but a peak's (length, GC) candidates do not, and recomputing
    them per round is what makes this otherwise quadratic on a 2,000-peak
    template.
    """
    pool_length = pool.length.to_numpy()
    pool_gc = pool.gc.to_numpy(dtype=float)
    pool_name = pool.peak.to_numpy()
    tpl_length = template.length.to_numpy()
    tpl_gc = template.gc.to_numpy(dtype=float)

    candidates: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    for length, gc in zip(tpl_length, tpl_gc):
        index = np.flatnonzero((pool_length == length) & (np.abs(pool_gc - gc) <= caliper))
        candidates.append(index)
        distances.append(np.abs(pool_gc[index] - gc))

    available = np.ones(len(pool), dtype=bool)
    foreground: list[str] = []
    background: list[str] = []

    def serve(row: int) -> bool:
        picked: list[int] = []
        for _ in range(2):
            pick = C._serve_one(candidates[row], distances[row], available, rng)
            if pick is None:
                break
            picked.append(pick)
        if len(picked) < 2:
            for pick in picked:
                available[pick] = True
            return False
        foreground.append(str(pool_name[picked[0]]))
        background.append(str(pool_name[picked[1]]))
        return True

    order = sorted(range(len(candidates)), key=lambda i: (len(candidates[i]), i))
    for row in order:
        if len(foreground) >= target:
            break
        serve(row)

    rounds = 0
    while len(foreground) < target and rounds < max_rounds:
        rounds += 1
        for row in rng.permutation(len(candidates)):
            if len(foreground) >= target:
                break
            serve(int(row))
    return foreground, background


def measure(
    template: pd.DataFrame,
    pool: pd.DataFrame,
    caliper: float,
    target: int,
    drawer: str,
    n_probe: int,
    seed: int,
    model_index: int,
) -> tuple[int, int, float]:
    fn = draw_paired_matched_sets if drawer == "paired" else C.draw_resampled_matched_sets
    achieved = []
    for repeat in range(n_probe):
        rng = np.random.default_rng([seed, model_index, repeat])
        fg, _ = fn(template, pool, rng, caliper, target)
        achieved.append(len(fg))
    n_exact = sum(1 for value in achieved if value == target)
    return n_exact, int(max(achieved)), n_exact / n_probe


def balance_of_draw(
    template: pd.DataFrame,
    pool: pd.DataFrame,
    coords: pd.DataFrame,
    obs_fg: list[str],
    caliper: float,
    seed: int,
    model_index: int,
) -> dict:
    """Is the exact-sized null drawn from the SAME covariate profile as the observation?

    Reaching the target size is not enough: the null has to be comparable to the
    observed Fisher table, and the observed table is built from
    `matched_foreground_background` -> the OBSERVED MATCHED FOREGROUND, i.e. the
    Top-K peaks that found a partner. That set is already narrower in GC than the
    raw Top-K set, because the unmatchable extremes are dropped. So the correct
    baseline for balance is that matched foreground, NOT the raw ranking.

    Reported per cell:
      gc_ks_p            two-sample KS of the drawn foreground's GC against the
                         observed matched foreground's GC. Above 0.05 = the null
                         is drawn from the same GC profile as the observation.
      max_fg_bg_gc_gap   the largest |fg GC - bg GC| in the draw; must stay inside
                         `caliper`, since the pairing rule is the whole reason the
                         null is comparable.
      balance            "ok" / "marginal" / "imbalanced", from gc_ks_p.

    A residual imbalance is expected to grow with the cut-off: the draw consumes
    2 * target pool peaks, so at Top2000 that is ~4,000 of the ~8,000 non-top-K
    peaks and the GC tails have less room. It is measured here rather than assumed
    away, and it never fails the run -- it is a property of the data's pool sizes.
    """
    target = len(obs_fg)
    if not target:
        return {"obs_fg_gc_median": np.nan, "null_fg_gc_median": np.nan,
                "gc_ks_p": np.nan, "max_fg_bg_gc_gap": np.nan,
                "caliper": caliper, "balance": "no_observed_foreground"}
    dfg, dbg = C.draw_resampled_matched_sets(
        template, pool, np.random.default_rng([seed, model_index]), caliper, target
    )
    if len(dfg) != target:
        return {"obs_fg_gc_median": float(coords.loc[obs_fg].gc.median()),
                "null_fg_gc_median": np.nan, "gc_ks_p": np.nan,
                "max_fg_bg_gc_gap": np.nan, "caliper": caliper,
                "balance": "cannot_draw_at_target"}
    from scipy import stats

    o_gc = coords.loc[obs_fg].gc.to_numpy(dtype=float)
    d_gc = coords.loc[dfg].gc.to_numpy(dtype=float)
    b_gc = coords.loc[dbg].gc.to_numpy(dtype=float)
    p_value = float(stats.ks_2samp(d_gc, o_gc).pvalue)
    return {
        "obs_fg_gc_median": float(np.median(o_gc)),
        "null_fg_gc_median": float(np.median(d_gc)),
        "gc_ks_p": p_value,
        "max_fg_bg_gc_gap": float(np.max(np.abs(d_gc - b_gc))),
        "caliper": caliper,
        "balance": "ok" if p_value > 0.05 else ("marginal" if p_value > 0.001 else "imbalanced"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repeats", type=int, default=20,
                        help="probe draws per cell; these are NOT the run's repeats")
    parser.add_argument("--topk", type=int, nargs="*", default=None)
    parser.add_argument("--drawer", nargs="*", default=list(DRAWERS),
                        help=f"subset of {DRAWERS}")
    parser.add_argument("--out", default=None,
                        help="default: result_v3/PROBE_EXACT_NULL.tsv")
    args = parser.parse_args()

    arms = tuple(args.topk) if args.topk else ARMS
    drawers = tuple(args.drawer) if args.drawer else DRAWERS
    out = Path(args.out) if args.out else (ROOT / "result_v3" / "PROBE_EXACT_NULL.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for topk in arms:
        config = C.load_config(CONFIG_V3 / f"top{topk}.yaml")
        seed = C.seed(config)
        caliper = float(config["model_analysis"]["background"]["gc_caliper"])
        manifest, rankings = C.load_rankings(config)
        manifest, dropped = C.apply_species_filter(config, manifest)
        if len(dropped):
            C.log(f"top{topk}: eligibility dropped", len(dropped), "model(s)")
        universes = C.model_species_peaks(manifest, rankings)

        print(f"\n=== top{topk} (n_peaks={config['random_control']['n_peaks']}, "
              f"target_K={topk}, min_peaks_per_repeat={config['random_control']['min_peaks_per_repeat']}) ===")

        for species, peaks in universes.items():
            genome = C.species_table(config)[species]
            sequences = C.read_sequences(config, genome, peaks, use_cache=True)
            valid = sequences.valid.astype(bool).to_numpy()
            valid_peaks = set(sequences.peak.to_numpy()[valid])
            coords = sequences[valid].set_index("peak")[["length", "gc"]]
            for row in manifest.itertuples():
                if row.species != species:
                    continue
                index = int(row.Index)
                ranking = rankings[row.model]
                top = ranking[ranking["rank"] <= topk]
                top_valid = top.peak[top.peak.isin(valid_peaks)]
                own = ranking[ranking["rank"] > topk]
                pool = coords.loc[coords.index.intersection(set(own.peak))].reset_index()
                if top_valid.empty or pool.empty:
                    continue
                template = (
                    coords.loc[coords.index.intersection(set(top_valid))]
                    .reset_index()[["peak", "length", "gc"]]
                    .sort_values("peak").reset_index(drop=True)
                )

                # the OBSERVED matched foreground: how many Top-K peaks really found
                # a partner under the same rule. This is the size a size-exact null
                # would have to reproduce, and it is not K.
                obs_rng = np.random.default_rng([seed, index])
                obs_fg, _, _ = C.matched_foreground_background(template, pool, obs_rng, caliper)
                obs_n = len(obs_fg)

                if len(template) > topk:
                    template = (template.sample(n=topk, random_state=index)
                                .sort_values("peak").reset_index(drop=True))

                for drawer in drawers:
                    n_k, best_k, rate_k = measure(template, pool, caliper, topk,
                                                  drawer, args.repeats, seed, index)
                    n_o, best_o, rate_o = measure(template, pool, caliper, obs_n,
                                                  drawer, args.repeats, seed, index + 1000)
                    rows.append({
                        "arm": f"top{topk}", "topk": topk, "model": row.model,
                        "analysis": row.analysis, "species": species,
                        "n_ranked": int(row.n_ranked), "n_top": int(len(top)),
                        "n_top_valid": int(len(top_valid)), "n_pool": int(len(pool)),
                        "target_K": topk, "obs_fg_matched": obs_n,
                        "obs_shortfall": int(len(template) - obs_n),
                        "drawer": drawer, "n_probe": args.repeats,
                        "exact_at_K": n_k, "rate_at_K": rate_k, "best_at_K": best_k,
                        "exact_at_obs": n_o, "rate_at_obs": rate_o, "best_at_obs": best_o,
                        "pool_headroom": int(len(pool) - 2 * topk),
                        **balance_of_draw(template, pool, coords, obs_fg, caliper,
                                          seed, index + 2000),
                        "verdict": (
                            "exact_at_obs" if rate_o == 1.0
                            else "partial_at_obs" if n_o
                            else "cannot_reach_obs"
                        ),
                    })
            print(f"  {species}: {len(valid_peaks)}/{len(peaks)} peaks with sequence")

    frame = pd.DataFrame(rows)[COLUMNS].sort_values(["topk", "drawer", "model"])
    C.save_table(frame, out, sep="\t")
    print(f"\nwrote {C.display_path(out)} ({len(frame)} rows)\n")

    print("=== observed matched foreground size (the size a size-exact null must hit) ===")
    obs = frame.drop_duplicates(["topk", "model"])
    for k, g in obs.groupby("topk"):
        short = g.obs_shortfall
        print(f"  K={k}: obs_fg_matched = {int(g.obs_fg_matched.min())}-{int(g.obs_fg_matched.max())} "
              f"(short by {int(short.min())}-{int(short.max())} of the Top-K peaks), "
              f"median shortfall {int(short.median())}")

    print("\n=== can each drawer hit the target EXACTLY? (fraction of probe draws) ===")
    for (k, drawer), g in frame.groupby(["topk", "drawer"]):
        all_k = int((g.rate_at_K == 1.0).sum())
        all_o = int((g.rate_at_obs == 1.0).sum())
        any_o = int((g.rate_at_obs > 0).sum())
        print(f"  K={k:5d}  {drawer:9s}  exact@K: {all_k:2d}/{len(g)} models can   "
              f"exact@observed: {all_o:2d}/{len(g)} always, {any_o:2d}/{len(g)} sometimes")

    bad = frame[(frame.drawer == "resampled") & (frame.rate_at_obs < 1.0)]
    if len(bad):
        print("\nmodels a RESAMPLED drawer still cannot draw exactly at the observed size:")
        print(bad[["topk", "model", "n_pool", "obs_fg_matched", "best_at_obs",
                   "pool_headroom"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
