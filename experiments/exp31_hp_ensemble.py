"""Experiment 31: ensemble over hyperparameters instead of picking one setting.

MOTIVATION
exp26 gridded alpha, MIN_YEAR and recency half-life and found a consistent
pattern: every cell that improved 2021-2025 made 2005-2019 worse and vice versa.
I read that as "no setting is better" and moved on. But there is a second
reading. If setting A is right for one regime and setting B for another, and I
cannot know in advance which regime the forecast year belongs to, then the
variance-minimising choice is not to pick -- it is to AVERAGE. That is exactly
the argument that already justifies averaging over feature bases and model forms
in v2 ("they fail on different eras, so averaging beats choosing"), and it was
never applied to the hyperparameters themselves.

Concretely the shipped model fixes ALPHA=100, MIN_YEAR=1990, CLIM_K=12. Each is
a point estimate of something genuinely uncertain, and each has a defensible
range. Averaging predictions across that range costs nothing in bias if the
settings straddle the optimum, and cuts the variance contributed by having
guessed.

WHAT IS TESTED
Axes added one at a time so the source of any gain is attributable:
  * alpha   {30, 100, 300}   -- exp26 showed 30 favours rust/recent, 300 favours
                                septoria/recent, 100 is the compromise
  * min_year{1980, 1990}     -- how much pre-modern-era data to admit
  * clim_k  {8, 12, 16}      -- climatology window, never tuned at all in v2

ADOPTION RULE (unchanged from exp26/exp28): must improve BOTH the 2005-2019 and
2021-2025 windows. Averaging is only worth adopting if it is robust, and a
variant that wins overall by trading windows is doing the thing this experiment
is supposed to avoid.
"""
import itertools
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

from common import TARGETS, SEPTORIA, INCIDENCE, to_long, score
import final_model_v2 as FM

EVAL = [y for y in range(2005, 2026) if y != 2020]
W19 = list(range(2005, 2020))
W25 = [2021, 2022, 2023, 2024, 2025]
truth = to_long(FM.OBS)
BASE = (FM.ALPHA, FM.MIN_YEAR, FM.CLIM_K)


def hp_predict(grid):
    """Mean over FM.predict evaluated at each hyperparameter setting.

    Averaging happens at member level, BEFORE enforce_leaf_order, so the
    constraint is applied once to the final ensemble rather than per setting.
    """
    def fn(target, T):
        ps = []
        for a, my, ck in grid:
            FM.ALPHA, FM.MIN_YEAR, FM.CLIM_K = a, my, ck
            try:
                p = FM.predict(target, T)
            finally:
                FM.ALPHA, FM.MIN_YEAR, FM.CLIM_K = BASE
            if p is not None:
                ps.append(p.sort_values(["Year", "Region"]).reset_index(drop=True))
        if not ps:
            return None
        out = ps[0][["Year", "Region", "target"]].copy()
        v = np.mean([p.value.to_numpy() for p in ps], axis=0)
        out["value"] = np.clip(v, 0, 100 if target in INCIDENCE else None)
        return out
    return fn


def evaluate(grid, label):
    p = FM.run(EVAL, hp_predict(grid))
    rec = {"variant": label, "n_members": len(grid) * 6}
    for lab, yrs in [("05_25", EVAL), ("05_19", W19), ("21_25", W25)]:
        _, s = score(p, truth, yrs)
        rec[f"sept_{lab}"] = s["septoria_pooled"]
        rec[f"rust_{lab}"] = s["rust_pooled"]
    return rec


if __name__ == "__main__":
    A, MY, CK = [30.0, 100.0, 300.0], [1980, 1990], [8, 12, 16]
    variants = [
        ([(100.0, 1990, 12)], "v2 shipped (single setting)"),
        ([(a, 1990, 12) for a in A], "avg alpha {30,100,300}"),
        ([(100.0, my, 12) for my in MY], "avg min_year {1980,1990}"),
        ([(100.0, 1990, ck) for ck in CK], "avg clim_k {8,12,16}"),
        ([(a, my, 12) for a in A for my in MY], "avg alpha x min_year"),
        ([(a, 1990, ck) for a in A for ck in CK], "avg alpha x clim_k"),
        (list(itertools.product(A, MY, CK)), "avg all three"),
    ]
    rows = []
    for grid, lab in variants:
        rows.append(evaluate(grid, lab))
        print(f"  done {lab} ({len(grid)} settings)", flush=True)

    df = pd.DataFrame(rows)
    ref = df.iloc[0]
    for g in ["sept", "rust"]:
        df[f"d_{g}19"] = df[f"{g}_05_19"] - ref[f"{g}_05_19"]
        df[f"d_{g}25"] = df[f"{g}_21_25"] - ref[f"{g}_21_25"]
        df[f"{g}_both"] = np.where((df[f"d_{g}19"] < 0) & (df[f"d_{g}25"] < 0), "YES", "")

    print("\n" + "=" * 132)
    print("HYPERPARAMETER ENSEMBLING   ('both' = improves 2005-19 AND 2021-25, the "
          "adoption bar)")
    print("=" * 132)
    print(df[["variant", "n_members", "sept_05_25", "sept_05_19", "sept_21_25",
              "sept_both", "rust_05_25", "rust_05_19", "rust_21_25", "rust_both"]]
          .round(3).to_string(index=False))
