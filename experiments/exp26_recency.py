"""Experiment 26: does down-weighting old training years help?

MOTIVATION
Both diseases are non-stationary, but for different reasons and on different
timescales. Septoria drifted up gradually (decade means 51.9 -> 75.6). Yellow
rust did something sharper: 22.0 in the 1970s, 0.8 in the 2000s as resistant
varieties spread, then back to 9.4 in the 2020s after the Warrior race incursion
around 2011. Rust's weather->disease slope is therefore fitted partly on a
1990s/2000s era whose host genetics no longer exist, which would explain why the
model is WORSE than climatology on rust for 2005-2019 (2.518 vs 2.340) and only
wins on 2021-2025.

An exponential recency weight on training rows is the obvious response: keep all
the data (only ~450 rows, throwing any away is expensive) but let recent years
dominate the fit. exp09 tried this under the OLD architecture (alpha=10, no
ensemble, no mechanistic basis) and it was not adopted; it has never been tested
inside the current ensemble, where the as-of baseline features already absorb
some of the level shift and may leave recency with nothing to do.

Gridded jointly with MIN_YEAR (hard cutoff -- the crude version of the same
idea) and ALPHA, since a recency weight shrinks the effective sample size and so
changes the right amount of ridge penalty.

READING THE TABLE -- this grid is scored on the same folds it selects from, so
the best cell is optimistically biased. A cell is only worth adopting if it
improves BOTH the 2005-2019 and 2021-2025 sub-windows; a cell that wins overall
by trading one window against the other is fitting the fold split, not the
problem. Septoria and rust are decided separately, since they are separate
models sharing a harness.
"""
import warnings
import itertools
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

from common import TARGETS, to_long, score
import final_model_v2 as FM

EVAL = [y for y in range(2005, 2026) if y != 2020]
W19 = list(range(2005, 2020))
W25 = [2021, 2022, 2023, 2024, 2025]
truth = to_long(FM.OBS)

BASE = dict(halflife=FM.HALFLIFE, min_year=FM.MIN_YEAR, alpha=FM.ALPHA)


def evaluate(halflife, min_year, alpha):
    FM.HALFLIFE, FM.MIN_YEAR, FM.ALPHA = halflife, min_year, alpha
    try:
        p = FM.run(EVAL)
    finally:
        FM.HALFLIFE, FM.MIN_YEAR, FM.ALPHA = (
            BASE["halflife"], BASE["min_year"], BASE["alpha"])
    rec = {"halflife": halflife or 0, "min_year": min_year, "alpha": alpha}
    for lab, yrs in [("05_25", EVAL), ("05_19", W19), ("21_25", W25)]:
        _, s = score(p, truth, yrs)
        rec[f"sept_{lab}"] = s["septoria_pooled"]
        rec[f"rust_{lab}"] = s["rust_pooled"]
    return rec


if __name__ == "__main__":
    grid = list(itertools.product([None, 30, 20, 15, 10],
                                  [1990],
                                  [100.0]))
    grid += [(hl, my, 100.0) for hl in [None, 20] for my in [1980, 1995, 2000]]
    grid += [(hl, 1990, a) for hl in [15, 20] for a in [30.0, 300.0]]

    rows = []
    for hl, my, a in grid:
        rows.append(evaluate(hl, my, a))
        print(f"  done halflife={hl} min_year={my} alpha={a}", flush=True)

    df = pd.DataFrame(rows).drop_duplicates(subset=["halflife", "min_year", "alpha"])
    ref = df[(df.halflife == 0) & (df.min_year == 1990) & (df.alpha == 100.0)].iloc[0]

    print("\n" + "=" * 118)
    print("RECENCY WEIGHTING GRID   (halflife 0 = equal weights = shipped v2)")
    print("=" * 118)
    for grp in ["sept", "rust"]:
        d = df.sort_values(f"{grp}_05_25")
        d = d.assign(**{
            "d_05_19": d[f"{grp}_05_19"] - ref[f"{grp}_05_19"],
            "d_21_25": d[f"{grp}_21_25"] - ref[f"{grp}_21_25"]})
        d["both_better"] = np.where((d.d_05_19 < 0) & (d.d_21_25 < 0), "YES", "")
        print(f"\n--- {grp.upper()}  (shipped: {ref[f'{grp}_05_25']:.3f} | "
              f"{ref[f'{grp}_05_19']:.3f} | {ref[f'{grp}_21_25']:.3f})")
        print(d[["halflife", "min_year", "alpha", f"{grp}_05_25", f"{grp}_05_19",
                 f"{grp}_21_25", "d_05_19", "d_21_25", "both_better"]]
              .round(3).to_string(index=False))
