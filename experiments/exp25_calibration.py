"""Experiment 25: is the ensemble mis-calibrated in scale?

MOTIVATION
exp24's bias table showed the ensemble missing 2023 septoria L1 incidence by
-30.9 and 2025 by +11.4 -- big errors in OPPOSITE directions, in a wet year and
a dry year respectively. That is the signature of a prediction that moves the
right way but not far enough: an under-dispersed forecast.

Ridge is deliberately over-shrunk (alpha=100), and averaging 6 ensemble members
shrinks further, so predicted anomalies being too small a priori is plausible.

TEST
Write every prediction as climatology plus an anomaly, and ask what multiplier on
that anomaly would have been optimal:

    y ~ clim + beta * (pred - clim)

beta > 1  => under-dispersed, inflating the anomaly helps
beta < 1  => over-confident, blending back toward climatology helps
beta = 1  => already calibrated

beta is fitted with NO intercept, so it is a pure scale question. Note that
beta is exactly the climatology-blend weight, so this single experiment also
answers "should target X just be shrunk toward climatology?" (the per-target
skill table has rust L1 severity at -4.2 %, i.e. worse than climatology).

HONESTY
Two numbers are reported for each target:
  * an ORACLE beta fitted on all 20 backtest folds -- an upper bound on what
    recalibration could ever buy, not an achievable score;
  * an AS-OF beta re-estimated at each year T using only folds < T and shrunk
    toward 1, which is a legitimate forecast and is what would ship.
If the as-of version does not beat the raw ensemble, the idea is dead regardless
of how good the oracle looks.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

from common import TARGETS, SEPTORIA, INCIDENCE, to_long, score, rmse
import final_model_v2 as FM

EVAL = [y for y in range(2005, 2026) if y != 2020]
OBS = FM.OBS
truth = to_long(OBS)


def clim_fn(t, T):
    p, te = FM._fit(t, T, "clim", "fix")
    return None if p is None else te[["Year", "Region"]].assign(target=t, value=p)


print("running backtest ...", flush=True)
BT = FM.run(EVAL)
CL = FM.run(EVAL, clim_fn)

# one tidy frame: actual, model, climatology
M = (truth[truth.Year.isin(EVAL)]
     .merge(BT.rename(columns={"value": "pred"}), on=["Year", "Region", "target"])
     .merge(CL.rename(columns={"value": "clim"}), on=["Year", "Region", "target"])
     .dropna(subset=["value"]))
M["dp"] = M.pred - M.clim          # predicted anomaly
M["dy"] = M.value - M.clim         # actual anomaly


def beta_of(g):
    """OLS slope through the origin of actual anomaly on predicted anomaly."""
    d = (g.dp ** 2).sum()
    return np.nan if d < 1e-9 else (g.dp * g.dy).sum() / d


print("\n" + "=" * 104)
print("ORACLE calibration slope per target (fitted on ALL folds -- upper bound, not a score)")
print("=" * 104)
print(f"{'target':<45}{'beta':>8}{'raw RMSE':>11}{'calib':>9}{'gain':>8}")
rows = []
for t in TARGETS:
    g = M[M.target == t]
    b = beta_of(g)
    r0 = rmse(g.value, g.pred)
    r1 = rmse(g.value, np.clip(g.clim + b * g.dp, 0, 100 if t in INCIDENCE else None))
    rows.append({"target": t, "beta": b, "raw": r0, "cal": r1})
    print(f"{t:<45}{b:>8.2f}{r0:>11.3f}{r1:>9.3f}{100*(1-r1/r0):>7.1f}%")

# Also at the level the variance actually lives at: the national year effect.
print("\nSame slope computed on NATIONAL YEAR MEANS (where 51-79 % of variance sits):")
for t in TARGETS[:1] + [x for x in TARGETS if "Crop_Incidence" in x]:
    g = M[M.target == t].groupby("Year")[["value", "pred", "clim"]].mean()
    g = g.assign(dp=g.pred - g.clim, dy=g.value - g.clim)
    print(f"  {t:<45}beta {beta_of(g):>6.2f}   corr(dp,dy) {g.dp.corr(g.dy):>6.2f}")


# ---------------------------------------------------------------- as-of version
def asof_calibrated(shrink_n, min_folds=6):
    """At each year T estimate beta from folds < T only, shrunk toward 1.

    beta_used = (S_xy + shrink_n * S_xx / n * 1) / (S_xx + shrink_n * S_xx / n)
    i.e. a pseudo-count of `shrink_n` observations asserting beta == 1.
    """
    out = []
    for t in TARGETS:
        g = M[M.target == t].sort_values("Year")
        for T in EVAL:
            cur, past = g[g.Year == T], g[g.Year < T]
            if len(cur) == 0:
                continue
            n = len(past)
            if n < min_folds * 9:
                b = 1.0
            else:
                sxx, sxy = (past.dp ** 2).sum(), (past.dp * past.dy).sum()
                prior = shrink_n * sxx / n
                b = (sxy + prior) / (sxx + prior) if sxx + prior > 0 else 1.0
                b = float(np.clip(b, 0.0, 2.5))
            out.append(cur.assign(cal=np.clip(cur.clim + b * cur.dp, 0,
                                              100 if t in INCIDENCE else None),
                                  beta=b))
    return pd.concat(out, ignore_index=True)


print("\n" + "=" * 104)
print("AS-OF calibration (beta re-estimated each year from earlier folds only) -- a real forecast")
print("=" * 104)
print(f"{'shrink_n':>9}{'sept 05_25':>12}{'sept 05_19':>12}{'sept 21_25':>12}"
      f"{'rust 05_25':>12}{'rust 05_19':>12}{'rust 21_25':>12}")
_, b_all = score(BT, truth, EVAL)
_, b_19 = score(BT, truth, list(range(2005, 2020)))
_, b_25 = score(BT, truth, [2021, 2022, 2023, 2024, 2025])
print(f"{'raw v2':>9}{b_all['septoria_pooled']:>12.3f}{b_19['septoria_pooled']:>12.3f}"
      f"{b_25['septoria_pooled']:>12.3f}{b_all['rust_pooled']:>12.3f}"
      f"{b_19['rust_pooled']:>12.3f}{b_25['rust_pooled']:>12.3f}")

for sn in [0, 25, 50, 100, 200, 400]:
    c = asof_calibrated(sn)
    p = c[["Year", "Region", "target"]].assign(value=c.cal)
    _, a = score(p, truth, EVAL)
    _, a19 = score(p, truth, list(range(2005, 2020)))
    _, a25 = score(p, truth, [2021, 2022, 2023, 2024, 2025])
    print(f"{sn:>9}{a['septoria_pooled']:>12.3f}{a19['septoria_pooled']:>12.3f}"
          f"{a25['septoria_pooled']:>12.3f}{a['rust_pooled']:>12.3f}"
          f"{a19['rust_pooled']:>12.3f}{a25['rust_pooled']:>12.3f}")

c = asof_calibrated(100)
print("\nbeta actually used at each year (shrink_n=100), septoria L1 incidence:")
s = c[c.target == "L1_Zymoseptoria_tritici_Crop_Incidence"].groupby("Year").beta.first()
print("  " + "  ".join(f"{y}:{v:.2f}" for y, v in s.items()))
