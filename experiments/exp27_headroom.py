"""Experiment 27: where is the remaining error, and what is it worth chasing?

After three consecutive negative results (cross-disease features, anomaly
recalibration, recency weighting) the useful question stops being "what else can
I bolt on" and becomes "how much error is left of each KIND, and which kinds are
even addressable".

Every prediction error decomposes into two parts that call for completely
different work:

  YEAR-LEVEL error   the national mean for that year is wrong. This is a
                     forecasting problem -- better weather signal, better regime
                     handling. Addressable in principle.

  WITHIN-YEAR error  given the national mean, the spread ACROSS REGIONS is
                     wrong. Addressable only with region-resolved drivers
                     (per-region weather, variety mix, fungicide use).

So: RMSE^2 = (year-level MSE) + (within-year MSE), computed by comparing each
prediction against the actual national mean of that year.

Three reference points are computed for each:
  * the current model
  * climatology (the do-nothing benchmark)
  * ORACLE-YEAR -- the model given the true national mean for each year, with
    only its regional deviations kept. This is the score a perfect year
    forecaster would get, i.e. the hard ceiling on all weather-signal work.
  * ORACLE-REGION -- true regional pattern, model's year level. The ceiling on
    all region-resolution work (the stalled ERA5 download, variety data).

Whichever ceiling is closer to the current score is the one that is nearly used
up, and effort should go to the other.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

from common import TARGETS, SEPTORIA, RUST, INCIDENCE, to_long, score, rmse
import final_model_v2 as FM

EVAL = [y for y in range(2005, 2026) if y != 2020]
truth = to_long(FM.OBS)


def clim_fn(t, T):
    p, te = FM._fit(t, T, "clim", "fix")
    return None if p is None else te[["Year", "Region"]].assign(target=t, value=p)


print("running backtest ...", flush=True)
BT = FM.run(EVAL)
CL = FM.run(EVAL, clim_fn)

M = (truth[truth.Year.isin(EVAL)]
     .merge(BT.rename(columns={"value": "pred"}), on=["Year", "Region", "target"])
     .merge(CL.rename(columns={"value": "clim"}), on=["Year", "Region", "target"])
     .dropna(subset=["value"]))

# national mean per (target, year), for truth and for the model
M["y_true"] = M.groupby(["target", "Year"]).value.transform("mean")
M["y_pred"] = M.groupby(["target", "Year"]).pred.transform("mean")
M["y_clim"] = M.groupby(["target", "Year"]).clim.transform("mean")

# oracle variants: swap in the true year level / true regional deviation
M["oracle_year"] = M.y_true + (M.pred - M.y_pred)      # perfect year forecast
M["oracle_reg"] = M.y_pred + (M.value - M.y_true)      # perfect regional pattern


def pooled(cols, a, b):
    s = M[M.target.isin(cols)]
    return rmse(s[a], s[b])


print("\n" + "=" * 100)
print("ERROR DECOMPOSITION, pooled per disease, 2005-2025")
print("=" * 100)
print(f"{'':<22}{'total RMSE':>12}{'year-level':>12}{'within-year':>13}{'% of MSE':>11}")
for name, cols in [("SEPTORIA", SEPTORIA), ("YELLOW RUST", RUST)]:
    s = M[M.target.isin(cols)]
    for lab, col in [("model", "pred"), ("climatology", "clim")]:
        tot = rmse(s.value, s[col])
        yr = rmse(s.y_true, s[f"y_{'pred' if col == 'pred' else 'clim'}"])
        wi = np.sqrt(max(tot ** 2 - yr ** 2, 0))
        print(f"{name + ' ' + lab:<22}{tot:>12.3f}{yr:>12.3f}{wi:>13.3f}"
              f"{100 * yr ** 2 / tot ** 2:>10.0f}%")
    print()

print("=" * 100)
print("CEILINGS -- what perfect information of each kind would be worth")
print("=" * 100)
print(f"{'':<22}{'septoria':>12}{'vs now':>10}{'rust':>12}{'vs now':>10}")
now_s, now_r = pooled(SEPTORIA, "value", "pred"), pooled(RUST, "value", "pred")
rows = [("current model", "pred"), ("climatology", "clim"),
        ("+ perfect YEAR level", "oracle_year"),
        ("+ perfect REGION pattern", "oracle_reg")]
for lab, col in rows:
    s, r = pooled(SEPTORIA, "value", col), pooled(RUST, "value", col)
    print(f"{lab:<22}{s:>12.3f}{s - now_s:>+10.3f}{r:>12.3f}{r - now_r:>+10.3f}")

print("\n" + "=" * 100)
print("How much of the ADDRESSABLE gap is already captured?")
print("=" * 100)
for name, cols in [("septoria", SEPTORIA), ("rust", RUST)]:
    cl = pooled(cols, "value", "clim")
    now = pooled(cols, "value", "pred")
    oy = pooled(cols, "value", "oracle_year")
    orr = pooled(cols, "value", "oracle_reg")
    print(f"\n{name}:  climatology {cl:.2f} -> model {now:.2f}")
    print(f"   year-effect work:   ceiling {oy:.2f}, "
          f"captured {100*(cl-now)/max(cl-oy,1e-9):.0f}% of the {cl-oy:.2f} available")
    print(f"   region-detail work: ceiling {orr:.2f}, "
          f"remaining headroom {now-orr:.2f}")

# per-year national miss, to see whether the year errors are a few bad years
print("\n" + "=" * 100)
print("National-mean miss by year, septoria L1 incidence (model vs climatology)")
print("=" * 100)
t = "L1_Zymoseptoria_tritici_Crop_Incidence"
g = M[M.target == t].groupby("Year")[["y_true", "y_pred", "y_clim"]].first()
g["model_err"] = g.y_pred - g.y_true
g["clim_err"] = g.y_clim - g.y_true
print(g.round(1).to_string())
print(f"\nmodel  RMSE of year level: {rmse(g.y_true, g.y_pred):.2f}")
print(f"clim   RMSE of year level: {rmse(g.y_true, g.y_clim):.2f}")
