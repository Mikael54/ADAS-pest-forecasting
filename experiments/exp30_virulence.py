"""Experiment 30: does UKCPVS virulence data explain yellow rust?

exp27 said rust has captured only 21 % of its available year-effect gap because
its driver is the pathogen population, not the weather. fetch_ukcpvs.py now
supplies the pathogen population: % of UKCPVS isolates virulent on each Yr gene
and named variety, 2010-2024.

⚠️ POWER WARNING, STATED UP FRONT
The series is 15 annual values. Lagged by one year and intersected with the eval
window it leaves ~14 usable year-pairs. A correlation needs |r| > 0.53 to clear
p < 0.05 at n = 14. So this experiment can only detect a LARGE effect, and any
modest-looking correlation here is indistinguishable from noise. It is sequenced
correlation-first precisely so that a full backtest is only run if something
survives -- otherwise the backtest would just be an expensive way to fit noise.

THE INFORMATION RULE MATTERS MORE THAN USUAL HERE
UKCPVS isolates for year Y are collected during year Y's epidemic and published
the following year. Using year-Y virulence to predict year-Y disease would be
circular: the isolates ARE that epidemic. Everything here is therefore lagged at
least one year, which is also what a real 2026 forecast would have (the 2024
report is the latest published).

THE SHARP TEST
Correlating virulence with rust incidence is not the question -- the model may
already capture that level through its as-of baselines. The question is whether
virulence explains what the model gets WRONG. So the headline test regresses the
current model's out-of-sample RESIDUAL on lagged virulence. If that correlation
is ~0, this data adds nothing regardless of how well it correlates with the raw
level, and no amount of feature engineering will change that.
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

from common import TARGETS, RUST, INCIDENCE, to_long, score, rmse
import final_model_v2 as FM

ROOT = Path(__file__).resolve().parent.parent
V = pd.read_csv(ROOT / "data_external" / "ukcpvs_virulence.csv", index_col="Year")
EVAL = [y for y in range(2005, 2026) if y != 2020]
truth = to_long(FM.OBS)

# Variety rows describe virulence on actual commercial wheat, which is what the
# crop's susceptibility depends on; Yr-gene rows are mostly pinned at 0 or 100.
GENES = [c for c in V.columns if c.startswith("Yr")]
VARIETIES = [c for c in V.columns if not c.startswith("Yr")
             and "Number" not in c and "Total" not in c]
usable = [c for c in GENES if V[c].std() > 5]

IDX = pd.DataFrame(index=V.index)
IDX["vir_variety_mean"] = V[VARIETIES].mean(axis=1)
IDX["vir_gene_mean"] = V[usable].mean(axis=1)
IDX["vir_warrior"] = V["Warrior"] if "Warrior" in V else np.nan
IDX["vir_breadth"] = (V[VARIETIES] > 50).mean(axis=1) * 100   # varieties mostly beaten
IDX["vir_n_isolates"] = V[[c for c in V.columns if "Number" in c]].mean(axis=1)

print("=" * 104)
print("UKCPVS virulence indices (national, annual)")
print("=" * 104)
print(IDX.round(1).to_string())

# ---------------------------------------------------------------- rust levels
nat = FM.OBS.groupby("Year")[TARGETS].mean()
rust_inc = [t for t in RUST if t in INCIDENCE]

print("\n" + "=" * 104)
print("Correlation with NATIONAL rust level (lag 1 = honest; lag 0 = circular, shown "
      "only as a ceiling)")
print("=" * 104)
print(f"{'index':<20}{'target':<34}{'lag0 (circular)':>17}{'lag1 (usable)':>15}{'n':>5}")
for k in ["vir_variety_mean", "vir_gene_mean", "vir_warrior", "vir_breadth"]:
    for t in rust_inc:
        s = IDX[k].dropna()
        r0 = pd.concat([s, nat[t]], axis=1).dropna()
        s1 = s.copy(); s1.index = s1.index + 1
        r1 = pd.concat([s1.rename(k), nat[t]], axis=1).dropna()
        c0 = r0.corr().iloc[0, 1] if len(r0) > 3 else np.nan
        c1 = r1.corr().iloc[0, 1] if len(r1) > 3 else np.nan
        print(f"{k:<20}{t:<34}{c0:>17.2f}{c1:>15.2f}{len(r1):>5}")

# ------------------------------------------------- THE SHARP TEST: residuals
print("\n" + "=" * 104)
print("THE SHARP TEST -- does lagged virulence explain the model's RESIDUAL?")
print("=" * 104)
print("running backtest ...", flush=True)
BT = FM.run(EVAL)
M = (truth[truth.Year.isin(EVAL)]
     .merge(BT.rename(columns={"value": "pred"}), on=["Year", "Region", "target"])
     .dropna(subset=["value"]))
M["resid"] = M.value - M.pred

print(f"\n{'index':<20}{'target':<34}{'corr(resid, vir_lag1)':>24}{'n yrs':>7}")
hits = []
for k in ["vir_variety_mean", "vir_gene_mean", "vir_warrior", "vir_breadth"]:
    s = IDX[k].dropna(); s.index = s.index + 1
    for t in rust_inc:
        g = M[M.target == t].groupby("Year").resid.mean()
        j = pd.concat([s.rename("v"), g.rename("r")], axis=1).dropna()
        if len(j) < 5:
            continue
        c = j.corr().iloc[0, 1]
        flag = "  <-- would clear p<.05" if abs(c) > 2 / np.sqrt(len(j)) else ""
        hits.append(abs(c))
        print(f"{k:<20}{t:<34}{c:>24.2f}{len(j):>7}{flag}")

print(f"\nlargest |corr| with residual: {max(hits):.2f}  "
      f"(needs ~0.53 at n=14 for p<0.05)")

# For scale: how much rust RMSE could a PERFECT year-level correction buy?
g = M[M.target.isin(rust_inc)]
yr_bias = g.groupby("Year").resid.mean()
per_year_fixed = g.merge(yr_bias.rename("b"), on="Year")
print(f"\nrust incidence RMSE now                : {rmse(g.value, g.pred):.3f}")
print(f"if every YEAR's mean residual were fixed: "
      f"{rmse(per_year_fixed.value, per_year_fixed.pred + per_year_fixed.b):.3f}")
print("(the second number is the ceiling for ANY national annual covariate, "
      "virulence included)")
