"""Experiment 15: final model selection -- ensembles and per-target assignment.

exp14 left three candidate model forms with complementary biases:
  add  additive weather + baseline      -- best in the recent high-disease years
  rel  multiplicative (ratio) form      -- best in drought years, poor in 21-25
  int  additive + drought x baseline    -- best overall, good compromise
and climatology, which still wins for the two yellow-rust SEVERITY targets.

Since the differences between add and int are inside the noise of a 944-point
backtest, this checks whether an ensemble is more robust than picking one, and
fixes the per-target assignment on evidence that is consistent across windows.
"""
import warnings
import numpy as np
import pandas as pd

from common import (TARGETS, SEVERITY, INCIDENCE, SEPTORIA, RUST, load_pest,
                    rmse, to_long, score)
from exp14_relative import run, DRY, EVAL, OBS, NATW

warnings.filterwarnings("ignore")
pd.set_option("display.width", 260)

truth = to_long(OBS)
MODES = ["clim", "add", "rel", "int"]
STORE = {m: run(m) for m in MODES}
KEY = ["Year", "Region", "target"]


def combine(names, weights=None):
    ps = [STORE[n].sort_values(KEY).reset_index(drop=True) for n in names]
    w = np.ones(len(ps)) / len(ps) if weights is None else np.array(weights, float)
    w = w / w.sum()
    v = sum(wi * p.value.to_numpy() for wi, p in zip(w, ps))
    return ps[0][KEY].assign(value=v)


CANDIDATES = {
    **{m: STORE[m] for m in MODES},
    "avg(add,int)": combine(["add", "int"]),
    "avg(add,rel,int)": combine(["add", "rel", "int"]),
    "avg(add,int)+clim.25": combine(["add", "int", "clim"], [0.375, 0.375, 0.25]),
    "avg(int,rel)": combine(["int", "rel"]),
}

WINDOWS = [("05_25", EVAL), ("05_19", list(range(2005, 2020))),
           ("21_25", [2021, 2022, 2023, 2024, 2025]), ("DRY", DRY)]

print("=" * 118)
print("CANDIDATE MODEL FORMS -- pooled RMSE by window")
print("=" * 118)
rows = []
for name, p in CANDIDATES.items():
    rec = {"candidate": name}
    for lab, yrs in WINDOWS:
        _, s = score(p, truth, yrs)
        rec[f"sept_{lab}"] = s["septoria_pooled"]
        rec[f"rust_{lab}"] = s["rust_pooled"]
    rows.append(rec)
res = pd.DataFrame(rows)
print(res[["candidate"] + [f"sept_{l}" for l, _ in WINDOWS]].round(3).to_string(index=False))
print()
print(res[["candidate"] + [f"rust_{l}" for l, _ in WINDOWS]].round(3).to_string(index=False))

# ---- per-target: which candidate is consistently best? --------------------
print("\n" + "=" * 118)
print("PER-TARGET RMSE (2005-2025 / 2005-2019 / 2021-2025)")
print("=" * 118)
names = list(CANDIDATES)
tab = []
for t in TARGETS:
    rec = {"target": t}
    for name in names:
        p = CANDIDATES[name]
        p = p[p.target == t]
        vals = []
        for lab, yrs in WINDOWS[:3]:
            tt = truth[(truth.target == t) & truth.Year.isin(yrs)]
            m = tt.merge(p, on=KEY, suffixes=("_t", "_p")).dropna(subset=["value_t"])
            vals.append(rmse(m.value_t, m.value_p))
        rec[name] = vals[0]
        rec[name + "_19"] = vals[1]
        rec[name + "_25"] = vals[2]
    tab.append(rec)
T = pd.DataFrame(tab)
print(T[["target"] + names].round(4).to_string(index=False))

print("\nwins per window (target -> best candidate):")
for t in TARGETS:
    r = T[T.target == t].iloc[0]
    b_all = min(names, key=lambda n: r[n])
    b_19 = min(names, key=lambda n: r[n + "_19"])
    b_25 = min(names, key=lambda n: r[n + "_25"])
    print(f"  {t:<44} 05_25={b_all:<20} 05_19={b_19:<20} 21_25={b_25}")

# ---- proposed assignment, justified on consistency ------------------------
ASSIGN = {t: ("clim" if (t in RUST and t in SEVERITY) else "int") for t in TARGETS}
hyb = pd.concat([CANDIDATES[ASSIGN[t]][CANDIDATES[ASSIGN[t]].target == t]
                 for t in TARGETS], ignore_index=True)
print("\n" + "=" * 118)
print("PROPOSED: 'int' everywhere EXCEPT the two rust severity targets -> climatology")
print("=" * 118)
for lab, yrs in WINDOWS:
    _, s = score(hyb, truth, yrs)
    _, c = score(STORE["clim"], truth, yrs)
    print(f"  {lab:6s} septoria {s['septoria_pooled']:7.3f} (clim {c['septoria_pooled']:7.3f})"
          f"   rust {s['rust_pooled']:7.3f} (clim {c['rust_pooled']:7.3f})")
