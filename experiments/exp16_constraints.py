"""Experiment 16: enforce the L2 >= L1 structural constraint.

Leaf 2 emerges before the flag leaf (leaf 1), sits lower in the canopy and so
nearer the splash-dispersed inoculum, and has been exposed for longer. Both
diseases should therefore always be at least as prevalent and severe on L2 as
on L1. The 8 targets are fitted independently, so nothing enforces this, and
the 2026 forecast produced East L2 septoria severity = 0.00 against L1 = 0.24 --
structurally impossible.

This checks the constraint holds in the observed data, then tests whether
imposing it (isotonic repair of each L1/L2 pair) improves backtest RMSE.
"""
import warnings
import numpy as np
import pandas as pd

from common import (TARGETS, SEVERITY, INCIDENCE, SEPTORIA, RUST, load_pest,
                    rmse, to_long, score)
import final_model as FM

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

OBS = FM.OBS
PAIRS = [("L1_Zymoseptoria_tritici_Disease_Severity", "L2_Zymoseptoria_tritici_Disease_Severity"),
         ("L1_Zymoseptoria_tritici_Crop_Incidence", "L2_Zymoseptoria_tritici_Crop_Incidence"),
         ("L1_Yellow_rust_Disease_Severity", "L2_Yellow_rust_Disease_Severity"),
         ("L1_Yellow_rust_Crop_Incidence", "L2_Yellow_rust_Crop_Incidence")]

print("=" * 96)
print("Does L2 >= L1 hold in the observed data? (1971-2025)")
print("=" * 96)
for a, b in PAIRS:
    d = OBS[[a, b]].dropna()
    frac = (d[b] >= d[a]).mean()
    viol = d[d[b] < d[a]]
    print(f"{a.replace('L1_',''):<38} L2>=L1 in {100*frac:5.1f}% of {len(d)} rows"
          f"   mean L1={d[a].mean():7.3f}  mean L2={d[b].mean():7.3f}"
          f"   worst violation={-(viol[b]-viol[a]).min() if len(viol) else 0:.3f}")


def enforce(preds):
    """Repair each L1/L2 pair: if L2 < L1, replace both with their mean
    (the least-squares projection onto the constraint set L2 >= L1)."""
    w = preds.pivot_table(index=["Year", "Region"], columns="target", values="value",
                          observed=True)
    for a, b in PAIRS:
        if a not in w.columns or b not in w.columns:
            continue
        bad = w[b] < w[a]
        mid = (w.loc[bad, a] + w.loc[bad, b]) / 2
        w.loc[bad, a] = mid
        w.loc[bad, b] = mid
    out = w.reset_index().melt(id_vars=["Year", "Region"], var_name="target",
                               value_name="value")
    return out.dropna(subset=["value"])


if __name__ == "__main__":
    EVAL = [y for y in range(2005, 2026) if y != 2020]
    truth = to_long(OBS)
    bt = FM.run(EVAL)
    bt_c = enforce(bt)

    n_fixed = 0
    w = bt.pivot_table(index=["Year", "Region"], columns="target", values="value", observed=True)
    for a, b in PAIRS:
        n_fixed += int((w[b] < w[a]).sum())
    print(f"\nconstraint violations in backtest predictions: {n_fixed} of {len(w)*len(PAIRS)}")

    print("\n" + "=" * 96)
    print("RMSE with vs without the L2>=L1 constraint")
    print("=" * 96)
    for label, yrs in [("2005-2025", EVAL), ("2005-2019", list(range(2005, 2020))),
                       ("2021-2025", [2021, 2022, 2023, 2024, 2025])]:
        _, a = score(bt, truth, yrs)
        _, c = score(bt_c, truth, yrs)
        print(f"  {label}:  septoria {a['septoria_pooled']:7.4f} -> {c['septoria_pooled']:7.4f}"
              f"   |  rust {a['rust_pooled']:7.4f} -> {c['rust_pooled']:7.4f}")

    pt, _ = score(bt, truth, EVAL)
    ptc, _ = score(bt_c, truth, EVAL)
    print("\nper-target (2005-2025):")
    for t in TARGETS:
        print(f"  {t:<45}{pt[t]:>9.4f} -> {ptc[t]:>9.4f}")

    print("\n" + "=" * 96)
    print("2026 forecast, before vs after constraint")
    print("=" * 96)
    fc = FM.run([2026])
    fcc = enforce(fc)
    m = fc.merge(fcc, on=["Year", "Region", "target"], suffixes=("_raw", "_fix"))
    chg = m[np.abs(m.value_raw - m.value_fix) > 1e-9]
    if len(chg):
        print(chg[["Region", "target", "value_raw", "value_fix"]].round(3).to_string(index=False))
    else:
        print("  (no changes)")
