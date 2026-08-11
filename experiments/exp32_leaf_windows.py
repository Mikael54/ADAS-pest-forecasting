"""Experiment 32: give L1 and L2 the weather windows they were actually exposed to.

THE OVERSIGHT
Leaf 2 emerges around 25 April; leaf 1 (the flag leaf) around 15 May. They are
therefore exposed to different weather, and septoria is splash-dispersed UP the
canopy, so what matters for each leaf is rain falling AFTER it emerged. exp17
encoded this properly in the mechanistic block:

    EXPOSURE_L2 = {Apr: 0.2, May: 1.0, Jun: 1.0}
    EXPOSURE_L1 = {         May: 0.5, Jun: 1.0}

and built m_sept_L1/L2 and m_splash_L1/L2 from it. But the model then hands the
SAME feature list to all four septoria targets: M_SEPT contains m_sept_L1_ranom,
m_sept_L2_ranom and m_splash_L2_ranom for every target, and m_splash_L1_ranom is
not used at all. So an L1 target is fitted partly on L2's exposure window and
vice versa, and ridge has to sort it out from ~450 rows.

The statistical block has the same problem more crudely: e_rain_apr_may is an
L2-appropriate window applied to L1 targets too, and there is no May-June
aggregate at all.

WHAT THIS CHANGES
  mechanistic: L1 targets get m_sept_L1 / m_splash_L1 / m_liebig_L1;
               L2 targets get m_sept_L2 / m_splash_L2 / m_liebig_L2.
               (m_liebig_L1 has to be built -- only the L2 one existed.)
  statistical: L2 keeps Apr-May moisture and sunshine;
               L1 gets new May-Jun equivalents.

This is a pure re-allocation of existing information, not new data, and it
REDUCES the feature count per target. If the epidemiology is right it should
help; if ridge was already handling it, it will be neutral.

ADOPTION RULE: improve both 2005-2019 and 2021-2025 (as exp26/28/31).
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

from common import TARGETS, SEPTORIA, RUST, to_long, score
from features import season_monthly, epi_features, add_climatology_anomalies
from exp18_rolling_mech import rolling_anomalies, M_SEPT, E_ROLL
import final_model_v2 as FM

EVAL = [y for y in range(2005, 2026) if y != 2020]
W19 = list(range(2005, 2020))
W25 = [2021, 2022, 2023, 2024, 2025]
truth = to_long(FM.OBS)


# ---------------------------------------------------------------- extra features
def _sum(w, var, months):
    return sum(w[f"w_{var}_{m}"] for m in months)


def _mean(w, var, months):
    return sum(w[f"w_{var}_{m}"] for m in months) / len(months)


def extra_features():
    """May-June statistical windows (the L1 exposure period) + m_liebig_L1."""
    w = season_monthly()
    f = pd.DataFrame({"Region": w.Region, "Year": w.Year})
    MJ = ["s05", "s06"]
    f["e_rain_may_jun"] = _sum(w, "Rainfall", MJ)
    f["e_rdays_may_jun"] = _sum(w, "Raindays1mm", MJ)
    f["e_sun_may_jun"] = _sum(w, "Sunshine", MJ)
    f["e_tmean_may_jun"] = _mean(w, "Tmean", MJ)
    f["e_wetness_may_jun"] = f["e_rdays_may_jun"] / (f["e_sun_may_jun"] / 100 + 1)
    f = add_climatology_anomalies(f)
    f = rolling_anomalies(f, "e_")
    keep = [c for c in f.columns if c.endswith(("_anom", "_ranom"))]
    return f[["Region", "Year"] + keep]


NEW = extra_features()
DF = FM.DF.merge(NEW, on=["Year", "Region"], how="left")

# m_liebig_L1: the limiting-factor index on L1's exposure window. Only the L2
# version was ever built. Same construction -- min of STANDARDISED components.
if {"m_cyc_spring_ranom", "m_splash_L1_ranom"} <= set(DF.columns):
    DF["m_liebig_L1_ranom"] = np.minimum(DF["m_cyc_spring_ranom"],
                                         DF["m_splash_L1_ranom"])
for c in [c for c in DF.columns if c.endswith(("_anom", "_ranom"))]:
    DF[c] = DF[c].clip(-5, 5)
FM.DF = DF

MISSING = [c for c in ["m_splash_L1_ranom", "m_liebig_L1_ranom",
                       "e_rain_may_jun_ranom"] if c not in DF.columns]
print(f"added {len(NEW.columns) - 2} statistical cols; missing: {MISSING or 'none'}")

# ---------------------------------------------------------------- column choice
E_FIX = FM.E_FIX
_orig = FM.weather_cols


def leaf_of(target):
    return "L1" if target.startswith("L1_") else "L2"


def have(cols):
    return [c for c in cols if c in DF.columns]


def mech_leaf(target):
    """Mechanistic block with only this leaf's exposure terms."""
    leaf = leaf_of(target)
    other = "L2" if leaf == "L1" else "L1"
    block = [c for c in M_SEPT if f"_{other}_" not in c]
    block += [f"m_sept_{leaf}_ranom", f"m_splash_{leaf}_ranom",
              f"m_liebig_{leaf}_ranom"]
    return have(list(dict.fromkeys(block)))


def stat_leaf(target, roll):
    """Statistical block with this leaf's moisture/sunshine window."""
    sfx = "_ranom" if roll else "_anom"
    base = [c for c in (E_ROLL if roll else E_FIX)
            if "apr_may" not in c and "latespring" not in c and "wetness" not in c]
    if leaf_of(target) == "L1":
        win = [f"e_rain_may_jun{sfx}", f"e_sun_may_jun{sfx}",
               f"e_rdays_may_jun{sfx}", f"e_wetness_may_jun{sfx}"]
    else:
        win = [f"e_rain_apr_may{sfx}", f"e_sun_apr_may{sfx}",
               f"e_rdays_latespring{sfx}", f"e_wetness_idx{sfx}"]
    return have(list(dict.fromkeys(base + win)))


def make_cols(mech_on, stat_on):
    def fn(target, basis):
        if target not in SEPTORIA:
            return _orig(target, basis)
        if basis == "mech":
            return mech_leaf(target) if mech_on else _orig(target, basis)
        if basis in ("fix", "roll") and stat_on:
            return stat_leaf(target, basis == "roll")
        return _orig(target, basis)
    return fn


def evaluate(fn, label):
    FM.weather_cols = fn
    try:
        p = FM.run(EVAL)
    finally:
        FM.weather_cols = _orig
    rec = {"variant": label}
    for lab, yrs in [("05_25", EVAL), ("05_19", W19), ("21_25", W25)]:
        _, s = score(p, truth, yrs)
        rec[f"sept_{lab}"] = s["septoria_pooled"]
        rec[f"rust_{lab}"] = s["rust_pooled"]
    for t in ["L1_Zymoseptoria_tritici_Crop_Incidence",
              "L2_Zymoseptoria_tritici_Crop_Incidence"]:
        pt, _ = score(p, truth, EVAL)
        rec[t[:2] + "_inc"] = pt[t]
    return rec


if __name__ == "__main__":
    variants = [
        (_orig, "v2 shipped"),
        (make_cols(True, False), "leaf-aware mechanistic"),
        (make_cols(False, True), "leaf-aware statistical"),
        (make_cols(True, True), "leaf-aware both"),
    ]
    rows = []
    for fn, lab in variants:
        rows.append(evaluate(fn, lab))
        print(f"  done {lab}", flush=True)

    df = pd.DataFrame(rows)
    ref = df.iloc[0]
    df["d_19"] = df.sept_05_19 - ref.sept_05_19
    df["d_25"] = df.sept_21_25 - ref.sept_21_25
    df["both"] = np.where((df.d_19 < 0) & (df.d_25 < 0), "YES", "")
    print("\n" + "=" * 124)
    print("LEAF-SPECIFIC EXPOSURE WINDOWS  (rust untouched, shown as a control)")
    print("=" * 124)
    print(df[["variant", "sept_05_25", "sept_05_19", "sept_21_25", "both",
              "L1_inc", "L2_inc", "rust_05_25"]].round(3).to_string(index=False))
