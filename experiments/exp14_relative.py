"""Experiment 14: relative (multiplicative) modelling + drought x baseline.

exp13 showed the remaining big septoria misses are extreme years where the
weather signal and the recent baseline disagree -- above all 2025 (spring
dryness +5.1 SD, actual L1 incidence 35.0, predicted 53.6). The additive model
lets the elevated 2020s baseline hold the prediction up even when the weather
says "collapse".

2026 is itself a drought year (dryness +3.9 SD), so this is exactly the regime
that will be scored. Two candidate fixes:

  REL   model the RATIO y / baseline, so weather acts multiplicatively on the
        prevailing level rather than being added to it;
  INT   keep the additive model but add drought x baseline interactions, so a
        dry spring is allowed to cancel a high baseline.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from common import (TARGETS, SEVERITY, INCIDENCE, SEPTORIA, RUST, load_pest,
                    rmse, to_long, score)
from features import build_weather_features
from exp09_baseline_feats import as_of_baselines

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

SEPT_FEATS = ["e_rain_spring_anom", "e_rain_apr_may_anom", "e_rdays_latespring_anom",
              "e_sun_apr_may_anom", "e_tmean_spring_anom", "e_frost_winter_anom",
              "e_wetness_idx_anom", "e_dry_bright_spring_anom"]
RUST_FEATS = ["e_tmin_winter_anom", "e_frost_winter_anom", "e_tmean_jun_anom",
              "e_tmean_spring_anom", "e_rain_spring_anom", "e_mild_winter_anom"]
NL_RUST = ["nl_junheat", "nl_mildwin", "nl_drought"]

PEST = load_pest()
OBS = PEST[PEST.Year <= 2025].reset_index(drop=True)
WF = build_weather_features()
BL = as_of_baselines(OBS, TARGETS)
DF = OBS.merge(WF, on=["Year", "Region"], how="left").merge(BL, on=["Year", "Region"], how="left")
DF["trend"] = (DF.Year - 2000) / 25.0
DF["nl_drought"] = np.clip(DF["e_sun_apr_may_anom"] - 1.0, 0, None)
DF["nl_dry2"] = np.clip(-DF["e_rain_apr_may_anom"] - 0.75, 0, None)
DF["nl_junheat"] = np.clip(DF["e_tmean_jun_anom"] - 1.0, 0, None)
DF["nl_mildwin"] = np.clip(-DF["e_frost_winter_anom"] - 0.5, 0, None)
REGIONS = sorted(OBS.Region.unique())
EVAL = [y for y in range(2005, 2026) if y != 2020]
NATW = WF.groupby("Year")[["e_rain_apr_may_anom", "e_sun_apr_may_anom"]].mean()
NATW["dryness"] = NATW.e_sun_apr_may_anom - NATW.e_rain_apr_may_anom
DRY = [y for y in EVAL if NATW.dryness.get(y, 0) >= NATW.dryness.reindex(EVAL).quantile(0.7)]


def base_cols(t):
    return [f"bl_reg4_{t}", f"bl_reg10_{t}", f"bl_nat4_{t}", f"bl_nat10_{t}"]


def predict(target, T, mode):
    tr = DF[(DF.Year < T) & (DF.Year >= 1990) & DF[target].notna()].copy()
    te = DF[DF.Year == T].copy()
    if len(te) == 0 or len(tr) < 30:
        return None
    cy = sorted(tr.Year.unique())[-12:]
    cl = tr[tr.Year.isin(cy)]
    nc, rc = cl[target].mean(), cl.groupby("Region")[target].mean()
    base = 0.5 * te.Region.map(rc).fillna(nc).to_numpy() + 0.5 * nc
    if mode == "clim":
        return te[["Year", "Region"]].assign(target=target, value=base)

    wcols = list(SEPT_FEATS if target in SEPTORIA else RUST_FEATS)
    if target in RUST:
        wcols += NL_RUST
    cols = wcols + base_cols(target) + ["trend"]

    if mode == "int":
        # drought x prevailing level: let a dry spring cancel a high baseline
        for df_ in (tr, te):
            df_["ix_dr_bl"] = df_["nl_drought"] * df_[f"bl_nat4_{target}"]
            df_["ix_dry_bl"] = df_["nl_dry2"] * df_[f"bl_nat4_{target}"]
            df_["ix_rain_bl"] = df_["e_rain_apr_may_anom"] * df_[f"bl_nat4_{target}"]
        cols = cols + ["ix_dr_bl", "ix_dry_bl", "ix_rain_bl"]

    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med).fillna(0.0), Xte.fillna(med).fillna(0.0)
    for r in REGIONS[1:]:
        Xtr[f"R_{r}"] = (tr.Region == r).astype(float)
        Xte[f"R_{r}"] = (te.Region == r).astype(float)
    sc = StandardScaler().fit(Xtr)

    if mode == "rel":
        # model log(y / baseline): weather multiplies the prevailing level
        b_tr = tr[f"bl_reg10_{target}"].fillna(tr[f"bl_nat10_{target}"])
        b_te = te[f"bl_reg10_{target}"].fillna(te[f"bl_nat10_{target}"])
        eps = 0.05 * max(tr[target].mean(), 1e-3)
        z = np.log((tr[target].to_numpy(float) + eps) / (b_tr.to_numpy(float) + eps))
        z = np.clip(z, -4, 4)
        m = Ridge(alpha=100.0).fit(sc.transform(Xtr), z)
        pv = (b_te.to_numpy(float) + eps) * np.exp(np.clip(m.predict(sc.transform(Xte)), -4, 4)) - eps
    else:
        m = Ridge(alpha=100.0).fit(sc.transform(Xtr), tr[target].to_numpy(float))
        pv = m.predict(sc.transform(Xte))
    pv = np.clip(pv, 0, 100 if target in INCIDENCE else None)
    return te[["Year", "Region"]].assign(target=target, value=pv)


def run(mode, targets=TARGETS, years=EVAL):
    out = [p for t in targets for T in years if (p := predict(t, T, mode)) is not None]
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    truth = to_long(OBS)
    modes = ["clim", "add", "rel", "int"]
    store = {m: run(m) for m in modes}

    print("=" * 116)
    print("ADDITIVE vs RELATIVE vs INTERACTION")
    print("=" * 116)
    rows = []
    for lab, yrs in [("05_25", EVAL), ("05_19", list(range(2005, 2020))),
                     ("21_25", [2021, 2022, 2023, 2024, 2025]), ("DRY", DRY)]:
        rec = {"window": lab}
        for m in modes:
            _, s = score(store[m], truth, yrs)
            rec[f"{m}_sept"] = s["septoria_pooled"]
            rec[f"{m}_rust"] = s["rust_pooled"]
        rows.append(rec)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\nNational septoria L1 incidence: actual vs each mode")
    t = "L1_Zymoseptoria_tritici_Crop_Incidence"
    d = pd.DataFrame({"actual": OBS[OBS.Year.isin(EVAL)].groupby("Year")[t].mean()})
    for m in modes:
        d[m] = store[m][store[m].target == t].groupby("Year").value.mean()
    d["dryness"] = NATW.dryness.reindex(d.index)
    print(d.round(1).to_string())

    print("\nper-target RMSE 2005-2025:")
    out = {"target": TARGETS}
    for m in modes:
        pt, _ = score(store[m], truth, EVAL)
        out[m] = [pt[t] for t in TARGETS]
    print(pd.DataFrame(out).round(4).to_string(index=False))
