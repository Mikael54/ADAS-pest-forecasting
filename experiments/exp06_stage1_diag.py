"""Experiment 06: how well can the NATIONAL year level actually be predicted?

exp05's two-stage model underperformed, which means stage 1 is weak. This
isolates stage 1: rolling-origin prediction of the national mean of each target,
scored against the climatology benchmark, plus a look at whether the
weather->disease response is linear or threshold-like.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from common import TARGETS, SEVERITY, INCIDENCE, load_pest, rmse
from features import build_weather_features

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)

WF = build_weather_features()
PEST = load_pest()
EPI_ANOM = [c for c in WF.columns if c.startswith("e_") and c.endswith("_anom")]
NATW = WF.groupby("Year")[EPI_ANOM].mean()
NAT = PEST[PEST.Year <= 2025].groupby("Year")[TARGETS].mean()

SEPT_FEATS = ["e_rain_spring_anom", "e_rain_apr_may_anom", "e_rdays_latespring_anom",
              "e_sun_apr_may_anom", "e_tmean_spring_anom", "e_frost_winter_anom",
              "e_wetness_idx_anom", "e_dry_bright_spring_anom"]
RUST_FEATS = ["e_tmin_winter_anom", "e_frost_winter_anom", "e_tmean_jun_anom",
              "e_tmax_jun_anom", "e_tmean_spring_anom", "e_rain_spring_anom",
              "e_mild_winter_anom", "e_rust_window_anom"]

EVAL = [y for y in range(2005, 2026) if y != 2020]


def rolling_stage1(target, cols, alpha, transform, min_year=1971):
    preds, actuals, clim = [], [], []
    for T in EVAL:
        tr = NAT[(NAT.index < T) & (NAT.index >= min_year)][target].dropna()
        if T not in NAT.index or np.isnan(NAT.loc[T, target]):
            continue
        X = NATW.reindex(tr.index)[cols]
        ok = X.notna().all(axis=1)
        X, y = X[ok], tr[ok]
        z = (np.log1p(y) if target in SEVERITY else
             np.log(np.clip(y/100, .005, .995)/(1-np.clip(y/100, .005, .995)))) \
            if transform else y.to_numpy(float)
        m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X, z)
        xp = NATW.reindex([T])[cols].fillna(X.median())
        pz = m.predict(xp)[0]
        if transform:
            pv = np.expm1(pz) if target in SEVERITY else 100/(1+np.exp(-np.clip(pz, -12, 12)))
        else:
            pv = pz
        pv = np.clip(pv, 0, 100 if target in INCIDENCE else None)
        preds.append(pv); actuals.append(NAT.loc[T, target]); clim.append(y.tail(12).mean())
    return np.array(preds), np.array(actuals), np.array(clim)


print("=" * 104)
print("STAGE-1 SKILL: rolling-origin prediction of the NATIONAL year mean, 2005-2025")
print("=" * 104)
print(f"{'target':<43}{'model':>9}{'clim':>9}{'skill%':>9}{'corr':>8}")
for t in TARGETS:
    cols = SEPT_FEATS if "Zymoseptoria" in t else RUST_FEATS
    best = None
    for alpha in [1, 3, 10, 30, 100]:
        for tr_ in [False, True]:
            p, a, c = rolling_stage1(t, cols, alpha, tr_)
            r_m, r_c = rmse(a, p), rmse(a, c)
            if best is None or r_m < best[0]:
                best = (r_m, r_c, alpha, tr_, np.corrcoef(a, p)[0, 1])
    r_m, r_c, alpha, tr_, cr = best
    print(f"{t:<43}{r_m:>9.3f}{r_c:>9.3f}{100*(1-r_m/r_c):>8.0f}%{cr:>8.2f}"
          f"   (best alpha={alpha}, transform={tr_})")

print()
print("=" * 104)
print("Is the weather response LINEAR? national septoria L1 incidence vs spring rain anomaly")
print("=" * 104)
d = pd.DataFrame({
    "rain_spring_anom": NATW["e_rain_spring_anom"],
    "sun_aprmay_anom": NATW["e_sun_apr_may_anom"],
    "tmean_spring_anom": NATW["e_tmean_spring_anom"],
}).join(NAT[["L1_Zymoseptoria_tritici_Crop_Incidence",
             "L1_Zymoseptoria_tritici_Disease_Severity",
             "L2_Yellow_rust_Crop_Incidence"]]).dropna()
d["decile"] = pd.qcut(d.rain_spring_anom, 5, labels=False)
print(d.groupby("decile").agg(
    n=("rain_spring_anom", "size"),
    rain=("rain_spring_anom", "mean"),
    sept_inc=("L1_Zymoseptoria_tritici_Crop_Incidence", "mean"),
    sept_sev=("L1_Zymoseptoria_tritici_Disease_Severity", "mean"),
).round(2).to_string())

print("\nSame, binned by Apr-May SUNSHINE anomaly (the strongest single septoria signal):")
d["sdec"] = pd.qcut(d.sun_aprmay_anom, 5, labels=False)
print(d.groupby("sdec").agg(
    n=("sun_aprmay_anom", "size"), sun=("sun_aprmay_anom", "mean"),
    sept_inc=("L1_Zymoseptoria_tritici_Crop_Incidence", "mean"),
    sept_sev=("L1_Zymoseptoria_tritici_Disease_Severity", "mean"),
).round(2).to_string())

print()
print("=" * 104)
print("NON-STATIONARITY: national means by decade (is the historical era even relevant?)")
print("=" * 104)
nn = NAT.copy(); nn["decade"] = (nn.index // 10) * 10
print(nn.groupby("decade")[TARGETS].mean().round(2).to_string())

print()
print("2026 national weather anomalies (what the model will see):")
print(NATW.reindex([2021, 2022, 2023, 2024, 2025, 2026])[
    ["e_rain_spring_anom", "e_rain_apr_may_anom", "e_sun_apr_may_anom",
     "e_rdays_latespring_anom", "e_tmean_spring_anom", "e_tmin_winter_anom",
     "e_frost_winter_anom", "e_tmean_jun_anom"]].round(2).to_string())
