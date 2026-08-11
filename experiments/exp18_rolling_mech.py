"""Experiment 18: rolling climatology + limiting-factor mechanistic indices.

exp17 exposed two flaws in the mechanistic index:

 1. FIXED 1961-2000 anomaly baseline + a warming climate means the degree-day
    term drifts upward every year, so "cycles" anomalies are inflated for all
    recent years regardless of moisture. 2026 scored +1.48 on septoria
    potential despite a drought. Fix: express anomalies against a TRAILING
    30-year climatology, which is both standard meteorological practice and
    strictly backward-looking.

 2. PRODUCT form (cycles x splash) lets warmth compensate for dryness. Septoria
    needs both, and in a UK spring moisture is essentially always the limiting
    factor. Fix: also build a Liebig limiting-factor form, min(cycles, splash).

Tests fixed vs rolling baseline, and statistical vs mechanistic vs both.
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from common import (TARGETS, SEVERITY, INCIDENCE, SEPTORIA, RUST, load_pest,
                    rmse, to_long, score)
from features import season_monthly, epi_features
from exp09_baseline_feats import as_of_baselines
from exp17_mechanistic import mechanistic_features

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)


def rolling_anomalies(f, prefix, window=30, min_years=15):
    """Anomaly vs each region's TRAILING `window`-year climatology.

    Uses only years strictly before the row's own year, so it is valid at any
    forecast origin and removes the warming trend from level comparisons.
    """
    cols = [c for c in f.columns if c.startswith(prefix)]
    f = f.sort_values(["Region", "Year"]).reset_index(drop=True)
    out = f.copy()
    for c in cols:
        mu = (f.groupby("Region")[c]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=min_years).mean()))
        sd = (f.groupby("Region")[c]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=min_years).std()))
        out[c + "_ranom"] = (f[c] - mu) / sd.replace(0, np.nan)
    return out


def build_features():
    from features import add_climatology_anomalies
    # --- statistical epi block: keep BOTH fixed and rolling baselines ---
    w = season_monthly()
    e = epi_features(w)
    e = add_climatology_anomalies(e)        # -> *_anom  (fixed 1961-2000)
    e = rolling_anomalies(e, "e_")          # -> *_ranom (trailing 30y)
    e = e[["Region", "Year"] + [c for c in e.columns
                                if c.endswith(("_anom", "_ranom"))]]
    # --- mechanistic block ---
    m = mechanistic_features()
    m["m_moist_limited"] = m["m_splash_spring"] / (1 + m["m_sun_per_rainday"])
    m = rolling_anomalies(m, "m_")
    # Liebig limiting-factor form: build it from STANDARDISED components, so the
    # minimum reflects which resource is scarce rather than which happens to be
    # measured in bigger units. (Raw cycles ~4 vs raw splash-days ~40 meant the
    # min was always the thermal term, making the index track warming only.)
    m["m_liebig_spring_ranom"] = np.minimum(m["m_cyc_spring_ranom"],
                                            m["m_splash_spring_ranom"])
    m["m_liebig_L2_ranom"] = np.minimum(m["m_cyc_spring_ranom"],
                                        m["m_splash_L2_ranom"])
    return e.merge(m, on=["Region", "Year"], how="outer")


FEAT = build_features()
PEST = load_pest()
OBS = PEST[PEST.Year <= 2025].reset_index(drop=True)
REGIONS = sorted(OBS.Region.unique())
BL = as_of_baselines(PEST.reset_index(drop=True), TARGETS)
DF = PEST.reset_index(drop=True).merge(FEAT, on=["Year", "Region"], how="left") \
                                .merge(BL, on=["Year", "Region"], how="left")
DF["trend"] = (DF.Year - 2000) / 25.0

E_FIX = ["e_rain_spring_anom", "e_rain_apr_may_anom", "e_rdays_latespring_anom",
         "e_sun_apr_may_anom", "e_tmean_spring_anom", "e_frost_winter_anom",
         "e_wetness_idx_anom", "e_dry_bright_spring_anom"]
E_ROLL = [c.replace("_anom", "_ranom") for c in E_FIX]
R_FIX = ["e_tmin_winter_anom", "e_frost_winter_anom", "e_tmean_jun_anom",
         "e_tmean_spring_anom", "e_rain_spring_anom", "e_mild_winter_anom"]
R_ROLL = [c.replace("_anom", "_ranom") for c in R_FIX]
M_SEPT = ["m_sept_spring_ranom", "m_sept_L1_ranom", "m_sept_L2_ranom",
          "m_splash_spring_ranom", "m_splash_L2_ranom", "m_sun_per_rainday_ranom",
          "m_liebig_spring_ranom", "m_liebig_L2_ranom", "m_moist_limited_ranom",
          "m_sept_autumn_ranom"]
M_RUST = ["m_rust_potential_ranom", "m_survive_x_cyc_ranom", "m_frost_win_ranom",
          "m_heat_kill_ranom", "m_jun_tmax_ranom", "m_rust_winter_cyc_ranom",
          "m_rust_cyc_ranom"]

EVAL = [y for y in range(2005, 2026) if y != 2020]


def predict(target, T, cfg):
    sept = target in SEPTORIA
    fs = list(cfg["sept"] if sept else cfg["rust"])
    tr = DF[(DF.Year < T) & (DF.Year >= 1990) & DF[target].notna()].copy()
    te = DF[DF.Year == T].copy()
    if len(te) == 0 or len(tr) < 30:
        return None
    cy = sorted(tr.Year.unique())[-12:]
    cl = tr[tr.Year.isin(cy)]
    nc, rc = cl[target].mean(), cl.groupby("Region")[target].mean()
    base = 0.5 * te.Region.map(rc).fillna(nc).to_numpy() + 0.5 * nc
    if not fs:
        return te[["Year", "Region"]].assign(target=target, value=base)
    cols = fs + [f"bl_reg4_{target}", f"bl_reg10_{target}",
                 f"bl_nat4_{target}", f"bl_nat10_{target}", "trend"]
    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med).fillna(0.0), Xte.fillna(med).fillna(0.0)
    for r in REGIONS[1:]:
        Xtr[f"R_{r}"] = (tr.Region == r).astype(float)
        Xte[f"R_{r}"] = (te.Region == r).astype(float)
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=100.0).fit(sc.transform(Xtr), tr[target].to_numpy(float))
    v = np.clip(m.predict(sc.transform(Xte)), 0, 100 if target in INCIDENCE else None)
    return te[["Year", "Region"]].assign(target=target, value=v)


def run(cfg, years=EVAL):
    out = [p for t in TARGETS for T in years if (p := predict(t, T, cfg)) is not None]
    return pd.concat(out, ignore_index=True)


CONFIGS = {
    "S1 statistical, FIXED baseline":  dict(sept=E_FIX, rust=R_FIX),
    "S2 statistical, ROLLING baseline": dict(sept=E_ROLL, rust=R_ROLL),
    "S3 mechanistic only":             dict(sept=M_SEPT, rust=M_RUST),
    "S4 mech + statistical(rolling)":  dict(sept=M_SEPT + E_ROLL, rust=M_RUST + R_ROLL),
    "S5 mech + statistical(fixed)":    dict(sept=M_SEPT + E_FIX, rust=M_RUST + R_FIX),
    "S6 climatology":                  dict(sept=[], rust=[]),
}

if __name__ == "__main__":
    truth = to_long(OBS)
    NATM = FEAT.groupby("Year")[[c for c in FEAT.columns if c.endswith("_ranom")]].mean()
    print("=== 2026 mechanistic signal with ROLLING baseline (fixes the warming drift) ===")
    print(NATM.reindex([2021, 2022, 2023, 2024, 2025, 2026])[
        ["m_sept_spring_ranom", "m_liebig_spring_ranom", "m_moist_limited_ranom",
         "m_splash_spring_ranom", "m_rust_potential_ranom", "m_heat_kill_ranom"]]
        .round(2).to_string())

    rows = []
    for name, cfg in CONFIGS.items():
        p = run(cfg)
        rec = {"config": name}
        for lab, yrs in [("05_25", EVAL), ("05_19", list(range(2005, 2020))),
                         ("21_25", [2021, 2022, 2023, 2024, 2025])]:
            _, s = score(p, truth, yrs)
            rec[f"sept_{lab}"] = s["septoria_pooled"]
            rec[f"rust_{lab}"] = s["rust_pooled"]
        rows.append(rec)
    print("\n" + "=" * 110)
    print("ROLLING BASELINE + MECHANISTIC FEATURES   (current best: sept 13.87 | rust 5.38)")
    print("=" * 110)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
