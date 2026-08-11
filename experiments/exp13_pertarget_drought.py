"""Experiment 13: per-target model choice, and behaviour in DROUGHT years.

Two issues surfaced by final_model.py:

  (a) the two yellow-rust SEVERITY targets have NEGATIVE skill vs climatology
      (-24.6 % and -12.4 %) while the rust INCIDENCE targets are positive.
      So the model choice should be made per target, not per disease.

  (b) 2026's spring is a drought (spring rain anomaly -1.24 SD, Apr-May
      sunshine +2.68 SD). The closest analogue, 2025 (-2.09 / +3.77), produced
      the lowest septoria in the record -- and exp08 showed the model
      OVER-predicted 2025 badly (predicted ~61 incidence, actual 35).
      Since 2026 is the year being scored, drought-year behaviour matters far
      more than average-year behaviour. This checks whether adding the drought
      threshold term to septoria fixes that tail.
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
NL_SEPT = ["nl_drought", "nl_dry2", "nl_verywet"]
NL_RUST = ["nl_junheat", "nl_mildwin", "nl_drought"]

PEST = load_pest()
OBS = PEST[PEST.Year <= 2025].reset_index(drop=True)
WF = build_weather_features()
BL = as_of_baselines(OBS, TARGETS)
DF = OBS.merge(WF, on=["Year", "Region"], how="left").merge(BL, on=["Year", "Region"], how="left")
DF["trend"] = (DF.Year - 2000) / 25.0
DF["nl_drought"] = np.clip(DF["e_sun_apr_may_anom"] - 1.0, 0, None)
DF["nl_dry2"] = np.clip(-DF["e_rain_apr_may_anom"] - 0.75, 0, None)
DF["nl_verywet"] = np.clip(DF["e_rain_apr_may_anom"] - 0.75, 0, None)
DF["nl_junheat"] = np.clip(DF["e_tmean_jun_anom"] - 1.0, 0, None)
DF["nl_mildwin"] = np.clip(-DF["e_frost_winter_anom"] - 0.5, 0, None)
REGIONS = sorted(OBS.Region.unique())
EVAL = [y for y in range(2005, 2026) if y != 2020]

# national spring-dryness ranking, to define "drought years"
NATW = WF.groupby("Year")[["e_rain_apr_may_anom", "e_sun_apr_may_anom"]].mean()
NATW["dryness"] = NATW.e_sun_apr_may_anom - NATW.e_rain_apr_may_anom
DRY_YEARS = [y for y in EVAL if NATW.dryness.get(y, 0) >= NATW.dryness.reindex(EVAL).quantile(0.7)]
WET_YEARS = [y for y in EVAL if NATW.dryness.get(y, 0) <= NATW.dryness.reindex(EVAL).quantile(0.3)]


def predict(target, T, mode):
    """mode: 'clim' | 'model' | 'model_nl'"""
    tr = DF[(DF.Year < T) & (DF.Year >= 1990) & DF[target].notna()]
    te = DF[DF.Year == T]
    if len(te) == 0 or len(tr) < 30:
        return None
    cy = sorted(tr.Year.unique())[-12:]
    cl = tr[tr.Year.isin(cy)]
    nc, rc = cl[target].mean(), cl.groupby("Region")[target].mean()
    base = 0.5 * te.Region.map(rc).fillna(nc).to_numpy() + 0.5 * nc
    if mode == "clim":
        return te[["Year", "Region"]].assign(target=target, value=base)

    cols = list(SEPT_FEATS if target in SEPTORIA else RUST_FEATS)
    if mode == "model_nl":
        cols += (NL_SEPT if target in SEPTORIA else NL_RUST)
    cols += [f"bl_reg4_{target}", f"bl_reg10_{target}",
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


def run(mode, targets=TARGETS, years=EVAL):
    out = [p for t in targets for T in years if (p := predict(t, T, mode)) is not None]
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    truth = to_long(OBS)
    modes = ["clim", "model", "model_nl"]
    store = {m: run(m) for m in modes}

    print("=" * 112)
    print("PER-TARGET RMSE by mode, over three windows.  Pick the mode that wins CONSISTENTLY.")
    print("=" * 112)
    wins = {}
    rows = []
    for t in TARGETS:
        rec = {"target": t}
        for m in modes:
            for lab, yrs in [("05_25", EVAL), ("05_19", list(range(2005, 2020))),
                             ("21_25", [2021, 2022, 2023, 2024, 2025])]:
                p = store[m][store[m].target == t]
                tt = truth[(truth.target == t) & truth.Year.isin(yrs)]
                mm = tt.merge(p, on=["Year", "Region", "target"],
                              suffixes=("_t", "_p")).dropna(subset=["value_t"])
                rec[f"{m}_{lab}"] = rmse(mm.value_t, mm.value_p)
        # consistent winner = lowest on 05_25 AND not worst on either sub-window
        best = min(modes, key=lambda m: rec[f"{m}_05_25"])
        wins[t] = best
        rec["choice"] = best
        rows.append(rec)
    r = pd.DataFrame(rows)
    print(r[["target"] + [f"{m}_05_25" for m in modes] + ["choice"]].round(4).to_string(index=False))
    print("\nsub-window detail:")
    print(r[["target"] + [f"{m}_{w}" for w in ("05_19", "21_25") for m in modes]].round(4).to_string(index=False))

    # hybrid: per-target best
    hyb = pd.concat([store[wins[t]][store[wins[t]].target == t] for t in TARGETS],
                    ignore_index=True)
    print("\n" + "=" * 112)
    print("POOLED, per-target hybrid vs single-mode")
    print("=" * 112)
    for lab, yrs in [("05_25", EVAL), ("05_19", list(range(2005, 2020))),
                     ("21_25", [2021, 2022, 2023, 2024, 2025])]:
        line = f"{lab}: "
        for nm, p in [("clim", store["clim"]), ("model", store["model"]),
                      ("model_nl", store["model_nl"]), ("HYBRID", hyb)]:
            _, s = score(p, truth, yrs)
            line += f"  {nm} sept={s['septoria_pooled']:6.3f} rust={s['rust_pooled']:6.3f} |"
        print(line)

    # ---- drought-year behaviour ----
    print("\n" + "=" * 112)
    print(f"DROUGHT YEARS {DRY_YEARS}\nWET YEARS {WET_YEARS}")
    print("=" * 112)
    for lab, yrs in [("DRY", DRY_YEARS), ("WET", WET_YEARS)]:
        line = f"{lab:4s}: "
        for nm in modes:
            _, s = score(store[nm], truth, yrs)
            line += f"  {nm} sept={s['septoria_pooled']:6.3f} rust={s['rust_pooled']:6.3f} |"
        print(line)

    print("\nNational septoria L1 incidence, actual vs predicted, by year:")
    t = "L1_Zymoseptoria_tritici_Crop_Incidence"
    a = OBS[OBS.Year.isin(EVAL)].groupby("Year")[t].mean()
    d = pd.DataFrame({"actual": a})
    for m in modes:
        d[m] = store[m][store[m].target == t].groupby("Year").value.mean()
    d["dryness"] = NATW.dryness.reindex(d.index)
    print(d.round(1).to_string())
