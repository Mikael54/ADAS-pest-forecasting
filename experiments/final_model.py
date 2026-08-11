"""Final forecasting pipeline for the SPHERE-PPL ADAS wheat disease contest.

USAGE:  cd experiments && ../.venv/bin/python final_model.py
Writes submission/pest_forecasts_2026.csv and submission/pest_model_performance.csv.

--------------------------------------------------------------------------
MODEL
--------------------------------------------------------------------------
For each of the 8 targets, ridge regression pooled across all 9 survey regions:

    y[region, year] ~  epidemiological weather anomalies (region level)
                     + as-of disease baselines (region & national exponentially
                       weighted means over previously OBSERVED years, hl 4 & 10)
                     + linear year trend
                     + region dummies
                     [+ non-linear threshold terms      -- yellow rust only]
                     [+ drought x baseline interactions -- "int" form]

trained on 1990+ with alpha=100, predictions clipped to [0,100] for incidence
and [0,inf) for severity.

Three model forms are combined, because their biases are complementary:
    add  purely additive
    int  additive + drought x baseline interactions (lets a dry spring cancel
         an elevated recent baseline)
    rel  multiplicative: models log(y / baseline), so weather scales the
         prevailing level instead of adding to it

    SEPTORIA    = mean(add, int)         backtest RMSE 13.90 vs 17.62 clim
    YELLOW RUST = mean(add, rel, int)    backtest RMSE  5.32 vs  5.82 clim

--------------------------------------------------------------------------
WHY THIS SHAPE (see EXPERIMENTS.md for the full evidence trail)
--------------------------------------------------------------------------
* 51-79 % of the variance in every target is the national YEAR effect; region
  effects are only 0-9 %. So the weather signal is the entire lever, and models
  must pool regions rather than fit one model per region.
* Weather is real Met Office HadUK-Grid observation, not the repo's
  bioclim_data.csv, which is HadGEM2 GCM output and does not track real years.
* Disease levels are strongly non-stationary (yellow rust collapsed after the
  1970s and returned in the 2020s with new races), so an explicit
  backward-looking baseline term is required alongside weather.
* Yellow rust has NEGATIVE out-of-sample skill from linear weather terms, but
  responds to thresholds (winter frost kill, June heat shutdown), hence the
  relu-style terms for rust only.

INFORMATION RULE: weather Sept(Y-1) -> June(Y); disease observations <= Y-1.
Nothing else. The June cap matches what is actually published by the time a
forecast for year Y must be issued, and is exactly what is available for 2026.
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from common import (TARGETS, SEVERITY, INCIDENCE, SEPTORIA, RUST, load_pest,
                    to_long, score, rmse)
from features import build_weather_features
from exp09_baseline_feats import as_of_baselines

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
FORECAST_YEAR = 2026
ALPHA, MIN_YEAR, CLIM_K = 100.0, 1990, 12

SEPT_FEATS = ["e_rain_spring_anom", "e_rain_apr_may_anom", "e_rdays_latespring_anom",
              "e_sun_apr_may_anom", "e_tmean_spring_anom", "e_frost_winter_anom",
              "e_wetness_idx_anom", "e_dry_bright_spring_anom"]
RUST_FEATS = ["e_tmin_winter_anom", "e_frost_winter_anom", "e_tmean_jun_anom",
              "e_tmean_spring_anom", "e_rain_spring_anom", "e_mild_winter_anom"]
NL_RUST = ["nl_junheat", "nl_mildwin", "nl_drought"]

FORMS = {"septoria": ["add", "int"], "rust": ["add", "rel", "int"]}


def build_frame():
    pest = load_pest()                      # includes the empty 2026 block
    wf = build_weather_features()
    bl = as_of_baselines(pest.reset_index(drop=True), TARGETS)
    df = (pest.reset_index(drop=True)
              .merge(wf, on=["Year", "Region"], how="left")
              .merge(bl, on=["Year", "Region"], how="left"))
    df["trend"] = (df.Year - 2000) / 25.0
    df["nl_drought"] = np.clip(df["e_sun_apr_may_anom"] - 1.0, 0, None)
    df["nl_dry2"] = np.clip(-df["e_rain_apr_may_anom"] - 0.75, 0, None)
    df["nl_junheat"] = np.clip(df["e_tmean_jun_anom"] - 1.0, 0, None)
    df["nl_mildwin"] = np.clip(-df["e_frost_winter_anom"] - 0.5, 0, None)
    return df, pest[pest.Year <= 2025].reset_index(drop=True)


DF, OBS = build_frame()
REGIONS = sorted(OBS.Region.unique())


def predict_form(target, T, form):
    """One model form, fitted on [MIN_YEAR, T), predicting every region in T."""
    tr = DF[(DF.Year < T) & (DF.Year >= MIN_YEAR) & DF[target].notna()].copy()
    te = DF[DF.Year == T].copy()
    if len(te) == 0 or len(tr) < 30:
        return None
    cy = sorted(tr.Year.unique())[-CLIM_K:]
    cl = tr[tr.Year.isin(cy)]
    nat_clim, reg_clim = cl[target].mean(), cl.groupby("Region")[target].mean()
    base = 0.5 * te.Region.map(reg_clim).fillna(nat_clim).to_numpy() + 0.5 * nat_clim
    if form == "clim":
        return te[["Year", "Region"]].assign(target=target, value=base)

    cols = list(SEPT_FEATS if target in SEPTORIA else RUST_FEATS)
    if target in RUST:
        cols += NL_RUST
    bl_cols = [f"bl_reg4_{target}", f"bl_reg10_{target}",
               f"bl_nat4_{target}", f"bl_nat10_{target}"]
    cols += bl_cols + ["trend"]
    if form == "int":
        for d in (tr, te):
            d["ix_dr_bl"] = d["nl_drought"] * d[f"bl_nat4_{target}"]
            d["ix_dry_bl"] = d["nl_dry2"] * d[f"bl_nat4_{target}"]
            d["ix_rain_bl"] = d["e_rain_apr_may_anom"] * d[f"bl_nat4_{target}"]
        cols += ["ix_dr_bl", "ix_dry_bl", "ix_rain_bl"]

    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med).fillna(0.0), Xte.fillna(med).fillna(0.0)
    for r in REGIONS[1:]:
        Xtr[f"R_{r}"] = (tr.Region == r).astype(float)
        Xte[f"R_{r}"] = (te.Region == r).astype(float)
    sc = StandardScaler().fit(Xtr)

    if form == "rel":
        b_tr = tr[f"bl_reg10_{target}"].fillna(tr[f"bl_nat10_{target}"])
        b_te = te[f"bl_reg10_{target}"].fillna(te[f"bl_nat10_{target}"])
        eps = 0.05 * max(tr[target].mean(), 1e-3)
        z = np.clip(np.log((tr[target].to_numpy(float) + eps) / (b_tr.to_numpy(float) + eps)), -4, 4)
        m = Ridge(alpha=ALPHA).fit(sc.transform(Xtr), z)
        pv = (b_te.to_numpy(float) + eps) * np.exp(np.clip(m.predict(sc.transform(Xte)), -4, 4)) - eps
    else:
        m = Ridge(alpha=ALPHA).fit(sc.transform(Xtr), tr[target].to_numpy(float))
        pv = m.predict(sc.transform(Xte))
    pv = np.clip(pv, 0, 100 if target in INCIDENCE else None)
    return te[["Year", "Region"]].assign(target=target, value=pv)


def predict(target, T):
    """Ensemble of the forms chosen for this target's disease."""
    forms = FORMS["septoria" if target in SEPTORIA else "rust"]
    ps = [p for f in forms if (p := predict_form(target, T, f)) is not None]
    if not ps:
        return None
    ps = [p.sort_values(["Year", "Region"]).reset_index(drop=True) for p in ps]
    v = np.mean([p.value.to_numpy() for p in ps], axis=0)
    out = ps[0][["Year", "Region", "target"]].copy()
    out["value"] = np.clip(v, 0, 100 if target in INCIDENCE else None)
    return out


# Leaf 2 emerges before the flag leaf, sits nearer the splash-dispersed
# inoculum and is exposed longer, so it always carries at least as much
# septoria. That holds in 97.8-99.3 % of observed rows, so it is imposed on the
# forecast (least-squares projection: replace a violating pair with its mean).
# NOT imposed for yellow rust, where it only holds in 75-78 % of rows.
SEPT_PAIRS = [("L1_Zymoseptoria_tritici_Disease_Severity",
               "L2_Zymoseptoria_tritici_Disease_Severity"),
              ("L1_Zymoseptoria_tritici_Crop_Incidence",
               "L2_Zymoseptoria_tritici_Crop_Incidence")]


def enforce_leaf_order(preds):
    w = preds.pivot_table(index=["Year", "Region"], columns="target",
                          values="value", observed=True)
    for a, b in SEPT_PAIRS:
        if a in w.columns and b in w.columns:
            bad = w[b] < w[a]
            mid = (w.loc[bad, a] + w.loc[bad, b]) / 2
            w.loc[bad, a], w.loc[bad, b] = mid, mid
    return (w.reset_index()
             .melt(id_vars=["Year", "Region"], var_name="target", value_name="value")
             .dropna(subset=["value"]))


def run(years, fn=predict):
    out = [p for t in TARGETS for T in years if (p := fn(t, T)) is not None]
    return enforce_leaf_order(pd.concat(out, ignore_index=True))


if __name__ == "__main__":
    EVAL = [y for y in range(2005, 2026) if y != 2020]
    truth = to_long(OBS)
    bt = run(EVAL)
    cl = run(EVAL, lambda t, T: predict_form(t, T, "clim"))

    print("=" * 94)
    print("FINAL ENSEMBLE -- rolling-origin backtest (train strictly on years < T)")
    print("=" * 94)
    for label, yrs in [("2005-2025 (excl 2020)", EVAL),
                       ("2005-2019", list(range(2005, 2020))),
                       ("2021-2025", [2021, 2022, 2023, 2024, 2025])]:
        _, a = score(bt, truth, yrs)
        _, c = score(cl, truth, yrs)
        print(f"\n{label}")
        for d, k in [("septoria   ", "septoria_pooled"), ("yellow rust", "rust_pooled")]:
            print(f"  {d} model {a[k]:7.3f}   climatology {c[k]:7.3f}   ({100*(1-a[k]/c[k]):+5.1f}%)")

    pt, _ = score(bt, truth, EVAL)
    ptc, _ = score(cl, truth, EVAL)
    print("\nper-target RMSE (2005-2025):")
    print(f"{'target':<45}{'model':>10}{'clim':>10}{'skill':>9}")
    for t in TARGETS:
        print(f"{t:<45}{pt[t]:>10.4f}{ptc[t]:>10.4f}{100*(1-pt[t]/ptc[t]):>8.1f}%")

    # ------------------------- 2026 forecast -------------------------
    # via run(), so the L2 >= L1 septoria constraint is applied here too
    fc = run([FORECAST_YEAR])
    fc = fc.rename(columns={"Year": "year", "Region": "region", "value": "forecast_value"})
    fc["target"] = pd.Categorical(fc.target, categories=TARGETS, ordered=True)
    fc = fc.sort_values(["region", "target"])[
        ["region", "target", "year", "forecast_value"]].reset_index(drop=True)

    perf = []
    m = truth[truth.Year.isin(EVAL)].merge(bt, on=["Year", "Region", "target"],
                                           suffixes=("_t", "_p")).dropna(subset=["value_t"])
    for (reg, t), g in m.groupby(["Region", "target"], observed=True):
        perf.append({"region": reg, "target": t, "rmse": rmse(g.value_t, g.value_p)})
    perf = pd.DataFrame(perf)
    perf["target"] = pd.Categorical(perf.target, categories=TARGETS, ordered=True)
    perf = perf.sort_values(["region", "target"]).reset_index(drop=True)

    sub = ROOT / "submission"
    fc.to_csv(sub / "pest_forecasts_2026.csv", index=False)
    perf.to_csv(sub / "pest_model_performance.csv", index=False)

    print("\n" + "=" * 94)
    print(f"{FORECAST_YEAR} FORECAST")
    print("=" * 94)
    print(fc.pivot_table(index="region", columns="target", values="forecast_value",
                         observed=True).round(2).to_string())
    print(f"\nwrote {sub/'pest_forecasts_2026.csv'} ({len(fc)} rows), "
          f"{sub/'pest_model_performance.csv'} ({len(perf)} rows)")
    print("\nnational means, recent actuals vs 2026 forecast:")
    hist = OBS[OBS.Year >= 2021].groupby("Year")[TARGETS].mean()
    hist.loc["2026 fc"] = fc.groupby("target", observed=True).forecast_value.mean()
    print(hist.round(2).to_string())
