"""Experiment 05: two-stage year-effect model.

exp04 showed 51-79% of variance is the national year effect and only 0-9% is
the region effect, and that a perfect year effect would score septoria ~10.2 vs
~16.7 for perfect climatology. So:

  Stage 1  predict the NATIONAL level of each target for year Y from national
           weather. ~50 annual observations, so keep it very low dimensional
           and heavily regularised.
  Stage 2  distribute that national level over regions using each region's
           climatological share, optionally tilted by the region's own weather
           anomaly relative to the national anomaly.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

from common import (TARGETS, SEVERITY, INCIDENCE, load_pest, backtest, report,
                    EVAL_YEARS, RECENT_EVAL_YEARS)
from features import build_weather_features

warnings.filterwarnings("ignore")

WF = build_weather_features()
PEST = load_pest()
REGIONS = sorted(PEST.Region.unique())

EPI_ANOM = [c for c in WF.columns if c.startswith("e_") and c.endswith("_anom")]

# Compact, epidemiology-led predictor sets. With ~50 annual rows, fewer is better.
SEPT_FEATS = ["e_rain_spring_anom", "e_rain_apr_may_anom", "e_rdays_latespring_anom",
              "e_sun_apr_may_anom", "e_tmean_spring_anom", "e_frost_winter_anom",
              "e_wetness_idx_anom", "e_dry_bright_spring_anom"]
RUST_FEATS = ["e_tmin_winter_anom", "e_frost_winter_anom", "e_tmean_jun_anom",
              "e_tmax_jun_anom", "e_tmean_spring_anom", "e_rain_spring_anom",
              "e_mild_winter_anom", "e_rust_window_anom"]


def national_weather(years=None):
    """Cross-region mean weather per year (the national anomaly signal)."""
    n = WF.groupby("Year")[EPI_ANOM].mean()
    return n if years is None else n.reindex(years)


NATW = national_weather()


def feats_for(target):
    return SEPT_FEATS if "Zymoseptoria" in target else RUST_FEATS


def stage1_fit_predict(train, target, pred_years, alpha, transform, model="ridge",
                       n_comp=3, min_year=1971):
    """Predict the national level of `target` for `pred_years`."""
    nat = (train[train.Year >= min_year].groupby("Year")[target].mean().dropna())
    cols = feats_for(target)
    X = NATW.reindex(nat.index)[cols]
    ok = X.notna().all(axis=1)
    X, y = X[ok], nat[ok]
    if len(y) < 12:
        return np.full(len(pred_years), nat.mean())
    z = _fwd(target, y.to_numpy()) if transform else y.to_numpy(float)

    Xp = NATW.reindex(pred_years)[cols]
    Xp = Xp.fillna(X.median())

    if model == "pls":
        sc = StandardScaler().fit(X)
        m = PLSRegression(n_components=min(n_comp, X.shape[1]))
        m.fit(sc.transform(X), z)
        pz = m.predict(sc.transform(Xp)).ravel()
    elif model == "pcr":
        sc = StandardScaler().fit(X)
        pca = PCA(n_components=min(n_comp, X.shape[1])).fit(sc.transform(X))
        r = Ridge(alpha=alpha).fit(pca.transform(sc.transform(X)), z)
        pz = r.predict(pca.transform(sc.transform(Xp)))
    else:
        m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X, z)
        pz = m.predict(Xp)

    pv = _inv(target, pz) if transform else pz
    return np.clip(pv, 0, 100 if target in INCIDENCE else None)


def _fwd(t, y):
    if t in SEVERITY:
        return np.log1p(np.clip(y, 0, None))
    p = np.clip(np.asarray(y, float), 0, 100) / 100
    return np.log(np.clip(p, .005, .995) / (1 - np.clip(p, .005, .995)))


def _inv(t, z):
    if t in SEVERITY:
        return np.clip(np.expm1(z), 0, None)
    return 100 / (1 + np.exp(-np.clip(z, -12, 12)))


def make_two_stage(alpha=10.0, transform=False, shrink=0.35, clim_k=12,
                   region_mode="ratio", region_shrink=0.5, model="ridge",
                   n_comp=3, min_year=1971):
    """shrink       : pull the stage-1 national level toward recent climatology
       region_mode  : 'ratio' multiplicative, 'add' additive, 'none' flat
       region_shrink: how much of the raw region deviation to keep
    """
    def predict(train, test_rows):
        yrs = sorted(test_rows.Year.unique())
        clim_years = sorted(train.Year.unique())[-clim_k:]
        clim = train[train.Year.isin(clim_years)]
        out = []
        for t in TARGETS:
            nat_pred = stage1_fit_predict(train, t, yrs, alpha, transform,
                                          model, n_comp, min_year)
            nat_pred = dict(zip(yrs, nat_pred))
            nat_clim = clim[t].mean()
            # stage-1 shrinkage toward the recent national climatology
            lvl = {y: (1 - shrink) * v + shrink * nat_clim for y, v in nat_pred.items()}

            # stage 2: region share
            reg_mean = clim.groupby("Region")[t].mean()
            if region_mode == "ratio":
                share = (reg_mean / nat_clim).clip(0.3, 3.0)
                share = 1 + region_shrink * (share - 1)
                vals = test_rows.Year.map(lvl).to_numpy() * \
                    test_rows.Region.map(share).fillna(1.0).to_numpy()
            elif region_mode == "add":
                off = (reg_mean - nat_clim) * region_shrink
                vals = test_rows.Year.map(lvl).to_numpy() + \
                    test_rows.Region.map(off).fillna(0.0).to_numpy()
            else:
                vals = test_rows.Year.map(lvl).to_numpy()
            vals = np.clip(vals, 0, 100 if t in INCIDENCE else None)
            d = test_rows.copy()
            d["target"], d["value"] = t, vals
            out.append(d)
        return pd.concat(out, ignore_index=True)
    return predict


if __name__ == "__main__":
    runs = [
        ("T1 2stage ridge a=10 raw shr.35 ratio",  make_two_stage()),
        ("T2 2stage ridge a=10 raw shr.0  ratio",  make_two_stage(shrink=0.0)),
        ("T3 2stage ridge a=10 raw shr.6  ratio",  make_two_stage(shrink=0.6)),
        ("T4 2stage ridge a=3  raw shr.35 ratio",  make_two_stage(alpha=3.0)),
        ("T5 2stage ridge a=30 raw shr.35 ratio",  make_two_stage(alpha=30.0)),
        ("T6 2stage ridge a=10 TRANS shr.35",      make_two_stage(transform=True)),
        ("T7 2stage ridge a=10 raw shr.35 add",    make_two_stage(region_mode="add")),
        ("T8 2stage ridge a=10 raw shr.35 none",   make_two_stage(region_mode="none")),
        ("T9 2stage ridge regshrink=1.0",          make_two_stage(region_shrink=1.0)),
        ("T10 2stage pls n=2",                     make_two_stage(model="pls", n_comp=2)),
        ("T11 2stage pls n=3",                     make_two_stage(model="pls", n_comp=3)),
        ("T12 2stage pcr n=3",                     make_two_stage(model="pcr", n_comp=3)),
        ("T13 2stage ridge 1990+",                 make_two_stage(min_year=1990)),
        ("T14 2stage ridge clim_k=6",              make_two_stage(clim_k=6)),
        ("T15 2stage ridge clim_k=20",             make_two_stage(clim_k=20)),
    ]
    rows = []
    for name, fn in runs:
        preds, truth = backtest(fn)
        r = report(name, preds, truth)
        rows.append({"model": name,
                     "sept_all": r["all"]["septoria_pooled"], "sept_rec": r["recent"]["septoria_pooled"],
                     "rust_all": r["all"]["rust_pooled"], "rust_rec": r["recent"]["rust_pooled"]})
    print("\n\n" + "=" * 95)
    print("SUMMARY   naive best: sept 17.71/18.06  rust 6.39/9.23")
    print("          exp03 best: sept 15.56/15.03  rust 7.17/11.05")
    print("          ORACLE:     sept 10.20/ 9.05  rust 3.85/5.53")
    print("=" * 95)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
