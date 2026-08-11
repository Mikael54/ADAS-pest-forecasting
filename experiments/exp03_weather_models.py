"""Experiment 03: pooled (all-region) models on real observed weather.

Key departures from the repo's example pipeline:
  * pool all regions into one model with region effects, instead of fitting a
    separate ElasticNet per region on ~40 rows;
  * use real HadUK-Grid observations instead of HadGEM2 GCM output;
  * transform the targets (severity is very skewed, incidence is a bounded %);
  * blend the model toward climatology, which regularises small-sample noise.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from common import (TARGETS, SEVERITY, INCIDENCE, load_pest, backtest, report,
                    to_long, EVAL_YEARS, RECENT_EVAL_YEARS)
from features import build_weather_features, add_lag_features

warnings.filterwarnings("ignore")

WF = build_weather_features()
PEST = load_pest()
REGIONS = sorted(PEST.Region.unique())

EPI = [c for c in WF.columns if c.startswith("e_") and c.endswith("_anom")]
EPI_RAW = [c for c in WF.columns if c.startswith("e_") and not c.endswith("_anom")]
MONTHLY = [c for c in WF.columns if c.startswith("w_")]


# ---- target transforms ----------------------------------------------------
def fwd(t, y):
    if t in SEVERITY:
        return np.log1p(np.clip(y, 0, None))
    p = np.clip(np.asarray(y, float), 0, 100) / 100
    p = np.clip(p, 0.005, 0.995)
    return np.log(p / (1 - p))


def inv(t, z):
    if t in SEVERITY:
        return np.clip(np.expm1(z), 0, None)
    return 100 / (1 + np.exp(-np.clip(z, -12, 12)))


# ---- design matrix --------------------------------------------------------
def design(rows, feat_cols, use_region=True, use_year=True):
    X = rows[feat_cols].copy()
    if use_region:
        for r in REGIONS[1:]:
            X[f"R_{r}"] = (rows.Region == r).astype(float)
    if use_year:
        X["yr"] = (rows.Year - 2000) / 25.0
    return X


def make_model(kind):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=30.0))
    if kind == "ridge_soft":
        return make_pipeline(StandardScaler(), Ridge(alpha=5.0))
    if kind == "ridge_hard":
        return make_pipeline(StandardScaler(), Ridge(alpha=150.0))
    if kind == "enet":
        return make_pipeline(StandardScaler(), ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=20000))
    if kind == "rf":
        return RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                                     max_features=0.35, random_state=0, n_jobs=-1)
    if kind == "gbm":
        return GradientBoostingRegressor(n_estimators=300, learning_rate=0.03,
                                         max_depth=3, subsample=0.8, random_state=0)
    raise ValueError(kind)


def make_predictor(kind, feat_cols, transform=True, shrink=0.0, clim_k=10,
                   use_region=True, use_year=True, min_year=1971):
    """shrink: weight on recent-climatology blended into the model prediction."""
    def predict(train, test_rows):
        tr = train[train.Year >= min_year].merge(WF, on=["Year", "Region"], how="left")
        te = test_rows.merge(WF, on=["Year", "Region"], how="left")
        Xtr_all = design(tr, feat_cols, use_region, use_year)
        Xte = design(te, feat_cols, use_region, use_year)
        med = Xtr_all.median()
        Xte = Xte.fillna(med).fillna(0.0)

        clim_years = sorted(train.Year.unique())[-clim_k:]
        clim = train[train.Year.isin(clim_years)]

        out = []
        for t in TARGETS:
            ok = tr[t].notna() & Xtr_all.notna().all(axis=1)
            Xtr, ytr = Xtr_all[ok], tr.loc[ok, t]
            z = fwd(t, ytr) if transform else ytr.to_numpy(float)
            m = make_model(kind)
            m.fit(Xtr, z)
            pz = m.predict(Xte)
            pv = inv(t, pz) if transform else pz
            if transform:
                pv = np.clip(pv, 0, 100 if t in INCIDENCE else None)
            else:
                pv = np.clip(pv, 0, 100 if t in INCIDENCE else None)
            if shrink > 0:
                pv = (1 - shrink) * pv + shrink * clim[t].mean()
            d = test_rows.copy()
            d["target"], d["value"] = t, pv
            out.append(d)
        return pd.concat(out, ignore_index=True)
    return predict


if __name__ == "__main__":
    runs = [
        ("M1 ridge, epi-anom",            make_predictor("ridge", EPI)),
        ("M2 ridge, epi-anom + raw",      make_predictor("ridge", EPI + EPI_RAW)),
        ("M3 ridge, monthly raw",         make_predictor("ridge", MONTHLY)),
        ("M4 ridge, monthly + epi",       make_predictor("ridge", MONTHLY + EPI)),
        ("M5 ridge-hard, monthly + epi",  make_predictor("ridge_hard", MONTHLY + EPI)),
        ("M6 enet, monthly + epi",        make_predictor("enet", MONTHLY + EPI)),
        ("M7 gbm, epi",                   make_predictor("gbm", EPI + EPI_RAW)),
        ("M8 rf, epi",                    make_predictor("rf", EPI + EPI_RAW)),
        ("M9 ridge epi, NO transform",    make_predictor("ridge", EPI, transform=False)),
        ("M10 ridge epi, shrink 0.3",     make_predictor("ridge", EPI, shrink=0.3)),
        ("M11 ridge epi, shrink 0.5",     make_predictor("ridge", EPI, shrink=0.5)),
        ("M12 ridge epi, 1990+",          make_predictor("ridge", EPI, min_year=1990)),
    ]
    rows = []
    for name, fn in runs:
        preds, truth = backtest(fn)
        r = report(name, preds, truth)
        rows.append({"model": name,
                     "sept_all": r["all"]["septoria_pooled"], "rust_all": r["all"]["rust_pooled"],
                     "sept_rec": r["recent"]["septoria_pooled"], "rust_rec": r["recent"]["rust_pooled"]})
    print("\n\n" + "=" * 90 + "\nSUMMARY  (naive best: sept 17.71 / 18.06,  rust 6.39 / 9.23)\n" + "=" * 90)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
